//! Device-owned swapchain resources, frame synchronization and presentation.

use std::{rc::Rc, sync::Arc};

use crate::runtime::{Device, RecordedCommandBuffer};
use crate::{Error, Result};

/// Vulkan values selected from one exact surface query for swapchain creation.
pub(in crate::runtime) struct SwapchainConfig {
	pub(in crate::runtime) extent: ash::vk::Extent2D,
	pub(in crate::runtime) image_count: u32,
	pub(in crate::runtime) format: ash::vk::SurfaceFormatKHR,
	pub(in crate::runtime) present_mode: ash::vk::PresentModeKHR,
	pub(in crate::runtime) composite_alpha: ash::vk::CompositeAlphaFlagsKHR,
	pub(in crate::runtime) pre_transform: ash::vk::SurfaceTransformFlagsKHR,
}

pub(in crate::runtime) struct PresentationSwapchain {
	_device: Arc<Device>,
	loader: ash::khr::swapchain::Device,
	handle: ash::vk::SwapchainKHR,
	extent: ash::vk::Extent2D,
	images: Vec<ash::vk::Image>,
	views: Vec<ash::vk::ImageView>,
	frames: Vec<PresentationFrameSync>,
	next_frame: usize,
}

struct PresentationFrameSync {
	image_available: ash::vk::Semaphore,
	render_finished: ash::vk::Semaphore,
	in_flight: ash::vk::Fence,
	command: Option<RecordedCommandBuffer>,
	source: Option<Rc<dyn std::any::Any>>,
}

const FRAMES_IN_FLIGHT: usize = 2;

impl PresentationSwapchain {
	pub(in crate::runtime) fn create(
		device: &Arc<Device>,
		surface: ash::vk::SurfaceKHR,
		present_queue_family: Option<u32>,
		config: SwapchainConfig,
	) -> Result<Self> {
		let loader = ash::khr::swapchain::Device::new(device.instance().raw(), device.raw());

		// Use CONCURRENT sharing when graphics and present are distinct families;
		// EXCLUSIVE otherwise (the common case).
		let graphics_family = device.physical().graphics_queue_family;
		let present_family = present_queue_family;
		let distinct_families = graphics_family
			.zip(present_family)
			.is_some_and(|(g, p)| g != p);
		let sharing_families: Vec<u32> = if distinct_families {
			[graphics_family, present_family]
				.iter()
				.filter_map(|f| *f)
				.collect()
		} else {
			vec![]
		};
		let (sharing_mode, queue_family_indices) = if distinct_families {
			(
				ash::vk::SharingMode::CONCURRENT,
				sharing_families.as_slice(),
			)
		} else {
			(ash::vk::SharingMode::EXCLUSIVE, [].as_slice())
		};
		let create_info = ash::vk::SwapchainCreateInfoKHR::default()
			.surface(surface)
			.min_image_count(config.image_count)
			.image_format(config.format.format)
			.image_color_space(config.format.color_space)
			.image_extent(config.extent)
			.image_array_layers(1)
			.image_usage(
				ash::vk::ImageUsageFlags::COLOR_ATTACHMENT | ash::vk::ImageUsageFlags::TRANSFER_DST,
			)
			.image_sharing_mode(sharing_mode)
			.queue_family_indices(queue_family_indices)
			.pre_transform(config.pre_transform)
			.composite_alpha(config.composite_alpha)
			.present_mode(config.present_mode)
			.clipped(true);
		let handle = unsafe { loader.create_swapchain(&create_info, None) }
			.map_err(|source| Error::backend_failure("Vulkan", "swapchain creation", source))?;
		let images = match unsafe { loader.get_swapchain_images(handle) } {
			Ok(images) => images,
			Err(source) => {
				unsafe { loader.destroy_swapchain(handle, None) };
				return Err(Error::backend_failure(
					"Vulkan",
					"swapchain image query",
					source,
				));
			}
		};
		let mut views = Vec::with_capacity(images.len());
		for image in &images {
			let view_info = ash::vk::ImageViewCreateInfo::default()
				.image(*image)
				.view_type(ash::vk::ImageViewType::TYPE_2D)
				.format(config.format.format)
				.subresource_range(
					ash::vk::ImageSubresourceRange::default()
						.aspect_mask(ash::vk::ImageAspectFlags::COLOR)
						.level_count(1)
						.layer_count(1),
				);
			match unsafe { device.raw().create_image_view(&view_info, None) } {
				Ok(view) => views.push(view),
				Err(source) => {
					for view in views {
						unsafe { device.raw().destroy_image_view(view, None) }
					}
					unsafe { loader.destroy_swapchain(handle, None) };
					return Err(Error::backend_failure(
						"Vulkan",
						"swapchain image-view creation",
						source,
					));
				}
			}
		}
		let mut frames = Vec::with_capacity(FRAMES_IN_FLIGHT);
		for _ in 0..FRAMES_IN_FLIGHT {
			let semaphore_info = ash::vk::SemaphoreCreateInfo::default();
			let image_available = match unsafe { device.raw().create_semaphore(&semaphore_info, None) } {
				Ok(semaphore) => semaphore,
				Err(source) => {
					destroy_partial_swapchain(device.raw(), &mut views, &mut frames, &loader, handle);
					return Err(Error::backend_failure(
						"Vulkan",
						"present image-available semaphore creation",
						source,
					));
				}
			};
			let render_finished = match unsafe { device.raw().create_semaphore(&semaphore_info, None) } {
				Ok(semaphore) => semaphore,
				Err(source) => {
					unsafe { device.raw().destroy_semaphore(image_available, None) };
					destroy_partial_swapchain(device.raw(), &mut views, &mut frames, &loader, handle);
					return Err(Error::backend_failure(
						"Vulkan",
						"present render-finished semaphore creation",
						source,
					));
				}
			};
			let fence_info =
				ash::vk::FenceCreateInfo::default().flags(ash::vk::FenceCreateFlags::SIGNALED);
			let in_flight = match unsafe { device.raw().create_fence(&fence_info, None) } {
				Ok(fence) => fence,
				Err(source) => {
					unsafe {
						device.raw().destroy_semaphore(render_finished, None);
						device.raw().destroy_semaphore(image_available, None);
					}
					destroy_partial_swapchain(device.raw(), &mut views, &mut frames, &loader, handle);
					return Err(Error::backend_failure(
						"Vulkan",
						"present in-flight fence creation",
						source,
					));
				}
			};
			frames.push(PresentationFrameSync {
				image_available,
				render_finished,
				in_flight,
				command: None,
				source: None,
			});
		}
		Ok(Self {
			_device: device.clone(),
			loader,
			handle,
			extent: config.extent,
			images,
			views,
			frames,
			next_frame: 0,
		})
	}
}

impl PresentationSwapchain {
	pub(in crate::runtime) fn is_valid(&self) -> bool {
		self.handle != ash::vk::SwapchainKHR::null()
	}

	pub(in crate::runtime) fn image_count(&self) -> usize {
		self.views.len()
	}

	pub(in crate::runtime) fn extent_px(&self) -> [u32; 2] {
		[self.extent.width, self.extent.height]
	}

	pub(in crate::runtime) fn frames_in_flight(&self) -> usize {
		self.frames.len()
	}

	pub(in crate::runtime) fn has_submitted_frames(&self) -> bool {
		self.frames.iter().any(|frame| frame.command.is_some())
	}

	pub(in crate::runtime) fn present_clear(&mut self, color: [f32; 4]) -> Result<bool> {
		let slot = self.next_frame;
		let frame = &mut self.frames[slot];
		unsafe {
			self
				._device
				.raw()
				.wait_for_fences(&[frame.in_flight], true, u64::MAX)
		}
		.map_err(|source| Error::backend_failure("Vulkan", "presentation fence wait", source))?;
		if let Some(command) = frame.command.take() {
			self._device.free(command);
		}
		frame.source = None;
		let (image_index, acquire_suboptimal) = match unsafe {
			self.loader.acquire_next_image(
				self.handle,
				u64::MAX,
				frame.image_available,
				ash::vk::Fence::null(),
			)
		} {
			Ok(acquired) => acquired,
			Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
			Err(source) => {
				return Err(Error::backend_failure(
					"Vulkan",
					"swapchain image acquisition",
					source,
				));
			}
		};
		let image = *self.images.get(image_index as usize).ok_or_else(|| {
			Error::invalid_argument("Vulkan returned an out-of-range swapchain image index")
		})?;
		unsafe { self._device.raw().reset_fences(&[frame.in_flight]) }
			.map_err(|source| Error::backend_failure("Vulkan", "presentation fence reset", source))?;
		let command = self._device.record_present_clear(image, color)?;
		self._device.submit_present(
			&command,
			frame.image_available,
			frame.render_finished,
			frame.in_flight,
		)?;
		frame.command = Some(command);
		let waits = [frame.render_finished];
		let chains = [self.handle];
		let indices = [image_index];
		let info = ash::vk::PresentInfoKHR::default()
			.wait_semaphores(&waits)
			.swapchains(&chains)
			.image_indices(&indices);
		let present_suboptimal = match unsafe {
			self
				.loader
				.queue_present(self._device.graphics_queue_handle()?, &info)
		} {
			Ok(suboptimal) => suboptimal,
			Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
			Err(source) => {
				return Err(Error::backend_failure(
					"Vulkan",
					"queue presentation",
					source,
				));
			}
		};
		self.next_frame = (self.next_frame + 1) % self.frames.len();
		Ok(acquire_suboptimal || present_suboptimal)
	}

	pub(in crate::runtime) fn present_render_target(
		&mut self,
		source: ash::vk::Image,
		source_extent: [u32; 2],
		retention: Rc<dyn std::any::Any>,
	) -> Result<bool> {
		let slot = self.next_frame;
		let frame = &mut self.frames[slot];
		unsafe {
			self
				._device
				.raw()
				.wait_for_fences(&[frame.in_flight], true, u64::MAX)
		}
		.map_err(|source| Error::backend_failure("Vulkan", "presentation fence wait", source))?;
		if let Some(command) = frame.command.take() {
			self._device.free(command);
		}
		frame.source = None;
		let (image_index, acquire_suboptimal) = match unsafe {
			self.loader.acquire_next_image(
				self.handle,
				u64::MAX,
				frame.image_available,
				ash::vk::Fence::null(),
			)
		} {
			Ok(acquired) => acquired,
			Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
			Err(source) => {
				return Err(Error::backend_failure(
					"Vulkan",
					"swapchain image acquisition",
					source,
				));
			}
		};
		let target = *self.images.get(image_index as usize).ok_or_else(|| {
			Error::invalid_argument("Vulkan returned an out-of-range swapchain image index")
		})?;
		unsafe { self._device.raw().reset_fences(&[frame.in_flight]) }
			.map_err(|source| Error::backend_failure("Vulkan", "presentation fence reset", source))?;
		let command = self._device.record_present_blit(
			source,
			source_extent,
			target,
			[self.extent.width, self.extent.height],
		)?;
		self._device.submit_present(
			&command,
			frame.image_available,
			frame.render_finished,
			frame.in_flight,
		)?;
		frame.command = Some(command);
		frame.source = Some(retention);
		let waits = [frame.render_finished];
		let chains = [self.handle];
		let indices = [image_index];
		let info = ash::vk::PresentInfoKHR::default()
			.wait_semaphores(&waits)
			.swapchains(&chains)
			.image_indices(&indices);
		let present_suboptimal = match unsafe {
			self
				.loader
				.queue_present(self._device.graphics_queue_handle()?, &info)
		} {
			Ok(suboptimal) => suboptimal,
			Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
			Err(source) => {
				return Err(Error::backend_failure(
					"Vulkan",
					"queue presentation",
					source,
				));
			}
		};
		self.next_frame = (self.next_frame + 1) % self.frames.len();
		Ok(acquire_suboptimal || present_suboptimal)
	}

	pub(in crate::runtime) fn present_texture(
		&mut self,
		texture: Rc<crate::runtime::NativeRgbaImage>,
	) -> Result<bool> {
		let slot = self.next_frame;
		let frame = &mut self.frames[slot];
		unsafe {
			self
				._device
				.raw()
				.wait_for_fences(&[frame.in_flight], true, u64::MAX)
		}
		.map_err(|source| Error::backend_failure("Vulkan", "presentation fence wait", source))?;
		if let Some(command) = frame.command.take() {
			self._device.free(command);
		}
		frame.source = None;
		let (image_index, acquire_suboptimal) = match unsafe {
			self.loader.acquire_next_image(
				self.handle,
				u64::MAX,
				frame.image_available,
				ash::vk::Fence::null(),
			)
		} {
			Ok(acquired) => acquired,
			Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
			Err(source) => {
				return Err(Error::backend_failure(
					"Vulkan",
					"swapchain image acquisition",
					source,
				));
			}
		};
		let image = *self.images.get(image_index as usize).ok_or_else(|| {
			Error::invalid_argument("Vulkan returned an out-of-range swapchain image index")
		})?;
		unsafe { self._device.raw().reset_fences(&[frame.in_flight]) }
			.map_err(|source| Error::backend_failure("Vulkan", "presentation fence reset", source))?;
		let command = self._device.record_present_blit(
			texture.handle,
			[texture.width, texture.height],
			image,
			[self.extent.width, self.extent.height],
		)?;
		self._device.submit_present(
			&command,
			frame.image_available,
			frame.render_finished,
			frame.in_flight,
		)?;
		frame.command = Some(command);
		frame.source = Some(texture);
		let waits = [frame.render_finished];
		let chains = [self.handle];
		let indices = [image_index];
		let info = ash::vk::PresentInfoKHR::default()
			.wait_semaphores(&waits)
			.swapchains(&chains)
			.image_indices(&indices);
		let present_suboptimal = match unsafe {
			self
				.loader
				.queue_present(self._device.graphics_queue_handle()?, &info)
		} {
			Ok(suboptimal) => suboptimal,
			Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
			Err(source) => {
				return Err(Error::backend_failure(
					"Vulkan",
					"queue presentation",
					source,
				));
			}
		};
		self.next_frame = (self.next_frame + 1) % self.frames.len();
		Ok(acquire_suboptimal || present_suboptimal)
	}

	/// Compose the UI onto the swapchain in a single frame without a readback.
	///
	/// `record` fills one compute command buffer with UI dispatch work that
	/// writes into `compose_image` (RGBA8 STORAGE_IMAGE, UNDEFINED at entry).
	/// The swapchain acquire, compute barrier, blit, and present are submitted
	/// on the graphics queue as one serialized command buffer — no host readback.
	pub(in crate::runtime) fn present_compose(
		&mut self,
		compose_image: ash::vk::Image,
		compose_width: u32,
		compose_height: u32,
		descriptor_set: ash::vk::DescriptorSet,
		pipeline_layout: ash::vk::PipelineLayout,
		record: impl FnOnce(&ash::Device, ash::vk::CommandBuffer) -> Result<()>,
	) -> Result<bool> {
		let slot = self.next_frame;
		let frame = &mut self.frames[slot];
		unsafe {
			self
				._device
				.raw()
				.wait_for_fences(&[frame.in_flight], true, u64::MAX)
		}
		.map_err(|source| Error::backend_failure("Vulkan", "presentation fence wait", source))?;
		if let Some(command) = frame.command.take() {
			self._device.free(command);
		}
		frame.source = None;
		let (image_index, acquire_suboptimal) = match unsafe {
			self.loader.acquire_next_image(
				self.handle,
				u64::MAX,
				frame.image_available,
				ash::vk::Fence::null(),
			)
		} {
			Ok(acquired) => acquired,
			Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
			Err(source) => {
				return Err(Error::backend_failure(
					"Vulkan",
					"swapchain image acquisition",
					source,
				));
			}
		};
		let target = *self.images.get(image_index as usize).ok_or_else(|| {
			Error::invalid_argument("Vulkan returned an out-of-range swapchain image index")
		})?;
		unsafe { self._device.raw().reset_fences(&[frame.in_flight]) }
			.map_err(|source| Error::backend_failure("Vulkan", "presentation fence reset", source))?;

		let swapchain_extent = [self.extent.width, self.extent.height];
		let device = self._device.clone();

		let command = device.record_compute_commands(|cmd| {
			// Bind the shared bindless descriptor set.
			unsafe {
				device.raw().cmd_bind_descriptor_sets(
					cmd,
					ash::vk::PipelineBindPoint::COMPUTE,
					pipeline_layout,
					0,
					&[descriptor_set],
					&[],
				);
			}
			// Run UI compute dispatches (writes compose_image).
			record(device.raw(), cmd)?;

			let subresource_range = ash::vk::ImageSubresourceRange::default()
				.aspect_mask(ash::vk::ImageAspectFlags::COLOR)
				.level_count(1)
				.layer_count(1);
			let subresource_layers = ash::vk::ImageSubresourceLayers::default()
				.aspect_mask(ash::vk::ImageAspectFlags::COLOR)
				.layer_count(1);

			// Transition compose: GENERAL → TRANSFER_SRC
			// Transition swapchain target: UNDEFINED → TRANSFER_DST
			let compose_to_src = ash::vk::ImageMemoryBarrier2::default()
				.src_stage_mask(ash::vk::PipelineStageFlags2::COMPUTE_SHADER)
				.src_access_mask(ash::vk::AccessFlags2::SHADER_STORAGE_WRITE)
				.dst_stage_mask(ash::vk::PipelineStageFlags2::BLIT)
				.dst_access_mask(ash::vk::AccessFlags2::TRANSFER_READ)
				.old_layout(ash::vk::ImageLayout::GENERAL)
				.new_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
				.image(compose_image)
				.subresource_range(subresource_range);
			let swapchain_to_dst = ash::vk::ImageMemoryBarrier2::default()
				.src_stage_mask(ash::vk::PipelineStageFlags2::NONE)
				.dst_stage_mask(ash::vk::PipelineStageFlags2::BLIT)
				.dst_access_mask(ash::vk::AccessFlags2::TRANSFER_WRITE)
				.old_layout(ash::vk::ImageLayout::UNDEFINED)
				.new_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
				.image(target)
				.subresource_range(subresource_range);
			unsafe {
				device.pipeline_barrier(
					cmd,
					&ash::vk::DependencyInfo::default()
						.image_memory_barriers(&[compose_to_src, swapchain_to_dst]),
				);
			}

			// Blit (with linear filter to handle DPI scaling).
			let blit = ash::vk::ImageBlit::default()
				.src_subresource(subresource_layers)
				.src_offsets([
					ash::vk::Offset3D::default(),
					ash::vk::Offset3D {
						x: compose_width as i32,
						y: compose_height as i32,
						z: 1,
					},
				])
				.dst_subresource(subresource_layers)
				.dst_offsets([
					ash::vk::Offset3D::default(),
					ash::vk::Offset3D {
						x: swapchain_extent[0] as i32,
						y: swapchain_extent[1] as i32,
						z: 1,
					},
				]);
			unsafe {
				device.raw().cmd_blit_image(
					cmd,
					compose_image,
					ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
					target,
					ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
					&[blit],
					ash::vk::Filter::LINEAR,
				);
			}

			// Transition swapchain → PRESENT_SRC.
			let swapchain_to_present = ash::vk::ImageMemoryBarrier2::default()
				.src_stage_mask(ash::vk::PipelineStageFlags2::BLIT)
				.src_access_mask(ash::vk::AccessFlags2::TRANSFER_WRITE)
				.dst_stage_mask(ash::vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
				.old_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
				.new_layout(ash::vk::ImageLayout::PRESENT_SRC_KHR)
				.image(target)
				.subresource_range(subresource_range);
			unsafe {
				device.pipeline_barrier(
					cmd,
					&ash::vk::DependencyInfo::default()
						.image_memory_barriers(std::slice::from_ref(&swapchain_to_present)),
				);
			}
			Ok(())
		})?;

		self._device.submit_present(
			&command,
			frame.image_available,
			frame.render_finished,
			frame.in_flight,
		)?;
		frame.command = Some(command);

		let waits = [frame.render_finished];
		let chains = [self.handle];
		let indices = [image_index];
		let info = ash::vk::PresentInfoKHR::default()
			.wait_semaphores(&waits)
			.swapchains(&chains)
			.image_indices(&indices);
		let present_suboptimal = match unsafe {
			self
				.loader
				.queue_present(self._device.graphics_queue_handle()?, &info)
		} {
			Ok(suboptimal) => suboptimal,
			Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
			Err(source) => {
				return Err(Error::backend_failure(
					"Vulkan",
					"queue presentation",
					source,
				));
			}
		};
		self.next_frame = (self.next_frame + 1) % self.frames.len();
		Ok(acquire_suboptimal || present_suboptimal)
	}

	pub(in crate::runtime) fn wait_idle(&mut self) -> Result<()> {
		unsafe { self._device.raw().device_wait_idle() }.map_err(|source| {
			Error::backend_failure("Vulkan", "presentation device-idle wait", source)
		})?;
		for frame in &mut self.frames {
			if let Some(command) = frame.command.take() {
				self._device.free(command);
			}
			frame.source = None;
		}
		Ok(())
	}
}

fn destroy_partial_swapchain(
	device: &ash::Device,
	views: &mut Vec<ash::vk::ImageView>,
	frames: &mut Vec<PresentationFrameSync>,
	loader: &ash::khr::swapchain::Device,
	handle: ash::vk::SwapchainKHR,
) {
	for frame in frames.drain(..) {
		unsafe {
			device.destroy_fence(frame.in_flight, None);
			device.destroy_semaphore(frame.render_finished, None);
			device.destroy_semaphore(frame.image_available, None);
		}
	}
	for view in views.drain(..) {
		unsafe { device.destroy_image_view(view, None) };
	}
	unsafe { loader.destroy_swapchain(handle, None) };
}

impl Drop for PresentationSwapchain {
	fn drop(&mut self) {
		debug_assert!(
			!self.has_submitted_frames(),
			"submitted presentation work must be retired by close_swapchain before destruction"
		);
		for frame in self.frames.drain(..) {
			unsafe {
				self._device.raw().destroy_fence(frame.in_flight, None);
				self
					._device
					.raw()
					.destroy_semaphore(frame.render_finished, None);
				self
					._device
					.raw()
					.destroy_semaphore(frame.image_available, None);
			}
		}
		for view in self.views.drain(..) {
			unsafe { self._device.raw().destroy_image_view(view, None) };
		}
		unsafe { self.loader.destroy_swapchain(self.handle, None) };
	}
}
