//! Device-owned swapchain resources, frame synchronization and presentation.

use std::{rc::Rc, sync::Arc};

use crate::runtime::{Device, RecordedCommandBuffer};
use crate::{Error, Result};

/// Vulkan values selected from one exact surface query for swapchain creation.
pub(in crate::runtime) struct SwapchainConfig {
	pub(in crate::runtime) extent: vk::Extent2D,
	pub(in crate::runtime) image_count: u32,
	pub(in crate::runtime) format: vk::SurfaceFormatKHR,
	pub(in crate::runtime) present_mode: vk::PresentModeKHR,
	pub(in crate::runtime) transparent: bool,
	pub(in crate::runtime) composite_alpha: vk::CompositeAlphaFlagsKHR,
	pub(in crate::runtime) pre_transform: vk::SurfaceTransformFlagsKHR,
}

pub(in crate::runtime) struct PresentationSwapchain {
	_device: Arc<Device>,
	handle: vk::SwapchainKHR,
	extent: vk::Extent2D,
	images: Vec<vk::Image>,
	views: Vec<vk::ImageView>,
	// Reacquiring an image proves its previous presentation wait is consumed.
	present_finished: Vec<vk::Semaphore>,
	frames: Vec<PresentationFrameSync>,
	next_frame: usize,
	composition_failed: bool,
	transparent: bool,
}

struct PresentationFrameSync {
	image_available: vk::Semaphore,
	in_flight: vk::Fence,
	command: Option<RecordedCommandBuffer>,
	source: Option<Rc<dyn std::any::Any>>,
}

const FRAMES_IN_FLIGHT: usize = 2;

impl PresentationSwapchain {
	pub(in crate::runtime) fn is_transparent(&self) -> bool {
		self.transparent
	}

	pub(in crate::runtime) fn create(
		device: &Arc<Device>,
		surface: vk::SurfaceKHR,
		present_queue_family: Option<u32>,
		config: SwapchainConfig,
	) -> Result<Self> {
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
			(vk::SharingMode::CONCURRENT, sharing_families.as_slice())
		} else {
			(vk::SharingMode::EXCLUSIVE, [].as_slice())
		};
		let create_info = vk::SwapchainCreateInfoKHR::default()
			.surface(surface)
			.min_image_count(config.image_count)
			.image_format(config.format.format)
			.image_color_space(config.format.color_space)
			.image_extent(config.extent)
			.image_array_layers(1)
			.image_usage(
				vk::ImageUsageFlagBits::COLOR_ATTACHMENT_BIT | vk::ImageUsageFlagBits::TRANSFER_DST_BIT,
			)
			.image_sharing_mode(sharing_mode)
			.queue_family_indices(queue_family_indices)
			.map_err(Error::from)?
			.pre_transform(config.pre_transform)
			.composite_alpha(config.composite_alpha)
			.present_mode(config.present_mode)
			.clipped(vk::TRUE);
		// SAFETY: `create_info` references live for the call; no callbacks are passed.
		let mut handle = vk::SwapchainKHR::default();
		match unsafe {
			device
				.raw()
				.create_swapchain_khr(&create_info, std::ptr::null(), &mut handle)
		} {
			Ok(vk::Result::SUCCESS) => {}
			other => {
				return Err(Error::backend_dispatch_failure(
					"Vulkan",
					"swapchain creation",
					other,
				));
			}
		}
		let images = match swapchain_images(device.raw(), handle) {
			Ok(images) => images,
			Err(source) => {
				// SAFETY: swapchain was just created; no images were bound.
				let _ = unsafe { device.raw().destroy_swapchain_khr(handle, std::ptr::null()) };
				return Err(Error::backend_failure(
					"Vulkan",
					"swapchain image query",
					source,
				));
			}
		};
		let mut views = Vec::with_capacity(images.len());
		for image in &images {
			let view_info = vk::ImageViewCreateInfo::default()
				.image(*image)
				.view_type(vk::ImageViewType::_2D)
				.format(config.format.format)
				.subresource_range(
					vk::ImageSubresourceRange::default()
						.aspect_mask(vk::ImageAspectFlagBits::COLOR_BIT)
						.level_count(1)
						.layer_count(1),
				);
			match unsafe { device.raw().create_image_view(&view_info, None) } {
				Ok(view) => views.push(view),
				Err(source) => {
					for view in views {
						let _ = unsafe { device.raw().destroy_image_view(view, None) };
					}
					let _ = unsafe { device.raw().destroy_swapchain_khr(handle, std::ptr::null()) };
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
			let semaphore_info = vk::SemaphoreCreateInfo::default();
			let image_available = match unsafe { device.raw().create_semaphore(&semaphore_info, None) } {
				Ok(semaphore) => semaphore,
				Err(source) => {
					destroy_partial_swapchain(device.raw(), &mut views, &mut frames, handle);
					return Err(Error::backend_failure(
						"Vulkan",
						"present image-available semaphore creation",
						source,
					));
				}
			};
			let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlagBits::SIGNALED_BIT);
			let in_flight = match unsafe { device.raw().create_fence(&fence_info, None) } {
				Ok(fence) => fence,
				Err(source) => {
					unsafe {
						device.raw().destroy_semaphore(image_available, None)?;
					}
					destroy_partial_swapchain(device.raw(), &mut views, &mut frames, handle);
					return Err(Error::backend_failure(
						"Vulkan",
						"present in-flight fence creation",
						source,
					));
				}
			};
			frames.push(PresentationFrameSync {
				image_available,
				in_flight,
				command: None,
				source: None,
			});
		}
		let mut present_finished = Vec::with_capacity(images.len());
		for _ in &images {
			// SAFETY: each image owns one binary semaphore until explicit WSI retirement.
			match unsafe {
				device
					.raw()
					.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
			} {
				Ok(semaphore) => present_finished.push(semaphore),
				Err(source) => {
					for semaphore in present_finished {
						unsafe {
							device.raw().destroy_semaphore(semaphore, None)?;
						}
					}
					destroy_partial_swapchain(device.raw(), &mut views, &mut frames, handle);
					return Err(Error::backend_failure(
						"Vulkan",
						"present image semaphore creation",
						source,
					));
				}
			}
		}
		Ok(Self {
			_device: device.clone(),
			handle,
			extent: config.extent,
			images,
			views,
			present_finished,
			frames,
			next_frame: 0,
			composition_failed: false,
			transparent: config.transparent,
		})
	}
}

impl PresentationSwapchain {
	pub(in crate::runtime) fn is_valid(&self) -> bool {
		self.handle != vk::SwapchainKHR::default()
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
		if self.composition_failed {
			return Err(Error::failed_precondition(
				"presentation composition failed after acquisition; recreate the swapchain",
			));
		}
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
		let mut image_index = 0_u32;
		let acquire_result = unsafe {
			self._device.raw().acquire_next_image_khr(
				self.handle,
				u64::MAX,
				frame.image_available,
				vk::Fence::default(),
				&mut image_index,
			)
		};
		let acquire_suboptimal = match acquire_result {
			Ok(vk::Result::SUCCESS) => false,
			Ok(vk::Result::SUBOPTIMAL_KHR) => true,
			Err(vk::CallError::Vulkan(vk::Result::ERROR_OUT_OF_DATE_KHR))
			| Ok(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
			other => {
				return Err(Error::backend_dispatch_failure(
					"Vulkan",
					"swapchain image acquisition",
					other,
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
			self.present_finished[image_index as usize],
			frame.in_flight,
		)?;
		frame.command = Some(command);
		let waits = [self.present_finished[image_index as usize]];
		let chains = [self.handle];
		let indices = [image_index];
		let info = vk::PresentInfoKHR::default()
			.wait_semaphores(&waits)
			.map_err(Error::from)?
			.swapchains(&chains, &indices, None)
			.map_err(Error::from)?;
		let present_suboptimal = match unsafe {
			self._device.raw().queue_present_khr(
				self._device.graphics_queue_handle()?,
				std::ptr::from_ref(&info),
			)
		} {
			Ok(vk::Result::SUCCESS) => false,
			Ok(vk::Result::SUBOPTIMAL_KHR) => true,
			Err(vk::CallError::Vulkan(vk::Result::ERROR_OUT_OF_DATE_KHR))
			| Ok(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
			other => {
				return Err(Error::backend_dispatch_failure(
					"Vulkan",
					"queue presentation",
					other,
				));
			}
		};
		self.next_frame = (self.next_frame + 1) % self.frames.len();
		Ok(acquire_suboptimal || present_suboptimal)
	}

	pub(in crate::runtime) fn present_render_target(
		&mut self,
		source: vk::Image,
		source_extent: [u32; 2],
		retention: Rc<dyn std::any::Any>,
	) -> Result<bool> {
		if self.composition_failed {
			return Err(Error::failed_precondition(
				"presentation composition failed after acquisition; recreate the swapchain",
			));
		}
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
		let mut image_index = 0_u32;
		let acquire_result = unsafe {
			self._device.raw().acquire_next_image_khr(
				self.handle,
				u64::MAX,
				frame.image_available,
				vk::Fence::default(),
				&mut image_index,
			)
		};
		let acquire_suboptimal = match acquire_result {
			Ok(vk::Result::SUCCESS) => false,
			Ok(vk::Result::SUBOPTIMAL_KHR) => true,
			Err(vk::CallError::Vulkan(vk::Result::ERROR_OUT_OF_DATE_KHR))
			| Ok(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
			other => {
				return Err(Error::backend_dispatch_failure(
					"Vulkan",
					"swapchain image acquisition",
					other,
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
			self.present_finished[image_index as usize],
			frame.in_flight,
		)?;
		frame.command = Some(command);
		frame.source = Some(retention);
		let waits = [self.present_finished[image_index as usize]];
		let chains = [self.handle];
		let indices = [image_index];
		let info = vk::PresentInfoKHR::default()
			.wait_semaphores(&waits)
			.map_err(Error::from)?
			.swapchains(&chains, &indices, None)
			.map_err(Error::from)?;
		let present_suboptimal = match unsafe {
			self._device.raw().queue_present_khr(
				self._device.graphics_queue_handle()?,
				std::ptr::from_ref(&info),
			)
		} {
			Ok(vk::Result::SUCCESS) => false,
			Ok(vk::Result::SUBOPTIMAL_KHR) => true,
			Err(vk::CallError::Vulkan(vk::Result::ERROR_OUT_OF_DATE_KHR))
			| Ok(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
			other => {
				return Err(Error::backend_dispatch_failure(
					"Vulkan",
					"queue presentation",
					other,
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
		if self.composition_failed {
			return Err(Error::failed_precondition(
				"presentation composition failed after acquisition; recreate the swapchain",
			));
		}
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
		let mut image_index = 0_u32;
		let acquire_result = unsafe {
			self._device.raw().acquire_next_image_khr(
				self.handle,
				u64::MAX,
				frame.image_available,
				vk::Fence::default(),
				&mut image_index,
			)
		};
		let acquire_suboptimal = match acquire_result {
			Ok(vk::Result::SUCCESS) => false,
			Ok(vk::Result::SUBOPTIMAL_KHR) => true,
			Err(vk::CallError::Vulkan(vk::Result::ERROR_OUT_OF_DATE_KHR))
			| Ok(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
			other => {
				return Err(Error::backend_dispatch_failure(
					"Vulkan",
					"swapchain image acquisition",
					other,
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
			self.present_finished[image_index as usize],
			frame.in_flight,
		)?;
		frame.command = Some(command);
		frame.source = Some(texture);
		let waits = [self.present_finished[image_index as usize]];
		let chains = [self.handle];
		let indices = [image_index];
		let info = vk::PresentInfoKHR::default()
			.wait_semaphores(&waits)
			.map_err(Error::from)?
			.swapchains(&chains, &indices, None)
			.map_err(Error::from)?;
		let present_suboptimal = match unsafe {
			self._device.raw().queue_present_khr(
				self._device.graphics_queue_handle()?,
				std::ptr::from_ref(&info),
			)
		} {
			Ok(vk::Result::SUCCESS) => false,
			Ok(vk::Result::SUBOPTIMAL_KHR) => true,
			Err(vk::CallError::Vulkan(vk::Result::ERROR_OUT_OF_DATE_KHR))
			| Ok(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
			other => {
				return Err(Error::backend_dispatch_failure(
					"Vulkan",
					"queue presentation",
					other,
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
	#[allow(clippy::too_many_arguments)]
	pub(in crate::runtime) fn present_compose(
		&mut self,
		compose_image: vk::Image,
		compose_width: u32,
		compose_height: u32,
		descriptor_set: vk::DescriptorSet,
		pipeline_layout: vk::PipelineLayout,
		retention: Option<Rc<dyn std::any::Any>>,
		record: impl FnOnce(&vk::Device, vk::CommandBuffer) -> Result<()>,
	) -> Result<bool> {
		if self.composition_failed {
			return Err(Error::failed_precondition(
				"presentation composition failed after acquisition; recreate the swapchain",
			));
		}
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
		let mut image_index = 0_u32;
		let acquire_result = unsafe {
			self._device.raw().acquire_next_image_khr(
				self.handle,
				u64::MAX,
				frame.image_available,
				vk::Fence::default(),
				&mut image_index,
			)
		};
		let acquire_suboptimal = match acquire_result {
			Ok(vk::Result::SUCCESS) => false,
			Ok(vk::Result::SUBOPTIMAL_KHR) => true,
			Err(vk::CallError::Vulkan(vk::Result::ERROR_OUT_OF_DATE_KHR))
			| Ok(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
			other => {
				return Err(Error::backend_dispatch_failure(
					"Vulkan",
					"swapchain image acquisition",
					other,
				));
			}
		};
		// Any subsequent error requires recreation: the acquisition semaphore
		// cannot be reused until a submission has consumed its signal.
		self.composition_failed = true;
		let target = *self.images.get(image_index as usize).ok_or_else(|| {
			Error::invalid_argument("Vulkan returned an out-of-range swapchain image index")
		})?;

		let swapchain_extent = [self.extent.width, self.extent.height];
		let device = self._device.clone();

		let command = device.record_presentation_commands(|cmd| {
			// Strict binds a shared set; bounded composition binds per command.
			if descriptor_set != vk::DescriptorSet::default() {
				unsafe {
					device.raw().cmd_bind_descriptor_sets(
						cmd,
						vk::PipelineBindPoint::COMPUTE,
						pipeline_layout,
						0,
						&[descriptor_set],
						&[],
					)?;
				}
			}
			// Run UI compute dispatches (writes compose_image).
			record(device.raw(), cmd)?;

			let subresource_range = vk::ImageSubresourceRange::default()
				.aspect_mask(vk::ImageAspectFlagBits::COLOR_BIT)
				.level_count(1)
				.layer_count(1);
			let subresource_layers = vk::ImageSubresourceLayers::default()
				.aspect_mask(vk::ImageAspectFlagBits::COLOR_BIT)
				.layer_count(1);

			// Transition compose: GENERAL → TRANSFER_SRC
			// Transition swapchain target: UNDEFINED → TRANSFER_DST
			let compose_to_src = vk::ImageMemoryBarrier2::default()
				.src_stage_mask(vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT)
				.src_access_mask(vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT)
				.dst_stage_mask(vk::PipelineStageFlagBits2::BLIT_BIT)
				.dst_access_mask(vk::AccessFlagBits2::TRANSFER_READ_BIT)
				.old_layout(vk::ImageLayout::GENERAL)
				.new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
				.image(compose_image)
				.subresource_range(subresource_range);
			let swapchain_to_dst = vk::ImageMemoryBarrier2::default()
				.src_stage_mask(vk::PipelineStageFlagBits2::NONE)
				.dst_stage_mask(vk::PipelineStageFlagBits2::BLIT_BIT)
				.dst_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
				.old_layout(vk::ImageLayout::UNDEFINED)
				.new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
				.image(target)
				.subresource_range(subresource_range);
			unsafe {
				device.pipeline_barrier(
					cmd,
					&(vk::DependencyInfo::default())
						.image_memory_barriers(&[compose_to_src, swapchain_to_dst])
						.map_err(Error::from)?,
				)?;
			}

			// Blit (with linear filter to handle DPI scaling).
			let blit = vk::ImageBlit::default()
				.src_subresource(subresource_layers)
				.src_offsets([
					vk::Offset3D::default(),
					vk::Offset3D {
						x: compose_width as i32,
						y: compose_height as i32,
						z: 1,
					},
				])
				.dst_subresource(subresource_layers)
				.dst_offsets([
					vk::Offset3D::default(),
					vk::Offset3D {
						x: swapchain_extent[0] as i32,
						y: swapchain_extent[1] as i32,
						z: 1,
					},
				]);
			unsafe {
				device.raw().cmd_blit_image(
					cmd,
					compose_image,
					vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
					target,
					vk::ImageLayout::TRANSFER_DST_OPTIMAL,
					&[blit],
					vk::Filter::LINEAR,
				)?;
			}

			// Transition swapchain → PRESENT_SRC.
			let swapchain_to_present = vk::ImageMemoryBarrier2::default()
				.src_stage_mask(vk::PipelineStageFlagBits2::BLIT_BIT)
				.src_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
				.dst_stage_mask(vk::PipelineStageFlagBits2::BOTTOM_OF_PIPE_BIT)
				.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
				.new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
				.image(target)
				.subresource_range(subresource_range);
			unsafe {
				device.pipeline_barrier(
					cmd,
					&(vk::DependencyInfo::default())
						.image_memory_barriers(std::slice::from_ref(&swapchain_to_present))
						.map_err(Error::from)?,
				)?;
			}
			Ok(())
		})?;

		// Retain resources before submission, including uncertain device-loss
		// failures. No error path drops work that may still reference them.
		frame.command = Some(command);
		frame.source = retention;
		unsafe { self._device.raw().reset_fences(&[frame.in_flight]) }
			.map_err(|source| Error::backend_failure("Vulkan", "presentation fence reset", source))?;
		self._device.submit_present(
			frame
				.command
				.as_ref()
				.expect("recorded composition retained"),
			frame.image_available,
			self.present_finished[image_index as usize],
			frame.in_flight,
		)?;

		let waits = [self.present_finished[image_index as usize]];
		let chains = [self.handle];
		let indices = [image_index];
		let info = vk::PresentInfoKHR::default()
			.wait_semaphores(&waits)
			.map_err(Error::from)?
			.swapchains(&chains, &indices, None)
			.map_err(Error::from)?;
		let present_suboptimal = match unsafe {
			self._device.raw().queue_present_khr(
				self._device.graphics_queue_handle()?,
				std::ptr::from_ref(&info),
			)
		} {
			Ok(vk::Result::SUCCESS) => false,
			Ok(vk::Result::SUBOPTIMAL_KHR) => true,
			Err(vk::CallError::Vulkan(vk::Result::ERROR_OUT_OF_DATE_KHR))
			| Ok(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
			other => {
				return Err(Error::backend_dispatch_failure(
					"Vulkan",
					"queue presentation",
					other,
				));
			}
		};
		self.composition_failed = false;
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

fn swapchain_images(device: &vk::Device, handle: vk::SwapchainKHR) -> Result<Vec<vk::Image>> {
	let mut count = 0_u32;
	// SAFETY: null output pointer requests only the element count.
	match unsafe { device.get_swapchain_images_khr(handle, &mut count, std::ptr::null_mut()) } {
		Ok(vk::Result::SUCCESS) => {}
		other => {
			return Err(Error::backend_dispatch_failure(
				"Vulkan",
				"swapchain image count",
				other,
			));
		}
	}
	let mut images = vec![vk::Image::default(); count as usize];
	if count == 0 {
		return Ok(images);
	}
	// SAFETY: `images` provides `count` writable handles; Vulkan writes at most
	// the passed count and reports the actual number through the out count.
	match unsafe { device.get_swapchain_images_khr(handle, &mut count, images.as_mut_ptr()) } {
		Ok(vk::Result::SUCCESS) => {
			images.truncate(count as usize);
			Ok(images)
		}
		other => Err(Error::backend_dispatch_failure(
			"Vulkan",
			"swapchain image query",
			other,
		)),
	}
}

fn destroy_partial_swapchain(
	device: &vk::Device,
	views: &mut Vec<vk::ImageView>,
	frames: &mut Vec<PresentationFrameSync>,
	handle: vk::SwapchainKHR,
) {
	for frame in frames.drain(..) {
		unsafe {
			let _ = device.destroy_fence(frame.in_flight, None);
			let _ = device.destroy_semaphore(frame.image_available, None);
		}
	}
	for view in views.drain(..) {
		let _ = unsafe { device.destroy_image_view(view, None) };
	}
	let _ = unsafe { device.destroy_swapchain_khr(handle, std::ptr::null()) };
}

impl Drop for PresentationSwapchain {
	fn drop(&mut self) {
		debug_assert!(
			!self.has_submitted_frames(),
			"submitted presentation work must be retired by close_swapchain before destruction"
		);
		for semaphore in self.present_finished.drain(..) {
			// SAFETY: explicit close established WSI retirement before destruction.
			let _ = unsafe { self._device.raw().destroy_semaphore(semaphore, None) };
		}
		for frame in self.frames.drain(..) {
			unsafe {
				let _ = self._device.raw().destroy_fence(frame.in_flight, None);
				let _ = self
					._device
					.raw()
					.destroy_semaphore(frame.image_available, None);
			}
		}
		for view in self.views.drain(..) {
			let _ = unsafe { self._device.raw().destroy_image_view(view, None) };
		}
		let _ = unsafe {
			self
				._device
				.raw()
				.destroy_swapchain_khr(self.handle, std::ptr::null())
		};
	}
}
