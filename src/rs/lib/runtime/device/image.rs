use std::sync::Arc;

use crate::{Error, Result};

use super::pipeline::{RENDER_COLOR_FORMAT, RENDER_DEPTH_FORMAT};
use crate::runtime::Device;

/// One render target slot: device-local color (R8G8B8A8_UNORM + SAMPLED) and
/// depth (D32_SFLOAT) images with their views, plus a host-visible readback
/// buffer for explicit color readback.
///
/// Donor: oacpp/source/cpp/lib/oa/render/renderer3d.cpp `RenderTarget` /
/// `createTarget`, single-sample (no MSAA) path only for this first slice.
pub(crate) struct RenderTarget {
	device: Arc<Device>,
	pub(crate) color_image: vk::Image,
	pub(crate) color_view: vk::ImageView,
	color_allocation: Option<vk::mem::Allocation>,
	pub(crate) depth_image: vk::Image,
	pub(crate) depth_view: vk::ImageView,
	depth_allocation: Option<vk::mem::Allocation>,
	/// Host-visible readback buffer: `width * height * 4` bytes.
	readback: vk::Buffer,
	readback_allocation: Option<vk::mem::Allocation>,
	pub(crate) width: u32,
	pub(crate) height: u32,
	/// Tracks the current Vulkan image layout so barriers can derive the correct
	/// `old_layout` on the next frame without assuming UNDEFINED every time.
	pub(crate) color_layout: vk::ImageLayout,
}

impl RenderTarget {
	pub(in crate::runtime) fn new(device: &Arc<Device>, width: u32, height: u32) -> Result<Self> {
		// Color: R8G8B8A8_UNORM — color attachment, transfer src, sampled
		let color_usage = vk::ImageUsageFlagBits::COLOR_ATTACHMENT_BIT
			| vk::ImageUsageFlagBits::TRANSFER_SRC_BIT
			| vk::ImageUsageFlagBits::SAMPLED_BIT;
		let (color_image, mut color_alloc) =
			alloc_image(device, width, height, RENDER_COLOR_FORMAT, color_usage)?;
		let color_view = match create_view(
			device.raw(),
			color_image,
			RENDER_COLOR_FORMAT,
			vk::ImageAspectFlagBits::COLOR_BIT,
		) {
			Ok(v) => v,
			Err(e) => {
				unsafe {
					device
						.allocator()
						.with_allocator(|a| a.destroy_image(color_image, &mut color_alloc))
				}??;
				return Err(e);
			}
		};

		// Depth: D32_SFLOAT — depth-stencil attachment + transfer src for readback
		let depth_usage = vk::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT_BIT
			| vk::ImageUsageFlagBits::TRANSFER_SRC_BIT;
		let (depth_image, mut depth_alloc) =
			match alloc_image(device, width, height, RENDER_DEPTH_FORMAT, depth_usage) {
				Ok(pair) => pair,
				Err(e) => {
					unsafe {
						device.raw().destroy_image_view(color_view, None)?;
						device
							.allocator()
							.with_allocator(|a| a.destroy_image(color_image, &mut color_alloc))??;
					}
					return Err(e);
				}
			};
		let depth_view = match create_view(
			device.raw(),
			depth_image,
			RENDER_DEPTH_FORMAT,
			vk::ImageAspectFlagBits::DEPTH_BIT,
		) {
			Ok(v) => v,
			Err(e) => {
				unsafe {
					device
						.allocator()
						.with_allocator(|a| a.destroy_image(depth_image, &mut depth_alloc))??;
					device.raw().destroy_image_view(color_view, None)?;
					device
						.allocator()
						.with_allocator(|a| a.destroy_image(color_image, &mut color_alloc))??;
				}
				return Err(e);
			}
		};

		// Readback buffer: host-visible, width * height * 4 bytes
		let readback_bytes = (width as u64)
			.checked_mul(height as u64)
			.and_then(|pixels| pixels.checked_mul(4))
			.ok_or_else(|| Error::out_of_range("render target readback size overflows u64"))?;
		let buf_info = vk::BufferCreateInfo::new()
			.set_size(readback_bytes)
			.set_usage(vk::BufferUsageFlagBits::TRANSFER_DST_BIT)
			.set_sharing_mode(vk::SharingMode::EXCLUSIVE);
		let alloc_info = vk::mem::AllocationCreateInfo::new()
			.set_usage(vk::mem::MemoryUsage::PreferHost)
			.set_host_access(vk::mem::HostAccess::Random)
			.set_required_flags(vk::MemoryPropertyFlagBits::HOST_VISIBLE_BIT);
		let (readback, readback_alloc) = match unsafe {
			device
				.allocator()
				.with_allocator(|a| a.create_buffer_with_options(&buf_info, &alloc_info))
		} {
			Ok(Ok(pair)) => pair,
			Ok(Err(source)) | Err(source) => {
				unsafe {
					device.raw().destroy_image_view(depth_view, None)?;
					device
						.allocator()
						.with_allocator(|a| a.destroy_image(depth_image, &mut depth_alloc))??;
					device.raw().destroy_image_view(color_view, None)?;
					device
						.allocator()
						.with_allocator(|a| a.destroy_image(color_image, &mut color_alloc))??;
				}
				return Err(Error::backend_failure(
					"Vulkan",
					"render target readback buffer allocation",
					source,
				));
			}
		};

		Ok(Self {
			device: device.clone(),
			color_image,
			color_view,
			color_allocation: Some(color_alloc),
			depth_image,
			depth_view,
			depth_allocation: Some(depth_alloc),
			readback,
			readback_allocation: Some(readback_alloc),
			width,
			height,
			color_layout: vk::ImageLayout::UNDEFINED,
		})
	}

	/// Read back the color buffer into a host `Vec<u8>` after the producer event
	/// has completed. Flushes and maps the readback buffer for a single frame.
	///
	/// # Safety
	/// The caller must ensure the readback buffer is not in active GPU use.
	pub(crate) fn read_color_rgba8(&self) -> Result<Vec<u8>> {
		let alloc = self
			.readback_allocation
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("render target readback allocation is gone"))?;
		// SAFETY: the caller contract makes this readback buffer idle; the
		// allocation is live and uniquely owned by this render target.
		let ptr = unsafe { self.device.allocator().with_allocator(|a| a.map(alloc)) }
			.map_err(|s| Error::backend_failure("Vulkan", "render target readback map", s))?;
		let byte_count = (self.width as usize) * (self.height as usize) * 4;
		let result =
			unsafe { std::slice::from_raw_parts(ptr.map_err(Error::from)?.cast::<u8>(), byte_count) }
				.to_vec();
		// SAFETY: balances the map above; the copied Vec owns all read bytes.
		unsafe {
			self
				.device
				.allocator()
				.with_allocator(|a| a.unmap(alloc))??;
		}
		Ok(result)
	}

	pub(crate) const fn readback_buffer(&self) -> vk::Buffer {
		self.readback
	}
}

impl Drop for RenderTarget {
	fn drop(&mut self) {
		unsafe {
			let _ = self.device.raw().destroy_image_view(self.depth_view, None);
			if let Some(mut alloc) = self.depth_allocation.take() {
				let _ = self
					.device
					.allocator()
					.with_allocator(|a| a.destroy_image(self.depth_image, &mut alloc));
			}
			let _ = self.device.raw().destroy_image_view(self.color_view, None);
			if let Some(mut alloc) = self.color_allocation.take() {
				let _ = self
					.device
					.allocator()
					.with_allocator(|a| a.destroy_image(self.color_image, &mut alloc));
			}
			if let Some(mut alloc) = self.readback_allocation.take() {
				let _ = self
					.device
					.allocator()
					.with_allocator(|a| a.destroy_buffer(self.readback, &mut alloc));
			}
		}
	}
}

fn alloc_image(
	device: &Arc<Device>,
	width: u32,
	height: u32,
	format: vk::Format,
	usage: vk::ImageUsageFlags,
) -> Result<(vk::Image, vk::mem::Allocation)> {
	let info = vk::ImageCreateInfo::new()
		.set_image_type(vk::ImageType::_2D)
		.set_format(format)
		.set_extent(vk::Extent3D {
			width,
			height,
			depth: 1,
		})
		.set_mip_levels(1)
		.set_array_layers(1)
		.set_samples(vk::SampleCountFlagBits::_1_BIT)
		.set_tiling(vk::ImageTiling::OPTIMAL)
		.set_usage(usage)
		.set_sharing_mode(vk::SharingMode::EXCLUSIVE)
		.set_initial_layout(vk::ImageLayout::UNDEFINED);
	let alloc_info = vk::mem::AllocationCreateInfo::new()
		.set_usage(vk::mem::MemoryUsage::PreferDevice)
		.set_required_flags(vk::MemoryPropertyFlagBits::DEVICE_LOCAL_BIT);
	unsafe {
		device
			.allocator()
			.with_allocator(|a| a.create_image_with_options(&info, &alloc_info))
	}
	.map_err(|s| Error::backend_failure("Vulkan", "render target image allocation", s))?
	.map_err(|s| Error::backend_failure("Vulkan", "render target image allocation", s))
}

fn create_view(
	device: &vk::Device,
	image: vk::Image,
	format: vk::Format,
	aspect: vk::ImageAspectFlags,
) -> Result<vk::ImageView> {
	let range = vk::ImageSubresourceRange::new()
		.set_aspect_mask(aspect)
		.set_level_count(1)
		.set_layer_count(1);
	let info = vk::ImageViewCreateInfo::new()
		.set_image(image)
		.set_view_type(vk::ImageViewType::_2D)
		.set_format(format)
		.set_subresource_range(range);
	unsafe { device.create_image_view(&info, None) }
		.map_err(|s| Error::backend_failure("Vulkan", "render target image-view creation", s))
}

pub(crate) struct NativeRgbaImage {
	device: Arc<Device>,
	pub(in crate::runtime) handle: vk::Image,
	pub(in crate::runtime) view: vk::ImageView,
	allocation: Option<vk::mem::Allocation>,
	pub(in crate::runtime) width: u32,
	pub(in crate::runtime) height: u32,
}

impl NativeRgbaImage {
	pub(in crate::runtime) fn new(device: &Arc<Device>, width: u32, height: u32) -> Result<Self> {
		let info = vk::ImageCreateInfo::new()
			.set_image_type(vk::ImageType::_2D)
			.set_format(vk::Format::R8G8B8A8_UNORM)
			.set_extent(vk::Extent3D {
				width,
				height,
				depth: 1,
			})
			.set_mip_levels(1)
			.set_array_layers(1)
			.set_samples(vk::SampleCountFlagBits::_1_BIT)
			.set_tiling(vk::ImageTiling::OPTIMAL)
			.set_usage(
				vk::ImageUsageFlagBits::TRANSFER_DST_BIT
					| vk::ImageUsageFlagBits::TRANSFER_SRC_BIT
					| vk::ImageUsageFlagBits::SAMPLED_BIT,
			)
			.set_sharing_mode(vk::SharingMode::EXCLUSIVE)
			.set_initial_layout(vk::ImageLayout::UNDEFINED);
		let allocation_info = vk::mem::AllocationCreateInfo::new()
			.set_usage(vk::mem::MemoryUsage::PreferDevice)
			.set_required_flags(vk::MemoryPropertyFlagBits::DEVICE_LOCAL_BIT);
		let (handle, allocation) = unsafe {
			device
				.allocator()
				.with_allocator(|a| a.create_image_with_options(&info, &allocation_info))
		}
		.map_err(|source| Error::backend_failure("Vulkan", "RGBA texture-image allocation", source))?
		.map_err(|source| Error::backend_failure("Vulkan", "RGBA texture-image allocation", source))?;
		let range = vk::ImageSubresourceRange::new()
			.set_aspect_mask(vk::ImageAspectFlagBits::COLOR_BIT)
			.set_level_count(1)
			.set_layer_count(1);
		let view_info = vk::ImageViewCreateInfo::new()
			.set_image(handle)
			.set_view_type(vk::ImageViewType::_2D)
			.set_format(vk::Format::R8G8B8A8_UNORM)
			.set_subresource_range(range);
		let view = match unsafe { device.raw().create_image_view(&view_info, None) } {
			Ok(view) => view,
			Err(source) => {
				let mut allocation = allocation;
				unsafe {
					device
						.allocator()
						.with_allocator(|a| a.destroy_image(handle, &mut allocation))
				}??;
				return Err(Error::backend_failure(
					"Vulkan",
					"RGBA texture-image view creation",
					source,
				));
			}
		};
		Ok(Self {
			device: device.clone(),
			handle,
			view,
			allocation: Some(allocation),
			width,
			height,
		})
	}

	pub(crate) const fn view(&self) -> vk::ImageView {
		self.view
	}
}

impl Drop for NativeRgbaImage {
	fn drop(&mut self) {
		let _ = unsafe { self.device.raw().destroy_image_view(self.view, None) };
		if let Some(mut allocation) = self.allocation.take() {
			let _ = unsafe {
				self
					.device
					.allocator()
					.with_allocator(|a| a.destroy_image(self.handle, &mut allocation))
			};
		}
	}
}
