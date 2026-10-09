use std::sync::Arc;

use vk_mem::Alloc;

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
	pub(crate) color_image: ash::vk::Image,
	pub(crate) color_view: ash::vk::ImageView,
	color_allocation: Option<vk_mem::Allocation>,
	pub(crate) depth_image: ash::vk::Image,
	pub(crate) depth_view: ash::vk::ImageView,
	depth_allocation: Option<vk_mem::Allocation>,
	/// Host-visible readback buffer: `width * height * 4` bytes.
	readback: ash::vk::Buffer,
	readback_allocation: Option<vk_mem::Allocation>,
	pub(crate) width: u32,
	pub(crate) height: u32,
	/// Tracks the current Vulkan image layout so barriers can derive the correct
	/// `old_layout` on the next frame without assuming UNDEFINED every time.
	pub(crate) color_layout: ash::vk::ImageLayout,
}

impl RenderTarget {
	pub(in crate::runtime) fn new(device: &Arc<Device>, width: u32, height: u32) -> Result<Self> {
		// Color: R8G8B8A8_UNORM — color attachment, transfer src, sampled
		let color_usage = ash::vk::ImageUsageFlags::COLOR_ATTACHMENT
			| ash::vk::ImageUsageFlags::TRANSFER_SRC
			| ash::vk::ImageUsageFlags::SAMPLED;
		let (color_image, mut color_alloc) =
			alloc_image(device, width, height, RENDER_COLOR_FORMAT, color_usage)?;
		let color_view = match create_view(
			device.raw(),
			color_image,
			RENDER_COLOR_FORMAT,
			ash::vk::ImageAspectFlags::COLOR,
		) {
			Ok(v) => v,
			Err(e) => {
				unsafe {
					device
						.allocator()
						.destroy_image(color_image, &mut color_alloc)
				};
				return Err(e);
			}
		};

		// Depth: D32_SFLOAT — depth-stencil attachment + transfer src for readback
		let depth_usage =
			ash::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | ash::vk::ImageUsageFlags::TRANSFER_SRC;
		let (depth_image, mut depth_alloc) =
			match alloc_image(device, width, height, RENDER_DEPTH_FORMAT, depth_usage) {
				Ok(pair) => pair,
				Err(e) => {
					unsafe {
						device.raw().destroy_image_view(color_view, None);
						device
							.allocator()
							.destroy_image(color_image, &mut color_alloc);
					}
					return Err(e);
				}
			};
		let depth_view = match create_view(
			device.raw(),
			depth_image,
			RENDER_DEPTH_FORMAT,
			ash::vk::ImageAspectFlags::DEPTH,
		) {
			Ok(v) => v,
			Err(e) => {
				unsafe {
					device
						.allocator()
						.destroy_image(depth_image, &mut depth_alloc);
					device.raw().destroy_image_view(color_view, None);
					device
						.allocator()
						.destroy_image(color_image, &mut color_alloc);
				}
				return Err(e);
			}
		};

		// Readback buffer: host-visible, width * height * 4 bytes
		let readback_bytes = (width as u64)
			.checked_mul(height as u64)
			.and_then(|pixels| pixels.checked_mul(4))
			.ok_or_else(|| Error::out_of_range("render target readback size overflows u64"))?;
		let buf_info = ash::vk::BufferCreateInfo::default()
			.size(readback_bytes)
			.usage(ash::vk::BufferUsageFlags::TRANSFER_DST)
			.sharing_mode(ash::vk::SharingMode::EXCLUSIVE);
		let alloc_info = vk_mem::AllocationCreateInfo {
			usage: vk_mem::MemoryUsage::AutoPreferHost,
			flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
			required_flags: ash::vk::MemoryPropertyFlags::HOST_VISIBLE,
			..Default::default()
		};
		let (readback, readback_alloc) =
			match unsafe { device.allocator().create_buffer(&buf_info, &alloc_info) } {
				Ok(pair) => pair,
				Err(source) => {
					unsafe {
						device.raw().destroy_image_view(depth_view, None);
						device
							.allocator()
							.destroy_image(depth_image, &mut depth_alloc);
						device.raw().destroy_image_view(color_view, None);
						device
							.allocator()
							.destroy_image(color_image, &mut color_alloc);
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
			color_layout: ash::vk::ImageLayout::UNDEFINED,
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
		let mut alloc_ref = *alloc;
		let ptr = unsafe { self.device.allocator().map_memory(&mut alloc_ref) }
			.map_err(|s| Error::backend_failure("Vulkan", "render target readback map", s))?;
		let byte_count = (self.width as usize) * (self.height as usize) * 4;
		let result = unsafe { std::slice::from_raw_parts(ptr, byte_count) }.to_vec();
		unsafe { self.device.allocator().unmap_memory(&mut alloc_ref) };
		Ok(result)
	}

	pub(crate) const fn readback_buffer(&self) -> ash::vk::Buffer {
		self.readback
	}
}

impl Drop for RenderTarget {
	fn drop(&mut self) {
		unsafe {
			self.device.raw().destroy_image_view(self.depth_view, None);
			if let Some(mut a) = self.depth_allocation.take() {
				self
					.device
					.allocator()
					.destroy_image(self.depth_image, &mut a);
			}
			self.device.raw().destroy_image_view(self.color_view, None);
			if let Some(mut a) = self.color_allocation.take() {
				self
					.device
					.allocator()
					.destroy_image(self.color_image, &mut a);
			}
			if let Some(mut a) = self.readback_allocation.take() {
				self
					.device
					.allocator()
					.destroy_buffer(self.readback, &mut a);
			}
		}
	}
}

fn alloc_image(
	device: &Arc<Device>,
	width: u32,
	height: u32,
	format: ash::vk::Format,
	usage: ash::vk::ImageUsageFlags,
) -> Result<(ash::vk::Image, vk_mem::Allocation)> {
	let info = ash::vk::ImageCreateInfo::default()
		.image_type(ash::vk::ImageType::TYPE_2D)
		.format(format)
		.extent(ash::vk::Extent3D {
			width,
			height,
			depth: 1,
		})
		.mip_levels(1)
		.array_layers(1)
		.samples(ash::vk::SampleCountFlags::TYPE_1)
		.tiling(ash::vk::ImageTiling::OPTIMAL)
		.usage(usage)
		.sharing_mode(ash::vk::SharingMode::EXCLUSIVE)
		.initial_layout(ash::vk::ImageLayout::UNDEFINED);
	let alloc_info = vk_mem::AllocationCreateInfo {
		usage: vk_mem::MemoryUsage::AutoPreferDevice,
		required_flags: ash::vk::MemoryPropertyFlags::DEVICE_LOCAL,
		..Default::default()
	};
	unsafe { device.allocator().create_image(&info, &alloc_info) }
		.map_err(|s| Error::backend_failure("Vulkan", "render target image allocation", s))
}

fn create_view(
	device: &ash::Device,
	image: ash::vk::Image,
	format: ash::vk::Format,
	aspect: ash::vk::ImageAspectFlags,
) -> Result<ash::vk::ImageView> {
	let range = ash::vk::ImageSubresourceRange::default()
		.aspect_mask(aspect)
		.level_count(1)
		.layer_count(1);
	let info = ash::vk::ImageViewCreateInfo::default()
		.image(image)
		.view_type(ash::vk::ImageViewType::TYPE_2D)
		.format(format)
		.subresource_range(range);
	unsafe { device.create_image_view(&info, None) }
		.map_err(|s| Error::backend_failure("Vulkan", "render target image-view creation", s))
}

pub(crate) struct NativeRgbaImage {
	device: Arc<Device>,
	pub(in crate::runtime) handle: ash::vk::Image,
	pub(in crate::runtime) view: ash::vk::ImageView,
	allocation: Option<vk_mem::Allocation>,
	pub(in crate::runtime) width: u32,
	pub(in crate::runtime) height: u32,
}

impl NativeRgbaImage {
	pub(in crate::runtime) fn new(device: &Arc<Device>, width: u32, height: u32) -> Result<Self> {
		let info = ash::vk::ImageCreateInfo::default()
			.image_type(ash::vk::ImageType::TYPE_2D)
			.format(ash::vk::Format::R8G8B8A8_UNORM)
			.extent(ash::vk::Extent3D {
				width,
				height,
				depth: 1,
			})
			.mip_levels(1)
			.array_layers(1)
			.samples(ash::vk::SampleCountFlags::TYPE_1)
			.tiling(ash::vk::ImageTiling::OPTIMAL)
			.usage(
				ash::vk::ImageUsageFlags::TRANSFER_DST
					| ash::vk::ImageUsageFlags::TRANSFER_SRC
					| ash::vk::ImageUsageFlags::SAMPLED,
			)
			.sharing_mode(ash::vk::SharingMode::EXCLUSIVE)
			.initial_layout(ash::vk::ImageLayout::UNDEFINED);
		let allocation_info = vk_mem::AllocationCreateInfo {
			usage: vk_mem::MemoryUsage::AutoPreferDevice,
			required_flags: ash::vk::MemoryPropertyFlags::DEVICE_LOCAL,
			..Default::default()
		};
		let (handle, allocation) = unsafe { device.allocator().create_image(&info, &allocation_info) }
			.map_err(|source| {
				Error::backend_failure("Vulkan", "RGBA texture-image allocation", source)
			})?;
		let range = ash::vk::ImageSubresourceRange::default()
			.aspect_mask(ash::vk::ImageAspectFlags::COLOR)
			.level_count(1)
			.layer_count(1);
		let view_info = ash::vk::ImageViewCreateInfo::default()
			.image(handle)
			.view_type(ash::vk::ImageViewType::TYPE_2D)
			.format(ash::vk::Format::R8G8B8A8_UNORM)
			.subresource_range(range);
		let view = match unsafe { device.raw().create_image_view(&view_info, None) } {
			Ok(view) => view,
			Err(source) => {
				let mut allocation = allocation;
				unsafe { device.allocator().destroy_image(handle, &mut allocation) };
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

	pub(crate) const fn view(&self) -> ash::vk::ImageView {
		self.view
	}
}

impl Drop for NativeRgbaImage {
	fn drop(&mut self) {
		unsafe { self.device.raw().destroy_image_view(self.view, None) };
		if let Some(mut allocation) = self.allocation.take() {
			unsafe {
				self
					.device
					.allocator()
					.destroy_image(self.handle, &mut allocation)
			};
		}
	}
}
