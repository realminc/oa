//! Vulkan image allocation shared by encode and decode sessions.

use std::sync::Arc;

use crate::{
	Error, Result,
	runtime::{Device, device::NativeFlagBits},
	video,
};

pub(super) struct VideoImage {
	pub(super) handle: vk::Image,
	pub(super) view: vk::ImageView,
	allocation: vk::mem::Allocation,
	device: Arc<Device>,
	pub(super) array_layers: u32,
	pub(super) format: vk::Format,
	pub(super) sampled: bool,
}

pub(super) fn create_video_image(
	device: &Arc<Device>,
	profile: &vk::VideoProfileInfoKHR<'_>,
	properties: &vk::VideoFormatPropertiesKHR<'_>,
	extent: video::VideoExtent,
	array_layers: u32,
	usage: vk::ImageUsageFlags,
) -> Result<VideoImage> {
	let profiles = std::slice::from_ref(profile);
	let mut profile_list = vk::VideoProfileListInfoKHR::new()
		.set_profiles(profiles)
		.map_err(Error::from)?;
	let image_info = vk::ImageCreateInfo::new()
		.set_flags(
			properties.image_create_flags & !vk::ImageCreateFlagBits::VIDEO_PROFILE_INDEPENDENT_BIT_KHR,
		)
		.set_image_type(properties.image_type)
		.set_format(properties.format)
		.set_extent(vk::Extent3D {
			width: extent.width,
			height: extent.height,
			depth: 1,
		})
		.set_mip_levels(1)
		.set_array_layers(array_layers)
		.set_samples(vk::SampleCountFlagBits::_1_BIT)
		.set_tiling(properties.image_tiling)
		.set_usage(usage)
		.set_sharing_mode(vk::SharingMode::EXCLUSIVE)
		.set_initial_layout(vk::ImageLayout::UNDEFINED)
		.push_next(&mut profile_list)
		.map_err(Error::from)?;
	let allocation_info = vk::mem::AllocationCreateInfo::new()
		.set_usage(vk::mem::MemoryUsage::PreferDevice)
		.set_required_flags(vk::MemoryPropertyFlagBits::DEVICE_LOCAL_BIT);
	// SAFETY: the exact video profile and queried format properties remain live,
	// the extent/layer count are nonzero and session-bounded, and VMA retains the
	// device/allocator used to bind the returned image allocation.
	let (handle, allocation) = unsafe {
		device
			.allocator()
			.with_allocator(|a| a.create_image_with_options(&(image_info), &allocation_info))
			.map_err(Error::from)?
	}
	.map_err(|source| Error::backend_failure("Vulkan", "video image allocation", source))?;
	let mut image = VideoImage {
		handle,
		view: vk::ImageView::default(),
		allocation,
		device: device.clone(),
		array_layers,
		format: properties.format,
		sampled: usage.contains(vk::ImageUsageFlagBits::SAMPLED_BIT),
	};
	// The decoder's generic view does not sample. Restrict its usage so a
	// sampled multi-planar image does not require a color-specific conversion
	// on this view; immutable YCbCr views are created separately per matrix/range.
	let mut view_usage =
		vk::ImageViewUsageCreateInfo::new().set_usage(usage & !vk::ImageUsageFlagBits::SAMPLED_BIT);
	let view_info = vk::ImageViewCreateInfo::new()
		.set_image(handle)
		.set_view_type(if array_layers == 1 {
			vk::ImageViewType::_2D
		} else {
			vk::ImageViewType::_2D_ARRAY
		})
		.set_format(properties.format)
		.set_components(properties.component_mapping)
		.set_subresource_range(vk::ImageSubresourceRange {
			aspect_mask: vk::ImageAspectFlagBits::COLOR_BIT,
			base_mip_level: 0,
			level_count: 1,
			base_array_layer: 0,
			layer_count: array_layers,
		})
		.push_next(&mut view_usage)
		.map_err(Error::from)?;
	// SAFETY: `handle` is a live 2D image with the queried format, and the view
	// covers its sole mip and complete valid layer range.
	image.view = unsafe { device.raw().create_image_view(&view_info, None) }
		.map_err(|source| Error::backend_failure("Vulkan", "video image view creation", source))?;
	Ok(image)
}

impl Drop for VideoImage {
	fn drop(&mut self) {
		// SAFETY: the image is uniquely owned. Session and frame retention keep
		// it live through retirement and destroy dependent views first.
		unsafe {
			if self.view != vk::ImageView::default() {
				let _ = self.device.raw().destroy_image_view(self.view, None);
			}
			let _ = self
				.device
				.allocator()
				.with_allocator(|allocator| allocator.destroy_image(self.handle, &mut self.allocation));
		}
	}
}
