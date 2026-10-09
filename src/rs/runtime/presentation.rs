//! SDL3-created Vulkan surface retained by the public Presenter session.

use std::sync::Arc;

use std::marker::PhantomData;

use crate::{Error, Result, runtime::SdlWindow};

use super::{
	Device,
	device::swapchain::{PresentationSwapchain, SwapchainConfig},
};

/// Backend-neutral surface capabilities used to validate a future swapchain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentationCapabilities {
	pub min_image_count: u32,
	pub max_image_count: Option<u32>,
	pub current_extent: Option<[u32; 2]>,
	pub min_extent: [u32; 2],
	pub max_extent: [u32; 2],
	pub format_count: usize,
	pub present_mode_count: usize,
	pub supports_color_attachment: bool,
	pub supports_transfer_dst: bool,
}

pub struct PresentationSurface<'window> {
	_device: Arc<Device>,
	loader: ash::khr::surface::Instance,
	handle: ash::vk::SurfaceKHR,
	graphics_present_supported: bool,
	/// The queue-family index that can present to this surface.
	/// May differ from the graphics queue family on some devices.
	present_queue_family: Option<u32>,
	_window: PhantomData<&'window SdlWindow>,
}

impl<'window> PresentationSurface<'window> {
	pub(in crate::runtime) fn create(
		device: &Arc<Device>,
		window: &'window SdlWindow,
	) -> Result<Self> {
		let instance = device.instance();
		let loader = ash::khr::surface::Instance::new(instance.entry(), instance.raw());
		// SAFETY: the SDL window and Vulkan instance are retained by the two
		// lifetimes carried by this object; SDL received the required extensions.
		let handle = unsafe { window.window.vulkan_create_surface(instance.raw().handle()) }
			.map_err(|source| Error::backend_failure("SDL3", "Vulkan surface creation", source))?;

		// Determine whether the graphics queue family can present to this surface.
		// If it cannot, search all queue families for one that can.
		let physical = device.physical();
		let graphics_family = physical.graphics_queue_family;
		let graphics_present_supported = graphics_family
			.map(|family| unsafe {
				loader.get_physical_device_surface_support(physical.handle, family, handle)
			})
			.transpose()
			.map_err(|source| Error::backend_failure("Vulkan", "surface present-support query", source))?
			.unwrap_or(false);

		// Find a dedicated present family when the graphics family cannot present.
		// Enumerate all queue families and pick the first that supports presenting
		// to this surface (including the graphics family itself as a fallback).
		let present_queue_family = if graphics_present_supported {
			graphics_family
		} else {
			let queue_families = unsafe {
				instance
					.raw()
					.get_physical_device_queue_family_properties(physical.handle)
			};
			let mut found = None;
			for (index, _props) in queue_families.iter().enumerate() {
				let supported = unsafe {
					loader.get_physical_device_surface_support(physical.handle, index as u32, handle)
				}
				.unwrap_or(false);
				if supported {
					found = Some(index as u32);
					break;
				}
			}
			found
		};

		Ok(Self {
			_device: device.clone(),
			loader,
			handle,
			graphics_present_supported,
			present_queue_family,
			_window: PhantomData,
		})
	}

	/// Return the queue-family index that can present to this surface, if known.
	pub(in crate::runtime) fn present_queue_family(&self) -> Option<u32> {
		self.present_queue_family
	}

	pub const fn graphics_present_supported(&self) -> bool {
		self.graphics_present_supported
	}

	pub(in crate::runtime) fn has_graphics(&self) -> bool {
		self._device.has_graphics_queue()
	}

	pub(in crate::runtime) fn supports_swapchain(&self) -> bool {
		self._device.supports_swapchain()
	}

	pub fn capabilities(&self) -> Result<PresentationCapabilities> {
		let (raw, formats, modes) = self.raw_capabilities()?;
		Ok(PresentationCapabilities {
			min_image_count: raw.min_image_count,
			max_image_count: (raw.max_image_count != 0).then_some(raw.max_image_count),
			current_extent: (raw.current_extent.width != u32::MAX)
				.then_some([raw.current_extent.width, raw.current_extent.height]),
			min_extent: [raw.min_image_extent.width, raw.min_image_extent.height],
			max_extent: [raw.max_image_extent.width, raw.max_image_extent.height],
			format_count: formats.len(),
			present_mode_count: modes.len(),
			supports_color_attachment: raw
				.supported_usage_flags
				.contains(ash::vk::ImageUsageFlags::COLOR_ATTACHMENT),
			supports_transfer_dst: raw
				.supported_usage_flags
				.contains(ash::vk::ImageUsageFlags::TRANSFER_DST),
		})
	}

	/// Select the valid swapchain extent for one SDL drawable-pixel extent.
	///
	/// `None` means the window is minimized and callers must defer allocation or
	/// acquisition until SDL reports a non-zero drawable extent.
	pub fn swapchain_extent(&self, drawable_extent: [u32; 2]) -> Result<Option<[u32; 2]>> {
		let (raw, _, _) = self.raw_capabilities()?;
		if raw.current_extent.width != u32::MAX {
			return Ok(Some([raw.current_extent.width, raw.current_extent.height]));
		}
		if drawable_extent.contains(&0) {
			return Ok(None);
		}
		Ok(Some([
			drawable_extent[0].clamp(raw.min_image_extent.width, raw.max_image_extent.width),
			drawable_extent[1].clamp(raw.min_image_extent.height, raw.max_image_extent.height),
		]))
	}

	pub(in crate::runtime) fn swapchain_config(
		&self,
		drawable_extent: [u32; 2],
	) -> Result<Option<SwapchainConfig>> {
		let (raw, formats, modes) = self.raw_capabilities()?;
		let extent = if raw.current_extent.width != u32::MAX {
			ash::vk::Extent2D::default()
				.width(raw.current_extent.width)
				.height(raw.current_extent.height)
		} else {
			if drawable_extent.contains(&0) {
				return Ok(None);
			}
			ash::vk::Extent2D::default()
				.width(drawable_extent[0].clamp(raw.min_image_extent.width, raw.max_image_extent.width))
				.height(drawable_extent[1].clamp(raw.min_image_extent.height, raw.max_image_extent.height))
		};
		let format = formats
			.iter()
			.copied()
			.find(|format| {
				format.format == ash::vk::Format::B8G8R8A8_SRGB
					&& format.color_space == ash::vk::ColorSpaceKHR::SRGB_NONLINEAR
			})
			.or_else(|| formats.first().copied())
			.ok_or_else(|| Error::missing_capability("surface exposes no swapchain formats"))?;
		if !modes.contains(&ash::vk::PresentModeKHR::FIFO) {
			return Err(Error::missing_capability(
				"surface does not expose mandatory FIFO presentation",
			));
		}
		if !raw
			.supported_usage_flags
			.contains(ash::vk::ImageUsageFlags::COLOR_ATTACHMENT | ash::vk::ImageUsageFlags::TRANSFER_DST)
		{
			return Err(Error::missing_capability(
				"surface swapchain images do not support color-attachment and transfer-destination usage",
			));
		}
		let source_features = ash::vk::FormatFeatureFlags::BLIT_SRC
			| ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR;
		if !self
			._device
			.optimal_format_supports(ash::vk::Format::R8G8B8A8_UNORM, source_features)
		{
			return Err(Error::missing_capability(
				"R8G8B8A8_UNORM images do not support linear-filtered blit sources",
			));
		}
		if !self
			._device
			.optimal_format_supports(format.format, ash::vk::FormatFeatureFlags::BLIT_DST)
		{
			return Err(Error::missing_capability(format!(
				"selected swapchain format {:?} does not support blit destinations",
				format.format
			)));
		}
		let composite_alpha = [
			ash::vk::CompositeAlphaFlagsKHR::OPAQUE,
			ash::vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
			ash::vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
			ash::vk::CompositeAlphaFlagsKHR::INHERIT,
		]
		.into_iter()
		.find(|candidate| raw.supported_composite_alpha.contains(*candidate))
		.ok_or_else(|| Error::missing_capability("surface exposes no composite-alpha mode"))?;
		let image_count = raw
			.min_image_count
			.saturating_add(1)
			.min(if raw.max_image_count == 0 {
				u32::MAX
			} else {
				raw.max_image_count
			});
		Ok(Some(SwapchainConfig {
			extent,
			image_count,
			format,
			present_mode: ash::vk::PresentModeKHR::FIFO,
			composite_alpha,
			pre_transform: raw.current_transform,
		}))
	}

	pub(in crate::runtime) fn create_swapchain(
		&self,
		drawable_extent: [u32; 2],
	) -> Result<Option<PresentationSwapchain>> {
		// Require a present-capable queue (may differ from the graphics family).
		if self.present_queue_family.is_none() || !self.supports_swapchain() {
			return Err(Error::missing_capability(
				"selected device cannot create a presentation swapchain",
			));
		}
		let Some(config) = self.swapchain_config(drawable_extent)? else {
			return Ok(None);
		};
		PresentationSwapchain::create(
			&self._device,
			self.handle,
			self.present_queue_family,
			config,
		)
		.map(Some)
	}

	fn raw_capabilities(
		&self,
	) -> Result<(
		ash::vk::SurfaceCapabilitiesKHR,
		Vec<ash::vk::SurfaceFormatKHR>,
		Vec<ash::vk::PresentModeKHR>,
	)> {
		let physical = self._device.physical();
		let raw = unsafe {
			self
				.loader
				.get_physical_device_surface_capabilities(physical.handle, self.handle)
		}
		.map_err(|source| Error::backend_failure("Vulkan", "surface-capability query", source))?;
		let formats = unsafe {
			self
				.loader
				.get_physical_device_surface_formats(physical.handle, self.handle)
		}
		.map_err(|source| Error::backend_failure("Vulkan", "surface-format query", source))?;
		let modes = unsafe {
			self
				.loader
				.get_physical_device_surface_present_modes(physical.handle, self.handle)
		}
		.map_err(|source| Error::backend_failure("Vulkan", "surface present-mode query", source))?;
		Ok((raw, formats, modes))
	}
}

impl Drop for PresentationSurface<'_> {
	fn drop(&mut self) {
		// SAFETY: this object uniquely owns the SDL-created surface and retains its instance.
		unsafe { self.loader.destroy_surface(self.handle, None) };
	}
}
