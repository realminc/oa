//! SDL3-created Vulkan surface retained by the public Presenter session.

use super::device::NativeFlagBits;
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
	handle: vk::SurfaceKHR,
	graphics_present_supported: bool,
	prefer_transparent: bool,
	/// The queue-family index that can present to this surface.
	/// May differ from the graphics queue family on some devices.
	present_queue_family: Option<u32>,
	_window: PhantomData<&'window SdlWindow>,
}

impl<'window> PresentationSurface<'window> {
	pub(in crate::runtime) fn validate_ui_engine(&self, engine: &super::EngineHandle) -> Result<()> {
		engine.validate_ui_presentation(&self._device)
	}

	pub(in crate::runtime) fn create(
		device: &Arc<Device>,
		window: &'window SdlWindow,
	) -> Result<Self> {
		let instance = device.instance();
		// SAFETY: the SDL window and Vulkan instance are retained by the two
		// lifetimes carried by this object; SDL received the required extensions.
		// SDL's surface factory takes the raw VkInstance handle.
		let handle = unsafe {
			window
				.window
				.vulkan_create_surface(instance.raw().handle().0.cast())
		}
		.map_err(|source| Error::backend_failure("SDL3", "Vulkan surface creation", source))?;
		let handle = vk::SurfaceKHR(handle as u64);

		// Determine whether the graphics queue family can present to this surface.
		// If it cannot, search all queue families for one that can.
		let physical = device.physical();
		let graphics_family = physical.graphics_queue_family;
		let graphics_present_supported = graphics_family
			.map(|family| surface_support(instance.raw(), physical.handle, family, handle))
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
			}
			.map_err(|source| Error::backend_failure("Vulkan", "queue-family property query", source))?;
			let mut found = None;
			for (index, _props) in queue_families.iter().enumerate() {
				let supported =
					surface_support(instance.raw(), physical.handle, index as u32, handle).unwrap_or(false);
				if supported {
					found = Some(index as u32);
					break;
				}
			}
			found
		};

		Ok(Self {
			_device: device.clone(),
			handle,
			graphics_present_supported,
			prefer_transparent: window.window.window_flags().0
				& sdl3::sys::video::SDL_WINDOW_TRANSPARENT.0
				!= 0,
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
				.contains(vk::ImageUsageFlagBits::COLOR_ATTACHMENT_BIT),
			supports_transfer_dst: raw
				.supported_usage_flags
				.contains(vk::ImageUsageFlagBits::TRANSFER_DST_BIT),
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
			vk::Extent2D::default()
				.width(raw.current_extent.width)
				.height(raw.current_extent.height)
		} else {
			if drawable_extent.contains(&0) {
				return Ok(None);
			}
			vk::Extent2D::default()
				.width(drawable_extent[0].clamp(raw.min_image_extent.width, raw.max_image_extent.width))
				.height(drawable_extent[1].clamp(raw.min_image_extent.height, raw.max_image_extent.height))
		};
		// Donor: oa/runtime/engine/presenter.cpp::recreateSwapchain.
		// Texture/UI pixels are already sRGB-encoded RGBA8. A blit into an
		// *_SRGB image would apply the transfer function again (128 → ~188).
		// UNORM preserves encoded values; SRGB_NONLINEAR tells WSI how to
		// display them. Storage format and presentation color space differ.
		let format = formats
			.iter()
			.copied()
			.map(|mut format| {
				// The legacy single-UNDEFINED offer allows application selection.
				if formats.len() == 1 && format.format == vk::Format::UNDEFINED {
					format.format = vk::Format::B8G8R8A8_UNORM;
				}
				format
			})
			.find(|format| {
				matches!(
					format.format,
					vk::Format::B8G8R8A8_UNORM | vk::Format::R8G8B8A8_UNORM
				)
					&& format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR_KHR
			})
			.ok_or_else(|| {
				Error::missing_capability(
					"encoded RGBA8 presentation requires an RGBA8/BGRA8 UNORM surface with sRGB nonlinear color space; other formats need an explicit color-conversion path",
				)
			})?;
		if !modes.contains(&vk::PresentModeKHR::FIFO_KHR) {
			return Err(Error::missing_capability(
				"surface does not expose mandatory FIFO presentation",
			));
		}
		if !raw.supported_usage_flags.contains(
			vk::ImageUsageFlagBits::COLOR_ATTACHMENT_BIT | vk::ImageUsageFlagBits::TRANSFER_DST_BIT,
		) {
			return Err(Error::missing_capability(
				"surface swapchain images do not support color-attachment and transfer-destination usage",
			));
		}
		let source_features = vk::FormatFeatureFlagBits::BLIT_SRC_BIT
			| vk::FormatFeatureFlagBits::SAMPLED_IMAGE_FILTER_LINEAR_BIT;
		if !self
			._device
			.optimal_format_supports(vk::Format::R8G8B8A8_UNORM, source_features)
		{
			return Err(Error::missing_capability(
				"R8G8B8A8_UNORM images do not support linear-filtered blit sources",
			));
		}
		if !self
			._device
			.optimal_format_supports(format.format, vk::FormatFeatureFlagBits::BLIT_DST_BIT)
		{
			return Err(Error::missing_capability(format!(
				"selected swapchain format {:?} does not support blit destinations",
				format.format
			)));
		}
		// Donor Presenter: request premultiplied alpha only for a transparent
		// SDL client frame; ordinary windows keep the opaque-first policy.
		let preferred = if self.prefer_transparent {
			vk::CompositeAlphaFlagBitsKHR::PRE_MULTIPLIED_BIT_KHR
		} else {
			vk::CompositeAlphaFlagBitsKHR::OPAQUE_BIT_KHR
		};
		let composite_alpha = [
			preferred,
			vk::CompositeAlphaFlagBitsKHR::OPAQUE_BIT_KHR,
			vk::CompositeAlphaFlagBitsKHR::PRE_MULTIPLIED_BIT_KHR,
			vk::CompositeAlphaFlagBitsKHR::POST_MULTIPLIED_BIT_KHR,
			vk::CompositeAlphaFlagBitsKHR::INHERIT_BIT_KHR,
		]
		.into_iter()
		.find(|candidate| (raw.supported_composite_alpha & *candidate) == *candidate)
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
			present_mode: vk::PresentModeKHR::FIFO_KHR,
			composite_alpha,
			transparent: self.prefer_transparent
				&& composite_alpha == vk::CompositeAlphaFlagBitsKHR::PRE_MULTIPLIED_BIT_KHR,
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
		vk::SurfaceCapabilitiesKHR,
		Vec<vk::SurfaceFormatKHR>,
		Vec<vk::PresentModeKHR>,
	)> {
		let physical = self._device.physical();
		let instance = self._device.instance().raw();
		let mut raw = vk::SurfaceCapabilitiesKHR::default();
		// SAFETY: the retained instance owns both device and surface.
		match unsafe {
			instance.get_physical_device_surface_capabilities_khr(physical.handle, self.handle, &mut raw)
		} {
			Ok(vk::Result::SUCCESS) => {}
			other => {
				return Err(Error::backend_dispatch_failure(
					"Vulkan",
					"surface-capability query",
					other,
				));
			}
		}
		let formats = surface_formats(instance, physical.handle, self.handle)
			.map_err(|source| Error::backend_failure("Vulkan", "surface-format query", source))?;
		let modes = surface_present_modes(instance, physical.handle, self.handle)
			.map_err(|source| Error::backend_failure("Vulkan", "surface present-mode query", source))?;
		Ok((raw, formats, modes))
	}
}

impl Drop for PresentationSurface<'_> {
	fn drop(&mut self) {
		// SAFETY: this object uniquely owns the SDL-created surface and retains its instance.
		unsafe {
			let _ = self
				._device
				.instance()
				.raw()
				.destroy_surface_khr(self.handle, std::ptr::null());
		}
	}
}

fn surface_support(
	instance: &vk::Instance,
	physical: vk::PhysicalDevice,
	family: u32,
	surface: vk::SurfaceKHR,
) -> Result<bool> {
	let mut supported = 0_u32;
	// SAFETY: `supported` is a writable Bool32 out-param.
	unsafe {
		instance.get_physical_device_surface_support_khr(physical, family, surface, &mut supported)
	}
	.map(|_| supported != vk::FALSE)
	.map_err(|source| Error::backend_failure("Vulkan", "surface present-support query", source))
}

fn surface_formats(
	instance: &vk::Instance,
	physical: vk::PhysicalDevice,
	surface: vk::SurfaceKHR,
) -> Result<Vec<vk::SurfaceFormatKHR>> {
	let mut count = 0_u32;
	// SAFETY: null output pointer requests only the element count.
	match unsafe {
		instance.get_physical_device_surface_formats_khr(
			physical,
			surface,
			&mut count,
			std::ptr::null_mut(),
		)
	} {
		Ok(vk::Result::SUCCESS) => {}
		other => {
			return Err(Error::backend_dispatch_failure(
				"Vulkan",
				"surface-format count",
				other,
			));
		}
	}
	let mut formats = vec![vk::SurfaceFormatKHR::default(); count as usize];
	// SAFETY: `formats` provides `count` writable records.
	match unsafe {
		instance.get_physical_device_surface_formats_khr(
			physical,
			surface,
			&mut count,
			formats.as_mut_ptr(),
		)
	} {
		Ok(vk::Result::SUCCESS) => {
			formats.truncate(count as usize);
			Ok(formats)
		}
		other => Err(Error::backend_dispatch_failure(
			"Vulkan",
			"surface-format query",
			other,
		)),
	}
}

fn surface_present_modes(
	instance: &vk::Instance,
	physical: vk::PhysicalDevice,
	surface: vk::SurfaceKHR,
) -> Result<Vec<vk::PresentModeKHR>> {
	let mut count = 0_u32;
	// SAFETY: null output pointer requests only the element count.
	match unsafe {
		instance.get_physical_device_surface_present_modes_khr(
			physical,
			surface,
			&mut count,
			std::ptr::null_mut(),
		)
	} {
		Ok(vk::Result::SUCCESS) => {}
		other => {
			return Err(Error::backend_dispatch_failure(
				"Vulkan",
				"present-mode count",
				other,
			));
		}
	}
	let mut modes = vec![vk::PresentModeKHR::default(); count as usize];
	// SAFETY: `modes` provides `count` writable records.
	match unsafe {
		instance.get_physical_device_surface_present_modes_khr(
			physical,
			surface,
			&mut count,
			modes.as_mut_ptr(),
		)
	} {
		Ok(vk::Result::SUCCESS) => {
			modes.truncate(count as usize);
			Ok(modes)
		}
		other => Err(Error::backend_dispatch_failure(
			"Vulkan",
			"present-mode query",
			other,
		)),
	}
}
