//! SDL3-backed Vulkan presentation session.

use crate::{Engine, Result, Texture};

use super::{PresentationCapabilities, PresentationSurface, SdlWindow};

/// Engine-borrowing presentation session.
///
/// Construction creates and owns the SDL3 Vulkan surface, then records whether
/// the Engine's retained graphics queue can present to that exact surface.
/// Swapchain acquire/present is intentionally the next lifecycle phase rather
/// than a fabricated method on this real surface session.
pub struct Presenter<'window> {
	swapchain: Option<super::PresentationSwapchain>,
	surface: Option<PresentationSurface<'window>>,
	pending_drawable_extent: Option<[u32; 2]>,
}

impl<'window> Presenter<'window> {
	/// Create a presentation session for one SDL3 window.
	///
	/// The Engine must have been built using `EngineRequirements::presentation()`
	/// and `window.required_instance_extensions()`.
	pub fn new(engine: &Engine, window: &'window SdlWindow) -> Result<Self> {
		Ok(Self {
			swapchain: None,
			surface: Some(engine.create_presentation_surface(window)?),
			pending_drawable_extent: None,
		})
	}

	/// Return whether the Engine retained a graphics-capable queue.
	///
	/// A surface can exist without this capability; swapchain allocation will
	/// require a graphics queue that can also present to this exact surface.
	pub fn has_graphics(&self) -> bool {
		self.surface().has_graphics()
	}

	/// Return whether the selected device enabled `VK_KHR_swapchain`.
	pub fn supports_swapchain(&self) -> bool {
		self.surface().supports_swapchain()
	}

	/// Return whether any queue family can present to this surface.
	///
	/// True when the graphics queue can present directly (same-family path) or
	/// when a distinct present-capable family was found during surface creation.
	pub fn has_present(&self) -> bool {
		self.surface().present_queue_family().is_some()
	}

	/// Query the exact surface capabilities before swapchain allocation.
	pub fn capabilities(&self) -> Result<PresentationCapabilities> {
		self.surface().capabilities()
	}

	/// Select the currently valid drawable extent for a future swapchain.
	pub fn swapchain_extent(&self, window: &SdlWindow) -> Result<Option<[u32; 2]>> {
		self.surface().swapchain_extent(window.drawable_extent())
	}

	/// Validate whether a swapchain can be allocated at the current drawable size.
	///
	/// `false` is the normal minimized-window state; unsupported WSI capability
	/// remains an error rather than being hidden behind a fallback.
	pub fn can_create_swapchain(&self, window: &SdlWindow) -> Result<bool> {
		if !self.has_graphics() || !self.has_present() || !self.supports_swapchain() {
			return Ok(false);
		}
		Ok(
			self
				.surface()
				.swapchain_config(window.drawable_extent())?
				.is_some(),
		)
	}

	/// Allocate the negotiated swapchain for the current drawable extent.
	pub fn init_swapchain(&mut self, window: &SdlWindow) -> Result<bool> {
		self.close_swapchain()?;
		let extent = self
			.pending_drawable_extent
			.take()
			.unwrap_or_else(|| window.drawable_extent());
		self.swapchain = self.surface().create_swapchain(extent)?;
		Ok(self.swapchain.is_some())
	}

	/// Wait for submitted WSI work before retiring or replacing swapchain resources.
	pub fn wait_presentation_idle(&mut self) -> Result<()> {
		if let Some(swapchain) = &mut self.swapchain {
			swapchain.wait_idle()?;
		}
		Ok(())
	}

	/// Explicitly retire the current swapchain after its submitted WSI work completes.
	pub fn close_swapchain(&mut self) -> Result<()> {
		self.wait_presentation_idle()?;
		self.swapchain = None;
		Ok(())
	}

	/// Retire the old swapchain and allocate one for the current drawable extent.
	pub fn recreate_swapchain(&mut self, window: &SdlWindow) -> Result<bool> {
		self.close_swapchain()?;
		self.init_swapchain(window)
	}

	/// Record SDL's pixel-size change for explicit swapchain retirement/recreation.
	///
	/// The caller must recreate only after all presentation work using the old
	/// images has completed; this notification itself never submits or waits.
	pub fn notify_pixel_size_changed(&mut self, width: u32, height: u32) {
		self.pending_drawable_extent = Some([width, height]);
	}

	pub fn needs_swapchain_recreate(&self) -> bool {
		self.pending_drawable_extent.is_some()
	}

	pub fn has_swapchain(&self) -> bool {
		self
			.swapchain
			.as_ref()
			.is_some_and(super::PresentationSwapchain::is_valid)
	}

	/// Return the number of owned swapchain image views.
	pub fn swapchain_image_count(&self) -> usize {
		self
			.swapchain
			.as_ref()
			.map_or(0, super::PresentationSwapchain::image_count)
	}

	/// Number of independent WSI frame slots owned by this Presenter.
	pub fn frames_in_flight(&self) -> usize {
		self
			.swapchain
			.as_ref()
			.map_or(0, super::PresentationSwapchain::frames_in_flight)
	}

	/// Acquire, transition, submit, and present one empty WSI frame.
	/// Returns `true` when the surface became suboptimal and should be recreated.
	pub fn present_empty_frame(&mut self) -> Result<bool> {
		self.present_clear_frame(crate::Color::default())
	}

	/// Acquire, clear with a linear RGBA color, submit, and present one WSI frame.
	pub fn present_clear_frame(&mut self, color: crate::Color) -> Result<bool> {
		if !color
			.to_array()
			.iter()
			.all(|value| value.is_finite() && (0.0..=1.0).contains(value))
		{
			return Err(crate::Error::invalid_argument(
				"Presenter clear color must be finite RGBA values in [0, 1]",
			));
		}
		self
			.swapchain
			.as_mut()
			.ok_or_else(|| crate::Error::invalid_argument("Presenter has no initialized swapchain"))?
			.present_clear(color.to_array())
	}

	/// Scale and present one native image-backed Texture.
	pub fn present_texture(&mut self, texture: &Texture) -> Result<bool> {
		let swapchain = self
			.swapchain
			.as_mut()
			.ok_or_else(|| crate::Error::invalid_argument("Presenter has no initialized swapchain"))?;
		if let Some(native) = texture.native() {
			return swapchain.present_texture(native);
		}
		if let Some((image, _view, width, height, retention)) = texture.render_target_image() {
			return swapchain.present_render_target(image, [width, height], retention);
		}
		Err(crate::Error::failed_precondition(
			"Presenter::present_texture requires a buffer-backed or render-target-backed Texture",
		))
	}

	/// Compose the UI layer and present it directly to the swapchain.
	///
	/// `record` fills a compute command buffer with the UI dispatches that
	/// write into `compose_image`.  The swapchain blit and present run in the
	/// same submission — no host readback, no extra queue round-trip.
	pub fn present_compose(
		&mut self,
		compose_image: ash::vk::Image,
		compose_width: u32,
		compose_height: u32,
		descriptor_set: ash::vk::DescriptorSet,
		pipeline_layout: ash::vk::PipelineLayout,
		record: impl FnOnce(&ash::Device, ash::vk::CommandBuffer) -> crate::Result<()>,
	) -> crate::Result<bool> {
		self
			.swapchain
			.as_mut()
			.ok_or_else(|| crate::Error::invalid_argument("Presenter has no initialized swapchain"))?
			.present_compose(
				compose_image,
				compose_width,
				compose_height,
				descriptor_set,
				pipeline_layout,
				record,
			)
	}

	/// Return the current swapchain image extent, or `None` when none is allocated.
	pub fn swapchain_extent_px(&self) -> Option<[u32; 2]> {
		self.swapchain.as_ref().map(|sc| sc.extent_px())
	}

	fn surface(&self) -> &PresentationSurface<'window> {
		self.surface.as_ref().expect("Presenter surface is live")
	}
}

impl Drop for Presenter<'_> {
	fn drop(&mut self) {
		if self
			.swapchain
			.as_ref()
			.is_some_and(super::PresentationSwapchain::has_submitted_frames)
		{
			// Drop cannot wait for the presentation engine. Retain the complete WSI
			// ownership graph rather than destroying a swapchain, semaphores, or its
			// surface while presentation may still reference them. `close_swapchain`
			// is the normal explicit, failure-bearing retirement boundary.
			if let Some(swapchain) = self.swapchain.take() {
				std::mem::forget(swapchain);
			}
			if let Some(surface) = self.surface.take() {
				std::mem::forget(surface);
			}
			return;
		}
		// Explicit field order: destroy an unused/explicitly-idle swapchain before
		// destroying the surface it was created from.
		self.swapchain = None;
		self.surface = None;
	}
}
