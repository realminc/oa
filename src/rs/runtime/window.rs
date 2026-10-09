//! SDL3 platform window used by [`super::Presenter`].

use crate::{Error, Result};

/// SDL3-owned Vulkan-capable window and input/event source.
pub struct SdlWindow {
	_sdl: sdl3::Sdl,
	_video: sdl3::VideoSubsystem,
	pub(crate) window: sdl3::video::Window,
}

impl SdlWindow {
	/// Create a resizable Vulkan-capable SDL3 window.
	pub fn new(title: &str, width: u32, height: u32) -> Result<Self> {
		if width == 0 || height == 0 {
			return Err(Error::invalid_argument(
				"runtime::SdlWindow requires non-zero extent",
			));
		}
		let sdl =
			sdl3::init().map_err(|source| Error::backend_failure("SDL3", "initialization", source))?;
		let video = sdl
			.video()
			.map_err(|source| Error::backend_failure("SDL3", "video initialization", source))?;
		let window = video
			.window(title, width, height)
			.vulkan()
			.resizable()
			.high_pixel_density()
			.build()
			.map_err(|source| Error::backend_failure("SDL3", "Vulkan-window creation", source))?;
		Ok(Self {
			_sdl: sdl,
			_video: video,
			window,
		})
	}

	/// Return the Vulkan instance extensions SDL requires for this window.
	pub fn required_instance_extensions(&self) -> Result<Vec<String>> {
		self
			.window
			.vulkan_instance_extensions()
			.map_err(|source| Error::backend_failure("SDL3", "Vulkan instance-extension query", source))
	}

	/// Return the drawable pixel extent used by Vulkan presentation.
	///
	/// This deliberately differs from SDL's logical window size on high-density
	/// displays. A zero extent means the window is minimized and must not cause
	/// swapchain allocation or acquisition.
	pub fn drawable_extent(&self) -> [u32; 2] {
		let (width, height) = self.window.size_in_pixels();
		[width, height]
	}

	/// Obtain SDL's process-wide event pump for Presenter/UI input.
	pub fn event_pump(&self) -> Result<sdl3::EventPump> {
		self
			._sdl
			.event_pump()
			.map_err(|source| Error::backend_failure("SDL3", "event-pump creation", source))
	}
}
