//! SDL3 platform window used by [`super::Presenter`].

use crate::{Error, Result};

pub(crate) mod decoration;

/// SDL3-owned Vulkan-capable window and input/event source.
pub struct SdlWindow {
	_sdl: sdl3::Sdl,
	_video: sdl3::VideoSubsystem,
	pub(crate) window: sdl3::video::Window,
	pub(crate) client_frame: bool,
}

impl SdlWindow {
	/// Create a resizable Vulkan-capable SDL3 window.
	pub fn new(title: &str, width: u32, height: u32) -> Result<Self> {
		Self::with_client_frame(title, width, height, false)
	}

	pub(crate) fn with_client_frame(
		title: &str,
		width: u32,
		height: u32,
		requested: bool,
	) -> Result<Self> {
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
		let create = |borderless: bool, transparent: bool| {
			let mut builder = video.window(
				title,
				if requested { width.max(320) } else { width },
				if requested { height.max(240) } else { height },
			);
			builder.vulkan().resizable().high_pixel_density();
			if borderless {
				builder.borderless();
			}
			if transparent {
				let flags = builder.flags() | sdl3::video::WindowFlags::TRANSPARENT;
				builder.set_flags(flags);
			}
			builder
				.build()
				.map_err(|source| Error::backend_failure("SDL3", "Vulkan-window creation", source))
		};
		let mut window = if requested {
			create(true, true).or_else(|_| create(true, false))?
		} else {
			create(false, false)?
		};
		let mut client_frame = false;
		if requested {
			// SAFETY: SDL invokes this stateless callback only while this exact
			// owned window lives. It retains no borrowed Rust state or userdata.
			client_frame = unsafe {
				sdl3::sys::video::SDL_SetWindowHitTest(
					window.raw(),
					Some(client_frame_hit_test),
					std::ptr::null_mut(),
				)
			};
			if client_frame {
				window
					.set_minimum_size(320, 240)
					.map_err(|source| Error::backend_failure("SDL3", "viewer minimum extent", source))?;
			} else {
				window = create(false, false)?;
			}
		}
		Ok(Self {
			_sdl: sdl,
			_video: video,
			window,
			client_frame,
		})
	}

	/// Native requests mutate SDL's window, not the retained Rust owner. The
	/// shared surface borrow still prevents destruction while a Presenter exists.
	pub(crate) fn minimize(&self) -> Result<()> {
		// SAFETY: live owned SDL window on the event thread.
		if unsafe { sdl3::sys::video::SDL_MinimizeWindow(self.window.raw()) } {
			Ok(())
		} else {
			Err(Error::backend_failure(
				"SDL3",
				"window minimize",
				sdl3::get_error(),
			))
		}
	}

	pub(crate) fn toggle_maximized(&self) -> Result<()> {
		// SAFETY: live owned SDL window on the event thread; no Rust field is mutated.
		let success = unsafe {
			if self.window.is_maximized() {
				sdl3::sys::video::SDL_RestoreWindow(self.window.raw())
			} else {
				sdl3::sys::video::SDL_MaximizeWindow(self.window.raw())
			}
		};
		if success {
			Ok(())
		} else {
			Err(Error::backend_failure(
				"SDL3",
				"window maximize/restore",
				sdl3::get_error(),
			))
		}
	}

	pub(crate) fn toggle_fullscreen(&self) -> Result<()> {
		let fullscreen = self.window.fullscreen_state() != sdl3::video::FullscreenType::Off;
		// SAFETY: live owned SDL window on the event thread; compositor owns the transition.
		if unsafe { sdl3::sys::video::SDL_SetWindowFullscreen(self.window.raw(), !fullscreen) } {
			Ok(())
		} else {
			Err(Error::backend_failure(
				"SDL3",
				"window fullscreen",
				sdl3::get_error(),
			))
		}
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

	/// Enable or disable platform text input (`SDL_StartTextInput` /
	/// `SDL_StopTextInput`).  Keep it enabled only while a text widget holds
	/// focus so the OS does not steal key events or show stray IME UI.
	pub fn set_text_input_active(&self, active: bool) {
		let util = self._video.text_input();
		if active {
			util.start(&self.window);
		} else {
			util.stop(&self.window);
		}
	}

	/// Anchor the OS IME candidate window at a physical-pixel rect.
	///
	/// `cursor_offset` is the caret's pixel offset from the rect's left edge.
	/// SDL expects window-logical coordinates, so the rect is divided by the
	/// window's pixel scale before the call.
	pub fn set_text_input_area(&self, x: i32, y: i32, w: u32, h: u32, cursor_offset: i32) {
		let (sx, sy) = crate::input::sdl_events::window_pixel_scale(&self.window);
		let logical = sdl3::rect::Rect::new(
			(x as f32 / sx) as i32,
			(y as f32 / sy) as i32,
			(w as f32 / sx) as u32,
			(h as f32 / sy) as u32,
		);
		self
			._video
			.text_input()
			.set_rect(&self.window, logical, (cursor_offset as f32 / sx) as i32);
	}

	/// Ids of standard-layout gamepads already connected — open each with
	/// [`SdlWindow::open_gamepad`] since already-present pads don't re-emit
	/// `GamepadAdded`.
	pub fn connected_gamepads(&self) -> Vec<u32> {
		self
			._sdl
			.gamepad()
			.and_then(|subsystem| subsystem.gamepads())
			.map(|ids| ids.iter().map(|id| id.raw()).collect())
			.unwrap_or_default()
	}

	/// Open a connected standard-layout gamepad.
	///
	/// Call with the instance id from a `GamepadAdded` event and keep the
	/// returned handle alive for as long as the pad stays connected — dropping
	/// it closes the device. Button/axis state arrives through the event pump
	/// as `Gamepad*` events.
	pub fn open_gamepad(&self, id: u32) -> Result<sdl3::gamepad::Gamepad> {
		let subsystem = self
			._sdl
			.gamepad()
			.map_err(|source| Error::backend_failure("SDL3", "gamepad subsystem", source))?;
		subsystem
			.open(sdl3::joystick::JoystickId::new(id))
			.map_err(|source| Error::backend_failure("SDL3", "gamepad open", source))
	}
}

// SDL supplies the current native dimensions/flags during interactive resizing,
// so no copied geometry, RefCell borrow, or captured pointer needs refreshing.
unsafe extern "C" fn client_frame_hit_test(
	window: *mut sdl3::sys::video::SDL_Window,
	point: *const sdl3::sys::rect::SDL_Point,
	_: *mut std::ffi::c_void,
) -> sdl3::sys::video::SDL_HitTestResult {
	use sdl3::sys::video::*;
	if window.is_null() || point.is_null() {
		return SDL_HITTEST_NORMAL;
	}
	let mut width = 0;
	let mut height = 0;
	// SAFETY: SDL owns the callback invocation and supplies live window/point pointers.
	unsafe {
		if !SDL_GetWindowSize(window, &mut width, &mut height) {
			return SDL_HITTEST_NORMAL;
		}
		let flags = SDL_GetWindowFlags(window);
		decoration::hit_test(
			(*point).x,
			(*point).y,
			width,
			height,
			flags.0 & SDL_WINDOW_RESIZABLE.0 != 0,
			flags.0 & SDL_WINDOW_MAXIMIZED.0 != 0,
		)
		.to_ll()
	}
}
