//! Camera capture sessions producing engine-owned RGBA8 video frames.
//!
//! Donor: `source/cpp/include/oa/vision/cameraCapture.h` and
//! `source/cpp/lib/oa/vision/capture/cameraCapture.cpp`.
//!
//! This slice owns the donor's SDL capture route: frames arrive as SDL
//! surfaces, are repacked into RGBA8, and upload into packed textures through
//! the session's Engine. The V4L2 MMAP/DMA-BUF zero-copy ring and reconnect
//! policy remain a later slice gated on external-memory admission.

use std::ptr::NonNull;

pub use crate::vision::types::CameraCaptureConfig;
use sdl3::sys::camera::{
	SDL_AcquireCameraFrame, SDL_Camera, SDL_CameraID, SDL_CameraPermissionState, SDL_CameraSpec,
	SDL_CloseCamera, SDL_GetCameraFormat, SDL_GetCameraPermissionState, SDL_GetCameras,
	SDL_OpenCamera, SDL_ReleaseCameraFrame,
};
use sdl3::sys::pixels::SDL_PixelFormat;
use sdl3::sys::stdinc::SDL_free;
use sdl3::sys::surface::SDL_Surface;

use crate::runtime::EngineHandle;
use crate::video::{
	self, VideoColorInfo, VideoColorMatrix, VideoColorRange, VideoFrame, VideoFrameTiming,
};
use crate::{DType, Engine, Error, Matrix, Result};

impl Default for CameraCaptureConfig {
	fn default() -> Self {
		Self {
			device_index: 0,
			width: 1280,
			height: 720,
			fps: 30,
		}
	}
}

/// Stateful camera capture session over SDL's camera backend.
///
/// `poll` acquires the newest queued frame, uploads it into an owned RGBA8
/// texture, and returns it as a [`VideoFrame`]; the SDL surface is released
/// immediately, so no ring-slot release contract exists on this route. The
/// returned frame's presentation timestamp is the camera's hardware capture
/// timestamp in microseconds and its duration follows the negotiated frame
/// rate.
pub struct CameraCapture {
	engine: EngineHandle,
	// SDL lifetime owners: dropping the subsystem deinitializes the camera
	// backend, and dropping the context runs SDL_Quit when it is the last.
	_sdl: sdl3::Sdl,
	_camera_subsystem: sdl3::CameraSubsystem,
	camera: Option<NonNull<SDL_Camera>>,
	width: u32,
	height: u32,
	fps_numerator: u32,
	fps_denominator: u32,
}

impl CameraCapture {
	/// Open the selected camera and negotiate the closest supported format.
	///
	/// The session requests RGBA8888 output so no pixel conversion happens
	/// outside SDL. On platforms with a camera permission gate the camera opens
	/// before approval; `poll` reports [`ErrorKind::MissingCapability`] when the
	/// user denies access and `None` while approval is pending.
	///
	/// # Errors
	///
	/// Returns an error for a zero extent, when no camera device matches
	/// `device_index`, or when SDL cannot open or describe the device.
	///
	/// [`ErrorKind::MissingCapability`]: crate::ErrorKind::MissingCapability
	pub fn open(engine: &Engine, config: &CameraCaptureConfig) -> Result<Self> {
		if config.width == 0 || config.height == 0 {
			return Err(Error::invalid_argument(
				"camera capture extent must be non-zero",
			));
		}
		if config.fps == 0 {
			return Err(Error::invalid_argument(
				"camera capture frame rate must be non-zero",
			));
		}
		let sdl =
			sdl3::init().map_err(|source| Error::backend_failure("SDL3", "initialization", source))?;
		let subsystem = sdl
			.camera()
			.map_err(|source| Error::backend_failure("SDL3", "camera subsystem", source))?;
		let camera_id = camera_id_for_index(config.device_index)?;
		let request = SDL_CameraSpec {
			format: SDL_PixelFormat::RGBA8888,
			width: config.width as i32,
			height: config.height as i32,
			framerate_numerator: config.fps as i32,
			framerate_denominator: 1,
			..Default::default()
		};
		// SAFETY: `request` outlives the call; ownership of a non-null camera
		// moves into the session.
		let raw = unsafe { SDL_OpenCamera(camera_id, &request) };
		let camera = NonNull::new(raw)
			.ok_or_else(|| Error::backend_failure("SDL3", "camera open", sdl3::get_error()))?;
		let mut actual = SDL_CameraSpec::default();
		// SAFETY: `camera` was just opened and `actual` is a valid out pointer.
		if !unsafe { SDL_GetCameraFormat(camera.as_ptr(), &mut actual) } {
			let source = sdl3::get_error();
			// SAFETY: the camera was opened above and is not yet owned.
			unsafe { SDL_CloseCamera(camera.as_ptr()) };
			return Err(Error::backend_failure(
				"SDL3",
				"camera format query",
				source,
			));
		}
		let fps_numerator = actual.framerate_numerator.max(0) as u32;
		let fps_denominator = actual.framerate_denominator.max(0) as u32;
		Ok(Self {
			engine: engine.handle().clone(),
			_sdl: sdl,
			_camera_subsystem: subsystem,
			camera: Some(camera),
			width: actual.width.max(0) as u32,
			height: actual.height.max(0) as u32,
			fps_numerator: fps_numerator.max(1),
			fps_denominator: fps_denominator.max(1),
		})
	}

	/// Acquire and upload the newest queued camera frame.
	///
	/// Returns `Ok(None)` when the camera has not produced another frame yet —
	/// this is also the steady state while a platform permission request is
	/// pending. Callers drive this in their own pacing loop; `poll` never
	/// blocks.
	///
	/// # Errors
	///
	/// Returns an error after close, when camera access was denied, for an
	/// unexpected frame layout, or when the texture upload fails.
	pub fn poll(&mut self) -> Result<Option<VideoFrame>> {
		let camera = self
			.camera
			.ok_or_else(|| Error::failed_precondition("CameraCapture is closed"))?;
		let mut timestamp_ns = 0_u64;
		// SAFETY: `camera` is open and `timestamp_ns` is a valid out pointer;
		// a non-null surface is released below before returning.
		let frame_ptr = unsafe { SDL_AcquireCameraFrame(camera.as_ptr(), &mut timestamp_ns) };
		if frame_ptr.is_null() {
			// SAFETY: the camera is open.
			return match unsafe { SDL_GetCameraPermissionState(camera.as_ptr()) } {
				SDL_CameraPermissionState::DENIED => Err(Error::missing_capability(
					"camera access was denied by the platform",
				)),
				_ => Ok(None),
			};
		}
		// SAFETY: SDL returns a live surface until released; reading its pixels
		// and releasing it both happen before this function returns.
		let result = unsafe { self.upload_surface(&*frame_ptr, timestamp_ns) };
		// SAFETY: `frame_ptr` came from SDL_AcquireCameraFrame above.
		unsafe { SDL_ReleaseCameraFrame(camera.as_ptr(), frame_ptr) };
		result.map(Some)
	}

	/// Close the session and release the camera.
	///
	/// Frames already returned remain owned by their callers; SDL surfaces are
	/// released eagerly at `poll` time so nothing is deferred to close.
	pub fn close(&mut self) {
		if let Some(camera) = self.camera.take() {
			// SAFETY: the session owns this opened camera.
			unsafe { SDL_CloseCamera(camera.as_ptr()) };
		}
	}

	/// Return whether the session still owns an open camera.
	#[must_use]
	pub const fn is_streaming(&self) -> bool {
		self.camera.is_some()
	}

	/// Return the negotiated frame width in pixels.
	#[must_use]
	pub const fn width(&self) -> u32 {
		self.width
	}

	/// Return the negotiated frame height in pixels.
	#[must_use]
	pub const fn height(&self) -> u32 {
		self.height
	}

	/// Return the negotiated frame rate, or `0.0` when the device reported none.
	#[must_use]
	pub fn frame_rate(&self) -> f64 {
		if self.fps_denominator == 0 {
			return 0.0;
		}
		f64::from(self.fps_numerator) / f64::from(self.fps_denominator)
	}

	unsafe fn upload_surface(&self, surface: &SDL_Surface, timestamp_ns: u64) -> Result<VideoFrame> {
		if surface.format != SDL_PixelFormat::RGBA8888 {
			return Err(Error::backend_failure(
				"SDL3",
				"camera frame format",
				std::io::Error::other(format!("expected RGBA8888, got {:?}", surface.format)),
			));
		}
		let width = usize::try_from(surface.w)
			.map_err(|_| Error::out_of_range("camera frame width is negative"))?;
		let height = usize::try_from(surface.h)
			.map_err(|_| Error::out_of_range("camera frame height is negative"))?;
		let pitch = usize::try_from(surface.pitch)
			.map_err(|_| Error::out_of_range("camera frame pitch is negative"))?;
		if surface.pixels.is_null() || width == 0 || height == 0 {
			return Err(Error::data_loss("camera produced an empty frame"));
		}
		let row_bytes = width
			.checked_mul(4)
			.ok_or_else(|| Error::out_of_range("camera frame row size overflows"))?;
		if pitch < row_bytes {
			return Err(Error::data_loss(
				"camera frame pitch is smaller than its RGBA8 row",
			));
		}
		// SAFETY: SDL owns `surface` for this call and guarantees `pitch * h`
		// readable bytes for a live camera frame.
		let source = unsafe { std::slice::from_raw_parts(surface.pixels.cast::<u8>(), pitch * height) };
		let rgba = if pitch == row_bytes {
			source.to_vec()
		} else {
			let mut packed = Vec::with_capacity(row_bytes * height);
			for row in source.chunks_exact(pitch) {
				packed.extend_from_slice(&row[..row_bytes]);
			}
			packed
		};
		let element_count = width
			.checked_mul(height)
			.and_then(|pixels| pixels.checked_mul(4))
			.ok_or_else(|| Error::out_of_range("camera frame byte count overflows"))?;
		let data = Matrix::allocate(
			&self.engine,
			vec![height, width, 4],
			element_count,
			DType::U8,
		)?;
		data.write_values(&rgba)?;
		let texture = crate::render::texture_from_matrix(data, width, height)?;
		let duration_micros =
			u64::from(self.fps_denominator).saturating_mul(1_000_000) / u64::from(self.fps_numerator);
		let timing = VideoFrameTiming::new_from_microseconds(timestamp_ns / 1_000, duration_micros);
		video::from_texture(
			&texture,
			timing,
			VideoColorInfo::new(VideoColorMatrix::Bt709, VideoColorRange::Full),
		)
	}
}

impl Drop for CameraCapture {
	fn drop(&mut self) {
		self.close();
	}
}

fn camera_id_for_index(index: u32) -> Result<SDL_CameraID> {
	let mut count = 0_i32;
	// SAFETY: `count` is a valid out pointer; the returned array is SDL-owned
	// and freed below.
	let list = unsafe { SDL_GetCameras(&mut count) };
	if list.is_null() {
		let source = sdl3::get_error();
		return if source.is_empty() {
			Err(Error::missing_capability("no camera devices are present"))
		} else {
			Err(Error::backend_failure("SDL3", "camera enumeration", source))
		};
	}
	// SAFETY: SDL guarantees `count` entries in the returned allocation.
	let ids = unsafe { std::slice::from_raw_parts(list, count.max(0) as usize) };
	let found = ids.get(index as usize).copied();
	// SAFETY: `list` was returned by SDL_GetCameras and is freed exactly once.
	unsafe { SDL_free(list.cast()) };
	found.ok_or_else(|| {
		Error::invalid_argument(format!(
			"camera device index {index} exceeds the {count} enumerated devices"
		))
	})
}
