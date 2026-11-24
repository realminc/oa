//! Screen capture sessions producing engine-owned video frames.
//!
//! Donor: `source/cpp/include/oa/vision/screenCapture.h` and
//! `source/cpp/lib/oa/vision/capture/screenCapture.cpp`.
//!
//! Donor mapped-memory route, with portal consent and nonblocking PipeWire
//! polling on the Engine-owning thread. Returned textures own their pixels;
//! producer buffers are copied and released before `poll` returns. DMA-BUF
//! import remains separate, unimplemented qualification work.

#[cfg(all(target_os = "linux", feature = "pipewire"))]
mod pipewire;
#[cfg(all(target_os = "linux", feature = "pipewire"))]
mod portal;

#[cfg(all(target_os = "linux", feature = "pipewire"))]
use crate::runtime::EngineHandle;
use crate::video::VideoFrame;
#[cfg(all(target_os = "linux", feature = "pipewire"))]
use crate::video::{VideoColorInfo, VideoColorMatrix, VideoColorRange, VideoFrameTiming};
pub use crate::vision::types::ScreenCaptureConfig;
#[cfg(all(target_os = "linux", feature = "pipewire"))]
use crate::{DType, Matrix};
use crate::{Engine, Error, Result};

/// Whether the capture targets whatever the compositor offers, an entire
/// monitor, or an individual window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScreenCaptureTarget {
	/// Let the compositor present its own monitor-or-window picker.
	#[default]
	MonitorOrWindow,
	/// Capture an entire monitor.
	Monitor,
	/// Capture an individual window.
	Window,
}

/// Whether captured frames composite the cursor into the image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScreenCaptureCursor {
	/// Frames never contain the cursor.
	Hidden,
	/// Frames contain the cursor wherever the compositor draws it.
	#[default]
	Embedded,
}

impl Default for ScreenCaptureConfig {
	fn default() -> Self {
		Self {
			target: ScreenCaptureTarget::MonitorOrWindow,
			cursor: ScreenCaptureCursor::Embedded,
			preferred_width: 1920,
			preferred_height: 1080,
			preferred_fps: 30,
			ring_frames: 4,
		}
	}
}

/// Portal-approved screen or window capture producing owned RGBA8 frames.
///
/// `open` may show the desktop picker. `poll` is nonblocking and never retains
/// a PipeWire producer slot after returning. Drop releases host resources only;
/// `close` explicitly disconnects the stream and reports a disconnect failure.
pub struct ScreenCapture {
	#[cfg(all(target_os = "linux", feature = "pipewire"))]
	engine: EngineHandle,
	#[cfg(all(target_os = "linux", feature = "pipewire"))]
	fps: u32,
	width: u32,
	height: u32,
	#[cfg(all(target_os = "linux", feature = "pipewire"))]
	backend: Option<pipewire::Backend>,
}

impl ScreenCapture {
	/// Whether this platform has the portal adapter and a session bus.
	/// This does not request permission or guarantee that a portal service or
	/// a selected source is available. Actual admission happens in `open`.
	#[must_use]
	pub fn is_supported() -> bool {
		#[cfg(all(target_os = "linux", feature = "pipewire"))]
		{
			std::env::var_os("DBUS_SESSION_BUS_ADDRESS").is_some() && portal::Portal::is_supported()
		}
		#[cfg(not(all(target_os = "linux", feature = "pipewire")))]
		{
			false
		}
	}

	/// Request desktop consent and open one selected screen/window stream.
	///
	/// # Errors
	/// Rejects invalid extent, FPS or ring hints, missing platform support,
	/// cancelled/denied permission, and portal/PipeWire setup failures.
	pub fn open(engine: &Engine, config: &ScreenCaptureConfig) -> Result<Self> {
		if config.preferred_width == 0
			|| config.preferred_height == 0
			|| config.preferred_width > 16384
			|| config.preferred_height > 16384
		{
			return Err(Error::invalid_argument(
				"screen capture preferred extent must be in 1..=16384",
			));
		}
		if config.preferred_fps == 0 || config.preferred_fps > 240 {
			return Err(Error::invalid_argument(
				"screen capture preferred frame rate must be in 1..=240",
			));
		}
		if config.ring_frames == 0 || config.ring_frames > 64 {
			return Err(Error::invalid_argument(
				"screen capture ring depth must be in 1..=64",
			));
		}
		if !Self::is_supported() {
			return Err(Error::missing_capability(
				"screen capture requires Linux, libportal and a desktop session bus",
			));
		}
		#[cfg(all(target_os = "linux", feature = "pipewire"))]
		{
			Ok(Self {
				engine: engine.handle().clone(),
				fps: config.preferred_fps,
				width: 0,
				height: 0,
				backend: Some(pipewire::Backend::open(config)?),
			})
		}
		#[cfg(not(all(target_os = "linux", feature = "pipewire")))]
		{
			let _ = engine;
			Err(Error::missing_capability(
				"screen capture requires the Linux portal/PipeWire adapter",
			))
		}
	}

	/// Poll once without waiting; return only a new caller-owned RGBA8 frame.
	///
	/// # Errors
	/// Reports closed/revoked streams, malformed frames or texture allocation.
	pub fn poll(&mut self) -> Result<Option<VideoFrame>> {
		#[cfg(all(target_os = "linux", feature = "pipewire"))]
		{
			let backend = self
				.backend
				.as_mut()
				.ok_or_else(|| Error::failed_precondition("ScreenCapture is closed"))?;
			let Some(frame) = backend.poll()? else {
				return Ok(None);
			};
			let width = frame.width as usize;
			let height = frame.height as usize;
			let data = Matrix::allocate(
				&self.engine,
				vec![height, width, 4],
				frame.rgba.len(),
				DType::U8,
			)?;
			data.write_values(&frame.rgba)?;
			let texture = crate::render::texture_from_matrix(data, width, height)?;
			self.width = frame.width;
			self.height = frame.height;
			crate::video::from_texture(
				&texture,
				VideoFrameTiming::new_from_microseconds(frame.pts, 1_000_000 / u64::from(self.fps)),
				VideoColorInfo::new(VideoColorMatrix::Bt709, VideoColorRange::Full),
			)
			.map(Some)
		}
		#[cfg(not(all(target_os = "linux", feature = "pipewire")))]
		{
			Err(Error::failed_precondition("ScreenCapture is closed"))
		}
	}

	/// Explicitly disconnect and release the portal grant. Idempotent.
	/// # Errors
	/// Reports a PipeWire disconnect failure after releasing the session.
	pub fn close(&mut self) -> Result<()> {
		#[cfg(all(target_os = "linux", feature = "pipewire"))]
		if let Some(backend) = self.backend.take() {
			return backend.close();
		}
		Ok(())
	}
	/// Whether PipeWire currently reports a streaming source.
	#[must_use]
	pub fn is_streaming(&self) -> bool {
		#[cfg(all(target_os = "linux", feature = "pipewire"))]
		{
			self
				.backend
				.as_ref()
				.is_some_and(pipewire::Backend::is_streaming)
		}
		#[cfg(not(all(target_os = "linux", feature = "pipewire")))]
		{
			false
		}
	}
	/// Latest delivered frame width; zero before the first frame.
	pub const fn width(&self) -> u32 {
		self.width
	}
	/// Latest delivered frame height; zero before the first frame.
	pub const fn height(&self) -> u32 {
		self.height
	}
}
