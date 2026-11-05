//! Screen capture sessions producing engine-owned video frames.
//!
//! Donor: `source/cpp/include/oa/vision/screenCapture.h` and
//! `source/cpp/lib/oa/vision/capture/screenCapture.cpp`.
//!
//! The donor's only backend is xdg-desktop-portal plus PipeWire DMA-BUF on
//! Wayland, where the portal grant is interactive and frame slots stay
//! producer-owned until exact consumer completion. No portal, PipeWire, or
//! external-memory-import route exists in this build, so `open` reports
//! [`ErrorKind::MissingCapability`] and [`ScreenCapture::is_supported`]
//! returns `false`; the capability-gated surface keeps the donor's public
//! shape stable for the backend vertical that lands it.

pub use crate::vision::types::ScreenCaptureConfig;
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

/// Stateful screen capture session.
///
/// Sessions are constructed by [`ScreenCapture::open`]. This build links no
/// platform capture backend, so opening always reports
/// [`ErrorKind::MissingCapability`] and no session can exist; `poll`,
/// `release`, and `close` land with the portal/PipeWire backend that produces
/// them.
pub struct ScreenCapture(());

impl ScreenCapture {
	/// Return whether this build and platform can open a screen capture
	/// session.
	///
	/// The donor route requires Wayland, xdg-desktop-portal, PipeWire, and
	/// Vulkan external-memory import. It returns `false` until that backend
	/// is linked.
	#[must_use]
	pub const fn is_supported() -> bool {
		false
	}

	/// Open a screen capture session against the engine's device.
	///
	/// The donor contract allows the compositor's interactive picker to run
	/// during this call.
	///
	/// # Errors
	///
	/// Returns [`ErrorKind::InvalidArgument`] for a zero preferred extent,
	/// frame rate, or ring depth, and [`ErrorKind::MissingCapability`] while
	/// no platform backend is linked.
	pub fn open(engine: &Engine, config: &ScreenCaptureConfig) -> Result<Self> {
		if config.preferred_width == 0 || config.preferred_height == 0 {
			return Err(Error::invalid_argument(
				"screen capture preferred extent must be non-zero",
			));
		}
		if config.preferred_fps == 0 {
			return Err(Error::invalid_argument(
				"screen capture preferred frame rate must be non-zero",
			));
		}
		if config.ring_frames == 0 {
			return Err(Error::invalid_argument(
				"screen capture ring depth must be non-zero",
			));
		}
		let _engine = engine.handle().clone();
		Err(Error::missing_capability(
			"screen capture requires a portal/PipeWire backend that is not \
			 linked into this build",
		))
	}
}
