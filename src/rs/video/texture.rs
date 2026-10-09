//! Stateless Render-to-Video frame adaptation.

use crate::{Error, Event, Result, Texture, VideoFrame};

use super::{VideoColorInfo, VideoFrameTiming};

/// Retain a render Texture as one timestamped video frame.
///
/// The returned frame shares a buffer-backed Texture's packed RGBA8 storage,
/// semantic identity, and exact Matrix producer readiness. It performs no
/// pixel copy, submission, wait, or Image conversion. Render targets require
/// [`from_texture_with_ready`] so their producer completion is not lost.
///
/// # Errors
///
/// Returns an error for a render-target-backed Texture.
pub fn from_texture(
	texture: &Texture,
	timing: VideoFrameTiming,
	color: VideoColorInfo,
) -> Result<VideoFrame> {
	if texture.is_render_target() {
		return Err(Error::invalid_argument(
			"render-target-backed video frame requires a producer Event",
		));
	}
	Ok(VideoFrame::from_texture(
		texture.clone(),
		timing,
		color,
		None,
	))
}

/// Retain a texture and its exact producer completion in one video frame.
///
/// This is the render-target path: it shares the texture and event without a
/// readback, wait, or pixel copy. The event must belong to the texture's Engine.
///
/// # Errors
///
/// Returns an error when `ready` comes from a different Engine.
pub fn from_texture_with_ready(
	texture: &Texture,
	timing: VideoFrameTiming,
	color: VideoColorInfo,
	ready: &Event,
) -> Result<VideoFrame> {
	if !texture.engine_handle().owns_event(ready) {
		return Err(Error::invalid_argument(
			"video frame producer event belongs to another Engine",
		));
	}
	Ok(VideoFrame::from_texture(
		texture.clone(),
		timing,
		color,
		Some(ready.clone()),
	))
}
