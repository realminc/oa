//! Stateless Render-to-Video frame adaptation.

use crate::{Texture, VideoFrame};

use super::{VideoColorInfo, VideoFrameTiming};

/// Retain a render Texture as one timestamped video frame.
///
/// The returned frame shares the Texture's packed RGBA8 storage, semantic
/// identity, and exact Matrix producer readiness. It performs no pixel copy,
/// submission, wait, or Image conversion.
pub fn from_texture(
	texture: &Texture,
	timing: VideoFrameTiming,
	color: VideoColorInfo,
) -> VideoFrame {
	VideoFrame::from_texture(texture.clone(), timing, color)
}
