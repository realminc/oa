//! UI draw commands accumulated during one frame.
//!
//! Each variant maps directly to one compute dispatch.
//! `record_into` walks the command list and records Vulkan dispatches.
//!
//! Donor reference: `oa::Ui::recordRender` in `ui.cpp`

use super::types::PixelRect;
use crate::Color;

// ── Command enum ──────────────────────────────────────────────────────────────

/// One compositor draw command.
#[derive(Clone, Debug)]
pub(crate) enum UiCmd {
	/// Fill the compose image with `color`.
	Clear { color: Color },
	/// Alpha-blend a filled rectangle.
	Rect {
		rect: PixelRect,
		color: Color,
		clip: PixelRect,
		corner_radius: f32,
	},
	/// Alpha-blend a rectangle outline.
	RectOutline {
		rect: PixelRect,
		color: Color,
		thickness: u32,
		clip: PixelRect,
		corner_radius: f32,
	},
	/// Anti-aliased line segment.
	Line {
		x0: f32,
		y0: f32,
		x1: f32,
		y1: f32,
		color: Color,
		thickness: f32,
		bounds: PixelRect,
	},
	/// Blit a host RGBA8 buffer into the compose image at `(dst_x, dst_y)`.
	BlitRgba {
		/// Bindless buffer index of the RGBA8 source bytes.
		src_idx: u32,
		src_w: u32,
		src_h: u32,
		dst_x: i32,
		dst_y: i32,
		dst_w: u32,
		dst_h: u32,
		clip: PixelRect,
		/// 0 = RGBA; 1..=4 select R/G/B/A and replicate it to RGB.
		channel: u32,
	},
	/// GPU line plot over a bindless F32 sample buffer.
	PlotLine {
		/// Bindless buffer index of the F32 sample array.
		values_idx: u32,
		count: u32,
		rect: PixelRect,
		clip: PixelRect,
		x_min: f32,
		x_max: f32,
		y_min: f32,
		y_max: f32,
		color: Color,
		fill: bool,
		explicit_xy: bool,
		antialias_samples: u32,
		line_width: f32,
	},
	/// Hinted grayscale glyph quads sourced from the embedded coverage atlas.
	Glyphs {
		glyph_idx: u32,
		atlas_idx: u32,
		count: u32,
		clip: PixelRect,
		atlas_w: u32,
		atlas_h: u32,
	},
}

// ── Command list ──────────────────────────────────────────────────────────────

/// Ordered list of UI draw commands for one frame.
#[derive(Default)]
pub(crate) struct UiCmdList {
	pub(crate) cmds: Vec<UiCmd>,
}

impl UiCmdList {
	pub(crate) fn clear_all(&mut self) {
		self.cmds.clear();
	}

	pub(crate) fn push(&mut self, cmd: UiCmd) {
		self.cmds.push(cmd);
	}
}

// ── Dispatch helpers ──────────────────────────────────────────────────────────

/// Ceil-divide `n` by `d`.
#[inline]
pub(super) const fn div_ceil(n: u32, d: u32) -> u32 {
	n.div_ceil(d)
}

/// Pack `PixelRect` into the four clip push-constant words.
#[inline]
pub(super) fn clip_words(clip: PixelRect) -> [i32; 4] {
	[clip.x, clip.y, clip.w as i32, clip.h as i32]
}
