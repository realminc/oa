//! UI draw commands accumulated during one frame.
//!
//! Each variant maps directly to one compute dispatch.
//! `record_into` walks the command list and records Vulkan dispatches.
//!
//! Donor reference: `oa::Ui::recordRender` in `ui.cpp`

pub use super::types::PixelRect;
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
	/// Premultiplied rounded viewport mask, applied after composition.
	RoundedMask {
		rect: PixelRect,
		radius: f32,
		outside: Color,
		clip: PixelRect,
	},
	/// Donor antialiased filled icon triangle.
	Triangle {
		points: [[f32; 2]; 3],
		color: Color,
		bounds: PixelRect,
	},
	/// Donor decimal grid and gradient in one image dispatch.
	CanvasGrid { rect: PixelRect, push: [u32; 30] },
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
	/// Direct retained matrix visualization, exact donor ABI.
	Heatmap {
		values_idx: u32,
		rect: PixelRect,
		push: [u32; 18],
	},
	Waveform {
		values_idx: u32,
		rect: PixelRect,
		push: [u32; 14],
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
	///
	/// `dst` is the normalized source-image rectangle the anchors map into;
	/// `None` maps anchors over the full compose image.
	Glyphs {
		glyph_idx: u32,
		atlas_idx: u32,
		count: u32,
		dst: Option<PixelRect>,
		clip: PixelRect,
		atlas_w: u32,
		atlas_h: u32,
	},
	/// Composite planar multi-dtype image planes into one image.
	BlitPlanar {
		/// Bindless buffer index of the R/gray plane.
		r_idx: u32,
		/// Bindless buffer index of the G plane (`u32::MAX` when unused).
		g_idx: u32,
		/// Bindless buffer index of the B plane (`u32::MAX` when unused).
		b_idx: u32,
		/// Bindless buffer index of the A/depth plane (`u32::MAX` when unused).
		a_idx: u32,
		src_w: u32,
		src_h: u32,
		dst_x: i32,
		dst_y: i32,
		/// Per-channel `ImageDtype` bits, two bits each starting at channel 0.
		dtypes: u32,
		channels: u32,
		dst_w: u32,
		dst_h: u32,
		clip: PixelRect,
	},
	/// Batched normalized rectangle borders read from one GPU record buffer.
	RectOutlines {
		/// Bindless buffer index of the packed 32-byte record array.
		rect_idx: u32,
		count: u32,
		dst: PixelRect,
		clip: PixelRect,
		thickness: u32,
		rgba: u32,
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

/// Intersect a destination rect with its clip rect.
///
/// Returns `(visible_w, visible_h, base_x, base_y)` where the base is the
/// visible origin expressed as an offset inside the destination rect — the
/// shader adds it to each thread index so source mapping still sees the full
/// scaled rectangle. `i64` intermediates keep `dst + extent` overflow-safe.
pub(super) fn visible_blit_region(
	dst_x: i32,
	dst_y: i32,
	dst_w: u32,
	dst_h: u32,
	clip: PixelRect,
) -> (u32, u32, u32, u32) {
	let clip_x1 = i64::from(clip.x) + i64::from(clip.w);
	let clip_y1 = i64::from(clip.y) + i64::from(clip.h);
	let vis_x0 = i64::from(dst_x).max(i64::from(clip.x));
	let vis_y0 = i64::from(dst_y).max(i64::from(clip.y));
	let vis_x1 = (i64::from(dst_x) + i64::from(dst_w)).min(clip_x1);
	let vis_y1 = (i64::from(dst_y) + i64::from(dst_h)).min(clip_y1);
	if vis_x1 <= vis_x0 || vis_y1 <= vis_y0 {
		return (0, 0, 0, 0);
	}
	(
		(vis_x1 - vis_x0) as u32,
		(vis_y1 - vis_y0) as u32,
		(vis_x0 - i64::from(dst_x)) as u32,
		(vis_y0 - i64::from(dst_y)) as u32,
	)
}
