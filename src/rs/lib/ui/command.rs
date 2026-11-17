//! UI draw commands accumulated during one frame.
//!
//! Each variant maps to one draw (RoundedMask to two) inside the single
//! dynamic-rendering pass recorded by `Ui::record_render`.
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
	/// Blit a sampled RGBA8 image into the compose image at `(dst_x, dst_y)`.
	BlitRgba {
		/// Sampled-image view of the source; the owning `Texture` is retained
		/// in `frame_textures` for the submission.
		view: vk::ImageView,
		src_w: u32,
		src_h: u32,
		dst_x: i32,
		dst_y: i32,
		dst_w: u32,
		dst_h: u32,
		clip: PixelRect,
		/// 0 = RGBA; 1..=4 select R/G/B/A and replicate it to RGB.
		channel: u32,
		/// Bilinear sampling when the destination downscales the source;
		/// exact nearest otherwise.
		linear: bool,
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
		/// Band fill between two `count`-sample y blocks (`lo` then `hi`).
		band: bool,
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
