//! Legacy CPU reference CV compositing.
//!
//! Donor: `oa/ui/cv.h` and `oa/ui/cv.cpp`.
//!
//! `CvFrame` remains a diagnostic/reference path for saved images with CPU
//! bounding-box annotation. It reads back and uploads a complete RGBA frame
//! and must not be used for realtime playback — use
//! [`crate::ui::DetectionOverlay`] for completion-safe, GPU-resident boxes
//! and hinted coverage labels over image/video views.
//!
//! ```ignore
//! let mut frame = CvFrame::default();
//! frame.w = 1280;
//! frame.h = 720;
//! frame.add_bboxes(detections, &CvBboxesConfig { show_labels: true, ..Default::default() });
//! let annotated = frame.render(&engine)?;
//! ```

use crate::render::{Texture, texture_from_rgba8};
pub use crate::ui::type_gen::CvBboxesConfig;
use crate::{Color, Engine, Error, Result};

/// One normalized-coordinate bounding box with optional label metadata.
///
/// Donor: `oa::CvBbox`.
#[derive(Clone, Debug)]
pub struct CvBbox {
	pub x: f32,
	pub y: f32,
	pub w: f32,
	pub h: f32,
	pub score: f32,
	pub class_id: i32,
	pub label: String,
}

impl Default for CvBbox {
	fn default() -> Self {
		Self {
			x: 0.0,
			y: 0.0,
			w: 0.0,
			h: 0.0,
			score: 1.0,
			class_id: 0,
			label: String::new(),
		}
	}
}

impl Default for CvBboxesConfig {
	fn default() -> Self {
		Self {
			color: Color::error(),
			line_width: 1.5,
			alpha: 1.0,
			label_scale: 1,
			show_labels: true,
			show_scores: true,
		}
	}
}

/// One overlay batch: shared config plus its boxes.
///
/// Donor: `oa::CvOverlayBboxes` (the only `CvOverlay` kind).
#[derive(Clone, Debug)]
pub struct CvOverlay {
	pub config: CvBboxesConfig,
	pub boxes: Vec<CvBbox>,
}

/// Diagnostic frame: borrowed base pixels plus ordered CPU overlays.
///
/// `base` borrows a packed RGBA8 texture; `render` reads it back before
/// compositing. A `None` base composites over a black canvas.
///
/// Donor: `oa::CvFrame`.
#[derive(Default)]
pub struct CvFrame<'a> {
	/// Borrowed RGBA8 device image read back during [`Self::render`].
	pub base: Option<&'a Texture>,
	pub w: i32,
	pub h: i32,
	pub overlays: Vec<CvOverlay>,
}

impl<'a> CvFrame<'a> {
	/// Append a bbox overlay batch, drawn after all previous overlays.
	pub fn add_bboxes(&mut self, boxes: Vec<CvBbox>, config: &CvBboxesConfig) {
		self.overlays.push(CvOverlay {
			config: *config,
			boxes,
		});
	}

	/// Drop every overlay.
	pub fn clear_overlays(&mut self) {
		self.overlays.clear();
	}

	/// Composite overlays over the borrowed base and upload the result.
	///
	/// # Errors
	///
	/// Returns an error for a non-positive extent, an undersized or
	/// unreadable base, or an invalid overlay config or box.
	pub fn render(&self, engine: &Engine) -> Result<Texture> {
		self.render_with_rgba(engine, &[])
	}

	/// Composite overlays over explicit host RGBA8 pixels (or the borrowed
	/// device base when `base_rgba` is empty).
	///
	/// # Errors
	///
	/// Returns an error for a non-positive extent, an undersized base, or an
	/// invalid overlay config or box.
	pub fn render_with_rgba(&self, engine: &Engine, base_rgba: &[u8]) -> Result<Texture> {
		render_frame(self, engine, base_rgba)
	}
}

// ── CPU drawing helpers ──────────────────────────────────────────────────────

fn blend_pixel(dst: &mut [u8], r: u8, g: u8, b: u8, a: u8) {
	let alpha = f32::from(a) / 255.0;
	dst[0] = (f32::from(dst[0]) * (1.0 - alpha) + f32::from(r) * alpha) as u8;
	dst[1] = (f32::from(dst[1]) * (1.0 - alpha) + f32::from(g) * alpha) as u8;
	dst[2] = (f32::from(dst[2]) * (1.0 - alpha) + f32::from(b) * alpha) as u8;
	dst[3] = 255;
}

fn draw_hline(buf: &mut [u8], w: i32, h: i32, x0: i64, x1: i64, y: i64, color: [u8; 4]) {
	if y < 0 || y >= i64::from(h) {
		return;
	}
	let x0 = x0.max(0);
	let x1 = x1.min(i64::from(w) - 1);
	for x in x0..=x1 {
		let offset = (y as usize * w as usize + x as usize) * 4;
		blend_pixel(
			&mut buf[offset..offset + 4],
			color[0],
			color[1],
			color[2],
			color[3],
		);
	}
}

fn draw_vline(buf: &mut [u8], w: i32, h: i32, x: i64, y0: i64, y1: i64, color: [u8; 4]) {
	if x < 0 || x >= i64::from(w) {
		return;
	}
	let y0 = y0.max(0);
	let y1 = y1.min(i64::from(h) - 1);
	for y in y0..=y1 {
		let offset = (y as usize * w as usize + x as usize) * 4;
		blend_pixel(
			&mut buf[offset..offset + 4],
			color[0],
			color[1],
			color[2],
			color[3],
		);
	}
}

#[allow(clippy::too_many_arguments)]
fn draw_rect(
	buf: &mut [u8],
	w: i32,
	h: i32,
	x: i64,
	y: i64,
	rw: i64,
	rh: i64,
	thickness: i32,
	color: [u8; 4],
) {
	for t in 0..i64::from(thickness) {
		draw_hline(buf, w, h, x, x + rw - 1, y + t, color);
		draw_hline(buf, w, h, x, x + rw - 1, y + rh - 1 - t, color);
		draw_vline(buf, w, h, x + t, y, y + rh - 1, color);
		draw_vline(buf, w, h, x + rw - 1 - t, y, y + rh - 1, color);
	}
}

#[allow(clippy::too_many_arguments)]
fn fill_rect(buf: &mut [u8], w: i32, h: i32, x: i64, y: i64, rw: i64, rh: i64, color: [u8; 4]) {
	let y0 = y.max(0);
	let y1 = (y + rh).min(i64::from(h));
	for row in y0..y1 {
		draw_hline(buf, w, h, x, x + rw - 1, row, color);
	}
}

// Compact 5x7 label font: digits, ':', '.', ' ', '%', A-Z.
const FONT_5X7: [[u8; 7]; 40] = [
	[
		0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
	], // 0
	[
		0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
	], // 1
	[
		0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
	], // 2
	[
		0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110,
	], // 3
	[
		0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
	], // 4
	[
		0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
	], // 5
	[
		0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
	], // 6
	[
		0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
	], // 7
	[
		0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
	], // 8
	[
		0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
	], // 9
	[
		0b00000, 0b00100, 0b00000, 0b00000, 0b00100, 0b00000, 0b00000,
	], // :
	[
		0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100,
	], // .
	[
		0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000,
	], // space
	[
		0b11000, 0b11001, 0b00010, 0b00100, 0b01000, 0b10011, 0b00011,
	], // %
	[
		0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
	], // A
	[
		0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
	], // B
	[
		0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
	], // C
	[
		0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
	], // D
	[
		0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
	], // E
	[
		0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
	], // F
	[
		0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110,
	], // G
	[
		0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
	], // H
	[
		0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
	], // I
	[
		0b00001, 0b00001, 0b00001, 0b00001, 0b10001, 0b10001, 0b01110,
	], // J
	[
		0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
	], // K
	[
		0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
	], // L
	[
		0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
	], // M
	[
		0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
	], // N
	[
		0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
	], // O
	[
		0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
	], // P
	[
		0b01110, 0b10001, 0b10001, 0b10101, 0b10001, 0b10010, 0b01101,
	], // Q
	[
		0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
	], // R
	[
		0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
	], // S
	[
		0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
	], // T
	[
		0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
	], // U
	[
		0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
	], // V
	[
		0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b10101, 0b01010,
	], // W
	[
		0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
	], // X
	[
		0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
	], // Y
	[
		0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
	], // Z
];

fn char_index(c: u8) -> usize {
	if c.is_ascii_digit() {
		return usize::from(c - b'0');
	}
	match c {
		b':' => 10,
		b'.' => 11,
		b' ' => 12,
		b'%' => 13,
		_ => {
			let upper = c.to_ascii_uppercase();
			if upper.is_ascii_uppercase() {
				14 + usize::from(upper - b'A')
			} else {
				12 // fallback: space
			}
		}
	}
}

#[allow(clippy::too_many_arguments)]
fn draw_char(
	buf: &mut [u8],
	w: i32,
	h: i32,
	x: i64,
	y: i64,
	c: u8,
	scale: i32,
	r: u8,
	g: u8,
	b: u8,
) {
	let idx = char_index(c);
	for (row, bits) in FONT_5X7[idx].iter().enumerate() {
		for col in 0..5_i64 {
			if bits & (0b10000_u8 >> col) != 0 {
				for sy in 0..i64::from(scale) {
					for sx in 0..i64::from(scale) {
						let px = x + col * i64::from(scale) + sx;
						let py = y + row as i64 * i64::from(scale) + sy;
						if px >= 0 && px < i64::from(w) && py >= 0 && py < i64::from(h) {
							let offset = (py as usize * w as usize + px as usize) * 4;
							buf[offset] = r;
							buf[offset + 1] = g;
							buf[offset + 2] = b;
							buf[offset + 3] = 255;
						}
					}
				}
			}
		}
	}
}

#[allow(clippy::too_many_arguments)]
fn draw_string(
	buf: &mut [u8],
	w: i32,
	h: i32,
	x: i64,
	y: i64,
	text: &str,
	scale: i32,
	color: [u8; 3],
) {
	let mut cx = x;
	for &c in text.as_bytes() {
		draw_char(buf, w, h, cx, y, c, scale, color[0], color[1], color[2]);
		cx += 6 * i64::from(scale);
	}
}

fn render_frame(frame: &CvFrame<'_>, engine: &Engine, base_rgba: &[u8]) -> Result<Texture> {
	let w = frame.w;
	let h = frame.h;
	if w <= 0 || h <= 0 {
		return Err(Error::invalid_argument("CvFrame: invalid W/H"));
	}
	let n_bytes = w as usize * h as usize * 4;
	let mut pixels = vec![0_u8; n_bytes];

	if !base_rgba.is_empty() && base_rgba.len() < n_bytes {
		return Err(Error::invalid_argument(
			"CvFrame::render: host base is smaller than W*H*4",
		));
	}
	if base_rgba.len() >= n_bytes {
		pixels.copy_from_slice(&base_rgba[..n_bytes]);
	} else if let Some(base) = frame.base {
		let readback = base.read_rgba8()?;
		if readback.len() < n_bytes {
			return Err(Error::invalid_argument(
				"CvFrame::render: base is null or smaller than W*H*4",
			));
		}
		pixels.copy_from_slice(&readback[..n_bytes]);
	} else {
		// Black canvas fallback — keep alpha=255 so overlays show on it.
		for index in (3..pixels.len()).step_by(4) {
			pixels[index] = 255;
		}
	}

	for overlay in &frame.overlays {
		let config = &overlay.config;
		if !config.alpha.is_finite()
			|| !(0.0..=1.0).contains(&config.alpha)
			|| !config.line_width.is_finite()
			|| !(0.0..=1024.0).contains(&config.line_width)
			|| config.line_width <= 0.0
			|| !(1..=16).contains(&config.label_scale)
			|| ![
				config.color.r,
				config.color.g,
				config.color.b,
				config.color.a,
			]
			.iter()
			.all(|component| component.is_finite())
		{
			return Err(Error::invalid_argument(
				"CvFrame: invalid bounding-box overlay config",
			));
		}
		let packed = config.color.to_u32();
		let r = (packed >> 24) as u8;
		let g = (packed >> 16) as u8;
		let b = (packed >> 8) as u8;
		let a = (config.alpha * 255.0) as u8;
		let thickness = config.line_width.max(1.0) as i32;

		for bbox in &overlay.boxes {
			let right = f64::from(bbox.x) + f64::from(bbox.w);
			let bottom = f64::from(bbox.y) + f64::from(bbox.h);
			let min_coordinate = f64::from(i32::MIN);
			let max_coordinate = f64::from(i32::MAX);
			if !bbox.x.is_finite()
				|| !bbox.y.is_finite()
				|| !bbox.w.is_finite()
				|| !bbox.h.is_finite()
				|| !bbox.score.is_finite()
				|| bbox.w <= 0.0
				|| bbox.h <= 0.0
				|| f64::from(bbox.x) < min_coordinate
				|| f64::from(bbox.y) < min_coordinate
				|| f64::from(bbox.w) > max_coordinate
				|| f64::from(bbox.h) > max_coordinate
				|| right > max_coordinate
				|| bottom > max_coordinate
			{
				return Err(Error::invalid_argument("CvFrame: invalid bounding box"));
			}
			let bx = i64::from(bbox.x as i32);
			let by = i64::from(bbox.y as i32);
			let bw = i64::from(bbox.w as i32);
			let bh = i64::from(bbox.h as i32);
			if bw <= 0 || bh <= 0 {
				return Err(Error::invalid_argument(
					"CvFrame: bounding box has no whole-pixel extent",
				));
			}
			draw_rect(&mut pixels, w, h, bx, by, bw, bh, thickness, [r, g, b, a]);

			if config.show_labels || config.show_scores {
				let percent = bbox.score * 100.0;
				let label_text = if bbox.label.is_empty() {
					"obj"
				} else {
					&bbox.label
				};
				let text = match (config.show_labels, config.show_scores) {
					(true, true) => format!("{label_text} {percent:.0}%"),
					(true, false) => label_text.to_string(),
					(false, true) => format!("{percent:.0}%"),
					(false, false) => String::new(),
				};
				let label_scale = config.label_scale.max(1);
				let label_width = text.len() as i64 * 6 * i64::from(label_scale) + 4;
				let label_height = 7 * i64::from(label_scale) + 4;
				let lx = bx + 2;
				let mut ly = by - label_height;
				if ly < 0 {
					ly = by + 2;
				}
				fill_rect(
					&mut pixels,
					w,
					h,
					lx - 2,
					ly - 2,
					label_width,
					label_height,
					[0, 0, 0, 192],
				);
				draw_string(&mut pixels, w, h, lx, ly, &text, label_scale, [r, g, b]);
			}
		}
	}

	texture_from_rgba8(engine, &pixels, w as usize, h as usize)
}
