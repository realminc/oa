//! Completion-safe GPU boxes and labels for image/video views.
//!
//! Geometry is normalized to the source image. Rectangle and glyph records
//! are compact host-upload adapters; native GPU inference/NMS can write the
//! same [`crate::Detection`] record layout without changing the display
//! consumer. A three-slot completion-tracked ring prevents the CPU from
//! overwriting records still consumed by a previous GPU frame.
//!
//! Typical use from a live source:
//! ```ignore
//! let mut overlay = DetectionOverlay::create(&engine, Default::default())?;
//! overlay.update(&items, &atlas)?;
//! // … draw image into destination …
//! overlay.draw(&mut ui, &atlas, destination, clip)?;
//! // after submission:
//! overlay.mark_consumed(&completion)?;
//! ```
//!
//! Donor: `oacpp/source/cpp/include/oa/ui/detectionOverlay.h`.

pub use crate::ui::type_gen::DetectionOverlayConfig;
use crate::{
	Color, Engine, Error, Event, Result,
	ui::{
		FontId, GlyphBuffer, GlyphInstance, PixelRect, TextAtlas, TextLayout, TextLayoutConfig, Ui,
	},
	vision::{Detection, DetectionBuffer},
};

const RING_SIZE: usize = 3;

/// One detection record with its display label.
#[derive(Clone, Debug)]
pub struct DetectionOverlayItem {
	pub detection: Detection,
	pub label: String,
}

impl Default for DetectionOverlayConfig {
	fn default() -> Self {
		Self {
			max_detections: 256,
			max_glyphs: 8192,
			thickness_pixels: 3.0,
			font_size: 18.0,
			label_padding_x: 4.0,
			label_padding_y: 2.0,
			box_color: Color::success(),
			label_text_color: Color {
				r: 0.0,
				g: 0.0,
				b: 0.0,
				a: 1.0,
			},
			show_labels: true,
		}
	}
}

struct OverlaySlot {
	detections: DetectionBuffer,
	glyphs: GlyphBuffer,
}

impl OverlaySlot {
	fn is_ready(&self) -> Result<bool> {
		Ok(self.detections.is_ready()? && self.glyphs.is_ready()?)
	}
}

/// Ring-buffered detection box and label overlay.
///
/// Constructed by [`DetectionOverlay::create`]; records are uploaded through
/// the non-blocking [`DetectionOverlay::update`], composed into a `Ui` frame
/// through [`DetectionOverlay::draw`], and released back for overwrite through
/// [`DetectionOverlay::mark_consumed`].
pub struct DetectionOverlay {
	config: DetectionOverlayConfig,
	slots: Vec<OverlaySlot>,
	active_slot: Option<usize>,
	next_slot: usize,
	count: u32,
}

impl DetectionOverlay {
	/// Allocate the three-slot detection and glyph upload ring.
	///
	/// # Errors
	///
	/// Returns an error when either capacity is zero or allocation fails.
	pub fn create(engine: &Engine, config: DetectionOverlayConfig) -> Result<Self> {
		if config.max_detections == 0 || config.max_glyphs == 0 {
			return Err(Error::invalid_argument(
				"oa::ui::DetectionOverlay: capacities must be non-zero",
			));
		}
		let mut slots = Vec::with_capacity(RING_SIZE);
		for _ in 0..RING_SIZE {
			slots.push(OverlaySlot {
				detections: DetectionBuffer::create_host_upload(engine, config.max_detections)?,
				glyphs: GlyphBuffer::create_host_upload(engine, config.max_glyphs)?,
			});
		}
		Ok(Self {
			config,
			slots,
			active_slot: None,
			next_slot: 0,
			count: 0,
		})
	}

	/// Upload `items` into the first ready ring slot without blocking.
	///
	/// Non-finite detection geometry is rejected and zero-extent boxes are
	/// skipped; remaining fields are clamped to `[0, 1]` and a zero
	/// `color_rgba` receives the configured box color.
	///
	/// # Errors
	///
	/// Returns an error when `items` exceeds `max_detections`, shaped labels
	/// exceed `max_glyphs`, geometry is non-finite, or all three upload slots
	/// are still consumed by in-flight frames.
	pub fn update(&mut self, items: &[DetectionOverlayItem], atlas: &TextAtlas) -> Result<()> {
		if items.len() > usize::try_from(self.config.max_detections).unwrap_or(usize::MAX) {
			return Err(Error::resource_exhausted(
				"oa::ui::DetectionOverlay: detection capacity exceeded",
			));
		}
		let mut selected = None;
		for offset in 0..RING_SIZE {
			let index = (self.next_slot + offset) % RING_SIZE;
			if self.slots[index].is_ready()? {
				selected = Some(index);
				break;
			}
		}
		let Some(index) = selected else {
			return Err(Error::not_ready(
				"oa::ui::DetectionOverlay: all upload slots are in flight",
			));
		};

		let mut detections = Vec::with_capacity(items.len());
		let mut glyphs = Vec::with_capacity(
			usize::try_from(self.config.max_glyphs)
				.unwrap_or(usize::MAX)
				.min(items.len() * 24),
		);
		for item in items {
			if !is_finite(&item.detection) {
				return Err(Error::invalid_argument(
					"oa::ui::DetectionOverlay: non-finite detection geometry",
				));
			}
			let detection = canonicalize(&item.detection, self.config.box_color.to_u32());
			if detection.width <= 0.0 || detection.height <= 0.0 {
				continue;
			}
			detections.push(detection);
			append_label(atlas, &self.config, &detection, &item.label, &mut glyphs)?;
		}
		if glyphs.len() > usize::try_from(self.config.max_glyphs).unwrap_or(usize::MAX) {
			return Err(Error::resource_exhausted(
				"oa::ui::DetectionOverlay: glyph capacity exceeded",
			));
		}

		let slot = &mut self.slots[index];
		slot.detections.upload(&detections)?;
		slot.glyphs.upload(&glyphs)?;
		self.active_slot = Some(index);
		self.next_slot = (index + 1) % RING_SIZE;
		self.count = u32::try_from(detections.len())
			.map_err(|_| Error::invalid_argument("detection count exceeds u32"))?;
		Ok(())
	}

	/// Emit the active slot's box outlines and glyph batch into `ui` through
	/// its shared coverage atlas.
	///
	/// `destination` maps normalized source-image geometry into the compose
	/// image; `clip` bounds rasterization.
	pub fn draw(&self, ui: &mut Ui, destination: PixelRect, clip: PixelRect) -> Result<()> {
		let Some(index) = self.active_slot else {
			return Ok(());
		};
		let slot = &self.slots[index];
		ui.rect_outlines(
			&slot.detections,
			destination,
			clip,
			self.config.box_color,
			self.config.thickness_pixels.round().max(1.0) as u32,
		)?;
		ui.glyphs(&slot.glyphs, destination, clip)
	}

	/// Pin `completion` on the active slot so its records cannot be
	/// overwritten until the consuming frame completes.
	///
	/// # Errors
	///
	/// Returns an error when `completion` does not belong to the owning
	/// engine.
	pub fn mark_consumed(&mut self, completion: &Event) -> Result<()> {
		let Some(index) = self.active_slot else {
			return Ok(());
		};
		let slot = &mut self.slots[index];
		slot.detections.mark_consumed(completion)?;
		slot.glyphs.mark_consumed(completion)
	}

	/// Return the number of records written by the last [`update`](Self::update).
	pub const fn count(&self) -> u32 {
		self.count
	}
}

fn canonicalize(detection: &Detection, fallback_color: u32) -> Detection {
	Detection {
		center_x: detection.center_x.clamp(0.0, 1.0),
		center_y: detection.center_y.clamp(0.0, 1.0),
		width: detection.width.clamp(0.0, 1.0),
		height: detection.height.clamp(0.0, 1.0),
		confidence: detection.confidence.clamp(0.0, 1.0),
		class_id: detection.class_id,
		color_rgba: if detection.color_rgba == 0 {
			fallback_color
		} else {
			detection.color_rgba
		},
		track_id: detection.track_id,
	}
}

fn is_finite(detection: &Detection) -> bool {
	detection.center_x.is_finite()
		&& detection.center_y.is_finite()
		&& detection.width.is_finite()
		&& detection.height.is_finite()
		&& detection.confidence.is_finite()
}

// Emit the label background box followed by one glyph quad per shaped
// codepoint. The background uses zero atlas extents — the draw kernel clamps
// to the atlas' corner texel, which is reserved as opaque coverage.
fn append_label(
	atlas: &TextAtlas,
	config: &DetectionOverlayConfig,
	detection: &Detection,
	label: &str,
	glyphs: &mut Vec<GlyphInstance>,
) -> Result<()> {
	if !config.show_labels || label.is_empty() {
		return Ok(());
	}
	let anchor_x = detection.center_x - detection.width * 0.5;
	let anchor_y = detection.center_y - detection.height * 0.5;
	let layout_config = TextLayoutConfig {
		font: FontId::Sans,
		size: config.font_size,
		..TextLayoutConfig::default()
	};
	let positioned = TextLayout::shape(
		atlas,
		label,
		[0.0, 0.0],
		layout_config,
		config.label_text_color,
	)?;
	if positioned.is_empty() {
		return Ok(());
	}
	let text_width = positioned.last().map_or(0.0, |item| {
		let advance = atlas
			.find_glyph(item.font, item.codepoint, config.font_size)
			.map_or(0.0, |glyph| {
				glyph.advance * (config.font_size / glyph.raster_size)
			});
		item.x + advance
	});
	let mut text_top = 1.0e9_f32;
	let mut text_bottom = -1.0e9_f32;
	for item in &positioned {
		let Some(glyph) = atlas.find_glyph(item.font, item.codepoint, config.font_size) else {
			continue;
		};
		let scale = config.font_size / glyph.raster_size;
		let glyph_top = -glyph.bearing_y * scale;
		text_top = text_top.min(glyph_top);
		text_bottom = text_bottom.max(glyph_top + glyph.ink_h * scale);
	}
	if text_bottom <= text_top {
		return Ok(());
	}

	let label_width = text_width + config.label_padding_x * 2.0;
	let label_height = config.font_size + config.label_padding_y * 2.0;
	let text_height = text_bottom - text_top;
	let baseline = -label_height + (label_height - text_height) * 0.5 - text_top;
	glyphs.push(GlyphInstance {
		anchor_x,
		anchor_y,
		offset_x: 0.0,
		offset_y: -label_height,
		width: label_width,
		height: label_height,
		color: detection.color_rgba,
		..GlyphInstance::default()
	});

	for item in &positioned {
		let Some(glyph) = atlas.find_glyph(item.font, item.codepoint, config.font_size) else {
			continue;
		};
		let scale = config.font_size / glyph.raster_size;
		glyphs.push(GlyphInstance {
			anchor_x,
			anchor_y,
			offset_x: config.label_padding_x + item.x + glyph.bearing_x * scale,
			offset_y: baseline - glyph.bearing_y * scale,
			width: glyph.atlas_w * scale,
			height: glyph.atlas_h * scale,
			atlas_x: glyph.atlas_x as u32,
			atlas_y: glyph.atlas_y as u32,
			atlas_w: glyph.atlas_w as u32,
			atlas_h: glyph.atlas_h as u32,
			color: config.label_text_color.to_u32(),
			flags: 0,
		});
	}
	Ok(())
}
