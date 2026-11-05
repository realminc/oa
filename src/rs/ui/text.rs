//! Embedded hinted text atlas and OpenType shaping.
//!
//! Donor: `oa/ui/text.h` and `oa/ui/text.cpp`. Rasterization remains an
//! offline build input; runtime shaping uses the same embedded face bytes.

use std::sync::OnceLock;

pub use crate::ui::type_gen::{GlyphInfo, TextLayoutConfig};
use crate::{
	Color, Error, Result,
	runtime::{EngineHandle, Storage},
};

const COVERAGE: &[u8] = include_bytes!("../../font/coverage.bin");
const GLYPHS: &[u8] = include_bytes!("../../font/glyphs.bin");
const SUPPORT: &[u8] = include_bytes!("../../font/support.bin");
const METADATA: &str = include_str!("../../font/metadata.json");
const SANS: &[u8] = include_bytes!("../../../sdk/asset/font/IBMPlexSans/IBMPlexSans-Regular.ttf");
const MONO: &[u8] = include_bytes!("../../../sdk/asset/font/IntelOneMono/IntelOneMono-Regular.ttf");
const SANS_SEMIBOLD: &[u8] =
	include_bytes!("../../../sdk/asset/font/IBMPlexSans/IBMPlexSans-SemiBold.ttf");
const GLYPH_RECORD_BYTES: usize = 40;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum FontId {
	#[default]
	Sans = 0,
	Mono = 1,
	SansSemibold = 2,
}

struct AtlasMetadata {
	width: u32,
	height: u32,
	strikes: Vec<u32>,
	codepoints: Vec<u32>,
}

fn metadata() -> &'static AtlasMetadata {
	static META: OnceLock<AtlasMetadata> = OnceLock::new();
	META.get_or_init(|| {
		let value: serde_json::Value = serde_json::from_str(METADATA).expect("generated text metadata");
		AtlasMetadata {
			width: value["width"].as_u64().expect("atlas width") as u32,
			height: value["height"].as_u64().expect("atlas height") as u32,
			strikes: value["strikes"]
				.as_array()
				.expect("atlas strikes")
				.iter()
				.map(|v| v.as_u64().unwrap() as u32)
				.collect(),
			codepoints: value["codepoints"]
				.as_array()
				.expect("atlas codepoints")
				.iter()
				.map(|v| v.as_u64().unwrap() as u32)
				.collect(),
		}
	})
}

pub struct TextAtlas {
	_storage: Storage,
	descriptor: u32,
}

impl TextAtlas {
	pub(super) fn storage(&self) -> &Storage {
		&self._storage
	}

	pub(crate) fn new(engine: &EngineHandle) -> Result<Self> {
		let storage = engine.create_storage(COVERAGE)?;
		let descriptor = engine.storage_descriptor_index(&storage)?;
		Ok(Self {
			_storage: storage,
			descriptor,
		})
	}

	pub fn width(&self) -> u32 {
		metadata().width
	}
	pub fn height(&self) -> u32 {
		metadata().height
	}
	pub(crate) fn descriptor(&self) -> u32 {
		self.descriptor
	}

	pub fn find_glyph(&self, font: FontId, codepoint: char, pixel_size: f32) -> Option<GlyphInfo> {
		if !pixel_size.is_finite() || pixel_size <= 0.0 {
			return None;
		}
		let meta = metadata();
		let cp_index = meta.codepoints.binary_search(&(codepoint as u32)).ok()?;
		let font_index = font as usize;
		if SUPPORT
			.get(font_index * meta.codepoints.len() + cp_index)
			.copied()?
			== 0
		{
			return None;
		}
		let strike_index = meta
			.strikes
			.iter()
			.enumerate()
			.min_by(|(_, a), (_, b)| {
				((**a as f32 - pixel_size).abs()).total_cmp(&((**b as f32 - pixel_size).abs()))
			})?
			.0;
		let index = (font_index * meta.strikes.len() + strike_index) * meta.codepoints.len() + cp_index;
		let record = GLYPHS.get(index * GLYPH_RECORD_BYTES..(index + 1) * GLYPH_RECORD_BYTES)?;
		let u32_at = |offset| u32::from_le_bytes(record[offset..offset + 4].try_into().unwrap());
		let f32_at = |offset| f32::from_le_bytes(record[offset..offset + 4].try_into().unwrap());
		Some(GlyphInfo {
			codepoint: u32_at(8),
			atlas_x: u32_at(12) as f32,
			atlas_y: u32_at(16) as f32,
			atlas_w: u32_at(20) as f32,
			atlas_h: u32_at(24) as f32,
			bearing_x: f32_at(28),
			bearing_y: f32_at(32),
			advance: f32_at(36),
			ink_w: u32_at(20) as f32,
			ink_h: u32_at(24) as f32,
			raster_size: meta.strikes[strike_index] as f32,
		})
	}
}

impl Default for TextLayoutConfig {
	fn default() -> Self {
		Self {
			font: FontId::Sans,
			size: 14.0,
			wrap_width: 0.0,
			monospace: false,
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionedGlyph {
	pub x: f32,
	pub y: f32,
	/// Pen advance applied after this glyph, in scaled pixels.
	pub advance: f32,
	pub codepoint: char,
	pub cluster: u32,
	pub color: Color,
	pub font: FontId,
}

pub struct TextLayout;

impl TextLayout {
	pub fn shape(
		atlas: &TextAtlas,
		text: &str,
		origin: [f32; 2],
		config: TextLayoutConfig,
		color: Color,
	) -> Result<Vec<PositionedGlyph>> {
		if !config.size.is_finite()
			|| config.size <= 0.0
			|| !config.wrap_width.is_finite()
			|| config.wrap_width < 0.0
		{
			return Err(Error::invalid_argument(
				"text size must be positive and wrap width non-negative",
			));
		}
		let font = if config.monospace {
			FontId::Mono
		} else {
			config.font
		};
		let bytes = match font {
			FontId::Sans => SANS,
			FontId::Mono => MONO,
			FontId::SansSemibold => SANS_SEMIBOLD,
		};
		let face = rustybuzz::Face::from_slice(bytes, 0)
			.ok_or_else(|| Error::failed_precondition("embedded UI font is invalid"))?;
		let mut reverse: std::collections::HashMap<u32, char> = std::collections::HashMap::new();
		for &cp in &metadata().codepoints {
			if let Some(ch) = char::from_u32(cp)
				&& let Some(glyph) = face.glyph_index(ch)
			{
				reverse.entry(u32::from(glyph.0)).or_insert(ch);
			}
		}
		let mut buffer = rustybuzz::UnicodeBuffer::new();
		buffer.push_str(text);
		buffer.set_direction(rustybuzz::Direction::LeftToRight);
		let output = rustybuzz::shape(&face, &[], buffer);
		let scale = config.size / face.units_per_em() as f32;
		let mut x = origin[0];
		let mut result = Vec::with_capacity(output.len());
		for (info, position) in output.glyph_infos().iter().zip(output.glyph_positions()) {
			let ch = reverse.get(&info.glyph_id).copied().unwrap_or('?');
			if atlas.find_glyph(font, ch, config.size).is_none() {
				continue;
			}
			let advance = position.x_advance as f32 * scale;
			result.push(PositionedGlyph {
				x: x + position.x_offset as f32 * scale,
				y: origin[1] - position.y_offset as f32 * scale,
				advance,
				codepoint: ch,
				cluster: info.cluster,
				color,
				font,
			});
			x += advance;
		}
		Ok(result)
	}
}

// ── GPU glyph batches ─────────────────────────────────────────────────────────
//
// Donor: `oa/ui/text.h` `GlyphInstance`/`GlyphBuffer`. One source-anchored
// glyph quad: the anchor is normalized to the source image while offsets and
// dimensions are compose-target pixels, so labels remain readable while the
// image camera pans and zooms.

/// One source-anchored glyph quad record, exactly 48 bytes.
///
/// `anchor_x`/`anchor_y` are normalized to the source image;
/// `offset_x`/`offset_y` and `width`/`height` are compose-target pixels.
/// `flags` bit 0 rotates the source coverage counter-clockwise for
/// bottom-to-top text; higher bits are reserved for `Ui`'s deterministic
/// non-overlap batching.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlyphInstance {
	pub anchor_x: f32,
	pub anchor_y: f32,
	pub offset_x: f32,
	pub offset_y: f32,
	pub width: f32,
	pub height: f32,
	pub atlas_x: u32,
	pub atlas_y: u32,
	pub atlas_w: u32,
	pub atlas_h: u32,
	pub color: u32,
	pub flags: u32,
}

const _: () = assert!(std::mem::size_of::<GlyphInstance>() == 48);

impl Default for GlyphInstance {
	fn default() -> Self {
		Self {
			anchor_x: 0.0,
			anchor_y: 0.0,
			offset_x: 0.0,
			offset_y: 0.0,
			width: 0.0,
			height: 0.0,
			atlas_x: 0,
			atlas_y: 0,
			atlas_w: 0,
			atlas_h: 0,
			color: 0xF5F5F5FF,
			flags: 0,
		}
	}
}

/// A fixed-capacity GPU-consumable glyph-instance buffer.
///
/// Records are replaced through [`GlyphBuffer::upload`] while
/// [`GlyphBuffer::mark_consumed`] pins the exact [`crate::Event`] a GPU
/// consumer must complete before the next overwrite. Dropping the buffer
/// never waits; the engine's retirement service keeps submitted resources
/// alive through completion.
pub struct GlyphBuffer {
	engine: EngineHandle,
	storage: crate::Matrix,
	consumer: Option<crate::Event>,
	count: u32,
	capacity: u32,
}

impl GlyphBuffer {
	/// Allocate a host-uploadable buffer for `capacity` glyph records.
	///
	/// # Errors
	///
	/// Returns an error when `capacity` is zero, the capacity byte extent
	/// overflows, or allocation fails.
	pub fn create_host_upload(engine: &crate::Engine, capacity: u32) -> Result<Self> {
		if capacity == 0 {
			return Err(Error::invalid_argument(
				"oa::ui::GlyphBuffer::create_host_upload: capacity must be non-zero",
			));
		}
		let byte_len = usize::try_from(capacity)
			.ok()
			.and_then(|count| count.checked_mul(std::mem::size_of::<GlyphInstance>()))
			.ok_or_else(|| {
				Error::invalid_argument(
					"oa::ui::GlyphBuffer::create_host_upload: capacity byte extent overflows usize",
				)
			})?;
		let storage = crate::Matrix::new_from_slice(engine, [byte_len], &vec![0_u8; byte_len])?;
		Ok(Self {
			engine: engine.handle(),
			storage,
			consumer: None,
			count: 0,
			capacity,
		})
	}

	/// Replace the current records.
	///
	/// # Errors
	///
	/// Returns [`crate::ErrorKind::NotReady`] while a previous render
	/// submission still consumes this slot,
	/// [`crate::ErrorKind::ResourceExhausted`] when `glyphs` exceeds capacity,
	/// or an upload error.
	pub fn upload(&mut self, glyphs: &[GlyphInstance]) -> Result<()> {
		if !self.is_ready()? {
			return Err(Error::not_ready(
				"oa::ui::GlyphBuffer::upload: buffer is still consumed by the GPU",
			));
		}
		if glyphs.len() > usize::try_from(self.capacity).unwrap_or(usize::MAX) {
			return Err(Error::resource_exhausted(
				"oa::ui::GlyphBuffer::upload: glyph count exceeds capacity",
			));
		}
		let used = std::mem::size_of_val(glyphs);
		let mut bytes = vec![0_u8; self.storage.num_elements()];
		if used > 0 {
			// SAFETY: `GlyphInstance` is `repr(C)`, plain data with a
			// statically checked layout; the used prefix never exceeds the
			// byte capacity.
			let source = unsafe { std::slice::from_raw_parts(glyphs.as_ptr().cast::<u8>(), used) };
			bytes[..used].copy_from_slice(source);
		}
		self.storage.write_values::<u8>(&bytes)?;
		self.count = u32::try_from(glyphs.len())
			.map_err(|_| Error::invalid_argument("glyph count exceeds u32"))?;
		self.consumer = None;
		Ok(())
	}

	/// Pin `completion` as the event that must complete before the next
	/// [`upload`](Self::upload).
	///
	/// # Errors
	///
	/// Returns an error when `completion` does not belong to this buffer's
	/// engine or completion cannot be queried.
	pub fn mark_consumed(&mut self, completion: &crate::Event) -> Result<()> {
		if !self.engine.owns_event(completion) {
			return Err(Error::invalid_argument(
				"oa::ui::GlyphBuffer::mark_consumed requires an event from its engine",
			));
		}
		self.consumer = if completion.is_complete()? {
			None
		} else {
			Some(completion.clone())
		};
		Ok(())
	}

	/// Return whether no pinned consumer event is still in flight.
	///
	/// # Errors
	///
	/// Returns an error when consumer completion cannot be queried.
	pub fn is_ready(&self) -> Result<bool> {
		match &self.consumer {
			Some(completion) => completion.is_complete(),
			None => Ok(true),
		}
	}

	/// Return the number of records written by the last [`upload`](Self::upload).
	pub const fn count(&self) -> u32 {
		self.count
	}

	/// Return the record capacity fixed at creation.
	pub const fn capacity(&self) -> u32 {
		self.capacity
	}

	/// Bindless storage-buffer descriptor index for UI draw consumers.
	pub(crate) fn bindless_index(&self) -> Result<u32> {
		self.engine.storage_descriptor_index(self.storage.storage())
	}

	/// The packed record storage retained by UI draw commands.
	pub(crate) const fn storage(&self) -> &crate::Matrix {
		&self.storage
	}

	/// The engine that owns this buffer's storage.
	pub(crate) const fn engine_handle(&self) -> &EngineHandle {
		&self.engine
	}
}
