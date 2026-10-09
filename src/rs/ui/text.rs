//! Embedded hinted text atlas and OpenType shaping.
//!
//! Donor: `oa/ui/text.h` and `oa/ui/text.cpp`. Rasterization remains an
//! offline build input; runtime shaping uses the same embedded face bytes.

use std::sync::OnceLock;

use crate::{
	Color, Error, Result,
	runtime::{EngineHandle, Storage},
};

const COVERAGE: &[u8] = include_bytes!("generated/coverage.bin");
const GLYPHS: &[u8] = include_bytes!("generated/glyphs.bin");
const SUPPORT: &[u8] = include_bytes!("generated/support.bin");
const METADATA: &str = include_str!("generated/metadata.json");
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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GlyphInfo {
	pub codepoint: u32,
	pub atlas_x: f32,
	pub atlas_y: f32,
	pub atlas_w: f32,
	pub atlas_h: f32,
	pub bearing_x: f32,
	pub bearing_y: f32,
	pub advance: f32,
	pub ink_w: f32,
	pub ink_h: f32,
	pub raster_size: f32,
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextLayoutConfig {
	pub font: FontId,
	pub size: f32,
	pub wrap_width: f32,
	pub monospace: bool,
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
			result.push(PositionedGlyph {
				x: x + position.x_offset as f32 * scale,
				y: origin[1] - position.y_offset as f32 * scale,
				codepoint: ch,
				cluster: info.cluster,
				color,
				font,
			});
			x += position.x_advance as f32 * scale;
		}
		Ok(result)
	}
}
