//! Backend-neutral VP9 frame parsing.

use crate::{Error, Result};

const FRAME_MARKER: u32 = 2;
const FRAME_SYNC_CODE: u32 = 0x49_83_42;
const MAX_PROBABILITY: u8 = 255;
const MIN_TILE_WIDTH_B64: u32 = 4;
const MAX_TILE_WIDTH_B64: u32 = 64;
const REFERENCE_BUFFERS: usize = 8;
const REFERENCES_PER_FRAME: usize = 3;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Vp9FrameType {
	#[default]
	Key,
	Inter,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Vp9InterpolationFilter {
	EightTapSmooth,
	EightTap,
	EightTapSharp,
	Bilinear,
	#[default]
	Switchable,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Vp9ColorConfig {
	pub bit_depth: u8,
	pub subsampling_x: bool,
	pub subsampling_y: bool,
	pub full_range: bool,
	pub color_space: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Vp9LoopFilter {
	pub level: u8,
	pub sharpness: u8,
	pub delta_enabled: bool,
	pub delta_update: bool,
	pub update_reference_delta: u8,
	pub reference_deltas: [i8; 4],
	pub update_mode_delta: u8,
	pub mode_deltas: [i8; 2],
}

impl Default for Vp9LoopFilter {
	fn default() -> Self {
		Self {
			level: 0,
			sharpness: 0,
			delta_enabled: false,
			delta_update: false,
			update_reference_delta: 0,
			reference_deltas: [1, 0, -1, -1],
			update_mode_delta: 0,
			mode_deltas: [0; 2],
		}
	}
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Vp9Segmentation {
	pub update_map: bool,
	pub temporal_update: bool,
	pub update_data: bool,
	pub absolute_or_delta_update: bool,
	pub tree_probabilities: [u8; 7],
	pub prediction_probabilities: [u8; 3],
	pub feature_enabled: [u8; 8],
	pub feature_data: [[i16; 4]; 8],
}

impl Default for Vp9Segmentation {
	fn default() -> Self {
		Self {
			update_map: false,
			temporal_update: false,
			update_data: false,
			absolute_or_delta_update: false,
			tree_probabilities: [MAX_PROBABILITY; 7],
			prediction_probabilities: [MAX_PROBABILITY; 3],
			feature_enabled: [0; 8],
			feature_data: [[0; 4]; 8],
		}
	}
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Vp9Picture {
	pub frame_offset: usize,
	pub frame_size: usize,
	pub timestamp: u64,
	pub show_existing_frame: bool,
	pub frame_to_show_map_index: u8,
	pub frame_width: u32,
	pub frame_height: u32,
	pub render_width: u32,
	pub render_height: u32,
	pub profile: u8,
	pub frame_type: Vp9FrameType,
	pub show_frame: bool,
	pub error_resilient_mode: bool,
	pub intra_only: bool,
	pub allow_high_precision_motion_vectors: bool,
	pub refresh_frame_context: bool,
	pub frame_parallel_decoding_mode: bool,
	pub segmentation_enabled: bool,
	pub use_previous_frame_motion_vectors: bool,
	pub frame_context_index: u8,
	pub reset_frame_context: u8,
	pub refresh_frame_flags: u8,
	pub reference_frame_sign_bias_mask: u8,
	pub interpolation_filter: Vp9InterpolationFilter,
	pub base_q_index: u8,
	pub delta_q_y_dc: i8,
	pub delta_q_uv_dc: i8,
	pub delta_q_uv_ac: i8,
	pub tile_columns_log2: u8,
	pub tile_rows_log2: u8,
	pub color: Vp9ColorConfig,
	pub loop_filter: Vp9LoopFilter,
	pub segmentation: Vp9Segmentation,
	pub reference_frame_indices: [u8; REFERENCES_PER_FRAME],
	pub frame_is_intra: bool,
	pub uncompressed_header_offset: u32,
	pub compressed_header_offset: u32,
	pub tiles_offset: u32,
	pub compressed_header_size: u32,
	pub tile_count: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct Vp9Parser {
	loop_filter_reference_deltas: [i8; 4],
	loop_filter_mode_deltas: [i8; 2],
	last_frame_width: u32,
	last_frame_height: u32,
	last_show_frame: bool,
	reference_widths: [u32; REFERENCE_BUFFERS],
	reference_heights: [u32; REFERENCE_BUFFERS],
	color: Option<Vp9ColorConfig>,
	segmentation_feature_enabled: [u8; 8],
	segmentation_feature_data: [[i16; 4]; 8],
}

struct FramePrefix {
	frame_type: Vp9FrameType,
	show_frame: bool,
	error_resilient_mode: bool,
	intra_only: bool,
	frame_is_intra: bool,
	frame_width: u32,
	frame_height: u32,
	render_width: u32,
	render_height: u32,
	color: Vp9ColorConfig,
	reset_frame_context: u8,
	refresh_frame_flags: u8,
	reference_frame_indices: [u8; REFERENCES_PER_FRAME],
	reference_frame_sign_bias_mask: u8,
	allow_high_precision_motion_vectors: bool,
	interpolation_filter: Vp9InterpolationFilter,
}

impl Default for Vp9Parser {
	fn default() -> Self {
		Self {
			loop_filter_reference_deltas: [1, 0, -1, -1],
			loop_filter_mode_deltas: [0; 2],
			last_frame_width: 0,
			last_frame_height: 0,
			last_show_frame: false,
			reference_widths: [0; REFERENCE_BUFFERS],
			reference_heights: [0; REFERENCE_BUFFERS],
			color: None,
			segmentation_feature_enabled: [0; 8],
			segmentation_feature_data: [[0; 4]; 8],
		}
	}
}

impl Vp9Picture {
	#[allow(clippy::too_many_arguments)]
	fn show_existing(
		frame_offset: usize,
		frame_size: usize,
		timestamp: u64,
		profile: u8,
		map_index: u8,
		width: u32,
		height: u32,
		color: Vp9ColorConfig,
	) -> Self {
		Self {
			frame_offset,
			frame_size,
			timestamp,
			show_existing_frame: true,
			frame_to_show_map_index: map_index,
			frame_width: width,
			frame_height: height,
			render_width: width,
			render_height: height,
			profile,
			frame_type: Vp9FrameType::Inter,
			show_frame: true,
			error_resilient_mode: false,
			intra_only: false,
			allow_high_precision_motion_vectors: false,
			refresh_frame_context: false,
			frame_parallel_decoding_mode: false,
			segmentation_enabled: false,
			use_previous_frame_motion_vectors: false,
			frame_context_index: 0,
			reset_frame_context: 0,
			refresh_frame_flags: 0,
			reference_frame_sign_bias_mask: 0,
			interpolation_filter: Vp9InterpolationFilter::Switchable,
			base_q_index: 0,
			delta_q_y_dc: 0,
			delta_q_uv_dc: 0,
			delta_q_uv_ac: 0,
			tile_columns_log2: 0,
			tile_rows_log2: 0,
			color,
			loop_filter: Vp9LoopFilter::default(),
			segmentation: Vp9Segmentation::default(),
			reference_frame_indices: [0; REFERENCES_PER_FRAME],
			frame_is_intra: false,
			uncompressed_header_offset: 0,
			compressed_header_offset: 0,
			tiles_offset: 0,
			compressed_header_size: 0,
			tile_count: 0,
		}
	}
}

impl Vp9Parser {
	pub fn parse_access_unit(&mut self, bytes: &[u8]) -> Result<Vec<Vp9Picture>> {
		let mut next = self.clone();
		let (payload_offset, payload, timestamp) = extract_ivf_frame(bytes)?;
		let frame_ranges = split_superframe(payload)?;
		let mut pictures = Vec::new();
		pictures
			.try_reserve_exact(frame_ranges.len())
			.map_err(|_| Error::resource_exhausted("VP9 picture allocation failed"))?;
		for (relative_offset, size) in frame_ranges {
			let start = payload_offset
				.checked_add(relative_offset)
				.ok_or_else(|| Error::out_of_range("VP9 frame offset overflowed"))?;
			let end = start
				.checked_add(size)
				.ok_or_else(|| Error::out_of_range("VP9 frame range overflowed"))?;
			let frame = bytes
				.get(start..end)
				.ok_or_else(|| Error::data_loss("VP9 frame range exceeds the access unit"))?;
			pictures.push(next.parse_frame(frame, start, timestamp)?);
		}
		*self = next;
		Ok(pictures)
	}
}

#[cfg(test)]
#[allow(
	clippy::items_after_test_module,
	reason = "parser tests exercise private state before the bit-reader implementation"
)]
mod tests {
	use std::path::PathBuf;

	use super::*;
	use crate::video::VideoDemuxer;

	#[test]
	fn splits_complete_superframes_and_rejects_inconsistent_indices() -> Result<()> {
		let marker = 0xc1;
		let bytes = [0xaa, 0xbb, 0xcc, 0xdd, 0xee, marker, 2, 3, marker];
		assert_eq!(split_superframe(&bytes)?, [(0, 2), (2, 3)]);
		let malformed = [0xaa, 0xbb, marker, 1, 2, marker];
		assert!(split_superframe(&malformed).is_err());
		Ok(())
	}

	#[test]
	fn malformed_frame_does_not_mutate_parser_state() {
		let mut parser = Vp9Parser {
			last_frame_width: 640,
			..Default::default()
		};
		parser.reference_widths[3] = 640;
		assert!(parser.parse_access_unit(&[0xff]).is_err());
		assert_eq!(parser.last_frame_width, 640);
		assert_eq!(parser.reference_widths[3], 640);
	}

	#[test]
	fn extracts_one_complete_ivf_frame_with_timestamp() -> Result<()> {
		let mut bytes = vec![0_u8; 45];
		bytes[..4].copy_from_slice(b"DKIF");
		bytes[6..8].copy_from_slice(&32_u16.to_le_bytes());
		bytes[8..12].copy_from_slice(b"VP90");
		bytes[32..36].copy_from_slice(&1_u32.to_le_bytes());
		bytes[36..44].copy_from_slice(&17_u64.to_le_bytes());
		bytes[44] = 0x82;
		let (offset, payload, timestamp) = extract_ivf_frame(&bytes)?;
		assert_eq!(offset, 44);
		assert_eq!(payload, [0x82]);
		assert_eq!(timestamp, 17);
		Ok(())
	}

	#[test]
	#[ignore = "requires OA donor VP9 fixture"]
	fn donor_packets_have_complete_bounded_picture_inventory() -> Result<()> {
		let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
			.with_file_name("oa")
			.join("sdk/asset/video/clip/shibuya_720p_30fps_vp9_profile0_8bit_420.mp4");
		let mut demuxer = VideoDemuxer::open(path)?;
		let mut parser = Vp9Parser::default();
		let mut coded = 0_usize;
		let mut shown = 0_usize;
		let mut hidden = 0_usize;
		let mut existing = 0_usize;
		let mut packet_index = 0_usize;
		while let Some(packet) = demuxer.read_next_packet()? {
			let pictures = parser
				.parse_access_unit(packet.data())
				.map_err(|source| Error::data_loss(format!("VP9 donor packet {packet_index}: {source}")))?;
			for picture in pictures {
				assert_eq!(picture.profile, 0);
				assert_eq!((picture.frame_width, picture.frame_height), (1280, 720));
				assert_eq!(picture.color.bit_depth, 8);
				assert!(picture.color.subsampling_x && picture.color.subsampling_y);
				let end = picture
					.frame_offset
					.checked_add(picture.frame_size)
					.expect("VP9 picture range must not overflow");
				assert!(end <= packet.data().len());
				if picture.show_existing_frame {
					existing += 1;
					continue;
				}
				coded += 1;
				assert!(usize::try_from(picture.tiles_offset).unwrap() <= picture.frame_size);
				assert!(picture.tile_count > 0);
				if picture.show_frame {
					shown += 1;
				} else {
					hidden += 1;
				}
			}
			if packet_index == 0 {
				let shown = parser.parse_access_unit(&[0x88, 0])?;
				assert_eq!(shown.len(), 1);
				assert!(shown[0].show_existing_frame);
				assert_eq!((shown[0].frame_width, shown[0].frame_height), (1280, 720));
			}
			packet_index += 1;
		}
		assert_eq!(coded + existing, shown + hidden + existing);
		assert_eq!(shown + existing, 60);
		assert!(coded >= shown);
		Ok(())
	}
}

struct BitReader<'a> {
	data: &'a [u8],
	bit_offset: usize,
	bit_len: usize,
}

impl<'a> BitReader<'a> {
	fn new(data: &'a [u8]) -> Result<Self> {
		let bit_len = data
			.len()
			.checked_mul(8)
			.ok_or_else(|| Error::out_of_range("VP9 payload bit length overflowed"))?;
		Ok(Self {
			data,
			bit_offset: 0,
			bit_len,
		})
	}

	fn bit(&mut self, field: &str) -> Result<bool> {
		Ok(self.bits(1, field)? != 0)
	}

	fn bits(&mut self, count: u32, field: &str) -> Result<u32> {
		if count > 32 {
			return Err(Error::internal("VP9 bit reader request exceeds 32 bits"));
		}
		let count =
			usize::try_from(count).map_err(|_| Error::out_of_range("VP9 bit count exceeds usize"))?;
		let end = self
			.bit_offset
			.checked_add(count)
			.ok_or_else(|| Error::out_of_range("VP9 bit-reader offset overflowed"))?;
		if end > self.bit_len {
			return Err(Error::data_loss(format!("truncated VP9 {field}")));
		}
		let mut value = 0_u32;
		while self.bit_offset < end {
			let byte = self.data[self.bit_offset >> 3];
			value = (value << 1) | u32::from((byte >> (7 - (self.bit_offset & 7))) & 1);
			self.bit_offset += 1;
		}
		Ok(value)
	}

	fn consumed_bytes(&self) -> usize {
		self.bit_offset.div_ceil(8)
	}
}

fn extract_ivf_frame(bytes: &[u8]) -> Result<(usize, &[u8], u64)> {
	if bytes.len() >= 32 && bytes.get(..4) == Some(b"DKIF") && bytes.get(8..12) == Some(b"VP90") {
		let header_size = usize::from(u16::from_le_bytes([bytes[6], bytes[7]]));
		let payload_offset = header_size
			.checked_add(12)
			.ok_or_else(|| Error::out_of_range("VP9 IVF frame offset overflowed"))?;
		if header_size < 32 || payload_offset > bytes.len() {
			return Err(Error::data_loss("invalid VP9 IVF header size"));
		}
		let size = usize::try_from(u32::from_le_bytes([
			bytes[header_size],
			bytes[header_size + 1],
			bytes[header_size + 2],
			bytes[header_size + 3],
		]))
		.map_err(|_| Error::out_of_range("VP9 IVF frame size exceeds usize"))?;
		let end = payload_offset
			.checked_add(size)
			.ok_or_else(|| Error::out_of_range("VP9 IVF frame range overflowed"))?;
		let payload = bytes
			.get(payload_offset..end)
			.ok_or_else(|| Error::data_loss("VP9 IVF frame exceeds the input"))?;
		let timestamp = u64::from_le_bytes(
			bytes[header_size + 4..header_size + 12]
				.try_into()
				.map_err(|_| Error::internal("VP9 IVF timestamp range is invalid"))?,
		);
		return Ok((payload_offset, payload, timestamp));
	}
	if bytes.is_empty() {
		return Err(Error::invalid_argument("VP9 access unit is empty"));
	}
	Ok((0, bytes, 0))
}

fn split_superframe(bytes: &[u8]) -> Result<Vec<(usize, usize)>> {
	let marker = *bytes
		.last()
		.ok_or_else(|| Error::invalid_argument("VP9 frame is empty"))?;
	if marker & 0xe0 != 0xc0 {
		return Ok(vec![(0, bytes.len())]);
	}
	let frame_count = usize::from(marker & 7) + 1;
	let magnitude = usize::from((marker >> 3) & 3) + 1;
	let index_size = 2_usize
		.checked_add(
			magnitude
				.checked_mul(frame_count)
				.ok_or_else(|| Error::out_of_range("VP9 superframe index overflowed"))?,
		)
		.ok_or_else(|| Error::out_of_range("VP9 superframe index overflowed"))?;
	if index_size > bytes.len() || bytes[bytes.len() - index_size] != marker {
		return Ok(vec![(0, bytes.len())]);
	}
	parse_superframe_index(bytes, marker, frame_count, magnitude, index_size)
}

fn parse_superframe_index(
	bytes: &[u8],
	marker: u8,
	frame_count: usize,
	magnitude: usize,
	index_size: usize,
) -> Result<Vec<(usize, usize)>> {
	let payload_size = bytes.len() - index_size;
	let mut cursor = payload_size + 1;
	let mut offset = 0_usize;
	let mut ranges = Vec::new();
	ranges
		.try_reserve_exact(frame_count)
		.map_err(|_| Error::resource_exhausted("VP9 superframe allocation failed"))?;
	for _ in 0..frame_count {
		let mut size = 0_usize;
		for byte_index in 0..magnitude {
			size |= usize::from(bytes[cursor]) << (byte_index * 8);
			cursor += 1;
		}
		if size == 0 || size > payload_size.saturating_sub(offset) {
			return Err(Error::data_loss("invalid VP9 superframe component size"));
		}
		ranges.push((offset, size));
		offset += size;
	}
	if offset != payload_size || bytes[cursor] != marker {
		return Err(Error::data_loss(
			"VP9 superframe index does not cover its payload",
		));
	}
	Ok(ranges)
}

fn parse_color_config(reader: &mut BitReader<'_>, profile: u8) -> Result<Vp9ColorConfig> {
	let bit_depth = if profile >= 2 {
		if reader.bit("12-bit color-depth flag")? {
			12
		} else {
			10
		}
	} else {
		8
	};
	let color_space = to_u8(reader.bits(3, "color space")?)?;
	if color_space == 7 {
		if !matches!(profile, 1 | 3) {
			return Err(Error::data_loss(
				"VP9 RGB color space requires profile 1 or 3",
			));
		}
		if reader.bit("RGB color-config reserved bit")? {
			return Err(Error::data_loss(
				"invalid VP9 RGB color-config reserved bit",
			));
		}
		return Ok(Vp9ColorConfig {
			bit_depth,
			subsampling_x: false,
			subsampling_y: false,
			full_range: true,
			color_space,
		});
	}
	parse_yuv_color_config(reader, profile, bit_depth, color_space)
}

fn parse_yuv_color_config(
	reader: &mut BitReader<'_>,
	profile: u8,
	bit_depth: u8,
	color_space: u8,
) -> Result<Vp9ColorConfig> {
	let full_range = reader.bit("color-range flag")?;
	let (subsampling_x, subsampling_y) = if matches!(profile, 1 | 3) {
		let x = reader.bit("horizontal chroma subsampling")?;
		let y = reader.bit("vertical chroma subsampling")?;
		if reader.bit("color-config reserved bit")? {
			return Err(Error::data_loss("invalid VP9 color-config reserved bit"));
		}
		(x, y)
	} else {
		(true, true)
	};
	Ok(Vp9ColorConfig {
		bit_depth,
		subsampling_x,
		subsampling_y,
		full_range,
		color_space,
	})
}

fn check_sync_code(reader: &mut BitReader<'_>, field: &str) -> Result<()> {
	if reader.bits(24, field)? != FRAME_SYNC_CODE {
		return Err(Error::data_loss(format!("invalid VP9 {field}")));
	}
	Ok(())
}

fn parse_frame_and_render_size(reader: &mut BitReader<'_>) -> Result<(u32, u32, u32, u32)> {
	let width = reader.bits(16, "frame width")? + 1;
	let height = reader.bits(16, "frame height")? + 1;
	let (render_width, render_height) = parse_render_size(reader, width, height)?;
	Ok((width, height, render_width, render_height))
}

fn parse_render_size(
	reader: &mut BitReader<'_>,
	frame_width: u32,
	frame_height: u32,
) -> Result<(u32, u32)> {
	if reader.bit("render-size-different flag")? {
		Ok((
			reader.bits(16, "render width")? + 1,
			reader.bits(16, "render height")? + 1,
		))
	} else {
		Ok((frame_width, frame_height))
	}
}

fn parse_interpolation_filter(reader: &mut BitReader<'_>) -> Result<Vp9InterpolationFilter> {
	if reader.bit("switchable interpolation-filter flag")? {
		return Ok(Vp9InterpolationFilter::Switchable);
	}
	match reader.bits(2, "interpolation filter")? {
		0 => Ok(Vp9InterpolationFilter::EightTapSmooth),
		1 => Ok(Vp9InterpolationFilter::EightTap),
		2 => Ok(Vp9InterpolationFilter::EightTapSharp),
		3 => Ok(Vp9InterpolationFilter::Bilinear),
		_ => Err(Error::internal("unreachable VP9 interpolation filter")),
	}
}

fn read_delta_q(reader: &mut BitReader<'_>) -> Result<i8> {
	if !reader.bit("delta-quantizer presence flag")? {
		return Ok(0);
	}
	read_signed(reader, 4, "delta quantizer")
}

fn read_signed(reader: &mut BitReader<'_>, bits: u32, field: &str) -> Result<i8> {
	let magnitude = i8::try_from(reader.bits(bits, field)?)
		.map_err(|_| Error::out_of_range(format!("VP9 {field} exceeds i8")))?;
	Ok(if reader.bit(&format!("{field} sign"))? {
		-magnitude
	} else {
		magnitude
	})
}

fn parse_tile_layout(reader: &mut BitReader<'_>, frame_width: u32) -> Result<(u8, u8)> {
	let mi_columns = frame_width.div_ceil(8);
	let superblock_columns = mi_columns.div_ceil(8);
	let mut minimum_columns_log2 = 0_u32;
	while MAX_TILE_WIDTH_B64
		.checked_shl(minimum_columns_log2)
		.is_some_and(|width| width < superblock_columns)
	{
		minimum_columns_log2 += 1;
	}
	let mut maximum_columns_log2 = 1_u32;
	while superblock_columns >> maximum_columns_log2 >= MIN_TILE_WIDTH_B64 {
		maximum_columns_log2 += 1;
	}
	maximum_columns_log2 -= 1;

	let mut columns_log2 = minimum_columns_log2;
	while columns_log2 < maximum_columns_log2 && reader.bit("increment tile-column count")? {
		columns_log2 += 1;
	}
	let mut rows_log2 = u32::from(reader.bit("multiple tile rows")?);
	if rows_log2 != 0 {
		rows_log2 += u32::from(reader.bit("four tile rows")?);
	}
	Ok((to_u8(columns_log2)?, to_u8(rows_log2)?))
}

fn parse_segmentation_probabilities(
	reader: &mut BitReader<'_>,
	segmentation: &mut Vp9Segmentation,
) -> Result<()> {
	for probability in &mut segmentation.tree_probabilities {
		*probability = if reader.bit("segmentation tree-probability update flag")? {
			to_u8(reader.bits(8, "segmentation tree probability")?)?
		} else {
			MAX_PROBABILITY
		};
	}
	segmentation.temporal_update = reader.bit("segmentation temporal-update flag")?;
	for probability in &mut segmentation.prediction_probabilities {
		*probability = if segmentation.temporal_update
			&& reader.bit("segmentation prediction-probability update flag")?
		{
			to_u8(reader.bits(8, "segmentation prediction probability")?)?
		} else {
			MAX_PROBABILITY
		};
	}
	Ok(())
}

fn parse_segmentation_features(
	reader: &mut BitReader<'_>,
	segmentation: &mut Vp9Segmentation,
) -> Result<()> {
	const FEATURE_BITS: [u32; 4] = [8, 6, 2, 0];
	const FEATURE_SIGNED: [bool; 4] = [true, true, false, false];
	segmentation.feature_enabled = [0; 8];
	segmentation.feature_data = [[0; 4]; 8];
	for segment in 0..8 {
		for feature in 0..4 {
			if !reader.bit("segmentation feature-enabled flag")? {
				continue;
			}
			segmentation.feature_enabled[segment] |= 1 << feature;
			let magnitude =
				i16::try_from(reader.bits(FEATURE_BITS[feature], "segmentation feature data")?)
					.map_err(|_| Error::out_of_range("VP9 segmentation feature exceeds i16"))?;
			segmentation.feature_data[segment][feature] =
				if FEATURE_SIGNED[feature] && reader.bit("segmentation feature sign")? {
					-magnitude
				} else {
					magnitude
				};
		}
	}
	Ok(())
}

fn to_u8(value: u32) -> Result<u8> {
	u8::try_from(value).map_err(|_| Error::out_of_range("VP9 value exceeds u8"))
}

fn to_u32(value: usize) -> Result<u32> {
	u32::try_from(value).map_err(|_| Error::out_of_range("VP9 offset exceeds u32"))
}

impl Vp9Parser {
	fn parse_frame(
		&mut self,
		data: &[u8],
		frame_offset: usize,
		timestamp: u64,
	) -> Result<Vp9Picture> {
		if data.len() < 2 {
			return Err(Error::data_loss("VP9 frame is too small"));
		}
		let mut reader = BitReader::new(data)?;
		if reader.bits(2, "frame marker")? != FRAME_MARKER {
			return Err(Error::data_loss("invalid VP9 frame marker"));
		}
		let profile =
			to_u8(reader.bits(1, "profile low bit")? | (reader.bits(1, "profile high bit")? << 1))?;
		self.parse_profiled_frame(data, frame_offset, timestamp, profile, reader)
	}

	fn parse_profiled_frame(
		&mut self,
		data: &[u8],
		frame_offset: usize,
		timestamp: u64,
		profile: u8,
		mut reader: BitReader<'_>,
	) -> Result<Vp9Picture> {
		if profile == 3 && reader.bit("profile reserved bit")? {
			return Err(Error::data_loss("invalid VP9 profile 3 reserved bit"));
		}
		if !reader.bit("show-existing flag")? {
			return self.parse_coded_frame(data, frame_offset, timestamp, profile, reader);
		}
		let map_index = to_u8(reader.bits(3, "show-existing map index")?)?;
		let index = usize::from(map_index);
		let width = self.reference_widths[index];
		let height = self.reference_heights[index];
		if width == 0 || height == 0 {
			return Err(Error::failed_precondition(
				"VP9 show-existing frame references an uninitialized buffer",
			));
		}
		Ok(Vp9Picture::show_existing(
			frame_offset,
			data.len(),
			timestamp,
			profile,
			map_index,
			width,
			height,
			self.color.unwrap_or_default(),
		))
	}

	fn parse_coded_frame(
		&mut self,
		data: &[u8],
		frame_offset: usize,
		timestamp: u64,
		profile: u8,
		mut reader: BitReader<'_>,
	) -> Result<Vp9Picture> {
		let prefix = self.parse_frame_prefix(&mut reader, profile)?;
		let use_previous_frame_motion_vectors = self.last_frame_width == prefix.frame_width
			&& self.last_frame_height == prefix.frame_height
			&& self.last_show_frame
			&& !prefix.error_resilient_mode
			&& !prefix.frame_is_intra;
		self.last_frame_width = prefix.frame_width;
		self.last_frame_height = prefix.frame_height;
		self.last_show_frame = prefix.show_frame;
		let (refresh_frame_context, frame_parallel_decoding_mode) = if prefix.error_resilient_mode {
			(false, true)
		} else {
			(
				reader.bit("refresh-frame-context flag")?,
				reader.bit("parallel-decoding-mode flag")?,
			)
		};
		let mut frame_context_index = to_u8(reader.bits(2, "frame-context index")?)?;
		if prefix.frame_is_intra || prefix.error_resilient_mode {
			frame_context_index = 0;
			self.segmentation_feature_enabled = [0; 8];
			self.segmentation_feature_data = [[0; 4]; 8];
			self.loop_filter_reference_deltas = [1, 0, -1, -1];
			self.loop_filter_mode_deltas = [0; 2];
		}
		let loop_filter = self.parse_loop_filter(&mut reader)?;
		let base_q_index = to_u8(reader.bits(8, "base quantizer")?)?;
		let delta_q_y_dc = read_delta_q(&mut reader)?;
		let delta_q_uv_dc = read_delta_q(&mut reader)?;
		let delta_q_uv_ac = read_delta_q(&mut reader)?;
		let (segmentation_enabled, segmentation) = self.parse_segmentation(&mut reader)?;
		let (tile_columns_log2, tile_rows_log2) = parse_tile_layout(&mut reader, prefix.frame_width)?;
		let compressed_header_size = reader.bits(16, "compressed-header size")?;
		let compressed_header_offset = to_u32(reader.consumed_bytes())?;
		let tiles_offset = compressed_header_offset
			.checked_add(compressed_header_size)
			.ok_or_else(|| Error::out_of_range("VP9 tile offset overflowed"))?;
		if usize::try_from(tiles_offset).map_or(true, |offset| offset > data.len()) {
			return Err(Error::data_loss("VP9 headers exceed the frame payload"));
		}
		let tile_count = (1_u32 << tile_columns_log2) * (1_u32 << tile_rows_log2);
		if prefix.frame_is_intra {
			self.color = Some(prefix.color);
		}
		for reference in 0..REFERENCE_BUFFERS {
			if prefix.refresh_frame_flags & (1 << reference) != 0 {
				self.reference_widths[reference] = prefix.frame_width;
				self.reference_heights[reference] = prefix.frame_height;
			}
		}
		Ok(Vp9Picture {
			frame_offset,
			frame_size: data.len(),
			timestamp,
			show_existing_frame: false,
			frame_to_show_map_index: 0,
			frame_width: prefix.frame_width,
			frame_height: prefix.frame_height,
			render_width: prefix.render_width,
			render_height: prefix.render_height,
			profile,
			frame_type: prefix.frame_type,
			show_frame: prefix.show_frame,
			error_resilient_mode: prefix.error_resilient_mode,
			intra_only: prefix.intra_only,
			allow_high_precision_motion_vectors: prefix.allow_high_precision_motion_vectors,
			refresh_frame_context,
			frame_parallel_decoding_mode,
			segmentation_enabled,
			use_previous_frame_motion_vectors,
			frame_context_index,
			reset_frame_context: prefix.reset_frame_context,
			refresh_frame_flags: prefix.refresh_frame_flags,
			reference_frame_sign_bias_mask: prefix.reference_frame_sign_bias_mask,
			interpolation_filter: prefix.interpolation_filter,
			base_q_index,
			delta_q_y_dc,
			delta_q_uv_dc,
			delta_q_uv_ac,
			tile_columns_log2,
			tile_rows_log2,
			color: prefix.color,
			loop_filter,
			segmentation,
			reference_frame_indices: prefix.reference_frame_indices,
			frame_is_intra: prefix.frame_is_intra,
			uncompressed_header_offset: 0,
			compressed_header_offset,
			tiles_offset,
			compressed_header_size,
			tile_count,
		})
	}

	fn parse_frame_prefix(&self, reader: &mut BitReader<'_>, profile: u8) -> Result<FramePrefix> {
		let frame_type = if reader.bit("frame type")? {
			Vp9FrameType::Inter
		} else {
			Vp9FrameType::Key
		};
		let show_frame = reader.bit("show-frame flag")?;
		let error_resilient_mode = reader.bit("error-resilient flag")?;
		if frame_type == Vp9FrameType::Key {
			check_sync_code(reader, "key-frame sync code")?;
			let color = parse_color_config(reader, profile)?;
			let (frame_width, frame_height, render_width, render_height) =
				parse_frame_and_render_size(reader)?;
			return Ok(FramePrefix {
				frame_type,
				show_frame,
				error_resilient_mode,
				intra_only: false,
				frame_is_intra: true,
				frame_width,
				frame_height,
				render_width,
				render_height,
				color,
				reset_frame_context: 0,
				refresh_frame_flags: u8::MAX,
				reference_frame_indices: [0; REFERENCES_PER_FRAME],
				reference_frame_sign_bias_mask: 0,
				allow_high_precision_motion_vectors: false,
				interpolation_filter: Vp9InterpolationFilter::Switchable,
			});
		}
		self.parse_inter_prefix(reader, profile, show_frame, error_resilient_mode)
	}

	fn parse_inter_prefix(
		&self,
		reader: &mut BitReader<'_>,
		profile: u8,
		show_frame: bool,
		error_resilient_mode: bool,
	) -> Result<FramePrefix> {
		let intra_only = !show_frame && reader.bit("intra-only flag")?;
		let reset_frame_context = if error_resilient_mode {
			0
		} else {
			to_u8(reader.bits(2, "reset-frame-context")?)?
		};
		let mut color = self.color.unwrap_or_default();
		let mut reference_frame_indices = [0_u8; REFERENCES_PER_FRAME];
		let mut reference_frame_sign_bias_mask = 0_u8;
		let mut allow_high_precision_motion_vectors = false;
		let mut interpolation_filter = Vp9InterpolationFilter::Switchable;
		let refresh_frame_flags;
		let geometry;
		if intra_only {
			check_sync_code(reader, "intra-only sync code")?;
			color = if profile > 0 {
				parse_color_config(reader, profile)?
			} else {
				Vp9ColorConfig {
					bit_depth: 8,
					subsampling_x: true,
					subsampling_y: true,
					full_range: false,
					color_space: 1,
				}
			};
			refresh_frame_flags = to_u8(reader.bits(8, "refresh-frame flags")?)?;
			geometry = parse_frame_and_render_size(reader)?;
		} else {
			refresh_frame_flags = to_u8(reader.bits(8, "refresh-frame flags")?)?;
			geometry = self.parse_reference_geometry(
				reader,
				&mut reference_frame_indices,
				&mut reference_frame_sign_bias_mask,
			)?;
			allow_high_precision_motion_vectors = reader.bit("high-precision motion-vector flag")?;
			interpolation_filter = parse_interpolation_filter(reader)?;
		}
		Ok(FramePrefix {
			frame_type: Vp9FrameType::Inter,
			show_frame,
			error_resilient_mode,
			intra_only,
			frame_is_intra: intra_only,
			frame_width: geometry.0,
			frame_height: geometry.1,
			render_width: geometry.2,
			render_height: geometry.3,
			color,
			reset_frame_context,
			refresh_frame_flags,
			reference_frame_indices,
			reference_frame_sign_bias_mask,
			allow_high_precision_motion_vectors,
			interpolation_filter,
		})
	}

	fn parse_reference_geometry(
		&self,
		reader: &mut BitReader<'_>,
		reference_frame_indices: &mut [u8; REFERENCES_PER_FRAME],
		reference_frame_sign_bias_mask: &mut u8,
	) -> Result<(u32, u32, u32, u32)> {
		for (name, reference_index) in reference_frame_indices.iter_mut().enumerate() {
			*reference_index = to_u8(reader.bits(3, "reference-frame index")?)?;
			if reader.bit("reference-frame sign bias")? {
				*reference_frame_sign_bias_mask |= 1 << (name + 1);
			}
		}
		for reference_index in *reference_frame_indices {
			if !reader.bit("reference-size match")? {
				continue;
			}
			let index = usize::from(reference_index);
			let width = self.reference_widths[index];
			let height = self.reference_heights[index];
			if width == 0 || height == 0 {
				return Err(Error::failed_precondition(
					"VP9 frame derives size from an uninitialized reference",
				));
			}
			let (render_width, render_height) = parse_render_size(reader, width, height)?;
			return Ok((width, height, render_width, render_height));
		}
		parse_frame_and_render_size(reader)
	}

	fn parse_loop_filter(&mut self, reader: &mut BitReader<'_>) -> Result<Vp9LoopFilter> {
		let level = to_u8(reader.bits(6, "loop-filter level")?)?;
		let sharpness = to_u8(reader.bits(3, "loop-filter sharpness")?)?;
		let delta_enabled = reader.bit("loop-filter delta-enabled flag")?;
		let mut delta_update = false;
		let mut update_reference_delta = 0_u8;
		let mut update_mode_delta = 0_u8;
		if delta_enabled {
			delta_update = reader.bit("loop-filter delta-update flag")?;
			if delta_update {
				for index in 0..4 {
					if reader.bit("loop-filter reference-delta update flag")? {
						update_reference_delta |= 1 << index;
						self.loop_filter_reference_deltas[index] =
							read_signed(reader, 6, "loop-filter reference delta")?;
					}
				}
				for index in 0..2 {
					if reader.bit("loop-filter mode-delta update flag")? {
						update_mode_delta |= 1 << index;
						self.loop_filter_mode_deltas[index] = read_signed(reader, 6, "loop-filter mode delta")?;
					}
				}
			}
		}
		Ok(Vp9LoopFilter {
			level,
			sharpness,
			delta_enabled,
			delta_update,
			update_reference_delta,
			reference_deltas: self.loop_filter_reference_deltas,
			update_mode_delta,
			mode_deltas: self.loop_filter_mode_deltas,
		})
	}

	fn parse_segmentation(&mut self, reader: &mut BitReader<'_>) -> Result<(bool, Vp9Segmentation)> {
		let enabled = reader.bit("segmentation-enabled flag")?;
		let mut segmentation = Vp9Segmentation {
			feature_enabled: self.segmentation_feature_enabled,
			feature_data: self.segmentation_feature_data,
			..Vp9Segmentation::default()
		};
		if !enabled {
			return Ok((false, segmentation));
		}
		segmentation.update_map = reader.bit("segmentation map-update flag")?;
		if segmentation.update_map {
			parse_segmentation_probabilities(reader, &mut segmentation)?;
		}
		segmentation.update_data = reader.bit("segmentation data-update flag")?;
		if segmentation.update_data {
			segmentation.absolute_or_delta_update = reader.bit("segmentation absolute-value flag")?;
			parse_segmentation_features(reader, &mut segmentation)?;
			self.segmentation_feature_enabled = segmentation.feature_enabled;
			self.segmentation_feature_data = segmentation.feature_data;
		}
		Ok((true, segmentation))
	}
}
