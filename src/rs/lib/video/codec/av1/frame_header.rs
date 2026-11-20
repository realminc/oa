//! Backend-neutral AV1 uncompressed frame-header parsing.

use crate::{Error, Result};

use super::{Av1BitReader, Av1CodingToolChoice, Av1Obu, Av1ObuType, Av1SequenceHeader};
use crate::video::VideoChromaSubsampling;

const REFERENCE_SLOTS: usize = 8;
const REFERENCES_PER_FRAME: usize = 7;
const SEGMENTS: usize = 8;
const SEGMENT_FEATURES: usize = 8;
const AV1_PRIMARY_REFERENCE_NONE: u8 = 7;
const MAX_TILE_COLUMNS: usize = 64;
const MAX_TILE_ROWS: usize = 64;

/// AV1 uncompressed-header frame identity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Av1FrameType {
	#[default]
	Key,
	Inter,
	IntraOnly,
	Switch,
}

/// AV1 interpolation-filter selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Av1InterpolationFilter {
	EightTap,
	EightTapSmooth,
	EightTapSharp,
	Bilinear,
	#[default]
	Switchable,
}

/// AV1 loop-restoration mode for one plane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Av1RestorationType {
	#[default]
	None,
	Wiener,
	Sgrproj,
	Switchable,
}

/// AV1 transform-mode selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Av1TransformMode {
	#[default]
	Largest,
	Select,
}

/// AV1 global-motion model for one reference of an inter frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Av1GlobalMotionType {
	#[default]
	Identity,
	Translation,
	RotZoom,
	Affine,
}

const GM_PARAMS_IDENTITY: [i32; 6] = [0, 0, 1 << 16, 0, 0, 1 << 16];

/// Logical AV1 reference-map order hints and global-motion state supplied to a
/// stateless frame parse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Av1ReferenceState {
	order_hints: [Option<u8>; REFERENCE_SLOTS],
	global_motion: [[[i32; 6]; REFERENCES_PER_FRAME]; REFERENCE_SLOTS],
}

impl Default for Av1ReferenceState {
	fn default() -> Self {
		Self::new([None; REFERENCE_SLOTS])
	}
}

/// Exact access-unit byte ranges for one AV1 tile group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Av1TileGroup {
	pub first_tile: u16,
	pub tile_offsets: Vec<u32>,
	pub tile_sizes: Vec<u32>,
}

impl Av1ReferenceState {
	/// Construct state from the eight logical AV1 reference-map order hints.
	pub const fn new(order_hints: [Option<u8>; REFERENCE_SLOTS]) -> Self {
		Self {
			order_hints,
			global_motion: [[GM_PARAMS_IDENTITY; REFERENCES_PER_FRAME]; REFERENCE_SLOTS],
		}
	}

	/// Return the order hint currently associated with one logical map slot.
	pub const fn order_hint(&self, slot: usize) -> Option<u8> {
		if slot < REFERENCE_SLOTS {
			self.order_hints[slot]
		} else {
			None
		}
	}

	/// Return the saved global-motion matrices of the picture in one filled
	/// logical map slot, or `None` when the slot has never been refreshed.
	pub fn global_motion(&self, slot: usize) -> Option<&[[i32; 6]; REFERENCES_PER_FRAME]> {
		if slot < REFERENCE_SLOTS && self.order_hints[slot].is_some() {
			Some(&self.global_motion[slot])
		} else {
			None
		}
	}

	/// Apply the refresh mask of one newly decoded picture.
	pub fn refresh(&mut self, frame: &Av1FrameHeader) {
		if frame.show_existing_frame {
			return;
		}
		for (slot, order_hint) in self.order_hints.iter_mut().enumerate() {
			if frame.refresh_frame_flags & (1 << slot) != 0 {
				*order_hint = Some(frame.order_hint);
				self.global_motion[slot] = frame.global_motion_params;
			}
		}
	}
}

/// Parsed AV1 uncompressed frame header required by picture submission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Av1FrameHeader {
	pub header_size: usize,
	pub show_existing_frame: bool,
	pub frame_to_show_map_idx: u8,
	pub frame_type: Av1FrameType,
	pub show_frame: bool,
	pub showable_frame: bool,
	pub error_resilient_mode: bool,
	pub disable_cdf_update: bool,
	pub disable_frame_end_update_cdf: bool,
	pub allow_screen_content_tools: bool,
	pub force_integer_motion_vectors: bool,
	pub frame_size_override: bool,
	pub order_hint: u8,
	pub primary_reference_frame: u8,
	pub refresh_frame_flags: u8,
	pub reference_name_slot_indices: [i8; REFERENCES_PER_FRAME],
	pub reference_order_hints: [u8; REFERENCE_SLOTS],
	pub reference_frame_sign_bias: u8,
	pub frame_references_short_signaling: bool,
	pub use_superres: bool,
	pub coded_denom: u8,
	pub render_and_frame_size_different: bool,
	pub allow_intra_block_copy: bool,
	pub allow_high_precision_motion_vectors: bool,
	pub interpolation_filter: Av1InterpolationFilter,
	pub motion_mode_switchable: bool,
	pub use_reference_frame_motion_vectors: bool,
	pub tile_columns: u8,
	pub tile_rows: u8,
	pub tile_columns_log2: u8,
	pub tile_rows_log2: u8,
	pub uniform_tile_spacing: bool,
	pub tile_column_starts_superblock: [u16; MAX_TILE_COLUMNS + 1],
	pub tile_row_starts_superblock: [u16; MAX_TILE_ROWS + 1],
	pub context_update_tile_id: u16,
	pub tile_size_bytes_minus_1: u8,
	pub base_q_idx: u8,
	pub delta_q_y_dc: i8,
	pub delta_q_u_dc: i8,
	pub delta_q_u_ac: i8,
	pub delta_q_v_dc: i8,
	pub delta_q_v_ac: i8,
	pub using_q_matrix: bool,
	pub different_uv_delta: bool,
	pub qm_y: u8,
	pub qm_u: u8,
	pub qm_v: u8,
	pub segmentation_enabled: bool,
	pub segmentation_update_map: bool,
	pub segmentation_temporal_update: bool,
	pub segmentation_update_data: bool,
	pub segment_feature_enabled: [[bool; SEGMENT_FEATURES]; SEGMENTS],
	pub segment_feature_data: [[i16; SEGMENT_FEATURES]; SEGMENTS],
	pub delta_q_present: bool,
	pub delta_q_resolution: u8,
	pub delta_loop_filter_present: bool,
	pub delta_loop_filter_resolution: u8,
	pub delta_loop_filter_multi: bool,
	pub loop_filter_levels: [u8; 4],
	pub loop_filter_sharpness: u8,
	pub loop_filter_delta_enabled: bool,
	pub loop_filter_delta_update: bool,
	pub loop_filter_update_reference_delta: [bool; 8],
	pub loop_filter_reference_deltas: [i8; 8],
	pub loop_filter_update_mode_delta: [bool; 2],
	pub loop_filter_mode_deltas: [i8; 2],
	pub cdef_damping_minus_3: u8,
	pub cdef_bits: u8,
	pub cdef_y_primary_strength: [u8; 8],
	pub cdef_y_secondary_strength: [u8; 8],
	pub cdef_uv_primary_strength: [u8; 8],
	pub cdef_uv_secondary_strength: [u8; 8],
	pub restoration_types: [Av1RestorationType; 3],
	pub restoration_unit_size_log2_minus_5: [u16; 3],
	pub transform_mode: Av1TransformMode,
	pub reference_select: bool,
	pub skip_mode_present: bool,
	pub skip_mode_frame: [u8; 2],
	pub allow_warped_motion: bool,
	pub reduced_transform_set: bool,
	pub global_motion_types: [Av1GlobalMotionType; REFERENCES_PER_FRAME],
	pub global_motion_params: [[i32; 6]; REFERENCES_PER_FRAME],
	pub apply_grain: bool,
}

impl Default for Av1FrameHeader {
	fn default() -> Self {
		Self {
			header_size: 0,
			show_existing_frame: false,
			frame_to_show_map_idx: 0,
			frame_type: Av1FrameType::Key,
			show_frame: false,
			showable_frame: false,
			error_resilient_mode: false,
			disable_cdf_update: false,
			disable_frame_end_update_cdf: false,
			allow_screen_content_tools: false,
			force_integer_motion_vectors: true,
			frame_size_override: false,
			order_hint: 0,
			primary_reference_frame: 7,
			refresh_frame_flags: 0xff,
			reference_name_slot_indices: [-1; REFERENCES_PER_FRAME],
			reference_order_hints: [0; REFERENCE_SLOTS],
			reference_frame_sign_bias: 0,
			frame_references_short_signaling: false,
			use_superres: false,
			coded_denom: 0,
			render_and_frame_size_different: false,
			allow_intra_block_copy: false,
			allow_high_precision_motion_vectors: false,
			interpolation_filter: Av1InterpolationFilter::Switchable,
			motion_mode_switchable: false,
			use_reference_frame_motion_vectors: false,
			tile_columns: 1,
			tile_rows: 1,
			tile_columns_log2: 0,
			tile_rows_log2: 0,
			uniform_tile_spacing: true,
			tile_column_starts_superblock: [0; MAX_TILE_COLUMNS + 1],
			tile_row_starts_superblock: [0; MAX_TILE_ROWS + 1],
			context_update_tile_id: 0,
			tile_size_bytes_minus_1: 0,
			base_q_idx: 128,
			delta_q_y_dc: 0,
			delta_q_u_dc: 0,
			delta_q_u_ac: 0,
			delta_q_v_dc: 0,
			delta_q_v_ac: 0,
			using_q_matrix: false,
			different_uv_delta: false,
			qm_y: 0,
			qm_u: 0,
			qm_v: 0,
			segmentation_enabled: false,
			segmentation_update_map: false,
			segmentation_temporal_update: false,
			segmentation_update_data: false,
			segment_feature_enabled: [[false; SEGMENT_FEATURES]; SEGMENTS],
			segment_feature_data: [[0; SEGMENT_FEATURES]; SEGMENTS],
			delta_q_present: false,
			delta_q_resolution: 0,
			delta_loop_filter_present: false,
			delta_loop_filter_resolution: 0,
			delta_loop_filter_multi: false,
			loop_filter_levels: [0; 4],
			loop_filter_sharpness: 0,
			loop_filter_delta_enabled: false,
			loop_filter_delta_update: false,
			loop_filter_update_reference_delta: [false; 8],
			loop_filter_reference_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
			loop_filter_update_mode_delta: [false; 2],
			loop_filter_mode_deltas: [0; 2],
			cdef_damping_minus_3: 0,
			cdef_bits: 0,
			cdef_y_primary_strength: [0; 8],
			cdef_y_secondary_strength: [0; 8],
			cdef_uv_primary_strength: [0; 8],
			cdef_uv_secondary_strength: [0; 8],
			restoration_types: [Av1RestorationType::None; 3],
			restoration_unit_size_log2_minus_5: [0; 3],
			transform_mode: Av1TransformMode::Largest,
			reference_select: false,
			skip_mode_present: false,
			skip_mode_frame: [0; 2],
			allow_warped_motion: false,
			reduced_transform_set: false,
			global_motion_types: [Av1GlobalMotionType::Identity; REFERENCES_PER_FRAME],
			global_motion_params: [GM_PARAMS_IDENTITY; REFERENCES_PER_FRAME],
			apply_grain: false,
		}
	}
}

/// Parse one AV1 frame or frame-header OBU payload against explicit reference state.
///
/// # Errors
///
/// Returns an error for truncated or inconsistent syntax and for donor features
/// that are not yet represented: frame IDs, decoder-model removal timing,
/// short reference signaling, non-uniform tiles, and film grain.
pub fn parse_av1_frame_header(
	payload: &[u8],
	sequence: &Av1SequenceHeader,
	references: &Av1ReferenceState,
) -> Result<Av1FrameHeader> {
	if payload.is_empty() {
		return Err(Error::invalid_argument("AV1 frame-header payload is empty"));
	}
	if sequence.frame_id_numbers_present {
		return Err(Error::missing_capability(
			"AV1 frame-ID syntax is not implemented",
		));
	}
	if sequence.decoder_model_info_present {
		return Err(Error::missing_capability(
			"AV1 decoder-model frame timing is not implemented",
		));
	}

	let mut reader = Av1BitReader::new(payload)?;
	let mut output = Av1FrameHeader::default();
	if !sequence.reduced_still_picture_header && reader.bit("show-existing-frame flag")? {
		output.show_existing_frame = true;
		output.frame_to_show_map_idx = to_u8(reader.bits(3, "frame-to-show map index")?)?;
		output.header_size = reader.byte_offset();
		return Ok(output);
	}

	output.frame_type = match reader.bits(2, "frame type")? {
		0 => Av1FrameType::Key,
		1 => Av1FrameType::Inter,
		2 => Av1FrameType::IntraOnly,
		_ => Av1FrameType::Switch,
	};
	output.show_frame = reader.bit("show-frame flag")?;
	if !output.show_frame {
		output.showable_frame = reader.bit("showable-frame flag")?;
	}
	let frame_is_intra = matches!(
		output.frame_type,
		Av1FrameType::Key | Av1FrameType::IntraOnly
	);
	let frame_is_switch = output.frame_type == Av1FrameType::Switch;
	let shown_key = output.frame_type == Av1FrameType::Key && output.show_frame;
	output.error_resilient_mode = if frame_is_switch || shown_key {
		true
	} else {
		reader.bit("error-resilient flag")?
	};
	output.disable_cdf_update = reader.bit("disable-CDF-update flag")?;
	output.allow_screen_content_tools = match sequence.screen_content_tools {
		Av1CodingToolChoice::Disabled => false,
		Av1CodingToolChoice::Enabled => true,
		Av1CodingToolChoice::SelectPerFrame => reader.bit("screen-content frame flag")?,
	};
	output.force_integer_motion_vectors = if !output.allow_screen_content_tools {
		false
	} else {
		match sequence.integer_motion_vectors {
			Av1CodingToolChoice::Disabled => false,
			Av1CodingToolChoice::Enabled => true,
			Av1CodingToolChoice::SelectPerFrame => reader.bit("force-integer-MV flag")?,
		}
	};
	if frame_is_intra {
		output.force_integer_motion_vectors = true;
	}
	output.frame_size_override = if frame_is_switch {
		true
	} else if sequence.reduced_still_picture_header {
		false
	} else {
		reader.bit("frame-size-override flag")?
	};
	if sequence.order_hint_bits > 0 {
		output.order_hint = to_u8(reader.bits(u32::from(sequence.order_hint_bits), "order hint")?)?;
	}
	if !frame_is_intra && !output.error_resilient_mode {
		output.primary_reference_frame = to_u8(reader.bits(3, "primary reference frame")?)?;
	}
	output.refresh_frame_flags = if frame_is_switch || shown_key {
		0xff
	} else {
		to_u8(reader.bits(8, "refresh-frame flags")?)?
	};
	let mut logical_order_hints = [0_u8; REFERENCE_SLOTS];
	for (index, hint) in logical_order_hints.iter_mut().enumerate() {
		*hint = references.order_hint(index).unwrap_or(0);
	}
	if (!frame_is_intra || output.refresh_frame_flags != 0xff)
		&& output.error_resilient_mode
		&& sequence.enable_order_hint
	{
		for hint in &mut logical_order_hints {
			*hint = to_u8(reader.bits(u32::from(sequence.order_hint_bits), "reference order hint")?)?;
		}
	}

	if frame_is_intra {
		read_frame_size(
			&mut reader,
			sequence,
			output.frame_size_override,
			&mut output,
		)?;
		read_render_size(&mut reader, &mut output)?;
		if output.allow_screen_content_tools {
			output.allow_intra_block_copy = reader.bit("allow-intrabc flag")?;
		}
	} else {
		if sequence.enable_order_hint {
			output.frame_references_short_signaling = reader.bit("short-reference-signaling flag")?;
			if output.frame_references_short_signaling {
				return Err(Error::missing_capability(
					"AV1 short reference signaling is not implemented",
				));
			}
		}
		for reference in &mut output.reference_name_slot_indices {
			*reference = to_i8(reader.bits(3, "reference-frame index")?)?;
		}
		output.reference_order_hints[0] = 0;
		for (index, logical) in output.reference_name_slot_indices.iter().enumerate() {
			output.reference_order_hints[index + 1] = logical_order_hints[*logical as usize];
		}
		if output.frame_size_override && !output.error_resilient_mode {
			let mut found_reference = false;
			for _ in 0..REFERENCES_PER_FRAME {
				found_reference = reader.bit("found-reference flag")?;
				if found_reference {
					break;
				}
			}
			if found_reference {
				read_superres(&mut reader, sequence, &mut output)?;
			} else {
				read_frame_size(&mut reader, sequence, true, &mut output)?;
				read_render_size(&mut reader, &mut output)?;
			}
		} else {
			read_frame_size(
				&mut reader,
				sequence,
				output.frame_size_override,
				&mut output,
			)?;
			read_render_size(&mut reader, &mut output)?;
		}
		if !output.force_integer_motion_vectors {
			output.allow_high_precision_motion_vectors = reader.bit("high-precision-MV flag")?;
		}
		output.interpolation_filter = if reader.bit("switchable-interpolation-filter flag")? {
			Av1InterpolationFilter::Switchable
		} else {
			match reader.bits(2, "interpolation filter")? {
				0 => Av1InterpolationFilter::EightTap,
				1 => Av1InterpolationFilter::EightTapSmooth,
				2 => Av1InterpolationFilter::EightTapSharp,
				_ => Av1InterpolationFilter::Bilinear,
			}
		};
		output.motion_mode_switchable = reader.bit("motion-mode-switchable flag")?;
		if !output.error_resilient_mode && sequence.enable_reference_frame_motion_vectors {
			output.use_reference_frame_motion_vectors = reader.bit("use-reference-frame-MVs flag")?;
		}
	}

	output.disable_frame_end_update_cdf =
		if sequence.reduced_still_picture_header || output.disable_cdf_update {
			true
		} else {
			reader.bit("disable-frame-end-CDF-update flag")?
		};
	parse_tile_info(&mut reader, sequence, &mut output)?;
	parse_quantization(&mut reader, sequence, &mut output)?;
	parse_segmentation(&mut reader, &mut output)?;
	parse_delta_quantization(&mut reader, &mut output)?;
	let all_lossless = (0..SEGMENTS).all(|segment| {
		let index = i32::from(output.base_q_idx)
			+ if output.segmentation_enabled && output.segment_feature_enabled[segment][0] {
				i32::from(output.segment_feature_data[segment][0])
			} else {
				0
			};
		index.clamp(0, 255) == 0
			&& output.delta_q_y_dc == 0
			&& output.delta_q_u_dc == 0
			&& output.delta_q_u_ac == 0
			&& output.delta_q_v_dc == 0
			&& output.delta_q_v_ac == 0
	});
	if all_lossless || output.allow_intra_block_copy {
		output.loop_filter_delta_enabled = true;
		output.loop_filter_delta_update = true;
	} else {
		parse_loop_filter(&mut reader, sequence, &mut output)?;
	}
	if !all_lossless && sequence.enable_cdef && !output.allow_intra_block_copy {
		parse_cdef(&mut reader, sequence, &mut output)?;
	}
	let restoration_permitted = sequence.enable_restoration
		&& !output.allow_intra_block_copy
		&& (!all_lossless || output.use_superres);
	if restoration_permitted {
		parse_restoration(&mut reader, sequence, &mut output)?;
	}
	output.transform_mode = if reader.bit("transform-mode flag")? {
		Av1TransformMode::Select
	} else {
		Av1TransformMode::Largest
	};
	if !frame_is_intra {
		output.reference_select = reader.bit("reference-select flag")?;
	}
	parse_skip_mode(
		&mut reader,
		sequence,
		references,
		frame_is_intra,
		&mut output,
	)?;
	if !frame_is_intra && !output.error_resilient_mode && sequence.enable_warped_motion {
		output.allow_warped_motion = reader.bit("allow-warped-motion flag")?;
	}
	output.reduced_transform_set = reader.bit("reduced-transform-set flag")?;
	if !frame_is_intra {
		parse_global_motion(&mut reader, &mut output, references)?;
	}
	if sequence.film_grain_params_present && (output.show_frame || output.showable_frame) {
		output.apply_grain = reader.bit("apply-grain flag")?;
		if output.apply_grain {
			return Err(Error::missing_capability(
				"AV1 film-grain parameters are not implemented",
			));
		}
	}
	reader.byte_align()?;
	output.header_size = reader.byte_offset();
	Ok(output)
}

/// Read one `ns(max)` uniform value in `[0, max)` per the AV1 syntax.
fn read_ns(reader: &mut Av1BitReader<'_>, max: u32, field: &str) -> Result<u32> {
	if max <= 1 {
		return Ok(0);
	}
	let width = u32::BITS - max.leading_zeros();
	let bound = (1_u32 << width) - max;
	let value = reader.bits(width - 1, field)?;
	Ok(if value < bound {
		value
	} else {
		(value << 1) - bound + u32::from(reader.bit(field)?)
	})
}

/// Undo the `inverse_recenter` mapping of `decode_unsigned_subexp_with_ref`.
fn inv_recenter(reference: u32, value: u32) -> u32 {
	if value > 2 * reference {
		value
	} else if value & 1 == 0 {
		(value >> 1) + reference
	} else {
		reference - ((value + 1) >> 1)
	}
}

/// Decode one `decode_unsigned_subexp_with_ref` value where `limit` counts the
/// maximum symbol index (spec `numSyms - 1`).
fn read_subexp_unsigned(
	reader: &mut Av1BitReader<'_>,
	reference: u32,
	limit: u32,
	field: &str,
) -> Result<u32> {
	let mut value = 0_u32;
	let mut index = 0_u32;
	loop {
		let width = if index == 0 { 3 } else { index + 2 };
		if limit < value + 3 * (1_u32 << width) {
			value += read_ns(reader, limit - value + 1, field)?;
			break;
		}
		if !reader.bit(field)? {
			value += reader.bits(width, field)?;
			break;
		}
		value += 1_u32 << width;
		index += 1;
	}
	Ok(if reference * 2 <= limit {
		inv_recenter(reference, value)
	} else {
		limit - inv_recenter(limit - reference, value)
	})
}

/// Decode one `decode_signed_subexp_with_ref` symbol in
/// `[-(1 << bits), 1 << bits]`.
fn read_subexp(reader: &mut Av1BitReader<'_>, reference: i32, bits: u32) -> Result<i32> {
	let range = 1_i64 << bits;
	let unsigned_reference = u32::try_from(i64::from(reference) + range).map_err(|_| {
		Error::out_of_range("AV1 global-motion subexp reference is outside its symbol range")
	})?;
	let decoded = read_subexp_unsigned(
		reader,
		unsigned_reference,
		u32::try_from(2 * range)
			.map_err(|_| Error::out_of_range("AV1 global-motion subexp range exceeds u32"))?,
		"global-motion subexp",
	)?;
	i32::try_from(i64::from(decoded) - range)
		.map_err(|_| Error::out_of_range("AV1 global-motion subexp result exceeds i32"))
}

/// Parse the seven per-reference global-motion models of one inter frame.
fn parse_global_motion(
	reader: &mut Av1BitReader<'_>,
	output: &mut Av1FrameHeader,
	references: &Av1ReferenceState,
) -> Result<()> {
	for index in 0..REFERENCES_PER_FRAME {
		let motion_type = if !reader.bit("global-motion is-global flag")? {
			Av1GlobalMotionType::Identity
		} else if reader.bit("global-motion rot-zoom flag")? {
			Av1GlobalMotionType::RotZoom
		} else if reader.bit("global-motion translation flag")? {
			Av1GlobalMotionType::Translation
		} else {
			Av1GlobalMotionType::Affine
		};
		output.global_motion_types[index] = motion_type;
		if motion_type == Av1GlobalMotionType::Identity {
			continue;
		}
		let previous = if output.primary_reference_frame == 7 {
			GM_PARAMS_IDENTITY
		} else {
			let slot = output.reference_name_slot_indices[usize::from(output.primary_reference_frame)];
			let slot = usize::try_from(slot).unwrap_or(usize::MAX);
			references
				.global_motion(slot)
				.ok_or_else(|| Error::data_loss("AV1 global motion references an unrefreshed DPB slot"))?[index]
		};
		let mut matrix = output.global_motion_params[index];
		let (bits, shift) = if matches!(
			motion_type,
			Av1GlobalMotionType::RotZoom | Av1GlobalMotionType::Affine
		) {
			matrix[2] = (1 << 16) + 2 * read_subexp(reader, (previous[2] - (1 << 16)) >> 1, 12)?;
			matrix[3] = 2 * read_subexp(reader, previous[3] >> 1, 12)?;
			(12_u32, 10_u32)
		} else {
			let precision = u32::from(!output.allow_high_precision_motion_vectors);
			(9 - precision, 13 + precision)
		};
		if motion_type == Av1GlobalMotionType::Affine {
			matrix[4] = 2 * read_subexp(reader, previous[4] >> 1, 12)?;
			matrix[5] = (1 << 16) + 2 * read_subexp(reader, (previous[5] - (1 << 16)) >> 1, 12)?;
		} else {
			matrix[4] = -matrix[3];
			matrix[5] = matrix[2];
		}
		matrix[0] = read_subexp(reader, previous[0] >> shift, bits)? << shift;
		matrix[1] = read_subexp(reader, previous[1] >> shift, bits)? << shift;
		output.global_motion_params[index] = matrix;
	}
	Ok(())
}

/// Parse the tile ranges carried by one frame or tile-group OBU.
///
/// Returned offsets are relative to the access-unit byte slice originally
/// passed to [`super::parse_av1_obus`], matching Vulkan Video's source-buffer
/// offset convention.
///
/// # Errors
///
/// Returns an error for an incompatible OBU type, missing payload, invalid tile
/// count/range/size width, truncated size fields, payload overruns, or offsets
/// that do not fit Vulkan's 32-bit tile arrays.
pub fn parse_av1_tile_group(obu: Av1Obu<'_>, frame: &Av1FrameHeader) -> Result<Av1TileGroup> {
	let header_bytes = match obu.type_() {
		Av1ObuType::Frame => frame.header_size,
		Av1ObuType::TileGroup => 0,
		_ => {
			return Err(Error::invalid_argument(
				"AV1 tile parsing requires a frame or tile-group OBU",
			));
		}
	};
	let payload = obu
		.payload()
		.get(header_bytes..)
		.ok_or_else(|| Error::data_loss("AV1 frame header exceeds its OBU payload"))?;
	if payload.is_empty() {
		return Err(Error::data_loss("AV1 tile group has no tile payload"));
	}
	let base_offset = obu
		.header_offset()
		.checked_add(obu.header_size())
		.and_then(|value| value.checked_add(header_bytes))
		.ok_or_else(|| Error::out_of_range("AV1 tile-group offset overflowed"))?;
	let tile_count = usize::from(frame.tile_columns)
		.checked_mul(usize::from(frame.tile_rows))
		.ok_or_else(|| Error::out_of_range("AV1 tile count overflowed"))?;
	if tile_count == 0 || tile_count > 4096 {
		return Err(Error::data_loss(
			"AV1 tile count is outside standard limits",
		));
	}
	if tile_count == 1 {
		return Ok(Av1TileGroup {
			first_tile: 0,
			tile_offsets: vec![to_u32(base_offset)?],
			tile_sizes: vec![to_u32(payload.len())?],
		});
	}

	let mut reader = Av1BitReader::new(payload)?;
	let has_explicit_range = reader.bit("tile-group range flag")?;
	let tile_number_bits = u32::from(frame.tile_columns_log2)
		.checked_add(u32::from(frame.tile_rows_log2))
		.ok_or_else(|| Error::out_of_range("AV1 tile-number width overflowed"))?;
	let (tile_start, tile_end) = if has_explicit_range {
		let start = usize::try_from(reader.bits(tile_number_bits, "tile-group start")?)
			.map_err(|_| Error::out_of_range("AV1 tile-group start exceeds usize"))?;
		let end = usize::try_from(reader.bits(tile_number_bits, "tile-group end")?)
			.map_err(|_| Error::out_of_range("AV1 tile-group end exceeds usize"))?;
		if start > end || end >= tile_count {
			return Err(Error::data_loss("AV1 tile-group range is invalid"));
		}
		(start, end)
	} else {
		(0, tile_count - 1)
	};
	reader.byte_align()?;
	let mut cursor = reader.byte_offset();
	let tile_size_bytes = usize::from(frame.tile_size_bytes_minus_1) + 1;
	if tile_size_bytes > 4 {
		return Err(Error::data_loss("AV1 tile-size field exceeds four bytes"));
	}
	let range_count = tile_end - tile_start + 1;
	let mut tile_offsets = Vec::new();
	let mut tile_sizes = Vec::new();
	tile_offsets
		.try_reserve_exact(range_count)
		.map_err(|_| Error::resource_exhausted("AV1 tile-offset allocation failed"))?;
	tile_sizes
		.try_reserve_exact(range_count)
		.map_err(|_| Error::resource_exhausted("AV1 tile-size allocation failed"))?;
	for tile in tile_start..=tile_end {
		let tile_size = if tile == tile_end {
			payload
				.len()
				.checked_sub(cursor)
				.ok_or_else(|| Error::data_loss("AV1 tile cursor exceeds payload"))?
		} else {
			let size_end = cursor
				.checked_add(tile_size_bytes)
				.ok_or_else(|| Error::out_of_range("AV1 tile-size range overflowed"))?;
			let size_bytes = payload
				.get(cursor..size_end)
				.ok_or_else(|| Error::data_loss("truncated AV1 tile-size field"))?;
			cursor = size_end;
			usize::try_from(read_little_endian(size_bytes))
				.map_err(|_| Error::out_of_range("AV1 tile size exceeds usize"))?
				.checked_add(1)
				.ok_or_else(|| Error::out_of_range("AV1 tile size overflowed"))?
		};
		if tile_size == 0 {
			return Err(Error::data_loss("AV1 tile payload is empty"));
		}
		let tile_end_offset = cursor
			.checked_add(tile_size)
			.ok_or_else(|| Error::out_of_range("AV1 tile payload range overflowed"))?;
		if tile_end_offset > payload.len() {
			return Err(Error::data_loss("AV1 tile payload exceeds tile group"));
		}
		let tile_offset = base_offset
			.checked_add(cursor)
			.ok_or_else(|| Error::out_of_range("AV1 tile offset overflowed"))?;
		tile_offsets.push(to_u32(tile_offset)?);
		tile_sizes.push(to_u32(tile_size)?);
		cursor = tile_end_offset;
	}
	Ok(Av1TileGroup {
		first_tile: u16::try_from(tile_start)
			.map_err(|_| Error::data_loss("AV1 tile-group start exceeds u16"))?,
		tile_offsets,
		tile_sizes,
	})
}

fn read_superres(
	reader: &mut Av1BitReader<'_>,
	sequence: &Av1SequenceHeader,
	output: &mut Av1FrameHeader,
) -> Result<()> {
	if sequence.enable_superres {
		output.use_superres = reader.bit("super-resolution flag")?;
		if output.use_superres {
			output.coded_denom = to_u8(reader.bits(3, "super-resolution denominator")?)?;
		}
	}
	Ok(())
}

fn read_frame_size(
	reader: &mut Av1BitReader<'_>,
	sequence: &Av1SequenceHeader,
	override_size: bool,
	output: &mut Av1FrameHeader,
) -> Result<()> {
	if override_size {
		reader.skip(
			u32::from(sequence.frame_width_bits_minus_1) + 1,
			"frame width",
		)?;
		reader.skip(
			u32::from(sequence.frame_height_bits_minus_1) + 1,
			"frame height",
		)?;
	}
	read_superres(reader, sequence, output)
}

fn read_render_size(reader: &mut Av1BitReader<'_>, output: &mut Av1FrameHeader) -> Result<()> {
	output.render_and_frame_size_different = reader.bit("render-size flag")?;
	if output.render_and_frame_size_different {
		reader.skip(32, "render size")?;
	}
	Ok(())
}

fn parse_tile_info(
	reader: &mut Av1BitReader<'_>,
	sequence: &Av1SequenceHeader,
	output: &mut Av1FrameHeader,
) -> Result<()> {
	let mi_columns = (sequence.coded_width() + 7) >> 3 << 1;
	let mi_rows = (sequence.coded_height() + 7) >> 3 << 1;
	let superblock_shift = if sequence.use_128x128_superblock {
		5
	} else {
		4
	};
	let superblock_columns = (mi_columns + (1 << superblock_shift) - 1) >> superblock_shift;
	let superblock_rows = (mi_rows + (1 << superblock_shift) - 1) >> superblock_shift;
	let max_tile_width = if sequence.use_128x128_superblock {
		32
	} else {
		64
	};
	let max_tile_area = if sequence.use_128x128_superblock {
		576
	} else {
		2304
	};
	let minimum_column_log2 = ceil_log2_div(superblock_columns, max_tile_width);
	let maximum_column_log2 =
		ceil_log2_div(superblock_columns.min(MAX_TILE_COLUMNS as u32).max(1), 1);
	let maximum_row_log2 = ceil_log2_div(superblock_rows.min(MAX_TILE_ROWS as u32).max(1), 1);
	let minimum_tile_log2 = minimum_column_log2.max(ceil_log2_div(
		superblock_rows * superblock_columns,
		max_tile_area,
	));
	output.uniform_tile_spacing = reader.bit("uniform-tile-spacing flag")?;
	if output.uniform_tile_spacing {
		let mut column_log2 = minimum_column_log2;
		while column_log2 < maximum_column_log2 && reader.bit("tile-column increment")? {
			column_log2 += 1;
		}
		let tile_width = 1 + ((superblock_columns - 1) >> column_log2);
		let mut column = 0_usize;
		let mut start = 0_u32;
		while start < superblock_columns {
			output.tile_column_starts_superblock[column] =
				u16::try_from(start).map_err(|_| Error::data_loss("AV1 tile column exceeds u16"))?;
			column += 1;
			start += tile_width;
		}
		output.tile_columns_log2 = to_u8(column_log2)?;
		output.tile_columns =
			u8::try_from(column).map_err(|_| Error::data_loss("AV1 tile columns exceed u8"))?;
		let mut row_log2 = minimum_tile_log2.saturating_sub(column_log2);
		while row_log2 < maximum_row_log2 && reader.bit("tile-row increment")? {
			row_log2 += 1;
		}
		let tile_height = 1 + ((superblock_rows - 1) >> row_log2);
		let mut row = 0_usize;
		let mut start = 0_u32;
		while start < superblock_rows {
			output.tile_row_starts_superblock[row] =
				u16::try_from(start).map_err(|_| Error::data_loss("AV1 tile row exceeds u16"))?;
			row += 1;
			start += tile_height;
		}
		output.tile_rows_log2 = to_u8(row_log2)?;
		output.tile_rows =
			u8::try_from(row).map_err(|_| Error::data_loss("AV1 tile rows exceed u8"))?;
	} else {
		let mut widest_tile = 0_u32;
		let mut start = 0_u32;
		let mut column = 0_usize;
		while start < superblock_columns && column < MAX_TILE_COLUMNS {
			output.tile_column_starts_superblock[column] =
				u16::try_from(start).map_err(|_| Error::data_loss("AV1 tile column exceeds u16"))?;
			let maximum_width = (superblock_columns - start).min(max_tile_width);
			let width = if maximum_width > 1 {
				1_u32
					.checked_add(reader.uniform(maximum_width, "tile column width")?)
					.ok_or_else(|| Error::out_of_range("AV1 tile column width overflowed"))?
			} else {
				1
			};
			widest_tile = widest_tile.max(width);
			start = start
				.checked_add(width)
				.ok_or_else(|| Error::out_of_range("AV1 tile column start overflowed"))?;
			column += 1;
		}
		if start < superblock_columns {
			return Err(Error::missing_capability(
				"AV1 stream exceeds the supported tile-column limit",
			));
		}
		output.tile_columns =
			u8::try_from(column).map_err(|_| Error::data_loss("AV1 tile columns exceed u8"))?;
		output.tile_columns_log2 = to_u8(ceil_log2_div(u32::from(output.tile_columns), 1))?;
		let mut adjusted_area = superblock_rows * superblock_columns;
		if minimum_tile_log2 > 0 {
			adjusted_area >>= minimum_tile_log2 + 1;
		}
		let max_tile_height = (adjusted_area / widest_tile.max(1)).max(1);
		let mut start = 0_u32;
		let mut row = 0_usize;
		while start < superblock_rows && row < MAX_TILE_ROWS {
			output.tile_row_starts_superblock[row] =
				u16::try_from(start).map_err(|_| Error::data_loss("AV1 tile row exceeds u16"))?;
			let maximum_height = (superblock_rows - start).min(max_tile_height);
			let height = if maximum_height > 1 {
				1_u32
					.checked_add(reader.uniform(maximum_height, "tile row height")?)
					.ok_or_else(|| Error::out_of_range("AV1 tile row height overflowed"))?
			} else {
				1
			};
			start = start
				.checked_add(height)
				.ok_or_else(|| Error::out_of_range("AV1 tile row start overflowed"))?;
			row += 1;
		}
		if start < superblock_rows {
			return Err(Error::missing_capability(
				"AV1 stream exceeds the supported tile-row limit",
			));
		}
		output.tile_rows =
			u8::try_from(row).map_err(|_| Error::data_loss("AV1 tile rows exceed u8"))?;
		output.tile_rows_log2 = to_u8(ceil_log2_div(u32::from(output.tile_rows), 1))?;
	}
	output.tile_column_starts_superblock[usize::from(output.tile_columns)] =
		u16::try_from(superblock_columns)
			.map_err(|_| Error::data_loss("AV1 tile column end exceeds u16"))?;
	output.tile_row_starts_superblock[usize::from(output.tile_rows)] =
		u16::try_from(superblock_rows).map_err(|_| Error::data_loss("AV1 tile row end exceeds u16"))?;
	let tile_id_bits = u32::from(output.tile_columns_log2) + u32::from(output.tile_rows_log2);
	if tile_id_bits > 0 {
		output.context_update_tile_id =
			u16::try_from(reader.bits(tile_id_bits, "context-update tile ID")?)
				.map_err(|_| Error::data_loss("AV1 context-update tile ID exceeds u16"))?;
		if output.context_update_tile_id >= u16::from(output.tile_columns) * u16::from(output.tile_rows)
		{
			return Err(Error::data_loss(
				"AV1 context-update tile ID exceeds the tile count",
			));
		}
		output.tile_size_bytes_minus_1 = to_u8(reader.bits(2, "tile-size byte count")?)?;
	}
	Ok(())
}

fn parse_quantization(
	reader: &mut Av1BitReader<'_>,
	sequence: &Av1SequenceHeader,
	output: &mut Av1FrameHeader,
) -> Result<()> {
	output.base_q_idx = to_u8(reader.bits(8, "base quantizer index")?)?;
	output.delta_q_y_dc = read_delta_q(reader)?;
	if !sequence.color.monochrome {
		output.different_uv_delta =
			sequence.color.separate_uv_delta_q && reader.bit("different-UV-delta flag")?;
		output.delta_q_u_dc = read_delta_q(reader)?;
		output.delta_q_u_ac = read_delta_q(reader)?;
		if output.different_uv_delta {
			output.delta_q_v_dc = read_delta_q(reader)?;
			output.delta_q_v_ac = read_delta_q(reader)?;
		} else {
			output.delta_q_v_dc = output.delta_q_u_dc;
			output.delta_q_v_ac = output.delta_q_u_ac;
		}
	}
	output.using_q_matrix = reader.bit("quantization-matrix flag")?;
	if output.using_q_matrix {
		output.qm_y = to_u8(reader.bits(4, "luma quantization matrix")?)?;
		output.qm_u = to_u8(reader.bits(4, "chroma quantization matrix")?)?;
		output.qm_v = output.qm_u;
	}
	Ok(())
}

fn parse_segmentation(reader: &mut Av1BitReader<'_>, output: &mut Av1FrameHeader) -> Result<()> {
	output.segmentation_enabled = reader.bit("segmentation flag")?;
	if !output.segmentation_enabled {
		return Ok(());
	}
	if output.primary_reference_frame == AV1_PRIMARY_REFERENCE_NONE {
		output.segmentation_update_map = true;
		output.segmentation_temporal_update = false;
		output.segmentation_update_data = true;
	} else {
		output.segmentation_update_map = reader.bit("segmentation-update-map flag")?;
		output.segmentation_temporal_update =
			output.segmentation_update_map && reader.bit("segmentation-temporal-update flag")?;
		output.segmentation_update_data = reader.bit("segmentation-update-data flag")?;
	}
	if !output.segmentation_update_data {
		return Ok(());
	}
	const FEATURE_BITS: [u32; SEGMENT_FEATURES] = [8, 6, 6, 6, 6, 3, 0, 0];
	for segment in 0..SEGMENTS {
		for (feature, bits) in FEATURE_BITS.into_iter().enumerate() {
			let enabled = reader.bit("segmentation feature flag")?;
			output.segment_feature_enabled[segment][feature] = enabled;
			if enabled && feature < 5 {
				let magnitude = reader.bits(bits, "signed segmentation feature")? as i16;
				output.segment_feature_data[segment][feature] =
					if reader.bit("segmentation feature sign")? {
						-magnitude
					} else {
						magnitude
					};
			} else if enabled && bits > 0 {
				output.segment_feature_data[segment][feature] =
					reader.bits(bits, "segmentation feature")? as i16;
			}
		}
	}
	Ok(())
}

fn parse_delta_quantization(
	reader: &mut Av1BitReader<'_>,
	output: &mut Av1FrameHeader,
) -> Result<()> {
	if output.base_q_idx > 0 {
		output.delta_q_present = reader.bit("delta-Q-present flag")?;
		if output.delta_q_present {
			output.delta_q_resolution = to_u8(reader.bits(2, "delta-Q resolution")?)?;
		}
	}
	if output.delta_q_present && !output.allow_intra_block_copy {
		output.delta_loop_filter_present = reader.bit("delta-loop-filter-present flag")?;
		if output.delta_loop_filter_present {
			output.delta_loop_filter_resolution = to_u8(reader.bits(2, "delta-loop-filter resolution")?)?;
			output.delta_loop_filter_multi = reader.bit("delta-loop-filter-multi flag")?;
		}
	}
	Ok(())
}

fn parse_loop_filter(
	reader: &mut Av1BitReader<'_>,
	sequence: &Av1SequenceHeader,
	output: &mut Av1FrameHeader,
) -> Result<()> {
	output.loop_filter_levels[0] = to_u8(reader.bits(6, "loop-filter level")?)?;
	output.loop_filter_levels[1] = to_u8(reader.bits(6, "loop-filter level")?)?;
	if !sequence.color.monochrome
		&& (output.loop_filter_levels[0] != 0 || output.loop_filter_levels[1] != 0)
	{
		output.loop_filter_levels[2] = to_u8(reader.bits(6, "chroma loop-filter level")?)?;
		output.loop_filter_levels[3] = to_u8(reader.bits(6, "chroma loop-filter level")?)?;
	}
	output.loop_filter_sharpness = to_u8(reader.bits(3, "loop-filter sharpness")?)?;
	output.loop_filter_delta_enabled = reader.bit("loop-filter-delta-enabled flag")?;
	if output.loop_filter_delta_enabled {
		output.loop_filter_delta_update = reader.bit("loop-filter-delta-update flag")?;
		if output.loop_filter_delta_update {
			for index in 0..8 {
				let update = reader.bit("loop-filter reference-delta update flag")?;
				output.loop_filter_update_reference_delta[index] = update;
				if update {
					output.loop_filter_reference_deltas[index] = read_inverse_signed(reader, 6)?;
				}
			}
			for index in 0..2 {
				let update = reader.bit("loop-filter mode-delta update flag")?;
				output.loop_filter_update_mode_delta[index] = update;
				if update {
					output.loop_filter_mode_deltas[index] = read_inverse_signed(reader, 6)?;
				}
			}
		}
	}
	Ok(())
}

fn parse_cdef(
	reader: &mut Av1BitReader<'_>,
	sequence: &Av1SequenceHeader,
	output: &mut Av1FrameHeader,
) -> Result<()> {
	if !sequence.enable_cdef {
		return Ok(());
	}
	output.cdef_damping_minus_3 = to_u8(reader.bits(2, "CDEF damping")?)?;
	output.cdef_bits = to_u8(reader.bits(2, "CDEF strength count")?)?;
	for index in 0..(1_usize << output.cdef_bits) {
		output.cdef_y_primary_strength[index] = to_u8(reader.bits(4, "CDEF luma primary")?)?;
		output.cdef_y_secondary_strength[index] = to_u8(reader.bits(2, "CDEF luma secondary")?)?;
		output.cdef_uv_primary_strength[index] = to_u8(reader.bits(4, "CDEF chroma primary")?)?;
		output.cdef_uv_secondary_strength[index] = to_u8(reader.bits(2, "CDEF chroma secondary")?)?;
	}
	Ok(())
}

fn parse_restoration(
	reader: &mut Av1BitReader<'_>,
	sequence: &Av1SequenceHeader,
	output: &mut Av1FrameHeader,
) -> Result<()> {
	for restoration in &mut output.restoration_types {
		*restoration = match reader.bits(2, "restoration type")? {
			0 => Av1RestorationType::None,
			1 => Av1RestorationType::Switchable,
			2 => Av1RestorationType::Wiener,
			3 => Av1RestorationType::Sgrproj,
			_ => unreachable!("two-bit restoration type"),
		};
	}
	let uses_restoration = output
		.restoration_types
		.iter()
		.any(|value| *value != Av1RestorationType::None);
	if !uses_restoration {
		return Ok(());
	}
	let mut unit_shift = u16::from(reader.bit("restoration unit shift")?);
	if sequence.use_128x128_superblock {
		unit_shift += 1;
	} else if unit_shift > 0 {
		unit_shift += u16::from(reader.bit("restoration unit extra shift")?);
	}
	let luma = 1 + unit_shift;
	output.restoration_unit_size_log2_minus_5[0] = luma;
	let chroma_used = output.restoration_types[1..]
		.iter()
		.any(|value| *value != Av1RestorationType::None);
	let chroma_shift = u16::from(
		chroma_used
			&& sequence.color.chroma_subsampling == VideoChromaSubsampling::Yuv420
			&& reader.bit("chroma restoration unit shift")?,
	);
	output.restoration_unit_size_log2_minus_5[1] = luma - chroma_shift;
	output.restoration_unit_size_log2_minus_5[2] = luma - chroma_shift;
	Ok(())
}

fn parse_skip_mode(
	reader: &mut Av1BitReader<'_>,
	sequence: &Av1SequenceHeader,
	references: &Av1ReferenceState,
	frame_is_intra: bool,
	output: &mut Av1FrameHeader,
) -> Result<()> {
	if frame_is_intra || !sequence.enable_order_hint {
		return Ok(());
	}
	let mut nearest_forward: Option<(usize, u8)> = None;
	let mut nearest_backward: Option<(usize, u8)> = None;
	for (name, logical) in output.reference_name_slot_indices.iter().enumerate() {
		let Some(hint) = references.order_hint(*logical as usize) else {
			continue;
		};
		let distance = relative_distance(hint, output.order_hint, sequence.order_hint_bits);
		if distance < 0
			&& nearest_forward
				.is_none_or(|(_, current)| relative_distance(hint, current, sequence.order_hint_bits) > 0)
		{
			nearest_forward = Some((name, hint));
		}
		if distance > 0
			&& nearest_backward
				.is_none_or(|(_, current)| relative_distance(hint, current, sequence.order_hint_bits) < 0)
		{
			nearest_backward = Some((name, hint));
		}
		if distance > 0 {
			output.reference_frame_sign_bias |= 1 << (name + 1);
		}
	}
	if !output.reference_select {
		return Ok(());
	}
	let mut second_forward = None;
	if let (Some((_, nearest_hint)), None) = (nearest_forward, nearest_backward) {
		for (name, logical) in output.reference_name_slot_indices.iter().enumerate() {
			let Some(hint) = references.order_hint(*logical as usize) else {
				continue;
			};
			if relative_distance(hint, nearest_hint, sequence.order_hint_bits) < 0
				&& second_forward
					.is_none_or(|(_, current)| relative_distance(hint, current, sequence.order_hint_bits) > 0)
			{
				second_forward = Some((name, hint));
			}
		}
	}
	let other = nearest_backward.or(second_forward);
	if let (Some((first, _)), Some((second, _))) = (nearest_forward, other) {
		output.skip_mode_present = reader.bit("skip-mode-present flag")?;
		let first = first as u8 + 1;
		let second = second as u8 + 1;
		output.skip_mode_frame = [first.min(second), first.max(second)];
	}
	Ok(())
}

fn read_delta_q(reader: &mut Av1BitReader<'_>) -> Result<i8> {
	if !reader.bit("quantization-delta flag")? {
		return Ok(0);
	}
	let value = reader.bits(7, "quantization delta")?;
	let magnitude = ((value + 1) >> 1) as i8;
	Ok(if value & 1 != 0 {
		-magnitude
	} else {
		(value >> 1) as i8
	})
}

fn read_inverse_signed(reader: &mut Av1BitReader<'_>, bits: u32) -> Result<i8> {
	let value = reader.bits(bits, "signed literal")?;
	Ok(if value & 1 != 0 {
		-(((value >> 1) + 1) as i8)
	} else {
		(value >> 1) as i8
	})
}

fn relative_distance(a: u8, b: u8, bits: u8) -> i32 {
	if bits == 0 {
		return 0;
	}
	let modulus = 1_u32 << bits;
	let difference = (u32::from(a).wrapping_sub(u32::from(b))) & (modulus - 1);
	if difference & (modulus >> 1) != 0 {
		difference as i32 - modulus as i32
	} else {
		difference as i32
	}
}

fn floor_log2(value: u32) -> u32 {
	u32::BITS - 1 - value.max(1).leading_zeros()
}

fn ceil_log2_div(count: u32, divisor: u32) -> u32 {
	let groups = count.max(1).div_ceil(divisor.max(1));
	if groups <= 1 {
		0
	} else {
		floor_log2(groups - 1) + 1
	}
}

fn to_u8(value: u32) -> Result<u8> {
	u8::try_from(value).map_err(|_| Error::data_loss("AV1 field exceeds u8 storage"))
}

fn to_i8(value: u32) -> Result<i8> {
	i8::try_from(value).map_err(|_| Error::data_loss("AV1 field exceeds i8 storage"))
}

fn to_u32(value: usize) -> Result<u32> {
	u32::try_from(value).map_err(|_| Error::out_of_range("AV1 byte range exceeds u32"))
}

fn read_little_endian(bytes: &[u8]) -> u32 {
	bytes
		.iter()
		.enumerate()
		.fold(0_u32, |value, (index, byte)| {
			value | (u32::from(*byte) << (index * 8))
		})
}
