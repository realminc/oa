//! Backend-neutral H.264 parameter-set parsing.

use crate::{Error, Result};

use super::bitstream::{BitReader, remove_emulation_prevention};

const MAX_H264_POC_CYCLE: usize = 256;

/// Parsed H.264 sequence parameter set required by decode-session setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H264SequenceParameterSet {
	pub id: u32,
	pub profile_idc: u32,
	pub level_idc: u32,
	pub constraint_flags: u32,
	pub chroma_format_idc: u32,
	pub bit_depth_luma_minus_8: u32,
	pub bit_depth_chroma_minus_8: u32,
	pub log2_max_frame_num_minus_4: u32,
	pub pic_order_count_type: u32,
	pub offset_for_non_ref_pic: i32,
	pub offset_for_top_to_bottom_field: i32,
	pub log2_max_pic_order_count_lsb_minus_4: u32,
	pub offset_for_ref_frame: Vec<i32>,
	pub width_in_macroblocks: u32,
	pub height_in_map_units: u32,
	pub max_num_ref_frames: u32,
	pub delta_pic_order_always_zero: bool,
	pub separate_colour_plane: bool,
	pub qpprime_y_zero_transform_bypass: bool,
	pub scaling_lists: Option<H264ScalingLists>,
	pub gaps_in_frame_num_value_allowed: bool,
	pub frame_mbs_only: bool,
	pub mb_adaptive_frame_field: bool,
	pub direct_8x8_inference: bool,
	pub frame_crop_left_offset: u32,
	pub frame_crop_right_offset: u32,
	pub frame_crop_top_offset: u32,
	pub frame_crop_bottom_offset: u32,
	pub vui: Option<H264VuiParameters>,
}

/// H.264 sequence- or picture-level scaling-list syntax in scan order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H264ScalingLists {
	pub present_mask: u16,
	pub use_default_mask: u16,
	pub list_4x4: [[u8; 16]; 6],
	pub list_8x8: [[u8; 64]; 6],
}

/// H.264 VUI aspect-ratio syntax retained with an optional extended SAR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct H264AspectRatio {
	pub idc: u8,
	pub sar_width: u16,
	pub sar_height: u16,
}

/// Optional H.264 VUI colour-description triplet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct H264ColourDescription {
	pub colour_primaries: u8,
	pub transfer_characteristics: u8,
	pub matrix_coefficients: u8,
}

/// H.264 VUI video-signal syntax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct H264VideoSignal {
	pub video_format: u8,
	pub full_range: bool,
	pub colour_description: Option<H264ColourDescription>,
}

/// H.264 VUI chroma sample locations for top and bottom fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct H264ChromaLocation {
	pub top_field: u32,
	pub bottom_field: u32,
}

/// H.264 VUI timing syntax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct H264TimingInfo {
	pub num_units_in_tick: u32,
	pub time_scale: u32,
	pub fixed_frame_rate: bool,
}

/// One H.264 hypothetical-reference-decoder CPB schedule entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct H264CpbEntry {
	pub bit_rate_value_minus_1: u32,
	pub cpb_size_value_minus_1: u32,
	pub constant_bit_rate: bool,
}

/// H.264 hypothetical-reference-decoder parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H264HrdParameters {
	pub bit_rate_scale: u8,
	pub cpb_size_scale: u8,
	pub entries: Vec<H264CpbEntry>,
	pub initial_cpb_removal_delay_length_minus_1: u8,
	pub cpb_removal_delay_length_minus_1: u8,
	pub dpb_output_delay_length_minus_1: u8,
	pub time_offset_length: u8,
}

/// H.264 VUI bitstream-restriction syntax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct H264BitstreamRestriction {
	pub motion_vectors_over_picture_boundaries: bool,
	pub max_bytes_per_picture_denom: u32,
	pub max_bits_per_macroblock_denom: u32,
	pub log2_max_motion_vector_length_horizontal: u32,
	pub log2_max_motion_vector_length_vertical: u32,
	pub max_num_reorder_frames: u32,
	pub max_dec_frame_buffering: u32,
}

/// Parsed H.264 video-usability information required by decoder backends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H264VuiParameters {
	pub aspect_ratio: Option<H264AspectRatio>,
	pub overscan_appropriate: Option<bool>,
	pub video_signal: Option<H264VideoSignal>,
	pub chroma_location: Option<H264ChromaLocation>,
	pub timing: Option<H264TimingInfo>,
	pub nal_hrd: Option<H264HrdParameters>,
	pub vcl_hrd: Option<H264HrdParameters>,
	pub low_delay_hrd: Option<bool>,
	pub picture_structure_present: bool,
	pub bitstream_restriction: Option<H264BitstreamRestriction>,
}

impl H264SequenceParameterSet {
	/// Return the coded width in luma samples before cropping.
	pub fn coded_width(&self) -> Result<u32> {
		self.width_in_macroblocks
			.checked_mul(16)
			.ok_or_else(|| Error::data_loss("H.264 coded width overflows u32"))
	}

	/// Return the coded height in luma samples before cropping.
	pub fn coded_height(&self) -> Result<u32> {
		self.height_in_map_units
			.checked_mul(if self.frame_mbs_only { 16 } else { 32 })
			.ok_or_else(|| Error::data_loss("H.264 coded height overflows u32"))
	}
}

/// Parsed H.264 picture parameter set required by decode-session setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H264PictureParameterSet {
	pub id: u32,
	pub sequence_parameter_set_id: u32,
	pub num_ref_idx_l0_default_active_minus_1: u32,
	pub num_ref_idx_l1_default_active_minus_1: u32,
	pub weighted_bipred_idc: u32,
	pub pic_init_qp_minus_26: i32,
	pub pic_init_qs_minus_26: i32,
	pub chroma_qp_index_offset: i32,
	pub second_chroma_qp_index_offset: i32,
	pub entropy_coding_mode: bool,
	pub bottom_field_pic_order_in_frame_present: bool,
	pub weighted_pred: bool,
	pub deblocking_filter_control_present: bool,
	pub constrained_intra_pred: bool,
	pub redundant_pic_count_present: bool,
	pub transform_8x8_mode: bool,
	pub scaling_lists: Option<H264ScalingLists>,
}

/// Normalized H.264 primary slice type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H264SliceType {
	P,
	B,
	I,
	Sp,
	Si,
}

/// One decoded-reference-picture marking command from an H.264 slice header.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct H264MemoryManagementControl {
	pub operation: u32,
	pub difference_of_picture_numbers_minus_1: u32,
	pub long_term_picture_number: u32,
	pub long_term_frame_index: u32,
	pub max_long_term_frame_index_plus_1: u32,
}

/// Parsed H.264 slice fields required by picture submission and DPB updates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H264SliceHeader {
	pub first_macroblock_in_slice: u32,
	pub slice_type: H264SliceType,
	pub picture_parameter_set_id: u32,
	pub frame_number: u32,
	pub idr_picture_id: Option<u32>,
	pub picture_order_count_lsb: Option<u32>,
	pub field_picture: bool,
	pub bottom_field: bool,
	pub is_idr: bool,
	pub is_reference: bool,
	pub no_output_of_prior_pictures: bool,
	pub long_term_reference: bool,
	pub adaptive_reference_picture_marking: bool,
	pub memory_management: Vec<H264MemoryManagementControl>,
}

/// Parse one raw H.264 SPS NAL unit without an Annex-B start code.
pub fn parse_h264_sps(nal: &[u8]) -> Result<H264SequenceParameterSet> {
	validate_nal(nal, 7, "SPS")?;
	let rbsp = remove_emulation_prevention(&nal[1..]);
	let mut bits = BitReader::new(&rbsp);
	let profile_idc = bits.read_bits(8)?;
	let constraint_flags = bits.read_bits(8)?;
	let level_idc = bits.read_bits(8)?;
	let id = bits.read_ue()?;
	if id > 31 {
		return Err(Error::data_loss("H.264 SPS id exceeds 31"));
	}
	let mut chroma_format_idc = 1;
	let mut bit_depth_luma_minus_8 = 0;
	let mut bit_depth_chroma_minus_8 = 0;
	let mut separate_colour_plane = false;
	let mut qpprime_y_zero_transform_bypass = false;
	let mut scaling_lists = None;
	if high_profile(profile_idc) {
		chroma_format_idc = bits.read_ue()?;
		if chroma_format_idc > 3 {
			return Err(Error::data_loss("H.264 SPS chroma format exceeds 3"));
		}
		if chroma_format_idc == 3 {
			separate_colour_plane = bits.read_bits(1)? != 0;
		}
		bit_depth_luma_minus_8 = bits.read_ue()?;
		bit_depth_chroma_minus_8 = bits.read_ue()?;
		qpprime_y_zero_transform_bypass = bits.read_bits(1)? != 0;
		if bits.read_bits(1)? != 0 {
			scaling_lists = Some(parse_h264_scaling_lists(
				&mut bits,
				if chroma_format_idc == 3 { 12 } else { 8 },
			)?);
		}
	}
	let log2_max_frame_num_minus_4 = bits.read_ue()?;
	let pic_order_count_type = bits.read_ue()?;
	let mut log2_max_pic_order_count_lsb_minus_4 = 0;
	let mut delta_pic_order_always_zero = false;
	let mut offset_for_non_ref_pic = 0;
	let mut offset_for_top_to_bottom_field = 0;
	let mut offset_for_ref_frame = Vec::new();
	if pic_order_count_type == 0 {
		log2_max_pic_order_count_lsb_minus_4 = bits.read_ue()?;
	} else if pic_order_count_type == 1 {
		delta_pic_order_always_zero = bits.read_bits(1)? != 0;
		offset_for_non_ref_pic = bits.read_se()?;
		offset_for_top_to_bottom_field = bits.read_se()?;
		let count = usize::try_from(bits.read_ue()?)
			.map_err(|_| Error::data_loss("H.264 POC cycle count exceeds usize"))?;
		if count > MAX_H264_POC_CYCLE {
			return Err(Error::data_loss("H.264 POC cycle exceeds safety bound"));
		}
		offset_for_ref_frame
			.try_reserve_exact(count)
			.map_err(|_| Error::resource_exhausted("H.264 POC cycle allocation failed"))?;
		for _ in 0..count {
			offset_for_ref_frame.push(bits.read_se()?);
		}
	} else if pic_order_count_type > 2 {
		return Err(Error::data_loss(
			"H.264 SPS picture-order-count type exceeds 2",
		));
	}
	let max_num_ref_frames = bits.read_ue()?;
	let gaps_in_frame_num_value_allowed = bits.read_bits(1)? != 0;
	let width_in_macroblocks = bits
		.read_ue()?
		.checked_add(1)
		.ok_or_else(|| Error::data_loss("H.264 macroblock width overflows"))?;
	let height_in_map_units = bits
		.read_ue()?
		.checked_add(1)
		.ok_or_else(|| Error::data_loss("H.264 map-unit height overflows"))?;
	let frame_mbs_only = bits.read_bits(1)? != 0;
	let mb_adaptive_frame_field = !frame_mbs_only && bits.read_bits(1)? != 0;
	let direct_8x8_inference = bits.read_bits(1)? != 0;
	let frame_cropping = bits.read_bits(1)? != 0;
	let (
		frame_crop_left_offset,
		frame_crop_right_offset,
		frame_crop_top_offset,
		frame_crop_bottom_offset,
	) = if frame_cropping {
		(
			bits.read_ue()?,
			bits.read_ue()?,
			bits.read_ue()?,
			bits.read_ue()?,
		)
	} else {
		(0, 0, 0, 0)
	};
	let vui = if bits.read_bits(1)? != 0 {
		Some(parse_h264_vui(&mut bits)?)
	} else {
		None
	};
	let parsed = H264SequenceParameterSet {
		id,
		profile_idc,
		level_idc,
		constraint_flags,
		chroma_format_idc,
		bit_depth_luma_minus_8,
		bit_depth_chroma_minus_8,
		log2_max_frame_num_minus_4,
		pic_order_count_type,
		offset_for_non_ref_pic,
		offset_for_top_to_bottom_field,
		log2_max_pic_order_count_lsb_minus_4,
		offset_for_ref_frame,
		width_in_macroblocks,
		height_in_map_units,
		max_num_ref_frames,
		delta_pic_order_always_zero,
		separate_colour_plane,
		qpprime_y_zero_transform_bypass,
		scaling_lists,
		gaps_in_frame_num_value_allowed,
		frame_mbs_only,
		mb_adaptive_frame_field,
		direct_8x8_inference,
		frame_crop_left_offset,
		frame_crop_right_offset,
		frame_crop_top_offset,
		frame_crop_bottom_offset,
		vui,
	};
	parsed.coded_width()?;
	parsed.coded_height()?;
	Ok(parsed)
}

/// Parse one raw H.264 PPS NAL unit without an Annex-B start code.
pub fn parse_h264_pps(
	nal: &[u8],
	sps: &H264SequenceParameterSet,
) -> Result<H264PictureParameterSet> {
	validate_nal(nal, 8, "PPS")?;
	let rbsp = remove_emulation_prevention(&nal[1..]);
	let mut bits = BitReader::new(&rbsp);
	let id = bits.read_ue()?;
	let sequence_parameter_set_id = bits.read_ue()?;
	if id > 255 || sequence_parameter_set_id > 31 {
		return Err(Error::data_loss(
			"H.264 PPS or referenced SPS id is out of range",
		));
	}
	if sequence_parameter_set_id != sps.id {
		return Err(Error::invalid_argument(
			"H.264 PPS does not reference the supplied SPS",
		));
	}
	let entropy_coding_mode = bits.read_bits(1)? != 0;
	let bottom_field_pic_order_in_frame_present = bits.read_bits(1)? != 0;
	if bits.read_ue()? != 0 {
		return Err(Error::missing_capability(
			"H.264 slice groups are not implemented",
		));
	}
	let num_ref_idx_l0_default_active_minus_1 = bits.read_ue()?;
	let num_ref_idx_l1_default_active_minus_1 = bits.read_ue()?;
	if num_ref_idx_l0_default_active_minus_1 >= 32 || num_ref_idx_l1_default_active_minus_1 >= 32 {
		return Err(Error::data_loss(
			"H.264 PPS reference-list count exceeds 32",
		));
	}
	let weighted_pred = bits.read_bits(1)? != 0;
	let weighted_bipred_idc = bits.read_bits(2)?;
	let pic_init_qp_minus_26 = bits.read_se()?;
	let pic_init_qs_minus_26 = bits.read_se()?;
	let chroma_qp_index_offset = bits.read_se()?;
	let deblocking_filter_control_present = bits.read_bits(1)? != 0;
	let constrained_intra_pred = bits.read_bits(1)? != 0;
	let redundant_pic_count_present = bits.read_bits(1)? != 0;
	let mut transform_8x8_mode = false;
	let mut scaling_lists = None;
	let mut second_chroma_qp_index_offset = chroma_qp_index_offset;
	if bits.more_rbsp_data() {
		transform_8x8_mode = bits.read_bits(1)? != 0;
		if bits.read_bits(1)? != 0 {
			let eight_by_eight_lists = if sps.chroma_format_idc == 3 { 6 } else { 2 };
			scaling_lists = Some(parse_h264_scaling_lists(
				&mut bits,
				6 + usize::from(transform_8x8_mode) * eight_by_eight_lists,
			)?);
		}
		second_chroma_qp_index_offset = bits.read_se()?;
	}
	Ok(H264PictureParameterSet {
		id,
		sequence_parameter_set_id,
		num_ref_idx_l0_default_active_minus_1,
		num_ref_idx_l1_default_active_minus_1,
		weighted_bipred_idc,
		pic_init_qp_minus_26,
		pic_init_qs_minus_26,
		chroma_qp_index_offset,
		second_chroma_qp_index_offset,
		entropy_coding_mode,
		bottom_field_pic_order_in_frame_present,
		weighted_pred,
		deblocking_filter_control_present,
		constrained_intra_pred,
		redundant_pic_count_present,
		transform_8x8_mode,
		scaling_lists,
	})
}

/// Parse one raw H.264 coded-slice NAL using its resolved SPS and PPS.
///
/// The returned picture-order-count field contains the coded LSB for POC type
/// zero. POC types one and two require prior-picture state and therefore return
/// `None`; the decoder session must derive the final POC rather than inventing
/// a stateless approximation.
pub fn parse_h264_slice_header(
	nal: &[u8],
	sps: &H264SequenceParameterSet,
	pps: &H264PictureParameterSet,
) -> Result<H264SliceHeader> {
	let Some(&nal_header) = nal.first() else {
		return Err(Error::data_loss("empty H.264 slice NAL"));
	};
	let nal_type = nal_header & 0x1f;
	if nal_header & 0x80 != 0 || !matches!(nal_type, 1 | 5) {
		return Err(Error::invalid_argument(
			"input is not an H.264 coded-slice NAL",
		));
	}
	if pps.sequence_parameter_set_id != sps.id {
		return Err(Error::invalid_argument(
			"H.264 PPS does not reference the supplied SPS",
		));
	}
	let rbsp = remove_emulation_prevention(nal);
	let mut bits = BitReader::new(&rbsp);
	bits.skip_bits(8)?;
	let first_macroblock_in_slice = bits.read_ue()?;
	let raw_slice_type = bits.read_ue()?;
	if raw_slice_type > 9 {
		return Err(Error::data_loss("H.264 slice type exceeds 9"));
	}
	let slice_type = match raw_slice_type % 5 {
		0 => H264SliceType::P,
		1 => H264SliceType::B,
		2 => H264SliceType::I,
		3 => H264SliceType::Sp,
		4 => H264SliceType::Si,
		_ => unreachable!(),
	};
	let picture_parameter_set_id = bits.read_ue()?;
	if picture_parameter_set_id != pps.id {
		return Err(Error::invalid_argument(
			"H.264 slice references a different PPS",
		));
	}
	let frame_number_bits = sps
		.log2_max_frame_num_minus_4
		.checked_add(4)
		.ok_or_else(|| Error::data_loss("H.264 frame-number width overflows"))?;
	if !(4..=16).contains(&frame_number_bits) {
		return Err(Error::data_loss(
			"H.264 frame-number width is outside 4..=16",
		));
	}
	let frame_number = bits.read_bits(frame_number_bits as usize)?;
	let mut field_picture = false;
	let mut bottom_field = false;
	if !sps.frame_mbs_only {
		field_picture = bits.read_bits(1)? != 0;
		if field_picture {
			bottom_field = bits.read_bits(1)? != 0;
		}
	}
	let is_idr = nal_type == 5;
	let idr_picture_id = is_idr.then(|| bits.read_ue()).transpose()?;
	let picture_order_count_lsb = if sps.pic_order_count_type == 0 {
		let count = sps
			.log2_max_pic_order_count_lsb_minus_4
			.checked_add(4)
			.ok_or_else(|| Error::data_loss("H.264 POC width overflows"))?;
		if !(4..=16).contains(&count) {
			return Err(Error::data_loss("H.264 POC width is outside 4..=16"));
		}
		let value = bits.read_bits(count as usize)?;
		if pps.bottom_field_pic_order_in_frame_present && !field_picture {
			bits.read_se()?;
		}
		Some(value)
	} else {
		if sps.pic_order_count_type == 1 && !sps.delta_pic_order_always_zero {
			bits.read_se()?;
			if pps.bottom_field_pic_order_in_frame_present && !field_picture {
				bits.read_se()?;
			}
		}
		None
	};
	if pps.redundant_pic_count_present {
		bits.read_ue()?;
	}
	let is_b = slice_type == H264SliceType::B;
	let has_list_0 = matches!(
		slice_type,
		H264SliceType::P | H264SliceType::Sp | H264SliceType::B
	);
	if is_b {
		bits.skip_bits(1)?;
	}
	let mut list_0_count = pps.num_ref_idx_l0_default_active_minus_1;
	let mut list_1_count = pps.num_ref_idx_l1_default_active_minus_1;
	if has_list_0 && bits.read_bits(1)? != 0 {
		list_0_count = bits.read_ue()?;
		if is_b {
			list_1_count = bits.read_ue()?;
		}
	}
	validate_reference_count(list_0_count)?;
	if is_b {
		validate_reference_count(list_1_count)?;
	}
	if has_list_0 {
		skip_reference_list_modification(&mut bits)?;
	}
	if is_b {
		skip_reference_list_modification(&mut bits)?;
	}
	if (matches!(slice_type, H264SliceType::P | H264SliceType::Sp) && pps.weighted_pred)
		|| (is_b && pps.weighted_bipred_idc == 1)
	{
		skip_prediction_weight_table(&mut bits, sps, list_0_count, list_1_count, is_b)?;
	}
	let is_reference = (nal_header >> 5) & 3 != 0;
	let mut no_output_of_prior_pictures = false;
	let mut long_term_reference = false;
	let mut adaptive_reference_picture_marking = false;
	let mut memory_management = Vec::new();
	if is_reference {
		if is_idr {
			no_output_of_prior_pictures = bits.read_bits(1)? != 0;
			long_term_reference = bits.read_bits(1)? != 0;
		} else {
			adaptive_reference_picture_marking = bits.read_bits(1)? != 0;
			if adaptive_reference_picture_marking {
				for _ in 0..64 {
					let operation = bits.read_ue()?;
					if operation == 0 {
						return Ok(H264SliceHeader {
							first_macroblock_in_slice,
							slice_type,
							picture_parameter_set_id,
							frame_number,
							idr_picture_id,
							picture_order_count_lsb,
							field_picture,
							bottom_field,
							is_idr,
							is_reference,
							no_output_of_prior_pictures,
							long_term_reference,
							adaptive_reference_picture_marking,
							memory_management,
						});
					}
					memory_management.push(read_memory_management_control(&mut bits, operation)?);
				}
				return Err(Error::data_loss(
					"H.264 memory-management command list exceeds 64 entries",
				));
			}
		}
	}
	Ok(H264SliceHeader {
		first_macroblock_in_slice,
		slice_type,
		picture_parameter_set_id,
		frame_number,
		idr_picture_id,
		picture_order_count_lsb,
		field_picture,
		bottom_field,
		is_idr,
		is_reference,
		no_output_of_prior_pictures,
		long_term_reference,
		adaptive_reference_picture_marking,
		memory_management,
	})
}

fn validate_reference_count(count_minus_1: u32) -> Result<()> {
	if count_minus_1 >= 32 {
		return Err(Error::data_loss(
			"H.264 active reference-list count exceeds 32",
		));
	}
	Ok(())
}

fn parse_h264_vui(bits: &mut BitReader<'_>) -> Result<H264VuiParameters> {
	let aspect_ratio = if bits.read_bits(1)? != 0 {
		let idc = bits.read_bits(8)? as u8;
		let (sar_width, sar_height) = if idc == 255 {
			(bits.read_bits(16)? as u16, bits.read_bits(16)? as u16)
		} else {
			(0, 0)
		};
		Some(H264AspectRatio {
			idc,
			sar_width,
			sar_height,
		})
	} else {
		None
	};
	let overscan_appropriate = if bits.read_bits(1)? != 0 {
		Some(bits.read_bits(1)? != 0)
	} else {
		None
	};
	let video_signal = if bits.read_bits(1)? != 0 {
		let video_format = bits.read_bits(3)? as u8;
		let full_range = bits.read_bits(1)? != 0;
		let colour_description = if bits.read_bits(1)? != 0 {
			Some(H264ColourDescription {
				colour_primaries: bits.read_bits(8)? as u8,
				transfer_characteristics: bits.read_bits(8)? as u8,
				matrix_coefficients: bits.read_bits(8)? as u8,
			})
		} else {
			None
		};
		Some(H264VideoSignal {
			video_format,
			full_range,
			colour_description,
		})
	} else {
		None
	};
	let chroma_location = if bits.read_bits(1)? != 0 {
		let top_field = bits.read_ue()?;
		let bottom_field = bits.read_ue()?;
		if top_field > 5 || bottom_field > 5 {
			return Err(Error::data_loss(
				"H.264 VUI chroma sample location exceeds 5",
			));
		}
		Some(H264ChromaLocation {
			top_field,
			bottom_field,
		})
	} else {
		None
	};
	let timing = if bits.read_bits(1)? != 0 {
		Some(H264TimingInfo {
			num_units_in_tick: bits.read_bits(32)?,
			time_scale: bits.read_bits(32)?,
			fixed_frame_rate: bits.read_bits(1)? != 0,
		})
	} else {
		None
	};
	let nal_hrd = if bits.read_bits(1)? != 0 {
		Some(parse_h264_hrd(bits)?)
	} else {
		None
	};
	let vcl_hrd = if bits.read_bits(1)? != 0 {
		Some(parse_h264_hrd(bits)?)
	} else {
		None
	};
	let low_delay_hrd = if nal_hrd.is_some() || vcl_hrd.is_some() {
		Some(bits.read_bits(1)? != 0)
	} else {
		None
	};
	let picture_structure_present = bits.read_bits(1)? != 0;
	let bitstream_restriction = if bits.read_bits(1)? != 0 {
		Some(H264BitstreamRestriction {
			motion_vectors_over_picture_boundaries: bits.read_bits(1)? != 0,
			max_bytes_per_picture_denom: bits.read_ue()?,
			max_bits_per_macroblock_denom: bits.read_ue()?,
			log2_max_motion_vector_length_horizontal: bits.read_ue()?,
			log2_max_motion_vector_length_vertical: bits.read_ue()?,
			max_num_reorder_frames: bits.read_ue()?,
			max_dec_frame_buffering: bits.read_ue()?,
		})
	} else {
		None
	};
	Ok(H264VuiParameters {
		aspect_ratio,
		overscan_appropriate,
		video_signal,
		chroma_location,
		timing,
		nal_hrd,
		vcl_hrd,
		low_delay_hrd,
		picture_structure_present,
		bitstream_restriction,
	})
}

fn parse_h264_scaling_lists(
	bits: &mut BitReader<'_>,
	list_count: usize,
) -> Result<H264ScalingLists> {
	if list_count > 12 {
		return Err(Error::data_loss("H.264 scaling-list count exceeds 12"));
	}
	let mut parsed = H264ScalingLists {
		present_mask: 0,
		use_default_mask: 0,
		list_4x4: [[0; 16]; 6],
		list_8x8: [[0; 64]; 6],
	};
	for index in 0..list_count {
		if bits.read_bits(1)? == 0 {
			continue;
		}
		parsed.present_mask |= 1_u16 << index;
		let use_default = if index < 6 {
			read_h264_scaling_list(bits, &mut parsed.list_4x4[index])?
		} else {
			read_h264_scaling_list(bits, &mut parsed.list_8x8[index - 6])?
		};
		if use_default {
			parsed.use_default_mask |= 1_u16 << index;
		}
	}
	Ok(parsed)
}

fn read_h264_scaling_list(bits: &mut BitReader<'_>, output: &mut [u8]) -> Result<bool> {
	let mut last_scale = 8_i32;
	let mut next_scale = 8_i32;
	let mut use_default = false;
	for (index, value) in output.iter_mut().enumerate() {
		if next_scale != 0 {
			next_scale = (last_scale + bits.read_se()? + 256) % 256;
			use_default = index == 0 && next_scale == 0;
		}
		let scale = if next_scale == 0 {
			last_scale
		} else {
			next_scale
		};
		*value = u8::try_from(scale)
			.map_err(|_| Error::data_loss("H.264 scaling-list value exceeds u8"))?;
		last_scale = scale;
	}
	Ok(use_default)
}

fn parse_h264_hrd(bits: &mut BitReader<'_>) -> Result<H264HrdParameters> {
	let entry_count = bits
		.read_ue()?
		.checked_add(1)
		.ok_or_else(|| Error::data_loss("H.264 HRD CPB count overflows"))?;
	if entry_count > 32 {
		return Err(Error::data_loss("H.264 HRD CPB count exceeds 32"));
	}
	let bit_rate_scale = bits.read_bits(4)? as u8;
	let cpb_size_scale = bits.read_bits(4)? as u8;
	let mut entries = Vec::new();
	entries
		.try_reserve_exact(entry_count as usize)
		.map_err(|_| Error::resource_exhausted("H.264 HRD entry allocation failed"))?;
	for _ in 0..entry_count {
		entries.push(H264CpbEntry {
			bit_rate_value_minus_1: bits.read_ue()?,
			cpb_size_value_minus_1: bits.read_ue()?,
			constant_bit_rate: bits.read_bits(1)? != 0,
		});
	}
	Ok(H264HrdParameters {
		bit_rate_scale,
		cpb_size_scale,
		entries,
		initial_cpb_removal_delay_length_minus_1: bits.read_bits(5)? as u8,
		cpb_removal_delay_length_minus_1: bits.read_bits(5)? as u8,
		dpb_output_delay_length_minus_1: bits.read_bits(5)? as u8,
		time_offset_length: bits.read_bits(5)? as u8,
	})
}

fn skip_reference_list_modification(bits: &mut BitReader<'_>) -> Result<()> {
	if bits.read_bits(1)? == 0 {
		return Ok(());
	}
	for _ in 0..256 {
		match bits.read_ue()? {
			3 => return Ok(()),
			0..=2 => {
				bits.read_ue()?;
			}
			_ => {
				return Err(Error::data_loss(
					"invalid H.264 reference-list modification",
				));
			}
		}
	}
	Err(Error::data_loss(
		"H.264 reference-list modification exceeds 256 entries",
	))
}

fn skip_prediction_weight_table(
	bits: &mut BitReader<'_>,
	sps: &H264SequenceParameterSet,
	list_0_count_minus_1: u32,
	list_1_count_minus_1: u32,
	has_list_1: bool,
) -> Result<()> {
	bits.read_ue()?;
	let chroma_array_type = if sps.separate_colour_plane {
		0
	} else {
		sps.chroma_format_idc
	};
	if chroma_array_type != 0 {
		bits.read_ue()?;
	}
	let mut skip_list = |count_minus_1| -> Result<()> {
		for _ in 0..=count_minus_1 {
			if bits.read_bits(1)? != 0 {
				bits.read_se()?;
				bits.read_se()?;
			}
			if chroma_array_type != 0 && bits.read_bits(1)? != 0 {
				for _ in 0..2 {
					bits.read_se()?;
					bits.read_se()?;
				}
			}
		}
		Ok(())
	};
	skip_list(list_0_count_minus_1)?;
	if has_list_1 {
		skip_list(list_1_count_minus_1)?;
	}
	Ok(())
}

fn read_memory_management_control(
	bits: &mut BitReader<'_>,
	operation: u32,
) -> Result<H264MemoryManagementControl> {
	let mut command = H264MemoryManagementControl {
		operation,
		..H264MemoryManagementControl::default()
	};
	match operation {
		1 => command.difference_of_picture_numbers_minus_1 = bits.read_ue()?,
		2 => command.long_term_picture_number = bits.read_ue()?,
		3 => {
			command.difference_of_picture_numbers_minus_1 = bits.read_ue()?;
			command.long_term_frame_index = bits.read_ue()?;
		}
		4 => command.max_long_term_frame_index_plus_1 = bits.read_ue()?,
		5 => {}
		6 => command.long_term_frame_index = bits.read_ue()?,
		_ => {
			return Err(Error::data_loss(
				"invalid H.264 memory-management operation",
			));
		}
	}
	Ok(command)
}

fn validate_nal(nal: &[u8], expected_type: u8, label: &str) -> Result<()> {
	let Some(&header) = nal.first() else {
		return Err(Error::data_loss(format!("empty H.264 {label} NAL")));
	};
	if header & 0x80 != 0 || header & 0x1f != expected_type {
		return Err(Error::invalid_argument(format!(
			"input is not a valid H.264 {label} NAL"
		)));
	}
	Ok(())
}

const fn high_profile(profile_idc: u32) -> bool {
	matches!(
		profile_idc,
		100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
	)
}

#[cfg(test)]
mod tests {
	use super::{BitReader, read_h264_scaling_list};

	#[test]
	fn scaling_list_retains_explicit_and_default_syntax() -> crate::Result<()> {
		let mut explicit = [0_u8; 16];
		let mut explicit_bits = BitReader::new(&[0xff, 0xff]);
		assert!(!read_h264_scaling_list(&mut explicit_bits, &mut explicit)?);
		assert_eq!(explicit, [8; 16]);

		let mut default = [0_u8; 16];
		let mut default_bits = BitReader::new(&[0b0000_1000, 0b1000_0000]);
		assert!(read_h264_scaling_list(&mut default_bits, &mut default)?);
		assert_eq!(default, [8; 16]);
		Ok(())
	}
}
