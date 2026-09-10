//! Backend-neutral H.265 parameter-set parsing.

use crate::{Error, Result};

use super::bitstream::{BitReader, remove_emulation_prevention};

const MAX_SUB_LAYERS: usize = 7;
const MAX_SHORT_TERM_REFERENCE_SETS: usize = 64;

const H265_DIAGONAL_SCAN_4X4_X: [usize; 16] = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 2, 3, 3];
const H265_DIAGONAL_SCAN_4X4_Y: [usize; 16] = [0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 3, 2, 3];
const H265_DIAGONAL_SCAN_8X8_X: [usize; 64] = [
	0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3,
	4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 4, 5, 6, 7, 5, 6, 7, 6, 7, 7,
];
const H265_DIAGONAL_SCAN_8X8_Y: [usize; 64] = [
	0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4,
	3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 7, 6, 5, 4, 7, 6, 5, 7, 6, 7,
];
const H265_DEFAULT_SCALING_LIST_INTRA: [u8; 64] = [
	16, 16, 16, 16, 17, 18, 21, 24, 16, 16, 16, 16, 17, 19, 22, 25, 16, 16, 17, 18, 20, 22, 25, 29,
	16, 16, 18, 21, 24, 27, 31, 36, 17, 17, 20, 24, 30, 35, 41, 47, 18, 19, 22, 27, 35, 44, 54, 65,
	21, 22, 25, 31, 41, 54, 70, 88, 24, 25, 29, 36, 47, 65, 88, 115,
];
const H265_DEFAULT_SCALING_LIST_INTER: [u8; 64] = [
	16, 16, 16, 16, 17, 18, 20, 24, 16, 16, 16, 17, 18, 20, 24, 25, 16, 16, 17, 18, 20, 24, 25, 28,
	16, 17, 18, 20, 24, 25, 28, 33, 17, 18, 20, 24, 25, 28, 33, 41, 18, 20, 24, 25, 28, 33, 41, 54,
	20, 24, 25, 28, 33, 41, 54, 71, 24, 25, 28, 33, 41, 54, 71, 91,
];

/// General H.265 profile-tier-level fields carried by a VPS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct H265ProfileTierLevel {
	pub profile_idc: u32,
	pub level_idc: u32,
	pub high_tier: bool,
	pub progressive_source: bool,
	pub interlaced_source: bool,
	pub non_packed_constraint: bool,
	pub frame_only_constraint: bool,
}

/// Fully resolved H.265 scaling matrices in raster order.
///
/// Prediction-mode references and diagonal coefficient walks are resolved by
/// the parser, so backend lowering can copy these arrays without retaining
/// bitstream-local prediction state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H265ScalingLists {
	pub list_4x4: [[u8; 16]; 6],
	pub list_8x8: [[u8; 64]; 6],
	pub list_16x16: [[u8; 64]; 6],
	pub list_32x32: [[u8; 64]; 2],
	pub dc_16x16: [u8; 6],
	pub dc_32x32: [u8; 2],
}

/// One fully parsed H.265 short-term reference-picture set.
///
/// Syntax masks are retained for standard-video lowering while `delta_pocs`
/// contains the resolved candidate order required to parse a following
/// inter-predicted set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct H265ShortTermReferencePictureSet {
	pub inter_ref_pic_set_prediction: bool,
	pub delta_index_minus_1: u32,
	pub delta_rps_sign: bool,
	pub abs_delta_rps_minus_1: u32,
	pub used_by_current_mask: u32,
	pub use_delta_mask: u32,
	pub negative_delta_poc_minus_1: Vec<u32>,
	pub positive_delta_poc_minus_1: Vec<u32>,
	pub used_by_current_negative_mask: u32,
	pub used_by_current_positive_mask: u32,
	pub delta_pocs: Vec<i32>,
}

/// One long-term H.265 reference picture declared by an SPS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct H265LongTermReferencePicture {
	pub picture_order_count_lsb: u32,
	pub used_by_current: bool,
}

impl Default for H265ScalingLists {
	fn default() -> Self {
		let mut lists = Self {
			list_4x4: [[16; 16]; 6],
			list_8x8: [[0; 64]; 6],
			list_16x16: [[0; 64]; 6],
			list_32x32: [
				H265_DEFAULT_SCALING_LIST_INTRA,
				H265_DEFAULT_SCALING_LIST_INTER,
			],
			dc_16x16: [16; 6],
			dc_32x32: [16; 2],
		};
		for matrix in 0..6 {
			let default = if matrix < 3 {
				H265_DEFAULT_SCALING_LIST_INTRA
			} else {
				H265_DEFAULT_SCALING_LIST_INTER
			};
			lists.list_8x8[matrix] = default;
			lists.list_16x16[matrix] = default;
		}
		lists
	}
}

/// Parsed H.265 video parameter set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H265VideoParameterSet {
	pub id: u32,
	pub max_sub_layers_minus_1: u32,
	pub temporal_id_nesting: bool,
	pub profile_tier_level: H265ProfileTierLevel,
}

/// Parsed H.265 sequence parameter set required by decode-session setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H265SequenceParameterSet {
	pub id: u32,
	pub video_parameter_set_id: u32,
	pub max_sub_layers_minus_1: u32,
	pub chroma_format_idc: u32,
	pub width: u32,
	pub height: u32,
	pub coded_width: u32,
	pub coded_height: u32,
	pub conformance_window: [u32; 4],
	pub bit_depth_luma_minus_8: u32,
	pub bit_depth_chroma_minus_8: u32,
	pub log2_max_pic_order_count_lsb_minus_4: u32,
	pub log2_min_luma_coding_block_size_minus_3: u32,
	pub log2_diff_max_min_luma_coding_block_size: u32,
	pub log2_min_luma_transform_block_size_minus_2: u32,
	pub log2_diff_max_min_luma_transform_block_size: u32,
	pub max_transform_hierarchy_depth_inter: u32,
	pub max_transform_hierarchy_depth_intra: u32,
	pub short_term_reference_picture_sets: Vec<H265ShortTermReferencePictureSet>,
	pub temporal_id_nesting: bool,
	pub separate_colour_plane: bool,
	pub sub_layer_ordering_info_present: bool,
	pub scaling_list_enabled: bool,
	pub scaling_lists: Option<H265ScalingLists>,
	pub asymmetric_motion_partitions_enabled: bool,
	pub sample_adaptive_offset_enabled: bool,
	pub pcm_enabled: bool,
	pub long_term_reference_pictures: Vec<H265LongTermReferencePicture>,
	pub temporal_mvp_enabled: bool,
	pub strong_intra_smoothing_enabled: bool,
	pub max_decoded_picture_buffering_minus_1: [u32; MAX_SUB_LAYERS],
	pub max_num_reorder_pictures: [u32; MAX_SUB_LAYERS],
	pub max_latency_increase_plus_1: [u32; MAX_SUB_LAYERS],
}

/// Parsed H.265 picture parameter set required by decode-session setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H265PictureParameterSet {
	pub id: u32,
	pub sequence_parameter_set_id: u32,
	pub num_extra_slice_header_bits: u32,
	pub num_ref_idx_l0_default_active_minus_1: u32,
	pub num_ref_idx_l1_default_active_minus_1: u32,
	pub init_qp_minus_26: i32,
	pub diff_cu_qp_delta_depth: u32,
	pub cb_qp_offset: i32,
	pub cr_qp_offset: i32,
	pub beta_offset_div_2: i32,
	pub tc_offset_div_2: i32,
	pub log2_parallel_merge_level_minus_2: u32,
	pub num_tile_columns_minus_1: u32,
	pub num_tile_rows_minus_1: u32,
	pub dependent_slice_segments_enabled: bool,
	pub output_flag_present: bool,
	pub sign_data_hiding_enabled: bool,
	pub cabac_init_present: bool,
	pub constrained_intra_pred: bool,
	pub transform_skip_enabled: bool,
	pub cu_qp_delta_enabled: bool,
	pub slice_chroma_qp_offsets_present: bool,
	pub weighted_pred: bool,
	pub weighted_bipred: bool,
	pub transquant_bypass_enabled: bool,
	pub tiles_enabled: bool,
	pub entropy_coding_sync_enabled: bool,
	pub uniform_spacing: bool,
	pub loop_filter_across_tiles_enabled: bool,
	pub loop_filter_across_slices_enabled: bool,
	pub deblocking_filter_control_present: bool,
	pub deblocking_filter_override_enabled: bool,
	pub deblocking_filter_disabled: bool,
	pub scaling_lists: Option<H265ScalingLists>,
	pub lists_modification_present: bool,
	pub slice_segment_header_extension_present: bool,
	pub extension_present: bool,
}

/// H.265 primary slice type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H265SliceType {
	B,
	P,
	I,
}

/// Parsed H.265 slice fields required by picture submission and DPB updates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H265SliceHeader {
	pub picture_parameter_set_id: u32,
	pub sequence_parameter_set_id: u32,
	pub video_parameter_set_id: u32,
	pub slice_type: H265SliceType,
	pub slice_segment_address: u32,
	pub picture_order_count_lsb: Option<u32>,
	pub short_term_reference_picture_set_sps: bool,
	pub short_term_reference_picture_set_index: Option<u32>,
	pub short_term_reference_picture_set_bits: u16,
	pub short_term_current_before_delta_pocs: Vec<i32>,
	pub short_term_current_after_delta_pocs: Vec<i32>,
	pub short_term_following_delta_pocs: Vec<i32>,
	pub nal_unit_type: u8,
	pub temporal_id: u8,
	pub first_slice_segment_in_picture: bool,
	pub is_irap: bool,
	pub is_idr: bool,
	pub no_output_of_prior_pictures: bool,
	pub is_reference: bool,
	pub picture_output: bool,
	pub colour_plane_id: Option<u8>,
}

/// Parse one raw H.265 VPS NAL unit without an Annex-B start code.
pub fn parse_h265_vps(nal: &[u8]) -> Result<H265VideoParameterSet> {
	validate_nal(nal, 32, "VPS")?;
	let rbsp = remove_emulation_prevention(nal);
	let mut bits = BitReader::new(&rbsp);
	bits.skip_bits(16)?;
	let id = bits.read_bits(4)?;
	bits.skip_bits(8)?;
	let max_sub_layers_minus_1 = bits.read_bits(3)?;
	validate_sub_layers(max_sub_layers_minus_1)?;
	let temporal_id_nesting = bits.read_bits(1)? != 0;
	bits.skip_bits(16)?;
	let profile_tier_level = read_profile_tier_level(&mut bits, max_sub_layers_minus_1)?;
	Ok(H265VideoParameterSet {
		id,
		max_sub_layers_minus_1,
		temporal_id_nesting,
		profile_tier_level,
	})
}

/// Parse one raw H.265 SPS NAL unit without an Annex-B start code.
pub fn parse_h265_sps(nal: &[u8]) -> Result<H265SequenceParameterSet> {
	validate_nal(nal, 33, "SPS")?;
	let rbsp = remove_emulation_prevention(nal);
	let mut bits = BitReader::new(&rbsp);
	bits.skip_bits(16)?;
	let video_parameter_set_id = bits.read_bits(4)?;
	let max_sub_layers_minus_1 = bits.read_bits(3)?;
	validate_sub_layers(max_sub_layers_minus_1)?;
	let temporal_id_nesting = bits.read_bits(1)? != 0;
	read_profile_tier_level(&mut bits, max_sub_layers_minus_1)?;
	let id = bits.read_ue()?;
	if id > 15 {
		return Err(Error::data_loss("H.265 SPS identifier exceeds 15"));
	}
	let chroma_format_idc = bits.read_ue()?;
	if chroma_format_idc > 3 {
		return Err(Error::data_loss("H.265 SPS chroma format exceeds 3"));
	}
	let separate_colour_plane = chroma_format_idc == 3 && bits.read_bits(1)? != 0;
	let coded_width = bits.read_ue()?;
	let coded_height = bits.read_ue()?;
	if coded_width == 0 || coded_height == 0 {
		return Err(Error::data_loss("H.265 SPS has zero coded extent"));
	}
	let mut width = coded_width;
	let mut height = coded_height;
	let mut conformance_window = [0; 4];
	if bits.read_bits(1)? != 0 {
		for value in &mut conformance_window {
			*value = bits.read_ue()?;
		}
		let (crop_unit_x, crop_unit_y) = match chroma_format_idc {
			0 | 3 => (1, 1),
			1 => (2, 2),
			2 => (2, 1),
			_ => unreachable!(),
		};
		let crop_width = conformance_window[0]
			.checked_add(conformance_window[1])
			.and_then(|value| value.checked_mul(crop_unit_x))
			.ok_or_else(|| Error::data_loss("H.265 conformance width overflows"))?;
		let crop_height = conformance_window[2]
			.checked_add(conformance_window[3])
			.and_then(|value| value.checked_mul(crop_unit_y))
			.ok_or_else(|| Error::data_loss("H.265 conformance height overflows"))?;
		width = width
			.checked_sub(crop_width)
			.ok_or_else(|| Error::data_loss("H.265 crop exceeds coded width"))?;
		height = height
			.checked_sub(crop_height)
			.ok_or_else(|| Error::data_loss("H.265 crop exceeds coded height"))?;
	}
	let bit_depth_luma_minus_8 = bits.read_ue()?;
	let bit_depth_chroma_minus_8 = bits.read_ue()?;
	let log2_max_pic_order_count_lsb_minus_4 = bits.read_ue()?;
	let sub_layer_ordering_info_present = bits.read_bits(1)? != 0;
	let ordering_start = if sub_layer_ordering_info_present {
		0
	} else {
		max_sub_layers_minus_1 as usize
	};
	let mut max_decoded_picture_buffering_minus_1 = [0; MAX_SUB_LAYERS];
	let mut max_num_reorder_pictures = [0; MAX_SUB_LAYERS];
	let mut max_latency_increase_plus_1 = [0; MAX_SUB_LAYERS];
	for index in ordering_start..=max_sub_layers_minus_1 as usize {
		max_decoded_picture_buffering_minus_1[index] = bits.read_ue()?;
		max_num_reorder_pictures[index] = bits.read_ue()?;
		max_latency_increase_plus_1[index] = bits.read_ue()?;
	}
	if !sub_layer_ordering_info_present {
		for index in 0..ordering_start {
			max_decoded_picture_buffering_minus_1[index] =
				max_decoded_picture_buffering_minus_1[ordering_start];
			max_num_reorder_pictures[index] = max_num_reorder_pictures[ordering_start];
			max_latency_increase_plus_1[index] = max_latency_increase_plus_1[ordering_start];
		}
	}
	let log2_min_luma_coding_block_size_minus_3 = bits.read_ue()?;
	let log2_diff_max_min_luma_coding_block_size = bits.read_ue()?;
	let log2_min_luma_transform_block_size_minus_2 = bits.read_ue()?;
	let log2_diff_max_min_luma_transform_block_size = bits.read_ue()?;
	let max_transform_hierarchy_depth_inter = bits.read_ue()?;
	let max_transform_hierarchy_depth_intra = bits.read_ue()?;
	let scaling_list_enabled = bits.read_bits(1)? != 0;
	let scaling_list_data_present = scaling_list_enabled && bits.read_bits(1)? != 0;
	let scaling_lists = scaling_list_data_present
		.then(|| parse_h265_scaling_lists(&mut bits))
		.transpose()?;
	let asymmetric_motion_partitions_enabled = bits.read_bits(1)? != 0;
	let sample_adaptive_offset_enabled = bits.read_bits(1)? != 0;
	let pcm_enabled = bits.read_bits(1)? != 0;
	if pcm_enabled {
		bits.skip_bits(8)?;
		bits.read_ue()?;
		bits.read_ue()?;
		bits.skip_bits(1)?;
	}
	let num_short_term_reference_picture_sets = bits.read_ue()?;
	let count = num_short_term_reference_picture_sets as usize;
	if count > MAX_SHORT_TERM_REFERENCE_SETS {
		return Err(Error::data_loss(
			"H.265 short-term RPS count exceeds safety bound",
		));
	}
	let mut short_term_reference_picture_sets = Vec::new();
	short_term_reference_picture_sets
		.try_reserve_exact(count)
		.map_err(|_| Error::resource_exhausted("H.265 reference-set allocation failed"))?;
	for index in 0..count {
		let set = parse_short_term_reference_picture_set(
			&mut bits,
			index,
			&short_term_reference_picture_sets,
		)?;
		if set.delta_pocs.len()
			> max_decoded_picture_buffering_minus_1[max_sub_layers_minus_1 as usize] as usize
		{
			return Err(Error::data_loss(
				"H.265 reference set exceeds the SPS decoded-picture buffer",
			));
		}
		short_term_reference_picture_sets.push(set);
	}
	let long_term_reference_pictures_present = bits.read_bits(1)? != 0;
	let mut long_term_reference_pictures = Vec::new();
	if long_term_reference_pictures_present {
		let count = bits.read_ue()?;
		if count > 32 {
			return Err(Error::data_loss(
				"H.265 long-term reference count exceeds safety bound",
			));
		}
		long_term_reference_pictures
			.try_reserve_exact(count as usize)
			.map_err(|_| {
				Error::resource_exhausted("H.265 long-term reference allocation failed")
			})?;
		let poc_bits = log2_max_pic_order_count_lsb_minus_4
			.checked_add(4)
			.ok_or_else(|| Error::data_loss("H.265 long-term POC width overflows"))?;
		if !(4..=32).contains(&poc_bits) {
			return Err(Error::data_loss(
				"H.265 long-term POC width is outside 4..=32",
			));
		}
		for _ in 0..count {
			long_term_reference_pictures.push(H265LongTermReferencePicture {
				picture_order_count_lsb: bits.read_bits(poc_bits as usize)?,
				used_by_current: bits.read_bits(1)? != 0,
			});
		}
	}
	let temporal_mvp_enabled = bits.read_bits(1)? != 0;
	let strong_intra_smoothing_enabled = bits.read_bits(1)? != 0;
	Ok(H265SequenceParameterSet {
		id,
		video_parameter_set_id,
		max_sub_layers_minus_1,
		chroma_format_idc,
		width,
		height,
		coded_width,
		coded_height,
		conformance_window,
		bit_depth_luma_minus_8,
		bit_depth_chroma_minus_8,
		log2_max_pic_order_count_lsb_minus_4,
		log2_min_luma_coding_block_size_minus_3,
		log2_diff_max_min_luma_coding_block_size,
		log2_min_luma_transform_block_size_minus_2,
		log2_diff_max_min_luma_transform_block_size,
		max_transform_hierarchy_depth_inter,
		max_transform_hierarchy_depth_intra,
		short_term_reference_picture_sets,
		temporal_id_nesting,
		separate_colour_plane,
		sub_layer_ordering_info_present,
		scaling_list_enabled,
		scaling_lists,
		asymmetric_motion_partitions_enabled,
		sample_adaptive_offset_enabled,
		pcm_enabled,
		long_term_reference_pictures,
		temporal_mvp_enabled,
		strong_intra_smoothing_enabled,
		max_decoded_picture_buffering_minus_1,
		max_num_reorder_pictures,
		max_latency_increase_plus_1,
	})
}

/// Parse one raw H.265 PPS NAL unit without an Annex-B start code.
pub fn parse_h265_pps(nal: &[u8]) -> Result<H265PictureParameterSet> {
	validate_nal(nal, 34, "PPS")?;
	let rbsp = remove_emulation_prevention(nal);
	let mut bits = BitReader::new(&rbsp);
	bits.skip_bits(16)?;
	let id = bits.read_ue()?;
	let sequence_parameter_set_id = bits.read_ue()?;
	if id > 63 || sequence_parameter_set_id > 15 {
		return Err(Error::data_loss("H.265 PPS identifier is out of range"));
	}
	let dependent_slice_segments_enabled = bits.read_bits(1)? != 0;
	let output_flag_present = bits.read_bits(1)? != 0;
	let num_extra_slice_header_bits = bits.read_bits(3)?;
	let sign_data_hiding_enabled = bits.read_bits(1)? != 0;
	let cabac_init_present = bits.read_bits(1)? != 0;
	let num_ref_idx_l0_default_active_minus_1 = bits.read_ue()?;
	let num_ref_idx_l1_default_active_minus_1 = bits.read_ue()?;
	let init_qp_minus_26 = bits.read_se()?;
	let constrained_intra_pred = bits.read_bits(1)? != 0;
	let transform_skip_enabled = bits.read_bits(1)? != 0;
	let cu_qp_delta_enabled = bits.read_bits(1)? != 0;
	let diff_cu_qp_delta_depth = if cu_qp_delta_enabled {
		bits.read_ue()?
	} else {
		0
	};
	let cb_qp_offset = bits.read_se()?;
	let cr_qp_offset = bits.read_se()?;
	let slice_chroma_qp_offsets_present = bits.read_bits(1)? != 0;
	let weighted_pred = bits.read_bits(1)? != 0;
	let weighted_bipred = bits.read_bits(1)? != 0;
	let transquant_bypass_enabled = bits.read_bits(1)? != 0;
	let tiles_enabled = bits.read_bits(1)? != 0;
	let entropy_coding_sync_enabled = bits.read_bits(1)? != 0;
	let (mut num_tile_columns_minus_1, mut num_tile_rows_minus_1) = (0, 0);
	let mut uniform_spacing = true;
	let mut loop_filter_across_tiles_enabled = false;
	if tiles_enabled {
		num_tile_columns_minus_1 = bits.read_ue()?;
		num_tile_rows_minus_1 = bits.read_ue()?;
		if num_tile_columns_minus_1 > 1024 || num_tile_rows_minus_1 > 1024 {
			return Err(Error::data_loss(
				"H.265 PPS tile count exceeds safety bound",
			));
		}
		uniform_spacing = bits.read_bits(1)? != 0;
		if !uniform_spacing {
			for _ in 0..num_tile_columns_minus_1 {
				bits.read_ue()?;
			}
			for _ in 0..num_tile_rows_minus_1 {
				bits.read_ue()?;
			}
		}
		loop_filter_across_tiles_enabled = bits.read_bits(1)? != 0;
	}
	let loop_filter_across_slices_enabled = bits.read_bits(1)? != 0;
	let deblocking_filter_control_present = bits.read_bits(1)? != 0;
	let (mut deblocking_filter_override_enabled, mut deblocking_filter_disabled) = (false, false);
	let (mut beta_offset_div_2, mut tc_offset_div_2) = (0, 0);
	if deblocking_filter_control_present {
		deblocking_filter_override_enabled = bits.read_bits(1)? != 0;
		deblocking_filter_disabled = bits.read_bits(1)? != 0;
		if !deblocking_filter_disabled {
			beta_offset_div_2 = bits.read_se()?;
			tc_offset_div_2 = bits.read_se()?;
		}
	}
	let scaling_list_data_present = bits.read_bits(1)? != 0;
	let scaling_lists = scaling_list_data_present
		.then(|| parse_h265_scaling_lists(&mut bits))
		.transpose()?;
	let lists_modification_present = bits.read_bits(1)? != 0;
	let log2_parallel_merge_level_minus_2 = bits.read_ue()?;
	let slice_segment_header_extension_present = bits.read_bits(1)? != 0;
	let extension_present = bits.read_bits(1)? != 0;
	Ok(H265PictureParameterSet {
		id,
		sequence_parameter_set_id,
		num_extra_slice_header_bits,
		num_ref_idx_l0_default_active_minus_1,
		num_ref_idx_l1_default_active_minus_1,
		init_qp_minus_26,
		diff_cu_qp_delta_depth,
		cb_qp_offset,
		cr_qp_offset,
		beta_offset_div_2,
		tc_offset_div_2,
		log2_parallel_merge_level_minus_2,
		num_tile_columns_minus_1,
		num_tile_rows_minus_1,
		dependent_slice_segments_enabled,
		output_flag_present,
		sign_data_hiding_enabled,
		cabac_init_present,
		constrained_intra_pred,
		transform_skip_enabled,
		cu_qp_delta_enabled,
		slice_chroma_qp_offsets_present,
		weighted_pred,
		weighted_bipred,
		transquant_bypass_enabled,
		tiles_enabled,
		entropy_coding_sync_enabled,
		uniform_spacing,
		loop_filter_across_tiles_enabled,
		loop_filter_across_slices_enabled,
		deblocking_filter_control_present,
		deblocking_filter_override_enabled,
		deblocking_filter_disabled,
		scaling_lists,
		lists_modification_present,
		slice_segment_header_extension_present,
		extension_present,
	})
}

/// Parse one raw H.265 coded-slice NAL using its resolved SPS and PPS.
///
/// Dependent slice segments and inline inter-predicted short-term reference
/// sets require additional retained state and fail with `MissingCapability`.
pub fn parse_h265_slice_header(
	nal: &[u8],
	sps: &H265SequenceParameterSet,
	pps: &H265PictureParameterSet,
) -> Result<H265SliceHeader> {
	if nal.len() < 2 {
		return Err(Error::data_loss("truncated H.265 slice NAL"));
	}
	let nal_unit_type = (nal[0] >> 1) & 0x3f;
	let temporal_id_plus_1 = nal[1] & 7;
	if nal[0] & 0x80 != 0 || nal_unit_type >= 32 || temporal_id_plus_1 == 0 {
		return Err(Error::invalid_argument(
			"input is not a valid H.265 coded-slice NAL",
		));
	}
	if pps.sequence_parameter_set_id != sps.id {
		return Err(Error::invalid_argument(
			"H.265 PPS does not reference the supplied SPS",
		));
	}
	let rbsp = remove_emulation_prevention(nal);
	let mut bits = BitReader::new(&rbsp);
	bits.skip_bits(16)?;
	let first_slice_segment_in_picture = bits.read_bits(1)? != 0;
	let is_irap = (16..=23).contains(&nal_unit_type);
	let is_idr = matches!(nal_unit_type, 19 | 20);
	let is_reference = is_irap || nal_unit_type & 1 != 0;
	let no_output_of_prior_pictures = is_irap && bits.read_bits(1)? != 0;
	let picture_parameter_set_id = bits.read_ue()?;
	if picture_parameter_set_id != pps.id {
		return Err(Error::invalid_argument(
			"H.265 slice references a different PPS",
		));
	}
	let mut slice_segment_address = 0;
	if !first_slice_segment_in_picture {
		let dependent = pps.dependent_slice_segments_enabled && bits.read_bits(1)? != 0;
		let ctb_log2 = sps
			.log2_min_luma_coding_block_size_minus_3
			.checked_add(3)
			.and_then(|value| value.checked_add(sps.log2_diff_max_min_luma_coding_block_size))
			.ok_or_else(|| Error::data_loss("H.265 CTB exponent overflows"))?;
		let ctb_size = 1_u32
			.checked_shl(ctb_log2)
			.ok_or_else(|| Error::data_loss("H.265 CTB size exceeds u32"))?;
		let ctb_width = sps
			.coded_width
			.checked_add(ctb_size - 1)
			.ok_or_else(|| Error::data_loss("H.265 CTB width rounds past u32"))?
			/ ctb_size;
		let ctb_height = sps
			.coded_height
			.checked_add(ctb_size - 1)
			.ok_or_else(|| Error::data_loss("H.265 CTB height rounds past u32"))?
			/ ctb_size;
		let picture_ctbs = ctb_width
			.checked_mul(ctb_height)
			.ok_or_else(|| Error::data_loss("H.265 CTB count overflows"))?;
		let address_bits = ceil_log2(picture_ctbs);
		slice_segment_address = bits.read_bits(address_bits as usize)?;
		if dependent {
			return Err(Error::missing_capability(
				"H.265 dependent slice segments are not implemented",
			));
		}
	}
	bits.skip_bits(pps.num_extra_slice_header_bits as usize)?;
	let slice_type = match bits.read_ue()? {
		0 => H265SliceType::B,
		1 => H265SliceType::P,
		2 => H265SliceType::I,
		_ => return Err(Error::data_loss("H.265 slice type exceeds 2")),
	};
	let picture_output = !pps.output_flag_present || bits.read_bits(1)? != 0;
	let colour_plane_id = sps
		.separate_colour_plane
		.then(|| bits.read_bits(2).map(|value| value as u8))
		.transpose()?;
	let mut picture_order_count_lsb = None;
	let mut short_term_reference_picture_set_sps = false;
	let mut short_term_reference_picture_set_index = None;
	let mut short_term_reference_picture_set_bits = 0;
	let mut short_term_current_before_delta_pocs = Vec::new();
	let mut short_term_current_after_delta_pocs = Vec::new();
	let mut short_term_following_delta_pocs = Vec::new();
	if !is_idr {
		let poc_bits = sps
			.log2_max_pic_order_count_lsb_minus_4
			.checked_add(4)
			.ok_or_else(|| Error::data_loss("H.265 POC width overflows"))?;
		if !(4..=32).contains(&poc_bits) {
			return Err(Error::data_loss("H.265 POC width is outside 4..=32"));
		}
		picture_order_count_lsb = Some(bits.read_bits(poc_bits as usize)?);
		let reference_set_count = u32::try_from(sps.short_term_reference_picture_sets.len())
			.map_err(|_| Error::data_loss("H.265 SPS reference-set count exceeds u32"))?;
		short_term_reference_picture_set_sps = reference_set_count != 0 && bits.read_bits(1)? != 0;
		if short_term_reference_picture_set_sps {
			if reference_set_count == 0 {
				return Err(Error::data_loss(
					"H.265 slice selects an SPS reference set but the SPS has none",
				));
			}
			let index_bits = ceil_log2(reference_set_count);
			short_term_reference_picture_set_index = Some(if index_bits == 0 {
				0
			} else {
				bits.read_bits(index_bits as usize)?
			});
			if short_term_reference_picture_set_index
				.is_some_and(|index| index >= reference_set_count)
			{
				return Err(Error::data_loss(
					"H.265 short-term reference-set index is out of range",
				));
			}
		} else {
			if reference_set_count != 0 {
				return Err(Error::missing_capability(
					"H.265 inline inter-predicted short-term reference sets require retained SPS RPS state",
				));
			}
			let start = bits.position();
			let negative_count = bits.read_ue()?;
			let positive_count = bits.read_ue()?;
			if negative_count > 16 || positive_count > 16 {
				return Err(Error::data_loss(
					"H.265 inline short-term reference set exceeds 16 entries per direction",
				));
			}
			let mut delta_poc = 0_i32;
			for _ in 0..negative_count {
				let delta = i32::try_from(bits.read_ue()?)
					.map_err(|_| Error::data_loss("H.265 negative delta POC exceeds i32"))?
					.checked_add(1)
					.ok_or_else(|| Error::data_loss("H.265 negative delta POC overflows"))?;
				delta_poc = delta_poc
					.checked_sub(delta)
					.ok_or_else(|| Error::data_loss("H.265 negative delta POC sum overflows"))?;
				if bits.read_bits(1)? != 0 {
					short_term_current_before_delta_pocs.push(delta_poc);
				} else {
					short_term_following_delta_pocs.push(delta_poc);
				}
			}
			delta_poc = 0;
			for _ in 0..positive_count {
				let delta = i32::try_from(bits.read_ue()?)
					.map_err(|_| Error::data_loss("H.265 positive delta POC exceeds i32"))?
					.checked_add(1)
					.ok_or_else(|| Error::data_loss("H.265 positive delta POC overflows"))?;
				delta_poc = delta_poc
					.checked_add(delta)
					.ok_or_else(|| Error::data_loss("H.265 positive delta POC sum overflows"))?;
				if bits.read_bits(1)? != 0 {
					short_term_current_after_delta_pocs.push(delta_poc);
				} else {
					short_term_following_delta_pocs.push(delta_poc);
				}
			}
			short_term_reference_picture_set_bits = u16::try_from(bits.position() - start)
				.map_err(|_| Error::data_loss("H.265 inline reference set exceeds 65535 bits"))?;
		}
	}
	Ok(H265SliceHeader {
		picture_parameter_set_id,
		sequence_parameter_set_id: sps.id,
		video_parameter_set_id: sps.video_parameter_set_id,
		slice_type,
		slice_segment_address,
		picture_order_count_lsb,
		short_term_reference_picture_set_sps,
		short_term_reference_picture_set_index,
		short_term_reference_picture_set_bits,
		short_term_current_before_delta_pocs,
		short_term_current_after_delta_pocs,
		short_term_following_delta_pocs,
		nal_unit_type,
		temporal_id: temporal_id_plus_1 - 1,
		first_slice_segment_in_picture,
		is_irap,
		is_idr,
		no_output_of_prior_pictures,
		is_reference,
		picture_output,
		colour_plane_id,
	})
}

const fn ceil_log2(value: u32) -> u32 {
	if value <= 1 {
		0
	} else {
		u32::BITS - (value - 1).leading_zeros()
	}
}

fn read_profile_tier_level(bits: &mut BitReader<'_>, layers: u32) -> Result<H265ProfileTierLevel> {
	bits.skip_bits(2)?;
	let high_tier = bits.read_bits(1)? != 0;
	let profile_idc = bits.read_bits(5)?;
	bits.skip_bits(32)?;
	let progressive_source = bits.read_bits(1)? != 0;
	let interlaced_source = bits.read_bits(1)? != 0;
	let non_packed_constraint = bits.read_bits(1)? != 0;
	let frame_only_constraint = bits.read_bits(1)? != 0;
	bits.skip_bits(32)?;
	bits.skip_bits(12)?;
	let level_idc = bits.read_bits(8)?;
	let count = layers as usize;
	let mut profile_present = [false; 8];
	let mut level_present = [false; 8];
	for index in 0..count {
		profile_present[index] = bits.read_bits(1)? != 0;
		level_present[index] = bits.read_bits(1)? != 0;
	}
	if count > 0 {
		for _ in count..8 {
			bits.skip_bits(2)?;
		}
	}
	for index in 0..count {
		if profile_present[index] {
			bits.skip_bits(88)?;
		}
		if level_present[index] {
			bits.skip_bits(8)?;
		}
	}
	Ok(H265ProfileTierLevel {
		profile_idc,
		level_idc,
		high_tier,
		progressive_source,
		interlaced_source,
		non_packed_constraint,
		frame_only_constraint,
	})
}

fn parse_h265_scaling_lists(bits: &mut BitReader<'_>) -> Result<H265ScalingLists> {
	let mut lists = H265ScalingLists::default();
	for size_id in 0..4 {
		let matrix_step = if size_id == 3 { 3 } else { 1 };
		for matrix_id in (0..6).step_by(matrix_step) {
			let dense_matrix_id = matrix_id / matrix_step;
			if bits.read_bits(1)? == 0 {
				let delta = usize::try_from(bits.read_ue()?).map_err(|_| {
					Error::data_loss("H.265 scaling-list predictor delta exceeds usize")
				})?;
				if delta > dense_matrix_id {
					return Err(Error::data_loss(
						"H.265 scaling-list predictor precedes the first matrix",
					));
				}
				if delta != 0 {
					copy_h265_scaling_list(
						&mut lists,
						size_id,
						dense_matrix_id,
						dense_matrix_id - delta,
					);
				}
				continue;
			}

			let mut next_coefficient = 8_u8;
			if size_id > 1 {
				let dc_minus_8 = bits.read_se()?;
				if !(-7..=247).contains(&dc_minus_8) {
					return Err(Error::data_loss(
						"H.265 scaling-list DC coefficient is outside 1..=255",
					));
				}
				next_coefficient = (dc_minus_8 + 8) as u8;
				set_h265_scaling_dc(&mut lists, size_id, dense_matrix_id, next_coefficient);
			}
			let coefficient_count = if size_id == 0 { 16 } else { 64 };
			for coefficient in 0..coefficient_count {
				let delta = bits.read_se()?;
				next_coefficient =
					(i64::from(next_coefficient) + i64::from(delta)).rem_euclid(256) as u8;
				let position = if size_id == 0 {
					4 * H265_DIAGONAL_SCAN_4X4_Y[coefficient]
						+ H265_DIAGONAL_SCAN_4X4_X[coefficient]
				} else {
					8 * H265_DIAGONAL_SCAN_8X8_Y[coefficient]
						+ H265_DIAGONAL_SCAN_8X8_X[coefficient]
				};
				set_h265_scaling_value(
					&mut lists,
					size_id,
					dense_matrix_id,
					position,
					next_coefficient,
				);
			}
		}
	}
	Ok(lists)
}

fn copy_h265_scaling_list(
	lists: &mut H265ScalingLists,
	size_id: usize,
	destination: usize,
	source: usize,
) {
	match size_id {
		0 => lists.list_4x4[destination] = lists.list_4x4[source],
		1 => lists.list_8x8[destination] = lists.list_8x8[source],
		2 => {
			lists.list_16x16[destination] = lists.list_16x16[source];
			lists.dc_16x16[destination] = lists.dc_16x16[source];
		}
		3 => {
			lists.list_32x32[destination] = lists.list_32x32[source];
			lists.dc_32x32[destination] = lists.dc_32x32[source];
		}
		_ => unreachable!(),
	}
}

fn set_h265_scaling_dc(lists: &mut H265ScalingLists, size_id: usize, matrix_id: usize, value: u8) {
	match size_id {
		2 => lists.dc_16x16[matrix_id] = value,
		3 => lists.dc_32x32[matrix_id] = value,
		_ => unreachable!(),
	}
}

fn set_h265_scaling_value(
	lists: &mut H265ScalingLists,
	size_id: usize,
	matrix_id: usize,
	position: usize,
	value: u8,
) {
	match size_id {
		0 => lists.list_4x4[matrix_id][position] = value,
		1 => lists.list_8x8[matrix_id][position] = value,
		2 => lists.list_16x16[matrix_id][position] = value,
		3 => lists.list_32x32[matrix_id][position] = value,
		_ => unreachable!(),
	}
}

fn parse_short_term_reference_picture_set(
	bits: &mut BitReader<'_>,
	index: usize,
	previous: &[H265ShortTermReferencePictureSet],
) -> Result<H265ShortTermReferencePictureSet> {
	let inter_ref_pic_set_prediction = index != 0 && bits.read_bits(1)? != 0;
	if inter_ref_pic_set_prediction {
		let reference = previous
			.get(index - 1)
			.ok_or_else(|| Error::data_loss("H.265 predicted reference set has no predecessor"))?;
		if reference.delta_pocs.len() >= u32::BITS as usize {
			return Err(Error::data_loss(
				"H.265 predicted reference set exceeds syntax-mask storage",
			));
		}
		let delta_rps_sign = bits.read_bits(1)? != 0;
		let abs_delta_rps_minus_1 = bits.read_ue()?;
		if abs_delta_rps_minus_1 > 32_767 {
			return Err(Error::data_loss(
				"H.265 reference delta exceeds the 15-bit syntax range",
			));
		}
		let delta_rps_magnitude = i32::try_from(abs_delta_rps_minus_1)
			.map_err(|_| Error::data_loss("H.265 reference delta exceeds i32"))?
			.checked_add(1)
			.ok_or_else(|| Error::data_loss("H.265 reference delta overflows"))?;
		let delta_rps = if delta_rps_sign {
			-delta_rps_magnitude
		} else {
			delta_rps_magnitude
		};
		let mut used_by_current_mask = 0_u32;
		let mut use_delta_mask = 0_u32;
		let mut included = Vec::new();
		included
			.try_reserve_exact(reference.delta_pocs.len() + 1)
			.map_err(|_| {
				Error::resource_exhausted("H.265 predicted reference allocation failed")
			})?;
		for candidate in 0..=reference.delta_pocs.len() {
			let used_by_current = bits.read_bits(1)? != 0;
			let use_delta = used_by_current || bits.read_bits(1)? != 0;
			if used_by_current {
				used_by_current_mask |= 1 << candidate;
			}
			if use_delta {
				use_delta_mask |= 1 << candidate;
				let reference_delta = reference.delta_pocs.get(candidate).copied().unwrap_or(0);
				let delta = delta_rps.checked_add(reference_delta).ok_or_else(|| {
					Error::data_loss("H.265 predicted picture-order delta overflows")
				})?;
				if delta != 0 {
					included.push((delta, used_by_current));
				}
			}
		}
		included.sort_by_key(|(delta, _)| *delta);
		let negative_count = included.partition_point(|(delta, _)| *delta < 0);
		included[..negative_count].reverse();
		let (used_by_current_negative_mask, used_by_current_positive_mask) =
			derived_reference_masks(&included, negative_count);
		return Ok(H265ShortTermReferencePictureSet {
			inter_ref_pic_set_prediction,
			delta_index_minus_1: 0,
			delta_rps_sign,
			abs_delta_rps_minus_1,
			used_by_current_mask,
			use_delta_mask,
			negative_delta_poc_minus_1: Vec::new(),
			positive_delta_poc_minus_1: Vec::new(),
			used_by_current_negative_mask,
			used_by_current_positive_mask,
			delta_pocs: included.into_iter().map(|(delta, _)| delta).collect(),
		});
	}

	let negative_count = usize::try_from(bits.read_ue()?)
		.map_err(|_| Error::data_loss("H.265 negative RPS count exceeds usize"))?;
	let positive_count = usize::try_from(bits.read_ue()?)
		.map_err(|_| Error::data_loss("H.265 positive RPS count exceeds usize"))?;
	if negative_count > 16 || positive_count > 16 {
		return Err(Error::data_loss(
			"H.265 RPS exceeds 16 entries in one direction",
		));
	}
	let total = negative_count
		.checked_add(positive_count)
		.ok_or_else(|| Error::data_loss("H.265 RPS entry count overflows"))?;
	let mut set = H265ShortTermReferencePictureSet::default();
	set.negative_delta_poc_minus_1
		.try_reserve_exact(negative_count)
		.map_err(|_| Error::resource_exhausted("H.265 negative RPS allocation failed"))?;
	set.positive_delta_poc_minus_1
		.try_reserve_exact(positive_count)
		.map_err(|_| Error::resource_exhausted("H.265 positive RPS allocation failed"))?;
	set.delta_pocs
		.try_reserve_exact(total)
		.map_err(|_| Error::resource_exhausted("H.265 RPS delta allocation failed"))?;
	let mut delta_poc = 0_i32;
	for entry in 0..negative_count {
		let delta_minus_1 = bits.read_ue()?;
		if delta_minus_1 > 32_767 {
			return Err(Error::data_loss(
				"H.265 negative RPS delta exceeds the 15-bit syntax range",
			));
		}
		let delta = i32::try_from(delta_minus_1)
			.map_err(|_| Error::data_loss("H.265 negative RPS delta exceeds i32"))?
			.checked_add(1)
			.ok_or_else(|| Error::data_loss("H.265 negative RPS delta overflows"))?;
		delta_poc = delta_poc
			.checked_sub(delta)
			.ok_or_else(|| Error::data_loss("H.265 negative RPS sum overflows"))?;
		set.negative_delta_poc_minus_1.push(delta_minus_1);
		set.delta_pocs.push(delta_poc);
		if bits.read_bits(1)? != 0 {
			set.used_by_current_negative_mask |= 1 << entry;
		}
	}
	delta_poc = 0;
	for entry in 0..positive_count {
		let delta_minus_1 = bits.read_ue()?;
		if delta_minus_1 > 32_767 {
			return Err(Error::data_loss(
				"H.265 positive RPS delta exceeds the 15-bit syntax range",
			));
		}
		let delta = i32::try_from(delta_minus_1)
			.map_err(|_| Error::data_loss("H.265 positive RPS delta exceeds i32"))?
			.checked_add(1)
			.ok_or_else(|| Error::data_loss("H.265 positive RPS delta overflows"))?;
		delta_poc = delta_poc
			.checked_add(delta)
			.ok_or_else(|| Error::data_loss("H.265 positive RPS sum overflows"))?;
		set.positive_delta_poc_minus_1.push(delta_minus_1);
		set.delta_pocs.push(delta_poc);
		if bits.read_bits(1)? != 0 {
			set.used_by_current_positive_mask |= 1 << entry;
		}
	}
	Ok(set)
}

fn derived_reference_masks(included: &[(i32, bool)], negative_count: usize) -> (u32, u32) {
	let mut negative = 0_u32;
	let mut positive = 0_u32;
	for (index, (_, used)) in included.iter().copied().enumerate() {
		if !used {
			continue;
		}
		if index < negative_count {
			negative |= 1 << index;
		} else {
			positive |= 1 << (index - negative_count);
		}
	}
	(negative, positive)
}

fn validate_nal(nal: &[u8], expected_type: u8, label: &str) -> Result<()> {
	if nal.len() < 2 {
		return Err(Error::data_loss(format!("truncated H.265 {label} NAL")));
	}
	if nal[0] & 0x80 != 0 || (nal[0] >> 1) & 0x3f != expected_type || nal[1] & 7 == 0 {
		return Err(Error::invalid_argument(format!(
			"input is not a valid H.265 {label} NAL"
		)));
	}
	Ok(())
}

fn validate_sub_layers(value: u32) -> Result<()> {
	if value >= MAX_SUB_LAYERS as u32 {
		return Err(Error::data_loss("H.265 sub-layer count exceeds seven"));
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::{
		BitReader, H265_DEFAULT_SCALING_LIST_INTER, H265_DEFAULT_SCALING_LIST_INTRA,
		H265_DIAGONAL_SCAN_4X4_X, H265_DIAGONAL_SCAN_4X4_Y, parse_h265_scaling_lists,
		parse_short_term_reference_picture_set,
	};

	#[test]
	fn scaling_lists_resolve_defaults_predictions_and_diagonal_order() -> crate::Result<()> {
		let mut syntax = Vec::new();
		for size_id in 0..4 {
			let matrix_count = if size_id == 3 { 2 } else { 6 };
			for matrix_id in 0..matrix_count {
				if size_id == 0 && matrix_id == 0 {
					syntax.push(true);
					for _ in 0..16 {
						push_se(&mut syntax, 1);
					}
				} else if size_id == 0 && matrix_id == 1 {
					syntax.push(false);
					push_ue(&mut syntax, 1);
				} else if size_id == 2 && matrix_id == 0 {
					syntax.push(true);
					push_se(&mut syntax, -7);
					for _ in 0..64 {
						push_se(&mut syntax, 0);
					}
				} else {
					syntax.push(false);
					push_ue(&mut syntax, 0);
				}
			}
		}
		let bytes = pack_bits(&syntax);
		let mut bits = BitReader::new(&bytes);
		let lists = parse_h265_scaling_lists(&mut bits)?;

		for coefficient in 0..16 {
			let position =
				4 * H265_DIAGONAL_SCAN_4X4_Y[coefficient] + H265_DIAGONAL_SCAN_4X4_X[coefficient];
			assert_eq!(lists.list_4x4[0][position], 9 + coefficient as u8);
		}
		assert_eq!(lists.list_4x4[1], lists.list_4x4[0]);
		assert_eq!(lists.list_8x8[0], H265_DEFAULT_SCALING_LIST_INTRA);
		assert_eq!(lists.list_8x8[3], H265_DEFAULT_SCALING_LIST_INTER);
		assert_eq!(lists.dc_16x16[0], 1);
		assert_eq!(lists.list_16x16[0], [1; 64]);
		assert_eq!(lists.dc_32x32, [16; 2]);
		Ok(())
	}

	#[test]
	fn scaling_list_rejects_a_predictor_before_the_first_matrix() {
		let mut syntax = vec![false];
		push_ue(&mut syntax, 1);
		let bytes = pack_bits(&syntax);
		let mut bits = BitReader::new(&bytes);
		assert!(parse_h265_scaling_lists(&mut bits).is_err());
	}

	#[test]
	fn short_term_reference_sets_retain_and_resolve_prediction() -> crate::Result<()> {
		let mut direct_syntax = Vec::new();
		push_ue(&mut direct_syntax, 2);
		push_ue(&mut direct_syntax, 1);
		push_ue(&mut direct_syntax, 0);
		direct_syntax.push(true);
		push_ue(&mut direct_syntax, 1);
		direct_syntax.push(false);
		push_ue(&mut direct_syntax, 1);
		direct_syntax.push(true);
		let direct_bytes = pack_bits(&direct_syntax);
		let mut direct_bits = BitReader::new(&direct_bytes);
		let direct = parse_short_term_reference_picture_set(&mut direct_bits, 0, &[])?;
		assert!(!direct.inter_ref_pic_set_prediction);
		assert_eq!(direct.negative_delta_poc_minus_1, [0, 1]);
		assert_eq!(direct.positive_delta_poc_minus_1, [1]);
		assert_eq!(direct.delta_pocs, [-1, -3, 2]);
		assert_eq!(direct.used_by_current_negative_mask, 0b01);
		assert_eq!(direct.used_by_current_positive_mask, 0b1);

		let mut predicted_syntax = vec![true, false];
		push_ue(&mut predicted_syntax, 0);
		predicted_syntax.extend([false, false, true, false, true, true]);
		let predicted_bytes = pack_bits(&predicted_syntax);
		let mut predicted_bits = BitReader::new(&predicted_bytes);
		let predicted = parse_short_term_reference_picture_set(&mut predicted_bits, 1, &[direct])?;
		assert!(predicted.inter_ref_pic_set_prediction);
		assert!(!predicted.delta_rps_sign);
		assert_eq!(predicted.abs_delta_rps_minus_1, 0);
		assert_eq!(predicted.used_by_current_mask, 0b1010);
		assert_eq!(predicted.use_delta_mask, 0b1110);
		assert_eq!(predicted.delta_pocs, [-2, 1, 3]);
		assert_eq!(predicted.used_by_current_negative_mask, 0b1);
		assert_eq!(predicted.used_by_current_positive_mask, 0b01);
		Ok(())
	}

	fn push_ue(bits: &mut Vec<bool>, value: u32) {
		let code = value + 1;
		let width = u32::BITS - code.leading_zeros();
		bits.extend(std::iter::repeat_n(false, width as usize - 1));
		for shift in (0..width).rev() {
			bits.push(code & (1 << shift) != 0);
		}
	}

	fn push_se(bits: &mut Vec<bool>, value: i32) {
		let code = if value > 0 {
			u32::try_from(value).expect("positive test coefficient") * 2 - 1
		} else {
			value.unsigned_abs() * 2
		};
		push_ue(bits, code);
	}

	fn pack_bits(bits: &[bool]) -> Vec<u8> {
		let mut bytes = vec![0; bits.len().div_ceil(8)];
		for (index, bit) in bits.iter().copied().enumerate() {
			if bit {
				bytes[index / 8] |= 1 << (7 - index % 8);
			}
		}
		bytes
	}
}
