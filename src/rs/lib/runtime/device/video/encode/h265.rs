//! H.265 Vulkan Video encode parameters and picture recording.

use crate::{Error, Result, video};

use super::super::profile::h265_profile;
use super::{
	EncodePictureArgs, EncodeSession, bit_depth_minus8, record_coding, reference_resources,
};

pub(super) fn create_parameters(
	session: &EncodeSession,
	settings: &video::VideoEncodeConfig,
	caps: &video::VideoEncodeCapabilities,
) -> Result<vk::VideoSessionParametersKHR> {
	let mut quality =
		vk::VideoEncodeQualityLevelInfoKHR::new().set_quality_level(settings.quality_level);
	let video::VideoEncodeCodecCapabilities::H265 {
		ctb_size_16,
		ctb_size_32,
		ctb_size_64,
		transform_size_4,
		transform_size_8,
		transform_size_16,
		transform_size_32,
		..
	} = caps.codec()
	else {
		return Err(Error::invalid_argument(
			"H.265 session parameters require H.265 codec capabilities",
		));
	};
	let mut parameters = build_h265_parameters(
		settings,
		session.coded_extent,
		session.dpb_slot_capacity,
		ctb_size_16,
		ctb_size_32,
		ctb_size_64,
		transform_size_4,
		transform_size_8,
		transform_size_16,
		transform_size_32,
	);
	// Link the pointer fields in place before copying the sets —
	// the copies then reference this stack frame for the call.
	parameters.link();
	let vpss = [parameters.vps];
	let spss = [parameters.sps];
	let ppss = [parameters.pps];
	let add = vk::VideoEncodeH265SessionParametersAddInfoKHR::new()
		.set_std_vp_ss(&vpss)
		.map_err(Error::from)?
		.set_std_sp_ss(&spss)
		.map_err(Error::from)?
		.set_std_pp_ss(&ppss)
		.map_err(Error::from)?;
	let mut codec = vk::VideoEncodeH265SessionParametersCreateInfoKHR::new()
		.set_max_std_vps_count(1)
		.set_max_std_sps_count(1)
		.set_max_std_pps_count(1)
		.set_parameters_add_info(&add);
	let create = vk::VideoSessionParametersCreateInfoKHR::new()
		.set_video_session(session.handle)
		.push_next(&mut codec)
		.map_err(Error::from)?
		.push_next(&mut quality)
		.map_err(Error::from)?;
	session.create_parameter_handle(&create)
}

pub(super) fn cache_headers(session: &mut EncodeSession) -> Result<()> {
	let mut codec = vk::VideoEncodeH265SessionParametersGetInfoKHR::new()
		.set_write_std_vps(vk::TRUE)
		.set_write_std_sps(vk::TRUE)
		.set_write_std_pps(vk::TRUE)
		.set_std_vps_id(0)
		.set_std_sps_id(0)
		.set_std_pps_id(0);
	let get = vk::VideoEncodeSessionParametersGetInfoKHR::new()
		.set_video_session_parameters(session.parameters)
		.push_next(&mut codec)
		.map_err(Error::from)?;
	session.cache_encoded_headers(&get)
}

pub(super) fn record_frame(
	device: &vk::Device,
	command_buffer: vk::CommandBuffer,
	args: &EncodePictureArgs,
) -> Result<()> {
	let [current_resource, previous_resource] = reference_resources(args);
	let mut current_ref = vk::StdVideoEncodeH265ReferenceInfo::new();
	let mut previous_ref = vk::StdVideoEncodeH265ReferenceInfo::new();
	let mut current = vk::VideoEncodeH265DpbSlotInfoKHR::new();
	let mut previous = vk::VideoEncodeH265DpbSlotInfoKHR::new();
	let mut slots = [
		vk::VideoReferenceSlotInfoKHR::new(),
		vk::VideoReferenceSlotInfoKHR::new(),
	];
	// Bind the current picture as inactive and list the single active
	// reference only for a predicted picture.
	// The setup slot describes DPB reference state, not the NAL picture
	// type — the donor leaves it P even for the first IDR (Intel expects
	// this exact convention).
	current_ref.pic_type = vk::StdVideoH265PictureType::P;
	current_ref.pic_order_cnt_val = if args.is_idr {
		0
	} else {
		args.frame_index as i32
	};
	current_ref.temporal_id = 0;
	current = current.set_std_reference_info(&current_ref);
	let slot0 = vk::VideoReferenceSlotInfoKHR::new()
		.set_slot_index(-1)
		.set_picture_resource(&current_resource)
		.push_next(&mut current)
		.map_err(Error::from)?;
	if !args.is_idr {
		previous_ref.pic_type = vk::StdVideoH265PictureType::P;
		previous_ref.pic_order_cnt_val = args.frame_index as i32 - 1;
		previous_ref.temporal_id = 0;
		previous = previous.set_std_reference_info(&previous_ref);
		let slot1 = vk::VideoReferenceSlotInfoKHR::new()
			.set_slot_index(args.previous_dpb_slot as i32)
			.set_picture_resource(&previous_resource)
			.push_next(&mut previous)
			.map_err(Error::from)?;
		slots[1] = slot1;
	}
	slots[0] = slot0;
	let layer = vk::VideoEncodeH265RateControlLayerInfoKHR::new()
		.set_use_min_qp(vk::FALSE)
		.set_use_max_qp(vk::FALSE)
		.set_use_max_frame_size(vk::FALSE);
	let reference_count = usize::from(!args.is_idr) + 1;
	record_coding(
		device,
		command_buffer,
		args,
		&slots[..reference_count],
		|| {
			vk::VideoEncodeH265RateControlInfoKHR::new()
				.set_gop_frame_count(args.gop_size)
				.set_idr_period(args.gop_size)
				.set_consecutive_b_frame_count(args.max_b_frames)
				.set_sub_layer_count(1)
		},
		layer,
		record_picture,
	)
}

#[allow(clippy::field_reassign_with_default)]
fn record_picture(
	device: &vk::Device,
	command_buffer: vk::CommandBuffer,
	args: &EncodePictureArgs,
	src_resource: vk::VideoPictureResourceInfoKHR<'_>,
) -> Result<()> {
	let mut ref_lists = vk::StdVideoEncodeH265ReferenceListsInfo::new();
	ref_lists
		.ref_pic_list0
		.fill(vk::STD_VIDEO_H265_NO_REFERENCE_PICTURE);
	ref_lists
		.ref_pic_list1
		.fill(vk::STD_VIDEO_H265_NO_REFERENCE_PICTURE);
	ref_lists
		.list_entry_l0
		.fill(vk::STD_VIDEO_H265_NO_REFERENCE_PICTURE);
	ref_lists
		.list_entry_l1
		.fill(vk::STD_VIDEO_H265_NO_REFERENCE_PICTURE);
	if !args.is_idr {
		ref_lists.num_ref_idx_l0_active_minus1 = 0;
		ref_lists.ref_pic_list0[0] = args.previous_dpb_slot as u8;
	}
	let picture_poc = if args.is_idr {
		0
	} else {
		args.frame_index as i32
	};
	let mut pic_flags = vk::StdVideoEncodeH265PictureInfoFlags::new();
	pic_flags.set_is_reference(1);
	pic_flags.set_irap_pic_flag(u32::from(args.is_idr));
	pic_flags.set_pic_output_flag(1);
	pic_flags.set_no_output_of_prior_pics_flag(u32::from(args.is_idr && !args.first_frame));
	pic_flags.set_short_term_ref_pic_set_sps_flag(1);
	let mut pic_info = vk::StdVideoEncodeH265PictureInfo::new();
	pic_info.flags = pic_flags;
	pic_info.pic_type = if args.is_idr {
		vk::StdVideoH265PictureType::IDR
	} else {
		vk::StdVideoH265PictureType::P
	};
	pic_info.sps_video_parameter_set_id = 0;
	pic_info.pps_seq_parameter_set_id = 0;
	pic_info.pps_pic_parameter_set_id = 0;
	pic_info.short_term_ref_pic_set_idx = 0;
	pic_info.pic_order_cnt_val = picture_poc;
	pic_info.temporal_id = 0;
	pic_info.p_ref_lists = if args.is_idr {
		std::ptr::null()
	} else {
		&ref_lists
	};

	let mut slice_flags = vk::StdVideoEncodeH265SliceSegmentHeaderFlags::new();
	slice_flags.set_first_slice_segment_in_pic_flag(1);
	slice_flags.set_slice_sao_luma_flag(1);
	slice_flags.set_slice_sao_chroma_flag(1);
	let mut slice_header = vk::StdVideoEncodeH265SliceSegmentHeader::new();
	slice_header.flags = slice_flags;
	slice_header.slice_type = if args.is_idr {
		vk::StdVideoH265SliceType::I
	} else {
		vk::StdVideoH265SliceType::P
	};
	slice_header.max_num_merge_cand = 5;
	let constant_qp =
		if args.rate_control_mode == vk::VideoEncodeRateControlModeFlagBitsKHR::DISABLED_BIT_KHR {
			args.constant_qp as i32
		} else {
			0
		};
	// The fixed-QP PPS already carries the requested base QP. Match it here;
	// a rate-controlled session retains the default PPS base of 26.
	slice_header.slice_qp_delta =
		if args.rate_control_mode == vk::VideoEncodeRateControlModeFlagBitsKHR::DISABLED_BIT_KHR {
			0
		} else {
			-26
		};

	let nalu = vk::VideoEncodeH265NaluSliceSegmentInfoKHR::new()
		.set_constant_qp(constant_qp)
		.set_std_slice_segment_header(&slice_header);
	let nalus = std::slice::from_ref(&nalu);
	let mut h265_picture = vk::VideoEncodeH265PictureInfoKHR::new()
		.set_nalu_slice_segment_entries(nalus)
		.map_err(Error::from)?
		.set_std_picture_info(&pic_info);

	let mut current_ref = vk::StdVideoEncodeH265ReferenceInfo::new();
	current_ref.pic_type = vk::StdVideoH265PictureType::P;
	current_ref.pic_order_cnt_val = picture_poc;
	current_ref.temporal_id = 0;
	let mut current_slot =
		vk::VideoEncodeH265DpbSlotInfoKHR::new().set_std_reference_info(&current_ref);
	let setup_resource = vk::VideoPictureResourceInfoKHR::new()
		.set_coded_extent(vk::Extent2D {
			width: args.coded.width,
			height: args.coded.height,
		})
		.set_base_array_layer(args.current_dpb_slot)
		.set_image_view_binding(args.dpb_view);
	let setup_slot = vk::VideoReferenceSlotInfoKHR::new()
		.set_slot_index(args.current_dpb_slot as i32)
		.set_picture_resource(&setup_resource)
		.push_next(&mut current_slot)
		.map_err(Error::from)?;

	let mut previous_ref = vk::StdVideoEncodeH265ReferenceInfo::new();
	previous_ref.pic_type = vk::StdVideoH265PictureType::P;
	previous_ref.pic_order_cnt_val = picture_poc - 1;
	previous_ref.temporal_id = 0;
	let mut previous_slot_info =
		vk::VideoEncodeH265DpbSlotInfoKHR::new().set_std_reference_info(&previous_ref);
	let previous_resource = vk::VideoPictureResourceInfoKHR::new()
		.set_coded_extent(vk::Extent2D {
			width: args.coded.width,
			height: args.coded.height,
		})
		.set_base_array_layer(args.previous_dpb_slot)
		.set_image_view_binding(args.dpb_view);
	let previous_slot = vk::VideoReferenceSlotInfoKHR::new()
		.set_slot_index(args.previous_dpb_slot as i32)
		.set_picture_resource(&previous_resource)
		.push_next(&mut previous_slot_info)
		.map_err(Error::from)?;
	let reference_slots = [previous_slot];
	let mut encode = vk::VideoEncodeInfoKHR::new()
		.set_dst_buffer(args.bitstream)
		.set_dst_buffer_offset(0)
		.set_dst_buffer_range(args.bitstream_range)
		.set_src_picture_resource(src_resource)
		.set_setup_reference_slot(&setup_slot)
		.push_next(&mut h265_picture)
		.map_err(Error::from)?;
	if !args.is_idr {
		encode = encode
			.set_reference_slots(&reference_slots)
			.map_err(Error::from)?;
	}
	// SAFETY: identical session/scope/lifetime contract as the H.264 path.
	unsafe {
		device.cmd_encode_video_khr(command_buffer, &encode)?;
	}
	Ok(())
}

struct H265Parameters<'a> {
	profile_tier_level: vk::StdVideoH265ProfileTierLevel,
	dpb_manager: vk::StdVideoH265DecPicBufMgr,
	short_term_ref_pic_set: vk::StdVideoH265ShortTermRefPicSet,
	vps: vk::StdVideoH265VideoParameterSet<'a>,
	sps: vk::StdVideoH265SequenceParameterSet<'a>,
	pps: vk::StdVideoH265PictureParameterSet<'a>,
}

impl H265Parameters<'_> {
	/// Point the parameter sets at this arena's leaf storage. Call after the
	/// value has reached its final stack position; the copies handed to the
	/// driver then reference `self`, not the build frame.
	fn link(&mut self) {
		self.vps.p_profile_tier_level = std::ptr::from_ref(&self.profile_tier_level);
		self.vps.p_dec_pic_buf_mgr = std::ptr::from_ref(&self.dpb_manager);
		self.sps.p_profile_tier_level = std::ptr::from_ref(&self.profile_tier_level);
		self.sps.p_dec_pic_buf_mgr = std::ptr::from_ref(&self.dpb_manager);
		self.sps.p_short_term_ref_pic_set = std::ptr::from_ref(&self.short_term_ref_pic_set);
	}
}

#[allow(clippy::too_many_arguments)]
fn build_h265_parameters(
	settings: &video::VideoEncodeConfig,
	coded: video::VideoExtent,
	dpb_slots: u32,
	ctb_16: bool,
	ctb_32: bool,
	ctb_64: bool,
	ts_4: bool,
	ts_8: bool,
	ts_16: bool,
	ts_32: bool,
) -> H265Parameters<'static> {
	let ctb_log2 = if ctb_16 {
		4
	} else if ctb_32 {
		5
	} else if ctb_64 {
		6
	} else {
		// No advertised CTB size — keep the donor's TGL-fallback geometry;
		// preflight admits at least one size on every qualified device.
		6
	};
	let min_transform_log2 = if ts_4 {
		2
	} else if ts_8 {
		3
	} else {
		4
	};
	let max_transform_log2 = if ctb_log2 >= 5 && ts_32 {
		5
	} else if ts_16 {
		4
	} else if ts_8 {
		3
	} else {
		min_transform_log2
	};
	let mut out = H265Parameters {
		profile_tier_level: vk::StdVideoH265ProfileTierLevel::new(),
		dpb_manager: vk::StdVideoH265DecPicBufMgr::new(),
		short_term_ref_pic_set: vk::StdVideoH265ShortTermRefPicSet::new(),
		vps: vk::StdVideoH265VideoParameterSet::new(),
		sps: vk::StdVideoH265SequenceParameterSet::new(),
		pps: vk::StdVideoH265PictureParameterSet::new(),
	};
	let mut ptl_flags = vk::StdVideoH265ProfileTierLevelFlags::new();
	ptl_flags.set_general_progressive_source_flag(1);
	ptl_flags.set_general_frame_only_constraint_flag(1);
	out.profile_tier_level.flags = ptl_flags;
	out.profile_tier_level.general_profile_idc = match settings.profile {
		video::VideoEncodeProfile::H265 { profile, .. } => h265_profile(profile),
		_ => vk::StdVideoH265ProfileIdc::MAIN,
	};
	out.profile_tier_level.general_level_idc = vk::StdVideoH265LevelIdc::_4_1;
	out.dpb_manager.max_dec_pic_buffering_minus1[0] = dpb_slots.saturating_sub(1) as u8;
	out.dpb_manager.max_num_reorder_pics[0] = 0;
	out.dpb_manager.max_latency_increase_plus1[0] = 0;
	// One flat L0 reference: the immediately preceding I/P picture.
	out.short_term_ref_pic_set.used_by_curr_pic_s0_flag = 1;
	out.short_term_ref_pic_set.num_negative_pics = 1;
	out.short_term_ref_pic_set.num_positive_pics = 0;
	out.short_term_ref_pic_set.delta_poc_s0_minus1[0] = 0;
	let mut vps_flags = vk::StdVideoH265VpsFlags::new();
	vps_flags.set_vps_temporal_id_nesting_flag(1);
	vps_flags.set_vps_sub_layer_ordering_info_present_flag(1);
	out.vps.flags = vps_flags;
	out.vps.vps_video_parameter_set_id = 0;
	out.vps.vps_max_sub_layers_minus1 = 0;
	let mut sps_flags = vk::StdVideoH265SpsFlags::new();
	sps_flags.set_sps_temporal_id_nesting_flag(1);
	sps_flags.set_sps_sub_layer_ordering_info_present_flag(1);
	sps_flags.set_amp_enabled_flag(1);
	sps_flags.set_sample_adaptive_offset_enabled_flag(1);
	let is_range_extension = matches!(
		settings.profile,
		video::VideoEncodeProfile::H265 {
			profile: video::H265Profile::FormatRangeExtensions,
			..
		}
	);
	sps_flags.set_sps_range_extension_flag(u32::from(is_range_extension));
	sps_flags.set_conformance_window_flag(u32::from(
		coded.width != settings.width || coded.height != settings.height,
	));
	out.sps.flags = sps_flags;
	out.sps.chroma_format_idc = vk::StdVideoH265ChromaFormatIdc::_420;
	out.sps.pic_width_in_luma_samples = coded.width;
	out.sps.pic_height_in_luma_samples = coded.height;
	out.sps.sps_video_parameter_set_id = 0;
	out.sps.sps_max_sub_layers_minus1 = 0;
	out.sps.sps_seq_parameter_set_id = 0;
	let (luma_bit_depth, chroma_bit_depth) = match settings.profile {
		video::VideoEncodeProfile::H265 {
			luma_bit_depth,
			chroma_bit_depth,
			..
		} => (luma_bit_depth, chroma_bit_depth),
		_ => (
			video::VideoComponentBitDepth::Eight,
			video::VideoComponentBitDepth::Eight,
		),
	};
	out.sps.bit_depth_luma_minus8 = bit_depth_minus8(luma_bit_depth);
	out.sps.bit_depth_chroma_minus8 = bit_depth_minus8(chroma_bit_depth);
	out.sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	// MinCb stays 16 while CTB/transform geometry follows the capability
	// masks — hard-coding 16 on TGL produces a valid-looking hang.
	const MIN_CODING_BLOCK_LOG2: u32 = 4;
	out.sps.log2_min_luma_coding_block_size_minus3 = (MIN_CODING_BLOCK_LOG2 - 3) as u8;
	out.sps.log2_diff_max_min_luma_coding_block_size = (ctb_log2 - MIN_CODING_BLOCK_LOG2) as u8;
	out.sps.log2_min_luma_transform_block_size_minus2 = (min_transform_log2 - 2) as u8;
	out.sps.log2_diff_max_min_luma_transform_block_size =
		(max_transform_log2 - min_transform_log2) as u8;
	out.sps.max_transform_hierarchy_depth_inter = (ctb_log2 - min_transform_log2).max(1) as u8;
	out.sps.max_transform_hierarchy_depth_intra = 3;
	out.sps.pcm_sample_bit_depth_luma_minus1 = bit_depth_minus8(luma_bit_depth) + 7;
	out.sps.pcm_sample_bit_depth_chroma_minus1 = bit_depth_minus8(chroma_bit_depth) + 7;
	out.sps.log2_min_pcm_luma_coding_block_size_minus3 = (MIN_CODING_BLOCK_LOG2 - 3) as u8;
	out.sps.log2_diff_max_min_pcm_luma_coding_block_size = (ctb_log2 - MIN_CODING_BLOCK_LOG2) as u8;
	out.sps.num_short_term_ref_pic_sets = 1;
	out.sps.conf_win_right_offset = (coded.width - settings.width) / 2;
	out.sps.conf_win_bottom_offset = (coded.height - settings.height) / 2;
	let mut pps_flags = vk::StdVideoH265PpsFlags::new();
	pps_flags.set_cabac_init_present_flag(1);
	pps_flags.set_transform_skip_enabled_flag(1);
	pps_flags.set_cu_qp_delta_enabled_flag(1);
	pps_flags.set_pps_loop_filter_across_slices_enabled_flag(1);
	pps_flags.set_deblocking_filter_control_present_flag(1);
	out.pps.flags = pps_flags;
	out.pps.pps_pic_parameter_set_id = 0;
	out.pps.pps_seq_parameter_set_id = 0;
	out.pps.sps_video_parameter_set_id = 0;
	out.pps.num_ref_idx_l0_default_active_minus1 = 0;
	out.pps.num_ref_idx_l1_default_active_minus1 = 0;
	// Keep the PPS base QP consistent with fixed-QP encoding. Otherwise Xe
	// reconstructs around QP 26 even when the emitted slice advertises QP 10.
	// ABI adaptation matches the independent FFmpeg Vulkan HEVC setup.
	out.pps.init_qp_minus26 = if settings.rate_control == video::VideoRateControl::ConstantQp {
		(settings.constant_qp as i32 - 26) as i8
	} else {
		0
	};
	out.pps.log2_parallel_merge_level_minus2 = 0;
	out
}
