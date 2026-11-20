//! H.264 Vulkan Video encode parameters and picture recording.

use crate::{Error, Result, video};

use super::super::profile::{h264_profile, std_h264_level};
use super::{
	EncodePictureArgs, EncodeSession, bit_depth_minus8, record_coding, reference_resources,
};

pub(super) fn create_parameters(
	session: &EncodeSession,
	settings: &video::VideoEncodeConfig,
	plan: video::VideoEncodePlan,
) -> Result<vk::VideoSessionParametersKHR> {
	let mut quality =
		vk::VideoEncodeQualityLevelInfoKHR::new().set_quality_level(settings.quality_level);
	let level = plan
		.h264_level()
		.ok_or_else(|| Error::invalid_argument("H.264 encode plan lacks a level"))?;
	let sps = build_h264_sps(session.coded_extent, level, settings.profile)?;
	let pps = build_h264_pps();
	let spss = [sps];
	let ppss = [pps];
	let add = vk::VideoEncodeH264SessionParametersAddInfoKHR::new()
		.set_std_sp_ss(&spss)
		.map_err(Error::from)?
		.set_std_pp_ss(&ppss)
		.map_err(Error::from)?;
	let mut codec = vk::VideoEncodeH264SessionParametersCreateInfoKHR::new()
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
	let mut codec = vk::VideoEncodeH264SessionParametersGetInfoKHR::new()
		.set_write_std_sps(vk::TRUE)
		.set_write_std_pps(vk::TRUE)
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
	let mut current_ref = vk::StdVideoEncodeH264ReferenceInfo::new();
	let mut previous_ref = vk::StdVideoEncodeH264ReferenceInfo::new();
	let mut current = vk::VideoEncodeH264DpbSlotInfoKHR::new();
	let mut previous = vk::VideoEncodeH264DpbSlotInfoKHR::new();
	let mut slots = [
		vk::VideoReferenceSlotInfoKHR::new(),
		vk::VideoReferenceSlotInfoKHR::new(),
	];
	// Bind the current picture as inactive and list the single active
	// reference only for a predicted picture.
	current_ref.primary_pic_type = if args.is_idr {
		vk::StdVideoH264PictureType::IDR
	} else {
		vk::StdVideoH264PictureType::P
	};
	current_ref.frame_num = if args.is_idr { 0 } else { args.frame_index };
	current_ref.pic_order_cnt = if args.is_idr {
		0
	} else {
		args.frame_index as i32
	};
	current = current.set_std_reference_info(&current_ref);
	let slot0 = vk::VideoReferenceSlotInfoKHR::new()
		.set_slot_index(-1)
		.set_picture_resource(&current_resource)
		.push_next(&mut current)
		.map_err(Error::from)?;
	if !args.is_idr {
		previous_ref.primary_pic_type = vk::StdVideoH264PictureType::P;
		previous_ref.frame_num = args.frame_index - 1;
		previous_ref.pic_order_cnt = args.frame_index as i32 - 1;
		previous = previous.set_std_reference_info(&previous_ref);
		let slot1 = vk::VideoReferenceSlotInfoKHR::new()
			.set_slot_index(args.previous_dpb_slot as i32)
			.set_picture_resource(&previous_resource)
			.push_next(&mut previous)
			.map_err(Error::from)?;
		slots[1] = slot1;
	}
	slots[0] = slot0;
	let layer = vk::VideoEncodeH264RateControlLayerInfoKHR::new()
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
			vk::VideoEncodeH264RateControlInfoKHR::new()
				.set_gop_frame_count(args.gop_size)
				.set_idr_period(args.gop_size)
				.set_consecutive_b_frame_count(args.max_b_frames)
				.set_temporal_layer_count(1)
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
	// B-picture *emission* needs display-order reorder and a second DPB
	// reference — the donor budgets maxBFrames in rate control but encodes
	// only I/P slices.
	let mut ref_lists = vk::StdVideoEncodeH264ReferenceListsInfo::new();
	ref_lists.num_ref_idx_l0_active_minus1 = 0;
	ref_lists.num_ref_idx_l1_active_minus1 = 0;
	ref_lists
		.ref_pic_list0
		.fill(vk::STD_VIDEO_H264_NO_REFERENCE_PICTURE);
	ref_lists
		.ref_pic_list1
		.fill(vk::STD_VIDEO_H264_NO_REFERENCE_PICTURE);
	if !args.is_idr {
		ref_lists.ref_pic_list0[0] = args.previous_dpb_slot as u8;
	}
	let mut pic_flags = vk::StdVideoEncodeH264PictureInfoFlags::new();
	pic_flags.set_idr_pic_flag(u32::from(args.is_idr));
	pic_flags.set_is_reference(1);
	pic_flags.set_no_output_of_prior_pics_flag(u32::from(args.is_idr));
	pic_flags.set_long_term_reference_flag(0);
	pic_flags.set_adaptive_ref_pic_marking_mode_flag(0);
	let mut pic_info = vk::StdVideoEncodeH264PictureInfo::new();
	pic_info.flags = pic_flags;
	pic_info.seq_parameter_set_id = 0;
	pic_info.pic_parameter_set_id = 0;
	pic_info.idr_pic_id = args.idr_pic_id;
	pic_info.primary_pic_type = if args.is_idr {
		vk::StdVideoH264PictureType::IDR
	} else {
		vk::StdVideoH264PictureType::P
	};
	pic_info.frame_num = if args.is_idr { 0 } else { args.frame_index };
	pic_info.pic_order_cnt = if args.is_idr {
		0
	} else {
		args.frame_index as i32
	};
	pic_info.p_ref_lists = &ref_lists;

	let mut slice_flags = vk::StdVideoEncodeH264SliceHeaderFlags::new();
	slice_flags.set_direct_spatial_mv_pred_flag(0);
	slice_flags.set_num_ref_idx_active_override_flag(0);
	let mut slice_header = vk::StdVideoEncodeH264SliceHeader::new();
	slice_header.flags = slice_flags;
	slice_header.first_mb_in_slice = 0;
	slice_header.slice_type = if args.is_idr {
		vk::StdVideoH264SliceType::I
	} else {
		vk::StdVideoH264SliceType::P
	};
	slice_header.cabac_init_idc = vk::StdVideoH264CabacInitIdc::_0;
	slice_header.disable_deblocking_filter_idc = vk::StdVideoH264DisableDeblockingFilterIdc::DISABLED;
	slice_header.slice_alpha_c0_offset_div2 = 0;
	slice_header.slice_beta_offset_div2 = 0;
	let constant_qp =
		if args.rate_control_mode == vk::VideoEncodeRateControlModeFlagBitsKHR::DISABLED_BIT_KHR {
			args.constant_qp as i32
		} else {
			0
		};
	// PPS pic_init_qp_minus26 is zero. Keep the StdVideo slice and Vulkan
	// QP controls consistent; a rate-controlled operation requires constantQp=0.
	slice_header.slice_qp_delta = (constant_qp - 26) as i8;

	let nalu = vk::VideoEncodeH264NaluSliceInfoKHR::new()
		.set_constant_qp(constant_qp)
		.set_std_slice_header(&slice_header);
	let nalus = std::slice::from_ref(&nalu);
	let mut h264_picture = vk::VideoEncodeH264PictureInfoKHR::new()
		.set_nalu_slice_entries(nalus)
		.map_err(Error::from)?
		.set_std_picture_info(&pic_info)
		.set_generate_prefix_nalu(vk::FALSE);

	// Setup slot: the current DPB layer being written this frame.
	let mut current_ref = vk::StdVideoEncodeH264ReferenceInfo::new();
	current_ref.primary_pic_type = if args.is_idr {
		vk::StdVideoH264PictureType::IDR
	} else {
		vk::StdVideoH264PictureType::P
	};
	current_ref.frame_num = pic_info.frame_num;
	current_ref.pic_order_cnt = pic_info.pic_order_cnt;
	let mut current_slot =
		vk::VideoEncodeH264DpbSlotInfoKHR::new().set_std_reference_info(&current_ref);
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

	let mut previous_ref = vk::StdVideoEncodeH264ReferenceInfo::new();
	previous_ref.primary_pic_type = vk::StdVideoH264PictureType::P;
	previous_ref.frame_num = args.frame_index.saturating_sub(1);
	previous_ref.pic_order_cnt = args.frame_index as i32 - 1;
	let mut previous_slot_info =
		vk::VideoEncodeH264DpbSlotInfoKHR::new().set_std_reference_info(&previous_ref);
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
		.push_next(&mut h264_picture)
		.map_err(Error::from)?;
	if !args.is_idr {
		encode = encode
			.set_reference_slots(&reference_slots)
			.map_err(Error::from)?;
	}
	// SAFETY: session coding scope is open; every referenced slot resource,
	// parameter chain, and bitstream buffer outlives submission via the
	// recorded-command retention on the engine timeline.
	unsafe {
		device.cmd_encode_video_khr(command_buffer, &encode)?;
	}
	Ok(())
}

#[allow(clippy::field_reassign_with_default)]
fn build_h264_sps(
	coded: video::VideoExtent,
	level: u32,
	profile: video::VideoEncodeProfile,
) -> Result<vk::StdVideoH264SequenceParameterSet<'static>> {
	let (profile_idc, luma_bit_depth, chroma_bit_depth) = match profile {
		video::VideoEncodeProfile::H264 {
			profile,
			luma_bit_depth,
			chroma_bit_depth,
			..
		} => (profile, luma_bit_depth, chroma_bit_depth),
		_ => (
			video::H264Profile::High,
			video::VideoComponentBitDepth::Eight,
			video::VideoComponentBitDepth::Eight,
		),
	};
	let mut flags = vk::StdVideoH264SpsFlags::new();
	flags.set_direct_8x8_inference_flag(1);
	flags.set_frame_mbs_only_flag(1);
	let mut sps = vk::StdVideoH264SequenceParameterSet::new();
	sps.flags = flags;
	sps.profile_idc = h264_profile(profile_idc);
	sps.level_idc = std_h264_level(level)?;
	sps.chroma_format_idc = vk::StdVideoH264ChromaFormatIdc::_420;
	sps.seq_parameter_set_id = 0;
	sps.bit_depth_luma_minus8 = bit_depth_minus8(luma_bit_depth);
	sps.bit_depth_chroma_minus8 = bit_depth_minus8(chroma_bit_depth);
	sps.log2_max_frame_num_minus4 = 4;
	sps.pic_order_cnt_type = vk::StdVideoH264PocType::_0;
	sps.offset_for_non_ref_pic = 0;
	sps.offset_for_top_to_bottom_field = 0;
	sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	sps.num_ref_frames_in_pic_order_cnt_cycle = 0;
	sps.max_num_ref_frames = 1;
	let width_mbs = coded.width.div_ceil(16);
	let height_mbs = coded.height.div_ceil(16);
	sps.pic_width_in_mbs_minus1 = width_mbs.saturating_sub(1);
	sps.pic_height_in_map_units_minus1 = height_mbs.saturating_sub(1);
	Ok(sps)
}

#[allow(clippy::field_reassign_with_default)]
fn build_h264_pps() -> vk::StdVideoH264PictureParameterSet<'static> {
	let mut flags = vk::StdVideoH264PpsFlags::new();
	flags.set_transform_8x8_mode_flag(1);
	flags.set_deblocking_filter_control_present_flag(1);
	flags.set_entropy_coding_mode_flag(1); // CABAC
	let mut pps = vk::StdVideoH264PictureParameterSet::new();
	pps.flags = flags;
	pps.seq_parameter_set_id = 0;
	pps.pic_parameter_set_id = 0;
	pps.num_ref_idx_l0_default_active_minus1 = 0;
	pps.num_ref_idx_l1_default_active_minus1 = 0;
	pps.weighted_bipred_idc = vk::StdVideoH264WeightedBipredIdc::DEFAULT;
	pps
}
