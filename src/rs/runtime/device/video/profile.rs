//! Private Vulkan Video profile, format, and StdVideo translation.

use crate::runtime::device::video::*;

pub(super) struct DecodeCapabilityQuery {
	pub(super) capabilities: video::VideoDecodeCapabilities,
	pub(super) std_header_version: ash::vk::ExtensionProperties,
}

pub(in crate::runtime) fn query_decode_capabilities(
	instance: &Instance,
	physical: &DevicePhysical,
	profile: video::VideoDecodeProfile,
) -> Result<video::VideoDecodeCapabilities> {
	Ok(query_decode_details(instance, physical, profile)?.capabilities)
}

pub(super) fn query_decode_details(
	instance: &Instance,
	physical: &DevicePhysical,
	profile: video::VideoDecodeProfile,
) -> Result<DecodeCapabilityQuery> {
	ensure_extension_advertised(physical, profile)?;
	let loader = ash::khr::video_queue::Instance::new(instance.entry(), instance.raw());

	match profile {
		video::VideoDecodeProfile::H264 {
			profile: codec_profile,
			picture_layout,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec_profile = ash::vk::VideoDecodeH264ProfileInfoKHR::default()
				.std_profile_idc(h264_profile(codec_profile))
				.picture_layout(h264_picture_layout(picture_layout));
			let vk_profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::DECODE_H264,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec_profile);
			let mut decode = ash::vk::VideoDecodeCapabilitiesKHR::default();
			let mut codec = ash::vk::VideoDecodeH264CapabilitiesKHR::default();
			let common = {
				let mut capabilities = ash::vk::VideoCapabilitiesKHR::default()
					.push_next(&mut decode)
					.push_next(&mut codec);
				query(&loader, physical.handle, &vk_profile, &mut capabilities)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(DecodeCapabilityQuery {
				capabilities: convert_capabilities(
					profile,
					common,
					&decode,
					video::VideoDecodeLevel::H264(codec.max_level_idc),
					Some((
						codec.field_offset_granularity.x,
						codec.field_offset_granularity.y,
					)),
				),
				std_header_version: common.std_header_version,
			})
		}
		video::VideoDecodeProfile::H265 {
			profile: codec_profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec_profile = ash::vk::VideoDecodeH265ProfileInfoKHR::default()
				.std_profile_idc(h265_profile(codec_profile));
			let vk_profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::DECODE_H265,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec_profile);
			let mut decode = ash::vk::VideoDecodeCapabilitiesKHR::default();
			let mut codec = ash::vk::VideoDecodeH265CapabilitiesKHR::default();
			let common = {
				let mut capabilities = ash::vk::VideoCapabilitiesKHR::default()
					.push_next(&mut decode)
					.push_next(&mut codec);
				query(&loader, physical.handle, &vk_profile, &mut capabilities)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(DecodeCapabilityQuery {
				capabilities: convert_capabilities(
					profile,
					common,
					&decode,
					video::VideoDecodeLevel::H265(codec.max_level_idc),
					None,
				),
				std_header_version: common.std_header_version,
			})
		}
		video::VideoDecodeProfile::Av1 {
			profile: codec_profile,
			film_grain_support,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec_profile = ash::vk::VideoDecodeAV1ProfileInfoKHR::default()
				.std_profile(av1_profile(codec_profile))
				.film_grain_support(film_grain_support);
			let vk_profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::DECODE_AV1,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec_profile);
			let mut decode = ash::vk::VideoDecodeCapabilitiesKHR::default();
			let mut codec = ash::vk::VideoDecodeAV1CapabilitiesKHR::default();
			let common = {
				let mut capabilities = ash::vk::VideoCapabilitiesKHR::default()
					.push_next(&mut decode)
					.push_next(&mut codec);
				query(&loader, physical.handle, &vk_profile, &mut capabilities)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(DecodeCapabilityQuery {
				capabilities: convert_capabilities(
					profile,
					common,
					&decode,
					video::VideoDecodeLevel::Av1(codec.max_level),
					None,
				),
				std_header_version: common.std_header_version,
			})
		}
		video::VideoDecodeProfile::Vp9 {
			profile: codec_profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec_profile = ash_vp9::vk::VideoDecodeVP9ProfileInfoKHR::default()
				.std_profile(vp9_profile(codec_profile));
			let mut vk_profile = common_profile(
				vp9_operation(),
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			);
			vk_profile.p_next = std::ptr::from_mut(&mut codec_profile).cast();
			let mut codec = ash_vp9::vk::VideoDecodeVP9CapabilitiesKHR::default();
			let mut decode = ash::vk::VideoDecodeCapabilitiesKHR {
				p_next: std::ptr::from_mut(&mut codec).cast(),
				..Default::default()
			};
			let common = {
				let mut capabilities = ash::vk::VideoCapabilitiesKHR {
					p_next: std::ptr::from_mut(&mut decode).cast(),
					..Default::default()
				};
				query(&loader, physical.handle, &vk_profile, &mut capabilities)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(DecodeCapabilityQuery {
				capabilities: convert_capabilities(
					profile,
					common,
					&decode,
					video::VideoDecodeLevel::Vp9(codec.max_level),
					None,
				),
				std_header_version: common.std_header_version,
			})
		}
	}
}

pub(in crate::runtime) fn query_decode_formats(
	instance: &Instance,
	physical: &DevicePhysical,
	profile: video::VideoDecodeProfile,
) -> Result<video::VideoDecodeFormats> {
	ensure_extension_advertised(physical, profile)?;
	let loader = ash::khr::video_queue::Instance::new(instance.entry(), instance.raw());
	with_decode_profile(profile, |vk_profile| {
		let output = query_formats(
			&loader,
			physical.handle,
			vk_profile,
			ash::vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR,
		)?;
		let dpb = query_formats(
			&loader,
			physical.handle,
			vk_profile,
			ash::vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR,
		)?;
		let (output, unrecognized_output_formats) = convert_formats(&output);
		let (dpb, unrecognized_dpb_formats) = convert_formats(&dpb);
		Ok(video::VideoDecodeFormats {
			profile,
			output,
			dpb,
			unrecognized_output_formats,
			unrecognized_dpb_formats,
		})
	})
}

pub(in crate::runtime) fn query_encode_capabilities(
	instance: &Instance,
	physical: &DevicePhysical,
	profile: video::VideoEncodeProfile,
) -> Result<video::VideoEncodeCapabilities> {
	ensure_encode_extension_advertised(physical, profile)?;
	let loader = ash::khr::video_queue::Instance::new(instance.entry(), instance.raw());
	match profile {
		video::VideoEncodeProfile::H264 {
			profile: codec_profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec_profile = ash::vk::VideoEncodeH264ProfileInfoKHR::default()
				.std_profile_idc(h264_profile(codec_profile));
			let vk_profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec_profile);
			let mut encode = ash::vk::VideoEncodeCapabilitiesKHR::default();
			let mut codec = ash::vk::VideoEncodeH264CapabilitiesKHR::default();
			let common = {
				let mut capabilities = ash::vk::VideoCapabilitiesKHR::default()
					.push_next(&mut encode)
					.push_next(&mut codec);
				query(&loader, physical.handle, &vk_profile, &mut capabilities)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(convert_encode_capabilities(
				profile,
				common,
				&encode,
				video::VideoEncodeCodecCapabilities::H264 {
					max_level: codec.max_level_idc,
					max_slice_count: codec.max_slice_count,
					max_p_l0_references: codec.max_p_picture_l0_reference_count,
					max_b_l0_references: codec.max_b_picture_l0_reference_count,
					max_l1_references: codec.max_l1_reference_count,
					max_temporal_layers: codec.max_temporal_layer_count,
					min_qp: codec.min_qp,
					max_qp: codec.max_qp,
				},
			))
		}
		video::VideoEncodeProfile::H265 {
			profile: codec_profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec_profile = ash::vk::VideoEncodeH265ProfileInfoKHR::default()
				.std_profile_idc(h265_profile(codec_profile));
			let vk_profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::ENCODE_H265,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec_profile);
			let mut encode = ash::vk::VideoEncodeCapabilitiesKHR::default();
			let mut codec = ash::vk::VideoEncodeH265CapabilitiesKHR::default();
			let common = {
				let mut capabilities = ash::vk::VideoCapabilitiesKHR::default()
					.push_next(&mut encode)
					.push_next(&mut codec);
				query(&loader, physical.handle, &vk_profile, &mut capabilities)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(convert_encode_capabilities(
				profile,
				common,
				&encode,
				video::VideoEncodeCodecCapabilities::H265 {
					max_level: codec.max_level_idc,
					max_slice_segment_count: codec.max_slice_segment_count,
					max_tiles: extent(codec.max_tiles),
					ctb_size_16: codec
						.ctb_sizes
						.contains(ash::vk::VideoEncodeH265CtbSizeFlagsKHR::TYPE_16),
					ctb_size_32: codec
						.ctb_sizes
						.contains(ash::vk::VideoEncodeH265CtbSizeFlagsKHR::TYPE_32),
					ctb_size_64: codec
						.ctb_sizes
						.contains(ash::vk::VideoEncodeH265CtbSizeFlagsKHR::TYPE_64),
					transform_size_4: codec
						.transform_block_sizes
						.contains(ash::vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_4),
					transform_size_8: codec
						.transform_block_sizes
						.contains(ash::vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_8),
					transform_size_16: codec
						.transform_block_sizes
						.contains(ash::vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_16),
					transform_size_32: codec
						.transform_block_sizes
						.contains(ash::vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_32),
					max_p_l0_references: codec.max_p_picture_l0_reference_count,
					max_b_l0_references: codec.max_b_picture_l0_reference_count,
					max_l1_references: codec.max_l1_reference_count,
					max_sub_layers: codec.max_sub_layer_count,
					min_qp: codec.min_qp,
					max_qp: codec.max_qp,
				},
			))
		}
	}
}

pub(in crate::runtime) fn query_encode_formats(
	instance: &Instance,
	physical: &DevicePhysical,
	profile: video::VideoEncodeProfile,
) -> Result<video::VideoEncodeFormats> {
	ensure_encode_extension_advertised(physical, profile)?;
	let loader = ash::khr::video_queue::Instance::new(instance.entry(), instance.raw());
	with_encode_profile(profile, |vk_profile| {
		let input = query_formats(
			&loader,
			physical.handle,
			vk_profile,
			ash::vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR,
		)?;
		let dpb = query_formats(
			&loader,
			physical.handle,
			vk_profile,
			ash::vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
		)?;
		let (input, unrecognized_input_formats) = convert_formats(&input);
		let (dpb, unrecognized_dpb_formats) = convert_formats(&dpb);
		Ok(video::VideoEncodeFormats {
			profile,
			input,
			dpb,
			unrecognized_input_formats,
			unrecognized_dpb_formats,
		})
	})
}

pub(super) struct StdH265Hrd {
	_nal: Box<[ash::vk::native::StdVideoH265SubLayerHrdParameters]>,
	_vcl: Box<[ash::vk::native::StdVideoH265SubLayerHrdParameters]>,
	value: ash::vk::native::StdVideoH265HrdParameters,
}

impl StdH265Hrd {
	pub(super) fn as_ptr(&self) -> *const ash::vk::native::StdVideoH265HrdParameters {
		&self.value
	}
}

pub(super) fn std_h265_profile_tier_level(
	profile: &video::H265ProfileTierLevel,
) -> Result<ash::vk::native::StdVideoH265ProfileTierLevel> {
	let profile_idc = match profile.profile_idc {
		1 | 2 | 3 | 4 | 9 => profile.profile_idc,
		value => {
			return Err(Error::missing_capability(format!(
				"H.265 profile_idc {value} has no supported StdVideo mapping"
			)));
		}
	};
	Ok(ash::vk::native::StdVideoH265ProfileTierLevel {
		flags: ash::vk::native::StdVideoH265ProfileTierLevelFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoH265ProfileTierLevelFlags::new_bitfield_1(
				u32::from(profile.high_tier),
				u32::from(profile.progressive_source),
				u32::from(profile.interlaced_source),
				u32::from(profile.non_packed_constraint),
				u32::from(profile.frame_only_constraint),
			),
			__bindgen_padding_0: [0; 3],
		},
		general_profile_idc: profile_idc,
		general_level_idc: std_h265_level(profile.level_idc)?,
	})
}

pub(super) fn std_h265_level(level_idc: u32) -> Result<ash::vk::native::StdVideoH265LevelIdc> {
	match level_idc {
		30 => Ok(0),
		60 => Ok(1),
		63 => Ok(2),
		90 => Ok(3),
		93 => Ok(4),
		120 => Ok(5),
		123 => Ok(6),
		150 => Ok(7),
		153 => Ok(8),
		156 => Ok(9),
		180 => Ok(10),
		183 => Ok(11),
		186 => Ok(12),
		value => Err(Error::missing_capability(format!(
			"H.265 level_idc {value} has no Khronos StdVideo mapping"
		))),
	}
}

pub(super) fn std_h265_dpb(
	dpb: &video::H265DecodedPictureBuffer,
) -> Result<ash::vk::native::StdVideoH265DecPicBufMgr> {
	let mut max_dec_pic_buffering_minus1 = [0_u8; 7];
	let mut max_num_reorder_pics = [0_u8; 7];
	for index in 0..7 {
		max_dec_pic_buffering_minus1[index] =
			u8::try_from(dpb.max_decoded_picture_buffering_minus_1[index])
				.map_err(|_| Error::data_loss("H.265 DPB capacity exceeds StdVideo storage"))?;
		max_num_reorder_pics[index] = u8::try_from(dpb.max_num_reorder_pictures[index])
			.map_err(|_| Error::data_loss("H.265 reorder count exceeds StdVideo storage"))?;
	}
	Ok(ash::vk::native::StdVideoH265DecPicBufMgr {
		max_latency_increase_plus1: dpb.max_latency_increase_plus_1,
		max_dec_pic_buffering_minus1,
		max_num_reorder_pics,
	})
}

pub(super) fn std_h265_hrd(hrd: &video::H265HrdParameters) -> Result<StdH265Hrd> {
	if hrd.sub_layers.len() > 7 {
		return Err(Error::data_loss("H.265 HRD sub-layer count exceeds seven"));
	}
	let mut cpb_cnt_minus1 = [0_u8; 7];
	let mut elemental_duration_in_tc_minus1 = [0_u16; 7];
	let mut fixed_general = 0_u32;
	let mut fixed_within = 0_u32;
	let mut low_delay = 0_u32;
	let mut nal = Vec::with_capacity(hrd.sub_layers.len());
	let mut vcl = Vec::with_capacity(hrd.sub_layers.len());
	for (index, layer) in hrd.sub_layers.iter().enumerate() {
		fixed_general |= u32::from(layer.fixed_picture_rate_general) << index;
		fixed_within |= u32::from(layer.fixed_picture_rate_within_cvs) << index;
		low_delay |= u32::from(layer.low_delay) << index;
		elemental_duration_in_tc_minus1[index] = u16::try_from(layer.elemental_duration_in_tc_minus_1)
			.map_err(|_| Error::data_loss("H.265 HRD elemental duration exceeds StdVideo storage"))?;
		let entries = if hrd.nal_parameters_present {
			&layer.nal_entries
		} else {
			&layer.vcl_entries
		};
		let count_minus_1 = entries
			.len()
			.checked_sub(1)
			.ok_or_else(|| Error::data_loss("H.265 HRD sub-layer has no CPB entries"))?;
		cpb_cnt_minus1[index] = u8::try_from(count_minus_1)
			.map_err(|_| Error::data_loss("H.265 HRD CPB count exceeds StdVideo storage"))?;
		if hrd.nal_parameters_present {
			nal.push(std_h265_sub_layer_hrd(&layer.nal_entries)?);
		}
		if hrd.vcl_parameters_present {
			if hrd.nal_parameters_present && layer.nal_entries.len() != layer.vcl_entries.len() {
				return Err(Error::data_loss(
					"H.265 NAL and VCL HRD tables use different CPB counts",
				));
			}
			vcl.push(std_h265_sub_layer_hrd(&layer.vcl_entries)?);
		}
	}
	let nal = nal.into_boxed_slice();
	let vcl = vcl.into_boxed_slice();
	let value = ash::vk::native::StdVideoH265HrdParameters {
		flags: ash::vk::native::StdVideoH265HrdFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoH265HrdFlags::new_bitfield_1(
				u32::from(hrd.nal_parameters_present),
				u32::from(hrd.vcl_parameters_present),
				u32::from(hrd.sub_picture_parameters_present),
				u32::from(hrd.sub_picture_cpb_parameters_in_picture_timing_sei),
				fixed_general,
				fixed_within,
				low_delay,
			),
		},
		tick_divisor_minus2: hrd.tick_divisor_minus_2,
		du_cpb_removal_delay_increment_length_minus1: hrd.du_cpb_removal_delay_increment_length_minus_1,
		dpb_output_delay_du_length_minus1: hrd.dpb_output_delay_du_length_minus_1,
		bit_rate_scale: hrd.bit_rate_scale,
		cpb_size_scale: hrd.cpb_size_scale,
		cpb_size_du_scale: hrd.cpb_size_du_scale,
		initial_cpb_removal_delay_length_minus1: hrd.initial_cpb_removal_delay_length_minus_1,
		au_cpb_removal_delay_length_minus1: hrd.au_cpb_removal_delay_length_minus_1,
		dpb_output_delay_length_minus1: hrd.dpb_output_delay_length_minus_1,
		cpb_cnt_minus1,
		elemental_duration_in_tc_minus1,
		reserved: [0; 3],
		pSubLayerHrdParametersNal: if nal.is_empty() {
			std::ptr::null()
		} else {
			nal.as_ptr()
		},
		pSubLayerHrdParametersVcl: if vcl.is_empty() {
			std::ptr::null()
		} else {
			vcl.as_ptr()
		},
	};
	Ok(StdH265Hrd {
		_nal: nal,
		_vcl: vcl,
		value,
	})
}

pub(super) fn std_h265_sub_layer_hrd(
	entries: &[video::H265CpbEntry],
) -> Result<ash::vk::native::StdVideoH265SubLayerHrdParameters> {
	if entries.is_empty() || entries.len() > 32 {
		return Err(Error::data_loss("H.265 HRD CPB count is outside 1..=32"));
	}
	let mut result = ash::vk::native::StdVideoH265SubLayerHrdParameters {
		bit_rate_value_minus1: [0; 32],
		cpb_size_value_minus1: [0; 32],
		cpb_size_du_value_minus1: [0; 32],
		bit_rate_du_value_minus1: [0; 32],
		cbr_flag: 0,
	};
	for (index, entry) in entries.iter().enumerate() {
		result.bit_rate_value_minus1[index] = entry.bit_rate_value_minus_1;
		result.cpb_size_value_minus1[index] = entry.cpb_size_value_minus_1;
		result.cpb_size_du_value_minus1[index] = entry.cpb_size_du_value_minus_1;
		result.bit_rate_du_value_minus1[index] = entry.bit_rate_du_value_minus_1;
		result.cbr_flag |= u32::from(entry.constant_bit_rate) << index;
	}
	Ok(result)
}

pub(super) fn std_h265_vps(
	vps: &video::H265VideoParameterSet,
	profile: &ash::vk::native::StdVideoH265ProfileTierLevel,
	dpb: &ash::vk::native::StdVideoH265DecPicBufMgr,
	hrd: *const ash::vk::native::StdVideoH265HrdParameters,
) -> Result<ash::vk::native::StdVideoH265VideoParameterSet> {
	let timing = vps.timing.unwrap_or_default();
	Ok(ash::vk::native::StdVideoH265VideoParameterSet {
		flags: ash::vk::native::StdVideoH265VpsFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoH265VpsFlags::new_bitfield_1(
				u32::from(vps.temporal_id_nesting),
				u32::from(vps.sub_layer_ordering_info_present),
				u32::from(vps.timing.is_some()),
				u32::from(timing.num_ticks_poc_diff_one_minus_1.is_some()),
			),
			__bindgen_padding_0: [0; 3],
		},
		vps_video_parameter_set_id: u8::try_from(vps.id)
			.map_err(|_| Error::data_loss("H.265 VPS id exceeds StdVideo storage"))?,
		vps_max_sub_layers_minus1: u8::try_from(vps.max_sub_layers_minus_1)
			.map_err(|_| Error::data_loss("H.265 VPS sub-layer count exceeds StdVideo storage"))?,
		reserved1: 0,
		reserved2: 0,
		vps_num_units_in_tick: timing.num_units_in_tick,
		vps_time_scale: timing.time_scale,
		vps_num_ticks_poc_diff_one_minus1: timing.num_ticks_poc_diff_one_minus_1.unwrap_or(0),
		reserved3: 0,
		pDecPicBufMgr: dpb,
		pHrdParameters: hrd,
		pProfileTierLevel: profile,
	})
}

pub(super) fn std_h265_sps(
	sps: &video::H265SequenceParameterSet,
	profile: &ash::vk::native::StdVideoH265ProfileTierLevel,
	dpb: &ash::vk::native::StdVideoH265DecPicBufMgr,
	scaling: *const ash::vk::native::StdVideoH265ScalingLists,
	short_term: &[ash::vk::native::StdVideoH265ShortTermRefPicSet],
	long_term: Option<&ash::vk::native::StdVideoH265LongTermRefPicsSps>,
	vui: Option<&ash::vk::native::StdVideoH265SequenceParameterSetVui>,
) -> Result<ash::vk::native::StdVideoH265SequenceParameterSet> {
	let pcm = sps.pcm.unwrap_or_default();
	Ok(ash::vk::native::StdVideoH265SequenceParameterSet {
		flags: ash::vk::native::StdVideoH265SpsFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoH265SpsFlags::new_bitfield_1(
				u32::from(sps.temporal_id_nesting),
				u32::from(sps.separate_colour_plane),
				u32::from(sps.conformance_window != [0; 4]),
				u32::from(sps.sub_layer_ordering_info_present),
				u32::from(sps.scaling_list_enabled),
				u32::from(sps.scaling_lists.is_some()),
				u32::from(sps.asymmetric_motion_partitions_enabled),
				u32::from(sps.sample_adaptive_offset_enabled),
				u32::from(sps.pcm.is_some()),
				u32::from(pcm.loop_filter_disabled),
				u32::from(!sps.long_term_reference_pictures.is_empty()),
				u32::from(sps.temporal_mvp_enabled),
				u32::from(sps.strong_intra_smoothing_enabled),
				u32::from(sps.vui.is_some()),
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
			),
		},
		chroma_format_idc: sps.chroma_format_idc,
		pic_width_in_luma_samples: sps.coded_width,
		pic_height_in_luma_samples: sps.coded_height,
		sps_video_parameter_set_id: h265_u8(sps.video_parameter_set_id, "VPS id")?,
		sps_max_sub_layers_minus1: h265_u8(sps.max_sub_layers_minus_1, "sub-layer count")?,
		sps_seq_parameter_set_id: h265_u8(sps.id, "SPS id")?,
		bit_depth_luma_minus8: h265_u8(sps.bit_depth_luma_minus_8, "luma bit depth")?,
		bit_depth_chroma_minus8: h265_u8(sps.bit_depth_chroma_minus_8, "chroma bit depth")?,
		log2_max_pic_order_cnt_lsb_minus4: h265_u8(
			sps.log2_max_pic_order_count_lsb_minus_4,
			"POC width",
		)?,
		log2_min_luma_coding_block_size_minus3: h265_u8(
			sps.log2_min_luma_coding_block_size_minus_3,
			"minimum coding-block size",
		)?,
		log2_diff_max_min_luma_coding_block_size: h265_u8(
			sps.log2_diff_max_min_luma_coding_block_size,
			"coding-block size difference",
		)?,
		log2_min_luma_transform_block_size_minus2: h265_u8(
			sps.log2_min_luma_transform_block_size_minus_2,
			"minimum transform-block size",
		)?,
		log2_diff_max_min_luma_transform_block_size: h265_u8(
			sps.log2_diff_max_min_luma_transform_block_size,
			"transform-block size difference",
		)?,
		max_transform_hierarchy_depth_inter: h265_u8(
			sps.max_transform_hierarchy_depth_inter,
			"inter transform depth",
		)?,
		max_transform_hierarchy_depth_intra: h265_u8(
			sps.max_transform_hierarchy_depth_intra,
			"intra transform depth",
		)?,
		num_short_term_ref_pic_sets: u8::try_from(short_term.len())
			.map_err(|_| Error::data_loss("H.265 short-term RPS count exceeds StdVideo storage"))?,
		num_long_term_ref_pics_sps: u8::try_from(sps.long_term_reference_pictures.len())
			.map_err(|_| Error::data_loss("H.265 long-term RPS count exceeds StdVideo storage"))?,
		pcm_sample_bit_depth_luma_minus1: pcm.sample_bit_depth_luma_minus_1,
		pcm_sample_bit_depth_chroma_minus1: pcm.sample_bit_depth_chroma_minus_1,
		log2_min_pcm_luma_coding_block_size_minus3: h265_u8(
			pcm.log2_min_luma_coding_block_size_minus_3,
			"minimum PCM block size",
		)?,
		log2_diff_max_min_pcm_luma_coding_block_size: h265_u8(
			pcm.log2_diff_max_min_luma_coding_block_size,
			"PCM block-size difference",
		)?,
		reserved1: 0,
		reserved2: 0,
		palette_max_size: 0,
		delta_palette_max_predictor_size: 0,
		motion_vector_resolution_control_idc: 0,
		sps_num_palette_predictor_initializers_minus1: 0,
		conf_win_left_offset: sps.conformance_window[0],
		conf_win_right_offset: sps.conformance_window[1],
		conf_win_top_offset: sps.conformance_window[2],
		conf_win_bottom_offset: sps.conformance_window[3],
		pProfileTierLevel: profile,
		pDecPicBufMgr: dpb,
		pScalingLists: scaling,
		pShortTermRefPicSet: if short_term.is_empty() {
			std::ptr::null()
		} else {
			short_term.as_ptr()
		},
		pLongTermRefPicsSps: long_term.map_or(std::ptr::null(), |value| value as *const _),
		pSequenceParameterSetVui: vui.map_or(std::ptr::null(), |value| value as *const _),
		pPredictorPaletteEntries: std::ptr::null(),
	})
}

pub(super) fn std_h265_short_term_reference_set(
	set: &video::H265ShortTermReferencePictureSet,
) -> Result<ash::vk::native::StdVideoH265ShortTermRefPicSet> {
	if set.negative_delta_poc_minus_1.len() > 16 || set.positive_delta_poc_minus_1.len() > 16 {
		return Err(Error::data_loss(
			"H.265 short-term RPS exceeds StdVideo's sixteen-entry arrays",
		));
	}
	let mut delta_poc_s0_minus1 = [0_u16; 16];
	let mut delta_poc_s1_minus1 = [0_u16; 16];
	for (output, input) in delta_poc_s0_minus1
		.iter_mut()
		.zip(&set.negative_delta_poc_minus_1)
	{
		*output = u16::try_from(*input)
			.map_err(|_| Error::data_loss("H.265 negative delta POC exceeds StdVideo storage"))?;
	}
	for (output, input) in delta_poc_s1_minus1
		.iter_mut()
		.zip(&set.positive_delta_poc_minus_1)
	{
		*output = u16::try_from(*input)
			.map_err(|_| Error::data_loss("H.265 positive delta POC exceeds StdVideo storage"))?;
	}
	Ok(ash::vk::native::StdVideoH265ShortTermRefPicSet {
		flags: ash::vk::native::StdVideoH265ShortTermRefPicSetFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoH265ShortTermRefPicSetFlags::new_bitfield_1(
				u32::from(set.inter_ref_pic_set_prediction),
				u32::from(set.delta_rps_sign),
			),
			__bindgen_padding_0: [0; 3],
		},
		delta_idx_minus1: set.delta_index_minus_1,
		use_delta_flag: u16::try_from(set.use_delta_mask)
			.map_err(|_| Error::data_loss("H.265 use-delta mask exceeds StdVideo storage"))?,
		abs_delta_rps_minus1: u16::try_from(set.abs_delta_rps_minus_1)
			.map_err(|_| Error::data_loss("H.265 delta RPS exceeds StdVideo storage"))?,
		used_by_curr_pic_flag: u16::try_from(set.used_by_current_mask)
			.map_err(|_| Error::data_loss("H.265 used-by-current mask exceeds StdVideo storage"))?,
		used_by_curr_pic_s0_flag: u16::try_from(set.used_by_current_negative_mask)
			.map_err(|_| Error::data_loss("H.265 negative RPS mask exceeds StdVideo storage"))?,
		used_by_curr_pic_s1_flag: u16::try_from(set.used_by_current_positive_mask)
			.map_err(|_| Error::data_loss("H.265 positive RPS mask exceeds StdVideo storage"))?,
		reserved1: 0,
		reserved2: 0,
		reserved3: 0,
		num_negative_pics: u8::try_from(set.negative_delta_poc_minus_1.len())
			.map_err(|_| Error::data_loss("H.265 negative RPS count exceeds StdVideo storage"))?,
		num_positive_pics: u8::try_from(set.positive_delta_poc_minus_1.len())
			.map_err(|_| Error::data_loss("H.265 positive RPS count exceeds StdVideo storage"))?,
		delta_poc_s0_minus1,
		delta_poc_s1_minus1,
	})
}

pub(super) fn std_h265_long_term_references(
	values: &[video::H265LongTermReferencePicture],
) -> Result<Option<ash::vk::native::StdVideoH265LongTermRefPicsSps>> {
	if values.is_empty() {
		return Ok(None);
	}
	if values.len() > 32 {
		return Err(Error::data_loss("H.265 long-term RPS count exceeds 32"));
	}
	let mut used_by_curr_pic_lt_sps_flag = 0_u32;
	let mut lt_ref_pic_poc_lsb_sps = [0_u32; 32];
	for (index, value) in values.iter().enumerate() {
		used_by_curr_pic_lt_sps_flag |= u32::from(value.used_by_current) << index;
		lt_ref_pic_poc_lsb_sps[index] = value.picture_order_count_lsb;
	}
	Ok(Some(ash::vk::native::StdVideoH265LongTermRefPicsSps {
		used_by_curr_pic_lt_sps_flag,
		lt_ref_pic_poc_lsb_sps,
	}))
}

pub(super) fn std_h265_scaling(
	scaling: &video::H265ScalingLists,
) -> ash::vk::native::StdVideoH265ScalingLists {
	ash::vk::native::StdVideoH265ScalingLists {
		ScalingList4x4: scaling.list_4x4,
		ScalingList8x8: scaling.list_8x8,
		ScalingList16x16: scaling.list_16x16,
		ScalingList32x32: scaling.list_32x32,
		ScalingListDCCoef16x16: scaling.dc_16x16,
		ScalingListDCCoef32x32: scaling.dc_32x32,
	}
}

pub(super) fn std_h265_vui(
	vui: &video::H265VuiParameters,
	hrd: *const ash::vk::native::StdVideoH265HrdParameters,
) -> Result<ash::vk::native::StdVideoH265SequenceParameterSetVui> {
	let aspect = vui.aspect_ratio.unwrap_or_default();
	let signal = vui.video_signal.unwrap_or_default();
	let colour = signal.colour_description.unwrap_or_default();
	let chroma = vui.chroma_location.unwrap_or_default();
	let display = vui.default_display_window.unwrap_or([0; 4]);
	let timing = vui.timing.unwrap_or_default();
	let restriction = vui.bitstream_restriction.unwrap_or_default();
	Ok(ash::vk::native::StdVideoH265SequenceParameterSetVui {
		flags: ash::vk::native::StdVideoH265SpsVuiFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoH265SpsVuiFlags::new_bitfield_1(
				u32::from(vui.aspect_ratio.is_some()),
				u32::from(vui.overscan_appropriate.is_some()),
				u32::from(vui.overscan_appropriate.unwrap_or(false)),
				u32::from(vui.video_signal.is_some()),
				u32::from(signal.full_range),
				u32::from(signal.colour_description.is_some()),
				u32::from(vui.chroma_location.is_some()),
				u32::from(vui.neutral_chroma_indication),
				u32::from(vui.field_sequence),
				u32::from(vui.frame_field_info_present),
				u32::from(vui.default_display_window.is_some()),
				u32::from(vui.timing.is_some()),
				u32::from(timing.num_ticks_poc_diff_one_minus_1.is_some()),
				u32::from(vui.hrd.is_some()),
				u32::from(vui.bitstream_restriction.is_some()),
				u32::from(restriction.tiles_fixed_structure),
				u32::from(restriction.motion_vectors_over_picture_boundaries),
				u32::from(restriction.restricted_reference_picture_lists),
			),
			__bindgen_padding_0: 0,
		},
		aspect_ratio_idc: u32::from(aspect.idc),
		sar_width: aspect.sar_width,
		sar_height: aspect.sar_height,
		video_format: signal.video_format,
		colour_primaries: colour.colour_primaries,
		transfer_characteristics: colour.transfer_characteristics,
		matrix_coeffs: colour.matrix_coefficients,
		chroma_sample_loc_type_top_field: h265_u8(chroma.top_field, "top chroma location")?,
		chroma_sample_loc_type_bottom_field: h265_u8(chroma.bottom_field, "bottom chroma location")?,
		reserved1: 0,
		reserved2: 0,
		def_disp_win_left_offset: u16::try_from(display[0])
			.map_err(|_| Error::data_loss("H.265 display-window left offset exceeds StdVideo storage"))?,
		def_disp_win_right_offset: u16::try_from(display[1]).map_err(|_| {
			Error::data_loss("H.265 display-window right offset exceeds StdVideo storage")
		})?,
		def_disp_win_top_offset: u16::try_from(display[2])
			.map_err(|_| Error::data_loss("H.265 display-window top offset exceeds StdVideo storage"))?,
		def_disp_win_bottom_offset: u16::try_from(display[3]).map_err(|_| {
			Error::data_loss("H.265 display-window bottom offset exceeds StdVideo storage")
		})?,
		vui_num_units_in_tick: timing.num_units_in_tick,
		vui_time_scale: timing.time_scale,
		vui_num_ticks_poc_diff_one_minus1: timing.num_ticks_poc_diff_one_minus_1.unwrap_or(0),
		min_spatial_segmentation_idc: u16::try_from(restriction.min_spatial_segmentation_idc).map_err(
			|_| Error::data_loss("H.265 minimum spatial segmentation exceeds StdVideo storage"),
		)?,
		reserved3: 0,
		max_bytes_per_pic_denom: h265_u8(
			restriction.max_bytes_per_picture_denom,
			"maximum bytes-per-picture denominator",
		)?,
		max_bits_per_min_cu_denom: h265_u8(
			restriction.max_bits_per_min_coding_unit_denom,
			"maximum bits-per-min-CU denominator",
		)?,
		log2_max_mv_length_horizontal: h265_u8(
			restriction.log2_max_motion_vector_length_horizontal,
			"horizontal motion-vector length",
		)?,
		log2_max_mv_length_vertical: h265_u8(
			restriction.log2_max_motion_vector_length_vertical,
			"vertical motion-vector length",
		)?,
		pHrdParameters: hrd,
	})
}

pub(super) fn std_h265_pps(
	pps: &video::H265PictureParameterSet,
	video_parameter_set_id: u32,
	scaling: *const ash::vk::native::StdVideoH265ScalingLists,
) -> Result<ash::vk::native::StdVideoH265PictureParameterSet> {
	let mut column_width_minus1 = [0_u16; 19];
	let mut row_height_minus1 = [0_u16; 21];
	for (output, input) in column_width_minus1
		.iter_mut()
		.zip(&pps.column_width_minus_1)
	{
		*output = u16::try_from(*input)
			.map_err(|_| Error::data_loss("H.265 tile-column width exceeds StdVideo storage"))?;
	}
	for (output, input) in row_height_minus1.iter_mut().zip(&pps.row_height_minus_1) {
		*output = u16::try_from(*input)
			.map_err(|_| Error::data_loss("H.265 tile-row height exceeds StdVideo storage"))?;
	}
	Ok(ash::vk::native::StdVideoH265PictureParameterSet {
		flags: ash::vk::native::StdVideoH265PpsFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoH265PpsFlags::new_bitfield_1(
				u32::from(pps.dependent_slice_segments_enabled),
				u32::from(pps.output_flag_present),
				u32::from(pps.sign_data_hiding_enabled),
				u32::from(pps.cabac_init_present),
				u32::from(pps.constrained_intra_pred),
				u32::from(pps.transform_skip_enabled),
				u32::from(pps.cu_qp_delta_enabled),
				u32::from(pps.slice_chroma_qp_offsets_present),
				u32::from(pps.weighted_pred),
				u32::from(pps.weighted_bipred),
				u32::from(pps.transquant_bypass_enabled),
				u32::from(pps.tiles_enabled),
				u32::from(pps.entropy_coding_sync_enabled),
				u32::from(pps.uniform_spacing),
				u32::from(pps.loop_filter_across_tiles_enabled),
				u32::from(pps.loop_filter_across_slices_enabled),
				u32::from(pps.deblocking_filter_control_present),
				u32::from(pps.deblocking_filter_override_enabled),
				u32::from(pps.deblocking_filter_disabled),
				u32::from(pps.scaling_lists.is_some()),
				u32::from(pps.lists_modification_present),
				u32::from(pps.slice_segment_header_extension_present),
				u32::from(pps.extension_present),
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
			),
		},
		pps_pic_parameter_set_id: h265_u8(pps.id, "PPS id")?,
		pps_seq_parameter_set_id: h265_u8(pps.sequence_parameter_set_id, "SPS id")?,
		sps_video_parameter_set_id: h265_u8(video_parameter_set_id, "VPS id")?,
		num_extra_slice_header_bits: h265_u8(pps.num_extra_slice_header_bits, "extra slice bits")?,
		num_ref_idx_l0_default_active_minus1: h265_u8(
			pps.num_ref_idx_l0_default_active_minus_1,
			"default L0 reference count",
		)?,
		num_ref_idx_l1_default_active_minus1: h265_u8(
			pps.num_ref_idx_l1_default_active_minus_1,
			"default L1 reference count",
		)?,
		init_qp_minus26: i8::try_from(pps.init_qp_minus_26)
			.map_err(|_| Error::data_loss("H.265 initial QP exceeds StdVideo storage"))?,
		diff_cu_qp_delta_depth: h265_u8(pps.diff_cu_qp_delta_depth, "CU QP delta depth")?,
		pps_cb_qp_offset: i8::try_from(pps.cb_qp_offset)
			.map_err(|_| Error::data_loss("H.265 Cb QP offset exceeds StdVideo storage"))?,
		pps_cr_qp_offset: i8::try_from(pps.cr_qp_offset)
			.map_err(|_| Error::data_loss("H.265 Cr QP offset exceeds StdVideo storage"))?,
		pps_beta_offset_div2: i8::try_from(pps.beta_offset_div_2)
			.map_err(|_| Error::data_loss("H.265 beta offset exceeds StdVideo storage"))?,
		pps_tc_offset_div2: i8::try_from(pps.tc_offset_div_2)
			.map_err(|_| Error::data_loss("H.265 tc offset exceeds StdVideo storage"))?,
		log2_parallel_merge_level_minus2: h265_u8(
			pps.log2_parallel_merge_level_minus_2,
			"parallel merge level",
		)?,
		log2_max_transform_skip_block_size_minus2: 0,
		diff_cu_chroma_qp_offset_depth: 0,
		chroma_qp_offset_list_len_minus1: 0,
		cb_qp_offset_list: [0; 6],
		cr_qp_offset_list: [0; 6],
		log2_sao_offset_scale_luma: 0,
		log2_sao_offset_scale_chroma: 0,
		pps_act_y_qp_offset_plus5: 0,
		pps_act_cb_qp_offset_plus5: 0,
		pps_act_cr_qp_offset_plus3: 0,
		pps_num_palette_predictor_initializers: 0,
		luma_bit_depth_entry_minus8: 0,
		chroma_bit_depth_entry_minus8: 0,
		num_tile_columns_minus1: h265_u8(pps.num_tile_columns_minus_1, "tile-column count")?,
		num_tile_rows_minus1: h265_u8(pps.num_tile_rows_minus_1, "tile-row count")?,
		reserved1: 0,
		reserved2: 0,
		column_width_minus1,
		row_height_minus1,
		reserved3: 0,
		pScalingLists: scaling,
		pPredictorPaletteEntries: std::ptr::null(),
	})
}

pub(super) fn h265_u8(value: u32, field: &str) -> Result<u8> {
	u8::try_from(value)
		.map_err(|_| Error::data_loss(format!("H.265 {field} exceeds StdVideo storage")))
}

pub(super) fn std_h264_sps(
	sps: &video::H264SequenceParameterSet,
	std_scaling: *const ash::vk::native::StdVideoH264ScalingLists,
	std_vui: *const ash::vk::native::StdVideoH264SequenceParameterSetVui,
) -> Result<ash::vk::native::StdVideoH264SequenceParameterSet> {
	let flags = ash::vk::native::StdVideoH264SpsFlags {
		_bitfield_align_1: [],
		_bitfield_1: ash::vk::native::StdVideoH264SpsFlags::new_bitfield_1(
			u32::from(sps.constraint_flags & 0x80 != 0),
			u32::from(sps.constraint_flags & 0x40 != 0),
			u32::from(sps.constraint_flags & 0x20 != 0),
			u32::from(sps.constraint_flags & 0x10 != 0),
			u32::from(sps.constraint_flags & 0x08 != 0),
			u32::from(sps.constraint_flags & 0x04 != 0),
			u32::from(sps.direct_8x8_inference),
			u32::from(sps.mb_adaptive_frame_field),
			u32::from(sps.frame_mbs_only),
			u32::from(sps.delta_pic_order_always_zero),
			u32::from(sps.separate_colour_plane),
			u32::from(sps.gaps_in_frame_num_value_allowed),
			u32::from(sps.qpprime_y_zero_transform_bypass),
			u32::from(
				sps.frame_crop_left_offset != 0
					|| sps.frame_crop_right_offset != 0
					|| sps.frame_crop_top_offset != 0
					|| sps.frame_crop_bottom_offset != 0,
			),
			u32::from(sps.scaling_lists.is_some()),
			u32::from(sps.vui.is_some()),
		),
		__bindgen_padding_0: 0,
	};
	Ok(ash::vk::native::StdVideoH264SequenceParameterSet {
		flags,
		profile_idc: sps.profile_idc,
		level_idc: std_h264_level(sps.level_idc)?,
		chroma_format_idc: sps.chroma_format_idc,
		seq_parameter_set_id: u8::try_from(sps.id)
			.map_err(|_| Error::data_loss("H.264 SPS id exceeds StdVideo storage"))?,
		bit_depth_luma_minus8: u8::try_from(sps.bit_depth_luma_minus_8)
			.map_err(|_| Error::data_loss("H.264 luma bit depth exceeds StdVideo storage"))?,
		bit_depth_chroma_minus8: u8::try_from(sps.bit_depth_chroma_minus_8)
			.map_err(|_| Error::data_loss("H.264 chroma bit depth exceeds StdVideo storage"))?,
		log2_max_frame_num_minus4: u8::try_from(sps.log2_max_frame_num_minus_4)
			.map_err(|_| Error::data_loss("H.264 frame-number width exceeds StdVideo storage"))?,
		pic_order_cnt_type: sps.pic_order_count_type,
		offset_for_non_ref_pic: sps.offset_for_non_ref_pic,
		offset_for_top_to_bottom_field: sps.offset_for_top_to_bottom_field,
		log2_max_pic_order_cnt_lsb_minus4: u8::try_from(sps.log2_max_pic_order_count_lsb_minus_4)
			.map_err(|_| Error::data_loss("H.264 POC width exceeds StdVideo storage"))?,
		num_ref_frames_in_pic_order_cnt_cycle: u8::try_from(sps.offset_for_ref_frame.len())
			.map_err(|_| Error::data_loss("H.264 POC cycle exceeds StdVideo storage"))?,
		max_num_ref_frames: u8::try_from(sps.max_num_ref_frames)
			.map_err(|_| Error::data_loss("H.264 reference count exceeds StdVideo storage"))?,
		reserved1: 0,
		pic_width_in_mbs_minus1: sps
			.width_in_macroblocks
			.checked_sub(1)
			.ok_or_else(|| Error::data_loss("H.264 SPS has zero macroblock width"))?,
		pic_height_in_map_units_minus1: sps
			.height_in_map_units
			.checked_sub(1)
			.ok_or_else(|| Error::data_loss("H.264 SPS has zero map-unit height"))?,
		frame_crop_left_offset: sps.frame_crop_left_offset,
		frame_crop_right_offset: sps.frame_crop_right_offset,
		frame_crop_top_offset: sps.frame_crop_top_offset,
		frame_crop_bottom_offset: sps.frame_crop_bottom_offset,
		reserved2: 0,
		pOffsetForRefFrame: if sps.offset_for_ref_frame.is_empty() {
			std::ptr::null()
		} else {
			sps.offset_for_ref_frame.as_ptr()
		},
		pScalingLists: std_scaling,
		pSequenceParameterSetVui: std_vui,
	})
}

pub(super) fn std_h264_hrd(
	vui: &video::H264VuiParameters,
) -> Result<Option<ash::vk::native::StdVideoH264HrdParameters>> {
	let hrd = match (&vui.nal_hrd, &vui.vcl_hrd) {
		(Some(nal), Some(vcl)) if nal != vcl => {
			return Err(Error::missing_capability(
				"distinct H.264 NAL and VCL HRD parameters cannot share one StdVideo table",
			));
		}
		(Some(hrd), _) | (_, Some(hrd)) => hrd,
		(None, None) => return Ok(None),
	};
	let cpb_count_minus_1 = hrd
		.entries
		.len()
		.checked_sub(1)
		.ok_or_else(|| Error::data_loss("H.264 HRD has no CPB entries"))?;
	let mut bit_rate_value_minus1 = [0_u32; 32];
	let mut cpb_size_value_minus1 = [0_u32; 32];
	let mut cbr_flag = [0_u8; 32];
	for (index, entry) in hrd.entries.iter().enumerate() {
		let Some(bit_rate) = bit_rate_value_minus1.get_mut(index) else {
			return Err(Error::data_loss("H.264 HRD CPB count exceeds 32"));
		};
		*bit_rate = entry.bit_rate_value_minus_1;
		cpb_size_value_minus1[index] = entry.cpb_size_value_minus_1;
		cbr_flag[index] = u8::from(entry.constant_bit_rate);
	}
	Ok(Some(ash::vk::native::StdVideoH264HrdParameters {
		cpb_cnt_minus1: u8::try_from(cpb_count_minus_1)
			.map_err(|_| Error::data_loss("H.264 HRD CPB count exceeds StdVideo storage"))?,
		bit_rate_scale: hrd.bit_rate_scale,
		cpb_size_scale: hrd.cpb_size_scale,
		reserved1: 0,
		bit_rate_value_minus1,
		cpb_size_value_minus1,
		cbr_flag,
		initial_cpb_removal_delay_length_minus1: u32::from(
			hrd.initial_cpb_removal_delay_length_minus_1,
		),
		cpb_removal_delay_length_minus1: u32::from(hrd.cpb_removal_delay_length_minus_1),
		dpb_output_delay_length_minus1: u32::from(hrd.dpb_output_delay_length_minus_1),
		time_offset_length: u32::from(hrd.time_offset_length),
	}))
}

pub(super) fn std_h264_vui(
	vui: &video::H264VuiParameters,
	std_hrd: *const ash::vk::native::StdVideoH264HrdParameters,
) -> Result<ash::vk::native::StdVideoH264SequenceParameterSetVui> {
	let aspect = vui.aspect_ratio.unwrap_or(video::H264AspectRatio {
		idc: 0,
		sar_width: 0,
		sar_height: 0,
	});
	let signal = vui.video_signal.unwrap_or(video::H264VideoSignal {
		video_format: 0,
		full_range: false,
		colour_description: None,
	});
	let colour = signal
		.colour_description
		.unwrap_or(video::H264ColourDescription {
			colour_primaries: 0,
			transfer_characteristics: 0,
			matrix_coefficients: 0,
		});
	let timing = vui.timing.unwrap_or(video::H264TimingInfo {
		num_units_in_tick: 0,
		time_scale: 0,
		fixed_frame_rate: false,
	});
	let chroma = vui.chroma_location.unwrap_or(video::H264ChromaLocation {
		top_field: 0,
		bottom_field: 0,
	});
	let restriction = vui
		.bitstream_restriction
		.unwrap_or(video::H264BitstreamRestriction {
			motion_vectors_over_picture_boundaries: false,
			max_bytes_per_picture_denom: 0,
			max_bits_per_macroblock_denom: 0,
			log2_max_motion_vector_length_horizontal: 0,
			log2_max_motion_vector_length_vertical: 0,
			max_num_reorder_frames: 0,
			max_dec_frame_buffering: 0,
		});
	let flags = ash::vk::native::StdVideoH264SpsVuiFlags {
		_bitfield_align_1: [],
		_bitfield_1: ash::vk::native::StdVideoH264SpsVuiFlags::new_bitfield_1(
			u32::from(vui.aspect_ratio.is_some()),
			u32::from(vui.overscan_appropriate.is_some()),
			u32::from(vui.overscan_appropriate.unwrap_or(false)),
			u32::from(vui.video_signal.is_some()),
			u32::from(signal.full_range),
			u32::from(signal.colour_description.is_some()),
			u32::from(vui.chroma_location.is_some()),
			u32::from(vui.timing.is_some()),
			u32::from(timing.fixed_frame_rate),
			u32::from(vui.bitstream_restriction.is_some()),
			u32::from(vui.nal_hrd.is_some()),
			u32::from(vui.vcl_hrd.is_some()),
		),
		__bindgen_padding_0: 0,
	};
	Ok(ash::vk::native::StdVideoH264SequenceParameterSetVui {
		flags,
		aspect_ratio_idc: u32::from(aspect.idc),
		sar_width: aspect.sar_width,
		sar_height: aspect.sar_height,
		video_format: signal.video_format,
		colour_primaries: colour.colour_primaries,
		transfer_characteristics: colour.transfer_characteristics,
		matrix_coefficients: colour.matrix_coefficients,
		num_units_in_tick: timing.num_units_in_tick,
		time_scale: timing.time_scale,
		max_num_reorder_frames: u8::try_from(restriction.max_num_reorder_frames)
			.map_err(|_| Error::data_loss("H.264 reorder count exceeds StdVideo storage"))?,
		max_dec_frame_buffering: u8::try_from(restriction.max_dec_frame_buffering)
			.map_err(|_| Error::data_loss("H.264 DPB limit exceeds StdVideo storage"))?,
		chroma_sample_loc_type_top_field: u8::try_from(chroma.top_field)
			.map_err(|_| Error::data_loss("H.264 top chroma location exceeds StdVideo storage"))?,
		chroma_sample_loc_type_bottom_field: u8::try_from(chroma.bottom_field)
			.map_err(|_| Error::data_loss("H.264 bottom chroma location exceeds StdVideo storage"))?,
		reserved1: 0,
		pHrdParameters: std_hrd,
	})
}

pub(super) fn std_h264_pps(
	pps: &video::H264PictureParameterSet,
	std_scaling: *const ash::vk::native::StdVideoH264ScalingLists,
) -> Result<ash::vk::native::StdVideoH264PictureParameterSet> {
	let flags = ash::vk::native::StdVideoH264PpsFlags {
		_bitfield_align_1: [],
		_bitfield_1: ash::vk::native::StdVideoH264PpsFlags::new_bitfield_1(
			u32::from(pps.transform_8x8_mode),
			u32::from(pps.redundant_pic_count_present),
			u32::from(pps.constrained_intra_pred),
			u32::from(pps.deblocking_filter_control_present),
			u32::from(pps.weighted_pred),
			u32::from(pps.bottom_field_pic_order_in_frame_present),
			u32::from(pps.entropy_coding_mode),
			u32::from(pps.scaling_lists.is_some()),
		),
		__bindgen_padding_0: [0; 3],
	};
	Ok(ash::vk::native::StdVideoH264PictureParameterSet {
		flags,
		seq_parameter_set_id: u8::try_from(pps.sequence_parameter_set_id)
			.map_err(|_| Error::data_loss("H.264 PPS SPS id exceeds StdVideo storage"))?,
		pic_parameter_set_id: u8::try_from(pps.id)
			.map_err(|_| Error::data_loss("H.264 PPS id exceeds StdVideo storage"))?,
		num_ref_idx_l0_default_active_minus1: u8::try_from(pps.num_ref_idx_l0_default_active_minus_1)
			.map_err(|_| {
			Error::data_loss("H.264 L0 reference count exceeds StdVideo storage")
		})?,
		num_ref_idx_l1_default_active_minus1: u8::try_from(pps.num_ref_idx_l1_default_active_minus_1)
			.map_err(|_| {
			Error::data_loss("H.264 L1 reference count exceeds StdVideo storage")
		})?,
		weighted_bipred_idc: pps.weighted_bipred_idc,
		pic_init_qp_minus26: i8::try_from(pps.pic_init_qp_minus_26)
			.map_err(|_| Error::data_loss("H.264 initial QP exceeds StdVideo storage"))?,
		pic_init_qs_minus26: i8::try_from(pps.pic_init_qs_minus_26)
			.map_err(|_| Error::data_loss("H.264 initial QS exceeds StdVideo storage"))?,
		chroma_qp_index_offset: i8::try_from(pps.chroma_qp_index_offset)
			.map_err(|_| Error::data_loss("H.264 chroma QP offset exceeds StdVideo storage"))?,
		second_chroma_qp_index_offset: i8::try_from(pps.second_chroma_qp_index_offset)
			.map_err(|_| Error::data_loss("H.264 second chroma QP offset exceeds StdVideo storage"))?,
		pScalingLists: std_scaling,
	})
}

pub(super) fn std_h264_scaling(
	scaling: &video::H264ScalingLists,
) -> ash::vk::native::StdVideoH264ScalingLists {
	ash::vk::native::StdVideoH264ScalingLists {
		scaling_list_present_mask: scaling.present_mask,
		use_default_scaling_matrix_mask: scaling.use_default_mask,
		ScalingList4x4: scaling.list_4x4,
		ScalingList8x8: scaling.list_8x8,
	}
}

pub(super) fn std_h264_level(level_idc: u32) -> Result<ash::vk::native::StdVideoH264LevelIdc> {
	match level_idc {
		10 => Ok(0),
		11 => Ok(1),
		12 => Ok(2),
		13 => Ok(3),
		20 => Ok(4),
		21 => Ok(5),
		22 => Ok(6),
		30 => Ok(7),
		31 => Ok(8),
		32 => Ok(9),
		40 => Ok(10),
		41 => Ok(11),
		42 => Ok(12),
		50 => Ok(13),
		51 => Ok(14),
		52 => Ok(15),
		60 => Ok(16),
		61 => Ok(17),
		62 => Ok(18),
		_ => Err(Error::missing_capability(format!(
			"H.264 level_idc {level_idc} has no Khronos StdVideo mapping"
		))),
	}
}

pub(super) fn with_decode_profile<T>(
	profile: video::VideoDecodeProfile,
	query: impl FnOnce(&ash::vk::VideoProfileInfoKHR<'_>) -> Result<T>,
) -> Result<T> {
	match profile {
		video::VideoDecodeProfile::H264 {
			profile,
			picture_layout,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec = ash::vk::VideoDecodeH264ProfileInfoKHR::default()
				.std_profile_idc(h264_profile(profile))
				.picture_layout(h264_picture_layout(picture_layout));
			let profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::DECODE_H264,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec);
			query(&profile)
		}
		video::VideoDecodeProfile::H265 {
			profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec =
				ash::vk::VideoDecodeH265ProfileInfoKHR::default().std_profile_idc(h265_profile(profile));
			let profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::DECODE_H265,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec);
			query(&profile)
		}
		video::VideoDecodeProfile::Av1 {
			profile,
			film_grain_support,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec = ash::vk::VideoDecodeAV1ProfileInfoKHR::default()
				.std_profile(av1_profile(profile))
				.film_grain_support(film_grain_support);
			let profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::DECODE_AV1,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec);
			query(&profile)
		}
		video::VideoDecodeProfile::Vp9 {
			profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec =
				ash_vp9::vk::VideoDecodeVP9ProfileInfoKHR::default().std_profile(vp9_profile(profile));
			let mut profile = common_profile(
				vp9_operation(),
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			);
			profile.p_next = std::ptr::from_mut(&mut codec).cast();
			query(&profile)
		}
	}
}

pub(super) fn with_encode_profile<T>(
	profile: video::VideoEncodeProfile,
	query: impl FnOnce(&ash::vk::VideoProfileInfoKHR<'_>) -> Result<T>,
) -> Result<T> {
	match profile {
		video::VideoEncodeProfile::H264 {
			profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec =
				ash::vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(h264_profile(profile));
			let profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec);
			query(&profile)
		}
		video::VideoEncodeProfile::H265 {
			profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec =
				ash::vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(h265_profile(profile));
			let profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::ENCODE_H265,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec);
			query(&profile)
		}
	}
}

pub(super) fn query_formats(
	loader: &ash::khr::video_queue::Instance,
	physical: ash::vk::PhysicalDevice,
	profile: &ash::vk::VideoProfileInfoKHR<'_>,
	usage: ash::vk::ImageUsageFlags,
) -> Result<Vec<ash::vk::VideoFormatPropertiesKHR<'static>>> {
	let profiles = [*profile];
	let mut profile_list = ash::vk::VideoProfileListInfoKHR::default().profiles(&profiles);
	let format_info = ash::vk::PhysicalDeviceVideoFormatInfoKHR::default()
		.image_usage(usage)
		.push_next(&mut profile_list);
	let mut count = 0_u32;
	// SAFETY: every pointer in `format_info` remains live, and a null output
	// pointer requests only the required element count.
	let result = unsafe {
		(loader.fp().get_physical_device_video_format_properties_khr)(
			physical,
			&format_info,
			&mut count,
			std::ptr::null_mut(),
		)
	};
	if result != ash::vk::Result::SUCCESS {
		return Err(Error::backend_failure(
			"Vulkan",
			"video-format count query",
			result,
		));
	}
	if count > MAX_VIDEO_FORMATS {
		return Err(Error::resource_exhausted(format!(
			"Vulkan reported {count} video formats, above OA's {MAX_VIDEO_FORMATS} safety bound"
		)));
	}
	let mut formats = vec![ash::vk::VideoFormatPropertiesKHR::default(); count as usize];
	if count == 0 {
		return Ok(formats);
	}
	// SAFETY: `formats` contains `count` initialized output structures with valid
	// sType values; Vulkan writes at most the count passed by mutable pointer.
	let result = unsafe {
		(loader.fp().get_physical_device_video_format_properties_khr)(
			physical,
			&format_info,
			&mut count,
			formats.as_mut_ptr(),
		)
	};
	if result != ash::vk::Result::SUCCESS {
		return Err(Error::backend_failure(
			"Vulkan",
			"video-format enumeration",
			result,
		));
	}
	formats.truncate(count as usize);
	Ok(formats)
}

pub(super) fn convert_formats(
	formats: &[ash::vk::VideoFormatPropertiesKHR<'_>],
) -> (Vec<video::VideoImageFormat>, usize) {
	let mut converted = Vec::new();
	let mut unrecognized = 0;
	for format in formats {
		let Some(pixel_format) = pixel_format(format.format) else {
			unrecognized += 1;
			continue;
		};
		converted.push(video::VideoImageFormat {
			pixel_format,
			optimal_tiling: format.image_tiling == ash::vk::ImageTiling::OPTIMAL,
			sampled: format
				.image_usage_flags
				.contains(ash::vk::ImageUsageFlags::SAMPLED),
			storage: format
				.image_usage_flags
				.contains(ash::vk::ImageUsageFlags::STORAGE),
			transfer_source: format
				.image_usage_flags
				.contains(ash::vk::ImageUsageFlags::TRANSFER_SRC),
			transfer_destination: format
				.image_usage_flags
				.contains(ash::vk::ImageUsageFlags::TRANSFER_DST),
		});
	}
	converted.sort_by_key(|format| format.pixel_format);
	converted.dedup();
	(converted, unrecognized)
}

pub(super) fn pixel_format(format: ash::vk::Format) -> Option<video::VideoPixelFormat> {
	match format {
		ash::vk::Format::G8_B8_R8_3PLANE_420_UNORM => Some(video::VideoPixelFormat::Yuv420Planar8),
		ash::vk::Format::G8_B8R8_2PLANE_420_UNORM => Some(video::VideoPixelFormat::Nv12),
		ash::vk::Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16 => {
			Some(video::VideoPixelFormat::Yuv420Planar10)
		}
		ash::vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 => {
			Some(video::VideoPixelFormat::P010)
		}
		ash::vk::Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16 => {
			Some(video::VideoPixelFormat::P012)
		}
		_ => None,
	}
}

pub(super) fn ensure_extension_advertised(
	physical: &DevicePhysical,
	profile: video::VideoDecodeProfile,
) -> Result<()> {
	let advertised = match profile {
		video::VideoDecodeProfile::H264 { .. } => physical.video.h264_decode,
		video::VideoDecodeProfile::H265 { .. } => physical.video.h265_decode,
		video::VideoDecodeProfile::Av1 { .. } => physical.video.av1_decode,
		video::VideoDecodeProfile::Vp9 { .. } => physical.video.vp9_decode,
	};
	if physical.video.decode_queue_family.is_none() || !advertised {
		return Err(Error::missing_capability(format!(
			"the selected device does not advertise the requested {profile:?} decoder extension"
		)));
	}
	Ok(())
}

pub(super) fn ensure_encode_extension_advertised(
	physical: &DevicePhysical,
	profile: video::VideoEncodeProfile,
) -> Result<()> {
	let advertised = match profile {
		video::VideoEncodeProfile::H264 { .. } => physical.video.h264_encode,
		video::VideoEncodeProfile::H265 { .. } => physical.video.h265_encode,
	};
	if physical.video.encode_queue_family.is_none() || !advertised {
		return Err(Error::missing_capability(format!(
			"the selected device does not advertise the requested {profile:?} encoder extension"
		)));
	}
	Ok(())
}

pub(super) fn common_profile<'a>(
	operation: ash::vk::VideoCodecOperationFlagsKHR,
	chroma: video::VideoChromaSubsampling,
	luma_depth: video::VideoComponentBitDepth,
	chroma_depth: video::VideoComponentBitDepth,
) -> ash::vk::VideoProfileInfoKHR<'a> {
	ash::vk::VideoProfileInfoKHR::default()
		.video_codec_operation(operation)
		.chroma_subsampling(chroma_subsampling(chroma))
		.luma_bit_depth(bit_depth(luma_depth))
		.chroma_bit_depth(bit_depth(chroma_depth))
}

pub(super) fn query(
	loader: &ash::khr::video_queue::Instance,
	physical: ash::vk::PhysicalDevice,
	profile: &ash::vk::VideoProfileInfoKHR<'_>,
	capabilities: &mut ash::vk::VideoCapabilitiesKHR<'_>,
) -> Result<()> {
	// SAFETY: the loader was created from the live instance that owns `physical`;
	// both root structures and their required decode/codec-specific pNext chains
	// remain live and writable for the duration of the call.
	let result = unsafe {
		(loader.fp().get_physical_device_video_capabilities_khr)(physical, profile, capabilities)
	};
	match result {
		ash::vk::Result::SUCCESS => Ok(()),
		ash::vk::Result::ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR
		| ash::vk::Result::ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR
		| ash::vk::Result::ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR => Err(Error::missing_capability(
			format!("Vulkan rejected the requested video profile: {result}"),
		)),
		_ => Err(Error::backend_failure(
			"Vulkan",
			"video-profile capability query",
			result,
		)),
	}
}

pub(super) fn convert_capabilities(
	profile: video::VideoDecodeProfile,
	capabilities: CommonCapabilities,
	decode: &ash::vk::VideoDecodeCapabilitiesKHR<'_>,
	level: video::VideoDecodeLevel,
	field_offset_granularity: Option<(i32, i32)>,
) -> video::VideoDecodeCapabilities {
	video::VideoDecodeCapabilities {
		profile,
		min_coded_extent: capabilities.min_coded_extent,
		max_coded_extent: capabilities.max_coded_extent,
		picture_access_granularity: capabilities.picture_access_granularity,
		min_bitstream_offset_alignment: capabilities.min_bitstream_buffer_offset_alignment,
		min_bitstream_size_alignment: capabilities.min_bitstream_buffer_size_alignment,
		max_dpb_slots: capabilities.max_dpb_slots,
		max_active_reference_pictures: capabilities.max_active_reference_pictures,
		level,
		field_offset_granularity,
		dpb_and_output_coincide: decode
			.flags
			.contains(ash::vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE),
		dpb_and_output_distinct: decode
			.flags
			.contains(ash::vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_DISTINCT),
		protected_content: capabilities
			.flags
			.contains(ash::vk::VideoCapabilityFlagsKHR::PROTECTED_CONTENT),
		separate_reference_images: capabilities
			.flags
			.contains(ash::vk::VideoCapabilityFlagsKHR::SEPARATE_REFERENCE_IMAGES),
	}
}

pub(super) fn convert_encode_capabilities(
	profile: video::VideoEncodeProfile,
	capabilities: CommonCapabilities,
	encode: &ash::vk::VideoEncodeCapabilitiesKHR<'_>,
	codec: video::VideoEncodeCodecCapabilities,
) -> video::VideoEncodeCapabilities {
	video::VideoEncodeCapabilities {
		profile,
		min_coded_extent: capabilities.min_coded_extent,
		max_coded_extent: capabilities.max_coded_extent,
		picture_access_granularity: capabilities.picture_access_granularity,
		input_picture_granularity: extent(encode.encode_input_picture_granularity),
		min_bitstream_offset_alignment: capabilities.min_bitstream_buffer_offset_alignment,
		min_bitstream_size_alignment: capabilities.min_bitstream_buffer_size_alignment,
		max_dpb_slots: capabilities.max_dpb_slots,
		max_active_reference_pictures: capabilities.max_active_reference_pictures,
		max_rate_control_layers: encode.max_rate_control_layers,
		max_bitrate: encode.max_bitrate,
		max_quality_levels: encode.max_quality_levels,
		constant_qp: encode
			.rate_control_modes
			.contains(ash::vk::VideoEncodeRateControlModeFlagsKHR::DISABLED),
		cbr: encode
			.rate_control_modes
			.contains(ash::vk::VideoEncodeRateControlModeFlagsKHR::CBR),
		vbr: encode
			.rate_control_modes
			.contains(ash::vk::VideoEncodeRateControlModeFlagsKHR::VBR),
		feedback_offset: encode
			.supported_encode_feedback_flags
			.contains(ash::vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET),
		feedback_bytes_written: encode
			.supported_encode_feedback_flags
			.contains(ash::vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN),
		feedback_overrides: encode
			.supported_encode_feedback_flags
			.contains(ash::vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_HAS_OVERRIDES),
		protected_content: capabilities
			.flags
			.contains(ash::vk::VideoCapabilityFlagsKHR::PROTECTED_CONTENT),
		separate_reference_images: capabilities
			.flags
			.contains(ash::vk::VideoCapabilityFlagsKHR::SEPARATE_REFERENCE_IMAGES),
		codec,
	}
}

#[derive(Clone, Copy)]
pub(super) struct CommonCapabilities {
	pub(super) flags: ash::vk::VideoCapabilityFlagsKHR,
	pub(super) std_header_version: ash::vk::ExtensionProperties,
	pub(super) min_bitstream_buffer_offset_alignment: u64,
	pub(super) min_bitstream_buffer_size_alignment: u64,
	pub(super) picture_access_granularity: video::VideoExtent,
	pub(super) min_coded_extent: video::VideoExtent,
	pub(super) max_coded_extent: video::VideoExtent,
	pub(super) max_dpb_slots: u32,
	pub(super) max_active_reference_pictures: u32,
}

impl From<&ash::vk::VideoCapabilitiesKHR<'_>> for CommonCapabilities {
	fn from(value: &ash::vk::VideoCapabilitiesKHR<'_>) -> Self {
		Self {
			flags: value.flags,
			std_header_version: value.std_header_version,
			min_bitstream_buffer_offset_alignment: value.min_bitstream_buffer_offset_alignment,
			min_bitstream_buffer_size_alignment: value.min_bitstream_buffer_size_alignment,
			picture_access_granularity: extent(value.picture_access_granularity),
			min_coded_extent: extent(value.min_coded_extent),
			max_coded_extent: extent(value.max_coded_extent),
			max_dpb_slots: value.max_dpb_slots,
			max_active_reference_pictures: value.max_active_reference_pictures,
		}
	}
}

pub(super) const fn extent(value: ash::vk::Extent2D) -> video::VideoExtent {
	video::VideoExtent {
		width: value.width,
		height: value.height,
	}
}

pub(super) const fn chroma_subsampling(
	value: video::VideoChromaSubsampling,
) -> ash::vk::VideoChromaSubsamplingFlagsKHR {
	match value {
		video::VideoChromaSubsampling::Monochrome => {
			ash::vk::VideoChromaSubsamplingFlagsKHR::MONOCHROME
		}
		video::VideoChromaSubsampling::Yuv420 => ash::vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
		video::VideoChromaSubsampling::Yuv422 => ash::vk::VideoChromaSubsamplingFlagsKHR::TYPE_422,
		video::VideoChromaSubsampling::Yuv444 => ash::vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
	}
}

pub(super) const fn bit_depth(
	value: video::VideoComponentBitDepth,
) -> ash::vk::VideoComponentBitDepthFlagsKHR {
	match value {
		video::VideoComponentBitDepth::Eight => ash::vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
		video::VideoComponentBitDepth::Ten => ash::vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
		video::VideoComponentBitDepth::Twelve => ash::vk::VideoComponentBitDepthFlagsKHR::TYPE_12,
	}
}

pub(super) const fn h264_profile(
	value: video::H264Profile,
) -> ash::vk::native::StdVideoH264ProfileIdc {
	match value {
		video::H264Profile::Baseline => {
			ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_BASELINE
		}
		video::H264Profile::Main => {
			ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN
		}
		video::H264Profile::High => {
			ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH
		}
		video::H264Profile::High444Predictive => {
			ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE
		}
	}
}

pub(super) const fn h264_picture_layout(
	value: video::H264PictureLayout,
) -> ash::vk::VideoDecodeH264PictureLayoutFlagsKHR {
	match value {
		video::H264PictureLayout::Progressive => {
			ash::vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE
		}
		video::H264PictureLayout::InterlacedInterleavedLines => {
			ash::vk::VideoDecodeH264PictureLayoutFlagsKHR::INTERLACED_INTERLEAVED_LINES
		}
		video::H264PictureLayout::InterlacedSeparatePlanes => {
			ash::vk::VideoDecodeH264PictureLayoutFlagsKHR::INTERLACED_SEPARATE_PLANES
		}
	}
}

pub(super) const fn h265_profile(
	value: video::H265Profile,
) -> ash::vk::native::StdVideoH265ProfileIdc {
	match value {
		video::H265Profile::Main => {
			ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN
		}
		video::H265Profile::Main10 => {
			ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10
		}
		video::H265Profile::MainStillPicture => {
			ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_STILL_PICTURE
		}
		video::H265Profile::FormatRangeExtensions => {
			ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS
		}
		video::H265Profile::ScreenContentCodingExtensions => {
			ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_SCC_EXTENSIONS
		}
	}
}

pub(super) const fn av1_profile(value: video::Av1Profile) -> ash::vk::native::StdVideoAV1Profile {
	match value {
		video::Av1Profile::Main => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN,
		video::Av1Profile::High => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_HIGH,
		video::Av1Profile::Professional => {
			ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_PROFESSIONAL
		}
	}
}

pub(super) const fn vp9_operation() -> ash::vk::VideoCodecOperationFlagsKHR {
	ash::vk::VideoCodecOperationFlagsKHR::from_raw(0b1000)
}

pub(super) const fn vp9_profile(
	value: video::Vp9Profile,
) -> ash_vp9::vk::native::StdVideoVP9Profile {
	match value {
		video::Vp9Profile::Profile0 => ash_vp9::vk::native::StdVideoVP9Profile_STD_VIDEO_VP9_PROFILE_0,
		video::Vp9Profile::Profile1 => ash_vp9::vk::native::StdVideoVP9Profile_STD_VIDEO_VP9_PROFILE_1,
		video::Vp9Profile::Profile2 => ash_vp9::vk::native::StdVideoVP9Profile_STD_VIDEO_VP9_PROFILE_2,
		video::Vp9Profile::Profile3 => ash_vp9::vk::native::StdVideoVP9Profile_STD_VIDEO_VP9_PROFILE_3,
	}
}

#[allow(dead_code, reason = "wired by the next AV1 frame decode checkpoint")]
pub(super) fn std_av1_color(
	color: &video::Av1ColorConfig,
) -> ash::vk::native::StdVideoAV1ColorConfig {
	let description = color
		.color_description
		.unwrap_or(video::Av1ColorDescription {
			color_primaries: 2,
			transfer_characteristics: 2,
			matrix_coefficients: 2,
		});
	let (subsampling_x, subsampling_y) = match color.chroma_subsampling {
		video::VideoChromaSubsampling::Monochrome | video::VideoChromaSubsampling::Yuv420 => (1, 1),
		video::VideoChromaSubsampling::Yuv422 => (1, 0),
		video::VideoChromaSubsampling::Yuv444 => (0, 0),
	};
	let chroma_sample_position = match color.chroma_sample_position {
		video::Av1ChromaSamplePosition::Unknown => 0,
		video::Av1ChromaSamplePosition::Vertical => 1,
		video::Av1ChromaSamplePosition::Colocated => 2,
		video::Av1ChromaSamplePosition::Reserved => 3,
	};
	ash::vk::native::StdVideoAV1ColorConfig {
		flags: ash::vk::native::StdVideoAV1ColorConfigFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoAV1ColorConfigFlags::new_bitfield_1(
				u32::from(color.monochrome),
				u32::from(color.full_range),
				u32::from(color.separate_uv_delta_q),
				u32::from(color.color_description.is_some()),
				0,
			),
		},
		BitDepth: match color.bit_depth {
			video::VideoComponentBitDepth::Eight => 8,
			video::VideoComponentBitDepth::Ten => 10,
			video::VideoComponentBitDepth::Twelve => 12,
		},
		subsampling_x,
		subsampling_y,
		reserved1: 0,
		color_primaries: u32::from(description.color_primaries),
		transfer_characteristics: u32::from(description.transfer_characteristics),
		matrix_coefficients: u32::from(description.matrix_coefficients),
		chroma_sample_position,
	}
}

#[allow(dead_code, reason = "wired by the next AV1 frame decode checkpoint")]
pub(super) fn std_av1_timing(
	timing: &video::Av1TimingInfo,
) -> ash::vk::native::StdVideoAV1TimingInfo {
	ash::vk::native::StdVideoAV1TimingInfo {
		flags: ash::vk::native::StdVideoAV1TimingInfoFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoAV1TimingInfoFlags::new_bitfield_1(
				u32::from(timing.equal_picture_interval),
				0,
			),
		},
		num_units_in_display_tick: timing.num_units_in_display_tick,
		time_scale: timing.time_scale,
		num_ticks_per_picture_minus_1: timing.num_ticks_per_picture_minus_1.unwrap_or(0),
	}
}

#[allow(dead_code, reason = "wired by the next AV1 frame decode checkpoint")]
pub(super) fn std_av1_sequence(
	sequence: &video::Av1SequenceHeader,
	color: &ash::vk::native::StdVideoAV1ColorConfig,
	timing: *const ash::vk::native::StdVideoAV1TimingInfo,
) -> Result<ash::vk::native::StdVideoAV1SequenceHeader> {
	if sequence.frame_width_bits_minus_1 > 15 || sequence.frame_height_bits_minus_1 > 15 {
		return Err(Error::data_loss(
			"AV1 frame extent bit width exceeds 16 bits",
		));
	}
	let max_width = (1_u32 << (u32::from(sequence.frame_width_bits_minus_1) + 1)) - 1;
	let max_height = (1_u32 << (u32::from(sequence.frame_height_bits_minus_1) + 1)) - 1;
	if u32::from(sequence.max_frame_width_minus_1) > max_width
		|| u32::from(sequence.max_frame_height_minus_1) > max_height
	{
		return Err(Error::data_loss(
			"AV1 coded extent exceeds its declared field width",
		));
	}
	if sequence.enable_order_hint != (sequence.order_hint_bits != 0) || sequence.order_hint_bits > 8 {
		return Err(Error::data_loss(
			"AV1 order-hint flag and bit width are inconsistent",
		));
	}
	if sequence.timing.is_none() != timing.is_null() {
		return Err(Error::internal(
			"AV1 timing pointer does not match sequence timing metadata",
		));
	}
	let tool_choice = |choice| match choice {
		video::Av1CodingToolChoice::Disabled => 0,
		video::Av1CodingToolChoice::Enabled => 1,
		video::Av1CodingToolChoice::SelectPerFrame => 2,
	};
	Ok(ash::vk::native::StdVideoAV1SequenceHeader {
		flags: ash::vk::native::StdVideoAV1SequenceHeaderFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoAV1SequenceHeaderFlags::new_bitfield_1(
				u32::from(sequence.still_picture),
				u32::from(sequence.reduced_still_picture_header),
				u32::from(sequence.use_128x128_superblock),
				u32::from(sequence.enable_filter_intra),
				u32::from(sequence.enable_intra_edge_filter),
				u32::from(sequence.enable_inter_intra_compound),
				u32::from(sequence.enable_masked_compound),
				u32::from(sequence.enable_warped_motion),
				u32::from(sequence.enable_dual_filter),
				u32::from(sequence.enable_order_hint),
				u32::from(sequence.enable_joint_compound),
				u32::from(sequence.enable_reference_frame_motion_vectors),
				u32::from(sequence.frame_id_numbers_present),
				u32::from(sequence.enable_superres),
				u32::from(sequence.enable_cdef),
				u32::from(sequence.enable_restoration),
				u32::from(sequence.film_grain_params_present),
				u32::from(sequence.timing.is_some()),
				u32::from(sequence.initial_display_delay_present),
				0,
			),
		},
		seq_profile: av1_profile(sequence.profile),
		frame_width_bits_minus_1: sequence.frame_width_bits_minus_1,
		frame_height_bits_minus_1: sequence.frame_height_bits_minus_1,
		max_frame_width_minus_1: sequence.max_frame_width_minus_1,
		max_frame_height_minus_1: sequence.max_frame_height_minus_1,
		delta_frame_id_length_minus_2: sequence.delta_frame_id_length_minus_2,
		additional_frame_id_length_minus_1: sequence.additional_frame_id_length_minus_1,
		order_hint_bits_minus_1: sequence.order_hint_bits.saturating_sub(1),
		seq_force_integer_mv: tool_choice(sequence.integer_motion_vectors),
		seq_force_screen_content_tools: tool_choice(sequence.screen_content_tools),
		reserved1: [0; 5],
		pColorConfig: color,
		pTimingInfo: timing,
	})
}
