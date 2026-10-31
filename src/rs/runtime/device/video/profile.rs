//! Private Vulkan Video profile, format, and StdVideo translation.

use crate::runtime::device::NativeFlagBits;
use crate::runtime::device::video::*;

pub(super) struct DecodeCapabilityQuery {
	pub(super) capabilities: video::VideoDecodeCapabilities,
	pub(super) std_header_version: vk::ExtensionProperties,
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
	let loader = instance.raw();

	match profile {
		video::VideoDecodeProfile::H264 {
			profile: codec_profile,
			picture_layout,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec_profile = vk::VideoDecodeH264ProfileInfoKHR::default()
				.std_profile_idc(h264_profile(codec_profile))
				.picture_layout(h264_picture_layout(picture_layout));
			let vk_profile = common_profile(
				vk::VideoCodecOperationFlagBitsKHR::DECODE_H264_BIT_KHR,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec_profile);
			let mut decode = vk::VideoDecodeCapabilitiesKHR::default();
			let mut codec = vk::VideoDecodeH264CapabilitiesKHR::default();
			let common = {
				let mut capabilities = vk::VideoCapabilitiesKHR::default()
					.push_next(&mut decode)
					.map_err(Error::from)?
					.push_next(&mut codec)
					.map_err(Error::from)?;
				query(
					loader,
					physical.handle,
					&(vk_profile).map_err(Error::from)?,
					&mut capabilities,
				)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(DecodeCapabilityQuery {
				capabilities: convert_capabilities(
					profile,
					common,
					&decode,
					video::VideoDecodeLevel::H264(codec.max_level_idc.0 as u32),
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
			let mut codec_profile =
				vk::VideoDecodeH265ProfileInfoKHR::default().std_profile_idc(h265_profile(codec_profile));
			let vk_profile = common_profile(
				vk::VideoCodecOperationFlagBitsKHR::DECODE_H265_BIT_KHR,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec_profile);
			let mut decode = vk::VideoDecodeCapabilitiesKHR::default();
			let mut codec = vk::VideoDecodeH265CapabilitiesKHR::default();
			let common = {
				let mut capabilities = vk::VideoCapabilitiesKHR::default()
					.push_next(&mut decode)
					.map_err(Error::from)?
					.push_next(&mut codec)
					.map_err(Error::from)?;
				query(
					loader,
					physical.handle,
					&(vk_profile).map_err(Error::from)?,
					&mut capabilities,
				)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(DecodeCapabilityQuery {
				capabilities: convert_capabilities(
					profile,
					common,
					&decode,
					video::VideoDecodeLevel::H265(codec.max_level_idc.0 as u32),
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
			let mut codec_profile = vk::VideoDecodeAV1ProfileInfoKHR::default()
				.std_profile(av1_profile(codec_profile))
				.film_grain_support(u32::from(film_grain_support));
			let vk_profile = common_profile(
				vk::VideoCodecOperationFlagBitsKHR::DECODE_AV1_BIT_KHR,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec_profile);
			let mut decode = vk::VideoDecodeCapabilitiesKHR::default();
			let mut codec = vk::VideoDecodeAV1CapabilitiesKHR::default();
			let common = {
				let mut capabilities = vk::VideoCapabilitiesKHR::default()
					.push_next(&mut decode)
					.map_err(Error::from)?
					.push_next(&mut codec)
					.map_err(Error::from)?;
				query(
					loader,
					physical.handle,
					&(vk_profile).map_err(Error::from)?,
					&mut capabilities,
				)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(DecodeCapabilityQuery {
				capabilities: convert_capabilities(
					profile,
					common,
					&decode,
					video::VideoDecodeLevel::Av1(codec.max_level.0 as u32),
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
			let mut codec_profile =
				vk::VideoDecodeVP9ProfileInfoKHR::default().std_profile(vp9_profile(codec_profile));
			let mut vk_profile = common_profile(
				vp9_operation(),
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			);
			vk_profile.p_next = std::ptr::from_mut(&mut codec_profile).cast();
			let mut codec = vk::VideoDecodeVP9CapabilitiesKHR::default();
			let mut decode = {
				vk::VideoDecodeCapabilitiesKHR {
					p_next: std::ptr::from_mut(&mut codec).cast(),
					..Default::default()
				}
			};
			let common = {
				let mut capabilities = {
					vk::VideoCapabilitiesKHR {
						p_next: std::ptr::from_mut(&mut decode).cast(),
						..Default::default()
					}
				};
				query(loader, physical.handle, &vk_profile, &mut capabilities)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(DecodeCapabilityQuery {
				capabilities: convert_capabilities(
					profile,
					common,
					&decode,
					video::VideoDecodeLevel::Vp9(codec.max_level.0 as u32),
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
	let loader = instance.raw();
	with_decode_profile(profile, |vk_profile| {
		let output = query_formats(
			loader,
			physical.handle,
			vk_profile,
			vk::ImageUsageFlagBits::VIDEO_DECODE_DST_BIT_KHR,
		)?;
		let dpb = query_formats(
			loader,
			physical.handle,
			vk_profile,
			vk::ImageUsageFlagBits::VIDEO_DECODE_DPB_BIT_KHR,
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
	let loader = instance.raw();
	match profile {
		video::VideoEncodeProfile::H264 {
			profile: codec_profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec_profile =
				vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(h264_profile(codec_profile));
			let vk_profile = common_profile(
				vk::VideoCodecOperationFlagBitsKHR::ENCODE_H264_BIT_KHR,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec_profile);
			let mut encode = vk::VideoEncodeCapabilitiesKHR::default();
			let mut codec = vk::VideoEncodeH264CapabilitiesKHR::default();
			let common = {
				let mut capabilities = vk::VideoCapabilitiesKHR::default()
					.push_next(&mut encode)
					.map_err(Error::from)?
					.push_next(&mut codec)
					.map_err(Error::from)?;
				query(
					loader,
					physical.handle,
					&(vk_profile).map_err(Error::from)?,
					&mut capabilities,
				)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(convert_encode_capabilities(
				profile,
				common,
				&encode,
				video::VideoEncodeCodecCapabilities::H264 {
					max_level: codec.max_level_idc.0 as u32,
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
			let mut codec_profile =
				vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(h265_profile(codec_profile));
			let vk_profile = common_profile(
				vk::VideoCodecOperationFlagBitsKHR::ENCODE_H265_BIT_KHR,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec_profile);
			let mut encode = vk::VideoEncodeCapabilitiesKHR::default();
			let mut codec = vk::VideoEncodeH265CapabilitiesKHR::default();
			let common = {
				let mut capabilities = vk::VideoCapabilitiesKHR::default()
					.push_next(&mut encode)
					.map_err(Error::from)?
					.push_next(&mut codec)
					.map_err(Error::from)?;
				query(
					loader,
					physical.handle,
					&(vk_profile).map_err(Error::from)?,
					&mut capabilities,
				)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(convert_encode_capabilities(
				profile,
				common,
				&encode,
				video::VideoEncodeCodecCapabilities::H265 {
					max_level: codec.max_level_idc.0 as u32,
					max_slice_segment_count: codec.max_slice_segment_count,
					max_tiles: extent(codec.max_tiles),
					ctb_size_16: codec
						.ctb_sizes
						.contains(vk::VideoEncodeH265CtbSizeFlagBitsKHR::_16_BIT_KHR),
					ctb_size_32: codec
						.ctb_sizes
						.contains(vk::VideoEncodeH265CtbSizeFlagBitsKHR::_32_BIT_KHR),
					ctb_size_64: codec
						.ctb_sizes
						.contains(vk::VideoEncodeH265CtbSizeFlagBitsKHR::_64_BIT_KHR),
					transform_size_4: codec
						.transform_block_sizes
						.contains(vk::VideoEncodeH265TransformBlockSizeFlagBitsKHR::_4_BIT_KHR),
					transform_size_8: codec
						.transform_block_sizes
						.contains(vk::VideoEncodeH265TransformBlockSizeFlagBitsKHR::_8_BIT_KHR),
					transform_size_16: codec
						.transform_block_sizes
						.contains(vk::VideoEncodeH265TransformBlockSizeFlagBitsKHR::_16_BIT_KHR),
					transform_size_32: codec
						.transform_block_sizes
						.contains(vk::VideoEncodeH265TransformBlockSizeFlagBitsKHR::_32_BIT_KHR),
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
	let loader = instance.raw();
	with_encode_profile(profile, |vk_profile| {
		let input = query_formats(
			loader,
			physical.handle,
			vk_profile,
			vk::ImageUsageFlagBits::VIDEO_ENCODE_SRC_BIT_KHR,
		)?;
		let dpb = query_formats(
			loader,
			physical.handle,
			vk_profile,
			vk::ImageUsageFlagBits::VIDEO_ENCODE_DPB_BIT_KHR,
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
	_nal: Box<[vk::StdVideoH265SubLayerHrdParameters]>,
	_vcl: Box<[vk::StdVideoH265SubLayerHrdParameters]>,
	value: vk::StdVideoH265HrdParameters<'static>,
}

impl StdH265Hrd {
	pub(super) fn as_ptr(&self) -> *const vk::StdVideoH265HrdParameters<'static> {
		&self.value
	}
}

pub(super) fn std_h265_profile_tier_level(
	profile: &video::H265ProfileTierLevel,
) -> Result<vk::StdVideoH265ProfileTierLevel> {
	let profile_idc = match profile.profile_idc {
		1 | 2 | 3 | 4 | 9 => profile.profile_idc,
		value => {
			return Err(Error::missing_capability(format!(
				"H.265 profile_idc {value} has no supported StdVideo mapping"
			)));
		}
	};
	Ok(vk::StdVideoH265ProfileTierLevel {
		flags: {
			let mut flags = vk::StdVideoH265ProfileTierLevelFlags::default();
			flags.set_general_tier_flag(u32::from(profile.high_tier) as _);
			flags.set_general_progressive_source_flag(u32::from(profile.progressive_source) as _);
			flags.set_general_interlaced_source_flag(u32::from(profile.interlaced_source) as _);
			flags.set_general_non_packed_constraint_flag(u32::from(profile.non_packed_constraint) as _);
			flags.set_general_frame_only_constraint_flag(u32::from(profile.frame_only_constraint) as _);
			flags
		},
		general_profile_idc: vk::StdVideoH265ProfileIdc(profile_idc as i32),
		general_level_idc: std_h265_level(profile.level_idc)?,
	})
}

pub(super) fn std_h265_level(level_idc: u32) -> Result<vk::StdVideoH265LevelIdc> {
	match level_idc {
		30 => Ok(vk::StdVideoH265LevelIdc(0)),
		60 => Ok(vk::StdVideoH265LevelIdc(1)),
		63 => Ok(vk::StdVideoH265LevelIdc(2)),
		90 => Ok(vk::StdVideoH265LevelIdc(3)),
		93 => Ok(vk::StdVideoH265LevelIdc(4)),
		120 => Ok(vk::StdVideoH265LevelIdc(5)),
		123 => Ok(vk::StdVideoH265LevelIdc(6)),
		150 => Ok(vk::StdVideoH265LevelIdc(7)),
		153 => Ok(vk::StdVideoH265LevelIdc(8)),
		156 => Ok(vk::StdVideoH265LevelIdc(9)),
		180 => Ok(vk::StdVideoH265LevelIdc(10)),
		183 => Ok(vk::StdVideoH265LevelIdc(11)),
		186 => Ok(vk::StdVideoH265LevelIdc(12)),
		value => Err(Error::missing_capability(format!(
			"H.265 level_idc {value} has no Khronos StdVideo mapping"
		))),
	}
}

pub(super) fn std_h265_dpb(
	dpb: &video::H265DecodedPictureBuffer,
) -> Result<vk::StdVideoH265DecPicBufMgr> {
	let mut max_dec_pic_buffering_minus1 = [0_u8; 7];
	let mut max_num_reorder_pics = [0_u8; 7];
	for index in 0..7 {
		max_dec_pic_buffering_minus1[index] =
			u8::try_from(dpb.max_decoded_picture_buffering_minus_1[index])
				.map_err(|_| Error::data_loss("H.265 DPB capacity exceeds StdVideo storage"))?;
		max_num_reorder_pics[index] = u8::try_from(dpb.max_num_reorder_pictures[index])
			.map_err(|_| Error::data_loss("H.265 reorder count exceeds StdVideo storage"))?;
	}
	Ok(vk::StdVideoH265DecPicBufMgr {
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
	let value = {
		vk::StdVideoH265HrdParameters {
			flags: {
				let mut flags = vk::StdVideoH265HrdFlags::default();
				flags.set_nal_hrd_parameters_present_flag(u32::from(hrd.nal_parameters_present) as _);
				flags.set_vcl_hrd_parameters_present_flag(u32::from(hrd.vcl_parameters_present) as _);
				flags
					.set_sub_pic_hrd_params_present_flag(u32::from(hrd.sub_picture_parameters_present) as _);
				flags.set_sub_pic_cpb_params_in_pic_timing_sei_flag(u32::from(
					hrd.sub_picture_cpb_parameters_in_picture_timing_sei,
				) as _);
				flags.set_fixed_pic_rate_general_flag(fixed_general as _);
				flags.set_fixed_pic_rate_within_cvs_flag(fixed_within as _);
				flags.set_low_delay_hrd_flag(low_delay as _);
				flags
			},
			tick_divisor_minus2: hrd.tick_divisor_minus_2,
			du_cpb_removal_delay_increment_length_minus1: hrd
				.du_cpb_removal_delay_increment_length_minus_1,
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
			p_sub_layer_hrd_parameters_nal: if nal.is_empty() {
				std::ptr::null()
			} else {
				nal.as_ptr()
			},
			p_sub_layer_hrd_parameters_vcl: if vcl.is_empty() {
				std::ptr::null()
			} else {
				vcl.as_ptr()
			},
			..Default::default()
		}
	};
	Ok(StdH265Hrd {
		_nal: nal,
		_vcl: vcl,
		value,
	})
}

pub(super) fn std_h265_sub_layer_hrd(
	entries: &[video::H265CpbEntry],
) -> Result<vk::StdVideoH265SubLayerHrdParameters> {
	if entries.is_empty() || entries.len() > 32 {
		return Err(Error::data_loss("H.265 HRD CPB count is outside 1..=32"));
	}
	let mut result = vk::StdVideoH265SubLayerHrdParameters {
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

pub(super) fn std_h265_vps<'a>(
	vps: &video::H265VideoParameterSet,
	profile: &vk::StdVideoH265ProfileTierLevel,
	dpb: &vk::StdVideoH265DecPicBufMgr,
	hrd: *const vk::StdVideoH265HrdParameters<'a>,
) -> Result<vk::StdVideoH265VideoParameterSet<'a>> {
	let timing = vps.timing.unwrap_or_default();
	Ok({
		vk::StdVideoH265VideoParameterSet {
			flags: {
				let mut flags = vk::StdVideoH265VpsFlags::default();
				flags.set_vps_temporal_id_nesting_flag(u32::from(vps.temporal_id_nesting) as _);
				flags.set_vps_sub_layer_ordering_info_present_flag(u32::from(
					vps.sub_layer_ordering_info_present,
				) as _);
				flags.set_vps_timing_info_present_flag(u32::from(vps.timing.is_some()) as _);
				flags.set_vps_poc_proportional_to_timing_flag(u32::from(
					timing.num_ticks_poc_diff_one_minus_1.is_some(),
				) as _);
				flags
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
			p_dec_pic_buf_mgr: dpb,
			p_hrd_parameters: hrd,
			p_profile_tier_level: profile,
			..Default::default()
		}
	})
}

pub(super) fn std_h265_sps<'a>(
	sps: &video::H265SequenceParameterSet,
	profile: &vk::StdVideoH265ProfileTierLevel,
	dpb: &vk::StdVideoH265DecPicBufMgr,
	scaling: *const vk::StdVideoH265ScalingLists,
	short_term: &'a [vk::StdVideoH265ShortTermRefPicSet],
	long_term: Option<&'a vk::StdVideoH265LongTermRefPicsSps>,
	vui: Option<&'a vk::StdVideoH265SequenceParameterSetVui<'a>>,
) -> Result<vk::StdVideoH265SequenceParameterSet<'a>> {
	let pcm = sps.pcm.unwrap_or_default();
	Ok({
		vk::StdVideoH265SequenceParameterSet {
			flags: {
				let mut flags = vk::StdVideoH265SpsFlags::default();
				flags.set_sps_temporal_id_nesting_flag(u32::from(sps.temporal_id_nesting) as _);
				flags.set_separate_colour_plane_flag(u32::from(sps.separate_colour_plane) as _);
				flags.set_conformance_window_flag(u32::from(sps.conformance_window != [0; 4]) as _);
				flags.set_sps_sub_layer_ordering_info_present_flag(u32::from(
					sps.sub_layer_ordering_info_present,
				) as _);
				flags.set_scaling_list_enabled_flag(u32::from(sps.scaling_list_enabled) as _);
				flags.set_sps_scaling_list_data_present_flag(u32::from(sps.scaling_lists.is_some()) as _);
				flags.set_amp_enabled_flag(u32::from(sps.asymmetric_motion_partitions_enabled) as _);
				flags.set_sample_adaptive_offset_enabled_flag(
					u32::from(sps.sample_adaptive_offset_enabled) as _,
				);
				flags.set_pcm_enabled_flag(u32::from(sps.pcm.is_some()) as _);
				flags.set_pcm_loop_filter_disabled_flag(u32::from(pcm.loop_filter_disabled) as _);
				flags.set_long_term_ref_pics_present_flag(u32::from(
					!sps.long_term_reference_pictures.is_empty(),
				) as _);
				flags.set_sps_temporal_mvp_enabled_flag(u32::from(sps.temporal_mvp_enabled) as _);
				flags.set_strong_intra_smoothing_enabled_flag(
					u32::from(sps.strong_intra_smoothing_enabled) as _,
				);
				flags.set_vui_parameters_present_flag(u32::from(sps.vui.is_some()) as _);
				flags.set_sps_extension_present_flag(0 as _);
				flags.set_sps_range_extension_flag(0 as _);
				flags.set_transform_skip_rotation_enabled_flag(0 as _);
				flags.set_transform_skip_context_enabled_flag(0 as _);
				flags.set_implicit_rdpcm_enabled_flag(0 as _);
				flags.set_explicit_rdpcm_enabled_flag(0 as _);
				flags.set_extended_precision_processing_flag(0 as _);
				flags.set_intra_smoothing_disabled_flag(0 as _);
				flags.set_high_precision_offsets_enabled_flag(0 as _);
				flags.set_persistent_rice_adaptation_enabled_flag(0 as _);
				flags.set_cabac_bypass_alignment_enabled_flag(0 as _);
				flags.set_sps_scc_extension_flag(0 as _);
				flags.set_sps_curr_pic_ref_enabled_flag(0 as _);
				flags.set_palette_mode_enabled_flag(0 as _);
				flags.set_sps_palette_predictor_initializers_present_flag(0 as _);
				flags.set_intra_boundary_filtering_disabled_flag(0 as _);
				flags
			},
			chroma_format_idc: vk::StdVideoH265ChromaFormatIdc(sps.chroma_format_idc as i32),
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
			p_profile_tier_level: profile,
			p_dec_pic_buf_mgr: dpb,
			p_scaling_lists: scaling,
			p_short_term_ref_pic_set: if short_term.is_empty() {
				std::ptr::null()
			} else {
				short_term.as_ptr()
			},
			p_long_term_ref_pics_sps: long_term.map_or(std::ptr::null(), |value| value as *const _),
			p_sequence_parameter_set_vui: vui.map_or(std::ptr::null(), |value| value as *const _),
			p_predictor_palette_entries: std::ptr::null(),
			..Default::default()
		}
	})
}

pub(super) fn std_h265_short_term_reference_set(
	set: &video::H265ShortTermReferencePictureSet,
) -> Result<vk::StdVideoH265ShortTermRefPicSet> {
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
	Ok(vk::StdVideoH265ShortTermRefPicSet {
		flags: {
			let mut flags = vk::StdVideoH265ShortTermRefPicSetFlags::default();
			flags.set_inter_ref_pic_set_prediction_flag(u32::from(set.inter_ref_pic_set_prediction) as _);
			flags.set_delta_rps_sign(u32::from(set.delta_rps_sign) as _);
			flags
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
) -> Result<Option<vk::StdVideoH265LongTermRefPicsSps>> {
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
	Ok(Some(vk::StdVideoH265LongTermRefPicsSps {
		used_by_curr_pic_lt_sps_flag,
		lt_ref_pic_poc_lsb_sps,
	}))
}

pub(super) fn std_h265_scaling(scaling: &video::H265ScalingLists) -> vk::StdVideoH265ScalingLists {
	vk::StdVideoH265ScalingLists {
		scaling_list4x4: scaling.list_4x4,
		scaling_list8x8: scaling.list_8x8,
		scaling_list16x16: scaling.list_16x16,
		scaling_list32x32: scaling.list_32x32,
		scaling_list_dc_coef16x16: scaling.dc_16x16,
		scaling_list_dc_coef32x32: scaling.dc_32x32,
	}
}

pub(super) fn std_h265_vui<'a>(
	vui: &video::H265VuiParameters,
	hrd: *const vk::StdVideoH265HrdParameters<'a>,
) -> Result<vk::StdVideoH265SequenceParameterSetVui<'a>> {
	let aspect = vui.aspect_ratio.unwrap_or_default();
	let signal = vui.video_signal.unwrap_or_default();
	let colour = signal.colour_description.unwrap_or_default();
	let chroma = vui.chroma_location.unwrap_or_default();
	let display = vui.default_display_window.unwrap_or([0; 4]);
	let timing = vui.timing.unwrap_or_default();
	let restriction = vui.bitstream_restriction.unwrap_or_default();
	Ok({
		vk::StdVideoH265SequenceParameterSetVui {
			flags: {
				let mut flags = vk::StdVideoH265SpsVuiFlags::default();
				flags.set_aspect_ratio_info_present_flag(u32::from(vui.aspect_ratio.is_some()) as _);
				flags.set_overscan_info_present_flag(u32::from(vui.overscan_appropriate.is_some()) as _);
				flags
					.set_overscan_appropriate_flag(u32::from(vui.overscan_appropriate.unwrap_or(false)) as _);
				flags.set_video_signal_type_present_flag(u32::from(vui.video_signal.is_some()) as _);
				flags.set_video_full_range_flag(u32::from(signal.full_range) as _);
				flags
					.set_colour_description_present_flag(u32::from(signal.colour_description.is_some()) as _);
				flags.set_chroma_loc_info_present_flag(u32::from(vui.chroma_location.is_some()) as _);
				flags.set_neutral_chroma_indication_flag(u32::from(vui.neutral_chroma_indication) as _);
				flags.set_field_seq_flag(u32::from(vui.field_sequence) as _);
				flags.set_frame_field_info_present_flag(u32::from(vui.frame_field_info_present) as _);
				flags.set_default_display_window_flag(u32::from(vui.default_display_window.is_some()) as _);
				flags.set_vui_timing_info_present_flag(u32::from(vui.timing.is_some()) as _);
				flags.set_vui_poc_proportional_to_timing_flag(u32::from(
					timing.num_ticks_poc_diff_one_minus_1.is_some(),
				) as _);
				flags.set_vui_hrd_parameters_present_flag(u32::from(vui.hrd.is_some()) as _);
				flags.set_bitstream_restriction_flag(u32::from(vui.bitstream_restriction.is_some()) as _);
				flags.set_tiles_fixed_structure_flag(u32::from(restriction.tiles_fixed_structure) as _);
				flags.set_motion_vectors_over_pic_boundaries_flag(u32::from(
					restriction.motion_vectors_over_picture_boundaries,
				) as _);
				flags.set_restricted_ref_pic_lists_flag(u32::from(
					restriction.restricted_reference_picture_lists,
				) as _);
				flags
			},
			aspect_ratio_idc: vk::StdVideoH265AspectRatioIdc(aspect.idc as i32),
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
			def_disp_win_left_offset: u16::try_from(display[0]).map_err(|_| {
				Error::data_loss("H.265 display-window left offset exceeds StdVideo storage")
			})?,
			def_disp_win_right_offset: u16::try_from(display[1]).map_err(|_| {
				Error::data_loss("H.265 display-window right offset exceeds StdVideo storage")
			})?,
			def_disp_win_top_offset: u16::try_from(display[2]).map_err(|_| {
				Error::data_loss("H.265 display-window top offset exceeds StdVideo storage")
			})?,
			def_disp_win_bottom_offset: u16::try_from(display[3]).map_err(|_| {
				Error::data_loss("H.265 display-window bottom offset exceeds StdVideo storage")
			})?,
			vui_num_units_in_tick: timing.num_units_in_tick,
			vui_time_scale: timing.time_scale,
			vui_num_ticks_poc_diff_one_minus1: timing.num_ticks_poc_diff_one_minus_1.unwrap_or(0),
			min_spatial_segmentation_idc: u16::try_from(restriction.min_spatial_segmentation_idc)
				.map_err(|_| {
					Error::data_loss("H.265 minimum spatial segmentation exceeds StdVideo storage")
				})?,
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
			p_hrd_parameters: hrd,
			..Default::default()
		}
	})
}

pub(super) fn std_h265_pps(
	pps: &video::H265PictureParameterSet,
	video_parameter_set_id: u32,
	scaling: *const vk::StdVideoH265ScalingLists,
) -> Result<vk::StdVideoH265PictureParameterSet<'static>> {
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
	Ok({
		vk::StdVideoH265PictureParameterSet {
			flags: {
				let mut flags = vk::StdVideoH265PpsFlags::default();
				flags
					.set_dependent_slice_segments_enabled_flag((pps.dependent_slice_segments_enabled) as _);
				flags.set_output_flag_present_flag((pps.output_flag_present) as _);
				flags.set_sign_data_hiding_enabled_flag((pps.sign_data_hiding_enabled) as _);
				flags.set_cabac_init_present_flag((pps.cabac_init_present) as _);
				flags.set_constrained_intra_pred_flag((pps.constrained_intra_pred) as _);
				flags.set_transform_skip_enabled_flag((pps.transform_skip_enabled) as _);
				flags.set_cu_qp_delta_enabled_flag((pps.cu_qp_delta_enabled) as _);
				flags
					.set_pps_slice_chroma_qp_offsets_present_flag((pps.slice_chroma_qp_offsets_present) as _);
				flags.set_weighted_pred_flag((pps.weighted_pred) as _);
				flags.set_weighted_bipred_flag((pps.weighted_bipred) as _);
				flags.set_transquant_bypass_enabled_flag((pps.transquant_bypass_enabled) as _);
				flags.set_tiles_enabled_flag((pps.tiles_enabled) as _);
				flags.set_entropy_coding_sync_enabled_flag((pps.entropy_coding_sync_enabled) as _);
				flags.set_uniform_spacing_flag((pps.uniform_spacing) as _);
				flags
					.set_loop_filter_across_tiles_enabled_flag((pps.loop_filter_across_tiles_enabled) as _);
				flags.set_pps_loop_filter_across_slices_enabled_flag(
					(pps.loop_filter_across_slices_enabled) as _,
				);
				flags
					.set_deblocking_filter_control_present_flag((pps.deblocking_filter_control_present) as _);
				flags.set_deblocking_filter_override_enabled_flag(
					(pps.deblocking_filter_override_enabled) as _,
				);
				flags.set_pps_deblocking_filter_disabled_flag((pps.deblocking_filter_disabled) as _);
				flags.set_pps_scaling_list_data_present_flag((pps.scaling_lists.is_some()) as _);
				flags.set_lists_modification_present_flag((pps.lists_modification_present) as _);
				flags.set_slice_segment_header_extension_present_flag(
					(pps.slice_segment_header_extension_present) as _,
				);
				flags.set_pps_extension_present_flag((pps.extension_present) as _);
				flags
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
			p_scaling_lists: scaling,
			p_predictor_palette_entries: std::ptr::null(),
			..Default::default()
		}
	})
}

pub(super) fn h265_u8(value: u32, field: &str) -> Result<u8> {
	u8::try_from(value)
		.map_err(|_| Error::data_loss(format!("H.265 {field} exceeds StdVideo storage")))
}

pub(super) fn std_h264_sps<'a>(
	sps: &video::H264SequenceParameterSet,
	std_scaling: *const vk::StdVideoH264ScalingLists,
	std_vui: *const vk::StdVideoH264SequenceParameterSetVui<'a>,
) -> Result<vk::StdVideoH264SequenceParameterSet<'a>> {
	let mut flags = vk::StdVideoH264SpsFlags::default();
	flags.set_constraint_set0_flag((sps.constraint_flags & 0x80 != 0) as _);
	flags.set_constraint_set1_flag((sps.constraint_flags & 0x40 != 0) as _);
	flags.set_constraint_set2_flag((sps.constraint_flags & 0x20 != 0) as _);
	flags.set_constraint_set3_flag((sps.constraint_flags & 0x10 != 0) as _);
	flags.set_constraint_set4_flag((sps.constraint_flags & 0x08 != 0) as _);
	flags.set_constraint_set5_flag((sps.constraint_flags & 0x04 != 0) as _);
	flags.set_direct_8x8_inference_flag((sps.direct_8x8_inference) as _);
	flags.set_mb_adaptive_frame_field_flag((sps.mb_adaptive_frame_field) as _);
	flags.set_frame_mbs_only_flag((sps.frame_mbs_only) as _);
	flags.set_delta_pic_order_always_zero_flag((sps.delta_pic_order_always_zero) as _);
	flags.set_separate_colour_plane_flag((sps.separate_colour_plane) as _);
	flags.set_gaps_in_frame_num_value_allowed_flag((sps.gaps_in_frame_num_value_allowed) as _);
	flags.set_qpprime_y_zero_transform_bypass_flag((sps.qpprime_y_zero_transform_bypass) as _);
	flags.set_frame_cropping_flag(
		(sps.frame_crop_left_offset != 0
			|| sps.frame_crop_right_offset != 0
			|| sps.frame_crop_top_offset != 0
			|| sps.frame_crop_bottom_offset != 0) as _,
	);
	flags.set_seq_scaling_matrix_present_flag((sps.scaling_lists.is_some()) as _);
	flags.set_vui_parameters_present_flag((sps.vui.is_some()) as _);
	let flags = flags;
	Ok({
		vk::StdVideoH264SequenceParameterSet {
			flags,
			profile_idc: vk::StdVideoH264ProfileIdc(sps.profile_idc as i32),
			level_idc: std_h264_level(sps.level_idc)?,
			chroma_format_idc: vk::StdVideoH264ChromaFormatIdc(sps.chroma_format_idc as i32),
			seq_parameter_set_id: u8::try_from(sps.id)
				.map_err(|_| Error::data_loss("H.264 SPS id exceeds StdVideo storage"))?,
			bit_depth_luma_minus8: u8::try_from(sps.bit_depth_luma_minus_8)
				.map_err(|_| Error::data_loss("H.264 luma bit depth exceeds StdVideo storage"))?,
			bit_depth_chroma_minus8: u8::try_from(sps.bit_depth_chroma_minus_8)
				.map_err(|_| Error::data_loss("H.264 chroma bit depth exceeds StdVideo storage"))?,
			log2_max_frame_num_minus4: u8::try_from(sps.log2_max_frame_num_minus_4)
				.map_err(|_| Error::data_loss("H.264 frame-number width exceeds StdVideo storage"))?,
			pic_order_cnt_type: vk::StdVideoH264PocType(sps.pic_order_count_type as i32),
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
			p_offset_for_ref_frame: if sps.offset_for_ref_frame.is_empty() {
				std::ptr::null()
			} else {
				sps.offset_for_ref_frame.as_ptr()
			},
			p_scaling_lists: std_scaling,
			p_sequence_parameter_set_vui: std_vui,
			..Default::default()
		}
	})
}
pub(super) fn std_h264_hrd(
	vui: &video::H264VuiParameters,
) -> Result<Option<vk::StdVideoH264HrdParameters>> {
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
	Ok(Some(vk::StdVideoH264HrdParameters {
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
pub(super) fn std_h264_pps(
	pps: &video::H264PictureParameterSet,
	std_scaling: *const vk::StdVideoH264ScalingLists,
) -> Result<vk::StdVideoH264PictureParameterSet<'static>> {
	let flags = {
		let mut flags = vk::StdVideoH264PpsFlags::default();
		flags.set_transform_8x8_mode_flag(u32::from(pps.transform_8x8_mode) as _);
		flags.set_redundant_pic_cnt_present_flag(u32::from(pps.redundant_pic_count_present) as _);
		flags.set_constrained_intra_pred_flag(u32::from(pps.constrained_intra_pred) as _);
		flags.set_deblocking_filter_control_present_flag(u32::from(
			pps.deblocking_filter_control_present,
		) as _);
		flags.set_weighted_pred_flag(u32::from(pps.weighted_pred) as _);
		flags.set_bottom_field_pic_order_in_frame_present_flag(u32::from(
			pps.bottom_field_pic_order_in_frame_present,
		) as _);
		flags.set_entropy_coding_mode_flag(u32::from(pps.entropy_coding_mode) as _);
		flags.set_pic_scaling_matrix_present_flag(u32::from(pps.scaling_lists.is_some()) as _);
		flags
	};
	Ok({
		vk::StdVideoH264PictureParameterSet {
			flags,
			seq_parameter_set_id: u8::try_from(pps.sequence_parameter_set_id)
				.map_err(|_| Error::data_loss("H.264 PPS SPS id exceeds StdVideo storage"))?,
			pic_parameter_set_id: u8::try_from(pps.id)
				.map_err(|_| Error::data_loss("H.264 PPS id exceeds StdVideo storage"))?,
			num_ref_idx_l0_default_active_minus1: u8::try_from(pps.num_ref_idx_l0_default_active_minus_1)
				.map_err(|_| Error::data_loss("H.264 L0 reference count exceeds StdVideo storage"))?,
			num_ref_idx_l1_default_active_minus1: u8::try_from(pps.num_ref_idx_l1_default_active_minus_1)
				.map_err(|_| Error::data_loss("H.264 L1 reference count exceeds StdVideo storage"))?,
			weighted_bipred_idc: vk::StdVideoH264WeightedBipredIdc(pps.weighted_bipred_idc as i32),
			pic_init_qp_minus26: i8::try_from(pps.pic_init_qp_minus_26)
				.map_err(|_| Error::data_loss("H.264 initial QP exceeds StdVideo storage"))?,
			pic_init_qs_minus26: i8::try_from(pps.pic_init_qs_minus_26)
				.map_err(|_| Error::data_loss("H.264 initial QS exceeds StdVideo storage"))?,
			chroma_qp_index_offset: i8::try_from(pps.chroma_qp_index_offset)
				.map_err(|_| Error::data_loss("H.264 chroma QP offset exceeds StdVideo storage"))?,
			second_chroma_qp_index_offset: i8::try_from(pps.second_chroma_qp_index_offset)
				.map_err(|_| Error::data_loss("H.264 second chroma QP offset exceeds StdVideo storage"))?,
			p_scaling_lists: std_scaling,
			..Default::default()
		}
	})
}

pub(super) fn std_h264_scaling(scaling: &video::H264ScalingLists) -> vk::StdVideoH264ScalingLists {
	vk::StdVideoH264ScalingLists {
		scaling_list_present_mask: scaling.present_mask,
		use_default_scaling_matrix_mask: scaling.use_default_mask,
		scaling_list4x4: scaling.list_4x4,
		scaling_list8x8: scaling.list_8x8,
	}
}

pub(super) fn std_h264_vui<'a>(
	vui: &video::H264VuiParameters,
	std_hrd: *const vk::StdVideoH264HrdParameters,
) -> Result<vk::StdVideoH264SequenceParameterSetVui<'a>> {
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
	let flags = {
		let mut flags = vk::StdVideoH264SpsVuiFlags::default();
		flags.set_aspect_ratio_info_present_flag((vui.aspect_ratio.is_some()) as _);
		flags.set_overscan_info_present_flag((vui.overscan_appropriate.is_some()) as _);
		flags.set_overscan_appropriate_flag((vui.overscan_appropriate.unwrap_or(false)) as _);
		flags.set_video_signal_type_present_flag((vui.video_signal.is_some()) as _);
		flags.set_video_full_range_flag((signal.full_range) as _);
		flags.set_color_description_present_flag((signal.colour_description.is_some()) as _);
		flags.set_chroma_loc_info_present_flag((vui.chroma_location.is_some()) as _);
		flags.set_timing_info_present_flag((vui.timing.is_some()) as _);
		flags.set_fixed_frame_rate_flag((timing.fixed_frame_rate) as _);
		flags.set_bitstream_restriction_flag((vui.bitstream_restriction.is_some()) as _);
		flags.set_nal_hrd_parameters_present_flag((vui.nal_hrd.is_some()) as _);
		flags.set_vcl_hrd_parameters_present_flag((vui.vcl_hrd.is_some()) as _);
		flags
	};
	Ok({
		vk::StdVideoH264SequenceParameterSetVui {
			flags,
			aspect_ratio_idc: vk::StdVideoH264AspectRatioIdc(aspect.idc as i32),
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
			p_hrd_parameters: std_hrd,
			..Default::default()
		}
	})
}

pub(super) fn std_h264_level(level_idc: u32) -> Result<vk::StdVideoH264LevelIdc> {
	match level_idc {
		10 => Ok(vk::StdVideoH264LevelIdc(0)),
		11 => Ok(vk::StdVideoH264LevelIdc(1)),
		12 => Ok(vk::StdVideoH264LevelIdc(2)),
		13 => Ok(vk::StdVideoH264LevelIdc(3)),
		20 => Ok(vk::StdVideoH264LevelIdc(4)),
		21 => Ok(vk::StdVideoH264LevelIdc(5)),
		22 => Ok(vk::StdVideoH264LevelIdc(6)),
		30 => Ok(vk::StdVideoH264LevelIdc(7)),
		31 => Ok(vk::StdVideoH264LevelIdc(8)),
		32 => Ok(vk::StdVideoH264LevelIdc(9)),
		40 => Ok(vk::StdVideoH264LevelIdc(10)),
		41 => Ok(vk::StdVideoH264LevelIdc(11)),
		42 => Ok(vk::StdVideoH264LevelIdc(12)),
		50 => Ok(vk::StdVideoH264LevelIdc(13)),
		51 => Ok(vk::StdVideoH264LevelIdc(14)),
		52 => Ok(vk::StdVideoH264LevelIdc(15)),
		60 => Ok(vk::StdVideoH264LevelIdc(16)),
		61 => Ok(vk::StdVideoH264LevelIdc(17)),
		62 => Ok(vk::StdVideoH264LevelIdc(18)),
		_ => Err(Error::missing_capability(format!(
			"H.264 level_idc {level_idc} has no Khronos StdVideo mapping"
		))),
	}
}

pub(super) fn with_decode_profile<T>(
	profile: video::VideoDecodeProfile,
	query: impl FnOnce(&vk::VideoProfileInfoKHR<'_>) -> Result<T>,
) -> Result<T> {
	match profile {
		video::VideoDecodeProfile::H264 {
			profile,
			picture_layout,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec = vk::VideoDecodeH264ProfileInfoKHR::default()
				.std_profile_idc(h264_profile(profile))
				.picture_layout(h264_picture_layout(picture_layout));
			let profile = common_profile(
				vk::VideoCodecOperationFlagBitsKHR::DECODE_H264_BIT_KHR,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec);
			query(&(profile).map_err(Error::from)?)
		}
		video::VideoDecodeProfile::H265 {
			profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec =
				vk::VideoDecodeH265ProfileInfoKHR::default().std_profile_idc(h265_profile(profile));
			let profile = common_profile(
				vk::VideoCodecOperationFlagBitsKHR::DECODE_H265_BIT_KHR,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec);
			query(&(profile).map_err(Error::from)?)
		}
		video::VideoDecodeProfile::Av1 {
			profile,
			film_grain_support,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec = vk::VideoDecodeAV1ProfileInfoKHR::default()
				.std_profile(av1_profile(profile))
				.film_grain_support(u32::from(film_grain_support));
			let profile = common_profile(
				vk::VideoCodecOperationFlagBitsKHR::DECODE_AV1_BIT_KHR,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec);
			query(&(profile).map_err(Error::from)?)
		}
		video::VideoDecodeProfile::Vp9 {
			profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec = vk::VideoDecodeVP9ProfileInfoKHR::default().std_profile(vp9_profile(profile));
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
	query: impl FnOnce(&vk::VideoProfileInfoKHR<'_>) -> Result<T>,
) -> Result<T> {
	match profile {
		video::VideoEncodeProfile::H264 {
			profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec =
				vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(h264_profile(profile));
			let profile = common_profile(
				vk::VideoCodecOperationFlagBitsKHR::ENCODE_H264_BIT_KHR,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec);
			query(&(profile).map_err(Error::from)?)
		}
		video::VideoEncodeProfile::H265 {
			profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec =
				vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(h265_profile(profile));
			let profile = common_profile(
				vk::VideoCodecOperationFlagBitsKHR::ENCODE_H265_BIT_KHR,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec);
			query(&(profile).map_err(Error::from)?)
		}
	}
}

pub(super) fn query_formats(
	loader: &vk::Instance,
	physical: vk::PhysicalDevice,
	profile: &vk::VideoProfileInfoKHR<'_>,
	usage: vk::ImageUsageFlags,
) -> Result<Vec<vk::VideoFormatPropertiesKHR<'static>>> {
	let profiles = std::slice::from_ref(profile);
	let mut profile_list = vk::VideoProfileListInfoKHR::default()
		.profiles(profiles)
		.map_err(Error::from)?;
	let format_info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
		.image_usage(usage)
		.push_next(&mut profile_list)
		.map_err(Error::from)?;
	let mut count = 0_u32;
	// safety: every pointer in `format_info` remains live, and a null output
	// pointer requests only the required element count.
	let result = unsafe {
		loader.get_physical_device_video_format_properties_khr(
			physical,
			&(format_info),
			&mut count,
			std::ptr::null_mut(),
		)
	};
	match result {
		Ok(vk::Result::SUCCESS) => {}
		_ => {
			return Err(Error::backend_dispatch_failure(
				"Vulkan",
				"video-format count query",
				result,
			));
		}
	}
	if count > MAX_VIDEO_FORMATS {
		return Err(Error::resource_exhausted(format!(
			"Vulkan reported {count} video formats, above OA's {MAX_VIDEO_FORMATS} safety bound"
		)));
	}
	let mut formats = (0..count)
		.map(|_| vk::VideoFormatPropertiesKHR::default())
		.collect::<Vec<_>>();
	if count == 0 {
		return Ok(formats);
	}
	// safety: `formats` contains `count` initialized output structures with valid
	// sType values; Vulkan writes at most the count passed by mutable pointer.
	let result = unsafe {
		loader.get_physical_device_video_format_properties_khr(
			physical,
			&(format_info),
			&mut count,
			formats.as_mut_ptr(),
		)
	};
	match result {
		Ok(vk::Result::SUCCESS) => {}
		_ => {
			return Err(Error::backend_dispatch_failure(
				"Vulkan",
				"video-format enumeration",
				result,
			));
		}
	}
	formats.truncate(count as usize);
	Ok(formats)
}

pub(super) fn convert_formats(
	formats: &[vk::VideoFormatPropertiesKHR<'_>],
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
			optimal_tiling: format.image_tiling == vk::ImageTiling::OPTIMAL,
			sampled: format
				.image_usage_flags
				.contains(vk::ImageUsageFlagBits::SAMPLED_BIT),
			storage: format
				.image_usage_flags
				.contains(vk::ImageUsageFlagBits::STORAGE_BIT),
			transfer_source: format
				.image_usage_flags
				.contains(vk::ImageUsageFlagBits::TRANSFER_SRC_BIT),
			transfer_destination: format
				.image_usage_flags
				.contains(vk::ImageUsageFlagBits::TRANSFER_DST_BIT),
		});
	}
	converted.sort_by_key(|format| format.pixel_format);
	converted.dedup();
	(converted, unrecognized)
}

pub(super) fn pixel_format(format: vk::Format) -> Option<video::VideoPixelFormat> {
	match format {
		vk::Format::G8_B8_R8_3PLANE_420_UNORM => Some(video::VideoPixelFormat::Yuv420Planar8),
		vk::Format::G8_B8R8_2PLANE_420_UNORM => Some(video::VideoPixelFormat::Nv12),
		vk::Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16 => {
			Some(video::VideoPixelFormat::Yuv420Planar10)
		}
		vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 => Some(video::VideoPixelFormat::P010),
		vk::Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16 => Some(video::VideoPixelFormat::P012),
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
	operation: vk::VideoCodecOperationFlagsKHR,
	chroma: video::VideoChromaSubsampling,
	luma_depth: video::VideoComponentBitDepth,
	chroma_depth: video::VideoComponentBitDepth,
) -> vk::VideoProfileInfoKHR<'a> {
	vk::VideoProfileInfoKHR::default()
		.video_codec_operation(operation)
		.chroma_subsampling(chroma_subsampling(chroma))
		.luma_bit_depth(bit_depth(luma_depth))
		.chroma_bit_depth(bit_depth(chroma_depth))
}

pub(super) fn query(
	loader: &vk::Instance,
	physical: vk::PhysicalDevice,
	profile: &vk::VideoProfileInfoKHR<'_>,
	capabilities: &mut vk::VideoCapabilitiesKHR<'_>,
) -> Result<()> {
	// safety: the loader was created from the live instance that owns `physical`;
	// both root structures and their required decode/codec-specific pNext chains
	// remain live and writable for the duration of the call.
	let result =
		unsafe { loader.get_physical_device_video_capabilities_khr(physical, profile, capabilities) };
	match result {
		Ok(vk::Result::SUCCESS) => Ok(()),
		Err(vk::CallError::Vulkan(
			vk::Result::ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR
			| vk::Result::ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR
			| vk::Result::ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR,
		)) => Err(Error::missing_capability(format!(
			"Vulkan rejected the requested video profile: {result:?}"
		))),
		_ => Err(Error::backend_dispatch_failure(
			"Vulkan",
			"video-profile capability query",
			result,
		)),
	}
}

pub(super) fn convert_capabilities(
	profile: video::VideoDecodeProfile,
	capabilities: CommonCapabilities,
	decode: &vk::VideoDecodeCapabilitiesKHR<'_>,
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
			.contains(vk::VideoDecodeCapabilityFlagBitsKHR::DPB_AND_OUTPUT_COINCIDE_BIT_KHR),
		dpb_and_output_distinct: decode
			.flags
			.contains(vk::VideoDecodeCapabilityFlagBitsKHR::DPB_AND_OUTPUT_DISTINCT_BIT_KHR),
		protected_content: capabilities
			.flags
			.contains(vk::VideoCapabilityFlagBitsKHR::PROTECTED_CONTENT_BIT_KHR),
		separate_reference_images: capabilities
			.flags
			.contains(vk::VideoCapabilityFlagBitsKHR::SEPARATE_REFERENCE_IMAGES_BIT_KHR),
	}
}

pub(super) fn convert_encode_capabilities(
	profile: video::VideoEncodeProfile,
	capabilities: CommonCapabilities,
	encode: &vk::VideoEncodeCapabilitiesKHR<'_>,
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
			.contains(vk::VideoEncodeRateControlModeFlagBitsKHR::DISABLED_BIT_KHR),
		cbr: encode
			.rate_control_modes
			.contains(vk::VideoEncodeRateControlModeFlagBitsKHR::CBR_BIT_KHR),
		vbr: encode
			.rate_control_modes
			.contains(vk::VideoEncodeRateControlModeFlagBitsKHR::VBR_BIT_KHR),
		feedback_offset: encode
			.supported_encode_feedback_flags
			.contains(vk::VideoEncodeFeedbackFlagBitsKHR::BITSTREAM_BUFFER_OFFSET_BIT_KHR),
		feedback_bytes_written: encode
			.supported_encode_feedback_flags
			.contains(vk::VideoEncodeFeedbackFlagBitsKHR::BITSTREAM_BYTES_WRITTEN_BIT_KHR),
		feedback_overrides: encode
			.supported_encode_feedback_flags
			.contains(vk::VideoEncodeFeedbackFlagBitsKHR::BITSTREAM_HAS_OVERRIDES_BIT_KHR),
		protected_content: capabilities
			.flags
			.contains(vk::VideoCapabilityFlagBitsKHR::PROTECTED_CONTENT_BIT_KHR),
		separate_reference_images: capabilities
			.flags
			.contains(vk::VideoCapabilityFlagBitsKHR::SEPARATE_REFERENCE_IMAGES_BIT_KHR),
		codec,
	}
}

#[derive(Clone, Copy)]
pub(super) struct CommonCapabilities {
	pub(super) flags: vk::VideoCapabilityFlagsKHR,
	pub(super) std_header_version: vk::ExtensionProperties,
	pub(super) min_bitstream_buffer_offset_alignment: u64,
	pub(super) min_bitstream_buffer_size_alignment: u64,
	pub(super) picture_access_granularity: video::VideoExtent,
	pub(super) min_coded_extent: video::VideoExtent,
	pub(super) max_coded_extent: video::VideoExtent,
	pub(super) max_dpb_slots: u32,
	pub(super) max_active_reference_pictures: u32,
}

impl From<&vk::VideoCapabilitiesKHR<'_>> for CommonCapabilities {
	fn from(value: &vk::VideoCapabilitiesKHR<'_>) -> Self {
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

pub(super) const fn extent(value: vk::Extent2D) -> video::VideoExtent {
	video::VideoExtent {
		width: value.width,
		height: value.height,
	}
}

pub(super) const fn chroma_subsampling(
	value: video::VideoChromaSubsampling,
) -> vk::VideoChromaSubsamplingFlagsKHR {
	match value {
		video::VideoChromaSubsampling::Monochrome => {
			vk::VideoChromaSubsamplingFlagBitsKHR::MONOCHROME_BIT_KHR
		}
		video::VideoChromaSubsampling::Yuv420 => vk::VideoChromaSubsamplingFlagBitsKHR::_420_BIT_KHR,
		video::VideoChromaSubsampling::Yuv422 => vk::VideoChromaSubsamplingFlagBitsKHR::_422_BIT_KHR,
		video::VideoChromaSubsampling::Yuv444 => vk::VideoChromaSubsamplingFlagBitsKHR::_444_BIT_KHR,
	}
}

pub(super) const fn bit_depth(
	value: video::VideoComponentBitDepth,
) -> vk::VideoComponentBitDepthFlagsKHR {
	match value {
		video::VideoComponentBitDepth::Eight => vk::VideoComponentBitDepthFlagBitsKHR::_8_BIT_KHR,
		video::VideoComponentBitDepth::Ten => vk::VideoComponentBitDepthFlagBitsKHR::_10_BIT_KHR,
		video::VideoComponentBitDepth::Twelve => vk::VideoComponentBitDepthFlagBitsKHR::_12_BIT_KHR,
	}
}

pub(super) const fn h264_profile(value: video::H264Profile) -> vk::StdVideoH264ProfileIdc {
	match value {
		video::H264Profile::Baseline => vk::StdVideoH264ProfileIdc::BASELINE,
		video::H264Profile::Main => vk::StdVideoH264ProfileIdc::MAIN,
		video::H264Profile::High => vk::StdVideoH264ProfileIdc::HIGH,
		video::H264Profile::High444Predictive => vk::StdVideoH264ProfileIdc::HIGH_444_PREDICTIVE,
	}
}

pub(super) const fn h264_picture_layout(
	value: video::H264PictureLayout,
) -> vk::VideoDecodeH264PictureLayoutFlagsKHR {
	match value {
		video::H264PictureLayout::Progressive => {
			vk::VideoDecodeH264PictureLayoutFlagBitsKHR::PROGRESSIVE_KHR
		}
		video::H264PictureLayout::InterlacedInterleavedLines => {
			vk::VideoDecodeH264PictureLayoutFlagBitsKHR::INTERLACED_INTERLEAVED_LINES_BIT_KHR
		}
		video::H264PictureLayout::InterlacedSeparatePlanes => {
			vk::VideoDecodeH264PictureLayoutFlagBitsKHR::INTERLACED_SEPARATE_PLANES_BIT_KHR
		}
	}
}

pub(super) const fn h265_profile(value: video::H265Profile) -> vk::StdVideoH265ProfileIdc {
	match value {
		video::H265Profile::Main => vk::StdVideoH265ProfileIdc::MAIN,
		video::H265Profile::Main10 => vk::StdVideoH265ProfileIdc::MAIN_10,
		video::H265Profile::MainStillPicture => vk::StdVideoH265ProfileIdc::MAIN_STILL_PICTURE,
		video::H265Profile::FormatRangeExtensions => {
			vk::StdVideoH265ProfileIdc::FORMAT_RANGE_EXTENSIONS
		}
		video::H265Profile::ScreenContentCodingExtensions => vk::StdVideoH265ProfileIdc::SCC_EXTENSIONS,
	}
}

pub(super) const fn av1_profile(value: video::Av1Profile) -> vk::StdVideoAV1Profile {
	match value {
		video::Av1Profile::Main => vk::StdVideoAV1Profile::MAIN,
		video::Av1Profile::High => vk::StdVideoAV1Profile::HIGH,
		video::Av1Profile::Professional => vk::StdVideoAV1Profile::PROFESSIONAL,
	}
}

pub(super) const fn vp9_operation() -> vk::VideoCodecOperationFlagsKHR {
	vk::VideoCodecOperationFlagsKHR::from_raw(0b1000)
}

pub(super) const fn vp9_profile(value: video::Vp9Profile) -> vk::StdVideoVP9Profile {
	match value {
		video::Vp9Profile::Profile0 => vk::StdVideoVP9Profile::_0,
		video::Vp9Profile::Profile1 => vk::StdVideoVP9Profile::_1,
		video::Vp9Profile::Profile2 => vk::StdVideoVP9Profile::_2,
		video::Vp9Profile::Profile3 => vk::StdVideoVP9Profile::_3,
	}
}

#[allow(dead_code, reason = "wired by the next AV1 frame decode checkpoint")]
pub(super) fn std_av1_color(color: &video::Av1ColorConfig) -> vk::StdVideoAV1ColorConfig {
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
	vk::StdVideoAV1ColorConfig {
		flags: {
			let mut flags = vk::StdVideoAV1ColorConfigFlags::default();
			flags.set_mono_chrome(u32::from(color.monochrome) as _);
			flags.set_color_range(u32::from(color.full_range) as _);
			flags.set_separate_uv_delta_q(u32::from(color.separate_uv_delta_q) as _);
			flags.set_color_description_present_flag(u32::from(color.color_description.is_some()) as _);
			flags.set_reserved(0 as _);
			flags
		},
		bit_depth: match color.bit_depth {
			video::VideoComponentBitDepth::Eight => 8,
			video::VideoComponentBitDepth::Ten => 10,
			video::VideoComponentBitDepth::Twelve => 12,
		},
		subsampling_x,
		subsampling_y,
		reserved1: 0,
		color_primaries: vk::StdVideoAV1ColorPrimaries(description.color_primaries as i32),
		transfer_characteristics: vk::StdVideoAV1TransferCharacteristics(
			description.transfer_characteristics as i32,
		),
		matrix_coefficients: vk::StdVideoAV1MatrixCoefficients(description.matrix_coefficients as i32),
		chroma_sample_position: vk::StdVideoAV1ChromaSamplePosition(chroma_sample_position),
	}
}

#[allow(dead_code, reason = "wired by the next AV1 frame decode checkpoint")]
pub(super) fn std_av1_timing(timing: &video::Av1TimingInfo) -> vk::StdVideoAV1TimingInfo {
	vk::StdVideoAV1TimingInfo {
		flags: {
			let mut flags = vk::StdVideoAV1TimingInfoFlags::default();
			flags.set_equal_picture_interval(u32::from(timing.equal_picture_interval) as _);
			flags.set_reserved(0 as _);
			flags
		},
		num_units_in_display_tick: timing.num_units_in_display_tick,
		time_scale: timing.time_scale,
		num_ticks_per_picture_minus_1: timing.num_ticks_per_picture_minus_1.unwrap_or(0),
	}
}

#[allow(dead_code, reason = "wired by the next AV1 frame decode checkpoint")]
pub(super) fn std_av1_sequence<'a>(
	sequence: &video::Av1SequenceHeader,
	color: &'a vk::StdVideoAV1ColorConfig,
	timing: *const vk::StdVideoAV1TimingInfo,
) -> Result<vk::StdVideoAV1SequenceHeader<'a>> {
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
	Ok({
		vk::StdVideoAV1SequenceHeader {
			flags: {
				let mut flags = vk::StdVideoAV1SequenceHeaderFlags::default();
				flags.set_still_picture(u32::from(sequence.still_picture) as _);
				flags
					.set_reduced_still_picture_header(u32::from(sequence.reduced_still_picture_header) as _);
				flags.set_use_128x128_superblock(u32::from(sequence.use_128x128_superblock) as _);
				flags.set_enable_filter_intra(u32::from(sequence.enable_filter_intra) as _);
				flags.set_enable_intra_edge_filter(u32::from(sequence.enable_intra_edge_filter) as _);
				flags.set_enable_interintra_compound(u32::from(sequence.enable_inter_intra_compound) as _);
				flags.set_enable_masked_compound(u32::from(sequence.enable_masked_compound) as _);
				flags.set_enable_warped_motion(u32::from(sequence.enable_warped_motion) as _);
				flags.set_enable_dual_filter(u32::from(sequence.enable_dual_filter) as _);
				flags.set_enable_order_hint(u32::from(sequence.enable_order_hint) as _);
				flags.set_enable_jnt_comp(u32::from(sequence.enable_joint_compound) as _);
				flags
					.set_enable_ref_frame_mvs(u32::from(sequence.enable_reference_frame_motion_vectors) as _);
				flags.set_frame_id_numbers_present_flag(u32::from(sequence.frame_id_numbers_present) as _);
				flags.set_enable_superres(u32::from(sequence.enable_superres) as _);
				flags.set_enable_cdef(u32::from(sequence.enable_cdef) as _);
				flags.set_enable_restoration(u32::from(sequence.enable_restoration) as _);
				flags.set_film_grain_params_present(u32::from(sequence.film_grain_params_present) as _);
				flags.set_timing_info_present_flag(u32::from(sequence.timing.is_some()) as _);
				flags.set_initial_display_delay_present_flag(u32::from(
					sequence.initial_display_delay_present,
				) as _);
				flags.set_reserved(0 as _);
				flags
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
			p_color_config: color,
			p_timing_info: timing,
			..Default::default()
		}
	})
}
