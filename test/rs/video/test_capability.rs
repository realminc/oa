test_vk!(
	video_capability_query_separates_hardware_from_enabled_sessions,
	engine,
	{
		let capabilities = oa::video::query_device_capabilities(&engine)?;
		assert_eq!(
			capabilities.video_queues_enabled(),
			capabilities.decode_queue_family().is_some() || capabilities.encode_queue_family().is_some()
		);
		assert_eq!(
			capabilities.decoder_sessions_available(),
			capabilities.video_queues_enabled()
				&& capabilities.supports_decode_result_status_queries()
				&& (capabilities.supports_h264_decode()
					|| capabilities.supports_h265_decode()
					|| capabilities.supports_av1_decode()
					|| capabilities.supports_vp9_decode())
		);
		assert_eq!(
			capabilities.encoder_sessions_available(),
			capabilities.encode_queue_family().is_some()
				&& (capabilities.supports_h264_encode() || capabilities.supports_h265_encode())
		);

		let any_decode = capabilities.supports_h264_decode()
			|| capabilities.supports_h265_decode()
			|| capabilities.supports_av1_decode()
			|| capabilities.supports_vp9_decode();
		if any_decode {
			assert!(capabilities.decode_queue_family().is_some());
		}
		let any_encode = capabilities.supports_h264_encode()
			|| capabilities.supports_h265_encode()
			|| capabilities.supports_av1_encode();
		if any_encode {
			assert!(capabilities.encode_queue_family().is_some());
		}
		if capabilities.supports_decode_result_status_queries() {
			assert!(capabilities.decode_queue_family().is_some());
		}
		if capabilities.supports_encode_result_status_queries() {
			assert!(capabilities.encode_queue_family().is_some());
		}
		Ok(())
	}
);

fn assert_profile_limits(capabilities: oa::video::VideoDecodeCapabilities) {
	let minimum = capabilities.min_coded_extent();
	let maximum = capabilities.max_coded_extent();
	assert!(minimum.width > 0 && minimum.height > 0);
	assert!(maximum.width >= minimum.width);
	assert!(maximum.height >= minimum.height);
	assert!(capabilities.picture_access_granularity().width > 0);
	assert!(capabilities.picture_access_granularity().height > 0);
	assert!(
		capabilities
			.min_bitstream_offset_alignment()
			.is_power_of_two()
	);
	assert!(
		capabilities
			.min_bitstream_size_alignment()
			.is_power_of_two()
	);
	assert!(capabilities.max_dpb_slots() > 0);
	assert!(capabilities.max_active_reference_pictures() > 0);
	assert!(capabilities.dpb_and_output_coincide() || capabilities.dpb_and_output_distinct());
}

fn assert_profile_formats(formats: &oa::video::VideoDecodeFormats) {
	assert!(
		!formats.output().is_empty() || formats.unrecognized_output_formats() > 0,
		"the supported decode profile must expose an output format"
	);
	assert!(
		!formats.dpb().is_empty() || formats.unrecognized_dpb_formats() > 0,
		"the supported decode profile must expose a DPB format"
	);
}

fn assert_encode_limits(capabilities: oa::video::VideoEncodeCapabilities) {
	let minimum = capabilities.min_coded_extent();
	let maximum = capabilities.max_coded_extent();
	assert!(minimum.width > 0 && minimum.height > 0);
	assert!(maximum.width >= minimum.width && maximum.height >= minimum.height);
	assert!(capabilities.picture_access_granularity().width > 0);
	assert!(capabilities.picture_access_granularity().height > 0);
	assert!(capabilities.input_picture_granularity().width > 0);
	assert!(capabilities.input_picture_granularity().height > 0);
	assert!(
		capabilities
			.min_bitstream_offset_alignment()
			.is_power_of_two()
	);
	assert!(
		capabilities
			.min_bitstream_size_alignment()
			.is_power_of_two()
	);
	assert!(capabilities.max_dpb_slots() > 0);
	assert!(capabilities.max_active_reference_pictures() > 0);
	assert!(capabilities.max_quality_levels() > 0);
	assert!(
		capabilities.supports_constant_qp()
			|| capabilities.supports_cbr()
			|| capabilities.supports_vbr()
	);
	assert!(capabilities.supports_feedback_bytes_written());
}

fn assert_encode_formats(formats: &oa::video::VideoEncodeFormats) {
	assert!(
		!formats.input().is_empty() || formats.unrecognized_input_formats() > 0,
		"the supported encode profile must expose an input format"
	);
	assert!(
		!formats.dpb().is_empty() || formats.unrecognized_dpb_formats() > 0,
		"the supported encode profile must expose a DPB format"
	);
}

test_vk!(
	exact_encode_profile_queries_return_codec_specific_limits,
	engine,
	{
		let advertised = oa::video::query_device_capabilities(&engine)?;
		if advertised.supports_h264_encode() {
			let profile = oa::video::VideoEncodeProfile::h264_high_420_8bit();
			let capabilities = oa::video::query_encode_capabilities(&engine, profile)?;
			assert_eq!(capabilities.profile(), profile);
			assert!(matches!(
				capabilities.codec(),
				oa::video::VideoEncodeCodecCapabilities::H264 { .. }
			));
			assert_encode_limits(capabilities);
			let formats = oa::video::query_encode_formats(&engine, profile)?;
			assert_eq!(formats.profile(), profile);
			assert_encode_formats(&formats);
		}
		if advertised.supports_h265_encode() {
			let profile = oa::video::VideoEncodeProfile::h265_main_420_8bit();
			let capabilities = oa::video::query_encode_capabilities(&engine, profile)?;
			assert_eq!(capabilities.profile(), profile);
			assert!(matches!(
				capabilities.codec(),
				oa::video::VideoEncodeCodecCapabilities::H265 { .. }
			));
			assert_encode_limits(capabilities);
			let formats = oa::video::query_encode_formats(&engine, profile)?;
			assert_eq!(formats.profile(), profile);
			assert_encode_formats(&formats);
		}
		Ok(())
	}
);

test_vk!(
	exact_decode_profile_queries_return_codec_specific_limits,
	engine,
	{
		let advertised = oa::video::query_device_capabilities(&engine)?;
		if advertised.supports_h264_decode() {
			let profile = oa::video::VideoDecodeProfile::h264_420_8bit(oa::video::H264Profile::High);
			let capabilities = oa::video::query_decode_capabilities(&engine, profile)?;
			assert_eq!(capabilities.profile(), profile);
			assert!(matches!(
				capabilities.level(),
				oa::video::VideoDecodeLevel::H264(_)
			));
			assert!(capabilities.field_offset_granularity().is_some());
			assert_profile_limits(capabilities);
			let formats = oa::video::query_decode_formats(&engine, profile)?;
			assert_eq!(formats.profile(), profile);
			assert_profile_formats(&formats);
		}
		if advertised.supports_h265_decode() {
			let profile = oa::video::VideoDecodeProfile::h265_420(
				oa::video::H265Profile::Main,
				oa::video::VideoComponentBitDepth::Eight,
			);
			let capabilities = oa::video::query_decode_capabilities(&engine, profile)?;
			assert_eq!(capabilities.profile(), profile);
			assert!(matches!(
				capabilities.level(),
				oa::video::VideoDecodeLevel::H265(_)
			));
			assert_eq!(capabilities.field_offset_granularity(), None);
			assert_profile_limits(capabilities);
			let formats = oa::video::query_decode_formats(&engine, profile)?;
			assert_eq!(formats.profile(), profile);
			assert_profile_formats(&formats);
		}
		if advertised.supports_av1_decode() {
			let profile = oa::video::VideoDecodeProfile::av1_420(
				oa::video::Av1Profile::Main,
				oa::video::VideoComponentBitDepth::Eight,
				false,
			);
			let capabilities = oa::video::query_decode_capabilities(&engine, profile)?;
			assert_eq!(capabilities.profile(), profile);
			assert!(matches!(
				capabilities.level(),
				oa::video::VideoDecodeLevel::Av1(_)
			));
			assert_eq!(capabilities.field_offset_granularity(), None);
			assert_profile_limits(capabilities);
			let formats = oa::video::query_decode_formats(&engine, profile)?;
			assert_eq!(formats.profile(), profile);
			assert_profile_formats(&formats);
		}
		if advertised.supports_vp9_decode() {
			let profile = oa::video::VideoDecodeProfile::vp9_420(
				oa::video::Vp9Profile::Profile0,
				oa::video::VideoComponentBitDepth::Eight,
			);
			let capabilities = oa::video::query_decode_capabilities(&engine, profile)?;
			assert_eq!(capabilities.profile(), profile);
			assert!(matches!(
				capabilities.level(),
				oa::video::VideoDecodeLevel::Vp9(_)
			));
			assert_eq!(capabilities.field_offset_granularity(), None);
			assert_profile_limits(capabilities);
			let formats = oa::video::query_decode_formats(&engine, profile)?;
			assert_eq!(formats.profile(), profile);
			assert_profile_formats(&formats);
		}
		Ok(())
	}
);

test_vk!(
	video_decode_profiles_report_exact_support_across_codecs,
	engine,
	{
		let capabilities = oa::video::query_device_capabilities(&engine)?;
		if !capabilities.video_queues_enabled() {
			return Ok(());
		}
		use oa::video::{
			Av1Profile, H264PictureLayout, H264Profile, H265Profile, VideoChromaSubsampling,
			VideoComponentBitDepth, VideoDecodeProfile, Vp9Profile,
		};
		type ProfileProbe = (
			VideoDecodeProfile,
			fn(oa::video::VideoDecodeCapabilities) -> bool,
		);
		let mut profiles: Vec<ProfileProbe> = Vec::new();
		for profile in [
			H264Profile::Baseline,
			H264Profile::Main,
			H264Profile::High,
			H264Profile::High444Predictive,
		] {
			for bit_depth in [VideoComponentBitDepth::Eight, VideoComponentBitDepth::Ten] {
				profiles.push((
					VideoDecodeProfile::H264 {
						profile,
						picture_layout: H264PictureLayout::Progressive,
						chroma_subsampling: VideoChromaSubsampling::Yuv420,
						luma_bit_depth: bit_depth,
						chroma_bit_depth: bit_depth,
					},
					|_| true,
				));
			}
		}
		for profile in [
			H265Profile::Main,
			H265Profile::Main10,
			H265Profile::MainStillPicture,
			H265Profile::FormatRangeExtensions,
			H265Profile::ScreenContentCodingExtensions,
		] {
			for bit_depth in [VideoComponentBitDepth::Eight, VideoComponentBitDepth::Ten] {
				profiles.push((VideoDecodeProfile::h265_420(profile, bit_depth), |_| true));
			}
		}
		for profile in [Av1Profile::Main, Av1Profile::High, Av1Profile::Professional] {
			for bit_depth in [VideoComponentBitDepth::Eight, VideoComponentBitDepth::Ten] {
				for film_grain_support in [false, true] {
					profiles.push((
						VideoDecodeProfile::av1_420(profile, bit_depth, film_grain_support),
						|_| true,
					));
				}
			}
		}
		for profile in [
			Vp9Profile::Profile0,
			Vp9Profile::Profile1,
			Vp9Profile::Profile2,
			Vp9Profile::Profile3,
		] {
			for bit_depth in [
				VideoComponentBitDepth::Eight,
				VideoComponentBitDepth::Ten,
				VideoComponentBitDepth::Twelve,
			] {
				profiles.push((VideoDecodeProfile::vp9_420(profile, bit_depth), |_| true));
			}
		}

		let mut supported_per_codec = [0_usize; 4];
		for (profile, codec) in &profiles {
			let codec_index = match profile {
				VideoDecodeProfile::H264 { .. } => 0,
				VideoDecodeProfile::H265 { .. } => 1,
				VideoDecodeProfile::Av1 { .. } => 2,
				VideoDecodeProfile::Vp9 { .. } => 3,
			};
			let _ = codec;
			match oa::video::query_decode_capabilities(&engine, *profile) {
				Ok(capability) => {
					assert_profile_limits(capability);
					let formats = oa::video::query_decode_formats(&engine, *profile)?;
					assert_profile_formats(&formats);
					supported_per_codec[codec_index] += 1;
				}
				Err(error) => {
					assert!(
						matches!(
							error.kind(),
							oa::ErrorKind::MissingCapability | oa::ErrorKind::InvalidArgument
						),
						"unsupported profile must fail closed, got {error:?} for {profile:?}"
					);
				}
			}
		}
		let flags = [
			capabilities.supports_h264_decode(),
			capabilities.supports_h265_decode(),
			capabilities.supports_av1_decode(),
			capabilities.supports_vp9_decode(),
		];
		for (index, flag) in flags.iter().enumerate() {
			assert_eq!(
				*flag,
				supported_per_codec[index] > 0,
				"aggregate codec flag disagrees with exact profile queries at index {index}: \
				 flag {flag}, {} profiles",
				supported_per_codec[index]
			);
		}
		eprintln!(
			"decode profiles supported: h264={}, h265={}, av1={}, vp9={} of {} queried",
			supported_per_codec[0],
			supported_per_codec[1],
			supported_per_codec[2],
			supported_per_codec[3],
			profiles.len()
		);
		Ok(())
	}
);

test_vk!(
	video_encode_profiles_report_exact_support_and_rate_control,
	engine,
	{
		let capabilities = oa::video::query_device_capabilities(&engine)?;
		if !capabilities.video_queues_enabled() {
			return Ok(());
		}
		use oa::video::{
			H264Profile, H265Profile, VideoChromaSubsampling, VideoComponentBitDepth, VideoEncodeProfile,
		};
		let mut profiles = Vec::new();
		for profile in [H264Profile::Baseline, H264Profile::Main, H264Profile::High] {
			profiles.push((
				VideoEncodeProfile::H264 {
					profile,
					chroma_subsampling: VideoChromaSubsampling::Yuv420,
					luma_bit_depth: VideoComponentBitDepth::Eight,
					chroma_bit_depth: VideoComponentBitDepth::Eight,
				},
				true,
			));
		}
		for (profile, bit_depth) in [
			(H265Profile::Main, VideoComponentBitDepth::Eight),
			(H265Profile::Main10, VideoComponentBitDepth::Ten),
		] {
			profiles.push((
				VideoEncodeProfile::H265 {
					profile,
					chroma_subsampling: VideoChromaSubsampling::Yuv420,
					luma_bit_depth: bit_depth,
					chroma_bit_depth: bit_depth,
				},
				false,
			));
		}

		let mut supported_per_codec = [0_usize; 2];
		let mut rate_modes = (false, false, false);
		for (profile, is_h264) in &profiles {
			match oa::video::query_encode_capabilities(&engine, *profile) {
				Ok(capability) => {
					assert_encode_limits(capability);
					let formats = oa::video::query_encode_formats(&engine, *profile)?;
					assert_encode_formats(&formats);
					supported_per_codec[usize::from(!is_h264)] += 1;
					rate_modes.0 |= capability.supports_constant_qp();
					rate_modes.1 |= capability.supports_cbr();
					rate_modes.2 |= capability.supports_vbr();
				}
				Err(error) => {
					assert!(
						matches!(
							error.kind(),
							oa::ErrorKind::MissingCapability | oa::ErrorKind::InvalidArgument
						),
						"unsupported encode profile must fail closed, got {error:?} for {profile:?}"
					);
				}
			}
		}
		assert_eq!(
			capabilities.supports_h264_encode(),
			supported_per_codec[0] > 0,
			"aggregate H.264 encode flag disagrees with exact profile queries"
		);
		assert_eq!(
			capabilities.supports_h265_encode(),
			supported_per_codec[1] > 0,
			"aggregate H.265 encode flag disagrees with exact profile queries"
		);
		// Preflight must fail closed for every rate-control mode the device
		// declined to advertise, with a typed MissingCapability.
		for mode in [
			oa::video::VideoRateControl::ConstantQp,
			oa::video::VideoRateControl::Cbr,
			oa::video::VideoRateControl::Vbr,
		] {
			let config = oa::video::VideoEncodeConfig {
				rate_control: mode,
				..oa::video::VideoEncodeConfig::new(VideoEncodeProfile::h264_high_420_8bit(), 320, 240)
			};
			let capability =
				oa::video::query_encode_capabilities(&engine, VideoEncodeProfile::h264_high_420_8bit());
			if let Ok(capability) = capability {
				let supported = match mode {
					oa::video::VideoRateControl::ConstantQp => capability.supports_constant_qp(),
					oa::video::VideoRateControl::Cbr => capability.supports_cbr(),
					oa::video::VideoRateControl::Vbr => capability.supports_vbr(),
				};
				match config.validate_for_device(capability) {
					Ok(_) => assert!(supported, "{mode:?} accepted without device support"),
					Err(error) => {
						assert!(
							!supported,
							"{mode:?} rejected although the device supports it: {error:?}"
						);
						assert_eq!(error.kind(), oa::ErrorKind::MissingCapability);
					}
				}
			}
		}
		eprintln!(
			"encode profiles supported: h264={}, h265={}; rate control: constant_qp={}, cbr={}, vbr={}",
			supported_per_codec[0], supported_per_codec[1], rate_modes.0, rate_modes.1, rate_modes.2
		);
		Ok(())
	}
);
