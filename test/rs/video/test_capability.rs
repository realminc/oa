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
		assert!(!capabilities.encoder_sessions_available());

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
