use std::any::TypeId;
use std::path::PathBuf;

use oa::{
	ErrorKind,
	video::{
		Av1Profile, H264Profile, H265Profile, VideoCodec, VideoComponentBitDepth,
		VideoDecodeProfile, VideoDemuxer, length_prefixed_to_annex_b, parse_nal_annex_b,
	},
};

#[test]
fn root_demuxer_export_is_the_video_session_identity() {
	assert_eq!(
		TypeId::of::<oa::VideoDemuxer>(),
		TypeId::of::<oa::video::VideoDemuxer>()
	);
}

#[test]
fn length_prefixed_nals_are_bounded_and_canonicalized() -> oa::Result<()> {
	assert_eq!(
		length_prefixed_to_annex_b(&[0, 0, 0, 2, 0x65, 0xaa, 0, 0, 0, 1, 0x06], 4)?,
		[0, 0, 0, 1, 0x65, 0xaa, 0, 0, 0, 1, 0x06]
	);
	assert_eq!(
		length_prefixed_to_annex_b(&[2, 0x65, 0xaa, 1, 0x06], 1)?,
		[0, 0, 0, 1, 0x65, 0xaa, 0, 0, 0, 1, 0x06]
	);
	for (sample, width) in [(&[][..], 4), (&[0, 0, 0][..], 4), (&[0, 2, 0x65][..], 2)] {
		assert_eq!(
			length_prefixed_to_annex_b(sample, width)
				.expect_err("malformed NAL sample was accepted")
				.kind(),
			ErrorKind::DataLoss
		);
	}
	assert_eq!(
		length_prefixed_to_annex_b(&[1, 0], 3)
			.expect_err("invalid NAL width was accepted")
			.kind(),
		ErrorKind::InvalidArgument
	);
	Ok(())
}

#[test]
#[ignore = "requires OA donor video fixtures"]
fn mp4_demux_reads_and_seeks_all_donor_codecs() -> oa::Result<()> {
	for (name, codec) in [
		(
			"shibuya_720p_30fps_h264_high_8bit_420.mp4",
			VideoCodec::H264,
		),
		(
			"shibuya_720p_30fps_h265_main_8bit_420.mp4",
			VideoCodec::H265,
		),
		("shibuya_720p_30fps_av1_main_8bit_420.mp4", VideoCodec::Av1),
		(
			"shibuya_720p_30fps_vp9_profile0_8bit_420.mp4",
			VideoCodec::Vp9,
		),
	] {
		let mut demuxer = VideoDemuxer::open(fixture(name))?;
		let info = demuxer.info();
		assert_eq!(info.codec(), codec);
		let expected_profile = match codec {
			VideoCodec::H264 => Some(VideoDecodeProfile::h264_420_8bit(H264Profile::High)),
			VideoCodec::H265 => Some(VideoDecodeProfile::h265_420(
				H265Profile::Main,
				VideoComponentBitDepth::Eight,
			)),
			VideoCodec::Av1 => Some(VideoDecodeProfile::av1_420(
				Av1Profile::Main,
				VideoComponentBitDepth::Eight,
				false,
			)),
			VideoCodec::Vp9 => None,
		};
		assert_eq!(info.decode_profile(), expected_profile);
		assert_eq!((info.width(), info.height()), (1280, 720));
		assert!(info.sample_count() > 1);
		assert!(info.duration() > 0);
		assert!(info.frame_rate() > 1.0);
		let first = demuxer
			.read_next_packet()?
			.expect("fixture must contain a first packet");
		assert!(!first.data().is_empty());
		assert!(first.is_keyframe());
		if matches!(codec, VideoCodec::H264 | VideoCodec::H265) {
			assert!(!parse_nal_annex_b(first.data()).is_empty());
		}

		demuxer.seek(info.duration() / 2)?;
		let sought = demuxer
			.read_next_packet()?
			.expect("seek target must have a packet");
		assert!(sought.is_keyframe());
		assert!(sought.presentation_timestamp() <= info.duration() / 2);
		demuxer.close();
		assert_eq!(
			demuxer
				.read_next_packet()
				.expect_err("closed demuxer was readable")
				.kind(),
			ErrorKind::FailedPrecondition
		);
	}
	Ok(())
}

#[test]
#[ignore = "requires a hardware Vulkan Video device and OA donor video fixtures"]
fn demuxed_profiles_drive_exact_device_queries() -> oa::Result<()> {
	let engine = oa::Engine::new()?;
	for name in [
		"shibuya_720p_30fps_h264_high_8bit_420.mp4",
		"shibuya_720p_30fps_h265_main_8bit_420.mp4",
		"shibuya_720p_30fps_av1_main_8bit_420.mp4",
	] {
		let demuxer = VideoDemuxer::open(fixture(name))?;
		let profile = demuxer
			.info()
			.decode_profile()
			.expect("the admitted donor stream must expose a typed decode profile");
		let capabilities = oa::video::query_decode_capabilities(&engine, profile)?;
		assert_eq!(capabilities.profile(), profile);
		let formats = oa::video::query_decode_formats(&engine, profile)?;
		assert_eq!(formats.profile(), profile);
		assert!(!formats.output().is_empty() || formats.unrecognized_output_formats() > 0);
		assert!(!formats.dpb().is_empty() || formats.unrecognized_dpb_formats() > 0);
	}
	Ok(())
}

fn fixture(name: &str) -> PathBuf {
	PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip")
		.join(name)
}
