use std::any::TypeId;
use std::{path::PathBuf, process::Command};

use oa::{
	ErrorKind,
	video::{
		Av1Profile, H264Profile, H265Profile, VideoCodec, VideoComponentBitDepth, VideoContainerKind,
		VideoDecodeProfile, VideoDemuxer, Vp9Profile, length_prefixed_to_annex_b, parse_nal_annex_b,
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
			VideoCodec::Vp9 => Some(VideoDecodeProfile::vp9_420(
				Vp9Profile::Profile0,
				VideoComponentBitDepth::Eight,
			)),
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
#[ignore = "requires OA donor fixtures and FFmpeg remux"]
fn matroska_and_webm_index_donor_packets_without_pixel_decode() -> oa::Result<()> {
	for (source, extension, codec, kind) in [
		(
			"shibuya_720p_30fps_h264_high_8bit_420.mp4",
			"mkv",
			VideoCodec::H264,
			VideoContainerKind::Matroska,
		),
		(
			"shibuya_720p_30fps_h265_main_8bit_420.mp4",
			"mkv",
			VideoCodec::H265,
			VideoContainerKind::Matroska,
		),
		(
			"shibuya_720p_30fps_av1_main_8bit_420.mp4",
			"mkv",
			VideoCodec::Av1,
			VideoContainerKind::Matroska,
		),
		(
			"shibuya_720p_30fps_vp9_profile0_8bit_420.mp4",
			"webm",
			VideoCodec::Vp9,
			VideoContainerKind::WebM,
		),
	] {
		let output = std::env::temp_dir().join(format!(
			"oa_matroska_test_{}_{}.{}",
			std::process::id(),
			source,
			extension
		));
		let status = Command::new("ffmpeg")
			.args(["-v", "error", "-y", "-i"])
			.arg(fixture(source))
			.args(["-map", "0:v:0", "-c", "copy", "-an", "-t", "1"])
			.arg(&output)
			.status()
			.expect("FFmpeg must be installed for the ignored remux test");
		assert!(status.success(), "FFmpeg fixture remux failed for {source}");
		let result = (|| -> oa::Result<()> {
			let mut demuxer = VideoDemuxer::open(&output)?;
			let info = demuxer.info();
			assert_eq!(info.kind(), kind);
			assert_eq!(info.codec(), codec);
			let expected_profile = match codec {
				VideoCodec::H264 => VideoDecodeProfile::h264_420_8bit(H264Profile::High),
				VideoCodec::H265 => {
					VideoDecodeProfile::h265_420(H265Profile::Main, VideoComponentBitDepth::Eight)
				}
				VideoCodec::Av1 => {
					VideoDecodeProfile::av1_420(Av1Profile::Main, VideoComponentBitDepth::Eight, false)
				}
				VideoCodec::Vp9 => {
					VideoDecodeProfile::vp9_420(Vp9Profile::Profile0, VideoComponentBitDepth::Eight)
				}
			};
			assert_eq!(info.decode_profile(), Some(expected_profile));
			assert_eq!((info.width(), info.height()), (1280, 720));
			assert!(info.sample_count() > 0);
			assert!(info.duration() > 0);
			let first = demuxer
				.read_next_packet()?
				.expect("remuxed stream must have a packet");
			assert!(first.is_keyframe());
			assert!(!first.data().is_empty());
			if matches!(codec, VideoCodec::H264 | VideoCodec::H265) {
				assert!(!parse_nal_annex_b(first.data()).is_empty());
			}
			demuxer.seek(info.duration() / 2)?;
			assert!(demuxer.read_next_packet()?.is_some());
			Ok(())
		})();
		let _ = std::fs::remove_file(&output);
		result?;
	}
	Ok(())
}

#[test]
#[ignore = "requires OA donor fixtures and FFmpeg remux"]
fn fragmented_mp4_indexes_donor_packets_for_four_codecs() -> oa::Result<()> {
	for (source, codec) in [
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
		let output = std::env::temp_dir().join(format!(
			"oa_fragmented_mp4_test_{}_{}.mp4",
			std::process::id(),
			source
		));
		let status = Command::new("ffmpeg")
			.args(["-v", "error", "-y", "-i"])
			.arg(fixture(source))
			.args([
				"-map",
				"0:v:0",
				"-c",
				"copy",
				"-an",
				"-movflags",
				"frag_keyframe+empty_moov+default_base_moof",
			])
			.arg(&output)
			.status()
			.expect("FFmpeg must be installed for the ignored remux test");
		assert!(status.success(), "FFmpeg remux failed for {source}");
		let result = (|| -> oa::Result<()> {
			let mut demuxer = VideoDemuxer::open(&output)?;
			let info = demuxer.info();
			assert_eq!(info.kind(), VideoContainerKind::Mp4);
			assert_eq!(info.codec(), codec);
			assert!(info.sample_count() > 0);
			let first = demuxer
				.read_next_packet()?
				.expect("fragment must contain a packet");
			assert!(first.is_keyframe());
			if matches!(codec, VideoCodec::H264 | VideoCodec::H265) {
				assert!(!parse_nal_annex_b(first.data()).is_empty());
			} else {
				assert!(!first.data().is_empty());
			}
			demuxer.close();
			if codec == VideoCodec::H264 {
				let mut bytes = std::fs::read(&output).expect("remuxed fixture must be readable");
				let trun = bytes
					.windows(4)
					.position(|window| window == b"trun")
					.expect("fragment must contain a trun box");
				let flags = u32::from_be_bytes(bytes[trun + 4..trun + 8].try_into().unwrap());
				assert_ne!(flags & 1, 0, "fixture must carry an explicit data offset");
				bytes[trun + 12..trun + 16].copy_from_slice(&i32::MAX.to_be_bytes());
				std::fs::write(&output, bytes).expect("malformed fixture write must succeed");
				let error = match VideoDemuxer::open(&output) {
					Ok(_) => panic!("out-of-mdat packet accepted"),
					Err(error) => error,
				};
				assert_eq!(error.kind(), ErrorKind::DataLoss);
			}
			Ok(())
		})();
		let _ = std::fs::remove_file(&output);
		result?;
	}
	Ok(())
}

#[test]
#[ignore = "requires OA donor H.264/H.265 fixtures and FFmpeg remux"]
fn mpeg_ts_streams_donor_h264_and_h265_pes() -> oa::Result<()> {
	for (source, codec) in [
		(
			"shibuya_720p_30fps_h264_high_8bit_420.mp4",
			VideoCodec::H264,
		),
		(
			"shibuya_720p_30fps_h265_main_8bit_420.mp4",
			VideoCodec::H265,
		),
	] {
		let output =
			std::env::temp_dir().join(format!("oa_ts_test_{}_{}.ts", std::process::id(), source));
		let status = Command::new("ffmpeg")
			.args(["-v", "error", "-y", "-i"])
			.arg(fixture(source))
			.args([
				"-map", "0:v:0", "-c", "copy", "-an", "-t", "1", "-f", "mpegts",
			])
			.arg(&output)
			.status()
			.expect("FFmpeg must be installed for the ignored remux test");
		assert!(status.success());
		let result = (|| -> oa::Result<()> {
			let mut demuxer = VideoDemuxer::open(&output)?;
			let info = demuxer.info();
			assert_eq!(info.kind(), VideoContainerKind::MpegTs);
			assert_eq!(info.codec(), codec);
			assert_eq!((info.width(), info.height()), (1280, 720));
			let first = demuxer
				.read_next_packet()?
				.expect("TS must contain first PES");
			assert!(first.is_keyframe());
			assert!(!parse_nal_annex_b(first.data()).is_empty());
			let second = demuxer
				.read_next_packet()?
				.expect("TS must contain second PES");
			assert!(second.presentation_timestamp() > 0);
			assert_eq!(
				demuxer
					.seek(90_000)
					.expect_err("nonzero TS seek was accepted")
					.kind(),
				ErrorKind::MissingCapability
			);
			demuxer.seek(0)?;
			assert_eq!(
				demuxer
					.read_next_packet()?
					.expect("TS rewind failed")
					.data(),
				first.data()
			);
			Ok(())
		})();
		let _ = std::fs::remove_file(&output);
		result?;
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
		"shibuya_720p_30fps_vp9_profile0_8bit_420.mp4",
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
	let sibling = PathBuf::from(env!("CARGO_MANIFEST_DIR")).with_file_name("oacpp");
	let donor = if sibling.exists() {
		sibling
	} else {
		PathBuf::from(env!("CARGO_MANIFEST_DIR")).with_file_name("oa")
	};
	donor.join("sdk/asset/video/clip").join(name)
}
