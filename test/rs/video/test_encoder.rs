//! External contracts for the public hardware video encoder and recorder.

use std::{
	path::PathBuf,
	time::{SystemTime, UNIX_EPOCH},
};

use oa::{
	ErrorKind,
	video::{
		VideoCodec, VideoEncodeConfig, VideoEncodeProfile, VideoEncoder, VideoExtent, VideoRateControl,
		VideoRecorder, VideoRecorderConfig,
	},
};

#[test]
fn encode_settings_defaults_match_the_donor_profile() {
	let settings = VideoEncodeConfig::new(VideoEncodeProfile::h264_high_420_8bit(), 1920, 1080);
	assert_eq!(settings.frame_rate, 30);
	assert_eq!(settings.gop_size, 30);
	assert_eq!(settings.rate_control, VideoRateControl::ConstantQp);
	assert_eq!(settings.bitrate, 4_000_000);
	assert_eq!(settings.max_bitrate, 0);
	assert_eq!(settings.constant_qp, 26);
	assert_eq!(settings.quality_level, 0);
	assert_eq!(settings.async_depth, 3);
	assert_eq!(settings.max_b_frames, 0);
	assert_eq!(settings.max_dpb_slots, 0);
}

#[test]
fn recorder_config_defaults_keep_the_donor_shape() {
	let config = VideoRecorderConfig::default();
	assert_eq!(config.output_path, "output.mp4");
	assert!(config.audio.is_none());
	assert!(!config.full_range);
}

test_vk!(
	video_encoder_session_is_capability_gated_and_encodes,
	engine,
	{
		let caps = oa::video::query_device_capabilities(&engine)?;
		let settings = VideoEncodeConfig::new(VideoEncodeProfile::h264_high_420_8bit(), 320, 240);
		if !caps.encoder_sessions_available() || !caps.supports_h264_encode() {
			let err = match VideoEncoder::new(&engine, settings) {
				Ok(_) => panic!("session creation must fail without H.264 encode"),
				Err(err) => err,
			};
			assert!(
				matches!(
					err.kind(),
					ErrorKind::MissingCapability | ErrorKind::InvalidArgument
				),
				"unsupported device must report capability admission, got {err:?}"
			);
			return Ok(());
		}

		let mut encoder = VideoEncoder::new(&engine, settings)?;
		assert!(encoder.is_open());
		assert_eq!(
			encoder.visible_extent(),
			Some(VideoExtent {
				width: 320,
				height: 240
			})
		);
		let coded = encoder
			.coded_extent()
			.expect("open session has a coded extent");
		assert!(coded.width >= 320 && coded.height >= 240);
		assert!(
			encoder.codec_headers().is_some_and(|h| !h.is_empty()),
			"an open H.264 session caches SPS/PPS headers"
		);

		let rgba = vec![0x7f_u8; 320 * 240 * 4];
		let packet = encoder.encode(&rgba, 0)?;
		assert!(packet.is_keyframe(), "first frame is the GOP keyframe");
		assert!(
			packet.bitstream().windows(4).any(|w| w == [0, 0, 0, 1]),
			"keyframe access unit must be Annex-B framed"
		);
		assert!(
			encoder.flush()?.is_empty(),
			"synchronous encode leaves nothing pending"
		);

		let second = encoder.encode(&rgba, 33_333)?;
		assert!(!second.is_keyframe());
		assert_eq!(second.presentation_timestamp_micros(), 33_333);

		let err = match encoder.encode(&rgba[..rgba.len() - 4], 66_666) {
			Ok(_) => panic!("undersized RGBA input must fail"),
			Err(err) => err,
		};
		assert_eq!(err.kind(), ErrorKind::InvalidArgument);

		encoder.close();
		assert!(!encoder.is_open());
		let closed = match encoder.encode(&rgba, 100_000) {
			Ok(_) => panic!("a closed encoder rejects frames"),
			Err(err) => err,
		};
		assert_eq!(closed.kind(), ErrorKind::FailedPrecondition);
		Ok(())
	}
);

test_vk!(
	video_recorder_validates_and_writes_a_demuxable_mp4,
	engine,
	{
		let path = temporary_mp4("recorder");
		let settings = VideoEncodeConfig::new(VideoEncodeProfile::h264_high_420_8bit(), 320, 240);
		let unsupported = match VideoRecorder::new(
			&engine,
			VideoRecorderConfig {
				output_path: String::new(),
				encode: settings,
				..Default::default()
			},
		) {
			Ok(_) => panic!("an empty output path must fail"),
			Err(err) => err,
		};
		assert_eq!(unsupported.kind(), ErrorKind::InvalidArgument);

		let zero_extent = match VideoRecorder::new(
			&engine,
			VideoRecorderConfig {
				output_path: path.to_string_lossy().into_owned(),
				encode: VideoEncodeConfig::new(VideoEncodeProfile::h264_high_420_8bit(), 0, 0),
				..Default::default()
			},
		) {
			Ok(_) => panic!("a zero encode extent must fail"),
			Err(err) => err,
		};
		assert_eq!(zero_extent.kind(), ErrorKind::InvalidArgument);

		let caps = oa::video::query_device_capabilities(&engine)?;
		let mut recorder = match VideoRecorder::new(
			&engine,
			VideoRecorderConfig {
				output_path: path.to_string_lossy().into_owned(),
				encode: settings,
				..Default::default()
			},
		) {
			Ok(recorder) => recorder,
			Err(err) => {
				assert!(
					!caps.encoder_sessions_available() || !caps.supports_h264_encode(),
					"capable device must create the recorder, got {err:?}"
				);
				return Ok(());
			}
		};
		assert!(recorder.is_open());
		assert_eq!(recorder.frame_count(), 0);

		let rgba = vec![0x33_u8; 320 * 240 * 4];
		recorder.write_rgba(&rgba, 0)?;
		recorder.write_rgba(&rgba, 33_333)?;
		recorder.write_rgba(&rgba, 66_666)?;
		assert_eq!(recorder.frame_count(), 3);
		recorder.finalize()?;
		assert!(!recorder.is_open());
		recorder.finalize()?;
		recorder.close();

		let demuxer = oa::video::VideoDemuxer::open(&path)?;
		let info = demuxer.info();
		assert_eq!(info.codec(), VideoCodec::H264);
		assert_eq!(info.sample_count(), 3);
		let _ = std::fs::remove_file(&path);
		Ok(())
	}
);

fn temporary_mp4(label: &str) -> PathBuf {
	let nonce = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.expect("system clock precedes Unix epoch")
		.as_nanos();
	std::env::temp_dir().join(format!("oars_{label}_{}_{nonce}.mp4", std::process::id()))
}
