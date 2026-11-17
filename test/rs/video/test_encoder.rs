//! External contracts for the public hardware video encoder and recorder.

use std::{
	path::PathBuf,
	time::{SystemTime, UNIX_EPOCH},
};

use oa::{
	ErrorKind,
	video::{
		H265Profile, VideoCodec, VideoComponentBitDepth, VideoEncodeConfig, VideoEncodeProfile,
		VideoEncoder, VideoExtent, VideoRateControl, VideoRecorder, VideoRecorderConfig,
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

test_vk!(
	hevc_repeated_720p_sessions_preserve_completion_and_packets,
	engine,
	"requires HEVC Main/Main10/Rext encode at 720p; repeated-session GPU stress",
	{
		const WIDTH: usize = 1280;
		const HEIGHT: usize = 720;
		let mut rgba8 = vec![0_u8; WIDTH * HEIGHT * 4];
		let mut rgba16 = vec![0_u16; WIDTH * HEIGHT * 4];
		for profile in [
			VideoEncodeProfile::h265_main_420_8bit(),
			VideoEncodeProfile::H265 {
				profile: H265Profile::Main10,
				chroma_subsampling: oa::video::VideoChromaSubsampling::Yuv420,
				luma_bit_depth: VideoComponentBitDepth::Ten,
				chroma_bit_depth: VideoComponentBitDepth::Ten,
			},
			VideoEncodeProfile::H265 {
				profile: H265Profile::FormatRangeExtensions,
				chroma_subsampling: oa::video::VideoChromaSubsampling::Yuv420,
				luma_bit_depth: VideoComponentBitDepth::Ten,
				chroma_bit_depth: VideoComponentBitDepth::Ten,
			},
		] {
			oa::video::query_encode_capabilities(&engine, profile)?;
			let wide = profile != VideoEncodeProfile::h265_main_420_8bit();
			for session in 0..32 {
				let config = VideoEncodeConfig::new(profile, WIDTH as u32, HEIGHT as u32);
				let mut encoder = VideoEncoder::new(&engine, config)?;
				for frame in 0..60 {
					for y in 0..HEIGHT {
						for x in 0..WIDTH {
							let index = (y * WIDTH + x) * 4;
							let values = [x + frame * 3, y + frame * 5, x + y + frame * 7];
							for (channel, value) in values.into_iter().enumerate() {
								rgba8[index + channel] = (value & 0xff) as u8;
								rgba16[index + channel] = ((value & 0x3ff) << 6) as u16;
							}
							rgba8[index + 3] = 255;
							rgba16[index + 3] = u16::MAX;
						}
					}
					let pts = frame as u64 * 33_333;
					let result = if wide {
						encoder.encode_rgba16(&rgba16, pts)
					} else {
						encoder.encode(&rgba8, pts)
					};
					let packet = result.unwrap_or_else(|error| {
						panic!("{profile:?} session {session} frame {frame}: {error:?}")
					});
					assert_eq!(packet.presentation_timestamp_micros(), pts);
					assert_eq!(packet.is_keyframe(), frame % 30 == 0);
					assert!(!packet.bitstream().is_empty());
				}
				assert!(encoder.flush()?.is_empty());
				encoder.close();
				eprintln!("HEVC stress {profile:?} session {session}: 60 packets complete");
			}
		}
		Ok(())
	}
);

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

test_vk!(
	h264_and_h265_recordings_match_independent_ffmpeg_pixels,
	engine,
	"requires both H.264/H.265 hardware encode profiles and FFmpeg",
	{
		let caps = oa::video::query_device_capabilities(&engine)?;
		assert!(caps.supports_h264_encode() && caps.supports_h265_encode());
		for (profile, codec) in [
			(VideoEncodeProfile::h264_high_420_8bit(), VideoCodec::H264),
			(VideoEncodeProfile::h265_main_420_8bit(), VideoCodec::H265),
		] {
			let path = temporary_mp4("encode_oracle");
			let mut encode = VideoEncodeConfig::new(profile, 320, 240);
			encode.gop_size = 2;
			// A low QP keeps the pixel oracle sensitive to ordering/conversion
			// defects rather than the default encoder's lossy quantization.
			encode.constant_qp = 10;
			let mut recorder = VideoRecorder::new(
				&engine,
				VideoRecorderConfig {
					output_path: path.to_string_lossy().into_owned(),
					encode,
					..Default::default()
				},
			)?;
			let levels = [32_u8, 64, 96, 128, 160, 192, 224];
			for (index, level) in levels.iter().enumerate() {
				let mut rgba = vec![*level; 320 * 240 * 4];
				for pixel in rgba.as_chunks_mut::<4>().0 {
					pixel[3] = 255;
				}
				recorder.write_rgba(&rgba, index as u64 * 33_333)?;
			}
			recorder.finalize()?;
			let demuxer = oa::video::VideoDemuxer::open(&path)?;
			assert_eq!(demuxer.info().codec(), codec);
			assert_eq!(demuxer.info().sample_count() as usize, levels.len());
			let decoded = std::process::Command::new("ffmpeg")
				.args(["-v", "error", "-i"])
				.arg(&path)
				.args(["-map", "0:v:0", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
				.output()
				.expect("FFmpeg is required for the independent encode oracle");
			assert!(
				decoded.status.success(),
				"{codec:?}: {}",
				String::from_utf8_lossy(&decoded.stderr)
			);
			let frame_size = 320 * 240 * 3;
			assert_eq!(decoded.stdout.len(), frame_size * levels.len(), "{codec:?}");
			for (index, (frame, level)) in decoded
				.stdout
				.chunks_exact(frame_size)
				.zip(levels)
				.enumerate()
			{
				let maximum_error = frame
					.iter()
					.map(|&value| value.abs_diff(level))
					.max()
					.unwrap();
				assert!(
					maximum_error <= 6,
					"{codec:?} frame {index}: maximum grayscale error {maximum_error}"
				);
			}
			std::fs::remove_file(path).expect("remove completed encode oracle fixture");
		}
		Ok(())
	}
);

test_vk!(
	h265_ten_bit_encode_matches_independent_ffmpeg_pixels,
	engine,
	"requires H.265 wide-component hardware encode and FFmpeg",
	{
		for (codec_profile, expected) in [
			(H265Profile::Main10, "Main 10"),
			(H265Profile::FormatRangeExtensions, "Rext"),
		] {
			let profile = VideoEncodeProfile::H265 {
				profile: codec_profile,
				chroma_subsampling: oa::video::VideoChromaSubsampling::Yuv420,
				luma_bit_depth: VideoComponentBitDepth::Ten,
				chroma_bit_depth: VideoComponentBitDepth::Ten,
			};
			if oa::video::query_encode_capabilities(&engine, profile).is_err() {
				continue;
			}
			assert_ten_bit_encode_oracle(&engine, profile, expected)?;
		}
		Ok(())
	}
);

fn assert_ten_bit_encode_oracle(
	engine: &oa::Engine,
	profile: VideoEncodeProfile,
	expected_profile: &str,
) -> oa::Result<()> {
	{
		let mut encode = VideoEncodeConfig::new(profile, 320, 240);
		encode.gop_size = 2;
		encode.constant_qp = 10;
		let mut encoder = VideoEncoder::new(engine, encode)?;
		// Mid-range 10-bit grays stay inside the limited-range encode window.
		let levels = [128_u16, 256, 384, 512, 640, 768, 896];
		let mut bitstream = Vec::new();
		if let Some(headers) = encoder.codec_headers() {
			bitstream.extend_from_slice(headers);
		}
		for (index, level) in levels.iter().enumerate() {
			let gray = level << 6;
			let mut rgba = vec![0_u16; 320 * 240 * 4];
			for pixel in rgba.as_chunks_mut::<4>().0 {
				*pixel = [gray, gray, gray, 0xffff];
			}
			let packet = encoder.encode_rgba16(&rgba, index as u64 * 33_333)?;
			if index == 0 {
				assert!(packet.is_keyframe(), "first frame is the GOP keyframe");
			}
			bitstream.extend_from_slice(packet.bitstream());
		}
		encoder.close();
		let path = temporary_stream("encode_main10", "h265");
		std::fs::write(&path, &bitstream).expect("write elementary stream");
		let probed = std::process::Command::new("ffprobe")
			.args([
				"-v",
				"error",
				"-show_entries",
				"stream=profile,pix_fmt",
				"-of",
				"csv=p=0",
			])
			.arg(&path)
			.output()
			.expect("FFprobe is required for the independent encode oracle");
		let probe_stdout = String::from_utf8_lossy(&probed.stdout).into_owned();
		assert!(
			probe_stdout.contains(expected_profile),
			"encoded stream must report {expected_profile}, got: {probe_stdout}"
		);
		let decoded = std::process::Command::new("ffmpeg")
			.args(["-v", "error", "-i"])
			.arg(&path)
			.args([
				"-map", "0:v:0", "-pix_fmt", "rgb48le", "-f", "rawvideo", "-",
			])
			.output()
			.expect("FFmpeg is required for the independent encode oracle");
		std::fs::remove_file(&path).expect("remove elementary stream");
		assert!(
			decoded.status.success(),
			"{}",
			String::from_utf8_lossy(&decoded.stderr)
		);
		let frame_size = 320 * 240 * 3 * 2;
		assert_eq!(decoded.stdout.len(), frame_size * levels.len());
		for (index, (frame, level)) in decoded
			.stdout
			.chunks_exact(frame_size)
			.zip(levels)
			.enumerate()
		{
			let expected = level << 6;
			let maximum_error = frame
				.as_chunks::<2>()
				.0
				.iter()
				.map(|channel| u16::from_le_bytes(*channel).abs_diff(expected))
				.max()
				.unwrap();
			assert!(
				maximum_error <= 2048,
				"frame {index}: maximum 16-bit grayscale error {maximum_error}"
			);
		}
		Ok(())
	}
}

fn temporary_mp4(label: &str) -> PathBuf {
	temporary_stream(label, "mp4")
}

fn temporary_stream(label: &str, extension: &str) -> PathBuf {
	let nonce = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.expect("system clock precedes Unix epoch")
		.as_nanos();
	std::env::temp_dir().join(format!(
		"oars_{label}_{}_{nonce}.{extension}",
		std::process::id()
	))
}
