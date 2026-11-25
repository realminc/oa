//! External contracts for the public hardware video encoder and recorder.

use std::{
	path::PathBuf,
	time::{SystemTime, UNIX_EPOCH},
};

use oa::{
	Error, ErrorKind, Matrix, VideoDecoder, VideoDemuxer,
	video::{
		H265Profile, VideoCodec, VideoComponentBitDepth, VideoEncodeConfig, VideoEncodeProfile,
		VideoEncoder, VideoExtent, VideoRateControl, VideoRecorder, VideoRecorderConfig,
	},
};
use std::io::Write;

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
	runtime,
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
			oa::video::query_encode_capabilities(&runtime, profile)?;
			let wide = profile != VideoEncodeProfile::h265_main_420_8bit();
			for session in 0..32 {
				let config = VideoEncodeConfig::new(profile, WIDTH as u32, HEIGHT as u32);
				let mut encoder = VideoEncoder::new(&runtime, config)?;
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
	runtime,
	{
		let caps = oa::video::query_device_capabilities(&runtime)?;
		let settings = VideoEncodeConfig::new(VideoEncodeProfile::h264_high_420_8bit(), 320, 240);
		if !caps.encoder_sessions_available() || !caps.supports_h264_encode() {
			let err = match VideoEncoder::new(&runtime, settings) {
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

		let mut encoder = VideoEncoder::new(&runtime, settings)?;
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
		let capture_error = runtime
			.capture(|| encoder.encode(&rgba, 0))
			.expect_err("immediate encode must fail before changing the session in capture");
		assert_eq!(capture_error.kind(), ErrorKind::FailedPrecondition);
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
	video_encoder_capture_replays_conversion_and_encode,
	runtime,
	{
		let caps = oa::video::query_device_capabilities(&runtime)?;
		let settings = VideoEncodeConfig::new(VideoEncodeProfile::h264_high_420_8bit(), 320, 240);
		if !caps.encoder_sessions_available() || !caps.supports_h264_encode() {
			let err = match VideoEncoder::new(&runtime, settings) {
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

		let mut encoder = VideoEncoder::new(&runtime, settings)?;
		let mut rgba = vec![0_u8; 320 * 240 * 4];
		for y in 0..240_u32 {
			for x in 0..320_u32 {
				let index = ((y * 320 + x) * 4) as usize;
				rgba[index] = (x & 0xff) as u8;
				rgba[index + 1] = (y & 0xff) as u8;
				rgba[index + 2] = ((x + y) & 0xff) as u8;
				rgba[index + 3] = 255;
			}
		}
		let frames: Vec<(&[u8], u64)> = (0..4_u64)
			.map(|frame| (rgba.as_slice(), frame * 33_333))
			.collect();
		let (plan, ()) = runtime.capture(|| encoder.record_frames(&frames))?;
		let report: serde_json::Value =
			serde_json::from_str(&plan.debug_report_json("h264-capture")).unwrap();
		let encode_commands = report["nodes"]
			.as_array()
			.unwrap()
			.iter()
			.filter(|node| node["kind"] == "video_encode")
			.count();
		assert_eq!(encode_commands, frames.len());
		plan.validate_serialized_descriptor(&plan.serialize_descriptor()?)?;

		// Two replays on the timeline: each emits conversion on the compute
		// family then the encode picture, with packets materializing at the
		// event boundary.
		let first = runtime.submit(&plan)?;
		let second = runtime.submit(&plan)?;
		let first_packets = first.encoded_packets()?;
		let second_packets = second.encoded_packets()?;
		for (label, packets) in [("first", first_packets), ("second", second_packets)] {
			assert_eq!(packets.len(), frames.len(), "{label} replay packet count");
			for (index, packet) in packets.iter().enumerate() {
				assert_eq!(
					packet.presentation_timestamp_micros(),
					(index as u64) * 33_333,
					"{label} frame {index} timestamp"
				);
				assert_eq!(
					packet.is_keyframe(),
					index == 0,
					"{label} frame {index} keyframe"
				);
				assert!(
					packet.bitstream().windows(4).any(|w| w == [0, 0, 0, 1]),
					"{label} frame {index} must be Annex-B framed"
				);
			}
		}

		// The live session is independent of the captured replay session.
		let live = encoder.encode(&rgba, 0)?;
		assert!(live.is_keyframe());
		encoder.close();
		Ok(())
	}
);

test_vk!(
	video_recorder_validates_and_writes_a_demuxable_mp4,
	runtime,
	{
		let path = temporary_mp4("recorder");
		let settings = VideoEncodeConfig::new(VideoEncodeProfile::h264_high_420_8bit(), 320, 240);
		let unsupported = match VideoRecorder::new(
			&runtime,
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
			&runtime,
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

		let caps = oa::video::query_device_capabilities(&runtime)?;
		let mut recorder = match VideoRecorder::new(
			&runtime,
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
	abandoned_recordings_retire_in_flight_encode_resources,
	runtime,
	{
		let rgba = vec![96; 320 * 240 * 4];
		for explicit_close in [false, true] {
			let path = temporary_mp4("abandoned");
			let mut recorder = VideoRecorder::new(
				&runtime,
				VideoRecorderConfig {
					output_path: path.to_string_lossy().into_owned(),
					encode: VideoEncodeConfig::new(VideoEncodeProfile::h264_high_420_8bit(), 320, 240),
					..Default::default()
				},
			)?;
			recorder.write_rgba(&rgba, 0)?;
			if explicit_close {
				recorder.close();
			}
			drop(recorder);
			runtime.checkpoint()?.wait()?;
			std::fs::remove_file(path).expect("delete abandoned recording");
		}
		Ok(())
	}
);

test_vk!(
	h264_and_h265_recordings_match_independent_ffmpeg_pixels,
	runtime,
	"requires both H.264/H.265 hardware encode profiles and FFmpeg",
	{
		let caps = oa::video::query_device_capabilities(&runtime)?;
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
				&runtime,
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
	runtime,
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
			if oa::video::query_encode_capabilities(&runtime, profile).is_err() {
				continue;
			}
			assert_ten_bit_encode_oracle(&runtime, profile, expected)?;
		}
		Ok(())
	}
);

fn assert_ten_bit_encode_oracle(
	runtime: &oa::Runtime,
	profile: VideoEncodeProfile,
	expected_profile: &str,
) -> oa::Result<()> {
	{
		let mut encode = VideoEncodeConfig::new(profile, 320, 240);
		encode.gop_size = 2;
		encode.constant_qp = 10;
		let mut encoder = VideoEncoder::new(runtime, encode)?;
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

test_vk!(
	captured_decode_planes_feed_encode_in_one_plan,
	runtime,
	"requires Vulkan H.264 decode and encode hardware plus the OA donor fixture",
	{
		let caps = oa::video::query_device_capabilities(&runtime)?;
		if !(caps.decoder_sessions_available()
			&& caps.supports_h264_decode()
			&& caps.encoder_sessions_available()
			&& caps.supports_h264_encode())
		{
			return Ok(());
		}
		let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
			.with_file_name("oa")
			.join("sdk/asset/video/clip/shibuya_720p_30fps_h264_high_8bit_420.mp4");
		let mut demuxer = VideoDemuxer::open(&fixture)?;
		let info = demuxer.info();
		let mut packets = Vec::new();
		for _ in 0..4 {
			match demuxer.read_next_packet()? {
				Some(packet) => packets.push(packet),
				None => break,
			}
		}
		assert!(packets.len() >= 2, "fixture must carry a decodable run");

		let mut decoder = VideoDecoder::new(&runtime, info)?;
		let mut encoder = VideoEncoder::new(
			&runtime,
			VideoEncodeConfig::new(
				VideoEncodeProfile::h264_high_420_8bit(),
				info.width(),
				info.height(),
			),
		)?;

		// One captured plan: decode pictures on the video family, plane buffers
		// staged through the compute family, encode-input copies on compute,
		// and encode pictures on the encode family — timeline-chained with no
		// host waits inside the graph.
		let (plan, frame_count) = runtime.capture(|| {
			let frames = decoder.record_to_planes(&packets)?;
			let inputs = frames
				.iter()
				.map(|frame| {
					let planes = frame
						.planes()
						.ok_or_else(|| Error::internal("recorded decode produced no planes"))?;
					let pts = u64::try_from(frame.timing().presentation_timestamp().as_micros())
						.map_err(|_| Error::out_of_range("frame timestamp overflows u64"))?;
					Ok((planes, pts))
				})
				.collect::<oa::Result<Vec<(&[Matrix], u64)>>>()?;
			let count = inputs.len();
			encoder.record_planes(&inputs)?;
			Ok(count)
		})?;
		assert!(frame_count > 0, "capture produced no encodable frames");

		let report: serde_json::Value =
			serde_json::from_str(&plan.debug_report_json("decode-encode")).unwrap();
		let node_kinds: Vec<&str> = report["nodes"]
			.as_array()
			.unwrap()
			.iter()
			.map(|node| node["kind"].as_str().unwrap())
			.collect();
		assert!(
			node_kinds.contains(&"video_decode"),
			"plan must carry decode nodes: {node_kinds:?}"
		);
		assert_eq!(
			node_kinds
				.iter()
				.filter(|kind| **kind == "video_encode")
				.count(),
			frame_count
		);
		assert!(
			plan.diagnostics().queue_run_count() >= 2,
			"decode/compute/encode must split into multiple queue runs"
		);
		plan.validate_serialized_descriptor(&plan.serialize_descriptor()?)?;

		let first = runtime.submit(&plan)?;
		let second = runtime.submit(&plan)?;
		for (label, packets) in [
			("first", first.encoded_packets()?),
			("second", second.encoded_packets()?),
		] {
			assert_eq!(packets.len(), frame_count, "{label} replay packet count");
			for (index, packet) in packets.iter().enumerate() {
				assert_eq!(
					packet.is_keyframe(),
					index == 0,
					"{label} frame {index} keyframe"
				);
				assert!(
					packet.bitstream().windows(4).any(|w| w == [0, 0, 0, 1]),
					"{label} frame {index} must be Annex-B framed"
				);
			}
		}

		decoder.close();
		encoder.close();
		Ok(())
	}
);

test_vk!(
	decoded_planes_flow_through_model_scores_into_encode,
	runtime,
	"requires Vulkan video decode/encode hardware plus FFmpeg",
	{
		let caps = oa::video::query_device_capabilities(&runtime)?;
		let decodes = [
			(
				"shibuya_720p_30fps_h264_high_8bit_420.mp4",
				VideoCodec::H264,
				caps.supports_h264_decode(),
			),
			(
				"shibuya_720p_30fps_h265_main_8bit_420.mp4",
				VideoCodec::H265,
				caps.supports_h265_decode(),
			),
			(
				"shibuya_720p_30fps_vp9_profile0_8bit_420.mp4",
				VideoCodec::Vp9,
				caps.supports_vp9_decode(),
			),
			(
				"shibuya_720p_30fps_av1_main_8bit_420.mp4",
				VideoCodec::Av1,
				caps.supports_av1_decode(),
			),
		];
		let encodes = [
			(
				VideoEncodeProfile::h264_high_420_8bit(),
				"h264",
				caps.supports_h264_encode(),
			),
			(
				VideoEncodeProfile::h265_main_420_8bit(),
				"hevc",
				caps.supports_h265_encode(),
			),
		];
		let mut covered = 0usize;
		for (fixture, codec, decode_ok) in decodes {
			for (profile, ffmpeg_codec, encode_ok) in &encodes {
				if !(caps.decoder_sessions_available()
					&& decode_ok
					&& caps.encoder_sessions_available()
					&& *encode_ok)
				{
					continue;
				}
				eprintln!("pipeline {fixture} -> {ffmpeg_codec}");
				run_model_scores_pipeline(&runtime, fixture, codec, *profile, ffmpeg_codec)?;
				covered += 1;
			}
		}
		assert!(covered > 0, "no admitted decode/encode pair on this device");
		Ok(())
	}
);

/// One CC-31 matrix cell: `fixture` decodes on the video queue, a luma-input
/// affine score head and `overlay_scores` run on compute, and `profile`
/// encodes on the encode queue — one captured plan, two replays, verified
/// against an independently decoded reference composite through FFmpeg.
fn run_model_scores_pipeline(
	runtime: &oa::Runtime,
	fixture: &str,
	codec: VideoCodec,
	profile: VideoEncodeProfile,
	ffmpeg_codec: &str,
) -> oa::Result<()> {
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip")
		.join(fixture);
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	assert_eq!(info.codec(), codec, "fixture codec");
	let mut packets = Vec::new();
	for _ in 0..4 {
		match demuxer.read_next_packet()? {
			Some(packet) => packets.push(packet),
			None => break,
		}
	}
	assert!(packets.len() >= 2, "fixture must carry a decodable run");
	let (width, height) = (info.width() as usize, info.height() as usize);

	// Independent reference planes: the same packets decoded on a second
	// session, harvested to packed I420 for the expected-composite oracle.
	let mut reference_decoder = VideoDecoder::new(runtime, info)?;
	let mut reference = Vec::new();
	for packet in &packets {
		if let Some(frame) = reference_decoder.decode_native(packet)? {
			reference.push(reference_decoder.read_native_yuv420p(&frame)?);
		}
	}
	for frame in reference_decoder.flush()? {
		reference.push(reference_decoder.read_native_yuv420p(&frame)?);
	}
	assert!(!reference.is_empty(), "reference decode produced no frames");

	const THRESHOLD: f32 = 0.5;
	const FILL: [u8; 3] = [235, 90, 160];
	let luma_config = oa::video::VideoMatrixConfig {
		channels: oa::video::MatrixChannels::Luma,
		..Default::default()
	};

	// Encode input buffers must cover the codec's coded extent (block
	// granularity), which can exceed the visible extent — pad when needed.
	let enc_caps = oa::video::query_encode_capabilities(runtime, profile)?;
	let gran = enc_caps.picture_access_granularity();
	let gw = gran.width.max(16) as usize;
	let gh = gran.height.max(16) as usize;
	let coded_w = width.div_ceil(gw) * gw;
	let coded_h = height.div_ceil(gh) * gh;

	let mut decoder = VideoDecoder::new(runtime, info)?;
	let mut encode = VideoEncodeConfig::new(profile, info.width(), info.height());
	encode.gop_size = 4;
	encode.constant_qp = 10;
	let mut encoder = VideoEncoder::new(runtime, encode)?;

	// One captured plan: decode on the video queue, the CC-30 conversion
	// region plus a 1x1 affine score head and the score-driven stencil on
	// compute, encode-input staging and encode pictures on their families —
	// timeline-chained with no host waits.
	let (plan, expected_pts) = runtime.capture(|| {
		let frames = decoder.record_to_planes(&packets)?;
		let mut overlaid = Vec::with_capacity(frames.len());
		for frame in &frames {
			let tensor = oa::video::to_matrix(std::slice::from_ref(frame), &luma_config)?;
			let flat = oa::matrix::reshape(&tensor, [width * height])?;
			let scores = oa::matrix::sub_scalar(&oa::matrix::scale(&flat, 1.25)?, 0.125)?;
			let mut stamped = oa::video::overlay_scores(frame, &scores, THRESHOLD, FILL)?;
			if coded_w != width || coded_h != height {
				stamped = oa::video::pad(
					&stamped,
					oa::video::VideoPadding {
						left: 0,
						top: 0,
						right: (coded_w - width) as u32,
						bottom: (coded_h - height) as u32,
					},
					[16, 128, 128],
				)?;
			}
			let pts = u64::try_from(frame.timing().presentation_timestamp().as_micros())
				.map_err(|_| Error::out_of_range("frame timestamp overflows u64"))?;
			overlaid.push((stamped, pts));
		}
		let mut inputs = Vec::with_capacity(overlaid.len());
		for (frame, pts) in &overlaid {
			let planes = frame
				.planes()
				.ok_or_else(|| Error::internal("overlay produced no planes"))?;
			inputs.push((planes, *pts));
		}
		let expected_pts = inputs.iter().map(|(_, pts)| *pts).collect::<Vec<_>>();
		encoder.record_planes(&inputs)?;
		Ok(expected_pts)
	})?;
	let frame_count = expected_pts.len();
	assert_eq!(
		frame_count,
		reference.len(),
		"capture produced every reference frame"
	);

	let report: serde_json::Value =
		serde_json::from_str(&plan.debug_report_json("decode-model-encode")).unwrap();
	let node_kinds: Vec<&str> = report["nodes"]
		.as_array()
		.unwrap()
		.iter()
		.filter_map(|node| node["kind"].as_str())
		.collect();
	assert!(
		node_kinds.contains(&"video_decode") && node_kinds.contains(&"video_encode"),
		"{codec:?}/{ffmpeg_codec}: plan must carry codec nodes: {node_kinds:?}"
	);
	assert!(
		plan.diagnostics().queue_run_count() >= 2,
		"{codec:?}/{ffmpeg_codec}: decode/compute/encode must split into multiple queue runs"
	);

	let first = runtime.submit(&plan)?;
	let second = runtime.submit(&plan)?;
	let first_packets = first.encoded_packets()?;
	let second_packets = second.encoded_packets()?;
	let check_packets = |packets: &[oa::video::EncodedVideoPacket]| {
		assert_eq!(packets.len(), frame_count);
		for (packet, &pts) in packets.iter().zip(&expected_pts) {
			assert_eq!(packet.presentation_timestamp_micros(), pts);
			assert!(!packet.bitstream().is_empty());
		}
	};
	check_packets(&first_packets);
	check_packets(&second_packets);

	// Independent FFmpeg decode of the first replay's elementary stream, then
	// a pixel comparison against the expected stencil composite.
	let mut stream = Vec::new();
	for packet in &first_packets {
		stream.extend_from_slice(packet.bitstream());
	}
	let mut ffmpeg = std::process::Command::new("ffmpeg")
		.args([
			"-v",
			"error",
			"-f",
			ffmpeg_codec,
			"-i",
			"pipe:0",
			"-pix_fmt",
			"yuv420p",
			"-f",
			"rawvideo",
			"-",
		])
		.stdin(std::process::Stdio::piped())
		.stdout(std::process::Stdio::piped())
		.spawn()
		.expect("FFmpeg is required for the independent encode oracle");
	ffmpeg
		.stdin
		.as_mut()
		.unwrap()
		.write_all(&stream)
		.expect("write encoded stream");
	drop(ffmpeg.stdin.take());
	let decoded = ffmpeg.wait_with_output().expect("FFmpeg output");
	assert!(
		decoded.status.success(),
		"{codec:?}/{ffmpeg_codec}: ffmpeg decode failed: {}",
		String::from_utf8_lossy(&decoded.stderr)
	);
	let (cw, chh) = (width.div_ceil(2), height.div_ceil(2));
	let frame_bytes = width * height + 2 * cw * chh;
	assert_eq!(
		decoded.stdout.len(),
		frame_bytes * frame_count,
		"{codec:?}/{ffmpeg_codec} decoded extent"
	);
	for (index, (out, src)) in decoded
		.stdout
		.chunks_exact(frame_bytes)
		.zip(reference.iter())
		.enumerate()
	{
		let (out_y, rest) = out.split_at(width * height);
		let (out_u, out_v) = rest.split_at(cw * chh);
		let src_y = &src[..width * height];
		let src_u = &src[width * height..width * height + cw * chh];
		let src_v = &src[width * height + cw * chh..];
		let mut mismatched = 0usize;
		for y in 0..height {
			for x in 0..width {
				let luma = src_y[y * width + x];
				let score = (f64::from(luma) - 16.0) / 219.0 * 1.25 - 0.125;
				let expected = if score >= f64::from(THRESHOLD) {
					FILL[0]
				} else {
					luma
				};
				if out_y[y * width + x].abs_diff(expected) > 16 {
					mismatched += 1;
				}
			}
		}
		assert!(
			mismatched * 20 < width * height,
			"{codec:?}/{ffmpeg_codec} frame {index}: {mismatched} luma pixels diverge"
		);
		let mut chroma_mismatch = 0usize;
		for cy in 0..chh {
			for cx in 0..cw {
				let luma = src_y[(cy * 2) * width + cx * 2];
				let score = (f64::from(luma) - 16.0) / 219.0 * 1.25 - 0.125;
				let pass = score >= f64::from(THRESHOLD);
				for (out_plane, src_plane, fill_code) in [(out_u, src_u, FILL[1]), (out_v, src_v, FILL[2])]
				{
					let expected = if pass {
						fill_code
					} else {
						src_plane[cy * cw + cx]
					};
					if out_plane[cy * cw + cx].abs_diff(expected) > 16 {
						chroma_mismatch += 1;
					}
				}
			}
		}
		assert!(
			chroma_mismatch * 20 < 2 * cw * chh,
			"{codec:?}/{ffmpeg_codec} frame {index}: {chroma_mismatch} chroma pixels diverge"
		);
	}

	crate::bench_media::record(
		runtime,
		&plan,
		serde_json::json!({
			"kind": "decode_model_encode",
			"decode": format!("{codec:?}"),
			"encode": ffmpeg_codec,
			"width": width,
			"height": height,
			"frames": frame_count,
			"fixture": fixture.file_name().and_then(|name| name.to_str()),
		}),
		|event| {
			check_packets(&event.encoded_packets()?);
			Ok(())
		},
	)?;
	decoder.close();
	encoder.close();
	reference_decoder.close();
	Ok(())
}
