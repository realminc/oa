#[macro_use]
#[path = "support/test_vk.rs"]
mod test_vk_fixture;

#[path = "support/bench_media.rs"]
mod bench_media;

use std::{any::TypeId, path::PathBuf, time::Duration};

use oa::{
	MediaPlayer,
	media::{MediaPlayerConfig, MediaTimestamp},
	video::{VideoFrameTiming, VideoTimeBase},
};

#[test]
fn public_media_session_has_one_identity_and_explicit_track_sources() {
	assert_eq!(
		TypeId::of::<MediaPlayer>(),
		TypeId::of::<oa::media::MediaPlayer>()
	);
	let config = MediaPlayerConfig::new("video.mp4", "audio.wav");
	assert_eq!(config.video_path, PathBuf::from("video.mp4"));
	assert_eq!(config.audio_path, PathBuf::from("audio.wav"));

	let embedded = MediaPlayerConfig::embedded("movie.mp4");
	assert_eq!(embedded.video_path, PathBuf::from("movie.mp4"));
	assert_eq!(embedded.audio_path, embedded.video_path);
}

#[test]
fn video_never_advances_before_its_next_presentation_boundary() {
	let timing = VideoFrameTiming::new(Duration::from_millis(40), Some(Duration::from_millis(20)))
		.expect("valid timing");
	assert!(
		!MediaTimestamp::new_from_microseconds(59_000)
			.frame_is_due(timing)
			.unwrap()
	);
	assert!(
		MediaTimestamp::new_from_microseconds(60_000)
			.frame_is_due(timing)
			.unwrap()
	);
	assert!(
		MediaTimestamp::new_from_microseconds(90_000)
			.frame_is_due(timing)
			.unwrap()
	);
	let unknown = VideoFrameTiming::new(Duration::ZERO, None).unwrap();
	assert!(
		MediaTimestamp::new_from_microseconds(1_000_000)
			.frame_is_due(unknown)
			.is_err()
	);
}

#[test]
fn microsecond_seek_uses_exact_rational_time_base() {
	let base = VideoTimeBase::new(1, 90_000).unwrap();
	assert_eq!(
		MediaTimestamp::new_from_microseconds(1_000_000)
			.to_video_ticks(base)
			.unwrap(),
		90_000
	);
	assert_eq!(
		MediaTimestamp::new_from_microseconds(33_333)
			.to_video_ticks(base)
			.unwrap(),
		2_999
	);
	let tiny_unit = VideoTimeBase::new(1, u32::MAX).unwrap();
	assert!(
		MediaTimestamp::new_from_microseconds(u64::MAX)
			.to_video_ticks(tiny_unit)
			.is_err()
	);
}

test_vk!(
	embedded_container_audio_decodes_and_video_demuxes_the_same_file,
	engine,
	{
		for (fixture, bytes, codec, extension) in [
			(
				"embedded_av.mp4" as &str,
				include_bytes!("../fixtures/media/embedded_av.mp4").as_slice(),
				oa::video::VideoCodec::H264,
				"mp4" as &str,
			),
			(
				"embedded_av.mkv",
				include_bytes!("../fixtures/media/embedded_av.mkv").as_slice(),
				oa::video::VideoCodec::Vp9,
				"mkv",
			),
		] {
			let audio = oa::audio::decode_memory(&engine, bytes)?;
			assert_eq!(audio.channels(), 1, "{extension} channel count");
			assert_eq!(audio.sample_rate(), 44_100, "{extension} sample rate");
			assert!(audio.samples() > 40_000, "{extension} sample count");
			let samples = audio.as_matrix().read_f32()?;
			assert!(samples.iter().all(|sample| sample.is_finite()));
			let mean_square = samples
				.iter()
				.map(|sample| f64::from(*sample) * f64::from(*sample))
				.sum::<f64>()
				/ samples.len() as f64;
			assert!(mean_square.sqrt() > 0.05, "{extension} is not silent");

			let mut demuxer = oa::video::VideoDemuxer::open(
				PathBuf::from(env!("CARGO_MANIFEST_DIR"))
					.join("test/fixtures/media")
					.join(fixture),
			)?;
			assert_eq!(demuxer.info().codec(), codec, "{extension} video codec");
			assert!(demuxer.read_next_packet()?.is_some());
		}
		Ok(())
	}
);

#[test]
#[ignore = "requires native H.264/VP9 Vulkan Video and an audio output device"]
fn embedded_playback_survives_seek_pause_and_audio_drain() -> oa::Result<()> {
	let engine = oa::Engine::new()?;
	for fixture in ["embedded_av.mp4", "embedded_av.mkv"] {
		let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
			.join("test/fixtures/media")
			.join(fixture);
		let mut player = MediaPlayer::open(&engine, MediaPlayerConfig::embedded(path))?;
		assert_eq!(player.position_us()?, 0);
		player.seek_us(400_000)?;
		let at_seek = player.current_frame()?.timing().presentation_timestamp();
		assert!(at_seek <= Duration::from_millis(425));
		assert_eq!(player.tick(2)?, 0);
		player.play()?;
		let deadline = std::time::Instant::now() + Duration::from_secs(4);
		let mut last_position = 400_000;
		let mut frames = 0;
		while !player.is_done() && std::time::Instant::now() < deadline {
			let advanced = player.tick(2)?;
			assert!(advanced <= 2);
			frames += advanced;
			let position = player.position_us()?;
			assert!(position >= last_position);
			last_position = position;
			std::thread::sleep(Duration::from_millis(4));
		}
		assert!(player.is_done(), "{fixture} stopped at {last_position} µs");
		assert!(frames > 3, "{fixture} advanced {frames} frames");
		assert!(last_position >= 950_000, "{fixture} stopped too early");
		player.play()?;
		assert!(player.position_us()? < 10_000);
		player.pause()?;
		let paused = player.position_us()?;
		std::thread::sleep(Duration::from_millis(80));
		assert_eq!(player.tick(2)?, 0);
		assert_eq!(player.position_us()?, paused);
		player.close()?;
	}
	Ok(())
}

#[test]
#[ignore = "requires native H.264 Vulkan Video, an audio output device and FFmpeg"]
fn sequential_transport_plays_the_longer_track_to_completion() -> oa::Result<()> {
	let engine = oa::Engine::new()?;
	let dir = std::env::temp_dir().join(format!("oa-media-tails-{}", std::process::id()));
	std::fs::create_dir(&dir).expect("temporary media directory");
	let ts = dir.join("video.ts");
	let wav = dir.join("audio.wav");
	let fixture =
		PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/fixtures/media/embedded_av.mp4");
	let ffmpeg = || {
		let mut cmd = std::process::Command::new("ffmpeg");
		cmd.args(["-v", "error", "-y"]);
		cmd
	};
	assert!(
		ffmpeg()
			.arg("-i")
			.arg(fixture)
			.args(["-an", "-c:v", "copy"])
			.arg(&ts)
			.status()
			.unwrap()
			.success()
	);
	for duration in ["0.2", "1.6"] {
		assert!(
			ffmpeg()
				.args([
					"-f",
					"lavfi",
					"-i",
					"sine=frequency=440:sample_rate=44100",
					"-t",
					duration
				])
				.arg(&wav)
				.status()
				.unwrap()
				.success()
		);
		let mut player = MediaPlayer::open(&engine, MediaPlayerConfig::new(&ts, &wav))?;
		let origin = player.current_frame()?.timing().presentation_timestamp();
		let mut displayed = 1;
		player.play()?;
		let deadline = std::time::Instant::now() + Duration::from_secs(5);
		while !player.is_done() && std::time::Instant::now() < deadline {
			let advanced = player.tick(2)?;
			assert!(advanced <= 2);
			displayed += advanced;
			let frame_time = player.current_frame()?.timing().presentation_timestamp() - origin;
			assert!(
				frame_time.as_micros() <= u128::from(player.position_us()?),
				"frame at {frame_time:?} appeared before the media clock"
			);

			std::thread::sleep(Duration::from_millis(4));
		}
		assert!(player.is_done());
		assert_eq!(displayed, 10);
		let minimum = if duration == "0.2" {
			995_000
		} else {
			1_599_000
		};
		assert!(
			player.position_us()? >= minimum,
			"audio {duration}s, ended at {} µs",
			player.position_us()?
		);
		player.close()?;
	}
	std::fs::remove_file(wav).unwrap();
	std::fs::remove_file(ts).unwrap();
	std::fs::remove_dir(dir).unwrap();
	Ok(())
}

test_vk!(
	captured_multimodal_plan_carries_demuxed_streams_to_model_output,
	engine,
	{
		// Reference audio-visual workload: one mp4 carries an H.264 video
		// elementary stream and an AAC audio track. Host code demuxes the
		// video packets and decodes the audio; every GPU step — native video
		// decode, video preprocessing, mel features, and the event-detector
		// head — records into a single captured plan.
		let fixture =
			PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/fixtures/media/embedded_av.mp4");
		let mut demuxer = oa::video::VideoDemuxer::open(&fixture)?;
		let info = demuxer.info();
		let mut packets = Vec::new();
		while let Some(packet) = demuxer.read_next_packet()? {
			packets.push(packet);
		}
		assert!(!packets.is_empty(), "fixture produced no video packets");
		let decoder = oa::video::VideoDecoder::new(&engine, info)?;
		let audio = oa::audio::decode_file(&engine, &fixture)?;

		let video_config = oa::video::VideoMatrixConfig {
			size: Some((16, 16)),
			..oa::video::VideoMatrixConfig::default()
		};
		let mel_config = oa::audio::MelConfig {
			fft_size: 512,
			hop_size: 128,
			num_mels: 16,
			..oa::audio::MelConfig::default()
		};
		// Probe the mel shape on the host so the affine head can be sized and
		// uploaded before capture; the plan re-records the real computation.
		let mel_probe = oa::audio::mel_spectrogram(&audio, mel_config)?;
		let audio_features = mel_probe.shape().iter().product::<usize>();
		// The probe records eager dispatches; the read drains them so capture
		// starts with no pending work.
		let _ = mel_probe.read_f32()?;
		let video_features = 3 * 16 * 16;
		let class_count = 2_usize;
		let make_weight = |features: usize, salt: u32| -> oa::Result<oa::Matrix> {
			let values = (0..class_count * features)
				.map(|index| {
					let value = (index as u32).wrapping_mul(1103515245).wrapping_add(salt);
					(value % 251) as f32 / 1000.0 - 0.125
				})
				.collect::<Vec<_>>();
			oa::Matrix::new_from_f32(&engine, [class_count, features], &values)
		};
		let video_weight = make_weight(video_features, 7)?;
		let audio_weight = make_weight(audio_features, 13)?;

		let (plan, (video_frames, probabilities)) = engine.capture(|| {
			let frames = decoder.record_to_planes(&packets)?;
			let video_score = oa::video::to_matrix(&frames, &video_config)?
				.reshape(vec![frames.len(), video_features])?
				.mat_mul_nt(&video_weight)?
				.mean(0)?
				.reshape(vec![1, class_count])?;
			let audio_score = oa::audio::mel_spectrogram(&audio, mel_config)?
				.reshape(vec![1, audio_features])?
				.mat_mul_nt(&audio_weight)?;
			Ok((frames.len(), video_score.add(&audio_score)?.softmax(-1)?))
		})?;

		let report: serde_json::Value =
			serde_json::from_str(&plan.debug_report_json("multimodal-event")).unwrap();
		let kinds = report["nodes"]
			.as_array()
			.unwrap()
			.iter()
			.filter_map(|node| node["kind"].as_str())
			.collect::<Vec<_>>();
		assert!(
			kinds.contains(&"video_decode"),
			"plan carries no video_decode node: {kinds:?}"
		);
		let compute_nodes = report["nodes"]
			.as_array()
			.unwrap()
			.iter()
			.filter(|node| node["kernel"].is_string())
			.count();
		assert!(compute_nodes > 0, "plan carries no compute nodes");

		// Two correctness replays must agree bitwise.
		let captured_bytes = plan
			.captured_resources()
			.iter()
			.map(|resource| resource.byte_len() as u64)
			.sum::<u64>();
		let started = std::time::Instant::now();
		let _event = engine.submit(&plan)?;
		plan.wait()?;
		let submit_ms = started.elapsed().as_secs_f64() * 1_000.0;
		let first = probabilities.read_f32()?;
		let _event = engine.submit(&plan)?;
		plan.wait()?;
		let second = probabilities.read_f32()?;
		assert_eq!(first.len(), class_count);
		assert!(first.iter().all(|value| value.is_finite()), "{first:?}");
		let mass = first.iter().copied().sum::<f32>();
		assert!(
			(mass - 1.0).abs() < 1.0e-4,
			"softmax mass {mass} deviates from one"
		);
		assert_eq!(first, second, "replay diverged from first submission");
		eprintln!(
			"multimodal plan: {} nodes, {captured_bytes} captured bytes, {submit_ms:.3} ms submit+wait, logits {first:?}",
			kinds.len()
		);
		bench_media::record(
			&engine,
			&plan,
			serde_json::json!({
				"kind": "multimodal",
				"fixture": "embedded_av.mp4",
				"width": info.width(),
				"height": info.height(),
				"frames": video_frames,
				"audio_sample_rate": audio.sample_rate(),
				"audio_channels": audio.channels(),
				"audio_samples_per_channel": audio.as_matrix().shape()[1],
			}),
			|event| {
				event.wait()?;
				assert_eq!(probabilities.read_f32()?, first);
				Ok(())
			},
		)?;
		Ok(())
	}
);
