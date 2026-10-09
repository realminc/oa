use std::{any::TypeId, time::Duration};

use oa::{Engine, VideoDemuxer, VideoPlayer, video::VideoPlayerConfig};

fn donor_h264_fixture() -> std::path::PathBuf {
	let sibling = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
	let donor = if sibling.with_file_name("oacpp").is_dir() {
		sibling.with_file_name("oacpp")
	} else {
		sibling.with_file_name("oa")
	};
	donor.join("sdk/asset/video/clip/shibuya_720p_30fps_h264_high_8bit_420.mp4")
}

#[test]
fn root_player_export_is_the_video_session_identity() {
	assert_eq!(
		TypeId::of::<oa::VideoPlayer>(),
		TypeId::of::<oa::video::VideoPlayer>()
	);
}

#[test]
fn player_cache_defaults_match_the_bounded_donor_policy() {
	let config = VideoPlayerConfig::default();
	assert_eq!(config.presentation_cache_frames, 32);
	assert_eq!(config.presentation_cache_bytes, 256 * 1024 * 1024);
}

#[test]
#[ignore = "requires Vulkan H.264 hardware and OA donor fixture"]
fn player_steps_through_cache_and_replays_evicted_history() -> oa::Result<()> {
	let engine = Engine::new()?;
	let fixture = donor_h264_fixture();
	let mut player = VideoPlayer::open(
		&engine,
		fixture,
		VideoPlayerConfig {
			loop_playback: false,
			frame_rate_override: None,
			start_playing: false,
			presentation_cache_frames: 4,
			presentation_cache_bytes: 8 * 1024 * 1024,
		},
	)?;
	let frame_hash = |player: &VideoPlayer| -> oa::Result<String> {
		Ok(
			oa::cryptography::hash(
				player
					.current_frame()?
					.as_yuv420p()
					.expect("player retains decoded planar frame"),
			)
			.to_hex(),
		)
	};
	let mut hashes = vec![frame_hash(&player)?];
	for _ in 0..5 {
		assert!(player.advance()?);
		hashes.push(frame_hash(&player)?);
	}
	assert_eq!(player.current_frame_index()?, 5);
	let decoded_at_cursor = player
		.stats()
		.expect("open player has stats")
		.decoded_packets;

	player.step_backward()?;
	assert_eq!(player.current_frame_index()?, 4);
	assert_eq!(frame_hash(&player)?, hashes[4]);
	let stats = player.stats().expect("open player has stats");
	assert_eq!(stats.decoded_packets, decoded_at_cursor);
	assert_eq!(stats.presentation_cache_hits, 1);
	assert_eq!(stats.presentation_cache_resident, 4);
	assert_eq!(stats.presentation_cache_capacity, 4);

	assert!(player.advance()?);
	assert_eq!(player.current_frame_index()?, 5);
	assert_eq!(frame_hash(&player)?, hashes[5]);
	let stats = player.stats().expect("open player has stats");
	assert_eq!(stats.decoded_packets, decoded_at_cursor);
	assert_eq!(stats.presentation_cache_hits, 2);

	player.step_frames(-4)?;
	assert_eq!(player.current_frame_index()?, 1);
	assert_eq!(frame_hash(&player)?, hashes[1]);
	let stats = player.stats().expect("open player has stats");
	assert_eq!(stats.presentation_cache_misses, 1);
	assert_eq!(stats.seek_replay_frames, 2);
	assert!(stats.decoded_packets > decoded_at_cursor);
	Ok(())
}

#[test]
#[ignore = "requires Vulkan H.264 hardware and OA donor fixture"]
fn player_composes_decode_pacing_eos_seek_and_close() -> oa::Result<()> {
	let engine = Engine::new()?;
	let fixture = donor_h264_fixture();
	let mut player = VideoPlayer::open(
		&engine,
		&fixture,
		VideoPlayerConfig {
			loop_playback: false,
			frame_rate_override: None,
			start_playing: false,
			presentation_cache_frames: 4,
			presentation_cache_bytes: 8 * 1024 * 1024,
		},
	)?;
	let info = player.info().expect("open player retains stream metadata");
	assert_eq!(player.current_frame_index()?, 0);
	let first_boundary = player
		.next_presentation_boundary()?
		.expect("fixture contains a second frame");
	assert!(!player.is_playing());
	assert_eq!(player.tick(Duration::from_secs(1))?, 0);
	assert_eq!(
		oa::cryptography::hash(
			player
				.current_frame()?
				.as_yuv420p()
				.expect("player retains decoded planar frame")
		)
		.to_hex(),
		"360d151e07a39eac2314dd527bf7ca1802e5ed3b980e42409345b6668736e8fd"
	);

	let mut previous_pts = player.current_frame()?.timing().presentation_timestamp();
	let mut frame_count = 1_usize;
	let mut frame_30_hash = None;
	while player.advance()? {
		let pts = player.current_frame()?.timing().presentation_timestamp();
		assert!(pts >= previous_pts);
		previous_pts = pts;
		if player.current_frame_index()? == 30 {
			frame_30_hash = Some(
				oa::cryptography::hash(
					player
						.current_frame()?
						.as_yuv420p()
						.expect("player retains decoded planar frame"),
				)
				.to_hex(),
			);
		}
		frame_count += 1;
	}
	assert_eq!(frame_count, info.sample_count() as usize);
	assert!(player.is_done());
	assert_eq!(
		player
			.stats()
			.expect("open player has stats")
			.decoded_packets,
		60
	);
	let mut source = VideoDemuxer::open(&fixture)?;
	let mut timestamps = Vec::new();
	while let Some(packet) = source.read_next_packet()? {
		timestamps.push(packet.presentation_timestamp());
	}
	timestamps.sort_unstable();
	let between = timestamps[0] + (timestamps[1] - timestamps[0]) / 2;
	assert!(between < timestamps[1]);
	player.seek_at_or_before(between)?;
	assert_eq!(player.current_frame_index()?, 0);
	player.seek_at_or_before(timestamps[1])?;
	assert_eq!(player.current_frame_index()?, 1);
	assert_eq!(
		player.current_frame()?.timing().presentation_timestamp(),
		first_boundary
	);
	player.seek_frame(30)?;
	assert_eq!(player.current_frame_index()?, 30);
	assert_eq!(
		oa::cryptography::hash(
			player
				.current_frame()?
				.as_yuv420p()
				.expect("player retains decoded planar frame")
		)
		.to_hex(),
		frame_30_hash.expect("frame 30 was observed during linear playback")
	);
	assert!(player.seek_frame(u64::from(info.sample_count())).is_err());

	player.reset()?;
	assert_eq!(player.current_frame_index()?, 0);
	assert!(!player.is_done());
	player.play()?;
	assert!(player.is_playing());
	assert_eq!(player.tick(Duration::from_millis(34))?, 1);
	player.pause()?;
	assert!(!player.is_playing());
	player.close();
	assert!(!player.is_open());
	assert!(player.info().is_none());
	assert!(player.current_frame().is_err());
	Ok(())
}

#[test]
#[ignore = "requires Vulkan H.264 hardware and OA donor fixture"]
fn native_player_retains_decoder_frames_without_host_pixels() -> oa::Result<()> {
	let engine = Engine::new()?;
	let mut player = VideoPlayer::open_native(
		&engine,
		donor_h264_fixture(),
		VideoPlayerConfig {
			loop_playback: false,
			start_playing: false,
			..VideoPlayerConfig::default()
		},
	)?;
	assert_eq!(player.stats().unwrap().presentation_cache_capacity, 0);
	let first = player.current_frame()?.clone();
	assert!(first.native_format().is_some());
	assert!(first.ready_event().is_some());
	assert!(first.as_yuv420p().is_none());
	assert!(player.advance()?);
	assert_eq!(player.current_frame_index()?, 1);
	assert!(player.current_frame()?.native_format().is_some());
	assert!(first.native_format().is_some());
	drop(first);
	player.step_backward()?;
	assert_eq!(player.current_frame_index()?, 0);
	assert_eq!(player.stats().unwrap().presentation_cache_resident, 0);
	player.close();
	Ok(())
}

#[test]
#[ignore = "requires Vulkan H.264 hardware, OA donor fixture, and FFmpeg remux"]
fn player_streams_mpeg_ts_without_a_display_index() -> oa::Result<()> {
	let output = std::env::temp_dir().join(format!("oa_player_ts_{}.ts", std::process::id()));
	let status = std::process::Command::new("ffmpeg")
		.args(["-v", "error", "-y", "-i"])
		.arg(donor_h264_fixture())
		.args([
			"-map", "0:v:0", "-c", "copy", "-an", "-t", "1", "-f", "mpegts",
		])
		.arg(&output)
		.status()
		.expect("FFmpeg must be installed for the ignored remux test");
	assert!(status.success());
	let result = (|| -> oa::Result<()> {
		let engine = Engine::new()?;
		let mut player = VideoPlayer::open(
			&engine,
			&output,
			VideoPlayerConfig {
				loop_playback: false,
				start_playing: false,
				..VideoPlayerConfig::default()
			},
		)?;
		assert_eq!(player.current_frame_index()?, 0);
		assert_eq!(player.stats().unwrap().presentation_cache_capacity, 0);
		assert!(player.current_frame()?.as_yuv420p().is_some());
		assert!(player.advance()?);
		assert_eq!(player.current_frame_index()?, 1);
		assert!(player.step_backward().is_err());
		assert!(player.seek_frame(1).is_err());
		assert!(player.seek(1).is_err());
		player.reset()?;
		assert_eq!(player.current_frame_index()?, 0);
		player.close();
		Ok(())
	})();
	let _ = std::fs::remove_file(output);
	result
}
