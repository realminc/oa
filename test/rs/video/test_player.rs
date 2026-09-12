use std::{any::TypeId, time::Duration};

use oa::{Engine, VideoPlayer, video::VideoPlayerConfig};

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
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_h264_high_8bit_420.mp4");
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
		Ok(oa::cryptography::hash(
			player
				.current_frame()?
				.as_yuv420p()
				.expect("player retains decoded planar frame"),
		)
		.to_hex())
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
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_h264_high_8bit_420.mp4");
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
