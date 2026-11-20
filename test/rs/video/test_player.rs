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
			oa::crypto::hash(
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
		oa::crypto::hash(
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
				oa::crypto::hash(
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
		oa::crypto::hash(
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
#[ignore = "requires Vulkan AV1 hardware, OA donor fixture, and the Core profile"]
fn native_player_presents_av1_frames_as_textures() -> oa::Result<()> {
	let engine = Engine::new()?;
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_av1_main_8bit_420.mp4");
	let mut player = VideoPlayer::new_native(
		&engine,
		fixture,
		VideoPlayerConfig {
			loop_playback: false,
			start_playing: false,
			..VideoPlayerConfig::default()
		},
	)?;
	let mut presented = 0_u32;
	loop {
		let texture = player.current_frame_texture()?;
		assert_eq!(texture.width(), player.current_frame()?.width());
		assert_eq!(texture.height(), player.current_frame()?.height());
		assert_eq!(texture.dtype(), oa::DType::U8);
		presented += 1;
		if !player.advance()? {
			break;
		}
	}
	assert!(presented >= 2);
	player.close();
	Ok(())
}

#[test]
#[ignore = "requires Vulkan AV1 8/10-bit decode; exercises repeated reverse scrubbing"]
fn av1_native_reverse_scrub_preserves_pixels_across_repeated_seeks() -> oa::Result<()> {
	let engine = Engine::new()?;
	for bits in [8, 10] {
		let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
			"sdk/asset/video/clip/shibuya_720p_30fps_av1_main_{bits}bit_420.mp4"
		));
		let mut player = VideoPlayer::new_native(
			&engine,
			fixture,
			VideoPlayerConfig {
				loop_playback: false,
				start_playing: false,
				..VideoPlayerConfig::default()
			},
		)?;
		let mut hashes = Vec::new();
		let mut retained = Vec::new();
		loop {
			let texture = player.current_frame_texture()?;
			hashes.push(oa::crypto::hash(&texture.read_rgba8()?).to_hex());
			if retained.len() < 4 {
				retained.push(texture);
			}
			if !player.advance()? {
				break;
			}
		}
		assert_eq!(hashes.len(), 60);
		for round in 0..3 {
			eprintln!("AV1 {bits}-bit reverse scrub round {round}");
			player.seek_frame(59)?;
			for index in (0..59).rev() {
				player.step_backward().unwrap_or_else(|error| {
					panic!("AV1 {bits}-bit round {round} reverse target {index}: {error}")
				});
				assert_eq!(player.current_frame_index()?, index);
				let actual = player.current_frame_texture()?.read_rgba8()?;
				assert_eq!(
					oa::crypto::hash(&actual).to_hex(),
					hashes[index as usize],
					"{bits}-bit round {round} frame {index}"
				);
			}
			for index in [47, 3, 58, 17, 0, 31, 59] {
				player.seek_frame(index)?;
				let actual = player.current_frame_texture()?.read_rgba8()?;
				assert_eq!(oa::crypto::hash(&actual).to_hex(), hashes[index as usize]);
			}
		}
		player.close();
		assert!(!retained[0].read_rgba8()?.is_empty());
	}
	Ok(())
}

#[test]
#[ignore = "requires all four Vulkan decode codecs and Core presentation"]
fn native_cached_scrubbing_uses_no_decode_or_conversion_work() -> oa::Result<()> {
	let engine = Engine::new()?;
	let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("sdk/asset/video/clip");
	let mut paths = std::fs::read_dir(directory)
		.expect("video fixtures")
		.map(|entry| entry.expect("fixture entry").path())
		.filter(|path| path.extension().is_some_and(|ext| ext == "mp4"))
		.collect::<Vec<_>>();
	paths.sort();
	assert_eq!(paths.len(), 9);
	for path in paths {
		let mut player = VideoPlayer::new_native(
			&engine,
			&path,
			VideoPlayerConfig {
				loop_playback: false,
				start_playing: false,
				..VideoPlayerConfig::default()
			},
		)?;
		loop {
			player.current_frame_texture()?;
			if !player.advance()? {
				break;
			}
		}
		engine.checkpoint()?.wait()?;
		let before = player.stats().expect("open player");
		assert_eq!(before.presentation_cache_resident, 32);
		let mut elapsed = Vec::new();
		for _ in 0..10 {
			for direction in [-1, 1] {
				for _ in 0..16 {
					let started = std::time::Instant::now();
					player.step_frames(direction)?;
					// A cached step already carries its texture; presenting it
					// cannot implicitly convert a decoder-owned frame again.
					assert!(player.current_frame()?.as_texture().is_some());
					player.current_frame_texture()?;
					elapsed.push(started.elapsed().as_nanos());
				}
			}
		}
		let after = player.stats().expect("open player");
		assert_eq!(after.decoded_packets, before.decoded_packets);
		assert_eq!(after.seek_resets, before.seek_resets);
		assert_eq!(
			after.presentation_cache_misses,
			before.presentation_cache_misses
		);
		assert_eq!(
			after.presentation_cache_hits - before.presentation_cache_hits,
			320
		);
		elapsed.sort();
		eprintln!(
			"SCRUB clip={} cached_steps=320 p50_ns={} p95_ns={} decoded_packets=0 seeks=0 conversions=0",
			path.display(),
			elapsed[160],
			elapsed[304]
		);
		player.close();
	}
	Ok(())
}

#[test]
#[ignore = "requires Vulkan AV1 decode and Core presentation"]
fn av1_native_reverse_replay_without_history_does_not_exhaust_dpb() -> oa::Result<()> {
	let engine = Engine::new()?;
	for bits in [8, 10] {
		let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
			"sdk/asset/video/clip/shibuya_720p_30fps_av1_main_{bits}bit_420.mp4"
		));
		let mut player = VideoPlayer::new_native(
			&engine,
			path,
			VideoPlayerConfig {
				loop_playback: false,
				start_playing: false,
				presentation_cache_frames: 0,
				..VideoPlayerConfig::default()
			},
		)?;
		player.seek_frame(59)?;
		for index in (43..59).rev() {
			player.step_backward()?;
			assert_eq!(player.current_frame_index()?, index);
			player.current_frame_texture()?.read_rgba8()?;
		}
		assert_eq!(player.stats().expect("open").presentation_cache_resident, 0);
		player.close();
	}
	Ok(())
}

#[test]
#[ignore = "requires four Vulkan decode codecs, Core profile, FFmpeg and video fixtures"]
fn native_presentation_pixels_match_independent_planar_oracle() -> oa::Result<()> {
	use oa::video::{VideoColorMatrix, VideoColorRange};
	let engine = Engine::new()?;
	let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("sdk/asset/video/clip");
	let mut fixtures = std::fs::read_dir(directory)
		.expect("video fixtures")
		.map(|entry| entry.expect("fixture entry").path())
		.filter(|path| path.extension().is_some_and(|ext| ext == "mp4"))
		.collect::<Vec<_>>();
	fixtures.sort();
	assert_eq!(fixtures.len(), 9, "cover every qualified decode profile");
	for path in fixtures {
		let wide = path.to_string_lossy().contains("10bit");
		let bits = if wide { 10 } else { 8 };
		let output = std::process::Command::new("ffmpeg")
			.args(["-v", "error", "-i"])
			.arg(&path)
			.args([
				"-frames:v",
				"1",
				"-pix_fmt",
				if wide { "yuv420p10le" } else { "yuv420p" },
				"-f",
				"rawvideo",
				"-",
			])
			.output()
			.expect("FFmpeg oracle");
		assert!(
			output.status.success(),
			"{}",
			String::from_utf8_lossy(&output.stderr)
		);
		let mut player = VideoPlayer::new_native(
			&engine,
			&path,
			VideoPlayerConfig {
				start_playing: false,
				..VideoPlayerConfig::default()
			},
		)?;
		let frame = player.current_frame()?;
		let (w, h) = (frame.width(), frame.height());
		let color = frame.color_info();
		let (kr, kb) = match color.matrix() {
			VideoColorMatrix::Bt601 => (0.299_f64, 0.114_f64),
			VideoColorMatrix::Bt2020 => (0.2627, 0.0593),
			VideoColorMatrix::Bt709 | VideoColorMatrix::Unspecified => (0.2126, 0.0722),
		};
		let full = color.range() == VideoColorRange::Full;
		let rgba = player.current_frame_texture()?.read_rgba8()?;
		assert_eq!(rgba.len(), w * h * 4);
		let samples = if wide {
			output
				.stdout
				.as_chunks::<2>()
				.0
				.iter()
				.map(|b| u16::from_le_bytes([b[0], b[1]]))
				.collect::<Vec<_>>()
		} else {
			output
				.stdout
				.iter()
				.map(|&b| u16::from(b))
				.collect::<Vec<_>>()
		};
		let chroma_w = w.div_ceil(2);
		let luma_len = w * h;
		let chroma_len = chroma_w * h.div_ceil(2);
		assert_eq!(samples.len(), luma_len + 2 * chroma_len);
		let scale = f64::from(1_u32 << (bits - 8));
		let max = f64::from((1_u32 << bits) - 1);
		let mut max_error = 0;
		for y in 0..h {
			for x in 0..w {
				let site = y / 2 * chroma_w + x / 2;
				let raw_y = f64::from(samples[y * w + x]);
				let raw_u = f64::from(samples[luma_len + site]);
				let raw_v = f64::from(samples[luma_len + chroma_len + site]);
				let luma = if full {
					raw_y / max
				} else {
					(raw_y - 16.0 * scale) / (219.0 * scale)
				};
				let divisor = if full { max } else { 224.0 * scale };
				let u = (raw_u - 128.0 * scale) / divisor;
				let v = (raw_v - 128.0 * scale) / divisor;
				let r = luma + 2.0 * (1.0 - kr) * v;
				let b = luma + 2.0 * (1.0 - kb) * u;
				let g = (luma - kr * r - kb * b) / (1.0 - kr - kb);
				let offset = (y * w + x) * 4;
				for (channel, value) in [r, g, b].into_iter().enumerate() {
					let expected = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
					max_error = max_error.max(rgba[offset + channel].abs_diff(expected));
				}
				assert_eq!(rgba[offset + 3], 255);
			}
		}
		eprintln!(
			"presentation oracle {}: max channel error={max_error}",
			path.display()
		);
		assert!(
			max_error <= 1,
			"{}: presentation max channel error {max_error}",
			path.display()
		);
		player.close();
	}
	Ok(())
}

#[test]
#[ignore = "requires Vulkan H.264 hardware and OA donor fixture"]
fn native_player_retains_decoder_frames_without_host_pixels() -> oa::Result<()> {
	let engine = Engine::new()?;
	let mut player = VideoPlayer::new_native(
		&engine,
		donor_h264_fixture(),
		VideoPlayerConfig {
			loop_playback: false,
			start_playing: false,
			..VideoPlayerConfig::default()
		},
	)?;
	assert_eq!(player.stats().unwrap().presentation_cache_capacity, 32);
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
	assert_eq!(player.stats().unwrap().presentation_cache_resident, 1);
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

#[test]
#[ignore = "requires FFmpeg x264/x265 and Vulkan H.264 and H.265 decode"]
fn cropped_avc_and_open_gop_hevc_seek_match_ffmpeg() -> oa::Result<()> {
	use std::process::Command;
	let engine = Engine::new()?;
	for (encoder, extension, pixel_format, encoder_args) in [
		("libx264", "mp4", "yuv420p", &["-g", "24"][..]),
		(
			"libx265",
			"mkv",
			"yuv420p10le",
			&[
				"-x265-params",
				"keyint=24:min-keyint=24:scenecut=0:open-gop=1:bframes=3:pools=1:frame-threads=1:log-level=error",
			][..],
		),
	] {
		let path = std::env::temp_dir().join(format!(
			"oa-{encoder}-seek-{}.{extension}",
			std::process::id()
		));
		let status = Command::new("ffmpeg")
			.args([
				"-nostdin",
				"-v",
				"error",
				"-y",
				"-f",
				"lavfi",
				"-i",
				"testsrc2=size=96x70:rate=24",
				"-t",
				"5",
				"-c:v",
				encoder,
				"-pix_fmt",
				pixel_format,
				"-threads",
				"2",
			])
			.args(encoder_args)
			.arg(&path)
			.status()
			.expect("FFmpeg available");
		assert!(status.success());
		let result = (|| -> oa::Result<()> {
			let reference = Command::new("ffmpeg")
				.args(["-nostdin", "-v", "error", "-i"])
				.arg(&path)
				.args(["-pix_fmt", pixel_format, "-f", "rawvideo", "pipe:1"])
				.output()
				.expect("FFmpeg reference");
			assert!(reference.status.success());
			let frame_size = 96 * 70 * 3 / 2 * if pixel_format == "yuv420p" { 1 } else { 2 };
			assert_eq!(reference.stdout.len(), 120 * frame_size);
			let cfg = VideoPlayerConfig {
				loop_playback: false,
				start_playing: false,
				..Default::default()
			};
			let mut player = VideoPlayer::open(&engine, &path, cfg)?;
			let mut native = VideoPlayer::new_native(&engine, &path, cfg)?;
			for index in [97, 74, 110, 24, 0, 119] {
				player.seek_frame(index)?;
				native.seek_frame(index)?;
				let frame = player.current_frame()?;
				assert_eq!(player.current_frame_index()?, index);
				let expected = &reference.stdout[index as usize * frame_size..][..frame_size];
				if let Some(bytes) = frame.as_yuv420p() {
					assert_eq!(bytes, expected, "{encoder} at {index}");
				} else {
					let samples = frame.as_yuv420p10().expect("wide planes");
					assert!(
						samples
							.iter()
							.zip(expected.as_chunks::<2>().0)
							.all(|(a, b)| *a == u16::from_le_bytes(*b)),
						"{encoder} at {index}"
					);
				}
				let texture = native.current_frame_texture()?;
				assert_eq!((texture.width(), texture.height()), (96, 70));
			}
			player.close();
			native.close();
			Ok(())
		})();
		let _ = std::fs::remove_file(&path);
		result?;
	}
	Ok(())
}
