use super::*;

#[test]
#[ignore = "requires Vulkan texture upload and UI raster composition"]
fn live_canvas_preserves_navigation_and_uses_shared_image_channels() -> Result<()> {
	let engine = Engine::builder().build()?;
	let config = ViewerConfig {
		mode: ViewerMode::Live,
		show_stats: false,
		show_help: false,
		..Default::default()
	};
	let mut media = MediaState::open(&engine, &config, None)?;
	let mut nav = Navigation::new();
	nav.set_viewport_size(320.0, 240.0)?;
	let texture = crate::render::texture_from_rgba8(&[231, 47, 13, 255].repeat(17 * 11), 17, 11)?;
	media.update_live_texture(Some(&texture), &mut nav)?;
	assert_eq!(media.content_size(), Some((17, 11)));
	assert!((nav.zoom() - 320.0 / 17.0).abs() < 0.001);
	nav.pan_by(23.0, -9.0);
	nav.keyboard_zoom_in();
	nav.update(1000.0);
	let navigation = (nav.pan_x(), nav.pan_y(), nav.zoom());
	let replacement = crate::render::texture_from_rgba8(&[231, 47, 13, 255].repeat(17 * 11), 17, 11)?;
	media.update_live_texture(Some(&replacement), &mut nav)?;
	assert_eq!(
		(nav.pan_x(), nav.pan_y(), nav.zoom()),
		navigation,
		"new live frames reset the camera"
	);
	media.handle_event(
		&crate::input::InputEvent {
			kind: InputEventKind::KeyDown,
			key: config.key_red,
			..Default::default()
		},
		&config,
	)?;
	let resized = crate::render::texture_from_rgba8(&[231, 47, 13, 255].repeat(23 * 13), 23, 13)?;
	media.update_live_texture(Some(&resized), &mut nav)?;
	assert_eq!(media.content_size(), Some((23, 13)));
	assert!((nav.zoom() - 320.0 / 23.0).abs() < 0.001);
	assert!(matches!(
		&media,
		MediaState::Live {
			channel_mode: image_mode::ChannelMode::Red,
			..
		}
	));
	let mut ui = Ui::init(&engine, 320, 240)?;
	ui.begin_frame(16.0);
	render_frame(
		&mut ui,
		&mut media,
		&mut nav,
		&config,
		320,
		240,
		1.0,
		false,
		&mut timeline::TransportState::default(),
	)?;
	ui.end_frame();
	let pixels = ui.render_rgba8()?;
	let center = &pixels[(120 * 320 + 160) * 4..(120 * 320 + 160) * 4 + 4];
	for &channel in &center[..3] {
		assert!(channel.abs_diff(231) <= 1, "{center:?}");
	}
	assert_eq!(center[3], 255);
	ui.close()?;
	Ok(())
}

#[test]
#[ignore = "requires 10-bit Vulkan Video and UI raster composition"]
fn yuv_canvas_matches_rgba_and_keeps_reverse_history() -> Result<()> {
	use crate::video::{VideoPlayer, VideoPlayerConfig};

	let engine = Engine::builder()
		.requirements(crate::EngineRequirements::default().graphics())
		.build()?;
	let mut ui = Ui::init(&engine, 1280, 720)?;
	let fixtures = [
		"shibuya_720p_30fps_h265_main10_10bit_420.mp4",
		"shibuya_720p_30fps_av1_main_10bit_420.mp4",
		"shibuya_720p_30fps_vp9_profile2_10bit_420.mp4",
	];
	for fixture in fixtures {
		let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
			.join("sdk/asset/video/clip")
			.join(fixture);
		let config = VideoPlayerConfig::default();
		let mut reference = VideoPlayer::new_native(&engine, &path, config)?;
		let mut direct = VideoPlayer::new_native(&engine, &path, config)?;
		direct.enable_canvas_yuv()?;
		let geometry = [
			PixelRect::new(0, 0, 1280, 720),
			PixelRect::new(-23, 31, 997, 557),
			PixelRect::new(-63, -17, 1589, 911),
		];
		let mut first = None;
		for dst in geometry {
			let expected_texture = reference.current_frame_texture()?;
			ui.begin_frame(0.0);
			ui.image_at(&expected_texture, dst)?;
			ui.end_frame();
			let expected = ui.render_rgba8()?;

			let staged = direct.prepare_viewer_frame()?;
			assert!(staged.canvas_planes().is_some(), "{fixture}");
			ui.begin_frame(0.0);
			ui.video_planes_at(&staged, dst)?;
			ui.end_frame();
			let actual = ui.render_rgba8()?;
			assert_eq!(actual.len(), expected.len());
			let largest_error = actual
				.into_iter()
				.zip(expected)
				.map(|(a, b)| a.abs_diff(b))
				.max();
			assert!(
				largest_error <= Some(1),
				"{fixture} at {dst:?}: {largest_error:?}"
			);
			first.get_or_insert(staged);
		}
		let index = direct.current_frame_index()?;
		direct.step_frames(1)?;
		direct.prepare_viewer_frame()?;
		direct.step_frames(-1)?;
		assert_eq!(direct.current_frame_index()?, index);
		let cached = direct.prepare_viewer_frame()?;
		let (cached_planes, _) = cached.canvas_planes().expect("cached YUV frame");
		let (first_planes, _) = first.as_ref().unwrap().canvas_planes().unwrap();
		assert!(cached_planes.same_as(first_planes), "{fixture}");
	}
	Ok(())
}

#[test]
#[ignore = "fixed-clock offscreen canvas throughput experiment"]
fn measure_yuv_canvas_throughput() -> Result<()> {
	use crate::video::{VideoPlayer, VideoPlayerConfig};
	use std::{path::Path, time::Instant};

	let engine = Engine::builder()
		.requirements(crate::EngineRequirements::default().graphics())
		.build()?;
	let mut ui = Ui::init(&engine, 1280, 720)?;
	let dst = PixelRect::new(0, 0, 1280, 720);
	let mut measure = |path: &Path, trial: bool| -> Result<f64> {
		let started = Instant::now();
		let mut player = VideoPlayer::new_native(&engine, path, VideoPlayerConfig::default())?;
		if trial {
			player.enable_canvas_yuv()?;
		}
		for index in 0..60 {
			let frame = player.prepare_viewer_frame()?;
			assert_eq!(frame.canvas_planes().is_some(), trial);
			assert_eq!(player.current_frame_index()?, index);
			ui.begin_frame(0.0);
			if let Some(texture) = frame.as_texture() {
				ui.image_at(texture, dst)?;
			} else {
				ui.video_planes_at(&frame, dst)?;
			}
			ui.end_frame();
			ui.render_rgba8()?;
			if index < 59 {
				player.step_frames(1)?;
			}
		}
		player.close();
		Ok(started.elapsed().as_secs_f64() * 1000.0)
	};

	for codec in ["h265_main10", "av1_main", "vp9_profile2"] {
		let path = Path::new(env!("CARGO_MANIFEST_DIR"))
			.join("sdk/asset/video/clip")
			.join(format!("shibuya_720p_30fps_{codec}_10bit_420.mp4"));
		measure(&path, false)?;
		measure(&path, true)?;
		let mut baseline = Vec::new();
		let mut trial = Vec::new();
		for pair in 0..5 {
			for direct in [pair % 2 != 0, pair % 2 == 0] {
				let elapsed = measure(&path, direct)?;
				if direct {
					trial.push(elapsed);
				} else {
					baseline.push(elapsed);
				}
			}
		}
		for (mode, mut times) in [("texture", baseline), ("yuv_canvas", trial)] {
			times.sort_by(f64::total_cmp);
			let median = times[2];
			let mut deviations: Vec<_> = times.iter().map(|time| (time - median).abs()).collect();
			deviations.sort_by(f64::total_cmp);
			println!(
				"OA_CANVAS_BENCH codec={codec} mode={mode} median_ms={median:.3} mad_ms={:.3} fps={:.1}",
				deviations[2],
				60000.0 / median
			);
		}
	}
	Ok(())
}
