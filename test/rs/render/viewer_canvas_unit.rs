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
	let texture =
		crate::render::texture_from_rgba8(&engine, &[231, 47, 13, 255].repeat(17 * 11), 17, 11)?;
	media.update_live_texture(Some(&texture), &mut nav)?;
	assert_eq!(media.content_size(), Some((17, 11)));
	assert!((nav.zoom() - 320.0 / 17.0).abs() < 0.001);
	nav.pan_by(23.0, -9.0);
	nav.keyboard_zoom_in();
	nav.update(1000.0);
	let navigation = (nav.pan_x(), nav.pan_y(), nav.zoom());
	let replacement =
		crate::render::texture_from_rgba8(&engine, &[231, 47, 13, 255].repeat(17 * 11), 17, 11)?;
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
	let resized =
		crate::render::texture_from_rgba8(&engine, &[231, 47, 13, 255].repeat(23 * 13), 23, 13)?;
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
