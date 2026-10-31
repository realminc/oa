use oa::{Engine, Presenter, SdlWindow};

#[test]
#[ignore = "requires a hardware Vulkan graphics device and live SDL display"]
fn presenter_blits_a_native_texture_through_a_real_swapchain() -> oa::Result<()> {
	let window = SdlWindow::new("OA Presenter qualification", 64, 64)?;
	let engine = Engine::builder()
		.instance_extensions(window.required_instance_extensions()?)
		.build()?;
	let mut presenter = Presenter::new(&engine, &window)?;

	assert!(presenter.has_graphics());
	assert!(presenter.has_present());
	assert!(presenter.supports_swapchain());
	let capabilities = presenter.capabilities()?;
	assert!(capabilities.supports_color_attachment);
	assert!(capabilities.supports_transfer_dst);
	assert!(presenter.init_swapchain(&window)?);
	assert!(presenter.swapchain_image_count() >= capabilities.min_image_count as usize);
	assert_eq!(presenter.frames_in_flight(), 2);

	let rgba = [
		255, 0, 0, 255, 0, 255, 0, 255, // row zero
		0, 0, 255, 255, 255, 255, 255, 255, // row one
	];
	let texture = oa::render::texture_from_rgba8(&engine, &rgba, 2, 2)?;
	let _suboptimal = presenter.present_texture(&texture)?;
	presenter.close_swapchain()?;
	assert!(!presenter.has_swapchain());
	Ok(())
}

#[test]
#[ignore = "requires a hardware Vulkan graphics device and live SDL display"]
fn presenter_displays_a_renderer_frame() -> oa::Result<()> {
	let window = SdlWindow::new("OA 3D viewer qualification", 64, 64)?;
	let engine = Engine::builder()
		.instance_extensions(window.required_instance_extensions()?)
		.build()?;
	let renderer = oa::render::Renderer::new(
		&engine,
		oa::render::RendererConfig {
			width: 64,
			height: 64,
			target_slot_count: 1,
			max_vertex_count: 4,
			max_index_count: 6,
			clear_color: oa::Color::new(0.0, 0.0, 0.0, 1.0),
			..Default::default()
		},
	)?;
	let mesh = oa::render::create_quad(1.0, 1.0)?;
	let camera = oa::render::Camera::new_orthographic(2.0, 2.0, 0.1, 10.0);
	let material = oa::render::FlatColorMaterial {
		color: oa::Color::new(0.0, 0.5, 1.0, 1.0),
		opacity: 1.0,
	};
	renderer.begin_frame_flat_color(&mesh, &camera, &material)?;
	let frame = renderer.submit_frame()?;

	let mut presenter = Presenter::new(&engine, &window)?;
	assert!(presenter.init_swapchain(&window)?);
	let _suboptimal = presenter.present_texture(frame.color())?;
	presenter.close_swapchain()?;
	let pixels = frame.consume_readback()?;
	assert!(
		pixels
			.as_chunks::<4>()
			.0
			.iter()
			.any(|pixel| pixel[1] > 80 && pixel[2] > 180),
		"presented Renderer frame did not contain the rendered geometry"
	);
	renderer.close()?;
	Ok(())
}

#[test]
#[ignore = "requires Vulkan presentation and a live SDL display"]
fn presenter_retains_compositor_frames_across_resize_and_ui_close() -> oa::Result<()> {
	let window = SdlWindow::new("OA compositor lifetime qualification", 64, 64)?;
	let selection = std::env::var("OA_TEST_DEVICE_INDEX")
		.ok()
		.map(|index| oa::DeviceSelection::Index(index.parse().expect("device index")))
		.unwrap_or_default();
	let engine = Engine::builder()
		.devices(selection)
		.requirements(oa::EngineRequirements::default().presentation())
		.instance_extensions(window.required_instance_extensions()?)
		.build()?;
	let mut presenter = Presenter::new(&engine, &window)?;
	assert!(presenter.init_swapchain(&window)?);
	let mut ui = oa::ui::Ui::init(&engine, 64, 64)?;
	let source = oa::render::texture_from_rgba8(&engine, &[255_u8; 16], 2, 2)?;
	for size in [64, 96, 64, 96] {
		ui.resize(size, size)?;
		ui.begin_frame(0.0);
		let clip = oa::ui::PixelRect::new(0, 0, size, size);
		let color = oa::Color::new(0.0, 1.0, 0.0, 1.0);
		ui.rect_raw(clip, color, clip, 0.0);
		ui.rect_outline_raw(clip, oa::Color::new(1.0, 0.0, 0.0, 1.0), 1, clip);
		ui.line_at([0.0, 0.0], [20.0, 20.0], color, 1.0, clip)?;
		ui.plot_line_at(&[0.0, 1.0, 0.5], clip, 0.0, 1.0, color, 1.0)?;
		ui.image_at(&source, oa::ui::PixelRect::new(0, 0, 8, 8))?;
		ui.text_at(
			"OA",
			[12.0, 24.0],
			oa::TextLayoutConfig::default(),
			color,
			clip,
		)?;
		ui.end_frame();
		let _suboptimal = ui.present_to(&mut presenter)?;
	}
	// The last two frames still own their pipelines, images and source buffers.
	ui.close()?;
	drop(source);
	presenter.close_swapchain()?;
	Ok(())
}
