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
