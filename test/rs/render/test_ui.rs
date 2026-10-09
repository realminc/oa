use oa::{
	Color,
	plot::{Figure, FigureConfig},
	ui::{PixelRect, Ui, UiEvent, UiEventKind},
};

#[test]
fn plot_figure_normalizes_config_and_clamps_cell_indices() {
	let figure = Figure::new(FigureConfig {
		rows: 0,
		cols: -4,
		h_spacing: -1,
		v_spacing: -2,
		padding: -3,
		..FigureConfig::default()
	});
	assert_eq!(figure.rows(), 1);
	assert_eq!(figure.cols(), 1);
	assert_eq!(figure.config().h_spacing, 0);
	assert_eq!(figure.config().v_spacing, 0);
	assert_eq!(figure.config().padding, 0);
	assert_eq!(
		figure.cell_rect(-8, 9, 100, 80),
		figure.cell_rect(0, 0, 100, 80)
	);
}

test_vk!(ui_compute_compositor_draws_panel_and_line, engine, {
	let mut ui = Ui::init(&engine, 64, 48)?;
	ui.begin_frame(16.0);
	ui.begin_panel(
		"panel",
		PixelRect {
			x: 8,
			y: 8,
			w: 48,
			h: 32,
		},
	);
	ui.end_panel();
	ui.plot_line_at(
		&[0.0, 1.0, 0.25, 0.75],
		PixelRect {
			x: 10,
			y: 10,
			w: 44,
			h: 28,
		},
		0.0,
		1.0,
		Color::new(1.0, 0.0, 0.0, 1.0),
		2.0,
	)?;
	ui.end_frame();
	let pixels = ui.render_rgba8()?;
	assert_eq!(pixels.len(), 64 * 48 * 4);
	let background = &pixels[0..4];
	assert!(background[0] < 32 && background[1] < 32 && background[2] < 32);
	assert!(pixels.as_chunks::<4>().0.iter().any(|pixel| {
		pixel[0] > 180
			&& pixel[0] > pixel[1].saturating_add(80)
			&& pixel[0] > pixel[2].saturating_add(80)
	}));
	ui.close()?;
	Ok(())
});

test_vk!(ui_compute_compositor_draws_hinted_glyphs, engine, {
	let mut ui = Ui::init(&engine, 96, 48)?;
	ui.begin_frame(16.0);
	ui.text_at(
		"OA",
		[10.0, 30.0],
		oa::TextLayoutConfig {
			size: 24.0,
			..Default::default()
		},
		Color::new(1.0, 1.0, 1.0, 1.0),
		PixelRect::new(0, 0, 96, 48),
	)?;
	ui.end_frame();
	let pixels = ui.render_rgba8()?;
	assert!(
		pixels
			.as_chunks::<4>()
			.0
			.iter()
			.any(|pixel| { pixel[0] > 180 && pixel[1] > 180 && pixel[2] > 180 })
	);
	// The compose image remains reusable after its TRANSFER_SRC readback state.
	ui.begin_frame(16.0);
	ui.text_at(
		"OA",
		[10.0, 30.0],
		oa::TextLayoutConfig {
			size: 24.0,
			..Default::default()
		},
		Color::new(1.0, 1.0, 1.0, 1.0),
		PixelRect::new(0, 0, 96, 48),
	)?;
	ui.end_frame();
	assert_eq!(ui.render_rgba8()?.len(), 96 * 48 * 4);
	ui.close()?;
	Ok(())
});

test_vk!(
	ui_pointer_release_survives_until_widget_evaluation,
	engine,
	{
		let mut ui = Ui::init(&engine, 96, 48)?;
		ui.route_event(&UiEvent {
			kind: UiEventKind::MouseMove,
			mouse_x: 8.0,
			mouse_y: 8.0,
			..UiEvent::default()
		});
		ui.begin_frame(0.0);
		ui.begin_panel("panel", PixelRect::new(0, 0, 96, 48));
		assert!(!ui.button("Run"));
		ui.end_panel();
		ui.end_frame();

		let down = UiEvent {
			kind: UiEventKind::MouseDown,
			mouse_x: 8.0,
			mouse_y: 8.0,
			button: 1,
			..UiEvent::default()
		};
		let up = UiEvent {
			kind: UiEventKind::MouseUp,
			mouse_x: 8.0,
			mouse_y: 8.0,
			button: 1,
			..UiEvent::default()
		};
		ui.route_event(&down);
		ui.route_event(&up);
		ui.begin_frame(0.0);
		ui.begin_panel("panel", PixelRect::new(0, 0, 96, 48));
		assert!(ui.button("Run"));
		ui.end_panel();
		ui.end_frame();
		ui.close()?;
		Ok(())
	}
);

test_vk!(plot_figure_line_sink_is_not_a_flat_placeholder, engine, {
	let mut figure = oa::plot::Figure::new(oa::plot::FigureConfig {
		width: 96,
		height: 64,
		padding: 8,
		..Default::default()
	});
	figure.ax(0, 0).plot(
		&[0.0, 1.0, 0.0, 1.0],
		oa::plot::LineStyle {
			color: Color::new(1.0, 0.0, 0.0, 1.0),
			width: 2.0,
			..Default::default()
		},
	);
	let image = figure.render(&engine)?;
	let values = image.as_matrix().read_f32()?;
	let plane = 96 * 64;
	assert!(
		values[..plane]
			.iter()
			.zip(&values[plane..plane * 2])
			.any(|(red, green)| *red > 0.7 && *red > *green + 0.3)
	);
	Ok(())
});
