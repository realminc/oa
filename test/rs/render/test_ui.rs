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

// A deferred texture producer must reach composition without a caller wait.
// Reuse and resize exercise fresh bounded descriptors and prior image layouts.
test_vk!(
	ui_compositor_observes_deferred_texture_and_resizes,
	engine,
	{
		let source = oa::render::texture_from_rgba8(&engine, &[0_u8; 4 * 3 * 4], 4, 3)?;
		let red = oa::render::clear(&source, Color::new(1.0, 0.0, 0.0, 1.0))?;
		let mut ui = Ui::init(&engine, 4, 3)?;
		for (width, height) in [(4, 3), (9, 7), (4, 3)] {
			ui.resize(width, height)?;
			ui.begin_frame(0.0);
			ui.image_at(&red, PixelRect::new(0, 0, width, height))?;
			ui.end_frame();
			let pixels = ui.render_rgba8()?;
			assert_eq!(pixels.len(), width as usize * height as usize * 4);
			assert!(
				pixels
					.as_chunks::<4>()
					.0
					.iter()
					.all(|pixel| *pixel == [255, 0, 0, 255])
			);
		}
		ui.close()?;
		Ok(())
	}
);

// Donor drawCanvasGrid.slang: independent host pixel oracle, including all
// tiers, off-center axes, clipping, alpha composition, gradient and falloff.
test_vk!(ui_canvas_grid_matches_donor_pixels, engine, {
	use oa::ui::UiGridConfig;
	let mut ui = Ui::init(&engine, 43, 31)?;
	let base = Color::new_from_u32(0x181818FF);
	ui.push_style(oa::ui::UiStyle {
		background: base,
		..Default::default()
	});
	let config = UiGridConfig {
		origin: [13.25, 10.75],
		minor_spacing: [4.5, 6.5],
		major_every: 2,
		super_major_every: 6,
		background_top: Color::new_from_u32(0x1F1F1FA6),
		background_bottom: Color::new_from_u32(0x181818A6),
		minor_color: Color::new_from_u32(0xFFFFFF0B),
		major_color: Color::new_from_u32(0xC7CCD129),
		super_major_color: Color::new_from_u32(0x0808089E),
		axis_color: Color::new_from_u32(0x000000D1),
		guide_fade_center: [13.25, 10.75],
		guide_fade_inner_radius: 5.0,
		guide_fade_outer_radius: 15.0,
		radial_guide_fade: true,
		dither_background: true,
		..Default::default()
	};
	ui.begin_frame(0.0);
	ui.grid(PixelRect::new(3, 2, 37, 27), config.clone());
	ui.end_frame();
	let pixels = ui.render_rgba8()?;
	let smooth = |a: f32, b: f32, x: f32| {
		let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
		t * t * (3.0 - 2.0 * t)
	};
	for y in 0..31_u32 {
		for x in 0..43_u32 {
			let mut expected = base.to_array();
			if (3..40).contains(&x) && (2..29).contains(&y) {
				let mut background = Color::lerp(
					config.background_top,
					config.background_bottom,
					(y - 2) as f32 / 26.0,
				)
				.to_array();
				let mut noise = x.wrapping_add(y.wrapping_mul(65537));
				noise = (noise ^ 61) ^ (noise >> 16);
				noise = noise.wrapping_mul(9);
				noise ^= noise >> 4;
				noise = noise.wrapping_mul(0x27d4eb2d);
				noise ^= noise >> 15;
				for v in &mut background[..3] {
					*v = (*v + ((noise & 255) as f32 / 255.0 - 0.5) * (2.0 / 255.0)).clamp(0.0, 1.0);
				}
				for (value, channel) in expected[..3].iter_mut().zip(&background[..3]) {
					*value = channel * background[3] + *value * (1.0 - background[3]);
				}
				let dx = x as f32 + 0.5 - config.origin[0];
				let dy = y as f32 + 0.5 - config.origin[1];
				let fade = 1.0 - smooth(5.0, 15.0, dx.hypot(dy));
				let periodic = |value: f32, spacing: f32| {
					let phase = value / spacing + 0.5;
					(phase - phase.floor() - 0.5).abs() * spacing
				};
				for (tier, thickness, color) in [
					(1.0, config.minor_thickness, config.minor_color),
					(2.0, config.major_thickness, config.major_color),
					(6.0, config.super_major_thickness, config.super_major_color),
					(0.0, config.axis_thickness, config.axis_color),
				] {
					let distance = if tier == 0.0 {
						dx.abs().min(dy.abs())
					} else {
						periodic(dx, config.minor_spacing[0] * tier)
							.min(periodic(dy, config.minor_spacing[1] * tier))
					};
					let alpha = color.a
						* fade * (1.0
						- smooth(
							(thickness * 0.5 - 0.75).max(0.0),
							thickness * 0.5 + 0.75,
							distance,
						));
					for (value, channel) in expected[..3].iter_mut().zip([color.r, color.g, color.b]) {
						*value = channel * alpha + *value * (1.0 - alpha);
					}
				}
			}
			for (channel, value) in expected.iter().enumerate() {
				let actual = pixels[((y * 43 + x) * 4) as usize + channel];
				let target = (value * 255.0).round() as u8;
				assert!(
					actual.abs_diff(target) <= 1,
					"pixel ({x},{y}) channel {channel}: {actual} != {target}"
				);
			}
		}
	}
	// Flags disable guides independently of the background; a negatively placed
	// rectangle is clipped safely and leaves every outside pixel unchanged.
	ui.begin_frame(0.0);
	ui.grid(
		PixelRect::new(-3, -2, 13, 9),
		UiGridConfig {
			background_top: Color::new_from_u32(0x283848FF),
			background_bottom: Color::new_from_u32(0x283848FF),
			draw_grid: false,
			draw_axes: false,
			..Default::default()
		},
	);
	ui.end_frame();
	let flat = ui.render_rgba8()?;
	for y in 0..31 {
		for x in 0..43 {
			let expected = if x < 10 && y < 7 {
				[40, 56, 72, 255]
			} else {
				[24, 24, 24, 255]
			};
			assert_eq!(&flat[(y * 43 + x) * 4..(y * 43 + x + 1) * 4], &expected);
		}
	}
	// Reject malformed tier relationships before any shader can execute them.
	ui.begin_frame(0.0);
	ui.grid(
		PixelRect::new(0, 0, 43, 31),
		UiGridConfig {
			major_every: 0,
			..Default::default()
		},
	);
	ui.end_frame();
	assert_eq!(
		ui.render_rgba8().unwrap_err().kind(),
		oa::ErrorKind::InvalidArgument
	);
	ui.close()?;
	Ok(())
});

// Covers the new donor source-buffer paths and their frame-owned lifetime.
test_vk!(ui_donor_audio_matrix_pixels_and_retention, engine, {
	let mut ui = Ui::init(&engine, 32, 24)?;
	let base = Color::new_from_u32(0x000000FF);
	ui.push_style(oa::ui::UiStyle {
		background: base,
		..Default::default()
	});
	ui.begin_frame(0.0);
	let values = oa::Matrix::new_from_slice(&engine, vec![2, 2], &[0.0_f32, 1.0, 0.5, 0.25])?;
	// Grayscale gives an independent exact range/row/column pixel oracle.
	ui.heatmap_matrix_at(&values, PixelRect::new(2, 3, 8, 6), [0.0, 1.0], 3)?;
	drop(values);
	let envelope = oa::Matrix::new_from_slice(&engine, vec![2, 2], &[-1.0_f32, 1.0, -0.5, 0.5])?;
	ui.waveform_matrix_at(
		&envelope,
		PixelRect::new(16, 3, 8, 9),
		0.5,
		Color::new_from_u32(0xFF0000FF),
		Color::new_from_u32(0x00FF00FF),
	)?;
	drop(envelope);
	let wrong_dtype = oa::Matrix::new_from_slice(&engine, vec![2, 2], &[0_u32; 4])?;
	assert!(
		ui.heatmap_matrix_at(&wrong_dtype, PixelRect::new(0, 0, 4, 4), [0.0, 1.0], 3)
			.is_err()
	);
	assert!(
		ui.waveform_matrix_at(&wrong_dtype, PixelRect::new(0, 0, 4, 4), 0.5, base, base)
			.is_err()
	);
	let valid = oa::Matrix::new_from_slice(&engine, vec![2, 2], &[0.0_f32; 4])?;
	assert!(
		ui.heatmap_matrix_at(&valid, PixelRect::new(0, 0, 33, 4), [0.0, 1.0], 3)
			.is_err()
	);
	assert!(
		ui.heatmap_matrix_at(&valid, PixelRect::new(0, 0, 4, 4), [1.0, 0.0], 3)
			.is_err()
	);
	ui.end_frame();
	let pixels = ui.render_rgba8()?;
	let pixel = |x: usize, y: usize| &pixels[(y * 32 + x) * 4..(y * 32 + x + 1) * 4];
	for (x, y, expected) in [(2, 3, 0_i16), (9, 3, 255), (2, 8, 128), (9, 8, 64)] {
		for value in &pixel(x, y)[..3] {
			assert!((i16::from(*value) - expected).abs() <= 1);
		}
		assert_eq!(pixel(x, y)[3], 255);
	}
	assert_eq!(pixel(16, 3), &[255, 0, 0, 255]);
	assert_eq!(pixel(19, 11), &[255, 0, 0, 255]);
	assert_eq!(pixel(20, 5), &[0, 255, 0, 255]);
	assert_eq!(pixel(23, 9), &[0, 255, 0, 255]);
	assert_eq!(pixel(20, 3), &[0, 0, 0, 255]);
	ui.close()?;
	Ok(())
});

test_vk!(ui_donor_icons_release_and_rounded_mask, engine, {
	use oa::ui::{UiIcon, UiIconButtonStyle};
	let mut ui = Ui::init(&engine, 64, 48)?;
	ui.push_style(oa::ui::UiStyle {
		background: Color::new_from_u32(0x000000FF),
		..Default::default()
	});
	let rect = PixelRect::new(16, 8, 32, 32);
	let draw =
		|ui: &mut Ui<'_>| ui.icon_button("play", rect, UiIcon::Play, true, UiIconButtonStyle::Overlay);
	ui.begin_frame(0.0);
	assert!(!draw(&mut ui));
	ui.mask_rounded_rect(PixelRect::new(0, 0, 64, 48), 15.0, Color::new_from_u32(0))?;
	ui.end_frame();
	let pixels = ui.render_rgba8()?;
	assert_eq!(&pixels[..4], &[0, 0, 0, 0]);
	let center = &pixels[(24 * 64 + 32) * 4..(24 * 64 + 32) * 4 + 4];
	assert!(center[..3].iter().all(|value| *value > 220));
	assert_eq!(center[3], 255);
	ui.route_event(&UiEvent {
		kind: UiEventKind::MouseDown,
		button: 1,
		mouse_x: 32.0,
		mouse_y: 24.0,
		..Default::default()
	});
	ui.begin_frame(16.0);
	assert!(!draw(&mut ui));
	ui.end_frame();
	ui.route_event(&UiEvent {
		kind: UiEventKind::MouseUp,
		button: 1,
		mouse_x: 1.0,
		mouse_y: 1.0,
		..Default::default()
	});
	ui.begin_frame(16.0);
	assert!(!draw(&mut ui));
	ui.end_frame();
	// A subsequent unrelated release inside cannot resurrect the old press.
	ui.route_event(&UiEvent {
		kind: UiEventKind::MouseUp,
		button: 1,
		mouse_x: 32.0,
		mouse_y: 24.0,
		..Default::default()
	});
	ui.begin_frame(16.0);
	assert!(!draw(&mut ui));
	ui.end_frame();
	ui.route_event(&UiEvent {
		kind: UiEventKind::MouseDown,
		button: 1,
		mouse_x: 32.0,
		mouse_y: 24.0,
		..Default::default()
	});
	ui.begin_frame(16.0);
	assert!(!draw(&mut ui));
	ui.end_frame();
	ui.route_event(&UiEvent {
		kind: UiEventKind::MouseUp,
		button: 1,
		mouse_x: 32.0,
		mouse_y: 24.0,
		..Default::default()
	});
	ui.begin_frame(16.0);
	assert!(draw(&mut ui));
	ui.end_frame();
	ui.close()?;
	Ok(())
});

// A popup at the lower-right corner must flip above its anchor and stay inside.
test_vk!(ui_donor_popup_stays_inside_viewport, engine, {
	let mut ui = Ui::init(&engine, 64, 48)?;
	ui.push_style(oa::ui::UiStyle {
		background: Color::new_from_u32(0x000000FF),
		..Default::default()
	});
	ui.begin_frame(0.0);
	// Different popup/clear colors make actual placement observable.
	ui.push_style(oa::ui::UiStyle {
		background: Color::new_from_u32(0x101010FF),
		..Default::default()
	});
	ui.open_popup_at("menu", PixelRect::new(58, 40, 4, 4));
	assert!(ui.begin_popup(
		"menu",
		oa::ui::UiPopupConfig {
			width: 24.0,
			height: 20.0,
			gap: 2.0,
			padding: 0.0
		}
	));
	ui.end_popup();
	ui.pop_style();
	ui.end_frame();
	let pixels = ui.render_rgba8()?;
	let pixel = |x: usize, y: usize| &pixels[(y * 64 + x) * 4..(y * 64 + x + 1) * 4];
	assert_eq!(pixel(40, 22), &[16, 16, 16, 255]);
	assert_eq!(pixel(61, 22), &[0, 0, 0, 255]);
	assert_eq!(pixel(40, 39), &[0, 0, 0, 255]);
	ui.close()?;
	Ok(())
});
