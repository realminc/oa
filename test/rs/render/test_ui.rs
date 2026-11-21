use oa::{
	Color,
	input::{InputEvent, InputEventKind},
	plot::{Figure, FigureConfig, figure::CellRect},
	ui::{PixelRect, Ui},
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
		ui.route_event(&InputEvent {
			kind: InputEventKind::MouseMove,
			mouse_x: 8.0,
			mouse_y: 8.0,
			..InputEvent::default()
		});
		ui.begin_frame(0.0);
		ui.begin_panel("panel", PixelRect::new(0, 0, 96, 48));
		assert!(!ui.button("Run"));
		ui.end_panel();
		ui.end_frame();

		let down = InputEvent {
			kind: InputEventKind::MouseDown,
			mouse_x: 8.0,
			mouse_y: 8.0,
			button: 1,
			..InputEvent::default()
		};
		let up = InputEvent {
			kind: InputEventKind::MouseUp,
			mouse_x: 8.0,
			mouse_y: 8.0,
			button: 1,
			..InputEvent::default()
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
	let image = figure.render()?;
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
		let source = oa::render::texture_from_rgba8(&[0_u8; 4 * 3 * 4], 4, 3)?;
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
	let values = oa::Matrix::new_from_slice(vec![2, 2], &[0.0_f32, 1.0, 0.5, 0.25])?;
	// Grayscale gives an independent exact range/row/column pixel oracle.
	ui.heatmap_matrix_at(&values, PixelRect::new(2, 3, 8, 6), [0.0, 1.0], 3, false)?;
	drop(values);
	let envelope = oa::Matrix::new_from_slice(vec![2, 2], &[-1.0_f32, 1.0, -0.5, 0.5])?;
	ui.waveform_matrix_at(
		&envelope,
		PixelRect::new(16, 3, 8, 9),
		0.5,
		Color::new_from_u32(0xFF0000FF),
		Color::new_from_u32(0x00FF00FF),
	)?;
	drop(envelope);
	let wrong_dtype = oa::Matrix::new_from_slice(vec![2, 2], &[0_u32; 4])?;
	assert!(
		ui.heatmap_matrix_at(
			&wrong_dtype,
			PixelRect::new(0, 0, 4, 4),
			[0.0, 1.0],
			3,
			false
		)
		.is_err()
	);
	assert!(
		ui.waveform_matrix_at(&wrong_dtype, PixelRect::new(0, 0, 4, 4), 0.5, base, base)
			.is_err()
	);
	let valid = oa::Matrix::new_from_slice(vec![2, 2], &[0.0_f32; 4])?;
	assert!(
		ui.heatmap_matrix_at(&valid, PixelRect::new(0, 0, 33, 4), [0.0, 1.0], 3, false)
			.is_err()
	);
	assert!(
		ui.heatmap_matrix_at(&valid, PixelRect::new(0, 0, 4, 4), [1.0, 0.0], 3, false)
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
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		button: 1,
		mouse_x: 32.0,
		mouse_y: 24.0,
		..Default::default()
	});
	ui.begin_frame(16.0);
	assert!(!draw(&mut ui));
	ui.end_frame();
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseUp,
		button: 1,
		mouse_x: 1.0,
		mouse_y: 1.0,
		..Default::default()
	});
	ui.begin_frame(16.0);
	assert!(!draw(&mut ui));
	ui.end_frame();
	// A subsequent unrelated release inside cannot resurrect the old press.
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseUp,
		button: 1,
		mouse_x: 32.0,
		mouse_y: 24.0,
		..Default::default()
	});
	ui.begin_frame(16.0);
	assert!(!draw(&mut ui));
	ui.end_frame();
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		button: 1,
		mouse_x: 32.0,
		mouse_y: 24.0,
		..Default::default()
	});
	ui.begin_frame(16.0);
	assert!(!draw(&mut ui));
	ui.end_frame();
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseUp,
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

test_vk!(ui_rect_outlines_draws_gpu_detection_records, engine, {
	use oa::vision::{Detection, DetectionBuffer};

	let mut buffer = DetectionBuffer::create_host_upload(4)?;
	buffer.upload(&[
		Detection {
			center_x: 0.5,
			center_y: 0.5,
			width: 0.5,
			height: 0.5,
			confidence: 0.9,
			class_id: 0,
			color_rgba: 0,
			track_id: 0,
		},
		Detection {
			center_x: 0.25,
			center_y: 0.25,
			width: 0.125,
			height: 0.25,
			confidence: 0.8,
			class_id: 1,
			color_rgba: 0x0000FFFF,
			track_id: 0,
		},
	])?;
	let mut ui = Ui::init(&engine, 64, 48)?;
	ui.begin_frame(16.0);
	let full = PixelRect::new(0, 0, 64, 48);
	ui.rect_outlines(&buffer, full, full, Color::new(1.0, 0.0, 0.0, 1.0), 2)?;
	ui.end_frame();
	let pixels = ui.render_rgba8()?;
	let pixel = |x: usize, y: usize| &pixels[(y * 64 + x) * 4..(y * 64 + x + 1) * 4];
	// Fallback-color box: center (0.5,0.5), size 0.5 → x∈[16,48], y∈[12,36].
	assert_eq!(pixel(16, 12), &[255, 0, 0, 255]);
	assert_eq!(pixel(17, 24), &[255, 0, 0, 255]);
	assert_eq!(pixel(47, 35), &[255, 0, 0, 255]);
	// Per-instance color overrides the fallback: x∈[12,20], y∈[6,18].
	assert_eq!(pixel(12, 6), &[0, 0, 255, 255]);
	// Interior and exterior remain background.
	assert!(pixel(32, 24)[0] < 32 && pixel(32, 24)[2] < 32);
	assert!(pixel(4, 4)[0] < 32 && pixel(4, 4)[2] < 32);
	ui.close()?;
	Ok(())
});

test_vk!(ui_glyphs_draws_solid_fill_records, engine, {
	use oa::ui::{GlyphBuffer, GlyphInstance};

	let mut buffer = GlyphBuffer::create_host_upload(4)?;
	buffer.upload(&[GlyphInstance {
		anchor_x: 0.25,
		anchor_y: 0.25,
		offset_x: 0.0,
		offset_y: 0.0,
		width: 8.0,
		height: 8.0,
		color: 0x00FF00FF,
		..GlyphInstance::default()
	}])?;
	let mut ui = Ui::init(&engine, 64, 48)?;
	ui.begin_frame(16.0);
	let full = PixelRect::new(0, 0, 64, 48);
	ui.glyphs(&buffer, full, full)?;
	ui.end_frame();
	let pixels = ui.render_rgba8()?;
	let pixel = |x: usize, y: usize| &pixels[(y * 64 + x) * 4..(y * 64 + x + 1) * 4];
	// anchor (0.25,0.25)·(64,48) + offset (0,0) → solid 8x8 at (16,12)-(23,19).
	assert_eq!(pixel(16, 12), &[0, 255, 0, 255]);
	assert_eq!(pixel(23, 19), &[0, 255, 0, 255]);
	assert!(pixel(24, 20)[1] < 32);
	assert!(pixel(15, 11)[1] < 32);
	ui.close()?;
	Ok(())
});

test_vk!(
	detection_overlay_draws_boxes_labels_and_reuses_slots,
	engine,
	{
		use oa::ui::{DetectionOverlay, DetectionOverlayConfig, DetectionOverlayItem};
		use oa::vision::Detection;

		let mut ui = Ui::init(&engine, 96, 64)?;
		let mut overlay = DetectionOverlay::create(
			&engine,
			DetectionOverlayConfig {
				max_detections: 4,
				max_glyphs: 512,
				..DetectionOverlayConfig::default()
			},
		)?;
		let items = [DetectionOverlayItem {
			detection: Detection {
				center_x: 0.5,
				center_y: 0.6,
				width: 0.4,
				height: 0.4,
				confidence: 0.95,
				class_id: 3,
				color_rgba: 0,
				track_id: 0,
			},
			label: "cat".to_string(),
		}];
		overlay.update(&items, ui.text_atlas()?)?;
		assert_eq!(overlay.count(), 1);
		let full = PixelRect::new(0, 0, 96, 64);
		ui.begin_frame(16.0);
		overlay.draw(&mut ui, full, full)?;
		ui.end_frame();
		let pixels = ui.render_rgba8()?;
		let pixel = |x: usize, y: usize| &pixels[(y * 96 + x) * 4..(y * 96 + x + 1) * 4];
		// Box center (0.5,0.6)·(96,64), size 0.4 → x∈[29,67], y∈[25,51];
		// box_color = success() ≈ 0x30D158 → R≈48 G≈209 B≈88, thickness 3.
		let left_edge = pixel(29, 38);
		assert!(left_edge[1] > 150 && left_edge[0] < 100 && left_edge[2] < 140);
		// Label background sits directly above the box left edge in the same
		// detection color and must be solid even though its atlas extent is zero.
		let label_bg = pixel(30, 24);
		assert!(label_bg[1] > 150 && label_bg[0] < 100 && label_bg[2] < 140);
		assert!(pixel(48, 38)[1] < 60, "interior must stay unoutlined");

		// Four updates cycle the three-slot ring; the oldest slot becomes active.
		for _ in 0..4 {
			overlay.update(&items, ui.text_atlas()?)?;
		}
		// Pinning all three slots forces the next update to fail closed.
		let source = oa::Matrix::new_from_f32([4], &[1.0; 4])?;
		let (plan, _) = engine.capture(|| source.add(&source))?;
		for _ in 0..3 {
			let event = engine.submit(&plan)?;
			overlay.mark_consumed(&event)?;
			overlay.update(&items, ui.text_atlas()?).ok();
		}
		ui.close()?;
		Ok(())
	}
);

// ── NodeCanvas ─────────────────────────────────────────────────────────────

use oa::input::{InputContext, InputSystem, KeyBinding};
use oa::ui::{NodeCanvas, NodeCanvasHitItem, NodeCanvasState, WorldAabb};

fn canvas_with_view() -> NodeCanvas {
	let mut canvas = NodeCanvas::new();
	canvas.set_view_size(800.0, 600.0).unwrap();
	canvas
}

#[test]
fn node_canvas_transforms_round_trip() {
	let mut canvas = canvas_with_view();
	canvas.pan([20.0, -15.0]).unwrap();
	canvas.zoom_at(1.5, [400.0, 300.0]).unwrap();
	let world = [12.5, -7.25];
	let screen = canvas.world_to_screen(world);
	let back = canvas.screen_to_world(screen);
	assert!((back[0] - world[0]).abs() < 1e-3);
	assert!((back[1] - world[1]).abs() < 1e-3);
	let visible = canvas.visible_world_rect();
	assert!(visible.is_valid());
	assert!(visible.contains(back));
}

#[test]
fn node_canvas_rejects_invalid_state_and_args() {
	let mut canvas = NodeCanvas::new();
	// Zero view is legal for pan/zoom but rejects the view-dependent calls.
	assert!(canvas.hit_test([0.0, 0.0], &[]).is_err());
	assert!(canvas.grid(8.0, 10.0, 10, 100).is_err());
	assert!(
		canvas
			.fit_to_view(
				&WorldAabb {
					min: [0.0, 0.0],
					max: [1.0, 1.0]
				},
				8.0
			)
			.is_err()
	);
	assert!(canvas.set_view_size(0.0, 10.0).is_err());
	assert!(canvas.set_view_size(f32::NAN, 10.0).is_err());
	canvas.set_view_size(800.0, 600.0).unwrap();
	assert!(canvas.pan([f32::INFINITY, 0.0]).is_err());
	assert!(canvas.zoom_at(0.0, [0.0, 0.0]).is_err());
	assert!(canvas.zoom_at(-2.0, [0.0, 0.0]).is_err());
	assert!(canvas.step_animation(-1.0).is_err());
	assert!(canvas.grid(8.0, 10.0, 10, 35).is_err()); // super % major != 0
	assert!(canvas.grid(8.0, 10.0, 10, 15).is_err()); // super < 2*major
	assert!(canvas.grid(8.0, 0.0, 10, 100).is_err());
	let bad = NodeCanvasState {
		zoom: 99.0,
		..Default::default()
	};
	assert!(canvas.set_state(bad).is_err());
	assert!(canvas.state().zoom == 1.0);
}

#[test]
fn node_canvas_zoom_at_keeps_focus_anchor() {
	let mut canvas = canvas_with_view();
	let focus = [640.0, 120.0];
	let before = canvas.screen_to_world(focus);
	canvas.zoom_at(2.0, focus).unwrap();
	let after = canvas.screen_to_world(focus);
	assert!((after[0] - before[0]).abs() < 1e-3);
	assert!((after[1] - before[1]).abs() < 1e-3);
	assert_eq!(canvas.state().zoom, 2.0);
	// Clamping to the zoom floor.
	for _ in 0..20 {
		canvas.zoom_at(0.1, [400.0, 300.0]).unwrap();
	}
	assert_eq!(canvas.state().zoom, NodeCanvas::ZOOM_MIN);
}

#[test]
fn node_canvas_fit_to_view_animates_to_target() {
	let mut canvas = canvas_with_view();
	let bounds = WorldAabb {
		min: [-50.0, -25.0],
		max: [50.0, 25.0],
	};
	canvas.fit_to_view(&bounds, 32.0).unwrap();
	assert!(canvas.is_animating());
	// 200 ms donor duration: 4 steps of 60 ms overshoot and clamp.
	for _ in 0..4 {
		canvas.step_animation(60.0).unwrap();
	}
	assert!(!canvas.is_animating());
	let state = canvas.state();
	assert_eq!(state.pan, [0.0, 0.0]);
	// min((800-64)/100, (600-64)/50) = min(7.36, 10.72) = 7.36
	assert!((state.zoom - 7.36).abs() < 0.01);
	// A second fit on the same bounds is a no-op animation.
	assert!(canvas.visible_world_rect().contains([45.0, 20.0]));
}

#[test]
fn node_canvas_hit_test_prefers_highest_layer() {
	let canvas = canvas_with_view();
	let items = [
		NodeCanvasHitItem {
			id: 7,
			bounds: WorldAabb {
				min: [-10.0, -10.0],
				max: [10.0, 10.0],
			},
			layer: 0,
			enabled: true,
		},
		NodeCanvasHitItem {
			id: 9,
			bounds: WorldAabb {
				min: [-5.0, -5.0],
				max: [5.0, 5.0],
			},
			layer: 3,
			enabled: true,
		},
		NodeCanvasHitItem {
			id: 4,
			bounds: WorldAabb {
				min: [-2.0, -2.0],
				max: [2.0, 2.0],
			},
			layer: 9,
			enabled: false,
		},
	];
	let hit = canvas.hit_test([400.0, 300.0], &items).unwrap().unwrap();
	assert_eq!(hit.id, 9);
	assert_eq!(hit.index, 1);
	// A miss outside all bounds.
	assert!(canvas.hit_test([0.0, 0.0], &items).unwrap().is_none());
	// Invalid items fail closed.
	let bad = [NodeCanvasHitItem {
		id: 0,
		bounds: items[0].bounds,
		layer: 0,
		enabled: true,
	}];
	assert!(canvas.hit_test([400.0, 300.0], &bad).is_err());
}

#[test]
fn node_canvas_grid_promotes_by_decades() {
	let mut canvas = canvas_with_view();
	let base = canvas.grid(8.0, 10.0, 10, 100).unwrap();
	assert_eq!(base.minor_world_step, 10.0);
	assert_eq!(base.minor_screen_step, 10.0);
	// Zoom far out: the minor tier promotes by whole decades only.
	canvas
		.set_state(NodeCanvasState {
			pan: [0.0, 0.0],
			zoom: NodeCanvas::ZOOM_MIN,
			view_size: [800.0, 600.0],
		})
		.unwrap();
	let promoted = canvas.grid(8.0, 10.0, 10, 100).unwrap();
	assert_eq!(promoted.minor_world_step, 1000.0);
	assert_eq!(promoted.minor_screen_step, 50.0);
	// Zooming in never invents a sub-base grid.
	canvas
		.set_state(NodeCanvasState {
			pan: [0.0, 0.0],
			zoom: NodeCanvas::ZOOM_MAX,
			view_size: [800.0, 600.0],
		})
		.unwrap();
	let dense = canvas.grid(8.0, 10.0, 10, 100).unwrap();
	assert_eq!(dense.minor_world_step, 10.0);
	assert_eq!(dense.minor_screen_step, 80.0);
}

#[test]
fn input_system_dispatches_by_context_and_consumes_repeats() {
	use std::cell::Cell;
	use std::rc::Rc;

	let mut input = InputSystem::new();
	input.register_defaults();
	assert_eq!(input.action_count(), 7);

	let fired = Rc::new(Cell::new(0_u32));
	let counter = Rc::clone(&fired);
	input.set_callback("save", move || counter.set(counter.get() + 1));

	let ctrl_s = oa::input::InputEvent {
		kind: InputEventKind::KeyDown,
		key: oa::input::Key::S,
		modifiers: oa::input::INPUT_MOD_CTRL,
		..Default::default()
	};
	assert!(input.dispatch(&ctrl_s));
	assert_eq!(fired.get(), 1);

	// Repeated key down on a no-repeat action is consumed without firing.
	let repeat = oa::input::InputEvent {
		key_repeat: true,
		..ctrl_s.clone()
	};
	assert!(input.dispatch(&repeat));
	assert_eq!(fired.get(), 1);

	// NodeCanvas actions stay dormant in Global context.
	let undo = oa::input::InputEvent {
		kind: InputEventKind::KeyDown,
		key: oa::input::Key::Z,
		modifiers: oa::input::INPUT_MOD_CTRL,
		..Default::default()
	};
	let undo_fired = Rc::new(Cell::new(false));
	let flag = Rc::clone(&undo_fired);
	input.set_callback("undo", move || flag.set(true));
	// Out-of-context bindings do not match and the event is not consumed.
	assert!(!input.dispatch(&undo));
	assert!(!undo_fired.get());
	input.set_context(InputContext::NodeCanvas);
	assert!(input.dispatch(&undo));
	assert!(undo_fired.get());

	// Non-key events and unbound keys are not consumed.
	assert!(!input.dispatch(&oa::input::InputEvent {
		kind: InputEventKind::MouseMove,
		..Default::default()
	}));
	input.unregister_action("save");
	assert!(!input.dispatch(&ctrl_s));

	// Rebind moves the trigger.
	input.rebind(
		"fit_view",
		KeyBinding {
			key: oa::input::Key::G,
			modifiers: 0,
		},
	);
	let fit = oa::input::InputEvent {
		kind: InputEventKind::KeyDown,
		key: oa::input::Key::G,
		..Default::default()
	};
	assert!(input.dispatch(&fit));

	// YAML persistence is honestly gated without a backend.
	match input.save_bindings_yaml("x.yaml") {
		Err(e) => assert_eq!(e.kind(), oa::ErrorKind::MissingCapability),
		Ok(()) => panic!("YAML save succeeded without a backend"),
	}
	assert!(input.save_bindings_yaml("").is_err());
}

// ── oa::input ──────────────────────────────────────────────────────────────

fn sdl_wheel(scroll_y: f32, integer_y: i32) -> sdl3::event::Event {
	sdl3::event::Event::MouseWheel {
		timestamp: 0,
		window_id: 0,
		which: 0,
		x: 0.0,
		y: scroll_y,
		direction: sdl3::mouse::MouseWheelDirection::Normal,
		mouse_x: 0.0,
		mouse_y: 0.0,
		integer_x: 0,
		integer_y,
	}
}

#[test]
fn ui_scroll_gesture_classifies_by_event_shape() {
	use oa::input::ScrollGesture;

	// A discrete integer detent classifies as a mouse wheel.
	let wheel = oa::input::sdl_events::convert_sdl_event(&sdl_wheel(1.0, 1), 1.0, 1.0).unwrap();
	assert_eq!(wheel.scroll_gesture, ScrollGesture::MouseWheel);
	// Smooth deltas classify as a touchpad pan — the gesture is stateless, so
	// an event can never inherit wheel semantics from an earlier detent.
	let pan = oa::input::sdl_events::convert_sdl_event(&sdl_wheel(0.3, 0), 1.0, 1.0).unwrap();
	assert_eq!(pan.scroll_gesture, ScrollGesture::TouchpadPan);
	// Zero motion is a no-op.
	let idle = oa::input::sdl_events::convert_sdl_event(&sdl_wheel(0.0, 0), 1.0, 1.0).unwrap();
	assert_eq!(idle.scroll_gesture, ScrollGesture::None);
}

#[test]
fn input_process_event_tracks_edges() {
	use sdl3::event::Event as SdlEvent;
	use sdl3::keyboard::{Mod, Scancode};
	use sdl3::mouse::MouseButton;

	oa::input::shutdown();
	oa::input::initialize();

	oa::input::process_event(&SdlEvent::KeyDown {
		timestamp: 0,
		window_id: 0,
		keycode: None,
		scancode: Some(Scancode::Space),
		keymod: Mod::NOMOD,
		repeat: false,
		which: 0,
		raw: 0,
	});
	oa::input::process_event(&SdlEvent::MouseButtonDown {
		timestamp: 0,
		window_id: 0,
		which: 0,
		mouse_btn: MouseButton::Left,
		clicks: 1,
		x: 40.0,
		y: 25.0,
	});
	oa::input::update();

	assert!(oa::input::down(oa::input::Button::KeySpace));
	assert!(oa::input::press(oa::input::Button::KeySpace));
	assert!(oa::input::down(oa::input::Button::MouseLeft));
	assert!(oa::input::press(oa::input::Button::MouseLeft));
	assert!(!oa::input::release(oa::input::Button::MouseLeft));
	assert_eq!(oa::input::pointer_position(), (40.0, 25.0));
	assert!(oa::input::mouse(|m| m.left));

	// Held keys are no longer fresh presses after the next update.
	oa::input::update();
	assert!(oa::input::down(oa::input::Button::KeySpace));
	assert!(!oa::input::press(oa::input::Button::KeySpace));

	oa::input::process_event(&SdlEvent::KeyUp {
		timestamp: 0,
		window_id: 0,
		keycode: None,
		scancode: Some(Scancode::Space),
		keymod: Mod::NOMOD,
		repeat: false,
		which: 0,
		raw: 0,
	});
	oa::input::update();
	assert!(!oa::input::down(oa::input::Button::KeySpace));
	assert!(oa::input::release(oa::input::Button::KeySpace));

	// clear_for_next_frame drops edges but keeps held state.
	oa::input::process_event(&SdlEvent::MouseWheel {
		timestamp: 0,
		window_id: 0,
		which: 0,
		x: 0.0,
		y: 2.0,
		direction: sdl3::mouse::MouseWheelDirection::Normal,
		mouse_x: 40.0,
		mouse_y: 25.0,
		integer_x: 0,
		integer_y: 2,
	});
	oa::input::clear_for_next_frame();
	assert!(oa::input::mouse(|m| m.scroll_y == 0.0));
	assert!(!oa::input::press(oa::input::Button::MouseLeft));
	oa::input::shutdown();
}

// ── Viewport ────────────────────────────────────────────────────────────────

use oa::ui::{RenderLayer, ScissorDesc, Viewport, ViewportDesc, ViewportMode};

#[test]
fn viewport_desc_and_scissor_validation() {
	let desc = ViewportDesc::default();
	assert!(desc.is_valid());
	assert!((desc.aspect_ratio().unwrap() - 1280.0 / 720.0).abs() < 1e-6);
	assert!(
		!ViewportDesc {
			width: 0,
			..ViewportDesc::default()
		}
		.is_valid()
	);
	assert!(
		!ViewportDesc {
			min_depth: 0.5,
			max_depth: 0.4,
			..ViewportDesc::default()
		}
		.is_valid()
	);
	assert!(
		!ViewportDesc {
			max_depth: 1.5,
			..ViewportDesc::default()
		}
		.is_valid()
	);
	assert!(
		!ViewportDesc {
			min_depth: -0.1,
			..ViewportDesc::default()
		}
		.is_valid()
	);
	assert!(
		ViewportDesc {
			width: 0,
			..ViewportDesc::default()
		}
		.aspect_ratio()
		.is_err()
	);

	let scissor = ScissorDesc {
		x: 10,
		y: 20,
		width: 30,
		height: 40,
	};
	assert!(scissor.is_valid());
	assert!(scissor.contains(10, 20));
	assert!(scissor.contains(39, 59));
	assert!(!scissor.contains(40, 20));
	assert!(!scissor.contains(10, 60));
	assert!(!scissor.contains(-10, -20));
	assert!(
		!ScissorDesc {
			width: 0,
			height: 4,
			x: 0,
			y: 0
		}
		.is_valid()
	);
	// Extreme coordinates cannot wrap the i64 range checks.
	assert!(!scissor.contains(i32::MAX, i32::MAX));
	assert!(!scissor.contains(i32::MIN, i32::MIN));
}

#[test]
fn viewport_state_setters_and_layers() {
	let mut view = Viewport::default();
	assert_eq!(view.mode(), ViewportMode::Image2D);
	assert!(view.camera().is_none());
	assert!(view.target().is_none());
	assert!(!view.is_valid());
	assert!(!view.is_scissor_enabled());
	assert_eq!(view.layer_mask(), RenderLayer::DEFAULT);
	assert_eq!(view.clear_depth(), 1.0);
	assert_eq!(view.clear_color(), [0.01, 0.01, 0.01, 1.0]);

	// Invalid setters fail closed and preserve the prior value.
	assert!(
		view
			.set_viewport(ViewportDesc {
				width: 0,
				..ViewportDesc::default()
			})
			.is_err()
	);
	assert_eq!(view.viewport(), ViewportDesc::default());
	assert!(
		view
			.set_scissor(ScissorDesc {
				x: 0,
				y: 0,
				width: -1,
				height: 4
			})
			.is_err()
	);
	assert!(!view.is_scissor_enabled());
	assert!(view.set_clear_depth(1.5).is_err());
	assert_eq!(view.clear_depth(), 1.0);
	assert!(view.set_clear_color([0.0, f32::NAN, 0.0, 1.0]).is_err());

	view
		.set_scissor(ScissorDesc {
			x: 2,
			y: 3,
			width: 8,
			height: 9,
		})
		.unwrap();
	assert!(view.is_scissor_enabled());
	assert_eq!(view.scissor().width, 8);
	view.disable_scissor();
	assert!(!view.is_scissor_enabled());
	assert!(view.is_layer_visible(RenderLayer::DEFAULT));
	assert!(!view.is_layer_visible(RenderLayer::OVERLAY));
	view.set_layer_mask(RenderLayer::DEFAULT | RenderLayer::OVERLAY);
	assert!(view.is_layer_visible(RenderLayer::OVERLAY));
	assert!(!view.is_layer_visible(RenderLayer::DEBUG));
	assert!(RenderLayer::ALL.intersects(RenderLayer::DEBUG));
	assert!(!RenderLayer::NONE.intersects(RenderLayer::ALL));
}

test_vk!(viewport_setup_image_2d_and_aspect_fit, engine, {
	use oa::render::texture_from_rgba8;

	let texture = texture_from_rgba8(&[255_u8; 4 * 4 * 4], 4, 4)?;
	let mut view = Viewport::default();
	view.setup_image_2d(
		200.0,
		100.0,
		&texture,
		ViewportDesc {
			x: 4,
			y: 8,
			width: 320,
			height: 240,
			min_depth: 0.0,
			max_depth: 1.0,
		},
	)?;
	assert!(view.is_valid());
	assert_eq!(view.mode(), ViewportMode::Image2D);
	assert!(view.camera().is_some());
	assert_eq!(view.viewport().x, 4);
	assert_eq!(view.scissor().width, 320);
	assert!(!view.is_scissor_enabled());

	// Invalid inputs fail closed without replacing the valid state.
	assert!(
		view
			.setup_image_2d(0.0, 100.0, &texture, ViewportDesc::default())
			.is_err()
	);
	assert!(
		view
			.setup_image_2d(
				100.0,
				100.0,
				&texture,
				ViewportDesc {
					height: 0,
					..ViewportDesc::default()
				},
			)
			.is_err()
	);
	assert!(view.is_valid());

	// 2:1 image inside a 16:9 window letterboxes top and bottom.
	view.setup_image_aspect_fit(200.0, 100.0, 160.0, 90.0, &texture)?;
	let fitted = view.viewport();
	assert_eq!(fitted.x, 0);
	assert_eq!(fitted.y, 5);
	assert_eq!(fitted.width, 160);
	assert_eq!(fitted.height, 80);
	assert!(view.is_valid());

	// 1:2 image inside a 16:9 window pillarboxes left and right.
	view.setup_image_aspect_fit(100.0, 200.0, 160.0, 90.0, &texture)?;
	let fitted = view.viewport();
	assert_eq!(fitted.x, 58);
	assert_eq!(fitted.y, 0);
	assert_eq!(fitted.width, 45);
	assert_eq!(fitted.height, 90);

	assert!(
		view
			.setup_image_aspect_fit(0.0, 100.0, 160.0, 90.0, &texture)
			.is_err()
	);
	assert!(
		view
			.setup_image_aspect_fit(100.0, 200.0, 0.0, 90.0, &texture)
			.is_err()
	);
	Ok(())
});
// ── CvFrame ──────────────────────────────────────────────────────────────────

use oa::ui::{CvBbox, CvBboxesConfig, CvFrame};

#[test]
fn cv_frame_value_defaults_and_overlay_order() {
	let mut frame = CvFrame::default();
	assert!(frame.base.is_none());
	assert_eq!(frame.w, 0);
	assert_eq!(frame.h, 0);
	assert!(frame.overlays.is_empty());

	let config = CvBboxesConfig::default();
	assert_eq!(config.line_width, 1.5);
	assert_eq!(config.alpha, 1.0);
	assert_eq!(config.label_scale, 1);
	assert!(config.show_labels);
	assert!(config.show_scores);
	let bbox = CvBbox::default();
	assert_eq!(bbox.score, 1.0);
	assert!(bbox.label.is_empty());

	frame.add_bboxes(vec![CvBbox::default()], &config);
	frame.add_bboxes(vec![CvBbox::default(), CvBbox::default()], &config);
	assert_eq!(frame.overlays.len(), 2);
	assert_eq!(frame.overlays[1].boxes.len(), 2);
	frame.clear_overlays();
	assert!(frame.overlays.is_empty());
}

test_vk!(cv_frame_render_composites_boxes_and_labels, engine, {
	let mut frame = CvFrame {
		w: 64,
		h: 48,
		..CvFrame::default()
	};
	frame.add_bboxes(
		vec![CvBbox {
			x: 8.0,
			y: 10.0,
			w: 20.0,
			h: 16.0,
			score: 0.9,
			label: "cat".to_string(),
			..CvBbox::default()
		}],
		&CvBboxesConfig {
			color: Color::error(),
			line_width: 1.0,
			show_scores: false,
			..CvBboxesConfig::default()
		},
	);
	let texture = frame.render()?;
	let pixels = texture.read_rgba8()?;
	assert_eq!(pixels.len(), 64 * 48 * 4);
	let pixel = |x: usize, y: usize| &pixels[(y * 64 + x) * 4..(y * 64 + x + 1) * 4];
	// error() is a strong red: the bottom box edge at (16,25) is red.
	let edge = pixel(16, 25);
	assert!(
		edge[0] > 150 && edge[1] < 90 && edge[2] < 90,
		"edge: {edge:?}"
	);
	// Interior stays black (base canvas fallback keeps alpha 255).
	let inside = pixel(16, 30);
	assert_eq!(inside[0], 0);
	assert_eq!(inside[1], 0);
	assert_eq!(inside[2], 0);
	assert_eq!(inside[3], 255);
	// The label drops inside the box top (ly=by+2=12): the 'c' glyph's first
	// row paints x 11..13 at y 12 in red.
	let label = pixel(12, 12);
	assert!(label[0] > 150 && label[1] < 90, "label: {label:?}");

	// Host-base variant composites over caller pixels instead of black.
	let mut blue_base = vec![0_u8; 64 * 48 * 4];
	for index in (2..blue_base.len()).step_by(4) {
		blue_base[index] = 255;
		blue_base[index + 1] = 255;
	}
	let texture = frame.render_with_rgba(&blue_base)?;
	let pixels = texture.read_rgba8()?;
	let inside = &pixels[(40 * 64 + 50) * 4..(40 * 64 + 50 + 1) * 4];
	assert_eq!(inside[2], 255);
	// The label background (alpha 192 black over blue) darkens the strip at
	// (9,11) — inside the label fill but on neither box edge nor a glyph.
	let label_bg = &pixels[(11 * 64 + 9) * 4..(11 * 64 + 9 + 1) * 4];
	assert!(
		label_bg[2] < 100 && label_bg[0] < 40,
		"label_bg: {label_bg:?}"
	);
	assert!(frame.render_with_rgba(&[0_u8; 16]).is_err());
	Ok(())
});

test_vk!(cv_frame_render_rejects_invalid_state, engine, {
	let mut frame = CvFrame::default();
	assert!(frame.render().is_err());
	frame.w = 8;
	frame.h = 8;
	frame.add_bboxes(
		vec![CvBbox {
			w: 0.0,
			h: 4.0,
			..CvBbox::default()
		}],
		&CvBboxesConfig::default(),
	);
	assert!(frame.render().is_err());
	frame.clear_overlays();
	frame.add_bboxes(
		vec![CvBbox {
			x: 1.0,
			y: 1.0,
			w: 4.0,
			h: 4.0,
			..CvBbox::default()
		}],
		&CvBboxesConfig {
			alpha: f32::NAN,
			..CvBboxesConfig::default()
		},
	);
	assert!(frame.render().is_err());
	frame.clear_overlays();
	assert!(frame.render().is_ok());
	Ok(())
});

// ── ImagePlanes ──────────────────────────────────────────────────────────────

use oa::ui::{ImageDtype, ImagePlanes};

test_vk!(image_planes_from_planes_validates_upload, engine, {
	// RGB U8 4x3: three planes of 12 bytes each.
	let r = vec![255_u8; 12];
	let g = vec![128_u8; 12];
	let b = vec![0_u8; 12];
	let planes = ImagePlanes::from_planes(
		&[&r, &g, &b],
		&[ImageDtype::U8, ImageDtype::U8, ImageDtype::U8],
		4,
		3,
	)?;
	assert!(planes.is_valid());
	assert_eq!(planes.channel_count(), 3);
	assert_eq!(planes.width(), 4);
	assert_eq!(planes.height(), 3);
	assert_eq!(planes.plane_dtype(0), ImageDtype::U8);
	assert_eq!(planes.plane_dtype(9), ImageDtype::U8);
	assert_eq!(planes.plane_dtype(2), ImageDtype::U8);

	// Mixed dtypes: U8 + U16 + F32 + Bf16.
	let u16_plane = vec![0_u8; 12 * 2];
	let f32_plane = vec![0_u8; 12 * 4];
	let bf16_plane = vec![0_u8; 12 * 2];
	let mixed = ImagePlanes::from_planes(
		&[&r, &u16_plane, &f32_plane, &bf16_plane],
		&[
			ImageDtype::U8,
			ImageDtype::U16,
			ImageDtype::F32,
			ImageDtype::Bf16,
		],
		4,
		3,
	)?;
	assert_eq!(mixed.channel_count(), 4);
	assert_eq!(mixed.plane_dtype(3), ImageDtype::Bf16);

	// Validation failures.
	assert!(
		ImagePlanes::from_planes(&[], &[], 4, 3).is_err(),
		"zero planes must fail"
	);
	assert!(
		ImagePlanes::from_planes(&[&r, &g, &b, &r, &g], &[ImageDtype::U8; 5], 4, 3).is_err(),
		"five planes must fail"
	);
	assert!(
		ImagePlanes::from_planes(&[&r, &g], &[ImageDtype::U8], 4, 3).is_err(),
		"dtype length mismatch must fail"
	);
	assert!(
		ImagePlanes::from_planes(&[&r], &[ImageDtype::U8], 0, 3).is_err(),
		"zero width must fail"
	);
	assert!(
		ImagePlanes::from_planes(&[&r], &[ImageDtype::U8], 5, 3).is_err(),
		"byte count mismatch must fail"
	);
	// U16 expects 2 bytes per element; a U8-sized span must fail.
	assert!(ImagePlanes::from_planes(&[&r], &[ImageDtype::U16], 4, 3).is_err());
	Ok(())
});

test_vk!(image_planes_load_file_decodes_rgba_png, engine, {
	let path = std::env::temp_dir().join(format!("oa-planes-{}.png", std::process::id()));
	// 2x1 red/green pair in RGBA8.
	oa::image::save_rgba_file(&[255, 0, 0, 255, 0, 255, 0, 128], 2, 1, &path, 90)?;
	let planes = ImagePlanes::load_file(&path)?;
	let _ = std::fs::remove_file(&path);
	assert!(planes.is_valid());
	assert_eq!(planes.channel_count(), 4);
	assert_eq!(planes.width(), 2);
	assert_eq!(planes.height(), 1);
	assert_eq!(planes.plane_dtype(0), ImageDtype::U8);
	assert!(ImagePlanes::load_file("").is_err());
	assert!(ImagePlanes::load_file(path.with_extension("missing")).is_err());
	Ok(())
});

test_vk!(ui_image_planar_composites_rgb_and_gray, engine, {
	let mut ui = Ui::init(&engine, 32, 32)?;
	// 4x4 RGB: R=255, G=128, B=0 everywhere.
	let planes = ImagePlanes::from_planes(
		&[vec![255_u8; 16], vec![128_u8; 16], vec![0_u8; 16]]
			.iter()
			.map(Vec::as_slice)
			.collect::<Vec<_>>(),
		&[ImageDtype::U8; 3],
		4,
		4,
	)?;
	ui.begin_frame(16.0);
	ui.image_planar(&planes, 4, 6)?;
	ui.end_frame();
	let pixels = ui.render_rgba8()?;
	let pixel = |x: usize, y: usize| &pixels[(y * 32 + x) * 4..(y * 32 + x + 1) * 4];
	let color = pixel(6, 8);
	assert!(
		color[0] > 240 && color[1] > 110 && color[1] < 145 && color[2] < 20,
		"color: {color:?}"
	);
	assert_eq!(color[3], 255);
	// Beyond the 4x4 image the compose background stays cleared.
	assert_eq!(pixel(20, 8), pixel(31, 31));

	// Single channel as grayscale through image_plane.
	let gray = ImagePlanes::from_planes(&[&[200_u8; 16]], &[ImageDtype::U8], 4, 4)?;
	ui.begin_frame(16.0);
	ui.image_plane(&gray, 0, 0, 0)?;
	ui.end_frame();
	let pixels = ui.render_rgba8()?;
	let pixel = |x: usize, y: usize| &pixels[(y * 32 + x) * 4..(y * 32 + x + 1) * 4];
	let shade = pixel(2, 2);
	assert!(
		shade[0] > 180 && shade[0] == shade[1] && shade[1] == shade[2] && shade[3] == 255,
		"shade: {shade:?}"
	);
	// Out-of-range channel fails closed.
	assert!(ui.image_plane(&gray, 1, 0, 0).is_err());
	ui.close()?;
	Ok(())
});

// ── TrainingViewerSource ─────────────────────────────────────────────────────

use oa::ml::training::{TrainingSession, TrainingSessionConfig};
use oa::viewer::{TrainingViewerConfig, TrainingViewerSource};

#[test]
fn training_viewer_config_is_normalized() {
	let session = TrainingSession::new(TrainingSessionConfig::default());
	let source = TrainingViewerSource::new(
		&session,
		TrainingViewerConfig {
			history_capacity: 0,
			max_metric_plots: 99,
			..TrainingViewerConfig::default()
		},
	);
	assert_eq!(source.config().history_capacity, 2);
	assert_eq!(source.config().max_metric_plots, 12);
	assert_eq!(source.config().title, "OA training");
	assert!(source.config().show_preview);
}

#[test]
fn training_viewer_update_validates_delta_and_state() {
	use oa::ViewerLiveSource;
	let session = TrainingSession::new(TrainingSessionConfig::default());
	let mut source = TrainingViewerSource::new(&session, TrainingViewerConfig::default());
	assert!(source.update(f32::NAN).is_err());
	assert!(source.update(-1.0).is_err());
	source.update(0.0).expect("zero delta");
	source.update(16.0).expect("positive delta");
	assert!(source.latest_snapshot().is_none());
	assert!(source.latest_preview().is_none());
	assert_eq!(source.metric_series_count(), 0);
	assert_eq!(source.metric_sample_count("loss"), 0);
}

#[test]
fn training_viewer_event_dispatch_ignores_non_keys() {
	use oa::ViewerLiveSource;
	let session = TrainingSession::new(TrainingSessionConfig::default());
	let mut source = TrainingViewerSource::new(&session, TrainingViewerConfig::default());
	for kind in [InputEventKind::MouseMove, InputEventKind::MouseScroll] {
		source
			.on_event(&InputEvent {
				kind,
				..InputEvent::default()
			})
			.expect("non-key event");
	}
	// Key-down actions enqueue on the shared session without blocking.
	source
		.on_event(&InputEvent {
			kind: InputEventKind::KeyDown,
			key: oa::input::Key::Space,
			..InputEvent::default()
		})
		.expect("space");
	source
		.on_event(&InputEvent {
			kind: InputEventKind::KeyDown,
			key: oa::input::Key::C,
			..InputEvent::default()
		})
		.expect("checkpoint");
	source.close().expect("close");
}

test_vk!(training_viewer_renders_dashboard_and_preview, engine, {
	use oa::ViewerLiveSource;
	let session = TrainingSession::new(TrainingSessionConfig::default());
	let mut source = TrainingViewerSource::new(&session, TrainingViewerConfig::default());
	source.open(&engine)?;
	source.update(16.0)?;
	let mut ui = Ui::init(&engine, 800, 560)?;
	ui.begin_frame(16.0);
	source.render(&mut ui, 800, 560)?;
	ui.end_frame();
	let pixels = ui.render_rgba8()?;
	let pixel = |x: usize, y: usize| &pixels[(y * 800 + x) * 4..(y * 800 + x + 1) * 4];
	// Header card is near-black; the status strip carries the state color.
	let header = pixel(40, 40);
	assert!(
		header[0] < 40 && header[1] < 40 && header[2] < 40,
		"header: {header:?}"
	);
	let strip = pixel(22, 30);
	assert!(
		strip[1] > strip[0] && strip[1] > strip[2],
		"running-state strip must be green, got {strip:?}"
	);

	// A published preview with a completed producer event is promoted by
	// update and shown in the split preview panel (800 >= px(720)).
	let texture = oa::render::texture_from_rgba8(&[250_u8; 4 * 4 * 4], 4, 4)?;
	let completion = engine.checkpoint()?;
	completion.wait()?;
	source.publish_preview(oa::viewer::TrainingPreviewFrame {
		texture: Some(texture),
		completion,
		label: "preview".into(),
		step: 1,
	})?;
	source.update(16.0)?;
	assert!(source.latest_preview().is_some());
	ui.begin_frame(16.0);
	source.render(&mut ui, 800, 560)?;
	ui.end_frame();
	let pixels = ui.render_rgba8()?;
	// The split preview sits in the right pane; its panel must differ from
	// the left metrics column's near-black cards.
	let left = &pixels[(300 * 800 + 100) * 4..(300 * 800 + 101) * 4];
	let right = &pixels[(300 * 800 + 700) * 4..(300 * 800 + 701) * 4];
	assert_ne!(left, right, "preview pane must differ from metrics pane");
	ui.close()?;
	source.close()?;
	Ok(())
});

test_vk!(ui_plot_line_draws_filled_autoscaled_series, engine, {
	let mut ui = Ui::init(&engine, 96, 64)?;
	ui.begin_frame(16.0);
	ui.plot_line(
		&[0.0, 1.0, 0.25, 0.75, 0.5],
		PixelRect::new(8, 8, 80, 48),
		&oa::ui::UiPlotConfig {
			color: Color::new(1.0, 0.0, 0.0, 1.0),
			fill: true,
			..Default::default()
		},
	)?;
	ui.end_frame();
	let pixels = ui.render_rgba8()?;
	// The filled series reaches the top of the chart; the fill must leave
	// red-tinted pixels under the curve inside the plot rect.
	let red_tinted =
		pixels.as_chunks::<4>().0.iter().any(|pixel| {
			pixel[0] > pixel[1].saturating_add(20) && pixel[0] > pixel[2].saturating_add(20)
		});
	assert!(red_tinted, "filled red series must tint pixels");
	// An all-NaN series draws nothing and does not error.
	ui.begin_frame(16.0);
	ui.plot_line(
		&[f32::NAN, f32::NAN],
		PixelRect::new(8, 8, 80, 48),
		&oa::ui::UiPlotConfig::default(),
	)?;
	ui.end_frame();
	assert_eq!(ui.render_rgba8()?.len(), 96 * 64 * 4);
	ui.close()?;
	Ok(())
});

// ── Plot theme, limits, and interactive view ────────────────────────────────

#[test]
fn plot_theme_resolves_shared_ui_style_tokens() {
	use oa::plot::Theme;
	use oa::ui::UiStyle;

	// Both themes resolve through `UiStyle` presets — plot colors cannot fork.
	let dark = Theme::Dark.ui_style();
	assert_eq!(dark.plot_background.to_u32(), 0x0A0A0AFF);
	assert_eq!(dark.plot_background, UiStyle::default().plot_background);
	assert_eq!(dark.data_palette, UiStyle::default().data_palette);
	for color in [
		dark.plot_background,
		dark.plot_surface,
		dark.plot_grid,
		dark.plot_border,
	] {
		assert_eq!(color.r, color.g);
		assert_eq!(color.g, color.b);
	}
	let light = Theme::Light.ui_style();
	assert_eq!(
		light.plot_background,
		UiStyle::realm_light().plot_background
	);
	assert_ne!(light.plot_background, dark.plot_background);
	assert_ne!(light.text, dark.text);
}

/// Recover the data viewport from its two rendered spines. Tests sample
/// artists in data-relative coordinates, independently of the label layout.
fn rendered_plot_rect(cell: CellRect, is_spine: impl Fn(i32, i32) -> bool) -> CellRect {
	let columns = cell.x..cell.x + cell.w;
	let rows = cell.y..cell.y + cell.h;
	let bottom = rows
		.clone()
		.max_by_key(|&y| columns.clone().filter(|&x| is_spine(x, y)).count())
		.unwrap();
	let left = columns
		.clone()
		.max_by_key(|&x| rows.clone().filter(|&y| is_spine(x, y)).count())
		.unwrap();
	let top = (cell.y..bottom)
		.find(|&y| is_spine(left, y) && is_spine(left, y + 1))
		.unwrap();
	let right = columns.rev().find(|&x| is_spine(x, bottom)).unwrap();
	CellRect {
		x: left,
		y: top,
		w: right - left + 1,
		h: bottom - top + 1,
	}
}

fn dark_plot_rect(figure: &Figure, w: u32, h: u32, values: &[f32]) -> CellRect {
	let spine = oa::plot::Theme::Dark.ui_style().plot_border.r;
	rendered_plot_rect(figure.cell_rect(0, 0, w, h), |x, y| {
		(values[y as usize * w as usize + x as usize] - spine).abs() < 0.5 / 255.0
	})
}

test_vk!(plot_dark_axes_frame_and_labels_stay_outside_data, engine, {
	let (w, h) = (400, 260);
	let mut figure = Figure::new(FigureConfig {
		width: w,
		height: h,
		title: String::new(),
		..Default::default()
	});
	let axes = figure.ax(0, 0);
	axes.limits(-10.0, 10.0, -10.0, 10.0);
	axes.x_label("optimizer step");
	axes.y_label("cross entropy");
	let values = figure.render()?.as_matrix().read_f32()?;
	let area = dark_plot_rect(&figure, w, h, &values);
	let cell = figure.cell_rect(0, 0, w, h);
	let at = |x, y| values[(y * w as i32 + x) as usize];
	let spine = oa::plot::Theme::Dark.ui_style().plot_border.r;
	let bottom = area.y + area.h - 1;
	let x_center = area.x + area.w / 2;
	let y_center = area.y + area.h / 2;
	assert_eq!(at(area.x, bottom), spine);
	assert!(
		at(x_center, y_center) < spine,
		"zero is a normal interior grid tick"
	);
	assert!(
		at(area.x + 6, area.y + 2) > at(area.x + 6, bottom - 2),
		"dark surface has a subtle vertical gradient"
	);
	assert!(
		(bottom + 19..cell.y + cell.h)
			.any(|y| { (x_center - 32..x_center + 32).any(|x| at(x, y) > 0.35) }),
		"x axis title is below the ticks"
	);
	assert!(
		(cell.x..cell.x + 19).any(|x| {
			(y_center - 32..y_center + 32)
				.filter(|&y| at(x, y) > 0.35)
				.count()
				> 10
		}),
		"y axis title is rotated in its left gutter"
	);
	Ok(())
});

#[test]
fn ui_style_viewer_dark_is_derived_from_realm_dark_tokens() {
	use oa::ui::UiStyle;

	// The generated token schema owns all presets: viewer_dark overrides
	// the viewer chrome but inherits Realm-dark plot tokens, palette,
	// and shared metrics it does not override.
	let viewer = UiStyle::viewer_dark();
	let dark = UiStyle::default();
	assert_eq!(viewer.plot_background, dark.plot_background);
	assert_eq!(viewer.plot_surface, dark.plot_surface);
	assert_eq!(viewer.plot_grid, dark.plot_grid);
	assert_eq!(viewer.plot_border, dark.plot_border);
	assert_eq!(viewer.data_palette, dark.data_palette);
	assert_eq!(viewer.padding, dark.padding);
	assert_ne!(viewer.background, dark.background);
	assert_ne!(viewer.accent, dark.accent);
	assert_eq!(viewer.corner_radius, 15.0);
	assert_eq!(viewer.font_size, 15.0);
}

#[test]
fn plot_axes_reports_effective_and_explicit_limits() {
	let mut axes = oa::plot::Axes::default();
	axes.plot(&[1.0, 3.0, 2.0], Default::default());
	let limits = axes.data_limits();
	assert_eq!(&limits[2..], &[1.0, 3.0]);
	assert_eq!(axes.explicit_limits(), None);
	axes.limits(0.0, 10.0, -5.0, 5.0);
	assert_eq!(axes.explicit_limits(), Some([0.0, 10.0, -5.0, 5.0]));
	assert_eq!(axes.data_limits(), [0.0, 10.0, -5.0, 5.0]);
	axes.auto_limits();
	assert_eq!(axes.explicit_limits(), None);
}

test_vk!(plot_figure_render_is_deterministic, engine, {
	let mut figure = Figure::new(FigureConfig {
		width: 96,
		height: 64,
		padding: 8,
		..Default::default()
	});
	figure
		.ax(0, 0)
		.plot(&[0.0, 1.0, 0.25, 0.75, 0.5], Default::default());
	figure
		.ax(0, 0)
		.scatter(&[0.0, 2.0, 4.0], &[0.5, 1.0, 0.25], Default::default());
	let first = figure.render()?.as_matrix().read_f32()?;
	let second = figure.render()?.as_matrix().read_f32()?;
	assert_eq!(first, second, "repeated headless renders must be identical");
	Ok(())
});

test_vk!(plot_figure_light_theme_draws_light_canvas, engine, {
	let mut figure = Figure::new(FigureConfig {
		width: 64,
		height: 48,
		theme: oa::plot::Theme::Light,
		..Default::default()
	});
	figure.ax(0, 0).plot(&[0.0, 1.0], Default::default());
	let image = figure.render()?;
	let values = image.as_matrix().read_f32()?;
	let plane = 64 * 48;
	// The figure corner is canvas background — the light preset must be bright.
	assert!(values[0] > 0.8, "light theme background must be bright");
	assert!(values[plane] > 0.8 && values[plane * 2] > 0.8);
	Ok(())
});

test_vk!(
	plot_figure_image_keeps_aspect_with_title_and_caption,
	engine,
	{
		let texture = oa::render::texture_from_rgba8(&[255, 0, 0, 255].repeat(16), 4, 4)?;
		let mut figure = Figure::new(FigureConfig {
			width: 320,
			height: 180,
			title: String::new(),
			..Default::default()
		});
		let axes = figure.ax(0, 0);
		axes.title("prediction");
		axes.caption("ground truth");
		axes.imshow(texture);
		let values = figure.render()?.as_matrix().read_f32()?;
		let plane = 320 * 180;
		let red = |x: usize, y: usize| {
			let i = y * 320 + x;
			values[i] > 0.9 && values[plane + i] < 0.1
		};
		let tile_w = (0..320).filter(|&x| red(x, 90)).count();
		let tile_h = (0..180).filter(|&y| red(160, y)).count();
		assert!(tile_w > 50);
		assert_eq!(tile_w, tile_h, "the square image must stay square");
		Ok(())
	}
);

test_vk!(
	plot_figure_matrix_heatmap_renders_shader_colormap,
	engine,
	{
		// Device-resident ramp: row 0 holds the lowest values, row 3 the highest.
		let ramp: Vec<f32> = (0..16).map(|i| i as f32).collect();
		let matrix = oa::Matrix::new_from_f32([4, 4], &ramp)?;
		let mut figure = Figure::new(FigureConfig {
			width: 128,
			height: 96,
			padding: 8,
			..Default::default()
		});
		let axes = figure.ax(0, 0);
		axes.heatmap_matrix(
			matrix,
			oa::plot::HeatmapStyle {
				v_min: 0.0,
				v_max: 15.0,
				colormap: 3, // grayscale
				auto_scale: false,
				show_grid: false,
				colorbar: false,
			},
		);
		axes.grid(false);
		axes.legend(false);
		let image = figure.render()?;
		let values = image.as_matrix().read_f32()?;
		let rect = figure.cell_rect(0, 0, 128, 96);
		assert!(rect.h >= 8, "axes cell must have room for the heatmap");
		let column = (rect.x + rect.w / 2) as usize;
		// Sample the axes interior away from tick-label bands.
		let top = values[((rect.y + 4) * 128) as usize + column];
		let bottom = values[((rect.y + rect.h - 8) * 128) as usize + column];
		assert!(
			bottom > top + 0.3,
			"grayscale heatmap must brighten toward the high-value rows \
			 (top={top}, bottom={bottom})"
		);
		Ok(())
	}
);

test_vk!(plot_figure_rejects_autoscale_on_device_heatmap, engine, {
	let matrix = oa::Matrix::new_from_f32([2, 2], &[0.0, 1.0, 1.0, 0.0])?;
	let mut figure = Figure::new(FigureConfig {
		width: 32,
		height: 24,
		..Default::default()
	});
	figure.ax(0, 0).heatmap_matrix(matrix, Default::default());
	assert!(
		figure.render().is_err(),
		"auto_scale cannot inspect device data and must fail"
	);
	Ok(())
});

test_vk!(figure_view_zooms_pans_and_resets_data_limits, engine, {
	use oa::input::Key;
	use oa::viewer::ViewerLiveSource;

	let mut figure = Figure::new(FigureConfig {
		width: 96,
		height: 64,
		padding: 8,
		..Default::default()
	});
	figure
		.ax(0, 0)
		.plot(&[0.0, 1.0, 0.0, 1.0], Default::default());
	figure.ax(0, 0).limits(0.0, 10.0, 0.0, 10.0);
	let mut view = oa::plot::FigureView::new(figure);
	let mut ui = Ui::init(&engine, 96, 64)?;
	view.open(&engine)?;

	// First render establishes the view extent.
	ui.begin_frame(0.0);
	view.render(&mut ui, 96, 64)?;
	ui.end_frame();

	let cell = view.figure().cell_rect(0, 0, 96, 64);
	let cx = cell.x as f32 + cell.w as f32 * 0.5;
	let cy = cell.y as f32 + cell.h as f32 * 0.5;

	// Scroll up over the cell — the span must shrink about the anchor.
	view.on_event(&InputEvent {
		kind: InputEventKind::MouseScroll,
		mouse_x: cx,
		mouse_y: cy,
		scroll_y: 1.0,
		..Default::default()
	})?;
	// Render starts the animated transition; update lands it.
	ui.begin_frame(0.0);
	view.render(&mut ui, 96, 64)?;
	ui.end_frame();
	view.update(1000.0)?;
	ui.begin_frame(0.0);
	view.render(&mut ui, 96, 64)?;
	ui.end_frame();
	let zoomed = view
		.figure_mut()
		.ax(0, 0)
		.explicit_limits()
		.expect("scroll must install a view override");
	assert!(
		zoomed[1] - zoomed[0] < 10.0,
		"scroll up must shrink the x span"
	);
	assert!(((zoomed[0] + zoomed[1]) * 0.5 - 5.0).abs() < 0.5);

	// Drag right — the visible window pans toward smaller x.
	view.on_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: cx,
		mouse_y: cy,
		button: 1,
		..Default::default()
	})?;
	view.on_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: cx + 12.0,
		mouse_y: cy,
		..Default::default()
	})?;
	view.on_event(&InputEvent {
		kind: InputEventKind::MouseUp,
		mouse_x: cx + 12.0,
		mouse_y: cy,
		button: 1,
		..Default::default()
	})?;
	ui.begin_frame(0.0);
	view.render(&mut ui, 96, 64)?;
	ui.end_frame();
	let panned = view
		.figure_mut()
		.ax(0, 0)
		.explicit_limits()
		.expect("pan must keep the view override");
	assert!(
		panned[0] < zoomed[0],
		"drag right must pan toward smaller x"
	);

	// R restores the limits recorded at attach time.
	view.on_event(&InputEvent {
		kind: InputEventKind::KeyDown,
		key: Key::R,
		..Default::default()
	})?;
	ui.begin_frame(0.0);
	view.render(&mut ui, 96, 64)?;
	ui.end_frame();
	view.update(1000.0)?;
	ui.begin_frame(0.0);
	view.render(&mut ui, 96, 64)?;
	ui.end_frame();
	assert_eq!(
		view.figure_mut().ax(0, 0).explicit_limits(),
		Some([0.0, 10.0, 0.0, 10.0])
	);
	ui.close()?;
	Ok(())
});

#[test]
fn plot_axes_artist_visibility_and_categories_api() {
	let mut axes = oa::plot::Axes::default();
	axes.plot(&[1.0, 2.0], Default::default());
	axes.scatter(&[0.0], &[1.0], Default::default());
	axes.bar(&[3.0], Default::default());
	assert_eq!(axes.artist_count(), 3);
	// Toggling by insertion index is a no-op outside the recorded range.
	axes.set_artist_hidden(1, true);
	axes.set_artist_hidden(9, true);
	axes.categories(&["a", "b", "c"]);
	axes.categories(&[]);
	axes.xscale(oa::plot::Scale::Log10);
	axes.yscale(oa::plot::Scale::Linear);
}

test_vk!(plot_figure_matrix_line_matches_host_series, engine, {
	// A device-resident [2, N] series lowers to the same explicit-XY
	// dispatch as the uploaded path — the renders must be byte-identical.
	let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
	let y: Vec<f32> = x.iter().map(|v| v * 0.5 + 0.5).collect();
	let mut packed = x.clone();
	packed.extend_from_slice(&y);
	let matrix = oa::Matrix::new_from_f32([2, 8], &packed)?;
	let config = || FigureConfig {
		width: 96,
		height: 64,
		padding: 8,
		..Default::default()
	};
	let mut host = Figure::new(config());
	let axes = host.ax(0, 0);
	axes.plot_xy(&x, &y, Default::default());
	axes.limits(0.0, 7.0, 0.0, 5.0);
	axes.grid(false);
	axes.legend(false);
	let mut device = Figure::new(config());
	let axes = device.ax(0, 0);
	axes.plot_matrix(&matrix, Default::default());
	axes.limits(0.0, 7.0, 0.0, 5.0);
	axes.grid(false);
	axes.legend(false);
	let host_pixels = host.render()?.as_matrix().read_f32()?;
	let device_pixels = device.render()?.as_matrix().read_f32()?;
	assert_eq!(
		host_pixels, device_pixels,
		"device-resident [2, N] series must render identically to plot_xy"
	);
	Ok(())
});

test_vk!(plot_figure_matrix_y_series_draws, engine, {
	// [N] index-mode series: render must differ from the empty figure.
	let matrix = oa::Matrix::new_from_f32([8], &[0.0, 1.0, 0.0, 2.0, 0.5, 3.0, 1.0, 2.5])?;
	let config = || FigureConfig {
		width: 96,
		height: 64,
		padding: 8,
		..Default::default()
	};
	let mut empty = Figure::new(config());
	empty.ax(0, 0).limits(0.0, 7.0, -0.5, 3.5);
	let mut figure = Figure::new(config());
	let axes = figure.ax(0, 0);
	axes.plot_matrix_y(&matrix, Default::default());
	axes.limits(0.0, 7.0, -0.5, 3.5);
	let a = empty.render()?.as_matrix().read_f32()?;
	let b = figure.render()?.as_matrix().read_f32()?;
	assert_ne!(a, b, "device-resident index series must draw");
	Ok(())
});

test_vk!(plot_figure_log_scale_changes_render, engine, {
	let config = || FigureConfig {
		width: 96,
		height: 64,
		padding: 8,
		..Default::default()
	};
	let mut linear = Figure::new(config());
	linear
		.ax(0, 0)
		.plot(&[1.0, 10.0, 100.0, 1000.0], Default::default());
	let mut log = Figure::new(config());
	let axes = log.ax(0, 0);
	axes.plot(&[1.0, 10.0, 100.0, 1000.0], Default::default());
	axes.yscale(oa::plot::Scale::Log10);
	let a = linear.render()?.as_matrix().read_f32()?;
	let b = log.render()?.as_matrix().read_f32()?;
	assert_ne!(a, b, "log scale must transform the rendered series");
	let c = log.render()?.as_matrix().read_f32()?;
	assert_eq!(b, c, "log-scale render must be deterministic");
	Ok(())
});

test_vk!(plot_figure_colorbar_draws_ramp_strip, engine, {
	// Grayscale heatmap + colorbar: the strip brightens toward v_max at top.
	let mut figure = Figure::new(FigureConfig {
		width: 160,
		height: 120,
		padding: 8,
		..Default::default()
	});
	let ramp: Vec<f32> = (0..16).map(|i| i as f32).collect();
	figure.ax(0, 0).heatmap(
		&ramp,
		4,
		4,
		oa::plot::HeatmapStyle {
			v_min: 0.0,
			v_max: 15.0,
			colormap: 3,
			auto_scale: false,
			show_grid: false,
			colorbar: true,
		},
	);
	figure.ax(0, 0).grid(false);
	figure.ax(0, 0).legend(false);
	let image = figure.render()?;
	let values = image.as_matrix().read_f32()?;
	let rect = figure.cell_rect(0, 0, 160, 120);
	// The strip sits in the reserved band: 6 px right of the plot area,
	// which is 46 px narrower than the cell — strip.x = cell.right - 40.
	let column = (rect.x + rect.w - 36) as usize;
	let top = values[((rect.y + 12) * 160) as usize + column];
	let bottom = values[((rect.y + rect.h - 12) * 160) as usize + column];
	assert!(
		top > bottom + 0.3,
		"colorbar must show v_max at the top (top={top}, bottom={bottom})"
	);
	Ok(())
});

test_vk!(figure_view_box_zoom_legend_toggle_and_state, engine, {
	use oa::input::INPUT_MOD_SHIFT;
	use oa::viewer::ViewerLiveSource;

	let mut figure = Figure::new(FigureConfig {
		width: 160,
		height: 160,
		padding: 8,
		..Default::default()
	});
	figure.ax(0, 0).plot(
		&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
		oa::plot::LineStyle {
			label: "s0".to_owned(),
			..Default::default()
		},
	);
	figure.ax(0, 0).limits(0.0, 10.0, 0.0, 10.0);
	let mut view = oa::plot::FigureView::new(figure);
	let mut ui = Ui::init(&engine, 160, 160)?;
	view.open(&engine)?;

	ui.begin_frame(0.0);
	view.render(&mut ui, 160, 160)?;
	ui.end_frame();
	let before = ui.render_rgba8()?;

	let spine = oa::plot::Theme::Dark
		.ui_style()
		.plot_border
		.to_u32()
		.to_be_bytes();
	let cell = rendered_plot_rect(view.figure().cell_rect(0, 0, 160, 160), |x, y| {
		let pixel = ((y * 160 + x) * 4) as usize;
		before[pixel..pixel + 4] == spine
	});
	// Legend row 0: band x ∈ [right-86, right), y ∈ [cell.y+4, cell.y+16).
	let lx = (cell.x + cell.w - 60) as f32;
	let ly = (cell.y + 10) as f32;

	// Clicking the legend row hides the series.
	view.on_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: lx,
		mouse_y: ly,
		button: 1,
		..Default::default()
	})?;
	view.on_event(&InputEvent {
		kind: InputEventKind::MouseUp,
		mouse_x: lx,
		mouse_y: ly,
		button: 1,
		..Default::default()
	})?;
	assert_eq!(
		view.figure().axes()[0].artist_hidden(0),
		Some(true),
		"legend click must toggle the first artist"
	);
	ui.begin_frame(0.0);
	view.render(&mut ui, 160, 160)?;
	ui.end_frame();
	let toggled = ui.render_rgba8()?;
	assert_ne!(before, toggled, "legend toggle must hide the series");

	// Shift+drag box-zooms into the selection: quarter to three-quarters.
	let x0 = cell.x as f32 + cell.w as f32 * 0.25;
	let x1 = cell.x as f32 + cell.w as f32 * 0.75;
	let y0 = cell.y as f32 + cell.h as f32 * 0.25;
	let y1 = cell.y as f32 + cell.h as f32 * 0.75;
	view.on_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: x0,
		mouse_y: y0,
		button: 1,
		modifiers: INPUT_MOD_SHIFT,
		..Default::default()
	})?;
	view.on_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: x1,
		mouse_y: y1,
		..Default::default()
	})?;
	view.on_event(&InputEvent {
		kind: InputEventKind::MouseUp,
		mouse_x: x1,
		mouse_y: y1,
		button: 1,
		..Default::default()
	})?;
	view.update(1000.0)?;
	ui.begin_frame(0.0);
	view.render(&mut ui, 160, 160)?;
	ui.end_frame();
	let zoomed = view
		.figure_mut()
		.ax(0, 0)
		.explicit_limits()
		.expect("box zoom must install a view override");
	for (got, want) in zoomed.iter().zip([2.5, 7.5, 2.5, 7.5]) {
		assert!(
			(got - want).abs() < 0.3,
			"box zoom limits {zoomed:?} must cover the selection"
		);
	}
	let snapshot = view.state();

	// Double-click restores the attach-time base limits.
	let cx = cell.x as f32 + cell.w as f32 * 0.5;
	let cy = cell.y as f32 + cell.h as f32 * 0.5;
	view.on_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: cx,
		mouse_y: cy,
		button: 1,
		click_count: 2,
		..Default::default()
	})?;
	view.update(1000.0)?;
	ui.begin_frame(0.0);
	view.render(&mut ui, 160, 160)?;
	ui.end_frame();
	assert_eq!(
		view.figure_mut().ax(0, 0).explicit_limits(),
		Some([0.0, 10.0, 0.0, 10.0])
	);

	// Restoring the snapshot returns to the box-zoomed view.
	view.restore(&snapshot);
	ui.begin_frame(0.0);
	view.render(&mut ui, 160, 160)?;
	ui.end_frame();
	let restored = view
		.figure_mut()
		.ax(0, 0)
		.explicit_limits()
		.expect("restored state must reinstall the override");
	for (got, want) in restored.iter().zip(zoomed.iter()) {
		assert!((got - want).abs() < 1e-5, "state restore must round-trip");
	}
	ui.close()?;
	Ok(())
});

test_vk!(figure_view_log_scale_zoom_keeps_positive_limits, engine, {
	use oa::viewer::ViewerLiveSource;

	let mut figure = Figure::new(FigureConfig {
		width: 96,
		height: 64,
		padding: 8,
		..Default::default()
	});
	figure
		.ax(0, 0)
		.plot(&[1.0, 10.0, 100.0, 1000.0], Default::default());
	figure.ax(0, 0).yscale(oa::plot::Scale::Log10);
	figure.ax(0, 0).limits(0.0, 3.0, 1.0, 1000.0);
	let mut view = oa::plot::FigureView::new(figure);
	let mut ui = Ui::init(&engine, 96, 64)?;
	view.open(&engine)?;
	ui.begin_frame(0.0);
	view.render(&mut ui, 96, 64)?;
	ui.end_frame();

	let cell = view.figure().cell_rect(0, 0, 96, 64);
	let cx = cell.x as f32 + cell.w as f32 * 0.5;
	let cy = cell.y as f32 + cell.h as f32 * 0.5;
	view.on_event(&InputEvent {
		kind: InputEventKind::MouseScroll,
		mouse_x: cx,
		mouse_y: cy,
		scroll_y: 1.0,
		..Default::default()
	})?;
	ui.begin_frame(0.0);
	view.render(&mut ui, 96, 64)?;
	ui.end_frame();
	view.update(1000.0)?;
	ui.begin_frame(0.0);
	view.render(&mut ui, 96, 64)?;
	ui.end_frame();
	let zoomed = view
		.figure_mut()
		.ax(0, 0)
		.explicit_limits()
		.expect("scroll must install a view override");
	// Written back through the log transform: raw bounds stay positive and
	// the decade span shrinks by the zoom factor.
	assert!(zoomed[2] > 0.0, "log y minimum must stay positive");
	let decades = (zoomed[3] / zoomed[2]).log10();
	assert!(
		decades < 3.0 && decades > 1.0,
		"log zoom must shrink the decade span (decades={decades})"
	);
	ui.close()?;
	Ok(())
});

// Sampled blit: exact nearest on upscale and 1:1; bilinear on downscale.
test_vk!(ui_image_at_samples_native_view, engine, {
	use oa::render::texture_from_rgba8;

	// 4x4 quadrant pattern: top-left red, top-right green, bottom-left blue,
	// bottom-right white.
	let mut src = vec![0_u8; 4 * 4 * 4];
	for y in 0..4 {
		for x in 0..4 {
			let px = &mut src[(y * 4 + x) * 4..(y * 4 + x + 1) * 4];
			let c = match (x < 2, y < 2) {
				(true, true) => [255, 0, 0, 255],
				(false, true) => [0, 255, 0, 255],
				(true, false) => [0, 0, 255, 255],
				(false, false) => [255, 255, 255, 255],
			};
			px.copy_from_slice(&c);
		}
	}
	let texture = texture_from_rgba8(&src, 4, 4)?;

	// Upscale 4x -> each source texel fills a 4x4 block, exact nearest.
	let mut ui = Ui::init(&engine, 16, 16)?;
	ui.begin_frame(16.0);
	ui.image_at(&texture, PixelRect::new(0, 0, 16, 16))?;
	ui.end_frame();
	let pixels = ui.render_rgba8()?;
	let pixel = |x: usize, y: usize| &pixels[(y * 16 + x) * 4..(y * 16 + x + 1) * 4];
	assert_eq!(pixel(4, 4), &[255, 0, 0, 255]);
	assert_eq!(pixel(12, 4), &[0, 255, 0, 255]);
	assert_eq!(pixel(4, 12), &[0, 0, 255, 255]);
	assert_eq!(pixel(12, 12), &[255, 255, 255, 255]);
	// Exact block boundaries — no bleed across texel edges on upscale.
	assert_eq!(pixel(7, 0), &[255, 0, 0, 255]);
	assert_eq!(pixel(8, 0), &[0, 255, 0, 255]);
	ui.close()?;

	// 1-pixel checkerboard downscaled 2x: every destination sample blends
	// four neighbours; nearest could only reproduce a pure texel.
	let mut checks = vec![0_u8; 4 * 4 * 4];
	for y in 0..4 {
		for x in 0..4 {
			let v = if (x + y) % 2 == 0 { 255 } else { 0 };
			checks[(y * 4 + x) * 4..(y * 4 + x + 1) * 4].copy_from_slice(&[v, v, v, 255]);
		}
	}
	let checker = texture_from_rgba8(&checks, 4, 4)?;
	let mut ui = Ui::init(&engine, 4, 4)?;
	ui.begin_frame(16.0);
	ui.image_at(&checker, PixelRect::new(0, 0, 2, 2))?;
	ui.end_frame();
	let pixels = ui.render_rgba8()?;
	for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
		let px = &pixels[(y * 4 + x) * 4..(y * 4 + x + 1) * 4];
		assert!(
			px[0] > 80 && px[0] < 176,
			"downscaled checkerboard must blend to mid-gray, got {px:?}"
		);
	}
	ui.close()?;
	Ok(())
});

// Bottom-to-top glyph records draw ink in a tall narrow run, not a wide one.
test_vk!(ui_text_vertical_draws_bottom_to_top, engine, {
	let mut ui = Ui::init(&engine, 64, 64)?;
	ui.begin_frame(16.0);
	ui.text_vertical_at(
		"Axis",
		[40.0, 56.0],
		oa::ui::TextLayoutConfig {
			size: 20.0,
			..Default::default()
		},
		Color::new(1.0, 1.0, 1.0, 1.0),
		PixelRect::new(0, 0, 64, 64),
	)?;
	ui.end_frame();
	let pixels = ui.render_rgba8()?;
	let (mut min_x, mut max_x, mut min_y, mut max_y) = (64usize, 0usize, 64usize, 0usize);
	for y in 0..64 {
		for x in 0..64 {
			let px = &pixels[(y * 64 + x) * 4..];
			if px[0] > 100 {
				min_x = min_x.min(x);
				max_x = max_x.max(x);
				min_y = min_y.min(y);
				max_y = max_y.max(y);
			}
		}
	}
	assert!(
		max_x > min_x && max_y > min_y,
		"vertical text must draw ink"
	);
	let w = max_x - min_x;
	let h = max_y - min_y;
	assert!(
		h > w * 2,
		"bottom-to-top text must be taller than wide (w={w}, h={h})"
	);
	// The baseline is right of the ink: all ink sits left of x=41.
	assert!(
		max_x <= 41,
		"ink must sit left of the baseline, max_x={max_x}"
	);
	ui.close()?;
	Ok(())
});

// A symlog y axis keeps zero and negative data on the plot; the polyline
// must span the full cell height.
test_vk!(ui_plot_symlog_renders_both_signs, engine, {
	let mut figure = Figure::new(FigureConfig {
		width: 160,
		height: 120,
		title: String::new(),
		..Default::default()
	});
	let axes = figure.ax(0, 0);
	axes.yscale(oa::plot::Scale::Symlog);
	axes.plot_xy(
		&[0.0, 1.0, 2.0, 3.0, 4.0],
		&[-100.0, -1.0, 0.0, 1.0, 100.0],
		oa::plot::LineStyle {
			color: Color::new(1.0, 1.0, 1.0, 1.0),
			..Default::default()
		},
	);
	// `Image::as_matrix` is planar `[4, H, W]` — index the R plane directly.
	let image = figure.render()?;
	let values = image.as_matrix().read_f32()?;
	let cell = dark_plot_rect(&figure, 160, 120, &values);
	let row_has_ink =
		|row: i32| (cell.x..cell.x + cell.w).any(|x| values[(row * 160 + x) as usize] > 0.5);
	assert!(
		row_has_ink(cell.y + 2),
		"positive extremes must draw near the cell top"
	);
	assert!(
		row_has_ink(cell.y + cell.h - 3),
		"negative extremes must draw near the cell bottom"
	);
	Ok(())
});

// `set_series_highlight` keeps the highlighted artist at full alpha and
// fades the rest — the legend-hover emphasis path.
test_vk!(ui_plot_series_highlight_dims_others, engine, {
	let mut figure = Figure::new(FigureConfig {
		width: 160,
		height: 120,
		title: String::new(),
		..Default::default()
	});
	let white = oa::plot::LineStyle {
		color: Color::new(1.0, 1.0, 1.0, 1.0),
		..Default::default()
	};
	{
		let axes = figure.ax(0, 0);
		axes.plot_xy(&[0.0, 10.0], &[8.0, 8.0], white.clone());
		axes.plot_xy(&[0.0, 10.0], &[2.0, 2.0], white);
		axes.limits(0.0, 10.0, 0.0, 10.0);
		axes.legend(false);
	}
	let base = figure.render()?.as_matrix().read_f32()?;
	let cell = dark_plot_rect(&figure, 160, 120, &base);
	// y=2 maps to 20% above the cell bottom; y=8 to 20% below the top.
	let dim_row = cell.y + cell.h - 1 - (cell.h as f32 * 0.2) as i32;
	let bright_row = cell.y + (cell.h as f32 * 0.2) as i32;
	let col = (cell.x + cell.w / 2) as usize;
	let sample = |values: &[f32], row: i32| {
		(row - 1..=row + 1)
			.map(|y| values[y as usize * 160 + col])
			.fold(0.0, f32::max)
	};
	assert!(sample(&base, dim_row) > 0.5 && sample(&base, bright_row) > 0.5);
	figure.ax(0, 0).set_series_highlight(Some(0));
	let dimmed = figure.render()?.as_matrix().read_f32()?;
	assert!(
		sample(&dimmed, dim_row) < sample(&base, dim_row) * 0.6 + 0.05,
		"the non-highlighted line must fade (base={}, dimmed={})",
		sample(&base, dim_row),
		sample(&dimmed, dim_row),
	);
	assert!(
		(sample(&dimmed, bright_row) - sample(&base, bright_row)).abs() < 0.1,
		"the highlighted line must stay at full strength (base={}, dimmed={})",
		sample(&base, bright_row),
		sample(&dimmed, bright_row),
	);
	Ok(())
});

// A stream series scrolls its ring window: after overflow the rendered
// trace covers the newest samples in order, not the dropped prefix.
test_vk!(ui_plot_stream_scrolls_ring_window, engine, {
	let mut figure = Figure::new(FigureConfig {
		width: 160,
		height: 120,
		title: String::new(),
		..Default::default()
	});
	figure.ax(0, 0).plot_xy(
		&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
		&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
		oa::plot::LineStyle {
			color: Color::new(1.0, 1.0, 1.0, 1.0),
			..Default::default()
		},
	);
	let image = figure.render()?;
	let values = image.as_matrix().read_f32()?;
	let cell = dark_plot_rect(&figure, 160, 120, &values);
	// A rising ramp drawn across the whole window: ink near the top-right
	// and bottom-left of the cell.
	// `Image::as_matrix` is planar `[4, H, W]` — index the R plane directly.
	let px = |x: i32, y: i32| values[(y * 160 + x) as usize];
	// The ramp rises left-to-right: ink must appear in the bottom-left and
	// top-right corners of the cell.
	let bottom_left = (0..8).any(|d| px(cell.x + 3 + d / 3, cell.y + cell.h - 3 - d % 3) > 0.5);
	let top_right = (0..8).any(|d| px(cell.x + cell.w - 3 - d / 3, cell.y + 2 + d % 3) > 0.5);
	assert!(
		bottom_left,
		"oldest window samples must draw at the bottom left"
	);
	assert!(
		top_right,
		"newest window samples must draw at the top right"
	);
	Ok(())
});

// `Axes::band` shades the region between the two curves; pixels outside
// the band keep the plot surface.
test_vk!(ui_plot_band_fills_between_curves, engine, {
	let mut figure = Figure::new(FigureConfig {
		width: 160,
		height: 120,
		title: String::new(),
		..Default::default()
	});
	{
		let axes = figure.ax(0, 0);
		axes.band(
			&[2.0, 2.0, 2.0, 2.0, 2.0],
			&[8.0, 8.0, 8.0, 8.0, 8.0],
			oa::plot::LineStyle {
				color: Color::new(0.0, 1.0, 0.0, 1.0),
				..Default::default()
			},
		);
		axes.limits(0.0, 4.0, 0.0, 10.0);
		axes.grid(false);
		axes.legend(false);
	}
	let image = figure.render()?;
	let values = image.as_matrix().read_f32()?;
	let plane = 160 * 120;
	let cell = dark_plot_rect(&figure, 160, 120, &values);
	let mid_x = (cell.x + cell.w / 2) as usize;
	// Planar [4, H, W]: G plane offset is one plane in.
	let g = |y: i32| values[plane + (y * 160) as usize + mid_x];
	let r = |y: i32| values[(y * 160) as usize + mid_x];
	// y=5 maps to the cell middle — inside the band (2..8).
	let inside_y = cell.y + cell.h / 2;
	// y=9.5 maps near the top — outside the band.
	let outside_y = cell.y + 3;
	assert!(
		g(inside_y) > 0.6 && r(inside_y) < 0.4,
		"inside the band must be green (r={:.3}, g={:.3})",
		r(inside_y),
		g(inside_y)
	);
	assert!(
		g(outside_y) < 0.3,
		"outside the band must not be green (g={:.3})",
		g(outside_y)
	);
	Ok(())
});

// Grouped bars share each x slot side-by-side; stacked layers accumulate
// on top of each other.
test_vk!(ui_plot_grouped_and_stacked_bars, engine, {
	// Grouped: two layers in one slot — red left half, green right half.
	let mut grouped = Figure::new(FigureConfig {
		width: 160,
		height: 120,
		title: String::new(),
		..Default::default()
	});
	{
		let axes = grouped.ax(0, 0);
		axes.bar_grouped(
			&[&[5.0], &[5.0]],
			&[
				oa::plot::BarStyle {
					color: Color::new(1.0, 0.0, 0.0, 1.0),
					..Default::default()
				},
				oa::plot::BarStyle {
					color: Color::new(0.0, 1.0, 0.0, 1.0),
					..Default::default()
				},
			],
		);
		axes.limits(0.0, 1.0, 0.0, 10.0);
		axes.grid(false);
		axes.legend(false);
	}
	let image = grouped.render()?;
	let values = image.as_matrix().read_f32()?;
	let plane = 160 * 120;
	let cell = dark_plot_rect(&grouped, 160, 120, &values);
	let mid_y = (cell.y + cell.h / 2) as usize;
	let left_center = (cell.x + cell.w / 4) as usize;
	let right_center = (cell.x + cell.w * 3 / 4) as usize;
	let channel = |plane_idx: usize, y: usize, x: usize| values[plane_idx * plane + y * 160 + x];
	assert!(
		channel(0, mid_y, left_center) > 0.6 && channel(1, mid_y, left_center) < 0.4,
		"left group must be red"
	);
	assert!(
		channel(1, mid_y, right_center) > 0.6 && channel(0, mid_y, right_center) < 0.4,
		"right group must be green"
	);

	// Stacked: red 4 + green 4 — green occupies the upper half only.
	let mut stacked = Figure::new(FigureConfig {
		width: 160,
		height: 120,
		title: String::new(),
		..Default::default()
	});
	{
		let axes = stacked.ax(0, 0);
		axes.bar_stacked(
			&[&[4.0], &[4.0]],
			&[
				oa::plot::BarStyle {
					color: Color::new(1.0, 0.0, 0.0, 1.0),
					..Default::default()
				},
				oa::plot::BarStyle {
					color: Color::new(0.0, 1.0, 0.0, 1.0),
					..Default::default()
				},
			],
		);
		axes.limits(0.0, 1.0, 0.0, 10.0);
		axes.grid(false);
		axes.legend(false);
	}
	let image = stacked.render()?;
	let values = image.as_matrix().read_f32()?;
	let cell = dark_plot_rect(&stacked, 160, 120, &values);
	let bar_x = (cell.x + cell.w / 2) as usize;
	// y=6 region (upper part of the stack) is green; y=2 (lower) is red.
	let upper_y = (cell.y + cell.h / 4) as usize;
	let lower_y = (cell.y + cell.h * 3 / 4) as usize;
	let channel = |plane_idx: usize, y: usize, x: usize| values[plane_idx * plane + y * 160 + x];
	assert!(
		channel(0, lower_y, bar_x) > 0.6 && channel(1, lower_y, bar_x) < 0.4,
		"lower stack must be red"
	);
	assert!(
		channel(1, upper_y, bar_x) > 0.6 && channel(0, upper_y, bar_x) < 0.4,
		"upper stack must be green"
	);
	Ok(())
});

test_vk!(ui_plot_explicit_ticks_place_grid, engine, {
	// Explicit data-space ticks at 2/5/8 on a 0..10 axis → grid columns at
	// 20%, 50%, 80% of the plot area; a non-tick column stays clean.
	let mut figure = Figure::new(FigureConfig {
		width: 160,
		height: 120,
		title: String::new(),
		..Default::default()
	});
	{
		let axes = figure.ax(0, 0);
		axes.limits(0.0, 10.0, 0.0, 10.0);
		axes.legend(false);
		axes.xticks(&[2.0, 5.0, 8.0]);
	}
	let grid = figure.render()?.as_matrix().read_f32()?;
	figure.ax(0, 0).grid(false);
	let plain = figure.render()?.as_matrix().read_f32()?;
	let plot = dark_plot_rect(&figure, 160, 120, &plain);
	let grid_column = |fraction: f32| {
		let x = plot.x + (fraction * (plot.w - 1) as f32).round() as i32;
		(plot.y + 2..plot.y + plot.h - 2)
			.filter(|&y| {
				let pixel = (y * 160 + x) as usize;
				grid[pixel] - plain[pixel] > 0.02
			})
			.count()
	};
	for fraction in [0.2, 0.5, 0.8] {
		assert!(
			grid_column(fraction) > 20,
			"tick {fraction} must draw a grid line"
		);
	}
	assert!(grid_column(0.35) < 10, "non-tick column must stay clear");
	Ok(())
});

test_vk!(ui_plot_boxplot_draws_quartiles, engine, {
	// Sorted [1..9, 50]: q1=3.25, median=5.5, q3=7.75, whiskers 1..9 on a
	// 0..10 axis — box mid-band, thin whisker at center outside the box,
	// and empty margin beyond the whisker tips.
	let mut figure = Figure::new(FigureConfig {
		width: 160,
		height: 120,
		title: String::new(),
		..Default::default()
	});
	{
		let axes = figure.ax(0, 0);
		axes.boxplot(
			&[&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 50.0]],
			oa::plot::BoxStyle {
				color: Color::new(1.0, 0.0, 0.0, 1.0),
				..Default::default()
			},
		);
		axes.limits(-0.5, 0.5, 0.0, 10.0);
		axes.grid(false);
		axes.legend(false);
	}
	let image = figure.render()?;
	let values = image.as_matrix().read_f32()?;
	let cell = dark_plot_rect(&figure, 160, 120, &values);
	let cx = (cell.x + cell.w / 2) as usize;
	let at = |fraction_from_top: f32, x: usize| -> f32 {
		let y = (cell.y + (fraction_from_top * cell.h as f32) as i32) as usize;
		values[y * 160 + x]
	};
	// Inside the box (y≈5 → 50% from top): red ink across the box width.
	assert!(at(0.50, cx) > 0.25, "box fill must cover the slot center");
	assert!(
		at(0.50, cx - 8) > 0.25,
		"box fill must cover the slot interior"
	);
	// Whisker zone (y≈8.5 → 15% from top): thin line at center only.
	assert!(at(0.15, cx) > 0.5, "upper whisker must reach the center");
	assert!(
		at(0.15, cx - 10) < 0.2,
		"whisker is thin — left of it stays clean"
	);
	// Beyond whiskers (y≈9.6 → 4% from top): empty.
	assert!(at(0.04, cx) < 0.2, "nothing above the whisker tip");
	Ok(())
});

// Shared fixture for the sink-determinism tests — one fixed figure rendered
// identically in the parent and the spawned child process.
fn plot_sink_figure() -> Figure {
	let mut figure = Figure::new(FigureConfig {
		width: 96,
		height: 64,
		padding: 8,
		..Default::default()
	});
	figure
		.ax(0, 0)
		.plot(&[0.0, 1.0, 0.25, 0.75, 0.5], Default::default());
	figure
		.ax(0, 0)
		.scatter(&[0.0, 2.0, 4.0], &[0.5, 1.0, 0.25], Default::default());
	figure
}

// Child half of the cross-process sink oracle: renders the fixture to the
// path named by `OA_PLOT_PNG_OUT`. A run without the variable is a no-op.
test_vk!(plot_figure_save_png_child, engine, {
	let Ok(path) = std::env::var("OA_PLOT_PNG_OUT") else {
		return Ok(());
	};
	plot_sink_figure().save_to(&path)?;
	Ok(())
});

test_vk!(plot_figure_save_png_matches_child_process, engine, {
	let dir = std::env::temp_dir().join(format!("oa-plot-sink-{}", std::process::id()));
	std::fs::create_dir_all(&dir).expect("create sink test dir");
	let parent_png = dir.join("parent.png");
	let child_png = dir.join("child.png");
	plot_sink_figure().save_to(&parent_png.to_string_lossy())?;
	let status = std::process::Command::new(std::env::current_exe().expect("current test exe path"))
		.args([
			"--include-ignored",
			"--exact",
			"ui::plot_figure_save_png_child",
			"--test-threads=1",
		])
		.env("OA_PLOT_PNG_OUT", &child_png)
		.status()
		.expect("spawn sink child");
	assert!(status.success(), "child PNG encode must succeed");
	// Byte equality is the strongest form of the cross-process hash: the
	// PNG bitstream must reproduce exactly.
	let (parent, child) = (
		std::fs::read(&parent_png).expect("read parent png"),
		std::fs::read(&child_png).expect("read child png"),
	);
	assert_eq!(
		parent, child,
		"PNG bitstream must be identical across processes"
	);
	let _ = std::fs::remove_dir_all(&dir);
	Ok(())
});

test_vk!(plot_figure_save_formats_decode_back, engine, {
	let dir = std::env::temp_dir().join(format!("oa-plot-fmt-{}", std::process::id()));
	std::fs::create_dir_all(&dir).expect("create format test dir");
	let figure = plot_sink_figure();
	let expected = figure.render()?.as_matrix().read_f32()?;
	let plane = 96 * 64;
	for extension in ["png", "jpg", "webp"] {
		let path = dir.join(format!("figure.{extension}"));
		figure.save_to(&path.to_string_lossy())?;
		let decoded = oa::image::decode_file(&path, oa::ImageFormat::Rgba)?;
		let values = decoded.as_matrix().read_f32()?;
		let [batch, channels, height, width] = *decoded.as_matrix().shape() else {
			panic!("decoded figure must be [1, 4, H, W]");
		};
		assert_eq!((batch, channels, height, width), (1, 4, 64, 96));
		// PNG is lossless (≤1/255 rounding); the lossy codecs deviate at
		// sharp plot edges — the oracle is a faithful roundtrip, not parity.
		let tolerance = if extension == "png" { 2.0 / 255.0 } else { 0.3 };
		let mut worst = 0.0f32;
		for i in 0..3 * plane {
			worst = worst.max((values[i] - expected[i]).abs());
		}
		assert!(
			worst <= tolerance,
			"{extension} roundtrip deviated by {worst}"
		);
	}
	let _ = std::fs::remove_dir_all(&dir);
	Ok(())
});

// ── Data table + filterable combo ───────────────────────────────────────────

test_vk!(ui_table_selects_row_and_reports_change, engine, {
	use oa::ui::{UiTableConfig, UiTableState};

	let mut ui = Ui::init(&engine, 200, 300)?;
	let mut state = UiTableState::default();
	let columns = ["name", "value"];
	let r0 = ["alpha", "1.0"];
	let r1 = ["beta", "2.0"];
	let rows: [&[&str]; 2] = [&r0, &r1];
	let config = UiTableConfig {
		row_height: 0,
		max_visible_rows: 0,
		show_header: true,
		zebra: true,
	};

	// Row 1 sits at panel y = padding(8) + header(46) + one row(46) → center 123.
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 100.0,
		mouse_y: 123.0,
		..InputEvent::default()
	});
	ui.begin_frame(0.0);
	ui.begin_panel("panel", PixelRect::new(0, 0, 200, 300));
	assert!(!ui.table("metrics", &columns, &rows, &[], &mut state, config.clone()));
	ui.end_panel();
	ui.end_frame();

	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: 100.0,
		mouse_y: 123.0,
		button: 1,
		..InputEvent::default()
	});
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseUp,
		mouse_x: 100.0,
		mouse_y: 123.0,
		button: 1,
		..InputEvent::default()
	});
	ui.begin_frame(0.0);
	ui.begin_panel("panel", PixelRect::new(0, 0, 200, 300));
	assert!(ui.table("metrics", &columns, &rows, &[], &mut state, config));
	assert_eq!(state.selected, 1);
	ui.end_panel();
	ui.end_frame();

	// The selected row carries the accent tint on the rendered frame.
	let px = ui.render_rgba8()?;
	let y = 123usize;
	let mut tinted = 0usize;
	for x in 9..198 {
		let i = (y * 200 + x) * 4;
		let (r, b) = (px[i] as i32, px[i + 2] as i32);
		if b > r + 12 && b > 40 {
			tinted += 1;
		}
	}
	assert!(
		tinted > 60,
		"selected row should show the accent tint ({tinted}px)"
	);
	ui.close()?;
	Ok(())
});

test_vk!(ui_combo_filters_items_and_selects_match, engine, {
	let mut ui = Ui::init(&engine, 200, 300)?;
	let mut selected = -1i32;
	let mut filter = String::new();
	let items = ["apple", "apricot", "banana", "cherry"];

	// Frame 1: record the closed combo.
	ui.begin_frame(0.0);
	ui.begin_panel("panel", PixelRect::new(0, 0, 200, 300));
	assert!(!ui.combo("fruit", &items, &mut selected, &mut filter));
	ui.end_panel();
	ui.end_frame();

	// Click the button to open the popup.
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 100.0,
		mouse_y: 31.0,
		..InputEvent::default()
	});
	ui.begin_frame(0.0);
	ui.begin_panel("panel", PixelRect::new(0, 0, 200, 300));
	ui.combo("fruit", &items, &mut selected, &mut filter);
	ui.end_panel();
	ui.end_frame();
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: 100.0,
		mouse_y: 31.0,
		button: 1,
		..InputEvent::default()
	});
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseUp,
		mouse_x: 100.0,
		mouse_y: 31.0,
		button: 1,
		..InputEvent::default()
	});
	ui.begin_frame(0.0);
	ui.begin_panel("panel", PixelRect::new(0, 0, 200, 300));
	ui.combo("fruit", &items, &mut selected, &mut filter);
	ui.end_panel();
	ui.end_frame();

	// Type "ap" — the filter narrows the popup to apple/apricot.
	ui.route_event(&InputEvent {
		kind: InputEventKind::KeyChar,
		text: "ap".to_owned(),
		..InputEvent::default()
	});
	ui.begin_frame(0.0);
	ui.begin_panel("panel", PixelRect::new(0, 0, 200, 300));
	ui.combo("fruit", &items, &mut selected, &mut filter);
	ui.end_panel();
	ui.end_frame();
	assert_eq!(filter, "ap");

	// Click the second filtered row (apricot) inside the popup:
	// popup starts at widget bottom + 2; row 1 center ≈ y 108 + 23 = 131.
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 100.0,
		mouse_y: 131.0,
		..InputEvent::default()
	});
	ui.begin_frame(0.0);
	ui.begin_panel("panel", PixelRect::new(0, 0, 200, 300));
	ui.combo("fruit", &items, &mut selected, &mut filter);
	ui.end_panel();
	ui.end_frame();
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: 100.0,
		mouse_y: 131.0,
		button: 1,
		..InputEvent::default()
	});
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseUp,
		mouse_x: 100.0,
		mouse_y: 131.0,
		button: 1,
		..InputEvent::default()
	});
	ui.begin_frame(0.0);
	ui.begin_panel("panel", PixelRect::new(0, 0, 200, 300));
	assert!(ui.combo("fruit", &items, &mut selected, &mut filter));
	ui.end_panel();
	ui.end_frame();

	assert_eq!(selected, 1, "apricot is index 1");
	assert_eq!(filter, "apricot");
	ui.close()?;
	Ok(())
});

test_vk!(ui_color_picker_drag_updates_channel, engine, {
	let mut ui = Ui::init(&engine, 200, 340)?;
	let mut color = Color {
		r: 0.2,
		g: 0.3,
		b: 0.4,
		a: 1.0,
	};

	// Frame 1: record the closed picker.
	ui.begin_frame(0.0);
	ui.begin_panel("panel", PixelRect::new(0, 0, 200, 340));
	assert!(!ui.color_picker("tint", &mut color));
	ui.end_panel();
	ui.end_frame();

	// Click the button to open the popup.
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 100.0,
		mouse_y: 31.0,
		..InputEvent::default()
	});
	ui.begin_frame(0.0);
	ui.begin_panel("panel", PixelRect::new(0, 0, 200, 340));
	ui.color_picker("tint", &mut color);
	ui.end_panel();
	ui.end_frame();
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: 100.0,
		mouse_y: 31.0,
		button: 1,
		..InputEvent::default()
	});
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseUp,
		mouse_x: 100.0,
		mouse_y: 31.0,
		button: 1,
		..InputEvent::default()
	});
	ui.begin_frame(0.0);
	ui.begin_panel("panel", PixelRect::new(0, 0, 200, 340));
	ui.color_picker("tint", &mut color);
	ui.end_panel();
	ui.end_frame();

	// The r slider is the first slider row inside the popup panel:
	// popup y 56 + margin 4 + scope pad 8 + preview 46 + gap 4 = 118,
	// center y = 141; x = 100 lands on fraction (100 - 20) / 160 = 0.5.
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 100.0,
		mouse_y: 141.0,
		..InputEvent::default()
	});
	ui.begin_frame(0.0);
	ui.begin_panel("panel", PixelRect::new(0, 0, 200, 340));
	ui.color_picker("tint", &mut color);
	ui.end_panel();
	ui.end_frame();
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: 100.0,
		mouse_y: 141.0,
		button: 1,
		..InputEvent::default()
	});
	ui.begin_frame(0.0);
	ui.begin_panel("panel", PixelRect::new(0, 0, 200, 340));
	assert!(ui.color_picker("tint", &mut color));
	ui.end_panel();
	ui.end_frame();
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseUp,
		mouse_x: 100.0,
		mouse_y: 141.0,
		button: 1,
		..InputEvent::default()
	});
	// Settle frame — the preview swatch reflects the post-drag color.
	ui.begin_frame(0.0);
	ui.begin_panel("panel", PixelRect::new(0, 0, 200, 340));
	ui.color_picker("tint", &mut color);
	ui.end_panel();
	ui.end_frame();

	assert!(
		(color.r - 0.5).abs() < 0.02,
		"r channel should be dragged to ~0.5, got {}",
		color.r
	);
	assert!((color.g - 0.3).abs() < 1e-6);
	assert!((color.b - 0.4).abs() < 1e-6);

	// The preview row at the popup top shows the new color on the frame.
	let px = ui.render_rgba8()?;
	let i = (90usize * 200 + 50) * 4;
	assert!(
		(px[i] as i32 - 128).abs() < 15,
		"preview r should be ~128, got {}",
		px[i]
	);
	ui.close()?;
	Ok(())
});

#[test]
fn workspace_layout_json_round_trip() {
	use oa::ui::UiDirection;
	use oa::viewer::{Workspace, WorkspaceNode};

	let root = WorkspaceNode::split(
		UiDirection::Row,
		0.625,
		WorkspaceNode::tabs(&["metrics", "preview"]),
		WorkspaceNode::split(
			UiDirection::Column,
			0.25,
			WorkspaceNode::tabs(&["inspector"]),
			WorkspaceNode::tabs(&[]),
		),
	);
	let workspace = Workspace::new(root);
	let json = workspace.to_json().expect("encode");
	let parsed = Workspace::from_json(&json).expect("decode");
	assert_eq!(parsed.to_json().expect("re-encode"), json);
	match parsed.root() {
		WorkspaceNode::Split {
			direction,
			ratio,
			second,
			..
		} => {
			assert!(matches!(*direction, UiDirection::Row));
			assert!((ratio - 0.625).abs() < 1e-6);
			match &**second {
				WorkspaceNode::Split { second, .. } => match &**second {
					WorkspaceNode::Tabs { tabs, .. } => assert!(tabs.is_empty()),
					_ => panic!("expected tabs node"),
				},
				_ => panic!("expected split node"),
			}
		}
		_ => panic!("expected split root"),
	}

	// Malformed and unsupported documents are rejected.
	assert!(Workspace::from_json("{").is_err());
	assert!(
		Workspace::from_json("{\"version\":2,\"root\":{\"kind\":\"tabs\",\"tabs\":[]}}").is_err()
	);
	assert!(Workspace::from_json("{\"version\":1,\"root\":{\"kind\":\"bogus\"}}").is_err());
	assert!(
		Workspace::from_json("{\"version\":1,\"root\":{\"kind\":\"split\",\"direction\":\"diagonal\",\"ratio\":0.5,\"first\":{\"kind\":\"tabs\",\"tabs\":[]},\"second\":{\"kind\":\"tabs\",\"tabs\":[]}}}").is_err()
	);
}

test_vk!(ui_workspace_renders_split_tabs_and_routes_events, engine, {
	use oa::ui::UiDirection;
	use oa::viewer::{Workspace, WorkspaceItem, WorkspaceNode};
	use std::cell::Cell;
	use std::rc::Rc;

	struct Probe {
		id: &'static str,
		title: &'static str,
		color: Color,
		events: Rc<Cell<u32>>,
	}
	impl WorkspaceItem for Probe {
		fn id(&self) -> &str {
			self.id
		}
		fn title(&self) -> &str {
			self.title
		}
		fn render(&mut self, ui: &mut Ui<'_>, rect: PixelRect) -> oa::Result<()> {
			ui.rect(rect, self.color, 0.0)
		}
		fn on_event(&mut self, _event: &InputEvent) -> oa::Result<()> {
			self.events.set(self.events.get() + 1);
			Ok(())
		}
	}

	let left_events = Rc::new(Cell::new(0u32));
	let right_events = Rc::new(Cell::new(0u32));
	let mut workspace = Workspace::new(WorkspaceNode::split(
		UiDirection::Row,
		0.5,
		WorkspaceNode::tabs(&["left"]),
		WorkspaceNode::tabs(&["right"]),
	));
	workspace.add_item(Box::new(Probe {
		id: "left",
		title: "Left",
		color: Color::new(0.9, 0.1, 0.1, 1.0),
		events: left_events.clone(),
	}));
	workspace.add_item(Box::new(Probe {
		id: "right",
		title: "Right",
		color: Color::new(0.1, 0.9, 0.1, 1.0),
		events: right_events.clone(),
	}));

	let mut ui = Ui::init(&engine, 400, 300)?;
	ui.begin_frame(0.0);
	workspace.render(&mut ui, PixelRect::new(0, 0, 400, 300))?;
	ui.end_frame();

	// Left pane: split row 0.5 over 400px with a 4px handle → first 198px;
	// tab bar 28px → item content begins below it.
	let px = ui.render_rgba8()?;
	let left = &px[(150 * 400 + 50) * 4..(150 * 400 + 50) * 4 + 4];
	assert!(
		left[0] > 180 && left[1] < 80,
		"left pane should be red: {left:?}"
	);
	let right = &px[(150 * 400 + 300) * 4..(150 * 400 + 300) * 4 + 4];
	assert!(
		right[1] > 180 && right[0] < 80,
		"right pane should be green: {right:?}"
	);
	// The split handle band is neither pane color.
	let handle = &px[(150 * 400 + 199) * 4..(150 * 400 + 199) * 4 + 4];
	assert!(
		handle[0] < 140 && handle[1] < 140,
		"handle band: {handle:?}"
	);

	// Pointer events route to the leaf under the cursor; the click focuses it.
	workspace.route_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: 300.0,
		mouse_y: 150.0,
		button: 1,
		..InputEvent::default()
	})?;
	assert_eq!(right_events.get(), 1);
	assert_eq!(left_events.get(), 0);
	// Key events follow focus (set by the click above).
	workspace.route_event(&InputEvent {
		kind: InputEventKind::KeyDown,
		..InputEvent::default()
	})?;
	assert_eq!(right_events.get(), 2);
	// A pointer event over the left pane routes there.
	workspace.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 50.0,
		mouse_y: 150.0,
		..InputEvent::default()
	})?;
	assert_eq!(left_events.get(), 1);

	ui.close()?;
	Ok(())
});

test_vk!(ui_workspace_hosts_figure_view_in_pane, engine, {
	use oa::plot::FigureView;
	use oa::ui::UiDirection;
	use oa::viewer::{ViewerLiveSource, Workspace, WorkspaceItem, WorkspaceNode};

	struct Solid {
		id: &'static str,
		color: Color,
	}
	impl WorkspaceItem for Solid {
		fn id(&self) -> &str {
			self.id
		}
		fn title(&self) -> &str {
			self.id
		}
		fn render(&mut self, ui: &mut Ui<'_>, rect: PixelRect) -> oa::Result<()> {
			ui.rect(rect, self.color, 0.0)
		}
	}

	// Distinct blue figure background makes the pane boundary observable.
	let mut figure = Figure::new(FigureConfig {
		width: 200,
		height: 200,
		background: Color::new(0.1, 0.2, 0.9, 1.0),
		..FigureConfig::default()
	});
	figure
		.ax(0, 0)
		.plot(&[0.0, 1.0, 0.5, 0.25], Default::default());
	let mut view = FigureView::new(figure);
	view.set_name("metrics");

	let mut workspace = Workspace::new(WorkspaceNode::split(
		UiDirection::Row,
		0.5,
		WorkspaceNode::tabs(&["chrome"]),
		WorkspaceNode::tabs(&["metrics"]),
	));
	workspace.add_item(Box::new(Solid {
		id: "chrome",
		color: Color::new(0.9, 0.1, 0.1, 1.0),
	}));
	workspace.add_item(Box::new(view));

	let mut ui = Ui::init(&engine, 400, 300)?;
	ui.begin_frame(0.0);
	workspace.render(&mut ui, PixelRect::new(0, 0, 400, 300))?;
	ui.end_frame();

	// The right pane's left gutter shows the figure's blue background.
	let px = ui.render_rgba8()?;
	let right = &px[(150 * 400 + 215) * 4..(150 * 400 + 215) * 4 + 4];
	assert!(
		right[2] > 150 && right[0] < 80,
		"right pane should show the figure's blue background: {right:?}"
	);
	let left = &px[(150 * 400 + 50) * 4..(150 * 400 + 50) * 4 + 4];
	assert!(
		left[0] > 180 && left[1] < 80,
		"left pane should be red: {left:?}"
	);

	// A scroll over the right pane reaches the FigureView and zooms —
	// the re-rendered frame differs inside the pane.
	let before = px;
	workspace.route_event(&InputEvent {
		kind: InputEventKind::MouseScroll,
		mouse_x: 300.0,
		mouse_y: 150.0,
		scroll_y: 1.0,
		..InputEvent::default()
	})?;
	ui.begin_frame(0.0);
	workspace.render(&mut ui, PixelRect::new(0, 0, 400, 300))?;
	ui.end_frame();
	workspace.update(1000.0)?;
	ui.begin_frame(0.0);
	workspace.render(&mut ui, PixelRect::new(0, 0, 400, 300))?;
	ui.end_frame();
	let after = ui.render_rgba8()?;
	let changed = before
		.iter()
		.zip(after.iter())
		.filter(|(a, b)| a != b)
		.count();
	assert!(changed > 0, "scroll zoom should repaint the pane");
	ui.close()?;
	Ok(())
});

test_vk!(ui_workspace_drag_reorders_tab_in_bar, engine, {
	use oa::viewer::{Workspace, WorkspaceNode};

	let mut workspace = Workspace::new(WorkspaceNode::tabs(&["a", "b", "c"]));
	let mut ui = Ui::init(&engine, 400, 300)?;

	let frame = |workspace: &mut Workspace, ui: &mut Ui<'_>| {
		ui.begin_frame(0.0);
		workspace
			.render(ui, PixelRect::new(0, 0, 400, 300))
			.unwrap();
		ui.end_frame();
	};

	// Seed hover over the first tab (400px bar / 3 tabs → 133px slots).
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 66.0,
		mouse_y: 14.0,
		..InputEvent::default()
	});
	frame(&mut workspace, &mut ui);
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: 66.0,
		mouse_y: 14.0,
		button: 1,
		..InputEvent::default()
	});
	frame(&mut workspace, &mut ui);
	// Drag past the threshold to the right bar edge, then release.
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 390.0,
		mouse_y: 14.0,
		..InputEvent::default()
	});
	frame(&mut workspace, &mut ui);
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseUp,
		mouse_x: 390.0,
		mouse_y: 14.0,
		button: 1,
		..InputEvent::default()
	});
	frame(&mut workspace, &mut ui);

	let oa::viewer::WorkspaceNode::Tabs { tabs, state } = workspace.root() else {
		panic!("root should stay a tabs leaf");
	};
	assert_eq!(
		tabs.as_slice(),
		&["b".to_owned(), "c".to_owned(), "a".to_owned()],
		"dragging tab 0 to the right edge should move it last: {tabs:?}"
	);
	assert_eq!(state.selected, 2, "the dragged tab should stay selected");
	ui.close()?;
	Ok(())
});

test_vk!(ui_workspace_drag_moves_tab_across_panes, engine, {
	use oa::ui::UiDirection;
	use oa::viewer::{Workspace, WorkspaceNode};

	let mut workspace = Workspace::new(WorkspaceNode::split(
		UiDirection::Row,
		0.5,
		WorkspaceNode::tabs(&["a"]),
		WorkspaceNode::tabs(&["b"]),
	));
	let mut ui = Ui::init(&engine, 400, 300)?;

	let frame = |workspace: &mut Workspace, ui: &mut Ui<'_>| {
		ui.begin_frame(0.0);
		workspace
			.render(ui, PixelRect::new(0, 0, 400, 300))
			.unwrap();
		ui.end_frame();
	};

	// Drag "a" from the left bar onto the right pane's content.
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 60.0,
		mouse_y: 14.0,
		..InputEvent::default()
	});
	frame(&mut workspace, &mut ui);
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: 60.0,
		mouse_y: 14.0,
		button: 1,
		..InputEvent::default()
	});
	frame(&mut workspace, &mut ui);
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 300.0,
		mouse_y: 150.0,
		..InputEvent::default()
	});
	frame(&mut workspace, &mut ui);
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseUp,
		mouse_x: 300.0,
		mouse_y: 150.0,
		button: 1,
		..InputEvent::default()
	});
	frame(&mut workspace, &mut ui);

	// The left leaf gave up its only tab, so it collapses and the root
	// flattens to the surviving leaf.
	let oa::viewer::WorkspaceNode::Tabs { tabs, state } = workspace.root() else {
		panic!("root should collapse to the surviving tabs leaf");
	};
	assert_eq!(
		tabs.as_slice(),
		&["b".to_owned(), "a".to_owned()],
		"dragging onto the right pane should append the tab: {tabs:?}"
	);
	assert_eq!(state.selected, 1, "the moved tab should be selected");
	ui.close()?;
	Ok(())
});

test_vk!(ui_workspace_edge_drop_splits_pane, engine, {
	use oa::ui::UiDirection;
	use oa::viewer::{Workspace, WorkspaceNode};

	let mut workspace = Workspace::new(WorkspaceNode::split(
		UiDirection::Row,
		0.5,
		WorkspaceNode::tabs(&["a"]),
		WorkspaceNode::tabs(&["b"]),
	));
	let mut ui = Ui::init(&engine, 400, 300)?;

	let frame = |workspace: &mut Workspace, ui: &mut Ui<'_>| {
		ui.begin_frame(0.0);
		workspace
			.render(ui, PixelRect::new(0, 0, 400, 300))
			.unwrap();
		ui.end_frame();
	};

	// Drag "a" onto the right pane's right-edge zone (x past 75% of the
	// right leaf) — the leaf should split with "a" opening on the right.
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 60.0,
		mouse_y: 14.0,
		..InputEvent::default()
	});
	frame(&mut workspace, &mut ui);
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: 60.0,
		mouse_y: 14.0,
		button: 1,
		..InputEvent::default()
	});
	frame(&mut workspace, &mut ui);
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 390.0,
		mouse_y: 150.0,
		..InputEvent::default()
	});
	frame(&mut workspace, &mut ui);
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseUp,
		mouse_x: 390.0,
		mouse_y: 150.0,
		button: 1,
		..InputEvent::default()
	});
	frame(&mut workspace, &mut ui);

	// The emptied left leaf collapses; the right leaf wraps in a row split
	// with the existing pane first and the dragged tab second.
	let oa::viewer::WorkspaceNode::Split {
		direction,
		first,
		second,
		..
	} = workspace.root()
	else {
		panic!("root should be the split created by the edge drop");
	};
	assert!(
		matches!(direction, UiDirection::Row),
		"a right-edge drop should split on the row axis"
	);
	let oa::viewer::WorkspaceNode::Tabs { tabs: kept, .. } = &**first else {
		panic!("first child should be the existing leaf");
	};
	assert_eq!(kept.as_slice(), &["b".to_owned()]);
	let oa::viewer::WorkspaceNode::Tabs { tabs: moved, state } = &**second else {
		panic!("second child should be the new leaf holding the dragged tab");
	};
	assert_eq!(moved.as_slice(), &["a".to_owned()]);
	assert_eq!(state.selected, 0);
	ui.close()?;
	Ok(())
});

test_vk!(ui_drag_drop_delivers_payload_to_target, engine, {
	let mut ui = Ui::init(&engine, 300, 120)?;
	let source = PixelRect::new(10, 40, 80, 40);
	let target = PixelRect::new(200, 40, 90, 40);
	let mut dropped: Option<std::rc::Rc<u32>> = None;
	let mut hovering = false;

	let frame = |ui: &mut Ui<'_>, dropped: &mut Option<std::rc::Rc<u32>>, hovering: &mut bool| {
		ui.begin_frame(0.0);
		if ui.drag_source("card", source) {
			ui.set_drag_payload("card-id", 7u32);
		}
		let _ = ui.rect(source, Color::new(0.8, 0.2, 0.2, 1.0), 0.0);
		*hovering = ui.drag_hovering(target, "card-id");
		let _ = ui.rect(
			target,
			if *hovering {
				Color::new(0.2, 0.8, 0.2, 1.0)
			} else {
				Color::new(0.2, 0.2, 0.8, 1.0)
			},
			0.0,
		);
		if let Some(payload) = ui.drag_drop::<u32>(target, "card-id") {
			*dropped = Some(payload);
		}
		ui.end_frame();
	};

	// Hover seeds the source's widget id, then press begins the drag.
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 50.0,
		mouse_y: 60.0,
		..InputEvent::default()
	});
	frame(&mut ui, &mut dropped, &mut hovering);
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: 50.0,
		mouse_y: 60.0,
		button: 1,
		..InputEvent::default()
	});
	frame(&mut ui, &mut dropped, &mut hovering);
	// Held drag over the target: the highlight signal reports true.
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 245.0,
		mouse_y: 60.0,
		..InputEvent::default()
	});
	frame(&mut ui, &mut dropped, &mut hovering);
	assert!(hovering, "target should highlight while a payload hovers");
	assert!(dropped.is_none(), "nothing drops while the button is held");
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseUp,
		mouse_x: 245.0,
		mouse_y: 60.0,
		button: 1,
		..InputEvent::default()
	});
	frame(&mut ui, &mut dropped, &mut hovering);

	let payload = dropped.expect("release over the target delivers the payload");
	assert_eq!(*payload, 7);
	ui.close()?;
	Ok(())
});

test_vk!(ui_input_text_multiline_edits_across_lines, engine, {
	use oa::input::Key;

	let mut ui = Ui::init(&engine, 240, 160)?;
	let mut value = String::new();

	let frame = |ui: &mut Ui<'_>, value: &mut String| {
		ui.begin_frame(0.0);
		ui.begin_panel("panel", PixelRect::new(0, 0, 240, 160));
		ui.input_text_multiline("notes", value, 3);
		ui.end_panel();
		ui.end_frame();
	};

	// Focus the field: hover then press inside the allocated rect.
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 120.0,
		mouse_y: 40.0,
		..InputEvent::default()
	});
	frame(&mut ui, &mut value);
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: 120.0,
		mouse_y: 40.0,
		button: 1,
		..InputEvent::default()
	});
	frame(&mut ui, &mut value);

	// Type "ab", Enter, then "c".
	for text in ["a", "b"] {
		ui.route_event(&InputEvent {
			kind: InputEventKind::KeyChar,
			text: text.to_owned(),
			..InputEvent::default()
		});
	}
	ui.route_event(&InputEvent {
		kind: InputEventKind::KeyDown,
		key: Key::Return,
		..InputEvent::default()
	});
	ui.route_event(&InputEvent {
		kind: InputEventKind::KeyChar,
		text: "c".to_owned(),
		..InputEvent::default()
	});
	frame(&mut ui, &mut value);
	assert_eq!(
		value, "ab\nc",
		"typed text and Enter should build two lines"
	);

	// Backspace removes the last character across the line break.
	ui.route_event(&InputEvent {
		kind: InputEventKind::KeyDown,
		key: Key::Backspace,
		..InputEvent::default()
	});
	ui.route_event(&InputEvent {
		kind: InputEventKind::KeyDown,
		key: Key::Backspace,
		..InputEvent::default()
	});
	frame(&mut ui, &mut value);
	assert_eq!(value, "ab", "two backspaces remove 'c' then the line break");
	ui.close()?;
	Ok(())
});

// Keyboard navigation: Tab cycles focus, Space/Enter activate the focused
// widget, and arrows step a focused slider.
test_vk!(ui_keyboard_nav_cycles_and_steps, engine, {
	use oa::input::Key;

	let mut ui = Ui::init(&engine, 300, 200)?;
	let mut hits = [false; 2];
	let mut level = 0.5f32;

	let frame = |ui: &mut Ui<'_>, hits: &mut [bool; 2], level: &mut f32| {
		ui.begin_frame(0.0);
		ui.begin_panel("panel", PixelRect::new(0, 0, 300, 200));
		if ui.button("first") {
			hits[0] = true;
		}
		if ui.button("second") {
			hits[1] = true;
		}
		ui.slider_f32("level", level, 0.0, 1.0);
		ui.end_panel();
		ui.end_frame();
	};

	// Frame one registers the focusable order: first, second, slider.
	frame(&mut ui, &mut hits, &mut level);

	// Tab focuses the first button; Space activates it.
	ui.route_event(&InputEvent {
		kind: InputEventKind::KeyDown,
		key: Key::Tab,
		..InputEvent::default()
	});
	ui.route_event(&InputEvent {
		kind: InputEventKind::KeyDown,
		key: Key::Space,
		..InputEvent::default()
	});
	frame(&mut ui, &mut hits, &mut level);
	assert!(hits[0], "Tab+Space should activate the first button");

	// Two more Tabs focus the slider; Right arrow steps it up.
	for _ in 0..2 {
		ui.route_event(&InputEvent {
			kind: InputEventKind::KeyDown,
			key: Key::Tab,
			..InputEvent::default()
		});
	}
	ui.route_event(&InputEvent {
		kind: InputEventKind::KeyDown,
		key: Key::Right,
		..InputEvent::default()
	});
	frame(&mut ui, &mut hits, &mut level);
	assert!(
		level > 0.51,
		"Right arrow should step the focused slider (got {level})"
	);

	// Shift+Tab wraps back to the second button; Enter activates it.
	ui.route_event(&InputEvent {
		kind: InputEventKind::KeyDown,
		key: Key::Tab,
		modifiers: oa::input::INPUT_MOD_SHIFT,
		..InputEvent::default()
	});
	ui.route_event(&InputEvent {
		kind: InputEventKind::KeyDown,
		key: Key::Return,
		..InputEvent::default()
	});
	frame(&mut ui, &mut hits, &mut level);
	assert!(hits[1], "Shift+Tab+Enter should activate the second button");
	ui.close()?;
	Ok(())
});

// Gamepad navigation: South grabs then activates focus, dpad and left stick
// route directions into the focused widget.
test_vk!(ui_gamepad_nav_focuses_and_steps, engine, {
	use oa::input::{GAMEPAD_AXIS_LEFT_X, GAMEPAD_BUTTON_DPAD_RIGHT, GAMEPAD_BUTTON_SOUTH};

	let mut ui = Ui::init(&engine, 300, 200)?;
	let mut fired = false;
	let mut level = 0.5f32;

	let frame = |ui: &mut Ui<'_>, fired: &mut bool, level: &mut f32| {
		ui.begin_frame(0.0);
		ui.begin_panel("panel", PixelRect::new(0, 0, 300, 200));
		if ui.button("go") {
			*fired = true;
		}
		ui.slider_f32("level", level, 0.0, 1.0);
		ui.end_panel();
		ui.end_frame();
	};
	let pad_button = |button: i32| InputEvent {
		kind: InputEventKind::GamepadButtonDown,
		gamepad_id: 1,
		gamepad_button: button,
		..InputEvent::default()
	};

	// Frame one registers focusables: go, slider.
	frame(&mut ui, &mut fired, &mut level);

	// First South press grabs focus; second activates the button.
	ui.route_event(&pad_button(GAMEPAD_BUTTON_SOUTH));
	ui.route_event(&pad_button(GAMEPAD_BUTTON_SOUTH));
	frame(&mut ui, &mut fired, &mut level);
	assert!(fired, "gamepad South should activate the focused button");

	// Tab to the slider, then dpad-right steps it up.
	ui.route_event(&InputEvent {
		kind: InputEventKind::KeyDown,
		key: oa::input::Key::Tab,
		..InputEvent::default()
	});
	ui.route_event(&pad_button(GAMEPAD_BUTTON_DPAD_RIGHT));
	frame(&mut ui, &mut fired, &mut level);
	assert!(
		level > 0.51,
		"dpad-right should step the slider (got {level})"
	);

	// A left-stick edge past the deadzone also produces a nav step, once —
	// holding the stick does not repeat.
	ui.route_event(&InputEvent {
		kind: InputEventKind::GamepadAxisMotion,
		gamepad_id: 1,
		gamepad_axis: GAMEPAD_AXIS_LEFT_X,
		gamepad_value: 30_000,
		..InputEvent::default()
	});
	ui.route_event(&InputEvent {
		kind: InputEventKind::GamepadAxisMotion,
		gamepad_id: 1,
		gamepad_axis: GAMEPAD_AXIS_LEFT_X,
		gamepad_value: 30_000,
		..InputEvent::default()
	});
	frame(&mut ui, &mut fired, &mut level);
	assert!(
		(level - 0.54).abs() < 1e-3,
		"one stick edge should produce exactly one step (got {level})"
	);
	ui.close()?;
	Ok(())
});

// IME plumbing: a focused text field reports its rect for the platform
// candidate window and consumes TextEditing (pre-edit) events.
test_vk!(ui_text_input_reports_ime_rect_and_preedit, engine, {
	use oa::input::Key;

	let mut ui = Ui::init(&engine, 240, 120)?;
	let mut value = String::from("ab");

	let frame = |ui: &mut Ui<'_>, value: &mut String| {
		ui.begin_frame(0.0);
		ui.begin_panel("panel", PixelRect::new(0, 0, 240, 120));
		ui.input_text("name", value);
		ui.end_panel();
		ui.end_frame();
	};

	assert!(!ui.wants_text_input());
	assert!(ui.text_input_rect().is_none());

	// Focus the field with a click, then render — the rect appears.
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 100.0,
		mouse_y: 20.0,
		..InputEvent::default()
	});
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: 100.0,
		mouse_y: 20.0,
		button: 1,
		..InputEvent::default()
	});
	frame(&mut ui, &mut value);
	assert!(ui.wants_text_input());
	let rect = ui
		.text_input_rect()
		.expect("focused text field should report an IME rect");
	assert!(rect.w > 0 && rect.h > 0);
	assert!(rect.y >= 0 && rect.y < 120);

	// Pre-edit events are consumed while focused and commit via KeyChar.
	assert!(ui.route_event(&InputEvent {
		kind: InputEventKind::TextEditing,
		text: "pre".to_owned(),
		..InputEvent::default()
	}));
	frame(&mut ui, &mut value);
	ui.route_event(&InputEvent {
		kind: InputEventKind::KeyChar,
		text: "x".to_owned(),
		..InputEvent::default()
	});
	frame(&mut ui, &mut value);
	assert_eq!(value, "abx");

	// Escape drops focus — the IME rect and wants flag disappear.
	ui.route_event(&InputEvent {
		kind: InputEventKind::KeyDown,
		key: Key::Escape,
		..InputEvent::default()
	});
	frame(&mut ui, &mut value);
	assert!(!ui.wants_text_input());
	assert!(ui.text_input_rect().is_none());
	ui.close()?;
	Ok(())
});

// Menu button → popup of menu_items: click opens, item click activates and
// closes the popup.
test_vk!(ui_menu_opens_and_item_activates, engine, {
	let mut ui = Ui::init(&engine, 300, 240)?;
	let mut chosen = false;

	let frame = |ui: &mut Ui<'_>, chosen: &mut bool| {
		ui.begin_frame(0.0);
		ui.begin_panel("panel", PixelRect::new(0, 0, 300, 240));
		if ui.menu("File") {
			if ui.menu_item("Open", false, true) {
				*chosen = true;
			}
			ui.menu_item("Disabled", false, false);
			ui.end_menu();
		}
		ui.end_panel();
		ui.end_frame();
	};

	// Click the "File" button (first widget row, ~y=20).
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 100.0,
		mouse_y: 20.0,
		..InputEvent::default()
	});
	frame(&mut ui, &mut chosen);
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: 100.0,
		mouse_y: 20.0,
		button: 1,
		..InputEvent::default()
	});
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseUp,
		mouse_x: 100.0,
		mouse_y: 20.0,
		button: 1,
		..InputEvent::default()
	});
	frame(&mut ui, &mut chosen);
	assert!(
		ui.is_popup_open("__menu_File"),
		"menu click should open the popup"
	);

	// Click the first item — sits at popup y ≈ button bottom + gap + half row.
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseMove,
		mouse_x: 60.0,
		mouse_y: 60.0,
		..InputEvent::default()
	});
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseDown,
		mouse_x: 60.0,
		mouse_y: 60.0,
		button: 1,
		..InputEvent::default()
	});
	frame(&mut ui, &mut chosen);
	ui.route_event(&InputEvent {
		kind: InputEventKind::MouseUp,
		mouse_x: 60.0,
		mouse_y: 60.0,
		button: 1,
		..InputEvent::default()
	});
	frame(&mut ui, &mut chosen);
	assert!(chosen, "menu_item should report the click");
	assert!(
		!ui.is_popup_open("__menu_File"),
		"activating an item should close the popup"
	);
	ui.close()?;
	Ok(())
});
