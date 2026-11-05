use oa::{
	Color,
	input::{InputEvent, InputEventKind},
	plot::{Figure, FigureConfig},
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

	let mut buffer = DetectionBuffer::create_host_upload(&engine, 4)?;
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

	let mut buffer = GlyphBuffer::create_host_upload(&engine, 4)?;
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
		let source = oa::Matrix::new_from_f32(&engine, [4], &[1.0; 4])?;
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

	let texture = texture_from_rgba8(&engine, &[255_u8; 4 * 4 * 4], 4, 4)?;
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
	let texture = frame.render(&engine)?;
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
	let texture = frame.render_with_rgba(&engine, &blue_base)?;
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
	assert!(frame.render_with_rgba(&engine, &[0_u8; 16]).is_err());
	Ok(())
});

test_vk!(cv_frame_render_rejects_invalid_state, engine, {
	let mut frame = CvFrame::default();
	assert!(frame.render(&engine).is_err());
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
	assert!(frame.render(&engine).is_err());
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
	assert!(frame.render(&engine).is_err());
	frame.clear_overlays();
	assert!(frame.render(&engine).is_ok());
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
		&engine,
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
		&engine,
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
		ImagePlanes::from_planes(&engine, &[], &[], 4, 3).is_err(),
		"zero planes must fail"
	);
	assert!(
		ImagePlanes::from_planes(&engine, &[&r, &g, &b, &r, &g], &[ImageDtype::U8; 5], 4, 3).is_err(),
		"five planes must fail"
	);
	assert!(
		ImagePlanes::from_planes(&engine, &[&r, &g], &[ImageDtype::U8], 4, 3).is_err(),
		"dtype length mismatch must fail"
	);
	assert!(
		ImagePlanes::from_planes(&engine, &[&r], &[ImageDtype::U8], 0, 3).is_err(),
		"zero width must fail"
	);
	assert!(
		ImagePlanes::from_planes(&engine, &[&r], &[ImageDtype::U8], 5, 3).is_err(),
		"byte count mismatch must fail"
	);
	// U16 expects 2 bytes per element; a U8-sized span must fail.
	assert!(ImagePlanes::from_planes(&engine, &[&r], &[ImageDtype::U16], 4, 3).is_err());
	Ok(())
});

test_vk!(image_planes_load_file_decodes_rgba_png, engine, {
	let path = std::env::temp_dir().join(format!("oa-planes-{}.png", std::process::id()));
	// 2x1 red/green pair in RGBA8.
	oa::image::save_rgba_file(&[255, 0, 0, 255, 0, 255, 0, 128], 2, 1, &path, 90)?;
	let planes = ImagePlanes::load_file(&engine, &path)?;
	let _ = std::fs::remove_file(&path);
	assert!(planes.is_valid());
	assert_eq!(planes.channel_count(), 4);
	assert_eq!(planes.width(), 2);
	assert_eq!(planes.height(), 1);
	assert_eq!(planes.plane_dtype(0), ImageDtype::U8);
	assert!(ImagePlanes::load_file(&engine, "").is_err());
	assert!(ImagePlanes::load_file(&engine, path.with_extension("missing")).is_err());
	Ok(())
});

test_vk!(ui_image_planar_composites_rgb_and_gray, engine, {
	let mut ui = Ui::init(&engine, 32, 32)?;
	// 4x4 RGB: R=255, G=128, B=0 everywhere.
	let planes = ImagePlanes::from_planes(
		&engine,
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
	let gray = ImagePlanes::from_planes(&engine, &[&[200_u8; 16]], &[ImageDtype::U8], 4, 4)?;
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
	let texture = oa::render::texture_from_rgba8(&engine, &[250_u8; 4 * 4 * 4], 4, 4)?;
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
