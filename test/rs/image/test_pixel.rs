use oa::{ErrorKind, Image, ImageFormat, ImageLayout, Matrix, image::NormalizationConfig};

#[test]
fn normalization_defaults_preserve_the_identity_transform() {
	let parameters = NormalizationConfig::default();
	assert_eq!(parameters.mean, [0.0; 3]);
	assert_eq!(parameters.std, [1.0; 3]);
}

test_vk!(
	captured_image_invert_clamp_matches_standalone_dispatches,
	runtime,
	{
		let samples: Vec<f32> = (0..273).map(|i| (i % 31 - 15) as f32 / 16.0).collect();
		let input = Image::new(
			Matrix::new_from_f32([1, 1, 13, 21], &samples)?,
			ImageLayout::Nchw,
			ImageFormat::Gray,
		)?;
		let inverted = input.invert(0.25)?;
		inverted.as_matrix().read_f32()?;
		let expected = inverted.clamp(-0.375, 0.625)?.as_matrix().read_f32()?;
		let (plan, output) = runtime.capture(|| input.invert(0.25)?.clamp(-0.375, 0.625))?;
		let report: serde_json::Value =
			serde_json::from_str(&plan.debug_report_json("image invert and clamp"))
				.expect("valid execution graph report");
		assert_eq!(report["stats"]["dispatches"], 1);
		assert_eq!(
			report["nodes"][0]["semantic_operations"],
			serde_json::json!([0, 1])
		);
		assert_eq!(output.layout(), input.layout());
		assert_eq!(output.format(), input.format());
		runtime.submit(&plan)?.wait()?;
		assert_eq!(output.as_matrix().read_f32()?, expected);
		Ok(())
	}
);

test_vk!(
	pointwise_pixel_family_matches_independent_scalar_oracles,
	runtime,
	{
		let values = [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0];
		let image = Image::new(
			Matrix::new_from_f32([1, 1, 1, values.len()], &values)?,
			ImageLayout::Nchw,
			ImageFormat::Gray,
		)?;
		let cases = [
			(
				oa::image::threshold_binary(&image, 0.5, 3.0)?,
				vec![0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0],
			),
			(
				oa::image::threshold_binary_inv(&image, 0.5, 3.0)?,
				vec![3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0],
			),
			(
				oa::image::threshold_truncate(&image, 0.5)?,
				vec![-1.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.5],
			),
			(
				oa::image::threshold_to_zero(&image, 0.5)?,
				vec![0.0, 0.0, 0.0, 0.0, 0.75, 1.0, 2.0],
			),
			(
				oa::image::threshold_to_zero_inv(&image, 0.5)?,
				vec![-1.0, 0.0, 0.25, 0.5, 0.0, 0.0, 0.0],
			),
			(
				oa::image::in_range(&image, 0.25, 0.75, 2.0)?,
				vec![0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
			),
			(
				oa::image::clamp(&image, 0.0, 1.0)?,
				vec![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0],
			),
			(
				oa::image::invert(&image, 1.0)?,
				values.iter().map(|value| 1.0 - value).collect(),
			),
			(
				oa::image::brightness_contrast(&image, 0.25, 2.0)?,
				values.iter().map(|value| value * 2.0 + 0.25).collect(),
			),
			(
				oa::image::gamma_contrast(&image, 2.0, 0.5)?,
				values
					.iter()
					.map(|value| value.max(0.0).powi(2) * 0.5)
					.collect(),
			),
			(
				oa::image::solarize(&image, 0.5, 1.0)?,
				vec![-1.0, 0.0, 0.25, 0.5, 0.25, 0.0, -1.0],
			),
			(
				oa::image::posterize(&image, 3, 0.0, 1.0)?,
				vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0],
			),
		];
		for (output, expected) in cases {
			assert_eq!(output.layout(), image.layout());
			assert_eq!(output.format(), image.format());
			for (actual, expected) in output.as_matrix().read_f32()?.iter().zip(expected) {
				assert!(
					(actual - expected).abs() <= 1e-6,
					"expected {expected}, found {actual}"
				);
			}
		}
		Ok(())
	}
);

test_vk!(
	pointwise_pixel_family_rejects_invalid_parameters,
	runtime,
	{
		let image = Image::new(
			Matrix::new_from_f32([1, 1, 1], &[0.5])?,
			ImageLayout::Chw,
			ImageFormat::Gray,
		)?;
		for error in [
			oa::image::threshold_binary(&image, f32::NAN, 1.0)
				.err()
				.expect("NaN accepted"),
			oa::image::in_range(&image, 2.0, 1.0, 1.0)
				.err()
				.expect("reversed range accepted"),
			oa::image::gamma_contrast(&image, 0.0, 1.0)
				.err()
				.expect("zero gamma accepted"),
			oa::image::posterize(&image, 1, 0.0, 1.0)
				.err()
				.expect("one level accepted"),
		] {
			assert_eq!(error.kind(), ErrorKind::InvalidArgument);
		}
		Ok(())
	}
);

test_vk!(
	channel_composite_and_color_twist_operations_match_oracles,
	runtime,
	{
		let rgb_values = [
			1.0, 0.0, 0.5, 0.25, // red
			0.0, 1.0, 0.5, 0.25, // green
			0.0, 0.0, 0.5, 0.25, // blue
		];
		let rgb = Image::new(
			Matrix::new_from_f32([1, 3, 2, 2], &rgb_values)?,
			ImageLayout::Nchw,
			ImageFormat::Rgb,
		)?;
		let gray = oa::image::grayscale(&rgb)?;
		assert_eq!(gray.format(), ImageFormat::Gray);
		assert_eq!(gray.as_matrix().shape(), [1, 1, 2, 2]);
		let expected_gray = [0.2126, 0.7152, 0.5, 0.25];
		for (actual, expected) in gray.as_matrix().read_f32()?.iter().zip(expected_gray) {
			assert!((actual - expected).abs() <= 1e-6);
		}

		let zeros = Image::new(
			Matrix::new_from_f32([1, 3, 2, 2], &[0.0; 12])?,
			ImageLayout::Nchw,
			ImageFormat::Rgb,
		)?;
		assert_eq!(
			oa::image::alpha_blend(&rgb, &zeros, 0.25)?
				.as_matrix()
				.read_f32()?,
			rgb_values.map(|value| value * 0.75)
		);
		let mask = Image::new(
			Matrix::new_from_f32([1, 1, 2, 2], &[0.0, 0.25, 0.75, 1.0])?,
			ImageLayout::Nchw,
			ImageFormat::Gray,
		)?;
		let composited = oa::image::composite(&rgb, &zeros, &mask)?
			.as_matrix()
			.read_f32()?;
		for channel in 0..3 {
			for pixel in 0..4 {
				let index = channel * 4 + pixel;
				let expected = rgb_values[index] * (1.0 - [0.0, 0.25, 0.75, 1.0][pixel]);
				assert!((composited[index] - expected).abs() <= 1e-6);
			}
		}
		assert_eq!(
			oa::image::erase(&rgb, 1, 0, 3, 1, -2.0)?
				.as_matrix()
				.read_f32()?,
			[
				1.0, -2.0, 0.5, 0.25, 0.0, -2.0, 0.5, 0.25, 0.0, -2.0, 0.5, 0.25
			]
		);
		let transform = Matrix::new_from_f32(
			[3, 4],
			&[0.0, 1.0, 0.0, 0.1, 0.0, 0.0, 1.0, 0.2, 1.0, 0.0, 0.0, 0.3],
		)?;
		let twisted = oa::image::color_twist(&rgb, &transform)?
			.as_matrix()
			.read_f32()?;
		for pixel in 0..4 {
			assert!((twisted[pixel] - (rgb_values[4 + pixel] + 0.1)).abs() <= 1e-6);
			assert!((twisted[4 + pixel] - (rgb_values[8 + pixel] + 0.2)).abs() <= 1e-6);
			assert!((twisted[8 + pixel] - (rgb_values[pixel] + 0.3)).abs() <= 1e-6);
		}
		Ok(())
	}
);

test_vk!(
	channel_reorder_and_seeded_noise_preserve_image_semantics,
	runtime,
	{
		let image = Image::new(
			Matrix::new_from_f32(
				[3, 1, 3],
				&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
			)?,
			ImageLayout::Chw,
			ImageFormat::Rgb,
		)?;
		let reordered = oa::image::channel_reorder(&image, [2, 1, 0, 0], ImageFormat::Bgr)?;
		assert_eq!(reordered.format(), ImageFormat::Bgr);
		assert_eq!(
			reordered.as_matrix().read_f32()?,
			[100.0, 200.0, 300.0, 10.0, 20.0, 30.0, 1.0, 2.0, 3.0]
		);
		assert_eq!(
			oa::image::gaussian_noise(&image, 0.25, 0.0, 99)?
				.as_matrix()
				.read_f32()?,
			[
				1.25, 2.25, 3.25, 10.25, 20.25, 30.25, 100.25, 200.25, 300.25
			]
		);
		assert_eq!(
			oa::image::salt_pepper_noise(&image, 0.0, 7.0, -7.0, 99)?
				.as_matrix()
				.read_f32()?,
			image.as_matrix().read_f32()?
		);
		let first = oa::image::gaussian_noise(&image, 0.0, 1.0, 0x1234_5678_9abc_def0)?;
		let second = oa::image::gaussian_noise(&image, 0.0, 1.0, 0x1234_5678_9abc_def0)?;
		assert_eq!(
			first.as_matrix().read_f32()?,
			second.as_matrix().read_f32()?
		);
		Ok(())
	}
);

test_vk!(normalize_matches_per_channel_oracle, runtime, {
	let image = Image::new(
		Matrix::new_from_f32([3, 1, 2], &[1.0, 3.0, 10.0, 14.0, -2.0, 4.0])?,
		ImageLayout::Chw,
		ImageFormat::Rgb,
	)?;
	let normalized = oa::image::normalize(
		&image,
		NormalizationConfig {
			mean: [1.0, 2.0, -2.0],
			std: [2.0, 4.0, 3.0],
		},
	)?;
	assert_eq!(
		normalized.as_matrix().read_f32()?,
		[0.0, 1.0, 2.0, 3.0, 0.0, 2.0]
	);
	assert_eq!(normalized.format(), ImageFormat::Rgb);
	Ok(())
});

test_vk!(
	semantic_color_compositions_preserve_typed_image_contracts,
	runtime,
	{
		let rgb = Image::new(
			Matrix::new_from_f32(
				[1, 3, 2, 2],
				&[
					1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
				],
			)?,
			ImageLayout::Nchw,
			ImageFormat::Rgb,
		)?;
		let bgr = oa::image::convert_color(&rgb, ImageFormat::Bgr)?;
		assert_eq!(bgr.format(), ImageFormat::Bgr);
		assert_eq!(
			bgr.as_matrix().read_f32()?,
			[
				100.0, 200.0, 300.0, 400.0, 10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0
			]
		);
		let composed = oa::image::resize_normalize(
			&rgb,
			2,
			2,
			NormalizationConfig {
				mean: [1.0, 2.0, 4.0],
				std: [1.0, 2.0, 4.0],
			},
		)?;
		assert_eq!(
			composed.as_matrix().read_f32()?,
			[
				0.0, 1.0, 2.0, 3.0, 4.0, 9.0, 14.0, 19.0, 24.0, 49.0, 74.0, 99.0
			]
		);
		let mask = Matrix::new_from_slice([1, 2, 2], &[0_i32, 1, -1, 3])?;
		let palette = Matrix::new_from_f32([2, 3], &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0])?;
		let overlay = oa::image::segmentation_overlay(&rgb, &mask, &palette, 0.5)?;
		assert_eq!(
			overlay.as_matrix().read_f32()?,
			[
				0.5, 1.5, 3.0, 4.0, 5.5, 10.0, 30.0, 40.0, 50.0, 100.5, 300.0, 400.0
			]
		);
		Ok(())
	}
);

test_vk!(cvt_color_rgb_to_gray_matches_luma_oracle, runtime, {
	let rgb = Image::new(
		Matrix::new_from_f32([1, 3, 1, 2], &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0])?,
		ImageLayout::Nchw,
		ImageFormat::Rgb,
	)?;
	let gray = oa::image::cvt_color(&rgb, oa::image::ColorConversion::RgbToGray)?;
	assert_eq!(gray.format(), ImageFormat::Gray);
	assert_eq!(gray.as_matrix().shape(), &[1, 1, 1, 2]);
	let result = gray.as_matrix().read_f32()?;
	assert!((result[0] - 0.299).abs() < 1e-5);
	assert!((result[1] - 0.701).abs() < 1e-5);
	Ok(())
});

test_vk!(cvt_color_rgb_hsv_round_trip_recovers_source, runtime, {
	let rgb = Image::new(
		Matrix::new_from_f32(
			[1, 3, 1, 3],
			&[0.8, 0.4, 0.1, 0.2, 0.9, 0.3, 0.5, 0.25, 0.75],
		)?,
		ImageLayout::Nchw,
		ImageFormat::Rgb,
	)?;
	let hsv = oa::image::cvt_color(&rgb, oa::image::ColorConversion::RgbToHsv)?;
	assert_eq!(hsv.as_matrix().shape(), &[1, 3, 1, 3]);
	let recovered = oa::image::cvt_color(&hsv, oa::image::ColorConversion::HsvToRgb)?;
	let source = rgb.as_matrix().read_f32()?;
	let round_trip = recovered.as_matrix().read_f32()?;
	for (expected, actual) in source.iter().zip(round_trip.iter()) {
		assert!((expected - actual).abs() < 1e-4, "{expected} != {actual}");
	}
	Ok(())
});

test_vk!(cvt_color_bgr_to_gray_uses_swapped_weights, runtime, {
	let bgr = Image::new(
		Matrix::new_from_f32([1, 3, 1, 1], &[1.0, 0.0, 0.0])?,
		ImageLayout::Nchw,
		ImageFormat::Bgr,
	)?;
	let gray = oa::image::cvt_color(&bgr, oa::image::ColorConversion::BgrToGray)?;
	let result = gray.as_matrix().read_f32()?;
	// Stored channel zero is blue under a BGR conversion.
	assert!((result[0] - 0.114).abs() < 1e-5);
	Ok(())
});

test_vk!(cvt_color_rejects_a_channel_count_mismatch, runtime, {
	let rgb = Image::new(
		Matrix::new_from_f32([1, 3, 1, 1], &[0.5; 3])?,
		ImageLayout::Nchw,
		ImageFormat::Rgb,
	)?;
	let error = oa::image::cvt_color(&rgb, oa::image::ColorConversion::GrayToRgb)
		.err()
		.unwrap();
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	Ok(())
});

test_vk!(apply_color_map_jet_maps_endpoints, runtime, {
	let gray = Image::new(
		Matrix::new_from_f32([1, 1, 1, 2], &[0.0, 1.0])?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)?;
	let mapped = oa::image::apply_color_map(&gray, oa::image::ColorMap::Jet)?;
	assert_eq!(mapped.format(), ImageFormat::Rgb);
	assert_eq!(mapped.as_matrix().shape(), &[1, 3, 1, 2]);
	let result = mapped.as_matrix().read_f32()?;
	// Planar RGB: Jet at 0.0 -> (0, 0, 0.5); at 1.0 -> (0.5, 0, 0).
	let expected = [0.0, 0.5, 0.0, 0.0, 0.5, 0.0];
	for (expected, actual) in expected.iter().zip(result.iter()) {
		assert!((expected - actual).abs() < 1e-5, "{expected} != {actual}");
	}
	Ok(())
});

test_vk!(apply_color_map_requires_a_single_channel, runtime, {
	let rgb = Image::new(
		Matrix::new_from_f32([1, 3, 1, 1], &[0.5; 3])?,
		ImageLayout::Nchw,
		ImageFormat::Rgb,
	)?;
	let error = oa::image::apply_color_map(&rgb, oa::image::ColorMap::Hot)
		.err()
		.unwrap();
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	Ok(())
});

test_vk!(flood_fill_grows_bounded_region, runtime, {
	// Left half = 0.4, right half = 0.9. Floating-range flood from (2, 2)
	// with lo/up = 0.05 fills only the left region.
	let width = 8;
	let height = 6;
	let mut plane = vec![0.0f32; width * height];
	for y in 0..height {
		for x in 0..width {
			plane[y * width + x] = if x < 4 { 0.4 } else { 0.9 };
		}
	}
	let image = Image::new(
		Matrix::new_from_f32_on(&runtime, [1, 1, height, width], &plane)?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)?;
	let filled = oa::image::flood_fill(
		&image,
		2,
		2,
		1.0,
		0.05,
		0.05,
		oa::image::Connectivity::Four,
		0,
	)?;
	let values = filled.as_matrix().read_f32()?;
	for y in 0..height {
		for x in 0..width {
			let expected = if x < 4 { 1.0 } else { 0.9 };
			assert_eq!(
				values[y * width + x],
				expected,
				"({x}, {y}) expected {expected}"
			);
		}
	}
	// Zero-diff seed-only fill on the right region propagates everywhere
	// because the region is constant.
	let filled_right = oa::image::flood_fill(
		&image,
		6,
		3,
		0.25,
		0.0,
		0.0,
		oa::image::Connectivity::Four,
		0,
	)?;
	let values = filled_right.as_matrix().read_f32()?;
	assert_eq!(values[3 * width + 4], 0.25);
	assert_eq!(values[0], 0.4);
	Ok(())
});

test_vk!(draw_line_marks_segment_band, runtime, {
	let width = 8;
	let height = 6;
	let image = Image::new(
		Matrix::new_from_f32_on(&runtime, [1, 1, height, width], &vec![0.0; width * height])?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)?;
	// Horizontal line y=2, x 1..6 with thickness 1 -> band of half a pixel.
	let out = oa::image::draw_line(&image, 1.5, 2.5, 6.5, 2.5, 1.0, 1.0)?;
	let values = out.as_matrix().read_f32()?;
	for x in 0..width {
		let expected = if (1..=6).contains(&x) { 1.0 } else { 0.0 };
		assert_eq!(values[2 * width + x], expected, "row2 x{x}");
		assert_eq!(values[x], 0.0, "row0 x{x}");
		assert_eq!(values[5 * width + x], 0.0, "row5 x{x}");
	}
	Ok(())
});

test_vk!(draw_rectangle_border_and_fill, runtime, {
	let width = 8;
	let height = 6;
	let image = Image::new(
		Matrix::new_from_f32_on(&runtime, [1, 1, height, width], &vec![0.0; width * height])?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)?;
	let out = oa::image::draw_rectangle(&image, 2.0, 1.0, 4.0, 3.0, 1.0, 1.0)?;
	let values = out.as_matrix().read_f32()?;
	for y in 0..height {
		for x in 0..width {
			let border =
				(2..=5).contains(&x) && (1..=3).contains(&y) && (x == 2 || x == 5 || y == 1 || y == 3);
			assert_eq!(
				values[y * width + x],
				if border { 1.0 } else { 0.0 },
				"({x},{y})"
			);
		}
	}
	let filled = oa::image::draw_rectangle(&image, 2.0, 1.0, 4.0, 3.0, 1.0, -1.0)?;
	let values = filled.as_matrix().read_f32()?;
	for y in 0..height {
		for x in 0..width {
			let inside = (2..=5).contains(&x) && (1..=3).contains(&y);
			assert_eq!(
				values[y * width + x],
				if inside { 1.0 } else { 0.0 },
				"({x},{y})"
			);
		}
	}
	Ok(())
});

test_vk!(draw_circle_disk_and_ring, runtime, {
	let width = 9;
	let height = 9;
	let image = Image::new(
		Matrix::new_from_f32_on(&runtime, [1, 1, height, width], &vec![0.0; width * height])?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)?;
	// Filled disk radius 2 at (4.5, 4.5): covers the center cross exactly.
	let disk = oa::image::draw_circle(&image, 4.5, 4.5, 2.0, 1.0, -1.0)?;
	let values = disk.as_matrix().read_f32()?;
	for y in 0..height {
		for x in 0..width {
			let d = ((x as f32 + 0.5 - 4.5).powi(2) + (y as f32 + 0.5 - 4.5).powi(2)).sqrt();
			let expected = if d <= 2.0 { 1.0 } else { 0.0 };
			assert_eq!(values[y * width + x], expected, "disk ({x},{y})");
		}
	}
	// Ring radius 3 thickness 1: band |d - 3| <= 0.5.
	let ring = oa::image::draw_circle(&image, 4.5, 4.5, 3.0, 1.0, 1.0)?;
	let values = ring.as_matrix().read_f32()?;
	for y in 0..height {
		for x in 0..width {
			let d = ((x as f32 + 0.5 - 4.5).powi(2) + (y as f32 + 0.5 - 4.5).powi(2)).sqrt();
			let expected = if (d - 3.0).abs() <= 0.5 { 1.0 } else { 0.0 };
			assert_eq!(values[y * width + x], expected, "ring ({x},{y})");
		}
	}
	Ok(())
});

test_vk!(fill_polygon_fills_triangle_interior, runtime, {
	let width = 8;
	let height = 8;
	let image = Image::new(
		Matrix::new_from_f32_on(&runtime, [1, 1, height, width], &vec![0.0; width * height])?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)?;
	// Triangle (1,1) (6,1) (1,6): even-odd fill covers x<=5-... interior points.
	let points = Matrix::new_from_f32_on(&runtime, [3, 2], &[1.0, 1.0, 6.0, 1.0, 1.0, 6.0])?;
	let out = oa::image::fill_polygon(&image, &points, 1.0)?;
	let values = out.as_matrix().read_f32()?;
	// Host even-odd oracle over the same rule.
	for y in 0..height {
		for x in 0..width {
			let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
			let verts = [(1.0f32, 1.0f32), (6.0, 1.0), (1.0, 6.0)];
			let mut inside = false;
			let mut j = 2;
			for i in 0..3 {
				let (xi, yi) = verts[i];
				let (xj, yj) = verts[j];
				if (yi > py) != (yj > py) && px < xj + (xi - xj) * (py - yj) / (yi - yj) {
					inside = !inside;
				}
				j = i;
			}
			assert_eq!(
				values[y * width + x],
				if inside { 1.0 } else { 0.0 },
				"({x},{y})"
			);
		}
	}
	Ok(())
});
