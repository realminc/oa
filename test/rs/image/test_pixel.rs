use oa::{ErrorKind, Image, ImageFormat, ImageLayout, Matrix, image::NormalizationParams};

test_vk!(
	pointwise_pixel_family_matches_independent_scalar_oracles,
	engine,
	{
		let values = [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0];
		let image = Image::new(
			Matrix::from_f32(&engine, [1, 1, 1, values.len()], &values)?,
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

test_vk!(pointwise_pixel_family_rejects_invalid_parameters, engine, {
	let image = Image::new(
		Matrix::from_f32(&engine, [1, 1, 1], &[0.5])?,
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
});

test_vk!(
	channel_composite_and_color_twist_operations_match_oracles,
	engine,
	{
		let rgb_values = [
			1.0, 0.0, 0.5, 0.25, // red
			0.0, 1.0, 0.5, 0.25, // green
			0.0, 0.0, 0.5, 0.25, // blue
		];
		let rgb = Image::new(
			Matrix::from_f32(&engine, [1, 3, 2, 2], &rgb_values)?,
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
			Matrix::from_f32(&engine, [1, 3, 2, 2], &[0.0; 12])?,
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
			Matrix::from_f32(&engine, [1, 1, 2, 2], &[0.0, 0.25, 0.75, 1.0])?,
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
		let transform = Matrix::from_f32(
			&engine,
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
	engine,
	{
		let image = Image::new(
			Matrix::from_f32(
				&engine,
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

test_vk!(normalize_matches_per_channel_oracle, engine, {
	let image = Image::new(
		Matrix::from_f32(&engine, [3, 1, 2], &[1.0, 3.0, 10.0, 14.0, -2.0, 4.0])?,
		ImageLayout::Chw,
		ImageFormat::Rgb,
	)?;
	let normalized = oa::image::normalize(
		&image,
		NormalizationParams {
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
	engine,
	{
		let rgb = Image::new(
			Matrix::from_f32(
				&engine,
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
			NormalizationParams {
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
		let mask = Matrix::from_slice(&engine, [1, 2, 2], &[0_i32, 1, -1, 3])?;
		let palette = Matrix::from_f32(&engine, [2, 3], &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0])?;
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
