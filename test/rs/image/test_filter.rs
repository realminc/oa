use oa::{ErrorKind, Image, ImageFormat, ImageLayout, Matrix, image::BorderMode};

test_vk!(
	complete_filter_family_matches_identity_and_fixed_kernel_oracles,
	engine,
	{
		let values = (0..9).map(|value| value as f32).collect::<Vec<_>>();
		let image = Image::new(
			Matrix::from_f32(&engine, [1, 1, 3, 3], &values)?,
			ImageLayout::Nchw,
			ImageFormat::Gray,
		)?;
		let one_2d = Matrix::from_f32(&engine, [1, 1], &[1.0])?;
		let one_1d = Matrix::from_f32(&engine, [1], &[1.0])?;
		let identity_outputs = [
			oa::image::convolve_2d(&image, &one_2d, BorderMode::Constant, 0.0)?,
			oa::image::separable_convolve_2d(&image, &one_1d, &one_1d, BorderMode::Constant, 0.0)?,
			oa::image::average_blur(&image, 1, 1, BorderMode::Reflect101)?,
			oa::image::erode(&image, 1, 1, BorderMode::Constant, -1.0)?,
			oa::image::dilate(&image, 1, 1, BorderMode::Constant, -1.0)?,
			oa::image::morphology_open(&image, 1, 1, BorderMode::Constant, -1.0)?,
			oa::image::morphology_close(&image, 1, 1, BorderMode::Constant, -1.0)?,
			oa::image::gaussian_blur(&image, 1.0, 1)?,
			oa::image::median_blur(&image, 1, BorderMode::Wrap)?,
			oa::image::bilateral_filter(&image, 1, 0.1, 1.0, BorderMode::Wrap)?,
			oa::image::unsharp_mask(&image, 1.0, 2.0, 1)?,
		];
		for output in identity_outputs {
			assert_eq!(output.as_matrix().read_f32()?, values);
		}
		for output in [
			oa::image::morphology_gradient(&image, 1, 1, BorderMode::Constant, 0.0)?,
			oa::image::morphology_top_hat(&image, 1, 1, BorderMode::Constant, 0.0)?,
			oa::image::morphology_black_hat(&image, 1, 1, BorderMode::Constant, 0.0)?,
		] {
			assert_eq!(output.as_matrix().read_f32()?, [0.0; 9]);
		}
		assert_eq!(
			oa::image::adaptive_threshold_mean(&image, 1, 0.1, 2.0, BorderMode::Replicate)?
				.as_matrix()
				.read_f32()?,
			[2.0; 9]
		);
		assert_eq!(
			oa::image::adaptive_threshold_gaussian(&image, 1, 0.1, 3.0, 0.0, BorderMode::Replicate,)?
				.as_matrix()
				.read_f32()?,
			[3.0; 9]
		);

		let fixed = [
			(
				oa::image::sobel(&image, 1, 0, BorderMode::Constant)?,
				[6.0, 6.0, -6.0, 16.0, 8.0, -16.0, 18.0, 6.0, -18.0],
			),
			(
				oa::image::scharr(&image, 1, 0, BorderMode::Constant)?,
				[22.0, 26.0, -22.0, 64.0, 32.0, -64.0, 82.0, 26.0, -82.0],
			),
			(
				oa::image::laplacian(&image, BorderMode::Constant)?,
				[4.0, 2.0, -2.0, -2.0, 0.0, -6.0, -14.0, -10.0, -20.0],
			),
		];
		for (output, expected) in fixed {
			assert_eq!(output.as_matrix().read_f32()?, expected);
		}
		assert_eq!(
			oa::image::sharpen(&image, 0.0, BorderMode::Constant)?
				.as_matrix()
				.read_f32()?,
			values
		);
		Ok(())
	}
);

test_vk!(
	filter_family_rejects_invalid_kernels_and_parameters,
	engine,
	{
		let image = Image::new(
			Matrix::from_f32(&engine, [1, 2, 2], &[0.0; 4])?,
			ImageLayout::Chw,
			ImageFormat::Gray,
		)?;
		let even = Matrix::from_f32(&engine, [2, 2], &[1.0; 4])?;
		for error in [
			oa::image::convolve_2d(&image, &even, BorderMode::Constant, 0.0)
				.err()
				.expect("even kernel accepted"),
			oa::image::average_blur(&image, 2, 3, BorderMode::Constant)
				.err()
				.expect("even average kernel accepted"),
			oa::image::sobel(&image, 1, 1, BorderMode::Constant)
				.err()
				.expect("mixed derivative accepted"),
			oa::image::gaussian_blur(&image, 0.0, 3)
				.err()
				.expect("zero sigma accepted"),
			oa::image::median_blur(&image, 17, BorderMode::Constant)
				.err()
				.expect("oversize neighborhood accepted"),
			oa::image::bilateral_filter(&image, 3, 0.0, 1.0, BorderMode::Constant)
				.err()
				.expect("zero color sigma accepted"),
		] {
			assert_eq!(error.kind(), ErrorKind::InvalidArgument);
		}
		Ok(())
	}
);

test_vk!(
	filter_kernels_reject_wrong_ranks_without_panicking,
	engine,
	{
		let image = Image::new(
			Matrix::from_f32(&engine, [1, 2, 3], &[0.0; 6])?,
			ImageLayout::Chw,
			ImageFormat::Gray,
		)?;
		let unit = Matrix::from_f32(&engine, [1], &[1.0])?;
		for shape in [vec![], vec![1], vec![1, 1, 1], vec![0, 1]] {
			let values = vec![1.0; shape.iter().product()];
			let kernel = Matrix::from_f32(&engine, shape, &values)?;
			let error = oa::image::convolve_2d(&image, &kernel, BorderMode::Constant, 0.0)
				.err()
				.expect("invalid 2D kernel shape must return an error");
			assert_eq!(error.kind(), ErrorKind::InvalidArgument);
		}
		for shape in [vec![], vec![1, 1, 1], vec![2, 2], vec![0]] {
			let values = vec![1.0; shape.iter().product()];
			let kernel = Matrix::from_f32(&engine, shape, &values)?;
			for (x, y) in [(&kernel, &unit), (&unit, &kernel)] {
				let error = oa::image::separable_convolve_2d(&image, x, y, BorderMode::Constant, 0.0)
					.err()
					.expect("invalid vector kernel shape must return an error");
				assert_eq!(error.kind(), ErrorKind::InvalidArgument);
			}
		}
		Ok(())
	}
);

test_vk!(rectangular_filter_extents_preserve_pixels, engine, {
	let values = (0..15).map(|index| index as f32).collect::<Vec<_>>();
	let image = Image::new(
		Matrix::from_f32(&engine, [1, 3, 5], &values)?,
		ImageLayout::Chw,
		ImageFormat::Gray,
	)?;
	let output = oa::image::average_blur(&image, 3, 1, BorderMode::Replicate)?;
	let mut expected = Vec::new();
	for row in values.as_chunks::<5>().0 {
		for column in 0_usize..5 {
			expected.push((row[column.saturating_sub(1)] + row[column] + row[(column + 1).min(4)]) / 3.0);
		}
	}
	assert_eq!(output.as_matrix().shape(), [1, 3, 5]);
	for (actual, expected) in output.as_matrix().read_f32()?.into_iter().zip(expected) {
		assert!((actual - expected).abs() < 1e-6);
	}
	Ok(())
});
