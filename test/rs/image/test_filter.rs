use oa::{ErrorKind, Image, ImageFormat, ImageLayout, Matrix, image::BorderMode};

test_vk!(
	complete_filter_family_matches_identity_and_fixed_kernel_oracles,
	runtime,
	{
		let values = (0..9).map(|value| value as f32).collect::<Vec<_>>();
		let image = Image::new(
			Matrix::new_from_f32([1, 1, 3, 3], &values)?,
			ImageLayout::Nchw,
			ImageFormat::Gray,
		)?;
		let one_2d = Matrix::new_from_f32([1, 1], &[1.0])?;
		let one_1d = Matrix::new_from_f32([1], &[1.0])?;
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
	runtime,
	{
		let image = Image::new(
			Matrix::new_from_f32([1, 2, 2], &[0.0; 4])?,
			ImageLayout::Chw,
			ImageFormat::Gray,
		)?;
		let even = Matrix::new_from_f32([2, 2], &[1.0; 4])?;
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
	runtime,
	{
		let image = Image::new(
			Matrix::new_from_f32([1, 2, 3], &[0.0; 6])?,
			ImageLayout::Chw,
			ImageFormat::Gray,
		)?;
		let unit = Matrix::new_from_f32([1], &[1.0])?;
		for shape in [vec![], vec![1], vec![1, 1, 1], vec![0, 1]] {
			let values = vec![1.0; shape.iter().product()];
			let kernel = Matrix::new_from_f32(shape, &values)?;
			let error = oa::image::convolve_2d(&image, &kernel, BorderMode::Constant, 0.0)
				.err()
				.expect("invalid 2D kernel shape must return an error");
			assert_eq!(error.kind(), ErrorKind::InvalidArgument);
		}
		for shape in [vec![], vec![1, 1, 1], vec![2, 2], vec![0]] {
			let values = vec![1.0; shape.iter().product()];
			let kernel = Matrix::new_from_f32(shape, &values)?;
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

test_vk!(rectangular_filter_extents_preserve_pixels, runtime, {
	let values = (0..15).map(|index| index as f32).collect::<Vec<_>>();
	let image = Image::new(
		Matrix::new_from_f32([1, 3, 5], &values)?,
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

fn reflect101(coordinate: i64, extent: i64) -> usize {
	if coordinate < 0 {
		(-coordinate) as usize
	} else if coordinate >= extent {
		(2 * extent - 2 - coordinate) as usize
	} else {
		coordinate as usize
	}
}

fn canny_host_oracle(plane: &[f32], width: usize, height: usize, low: f32, high: f32) -> Vec<f32> {
	let at =
		|x: i64, y: i64| plane[reflect101(y, height as i64) * width + reflect101(x, width as i64)];
	let mut magnitude = vec![0.0f32; width * height];
	let mut direction = vec![0u8; width * height];
	for y in 0..height as i64 {
		for x in 0..width as i64 {
			let gx = -at(x - 1, y - 1) - 2.0 * at(x - 1, y) - at(x - 1, y + 1)
				+ at(x + 1, y - 1)
				+ 2.0 * at(x + 1, y)
				+ at(x + 1, y + 1);
			let gy = -at(x - 1, y - 1) - 2.0 * at(x, y - 1) - at(x + 1, y - 1)
				+ at(x - 1, y + 1)
				+ 2.0 * at(x, y + 1)
				+ at(x + 1, y + 1);
			let index = y as usize * width + x as usize;
			magnitude[index] = gx.abs() + gy.abs();
			let (ax, ay) = (gx.abs(), gy.abs());
			direction[index] = if ay <= ax * 0.41421356 {
				0
			} else if ax <= ay * 0.41421356 {
				2
			} else if gx * gy > 0.0 {
				1
			} else {
				3
			};
		}
	}
	let mut mask = vec![0u8; width * height];
	for y in 0..height {
		for x in 0..width {
			let index = y * width + x;
			let (dx, dy): (i64, i64) = match direction[index] {
				0 => (1, 0),
				1 => (1, 1),
				2 => (0, 1),
				_ => (1, -1),
			};
			let px = ((x as i64 + dx).min(width as i64 - 1)).max(0) as usize;
			let py = ((y as i64 + dy).min(height as i64 - 1)).max(0) as usize;
			let nx = ((x as i64 - dx).min(width as i64 - 1)).max(0) as usize;
			let ny = ((y as i64 - dy).min(height as i64 - 1)).max(0) as usize;
			let mag = magnitude[index];
			let kept = if mag >= magnitude[py * width + px] && mag >= magnitude[ny * width + nx] {
				mag
			} else {
				0.0
			};
			mask[index] = if kept >= high {
				2
			} else if kept >= low {
				1
			} else {
				0
			};
		}
	}
	loop {
		let mut changed = false;
		let snapshot = mask.clone();
		for y in 0..height {
			for x in 0..width {
				let index = y * width + x;
				if snapshot[index] != 1 {
					continue;
				}
				let strong = (-1_i64..=1).any(|dy| {
					(-1_i64..=1).any(|dx| {
						if dx == 0 && dy == 0 {
							return false;
						}
						let (nx, ny) = (x as i64 + dx, y as i64 + dy);
						nx >= 0
							&& ny >= 0
							&& nx < width as i64
							&& ny < height as i64
							&& snapshot[ny as usize * width + nx as usize] == 2
					})
				});
				if strong {
					mask[index] = 2;
					changed = true;
				}
			}
		}
		if !changed {
			break;
		}
	}
	mask
		.into_iter()
		.map(|value| if value == 2 { 1.0 } else { 0.0 })
		.collect()
}

test_vk!(
	canny_detects_a_step_edge_matching_the_host_oracle,
	runtime,
	{
		// Vertical step: dark left half, bright right half — one edge line.
		let width = 12;
		let height = 10;
		let mut plane = vec![0.0f32; width * height];
		for y in 0..height {
			for x in 6..width {
				plane[y * width + x] = 1.0;
			}
		}
		let image = Image::new(
			Matrix::new_from_f32_on(&runtime, [1, 1, height, width], &plane)?,
			ImageLayout::Nchw,
			ImageFormat::Gray,
		)?;
		let output = oa::image::canny(&image, 0.5, 2.0, 0)?;
		let expected = canny_host_oracle(&plane, width, height, 0.5, 2.0);
		assert_eq!(output.as_matrix().read_f32()?, expected);
		// Threshold hysteresis keeps a weak ramp attached to a strong edge.
		let mut ramp = vec![0.0f32; width * height];
		for y in 0..height {
			for x in 0..width {
				ramp[y * width + x] = if x < 4 { 1.0 } else { (8.0 - x as f32) / 10.0 };
			}
		}
		let ramp_image = Image::new(
			Matrix::new_from_f32_on(&runtime, [1, 1, height, width], &ramp)?,
			ImageLayout::Nchw,
			ImageFormat::Gray,
		)?;
		let ramp_output = oa::image::canny(&ramp_image, 0.2, 1.5, 0)?;
		let ramp_expected = canny_host_oracle(&ramp, width, height, 0.2, 1.5);
		assert_eq!(ramp_output.as_matrix().read_f32()?, ramp_expected);
		Ok(())
	}
);

test_vk!(canny_rejects_invalid_thresholds, runtime, {
	let _ = runtime;
	let image = Image::new(
		Matrix::new_from_f32([1, 1, 4, 4], &[0.5f32; 16])?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)?;
	assert!(oa::image::canny(&image, 1.0, 0.5, 0).is_err());
	assert!(oa::image::canny(&image, -0.1, 0.5, 0).is_err());
	assert!(oa::image::canny(&image, 0.1, f32::NAN, 0).is_err());
	Ok(())
});
