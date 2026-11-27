use oa::{ErrorKind, Image, ImageFormat, ImageLayout, Matrix, image::HistogramComparison};

fn gray_image(runtime: &oa::Runtime, shape: [usize; 4], values: &[f32]) -> oa::Result<Image> {
	Image::new(
		Matrix::new_from_f32_on(runtime, shape, values)?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)
}

test_vk!(calc_hist_matches_an_independent_bucket_oracle, runtime, {
	// [1,1,2,4]: values 0,.25,.5,.75 / 1.5,.5,-.5,.999 over [0,1) in 4 bins.
	let values = [0.0, 0.25, 0.5, 0.75, 1.5, 0.5, -0.5, 0.999];
	let image = gray_image(&runtime, [1, 1, 2, 4], &values)?;
	let histogram = image.calc_hist(4, 0.0, 1.0)?;
	assert_eq!(histogram.shape(), &[1, 1, 4]);
	assert_eq!(histogram.dtype(), oa::DType::U32);
	let counts: Vec<u32> = histogram.read()?;
	assert_eq!(counts, vec![1, 1, 2, 2]);
	Ok(())
});

test_vk!(calc_hist_counts_each_plane_independently, runtime, {
	// [2,1,1,3]: batch plane 0 uses bins 0 and 2, plane 1 uses bins 1 and 3.
	let values = [0.0, 0.0, 0.5, 0.25, 0.75, 0.25];
	let image = gray_image(&runtime, [2, 1, 1, 3], &values)?;
	let histogram = oa::image::calc_hist(&image, 4, 0.0, 1.0)?;
	assert_eq!(histogram.shape(), &[2, 1, 4]);
	let counts: Vec<u32> = histogram.read()?;
	assert_eq!(counts, vec![2, 0, 1, 0, 0, 2, 0, 1]);
	Ok(())
});

test_vk!(calc_hist_rejects_invalid_parameters, runtime, {
	let _ = runtime;
	let values = [0.5f32; 4];
	let image = Image::new(
		Matrix::new_from_f32([1, 1, 1, 4], &values)?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)?;
	for (bins, lower, upper) in [
		(0, 0.0, 1.0),
		(257, 0.0, 1.0),
		(4, 1.0, 1.0),
		(4, 0.5, 0.0),
		(4, f32::NAN, 1.0),
		(4, 0.0, f32::INFINITY),
	] {
		let error = oa::image::calc_hist(&image, bins, lower, upper)
			.err()
			.unwrap();
		assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	}
	Ok(())
});

test_vk!(
	equalize_hist_spreads_the_cumulative_distribution,
	runtime,
	{
		// [1,1,1,4] = [0,0,0,0.9]: buckets 0,0,0,230; cdf_min=3, total=4.
		// lut[0]=(3-3)/1=0, lut[230]=(4-3)/1=1 -> [0,0,0,1].
		let image = gray_image(&runtime, [1, 1, 1, 4], &[0.0, 0.0, 0.0, 0.9])?;
		let equalized = image.equalize_hist()?;
		assert_eq!(equalized.layout(), image.layout());
		assert_eq!(equalized.format(), image.format());
		let result = equalized.as_matrix().read_f32()?;
		assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0]);
		Ok(())
	}
);

test_vk!(equalize_hist_leaves_a_degenerate_plane_at_zero, runtime, {
	// Single-valued plane: cdf_min == total -> all outputs 0.
	let image = gray_image(&runtime, [1, 1, 2, 3], &[0.5; 6])?;
	let result = image.equalize_hist()?.as_matrix().read_f32()?;
	assert_eq!(result, vec![0.0; 6]);
	Ok(())
});

test_vk!(compare_hist_matches_independent_metric_oracles, runtime, {
	// Orthogonal one-hot rows: correlation -0.5, chi-square 1.0,
	// intersection 0.0, bhattacharyya 1.0. Identical rows return the
	// perfect-match score for each metric.
	let first = Matrix::new_from_slice_on(&runtime, [1, 1, 3], &[1u32, 0, 0])?;
	let second = first.new_from_slice_on_same_runtime([1, 1, 3], &[0u32, 1, 0])?;
	for (method, expected) in [
		(HistogramComparison::Correlation, -0.5),
		(HistogramComparison::ChiSquare, 1.0),
		(HistogramComparison::Intersection, 0.0),
		(HistogramComparison::Bhattacharyya, 1.0),
	] {
		let scores = oa::image::compare_hist(&first, &second, method)?;
		assert_eq!(scores.shape(), &[1, 1]);
		let values = scores.read_f32()?;
		assert!(
			(values[0] - expected).abs() < 1e-6,
			"{method:?}: expected {expected}, got {}",
			values[0]
		);
	}
	for method in [
		HistogramComparison::Correlation,
		HistogramComparison::ChiSquare,
		HistogramComparison::Intersection,
		HistogramComparison::Bhattacharyya,
	] {
		let perfect = match method {
			HistogramComparison::Correlation | HistogramComparison::Intersection => 1.0,
			_ => 0.0,
		};
		let scores = oa::image::compare_hist(&first, &first, method)?;
		let values = scores.read_f32()?;
		assert!(
			(values[0] - perfect).abs() < 1e-6,
			"{method:?} self-match: expected {perfect}, got {}",
			values[0]
		);
	}
	Ok(())
});

test_vk!(
	compare_hist_rejects_mismatched_or_nonhistogram_inputs,
	runtime,
	{
		let _ = runtime;
		let first = Matrix::new_from_slice([2, 3], &[1u32, 0, 0, 0, 1, 0])?;
		let wrong_dtype = Matrix::new_from_f32([2, 3], &[1.0; 6])?;
		let wrong_shape = Matrix::new_from_slice([6], &[1u32; 6])?;
		let rank_one = Matrix::new_from_slice([3], &[1u32, 0, 0])?;
		assert_eq!(
			oa::image::compare_hist(&first, &wrong_dtype, HistogramComparison::Correlation)
				.err()
				.unwrap()
				.kind(),
			ErrorKind::InvalidArgument
		);
		assert_eq!(
			oa::image::compare_hist(&first, &wrong_shape, HistogramComparison::Correlation)
				.err()
				.unwrap()
				.kind(),
			ErrorKind::InvalidArgument
		);
		assert_eq!(
			oa::image::compare_hist(&rank_one, &rank_one, HistogramComparison::Correlation)
				.err()
				.unwrap()
				.kind(),
			ErrorKind::InvalidArgument
		);
		Ok(())
	}
);

test_vk!(captured_histogram_plan_replays_identical_counts, runtime, {
	let values = [0.0, 0.25, 0.5, 0.75, 1.5, 0.5, -0.5, 0.999];
	let image = gray_image(&runtime, [1, 1, 2, 4], &values)?;
	let expected = image.calc_hist(4, 0.0, 1.0)?.read::<u32>()?;
	let (plan, histogram) = runtime.capture(|| image.calc_hist(4, 0.0, 1.0))?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("image calc_hist"))
		.expect("valid execution graph report");
	assert_eq!(report["stats"]["dispatches"], 2);
	runtime.submit(&plan)?.wait()?;
	let first_run: Vec<u32> = histogram.read()?;
	assert_eq!(first_run, expected);
	// Replaying the same plan must clear the accumulator; a missing
	// clear stage would double every bucket.
	runtime.submit(&plan)?.wait()?;
	let second_run: Vec<u32> = histogram.read()?;
	assert_eq!(second_run, expected);
	Ok(())
});

test_vk!(
	captured_calc_hist_feeds_compare_hist_in_one_plan,
	runtime,
	{
		let values = [0.0, 0.25, 0.5, 0.75, 1.5, 0.5, -0.5, 0.999];
		let image = gray_image(&runtime, [1, 1, 2, 4], &values)?;
		let other = gray_image(
			&runtime,
			[1, 1, 2, 4],
			&[0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
		)?;
		let (plan, (first_histogram, scores)) = runtime.capture(|| {
			let first = image.calc_hist(4, 0.0, 1.0)?;
			let second = other.calc_hist(4, 0.0, 1.0)?;
			let score = oa::image::compare_hist(&first, &second, HistogramComparison::Intersection)?;
			Ok((first, score))
		})?;
		runtime.submit(&plan)?.wait()?;
		let counts: Vec<u32> = first_histogram.read()?;
		assert_eq!(counts, vec![1, 1, 2, 2]);
		// second = [4,0,4,0]: overlap = min(1,4)+min(1,0)+min(2,4)+min(2,0) = 3;
		// normalized by first total 6 -> 0.5.
		let value = scores.read_f32()?;
		assert!((value[0] - 0.5).abs() < 1e-6, "got {}", value[0]);
		Ok(())
	}
);

fn quantize_256(value: f32) -> usize {
	((value.clamp(0.0, 1.0) * 256.0) as usize).min(255)
}

fn histogram_256(values: &[f32]) -> [u32; 256] {
	let mut counts = [0u32; 256];
	for &value in values {
		counts[quantize_256(value)] += 1;
	}
	counts
}

fn otsu_oracle(counts: &[u32; 256]) -> u32 {
	let total: f32 = counts.iter().sum::<u32>() as f32;
	if total <= 0.0 {
		return 0;
	}
	let sum_all: f32 = counts
		.iter()
		.enumerate()
		.map(|(i, &c)| i as f32 * c as f32)
		.sum();
	let mut weight_back = 0.0;
	let mut sum_back = 0.0;
	let mut best = -1.0f32;
	let mut best_bin = 0u32;
	for (t, &count) in counts.iter().enumerate() {
		weight_back += count as f32;
		sum_back += t as f32 * count as f32;
		let weight_fore = total - weight_back;
		if weight_back <= 0.0 || weight_fore <= 0.0 {
			continue;
		}
		let mean_back = sum_back / weight_back;
		let mean_fore = (sum_all - sum_back) / weight_fore;
		let delta = mean_back - mean_fore;
		let variance = weight_back * weight_fore * delta * delta;
		if variance > best {
			best = variance;
			best_bin = t as u32;
		}
	}
	best_bin
}

test_vk!(threshold_auto_otsu_splits_a_bimodal_plane, runtime, {
	// 12 dark pixels at bucket 25, 12 bright at bucket 230.
	let mut values = vec![0.1f32; 12];
	values.extend(vec![0.9f32; 12]);
	let image = gray_image(&runtime, [1, 1, 4, 6], &values)?;
	let (output, thresholds) = image.threshold_auto(oa::image::AutoThreshold::Otsu, 1.0)?;
	assert_eq!(thresholds.shape(), &[1, 1]);
	let level = thresholds.read_f32()?;
	let expected = otsu_oracle(&histogram_256(&values));
	assert_eq!(level[0] as u32, expected, "otsu bucket mismatch");
	let result = output.as_matrix().read_f32()?;
	for (actual, &source) in result.iter().zip(values.iter()) {
		let expect = if quantize_256(source) as u32 > expected {
			1.0
		} else {
			0.0
		};
		assert_eq!(*actual, expect);
	}
	Ok(())
});

fn triangle_oracle(counts: &mut [u32; 256]) -> u32 {
	let mut left = (0..256).find(|&i| counts[i] > 0).unwrap_or(0);
	left = left.saturating_sub(1);
	let mut right = (0..256).rev().find(|&i| counts[i] > 0).unwrap_or(255);
	if right < 255 {
		right += 1;
	}
	let max_ind = (0..256).max_by_key(|&i| counts[i]).unwrap_or(0);
	let max = counts[max_ind];
	let flipped = (max_ind as i64 - left as i64) < (right as i64 - max_ind as i64);
	if flipped {
		counts.reverse();
		left = 255 - right;
	}
	let segment_peak = if flipped { 255 - max_ind } else { max_ind };
	let a = max as f64;
	let b = left as f64 - segment_peak as f64;
	let mut dist = 0.0;
	let mut thresh = left;
	for (i, &count) in counts
		.iter()
		.enumerate()
		.take(segment_peak + 1)
		.skip(left + 1)
	{
		let distance = a * i as f64 + b * f64::from(count);
		if distance > dist {
			dist = distance;
			thresh = i;
		}
	}
	thresh = thresh.saturating_sub(1);
	if flipped {
		(255 - thresh) as u32
	} else {
		thresh as u32
	}
}

test_vk!(threshold_auto_triangle_finds_the_skewed_knee, runtime, {
	// Heavily left-skewed: most pixels near 0.05, a tail toward 0.6.
	let mut values = vec![0.05f32; 40];
	values.extend([0.4f32, 0.5, 0.55, 0.6, 0.45, 0.5, 0.55, 0.6]);
	let image = gray_image(&runtime, [1, 1, 6, 8], &values)?;
	let (output, thresholds) = image.threshold_auto(oa::image::AutoThreshold::Triangle, 2.0)?;
	let expected = triangle_oracle(&mut histogram_256(&values));
	let level = thresholds.read_f32()?;
	assert_eq!(level[0] as u32, expected, "triangle bucket mismatch");
	let result = output.as_matrix().read_f32()?;
	for (actual, &source) in result.iter().zip(values.iter()) {
		let expect = if quantize_256(source) as u32 > expected {
			2.0
		} else {
			0.0
		};
		assert_eq!(*actual, expect);
	}
	Ok(())
});

test_vk!(threshold_auto_returns_one_level_per_plane, runtime, {
	// [2,1,1,4]: plane 0 splits dark/bright; plane 1 is uniform, so its
	// Otsu level stays at bucket 0 and every pixel quantizes above it.
	let values = [0.1, 0.1, 0.9, 0.9, 0.05, 0.05, 0.05, 0.05];
	let image = gray_image(&runtime, [2, 1, 1, 4], &values)?;
	let (output, thresholds) =
		oa::image::threshold_auto(&image, oa::image::AutoThreshold::Otsu, 1.0)?;
	assert_eq!(thresholds.shape(), &[2, 1]);
	let levels = thresholds.read_f32()?;
	assert_eq!(levels[0] as u32, otsu_oracle(&histogram_256(&values[0..4])));
	assert_eq!(levels[1] as u32, 0);
	let result = output.as_matrix().read_f32()?;
	assert_eq!(result[0..4], [0.0, 0.0, 1.0, 1.0]);
	assert_eq!(result[4..8], [1.0, 1.0, 1.0, 1.0]);
	Ok(())
});

test_vk!(threshold_auto_rejects_nonfinite_max_value, runtime, {
	let image = gray_image(&runtime, [1, 1, 1, 4], &[0.5f32; 4])?;
	for max_value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
		let error = image
			.threshold_auto(oa::image::AutoThreshold::Otsu, max_value)
			.err()
			.unwrap();
		assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	}
	Ok(())
});

test_vk!(integral_matches_a_host_summed_area_table, runtime, {
	let height = 3;
	let width = 4;
	let values = (0..height * width)
		.map(|value| ((value * 7) % 11) as f32 / 11.0)
		.collect::<Vec<_>>();
	let image = gray_image(&runtime, [1, 1, height, width], &values)?;
	let sat = image.integral()?;
	assert_eq!(sat.shape(), &[1, 1, height + 1, width + 1]);
	let result = sat.read_f32()?;
	let out_width = width + 1;
	for y in 0..=height {
		for x in 0..=width {
			let mut expected = 0.0f32;
			for sy in 0..y {
				for sx in 0..x {
					expected += values[sy * width + sx];
				}
			}
			let actual = result[y * out_width + x];
			assert!(
				(actual - expected).abs() < 1e-4,
				"sat[{y},{x}]: {actual} != {expected}"
			);
		}
	}
	Ok(())
});

test_vk!(integral_tracks_each_plane_independently, runtime, {
	let values = [1.0f32; 4 * 2];
	let image = gray_image(&runtime, [2, 1, 2, 2], &values)?;
	let sat = oa::image::integral(&image)?;
	assert_eq!(sat.shape(), &[2, 1, 3, 3]);
	let result = sat.read_f32()?;
	let expected_row = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 4.0];
	assert_eq!(result[0..9], expected_row);
	assert_eq!(result[9..18], expected_row);
	Ok(())
});

test_vk!(
	calc_back_project_writes_histogram_counts_per_pixel,
	runtime,
	{
		// Values in bins 0,1,2,3 over [0,1) with 4 bins; histogram counts
		// 1,1,1,1 map each pixel back to its bucket population.
		let values = [0.1, 0.4, 0.6, 0.9];
		let image = gray_image(&runtime, [1, 1, 1, 4], &values)?;
		let histogram = image.calc_hist(4, 0.0, 1.0)?;
		let projected = oa::image::calc_back_project(&image, &histogram, 0.0, 1.0)?;
		assert_eq!(projected.as_matrix().shape(), image.as_matrix().shape());
		assert_eq!(projected.as_matrix().read_f32()?, vec![1.0; 4]);
		Ok(())
	}
);

test_vk!(calc_back_project_zeros_out_of_range_pixels, runtime, {
	// Histogram over [0, 0.5): the 0.75 pixel quantizes out of range.
	let image = gray_image(&runtime, [1, 1, 1, 2], &[0.25, 0.75])?;
	let histogram = Matrix::new_from_slice_on(&runtime, [1, 1, 2], &[3u32, 7])?;
	let projected = oa::image::calc_back_project(&image, &histogram, 0.0, 0.5)?;
	// bin for 0.25 = floor(0.25*4)=1 -> count 7; 0.75 out of range -> 0.
	assert_eq!(projected.as_matrix().read_f32()?, vec![7.0, 0.0]);
	Ok(())
});

test_vk!(clahe_on_a_single_tile_matches_the_cdf_oracle, runtime, {
	// One tile: bins 0,64,128,255 each hold one pixel -> lut = cdf/4.
	let image = gray_image(&runtime, [1, 1, 1, 4], &[0.0, 0.5, 1.0, 0.25])?;
	let result = oa::image::clahe(&image, 40.0, 4, 1)?;
	assert_eq!(result.as_matrix().shape(), image.as_matrix().shape());
	let expected = [0.25, 0.75, 1.0, 0.5];
	for (expected, actual) in expected.iter().zip(result.as_matrix().read_f32()?.iter()) {
		assert!((expected - actual).abs() < 1e-5, "{expected} != {actual}");
	}
	Ok(())
});

test_vk!(clahe_keeps_a_uniform_plane_uniform, runtime, {
	let image = gray_image(&runtime, [1, 1, 4, 4], &[0.5; 16])?;
	let result = oa::image::clahe(&image, 2.0, 2, 2)?
		.as_matrix()
		.read_f32()?;
	let first = result[0];
	assert!(result.iter().all(|value| (*value - first).abs() < 1e-6));
	Ok(())
});

test_vk!(clahe_rejects_invalid_parameters, runtime, {
	let _ = runtime;
	let image = Image::new(
		Matrix::new_from_f32([1, 1, 2, 2], &[0.5; 4])?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)?;
	for (clip, tw, th) in [
		(0.0, 1, 1),
		(-1.0, 1, 1),
		(f32::NAN, 1, 1),
		(1.0, 0, 1),
		(1.0, 1, 0),
	] {
		let error = oa::image::clahe(&image, clip, tw, th).err().unwrap();
		assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	}
	Ok(())
});

test_vk!(accumulate_sums_two_planes_elementwise, runtime, {
	let first = gray_image(&runtime, [1, 1, 1, 4], &[0.1, 0.2, 0.3, 0.4])?;
	let second = gray_image(&runtime, [1, 1, 1, 4], &[1.0, 1.0, 1.0, 1.0])?;
	let sum = oa::image::accumulate(&first, &second)?;
	assert_eq!(sum.as_matrix().read_f32()?, vec![1.1, 1.2, 1.3, 1.4]);
	Ok(())
});

test_vk!(accumulate_weighted_blends_by_alpha, runtime, {
	let input = gray_image(&runtime, [1, 1, 1, 2], &[1.0, 0.0])?;
	let accumulator = gray_image(&runtime, [1, 1, 1, 2], &[0.0, 1.0])?;
	let blended = oa::image::accumulate_weighted(&input, &accumulator, 0.25)?;
	let result = blended.as_matrix().read_f32()?;
	assert!((result[0] - 0.25).abs() < 1e-6);
	assert!((result[1] - 0.75).abs() < 1e-6);
	let error = oa::image::accumulate_weighted(&input, &accumulator, f32::NAN)
		.err()
		.unwrap();
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	Ok(())
});

test_vk!(moments_matches_a_direct_spatial_oracle, runtime, {
	// 2x3 plane with distinct values.
	let values = [1.0, 2.0, 0.5, 0.0, 1.5, 2.5];
	let image = gray_image(&runtime, [1, 1, 2, 3], &values)?;
	let moments = image.moments()?;
	assert_eq!(moments.shape(), &[1, 1, 10]);
	let result = moments.read_f32()?;
	let (h, w) = (2usize, 3usize);
	let mut oracle = [0.0f32; 10];
	for y in 0..h {
		for x in 0..w {
			let v = values[y * w + x];
			let (fx, fy) = (x as f32, y as f32);
			oracle[0] += v;
			oracle[1] += fx * v;
			oracle[2] += fy * v;
			oracle[3] += fx * fx * v;
			oracle[4] += fx * fy * v;
			oracle[5] += fy * fy * v;
			oracle[6] += fx * fx * fx * v;
			oracle[7] += fx * fx * fy * v;
			oracle[8] += fx * fy * fy * v;
			oracle[9] += fy * fy * fy * v;
		}
	}
	for (expected, actual) in oracle.iter().zip(result.iter()) {
		assert!((expected - actual).abs() < 1e-4, "{expected} != {actual}");
	}
	Ok(())
});

test_vk!(psnr_returns_inf_for_identical_planes, runtime, {
	let image = gray_image(&runtime, [1, 1, 2, 2], &[0.25, 0.5, 0.75, 1.0])?;
	let result = oa::image::psnr(&image, &image, 1.0)?;
	assert_eq!(result.shape(), &[1, 1]);
	assert!(result.read_f32()?[0].is_infinite());
	Ok(())
});

test_vk!(psnr_matches_the_mse_oracle, runtime, {
	// mse = mean((a-b)^2) over four deltas of 0.1 -> psnr = 20 dB.
	let first = gray_image(&runtime, [1, 1, 2, 2], &[0.5, 0.5, 0.5, 0.5])?;
	let second = gray_image(&runtime, [1, 1, 2, 2], &[0.4, 0.6, 0.4, 0.6])?;
	let result = oa::image::psnr(&first, &second, 1.0)?.read_f32()?;
	assert!((result[0] - 20.0).abs() < 0.01, "{}", result[0]);
	Ok(())
});

test_vk!(ssim_scores_identical_planes_at_one, runtime, {
	let image = gray_image(&runtime, [1, 1, 2, 2], &[0.2, 0.4, 0.6, 0.8])?;
	let result = oa::image::ssim(&image, &image)?.read_f32()?;
	assert!((result[0] - 1.0).abs() < 1e-5);
	Ok(())
});

test_vk!(ssim_drops_below_one_on_perturbation, runtime, {
	let first = gray_image(&runtime, [1, 1, 4, 4], &[0.5; 16])?;
	let mut values = [0.5f32; 16];
	values[0] = 0.9;
	let second = gray_image(&runtime, [1, 1, 4, 4], &values)?;
	let result = oa::image::ssim(&first, &second)?.read_f32()?;
	assert!(result[0] < 1.0 && result[0] > 0.0, "{}", result[0]);
	Ok(())
});

test_vk!(match_template_finds_the_exact_patch, runtime, {
	// 1x3x4 plane; a 1x2 template copied from columns 1..3 scores
	// zero sqdiff at position (0,1).
	let image = gray_image(
		&runtime,
		[1, 1, 3, 4],
		&[0.0, 0.5, 0.9, 0.2, 0.1, 0.6, 1.0, 0.3, 0.2, 0.7, 0.8, 0.4],
	)?;
	let template = gray_image(&runtime, [1, 1, 1, 2], &[0.6, 1.0])?;
	let scores = oa::image::match_template(&image, &template, oa::image::MatchTemplate::Sqdiff)?;
	assert_eq!(scores.shape(), &[1, 1, 3, 3]);
	let result = scores.read_f32()?;
	// Exact copy at row 1, col 1 -> index 1*3+1 = 4.
	assert!(result[4].abs() < 1e-6, "{}", result[4]);
	// Every position score >= 0 and position 4 is the strict minimum.
	let minimum = result
		.iter()
		.enumerate()
		.min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
		.map(|(index, _)| index);
	assert_eq!(minimum, Some(4));
	Ok(())
});

test_vk!(
	match_template_ccorr_normed_peaks_on_self_similarity,
	runtime,
	{
		let image = gray_image(&runtime, [1, 1, 2, 3], &[0.2, 0.8, 0.1, 0.4, 0.5, 0.3])?;
		let template = gray_image(&runtime, [1, 1, 1, 2], &[0.8, 0.1])?;
		let scores =
			oa::image::match_template(&image, &template, oa::image::MatchTemplate::CcorrNormed)?;
		// Patch (0,0)=(0.2,0.8) is not proportional to template; (0,1)=(0.8,0.1)
		// is the template itself -> normalized correlation 1.0 at index 1.
		let result = scores.read_f32()?;
		assert!((result[1] - 1.0).abs() < 1e-4, "{}", result[1]);
		Ok(())
	}
);

test_vk!(match_template_rejects_an_oversized_template, runtime, {
	let image = gray_image(&runtime, [1, 1, 2, 2], &[0.5; 4])?;
	let template = gray_image(&runtime, [1, 1, 3, 3], &[0.5; 9])?;
	let error = oa::image::match_template(&image, &template, oa::image::MatchTemplate::Sqdiff)
		.err()
		.unwrap();
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	Ok(())
});

test_vk!(connected_components_labels_regions_per_plane, runtime, {
	// Two diagonal 8-connected blobs and one isolated pixel.
	let width = 8;
	let height = 6;
	let mut plane = vec![0.0f32; width * height];
	for (x, y) in [(1, 1), (2, 2), (3, 3), (6, 1), (6, 2), (6, 3), (0, 5)] {
		plane[y * width + x] = 1.0;
	}
	let image = Image::new(
		Matrix::new_from_f32_on(&runtime, [1, 1, height, width], &plane)?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)?;
	let values = oa::image::connected_components(&image, 0.5, oa::image::Connectivity::Eight, 0)?
		.read::<u32>()?;
	// blob A = min index 1*8+1=9; blob B = 1*8+6=14; isolated = 5*8+0=40.
	assert_eq!(values[9], 9);
	assert_eq!(values[2 * 8 + 2], 9);
	assert_eq!(values[3 * 8 + 3], 9);
	assert_eq!(values[8 + 6], 14);
	assert_eq!(values[3 * 8 + 6], 14);
	assert_eq!(values[5 * 8], 40);
	assert_eq!(values[0], u32::MAX);
	// 4-connectivity splits the diagonal blob.
	let four = oa::image::connected_components(&image, 0.5, oa::image::Connectivity::Four, 0)?;
	let four_values = four.read::<u32>()?;
	assert_ne!(four_values[9], four_values[2 * 8 + 2]);
	assert_eq!(four_values[2 * 8 + 6], four_values[3 * 8 + 6]);
	Ok(())
});

test_vk!(
	distance_transform_matches_independent_chamfer_oracles,
	runtime,
	{
		// Single background pixel at (4, 2): foreground distances radiate out.
		let width = 9;
		let height = 7;
		let mut plane = vec![1.0f32; width * height];
		plane[2 * width + 4] = 0.0;
		let image = Image::new(
			Matrix::new_from_f32_on(&runtime, [1, 1, height, width], &plane)?,
			ImageLayout::Nchw,
			ImageFormat::Gray,
		)?;
		let l1 = oa::image::distance_transform(&image, 0.5, oa::image::DistanceType::L1, 0)?;
		let l2 = oa::image::distance_transform(&image, 0.5, oa::image::DistanceType::L2, 0)?;
		let l1_values = l1.read_f32()?;
		let l2_values = l2.read_f32()?;
		let sqrt2 = 2.0f32.sqrt();
		for y in 0..height {
			for x in 0..width {
				let dx = (x as i32 - 4).abs() as f32;
				let dy = (y as i32 - 2).abs() as f32;
				let index = y * width + x;
				let expected_l1 = dx + dy;
				let expected_l2 = dx.min(dy) * sqrt2 + (dx - dy).abs();
				assert!(
					(l1_values[index] - expected_l1).abs() < 1e-4,
					"L1 at ({x},{y}): {} != {expected_l1}",
					l1_values[index]
				);
				assert!(
					(l2_values[index] - expected_l2).abs() < 1e-4,
					"L2 chamfer at ({x},{y}): {} != {expected_l2}",
					l2_values[index]
				);
			}
		}
		Ok(())
	}
);

test_vk!(hough_lines_detects_a_single_voting_peak, runtime, {
	// Vertical edge line at x = 4 across a 16x16 plane votes theta = 0, rho = 4.
	let width = 16;
	let height = 16;
	let mut plane = vec![0.0f32; width * height];
	for y in 0..height {
		plane[y * width + 4] = 1.0;
	}
	let image = Image::new(
		Matrix::new_from_f32_on(&runtime, [1, 1, height, width], &plane)?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)?;
	let lines = oa::image::hough_lines(&image, 0.5, 1.0, 1.0, 16, 4)?;
	assert_eq!(lines.shape(), &[1, 4, 3]);
	let values = lines.read_f32()?;
	let mut found = false;
	for row in 0..4 {
		let (rho, theta, votes) = (values[row * 3], values[row * 3 + 1], values[row * 3 + 2]);
		if votes <= 0.0 {
			continue;
		}
		// A vertical line also peaks near theta = 179 (rho = -4) and at
		// tilted lower-vote cells; assert the canonical peak is present.
		if (rho - 4.0).abs() < 0.6 && theta.abs() < 1.0 {
			found = true;
			assert_eq!(votes, 16.0);
		}
	}
	assert!(found);
	// A horizontal edge line votes theta = 90, rho = 3.
	let mut plane2 = vec![0.0f32; width * height];
	for x in 0..width {
		plane2[3 * width + x] = 1.0;
	}
	let image2 = Image::new(
		Matrix::new_from_f32_on(&runtime, [1, 1, height, width], &plane2)?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)?;
	let lines2 = oa::image::hough_lines(&image2, 0.5, 1.0, 1.0, 16, 4)?;
	let values2 = lines2.read_f32()?;
	let mut found2 = false;
	for row in 0..4 {
		let (rho, theta, votes) = (values2[row * 3], values2[row * 3 + 1], values2[row * 3 + 2]);
		// Quantized voting ties at theta = 89/90; either canonical cell wins.
		if (rho - 3.0).abs() < 0.6 && (theta - 90.0).abs() <= 2.0 {
			found2 = true;
			assert_eq!(votes, 16.0);
		}
	}
	assert!(found2);
	Ok(())
});

test_vk!(hough_lines_p_recovers_segment_endpoints, runtime, {
	// Horizontal edge run on row 5 from x = 2..=13 with a 1-pixel gap at x = 8.
	let width = 16;
	let height = 12;
	let mut plane = vec![0.0f32; width * height];
	for x in 2..=13 {
		if x == 8 {
			continue;
		}
		plane[5 * width + x] = 1.0;
	}
	let image = Image::new(
		Matrix::new_from_f32_on(&runtime, [1, 1, height, width], &plane)?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)?;
	let segments = oa::image::hough_lines_p(&image, 0.5, 1.0, 1.0, 10, 4, 8.0, 1)?;
	assert_eq!(segments.shape(), &[1, 4, 4]);
	let values = segments.read_f32()?;
	let mut found = false;
	for row in 0..4 {
		let (x1, y1, x2, y2) = (
			values[row * 4],
			values[row * 4 + 1],
			values[row * 4 + 2],
			values[row * 4 + 3],
		);
		if x1 == 0.0 && y1 == 0.0 && x2 == 0.0 && y2 == 0.0 {
			continue;
		}
		// The dominant run spans the full gapped segment at y = 5.
		if (x2 - x1).abs() >= 10.0 && (y1 - 5.0).abs() <= 1.5 && (y2 - 5.0).abs() <= 1.5 {
			found = true;
			assert!((x1 - 2.0).abs() <= 1.5, "x1 {x1}");
			assert!((x2 - 13.0).abs() <= 1.5, "x2 {x2}");
		}
	}
	assert!(found, "no qualifying segment in {values:?}");
	Ok(())
});

test_vk!(find_contours_traces_rectangle_boundary, runtime, {
	// Filled 5x4 rectangle at (x 3..=7, y 2..=5): Moore boundary is
	// 2*5 + 2*4 - 4 = 14 pixels; a second 2x2 blob adds 4.
	let width = 16;
	let height = 12;
	let mut plane = vec![0.0f32; width * height];
	for y in 2..=5 {
		for x in 3..=7 {
			plane[y * width + x] = 1.0;
		}
	}
	for y in 8..=9 {
		for x in 12..=13 {
			plane[y * width + x] = 1.0;
		}
	}
	let image = Image::new(
		Matrix::new_from_f32_on(&runtime, [1, 1, height, width], &plane)?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)?;
	let (points, lengths) =
		oa::image::find_contours(&image, 0.5, oa::image::Connectivity::Eight, 4, 0, 0)?;
	assert_eq!(points.shape(), &[1, 4, 112, 2]);
	assert_eq!(lengths.shape(), &[1, 4]);
	let counts: Vec<u32> = lengths.read()?;
	let mut traced: Vec<usize> = counts
		.iter()
		.map(|&c| c as usize)
		.filter(|&c| c > 0)
		.collect();
	traced.sort_unstable();
	assert_eq!(traced, vec![4, 14], "contour lengths {counts:?}");
	// The 14-point contour's points must all lie on the rectangle boundary.
	let coords = points.read_f32()?;
	let (long_slot, long_len) = counts
		.iter()
		.enumerate()
		.find(|&(_, &c)| c == 14)
		.map(|(i, &c)| (i, c as usize))
		.expect("14-point contour");
	for t in 0..long_len {
		let x = coords[(long_slot * 112 + t) * 2] as i32;
		let y = coords[(long_slot * 112 + t) * 2 + 1] as i32;
		let on_boundary =
			(3..=7).contains(&x) && (2..=5).contains(&y) && (x == 3 || x == 7 || y == 2 || y == 5);
		assert!(on_boundary, "point {t} = ({x}, {y}) not on boundary");
	}
	// First point is the component's minimum-index pixel: (3, 2).
	assert_eq!(coords[(long_slot * 112) * 2], 3.0);
	assert_eq!(coords[(long_slot * 112) * 2 + 1], 2.0);
	Ok(())
});

fn reflect101_index(coordinate: i32, extent: i32) -> usize {
	if coordinate < 0 {
		(-coordinate) as usize
	} else if coordinate >= extent {
		(2 * extent - 2 - coordinate) as usize
	} else {
		coordinate as usize
	}
}

fn corner_score_reference(
	values: &[f32],
	width: usize,
	height: usize,
	block_size: usize,
	k: f32,
	min_eigen: bool,
) -> Vec<f32> {
	let half = block_size as i32 / 2;
	let mut products = vec![0.0f32; width * height * 3];
	for y in 0..height as i32 {
		for x in 0..width as i32 {
			let sample = |dx: i32, dy: i32| {
				values
					[reflect101_index(y + dy, height as i32) * width + reflect101_index(x + dx, width as i32)]
			};
			let gx = -sample(-1, -1) - 2.0 * sample(-1, 0) - sample(-1, 1)
				+ sample(1, -1)
				+ 2.0 * sample(1, 0)
				+ sample(1, 1);
			let gy = -sample(-1, -1) - 2.0 * sample(0, -1) - sample(1, -1)
				+ sample(-1, 1)
				+ 2.0 * sample(0, 1)
				+ sample(1, 1);
			let base = (y as usize * width + x as usize) * 3;
			products[base] = gx * gx;
			products[base + 1] = gy * gy;
			products[base + 2] = gx * gy;
		}
	}
	let mut out = vec![0.0f32; width * height];
	for y in 0..height as i32 {
		for x in 0..width as i32 {
			let (mut sxx, mut syy, mut sxy) = (0.0f32, 0.0f32, 0.0f32);
			for dy in -half..=half {
				for dx in -half..=half {
					let base = (reflect101_index(y + dy, height as i32) * width
						+ reflect101_index(x + dx, width as i32))
						* 3;
					sxx += products[base];
					syy += products[base + 1];
					sxy += products[base + 2];
				}
			}
			out[y as usize * width + x as usize] = if min_eigen {
				let diff = sxx - syy;
				0.5 * (sxx + syy - (diff * diff + 4.0 * sxy * sxy).max(0.0).sqrt())
			} else {
				let det = sxx * syy - sxy * sxy;
				let trace = sxx + syy;
				det - k * trace * trace
			};
		}
	}
	out
}

test_vk!(
	corner_score_harris_and_min_eigen_match_host_oracle,
	runtime,
	{
		let width = 8;
		let height = 6;
		// Lower-right quadrant lit: corner structure at (3,2)-ish.
		let mut values = vec![0.0f32; width * height];
		for y in 0..height {
			for x in 0..width {
				if x >= 4 && y >= 3 {
					values[y * width + x] = 1.0;
				}
			}
		}
		let image = Image::new(
			Matrix::new_from_f32_on(&runtime, [1, 1, height, width], &values)?,
			ImageLayout::Nchw,
			ImageFormat::Gray,
		)?;
		for (score_type, min_eigen) in [
			(oa::image::CornerScore::Harris, false),
			(oa::image::CornerScore::MinEigen, true),
		] {
			let out = oa::image::corner_score(&image, 3, 0.04, score_type)?;
			let actual = out.as_matrix().read_f32()?;
			let expected = corner_score_reference(&values, width, height, 3, 0.04, min_eigen);
			for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
				assert!(
					(a - e).abs() <= e.abs().max(1.0) * 1e-4,
					"min_eigen={min_eigen} index {index}: {a} vs {e}"
				);
			}
			// The quadrant corner should be among the top responses.
			let best = actual
				.iter()
				.enumerate()
				.max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
				.unwrap();
			assert!(best.1 > &0.0, "min_eigen={min_eigen}: no positive response");
		}
		Ok(())
	}
);

test_vk!(corner_fast_scores_corner_not_flat, runtime, {
	let width = 9;
	let height = 9;
	// L-corner: bottom-right 5x5 block at 1.0 on 0.0 background. The pixel
	// at the inside corner of the L junction (x=4,y=4) sees a bright arc.
	let mut values = vec![0.0f32; width * height];
	for y in 4..height {
		for x in 4..width {
			values[y * width + x] = 1.0;
		}
	}
	let image = Image::new(
		Matrix::new_from_f32_on(&runtime, [1, 1, height, width], &values)?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)?;
	let out = oa::image::corner_fast(&image, 9)?;
	let actual = out.as_matrix().read_f32()?;
	// Host oracle mirroring the kernel: circle of 16, arc of 9.
	let circle: [(i32, i32); 16] = [
		(0, -3),
		(1, -3),
		(2, -2),
		(3, -1),
		(3, 0),
		(3, 1),
		(2, 2),
		(1, 3),
		(0, 3),
		(-1, 3),
		(-2, 2),
		(-3, 1),
		(-3, 0),
		(-3, -1),
		(-2, -2),
		(-1, -3),
	];
	let mut expected = vec![0.0f32; width * height];
	for y in 3..height - 3 {
		for x in 3..width - 3 {
			let center = values[y * width + x];
			let mut diffs = [0.0f32; 16];
			for (i, (dx, dy)) in circle.iter().enumerate() {
				diffs[i] = values[(y as i32 + dy) as usize * width + (x as i32 + dx) as usize] - center;
			}
			let mut score = 0.0f32;
			for start in 0..16 {
				let mut bright = f32::MAX;
				let mut dark = f32::MAX;
				for j in 0..9 {
					let d = diffs[(start + j) % 16];
					bright = bright.min(d);
					dark = dark.min(-d);
				}
				score = score.max(bright.max(dark));
			}
			expected[y * width + x] = score.max(0.0);
		}
	}
	for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
		assert_eq!(a, e, "index {index}");
	}
	// Corner pixel (4,4) region must score 1.0 (diffs of 1.0 along an arc).
	assert!(
		actual[4 * width + 4] > 0.9,
		"corner score {}",
		actual[4 * width + 4]
	);
	// Deep flat regions score 0.
	assert_eq!(actual[7 * width + 7], 0.0);
	assert_eq!(actual[0], 0.0);
	Ok(())
});

test_vk!(bf_match_finds_nearest_hamming_neighbors, runtime, {
	// 3 query descriptors x 4 bytes, 4 train descriptors x 4 bytes.
	let query = Matrix::new_from_slice_on(
		&runtime,
		[3, 4],
		&[
			0b0000_0000u8,
			0b0000_0000,
			0b0000_0000,
			0b0000_0000, // all zero
			0b1111_1111u8,
			0b1111_1111,
			0b0000_0000,
			0b0000_0000, // two set bytes
			0b0000_0001u8,
			0b0000_0010,
			0b0000_0100,
			0b0000_1000, // sparse bits
		],
	)?;
	let train = Matrix::new_from_slice_on(
		&runtime,
		[4, 4],
		&[
			0b0000_0001u8,
			0b0000_0010,
			0b0000_0100,
			0b0000_1000, // = query[2], dist 0
			0b1111_1111u8,
			0b1111_1111,
			0b0000_0000,
			0b0000_0001, // near query[1], dist 1
			0b0000_0000u8,
			0b0000_0000,
			0b0000_0000,
			0b0000_0011, // near query[0], dist 2
			0b1010_1010u8,
			0b1010_1010,
			0b1010_1010,
			0b1010_1010, // alternating, far
		],
	)?;
	let out = oa::image::bf_match(&query, &train)?;
	assert_eq!(out.shape(), &[3, 4]);
	let rows = out.read_f32()?;
	// query[0] (all zero): best is train[2] (dist 2), second train[0] (dist 4).
	assert_eq!(rows[0], 2.0);
	assert_eq!(rows[1], 2.0);
	assert_eq!(rows[2], 0.0);
	assert_eq!(rows[3], 4.0);
	// query[1]: best train[1] (dist 1), second train[2] or[0] (dist 10/9 bits... compute).
	// train[0] dist = popcount(ff^01)+popcount(ff^02)+popcount(00^04)+popcount(00^08)
	//   = 7+7+1+1 = 16; train[2] dist = 8+8+1+1 = 18? recompute below.
	assert_eq!(rows[4], 1.0);
	assert_eq!(rows[5], 1.0);
	// query[2]: best train[0] (dist 0), second train[2] (dist 4+2=... compute).
	assert_eq!(rows[8], 0.0);
	assert_eq!(rows[9], 0.0);
	// Host oracle check on second-best distances too.
	for q in 0..3 {
		let mut dists: Vec<(u32, u32)> = (0..4)
			.map(|t| {
				let mut d = 0u32;
				for b in 0..4 {
					let qb = match q {
						0 => 0u8,
						1 => [0b1111_1111u8, 0b1111_1111, 0, 0][b],
						_ => [0b0000_0001u8, 0b0000_0010, 0b0000_0100, 0b0000_1000][b],
					};
					let tb = match t {
						0 => [0b0000_0001u8, 0b0000_0010, 0b0000_0100, 0b0000_1000][b],
						1 => [0b1111_1111u8, 0b1111_1111, 0, 0b0000_0001][b],
						2 => [0u8, 0, 0, 0b0000_0011][b],
						_ => [0b1010_1010u8, 0b1010_1010, 0b1010_1010, 0b1010_1010][b],
					};
					d += (qb ^ tb).count_ones();
				}
				(t as u32, d)
			})
			.collect();
		dists.sort_by_key(|&(_, d)| d);
		assert_eq!(rows[q * 4], dists[0].0 as f32, "q{q} best index");
		assert_eq!(rows[q * 4 + 1], dists[0].1 as f32, "q{q} best dist");
		assert_eq!(rows[q * 4 + 2], dists[1].0 as f32, "q{q} second index");
		assert_eq!(rows[q * 4 + 3], dists[1].1 as f32, "q{q} second dist");
	}
	Ok(())
});

test_vk!(good_features_to_track_selects_spread_corners, runtime, {
	let width = 16;
	let height = 16;
	// Three score peaks: strong at (4,4) with a near-twin at (5,4),
	// isolated medium at (12,12), isolated weak at (4,12).
	let mut values = vec![0.0f32; width * height];
	values[4 * width + 4] = 1.0;
	values[4 * width + 5] = 0.9;
	values[12 * width + 12] = 0.5;
	values[12 * width + 4] = 0.2;
	let image = Image::new(
		Matrix::new_from_f32_on(&runtime, [1, 1, height, width], &values)?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)?;
	// quality=0.1 -> threshold 0.1; min_distance=3 suppresses the (5,4) twin.
	let (keypoints, counts) = oa::image::good_features_to_track(&image, 8, 0.1, 3.0)?;
	assert_eq!(keypoints.shape(), &[1, 8, 3]);
	let count_rows = counts.read::<u32>()?;
	assert_eq!(count_rows, vec![3], "three separated survivors");
	let rows = keypoints.read_f32()?;
	let mut found: Vec<(f32, f32, f32)> = rows[..9]
		.chunks(3)
		.map(|c| (c[0], c[1], c[2]))
		.filter(|&(x, y, s)| s > 0.0 || x > 0.0 || y > 0.0)
		.collect();
	found.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
	assert_eq!(found.len(), 3, "rows {:?}", found);
	assert_eq!(found[0], (4.0, 4.0, 1.0));
	assert_eq!(found[1], (12.0, 12.0, 0.5));
	assert_eq!(found[2], (4.0, 12.0, 0.2));
	// max_corners cap truncates.
	let (keypoints, counts) = oa::image::good_features_to_track(&image, 2, 0.1, 3.0)?;
	assert_eq!(keypoints.shape(), &[1, 2, 3]);
	let count_rows = counts.read::<u32>()?;
	assert_eq!(count_rows[0], 3, "true survivor count exceeds cap");
	let rows = keypoints.read_f32()?;
	assert!(rows[..6].chunks(3).all(|c| c[2] > 0.0), "two filled rows");
	Ok(())
});

test_vk!(orb_detects_and_describes_keypoints, runtime, {
	let width = 96;
	let height = 96;
	// Asymmetric bright blob off-center: the 31x31 patch moments have a
	// definite centroid direction.
	let mut values = vec![0.0f32; width * height];
	for y in 20..70 {
		for x in 30..85 {
			let falloff = 1.0 - ((x as f32 - 55.0) / 35.0).powi(2);
			if falloff > 0.0 {
				values[y * width + x] = 0.8 * falloff + 0.2;
			}
		}
	}
	values[45 * width + 42] = 1.0;
	values[46 * width + 43] = 1.0;
	let image = Image::new(
		Matrix::new_from_f32_on(&runtime, [1, 1, height, width], &values)?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)?;
	let (keypoints, descriptors, counts) = oa::image::orb(&image, 64, 0.01, 5.0, 1, 18)?;
	assert_eq!(keypoints.shape(), &[1, 64, 6]);
	assert_eq!(descriptors.shape(), &[1, 64, 8]);
	assert_eq!(counts.read::<u32>()?.len(), 1);
	let claimed = counts.read::<u32>()?[0] as usize;
	assert!(claimed >= 1, "orb emitted no keypoints");
	let rows = keypoints.read_f32()?;
	let mut kpts: Vec<(f32, f32, f32, f32)> = rows[..claimed.min(64) * 6]
		.chunks(6)
		.map(|r| (r[0], r[1], r[2], r[3]))
		.collect();
	kpts.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
	assert!(
		kpts.iter().all(|&(x, y, angle, _)| {
			(18.0..78.0).contains(&x) && (18.0..78.0).contains(&y) && angle.abs() <= std::f32::consts::PI
		}),
		"keypoints out of edge/angle contract: {:?}",
		kpts
	);
	// Host oracle: mirror the kernel's patch moments at the top keypoint.
	let (kx, ky, kangle, _) = kpts[0];
	let ix = kx.round() as i32;
	let iy = ky.round() as i32;
	let (mut m10, mut m01) = (0.0f64, 0.0f64);
	for dy in -15..=15 {
		for dx in -15..=15 {
			let v = values[(iy + dy) as usize * width + (ix + dx) as usize] as f64;
			m10 += dx as f64 * v;
			m01 += dy as f64 * v;
		}
	}
	let expected = (m01 as f32).atan2(m10 as f32);
	assert!(
		(kangle - expected).abs() < 1e-4,
		"angle {} vs oracle {}",
		kangle,
		expected
	);
	let words = descriptors.read::<u32>()?;
	assert!(
		words[..8].iter().any(|&w| w != 0),
		"top keypoint descriptor is all zeros"
	);
	// Descriptor self-matches at distance zero through bf_match on the raw bytes.
	let bytes: Vec<u8> = words[..8].iter().flat_map(|w| w.to_le_bytes()).collect();
	let top_desc = Matrix::new_from_slice_on(&runtime, [1, 32], &bytes)?;
	let matched = oa::image::bf_match(&top_desc, &top_desc)?;
	let matched_rows = matched.read_f32()?;
	assert_eq!(&matched_rows[..2], &[0.0, 0.0], "self-match mismatch");
	Ok(())
});

fn textured_plane(width: usize, height: usize) -> Vec<f32> {
	// Deterministic texture with structure everywhere so the polynomial
	// expansion and LK gradient windows stay well conditioned.
	(0..height)
		.flat_map(|y| {
			(0..width).map(move |x| {
				let x = x as f32;
				let y = y as f32;
				((x * 0.31 + y * 0.17).sin()
					+ (x * 0.11 - y * 0.23).cos()
					+ ((x * x + y * y).sqrt() * 0.09).sin())
					* 0.25
					+ 0.5
			})
		})
		.collect()
}

fn shifted_plane(width: usize, height: usize, dx: i64, dy: i64) -> Vec<f32> {
	(0..height)
		.flat_map(|y| {
			(0..width).map(move |x| {
				let source_x = x as i64 - dx;
				let source_y = y as i64 - dy;
				if source_x < 0 || source_y < 0 || source_x >= width as i64 || source_y >= height as i64 {
					return 0.0;
				}
				let x = source_x as f32;
				let y = source_y as f32;
				((x * 0.31 + y * 0.17).sin()
					+ (x * 0.11 - y * 0.23).cos()
					+ ((x * x + y * y).sqrt() * 0.09).sin())
					* 0.25
					+ 0.5
			})
		})
		.collect()
}

test_vk!(optical_flow_pyr_lk_tracks_a_translation, runtime, {
	let width = 64usize;
	let height = 64usize;
	let previous = gray_image(
		&runtime,
		[1, 1, height, width],
		&textured_plane(width, height),
	)?;
	let next = gray_image(
		&runtime,
		[1, 1, height, width],
		&shifted_plane(width, height, 3, -2),
	)?;
	// Interior points inside the textured support.
	let point_rows = [32.0f32, 32.0, 20.0, 24.0, 44.0, 40.0, 16.0, 48.0];
	let points = Matrix::new_from_f32_on(&runtime, [4, 2], &point_rows)?;
	let (next_points, status, error) =
		oa::image::optical_flow_pyr_lk(&previous, &next, &points, 3, 5, 20, 1e-4)?;
	assert_eq!(next_points.shape(), &[1, 4, 2]);
	assert_eq!(status.shape(), &[1, 4]);
	assert_eq!(error.shape(), &[1, 4]);
	let tracked = next_points.read_f32()?;
	let found = status.read_f32()?;
	let residuals = error.read_f32()?;
	for (point, source) in [32.0f32, 32.0, 20.0, 24.0, 44.0, 40.0, 16.0, 48.0]
		.chunks(2)
		.enumerate()
	{
		assert_eq!(found[point], 1.0, "point {point} reported lost");
		let tx = source[0] + 3.0;
		let ty = source[1] - 2.0;
		assert!(
			(tracked[point * 2] - tx).abs() < 0.15 && (tracked[point * 2 + 1] - ty).abs() < 0.15,
			"point {point} tracked to ({}, {}), expected ({tx}, {ty})",
			tracked[point * 2],
			tracked[point * 2 + 1],
		);
		assert!(residuals[point] < 0.1, "point {point} residual too large");
	}
	Ok(())
});

test_vk!(optical_flow_pyr_lk_reports_lost_points, runtime, {
	let width = 64usize;
	let height = 64usize;
	// Flatten the top-left corner of the source so the LK gradient window
	// there has no texture to solve against.
	let mut source = textured_plane(width, height);
	for y in 0..24usize {
		for x in 0..24usize {
			source[y * width + x] = 0.0;
		}
	}
	let previous = gray_image(&runtime, [1, 1, height, width], &source)?;
	let next = gray_image(
		&runtime,
		[1, 1, height, width],
		&textured_plane(width, height),
	)?;
	let point_rows = [12.0f32, 12.0, 32.0, 48.0];
	let points = Matrix::new_from_f32_on(&runtime, [2, 2], &point_rows)?;
	let (_tracked, status, _error) =
		oa::image::optical_flow_pyr_lk(&previous, &next, &points, 3, 5, 20, 1e-4)?;
	let found = status.read_f32()?;
	assert_eq!(found[0], 0.0, "flat-window point should report lost");
	assert_eq!(found[1], 1.0, "textured point should report found");
	Ok(())
});

test_vk!(optical_flow_pyr_lk_rejects_invalid_inputs, runtime, {
	let width = 64usize;
	let height = 64usize;
	let previous = gray_image(
		&runtime,
		[1, 1, height, width],
		&textured_plane(width, height),
	)?;
	let next = gray_image(
		&runtime,
		[1, 1, height, width],
		&textured_plane(width, height),
	)?;
	let points = Matrix::new_from_f32_on(&runtime, [1, 2], &[32.0f32, 32.0])?;
	for (levels, window, iterations, min_eigen) in [
		(0, 10, 20, 1e-4),
		(4, 0, 20, 1e-4),
		(4, 10, 0, 1e-4),
		(4, 10, 20, 0.0),
		(4, 10, 20, f32::NAN),
		// Coarsest level can no longer fit the window.
		(16, 10, 20, 1e-4),
	] {
		let error = oa::image::optical_flow_pyr_lk(
			&previous, &next, &points, levels, window, iterations, min_eigen,
		)
		.err()
		.unwrap();
		assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	}
	// Mismatched images and malformed point matrices.
	let other = gray_image(
		&runtime,
		[1, 1, height / 2, width],
		&textured_plane(width, height / 2),
	)?;
	assert_eq!(
		oa::image::optical_flow_pyr_lk(&previous, &other, &points, 4, 10, 20, 1e-4)
			.err()
			.unwrap()
			.kind(),
		ErrorKind::InvalidArgument
	);
	for (shape, values) in [
		(vec![0, 2], vec![]),
		(vec![4, 1], vec![0.0; 4]),
		(vec![1, 4, 2, 1], vec![0.0; 8]),
	] {
		let bad = Matrix::new_from_f32_on(&runtime, shape.clone(), &values)?;
		assert_eq!(
			oa::image::optical_flow_pyr_lk(&previous, &next, &bad, 4, 10, 20, 1e-4)
				.err()
				.unwrap()
				.kind(),
			ErrorKind::InvalidArgument,
			"points shape {shape:?} should be rejected"
		);
	}
	Ok(())
});

test_vk!(optical_flow_farneback_recovers_a_translation, runtime, {
	let width = 64usize;
	let height = 64usize;
	let previous = gray_image(
		&runtime,
		[1, 1, height, width],
		&textured_plane(width, height),
	)?;
	let next = gray_image(
		&runtime,
		[1, 1, height, width],
		&shifted_plane(width, height, 2, 1),
	)?;
	let flow = oa::image::optical_flow_farneback(&previous, &next, 3, 7, 3, 5, 1.1)?;
	assert_eq!(flow.shape(), &[1, 2, height, width]);
	let field = flow.read_f32()?;
	let plane_size = width * height;
	// Interior region away from the shifted-in zero border and the
	// reflect101 polynomial edges.
	let mut deltas = Vec::new();
	for y in 12..52usize {
		for x in 12..52usize {
			deltas.push((field[y * width + x], field[plane_size + y * width + x]));
		}
	}
	let mut xs: Vec<f32> = deltas.iter().map(|pair| pair.0).collect();
	let mut ys: Vec<f32> = deltas.iter().map(|pair| pair.1).collect();
	xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
	ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let median_x = xs[xs.len() / 2];
	let median_y = ys[ys.len() / 2];
	assert!(
		(median_x - 2.0).abs() < 0.5 && (median_y - 1.0).abs() < 0.5,
		"median flow ({median_x}, {median_y}), expected approximately (2, 1)"
	);
	let within: usize = deltas
		.iter()
		.filter(|&&(fx, fy)| (fx - 2.0).abs() < 0.75 && (fy - 1.0).abs() < 0.75)
		.count();
	assert!(
		within * 100 > deltas.len() * 80,
		"only {within}/{} interior vectors within tolerance",
		deltas.len()
	);
	Ok(())
});

test_vk!(
	optical_flow_farneback_returns_zero_for_identical_frames,
	runtime,
	{
		let width = 48usize;
		let height = 48usize;
		let frame = gray_image(
			&runtime,
			[1, 1, height, width],
			&textured_plane(width, height),
		)?;
		let flow = oa::image::optical_flow_farneback(&frame, &frame, 3, 5, 3, 5, 1.1)?;
		let field = flow.read_f32()?;
		let peak = field
			.iter()
			.fold(0.0f32, |peak, value| peak.max(value.abs()));
		assert!(
			peak < 0.05,
			"identical frames produced flow magnitude {peak}"
		);
		Ok(())
	}
);

test_vk!(optical_flow_farneback_rejects_invalid_inputs, runtime, {
	let width = 64usize;
	let height = 64usize;
	let previous = gray_image(
		&runtime,
		[1, 1, height, width],
		&textured_plane(width, height),
	)?;
	let next = gray_image(
		&runtime,
		[1, 1, height, width],
		&textured_plane(width, height),
	)?;
	for (levels, window, iterations, poly_n, poly_sigma) in [
		(0, 7, 3, 5, 1.1),
		(4, 0, 3, 5, 1.1),
		(4, 7, 0, 5, 1.1),
		(4, 7, 3, 0, 1.1),
		(4, 7, 3, 5, 0.0),
		(4, 7, 3, 5, f32::NAN),
		(16, 7, 3, 5, 1.1),
	] {
		let error = oa::image::optical_flow_farneback(
			&previous, &next, levels, window, iterations, poly_n, poly_sigma,
		)
		.err()
		.unwrap();
		assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	}
	Ok(())
});

test_vk!(stereo_bm_recovers_a_disparity, runtime, {
	let width = 64usize;
	let height = 64usize;
	// Constant disparity d = 8: right(y, x) = left(y, x + 8) so the
	// match for left(x) lands at right(x - 8).
	let disparity = 8i64;
	let left = gray_image(
		&runtime,
		[1, 1, height, width],
		&textured_plane(width, height),
	)?;
	let right = gray_image(
		&runtime,
		[1, 1, height, width],
		&shifted_plane(width, height, -disparity, 0),
	)?;
	let map = oa::image::stereo_bm(&left, &right, 16, 11, 5)?;
	assert_eq!(map.shape(), &[1, 1, height, width]);
	let field = map.read_f32()?;
	// Valid region starts at block_radius + num_disparities - 1 = 20.
	for y in 6..58usize {
		for x in 0..20usize {
			assert_eq!(
				field[y * width + x],
				-1.0,
				"border pixel ({x}, {y}) not marked invalid"
			);
		}
	}
	let mut xs: Vec<f32> = Vec::new();
	for y in 8..56usize {
		for x in 24..56usize {
			xs.push(field[y * width + x]);
		}
	}
	xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let median = xs[xs.len() / 2];
	assert!(
		(median - 8.0).abs() < 0.5,
		"median disparity {median}, expected approximately 8"
	);
	let within = xs.iter().filter(|&&v| (v - 8.0).abs() < 0.75).count();
	assert!(
		within * 100 > xs.len() * 80,
		"only {within}/{} interior disparities within tolerance",
		xs.len()
	);
	Ok(())
});

test_vk!(stereo_bm_marks_ambiguous_pixels, runtime, {
	let width = 64usize;
	let height = 64usize;
	// A flat plane gives identical SAD at every disparity, so the
	// uniqueness margin rejects every interior pixel.
	let flat = vec![0.5f32; width * height];
	let left = gray_image(&runtime, [1, 1, height, width], &flat)?;
	let right = gray_image(&runtime, [1, 1, height, width], &flat)?;
	let map = oa::image::stereo_bm(&left, &right, 16, 11, 5)?;
	let field = map.read_f32()?;
	assert!(
		field.iter().all(|&v| v == -1.0),
		"flat input produced non-(-1) disparities"
	);
	Ok(())
});

test_vk!(stereo_bm_rejects_invalid_inputs, runtime, {
	let width = 64usize;
	let height = 64usize;
	let left = gray_image(
		&runtime,
		[1, 1, height, width],
		&textured_plane(width, height),
	)?;
	let other = gray_image(
		&runtime,
		[1, 1, height, width / 2],
		&textured_plane(width / 2, height),
	)?;
	for (num_disparities, block_size) in [(0, 11), (16, 0), (16, 2), (16, 10), (128, 11)] {
		let error = oa::image::stereo_bm(&left, &left, num_disparities, block_size, 5)
			.err()
			.unwrap();
		assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	}
	let error = oa::image::stereo_bm(&left, &other, 16, 11, 5)
		.err()
		.unwrap();
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	Ok(())
});

test_vk!(reproject_image_to_3d_projects_points, runtime, {
	let height = 8usize;
	let width = 8usize;
	// Constant disparity 4 with Q that swaps/scales axes:
	// X = x + 2d, Y = y + d, Z = 3d, W = 1.
	let disparity = Matrix::new_from_f32_on(
		&runtime,
		[1, 1, height, width],
		&vec![4.0f32; height * width],
	)?;
	let q = Matrix::new_from_f32_on(
		&runtime,
		[4, 4],
		&[
			1.0, 0.0, 2.0, 0.0, //
			0.0, 1.0, 1.0, 0.0, //
			0.0, 0.0, 3.0, 0.0, //
			0.0, 0.0, 0.0, 1.0,
		],
	)?;
	let points = oa::image::reproject_image_to_3d(&disparity, &q)?;
	assert_eq!(points.shape(), &[1, 3, height, width]);
	let field = points.read_f32()?;
	let plane_size = height * width;
	for y in 0..height {
		for x in 0..width {
			let index = y * width + x;
			assert_eq!(field[index], x as f32 + 8.0);
			assert_eq!(field[plane_size + index], y as f32 + 4.0);
			assert_eq!(field[2 * plane_size + index], 12.0);
		}
	}
	Ok(())
});

test_vk!(reproject_image_to_3d_rejects_invalid_inputs, runtime, {
	let disparity = Matrix::new_from_f32_on(&runtime, [1, 2, 4, 4], &[0.0f32; 32])?;
	let q = Matrix::new_from_f32_on(&runtime, [4, 4], &[0.0f32; 16])?;
	let error = oa::image::reproject_image_to_3d(&disparity, &q)
		.err()
		.unwrap();
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	let disparity = Matrix::new_from_f32_on(&runtime, [1, 1, 4, 4], &[0.0f32; 16])?;
	let bad_q = Matrix::new_from_f32_on(&runtime, [3, 3], &[0.0f32; 9])?;
	let error = oa::image::reproject_image_to_3d(&disparity, &bad_q)
		.err()
		.unwrap();
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	Ok(())
});

// ---- fast_nl_means_denoising -------------------------------------------------

test_vk!(nl_means_suppresses_an_impulse, runtime, {
	// 16x16 zeros with a single 1.0 impulse at the center: neighbour
	// patches disagree on the hot pixel, so its own weight collapses
	// and the output must land near the clean zero background.
	let mut values = vec![0.0f32; 16 * 16];
	values[8 * 16 + 8] = 1.0;
	let image = gray_image(&runtime, [1, 1, 16, 16], &values)?;
	let out = oa::image::fast_nl_means_denoising(&image, 0.1, 7, 11)?;
	let data: Vec<f32> = out.as_matrix().read_f32()?;
	assert!(
		data[8 * 16 + 8] < 0.4,
		"impulse should be suppressed: got {}",
		data[8 * 16 + 8]
	);
	Ok(())
});

test_vk!(nl_means_preserves_a_flat_image, runtime, {
	let values = vec![0.375f32; 12 * 12];
	let image = gray_image(&runtime, [1, 1, 12, 12], &values)?;
	let out = oa::image::fast_nl_means_denoising(&image, 0.1, 5, 9)?;
	let data: Vec<f32> = out.as_matrix().read_f32()?;
	assert!(data.iter().all(|v| (*v - 0.375).abs() < 1e-4));
	Ok(())
});

test_vk!(nl_means_rejects_invalid_parameters, runtime, {
	let image = gray_image(&runtime, [1, 1, 8, 8], &vec![0.5f32; 64])?;
	for (h, tw, sw) in [
		(0.0, 7, 11),
		(f32::NAN, 7, 11),
		(0.1, 2, 11),
		(0.1, 8, 11),
		(0.1, 7, 10),
	] {
		assert!(oa::image::fast_nl_means_denoising(&image, h, tw, sw).is_err());
	}
	Ok(())
});

// ---- pyr_mean_shift_filtering -----------------------------------------------

test_vk!(mean_shift_keeps_region_levels, runtime, {
	// Two flat halves split vertically at x = 16. Deep-interior pixels
	// converge inside their own region and must keep their level;
	// the boundary column is the only place allowed to migrate.
	let mut values = vec![0.0f32; 16 * 32];
	for y in 0..16 {
		for x in 16..32 {
			values[y * 32 + x] = 0.8;
		}
	}
	let image = gray_image(&runtime, [1, 1, 16, 32], &values)?;
	let out = oa::image::pyr_mean_shift_filtering(&image, 4.0, 0.25, 1)?;
	let data: Vec<f32> = out.as_matrix().read_f32()?;
	for y in 2..14 {
		assert!(
			(data[y * 32 + 4] - 0.0).abs() < 0.05,
			"dark interior drifted: {}",
			data[y * 32 + 4]
		);
		assert!(
			(data[y * 32 + 28] - 0.8).abs() < 0.05,
			"bright interior drifted: {}",
			data[y * 32 + 28]
		);
	}
	Ok(())
});

test_vk!(mean_shift_smooths_zero_mean_noise, runtime, {
	// A flat 0.5 field plus small zero-mean dither: converged colors
	// must cluster near 0.5 rather than keep the per-pixel noise.
	let mut values = vec![0.5f32; 16 * 16];
	for (i, value) in values.iter_mut().enumerate() {
		*value += ((i * 37) % 11) as f32 * 0.02 - 0.1;
	}
	let image = gray_image(&runtime, [1, 1, 16, 16], &values)?;
	let out = oa::image::pyr_mean_shift_filtering(&image, 6.0, 0.5, 1)?;
	let data: Vec<f32> = out.as_matrix().read_f32()?;
	assert!(data.iter().all(|v| (*v - 0.5).abs() < 0.12));
	Ok(())
});

test_vk!(mean_shift_runs_a_two_level_pyramid, runtime, {
	let mut values = vec![0.0f32; 32 * 32];
	for y in 0..32 {
		for x in 16..32 {
			values[y * 32 + x] = 0.9;
		}
	}
	let image = gray_image(&runtime, [1, 1, 32, 32], &values)?;
	let out = oa::image::pyr_mean_shift_filtering(&image, 4.0, 0.2, 2)?;
	assert_eq!(out.as_matrix().shape(), &[1, 1, 32, 32]);
	let data: Vec<f32> = out.as_matrix().read_f32()?;
	assert!((data[8 * 32 + 4] - 0.0).abs() < 0.1);
	assert!((data[8 * 32 + 28] - 0.9).abs() < 0.1);
	Ok(())
});

test_vk!(mean_shift_rejects_invalid_parameters, runtime, {
	let image = gray_image(&runtime, [1, 1, 16, 16], &vec![0.5f32; 16 * 16])?;
	for (sp, sr, level) in [
		(0.0, 10.0, 1),
		(f32::NAN, 10.0, 1),
		(4.0, 0.0, 1),
		(4.0, 10.0, 0),
		(4.0, 10.0, 5),
	] {
		assert!(oa::image::pyr_mean_shift_filtering(&image, sp, sr, level).is_err());
	}
	Ok(())
});
