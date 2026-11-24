use oa::{ErrorKind, Image, ImageFormat, ImageLayout, Matrix, image::HistogramComparison};

fn gray_image(engine: &oa::Engine, shape: [usize; 4], values: &[f32]) -> oa::Result<Image> {
	Image::new(
		Matrix::new_from_f32_on(engine, shape, values)?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)
}

test_vk!(calc_hist_matches_an_independent_bucket_oracle, engine, {
	// [1,1,2,4]: values 0,.25,.5,.75 / 1.5,.5,-.5,.999 over [0,1) in 4 bins.
	let values = [0.0, 0.25, 0.5, 0.75, 1.5, 0.5, -0.5, 0.999];
	let image = gray_image(&engine, [1, 1, 2, 4], &values)?;
	let histogram = image.calc_hist(4, 0.0, 1.0)?;
	assert_eq!(histogram.shape(), &[1, 1, 4]);
	assert_eq!(histogram.dtype(), oa::DType::U32);
	let counts: Vec<u32> = histogram.read()?;
	assert_eq!(counts, vec![1, 1, 2, 2]);
	Ok(())
});

test_vk!(calc_hist_counts_each_plane_independently, engine, {
	// [2,1,1,3]: batch plane 0 uses bins 0 and 2, plane 1 uses bins 1 and 3.
	let values = [0.0, 0.0, 0.5, 0.25, 0.75, 0.25];
	let image = gray_image(&engine, [2, 1, 1, 3], &values)?;
	let histogram = oa::image::calc_hist(&image, 4, 0.0, 1.0)?;
	assert_eq!(histogram.shape(), &[2, 1, 4]);
	let counts: Vec<u32> = histogram.read()?;
	assert_eq!(counts, vec![2, 0, 1, 0, 0, 2, 0, 1]);
	Ok(())
});

test_vk!(calc_hist_rejects_invalid_parameters, engine, {
	let _ = engine;
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

test_vk!(equalize_hist_spreads_the_cumulative_distribution, engine, {
	// [1,1,1,4] = [0,0,0,0.9]: buckets 0,0,0,230; cdf_min=3, total=4.
	// lut[0]=(3-3)/1=0, lut[230]=(4-3)/1=1 -> [0,0,0,1].
	let image = gray_image(&engine, [1, 1, 1, 4], &[0.0, 0.0, 0.0, 0.9])?;
	let equalized = image.equalize_hist()?;
	assert_eq!(equalized.layout(), image.layout());
	assert_eq!(equalized.format(), image.format());
	let result = equalized.as_matrix().read_f32()?;
	assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0]);
	Ok(())
});

test_vk!(equalize_hist_leaves_a_degenerate_plane_at_zero, engine, {
	// Single-valued plane: cdf_min == total -> all outputs 0.
	let image = gray_image(&engine, [1, 1, 2, 3], &[0.5; 6])?;
	let result = image.equalize_hist()?.as_matrix().read_f32()?;
	assert_eq!(result, vec![0.0; 6]);
	Ok(())
});

test_vk!(compare_hist_matches_independent_metric_oracles, engine, {
	// Orthogonal one-hot rows: correlation -0.5, chi-square 1.0,
	// intersection 0.0, bhattacharyya 1.0. Identical rows return the
	// perfect-match score for each metric.
	let first = Matrix::new_from_slice_on(&engine, [1, 1, 3], &[1u32, 0, 0])?;
	let second = first.new_from_slice_on_same_engine([1, 1, 3], &[0u32, 1, 0])?;
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
	engine,
	{
		let _ = engine;
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

test_vk!(captured_histogram_plan_replays_identical_counts, engine, {
	let values = [0.0, 0.25, 0.5, 0.75, 1.5, 0.5, -0.5, 0.999];
	let image = gray_image(&engine, [1, 1, 2, 4], &values)?;
	let expected = image.calc_hist(4, 0.0, 1.0)?.read::<u32>()?;
	let (plan, histogram) = engine.capture(|| image.calc_hist(4, 0.0, 1.0))?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("image calc_hist"))
		.expect("valid execution graph report");
	assert_eq!(report["stats"]["dispatches"], 2);
	engine.submit(&plan)?.wait()?;
	let first_run: Vec<u32> = histogram.read()?;
	assert_eq!(first_run, expected);
	// Replaying the same plan must clear the accumulator; a missing
	// clear stage would double every bucket.
	engine.submit(&plan)?.wait()?;
	let second_run: Vec<u32> = histogram.read()?;
	assert_eq!(second_run, expected);
	Ok(())
});

test_vk!(captured_calc_hist_feeds_compare_hist_in_one_plan, engine, {
	let values = [0.0, 0.25, 0.5, 0.75, 1.5, 0.5, -0.5, 0.999];
	let image = gray_image(&engine, [1, 1, 2, 4], &values)?;
	let other = gray_image(
		&engine,
		[1, 1, 2, 4],
		&[0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
	)?;
	let (plan, (first_histogram, scores)) = engine.capture(|| {
		let first = image.calc_hist(4, 0.0, 1.0)?;
		let second = other.calc_hist(4, 0.0, 1.0)?;
		let score = oa::image::compare_hist(&first, &second, HistogramComparison::Intersection)?;
		Ok((first, score))
	})?;
	engine.submit(&plan)?.wait()?;
	let counts: Vec<u32> = first_histogram.read()?;
	assert_eq!(counts, vec![1, 1, 2, 2]);
	// second = [4,0,4,0]: overlap = min(1,4)+min(1,0)+min(2,4)+min(2,0) = 3;
	// normalized by first total 6 -> 0.5.
	let value = scores.read_f32()?;
	assert!((value[0] - 0.5).abs() < 1e-6, "got {}", value[0]);
	Ok(())
});
