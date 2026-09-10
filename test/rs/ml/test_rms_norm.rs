fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		let error = (actual - expected).abs();
		assert!(
			error <= tolerance,
			"{label} element {index}: expected {expected}, found {actual}, error {error}"
		);
	}
}

fn host_rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
	input
		.chunks_exact(weight.len())
		.flat_map(|row| {
			let mean_square =
				row.iter()
					.map(|value| f64::from(*value) * f64::from(*value))
					.sum::<f64>() / row.len() as f64;
			let inverse_rms = (mean_square + f64::from(epsilon)).sqrt().recip() as f32;
			row.iter()
				.zip(weight)
				.map(move |(value, weight)| value * inverse_rms * weight)
		})
		.collect()
}

fn host_loss(
	input: &[f32],
	rms_weight: &[f32],
	linear_weight: &[f32],
	linear_bias: &[f32],
	targets: &[u32],
	epsilon: f32,
) -> f32 {
	let normalized = host_rms_norm(input, rms_weight, epsilon);
	let width = rms_weight.len();
	let classes = linear_bias.len();
	let mut loss = 0.0_f64;
	for (row_index, row) in normalized.chunks_exact(width).enumerate() {
		let logits = (0..classes)
			.map(|class| {
				row.iter()
					.zip(&linear_weight[class * width..(class + 1) * width])
					.map(|(input, weight)| input * weight)
					.sum::<f32>() + linear_bias[class]
			})
			.collect::<Vec<_>>();
		let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
		let denominator = logits
			.iter()
			.map(|value| f64::from(*value - maximum).exp())
			.sum::<f64>();
		loss +=
			f64::from(maximum) + denominator.ln() - f64::from(logits[targets[row_index] as usize]);
	}
	(loss / targets.len() as f64) as f32
}

fn numerical_gradient(values: &[f32], loss: impl Fn(&[f32]) -> f32) -> Vec<f32> {
	const DELTA: f32 = 1.0e-3;
	(0..values.len())
		.map(|index| {
			let mut below = values.to_vec();
			let mut above = values.to_vec();
			below[index] -= DELTA;
			above[index] += DELTA;
			(loss(&above) - loss(&below)) / (2.0 * DELTA)
		})
		.collect()
}

test_vk!(
	rms_norm_matches_rank_three_forward_and_reverse_oracles,
	engine,
	{
		const EPSILON: f32 = 0.01;
		let input_values = [
			-1.2_f32, 0.3, 2.0, 0.5, -0.8, 1.1, 1.7, -0.4, 0.2, -0.1, 0.9, -1.4,
		];
		let rms_weight_values = [0.8_f32, -0.5, 1.3];
		let linear_weight_values = [0.4_f32, -0.2, 0.7, -0.6, 0.3, 0.1];
		let linear_bias_values = [0.05_f32, -0.15];
		let targets = [0_u32, 1, 0, 1];

		let input = oa::Matrix::from_f32(&engine, [2, 2, 3], &input_values)?;
		let rms_weight = oa::Matrix::from_f32(&engine, [3], &rms_weight_values)?;
		let forward = oa::ml::matrix::rms_norm(&input, &rms_weight, EPSILON)?;
		assert_eq!(forward.shape(), [2, 2, 3]);
		assert_close(
			&forward.read_f32()?,
			&host_rms_norm(&input_values, &rms_weight_values, EPSILON),
			2.0e-5,
			"RMSNorm forward",
		);

		let expected_input = numerical_gradient(&input_values, |values| {
			host_loss(
				values,
				&rms_weight_values,
				&linear_weight_values,
				&linear_bias_values,
				&targets,
				EPSILON,
			)
		});
		let expected_weight = numerical_gradient(&rms_weight_values, |values| {
			host_loss(
				&input_values,
				values,
				&linear_weight_values,
				&linear_bias_values,
				&targets,
				EPSILON,
			)
		});

		let indices = oa::Matrix::from_slice(&engine, [2, 2], &[0_u32, 1, 2, 3])?;
		let target_matrix = oa::Matrix::from_slice(&engine, [4], &targets)?;
		let embedding = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
			&engine,
			[4, 3],
			&input_values,
		)?)?;
		let rms_norm = oa::ml::nn::RmsNorm::from_matrix(
			oa::Matrix::from_f32(&engine, [3], &rms_weight_values)?,
			EPSILON,
		)?;
		let linear = oa::ml::nn::Linear::from_matrices(
			oa::Matrix::from_f32(&engine, [2, 3], &linear_weight_values)?,
			oa::Matrix::from_f32(&engine, [2], &linear_bias_values)?,
		)?;
		let tape = oa::ml::GradientTape::new();
		let embedded = embedding.forward(&indices)?;
		let normalized = rms_norm.forward(&embedded)?.reshape([4, 3])?;
		let loss = oa::ml::loss::cross_entropy(&linear.forward(&normalized)?, &target_matrix)?;
		tape.backward(&loss)?;
		assert_close(
			&embedding
				.weight()
				.gradient()
				.expect("RMSNorm input gradient is missing")
				.read_f32()?,
			&expected_input,
			1.2e-3,
			"RMSNorm input gradient",
		);
		assert_close(
			&rms_norm
				.weight()
				.gradient()
				.expect("RMSNorm weight gradient is missing")
				.read_f32()?,
			&expected_weight,
			1.2e-3,
			"RMSNorm weight gradient",
		);
		assert_eq!(rms_norm.dimension(), 3);
		assert_eq!(rms_norm.epsilon(), EPSILON);
		Ok(())
	}
);

test_vk!(rms_norm_rejects_invalid_contracts, engine, {
	let weight = oa::Matrix::from_f32(&engine, [3], &[1.0; 3])?;
	let input = oa::Matrix::from_f32(&engine, [2, 3], &[1.0; 6])?;
	for epsilon in [0.0_f32, -1.0, f32::NAN] {
		assert_eq!(
			oa::ml::matrix::rms_norm(&input, &weight, epsilon)
				.err()
				.expect("invalid RMSNorm epsilon was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	let wrong_weight = oa::Matrix::from_f32(&engine, [2], &[1.0; 2])?;
	assert_eq!(
		oa::ml::matrix::rms_norm(&input, &wrong_weight, 1.0e-6)
			.err()
			.expect("wrong RMSNorm weight shape was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});

test_vk!(rms_norm_weight_adjoint_uses_core_sum_axis, engine, {
	let indices = oa::Matrix::from_slice(&engine, [2, 2], &[0_u32, 1, 2, 3])?;
	let targets = oa::Matrix::from_slice(&engine, [4], &[0_u32, 1, 0, 1])?;
	let embedding = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
		&engine,
		[4, 3],
		&[
			-1.2_f32, 0.3, 2.0, 0.5, -0.8, 1.1, 1.7, -0.4, 0.2, -0.1, 0.9, -1.4,
		],
	)?)?;
	let rms_norm = oa::ml::nn::RmsNorm::from_matrix(
		oa::Matrix::from_f32(&engine, [3], &[0.8_f32, -0.5, 1.3])?,
		0.01,
	)?;
	let linear = oa::ml::nn::Linear::from_matrices(
		oa::Matrix::from_f32(&engine, [2, 3], &[0.4_f32, -0.2, 0.7, -0.6, 0.3, 0.1])?,
		oa::Matrix::from_f32(&engine, [2], &[0.05_f32, -0.15])?,
	)?;

	let (plan, ()) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let embedded = embedding.forward(&indices)?;
		let normalized = rms_norm.forward(&embedded)?.reshape([4, 3])?;
		let loss = oa::ml::loss::cross_entropy(&linear.forward(&normalized)?, &targets)?;
		tape.backward(&loss)
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("RmsNorm"))
		.expect("RMSNorm executable report must be valid JSON");
	let kernels = report["nodes"]
		.as_array()
		.expect("RMSNorm executable nodes must be an array")
		.iter()
		.map(|node| {
			node["kernel"]
				.as_str()
				.expect("kernel name must be a string")
		})
		.collect::<Vec<_>>();
	assert!(kernels.contains(&"matrix.sum_axis.f32"));
	assert!(
		!kernels
			.iter()
			.any(|kernel| kernel.contains("rms_norm_parameter_backward"))
	);
	let nodes = report["nodes"]
		.as_array()
		.expect("RMSNorm executable nodes must be an array");
	let sum_axis = nodes
		.iter()
		.find(|node| node["kernel"] == "matrix.sum_axis.f32")
		.expect("RMSNorm parameter adjoint must route through Core Sum");
	assert_eq!(
		sum_axis["physical_write"]["writes"][0]["domain"],
		"output_elements"
	);
	let rms_backward = nodes
		.iter()
		.find(|node| node["kernel"] == "ml.matrix.rms_norm_backward.f32")
		.expect("RMSNorm backward candidate must be visible");
	assert_eq!(
		rms_backward["physical_write"]["writes"]
			.as_array()
			.expect("RMSNorm writes must be an array")
			.len(),
		2
	);
	assert_eq!(
		rms_backward["physical_write"]["workspace"],
		"exclusive_per_workgroup"
	);
	engine.submit(&plan)?.wait()?;
	Ok(())
});
