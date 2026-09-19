use oa::ml::Module;

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= tolerance,
			"RoPE element {index}: expected {expected}, found {actual}"
		);
	}
}

fn host_rope(
	input: &[f32],
	tokens: usize,
	num_heads: usize,
	head_dim: usize,
	theta_base: f32,
	position_offset: u32,
) -> Vec<f32> {
	let mut output = vec![0.0; input.len()];
	let half_dim = head_dim / 2;
	for token in 0..tokens {
		for head in 0..num_heads {
			for pair in 0..half_dim {
				let frequency = (token as f32 + position_offset as f32)
					* theta_base.powf(-2.0 * pair as f32 / head_dim as f32);
				let (sine, cosine) = frequency.sin_cos();
				let first = token * num_heads * head_dim + head * head_dim + pair;
				let second = first + half_dim;
				output[first] = input[first] * cosine - input[second] * sine;
				output[second] = input[first] * sine + input[second] * cosine;
			}
		}
	}
	output
}

fn host_loss(input: &[f32], linear_weight: &[f32], linear_bias: &[f32], targets: &[u32]) -> f32 {
	let rotated = host_rope(input, 2, 1, 4, 100.0, 3);
	let mut loss = 0.0_f64;
	for (row_index, row) in rotated.as_chunks::<4>().0.iter().enumerate() {
		let logits = (0..2)
			.map(|class| {
				row
					.iter()
					.zip(&linear_weight[class * 4..(class + 1) * 4])
					.map(|(input, weight)| input * weight)
					.sum::<f32>()
					+ linear_bias[class]
			})
			.collect::<Vec<_>>();
		let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
		let denominator = logits
			.iter()
			.map(|value| f64::from(*value - maximum).exp())
			.sum::<f64>();
		loss += f64::from(maximum) + denominator.ln() - f64::from(logits[targets[row_index] as usize]);
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

test_vk!(rope_matches_forward_and_reverse_oracles, engine, {
	let values = [0.2_f32, -0.4, 0.8, 1.1, -0.7, 0.3, 0.5, -0.2];
	let input = oa::Matrix::from_f32(&engine, [2, 4], &values)?;
	let output = oa::ml::matrix::rope(&input, 1, 4, 100.0, 3)?;
	assert_close(
		&output.read_f32()?,
		&host_rope(&values, 2, 1, 4, 100.0, 3),
		2.0e-6,
	);

	let linear_weight = [0.4_f32, -0.2, 0.7, 0.1, -0.6, 0.3, 0.2, -0.5];
	let linear_bias = [0.05_f32, -0.15];
	let targets = [0_u32, 1];
	let expected = numerical_gradient(&values, |candidate| {
		host_loss(candidate, &linear_weight, &linear_bias, &targets)
	});
	let embedding =
		oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [2, 4], &values)?)?;
	let linear = oa::ml::nn::Linear::from_matrices(
		oa::Matrix::from_f32(&engine, [2, 4], &linear_weight)?,
		oa::Matrix::from_f32(&engine, [2], &linear_bias)?,
	)?;
	let indices = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
	let target_matrix = oa::Matrix::from_slice(&engine, [2], &targets)?;
	let tape = oa::ml::GradientTape::new();
	let rotated = oa::ml::matrix::rope(&embedding.forward(&indices)?, 1, 4, 100.0, 3)?;
	let loss = oa::ml::loss::cross_entropy(&linear.forward(&rotated)?, &target_matrix)?;
	tape.backward(&loss)?;
	assert_close(
		&embedding
			.weight()
			.gradient()
			.expect("RoPE input gradient is missing")
			.read_f32()?,
		&expected,
		5.0e-4,
	);

	let module = oa::ml::nn::Rope::new(1, 4, 100.0)?;
	assert_eq!(module.num_heads(), 1);
	assert_eq!(module.head_dim(), 4);
	assert_eq!(module.theta_base(), 100.0);
	assert!(module.all_parameters()?.is_empty());
	assert_close(
		&module.forward(&input)?.read_f32()?,
		&host_rope(&values, 2, 1, 4, 100.0, 0),
		2.0e-6,
	);
	Ok(())
});

test_vk!(rope_rejects_invalid_contracts, engine, {
	for (heads, head_dim, theta) in [(0, 4, 100.0), (1, 3, 100.0), (1, 4, 0.0)] {
		assert_eq!(
			oa::ml::nn::Rope::new(heads, head_dim, theta)
				.err()
				.expect("invalid RoPE configuration was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	let wrong_width = oa::Matrix::from_f32(&engine, [2, 3], &[1.0; 6])?;
	assert_eq!(
		oa::ml::matrix::rope(&wrong_width, 1, 4, 100.0, 0)
			.err()
			.expect("wrong-width RoPE input was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});
