use oa::ml::Module as _;

fn host_softmax(input: &[f32], rows: usize, columns: usize) -> Vec<f32> {
	let mut output = vec![0.0; input.len()];
	for row in 0..rows {
		let values = &input[row * columns..(row + 1) * columns];
		let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
		let sum = values
			.iter()
			.map(|value| (value - maximum).exp())
			.sum::<f32>();
		for column in 0..columns {
			output[row * columns + column] = (values[column] - maximum).exp() / sum;
		}
	}
	output
}

fn host_log_softmax(input: &[f32], rows: usize, columns: usize) -> Vec<f32> {
	let mut output = vec![0.0; input.len()];
	for row in 0..rows {
		let values = &input[row * columns..(row + 1) * columns];
		let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
		let log_normalizer = maximum
			+ values
				.iter()
				.map(|value| (value - maximum).exp())
				.sum::<f32>()
				.ln();
		for column in 0..columns {
			output[row * columns + column] = values[column] - log_normalizer;
		}
	}
	output
}

fn host_log_softmax_mse(input: &[f32], rows: usize, columns: usize, target: &[f32]) -> f32 {
	host_log_softmax(input, rows, columns)
		.iter()
		.zip(target)
		.map(|(actual, target)| (actual - target).powi(2))
		.sum::<f32>()
		/ input.len() as f32
}

fn host_loss(
	input: &[f32],
	rows: usize,
	columns: usize,
	weight: &[f32],
	bias: &[f32],
	targets: &[u32],
) -> f32 {
	let probabilities = host_softmax(input, rows, columns);
	let classes = bias.len();
	let mut loss = 0.0_f64;
	for row in 0..rows {
		let mut logits = vec![0.0_f32; classes];
		for class in 0..classes {
			logits[class] = bias[class];
			for column in 0..columns {
				logits[class] += probabilities[row * columns + column] * weight[class * columns + column];
			}
		}
		let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
		let denominator = logits
			.iter()
			.map(|value| f64::from(*value - maximum).exp())
			.sum::<f64>();
		loss += f64::from(maximum) + denominator.ln() - f64::from(logits[targets[row] as usize]);
	}
	(loss / rows as f64) as f32
}

test_vk!(softmax_module_and_adjoint_match_donor_contract, engine, {
	let input_values = [
		0.2_f32, -0.7, 1.1, 0.8, 0.3, -0.4, -1.2, 0.5, 0.9, 1.4, -0.2, 0.1,
	];
	let weight_values = [0.4_f32, -0.3, 0.7, -0.6, 0.9, 0.2];
	let bias_values = [0.1_f32, -0.2];
	let targets = [0_u32, 1, 0, 1];
	let embedding =
		oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [4, 3], &input_values)?)?;
	let indices = oa::Matrix::from_slice(&engine, [4], &[0_u32, 1, 2, 3])?;
	let target_matrix = oa::Matrix::from_slice(&engine, [4], &targets)?;
	let linear = oa::ml::nn::Linear::from_matrices(
		oa::Matrix::from_f32(&engine, [2, 3], &weight_values)?,
		oa::Matrix::from_f32(&engine, [2], &bias_values)?,
	)?;
	let softmax = oa::ml::nn::Softmax::new(-1);
	assert_eq!(softmax.dim(), -1);
	assert!(softmax.parameters().is_empty());

	let tape = oa::ml::GradientTape::new();
	let input = embedding.forward(&indices)?;
	let probabilities = softmax.forward(&input)?;
	let loss = oa::ml::loss::cross_entropy(&linear.forward(&probabilities)?, &target_matrix)?;
	tape.backward(&loss)?;
	let actual = embedding
		.weight()
		.gradient()
		.expect("Softmax input adjoint is missing")
		.read_f32()?;

	const DELTA: f32 = 1.0e-3;
	let expected = (0..input_values.len())
		.map(|index| {
			let mut below = input_values;
			let mut above = input_values;
			below[index] -= DELTA;
			above[index] += DELTA;
			(host_loss(&above, 4, 3, &weight_values, &bias_values, &targets)
				- host_loss(&below, 4, 3, &weight_values, &bias_values, &targets))
				/ (2.0 * DELTA)
		})
		.collect::<Vec<_>>();
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= 4.0e-4,
			"gradient {index}: expected {expected}, found {actual}"
		);
	}
	Ok(())
});

test_vk!(
	log_softmax_module_and_adjoint_match_donor_contract,
	engine,
	{
		let input_values = [
			0.2_f32, -0.7, 1.1, 0.8, 0.3, -0.4, -1.2, 0.5, 0.9, 1.4, -0.2, 0.1,
		];
		let target_values = [
			-0.5_f32, -1.5, -0.9, -1.0, -0.8, -1.7, -2.1, -1.0, -0.6, -0.4, -2.0, -1.3,
		];
		let embedding =
			oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [4, 3], &input_values)?)?;
		let indices = oa::Matrix::from_slice(&engine, [4], &[0_u32, 1, 2, 3])?;
		let target = oa::Matrix::from_f32(&engine, [4, 3], &target_values)?;
		let log_softmax = oa::ml::nn::LogSoftmax::new(-1);
		assert_eq!(log_softmax.dim(), -1);
		assert!(log_softmax.parameters().is_empty());

		let tape = oa::ml::GradientTape::new();
		let output = log_softmax.forward(&embedding.forward(&indices)?)?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		let actual = embedding
			.weight()
			.gradient()
			.expect("LogSoftmax input adjoint is missing")
			.read_f32()?;

		const DELTA: f32 = 1.0e-3;
		let expected = (0..input_values.len())
			.map(|index| {
				let mut below = input_values;
				let mut above = input_values;
				below[index] -= DELTA;
				above[index] += DELTA;
				(host_log_softmax_mse(&above, 4, 3, &target_values)
					- host_log_softmax_mse(&below, 4, 3, &target_values))
					/ (2.0 * DELTA)
			})
			.collect::<Vec<_>>();
		for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
			assert!(
				(actual - expected).abs() <= 5.0e-4,
				"gradient {index}: expected {expected}, found {actual}"
			);
		}
		Ok(())
	}
);
