fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, operation: &str) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		let error = (actual - expected).abs();
		assert!(
			error <= tolerance,
			"{operation} element {index}: expected {expected}, found {actual}, error {error}"
		);
	}
}

fn cross_entropy(logits: &[f32], targets: &[u32], classes: usize) -> f32 {
	let mut loss = 0.0_f64;
	for (row, target) in targets.iter().copied().enumerate() {
		let values = &logits[row * classes..(row + 1) * classes];
		let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
		let denominator = values
			.iter()
			.map(|value| f64::from(*value - maximum).exp())
			.sum::<f64>();
		loss += f64::from(maximum) + denominator.ln() - f64::from(values[target as usize]);
	}
	(loss / targets.len() as f64) as f32
}

fn activation_loss(
	values: &[f32],
	activation: fn(f32) -> f32,
	weight: &[f32; 2],
	bias: &[f32; 2],
	targets: &[u32],
) -> f32 {
	let mut logits = Vec::with_capacity(values.len() * 2);
	for value in values {
		let activated = activation(*value);
		logits.push(activated * weight[0] + bias[0]);
		logits.push(activated * weight[1] + bias[1]);
	}
	cross_entropy(&logits, targets, 2)
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

struct ActivationCase {
	name: &'static str,
	host: fn(f32) -> f32,
	device: fn(&oa::Matrix) -> oa::Result<oa::Matrix>,
}

fn host_silu(value: f32) -> f32 {
	value / (1.0 + (-value).exp())
}

fn host_relu(value: f32) -> f32 {
	value.max(0.0)
}

fn host_tanh(value: f32) -> f32 {
	value.tanh()
}

fn host_sigmoid(value: f32) -> f32 {
	1.0 / (1.0 + (-value).exp())
}

fn host_leaky_relu(value: f32) -> f32 {
	(value * 0.1).max(value)
}

fn device_leaky_relu(input: &oa::Matrix) -> oa::Result<oa::Matrix> {
	oa::ml::matrix::leaky_relu(input, 0.1)
}

fn host_elu(value: f32) -> f32 {
	if value > 0.0 {
		value
	} else {
		value.exp() - 1.0
	}
}

fn device_elu(input: &oa::Matrix) -> oa::Result<oa::Matrix> {
	oa::ml::matrix::elu(input, 1.0)
}

fn host_mish(value: f32) -> f32 {
	value * value.exp().ln_1p().tanh()
}

fn host_softplus(value: f32) -> f32 {
	value.max(0.0) + (-value.abs()).exp().ln_1p()
}

const CASES: [ActivationCase; 8] = [
	ActivationCase {
		name: "silu",
		host: host_silu,
		device: oa::ml::matrix::silu,
	},
	ActivationCase {
		name: "relu",
		host: host_relu,
		device: oa::ml::matrix::relu,
	},
	ActivationCase {
		name: "tanh",
		host: host_tanh,
		device: oa::ml::matrix::tanh,
	},
	ActivationCase {
		name: "sigmoid",
		host: host_sigmoid,
		device: oa::ml::matrix::sigmoid,
	},
	ActivationCase {
		name: "leaky_relu",
		host: host_leaky_relu,
		device: device_leaky_relu,
	},
	ActivationCase {
		name: "elu",
		host: host_elu,
		device: device_elu,
	},
	ActivationCase {
		name: "mish",
		host: host_mish,
		device: oa::ml::matrix::mish,
	},
	ActivationCase {
		name: "softplus",
		host: host_softplus,
		device: oa::ml::matrix::softplus,
	},
];

test_vk!(
	donor_activation_family_matches_forward_and_reverse_oracles,
	engine,
	{
		let forward_values = [-100.0_f32, -3.0, -0.25, 0.0, 0.5, 4.0, 100.0];
		let forward_input = oa::Matrix::from_f32(&engine, [forward_values.len()], &forward_values)?;
		for case in &CASES {
			let expected = forward_values.map(case.host);
			let actual = (case.device)(&forward_input)?.read_f32()?;
			assert_close(&actual, &expected, 2.0e-5, case.name);
		}

		let gradient_values = [-2.0_f32, -0.7, -0.1, 0.2, 0.8, 1.7];
		let indices = oa::Matrix::from_slice(&engine, [6], &[0_u32, 1, 2, 3, 4, 5])?;
		let targets = [0_u32, 1, 0, 1, 0, 1];
		let target_matrix = oa::Matrix::from_slice(&engine, [6], &targets)?;
		let weight = [0.4_f32, -0.6];
		let bias = [0.1_f32, -0.2];
		for case in &CASES {
			let expected = numerical_gradient(&gradient_values, |values| {
				activation_loss(values, case.host, &weight, &bias, &targets)
			});
			let embedding = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
				&engine,
				[6, 1],
				&gradient_values,
			)?)?;
			let linear = oa::ml::nn::Linear::from_matrices(
				oa::Matrix::from_f32(&engine, [2, 1], &weight)?,
				oa::Matrix::from_f32(&engine, [2], &bias)?,
			)?;
			let tape = oa::ml::GradientTape::new();
			let activated = (case.device)(&embedding.forward(&indices)?)?;
			let loss = oa::ml::loss::cross_entropy(&linear.forward(&activated)?, &target_matrix)?;
			tape.backward(&loss)?;
			let gradient = embedding.weight().gradient();
			let Some(gradient) = gradient else {
				panic!("{} input gradient is missing", case.name);
			};
			let actual = gradient.read_f32()?;
			assert_close(&actual, &expected, 7.0e-4, case.name);
		}
		Ok(())
	}
);

test_vk!(
	donor_activation_modules_are_parameterless_operation_adapters,
	engine,
	{
		use oa::ml::Module as _;

		let input = oa::Matrix::from_f32(&engine, [3], &[-1.0, 0.0, 2.0])?;
		let relu = oa::ml::nn::Relu::new();
		let gelu = oa::ml::nn::Gelu::new();
		let silu = oa::ml::nn::Silu::new();
		assert_eq!(relu.forward(&input)?.read_f32()?, [0.0, 0.0, 2.0]);
		assert_close(
			&gelu.forward(&input)?.read_f32()?,
			&[-1.0_f32, 0.0, 2.0].map(|value| {
				0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value.powi(3))).tanh())
			}),
			1.0e-6,
			"Gelu module",
		);
		assert_close(
			&silu.forward(&input)?.read_f32()?,
			&[-1.0_f32, 0.0, 2.0].map(host_silu),
			1.0e-6,
			"Silu module",
		);
		assert!(relu.parameters().is_empty());
		assert!(gelu.parameters().is_empty());
		assert!(silu.parameters().is_empty());
		Ok(())
	}
);
