fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		let error = (actual - expected).abs();
		assert!(
			error <= tolerance,
			"element {index}: expected {expected}, found {actual}, error {error}"
		);
	}
}

fn cpu_attention(
	q: &[f32],
	k: &[f32],
	v: &[f32],
	sequence: usize,
	width: usize,
	heads: usize,
) -> Vec<f32> {
	let batches = q.len() / (sequence * width);
	let head_width = width / heads;
	let scale = 1.0_f64 / (head_width as f64).sqrt();
	let mut output = vec![0.0; q.len()];
	for batch in 0..batches {
		for head in 0..heads {
			for query in 0..sequence {
				let mut scores = Vec::with_capacity(query + 1);
				for key in 0..=query {
					let mut score = 0.0_f64;
					for feature in 0..head_width {
						let q_index = (batch * sequence + query) * width + head * head_width + feature;
						let k_index = (batch * sequence + key) * width + head * head_width + feature;
						score += f64::from(q[q_index]) * f64::from(k[k_index]);
					}
					scores.push(score * scale);
				}
				let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
				let denominator = scores
					.iter()
					.map(|score| (score - maximum).exp())
					.sum::<f64>();
				for feature in 0..head_width {
					let mut value_sum = 0.0_f64;
					for (key, score) in scores.iter().copied().enumerate() {
						let probability = (score - maximum).exp() / denominator;
						let index = (batch * sequence + key) * width + head * head_width + feature;
						value_sum += probability * f64::from(v[index]);
					}
					let index = (batch * sequence + query) * width + head * head_width + feature;
					output[index] = value_sum as f32;
				}
			}
		}
	}
	output
}

fn cpu_cross_entropy(logits: &[f32], targets: &[u32], classes: usize) -> f32 {
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

fn cpu_linear(
	input: &[f32],
	weight: &[f32],
	bias: &[f32],
	width: usize,
	classes: usize,
) -> Vec<f32> {
	let mut output = vec![0.0; input.len() / width * classes];
	for row in 0..input.len() / width {
		for class in 0..classes {
			let mut value = bias[class];
			for feature in 0..width {
				value += input[row * width + feature] * weight[class * width + feature];
			}
			output[row * classes + class] = value;
		}
	}
	output
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
	causal_multi_head_attention_matches_host_and_finite_difference_oracles,
	engine,
	{
		const S: usize = 3;
		const D: usize = 4;
		const H: usize = 2;
		const C: usize = 2;
		let q = [
			0.2_f32, -0.1, 0.4, 0.3, -0.3, 0.5, 0.1, -0.2, 0.6, 0.2, -0.4, 0.7,
		];
		let k = [
			-0.2_f32, 0.3, 0.5, -0.1, 0.4, -0.6, 0.2, 0.3, 0.1, 0.7, -0.5, 0.2,
		];
		let v = [
			0.8_f32, -0.4, 0.2, 0.1, -0.3, 0.6, 0.7, -0.2, 0.5, 0.3, -0.1, 0.9,
		];
		let linear_weight = [0.3_f32, -0.2, 0.4, 0.1, -0.5, 0.6, 0.2, -0.3];
		let linear_bias = [0.02_f32, -0.04];
		let targets = [1_u32, 0, 1];
		let cpu_loss = |qv: &[f32], kv: &[f32], vv: &[f32]| {
			let context = cpu_attention(qv, kv, vv, S, D, H);
			cpu_cross_entropy(
				&cpu_linear(&context, &linear_weight, &linear_bias, D, C),
				&targets,
				C,
			)
		};
		let expected_q = numerical_gradient(&q, |values| cpu_loss(values, &k, &v));
		let expected_k = numerical_gradient(&k, |values| cpu_loss(&q, values, &v));
		let expected_v = numerical_gradient(&v, |values| cpu_loss(&q, &k, values));
		let indices = oa::Matrix::from_slice(&engine, [S], &[0_u32, 1, 2])?;
		let q_embedding =
			oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [S, D], &q)?)?;
		let k_embedding =
			oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [S, D], &k)?)?;
		let v_embedding =
			oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [S, D], &v)?)?;
		let linear = oa::ml::nn::Linear::from_matrices(
			oa::Matrix::from_f32(&engine, [C, D], &linear_weight)?,
			oa::Matrix::from_f32(&engine, [C], &linear_bias)?,
		)?;
		let target_matrix = oa::Matrix::from_slice(&engine, [S], &targets)?;
		let tape = oa::ml::GradientTape::new();
		let q_matrix = q_embedding.forward(&indices)?;
		let k_matrix = k_embedding.forward(&indices)?;
		let v_matrix = v_embedding.forward(&indices)?;
		let context =
			oa::ml::matrix::scaled_dot_product_attention_causal(&q_matrix, &k_matrix, &v_matrix, S, H)?;
		assert_close(
			&context.read_f32()?,
			&cpu_attention(&q, &k, &v, S, D, H),
			2.0e-5,
		);
		let loss = oa::ml::loss::cross_entropy(&linear.forward(&context)?, &target_matrix)?;
		tape.backward(&loss)?;
		assert_close(
			&q_embedding
				.weight()
				.gradient()
				.expect("missing Q gradient")
				.read_f32()?,
			&expected_q,
			8.0e-4,
		);
		assert_close(
			&k_embedding
				.weight()
				.gradient()
				.expect("missing K gradient")
				.read_f32()?,
			&expected_k,
			8.0e-4,
		);
		assert_close(
			&v_embedding
				.weight()
				.gradient()
				.expect("missing V gradient")
				.read_f32()?,
			&expected_v,
			8.0e-4,
		);
		Ok(())
	}
);

test_vk!(
	causal_attention_accepts_sequence_lengths_beyond_the_retired_monolithic_limit,
	engine,
	{
		let values = vec![0.0_f32; 1025];
		let query = oa::Matrix::from_f32(&engine, [1025, 1], &values)?;
		let key = oa::Matrix::from_f32(&engine, [1025, 1], &values)?;
		let value = oa::Matrix::from_f32(&engine, [1025, 1], &values)?;
		let output =
			oa::ml::matrix::scaled_dot_product_attention_causal(&query, &key, &value, 1025, 1)?;
		assert_eq!(output.shape(), [1025, 1]);
		assert!(output.read_f32()?.into_iter().all(|value| value == 0.0));
		Ok(())
	}
);

test_vk!(
	gelu_forward_and_backward_match_independent_oracles,
	engine,
	{
		let values = [-2.0_f32, -1.0, -0.1, 0.0, 0.5, 1.0, 2.0];
		let expected =
			values.map(|x| 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh()));
		let input = oa::Matrix::from_f32(&engine, [7], &values)?;
		assert_close(
			&oa::ml::matrix::gelu(&input)?.read_f32()?,
			&expected,
			1.0e-6,
		);
		let embedding =
			oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [7, 1], &values)?)?;
		let indices = oa::Matrix::from_slice(&engine, [7], &[0_u32, 1, 2, 3, 4, 5, 6])?;
		let linear_weight = [0.4_f32, -0.6];
		let linear_bias = [0.1_f32, -0.2];
		let targets = [0_u32, 1, 0, 1, 0, 1, 0];
		let host_loss = |input: &[f32]| {
			let activated = input
				.iter()
				.map(|x| 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh()))
				.collect::<Vec<_>>();
			cpu_cross_entropy(
				&cpu_linear(&activated, &linear_weight, &linear_bias, 1, 2),
				&targets,
				2,
			)
		};
		let expected_gradient = numerical_gradient(&values, host_loss);
		let linear = oa::ml::nn::Linear::from_matrices(
			oa::Matrix::from_f32(&engine, [2, 1], &linear_weight)?,
			oa::Matrix::from_f32(&engine, [2], &linear_bias)?,
		)?;
		let target_matrix = oa::Matrix::from_slice(&engine, [7], &targets)?;
		let tape = oa::ml::GradientTape::new();
		let activated = oa::ml::matrix::gelu(&embedding.forward(&indices)?)?;
		let loss = oa::ml::loss::cross_entropy(&linear.forward(&activated)?, &target_matrix)?;
		tape.backward(&loss)?;
		assert_close(
			&embedding
				.weight()
				.gradient()
				.expect("missing GELU input gradient")
				.read_f32()?,
			&expected_gradient,
			5.0e-4,
		);
		Ok(())
	}
);

test_vk!(
	swiglu_forward_and_adjoint_match_independent_oracles,
	engine,
	{
		let gate_values = [-2.0_f32, -0.5, 0.0, 0.25, 1.0, 2.0, 4.0];
		let up_values = [0.3_f32, -0.7, 1.2, 0.5, -0.4, 0.8, -1.1];
		let expected = std::array::from_fn::<_, 7, _>(|index| {
			let gate = gate_values[index];
			gate / (1.0 + (-gate).exp()) * up_values[index]
		});
		let gate = oa::Matrix::from_f32(&engine, [7], &gate_values)?;
		let up = oa::Matrix::from_f32(&engine, [7], &up_values)?;
		assert_close(
			&oa::ml::matrix::swiglu(&gate, &up)?.read_f32()?,
			&expected,
			1.0e-6,
		);
		let wrong_shape = oa::Matrix::from_f32(&engine, [1], &[1.0])?;
		assert_eq!(
			oa::ml::matrix::swiglu(&gate, &wrong_shape)
				.err()
				.map(|error| error.kind()),
			Some(oa::ErrorKind::InvalidArgument)
		);
		let wrong_dtype = oa::Matrix::from_slice(&engine, [7], &[1_u32; 7])?;
		assert_eq!(
			oa::ml::matrix::swiglu(&gate, &wrong_dtype)
				.err()
				.map(|error| error.kind()),
			Some(oa::ErrorKind::InvalidArgument)
		);

		let indices = oa::Matrix::from_slice(&engine, [7], &[0_u32, 1, 2, 3, 4, 5, 6])?;
		let gate_embedding =
			oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [7, 1], &gate_values)?)?;
		let up_embedding =
			oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [7, 1], &up_values)?)?;
		let linear_weight = [0.4_f32, -0.6];
		let linear_bias = [0.1_f32, -0.2];
		let targets = [0_u32, 1, 0, 1, 0, 1, 0];
		let host_loss = |gate: &[f32], up: &[f32]| {
			let activated = gate
				.iter()
				.zip(up)
				.map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
				.collect::<Vec<_>>();
			cpu_cross_entropy(
				&cpu_linear(&activated, &linear_weight, &linear_bias, 1, 2),
				&targets,
				2,
			)
		};
		let expected_gate = numerical_gradient(&gate_values, |values| host_loss(values, &up_values));
		let expected_up = numerical_gradient(&up_values, |values| host_loss(&gate_values, values));
		let linear = oa::ml::nn::Linear::from_matrices(
			oa::Matrix::from_f32(&engine, [2, 1], &linear_weight)?,
			oa::Matrix::from_f32(&engine, [2], &linear_bias)?,
		)?;
		let target_matrix = oa::Matrix::from_slice(&engine, [7], &targets)?;
		let tape = oa::ml::GradientTape::new();
		let activated = oa::ml::matrix::swiglu(
			&gate_embedding.forward(&indices)?,
			&up_embedding.forward(&indices)?,
		)?;
		let loss = oa::ml::loss::cross_entropy(&linear.forward(&activated)?, &target_matrix)?;
		tape.backward(&loss)?;
		assert_close(
			&gate_embedding
				.weight()
				.gradient()
				.expect("missing SwiGLU gate gradient")
				.read_f32()?,
			&expected_gate,
			5.0e-4,
		);
		assert_close(
			&up_embedding
				.weight()
				.gradient()
				.expect("missing SwiGLU up gradient")
				.read_f32()?,
			&expected_up,
			5.0e-4,
		);
		Ok(())
	}
);
