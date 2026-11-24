struct HostLstmCase<'a> {
	input: &'a [f32],
	batch: usize,
	sequence: usize,
	input_size: usize,
	hidden_size: usize,
	weight_ih: &'a [f32],
	weight_hh: &'a [f32],
	bias_ih: &'a [f32],
	bias_hh: &'a [f32],
}

fn sigmoid(value: f32) -> f32 {
	1.0 / (1.0 + (-value).exp())
}

fn host_lstm(case: HostLstmCase<'_>) -> Vec<f32> {
	let HostLstmCase {
		input,
		batch,
		sequence,
		input_size,
		hidden_size,
		weight_ih,
		weight_hh,
		bias_ih,
		bias_hh,
	} = case;
	let gate_size = 4 * hidden_size;
	let mut output = vec![0.0; batch * sequence * hidden_size];
	for batch_index in 0..batch {
		let mut hidden = vec![0.0; hidden_size];
		let mut cell = vec![0.0; hidden_size];
		for time in 0..sequence {
			let input_base = (batch_index * sequence + time) * input_size;
			let mut total = vec![0.0; gate_size];
			for gate in 0..gate_size {
				total[gate] = bias_ih[gate] + bias_hh[gate];
				for feature in 0..input_size {
					total[gate] += input[input_base + feature] * weight_ih[gate * input_size + feature];
				}
				for feature in 0..hidden_size {
					total[gate] += hidden[feature] * weight_hh[gate * hidden_size + feature];
				}
			}
			let output_base = (batch_index * sequence + time) * hidden_size;
			for feature in 0..hidden_size {
				let input_gate = sigmoid(total[feature]);
				let forget_gate = sigmoid(total[hidden_size + feature]);
				let candidate = (total[2 * hidden_size + feature]).tanh();
				let output_gate = sigmoid(total[3 * hidden_size + feature]);
				cell[feature] = forget_gate * cell[feature] + input_gate * candidate;
				hidden[feature] = output_gate * cell[feature].tanh();
				output[output_base + feature] = hidden[feature];
			}
		}
	}
	output
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= tolerance,
			"value {index} differs: actual={actual}, expected={expected}, tolerance={tolerance}"
		);
	}
}

test_vk!(lstm_forward_matches_the_host_reference, engine, {
	let input = oa::Matrix::new_from_f32(
		[2, 3, 2],
		&[
			0.4, -0.2, //
			0.1, 0.7, //
			-0.5, 0.3, //
			0.9, -0.4, //
			-0.1, 0.2, //
			0.6, -0.8,
		],
	)?;
	// Gate order is input, forget, candidate, output. Four blocks of 3 rows.
	let weight_ih = [
		0.2, -0.1, 0.1, 0.3, -0.3, 0.0, // input gate
		0.4, 0.1, -0.2, 0.2, 0.0, -0.1, // forget gate
		-0.1, 0.5, 0.2, -0.4, 0.3, 0.1, // candidate
		0.3, -0.2, -0.1, 0.1, 0.2, 0.4, // output gate
	];
	let weight_hh = [
		0.1, 0.0, -0.2, 0.2, 0.1, 0.1, -0.1, 0.3, 0.0, // input gate
		0.0, 0.2, 0.1, -0.1, 0.0, 0.2, 0.2, -0.2, 0.1, // forget gate
		0.3, -0.1, 0.0, 0.1, 0.2, -0.3, 0.0, 0.1, 0.2, // candidate
		-0.2, 0.1, 0.1, 0.0, -0.1, 0.2, 0.1, 0.0, -0.3, // output gate
	];
	let bias_ih = [
		0.05, -0.02, 0.01, //
		0.10, 0.00, -0.03, //
		-0.04, 0.06, 0.02, //
		0.03, -0.01, 0.07,
	];
	let bias_hh = [
		-0.01, 0.04, -0.02, //
		0.02, -0.05, 0.00, //
		0.01, 0.03, -0.06, //
		-0.02, 0.01, 0.04,
	];
	let layer = oa::ml::nn::Lstm::new_from_matrices(
		oa::Matrix::new_from_f32([12, 2], &weight_ih)?,
		oa::Matrix::new_from_f32([12, 3], &weight_hh)?,
		oa::Matrix::new_from_f32([12], &bias_ih)?,
		oa::Matrix::new_from_f32([12], &bias_hh)?,
	)?;
	let output = layer.forward(&input)?;
	assert_eq!(output.shape(), &[2, 3, 3]);
	let expected = host_lstm(HostLstmCase {
		input: &[
			0.4, -0.2, 0.1, 0.7, -0.5, 0.3, //
			0.9, -0.4, -0.1, 0.2, 0.6, -0.8,
		],
		batch: 2,
		sequence: 3,
		input_size: 2,
		hidden_size: 3,
		weight_ih: &weight_ih,
		weight_hh: &weight_hh,
		bias_ih: &bias_ih,
		bias_hh: &bias_hh,
	});
	assert_close(&output.read_f32()?, &expected, 1.0e-5);
	Ok(())
});

test_vk!(lstm_backward_produces_all_parameter_gradients, engine, {
	let layer = oa::ml::nn::Lstm::new_with_seed(2, 3, 1, true, 0x4c53_544d_0001)?;
	let input = oa::Matrix::new_from_f32([2, 4, 2], &[0.3; 16])?;
	let tape = oa::ml::GradientTape::new();
	let output = layer.forward(&input)?;
	let loss = output.sum(-1)?.reshape(Vec::<usize>::new())?;
	tape.backward(&loss)?;
	let parameters = layer.all_parameters()?;
	assert_eq!(parameters.len(), 4);
	for parameter in &parameters {
		let gradient = parameter
			.gradient()
			.expect("backward produced a parameter gradient");
		assert!(
			gradient.read_f32()?.iter().any(|value| *value != 0.0),
			"parameter gradient is entirely zero"
		);
	}
	Ok(())
});

test_vk!(lstm_cell_step_matches_the_scan_first_state, engine, {
	let weight_ih = oa::Matrix::new_from_f32([8, 2], &[0.1; 16])?;
	let weight_hh = oa::Matrix::new_from_f32([8, 2], &[0.05; 16])?;
	let bias_ih = oa::Matrix::new_from_f32([8], &[0.02; 8])?;
	let bias_hh = oa::Matrix::new_from_f32([8], &[-0.01; 8])?;
	let cell_layer = oa::ml::nn::LstmCell::new_from_matrices(
		weight_ih.clone(),
		weight_hh.clone(),
		bias_ih.clone(),
		bias_hh.clone(),
	)?;
	let scan_layer = oa::ml::nn::Lstm::new_from_matrices(weight_ih, weight_hh, bias_ih, bias_hh)?;

	let input_step = oa::Matrix::new_from_f32([2, 2], &[0.4, -0.2, 0.7, 0.1])?;
	let input_seq = input_step.reshape([2, 1, 2])?;
	let (hidden, cell) = cell_layer.zero_state(2)?;
	let (hidden_next, _) = cell_layer.step(&input_step, &hidden, &cell)?;
	let scan_output = scan_layer.forward(&input_seq)?.reshape([2, 2])?;
	assert_close(&hidden_next.read_f32()?, &scan_output.read_f32()?, 1.0e-5);
	Ok(())
});
