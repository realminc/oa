struct HostGruCase<'a> {
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

fn host_gru(case: HostGruCase<'_>) -> Vec<f32> {
	let HostGruCase {
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
	let gate_size = 3 * hidden_size;
	let mut output = vec![0.0; batch * sequence * hidden_size];
	for batch_index in 0..batch {
		let mut hidden = vec![0.0; hidden_size];
		for time in 0..sequence {
			let input_base = (batch_index * sequence + time) * input_size;
			let mut gates_i = vec![0.0; gate_size];
			let mut gates_h = vec![0.0; gate_size];
			for gate in 0..gate_size {
				gates_i[gate] = bias_ih[gate];
				gates_h[gate] = bias_hh[gate];
				for feature in 0..input_size {
					gates_i[gate] += input[input_base + feature] * weight_ih[gate * input_size + feature];
				}
				for feature in 0..hidden_size {
					gates_h[gate] += hidden[feature] * weight_hh[gate * hidden_size + feature];
				}
			}
			let output_base = (batch_index * sequence + time) * hidden_size;
			for feature in 0..hidden_size {
				let reset = 1.0 / (1.0 + (-(gates_i[feature] + gates_h[feature])).exp());
				let update =
					1.0 / (1.0 + (-(gates_i[hidden_size + feature] + gates_h[hidden_size + feature])).exp());
				let candidate =
					(gates_i[2 * hidden_size + feature] + reset * gates_h[2 * hidden_size + feature]).tanh();
				hidden[feature] = (1.0 - update) * candidate + update * hidden[feature];
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
			"value {index}: actual {actual}, expected {expected}, tolerance {tolerance}"
		);
	}
}

struct HostGruCellCase<'a> {
	input: &'a [f32],
	hidden: &'a [f32],
	input_size: usize,
	hidden_size: usize,
	weight_ih: &'a [f32],
	weight_hh: &'a [f32],
	bias_ih: &'a [f32],
	bias_hh: &'a [f32],
}

fn host_gru_cell(case: HostGruCellCase<'_>) -> Vec<f32> {
	let gate_size = 3 * case.hidden_size;
	let mut gates_i = vec![0.0; gate_size];
	let mut gates_h = vec![0.0; gate_size];
	for gate in 0..gate_size {
		gates_i[gate] = case.bias_ih[gate];
		gates_h[gate] = case.bias_hh[gate];
		for feature in 0..case.input_size {
			gates_i[gate] += case.input[feature] * case.weight_ih[gate * case.input_size + feature];
		}
		for feature in 0..case.hidden_size {
			gates_h[gate] += case.hidden[feature] * case.weight_hh[gate * case.hidden_size + feature];
		}
	}
	(0..case.hidden_size)
		.map(|feature| {
			let reset = 1.0 / (1.0 + (-(gates_i[feature] + gates_h[feature])).exp());
			let update = 1.0
				/ (1.0
					+ (-(gates_i[case.hidden_size + feature] + gates_h[case.hidden_size + feature])).exp());
			let candidate = (gates_i[2 * case.hidden_size + feature]
				+ reset * gates_h[2 * case.hidden_size + feature])
				.tanh();
			(1.0 - update) * candidate + update * case.hidden[feature]
		})
		.collect()
}

test_vk!(gru_matches_independent_host_scan, engine, {
	let input_values = [0.2, -0.4, 0.7, 0.1, -0.3, 0.5, 0.8, -0.2];
	let weight_ih = [
		0.1, -0.2, 0.3, 0.4, -0.5, 0.2, 0.6, -0.1, 0.2, 0.5, -0.4, 0.3,
	];
	let weight_hh = [
		0.2, -0.3, 0.4, 0.1, -0.2, 0.5, 0.3, 0.2, -0.1, 0.6, 0.5, -0.4,
	];
	let bias_ih = [0.01, -0.02, 0.03, 0.04, -0.05, 0.06];
	let bias_hh = [-0.02, 0.01, 0.05, -0.03, 0.02, -0.04];
	let expected = host_gru(HostGruCase {
		input: &input_values,
		batch: 2,
		sequence: 2,
		input_size: 2,
		hidden_size: 2,
		weight_ih: &weight_ih,
		weight_hh: &weight_hh,
		bias_ih: &bias_ih,
		bias_hh: &bias_hh,
	});
	let gru = oa::ml::nn::Gru::from_matrices(
		oa::Matrix::from_f32(&engine, [6, 2], &weight_ih)?,
		oa::Matrix::from_f32(&engine, [6, 2], &weight_hh)?,
		oa::Matrix::from_f32(&engine, [6], &bias_ih)?,
		oa::Matrix::from_f32(&engine, [6], &bias_hh)?,
	)?;
	let input = oa::Matrix::from_f32(&engine, [2, 2, 2], &input_values)?;
	let output = gru.forward(&input)?;
	assert_eq!(output.shape(), [2, 2, 2]);
	assert_close(&output.read_f32()?, &expected, 3.0e-5);
	Ok(())
});

test_vk!(gru_cell_matches_nonzero_hidden_state_oracle, engine, {
	let input = [0.2, -0.4];
	let hidden = [0.3, -0.25];
	let weight_ih = [
		0.1, -0.2, 0.3, 0.4, -0.5, 0.2, 0.6, -0.1, 0.2, 0.5, -0.4, 0.3,
	];
	let weight_hh = [
		0.2, -0.3, 0.4, 0.1, -0.2, 0.5, 0.3, 0.2, -0.1, 0.6, 0.5, -0.4,
	];
	let bias_ih = [0.01, -0.02, 0.03, 0.04, -0.05, 0.06];
	let bias_hh = [-0.02, 0.01, 0.05, -0.03, 0.02, -0.04];
	let expected = host_gru_cell(HostGruCellCase {
		input: &input,
		hidden: &hidden,
		input_size: 2,
		hidden_size: 2,
		weight_ih: &weight_ih,
		weight_hh: &weight_hh,
		bias_ih: &bias_ih,
		bias_hh: &bias_hh,
	});
	let cell = oa::ml::nn::GruCell::from_matrices(
		oa::Matrix::from_f32(&engine, [6, 2], &weight_ih)?,
		oa::Matrix::from_f32(&engine, [6, 2], &weight_hh)?,
		oa::Matrix::from_f32(&engine, [6], &bias_ih)?,
		oa::Matrix::from_f32(&engine, [6], &bias_hh)?,
	)?;
	assert_eq!(cell.input_size(), 2);
	assert_eq!(cell.hidden_size(), 2);
	assert!(cell.has_bias());
	assert_eq!(cell.all_parameters()?.len(), 4);
	let output = cell.step(
		&oa::Matrix::from_f32(&engine, [1, 2], &input)?,
		&oa::Matrix::from_f32(&engine, [1, 2], &hidden)?,
	)?;
	assert_close(&output.read_f32()?, &expected, 3.0e-5);
	assert_eq!(cell.zero_state(3)?.read_f32()?, vec![0.0; 6]);
	Ok(())
});

fn gpu_loss_and_gradients(
	engine: &oa::Engine,
	input: &[f32],
	weight_ih: &[f32],
	weight_hh: &[f32],
	bias_ih: &[f32],
	bias_hh: &[f32],
	backward: bool,
) -> oa::Result<(f32, Vec<Vec<f32>>)> {
	let gru = oa::ml::nn::Gru::from_matrices(
		oa::Matrix::from_f32(engine, [3, 1], weight_ih)?,
		oa::Matrix::from_f32(engine, [3, 1], weight_hh)?,
		oa::Matrix::from_f32(engine, [3], bias_ih)?,
		oa::Matrix::from_f32(engine, [3], bias_hh)?,
	)?;
	let tape = oa::ml::GradientTape::new();
	let output = gru.forward(&oa::Matrix::from_f32(engine, [1, 2, 1], input)?)?;
	let target = oa::Matrix::from_f32(engine, [1, 2, 1], &[0.0, 0.0])?;
	let loss = oa::ml::loss::mse(&output, &target)?;
	if backward {
		tape.backward(&loss)?;
	}
	let loss_value = loss.read_f32()?[0];
	let gradients = if backward {
		gru
			.layer_parameters(0)
			.expect("missing GRU layer")
			.into_iter()
			.map(|parameter| {
				parameter
					.gradient()
					.expect("missing GRU parameter gradient")
					.read_f32()
			})
			.collect::<oa::Result<Vec<_>>>()?
	} else {
		Vec::new()
	};
	Ok((loss_value, gradients))
}

fn gpu_cell_loss_and_gradients(
	engine: &oa::Engine,
	input: f32,
	hidden: f32,
	values: &[Vec<f32>; 4],
	backward: bool,
) -> oa::Result<(f32, Vec<Vec<f32>>)> {
	let cell = oa::ml::nn::GruCell::from_matrices(
		oa::Matrix::from_f32(engine, [3, 1], &values[0])?,
		oa::Matrix::from_f32(engine, [3, 1], &values[1])?,
		oa::Matrix::from_f32(engine, [3], &values[2])?,
		oa::Matrix::from_f32(engine, [3], &values[3])?,
	)?;
	let tape = oa::ml::GradientTape::new();
	let output = cell.step(
		&oa::Matrix::from_f32(engine, [1, 1], &[input])?,
		&oa::Matrix::from_f32(engine, [1, 1], &[hidden])?,
	)?;
	let target = oa::Matrix::from_f32(engine, [1, 1], &[0.0])?;
	let loss = oa::ml::loss::mse(&output, &target)?;
	if backward {
		tape.backward(&loss)?;
	}
	let loss_value = loss.read_f32()?[0];
	let gradients = if backward {
		cell
			.all_parameters()?
			.into_iter()
			.map(|parameter| {
				parameter
					.gradient()
					.expect("missing GRU cell parameter gradient")
					.read_f32()
			})
			.collect::<oa::Result<Vec<_>>>()?
	} else {
		Vec::new()
	};
	Ok((loss_value, gradients))
}

test_vk!(gru_parameter_gradients_match_finite_differences, engine, {
	let input = [0.25, -0.4];
	let mut values = [
		vec![0.2, -0.1, 0.35],
		vec![0.15, -0.25, 0.3],
		vec![0.01, -0.02, 0.04],
		vec![-0.03, 0.02, 0.05],
	];
	let (_, gradients) = gpu_loss_and_gradients(
		&engine, &input, &values[0], &values[1], &values[2], &values[3], true,
	)?;
	let epsilon = 1.0e-3_f32;
	for parameter_index in 0..values.len() {
		for value_index in 0..values[parameter_index].len() {
			values[parameter_index][value_index] += epsilon;
			let plus = gpu_loss_and_gradients(
				&engine, &input, &values[0], &values[1], &values[2], &values[3], false,
			)?
			.0;
			values[parameter_index][value_index] -= 2.0 * epsilon;
			let minus = gpu_loss_and_gradients(
				&engine, &input, &values[0], &values[1], &values[2], &values[3], false,
			)?
			.0;
			values[parameter_index][value_index] += epsilon;
			let numerical = (plus - minus) / (2.0 * epsilon);
			let actual = gradients[parameter_index][value_index];
			assert!(
				(actual - numerical).abs() <= 1.2e-3,
				"parameter {parameter_index} value {value_index}: actual {actual}, numerical {numerical}"
			);
		}
	}
	Ok(())
});

test_vk!(
	gru_cell_parameter_gradients_match_finite_differences,
	engine,
	{
		let mut values = [
			vec![0.2, -0.1, 0.35],
			vec![0.15, -0.25, 0.3],
			vec![0.01, -0.02, 0.04],
			vec![-0.03, 0.02, 0.05],
		];
		let (_, gradients) = gpu_cell_loss_and_gradients(&engine, 0.25, -0.4, &values, true)?;
		let epsilon = 1.0e-3_f32;
		for parameter_index in 0..values.len() {
			for value_index in 0..values[parameter_index].len() {
				values[parameter_index][value_index] += epsilon;
				let plus = gpu_cell_loss_and_gradients(&engine, 0.25, -0.4, &values, false)?.0;
				values[parameter_index][value_index] -= 2.0 * epsilon;
				let minus = gpu_cell_loss_and_gradients(&engine, 0.25, -0.4, &values, false)?.0;
				values[parameter_index][value_index] += epsilon;
				let numerical = (plus - minus) / (2.0 * epsilon);
				let actual = gradients[parameter_index][value_index];
				assert!(
					(actual - numerical).abs() <= 1.2e-3,
					"parameter {parameter_index} value {value_index}: actual {actual}, numerical {numerical}"
				);
			}
		}
		Ok(())
	}
);

test_vk!(
	stacked_and_bias_free_gru_preserve_module_contract,
	engine,
	{
		let biased = oa::ml::nn::Gru::with_seed(&engine, 2, 3, 2, true, 0x4752_5501)?;
		assert_eq!(biased.input_size(), 2);
		assert_eq!(biased.hidden_size(), 3);
		assert_eq!(biased.num_layers(), 2);
		assert!(biased.has_bias());
		assert_eq!(biased.all_parameters()?.len(), 8);
		let input = oa::Matrix::from_f32(&engine, [2, 4, 2], &[0.1; 16])?;
		let before_freeze = biased.forward(&input)?.read_f32()?;
		let layer_parameters = biased.layer_parameters(0).expect("missing layer");
		layer_parameters[2].set_requires_grad(false);
		layer_parameters[3].set_requires_grad(false);
		let after_freeze = biased.forward(&input)?.read_f32()?;
		assert_close(&after_freeze, &before_freeze, 1.0e-7);

		let bias_free = oa::ml::nn::Gru::with_seed(&engine, 2, 3, 2, false, 0x4752_5502)?;
		assert!(!bias_free.has_bias());
		assert_eq!(bias_free.all_parameters()?.len(), 4);
		assert_eq!(
			bias_free.layer_parameters(0).expect("missing layer").len(),
			2
		);
		assert_eq!(bias_free.forward(&input)?.shape(), [2, 4, 3]);
		assert_eq!(
			bias_free
				.forward(&oa::Matrix::from_f32(&engine, [2, 4, 1], &[0.0; 8])?)
				.err()
				.expect("wrong GRU input width was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		Ok(())
	}
);
