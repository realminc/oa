struct HostRnnCellCase<'a> {
	input: &'a [f32],
	hidden: &'a [f32],
	input_size: usize,
	hidden_size: usize,
	weight_ih: &'a [f32],
	weight_hh: &'a [f32],
	bias_ih: &'a [f32],
	bias_hh: &'a [f32],
}

fn host_rnn_cell(case: HostRnnCellCase<'_>) -> Vec<f32> {
	(0..case.hidden_size)
		.map(|output| {
			let mut value = case.bias_ih[output] + case.bias_hh[output];
			for input in 0..case.input_size {
				value += case.input[input] * case.weight_ih[output * case.input_size + input];
			}
			for hidden in 0..case.hidden_size {
				value += case.hidden[hidden] * case.weight_hh[output * case.hidden_size + hidden];
			}
			value.tanh()
		})
		.collect()
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

test_vk!(rnn_cell_matches_nonzero_hidden_state_oracle, engine, {
	let input = [0.2, -0.4];
	let hidden = [0.3, -0.25];
	let weight_ih = [0.1, -0.2, 0.3, 0.4];
	let weight_hh = [0.2, -0.3, 0.4, 0.1];
	let bias_ih = [0.01, -0.02];
	let bias_hh = [-0.03, 0.04];
	let expected = host_rnn_cell(HostRnnCellCase {
		input: &input,
		hidden: &hidden,
		input_size: 2,
		hidden_size: 2,
		weight_ih: &weight_ih,
		weight_hh: &weight_hh,
		bias_ih: &bias_ih,
		bias_hh: &bias_hh,
	});
	let cell = oa::ml::nn::RnnCell::from_matrices(
		oa::Matrix::from_f32(&engine, [2, 2], &weight_ih)?,
		oa::Matrix::from_f32(&engine, [2, 2], &weight_hh)?,
		oa::Matrix::from_f32(&engine, [2], &bias_ih)?,
		oa::Matrix::from_f32(&engine, [2], &bias_hh)?,
	)?;
	assert_eq!(cell.input_size(), 2);
	assert_eq!(cell.hidden_size(), 2);
	assert!(cell.has_bias());
	assert_eq!(cell.all_parameters()?.len(), 4);
	let output = cell.step(
		&oa::Matrix::from_f32(&engine, [1, 2], &input)?,
		&oa::Matrix::from_f32(&engine, [1, 2], &hidden)?,
	)?;
	assert_close(&output.read_f32()?, &expected, 2.0e-5);
	assert_eq!(cell.zero_state(3)?.read_f32()?, vec![0.0; 6]);
	Ok(())
});

fn gpu_cell_loss_and_gradients(
	engine: &oa::Engine,
	input: f32,
	hidden: f32,
	values: &[f32; 4],
	backward: bool,
) -> oa::Result<(f32, Vec<f32>)> {
	let cell = oa::ml::nn::RnnCell::from_matrices(
		oa::Matrix::from_f32(engine, [1, 1], &values[0..1])?,
		oa::Matrix::from_f32(engine, [1, 1], &values[1..2])?,
		oa::Matrix::from_f32(engine, [1], &values[2..3])?,
		oa::Matrix::from_f32(engine, [1], &values[3..4])?,
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
		cell.all_parameters()?
			.into_iter()
			.map(|parameter| {
				parameter
					.gradient()
					.expect("missing RNN cell parameter gradient")
					.read_f32()
					.map(|values| values[0])
			})
			.collect::<oa::Result<Vec<_>>>()?
	} else {
		Vec::new()
	};
	Ok((loss_value, gradients))
}

test_vk!(
	rnn_cell_parameter_gradients_match_finite_differences,
	engine,
	{
		let mut values = [0.2, 0.15, 0.01, -0.03];
		let (_, gradients) = gpu_cell_loss_and_gradients(&engine, 0.25, -0.4, &values, true)?;
		let epsilon = 1.0e-3_f32;
		for parameter in 0..values.len() {
			values[parameter] += epsilon;
			let plus = gpu_cell_loss_and_gradients(&engine, 0.25, -0.4, &values, false)?.0;
			values[parameter] -= 2.0 * epsilon;
			let minus = gpu_cell_loss_and_gradients(&engine, 0.25, -0.4, &values, false)?.0;
			values[parameter] += epsilon;
			let numerical = (plus - minus) / (2.0 * epsilon);
			let actual = gradients[parameter];
			assert!(
				(actual - numerical).abs() <= 8.0e-4,
				"parameter {parameter}: actual {actual}, numerical {numerical}"
			);
		}
		Ok(())
	}
);

test_vk!(rnn_cell_preserves_bias_and_bias_free_contracts, engine, {
	let cell = oa::ml::nn::RnnCell::with_seed(&engine, 2, 3, true, 0x524e_4e01)?;
	let input = oa::Matrix::from_f32(&engine, [2, 2], &[0.1; 4])?;
	let hidden = oa::Matrix::from_f32(&engine, [2, 3], &[0.2; 6])?;
	let before_freeze = cell.step(&input, &hidden)?.read_f32()?;
	let parameters = cell.all_parameters()?;
	parameters[2].set_requires_grad(false);
	parameters[3].set_requires_grad(false);
	let after_freeze = cell.step(&input, &hidden)?.read_f32()?;
	assert_close(&after_freeze, &before_freeze, 1.0e-7);

	let bias_free = oa::ml::nn::RnnCell::with_seed(&engine, 2, 3, false, 0x524e_4e02)?;
	assert!(!bias_free.has_bias());
	assert_eq!(bias_free.all_parameters()?.len(), 2);
	assert_eq!(bias_free.step(&input, &hidden)?.shape(), [2, 3]);
	assert_eq!(
		bias_free
			.step(&oa::Matrix::from_f32(&engine, [2, 1], &[0.0; 2])?, &hidden,)
			.err()
			.expect("wrong RNN cell input width was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});
