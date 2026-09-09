fn cpu_linear(
	input: &[f32],
	weight: &[f32],
	bias: &[f32],
	batch: usize,
	input_features: usize,
	output_features: usize,
) -> Vec<f32> {
	let mut output = vec![0.0; batch * output_features];
	for row in 0..batch {
		for column in 0..output_features {
			let mut value = bias[column];
			for inner in 0..input_features {
				value +=
					input[row * input_features + inner] * weight[column * input_features + inner];
			}
			output[row * output_features + column] = value;
		}
	}
	output
}

fn cpu_cross_entropy(logits: &[f32], targets: &[u32], classes: usize) -> f32 {
	let mut total = 0.0_f64;
	for (row, target) in targets.iter().copied().enumerate() {
		let begin = row * classes;
		let row_logits = &logits[begin..begin + classes];
		let maximum = row_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
		let exponential_sum = row_logits
			.iter()
			.map(|value| f64::from(*value - maximum).exp())
			.sum::<f64>();
		total += f64::from(maximum) + exponential_sum.ln() - f64::from(row_logits[target as usize]);
	}
	(total / targets.len() as f64) as f32
}

fn cpu_loss(
	input: &[f32],
	weight: &[f32],
	bias: &[f32],
	targets: &[u32],
	input_features: usize,
	output_features: usize,
) -> f32 {
	let logits = cpu_linear(
		input,
		weight,
		bias,
		targets.len(),
		input_features,
		output_features,
	);
	cpu_cross_entropy(&logits, targets, output_features)
}

fn numerical_gradients(
	input: &[f32],
	weight: &[f32],
	bias: &[f32],
	targets: &[u32],
	input_features: usize,
	output_features: usize,
) -> (Vec<f32>, Vec<f32>) {
	const EPSILON: f32 = 1.0e-3;
	let mut weight_gradient = vec![0.0; weight.len()];
	for index in 0..weight.len() {
		let mut below = weight.to_vec();
		let mut above = weight.to_vec();
		below[index] -= EPSILON;
		above[index] += EPSILON;
		weight_gradient[index] = (cpu_loss(
			input,
			&above,
			bias,
			targets,
			input_features,
			output_features,
		) - cpu_loss(
			input,
			&below,
			bias,
			targets,
			input_features,
			output_features,
		)) / (2.0 * EPSILON);
	}
	let mut bias_gradient = vec![0.0; bias.len()];
	for index in 0..bias.len() {
		let mut below = bias.to_vec();
		let mut above = bias.to_vec();
		below[index] -= EPSILON;
		above[index] += EPSILON;
		bias_gradient[index] = (cpu_loss(
			input,
			weight,
			&above,
			targets,
			input_features,
			output_features,
		) - cpu_loss(
			input,
			weight,
			&below,
			targets,
			input_features,
			output_features,
		)) / (2.0 * EPSILON);
	}
	(weight_gradient, bias_gradient)
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		let difference = (actual - expected).abs();
		assert!(
			difference <= tolerance,
			"element {index}: expected {expected}, found {actual}, difference {difference} exceeds {tolerance}"
		);
	}
}

fn cpu_first_adamw(parameter: &[f32], gradient: &[f32], learning_rate: f32) -> Vec<f32> {
	parameter
		.iter()
		.zip(gradient)
		.map(|(parameter, gradient)| {
			parameter - learning_rate * (gradient / (gradient.abs() + 1.0e-8) + 0.01 * parameter)
		})
		.collect()
}

fn cpu_embedding(weight: &[f32], indices: &[u32], embedding_dim: usize) -> Vec<f32> {
	let mut output = Vec::with_capacity(indices.len() * embedding_dim);
	for index in indices {
		let begin = *index as usize * embedding_dim;
		output.extend_from_slice(&weight[begin..begin + embedding_dim]);
	}
	output
}

fn cpu_embedding_linear_loss(
	embedding_weight: &[f32],
	indices: &[u32],
	linear_weight: &[f32],
	linear_bias: &[f32],
	targets: &[u32],
	embedding_dim: usize,
	classes: usize,
) -> f32 {
	let embedded = cpu_embedding(embedding_weight, indices, embedding_dim);
	let logits = cpu_linear(
		&embedded,
		linear_weight,
		linear_bias,
		indices.len(),
		embedding_dim,
		classes,
	);
	cpu_cross_entropy(&logits, targets, classes)
}

fn numerical_embedding_gradient(
	embedding_weight: &[f32],
	indices: &[u32],
	linear_weight: &[f32],
	linear_bias: &[f32],
	targets: &[u32],
	embedding_dim: usize,
	classes: usize,
) -> Vec<f32> {
	const EPSILON: f32 = 1.0e-3;
	let mut gradient = vec![0.0; embedding_weight.len()];
	for index in 0..embedding_weight.len() {
		let mut below = embedding_weight.to_vec();
		let mut above = embedding_weight.to_vec();
		below[index] -= EPSILON;
		above[index] += EPSILON;
		gradient[index] = (cpu_embedding_linear_loss(
			&above,
			indices,
			linear_weight,
			linear_bias,
			targets,
			embedding_dim,
			classes,
		) - cpu_embedding_linear_loss(
			&below,
			indices,
			linear_weight,
			linear_bias,
			targets,
			embedding_dim,
			classes,
		)) / (2.0 * EPSILON);
	}
	gradient
}

#[derive(Clone, Copy)]
struct CpuRnnCase<'a> {
	input: &'a [f32],
	weight_ih: &'a [f32],
	weight_hh: &'a [f32],
	bias_ih: &'a [f32],
	bias_hh: &'a [f32],
	batch: usize,
	sequence_length: usize,
	input_size: usize,
	hidden_size: usize,
}

fn cpu_rnn(case: CpuRnnCase<'_>) -> Vec<f32> {
	let mut output = vec![0.0; case.batch * case.sequence_length * case.hidden_size];
	for batch in 0..case.batch {
		let mut hidden = vec![0.0_f32; case.hidden_size];
		for time in 0..case.sequence_length {
			let mut next = vec![0.0_f32; case.hidden_size];
			for feature in 0..case.hidden_size {
				let mut value = case.bias_ih[feature] + case.bias_hh[feature];
				for input_feature in 0..case.input_size {
					value += case.input
						[(batch * case.sequence_length + time) * case.input_size + input_feature]
						* case.weight_ih[feature * case.input_size + input_feature];
				}
				for (hidden_feature, hidden_value) in hidden.iter().enumerate() {
					value +=
						hidden_value * case.weight_hh[feature * case.hidden_size + hidden_feature];
				}
				next[feature] = value.tanh();
				output[(batch * case.sequence_length + time) * case.hidden_size + feature] =
					next[feature];
			}
			hidden = next;
		}
	}
	output
}

#[derive(Clone, Copy)]
struct CpuCharRnnCase<'a> {
	embedding: &'a [f32],
	indices: &'a [u32],
	weight_ih: &'a [f32],
	weight_hh: &'a [f32],
	bias_ih: &'a [f32],
	bias_hh: &'a [f32],
	head_weight: &'a [f32],
	head_bias: &'a [f32],
	targets: &'a [u32],
	embedding_dim: usize,
	hidden_size: usize,
	classes: usize,
}

fn cpu_char_rnn_loss(case: CpuCharRnnCase<'_>) -> (Vec<f32>, f32) {
	let embedded = cpu_embedding(case.embedding, case.indices, case.embedding_dim);
	let recurrent = cpu_rnn(CpuRnnCase {
		input: &embedded,
		weight_ih: case.weight_ih,
		weight_hh: case.weight_hh,
		bias_ih: case.bias_ih,
		bias_hh: case.bias_hh,
		batch: 1,
		sequence_length: case.indices.len(),
		input_size: case.embedding_dim,
		hidden_size: case.hidden_size,
	});
	let logits = cpu_linear(
		&recurrent,
		case.head_weight,
		case.head_bias,
		case.indices.len(),
		case.hidden_size,
		case.classes,
	);
	let loss = cpu_cross_entropy(&logits, case.targets, case.classes);
	(recurrent, loss)
}

fn numerical_vector_gradient(values: &[f32], mut evaluate: impl FnMut(&[f32]) -> f32) -> Vec<f32> {
	const EPSILON: f32 = 1.0e-3;
	let mut gradient = vec![0.0; values.len()];
	for index in 0..values.len() {
		let mut below = values.to_vec();
		let mut above = values.to_vec();
		below[index] -= EPSILON;
		above[index] += EPSILON;
		gradient[index] = (evaluate(&above) - evaluate(&below)) / (2.0 * EPSILON);
	}
	gradient
}

test_vk!(
	linear_cross_entropy_backward_and_adamw_match_independent_oracles,
	engine,
	{
		const BATCH: usize = 2;
		const INPUT_FEATURES: usize = 2;
		const OUTPUT_FEATURES: usize = 3;
		const LEARNING_RATE: f32 = 0.01;
		let input_values = [1.0_f32, 2.0, -1.0, 0.5];
		let weight_values = [0.2_f32, -0.1, -0.3, 0.4, 0.1, 0.2];
		let bias_values = [0.01_f32, -0.02, 0.03];
		let target_values = [2_u32, 0];
		let expected_logits = cpu_linear(
			&input_values,
			&weight_values,
			&bias_values,
			BATCH,
			INPUT_FEATURES,
			OUTPUT_FEATURES,
		);
		let expected_loss = cpu_cross_entropy(&expected_logits, &target_values, OUTPUT_FEATURES);
		let (expected_weight_gradient, expected_bias_gradient) = numerical_gradients(
			&input_values,
			&weight_values,
			&bias_values,
			&target_values,
			INPUT_FEATURES,
			OUTPUT_FEATURES,
		);

		let input = oa::Matrix::from_f32(&engine, [BATCH, INPUT_FEATURES], &input_values)?;
		let weight =
			oa::Matrix::from_f32(&engine, [OUTPUT_FEATURES, INPUT_FEATURES], &weight_values)?;
		let bias = oa::Matrix::from_f32(&engine, [OUTPUT_FEATURES], &bias_values)?;
		let targets = oa::Matrix::from_slice(&engine, [BATCH], &target_values)?;
		assert_eq!(targets.read::<u32>()?, target_values);
		let layer = oa::ml::nn::Linear::from_matrices(weight, bias)?;
		let weight_parameter = layer.weight();
		let bias_parameter = layer.bias();
		let mut optimizer = oa::ml::AdamW::new(layer.parameters(), LEARNING_RATE)?;

		let tape = oa::ml::GradientTape::new();
		let logits = layer.forward(&input)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
		tape.backward(&loss)?;
		assert_close(&logits.read_f32()?, &expected_logits, 1.0e-5);
		assert_close(&loss.read_f32()?, &[expected_loss], 1.0e-5);
		let weight_gradient = weight_parameter
			.gradient()
			.expect("backward did not produce a weight gradient")
			.read_f32()?;
		let bias_gradient = bias_parameter
			.gradient()
			.expect("backward did not produce a bias gradient")
			.read_f32()?;
		assert_close(&weight_gradient, &expected_weight_gradient, 2.0e-4);
		assert_close(&bias_gradient, &expected_bias_gradient, 2.0e-4);
		assert_eq!(
			tape.backward(&loss).unwrap_err().kind(),
			oa::ErrorKind::FailedPrecondition
		);

		let expected_weight = cpu_first_adamw(&weight_values, &weight_gradient, LEARNING_RATE);
		let expected_bias = cpu_first_adamw(&bias_values, &bias_gradient, LEARNING_RATE);
		let zero = oa::matrix::full(&engine, [OUTPUT_FEATURES, INPUT_FEATURES], 0.0)?;
		let (stable_weight_plan, stable_weight) =
			engine.capture(|| oa::matrix::add(&weight_parameter.data(), &zero))?;
		optimizer.step()?;
		assert_eq!(optimizer.step_count(), 1);
		engine.submit(&stable_weight_plan)?.wait()?;
		assert_close(&stable_weight.read_f32()?, &expected_weight, 2.0e-5);
		assert_close(
			&weight_parameter.data().read_f32()?,
			&expected_weight,
			2.0e-5,
		);
		assert_close(&bias_parameter.data().read_f32()?, &expected_bias, 2.0e-5);
		optimizer.zero_grad();
		assert!(weight_parameter.gradient().is_none());
		assert!(bias_parameter.gradient().is_none());
		Ok(())
	}
);

test_vk!(
	linear_training_reduces_a_fixed_classification_loss,
	engine,
	{
		let input =
			oa::Matrix::from_f32(&engine, [4, 2], &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0])?;
		let targets = oa::Matrix::from_slice(&engine, [4], &[0_u32, 1, 1, 0])?;
		let layer = oa::ml::nn::Linear::with_seed(&engine, 2, 2, 0x4f41)?;
		let mut optimizer = oa::ml::AdamW::new(layer.parameters(), 0.05)?;
		let mut initial_loss = 0.0;
		let mut final_loss = 0.0;

		for step in 0..24 {
			optimizer.zero_grad();
			let tape = oa::ml::GradientTape::new();
			let logits = layer.forward(&input)?;
			let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
			tape.backward(&loss)?;
			let value = loss.read_f32()?[0];
			if step == 0 {
				initial_loss = value;
			}
			final_loss = value;
			optimizer.step()?;
		}
		assert!(
			final_loss < initial_loss * 0.7,
			"expected training loss to decrease materially; initial={initial_loss}, final={final_loss}"
		);
		Ok(())
	}
);

test_vk!(
	captured_training_program_matches_eager_adamw_and_reuses_one_command,
	engine,
	{
		const INPUTS: [[f32; 4]; 4] = [
			[1.0, 0.0, 0.0, 1.0],
			[-1.0, 0.5, 0.25, -0.75],
			[0.5, 1.0, -0.5, -1.0],
			[0.25, -0.5, 1.0, 0.75],
		];
		const TARGETS: [[u32; 2]; 4] = [[0, 1], [1, 0], [0, 1], [1, 0]];
		let captured_input = oa::Matrix::from_f32(&engine, [2, 2], &INPUTS[0])?;
		let captured_target = oa::Matrix::from_slice(&engine, [2], &TARGETS[0])?;
		let captured_layer = oa::ml::nn::Linear::with_seed(&engine, 2, 2, 0x5447)?;
		let eager_layer = oa::ml::nn::Linear::with_seed(&engine, 2, 2, 0x5447)?;
		let mut captured_optimizer = oa::ml::AdamW::new(captured_layer.parameters(), 0.01)?;
		let mut eager_optimizer = oa::ml::AdamW::new(eager_layer.parameters(), 0.01)?;
		let mut program =
			oa::ml::TrainingProgram::capture(&engine, &mut captured_optimizer, || {
				let tape = oa::ml::GradientTape::new();
				let logits = captured_layer.forward(&captured_input)?;
				let loss = oa::ml::loss::cross_entropy(&logits, &captured_target)?;
				tape.backward(&loss)?;
				Ok(loss)
			})?;
		assert_eq!(captured_optimizer.step_count(), 0);

		for step in 0..INPUTS.len() {
			if step != 0 {
				program.upload_input(&captured_input, &INPUTS[step])?;
				program.upload_input(&captured_target, &TARGETS[step])?;
			}
			let captured_loss = program.replay_and_wait(&engine, &mut captured_optimizer)?;

			let input = oa::Matrix::from_f32(&engine, [2, 2], &INPUTS[step])?;
			let target = oa::Matrix::from_slice(&engine, [2], &TARGETS[step])?;
			eager_optimizer.zero_grad();
			let tape = oa::ml::GradientTape::new();
			let logits = eager_layer.forward(&input)?;
			let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
			tape.backward(&loss)?;
			let eager_loss = loss.read_f32()?[0];
			eager_optimizer.step()?;
			assert!((captured_loss - eager_loss).abs() <= 1.0e-6);
		}

		assert_eq!(captured_optimizer.step_count(), 4);
		assert_eq!(eager_optimizer.step_count(), 4);
		for (captured, eager) in captured_layer
			.parameters()
			.iter()
			.zip(eager_layer.parameters().iter())
		{
			assert_close(
				&captured.data().read_f32()?,
				&eager.data().read_f32()?,
				1.0e-6,
			);
		}
		let diagnostics = program.diagnostics();
		assert_eq!(
			diagnostics.schema_owned_node_count() + diagnostics.compatibility_node_count(),
			diagnostics.node_count()
		);
		assert!(diagnostics.compatibility_node_count() > 0);
		assert!(diagnostics.schema_owned_node_count() > diagnostics.compatibility_node_count());
		assert!(diagnostics.semantic_autograd_attachment_count() > 0);
		assert_eq!(
			diagnostics.semantic_autograd_expanded_count(),
			diagnostics.semantic_autograd_attachment_count()
		);
		assert!(diagnostics.semantic_backward_operation_count() > 0);
		assert!(
			program
				.semantic_graph()
				.operations()
				.iter()
				.any(|operation| operation.backward_of().is_some())
		);
		assert!(diagnostics.dnn_captured_operation_count() > 0);
		assert!(diagnostics.dnn_recognized_partition_count() > 0);
		assert_eq!(diagnostics.observed_output_count(), 1);
		assert_eq!(
			diagnostics.semantic_binding_count(),
			program.semantic_graph().values().len()
		);
		assert_eq!(diagnostics.command_recording_count(), 1);
		assert_eq!(diagnostics.command_cache_hit_count(), 3);
		assert_eq!(diagnostics.submission_count(), 4);
		assert_eq!(diagnostics.input_upload_count(), 6);
		assert_eq!(program.replay_count(), 4);

		captured_optimizer.zero_grad();
		assert_eq!(
			program
				.replay(&engine, &mut captured_optimizer)
				.unwrap_err()
				.kind(),
			oa::ErrorKind::FailedPrecondition
		);
		Ok(())
	}
);

test_vk!(
	embedding_reshape_and_scatter_add_backward_match_independent_oracles,
	engine,
	{
		const VOCABULARY: usize = 4;
		const EMBEDDING_DIM: usize = 3;
		const CLASSES: usize = 2;
		let embedding_values = [
			0.1_f32, -0.2, 0.3, 0.4, 0.5, -0.6, -0.7, 0.8, 0.9, 0.2, -0.4, 0.6,
		];
		let index_values = [2_u32, 1, 2, 0];
		let linear_values = [0.3_f32, -0.1, 0.2, -0.4, 0.5, 0.1];
		let bias_values = [0.02_f32, -0.03];
		let target_values = [1_u32, 0, 1, 0];
		let expected_embedded = cpu_embedding(&embedding_values, &index_values, EMBEDDING_DIM);
		let expected_gradient = numerical_embedding_gradient(
			&embedding_values,
			&index_values,
			&linear_values,
			&bias_values,
			&target_values,
			EMBEDDING_DIM,
			CLASSES,
		);

		let embedding_weight =
			oa::Matrix::from_f32(&engine, [VOCABULARY, EMBEDDING_DIM], &embedding_values)?;
		let indices = oa::Matrix::from_slice(&engine, [2, 2], &index_values)?;
		let linear_weight =
			oa::Matrix::from_f32(&engine, [CLASSES, EMBEDDING_DIM], &linear_values)?;
		let linear_bias = oa::Matrix::from_f32(&engine, [CLASSES], &bias_values)?;
		let targets = oa::Matrix::from_slice(&engine, [4], &target_values)?;
		let embedding = oa::ml::nn::Embedding::from_matrix(embedding_weight)?;
		let projection = oa::ml::nn::Linear::from_matrices(linear_weight, linear_bias)?;

		let tape = oa::ml::GradientTape::new();
		let embedded = embedding.forward(&indices)?;
		assert_eq!(embedded.shape(), [2, 2, EMBEDDING_DIM]);
		let flattened = embedded.reshape([4, EMBEDDING_DIM])?;
		let logits = projection.forward(&flattened)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
		tape.backward(&loss)?;

		assert_close(&embedded.read_f32()?, &expected_embedded, 1.0e-6);
		assert_close(&flattened.read_f32()?, &expected_embedded, 1.0e-6);
		let actual_gradient = embedding
			.weight()
			.gradient()
			.expect("embedding backward did not reach the table")
			.read_f32()?;
		assert_close(&actual_gradient, &expected_gradient, 3.0e-4);
		assert_close(
			&actual_gradient[3 * EMBEDDING_DIM..4 * EMBEDDING_DIM],
			&[0.0; EMBEDDING_DIM],
			1.0e-7,
		);
		Ok(())
	}
);

test_vk!(
	character_rnn_forward_and_complete_bptt_match_independent_oracles,
	engine,
	{
		const VOCABULARY: usize = 3;
		const EMBEDDING_DIM: usize = 2;
		const HIDDEN_SIZE: usize = 2;
		const CLASSES: usize = 3;
		let embedding_values = [0.1_f32, -0.2, 0.3, 0.4, -0.5, 0.6];
		let indices_values = [0_u32, 1, 0];
		let weight_ih_values = [0.2_f32, -0.1, 0.3, 0.25];
		let weight_hh_values = [0.15_f32, -0.2, 0.05, 0.1];
		let bias_ih_values = [0.01_f32, -0.02];
		let bias_hh_values = [-0.03_f32, 0.04];
		let head_weight_values = [0.2_f32, -0.1, -0.3, 0.25, 0.1, 0.35];
		let head_bias_values = [0.01_f32, -0.02, 0.03];
		let target_values = [1_u32, 2, 0];
		let cpu_case = CpuCharRnnCase {
			embedding: &embedding_values,
			indices: &indices_values,
			weight_ih: &weight_ih_values,
			weight_hh: &weight_hh_values,
			bias_ih: &bias_ih_values,
			bias_hh: &bias_hh_values,
			head_weight: &head_weight_values,
			head_bias: &head_bias_values,
			targets: &target_values,
			embedding_dim: EMBEDDING_DIM,
			hidden_size: HIDDEN_SIZE,
			classes: CLASSES,
		};
		let (expected_recurrent, _) = cpu_char_rnn_loss(cpu_case);
		let expected_embedding_gradient = numerical_vector_gradient(&embedding_values, |values| {
			cpu_char_rnn_loss(CpuCharRnnCase {
				embedding: values,
				..cpu_case
			})
			.1
		});
		let expected_weight_ih_gradient = numerical_vector_gradient(&weight_ih_values, |values| {
			cpu_char_rnn_loss(CpuCharRnnCase {
				weight_ih: values,
				..cpu_case
			})
			.1
		});
		let expected_weight_hh_gradient = numerical_vector_gradient(&weight_hh_values, |values| {
			cpu_char_rnn_loss(CpuCharRnnCase {
				weight_hh: values,
				..cpu_case
			})
			.1
		});
		let expected_bias_ih_gradient = numerical_vector_gradient(&bias_ih_values, |values| {
			cpu_char_rnn_loss(CpuCharRnnCase {
				bias_ih: values,
				..cpu_case
			})
			.1
		});
		let expected_bias_hh_gradient = numerical_vector_gradient(&bias_hh_values, |values| {
			cpu_char_rnn_loss(CpuCharRnnCase {
				bias_hh: values,
				..cpu_case
			})
			.1
		});

		let embedding_weight =
			oa::Matrix::from_f32(&engine, [VOCABULARY, EMBEDDING_DIM], &embedding_values)?;
		let indices = oa::Matrix::from_slice(&engine, [1, 3], &indices_values)?;
		let weight_ih =
			oa::Matrix::from_f32(&engine, [HIDDEN_SIZE, EMBEDDING_DIM], &weight_ih_values)?;
		let weight_hh =
			oa::Matrix::from_f32(&engine, [HIDDEN_SIZE, HIDDEN_SIZE], &weight_hh_values)?;
		let bias_ih = oa::Matrix::from_f32(&engine, [HIDDEN_SIZE], &bias_ih_values)?;
		let bias_hh = oa::Matrix::from_f32(&engine, [HIDDEN_SIZE], &bias_hh_values)?;
		let head_weight =
			oa::Matrix::from_f32(&engine, [CLASSES, HIDDEN_SIZE], &head_weight_values)?;
		let head_bias = oa::Matrix::from_f32(&engine, [CLASSES], &head_bias_values)?;
		let targets = oa::Matrix::from_slice(&engine, [3], &target_values)?;
		let embedding = oa::ml::nn::Embedding::from_matrix(embedding_weight)?;
		let rnn = oa::ml::nn::Rnn::from_matrices(weight_ih, weight_hh, bias_ih, bias_hh)?;
		let head = oa::ml::nn::Linear::from_matrices(head_weight, head_bias)?;

		let tape = oa::ml::GradientTape::new();
		let embedded = embedding.forward(&indices)?;
		let recurrent = rnn.forward(&embedded)?;
		let logits = head.forward(&recurrent.reshape([3, HIDDEN_SIZE])?)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
		tape.backward(&loss)?;

		assert_eq!(recurrent.shape(), [1, 3, HIDDEN_SIZE]);
		assert_close(&recurrent.read_f32()?, &expected_recurrent, 2.0e-5);
		assert_close(
			&embedding
				.weight()
				.gradient()
				.expect("RNN backward did not reach embedding")
				.read_f32()?,
			&expected_embedding_gradient,
			7.0e-4,
		);
		let recurrent_parameters = rnn.layer_parameters(0).expect("missing RNN layer");
		for (parameter, expected) in recurrent_parameters.iter().zip([
			&expected_weight_ih_gradient,
			&expected_weight_hh_gradient,
			&expected_bias_ih_gradient,
			&expected_bias_hh_gradient,
		]) {
			assert_close(
				&parameter
					.gradient()
					.expect("RNN parameter gradient is missing")
					.read_f32()?,
				expected,
				7.0e-4,
			);
		}
		Ok(())
	}
);

test_vk!(
	stacked_rnn_preserves_batch_sequence_and_parameter_order,
	engine,
	{
		let rnn = oa::ml::nn::Rnn::with_seed(&engine, 2, 3, 2, 0x0052_4e4e)?;
		assert_eq!(rnn.input_size(), 2);
		assert_eq!(rnn.hidden_size(), 3);
		assert_eq!(rnn.num_layers(), 2);
		assert_eq!(rnn.all_parameters()?.len(), 8);
		let input = oa::Matrix::from_f32(&engine, [2, 3, 2], &[0.1; 12])?;
		let output = rnn.forward(&input)?;
		assert_eq!(output.shape(), [2, 3, 3]);
		assert!(output.read_f32()?.iter().all(|value| value.is_finite()));

		assert_eq!(
			oa::ml::nn::Rnn::with_seed(&engine, 2, 1025, 1, 7)
				.err()
				.expect("oversized hidden state was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		let wrong_width = oa::Matrix::from_f32(&engine, [1, 2, 4], &[0.0; 8])?;
		assert_eq!(
			rnn.forward(&wrong_width)
				.err()
				.expect("wrong RNN input width was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		Ok(())
	}
);

test_vk!(embedding_and_reshape_reject_invalid_contracts, engine, {
	let weight = oa::Matrix::from_f32(&engine, [3, 2], &[0.0; 6])?;
	let embedding = oa::ml::nn::Embedding::from_matrix(weight)?;
	let signed = oa::Matrix::from_slice(&engine, [2], &[0_i32; 2])?;
	assert_eq!(
		embedding
			.forward(&signed)
			.err()
			.expect("signed embedding indices were accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);

	let values = oa::Matrix::from_f32(&engine, [2, 3], &[0.0; 6])?;
	assert_eq!(
		values
			.reshape([5])
			.err()
			.expect("element-changing reshape was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	let scalar = values.reshape([]);
	assert_eq!(
		scalar
			.err()
			.expect("non-scalar matrix was reshaped to scalar")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});

test_vk!(
	out_of_range_embedding_index_produces_nan_without_oob_access,
	engine,
	{
		let weight = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 2.0, 3.0, 4.0])?;
		let indices = oa::Matrix::from_slice(&engine, [2], &[1_u32, 2])?;
		let embedding = oa::ml::nn::Embedding::from_matrix(weight)?;
		let values = embedding.forward(&indices)?.read_f32()?;
		assert_eq!(&values[..2], &[3.0, 4.0]);
		assert!(values[2..].iter().all(|value| value.is_nan()));
		Ok(())
	}
);

test_vk!(
	ml_training_contracts_reject_invalid_shapes_and_dtypes,
	engine,
	{
		let input = oa::Matrix::from_f32(&engine, [2, 2], &[0.0; 4])?;
		let weight = oa::Matrix::from_f32(&engine, [3, 2], &[0.0; 6])?;
		let wrong_bias = oa::Matrix::from_f32(&engine, [2], &[0.0; 2])?;
		assert_eq!(
			oa::ml::nn::Linear::from_matrices(weight, wrong_bias)
				.err()
				.expect("invalid linear contract was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);

		let logits = oa::Matrix::from_f32(&engine, [2, 3], &[0.0; 6])?;
		let signed_targets = oa::Matrix::from_slice(&engine, [2], &[0_i32; 2])?;
		let error = oa::ml::loss::cross_entropy(&logits, &signed_targets)
			.err()
			.expect("signed targets were accepted");
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
		let wrong_targets = oa::Matrix::from_slice(&engine, [1], &[0_u32])?;
		let error = oa::ml::loss::cross_entropy(&logits, &wrong_targets)
			.err()
			.expect("shape-mismatched targets were accepted");
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);

		let targets = oa::Matrix::from_slice(&engine, [2], &[0_u32, 2])?;
		let layer = oa::ml::nn::Linear::with_seed(&engine, 2, 3, 7)?;
		let output = layer.forward(&input)?;
		let foreign_engine = oa::Engine::new()?;
		let foreign_targets = oa::Matrix::from_slice(&foreign_engine, [2], &[0_u32, 2])?;
		let error = oa::ml::loss::cross_entropy(&output, &foreign_targets)
			.err()
			.expect("foreign targets were accepted");
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
		let valid_loss = oa::ml::loss::cross_entropy(&output, &targets)?;
		assert!(valid_loss.shape().is_empty());
		Ok(())
	}
);

test_vk!(
	out_of_range_cross_entropy_target_produces_nan_without_oob_access,
	engine,
	{
		let logits = oa::Matrix::from_f32(&engine, [1, 3], &[1.0, 2.0, 3.0])?;
		let targets = oa::Matrix::from_slice(&engine, [1], &[3_u32])?;
		let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
		assert!(loss.read_f32()?[0].is_nan());
		Ok(())
	}
);

test_vk!(
	backward_rejects_a_parameter_changed_after_forward,
	engine,
	{
		let input = oa::Matrix::from_f32(&engine, [1, 2], &[1.0, -1.0])?;
		let targets = oa::Matrix::from_slice(&engine, [1], &[0_u32])?;
		let layer = oa::ml::nn::Linear::with_seed(&engine, 2, 2, 99)?;
		let mut optimizer = oa::ml::AdamW::new(layer.parameters(), 0.01)?;

		let first_tape = oa::ml::GradientTape::new();
		let first_logits = layer.forward(&input)?;
		let first_loss = oa::ml::loss::cross_entropy(&first_logits, &targets)?;
		first_tape.backward(&first_loss)?;
		let _ = first_loss.read_f32()?;

		let stale_tape = oa::ml::GradientTape::new();
		let stale_logits = layer.forward(&input)?;
		let stale_loss = oa::ml::loss::cross_entropy(&stale_logits, &targets)?;
		optimizer.step()?;
		let error = stale_tape
			.backward(&stale_loss)
			.expect_err("backward accepted a changed saved parameter");
		assert_eq!(error.kind(), oa::ErrorKind::FailedPrecondition);
		assert!(error.message().contains("changed after"));
		Ok(())
	}
);

test_vk!(
	backward_rejects_an_embedding_changed_after_forward,
	engine,
	{
		let embedding_weight =
			oa::Matrix::from_f32(&engine, [3, 2], &[0.1, 0.2, -0.3, 0.4, 0.5, -0.6])?;
		let linear_weight = oa::Matrix::from_f32(&engine, [2, 2], &[0.2, -0.1, 0.3, 0.4])?;
		let linear_bias = oa::Matrix::from_f32(&engine, [2], &[0.0, 0.0])?;
		let indices = oa::Matrix::from_slice(&engine, [2], &[0_u32, 2])?;
		let targets = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
		let embedding = oa::ml::nn::Embedding::from_matrix(embedding_weight)?;
		let projection = oa::ml::nn::Linear::from_matrices(linear_weight, linear_bias)?;
		let mut optimizer = oa::ml::AdamW::new(embedding.parameters(), 0.01)?;

		let first_tape = oa::ml::GradientTape::new();
		let first_embedded = embedding.forward(&indices)?;
		let first_logits = projection.forward(&first_embedded)?;
		let first_loss = oa::ml::loss::cross_entropy(&first_logits, &targets)?;
		first_tape.backward(&first_loss)?;
		let _ = first_loss.read_f32()?;

		let stale_tape = oa::ml::GradientTape::new();
		let stale_embedded = embedding.forward(&indices)?;
		let stale_logits = projection.forward(&stale_embedded)?;
		let stale_loss = oa::ml::loss::cross_entropy(&stale_logits, &targets)?;
		optimizer.step()?;
		let error = stale_tape
			.backward(&stale_loss)
			.expect_err("backward accepted a changed saved embedding");
		assert_eq!(error.kind(), oa::ErrorKind::FailedPrecondition);
		assert!(error.message().contains("changed after"));
		Ok(())
	}
);
