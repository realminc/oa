use oa::ml::Module;

const EPSILON: f32 = 1.0e-5;

fn cpu_layer_norm(
	input: &[f32],
	weight: &[f32],
	bias: &[f32],
	columns: usize,
	epsilon: f32,
) -> Vec<f32> {
	let mut output = vec![0.0; input.len()];
	for (input_row, output_row) in input
		.chunks_exact(columns)
		.zip(output.chunks_exact_mut(columns))
	{
		let mean = input_row.iter().map(|value| f64::from(*value)).sum::<f64>() / columns as f64;
		let variance = input_row
			.iter()
			.map(|value| {
				let centered = f64::from(*value) - mean;
				centered * centered
			})
			.sum::<f64>()
			/ columns as f64;
		let inverse_stddev = 1.0 / (variance + f64::from(epsilon)).sqrt();
		for column in 0..columns {
			output_row[column] =
				((f64::from(input_row[column]) - mean) * inverse_stddev * f64::from(weight[column])
					+ f64::from(bias[column])) as f32;
		}
	}
	output
}

fn cpu_linear(
	input: &[f32],
	weight: &[f32],
	bias: &[f32],
	input_features: usize,
	output_features: usize,
) -> Vec<f32> {
	let rows = input.len() / input_features;
	let mut output = vec![0.0; rows * output_features];
	for row in 0..rows {
		for output_feature in 0..output_features {
			let mut value = bias[output_feature];
			for input_feature in 0..input_features {
				value += input[row * input_features + input_feature]
					* weight[output_feature * input_features + input_feature];
			}
			output[row * output_features + output_feature] = value;
		}
	}
	output
}

fn cpu_cross_entropy(logits: &[f32], targets: &[u32], classes: usize) -> f32 {
	let mut loss = 0.0_f64;
	for (row, target) in targets.iter().copied().enumerate() {
		let values = &logits[row * classes..(row + 1) * classes];
		let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
		let sum = values
			.iter()
			.map(|value| f64::from(*value - maximum).exp())
			.sum::<f64>();
		loss += f64::from(maximum) + sum.ln() - f64::from(values[target as usize]);
	}
	(loss / targets.len() as f64) as f32
}

#[derive(Clone, Copy)]
struct CpuCase<'a> {
	embedding: &'a [f32],
	indices: &'a [u32],
	norm_weight: &'a [f32],
	norm_bias: &'a [f32],
	linear_weight: &'a [f32],
	linear_bias: &'a [f32],
	targets: &'a [u32],
	features: usize,
	classes: usize,
}

fn cpu_loss(case: CpuCase<'_>) -> f32 {
	let mut embedded = Vec::with_capacity(case.indices.len() * case.features);
	for index in case.indices {
		let begin = *index as usize * case.features;
		embedded.extend_from_slice(&case.embedding[begin..begin + case.features]);
	}
	let normalized = cpu_layer_norm(
		&embedded,
		case.norm_weight,
		case.norm_bias,
		case.features,
		EPSILON,
	);
	let logits = cpu_linear(
		&normalized,
		case.linear_weight,
		case.linear_bias,
		case.features,
		case.classes,
	);
	cpu_cross_entropy(&logits, case.targets, case.classes)
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

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		let error = (actual - expected).abs();
		assert!(
			error <= tolerance,
			"element {index}: expected {expected}, found {actual}, error {error} exceeds {tolerance}"
		);
	}
}

test_vk!(
	layer_norm_forward_matches_independent_host_oracle,
	engine,
	{
		let input_values = [
			0.25_f32, -0.5, 1.25, 3.0, 2.0, -1.0, 0.125, 0.5, 0.75, -2.0, 4.0, 1.0,
		];
		let weight_values = [0.75_f32, -1.25, 0.5];
		let bias_values = [0.1_f32, -0.2, 0.3];
		let expected = cpu_layer_norm(&input_values, &weight_values, &bias_values, 3, EPSILON);
		let input = oa::Matrix::from_f32(&engine, [2, 2, 3], &input_values)?;
		let weight = oa::Matrix::from_f32(&engine, [3], &weight_values)?;
		let bias = oa::Matrix::from_f32(&engine, [3], &bias_values)?;
		let layer = oa::ml::nn::LayerNorm::from_matrices(weight, bias, EPSILON)?;
		let output = layer.forward(&input)?;

		assert_eq!(output.shape(), [2, 2, 3]);
		assert_eq!(layer.normalized_shape(), 3);
		assert_eq!(layer.epsilon(), EPSILON);
		assert_close(&output.read_f32()?, &expected, 2.0e-5);
		assert_eq!(
			layer
				.named_parameters()
				.iter()
				.map(|entry| entry.path())
				.collect::<Vec<_>>(),
			["weight", "bias"]
		);

		let offset_values = [10_000.25_f32, 9_999.5, 10_001.25];
		let offset_expected = cpu_layer_norm(&offset_values, &[1.0; 3], &[0.0; 3], 3, EPSILON);
		let offset_input = oa::Matrix::from_f32(&engine, [1, 3], &offset_values)?;
		let stable_layer = oa::ml::nn::LayerNorm::new(&engine, 3, EPSILON)?;
		assert_close(
			&stable_layer.forward(&offset_input)?.read_f32()?,
			&offset_expected,
			2.0e-3,
		);
		Ok(())
	}
);

test_vk!(layer_norm_backward_matches_finite_differences, engine, {
	const VOCABULARY: usize = 3;
	const FEATURES: usize = 3;
	const CLASSES: usize = 2;
	let embedding_values = [0.2_f32, -0.4, 0.7, -0.3, 0.6, 0.1, 0.8, -0.2, -0.5];
	let indices_values = [0_u32, 1, 0, 2];
	let norm_weight_values = [0.9_f32, -0.7, 1.2];
	let norm_bias_values = [0.05_f32, -0.1, 0.2];
	let linear_weight_values = [0.3_f32, -0.2, 0.4, -0.5, 0.6, 0.1];
	let linear_bias_values = [0.02_f32, -0.03];
	let target_values = [1_u32, 0, 1, 0];
	let cpu_case = CpuCase {
		embedding: &embedding_values,
		indices: &indices_values,
		norm_weight: &norm_weight_values,
		norm_bias: &norm_bias_values,
		linear_weight: &linear_weight_values,
		linear_bias: &linear_bias_values,
		targets: &target_values,
		features: FEATURES,
		classes: CLASSES,
	};
	let expected_embedding = numerical_gradient(&embedding_values, |values| {
		cpu_loss(CpuCase {
			embedding: values,
			..cpu_case
		})
	});
	let expected_weight = numerical_gradient(&norm_weight_values, |values| {
		cpu_loss(CpuCase {
			norm_weight: values,
			..cpu_case
		})
	});
	let expected_bias = numerical_gradient(&norm_bias_values, |values| {
		cpu_loss(CpuCase {
			norm_bias: values,
			..cpu_case
		})
	});

	let embedding = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
		&engine,
		[VOCABULARY, FEATURES],
		&embedding_values,
	)?)?;
	let norm = oa::ml::nn::LayerNorm::from_matrices(
		oa::Matrix::from_f32(&engine, [FEATURES], &norm_weight_values)?,
		oa::Matrix::from_f32(&engine, [FEATURES], &norm_bias_values)?,
		EPSILON,
	)?;
	let linear = oa::ml::nn::Linear::from_matrices(
		oa::Matrix::from_f32(&engine, [CLASSES, FEATURES], &linear_weight_values)?,
		oa::Matrix::from_f32(&engine, [CLASSES], &linear_bias_values)?,
	)?;
	let indices = oa::Matrix::from_slice(&engine, [2, 2], &indices_values)?;
	let targets = oa::Matrix::from_slice(&engine, [4], &target_values)?;

	let (plan, ()) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let embedded = embedding.forward(&indices)?;
		let normalized = norm.forward(&embedded)?;
		let logits = linear.forward(&normalized.reshape([4, FEATURES])?)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
		tape.backward(&loss)
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("LayerNorm"))
		.expect("LayerNorm executable report must be valid JSON");
	let sum_axis_nodes = report["nodes"]
		.as_array()
		.expect("LayerNorm executable nodes must be an array")
		.iter()
		.filter(|node| node["kernel"] == "matrix.sum_axis.f32")
		.collect::<Vec<_>>();
	assert_eq!(sum_axis_nodes.len(), 2);
	for node in sum_axis_nodes {
		assert_eq!(node["physical_write"]["writes"][0]["binding"], 1);
		assert_eq!(
			node["physical_write"]["writes"][0]["partition"],
			"exclusive_per_invocation"
		);
		assert_eq!(node["physical_write"]["workspace"], "none");
	}
	let layer_norm_backward = report["nodes"]
		.as_array()
		.expect("LayerNorm executable nodes must be an array")
		.iter()
		.find(|node| node["kernel"] == "ml.matrix.layer_norm_backward.f32")
		.expect("LayerNorm backward candidate must be visible");
	assert_eq!(
		layer_norm_backward["physical_write"]["writes"]
			.as_array()
			.expect("LayerNorm writes must be an array")
			.len(),
		2
	);
	assert_eq!(
		layer_norm_backward["physical_write"]["workspace"],
		"exclusive_per_workgroup"
	);
	assert!(!report.to_string().contains("layer_norm_parameter_backward"));
	engine.submit(&plan)?.wait()?;

	assert_close(
		&embedding
			.weight()
			.gradient()
			.expect("missing input-path gradient")
			.read_f32()?,
		&expected_embedding,
		6.0e-4,
	);
	assert_close(
		&norm
			.weight()
			.gradient()
			.expect("missing weight gradient")
			.read_f32()?,
		&expected_weight,
		6.0e-4,
	);
	assert_close(
		&norm
			.bias()
			.gradient()
			.expect("missing bias gradient")
			.read_f32()?,
		&expected_bias,
		6.0e-4,
	);

	let parameters = [norm.weight(), norm.bias()];
	let before = parameters
		.iter()
		.map(|parameter| parameter.data().read_f32())
		.collect::<oa::Result<Vec<_>>>()?;
	let mut optimizer = oa::ml::AdamW::new(parameters.to_vec(), 0.01)?;
	optimizer.step()?;
	for (parameter, before) in parameters.iter().zip(before) {
		assert_ne!(parameter.data().read_f32()?, before);
	}
	Ok(())
});

test_vk!(layer_norm_rejects_invalid_contracts, engine, {
	for result in [
		oa::ml::nn::LayerNorm::new(&engine, 0, EPSILON),
		oa::ml::nn::LayerNorm::new(&engine, 3, 0.0),
		oa::ml::nn::LayerNorm::new(&engine, 3, f32::NAN),
	] {
		assert_eq!(
			result
				.err()
				.expect("invalid constructor was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}

	let layer = oa::ml::nn::LayerNorm::new(&engine, 3, EPSILON)?;
	let wrong_width = oa::Matrix::from_f32(&engine, [2, 2], &[0.0; 4])?;
	let integer = oa::Matrix::from_slice(&engine, [2, 3], &[0_i32; 6])?;
	let empty = oa::Matrix::from_f32(&engine, [0, 3], &[])?;
	for input in [&wrong_width, &integer, &empty] {
		assert_eq!(
			layer
				.forward(input)
				.err()
				.expect("invalid input was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	Ok(())
});
