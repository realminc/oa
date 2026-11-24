//! ONNX import and named-state round-trip contracts.

use std::time::SystemTime;

// Minimal protobuf wire writer for ONNX test fixtures (length-delimited and
// varint fields only — the subset the importer parses).
fn varint_bytes(mut value: u64) -> Vec<u8> {
	let mut out = Vec::new();
	loop {
		let byte = (value & 0x7f) as u8;
		value >>= 7;
		out.push(if value == 0 { byte } else { byte | 0x80 });
		if value == 0 {
			return out;
		}
	}
}

fn key(number: u64, wire: u64) -> Vec<u8> {
	varint_bytes((number << 3) | wire)
}

fn field_varint(number: u64, value: u64) -> Vec<u8> {
	let mut out = key(number, 0);
	out.extend(varint_bytes(value));
	out
}

fn field_bytes(number: u64, payload: &[u8]) -> Vec<u8> {
	let mut out = key(number, 2);
	out.extend(varint_bytes(payload.len() as u64));
	out.extend_from_slice(payload);
	out
}

fn onnx_tensor(name: &str, dims: &[usize], values: &[f32]) -> Vec<u8> {
	let mut packed_dims = Vec::new();
	for dim in dims {
		packed_dims.extend(varint_bytes(*dim as u64));
	}
	let mut tensor = field_bytes(1, &packed_dims);
	tensor.extend(field_varint(2, 1)); // FLOAT
	tensor.extend(field_bytes(8, name.as_bytes()));
	let mut raw = Vec::with_capacity(values.len() * 4);
	for value in values {
		raw.extend_from_slice(&value.to_le_bytes());
	}
	tensor.extend(field_bytes(9, &raw));
	tensor
}

fn onnx_attribute_i(name: &str, value: i64) -> Vec<u8> {
	let mut attribute = field_bytes(1, name.as_bytes());
	attribute.extend(field_varint(3, value as u64));
	attribute
}

fn onnx_node(op_type: &str, inputs: &[&str], output: &str, attributes: &[Vec<u8>]) -> Vec<u8> {
	let mut node = Vec::new();
	for input in inputs {
		node.extend(field_bytes(1, input.as_bytes()));
	}
	node.extend(field_bytes(2, output.as_bytes()));
	node.extend(field_bytes(4, op_type.as_bytes()));
	for attribute in attributes {
		node.extend(field_bytes(5, attribute));
	}
	node
}

fn onnx_value_info(name: &str) -> Vec<u8> {
	field_bytes(1, name.as_bytes())
}

fn onnx_model(nodes: &[Vec<u8>], initializers: &[Vec<u8>], input: &str, output: &str) -> Vec<u8> {
	let mut graph = Vec::new();
	for node in nodes {
		graph.extend(field_bytes(1, node));
	}
	for initializer in initializers {
		graph.extend(field_bytes(5, initializer));
	}
	graph.extend(field_bytes(11, &onnx_value_info(input)));
	graph.extend(field_bytes(12, &onnx_value_info(output)));
	field_bytes(7, &graph)
}

test_vk!(
	onnx_imported_mlp_reproduces_reference_logits_on_device,
	engine,
	{
		// MLP: Relu(x @ W1 + b1) @ W2 + b2 — MatMul/Add/Relu nodes.
		let w1 = vec![
			0.10, -0.20, 0.30, 0.05, -0.15, 0.25, -0.05, 0.40, //
			-0.30, 0.12, -0.08, 0.22, 0.18, -0.33, 0.09, -0.11, //
			0.21, 0.07, -0.25, -0.13, 0.31, -0.04, 0.14, 0.27, //
			-0.09, 0.35, 0.02, -0.19, -0.28, 0.16, -0.22, 0.08,
		];
		let b1 = vec![0.05, -0.10, 0.15, -0.20, 0.25, -0.05, 0.10, -0.15];
		let w2 = vec![
			0.20, -0.15, 0.10, //
			-0.10, 0.25, -0.20, //
			0.15, -0.05, 0.30, //
			-0.25, 0.10, -0.15, //
			0.05, -0.20, 0.15, //
			-0.15, 0.30, -0.10, //
			0.10, -0.10, 0.20, //
			0.25, 0.05, -0.25,
		];
		let b2 = vec![0.10, -0.05, 0.20];
		let nodes = vec![
			onnx_node("MatMul", &["x", "w1"], "m1", &[]),
			onnx_node("Add", &["m1", "b1"], "a1", &[]),
			onnx_node("Relu", &["a1"], "h", &[]),
			onnx_node("MatMul", &["h", "w2"], "m2", &[]),
			onnx_node("Add", &["m2", "b2"], "y", &[]),
		];
		let initializers = vec![
			onnx_tensor("w1", &[4, 8], &w1),
			onnx_tensor("b1", &[8], &b1),
			onnx_tensor("w2", &[8, 3], &w2),
			onnx_tensor("b2", &[3], &b2),
		];
		let model = onnx_model(&nodes, &initializers, "x", "y");
		let graph = oa::ml::OnnxGraph::from_bytes_on(&engine, &model)?;
		assert_eq!(graph.input_name(), "x");
		assert_eq!(graph.output_name(), "y");
		assert_eq!(graph.num_nodes(), 5);

		let input_values = vec![
			0.5_f32, -0.3, 0.8, 0.1, //
			-0.7, 0.4, -0.2, 0.9, //
			0.2, 0.6, -0.5, -0.4,
		];
		let input = oa::Matrix::new_from_f32([3, 4], &input_values)?;
		let logits = graph.forward(&input)?;
		assert_eq!(logits.shape(), [3, 3]);

		// Host reference: same MLP in plain arithmetic.
		let mut expected = [0.0_f32; 9];
		for row in 0..3 {
			for hidden in 0..8 {
				let mut acc = b1[hidden];
				for column in 0..4 {
					acc += input_values[row * 4 + column] * w1[column * 8 + hidden];
				}
				let activated = acc.max(0.0);
				for out in 0..3 {
					expected[row * 3 + out] += activated * w2[hidden * 3 + out];
				}
			}
			for out in 0..3 {
				expected[row * 3 + out] += b2[out];
			}
		}
		let actual = logits.read_f32()?;
		for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
			assert!(
				(actual - expected).abs() <= 1.0e-5,
				"logit {index}: found {actual}, expected {expected}"
			);
		}
		Ok(())
	}
);

test_vk!(onnx_gemm_and_softmax_nodes_evaluate_in_order, engine, {
	// Gemm(x, W[2,3] transB=0) + bias[3] -> Softmax(axis=-1).
	let w = vec![0.3, -0.1, 0.2, -0.2, 0.4, 0.1];
	let bias = vec![0.1, 0.0, -0.1];
	let nodes = vec![
		onnx_node(
			"Gemm",
			&["x", "w", "bias"],
			"g",
			&[onnx_attribute_i("transB", 0)],
		),
		onnx_node("Softmax", &["g"], "y", &[onnx_attribute_i("axis", -1)]),
	];
	let initializers = vec![
		onnx_tensor("w", &[2, 3], &w),
		onnx_tensor("bias", &[3], &bias),
	];
	let model = onnx_model(&nodes, &initializers, "x", "y");
	let graph = oa::ml::OnnxGraph::from_bytes_on(&engine, &model)?;
	let input = oa::Matrix::new_from_f32([2, 2], &[1.0, -0.5, 0.3, 0.7])?;
	let probabilities = graph.forward(&input)?.read_f32()?;
	for row in probabilities.as_chunks::<3>().0 {
		assert!((row.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
	}
	// Host reference for row 0.
	let mut logits = [0.0_f32; 3];
	for out in 0..3 {
		logits[out] = bias[out] + 1.0 * w[out] + (-0.5) * w[3 + out];
	}
	let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
	let denominator: f32 = logits.iter().map(|value| (value - max).exp()).sum();
	for out in 0..3 {
		let expected = (logits[out] - max).exp() / denominator;
		assert!((probabilities[out] - expected).abs() < 1.0e-5);
	}
	Ok(())
});

test_vk!(
	onnx_import_rejects_unsupported_operators_and_bad_dtype,
	engine,
	{
		let nodes = vec![onnx_node("Conv", &["x", "w"], "y", &[])];
		let initializers = vec![onnx_tensor("w", &[1, 1], &[0.5])];
		let model = onnx_model(&nodes, &initializers, "x", "y");
		assert_eq!(
			oa::ml::OnnxGraph::from_bytes_on(&engine, &model)
				.err()
				.expect("import accepted an unsupported operator")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		Ok(())
	}
);

test_vk!(
	transformer_named_state_roundtrip_reproduces_logits,
	engine,
	{
		use oa::ml::{Module, nn::Transformer};

		let directory = std::env::temp_dir().join(format!(
			"oars-state-dict-{}-{}",
			std::process::id(),
			SystemTime::now()
				.duration_since(SystemTime::UNIX_EPOCH)
				.expect("system clock predates Unix epoch")
				.as_nanos()
		));
		std::fs::create_dir_all(&directory).expect("create state-dict test directory");
		let path = directory.join("transformer.oam");

		let source = Transformer::new_with_seed(24, 6, 8, 16, 2, 2, 1.0e-5, 0x57A7_E5D1)?;
		let source_optimizer = oa::ml::AdamW::new(source.all_parameters()?, 0.001)?;
		let tokens = oa::Matrix::new_from_slice([1, 6], &[1_u32, 5, 9, 13, 17, 21])?;
		let expected = source.forward(&tokens)?.read_f32()?;
		oa::ml::save_checkpoint(&path, &source, &source_optimizer)?;

		let destination = Transformer::new_with_seed(24, 6, 8, 16, 2, 2, 1.0e-5, 0xDE5E_C0DE)?;
		assert_ne!(
			destination.forward(&tokens)?.read_f32()?,
			expected,
			"destination weights must differ before the state-dict load"
		);
		let mut destination_optimizer = oa::ml::AdamW::new(destination.all_parameters()?, 0.001)?;
		oa::ml::load_checkpoint(&path, &destination, &mut destination_optimizer)?;
		let restored = destination.forward(&tokens)?.read_f32()?;
		for (index, (restored, expected)) in restored.iter().zip(expected.iter()).enumerate() {
			assert!(
				(restored - expected).abs() <= 1.0e-6,
				"logit {index}: restored {restored}, expected {expected}"
			);
		}
		Ok(())
	}
);
