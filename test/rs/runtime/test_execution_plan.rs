use oa::ml::Module;

test_vk!(captures_and_replays_an_immutable_plan, engine, {
	let one = oa::matrix::ones(&engine, [2, 3])?;
	let two = oa::matrix::full(&engine, [2, 3], 2.0)?;
	let (plan, six) = engine.capture(|| {
		let three = oa::matrix::add(&one, &two)?;
		oa::matrix::add(&three, &three)
	})?;
	assert!(plan.is_complete()?);
	let semantic = plan.semantic_graph();
	assert_eq!(semantic.operations().len(), 2);
	assert_eq!(semantic.values().len(), 4);
	assert_eq!(semantic.operations()[0].name(), "oa::FnMatrix::add");
	assert_eq!(
		semantic.operations()[0].contract_hash(),
		oa::core::operation::matrix::ADD.hash()
	);
	assert_eq!(
		semantic.operations()[1].inputs()[0],
		semantic.operations()[0].outputs().first().copied()
	);
	let lowering = plan.semantic_lowering();
	assert_eq!(lowering.schema_owned_node_count(), 1);
	assert_eq!(lowering.compatibility_node_count(), 0);
	assert_eq!(lowering.direct_op_count(), 0);
	assert_eq!(lowering.fused_op_count(), 2);
	assert_eq!(lowering.fused_node_count(), 1);
	assert_eq!(lowering.maximum_nodes_per_op(), 1);
	let report = plan.debug_report_json("graph\n\"probe");
	assert_eq!(report, plan.debug_report_json("graph\n\"probe"));
	let report: serde_json::Value =
		serde_json::from_str(&report).expect("executable report must be valid JSON");
	assert_eq!(report["schema"], "oa.execution_graph.v3");
	assert_eq!(report["name"], "graph\n\"probe");
	assert_eq!(report["compiled"], false);
	assert_eq!(report["completion"]["submitted"], false);
	assert_eq!(report["completion"]["timeline_value"], 0);
	assert_eq!(report["stats"]["dispatches"], 1);
	assert_eq!(report["stats"]["barriers"], 0);
	assert_eq!(report["stats"]["war_barriers"], 0);
	assert_eq!(report["stats"]["descriptor_sets"], 0);
	assert_eq!(
		report["resources"]
			.as_array()
			.expect("resources must be an array")
			.len(),
		4
	);
	assert_eq!(report["resources"][0]["id"], 0);
	assert_eq!(report["resources"][0]["hazard_domain"], 0);
	assert_eq!(report["alias_groups"], serde_json::json!([]));
	assert_eq!(
		report["barriers"]
			.as_array()
			.expect("barriers must be an array")
			.len(),
		0
	);
	assert_eq!(
		report["nodes"]
			.as_array()
			.expect("nodes must be an array")
			.len(),
		1
	);
	assert_eq!(report["nodes"][0]["operation"], "matrix.fused_chain.f32");
	assert_eq!(
		report["nodes"][0]["semantic_operations"],
		serde_json::json!([0, 1])
	);
	assert_eq!(report["nodes"][0]["kernel"], "matrix.fused_chain.f32");
	assert_eq!(report["nodes"][0]["dtype_class"], serde_json::Value::Null);
	assert_eq!(report["nodes"][0]["dtype"], "float32");
	assert_eq!(report["nodes"][0]["kernel_selection"], "direct");
	assert_eq!(
		report["nodes"][0]["physical_write"]["uniform_store"]["partition"],
		"exclusive_per_invocation"
	);
	let serialized = report.to_string();
	assert!(!serialized.contains("descriptor_index"));
	assert!(!serialized.contains("device_address"));
	assert!(!serialized.contains("push_constants"));

	assert_eq!(
		six.try_read_f32().unwrap_err().kind(),
		oa::ErrorKind::NotReady
	);
	assert_eq!(
		six.read_f32().unwrap_err().kind(),
		oa::ErrorKind::FailedPrecondition
	);
	let captured_input_error = match oa::matrix::add(&six, &six) {
		Ok(_) => panic!("captured output was used before plan submission"),
		Err(error) => error,
	};
	assert_eq!(
		captured_input_error.kind(),
		oa::ErrorKind::FailedPrecondition
	);

	let first = engine.submit(&plan)?;
	drop(first);
	plan.wait()?;
	assert!(plan.is_complete()?);
	assert_eq!(six.read_f32()?, [6.0; 6]);
	let submitted: serde_json::Value = serde_json::from_str(&plan.debug_report_json("submitted"))
		.expect("submitted executable report must be valid JSON");
	assert_eq!(submitted["compiled"], true);
	assert_eq!(submitted["completion"]["submitted"], true);
	assert!(
		submitted["completion"]["timeline_value"]
			.as_u64()
			.is_some_and(|value| value != 0)
	);

	let second = engine.submit(&plan)?;
	drop(second);
	plan.wait()?;
	assert_eq!(six.read_f32()?, [6.0; 6]);
	plan.reset()?;
	assert_eq!(six.read_f32()?, [6.0; 6]);
	Ok(())
});

test_vk!(fused_chain_materializes_observed_intermediates, engine, {
	let one = oa::matrix::ones(&engine, [4])?;
	let two = oa::matrix::full(&engine, [4], 2.0)?;
	let (plan, (three, six)) = engine.capture(|| {
		let three = oa::matrix::add(&one, &two)?;
		let six = oa::matrix::add(&three, &three)?;
		Ok((three, six))
	})?;
	let lowering = plan.semantic_lowering();
	assert_eq!(lowering.fused_op_count(), 2);
	assert_eq!(lowering.fused_node_count(), 1);
	let submitted = engine.submit(&plan)?;
	drop(submitted);
	plan.wait()?;
	assert_eq!(three.read_f32()?, [3.0; 4]);
	assert_eq!(six.read_f32()?, [6.0; 4]);
	Ok(())
});

test_vk!(epilogue_fuses_linear_residual_pair, engine, {
	let input = oa::Matrix::new_from_f32(&engine, [2, 3], &[1.0, -2.0, 0.5, -1.5, 2.0, 0.25])?;
	let weight = oa::Matrix::new_from_f32(
		&engine,
		[4, 3],
		&[
			0.5, -1.0, 2.0, //
			1.0, 2.0, -1.5, //
			-0.25, 0.5, -2.0, //
			1.75, -0.75, 0.25,
		],
	)?;
	let bias = oa::Matrix::new_from_f32(&engine, [4], &[0.5, -1.0, 2.0, -0.25])?;
	let residual = oa::Matrix::new_from_f32(
		&engine,
		[2, 4],
		&[1.0, 0.0, -1.0, 2.0, -0.5, 1.5, 0.25, -2.0],
	)?;
	let linear = oa::ml::nn::Linear::new_from_matrices(weight, bias)?;
	let reference = {
		let pre_activation = linear.forward(&input)?;
		oa::matrix::add(&pre_activation, &residual)?.read_f32()?
	};
	let (plan, output) = engine.capture(|| {
		let pre_activation = linear.forward(&input)?;
		oa::matrix::add(&pre_activation, &residual)
	})?;

	let lowering = plan.semantic_lowering();
	assert_eq!(lowering.fused_op_count(), 2);
	assert_eq!(lowering.fused_node_count(), 1);
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	assert_eq!(report["nodes"].as_array().expect("nodes").len(), 1);
	assert_eq!(report["nodes"][0]["kernel"], "ml.linear_residual.f32");

	engine.submit(&plan)?.wait()?;
	assert_eq!(output.read_f32()?, reference);
	Ok(())
});

test_vk!(epilogue_fuses_linear_activation_pair, engine, {
	let input = oa::Matrix::new_from_f32(&engine, [2, 3], &[1.0, -2.0, 0.5, -1.5, 2.0, 0.25])?;
	let weight = oa::Matrix::new_from_f32(
		&engine,
		[4, 3],
		&[
			0.5, -1.0, 2.0, //
			1.0, 2.0, -1.5, //
			-0.25, 0.5, -2.0, //
			1.75, -0.75, 0.25,
		],
	)?;
	let bias = oa::Matrix::new_from_f32(&engine, [4], &[0.5, -1.0, 2.0, -0.25])?;
	let linear = oa::ml::nn::Linear::new_from_matrices(weight, bias)?;
	let reference = {
		let pre_activation = linear.forward(&input)?;
		oa::ml::matrix::gelu(&pre_activation)?.read_f32()?
	};
	let (plan, output) = engine.capture(|| {
		let pre_activation = linear.forward(&input)?;
		oa::ml::matrix::gelu(&pre_activation)
	})?;

	let lowering = plan.semantic_lowering();
	assert_eq!(lowering.fused_op_count(), 2);
	assert_eq!(lowering.fused_node_count(), 1);
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	assert_eq!(report["nodes"].as_array().expect("nodes").len(), 1);
	assert_eq!(report["nodes"][0]["kernel"], "ml.linear_act.f32");

	engine.submit(&plan)?.wait()?;
	assert_eq!(output.read_f32()?, reference);
	Ok(())
});

test_vk!(epilogue_fuses_linear_backward_prologue_pair, engine, {
	let input_values: Vec<f32> = (0..12)
		.map(|i| ((i * 13 % 19) as f32 - 9.0) / 7.0)
		.collect();
	let target_values: Vec<f32> = (0..21)
		.map(|i| ((i * 11 % 13) as f32 - 6.0) / 8.0)
		.collect();
	for activate in [
		oa::ml::matrix::gelu as fn(&oa::Matrix) -> oa::Result<oa::Matrix>,
		oa::ml::matrix::relu,
		oa::ml::matrix::silu,
	] {
		let input = oa::Matrix::new_from_f32(&engine, [3, 4], &input_values)?;
		let target = oa::Matrix::new_from_f32(&engine, [3, 7], &target_values)?;
		// linear_up → linear_fused → act → linear_out → mse. In backward the
		// act adjoint feeds linear_fused's data adjoint, fusing into
		// linear_backward_prologue; its dX output feeds linear_up's backward, so
		// upstream parameter gradients cover both fused outputs transitively.
		let layers = |seed_offset: u64| -> oa::Result<Vec<_>> {
			Ok(vec![
				oa::ml::nn::Linear::new_with_seed(&engine, 4, 5, 0x984 + seed_offset)?,
				oa::ml::nn::Linear::new_with_seed(&engine, 5, 6, 0x985 + seed_offset)?,
				oa::ml::nn::Linear::new_with_seed(&engine, 6, 7, 0x986 + seed_offset)?,
			])
		};
		let reference = layers(0)?;
		let candidate = layers(0)?;
		// The unfused reference runs the same eager graph with epilogue
		// fusion disabled, so dX, dW and db come from the per-op kernels.
		unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
		let expected = {
			let tape = oa::ml::GradientTape::new();
			let hidden = reference[1].forward(&reference[0].forward(&input)?)?;
			let output = reference[2].forward(&activate(&hidden)?)?;
			let loss = oa::ml::loss::mse(&output, &target)?;
			tape.backward(&loss)?;
			reference
				.iter()
				.flat_map(|layer| layer.all_parameters().unwrap())
				.map(|p| p.gradient().unwrap().read_f32())
				.collect::<oa::Result<Vec<_>>>()?
		};
		unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
		let (plan, _) = engine.capture(|| {
			let tape = oa::ml::GradientTape::new();
			let hidden = candidate[1].forward(&candidate[0].forward(&input)?)?;
			let output = candidate[2].forward(&activate(&hidden)?)?;
			let loss = oa::ml::loss::mse(&output, &target)?;
			tape.backward(&loss)?;
			Ok(())
		})?;
		let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
			.expect("fused epilogue executable report must be valid JSON");
		assert!(
			report["nodes"]
				.as_array()
				.expect("nodes")
				.iter()
				.any(|node| node["kernel"]
					.as_str()
					.unwrap()
					.ends_with("linear_backward_prologue.f32")),
			"backward adjoint must fuse into linear_backward_prologue.f32"
		);
		engine.submit(&plan)?.wait()?;
		let actual = candidate
			.iter()
			.flat_map(|layer| layer.all_parameters().unwrap())
			.map(|p| p.gradient().unwrap().read_f32())
			.collect::<oa::Result<Vec<_>>>()?;
		for (actual, expected) in actual.iter().zip(&expected) {
			assert_eq!(actual, expected, "fused backward must match unfused");
		}
	}
	Ok(())
});

test_vk!(epilogue_fuses_linear_backward_silu_mul_pair, engine, {
	// linear_up → linear_fused(produces packed gate|up) → silu_mul →
	// linear_out → mse: the gated adjoint feeds the data adjoint of the
	// gate|up projection and fuses into linear_backward_prologue kind=9.
	let input_values: Vec<f32> = (0..12)
		.map(|i| ((i * 13 % 19) as f32 - 9.0) / 7.0)
		.collect();
	let target_values: Vec<f32> = (0..21)
		.map(|i| ((i * 11 % 13) as f32 - 6.0) / 8.0)
		.collect();
	let input = oa::Matrix::new_from_f32(&engine, [3, 4], &input_values)?;
	let target = oa::Matrix::new_from_f32(&engine, [3, 7], &target_values)?;
	let layers = || -> oa::Result<Vec<_>> {
		Ok(vec![
			oa::ml::nn::Linear::new_with_seed(&engine, 4, 5, 0x984)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 5, 12, 0x985)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 6, 7, 0x986)?,
		])
	};
	let reference = layers()?;
	let candidate = layers()?;
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let expected = {
		let tape = oa::ml::GradientTape::new();
		let hidden = reference[1].forward(&reference[0].forward(&input)?)?;
		let output = reference[2].forward(&oa::ml::matrix::silu_mul(&hidden, 6)?)?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		reference
			.iter()
			.flat_map(|layer| layer.all_parameters().unwrap())
			.map(|p| p.gradient().unwrap().read_f32())
			.collect::<oa::Result<Vec<_>>>()?
	};
	unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let hidden = candidate[1].forward(&candidate[0].forward(&input)?)?;
		let output = candidate[2].forward(&oa::ml::matrix::silu_mul(&hidden, 6)?)?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		Ok(())
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	assert!(
		report["nodes"]
			.as_array()
			.expect("nodes")
			.iter()
			.any(|node| node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("linear_backward_prologue.f32")),
		"gated backward adjoint must fuse into linear_backward_prologue.f32"
	);
	engine.submit(&plan)?.wait()?;
	let actual = candidate
		.iter()
		.flat_map(|layer| layer.all_parameters().unwrap())
		.map(|p| p.gradient().unwrap().read_f32())
		.collect::<oa::Result<Vec<_>>>()?;
	for (actual, expected) in actual.iter().zip(&expected) {
		assert_eq!(actual, expected, "fused backward must match unfused");
	}
	Ok(())
});

test_vk!(epilogue_fuses_linear_backward_add_pair, engine, {
	// lin1's output feeds two branch projections; its backward gradient is
	// the sum of both branch data adjoints, emitted as a real matrix.add
	// that directly feeds lin1's data adjoint — a pointwise producer
	// prologue (kind 10).
	let input_values: Vec<f32> = (0..12)
		.map(|i| ((i * 13 % 19) as f32 - 9.0) / 7.0)
		.collect();
	let target_values: Vec<f32> = (0..18)
		.map(|i| ((i * 11 % 13) as f32 - 6.0) / 8.0)
		.collect();
	let input = oa::Matrix::new_from_f32(&engine, [3, 4], &input_values)?;
	let target = oa::Matrix::new_from_f32(&engine, [3, 6], &target_values)?;
	let layers = || -> oa::Result<Vec<_>> {
		Ok(vec![
			oa::ml::nn::Linear::new_with_seed(&engine, 4, 5, 0x991)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 5, 6, 0x992)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 5, 6, 0x993)?,
		])
	};
	let reference = layers()?;
	let candidate = layers()?;
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let expected = {
		let tape = oa::ml::GradientTape::new();
		let shared = reference[0].forward(&input)?;
		let first = oa::ml::loss::mse(&reference[1].forward(&shared)?, &target)?;
		let second = oa::ml::loss::mse(&reference[2].forward(&shared)?, &target)?;
		let loss = oa::matrix::add(&first, &second)?;
		tape.backward(&loss)?;
		reference
			.iter()
			.flat_map(|layer| layer.all_parameters().unwrap())
			.map(|p| p.gradient().unwrap().read_f32())
			.collect::<oa::Result<Vec<_>>>()?
	};
	unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let shared = candidate[0].forward(&input)?;
		let first = oa::ml::loss::mse(&candidate[1].forward(&shared)?, &target)?;
		let second = oa::ml::loss::mse(&candidate[2].forward(&shared)?, &target)?;
		let loss = oa::matrix::add(&first, &second)?;
		tape.backward(&loss)?;
		Ok(())
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	assert!(
		report["nodes"]
			.as_array()
			.expect("nodes")
			.iter()
			.any(|node| node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("linear_backward_prologue.f32")),
		"accumulation add must fuse into linear_backward_prologue.f32"
	);
	engine.submit(&plan)?.wait()?;
	let actual = candidate
		.iter()
		.flat_map(|layer| layer.all_parameters().unwrap())
		.map(|p| p.gradient().unwrap().read_f32())
		.collect::<oa::Result<Vec<_>>>()?;
	for (actual, expected) in actual.iter().zip(&expected) {
		assert_eq!(actual, expected, "fused backward must match unfused");
	}
	Ok(())
});

test_vk!(epilogue_fuses_linear_backward_scalar_mul_pair, engine, {
	// The cross-entropy adjoint scales its per-element gradient by the
	// scalar loss gradient through a broadcast multiply that directly feeds
	// the linear data adjoint — a scalar splat producer prologue
	// (kind 13).
	let input_values: Vec<f32> = (0..12)
		.map(|i| ((i * 13 % 19) as f32 - 9.0) / 7.0)
		.collect();
	let input = oa::Matrix::new_from_f32(&engine, [3, 4], &input_values)?;
	let target = oa::Matrix::new_from_slice(&engine, [3], &[1_u32, 3, 4])?;
	let layers = || -> oa::Result<Vec<_>> {
		Ok(vec![
			oa::ml::nn::Linear::new_with_seed(&engine, 4, 5, 0x994)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 5, 6, 0x995)?,
		])
	};
	let reference = layers()?;
	let candidate = layers()?;
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let expected = {
		let tape = oa::ml::GradientTape::new();
		let logits = reference[1].forward(&reference[0].forward(&input)?)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		reference
			.iter()
			.flat_map(|layer| layer.all_parameters().unwrap())
			.map(|p| p.gradient().unwrap().read_f32())
			.collect::<oa::Result<Vec<_>>>()?
	};
	unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let logits = candidate[1].forward(&candidate[0].forward(&input)?)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		Ok(())
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	assert!(
		report["nodes"]
			.as_array()
			.expect("nodes")
			.iter()
			.any(|node| node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("linear_backward_prologue.f32")),
		"cross-entropy scalar multiply must fuse into linear_backward_prologue.f32"
	);
	engine.submit(&plan)?.wait()?;
	let actual = candidate
		.iter()
		.flat_map(|layer| layer.all_parameters().unwrap())
		.map(|p| p.gradient().unwrap().read_f32())
		.collect::<oa::Result<Vec<_>>>()?;
	for (actual, expected) in actual.iter().zip(&expected) {
		assert_eq!(actual, expected, "fused backward must match unfused");
	}
	Ok(())
});

test_vk!(epilogue_fuses_linear_backward_norm_pair, engine, {
	// lin1 → layer_norm → lin2 → mse: the layer norm's data adjoint writes
	// the upstream operand of lin1's data adjoint with only the sibling
	// sum_axis_pair reduction between them, fusing into
	// linear_backward_norm. The fused kernel recomputes the row statistics
	// the norm adjoint needs inside the GEMM tile load and materializes both
	// the norm's input gradient and the weight contribution the surviving
	// reduction reads.
	let input_values: Vec<f32> = (0..12)
		.map(|i| ((i * 13 % 19) as f32 - 9.0) / 7.0)
		.collect();
	let target_values: Vec<f32> = (0..18)
		.map(|i| ((i * 11 % 13) as f32 - 6.0) / 8.0)
		.collect();
	let input = oa::Matrix::new_from_f32(&engine, [3, 4], &input_values)?;
	let target = oa::Matrix::new_from_f32(&engine, [3, 6], &target_values)?;
	let layers = || -> oa::Result<(
		oa::ml::nn::Linear,
		oa::ml::nn::LayerNorm,
		oa::ml::nn::Linear,
	)> {
		Ok((
			oa::ml::nn::Linear::new_with_seed(&engine, 4, 5, 0x996)?,
			oa::ml::nn::LayerNorm::new(&engine, 5, 1e-5)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 5, 6, 0x997)?,
		))
	};
	let reference = layers()?;
	let candidate = layers()?;
	let parameters = |layers: &(
		oa::ml::nn::Linear,
		oa::ml::nn::LayerNorm,
		oa::ml::nn::Linear,
	)|
	 -> oa::Result<Vec<Vec<f32>>> {
		layers
			.0
			.all_parameters()
			.unwrap()
			.iter()
			.chain(layers.1.all_parameters().unwrap().iter())
			.chain(layers.2.all_parameters().unwrap().iter())
			.map(|p| p.gradient().unwrap().read_f32())
			.collect()
	};
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let expected = {
		let tape = oa::ml::GradientTape::new();
		let normalized = reference.1.forward(&reference.0.forward(&input)?)?;
		let output = reference.2.forward(&normalized)?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		parameters(&reference)?
	};
	unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let normalized = candidate.1.forward(&candidate.0.forward(&input)?)?;
		let output = candidate.2.forward(&normalized)?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		Ok(())
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	assert!(
		report["nodes"]
			.as_array()
			.expect("nodes")
			.iter()
			.any(|node| node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("linear_backward_norm.f32")),
		"layer norm data adjoint must fuse into linear_backward_norm.f32"
	);
	engine.submit(&plan)?.wait()?;
	let actual = parameters(&candidate)?;
	// The fused kernel's row statistics reduce in a different order than the
	// unfused 256-lane tree, so gradients agree within float tolerance
	// rather than bitwise.
	for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
		assert!(
			(actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs(),
			"fused backward must match unfused: {actual} vs {expected}"
		);
	}
	Ok(())
});

test_vk!(epilogue_fuses_linear_backward_bmm_chain_pair, engine, {
	// embed → {q,k,v} projections → scaled_dot_product_attention → proj →
	// mse: the attention backward emits bmm producers for dQ (NN layout),
	// dK and dV (TN layout) whose outputs feed the three projection data
	// adjoints at distances 7, 4 and 1 — the parameter adjoints and sibling
	// bmms sit between each pair. Every pair fuses into a dual-output
	// linear_backward_bmm_chain dispatch that republishes the bmm result
	// for its surviving parameter adjoint, keeping the donor k-tile and
	// feature accumulation orders bit-identical.
	let input_values: Vec<f32> = (0..512)
		.map(|i| ((i * 13 % 29) as f32 - 14.0) / 11.0)
		.collect();
	let target_values: Vec<f32> = (0..512)
		.map(|i| ((i * 11 % 23) as f32 - 11.0) / 9.0)
		.collect();
	let input = oa::Matrix::new_from_f32(&engine, [32, 16], &input_values)?;
	let target = oa::Matrix::new_from_f32(&engine, [32, 16], &target_values)?;
	let layers = || -> oa::Result<Vec<oa::ml::nn::Linear>> {
		Ok(vec![
			oa::ml::nn::Linear::new_with_seed(&engine, 16, 16, 0x9a1)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 16, 16, 0x9a2)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 16, 16, 0x9a3)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 16, 16, 0x9a4)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 16, 16, 0x9a5)?,
		])
	};
	let forward = |layers: &[oa::ml::nn::Linear]| -> oa::Result<oa::Matrix> {
		let shared = layers[0].forward(&input)?;
		let query = oa::ml::matrix::split_heads(&layers[1].forward(&shared)?, 2, 16, 1)?;
		let key = oa::ml::matrix::split_heads(&layers[2].forward(&shared)?, 2, 16, 1)?;
		let value = oa::ml::matrix::split_heads(&layers[3].forward(&shared)?, 2, 16, 1)?;
		let attended =
			oa::ml::matrix::scaled_dot_product_attention(&query, &key, &value, None, 0.25, true)?;
		let merged = oa::ml::matrix::merge_heads(&attended, 2, 16, 1)?;
		layers[4].forward(&merged)
	};
	let parameters = |layers: &[oa::ml::nn::Linear]| -> oa::Result<Vec<Vec<f32>>> {
		layers
			.iter()
			.flat_map(|layer| layer.all_parameters().unwrap())
			.map(|p| p.gradient().unwrap().read_f32())
			.collect()
	};
	let reference = layers()?;
	let candidate = layers()?;
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let expected = {
		let tape = oa::ml::GradientTape::new();
		let output = forward(&reference)?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		parameters(&reference)?
	};
	unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let output = forward(&candidate)?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		Ok(())
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	let chained = report["nodes"]
		.as_array()
		.expect("nodes")
		.iter()
		.filter(|node| {
			node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("linear_backward_bmm_chain.f32")
		})
		.count();
	assert_eq!(
		chained, 3,
		"all three projection data adjoints must fuse into linear_backward_bmm_chain.f32"
	);
	engine.submit(&plan)?.wait()?;
	let actual = parameters(&candidate)?;
	// The fused attention route (`small_causal_sdpa`) serializes the
	// per-row softmax where the unfused softmax tree-reduces, so
	// gradients carry ulp-scale accumulation-order differences on this
	// driver; the chained adjoints themselves keep the donor orders.
	for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
		assert!(
			(actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs(),
			"fused backward must match unfused: {actual} vs {expected}"
		);
	}
	Ok(())
});

test_vk!(epilogue_fuses_linear_backward_bmm_chain_tails, engine, {
	// Same chain pattern with odd geometry — batch 3, sequence 10, width 20
	// leaves partial row bands, a partial k tile and 20 of 32 shared
	// columns live; zero padding must reproduce the donor kernels' tails
	// bit-identically.
	let input_values: Vec<f32> = (0..600)
		.map(|i| ((i * 13 % 29) as f32 - 14.0) / 11.0)
		.collect();
	let target_values: Vec<f32> = (0..600)
		.map(|i| ((i * 11 % 23) as f32 - 11.0) / 9.0)
		.collect();
	let input = oa::Matrix::new_from_f32(&engine, [30, 20], &input_values)?;
	let target = oa::Matrix::new_from_f32(&engine, [30, 20], &target_values)?;
	let layers = || -> oa::Result<Vec<oa::ml::nn::Linear>> {
		Ok(vec![
			oa::ml::nn::Linear::new_with_seed(&engine, 20, 20, 0x9b1)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 20, 20, 0x9b2)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 20, 20, 0x9b3)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 20, 20, 0x9b4)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 20, 20, 0x9b5)?,
		])
	};
	let forward = |layers: &[oa::ml::nn::Linear]| -> oa::Result<oa::Matrix> {
		let shared = layers[0].forward(&input)?;
		let query = oa::ml::matrix::split_heads(&layers[1].forward(&shared)?, 3, 10, 1)?;
		let key = oa::ml::matrix::split_heads(&layers[2].forward(&shared)?, 3, 10, 1)?;
		let value = oa::ml::matrix::split_heads(&layers[3].forward(&shared)?, 3, 10, 1)?;
		let attended =
			oa::ml::matrix::scaled_dot_product_attention(&query, &key, &value, None, 0.25, true)?;
		let merged = oa::ml::matrix::merge_heads(&attended, 3, 10, 1)?;
		layers[4].forward(&merged)
	};
	let parameters = |layers: &[oa::ml::nn::Linear]| -> oa::Result<Vec<Vec<f32>>> {
		layers
			.iter()
			.flat_map(|layer| layer.all_parameters().unwrap())
			.map(|p| p.gradient().unwrap().read_f32())
			.collect()
	};
	let reference = layers()?;
	let candidate = layers()?;
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let expected = {
		let tape = oa::ml::GradientTape::new();
		let output = forward(&reference)?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		parameters(&reference)?
	};
	unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let output = forward(&candidate)?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		Ok(())
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	let chained = report["nodes"]
		.as_array()
		.expect("nodes")
		.iter()
		.filter(|node| {
			node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("linear_backward_bmm_chain.f32")
		})
		.count();
	assert_eq!(
		chained, 3,
		"odd-geometry adjoints must still fuse into linear_backward_bmm_chain.f32"
	);
	engine.submit(&plan)?.wait()?;
	let actual = parameters(&candidate)?;
	// The fused attention route (`small_causal_sdpa`) serializes the
	// per-row softmax where the unfused softmax tree-reduces, so
	// gradients carry ulp-scale accumulation-order differences on this
	// driver; the chained adjoints themselves keep the donor orders.
	for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
		assert!(
			(actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs(),
			"fused backward must match unfused: {actual} vs {expected}"
		);
	}
	Ok(())
});

test_vk!(epilogue_fuses_linear_backward_norm_add_pair, engine, {
	// lin1 → h, then h feeds both ln → lin2 and lin3 directly: h's
	// gradient accumulates the norm input adjoint and lin3's data
	// adjoint through a real matrix.add that feeds lin1's data adjoint —
	// the [layer_norm_backward, sum_axis_pair, add, linear_backward]
	// window fuses into linear_backward_norm_add while the surviving
	// sum_axis_pair keeps its unfused operands.
	let input_values: Vec<f32> = (0..20)
		.map(|i| ((i * 13 % 19) as f32 - 9.0) / 7.0)
		.collect();
	let target_values: Vec<f32> = (0..28)
		.map(|i| ((i * 11 % 13) as f32 - 6.0) / 8.0)
		.collect();
	let input = oa::Matrix::new_from_f32(&engine, [4, 5], &input_values)?;
	let target = oa::Matrix::new_from_f32(&engine, [4, 7], &target_values)?;
	let layers = || -> oa::Result<(
		oa::ml::nn::Linear,
		oa::ml::nn::LayerNorm,
		oa::ml::nn::Linear,
		oa::ml::nn::Linear,
	)> {
		Ok((
			oa::ml::nn::Linear::new_with_seed(&engine, 5, 5, 0x9c1)?,
			oa::ml::nn::LayerNorm::new(&engine, 5, 1e-5)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 5, 7, 0x9c2)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 5, 7, 0x9c3)?,
		))
	};
	let reference = layers()?;
	let candidate = layers()?;
	let parameters = |layers: &(
		oa::ml::nn::Linear,
		oa::ml::nn::LayerNorm,
		oa::ml::nn::Linear,
		oa::ml::nn::Linear,
	)|
	 -> oa::Result<Vec<Vec<f32>>> {
		layers
			.0
			.all_parameters()
			.unwrap()
			.iter()
			.chain(layers.1.all_parameters().unwrap().iter())
			.chain(layers.2.all_parameters().unwrap().iter())
			.chain(layers.3.all_parameters().unwrap().iter())
			.map(|p| p.gradient().unwrap().read_f32())
			.collect()
	};
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let expected = {
		let tape = oa::ml::GradientTape::new();
		let shared = reference.0.forward(&input)?;
		let first = oa::ml::loss::mse(
			&reference.2.forward(&reference.1.forward(&shared)?)?,
			&target,
		)?;
		let second = oa::ml::loss::mse(&reference.3.forward(&shared)?, &target)?;
		let loss = oa::matrix::add(&first, &second)?;
		tape.backward(&loss)?;
		parameters(&reference)?
	};
	unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let shared = candidate.0.forward(&input)?;
		let first = oa::ml::loss::mse(
			&candidate.2.forward(&candidate.1.forward(&shared)?)?,
			&target,
		)?;
		let second = oa::ml::loss::mse(&candidate.3.forward(&shared)?, &target)?;
		let loss = oa::matrix::add(&first, &second)?;
		tape.backward(&loss)?;
		Ok(())
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	assert!(
		report["nodes"]
			.as_array()
			.expect("nodes")
			.iter()
			.any(|node| node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("linear_backward_norm_add.f32")),
		"norm accumulation window must fuse into linear_backward_norm_add.f32"
	);
	assert!(
		report["nodes"]
			.as_array()
			.expect("nodes")
			.iter()
			.any(|node| node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("sum_axis_pair.f32")),
		"the sibling dγ/dβ reduction must survive the fusion"
	);
	engine.submit(&plan)?.wait()?;
	let actual = parameters(&candidate)?;
	// The fused kernel's row statistics reduce in a different order than
	// the unfused 256-lane tree, so gradients agree within float
	// tolerance rather than bitwise.
	for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
		assert!(
			(actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs(),
			"fused backward must match unfused: {actual} vs {expected}"
		);
	}
	Ok(())
});

test_vk!(epilogue_fuses_layer_norm_backward_sum_chain, engine, {
	// A layer_norm feeding three linears — the QKV projection shape —
	// accumulates three data adjoints into the norm's dY through
	// add(add(dXq, dXk), dXv): the whole window fuses into
	// layer_norm_backward_sum at the inner add's slot while the
	// surviving sum_axis_pair keeps reading the republished dY.
	let input_values: Vec<f32> = (0..20)
		.map(|i| ((i * 13 % 19) as f32 - 9.0) / 7.0)
		.collect();
	let target_values: Vec<f32> = (0..28)
		.map(|i| ((i * 11 % 13) as f32 - 6.0) / 8.0)
		.collect();
	let input = oa::Matrix::new_from_f32(&engine, [4, 5], &input_values)?;
	let target = oa::Matrix::new_from_f32(&engine, [4, 7], &target_values)?;
	let layers = || -> oa::Result<(
		oa::ml::nn::LayerNorm,
		oa::ml::nn::Linear,
		oa::ml::nn::Linear,
		oa::ml::nn::Linear,
	)> {
		Ok((
			oa::ml::nn::LayerNorm::new(&engine, 5, 1e-5)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 5, 7, 0x9d1)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 5, 7, 0x9d2)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 5, 7, 0x9d3)?,
		))
	};
	let reference = layers()?;
	let candidate = layers()?;
	let parameters = |layers: &(
		oa::ml::nn::LayerNorm,
		oa::ml::nn::Linear,
		oa::ml::nn::Linear,
		oa::ml::nn::Linear,
	)|
	 -> oa::Result<Vec<Vec<f32>>> {
		layers
			.0
			.all_parameters()
			.unwrap()
			.iter()
			.chain(layers.1.all_parameters().unwrap().iter())
			.chain(layers.2.all_parameters().unwrap().iter())
			.chain(layers.3.all_parameters().unwrap().iter())
			.map(|p| p.gradient().unwrap().read_f32())
			.collect()
	};
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let expected = {
		let tape = oa::ml::GradientTape::new();
		let shared = reference.0.forward(&input)?;
		let loss = oa::matrix::add(
			&oa::matrix::add(
				&oa::ml::loss::mse(&reference.1.forward(&shared)?, &target)?,
				&oa::ml::loss::mse(&reference.2.forward(&shared)?, &target)?,
			)?,
			&oa::ml::loss::mse(&reference.3.forward(&shared)?, &target)?,
		)?;
		tape.backward(&loss)?;
		parameters(&reference)?
	};
	unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let shared = candidate.0.forward(&input)?;
		let loss = oa::matrix::add(
			&oa::matrix::add(
				&oa::ml::loss::mse(&candidate.1.forward(&shared)?, &target)?,
				&oa::ml::loss::mse(&candidate.2.forward(&shared)?, &target)?,
			)?,
			&oa::ml::loss::mse(&candidate.3.forward(&shared)?, &target)?,
		)?;
		tape.backward(&loss)?;
		Ok(())
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	assert!(
		report["nodes"]
			.as_array()
			.expect("nodes")
			.iter()
			.any(|node| node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("layer_norm_backward_sum.f32")),
		"the three-addend dY window must fuse into layer_norm_backward_sum.f32"
	);
	assert!(
		report["nodes"]
			.as_array()
			.expect("nodes")
			.iter()
			.any(|node| node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("sum_axis_pair.f32")),
		"the sibling dγ/dβ reduction must survive the fusion"
	);
	assert!(
		!report["nodes"]
			.as_array()
			.expect("nodes")
			.iter()
			.any(|node| node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("matrix.layer_norm_backward.f32")),
		"no unfused layer_norm_backward dispatch may remain"
	);
	engine.submit(&plan)?.wait()?;
	let actual = parameters(&candidate)?;
	// The fused kernel's row statistics reduce in a different order than
	// the unfused 256-lane tree, so gradients agree within float
	// tolerance rather than bitwise.
	for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
		assert!(
			(actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs(),
			"fused backward must match unfused: {actual} vs {expected}"
		);
	}
	Ok(())
});

test_vk!(epilogue_fuses_layer_norm_backward_sum_pair, engine, {
	// A layer_norm feeding two linears accumulates dX1 + dX2 into the
	// norm's dY through one matrix.add: the two-addend variant of the
	// same window fuses into layer_norm_backward_sum.
	let input_values: Vec<f32> = (0..20)
		.map(|i| ((i * 13 % 19) as f32 - 9.0) / 7.0)
		.collect();
	let target_values: Vec<f32> = (0..28)
		.map(|i| ((i * 11 % 13) as f32 - 6.0) / 8.0)
		.collect();
	let input = oa::Matrix::new_from_f32(&engine, [4, 5], &input_values)?;
	let target = oa::Matrix::new_from_f32(&engine, [4, 7], &target_values)?;
	let layers = || -> oa::Result<(
		oa::ml::nn::LayerNorm,
		oa::ml::nn::Linear,
		oa::ml::nn::Linear,
	)> {
		Ok((
			oa::ml::nn::LayerNorm::new(&engine, 5, 1e-5)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 5, 7, 0x9e1)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 5, 7, 0x9e2)?,
		))
	};
	let reference = layers()?;
	let candidate = layers()?;
	let parameters = |layers: &(
		oa::ml::nn::LayerNorm,
		oa::ml::nn::Linear,
		oa::ml::nn::Linear,
	)|
	 -> oa::Result<Vec<Vec<f32>>> {
		layers
			.0
			.all_parameters()
			.unwrap()
			.iter()
			.chain(layers.1.all_parameters().unwrap().iter())
			.chain(layers.2.all_parameters().unwrap().iter())
			.map(|p| p.gradient().unwrap().read_f32())
			.collect()
	};
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let expected = {
		let tape = oa::ml::GradientTape::new();
		let shared = reference.0.forward(&input)?;
		let loss = oa::matrix::add(
			&oa::ml::loss::mse(&reference.1.forward(&shared)?, &target)?,
			&oa::ml::loss::mse(&reference.2.forward(&shared)?, &target)?,
		)?;
		tape.backward(&loss)?;
		parameters(&reference)?
	};
	unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let shared = candidate.0.forward(&input)?;
		let loss = oa::matrix::add(
			&oa::ml::loss::mse(&candidate.1.forward(&shared)?, &target)?,
			&oa::ml::loss::mse(&candidate.2.forward(&shared)?, &target)?,
		)?;
		tape.backward(&loss)?;
		Ok(())
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	assert!(
		report["nodes"]
			.as_array()
			.expect("nodes")
			.iter()
			.any(|node| node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("layer_norm_backward_sum.f32")),
		"the two-addend dY window must fuse into layer_norm_backward_sum.f32"
	);
	engine.submit(&plan)?.wait()?;
	let actual = parameters(&candidate)?;
	for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
		assert!(
			(actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs(),
			"fused backward must match unfused: {actual} vs {expected}"
		);
	}
	Ok(())
});

test_vk!(epilogue_fuses_grouped_backward_prologue_pair, engine, {
	// grouped_linear_m → silu_mul → grouped_linear_m on tape: the
	// [silu_mul_backward → grouped data adjoint] pair fuses into
	// grouped_backward_prologue while the grouped parameter adjoint keeps
	// its republished upstream operand. 120 rows over three experts with
	// offsets [0, 40, 80, 120] places two 32-row bands inside one expert
	// and two across expert boundaries, covering the tiled and the
	// mixed-expert fallback paths of the fused kernel.
	const ROWS: usize = 120;
	const WIDTH: usize = 2;
	const HIDDEN: usize = 2;
	const OUTPUT: usize = 3;
	let pseudo = |index: usize, salt: usize| {
		((index.wrapping_mul(31).wrapping_add(salt * 17) % 43) as f32 - 21.0) / 11.0
	};
	let input_values: Vec<f32> = (0..ROWS * WIDTH).map(|i| pseudo(i, 1)).collect();
	let gate_up_weight: Vec<f32> = (0..3 * 2 * HIDDEN * WIDTH)
		.map(|i| pseudo(i, 2) * 0.4)
		.collect();
	let gate_up_bias: Vec<f32> = (0..3 * 2 * HIDDEN).map(|i| pseudo(i, 3) * 0.1).collect();
	let down_weight: Vec<f32> = (0..3 * OUTPUT * HIDDEN)
		.map(|i| pseudo(i, 4) * 0.4)
		.collect();
	let down_bias: Vec<f32> = (0..3 * OUTPUT).map(|i| pseudo(i, 5) * 0.1).collect();
	let target_values: Vec<f32> = (0..ROWS * OUTPUT).map(|i| pseudo(i, 6)).collect();
	let offsets_values = [0_u32, 40, 80, 120];
	let target = oa::Matrix::new_from_f32(&engine, [ROWS, OUTPUT], &target_values)?;
	let offsets = oa::Matrix::new_from_slice(&engine, [4], &offsets_values)?;
	let expert_ids = oa::Matrix::new_from_slice(&engine, [3], &[0_u32, 1, 2])?;
	let input = oa::Matrix::new_from_f32(&engine, [ROWS, WIDTH], &input_values)?;
	let build = || -> oa::Result<(
		oa::ml::nn::Embedding,
		oa::ml::nn::Embedding,
		oa::ml::nn::Embedding,
		oa::ml::nn::Embedding,
	)> {
		Ok((
			oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32(
				&engine,
				[3, 2 * HIDDEN * WIDTH],
				&gate_up_weight,
			)?)?,
			oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32(
				&engine,
				[3, 2 * HIDDEN],
				&gate_up_bias,
			)?)?,
			oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32(
				&engine,
				[3, OUTPUT * HIDDEN],
				&down_weight,
			)?)?,
			oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32(
				&engine,
				[3, OUTPUT],
				&down_bias,
			)?)?,
		))
	};
	let forward = |weights: &(
		oa::ml::nn::Embedding,
		oa::ml::nn::Embedding,
		oa::ml::nn::Embedding,
		oa::ml::nn::Embedding,
	),
	               input: &oa::Matrix|
	 -> oa::Result<oa::Matrix> {
		let gate_up_weight = weights
			.0
			.forward(&expert_ids)?
			.reshape([3, 2 * HIDDEN, WIDTH])?;
		let gate_up_bias = weights.1.forward(&expert_ids)?.reshape([3, 2 * HIDDEN])?;
		let down_weight = weights
			.2
			.forward(&expert_ids)?
			.reshape([3, OUTPUT, HIDDEN])?;
		let down_bias = weights.3.forward(&expert_ids)?.reshape([3, OUTPUT])?;
		let gate_up =
			oa::ml::matrix::grouped_linear_m(input, &gate_up_weight, &gate_up_bias, &offsets)?;
		let hidden = oa::ml::matrix::silu_mul(&gate_up, HIDDEN)?;
		oa::ml::matrix::grouped_linear_m(&hidden, &down_weight, &down_bias, &offsets)
	};
	let gradients = |weights: &(
		oa::ml::nn::Embedding,
		oa::ml::nn::Embedding,
		oa::ml::nn::Embedding,
		oa::ml::nn::Embedding,
	)|
	 -> oa::Result<Vec<Vec<f32>>> {
		[
			weights.0.weight(),
			weights.1.weight(),
			weights.2.weight(),
			weights.3.weight(),
		]
		.iter()
		.map(|parameter| parameter.gradient().unwrap().read_f32())
		.collect()
	};
	let reference = build()?;
	let candidate = build()?;
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let expected = {
		let tape = oa::ml::GradientTape::new();
		let loss = oa::ml::loss::mse(&forward(&reference, &input)?, &target)?;
		tape.backward(&loss)?;
		gradients(&reference)?
	};
	unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let loss = oa::ml::loss::mse(&forward(&candidate, &input)?, &target)?;
		tape.backward(&loss)?;
		Ok(())
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	assert!(
		report["nodes"]
			.as_array()
			.expect("nodes")
			.iter()
			.any(|node| node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("grouped_backward_prologue.f32")),
		"gated grouped adjoint must fuse into grouped_backward_prologue.f32"
	);
	assert!(
		report["nodes"]
			.as_array()
			.expect("nodes")
			.iter()
			.any(|node| node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("grouped_linear_m_parameter_backward.f32")),
		"the grouped parameter adjoint must survive the fusion"
	);
	engine.submit(&plan)?.wait()?;
	let actual = gradients(&candidate)?;
	for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
		assert_eq!(
			actual, expected,
			"fused grouped backward must match unfused"
		);
	}
	Ok(())
});

test_vk!(epilogue_fuses_residual_layer_norm_pair, engine, {
	// [512, 4]: the elementwise add launches eight workgroups while the fused
	// kernel needs one per row — a mismatched dispatch leaves 504 rows unwritten.
	let input_a = oa::Matrix::new_from_f32(
		&engine,
		[512, 4],
		&(0..2048)
			.map(|index| (index % 97) as f32 * 0.05 - 2.0)
			.collect::<Vec<_>>(),
	)?;
	let input_b = oa::Matrix::new_from_f32(
		&engine,
		[512, 4],
		&(0..2048)
			.map(|index| (index % 53) as f32 * -0.04 + 1.0)
			.collect::<Vec<_>>(),
	)?;
	let weight = oa::Matrix::new_from_f32(&engine, [4], &[1.0, -0.5, 2.0, 0.75])?;
	let bias = oa::Matrix::new_from_f32(&engine, [4], &[0.25, -1.0, 0.5, -0.125])?;
	let norm = oa::ml::nn::LayerNorm::new_from_matrices(weight, bias, 1.0e-5)?;
	let reference = {
		let summed = oa::matrix::add(&input_a, &input_b)?;
		norm.forward(&summed)?.read_f32()?
	};
	let (plan, output) = engine.capture(|| {
		let summed = oa::matrix::add(&input_a, &input_b)?;
		norm.forward(&summed)
	})?;

	let lowering = plan.semantic_lowering();
	assert_eq!(lowering.fused_op_count(), 2);
	assert_eq!(lowering.fused_node_count(), 1);
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	assert_eq!(report["nodes"].as_array().expect("nodes").len(), 1);
	assert_eq!(report["nodes"][0]["kernel"], "ml.residual_layer_norm.f32");

	engine.submit(&plan)?.wait()?;
	assert_eq!(output.read_f32()?, reference);
	Ok(())
});

test_vk!(epilogue_fuses_linear_residual_norm_chain, engine, {
	// residual + lin1 → layer_norm → lin2 → mse: the forward
	// [linear, add, layer_norm] triple that forms the transformer
	// sublayer epilogue fuses into one linear_residual_norm dispatch
	// while the backward keeps its unfused route.
	let input_values: Vec<f32> = (0..30)
		.map(|i| ((i * 13 % 17) as f32 - 8.0) / 6.0)
		.collect();
	let residual_values: Vec<f32> = (0..42)
		.map(|i| ((i * 11 % 13) as f32 - 6.0) / 5.0)
		.collect();
	let target_values: Vec<f32> = (0..24).map(|i| ((i * 7 % 11) as f32 - 5.0) / 4.0).collect();
	let input = oa::Matrix::new_from_f32(&engine, [6, 5], &input_values)?;
	let residual = oa::Matrix::new_from_f32(&engine, [6, 7], &residual_values)?;
	let target = oa::Matrix::new_from_f32(&engine, [6, 4], &target_values)?;
	let layers = || -> oa::Result<(
		oa::ml::nn::Linear,
		oa::ml::nn::LayerNorm,
		oa::ml::nn::Linear,
	)> {
		Ok((
			oa::ml::nn::Linear::new_with_seed(&engine, 5, 7, 0x8a1)?,
			oa::ml::nn::LayerNorm::new(&engine, 7, 1e-5)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 7, 4, 0x8a2)?,
		))
	};
	let reference = layers()?;
	let candidate = layers()?;
	let parameters = |layers: &(
		oa::ml::nn::Linear,
		oa::ml::nn::LayerNorm,
		oa::ml::nn::Linear,
	)|
	 -> oa::Result<Vec<Vec<f32>>> {
		layers
			.0
			.all_parameters()
			.unwrap()
			.iter()
			.chain(layers.1.all_parameters().unwrap().iter())
			.chain(layers.2.all_parameters().unwrap().iter())
			.map(|p| p.gradient().unwrap().read_f32())
			.collect()
	};
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let (expected_loss, expected) = {
		let tape = oa::ml::GradientTape::new();
		let summed = oa::matrix::add(&residual, &reference.0.forward(&input)?)?;
		let loss = oa::ml::loss::mse(
			&reference.2.forward(&reference.1.forward(&summed)?)?,
			&target,
		)?;
		let value = loss.read_f32()?;
		tape.backward(&loss)?;
		(value, parameters(&reference)?)
	};
	unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
	let (plan, loss) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let summed = oa::matrix::add(&residual, &candidate.0.forward(&input)?)?;
		let loss = oa::ml::loss::mse(
			&candidate.2.forward(&candidate.1.forward(&summed)?)?,
			&target,
		)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	assert!(
		report["nodes"]
			.as_array()
			.expect("nodes")
			.iter()
			.any(|node| node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("linear_residual_norm.f32")),
		"the [linear, add, layer_norm] window must fuse into linear_residual_norm.f32"
	);
	engine.submit(&plan)?.wait()?;
	let actual = parameters(&candidate)?;
	// The fused forward replays the donor layer_norm reduction order, but
	// exact bit parity still depends on the device: on some drivers the
	// unfused layer_norm itself produces per-lane divergent statistics, so
	// the loss agrees within float tolerance rather than bitwise.
	let actual_loss = loss.read_f32()?;
	assert_eq!(actual_loss.len(), expected_loss.len());
	for (actual, expected) in actual_loss.iter().zip(expected_loss.iter()) {
		assert!(
			(actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs(),
			"fused forward loss must match unfused: {actual} vs {expected}"
		);
	}
	// The candidate backward still runs the prologue matchers the
	// disabled reference skipped, so gradients agree within float
	// tolerance rather than bitwise.
	for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
		assert!(
			(actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs(),
			"fused backward must match unfused: {actual} vs {expected}"
		);
	}
	Ok(())
});

test_vk!(epilogue_fuses_shared_input_linear_triple, engine, {
	let input = oa::Matrix::new_from_f32(&engine, [2, 3], &[1.0, -2.0, 0.5, -1.5, 2.0, 0.25])?;
	let weights: Vec<oa::Matrix> = (0..3)
		.map(|index| {
			oa::Matrix::new_from_f32(
				&engine,
				[4, 3],
				&[
					0.5 + index as f32,
					-1.0,
					2.0, //
					1.0,
					2.0,
					-1.5, //
					-0.25,
					0.5,
					-2.0 + index as f32, //
					1.75,
					-0.75,
					0.25,
				],
			)
		})
		.collect::<oa::Result<_>>()?;
	let biases: Vec<oa::Matrix> = (0..3)
		.map(|index| oa::Matrix::new_from_f32(&engine, [4], &[0.5, -1.0 + index as f32, 2.0, -0.25]))
		.collect::<oa::Result<_>>()?;
	let projections: Vec<oa::ml::nn::Linear> = weights
		.into_iter()
		.zip(biases)
		.map(|(weight, bias)| oa::ml::nn::Linear::new_from_matrices(weight, bias))
		.collect::<oa::Result<_>>()?;
	let reference: Vec<Vec<f32>> = projections
		.iter()
		.map(|projection| projection.forward(&input)?.read_f32())
		.collect::<oa::Result<_>>()?;
	let (plan, (out_q, out_k, out_v)) = engine.capture(|| {
		Ok((
			projections[0].forward(&input)?,
			projections[1].forward(&input)?,
			projections[2].forward(&input)?,
		))
	})?;

	let lowering = plan.semantic_lowering();
	assert_eq!(lowering.fused_op_count(), 3);
	assert_eq!(lowering.fused_node_count(), 1);
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused triple executable report must be valid JSON");
	assert_eq!(report["nodes"].as_array().expect("nodes").len(), 1);
	assert_eq!(report["nodes"][0]["kernel"], "ml.linear_triple.f32");
	assert_eq!(report["nodes"][0]["groups"][2], 3);

	engine.submit(&plan)?.wait()?;
	assert_eq!(out_q.read_f32()?, reference[0]);
	assert_eq!(out_k.read_f32()?, reference[1]);
	assert_eq!(out_v.read_f32()?, reference[2]);
	Ok(())
});

test_vk!(epilogue_fuses_embedding_pair_sum, engine, {
	let tokens = oa::Matrix::new_from_slice(&engine, [4], &[0_u32, 2, 1, 3])?;
	let positions = oa::Matrix::new_from_slice(&engine, [4], &[0_u32, 1, 2, 3])?;
	let token_table = oa::Matrix::new_from_f32(
		&engine,
		[4, 3],
		&[
			0.5, -1.0, 2.0, //
			1.0, 2.0, -1.5, //
			-0.25, 0.5, -2.0, //
			1.75, -0.75, 0.25,
		],
	)?;
	let position_table = oa::Matrix::new_from_f32(
		&engine,
		[4, 3],
		&[
			1.0, 0.0, -1.0, //
			2.0, -0.5, 1.5, //
			0.25, -2.0, 0.75, //
			-1.25, 0.5, 1.0,
		],
	)?;
	let token_embedding = oa::ml::nn::Embedding::new_from_matrix(token_table)?;
	let position_embedding = oa::ml::nn::Embedding::new_from_matrix(position_table)?;
	let reference = {
		let token_values = token_embedding.forward(&tokens)?;
		let position_values = position_embedding.forward(&positions)?;
		oa::matrix::add(&token_values, &position_values)?.read_f32()?
	};
	let (plan, output) = engine.capture(|| {
		let token_values = token_embedding.forward(&tokens)?;
		let position_values = position_embedding.forward(&positions)?;
		oa::matrix::add(&token_values, &position_values)
	})?;

	let lowering = plan.semantic_lowering();
	assert_eq!(lowering.fused_node_count(), 1);
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused embedding report must be valid JSON");
	assert_eq!(report["nodes"].as_array().expect("nodes").len(), 1);
	assert_eq!(report["nodes"][0]["kernel"], "ml.embedding_pair.f32");

	engine.submit(&plan)?.wait()?;
	assert_eq!(output.read_f32()?, reference);
	Ok(())
});

test_vk!(epilogue_fuses_embedding_pair_norm_chain, engine, {
	// token + position embeddings summed then normalized: the
	// [embedding, embedding, add, layer_norm] head that opens the
	// transformer fuses into one embedding_pair_norm dispatch instead
	// of the embedding_pair kernel plus a separate layer_norm.
	let tokens = oa::Matrix::new_from_slice(&engine, [4], &[0_u32, 2, 1, 3])?;
	let positions = oa::Matrix::new_from_slice(&engine, [4], &[0_u32, 1, 2, 3])?;
	let target_values: Vec<f32> = (0..8).map(|i| ((i * 7 % 9) as f32 - 4.0) / 3.0).collect();
	let target = oa::Matrix::new_from_f32(&engine, [4, 2], &target_values)?;
	let layers = || -> oa::Result<(
		oa::ml::nn::Embedding,
		oa::ml::nn::Embedding,
		oa::ml::nn::LayerNorm,
		oa::ml::nn::Linear,
	)> {
		Ok((
			oa::ml::nn::Embedding::new_with_seed(&engine, 4, 3, 0x6f1)?,
			oa::ml::nn::Embedding::new_with_seed(&engine, 4, 3, 0x6f2)?,
			oa::ml::nn::LayerNorm::new(&engine, 3, 1e-5)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 3, 2, 0x6f3)?,
		))
	};
	let reference = layers()?;
	let candidate = layers()?;
	let parameters = |layers: &(
		oa::ml::nn::Embedding,
		oa::ml::nn::Embedding,
		oa::ml::nn::LayerNorm,
		oa::ml::nn::Linear,
	)|
	 -> oa::Result<Vec<Vec<f32>>> {
		layers
			.0
			.parameters()
			.iter()
			.chain(layers.1.parameters().iter())
			.chain(layers.2.all_parameters().unwrap().iter())
			.chain(layers.3.all_parameters().unwrap().iter())
			.map(|p| p.gradient().unwrap().read_f32())
			.collect()
	};
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let (expected_loss, expected) = {
		let tape = oa::ml::GradientTape::new();
		let token_values = reference.0.forward(&tokens)?;
		let position_values = reference.1.forward(&positions)?;
		let summed = oa::matrix::add(&token_values, &position_values)?;
		let loss = oa::ml::loss::mse(
			&reference.3.forward(&reference.2.forward(&summed)?)?,
			&target,
		)?;
		let value = loss.read_f32()?;
		tape.backward(&loss)?;
		(value, parameters(&reference)?)
	};
	unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
	let (plan, loss) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let token_values = candidate.0.forward(&tokens)?;
		let position_values = candidate.1.forward(&positions)?;
		let summed = oa::matrix::add(&token_values, &position_values)?;
		let loss = oa::ml::loss::mse(
			&candidate.3.forward(&candidate.2.forward(&summed)?)?,
			&target,
		)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	assert!(
		report["nodes"]
			.as_array()
			.expect("nodes")
			.iter()
			.any(|node| node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("embedding_pair_norm.f32")),
		"the [embedding, embedding, add, layer_norm] window must fuse into embedding_pair_norm.f32"
	);
	engine.submit(&plan)?.wait()?;
	let actual = parameters(&candidate)?;
	assert_eq!(loss.read_f32()?, expected_loss);
	// The candidate backward still runs the prologue matchers the
	// disabled reference skipped, so gradients agree within float
	// tolerance rather than bitwise.
	for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
		assert!(
			(actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs(),
			"fused backward must match unfused: {actual} vs {expected}"
		);
	}
	Ok(())
});

test_vk!(epilogue_fuses_embedding_backward_sum_pair, engine, {
	// Two embeddings summed into a head that feeds two downstream
	// consumers: the accumulated gradient of the shared sum is one
	// matrix.add read by both embedding_backward scatters. Each fused
	// consumer sums the two addends inside its position scan, so the
	// add is absorbed and its buffer never materializes — a partial
	// fuse is impossible, matching the real transformer head where the
	// token and position table adjoints share the same combined dY.
	let tokens = oa::Matrix::new_from_slice(&engine, [4], &[0_u32, 2, 1, 3])?;
	let positions = oa::Matrix::new_from_slice(&engine, [4], &[0_u32, 1, 2, 3])?;
	let target_values: Vec<f32> = (0..8).map(|i| ((i * 7 % 9) as f32 - 4.0) / 3.0).collect();
	let target = oa::Matrix::new_from_f32(&engine, [4, 2], &target_values)?;
	let layers = || -> oa::Result<(
		oa::ml::nn::Embedding,
		oa::ml::nn::Embedding,
		oa::ml::nn::Linear,
		oa::ml::nn::Linear,
	)> {
		Ok((
			oa::ml::nn::Embedding::new_with_seed(&engine, 4, 3, 0x7a1)?,
			oa::ml::nn::Embedding::new_with_seed(&engine, 4, 3, 0x7a2)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 3, 2, 0x7a3)?,
			oa::ml::nn::Linear::new_with_seed(&engine, 3, 2, 0x7a4)?,
		))
	};
	let reference = layers()?;
	let candidate = layers()?;
	let parameters = |layers: &(
		oa::ml::nn::Embedding,
		oa::ml::nn::Embedding,
		oa::ml::nn::Linear,
		oa::ml::nn::Linear,
	)|
	 -> oa::Result<Vec<Vec<f32>>> {
		layers
			.0
			.parameters()
			.iter()
			.chain(layers.1.parameters().iter())
			.chain(layers.2.all_parameters().unwrap().iter())
			.chain(layers.3.all_parameters().unwrap().iter())
			.map(|p| p.gradient().unwrap().read_f32())
			.collect()
	};
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let expected = {
		let tape = oa::ml::GradientTape::new();
		let summed = oa::matrix::add(
			&reference.0.forward(&tokens)?,
			&reference.1.forward(&positions)?,
		)?;
		let loss = oa::matrix::add(
			&oa::ml::loss::mse(&reference.2.forward(&summed)?, &target)?,
			&oa::ml::loss::mse(&reference.3.forward(&summed)?, &target)?,
		)?;
		tape.backward(&loss)?;
		parameters(&reference)?
	};
	unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let summed = oa::matrix::add(
			&candidate.0.forward(&tokens)?,
			&candidate.1.forward(&positions)?,
		)?;
		let loss = oa::matrix::add(
			&oa::ml::loss::mse(&candidate.2.forward(&summed)?, &target)?,
			&oa::ml::loss::mse(&candidate.3.forward(&summed)?, &target)?,
		)?;
		tape.backward(&loss)?;
		Ok(())
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	let fused = report["nodes"]
		.as_array()
		.expect("nodes")
		.iter()
		.filter(|node| {
			node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("embedding_backward_sum.f32")
		})
		.count();
	assert_eq!(
		fused, 2,
		"both table adjoints must fuse the accumulation prologue"
	);
	assert!(
		!report["nodes"]
			.as_array()
			.expect("nodes")
			.iter()
			.any(|node| node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("matrix.embedding_backward.f32")),
		"no unfused embedding_backward dispatch may remain"
	);
	engine.submit(&plan)?.wait()?;
	let actual = parameters(&candidate)?;
	// The fused kernel's per-position sum is evaluated elementwise in
	// the same order as the unfused add, so gradients agree within
	// float tolerance across the remaining backward fusions.
	for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
		assert!(
			(actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs(),
			"fused backward must match unfused: {actual} vs {expected}"
		);
	}
	Ok(())
});

test_vk!(epilogue_fuses_conv_1d_bias_transpose_pair, engine, {
	// Every Conv1d lowering ends with an in-place bias add over the
	// projected GEMM buffer followed by the NCL transpose. The fused
	// kernel adds bias[channel] at the tile load, so the projected
	// buffer stays raw and the biased [B, C, T] output is written in
	// one pass — bit-identical to the unfused pair.
	let input = oa::Matrix::new_from_f32(
		&engine,
		[2, 3, 12],
		&vec![0.0; 2 * 3 * 12]
			.iter()
			.enumerate()
			.map(|(i, _)| ((i * 29 % 17) as f32 - 8.0) / 7.0)
			.collect::<Vec<f32>>(),
	)?;
	let target = oa::Matrix::new_from_f32(&engine, [2, 4, 12], &vec![0.0; 2 * 4 * 12])?;
	let layers = || -> oa::Result<(oa::ml::nn::Conv1d, oa::ml::nn::Conv1d)> {
		Ok((
			oa::ml::nn::Conv1d::new_with_seed(&engine, 3, 5, 3, 1, 1, 1, 0x5c1)?,
			oa::ml::nn::Conv1d::new_with_seed(&engine, 5, 4, 3, 1, 1, 1, 0x5c2)?,
		))
	};
	let reference = layers()?;
	let candidate = layers()?;
	let gradients = |layers: &(oa::ml::nn::Conv1d, oa::ml::nn::Conv1d)| -> oa::Result<Vec<Vec<f32>>> {
		layers
			.0
			.parameters()
			.iter()
			.chain(layers.1.parameters().iter())
			.map(|p| p.gradient().unwrap().read_f32())
			.collect()
	};
	let run = |layers: &(oa::ml::nn::Conv1d, oa::ml::nn::Conv1d)| -> oa::Result<Vec<Vec<f32>>> {
		let tape = oa::ml::GradientTape::new();
		let hidden = oa::ml::matrix::relu(&layers.0.forward(&input)?)?;
		let reconstruction = layers.1.forward(&hidden)?;
		let loss = oa::ml::loss::mse(&reconstruction, &target)?;
		tape.backward(&loss)?;
		gradients(layers)
	};
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let expected = run(&reference)?;
	unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
	let (plan, _) = engine.capture(|| {
		let _ = run(&candidate);
		Ok(())
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	let nodes = report["nodes"].as_array().expect("nodes");
	// The first Conv1d feeds ReLU directly, so its tail extends to the
	// activated variant; the second Conv1d feeds the loss and keeps the
	// bias/transpose pair.
	let fused_relu = nodes
		.iter()
		.filter(|node| {
			node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("conv_1d_bias_transpose_relu.f32")
		})
		.count();
	assert_eq!(
		fused_relu, 1,
		"the Conv1d-ReLU tail must fuse all three nodes"
	);
	let fused_pair = nodes
		.iter()
		.filter(|node| {
			node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("conv_1d_bias_transpose.f32")
		})
		.count();
	assert_eq!(
		fused_pair, 1,
		"the plain Conv1d tail must fuse bias and transpose"
	);
	assert!(
		!nodes.iter().any(|node| {
			let kernel = node["kernel"].as_str().unwrap();
			kernel.ends_with("conv_1d_bias_add.f32")
				|| kernel.ends_with("conv_1d_transpose.f32")
				|| kernel.ends_with("matrix.relu.f32")
		}),
		"no unfused Conv1d bias add, transpose or ReLU may remain"
	);
	engine.submit(&plan)?.wait()?;
	let actual = gradients(&candidate)?;
	for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
		assert!(
			(actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs(),
			"fused Conv1d tail must match unfused: {actual} vs {expected}"
		);
	}
	Ok(())
});

test_vk!(
	epilogue_fuses_conv_1d_bias_transpose_residual_relu,
	engine,
	{
		// The ALM encoder's residual block feeds relu(conv(x)) + residual
		// into the next stage. The fused kernel republishes the activated
		// output the ReLU adjoint saves and writes the residual sum in the
		// same tile pass — the standalone matrix.add must disappear while
		// the activated intermediate stays bit-identical.
		let input = oa::Matrix::new_from_f32(
			&engine,
			[2, 3, 12],
			&vec![0.0; 2 * 3 * 12]
				.iter()
				.enumerate()
				.map(|(i, _)| ((i * 29 % 17) as f32 - 8.0) / 7.0)
				.collect::<Vec<f32>>(),
		)?;
		let residual = oa::Matrix::new_from_f32(
			&engine,
			[2, 5, 12],
			&vec![0.0; 2 * 5 * 12]
				.iter()
				.enumerate()
				.map(|(i, _)| ((i * 23 % 13) as f32 - 6.0) / 9.0)
				.collect::<Vec<f32>>(),
		)?;
		let target = oa::Matrix::new_from_f32(&engine, [2, 4, 12], &vec![0.0; 2 * 4 * 12])?;
		let layers = || -> oa::Result<(oa::ml::nn::Conv1d, oa::ml::nn::Conv1d)> {
			Ok((
				oa::ml::nn::Conv1d::new_with_seed(&engine, 3, 5, 3, 1, 1, 1, 0x8c1)?,
				oa::ml::nn::Conv1d::new_with_seed(&engine, 5, 4, 3, 1, 1, 1, 0x8c2)?,
			))
		};
		let reference = layers()?;
		let candidate = layers()?;
		let gradients =
			|layers: &(oa::ml::nn::Conv1d, oa::ml::nn::Conv1d)| -> oa::Result<Vec<Vec<f32>>> {
				layers
					.0
					.parameters()
					.iter()
					.chain(layers.1.parameters().iter())
					.map(|p| p.gradient().unwrap().read_f32())
					.collect()
			};
		let run = |layers: &(oa::ml::nn::Conv1d, oa::ml::nn::Conv1d)| -> oa::Result<Vec<Vec<f32>>> {
			let tape = oa::ml::GradientTape::new();
			let hidden = oa::ml::matrix::relu(&layers.0.forward(&input)?)?;
			let summed = oa::matrix::add(&hidden, &residual)?;
			let reconstruction = layers.1.forward(&summed)?;
			let loss = oa::ml::loss::mse(&reconstruction, &target)?;
			tape.backward(&loss)?;
			gradients(layers)
		};
		unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
		let expected = run(&reference)?;
		unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
		let (plan, _) = engine.capture(|| {
			let _ = run(&candidate);
			Ok(())
		})?;
		let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
			.expect("fused epilogue executable report must be valid JSON");
		let nodes = report["nodes"].as_array().expect("nodes");
		let fused = nodes
			.iter()
			.filter(|node| {
				node["kernel"]
					.as_str()
					.unwrap()
					.ends_with("conv_1d_bias_transpose_residual_relu.f32")
			})
			.count();
		assert_eq!(
			fused, 1,
			"the Conv1d-ReLU-residual tail must fuse all four nodes"
		);
		assert!(
			!nodes.iter().any(|node| {
				let kernel = node["kernel"].as_str().unwrap();
				kernel.ends_with("conv_1d_bias_add.f32")
					|| kernel.ends_with("conv_1d_transpose.f32")
					|| kernel.ends_with("matrix.relu.f32")
			}),
			"no unfused Conv1d bias add, transpose or ReLU may remain"
		);
		engine.submit(&plan)?.wait()?;
		let actual = gradients(&candidate)?;
		for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
			assert!(
				(actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs(),
				"fused Conv1d residual tail must match unfused: {actual} vs {expected}"
			);
		}
		Ok(())
	}
);

test_vk!(epilogue_fuses_conv_1d_bias_transpose_norm_relu, engine, {
	// The ALM encoder applies channel normalization with fused ReLU
	// directly after Conv1d. The fused kernel reduces over the channels
	// of the biased projected row — the same lane-strided order the
	// unfused normalization applies to the transposed output — and
	// republishes the biased conv output because the backward adjoint
	// consumes it as saved state.
	let input = oa::Matrix::new_from_f32(
		&engine,
		[2, 3, 11],
		&vec![0.0; 2 * 3 * 11]
			.iter()
			.enumerate()
			.map(|(i, _)| ((i * 31 % 19) as f32 - 9.0) / 8.0)
			.collect::<Vec<f32>>(),
	)?;
	let target = oa::Matrix::new_from_f32(&engine, [2, 5, 11], &vec![0.0; 2 * 5 * 11])?;
	let layers =
		|| -> oa::Result<(oa::ml::nn::Conv1d, oa::ml::nn::LayerNorm, oa::ml::nn::Conv1d)> {
			Ok((
				oa::ml::nn::Conv1d::new_with_seed(&engine, 3, 5, 3, 1, 1, 1, 0x6d1)?,
				oa::ml::nn::LayerNorm::new(&engine, 5, 1e-5)?,
				oa::ml::nn::Conv1d::new_with_seed(&engine, 5, 5, 3, 1, 1, 1, 0x6d2)?,
			))
		};
	let reference = layers()?;
	let candidate = layers()?;
	let gradients = |layers: &(
		oa::ml::nn::Conv1d,
		oa::ml::nn::LayerNorm,
		oa::ml::nn::Conv1d,
	)|
	 -> oa::Result<Vec<Vec<f32>>> {
		layers
			.0
			.parameters()
			.iter()
			.chain(layers.1.parameters().iter())
			.chain(layers.2.parameters().iter())
			.map(|p| p.gradient().unwrap().read_f32())
			.collect()
	};
	let run = |layers: &(
		oa::ml::nn::Conv1d,
		oa::ml::nn::LayerNorm,
		oa::ml::nn::Conv1d,
	)|
	 -> oa::Result<Vec<Vec<f32>>> {
		let tape = oa::ml::GradientTape::new();
		let hidden = layers.1.forward_channel_relu(&layers.0.forward(&input)?)?;
		let reconstruction = layers.2.forward(&hidden)?;
		let loss = oa::ml::loss::mse(&reconstruction, &target)?;
		tape.backward(&loss)?;
		gradients(layers)
	};
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let expected = run(&reference)?;
	unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
	let (plan, _) = engine.capture(|| {
		let _ = run(&candidate);
		Ok(())
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	let nodes = report["nodes"].as_array().expect("nodes");
	let fused = nodes
		.iter()
		.filter(|node| {
			node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("conv_1d_bias_transpose_norm_relu.f32")
		})
		.count();
	assert_eq!(
		fused, 1,
		"the Conv1d-norm-ReLU tail must fuse all three nodes"
	);
	assert!(
		!nodes.iter().any(|node| {
			let kernel = node["kernel"].as_str().unwrap();
			kernel.ends_with("conv_1d_bias_add.f32")
				|| kernel.ends_with("conv_1d_transpose.f32")
				|| kernel.ends_with("channel_norm_relu.f32")
		}),
		"no unfused Conv1d bias add, transpose or channel norm may remain"
	);
	engine.submit(&plan)?.wait()?;
	let actual = gradients(&candidate)?;
	// The fused norm keeps the donor's lane-strided reduction order, so
	// forward is bit-identical; backward runs the other matchers the
	// disabled reference skipped, so gradients agree within tolerance.
	for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
		assert!(
			(actual - expected).abs() <= 1e-4 + 1e-4 * expected.abs(),
			"fused Conv1d-norm tail must match unfused: {actual} vs {expected}"
		);
	}
	Ok(())
});

test_vk!(epilogue_fuses_conv_1d_bias_copy, engine, {
	// The ALM decoder transposes a Conv1d output straight back to
	// [B, T, C]: the NCL transpose followed by its exact inverse is a
	// biased copy of the projected buffer, so the fused kernel drops
	// both transposes and writes the final layout directly.
	let input = oa::Matrix::new_from_f32(
		&engine,
		[2, 3, 9],
		&vec![0.0; 2 * 3 * 9]
			.iter()
			.enumerate()
			.map(|(i, _)| ((i * 23 % 13) as f32 - 6.0) / 5.0)
			.collect::<Vec<f32>>(),
	)?;
	let target = oa::Matrix::new_from_f32(&engine, [2, 9, 4], &vec![0.0; 2 * 9 * 4])?;
	let layers = || -> oa::Result<oa::ml::nn::Conv1d> {
		oa::ml::nn::Conv1d::new_with_seed(&engine, 3, 4, 3, 1, 1, 1, 0x7d1)
	};
	let reference = layers()?;
	let candidate = layers()?;
	let gradients = |layer: &oa::ml::nn::Conv1d| -> oa::Result<Vec<Vec<f32>>> {
		layer
			.parameters()
			.iter()
			.map(|p| p.gradient().unwrap().read_f32())
			.collect()
	};
	let run = |layer: &oa::ml::nn::Conv1d| -> oa::Result<Vec<Vec<f32>>> {
		let tape = oa::ml::GradientTape::new();
		let transposed = oa::matrix::transpose(&layer.forward(&input)?, 1, 2)?;
		let loss = oa::ml::loss::mse(&transposed, &target)?;
		tape.backward(&loss)?;
		gradients(layer)
	};
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let expected = run(&reference)?;
	unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
	let (plan, _) = engine.capture(|| {
		let _ = run(&candidate);
		Ok(())
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused epilogue executable report must be valid JSON");
	let nodes = report["nodes"].as_array().expect("nodes");
	let fused = nodes
		.iter()
		.filter(|node| {
			node["kernel"]
				.as_str()
				.unwrap()
				.ends_with("conv_1d_bias_copy.f32")
		})
		.count();
	assert_eq!(
		fused, 1,
		"the inverse-transpose tail must reduce to a biased copy"
	);
	assert!(
		!nodes.iter().any(|node| {
			let kernel = node["kernel"].as_str().unwrap();
			kernel.ends_with("conv_1d_bias_add.f32") || kernel.ends_with("conv_1d_transpose.f32")
		}),
		"no unfused Conv1d bias add or NCL transpose may remain"
	);
	engine.submit(&plan)?.wait()?;
	let actual = gradients(&candidate)?;
	for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
		assert!(
			(actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs(),
			"fused Conv1d copy tail must match unfused: {actual} vs {expected}"
		);
	}
	Ok(())
});

test_vk!(
	epilogue_fuses_conv_2d_relu_and_pool_relu_backward,
	engine,
	{
		// The commands-CNN block `conv2d → relu → max_pool` runs both new
		// fusions: the forward activation applies at the conv store and the
		// backward mask applies at the pool adjoint's store. Neither the raw
		// conv output nor the unmasked pool gradient is materialized.
		let input = oa::Matrix::new_from_f32(
			&engine,
			[2, 3, 8, 9],
			&vec![0.0; 2 * 3 * 8 * 9]
				.iter()
				.enumerate()
				.map(|(i, _)| ((i * 31 % 19) as f32 - 9.0) / 8.0)
				.collect::<Vec<f32>>(),
		)?;
		let target = oa::Matrix::new_from_f32(&engine, [2, 4, 4, 4], &vec![0.0; 2 * 4 * 4 * 4])?;
		let layers = || -> oa::Result<oa::ml::nn::Conv2d> {
			oa::ml::nn::Conv2d::new_with_seed(&engine, 3, 4, 3, 1, 1, 1, 0x8c1)
		};
		let reference = layers()?;
		let candidate = layers()?;
		let gradients = |layer: &oa::ml::nn::Conv2d| -> oa::Result<Vec<Vec<f32>>> {
			layer
				.parameters()
				.iter()
				.map(|p| p.gradient().unwrap().read_f32())
				.collect()
		};
		let run = |layer: &oa::ml::nn::Conv2d| -> oa::Result<Vec<Vec<f32>>> {
			let tape = oa::ml::GradientTape::new();
			let activated = oa::ml::matrix::relu(&layer.forward(&input)?)?;
			let pooled = oa::ml::nn::MaxPool2d::new(2)?.forward(&activated)?;
			let loss = oa::ml::loss::mse(&pooled, &target)?;
			tape.backward(&loss)?;
			gradients(layer)
		};
		unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
		let expected = run(&reference)?;
		unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
		let (plan, _) = engine.capture(|| {
			let _ = run(&candidate);
			Ok(())
		})?;
		let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
			.expect("fused epilogue executable report must be valid JSON");
		let nodes = report["nodes"].as_array().expect("nodes");
		for kernel in ["conv_2d_relu.f32", "max_pool_2d_relu_backward.f32"] {
			let fused = nodes
				.iter()
				.filter(|node| node["kernel"].as_str().unwrap().ends_with(kernel))
				.count();
			assert_eq!(fused, 1, "expected one fused {kernel} dispatch");
		}
		assert!(
			!nodes.iter().any(|node| {
				let kernel = node["kernel"].as_str().unwrap();
				kernel.ends_with("matrix.conv_2d.f32")
					|| kernel.ends_with("matrix.relu.f32")
					|| kernel.ends_with("matrix.relu_backward.f32")
					|| kernel.ends_with("matrix.max_pool_2d_backward.f32")
			}),
			"no unfused conv, relu or pool adjoint may remain"
		);
		engine.submit(&plan)?.wait()?;
		let actual = gradients(&candidate)?;
		for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
			assert!(
				(actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs(),
				"fused conv/pool adjoints must match unfused: {actual} vs {expected}"
			);
		}
		Ok(())
	}
);

test_vk!(
	epilogue_preserves_externally_retained_conv_outputs,
	engine,
	{
		// A structural provider that stops materializing an intermediate may only
		// do so when nothing outside the pending execution owns that storage. A
		// caller that keeps the raw convolution output beside its activated,
		// residual or transposed consumer must read the unfused values, in both
		// captured and eager submission.
		let ramp = |count: usize, modulus: usize, scale: f32| -> Vec<f32> {
			(0..count)
				.map(|i| ((i * 29 % modulus) as f32 - (modulus / 2) as f32) / scale)
				.collect()
		};
		let input_1d = oa::Matrix::new_from_f32(&engine, [2, 3, 10], &ramp(60, 17, 7.0))?;
		let residual = oa::Matrix::new_from_f32(&engine, [2, 4, 10], &ramp(80, 13, 9.0))?;
		let input_2d = oa::Matrix::new_from_f32(&engine, [2, 3, 6, 7], &ramp(252, 19, 8.0))?;
		let conv_1d = oa::ml::nn::Conv1d::new_with_seed(&engine, 3, 4, 3, 1, 1, 1, 0xa11)?;
		let conv_2d = oa::ml::nn::Conv2d::new_with_seed(&engine, 3, 4, 3, 1, 1, 1, 0xa12)?;
		type Case<'a> = (&'a str, Box<dyn Fn() -> oa::Result<Vec<oa::Matrix>> + 'a>);
		let cases: Vec<Case> = vec![
			(
				"conv1d relu",
				Box::new(|| {
					let conv = conv_1d.forward(&input_1d)?;
					let activated = oa::ml::matrix::relu(&conv)?;
					Ok(vec![conv, activated])
				}),
			),
			(
				"conv1d relu residual",
				Box::new(|| {
					let conv = conv_1d.forward(&input_1d)?;
					let summed = oa::matrix::add(&oa::ml::matrix::relu(&conv)?, &residual)?;
					Ok(vec![conv, summed])
				}),
			),
			(
				"conv1d inverse transpose",
				Box::new(|| {
					let conv = conv_1d.forward(&input_1d)?;
					let transposed = oa::matrix::transpose(&conv, 1, 2)?;
					Ok(vec![conv, transposed])
				}),
			),
			(
				"conv2d relu",
				Box::new(|| {
					let conv = conv_2d.forward(&input_2d)?;
					let activated = oa::ml::matrix::relu(&conv)?;
					Ok(vec![conv, activated])
				}),
			),
		];
		let read = |outputs: &[oa::Matrix]| -> oa::Result<Vec<Vec<f32>>> {
			outputs.iter().map(oa::Matrix::read_f32).collect()
		};
		for (name, build) in &cases {
			unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
			let expected = read(&build()?);
			unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
			let expected = expected?;
			assert!(
				expected[0].iter().any(|value| *value != 0.0),
				"{name}: the reference convolution output must be non-trivial"
			);
			let (plan, captured) = engine.capture(build)?;
			engine.submit(&plan)?.wait()?;
			assert_eq!(
				read(&captured)?,
				expected,
				"{name}: captured fusion must preserve retained outputs"
			);
			assert_eq!(
				read(&build()?)?,
				expected,
				"{name}: eager fusion must preserve retained outputs"
			);
		}
		Ok(())
	}
);

test_vk!(epilogue_fuses_sum_axis_scale_pair, engine, {
	let input = oa::Matrix::new_from_f32(&engine, [2, 3], &[1.0, -2.0, 0.5, -1.5, 2.0, 0.25])?;
	let reference = {
		let summed = oa::matrix::sum(&input, 1)?;
		oa::matrix::scale(&summed, 0.25)?.read_f32()?
	};
	let (plan, output) = engine.capture(|| {
		let summed = oa::matrix::sum(&input, 1)?;
		oa::matrix::scale(&summed, 0.25)
	})?;

	let lowering = plan.semantic_lowering();
	assert_eq!(lowering.fused_op_count(), 2);
	assert_eq!(lowering.fused_node_count(), 1);
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
		.expect("fused reduction report must be valid JSON");
	assert_eq!(report["nodes"].as_array().expect("nodes").len(), 1);
	assert_eq!(report["nodes"][0]["kernel"], "ml.sum_axis_scale.f32");

	engine.submit(&plan)?.wait()?;
	assert_eq!(output.read_f32()?, reference);
	Ok(())
});

/// Executable kernel histogram of a captured plan.
fn kernel_histogram(
	plan: &oa::ExecutionPlan,
	name: &str,
) -> (usize, std::collections::BTreeMap<String, usize>) {
	let report: serde_json::Value =
		serde_json::from_str(&plan.debug_report_json(name)).expect("plan report must be valid JSON");
	let nodes = report["nodes"].as_array().expect("nodes");
	let mut histogram = std::collections::BTreeMap::new();
	for node in nodes {
		*histogram
			.entry(node["kernel"].as_str().expect("kernel").to_owned())
			.or_default() += 1;
	}
	(nodes.len(), histogram)
}

test_vk!(alm_tokenizer_plan_keeps_its_conv_tail_fusions, engine, {
	// Pins the fused ALM training plan recorded in jit_fusion_census.md
	// (202 nodes vs 248 unfused): every Conv1d lowering tail sits inside a
	// fused variant and only the bias-free ConvTranspose1d backward
	// transposes remain.
	let config = oa::sdk::ml::alm::AlmTokenizerConfig {
		input_dim: 16,
		width: 32,
		code_dim: 16,
		num_codes: 64,
		downsample_stages: 2,
		depth: 2,
		commitment_beta: 0.25,
		ema_decay: 0.99,
		ema_epsilon: 1e-5,
		dead_threshold: 2.0,
	};
	let model = oa::sdk::ml::alm::AlmTokenizer::new(&engine, config)?;
	let input = oa::Matrix::new_from_f32(
		&engine,
		[2, 32, 16],
		&vec![0.05; 2 * 32 * 16]
			.iter()
			.enumerate()
			.map(|(i, v)| v * ((i * 37 % 23) as f32 - 11.0))
			.collect::<Vec<f32>>(),
	)?;
	let target = oa::Matrix::new_from_f32(&engine, [2, 32, 16], &vec![0.0; 2 * 32 * 16])?;
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let reconstruction = model.forward(&input)?;
		let loss = oa::ml::loss::mse(&reconstruction, &target)?;
		tape.backward(&loss)?;
		Ok(())
	})?;
	let (total, census) = kernel_histogram(&plan, "alm");
	assert_eq!(
		total, 202,
		"fused ALM training plan size changed: {census:?}"
	);
	assert!(
		!census.contains_key("ml.conv_1d_bias_add.f32"),
		"no unfused Conv1d bias add may remain"
	);
	assert_eq!(
		census.get("ml.conv_1d_transpose.f32").copied().unwrap_or(0),
		2,
		"only the bias-free ConvTranspose1d backward transposes may remain"
	);
	assert!(
		census.contains_key("ml.conv_1d_bias_transpose_residual_relu.f32"),
		"the residual Conv1d tail must stay fused"
	);
	Ok(())
});

test_vk!(commands_cnn_plan_keeps_its_conv_and_pool_fusions, engine, {
	// Pins the fused commands-CNN training plan recorded in
	// jit_fusion_census.md (23 nodes vs 27 unfused): each Conv2d block
	// applies ReLU at the conv store and folds the ReLU mask into the pool
	// adjoint.
	let model = oa::sdk::ml::audio::commands_cnn(&engine, 40, 62, 8)?;
	let input = oa::Matrix::new_from_f32(
		&engine,
		[4, 1, 40, 62],
		&vec![0.05; 4 * 40 * 62]
			.iter()
			.enumerate()
			.map(|(i, v)| v * ((i * 37 % 23) as f32 - 11.0))
			.collect::<Vec<f32>>(),
	)?;
	let target = oa::Matrix::new_from_slice(&engine, [4], &[0_u32, 2, 1, 3])?;
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&input)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		Ok(())
	})?;
	let (total, census) = kernel_histogram(&plan, "cnn");
	assert_eq!(
		total, 23,
		"fused commands-CNN training plan size changed: {census:?}"
	);
	for (kernel, count) in [
		("ml.conv_2d_relu.f32", 2),
		("ml.max_pool_2d_relu_backward.f32", 2),
		("ml.matrix.conv_2d.f32", 0),
		("ml.matrix.relu_backward.f32", 0),
	] {
		assert_eq!(
			census.get(kernel).copied().unwrap_or(0),
			count,
			"{kernel}: {census:?}"
		);
	}
	Ok(())
});

test_vk!(capture_preserves_metadata_view_provenance, engine, {
	let source = oa::matrix::ones(&engine, [2, 3])?;
	let view = source.reshape([3, 2])?;
	let (plan, output) = engine.capture(|| oa::matrix::scale(&view, 2.0))?;

	let semantic = plan.semantic_graph();
	assert_eq!(semantic.values().len(), 3);
	assert_eq!(semantic.view_count(), 1);
	assert_eq!(semantic.operations().len(), 1);
	let view_id = semantic.values()[1].id().expect("captured view has an id");
	assert_eq!(
		semantic.values()[1].view_source(),
		semantic.values()[0].id()
	);
	assert_eq!(semantic.operations()[0].inputs(), &[Some(view_id)]);
	assert_eq!(semantic.operations()[0].attributes().len(), 1);
	assert_eq!(semantic.operations()[0].attributes()[0].name(), "scalar");
	assert_eq!(plan.semantic_lowering().direct_op_count(), 1);

	engine.submit(&plan)?.wait()?;
	assert_eq!(output.read_f32()?, [2.0; 6]);
	Ok(())
});

test_vk!(
	timed_replays_report_exact_device_durations,
	engine,
	"requires a hardware Vulkan 1.3 compute device with timestamp support",
	{
		use std::time::Duration;

		let one = oa::matrix::ones(&engine, [262_147])?;
		let two = oa::matrix::full(&engine, [262_147], 2.0)?;
		let (plan, output) = engine.capture(|| {
			let three = oa::matrix::add(&one, &two)?;
			oa::matrix::add(&three, &three)
		})?;

		let untimed = engine.submit(&plan)?;
		assert_eq!(
			untimed.device_duration().unwrap_err().kind(),
			oa::ErrorKind::FailedPrecondition
		);

		let first = engine.submit_timed(&plan)?;
		let second = engine.submit_timed(&plan)?;
		let first_duration = first.device_duration()?;
		let second_duration = second.device_duration()?;
		assert!(first_duration > Duration::ZERO);
		assert!(second_duration > Duration::ZERO);
		assert_eq!(first.try_device_duration()?, Some(first_duration));
		assert_eq!(second.try_device_duration()?, Some(second_duration));
		assert_eq!(output.read_f32()?, vec![6.0; 262_147]);

		let surviving_event = {
			let engine = oa::Engine::new()?;
			let input = oa::matrix::ones(&engine, [262_147])?;
			let (plan, _output) = engine.capture(|| oa::matrix::add(&input, &input))?;
			engine.submit_timed(&plan)?
		};
		assert!(surviving_event.device_duration()? > Duration::ZERO);
		Ok(())
	}
);

test_vk!(
	capture_rejects_dirty_empty_nested_and_failed_scopes,
	engine,
	{
		use std::panic::{AssertUnwindSafe, catch_unwind};

		let one = oa::matrix::ones(&engine, [2])?;
		let two = oa::matrix::full(&engine, [2], 2.0)?;
		let pending = oa::matrix::add(&one, &two)?;

		let dirty = engine.capture(|| Ok(()));
		assert_eq!(dirty.unwrap_err().kind(), oa::ErrorKind::FailedPrecondition);
		assert_eq!(pending.read_f32()?, [3.0; 2]);

		let empty = engine.capture(|| Ok(()));
		assert_eq!(empty.unwrap_err().kind(), oa::ErrorKind::FailedPrecondition);

		let nested = engine.capture(|| {
			let (_plan, _output) = engine.capture(|| Ok(()))?;
			Ok(())
		});
		assert_eq!(
			nested.unwrap_err().kind(),
			oa::ErrorKind::FailedPrecondition
		);

		let mismatched = oa::matrix::ones(&engine, [3])?;
		let mut escaped = None;
		let failed = engine.capture(|| {
			escaped = Some(oa::matrix::add(&one, &two)?);
			oa::matrix::add(&one, &mismatched)?;
			Ok(())
		});
		assert_eq!(failed.unwrap_err().kind(), oa::ErrorKind::InvalidArgument);
		let Some(escaped) = escaped else {
			panic!("captured output did not escape");
		};
		assert_eq!(
			escaped.read_f32().unwrap_err().kind(),
			oa::ErrorKind::FailedPrecondition
		);

		let mut panic_escaped = None;
		let unwound = catch_unwind(AssertUnwindSafe(|| {
			let _capture = engine.capture(|| -> oa::Result<()> {
				panic_escaped = Some(oa::matrix::add(&one, &two)?);
				panic!("capture unwind probe");
			});
		}));
		assert!(unwound.is_err());
		let Some(panic_escaped) = panic_escaped else {
			panic!("unwound captured output did not escape");
		};
		assert_eq!(
			panic_escaped.read_f32().unwrap_err().kind(),
			oa::ErrorKind::FailedPrecondition
		);
		assert_eq!(oa::matrix::add(&one, &two)?.read_f32()?, [3.0; 2]);
		Ok(())
	}
);

test_vk!(rejects_foreign_plan_without_poisoning_its_origin, origin, {
	let input = oa::matrix::ones(&origin, [3])?;
	let (plan, output) = origin.capture(|| oa::matrix::add(&input, &input))?;
	let foreign = oa::Engine::new()?;

	let error = foreign.submit(&plan).unwrap_err();
	assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	origin.submit(&plan)?.wait()?;
	assert_eq!(output.read_f32()?, [2.0; 3]);
	Ok(())
});

test_vk!(reuses_commands_and_rebinds_stable_matrix_inputs, engine, {
	let one = oa::matrix::ones(&engine, [2, 3])?;
	let two = oa::matrix::full(&engine, [2, 3], 2.0)?;
	let (mut plan, output) = engine.capture(|| {
		let three = oa::matrix::add(&one, &two)?;
		oa::matrix::add(&three, &one)
	})?;

	let initial = plan.diagnostics();
	assert_ne!(initial.graph_id(), 0);
	assert_eq!(initial.node_count(), 1);
	assert_eq!(initial.barrier_count(), 0);
	assert_eq!(initial.semantic_operation_count(), 2);
	assert_eq!(initial.schema_owned_node_count(), 1);
	assert_eq!(initial.compatibility_node_count(), 0);
	assert_ne!(initial.dnn_graph_hash(), 0);
	assert_eq!(initial.dnn_value_count(), 4);
	assert_eq!(initial.dnn_external_value_count(), 2);
	assert_eq!(initial.dnn_virtual_value_count(), 0);
	assert_eq!(initial.dnn_partition_count(), 2);
	assert_eq!(initial.dnn_recognized_partition_count(), 0);
	assert_eq!(initial.dnn_portable_partition_count(), 2);
	assert_eq!(initial.dnn_captured_operation_count(), 2);
	assert_eq!(initial.input_binding_count(), 2);
	assert_eq!(initial.captured_resource_count(), 5);
	assert_eq!(initial.semantic_binding_count(), 4);
	assert_eq!(initial.observed_output_count(), 0);
	assert_eq!(plan.semantic_bindings().len(), 4);
	assert_eq!(
		plan
			.captured_resources()
			.iter()
			.filter(|resource| resource.semantic_external())
			.count(),
		2
	);
	assert_eq!(initial.command_recording_count(), 0);
	assert_eq!(initial.command_cache_hit_count(), 0);
	assert_eq!(initial.submission_count(), 0);
	assert_eq!(initial.fallback_count(), 0);
	let duplicate_one = oa::matrix::ones(&engine, [2, 3])?;
	let duplicate_two = oa::matrix::full(&engine, [2, 3], 2.0)?;
	let (equivalent_plan, _equivalent_output) = engine.capture(|| {
		let three = oa::matrix::add(&duplicate_one, &duplicate_two)?;
		oa::matrix::add(&three, &duplicate_one)
	})?;
	assert_eq!(equivalent_plan.diagnostics().graph_id(), initial.graph_id());
	drop(equivalent_plan);

	let first = engine.submit(&plan)?;
	let second = engine.submit(&plan)?;
	first.wait()?;
	second.wait()?;
	assert_eq!(output.read_f32()?, [4.0; 6]);
	let reused = plan.diagnostics();
	assert_eq!(reused.command_recording_count(), 1);
	assert_eq!(reused.command_cache_hit_count(), 1);
	assert_eq!(reused.submission_count(), 2);

	let four = oa::matrix::full(&engine, [2, 3], 4.0)?;
	plan.bind_matrix_input(&one, &four)?;
	let rebound = plan.diagnostics();
	assert_eq!(rebound.graph_id(), initial.graph_id());
	assert_eq!(rebound.input_binding_count(), 2);
	assert_eq!(rebound.input_rebinding_count(), 1);
	let rebound_report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("rebound"))
		.expect("rebound executable report must be valid JSON");
	assert_eq!(rebound_report["compiled"], false);

	let third = engine.submit(&plan)?;
	let fourth = engine.submit(&plan)?;
	plan.bind_matrix_input(&one, &one)?;
	let restored_report: serde_json::Value =
		serde_json::from_str(&plan.debug_report_json("restored"))
			.expect("restored executable report must be valid JSON");
	assert_eq!(restored_report["compiled"], true);
	let fifth = engine.submit(&plan)?;
	let sixth = engine.submit(&plan)?;
	let final_diagnostics = plan.diagnostics();
	assert_eq!(final_diagnostics.command_recording_count(), 2);
	assert_eq!(final_diagnostics.command_cache_hit_count(), 4);
	assert_eq!(final_diagnostics.submission_count(), 6);
	assert_eq!(final_diagnostics.fallback_count(), 0);
	drop(plan);
	third.wait()?;
	fourth.wait()?;
	fifth.wait()?;
	sixth.wait()?;
	assert_eq!(output.read_f32()?, [4.0; 6]);
	Ok(())
});

test_vk!(dnn_analysis_recognizes_a_schema_owned_qkv_region, engine, {
	let activation = oa::matrix::ones(&engine, [2, 3])?;
	let q_weight = oa::matrix::ones(&engine, [4, 3])?;
	let k_weight = oa::matrix::full(&engine, [4, 3], 2.0)?;
	let v_weight = oa::matrix::full(&engine, [4, 3], 3.0)?;
	let (plan, output) = engine.capture(|| {
		let _query = oa::matrix::mat_mul_nt(&activation, &q_weight)?;
		let _key = oa::matrix::mat_mul_nt(&activation, &k_weight)?;
		oa::matrix::mat_mul_nt(&activation, &v_weight)
	})?;

	let diagnostics = plan.diagnostics();
	assert_eq!(diagnostics.semantic_operation_count(), 3);
	assert_eq!(diagnostics.dnn_captured_operation_count(), 3);
	assert_eq!(diagnostics.dnn_value_count(), 7);
	assert_eq!(diagnostics.dnn_external_value_count(), 4);
	assert_eq!(diagnostics.dnn_virtual_value_count(), 0);
	assert_eq!(diagnostics.dnn_partition_count(), 1);
	assert_eq!(diagnostics.dnn_recognized_partition_count(), 1);
	assert_eq!(diagnostics.dnn_portable_partition_count(), 0);
	assert_eq!(diagnostics.dnn_applied_partition_count(), 0);
	assert_eq!(diagnostics.dnn_fallback_partition_count(), 1);
	assert_eq!(diagnostics.dnn_unexpected_fallback_count(), 0);

	engine.submit(&plan)?.wait()?;
	assert_eq!(output.read_f32()?, [9.0; 8]);
	Ok(())
});

test_vk!(
	dnn_lowering_fuses_the_donor_qualified_qkv_projection,
	engine,
	{
		let input = oa::matrix::ones(&engine, [1024, 32])?;
		let query = oa::ml::nn::Linear::new_from_matrices(
			oa::matrix::ones(&engine, [32, 32])?,
			oa::matrix::full(&engine, [32], 0.5)?,
		)?;
		let key = oa::ml::nn::Linear::new_from_matrices(
			oa::matrix::full(&engine, [32, 32], 2.0)?,
			oa::matrix::full(&engine, [32], 1.0)?,
		)?;
		let value = oa::ml::nn::Linear::new_from_matrices(
			oa::matrix::full(&engine, [32, 32], 3.0)?,
			oa::matrix::full(&engine, [32], 1.5)?,
		)?;
		let capture = || {
			engine.capture(|| {
				Ok((
					query.forward(&input)?,
					key.forward(&input)?,
					value.forward(&input)?,
				))
			})
		};
		let check = |outputs: (oa::Matrix, oa::Matrix, oa::Matrix)| -> oa::Result<()> {
			assert_eq!(outputs.0.read_f32()?, vec![32.5; 1024 * 32]);
			assert_eq!(outputs.1.read_f32()?, vec![65.0; 1024 * 32]);
			assert_eq!(outputs.2.read_f32()?, vec![97.5; 1024 * 32]);
			Ok(())
		};

		// L1 owns shared-input projection epilogues (OA-JIT-05 owner gate):
		// the submit-time `linear_triple` absorbs the three projections before
		// the captured-plan DNN analyzer sees them.
		let (plan, outputs) = capture()?;
		let diagnostics = plan.diagnostics();
		assert_eq!(diagnostics.node_count(), 1);
		assert_eq!(diagnostics.dnn_applied_partition_count(), 0);
		let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("triple"))
			.expect("fused triple executable report must be valid JSON");
		assert_eq!(report["nodes"][0]["kernel"], "ml.linear_triple.f32");
		engine.submit(&plan)?.wait()?;
		check(outputs)?;

		// With L1 disabled, the L3 donor-qualified QKV lowering applies.
		unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
		let captured = capture();
		unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
		let (plan, (query_output, key_output, value_output)) = captured?;

		let diagnostics = plan.diagnostics();
		assert_eq!(diagnostics.semantic_operation_count(), 3);
		assert_eq!(diagnostics.node_count(), 1);
		assert_eq!(diagnostics.dnn_recognized_partition_count(), 1);
		assert_eq!(diagnostics.dnn_applied_partition_count(), 1);
		assert_eq!(diagnostics.dnn_inherited_partition_count(), 0);
		assert_eq!(diagnostics.dnn_fallback_partition_count(), 0);
		assert_eq!(diagnostics.dnn_unexpected_fallback_count(), 0);
		assert_eq!(plan.semantic_lowering().fused_op_count(), 3);
		assert_eq!(plan.semantic_lowering().fused_node_count(), 1);
		assert_eq!(plan.semantic_lowering().maximum_ops_per_node(), 3);
		let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("qkv"))
			.expect("fused QKV executable report must be valid JSON");
		assert_eq!(
			report["nodes"][0]["operation"],
			"oa::Dnn::QkvProjectionBias"
		);
		assert_eq!(report["nodes"][0]["kernel"], "ml.qkv_projection_bias.f32");
		assert_eq!(
			report["nodes"][0]["semantic_operations"],
			serde_json::json!([0, 1, 2])
		);

		engine.submit(&plan)?.wait()?;
		check((query_output, key_output, value_output))
	}
);

test_vk!(
	dnn_lowering_fuses_the_donor_qualified_gate_up_swiglu,
	engine,
	{
		let input = oa::matrix::ones(&engine, [1024, 32])?;
		let gate = oa::ml::nn::Linear::new_from_matrices(
			oa::matrix::full(&engine, [64, 32], 0.01)?,
			oa::matrix::full(&engine, [64], 0.1)?,
		)?;
		let up = oa::ml::nn::Linear::new_from_matrices(
			oa::matrix::full(&engine, [64, 32], 0.02)?,
			oa::matrix::full(&engine, [64], 0.2)?,
		)?;
		let (plan, output) = engine.capture(|| {
			let gate_output = gate.forward(&input)?;
			let up_output = up.forward(&input)?;
			oa::ml::matrix::swiglu(&gate_output, &up_output)
		})?;

		let diagnostics = plan.diagnostics();
		assert_eq!(diagnostics.semantic_operation_count(), 3);
		assert_eq!(diagnostics.node_count(), 1);
		assert_eq!(diagnostics.dnn_applied_partition_count(), 1);
		assert_eq!(diagnostics.dnn_fallback_partition_count(), 0);
		assert_eq!(plan.semantic_lowering().fused_op_count(), 3);
		let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("gate_up"))
			.expect("fused gate/up executable report must be valid JSON");
		assert_eq!(report["nodes"][0]["operation"], "oa::Dnn::GateUpSwigluBias");
		assert_eq!(report["nodes"][0]["kernel"], "ml.gate_up_swiglu_bias.f32");
		assert_eq!(
			report["nodes"][0]["semantic_operations"],
			serde_json::json!([0, 1, 2])
		);

		engine.submit(&plan)?.wait()?;
		let expected = 0.42_f32 / (1.0 + (-0.42_f32).exp()) * 0.84;
		for actual in output.read_f32()? {
			assert!((actual - expected).abs() <= 1.0e-5);
		}
		Ok(())
	}
);

test_vk!(
	dnn_gate_up_retains_source_nodes_when_an_intermediate_escapes,
	engine,
	{
		let input = oa::matrix::ones(&engine, [1024, 32])?;
		let gate = oa::ml::nn::Linear::new_from_matrices(
			oa::matrix::full(&engine, [64, 32], 0.01)?,
			oa::matrix::full(&engine, [64], 0.1)?,
		)?;
		let up = oa::ml::nn::Linear::new_from_matrices(
			oa::matrix::full(&engine, [64, 32], 0.02)?,
			oa::matrix::full(&engine, [64], 0.2)?,
		)?;
		let (plan, (_escaped, output)) = engine.capture(|| {
			let gate_output = gate.forward(&input)?;
			let up_output = up.forward(&input)?;
			let output = oa::ml::matrix::swiglu(&gate_output, &up_output)?;
			Ok((gate_output, output))
		})?;
		let diagnostics = plan.diagnostics();
		assert_eq!(diagnostics.node_count(), 3);
		assert_eq!(diagnostics.dnn_applied_partition_count(), 0);
		assert_eq!(diagnostics.dnn_fallback_partition_count(), 1);
		assert_eq!(diagnostics.dnn_unexpected_fallback_count(), 0);
		engine.submit(&plan)?.wait()?;
		assert_eq!(output.read_f32()?.len(), 1024 * 64);
		Ok(())
	}
);

test_vk!(
	rejects_invalid_matrix_input_rebinding_without_mutating_the_plan,
	engine,
	{
		let one = oa::matrix::ones(&engine, [2, 3])?;
		let two = oa::matrix::full(&engine, [2, 3], 2.0)?;
		let (mut plan, output) = engine.capture(|| oa::matrix::add(&one, &two))?;
		let graph_id = plan.diagnostics().graph_id();

		let wrong_shape = oa::matrix::ones(&engine, [3, 2])?;
		assert_eq!(
			plan
				.bind_matrix_input(&one, &wrong_shape)
				.unwrap_err()
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		assert_eq!(
			plan.bind_matrix_input(&one, &two).unwrap_err().kind(),
			oa::ErrorKind::InvalidArgument
		);
		let replacement_output = oa::matrix::ones(&engine, [2, 3])?;
		assert_eq!(
			plan
				.bind_matrix_input(&output, &replacement_output)
				.unwrap_err()
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		let unrelated = oa::matrix::ones(&engine, [2, 3])?;
		assert_eq!(
			plan
				.bind_matrix_input(&unrelated, &unrelated)
				.unwrap_err()
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		let foreign = oa::Engine::new()?;
		let foreign_input = oa::matrix::ones(&foreign, [2, 3])?;
		assert_eq!(
			plan
				.bind_matrix_input(&one, &foreign_input)
				.unwrap_err()
				.kind(),
			oa::ErrorKind::InvalidArgument
		);

		let diagnostics = plan.diagnostics();
		assert_eq!(diagnostics.graph_id(), graph_id);
		assert_eq!(diagnostics.input_binding_count(), 2);
		assert_eq!(diagnostics.input_rebinding_count(), 0);
		engine.submit(&plan)?.wait()?;
		assert_eq!(output.read_f32()?, [3.0; 6]);
		Ok(())
	}
);

test_vk!(
	timed_replay_reuses_graph_command_and_retains_independent_queries,
	engine,
	{
		let x = oa::matrix::ones(&engine, [17])?;
		let (plan, y) = engine.capture(|| oa::matrix::scale(&x, 3.))?;
		let first = engine.submit_timed(&plan)?;
		let second = engine.submit_timed(&plan)?;
		let a = first.device_duration()?;
		let b = second.device_duration()?;
		assert!(!a.is_zero() && !b.is_zero());
		assert_eq!(first.device_duration()?, a);
		assert_eq!(plan.diagnostics().command_recording_count(), 1);
		assert_eq!(plan.diagnostics().command_cache_hit_count(), 1);
		assert_eq!(y.read_f32()?, vec![3.; 17]);
		drop(plan);
		assert_eq!(second.device_duration()?, b);
		Ok(())
	}
);

test_vk!(
	dnn_filtered_storage_root_keeps_view_on_source_path,
	engine,
	{
		let scalar = oa::Matrix::new_from_f32(&engine, [], &[3.0])?;
		let (plan, output) = engine.capture(|| {
			let view = oa::matrix::reshape(&scalar, [1])?;
			oa::matrix::add_scalar(&view, 2.0)
		})?;
		assert_eq!(plan.diagnostics().dnn_captured_operation_count(), 0);
		engine.submit(&plan)?.wait()?;
		assert_eq!(output.read_f32()?, vec![5.0]);
		Ok(())
	}
);

test_vk!(
	dnn_lowering_fuses_the_donor_qualified_vision_preprocess,
	engine,
	{
		let rgb = oa::Image::new(
			oa::Matrix::new_from_f32(
				&engine,
				[1, 3, 2, 4],
				&[
					1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
					10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, //
					100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0,
				],
			)?,
			oa::ImageLayout::Nchw,
			oa::ImageFormat::Rgb,
		)?;
		let parameters = oa::image::NormalizationConfig {
			mean: [1.0, 2.0, 4.0],
			std: [2.0, 4.0, 8.0],
		};
		let reference = oa::image::resize_normalize(
			&oa::image::convert_color(&rgb, oa::ImageFormat::Bgr)?,
			2,
			2,
			parameters,
		)?
		.as_matrix()
		.read_f32()?;
		let (plan, output) = engine.capture(|| {
			let converted = oa::image::convert_color(&rgb, oa::ImageFormat::Bgr)?;
			oa::image::resize_normalize(&converted, 2, 2, parameters)
		})?;

		let diagnostics = plan.diagnostics();
		assert_eq!(diagnostics.dnn_recognized_partition_count(), 1);
		assert_eq!(diagnostics.dnn_applied_partition_count(), 1);
		assert_eq!(diagnostics.dnn_fallback_partition_count(), 0);
		assert_eq!(diagnostics.dnn_unexpected_fallback_count(), 0);
		assert_eq!(plan.semantic_lowering().fused_op_count(), 2);
		assert_eq!(plan.semantic_lowering().fused_node_count(), 1);
		let report: serde_json::Value =
			serde_json::from_str(&plan.debug_report_json("vision_preprocess"))
				.expect("fused vision preprocess executable report must be valid JSON");
		assert_eq!(report["nodes"][0]["operation"], "oa::Dnn::VisionPreprocess");
		assert_eq!(report["nodes"][0]["kernel"], "image.resize_normalize.f32");

		engine.submit(&plan)?.wait()?;
		assert_eq!(output.as_matrix().read_f32()?, reference);
		Ok(())
	}
);

test_vk!(dnn_lowering_fuses_linear_bias_epilogue, engine, {
	let a = oa::Matrix::new_from_f32(
		&engine,
		[4, 8],
		&[
			1.0, -2.0, 0.5, 3.0, -1.5, 2.0, 0.25, 4.0, //
			-3.0, 1.0, 2.5, -0.5, 1.25, -2.5, 3.5, 0.75, //
			2.0, 0.0, -1.0, 1.5, -3.5, 0.5, 2.25, -4.0, //
			0.5, 1.5, -2.25, 3.25, 4.5, -0.75, -1.25, 2.75,
		],
	)?;
	let b = oa::Matrix::new_from_f32(
		&engine,
		[6, 8],
		&[
			0.5, -1.0, 2.0, 1.0, -0.5, 3.0, -2.0, 1.5, //
			1.0, 2.0, -1.5, 0.25, 2.5, -3.0, 0.75, -1.25, //
			-0.25, 3.5, 1.25, -2.5, 4.0, 0.5, -1.75, 2.25, //
			2.5, -0.75, -3.25, 1.75, -4.5, 2.75, 0.25, -0.5, //
			-1.5, 0.25, 3.75, -2.25, 1.75, -0.25, 4.25, -3.5, //
			0.75, -2.75, -0.5, 3.25, -3.75, 1.25, -1.0, 0.5,
		],
	)?;
	let bias = oa::Matrix::new_from_f32(&engine, [6], &[0.5, -1.0, 2.0, -0.25, 1.5, -3.0])?;
	let reference = oa::matrix::bias_add(&oa::matrix::mat_mul_nt(&a, &b)?, &bias)?.read_f32()?;
	let (plan, output) = engine.capture(|| {
		let product = oa::matrix::mat_mul_nt(&a, &b)?;
		oa::matrix::bias_add(&product, &bias)
	})?;

	let diagnostics = plan.diagnostics();
	assert_eq!(diagnostics.dnn_recognized_partition_count(), 1);
	assert_eq!(diagnostics.dnn_applied_partition_count(), 1);
	assert_eq!(diagnostics.dnn_fallback_partition_count(), 0);
	assert_eq!(diagnostics.dnn_unexpected_fallback_count(), 0);
	assert_eq!(plan.semantic_lowering().fused_op_count(), 2);
	assert_eq!(plan.semantic_lowering().fused_node_count(), 1);
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("linear_epilogue"))
		.expect("fused linear epilogue executable report must be valid JSON");
	assert_eq!(report["nodes"][0]["operation"], "oa::Dnn::LinearEpilogue");
	assert_eq!(report["nodes"][0]["kernel"], "matrix.mat_mul_nt_bias.f32");

	engine.submit(&plan)?.wait()?;
	assert_eq!(output.read_f32()?, reference);
	Ok(())
});

test_vk!(
	dnn_lowering_fuses_linear_bias_activation_epilogues,
	engine,
	{
		let a = oa::Matrix::new_from_f32(
			&engine,
			[4, 8],
			&[
				1.0, -2.0, 0.5, 3.0, -1.5, 2.0, 0.25, 4.0, //
				-3.0, 1.0, 2.5, -0.5, 1.25, -2.5, 3.5, 0.75, //
				2.0, 0.0, -1.0, 1.5, -3.5, 0.5, 2.25, -4.0, //
				0.5, 1.5, -2.25, 3.25, 4.5, -0.75, -1.25, 2.75,
			],
		)?;
		let b = oa::Matrix::new_from_f32(
			&engine,
			[6, 8],
			&[
				0.5, -1.0, 2.0, 1.0, -0.5, 3.0, -2.0, 1.5, //
				1.0, 2.0, -1.5, 0.25, 2.5, -3.0, 0.75, -1.25, //
				-0.25, 3.5, 1.25, -2.5, 4.0, 0.5, -1.75, 2.25, //
				2.5, -0.75, -3.25, 1.75, -4.5, 2.75, 0.25, -0.5, //
				-1.5, 0.25, 3.75, -2.25, 1.75, -0.25, 4.25, -3.5, //
				0.75, -2.75, -0.5, 3.25, -3.75, 1.25, -1.0, 0.5,
			],
		)?;
		let bias = oa::Matrix::new_from_f32(&engine, [6], &[0.5, -1.0, 2.0, -0.25, 1.5, -3.0])?;
		for (activate, kernel) in [
			(
				oa::ml::matrix::relu as fn(&oa::Matrix) -> oa::Result<oa::Matrix>,
				"matrix.mat_mul_nt_bias_relu.f32",
			),
			(oa::ml::matrix::gelu, "matrix.mat_mul_nt_bias_gelu.f32"),
			(oa::ml::matrix::silu, "matrix.mat_mul_nt_bias_silu.f32"),
		] {
			let reference = activate(&oa::matrix::bias_add(
				&oa::matrix::mat_mul_nt(&a, &b)?,
				&bias,
			)?)?
			.read_f32()?;
			let (plan, output) = engine.capture(|| {
				let product = oa::matrix::mat_mul_nt(&a, &b)?;
				activate(&oa::matrix::bias_add(&product, &bias)?)
			})?;

			let diagnostics = plan.diagnostics();
			assert_eq!(diagnostics.dnn_recognized_partition_count(), 1);
			assert_eq!(diagnostics.dnn_applied_partition_count(), 1);
			assert_eq!(diagnostics.dnn_fallback_partition_count(), 0);
			assert_eq!(diagnostics.dnn_unexpected_fallback_count(), 0);
			assert_eq!(plan.semantic_lowering().fused_op_count(), 3);
			assert_eq!(plan.semantic_lowering().fused_node_count(), 1);
			let report: serde_json::Value =
				serde_json::from_str(&plan.debug_report_json("linear_epilogue"))
					.expect("fused linear epilogue executable report must be valid JSON");
			assert_eq!(report["nodes"][0]["operation"], "oa::Dnn::LinearEpilogue");
			assert_eq!(report["nodes"][0]["kernel"], kernel);

			engine.submit(&plan)?.wait()?;
			let fused = output.read_f32()?;
			for (index, (actual, expected)) in fused.iter().zip(&reference).enumerate() {
				assert!(
					(actual - expected).abs() <= 2e-5,
					"{kernel} output[{index}]: {actual} != {expected}"
				);
			}
		}
		Ok(())
	}
);
