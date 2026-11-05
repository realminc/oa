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
		let (plan, (query_output, key_output, value_output)) = engine.capture(|| {
			Ok((
				query.forward(&input)?,
				key.forward(&input)?,
				value.forward(&input)?,
			))
		})?;

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
		assert_eq!(query_output.read_f32()?, vec![32.5; 1024 * 32]);
		assert_eq!(key_output.read_f32()?, vec![65.0; 1024 * 32]);
		assert_eq!(value_output.read_f32()?, vec![97.5; 1024 * 32]);
		Ok(())
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
