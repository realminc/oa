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
	assert_eq!(semantic.operations()[0].name(), "oa::matrix::add");
	assert_eq!(
		semantic.operations()[0].contract_hash(),
		oa::core::operation::matrix::ADD.hash()
	);
	assert_eq!(
		semantic.operations()[1].inputs()[0],
		semantic.operations()[0].outputs().first().copied()
	);
	let lowering = plan.semantic_lowering();
	assert_eq!(lowering.schema_owned_node_count(), 2);
	assert_eq!(lowering.compatibility_node_count(), 0);
	assert_eq!(lowering.direct_op_count(), 2);
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
	assert_eq!(report["stats"]["dispatches"], 2);
	assert_eq!(report["stats"]["barriers"], 1);
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
	assert_eq!(report["barriers"][0]["reason"], "read_after_write");
	assert_eq!(
		report["barriers"][0]["source_nodes"],
		serde_json::json!([0, 0])
	);
	assert_eq!(report["barriers"][0]["destination_node"], 1);
	assert_eq!(report["barriers"][0]["source_resource"], 2);
	assert_eq!(report["barriers"][0]["destination_resource"], 2);
	assert_eq!(
		report["barriers"][0]["source_stages"],
		serde_json::json!(["compute_shader"])
	);
	assert_eq!(report["nodes"][0]["operation"], "oa::matrix::add");
	assert_eq!(
		report["nodes"][0]["semantic_operations"],
		serde_json::json!([0])
	);
	assert_eq!(report["nodes"][0]["kernel"], "matrix.add.f32");
	assert_eq!(report["nodes"][0]["dtype_class"], serde_json::Value::Null);
	assert_eq!(report["nodes"][0]["dtype"], "float32");
	assert_eq!(report["nodes"][0]["kernel_selection"], "direct");
	assert_eq!(
		report["nodes"][0]["physical_write"],
		serde_json::Value::Null
	);
	assert_eq!(report["nodes"][0]["effects"][0]["resource"], 0);
	assert_eq!(report["nodes"][0]["effects"][0]["access"], "read");
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

		let mismatched = oa::matrix::ones(&engine, [1, 2])?;
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
	assert_eq!(initial.node_count(), 2);
	assert_eq!(initial.barrier_count(), 1);
	assert_eq!(initial.semantic_operation_count(), 2);
	assert_eq!(initial.schema_owned_node_count(), 2);
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
	assert_eq!(initial.captured_resource_count(), 4);
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
	let final_diagnostics = plan.diagnostics();
	assert_eq!(final_diagnostics.command_recording_count(), 2);
	assert_eq!(final_diagnostics.command_cache_hit_count(), 2);
	assert_eq!(final_diagnostics.submission_count(), 4);
	assert_eq!(final_diagnostics.fallback_count(), 0);
	drop(plan);
	third.wait()?;
	fourth.wait()?;
	assert_eq!(output.read_f32()?, [10.0; 6]);
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
		let query = oa::ml::nn::Linear::from_matrices(
			oa::matrix::ones(&engine, [32, 32])?,
			oa::matrix::full(&engine, [32], 0.5)?,
		)?;
		let key = oa::ml::nn::Linear::from_matrices(
			oa::matrix::full(&engine, [32, 32], 2.0)?,
			oa::matrix::full(&engine, [32], 1.0)?,
		)?;
		let value = oa::ml::nn::Linear::from_matrices(
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
		let gate = oa::ml::nn::Linear::from_matrices(
			oa::matrix::full(&engine, [64, 32], 0.01)?,
			oa::matrix::full(&engine, [64], 0.1)?,
		)?;
		let up = oa::ml::nn::Linear::from_matrices(
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
		let gate = oa::ml::nn::Linear::from_matrices(
			oa::matrix::full(&engine, [64, 32], 0.01)?,
			oa::matrix::full(&engine, [64], 0.1)?,
		)?;
		let up = oa::ml::nn::Linear::from_matrices(
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
