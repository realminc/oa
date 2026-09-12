use oa::ml::loss::{DqnLossConfig, dqn};

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= tolerance,
			"DQN mismatch at {index}: expected {expected}, found {actual}"
		);
	}
}

test_vk!(dqn_matches_donor_boundary_oracle_and_q_adjoint, engine, {
	let q_parameter = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
		&engine,
		[3, 2],
		&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
	)?)?;
	let next_q_parameter = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
		&engine,
		[3, 2],
		&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
	)?)?;
	let rows = oa::Matrix::from_slice(&engine, [3], &[0_u32, 1, 2])?;
	let action = oa::Matrix::from_slice(&engine, [3], &[0_i32, 1, 0])?;
	let reward = oa::Matrix::from_f32(&engine, [3], &[1.0, 2.0, 3.0])?;
	let terminated = oa::Matrix::from_slice(&engine, [3], &[0_u8, 1, 0])?;
	let truncated = oa::Matrix::from_slice(&engine, [3], &[0_u8, 0, 1])?;
	let tape = oa::ml::GradientTape::new();
	let q = q_parameter.forward(&rows)?;
	let next_q = next_q_parameter.forward(&rows)?;
	engine.checkpoint()?.wait()?;

	let (plan, result) = engine.capture(|| {
		dqn(
			&q,
			&action,
			&reward,
			&next_q,
			&terminated,
			&truncated,
			DqnLossConfig { discount: 0.5 },
		)
	})?;
	assert_eq!(plan.diagnostics().semantic_operation_count(), 1);
	assert_eq!(plan.diagnostics().node_count(), 3);
	assert_eq!(
		plan.diagnostics().schema_owned_node_count(),
		plan.diagnostics().node_count()
	);
	let operation = &plan.semantic_graph().operations()[0];
	assert_eq!(operation.name(), "oa::ml::loss::dqn");
	assert_eq!(operation.attributes().len(), 1);
	engine.submit(&plan)?.wait()?;

	assert_close(&result.target_q.read_f32()?, &[11.0, 2.0, 33.0], 1.0e-6);
	assert_close(&result.selected_q.read_f32()?, &[1.0, 4.0, 5.0], 1.0e-6);
	assert_close(&result.loss.read_f32()?, &[38.5 / 3.0], 2.0e-6);

	tape.backward(&result.loss)?;
	assert_close(
		&q_parameter
			.weight()
			.gradient()
			.expect("DQN omitted the selected-Q gradient")
			.read_f32()?,
		&[-1.0 / 3.0, 0.0, 0.0, 1.0 / 3.0, -1.0 / 3.0, 0.0],
		2.0e-6,
	);
	assert!(
		next_q_parameter.weight().gradient().is_none(),
		"DQN target path must remain detached"
	);
	Ok(())
});

test_vk!(dqn_rejects_invalid_contracts, engine, {
	let q = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 2.0, 3.0, 4.0])?;
	let next_q = oa::Matrix::from_f32(&engine, [2, 2], &[4.0, 3.0, 2.0, 1.0])?;
	let action = oa::Matrix::from_slice(&engine, [2], &[0_i32, 1])?;
	let action_f32 = oa::Matrix::from_f32(&engine, [2], &[0.0, 1.0])?;
	let reward = oa::Matrix::from_f32(&engine, [2], &[1.0, 1.0])?;
	let boundary = oa::Matrix::from_slice(&engine, [2], &[0_u8, 0])?;

	for invalid in [
		dqn(
			&q,
			&action_f32,
			&reward,
			&next_q,
			&boundary,
			&boundary,
			DqnLossConfig::default(),
		),
		dqn(
			&q,
			&action,
			&reward,
			&next_q,
			&boundary,
			&boundary,
			DqnLossConfig { discount: -0.1 },
		),
		dqn(
			&q,
			&action,
			&reward,
			&next_q,
			&boundary,
			&boundary,
			DqnLossConfig { discount: f32::NAN },
		),
	] {
		let error = match invalid {
			Ok(_) => panic!("invalid DQN contract was accepted"),
			Err(error) => error,
		};
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	}
	Ok(())
});
