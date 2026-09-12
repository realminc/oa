use oa::ml::loss::{SacLossConfig, sac_actor, sac_critic};

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= tolerance,
			"SAC mismatch at {index}: expected {expected}, found {actual}"
		);
	}
}

fn parameter_vector(
	engine: &oa::Engine,
	values: &[f32],
) -> oa::Result<(oa::ml::nn::Embedding, oa::Matrix)> {
	let parameter = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
		engine,
		[values.len(), 1],
		values,
	)?)?;
	let rows = oa::Matrix::from_slice(
		engine,
		[values.len()],
		&(0..values.len() as u32).collect::<Vec<_>>(),
	)?;
	let vector = oa::matrix::reshape(&parameter.forward(&rows)?, vec![values.len()])?;
	Ok((parameter, vector))
}

test_vk!(
	sac_critic_matches_donor_target_and_detached_adjoint,
	engine,
	{
		let tape = oa::ml::GradientTape::new();
		let (q1_parameter, q1) = parameter_vector(&engine, &[1.0, 2.0, 3.0])?;
		let (q2_parameter, q2) = parameter_vector(&engine, &[1.5, 1.0, 4.0])?;
		let (next_q1_parameter, next_q1) = parameter_vector(&engine, &[10.0, 30.0, 50.0])?;
		let (next_q2_parameter, next_q2) = parameter_vector(&engine, &[20.0, 40.0, 60.0])?;
		let (next_log_parameter, next_log_probability) =
			parameter_vector(&engine, &[-1.0, -2.0, -3.0])?;
		let reward = oa::Matrix::from_f32(&engine, [3], &[1.0, 2.0, 3.0])?;
		let terminated = oa::Matrix::from_slice(&engine, [3], &[0_u8, 1, 0])?;
		let truncated = oa::Matrix::from_slice(&engine, [3], &[0_u8, 0, 1])?;
		engine.checkpoint()?.wait()?;

		let (plan, result) = engine.capture(|| {
			sac_critic(
				&q1,
				&q2,
				&reward,
				&next_q1,
				&next_q2,
				&next_log_probability,
				&terminated,
				&truncated,
				SacLossConfig {
					discount: 0.5,
					entropy_coefficient: 0.2,
				},
			)
		})?;
		assert_eq!(plan.diagnostics().semantic_operation_count(), 1);
		assert_eq!(plan.diagnostics().node_count(), 8);
		assert_eq!(
			plan.diagnostics().schema_owned_node_count(),
			plan.diagnostics().node_count()
		);
		let operation = &plan.semantic_graph().operations()[0];
		assert_eq!(operation.name(), "oa::ml::loss::sac_critic");
		assert_eq!(operation.attributes().len(), 2);
		engine.submit(&plan)?.wait()?;

		assert_close(&result.target_q.read_f32()?, &[6.1, 2.0, 28.3], 1.0e-5);
		assert_close(&result.q1_loss.read_f32()?, &[666.1 / 3.0], 2.0e-5);
		assert_close(&result.q2_loss.read_f32()?, &[612.65 / 3.0], 2.0e-5);
		assert_close(&result.total_loss.read_f32()?, &[426.25], 3.0e-5);

		tape.backward(&result.total_loss)?;
		assert_close(
			&q1_parameter
				.weight()
				.gradient()
				.expect("SAC omitted first critic gradient")
				.read_f32()?,
			&[-3.4, 0.0, -50.6 / 3.0],
			3.0e-5,
		);
		assert_close(
			&q2_parameter
				.weight()
				.gradient()
				.expect("SAC omitted second critic gradient")
				.read_f32()?,
			&[-9.2 / 3.0, -2.0 / 3.0, -16.2],
			3.0e-5,
		);
		for parameter in [next_q1_parameter, next_q2_parameter, next_log_parameter] {
			assert!(
				parameter.weight().gradient().is_none(),
				"SAC critic target path must remain detached"
			);
		}
		Ok(())
	}
);

test_vk!(
	sac_actor_matches_donor_value_and_complete_adjoint,
	engine,
	{
		let tape = oa::ml::GradientTape::new();
		let (q1_parameter, q1) = parameter_vector(&engine, &[2.0, 4.0])?;
		let (q2_parameter, q2) = parameter_vector(&engine, &[3.0, 1.0])?;
		let (log_parameter, log_probability) = parameter_vector(&engine, &[-0.5, -1.0])?;
		engine.checkpoint()?.wait()?;
		let (plan, loss) = engine.capture(|| sac_actor(&q1, &q2, &log_probability, 0.2))?;
		assert_eq!(plan.diagnostics().semantic_operation_count(), 1);
		assert!(plan.diagnostics().node_count() > 1);
		assert_eq!(
			plan.diagnostics().schema_owned_node_count(),
			plan.diagnostics().node_count()
		);
		assert_eq!(
			plan.semantic_graph().operations()[0].name(),
			"oa::ml::loss::sac_actor"
		);
		engine.submit(&plan)?.wait()?;
		assert_close(&loss.read_f32()?, &[-1.65], 1.0e-6);

		tape.backward(&loss)?;
		assert_close(
			&q1_parameter
				.weight()
				.gradient()
				.expect("SAC actor omitted first Q gradient")
				.read_f32()?,
			&[-0.5, 0.0],
			2.0e-6,
		);
		assert_close(
			&q2_parameter
				.weight()
				.gradient()
				.expect("SAC actor omitted second Q gradient")
				.read_f32()?,
			&[0.0, -0.5],
			2.0e-6,
		);
		assert_close(
			&log_parameter
				.weight()
				.gradient()
				.expect("SAC actor omitted log-probability gradient")
				.read_f32()?,
			&[0.1, 0.1],
			2.0e-6,
		);
		Ok(())
	}
);

test_vk!(sac_losses_reject_invalid_contracts, engine, {
	let vector = oa::Matrix::from_f32(&engine, [2], &[1.0, 2.0])?;
	let short = oa::Matrix::from_f32(&engine, [1], &[1.0])?;
	let boundary = oa::Matrix::from_slice(&engine, [2], &[0_u8, 0])?;
	assert!(sac_actor(&vector, &short, &vector, 0.2).is_err());
	assert!(sac_actor(&vector, &vector, &vector, -0.1).is_err());
	assert!(
		sac_critic(
			&vector,
			&vector,
			&vector,
			&vector,
			&vector,
			&vector,
			&boundary,
			&boundary,
			SacLossConfig {
				discount: 1.1,
				..SacLossConfig::default()
			},
		)
		.is_err()
	);
	Ok(())
});
