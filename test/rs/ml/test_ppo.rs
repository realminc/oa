use oa::ml::loss::{PpoLossConfig, ppo, ppo_clipped_policy, ppo_clipped_policy_backward};

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= tolerance,
			"PPO mismatch at {index}: expected {expected}, found {actual}"
		);
	}
}

test_vk!(
	clipped_policy_and_explicit_adjoint_match_cpp_oracle,
	engine,
	{
		let ratios = [1.3_f32, 0.5, 1.1, 0.9];
		let new_log = oa::Matrix::from_f32(&engine, [4], &ratios.map(f32::ln))?;
		let old_log = oa::Matrix::from_f32(&engine, [4], &[0.0; 4])?;
		let advantage = oa::Matrix::from_f32(&engine, [4], &[1.0, -1.0, 2.0, -2.0])?;
		let (plan, loss) =
			engine.capture(|| ppo_clipped_policy(&new_log, &old_log, &advantage, 0.2))?;
		assert_eq!(plan.diagnostics().semantic_operation_count(), 1);
		assert_eq!(plan.diagnostics().node_count(), 3);
		assert_eq!(
			plan.semantic_graph().operations()[0].name(),
			"oa::ml::loss::ppo_clipped_policy"
		);
		engine.submit(&plan)?.wait()?;
		assert_close(&loss.read_f32()?, &[-0.2], 1.0e-6);
		assert_close(
			&ppo_clipped_policy_backward(&new_log, &old_log, &advantage, 0.2)?.read_f32()?,
			&[0.0, 0.0, -0.55, 0.45],
			1.0e-6,
		);
		Ok(())
	}
);

test_vk!(
	ppo_composition_and_reverse_match_independent_oracle,
	engine,
	{
		let new_log_parameter = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
			&engine,
			[1, 4],
			&[1.3_f32.ln(), 0.5_f32.ln(), 1.1_f32.ln(), 0.9_f32.ln()],
		)?)?;
		let value_parameter = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
			&engine,
			[1, 4],
			&[0.5, -0.5, 1.0, -1.0],
		)?)?;
		let entropy_parameter = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
			&engine,
			[1, 4],
			&[0.2, 0.4, 0.6, 0.8],
		)?)?;
		let row = oa::Matrix::from_slice(&engine, [1], &[0_u32])?;
		let old_log = oa::Matrix::from_f32(&engine, [1, 4], &[0.0; 4])?;
		let advantage = oa::Matrix::from_f32(&engine, [1, 4], &[1.0, -1.0, 2.0, -2.0])?;
		let target_return = oa::Matrix::from_f32(&engine, [1, 4], &[0.0; 4])?;
		let tape = oa::ml::GradientTape::new();
		let new_log = new_log_parameter.forward(&row)?;
		let value = value_parameter.forward(&row)?;
		let entropy = entropy_parameter.forward(&row)?;
		engine.checkpoint()?.wait()?;
		let config = PpoLossConfig::default();
		let (plan, result) = engine.capture(|| {
			ppo(
				&new_log,
				&old_log,
				&advantage,
				&value,
				&target_return,
				&entropy,
				config,
			)
		})?;
		assert_eq!(plan.diagnostics().semantic_operation_count(), 1);
		assert_eq!(
			plan.semantic_graph().operations()[0].name(),
			"oa::ml::loss::ppo"
		);
		assert_eq!(plan.semantic_graph().operations()[0].attributes().len(), 3);
		engine.submit(&plan)?.wait()?;
		assert_close(&result.policy_loss.read_f32()?, &[-0.2], 1.0e-6);
		assert_close(&result.value_loss.read_f32()?, &[0.625], 1.0e-6);
		assert_close(&result.entropy.read_f32()?, &[0.5], 1.0e-6);
		assert_close(&result.total_loss.read_f32()?, &[0.1075], 1.0e-6);

		tape.backward(&result.total_loss)?;
		assert_close(
			&new_log_parameter
				.weight()
				.gradient()
				.expect("PPO omitted policy gradient")
				.read_f32()?,
			&[0.0, 0.0, -0.55, 0.45],
			2.0e-6,
		);
		assert_close(
			&value_parameter
				.weight()
				.gradient()
				.expect("PPO omitted value gradient")
				.read_f32()?,
			&[0.125, -0.125, 0.25, -0.25],
			2.0e-6,
		);
		assert_close(
			&entropy_parameter
				.weight()
				.gradient()
				.expect("PPO omitted entropy gradient")
				.read_f32()?,
			&[-0.0025; 4],
			2.0e-6,
		);
		Ok(())
	}
);

test_vk!(ppo_rejects_invalid_contracts, engine, {
	let values = oa::Matrix::from_f32(&engine, [2], &[0.0; 2])?;
	let short = oa::Matrix::from_f32(&engine, [1], &[0.0])?;
	assert!(ppo_clipped_policy(&values, &short, &values, 0.2).is_err());
	assert!(ppo_clipped_policy(&values, &values, &values, 1.0).is_err());
	assert!(
		ppo(
			&values,
			&values,
			&values,
			&values,
			&values,
			&values,
			PpoLossConfig {
				value_coefficient: -1.0,
				..PpoLossConfig::default()
			},
		)
		.is_err()
	);
	Ok(())
});
