use oa::ml::{EnvironmentSpace, EnvironmentSpaceKind, EnvironmentSpec, EnvironmentTransition};

fn cart_pole_spec() -> oa::Result<EnvironmentSpec> {
	EnvironmentSpec::new(
		EnvironmentSpace::continuous(
			"observation",
			[4],
			oa::DType::F32,
			f64::NEG_INFINITY,
			f64::INFINITY,
		)?,
		EnvironmentSpace::discrete("action", 2, oa::DType::I32)?,
		EnvironmentSpace::continuous(
			"reward",
			[],
			oa::DType::F32,
			f64::NEG_INFINITY,
			f64::INFINITY,
		)?,
		EnvironmentSpace::binary("terminated", [])?,
		EnvironmentSpace::binary("truncated", [])?,
	)
}

#[test]
fn environment_spaces_preserve_donor_shape_and_dtype_contracts() -> oa::Result<()> {
	let observation =
		EnvironmentSpace::continuous("observation", [4], oa::DType::F32, -10.0, 10.0)?;
	assert_eq!(observation.kind(), EnvironmentSpaceKind::Box);
	assert_eq!(observation.name(), "observation");
	assert_eq!(observation.shape(), [4]);
	assert_eq!(observation.elements_per_environment(), 4);
	assert_eq!(observation.batched_shape(8)?, [8, 4]);
	let action = EnvironmentSpace::discrete("action", 3, oa::DType::U32)?;
	assert_eq!(action.kind(), EnvironmentSpaceKind::Discrete);
	assert!(action.shape().is_empty());
	assert_eq!(action.minimum(), 0.0);
	assert_eq!(action.maximum(), 2.0);
	assert_eq!(action.cardinality(), 3);
	let terminated = EnvironmentSpace::binary("terminated", [])?;
	assert_eq!(terminated.dtype(), oa::DType::U8);
	assert_eq!(terminated.batched_shape(8)?, [8]);

	for invalid in [
		EnvironmentSpace::continuous("", [4], oa::DType::F32, -1.0, 1.0),
		EnvironmentSpace::continuous("box", [0], oa::DType::F32, -1.0, 1.0),
		EnvironmentSpace::continuous("box", [4], oa::DType::I32, -1.0, 1.0),
		EnvironmentSpace::continuous("box", [4], oa::DType::F32, f64::NAN, 1.0),
		EnvironmentSpace::continuous("box", [4], oa::DType::F32, 2.0, 1.0),
		EnvironmentSpace::discrete("action", 0, oa::DType::I32),
		EnvironmentSpace::discrete("action", 2, oa::DType::F32),
	] {
		assert_eq!(
			invalid
				.expect_err("invalid environment space was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	assert_eq!(
		observation
			.batched_shape(0)
			.expect_err("zero environment count was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	let shaped_reward = EnvironmentSpace::continuous(
		"reward",
		[1],
		oa::DType::F32,
		f64::NEG_INFINITY,
		f64::INFINITY,
	)?;
	assert_eq!(
		EnvironmentSpec::new(
			observation,
			action,
			shaped_reward,
			terminated.clone(),
			terminated,
		)
		.expect_err("vector reward was accepted")
		.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
}

test_vk!(
	environment_spec_validates_complete_batched_transition,
	engine,
	{
		let spec = cart_pole_spec()?;
		let observation = oa::Matrix::from_f32(&engine, [3, 4], &[0.0; 12])?;
		let next_observation = oa::Matrix::from_f32(&engine, [3, 4], &[0.1; 12])?;
		let action = oa::Matrix::from_slice(&engine, [3], &[0_i32, 1, 0])?;
		let reward = oa::Matrix::from_f32(&engine, [3], &[1.0; 3])?;
		let terminated = oa::Matrix::from_slice(&engine, [3], &[0_u8, 1, 0])?;
		let truncated = oa::Matrix::from_slice(&engine, [3], &[0_u8, 0, 1])?;
		let transition = EnvironmentTransition::new(
			observation.clone(),
			next_observation,
			reward,
			terminated,
			truncated,
		);
		spec.validate_reset(&observation, 3)?;
		spec.validate_action(&action, 3)?;
		spec.validate_transition(&action, &transition, 3)?;
		assert_eq!(transition.observation().shape(), [3, 4]);
		assert_eq!(transition.next_observation().shape(), [3, 4]);
		assert_eq!(transition.reward().shape(), [3]);
		assert_eq!(transition.terminated().read::<u8>()?, [0, 1, 0]);
		assert_eq!(transition.truncated().read::<u8>()?, [0, 0, 1]);

		let wrong_action = oa::Matrix::from_slice(&engine, [3], &[0_u32, 1, 0])?;
		assert_eq!(
			spec.validate_action(&wrong_action, 3)
				.expect_err("wrong action dtype was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		let wrong_observation = oa::Matrix::from_f32(&engine, [2, 4], &[0.0; 8])?;
		assert_eq!(
			spec.validate_reset(&wrong_observation, 3)
				.expect_err("wrong observation batch was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		Ok(())
	}
);
