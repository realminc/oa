use oa::ml::environment::Environment;
use oa::sdk::ml::rl::{CartPole, CartPoleConfig};

#[derive(Clone, Copy)]
struct CpuStep {
	state: [f32; 4],
	reward: f32,
	terminated: u8,
	truncated: u8,
}

fn step_cpu(state: &[f32], action: i32, episode_steps: u32, config: CartPoleConfig) -> CpuStep {
	let [mut x, mut x_velocity, mut angle, mut angle_velocity] =
		<[f32; 4]>::try_from(state).expect("CartPole state width");
	let valid_action = action == 0 || action == 1;
	let force = if action == 1 {
		config.force_magnitude
	} else {
		-config.force_magnitude
	};
	let total_mass = config.cart_mass + config.pole_mass;
	let pole_mass_length = config.pole_mass * config.half_pole_length;
	let cosine = angle.cos();
	let sine = angle.sin();
	let temporary =
		(force + pole_mass_length * angle_velocity * angle_velocity * sine) / total_mass;
	let angle_acceleration = (config.gravity * sine - cosine * temporary)
		/ (config.half_pole_length * (4.0 / 3.0 - config.pole_mass * cosine * cosine / total_mass));
	let x_acceleration = temporary - pole_mass_length * angle_acceleration * cosine / total_mass;
	x += config.time_step * x_velocity;
	x_velocity += config.time_step * x_acceleration;
	angle += config.time_step * angle_velocity;
	angle_velocity += config.time_step * angle_acceleration;
	let state = [x, x_velocity, angle, angle_velocity];
	let invalid_state = state.into_iter().any(f32::is_nan);
	let terminated = (!valid_action
		|| invalid_state
		|| x.abs() > config.position_threshold
		|| angle.abs() > config.angle_threshold_radians) as u8;
	let truncated = (terminated == 0 && episode_steps + 1 >= config.max_episode_steps) as u8;
	CpuStep {
		state,
		reward: if valid_action { 1.0 } else { 0.0 },
		terminated,
		truncated,
	}
}

#[test]
fn dynamics_identity_covers_behavior_but_not_lane_or_seed_policy() {
	let reference = CartPoleConfig::default();
	let identity = reference.dynamics_identity();
	let changed = [
		CartPoleConfig {
			max_episode_steps: 501,
			..reference
		},
		CartPoleConfig {
			gravity: 9.81,
			..reference
		},
		CartPoleConfig {
			cart_mass: 1.01,
			..reference
		},
		CartPoleConfig {
			pole_mass: 0.11,
			..reference
		},
		CartPoleConfig {
			half_pole_length: 0.51,
			..reference
		},
		CartPoleConfig {
			force_magnitude: 10.1,
			..reference
		},
		CartPoleConfig {
			time_step: 0.021,
			..reference
		},
		CartPoleConfig {
			position_threshold: 2.5,
			..reference
		},
		CartPoleConfig {
			angle_threshold_radians: 0.21,
			..reference
		},
	];
	assert!(
		changed
			.into_iter()
			.all(|config| config.dynamics_identity() != identity)
	);
	assert_eq!(
		CartPoleConfig {
			environments: 7,
			..reference
		}
		.dynamics_identity(),
		identity
	);
	assert_eq!(
		CartPoleConfig {
			seed: 99,
			..reference
		}
		.dynamics_identity(),
		identity
	);
}

test_vk!(cart_pole_gpu_step_matches_scalar_cpu_oracle, engine, {
	let config = CartPoleConfig {
		environments: 4,
		seed: 0x1234_5678_9abc_def0,
		..CartPoleConfig::default()
	};
	let mut environment = CartPole::new(&engine, config)?;
	let initial_reset = environment.submit()?;
	environment.wait(&initial_reset)?;
	let initial = environment.observation().read_f32()?;
	let actions = [0_i32, 1, 1, 0];
	let action = oa::Matrix::from_slice(&engine, [4], &actions)?;
	let transition = environment.step(&action)?;
	let completion = environment.submit()?;
	environment.wait(&completion)?;

	assert_eq!(transition.observation().read_f32()?, initial);
	let actual_state = transition.next_observation().read_f32()?;
	let actual_reward = transition.reward().read_f32()?;
	let actual_terminated = transition.terminated().read::<u8>()?;
	let actual_truncated = transition.truncated().read::<u8>()?;
	let actual_done = environment.done().read::<u8>()?;
	assert_eq!(environment.episode_steps().read::<u32>()?, [1, 1, 1, 1]);
	for lane in 0..config.environments as usize {
		let expected = step_cpu(&initial[lane * 4..lane * 4 + 4], actions[lane], 0, config);
		for component in 0..4 {
			let actual = actual_state[lane * 4 + component];
			let expected = expected.state[component];
			assert!(
				(actual - expected).abs() <= 1.0e-5,
				"lane {lane} component {component}: {actual} != {expected}"
			);
		}
		assert_eq!(actual_reward[lane], expected.reward);
		assert_eq!(actual_terminated[lane], expected.terminated);
		assert_eq!(actual_truncated[lane], expected.truncated);
		assert_eq!(actual_done[lane], expected.terminated | expected.truncated);
	}
	environment.close()?;
	Ok(())
});

test_vk!(
	cart_pole_reset_and_done_lifecycle_is_deterministic,
	engine,
	{
		let config = CartPoleConfig {
			environments: 3,
			max_episode_steps: 1,
			seed: 918_273_645,
			..CartPoleConfig::default()
		};
		let mut environment = CartPole::new(&engine, config)?;
		let event = environment.submit()?;
		environment.wait(&event)?;
		let first_initial = environment.observation().read_f32()?;

		environment.reset(config.seed)?;
		let event = environment.submit()?;
		environment.wait(&event)?;
		assert_eq!(environment.observation().read_f32()?, first_initial);
		assert_eq!(environment.episode_index().read::<u32>()?, [0, 0, 0]);

		let actions = oa::Matrix::from_slice(&engine, [3], &[0_i32, 1, 0])?;
		environment.step(&actions)?;
		let event = environment.submit()?;
		environment.wait(&event)?;
		assert_eq!(environment.done().read::<u8>()?, [1, 1, 1]);
		assert_eq!(environment.episode_steps().read::<u32>()?, [1, 1, 1]);
		let terminal_state = environment.observation().read_f32()?;

		let ignored = environment.step(&actions)?;
		let event = environment.submit()?;
		environment.wait(&event)?;
		assert_eq!(environment.observation().read_f32()?, terminal_state);
		assert_eq!(ignored.reward().read_f32()?, [0.0, 0.0, 0.0]);
		assert_eq!(environment.episode_steps().read::<u32>()?, [1, 1, 1]);

		environment.reset_completed()?;
		let event = environment.submit()?;
		environment.wait(&event)?;
		assert_eq!(environment.done().read::<u8>()?, [0, 0, 0]);
		assert_eq!(environment.episode_steps().read::<u32>()?, [0, 0, 0]);
		assert_eq!(environment.episode_index().read::<u32>()?, [1, 1, 1]);
		assert_ne!(environment.observation().read_f32()?, first_initial);
		environment.close()?;
		Ok(())
	}
);

test_vk!(cart_pole_invalid_action_terminates_only_its_lane, engine, {
	let mut environment = CartPole::new(
		&engine,
		CartPoleConfig {
			environments: 2,
			seed: 7,
			..CartPoleConfig::default()
		},
	)?;
	let action = oa::Matrix::from_slice(&engine, [2], &[7_i32, 1])?;
	let transition = environment.step(&action)?;
	let event = environment.submit()?;
	environment.wait(&event)?;
	assert_eq!(transition.reward().read_f32()?, [0.0, 1.0]);
	assert_eq!(transition.terminated().read::<u8>()?, [1, 0]);
	assert_eq!(environment.done().read::<u8>()?, [1, 0]);
	environment.close()?;
	Ok(())
});

test_vk!(
	cart_pole_records_multiple_steps_in_one_ssa_transaction,
	engine,
	{
		let mut environment = CartPole::new(
			&engine,
			CartPoleConfig {
				environments: 2,
				seed: 11,
				..CartPoleConfig::default()
			},
		)?;
		let actions = oa::Matrix::from_slice(&engine, [2], &[0_i32, 1])?;
		environment.step(&actions)?;
		environment.reset_completed()?;
		environment.step(&actions)?;
		let event = environment.submit()?;
		environment.wait(&event)?;
		assert_eq!(environment.episode_steps().read::<u32>()?, [2, 2]);
		environment.close()?;
		Ok(())
	}
);
