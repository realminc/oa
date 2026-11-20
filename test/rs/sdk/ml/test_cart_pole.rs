use oa::ml::environment::Environment;
use sdk::ml::rl::{CartPole, CartPoleConfig};

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
	let temporary = (force + pole_mass_length * angle_velocity * angle_velocity * sine) / total_mass;
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
	let action = oa::Matrix::new_from_slice(&engine, [4], &actions)?;
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

		let actions = oa::Matrix::new_from_slice(&engine, [3], &[0_i32, 1, 0])?;
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
	let action = oa::Matrix::new_from_slice(&engine, [2], &[7_i32, 1])?;
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
		let actions = oa::Matrix::new_from_slice(&engine, [2], &[0_i32, 1])?;
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

test_vk!(
	cart_pole_cursor_replay_matches_eager_reset_and_step,
	engine,
	{
		use oa::ml::environment::CompiledEnvironment;
		let seed = 0x9e37_79b9_85eb_ca6b_u64;
		let config = CartPoleConfig {
			environments: 4,
			max_episode_steps: 500,
			..CartPoleConfig::default()
		};
		// One environment transaction owns the engine capture slot at a time:
		// submit and wait each construction reset before opening the next.
		let mut eager = CartPole::new(&engine, CartPoleConfig { seed, ..config })?;
		let event = eager.submit()?;
		eager.wait(&event)?;
		let mut replay = CartPole::new(&engine, CartPoleConfig { seed: 0, ..config })?;
		let event = replay.submit()?;
		replay.wait(&event)?;
		let cursor = oa::Matrix::new_from_slice(
			&engine,
			[7],
			&[0_u32, 1, 0, seed as u32, (seed >> 32) as u32, 0, 0],
		)?;

		// Full reset through the captured cursor program is bit-exact with the
		// eager push-seeded reset under the same collection seed.
		let (plan, ()) = engine.capture(|| replay.record_reset_replay(false, &cursor))?;
		let event = engine.submit(&plan)?;
		event.wait()?;
		assert_eq!(
			replay.observation().read_f32()?,
			eager.observation().read_f32()?
		);

		// A captured deterministic step replays bit-exact against the eager step;
		// lane 0's invalid action terminates it for the completed-reset check.
		let action = oa::Matrix::new_from_slice(&engine, [4], &[5_i32, 1, 0, 1])?;
		let expected = eager.step(&action)?;
		let event = eager.submit()?;
		eager.wait(&event)?;
		let (plan, step) = engine.capture(|| replay.record_step_replay(&action, &cursor))?;
		let event = engine.submit(&plan)?;
		event.wait()?;
		assert_eq!(
			step.observation().read_f32()?,
			expected.observation().read_f32()?
		);
		assert_eq!(
			replay.observation().read_f32()?,
			eager.observation().read_f32()?
		);
		assert_eq!(step.reward().read_f32()?, expected.reward().read_f32()?);
		assert_eq!(
			step.terminated().read::<u8>()?,
			expected.terminated().read::<u8>()?
		);
		assert_eq!(replay.done().read::<u8>()?, eager.done().read::<u8>()?);

		// Completed-lane reset through the captured cursor program restores only
		// the done lane with its advanced episode index, bit-exact with eager.
		eager.reset_completed()?;
		let event = eager.submit()?;
		eager.wait(&event)?;
		let (plan, ()) = engine.capture(|| replay.record_reset_replay(true, &cursor))?;
		let event = engine.submit(&plan)?;
		event.wait()?;
		assert_eq!(
			replay.observation().read_f32()?,
			eager.observation().read_f32()?
		);
		assert_eq!(
			replay.episode_index().read::<u32>()?,
			eager.episode_index().read::<u32>()?
		);
		assert_eq!(replay.done().read::<u8>()?, eager.done().read::<u8>()?);
		eager.close()?;
		replay.close()?;
		Ok(())
	}
);

test_vk!(
	cart_pole_compiled_collect_matches_eager_bit_exact,
	engine,
	{
		use oa::ml::training::{PpoTrainer, PpoTrainerConfig};
		use oa::ml::{
			AdamW, CategoricalActorCritic, CategoricalActorCriticConfig, CompiledRollout, Module,
		};
		const ENVS: u32 = 4;
		const HORIZON: usize = 4;
		const SEED: u64 = 0x5eed_5eed;
		let mut batches = Vec::new();
		for compiled in [false, true] {
			let model = CategoricalActorCritic::new(
				&engine,
				CategoricalActorCriticConfig {
					observation_size: 4,
					action_count: 2,
					hidden_size: 8,
					seed: 77,
				},
			)?;
			let mut optimizer = AdamW::new(model.all_parameters()?, 0.001)?;
			let mut environment = CartPole::new(
				&engine,
				CartPoleConfig {
					environments: ENVS,
					seed: SEED,
					..CartPoleConfig::default()
				},
			)?;
			let event = environment.submit()?;
			environment.wait(&event)?;
			let config = PpoTrainerConfig {
				rollouts: 1,
				horizon: HORIZON,
				environments: ENVS as usize,
				update_epochs: 1,
				observation_shape: vec![4],
				seed: SEED,
				..PpoTrainerConfig::default()
			};
			let mut trainer = PpoTrainer::new(&engine, &model, &mut optimizer, config)?;
			if compiled {
				let mut program = CompiledRollout::new();
				assert!(program.diagnostics().is_none());
				trainer.collect_compiled(&mut environment, &mut program, SEED, 0, 0.0)?;
				assert_eq!(program.replay_count(), HORIZON as u64);
				assert_eq!(program.command_recording_count(), 3);
				let diagnostics = program
					.diagnostics()
					.expect("compiled collection exposes captured plan diagnostics");
				assert!(diagnostics.step().node_count() > 0);
				assert!(diagnostics.step().peak_memory_bytes() > 0);
				assert!(diagnostics.gae().node_count() > 0);
				assert!(diagnostics.resume().is_none());
			} else {
				trainer.begin_collection()?;
				engine.checkpoint()?.wait()?;
				environment.reset(SEED)?;
				for _ in 0..HORIZON {
					let observation = environment.observation().clone();
					let policy = trainer.act(&observation)?;
					let transition = environment.step(&policy.action)?;
					trainer.observe(
						transition.observation(),
						transition.next_observation(),
						transition.reward(),
						transition.terminated(),
						transition.truncated(),
						&policy,
					)?;
					environment.reset_completed()?;
				}
				trainer.end_collection()?;
				let event = environment.submit()?;
				environment.wait(&event)?;
			}
			let batch = trainer.batch();
			let actions = batch.action().read::<i32>()?;
			let floats = [
				batch.observation(),
				batch.reward(),
				batch.value(),
				batch.next_value(),
				batch.old_log_probability(),
				batch.advantage(),
				batch.returns(),
			]
			.iter()
			.map(|matrix| matrix.read_f32())
			.collect::<oa::Result<Vec<_>>>()?;
			batches.push((actions, floats));
			environment.close()?;
		}
		assert_eq!(batches[0], batches[1]);
		Ok(())
	}
);

test_vk!(
	cart_pole_compiled_continuing_matches_eager_across_chunks,
	engine,
	{
		use oa::ml::training::{PpoTrainer, PpoTrainerConfig};
		use oa::ml::{
			AdamW, CategoricalActorCritic, CategoricalActorCriticConfig, CompiledRollout, Module,
		};
		const ENVS: u32 = 4;
		const HORIZON: usize = 3;
		const SEED: u64 = 0x0c0f_fee1;
		let mut outcomes = Vec::new();
		for compiled in [false, true] {
			let mut environment = CartPole::new(
				&engine,
				CartPoleConfig {
					environments: ENVS,
					seed: SEED,
					..CartPoleConfig::default()
				},
			)?;
			let event = environment.submit()?;
			environment.wait(&event)?;
			let model = CategoricalActorCritic::new(
				&engine,
				CategoricalActorCriticConfig {
					observation_size: 4,
					action_count: 2,
					hidden_size: 8,
					seed: 77,
				},
			)?;
			let mut optimizer = AdamW::new(model.all_parameters()?, 0.001)?;
			let config = PpoTrainerConfig {
				rollouts: 2,
				horizon: HORIZON,
				environments: ENVS as usize,
				update_epochs: 1,
				observation_shape: vec![4],
				seed: SEED,
				..PpoTrainerConfig::default()
			};
			let mut trainer = PpoTrainer::new(&engine, &model, &mut optimizer, config)?;
			let mut program = CompiledRollout::new();
			let mut chunks = Vec::new();
			for chunk in 0..2_u64 {
				if compiled {
					trainer.collect_compiled_continuing(&mut environment, &mut program, SEED, 0, 0.0)?;
				} else {
					trainer.begin_collection()?;
					engine.checkpoint()?.wait()?;
					if chunk == 0 {
						environment.reset(SEED)?;
					} else {
						// Continuing collection resumes on carried state; opening
						// the transaction mirrors the compiled resume plan, which
						// touches rollout storage only.
						oa::ml::environment::Environment::begin(&mut environment)?;
					}
					for _ in 0..HORIZON {
						let observation = environment.observation().clone();
						let policy = trainer.act(&observation)?;
						let transition = environment.step(&policy.action)?;
						trainer.observe(
							transition.observation(),
							transition.next_observation(),
							transition.reward(),
							transition.terminated(),
							transition.truncated(),
							&policy,
						)?;
						environment.reset_completed()?;
					}
					trainer.end_collection()?;
					let event = environment.submit()?;
					environment.wait(&event)?;
				}
				let batch = trainer.batch();
				chunks.push((
					batch.action().read::<i32>()?,
					[
						batch.observation(),
						batch.reward(),
						batch.value(),
						batch.next_value(),
						batch.old_log_probability(),
						batch.advantage(),
						batch.returns(),
					]
					.iter()
					.map(|matrix| matrix.read_f32())
					.collect::<oa::Result<Vec<_>>>()?,
				));
				trainer.update()?;
			}
			if compiled {
				// Two chunks replay the captured step twice each; the resume plan
				// joins the reset/step/GAE recordings after the first chunk.
				assert_eq!(program.replay_count(), (HORIZON * 2) as u64);
				assert_eq!(program.command_recording_count(), 3);
				assert!(
					program
						.diagnostics()
						.expect("captured plans")
						.resume()
						.is_some()
				);
			}
			outcomes.push(chunks);
			environment.close()?;
		}
		assert_eq!(outcomes[0], outcomes[1]);
		Ok(())
	}
);
