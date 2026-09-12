use oa::sdk::ml::rl::lunar_lander::{integrate, observe, scripted_landing_action};
use oa::sdk::ml::rl::{
	LUNAR_OBSERVATION_SIZE, LunarAction, LunarEpisodeManifest, LunarFirstEpisodeEvaluationConfig,
	LunarLander3dConfig, LunarLander3dState, LunarLander3dVector, LunarLander3dVectorConfig,
	LunarRandomPurpose, LunarRewardTerms, LunarScalarEnvironment, LunarTeacherConfig, LunarTerrain,
	LunarTerrainConfig, LunarTerrainTriangle, collect_ppo_rollout, evaluate_first_episodes,
	pretrain_scripted_teacher,
};
use oa::{ErrorKind, Matrix, ml::environment::Environment};

fn assert_f32_close(actual: f32, expected: f32, tolerance: f32, context: &str) {
	assert!(
		(actual - expected).abs() <= tolerance,
		"{context}: {actual} != {expected} (tolerance {tolerance})"
	);
}

fn result_error_kind<T>(result: oa::Result<T>) -> ErrorKind {
	match result {
		Ok(_) => panic!("operation unexpectedly succeeded"),
		Err(error) => error.kind(),
	}
}

fn lunar_observation_tolerance(component: usize) -> f32 {
	match component {
		0..=5 => 8.0e-4,
		6..=14 => 1.5e-3,
		15..=27 => 2.0e-3,
		28..=31 => 0.0,
		32 => 8.0e-5,
		_ => unreachable!("Lunar observation component is schema-bounded"),
	}
}

fn assert_lunar_observation_close(
	actual: &[f32],
	expected: &[f32; LUNAR_OBSERVATION_SIZE],
	lane: usize,
	step: u32,
) {
	for (component, expected) in expected.iter().copied().enumerate() {
		let actual = actual[component];
		let tolerance = lunar_observation_tolerance(component);
		if tolerance == 0.0 {
			assert_eq!(
				actual, expected,
				"lane {lane} step {step} component {component}"
			);
		} else {
			assert_f32_close(
				actual,
				expected,
				tolerance,
				&format!("lane {lane} step {step} observation {component}"),
			);
		}
	}
}

fn assert_lunar_contact_observation_close(
	actual: &[f32],
	expected: &[f32; LUNAR_OBSERVATION_SIZE],
	lane: usize,
	step: u32,
) {
	for (component, expected) in expected.iter().copied().enumerate() {
		let actual = actual[component];
		let tolerance = match component {
			0..=2 => 3.0e-3,
			3..=5 => 1.5e-2,
			6..=14 => 3.0e-2,
			15..=27 => 2.0e-2,
			28..=31 => 0.0,
			32 => 1.0e-3,
			_ => unreachable!("Lunar observation component is schema-bounded"),
		};
		if tolerance == 0.0 {
			assert_eq!(
				actual, expected,
				"lane {lane} step {step} component {component}"
			);
		} else {
			assert_f32_close(
				actual,
				expected,
				tolerance,
				&format!("lane {lane} step {step} contact observation {component}"),
			);
		}
	}
}

fn run_lunar_vector_episode_differential(
	engine: &oa::Engine,
	environment_config: LunarLander3dConfig,
	seed: u64,
	scripted: bool,
	expected_reason: oa::sdk::ml::rl::LunarEndReason,
) -> oa::Result<()> {
	const ENVIRONMENTS: usize = 4;
	let config = LunarLander3dVectorConfig {
		environments: ENVIRONMENTS as u32,
		seed,
		environment: environment_config,
	};
	let mut vector = LunarLander3dVector::flat(engine, config)?;
	let event = vector.submit()?;
	vector.wait(&event)?;
	let mut scalar = (0..ENVIRONMENTS)
		.map(|lane| {
			LunarScalarEnvironment::flat(
				environment_config,
				LunarEpisodeManifest::derive(
					seed,
					lane as u32,
					0,
					environment_config.contract_fingerprint(),
				),
			)
		})
		.collect::<oa::Result<Vec<_>>>()?;
	let mut contact_phase = [false; ENVIRONMENTS];
	let mut completed = false;
	for step_index in 0..environment_config.max_episode_steps {
		let mut actions = [0_i32; ENVIRONMENTS];
		let mut expected_observations = [[0.0_f32; LUNAR_OBSERVATION_SIZE]; ENVIRONMENTS];
		let mut expected_rewards = [0.0_f32; ENVIRONMENTS];
		let mut expected_terminated = [false; ENVIRONMENTS];
		let mut expected_truncated = [false; ENVIRONMENTS];
		let mut expected_reasons = [oa::sdk::ml::rl::LunarEndReason::None; ENVIRONMENTS];
		for lane in 0..ENVIRONMENTS {
			let state = *scalar[lane].state();
			if state.terminated || state.truncated {
				expected_observations[lane] = scalar[lane].observation();
				expected_terminated[lane] = state.terminated;
				expected_truncated[lane] = state.truncated;
				expected_reasons[lane] = state.end_reason;
				continue;
			}
			let action = if scripted {
				scripted_landing_action(&environment_config, &state)
			} else {
				LunarAction::Coast
			};
			actions[lane] = action as i32;
			let transition = scalar[lane].step(action as u32, false)?;
			expected_observations[lane] = transition.observation;
			expected_rewards[lane] = transition.reward as f32;
			expected_terminated[lane] = transition.terminated;
			expected_truncated[lane] = transition.truncated;
			expected_reasons[lane] = transition.end_reason;
			contact_phase[lane] |= transition.contact.foot_contact_occurred
				|| transition.contact.body_contact_occurred;
		}

		let action = Matrix::from_slice(engine, [ENVIRONMENTS], &actions)?;
		let transition = vector.step(&action)?;
		let event = vector.submit()?;
		vector.wait(&event)?;
		let observations = transition.next_observation.read_f32()?;
		let rewards = transition.reward.read_f32()?;
		let terminated = transition.terminated.read::<u8>()?;
		let truncated = transition.truncated.read::<u8>()?;
		let reasons = transition.end_reason.read::<u32>()?;
		for lane in 0..ENVIRONMENTS {
			let begin = lane * LUNAR_OBSERVATION_SIZE;
			let actual = &observations[begin..begin + LUNAR_OBSERVATION_SIZE];
			if contact_phase[lane] {
				assert_lunar_contact_observation_close(
					actual,
					&expected_observations[lane],
					lane,
					step_index + 1,
				);
			} else {
				assert_lunar_observation_close(
					actual,
					&expected_observations[lane],
					lane,
					step_index + 1,
				);
			}
			assert_f32_close(
				rewards[lane],
				expected_rewards[lane],
				6.0e-3,
				&format!("lane {lane} step {step_index} reward"),
			);
			assert_eq!(terminated[lane], u8::from(expected_terminated[lane]));
			assert_eq!(truncated[lane], u8::from(expected_truncated[lane]));
			assert_eq!(reasons[lane], expected_reasons[lane] as u32);
		}
		completed = scalar
			.iter()
			.all(|lane| lane.state().terminated || lane.state().truncated);
		if completed {
			break;
		}
	}
	assert!(completed, "all scalar lanes must reach a boundary");
	for (lane, (actual, expected)) in vector
		.copy_episode_telemetry()?
		.into_iter()
		.zip(scalar.iter().map(LunarScalarEnvironment::state))
		.enumerate()
	{
		assert_eq!(actual.end_reason, expected_reason, "lane {lane}");
		assert_eq!(actual.end_reason, expected.end_reason, "lane {lane}");
		assert_eq!(actual.terminated, expected.terminated, "lane {lane}");
		assert_eq!(actual.truncated, expected.truncated, "lane {lane}");
		assert_eq!(actual.episode_step, expected.episode_step, "lane {lane}");
		let return_tolerance = 0.2_f32.max((expected.episode_return.abs() as f32) * 2.0e-3);
		assert_f32_close(
			actual.episode_return,
			expected.episode_return as f32,
			return_tolerance,
			&format!("lane {lane} episode return"),
		);
		assert_f32_close(
			actual.fuel_remaining,
			expected.fuel as f32,
			5.0e-2,
			&format!("lane {lane} fuel"),
		);
	}
	vector.close()?;
	Ok(())
}

fn flight_state(config: LunarLander3dConfig, height: f64) -> LunarLander3dState {
	LunarLander3dState {
		position: oa::vlm::DVec3 {
			x: 0.0,
			y: height,
			z: 0.0,
		},
		fuel: config.fuel_capacity,
		..LunarLander3dState::default()
	}
}

fn manifest_for(config: LunarLander3dConfig) -> LunarEpisodeManifest {
	LunarEpisodeManifest::derive(0x1234_5678_9abc_def0, 3, 7, config.contract_fingerprint())
}

struct LunarTraceDigest(u64);

impl LunarTraceDigest {
	fn new() -> Self {
		Self(1_469_598_103_934_665_603)
	}

	fn u64(&mut self, value: u64) {
		for byte in value.to_le_bytes() {
			self.0 ^= u64::from(byte);
			self.0 = self.0.wrapping_mul(1_099_511_628_211);
		}
	}

	fn u32(&mut self, value: u32) {
		self.u64(u64::from(value));
	}

	fn bool(&mut self, value: bool) {
		self.u64(u64::from(value));
	}

	fn f64(&mut self, value: f64) {
		self.u64((value * 1.0e9).round() as i64 as u64);
	}

	fn f32(&mut self, value: f32) {
		self.u64((f64::from(value) * 1.0e6).round() as i64 as u64);
	}
}

fn digest_vec3(digest: &mut LunarTraceDigest, value: oa::vlm::DVec3) {
	digest.f64(value.x);
	digest.f64(value.y);
	digest.f64(value.z);
}

fn digest_state(digest: &mut LunarTraceDigest, state: LunarLander3dState) {
	digest_vec3(digest, state.position);
	digest_vec3(digest, state.linear_velocity);
	digest.f64(state.orientation.w);
	digest.f64(state.orientation.x);
	digest.f64(state.orientation.y);
	digest.f64(state.orientation.z);
	digest_vec3(digest, state.angular_velocity_body);
	digest.f64(state.fuel);
	digest.u32(state.last_action as u32);
	digest.f64(state.main_throttle);
	digest_vec3(digest, state.attitude_command_body);
	for value in state.body_contacts {
		digest.bool(value);
	}
	for value in state.body_contact_impulses {
		digest.f64(value);
	}
	for value in state.foot_contacts {
		digest.bool(value);
	}
	for value in state.feet_on_pad {
		digest.bool(value);
	}
	for value in state.foot_contact_impulses {
		digest.f64(value);
	}
	for value in state.foot_contact_rewarded {
		digest.bool(value);
	}
	digest.u32(state.episode_step);
	digest.u32(state.stable_dwell);
	digest.bool(state.terminated);
	digest.bool(state.truncated);
	digest.u32(state.end_reason as u32);
	digest.f64(state.episode_return);
}

fn digest_reward(digest: &mut LunarTraceDigest, reward: LunarRewardTerms) {
	for value in [
		reward.potential_before,
		reward.potential_after,
		reward.shaping,
		reward.main_fuel_cost,
		reward.attitude_fuel_cost,
		reward.soft_foot_contact,
		reward.stable_dwell,
		reward.terminal,
		reward.total,
	] {
		digest.f64(value);
	}
}

fn digest_manifest(digest: &mut LunarTraceDigest, manifest: LunarEpisodeManifest) {
	digest.u32(manifest.environment_version());
	digest.u32(manifest.random_version());
	digest.u32(manifest.terrain_version());
	digest.u32(manifest.physics_version());
	digest.u32(manifest.observation_version());
	digest.u32(manifest.reward_version());
	digest.u64(manifest.config_fingerprint());
	digest.u64(manifest.base_seed());
	digest.u32(manifest.environment_lane());
	digest.u64(manifest.episode_index());
	digest.u64(manifest.terrain_seed());
	digest.u64(manifest.spawn_seed());
	digest.u64(manifest.domain_seed());
}

#[test]
fn lunar_manifest_matches_donor_seed_vectors() -> oa::Result<()> {
	let manifest = LunarEpisodeManifest::derive(0x1234_5678_9abc_def0, 3, 7, 0);
	manifest.validate()?;
	assert_eq!(manifest.terrain_seed(), 0x87b6_d5a4_24c9_738b);
	assert_eq!(manifest.spawn_seed(), 0xdb0f_56e4_7eab_3480);
	assert_eq!(manifest.domain_seed(), 0x2ce2_d29d_6f07_62e7);
	assert_eq!(
		manifest.sample_01(LunarRandomPurpose::Spawn, 11),
		0.109_155_029_381_090_56
	);
	assert_ne!(
		LunarEpisodeManifest::derive(manifest.base_seed(), 4, manifest.episode_index(), 0),
		manifest
	);
	assert!(LunarAction::try_from(7).is_ok());
	assert!(LunarAction::try_from(8).is_err());
	Ok(())
}

#[test]
fn lunar_manifest_rejects_unsupported_persisted_versions() {
	let unsupported = LunarEpisodeManifest::derive_versioned(1, 0, 0, 2, 1, 1, 1, 1, 0);
	assert!(unsupported.validate().is_err());
}

#[test]
fn lunar_terrain_freezes_vertices_edges_diagonals_and_normals() -> oa::Result<()> {
	let config = LunarTerrainConfig {
		cells_x: 2,
		cells_z: 2,
		cell_size: 1.0,
		max_abs_height: 10.0,
		max_slope: 20.0,
		pad_half_extent: 0.0,
		pad_transition_width: 0.0,
	};
	let terrain =
		LunarTerrain::from_heights(config, vec![0.0, 1.0, 4.0, 2.0, 4.0, 8.0, 3.0, 6.0, 9.0])?;
	let vertex = terrain.query(0.0, 0.0);
	assert_eq!((vertex.cell_x, vertex.cell_z), (1, 1));
	assert_eq!(
		(vertex.local_x, vertex.local_z, vertex.height),
		(0.0, 0.0, 4.0)
	);
	let edge = terrain.query(0.0, -0.5);
	assert_eq!(edge.triangle, LunarTerrainTriangle::UpperLeft);
	assert_eq!(edge.height, 2.5);
	let diagonal = terrain.query(-0.5, -0.5);
	assert_eq!(diagonal.triangle, LunarTerrainTriangle::LowerRight);
	assert_eq!(diagonal.height, 2.0);
	let lower = terrain.query(-0.25, -0.75);
	let upper = terrain.query(-0.75, -0.25);
	assert_eq!(lower.height, 1.5);
	assert_eq!(upper.height, 2.0);
	let scale = 1.0 / 11.0_f64.sqrt();
	assert!((lower.normal.x + scale).abs() <= 1.0e-15);
	assert!((lower.normal.y - scale).abs() <= 1.0e-15);
	assert!((lower.normal.z + 3.0 * scale).abs() <= 1.0e-15);
	assert!((upper.normal.x + 2.0 / 3.0).abs() <= 1.0e-15);
	assert!((upper.normal.y - 1.0 / 3.0).abs() <= 1.0e-15);
	assert!((upper.normal.z + 2.0 / 3.0).abs() <= 1.0e-15);
	assert_eq!(terrain.query(1.0, 1.0).height, 9.0);
	assert!(!terrain.query(terrain.min_x() - 0.01, 0.0).in_bounds);
	Ok(())
}

#[test]
fn seeded_lunar_terrain_is_bounded_deterministic_and_pad_flat() -> oa::Result<()> {
	let config = LunarTerrainConfig::default();
	let manifest = LunarEpisodeManifest::derive(0x1234_5678_9abc_def0, 3, 7, 0);
	let first = LunarTerrain::seeded(config, manifest)?;
	let repeated = LunarTerrain::seeded(config, manifest)?;
	assert_eq!(first.heights(), repeated.heights());
	assert!(
		first
			.heights()
			.iter()
			.all(|height| height.abs() <= config.max_abs_height + 1.0e-12)
	);
	for x in [-3.0, -1.0, 0.0, 1.0, 3.0] {
		for z in [-3.0, -1.0, 0.0, 1.0, 3.0] {
			let sample = first.query(x, z);
			assert!(sample.in_bounds);
			assert_eq!(sample.height, 0.0);
			assert_eq!(
				(sample.normal.x, sample.normal.y, sample.normal.z),
				(0.0, 1.0, 0.0)
			);
			assert!(first.is_on_pad(x, z));
		}
	}
	Ok(())
}

#[test]
fn lunar_scalar_free_fall_matches_semi_implicit_solution() -> oa::Result<()> {
	let config = LunarLander3dConfig::default();
	config.validate()?;
	let terrain = LunarTerrain::flat(config.terrain)?;
	let mut state = flight_state(config, 20.0);
	state.linear_velocity = oa::vlm::DVec3 {
		x: 0.25,
		y: 1.0,
		z: -0.5,
	};
	let initial = state;
	integrate(&config, &terrain, LunarAction::Coast, &mut state)?;
	let substep = config.policy_time_step / f64::from(config.physics_substeps);
	let count = f64::from(config.physics_substeps);
	let expected_y = initial.position.y + initial.linear_velocity.y * config.policy_time_step
		- config.gravity * substep * substep * count * (count + 1.0) * 0.5;
	assert!(
		(state.position.x - (initial.position.x + 0.25 * config.policy_time_step)).abs() < 1.0e-14
	);
	assert!((state.position.y - expected_y).abs() < 1.0e-14);
	assert!(
		(state.position.z - (initial.position.z - 0.5 * config.policy_time_step)).abs() < 1.0e-14
	);
	assert!(
		(state.linear_velocity.y - (1.0 - config.gravity * config.policy_time_step)).abs()
			< 1.0e-14
	);
	Ok(())
}

#[test]
fn lunar_scalar_thrusters_preserve_donor_fuel_and_axis_contracts() -> oa::Result<()> {
	let config = LunarLander3dConfig {
		gravity: 0.0,
		..LunarLander3dConfig::default()
	};
	let terrain = LunarTerrain::flat(config.terrain)?;
	let initial = flight_state(config, 20.0);
	let mut main = initial;
	let main_result = integrate(&config, &terrain, LunarAction::MainEngine, &mut main)?;
	let acceleration = config.main_thrust / config.mass;
	assert!((main.linear_velocity.y - acceleration * config.policy_time_step).abs() < 1.0e-14);
	assert_eq!(main.linear_velocity.x, 0.0);
	assert_eq!(main.linear_velocity.z, 0.0);
	assert!(
		(main.fuel - (config.fuel_capacity - config.main_fuel_rate * config.policy_time_step))
			.abs() < 1.0e-12
	);
	assert!(
		(main_result.main_fuel_used - config.main_fuel_rate * config.policy_time_step).abs()
			< 1.0e-14
	);

	let mut pitch = initial;
	let pitch_result = integrate(&config, &terrain, LunarAction::PitchPositive, &mut pitch)?;
	let expected = config.attitude_torque / config.diagonal_inertia.x * config.policy_time_step;
	assert!((pitch.angular_velocity_body.x - expected).abs() < 1.0e-14);
	assert_eq!(pitch.angular_velocity_body.y, 0.0);
	assert_eq!(pitch.angular_velocity_body.z, 0.0);
	assert_eq!(pitch.linear_velocity.length_squared(), 0.0);
	assert!((pitch.orientation.norm() - 1.0).abs() < 1.0e-15);
	assert!(
		(pitch_result.attitude_fuel_used - config.attitude_fuel_rate * config.policy_time_step)
			.abs() < 1.0e-14
	);
	Ok(())
}

#[test]
fn lunar_scalar_observation_has_frozen_thirty_three_value_layout() -> oa::Result<()> {
	let config = LunarLander3dConfig::default();
	let terrain = LunarTerrain::flat(config.terrain)?;
	let mut state = flight_state(config, 6.0);
	state.position.x = 2.0;
	state.position.z = -3.0;
	state.linear_velocity = oa::vlm::DVec3 {
		x: 0.4,
		y: -0.8,
		z: 1.2,
	};
	state.angular_velocity_body = oa::vlm::DVec3 {
		x: 0.1,
		y: -0.2,
		z: 0.3,
	};
	state.fuel = config.fuel_capacity * 0.5;
	state.foot_contacts = [true, false, true, false];
	let observation = observe(&config, &terrain, &state);
	assert_eq!(observation.len(), LUNAR_OBSERVATION_SIZE);
	assert!(
		observation
			.iter()
			.all(|value| value.is_finite() && (-1.0..=1.0).contains(value))
	);
	assert_eq!(
		observation[0],
		(state.position.x / config.position_observation_scale) as f32
	);
	assert_eq!(&observation[6..12], &[0.0, 1.0, 0.0, 0.0, 0.0, -1.0]);
	assert_eq!(&observation[28..32], &[1.0, 0.0, 1.0, 0.0]);
	assert_eq!(observation[32], 0.5);
	Ok(())
}

#[test]
fn lunar_scalar_contact_solver_is_finite_fixed_and_bounded() -> oa::Result<()> {
	let config = LunarLander3dConfig {
		hard_foot_impact_speed: 10.0,
		safe_dwell_steps: 1000,
		..LunarLander3dConfig::default()
	};
	let terrain = LunarTerrain::flat(config.terrain)?;
	let mut state = flight_state(config, 1.10);
	state.linear_velocity.y = -0.1;
	let result = integrate(&config, &terrain, LunarAction::Coast, &mut state)?;
	assert!(state.is_finite());
	assert!(result.contact.is_finite());
	assert!(result.contact.bounded);
	assert!(result.contact.foot_contact_occurred);
	assert!(result.contact.contact_count > 0);
	assert!(result.contact.maximum_normal_impulse <= config.max_contact_impulse);
	assert!(result.contact.maximum_friction_impulse <= config.max_contact_impulse);
	let maximum_correction = f64::from(config.physics_substeps * config.contact_iterations)
		* (config.body_supports.len() + config.foot_supports.len()) as f64
		* config.max_position_correction_per_contact;
	assert!(result.contact.total_position_correction <= maximum_correction + 1.0e-12);
	Ok(())
}

#[test]
fn lunar_scalar_fuel_exhaustion_clamps_thrust_without_changing_mass() -> oa::Result<()> {
	let config = LunarLander3dConfig {
		gravity: 0.0,
		fuel_capacity: 0.01,
		..LunarLander3dConfig::default()
	};
	let terrain = LunarTerrain::flat(config.terrain)?;
	let mut state = flight_state(config, 20.0);
	integrate(&config, &terrain, LunarAction::MainEngine, &mut state)?;
	let burn_duration = config.fuel_capacity / config.main_fuel_rate;
	let expected_velocity = config.main_thrust / config.mass * burn_duration;
	assert_eq!(state.fuel, 0.0);
	assert!((state.linear_velocity.y - expected_velocity).abs() < 1.0e-14);
	let velocity_before = state.linear_velocity.y;
	integrate(&config, &terrain, LunarAction::MainEngine, &mut state)?;
	assert_eq!(state.fuel, 0.0);
	assert_eq!(state.linear_velocity.y, velocity_before);
	Ok(())
}

#[test]
fn lunar_scalar_quaternion_stays_normalized_through_long_torque_trace() -> oa::Result<()> {
	let config = LunarLander3dConfig {
		gravity: 0.0,
		max_episode_steps: 2000,
		..LunarLander3dConfig::default()
	};
	let terrain = LunarTerrain::flat(config.terrain)?;
	let mut state = flight_state(config, 20.0);
	let actions = [
		LunarAction::PitchPositive,
		LunarAction::RollNegative,
		LunarAction::YawPositive,
		LunarAction::PitchNegative,
		LunarAction::RollPositive,
		LunarAction::YawNegative,
	];
	for step in 0..600 {
		integrate(&config, &terrain, actions[step % actions.len()], &mut state)?;
		assert!((state.orientation.norm() - 1.0).abs() <= 2.0e-15);
	}
	Ok(())
}

#[test]
fn lunar_scalar_config_rejects_invalid_contract_fields() {
	assert!(LunarLander3dConfig::default().validate().is_ok());
	assert!(
		LunarLander3dConfig {
			mass: 0.0,
			..LunarLander3dConfig::default()
		}
		.validate()
		.is_err()
	);
	assert!(
		LunarLander3dConfig {
			physics_substeps: 0,
			..LunarLander3dConfig::default()
		}
		.validate()
		.is_err()
	);
	assert!(
		LunarLander3dConfig {
			success_reward: -1.0,
			..LunarLander3dConfig::default()
		}
		.validate()
		.is_err()
	);
}

#[test]
fn lunar_scalar_environment_matches_frozen_donor_trace() -> oa::Result<()> {
	let config = LunarLander3dConfig {
		max_episode_steps: 96,
		..LunarLander3dConfig::default()
	};
	let manifest = manifest_for(config);
	let mut environment = LunarScalarEnvironment::seeded(config, manifest)?;
	let mut digest = LunarTraceDigest::new();
	digest_manifest(&mut digest, manifest);
	for height in environment.terrain().heights() {
		digest.f64(*height);
	}
	digest_state(&mut digest, *environment.state());
	let actions = [
		LunarAction::Coast,
		LunarAction::MainEngine,
		LunarAction::PitchPositive,
		LunarAction::PitchNegative,
		LunarAction::RollPositive,
		LunarAction::RollNegative,
		LunarAction::YawPositive,
		LunarAction::YawNegative,
	];
	for step in 0..config.max_episode_steps as usize {
		let action = actions[step % actions.len()];
		digest.u32(action as u32);
		let transition = environment.step(action as u32, false)?;
		digest_state(&mut digest, *environment.state());
		digest_reward(&mut digest, transition.reward_terms);
		digest.f64(transition.reward);
		digest.bool(transition.terminated);
		digest.bool(transition.truncated);
		digest.u32(transition.end_reason as u32);
		for value in transition.observation {
			digest.f32(value);
		}
	}
	assert!(!environment.state().terminated);
	assert!(environment.state().truncated);
	assert_eq!(environment.state().end_reason as u32, 6);
	assert_eq!(digest.0, 0xa85e_a7bf_d77f_9ab5);
	Ok(())
}

#[test]
fn lunar_scalar_environment_preserves_invalid_action_and_external_stop_semantics() -> oa::Result<()>
{
	let config = LunarLander3dConfig::default();
	let manifest = manifest_for(config);
	let mut invalid = LunarScalarEnvironment::flat(config, manifest)?;
	let before = *invalid.state();
	let transition = invalid.step(8, false)?;
	assert!(transition.terminated);
	assert!(!transition.truncated);
	assert_eq!(transition.end_reason as u32, 8);
	assert_eq!(transition.reward, config.failure_penalty);
	assert_eq!(invalid.state().position, before.position);
	assert_eq!(invalid.state().episode_step, before.episode_step + 1);
	assert!(invalid.step(0, false).is_err());

	let mut stopped = LunarScalarEnvironment::flat(config, manifest)?;
	let before = *stopped.state();
	let transition = stopped.step(0, true)?;
	assert!(!transition.terminated);
	assert!(transition.truncated);
	assert_eq!(transition.end_reason as u32, 7);
	assert_eq!(
		*stopped.state(),
		LunarLander3dState {
			truncated: true,
			end_reason: transition.end_reason,
			..before
		}
	);
	Ok(())
}

#[test]
fn lunar_scalar_environment_keeps_end_reasons_distinct() -> oa::Result<()> {
	let success_config = LunarLander3dConfig {
		gravity: 0.0,
		safe_dwell_steps: 2,
		..LunarLander3dConfig::default()
	};
	let mut success = LunarScalarEnvironment::flat(success_config, manifest_for(success_config))?;
	success.set_state(flight_state(success_config, 1.15))?;
	assert!(!success.step(0, false)?.terminated);
	let landed = success.step(0, false)?;
	assert!(landed.terminated);
	assert!(!landed.truncated);
	assert_eq!(landed.end_reason as u32, 1);
	assert_eq!(landed.reward_terms.terminal, success_config.success_reward);

	let config = LunarLander3dConfig {
		gravity: 0.0,
		..LunarLander3dConfig::default()
	};
	let manifest = manifest_for(config);
	let mut body = LunarScalarEnvironment::flat(config, manifest)?;
	body.set_state(flight_state(config, 0.60))?;
	assert_eq!(body.step(0, false)?.end_reason as u32, 2);

	let mut hard = LunarScalarEnvironment::flat(config, manifest)?;
	let mut state = flight_state(config, 1.17);
	state.linear_velocity.y = -3.0;
	hard.set_state(state)?;
	assert_eq!(hard.step(0, false)?.end_reason as u32, 3);

	let mut outside = LunarScalarEnvironment::flat(config, manifest)?;
	let mut state = flight_state(config, 20.0);
	state.position.x = outside.terrain().max_x() + 0.5;
	outside.set_state(state)?;
	assert_eq!(outside.step(0, false)?.end_reason as u32, 4);

	let horizon_config = LunarLander3dConfig {
		gravity: 0.0,
		max_episode_steps: 1,
		..LunarLander3dConfig::default()
	};
	let mut horizon = LunarScalarEnvironment::flat(horizon_config, manifest_for(horizon_config))?;
	horizon.set_state(flight_state(horizon_config, 20.0))?;
	let timeout = horizon.step(0, false)?;
	assert!(!timeout.terminated);
	assert!(timeout.truncated);
	assert_eq!(timeout.end_reason as u32, 6);
	assert_eq!(timeout.reward_terms.terminal, 0.0);
	assert_eq!(
		timeout.reward_terms.shaping,
		horizon_config.reward_gamma * timeout.reward_terms.potential_after
			- timeout.reward_terms.potential_before
	);
	Ok(())
}

#[test]
fn lunar_scripted_controller_reaches_safe_landing() -> oa::Result<()> {
	let config = LunarLander3dConfig {
		safe_dwell_steps: 12,
		max_episode_steps: 1200,
		..LunarLander3dConfig::default()
	};
	let mut environment = LunarScalarEnvironment::flat(config, manifest_for(config))?;
	let mut state = flight_state(config, 4.0);
	state.linear_velocity.y = -0.2;
	environment.set_state(state)?;
	for _ in 0..config.max_episode_steps {
		if environment.state().terminated || environment.state().truncated {
			break;
		}
		let action = scripted_landing_action(&config, environment.state());
		environment.step(action as u32, false)?;
	}
	assert!(environment.state().terminated);
	assert!(!environment.state().truncated);
	assert_eq!(environment.state().end_reason as u32, 1);
	Ok(())
}

test_vk!(
	lunar_vector_reset_and_step_match_scalar_lane_oracles,
	engine,
	{
		let scalar_config = LunarLander3dConfig::default();
		let config = LunarLander3dVectorConfig {
			environments: 4,
			seed: 0x1234_5678_9abc_def0,
			environment: scalar_config,
		};
		let mut vector = LunarLander3dVector::flat(&engine, config)?;
		let event = vector.submit()?;
		vector.wait(&event)?;
		let initial = vector.observation().read_f32()?;
		let mut scalar = (0..config.environments)
			.map(|lane| {
				LunarScalarEnvironment::flat(
					scalar_config,
					LunarEpisodeManifest::derive(
						config.seed,
						lane,
						0,
						scalar_config.contract_fingerprint(),
					),
				)
			})
			.collect::<oa::Result<Vec<_>>>()?;
		for (lane, environment) in scalar.iter().enumerate() {
			for (component, expected) in environment.observation().into_iter().enumerate() {
				assert_f32_close(
					initial[lane * LUNAR_OBSERVATION_SIZE + component],
					expected,
					2.0e-5,
					&format!("reset lane {lane} observation {component}"),
				);
			}
		}

		let actions = [0_i32, 1, 2, 7];
		let action = Matrix::from_slice(&engine, [config.environments as usize], &actions)?;
		let transition = vector.step(&action)?;
		let event = vector.submit()?;
		vector.wait(&event)?;
		assert_eq!(transition.observation.read_f32()?, initial);
		let next = transition.next_observation.read_f32()?;
		let rewards = transition.reward.read_f32()?;
		let terminated = transition.terminated.read::<u8>()?;
		let truncated = transition.truncated.read::<u8>()?;
		let reasons = transition.end_reason.read::<u32>()?;
		for (lane, environment) in scalar.iter_mut().enumerate() {
			let expected = environment.step(actions[lane] as u32, false)?;
			for (component, expected_value) in expected.observation.into_iter().enumerate() {
				assert_f32_close(
					next[lane * LUNAR_OBSERVATION_SIZE + component],
					expected_value,
					1.0e-4,
					&format!("step lane {lane} observation {component}"),
				);
			}
			assert_f32_close(
				rewards[lane],
				expected.reward as f32,
				1.0e-4,
				&format!("step lane {lane} reward"),
			);
			assert_eq!(terminated[lane], u8::from(expected.terminated));
			assert_eq!(truncated[lane], u8::from(expected.truncated));
			assert_eq!(reasons[lane], expected.end_reason as u32);
		}
		vector.close()?;
		Ok(())
	}
);

test_vk!(
	lunar_vector_terminal_precedence_telemetry_and_completed_reset,
	engine,
	{
		let config = LunarLander3dVectorConfig {
			environments: 3,
			seed: 918_273_645,
			environment: LunarLander3dConfig {
				max_episode_steps: 1,
				..LunarLander3dConfig::default()
			},
		};
		let mut vector = LunarLander3dVector::flat(&engine, config)?;
		let event = vector.submit()?;
		vector.wait(&event)?;
		let initial = vector.observation().read_f32()?;
		let action = Matrix::from_slice(&engine, [3], &[0_i32, 8, 0])?;
		let external_stop = Matrix::from_slice(&engine, [3], &[1_u8, 0, 0])?;
		let transition = vector.step_with_external_stop(&action, &external_stop)?;
		assert!(vector.copy_episode_telemetry().is_err());
		let event = vector.submit()?;
		assert!(vector.copy_episode_telemetry().is_err());
		vector.wait(&event)?;
		assert_eq!(transition.terminated.read::<u8>()?, [0, 1, 0]);
		assert_eq!(transition.truncated.read::<u8>()?, [1, 0, 1]);
		assert_eq!(transition.end_reason.read::<u32>()?, [7, 8, 6]);
		let telemetry = vector.copy_episode_telemetry()?;
		assert_eq!(telemetry.len(), 3);
		assert!(telemetry.into_iter().all(|value| value.is_valid()));

		vector.reset_completed()?;
		let event = vector.submit()?;
		vector.wait(&event)?;
		assert_ne!(vector.observation().read_f32()?, initial);
		assert_eq!(vector.end_reason().read::<u32>()?, [0, 0, 0]);
		assert!(
			vector
				.copy_episode_telemetry()?
				.into_iter()
				.all(|value| !value.terminated && !value.truncated && value.episode_step == 0)
		);
		vector.close()?;
		Ok(())
	}
);

test_vk!(
	lunar_vector_rejects_unrepresentable_fp32_contracts,
	engine,
	{
		let mut config = LunarLander3dVectorConfig::default();
		config.environment.gravity = f64::from(f32::MIN_POSITIVE) * 0.5;
		assert_eq!(
			result_error_kind(LunarLander3dVector::flat(&engine, config)),
			ErrorKind::OutOfRange
		);

		let mut config = LunarLander3dVectorConfig::default();
		config.environment.position_potential_weight = f64::from(f32::MAX);
		assert_eq!(
			result_error_kind(LunarLander3dVector::flat(&engine, config)),
			ErrorKind::OutOfRange
		);

		let mut config = LunarLander3dVectorConfig::default();
		config.environment.policy_time_step = f64::from(f32::MIN_POSITIVE);
		config.environment.physics_substeps = 64;
		assert_eq!(
			result_error_kind(LunarLander3dVector::flat(&engine, config)),
			ErrorKind::OutOfRange
		);

		let mut config = LunarLander3dVectorConfig::default();
		config.environment.fuel_capacity = 1.0e8;
		config.environment.main_fuel_rate = 100.0;
		config.environment.attitude_fuel_rate = 1.0e6;
		assert_eq!(
			result_error_kind(LunarLander3dVector::flat(&engine, config)),
			ErrorKind::OutOfRange
		);
		Ok(())
	}
);

test_vk!(
	lunar_vector_fixed_action_trace_matches_scalar_fp64_oracle,
	engine,
	{
		const ENVIRONMENTS: usize = 5;
		const TRACE_STEPS: u32 = 24;
		let config = LunarLander3dVectorConfig {
			environments: ENVIRONMENTS as u32,
			seed: 0xc34d_8217_a695_0bef,
			environment: LunarLander3dConfig::default(),
		};
		let mut vector = LunarLander3dVector::flat(&engine, config)?;
		let event = vector.submit()?;
		vector.wait(&event)?;
		let mut scalar = (0..ENVIRONMENTS)
			.map(|lane| {
				LunarScalarEnvironment::flat(
					config.environment,
					LunarEpisodeManifest::derive(
						config.seed,
						lane as u32,
						0,
						config.environment.contract_fingerprint(),
					),
				)
			})
			.collect::<oa::Result<Vec<_>>>()?;
		for step_index in 0..TRACE_STEPS {
			let actions: Vec<i32> = (0..ENVIRONMENTS)
				.map(|lane| ((step_index + lane as u32 * 3) % 8) as i32)
				.collect();
			let action = Matrix::from_slice(&engine, [ENVIRONMENTS], &actions)?;
			let transition = vector.step(&action)?;
			let event = vector.submit()?;
			vector.wait(&event)?;
			let observations = transition.next_observation.read_f32()?;
			let rewards = transition.reward.read_f32()?;
			let terminated = transition.terminated.read::<u8>()?;
			let truncated = transition.truncated.read::<u8>()?;
			let reasons = transition.end_reason.read::<u32>()?;
			for lane in 0..ENVIRONMENTS {
				let expected = scalar[lane].step(actions[lane] as u32, false)?;
				let begin = lane * LUNAR_OBSERVATION_SIZE;
				assert_lunar_observation_close(
					&observations[begin..begin + LUNAR_OBSERVATION_SIZE],
					&expected.observation,
					lane,
					step_index + 1,
				);
				assert_f32_close(
					rewards[lane],
					expected.reward as f32,
					4.0e-3,
					&format!("lane {lane} step {step_index} reward"),
				);
				assert_eq!(terminated[lane], u8::from(expected.terminated));
				assert_eq!(truncated[lane], u8::from(expected.truncated));
				assert_eq!(reasons[lane], expected.end_reason as u32);
			}
		}
		vector.close()?;
		Ok(())
	}
);

test_vk!(
	lunar_vector_cancel_reseed_and_recovery_are_transactional,
	engine,
	{
		let seed_a = 0x1021_3243_5465_7687;
		let seed_b = 0x8091_a2b3_c4d5_e6f7;
		let seed_c = 0x4051_6273_8495_a6b7;
		let config = LunarLander3dVectorConfig {
			environments: 2,
			seed: seed_a,
			environment: LunarLander3dConfig::default(),
		};
		let mut cancelled = LunarLander3dVector::flat(&engine, config)?;
		cancelled.cancel()?;
		let actions = Matrix::from_slice(&engine, [2], &[0_i32, 0])?;
		assert_eq!(
			result_error_kind(cancelled.step(&actions)),
			ErrorKind::FailedPrecondition
		);
		assert_eq!(
			cancelled.reset_completed().unwrap_err().kind(),
			ErrorKind::FailedPrecondition
		);
		cancelled.reset(seed_a)?;
		let event = cancelled.submit()?;
		cancelled.wait(&event)?;

		cancelled.reset(seed_b)?;
		assert_eq!(cancelled.config().seed, seed_a);
		let event = cancelled.submit()?;
		cancelled.wait(&event)?;
		assert_eq!(cancelled.config().seed, seed_b);
		let seed_b_observation = cancelled.observation().read_f32()?;

		cancelled.reset(seed_c)?;
		assert_eq!(cancelled.config().seed, seed_b);
		cancelled.cancel()?;
		assert_eq!(cancelled.config().seed, seed_b);
		cancelled.reset(seed_b)?;
		let event = cancelled.submit()?;
		cancelled.wait(&event)?;
		assert_eq!(cancelled.observation().read_f32()?, seed_b_observation);
		cancelled.close()?;
		Ok(())
	}
);

test_vk!(
	lunar_vector_odd_lane_reuse_and_poison_stay_bounded,
	engine,
	{
		const ENVIRONMENTS: usize = 257;
		let mut vector = LunarLander3dVector::flat(
			&engine,
			LunarLander3dVectorConfig {
				environments: ENVIRONMENTS as u32,
				seed: 0x5d3e_8c71_4a09_b26f,
				environment: LunarLander3dConfig::default(),
			},
		)?;
		let event = vector.submit()?;
		vector.wait(&event)?;
		for reuse in 0..8 {
			let mut actions = vec![(reuse % 8) as i32; ENVIRONMENTS];
			for lane in (reuse..ENVIRONMENTS).step_by(31) {
				actions[lane] = if lane & 1 == 0 { i32::MIN } else { i32::MAX };
			}
			let action = Matrix::from_slice(&engine, [ENVIRONMENTS], &actions)?;
			let transition = vector.step(&action)?;
			let event = vector.submit()?;
			vector.wait(&event)?;
			assert!(
				transition
					.next_observation
					.read_f32()?
					.into_iter()
					.all(f32::is_finite)
			);
			assert!(
				transition
					.reward
					.read_f32()?
					.into_iter()
					.all(f32::is_finite)
			);
			vector.reset_completed()?;
			let event = vector.submit()?;
			vector.wait(&event)?;
		}
		vector.close()?;
		Ok(())
	}
);

test_vk!(
	lunar_vector_matches_scalar_through_contact_and_terminal_episodes,
	engine,
	{
		let scripted = LunarLander3dConfig {
			safe_dwell_steps: 12,
			..LunarLander3dConfig::default()
		};
		run_lunar_vector_episode_differential(
			&engine,
			scripted,
			0x5049_4c4f_545f_4556,
			true,
			oa::sdk::ml::rl::LunarEndReason::SafeLanding,
		)?;

		run_lunar_vector_episode_differential(
			&engine,
			LunarLander3dConfig::default(),
			0x4841_5244_5f46_4f4f,
			false,
			oa::sdk::ml::rl::LunarEndReason::HardFootImpact,
		)?;

		let mut body_impact = LunarLander3dConfig::default();
		for foot in &mut body_impact.foot_supports {
			foot.body_offset.y = 1.0;
		}
		run_lunar_vector_episode_differential(
			&engine,
			body_impact,
			0x424f_4459_5f49_4d50,
			false,
			oa::sdk::ml::rl::LunarEndReason::BodyImpact,
		)?;

		run_lunar_vector_episode_differential(
			&engine,
			LunarLander3dConfig {
				gravity: 0.0,
				max_episode_steps: 3,
				..LunarLander3dConfig::default()
			},
			0x5449_4d45_5f4c_494d,
			false,
			oa::sdk::ml::rl::LunarEndReason::TimeLimit,
		)?;

		run_lunar_vector_episode_differential(
			&engine,
			LunarLander3dConfig {
				task_maximum_y: 4.0,
				..LunarLander3dConfig::default()
			},
			0x4f55_545f_4f46_5f42,
			false,
			oa::sdk::ml::rl::LunarEndReason::OutOfBounds,
		)?;
		Ok(())
	}
);

test_vk!(
	lunar_ppo_smoke_teacher_and_checkpoint_match_donor_workflow,
	engine,
	{
		use oa::ml::{
			AdamW, CategoricalActorCritic, CategoricalActorCriticConfig, Module,
			training::{PpoTrainer, PpoTrainerConfig},
		};

		const SEED: u64 = 0x1a2b_3c4d;
		let model = CategoricalActorCritic::new(
			&engine,
			CategoricalActorCriticConfig {
				observation_size: LUNAR_OBSERVATION_SIZE,
				action_count: 8,
				hidden_size: 64,
				seed: SEED,
			},
		)?;
		let mut optimizer =
			AdamW::with_hyperparameters(model.all_parameters()?, 2.5e-4, 0.9, 0.999, 1.0e-8, 0.0)?;
		let teacher = pretrain_scripted_teacher(
			&engine,
			&model,
			LunarTeacherConfig {
				episodes: 16,
				epochs: 2,
				batch_size: 512,
				maximum_samples: 8192,
				..LunarTeacherConfig::default()
			},
		)?;
		assert!(teacher.samples > 0);
		assert!(teacher.optimizer_steps > 0);
		assert!(teacher.dataset_digest > 0);
		assert!(teacher.final_loss < teacher.initial_loss);
		assert_eq!(optimizer.step_count(), 0);

		let mut environment = LunarLander3dVector::flat(
			&engine,
			LunarLander3dVectorConfig {
				environments: 7,
				seed: SEED,
				environment: LunarLander3dConfig::default(),
			},
		)?;
		{
			let mut trainer = PpoTrainer::new(
				&engine,
				&model,
				&mut optimizer,
				PpoTrainerConfig {
					rollouts: 1,
					horizon: 16,
					environments: 7,
					update_epochs: 1,
					observation_shape: vec![LUNAR_OBSERVATION_SIZE],
					seed: SEED,
					..PpoTrainerConfig::default()
				},
			)?;
			collect_ppo_rollout(&mut environment, &mut trainer)?;
			assert!(trainer.update()?);
			let metrics = trainer.metrics();
			assert!(metrics.total_loss.is_finite());
			assert!(metrics.policy_loss.is_finite());
			assert!(metrics.value_loss.is_finite());
			assert!(metrics.entropy.is_finite());
			assert_eq!(metrics.rollout, 1);
			trainer.finish()?;
		}
		environment.close()?;
		assert_eq!(optimizer.step_count(), 1);

		let evaluation_config = LunarFirstEpisodeEvaluationConfig {
			environments: 7,
			horizon: 64,
			submission_chunk_steps: 8,
			environment_seed: 0x4c55_4e41_525f_4556,
		};
		let evaluation = evaluate_first_episodes(&engine, &model, evaluation_config)?;
		assert_eq!(
			evaluation.completed_episodes + evaluation.incomplete_episodes,
			evaluation.expected_episodes
		);
		assert!(evaluation.action_trace_digest > 0);
		assert!(evaluation.value_trace_digest > 0);
		assert!(evaluation.submissions <= 8);

		let checkpoint = std::env::temp_dir().join("oars_lunar_lander_ppo_smoke.oam");
		if checkpoint.exists() {
			std::fs::remove_file(&checkpoint).expect("remove stale Lunar smoke checkpoint");
		}
		oa::ml::save_checkpoint(&checkpoint, &model, &optimizer)?;
		let restored_model = CategoricalActorCritic::new(
			&engine,
			CategoricalActorCriticConfig {
				observation_size: LUNAR_OBSERVATION_SIZE,
				action_count: 8,
				hidden_size: 64,
				seed: SEED,
			},
		)?;
		let mut restored_optimizer = AdamW::with_hyperparameters(
			restored_model.all_parameters()?,
			2.5e-4,
			0.9,
			0.999,
			1.0e-8,
			0.0,
		)?;
		oa::ml::load_checkpoint(
			&engine,
			&checkpoint,
			&restored_model,
			&mut restored_optimizer,
		)?;
		let restored = evaluate_first_episodes(&engine, &restored_model, evaluation_config)?;
		std::fs::remove_file(&checkpoint).expect("remove completed Lunar smoke checkpoint");
		assert_eq!(restored, evaluation);
		assert_eq!(restored_optimizer.step_count(), optimizer.step_count());
		Ok(())
	}
);
