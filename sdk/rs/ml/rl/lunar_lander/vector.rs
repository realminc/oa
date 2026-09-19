//! Batched FP32 Lunar Lander environment over the canonical Engine lifecycle.

use crate::ml::environment::{Environment, EnvironmentExecution};
use crate::ml::lowering::environment as lowering;
use crate::ml::{EnvironmentSpace, EnvironmentSpec, EnvironmentTransition};
use crate::{DType, Engine, Error, Matrix, Result};

use super::{
	LUNAR_OBSERVATION_SIZE, LUNAR_RANDOM_VERSION, LUNAR_TERRAIN_VERSION, LunarEndReason,
	LunarLander3dConfig, LunarTerrain,
};

/// Current packed vector-configuration ABI.
pub const LUNAR_VECTOR_CONFIG_LAYOUT_VERSION: u32 = 1;
/// Current packed vector-state ABI.
pub const LUNAR_VECTOR_STATE_LAYOUT_VERSION: u32 = 1;

const CONFIG_F32_COUNT: usize = 75;
const CONFIG_U32_COUNT: usize = 17;
const STATE_F32_WIDTH: usize = 32;
const STATE_U32_WIDTH: usize = 16;

/// Vector-lane count, reset seed, and versioned environment behavior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LunarLander3dVectorConfig {
	pub environments: u32,
	pub seed: u64,
	pub environment: LunarLander3dConfig,
}

impl Default for LunarLander3dVectorConfig {
	fn default() -> Self {
		Self {
			environments: 1,
			seed: 1,
			environment: LunarLander3dConfig::default(),
		}
	}
}

/// Device-resident values produced by one vector step.
#[derive(Clone)]
pub struct LunarLander3dVectorStep {
	pub observation: Matrix,
	pub next_observation: Matrix,
	pub reward: Matrix,
	pub terminated: Matrix,
	pub truncated: Matrix,
	pub end_reason: Matrix,
}

/// Compact synchronized host telemetry for one vector lane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LunarLander3dEpisodeTelemetry {
	pub episode_return: f32,
	pub fuel_remaining: f32,
	pub terminal_linear_speed: f32,
	pub terminal_angular_speed: f32,
	pub maximum_foot_impulse: f32,
	pub episode_step: u32,
	pub terminated: bool,
	pub truncated: bool,
	pub end_reason: LunarEndReason,
}

impl LunarLander3dEpisodeTelemetry {
	/// Return whether all continuous telemetry is finite and non-negative where required.
	pub fn is_valid(self) -> bool {
		self.episode_return.is_finite()
			&& self.fuel_remaining.is_finite()
			&& self.fuel_remaining >= 0.0
			&& self.terminal_linear_speed.is_finite()
			&& self.terminal_linear_speed >= 0.0
			&& self.terminal_angular_speed.is_finite()
			&& self.terminal_angular_speed >= 0.0
			&& self.maximum_foot_impulse.is_finite()
			&& self.maximum_foot_impulse >= 0.0
	}
}

/// Native vectorized flat-terrain Lunar Lander workload.
#[must_use]
pub struct LunarLander3dVector<'engine> {
	engine: &'engine Engine,
	execution: EnvironmentExecution,
	config: LunarLander3dVectorConfig,
	spec: EnvironmentSpec,
	config_f32: Matrix,
	config_u32: Matrix,
	terrain_f32: Matrix,
	state_f32: Matrix,
	state_u32: Matrix,
	observation: Matrix,
	transition_observation: Matrix,
	reward: Matrix,
	terminated: Matrix,
	truncated: Matrix,
	end_reason: Matrix,
	no_external_stop: Matrix,
	pending_seed: Option<u64>,
	has_committed_state: bool,
	has_pending_full_reset: bool,
}

impl<'engine> LunarLander3dVector<'engine> {
	/// Allocate a flat-terrain vector environment and record its initial reset.
	///
	/// # Errors
	///
	/// Returns an error for an invalid or non-representable configuration,
	/// allocation failure, schema failure, or command-recording failure.
	pub fn flat(engine: &'engine Engine, config: LunarLander3dVectorConfig) -> Result<Self> {
		if config.environments == 0 {
			return Err(Error::invalid_argument(
				"Lunar Lander 3D requires at least one environment lane",
			));
		}
		config.environment.validate()?;
		let terrain = LunarTerrain::flat(config.environment.terrain)?;
		let config_f32_values = serialize_config_f32(&config.environment, &terrain)?;
		let config_u32_values = serialize_config_u32(&config.environment);
		let terrain_values: Vec<f32> = terrain
			.heights()
			.iter()
			.map(|value| *value as f32)
			.collect();
		let lanes = config.environments as usize;
		let state_f32_count = lanes
			.checked_mul(STATE_F32_WIDTH)
			.ok_or_else(|| Error::resource_exhausted("Lunar Lander FP32 state size overflows"))?;
		let state_u32_count = lanes
			.checked_mul(STATE_U32_WIDTH)
			.ok_or_else(|| Error::resource_exhausted("Lunar Lander U32 state size overflows"))?;
		let observation_count = lanes
			.checked_mul(LUNAR_OBSERVATION_SIZE)
			.ok_or_else(|| Error::resource_exhausted("Lunar Lander observation size overflows"))?;
		let config_f32 = Matrix::from_f32(engine, [CONFIG_F32_COUNT], &config_f32_values)?;
		let config_u32 = Matrix::from_slice(engine, [CONFIG_U32_COUNT], &config_u32_values)?;
		let terrain_f32 = Matrix::from_f32(engine, [terrain_values.len()], &terrain_values)?;
		let state_f32 = Matrix::from_f32(
			engine,
			[lanes, STATE_F32_WIDTH],
			&vec![0.0; state_f32_count],
		)?;
		let state_u32 = Matrix::from_slice(
			engine,
			[lanes, STATE_U32_WIDTH],
			&vec![0_u32; state_u32_count],
		)?;
		let observation = Matrix::from_f32(
			engine,
			[lanes, LUNAR_OBSERVATION_SIZE],
			&vec![0.0; observation_count],
		)?;
		let transition_observation = Matrix::from_f32(
			engine,
			[lanes, LUNAR_OBSERVATION_SIZE],
			&vec![0.0; observation_count],
		)?;
		let reward = Matrix::from_f32(engine, [lanes], &vec![0.0; lanes])?;
		let terminated = Matrix::from_slice(engine, [lanes], &vec![0_u8; lanes])?;
		let truncated = Matrix::from_slice(engine, [lanes], &vec![0_u8; lanes])?;
		let end_reason = Matrix::from_slice(engine, [lanes], &vec![0_u32; lanes])?;
		let no_external_stop = Matrix::from_slice(engine, [lanes], &vec![0_u8; lanes])?;
		let spec = EnvironmentSpec::new(
			EnvironmentSpace::continuous(
				"observation",
				[LUNAR_OBSERVATION_SIZE],
				DType::F32,
				-1.0,
				1.0,
			)?,
			EnvironmentSpace::discrete("action", 8, DType::I32)?,
			EnvironmentSpace::continuous("reward", [], DType::F32, f64::NEG_INFINITY, f64::INFINITY)?,
			EnvironmentSpace::binary("terminated", [])?,
			EnvironmentSpace::binary("truncated", [])?,
		)?;
		let mut result = Self {
			engine,
			execution: EnvironmentExecution::new(engine),
			config,
			spec,
			config_f32,
			config_u32,
			terrain_f32,
			state_f32,
			state_u32,
			observation,
			transition_observation,
			reward,
			terminated,
			truncated,
			end_reason,
			no_external_stop,
			pending_seed: None,
			has_committed_state: false,
			has_pending_full_reset: false,
		};
		Environment::reset(&mut result, config.seed)?;
		Ok(result)
	}

	/// Return the complete vector environment configuration.
	pub const fn config(&self) -> LunarLander3dVectorConfig {
		self.config
	}

	/// Return the device-resident end-reason vector.
	pub const fn end_reason(&self) -> &Matrix {
		&self.end_reason
	}

	/// Record a step and retain Lunar-specific end-reason output.
	pub fn step(&mut self, action: &Matrix) -> Result<LunarLander3dVectorStep> {
		let transition = Environment::step(self, action)?;
		Ok(self.wrap_step(transition))
	}

	/// Record a step with an explicit per-lane external-stop mask.
	pub fn step_with_external_stop(
		&mut self,
		action: &Matrix,
		external_stop: &Matrix,
	) -> Result<LunarLander3dVectorStep> {
		Environment::begin(self)?;
		if let Err(error) = self
			.spec
			.validate_action(action, self.config.environments)
			.and_then(|()| {
				EnvironmentSpace::binary("external_stop", [])?
					.validate_matrix(external_stop, self.config.environments)
			}) {
			let _ = Environment::cancel(self);
			return Err(error);
		}
		match self.record_step_impl(action, external_stop) {
			Ok(step) => {
				let transition = EnvironmentTransition::new(
					step.observation.clone(),
					step.next_observation.clone(),
					step.reward.clone(),
					step.terminated.clone(),
					step.truncated.clone(),
				);
				if let Err(error) =
					self
						.spec
						.validate_transition(action, &transition, self.config.environments)
				{
					let _ = Environment::cancel(self);
					return Err(error);
				}
				Ok(step)
			}
			Err(error) => {
				let _ = Environment::cancel(self);
				Err(error)
			}
		}
	}

	/// Copy compact lane telemetry after exact completion.
	pub fn copy_episode_telemetry(&self) -> Result<Vec<LunarLander3dEpisodeTelemetry>> {
		if self.execution.has_active_recording() || self.execution.has_pending_event() {
			return Err(Error::failed_precondition(
				"Lunar Lander telemetry requires no active recording or pending event",
			));
		}
		let state_f32 = self.state_f32.read::<f32>()?;
		let state_u32 = self.state_u32.read::<u32>()?;
		let mut output = Vec::with_capacity(self.config.environments as usize);
		for lane in 0..self.config.environments as usize {
			let f = lane * STATE_F32_WIDTH;
			let u = lane * STATE_U32_WIDTH;
			let length = |offset: usize| {
				let x = state_f32[f + offset];
				let y = state_f32[f + offset + 1];
				let z = state_f32[f + offset + 2];
				(x * x + y * y + z * z).sqrt()
			};
			let raw_reason = state_u32[u + 9];
			let end_reason = end_reason_from_u32(raw_reason)?;
			let terminated = state_u32[u + 7] != 0;
			let truncated = state_u32[u + 8] != 0;
			validate_terminal_state(raw_reason, state_u32[u + 7], state_u32[u + 8])?;
			let telemetry = LunarLander3dEpisodeTelemetry {
				episode_return: state_f32[f + 25],
				fuel_remaining: state_f32[f + 13],
				terminal_linear_speed: length(3),
				terminal_angular_speed: length(10),
				maximum_foot_impulse: state_f32[f + 21..f + 25]
					.iter()
					.copied()
					.fold(0.0_f32, f32::max),
				episode_step: state_u32[u + 5],
				terminated,
				truncated,
				end_reason,
			};
			if !telemetry.is_valid() {
				return Err(Error::failed_precondition(
					"Lunar Lander telemetry contains invalid values",
				));
			}
			output.push(telemetry);
		}
		Ok(output)
	}

	fn wrap_step(&self, transition: EnvironmentTransition) -> LunarLander3dVectorStep {
		LunarLander3dVectorStep {
			observation: transition.observation().clone(),
			next_observation: transition.next_observation().clone(),
			reward: transition.reward().clone(),
			terminated: transition.terminated().clone(),
			truncated: transition.truncated().clone(),
			end_reason: self.end_reason.clone(),
		}
	}

	fn record_reset_impl(&mut self, only_completed: bool) -> Result<()> {
		if only_completed && !self.has_committed_state && !self.has_pending_full_reset {
			return Err(Error::failed_precondition(
				"Lunar Lander completed reset requires submitted state or an earlier full reset",
			));
		}
		if !only_completed {
			self.has_pending_full_reset = true;
		}
		lowering::lunar_lander_reset(
			&self.config_f32,
			&self.config_u32,
			&self.terrain_f32,
			&self.state_f32,
			&self.state_u32,
			&self.observation,
			&self.end_reason,
			self.config.environments,
			self.pending_seed.unwrap_or(self.config.seed),
			only_completed,
			self.config.environment.environment_version,
			LUNAR_VECTOR_STATE_LAYOUT_VERSION,
		)
	}

	fn record_step_impl(
		&mut self,
		action: &Matrix,
		external_stop: &Matrix,
	) -> Result<LunarLander3dVectorStep> {
		if !self.has_committed_state && !self.has_pending_full_reset {
			return Err(Error::failed_precondition(
				"Lunar Lander step requires submitted state or an earlier full reset",
			));
		}
		let config = self.config.environment;
		lowering::lunar_lander_step(
			action,
			external_stop,
			&self.config_f32,
			&self.config_u32,
			&self.terrain_f32,
			&self.state_f32,
			&self.state_u32,
			&self.transition_observation,
			&self.observation,
			&self.reward,
			&self.terminated,
			&self.truncated,
			&self.end_reason,
			self.config.environments,
			config.environment_version,
			config.physics_version,
			config.observation_version,
			config.reward_version,
			LUNAR_VECTOR_STATE_LAYOUT_VERSION,
			config.contract_fingerprint(),
			config.max_episode_steps,
			config.failure_penalty,
		)?;
		Ok(LunarLander3dVectorStep {
			observation: self.transition_observation.clone(),
			next_observation: self.observation.clone(),
			reward: self.reward.clone(),
			terminated: self.terminated.clone(),
			truncated: self.truncated.clone(),
			end_reason: self.end_reason.clone(),
		})
	}
}

impl Environment for LunarLander3dVector<'_> {
	fn engine(&self) -> &Engine {
		self.engine
	}
	fn spec(&self) -> &EnvironmentSpec {
		&self.spec
	}
	fn environments(&self) -> u32 {
		self.config.environments
	}
	fn observation(&self) -> &Matrix {
		&self.observation
	}
	fn execution(&self) -> &EnvironmentExecution {
		&self.execution
	}
	fn execution_mut(&mut self) -> &mut EnvironmentExecution {
		&mut self.execution
	}

	fn record_reset(&mut self, seed: u64) -> Result<()> {
		self.pending_seed = Some(seed);
		self.record_reset_impl(false)
	}

	fn record_step(&mut self, action: &Matrix) -> Result<EnvironmentTransition> {
		let stop = self.no_external_stop.clone();
		let step = self.record_step_impl(action, &stop)?;
		Ok(EnvironmentTransition::new(
			step.observation,
			step.next_observation,
			step.reward,
			step.terminated,
			step.truncated,
		))
	}

	fn record_reset_completed(&mut self) -> Result<()> {
		self.record_reset_impl(true)
	}

	fn commit_recorded_state(&mut self) {
		if let Some(seed) = self.pending_seed.take() {
			self.config.seed = seed;
		}
		if self.has_pending_full_reset {
			self.has_committed_state = true;
		}
		self.has_pending_full_reset = false;
	}

	fn rollback_recorded_state(&mut self) {
		self.pending_seed = None;
		self.has_pending_full_reset = false;
	}
}

fn serialize_config_f32(config: &LunarLander3dConfig, terrain: &LunarTerrain) -> Result<Vec<f32>> {
	let mut values = Vec::with_capacity(CONFIG_F32_COUNT);
	for value in [
		config.policy_time_step,
		config.gravity,
		config.mass,
		config.diagonal_inertia.x,
		config.diagonal_inertia.y,
		config.diagonal_inertia.z,
		config.main_thrust,
		config.attitude_torque,
		config.fuel_capacity,
		config.main_fuel_rate,
		config.attitude_fuel_rate,
		config.restitution,
		config.friction,
		config.contact_slop,
		config.penetration_correction_fraction,
		config.max_position_correction_per_contact,
		config.max_contact_impulse,
		config.max_bias_speed,
		config.task_minimum_y,
		config.task_maximum_y,
		config.safe_linear_speed,
		config.safe_angular_speed,
		config.safe_tilt_radians,
		config.hard_foot_impact_speed,
		config.position_observation_scale,
		config.velocity_observation_scale,
		config.angular_velocity_observation_scale,
		config.terrain_clearance_observation_scale,
		config.foot_clearance_observation_scale,
		config.terrain_probe_spacing,
		config.reward_gamma,
		config.position_potential_weight,
		config.velocity_potential_weight,
		config.tilt_potential_weight,
		config.angular_potential_weight,
		config.main_fuel_cost_weight,
		config.attitude_fuel_cost_weight,
		config.soft_foot_contact_reward,
		config.stable_dwell_reward,
		config.success_reward,
		config.failure_penalty,
		config.terrain.cell_size,
		terrain.min_x(),
		terrain.max_x(),
		terrain.min_z(),
		terrain.max_z(),
		config.terrain.pad_half_extent,
	] {
		let converted = value as f32;
		if !converted.is_finite()
			|| (value != 0.0 && converted == 0.0)
			|| (converted != 0.0 && converted.is_subnormal())
		{
			return Err(Error::out_of_range(
				"Lunar Lander configuration cannot be represented as normal finite FP32",
			));
		}
		values.push(converted);
	}
	for support in config.body_supports.into_iter().chain(config.foot_supports) {
		for value in [
			support.body_offset.x,
			support.body_offset.y,
			support.body_offset.z,
			support.radius,
		] {
			let converted = value as f32;
			if !converted.is_finite() || (value != 0.0 && (converted == 0.0 || converted.is_subnormal()))
			{
				return Err(Error::out_of_range(
					"Lunar Lander support cannot be represented as normal finite FP32",
				));
			}
			values.push(converted);
		}
	}
	debug_assert_eq!(values.len(), CONFIG_F32_COUNT);
	validate_serialized_config(&values, config)?;
	Ok(values)
}

fn serialize_config_u32(config: &LunarLander3dConfig) -> Vec<u32> {
	let fingerprint = config.contract_fingerprint();
	vec![
		LUNAR_VECTOR_CONFIG_LAYOUT_VERSION,
		config.environment_version,
		LUNAR_RANDOM_VERSION,
		LUNAR_TERRAIN_VERSION,
		config.physics_version,
		config.observation_version,
		config.reward_version,
		config.physics_substeps,
		config.contact_iterations,
		config.safe_dwell_steps,
		config.max_episode_steps,
		config.terrain.cells_x,
		config.terrain.cells_z,
		fingerprint as u32,
		(fingerprint >> 32) as u32,
		STATE_F32_WIDTH as u32,
		STATE_U32_WIDTH as u32,
	]
}

fn validate_serialized_config(values: &[f32], config: &LunarLander3dConfig) -> Result<()> {
	let reciprocal_safe = |index: usize| values[index] > 0.0 && (1.0 / values[index]).is_finite();
	let denominators = [0, 2, 3, 4, 5, 8, 24, 25, 26, 27, 28, 41]
		.into_iter()
		.all(reciprocal_safe);
	let bounds = values[18] < values[19] && values[42] < values[43] && values[44] < values[45];
	let substep = values[0] / config.physics_substeps as f32;
	let representable_substep =
		substep > 0.0 && substep.is_finite() && !substep.is_subnormal() && (1.0 / substep).is_finite();
	let debit_valid = |rate: f32| {
		if rate == 0.0 {
			true
		} else {
			let debit = rate * substep;
			debit > 0.0 && debit.is_finite() && !debit.is_subnormal() && values[8] - debit != values[8]
		}
	};
	let supports = (0..7).all(|support| values[47 + support * 4 + 3] > 0.0);
	let potential = f64::from(values[31] + values[32] + values[33] + values[34]);
	let fuel = f64::from(values[8]) * f64::from(values[35] + values[36]);
	let terminal = f64::from(values[39].max(values[40].abs()));
	let reward = (1.0 + f64::from(values[30])) * potential
		+ fuel
		+ 4.0 * f64::from(values[37])
		+ f64::from(values[38])
		+ terminal;
	let bounded = reward.is_finite()
		&& reward <= f64::from(f32::MAX)
		&& (reward * f64::from(config.max_episode_steps)).is_finite()
		&& reward * f64::from(config.max_episode_steps) <= f64::from(f32::MAX);
	if !denominators
		|| !bounds
		|| !representable_substep
		|| !debit_valid(values[9])
		|| !debit_valid(values[10])
		|| !supports
		|| !bounded
	{
		return Err(Error::out_of_range(
			"Lunar Lander configuration loses required FP32 relationships or reward bounds",
		));
	}
	Ok(())
}

fn end_reason_from_u32(value: u32) -> Result<LunarEndReason> {
	match value {
		0 => Ok(LunarEndReason::None),
		1 => Ok(LunarEndReason::SafeLanding),
		2 => Ok(LunarEndReason::BodyImpact),
		3 => Ok(LunarEndReason::HardFootImpact),
		4 => Ok(LunarEndReason::OutOfBounds),
		5 => Ok(LunarEndReason::NumericalFailure),
		6 => Ok(LunarEndReason::TimeLimit),
		7 => Ok(LunarEndReason::ExternalStop),
		8 => Ok(LunarEndReason::InvalidAction),
		_ => Err(Error::failed_precondition(
			"Lunar Lander telemetry has invalid end reason",
		)),
	}
}

fn validate_terminal_state(reason: u32, terminated: u32, truncated: u32) -> Result<()> {
	let completed = terminated != 0 || truncated != 0;
	let truncated_reason = matches!(reason, 6 | 7);
	let terminated_reason = reason != 0 && !truncated_reason;
	if terminated > 1
		|| truncated > 1
		|| (terminated != 0 && truncated != 0)
		|| completed != (reason != 0)
		|| (terminated != 0) != terminated_reason
		|| (truncated != 0) != truncated_reason
	{
		return Err(Error::failed_precondition(
			"Lunar Lander telemetry has invalid terminal state",
		));
	}
	Ok(())
}
