//! SDK-owned vectorized CartPole-v1 environment.

use crate::ml::environment::{Environment, EnvironmentExecution};
use crate::ml::lowering::environment as lowering;
use crate::ml::{EnvironmentSpace, EnvironmentSpec, EnvironmentTransition};
use crate::{DType, Engine, Error, Matrix, Result};

/// Versioned dynamics and vector-lane policy for CartPole-v1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CartPoleConfig {
	/// Number of independent GPU environment lanes.
	pub environments: u32,
	/// Time-limit truncation threshold per episode.
	pub max_episode_steps: u32,
	/// Deterministic reset seed.
	pub seed: u64,
	/// Gravitational acceleration.
	pub gravity: f32,
	/// Cart mass.
	pub cart_mass: f32,
	/// Pole mass.
	pub pole_mass: f32,
	/// Half pole length used by the standard equations.
	pub half_pole_length: f32,
	/// Magnitude of the left/right action force.
	pub force_magnitude: f32,
	/// Euler integration interval.
	pub time_step: f32,
	/// Absolute cart-position termination threshold.
	pub position_threshold: f32,
	/// Absolute pole-angle termination threshold in radians.
	pub angle_threshold_radians: f32,
}

impl CartPoleConfig {
	/// Version of the complete step-dynamics identity.
	pub const DYNAMICS_VERSION: u32 = 1;

	/// Return a stable identity over every behavior-changing dynamics field.
	pub fn dynamics_identity(self) -> u64 {
		const OFFSET: u64 = 14_695_981_039_346_656_037;
		const PRIME: u64 = 1_099_511_628_211;
		fn hash_u32(mut hash: u64, value: u32) -> u64 {
			for byte in value.to_le_bytes() {
				hash ^= u64::from(byte);
				hash = hash.wrapping_mul(PRIME);
			}
			hash
		}
		[
			Self::DYNAMICS_VERSION,
			self.max_episode_steps,
			self.gravity.to_bits(),
			self.cart_mass.to_bits(),
			self.pole_mass.to_bits(),
			self.half_pole_length.to_bits(),
			self.force_magnitude.to_bits(),
			self.time_step.to_bits(),
			self.position_threshold.to_bits(),
			self.angle_threshold_radians.to_bits(),
		]
		.into_iter()
		.fold(OFFSET, hash_u32)
	}
}

impl Default for CartPoleConfig {
	fn default() -> Self {
		Self {
			environments: 1,
			max_episode_steps: 500,
			seed: 1,
			gravity: 9.8,
			cart_mass: 1.0,
			pole_mass: 0.1,
			half_pole_length: 0.5,
			force_magnitude: 10.0,
			time_step: 0.02,
			position_threshold: 2.4,
			angle_threshold_radians: 0.209_439_52,
		}
	}
}

/// Vectorized GPU CartPole workload over the reusable Environment lifecycle.
#[must_use]
pub struct CartPole<'engine> {
	engine: &'engine Engine,
	execution: EnvironmentExecution,
	config: CartPoleConfig,
	spec: EnvironmentSpec,
	state: Matrix,
	transition_observation: Matrix,
	reward: Matrix,
	terminated: Matrix,
	truncated: Matrix,
	done: Matrix,
	episode_steps: Matrix,
	episode_index: Matrix,
	pending_seed: Option<u64>,
	has_committed_state: bool,
	has_pending_full_reset: bool,
}

impl<'engine> CartPole<'engine> {
	/// Construct CartPole storage and record its deterministic initial reset.
	///
	/// # Errors
	///
	/// Returns an error for invalid dynamics, allocation, schema, or recording.
	pub fn new(engine: &'engine Engine, config: CartPoleConfig) -> Result<Self> {
		validate_config(config)?;
		let environments = config.environments as usize;
		let state = Matrix::from_f32(engine, [environments, 4], &vec![0.0; environments * 4])?;
		let transition_observation =
			Matrix::from_f32(engine, [environments, 4], &vec![0.0; environments * 4])?;
		let reward = Matrix::from_f32(engine, [environments], &vec![0.0; environments])?;
		let terminated = Matrix::from_slice(engine, [environments], &vec![0_u8; environments])?;
		let truncated = Matrix::from_slice(engine, [environments], &vec![0_u8; environments])?;
		let done = Matrix::from_slice(engine, [environments], &vec![0_u8; environments])?;
		let episode_steps = Matrix::from_slice(engine, [environments], &vec![0_u32; environments])?;
		let episode_index = Matrix::from_slice(engine, [environments], &vec![0_u32; environments])?;
		let spec = EnvironmentSpec::new(
			EnvironmentSpace::continuous(
				"observation",
				[4],
				DType::F32,
				f64::NEG_INFINITY,
				f64::INFINITY,
			)?,
			EnvironmentSpace::discrete("action", 2, DType::I32)?,
			EnvironmentSpace::continuous("reward", [], DType::F32, 0.0, 1.0)?,
			EnvironmentSpace::binary("terminated", [])?,
			EnvironmentSpace::binary("truncated", [])?,
		)?;
		let mut result = Self {
			engine,
			execution: EnvironmentExecution::new(engine),
			config,
			spec,
			state,
			transition_observation,
			reward,
			terminated,
			truncated,
			done,
			episode_steps,
			episode_index,
			pending_seed: None,
			has_committed_state: false,
			has_pending_full_reset: false,
		};
		Environment::reset(&mut result, config.seed)?;
		Ok(result)
	}

	/// Return the complete dynamics configuration.
	pub const fn config(&self) -> CartPoleConfig {
		self.config
	}

	/// Return the done mask used by completed-lane reset.
	pub const fn done(&self) -> &Matrix {
		&self.done
	}

	/// Return per-lane steps in the current episode.
	pub const fn episode_steps(&self) -> &Matrix {
		&self.episode_steps
	}

	/// Return per-lane deterministic episode indices.
	pub const fn episode_index(&self) -> &Matrix {
		&self.episode_index
	}

	fn effective_seed(&self) -> u64 {
		self.pending_seed.unwrap_or(self.config.seed)
	}

	fn record_reset_impl(&mut self, only_done: bool) -> Result<()> {
		if only_done && !self.has_committed_state && !self.has_pending_full_reset {
			return Err(Error::failed_precondition(
				"CartPole completed reset requires submitted state or an earlier full reset",
			));
		}
		if !only_done {
			self.has_pending_full_reset = true;
		}
		lowering::cart_pole_reset(
			&self.done,
			&self.state,
			&self.episode_steps,
			&self.episode_index,
			self.config.environments,
			self.effective_seed(),
			only_done,
		)
	}
}

impl Environment for CartPole<'_> {
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
		&self.state
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
		if !self.has_committed_state && !self.has_pending_full_reset {
			return Err(Error::failed_precondition(
				"CartPole step requires submitted state or an earlier full reset",
			));
		}
		lowering::cart_pole_step(
			action,
			&self.state,
			&self.transition_observation,
			&self.reward,
			&self.terminated,
			&self.truncated,
			&self.done,
			&self.episode_steps,
			self.config.environments,
			self.config.max_episode_steps,
			self.config.gravity,
			self.config.cart_mass,
			self.config.pole_mass,
			self.config.half_pole_length,
			self.config.force_magnitude,
			self.config.time_step,
			self.config.position_threshold,
			self.config.angle_threshold_radians,
			self.config.dynamics_identity(),
		)?;
		Ok(EnvironmentTransition::new(
			self.transition_observation.clone(),
			self.state.clone(),
			self.reward.clone(),
			self.terminated.clone(),
			self.truncated.clone(),
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

fn validate_config(config: CartPoleConfig) -> Result<()> {
	let finite = [
		config.gravity,
		config.cart_mass,
		config.pole_mass,
		config.half_pole_length,
		config.force_magnitude,
		config.time_step,
		config.position_threshold,
		config.angle_threshold_radians,
	]
	.into_iter()
	.all(f32::is_finite);
	if !finite
		|| config.environments == 0
		|| config.max_episode_steps == 0
		|| config.environments > u32::MAX / 4
		|| config.cart_mass <= 0.0
		|| config.pole_mass <= 0.0
		|| config.half_pole_length <= 0.0
		|| config.force_magnitude <= 0.0
		|| config.time_step <= 0.0
		|| config.position_threshold <= 0.0
		|| config.angle_threshold_radians <= 0.0
	{
		return Err(Error::invalid_argument(
			"CartPole received an invalid environment configuration",
		));
	}
	Ok(())
}
