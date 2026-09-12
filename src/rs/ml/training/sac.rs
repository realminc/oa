//! Fixed-temperature Soft Actor-Critic update coordinator.
//!
//! Port provenance: OA C++ `oa/ml/sacTrainer.{h,cpp}`. Actor and twin-critic
//! optimizer units remain separate [`ItTraining`] lifecycles while sharing the
//! caller's one Engine and replay owner.

use crate::{DType, Engine, Error, Matrix, Result, matrix};

use super::{ItTraining, ItTrainingConfig, Optimizer, target};
use crate::ml::{GradientTape, Module, ReplayBuffer, loss, policy};

/// Fixed SAC update, shape, action-range, target, and loss policy.
#[derive(Clone, Debug, PartialEq)]
pub struct SacTrainerConfig {
	/// Number of critic/actor update pairs to complete.
	pub updates: u64,
	/// Replay transitions consumed by each pair.
	pub batch_size: usize,
	/// Number of scalar continuous action dimensions.
	pub action_dimensions: usize,
	/// Completed-pair interval between exact critic-to-target copies.
	pub target_update_interval: u64,
	/// Per-transition observation shape expected from replay.
	pub observation_shape: Vec<usize>,
	/// Inclusive lower action mapping bound.
	pub action_minimum: f32,
	/// Inclusive upper action mapping bound.
	pub action_maximum: f32,
	/// Base deterministic replay and policy seed.
	pub seed: u64,
	/// Critic discount and fixed entropy coefficient.
	pub loss: loss::SacLossConfig,
}

impl Default for SacTrainerConfig {
	fn default() -> Self {
		Self {
			updates: 0,
			batch_size: 0,
			action_dimensions: 0,
			target_update_interval: 1,
			observation_shape: Vec::new(),
			action_minimum: -1.0,
			action_maximum: 1.0,
			seed: 0,
			loss: loss::SacLossConfig::default(),
		}
	}
}

/// Last completed SAC actor/critic update pair.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SacTrainerMetrics {
	/// One-based completed update-pair count.
	pub update: u64,
	/// Last synchronized actor loss.
	pub actor_loss: f32,
	/// Last synchronized twin-critic loss.
	pub critic_loss: f32,
}

/// Stateful fixed-temperature SAC coordinator over caller-owned components.
#[must_use]
pub struct SacTrainer<'engine, 'hooks> {
	engine: &'engine Engine,
	actor: &'engine dyn Module,
	critic1: &'engine dyn Module,
	critic2: &'engine dyn Module,
	target_critic1: &'engine dyn Module,
	target_critic2: &'engine dyn Module,
	replay: &'engine ReplayBuffer,
	config: SacTrainerConfig,
	observation_elements: usize,
	critic_training: ItTraining<'engine, 'hooks>,
	actor_training: ItTraining<'engine, 'hooks>,
	metrics: SacTrainerMetrics,
}

impl<'engine, 'hooks> SacTrainer<'engine, 'hooks> {
	/// Validate all owners, construct two training lifecycles, and synchronize targets.
	///
	/// # Errors
	///
	/// Returns an error for invalid shape/range/update policy, incompatible
	/// critic schemas, optimizer ownership, overflow, or target-copy failure.
	#[allow(
		clippy::too_many_arguments,
		reason = "the SAC owner contract keeps actor, twin critics, twin targets, and optimizers explicit"
	)]
	pub fn new(
		engine: &'engine Engine,
		actor: &'engine dyn Module,
		critic1: &'engine dyn Module,
		critic2: &'engine dyn Module,
		target_critic1: &'engine dyn Module,
		target_critic2: &'engine dyn Module,
		actor_optimizer: &'engine mut dyn Optimizer,
		critic_optimizer: &'engine mut dyn Optimizer,
		replay: &'engine ReplayBuffer,
		config: SacTrainerConfig,
	) -> Result<Self> {
		validate_config(replay, &config)?;
		let observation_elements = shape_elements(&config.observation_shape, "observation")?;
		target::sync_exact(
			engine,
			&[(critic1, target_critic1), (critic2, target_critic2)],
			"SAC critic",
		)?;
		let batch_size = u64::try_from(config.batch_size)
			.map_err(|_| Error::resource_exhausted("SAC batch size exceeds u64"))?;
		let critic_training = ItTraining::new_eager(
			engine,
			critic_optimizer,
			ItTrainingConfig {
				total_steps: config.updates,
				batch_size,
				timer_name: "sac_critic_update".into(),
				..ItTrainingConfig::default()
			},
		)?;
		let actor_training = ItTraining::new_eager(
			engine,
			actor_optimizer,
			ItTrainingConfig {
				total_steps: config.updates,
				batch_size,
				timer_name: "sac_actor_update".into(),
				..ItTrainingConfig::default()
			},
		)?;
		Ok(Self {
			engine,
			actor,
			critic1,
			critic2,
			target_critic1,
			target_critic2,
			replay,
			config,
			observation_elements,
			critic_training,
			actor_training,
			metrics: SacTrainerMetrics::default(),
		})
	}

	/// Complete one critic update followed by one actor update.
	///
	/// Returns `false` when either coordinated loop has stopped or exhausted its
	/// budget. Next-state policy and target-critic work is recorded before either
	/// tape, preserving the donor's detached Bellman target.
	///
	/// # Errors
	///
	/// Returns an error for insufficient replay, module output drift, loss,
	/// reverse-mode, optimizer, callback, synchronization, or runtime failure.
	pub fn update(&mut self) -> Result<bool> {
		if self.is_done() {
			return Ok(false);
		}
		if self.replay.len() < self.config.batch_size {
			return Err(Error::failed_precondition(
				"SAC replay does not contain one complete batch",
			));
		}
		if !self.actor_training.begin_step()? {
			self.critic_training.request_stop();
			return Ok(false);
		}
		if !self.critic_training.begin_step()? {
			self.actor_training.cancel_composed_step();
			return Ok(false);
		}

		if let Err(error) = self.record_update() {
			self.engine.abort_pending_work();
			self.actor_training.cancel_composed_step();
			return self.critic_training.fail_composed_step(error);
		}
		let update = self.critic_training.snapshot().step_count();
		self.metrics.update = update;
		if update.is_multiple_of(self.config.target_update_interval) {
			self.sync_targets()?;
		}
		Ok(true)
	}

	fn record_update(&mut self) -> Result<()> {
		let update = self.metrics.update;
		let sample_seed = checked_seed(self.config.seed, update + 1, "replay")?;
		let sampled = self.replay.sample(self.config.batch_size, sample_seed)?;
		let shape = vec![self.config.batch_size, self.observation_elements];
		let observation = matrix::reshape(sampled.observation(), shape.clone())?;
		let next_observation = matrix::reshape(sampled.next_observation(), shape)?;

		let next_policy = self.actor_policy(
			&next_observation,
			checked_seed(self.config.seed, 0x1_0000_0001 + update, "next policy")?,
		)?;
		let next_input = critic_input(&next_observation, &next_policy.action)?;
		let next_q1 = vector_q(
			self.target_critic1.forward(&next_input)?,
			self.config.batch_size,
		)?;
		let next_q2 = vector_q(
			self.target_critic2.forward(&next_input)?,
			self.config.batch_size,
		)?;

		self.critic_training.zero_grad();
		let critic_tape = GradientTape::new();
		let stored_input = critic_input(&observation, sampled.action())?;
		let q1 = vector_q(self.critic1.forward(&stored_input)?, self.config.batch_size)?;
		let q2 = vector_q(self.critic2.forward(&stored_input)?, self.config.batch_size)?;
		let critic = loss::sac_critic(
			&q1,
			&q2,
			sampled.reward(),
			&next_q1,
			&next_q2,
			&next_policy.log_probability,
			sampled.terminated(),
			sampled.truncated(),
			self.config.loss,
		)?;
		critic_tape.backward(&critic.total_loss)?;
		self.critic_training.complete_step(&critic.total_loss)?;
		self.metrics.critic_loss = self.critic_training.snapshot().last_loss().unwrap_or(0.0);

		self.actor_training.zero_grad();
		self.critic_training.zero_grad();
		let actor_tape = GradientTape::new();
		let actor_policy = self.actor_policy(
			&observation,
			checked_seed(self.config.seed, 0x2_0000_0001 + update, "actor policy")?,
		)?;
		let policy_input = critic_input(&observation, &actor_policy.action)?;
		let actor_q1 = vector_q(self.critic1.forward(&policy_input)?, self.config.batch_size)?;
		let actor_q2 = vector_q(self.critic2.forward(&policy_input)?, self.config.batch_size)?;
		let actor_loss = loss::sac_actor(
			&actor_q1,
			&actor_q2,
			&actor_policy.log_probability,
			self.config.loss.entropy_coefficient,
		)?;
		actor_tape.backward(&actor_loss)?;
		self.actor_training.complete_step(&actor_loss)?;
		self.metrics.actor_loss = self.actor_training.snapshot().last_loss().unwrap_or(0.0);
		Ok(())
	}

	fn actor_policy(
		&self,
		observation: &Matrix,
		seed: u64,
	) -> Result<policy::ContinuousPolicyResult> {
		let output = self.actor.forward(observation)?;
		let expected_width = self
			.config
			.action_dimensions
			.checked_mul(2)
			.ok_or_else(|| Error::resource_exhausted("SAC actor width overflows usize"))?;
		if output.shape() != [self.config.batch_size, expected_width]
			|| output.dtype() != DType::F32
		{
			return Err(Error::invalid_argument(
				"SAC actor must return F32 [batch, 2 * action_dimensions]",
			));
		}
		let split = i64::try_from(self.config.action_dimensions)
			.map_err(|_| Error::resource_exhausted("SAC action width exceeds i64"))?;
		let end = i64::try_from(expected_width)
			.map_err(|_| Error::resource_exhausted("SAC actor width exceeds i64"))?;
		let mean = matrix::slice(&output, 1, 0, split)?;
		let log_stddev = matrix::slice(&output, 1, split, end)?;
		let value = matrix::full(self.engine, [self.config.batch_size], 0.0)?;
		policy::sample_tanh_normal(
			&mean,
			&log_stddev,
			&value,
			self.config.action_minimum,
			self.config.action_maximum,
			seed,
			1.0e-6,
		)
	}

	/// Exactly synchronize both online critics into their target peers.
	pub fn sync_targets(&self) -> Result<()> {
		target::sync_exact(
			self.engine,
			&[
				(self.critic1, self.target_critic1),
				(self.critic2, self.target_critic2),
			],
			"SAC critic",
		)
	}

	/// Return whether either coordinated loop has stopped or reached its budget.
	pub fn is_done(&self) -> bool {
		let critic = self.critic_training.snapshot().step_count();
		let actor = self.actor_training.snapshot().step_count();
		self.critic_training.stop_requested()
			|| self.actor_training.stop_requested()
			|| critic >= self.config.updates
			|| actor >= self.config.updates
	}

	/// Return the last completed update-pair metrics.
	pub const fn metrics(&self) -> SacTrainerMetrics {
		self.metrics
	}

	/// Borrow the primary critic update lifecycle.
	pub const fn training_loop(&self) -> &ItTraining<'engine, 'hooks> {
		&self.critic_training
	}

	/// Mutably borrow the primary critic update lifecycle.
	pub fn training_loop_mut(&mut self) -> &mut ItTraining<'engine, 'hooks> {
		&mut self.critic_training
	}

	/// Borrow the independently observable actor update lifecycle.
	pub const fn actor_training_loop(&self) -> &ItTraining<'engine, 'hooks> {
		&self.actor_training
	}

	/// Mutably borrow the actor update lifecycle.
	pub fn actor_training_loop_mut(&mut self) -> &mut ItTraining<'engine, 'hooks> {
		&mut self.actor_training
	}

	/// Fire both explicit terminal boundaries and return critic then actor snapshots.
	///
	/// # Errors
	///
	/// Returns an error for incomplete steps or terminal callback failure.
	pub fn finish(self) -> Result<(super::TrainingSnapshot, super::TrainingSnapshot)> {
		let critic = self.critic_training.finish()?;
		let actor = self.actor_training.finish()?;
		Ok((critic, actor))
	}
}

fn validate_config(replay: &ReplayBuffer, config: &SacTrainerConfig) -> Result<()> {
	let replay_config = replay.config();
	if config.updates == 0
		|| config.batch_size == 0
		|| config.action_dimensions == 0
		|| config.target_update_interval == 0
		|| config.observation_shape != replay_config.observation_shape
		|| replay_config.action_dtype != DType::F32
		|| replay_config.action_shape != [config.action_dimensions]
		|| !config.action_minimum.is_finite()
		|| !config.action_maximum.is_finite()
		|| config.action_minimum >= config.action_maximum
	{
		return Err(Error::invalid_argument(
			"SAC trainer configuration does not match continuous replay storage",
		));
	}
	shape_elements(&config.observation_shape, "observation")?;
	Ok(())
}

fn shape_elements(shape: &[usize], label: &'static str) -> Result<usize> {
	shape.iter().try_fold(1_usize, |count, extent| {
		count
			.checked_mul(*extent)
			.ok_or_else(|| Error::resource_exhausted(format!("SAC {label} size overflows usize")))
	})
}

fn checked_seed(base: u64, offset: u64, label: &'static str) -> Result<u64> {
	base.checked_add(offset)
		.ok_or_else(|| Error::resource_exhausted(format!("SAC {label} seed exhausted")))
}

fn critic_input(observation: &Matrix, action: &Matrix) -> Result<Matrix> {
	matrix::concat(&[observation.clone(), action.clone()], 1)
}

fn vector_q(input: Matrix, batch_size: usize) -> Result<Matrix> {
	if input.dtype() != DType::F32 {
		return Err(Error::invalid_argument("SAC critic output must be F32"));
	}
	if input.shape() == [batch_size] {
		return Ok(input);
	}
	if input.shape() == [batch_size, 1] {
		return matrix::reshape(&input, [batch_size]);
	}
	Err(Error::invalid_argument(
		"SAC critics must return [batch] or [batch, 1]",
	))
}
