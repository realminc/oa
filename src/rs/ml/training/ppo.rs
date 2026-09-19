//! Environment-neutral categorical Proximal Policy Optimization coordinator.
//!
//! Port provenance: OA C++ `oa/ml/ppoTrainer.{h,cpp}`. Environment stepping
//! remains caller-owned; this trainer composes one retained rollout, the
//! Actor-Critic behavior contract, reverse mode, and [`ItRolloutTraining`].

use crate::{Engine, Error, Matrix, Result, matrix};

use super::{
	ItRolloutTraining, ItRolloutTrainingConfig, Optimizer, RolloutTrainingPhase, TrainingCallback,
	TrainingMetric, TrainingSnapshot,
};
use crate::ml::{
	ActorCritic, GradientTape, PolicyResult, RolloutBatch, RolloutBuffer, RolloutConfig,
	RolloutTransition, advantage, loss, policy,
};

/// Complete categorical PPO collection and update policy.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PpoTrainerConfig {
	/// Number of complete rollout/update cycles.
	pub rollouts: u64,
	/// Time steps collected in each rollout.
	pub horizon: usize,
	/// Vector environments represented by each step.
	pub environments: usize,
	/// Full-batch optimizer passes over each finalized rollout.
	pub update_epochs: u64,
	/// Per-environment observation shape, excluding the environment axis.
	pub observation_shape: Vec<usize>,
	/// Base deterministic categorical sampling seed.
	pub seed: u64,
	/// Whether optimizer updates collect Vulkan timestamp statistics.
	pub enable_gpu_timing: bool,
	/// Generalized-advantage estimator policy.
	pub gae: advantage::GaeConfig,
	/// Clipped PPO objective coefficients.
	pub loss: loss::PpoLossConfig,
}

/// Scalar results from the last completed PPO update epoch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PpoTrainerMetrics {
	/// One-based count of fully updated rollouts.
	pub rollout: u64,
	/// One-based update epoch within the current/final rollout.
	pub update_epoch: u64,
	/// Last synchronized composite loss.
	pub total_loss: f32,
	/// Last synchronized clipped policy loss.
	pub policy_loss: f32,
	/// Last synchronized critic MSE.
	pub value_loss: f32,
	/// Last synchronized mean categorical entropy.
	pub entropy: f32,
}

/// Complete caller-driven categorical PPO lifecycle.
///
/// The Engine, model, and optimizer remain caller-owned. The trainer owns only
/// its fixed-capacity rollout and lifecycle state. Drop performs no submission,
/// waiting, or terminal callbacks; call [`PpoTrainer::finish`] explicitly.
#[must_use]
pub struct PpoTrainer<'engine, 'hooks> {
	engine: &'engine Engine,
	model: &'engine dyn ActorCritic,
	config: PpoTrainerConfig,
	observation_elements: usize,
	rollout: RolloutBuffer,
	training: ItRolloutTraining<'engine, 'hooks>,
	metrics: PpoTrainerMetrics,
	action_index: u64,
	collection_action_index: u64,
	collecting: bool,
	collection_abortable: bool,
}

impl<'engine, 'hooks> PpoTrainer<'engine, 'hooks> {
	/// Construct a checked PPO trainer and its retained rollout storage.
	///
	/// # Errors
	///
	/// Returns an error for invalid schedule/observation/loss policy, model
	/// ownership mismatch, rollout allocation, or optimizer ownership failure.
	pub fn new(
		engine: &'engine Engine,
		model: &'engine dyn ActorCritic,
		optimizer: &'engine mut dyn Optimizer,
		config: PpoTrainerConfig,
	) -> Result<Self> {
		let observation_elements = validate_config(&config)?;
		for parameter in model.all_parameters()? {
			if !engine.owns_matrix(&parameter.data()) {
				return Err(Error::invalid_argument(
					"PPO model parameters must belong to the selected engine",
				));
			}
		}
		let rollout = RolloutBuffer::new(
			engine,
			RolloutConfig {
				time: config.horizon,
				environments: config.environments,
				observation_shape: config.observation_shape.clone(),
			},
		)?;
		let training = ItRolloutTraining::new(
			engine,
			optimizer,
			ItRolloutTrainingConfig {
				rollouts: config.rollouts,
				horizon: config.horizon,
				environments: config.environments,
				update_epochs: config.update_epochs,
				timer_name: "ppo_update".into(),
				enable_gpu_timing: config.enable_gpu_timing,
			},
		)?;
		Ok(Self {
			engine,
			model,
			config,
			observation_elements,
			rollout,
			training,
			metrics: PpoTrainerMetrics::default(),
			action_index: 0,
			collection_action_index: 0,
			collecting: false,
			collection_abortable: false,
		})
	}

	/// Register one borrowed metric on the optimizer-update lifecycle.
	pub fn add_metric(&mut self, metric: &'hooks mut dyn TrainingMetric) {
		self.training.update_loop_mut().add_metric(metric);
	}

	/// Register one borrowed callback on the optimizer-update lifecycle.
	pub fn add_callback(&mut self, callback: &'hooks mut dyn TrainingCallback) {
		self.training.update_loop_mut().add_callback(callback);
	}

	/// Reset and begin the next collection cycle.
	///
	/// # Errors
	///
	/// Returns an error unless the trainer needs collection or reset recording
	/// fails.
	pub fn begin_collection(&mut self) -> Result<()> {
		if self.collecting || self.training.phase() != RolloutTrainingPhase::Collect {
			return Err(Error::failed_precondition(
				"PPO begin_collection requires an idle Collect phase",
			));
		}
		self.training.begin_rollout(&mut self.rollout)?;
		self.collection_action_index = self.action_index;
		self.collecting = true;
		self.collection_abortable = true;
		Ok(())
	}

	/// Sample one action and its stored policy statistics for a vector observation.
	///
	/// # Errors
	///
	/// Returns an error outside active collection, on seed overflow, or from
	/// observation reshape, model evaluation, or categorical sampling.
	pub fn act(&mut self, observation: &Matrix) -> Result<PolicyResult> {
		if !self.collecting {
			return Err(Error::failed_precondition(
				"PPO act requires active collection",
			));
		}
		let flat = matrix::reshape(
			observation,
			[self.config.environments, self.observation_elements],
		)?;
		let network = self.model.evaluate(&flat)?;
		self.action_index = self
			.action_index
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("PPO action seed index exhausted"))?;
		let seed = self
			.config
			.seed
			.checked_add(self.action_index)
			.ok_or_else(|| Error::resource_exhausted("PPO action seed exhausted"))?;
		policy::sample_categorical(&network.logits, &network.value, seed)
	}

	/// Append one environment step using statistics returned by [`Self::act`].
	///
	/// The next critic evaluation remains outside a gradient tape.
	///
	/// # Errors
	///
	/// Returns an error outside collection or for invalid observation, policy,
	/// transition, model, or device-recording contracts.
	#[allow(
		clippy::too_many_arguments,
		reason = "the PPO transition boundary keeps observation, reward, and both terminal meanings explicit"
	)]
	pub fn observe(
		&mut self,
		observation: &Matrix,
		next_observation: &Matrix,
		reward: &Matrix,
		terminated: &Matrix,
		truncated: &Matrix,
		policy_result: &PolicyResult,
	) -> Result<()> {
		if !self.collecting {
			return Err(Error::failed_precondition(
				"PPO observe requires active collection",
			));
		}
		let next_flat = matrix::reshape(
			next_observation,
			[self.config.environments, self.observation_elements],
		)?;
		let next = self.model.evaluate(&next_flat)?;
		self.rollout.append(&RolloutTransition::new(
			observation.clone(),
			policy_result.action.clone(),
			reward.clone(),
			policy_result.value.clone(),
			next.value,
			policy_result.log_probability.clone(),
			terminated.clone(),
			truncated.clone(),
		))
	}

	/// Finalize a full collection with GAE and enter Update.
	///
	/// # Errors
	///
	/// Returns an error outside collection, before the rollout is full, or when
	/// GAE recording fails.
	pub fn end_collection(&mut self) -> Result<()> {
		if !self.collecting {
			return Err(Error::failed_precondition(
				"PPO end_collection requires active collection",
			));
		}
		self
			.training
			.finalize_rollout(&mut self.rollout, self.config.gae)?;
		self.collecting = false;
		Ok(())
	}

	/// Rewind collection state after the enclosing Engine transaction rejects.
	///
	/// This remains valid after finalization until the first update begins.
	///
	/// # Errors
	///
	/// Returns an error when there is no abortable collection or the underlying
	/// rollout lifecycle has already begun updating.
	pub fn abort_collection(&mut self) -> Result<()> {
		if !self.collection_abortable {
			return Err(Error::failed_precondition(
				"PPO has no unsubmitted collection to abort",
			));
		}
		self.training.abort_rollout(&mut self.rollout)?;
		self.action_index = self.collection_action_index;
		self.collecting = false;
		self.collection_abortable = false;
		Ok(())
	}

	/// Perform one full-batch PPO update epoch.
	///
	/// Returns `false` when paused, stopped, or complete without authoring an
	/// update. The caller repeats this until collection is needed or training is
	/// done.
	///
	/// # Errors
	///
	/// Returns an error outside Update or from model evaluation, advantage
	/// normalization, loss, reverse mode, optimizer, runtime, metrics, or callbacks.
	pub fn update(&mut self) -> Result<bool> {
		if self.training.phase() != RolloutTrainingPhase::Update {
			return Err(Error::failed_precondition(
				"PPO update requires the Update phase",
			));
		}
		if !self.training.begin_update()? {
			return Ok(false);
		}
		self.collection_abortable = false;
		let result = self.record_update();
		if let Err(error) = result {
			self.engine.abort_pending_work();
			return self.training.fail_update(error);
		}
		Ok(true)
	}

	fn record_update(&mut self) -> Result<()> {
		let batch = self
			.config
			.horizon
			.checked_mul(self.config.environments)
			.ok_or_else(|| Error::resource_exhausted("PPO batch size overflows usize"))?;
		let rollout = self.rollout.batch();
		let observation = matrix::reshape(rollout.observation(), [batch, self.observation_elements])?;
		let action = matrix::reshape(rollout.action(), [batch])?;
		let old_log_probability = matrix::reshape(rollout.old_log_probability(), [batch])?;
		let advantage = matrix::reshape(rollout.advantage(), [batch])?;
		let target_return = matrix::reshape(rollout.returns(), [batch])?;

		self.training.update_loop().zero_grad();
		let tape = GradientTape::new();
		let network = self.model.evaluate(&observation)?;
		let policy = policy::evaluate_categorical(&network.logits, &action, &network.value)?;
		let normalized_advantage = advantage::normalize(&advantage, 1.0e-8)?;
		let loss = loss::ppo(
			&policy.log_probability,
			&old_log_probability,
			&normalized_advantage,
			&policy.value,
			&target_return,
			&policy.entropy,
			self.config.loss,
		)?;
		tape.backward(&loss.total_loss)?;
		self.training.complete_update(&loss.total_loss)?;
		self.metrics = PpoTrainerMetrics {
			rollout: self.training.rollout_index(),
			update_epoch: self.training.update_epoch(),
			total_loss: self
				.training
				.update_loop()
				.snapshot()
				.last_loss()
				.unwrap_or(0.0),
			policy_loss: loss.policy_loss.read_f32()?[0],
			value_loss: loss.value_loss.read_f32()?[0],
			entropy: loss.entropy.read_f32()?[0],
		};
		Ok(())
	}

	/// Return whether another collection must begin before an update.
	pub fn needs_collection(&self) -> bool {
		!self.collecting && self.training.phase() == RolloutTrainingPhase::Collect
	}

	/// Return whether every configured rollout/update cycle has completed.
	pub fn is_done(&self) -> bool {
		self.training.is_done()
	}

	/// Return the current Collect/Update/Complete phase.
	pub const fn phase(&self) -> RolloutTrainingPhase {
		self.training.phase()
	}

	/// Return the immutable PPO policy.
	pub const fn config(&self) -> &PpoTrainerConfig {
		&self.config
	}

	/// Return the last synchronized update metrics.
	pub const fn metrics(&self) -> PpoTrainerMetrics {
		self.metrics
	}

	/// Return the retained rollout matrices.
	pub const fn batch(&self) -> &RolloutBatch {
		self.rollout.batch()
	}

	/// Borrow the shared optimizer/update lifecycle.
	pub const fn training_loop(&self) -> &super::ItTraining<'engine, 'hooks> {
		self.training.update_loop()
	}

	/// Finish a complete/stopped PPO lifecycle and fire terminal callbacks.
	///
	/// # Errors
	///
	/// Returns an error before completion or from terminal callbacks.
	pub fn finish(self) -> Result<TrainingSnapshot> {
		self.training.finish()
	}
}

fn validate_config(config: &PpoTrainerConfig) -> Result<usize> {
	if config.rollouts == 0
		|| config.horizon == 0
		|| config.environments == 0
		|| config.update_epochs == 0
		|| config.observation_shape.is_empty()
	{
		return Err(Error::invalid_argument(
			"PPO requires nonzero rollout dimensions and a nonempty observation shape",
		));
	}
	let elements = config
		.observation_shape
		.iter()
		.try_fold(1_usize, |count, extent| count.checked_mul(*extent))
		.ok_or_else(|| Error::resource_exhausted("PPO observation size overflows usize"))?;
	if elements == 0 {
		return Err(Error::invalid_argument(
			"PPO observation shape must contain elements",
		));
	}
	config
		.horizon
		.checked_mul(config.environments)
		.ok_or_else(|| Error::resource_exhausted("PPO batch size overflows usize"))?;
	if !config.gae.gamma.is_finite()
		|| !(0.0..=1.0).contains(&config.gae.gamma)
		|| !config.gae.lambda.is_finite()
		|| !(0.0..=1.0).contains(&config.gae.lambda)
		|| !config.loss.clip_epsilon.is_finite()
		|| !(0.0..1.0).contains(&config.loss.clip_epsilon)
		|| !config.loss.value_coefficient.is_finite()
		|| config.loss.value_coefficient < 0.0
		|| !config.loss.entropy_coefficient.is_finite()
		|| config.loss.entropy_coefficient < 0.0
	{
		return Err(Error::invalid_argument("PPO GAE/loss policy is invalid"));
	}
	Ok(elements)
}
