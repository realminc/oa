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
	/// Maximum sequences evaluated by one update tape; zero means full batch.
	/// Minibatches accumulate sample-weighted gradients with unchanged parameters
	/// and one optimizer step per epoch. Advantages normalize over the rollout.
	/// Each minibatch completes before the next to bound live device memory.
	pub update_minibatch_size: usize,
	/// Per-environment observation shape, excluding the environment axis.
	pub observation_shape: Vec<usize>,
	/// Base deterministic categorical sampling seed.
	pub seed: u64,
	/// Whether optimizer updates collect Vulkan timestamp statistics.
	/// Time all update submissions, summing disjoint minibatch and optimizer
	/// intervals into one sample per epoch. Collection remains caller-owned.
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
	compiled: Option<&'hooks mut super::PpoUpdateProgram>,
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
		let training = ItRolloutTraining::new(engine, optimizer, rollout_training_config(&config))?;
		Self::new_with_training(engine, model, config, training, None)
	}

	/// Construct PPO with a reusable bounded AdamW update program.
	/// Keep `program` across rollout/checkpoint boundaries; compilation happens
	/// once per fixed topology/geometry, not once per minibatch or rollout.
	/// # Errors
	/// Returns the errors of `new`; update capture failures are reported by `update`.
	pub fn new_compiled(
		engine: &'engine Engine,
		model: &'engine dyn ActorCritic,
		optimizer: &'engine mut super::AdamW,
		program: &'hooks mut super::PpoUpdateProgram,
		config: PpoTrainerConfig,
	) -> Result<Self> {
		let training =
			ItRolloutTraining::new_compiled(engine, optimizer, rollout_training_config(&config))?;
		Self::new_with_training(engine, model, config, training, Some(program))
	}

	fn new_with_training(
		engine: &'engine Engine,
		model: &'engine dyn ActorCritic,
		config: PpoTrainerConfig,
		mut training: ItRolloutTraining<'engine, 'hooks>,
		mut compiled: Option<&'hooks mut super::PpoUpdateProgram>,
	) -> Result<Self> {
		let observation_elements = validate_config(&config)?;
		for parameter in model.all_parameters()? {
			if !engine.owns_matrix(&parameter.data()) {
				return Err(Error::invalid_argument(
					"PPO model must belong to selected engine",
				));
			}
		}
		let expected = RolloutConfig {
			time: config.horizon,
			environments: config.environments,
			observation_shape: config.observation_shape.clone(),
		};
		let rollout = match compiled.as_deref_mut() {
			Some(program) => match program.rollout.take() {
				Some(rollout) if rollout.config() == &expected => rollout,
				Some(rollout) => {
					program.rollout = Some(rollout);
					return Err(Error::failed_precondition(
						"compiled PPO rollout geometry changed",
					));
				}
				None if program.is_compiled() => {
					return Err(Error::failed_precondition(
						"compiled PPO trainer must finish before reuse",
					));
				}
				None => RolloutBuffer::new(engine, expected)?,
			},
			None => RolloutBuffer::new(engine, expected)?,
		};
		if compiled.is_some()
			|| (config.update_minibatch_size != 0
				&& config.update_minibatch_size < config.horizon * config.environments)
		{
			training.update_loop_mut().disable_eager_resource_reuse();
		}
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
			compiled,
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

	/// Collect a complete trading rollout through fixed reset/step/GAE programs.
	/// The final decision closes every lane exactly as `step_terminal` does.
	/// # Errors
	/// Rejects changed owners/geometry, stale execution, unhealthy lanes or any
	/// replay failure. Submitted collection is not rollbackable; resume from the
	/// last completed checkpoint after failure.
	pub fn collect_trading(
		&mut self,
		environment: &mut crate::ml::environment::TradingEnvironment<'_>,
		program: &mut super::TradingRolloutProgram,
		reset_seed: u64,
		check_every: u32,
		reward_offset: f32,
	) -> Result<std::time::Duration> {
		self.collect_trading_impl(
			environment,
			program,
			super::trading_rollout::CollectionOptions {
				seed: reset_seed,
				check_every,
				shaping: reward_offset,
				continuing: false,
			},
		)
	}

	/// Collect a bounded PPO chunk while retaining positions, pending execution,
	/// account state and episode cursor across chunks. Only actual environment
	/// episode boundaries reset lanes; GAE bootstraps the final nonterminal value.
	/// # Errors
	/// Same ownership/geometry/completion gates as `collect_trading`.
	pub fn collect_trading_continuing(
		&mut self,
		environment: &mut crate::ml::environment::TradingEnvironment<'_>,
		program: &mut super::TradingRolloutProgram,
		reset_seed: u64,
		check_every: u32,
		reward_offset: f32,
	) -> Result<std::time::Duration> {
		self.collect_trading_impl(
			environment,
			program,
			super::trading_rollout::CollectionOptions {
				seed: reset_seed,
				check_every,
				shaping: reward_offset,
				continuing: true,
			},
		)
	}

	fn collect_trading_impl(
		&mut self,
		environment: &mut crate::ml::environment::TradingEnvironment<'_>,
		program: &mut super::TradingRolloutProgram,
		options: super::trading_rollout::CollectionOptions,
	) -> Result<std::time::Duration> {
		if self.collecting || self.training.phase() != RolloutTrainingPhase::Collect {
			return Err(Error::failed_precondition(
				"compiled trading collection requires idle Collect phase",
			));
		}
		self.training.begin_replayed_rollout(&mut self.rollout)?;
		self.collection_action_index = self.action_index;
		self.collecting = true;
		self.collection_abortable = false;
		let mut config = self.config.clone();
		config.seed = config
			.seed
			.checked_add(self.action_index)
			.ok_or_else(|| Error::resource_exhausted("PPO action seed exhausted"))?;
		let elapsed = program.collect(
			self.engine,
			self.model,
			environment,
			&mut self.rollout,
			&config,
			options,
		)?;
		self.action_index = self
			.action_index
			.checked_add(self.config.horizon as u64)
			.ok_or_else(|| Error::resource_exhausted("PPO action cursor exhausted"))?;
		self
			.training
			.finalize_rollout(&mut self.rollout, self.config.gae)?;
		self.collecting = false;
		Ok(elapsed)
	}

	/// Number of vector decisions consumed by this trainer's policy RNG stream.
	pub const fn action_index(&self) -> u64 {
		self.action_index
	}

	/// Restore a persisted policy cursor before collecting the next chunk.
	/// # Errors
	/// Requires idle Collect phase and an unexhausted checked seed range.
	pub fn restore_action_index(&mut self, index: u64) -> Result<()> {
		if self.collecting || self.training.phase() != RolloutTrainingPhase::Collect {
			return Err(Error::failed_precondition(
				"PPO cursor restore requires idle Collect phase",
			));
		}
		self
			.config
			.seed
			.checked_add(index)
			.and_then(|v| v.checked_add(self.config.horizon as u64))
			.ok_or_else(|| Error::resource_exhausted("PPO restored seed exhausted"))?;
		self.action_index = index;
		self.collection_action_index = index;
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

	/// Perform one PPO epoch with sample-weighted gradient accumulation.
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
		// Collection is caller-owned. Flush its pending reset/GAE work before
		// begin_update opens the next stable resource frame. This keeps it out
		// of update timestamps without submitting an unsealed training frame.
		if self.config.enable_gpu_timing
			&& self.engine.has_pending_work()
			&& let Err(error) = self.engine.checkpoint().and_then(|event| event.wait())
		{
			self.engine.abort_pending_work();
			return self.training.fail_update(error);
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
		if let Some(program) = self.compiled.as_deref_mut() {
			self.metrics = program.update(
				self.engine,
				self.model,
				&self.config,
				self.rollout.batch(),
				&mut self.training,
			)?;
			return Ok(());
		}
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

		let normalized_advantage = advantage::normalize(&advantage, 1.0e-8)?;
		let minibatch = if self.config.update_minibatch_size == 0 {
			batch
		} else {
			self.config.update_minibatch_size
		};
		self.training.update_loop().zero_grad();
		let mut metrics = [0.0_f64; 4];
		let mut device_prefix = std::time::Duration::ZERO;
		for start in (0..batch).step_by(minibatch) {
			let end = start.saturating_add(minibatch).min(batch);
			let count = end - start;
			let [observations, actions, old_logs, advantages, returns] = if count == batch {
				[
					observation.clone(),
					action.clone(),
					old_log_probability.clone(),
					normalized_advantage.clone(),
					target_return.clone(),
				]
			} else {
				crate::ml::rollout::minibatch(
					[
						&observation,
						&action,
						&old_log_probability,
						&normalized_advantage,
						&target_return,
					],
					start,
					count,
				)?
			};
			let tape = GradientTape::new();
			let network = self.model.evaluate(&observations)?;
			let policy = policy::evaluate_categorical(&network.logits, &actions, &network.value)?;
			let loss = loss::ppo(
				&policy.log_probability,
				&old_logs,
				&advantages,
				&policy.value,
				&returns,
				&policy.entropy,
				self.config.loss,
			)?;
			let total_loss = self.model.auxiliary_loss(&loss.total_loss)?;
			let weight = count as f64 / batch as f64;
			let weighted = matrix::scale(&total_loss, weight as f32)?;
			tape.backward(&weighted)?;
			// Exact completion releases the preceding tape's transient storage.
			// Without this boundary Engine would retain every minibatch at once.
			if minibatch == batch {
				self.training.complete_update(&total_loss)?;
			} else if self.config.enable_gpu_timing {
				let elapsed = self.engine.checkpoint_timed()?.device_duration()?;
				device_prefix = device_prefix
					.checked_add(elapsed)
					.ok_or_else(|| Error::resource_exhausted("PPO minibatch device time overflow"))?;
			} else {
				self.engine.checkpoint()?.wait()?;
			}
			for (sum, value) in metrics.iter_mut().zip([
				&total_loss,
				&loss.policy_loss,
				&loss.value_loss,
				&loss.entropy,
			]) {
				let scalar = value.read_f32()?[0];
				if !scalar.is_finite() {
					return Err(Error::failed_precondition(
						"PPO update metric is not finite",
					));
				}
				*sum += weight * f64::from(scalar);
			}
		}
		if minibatch < batch {
			let total_loss = Matrix::new_from_f32(self.engine, [], &[metrics[0] as f32])?;
			self
				.training
				.complete_update_with_device_prefix(&total_loss, device_prefix)?;
		}
		self.metrics = PpoTrainerMetrics {
			rollout: self.training.rollout_index(),
			update_epoch: self.training.update_epoch(),
			total_loss: metrics[0] as f32,
			policy_loss: metrics[1] as f32,
			value_loss: metrics[2] as f32,
			entropy: metrics[3] as f32,
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
		let snapshot = self.training.finish()?;
		if let Some(program) = self.compiled {
			program.rollout = Some(self.rollout);
		}
		Ok(snapshot)
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
	let batch = config.horizon * config.environments;
	if config.update_minibatch_size > batch {
		return Err(Error::invalid_argument(
			"PPO update minibatch size exceeds rollout batch",
		));
	}
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

fn rollout_training_config(config: &PpoTrainerConfig) -> ItRolloutTrainingConfig {
	ItRolloutTrainingConfig {
		rollouts: config.rollouts,
		horizon: config.horizon,
		environments: config.environments,
		update_epochs: config.update_epochs,
		timer_name: "ppo_update".into(),
		enable_gpu_timing: config.enable_gpu_timing,
	}
}
