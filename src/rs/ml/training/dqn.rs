//! Environment-neutral DQN update coordinator.
//!
//! Port provenance: OA C++ `oa/ml/dqnTrainer.{h,cpp}`. The trainer composes
//! replay, modules, reverse mode, an optimizer, and [`ItTraining`]; it owns no
//! second scheduler or execution runtime.

use crate::{DType, Engine, Error, Result, matrix};

use super::{ItTraining, ItTrainingConfig, Optimizer, TrainingCallback, TrainingMetric, target};
use crate::ml::{GradientTape, Module, ReplayBuffer, loss};

/// Fixed update budget, replay batch, target cadence, and Bellman policy.
#[derive(Clone, Debug, PartialEq)]
pub struct DqnTrainerConfig {
	/// Number of optimizer updates to complete.
	pub updates: u64,
	/// Replay transitions consumed by one update.
	pub batch_size: usize,
	/// Completed-update interval between exact online-to-target copies.
	pub target_update_interval: u64,
	/// Per-transition observation shape expected from replay.
	pub observation_shape: Vec<usize>,
	/// Base seed used by deterministic replay sampling.
	pub seed: u64,
	/// Bellman-target configuration.
	pub loss: loss::DqnLossConfig,
}

impl Default for DqnTrainerConfig {
	fn default() -> Self {
		Self {
			updates: 0,
			batch_size: 0,
			target_update_interval: 100,
			observation_shape: Vec::new(),
			seed: 0,
			loss: loss::DqnLossConfig::default(),
		}
	}
}

/// Last completed DQN update and its synchronized scalar loss.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DqnTrainerMetrics {
	/// One-based completed update count.
	pub update: u64,
	/// Last mean Smooth-L1 Bellman loss.
	pub loss: f32,
}

/// Stateful DQN training coordinator over caller-owned components.
///
/// The online module, target module, optimizer, replay storage, and engine
/// remain caller-owned and must outlive this coordinator. Dropping the trainer
/// does not submit, wait, synchronize targets, or fire terminal callbacks; use
/// [`DqnTrainer::finish`] for the explicit terminal boundary.
#[must_use]
pub struct DqnTrainer<'engine, 'hooks> {
	engine: &'engine Engine,
	online: &'engine dyn Module,
	target: &'engine dyn Module,
	replay: &'engine ReplayBuffer,
	config: DqnTrainerConfig,
	observation_elements: usize,
	training: ItTraining<'engine, 'hooks>,
	metrics: DqnTrainerMetrics,
}

impl<'engine, 'hooks> DqnTrainer<'engine, 'hooks> {
	/// Validate and construct a trainer, then deep-copy the online model into the target.
	///
	/// # Errors
	///
	/// Returns an error for invalid update settings, a replay contract mismatch,
	/// incompatible module schemas, optimizer ownership failure, overflow, or
	/// device-copy failure.
	pub fn new(
		engine: &'engine Engine,
		online: &'engine dyn Module,
		target: &'engine dyn Module,
		optimizer: &'engine mut dyn Optimizer,
		replay: &'engine ReplayBuffer,
		config: DqnTrainerConfig,
	) -> Result<Self> {
		validate_config(replay, &config)?;
		let observation_elements = observation_elements(&config.observation_shape)?;
		target::sync_exact(engine, &[(online, target)], "DQN")?;
		let batch_size = u64::try_from(config.batch_size)
			.map_err(|_| Error::resource_exhausted("DQN batch size exceeds u64"))?;
		let training = ItTraining::new_eager(
			engine,
			optimizer,
			ItTrainingConfig {
				total_steps: config.updates,
				batch_size,
				timer_name: "dqn_update".into(),
				..ItTrainingConfig::default()
			},
		)?;
		Ok(Self {
			engine,
			online,
			target,
			replay,
			config,
			observation_elements,
			training,
			metrics: DqnTrainerMetrics::default(),
		})
	}

	/// Register one borrowed metric on the composed training lifecycle.
	pub fn add_metric(&mut self, metric: &'hooks mut dyn TrainingMetric) {
		self.training.add_metric(metric);
	}

	/// Register one borrowed callback on the composed training lifecycle.
	pub fn add_callback(&mut self, callback: &'hooks mut dyn TrainingCallback) {
		self.training.add_callback(callback);
	}

	/// Complete one DQN replay update.
	///
	/// Returns `false` without doing work after the update budget or a callback
	/// stop decision. Target inference is authored before the tape is opened, so
	/// Bellman targets remain detached without a second no-grad mechanism.
	///
	/// # Errors
	///
	/// Returns an error when replay has less than one batch or any replay,
	/// module, loss, reverse-mode, optimizer, callback, or runtime step fails.
	pub fn update(&mut self) -> Result<bool> {
		if self.is_done() {
			return Ok(false);
		}
		if self.replay.len() < self.config.batch_size {
			return Err(Error::failed_precondition(
				"DQN replay does not contain one complete batch",
			));
		}
		if !self.training.begin_step()? {
			return Ok(false);
		}

		let result = self.record_update();
		if let Err(error) = result {
			self.engine.abort_pending_work();
			return self.training.fail_composed_step(error);
		}

		let update = self.training.snapshot().step_count();
		self.metrics = DqnTrainerMetrics {
			update,
			loss: self.training.snapshot().last_loss().unwrap_or(0.0),
		};
		if update.is_multiple_of(self.config.target_update_interval) {
			self.sync_target()?;
		}
		Ok(true)
	}

	fn record_update(&mut self) -> Result<()> {
		let step = self.training.snapshot().step_count();
		let seed = self
			.config
			.seed
			.checked_add(step - 1)
			.ok_or_else(|| Error::resource_exhausted("DQN replay seed exhausted"))?;
		let sampled = self.replay.sample(self.config.batch_size, seed)?;
		let shape = vec![self.config.batch_size, self.observation_elements];
		let observation = matrix::reshape(sampled.observation(), shape.clone())?;
		let next_observation = matrix::reshape(sampled.next_observation(), shape)?;
		let next_q = self.target.forward(&next_observation)?;

		self.training.zero_grad();
		let tape = GradientTape::new();
		let q = self.online.forward(&observation)?;
		let result = loss::dqn(
			&q,
			sampled.action(),
			sampled.reward(),
			&next_q,
			sampled.terminated(),
			sampled.truncated(),
			self.config.loss,
		)?;
		tape.backward(&result.loss)?;
		self.training.complete_step(&result.loss)
	}

	/// Deep-copy every online parameter into its matching target parameter.
	///
	/// # Errors
	///
	/// Returns an error for schema drift, copy submission, or completion failure.
	pub fn sync_target(&self) -> Result<()> {
		target::sync_exact(self.engine, &[(self.online, self.target)], "DQN")
	}

	/// Return whether the configured budget or a callback stop has been reached.
	pub fn is_done(&self) -> bool {
		let snapshot = self.training.snapshot();
		self.training.stop_requested() || snapshot.step_count() >= self.config.updates
	}

	/// Return the last completed update metrics.
	pub const fn metrics(&self) -> DqnTrainerMetrics {
		self.metrics
	}

	/// Borrow the composed lifecycle for callbacks, sessions, and diagnostics.
	pub const fn training_loop(&self) -> &ItTraining<'engine, 'hooks> {
		&self.training
	}

	/// Mutably borrow the composed lifecycle before or between steps.
	pub fn training_loop_mut(&mut self) -> &mut ItTraining<'engine, 'hooks> {
		&mut self.training
	}

	/// Fire the explicit terminal callback boundary and return its final snapshot.
	///
	/// # Errors
	///
	/// Returns an error for an incomplete step or terminal callback failure.
	pub fn finish(self) -> Result<super::TrainingSnapshot> {
		self.training.finish()
	}
}

fn validate_config(replay: &ReplayBuffer, config: &DqnTrainerConfig) -> Result<()> {
	let replay_config = replay.config();
	if config.updates == 0
		|| config.batch_size == 0
		|| config.target_update_interval == 0
		|| config.observation_shape != replay_config.observation_shape
		|| !replay_config.action_shape.is_empty()
		|| replay_config.action_dtype != DType::I32
	{
		return Err(Error::invalid_argument(
			"DQN trainer expects positive update settings and scalar I32 replay actions",
		));
	}
	Ok(())
}

fn observation_elements(shape: &[usize]) -> Result<usize> {
	shape.iter().try_fold(1_usize, |count, extent| {
		count
			.checked_mul(*extent)
			.ok_or_else(|| Error::resource_exhausted("DQN observation size overflows usize"))
	})
}
