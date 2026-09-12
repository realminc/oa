//! Same-device rollout collection sessions.

use crate::{Error, Event, Result, matrix};

use super::{
	ActorCritic, EnvironmentSpaceKind, RolloutBuffer, RolloutTransition, advantage::GaeConfig,
	environment::Environment, policy,
};

/// Fixed categorical collection horizon and sampling policy.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RolloutCollectorConfig {
	/// Number of vector-environment steps in each collection.
	pub horizon: usize,
	/// Base seed for initial reset and categorical samples.
	pub seed: u64,
	/// Generalized-advantage estimator policy.
	pub gae: GaeConfig,
}

/// Accepted rollout and transition counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RolloutCollectorMetrics {
	/// Successfully submitted complete rollouts.
	pub collections: u64,
	/// Vector-environment steps recorded across accepted rollouts.
	pub environment_steps: u64,
	/// Individual lane transitions across accepted rollouts.
	pub transitions: u64,
}

/// Environment-neutral categorical collector over borrowed owners.
///
/// Collection records one complete transaction and returns its exact Event
/// without waiting. It owns no environment, model, rollout storage, Engine, or
/// queue.
#[must_use]
pub struct RolloutCollector<'environment, 'model> {
	environment: &'environment mut dyn Environment,
	model: &'model dyn ActorCritic,
	config: RolloutCollectorConfig,
	metrics: RolloutCollectorMetrics,
	action_index: u64,
}

impl<'environment, 'model> RolloutCollector<'environment, 'model> {
	/// Construct a checked same-device categorical collector.
	///
	/// # Errors
	///
	/// Returns an error for zero horizon/lanes, invalid environment metadata, a
	/// non-discrete action space, or model parameters owned by another Engine.
	pub fn new(
		environment: &'environment mut dyn Environment,
		model: &'model dyn ActorCritic,
		config: RolloutCollectorConfig,
	) -> Result<Self> {
		if config.horizon == 0 || environment.environments() == 0 {
			return Err(Error::invalid_argument(
				"rollout collector requires a nonempty horizon and vector environment",
			));
		}
		if environment.spec().action().kind() != EnvironmentSpaceKind::Discrete {
			return Err(Error::invalid_argument(
				"rollout collector requires a discrete action space",
			));
		}
		for parameter in model.all_parameters()? {
			if !environment.engine().owns_matrix(&parameter.data()) {
				return Err(Error::invalid_argument(
					"rollout collector model must belong to the environment Engine",
				));
			}
		}
		Ok(Self {
			environment,
			model,
			config,
			metrics: RolloutCollectorMetrics::default(),
			action_index: 0,
		})
	}

	/// Record and submit one complete categorical rollout without waiting.
	///
	/// # Errors
	///
	/// Returns an error for rollout mismatch, seed/counter overflow, environment,
	/// model, policy, GAE, capture, or queue submission failure. Recording errors
	/// cancel the complete transaction and rewind rollout/action host state.
	pub fn collect(&mut self, rollout: &mut RolloutBuffer) -> Result<Event> {
		let environments = self.environment.environments();
		let rollout_config = rollout.config();
		if rollout_config.time != self.config.horizon
			|| rollout_config.environments != environments as usize
			|| rollout_config.observation_shape != self.environment.spec().observation().shape()
		{
			return Err(Error::invalid_argument(
				"rollout collector storage does not match environment/horizon schema",
			));
		}
		let observation_elements = self
			.environment
			.spec()
			.observation()
			.elements_per_environment();
		let starting_action_index = self.action_index;
		let recorded = self.record(rollout, environments as usize, observation_elements);
		if let Err(error) = recorded {
			let _ = self.environment.cancel();
			rollout.abort_unsubmitted();
			self.action_index = starting_action_index;
			return Err(error);
		}
		let event = self.environment.submit()?;
		self.metrics.collections = self
			.metrics
			.collections
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("rollout collection count exhausted"))?;
		self.metrics.environment_steps = self
			.metrics
			.environment_steps
			.checked_add(self.config.horizon as u64)
			.ok_or_else(|| Error::resource_exhausted("environment step count exhausted"))?;
		let transitions = (self.config.horizon as u64)
			.checked_mul(u64::from(environments))
			.ok_or_else(|| Error::resource_exhausted("rollout transition count overflows"))?;
		self.metrics.transitions = self
			.metrics
			.transitions
			.checked_add(transitions)
			.ok_or_else(|| Error::resource_exhausted("rollout transition count exhausted"))?;
		Ok(event)
	}

	fn record(
		&mut self,
		rollout: &mut RolloutBuffer,
		environments: usize,
		observation_elements: usize,
	) -> Result<()> {
		if self.metrics.collections == 0 {
			self.environment.reset(self.config.seed)?;
		} else {
			self.environment.begin()?;
		}
		rollout.reset()?;
		for _ in 0..self.config.horizon {
			let observation = self.environment.observation().clone();
			let flat = matrix::reshape(&observation, [environments, observation_elements])?;
			let network = self.model.evaluate(&flat)?;
			self.action_index = self
				.action_index
				.checked_add(1)
				.ok_or_else(|| Error::resource_exhausted("collector action index exhausted"))?;
			let seed = self
				.config
				.seed
				.checked_add(self.action_index)
				.ok_or_else(|| Error::resource_exhausted("collector action seed exhausted"))?;
			let selected = policy::sample_categorical(&network.logits, &network.value, seed)?;
			let transition = self.environment.step(&selected.action)?;
			let next_flat = matrix::reshape(
				transition.next_observation(),
				[environments, observation_elements],
			)?;
			let next = self.model.evaluate(&next_flat)?;
			rollout.append(&RolloutTransition::new(
				transition.observation().clone(),
				selected.action,
				transition.reward().clone(),
				selected.value,
				next.value,
				selected.log_probability,
				transition.terminated().clone(),
				transition.truncated().clone(),
			))?;
			self.environment.reset_completed()?;
		}
		rollout.finalize(self.config.gae)
	}

	/// Return accepted collection counters.
	pub const fn metrics(&self) -> RolloutCollectorMetrics {
		self.metrics
	}
}
