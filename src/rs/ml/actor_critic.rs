//! Environment-neutral actor/critic module contracts.

use std::rc::Rc;

use crate::{DType, Engine, Error, Matrix, Result, matrix as core_matrix};

use super::{Module, ModuleRegistry, matrix, nn::Linear};

/// Policy logits and scalar critic value produced from one observation batch.
pub struct ActorCriticOutput {
	/// FP32 categorical policy logits shaped `[batch, actions]`.
	pub logits: Matrix,
	/// FP32 critic estimates shaped `[batch]`.
	pub value: Matrix,
}

/// Environment-neutral discrete actor/critic behavior.
///
/// PPO depends on this contract rather than a concrete network architecture.
pub trait ActorCritic: Module {
	/// Evaluate policy logits and critic values without submitting or waiting.
	///
	/// # Errors
	///
	/// Returns an error when the observation contract or a child operation fails.
	fn evaluate(&self, observation: &Matrix) -> Result<ActorCriticOutput>;
}

/// Dimensions and deterministic initialization seed for the default MLP.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CategoricalActorCriticConfig {
	/// Number of scalar features in each observation.
	pub observation_size: usize,
	/// Number of discrete actions; must be at least two.
	pub action_count: usize,
	/// Width of both hidden layers in each independent tower.
	pub hidden_size: usize,
	/// Seed for the first of six independent Xavier-uniform projections.
	pub seed: u64,
}

/// Default two-tower categorical actor/critic MLP.
///
/// Policy and value paths are independent
/// `Linear -> ReLU -> Linear -> ReLU -> Linear` towers, matching OA C++.
pub struct CategoricalActorCritic {
	config: CategoricalActorCriticConfig,
	policy_0: Rc<Linear>,
	policy_1: Rc<Linear>,
	policy: Rc<Linear>,
	value_0: Rc<Linear>,
	value_1: Rc<Linear>,
	value: Rc<Linear>,
	registry: ModuleRegistry,
}

impl CategoricalActorCritic {
	/// Construct the checked default categorical actor/critic.
	///
	/// # Errors
	///
	/// Returns an error for zero observation/hidden width, fewer than two
	/// actions, initializer overflow, allocation failure, or invalid child
	/// registration.
	pub fn new(engine: &Engine, config: CategoricalActorCriticConfig) -> Result<Self> {
		if config.observation_size == 0 || config.action_count < 2 || config.hidden_size == 0 {
			return Err(Error::invalid_argument(
				"categorical actor-critic requires positive observation/hidden sizes and at least two actions",
			));
		}
		let policy_0 = Rc::new(Linear::with_seed(
			engine,
			config.observation_size,
			config.hidden_size,
			config.seed,
		)?);
		let policy_1 = Rc::new(Linear::with_seed(
			engine,
			config.hidden_size,
			config.hidden_size,
			config.seed.wrapping_add(1),
		)?);
		let policy = Rc::new(Linear::with_seed(
			engine,
			config.hidden_size,
			config.action_count,
			config.seed.wrapping_add(2),
		)?);
		let value_0 = Rc::new(Linear::with_seed(
			engine,
			config.observation_size,
			config.hidden_size,
			config.seed.wrapping_add(3),
		)?);
		let value_1 = Rc::new(Linear::with_seed(
			engine,
			config.hidden_size,
			config.hidden_size,
			config.seed.wrapping_add(4),
		)?);
		let value = Rc::new(Linear::with_seed(
			engine,
			config.hidden_size,
			1,
			config.seed.wrapping_add(5),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("policy_0", policy_0.clone())?;
		registry.register_module("policy_1", policy_1.clone())?;
		registry.register_module("policy", policy.clone())?;
		registry.register_module("value_0", value_0.clone())?;
		registry.register_module("value_1", value_1.clone())?;
		registry.register_module("value", value.clone())?;
		Ok(Self {
			config,
			policy_0,
			policy_1,
			policy,
			value_0,
			value_1,
			value,
			registry,
		})
	}

	/// Evaluate the independent policy and value towers.
	///
	/// # Errors
	///
	/// Returns an error unless observations are nonempty FP32 `[batch, O]`,
	/// or when a child Matrix operation fails.
	pub fn evaluate(&self, observation: &Matrix) -> Result<ActorCriticOutput> {
		let [batch, features] = observation.shape() else {
			return Err(Error::invalid_argument(
				"categorical actor-critic observations must have rank two",
			));
		};
		if *batch == 0
			|| *features != self.config.observation_size
			|| observation.dtype() != DType::F32
		{
			return Err(Error::invalid_argument(format!(
				"categorical actor-critic requires nonempty FP32 observations [batch, {}]",
				self.config.observation_size
			)));
		}
		let policy_hidden = matrix::relu(&self.policy_0.forward(observation)?)?;
		let policy_hidden = matrix::relu(&self.policy_1.forward(&policy_hidden)?)?;
		let value_hidden = matrix::relu(&self.value_0.forward(observation)?)?;
		let value_hidden = matrix::relu(&self.value_1.forward(&value_hidden)?)?;
		let value = self.value.forward(&value_hidden)?;
		Ok(ActorCriticOutput {
			logits: self.policy.forward(&policy_hidden)?,
			value: core_matrix::reshape(&value, [*batch])?,
		})
	}

	/// Return the immutable architecture and seed.
	pub const fn config(&self) -> CategoricalActorCriticConfig {
		self.config
	}
}

impl Module for CategoricalActorCritic {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Ok(self.evaluate(input)?.logits)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

impl ActorCritic for CategoricalActorCritic {
	fn evaluate(&self, observation: &Matrix) -> Result<ActorCriticOutput> {
		CategoricalActorCritic::evaluate(self, observation)
	}
}
