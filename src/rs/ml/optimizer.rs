use std::sync::atomic::{AtomicU64, Ordering};

use crate::{DType, Engine, Error, Matrix, Result};

use super::{
	Parameter,
	lowering::optim::{self as dispatch, AdamScalars, AdamWScalars, MuonScalars, SgdScalars},
};

/// Common behavior exposed by an OA optimizer to training policy.
///
/// This contract deliberately excludes captured-program and persistence
/// internals. Those capabilities depend on optimizer-specific state and are
/// admitted separately only when their complete implementation exists.
pub trait Optimizer {
	/// Discard every currently accumulated parameter gradient.
	fn zero_grad(&self);

	/// Record one logical parameter update.
	///
	/// # Errors
	///
	/// Returns an optimizer, validation, allocation, or runtime recording error.
	fn step(&mut self) -> Result<()>;

	/// Return the learning rate used by the next optimizer step.
	fn learning_rate(&self) -> f32;

	/// Set the learning rate used by subsequent optimizer steps.
	///
	/// # Errors
	///
	/// Returns an error when the value is outside the optimizer's contract or
	/// optimizer state cannot be updated safely.
	fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()>;

	/// Return the number of completed logical optimizer steps.
	fn step_count(&self) -> u64;

	/// Return whether this optimizer can participate in work owned by `engine`.
	fn belongs_to(&self, engine: &Engine) -> bool;
}

pub struct OptimizerCheckpoint {
	pub(crate) kind: &'static str,
	pub(crate) step: u64,
	pub(crate) learning_rate: f32,
	pub(crate) beta1: f32,
	pub(crate) beta2: f32,
	pub(crate) epsilon: f32,
	pub(crate) weight_decay: f32,
	pub(crate) parameters: Vec<Parameter>,
	pub(crate) first_state: Vec<Matrix>,
	pub(crate) second_state: Vec<Matrix>,
}

pub struct OptimizerRestore {
	pub(crate) kind: String,
	pub(crate) step: u64,
	pub(crate) learning_rate: f32,
	pub(crate) beta1: f32,
	pub(crate) beta2: f32,
	pub(crate) epsilon: f32,
	pub(crate) weight_decay: f32,
	pub(crate) first_state: Vec<Matrix>,
	pub(crate) second_state: Vec<Matrix>,
}

pub(super) mod checkpoint_capability {
	use super::{OptimizerCheckpoint, OptimizerRestore};
	use crate::Result;

	pub trait Persistence {
		fn checkpoint_state(&self) -> Result<OptimizerCheckpoint>;
		fn validate_checkpoint_state(&self, state: &OptimizerRestore) -> Result<()>;
		fn restore_checkpoint_state(&mut self, state: OptimizerRestore) -> Result<()>;
	}
}

/// Optimizer with a complete native `.oam` save-and-restore contract.
///
/// This capability is sealed because persistence must remain synchronized with
/// OA's versioned model-file codec. Custom optimizers can implement
/// [`Optimizer`] for eager training without claiming wire compatibility.
#[allow(private_bounds)]
pub trait CheckpointOptimizer: Optimizer + checkpoint_capability::Persistence {}

impl<T> CheckpointOptimizer for T where T: Optimizer + checkpoint_capability::Persistence {}

/// Optimizer for externally authored parameter updates.
///
/// `NoOpOptimizer` lets callers use [`super::ItTraining`] for cadence,
/// metrics, and callbacks while the step body owns all parameter mutation.
#[derive(Clone, Copy, Debug)]
pub struct NoOpOptimizer {
	learning_rate: f32,
}

impl NoOpOptimizer {
	/// Construct a no-op optimizer with the requested policy-visible rate.
	///
	/// # Errors
	///
	/// Returns an error when `learning_rate` is not finite and non-negative.
	pub fn new(learning_rate: f32) -> Result<Self> {
		if !learning_rate.is_finite() || learning_rate < 0.0 {
			return Err(Error::invalid_argument(
				"no-op optimizer learning rate must be finite and non-negative",
			));
		}
		Ok(Self { learning_rate })
	}
}

impl Default for NoOpOptimizer {
	fn default() -> Self {
		Self { learning_rate: 0.0 }
	}
}

impl Optimizer for NoOpOptimizer {
	fn zero_grad(&self) {}

	fn step(&mut self) -> Result<()> {
		Ok(())
	}

	fn learning_rate(&self) -> f32 {
		self.learning_rate
	}

	fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
		if !learning_rate.is_finite() || learning_rate < 0.0 {
			return Err(Error::invalid_argument(
				"no-op optimizer learning rate must be finite and non-negative",
			));
		}
		self.learning_rate = learning_rate;
		Ok(())
	}

	fn step_count(&self) -> u64 {
		0
	}

	fn belongs_to(&self, _engine: &Engine) -> bool {
		true
	}
}

struct MomentParameterState {
	parameter: Parameter,
	first_moment: Matrix,
	second_moment: Matrix,
}

struct SgdParameterState {
	parameter: Parameter,
	momentum: Option<Matrix>,
}

/// Stochastic gradient descent over stable [`Parameter`] handles.
pub struct Sgd {
	parameters: Vec<SgdParameterState>,
	learning_rate: f32,
	momentum: f32,
	weight_decay: f32,
	step: u64,
}

impl Sgd {
	/// Bind SGD to a nonempty FP32 parameter set.
	///
	/// A positive momentum coefficient allocates one zero-initialized momentum
	/// tensor per parameter. Weight decay follows the donor kernels exactly.
	///
	/// # Errors
	///
	/// Returns an error for invalid hyperparameters, duplicate parameters,
	/// mixed-engine ownership, or momentum allocation/upload failure.
	pub fn new(
		parameters: impl IntoIterator<Item = Parameter>,
		learning_rate: f32,
		momentum: f32,
		weight_decay: f32,
	) -> Result<Self> {
		if !learning_rate.is_finite() || learning_rate < 0.0 {
			return Err(Error::invalid_argument(
				"SGD learning rate must be finite and non-negative",
			));
		}
		if !momentum.is_finite() || momentum < 0.0 {
			return Err(Error::invalid_argument(
				"SGD momentum must be finite and non-negative",
			));
		}
		if !weight_decay.is_finite() || weight_decay < 0.0 {
			return Err(Error::invalid_argument(
				"SGD weight decay must be finite and non-negative",
			));
		}
		let parameters = parameters.into_iter().collect::<Vec<_>>();
		if parameters.is_empty() {
			return Err(Error::invalid_argument(
				"SGD requires at least one parameter",
			));
		}
		for (index, parameter) in parameters.iter().enumerate() {
			if parameters[..index]
				.iter()
				.any(|existing| existing.same_as(parameter))
			{
				return Err(Error::invalid_argument(
					"SGD parameter handle appears more than once",
				));
			}
		}
		let engine = parameters[0].data().engine_handle().clone();
		if parameters
			.iter()
			.any(|parameter| !engine.same_as(parameter.data().engine_handle()))
		{
			return Err(Error::invalid_argument(
				"SGD parameters must belong to one engine",
			));
		}
		let mut states = Vec::with_capacity(parameters.len());
		for parameter in parameters {
			let data = parameter.data();
			let momentum = (momentum > 0.0)
				.then(|| {
					Matrix::allocate(
						data.engine_handle(),
						data.shape().to_vec(),
						data.num_elements(),
						DType::F32,
					)
				})
				.transpose()?;
			states.push(SgdParameterState {
				parameter,
				momentum,
			});
		}
		Ok(Self {
			parameters: states,
			learning_rate,
			momentum,
			weight_decay,
			step: 0,
		})
	}

	/// Discard every currently accumulated parameter gradient.
	pub fn zero_grad(&self) {
		for state in &self.parameters {
			state.parameter.clear_gradient();
		}
	}

	/// Record one SGD update for every parameter carrying a gradient.
	///
	/// # Errors
	///
	/// Returns an error when the step counter is exhausted, a parameter contract
	/// changed, or runtime recording fails.
	pub fn step(&mut self) -> Result<()> {
		let next_step = self
			.step
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("SGD step counter exhausted"))?;
		let scalars = SgdScalars {
			learning_rate: self.learning_rate,
			momentum: self.momentum,
			weight_decay: self.weight_decay,
		};
		let updates = self
			.parameters
			.iter()
			.filter_map(|state| {
				state
					.parameter
					.gradient()
					.map(|gradient| (state, state.parameter.data(), gradient))
			})
			.collect::<Vec<_>>();
		for (state, _, _) in &updates {
			state.parameter.validate_can_update()?;
		}
		for (state, parameter, gradient) in &updates {
			match &state.momentum {
				Some(momentum) => dispatch::sgd_momentum(parameter, gradient, momentum, scalars)?,
				None => dispatch::sgd(parameter, gradient, scalars)?,
			}
		}
		for (state, _, _) in &updates {
			state.parameter.mark_updated()?;
		}
		self.step = next_step;
		Ok(())
	}

	/// Return the learning rate used by the next step.
	pub const fn learning_rate(&self) -> f32 {
		self.learning_rate
	}

	/// Set the learning rate used by subsequent steps.
	///
	/// # Errors
	///
	/// Returns an error when the value is not finite and non-negative.
	pub fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
		if !learning_rate.is_finite() || learning_rate < 0.0 {
			return Err(Error::invalid_argument(
				"SGD learning rate must be finite and non-negative",
			));
		}
		self.learning_rate = learning_rate;
		Ok(())
	}

	/// Return the number of completed logical steps.
	pub const fn step_count(&self) -> u64 {
		self.step
	}

	fn checkpoint_state(&self) -> Result<OptimizerCheckpoint> {
		if self.momentum > 0.0 && self.step > 0 {
			return Err(Error::failed_precondition(
				"native .oam cannot represent live SGD momentum state",
			));
		}
		Ok(OptimizerCheckpoint {
			kind: "SGD",
			step: self.step,
			learning_rate: self.learning_rate,
			beta1: self.momentum,
			beta2: 0.0,
			epsilon: 0.0,
			weight_decay: self.weight_decay,
			parameters: self
				.parameters
				.iter()
				.map(|state| state.parameter.clone())
				.collect(),
			first_state: Vec::new(),
			second_state: Vec::new(),
		})
	}

	fn validate_checkpoint_state(&self, state: &OptimizerRestore) -> Result<()> {
		if state.kind != "SGD"
			|| !state.first_state.is_empty()
			|| !state.second_state.is_empty()
			|| !state.learning_rate.is_finite()
			|| state.learning_rate < 0.0
			|| !state.beta1.is_finite()
			|| state.beta1 < 0.0
			|| state.beta1 != self.momentum
			|| (state.beta1 > 0.0 && state.step > 0)
			|| !state.weight_decay.is_finite()
			|| state.weight_decay < 0.0
		{
			return Err(Error::invalid_argument("invalid SGD checkpoint state"));
		}
		Ok(())
	}

	fn restore_checkpoint_state(&mut self, state: OptimizerRestore) -> Result<()> {
		self.validate_checkpoint_state(&state)?;
		self.step = state.step;
		self.learning_rate = state.learning_rate;
		self.weight_decay = state.weight_decay;
		Ok(())
	}

	fn belongs_to_engine(&self, engine: &Engine) -> bool {
		engine.same_as_handle(self.parameters[0].parameter.data().engine_handle())
	}
}

impl Optimizer for Sgd {
	fn zero_grad(&self) {
		Self::zero_grad(self);
	}

	fn step(&mut self) -> Result<()> {
		Self::step(self)
	}

	fn learning_rate(&self) -> f32 {
		Self::learning_rate(self)
	}

	fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
		Self::set_learning_rate(self, learning_rate)
	}

	fn step_count(&self) -> u64 {
		Self::step_count(self)
	}

	fn belongs_to(&self, engine: &Engine) -> bool {
		self.belongs_to_engine(engine)
	}
}

impl checkpoint_capability::Persistence for Sgd {
	fn checkpoint_state(&self) -> Result<OptimizerCheckpoint> {
		Self::checkpoint_state(self)
	}

	fn validate_checkpoint_state(&self, state: &OptimizerRestore) -> Result<()> {
		Self::validate_checkpoint_state(self, state)
	}

	fn restore_checkpoint_state(&mut self, state: OptimizerRestore) -> Result<()> {
		Self::restore_checkpoint_state(self, state)
	}
}

/// Adam optimizer over stable [`Parameter`] handles.
pub struct Adam {
	parameters: Vec<MomentParameterState>,
	learning_rate: f32,
	beta1: f32,
	beta2: f32,
	epsilon: f32,
	step: u32,
}

impl Adam {
	/// Bind Adam to a nonempty parameter set using standard OA defaults.
	///
	/// # Errors
	///
	/// Returns an error for invalid ownership, duplicate parameters, or moment
	/// allocation failure.
	pub fn new(
		parameters: impl IntoIterator<Item = Parameter>,
		learning_rate: f32,
	) -> Result<Self> {
		Self::with_hyperparameters(parameters, learning_rate, 0.9, 0.999, 1.0e-8)
	}

	/// Bind Adam with explicit scalar hyperparameters.
	///
	/// # Errors
	///
	/// Returns an error for non-finite or out-of-range scalars, an empty or
	/// duplicate parameter set, mixed-engine ownership, or moment allocation.
	pub fn with_hyperparameters(
		parameters: impl IntoIterator<Item = Parameter>,
		learning_rate: f32,
		beta1: f32,
		beta2: f32,
		epsilon: f32,
	) -> Result<Self> {
		if !learning_rate.is_finite() || learning_rate < 0.0 {
			return Err(Error::invalid_argument(
				"Adam learning rate must be finite and non-negative",
			));
		}
		if !beta1.is_finite() || !(0.0..1.0).contains(&beta1) {
			return Err(Error::invalid_argument("Adam beta1 must be in [0, 1)"));
		}
		if !beta2.is_finite() || !(0.0..1.0).contains(&beta2) {
			return Err(Error::invalid_argument("Adam beta2 must be in [0, 1)"));
		}
		if !epsilon.is_finite() || epsilon <= 0.0 {
			return Err(Error::invalid_argument(
				"Adam epsilon must be finite and positive",
			));
		}
		let parameters = parameters.into_iter().collect::<Vec<_>>();
		if parameters.is_empty() {
			return Err(Error::invalid_argument(
				"Adam requires at least one parameter",
			));
		}
		for (index, parameter) in parameters.iter().enumerate() {
			if parameters[..index]
				.iter()
				.any(|existing| existing.same_as(parameter))
			{
				return Err(Error::invalid_argument(
					"Adam parameter handle appears more than once",
				));
			}
		}
		let engine = parameters[0].data().engine_handle().clone();
		if parameters
			.iter()
			.any(|parameter| !engine.same_as(parameter.data().engine_handle()))
		{
			return Err(Error::invalid_argument(
				"Adam parameters must belong to one engine",
			));
		}
		let mut states = Vec::with_capacity(parameters.len());
		for parameter in parameters {
			let data = parameter.data();
			let first_moment = Matrix::allocate(
				data.engine_handle(),
				data.shape().to_vec(),
				data.num_elements(),
				DType::F32,
			)?;
			let second_moment = Matrix::allocate(
				data.engine_handle(),
				data.shape().to_vec(),
				data.num_elements(),
				DType::F32,
			)?;
			states.push(MomentParameterState {
				parameter,
				first_moment,
				second_moment,
			});
		}
		Ok(Self {
			parameters: states,
			learning_rate,
			beta1,
			beta2,
			epsilon,
			step: 0,
		})
	}

	/// Discard every currently accumulated parameter gradient.
	pub fn zero_grad(&self) {
		for state in &self.parameters {
			state.parameter.clear_gradient();
		}
	}

	/// Record one Adam update for every parameter carrying a gradient.
	///
	/// # Errors
	///
	/// Returns an error when the step counter is exhausted, a parameter contract
	/// changed, or runtime recording fails.
	pub fn step(&mut self) -> Result<()> {
		let step = self
			.step
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("Adam step counter exhausted"))?;
		let scalars = AdamScalars {
			step,
			learning_rate: self.learning_rate,
			beta1: self.beta1,
			beta2: self.beta2,
			epsilon: self.epsilon,
		};
		let updates = self
			.parameters
			.iter()
			.filter_map(|state| {
				state
					.parameter
					.gradient()
					.map(|gradient| (state, state.parameter.data(), gradient))
			})
			.collect::<Vec<_>>();
		for (state, _, _) in &updates {
			state.parameter.validate_can_update()?;
		}
		for (state, parameter, gradient) in &updates {
			dispatch::adam(
				parameter,
				gradient,
				&state.first_moment,
				&state.second_moment,
				scalars,
			)?;
		}
		for (state, _, _) in &updates {
			state.parameter.mark_updated()?;
		}
		self.step = step;
		Ok(())
	}

	/// Return the learning rate used by the next step.
	pub const fn learning_rate(&self) -> f32 {
		self.learning_rate
	}

	/// Set the learning rate used by subsequent steps.
	///
	/// # Errors
	///
	/// Returns an error when the value is not finite and non-negative.
	pub fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
		if !learning_rate.is_finite() || learning_rate < 0.0 {
			return Err(Error::invalid_argument(
				"Adam learning rate must be finite and non-negative",
			));
		}
		self.learning_rate = learning_rate;
		Ok(())
	}

	/// Return the number of completed logical steps.
	pub const fn step_count(&self) -> u32 {
		self.step
	}

	fn checkpoint_state(&self) -> Result<OptimizerCheckpoint> {
		Ok(OptimizerCheckpoint {
			kind: "Adam",
			step: u64::from(self.step),
			learning_rate: self.learning_rate,
			beta1: self.beta1,
			beta2: self.beta2,
			epsilon: self.epsilon,
			weight_decay: 0.0,
			parameters: self
				.parameters
				.iter()
				.map(|state| state.parameter.clone())
				.collect(),
			first_state: self
				.parameters
				.iter()
				.map(|state| state.first_moment.clone())
				.collect(),
			second_state: self
				.parameters
				.iter()
				.map(|state| state.second_moment.clone())
				.collect(),
		})
	}

	fn validate_checkpoint_state(&self, state: &OptimizerRestore) -> Result<()> {
		if state.kind != "Adam"
			|| state.first_state.len() != self.parameters.len()
			|| state.second_state.len() != self.parameters.len()
			|| state.step > u64::from(u32::MAX)
			|| !state.learning_rate.is_finite()
			|| state.learning_rate < 0.0
			|| !state.beta1.is_finite()
			|| !(0.0..1.0).contains(&state.beta1)
			|| !state.beta2.is_finite()
			|| !(0.0..1.0).contains(&state.beta2)
			|| !state.epsilon.is_finite()
			|| state.epsilon <= 0.0
		{
			return Err(Error::invalid_argument("invalid Adam checkpoint state"));
		}
		for ((parameter_state, first), second) in self
			.parameters
			.iter()
			.zip(&state.first_state)
			.zip(&state.second_state)
		{
			let data = parameter_state.parameter.data();
			for moment in [first, second] {
				if moment.shape() != data.shape()
					|| moment.dtype() != DType::F32
					|| !moment.engine_handle().same_as(data.engine_handle())
				{
					return Err(Error::invalid_argument(
						"Adam checkpoint moment contract mismatch",
					));
				}
			}
		}
		Ok(())
	}

	fn restore_checkpoint_state(&mut self, state: OptimizerRestore) -> Result<()> {
		self.validate_checkpoint_state(&state)?;
		for ((parameter_state, first), second) in self
			.parameters
			.iter_mut()
			.zip(state.first_state)
			.zip(state.second_state)
		{
			parameter_state.first_moment = first;
			parameter_state.second_moment = second;
		}
		self.step = state.step as u32;
		self.learning_rate = state.learning_rate;
		self.beta1 = state.beta1;
		self.beta2 = state.beta2;
		self.epsilon = state.epsilon;
		Ok(())
	}

	fn belongs_to_engine(&self, engine: &Engine) -> bool {
		engine.same_as_handle(self.parameters[0].parameter.data().engine_handle())
	}
}

impl Optimizer for Adam {
	fn zero_grad(&self) {
		Self::zero_grad(self);
	}

	fn step(&mut self) -> Result<()> {
		Self::step(self)
	}

	fn learning_rate(&self) -> f32 {
		Self::learning_rate(self)
	}

	fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
		Self::set_learning_rate(self, learning_rate)
	}

	fn step_count(&self) -> u64 {
		u64::from(Self::step_count(self))
	}

	fn belongs_to(&self, engine: &Engine) -> bool {
		self.belongs_to_engine(engine)
	}
}

impl checkpoint_capability::Persistence for Adam {
	fn checkpoint_state(&self) -> Result<OptimizerCheckpoint> {
		Self::checkpoint_state(self)
	}

	fn validate_checkpoint_state(&self, state: &OptimizerRestore) -> Result<()> {
		Self::validate_checkpoint_state(self, state)
	}

	fn restore_checkpoint_state(&mut self, state: OptimizerRestore) -> Result<()> {
		Self::restore_checkpoint_state(self, state)
	}
}

struct MuonParameterState {
	parameter: Parameter,
	momentum: Matrix,
}

/// Muon optimizer with Nesterov momentum and rank-two NS5 orthogonalization.
///
/// Rank-two parameters use the donor Muon matrix pipeline. Other ranks use its
/// fused momentum update; OA never silently delegates them to AdamW.
pub struct Muon {
	parameters: Vec<MuonParameterState>,
	learning_rate: f32,
	beta: f32,
	weight_decay: f32,
	epsilon: f32,
	ns5_iterations: u32,
	step: u64,
}

impl Muon {
	/// Bind Muon to a nonempty FP32 parameter set using OA defaults.
	///
	/// # Errors
	///
	/// Returns an error for invalid ownership, duplicate parameters, or momentum
	/// allocation failure.
	pub fn new(
		parameters: impl IntoIterator<Item = Parameter>,
		learning_rate: f32,
	) -> Result<Self> {
		Self::with_hyperparameters(parameters, learning_rate, 0.95, 0.1, 1.0e-7, 5)
	}

	/// Bind Muon with explicit donor scalar policy.
	///
	/// Zero NS5 iterations selects the fused vector update for every parameter.
	///
	/// # Errors
	///
	/// Returns an error for invalid scalars, an empty or duplicate parameter set,
	/// non-F32 parameters, mixed-engine ownership, or momentum allocation failure.
	pub fn with_hyperparameters(
		parameters: impl IntoIterator<Item = Parameter>,
		learning_rate: f32,
		beta: f32,
		weight_decay: f32,
		epsilon: f32,
		ns5_iterations: u32,
	) -> Result<Self> {
		if !learning_rate.is_finite() || learning_rate < 0.0 {
			return Err(Error::invalid_argument(
				"Muon learning rate must be finite and non-negative",
			));
		}
		if !beta.is_finite() || !(0.0..1.0).contains(&beta) {
			return Err(Error::invalid_argument("Muon beta must be in [0, 1)"));
		}
		if !weight_decay.is_finite() || weight_decay < 0.0 {
			return Err(Error::invalid_argument(
				"Muon weight decay must be finite and non-negative",
			));
		}
		if !epsilon.is_finite() || epsilon <= 0.0 {
			return Err(Error::invalid_argument(
				"Muon epsilon must be finite and positive",
			));
		}
		let parameters = parameters.into_iter().collect::<Vec<_>>();
		if parameters.is_empty() {
			return Err(Error::invalid_argument(
				"Muon requires at least one parameter",
			));
		}
		for (index, parameter) in parameters.iter().enumerate() {
			if parameter.data().dtype() != DType::F32 {
				return Err(Error::invalid_argument("Muon requires FP32 parameters"));
			}
			if parameters[..index]
				.iter()
				.any(|existing| existing.same_as(parameter))
			{
				return Err(Error::invalid_argument(
					"Muon parameter handle appears more than once",
				));
			}
		}
		let engine = parameters[0].data().engine_handle().clone();
		if parameters
			.iter()
			.any(|parameter| !engine.same_as(parameter.data().engine_handle()))
		{
			return Err(Error::invalid_argument(
				"Muon parameters must belong to one engine",
			));
		}
		let mut states = Vec::with_capacity(parameters.len());
		for parameter in parameters {
			let data = parameter.data();
			let momentum = Matrix::allocate(
				data.engine_handle(),
				data.shape().to_vec(),
				data.num_elements(),
				DType::F32,
			)?;
			states.push(MuonParameterState {
				parameter,
				momentum,
			});
		}
		Ok(Self {
			parameters: states,
			learning_rate,
			beta,
			weight_decay,
			epsilon,
			ns5_iterations,
			step: 0,
		})
	}

	/// Discard all accumulated gradients.
	pub fn zero_grad(&self) {
		for state in &self.parameters {
			state.parameter.clear_gradient();
		}
	}

	/// Record one Muon update for each parameter carrying a gradient.
	///
	/// # Errors
	///
	/// Returns an error when the step counter is exhausted, a parameter contract
	/// changed, or runtime recording/allocation fails.
	pub fn step(&mut self) -> Result<()> {
		let next_step = self
			.step
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("Muon step counter exhausted"))?;
		let scalars = MuonScalars {
			learning_rate: self.learning_rate,
			beta: self.beta,
			weight_decay: self.weight_decay,
			epsilon: self.epsilon,
			ns5_iterations: self.ns5_iterations,
		};
		let updates = self
			.parameters
			.iter()
			.filter_map(|state| {
				state
					.parameter
					.gradient()
					.map(|gradient| (state, state.parameter.data(), gradient))
			})
			.collect::<Vec<_>>();
		for (state, _, _) in &updates {
			state.parameter.validate_can_update()?;
		}
		for (state, parameter, gradient) in &updates {
			dispatch::muon(parameter, gradient, &state.momentum, scalars)?;
		}
		for (state, _, _) in &updates {
			state.parameter.mark_updated()?;
		}
		self.step = next_step;
		Ok(())
	}

	/// Return the learning rate used by the next step.
	pub const fn learning_rate(&self) -> f32 {
		self.learning_rate
	}

	/// Set the learning rate used by subsequent steps.
	///
	/// # Errors
	///
	/// Returns an error unless the value is finite and non-negative.
	pub fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
		if !learning_rate.is_finite() || learning_rate < 0.0 {
			return Err(Error::invalid_argument(
				"Muon learning rate must be finite and non-negative",
			));
		}
		self.learning_rate = learning_rate;
		Ok(())
	}

	/// Return the number of completed logical steps.
	pub const fn step_count(&self) -> u64 {
		self.step
	}

	fn checkpoint_state(&self) -> Result<OptimizerCheckpoint> {
		Ok(OptimizerCheckpoint {
			kind: "Muon",
			step: self.step,
			learning_rate: self.learning_rate,
			beta1: self.beta,
			beta2: 0.0,
			epsilon: self.epsilon,
			weight_decay: self.weight_decay,
			parameters: self
				.parameters
				.iter()
				.map(|state| state.parameter.clone())
				.collect(),
			first_state: self
				.parameters
				.iter()
				.map(|state| state.momentum.clone())
				.collect(),
			second_state: Vec::new(),
		})
	}

	fn validate_checkpoint_state(&self, state: &OptimizerRestore) -> Result<()> {
		if state.kind != "Muon"
			|| state.first_state.len() != self.parameters.len()
			|| !state.second_state.is_empty()
			|| !state.learning_rate.is_finite()
			|| state.learning_rate < 0.0
			|| !state.beta1.is_finite()
			|| !(0.0..1.0).contains(&state.beta1)
			|| state.beta2 != 0.0
			|| !state.epsilon.is_finite()
			|| state.epsilon <= 0.0
			|| !state.weight_decay.is_finite()
			|| state.weight_decay < 0.0
		{
			return Err(Error::invalid_argument("invalid Muon checkpoint state"));
		}
		for (parameter_state, momentum) in self.parameters.iter().zip(&state.first_state) {
			let data = parameter_state.parameter.data();
			if momentum.shape() != data.shape()
				|| momentum.dtype() != DType::F32
				|| !momentum.engine_handle().same_as(data.engine_handle())
			{
				return Err(Error::invalid_argument(
					"Muon checkpoint momentum contract mismatch",
				));
			}
		}
		Ok(())
	}

	fn restore_checkpoint_state(&mut self, state: OptimizerRestore) -> Result<()> {
		self.validate_checkpoint_state(&state)?;
		for (parameter_state, momentum) in self.parameters.iter_mut().zip(state.first_state) {
			parameter_state.momentum = momentum;
		}
		self.step = state.step;
		self.learning_rate = state.learning_rate;
		self.beta = state.beta1;
		self.epsilon = state.epsilon;
		self.weight_decay = state.weight_decay;
		Ok(())
	}

	fn belongs_to_engine(&self, engine: &Engine) -> bool {
		engine.same_as_handle(self.parameters[0].parameter.data().engine_handle())
	}
}

impl Optimizer for Muon {
	fn zero_grad(&self) {
		Self::zero_grad(self);
	}

	fn step(&mut self) -> Result<()> {
		Self::step(self)
	}

	fn learning_rate(&self) -> f32 {
		Self::learning_rate(self)
	}

	fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
		Self::set_learning_rate(self, learning_rate)
	}

	fn step_count(&self) -> u64 {
		Self::step_count(self)
	}

	fn belongs_to(&self, engine: &Engine) -> bool {
		self.belongs_to_engine(engine)
	}
}

impl checkpoint_capability::Persistence for Muon {
	fn checkpoint_state(&self) -> Result<OptimizerCheckpoint> {
		Self::checkpoint_state(self)
	}

	fn validate_checkpoint_state(&self, state: &OptimizerRestore) -> Result<()> {
		Self::validate_checkpoint_state(self, state)
	}

	fn restore_checkpoint_state(&mut self, state: OptimizerRestore) -> Result<()> {
		Self::restore_checkpoint_state(self, state)
	}
}

pub(super) struct AdamWProgramSignature {
	optimizer_id: u64,
	state_id: u64,
	parameter_ids: Vec<u64>,
	gradient_ids: Vec<u64>,
	moment_ids: Vec<(u64, u64)>,
	pub(super) base_step: u32,
}

/// Decoupled-weight-decay Adam optimizer over stable [`Parameter`] handles.
pub struct AdamW {
	id: u64,
	parameters: Vec<MomentParameterState>,
	learning_rate: f32,
	beta1: f32,
	beta2: f32,
	epsilon: f32,
	weight_decay: f32,
	step: u32,
	graph_state: Option<Matrix>,
}

impl AdamW {
	/// Bind AdamW to a nonempty parameter set using standard OA defaults.
	///
	/// # Errors
	///
	/// Returns an error when the parameter set is empty, learning rate is not
	/// finite and non-negative, a parameter handle appears more than once, or moment
	/// allocation fails.
	pub fn new(
		parameters: impl IntoIterator<Item = Parameter>,
		learning_rate: f32,
	) -> Result<Self> {
		if !learning_rate.is_finite() || learning_rate < 0.0 {
			return Err(Error::invalid_argument(
				"AdamW learning rate must be finite and non-negative",
			));
		}
		let parameters = parameters.into_iter().collect::<Vec<_>>();
		if parameters.is_empty() {
			return Err(Error::invalid_argument(
				"AdamW requires at least one parameter",
			));
		}
		for (index, parameter) in parameters.iter().enumerate() {
			if parameters[..index]
				.iter()
				.any(|existing| existing.same_as(parameter))
			{
				return Err(Error::invalid_argument(
					"AdamW parameter handle appears more than once",
				));
			}
		}
		let engine = parameters[0].data().engine_handle().clone();
		if parameters
			.iter()
			.any(|parameter| !engine.same_as(parameter.data().engine_handle()))
		{
			return Err(Error::invalid_argument(
				"AdamW parameters must belong to one engine",
			));
		}
		let mut states = Vec::with_capacity(parameters.len());
		for parameter in parameters {
			let data = parameter.data();
			let first_moment = Matrix::allocate(
				data.engine_handle(),
				data.shape().to_vec(),
				data.num_elements(),
				DType::F32,
			)?;
			let second_moment = Matrix::allocate(
				data.engine_handle(),
				data.shape().to_vec(),
				data.num_elements(),
				DType::F32,
			)?;
			states.push(MomentParameterState {
				parameter,
				first_moment,
				second_moment,
			});
		}
		Ok(Self {
			id: next_optimizer_id()?,
			parameters: states,
			learning_rate,
			beta1: 0.9,
			beta2: 0.999,
			epsilon: 1.0e-8,
			weight_decay: 0.01,
			step: 0,
			graph_state: None,
		})
	}

	/// Discard every currently accumulated parameter gradient.
	pub fn zero_grad(&self) {
		for state in &self.parameters {
			state.parameter.clear_gradient();
		}
	}

	/// Record one AdamW update for every parameter carrying a gradient.
	///
	/// Parameter and moment storage is updated in place while the stable
	/// [`Parameter`] handle advances its mutation version. No submission or waiting
	/// occurs here.
	///
	/// # Errors
	///
	/// Returns an error when the step counter is exhausted, a gradient contract
	/// changed, or allocation/runtime recording fails.
	pub fn step(&mut self) -> Result<()> {
		let step = self
			.step
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("AdamW step counter exhausted"))?;
		let capture_active = self.parameters[0]
			.parameter
			.data()
			.engine_handle()
			.capture_active();
		if capture_active {
			return self.capture_step(step);
		}
		let scalars = AdamWScalars {
			step,
			learning_rate: self.learning_rate,
			beta1: self.beta1,
			beta2: self.beta2,
			epsilon: self.epsilon,
			weight_decay: self.weight_decay,
		};
		let updates = self
			.parameters
			.iter()
			.filter_map(|state| {
				state
					.parameter
					.gradient()
					.map(|gradient| (state, state.parameter.data(), gradient))
			})
			.collect::<Vec<_>>();
		let (batches, remainder) = updates.as_chunks::<4>();
		for batch in batches {
			let entries = std::array::from_fn(|index| dispatch::AdamWBatchEntry {
				parameter: &batch[index].1,
				gradient: &batch[index].2,
				first_moment: &batch[index].0.first_moment,
				second_moment: &batch[index].0.second_moment,
			});
			dispatch::adamw_many4(&entries, scalars)?;
		}
		for (state, parameter, gradient) in remainder {
			dispatch::adamw(
				parameter,
				gradient,
				&state.first_moment,
				&state.second_moment,
				scalars,
			)?;
		}
		for (state, _, _) in &updates {
			state.parameter.mark_updated()?;
		}
		self.step = step;
		Ok(())
	}

	fn capture_step(&mut self, step: u32) -> Result<()> {
		let gradients = self
			.parameters
			.iter()
			.map(|state| {
				state.parameter.gradient().ok_or_else(|| {
					Error::failed_precondition(
						"captured AdamW requires one stable gradient for every parameter",
					)
				})
			})
			.collect::<Result<Vec<_>>>()?;
		let engine = self.parameters[0].parameter.data().engine_handle().clone();
		let state = Matrix::from_slice_handle(
			&engine,
			vec![6],
			&[
				step - 1,
				self.learning_rate.to_bits(),
				self.beta1.to_bits(),
				self.beta2.to_bits(),
				self.epsilon.to_bits(),
				self.weight_decay.to_bits(),
			],
		)?;
		dispatch::adamw_graph_advance(&state)?;
		let parameters = self
			.parameters
			.iter()
			.map(|parameter_state| parameter_state.parameter.data())
			.collect::<Vec<_>>();
		let mut indices = (0..self.parameters.len()).collect::<Vec<_>>().into_iter();
		while indices.len() >= 4 {
			let batch_indices: [usize; 4] =
				std::array::from_fn(|_| indices.next().expect("four indices remain"));
			let entries = std::array::from_fn(|batch| {
				let index = batch_indices[batch];
				dispatch::AdamWBatchEntry {
					parameter: &parameters[index],
					gradient: &gradients[index],
					first_moment: &self.parameters[index].first_moment,
					second_moment: &self.parameters[index].second_moment,
				}
			});
			dispatch::adamw_many4_graph(&entries, &state)?;
		}
		for index in indices {
			let parameter_state = &self.parameters[index];
			dispatch::adamw_graph(
				&parameters[index],
				&gradients[index],
				&parameter_state.first_moment,
				&parameter_state.second_moment,
				&state,
			)?;
		}
		self.graph_state = Some(state);
		Ok(())
	}

	/// Return the learning rate used by the next optimizer step.
	pub const fn learning_rate(&self) -> f32 {
		self.learning_rate
	}

	/// Set the learning rate used by subsequent eager or captured steps.
	///
	/// A captured program reads optimizer scalars from graph-resident state. This
	/// method updates that state only after its prior GPU use is complete, so a
	/// scheduler changes the existing program rather than silently changing only
	/// the host-side optimizer.
	///
	/// # Errors
	///
	/// Returns an error when `learning_rate` is not finite and non-negative, when a
	/// captured optimizer state is still in flight, or when its host upload fails.
	pub fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
		if !learning_rate.is_finite() || learning_rate < 0.0 {
			return Err(Error::invalid_argument(
				"AdamW learning rate must be finite and non-negative",
			));
		}
		if let Some(state) = &self.graph_state {
			state.storage().ensure_ready()?;
			state.write_values(&[
				self.step,
				learning_rate.to_bits(),
				self.beta1.to_bits(),
				self.beta2.to_bits(),
				self.epsilon.to_bits(),
				self.weight_decay.to_bits(),
			])?;
		}
		self.learning_rate = learning_rate;
		Ok(())
	}

	/// Return the number of completed logical optimizer steps.
	pub const fn step_count(&self) -> u32 {
		self.step
	}

	pub(super) fn belongs_to(&self, engine: &crate::Engine) -> bool {
		engine.same_as_handle(self.parameters[0].parameter.data().engine_handle())
	}

	pub(super) fn program_signature(&self) -> Result<AdamWProgramSignature> {
		let state = self.graph_state.as_ref().ok_or_else(|| {
			Error::failed_precondition("AdamW did not record graph-resident replay state")
		})?;
		let gradient_ids = self
			.parameters
			.iter()
			.map(|entry| {
				entry
					.parameter
					.gradient()
					.map(|gradient| gradient.value_id())
					.ok_or_else(|| {
						Error::failed_precondition(
							"captured AdamW lost a stable parameter gradient",
						)
					})
			})
			.collect::<Result<Vec<_>>>()?;
		Ok(AdamWProgramSignature {
			optimizer_id: self.id,
			state_id: state.value_id(),
			parameter_ids: self
				.parameters
				.iter()
				.map(|entry| entry.parameter.data().value_id())
				.collect(),
			gradient_ids,
			moment_ids: self
				.parameters
				.iter()
				.map(|entry| {
					(
						entry.first_moment.value_id(),
						entry.second_moment.value_id(),
					)
				})
				.collect(),
			base_step: self.step,
		})
	}

	pub(super) fn validate_program_replay(
		&self,
		signature: &AdamWProgramSignature,
		expected_step: u32,
		logical_step: u32,
	) -> Result<()> {
		let state_matches = self
			.graph_state
			.as_ref()
			.is_some_and(|state| state.value_id() == signature.state_id);
		let parameter_ids = self
			.parameters
			.iter()
			.map(|entry| entry.parameter.data().value_id())
			.collect::<Vec<_>>();
		let gradient_ids = self
			.parameters
			.iter()
			.map(|entry| {
				entry
					.parameter
					.gradient()
					.map(|gradient| gradient.value_id())
			})
			.collect::<Option<Vec<_>>>();
		let moment_ids = self
			.parameters
			.iter()
			.map(|entry| {
				(
					entry.first_moment.value_id(),
					entry.second_moment.value_id(),
				)
			})
			.collect::<Vec<_>>();
		if self.id != signature.optimizer_id
			|| self.step != expected_step
			|| !state_matches
			|| parameter_ids != signature.parameter_ids
			|| gradient_ids.as_deref() != Some(signature.gradient_ids.as_slice())
			|| moment_ids != signature.moment_ids
		{
			return Err(Error::failed_precondition(
				"AdamW state no longer matches the captured training program",
			));
		}
		if logical_step > self.step {
			for state in &self.parameters {
				state.parameter.validate_can_update()?;
			}
		}
		Ok(())
	}

	pub(super) fn complete_program_replay(
		&mut self,
		signature: &AdamWProgramSignature,
		logical_step: u32,
	) -> Result<()> {
		if logical_step > self.step {
			for state in &self.parameters {
				state.parameter.mark_updated()?;
			}
		}
		self.step = logical_step;
		debug_assert!(self.step > signature.base_step);
		Ok(())
	}

	pub(super) fn complete_capture_fallback(
		&mut self,
		signature: &AdamWProgramSignature,
	) -> Result<()> {
		let logical_step = signature
			.base_step
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("training step counter exhausted"))?;
		self.validate_program_replay(signature, signature.base_step, logical_step)?;
		self.complete_program_replay(signature, logical_step)
	}

	fn checkpoint_state(&self) -> Result<OptimizerCheckpoint> {
		Ok(OptimizerCheckpoint {
			kind: "AdamW",
			step: u64::from(self.step),
			learning_rate: self.learning_rate,
			beta1: self.beta1,
			beta2: self.beta2,
			epsilon: self.epsilon,
			weight_decay: self.weight_decay,
			parameters: self
				.parameters
				.iter()
				.map(|state| state.parameter.clone())
				.collect(),
			first_state: self
				.parameters
				.iter()
				.map(|state| state.first_moment.clone())
				.collect(),
			second_state: self
				.parameters
				.iter()
				.map(|state| state.second_moment.clone())
				.collect(),
		})
	}

	fn validate_checkpoint_state(&self, checkpoint: &OptimizerRestore) -> Result<()> {
		if checkpoint.kind != "AdamW"
			|| checkpoint.first_state.len() != self.parameters.len()
			|| checkpoint.second_state.len() != self.parameters.len()
			|| checkpoint.step > u64::from(u32::MAX)
			|| !checkpoint.learning_rate.is_finite()
			|| checkpoint.learning_rate < 0.0
			|| !checkpoint.beta1.is_finite()
			|| !checkpoint.beta2.is_finite()
			|| !checkpoint.epsilon.is_finite()
			|| checkpoint.epsilon <= 0.0
			|| !checkpoint.weight_decay.is_finite()
		{
			return Err(Error::invalid_argument("invalid AdamW checkpoint state"));
		}
		for ((state, first), second) in self
			.parameters
			.iter()
			.zip(&checkpoint.first_state)
			.zip(&checkpoint.second_state)
		{
			let data = state.parameter.data();
			for moment in [first, second] {
				if moment.shape() != data.shape()
					|| moment.dtype() != DType::F32
					|| !moment.engine_handle().same_as(data.engine_handle())
				{
					return Err(Error::invalid_argument(
						"AdamW checkpoint moment contract mismatch",
					));
				}
			}
		}
		Ok(())
	}

	fn restore_checkpoint_state(&mut self, checkpoint: OptimizerRestore) -> Result<()> {
		self.validate_checkpoint_state(&checkpoint)?;
		for ((state, first), second) in self
			.parameters
			.iter_mut()
			.zip(checkpoint.first_state)
			.zip(checkpoint.second_state)
		{
			state.first_moment = first;
			state.second_moment = second;
		}
		self.step = checkpoint.step as u32;
		self.learning_rate = checkpoint.learning_rate;
		self.beta1 = checkpoint.beta1;
		self.beta2 = checkpoint.beta2;
		self.epsilon = checkpoint.epsilon;
		self.weight_decay = checkpoint.weight_decay;
		self.graph_state = None;
		Ok(())
	}
}

impl checkpoint_capability::Persistence for AdamW {
	fn checkpoint_state(&self) -> Result<OptimizerCheckpoint> {
		Self::checkpoint_state(self)
	}

	fn validate_checkpoint_state(&self, state: &OptimizerRestore) -> Result<()> {
		Self::validate_checkpoint_state(self, state)
	}

	fn restore_checkpoint_state(&mut self, state: OptimizerRestore) -> Result<()> {
		Self::restore_checkpoint_state(self, state)
	}
}

impl Optimizer for AdamW {
	fn zero_grad(&self) {
		Self::zero_grad(self);
	}

	fn step(&mut self) -> Result<()> {
		Self::step(self)
	}

	fn learning_rate(&self) -> f32 {
		Self::learning_rate(self)
	}

	fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
		Self::set_learning_rate(self, learning_rate)
	}

	fn step_count(&self) -> u64 {
		u64::from(Self::step_count(self))
	}

	fn belongs_to(&self, engine: &Engine) -> bool {
		Self::belongs_to(self, engine)
	}
}

fn next_optimizer_id() -> Result<u64> {
	static NEXT_OPTIMIZER_ID: AtomicU64 = AtomicU64::new(1);
	NEXT_OPTIMIZER_ID
		.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
			(value != u64::MAX).then_some(value + 1)
		})
		.map_err(|_| Error::resource_exhausted("optimizer identity exhausted"))
}

#[cfg(test)]
mod tests {
	use super::{NoOpOptimizer, Optimizer};

	#[test]
	fn no_op_optimizer_is_an_object_safe_policy_owner() {
		let mut optimizer = NoOpOptimizer::new(0.25).expect("valid rate");
		let optimizer: &mut dyn Optimizer = &mut optimizer;
		assert_eq!(optimizer.learning_rate(), 0.25);
		assert_eq!(optimizer.step_count(), 0);
		optimizer.zero_grad();
		optimizer.step().expect("no-op step");
		optimizer.set_learning_rate(0.125).expect("valid rate");
		assert_eq!(optimizer.learning_rate(), 0.125);
		assert_eq!(optimizer.step_count(), 0);
	}

	#[test]
	fn no_op_optimizer_rejects_invalid_rates() {
		assert!(NoOpOptimizer::new(f32::NAN).is_err());
		let mut optimizer = NoOpOptimizer::default();
		assert!(optimizer.set_learning_rate(-1.0).is_err());
	}
}
