//! Adam optimizer.

use crate::ml::Parameter;
use crate::ml::lowering::optim::{self as dispatch, AdamScalars};
use crate::{DType, Engine, Error, Matrix, Result};

use super::optimizer::{
	MomentParameterState, Optimizer, OptimizerCheckpoint, OptimizerRestore, checkpoint_capability,
};

/// Adam optimizer over stable [`Parameter`] handles.
pub struct Adam {
	pub(super) parameters: Vec<MomentParameterState>,
	pub(super) learning_rate: f32,
	pub(super) beta1: f32,
	pub(super) beta2: f32,
	pub(super) epsilon: f32,
	pub(super) step: u32,
}

impl Adam {
	/// Bind Adam to a nonempty parameter set using standard OA defaults.
	///
	/// # Errors
	///
	/// Returns an error for invalid ownership, duplicate parameters, or moment
	/// allocation failure.
	pub fn new(parameters: impl IntoIterator<Item = Parameter>, learning_rate: f32) -> Result<Self> {
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

	pub(super) fn belongs_to_engine(&self, engine: &Engine) -> bool {
		engine.same_as_handle(self.parameters[0].parameter.data().engine_handle())
	}

	pub(super) fn checkpoint(&self) -> Result<OptimizerCheckpoint> {
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

	pub(super) fn validate_checkpoint(&self, state: &OptimizerRestore) -> Result<()> {
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

	pub(super) fn restore_checkpoint(&mut self, state: OptimizerRestore) -> Result<()> {
		self.validate_checkpoint(&state)?;
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
		Self::checkpoint(self)
	}

	fn validate_checkpoint_state(&self, state: &OptimizerRestore) -> Result<()> {
		Self::validate_checkpoint(self, state)
	}

	fn restore_checkpoint_state(&mut self, state: OptimizerRestore) -> Result<()> {
		Self::restore_checkpoint(self, state)
	}
}
