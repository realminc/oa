//! Stochastic gradient descent optimizer.

use crate::ml::Parameter;
use crate::ml::lowering::optim::{self as dispatch, SgdScalars};
use crate::{DType, Engine, Error, Matrix, Result};

use super::optimizer::{
	Optimizer, OptimizerCheckpoint, OptimizerRestore, SgdParameterState, checkpoint_capability,
};

/// Stochastic gradient descent over stable [`Parameter`] handles.
pub struct Sgd {
	pub(super) parameters: Vec<SgdParameterState>,
	pub(super) learning_rate: f32,
	pub(super) momentum: f32,
	pub(super) weight_decay: f32,
	pub(super) step: u64,
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
			let momentum_buf = (momentum > 0.0)
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
				momentum: momentum_buf,
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

	pub(super) fn belongs_to_engine(&self, engine: &Engine) -> bool {
		engine.same_as_handle(self.parameters[0].parameter.data().engine_handle())
	}

	pub(super) fn checkpoint(&self) -> Result<OptimizerCheckpoint> {
		if self.momentum > 0.0 && self.step > 0 {
			return Err(crate::Error::failed_precondition(
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

	pub(super) fn validate_checkpoint(&self, state: &OptimizerRestore) -> Result<()> {
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

	pub(super) fn restore_checkpoint(&mut self, state: OptimizerRestore) -> Result<()> {
		self.validate_checkpoint(&state)?;
		self.step = state.step;
		self.learning_rate = state.learning_rate;
		self.weight_decay = state.weight_decay;
		Ok(())
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
		Self::checkpoint(self)
	}

	fn validate_checkpoint_state(&self, state: &OptimizerRestore) -> Result<()> {
		Self::validate_checkpoint(self, state)
	}

	fn restore_checkpoint_state(&mut self, state: OptimizerRestore) -> Result<()> {
		Self::restore_checkpoint(self, state)
	}
}
