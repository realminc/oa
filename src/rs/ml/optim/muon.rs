//! Muon optimizer with Nesterov momentum and NS5 orthogonalization.

use crate::ml::Parameter;
use crate::ml::lowering::optim::{self as dispatch, MuonScalars};
use crate::{DType, Engine, Error, Matrix, Result};

use super::optimizer::{
	MuonParameterState, Optimizer, OptimizerCheckpoint, OptimizerRestore, checkpoint_capability,
};

/// Muon optimizer with Nesterov momentum and rank-two NS5 orthogonalization.
///
/// Rank-two parameters use the donor Muon matrix pipeline. Other ranks use its
/// fused momentum update; OA never silently delegates them to AdamW.
pub struct Muon {
	pub(super) parameters: Vec<MuonParameterState>,
	pub(super) learning_rate: f32,
	pub(super) beta: f32,
	pub(super) weight_decay: f32,
	pub(super) epsilon: f32,
	pub(super) ns5_iterations: u32,
	pub(super) step: u64,
}

impl Muon {
	/// Bind Muon to a nonempty FP32 parameter set using OA defaults.
	///
	/// # Errors
	///
	/// Returns an error for invalid ownership, duplicate parameters, or momentum
	/// allocation failure.
	pub fn new(parameters: impl IntoIterator<Item = Parameter>, learning_rate: f32) -> Result<Self> {
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

	pub(super) fn belongs_to_engine(&self, engine: &Engine) -> bool {
		engine.same_as_handle(self.parameters[0].parameter.data().engine_handle())
	}

	pub(super) fn checkpoint(&self) -> Result<OptimizerCheckpoint> {
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

	pub(super) fn validate_checkpoint(&self, state: &OptimizerRestore) -> Result<()> {
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

	pub(super) fn restore_checkpoint(&mut self, state: OptimizerRestore) -> Result<()> {
		self.validate_checkpoint(&state)?;
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
		Self::checkpoint(self)
	}

	fn validate_checkpoint_state(&self, state: &OptimizerRestore) -> Result<()> {
		Self::validate_checkpoint(self, state)
	}

	fn restore_checkpoint_state(&mut self, state: OptimizerRestore) -> Result<()> {
		Self::restore_checkpoint(self, state)
	}
}
