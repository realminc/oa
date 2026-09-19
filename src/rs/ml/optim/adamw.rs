//! AdamW optimizer with decoupled weight decay and graph-capture support.

use crate::ml::Parameter;
use crate::ml::lowering::optim::{self as dispatch, AdamWScalars};
use crate::{DType, Engine, Error, Matrix, Result};

use super::optimizer::{
	MomentParameterState, Optimizer, OptimizerCheckpoint, OptimizerRestore, checkpoint_capability,
	next_optimizer_id,
};

pub struct AdamWProgramSignature {
	pub(crate) optimizer_id: u64,
	pub(crate) state_id: u64,
	pub(crate) parameter_ids: Vec<u64>,
	pub(crate) gradient_ids: Vec<u64>,
	pub(crate) moment_ids: Vec<(u64, u64)>,
	pub(crate) base_step: u32,
}

/// Decoupled-weight-decay Adam optimizer over stable [`Parameter`] handles.
pub struct AdamW {
	pub(super) id: u64,
	pub(super) parameters: Vec<MomentParameterState>,
	pub(super) learning_rate: f32,
	pub(super) beta1: f32,
	pub(super) beta2: f32,
	pub(super) epsilon: f32,
	pub(super) weight_decay: f32,
	pub(super) step: u32,
	pub(super) graph_state: Option<Matrix>,
}

impl AdamW {
	/// Bind AdamW to a nonempty parameter set using standard OA defaults.
	///
	/// # Errors
	///
	/// Returns an error when the parameter set is empty, learning rate is not
	/// finite and non-negative, a parameter handle appears more than once, or moment
	/// allocation fails.
	pub fn new(parameters: impl IntoIterator<Item = Parameter>, learning_rate: f32) -> Result<Self> {
		Self::with_hyperparameters(parameters, learning_rate, 0.9, 0.999, 1.0e-8, 0.01)
	}

	/// Bind AdamW with explicit scalar hyperparameters.
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
		weight_decay: f32,
	) -> Result<Self> {
		if !learning_rate.is_finite() || learning_rate < 0.0 {
			return Err(Error::invalid_argument(
				"AdamW learning rate must be finite and non-negative",
			));
		}
		if !beta1.is_finite() || !(0.0..1.0).contains(&beta1) {
			return Err(Error::invalid_argument("AdamW beta1 must be in [0, 1)"));
		}
		if !beta2.is_finite() || !(0.0..1.0).contains(&beta2) {
			return Err(Error::invalid_argument("AdamW beta2 must be in [0, 1)"));
		}
		if !epsilon.is_finite() || epsilon <= 0.0 {
			return Err(Error::invalid_argument(
				"AdamW epsilon must be finite and positive",
			));
		}
		if !weight_decay.is_finite() || weight_decay < 0.0 {
			return Err(Error::invalid_argument(
				"AdamW weight decay must be finite and non-negative",
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
			beta1,
			beta2,
			epsilon,
			weight_decay,
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

	pub(crate) fn belongs_to(&self, engine: &Engine) -> bool {
		engine.same_as_handle(self.parameters[0].parameter.data().engine_handle())
	}

	pub(crate) fn program_signature(&self) -> Result<AdamWProgramSignature> {
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
						Error::failed_precondition("captured AdamW lost a stable parameter gradient")
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

	pub(crate) fn validate_program_replay(
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

	pub(crate) fn complete_program_replay(
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

	pub(crate) fn complete_capture_fallback(
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

	pub(super) fn checkpoint(&self) -> Result<OptimizerCheckpoint> {
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

	pub(super) fn validate_checkpoint(&self, checkpoint: &OptimizerRestore) -> Result<()> {
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

	pub(super) fn restore_checkpoint(&mut self, checkpoint: OptimizerRestore) -> Result<()> {
		self.validate_checkpoint(&checkpoint)?;
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

impl checkpoint_capability::Persistence for AdamW {
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
