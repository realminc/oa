use std::sync::atomic::{AtomicU64, Ordering};

use crate::{DType, Error, Matrix, Result};

use super::{
	Parameter,
	kernels::{self, AdamWScalars},
};

struct ParameterState {
	parameter: Parameter,
	first_moment: Matrix,
	second_moment: Matrix,
}

pub(crate) struct AdamWCheckpoint {
	pub(crate) step: u32,
	pub(crate) learning_rate: f32,
	pub(crate) beta1: f32,
	pub(crate) beta2: f32,
	pub(crate) epsilon: f32,
	pub(crate) weight_decay: f32,
	pub(crate) states: Vec<(Parameter, Matrix, Matrix)>,
}

pub(crate) struct AdamWRestore {
	pub(crate) step: u32,
	pub(crate) learning_rate: f32,
	pub(crate) beta1: f32,
	pub(crate) beta2: f32,
	pub(crate) epsilon: f32,
	pub(crate) weight_decay: f32,
	pub(crate) moments: Vec<(Matrix, Matrix)>,
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
	parameters: Vec<ParameterState>,
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
	/// finite and positive, a parameter handle appears more than once, or moment
	/// allocation fails.
	pub fn new(
		parameters: impl IntoIterator<Item = Parameter>,
		learning_rate: f32,
	) -> Result<Self> {
		if !learning_rate.is_finite() || learning_rate <= 0.0 {
			return Err(Error::invalid_argument(
				"AdamW learning rate must be finite and positive",
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
			states.push(ParameterState {
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
		for state in &self.parameters {
			let Some(gradient) = state.parameter.gradient() else {
				continue;
			};
			let parameter = state.parameter.data();
			kernels::adamw(
				&parameter,
				&gradient,
				&state.first_moment,
				&state.second_moment,
				AdamWScalars {
					step,
					learning_rate: self.learning_rate,
					beta1: self.beta1,
					beta2: self.beta2,
					epsilon: self.epsilon,
					weight_decay: self.weight_decay,
				},
			)?;
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
		kernels::adamw_graph_advance(&state)?;
		for (parameter_state, gradient) in self.parameters.iter().zip(&gradients) {
			let parameter = parameter_state.parameter.data();
			kernels::adamw_graph(
				&parameter,
				gradient,
				&parameter_state.first_moment,
				&parameter_state.second_moment,
				&state,
			)?;
		}
		self.graph_state = Some(state);
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

	pub(crate) fn checkpoint(&self) -> AdamWCheckpoint {
		AdamWCheckpoint {
			step: self.step,
			learning_rate: self.learning_rate,
			beta1: self.beta1,
			beta2: self.beta2,
			epsilon: self.epsilon,
			weight_decay: self.weight_decay,
			states: self
				.parameters
				.iter()
				.map(|state| {
					(
						state.parameter.clone(),
						state.first_moment.clone(),
						state.second_moment.clone(),
					)
				})
				.collect(),
		}
	}

	pub(crate) fn restore_checkpoint(&mut self, checkpoint: AdamWRestore) -> Result<()> {
		if checkpoint.moments.len() != self.parameters.len()
			|| !checkpoint.learning_rate.is_finite()
			|| checkpoint.learning_rate <= 0.0
			|| !checkpoint.beta1.is_finite()
			|| !checkpoint.beta2.is_finite()
			|| !checkpoint.epsilon.is_finite()
			|| checkpoint.epsilon <= 0.0
			|| !checkpoint.weight_decay.is_finite()
		{
			return Err(Error::invalid_argument("invalid AdamW checkpoint state"));
		}
		for (state, (first, second)) in self.parameters.iter().zip(&checkpoint.moments) {
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
		for (state, (first, second)) in self.parameters.iter_mut().zip(checkpoint.moments) {
			state.first_moment = first;
			state.second_moment = second;
		}
		self.step = checkpoint.step;
		self.learning_rate = checkpoint.learning_rate;
		self.beta1 = checkpoint.beta1;
		self.beta2 = checkpoint.beta2;
		self.epsilon = checkpoint.epsilon;
		self.weight_decay = checkpoint.weight_decay;
		self.graph_state = None;
		Ok(())
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
