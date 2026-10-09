//! AdamW optimizer with decoupled weight decay and graph-capture support.

use crate::ml::Parameter;
use crate::ml::validation::{shader_u32, validate_f32_same_engine};
use crate::runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch};
use crate::{DType, Engine, Error, Matrix, OpAttribute, Result};

use super::optimizer::{
	MomentParameterState, Optimizer, OptimizerCheckpoint, OptimizerRestore, checkpoint_capability,
	next_optimizer_id,
};

#[derive(Clone, Copy)]
pub(super) struct AdamWScalars {
	pub(super) step: u32,
	pub(super) learning_rate: f32,
	pub(super) beta1: f32,
	pub(super) beta2: f32,
	pub(super) epsilon: f32,
	pub(super) weight_decay: f32,
}

#[derive(Clone, Copy)]
pub(super) struct AdamWBatchEntry<'a> {
	pub(super) parameter: &'a Matrix,
	pub(super) gradient: &'a Matrix,
	pub(super) first_moment: &'a Matrix,
	pub(super) second_moment: &'a Matrix,
}

fn adamw_graph_advance(state: &Matrix) -> Result<()> {
	const OPERATION: &str = crate::core::operation::ml::ADAMW_GRAPH_ADVANCE.name();
	if state.dtype() != DType::U32 || state.shape() != [6] {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires U32 optimizer state [6]"
		)));
	}
	let buffers = [BufferBinding::read_write(state.storage())];
	{
		let inputs: &[&Matrix] = &[state];
		let outputs: &[&Matrix] = &[state];
		let attributes: &[OpAttribute] = &[];
		let dispatch = ComputeDispatch {
			kernel: KernelId::MlAdamWGraphAdvanceU32,
			buffers: &buffers,
			push_constants: &[],
			workgroups: [1, 1, 1],
		};
		let contract = dispatch.kernel.semantic_contract().ok_or_else(|| {
			Error::internal(format!(
				"lowering-only kernel {} cannot own one semantic ML operation",
				dispatch.kernel.report_name()
			))
		})?;
		let engine = inputs
			.first()
			.ok_or_else(|| Error::internal("semantic ML dispatch requires an input"))?
			.engine_handle();
		engine.record_semantic(
			dispatch,
			SemanticDispatch {
				contract,
				inputs,
				outputs,
				attributes,
			},
		)
	}
}

fn adamw_graph(
	parameter: &Matrix,
	gradient: &Matrix,
	first_moment: &Matrix,
	second_moment: &Matrix,
	state: &Matrix,
) -> Result<()> {
	const OPERATION: &str = crate::core::operation::ml::ADAMW_GRAPH.name();
	validate_f32_same_engine(
		OPERATION,
		&[parameter, gradient, first_moment, second_moment],
	)?;
	if state.dtype() != DType::U32 || !state.engine_handle().same_as(parameter.engine_handle()) {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires same-engine U32 optimizer state"
		)));
	}
	for candidate in [gradient, first_moment, second_moment] {
		if candidate.shape() != parameter.shape() {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} requires identical parameter, gradient, and moment shapes"
			)));
		}
	}
	if state.shape() != [6] {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires optimizer state [6]"
		)));
	}
	let element_count = shader_u32(parameter.num_elements(), "element count", OPERATION)?;
	if element_count == 0 {
		return Ok(());
	}
	let buffers = [
		BufferBinding::read_write(parameter.storage()),
		BufferBinding::read(gradient.storage()),
		BufferBinding::read_write(first_moment.storage()),
		BufferBinding::read_write(second_moment.storage()),
		BufferBinding::read(state.storage()),
	];
	let push_constants = [PushConstant::U32(element_count)];
	let kernel = KernelId::MlAdamWGraphF32;
	{
		let inputs: &[&Matrix] = &[parameter, gradient, first_moment, second_moment, state];
		let outputs: &[&Matrix] = &[parameter, first_moment, second_moment];
		let attributes: &[OpAttribute] = &[];
		let dispatch = ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(element_count),
		};
		let contract = dispatch.kernel.semantic_contract().ok_or_else(|| {
			Error::internal(format!(
				"lowering-only kernel {} cannot own one semantic ML operation",
				dispatch.kernel.report_name()
			))
		})?;
		let engine = inputs
			.first()
			.ok_or_else(|| Error::internal("semantic ML dispatch requires an input"))?
			.engine_handle();
		engine.record_semantic(
			dispatch,
			SemanticDispatch {
				contract,
				inputs,
				outputs,
				attributes,
			},
		)
	}
}

fn adamw_many4_graph(entries: &[AdamWBatchEntry<'_>; 4], state: &Matrix) -> Result<()> {
	const OPERATION: &str = "ml.adamw_many4_graph";
	let counts = validate_adamw_batch(entries, Some(state), OPERATION)?;
	let maximum = counts.iter().copied().max().unwrap_or(0);
	if maximum == 0 {
		return Ok(());
	}
	let buffers = adamw_many4_graph_buffers(entries, state);
	let push_constants = counts.map(PushConstant::U32);
	let input0 = adamw_graph_inputs(entries[0], state);
	let input1 = adamw_graph_inputs(entries[1], state);
	let input2 = adamw_graph_inputs(entries[2], state);
	let input3 = adamw_graph_inputs(entries[3], state);
	let output0 = adamw_outputs(entries[0]);
	let output1 = adamw_outputs(entries[1]);
	let output2 = adamw_outputs(entries[2]);
	let output3 = adamw_outputs(entries[3]);
	let contract = crate::core::operation::ml::ADAMW_GRAPH;
	let semantics = [
		SemanticDispatch {
			contract,
			inputs: &input0,
			outputs: &output0,
			attributes: &[],
		},
		SemanticDispatch {
			contract,
			inputs: &input1,
			outputs: &output1,
			attributes: &[],
		},
		SemanticDispatch {
			contract,
			inputs: &input2,
			outputs: &output2,
			attributes: &[],
		},
		SemanticDispatch {
			contract,
			inputs: &input3,
			outputs: &output3,
			attributes: &[],
		},
	];
	let kernel = KernelId::MlAdamWMany4GraphF32;
	entries[0].parameter.engine_handle().record_fused_semantic(
		ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(maximum),
		},
		&semantics,
	)
}

fn adamw(
	parameter: &Matrix,
	gradient: &Matrix,
	first_moment: &Matrix,
	second_moment: &Matrix,
	scalars: AdamWScalars,
) -> Result<()> {
	const OPERATION: &str = crate::core::operation::ml::ADAMW.name();
	validate_f32_same_engine(
		OPERATION,
		&[parameter, gradient, first_moment, second_moment],
	)?;
	for candidate in [gradient, first_moment, second_moment] {
		if candidate.shape() != parameter.shape() {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} requires identical parameter, gradient, and moment shapes"
			)));
		}
	}
	let element_count = shader_u32(parameter.num_elements(), "element count", OPERATION)?;
	if element_count != 0 {
		let buffers = [
			BufferBinding::read_write(parameter.storage()),
			BufferBinding::read(gradient.storage()),
			BufferBinding::read_write(first_moment.storage()),
			BufferBinding::read_write(second_moment.storage()),
		];
		let push_constants = [
			PushConstant::U32(element_count),
			PushConstant::U32(scalars.step),
			PushConstant::F32(scalars.learning_rate),
			PushConstant::F32(scalars.beta1),
			PushConstant::F32(scalars.beta2),
			PushConstant::F32(scalars.epsilon),
			PushConstant::F32(scalars.weight_decay),
		];
		let kernel = KernelId::MlAdamWF32;
		let attributes = adamw_attributes(scalars);
		{
			let inputs: &[&Matrix] = &[parameter, gradient, first_moment, second_moment];
			let outputs: &[&Matrix] = &[parameter, first_moment, second_moment];
			let attributes: &[OpAttribute] = &attributes;
			let dispatch = ComputeDispatch {
				kernel,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: kernel.linear_workgroups(element_count),
			};
			let contract = dispatch.kernel.semantic_contract().ok_or_else(|| {
				Error::internal(format!(
					"lowering-only kernel {} cannot own one semantic ML operation",
					dispatch.kernel.report_name()
				))
			})?;
			let engine = inputs
				.first()
				.ok_or_else(|| Error::internal("semantic ML dispatch requires an input"))?
				.engine_handle();
			engine.record_semantic(
				dispatch,
				SemanticDispatch {
					contract,
					inputs,
					outputs,
					attributes,
				},
			)
		}?;
	}
	Ok(())
}

fn adamw_many4(entries: &[AdamWBatchEntry<'_>; 4], scalars: AdamWScalars) -> Result<()> {
	const OPERATION: &str = "ml.adamw_many4";
	let counts = validate_adamw_batch(entries, None, OPERATION)?;
	let maximum = counts.iter().copied().max().unwrap_or(0);
	if maximum == 0 {
		return Ok(());
	}
	let buffers = adamw_many4_buffers(entries);
	let push_constants = [
		PushConstant::U32(counts[0]),
		PushConstant::U32(counts[1]),
		PushConstant::U32(counts[2]),
		PushConstant::U32(counts[3]),
		PushConstant::F32(scalars.learning_rate),
		PushConstant::F32(scalars.beta1),
		PushConstant::F32(scalars.beta2),
		PushConstant::F32(scalars.epsilon),
		PushConstant::F32(scalars.weight_decay),
		PushConstant::U32(scalars.step),
	];
	let attributes = adamw_attributes(scalars);
	let input0 = adamw_inputs(entries[0]);
	let input1 = adamw_inputs(entries[1]);
	let input2 = adamw_inputs(entries[2]);
	let input3 = adamw_inputs(entries[3]);
	let output0 = adamw_outputs(entries[0]);
	let output1 = adamw_outputs(entries[1]);
	let output2 = adamw_outputs(entries[2]);
	let output3 = adamw_outputs(entries[3]);
	let contract = crate::core::operation::ml::ADAMW;
	let semantics = [
		SemanticDispatch {
			contract,
			inputs: &input0,
			outputs: &output0,
			attributes: &attributes,
		},
		SemanticDispatch {
			contract,
			inputs: &input1,
			outputs: &output1,
			attributes: &attributes,
		},
		SemanticDispatch {
			contract,
			inputs: &input2,
			outputs: &output2,
			attributes: &attributes,
		},
		SemanticDispatch {
			contract,
			inputs: &input3,
			outputs: &output3,
			attributes: &attributes,
		},
	];
	let kernel = KernelId::MlAdamWMany4F32;
	entries[0].parameter.engine_handle().record_fused_semantic(
		ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(maximum),
		},
		&semantics,
	)
}

fn validate_adamw_batch(
	entries: &[AdamWBatchEntry<'_>; 4],
	state: Option<&Matrix>,
	operation: &'static str,
) -> Result<[u32; 4]> {
	let engine = entries[0].parameter.engine_handle();
	if let Some(state) = state
		&& (state.dtype() != DType::U32
			|| state.shape() != [6]
			|| !engine.same_as(state.engine_handle()))
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires same-engine U32 optimizer state [6]"
		)));
	}
	let mut counts = [0_u32; 4];
	for (index, entry) in entries.iter().enumerate() {
		validate_f32_same_engine(
			operation,
			&[
				entry.parameter,
				entry.gradient,
				entry.first_moment,
				entry.second_moment,
			],
		)?;
		if !engine.same_as(entry.parameter.engine_handle())
			|| [entry.gradient, entry.first_moment, entry.second_moment]
				.iter()
				.any(|candidate| candidate.shape() != entry.parameter.shape())
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires four same-engine parameter, gradient, and moment groups with identical per-group shapes"
			)));
		}
		counts[index] = shader_u32(entry.parameter.num_elements(), "element count", operation)?;
	}
	Ok(counts)
}

fn adamw_inputs(entry: AdamWBatchEntry<'_>) -> [&Matrix; 4] {
	[
		entry.parameter,
		entry.gradient,
		entry.first_moment,
		entry.second_moment,
	]
}

fn adamw_graph_inputs<'a>(entry: AdamWBatchEntry<'a>, state: &'a Matrix) -> [&'a Matrix; 5] {
	[
		entry.parameter,
		entry.gradient,
		entry.first_moment,
		entry.second_moment,
		state,
	]
}

fn adamw_outputs(entry: AdamWBatchEntry<'_>) -> [&Matrix; 3] {
	[entry.parameter, entry.first_moment, entry.second_moment]
}

fn adamw_many4_buffers<'a>(entries: &[AdamWBatchEntry<'a>; 4]) -> [BufferBinding<'a>; 16] {
	[
		BufferBinding::read_write(entries[0].parameter.storage()),
		BufferBinding::read(entries[0].gradient.storage()),
		BufferBinding::read_write(entries[0].first_moment.storage()),
		BufferBinding::read_write(entries[0].second_moment.storage()),
		BufferBinding::read_write(entries[1].parameter.storage()),
		BufferBinding::read(entries[1].gradient.storage()),
		BufferBinding::read_write(entries[1].first_moment.storage()),
		BufferBinding::read_write(entries[1].second_moment.storage()),
		BufferBinding::read_write(entries[2].parameter.storage()),
		BufferBinding::read(entries[2].gradient.storage()),
		BufferBinding::read_write(entries[2].first_moment.storage()),
		BufferBinding::read_write(entries[2].second_moment.storage()),
		BufferBinding::read_write(entries[3].parameter.storage()),
		BufferBinding::read(entries[3].gradient.storage()),
		BufferBinding::read_write(entries[3].first_moment.storage()),
		BufferBinding::read_write(entries[3].second_moment.storage()),
	]
}

fn adamw_many4_graph_buffers<'a>(
	entries: &[AdamWBatchEntry<'a>; 4],
	state: &'a Matrix,
) -> [BufferBinding<'a>; 17] {
	let base = adamw_many4_buffers(entries);
	[
		base[0],
		base[1],
		base[2],
		base[3],
		base[4],
		base[5],
		base[6],
		base[7],
		base[8],
		base[9],
		base[10],
		base[11],
		base[12],
		base[13],
		base[14],
		base[15],
		BufferBinding::read(state.storage()),
	]
}

fn adamw_attributes(scalars: AdamWScalars) -> [OpAttribute; 6] {
	[
		OpAttribute::UnsignedInteger {
			name: "step".into(),
			value: u64::from(scalars.step),
		},
		OpAttribute::Float {
			name: "learning_rate".into(),
			value: f64::from(scalars.learning_rate),
		},
		OpAttribute::Float {
			name: "beta1".into(),
			value: f64::from(scalars.beta1),
		},
		OpAttribute::Float {
			name: "beta2".into(),
			value: f64::from(scalars.beta2),
		},
		OpAttribute::Float {
			name: "epsilon".into(),
			value: f64::from(scalars.epsilon),
		},
		OpAttribute::Float {
			name: "weight_decay".into(),
			value: f64::from(scalars.weight_decay),
		},
	]
}

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
			let entries = std::array::from_fn(|index| AdamWBatchEntry {
				parameter: &batch[index].1,
				gradient: &batch[index].2,
				first_moment: &batch[index].0.first_moment,
				second_moment: &batch[index].0.second_moment,
			});
			adamw_many4(&entries, scalars)?;
		}
		for (state, parameter, gradient) in remainder {
			adamw(
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
		adamw_graph_advance(&state)?;
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
				AdamWBatchEntry {
					parameter: &parameters[index],
					gradient: &gradients[index],
					first_moment: &self.parameters[index].first_moment,
					second_moment: &self.parameters[index].second_moment,
				}
			});
			adamw_many4_graph(&entries, &state)?;
		}
		for index in indices {
			let parameter_state = &self.parameters[index];
			adamw_graph(
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
