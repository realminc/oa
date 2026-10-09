//! Muon optimizer with Nesterov momentum and NS5 orthogonalization.

use crate::ml::Parameter;
use crate::ml::validation::{shader_u32, validate_f32_same_engine};
use crate::runtime::{
	BufferAccess, BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch,
};
use crate::{DType, Engine, Error, Matrix, OpAttribute, Result};

use super::optimizer::{
	MuonParameterState, Optimizer, OptimizerCheckpoint, OptimizerRestore, checkpoint_capability,
};

#[derive(Clone, Copy)]
pub(super) struct MuonScalars {
	pub(super) learning_rate: f32,
	pub(super) beta: f32,
	pub(super) weight_decay: f32,
	pub(super) epsilon: f32,
	pub(super) ns5_iterations: u32,
}

#[derive(Clone, Copy)]
enum MuonSlot {
	Parameter,
	Gradient,
	Momentum,
	Temporary(usize),
}

struct MuonStage {
	kernel: KernelId,
	bindings: Vec<(MuonSlot, BufferAccess)>,
	push_constants: Vec<PushConstant>,
	workgroups: [u32; 3],
}

fn muon(
	parameter: &Matrix,
	gradient: &Matrix,
	momentum: &Matrix,
	scalars: MuonScalars,
) -> Result<()> {
	const OPERATION: &str = crate::core::operation::ml::MUON.name();
	validate_f32_same_engine(OPERATION, &[parameter, gradient, momentum])?;
	for candidate in [gradient, momentum] {
		if candidate.shape() != parameter.shape() {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} requires identical parameter, gradient, and momentum shapes"
			)));
		}
	}
	let element_count = shader_u32(parameter.num_elements(), "element count", OPERATION)?;
	if element_count == 0 {
		return Ok(());
	}
	let attributes = muon_attributes(scalars);
	if parameter.shape().len() != 2 || scalars.ns5_iterations == 0 {
		let buffers = [
			BufferBinding::read_write(parameter.storage()),
			BufferBinding::read(gradient.storage()),
			BufferBinding::read_write(momentum.storage()),
		];
		let push_constants = [
			PushConstant::U32(element_count),
			PushConstant::F32(scalars.learning_rate),
			PushConstant::F32(scalars.beta),
			PushConstant::F32(scalars.weight_decay),
		];
		let kernel = KernelId::MlMuonVectorF32;
		return {
			let inputs: &[&Matrix] = &[parameter, gradient, momentum];
			let outputs: &[&Matrix] = &[parameter, momentum];
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
		};
	}
	let [rows, columns] = parameter.shape() else {
		unreachable!("rank checked above")
	};
	let rows = shader_u32(*rows, "row count", OPERATION)?;
	let columns = shader_u32(*columns, "column count", OPERATION)?;
	muon_matrix(
		parameter,
		gradient,
		momentum,
		rows,
		columns,
		scalars,
		&attributes,
	)
}

fn muon_matrix(
	parameter: &Matrix,
	gradient: &Matrix,
	momentum: &Matrix,
	rows: u32,
	columns: u32,
	scalars: MuonScalars,
	attributes: &[OpAttribute],
) -> Result<()> {
	const NS_A: f32 = 3.4445;
	const NS_B: f32 = -4.7750;
	const NS_C: f32 = 2.0315;
	let count = rows
		.checked_mul(columns)
		.ok_or_else(|| Error::invalid_argument("ml.muon matrix size exceeds u32"))?;
	let transposed = rows > columns;
	let (operation_rows, operation_columns) = if transposed {
		(columns, rows)
	} else {
		(rows, columns)
	};
	let mut temporaries = Vec::new();
	let mut stages = Vec::new();
	let update = allocate_muon_temp(parameter, &[rows, columns], &mut temporaries)?;
	stages.push(MuonStage {
		kernel: KernelId::MlMuonNesterovF32,
		bindings: vec![
			(MuonSlot::Gradient, BufferAccess::Read),
			(MuonSlot::Momentum, BufferAccess::ReadWrite),
			(update, BufferAccess::Write),
		],
		push_constants: vec![PushConstant::U32(count), PushConstant::F32(scalars.beta)],
		workgroups: KernelId::MlMuonNesterovF32.linear_workgroups(count),
	});
	let mut z = if transposed {
		let transposed_update = allocate_muon_temp(
			parameter,
			&[operation_rows, operation_columns],
			&mut temporaries,
		)?;
		stages.push(muon_transpose_stage(
			update,
			transposed_update,
			rows,
			columns,
		));
		transposed_update
	} else {
		update
	};
	let norm = allocate_muon_temp(parameter, &[1], &mut temporaries)?;
	stages.push(MuonStage {
		kernel: KernelId::MlMuonNormalizeF32,
		bindings: vec![
			(z, BufferAccess::Read),
			(z, BufferAccess::ReadWrite),
			(norm, BufferAccess::Write),
		],
		push_constants: vec![
			PushConstant::U32(operation_rows),
			PushConstant::U32(operation_columns),
			PushConstant::F32(scalars.epsilon),
		],
		workgroups: [1, 1, 1],
	});
	for _ in 0..scalars.ns5_iterations {
		let a = allocate_muon_temp(
			parameter,
			&[operation_rows, operation_rows],
			&mut temporaries,
		)?;
		stages.push(muon_mat_mul_nt_stage(
			z,
			z,
			a,
			operation_rows,
			operation_rows,
			operation_columns,
		));
		let next_z = allocate_muon_temp(
			parameter,
			&[operation_rows, operation_columns],
			&mut temporaries,
		)?;
		if operation_rows <= 64 {
			let b = allocate_muon_temp(
				parameter,
				&[operation_rows, operation_rows],
				&mut temporaries,
			)?;
			stages.push(muon_mat_mul_axpby_stage(
				a,
				a,
				a,
				b,
				operation_rows,
				operation_rows,
				operation_rows,
				NS_C,
				NS_B,
			));
			stages.push(muon_mat_mul_axpby_stage(
				b,
				z,
				z,
				next_z,
				operation_rows,
				operation_columns,
				operation_rows,
				1.0,
				NS_A,
			));
		} else {
			let aa = allocate_muon_temp(
				parameter,
				&[operation_rows, operation_rows],
				&mut temporaries,
			)?;
			stages.push(muon_mat_mul_nt_stage(
				a,
				a,
				aa,
				operation_rows,
				operation_rows,
				operation_rows,
			));
			let b = allocate_muon_temp(
				parameter,
				&[operation_rows, operation_rows],
				&mut temporaries,
			)?;
			stages.push(muon_linear_stage(
				a,
				aa,
				b,
				operation_rows * operation_rows,
				NS_B,
				NS_C,
			));
			let z_transpose = allocate_muon_temp(
				parameter,
				&[operation_columns, operation_rows],
				&mut temporaries,
			)?;
			stages.push(muon_transpose_stage(
				z,
				z_transpose,
				operation_rows,
				operation_columns,
			));
			let bz = allocate_muon_temp(
				parameter,
				&[operation_rows, operation_columns],
				&mut temporaries,
			)?;
			stages.push(muon_mat_mul_nt_stage(
				b,
				z_transpose,
				bz,
				operation_rows,
				operation_columns,
				operation_rows,
			));
			stages.push(muon_linear_stage(z, bz, next_z, count, NS_A, 1.0));
		}
		z = next_z;
	}
	let orthogonal = if transposed {
		let output = allocate_muon_temp(parameter, &[rows, columns], &mut temporaries)?;
		stages.push(muon_transpose_stage(
			z,
			output,
			operation_rows,
			operation_columns,
		));
		output
	} else {
		z
	};
	stages.push(MuonStage {
		kernel: KernelId::MlMuonApplyF32,
		bindings: vec![
			(MuonSlot::Parameter, BufferAccess::ReadWrite),
			(orthogonal, BufferAccess::Read),
		],
		push_constants: vec![
			PushConstant::U32(count),
			PushConstant::F32(scalars.learning_rate),
			PushConstant::F32(scalars.weight_decay),
			PushConstant::F32(0.2 * (rows.max(columns) as f32).sqrt()),
		],
		workgroups: KernelId::MlMuonApplyF32.linear_workgroups(count),
	});
	let binding_sets = stages
		.iter()
		.map(|stage| {
			stage
				.bindings
				.iter()
				.map(|(slot, access)| {
					let matrix = match slot {
						MuonSlot::Parameter => parameter,
						MuonSlot::Gradient => gradient,
						MuonSlot::Momentum => momentum,
						MuonSlot::Temporary(index) => &temporaries[*index],
					};
					BufferBinding {
						storage: matrix.storage(),
						access: *access,
					}
				})
				.collect::<Vec<_>>()
		})
		.collect::<Vec<_>>();
	let dispatches = stages
		.iter()
		.zip(&binding_sets)
		.map(|(stage, bindings)| ComputeDispatch {
			kernel: stage.kernel,
			buffers: bindings,
			push_constants: &stage.push_constants,
			workgroups: stage.workgroups,
		})
		.collect::<Vec<_>>();
	let inputs = [parameter, gradient, momentum];
	let outputs = [parameter, momentum];
	parameter.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::ml::MUON,
			inputs: &inputs,
			outputs: &outputs,
			attributes,
		},
	)
}

fn allocate_muon_temp(
	parameter: &Matrix,
	shape: &[u32],
	temporaries: &mut Vec<Matrix>,
) -> Result<MuonSlot> {
	let shape = shape
		.iter()
		.map(|value| *value as usize)
		.collect::<Vec<_>>();
	let count = shape.iter().try_fold(1_usize, |count, extent| {
		count
			.checked_mul(*extent)
			.ok_or_else(|| Error::invalid_argument("ml.muon temporary size overflows usize"))
	})?;
	temporaries.push(Matrix::allocate(
		parameter.engine_handle(),
		shape,
		count,
		DType::F32,
	)?);
	Ok(MuonSlot::Temporary(temporaries.len() - 1))
}

fn muon_attributes(scalars: MuonScalars) -> Vec<OpAttribute> {
	vec![
		OpAttribute::Float {
			name: "learning_rate".into(),
			value: f64::from(scalars.learning_rate),
		},
		OpAttribute::Float {
			name: "beta".into(),
			value: f64::from(scalars.beta),
		},
		OpAttribute::Float {
			name: "weight_decay".into(),
			value: f64::from(scalars.weight_decay),
		},
		OpAttribute::Float {
			name: "epsilon".into(),
			value: f64::from(scalars.epsilon),
		},
		OpAttribute::UnsignedInteger {
			name: "ns5_iterations".into(),
			value: u64::from(scalars.ns5_iterations),
		},
	]
}

fn muon_transpose_stage(input: MuonSlot, output: MuonSlot, rows: u32, columns: u32) -> MuonStage {
	let count = rows * columns;
	MuonStage {
		kernel: KernelId::MlMuonTransposeF32,
		bindings: vec![(input, BufferAccess::Read), (output, BufferAccess::Write)],
		push_constants: vec![PushConstant::U32(rows), PushConstant::U32(columns)],
		workgroups: KernelId::MlMuonTransposeF32.linear_workgroups(count),
	}
}

fn muon_mat_mul_nt_stage(
	left: MuonSlot,
	right: MuonSlot,
	output: MuonSlot,
	m: u32,
	n: u32,
	k: u32,
) -> MuonStage {
	MuonStage {
		kernel: KernelId::MatrixMatMulNtTiledF32,
		bindings: vec![
			(left, BufferAccess::Read),
			(right, BufferAccess::Read),
			(output, BufferAccess::Write),
		],
		push_constants: vec![
			PushConstant::U32(m),
			PushConstant::U32(n),
			PushConstant::U32(k),
		],
		workgroups: KernelId::MatrixMatMulNtTiledF32.output_workgroups(m, n),
	}
}

#[allow(clippy::too_many_arguments)]
fn muon_mat_mul_axpby_stage(
	a: MuonSlot,
	b: MuonSlot,
	residual: MuonSlot,
	output: MuonSlot,
	m: u32,
	n: u32,
	k: u32,
	alpha: f32,
	beta: f32,
) -> MuonStage {
	MuonStage {
		kernel: KernelId::MlMuonMatMulAxpbyF32,
		bindings: vec![
			(a, BufferAccess::Read),
			(b, BufferAccess::Read),
			(residual, BufferAccess::Read),
			(output, BufferAccess::Write),
		],
		push_constants: vec![
			PushConstant::U32(m),
			PushConstant::U32(n),
			PushConstant::U32(k),
			PushConstant::F32(alpha),
			PushConstant::F32(beta),
		],
		workgroups: KernelId::MlMuonMatMulAxpbyF32.output_workgroups(m, n),
	}
}

fn muon_linear_stage(
	a: MuonSlot,
	b: MuonSlot,
	output: MuonSlot,
	count: u32,
	scale_a: f32,
	scale_b: f32,
) -> MuonStage {
	MuonStage {
		kernel: KernelId::MlMuonLinearCombinationF32,
		bindings: vec![
			(a, BufferAccess::Read),
			(b, BufferAccess::Read),
			(output, BufferAccess::Write),
		],
		push_constants: vec![
			PushConstant::U32(count),
			PushConstant::F32(scale_a),
			PushConstant::F32(scale_b),
		],
		workgroups: KernelId::MlMuonLinearCombinationF32.linear_workgroups(count),
	}
}

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
			muon(parameter, gradient, &state.momentum, scalars)?;
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
