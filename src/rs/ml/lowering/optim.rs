//! Private lowering for ML optimizer operations.

use crate::{
	DType, Error, Matrix, OpAttribute, Result,
	runtime::{
		BufferAccess, BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch,
	},
};

use super::common::{record_semantic, shader_u32, validate_f32_same_engine};

pub(in crate::ml) fn clip_grad_norm(gradients: &[&Matrix], max_norm: f32) -> Result<()> {
	const OPERATION: &str = crate::core::operation::ml::CLIP_GRAD_NORM.name();
	if !max_norm.is_finite() || max_norm < 0.0 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} max_norm must be finite and non-negative"
		)));
	}
	let gradients = gradients
		.iter()
		.copied()
		.filter(|gradient| gradient.num_elements() != 0)
		.collect::<Vec<_>>();
	if gradients.is_empty() {
		return Ok(());
	}
	if gradients.len() > 16 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} supports at most 16 nonempty gradients"
		)));
	}
	validate_f32_same_engine(OPERATION, &gradients)?;
	for (index, gradient) in gradients.iter().enumerate() {
		if gradients[..index]
			.iter()
			.any(|previous| previous.storage().same_as(gradient.storage()))
		{
			return Err(Error::invalid_argument(format!(
				"{OPERATION} gradients must not share storage"
			)));
		}
	}

	let mut params = [0_u32; 18];
	let gradient_count = u32::try_from(gradients.len())
		.map_err(|_| Error::invalid_argument(format!("{OPERATION} gradient count exceeds u32")))?;
	params[0] = gradient_count;
	for (index, gradient) in gradients.iter().enumerate() {
		params[index + 1] = shader_u32(gradient.num_elements(), "element count", OPERATION)?;
	}
	params[17] = max_norm.to_bits();

	let engine = gradients[0].engine_handle();
	let params = Matrix::from_slice_handle(engine, vec![18], &params)?;
	let partials = Matrix::allocate(engine, vec![16], 16, DType::F32)?;
	let mut reduce_buffers = Vec::with_capacity(18);
	reduce_buffers.push(BufferBinding::read(params.storage()));
	reduce_buffers.push(BufferBinding::write(partials.storage()));
	for index in 0..16 {
		reduce_buffers.push(BufferBinding::read(
			gradients
				.get(index)
				.copied()
				.unwrap_or(gradients[0])
				.storage(),
		));
	}
	let mut scale_buffers = Vec::with_capacity(18);
	scale_buffers.push(BufferBinding::read(params.storage()));
	scale_buffers.push(BufferBinding::read(partials.storage()));
	for index in 0..16 {
		let binding = gradients.get(index).copied().map_or_else(
			|| BufferBinding::read(gradients[0].storage()),
			|gradient| BufferBinding::read_write(gradient.storage()),
		);
		scale_buffers.push(binding);
	}

	let reduce = KernelId::MlClipGradNormReduceF32;
	let scale = KernelId::MlClipGradNormScaleF32;
	let dispatches = [
		ComputeDispatch {
			kernel: reduce,
			buffers: &reduce_buffers,
			push_constants: &[],
			workgroups: [gradient_count, 1, 1],
		},
		ComputeDispatch {
			kernel: scale,
			buffers: &scale_buffers,
			push_constants: &[],
			workgroups: [gradient_count, 1, 1],
		},
	];
	let attributes = [OpAttribute::Float {
		name: "max_norm".into(),
		value: f64::from(max_norm),
	}];
	engine.record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::ml::CLIP_GRAD_NORM,
			inputs: &gradients,
			outputs: &gradients,
			attributes: &attributes,
		},
	)
}

#[derive(Clone, Copy)]
pub(in crate::ml) struct AdamWScalars {
	pub(in crate::ml) step: u32,
	pub(in crate::ml) learning_rate: f32,
	pub(in crate::ml) beta1: f32,
	pub(in crate::ml) beta2: f32,
	pub(in crate::ml) epsilon: f32,
	pub(in crate::ml) weight_decay: f32,
}

#[derive(Clone, Copy)]
pub(in crate::ml) struct AdamWBatchEntry<'a> {
	pub(in crate::ml) parameter: &'a Matrix,
	pub(in crate::ml) gradient: &'a Matrix,
	pub(in crate::ml) first_moment: &'a Matrix,
	pub(in crate::ml) second_moment: &'a Matrix,
}

#[derive(Clone, Copy)]
pub(in crate::ml) struct SgdScalars {
	pub(in crate::ml) learning_rate: f32,
	pub(in crate::ml) momentum: f32,
	pub(in crate::ml) weight_decay: f32,
}

#[derive(Clone, Copy)]
pub(in crate::ml) struct AdamScalars {
	pub(in crate::ml) step: u32,
	pub(in crate::ml) learning_rate: f32,
	pub(in crate::ml) beta1: f32,
	pub(in crate::ml) beta2: f32,
	pub(in crate::ml) epsilon: f32,
}

#[derive(Clone, Copy)]
pub(in crate::ml) struct MuonScalars {
	pub(in crate::ml) learning_rate: f32,
	pub(in crate::ml) beta: f32,
	pub(in crate::ml) weight_decay: f32,
	pub(in crate::ml) epsilon: f32,
	pub(in crate::ml) ns5_iterations: u32,
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

pub(in crate::ml) fn muon(
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
		return record_semantic(
			&[parameter, gradient, momentum],
			&[parameter, momentum],
			&attributes,
			kernel,
			&buffers,
			&push_constants,
			kernel.linear_workgroups(element_count),
		);
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

pub(in crate::ml) fn sgd(parameter: &Matrix, gradient: &Matrix, scalars: SgdScalars) -> Result<()> {
	const OPERATION: &str = crate::core::operation::ml::SGD.name();
	validate_f32_same_engine(OPERATION, &[parameter, gradient])?;
	if gradient.shape() != parameter.shape() {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires identical parameter and gradient shapes"
		)));
	}
	let element_count = shader_u32(parameter.num_elements(), "element count", OPERATION)?;
	if element_count == 0 {
		return Ok(());
	}
	let buffers = [
		BufferBinding::read_write(parameter.storage()),
		BufferBinding::read(gradient.storage()),
	];
	let push_constants = [
		PushConstant::U32(element_count),
		PushConstant::F32(scalars.learning_rate),
		PushConstant::F32(scalars.weight_decay),
	];
	let attributes = [
		OpAttribute::Float {
			name: "learning_rate".into(),
			value: f64::from(scalars.learning_rate),
		},
		OpAttribute::Float {
			name: "weight_decay".into(),
			value: f64::from(scalars.weight_decay),
		},
	];
	let kernel = KernelId::MlSgdF32;
	record_semantic(
		&[parameter, gradient],
		&[parameter],
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(element_count),
	)
}

pub(in crate::ml) fn sgd_momentum(
	parameter: &Matrix,
	gradient: &Matrix,
	momentum: &Matrix,
	scalars: SgdScalars,
) -> Result<()> {
	const OPERATION: &str = crate::core::operation::ml::SGD_MOMENTUM.name();
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
	let buffers = [
		BufferBinding::read_write(parameter.storage()),
		BufferBinding::read(gradient.storage()),
		BufferBinding::read_write(momentum.storage()),
	];
	let push_constants = [
		PushConstant::U32(element_count),
		PushConstant::F32(scalars.learning_rate),
		PushConstant::F32(scalars.momentum),
		PushConstant::F32(scalars.weight_decay),
	];
	let attributes = [
		OpAttribute::Float {
			name: "learning_rate".into(),
			value: f64::from(scalars.learning_rate),
		},
		OpAttribute::Float {
			name: "momentum".into(),
			value: f64::from(scalars.momentum),
		},
		OpAttribute::Float {
			name: "weight_decay".into(),
			value: f64::from(scalars.weight_decay),
		},
	];
	let kernel = KernelId::MlSgdMomentumF32;
	record_semantic(
		&[parameter, gradient, momentum],
		&[parameter, momentum],
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(element_count),
	)
}

pub(in crate::ml) fn adam(
	parameter: &Matrix,
	gradient: &Matrix,
	first_moment: &Matrix,
	second_moment: &Matrix,
	scalars: AdamScalars,
) -> Result<()> {
	const OPERATION: &str = crate::core::operation::ml::ADAM.name();
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
	if element_count == 0 {
		return Ok(());
	}
	let buffers = [
		BufferBinding::read_write(parameter.storage()),
		BufferBinding::read(gradient.storage()),
		BufferBinding::read_write(first_moment.storage()),
		BufferBinding::read_write(second_moment.storage()),
	];
	let push_constants = [
		PushConstant::U32(element_count),
		PushConstant::F32(scalars.learning_rate),
		PushConstant::F32(scalars.beta1),
		PushConstant::F32(scalars.beta2),
		PushConstant::F32(scalars.epsilon),
		PushConstant::U32(scalars.step),
	];
	let attributes = [
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
	];
	let kernel = KernelId::MlAdamF32;
	record_semantic(
		&[parameter, gradient, first_moment, second_moment],
		&[parameter, first_moment, second_moment],
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(element_count),
	)
}

pub(in crate::ml) fn adamw_graph_advance(state: &Matrix) -> Result<()> {
	const OPERATION: &str = crate::core::operation::ml::ADAMW_GRAPH_ADVANCE.name();
	if state.dtype() != DType::U32 || state.shape() != [6] {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires U32 optimizer state [6]"
		)));
	}
	let buffers = [BufferBinding::read_write(state.storage())];
	record_semantic(
		&[state],
		&[state],
		&[],
		KernelId::MlAdamWGraphAdvanceU32,
		&buffers,
		&[],
		[1, 1, 1],
	)
}

pub(in crate::ml) fn adamw_graph(
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
	record_semantic(
		&[parameter, gradient, first_moment, second_moment, state],
		&[parameter, first_moment, second_moment],
		&[],
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(element_count),
	)
}

pub(in crate::ml) fn adamw_many4_graph(
	entries: &[AdamWBatchEntry<'_>; 4],
	state: &Matrix,
) -> Result<()> {
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

pub(in crate::ml) fn adamw(
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
		let attributes = [
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
		];
		record_semantic(
			&[parameter, gradient, first_moment, second_moment],
			&[parameter, first_moment, second_moment],
			&attributes,
			kernel,
			&buffers,
			&push_constants,
			kernel.linear_workgroups(element_count),
		)?;
	}
	Ok(())
}

pub(in crate::ml) fn adamw_many4(
	entries: &[AdamWBatchEntry<'_>; 4],
	scalars: AdamWScalars,
) -> Result<()> {
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
