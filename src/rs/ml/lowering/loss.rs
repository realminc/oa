//! Private lowering for ML loss operations.

use crate::{
	DType, Error, Matrix, OpAttribute, OperationContract, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

use super::common::{record_semantic, shader_u32};

pub(in crate::ml) fn ppo_clipped_policy(
	new_log_probability: &Matrix,
	old_log_probability: &Matrix,
	advantage: &Matrix,
	clip_epsilon: f32,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::PPO_CLIPPED_POLICY;
	let count = validate_ppo_inputs(
		new_log_probability,
		old_log_probability,
		advantage,
		clip_epsilon,
		contract.name(),
	)?;
	let per_sample = Matrix::allocate(
		new_log_probability.engine_handle(),
		new_log_probability.shape().to_vec(),
		new_log_probability.num_elements(),
		DType::F32,
	)?;
	let sum = Matrix::allocate(
		new_log_probability.engine_handle(),
		Vec::new(),
		1,
		DType::F32,
	)?;
	let output = Matrix::allocate(
		new_log_probability.engine_handle(),
		Vec::new(),
		1,
		DType::F32,
	)?;
	let policy_buffers = [
		BufferBinding::read(new_log_probability.storage()),
		BufferBinding::read(old_log_probability.storage()),
		BufferBinding::read(advantage.storage()),
		BufferBinding::write(per_sample.storage()),
	];
	let policy_push = [PushConstant::U32(count), PushConstant::F32(clip_epsilon)];
	let sum_buffers = [
		BufferBinding::read(per_sample.storage()),
		BufferBinding::write(sum.storage()),
	];
	let sum_push = [PushConstant::U32(count)];
	let mean_buffers = [
		BufferBinding::read(sum.storage()),
		BufferBinding::write(output.storage()),
	];
	let mean_push = [PushConstant::U32(1), PushConstant::F32(1.0 / count as f32)];
	let kernel = KernelId::MlPpoClippedPolicyF32;
	let dispatches = [
		ComputeDispatch {
			kernel,
			buffers: &policy_buffers,
			push_constants: &policy_push,
			workgroups: kernel.linear_workgroups(count),
		},
		ComputeDispatch {
			kernel: KernelId::MatrixSumF32,
			buffers: &sum_buffers,
			push_constants: &sum_push,
			workgroups: [1, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MatrixScaleF32,
			buffers: &mean_buffers,
			push_constants: &mean_push,
			workgroups: [1, 1, 1],
		},
	];
	let attributes = [OpAttribute::Float {
		name: "clip_epsilon".into(),
		value: f64::from(clip_epsilon),
	}];
	new_log_probability.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &[new_log_probability, old_log_probability, advantage],
			outputs: &[&output],
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

pub(in crate::ml) fn ppo_clipped_policy_backward(
	new_log_probability: &Matrix,
	old_log_probability: &Matrix,
	advantage: &Matrix,
	clip_epsilon: f32,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::PPO_CLIPPED_POLICY_BACKWARD;
	let count = validate_ppo_inputs(
		new_log_probability,
		old_log_probability,
		advantage,
		clip_epsilon,
		contract.name(),
	)?;
	let gradient = Matrix::allocate(
		new_log_probability.engine_handle(),
		new_log_probability.shape().to_vec(),
		new_log_probability.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(new_log_probability.storage()),
		BufferBinding::read(old_log_probability.storage()),
		BufferBinding::read(advantage.storage()),
		BufferBinding::write(gradient.storage()),
	];
	let push_constants = [PushConstant::U32(count), PushConstant::F32(clip_epsilon)];
	let attributes = [OpAttribute::Float {
		name: "clip_epsilon".into(),
		value: f64::from(clip_epsilon),
	}];
	let kernel = KernelId::MlPpoClippedPolicyBackwardF32;
	record_semantic(
		&[new_log_probability, old_log_probability, advantage],
		&[&gradient],
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(count),
	)?;
	Ok(gradient)
}

#[allow(
	clippy::too_many_arguments,
	reason = "the private DQN target lowering retains all four donor fields and its shape ABI"
)]
pub(in crate::ml) fn dqn_target(
	lowering: &crate::runtime::SemanticLoweringScope,
	reward: &Matrix,
	next_q: &Matrix,
	terminated: &Matrix,
	truncated: &Matrix,
	batch: u32,
	actions: u32,
	discount: f32,
) -> Result<Matrix> {
	let output = Matrix::allocate(
		reward.engine_handle(),
		reward.shape().to_vec(),
		reward.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(reward.storage()),
		BufferBinding::read(next_q.storage()),
		BufferBinding::read(terminated.storage()),
		BufferBinding::read(truncated.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(batch),
		PushConstant::U32(actions),
		PushConstant::F32(discount),
	];
	let kernel = KernelId::MlDqnTargetF32;
	lowering.record_physical(ComputeDispatch {
		kernel,
		buffers: &buffers,
		push_constants: &push_constants,
		workgroups: kernel.linear_workgroups(batch),
	})?;
	Ok(output)
}

#[allow(
	clippy::too_many_arguments,
	reason = "the private SAC target lowering retains all six donor fields and its scalar ABI"
)]
pub(in crate::ml) fn sac_target(
	lowering: &crate::runtime::SemanticLoweringScope,
	reward: &Matrix,
	next_q1: &Matrix,
	next_q2: &Matrix,
	next_log_probability: &Matrix,
	terminated: &Matrix,
	truncated: &Matrix,
	batch: u32,
	discount: f32,
	entropy_coefficient: f32,
) -> Result<Matrix> {
	let output = Matrix::allocate(
		reward.engine_handle(),
		reward.shape().to_vec(),
		reward.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(reward.storage()),
		BufferBinding::read(next_q1.storage()),
		BufferBinding::read(next_q2.storage()),
		BufferBinding::read(next_log_probability.storage()),
		BufferBinding::read(terminated.storage()),
		BufferBinding::read(truncated.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(batch),
		PushConstant::F32(discount),
		PushConstant::F32(entropy_coefficient),
	];
	let kernel = KernelId::MlSacTargetF32;
	lowering.record_physical(ComputeDispatch {
		kernel,
		buffers: &buffers,
		push_constants: &push_constants,
		workgroups: kernel.linear_workgroups(batch),
	})?;
	Ok(output)
}

fn validate_ppo_inputs(
	new_log_probability: &Matrix,
	old_log_probability: &Matrix,
	advantage: &Matrix,
	clip_epsilon: f32,
	operation: &'static str,
) -> Result<u32> {
	for input in [old_log_probability, advantage] {
		if input.shape() != new_log_probability.shape() || input.dtype() != DType::F32 {
			return Err(Error::invalid_argument(format!(
				"{operation} requires matching F32 inputs"
			)));
		}
		if !new_log_probability
			.engine_handle()
			.same_as(input.engine_handle())
		{
			return Err(Error::invalid_argument(format!(
				"{operation} inputs must belong to the same engine"
			)));
		}
	}
	if new_log_probability.dtype() != DType::F32 || new_log_probability.num_elements() == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires nonempty F32 inputs"
		)));
	}
	if !clip_epsilon.is_finite() || !(0.0..1.0).contains(&clip_epsilon) {
		return Err(Error::invalid_argument(format!(
			"{operation} requires clip epsilon in (0, 1)"
		)));
	}
	shader_u32(
		new_log_probability.num_elements(),
		"element count",
		operation,
	)
}

pub(in crate::ml) fn smooth_l1(prediction: &Matrix, target: &Matrix) -> Result<Matrix> {
	let operation = crate::core::operation::ml::SMOOTH_L1.name();
	let element_count = validate_pointwise_loss_inputs(prediction, target, operation)?;
	let output = Matrix::allocate(prediction.engine_handle(), Vec::new(), 1, DType::F32)?;
	let buffers = [
		BufferBinding::read(prediction.storage()),
		BufferBinding::read(target.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [PushConstant::U32(element_count)];
	let kernel = KernelId::MlSmoothL1MeanF32;
	record_semantic(
		&[prediction, target],
		&[&output],
		&[],
		kernel,
		&buffers,
		&push_constants,
		[1, 1, 1],
	)?;
	Ok(output)
}

pub(in crate::ml) fn smooth_l1_backward(prediction: &Matrix, target: &Matrix) -> Result<Matrix> {
	pointwise_loss_backward(
		prediction,
		target,
		KernelId::MlSmoothL1BackwardF32,
		crate::core::operation::ml::SMOOTH_L1_BACKWARD,
	)
}

pub(in crate::ml) fn mse(prediction: &Matrix, target: &Matrix) -> Result<Matrix> {
	mean_pointwise_loss(
		prediction,
		target,
		KernelId::MlMseF32,
		crate::core::operation::ml::MSE,
	)
}

pub(in crate::ml) fn mse_backward(prediction: &Matrix, target: &Matrix) -> Result<Matrix> {
	pointwise_loss_backward(
		prediction,
		target,
		KernelId::MlMseBackwardF32,
		crate::core::operation::ml::MSE_BACKWARD,
	)
}

pub(in crate::ml) fn l1(prediction: &Matrix, target: &Matrix) -> Result<Matrix> {
	mean_pointwise_loss(
		prediction,
		target,
		KernelId::MlL1F32,
		crate::core::operation::ml::L1,
	)
}

pub(in crate::ml) fn l1_backward(prediction: &Matrix, target: &Matrix) -> Result<Matrix> {
	pointwise_loss_backward(
		prediction,
		target,
		KernelId::MlL1BackwardF32,
		crate::core::operation::ml::L1_BACKWARD,
	)
}

pub(in crate::ml) fn bce(prediction: &Matrix, target: &Matrix) -> Result<Matrix> {
	mean_pointwise_loss(
		prediction,
		target,
		KernelId::MlBceF32,
		crate::core::operation::ml::BCE,
	)
}

pub(in crate::ml) fn bce_backward(prediction: &Matrix, target: &Matrix) -> Result<Matrix> {
	pointwise_loss_backward(
		prediction,
		target,
		KernelId::MlBceBackwardF32,
		crate::core::operation::ml::BCE_BACKWARD,
	)
}

fn mean_pointwise_loss(
	prediction: &Matrix,
	target: &Matrix,
	kernel: KernelId,
	contract: OperationContract,
) -> Result<Matrix> {
	let element_count = validate_pointwise_loss_inputs(prediction, target, contract.name())?;
	let per_element = Matrix::allocate(
		prediction.engine_handle(),
		prediction.shape().to_vec(),
		prediction.num_elements(),
		DType::F32,
	)?;
	let sum = Matrix::allocate(prediction.engine_handle(), Vec::new(), 1, DType::F32)?;
	let output = Matrix::allocate(prediction.engine_handle(), Vec::new(), 1, DType::F32)?;
	let loss_buffers = [
		BufferBinding::read(prediction.storage()),
		BufferBinding::read(target.storage()),
		BufferBinding::write(per_element.storage()),
	];
	let loss_push_constants = [PushConstant::U32(element_count)];
	let sum_buffers = [
		BufferBinding::read(per_element.storage()),
		BufferBinding::write(sum.storage()),
	];
	let sum_push_constants = [PushConstant::U32(element_count)];
	let mean_buffers = [
		BufferBinding::read(sum.storage()),
		BufferBinding::write(output.storage()),
	];
	let mean_push_constants = [
		PushConstant::U32(1),
		PushConstant::F32(1.0 / element_count as f32),
	];
	let dispatches = [
		ComputeDispatch {
			kernel,
			buffers: &loss_buffers,
			push_constants: &loss_push_constants,
			workgroups: kernel.linear_workgroups(element_count),
		},
		ComputeDispatch {
			kernel: KernelId::MatrixSumF32,
			buffers: &sum_buffers,
			push_constants: &sum_push_constants,
			workgroups: [1, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MatrixScaleF32,
			buffers: &mean_buffers,
			push_constants: &mean_push_constants,
			workgroups: [1, 1, 1],
		},
	];
	let inputs = [prediction, target];
	let outputs = [&output];
	prediction.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &[],
		},
	)?;
	Ok(output)
}

fn pointwise_loss_backward(
	prediction: &Matrix,
	target: &Matrix,
	kernel: KernelId,
	contract: OperationContract,
) -> Result<Matrix> {
	let element_count = validate_pointwise_loss_inputs(prediction, target, contract.name())?;
	let gradient = Matrix::allocate(
		prediction.engine_handle(),
		prediction.shape().to_vec(),
		prediction.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(prediction.storage()),
		BufferBinding::read(target.storage()),
		BufferBinding::write(gradient.storage()),
	];
	let push_constants = [PushConstant::U32(element_count)];
	record_semantic(
		&[prediction, target],
		&[&gradient],
		&[],
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(element_count),
	)?;
	Ok(gradient)
}

fn validate_pointwise_loss_inputs(
	prediction: &Matrix,
	target: &Matrix,
	operation: &'static str,
) -> Result<u32> {
	if prediction.shape() != target.shape() || prediction.num_elements() == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires matching nonempty prediction and target shapes; found {:?} and {:?}",
			prediction.shape(),
			target.shape()
		)));
	}
	if prediction.dtype() != DType::F32 || target.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires two F32 matrices; found {} and {}",
			prediction.dtype().token(),
			target.dtype().token()
		)));
	}
	if !prediction.engine_handle().same_as(target.engine_handle()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same engine"
		)));
	}
	shader_u32(prediction.num_elements(), "element count", operation)
}

pub(in crate::ml) fn cross_entropy(logits: &Matrix, targets: &Matrix) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::CROSS_ENTROPY.name();
	let (rows, classes) = validate_cross_entropy_inputs(logits, targets, OPERATION)?;
	let per_row = Matrix::allocate(
		logits.engine_handle(),
		vec![rows as usize],
		rows as usize,
		DType::F32,
	)?;
	let sum = Matrix::allocate(logits.engine_handle(), Vec::new(), 1, DType::F32)?;
	let output = Matrix::allocate(logits.engine_handle(), Vec::new(), 1, DType::F32)?;
	let loss_buffers = [
		BufferBinding::read(logits.storage()),
		BufferBinding::read(targets.storage()),
		BufferBinding::write(per_row.storage()),
	];
	let target_dtype = target_dtype(targets.dtype());
	let loss_push_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(classes),
		PushConstant::U32(target_dtype),
	];
	let sum_buffers = [
		BufferBinding::read(per_row.storage()),
		BufferBinding::write(sum.storage()),
	];
	let sum_push_constants = [PushConstant::U32(rows)];
	let mean_buffers = [
		BufferBinding::read(sum.storage()),
		BufferBinding::write(output.storage()),
	];
	let mean_push_constants = [PushConstant::U32(1), PushConstant::F32(1.0 / rows as f32)];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlCrossEntropyF32,
			buffers: &loss_buffers,
			push_constants: &loss_push_constants,
			workgroups: portable_row_workgroups(rows),
		},
		ComputeDispatch {
			kernel: KernelId::MlCrossEntropySumF32,
			buffers: &sum_buffers,
			push_constants: &sum_push_constants,
			workgroups: [1, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MatrixScaleF32,
			buffers: &mean_buffers,
			push_constants: &mean_push_constants,
			workgroups: [1, 1, 1],
		},
	];
	let inputs = [logits, targets];
	let outputs = [&output];
	logits.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::ml::CROSS_ENTROPY,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &[],
		},
	)?;
	Ok(output)
}

pub(in crate::ml) fn cross_entropy_backward(logits: &Matrix, targets: &Matrix) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::CROSS_ENTROPY_BACKWARD.name();
	let (rows, classes) = validate_cross_entropy_inputs(logits, targets, OPERATION)?;
	let gradient = Matrix::allocate(
		logits.engine_handle(),
		logits.shape().to_vec(),
		logits.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(logits.storage()),
		BufferBinding::read(targets.storage()),
		BufferBinding::write(gradient.storage()),
	];
	let push_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(classes),
		PushConstant::U32(target_dtype(targets.dtype())),
	];
	let kernel = KernelId::MlCrossEntropyBackwardF32;
	record_semantic(
		&[logits, targets],
		&[&gradient],
		&[],
		kernel,
		&buffers,
		&push_constants,
		portable_row_workgroups(rows),
	)?;
	Ok(gradient)
}

pub(in crate::ml) fn masked_cross_entropy(
	logits: &Matrix,
	targets: &Matrix,
	mask: &Matrix,
	valid_count: usize,
) -> Result<Matrix> {
	const CONTRACT: OperationContract = crate::core::operation::ml::MASKED_CROSS_ENTROPY;
	let (rows, classes, valid_count) =
		validate_masked_cross_entropy_inputs(logits, targets, mask, valid_count, CONTRACT.name())?;
	let per_row = Matrix::allocate(
		logits.engine_handle(),
		vec![rows as usize],
		rows as usize,
		DType::F32,
	)?;
	let sum = Matrix::allocate(logits.engine_handle(), Vec::new(), 1, DType::F32)?;
	let output = Matrix::allocate(logits.engine_handle(), Vec::new(), 1, DType::F32)?;
	let loss_buffers = [
		BufferBinding::read(logits.storage()),
		BufferBinding::read(targets.storage()),
		BufferBinding::read(mask.storage()),
		BufferBinding::write(per_row.storage()),
	];
	let loss_push_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(classes),
		PushConstant::U32(target_dtype(targets.dtype())),
	];
	let sum_buffers = [
		BufferBinding::read(per_row.storage()),
		BufferBinding::write(sum.storage()),
	];
	let sum_push_constants = [PushConstant::U32(rows)];
	let mean_buffers = [
		BufferBinding::read(sum.storage()),
		BufferBinding::write(output.storage()),
	];
	let mean_push_constants = [
		PushConstant::U32(1),
		PushConstant::F32(1.0 / valid_count as f32),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlMaskedCrossEntropyF32,
			buffers: &loss_buffers,
			push_constants: &loss_push_constants,
			workgroups: portable_row_workgroups(rows),
		},
		ComputeDispatch {
			kernel: KernelId::MatrixSumF32,
			buffers: &sum_buffers,
			push_constants: &sum_push_constants,
			workgroups: [1, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MatrixScaleF32,
			buffers: &mean_buffers,
			push_constants: &mean_push_constants,
			workgroups: [1, 1, 1],
		},
	];
	let attributes = [valid_count_attribute(valid_count)];
	let inputs = [logits, targets, mask];
	let outputs = [&output];
	logits.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: CONTRACT,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

pub(in crate::ml) fn masked_cross_entropy_backward(
	logits: &Matrix,
	targets: &Matrix,
	mask: &Matrix,
	valid_count: usize,
) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::MASKED_CROSS_ENTROPY_BACKWARD.name();
	let (rows, classes, valid_count) =
		validate_masked_cross_entropy_inputs(logits, targets, mask, valid_count, OPERATION)?;
	let gradient = Matrix::allocate(
		logits.engine_handle(),
		logits.shape().to_vec(),
		logits.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(logits.storage()),
		BufferBinding::read(targets.storage()),
		BufferBinding::read(mask.storage()),
		BufferBinding::write(gradient.storage()),
	];
	let push_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(classes),
		PushConstant::U32(target_dtype(targets.dtype())),
		PushConstant::U32(valid_count),
	];
	let attributes = [valid_count_attribute(valid_count)];
	let kernel = KernelId::MlMaskedCrossEntropyBackwardF32;
	record_semantic(
		&[logits, targets, mask],
		&[&gradient],
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		portable_row_workgroups(rows),
	)?;
	Ok(gradient)
}

fn validate_masked_cross_entropy_inputs(
	logits: &Matrix,
	targets: &Matrix,
	mask: &Matrix,
	valid_count: usize,
	operation: &'static str,
) -> Result<(u32, u32, u32)> {
	let (rows, classes) = validate_cross_entropy_inputs(logits, targets, operation)?;
	if mask.shape() != [rows as usize] || mask.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires an F32 mask [N]; found shape {:?} and dtype {}",
			mask.shape(),
			mask.dtype().token()
		)));
	}
	if !logits.engine_handle().same_as(mask.engine_handle()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same engine"
		)));
	}
	let valid_count_u32 = shader_u32(valid_count, "valid count", operation)?;
	if valid_count_u32 == 0 || valid_count_u32 > rows {
		return Err(Error::invalid_argument(format!(
			"{operation} valid count must be in 1..={rows}; found {valid_count}"
		)));
	}
	Ok((rows, classes, valid_count_u32))
}

fn valid_count_attribute(valid_count: u32) -> OpAttribute {
	OpAttribute::SignedInteger {
		name: "valid_count".into(),
		value: i64::from(valid_count),
	}
}

fn validate_cross_entropy_inputs(
	logits: &Matrix,
	targets: &Matrix,
	operation: &'static str,
) -> Result<(u32, u32)> {
	let [rows, classes] = logits.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} logits must have rank two; found {:?}",
			logits.shape()
		)));
	};
	if *rows == 0 || *classes == 0 || targets.shape() != [*rows] {
		return Err(Error::invalid_argument(format!(
			"{operation} requires nonempty logits [N, C] and targets [N]; found {:?} and {:?}",
			logits.shape(),
			targets.shape()
		)));
	}
	if logits.dtype() != DType::F32 || !matches!(targets.dtype(), DType::U32 | DType::I32) {
		return Err(Error::invalid_argument(format!(
			"{operation} requires F32 logits and U32 or non-negative I32 targets; found {} and {}",
			logits.dtype().token(),
			targets.dtype().token()
		)));
	}
	if !logits.engine_handle().same_as(targets.engine_handle()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same engine"
		)));
	}
	Ok((
		shader_u32(*rows, "row count", operation)?,
		shader_u32(*classes, "class count", operation)?,
	))
}

fn target_dtype(dtype: DType) -> u32 {
	debug_assert!(matches!(dtype, DType::U32 | DType::I32));
	1
}

fn portable_row_workgroups(rows: u32) -> [u32; 3] {
	const PORTABLE_WIDTH: u32 = 65_535;
	[rows.min(PORTABLE_WIDTH), rows.div_ceil(PORTABLE_WIDTH), 1]
}
