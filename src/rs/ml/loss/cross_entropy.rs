//! Cross-entropy and masked cross-entropy losses.

use crate::OpAttribute;
use crate::ml::autograd;
use crate::runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch};
use crate::{DType, Error, Matrix, OperationContract, Result};

use super::common::{portable_row_workgroups, shader_u32, target_dtype, valid_count_attribute};

/// Compute mean cross-entropy over rank-two FP32 logits and U32 or I32 class targets.
///
/// `logits` has shape `[N, C]`, `targets` has shape `[N]`, and the returned
/// matrix is an FP32 scalar. I32 targets must be non-negative; any negative or
/// out-of-range target produces NaN without an out-of-bounds access.
///
/// # Errors
///
/// Returns an error when ranks, shapes, dtypes, ownership, allocation, or
/// runtime recording are invalid.
pub fn cross_entropy(logits: &Matrix, targets: &Matrix) -> Result<Matrix> {
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
	let target_dtype_u32 = target_dtype(targets.dtype());
	let loss_push_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(classes),
		PushConstant::U32(target_dtype_u32),
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
	logits.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::ml::CROSS_ENTROPY,
			inputs: &[logits, targets],
			outputs: &[&output],
			attributes: &[],
		},
	)?;
	autograd::record_cross_entropy(logits, targets, &output)?;
	Ok(output)
}

/// Compute mean cross-entropy over the rows selected by a floating FP32 mask.
///
/// `logits` has shape `[N, C]`; `targets` and `mask` have shape `[N]`.
/// A zero mask value excludes its row and produces an exact-zero logits
/// adjoint. `valid_count` is the caller-provided normalization denominator and
/// must be in `1..=N`; it is not read back or recomputed from device storage.
/// Reverse mode differentiates only `logits`.
///
/// # Errors
///
/// Returns an error when ranks, shapes, dtypes, ownership, `valid_count`,
/// allocation, or runtime recording are invalid.
pub fn cross_entropy_masked(
	logits: &Matrix,
	targets: &Matrix,
	mask: &Matrix,
	valid_count: usize,
) -> Result<Matrix> {
	const CONTRACT: OperationContract = crate::core::operation::ml::MASKED_CROSS_ENTROPY;
	let (rows, classes, valid_count) =
		validate_cross_entropy_masked_inputs(logits, targets, mask, valid_count, CONTRACT.name())?;
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
	let attributes = [valid_count_attribute(valid_count)];
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
	logits.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: CONTRACT,
			inputs: &[logits, targets, mask],
			outputs: &[&output],
			attributes: &attributes,
		},
	)?;
	autograd::record_masked_cross_entropy(logits, targets, mask, valid_count as usize, &output)?;
	Ok(output)
}

// ---------------------------------------------------------------------------
// Backward passes (called by tape.rs during reverse-mode differentiation)
// ---------------------------------------------------------------------------

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
	{
		let inputs: &[&Matrix] = &[logits, targets];
		let outputs: &[&Matrix] = &[&gradient];
		let attributes: &[OpAttribute] = &[];
		let dispatch = ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: portable_row_workgroups(rows),
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
	Ok(gradient)
}

pub(in crate::ml) fn cross_entropy_masked_backward(
	logits: &Matrix,
	targets: &Matrix,
	mask: &Matrix,
	valid_count: usize,
) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::MASKED_CROSS_ENTROPY_BACKWARD.name();
	let (rows, classes, valid_count) =
		validate_cross_entropy_masked_inputs(logits, targets, mask, valid_count, OPERATION)?;
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
	{
		let inputs: &[&Matrix] = &[logits, targets, mask];
		let outputs: &[&Matrix] = &[&gradient];
		let attributes: &[OpAttribute] = &attributes;
		let dispatch = ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: portable_row_workgroups(rows),
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
	Ok(gradient)
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

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

fn validate_cross_entropy_masked_inputs(
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
