//! Shared mechanics for the loss-family implementations.

use crate::runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch};
use crate::{DType, Error, Matrix, OpAttribute, OperationContract, Result};

// ---------------------------------------------------------------------------
// Shared validation helpers
// ---------------------------------------------------------------------------

pub(super) fn validate_pointwise_loss_inputs(
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

pub(super) fn shader_u32(value: usize, label: &str, operation: &'static str) -> Result<u32> {
	u32::try_from(value)
		.map_err(|_| Error::invalid_argument(format!("{operation} {label} exceeds u32")))
}

pub(super) fn target_dtype(dtype: DType) -> u32 {
	debug_assert!(matches!(dtype, DType::U32 | DType::I32));
	1
}

pub(super) fn portable_row_workgroups(rows: u32) -> [u32; 3] {
	const PORTABLE_WIDTH: u32 = 65_535;
	[rows.min(PORTABLE_WIDTH), rows.div_ceil(PORTABLE_WIDTH), 1]
}

pub(super) fn valid_count_attribute(valid_count: u32) -> OpAttribute {
	OpAttribute::SignedInteger {
		name: "valid_count".into(),
		value: i64::from(valid_count),
	}
}

// ---------------------------------------------------------------------------
// Shared physical-dispatch patterns
// ---------------------------------------------------------------------------

/// Three-dispatch pipeline: per-element kernel → atomic sum → scale to mean.
pub(super) fn mean_pointwise_loss(
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

/// Single-dispatch backward kernel writing per-element gradients.
pub(super) fn pointwise_loss_backward(
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
	{
		let inputs: &[&Matrix] = &[prediction, target];
		let outputs: &[&Matrix] = &[&gradient];
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
	}?;
	Ok(gradient)
}
