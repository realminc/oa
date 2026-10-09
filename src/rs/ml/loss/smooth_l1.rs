//! Smooth L1 (Huber) loss.

use crate::Error;
use crate::OpAttribute;
use crate::ml::autograd;
use crate::runtime::ComputeDispatch;
use crate::runtime::SemanticDispatch;
use crate::runtime::{BufferBinding, KernelId, PushConstant};
use crate::{DType, Matrix, Result};

use super::common::{pointwise_loss_backward, validate_pointwise_loss_inputs};

/// Compute mean unit-beta Smooth L1 loss for matching FP32 matrices.
///
/// The target is treated as detached; reverse mode differentiates only the
/// prediction.
///
/// # Errors
///
/// Returns an error for empty, mismatched, non-FP32, or cross-engine inputs, or
/// when allocation or runtime recording fails.
pub fn smooth_l1(prediction: &Matrix, target: &Matrix) -> Result<Matrix> {
	let element_count = validate_pointwise_loss_inputs(
		prediction,
		target,
		crate::core::operation::ml::SMOOTH_L1.name(),
	)?;
	let output = Matrix::allocate(prediction.engine_handle(), Vec::new(), 1, DType::F32)?;
	let buffers = [
		BufferBinding::read(prediction.storage()),
		BufferBinding::read(target.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [PushConstant::U32(element_count)];
	let kernel = KernelId::MlSmoothL1MeanF32;
	{
		let inputs: &[&Matrix] = &[prediction, target];
		let outputs: &[&Matrix] = &[&output];
		let attributes: &[OpAttribute] = &[];
		let dispatch = ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
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
	}?;
	autograd::record_smooth_l1(prediction, target, &output)?;
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
