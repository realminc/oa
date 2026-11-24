//! Connectionist temporal classification loss.

use crate::OpAttribute;
use crate::ml::autograd;
use crate::runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch};
use crate::{DType, Error, Matrix, Result};

use super::common::{portable_row_workgroups, shader_u32};

/// Maximum label-sequence width the shared-memory lattice admits: the extended
/// sequence holds `2 * target_width + 1` states in at most 513 entries.
const MAX_TARGET_WIDTH: usize = 256;

/// Compute the mean connectionist temporal classification loss over a batch.
///
/// `log_probs` has shape `[T, N, C]` and holds log-probabilities over `C`
/// classes including the blank class. `targets` has shape `[N, S]` and holds
/// the label sequences, `input_lengths` and `target_lengths` have shape `[N]`
/// and hold each item's frame and label counts. `blank` is the blank class
/// index and must be below `C`. The returned matrix is an FP32 scalar: the
/// mean of the per-item negative log-likelihoods.
///
/// Label entries shorter than `S` and frames beyond an item's
/// `input_lengths` entry are ignored. Target classes outside `0..C` produce
/// a NaN item loss without an out-of-bounds read, matching the cross-entropy
/// invalid-target contract.
///
/// # Errors
///
/// Returns an error when ranks, shapes, dtypes, `blank`, `target_lengths`
/// width, ownership, allocation, or runtime recording are invalid.
pub fn ctc(
	log_probs: &Matrix,
	targets: &Matrix,
	input_lengths: &Matrix,
	target_lengths: &Matrix,
	blank: usize,
) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::CTC.name();
	let (times, batch, classes, width) = validate_ctc_inputs(
		log_probs,
		targets,
		input_lengths,
		target_lengths,
		blank,
		OPERATION,
	)?;
	let per_item = Matrix::allocate(
		log_probs.engine_handle(),
		vec![batch as usize],
		batch as usize,
		DType::F32,
	)?;
	let output = Matrix::allocate(log_probs.engine_handle(), Vec::new(), 1, DType::F32)?;
	let loss_buffers = [
		BufferBinding::read(log_probs.storage()),
		BufferBinding::read(targets.storage()),
		BufferBinding::read(input_lengths.storage()),
		BufferBinding::read(target_lengths.storage()),
		BufferBinding::write(per_item.storage()),
	];
	let loss_push_constants = [
		PushConstant::U32(times),
		PushConstant::U32(batch),
		PushConstant::U32(classes),
		PushConstant::U32(width),
		PushConstant::U32(blank as u32),
	];
	let mean_buffers = [
		BufferBinding::read(per_item.storage()),
		BufferBinding::write(output.storage()),
	];
	let mean_push_constants = [
		PushConstant::U32(batch),
		PushConstant::F32(1.0 / batch as f32),
	];
	let attributes = [OpAttribute::UnsignedInteger {
		name: "blank".into(),
		value: blank as u64,
	}];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlCtcF32,
			buffers: &loss_buffers,
			push_constants: &loss_push_constants,
			workgroups: portable_row_workgroups(batch),
			indirect: None,
		},
		ComputeDispatch {
			kernel: KernelId::MlSumScaleF32,
			buffers: &mean_buffers,
			push_constants: &mean_push_constants,
			workgroups: [1, 1, 1],
			indirect: None,
		},
	];
	log_probs.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::ml::CTC,
			inputs: &[log_probs, targets, input_lengths, target_lengths],
			outputs: &[&output],
			attributes: &attributes,
		},
	)?;
	autograd::record_ctc(
		log_probs,
		targets,
		input_lengths,
		target_lengths,
		blank,
		&output,
	)?;
	Ok(output)
}

// ---------------------------------------------------------------------------
// Backward pass (called by tape.rs during reverse-mode differentiation)
// ---------------------------------------------------------------------------

pub(in crate::ml) fn ctc_backward(
	log_probs: &Matrix,
	targets: &Matrix,
	input_lengths: &Matrix,
	target_lengths: &Matrix,
	blank: usize,
) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::CTC_BACKWARD.name();
	let (times, batch, classes, width) = validate_ctc_inputs(
		log_probs,
		targets,
		input_lengths,
		target_lengths,
		blank,
		OPERATION,
	)?;
	// The gradient accumulates class posteriors through shared atomic
	// contributors; it must start at exact zero.
	let gradient = log_probs.new_from_slice_on_same_engine(
		log_probs.shape().to_vec(),
		&vec![0.0_f32; log_probs.num_elements()],
	)?;
	let lattice = Matrix::allocate(
		log_probs.engine_handle(),
		vec![batch as usize, times as usize * (2 * width as usize + 1)],
		batch as usize * times as usize * (2 * width as usize + 1),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(log_probs.storage()),
		BufferBinding::read(targets.storage()),
		BufferBinding::read(input_lengths.storage()),
		BufferBinding::read(target_lengths.storage()),
		BufferBinding::write(lattice.storage()),
		BufferBinding::write(gradient.storage()),
	];
	let push_constants = [
		PushConstant::U32(times),
		PushConstant::U32(batch),
		PushConstant::U32(classes),
		PushConstant::U32(width),
		PushConstant::U32(blank as u32),
	];
	let attributes = [OpAttribute::UnsignedInteger {
		name: "blank".into(),
		value: blank as u64,
	}];
	let dispatch = ComputeDispatch {
		kernel: KernelId::MlCtcBackwardF32,
		buffers: &buffers,
		push_constants: &push_constants,
		workgroups: portable_row_workgroups(batch),
		indirect: None,
	};
	log_probs.engine_handle().record_semantic(
		dispatch,
		SemanticDispatch {
			contract: crate::core::operation::ml::CTC_BACKWARD,
			inputs: &[log_probs, targets, input_lengths, target_lengths],
			outputs: &[&gradient],
			attributes: &attributes,
		},
	)?;
	Ok(gradient)
}

fn validate_ctc_inputs(
	log_probs: &Matrix,
	targets: &Matrix,
	input_lengths: &Matrix,
	target_lengths: &Matrix,
	blank: usize,
	operation: &'static str,
) -> Result<(u32, u32, u32, u32)> {
	if log_probs.dtype() != DType::F32 || log_probs.shape().len() != 3 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires rank-three FP32 log_probs"
		)));
	}
	let shape = log_probs.shape();
	let times = shader_u32(shape[0], "times", operation)?;
	let batch = shader_u32(shape[1], "batch", operation)?;
	let classes = shader_u32(shape[2], "classes", operation)?;
	for (name, matrix) in [
		("targets", targets),
		("input_lengths", input_lengths),
		("target_lengths", target_lengths),
	] {
		if !matches!(matrix.dtype(), DType::I32 | DType::U32) {
			return Err(Error::invalid_argument(format!(
				"{operation} requires I32 or U32 {name}"
			)));
		}
		if !matrix.engine_handle().same_as(log_probs.engine_handle()) {
			return Err(Error::invalid_argument(format!(
				"{operation} inputs must belong to the same engine"
			)));
		}
	}
	if targets.shape().len() != 2 || targets.shape()[0] != shape[1] {
		return Err(Error::invalid_argument(format!(
			"{operation} targets must have shape [N, S] matching log_probs [T, N, C]"
		)));
	}
	for (name, matrix) in [
		("input_lengths", input_lengths),
		("target_lengths", target_lengths),
	] {
		if matrix.shape().len() != 1 || matrix.shape()[0] != shape[1] {
			return Err(Error::invalid_argument(format!(
				"{operation} {name} must have shape [N] matching log_probs [T, N, C]"
			)));
		}
	}
	let width = shader_u32(targets.shape()[1], "target_width", operation)?;
	if width as usize > MAX_TARGET_WIDTH {
		return Err(Error::invalid_argument(format!(
			"{operation} target width {width} exceeds the shared-lattice limit {MAX_TARGET_WIDTH}"
		)));
	}
	if blank >= classes as usize {
		return Err(Error::invalid_argument(format!(
			"{operation} blank index {blank} must be below the {classes} classes"
		)));
	}
	Ok((times, batch, classes, width))
}
