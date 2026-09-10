//! Shared semantic-dispatch mechanics for private ML lowering families.

use crate::{
	DType, Error, Matrix, OpAttribute, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

#[allow(
	clippy::too_many_arguments,
	reason = "keeps semantic values and physical dispatch fields explicit at each lowering"
)]
pub(super) fn record_semantic(
	inputs: &[&Matrix],
	outputs: &[&Matrix],
	attributes: &[OpAttribute],
	kernel: KernelId,
	buffers: &[BufferBinding<'_>],
	push_constants: &[PushConstant],
	workgroups: [u32; 3],
) -> Result<()> {
	let contract = kernel.semantic_contract().ok_or_else(|| {
		Error::internal(format!(
			"lowering-only kernel {} cannot own one semantic ML operation",
			kernel.report_name()
		))
	})?;
	let engine = inputs
		.first()
		.ok_or_else(|| Error::internal("semantic ML dispatch requires an input"))?
		.engine_handle();
	engine.record_semantic(
		ComputeDispatch {
			kernel,
			buffers,
			push_constants,
			workgroups,
		},
		SemanticDispatch {
			contract,
			inputs,
			outputs,
			attributes,
		},
	)
}

pub(super) fn validate_f32_same_engine(
	operation: &'static str,
	matrices: &[&Matrix],
) -> Result<()> {
	let Some(first) = matrices.first() else {
		return Err(Error::invalid_argument(
			"matrix validation requires an input",
		));
	};
	for matrix in matrices {
		if matrix.dtype() != DType::F32 {
			return Err(Error::invalid_argument(format!(
				"{operation} requires F32 matrices; found {}",
				matrix.dtype().token()
			)));
		}
		if !first.engine_handle().same_as(matrix.engine_handle()) {
			return Err(Error::invalid_argument(format!(
				"{operation} inputs must belong to the same engine"
			)));
		}
	}
	Ok(())
}

pub(super) fn shader_u32(value: usize, label: &str, operation: &'static str) -> Result<u32> {
	u32::try_from(value)
		.map_err(|_| Error::invalid_argument(format!("{operation} {label} exceeds u32")))
}
