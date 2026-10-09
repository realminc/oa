//! Stateless optimizer-adjacent transformations and stateful optimizer implementations.

mod adam;
mod adamw;
mod muon;
mod optimizer;
mod sgd;

pub use adam::Adam;
pub use adamw::{AdamW, AdamWProgramSignature};
pub use muon::Muon;
pub use optimizer::{
	CheckpointOptimizer, NoOpOptimizer, Optimizer, OptimizerCheckpoint, OptimizerRestore,
};
pub use sgd::Sgd;

use crate::ml::validation::{shader_u32, validate_f32_same_engine};
use crate::runtime::{BufferBinding, ComputeDispatch, KernelId, SemanticDispatch};
use crate::{Error, Matrix, OpAttribute, Result};

/// Clip the combined L2 norm of nonempty FP32 gradients in place.
///
/// The operation records one GPU reduction followed by one GPU scaling pass.
/// It does not read gradients back to the CPU or wait for completion. Empty
/// matrices are ignored; at most sixteen nonempty gradients are admitted.
///
/// # Errors
///
/// Returns an error when `max_norm` is negative or non-finite, more than
/// sixteen nonempty gradients are supplied, gradients do not share one engine,
/// a gradient is not FP32, storage is duplicated, a size exceeds the shader
/// ABI, or allocation and runtime recording fail.
pub fn clip_grad_norm(gradients: &[Matrix], max_norm: f32) -> Result<()> {
	const OPERATION: &str = crate::core::operation::ml::CLIP_GRAD_NORM.name();
	if !max_norm.is_finite() || max_norm < 0.0 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} max_norm must be finite and non-negative"
		)));
	}
	let gradients = gradients
		.iter()
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
	let partials = Matrix::allocate(engine, vec![16], 16, crate::DType::F32)?;
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
