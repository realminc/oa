//! Stateless optimizer-adjacent transformations and stateful optimizer implementations.

mod adam;
mod adamw;
mod grad_scaler;
mod muon;
mod optimizer;
mod sgd;

pub use adam::Adam;
pub use adamw::{AdamW, AdamWProgramSignature};
pub use grad_scaler::GradScaler;
pub use muon::Muon;
pub use optimizer::{
	CheckpointOptimizer, NoOpOptimizer, Optimizer, OptimizerCheckpoint, OptimizerRestore,
};
pub use sgd::Sgd;

use crate::ml::Parameter;
use crate::ml::matrix::detach;
use crate::ml::validation::{shader_u32, validate_f32_same_runtime};
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
/// sixteen nonempty gradients are supplied, gradients do not share one runtime,
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
	validate_f32_same_runtime(OPERATION, &gradients)?;
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
	let runtime = gradients[0].runtime();
	let params = Matrix::new_from_slice_handle(runtime, vec![18], &params)?;
	let partials = Matrix::allocate(runtime, vec![16], 16, crate::DType::F32)?;
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
			indirect: None,
		},
		ComputeDispatch {
			kernel: scale,
			buffers: &scale_buffers,
			push_constants: &[],
			workgroups: [gradient_count, 1, 1],
			indirect: None,
		},
	];
	let attributes = [OpAttribute::Float {
		name: "max_norm".into(),
		value: f64::from(max_norm),
	}];
	runtime.record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::ml::CLIP_GRAD_NORM,
			inputs: &gradients,
			outputs: &gradients,
			attributes: &attributes,
		},
	)
}

/// Clip the combined L2 norm of accumulated [`Parameter`] gradients.
///
/// This is the parameter-level form of [`clip_grad_norm`]: each parameter's
/// live gradient handle is collected, parameters without a gradient are
/// skipped, and gradient values are rescaled so a following
/// [`Optimizer::step`] observes clipped values. Up to sixteen nonempty
/// gradients scale in place through the fused two-pass kernel; larger models
/// take an equivalent device-resident reduction and broadcast-rescale path
/// that stores a detached replacement gradient.
///
/// Call between backward and the optimizer step. Inside
/// [`crate::ml::ItTraining::step`] the call belongs in the `record` closure so
/// the rescale is captured into the training program; in eager steps it
/// belongs before [`crate::ml::ItTraining::complete_step`].
///
/// # Errors
///
/// Returns an error when `max_norm` is negative or non-finite, gradients do
/// not share one runtime or are not FP32, storages are duplicated on the fused
/// path, a size exceeds the shader ABI, or allocation and runtime recording
/// fail.
pub fn clip_parameter_grad_norm(parameters: &[Parameter], max_norm: f32) -> Result<()> {
	const OPERATION: &str = "clip_parameter_grad_norm";
	if !max_norm.is_finite() || max_norm < 0.0 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} max_norm must be finite and non-negative"
		)));
	}
	let pairs = parameters
		.iter()
		.filter_map(|parameter| parameter.gradient().map(|gradient| (parameter, gradient)))
		.collect::<Vec<_>>();
	if pairs.is_empty() {
		return Ok(());
	}
	if pairs.len() <= 16 {
		let gradients = pairs
			.iter()
			.map(|(_, gradient)| gradient.clone())
			.collect::<Vec<_>>();
		return clip_grad_norm(&gradients, max_norm);
	}
	// The fused kernel admits at most sixteen gradient storages. Larger models
	// take the equivalent device-resident route: one full reduction over every
	// squared norm, then a broadcast rescale into detached replacement storage.
	let mut total_sq: Option<Matrix> = None;
	for (_, gradient) in &pairs {
		let sq = gradient.mul(gradient)?.sum(-1)?;
		total_sq = Some(match total_sq {
			Some(acc) => acc.add(&sq)?,
			None => sq,
		});
	}
	let Some(total_sq) = total_sq else {
		return Ok(());
	};
	let scale = total_sq
		.sqrt()?
		.add_scalar(1.0e-6)?
		.reciprocal()?
		.scale(max_norm)?
		.clamp_max(1.0)?;
	for (parameter, gradient) in &pairs {
		parameter.set_gradient(detach(&gradient.mul(&scale)?)?);
	}
	Ok(())
}
