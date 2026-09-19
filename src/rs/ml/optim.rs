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

use crate::Matrix;
use crate::Result;

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
	let gradients = gradients.iter().collect::<Vec<_>>();
	super::lowering::optim::clip_grad_norm(&gradients, max_norm)
}

#[cfg(test)]
mod tests {
	use super::{AdamW, NoOpOptimizer, Optimizer};
	use crate::ml::Parameter;

	#[test]
	fn no_op_optimizer_is_an_object_safe_policy_owner() {
		let mut optimizer = NoOpOptimizer::new(0.25).expect("valid rate");
		let optimizer: &mut dyn Optimizer = &mut optimizer;
		assert_eq!(optimizer.learning_rate(), 0.25);
		assert_eq!(optimizer.step_count(), 0);
		optimizer.zero_grad();
		optimizer.step().expect("no-op step");
		optimizer.set_learning_rate(0.125).expect("valid rate");
		assert_eq!(optimizer.learning_rate(), 0.125);
		assert_eq!(optimizer.step_count(), 0);
	}

	#[test]
	fn no_op_optimizer_rejects_invalid_rates() {
		assert!(NoOpOptimizer::new(f32::NAN).is_err());
		let mut optimizer = NoOpOptimizer::default();
		assert!(optimizer.set_learning_rate(-1.0).is_err());
	}

	#[test]
	fn adamw_explicit_hyperparameters_reject_invalid_scalars_before_allocation() {
		let empty = || std::iter::empty::<Parameter>();
		assert!(AdamW::with_hyperparameters(empty(), f32::NAN, 0.9, 0.999, 1.0e-8, 0.0).is_err());
		assert!(AdamW::with_hyperparameters(empty(), 1.0e-3, 1.0, 0.999, 1.0e-8, 0.0).is_err());
		assert!(AdamW::with_hyperparameters(empty(), 1.0e-3, 0.9, -0.1, 1.0e-8, 0.0).is_err());
		assert!(AdamW::with_hyperparameters(empty(), 1.0e-3, 0.9, 0.999, 0.0, 0.0).is_err());
		assert!(AdamW::with_hyperparameters(empty(), 1.0e-3, 0.9, 0.999, 1.0e-8, -0.1).is_err());
	}
}
