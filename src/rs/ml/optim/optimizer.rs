//! Shared optimizer contracts, state types, and trait implementations.
//!
//! Every concrete optimizer struct lives in its own sibling module and imports
//! from here. This file is the analog of `nn/module.rs` for the optimizer
//! subsystem — it owns the common vocabulary that all optimizers share.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Engine, Error, Matrix, Result};

use crate::ml::Parameter;

// ---------------------------------------------------------------------------
// Public contracts
// ---------------------------------------------------------------------------

/// Common behavior exposed by an OA optimizer to training policy.
///
/// This contract deliberately excludes captured-program and persistence
/// internals. Those capabilities depend on optimizer-specific state and are
/// admitted separately only when their complete implementation exists.
pub trait Optimizer {
	/// Discard every currently accumulated parameter gradient.
	fn zero_grad(&self);

	/// Record one logical parameter update.
	///
	/// # Errors
	///
	/// Returns an optimizer, validation, allocation, or runtime recording error.
	fn step(&mut self) -> Result<()>;

	/// Return the learning rate used by the next optimizer step.
	fn learning_rate(&self) -> f32;

	/// Set the learning rate used by subsequent optimizer steps.
	///
	/// # Errors
	///
	/// Returns an error when the value is outside the optimizer's contract or
	/// optimizer state cannot be updated safely.
	fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()>;

	/// Return the number of completed logical optimizer steps.
	fn step_count(&self) -> u64;

	/// Return whether this optimizer can participate in work owned by `engine`.
	fn belongs_to(&self, engine: &Engine) -> bool;
}

/// Optimizer with a complete native `.oam` save-and-restore contract.
///
/// This capability is sealed because persistence must remain synchronized with
/// OA's versioned model-file codec. Custom optimizers can implement
/// [`Optimizer`] for eager training without claiming wire compatibility.
#[allow(private_bounds)]
pub trait CheckpointOptimizer: Optimizer + checkpoint_capability::Persistence {}

impl<T> CheckpointOptimizer for T where T: Optimizer + checkpoint_capability::Persistence {}

// ---------------------------------------------------------------------------
// Checkpoint wire types
// ---------------------------------------------------------------------------

pub struct OptimizerCheckpoint {
	pub(crate) kind: &'static str,
	pub(crate) step: u64,
	pub(crate) learning_rate: f32,
	pub(crate) beta1: f32,
	pub(crate) beta2: f32,
	pub(crate) epsilon: f32,
	pub(crate) weight_decay: f32,
	pub(crate) parameters: Vec<Parameter>,
	pub(crate) first_state: Vec<Matrix>,
	pub(crate) second_state: Vec<Matrix>,
}

pub struct OptimizerRestore {
	pub(crate) kind: String,
	pub(crate) step: u64,
	pub(crate) learning_rate: f32,
	pub(crate) beta1: f32,
	pub(crate) beta2: f32,
	pub(crate) epsilon: f32,
	pub(crate) weight_decay: f32,
	pub(crate) first_state: Vec<Matrix>,
	pub(crate) second_state: Vec<Matrix>,
}

pub(super) mod checkpoint_capability {
	use super::{OptimizerCheckpoint, OptimizerRestore};
	use crate::Result;

	pub trait Persistence {
		fn checkpoint_state(&self) -> Result<OptimizerCheckpoint>;
		fn validate_checkpoint_state(&self, state: &OptimizerRestore) -> Result<()>;
		fn restore_checkpoint_state(&mut self, state: OptimizerRestore) -> Result<()>;
	}
}

// ---------------------------------------------------------------------------
// Shared per-parameter state types reused across optimizer modules
// ---------------------------------------------------------------------------

pub(super) struct MomentParameterState {
	pub(super) parameter: Parameter,
	pub(super) first_moment: Matrix,
	pub(super) second_moment: Matrix,
}

pub(super) struct SgdParameterState {
	pub(super) parameter: Parameter,
	pub(super) momentum: Option<Matrix>,
}

pub(super) struct MuonParameterState {
	pub(super) parameter: Parameter,
	pub(super) momentum: Matrix,
}

// ---------------------------------------------------------------------------
// Stable identity counter for AdamW graph capture
// ---------------------------------------------------------------------------

pub(super) fn next_optimizer_id() -> Result<u64> {
	static NEXT_OPTIMIZER_ID: AtomicU64 = AtomicU64::new(1);
	NEXT_OPTIMIZER_ID
		.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
			(value != u64::MAX).then_some(value + 1)
		})
		.map_err(|_| Error::resource_exhausted("optimizer identity exhausted"))
}

// ---------------------------------------------------------------------------
// NoOpOptimizer — lives here because it has no submodule of its own
// ---------------------------------------------------------------------------

/// Optimizer for externally authored parameter updates.
///
/// `NoOpOptimizer` lets callers use [`crate::ml::ItTraining`] for cadence,
/// metrics, and callbacks while the step body owns all parameter mutation.
#[derive(Clone, Copy, Debug)]
pub struct NoOpOptimizer {
	learning_rate: f32,
}

impl NoOpOptimizer {
	/// Construct a no-op optimizer with the requested policy-visible rate.
	///
	/// # Errors
	///
	/// Returns an error when `learning_rate` is not finite and non-negative.
	pub fn new(learning_rate: f32) -> Result<Self> {
		if !learning_rate.is_finite() || learning_rate < 0.0 {
			return Err(Error::invalid_argument(
				"no-op optimizer learning rate must be finite and non-negative",
			));
		}
		Ok(Self { learning_rate })
	}
}

impl Default for NoOpOptimizer {
	fn default() -> Self {
		Self { learning_rate: 0.0 }
	}
}

impl Optimizer for NoOpOptimizer {
	fn zero_grad(&self) {}

	fn step(&mut self) -> Result<()> {
		Ok(())
	}

	fn learning_rate(&self) -> f32 {
		self.learning_rate
	}

	fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
		if !learning_rate.is_finite() || learning_rate < 0.0 {
			return Err(Error::invalid_argument(
				"no-op optimizer learning rate must be finite and non-negative",
			));
		}
		self.learning_rate = learning_rate;
		Ok(())
	}

	fn step_count(&self) -> u64 {
		0
	}

	fn belongs_to(&self, _engine: &Engine) -> bool {
		true
	}
}
