use crate::{Error, Matrix, Result, matrix};

use super::super::{Module, ModuleRegistry};

/// Inverted Dropout with explicit train/eval behavior.
pub struct Dropout {
	probability: f32,
	seed: u64,
	registry: ModuleRegistry,
}

impl Dropout {
	/// Construct Dropout with the thread-local random seed source.
	///
	/// # Errors
	///
	/// Returns an error unless `probability` is finite and in `[0, 1)`.
	pub fn new(probability: f32) -> Result<Self> {
		Self::with_seed(probability, 0)
	}

	/// Construct Dropout with an explicit Philox seed.
	///
	/// A nonzero seed makes independently recorded eager calls reproducible.
	/// Captured programs still advance their private random counter on every
	/// replay. Zero consumes the thread-local seed source when training forward
	/// is recorded.
	///
	/// # Errors
	///
	/// Returns an error unless `probability` is finite and in `[0, 1)`.
	pub fn with_seed(probability: f32, seed: u64) -> Result<Self> {
		if !probability.is_finite() || !(0.0..1.0).contains(&probability) {
			return Err(Error::invalid_argument(
				"Dropout probability must be finite and in [0, 1)",
			));
		}
		Ok(Self {
			probability,
			seed,
			registry: ModuleRegistry::new(),
		})
	}

	/// Return the probability of dropping each element during training.
	pub const fn probability(&self) -> f32 {
		self.probability
	}

	/// Apply Dropout in training mode or return an identity handle in eval mode.
	///
	/// # Errors
	///
	/// Returns an error from the underlying Matrix operation.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		if !self.is_training() || self.probability == 0.0 {
			return Ok(input.clone());
		}
		matrix::dropout(input, self.probability, self.seed)
	}
}

impl Module for Dropout {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Dropout::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
