//! Gated linear unit (`nn.GLU`): split the last configured axis in half and
//! emit `a * sigmoid(b)`.

use crate::{Error, Matrix, Result, matrix};

use super::super::{Module, ModuleRegistry, matrix as ml_matrix};

/// Gated linear unit over one axis (donor default `dim = -1`).
pub struct Glu {
	dim: i32,
	registry: ModuleRegistry,
}

impl Glu {
	/// Construct a GLU over the last axis.
	pub const fn new() -> Self {
		Self::new_with_dim(-1)
	}

	/// Construct a GLU over `dim` (negative values index from the end).
	pub const fn new_with_dim(dim: i32) -> Self {
		Self {
			dim,
			registry: ModuleRegistry::new(),
		}
	}

	/// Return the configured split axis.
	pub const fn dim(&self) -> i32 {
		self.dim
	}

	/// Emit `a * sigmoid(b)` for the two halves of `dim`.
	///
	/// # Errors
	///
	/// Returns an error when the axis is out of range, its extent is odd, or
	/// the composed dispatch fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let rank = input.shape().len();
		let dim = if self.dim < 0 {
			self.dim + rank as i32
		} else {
			self.dim
		};
		if dim < 0 || dim as usize >= rank {
			return Err(Error::invalid_argument(format!(
				"Glu dim {} is outside rank {rank}",
				self.dim
			)));
		}
		let extent = input.shape()[dim as usize];
		if !extent.is_multiple_of(2) {
			return Err(Error::invalid_argument(format!(
				"Glu requires an even extent on the split axis; found {extent}"
			)));
		}
		let half = extent as i64 / 2;
		let gate = matrix::slice(input, dim, 0, half)?;
		let value = matrix::slice(input, dim, half, extent as i64)?;
		matrix::mul(&gate, &ml_matrix::sigmoid(&value)?)
	}
}

impl Default for Glu {
	fn default() -> Self {
		Self::new()
	}
}

impl Module for Glu {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Glu::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
