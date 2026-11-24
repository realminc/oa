//! Pairwise distance and cosine similarity over one axis.

use crate::{Error, Matrix, Result, matrix};

use super::super::{Module, ModuleRegistry};

/// Pairwise p-norm distance (`nn.PairwiseDistance`) over one axis.
///
/// Emits `||x1 - x2||_p` keeping the reduced axis at extent one, matching the
/// donor's `keepdim` shape.
pub struct PairwiseDistance {
	p: f32,
	dim: i32,
	eps: f32,
	registry: ModuleRegistry,
}

impl PairwiseDistance {
	/// Construct with the donor defaults (`p = 2`, `dim = 1`, `eps = 1e-6`).
	pub const fn new() -> Self {
		Self::new_with_options(2.0, 1, 1e-6)
	}

	/// Construct with explicit `p`, `dim`, and `eps`.
	pub const fn new_with_options(p: f32, dim: i32, eps: f32) -> Self {
		Self {
			p,
			dim,
			eps,
			registry: ModuleRegistry::new(),
		}
	}

	/// Compute `(||x1 - x2||_p + eps)` along `dim`.
	///
	/// `eps` is added inside the norm for gradient stability at zero, matching
	/// the donor contract.
	///
	/// # Errors
	///
	/// Returns an error for mismatched shapes, an out-of-range axis, or a
	/// composed dispatch failure.
	pub fn forward2(&self, input1: &Matrix, input2: &Matrix) -> Result<Matrix> {
		let operation = "PairwiseDistance";
		if input1.shape() != input2.shape() {
			return Err(Error::invalid_argument(format!(
				"{operation} requires matching input shapes; found {:?} and {:?}",
				input1.shape(),
				input2.shape()
			)));
		}
		let rank = input1.shape().len();
		let dim = if self.dim < 0 {
			self.dim + rank as i32
		} else {
			self.dim
		};
		if dim < 0 || dim as usize >= rank || !self.p.is_finite() || self.p <= 0.0 {
			return Err(Error::invalid_argument(format!(
				"{operation} dim {} is outside rank {rank} or p is invalid",
				self.dim
			)));
		}
		let difference = matrix::sub(input1, input2)?;
		let powered = matrix::pow(&matrix::abs(&difference)?, self.p)?;
		matrix::pow(
			&matrix::add_scalar(&matrix::sum(&powered, dim)?, self.eps)?,
			1.0 / self.p,
		)
	}
}

impl Default for PairwiseDistance {
	fn default() -> Self {
		Self::new()
	}
}

impl Module for PairwiseDistance {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let _ = input;
		Err(Error::invalid_argument(
			"PairwiseDistance requires two inputs; call forward2(input1, input2)",
		))
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Cosine similarity (`nn.CosineSimilarity`) over one axis.
///
/// Emits `x1 . x2 / (max(||x1||, eps) * max(||x2||, eps))` keeping the reduced
/// axis at extent one.
pub struct CosineSimilarity {
	dim: i32,
	eps: f32,
	registry: ModuleRegistry,
}

impl CosineSimilarity {
	/// Construct with the donor defaults (`dim = 1`, `eps = 1e-8`).
	pub const fn new() -> Self {
		Self::new_with_options(1, 1e-8)
	}

	/// Construct with explicit `dim` and `eps`.
	pub const fn new_with_options(dim: i32, eps: f32) -> Self {
		Self {
			dim,
			eps,
			registry: ModuleRegistry::new(),
		}
	}

	/// Compute the cosine similarity of `input1` and `input2` along `dim`.
	///
	/// # Errors
	///
	/// Returns an error for mismatched shapes, an out-of-range axis, or a
	/// composed dispatch failure.
	pub fn forward2(&self, input1: &Matrix, input2: &Matrix) -> Result<Matrix> {
		let operation = "CosineSimilarity";
		if input1.shape() != input2.shape() {
			return Err(Error::invalid_argument(format!(
				"{operation} requires matching input shapes; found {:?} and {:?}",
				input1.shape(),
				input2.shape()
			)));
		}
		let rank = input1.shape().len();
		let dim = if self.dim < 0 {
			self.dim + rank as i32
		} else {
			self.dim
		};
		if dim < 0 || dim as usize >= rank {
			return Err(Error::invalid_argument(format!(
				"{operation} dim {} is outside rank {rank}",
				self.dim
			)));
		}
		let norm = |input: &Matrix| -> Result<Matrix> {
			matrix::clamp_min(
				&matrix::sqrt(&matrix::sum(&matrix::mul(input, input)?, dim)?)?,
				self.eps,
			)
		};
		let numerator = matrix::sum(&matrix::mul(input1, input2)?, dim)?;
		matrix::div(&numerator, &matrix::mul(&norm(input1)?, &norm(input2)?)?)
	}
}

impl Default for CosineSimilarity {
	fn default() -> Self {
		Self::new()
	}
}

impl Module for CosineSimilarity {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let _ = input;
		Err(Error::invalid_argument(
			"CosineSimilarity requires two inputs; call forward2(input1, input2)",
		))
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
