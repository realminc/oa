//! Local response normalization (`nn.LocalResponseNorm`) over the channel axis.

use crate::{Error, Matrix, Result, matrix};

use super::super::{Module, ModuleRegistry};

/// Cross-channel normalization `x_c / (k + (alpha / size) * sum_j x_j^2)^beta`
/// where the sum spans the `size`-wide neighborhood centered at channel `c`,
/// clipped at the borders.
///
/// The window sum is composed over a zero `PadMode::Constant` channel pad and
/// `size` shifted slices, so gradients flow through the pad and slice
/// adjoints.
pub struct LocalResponseNorm {
	size: usize,
	alpha: f32,
	beta: f32,
	k: f32,
	registry: ModuleRegistry,
}

impl LocalResponseNorm {
	/// Construct with the donor defaults (`alpha = 1e-4`, `beta = 0.75`,
	/// `k = 1.0`).
	///
	/// # Errors
	///
	/// Returns an error when `size` is zero.
	pub fn new(size: usize) -> Result<Self> {
		Self::new_with_options(size, 1e-4, 0.75, 1.0)
	}

	/// Construct with explicit normalization coefficients.
	///
	/// # Errors
	///
	/// Returns an error when `size` is zero or a coefficient is not finite.
	pub fn new_with_options(size: usize, alpha: f32, beta: f32, k: f32) -> Result<Self> {
		if size == 0 {
			return Err(Error::invalid_argument(
				"local response norm requires a nonzero window size",
			));
		}
		if !alpha.is_finite() || !beta.is_finite() || !k.is_finite() {
			return Err(Error::invalid_argument(
				"local response norm coefficients must be finite",
			));
		}
		Ok(Self {
			size,
			alpha,
			beta,
			k,
			registry: ModuleRegistry::new(),
		})
	}

	/// Return the configured window size.
	pub const fn size(&self) -> usize {
		self.size
	}

	/// Normalize channels of a rank-at-least-two `[N, C, *]` input.
	///
	/// # Errors
	///
	/// Returns an error for a rank-below-two input or a composed dispatch
	/// failure.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let operation = "LocalResponseNorm";
		let shape = input.shape();
		if shape.len() < 2 {
			return Err(Error::invalid_argument(format!(
				"{operation} requires an input of rank at least two"
			)));
		}
		let half = self.size / 2;
		let squared = matrix::mul(input, input)?;
		let mut pads = vec![0_usize; 2 * shape.len()];
		pads[2] = half;
		pads[3] = half;
		let padded = matrix::pad(&squared, &pads, matrix::PadMode::Constant, 0.0)?;
		let channels = shape[1] as i64;
		let mut sum: Option<Matrix> = None;
		for offset in 0..self.size {
			let window = matrix::slice(&padded, 1, offset as i64, offset as i64 + channels)?;
			sum = Some(match sum {
				None => window,
				Some(total) => matrix::add(&total, &window)?,
			});
		}
		let sum = sum.ok_or_else(|| {
			Error::invalid_argument(format!("{operation} requires a nonzero window size"))
		})?;
		let denominator = matrix::pow(
			&matrix::add_scalar(&matrix::scale(&sum, self.alpha / self.size as f32)?, self.k)?,
			self.beta,
		)?;
		matrix::div(input, &denominator)
	}
}

impl Module for LocalResponseNorm {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		LocalResponseNorm::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
