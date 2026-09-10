use crate::{Matrix, Result};

pub use super::super::matrix::UpsampleMode;
use super::super::{Module, ModuleRegistry, matrix};

/// Parameterless spatial upsampling module for FP32 NCHW matrices.
pub struct Upsample {
	scale_factor: usize,
	mode: UpsampleMode,
	registry: ModuleRegistry,
}

impl Upsample {
	/// Construct bilinear Upsample with the requested integer scale factor.
	///
	/// # Errors
	///
	/// Returns an error when `scale_factor` is zero.
	pub fn new(scale_factor: usize) -> Result<Self> {
		Self::with_mode(scale_factor, UpsampleMode::Bilinear)
	}

	/// Construct Upsample with an explicit interpolation mode.
	///
	/// # Errors
	///
	/// Returns an error when `scale_factor` is zero.
	pub fn with_mode(scale_factor: usize, mode: UpsampleMode) -> Result<Self> {
		if scale_factor == 0 {
			return Err(crate::Error::invalid_argument(
				"Upsample scale factor must be nonzero",
			));
		}
		Ok(Self {
			scale_factor,
			mode,
			registry: ModuleRegistry::new(),
		})
	}

	/// Upsample one NCHW Matrix without submitting or waiting.
	///
	/// # Errors
	///
	/// Returns an error from the underlying Matrix operation.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		matrix::upsample_2d(input, self.scale_factor, self.mode)
	}

	/// Return the integer spatial scale factor.
	pub const fn scale_factor(&self) -> usize {
		self.scale_factor
	}

	/// Return the interpolation mode.
	pub const fn mode(&self) -> UpsampleMode {
		self.mode
	}
}

impl Module for Upsample {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Upsample::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
