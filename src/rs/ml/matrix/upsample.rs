use crate::{Matrix, Result};

use crate::ml::{autograd, lowering::upsample as dispatch};

/// Spatial interpolation mode for NCHW upsampling.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpsampleMode {
	/// Select the source pixel at the integer-scaled coordinate.
	Nearest,
	/// Apply align-corners-false half-pixel bilinear interpolation.
	Bilinear,
}

impl UpsampleMode {
	pub(crate) const fn token(self) -> &'static str {
		match self {
			Self::Nearest => "nearest",
			Self::Bilinear => "bilinear",
		}
	}
}

/// Upsample an FP32 NCHW Matrix by a positive integer scale factor.
///
/// # Errors
///
/// Returns an error unless the input is nonempty rank-four F32, the scale
/// factor is positive, all output geometry is representable, or runtime
/// recording fails.
pub fn upsample_2d(input: &Matrix, scale_factor: usize, mode: UpsampleMode) -> Result<Matrix> {
	let output = dispatch::upsample_2d(input, scale_factor, mode)?;
	autograd::record_upsample_2d(input, &output, scale_factor, mode)?;
	Ok(output)
}

pub(in crate::ml) fn upsample_2d_backward(
	input: &Matrix,
	output_gradient: &Matrix,
	scale_factor: usize,
	mode: UpsampleMode,
) -> Result<Matrix> {
	dispatch::upsample_2d_backward(input, output_gradient, scale_factor, mode)
}
