use crate::{Matrix, Result};

use super::{autograd, kernels};

/// Apply the tanh-approximate Gaussian error linear unit elementwise.
///
/// # Errors
///
/// Returns an error unless `input` is F32 or runtime recording fails.
pub fn gelu(input: &Matrix) -> Result<Matrix> {
	let output = kernels::gelu(input)?;
	autograd::record_gelu(input, &output)?;
	Ok(output)
}

/// Apply `SiLU(gate) * up` elementwise.
///
/// # Errors
///
/// Returns an error unless both inputs are equal-shape F32 matrices on the
/// same engine, or runtime recording fails.
pub fn swiglu(gate: &Matrix, up: &Matrix) -> Result<Matrix> {
	let output = kernels::swiglu(gate, up)?;
	autograd::record_swiglu(gate, up, &output)?;
	Ok(output)
}
