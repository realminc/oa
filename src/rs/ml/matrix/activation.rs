use crate::{Matrix, Result};

use crate::ml::{autograd, lowering::matrix as dispatch};

/// Apply the tanh-approximate Gaussian error linear unit elementwise.
///
/// # Errors
///
/// Returns an error unless `input` is F32 or runtime recording fails.
pub fn gelu(input: &Matrix) -> Result<Matrix> {
	let output = dispatch::gelu(input)?;
	autograd::record_gelu(input, &output)?;
	Ok(output)
}

/// Apply the sigmoid linear unit elementwise.
///
/// # Errors
///
/// Returns an error unless `input` is F32 or runtime recording fails.
pub fn silu(input: &Matrix) -> Result<Matrix> {
	let output = dispatch::silu(input)?;
	autograd::record_silu(input, &output)?;
	Ok(output)
}

/// Apply the rectified linear unit elementwise.
///
/// # Errors
///
/// Returns an error unless `input` is F32 or runtime recording fails.
pub fn relu(input: &Matrix) -> Result<Matrix> {
	let output = dispatch::relu(input)?;
	autograd::record_relu(input, &output, &output)?;
	Ok(output)
}

/// Apply hyperbolic tangent elementwise.
///
/// # Errors
///
/// Returns an error unless `input` is F32 or runtime recording fails.
pub fn tanh(input: &Matrix) -> Result<Matrix> {
	let output = dispatch::tanh(input)?;
	autograd::record_tanh(input, &output, &output)?;
	Ok(output)
}

/// Apply the logistic sigmoid elementwise.
///
/// # Errors
///
/// Returns an error unless `input` is F32 or runtime recording fails.
pub fn sigmoid(input: &Matrix) -> Result<Matrix> {
	let output = dispatch::sigmoid(input)?;
	autograd::record_sigmoid(input, &output, &output)?;
	Ok(output)
}

/// Apply Leaky ReLU elementwise with the requested negative slope.
///
/// # Errors
///
/// Returns an error unless `input` is F32 or runtime recording fails.
pub fn leaky_relu(input: &Matrix, alpha: f32) -> Result<Matrix> {
	let output = dispatch::leaky_relu(input, alpha)?;
	autograd::record_leaky_relu(input, alpha, &output)?;
	Ok(output)
}

/// Apply the exponential linear unit elementwise.
///
/// # Errors
///
/// Returns an error unless `input` is F32 or runtime recording fails.
pub fn elu(input: &Matrix, alpha: f32) -> Result<Matrix> {
	let output = dispatch::elu(input, alpha)?;
	autograd::record_elu(input, &output, alpha, &output)?;
	Ok(output)
}

/// Apply Mish elementwise.
///
/// # Errors
///
/// Returns an error unless `input` is F32 or runtime recording fails.
pub fn mish(input: &Matrix) -> Result<Matrix> {
	let output = dispatch::mish(input)?;
	autograd::record_mish(input, &output)?;
	Ok(output)
}

/// Apply the overflow-safe Softplus formulation elementwise.
///
/// # Errors
///
/// Returns an error unless `input` is F32 or runtime recording fails.
pub fn softplus(input: &Matrix) -> Result<Matrix> {
	let output = dispatch::softplus(input)?;
	autograd::record_softplus(input, &output, &output)?;
	Ok(output)
}

/// Apply `SiLU(gate) * up` elementwise.
///
/// # Errors
///
/// Returns an error unless both inputs are equal-shape F32 matrices on the
/// same engine, or runtime recording fails.
pub fn swiglu(gate: &Matrix, up: &Matrix) -> Result<Matrix> {
	let output = dispatch::swiglu(gate, up)?;
	autograd::record_swiglu(gate, up, &output)?;
	Ok(output)
}
