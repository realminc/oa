//! Sparse mixture-of-experts Matrix operations.

use crate::{Matrix, Result};

use crate::ml::{Parameter, autograd, lowering::moe as algebra};

/// Apply one bias-free projection per expert to expert-major packed rows.
///
/// `input` is F32 `[R, K]`, `weight` is F32 `[E, N, K]`, and `offsets` is
/// the U32 `[E+1]` boundary output from [`crate::matrix::moe_expert_plan`].
/// The result is F32 `[R, N]`.
///
/// # Errors
///
/// Returns an error for incompatible shapes, dtypes, engine ownership, shader
/// ABI overflow, allocation failure, or runtime recording failure.
pub fn grouped_gemm_m(input: &Matrix, weight: &Matrix, offsets: &Matrix) -> Result<Matrix> {
	let output = algebra::grouped_gemm_m(input, weight, offsets)?;
	autograd::record_grouped_gemm_m(input, &output, None, weight.clone(), offsets)?;
	Ok(output)
}

#[allow(dead_code, reason = "reserved for the first bias-free grouped module")]
pub(in crate::ml) fn grouped_gemm_m_parameterized(
	input: &Matrix,
	weight: (Parameter, Matrix, u64),
	offsets: &Matrix,
) -> Result<Matrix> {
	let (weight_parameter, weight_value, weight_version) = weight;
	let output = algebra::grouped_gemm_m(input, &weight_value, offsets)?;
	autograd::record_grouped_gemm_m(
		input,
		&output,
		Some((weight_parameter, weight_version)),
		weight_value,
		offsets,
	)?;
	Ok(output)
}

/// Apply one affine projection per expert to expert-major packed rows.
///
/// `input` is F32 `[R, K]`, `weight` is F32 `[E, N, K]`, `bias` is F32
/// `[E, N]`, and `offsets` is the U32 `[E+1]` boundary output from
/// [`crate::matrix::moe_expert_plan`]. The result is F32 `[R, N]`.
///
/// # Errors
///
/// Returns an error for incompatible shapes, dtypes, engine ownership, shader
/// ABI overflow, allocation failure, or runtime recording failure.
pub fn grouped_linear_m(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	offsets: &Matrix,
) -> Result<Matrix> {
	let output = algebra::grouped_linear_m(input, weight, bias, offsets)?;
	autograd::record_grouped_linear_m(
		input,
		&output,
		None,
		weight.clone(),
		None,
		bias.clone(),
		offsets,
	)?;
	Ok(output)
}

pub(in crate::ml) fn grouped_linear_m_parameterized(
	input: &Matrix,
	weight: (Parameter, Matrix, u64),
	bias: (Parameter, Matrix, u64),
	offsets: &Matrix,
) -> Result<Matrix> {
	let (weight_parameter, weight_value, weight_version) = weight;
	let (bias_parameter, bias_value, bias_version) = bias;
	let output = algebra::grouped_linear_m(input, &weight_value, &bias_value, offsets)?;
	autograd::record_grouped_linear_m(
		input,
		&output,
		Some((weight_parameter, weight_version)),
		weight_value,
		Some((bias_parameter, bias_version)),
		bias_value,
		offsets,
	)?;
	Ok(output)
}

/// Normalize probabilities over the selected experts for each token.
///
/// `probabilities` is F32 `[T, E]`; `expert_indices` is I32 `[T, K]` with
/// `1 <= K <= E`. The routing decision is detached from reverse mode, while
/// gradients propagate to the selected probability entries.
///
/// # Errors
///
/// Returns an error for incompatible shapes, dtypes, engine ownership, shader
/// ABI overflow, allocation failure, or runtime recording failure.
pub fn moe_route_weights(probabilities: &Matrix, expert_indices: &Matrix) -> Result<Matrix> {
	let output = algebra::route_weights(probabilities, expert_indices)?;
	autograd::record_moe_route_weights(probabilities, expert_indices, &output)?;
	Ok(output)
}

/// Pack token rows into stable expert-major route order.
///
/// `input` is F32 `[T, D]`; `packed_token` and `inverse` are the U32 `[T*K]`
/// maps produced by [`crate::matrix::moe_expert_plan`]. `inverse` is retained
/// for the deterministic reverse pass and is not read by the forward kernel.
///
/// # Errors
///
/// Returns an error for incompatible shapes, dtypes, engine ownership, shader
/// ABI overflow, allocation failure, or runtime recording failure.
pub fn moe_gather(input: &Matrix, packed_token: &Matrix, inverse: &Matrix) -> Result<Matrix> {
	let output = algebra::gather(input, packed_token, inverse)?;
	autograd::record_moe_gather(input, inverse, &output)?;
	Ok(output)
}

/// Combine expert-major packed rows into token-major output using route gates.
///
/// `packed` is F32 `[T*K, D]`, `route_gate` is F32 `[T, K]`, and the U32
/// maps come from [`crate::matrix::moe_expert_plan`].
///
/// # Errors
///
/// Returns an error for incompatible shapes, dtypes, engine ownership, shader
/// ABI overflow, allocation failure, or runtime recording failure.
pub fn moe_combine(
	packed: &Matrix,
	route_gate: &Matrix,
	inverse: &Matrix,
	packed_slot: &Matrix,
) -> Result<Matrix> {
	let output = algebra::combine(packed, route_gate, inverse, packed_slot)?;
	autograd::record_moe_combine(packed, route_gate, inverse, packed_slot, &output)?;
	Ok(output)
}

pub(in crate::ml) fn moe_route_weights_backward(
	output_gradient: &Matrix,
	probabilities: &Matrix,
	expert_indices: &Matrix,
	route_weights: &Matrix,
) -> Result<Matrix> {
	algebra::route_weights_backward(
		output_gradient,
		probabilities,
		expert_indices,
		route_weights,
	)
}

pub(in crate::ml) fn moe_gather_backward(
	output_gradient: &Matrix,
	inverse: &Matrix,
	output_rows: usize,
) -> Result<Matrix> {
	algebra::gather_backward(output_gradient, inverse, output_rows)
}

pub(in crate::ml) fn moe_combine_backward(
	output_gradient: &Matrix,
	packed: &Matrix,
	route_gate: &Matrix,
	inverse: &Matrix,
	packed_slot: &Matrix,
) -> Result<algebra::CombineBackward> {
	algebra::combine_backward(output_gradient, packed, route_gate, inverse, packed_slot)
}

pub(in crate::ml) fn grouped_linear_m_backward(
	output_gradient: &Matrix,
	input: &Matrix,
	weight: &Matrix,
	offsets: &Matrix,
) -> Result<algebra::GroupedLinearBackward> {
	algebra::grouped_linear_m_backward(output_gradient, input, weight, offsets)
}

pub(in crate::ml) fn grouped_gemm_m_backward(
	output_gradient: &Matrix,
	input: &Matrix,
	weight: &Matrix,
	offsets: &Matrix,
) -> Result<algebra::GroupedGemmBackward> {
	algebra::grouped_gemm_m_backward(output_gradient, input, weight, offsets)
}
