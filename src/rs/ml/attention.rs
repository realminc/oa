use crate::{Matrix, Result};

use super::{autograd, kernels};

/// Apply multi-head causal scaled dot-product attention to packed `[B*S, D]` values.
///
/// # Errors
///
/// Returns an error unless Q, K, and V are equal nonempty same-engine F32
/// matrices, their row count is divisible by `sequence_length`, their width is
/// divisible by `num_heads`, the sequence length exceeds 1024, or runtime
/// recording fails.
pub fn scaled_dot_product_attention_causal(
	query: &Matrix,
	key: &Matrix,
	value: &Matrix,
	sequence_length: usize,
	num_heads: usize,
) -> Result<Matrix> {
	let result = kernels::scaled_dot_product_attention_causal(
		query,
		key,
		value,
		sequence_length,
		num_heads,
	)?;
	autograd::record_attention(
		query,
		key,
		value,
		&result.output,
		result.probabilities,
		sequence_length,
		num_heads,
	)?;
	Ok(result.output)
}
