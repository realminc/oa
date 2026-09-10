use crate::{Matrix, Result};

use crate::ml::{autograd, lowering::attention as algebra};

/// Multiply corresponding matrices in rank-three batches.
///
/// Computes `[N, M, K] @ [N, K, P] -> [N, M, P]`.
///
/// # Errors
///
/// Returns an error unless inputs are compatible nonempty same-engine F32
/// matrices, shape arithmetic fits the shader ABI, or runtime recording fails.
pub fn bmm(left: &Matrix, right: &Matrix) -> Result<Matrix> {
	let output = algebra::bmm(left, right)?;
	autograd::record_bmm(left, right, &output)?;
	Ok(output)
}

/// Multiply rank-three batches with right-transposed storage.
///
/// Computes `[N, M, K] @ [N, P, K]^T -> [N, M, P]`.
///
/// # Errors
///
/// Returns an error unless inputs are compatible nonempty same-engine F32
/// matrices, shape arithmetic fits the shader ABI, or runtime recording fails.
pub fn bmm_nt(left: &Matrix, right: &Matrix) -> Result<Matrix> {
	let output = algebra::bmm_nt(left, right)?;
	autograd::record_bmm_nt(left, right, &output)?;
	Ok(output)
}

/// Multiply rank-three batches with left-transposed storage.
///
/// Computes `[N, K, M]^T @ [N, K, P] -> [N, M, P]`.
///
/// # Errors
///
/// Returns an error unless inputs are compatible nonempty same-engine F32
/// matrices, shape arithmetic fits the shader ABI, or runtime recording fails.
pub fn bmm_tn(left: &Matrix, right: &Matrix) -> Result<Matrix> {
	let output = algebra::bmm_tn(left, right)?;
	autograd::record_bmm_tn(left, right, &output)?;
	Ok(output)
}

/// Permute contiguous `[B*S, D]` storage into `[B*H, S, D/H]` heads.
///
/// The one-head case is a zero-copy differentiable view.
///
/// # Errors
///
/// Returns an error unless `input` is nonempty F32 `[B*S, D]`, all dimensions
/// are nonzero, `D` is divisible by `num_heads`, or runtime recording fails.
pub fn split_heads(
	input: &Matrix,
	batch: usize,
	sequence_length: usize,
	num_heads: usize,
) -> Result<Matrix> {
	let result = algebra::split_heads(input, batch, sequence_length, num_heads)?;
	if result.permuted {
		autograd::record_split_heads(input, &result.output, batch, sequence_length, num_heads)?;
	}
	Ok(result.output)
}

/// Invert [`split_heads`] into contiguous `[B*S, H*(D/H)]` storage.
///
/// The one-head case is a zero-copy differentiable view.
///
/// # Errors
///
/// Returns an error unless `input` is nonempty F32 `[B*H, S, D/H]` with the
/// supplied nonzero dimensions, or runtime recording fails.
pub fn merge_heads(
	input: &Matrix,
	batch: usize,
	sequence_length: usize,
	num_heads: usize,
) -> Result<Matrix> {
	let result = algebra::merge_heads(input, batch, sequence_length, num_heads)?;
	if result.permuted {
		autograd::record_merge_heads(input, &result.output, batch, sequence_length, num_heads)?;
	}
	Ok(result.output)
}

/// Compute `softmax(scores * scale + mask)` over the final axis.
///
/// The additive mask is detached from reverse mode.
///
/// # Errors
///
/// Returns an error unless both values are equal-shape, nonempty,
/// same-engine F32 matrices, or runtime recording fails.
pub fn softmax_scaled_masked(scores: &Matrix, mask: &Matrix, scale: f32) -> Result<Matrix> {
	let output = algebra::softmax_scaled_masked(scores, mask, scale)?;
	autograd::record_softmax_scaled_masked(scores, &output, scale)?;
	Ok(output)
}

/// Apply standard scaled dot-product attention to `[B*H, S, D/H]` values.
///
/// `additive_mask`, when present, is detached F32 `[B*H*S, S]` storage.
/// Causal visibility is applied without materializing another mask.
///
/// # Errors
///
/// Returns an error unless Q, K, and V are equal nonempty same-engine F32
/// rank-three matrices, the optional mask matches their flattened score rows,
/// or runtime recording fails.
pub fn scaled_dot_product_attention(
	query: &Matrix,
	key: &Matrix,
	value: &Matrix,
	additive_mask: Option<&Matrix>,
	scale: f32,
	causal: bool,
) -> Result<Matrix> {
	let result =
		algebra::scaled_dot_product_attention(query, key, value, additive_mask, scale, causal)?;
	autograd::record_scaled_dot_product_attention(
		query,
		key,
		value,
		&result.output,
		result.probabilities,
		scale,
	)?;
	Ok(result.output)
}

/// Apply the explicit causal Flash provider to `[BH,S,Dh]` values.
///
/// This compatibility entry point retains the same semantic scaled-dot-product
/// attention identity as [`scaled_dot_product_attention`]. It materializes only
/// the output and one FP32 log-sum-exp value per query row.
///
/// # Errors
///
/// Returns an error unless Q, K, and V are equal nonempty same-engine F32
/// matrices, the sequence length is at most 1024, or runtime recording fails.
pub fn flash_attention_causal(
	query: &Matrix,
	key: &Matrix,
	value: &Matrix,
	scale: f32,
) -> Result<Matrix> {
	let result = algebra::flash_attention_causal(query, key, value, scale)?;
	autograd::record_flash_attention(query, key, value, &result.output, result.log_sum_exp, scale)?;
	Ok(result.output)
}

/// Apply multi-head causal scaled dot-product attention to packed `[B*S, D]` values.
///
/// # Errors
///
/// Returns an error unless Q, K, and V are equal nonempty same-engine F32
/// matrices, their row count is divisible by `sequence_length`, their width is
/// divisible by `num_heads`, or runtime recording fails.
pub fn scaled_dot_product_attention_causal(
	query: &Matrix,
	key: &Matrix,
	value: &Matrix,
	sequence_length: usize,
	num_heads: usize,
) -> Result<Matrix> {
	let [rows, model_width] = query.shape() else {
		return Err(crate::Error::invalid_argument(
			"causal attention requires Q/K/V [B*S, D]",
		));
	};
	if *rows == 0
		|| *model_width == 0
		|| sequence_length == 0
		|| num_heads == 0
		|| !rows.is_multiple_of(sequence_length)
		|| !model_width.is_multiple_of(num_heads)
		|| key.shape() != query.shape()
		|| value.shape() != query.shape()
	{
		return Err(crate::Error::invalid_argument(
			"causal attention requires equal nonempty Q/K/V [B*S, D], rows divisible by S, and D divisible by H",
		));
	}
	let batch = rows / sequence_length;
	let query = split_heads(query, batch, sequence_length, num_heads)?;
	let key = split_heads(key, batch, sequence_length, num_heads)?;
	let value = split_heads(value, batch, sequence_length, num_heads)?;
	let scale = 1.0 / ((*model_width / num_heads) as f32).sqrt();
	let context = scaled_dot_product_attention(&query, &key, &value, None, scale, true)?;
	merge_heads(&context, batch, sequence_length, num_heads)
}
