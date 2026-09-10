use crate::{Matrix, Result};

use crate::ml::{autograd, lowering::matrix as dispatch};

/// Apply split-half rotary position embeddings to `[tokens, heads * head_dim]`.
///
/// `position_offset` selects the first absolute position, making the same
/// operation usable for full sequences and future KV-cache extension.
///
/// # Errors
///
/// Returns an error unless the input is nonempty F32, `head_dim` is even, the
/// final width matches `num_heads * head_dim`, and `theta_base` is finite and
/// positive.
pub fn rope(
	input: &Matrix,
	num_heads: usize,
	head_dim: usize,
	theta_base: f32,
	position_offset: u32,
) -> Result<Matrix> {
	let output = dispatch::rope(input, num_heads, head_dim, theta_base, position_offset)?;
	autograd::record_rope(
		input,
		&output,
		num_heads,
		head_dim,
		theta_base,
		position_offset,
	)?;
	Ok(output)
}
