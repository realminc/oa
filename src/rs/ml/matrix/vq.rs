//! Vector-quantization Matrix primitives.

use crate::{Matrix, Result};

use super::super::lowering::vq as dispatch;

/// Nearest-code indices and gathered code vectors.
pub struct VqAssignResult {
	/// I32 nearest-code index for every latent row.
	pub indices: Matrix,
	/// Gathered winning code for every latent row.
	pub quantized: Matrix,
}

/// Functionally updated persistent EMA state.
pub struct VqEmaState {
	/// Next per-code EMA sum.
	pub embed_sum: Matrix,
	/// Next per-code EMA population.
	pub cluster_size: Matrix,
	/// Next codebook derived from the EMA state.
	pub codebook: Matrix,
}

/// Return a zero-copy stop-gradient view.
///
/// The returned Matrix shares storage and forward values with `input`, but owns
/// a fresh semantic identity with no reverse-mode edge to `input`.
///
/// # Errors
///
/// Returns an error only if the input's existing shape metadata is invalid.
pub fn detach(input: &Matrix) -> Result<Matrix> {
	input.reshape_view(input.shape().to_vec())
}

/// Assign every FP32 latent row to its nearest FP32 codebook row.
///
/// Squared L2 distance is used and equal-distance ties select the lower code
/// index, matching OA C++. The hard assignment is detached from reverse mode.
///
/// # Errors
///
/// Returns an error unless inputs are nonempty same-engine FP32 `[N,D]` and
/// `[K,D]` matrices or recording/allocation fails.
pub fn vq_assign(latent: &Matrix, codebook: &Matrix) -> Result<VqAssignResult> {
	let result = dispatch::assign(latent, codebook)?;
	Ok(VqAssignResult {
		indices: result.indices,
		quantized: result.quantized,
	})
}

/// Gather codebook rows selected by I32 token indices.
///
/// Invalid indices yield zero rows without an out-of-bounds read. This pure
/// decode operation is detached from reverse mode because EMA owns codebook
/// learning.
///
/// # Errors
///
/// Returns an error unless the codebook is FP32 `[K,D]`, indices are I32 `[N]`,
/// both belong to one engine, or recording/allocation fails.
pub fn vq_lookup(codebook: &Matrix, indices: &Matrix) -> Result<Matrix> {
	dispatch::lookup(codebook, indices)
}

/// Produce the next VQ EMA state without mutating input Matrix values.
///
/// This preserves the donor update equations, optional unit-RMS codebook
/// normalization, and exact dead-code revival hash. The owning module installs
/// the returned values into its stable persistent buffer handles.
///
/// # Errors
///
/// Returns an error for incompatible shapes, dtypes, engines, invalid numerical
/// options, ABI overflow, allocation failure, or runtime recording failure.
#[allow(
	clippy::too_many_arguments,
	reason = "publicly preserves the donor VQ state transition"
)]
pub fn vq_ema_update(
	latent: &Matrix,
	indices: &Matrix,
	embed_sum: &Matrix,
	cluster_size: &Matrix,
	codebook: &Matrix,
	decay: f32,
	epsilon: f32,
	dead_threshold: f32,
	seed: u32,
	normalize: bool,
) -> Result<VqEmaState> {
	let state = dispatch::ema_update(
		latent,
		indices,
		embed_sum,
		cluster_size,
		codebook,
		decay,
		epsilon,
		dead_threshold,
		seed,
		normalize,
	)?;
	Ok(VqEmaState {
		embed_sum: state.embed_sum,
		cluster_size: state.cluster_size,
		codebook: state.codebook,
	})
}
