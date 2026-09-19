//! Deterministic batching contracts for ALM tokenizer and prior training.
//!
//! These host-side descriptions are the Rust equivalents of `trainalm`'s
//! tokenizer-window and true-boundary language-model window builders. They do
//! not own an [`crate::Engine`] or submit work.

use std::path::PathBuf;

use crate::{Error, Result, sdk::data::HumanMl3dDataset};

mod prior;
mod tokenizer;
mod validation;
mod workflow;

pub use prior::{
	PriorConditioning, PriorTrainingConfig, PriorTrainingReport, PriorValidation, train_prior,
	train_prior_with_validation,
};
pub use tokenizer::{
	TokenizerTrainingConfig, TokenizerTrainingReport, TokenizerValidation, train_tokenizer,
	train_tokenizer_with_validation,
};
pub use validation::{
	PriorValidationConfig, PriorValidationReport, TokenizerValidationConfig,
	TokenizerValidationReport, evaluate_prior, evaluate_tokenizer,
};
pub use workflow::{
	AlmTrainingConfig, AlmTrainingReport, AlmValidation, tokenize_corpus, train_alm,
	train_alm_with_native_text, train_alm_with_validation,
};

/// Native `.oam` persistence policy for one ALM training stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageCheckpointConfig {
	/// Root directory beneath the stage's model directory.
	pub directory: PathBuf,
	/// Portable stage model name used in checkpoint filenames.
	pub model_name: String,
	/// Optional portable run/context suffix.
	pub context: String,
	/// Maximum retained incremental checkpoints; zero disables rotation.
	pub max_keep: usize,
	/// Save a resumable mid-epoch checkpoint every N steps; zero disables it.
	pub save_every: u64,
	/// Restore the highest-step incremental checkpoint before training.
	pub resume: bool,
	/// Restore the best validation/epoch checkpoint after the final step.
	pub restore_best: bool,
	/// Print checkpoint decisions through the standard callback.
	pub verbose: bool,
}

impl StageCheckpointConfig {
	/// Construct the donor-style bounded epoch checkpoint policy.
	pub fn new(directory: impl Into<PathBuf>, model_name: impl Into<String>) -> Self {
		Self {
			directory: directory.into(),
			model_name: model_name.into(),
			context: String::new(),
			max_keep: 5,
			save_every: 0,
			resume: false,
			restore_best: true,
			verbose: true,
		}
	}
}

/// One fixed-length motion window used to train the ALM tokenizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenizerWindow {
	/// Dataset clip index.
	pub clip: usize,
	/// First frame in the clip.
	pub start: usize,
}

/// One true-boundary next-token window used to train the ALM prior.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PriorWindow {
	/// Token-sequence index.
	pub sequence: usize,
	/// First pair in `[SOM, motion..., EOM]`.
	pub start: usize,
	/// Number of valid next-token pairs; the remainder is padded and masked.
	pub valid: usize,
}

/// Reserved token identifiers for an ALM prior vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PriorSpecialTokens {
	/// Start-of-motion token.
	pub som: i32,
	/// End-of-motion token.
	pub eom: i32,
	/// Padding token.
	pub pad: i32,
}

impl PriorSpecialTokens {
	/// Construct the donor vocabulary suffix from a codebook size.
	///
	/// # Errors
	///
	/// Returns an error when the codebook size cannot be represented by the
	/// prior's signed 32-bit token matrices.
	pub fn after_codebook(num_codes: usize) -> Result<Self> {
		let som = i32::try_from(num_codes)
			.map_err(|_| Error::invalid_argument("ALM code count exceeds i32"))?;
		let eom = som
			.checked_add(1)
			.ok_or_else(|| Error::invalid_argument("ALM EOM token exceeds i32"))?;
		let pad = eom
			.checked_add(1)
			.ok_or_else(|| Error::invalid_argument("ALM PAD token exceeds i32"))?;
		Ok(Self { som, eom, pad })
	}
}

/// Owned host batch for one ALM prior training or validation step.
#[derive(Clone, Debug, PartialEq)]
pub struct PriorBatch {
	/// Row-major `[batch, window_len]` input tokens.
	pub input_ids: Vec<i32>,
	/// Row-major `[batch, window_len]` next-token targets.
	pub target_ids: Vec<i32>,
	/// Row-major loss mask containing only zero or one.
	pub loss_mask: Vec<f32>,
	/// Number of unmasked target tokens.
	pub valid_count: usize,
	/// Number of rows.
	pub batch_size: usize,
	/// Tokens per row.
	pub window_len: usize,
}

/// Build the donor tokenizer windows: 50% overlap and one exact tail window.
///
/// Clips shorter than `sequence_len` do not produce a window. The returned
/// order is stable: clip order, then ascending frame offset. Shuffling remains
/// an explicit training-policy concern.
///
/// # Errors
///
/// Returns an error when `sequence_len` is zero or the number of windows
/// overflows addressable host memory.
pub fn build_tokenizer_windows(
	frame_counts: impl IntoIterator<Item = usize>,
	sequence_len: usize,
) -> Result<Vec<TokenizerWindow>> {
	if sequence_len == 0 {
		return Err(Error::invalid_argument(
			"ALM tokenizer sequence length must be positive",
		));
	}
	let stride = (sequence_len / 2).max(1);
	let mut windows = Vec::new();
	for (clip, frames) in frame_counts.into_iter().enumerate() {
		if frames < sequence_len {
			continue;
		}
		let tail = frames - sequence_len;
		let mut start = 0_usize;
		loop {
			windows.try_reserve(1).map_err(|_| {
				Error::resource_exhausted("ALM tokenizer window inventory exceeds host memory")
			})?;
			windows.push(TokenizerWindow { clip, start });
			if tail - start < stride {
				break;
			}
			start += stride;
		}
		if windows
			.last()
			.is_none_or(|window| window.start != tail || window.clip != clip)
		{
			windows.try_reserve(1).map_err(|_| {
				Error::resource_exhausted("ALM tokenizer window inventory exceeds host memory")
			})?;
			windows.push(TokenizerWindow { clip, start: tail });
		}
	}
	Ok(windows)
}

/// Build tokenizer windows from a loaded HumanML3D dataset.
///
/// # Errors
///
/// Returns the same errors as [`build_tokenizer_windows`].
pub fn build_dataset_tokenizer_windows(
	dataset: &HumanMl3dDataset,
	sequence_len: usize,
) -> Result<Vec<TokenizerWindow>> {
	build_tokenizer_windows(
		(0..dataset.len()).filter_map(|index| dataset.clip_frames(index)),
		sequence_len,
	)
}

/// Gather a cyclic fixed-size tokenizer batch from normalized dataset clips.
///
/// # Errors
///
/// Returns an error for an empty window inventory, zero batch size, mismatched
/// sequence geometry, invalid window, or allocation overflow.
pub fn gather_tokenizer_batch(
	dataset: &HumanMl3dDataset,
	windows: &[TokenizerWindow],
	cursor: usize,
	batch_size: usize,
	sequence_len: usize,
) -> Result<Vec<f32>> {
	let clips = (0..dataset.len())
		.map(|index| {
			dataset.clip_data(index).ok_or_else(|| {
				Error::internal("HumanML3D dataset lost a clip inside its advertised length")
			})
		})
		.collect::<Result<Vec<_>>>()?;
	gather_tokenizer_clip_batch(
		&clips,
		dataset.feature_dim(),
		windows,
		cursor,
		batch_size,
		sequence_len,
	)
}

/// Gather a cyclic tokenizer batch from row-major normalized motion clips.
///
/// # Errors
///
/// Returns an error for zero geometry, malformed clips, empty or invalid
/// windows, cursor/allocation overflow, or a window outside its source clip.
pub fn gather_tokenizer_clip_batch<T: AsRef<[f32]>>(
	clips: &[T],
	feature_dim: usize,
	windows: &[TokenizerWindow],
	cursor: usize,
	batch_size: usize,
	sequence_len: usize,
) -> Result<Vec<f32>> {
	if windows.is_empty() {
		return Err(Error::invalid_argument(
			"ALM tokenizer batch requires at least one window",
		));
	}
	if batch_size == 0 || sequence_len == 0 || feature_dim == 0 {
		return Err(Error::invalid_argument(
			"ALM tokenizer batch and sequence lengths must be positive",
		));
	}
	if clips.iter().any(|clip| {
		clip.as_ref().is_empty()
			|| !clip.as_ref().len().is_multiple_of(feature_dim)
			|| clip.as_ref().iter().any(|value| !value.is_finite())
	}) {
		return Err(Error::invalid_argument(
			"ALM tokenizer clips must contain finite complete feature rows",
		));
	}
	let row_len = sequence_len
		.checked_mul(feature_dim)
		.ok_or_else(|| Error::invalid_argument("ALM tokenizer row size overflows usize"))?;
	let value_count = batch_size
		.checked_mul(row_len)
		.ok_or_else(|| Error::invalid_argument("ALM tokenizer batch size overflows usize"))?;
	let mut batch = vec![0.0; value_count];
	for row in 0..batch_size {
		let index = cursor
			.checked_add(row)
			.ok_or_else(|| Error::invalid_argument("ALM tokenizer cursor overflows usize"))?
			% windows.len();
		let window = windows[index];
		let clip = clips
			.get(window.clip)
			.map(AsRef::as_ref)
			.ok_or_else(|| Error::invalid_argument("ALM tokenizer window has invalid clip index"))?;
		let source_start = window
			.start
			.checked_mul(feature_dim)
			.ok_or_else(|| Error::invalid_argument("ALM tokenizer frame offset overflows usize"))?;
		let source_end = source_start
			.checked_add(row_len)
			.ok_or_else(|| Error::invalid_argument("ALM tokenizer window size overflows usize"))?;
		let source = clip
			.get(source_start..source_end)
			.ok_or_else(|| Error::invalid_argument("ALM tokenizer window exceeds its source clip"))?;
		batch[row * row_len..(row + 1) * row_len].copy_from_slice(source);
	}
	Ok(batch)
}

/// Build true-boundary next-token windows for every motion-token sequence.
///
/// A short sequence produces one padded window. A long sequence produces one
/// full window for every valid start position and never fabricates SOM or EOM
/// at an interior boundary.
///
/// # Errors
///
/// Returns an error when `window_len` is zero, sequence length arithmetic
/// overflows, or the inventory exceeds addressable host memory.
pub fn build_prior_windows(sequences: &[Vec<i32>], window_len: usize) -> Result<Vec<PriorWindow>> {
	if window_len == 0 {
		return Err(Error::invalid_argument(
			"ALM prior window length must be positive",
		));
	}
	let mut windows = Vec::new();
	for (sequence, codes) in sequences.iter().enumerate() {
		let pairs = codes
			.len()
			.checked_add(1)
			.ok_or_else(|| Error::invalid_argument("ALM prior pair count overflows usize"))?;
		if pairs <= window_len {
			windows
				.try_reserve(1)
				.map_err(|_| Error::resource_exhausted("ALM prior window inventory exceeds host memory"))?;
			windows.push(PriorWindow {
				sequence,
				start: 0,
				valid: pairs,
			});
			continue;
		}
		let count = pairs - window_len + 1;
		windows
			.try_reserve(count)
			.map_err(|_| Error::resource_exhausted("ALM prior window inventory exceeds host memory"))?;
		for start in 0..count {
			windows.push(PriorWindow {
				sequence,
				start,
				valid: window_len,
			});
		}
	}
	Ok(windows)
}

/// Gather a cyclic fixed-size prior batch with true-boundary padding.
///
/// # Errors
///
/// Returns an error for empty windows, zero geometry, invalid window
/// descriptors, invalid motion token IDs, or allocation overflow.
pub fn gather_prior_batch(
	sequences: &[Vec<i32>],
	windows: &[PriorWindow],
	cursor: usize,
	batch_size: usize,
	window_len: usize,
	special: PriorSpecialTokens,
) -> Result<PriorBatch> {
	if windows.is_empty() {
		return Err(Error::invalid_argument(
			"ALM prior batch requires at least one window",
		));
	}
	if batch_size == 0 || window_len == 0 {
		return Err(Error::invalid_argument(
			"ALM prior batch and window lengths must be positive",
		));
	}
	if special.som == special.eom || special.som == special.pad || special.eom == special.pad {
		return Err(Error::invalid_argument(
			"ALM prior reserved tokens must be distinct",
		));
	}
	let value_count = batch_size
		.checked_mul(window_len)
		.ok_or_else(|| Error::invalid_argument("ALM prior batch size overflows usize"))?;
	let mut batch = PriorBatch {
		input_ids: vec![special.pad; value_count],
		target_ids: vec![special.pad; value_count],
		loss_mask: vec![0.0; value_count],
		valid_count: 0,
		batch_size,
		window_len,
	};
	for row in 0..batch_size {
		let index = cursor
			.checked_add(row)
			.ok_or_else(|| Error::invalid_argument("ALM prior cursor overflows usize"))?
			% windows.len();
		let window = windows[index];
		let codes = sequences
			.get(window.sequence)
			.ok_or_else(|| Error::invalid_argument("ALM prior window has invalid sequence index"))?;
		let pairs = codes
			.len()
			.checked_add(1)
			.ok_or_else(|| Error::invalid_argument("ALM prior pair count overflows usize"))?;
		if window.valid > window_len
			|| window.start > pairs
			|| window.valid > pairs.saturating_sub(window.start)
		{
			return Err(Error::invalid_argument(
				"ALM prior window exceeds its token sequence",
			));
		}
		for (offset, token) in codes.iter().copied().enumerate() {
			if token < 0 || token >= special.som {
				return Err(Error::invalid_argument(format!(
					"ALM motion token {token} at offset {offset} is outside [0, SOM)"
				)));
			}
		}
		for column in 0..window.valid {
			let output = row * window_len + column;
			batch.input_ids[output] = stream_token(codes, window.start + column, special)?;
			batch.target_ids[output] = stream_token(codes, window.start + column + 1, special)?;
			batch.loss_mask[output] = 1.0;
		}
		batch.valid_count = batch
			.valid_count
			.checked_add(window.valid)
			.ok_or_else(|| Error::invalid_argument("ALM prior valid count overflows usize"))?;
	}
	Ok(batch)
}

fn stream_token(codes: &[i32], index: usize, special: PriorSpecialTokens) -> Result<i32> {
	if index == 0 {
		return Ok(special.som);
	}
	if index == codes.len() + 1 {
		return Ok(special.eom);
	}
	codes
		.get(index - 1)
		.copied()
		.ok_or_else(|| Error::invalid_argument("ALM prior stream offset exceeds sequence"))
}
