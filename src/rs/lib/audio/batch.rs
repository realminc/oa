//! Variable-length batching helpers composed from existing operations.
//!
//! `Audio` remains a rank-two `[channels, samples]` value; a batch of
//! variable-length clips is represented by stacking clips as channel rows
//! after `pad_to`/`crop_to` equalizes lengths. Every per-row operation —
//! `stft`, `stft_complex`, `istft`, `mel_spectrogram`, `mix_snr`, and the
//! signal family — then batches across items for free, and [`length_mask`]
//! provides the row-wise mask losses and reductions need.

use crate::{Audio, Error, Matrix, Result, matrix};

use super::audio::AudioChannelLayout;

/// Right-pad `input` with silence to exactly `samples` per channel.
///
/// # Errors
///
/// Returns an error when `samples` is smaller than the input length; use
/// [`crop_to`] to shorten instead. Allocation or dispatch failures propagate.
pub fn pad_to(input: &Audio, samples: usize) -> Result<Audio> {
	let current = input.samples();
	if samples < current {
		return Err(Error::invalid_argument(format!(
			"audio::pad_to target {samples} is smaller than input length {current}"
		)));
	}
	if samples == current {
		return Audio::new(
			input.as_matrix().clone(),
			input.sample_rate(),
			input.layout(),
		);
	}
	let zeros = Matrix::allocate(
		input.as_matrix().runtime(),
		vec![input.channels(), samples - current],
		input.channels() * (samples - current),
		crate::DType::F32,
	)?;
	let padded = matrix::concat(&[input.as_matrix().clone(), zeros], 1)?;
	Audio::new(padded, input.sample_rate(), input.layout())
}

/// Crop `input` to its first `samples` samples per channel.
///
/// # Errors
///
/// Returns an error when `samples` is zero or exceeds the input length; use
/// [`pad_to`] to extend instead.
pub fn crop_to(input: &Audio, samples: usize) -> Result<Audio> {
	let current = input.samples();
	if samples == 0 || samples > current {
		return Err(Error::invalid_argument(format!(
			"audio::crop_to target {samples} must be in 1..={current}"
		)));
	}
	let cropped = matrix::slice(
		input.as_matrix(),
		1,
		0,
		i64::try_from(samples)
			.map_err(|_| Error::out_of_range("audio::crop_to samples exceeds i64"))?,
	)?;
	Audio::new(cropped, input.sample_rate(), input.layout())
}

/// Stack clips as consecutive channel rows, producing one batched
/// `[total_channels, samples]` audio.
///
/// All inputs must share sample rate, sample count, and channel layout;
/// use [`pad_to`] or [`crop_to`] first for variable-length sources. Item
/// `i`'s channels occupy the contiguous rows following items `0..i`, which
/// keeps per-item channel grouping recoverable from the layouts.
///
/// # Errors
///
/// Returns an error for an empty input list, mismatched metadata, or a
/// failing concatenate dispatch.
pub fn stack(inputs: &[Audio]) -> Result<Audio> {
	let Some(first) = inputs.first() else {
		return Err(Error::invalid_argument(
			"audio::stack requires at least one input",
		));
	};
	let (rate, samples, layout) = (first.sample_rate(), first.samples(), first.layout());
	for (index, input) in inputs.iter().enumerate().skip(1) {
		if input.sample_rate() != rate || input.samples() != samples || input.layout() != layout {
			return Err(Error::invalid_argument(format!(
				"audio::stack input {index} must share rate, samples, and layout"
			)));
		}
	}
	let matrices: Vec<Matrix> = inputs
		.iter()
		.map(|input| input.as_matrix().clone())
		.collect();
	let channels: usize = inputs.iter().map(Audio::channels).sum();
	let stacked = matrix::concat(&matrices, 0)?;
	Audio::new(
		stacked,
		rate,
		AudioChannelLayout::for_channel_count(channels),
	)
}

/// Build a per-row `0/1` length mask matching `reference`'s channel count and
/// sample width: `mask[r, n] = 1` when `n < lengths[r]`.
///
/// Pair with [`stack`]: one length entry per output row marks each clip's
/// true sample count so masked losses and reductions can ignore padding.
/// The mask is deterministic metadata and is uploaded like the mel
/// filterbank.
///
/// # Errors
///
/// Returns an error when `lengths` does not have one entry per channel or any
/// entry exceeds the reference's sample count.
pub fn length_mask(reference: &Audio, lengths: &[usize]) -> Result<Matrix> {
	if lengths.len() != reference.channels() {
		return Err(Error::invalid_argument(format!(
			"audio::length_mask expects {} lengths; found {}",
			reference.channels(),
			lengths.len()
		)));
	}
	let samples = reference.samples();
	if let Some(length) = lengths.iter().find(|length| **length > samples) {
		return Err(Error::invalid_argument(format!(
			"audio::length_mask entry {length} exceeds sample count {samples}"
		)));
	}
	let mut values = vec![0.0_f32; lengths.len() * samples];
	for (row, length) in lengths.iter().enumerate() {
		for value in values.iter_mut().skip(row * samples).take(*length) {
			*value = 1.0;
		}
	}
	Matrix::new_from_slice_handle(
		reference.as_matrix().runtime(),
		vec![lengths.len(), samples],
		&values,
	)
}
