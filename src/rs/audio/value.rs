//! Whole-audio value and channel-layout metadata.

use crate::{DType, Engine, Error, Matrix, Result};

/// Speaker layout carried by an [`Audio`] value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioChannelLayout {
	/// One center channel.
	Mono,
	/// Left and right channels.
	Stereo,
	/// Left, right, and low-frequency-effects channels.
	Stereo21,
	/// 5.1 surround channels.
	Surround51,
	/// 7.1 surround channels.
	Surround71,
	/// The channel count does not determine an unambiguous speaker layout.
	Unknown,
}

impl AudioChannelLayout {
	/// Return the required channel count, or `None` for an unknown layout.
	#[must_use]
	pub const fn channel_count(self) -> Option<usize> {
		match self {
			Self::Mono => Some(1),
			Self::Stereo => Some(2),
			Self::Stereo21 => Some(3),
			Self::Surround51 => Some(6),
			Self::Surround71 => Some(8),
			Self::Unknown => None,
		}
	}

	/// Infer the unambiguous conventional layout for a channel count.
	///
	/// Three channels deliberately remain unknown because the count cannot
	/// distinguish L/R/LFE (2.1) from L/C/R (3.0).
	#[must_use]
	pub const fn for_channel_count(channels: usize) -> Self {
		match channels {
			1 => Self::Mono,
			2 => Self::Stereo,
			6 => Self::Surround51,
			8 => Self::Surround71,
			_ => Self::Unknown,
		}
	}
}

/// Whole-audio value backed by a planar FP32 `[channels, samples]` matrix.
///
/// Clones retain the same matrix storage and semantic audio metadata.
#[derive(Clone)]
pub struct Audio {
	data: Matrix,
	sample_rate: u32,
	layout: AudioChannelLayout,
}

impl Audio {
	/// Attach audio semantics to a planar FP32 matrix.
	///
	/// # Errors
	///
	/// Returns an error unless `data` is a non-empty rank-two FP32 matrix, the
	/// sample rate is non-zero, and a known layout agrees with the matrix's
	/// channel extent.
	pub fn new(data: Matrix, sample_rate: u32, layout: AudioChannelLayout) -> Result<Self> {
		validate_audio_matrix(&data, sample_rate, layout)?;
		Ok(Self {
			data,
			sample_rate,
			layout,
		})
	}

	/// Upload planar FP32 samples and attach audio metadata.
	///
	/// `samples` contains all samples for channel zero, followed by all samples
	/// for channel one, and so on.
	///
	/// # Errors
	///
	/// Returns an error for zero channels, an empty or incomplete planar
	/// buffer, a zero sample rate, incompatible layout metadata, checked-size
	/// overflow, or device allocation/upload failure.
	pub fn from_planar_f32(
		engine: &Engine,
		samples: &[f32],
		channel_count: usize,
		sample_rate: u32,
		layout: AudioChannelLayout,
	) -> Result<Self> {
		let samples_per_channel = validate_sample_buffer(samples, channel_count, sample_rate)?;
		if let Some(expected) = layout.channel_count()
			&& expected != channel_count
		{
			return Err(Error::invalid_argument(format!(
				"audio layout {layout:?} requires {expected} channels; received {channel_count}"
			)));
		}
		let data = Matrix::from_f32(engine, [channel_count, samples_per_channel], samples)?;
		Self::new(data, sample_rate, layout)
	}

	/// Return the zero-copy planar matrix view.
	#[must_use]
	pub const fn as_matrix(&self) -> &Matrix {
		&self.data
	}

	/// Return the number of channels.
	#[must_use]
	pub fn channels(&self) -> usize {
		self.data.shape()[0]
	}

	/// Return the sample count per channel.
	#[must_use]
	pub fn samples(&self) -> usize {
		self.data.shape()[1]
	}

	/// Return the sample rate in hertz.
	#[must_use]
	pub const fn sample_rate(&self) -> u32 {
		self.sample_rate
	}

	/// Return the semantic speaker layout.
	#[must_use]
	pub const fn layout(&self) -> AudioChannelLayout {
		self.layout
	}

	/// Return the clip duration in seconds.
	#[must_use]
	pub fn duration_seconds(&self) -> f64 {
		self.samples() as f64 / f64::from(self.sample_rate)
	}
}

pub(super) fn validate_sample_buffer(
	samples: &[f32],
	channel_count: usize,
	sample_rate: u32,
) -> Result<usize> {
	if sample_rate == 0 {
		return Err(Error::invalid_argument(
			"audio sample rate must be non-zero",
		));
	}
	if channel_count == 0 {
		return Err(Error::invalid_argument(
			"audio channel count must be non-zero",
		));
	}
	if samples.is_empty() || !samples.len().is_multiple_of(channel_count) {
		return Err(Error::invalid_argument(
			"audio samples must contain complete, non-empty frames",
		));
	}
	Ok(samples.len() / channel_count)
}

fn validate_audio_matrix(
	data: &Matrix,
	sample_rate: u32,
	layout: AudioChannelLayout,
) -> Result<()> {
	let [channels, samples] = data.shape() else {
		return Err(Error::invalid_argument(format!(
			"audio data must be rank two [channels, samples]; received {:?}",
			data.shape()
		)));
	};
	if *channels == 0 || *samples == 0 {
		return Err(Error::invalid_argument(
			"audio data must contain at least one channel and sample",
		));
	}
	if data.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"audio data must use f32 storage; received {}",
			data.dtype().token()
		)));
	}
	if sample_rate == 0 {
		return Err(Error::invalid_argument(
			"audio sample rate must be non-zero",
		));
	}
	if let Some(expected) = layout.channel_count()
		&& *channels != expected
	{
		return Err(Error::invalid_argument(format!(
			"audio layout {layout:?} requires {expected} channels; received {channels}"
		)));
	}
	Ok(())
}
