//! Synchronous encoding from host samples or semantic [`crate::Audio`] buffers.

use std::path::Path;

use crate::{Error, Result};

use super::super::{Audio, clip::validate_sample_buffer};

/// Encode non-empty interleaved FP32 frames as lossless IEEE-float WAV bytes.
///
/// # Errors
///
/// Returns an error for a zero sample rate or channel count, incomplete frames,
/// or values that exceed the RIFF/WAV 32-bit header limits.
pub fn encode_interleaved_wav_f32(
	samples: &[f32],
	sample_rate: u32,
	channel_count: usize,
) -> Result<Vec<u8>> {
	validate_sample_buffer(samples, channel_count, sample_rate)?;
	let channel_count_u16 = u16::try_from(channel_count).map_err(|_| {
		Error::invalid_argument("audio::encode_interleaved_wav_f32 channel count exceeds u16")
	})?;
	let data_bytes = samples
		.len()
		.checked_mul(size_of::<f32>())
		.ok_or_else(|| Error::invalid_argument("WAV sample byte size overflows usize"))?;
	let data_bytes_u32 = u32::try_from(data_bytes).map_err(|_| {
		Error::invalid_argument("audio::encode_interleaved_wav_f32 exceeds the RIFF size limit")
	})?;
	let riff_size = data_bytes_u32.checked_add(38).ok_or_else(|| {
		Error::invalid_argument("audio::encode_interleaved_wav_f32 exceeds the RIFF size limit")
	})?;
	let block_align = channel_count_u16.checked_mul(4).ok_or_else(|| {
		Error::invalid_argument("audio::encode_interleaved_wav_f32 block alignment overflows u16")
	})?;
	let byte_rate = sample_rate
		.checked_mul(u32::from(block_align))
		.ok_or_else(|| {
			Error::invalid_argument("audio::encode_interleaved_wav_f32 byte rate overflows u32")
		})?;
	let output_len = data_bytes
		.checked_add(46)
		.ok_or_else(|| Error::invalid_argument("WAV output size overflows usize"))?;
	let mut output = Vec::new();
	output
		.try_reserve_exact(output_len)
		.map_err(|_| Error::resource_exhausted("WAV output allocation failed"))?;
	output.extend_from_slice(b"RIFF");
	output.extend_from_slice(&riff_size.to_le_bytes());
	output.extend_from_slice(b"WAVEfmt ");
	output.extend_from_slice(&18_u32.to_le_bytes());
	output.extend_from_slice(&3_u16.to_le_bytes());
	output.extend_from_slice(&channel_count_u16.to_le_bytes());
	output.extend_from_slice(&sample_rate.to_le_bytes());
	output.extend_from_slice(&byte_rate.to_le_bytes());
	output.extend_from_slice(&block_align.to_le_bytes());
	output.extend_from_slice(&32_u16.to_le_bytes());
	output.extend_from_slice(&0_u16.to_le_bytes());
	output.extend_from_slice(b"data");
	output.extend_from_slice(&data_bytes_u32.to_le_bytes());
	for sample in samples {
		output.extend_from_slice(&sample.to_le_bytes());
	}
	debug_assert_eq!(output.len(), output_len);
	Ok(output)
}

/// Read planar device samples and encode lossless IEEE-float WAV bytes.
///
/// This is an explicit blocking host-observation boundary.
///
/// # Errors
///
/// Returns an error when readback fails or the resulting WAV exceeds checked
/// allocation or RIFF header limits.
pub fn encode_wav_f32(audio: &Audio) -> Result<Vec<u8>> {
	let planar = audio.as_matrix().read_f32()?;
	let channels = audio.channels();
	let samples = audio.samples();
	let element_count = channels
		.checked_mul(samples)
		.ok_or_else(|| Error::invalid_argument("audio shape overflows usize"))?;
	let mut interleaved = Vec::new();
	interleaved
		.try_reserve_exact(element_count)
		.map_err(|_| Error::resource_exhausted("WAV interleave allocation failed"))?;
	for sample in 0..samples {
		for channel in 0..channels {
			interleaved.push(planar[channel * samples + sample]);
		}
	}
	encode_interleaved_wav_f32(&interleaved, audio.sample_rate(), channels)
}

/// Encode and write one lossless IEEE-float WAV file.
///
/// This is an explicit blocking host-observation and filesystem boundary.
///
/// # Errors
///
/// Returns an error for an empty path, failed readback or encoding, or a host
/// filesystem write failure.
pub fn save_wav_f32(path: impl AsRef<Path>, audio: &Audio) -> Result<()> {
	let path = path.as_ref();
	if path.as_os_str().is_empty() {
		return Err(Error::invalid_argument("audio::save_wav_f32 path is empty"));
	}
	let encoded = encode_wav_f32(audio)?;
	std::fs::write(path, encoded).map_err(|source| Error::io("audio WAV write", source))
}
