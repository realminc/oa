//! PCM-S16 encoding from interleaved FP32 samples.

use crate::{Error, Result};

pub(in crate::audio) fn encode_s16(samples: &[f32]) -> Result<Vec<u8>> {
	let byte_count = samples
		.len()
		.checked_mul(size_of::<i16>())
		.ok_or_else(|| Error::invalid_argument("audio packet byte size exceeds usize"))?;
	let mut bitstream = Vec::new();
	bitstream
		.try_reserve_exact(byte_count)
		.map_err(|_| Error::resource_exhausted("audio packet allocation failed"))?;
	for &sample in samples {
		bitstream.extend_from_slice(&quantize_s16(sample).to_le_bytes());
	}
	Ok(bitstream)
}

fn quantize_s16(sample: f32) -> i16 {
	if sample.is_nan() {
		return 0;
	}
	if sample >= 1.0 {
		return i16::MAX;
	}
	if sample <= -1.0 {
		return i16::MIN;
	}
	let scaled = if sample < 0.0 {
		sample * 32_768.0
	} else {
		sample * 32_767.0
	};
	// f32::round matches the donor's lround for these finite values.
	scaled.round() as i16
}
