//! In-process AC-3 and E-AC-3 decoding and stereo playback.

mod syncframe;

use crate::{Error, Result};

pub(in crate::audio) struct DolbyDecoder {
	decoder: syncframe::Decoder,
	sample_rate: u32,
}

impl DolbyDecoder {
	pub(in crate::audio) fn new(sample_rate: u32) -> Self {
		Self {
			decoder: syncframe::Decoder::new(),
			sample_rate,
		}
	}

	pub(in crate::audio) fn reset(&mut self) {
		self.decoder.reset();
	}

	pub(in crate::audio) fn decode(&mut self, mut packet: &[u8], pcm: &mut Vec<f32>) -> Result<()> {
		pcm.clear();
		while !packet.is_empty() {
			let header = syncframe::parse_header(packet).map_err(codec_error)?;
			let frame = packet
				.get(..header.frame_bytes)
				.ok_or_else(|| Error::data_loss("truncated Dolby syncframe"))?;
			packet = &packet[frame.len()..];
			if !header.is_primary() {
				continue;
			}
			if header.sample_rate != self.sample_rate {
				return Err(Error::failed_precondition(
					"Dolby sample rate changed during playback",
				));
			}
			let decoded = self.decoder.decode(frame).map_err(codec_error)?;
			append_stereo(&decoded, pcm)?;
		}
		Ok(())
	}
}

fn codec_error(error: syncframe::Error) -> Error {
	match error {
		syncframe::Error::NoSync => Error::data_loss("Dolby syncword missing"),
		syncframe::Error::Unsupported(profile) => {
			Error::missing_capability(format!("Dolby decoder: {profile}"))
		}
		syncframe::Error::Invalid(reason) => {
			Error::data_loss(format!("invalid Dolby syncframe: {reason}"))
		}
	}
}

fn append_stereo(decoded: &syncframe::Decoded, pcm: &mut Vec<f32>) -> Result<()> {
	const MIX: f32 = std::f32::consts::FRAC_1_SQRT_2;
	const LEFT: [f32; 2] = [1.0, 0.0];
	const RIGHT: [f32; 2] = [0.0, 1.0];
	const CENTER: [f32; 2] = [MIX, MIX];

	let (fronts, weights): (usize, &[[f32; 2]]) = match decoded.header.acmod {
		0 | 2 => (2, &[LEFT, RIGHT]),
		1 => (1, &[[1.0, 1.0]]),
		3 => (3, &[LEFT, RIGHT, CENTER]),
		4 => (2, &[LEFT, RIGHT, CENTER]),
		5 => (3, &[LEFT, RIGHT, CENTER, CENTER]),
		6 => (2, &[LEFT, RIGHT, [MIX, 0.0], [0.0, MIX]]),
		7 => (3, &[LEFT, RIGHT, CENTER, [MIX, 0.0], [0.0, MIX]]),
		_ => return Err(Error::data_loss("invalid Dolby channel mode")),
	};
	let frames = decoded.header.samples();
	if decoded.channels.len() != decoded.header.channels()
		|| decoded
			.channels
			.iter()
			.any(|channel| channel.len() != frames)
	{
		return Err(Error::data_loss(
			"Dolby decoder returned an incomplete audio block",
		));
	}
	let [left, right] = weights
		.iter()
		.fold([0.0, 0.0], |[left, right], &[l, r]| [left + l, right + r]);
	let gain = left.max(right).recip();
	let start = pcm.len();
	let samples = frames * 2;
	pcm
		.try_reserve(samples)
		.map_err(|_| Error::resource_exhausted("Dolby playback PCM allocation failed"))?;
	pcm.resize(start + samples, 0.0);
	// The decoder returns WAV order with LFE immediately after the front
	// channels. The stereo speaker mix uses the full-bandwidth channels.
	let channels = decoded.channels[..fronts]
		.iter()
		.chain(decoded.channels[fronts + usize::from(decoded.header.lfeon)..].iter());
	for (channel, &[left, right]) in channels.zip(weights) {
		let left = left * gain;
		let right = right * gain;
		for (dst, &sample) in pcm[start..].as_chunks_mut::<2>().0.iter_mut().zip(channel) {
			dst[0] += sample * left;
			dst[1] += sample * right;
		}
	}
	Ok(())
}
