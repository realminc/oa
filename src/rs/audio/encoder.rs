//! Stateful native audio encoding.

use crate::{Error, Result};

/// Elementary audio representation produced by [`AudioEncoder`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum AudioCodec {
	/// Signed little-endian 16-bit PCM samples.
	#[default]
	PcmS16,
}

/// Configuration for one streaming audio encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioEncodeProfile {
	/// Elementary codec written into each packet.
	pub codec: AudioCodec,
	/// Input sample rate in hertz.
	pub sample_rate: u32,
	/// Number of interleaved input channels.
	pub channel_count: u32,
	/// Preferred number of frames emitted per packet.
	pub frames_per_packet: u32,
}

impl Default for AudioEncodeProfile {
	fn default() -> Self {
		Self {
			codec: AudioCodec::PcmS16,
			sample_rate: 48_000,
			channel_count: 2,
			frames_per_packet: 1_024,
		}
	}
}

/// One encoded elementary-stream packet with frame-domain timing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedAudioPacket {
	/// Encoded packet bytes.
	pub bitstream: Vec<u8>,
	/// Zero-based input frame corresponding to the packet's first sample.
	pub presentation_frame: i64,
	/// Number of interleaved frames represented by the packet.
	pub duration_frames: u32,
}

/// Stateful deterministic PCM-S16 audio encoder.
///
/// Input is interleaved FP32. Complete configured packets are emitted by
/// [`encode`](Self::encode); [`flush`](Self::flush) emits a final partial
/// packet. [`close`](Self::close) deliberately discards an unflushed partial
/// packet, matching the donor lifecycle contract. Dropping the encoder only
/// releases host state and never flushes implicitly.
#[derive(Debug)]
pub struct AudioEncoder {
	state: Option<EncoderState>,
}

#[derive(Debug)]
struct EncoderState {
	profile: AudioEncodeProfile,
	channels: usize,
	packet_samples: usize,
	pending: Vec<f32>,
	next_input_frame: i64,
}

impl AudioEncoder {
	/// Create one open encoder with a checked profile.
	///
	/// # Errors
	///
	/// Returns an error unless the profile has a nonzero sample rate, one to
	/// eight channels, a nonzero packet size, and an addressable packet shape.
	pub fn create(profile: AudioEncodeProfile) -> Result<Self> {
		if profile.sample_rate == 0
			|| !(1..=8).contains(&profile.channel_count)
			|| profile.frames_per_packet == 0
		{
			return Err(Error::invalid_argument(
				"audio encoder requires a sample rate, 1..=8 channels, and a packet size",
			));
		}
		let channels = usize::try_from(profile.channel_count)
			.map_err(|_| Error::invalid_argument("audio channel count exceeds usize"))?;
		let packet_samples = usize::try_from(profile.frames_per_packet)
			.ok()
			.and_then(|frames| frames.checked_mul(channels))
			.ok_or_else(|| Error::invalid_argument("audio packet shape exceeds usize"))?;
		packet_samples
			.checked_mul(size_of::<i16>())
			.ok_or_else(|| Error::invalid_argument("audio packet byte size exceeds usize"))?;
		Ok(Self {
			state: Some(EncoderState {
				profile,
				channels,
				packet_samples,
				pending: Vec::new(),
				next_input_frame: 0,
			}),
		})
	}

	/// Append complete interleaved frames and emit every full packet now ready.
	///
	/// # Errors
	///
	/// Returns an error when the encoder is closed, the input is empty or does
	/// not contain complete frames, buffered allocation fails, or timing exceeds
	/// the signed frame domain.
	pub fn encode(&mut self, interleaved: &[f32]) -> Result<Vec<EncodedAudioPacket>> {
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("audio encoder is closed"))?;
		if interleaved.is_empty() || !interleaved.len().is_multiple_of(state.channels) {
			return Err(Error::invalid_argument(
				"audio encoder input must contain complete non-empty interleaved frames",
			));
		}
		state
			.pending
			.try_reserve(interleaved.len())
			.map_err(|_| Error::resource_exhausted("audio encoder input allocation failed"))?;
		state.pending.extend_from_slice(interleaved);
		emit_complete_packets(state)
	}

	/// Emit the final partial packet, if any.
	///
	/// Calling `flush` on a closed encoder is idempotent and returns no packets.
	///
	/// # Errors
	///
	/// Returns an error when packet allocation fails or timing exceeds the
	/// signed frame domain.
	pub fn flush(&mut self) -> Result<Vec<EncodedAudioPacket>> {
		let Some(state) = self.state.as_mut() else {
			return Ok(Vec::new());
		};
		if state.pending.is_empty() {
			return Ok(Vec::new());
		}
		let frames = u32::try_from(state.pending.len() / state.channels)
			.map_err(|_| Error::out_of_range("audio flush frame count exceeds u32"))?;
		Ok(vec![emit_packet(state, frames)?])
	}

	/// Discard any unflushed partial packet and close the encoder.
	pub fn close(&mut self) {
		self.state = None;
	}

	/// Return whether this encoder still accepts input.
	#[must_use]
	pub const fn is_open(&self) -> bool {
		self.state.is_some()
	}

	/// Return the active profile, or `None` after close.
	#[must_use]
	pub fn profile(&self) -> Option<&AudioEncodeProfile> {
		self.state.as_ref().map(|state| &state.profile)
	}

	/// Return codec initialization bytes.
	///
	/// PCM-S16 has no codec configuration record.
	#[must_use]
	pub const fn codec_config(&self) -> &[u8] {
		&[]
	}

	/// Return encoder-delay frames.
	///
	/// PCM-S16 introduces no priming delay.
	#[must_use]
	pub const fn priming_frames(&self) -> u32 {
		0
	}
}

fn emit_complete_packets(state: &mut EncoderState) -> Result<Vec<EncodedAudioPacket>> {
	let packet_count = state.pending.len() / state.packet_samples;
	let mut packets = Vec::new();
	packets
		.try_reserve_exact(packet_count)
		.map_err(|_| Error::resource_exhausted("audio packet list allocation failed"))?;
	for _ in 0..packet_count {
		packets.push(emit_packet(state, state.profile.frames_per_packet)?);
	}
	Ok(packets)
}

fn emit_packet(state: &mut EncoderState, frames: u32) -> Result<EncodedAudioPacket> {
	let samples = usize::try_from(frames)
		.ok()
		.and_then(|frames| frames.checked_mul(state.channels))
		.ok_or_else(|| Error::invalid_argument("audio packet sample count exceeds usize"))?;
	let byte_count = samples
		.checked_mul(size_of::<i16>())
		.ok_or_else(|| Error::invalid_argument("audio packet byte size exceeds usize"))?;
	let mut bitstream = Vec::new();
	bitstream
		.try_reserve_exact(byte_count)
		.map_err(|_| Error::resource_exhausted("audio packet allocation failed"))?;
	for &sample in &state.pending[..samples] {
		bitstream.extend_from_slice(&quantize_pcm_s16(sample).to_le_bytes());
	}
	let packet = EncodedAudioPacket {
		bitstream,
		presentation_frame: state.next_input_frame,
		duration_frames: frames,
	};
	state.next_input_frame = state
		.next_input_frame
		.checked_add(i64::from(frames))
		.ok_or_else(|| Error::out_of_range("audio presentation frame exceeds i64"))?;
	state.pending.drain(..samples);
	Ok(packet)
}

fn quantize_pcm_s16(sample: f32) -> i16 {
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
	// f32::round follows the donor's lround behavior for finite values here:
	// halfway cases are rounded away from zero.
	scaled.round() as i16
}
