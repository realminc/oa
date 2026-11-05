//! Streaming MP4 recording session over the hardware video encoder.
//!
//! Donor: `source/cpp/include/oa/vision/videoRecorder.h` and
//! `source/cpp/lib/oa/vision/video/videoRecorder.cpp`.
//!
//! One recorder owns a [`VideoEncoder`], a [`VideoMuxer`], and an optional
//! [`AudioEncoder`]. Frames enter as packed RGBA8 bytes; interleaved F32 audio
//! enters independently and is aligned to the first video presentation
//! timestamp. `finalize` flushes the encoder and completes the container;
//! `close` abandons them deliberately and never manufactures a partial file
//! that appears finished.

use crate::audio::{AudioCaptureChunk, AudioCodec, AudioEncoder};
use crate::{Engine, Error, Result};

use super::{
	EncodedVideoPacket, VideoCodec, VideoColorMatrix, VideoEncodeConfig, VideoEncoder, VideoMuxer,
	VideoMuxerAudioConfig, VideoMuxerConfig, extract_pps, extract_pps_h265, extract_sps,
	extract_sps_h265, extract_vps_h265,
};
pub use crate::video::types::VideoRecorderConfig;

impl Default for VideoRecorderConfig {
	fn default() -> Self {
		Self {
			output_path: String::from("output.mp4"),
			encode: VideoEncodeConfig::new(super::VideoEncodeProfile::h264_high_420_8bit(), 0, 0),
			color_space: VideoColorMatrix::Unspecified,
			full_range: false,
			audio: None,
		}
	}
}

struct PendingAudioChunk {
	samples: Vec<f32>,
	pts_micros: u64,
}

struct RecorderState {
	encoder: VideoEncoder,
	muxer: VideoMuxer,
	audio_encoder: Option<AudioEncoder>,
	pending_audio: Vec<PendingAudioChunk>,
	audio_scratch: Vec<f32>,
	first_video_pts_micros: u64,
	next_audio_frame: u64,
	has_first_video_pts: bool,
	submitted_frame_count: u64,
	muxed_frame_count: u64,
	codec_config_written: bool,
	finalized: bool,
}

/// Stateful streaming MP4 recorder for hardware-encoded H.264/H.265 video.
///
/// `write_rgba` submits one packed RGBA8 frame asynchronously and muxes every
/// packet the slot ring has completed; `finalize` drains the remaining
/// encoder jobs and closes the container. Audio chunks arriving before the
/// first video frame are buffered until the first video presentation
/// timestamp establishes the shared timeline.
pub struct VideoRecorder {
	state: Option<RecorderState>,
	config: VideoRecorderConfig,
}

impl VideoRecorder {
	/// Create a recorder, its encoder session, and the output container.
	///
	/// The encode profile must carry a non-zero extent admitted by the
	/// engine's queried encode capabilities; audio, when configured, admits
	/// only the muxer's PCM-S16 track.
	///
	/// # Errors
	///
	/// Returns an error for an empty output path or zero extent, when the
	/// device does not admit the encode session, when the audio profile is
	/// not PCM-S16, or when the container cannot be created.
	pub fn new(engine: &Engine, config: VideoRecorderConfig) -> Result<Self> {
		if config.output_path.is_empty() {
			return Err(Error::invalid_argument(
				"VideoRecorder output path must not be empty",
			));
		}
		if config.encode.width == 0 || config.encode.height == 0 {
			return Err(Error::invalid_argument(
				"VideoRecorder requires a non-zero encode extent",
			));
		}
		let codec = encode_codec(&config);
		if let Some(profile) = &config.audio
			&& profile.codec != AudioCodec::PcmS16
		{
			return Err(Error::missing_capability(
				"the recorder's MP4 audio track admits PCM-S16 only",
			));
		}
		let encoder = VideoEncoder::new(engine, config.encode)?;
		let audio_encoder = config.audio.map(AudioEncoder::new).transpose()?;
		let mut mux_config = VideoMuxerConfig::new(codec, config.encode.width, config.encode.height)?;
		mux_config.frame_rate = config.encode.frame_rate;
		mux_config.audio = audio_encoder.as_ref().map(|encoder| VideoMuxerAudioConfig {
			sample_rate: encoder
				.profile()
				.map(|profile| profile.sample_rate)
				.unwrap_or_default(),
			channel_count: encoder
				.profile()
				.map(|profile| profile.channel_count)
				.unwrap_or_default(),
			priming_frames: encoder.priming_frames(),
		});
		let muxer = VideoMuxer::new(&config.output_path, mux_config)?;
		Ok(Self {
			state: Some(RecorderState {
				encoder,
				muxer,
				audio_encoder,
				pending_audio: Vec::new(),
				audio_scratch: Vec::new(),
				first_video_pts_micros: 0,
				next_audio_frame: 0,
				has_first_video_pts: false,
				submitted_frame_count: 0,
				muxed_frame_count: 0,
				codec_config_written: false,
				finalized: false,
			}),
			config,
		})
	}

	/// Submit one packed RGBA8 frame and mux every packet now complete.
	///
	/// `rgba` must contain exactly `encode.width * encode.height * 4` bytes.
	/// The encode job runs asynchronously on the session's slot ring; packets
	/// are muxed in completion order while `finalize` drains the tail.
	///
	/// # Errors
	///
	/// Returns an error after close or finalize, for a mismatched input size,
	/// or when submission, feedback, or muxing fails.
	pub fn write_rgba(&mut self, rgba: &[u8], pts_micros: u64) -> Result<()> {
		let (color_space, full_range) = color_selection(&self.config);
		let state = self.open_state()?;
		state.set_first_video_pts(pts_micros)?;
		let ready = state
			.encoder
			.submit_rgba(rgba, pts_micros, color_space, full_range)?;
		state.submitted_frame_count += 1;
		for packet in &ready {
			state.write_encoded(packet)?;
		}
		Ok(())
	}

	/// Append captured interleaved F32 audio aligned to the video timeline.
	///
	/// Sample format must match the configured audio profile. Chunks received
	/// before the first video frame are retained until the first video
	/// presentation timestamp arrives; gaps are filled with silence and
	/// overlapping samples are trimmed.
	///
	/// # Errors
	///
	/// Returns an error after close or finalize, when audio is not enabled,
	/// for a mismatched sample format or incomplete frames, or when encoding
	/// or muxing fails.
	pub fn write_audio(
		&mut self,
		interleaved: &[f32],
		sample_rate: u32,
		channel_count: u32,
		pts_micros: u64,
	) -> Result<()> {
		let profile = self
			.config
			.audio
			.ok_or_else(|| Error::failed_precondition("VideoRecorder audio was not enabled"))?;
		if sample_rate != profile.sample_rate || channel_count != profile.channel_count {
			return Err(Error::invalid_argument(
				"audio chunk format does not match the recorder profile",
			));
		}
		if interleaved.is_empty() || !interleaved.len().is_multiple_of(channel_count as usize) {
			return Err(Error::invalid_argument(
				"audio chunk contains incomplete interleaved frames",
			));
		}
		let state = self.open_state()?;
		if !state.has_first_video_pts {
			state.pending_audio.push(PendingAudioChunk {
				samples: interleaved.to_vec(),
				pts_micros,
			});
			return Ok(());
		}
		state.write_audio_aligned(interleaved, pts_micros)
	}

	/// Append one capture chunk through [`write_audio`](Self::write_audio).
	///
	/// # Errors
	///
	/// Returns an error under the same conditions as `write_audio`.
	pub fn write_audio_chunk(&mut self, chunk: &AudioCaptureChunk) -> Result<()> {
		self.write_audio(
			&chunk.interleaved,
			chunk.sample_rate,
			chunk.channel_count,
			chunk.presentation_timestamp_us,
		)
	}

	/// Drain the encoder, flush pending audio, and finalize the container.
	///
	/// Idempotent: a second call returns `Ok(())`. A recording with no muxed
	/// frame cannot be finalized — the file would carry a header and no
	/// samples.
	///
	/// # Errors
	///
	/// Returns an error after close, when any outstanding encode job fails,
	/// or when the container trailer cannot be written.
	pub fn finalize(&mut self) -> Result<()> {
		let Some(state) = self.state.as_mut() else {
			return Err(Error::failed_precondition("VideoRecorder is not open"));
		};
		if state.finalized {
			return Ok(());
		}
		for packet in &state.encoder.flush()? {
			state.write_encoded(packet)?;
		}
		if let Some(audio) = &mut state.audio_encoder {
			if !state.has_first_video_pts {
				state.pending_audio.clear();
			}
			for packet in &audio.flush()? {
				state.muxer.write_audio_packet(packet)?;
			}
		}
		if state.submitted_frame_count == 0 || state.muxed_frame_count == 0 {
			return Err(Error::failed_precondition(
				"VideoRecorder cannot finalize an empty recording",
			));
		}
		state.muxer.finalize()?;
		state.finalized = true;
		Ok(())
	}

	/// Abandon the recording and release encoder, muxer, and host state.
	///
	/// Closing an unfinished recording never writes a movie trailer; the
	/// output file remains at whatever was muxed so far. Call
	/// [`finalize`](Self::finalize) first when the recording is required.
	pub fn close(&mut self) {
		self.state = None;
	}

	/// Return whether the recorder still accepts video and audio input.
	#[must_use]
	pub fn is_open(&self) -> bool {
		self.state.as_ref().is_some_and(|state| !state.finalized)
	}

	/// Return the number of video frames submitted for encode.
	#[must_use]
	pub fn frame_count(&self) -> u64 {
		self
			.state
			.as_ref()
			.map_or(0, |state| state.submitted_frame_count)
	}

	/// Return the active configuration.
	#[must_use]
	pub const fn config(&self) -> &VideoRecorderConfig {
		&self.config
	}

	fn open_state(&mut self) -> Result<&mut RecorderState> {
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("VideoRecorder is not open"))?;
		if state.finalized {
			return Err(Error::failed_precondition("VideoRecorder is finalized"));
		}
		Ok(state)
	}
}

impl RecorderState {
	/// Mux one encoded packet, seeding the container's codec configuration
	/// from the first keyframe's embedded parameter sets.
	fn write_encoded(&mut self, packet: &EncodedVideoPacket) -> Result<()> {
		if !self.codec_config_written && packet.is_keyframe() {
			let bitstream = packet.bitstream();
			match self.muxer.config().codec {
				VideoCodec::H264 => {
					let sps = extract_sps(bitstream)
						.ok_or_else(|| Error::data_loss("encoded H.264 keyframe does not contain SPS/PPS"))?;
					let pps = extract_pps(bitstream)
						.ok_or_else(|| Error::data_loss("encoded H.264 keyframe does not contain SPS/PPS"))?;
					self.muxer.set_h264_codec_config(&sps, &pps)?;
				}
				VideoCodec::H265 => {
					let vps = extract_vps_h265(bitstream).ok_or_else(|| {
						Error::data_loss("encoded H.265 keyframe does not contain VPS/SPS/PPS")
					})?;
					let sps = extract_sps_h265(bitstream).ok_or_else(|| {
						Error::data_loss("encoded H.265 keyframe does not contain VPS/SPS/PPS")
					})?;
					let pps = extract_pps_h265(bitstream).ok_or_else(|| {
						Error::data_loss("encoded H.265 keyframe does not contain VPS/SPS/PPS")
					})?;
					self.muxer.set_h265_codec_config(&vps, &sps, &pps)?;
				}
				_ => {
					return Err(Error::missing_capability(
						"the recorder's container configuration supports H.264/H.265 only",
					));
				}
			}
			self.codec_config_written = true;
		}
		self.muxer.write_packet(packet)?;
		self.muxed_frame_count += 1;
		Ok(())
	}

	/// Establish the shared timeline's origin on the first video frame and
	/// release every audio chunk buffered while it was unknown.
	fn set_first_video_pts(&mut self, pts_micros: u64) -> Result<()> {
		if self.has_first_video_pts {
			return Ok(());
		}
		self.first_video_pts_micros = pts_micros;
		self.has_first_video_pts = true;
		let pending = std::mem::take(&mut self.pending_audio);
		for chunk in &pending {
			self.write_audio_aligned(&chunk.samples, chunk.pts_micros)?;
		}
		Ok(())
	}

	/// Encode `interleaved` at the timeline position `pts_micros` implies,
	/// filling gaps with silence and trimming overlap, then mux the packets.
	fn write_audio_aligned(&mut self, interleaved: &[f32], pts_micros: u64) -> Result<()> {
		let Some(audio) = &self.audio_encoder else {
			return Err(Error::failed_precondition(
				"VideoRecorder audio was not enabled",
			));
		};
		let channels = u64::from(
			audio
				.profile()
				.map(|profile| profile.channel_count)
				.unwrap_or_default(),
		);
		let sample_rate = i64::from(
			audio
				.profile()
				.map(|profile| profile.sample_rate)
				.unwrap_or_default(),
		);
		let input_frames = interleaved.len() as u64 / channels;
		let delta_micros = pts_micros
			.checked_sub(self.first_video_pts_micros)
			.map(i64::try_from)
			.transpose()
			.map_err(|_| Error::out_of_range("audio timeline offset exceeds i64"))?
			.unwrap_or_else(|| {
				-i64::try_from(self.first_video_pts_micros - pts_micros).unwrap_or(i64::MIN)
			});
		let desired_start = delta_micros
			.checked_mul(sample_rate)
			.ok_or_else(|| Error::out_of_range("audio timeline offset overflows"))?
			/ 1_000_000;
		let desired_end = desired_start + input_frames as i64;
		if desired_end <= self.next_audio_frame as i64 {
			return Ok(());
		}
		let mut trim_frames = 0_u64;
		if desired_start < self.next_audio_frame as i64 {
			trim_frames = self.next_audio_frame - desired_start as u64;
		}
		let audio = self.audio_encoder.as_mut().expect("audio checked");
		if desired_start > self.next_audio_frame as i64 {
			let gap_frames = desired_start as u64 - self.next_audio_frame;
			self
				.audio_scratch
				.resize(gap_frames as usize * channels as usize, 0.0);
			for packet in &audio.encode(&self.audio_scratch)? {
				self.muxer.write_audio_packet(packet)?;
			}
			self.next_audio_frame += gap_frames;
		}
		if trim_frames >= input_frames {
			return Ok(());
		}
		let accepted_frames = input_frames - trim_frames;
		let sample_offset = (trim_frames * channels) as usize;
		let sample_count = (accepted_frames * channels) as usize;
		for packet in &audio.encode(&interleaved[sample_offset..sample_offset + sample_count])? {
			self.muxer.write_audio_packet(packet)?;
		}
		self.next_audio_frame += accepted_frames;
		Ok(())
	}
}

/// Map a public color matrix to the conversion kernel's selector.
const fn color_selection(config: &VideoRecorderConfig) -> (u32, bool) {
	let space = match config.color_space {
		VideoColorMatrix::Bt601 => 0,
		VideoColorMatrix::Bt709 | VideoColorMatrix::Unspecified => 1,
		VideoColorMatrix::Bt2020 => 2,
	};
	(space, config.full_range)
}

fn encode_codec(config: &VideoRecorderConfig) -> VideoCodec {
	match config.encode.profile {
		super::VideoEncodeProfile::H264 { .. } => VideoCodec::H264,
		super::VideoEncodeProfile::H265 { .. } => VideoCodec::H265,
	}
}
