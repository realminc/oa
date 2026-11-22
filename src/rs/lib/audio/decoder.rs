//! Host decoding into finite Audio values or incremental playback PCM.

use std::{
	io::Cursor,
	path::{Path, PathBuf},
};

use symphonia::core::{
	codecs::audio::{
		AudioCodecParameters, AudioDecoder as SymphoniaDecoder, AudioDecoderOptions,
		well_known::{CODEC_ID_AC3, CODEC_ID_EAC3},
	},
	formats::{FormatReader, SeekMode, SeekTo, TrackType, probe::Hint},
	units::{Time, TimeBase},
};

use crate::{Engine, Error, Result};

use super::{Audio, AudioChannelLayout, codec::ac3, demux, validate_sample_buffer};

/// Decode one supported file and upload planar FP32 samples.
///
/// This includes WAV/PCM, FLAC, MP3, and supported AAC-LC or Vorbis container tracks.
///
/// This is a synchronous CPU codec boundary followed by device upload. It does
/// not introduce a decoder session or a hidden execution backend.
///
/// # Errors
///
/// Returns an error for an empty path, file I/O failure, unsupported or malformed
/// media, changing stream parameters, empty decoded audio, checked-size overflow,
/// or device allocation/upload failure.
///
/// This variant decodes on this thread's default engine. An engine
/// initialization failure is returned before decoding starts.
pub fn decode_file(path: impl AsRef<Path>) -> Result<Audio> {
	let path = path.as_ref();
	decode_file_on(&Engine::current()?, path)
}
/// Decode one supported file and upload planar FP32 samples on a particular
/// engine.
///
/// This is the [`decode_file`] contract evaluated on `engine` instead of this
/// thread's default engine.
///
/// # Errors
///
/// Returns the same errors as [`decode_file`].
pub fn decode_file_on(engine: &Engine, path: impl AsRef<Path>) -> Result<Audio> {
	let path = path.as_ref();
	if path.as_os_str().is_empty() {
		return Err(Error::invalid_argument("audio::decode_file path is empty"));
	}
	decode_source(
		engine,
		demux::probe_file(path, "audio file open", "audio format probe")?,
	)
}

/// Decode one supported in-memory media stream and upload planar FP32 samples
/// on this thread's default engine.
///
/// # Errors
///
/// Returns an error when the default engine cannot be initialized or for
/// empty, unsupported, malformed, or parameter-changing media, empty decoded
/// audio, checked-size overflow, or device allocation/upload failure.
pub fn decode_memory(encoded: &[u8]) -> Result<Audio> {
	decode_memory_on(&Engine::current()?, encoded)
}

/// Decode one supported in-memory media stream and upload planar FP32 samples
/// on a particular engine.
///
/// # Errors
///
/// Returns an error for empty, unsupported, malformed, or parameter-changing
/// media, empty decoded audio, checked-size overflow, or device
/// allocation/upload failure.
pub fn decode_memory_on(engine: &Engine, encoded: &[u8]) -> Result<Audio> {
	if encoded.is_empty() {
		return Err(Error::invalid_argument(
			"audio::decode_memory input buffer is empty",
		));
	}
	let source = Box::new(Cursor::new(encoded.to_vec()));
	decode_source(
		engine,
		demux::probe(source, &Hint::new(), "audio format probe")?,
	)
}

fn decode_source(engine: &Engine, mut format: Box<dyn FormatReader>) -> Result<Audio> {
	let track = format.default_track(TrackType::Audio).ok_or_else(|| {
		Error::invalid_argument("audio stream does not contain a decodable audio track")
	})?;
	let codec_parameters = track
		.codec_params
		.as_ref()
		.and_then(|parameters| parameters.audio())
		.ok_or_else(|| Error::invalid_argument("audio track has no codec parameters"))?;
	let mut decoder = symphonia_decoder(codec_parameters, "audio decoder creation")?;
	let track_id = track.id;
	let mut interleaved = Vec::<f32>::new();
	let mut stream_spec = None::<(usize, u32)>;

	while let Some(packet) = format
		.next_packet()
		.map_err(|source| Error::backend_failure("Symphonia", "audio packet read", source))?
	{
		if packet.track_id != track_id {
			continue;
		}
		let decoded = decoder
			.decode(&packet)
			.map_err(|source| Error::backend_failure("Symphonia", "audio packet decode", source))?;
		let channels = decoded.spec().channels().count();
		let sample_rate = decoded.spec().rate();
		if channels == 0 || sample_rate == 0 {
			return Err(Error::invalid_argument(
				"decoded audio has zero channels or sample rate",
			));
		}
		let current_spec = (channels, sample_rate);
		if let Some(expected) = stream_spec {
			if current_spec != expected {
				return Err(Error::failed_precondition(format!(
					"audio stream parameters changed from {expected:?} to {current_spec:?}"
				)));
			}
		} else {
			stream_spec = Some(current_spec);
		}
		let chunk_len = decoded.samples_interleaved();
		let start = interleaved.len();
		interleaved
			.try_reserve(chunk_len)
			.map_err(|_| Error::resource_exhausted("decoded audio allocation failed"))?;
		interleaved.resize(start + chunk_len, 0.0);
		decoded.copy_to_slice_interleaved(&mut interleaved[start..]);
	}

	let (channels, sample_rate) = stream_spec
		.ok_or_else(|| Error::invalid_argument("audio stream did not produce any decoded samples"))?;
	upload_interleaved(engine, &interleaved, channels, sample_rate)
}

fn upload_interleaved(
	engine: &Engine,
	interleaved: &[f32],
	channel_count: usize,
	sample_rate: u32,
) -> Result<Audio> {
	let samples_per_channel = validate_sample_buffer(interleaved, channel_count, sample_rate)?;
	let mut planar = Vec::new();
	planar
		.try_reserve_exact(interleaved.len())
		.map_err(|_| Error::resource_exhausted("planar audio allocation failed"))?;
	planar.resize(interleaved.len(), 0.0);
	for sample in 0..samples_per_channel {
		for channel in 0..channel_count {
			planar[channel * samples_per_channel + sample] =
				interleaved[sample * channel_count + channel];
		}
	}
	Audio::new_from_planar_f32(
		engine,
		&planar,
		channel_count,
		sample_rate,
		AudioChannelLayout::for_channel_count(channel_count),
	)
}

fn symphonia_decoder(
	parameters: &AudioCodecParameters,
	operation: &'static str,
) -> Result<Box<dyn SymphoniaDecoder>> {
	symphonia::default::get_codecs()
		.make_audio_decoder(parameters, &AudioDecoderOptions::default())
		.map_err(|source| Error::backend_failure("Symphonia", operation, source))
}

pub(super) struct StreamingDecoder {
	path: PathBuf,
	format: Box<dyn FormatReader>,
	decoder: PcmDecoder,
	pub(super) track_id: u32,
	pub(super) channels: usize,
	pub(super) sample_rate: u32,
	discard_frames: usize,
	timeline: Option<TimeBase>,
	next_frame: u64,
	samples: Vec<f32>,
	exhausted: bool,
}

enum PcmDecoder {
	Symphonia(Box<dyn SymphoniaDecoder>),
	Dolby(Box<ac3::DolbyDecoder>),
}

impl StreamingDecoder {
	pub(super) fn open(path: &Path, timestamped: bool) -> Result<Option<(Self, u64)>> {
		let format = demux::probe_file(path, "AudioPlayer source open", "AudioPlayer probe")?;
		let Some(track) = format.default_track(TrackType::Audio) else {
			return Ok(None);
		};
		let parameters = track
			.codec_params
			.as_ref()
			.and_then(|parameters| parameters.audio())
			.ok_or_else(|| Error::invalid_argument("AudioPlayer track has no audio parameters"))?;

		let sample_rate = parameters
			.sample_rate
			.filter(|&rate| rate != 0)
			.ok_or_else(|| Error::invalid_argument("AudioPlayer track has no sample rate"))?;
		let duration_us = track
			.time_base
			.zip(track.duration)
			.and_then(|(base, duration)| base.calc_duration(duration))
			.and_then(|duration| u64::try_from(duration.as_nanos() / 1_000).ok())
			.or_else(|| {
				track
					.num_frames
					.map(|frames| frames_to_us(frames, sample_rate))
			})
			.or_else(|| {
				let info = format.media_info();
				info
					.time_base
					.zip(info.duration)
					.and_then(|(base, duration)| base.calc_duration(duration))
					.and_then(|duration| u64::try_from(duration.as_nanos() / 1_000).ok())
			})
			.unwrap_or(0);
		let track_id = track.id;
		let timeline = if timestamped {
			Some(
				track
					.time_base
					.ok_or_else(|| Error::data_loss("audio track has no time base"))?,
			)
		} else {
			None
		};
		let (decoder, channels) = if matches!(parameters.codec, CODEC_ID_AC3 | CODEC_ID_EAC3) {
			(
				PcmDecoder::Dolby(Box::new(ac3::DolbyDecoder::new(sample_rate))),
				2,
			)
		} else {
			let channels = parameters
				.channels
				.as_ref()
				.ok_or_else(|| Error::invalid_argument("AudioPlayer track has no channel declaration"))?
				.count();
			let decoder = symphonia_decoder(parameters, "AudioPlayer decoder")?;
			(PcmDecoder::Symphonia(decoder), channels)
		};
		Ok(Some((
			Self {
				path: path.to_owned(),
				format,
				decoder,
				track_id,
				channels,
				sample_rate,
				discard_frames: 0,
				timeline,
				next_frame: 0,
				samples: Vec::new(),
				exhausted: false,
			},
			duration_us,
		)))
	}

	pub(super) fn read_frames(
		&mut self,
		cancelled: impl Fn() -> bool,
	) -> Result<Option<(u64, &[f32])>> {
		loop {
			if cancelled() {
				return Ok(Some((0, &[])));
			}
			let Some(packet) = self
				.format
				.next_packet()
				.map_err(|source| Error::backend_failure("Symphonia", "AudioPlayer packet read", source))?
			else {
				self.exhausted = true;
				return Ok(None);
			};
			if packet.track_id != self.track_id {
				continue;
			}
			let decoded = match &mut self.decoder {
				PcmDecoder::Dolby(decoder) => {
					decoder.decode(&packet.data, &mut self.samples)?;
					None
				}
				PcmDecoder::Symphonia(decoder) => {
					let decoded = decoder
						.decode(&packet)
						.map_err(|source| Error::backend_failure("Symphonia", "AudioPlayer decode", source))?;
					if decoded.spec().channels().count() != self.channels
						|| decoded.spec().rate() != self.sample_rate
					{
						return Err(Error::failed_precondition(
							"AudioPlayer stream parameters changed during playback",
						));
					}
					Some(decoded)
				}
			};
			let count = decoded
				.as_ref()
				.map_or(self.samples.len(), |decoded| decoded.samples_interleaved());
			let frames = count / self.channels;
			if frames == 0 {
				continue;
			}
			let (gap, discard) = if let Some(base) = self.timeline {
				let time = base
					.calc_time(packet.pts)
					.ok_or_else(|| Error::out_of_range("audio packet timestamp exceeds Time"))?;
				let start = time.as_nanos() * i128::from(self.sample_rate) / 1_000_000_000;
				let delta = start - i128::from(self.next_frame);
				// Container ticks can be coarser than PCM (e.g. Matroska ms).
				// Preserve contiguous samples across timestamp quantization.
				let tolerance = (u64::from(base.numer.get()) * u64::from(self.sample_rate))
					.div_ceil(u64::from(base.denom.get()));
				let (gap, discard) = pcm_discontinuity(delta, frames, tolerance)?;
				self.next_frame = self
					.next_frame
					.checked_add(gap)
					.and_then(|next| next.checked_add((frames - discard) as u64))
					.ok_or_else(|| Error::out_of_range("audio packet exceeds PCM timeline"))?;
				(gap, discard)
			} else {
				let discard = frames.min(self.discard_frames);
				self.discard_frames -= discard;
				(0, discard)
			};
			if discard == frames {
				continue;
			}
			if let Some(decoded) = decoded {
				self
					.samples
					.try_reserve(count.saturating_sub(self.samples.len()))
					.map_err(|_| Error::resource_exhausted("AudioPlayer PCM scratch allocation failed"))?;
				self.samples.resize(count, 0.0);
				decoded.copy_to_slice_interleaved(&mut self.samples);
			}
			return Ok(Some((gap, &self.samples[discard * self.channels..])));
		}
	}

	pub(super) fn seek(&mut self, timestamp_us: u64) -> Result<()> {
		let is_dolby = matches!(&self.decoder, PcmDecoder::Dolby(_));
		if self.exhausted && !is_dolby {
			// ISO BMFF clears its current media-data atom at EOF.
			self.reopen()?;
		}
		let actual_us = self.seek_with_preroll(timestamp_us, is_dolby)?;
		match &mut self.decoder {
			PcmDecoder::Symphonia(decoder) => decoder.reset(),
			PcmDecoder::Dolby(decoder) => decoder.reset(),
		}
		let requested_frame = us_to_frames(timestamp_us, self.sample_rate);
		self.next_frame = requested_frame;
		if self.timeline.is_none() {
			let actual_frame = us_to_frames(actual_us, self.sample_rate);
			self.discard_frames =
				usize::try_from(requested_frame.saturating_sub(actual_frame)).unwrap_or(usize::MAX);
		}
		Ok(())
	}

	fn reopen(&mut self) -> Result<()> {
		self.format = demux::probe_file(&self.path, "AudioPlayer source open", "AudioPlayer probe")?;
		self.exhausted = false;
		Ok(())
	}

	fn seek_with_preroll(&mut self, target_us: u64, is_dolby: bool) -> Result<u64> {
		const PREROLL_US: u64 = 100_000;
		let mut requested = target_us.saturating_sub(PREROLL_US);
		for _ in 0..8 {
			if is_dolby {
				// Symphonia's Matroska reader can retain queued blocks across a
				// seek. Each retry needs a fresh demux position and packet queue.
				self.reopen()?;
				if requested == 0 {
					return Ok(0);
				}
			}
			let seeked = self
				.format
				.seek(
					SeekMode::Accurate,
					SeekTo::Time {
						time: Time::from_micros_u64(requested),
						track_id: Some(self.track_id),
					},
				)
				.map_err(|source| Error::backend_failure("Symphonia", "AudioPlayer seek", source))?;
			let actual = self
				.format
				.tracks()
				.iter()
				.find(|track| track.id == seeked.track_id)
				.and_then(|track| track.time_base)
				.and_then(|base| base.calc_time(seeked.actual_ts))
				.map_or(0, |time| {
					u64::try_from((time.as_nanos() / 1_000).max(0)).unwrap_or(u64::MAX)
				});
			if !is_dolby || actual <= target_us {
				return Ok(actual);
			}
			// Matroska's cues can point at a video block after the required
			// audio block. Move back by the observed overshoot plus preroll.
			let backoff = (actual - requested).saturating_add(PREROLL_US);
			requested = requested.saturating_sub(backoff);
		}
		Err(Error::data_loss(
			"audio seek could not land before the requested sample",
		))
	}
}

fn pcm_discontinuity(delta: i128, frames: usize, tolerance: u64) -> Result<(u64, usize)> {
	if delta.unsigned_abs() <= u128::from(tolerance) {
		return Ok((0, 0));
	}
	if delta < 0 {
		return Ok((
			0,
			usize::try_from(delta.unsigned_abs())
				.unwrap_or(usize::MAX)
				.min(frames),
		));
	}
	let silence = u64::try_from(delta)
		.map_err(|_| Error::out_of_range("audio timestamp gap exceeds PCM timeline"))?;
	Ok((silence, 0))
}

pub(super) fn frames_to_us(frames: u64, sample_rate: u32) -> u64 {
	u64::try_from(u128::from(frames) * 1_000_000 / u128::from(sample_rate)).unwrap_or(u64::MAX)
}

pub(super) fn us_to_frames(timestamp: u64, sample_rate: u32) -> u64 {
	u64::try_from(u128::from(timestamp) * u128::from(sample_rate) / 1_000_000).unwrap_or(u64::MAX)
}

#[cfg(test)]
#[path = "../../../../test/rs/audio/decoder_unit.rs"]
mod tests;
