//! Synchronous CPU decode and WAV-F32 host boundaries.

use std::{fs::File, io::Cursor, path::Path};

use symphonia::core::{
	codecs::audio::AudioDecoderOptions,
	formats::{FormatOptions, TrackType, probe::Hint},
	io::{MediaSource, MediaSourceStream},
	meta::MetadataOptions,
};

use crate::{Engine, Error, Result};

use super::{Audio, AudioChannelLayout, value::validate_sample_buffer};

/// Decode one WAV/PCM, FLAC, or MP3 file and upload planar FP32 samples.
///
/// This is a synchronous CPU codec boundary followed by device upload. It does
/// not introduce a decoder session or a hidden execution backend.
///
/// # Errors
///
/// Returns an error for an empty path, file I/O failure, unsupported or malformed
/// media, changing stream parameters, empty decoded audio, checked-size overflow,
/// or device allocation/upload failure.
pub fn decode_file(engine: &Engine, path: impl AsRef<Path>) -> Result<Audio> {
	let path = path.as_ref();
	if path.as_os_str().is_empty() {
		return Err(Error::invalid_argument("audio::decode_file path is empty"));
	}
	let file = File::open(path).map_err(|source| Error::io("audio file open", source))?;
	let mut hint = Hint::new();
	if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
		hint.with_extension(extension);
	}
	decode_source(engine, Box::new(file), &hint)
}

/// Decode one in-memory WAV/PCM, FLAC, or MP3 stream and upload planar FP32
/// samples.
///
/// # Errors
///
/// Returns an error for empty, unsupported, malformed, or parameter-changing
/// media, empty decoded audio, checked-size overflow, or device
/// allocation/upload failure.
pub fn decode_memory(engine: &Engine, encoded: &[u8]) -> Result<Audio> {
	if encoded.is_empty() {
		return Err(Error::invalid_argument(
			"audio::decode_memory input buffer is empty",
		));
	}
	decode_source(
		engine,
		Box::new(Cursor::new(encoded.to_vec())),
		&Hint::new(),
	)
}

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

fn decode_source(engine: &Engine, source: Box<dyn MediaSource>, hint: &Hint) -> Result<Audio> {
	let stream = MediaSourceStream::new(source, Default::default());
	let mut format = symphonia::default::get_probe()
		.probe(
			hint,
			stream,
			FormatOptions::default(),
			MetadataOptions::default(),
		)
		.map_err(|source| Error::backend_failure("Symphonia", "audio format probe", source))?;
	let track = format.default_track(TrackType::Audio).ok_or_else(|| {
		Error::invalid_argument("audio stream does not contain a decodable audio track")
	})?;
	let codec_parameters = track
		.codec_params
		.as_ref()
		.and_then(|parameters| parameters.audio())
		.ok_or_else(|| Error::invalid_argument("audio track has no codec parameters"))?;
	let mut decoder = symphonia::default::get_codecs()
		.make_audio_decoder(codec_parameters, &AudioDecoderOptions::default())
		.map_err(|source| Error::backend_failure("Symphonia", "audio decoder creation", source))?;
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

	let (channels, sample_rate) = stream_spec.ok_or_else(|| {
		Error::invalid_argument("audio stream did not produce any decoded samples")
	})?;
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
	Audio::from_planar_f32(
		engine,
		&planar,
		channel_count,
		sample_rate,
		AudioChannelLayout::for_channel_count(channel_count),
	)
}
