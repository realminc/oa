//! Streaming ISO-BMFF/MP4 video muxing.

use std::{
	fs::File,
	io::{Seek, SeekFrom, Write},
	path::Path,
};

use crate::{Error, Result, audio::EncodedAudioPacket};

use super::{VideoCodec, VideoTimeBase};

const STREAM_HEADER_BYTES: u64 = 32;

/// Optional native PCM-S16 track configuration for [`VideoMuxer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoMuxerAudioConfig {
	/// Audio sample rate in hertz.
	pub sample_rate: u32,
	/// Number of interleaved channels.
	pub channel_count: u32,
	/// Encoder-delay frames omitted from presentation through an edit list.
	pub priming_frames: u32,
}

impl Default for VideoMuxerAudioConfig {
	fn default() -> Self {
		Self {
			sample_rate: 48_000,
			channel_count: 2,
			priming_frames: 0,
		}
	}
}

/// Configuration for one streaming MP4 muxer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoMuxerConfig {
	/// Elementary video codec. Only H.264 and H.265 are admitted.
	pub codec: VideoCodec,
	/// Coded width in pixels.
	pub width: u32,
	/// Coded height in pixels.
	pub height: u32,
	/// Nominal frame rate used for the final sample duration.
	pub frame_rate: u32,
	/// MP4 video-track time base. Its numerator must currently be one.
	pub time_base: VideoTimeBase,
	/// Optional native PCM-S16 audio track.
	pub audio: Option<VideoMuxerAudioConfig>,
}

impl VideoMuxerConfig {
	/// Construct the default 30-fps, 90-kHz video-only profile.
	///
	/// # Errors
	///
	/// Returns an error when the coded extent is zero.
	pub fn new(codec: VideoCodec, width: u32, height: u32) -> Result<Self> {
		if width == 0 || height == 0 {
			return Err(Error::invalid_argument(
				"video muxer coded extent must be non-zero",
			));
		}
		Ok(Self {
			codec,
			width,
			height,
			frame_rate: 30,
			time_base: VideoTimeBase::new(1, 90_000)?,
			audio: None,
		})
	}
}

/// One owned Annex-B video access unit with microsecond presentation timing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedVideoPacket {
	bitstream: Vec<u8>,
	presentation_timestamp_micros: u64,
	decode_timestamp_micros: u64,
	keyframe: bool,
}

impl EncodedVideoPacket {
	/// Construct a non-empty encoded access unit.
	///
	/// # Errors
	///
	/// Returns an error when `bitstream` is empty. The muxer validates Annex-B
	/// framing when the packet is written.
	pub fn new(
		bitstream: Vec<u8>,
		presentation_timestamp_micros: u64,
		keyframe: bool,
	) -> Result<Self> {
		if bitstream.is_empty() {
			return Err(Error::invalid_argument(
				"encoded video packet bitstream is empty",
			));
		}
		Ok(Self {
			bitstream,
			presentation_timestamp_micros,
			decode_timestamp_micros: presentation_timestamp_micros,
			keyframe,
		})
	}

	/// Construct an access unit with distinct presentation and decode timing.
	///
	/// Packets are written in decode order. Presentation timestamps may reorder;
	/// decode timestamps must increase when packets enter a muxer.
	///
	/// # Errors
	///
	/// Returns an error when `bitstream` is empty.
	pub fn with_timestamps(
		bitstream: Vec<u8>,
		presentation_timestamp_micros: u64,
		decode_timestamp_micros: u64,
		keyframe: bool,
	) -> Result<Self> {
		let mut packet = Self::new(bitstream, presentation_timestamp_micros, keyframe)?;
		packet.decode_timestamp_micros = decode_timestamp_micros;
		Ok(packet)
	}

	/// Borrow the encoded Annex-B access unit.
	pub fn bitstream(&self) -> &[u8] {
		&self.bitstream
	}

	/// Return the presentation timestamp in microseconds.
	pub const fn presentation_timestamp_micros(&self) -> u64 {
		self.presentation_timestamp_micros
	}

	/// Return the decode timestamp in microseconds.
	pub const fn decode_timestamp_micros(&self) -> u64 {
		self.decode_timestamp_micros
	}

	/// Return whether this packet is a random-access point.
	pub const fn is_keyframe(&self) -> bool {
		self.keyframe
	}

	/// Consume the packet and return its encoded bytes.
	pub fn into_bitstream(self) -> Vec<u8> {
		self.bitstream
	}
}

/// Stateful streaming MP4 muxer for H.264/H.265 and optional PCM-S16 audio.
///
/// Media payload is written as packets arrive. [`finalize`](Self::finalize)
/// appends the movie sample tables and closes the file. [`close`](Self::close)
/// deliberately abandons an unfinished file. Drop only releases the open file
/// handle; it never manufactures a movie trailer or reports success.
pub struct VideoMuxer {
	file: Option<File>,
	config: VideoMuxerConfig,
	mdat_payload_bytes: u64,
	video_offsets: Vec<u64>,
	video_sizes: Vec<u32>,
	video_presentation_timestamps_micros: Vec<u64>,
	video_decode_timestamps_micros: Vec<u64>,
	video_keyframes: Vec<bool>,
	audio_offsets: Vec<u64>,
	audio_durations: Vec<u32>,
	vps: Vec<u8>,
	sps: Vec<u8>,
	pps: Vec<u8>,
	finalized: bool,
}

impl VideoMuxer {
	/// Create a new streaming MP4 file and write its fixed header.
	///
	/// # Errors
	///
	/// Returns an error for an empty path, invalid or unsupported configuration,
	/// an unrepresentable MP4 field, or a file create/write failure.
	pub fn create(path: impl AsRef<Path>, config: VideoMuxerConfig) -> Result<Self> {
		let path = path.as_ref();
		validate_config(path, config)?;
		let mut file = File::create(path).map_err(|source| Error::io("MP4 output create", source))?;
		file
			.write_all(&stream_header())
			.map_err(|source| Error::io("MP4 stream header write", source))?;
		Ok(Self {
			file: Some(file),
			config,
			mdat_payload_bytes: 0,
			video_offsets: Vec::new(),
			video_sizes: Vec::new(),
			video_presentation_timestamps_micros: Vec::new(),
			video_decode_timestamps_micros: Vec::new(),
			video_keyframes: Vec::new(),
			audio_offsets: Vec::new(),
			audio_durations: Vec::new(),
			vps: Vec::new(),
			sps: Vec::new(),
			pps: Vec::new(),
			finalized: false,
		})
	}

	/// Install one AVC SPS/PPS pair, without Annex-B start codes.
	///
	/// # Errors
	///
	/// Returns an error for the wrong muxer codec, empty/oversized records, or
	/// records whose NAL unit types are not SPS and PPS.
	pub fn set_h264_codec_config(&mut self, sps: &[u8], pps: &[u8]) -> Result<()> {
		self.require_open()?;
		if self.config.codec != VideoCodec::H264 {
			return Err(Error::failed_precondition(
				"H.264 codec configuration requires an H.264 muxer",
			));
		}
		validate_nal_record(sps, 7, false, "H.264 SPS")?;
		validate_nal_record(pps, 8, false, "H.264 PPS")?;
		self.sps = copy_bytes(sps, "H.264 SPS")?;
		self.pps = copy_bytes(pps, "H.264 PPS")?;
		self.vps.clear();
		Ok(())
	}

	/// Install one HEVC VPS/SPS/PPS set, without Annex-B start codes.
	///
	/// # Errors
	///
	/// Returns an error for the wrong muxer codec, empty/oversized records, or
	/// records whose NAL unit types are not VPS, SPS, and PPS.
	pub fn set_h265_codec_config(&mut self, vps: &[u8], sps: &[u8], pps: &[u8]) -> Result<()> {
		self.require_open()?;
		if self.config.codec != VideoCodec::H265 {
			return Err(Error::failed_precondition(
				"H.265 codec configuration requires an H.265 muxer",
			));
		}
		validate_nal_record(vps, 32, true, "H.265 VPS")?;
		validate_nal_record(sps, 33, true, "H.265 SPS")?;
		validate_nal_record(pps, 34, true, "H.265 PPS")?;
		self.vps = copy_bytes(vps, "H.265 VPS")?;
		self.sps = copy_bytes(sps, "H.265 SPS")?;
		self.pps = copy_bytes(pps, "H.265 PPS")?;
		Ok(())
	}

	/// Stream one encoded video access unit into the MP4 media-data box.
	///
	/// # Errors
	///
	/// Returns an error after close/finalize, for malformed Annex-B input,
	/// non-increasing timestamps, metadata overflow/allocation failure, or I/O
	/// failure. An I/O failure closes and poisons this session.
	pub fn write_packet(&mut self, packet: &EncodedVideoPacket) -> Result<()> {
		self.require_open()?;
		if self
			.video_decode_timestamps_micros
			.last()
			.is_some_and(|last| packet.decode_timestamp_micros <= *last)
		{
			return Err(Error::invalid_argument(
				"video packet decode timestamps must be strictly increasing",
			));
		}
		let sample = annex_b_to_length_prefixed(&packet.bitstream)?;
		let sample_size = u32::try_from(sample.len())
			.map_err(|_| Error::out_of_range("MP4 video sample exceeds u32"))?;
		let offset = STREAM_HEADER_BYTES
			.checked_add(self.mdat_payload_bytes)
			.ok_or_else(|| Error::out_of_range("MP4 video sample offset exceeds u64"))?;
		let next_payload = self
			.mdat_payload_bytes
			.checked_add(u64::from(sample_size))
			.ok_or_else(|| Error::out_of_range("MP4 media-data size exceeds u64"))?;
		reserve_one(&mut self.video_offsets, "MP4 video offset table")?;
		reserve_one(&mut self.video_sizes, "MP4 video size table")?;
		reserve_one(
			&mut self.video_presentation_timestamps_micros,
			"MP4 video presentation timestamp table",
		)?;
		reserve_one(
			&mut self.video_decode_timestamps_micros,
			"MP4 video decode timestamp table",
		)?;
		reserve_one(&mut self.video_keyframes, "MP4 video keyframe table")?;
		if let Err(source) = self
			.file
			.as_mut()
			.ok_or_else(closed_error)?
			.write_all(&sample)
		{
			self.file = None;
			return Err(Error::io("MP4 video sample write", source));
		}
		self.video_offsets.push(offset);
		self.video_sizes.push(sample_size);
		self
			.video_presentation_timestamps_micros
			.push(packet.presentation_timestamp_micros);
		self
			.video_decode_timestamps_micros
			.push(packet.decode_timestamp_micros);
		self.video_keyframes.push(packet.keyframe);
		self.mdat_payload_bytes = next_payload;
		Ok(())
	}

	/// Stream one native PCM-S16 audio packet into the optional audio track.
	///
	/// # Errors
	///
	/// Returns an error after close/finalize, when no audio track was configured,
	/// for an inconsistent packet duration/byte count, overflow/allocation
	/// failure, or I/O failure. An I/O failure closes and poisons this session.
	pub fn write_audio_packet(&mut self, packet: &EncodedAudioPacket) -> Result<()> {
		self.require_open()?;
		let audio = self.config.audio.ok_or_else(|| {
			Error::failed_precondition("video muxer was not created with an audio track")
		})?;
		if packet.duration_frames == 0 {
			return Err(Error::invalid_argument(
				"audio packet duration must be non-zero",
			));
		}
		let expected = u64::from(packet.duration_frames)
			.checked_mul(u64::from(audio.channel_count))
			.and_then(|frames| frames.checked_mul(2))
			.ok_or_else(|| Error::out_of_range("PCM-S16 packet byte count exceeds u64"))?;
		if u64::try_from(packet.bitstream.len()).ok() != Some(expected) {
			return Err(Error::invalid_argument(
				"PCM-S16 packet byte count does not match duration and channel count",
			));
		}
		let offset = STREAM_HEADER_BYTES
			.checked_add(self.mdat_payload_bytes)
			.ok_or_else(|| Error::out_of_range("MP4 audio sample offset exceeds u64"))?;
		let next_payload = self
			.mdat_payload_bytes
			.checked_add(expected)
			.ok_or_else(|| Error::out_of_range("MP4 media-data size exceeds u64"))?;
		reserve_one(&mut self.audio_offsets, "MP4 audio offset table")?;
		reserve_one(&mut self.audio_durations, "MP4 audio duration table")?;
		if let Err(source) = self
			.file
			.as_mut()
			.ok_or_else(closed_error)?
			.write_all(&packet.bitstream)
		{
			self.file = None;
			return Err(Error::io("MP4 audio packet write", source));
		}
		self.audio_offsets.push(offset);
		self.audio_durations.push(packet.duration_frames);
		self.mdat_payload_bytes = next_payload;
		Ok(())
	}

	/// Finalize sample tables, append the movie box, flush, and close the file.
	///
	/// # Errors
	///
	/// Returns an error after close/finalize, for an empty stream, absent codec
	/// configuration, an unrepresentable sample table, allocation failure, or
	/// seek/write/flush failure. A finalization I/O failure closes the session.
	pub fn finalize(&mut self) -> Result<()> {
		self.require_open()?;
		if self.video_sizes.is_empty() {
			return Err(Error::failed_precondition("cannot finalize an empty video"));
		}
		if self.sps.is_empty()
			|| self.pps.is_empty()
			|| (self.config.codec == VideoCodec::H265 && self.vps.is_empty())
		{
			return Err(Error::failed_precondition(match self.config.codec {
				VideoCodec::H264 => "H.264 MP4 requires SPS and PPS codec configuration",
				VideoCodec::H265 => "H.265 MP4 requires VPS, SPS and PPS codec configuration",
				_ => "MP4 muxer codec is unsupported",
			}));
		}
		let moov = self.build_moov()?;
		let mdat_size = self
			.mdat_payload_bytes
			.checked_add(16)
			.ok_or_else(|| Error::out_of_range("MP4 mdat size exceeds u64"))?;
		let mut file = self.file.take().ok_or_else(closed_error)?;
		let io_result = (|| -> std::io::Result<()> {
			file.seek(SeekFrom::Start(24))?;
			file.write_all(&mdat_size.to_be_bytes())?;
			file.seek(SeekFrom::End(0))?;
			file.write_all(&moov)?;
			file.flush()
		})();
		if let Err(source) = io_result {
			return Err(Error::io("MP4 finalization", source));
		}
		self.finalized = true;
		Ok(())
	}

	/// Abandon the current output without adding a movie trailer.
	///
	/// A second close is idempotent. The incomplete output file is retained so
	/// callers can inspect or remove it explicitly.
	pub fn close(&mut self) {
		self.file = None;
	}

	/// Return the checked session configuration.
	pub const fn config(&self) -> &VideoMuxerConfig {
		&self.config
	}

	/// Return the number of video packets written so far.
	pub fn packet_count(&self) -> usize {
		self.video_sizes.len()
	}

	/// Return whether finalization completed successfully.
	pub const fn is_finalized(&self) -> bool {
		self.finalized
	}

	fn require_open(&self) -> Result<()> {
		if self.finalized {
			return Err(Error::failed_precondition("video muxer is finalized"));
		}
		if self.file.is_none() {
			return Err(closed_error());
		}
		Ok(())
	}

	fn build_moov(&self) -> Result<Vec<u8>> {
		let timescale = self.config.time_base.denominator();
		let deltas = sample_deltas(
			&self.video_decode_timestamps_micros,
			timescale,
			self.config.frame_rate,
		)?;
		let composition_offsets = composition_offsets(
			&self.video_presentation_timestamps_micros,
			&self.video_decode_timestamps_micros,
			timescale,
		)?;
		let video_duration = deltas.iter().try_fold(0_u32, |total, delta| {
			total
				.checked_add(*delta)
				.ok_or_else(|| Error::out_of_range("MP4 video duration exceeds u32"))
		})?;
		let audio_frames = self
			.audio_durations
			.iter()
			.try_fold(0_u64, |total, value| {
				total
					.checked_add(u64::from(*value))
					.ok_or_else(|| Error::out_of_range("MP4 audio duration exceeds u64"))
			})?;
		let audio_duration = if let Some(audio) = self.config.audio {
			let audible = audio_frames.saturating_sub(u64::from(audio.priming_frames));
			let ticks = audible
				.checked_mul(u64::from(timescale))
				.and_then(|value| value.checked_add(u64::from(audio.sample_rate / 2)))
				.ok_or_else(|| Error::out_of_range("MP4 audio movie duration exceeds u64"))?
				/ u64::from(audio.sample_rate);
			u32::try_from(ticks)
				.map_err(|_| Error::out_of_range("MP4 audio movie duration exceeds u32"))?
		} else {
			0
		};
		let mut payload = Vec::new();
		append(
			&mut payload,
			&movie_header(
				timescale,
				video_duration.max(audio_duration),
				!self.audio_offsets.is_empty(),
			)?,
		)?;
		append(
			&mut payload,
			&self.build_video_track(video_duration, &deltas, &composition_offsets)?,
		)?;
		if !self.audio_offsets.is_empty() {
			append(
				&mut payload,
				&build_audio_track(
					self
						.config
						.audio
						.ok_or_else(|| Error::internal("audio samples lost their track configuration"))?,
					&self.audio_offsets,
					&self.audio_durations,
					timescale,
				)?,
			)?;
		}
		make_box(*b"moov", &payload)
	}

	fn build_video_track(
		&self,
		duration: u32,
		deltas: &[u32],
		composition_offsets: &[i32],
	) -> Result<Vec<u8>> {
		let mut track = Vec::new();
		append(
			&mut track,
			&video_track_header(self.config.width, self.config.height, duration)?,
		)?;

		let mut media = Vec::new();
		append(
			&mut media,
			&media_header(self.config.time_base.denominator(), duration)?,
		)?;
		append(&mut media, &handler_box(*b"vide", b"VideoHandler\0")?)?;

		let mut information = Vec::new();
		append(&mut information, &video_media_header()?)?;
		append(
			&mut information,
			&self.build_video_sample_table(deltas, composition_offsets)?,
		)?;
		append(&mut media, &make_box(*b"minf", &information)?)?;
		append(&mut track, &make_box(*b"mdia", &media)?)?;
		make_box(*b"trak", &track)
	}

	fn build_video_sample_table(
		&self,
		deltas: &[u32],
		composition_offsets: &[i32],
	) -> Result<Vec<u8>> {
		let mut table = Vec::new();
		append(&mut table, &self.video_sample_description()?)?;
		append(&mut table, &time_to_sample(deltas)?)?;
		if composition_offsets.iter().any(|offset| *offset != 0) {
			append(
				&mut table,
				&composition_time_to_sample(composition_offsets)?,
			)?;
		}
		append(&mut table, &one_sample_per_chunk()?)?;
		append(&mut table, &sample_sizes(&self.video_sizes)?)?;
		append(&mut table, &chunk_offsets(&self.video_offsets)?)?;
		if self.video_keyframes.iter().any(|keyframe| *keyframe) {
			append(&mut table, &sync_samples(&self.video_keyframes)?)?;
		}
		make_box(*b"stbl", &table)
	}

	fn video_sample_description(&self) -> Result<Vec<u8>> {
		let codec_box = match self.config.codec {
			VideoCodec::H264 => build_avcc(&self.sps, &self.pps)?,
			VideoCodec::H265 => build_hvcc(&self.vps, &self.sps, &self.pps)?,
			_ => {
				return Err(Error::internal(
					"unsupported codec reached MP4 sample description",
				));
			}
		};
		let size = 86_usize
			.checked_add(codec_box.len())
			.ok_or_else(|| Error::out_of_range("MP4 video sample entry exceeds usize"))?;
		let mut entry = zeroed(size, "MP4 video sample entry")?;
		put_u32(&mut entry, 0, checked_u32(size, "MP4 video sample entry")?);
		entry[4..8].copy_from_slice(match self.config.codec {
			VideoCodec::H264 => b"avc1",
			VideoCodec::H265 => b"hvc1",
			_ => {
				return Err(Error::internal(
					"unsupported codec reached MP4 sample entry",
				));
			}
		});
		put_u16(&mut entry, 14, 1);
		put_u16(
			&mut entry,
			32,
			checked_u16(self.config.width, "MP4 video width")?,
		);
		put_u16(
			&mut entry,
			34,
			checked_u16(self.config.height, "MP4 video height")?,
		);
		put_u32(&mut entry, 36, 0x0048_0000);
		put_u32(&mut entry, 40, 0x0048_0000);
		put_u16(&mut entry, 48, 1);
		put_u16(&mut entry, 82, 0x18);
		put_u16(&mut entry, 84, u16::MAX);
		entry[86..].copy_from_slice(&codec_box);

		let mut payload = zeroed(8, "MP4 stsd prefix")?;
		put_u32(&mut payload, 4, 1);
		append(&mut payload, &entry)?;
		make_box(*b"stsd", &payload)
	}
}

fn validate_config(path: &Path, config: VideoMuxerConfig) -> Result<()> {
	if path.as_os_str().is_empty() {
		return Err(Error::invalid_argument("video muxer path is empty"));
	}
	if config.width == 0 || config.height == 0 || config.frame_rate == 0 {
		return Err(Error::invalid_argument(
			"video muxer requires a non-zero extent and frame rate",
		));
	}
	checked_u16(config.width, "MP4 video width")?;
	checked_u16(config.height, "MP4 video height")?;
	if config.time_base.numerator() != 1 {
		return Err(Error::invalid_argument(
			"MP4 video time-base numerator must currently be one",
		));
	}
	if !matches!(config.codec, VideoCodec::H264 | VideoCodec::H265) {
		return Err(Error::missing_capability(
			"MP4 muxing is implemented only for H.264 and H.265",
		));
	}
	if let Some(audio) = config.audio
		&& (audio.sample_rate == 0
			|| audio.sample_rate > u16::MAX.into()
			|| !(1..=8).contains(&audio.channel_count))
	{
		return Err(Error::invalid_argument(
			"MP4 PCM-S16 audio requires sample rate 1..=65535 and 1..=8 channels",
		));
	}
	Ok(())
}

fn closed_error() -> Error {
	Error::failed_precondition("video muxer is closed")
}

fn stream_header() -> [u8; STREAM_HEADER_BYTES as usize] {
	let mut header = [0_u8; STREAM_HEADER_BYTES as usize];
	put_u32(&mut header, 0, 16);
	header[4..8].copy_from_slice(b"ftyp");
	header[8..12].copy_from_slice(b"isom");
	put_u32(&mut header, 12, 512);
	put_u32(&mut header, 16, 1);
	header[20..24].copy_from_slice(b"mdat");
	header
}

fn annex_b_to_length_prefixed(bytes: &[u8]) -> Result<Vec<u8>> {
	let units = super::parse_nal_annex_b(bytes);
	if units.is_empty() {
		return Err(Error::data_loss(
			"MP4 video packet is not Annex-B framed or contains no NAL units",
		));
	}
	let total = units.iter().try_fold(0_usize, |total, unit| {
		let payload = trim_annex_b_trailing_zeroes(unit.payload());
		let length = u32::try_from(payload.len())
			.map_err(|_| Error::out_of_range("video NAL exceeds MP4 length field"))?;
		if length == 0 {
			return Err(Error::data_loss(
				"video access unit contains an empty NAL unit",
			));
		}
		total
			.checked_add(4)
			.and_then(|value| value.checked_add(payload.len()))
			.ok_or_else(|| Error::out_of_range("MP4 video sample exceeds usize"))
	})?;
	let mut output = Vec::new();
	output
		.try_reserve_exact(total)
		.map_err(|_| Error::resource_exhausted("MP4 video sample allocation failed"))?;
	for unit in units {
		let payload = trim_annex_b_trailing_zeroes(unit.payload());
		let length = u32::try_from(payload.len())
			.map_err(|_| Error::out_of_range("video NAL exceeds MP4 length field"))?;
		output.extend_from_slice(&length.to_be_bytes());
		output.extend_from_slice(payload);
	}
	Ok(output)
}

fn trim_annex_b_trailing_zeroes(mut payload: &[u8]) -> &[u8] {
	while payload.last() == Some(&0) {
		payload = &payload[..payload.len() - 1];
	}
	payload
}

fn validate_nal_record(bytes: &[u8], expected_type: u8, h265: bool, label: &str) -> Result<()> {
	if bytes.is_empty() || bytes.len() > usize::from(u16::MAX) {
		return Err(Error::invalid_argument(format!(
			"{label} must contain 1..=65535 bytes"
		)));
	}
	let nal_type = if h265 {
		(bytes[0] >> 1) & 0x3f
	} else {
		bytes[0] & 0x1f
	};
	if nal_type != expected_type {
		return Err(Error::invalid_argument(format!(
			"{label} has the wrong NAL unit type"
		)));
	}
	Ok(())
}

fn copy_bytes(bytes: &[u8], label: &str) -> Result<Vec<u8>> {
	let mut output = Vec::new();
	output
		.try_reserve_exact(bytes.len())
		.map_err(|_| Error::resource_exhausted(format!("{label} allocation failed")))?;
	output.extend_from_slice(bytes);
	Ok(output)
}

fn sample_deltas(timestamps: &[u64], timescale: u32, frame_rate: u32) -> Result<Vec<u32>> {
	let nominal = (timescale / frame_rate).max(1);
	let mut ticks = Vec::new();
	ticks
		.try_reserve_exact(timestamps.len())
		.map_err(|_| Error::resource_exhausted("MP4 timestamp conversion allocation failed"))?;
	for timestamp in timestamps {
		ticks.push(timestamp_to_ticks(*timestamp, timescale)?);
	}
	let mut deltas = Vec::new();
	deltas
		.try_reserve_exact(ticks.len())
		.map_err(|_| Error::resource_exhausted("MP4 sample delta allocation failed"))?;
	for pair in ticks.windows(2) {
		deltas.push(
			pair[1]
				.checked_sub(pair[0])
				.filter(|delta| *delta > 0)
				.ok_or_else(|| {
					Error::invalid_argument("video timestamps collapse in the configured MP4 time base")
				})?,
		);
	}
	if !timestamps.is_empty() {
		deltas.push(deltas.last().copied().unwrap_or(nominal));
	}
	Ok(deltas)
}

fn composition_offsets(
	presentation_timestamps: &[u64],
	decode_timestamps: &[u64],
	timescale: u32,
) -> Result<Vec<i32>> {
	if presentation_timestamps.len() != decode_timestamps.len() {
		return Err(Error::internal(
			"MP4 presentation and decode timestamp tables differ in length",
		));
	}
	let mut offsets = Vec::new();
	offsets
		.try_reserve_exact(presentation_timestamps.len())
		.map_err(|_| Error::resource_exhausted("MP4 composition offset allocation failed"))?;
	for (presentation, decode) in presentation_timestamps.iter().zip(decode_timestamps) {
		let presentation = i64::from(timestamp_to_ticks(*presentation, timescale)?);
		let decode = i64::from(timestamp_to_ticks(*decode, timescale)?);
		offsets.push(
			i32::try_from(presentation - decode)
				.map_err(|_| Error::out_of_range("MP4 composition offset exceeds i32"))?,
		);
	}
	Ok(offsets)
}

fn timestamp_to_ticks(timestamp_micros: u64, timescale: u32) -> Result<u32> {
	let value = timestamp_micros
		.checked_mul(u64::from(timescale))
		.and_then(|value| value.checked_add(500_000))
		.ok_or_else(|| Error::out_of_range("MP4 video timestamp conversion exceeds u64"))?
		/ 1_000_000;
	u32::try_from(value)
		.map_err(|_| Error::out_of_range("MP4 video timestamp exceeds version-zero time domain"))
}

fn movie_header(timescale: u32, duration: u32, has_audio: bool) -> Result<Vec<u8>> {
	let mut box_data = zeroed(108, "MP4 movie header")?;
	write_box_header(&mut box_data, *b"mvhd")?;
	put_u32(&mut box_data, 20, timescale);
	put_u32(&mut box_data, 24, duration);
	put_u32(&mut box_data, 28, 0x0001_0000);
	put_u16(&mut box_data, 32, 0x0100);
	put_identity_matrix(&mut box_data, 44);
	put_u32(&mut box_data, 104, if has_audio { 3 } else { 2 });
	Ok(box_data)
}

fn video_track_header(width: u32, height: u32, duration: u32) -> Result<Vec<u8>> {
	let mut box_data = zeroed(92, "MP4 video track header")?;
	write_box_header(&mut box_data, *b"tkhd")?;
	put_u32(&mut box_data, 8, 7);
	put_u32(&mut box_data, 20, 1);
	put_u32(&mut box_data, 28, duration);
	put_identity_matrix(&mut box_data, 48);
	put_u32(
		&mut box_data,
		84,
		width
			.checked_shl(16)
			.ok_or_else(|| Error::out_of_range("MP4 fixed-point width exceeds u32"))?,
	);
	put_u32(
		&mut box_data,
		88,
		height
			.checked_shl(16)
			.ok_or_else(|| Error::out_of_range("MP4 fixed-point height exceeds u32"))?,
	);
	Ok(box_data)
}

fn media_header(timescale: u32, duration: u32) -> Result<Vec<u8>> {
	let mut box_data = zeroed(32, "MP4 media header")?;
	write_box_header(&mut box_data, *b"mdhd")?;
	put_u32(&mut box_data, 20, timescale);
	put_u32(&mut box_data, 24, duration);
	put_u16(&mut box_data, 28, 0x55c4);
	Ok(box_data)
}

fn handler_box(kind: [u8; 4], name: &[u8]) -> Result<Vec<u8>> {
	let size = 32_usize
		.checked_add(name.len())
		.ok_or_else(|| Error::out_of_range("MP4 handler box exceeds usize"))?;
	let mut box_data = zeroed(size, "MP4 handler box")?;
	write_box_header(&mut box_data, *b"hdlr")?;
	box_data[16..20].copy_from_slice(&kind);
	box_data[32..].copy_from_slice(name);
	Ok(box_data)
}

fn video_media_header() -> Result<Vec<u8>> {
	let mut box_data = zeroed(20, "MP4 video media header")?;
	write_box_header(&mut box_data, *b"vmhd")?;
	put_u32(&mut box_data, 8, 1);
	Ok(box_data)
}

fn build_avcc(sps: &[u8], pps: &[u8]) -> Result<Vec<u8>> {
	let payload_size = 11_usize
		.checked_add(sps.len())
		.and_then(|value| value.checked_add(pps.len()))
		.ok_or_else(|| Error::out_of_range("avcC size exceeds usize"))?;
	let mut payload = zeroed(payload_size, "avcC payload")?;
	payload[0] = 1;
	if sps.len() >= 4 {
		payload[1..4].copy_from_slice(&sps[1..4]);
	} else {
		payload[1..4].copy_from_slice(&[0x42, 0xe0, 0x1e]);
	}
	payload[4] = 0xff;
	payload[5] = 0xe1;
	put_u16(&mut payload, 6, checked_u16_len(sps.len(), "H.264 SPS")?);
	payload[8..8 + sps.len()].copy_from_slice(sps);
	let pps_offset = 8 + sps.len();
	payload[pps_offset] = 1;
	put_u16(
		&mut payload,
		pps_offset + 1,
		checked_u16_len(pps.len(), "H.264 PPS")?,
	);
	payload[pps_offset + 3..].copy_from_slice(pps);
	make_box(*b"avcC", &payload)
}

fn build_hvcc(vps: &[u8], sps: &[u8], pps: &[u8]) -> Result<Vec<u8>> {
	let arrays_size = [vps, sps, pps].iter().try_fold(0_usize, |total, nal| {
		total
			.checked_add(5)
			.and_then(|value| value.checked_add(nal.len()))
			.ok_or_else(|| Error::out_of_range("hvcC arrays exceed usize"))
	})?;
	let mut payload = zeroed(
		23_usize
			.checked_add(arrays_size)
			.ok_or_else(|| Error::out_of_range("hvcC payload exceeds usize"))?,
		"hvcC payload",
	)?;
	payload[0] = 1;
	payload[1] = 1;
	put_u32(&mut payload, 2, 0x6000_0000);
	payload[12] = 123;
	payload[13] = 0xf0;
	payload[15] = 0xfc;
	payload[16] = 0xfd;
	payload[17] = 0xf8;
	payload[18] = 0xf8;
	payload[21] = 0x0f;
	payload[22] = 3;
	let mut offset = 23;
	for (nal_type, nal) in [(32_u8, vps), (33, sps), (34, pps)] {
		payload[offset] = 0x80 | nal_type;
		put_u16(&mut payload, offset + 1, 1);
		put_u16(
			&mut payload,
			offset + 3,
			checked_u16_len(nal.len(), "H.265 parameter set")?,
		);
		payload[offset + 5..offset + 5 + nal.len()].copy_from_slice(nal);
		offset += 5 + nal.len();
	}
	make_box(*b"hvcC", &payload)
}

fn time_to_sample(deltas: &[u32]) -> Result<Vec<u8>> {
	let mut runs: Vec<(u32, u32)> = Vec::new();
	for delta in deltas {
		if let Some((count, previous)) = runs.last_mut()
			&& previous == delta
		{
			*count = count
				.checked_add(1)
				.ok_or_else(|| Error::out_of_range("MP4 stts run count exceeds u32"))?;
		} else {
			reserve_one(&mut runs, "MP4 stts runs")?;
			runs.push((1, *delta));
		}
	}
	let mut payload = zeroed(
		8_usize
			.checked_add(
				runs
					.len()
					.checked_mul(8)
					.ok_or_else(|| Error::out_of_range("MP4 stts size exceeds usize"))?,
			)
			.ok_or_else(|| Error::out_of_range("MP4 stts size exceeds usize"))?,
		"MP4 stts",
	)?;
	put_u32(
		&mut payload,
		4,
		checked_u32(runs.len(), "MP4 stts run count")?,
	);
	for (index, (count, delta)) in runs.iter().enumerate() {
		put_u32(&mut payload, 8 + index * 8, *count);
		put_u32(&mut payload, 12 + index * 8, *delta);
	}
	make_box(*b"stts", &payload)
}

fn composition_time_to_sample(offsets: &[i32]) -> Result<Vec<u8>> {
	let mut runs: Vec<(u32, i32)> = Vec::new();
	for offset in offsets {
		if let Some((count, previous)) = runs.last_mut()
			&& previous == offset
		{
			*count = count
				.checked_add(1)
				.ok_or_else(|| Error::out_of_range("MP4 ctts run count exceeds u32"))?;
		} else {
			reserve_one(&mut runs, "MP4 ctts runs")?;
			runs.push((1, *offset));
		}
	}
	let mut payload = zeroed(
		8_usize
			.checked_add(
				runs
					.len()
					.checked_mul(8)
					.ok_or_else(|| Error::out_of_range("MP4 ctts size exceeds usize"))?,
			)
			.ok_or_else(|| Error::out_of_range("MP4 ctts size exceeds usize"))?,
		"MP4 ctts",
	)?;
	payload[0] = 1;
	put_u32(
		&mut payload,
		4,
		checked_u32(runs.len(), "MP4 ctts run count")?,
	);
	for (index, (count, offset)) in runs.iter().enumerate() {
		put_u32(&mut payload, 8 + index * 8, *count);
		put_u32(
			&mut payload,
			12 + index * 8,
			u32::from_be_bytes(offset.to_be_bytes()),
		);
	}
	make_box(*b"ctts", &payload)
}

fn one_sample_per_chunk() -> Result<Vec<u8>> {
	let mut payload = zeroed(20, "MP4 stsc")?;
	put_u32(&mut payload, 4, 1);
	put_u32(&mut payload, 8, 1);
	put_u32(&mut payload, 12, 1);
	put_u32(&mut payload, 16, 1);
	make_box(*b"stsc", &payload)
}

fn sample_sizes(sizes: &[u32]) -> Result<Vec<u8>> {
	let mut payload = zeroed(
		12_usize
			.checked_add(
				sizes
					.len()
					.checked_mul(4)
					.ok_or_else(|| Error::out_of_range("MP4 stsz exceeds usize"))?,
			)
			.ok_or_else(|| Error::out_of_range("MP4 stsz exceeds usize"))?,
		"MP4 stsz",
	)?;
	put_u32(
		&mut payload,
		8,
		checked_u32(sizes.len(), "MP4 video sample count")?,
	);
	for (index, size) in sizes.iter().enumerate() {
		put_u32(&mut payload, 12 + index * 4, *size);
	}
	make_box(*b"stsz", &payload)
}

fn chunk_offsets(offsets: &[u64]) -> Result<Vec<u8>> {
	let wide = offsets.iter().any(|offset| *offset > u64::from(u32::MAX));
	let stride = if wide { 8 } else { 4 };
	let mut payload = zeroed(
		8_usize
			.checked_add(
				offsets
					.len()
					.checked_mul(stride)
					.ok_or_else(|| Error::out_of_range("MP4 offset table exceeds usize"))?,
			)
			.ok_or_else(|| Error::out_of_range("MP4 offset table exceeds usize"))?,
		"MP4 offset table",
	)?;
	put_u32(
		&mut payload,
		4,
		checked_u32(offsets.len(), "MP4 chunk count")?,
	);
	for (index, offset) in offsets.iter().enumerate() {
		if wide {
			put_u64(&mut payload, 8 + index * 8, *offset);
		} else {
			put_u32(
				&mut payload,
				8 + index * 4,
				u32::try_from(*offset)
					.map_err(|_| Error::internal("32-bit MP4 chunk selection lost range"))?,
			);
		}
	}
	make_box(if wide { *b"co64" } else { *b"stco" }, &payload)
}

fn sync_samples(keyframes: &[bool]) -> Result<Vec<u8>> {
	let count = keyframes.iter().filter(|keyframe| **keyframe).count();
	let mut payload = zeroed(
		8_usize
			.checked_add(
				count
					.checked_mul(4)
					.ok_or_else(|| Error::out_of_range("MP4 stss exceeds usize"))?,
			)
			.ok_or_else(|| Error::out_of_range("MP4 stss exceeds usize"))?,
		"MP4 stss",
	)?;
	put_u32(
		&mut payload,
		4,
		checked_u32(count, "MP4 sync sample count")?,
	);
	let mut output = 0;
	for (index, keyframe) in keyframes.iter().enumerate() {
		if *keyframe {
			put_u32(
				&mut payload,
				8 + output * 4,
				checked_u32(index + 1, "MP4 sync sample index")?,
			);
			output += 1;
		}
	}
	make_box(*b"stss", &payload)
}

fn build_audio_track(
	config: VideoMuxerAudioConfig,
	offsets: &[u64],
	durations: &[u32],
	movie_timescale: u32,
) -> Result<Vec<u8>> {
	let media_duration_u64 = durations.iter().try_fold(0_u64, |total, duration| {
		total
			.checked_add(u64::from(*duration))
			.ok_or_else(|| Error::out_of_range("MP4 audio media duration exceeds u64"))
	})?;
	let media_duration = u32::try_from(media_duration_u64)
		.map_err(|_| Error::out_of_range("MP4 audio media duration exceeds u32"))?;
	let audible = media_duration_u64.saturating_sub(u64::from(config.priming_frames));
	let movie_duration = audible
		.checked_mul(u64::from(movie_timescale))
		.and_then(|value| value.checked_add(u64::from(config.sample_rate / 2)))
		.ok_or_else(|| Error::out_of_range("MP4 audio track duration exceeds u64"))?
		/ u64::from(config.sample_rate);
	let movie_duration = u32::try_from(movie_duration)
		.map_err(|_| Error::out_of_range("MP4 audio track duration exceeds u32"))?;

	let mut track = Vec::new();
	let mut tkhd = zeroed(92, "MP4 audio track header")?;
	write_box_header(&mut tkhd, *b"tkhd")?;
	put_u32(&mut tkhd, 8, 7);
	put_u32(&mut tkhd, 20, 2);
	put_u32(&mut tkhd, 28, movie_duration);
	put_u16(&mut tkhd, 44, 0x0100);
	put_identity_matrix(&mut tkhd, 48);
	append(&mut track, &tkhd)?;

	if config.priming_frames > 0 {
		let mut edit_payload = zeroed(20, "MP4 edit list")?;
		put_u32(&mut edit_payload, 4, 1);
		put_u32(&mut edit_payload, 8, movie_duration);
		put_u32(&mut edit_payload, 12, config.priming_frames);
		put_u16(&mut edit_payload, 16, 1);
		append(
			&mut track,
			&make_box(*b"edts", &make_box(*b"elst", &edit_payload)?)?,
		)?;
	}

	let mut media = Vec::new();
	append(
		&mut media,
		&media_header(config.sample_rate, media_duration)?,
	)?;
	append(&mut media, &handler_box(*b"soun", b"SoundHandler\0")?)?;
	let mut information = Vec::new();
	let mut smhd = zeroed(16, "MP4 sound media header")?;
	write_box_header(&mut smhd, *b"smhd")?;
	append(&mut information, &smhd)?;
	append(
		&mut information,
		&build_audio_sample_table(config, offsets, durations, media_duration)?,
	)?;
	append(&mut media, &make_box(*b"minf", &information)?)?;
	append(&mut track, &make_box(*b"mdia", &media)?)?;
	make_box(*b"trak", &track)
}

fn build_audio_sample_table(
	config: VideoMuxerAudioConfig,
	offsets: &[u64],
	durations: &[u32],
	media_duration: u32,
) -> Result<Vec<u8>> {
	let mut table = Vec::new();
	let mut entry = zeroed(50, "MP4 ipcm sample entry")?;
	put_u32(&mut entry, 0, 50);
	entry[4..8].copy_from_slice(b"ipcm");
	put_u16(&mut entry, 14, 1);
	put_u16(
		&mut entry,
		24,
		checked_u16(config.channel_count, "MP4 audio channel count")?,
	);
	put_u16(&mut entry, 26, 16);
	put_u32(
		&mut entry,
		32,
		config
			.sample_rate
			.checked_shl(16)
			.ok_or_else(|| Error::out_of_range("MP4 fixed-point audio rate exceeds u32"))?,
	);
	put_u32(&mut entry, 36, 14);
	entry[40..44].copy_from_slice(b"pcmC");
	entry[48] = 1;
	entry[49] = 16;
	let mut stsd_payload = zeroed(8, "MP4 audio stsd prefix")?;
	put_u32(&mut stsd_payload, 4, 1);
	append(&mut stsd_payload, &entry)?;
	append(&mut table, &make_box(*b"stsd", &stsd_payload)?)?;

	let mut stts_payload = zeroed(16, "MP4 audio stts")?;
	put_u32(&mut stts_payload, 4, 1);
	put_u32(&mut stts_payload, 8, media_duration);
	put_u32(&mut stts_payload, 12, 1);
	append(&mut table, &make_box(*b"stts", &stts_payload)?)?;

	let mut runs: Vec<(u32, u32)> = Vec::new();
	for (index, duration) in durations.iter().enumerate() {
		if runs
			.last()
			.is_some_and(|(_, previous)| previous == duration)
		{
			continue;
		}
		reserve_one(&mut runs, "MP4 audio chunk runs")?;
		runs.push((checked_u32(index + 1, "MP4 audio chunk index")?, *duration));
	}
	let mut stsc_payload = zeroed(
		8_usize
			.checked_add(
				runs
					.len()
					.checked_mul(12)
					.ok_or_else(|| Error::out_of_range("MP4 audio stsc exceeds usize"))?,
			)
			.ok_or_else(|| Error::out_of_range("MP4 audio stsc exceeds usize"))?,
		"MP4 audio stsc",
	)?;
	put_u32(
		&mut stsc_payload,
		4,
		checked_u32(runs.len(), "MP4 audio stsc run count")?,
	);
	for (index, (first_chunk, samples)) in runs.iter().enumerate() {
		put_u32(&mut stsc_payload, 8 + index * 12, *first_chunk);
		put_u32(&mut stsc_payload, 12 + index * 12, *samples);
		put_u32(&mut stsc_payload, 16 + index * 12, 1);
	}
	append(&mut table, &make_box(*b"stsc", &stsc_payload)?)?;

	let mut stsz_payload = zeroed(12, "MP4 audio stsz")?;
	put_u32(
		&mut stsz_payload,
		4,
		config
			.channel_count
			.checked_mul(2)
			.ok_or_else(|| Error::out_of_range("MP4 PCM frame size exceeds u32"))?,
	);
	put_u32(&mut stsz_payload, 8, media_duration);
	append(&mut table, &make_box(*b"stsz", &stsz_payload)?)?;
	append(&mut table, &chunk_offsets(offsets)?)?;
	make_box(*b"stbl", &table)
}

fn make_box(kind: [u8; 4], payload: &[u8]) -> Result<Vec<u8>> {
	let size = payload
		.len()
		.checked_add(8)
		.ok_or_else(|| Error::out_of_range("MP4 box size exceeds usize"))?;
	let mut output = Vec::new();
	output
		.try_reserve_exact(size)
		.map_err(|_| Error::resource_exhausted("MP4 box allocation failed"))?;
	output.extend_from_slice(&checked_u32(size, "MP4 box size")?.to_be_bytes());
	output.extend_from_slice(&kind);
	output.extend_from_slice(payload);
	Ok(output)
}

fn write_box_header(bytes: &mut [u8], kind: [u8; 4]) -> Result<()> {
	put_u32(bytes, 0, checked_u32(bytes.len(), "MP4 box size")?);
	bytes[4..8].copy_from_slice(&kind);
	Ok(())
}

fn append(output: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
	output
		.try_reserve(bytes.len())
		.map_err(|_| Error::resource_exhausted("MP4 metadata allocation failed"))?;
	output.extend_from_slice(bytes);
	Ok(())
}

fn zeroed(size: usize, label: &str) -> Result<Vec<u8>> {
	let mut output = Vec::new();
	output
		.try_reserve_exact(size)
		.map_err(|_| Error::resource_exhausted(format!("{label} allocation failed")))?;
	output.resize(size, 0);
	Ok(output)
}

fn reserve_one<T>(values: &mut Vec<T>, label: &str) -> Result<()> {
	values
		.try_reserve(1)
		.map_err(|_| Error::resource_exhausted(format!("{label} allocation failed")))
}

fn checked_u16(value: u32, label: &str) -> Result<u16> {
	u16::try_from(value).map_err(|_| Error::out_of_range(format!("{label} exceeds u16")))
}

fn checked_u16_len(value: usize, label: &str) -> Result<u16> {
	u16::try_from(value).map_err(|_| Error::out_of_range(format!("{label} exceeds u16")))
}

fn checked_u32(value: usize, label: &str) -> Result<u32> {
	u32::try_from(value).map_err(|_| Error::out_of_range(format!("{label} exceeds u32")))
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
	bytes[offset..offset + 2].copy_from_slice(&value.to_be_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
	bytes[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
	bytes[offset..offset + 8].copy_from_slice(&value.to_be_bytes());
}

fn put_identity_matrix(bytes: &mut [u8], offset: usize) {
	put_u32(bytes, offset, 0x0001_0000);
	put_u32(bytes, offset + 16, 0x0001_0000);
	put_u32(bytes, offset + 32, 0x4000_0000);
}

#[cfg(test)]
mod tests {
	use super::{annex_b_to_length_prefixed, sample_deltas};

	#[test]
	fn annex_b_samples_become_four_byte_lengths() -> crate::Result<()> {
		assert_eq!(
			annex_b_to_length_prefixed(&[0, 0, 1, 0x67, 1, 2, 0, 0, 0, 1, 0x68, 3])?,
			[0, 0, 0, 3, 0x67, 1, 2, 0, 0, 0, 2, 0x68, 3]
		);
		Ok(())
	}

	#[test]
	fn microsecond_timestamps_must_remain_distinct_in_the_track_time_base() -> crate::Result<()> {
		assert_eq!(
			sample_deltas(&[0, 33_333, 66_666], 90_000, 30)?,
			vec![3_000, 3_000, 3_000]
		);
		assert!(sample_deltas(&[0, 1], 1, 30).is_err());
		Ok(())
	}
}
