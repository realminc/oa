//! Bounded MP4 video packet demuxing.

use std::{
	collections::BTreeSet,
	fs::File,
	io::{Read, Seek, SeekFrom},
	path::Path,
};

use crate::{Error, Result};

use super::{
	Av1Profile, H264PictureLayout, H264Profile, H265Profile, VideoChromaSubsampling,
	VideoComponentBitDepth, VideoDecodeProfile,
};

const MAX_MP4_TABLE_ENTRIES: usize = 8 * 1024 * 1024;
const MAX_MOOV_BYTES: usize = 256 * 1024 * 1024;
const MAX_PACKET_BYTES: usize = 256 * 1024 * 1024;

/// Compressed video codec carried by a stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoCodec {
	/// Advanced Video Coding / H.264.
	H264,
	/// High Efficiency Video Coding / H.265.
	H265,
	/// AV1.
	Av1,
	/// VP9.
	Vp9,
}

/// Container kind recognized by the current demux boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoContainerKind {
	/// ISO Base Media File Format / MP4.
	Mp4,
}

/// Rational stream time base measured in seconds per timestamp tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoTimeBase {
	numerator: u32,
	denominator: u32,
}

impl VideoTimeBase {
	/// Construct a non-zero rational time base.
	///
	/// # Errors
	///
	/// Returns an error when either component is zero.
	pub fn new(numerator: u32, denominator: u32) -> Result<Self> {
		if numerator == 0 || denominator == 0 {
			return Err(Error::invalid_argument(
				"video time-base numerator and denominator must be non-zero",
			));
		}
		Ok(Self {
			numerator,
			denominator,
		})
	}

	/// Return the numerator in seconds per tick.
	pub const fn numerator(self) -> u32 {
		self.numerator
	}

	/// Return the denominator in seconds per tick.
	pub const fn denominator(self) -> u32 {
		self.denominator
	}
}

/// Immutable metadata for the selected video track.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoContainerInfo {
	kind: VideoContainerKind,
	codec: VideoCodec,
	width: u32,
	height: u32,
	time_base: VideoTimeBase,
	duration: u64,
	sample_count: u32,
	track_id: u32,
	track_count: u32,
	decode_profile: Option<VideoDecodeProfile>,
}

impl VideoContainerInfo {
	/// Return the container kind.
	pub const fn kind(self) -> VideoContainerKind {
		self.kind
	}

	/// Return the selected video codec.
	pub const fn codec(self) -> VideoCodec {
		self.codec
	}

	/// Return the coded width.
	pub const fn width(self) -> u32 {
		self.width
	}

	/// Return the coded height.
	pub const fn height(self) -> u32 {
		self.height
	}

	/// Return the selected track time base.
	pub const fn time_base(self) -> VideoTimeBase {
		self.time_base
	}

	/// Return the track duration in time-base ticks.
	pub const fn duration(self) -> u64 {
		self.duration
	}

	/// Return the selected track sample count.
	pub const fn sample_count(self) -> u32 {
		self.sample_count
	}

	/// Return the selected MP4 track identifier.
	pub const fn track_id(self) -> u32 {
		self.track_id
	}

	/// Return the number of tracks declared by the movie box.
	pub const fn track_count(self) -> u32 {
		self.track_count
	}

	/// Return the exact Vulkan-queryable stream profile when representable.
	///
	/// VP9 and codec profiles outside the current Vulkan standard-video binding
	/// return `None`; this does not mean the container or bitstream is invalid.
	pub const fn decode_profile(self) -> Option<VideoDecodeProfile> {
		self.decode_profile
	}

	/// Return the average frame rate derived from samples and track duration.
	pub fn frame_rate(self) -> f64 {
		if self.duration == 0 {
			return 0.0;
		}
		f64::from(self.sample_count) * f64::from(self.time_base.denominator)
			/ (self.duration as f64 * f64::from(self.time_base.numerator))
	}
}

/// One owned compressed video access unit in the selected track time base.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoPacket {
	data: Vec<u8>,
	presentation_timestamp: u64,
	decode_timestamp: u64,
	duration: u32,
	keyframe: bool,
	track_id: u32,
}

impl VideoPacket {
	/// Borrow Annex-B H.264/H.265 or native AV1/VP9 access-unit bytes.
	pub fn data(&self) -> &[u8] {
		&self.data
	}

	/// Consume the packet and return its owned bytes.
	pub fn into_data(self) -> Vec<u8> {
		self.data
	}

	/// Return the presentation timestamp in track time-base ticks.
	pub const fn presentation_timestamp(&self) -> u64 {
		self.presentation_timestamp
	}

	/// Return the decode timestamp in track time-base ticks.
	pub const fn decode_timestamp(&self) -> u64 {
		self.decode_timestamp
	}

	/// Return the sample duration in track time-base ticks.
	pub const fn duration(&self) -> u32 {
		self.duration
	}

	/// Return whether this sample is a random-access point.
	pub const fn is_keyframe(&self) -> bool {
		self.keyframe
	}

	/// Return the source MP4 track identifier.
	pub const fn track_id(&self) -> u32 {
		self.track_id
	}
}

/// Stateful, seekable packet source for one unfragmented MP4 video track.
///
/// Metadata is bounded and validated at open. Media payload is read only for
/// the current sample. H.264/H.265 length-prefixed samples are normalized to
/// Annex-B and codec parameter sets are prepended to the first keyframe after
/// open or seek.
pub struct VideoDemuxer {
	file: Option<File>,
	info: VideoContainerInfo,
	samples: Vec<Sample>,
	current_sample: usize,
	nal_length_size: Option<usize>,
	codec_config: Vec<u8>,
	need_codec_config: bool,
	eos: bool,
}

impl VideoDemuxer {
	/// Open and validate one unfragmented MP4 video track.
	///
	/// # Errors
	///
	/// Returns an error for an empty or unreadable path, a malformed or
	/// unsupported container/codec, hostile table sizes, inconsistent sample
	/// metadata, timestamp overflow, or allocation failure.
	pub fn open(path: impl AsRef<Path>) -> Result<Self> {
		let path = path.as_ref();
		if path.as_os_str().is_empty() {
			return Err(Error::invalid_argument("VideoDemuxer path is empty"));
		}
		let mut file = File::open(path).map_err(|source| Error::io("video file open", source))?;
		let file_size = file
			.metadata()
			.map_err(|source| Error::io("video file metadata", source))?
			.len();
		let moov = read_moov(&mut file, file_size)?;
		let parsed = parse_moov(&moov, file_size)?;
		Ok(Self {
			file: Some(file),
			info: parsed.info,
			samples: parsed.samples,
			current_sample: 0,
			nal_length_size: parsed.nal_length_size,
			codec_config: parsed.codec_config,
			need_codec_config: true,
			eos: false,
		})
	}

	/// Return selected-track metadata.
	pub const fn info(&self) -> VideoContainerInfo {
		self.info
	}

	/// Return whether the source has reached end of stream.
	pub const fn is_eos(&self) -> bool {
		self.eos
	}

	/// Return the zero-based index of the next sample.
	pub const fn current_sample_index(&self) -> usize {
		self.current_sample
	}

	/// Read the next compressed access unit, or `None` at end of stream.
	///
	/// # Errors
	///
	/// Returns an error after close or when seeking, allocation, reading, or NAL
	/// normalization fails.
	pub fn read_next_packet(&mut self) -> Result<Option<VideoPacket>> {
		let file = self
			.file
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("VideoDemuxer is closed"))?;
		let Some(sample) = self.samples.get(self.current_sample).copied() else {
			self.eos = true;
			return Ok(None);
		};
		let size = usize::try_from(sample.size)
			.map_err(|_| Error::out_of_range("video packet size exceeds usize"))?;
		if size == 0 || size > MAX_PACKET_BYTES {
			return Err(Error::data_loss(
				"video packet size is zero or exceeds the safety bound",
			));
		}
		let mut encoded = Vec::new();
		encoded
			.try_reserve_exact(size)
			.map_err(|_| Error::resource_exhausted("video packet allocation failed"))?;
		encoded.resize(size, 0);
		file.seek(SeekFrom::Start(sample.offset))
			.map_err(|source| Error::io("video packet seek", source))?;
		file.read_exact(&mut encoded)
			.map_err(|source| Error::io("video packet read", source))?;

		let mut data = if let Some(length_size) = self.nal_length_size {
			length_prefixed_to_annex_b(&encoded, length_size)?
		} else {
			encoded
		};
		if sample.keyframe && self.need_codec_config && !self.codec_config.is_empty() {
			let capacity = self
				.codec_config
				.len()
				.checked_add(data.len())
				.ok_or_else(|| {
					Error::out_of_range("video packet plus codec config overflows usize")
				})?;
			let mut prefixed = Vec::new();
			prefixed
				.try_reserve_exact(capacity)
				.map_err(|_| Error::resource_exhausted("video packet prefix allocation failed"))?;
			prefixed.extend_from_slice(&self.codec_config);
			prefixed.append(&mut data);
			data = prefixed;
			self.need_codec_config = false;
		}
		self.current_sample += 1;
		Ok(Some(VideoPacket {
			data,
			presentation_timestamp: sample.pts,
			decode_timestamp: sample.dts,
			duration: sample.duration,
			keyframe: sample.keyframe,
			track_id: self.info.track_id,
		}))
	}

	/// Seek to the closest preceding keyframe by presentation timestamp.
	///
	/// # Errors
	///
	/// Returns an error after close or when the stream contains no keyframe.
	pub fn seek(&mut self, presentation_timestamp: u64) -> Result<()> {
		if self.file.is_none() {
			return Err(Error::failed_precondition("VideoDemuxer is closed"));
		}
		let index = self
			.samples
			.iter()
			.enumerate()
			.filter(|(_, sample)| sample.keyframe && sample.pts <= presentation_timestamp)
			.max_by_key(|(_, sample)| sample.pts)
			.or_else(|| {
				self.samples
					.iter()
					.enumerate()
					.find(|(_, sample)| sample.keyframe)
			})
			.map(|(index, _)| index)
			.ok_or_else(|| Error::data_loss("video track has no random-access sample"))?;
		self.current_sample = index;
		self.need_codec_config = true;
		self.eos = false;
		Ok(())
	}

	/// Close this demuxer and discard unread packet state.
	///
	/// A second close is idempotent. Drop performs no I/O beyond releasing the
	/// already-open read-only file handle.
	pub fn close(&mut self) {
		self.file = None;
		self.eos = true;
	}
}

/// Convert an MP4 length-prefixed H.264/H.265 sample into Annex-B NAL units.
///
/// # Errors
///
/// Returns an error for a length width other than 1, 2, or 4, a truncated
/// length field, a zero-sized NAL unit, arithmetic overflow, or payload overrun.
pub fn length_prefixed_to_annex_b(sample: &[u8], length_size: usize) -> Result<Vec<u8>> {
	if !matches!(length_size, 1 | 2 | 4) {
		return Err(Error::invalid_argument(
			"NAL length-field size must be 1, 2, or 4 bytes",
		));
	}
	let mut cursor = 0_usize;
	let mut output = Vec::new();
	while cursor < sample.len() {
		let length_end = cursor
			.checked_add(length_size)
			.ok_or_else(|| Error::data_loss("NAL length-field offset overflows usize"))?;
		let length_bytes = sample
			.get(cursor..length_end)
			.ok_or_else(|| Error::data_loss("truncated NAL length field"))?;
		let mut length = 0_usize;
		for byte in length_bytes {
			length = length
				.checked_mul(256)
				.and_then(|value| value.checked_add(usize::from(*byte)))
				.ok_or_else(|| Error::data_loss("NAL payload length overflows usize"))?;
		}
		if length == 0 {
			return Err(Error::data_loss("zero-sized NAL unit"));
		}
		let payload_end = length_end
			.checked_add(length)
			.ok_or_else(|| Error::data_loss("NAL payload end overflows usize"))?;
		let payload = sample
			.get(length_end..payload_end)
			.ok_or_else(|| Error::data_loss("NAL payload exceeds the MP4 sample"))?;
		output
			.try_reserve(4 + payload.len())
			.map_err(|_| Error::resource_exhausted("Annex-B packet allocation failed"))?;
		output.extend_from_slice(&[0, 0, 0, 1]);
		output.extend_from_slice(payload);
		cursor = payload_end;
	}
	if output.is_empty() {
		return Err(Error::data_loss("MP4 sample contains no NAL units"));
	}
	Ok(output)
}

#[derive(Clone, Copy)]
struct Sample {
	offset: u64,
	size: u32,
	dts: u64,
	pts: u64,
	duration: u32,
	keyframe: bool,
}

struct ParsedMovie {
	info: VideoContainerInfo,
	samples: Vec<Sample>,
	nal_length_size: Option<usize>,
	codec_config: Vec<u8>,
}

struct ParsedTrack {
	track_id: u32,
	codec: VideoCodec,
	width: u32,
	height: u32,
	timescale: u32,
	duration: u64,
	samples: Vec<Sample>,
	nal_length_size: Option<usize>,
	codec_config: Vec<u8>,
	decode_profile: Option<VideoDecodeProfile>,
}

fn read_moov(file: &mut File, file_size: u64) -> Result<Vec<u8>> {
	let mut offset = 0_u64;
	let mut saw_ftyp = false;
	while offset < file_size {
		file.seek(SeekFrom::Start(offset))
			.map_err(|source| Error::io("MP4 box seek", source))?;
		let header = read_file_box_header(file, offset, file_size)?;
		if header.kind == *b"ftyp" {
			saw_ftyp = true;
		}
		if header.kind == *b"moov" {
			if !saw_ftyp {
				return Err(Error::data_loss(
					"MP4 movie box precedes or lacks file-type box",
				));
			}
			let payload_len = usize::try_from(header.payload_size)
				.map_err(|_| Error::out_of_range("MP4 movie box exceeds usize"))?;
			if payload_len > MAX_MOOV_BYTES {
				return Err(Error::resource_exhausted(
					"MP4 movie box exceeds the metadata bound",
				));
			}
			let mut payload = Vec::new();
			payload
				.try_reserve_exact(payload_len)
				.map_err(|_| Error::resource_exhausted("MP4 movie box allocation failed"))?;
			payload.resize(payload_len, 0);
			file.read_exact(&mut payload)
				.map_err(|source| Error::io("MP4 movie box read", source))?;
			return Ok(payload);
		}
		offset = header.end;
	}
	Err(Error::data_loss("MP4 file has no movie box"))
}

struct FileBoxHeader {
	kind: [u8; 4],
	payload_size: u64,
	end: u64,
}

fn read_file_box_header(file: &mut File, offset: u64, limit: u64) -> Result<FileBoxHeader> {
	let mut bytes = [0_u8; 8];
	file.read_exact(&mut bytes)
		.map_err(|source| Error::io("MP4 box header read", source))?;
	let size32 = u32::from_be_bytes(
		bytes[..4]
			.try_into()
			.map_err(|_| Error::internal("fixed MP4 box-size slice conversion failed"))?,
	);
	let kind = bytes[4..8]
		.try_into()
		.map_err(|_| Error::internal("fixed MP4 box-type slice conversion failed"))?;
	let (size, header_size) = if size32 == 1 {
		let mut extended = [0_u8; 8];
		file.read_exact(&mut extended)
			.map_err(|source| Error::io("MP4 extended box header read", source))?;
		(u64::from_be_bytes(extended), 16_u64)
	} else if size32 == 0 {
		(
			limit
				.checked_sub(offset)
				.ok_or_else(|| Error::data_loss("MP4 box offset exceeds file"))?,
			8,
		)
	} else {
		(u64::from(size32), 8)
	};
	if size < header_size {
		return Err(Error::data_loss("MP4 box is smaller than its header"));
	}
	let end = offset
		.checked_add(size)
		.ok_or_else(|| Error::data_loss("MP4 box end overflows u64"))?;
	if end > limit {
		return Err(Error::data_loss("MP4 box exceeds its containing file"));
	}
	Ok(FileBoxHeader {
		kind,
		payload_size: size - header_size,
		end,
	})
}

fn parse_moov(data: &[u8], file_size: u64) -> Result<ParsedMovie> {
	let mut tracks = Vec::new();
	let mut track_count = 0_u32;
	for child in boxes(data)? {
		if child.kind == *b"trak" {
			track_count = track_count
				.checked_add(1)
				.ok_or_else(|| Error::out_of_range("MP4 track count exceeds u32"))?;
			if let Some(track) = parse_trak(child.payload, file_size)? {
				tracks.push(track);
			}
		}
	}
	let track = tracks
		.into_iter()
		.next()
		.ok_or_else(|| Error::data_loss("MP4 contains no supported video track"))?;
	let sample_count = u32::try_from(track.samples.len())
		.map_err(|_| Error::out_of_range("MP4 sample count exceeds u32"))?;
	let time_base = VideoTimeBase::new(1, track.timescale)?;
	Ok(ParsedMovie {
		info: VideoContainerInfo {
			kind: VideoContainerKind::Mp4,
			codec: track.codec,
			width: track.width,
			height: track.height,
			time_base,
			duration: track.duration,
			sample_count,
			track_id: track.track_id,
			track_count,
			decode_profile: track.decode_profile,
		},
		samples: track.samples,
		nal_length_size: track.nal_length_size,
		codec_config: track.codec_config,
	})
}

fn parse_trak(data: &[u8], file_size: u64) -> Result<Option<ParsedTrack>> {
	let mut track_id = None;
	let mut mdia = None;
	for child in boxes(data)? {
		match &child.kind {
			b"tkhd" => track_id = parse_tkhd_track_id(child.payload),
			b"mdia" => mdia = Some(child.payload),
			_ => {}
		}
	}
	let Some(mdia) = mdia else {
		return Ok(None);
	};
	let mut handler_is_video = false;
	let mut timescale_duration = None;
	let mut minf = None;
	for child in boxes(mdia)? {
		match &child.kind {
			b"hdlr" => handler_is_video = child.payload.get(8..12) == Some(b"vide"),
			b"mdhd" => timescale_duration = parse_mdhd(child.payload),
			b"minf" => minf = Some(child.payload),
			_ => {}
		}
	}
	if !handler_is_video {
		return Ok(None);
	}
	let (timescale, duration) = timescale_duration
		.ok_or_else(|| Error::data_loss("MP4 video track lacks a valid media header"))?;
	let stbl = find_descendant(
		minf.ok_or_else(|| Error::data_loss("MP4 video track lacks media info"))?,
		b"stbl",
	)?
	.ok_or_else(|| Error::data_loss("MP4 video track lacks a sample table"))?;
	let tables = parse_sample_table(stbl, file_size)?;
	Ok(Some(ParsedTrack {
		track_id: track_id
			.ok_or_else(|| Error::data_loss("MP4 video track lacks a valid track id"))?,
		codec: tables.codec,
		width: tables.width,
		height: tables.height,
		timescale,
		duration,
		samples: tables.samples,
		nal_length_size: tables.nal_length_size,
		codec_config: tables.codec_config,
		decode_profile: tables.decode_profile,
	}))
}

fn parse_tkhd_track_id(data: &[u8]) -> Option<u32> {
	match *data.first()? {
		0 => read_u32(data, 12).ok().filter(|value| *value != 0),
		1 => read_u32(data, 20).ok().filter(|value| *value != 0),
		_ => None,
	}
}

fn parse_mdhd(data: &[u8]) -> Option<(u32, u64)> {
	let (timescale, duration) = match *data.first()? {
		0 => (
			read_u32(data, 12).ok()?,
			u64::from(read_u32(data, 16).ok()?),
		),
		1 => (read_u32(data, 20).ok()?, read_u64(data, 24).ok()?),
		_ => return None,
	};
	(timescale != 0).then_some((timescale, duration))
}

struct SampleTables {
	codec: VideoCodec,
	width: u32,
	height: u32,
	nal_length_size: Option<usize>,
	codec_config: Vec<u8>,
	decode_profile: Option<VideoDecodeProfile>,
	samples: Vec<Sample>,
}

fn parse_sample_table(data: &[u8], file_size: u64) -> Result<SampleTables> {
	let mut description = None;
	let mut durations = None;
	let mut composition_offsets = None;
	let mut chunk_map = None;
	let mut sizes = None;
	let mut chunks = None;
	let mut sync_samples = None;
	for child in boxes(data)? {
		match &child.kind {
			b"stsd" => description = Some(parse_stsd(child.payload)?),
			b"stts" => durations = Some(parse_stts(child.payload)?),
			b"ctts" => composition_offsets = Some(parse_ctts(child.payload)?),
			b"stsc" => chunk_map = Some(parse_stsc(child.payload)?),
			b"stsz" => sizes = Some(parse_stsz(child.payload)?),
			b"stco" => chunks = Some(parse_stco(child.payload)?),
			b"co64" => chunks = Some(parse_co64(child.payload)?),
			b"stss" => sync_samples = Some(parse_stss(child.payload)?),
			_ => {}
		}
	}
	let description = description.ok_or_else(|| Error::data_loss("MP4 sample table lacks stsd"))?;
	let durations = durations.ok_or_else(|| Error::data_loss("MP4 sample table lacks stts"))?;
	let chunk_map = chunk_map.ok_or_else(|| Error::data_loss("MP4 sample table lacks stsc"))?;
	let sizes = sizes.ok_or_else(|| Error::data_loss("MP4 sample table lacks stsz"))?;
	let chunks = chunks.ok_or_else(|| Error::data_loss("MP4 sample table lacks stco/co64"))?;
	let offsets = build_sample_offsets(&sizes, &chunks, &chunk_map, file_size)?;
	if durations.len() != sizes.len() {
		return Err(Error::data_loss("MP4 stts count does not match stsz"));
	}
	let composition_offsets = composition_offsets.unwrap_or_else(|| vec![0; sizes.len()]);
	if composition_offsets.len() != sizes.len() {
		return Err(Error::data_loss("MP4 ctts count does not match stsz"));
	}
	let all_sync = sync_samples.is_none();
	let sync_samples = sync_samples.unwrap_or_default();
	let mut dts = 0_u64;
	let mut samples = Vec::new();
	samples
		.try_reserve_exact(sizes.len())
		.map_err(|_| Error::resource_exhausted("MP4 sample index allocation failed"))?;
	for index in 0..sizes.len() {
		let pts = checked_signed_add(dts, composition_offsets[index])?;
		let sample_number = u32::try_from(index + 1)
			.map_err(|_| Error::out_of_range("MP4 sample number exceeds u32"))?;
		samples.push(Sample {
			offset: offsets[index],
			size: sizes[index],
			dts,
			pts,
			duration: durations[index],
			keyframe: all_sync || sync_samples.contains(&sample_number),
		});
		dts = dts
			.checked_add(u64::from(durations[index]))
			.ok_or_else(|| Error::data_loss("MP4 decode timestamp overflows u64"))?;
	}
	Ok(SampleTables {
		codec: description.codec,
		width: description.width,
		height: description.height,
		nal_length_size: description.nal_length_size,
		codec_config: description.codec_config,
		decode_profile: description.decode_profile,
		samples,
	})
}

struct SampleDescription {
	codec: VideoCodec,
	width: u32,
	height: u32,
	nal_length_size: Option<usize>,
	codec_config: Vec<u8>,
	decode_profile: Option<VideoDecodeProfile>,
}

fn parse_stsd(data: &[u8]) -> Result<SampleDescription> {
	let count = usize::try_from(read_u32(data, 4)?)
		.map_err(|_| Error::out_of_range("MP4 stsd entry count exceeds usize"))?;
	if count == 0 || count > 64 {
		return Err(Error::data_loss(
			"MP4 stsd has no entries or exceeds the safety bound",
		));
	}
	let entries = data
		.get(8..)
		.ok_or_else(|| Error::data_loss("truncated MP4 stsd header"))?;
	for entry in boxes(entries)?.into_iter().take(count) {
		let codec = match &entry.kind {
			b"avc1" | b"avc3" => VideoCodec::H264,
			b"hvc1" | b"hev1" => VideoCodec::H265,
			b"av01" => VideoCodec::Av1,
			b"vp09" => VideoCodec::Vp9,
			_ => continue,
		};
		let width = u32::from(read_u16(entry.payload, 24)?);
		let height = u32::from(read_u16(entry.payload, 26)?);
		if width == 0 || height == 0 {
			return Err(Error::data_loss(
				"MP4 video sample entry has zero dimensions",
			));
		}
		let children = entry
			.payload
			.get(78..)
			.ok_or_else(|| Error::data_loss("truncated MP4 visual sample entry"))?;
		let mut nal_length_size = None;
		let mut codec_config = Vec::new();
		let mut decode_profile = None;
		for child in boxes(children)? {
			match (&codec, &child.kind) {
				(VideoCodec::H264, b"avcC") => {
					let parsed = parse_avcc(child.payload)?;
					nal_length_size = Some(parsed.0);
					codec_config = parsed.1;
					decode_profile = parsed.2;
				}
				(VideoCodec::H265, b"hvcC") => {
					let parsed = parse_hvcc(child.payload)?;
					nal_length_size = Some(parsed.0);
					codec_config = parsed.1;
					decode_profile = parsed.2;
				}
				(VideoCodec::Av1, b"av1C") => {
					decode_profile = parse_av1c_profile(child.payload)?;
					codec_config = child.payload.get(4..).unwrap_or_default().to_vec();
				}
				_ => {}
			}
		}
		if matches!(codec, VideoCodec::H264 | VideoCodec::H265) && nal_length_size.is_none() {
			return Err(Error::data_loss(
				"MP4 AVC/HEVC sample entry lacks codec configuration",
			));
		}
		return Ok(SampleDescription {
			codec,
			width,
			height,
			nal_length_size,
			codec_config,
			decode_profile,
		});
	}
	Err(Error::data_loss(
		"MP4 stsd contains no supported video sample entry",
	))
}

fn parse_avcc(data: &[u8]) -> Result<(usize, Vec<u8>, Option<VideoDecodeProfile>)> {
	let length_size = usize::from(
		*data
			.get(4)
			.ok_or_else(|| Error::data_loss("truncated avcC"))?
			& 3,
	) + 1;
	if !matches!(length_size, 1 | 2 | 4) {
		return Err(Error::data_loss(
			"avcC declares an unsupported NAL length width",
		));
	}
	let sps_count = usize::from(
		*data
			.get(5)
			.ok_or_else(|| Error::data_loss("truncated avcC"))?
			& 0x1f,
	);
	let mut cursor = 6_usize;
	let mut config = Vec::new();
	for _ in 0..sps_count {
		append_config_nal(data, &mut cursor, &mut config)?;
	}
	let pps_count = usize::from(
		*data
			.get(cursor)
			.ok_or_else(|| Error::data_loss("truncated avcC PPS count"))?,
	);
	cursor += 1;
	for _ in 0..pps_count {
		append_config_nal(data, &mut cursor, &mut config)?;
	}
	let decode_profile = h264_profile_from_config(&config)?;
	Ok((length_size, config, decode_profile))
}

fn parse_hvcc(data: &[u8]) -> Result<(usize, Vec<u8>, Option<VideoDecodeProfile>)> {
	let length_size = usize::from(
		*data
			.get(21)
			.ok_or_else(|| Error::data_loss("truncated hvcC"))?
			& 3,
	) + 1;
	if !matches!(length_size, 1 | 2 | 4) {
		return Err(Error::data_loss(
			"hvcC declares an unsupported NAL length width",
		));
	}
	let arrays = usize::from(
		*data
			.get(22)
			.ok_or_else(|| Error::data_loss("truncated hvcC"))?,
	);
	let mut cursor = 23_usize;
	let mut config = Vec::new();
	for _ in 0..arrays {
		cursor = cursor
			.checked_add(1)
			.ok_or_else(|| Error::data_loss("hvcC array offset overflows usize"))?;
		let nal_count = usize::from(read_u16(data, cursor)?);
		cursor += 2;
		for _ in 0..nal_count {
			append_config_nal(data, &mut cursor, &mut config)?;
		}
	}
	let decode_profile = h265_profile_from_hvcc(data)?;
	Ok((length_size, config, decode_profile))
}

fn h265_profile_from_hvcc(data: &[u8]) -> Result<Option<VideoDecodeProfile>> {
	let profile = match data
		.get(1)
		.ok_or_else(|| Error::data_loss("truncated hvcC profile"))?
		& 0x1f
	{
		1 => H265Profile::Main,
		2 => H265Profile::Main10,
		3 => H265Profile::MainStillPicture,
		4 => H265Profile::FormatRangeExtensions,
		9 => H265Profile::ScreenContentCodingExtensions,
		_ => return Ok(None),
	};
	let chroma = parse_chroma(u32::from(
		*data
			.get(16)
			.ok_or_else(|| Error::data_loss("truncated hvcC chroma format"))?
			& 3,
	))?;
	let luma = parse_bit_depth(
		u32::from(
			*data
				.get(17)
				.ok_or_else(|| Error::data_loss("truncated hvcC luma depth"))?
				& 7,
		) + 8,
	)?;
	let chroma_depth = parse_bit_depth(
		u32::from(
			*data
				.get(18)
				.ok_or_else(|| Error::data_loss("truncated hvcC chroma depth"))?
				& 7,
		) + 8,
	)?;
	Ok(Some(VideoDecodeProfile::H265 {
		profile,
		chroma_subsampling: chroma,
		luma_bit_depth: luma,
		chroma_bit_depth: chroma_depth,
	}))
}

fn parse_av1c_profile(data: &[u8]) -> Result<Option<VideoDecodeProfile>> {
	let profile_byte = *data
		.get(1)
		.ok_or_else(|| Error::data_loss("truncated av1C profile"))?;
	let format_byte = *data
		.get(2)
		.ok_or_else(|| Error::data_loss("truncated av1C format"))?;
	let profile = match profile_byte >> 5 {
		0 => Av1Profile::Main,
		1 => Av1Profile::High,
		2 => Av1Profile::Professional,
		_ => return Ok(None),
	};
	let monochrome = format_byte & 0x10 != 0;
	let chroma = if monochrome {
		VideoChromaSubsampling::Monochrome
	} else {
		match ((format_byte >> 3) & 1, (format_byte >> 2) & 1) {
			(1, 1) => VideoChromaSubsampling::Yuv420,
			(1, 0) => VideoChromaSubsampling::Yuv422,
			(0, 0) => VideoChromaSubsampling::Yuv444,
			_ => return Ok(None),
		}
	};
	let depth = if format_byte & 0x40 == 0 {
		VideoComponentBitDepth::Eight
	} else if format_byte & 0x20 == 0 {
		VideoComponentBitDepth::Ten
	} else {
		VideoComponentBitDepth::Twelve
	};
	Ok(Some(VideoDecodeProfile::Av1 {
		profile,
		film_grain_support: false,
		chroma_subsampling: chroma,
		luma_bit_depth: depth,
		chroma_bit_depth: depth,
	}))
}

fn h264_profile_from_config(config: &[u8]) -> Result<Option<VideoDecodeProfile>> {
	let Some(sps) = super::parse_nal_annex_b(config).into_iter().find(|nal| {
		nal.payload()
			.first()
			.is_some_and(|header| header & 0x1f == 7)
	}) else {
		return Ok(None);
	};
	h264_profile_from_sps(sps.payload())
}

fn h264_profile_from_sps(nal: &[u8]) -> Result<Option<VideoDecodeProfile>> {
	let sps = super::parse_h264_sps(nal)?;
	let profile = match sps.profile_idc {
		66 => H264Profile::Baseline,
		77 => H264Profile::Main,
		100 => H264Profile::High,
		244 => H264Profile::High444Predictive,
		_ => return Ok(None),
	};
	Ok(Some(VideoDecodeProfile::H264 {
		profile,
		picture_layout: if sps.frame_mbs_only {
			H264PictureLayout::Progressive
		} else {
			H264PictureLayout::InterlacedInterleavedLines
		},
		chroma_subsampling: parse_chroma(sps.chroma_format_idc)?,
		luma_bit_depth: parse_bit_depth(sps.bit_depth_luma_minus_8 + 8)?,
		chroma_bit_depth: parse_bit_depth(sps.bit_depth_chroma_minus_8 + 8)?,
	}))
}

fn parse_chroma(value: u32) -> Result<VideoChromaSubsampling> {
	match value {
		0 => Ok(VideoChromaSubsampling::Monochrome),
		1 => Ok(VideoChromaSubsampling::Yuv420),
		2 => Ok(VideoChromaSubsampling::Yuv422),
		3 => Ok(VideoChromaSubsampling::Yuv444),
		_ => Err(Error::data_loss(
			"codec configuration has invalid chroma format",
		)),
	}
}

fn parse_bit_depth(value: u32) -> Result<VideoComponentBitDepth> {
	match value {
		8 => Ok(VideoComponentBitDepth::Eight),
		10 => Ok(VideoComponentBitDepth::Ten),
		12 => Ok(VideoComponentBitDepth::Twelve),
		_ => Err(Error::data_loss(
			"codec configuration has unsupported bit depth",
		)),
	}
}

fn append_config_nal(data: &[u8], cursor: &mut usize, output: &mut Vec<u8>) -> Result<()> {
	let length = usize::from(read_u16(data, *cursor)?);
	*cursor = cursor
		.checked_add(2)
		.ok_or_else(|| Error::data_loss("codec-config NAL offset overflows usize"))?;
	if length == 0 {
		return Err(Error::data_loss(
			"codec configuration contains an empty NAL unit",
		));
	}
	let end = cursor
		.checked_add(length)
		.ok_or_else(|| Error::data_loss("codec-config NAL end overflows usize"))?;
	let payload = data
		.get(*cursor..end)
		.ok_or_else(|| Error::data_loss("codec-config NAL exceeds its box"))?;
	output
		.try_reserve(4 + length)
		.map_err(|_| Error::resource_exhausted("codec-config allocation failed"))?;
	output.extend_from_slice(&[0, 0, 0, 1]);
	output.extend_from_slice(payload);
	*cursor = end;
	Ok(())
}

fn parse_stts(data: &[u8]) -> Result<Vec<u32>> {
	let entries = table_count(data, 8, "stts")?;
	let mut values = Vec::new();
	for index in 0..entries {
		let offset = 8 + index * 8;
		let count = usize::try_from(read_u32(data, offset)?)
			.map_err(|_| Error::out_of_range("stts sample count exceeds usize"))?;
		let delta = read_u32(data, offset + 4)?;
		if delta == 0
			|| values
				.len()
				.checked_add(count)
				.is_none_or(|len| len > MAX_MP4_TABLE_ENTRIES)
		{
			return Err(Error::data_loss(
				"MP4 stts expansion is invalid or exceeds the safety bound",
			));
		}
		values
			.try_reserve(count)
			.map_err(|_| Error::resource_exhausted("MP4 stts allocation failed"))?;
		values.extend(std::iter::repeat_n(delta, count));
	}
	Ok(values)
}

fn parse_ctts(data: &[u8]) -> Result<Vec<i64>> {
	let version = *data
		.first()
		.ok_or_else(|| Error::data_loss("truncated ctts"))?;
	if version > 1 {
		return Err(Error::data_loss("unsupported MP4 ctts version"));
	}
	let entries = table_count(data, 8, "ctts")?;
	let mut values = Vec::new();
	for index in 0..entries {
		let offset = 8 + index * 8;
		let count = usize::try_from(read_u32(data, offset)?)
			.map_err(|_| Error::out_of_range("ctts sample count exceeds usize"))?;
		let raw = read_u32(data, offset + 4)?;
		let value = if version == 1 {
			i64::from(i32::from_be_bytes(raw.to_be_bytes()))
		} else {
			i64::from(raw)
		};
		if values
			.len()
			.checked_add(count)
			.is_none_or(|len| len > MAX_MP4_TABLE_ENTRIES)
		{
			return Err(Error::data_loss(
				"MP4 ctts expansion exceeds the safety bound",
			));
		}
		values
			.try_reserve(count)
			.map_err(|_| Error::resource_exhausted("MP4 ctts allocation failed"))?;
		values.extend(std::iter::repeat_n(value, count));
	}
	Ok(values)
}

#[derive(Clone, Copy)]
struct SampleToChunk {
	first_chunk: u32,
	samples_per_chunk: u32,
}

fn parse_stsc(data: &[u8]) -> Result<Vec<SampleToChunk>> {
	let entries = table_count(data, 12, "stsc")?;
	let mut values = Vec::new();
	values
		.try_reserve_exact(entries)
		.map_err(|_| Error::resource_exhausted("MP4 stsc allocation failed"))?;
	for index in 0..entries {
		let offset = 8 + index * 12;
		let first_chunk = read_u32(data, offset)?;
		let samples_per_chunk = read_u32(data, offset + 4)?;
		if first_chunk == 0 || samples_per_chunk == 0 || (index == 0 && first_chunk != 1) {
			return Err(Error::data_loss("MP4 stsc contains invalid chunk mapping"));
		}
		if values
			.last()
			.is_some_and(|previous: &SampleToChunk| previous.first_chunk >= first_chunk)
		{
			return Err(Error::data_loss(
				"MP4 stsc first_chunk values are not increasing",
			));
		}
		values.push(SampleToChunk {
			first_chunk,
			samples_per_chunk,
		});
	}
	Ok(values)
}

fn parse_stsz(data: &[u8]) -> Result<Vec<u32>> {
	let constant = read_u32(data, 4)?;
	let count = bounded_count(read_u32(data, 8)?, "stsz")?;
	if constant != 0 {
		return Ok(vec![constant; count]);
	}
	ensure_table(data, 12, count, 4, "stsz")?;
	(0..count)
		.map(|index| read_u32(data, 12 + index * 4))
		.collect()
}

fn parse_stco(data: &[u8]) -> Result<Vec<u64>> {
	let count = table_count(data, 4, "stco")?;
	(0..count)
		.map(|index| read_u32(data, 8 + index * 4).map(u64::from))
		.collect()
}

fn parse_co64(data: &[u8]) -> Result<Vec<u64>> {
	let count = table_count(data, 8, "co64")?;
	(0..count)
		.map(|index| read_u64(data, 8 + index * 8))
		.collect()
}

fn parse_stss(data: &[u8]) -> Result<BTreeSet<u32>> {
	let count = table_count(data, 4, "stss")?;
	let mut values = BTreeSet::new();
	for index in 0..count {
		let sample = read_u32(data, 8 + index * 4)?;
		if sample == 0 {
			return Err(Error::data_loss("MP4 stss contains sample zero"));
		}
		values.insert(sample);
	}
	Ok(values)
}

fn build_sample_offsets(
	sizes: &[u32],
	chunks: &[u64],
	map: &[SampleToChunk],
	file_size: u64,
) -> Result<Vec<u64>> {
	if sizes.is_empty() || chunks.is_empty() || map.is_empty() {
		return Err(Error::data_loss("MP4 sample/chunk tables are empty"));
	}
	let mut offsets = Vec::new();
	offsets
		.try_reserve_exact(sizes.len())
		.map_err(|_| Error::resource_exhausted("MP4 sample-offset allocation failed"))?;
	let mut sample_index = 0_usize;
	let mut map_index = 0_usize;
	for (chunk_index, chunk_offset) in chunks.iter().copied().enumerate() {
		let chunk_number = u32::try_from(chunk_index + 1)
			.map_err(|_| Error::out_of_range("MP4 chunk number exceeds u32"))?;
		while map_index + 1 < map.len() && map[map_index + 1].first_chunk <= chunk_number {
			map_index += 1;
		}
		let mut offset = chunk_offset;
		for _ in 0..map[map_index].samples_per_chunk {
			let size = *sizes
				.get(sample_index)
				.ok_or_else(|| Error::data_loss("MP4 chunks reference too many samples"))?;
			let end = offset
				.checked_add(u64::from(size))
				.ok_or_else(|| Error::data_loss("MP4 sample end overflows u64"))?;
			if size == 0 || end > file_size {
				return Err(Error::data_loss("MP4 sample is empty or exceeds the file"));
			}
			offsets.push(offset);
			offset = end;
			sample_index += 1;
		}
	}
	if sample_index != sizes.len() {
		return Err(Error::data_loss("MP4 chunks do not cover every sample"));
	}
	Ok(offsets)
}

struct SliceBox<'a> {
	kind: [u8; 4],
	payload: &'a [u8],
}

fn boxes(mut data: &[u8]) -> Result<Vec<SliceBox<'_>>> {
	let mut output = Vec::new();
	while !data.is_empty() {
		if data.len() < 8 {
			return Err(Error::data_loss("truncated MP4 child-box header"));
		}
		let size32 = read_u32(data, 0)?;
		let kind = data[4..8]
			.try_into()
			.map_err(|_| Error::internal("fixed MP4 child-box type conversion failed"))?;
		let (size, header) = if size32 == 1 {
			(
				usize::try_from(read_u64(data, 8)?)
					.map_err(|_| Error::out_of_range("MP4 child box exceeds usize"))?,
				16,
			)
		} else if size32 == 0 {
			(data.len(), 8)
		} else {
			(
				usize::try_from(size32)
					.map_err(|_| Error::out_of_range("MP4 child box exceeds usize"))?,
				8,
			)
		};
		if size < header || size > data.len() {
			return Err(Error::data_loss("MP4 child box exceeds its parent"));
		}
		output.push(SliceBox {
			kind,
			payload: &data[header..size],
		});
		data = &data[size..];
	}
	Ok(output)
}

fn find_descendant<'a>(data: &'a [u8], kind: &[u8; 4]) -> Result<Option<&'a [u8]>> {
	Ok(boxes(data)?
		.into_iter()
		.find(|child| &child.kind == kind)
		.map(|child| child.payload))
}

fn table_count(data: &[u8], stride: usize, name: &'static str) -> Result<usize> {
	let count = bounded_count(read_u32(data, 4)?, name)?;
	ensure_table(data, 8, count, stride, name)?;
	Ok(count)
}

fn bounded_count(count: u32, name: &'static str) -> Result<usize> {
	let count = usize::try_from(count)
		.map_err(|_| Error::out_of_range(format!("MP4 {name} count exceeds usize")))?;
	if count > MAX_MP4_TABLE_ENTRIES {
		return Err(Error::resource_exhausted(format!(
			"MP4 {name} count exceeds the safety bound"
		)));
	}
	Ok(count)
}

fn ensure_table(
	data: &[u8],
	header: usize,
	count: usize,
	stride: usize,
	name: &'static str,
) -> Result<()> {
	let bytes = count
		.checked_mul(stride)
		.and_then(|value| header.checked_add(value))
		.ok_or_else(|| Error::data_loss(format!("MP4 {name} table size overflows usize")))?;
	if bytes > data.len() {
		return Err(Error::data_loss(format!("truncated MP4 {name} table")));
	}
	Ok(())
}

fn read_u16(data: &[u8], offset: usize) -> Result<u16> {
	let bytes = data
		.get(offset..offset.saturating_add(2))
		.ok_or_else(|| Error::data_loss("truncated MP4 u16 field"))?;
	Ok(u16::from_be_bytes(bytes.try_into().map_err(|_| {
		Error::internal("fixed MP4 u16 conversion failed")
	})?))
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32> {
	let bytes = data
		.get(offset..offset.saturating_add(4))
		.ok_or_else(|| Error::data_loss("truncated MP4 u32 field"))?;
	Ok(u32::from_be_bytes(bytes.try_into().map_err(|_| {
		Error::internal("fixed MP4 u32 conversion failed")
	})?))
}

fn read_u64(data: &[u8], offset: usize) -> Result<u64> {
	let bytes = data
		.get(offset..offset.saturating_add(8))
		.ok_or_else(|| Error::data_loss("truncated MP4 u64 field"))?;
	Ok(u64::from_be_bytes(bytes.try_into().map_err(|_| {
		Error::internal("fixed MP4 u64 conversion failed")
	})?))
}

fn checked_signed_add(value: u64, offset: i64) -> Result<u64> {
	if offset >= 0 {
		value
			.checked_add(
				u64::try_from(offset)
					.map_err(|_| Error::data_loss("positive ctts conversion failed"))?,
			)
			.ok_or_else(|| Error::data_loss("video presentation timestamp overflows u64"))
	} else {
		value
			.checked_sub(offset.unsigned_abs())
			.ok_or_else(|| Error::data_loss("video presentation timestamp is negative"))
	}
}
