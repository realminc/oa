//! Bounded Matroska/WebM SimpleBlock indexer.
//!
//! Donor: `source/cpp/lib/oa/vision/video/videoDemuxer.cpp`, the EBML,
//! Matroska track, and cluster parsers. Only the donor's unlaced SimpleBlock
//! path is admitted; media payload remains in the source file until read.

use std::{
	fs::File,
	io::{Read, Seek, SeekFrom},
	path::Path,
};

use crate::{Error, Result};

use super::{
	MAX_MP4_TABLE_ENTRIES, MAX_PACKET_BYTES, ParsedMovie, Sample, VideoChromaSubsampling, VideoCodec,
	VideoComponentBitDepth, VideoContainerInfo, VideoContainerKind, VideoDecodeProfile,
	VideoTimeBase, Vp9Profile, parse_av1c_profile, parse_avcc, parse_hvcc,
};

const MAX_CODEC_PRIVATE: u64 = 16 * 1024 * 1024;
const MAX_CLUSTERS: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy)]
struct Element {
	id: u64,
	data_offset: u64,
	end: u64,
}

#[derive(Default)]
struct Track {
	number: u64,
	kind: u64,
	codec: String,
	codec_private: Vec<u8>,
	width: u32,
	height: u32,
	default_duration_ns: u64,
}

struct CodecConfig {
	codec: VideoCodec,
	nal_length_size: Option<usize>,
	parameter_sets: Vec<u8>,
	decode_profile: Option<VideoDecodeProfile>,
}

pub(super) fn parse(file: &mut File, file_size: u64, path: &Path) -> Result<ParsedMovie> {
	seek(file, 0)?;
	let header = read_element(file, file_size)?
		.ok_or_else(|| Error::data_loss("Matroska EBML header is absent"))?;
	if header.id != 0x1a45_dfa3 {
		return Err(Error::data_loss("Matroska EBML header is absent"));
	}
	seek(file, header.end)?;
	let segment =
		read_element(file, file_size)?.ok_or_else(|| Error::data_loss("Matroska segment is absent"))?;
	if segment.id != 0x1853_8067 {
		return Err(Error::data_loss("Matroska segment is absent"));
	}
	let mut timecode_scale = 1_000_000_u64;
	let mut track = None;
	let mut track_count = 0_u32;
	let mut clusters = Vec::new();
	seek(file, segment.data_offset)?;
	while let Some(element) = read_element(file, segment.end)? {
		match element.id {
			0x1549_a966 => parse_info(file, element, &mut timecode_scale)?,
			0x1654_ae6b => parse_tracks(file, element, &mut track, &mut track_count)?,
			0x1f43_b675 => {
				if clusters.len() >= MAX_CLUSTERS {
					return Err(Error::resource_exhausted(
						"Matroska cluster count exceeds bound",
					));
				}
				clusters
					.try_reserve(1)
					.map_err(|_| Error::resource_exhausted("Matroska cluster index allocation failed"))?;
				clusters.push(element);
			}
			_ => {}
		}
		seek(file, element.end)?;
	}
	let track = track.ok_or_else(|| Error::data_loss("Matroska has no video track"))?;
	if track.number == 0 || track.width == 0 || track.height == 0 || timecode_scale == 0 {
		return Err(Error::data_loss("Matroska video metadata is invalid"));
	}
	let time_base = VideoTimeBase::new(
		u32::try_from(timecode_scale)
			.map_err(|_| Error::out_of_range("Matroska timecode scale exceeds u32"))?,
		1_000_000_000,
	)?;
	let codec_config = codec_config(&track)?;
	let mut decode_profile = codec_config.decode_profile;
	let mut samples = Vec::new();
	for cluster in clusters {
		parse_cluster(file, cluster, &track, timecode_scale, &mut samples)?;
	}
	if samples.is_empty() {
		return Err(Error::data_loss(
			"Matroska video track has no unlaced SimpleBlocks",
		));
	}
	if codec_config.codec == VideoCodec::Vp9 {
		decode_profile = vp9_profile(file, &samples)?;
	}
	let duration = samples.iter().try_fold(0_u64, |end, sample| {
		let sample_end = sample
			.pts
			.checked_add(u64::from(sample.duration))
			.ok_or_else(|| Error::out_of_range("Matroska duration overflow"))?;
		Ok::<u64, Error>(end.max(sample_end))
	})?;
	let sample_count = u32::try_from(samples.len())
		.map_err(|_| Error::out_of_range("Matroska sample count exceeds u32"))?;
	let track_id = u32::try_from(track.number)
		.map_err(|_| Error::out_of_range("Matroska video track number exceeds u32"))?;
	let kind = if path
		.extension()
		.is_some_and(|extension| extension.eq_ignore_ascii_case("webm"))
	{
		VideoContainerKind::WebM
	} else {
		VideoContainerKind::Matroska
	};
	Ok(ParsedMovie {
		info: VideoContainerInfo {
			kind,
			codec: codec_config.codec,
			width: track.width,
			height: track.height,
			time_base,
			duration,
			sample_count,
			track_id,
			track_count,
			decode_profile,
		},
		samples,
		nal_length_size: codec_config.nal_length_size,
		codec_config: codec_config.parameter_sets,
	})
}

fn codec_config(track: &Track) -> Result<CodecConfig> {
	match track.codec.as_str() {
		"V_MPEG4/ISO/AVC" => {
			let (length, config, profile) = parse_avcc(&track.codec_private)?;
			if config.is_empty() {
				return Err(Error::data_loss(
					"Matroska AVC codecPrivate lacks parameter sets",
				));
			}
			Ok(CodecConfig {
				codec: VideoCodec::H264,
				nal_length_size: Some(length),
				parameter_sets: config,
				decode_profile: profile,
			})
		}
		"V_MPEGH/ISO/HEVC" => {
			let (length, config, profile) = parse_hvcc(&track.codec_private)?;
			if config.is_empty() {
				return Err(Error::data_loss(
					"Matroska HEVC codecPrivate lacks parameter sets",
				));
			}
			Ok(CodecConfig {
				codec: VideoCodec::H265,
				nal_length_size: Some(length),
				parameter_sets: config,
				decode_profile: profile,
			})
		}
		"V_AV1" => {
			if track.codec_private.len() < 5 {
				return Err(Error::data_loss("Matroska AV1 codecPrivate is truncated"));
			}
			let profile = parse_av1c_profile(&track.codec_private)?;
			Ok(CodecConfig {
				codec: VideoCodec::Av1,
				nal_length_size: None,
				parameter_sets: Vec::new(),
				decode_profile: profile,
			})
		}
		"V_VP9" => Ok(CodecConfig {
			codec: VideoCodec::Vp9,
			nal_length_size: None,
			parameter_sets: Vec::new(),
			decode_profile: None,
		}),
		_ => Err(Error::missing_capability(
			"Matroska video codec is unsupported",
		)),
	}
}

fn vp9_profile(file: &mut File, samples: &[Sample]) -> Result<Option<VideoDecodeProfile>> {
	let Some(sample) = samples.iter().find(|sample| sample.keyframe) else {
		return Ok(None);
	};
	let mut bytes = Vec::new();
	bytes
		.try_reserve_exact(sample.size as usize)
		.map_err(|_| Error::resource_exhausted("Matroska VP9 profile packet allocation failed"))?;
	bytes.resize(sample.size as usize, 0);
	seek(file, sample.offset)?;
	file
		.read_exact(&mut bytes)
		.map_err(|source| Error::io("Matroska VP9 profile packet read", source))?;
	let Ok(pictures) = crate::video::vp9::Vp9Parser::default().parse_access_unit(&bytes) else {
		return Ok(None);
	};
	let Some(picture) = pictures.first() else {
		return Ok(None);
	};
	let profile = match picture.profile {
		0 => Vp9Profile::Profile0,
		1 => Vp9Profile::Profile1,
		2 => Vp9Profile::Profile2,
		3 => Vp9Profile::Profile3,
		_ => return Ok(None),
	};
	let depth = match picture.color.bit_depth {
		8 => VideoComponentBitDepth::Eight,
		10 => VideoComponentBitDepth::Ten,
		12 => VideoComponentBitDepth::Twelve,
		_ => return Ok(None),
	};
	let chroma = match (picture.color.subsampling_x, picture.color.subsampling_y) {
		(true, true) => VideoChromaSubsampling::Yuv420,
		(true, false) => VideoChromaSubsampling::Yuv422,
		(false, false) => VideoChromaSubsampling::Yuv444,
		(false, true) => return Ok(None),
	};
	Ok(Some(VideoDecodeProfile::Vp9 {
		profile,
		chroma_subsampling: chroma,
		luma_bit_depth: depth,
		chroma_bit_depth: depth,
	}))
}

fn parse_info(file: &mut File, parent: Element, scale: &mut u64) -> Result<()> {
	seek(file, parent.data_offset)?;
	while let Some(element) = read_element(file, parent.end)? {
		if element.id == 0x002a_d7b1 {
			*scale = read_unsigned(file, element)?;
		}
		seek(file, element.end)?;
	}
	Ok(())
}

fn parse_tracks(
	file: &mut File,
	parent: Element,
	selected: &mut Option<Track>,
	count: &mut u32,
) -> Result<()> {
	seek(file, parent.data_offset)?;
	while let Some(element) = read_element(file, parent.end)? {
		if element.id == 0xae {
			*count = count
				.checked_add(1)
				.ok_or_else(|| Error::out_of_range("Matroska track count overflow"))?;
			let track = parse_track(file, element)?;
			if selected.is_none() && track.kind == 1 {
				*selected = Some(track);
			}
		}
		seek(file, element.end)?;
	}
	Ok(())
}

fn parse_track(file: &mut File, parent: Element) -> Result<Track> {
	let mut track = Track::default();
	seek(file, parent.data_offset)?;
	while let Some(element) = read_element(file, parent.end)? {
		match element.id {
			0xd7 => track.number = read_unsigned(file, element)?,
			0x83 => track.kind = read_unsigned(file, element)?,
			0x86 => {
				if element.end - element.data_offset > 1024 {
					return Err(Error::resource_exhausted("Matroska codec id exceeds bound"));
				}
				track.codec = read_string(file, element)?;
			}
			0x63a2 => {
				if element.end - element.data_offset > MAX_CODEC_PRIVATE {
					return Err(Error::resource_exhausted(
						"Matroska codecPrivate exceeds bound",
					));
				}
				track.codec_private = read_bytes(file, element)?;
			}
			0x23e383 => track.default_duration_ns = read_unsigned(file, element)?,
			0xe0 => parse_video(file, element, &mut track)?,
			_ => {}
		}
		seek(file, element.end)?;
	}
	Ok(track)
}

fn parse_video(file: &mut File, parent: Element, track: &mut Track) -> Result<()> {
	seek(file, parent.data_offset)?;
	while let Some(element) = read_element(file, parent.end)? {
		match element.id {
			0xb0 => {
				track.width = u32::try_from(read_unsigned(file, element)?)
					.map_err(|_| Error::out_of_range("Matroska video width exceeds u32"))?
			}
			0xba => {
				track.height = u32::try_from(read_unsigned(file, element)?)
					.map_err(|_| Error::out_of_range("Matroska video height exceeds u32"))?
			}
			_ => {}
		}
		seek(file, element.end)?;
	}
	Ok(())
}

fn parse_cluster(
	file: &mut File,
	parent: Element,
	track: &Track,
	scale: u64,
	samples: &mut Vec<Sample>,
) -> Result<()> {
	let mut cluster_timecode = 0_u64;
	seek(file, parent.data_offset)?;
	while let Some(element) = read_element(file, parent.end)? {
		match element.id {
			0xe7 => cluster_timecode = read_unsigned(file, element)?,
			0xa3 => parse_simple_block(file, element, track, scale, cluster_timecode, samples)?,
			_ => {}
		}
		seek(file, element.end)?;
	}
	Ok(())
}

fn parse_simple_block(
	file: &mut File,
	block: Element,
	track: &Track,
	scale: u64,
	cluster_timecode: u64,
	samples: &mut Vec<Sample>,
) -> Result<()> {
	let size = block.end - block.data_offset;
	if size < 4 {
		return Err(Error::data_loss("Matroska SimpleBlock is truncated"));
	}
	let (track_number, track_bytes, unknown) = read_vint(file, false)?;
	if unknown || track_bytes + 3 >= size {
		return Err(Error::data_loss(
			"Matroska SimpleBlock has invalid track header",
		));
	}
	let mut header = [0_u8; 3];
	file
		.read_exact(&mut header)
		.map_err(|source| Error::io("Matroska block header read", source))?;
	if track_number != track.number || header[2] & 0x06 != 0 {
		return Ok(());
	}
	let relative = i64::from(i16::from_be_bytes([header[0], header[1]]));
	let timestamp = cluster_timecode
		.checked_add_signed(relative)
		.ok_or_else(|| {
			Error::data_loss("Matroska block presentation timestamp is negative or overflows")
		})?;
	let duration = if track.default_duration_ns == 0 {
		1
	} else {
		(track.default_duration_ns / scale).max(1)
	};
	let duration = u32::try_from(duration)
		.map_err(|_| Error::out_of_range("Matroska block duration exceeds u32"))?;
	let size = size - track_bytes - 3;
	if size > MAX_PACKET_BYTES as u64 || size > u64::from(u32::MAX) {
		return Err(Error::resource_exhausted("Matroska packet exceeds bound"));
	}
	if samples.len() >= MAX_MP4_TABLE_ENTRIES {
		return Err(Error::resource_exhausted(
			"Matroska sample count exceeds bound",
		));
	}
	samples
		.try_reserve(1)
		.map_err(|_| Error::resource_exhausted("Matroska sample index allocation failed"))?;
	samples.push(Sample {
		offset: block.data_offset + track_bytes + 3,
		size: size as u32,
		dts: timestamp,
		pts: timestamp,
		duration,
		keyframe: header[2] & 0x80 != 0,
	});
	Ok(())
}

fn read_element(file: &mut File, limit: u64) -> Result<Option<Element>> {
	let start = file
		.stream_position()
		.map_err(|source| Error::io("Matroska position", source))?;
	if start == limit {
		return Ok(None);
	}
	if start > limit {
		return Err(Error::data_loss("Matroska child exceeds parent"));
	}
	let (id, _, _) = read_vint(file, true)?;
	let (size, _, unknown) = read_vint(file, false)?;
	let data_offset = file
		.stream_position()
		.map_err(|source| Error::io("Matroska position", source))?;
	if data_offset > limit {
		return Err(Error::data_loss("Matroska element header exceeds parent"));
	}
	let end = if unknown {
		limit
	} else {
		data_offset
			.checked_add(size)
			.ok_or_else(|| Error::out_of_range("Matroska element end overflow"))?
	};
	if end > limit || end <= start {
		return Err(Error::data_loss("Matroska element exceeds parent"));
	}
	Ok(Some(Element {
		id,
		data_offset,
		end,
	}))
}

fn read_vint(file: &mut File, keep_marker: bool) -> Result<(u64, u64, bool)> {
	let mut first = [0_u8; 1];
	file
		.read_exact(&mut first)
		.map_err(|source| Error::io("Matroska variable integer read", source))?;
	let length = first[0].leading_zeros() as u64 + 1;
	if length > 8 || (keep_marker && length > 4) {
		return Err(Error::data_loss(
			"Matroska variable integer has invalid width",
		));
	}
	let marker = 0x80_u8 >> (length - 1);
	let mut value = if keep_marker {
		u64::from(first[0])
	} else {
		u64::from(first[0] & (marker - 1))
	};
	for _ in 1..length {
		file
			.read_exact(&mut first)
			.map_err(|source| Error::io("Matroska variable integer read", source))?;
		value = (value << 8) | u64::from(first[0]);
	}
	let unknown = !keep_marker && value == (1_u64 << (7 * length)) - 1;
	Ok((value, length, unknown))
}

fn read_unsigned(file: &mut File, element: Element) -> Result<u64> {
	let size = element.end - element.data_offset;
	if size == 0 || size > 8 {
		return Err(Error::data_loss(
			"Matroska unsigned field has invalid width",
		));
	}
	let mut value = 0_u64;
	let mut byte = [0_u8; 1];
	for _ in 0..size {
		file
			.read_exact(&mut byte)
			.map_err(|source| Error::io("Matroska unsigned field read", source))?;
		value = (value << 8) | u64::from(byte[0]);
	}
	Ok(value)
}

fn read_string(file: &mut File, element: Element) -> Result<String> {
	String::from_utf8(read_bytes(file, element)?)
		.map_err(|_| Error::data_loss("Matroska codec id is not UTF-8"))
}

fn read_bytes(file: &mut File, element: Element) -> Result<Vec<u8>> {
	let size = usize::try_from(element.end - element.data_offset)
		.map_err(|_| Error::out_of_range("Matroska field exceeds usize"))?;
	let mut bytes = Vec::new();
	bytes
		.try_reserve_exact(size)
		.map_err(|_| Error::resource_exhausted("Matroska field allocation failed"))?;
	bytes.resize(size, 0);
	file
		.read_exact(&mut bytes)
		.map_err(|source| Error::io("Matroska field read", source))?;
	Ok(bytes)
}

fn seek(file: &mut File, offset: u64) -> Result<()> {
	file
		.seek(SeekFrom::Start(offset))
		.map_err(|source| Error::io("Matroska seek", source))?;
	Ok(())
}
