//! Sequential MPEG-TS PAT/PMT and H.264/H.265 PES demux.
//!
//! Donor: `source/cpp/lib/oa/vision/video/videoDemuxer.cpp`, `parseTsPayload`,
//! `parseTsPat`, `parseTsPmt`, `readMpegTsPes`, and `initMpegTs`. Only one
//! local 188-byte-packet transport and one selected video PID are admitted.

use std::{
	fs::File,
	io::{Read, Seek, SeekFrom},
};

use crate::{Error, Result};

use super::{
	H265Profile, MAX_PACKET_BYTES, VideoCodec, VideoContainerInfo, VideoContainerKind,
	VideoDecodeProfile, VideoPacket, VideoTimeBase,
};

const PACKET_BYTES: usize = 188;
const NULL_PID: u16 = 0x1fff;

pub(super) struct TsState {
	file_size: u64,
	video_pid: u16,
	codec: VideoCodec,
	pending: Option<PendingPes>,
}

struct PendingPes {
	data: Vec<u8>,
	pts: u64,
	dts: u64,
}

struct Payload<'a> {
	pid: u16,
	start: bool,
	bytes: &'a [u8],
}

pub(super) fn open(file: &mut File, file_size: u64) -> Result<(VideoContainerInfo, TsState)> {
	if file_size < PACKET_BYTES as u64 || !file_size.is_multiple_of(PACKET_BYTES as u64) {
		return Err(Error::data_loss(
			"MPEG-TS size is not a multiple of 188 bytes",
		));
	}
	file
		.seek(SeekFrom::Start(0))
		.map_err(|source| Error::io("MPEG-TS scan rewind", source))?;
	let mut pmt_pid = NULL_PID;
	let mut selected = None;
	let mut packet = [0_u8; PACKET_BYTES];
	for _ in 0..(file_size / PACKET_BYTES as u64).min(8192) {
		file
			.read_exact(&mut packet)
			.map_err(|source| Error::io("MPEG-TS discovery read", source))?;
		let Some(payload) = parse_payload(&packet)? else {
			continue;
		};
		if payload.pid == 0 {
			if let Some(pid) = parse_pat(payload) {
				pmt_pid = pid;
			}
		} else if payload.pid == pmt_pid
			&& let Some(video) = parse_pmt(payload)
		{
			selected = Some(video);
			break;
		}
	}
	let (video_pid, codec) = selected
		.ok_or_else(|| Error::data_loss("MPEG-TS PAT/PMT has no supported H.264/H.265 stream"))?;
	let mut state = TsState {
		file_size,
		video_pid,
		codec,
		pending: None,
	};
	file
		.seek(SeekFrom::Start(0))
		.map_err(|source| Error::io("MPEG-TS probe rewind", source))?;
	let first = state
		.read_next(file)?
		.ok_or_else(|| Error::data_loss("MPEG-TS video PID contains no PES payload"))?;
	let (width, height, decode_profile) = stream_profile(codec, first.data())?;
	file
		.seek(SeekFrom::Start(0))
		.map_err(|source| Error::io("MPEG-TS data rewind", source))?;
	state.reset();
	Ok((
		VideoContainerInfo {
			kind: VideoContainerKind::MpegTs,
			codec,
			width,
			height,
			time_base: VideoTimeBase::new(1, 90_000)?,
			duration: 0,
			sample_count: 0,
			track_id: u32::from(video_pid),
			track_count: 1,
			decode_profile,
		},
		state,
	))
}

impl TsState {
	pub(super) fn reset(&mut self) {
		self.pending = None;
	}

	pub(super) fn read_next(&mut self, file: &mut File) -> Result<Option<VideoPacket>> {
		let mut packet = [0_u8; PACKET_BYTES];
		loop {
			let offset = file
				.stream_position()
				.map_err(|source| Error::io("MPEG-TS packet position", source))?;
			if offset == self.file_size {
				return Ok(
					self
						.pending
						.take()
						.filter(|pes| !pes.data.is_empty())
						.map(|pes| self.finish(pes)),
				);
			}
			if offset > self.file_size || self.file_size - offset < PACKET_BYTES as u64 {
				return Err(Error::data_loss("truncated MPEG-TS packet"));
			}
			file
				.read_exact(&mut packet)
				.map_err(|source| Error::io("MPEG-TS packet read", source))?;
			let Some(payload) = parse_payload(&packet)? else {
				continue;
			};
			if payload.pid != self.video_pid || payload.bytes.is_empty() {
				continue;
			}
			if payload.start {
				if self
					.pending
					.as_ref()
					.is_some_and(|pes| !pes.data.is_empty())
				{
					file
						.seek(SeekFrom::Current(-(PACKET_BYTES as i64)))
						.map_err(|source| Error::io("MPEG-TS PES boundary rewind", source))?;
					let previous = self
						.pending
						.take()
						.ok_or_else(|| Error::internal("missing PES after boundary test"))?;
					return Ok(Some(self.finish(previous)));
				}
				self.pending = parse_pes_header(payload.bytes)?;
				continue;
			}
			if let Some(pes) = self.pending.as_mut() {
				append_payload(&mut pes.data, payload.bytes)?;
			}
		}
	}

	fn finish(&self, pes: PendingPes) -> VideoPacket {
		let keyframe = super::super::parse_nal_annex_b(&pes.data)
			.iter()
			.any(|nal| {
				let Some(&header) = nal.payload().first() else {
					return false;
				};
				match self.codec {
					VideoCodec::H264 => header & 0x1f == 5,
					VideoCodec::H265 => matches!((header >> 1) & 0x3f, 19..=21),
					_ => false,
				}
			});
		VideoPacket {
			data: pes.data,
			presentation_timestamp: pes.pts,
			decode_timestamp: pes.dts,
			duration: 0,
			keyframe,
			track_id: u32::from(self.video_pid),
		}
	}
}

fn parse_payload(packet: &[u8; PACKET_BYTES]) -> Result<Option<Payload<'_>>> {
	if packet[0] != 0x47 {
		return Err(Error::data_loss("MPEG-TS sync byte is invalid"));
	}
	if packet[1] & 0x80 != 0 {
		return Ok(None);
	}
	let pid = (u16::from(packet[1] & 0x1f) << 8) | u16::from(packet[2]);
	let control = (packet[3] >> 4) & 3;
	if control == 0 {
		return Err(Error::data_loss("MPEG-TS adaptation control is reserved"));
	}
	if control == 2 {
		return Ok(None);
	}
	let offset = if control == 3 {
		5 + usize::from(packet[4])
	} else {
		4
	};
	if offset > PACKET_BYTES {
		return Err(Error::data_loss("MPEG-TS adaptation field exceeds packet"));
	}
	Ok(Some(Payload {
		pid,
		start: packet[1] & 0x40 != 0,
		bytes: &packet[offset..],
	}))
}

fn psi_section(payload: Payload<'_>) -> Option<&[u8]> {
	let offset = if payload.start {
		1 + usize::from(*payload.bytes.first()?)
	} else {
		0
	};
	payload.bytes.get(offset..)
}

fn parse_pat(payload: Payload<'_>) -> Option<u16> {
	let section = psi_section(payload)?;
	if section.len() < 12 || section[0] != 0 {
		return None;
	}
	let length = (usize::from(section[1] & 15) << 8) | usize::from(section[2]);
	if length < 9 || length + 3 > section.len() {
		return None;
	}
	let end = 3 + length - 4;
	for p in (8..end).step_by(4) {
		if p + 4 > end {
			break;
		}
		let program = u16::from_be_bytes([section[p], section[p + 1]]);
		if program != 0 {
			return Some((u16::from(section[p + 2] & 31) << 8) | u16::from(section[p + 3]));
		}
	}
	None
}

fn parse_pmt(payload: Payload<'_>) -> Option<(u16, VideoCodec)> {
	let section = psi_section(payload)?;
	if section.len() < 16 || section[0] != 2 {
		return None;
	}
	let length = (usize::from(section[1] & 15) << 8) | usize::from(section[2]);
	if length < 13 || length + 3 > section.len() {
		return None;
	}
	let end = 3 + length - 4;
	let program_info = (usize::from(section[10] & 15) << 8) | usize::from(section[11]);
	let mut p = 12 + program_info;
	while p + 5 <= end {
		let codec = match section[p] {
			0x1b => Some(VideoCodec::H264),
			0x24 => Some(VideoCodec::H265),
			_ => None,
		};
		let pid = (u16::from(section[p + 1] & 31) << 8) | u16::from(section[p + 2]);
		let info_len = (usize::from(section[p + 3] & 15) << 8) | usize::from(section[p + 4]);
		if let Some(codec) = codec {
			return Some((pid, codec));
		}
		p = p.checked_add(5 + info_len)?;
	}
	None
}

fn parse_pes_header(bytes: &[u8]) -> Result<Option<PendingPes>> {
	if bytes.len() < 9 || bytes[..3] != [0, 0, 1] {
		return Ok(None);
	}
	let flags = (bytes[7] >> 6) & 3;
	let header_len = usize::from(bytes[8]);
	let payload_offset = 9 + header_len;
	if payload_offset > bytes.len() {
		return Err(Error::data_loss("MPEG-TS PES header exceeds packet"));
	}
	let pts = if flags & 2 != 0 && header_len >= 5 {
		parse_timestamp(&bytes[9..14])?
	} else {
		0
	};
	let dts = if flags == 3 && header_len >= 10 {
		parse_timestamp(&bytes[14..19])?
	} else {
		pts
	};
	let mut pes = PendingPes {
		data: Vec::new(),
		pts,
		dts,
	};
	append_payload(&mut pes.data, &bytes[payload_offset..])?;
	Ok(Some(pes))
}

fn parse_timestamp(bytes: &[u8]) -> Result<u64> {
	if bytes.len() != 5 {
		return Err(Error::data_loss("MPEG-TS timestamp is truncated"));
	}
	Ok(
		(u64::from((bytes[0] >> 1) & 7) << 30)
			| (u64::from(bytes[1]) << 22)
			| (u64::from((bytes[2] >> 1) & 0x7f) << 15)
			| (u64::from(bytes[3]) << 7)
			| u64::from(bytes[4] >> 1),
	)
}

fn append_payload(output: &mut Vec<u8>, payload: &[u8]) -> Result<()> {
	if output
		.len()
		.checked_add(payload.len())
		.is_none_or(|size| size > MAX_PACKET_BYTES)
	{
		return Err(Error::resource_exhausted(
			"MPEG-TS PES exceeds packet bound",
		));
	}
	output
		.try_reserve(payload.len())
		.map_err(|_| Error::resource_exhausted("MPEG-TS PES allocation failed"))?;
	output.extend_from_slice(payload);
	Ok(())
}

fn stream_profile(
	codec: VideoCodec,
	bytes: &[u8],
) -> Result<(u32, u32, Option<VideoDecodeProfile>)> {
	for nal in super::super::parse_nal_annex_b(bytes) {
		let payload = nal.payload();
		let Some(&header) = payload.first() else {
			continue;
		};
		match codec {
			VideoCodec::H264 if header & 0x1f == 7 => {
				let sps = super::super::parse_h264_sps(payload)?;
				let width = sps.coded_width()?;
				let height = sps.coded_height()?;
				let frame_factor = if sps.frame_mbs_only { 1_u32 } else { 2 };
				let crop_x_unit = if matches!(sps.chroma_format_idc, 0 | 3) {
					1
				} else {
					2
				};
				let crop_y_unit = if sps.chroma_format_idc == 1 {
					2 * frame_factor
				} else {
					frame_factor
				};
				let crop_x = sps
					.frame_crop_left_offset
					.checked_add(sps.frame_crop_right_offset)
					.and_then(|crop| crop.checked_mul(crop_x_unit))
					.ok_or_else(|| Error::data_loss("H.264 TS crop overflows"))?;
				let crop_y = sps
					.frame_crop_top_offset
					.checked_add(sps.frame_crop_bottom_offset)
					.and_then(|crop| crop.checked_mul(crop_y_unit))
					.ok_or_else(|| Error::data_loss("H.264 TS crop overflows"))?;
				let width = width
					.checked_sub(crop_x)
					.filter(|value| *value > 0)
					.ok_or_else(|| Error::data_loss("H.264 TS width crop is invalid"))?;
				let height = height
					.checked_sub(crop_y)
					.filter(|value| *value > 0)
					.ok_or_else(|| Error::data_loss("H.264 TS height crop is invalid"))?;
				let profile = super::h264_profile_from_sps(payload)?;
				return Ok((width, height, profile));
			}
			VideoCodec::H265 if (header >> 1) & 0x3f == 33 => {
				let sps = super::super::parse_h265_sps(payload)?;
				let profile = match sps.profile_tier_level.profile_idc {
					1 => Some(H265Profile::Main),
					2 => Some(H265Profile::Main10),
					3 => Some(H265Profile::MainStillPicture),
					4 => Some(H265Profile::FormatRangeExtensions),
					9 => Some(H265Profile::ScreenContentCodingExtensions),
					_ => None,
				};
				let depth_luma = super::parse_bit_depth(sps.bit_depth_luma_minus_8 + 8)?;
				let depth_chroma = super::parse_bit_depth(sps.bit_depth_chroma_minus_8 + 8)?;
				let chroma = super::parse_chroma(sps.chroma_format_idc)?;
				let profile = profile.map(|profile| VideoDecodeProfile::H265 {
					profile,
					chroma_subsampling: chroma,
					luma_bit_depth: depth_luma,
					chroma_bit_depth: depth_chroma,
				});
				return Ok((sps.width, sps.height, profile));
			}
			_ => {}
		}
	}
	Err(Error::data_loss(
		"MPEG-TS first PES lacks a usable video SPS",
	))
}
