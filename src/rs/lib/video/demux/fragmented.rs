//! Bounded fragmented MP4 sample indexing.
//!
//! Donor: `source/cpp/lib/oa/vision/video/videoDemuxer.cpp::parseMoofBox`.
//! The initial `moov` retains codec configuration; `moof`/`traf`/`trun`
//! supply later packet offsets and timing. Payload stays on disk until read.

use std::{
	fs::File,
	io::{Read, Seek, SeekFrom},
};

use crate::{Error, Result};

use super::{
	MAX_MOOV_BYTES, MAX_MP4_TABLE_ENTRIES, MAX_PACKET_BYTES, ParsedMovie, Sample, boxes,
	checked_signed_add, read_file_box_header, read_u32, read_u64,
};

#[derive(Clone, Copy, Default)]
struct Defaults {
	track_id: u32,
	duration: u32,
	size: u32,
	flags: u32,
}

#[derive(Clone, Copy)]
struct FileRange {
	start: u64,
	payload_start: u64,
	end: u64,
}

pub(super) fn append(
	file: &mut File,
	file_size: u64,
	moov: &[u8],
	movie: &mut ParsedMovie,
) -> Result<()> {
	let defaults = parse_trex(moov, movie.info.track_id)?;
	let mut offset = 0_u64;
	let mut fragments = Vec::new();
	let mut media = Vec::new();
	while offset < file_size {
		file
			.seek(SeekFrom::Start(offset))
			.map_err(|source| Error::io("MP4 fragment box seek", source))?;
		let header = read_file_box_header(file, offset, file_size)?;
		if header.kind == *b"moof" || header.kind == *b"mdat" {
			let target = if header.kind == *b"moof" {
				&mut fragments
			} else {
				&mut media
			};
			if target.len() >= MAX_MP4_TABLE_ENTRIES {
				return Err(Error::resource_exhausted(
					"MP4 top-level box count exceeds bound",
				));
			}
			target
				.try_reserve(1)
				.map_err(|_| Error::resource_exhausted("MP4 box index allocation failed"))?;
			target.push(FileRange {
				start: offset,
				payload_start: file
					.stream_position()
					.map_err(|source| Error::io("MP4 box payload position", source))?,
				end: header.end,
			});
		}
		offset = header.end;
	}
	for fragment in fragments.iter().copied() {
		let len = usize::try_from(fragment.end - fragment.payload_start)
			.map_err(|_| Error::out_of_range("MP4 fragment box exceeds usize"))?;
		if len > MAX_MOOV_BYTES {
			return Err(Error::resource_exhausted(
				"MP4 fragment box exceeds metadata bound",
			));
		}
		let mut payload = Vec::new();
		payload
			.try_reserve_exact(len)
			.map_err(|_| Error::resource_exhausted("MP4 fragment allocation failed"))?;
		payload.resize(len, 0);
		file
			.seek(SeekFrom::Start(fragment.payload_start))
			.map_err(|source| Error::io("MP4 fragment payload seek", source))?;
		file
			.read_exact(&mut payload)
			.map_err(|source| Error::io("MP4 fragment read", source))?;
		let next_media = media.partition_point(|range| range.start < fragment.end);
		let implicit_offset = media
			.get(next_media)
			.map_or(fragment.end, |range| range.payload_start);
		parse_moof(
			&payload,
			fragment.start,
			implicit_offset,
			&media,
			defaults,
			movie,
		)?;
	}
	if fragments.is_empty() || movie.samples.is_empty() {
		return Err(Error::data_loss(
			"MP4 has no indexed movie samples or fragments",
		));
	}
	movie.info.sample_count = u32::try_from(movie.samples.len())
		.map_err(|_| Error::out_of_range("MP4 fragment sample count exceeds u32"))?;
	movie.info.duration = movie.samples.iter().try_fold(0_u64, |end, sample| {
		let sample_end = sample
			.pts
			.checked_add(u64::from(sample.duration))
			.ok_or_else(|| Error::out_of_range("MP4 fragment duration overflow"))?;
		Ok::<u64, Error>(end.max(sample_end))
	})?;
	Ok(())
}

fn parse_trex(moov: &[u8], track_id: u32) -> Result<Defaults> {
	for child in boxes(moov)? {
		if child.kind != *b"mvex" {
			continue;
		}
		for entry in boxes(child.payload)? {
			if entry.kind != *b"trex" || entry.payload.len() < 24 {
				continue;
			}
			if read_u32(entry.payload, 4)? == track_id {
				return Ok(Defaults {
					track_id,
					duration: read_u32(entry.payload, 12)?,
					size: read_u32(entry.payload, 16)?,
					flags: read_u32(entry.payload, 20)?,
				});
			}
		}
	}
	Err(Error::data_loss(
		"fragmented MP4 lacks matching trex defaults",
	))
}

fn parse_moof(
	data: &[u8],
	moof_offset: u64,
	implicit_offset: u64,
	media: &[FileRange],
	defaults: Defaults,
	movie: &mut ParsedMovie,
) -> Result<()> {
	for child in boxes(data)? {
		if child.kind == *b"traf" {
			parse_traf(
				child.payload,
				moof_offset,
				implicit_offset,
				media,
				defaults,
				movie,
			)?;
		}
	}
	Ok(())
}

fn parse_traf(
	data: &[u8],
	moof_offset: u64,
	implicit_offset: u64,
	media: &[FileRange],
	defaults: Defaults,
	movie: &mut ParsedMovie,
) -> Result<()> {
	let children = boxes(data)?;
	let mut track_id = 0_u32;
	let mut duration = defaults.duration;
	let mut size = defaults.size;
	let mut flags = defaults.flags;
	let mut base_offset = moof_offset;
	let mut dts = 0_u64;
	for child in &children {
		match &child.kind {
			b"tfhd" => {
				let tfhd = child.payload;
				let options = read_u32(tfhd, 0)? & 0x00ff_ffff;
				track_id = read_u32(tfhd, 4)?;
				let mut p = 8_usize;
				if options & 0x000001 != 0 {
					base_offset = read_u64(tfhd, p)?;
					p += 8;
				}
				if options & 0x000002 != 0 {
					let _ = read_u32(tfhd, p)?;
					p += 4;
				}
				if options & 0x000008 != 0 {
					duration = read_u32(tfhd, p)?;
					p += 4;
				}
				if options & 0x000010 != 0 {
					size = read_u32(tfhd, p)?;
					p += 4;
				}
				if options & 0x000020 != 0 {
					flags = read_u32(tfhd, p)?;
				}
			}
			b"tfdt" => {
				let tfdt = child.payload;
				dts = if *tfdt
					.first()
					.ok_or_else(|| Error::data_loss("truncated tfdt"))?
					== 1
				{
					read_u64(tfdt, 4)?
				} else {
					u64::from(read_u32(tfdt, 4)?)
				};
			}
			_ => {}
		}
	}
	if track_id != defaults.track_id {
		return Ok(());
	}
	let mut implicit_offset = implicit_offset;
	for child in &children {
		if child.kind != *b"trun" {
			continue;
		}
		let run = child.payload;
		let version = *run
			.first()
			.ok_or_else(|| Error::data_loss("truncated trun"))?;
		let options = read_u32(run, 0)? & 0x00ff_ffff;
		let count = read_u32(run, 4)? as usize;
		if count > MAX_MP4_TABLE_ENTRIES - movie.samples.len() {
			return Err(Error::resource_exhausted(
				"MP4 fragment sample count exceeds bound",
			));
		}
		let mut p = 8_usize;
		let mut data_offset = implicit_offset;
		if options & 0x000001 != 0 {
			let relative = i64::from(read_u32(run, p)? as i32);
			data_offset = checked_signed_add(base_offset, relative)?;
			p += 4;
		}
		let mut first_flags = flags;
		if options & 0x000004 != 0 {
			first_flags = read_u32(run, p)?;
			p += 4;
		}
		movie
			.samples
			.try_reserve(count)
			.map_err(|_| Error::resource_exhausted("MP4 fragment sample index allocation failed"))?;
		for index in 0..count {
			let sample_duration = if options & 0x000100 != 0 {
				let value = read_u32(run, p)?;
				p += 4;
				value
			} else {
				duration
			}
			.max(1);
			let sample_size = if options & 0x000200 != 0 {
				let value = read_u32(run, p)?;
				p += 4;
				value
			} else {
				size
			};
			let sample_flags = if options & 0x000400 != 0 {
				let value = read_u32(run, p)?;
				p += 4;
				value
			} else if index == 0 {
				first_flags
			} else {
				flags
			};
			let cts = if options & 0x000800 != 0 {
				let raw = read_u32(run, p)?;
				p += 4;
				if version == 1 {
					i64::from(raw as i32)
				} else {
					i64::from(raw)
				}
			} else {
				0
			};
			if sample_size == 0 || sample_size as usize > MAX_PACKET_BYTES {
				return Err(Error::data_loss("MP4 fragment sample has invalid size"));
			}
			let end = data_offset
				.checked_add(u64::from(sample_size))
				.ok_or_else(|| Error::out_of_range("MP4 fragment sample offset overflow"))?;
			let media_index = media.partition_point(|range| range.payload_start <= data_offset);
			if media_index == 0 || end > media[media_index - 1].end {
				return Err(Error::data_loss(
					"MP4 fragment sample is outside media payload",
				));
			}
			movie.samples.push(Sample {
				offset: data_offset,
				size: sample_size,
				dts,
				pts: checked_signed_add(dts, cts)?,
				duration: sample_duration,
				keyframe: sample_flags & 0x00010000 == 0,
			});
			data_offset = end;
			dts = dts
				.checked_add(u64::from(sample_duration))
				.ok_or_else(|| Error::out_of_range("MP4 fragment decode timestamp overflow"))?;
		}
		implicit_offset = data_offset;
	}
	Ok(())
}
