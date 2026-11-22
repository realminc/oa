//! ISO BMFF media-to-movie timeline mapping for selected playback tracks.

use std::{fs::File, path::Path, time::Duration};

use crate::{Error, Result};

use super::{
	boxes, find_descendant, parse_mdhd, parse_tkhd_track_id, read_moov, read_u32, read_u64,
};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Mp4TrackEdit {
	pub movie_start: Duration,
	pub media_start: Duration,
	pub length: Option<Duration>,
}

pub(crate) fn mp4_track_edits(
	path: &Path,
	track_ids: [Option<u32>; 2],
) -> Result<[Mp4TrackEdit; 2]> {
	let mut file = File::open(path).map_err(|source| Error::io("MP4 edit list open", source))?;
	let size = file
		.metadata()
		.map_err(|source| Error::io("MP4 edit list metadata", source))?
		.len();
	let moov = read_moov(&mut file, size)?;
	let mvhd =
		find_descendant(&moov, b"mvhd")?.ok_or_else(|| Error::data_loss("MP4 has no movie header"))?;
	let movie_scale = match mvhd.first() {
		Some(0) => read_u32(mvhd, 12)?,
		Some(1) => read_u32(mvhd, 20)?,
		_ => return Err(Error::data_loss("MP4 movie header has an unknown version")),
	};
	let mut edits = [Mp4TrackEdit::default(); 2];
	for trak in boxes(&moov)?
		.into_iter()
		.filter(|child| child.kind == *b"trak")
	{
		let Some(id) = find_descendant(trak.payload, b"tkhd")?.and_then(parse_tkhd_track_id) else {
			continue;
		};
		let Some(index) = track_ids.iter().position(|&selected| selected == Some(id)) else {
			continue;
		};
		let Some(edts) = find_descendant(trak.payload, b"edts")? else {
			continue;
		};
		let elst = find_descendant(edts, b"elst")?
			.ok_or_else(|| Error::data_loss("MP4 edit box lacks an edit list"))?;
		let mdia = find_descendant(trak.payload, b"mdia")?
			.ok_or_else(|| Error::data_loss("MP4 edited track lacks a media box"))?;
		let (media_scale, _) = find_descendant(mdia, b"mdhd")?
			.and_then(parse_mdhd)
			.ok_or_else(|| Error::data_loss("MP4 edited track lacks a valid media header"))?;
		edits[index] = parse_edit_list(elst, movie_scale, media_scale)?;
	}
	Ok(edits)
}

fn parse_edit_list(data: &[u8], movie_scale: u32, media_scale: u32) -> Result<Mp4TrackEdit> {
	let wide = match data.first() {
		Some(0) => false,
		Some(1) => true,
		_ => return Err(Error::data_loss("MP4 edit list has an unknown version")),
	};
	let count = read_u32(data, 4)?;
	let stride = if wide { 20 } else { 12 };
	if data.len().saturating_sub(8) / stride < count as usize {
		return Err(Error::data_loss("truncated MP4 edit list"));
	}
	let mut edit = Mp4TrackEdit::default();
	for entry in data[8..].chunks_exact(stride).take(count as usize) {
		let (length, start) = if wide {
			(read_u64(entry, 0)?, read_u64(entry, 8)?.cast_signed())
		} else {
			(
				u64::from(read_u32(entry, 0)?),
				i64::from(read_u32(entry, 4)?.cast_signed()),
			)
		};
		if edit.length.is_some()
			|| (start < 0 && start != -1)
			|| read_u32(entry, stride - 4)? != 0x0001_0000
		{
			return Err(Error::missing_capability(
				"MP4 playback requires one continuous, unit-rate media edit",
			));
		}
		let length = duration(length, movie_scale)?;
		if start == -1 {
			edit.movie_start = edit
				.movie_start
				.checked_add(length)
				.ok_or_else(|| Error::out_of_range("MP4 edit start exceeds Duration"))?;
		} else {
			edit.media_start = duration(start.unsigned_abs(), media_scale)?;
			edit.length = Some(length);
		}
	}
	if count != 0 && edit.length.is_none() {
		return Err(Error::missing_capability(
			"MP4 edit list has no playable segment",
		));
	}
	Ok(edit)
}

fn duration(ticks: u64, scale: u32) -> Result<Duration> {
	let scale = u64::from(scale);
	let seconds = ticks
		.checked_div(scale)
		.ok_or_else(|| Error::data_loss("MP4 timescale is zero"))?;
	Ok(Duration::new(
		seconds,
		((ticks % scale) * 1_000_000_000 / scale) as u32,
	))
}

#[cfg(test)]
#[path = "../../../../../test/rs/video/mp4_edits_unit.rs"]
mod tests;
