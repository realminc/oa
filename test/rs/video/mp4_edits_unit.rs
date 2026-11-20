use super::*;
use crate::ErrorKind;

fn edits(version: u8, entries: &[(u64, i64, u32)]) -> Vec<u8> {
	let mut data = vec![version, 0, 0, 0];
	data.extend((entries.len() as u32).to_be_bytes());
	for &(length, start, rate) in entries {
		if version == 1 {
			data.extend(length.to_be_bytes());
			data.extend(start.to_be_bytes());
		} else {
			data.extend((length as u32).to_be_bytes());
			data.extend((start as i32).to_be_bytes());
		}
		data.extend(rate.to_be_bytes());
	}
	data
}

#[test]
fn mp4_edits_map_delayed_tracks_and_aac_priming() -> Result<()> {
	let unit = 0x0001_0000;
	for version in [0, 1] {
		let data = edits(version, &[(250, -1, unit), (1750, 1024, unit)]);
		let edit = parse_edit_list(&data, 1000, 48_000)?;
		assert_eq!(edit.movie_start, Duration::from_millis(250));
		assert_eq!(edit.media_start, Duration::from_nanos(21_333_333));
		assert_eq!(edit.length, Some(Duration::from_millis(1750)));
	}
	let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("test/fixtures/media/embedded_av.mp4");
	let [video, audio] = mp4_track_edits(&path, [Some(1), Some(2)])?;
	assert_eq!(video.movie_start, Duration::ZERO);
	assert_eq!(video.length, Some(Duration::from_secs(1)));
	assert_eq!(audio.movie_start, Duration::ZERO);
	assert_eq!(audio.media_start, Duration::from_nanos(23_219_954));
	assert_eq!(audio.length, Some(Duration::from_secs(1)));
	Ok(())
}

#[test]
fn mp4_edits_reject_truncation_and_unsupported_rate_sequences() {
	let unit = 0x0001_0000;
	let data = edits(1, &[(250, -1, unit), (1750, 1024, unit)]);
	assert_eq!(
		parse_edit_list(&data[..data.len() - 1], 1000, 48_000)
			.unwrap_err()
			.kind(),
		ErrorKind::DataLoss
	);
	for entries in [vec![(1, 0, 0)], vec![(1, 0, unit), (1, 100, unit)]] {
		assert_eq!(
			parse_edit_list(&edits(0, &entries), 1000, 48_000)
				.unwrap_err()
				.kind(),
			ErrorKind::MissingCapability
		);
	}
}
