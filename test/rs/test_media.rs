use std::{any::TypeId, path::PathBuf, time::Duration};

use oa::{
	MediaPlayer,
	media::{MediaPlayerConfig, MediaTimestamp},
	video::{VideoFrameTiming, VideoTimeBase},
};

#[test]
fn public_media_session_has_one_identity_and_explicit_track_sources() {
	assert_eq!(
		TypeId::of::<MediaPlayer>(),
		TypeId::of::<oa::media::MediaPlayer>()
	);
	let config = MediaPlayerConfig::new("video.mp4", "audio.wav");
	assert_eq!(config.video_path, PathBuf::from("video.mp4"));
	assert_eq!(config.audio_path, PathBuf::from("audio.wav"));
}

#[test]
fn video_never_advances_before_its_next_presentation_boundary() {
	let timing = VideoFrameTiming::new(Duration::from_millis(40), Some(Duration::from_millis(20)))
		.expect("valid timing");
	assert!(
		!MediaTimestamp::from_microseconds(59_000)
			.frame_is_due(timing)
			.unwrap()
	);
	assert!(
		MediaTimestamp::from_microseconds(60_000)
			.frame_is_due(timing)
			.unwrap()
	);
	assert!(
		MediaTimestamp::from_microseconds(90_000)
			.frame_is_due(timing)
			.unwrap()
	);
	let unknown = VideoFrameTiming::new(Duration::ZERO, None).unwrap();
	assert!(
		MediaTimestamp::from_microseconds(1_000_000)
			.frame_is_due(unknown)
			.is_err()
	);
}

#[test]
fn microsecond_seek_uses_exact_rational_time_base() {
	let base = VideoTimeBase::new(1, 90_000).unwrap();
	assert_eq!(
		MediaTimestamp::from_microseconds(1_000_000)
			.to_video_ticks(base)
			.unwrap(),
		90_000
	);
	assert_eq!(
		MediaTimestamp::from_microseconds(33_333)
			.to_video_ticks(base)
			.unwrap(),
		2_999
	);
	let tiny_unit = VideoTimeBase::new(1, u32::MAX).unwrap();
	assert!(
		MediaTimestamp::from_microseconds(u64::MAX)
			.to_video_ticks(tiny_unit)
			.is_err()
	);
}
