#[macro_use]
#[path = "support/test_vk.rs"]
mod test_vk_fixture;

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

	let embedded = MediaPlayerConfig::embedded("movie.mp4");
	assert_eq!(embedded.video_path, PathBuf::from("movie.mp4"));
	assert_eq!(embedded.audio_path, embedded.video_path);
}

#[test]
fn video_never_advances_before_its_next_presentation_boundary() {
	let timing = VideoFrameTiming::new(Duration::from_millis(40), Some(Duration::from_millis(20)))
		.expect("valid timing");
	assert!(
		!MediaTimestamp::new_from_microseconds(59_000)
			.frame_is_due(timing)
			.unwrap()
	);
	assert!(
		MediaTimestamp::new_from_microseconds(60_000)
			.frame_is_due(timing)
			.unwrap()
	);
	assert!(
		MediaTimestamp::new_from_microseconds(90_000)
			.frame_is_due(timing)
			.unwrap()
	);
	let unknown = VideoFrameTiming::new(Duration::ZERO, None).unwrap();
	assert!(
		MediaTimestamp::new_from_microseconds(1_000_000)
			.frame_is_due(unknown)
			.is_err()
	);
}

#[test]
fn microsecond_seek_uses_exact_rational_time_base() {
	let base = VideoTimeBase::new(1, 90_000).unwrap();
	assert_eq!(
		MediaTimestamp::new_from_microseconds(1_000_000)
			.to_video_ticks(base)
			.unwrap(),
		90_000
	);
	assert_eq!(
		MediaTimestamp::new_from_microseconds(33_333)
			.to_video_ticks(base)
			.unwrap(),
		2_999
	);
	let tiny_unit = VideoTimeBase::new(1, u32::MAX).unwrap();
	assert!(
		MediaTimestamp::new_from_microseconds(u64::MAX)
			.to_video_ticks(tiny_unit)
			.is_err()
	);
}

test_vk!(
	embedded_container_audio_decodes_and_video_demuxes_the_same_file,
	engine,
	{
		for (fixture, bytes, codec, extension) in [
			(
				"embedded_av.mp4" as &str,
				include_bytes!("../fixtures/media/embedded_av.mp4").as_slice(),
				oa::video::VideoCodec::H264,
				"mp4" as &str,
			),
			(
				"embedded_av.mkv",
				include_bytes!("../fixtures/media/embedded_av.mkv").as_slice(),
				oa::video::VideoCodec::Vp9,
				"mkv",
			),
		] {
			let audio = oa::audio::decode_memory(&engine, bytes)?;
			assert_eq!(audio.channels(), 1, "{extension} channel count");
			assert_eq!(audio.sample_rate(), 44_100, "{extension} sample rate");
			assert!(audio.samples() > 40_000, "{extension} sample count");
			let samples = audio.as_matrix().read_f32()?;
			assert!(samples.iter().all(|sample| sample.is_finite()));
			let mean_square = samples
				.iter()
				.map(|sample| f64::from(*sample) * f64::from(*sample))
				.sum::<f64>()
				/ samples.len() as f64;
			assert!(mean_square.sqrt() > 0.05, "{extension} is not silent");

			let mut demuxer = oa::video::VideoDemuxer::open(
				PathBuf::from(env!("CARGO_MANIFEST_DIR"))
					.join("test/fixtures/media")
					.join(fixture),
			)?;
			assert_eq!(demuxer.info().codec(), codec, "{extension} video codec");
			assert!(demuxer.read_next_packet()?.is_some());
		}
		Ok(())
	}
);
