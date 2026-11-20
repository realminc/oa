use super::*;

#[test]
fn pcm_timestamp_jitter_gaps_and_preroll_are_distinct() -> Result<()> {
	assert_eq!(pcm_discontinuity(1, 1024, 1)?, (0, 0));
	assert_eq!(pcm_discontinuity(-44, 1024, 45)?, (0, 0));
	assert_eq!(pcm_discontinuity(4800, 1024, 1)?, (4800, 0));
	assert_eq!(pcm_discontinuity(-1500, 1024, 1)?, (0, 1024));
	assert_eq!(pcm_discontinuity(-480, 1024, 1)?, (0, 480));
	Ok(())
}

#[test]
fn embedded_audio_streams_decode_and_seek_with_a_reused_pcm_buffer() -> Result<()> {
	for name in ["embedded_av.mp4", "embedded_av.mkv"] {
		let path = Path::new(env!("CARGO_MANIFEST_DIR"))
			.join("test/fixtures/media")
			.join(name);
		let (mut decoder, _) = StreamingDecoder::open(&path, true)?.expect("embedded track");
		let rate = decoder.sample_rate;
		let mut frames = 0;
		let mut peak = 0_f32;
		while let Some((gap, samples)) = decoder.read_frames(|| false)? {
			assert!(gap < u64::from(rate) / 10, "{name}: unexpected PCM gap");
			frames += gap + samples.len() as u64;
			peak = samples.iter().fold(peak, |p, sample| p.max(sample.abs()));
		}
		assert!(
			frames.abs_diff(u64::from(rate)) <= 2_048,
			"{name}: {frames}"
		);
		assert!(peak > 0.08, "{name}: {peak}");
		let capacity = decoder.samples.capacity();
		decoder.seek(500_000)?;
		assert_eq!(decoder.next_frame, u64::from(rate) / 2);
		let (_, samples) = decoder.read_frames(|| false)?.expect("PCM after seek");
		assert!(samples.iter().any(|sample| sample.abs() > 0.01));
		assert_eq!(decoder.samples.capacity(), capacity);
	}
	Ok(())
}

#[test]
#[ignore = "requires FFmpeg AC-3 and E-AC-3 codecs"]
fn dolby_embedded_audio_seek_returns_stereo_from_the_requested_time() -> Result<()> {
	for (codec, extension) in [("ac3", "mp4"), ("eac3", "mkv")] {
		let path = std::env::temp_dir().join(format!(
			"oa-{codec}-seek-{}.{extension}",
			std::process::id()
		));
		let status = std::process::Command::new("ffmpeg")
			.args([
				"-nostdin",
				"-v",
				"error",
				"-y",
				"-f",
				"lavfi",
				"-i",
				"sine=frequency=440:sample_rate=48000:duration=2",
				"-af",
				"volume=0:enable='lt(t,1)'",
				"-ac",
				"6",
				"-c:a",
				codec,
			])
			.arg(&path)
			.status()
			.expect("FFmpeg available");
		assert!(status.success());
		let (mut decoder, duration) = StreamingDecoder::open(&path, true)?.expect("embedded track");
		assert_eq!(decoder.channels, 2);
		assert!(duration >= 2_000_000);
		let result = (|| -> Result<()> {
			for (target, silent) in [(500_000, true), (1_500_000, false), (250_000, true)] {
				decoder.seek(target)?;
				let (gap, pcm) = decoder.read_frames(|| false)?.expect("PCM after seek");
				assert_eq!(gap, 0);
				let peak = pcm
					.iter()
					.fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
				assert!(
					if silent { peak < 0.001 } else { peak > 0.01 },
					"{codec} at {target}: {peak}"
				);
			}
			Ok(())
		})();
		let _ = std::fs::remove_file(&path);
		result?;
	}
	Ok(())
}
