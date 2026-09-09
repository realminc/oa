use oa::{Audio, AudioChannelLayout, ErrorKind};

#[test]
fn channel_count_inference_preserves_ambiguous_three_channel_layout() {
	assert_eq!(
		AudioChannelLayout::for_channel_count(1),
		AudioChannelLayout::Mono
	);
	assert_eq!(
		AudioChannelLayout::for_channel_count(2),
		AudioChannelLayout::Stereo
	);
	assert_eq!(
		AudioChannelLayout::for_channel_count(3),
		AudioChannelLayout::Unknown
	);
	assert_eq!(AudioChannelLayout::Stereo21.channel_count(), Some(3));
}

#[test]
fn wav_f32_writer_emits_checked_ieee_float_header_and_payload() -> oa::Result<()> {
	let samples = [0.0_f32, -0.25, 0.5, 1.0];
	let wav = oa::audio::encode_interleaved_wav_f32(&samples, 48_000, 2)?;
	assert_eq!(&wav[0..4], b"RIFF");
	assert_eq!(&wav[8..16], b"WAVEfmt ");
	assert_eq!(u16::from_le_bytes([wav[20], wav[21]]), 3);
	assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 2);
	assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 48_000);
	assert_eq!(&wav[38..42], b"data");
	assert_eq!(u32::from_le_bytes(wav[42..46].try_into().unwrap()), 16);
	for (index, expected) in samples.iter().enumerate() {
		let offset = 46 + index * 4;
		assert_eq!(
			f32::from_le_bytes(wav[offset..offset + 4].try_into().unwrap()).to_bits(),
			expected.to_bits()
		);
	}
	Ok(())
}

#[test]
fn wav_writer_rejects_malformed_shapes_and_rates() {
	for (samples, rate, channels) in [
		(&[][..], 48_000, 1),
		(&[0.0, 0.1, 0.2][..], 48_000, 2),
		(&[0.0][..], 0, 1),
		(&[0.0][..], 48_000, 0),
	] {
		let error = oa::audio::encode_interleaved_wav_f32(samples, rate, channels)
			.expect_err("invalid audio input should fail");
		assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	}
}

test_vk!(
	wav_decode_round_trip_preserves_planar_samples_and_metadata,
	engine,
	{
		let interleaved = [0.0_f32, 1.0, 0.25, 0.75, -0.5, 0.5, -1.0, 0.0];
		let wav = oa::audio::encode_interleaved_wav_f32(&interleaved, 22_050, 2)?;
		let decoded = oa::audio::decode_memory(&engine, &wav)?;
		assert_eq!(decoded.channels(), 2);
		assert_eq!(decoded.samples(), 4);
		assert_eq!(decoded.sample_rate(), 22_050);
		assert_eq!(decoded.layout(), AudioChannelLayout::Stereo);
		assert_eq!(decoded.duration_seconds(), 4.0 / 22_050.0);
		assert_eq!(
			decoded.as_matrix().read_f32()?,
			[0.0, 0.25, -0.5, -1.0, 1.0, 0.75, 0.5, 0.0]
		);
		assert_eq!(oa::audio::encode_wav_f32(&decoded)?, wav);
		Ok(())
	}
);

test_vk!(audio_value_rejects_invalid_matrix_metadata, engine, {
	let mono = oa::Matrix::from_f32(&engine, [1, 4], &[0.0; 4])?;
	assert!(Audio::new(mono.clone(), 0, AudioChannelLayout::Mono).is_err());
	assert!(Audio::new(mono, 48_000, AudioChannelLayout::Stereo).is_err());

	let integer = oa::Matrix::from_slice(&engine, [1, 4], &[0_i32; 4])?;
	assert!(Audio::new(integer, 48_000, AudioChannelLayout::Mono).is_err());
	let rank_one = oa::Matrix::from_f32(&engine, [4], &[0.0; 4])?;
	assert!(Audio::new(rank_one, 48_000, AudioChannelLayout::Mono).is_err());
	assert!(oa::audio::decode_memory(&engine, &[]).is_err());
	assert!(oa::audio::decode_memory(&engine, b"not audio").is_err());
	Ok(())
});

test_vk!(
	real_wav_flac_and_mp3_assets_decode,
	engine,
	"requires a hardware Vulkan 1.3 compute device and OA_DATA_DIR/audio/oaNarration assets",
	{
		let asset_root = std::env::var_os("OA_DATA_DIR").expect("OA_DATA_DIR is required");
		let audio_root = std::path::PathBuf::from(asset_root).join("audio");
		let wav = oa::audio::decode_file(&engine, audio_root.join("oaNarration.wav"))?;
		let flac = oa::audio::decode_file(&engine, audio_root.join("oaNarration.flac"))?;
		assert_eq!(wav.sample_rate(), 24_000);
		assert_eq!(flac.sample_rate(), wav.sample_rate());
		assert_eq!(flac.channels(), wav.channels());
		assert_eq!(flac.samples(), wav.samples());
		let wav_samples = wav.as_matrix().read_f32()?;
		let flac_samples = flac.as_matrix().read_f32()?;
		for index in (0..wav_samples.len()).step_by(31) {
			assert_eq!(flac_samples[index], wav_samples[index]);
		}

		let mp3 = oa::audio::decode_file(&engine, audio_root.join("oaNarration.mp3"))?;
		assert_eq!(mp3.sample_rate(), 24_000);
		assert_eq!(mp3.channels(), 1);
		assert!(mp3.samples() > 240_000);
		let mp3_samples = mp3.as_matrix().read_f32()?;
		assert!(mp3_samples.iter().all(|sample| sample.is_finite()));
		let mean_square = mp3_samples
			.iter()
			.map(|sample| f64::from(*sample) * f64::from(*sample))
			.sum::<f64>()
			/ mp3_samples.len() as f64;
		assert!(mean_square.sqrt() > 0.01);
		Ok(())
	}
);
