use oa::{
	Audio, ErrorKind,
	audio::{
		AudioCaptureConfig, AudioChannelLayout, AudioEncodeProfile, AudioEncoder, AudioPlayerConfig,
	},
};

#[test]
fn root_exports_are_identities_of_audio_contracts() {
	use std::any::TypeId;

	assert_eq!(TypeId::of::<oa::Audio>(), TypeId::of::<oa::audio::Audio>());
	assert_eq!(
		TypeId::of::<oa::AudioEncoder>(),
		TypeId::of::<oa::audio::AudioEncoder>()
	);
	assert_eq!(
		TypeId::of::<oa::AudioPlayer>(),
		TypeId::of::<oa::audio::AudioPlayer>()
	);
	assert_eq!(
		TypeId::of::<oa::AudioCapture>(),
		TypeId::of::<oa::audio::AudioCapture>()
	);
}

#[test]
fn realtime_session_configs_preserve_donor_defaults() {
	let capture = AudioCaptureConfig::default();
	assert_eq!(capture.sample_rate, 48_000);
	assert_eq!(capture.channel_count, 2);
	assert_eq!(capture.ring_milliseconds, 500);

	let player = AudioPlayerConfig::new("voice.wav");
	assert_eq!(player.uri, std::path::PathBuf::from("voice.wav"));
	assert!(!player.loop_playback);
	assert_eq!(player.ring_milliseconds, 500);
}

#[test]
fn pcm_s16_encoder_packets_flush_and_close_explicitly() -> oa::Result<()> {
	let profile = AudioEncodeProfile {
		sample_rate: 48_000,
		channel_count: 2,
		frames_per_packet: 2,
		..AudioEncodeProfile::default()
	};
	let mut encoder = AudioEncoder::create(profile)?;
	assert!(encoder.is_open());
	assert_eq!(encoder.profile(), Some(&profile));
	assert!(encoder.codec_config().is_empty());
	assert_eq!(encoder.priming_frames(), 0);

	let packets = encoder.encode(&[-1.0, -0.5, 0.0, 0.5, 1.0, f32::NAN])?;
	assert_eq!(packets.len(), 1);
	assert_eq!(packets[0].presentation_frame, 0);
	assert_eq!(packets[0].duration_frames, 2);
	assert_eq!(
		packets[0].bitstream,
		[
			i16::MIN.to_le_bytes(),
			(-16_384_i16).to_le_bytes(),
			0_i16.to_le_bytes(),
			16_384_i16.to_le_bytes(),
		]
		.concat()
	);

	let tail = encoder.flush()?;
	assert_eq!(tail.len(), 1);
	assert_eq!(tail[0].presentation_frame, 2);
	assert_eq!(tail[0].duration_frames, 1);
	assert_eq!(
		tail[0].bitstream,
		[i16::MAX.to_le_bytes(), 0_i16.to_le_bytes()].concat()
	);
	assert!(encoder.flush()?.is_empty());

	encoder.close();
	assert!(!encoder.is_open());
	assert_eq!(encoder.profile(), None);
	assert_eq!(
		encoder.encode(&[0.0, 0.0]).unwrap_err().kind(),
		ErrorKind::FailedPrecondition
	);
	Ok(())
}

#[test]
fn pcm_s16_encoder_rejects_invalid_profiles_and_partial_frames() -> oa::Result<()> {
	for profile in [
		AudioEncodeProfile {
			sample_rate: 0,
			..AudioEncodeProfile::default()
		},
		AudioEncodeProfile {
			channel_count: 0,
			..AudioEncodeProfile::default()
		},
		AudioEncodeProfile {
			channel_count: 9,
			..AudioEncodeProfile::default()
		},
		AudioEncodeProfile {
			frames_per_packet: 0,
			..AudioEncodeProfile::default()
		},
	] {
		assert_eq!(
			AudioEncoder::create(profile).unwrap_err().kind(),
			ErrorKind::InvalidArgument
		);
	}

	let mut encoder = AudioEncoder::create(AudioEncodeProfile {
		channel_count: 2,
		..AudioEncodeProfile::default()
	})?;
	assert_eq!(
		encoder.encode(&[]).unwrap_err().kind(),
		ErrorKind::InvalidArgument
	);
	assert_eq!(
		encoder.encode(&[0.0]).unwrap_err().kind(),
		ErrorKind::InvalidArgument
	);
	Ok(())
}

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

test_vk!(audio_signal_operations_match_host_oracles, engine, {
	use oa::audio::{BiquadCoefficients, NormalizeAudioConfig, NormalizeAudioMode, ResampleConfig};

	fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
		assert_eq!(actual.len(), expected.len());
		for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
			assert!(
				(actual - expected).abs() <= tolerance,
				"sample {index}: expected {expected}, found {actual}"
			);
		}
	}

	let input = Audio::from_planar_f32(
		&engine,
		&[0.0, 0.25, -0.5, 1.0, 1.0, 0.5, -0.25, -1.0],
		2,
		8_000,
		AudioChannelLayout::Stereo,
	)?;
	close(
		&oa::audio::gain(&input, 6.020_6)?.as_matrix().read_f32()?,
		&[0.0, 0.5, -1.0, 2.0, 2.0, 1.0, -0.5, -2.0],
		2e-5,
	);
	close(
		&oa::audio::clip(&input, -0.3, 0.4)?.as_matrix().read_f32()?,
		&[0.0, 0.25, -0.3, 0.4, 0.4, 0.4, -0.25, -0.3],
		1e-6,
	);
	close(
		&oa::audio::pre_emphasis(&input, 0.5)?
			.as_matrix()
			.read_f32()?,
		&[0.0, 0.25, -0.625, 1.25, 1.0, 0.0, -0.5, -0.875],
		1e-6,
	);
	let mono = oa::audio::to_mono(&input)?;
	assert_eq!(mono.layout(), AudioChannelLayout::Mono);
	assert_eq!(mono.sample_rate(), input.sample_rate());
	close(
		&mono.as_matrix().read_f32()?,
		&[0.5, 0.375, -0.375, 0.0],
		1e-6,
	);
	close(
		&oa::audio::fade(&input, 2, 2)?.as_matrix().read_f32()?,
		&[0.0, 0.125, -0.25, 0.0, 0.0, 0.25, -0.125, 0.0],
		1e-6,
	);
	close(
		&oa::audio::mix(&input, &input, 0.25, 0.75)?
			.as_matrix()
			.read_f32()?,
		&input.as_matrix().read_f32()?,
		1e-6,
	);
	close(
		&oa::audio::biquad(&input, BiquadCoefficients::default())?
			.as_matrix()
			.read_f32()?,
		&input.as_matrix().read_f32()?,
		1e-6,
	);
	close(
		&oa::audio::sos_filter(
			&input,
			&[BiquadCoefficients::default(), BiquadCoefficients::default()],
		)?
		.as_matrix()
		.read_f32()?,
		&input.as_matrix().read_f32()?,
		1e-6,
	);
	let normalized = oa::audio::normalize(
		&input,
		NormalizeAudioConfig {
			target_db: 0.0,
			mode: NormalizeAudioMode::Peak,
		},
	)?;
	assert_eq!(
		normalized
			.as_matrix()
			.read_f32()?
			.into_iter()
			.map(f32::abs)
			.fold(0.0, f32::max),
		1.0
	);
	let db = oa::audio::amplitude_to_db(&input, -40.0)?.read_f32()?;
	close(&db[..4], &[-40.0, -12.041_2, -6.020_6, 0.0], 2e-4);
	assert_eq!(oa::audio::waveform_envelope(&input, 2)?.shape(), [2, 2]);
	close(
		&oa::audio::waveform_envelope(&input, 2)?.read_f32()?,
		&[0.0, 1.0, -1.0, 1.0],
		1e-6,
	);
	let resampled = oa::audio::resample(
		&input,
		ResampleConfig {
			out_rate: 16_000,
			filter_half_width: 8,
		},
	)?;
	assert_eq!(resampled.as_matrix().shape(), [2, 8]);
	assert!(
		resampled
			.as_matrix()
			.read_f32()?
			.iter()
			.all(|value| value.is_finite())
	);
	let dry_reverb = oa::audio::reverb(&input, 0.1, 0.0)?;
	assert_eq!(dry_reverb.samples(), 805);
	let rendered = dry_reverb.as_matrix().read_f32()?;
	close(&rendered[..4], &[0.0, 0.25, -0.5, 1.0], 1e-6);
	assert!(rendered[4..805].iter().all(|value| *value == 0.0));
	Ok(())
});

test_vk!(audio_feature_transforms_have_checked_layouts, engine, {
	use oa::audio::{MelConfig, MfccConfig, StftConfig};

	let samples = (0..32)
		.map(|index| (std::f32::consts::TAU * index as f32 / 8.0).sin())
		.collect::<Vec<_>>();
	let input = Audio::from_planar_f32(&engine, &samples, 1, 8_000, AudioChannelLayout::Mono)?;
	let stft_config = StftConfig {
		fft_size: 16,
		hop_size: 8,
		win_size: 16,
		center: false,
		..StftConfig::default()
	};
	let spectrum = oa::audio::stft(&input, stft_config)?;
	assert_eq!(spectrum.shape(), [1, 3, 9]);
	assert!(spectrum.read_f32()?.iter().all(|value| value.is_finite()));

	let mel_config = MelConfig {
		fft_size: 16,
		hop_size: 8,
		num_mels: 4,
		f_min: 0.0,
		f_max: 4_000.0,
		log_scale: true,
		normalize: true,
	};
	let mel = oa::audio::mel_spectrogram(&input, mel_config)?;
	assert_eq!(mel.shape(), [1, 4, 5]);
	assert!(mel.read_f32()?.iter().all(|value| value.is_finite()));
	let coefficients = oa::audio::mfcc(
		&input,
		MfccConfig {
			num_coeffs: 3,
			mel: mel_config,
		},
	)?;
	assert_eq!(coefficients.shape(), [1, 3, 5]);
	assert!(
		coefficients
			.read_f32()?
			.iter()
			.all(|value| value.is_finite())
	);
	Ok(())
});

test_vk!(
	default_audio_player_runs_incremental_session_lifecycle,
	engine,
	"requires Vulkan plus a default 48 kHz mono output device",
	{
		struct TempAudio(std::path::PathBuf);
		impl Drop for TempAudio {
			fn drop(&mut self) {
				let _ = std::fs::remove_file(&self.0);
			}
		}

		let path = std::env::temp_dir().join(format!("oa-player-{}.wav", std::process::id()));
		let temp = TempAudio(path);
		let wav = oa::audio::encode_interleaved_wav_f32(&vec![0.0; 4_800], 48_000, 1)?;
		std::fs::write(&temp.0, wav).map_err(|error| oa::Error::callback(error.to_string()))?;
		let mut player = oa::AudioPlayer::open_uri(&engine, &temp.0)?;
		assert!(player.is_open());
		assert_eq!(player.sample_rate(), 48_000);
		assert_eq!(player.channel_count(), 1);
		player.play()?;
		std::thread::sleep(std::time::Duration::from_millis(20));
		player.pause();
		player.seek(0)?;
		player.set_muted(true)?;
		assert!(player.is_muted());
		player.close()?;
		assert!(!player.is_open());
		Ok(())
	}
);

test_vk!(
	default_audio_capture_runs_explicit_session_lifecycle,
	engine,
	"requires Vulkan plus a default 48 kHz stereo input device",
	{
		let mut capture = oa::AudioCapture::open(&engine, AudioCaptureConfig::default())?;
		assert!(capture.is_open());
		assert!(!capture.is_started());
		capture.start()?;
		assert!(capture.is_started());
		std::thread::sleep(std::time::Duration::from_millis(20));
		if let Some(chunk) = capture.poll(4_096) {
			assert_eq!(chunk.sample_rate, 48_000);
			assert_eq!(chunk.channel_count, 2);
			assert_eq!(chunk.interleaved.len(), chunk.frame_count as usize * 2);
		}
		capture.stop()?;
		assert!(!capture.is_started());
		capture.close()?;
		assert!(!capture.is_open());
		Ok(())
	}
);

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
