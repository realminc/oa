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
	let mut encoder = AudioEncoder::new(profile)?;
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
			AudioEncoder::new(profile).unwrap_err().kind(),
			ErrorKind::InvalidArgument
		);
	}

	let mut encoder = AudioEncoder::new(AudioEncodeProfile {
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
		let decoded = oa::audio::decode_memory(&wav)?;
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
	let mono = oa::Matrix::new_from_f32([1, 4], &[0.0; 4])?;
	assert!(Audio::new(mono.clone(), 0, AudioChannelLayout::Mono).is_err());
	assert!(Audio::new(mono, 48_000, AudioChannelLayout::Stereo).is_err());

	let integer = oa::Matrix::new_from_slice([1, 4], &[0_i32; 4])?;
	assert!(Audio::new(integer, 48_000, AudioChannelLayout::Mono).is_err());
	let rank_one = oa::Matrix::new_from_f32([4], &[0.0; 4])?;
	assert!(Audio::new(rank_one, 48_000, AudioChannelLayout::Mono).is_err());
	assert!(oa::audio::decode_memory(&[]).is_err());
	assert!(oa::audio::decode_memory(b"not audio").is_err());
	Ok(())
});

test_vk!(
	captured_audio_gain_clip_matches_standalone_dispatches,
	engine,
	{
		let samples: Vec<f32> = (0..514).map(|i| (i % 31 - 15) as f32 / 16.0).collect();
		let input =
			Audio::new_from_planar_f32(&engine, &samples, 2, 48_000, AudioChannelLayout::Stereo)?;
		let (gain_db, minimum, maximum) = (3.75, -0.375, 0.625);
		let gain = input.gain(gain_db)?;
		gain.as_matrix().read_f32()?;
		let expected = gain.clip(minimum, maximum)?.as_matrix().read_f32()?;

		let (plan, output) = engine.capture(|| input.gain(gain_db)?.clip(minimum, maximum))?;
		let report: serde_json::Value =
			serde_json::from_str(&plan.debug_report_json("audio gain and clip"))
				.expect("valid execution graph report");
		assert_eq!(report["stats"]["dispatches"], 1);
		assert_eq!(
			report["nodes"][0]["semantic_operations"],
			serde_json::json!([0, 1])
		);
		assert_eq!(output.sample_rate(), input.sample_rate());
		assert_eq!(output.layout(), input.layout());

		engine.submit(&plan)?.wait()?;
		let actual = output.as_matrix().read_f32()?;
		assert_eq!(actual.len(), expected.len());
		for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
			assert!(
				(actual - expected).abs() <= 1e-6,
				"sample {index}: {actual} != {expected}"
			);
		}
		Ok(())
	}
);

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

	let input = Audio::new_from_planar_f32(
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

test_vk!(
	audio_upsampling_writes_past_the_input_dispatch_extent,
	engine,
	{
		let samples = vec![1.0_f32; 257];
		let input = Audio::new_from_planar_f32(&engine, &samples, 1, 8_000, AudioChannelLayout::Mono)?;
		let output = oa::audio::resample(
			&input,
			oa::audio::ResampleConfig {
				out_rate: 16_000,
				filter_half_width: 8,
			},
		)?;
		assert_eq!(output.samples(), 514);
		let values = output.as_matrix().read_f32()?;
		// At every integral source position the windowed sinc reproduces the sample,
		// including output 512, beyond the old input-sized two-workgroup dispatch.
		for index in (0..514).step_by(2) {
			assert!(
				(values[index] - 1.0).abs() < 2e-5,
				"output {index}: {}",
				values[index]
			);
		}
		Ok(())
	}
);

test_vk!(audio_feature_transforms_have_checked_layouts, engine, {
	use oa::audio::{MelConfig, MfccConfig, StftConfig};

	let samples = (0..32)
		.map(|index| (std::f32::consts::TAU * index as f32 / 8.0).sin())
		.collect::<Vec<_>>();
	let input = Audio::new_from_planar_f32(&engine, &samples, 1, 8_000, AudioChannelLayout::Mono)?;
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
		let mut player = oa::AudioPlayer::new_from_uri(&engine, &temp.0)?;
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
		while capture.poll(4_096).is_some() {}
		std::thread::sleep(std::time::Duration::from_millis(20));
		assert!(
			capture.poll(4_096).is_none(),
			"stopped capture must not deliver frames"
		);
		capture.start()?;
		assert!(capture.is_started());
		std::thread::sleep(std::time::Duration::from_millis(20));
		if let Some(chunk) = capture.poll(4_096) {
			assert_eq!(
				chunk.first_frame_index, 0,
				"restart resets the capture epoch"
			);
		}
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
		let wav = oa::audio::decode_file(audio_root.join("oaNarration.wav"))?;
		let flac = oa::audio::decode_file(audio_root.join("oaNarration.flac"))?;
		assert_eq!(wav.sample_rate(), 24_000);
		assert_eq!(flac.sample_rate(), wav.sample_rate());
		assert_eq!(flac.channels(), wav.channels());
		assert_eq!(flac.samples(), wav.samples());
		let wav_samples = wav.as_matrix().read_f32()?;
		let flac_samples = flac.as_matrix().read_f32()?;
		for index in (0..wav_samples.len()).step_by(31) {
			assert_eq!(flac_samples[index], wav_samples[index]);
		}

		let mp3 = oa::audio::decode_file(audio_root.join("oaNarration.mp3"))?;
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

test_vk!(
	stft_window_and_zero_win_size_match_independent_dft,
	engine,
	{
		use oa::audio::{StftConfig, StftWindow};
		let samples = (0..16)
			.map(|index| (std::f32::consts::TAU * index as f32 / 8.0).sin())
			.collect::<Vec<_>>();
		let input = Audio::new_from_planar_f32(&engine, &samples, 1, 8_000, AudioChannelLayout::Mono)?;
		for window in [
			StftWindow::Hann,
			StftWindow::Hamming,
			StftWindow::Blackman,
			StftWindow::Rectangular,
		] {
			let output = oa::audio::stft(
				&input,
				StftConfig {
					fft_size: 16,
					hop_size: 16,
					win_size: 0,
					window,
					center: false,
				},
			)?;
			assert_eq!(output.shape(), [1, 1, 9]);
			let actual = output.read_f32()?;
			for (bin, magnitude) in actual.into_iter().enumerate() {
				let mut real = 0.0_f64;
				let mut imaginary = 0.0_f64;
				for (index, sample) in samples.iter().enumerate() {
					let phase = std::f64::consts::TAU * index as f64 / 16.0;
					let weight = match window {
						StftWindow::Hann => 0.5 - 0.5 * phase.cos(),
						StftWindow::Hamming => 0.54 - 0.46 * phase.cos(),
						StftWindow::Blackman => 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos(),
						StftWindow::Rectangular => 1.0,
					};
					let angle = std::f64::consts::TAU * bin as f64 * index as f64 / 16.0;
					real += f64::from(*sample) * weight * angle.cos();
					imaginary -= f64::from(*sample) * weight * angle.sin();
				}
				let expected = real.hypot(imaginary);
				assert!(
					(f64::from(magnitude) - expected).abs() < 1e-4,
					"window {window:?}, bin {bin}: {magnitude} != {expected}"
				);
			}
		}
		Ok(())
	}
);

#[test]
fn generated_dsp_config_defaults_preserve_donor_contracts() {
	use oa::audio::{
		BiquadCoefficients, MelConfig, MfccConfig, NormalizeAudioConfig, NormalizeAudioMode,
		ResampleConfig, StftConfig, StftWindow,
	};
	let normalize = NormalizeAudioConfig::default();
	assert_eq!(normalize.target_db, -3.0);
	assert_eq!(normalize.mode, NormalizeAudioMode::Peak);
	assert_eq!(NormalizeAudioMode::default(), NormalizeAudioMode::Peak);
	let resample = ResampleConfig::default();
	assert_eq!(
		(resample.out_rate, resample.filter_half_width),
		(16_000, 64)
	);
	let biquad = BiquadCoefficients::default();
	assert_eq!(
		[biquad.b0, biquad.b1, biquad.b2, biquad.a1, biquad.a2],
		[1.0, 0.0, 0.0, 0.0, 0.0]
	);
	let stft = StftConfig::default();
	assert_eq!(
		(stft.fft_size, stft.hop_size, stft.win_size),
		(1_024, 256, 0)
	);
	assert_eq!(stft.window, StftWindow::Hann);
	assert_eq!(StftWindow::default(), StftWindow::Hann);
	assert!(stft.center);
	let mel = MelConfig::default();
	assert_eq!((mel.fft_size, mel.hop_size, mel.num_mels), (1_024, 256, 80));
	assert_eq!((mel.f_min, mel.f_max), (0.0, 0.0));
	assert!(mel.log_scale);
	assert!(!mel.normalize);
	let mfcc = MfccConfig::default();
	assert_eq!(mfcc.num_coeffs, 13);
	assert_eq!(mfcc.mel, mel);
}

test_vk!(stft_complex_matches_magnitude_spectrum, engine, {
	use oa::audio::{StftConfig, stft, stft_complex};
	let samples = (0..512)
		.map(|index| {
			let phase = index as f32;
			(0.4 * (std::f32::consts::TAU * phase / 16.0).sin())
				+ (0.2 * (std::f32::consts::TAU * phase / 7.0).cos())
		})
		.collect::<Vec<_>>();
	let input = Audio::new_from_planar_f32(&engine, &samples, 1, 48_000, AudioChannelLayout::Mono)?;
	let config = StftConfig {
		fft_size: 64,
		hop_size: 16,
		win_size: 0,
		center: true,
		..StftConfig::default()
	};
	let complex = stft_complex(&input, config)?;
	assert_eq!(complex.shape(), [1, 33, 33, 2]);
	let magnitude = stft(&input, config)?;
	assert_eq!(magnitude.shape(), [1, 33, 33]);
	let complex_values = complex.read_f32()?;
	let magnitude_values = magnitude.read_f32()?;
	for (bin, expected) in magnitude_values.iter().enumerate() {
		let real = complex_values[bin * 2];
		let imaginary = complex_values[bin * 2 + 1];
		let actual = (real * real + imaginary * imaginary).sqrt();
		assert!(
			(actual - expected).abs() < 1e-5,
			"bin {bin}: complex magnitude {actual} != stft {expected}"
		);
	}
	Ok(())
});

test_vk!(istft_roundtrip_reconstructs_audio, engine, {
	use oa::audio::{IstftConfig, StftConfig, StftWindow, istft, stft_complex};
	let sample_count = 1_024usize;
	let samples = (0..sample_count)
		.map(|index| {
			let phase = index as f32;
			(0.5 * (std::f32::consts::TAU * phase / 32.0).sin())
				+ (0.3 * (std::f32::consts::TAU * phase / 11.0).cos())
				+ 0.1 * (phase * 0.37).sin()
		})
		.collect::<Vec<_>>();
	let input = Audio::new_from_planar_f32(&engine, &samples, 1, 48_000, AudioChannelLayout::Mono)?;
	for window in [StftWindow::Hann, StftWindow::Rectangular] {
		let stft_config = StftConfig {
			fft_size: 256,
			hop_size: 64,
			win_size: 0,
			window,
			center: true,
		};
		let complex = stft_complex(&input, stft_config)?;
		let output = istft(
			&complex,
			IstftConfig {
				fft_size: 256,
				hop_size: 64,
				win_size: 0,
				window,
				center: true,
				sample_rate: 48_000,
			},
		)?;
		assert_eq!(output.channels(), 1);
		assert_eq!(output.samples(), sample_count);
		assert_eq!(output.sample_rate(), 48_000);
		let reconstructed = output.as_matrix().read_f32()?;
		for (index, (actual, expected)) in reconstructed.iter().zip(&samples).enumerate() {
			assert!(
				(actual - expected).abs() < 1e-4,
				"sample {index}: reconstructed {actual} != {expected}"
			);
		}
	}
	Ok(())
});

test_vk!(istft_rejects_invalid_inputs, engine, {
	use oa::audio::{IstftConfig, istft};
	let matrix = oa::Matrix::new_from_f32_on(&engine, [1, 4, 9, 2], &[0.0; 72])?;
	let config = IstftConfig {
		fft_size: 16,
		hop_size: 4,
		sample_rate: 48_000,
		..IstftConfig::default()
	};
	assert!(istft(&matrix, config).is_ok());
	assert!(
		istft(
			&matrix,
			IstftConfig {
				fft_size: 32,
				..config
			}
		)
		.is_err()
	);
	assert!(
		istft(
			&matrix,
			IstftConfig {
				fft_size: 12,
				..config
			}
		)
		.is_err()
	);
	assert!(
		istft(
			&matrix,
			IstftConfig {
				hop_size: 0,
				..config
			}
		)
		.is_err()
	);
	assert!(
		istft(
			&matrix,
			IstftConfig {
				sample_rate: 0,
				..config
			}
		)
		.is_err()
	);
	let rank_two = oa::Matrix::new_from_f32_on(&engine, [4, 9], &[0.0; 36])?;
	assert!(istft(&rank_two, config).is_err());
	Ok(())
});

test_vk!(noise_like_is_deterministic_per_seed, engine, {
	let samples: Vec<f32> = (0..128).map(|i| (i as f32 * 0.02).sin()).collect();
	let audio = Audio::new_from_planar_f32(&engine, &samples, 1, 8_000, AudioChannelLayout::Mono)?;
	let first = oa::audio::noise_like(&audio, oa::audio::NoiseDistribution::Gaussian, 0.1, 42)?;
	let repeat = oa::audio::noise_like(&audio, oa::audio::NoiseDistribution::Gaussian, 0.1, 42)?;
	let other = oa::audio::noise_like(&audio, oa::audio::NoiseDistribution::Gaussian, 0.1, 43)?;
	let uniform = oa::audio::noise_like(&audio, oa::audio::NoiseDistribution::Uniform, 0.5, 42)?;
	assert_eq!(
		first.as_matrix().read_f32()?,
		repeat.as_matrix().read_f32()?
	);
	assert_ne!(first.as_matrix().read_f32()?, other.as_matrix().read_f32()?);
	assert!(
		uniform
			.as_matrix()
			.read_f32()?
			.iter()
			.all(|v| v.abs() <= 0.5 + f32::EPSILON)
	);
	assert!(oa::audio::noise_like(&audio, oa::audio::NoiseDistribution::Gaussian, -1.0, 42).is_err());
	assert!(
		oa::audio::noise_like(&audio, oa::audio::NoiseDistribution::Gaussian, f32::NAN, 42).is_err()
	);
	Ok(())
});

test_vk!(mix_snr_reaches_target_signal_to_noise_ratio, engine, {
	let samples: Vec<f32> = (0..4096)
		.map(|i| ((i as f32 * 0.05).sin() + 0.5 * (i as f32 * 0.13).cos()) * 0.5)
		.collect();
	let signal = Audio::new_from_planar_f32(&engine, &samples, 1, 8_000, AudioChannelLayout::Mono)?;
	let noise = oa::audio::noise_like(&signal, oa::audio::NoiseDistribution::Gaussian, 0.2, 7)?;
	let snr_db = 10.0_f32;
	let mixed = oa::audio::mix_snr(&signal, &noise, snr_db)?;
	let signal_read = signal.as_matrix().read_f32()?;
	let mixed_read = mixed.as_matrix().read_f32()?;
	// Host oracle: measured SNR between signal and the injected noise component.
	let mut signal_power = 0.0f64;
	let mut residual_power = 0.0f64;
	for (s, m) in signal_read.iter().zip(mixed_read.iter()) {
		signal_power += f64::from(*s) * f64::from(*s);
		let residual = f64::from(m - s);
		residual_power += residual * residual;
	}
	let measured_snr_db = 10.0 * (signal_power / residual_power).log10() as f32;
	assert!(
		(measured_snr_db - snr_db).abs() < 0.25,
		"measured SNR {measured_snr_db} dB differs from target {snr_db} dB"
	);
	let mismatched = oa::audio::noise_like(
		&Audio::new_from_planar_f32(&engine, &samples[..64], 1, 8_000, AudioChannelLayout::Mono)?,
		oa::audio::NoiseDistribution::Gaussian,
		0.2,
		7,
	)?;
	assert!(oa::audio::mix_snr(&signal, &mismatched, snr_db).is_err());
	assert!(oa::audio::mix_snr(&signal, &noise, f32::INFINITY).is_err());
	Ok(())
});

fn deterministic_values(count: usize, seed: u64) -> Vec<f32> {
	let mut state = seed;
	(0..count)
		.map(|_| {
			state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
			((state >> 33) as f32 / u32::MAX as f32) * 0.2 - 0.1
		})
		.collect()
}

test_vk!(stft_complex_backward_matches_finite_difference, engine, {
	// Graph: y = x @ W^T -> audio -> stft_complex -> mse -> scalar loss.
	// The weight is a watched nn parameter, so its gradient surfaces through
	// the transform adjoint and can be checked against finite differences.
	let (n, out) = (32_usize, 128_usize);
	let config = oa::audio::StftConfig {
		fft_size: 64,
		hop_size: 16,
		win_size: 0,
		window: oa::audio::StftWindow::Hann,
		center: true,
	};
	let x = oa::Matrix::new_from_f32_on(&engine, [1, n], &deterministic_values(n, 7))?;
	let weights = deterministic_values(out * n, 11);
	let loss_of = |weights: &[f32]| -> oa::Result<f32> {
		let layer = oa::ml::nn::Linear::new_from_weight(oa::Matrix::new_from_f32_on(
			&engine,
			[out, n],
			weights,
		)?)?;
		let audio = Audio::new(layer.forward(&x)?, 8_000, AudioChannelLayout::Mono)?;
		let spectrum = oa::audio::stft_complex(&audio, config)?;
		let zeros = oa::Matrix::new_from_f32_on(
			&engine,
			spectrum.shape().to_vec(),
			&vec![0.0; spectrum.shape().iter().product::<usize>()],
		)?;
		oa::ml::loss::mse(&spectrum, &zeros)?
			.read_f32()
			.map(|values| values[0])
	};
	let tape = oa::ml::GradientTape::new();
	let layer =
		oa::ml::nn::Linear::new_from_weight(oa::Matrix::new_from_f32_on(&engine, [out, n], &weights)?)?;
	let audio = Audio::new(layer.forward(&x)?, 8_000, AudioChannelLayout::Mono)?;
	let spectrum = oa::audio::stft_complex(&audio, config)?;
	let zeros = oa::Matrix::new_from_f32_on(
		&engine,
		spectrum.shape().to_vec(),
		&vec![0.0; spectrum.shape().iter().product::<usize>()],
	)?;
	let loss = oa::ml::loss::mse(&spectrum, &zeros)?;
	tape.backward(&loss)?;
	let gradient = layer
		.weight()
		.gradient()
		.expect("stft_complex backward must reach the linear weight")
		.read_f32()?;
	let epsilon = 1.0e-3_f32;
	for index in [0_usize, n + 3, out * n - 7] {
		let mut plus = weights.clone();
		let mut minus = weights.clone();
		plus[index] += epsilon;
		minus[index] -= epsilon;
		let finite = (loss_of(&plus)? - loss_of(&minus)?) / (2.0 * epsilon);
		assert!(
			(finite - gradient[index]).abs() < 1.0e-2 + 0.05 * finite.abs(),
			"stft_complex backward weight {index}: finite {finite} vs tape {}",
			gradient[index]
		);
	}
	Ok(())
});

test_vk!(istft_backward_matches_finite_difference, engine, {
	// Graph: spec_flat = x @ W^T -> reshape [1, frames, bins, 2] -> istft ->
	// audio -> mse -> scalar loss. Same finite-difference check through the
	// synthesis adjoint.
	let (n, frames, bins) = (8_usize, 9_usize, 33_usize);
	let spec_count = frames * bins * 2;
	let config = oa::audio::IstftConfig {
		fft_size: 64,
		hop_size: 16,
		win_size: 0,
		window: oa::audio::StftWindow::Hann,
		center: true,
		sample_rate: 8_000,
	};
	let x = oa::Matrix::new_from_f32_on(&engine, [1, n], &deterministic_values(n, 13))?;
	let weights = deterministic_values(spec_count * n, 17);
	let forward =
		|weights: &[f32], engine: &oa::Engine| -> oa::Result<(oa::ml::nn::Linear, oa::Matrix)> {
			let layer = oa::ml::nn::Linear::new_from_weight(oa::Matrix::new_from_f32_on(
				engine,
				[spec_count, n],
				weights,
			)?)?;
			let spec = layer.forward(&x)?.reshape([1, frames, bins, 2])?;
			let audio = oa::audio::istft(&spec, config)?;
			let target = oa::Matrix::new_from_f32_on(
				engine,
				audio.as_matrix().shape().to_vec(),
				&deterministic_values(audio.as_matrix().shape().iter().product::<usize>(), 19),
			)?;
			Ok((layer, oa::ml::loss::mse(audio.as_matrix(), &target)?))
		};
	let loss_of = |weights: &[f32]| -> oa::Result<f32> {
		forward(weights, &engine).and_then(|(_, loss)| loss.read_f32().map(|values| values[0]))
	};
	let tape = oa::ml::GradientTape::new();
	let (layer, loss) = forward(&weights, &engine)?;
	tape.backward(&loss)?;
	let gradient = layer
		.weight()
		.gradient()
		.expect("istft backward must reach the linear weight")
		.read_f32()?;
	let epsilon = 1.0e-3_f32;
	for index in [0_usize, n + 5, spec_count * n - 11] {
		let mut plus = weights.clone();
		let mut minus = weights.clone();
		plus[index] += epsilon;
		minus[index] -= epsilon;
		let finite = (loss_of(&plus)? - loss_of(&minus)?) / (2.0 * epsilon);
		assert!(
			(finite - gradient[index]).abs() < 1.0e-2 + 0.05 * finite.abs(),
			"istft backward weight {index}: finite {finite} vs tape {}",
			gradient[index]
		);
	}
	Ok(())
});

test_vk!(batch_pad_crop_stack_and_mask, engine, {
	let clip_a = Audio::new(
		oa::Matrix::new_from_f32_on(&engine, [1, 3], &[1.0, 2.0, 3.0])?,
		48_000,
		AudioChannelLayout::Mono,
	)?;
	let clip_b = Audio::new(
		oa::Matrix::new_from_f32_on(
			&engine,
			[1, 16],
			&[
				4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0,
			],
		)?,
		48_000,
		AudioChannelLayout::Mono,
	)?;

	let padded = oa::audio::pad_to(&clip_a, clip_b.samples())?;
	assert_eq!(padded.samples(), 16);
	assert_eq!(padded.sample_rate(), 48_000);
	let tail = &padded.as_matrix().read_f32()?[3..];
	assert!(tail.iter().all(|value| *value == 0.0));

	let cropped = oa::audio::crop_to(&clip_b, 4)?;
	assert_eq!(cropped.as_matrix().read_f32()?, [4.0, 5.0, 6.0, 7.0]);

	let batch = oa::audio::stack(&[padded, clip_b])?;
	assert_eq!(batch.channels(), 2);
	assert_eq!(batch.samples(), 16);
	assert_eq!(batch.as_matrix().read_f32()?[..3], [1.0, 2.0, 3.0]);

	let mask = oa::audio::length_mask(&batch, &[3, 16])?;
	assert_eq!(mask.shape(), [2, 16]);
	let values = mask.read_f32()?;
	assert_eq!(values[..3], [1.0, 1.0, 1.0]);
	assert!(values[3..16].iter().all(|value| *value == 0.0));
	assert!(values[16..].iter().all(|value| *value == 1.0));

	// A stacked mono batch feeds per-row ops without special batching code.
	let spectrum = oa::audio::stft_complex(
		&batch,
		oa::audio::StftConfig {
			fft_size: 16,
			hop_size: 4,
			win_size: 16,
			window: oa::audio::StftWindow::Rectangular,
			center: false,
		},
	)?;
	assert_eq!(spectrum.shape(), [2, 1, 9, 2]);
	Ok(())
});

test_vk!(batch_ops_reject_mismatched_inputs, engine, {
	let clip = Audio::new(
		oa::Matrix::new_from_f32_on(&engine, [1, 4], &[1.0; 4])?,
		48_000,
		AudioChannelLayout::Mono,
	)?;
	let other_rate = Audio::new(
		oa::Matrix::new_from_f32_on(&engine, [1, 4], &[2.0; 4])?,
		44_100,
		AudioChannelLayout::Mono,
	)?;
	let other_length = Audio::new(
		oa::Matrix::new_from_f32_on(&engine, [1, 6], &[3.0; 6])?,
		48_000,
		AudioChannelLayout::Mono,
	)?;
	let stereo = Audio::new(
		oa::Matrix::new_from_f32_on(&engine, [2, 4], &[0.5; 8])?,
		48_000,
		AudioChannelLayout::Stereo,
	)?;

	assert!(oa::audio::pad_to(&clip, 2).is_err());
	assert!(oa::audio::crop_to(&clip, 0).is_err());
	assert!(oa::audio::crop_to(&clip, 5).is_err());
	assert!(oa::audio::stack(&[]).is_err());
	assert!(oa::audio::stack(&[clip.clone(), other_rate]).is_err());
	assert!(oa::audio::stack(&[clip.clone(), other_length]).is_err());
	assert!(oa::audio::stack(&[clip.clone(), stereo]).is_err());
	assert!(oa::audio::length_mask(&clip, &[4, 4]).is_err());
	assert!(oa::audio::length_mask(&clip, &[5]).is_err());
	Ok(())
});
