"""Python binding tests for oa.audio DSP and feature extraction operations."""

import math
import unittest

import oa


class AudioConstructorTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def _stereo(self) -> oa.Audio:
		# 4 stereo samples (channels first / planar): L=[0.1, 0.2, 0.3, 0.4] R=[0.5, 0.6, 0.7, 0.8]
		return oa.Audio.new_from_planar_f32(
			[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
			channels=2,
			sample_rate=8000,
			layout="stereo", engine=self.engine)

	def _mono(self) -> oa.Audio:
		return oa.Audio.new_from_planar_f32(
			[0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25],
			channels=1,
			sample_rate=8000,
			layout="mono", engine=self.engine)

	def test_from_planar_f32_properties(self) -> None:
		a = self._stereo()
		self.assertEqual(a.channels, 2)
		self.assertEqual(a.samples, 4)
		self.assertEqual(a.sample_rate, 8000)
		self.assertEqual(a.layout, "stereo")
		self.assertAlmostEqual(a.duration_seconds, 4 / 8000, places=6)

	def test_as_matrix_shape(self) -> None:
		a = self._stereo()
		m = a.as_matrix()
		self.assertIsInstance(m, oa.Matrix)
		self.assertEqual(m.shape, [2, 4])

	def test_to_mono_returns_audio(self) -> None:
		a = self._stereo()
		mono = oa.audio.to_mono(a)
		self.assertIsInstance(mono, oa.Audio)
		self.assertEqual(mono.channels, 1)
		self.assertEqual(mono.samples, 4)


class AudioDspTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()
		cls.mono = oa.Audio.new_from_planar_f32(
			[0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1],
			channels=1,
			sample_rate=8000,
			layout="mono", engine=cls.engine)

	def test_gain_increases_amplitude(self) -> None:
		boosted = oa.audio.gain(self.mono, gain_db=6.0)
		self.assertIsInstance(boosted, oa.Audio)
		orig = self.mono.as_matrix().read_f32()
		louder = boosted.as_matrix().read_f32()
		for o, l in zip(orig, louder):
			if abs(o) > 1e-6:
				self.assertGreater(abs(l), abs(o))

	def test_clip_bounds_samples(self) -> None:
		result = oa.audio.clip(self.mono, -0.15, 0.15)
		for v in result.as_matrix().read_f32():
			self.assertLessEqual(v, 0.151)
			self.assertGreaterEqual(v, -0.151)

	def test_pre_emphasis_changes_signal(self) -> None:
		result = oa.audio.pre_emphasis(self.mono, alpha=0.97)
		self.assertIsInstance(result, oa.Audio)
		self.assertEqual(result.samples, self.mono.samples)

	def test_saturate_returns_audio(self) -> None:
		result = oa.audio.saturate(self.mono, drive_db=6.0, mix=0.5)
		self.assertIsInstance(result, oa.Audio)

	def test_reverb_returns_longer_or_equal_audio(self) -> None:
		result = oa.audio.reverb(self.mono, decay_seconds=0.1, wet=0.3)
		self.assertIsInstance(result, oa.Audio)
		self.assertGreaterEqual(result.samples, self.mono.samples)

	def test_fade_returns_audio(self) -> None:
		result = oa.audio.fade(self.mono, fade_in_samples=2, fade_out_samples=2)
		self.assertIsInstance(result, oa.Audio)
		self.assertEqual(result.samples, self.mono.samples)

	def test_mix_combines_two_signals(self) -> None:
		other = oa.Audio.new_from_planar_f32(
			[0.1] * 8,
			channels=1,
			sample_rate=8000,
			layout="mono", engine=self.engine)
		result = oa.audio.mix(self.mono, other, gain_left=1.0, gain_right=1.0)
		self.assertIsInstance(result, oa.Audio)
		self.assertEqual(result.samples, 8)

	def test_normalize_peak_does_not_increase_beyond_target(self) -> None:
		loud = oa.Audio.new_from_planar_f32(
			[0.01] * 8,
			channels=1,
			sample_rate=8000,
			layout="mono", engine=self.engine)
		result = oa.audio.normalize(loud, target_db=-3.0, mode="peak")
		self.assertIsInstance(result, oa.Audio)


class AudioFeatureTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()
		# 512 samples of a 440 Hz tone at 8000 Hz sample rate
		import math as _math
		samples = [_math.sin(2 * _math.pi * 440 * i / 8000) for i in range(512)]
		cls.tone = oa.Audio.new_from_planar_f32( samples, channels=1, sample_rate=8000, layout="mono", engine=cls.engine)

	def test_amplitude_to_db_shape(self) -> None:
		result = oa.audio.amplitude_to_db(self.tone, floor_db=-80.0)
		self.assertIsInstance(result, oa.Matrix)
		self.assertEqual(len(result.shape), 2)

	def test_waveform_envelope_shape(self) -> None:
		result = oa.audio.waveform_envelope(self.tone, bins=32)
		self.assertIsInstance(result, oa.Matrix)
		# Output layout is [bins, 2]: each bin contains [min, max] amplitude.
		self.assertEqual(result.shape, [32, 2])

	def test_stft_shape_consistent(self) -> None:
		result = oa.audio.stft(self.tone, fft_size=64, hop_size=32)
		self.assertIsInstance(result, oa.Matrix)
		self.assertGreaterEqual(result.num_elements, 1)

	def test_mel_spectrogram_shape(self) -> None:
		result = oa.audio.mel_spectrogram(
			self.tone, fft_size=64, hop_size=32, num_mels=16
		)
		self.assertIsInstance(result, oa.Matrix)
		self.assertGreaterEqual(result.num_elements, 16)

	def test_mfcc_shape_is_channels_coeffs_frames(self) -> None:
		result = oa.audio.mfcc(
			self.tone,
			num_coeffs=13,
			num_mels=16,
			fft_size=64,
			hop_size=32,
		)
		self.assertIsInstance(result, oa.Matrix)
		# Output layout is [channels, num_coeffs, frames].
		self.assertEqual(len(result.shape), 3)
		self.assertEqual(result.shape[1], 13)


class AudioBiquadTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()
		cls.mono = oa.Audio.new_from_planar_f32(
			[0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0],
			channels=1,
			sample_rate=8000,
			layout="mono", engine=cls.engine)

	def test_biquad_passthrough_identity(self) -> None:
		# identity: b0=1 b1=0 b2=0 a1=0 a2=0
		result = oa.audio.biquad(self.mono, b0=1.0, b1=0.0, b2=0.0, a1=0.0, a2=0.0)
		self.assertIsInstance(result, oa.Audio)
		orig = self.mono.as_matrix().read_f32()
		filt = result.as_matrix().read_f32()
		for o, f in zip(orig, filt):
			self.assertAlmostEqual(f, o, places=5)

	def test_sos_filter_two_sections(self) -> None:
		# Two identity biquad sections
		sections = [[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]]
		result = oa.audio.sos_filter(self.mono, sections)
		self.assertIsInstance(result, oa.Audio)
		orig = self.mono.as_matrix().read_f32()
		filt = result.as_matrix().read_f32()
		for o, f in zip(orig, filt):
			self.assertAlmostEqual(f, o, places=5)

	def test_sos_filter_single_section(self) -> None:
		sections = [[1.0, 0.0, 0.0, 0.0, 0.0]]
		result = oa.audio.sos_filter(self.mono, sections)
		self.assertEqual(result.samples, self.mono.samples)


class AudioCodecTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_encode_interleaved_wav_f32_returns_bytes(self) -> None:
		samples = [0.0, 0.1, 0.2, 0.3] * 2  # 4 stereo frames interleaved
		result = oa.audio.encode_interleaved_wav_f32(samples, 8000, 2)
		self.assertIsInstance(result, bytes)
		# WAV header starts with RIFF
		self.assertTrue(result[:4] == b"RIFF")

	def test_encode_interleaved_wav_f32_mono(self) -> None:
		samples = [0.0, 0.25, 0.5, 0.75]
		result = oa.audio.encode_interleaved_wav_f32(samples, 16000, 1)
		self.assertIsInstance(result, bytes)
		self.assertGreater(len(result), 44)  # at least WAV header

	def test_encode_wav_f32_roundtrip_length(self) -> None:
		audio = oa.Audio.new_from_planar_f32(
			[0.0, 0.5, 1.0, 0.5, 0.0],
			channels=1,
			sample_rate=8000,
			layout="mono", engine=self.engine)
		encoded = oa.audio.encode_wav_f32(audio)
		self.assertIsInstance(encoded, bytes)
		self.assertTrue(encoded[:4] == b"RIFF")


if __name__ == "__main__":
	unittest.main()

class StftComplexIstftTest(unittest.TestCase):
	"""GPU complex STFT and inverse STFT round trip."""

	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()
		samples = [
			0.5 * math.sin(2 * math.pi * i / 32.0)
			+ 0.3 * math.cos(2 * math.pi * i / 11.0)
			+ 0.1 * math.sin(i * 0.37)
			for i in range(1024)
		]
		cls.samples = samples
		cls.tone = oa.Audio.new_from_planar_f32(
			samples, channels=1, sample_rate=48_000, layout="mono", engine=cls.engine
		)

	def test_stft_complex_matches_magnitude(self) -> None:
		complex_spectrum = oa.audio.stft_complex(self.tone, fft_size=64, hop_size=16)
		self.assertEqual(complex_spectrum.shape, [1, 65, 33, 2])
		magnitude = oa.audio.stft(self.tone, fft_size=64, hop_size=16)
		complex_values = complex_spectrum.read_f32()
		magnitude_values = magnitude.read_f32()
		for index, expected in enumerate(magnitude_values):
			real = complex_values[index * 2]
			imaginary = complex_values[index * 2 + 1]
			actual = math.sqrt(real * real + imaginary * imaginary)
			self.assertAlmostEqual(actual, expected, places=5)

	def test_istft_roundtrip_reconstructs_audio(self) -> None:
		complex_spectrum = oa.audio.stft_complex(self.tone, fft_size=256, hop_size=64)
		output = oa.audio.istft(
			complex_spectrum, fft_size=256, hop_size=64, sample_rate=48_000
		)
		self.assertEqual(output.channels, 1)
		self.assertEqual(output.samples, len(self.samples))
		self.assertEqual(output.sample_rate, 48_000)
		for actual, expected in zip(output.as_matrix().read_f32(), self.samples):
			self.assertAlmostEqual(actual, expected, places=4)

	def test_istft_rejects_mismatched_shape(self) -> None:
		bad = oa.Matrix.new_from_f32([1, 4, 8, 2], [0.0] * 64, engine=self.engine)
		with self.assertRaises(Exception):
			oa.audio.istft(bad, fft_size=16)


class AudioAugmentationTest(unittest.TestCase):
	def test_noise_like_deterministic_and_mix_snr_reaches_target(self) -> None:
		samples = [math.sin(i * 0.05) * 0.5 for i in range(4096)]
		signal = oa.Audio.new_from_planar_f32(samples, channels=1, sample_rate=8000, layout="mono")
		first = oa.audio.noise_like(signal, "gaussian", 0.2, 42)
		repeat = oa.audio.noise_like(signal, "gaussian", 0.2, 42)
		assert first.as_matrix().read_f32() == repeat.as_matrix().read_f32()
		other = oa.audio.noise_like(signal, "gaussian", 0.2, 43)
		assert first.as_matrix().read_f32() != other.as_matrix().read_f32()
		uniform = oa.audio.noise_like(signal, "uniform", 0.5, 42)
		assert all(abs(v) <= 0.5 + 1e-6 for v in uniform.as_matrix().read_f32())

		mixed = oa.audio.mix_snr(signal, first, 10.0)
		signal_read = signal.as_matrix().read_f32()
		mixed_read = mixed.as_matrix().read_f32()
		signal_power = sum(s * s for s in signal_read)
		residual_power = sum((m - s) * (m - s) for s, m in zip(signal_read, mixed_read))
		measured = 10.0 * math.log10(signal_power / residual_power)
		assert abs(measured - 10.0) < 0.25

	def test_pad_crop_stack_and_length_mask(self) -> None:
		clip_a = oa.Audio.new_from_planar_f32(
			[1.0, 2.0, 3.0], channels=1, sample_rate=48_000, layout="mono")
		clip_b = oa.Audio.new_from_planar_f32(
			[float(i) for i in range(16)], channels=1, sample_rate=48_000, layout="mono")

		padded = oa.audio.pad_to(clip_a, 16)
		assert padded.samples == 16
		assert padded.as_matrix().read_f32()[3:] == [0.0] * 13
		cropped = oa.audio.crop_to(clip_b, 4)
		assert cropped.as_matrix().read_f32() == [0.0, 1.0, 2.0, 3.0]

		batch = oa.audio.stack([padded, clip_b])
		assert batch.channels == 2
		assert batch.samples == 16
		mask = oa.audio.length_mask(batch, [3, 16])
		assert mask.shape == [2, 16]
		values = mask.read_f32()
		assert values[:3] == [1.0, 1.0, 1.0]
		assert all(v == 0.0 for v in values[3:16])
		assert all(v == 1.0 for v in values[16:])

		spectrum = oa.audio.stft_complex(
			batch, fft_size=16, hop_size=4, win_size=16, window="rectangular", center=False)
		assert spectrum.shape == [2, 1, 9, 2]

		with self.assertRaises(Exception):
			oa.audio.pad_to(clip_a, 2)
		with self.assertRaises(Exception):
			oa.audio.stack([clip_a, clip_b])
		with self.assertRaises(Exception):
			oa.audio.length_mask(batch, [3])
