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
		return oa.Audio.from_planar_f32(
			self.engine,
			[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
			channels=2,
			sample_rate=8000,
			layout="stereo",
		)

	def _mono(self) -> oa.Audio:
		return oa.Audio.from_planar_f32(
			self.engine,
			[0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25],
			channels=1,
			sample_rate=8000,
			layout="mono",
		)

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
		cls.mono = oa.Audio.from_planar_f32(
			cls.engine,
			[0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1],
			channels=1,
			sample_rate=8000,
			layout="mono",
		)

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
		other = oa.Audio.from_planar_f32(
			self.engine,
			[0.1] * 8,
			channels=1,
			sample_rate=8000,
			layout="mono",
		)
		result = oa.audio.mix(self.mono, other, gain_a=1.0, gain_b=1.0)
		self.assertIsInstance(result, oa.Audio)
		self.assertEqual(result.samples, 8)

	def test_normalize_peak_does_not_increase_beyond_target(self) -> None:
		loud = oa.Audio.from_planar_f32(
			self.engine,
			[0.01] * 8,
			channels=1,
			sample_rate=8000,
			layout="mono",
		)
		result = oa.audio.normalize(loud, target_db=-3.0, mode="peak")
		self.assertIsInstance(result, oa.Audio)


class AudioFeatureTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()
		# 512 samples of a 440 Hz tone at 8000 Hz sample rate
		import math as _math
		samples = [_math.sin(2 * _math.pi * 440 * i / 8000) for i in range(512)]
		cls.tone = oa.Audio.from_planar_f32(
			cls.engine, samples, channels=1, sample_rate=8000, layout="mono"
		)

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
		cls.mono = oa.Audio.from_planar_f32(
			cls.engine,
			[0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0],
			channels=1,
			sample_rate=8000,
			layout="mono",
		)

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
		audio = oa.Audio.from_planar_f32(
			self.engine,
			[0.0, 0.5, 1.0, 0.5, 0.0],
			channels=1,
			sample_rate=8000,
			layout="mono",
		)
		encoded = oa.audio.encode_wav_f32(audio)
		self.assertIsInstance(encoded, bytes)
		self.assertTrue(encoded[:4] == b"RIFF")


if __name__ == "__main__":
	unittest.main()
