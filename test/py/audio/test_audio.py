#!/usr/bin/env python3
"""Tests for OA's public root-parity Audio API.

Self-contained: builds a synthetic waveform on the GPU, exercises the FnAudio
DSP surface, and round-trips it through the WAV-F32 codec boundary — no external
audio asset required. Run from a checkout with:

	python -m pytest test/py/audio/test_audio.py

If the native extension is not on PYTHONPATH, set OA_PYTHON_BUILD_DIR to the
directory containing the private `_oa` extension.
"""

from __future__ import annotations

import math
import sys

import pytest

import oa_python_test  # noqa: F401 - bootstraps source builds

import oa
audio = oa.FnAudio
core = oa.FnMatrix
runtime = oa


@pytest.fixture(scope="session")
def engine():
	if not runtime.initComputeEngine():
		pytest.fail("GPU profile requires an initialized OA Vulkan engine")
	yield
	shutdown = getattr(runtime, "shutdownComputeEngine", None)
	if shutdown is not None:
		shutdown()


SAMPLE_RATE = 16000
NUM_SAMPLES = 4096


def _makeSine(freq: float = 440.0) -> "oa.Audio":
	# Mono [1, NUM_SAMPLES] tone plus a small deterministic broadband floor so
	# log-domain features (mel/MFCC) never hit a near-silent band.
	seed = 12345
	data = []
	for n in range(NUM_SAMPLES):
		seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
		noise = (seed / 0x7FFFFFFF - 0.5) * 0.02
		data.append(0.5 * math.sin(2.0 * math.pi * freq * n / SAMPLE_RATE) + noise)
	matrix = oa.FnMatrix.fromFloats(data, 1, NUM_SAMPLES)
	return oa.Audio(matrix, SAMPLE_RATE, oa.AudioChannelLayout.Mono)


def _biquadCoefficients(frequency: float, *, highpass: bool = False):
	q = 1.0 / math.sqrt(2.0)
	omega = 2.0 * math.pi * frequency / SAMPLE_RATE
	cosine = math.cos(omega)
	alpha = math.sin(omega) / (2.0 * q)
	a0 = 1.0 + alpha
	coefficients = oa.BiquadCoefficients()
	if highpass:
		coefficients.b0 = ((1.0 + cosine) * 0.5) / a0
		coefficients.b1 = -(1.0 + cosine) / a0
	else:
		coefficients.b0 = ((1.0 - cosine) * 0.5) / a0
		coefficients.b1 = (1.0 - cosine) / a0
	coefficients.b2 = coefficients.b0
	coefficients.a1 = (-2.0 * cosine) / a0
	coefficients.a2 = (1.0 - alpha) / a0
	return coefficients


def _cpuBiquad(samples: list[float], coefficients) -> list[float]:
	output = []
	state0 = 0.0
	state1 = 0.0
	for sample in samples:
		value = coefficients.b0 * sample + state0
		next0 = coefficients.b1 * sample - coefficients.a1 * value + state1
		next1 = coefficients.b2 * sample - coefficients.a2 * value
		output.append(value)
		state0 = next0
		state1 = next1
	return output


# ── Surface / construction (no GPU) ──────────────────────────────────────────


def test_audio_import_surface():
	for name in (
		"Audio",
		"AudioChannelLayout", "StftConfig", "MelConfig", "MfccConfig",
		"ResampleConfig", "NormalizeAudioConfig", "BiquadCoefficients",
		"AudioCodec", "AudioCapture", "AudioCaptureConfig",
		"AudioCaptureChunk", "AudioEncoder", "AudioEncodeProfile",
		"EncodedAudioPacket", "AudioPlayer", "AudioPlayerConfig",
	):
		assert hasattr(oa, name), name

	for name in (
		"decodeFile", "decodeMemory", "encodeWavF32", "saveWavF32",
		"stft", "melSpectrogram", "mfcc", "normalize", "resample", "gain",
		"clip", "saturate", "biquad", "sosFilter", "amplitudeToDb",
		"preEmphasis", "toMono", "fade", "mix", "reverb",
	):
		assert hasattr(oa.FnAudio, name), name

	assert oa.Audio.__module__ == "oa"
	assert audio.decodeFile is oa.FnAudio.decodeFile
	assert audio.normalize is oa.FnAudio.normalize


def test_config_structs_roundtrip_fields():
	stft = oa.StftConfig()
	stft.fftSize = 512
	stft.hopSize = 160
	stft.winSize = 512
	assert stft.fftSize == 512 and stft.hopSize == 160

	mel = oa.MelConfig()
	mel.numMels = 40
	mel.fftSize = 512
	mel.hopSize = 160
	assert mel.numMels == 40

	mfcc = oa.MfccConfig()
	mfcc.numCoeffs = 13
	mfcc.mel = mel
	assert mfcc.numCoeffs == 13 and mfcc.mel.numMels == 40

	biquad = oa.BiquadCoefficients()
	biquad.b0 = 0.25
	biquad.b1 = 0.5
	biquad.b2 = 0.25
	assert biquad.b0 == pytest.approx(0.25)
	assert biquad.b1 == pytest.approx(0.5)
	assert biquad.b2 == pytest.approx(0.25)


def test_channel_layout_helpers():
	assert oa.channelsForLayout(oa.AudioChannelLayout.Stereo) == 2
	assert oa.channelsForLayout(oa.AudioChannelLayout.Stereo21) == 3
	assert oa.layoutForChannels(1) == oa.AudioChannelLayout.Mono
	assert oa.layoutForChannels(2) == oa.AudioChannelLayout.Stereo
	assert oa.layoutForChannels(3) == oa.AudioChannelLayout.Unknown


def test_audio_encoder_packet_contract():
	profile = oa.AudioEncodeProfile()
	profile.sampleRate = 48_000
	profile.channelCount = 1
	profile.framesPerPacket = 4
	encoder = oa.AudioEncoder.create(profile)
	packets = encoder.encode([-1.0, 0.0, 0.5, 1.0])
	assert len(packets) == 1
	assert packets[0].durationFrames == 4
	assert len(packets[0].bitstream) == 8
	encoder.close()


# ── GPU DSP ──────────────────────────────────────────────────────────────────


def test_mel_spectrogram_shape(engine):
	x = _makeSine()
	cfg = oa.MelConfig()
	cfg.fftSize = 512
	cfg.hopSize = 160
	cfg.numMels = 40
	mel = oa.FnAudio.melSpectrogram(x, cfg)
	shape = mel.shape()
	# [Channels, NumMels, Frames]
	assert shape[0] == 1 and shape[1] == 40 and shape[2] > 0


def test_stft_shape(engine):
	x = _makeSine()
	cfg = oa.StftConfig()
	cfg.fftSize = 512
	cfg.hopSize = 160
	cfg.winSize = 512
	spec = oa.FnAudio.stft(x, cfg)
	shape = spec.shape()
	# [Channels, Frames, FftSize/2 + 1]
	assert shape[0] == 1 and shape[2] == 512 // 2 + 1


def test_signal_ops_shapes(engine):
	x = _makeSine()
	gained = oa.FnAudio.gain(x, -6.0)
	clipped = oa.FnAudio.clip(x, -0.5, 0.5)
	normalized = oa.FnAudio.normalize(x, -3.0, 0)
	mono = oa.FnAudio.toMono(x)
	faded = oa.FnAudio.fade(x, 128, 128)
	assert gained.matrix.shape() == [1, NUM_SAMPLES]
	assert clipped.matrix.shape() == [1, NUM_SAMPLES]
	assert normalized.matrix.shape() == [1, NUM_SAMPLES]
	assert mono.matrix.shape() == [1, NUM_SAMPLES]
	assert faded.matrix.shape() == [1, NUM_SAMPLES]
	assert gained.sampleRate == SAMPLE_RATE
	assert gained.layout == oa.AudioChannelLayout.Mono


def test_resample_length(engine):
	x = _makeSine()
	out = oa.FnAudio.resample(x, SAMPLE_RATE * 2, 64)
	# Upsample 2x → roughly double the sample count.
	assert out.matrix.shape()[1] == NUM_SAMPLES * 2
	assert out.sampleRate == SAMPLE_RATE * 2


def test_clip_bounds_values(engine):
	x = _makeSine()
	clipped = oa.FnAudio.clip(x, -0.1, 0.1)
	host = oa.FnMatrix.copyToHost(clipped.matrix)
	assert all(-0.1 - 1e-5 <= v <= 0.1 + 1e-5 for v in host)


def test_saturate_matches_waveshaper(engine):
	x = _makeSine()
	driveDb = 6.0
	mix = 0.7
	saturated = oa.FnAudio.saturate(x, driveDb, mix)
	dry = oa.FnMatrix.copyToHost(x.matrix)
	wet = oa.FnMatrix.copyToHost(saturated.matrix)
	drive = 10.0 ** (driveDb / 20.0)
	for source, actual in zip(dry, wet, strict=True):
		expected = source + (math.tanh(source * drive) - source) * mix
		assert actual == pytest.approx(expected, abs=2.0e-6)


def test_reverb_renders_finite_tail(engine):
	x = _makeSine()
	reverberated = oa.FnAudio.reverb(x, 0.125, 0.4)
	assert reverberated.sampleRate == SAMPLE_RATE
	assert reverberated.layout == oa.AudioChannelLayout.Mono
	assert reverberated.sampleCount == NUM_SAMPLES + SAMPLE_RATE // 8
	values = oa.FnMatrix.copyToHost(reverberated.matrix)
	assert all(math.isfinite(value) for value in values)
	assert any(abs(value) > 1.0e-5 for value in values[NUM_SAMPLES:])


def test_biquad_matches_zero_state_lowpass(engine):
	x = _makeSine()
	coefficients = _biquadCoefficients(2000.0)

	filtered = oa.FnAudio.biquad(x, coefficients)
	source = oa.FnMatrix.copyToHost(x.matrix)
	actual = oa.FnMatrix.copyToHost(filtered.matrix)
	expected = _cpuBiquad(source, coefficients)
	for index, value in enumerate(expected):
		assert actual[index] == pytest.approx(value, abs=3.0e-5)


def test_sos_filter_matches_cpu_cascade(engine):
	x = _makeSine(3000.0)
	sections = [
		_biquadCoefficients(100.0, highpass=True),
		_biquadCoefficients(6000.0),
		_biquadCoefficients(4000.0),
	]
	source = oa.FnMatrix.copyToHost(x.matrix)
	expected = source
	for section in sections:
		expected = _cpuBiquad(expected, section)

	filtered = oa.FnAudio.sosFilter(x, sections)
	actual = oa.FnMatrix.copyToHost(filtered.matrix)
	assert filtered.sampleRate == x.sampleRate
	assert filtered.layout == x.layout
	for index, value in enumerate(expected):
		assert actual[index] == pytest.approx(value, abs=8.0e-5)


# ── Codec round-trip ──────────────────────────────────────────────────────────


def test_wav_encode_decode_roundtrip(engine, tmp_path):
	x = _makeSine()

	# Encode to bytes and to a file; both go through the same synchronous sink.
	wavBytes = oa.FnAudio.encodeWavF32(x)
	assert isinstance(wavBytes, bytes) and len(wavBytes) > 44  # header + samples

	path = str(tmp_path / "tone.wav")
	oa.FnAudio.saveWavF32(path, x)

	decoded = oa.FnAudio.decodeFile(path)
	assert decoded.isValid()
	assert decoded.sampleRate == SAMPLE_RATE
	assert decoded.channelCount == 1
	assert decoded.sampleCount == NUM_SAMPLES

	assert decoded.layout == oa.AudioChannelLayout.Mono
	assert abs(decoded.durationSeconds() - NUM_SAMPLES / SAMPLE_RATE) < 1e-6

	# LoadMemory on the encoded bytes must agree with LoadFile.
	fromMem = oa.FnAudio.decodeMemory(wavBytes)
	assert fromMem.sampleCount == decoded.sampleCount


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
