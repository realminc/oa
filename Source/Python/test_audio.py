#!/usr/bin/env python3
"""Smoke tests for OA's public root-parity Audio API.

Self-contained: builds a synthetic waveform on the GPU, exercises the OaFnAudio
DSP surface, and round-trips it through the WAV-F32 codec boundary — no external
audio asset required. Run from a checkout with:

	python -m pytest Source/Python/test_audio.py

If the native extension is not on PYTHONPATH, set OA_PYTHON_BUILD_DIR to the
directory containing the private `_oa` extension.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _add_dev_paths() -> None:
	candidates: list[Path] = []

	build_dir = os.getenv("OA_PYTHON_BUILD_DIR")
	if build_dir:
		candidates.append(Path(build_dir).expanduser())

	candidates.extend(
		[
			REPO_ROOT / "Build" / "Release",
			REPO_ROOT / "Build" / "Debug",
			REPO_ROOT / "build",
			REPO_ROOT / "Source" / "Python",
		]
	)

	for path in candidates:
		if path.exists():
			path_str = str(path)
			if path_str not in sys.path:
				sys.path.insert(0, path_str)


_add_dev_paths()

oa = pytest.importorskip("oa", reason="oa Python package is not importable")
# Lowercase modules remain compatibility aliases during migration. The tests
# call the canonical root and OaFnAudio surfaces, then verify alias identity.
audio = oa.audio
core = oa.core
runtime = oa.runtime


@pytest.fixture(scope="session")
def engine():
	if not runtime.OaInitComputeEngine():
		pytest.skip("OA compute engine could not initialize, likely no Vulkan device")
	yield
	shutdown = getattr(runtime, "OaShutdownComputeEngine", None)
	if shutdown is not None:
		shutdown()


SAMPLE_RATE = 16000
NUM_SAMPLES = 4096


def _make_sine(freq: float = 440.0) -> "oa.OaAudio":
	# Mono [1, NUM_SAMPLES] tone plus a small deterministic broadband floor so
	# log-domain features (mel/MFCC) never hit a near-silent band.
	seed = 12345
	data = []
	for n in range(NUM_SAMPLES):
		seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
		noise = (seed / 0x7FFFFFFF - 0.5) * 0.02
		data.append(0.5 * math.sin(2.0 * math.pi * freq * n / SAMPLE_RATE) + noise)
	matrix = oa.OaFnMatrix.FromFloats(data, 1, NUM_SAMPLES)
	return oa.OaAudio(matrix, SAMPLE_RATE, oa.OaChannelLayout.Mono)


# ── Surface / construction (no GPU) ──────────────────────────────────────────


def test_audio_import_surface():
	for name in (
		"OaAudio", "OaAudioDecoder", "OaAudioEncoder",
		"OaChannelLayout", "OaStftConfig", "OaMelConfig", "OaMfccConfig",
		"OaResampleConfig", "OaNormalizeAudioConfig",
	):
		assert hasattr(oa, name), name

	for name in (
		"Stft", "MelSpectrogram", "Mfcc", "Normalize", "Resample", "Gain",
		"Clip", "AmplitudeToDb", "PreEmphasis", "ToMono", "Fade", "Mix",
	):
		assert hasattr(oa.OaFnAudio, name), name

	assert oa.OaAudio is audio.OaAudio
	assert oa.OaAudioDecoder is audio.OaAudioDecoder
	assert oa.OaFnAudio.Normalize is audio.Normalize


def test_config_structs_roundtrip_fields():
	stft = oa.OaStftConfig()
	stft.FftSize = 512
	stft.HopSize = 160
	stft.WinSize = 512
	assert stft.FftSize == 512 and stft.HopSize == 160

	mel = oa.OaMelConfig()
	mel.NumMels = 40
	mel.FftSize = 512
	mel.HopSize = 160
	assert mel.NumMels == 40

	mfcc = oa.OaMfccConfig()
	mfcc.NumCoeffs = 13
	mfcc.Mel = mel
	assert mfcc.NumCoeffs == 13 and mfcc.Mel.NumMels == 40


def test_channel_layout_helpers():
	assert oa.OaChannelsForLayout(oa.OaChannelLayout.Stereo) == 2
	assert oa.OaLayoutForChannels(1) == oa.OaChannelLayout.Mono
	assert oa.OaLayoutForChannels(2) == oa.OaChannelLayout.Stereo


# ── GPU DSP ──────────────────────────────────────────────────────────────────


def test_mel_spectrogram_shape(engine):
	x = _make_sine()
	cfg = oa.OaMelConfig()
	cfg.FftSize = 512
	cfg.HopSize = 160
	cfg.NumMels = 40
	with oa.Context():
		mel = oa.OaFnAudio.MelSpectrogram(x, cfg)
	shape = mel.Shape()
	# [Channels, NumMels, Frames]
	assert shape[0] == 1 and shape[1] == 40 and shape[2] > 0


def test_stft_shape(engine):
	x = _make_sine()
	cfg = oa.OaStftConfig()
	cfg.FftSize = 512
	cfg.HopSize = 160
	cfg.WinSize = 512
	with oa.Context():
		spec = oa.OaFnAudio.Stft(x, cfg)
	shape = spec.Shape()
	# [Channels, Frames, FftSize/2 + 1]
	assert shape[0] == 1 and shape[2] == 512 // 2 + 1


def test_signal_ops_shapes(engine):
	x = _make_sine()
	with oa.Context():
		gained = oa.OaFnAudio.Gain(x, -6.0)
		clipped = oa.OaFnAudio.Clip(x, -0.5, 0.5)
		normalized = oa.OaFnAudio.Normalize(x, -3.0, 0)
		mono = oa.OaFnAudio.ToMono(x)
		faded = oa.OaFnAudio.Fade(x, 128, 128)
	assert gained.Matrix.Shape() == [1, NUM_SAMPLES]
	assert clipped.Matrix.Shape() == [1, NUM_SAMPLES]
	assert normalized.Matrix.Shape() == [1, NUM_SAMPLES]
	assert mono.Matrix.Shape() == [1, NUM_SAMPLES]
	assert faded.Matrix.Shape() == [1, NUM_SAMPLES]
	assert gained.SampleRate == SAMPLE_RATE
	assert gained.Layout == oa.OaChannelLayout.Mono


def test_resample_length(engine):
	x = _make_sine()
	with oa.Context():
		out = oa.OaFnAudio.Resample(x, SAMPLE_RATE * 2, 64)
	# Upsample 2x → roughly double the sample count.
	assert out.Matrix.Shape()[1] == NUM_SAMPLES * 2
	assert out.SampleRate == SAMPLE_RATE * 2


def test_clip_bounds_values(engine):
	x = _make_sine()
	with oa.Context():
		clipped = oa.OaFnAudio.Clip(x, -0.1, 0.1)
	host = oa.OaFnMatrix.CopyToHost(clipped.Matrix)
	assert all(-0.1 - 1e-5 <= v <= 0.1 + 1e-5 for v in host)


# ── Codec round-trip ──────────────────────────────────────────────────────────


def test_wav_encode_decode_roundtrip(engine, tmp_path):
	x = _make_sine()

	# Encode to bytes and to a file; both go through the same synchronous sink.
	wav_bytes = oa.OaAudioEncoder.EncodeWavF32(x)
	assert isinstance(wav_bytes, bytes) and len(wav_bytes) > 44  # header + samples

	path = str(tmp_path / "tone.wav")
	oa.OaAudioEncoder.SaveWavF32(path, x)

	decoded = oa.OaAudioDecoder.LoadFile(path)
	assert decoded.IsValid()
	assert decoded.SampleRate == SAMPLE_RATE
	assert decoded.ChannelCount == 1
	assert decoded.SampleCount == NUM_SAMPLES

	assert decoded.Layout == oa.OaChannelLayout.Mono
	assert abs(decoded.DurationSeconds() - NUM_SAMPLES / SAMPLE_RATE) < 1e-6

	# LoadMemory on the encoded bytes must agree with LoadFile.
	from_mem = oa.OaAudioDecoder.LoadMemory(wav_bytes)
	assert from_mem.SampleCount == decoded.SampleCount


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
