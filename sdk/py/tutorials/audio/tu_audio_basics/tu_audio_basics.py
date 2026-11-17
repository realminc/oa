#!/usr/bin/env python3
"""Decode, normalize, and save one audio file with the OA audio API.

Mirrors sdk/cpp/tutorials/core/tuCoreEngine.cpp — basic Audio DSP.
"""

import os

import oa

engine = oa.Engine()

asset_root = os.environ.get("OA_ASSET", "asset")
asset = os.path.join(asset_root, "audio/oaNarration.flac")
output_root = os.environ.get("OA_VAR", "var")
output = os.path.join(output_root, "tutorial/audio/oaNarrationNormalized.wav")
os.makedirs(os.path.dirname(output), exist_ok=True)

audio = oa.audio.decode_file(engine, asset)
clean = oa.audio.normalize(audio, -3.0)
oa.audio.save_wav_f32(output, clean)

print(f"Channels: {clean.channels}  Sample-rate: {clean.sample_rate}  "
	  f"Duration: {clean.duration_seconds:.2f} s")
print(f"Saved: {os.path.abspath(output)}")
