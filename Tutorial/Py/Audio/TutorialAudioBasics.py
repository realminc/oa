#!/usr/bin/env python3
"""Decode, normalize, and save one audio file with the C++-parity OA API.
"""

# pyright: reportWildcardImportFromLibrary=false
from oa import *


asset = OaPaths.Asset("Audio/0_jackson_0.flac")
output = OaPaths.Var("tutorial/audio/oa_audio_normalized.wav")

audio = OaAudioDecoder.LoadFile(asset)
clean = OaFnAudio.Normalize(audio, -3.0)
OaAudioEncoder.SaveWavF32(output, clean)

print(OaFilesystem.Absolute(output))
