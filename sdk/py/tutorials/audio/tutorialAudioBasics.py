#!/usr/bin/env python3
"""Decode, normalize, and save one audio file with the C++-parity OA API.
"""

# pyright: reportWildcardImportFromLibrary=false
from oa import *


asset = Paths.asset("audio/oaNarration.flac")
output = Paths.var("tutorial/audio/oa_audio_normalized.wav")

audio = FnAudio.decodeFile(asset)
clean = FnAudio.normalize(audio, -3.0)
FnAudio.saveWavF32(output, clean)

print(Filesystem.absolute(output))
