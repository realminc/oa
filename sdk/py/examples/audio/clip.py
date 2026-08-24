#!/usr/bin/env python3
"""Clip a semantic audio value on the GPU and preserve its metadata.
"""

# OA_DOC_BEGIN: audio-clip
from oa import (
	Audio,
	AudioChannelLayout,
	FnAudio,
	FnMatrix,
	initComputeEngine,
	shutdownComputeEngine,
)


if not initComputeEngine():
	raise RuntimeError("OA could not create a Vulkan compute engine")

try:
	audio = Audio(
		FnMatrix.full([1, 8], 0.75),
		48_000,
		AudioChannelLayout.Mono,
	)
	clipped = FnAudio.clip(audio, -0.5, 0.5)
	values = FnMatrix.copyToHost(clipped.matrix)
	assert clipped.isValid()
	assert clipped.sampleRate == 48_000
	assert values == [0.5] * 8
	print("8 mono samples clipped to 0.5")
finally:
	shutdownComputeEngine()
# OA_DOC_END: audio-clip
