#!/usr/bin/env python3
"""Resize a semantic image on the GPU and preserve its metadata.
"""

# OA_DOC_BEGIN: vision-resize
from oa import (
	FnImage,
	FnMatrix,
	Image,
	ImageFormat,
	ImageLayout,
	initComputeEngine,
	shutdownComputeEngine,
)


if not initComputeEngine():
	raise RuntimeError("OA could not create a Vulkan compute engine")

try:
	image = Image(
		FnMatrix.full([1, 3, 2, 2], 0.25),
		ImageLayout.Nchw,
		ImageFormat.Rgb,
	)
	resized = FnImage.resize(image, 4, 3)
	values = FnMatrix.copyToHost(resized.asMatrix())
	assert resized.validate()
	assert resized.width() == 4
	assert resized.height() == 3
	assert values == [0.25] * 36
	print("RGB NCHW image resized from 2x2 to 4x3")
finally:
	shutdownComputeEngine()
# OA_DOC_END: vision-resize
