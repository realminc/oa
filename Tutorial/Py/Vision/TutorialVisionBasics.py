#!/usr/bin/env python3
"""Decode, resize, adjust, and inspect one image with OA Vision.
"""

# pyright: reportWildcardImportFromLibrary=false
from oa import *


asset = OaPaths.Asset("Image/VisionTestPattern320x180.jpg")
image = OaImageDecoder.LoadFile(asset)
small = OaFnImage.Resize(image, 160, 90)
adjusted = OaFnImage.BrightnessContrast(small, 0.05, 1.1)
values = OaFnMatrix.CopyToHost(adjusted.AsMatrix())

assert adjusted.AsMatrix().Shape() == [1, 3, 90, 160]
assert len(values) == 3 * 90 * 160
print(adjusted.AsMatrix().Shape(), min(values), max(values))
