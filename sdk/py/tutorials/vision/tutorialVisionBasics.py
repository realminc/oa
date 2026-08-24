#!/usr/bin/env python3
"""Decode, resize, adjust, and inspect one image with OA Vision.
"""

# pyright: reportWildcardImportFromLibrary=false
from oa import *


asset = Paths.asset("image/visionTestPattern320x180.jpg")
image = FnImage.decodeFile(asset)
small = FnImage.resize(image, 160, 90)
adjusted = FnImage.brightnessContrast(small, 0.05, 1.1)
values = FnMatrix.copyToHost(adjusted.asMatrix())

assert adjusted.asMatrix().shape() == [1, 3, 90, 160]
assert len(values) == 3 * 90 * 160
print(adjusted.asMatrix().shape(), min(values), max(values))
