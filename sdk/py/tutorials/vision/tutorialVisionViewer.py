#!/usr/bin/env python3
"""Load, display, resize, and display one image with Viewer."""

# OA_DOC_BEGIN: viewer-intro
# pyright: reportWildcardImportFromLibrary=false
from oa import *


asset = Paths.asset("image/coverMl.jpg")
image = FnImage.decodeFile(asset)
Viewer.show(image, title="Space Cathedral · original")

resized = FnImage.resize(image, 836, 471)
Viewer.show(resized, title="Space Cathedral · 836×471")
# OA_DOC_END: viewer-intro
