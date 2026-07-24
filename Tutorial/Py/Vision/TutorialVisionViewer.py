#!/usr/bin/env python3
"""Load, display, resize, and display one image with OaViewer.
"""

# pyright: reportWildcardImportFromLibrary=false
from oa import *


asset = OaPaths.Asset("Image/VisionTestPattern320x180.jpg")
image = OaImageDecoder.LoadFile(asset)
OaViewer.Show(image, Title="Original · 320×180")

resized = OaFnImage.Resize(image, 160, 90)
OaViewer.Show(resized, Title="Resized · 160×90")
