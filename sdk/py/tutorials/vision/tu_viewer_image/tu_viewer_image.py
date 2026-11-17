"""Open an image in the blocking OA Viewer.

Usage: python sdk/py/tutorials/vision/tu_viewer_image.py [image]
"""

import sys

import oa


path = sys.argv[1] if len(sys.argv) > 1 else "sdk/asset/image/coverMl.jpg"
oa.Viewer.preview(path)
