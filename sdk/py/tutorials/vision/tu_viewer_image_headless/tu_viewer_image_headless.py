"""Decode and save a semantic image without a window.

Usage: python sdk/py/tutorials/vision/tu_image_headless.py [input] [output]
"""

import sys

import oa


source = sys.argv[1] if len(sys.argv) > 1 else "sdk/asset/image/coverMl.jpg"
output = sys.argv[2] if len(sys.argv) > 2 else "/tmp/oa_viewer_headless.png"
engine = oa.Engine()
image = oa.image.decode_file(engine, source, "rgba")
oa.image.save_file(output, image)
print(f"wrote {output}")
