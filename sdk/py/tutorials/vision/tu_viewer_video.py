"""Open an H.264, H.265, VP9, or AV1 video.

Usage: python sdk/py/tutorials/vision/tu_viewer_video.py [video]
"""

import sys

import oa


path = (
	sys.argv[1]
	if len(sys.argv) > 1
	else "sdk/asset/clip/shibuya_720p_30fps_h264_high_8bit_420.mp4"
)
config = oa.ViewerConfig()
config.mode = oa.ViewerMode.Video
config.loop_media = False
config.title = "OA Viewer · Video"
oa.Viewer.preview(path, config)
