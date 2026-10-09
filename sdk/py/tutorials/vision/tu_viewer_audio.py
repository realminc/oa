"""Open audio with waveform, spectrum, or Mel analysis.

Usage: python sdk/py/tutorials/vision/tu_viewer_audio.py [audio] [waveform|spectrum|mel]
"""

import sys

import oa


path = sys.argv[1] if len(sys.argv) > 1 else "sdk/asset/audio/oaNarration.flac"
view = sys.argv[2] if len(sys.argv) > 2 else "waveform"
config = oa.ViewerConfig()
config.mode = oa.ViewerMode.Audio
config.audio_view = {
	"spectrum": oa.ViewerAudioView.Spectrum,
	"mel": oa.ViewerAudioView.Mel,
}.get(view, oa.ViewerAudioView.Waveform)
config.title = "OA Viewer · Audio"
oa.Viewer.preview(path, config)
