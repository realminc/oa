// OA Tutorial: TutorialViewerVideoH264 — H.264 playback through OaViewer.

#include "TutorialViewerVideo.h"

int main(int argc, char** argv) {
	return RunTutorialViewerVideo(
		argc,
		argv,
		TutorialVideoPath("shibuya_crossing_1080p30_h264.mp4"),
		"OA Viewer · H.264 source");
}
