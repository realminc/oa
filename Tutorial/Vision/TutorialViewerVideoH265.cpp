// OA Tutorial: TutorialViewerVideoH265 — H.265 playback through OaViewer.

#include "TutorialViewerVideo.h"

int main(int argc, char** argv) {
	return RunTutorialViewerVideo(
		argc,
		argv,
		TutorialVideoPath("shibuya_crossing_1080p30_h265.mp4"),
		"OA Viewer · H.265 source");
}
