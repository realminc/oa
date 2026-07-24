// OA Tutorial: TutorialViewerVideoVp9 — VP9 playback through OaViewer.

#include "TutorialViewerVideo.h"

int main(int argc, char** argv) {
	return RunTutorialViewerVideo(
		argc,
		argv,
		TutorialVideoPath("shibuya_crossing_1080p30_vp9.mp4"),
		"OA Viewer · VP9 source");
}
