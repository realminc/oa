// OA Tutorial: TutorialViewerVideoAv1 — AV1 playback through OaViewer.

#include "TutorialViewerVideo.h"

int main(int argc, char** argv) {
	return RunTutorialViewerVideo(
		argc,
		argv,
		TutorialVideoPath("shibuya_crossing_1080p30_av1.mp4"),
		"OA Viewer · AV1 source");
}
