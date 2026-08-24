// OA Tutorial: AV1 main, 8-bit 4:2:0 profile fixture through oa::Viewer.

#include "tutorialViewerVideo.h"

int main(int argc, char** argv) {
	return runTutorialViewerVideo(
		argc,
		argv,
		tutorialVideoAssetPath("shibuya_720p_av1_main_8bit_420.mp4"),
		"OA Viewer · AV1 main · 8-bit 4:2:0",
		false);
}
