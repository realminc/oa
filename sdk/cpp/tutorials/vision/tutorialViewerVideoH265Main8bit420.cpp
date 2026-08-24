// OA Tutorial: H.265 main, 8-bit 4:2:0 profile fixture through oa::Viewer.

#include "tutorialViewerVideo.h"

int main(int argc, char** argv) {
	return runTutorialViewerVideo(
		argc,
		argv,
		tutorialVideoAssetPath("shibuya_720p_h265_main_8bit_420.mp4"),
		"OA Viewer · H.265 main · 8-bit 4:2:0",
		false);
}
