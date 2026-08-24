// OA Tutorial: AV1 main, 8-bit 4:2:0 profile fixture through oa::Viewer.

#include "tutorialViewerVideo.h"

int main(int argc, char** argv) {
	return runTutorialViewerVideo(
		argc,
		argv,
		tutorialVideoAssetPath("shibuya720pAv1MainEightBit420.mp4"),
		"OA Viewer · AV1 main · 8-bit 4:2:0",
		false);
}
