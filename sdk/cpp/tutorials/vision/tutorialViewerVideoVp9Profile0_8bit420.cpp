// OA Tutorial: VP9 profile 0, 8-bit 4:2:0 profile fixture through oa::Viewer.

#include "tutorialViewerVideo.h"

int main(int argc, char** argv) {
	return runTutorialViewerVideo(
		argc,
		argv,
		tutorialVideoAssetPath("shibuya720pVp9Profile0EightBit420.mp4"),
		"OA Viewer · VP9 profile 0 · 8-bit 4:2:0",
		false);
}
