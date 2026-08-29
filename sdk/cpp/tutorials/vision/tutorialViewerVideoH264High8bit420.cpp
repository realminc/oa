// OA Tutorial: H.264 high, 8-bit 4:2:0 profile fixture through oa::Viewer.

#include "tutorialViewerVideo.h"

int main(int argc, char** argv) {
	return runTutorialViewerVideo(
		argc,
		argv,
		tutorialVideoAssetPath("clip/shibuya_720p_30fps_h264_high_8bit_420.mp4"),
		"OA Viewer · H.264 high · 8-bit 4:2:0",
		false);
}
