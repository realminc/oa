// OA Tutorial: H.264 High, 8-bit 4:2:0 profile fixture through OaViewer.

#include "TutorialViewerVideo.h"

int main(int argc, char** argv) {
	return RunTutorialViewerVideo(
		argc,
		argv,
		TutorialVideoAssetPath("shibuya_720p_h264_high_8bit_420.mp4"),
		"OA Viewer · H.264 High · 8-bit 4:2:0",
		false);
}
