// OA Tutorial: H.265 Main, 8-bit 4:2:0 profile fixture through OaViewer.

#include "TutorialViewerVideo.h"

int main(int argc, char** argv) {
	return RunTutorialViewerVideo(
		argc,
		argv,
		TutorialVideoAssetPath("shibuya_720p_h265_main_8bit_420.mp4"),
		"OA Viewer · H.265 Main · 8-bit 4:2:0",
		false);
}
