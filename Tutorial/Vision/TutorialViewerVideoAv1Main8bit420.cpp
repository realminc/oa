// OA Tutorial: AV1 Main, 8-bit 4:2:0 profile fixture through OaViewer.

#include "TutorialViewerVideo.h"

int main(int argc, char** argv) {
	return RunTutorialViewerVideo(
		argc,
		argv,
		TutorialVideoAssetPath("shibuya_720p_av1_main_8bit_420.mp4"),
		"OA Viewer · AV1 Main · 8-bit 4:2:0",
		false);
}
