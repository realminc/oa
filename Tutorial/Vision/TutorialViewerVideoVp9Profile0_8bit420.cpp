// OA Tutorial: VP9 Profile 0, 8-bit 4:2:0 profile fixture through OaViewer.

#include "TutorialViewerVideo.h"

int main(int argc, char** argv) {
	return RunTutorialViewerVideo(
		argc,
		argv,
		TutorialVideoAssetPath("shibuya_720p_vp9_profile0_8bit_420.mp4"),
		"OA Viewer · VP9 Profile 0 · 8-bit 4:2:0",
		false);
}
