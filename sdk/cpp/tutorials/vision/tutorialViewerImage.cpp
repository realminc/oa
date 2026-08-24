// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial: oa::Viewer — Universal GPU-Accelerated Image Viewer
// level 0 API — oa::Viewer (preconfigured app class)
// ═══════════════════════════════════════════════════════════════════════════
//
// Demonstrates the unified viewer application with interactive image
// navigation.
//
// features:
//   - oa::Navigation: LMB/MMB/wheel pan, Maya RMB zoom, ctrl+wheel dolly
//   - keyboard: +/- zoom, 0/F fit, 9 = 100%, arrows pan
//
// Parallel structure to OpenCV's image display tutorial:
//   https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html
//
//   OpenCV                           OA C++
//   ─────────────────────────────    ─────────────────────────────────────
//   cv::imread(path)                 oa::Viewer viewer(path);
//   cv::imshow("name", mat)          viewer.run();
//   cv::waitKey(0)
//
// usage:  ./Tutorial/TutorialViewerImage [image.jpg]
//
// Controls: same as oa::Viewer — see oa::navigationHelpLine() at startup
// ═══════════════════════════════════════════════════════════════════════════

// OA_DOC_BEGIN: viewer-intro
#include <oa/ui/viewer.h>

// ─── main ──────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
	const char* path = (argc > 1) ? argv[1] : "asset/image/SpaceCathedral.jpg";

	oa::Viewer viewer(path);

	return viewer.run().isOk() ? 0 : 1;
}
// OA_DOC_END: viewer-intro
