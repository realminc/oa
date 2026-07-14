// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial: OaViewer — Universal GPU-Accelerated Viewer (Timeline Animated)
// Level 0 API — OaViewer (preconfigured app class)
// ═══════════════════════════════════════════════════════════════════════════
//
// Demonstrates the unified viewport architecture with timeline-driven animation
// for smooth pan/zoom effects. Uses OaTimeline for deterministic time evaluation.
//
// Features:
//   - OaNavigation: LMB/MMB/wheel pan, Maya RMB zoom, Ctrl+wheel dolly
//   - Keyboard: +/- zoom, 0/F fit, 9 = 100%, arrows pan
//
// Architecture: CoreAnimationRenderArchitecture.md §9, §5.3
//
// Parallel structure to OpenCV's image display tutorial:
//   https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html
//
//   OpenCV                           OA C++
//   ─────────────────────────────    ─────────────────────────────────────
//   cv::imread(path)                 OaViewer viewer(path);
//   cv::imshow("name", mat)          viewer.Run();
//   cv::waitKey(0)
//
// Usage:  ./Tutorial/TutorialViewportImage [image.jpg]
//
// Controls: same as OaViewer — see OaNavigationHelpLine() at startup
// ═══════════════════════════════════════════════════════════════════════════

#include <Oa/Ui/ImageViewer.h>

// ─── main ──────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
	const char* path = (argc > 1) ? argv[1] : "Asset/Image/Realm1024px.jpg";

	OaImageViewer viewer(path);

	return viewer.Run().IsOk() ? 0 : 1;
}
