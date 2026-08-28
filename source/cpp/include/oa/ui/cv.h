// Ui — legacy CPU reference CV compositing.
//
// CvFrame remains a diagnostic/reference path for saved images with CPU
// bounding-box annotation. It reads back and uploads a complete RGBA frame and
// must not be used for realtime playback. Use <oa/ui/detectionOverlay.h> for
// completion-safe, GPU-resident boxes and hinted coverage labels over
// image/video views.
//
// OA does not depend on OpenCV.
//
// usage:
//   CvFrame frame;
//   frame.base = camera_buf;
//   frame.w = 1280;  frame.h = 720;
//   frame.addBboxes(detections, {.showLabels = true});
//   auto annotated = frame.render(engine);

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/core/color.h>

namespace oa { class Engine; }
namespace oa { class Texture; }
namespace oavk { class Buffer; }


// ─── Overlay config structs ───────────────────────────────────────────────────

namespace oa {

struct CvBbox {
	oa::F32      x         = 0.0F;
	oa::F32      y         = 0.0F;
	oa::F32      w         = 0.0F;
	oa::F32      h         = 0.0F;
	oa::F32      score     = 1.0F;
	oa::I32      classId   = 0;
	oa::String   label;
};

struct CvBboxesConfig {
	oa::Color color      = oa::Color::error();
	oa::F32   lineWidth  = 1.5F;
	oa::F32   alpha      = 1.0F;
	oa::I32   labelScale = 1;
	bool    showLabels = true;
	bool    showScores = true;
};

// ─── CvOverlay ──────────────────────────────────────────────────────────────

struct CvOverlayBboxes { CvBboxesConfig config; oa::Vector<CvBbox> boxes; };
using CvOverlay = CvOverlayBboxes;


// ─── CvFrame ────────────────────────────────────────────────────────────────

struct CvFrame {
	const oavk::Buffer*   base    = nullptr;  // borrowed RGBA8 device buffer
	oa::I32               w       = 0;
	oa::I32               h       = 0;
	oa::Vector<CvOverlay>  overlays;

	void addBboxes(oa::Vector<CvBbox>         inBoxes, const CvBboxesConfig& inCfg = {});

	void clearOverlays() noexcept { overlays.clear(); }

	// CPU reference/diagnostic composite -> upload RGBA8 oa::Texture.
	// pending work in the engine's matching private recorder completes before
	// reading base. Realtime consumers must use DetectionOverlay or another
	// resident path.
	[[nodiscard]] oa::Result<oa::Texture> render(oa::Engine& inRt) const;
	[[nodiscard]] oa::Result<oa::Texture> render(
		oa::Engine& inRt,
		oa::Span<const oa::U8> inBaseRgba) const;
};

}  // namespace oa
