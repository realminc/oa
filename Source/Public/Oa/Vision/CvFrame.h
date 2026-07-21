// OaCv::Frame — detection overlay drawing surface bound to an OaPlot::Axes.
//
// Architecture/OaArchitecture.md §10. Thin compatibility
// wrapper over OaCvFrame (Oa/Ui/Cv.h) that integrates the existing CPU
// composite path with the OaPlot canvas. The flow is:
//
//   OaPlot::Figure fig({.Rows=1, .Cols=1, .Width=W, .Height=H});
//   OaCv::Frame frame(fig.Ax(0,0), baseImage);
//   frame.BBoxes(boxes, {.Color = OaColor::Accent});
//   (void)frame.Commit(rt);   // renders overlay tex, assigns to axes
//   fig.Rasterize(rt);        // canvas now contains base + boxes
//   (void)fig.Show();         // same renderer body, different content
//   (void)fig.SaveFig(path);  // same renderer body, batch sink
//
// This plot/save adapter intentionally uses the CPU reference compositor.
// Realtime image/video display uses OaDetectionOverlay and never replaces the
// full decoded frame with a host-composited upload.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Color.h>
#include <Oa/Ui/Cv.h>       // OaCvBbox, OaCvFrame, OaCvBboxesConfig, OaCvMask
#include <Oa/Ui/Image.h>    // OaTexture
#include <Oa/Ui/Plot/Axes.h>   // OaPlot::Axes

class OaEngine;
class OaContext;

namespace OaCv {

// Re-export OaCvBbox as OaCv::BBox so call sites can stay namespaced.
using BBox = OaCvBbox;

struct BBoxStyle {
	OaColor Color      = OaColor::Accent();  // Realm Design System accent
	OaF32   LineWidth  = 2.0F;
	OaF32   Alpha      = 1.0F;
	bool    ShowLabels = true;
	bool    ShowScores = true;
};

class Frame {
public:
	// Default-construct an empty Frame — only useful as a placeholder before
	// the real frame is move-assigned in (e.g. an app's OnInit). Methods on a
	// default-constructed Frame are no-ops; Commit fails with "no axes bound".
	Frame() = default;

	// InAx must outlive this Frame. The Frame holds a borrowed reference and
	// assigns the rendered overlay to InAx on Commit().
	// InBase is the underlying RGBA8 image — bbox coords are interpreted in
	// its pixel space.
	Frame(OaPlot::Axes& InAx, const OaTexture& InBase);

	Frame(const Frame&)            = delete;
	Frame& operator=(const Frame&) = delete;
	// Custom move: OaCvFrame::Base is a raw pointer into our own Base_
	// member, so move-construction and move-assignment have to rebind it
	// to the new Base_ — otherwise it dangles into the moved-from object.
	Frame(Frame&& InOther) noexcept;
	Frame& operator=(Frame&& InOther) noexcept;
	~Frame() = default;

	// ── Overlay accumulators ─────────────────────────────────────────────

	// Add a group of boxes with shared style. Call multiple times to use
	// distinct colours per class.
	void BBoxes(OaVec<BBox> InBoxes, const BBoxStyle& InStyle = {});

	// Add segmentation masks. Routes to OaCvFrame::AddMasks.
	void Masks(OaVec<OaCvMask> InMasks, const OaCvMasksConfig& InCfg = {});

	void Stats();

	// ── Commit ────────────────────────────────────────────────────────────

	// Renders the overlay (Base + accumulated decorations) into a fresh
	// RGBA8 OaTexture and assigns it to the axes via OaPlot::Axes::Imshow.
	// The overlay handle is held internally; call Destroy() to free its GPU
	// memory after the figure has been Rasterized (and the canvas no longer
	// needs to read it back).
	[[nodiscard]] OaStatus Commit(OaContext& InContext);
	[[nodiscard]] OaStatus Commit(OaEngine& InRt);

	[[nodiscard]] const OaTexture& Overlay() const noexcept { return Overlay_; }

	// Releases the overlay's GPU memory. Safe to call multiple times.
	void Destroy(OaEngine& InRt);

private:
	OaPlot::Axes* Ax_  = nullptr;
	OaTexture     Base_;
	OaCvFrame     Cv_;
	OaTexture     Overlay_;
};

}  // namespace OaCv
