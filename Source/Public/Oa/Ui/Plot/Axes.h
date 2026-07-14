// OaPlot::Axes — one subplot inside an OaPlot::Figure.
//
// Phase-1 surface (UnifiedExecutionArchitecture.md §3.5 / OaUiFinalGlueBridge.md §5.4): records
// matplotlib-style draw calls. The parent Figure replays them either through
// a swapchain (Show) or into an off-screen compose image (SaveFig). Imshow
// records as a ctx.RecordBlit(source → axes subregion); text/lines/bars
// route through OaUi widgets when a UI is available.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Ui/Image.h>      // OaTexture
#include <Oa/Core/Color.h>    // OaColor

namespace OaPlot {

class Figure;

// ─── Per-call optional formatting ──────────────────────────────────────────

struct LineStyle {
	OaColor Color     = {0.388F, 0.400F, 0.945F, 1.0F};  // Realm accent
	OaF32     Width     = 1.5F;
};

struct BarStyle {
	OaColor Color     = {0.388F, 0.400F, 0.945F, 1.0F};
};

struct ScatterStyle {
	OaColor Color     = {0.388F, 0.400F, 0.945F, 1.0F};
	OaF32     Size      = 4.0F;
};

// ─── Axes ──────────────────────────────────────────────────────────────────

class Axes {
public:
	// Image grid call. The texture is blitted into this axes subregion through
	// OaContext::RecordBlit (cross-format vkCmdBlitImage when needed).
	void Imshow(const OaTexture& InTex);

	// Line plot of N values (y[i] vs i). Routes to OaUi::PlotLine when a UI
	// is present; ignored in headless SaveFig mode (Phase-1 limitation).
	void Plot(OaSpan<const OaF32> InY,
	          const LineStyle& InStyle = {});

	// Bar plot. UI-only in Phase 1.
	void Bar(OaSpan<const OaF32> InValues,
	         const BarStyle& InStyle = {});

	// Scatter plot. UI-only in Phase 1.
	void Scatter(OaSpan<const OaF32> InXs,
	             OaSpan<const OaF32> InYs,
	             const ScatterStyle& InStyle = {});

	// Title above the axes. Color controls the text color (red/green for the
	// classify-tutorial correct/wrong indicator).
	void Title(const char* InText,
	           OaColor InColor = {0.961F, 0.961F, 0.961F, 1.0F});

	// X / Y axis labels (below / left of the axes).
	void XLabel(const char* InText);
	void YLabel(const char* InText);

	// Additional caption line under the image (used by the classify tutorial
	// to show the ground-truth label when prediction is wrong).
	void Caption(const char* InText,
	             OaColor InColor = {0.565F, 0.565F, 0.565F, 1.0F});

	// Border / frame color around the axes rect. Defaults to no border.
	void BorderColor(OaColor InColor);

	// Default-constructible so OaPlot::Figure can hold OaVec<Axes>. Use
	// Figure::Ax(r, c) to obtain a reference rather than constructing one
	// yourself.
	Axes() = default;

private:
	friend class Figure;

	// ── Recorded state — kept tiny; one Axes per grid cell, replayed each
	//    frame (Show) or once (SaveFig). ────────────────────────────────────

	struct ImshowCmd  { OaTexture Tex; bool Present = false; };
	struct TextCmd    { OaString  Text; OaColor Color; bool Present = false; };
	struct LineCmd    { OaVec<OaF32> Y;    LineStyle Style;  bool Present = false; };
	struct BarCmd     { OaVec<OaF32> V;    BarStyle Style;   bool Present = false; };
	struct ScatterCmd { OaVec<OaF32> Xs, Ys; ScatterStyle Style; bool Present = false; };

	ImshowCmd  Image_;
	TextCmd    Title_;
	TextCmd    Caption_;
	TextCmd    XLabel_;
	TextCmd    YLabel_;
	LineCmd    Line_;
	BarCmd     Bar_;
	ScatterCmd Scatter_;
	OaColor  Border_      = {0.0F, 0.0F, 0.0F, 0.0F};  // transparent = no border
	bool       HasBorder_   = false;
};

}  // namespace OaPlot
