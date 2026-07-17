// OaPlot::Axes — one subplot inside an OaPlot::Figure.
//
// Records the compact image/line/heatmap surface. The parent Figure replays it
// through a swapchain (Show) or a deterministic PNG sink (SaveFig).

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

struct HeatmapStyle {
	OaF32 VMin = 0.0F;
	OaF32 VMax = 1.0F;
	OaU32 Colormap = 1; // viridis
	bool AutoScale = true;
	bool ShowGrid = true;
};

// ─── Axes ──────────────────────────────────────────────────────────────────

class Axes {
public:
	// Image grid call. The texture is blitted into this axes subregion through
	// OaContext::RecordBlit (cross-format vkCmdBlitImage when needed).
	void Imshow(const OaTexture& InTex);

	// Line plot of N values (y[i] vs i). Supported by both sinks.
	void Plot(OaSpan<const OaF32> InY,
	          const LineStyle& InStyle = {});

	// Dense row-major heatmap. This is the compact plotting contract for
	// confusion matrices and other evaluation tables.
	void Heatmap(OaSpan<const OaF32> InValues, OaI32 InRows, OaI32 InCols,
		const HeatmapStyle& InStyle = {});

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
	struct HeatmapCmd { OaVec<OaF32> V; OaI32 Rows = 0, Cols = 0; HeatmapStyle Style; bool Present = false; };

	ImshowCmd  Image_;
	TextCmd    Title_;
	TextCmd    Caption_;
	TextCmd    XLabel_;
	TextCmd    YLabel_;
	LineCmd    Line_;
	HeatmapCmd Heatmap_;
	OaColor  Border_      = {0.0F, 0.0F, 0.0F, 0.0F};  // transparent = no border
	bool       HasBorder_   = false;
};

}  // namespace OaPlot
