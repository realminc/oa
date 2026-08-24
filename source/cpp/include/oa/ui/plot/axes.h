// oa::plot::Axes — one subplot inside an oa::plot::Figure.
//
// Records one optional raster base plus ordered vector artists. The parent
// Figure lowers the same commands through oa::Ui's GPU compositor for live and
// terminal sinks.

#pragma once

#include <oa/core/types.h>
#include <oa/ui/image.h>      // oa::Texture
#include <oa/core/color.h>    // oa::Color

namespace oa::plot {

class Figure;

// ─── Per-call optional formatting ──────────────────────────────────────────

struct LineStyle {
	// alpha zero selects the next deterministic theme palette color.
	oa::Color  color = {0.0F, 0.0F, 0.0F, 0.0F};
	oa::String label;
	// Implemented compute-line width and fixed subpixel coverage count.
	oa::F32    width = 1.35F;
	oa::U32    antialiasSamples = 4U;
};

struct ScatterStyle {
	// alpha zero selects the next deterministic theme palette color.
	oa::Color  color = {0.0F, 0.0F, 0.0F, 0.0F};
	oa::String label;
	oa::F32    radius = 3.0F;
};

struct BarStyle {
	// alpha zero selects the next deterministic theme palette color.
	oa::Color  color = {0.0F, 0.0F, 0.0F, 0.0F};
	oa::String label;
	// fraction of each category interval left empty, clamped to [0, 0.95].
	oa::F32    gap = 0.18F;
};

struct HeatmapStyle {
	oa::F32 vMin = 0.0F;
	oa::F32 vMax = 1.0F;
	oa::U32 colormap = 1; // viridis
	bool autoScale = true;
	bool showGrid = true;
};

// ─── Axes ──────────────────────────────────────────────────────────────────

class Axes {
public:
	// Image grid call. The texture is sampled directly by oa::Ui into this axes
	// subregion; Figure does not read its pixels back for composition.
	void imshow(const oa::Texture& inTex);

	// Line plot of N values (y[i] vs i). Repeated calls append ordered series.
	void plot(oa::Span<const oa::F32> inY, const LineStyle& inStyle = {});
	// Explicit-X line plot. X and Y must be non-empty and have equal length.
	void plot(oa::Span<const oa::F32> inX, oa::Span<const oa::F32> inY, const LineStyle& inStyle = {});

	// Point cloud / metric scatter. X and Y must have equal length.
	void scatter(oa::Span<const oa::F32> inX, oa::Span<const oa::F32> inY, const ScatterStyle& inStyle = {});

	// category bars at x = 0..n-1. Negative values extend below zero.
	void bar(oa::Span<const oa::F32> inY, const BarStyle& inStyle = {});
	// histogram convenience: finite samples are binned on the host into an
	// ordered bar artist. Metric histories are intentionally small telemetry.
	void histogram(oa::Span<const oa::F32> inValues, oa::I32 inBins = 16, const BarStyle& inStyle = {});

	// Fixed data limits. Invalid or degenerate input restores automatic limits.
	void limits(oa::F32 inXMin, oa::F32 inXMax, oa::F32 inYMin, oa::F32 inYMax);
	void autoLimits();
	// grid and labeled-series legend are enabled by default.
	void grid(bool inVisible = true);
	void legend(bool inVisible = true);

	// Dense row-major heatmap. This is the compact plotting contract for
	// confusion matrices and other evaluation tables.
	void heatmap(oa::Span<const oa::F32> inValues, oa::I32 inRows, oa::I32 inCols, const HeatmapStyle& inStyle = {});

	// title above the axes. Color controls the text color (red/green for the
	// classify-tutorial correct/wrong indicator). alpha zero uses theme text.
	void title(const char* inText, oa::Color inColor = {0.0F, 0.0F, 0.0F, 0.0F});

	// Centered X / Y axis labels (below / left of the plot). Y labels render
	// bottom-to-top; both reserve space rather than overlaying plot content.
	void xLabel(const char* inText);
	void yLabel(const char* inText);

	// Additional caption line under the image (used by the classify tutorial
	// to show the ground-truth label when prediction is wrong). alpha zero uses
	// theme secondary text.
	void caption(const char* inText, oa::Color inColor = {0.0F, 0.0F, 0.0F, 0.0F});

	// border / frame color around the axes rect. Defaults to no border.
	void borderColor(oa::Color inColor);

	// Default-constructible so oa::plot::Figure can hold oa::Vec<Axes>. Use
	// Figure::ax(r, c) to obtain a reference rather than constructing one
	// yourself.
	Axes() = default;

private:
	friend class Figure;

	// ── Recorded state — kept tiny; one Axes per grid cell, replayed each
	//    frame (show) or once (saveTo). ─────────────────────────────────────

	struct ImshowCmd  { oa::Texture tex; bool present = false; };
	struct TextCmd    { oa::String text; oa::Color color; bool present = false; };
	struct LineCmd    { oa::Vec<oa::F32> x; oa::Vec<oa::F32> y; LineStyle style; };
	struct ScatterCmd { oa::Vec<oa::F32> x; oa::Vec<oa::F32> y; ScatterStyle style; };
	struct BarCmd     { oa::Vec<oa::F32> x; oa::Vec<oa::F32> y; BarStyle style; oa::F32 width = 0.8F; };
	struct HeatmapCmd { oa::Vec<oa::F32> v; oa::I32 rows = 0, cols = 0; HeatmapStyle style; bool present = false; };
	enum class ArtistKind : oa::U8 { Line, Scatter, Bar };
	struct ArtistRef { ArtistKind kind = ArtistKind::Line; oa::U32 index = 0U; };
	struct LimitsCmd {
		oa::F32 xMin = 0.0F, xMax = 1.0F, yMin = 0.0F, yMax = 1.0F;
		bool present = false;
	};

	ImshowCmd  image_;
	TextCmd    title_;
	TextCmd    caption_;
	TextCmd    xLabel_;
	TextCmd    yLabel_;
	HeatmapCmd heatmap_;
	oa::Vec<LineCmd> lines_;
	oa::Vec<ScatterCmd> scatters_;
	oa::Vec<BarCmd> bars_;
	oa::Vec<ArtistRef> artists_;
	LimitsCmd limits_;
	oa::Color border_ = {0.0F, 0.0F, 0.0F, 0.0F};
	bool hasBorder_ = false;
	bool showGrid_ = true;
	bool showLegend_ = true;
};

}  // namespace oa::plot
