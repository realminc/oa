// oa::plot::Figure — top-level container for a grid of oa::plot::Axes.
//
// Three terminal sinks (architecture/oaArchitecture.md §10):
//   show()    — replays the retained figure through oa::Ui's GPU compositor.
//   saveTo()  — performs one terminal readback and encodes a fixed-size image.
//   render()  — returns the same fixed-size composition as an oa::Image.

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/core/color.h>    // oa::Color
#include <oa/ui/plot/axes.h>

namespace oa {
class Engine;
class Ui;
}

namespace oa::plot {

enum class Theme : oa::U8 {
	Dark = 0,
	Light = 1,
};

struct FigureConfig {
	// Initial interactive window name. call Figure::title to also paint an
	// in-canvas title and reserve its layout band.
	oa::String title  = "oa::plot";
	oa::I32    rows   = 1;
	oa::I32    cols   = 1;
	oa::U32    width  = 800;
	oa::U32    height = 600;
	// Spacing between axes in pixels. total figure subdivides into
	// rows × cols equal cells minus this margin on each side.
	oa::I32    hSpacing = 20;
	oa::I32    vSpacing = 24;
	// Outer padding around the entire grid.
	oa::I32    padding  = 20;
	// Realm dark is the default visual language; light is an explicit peer.
	oa::plot::Theme theme = oa::plot::Theme::Dark;
	// alpha zero selects the theme background. An opaque color overrides it.
	oa::Color background = {0.0F, 0.0F, 0.0F, 0.0F};
};

class Figure {
public:
	explicit Figure(const FigureConfig& inConfig = {});
	~Figure();
	Figure(const Figure&)            = delete;
	Figure& operator=(const Figure&) = delete;
	Figure(Figure&&) noexcept;
	Figure& operator=(Figure&&) noexcept;

	// access the (row, col) axes. rows and cols are 0-indexed.
	[[nodiscard]] Axes& ax(oa::I32 inRow, oa::I32 inCol);

	// Centered figure-level title and labels. Y labels render bottom-to-top.
	// The title also replaces the interactive window name.
	void title(const char* inText);
	void xLabel(const char* inText);
	void yLabel(const char* inText);

	// ── Sinks ─────────────────────────────────────────────────────────────

	// open an interactive window, render until closed. Returns when the
	// user closes the window or the run loop exits.
	[[nodiscard]] oa::Status show();

	// Headless GPU render -> PNG. The explicit overload completes pending
	// producers through the supplied engine; the convenience overload uses the
	// active operation context. One readback occurs only after composition.
	// output includes raster bases, ordered line/scatter/bar artists, heatmaps,
	// legends, figure/axes titles and labels, and captions.
	[[nodiscard]] oa::Status saveTo(oa::Engine& inEngine, const char* inPath);
	[[nodiscard]] oa::Status saveTo(const char* inPath);

	// Render the fixed-size figure into a semantic RGBA image. The result is a
	// normalized Float32 [1, 4, height, width] oa::Image and may be passed to
	// oa::Viewer, Vision operations, or an image encoder. This semantic-image sink
	// performs one terminal GPU readback and upload; show() remains GPU-resident.
	// Every label uses the same generated coverage data as interactive text.
	[[nodiscard]] oa::Result<oa::Image> render(oa::Engine& inEngine);
	[[nodiscard]] oa::Result<oa::Image> render();

	// ── layout query (used by impl + tutorials) ──────────────────────────

	[[nodiscard]] const FigureConfig& config() const noexcept { return config_; }
	[[nodiscard]] oa::I32 rows() const noexcept { return config_.rows; }
	[[nodiscard]] oa::I32 cols() const noexcept { return config_.cols; }

	// Compute the pixel rect of the (row, col) cell inside an output of
	// (inW × inH), after explicit figure-level title/x/y bands are reserved.
	// Used by every terminal sink so the layout is identical across them.
	struct Rect { oa::I32 x = 0; oa::I32 y = 0; oa::I32 w = 0; oa::I32 h = 0; };
	[[nodiscard]] Rect cellRect(oa::I32 inRow, oa::I32 inCol,
	                            oa::U32 inW, oa::U32 inH) const noexcept;

	// replay the complete figure directly into an active oa::Ui GPU frame. The
	// configured aspect ratio and all title/label bands remain stable on resize.
	void renderFrame(oa::U32 inWidth, oa::U32 inHeight, ::oa::Ui& inUi);

private:
	class ShowSource;
	[[nodiscard]] oa::Status prepareImageSources(oa::Engine& inEngine) const;
	[[nodiscard]] oa::Status renderRgba(
		oa::Engine& inEngine,
		oa::Vec<oa::U8>& outRgba);

	struct Impl;
	oa::UniquePtr<Impl> impl_;
	FigureConfig      config_;
};

}  // namespace oa::plot
