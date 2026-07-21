// OaPlot::Figure — top-level container for a grid of OaPlot::Axes.
//
// Two terminal sinks (Architecture/OaArchitecture.md §10):
//   Show()    — opens the OaViewer application, blocks until the
//               user closes it. Re-renders every frame from the recorded
//               commands so the same figure can be reused on resize.
//   SaveFig() — replays image, line and heatmap commands into a fixed-size PNG.
//               Text glyphs remain interactive-only.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Color.h>    // OaColor
#include <Oa/Ui/Plot/Axes.h>

class OaUi;
class OaContext;

namespace OaPlot {

struct FigureConfig {
	OaString Title  = "OaPlot";
	OaI32    Rows   = 1;
	OaI32    Cols   = 1;
	OaU32    Width  = 800;
	OaU32    Height = 600;
	// Spacing between axes in pixels. Total figure subdivides into
	// rows × cols equal cells minus this margin on each side.
	OaI32    HSpacing = 0;
	OaI32    VSpacing = 0;
	// Outer padding around the entire grid.
	OaI32    Padding  = 0;
	// matplotlib-style off-white — contrasts with dark-background images
	// (Fashion-MNIST, raw heatmaps, etc.) by default. Override per-figure.
	OaColor Background = {0.94F, 0.94F, 0.94F, 1.0F};
};

class Figure {
public:
	explicit Figure(const FigureConfig& InConfig = {});
	~Figure();
	Figure(const Figure&)            = delete;
	Figure& operator=(const Figure&) = delete;
	Figure(Figure&&) noexcept;
	Figure& operator=(Figure&&) noexcept;

	// Access the (row, col) axes. Rows and Cols are 0-indexed.
	[[nodiscard]] Axes& Ax(OaI32 InRow, OaI32 InCol);

	// Figure-level title / labels (interactive sink; headless text is pending).
	void Title(const char* InText);
	void XLabel(const char* InText);
	void YLabel(const char* InText);

	// ── Sinks ─────────────────────────────────────────────────────────────

	// Open an interactive window, render until closed. Returns when the
	// user closes the window or the run loop exits.
	[[nodiscard]] OaStatus Show();

	// Headless render → PNG. The context overload completes pending producers
	// before readback; the compatibility overload uses the matching global/default
	// context. Output includes images, line plots and heatmaps; text is omitted.
	[[nodiscard]] OaStatus SaveFig(OaContext& InContext, const char* InPath);
	[[nodiscard]] OaStatus SaveFig(const char* InPath);

	// ── Layout query (used by impl + tutorials) ──────────────────────────

	[[nodiscard]] const FigureConfig& Config() const noexcept { return Config_; }
	[[nodiscard]] OaI32 Rows() const noexcept { return Config_.Rows; }
	[[nodiscard]] OaI32 Cols() const noexcept { return Config_.Cols; }

	// Compute the pixel rect of the (row, col) cell inside an output of
	// (InW × InH). Used by both the Show and SaveFig paths so the layout
	// is bit-identical across sinks.
	struct Rect { OaI32 X = 0; OaI32 Y = 0; OaI32 W = 0; OaI32 H = 0; };
	[[nodiscard]] Rect CellRect(OaI32 InRow, OaI32 InCol,
	                            OaU32 InW, OaU32 InH) const noexcept;

	// ── Rasterize ────────────────────────────────────────────────────────
	//
	// Composite the figure into its own fixed-size canvas (one CPU paint, one
	// GPU upload). Reads each Imshow tile back from the GPU into a host
	// framebuffer at Config_.Width × Config_.Height, then uploads the result
	// as a single OaTexture. After this returns, Canvas() is valid and the
	// figure is a frozen 2D image — Show() displays it as one textured panel
	// letterboxed inside the window, so window resize never affects cell
	// proportions. Call this from your OnInit after the engine is up and
	// after axes have been populated.
	[[nodiscard]] OaStatus Rasterize(OaContext& InContext);
	void Rasterize(class OaEngine& InRt);

	// The rasterized canvas. Valid after Rasterize(). Use BindlessIndex() +
	// (Width, Height) to draw the figure as a single OaUi::Image.
	[[nodiscard]] const OaTexture& Canvas() const noexcept;

	// Per-frame replay used by the internal Show() app. If Rasterize() has
	// been called, displays the canvas as a single textured panel letterboxed
	// in the window. Otherwise replays Axes commands as OaUi widgets (legacy
	// path; cell proportions follow the window).
	void RenderFrame(OaU32 InWidth, OaU32 InHeight, ::OaUi& InOui);

private:
	// Paint the figure into Impl_->Framebuffer_ (CPU-side, fixed size =
	// Config_.Width × Config_.Height). Used by both Rasterize (which then
	// uploads to Canvas_) and SaveFig (which writes PNG directly).
	[[nodiscard]] OaStatus CompositeFramebuffer(class OaEngine& InRt);
	[[nodiscard]] OaStatus SaveFigReady(
		class OaEngine& InRt,
		const char* InPath);
	[[nodiscard]] OaStatus RasterizeReady(class OaEngine& InRt);
	[[nodiscard]] OaStatus ValidateImageSources(
		class OaEngine& InRt,
		OaBool& OutNeedsCompletion) const;

	struct Impl;
	OaUniquePtr<Impl> Impl_;
	FigureConfig      Config_;
};

}  // namespace OaPlot
