// oa::plot::Figure — retained plot commands lowered through oa::Ui's GPU compositor.
// Interactive presentation stays device-resident. render() and saveTo() cross
// to the host once, at their explicit terminal sink boundary.

// Engine first — VK_NO_PROTOTYPES before any vulkan.h pull-in.
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/executionSession.h>

#include "../../runtime/textureAccess.h"

#include <oa/ui/plot/figure.h>

#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/render/renderer.h>
#include <oa/ui/renderConfig.h>
#include <oa/ui/style.h>
#include <oa/ui/ui.h>
#include <oa/ui/viewer.h>
#include <oa/vision/fnImage.h>

#include <stdio.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/scalarMath.h>
#include <oa/core/std/cString.h>
#include <oa/core/std/limits.h>

namespace oa::plot {

// ─── Figure::Impl ──────────────────────────────────────────────────────────

struct Figure::Impl {
	oa::Vector<Axes> axes_;
	oa::String    title_;
	oa::String    xLabel_;
	oa::String    yLabel_;
};


// ─── ctor / dtor / move ────────────────────────────────────────────────────

Figure::Figure(const FigureConfig& inConfig)
	: impl_(oa::makeUnique<Impl>())
	, config_(inConfig)
{
	if (config_.rows < 1) config_.rows = 1;
	if (config_.cols < 1) config_.cols = 1;
	config_.hSpacing = oa::max(0, config_.hSpacing);
	config_.vSpacing = oa::max(0, config_.vSpacing);
	config_.padding = oa::max(0, config_.padding);
	if (config_.background.a <= 0.0F) {
		config_.background = config_.theme == Theme::Light
			? oa::UiStyle::realmLight().background
			: oa::UiStyle::realmDark().background;
	}
	impl_->axes_.resize(
		static_cast<oa::Usize>(config_.rows)
		* static_cast<oa::Usize>(config_.cols));
}

Figure::~Figure() = default;
Figure::Figure(Figure&&) noexcept = default;
Figure& Figure::operator=(Figure&&) noexcept = default;


// ─── layout ────────────────────────────────────────────────────────────────

Axes& Figure::ax(oa::I32 inRow, oa::I32 inCol) {
	oa::I32 r = inRow < 0 ? 0 : (inRow >= config_.rows ? config_.rows - 1 : inRow);
	oa::I32 c = inCol < 0 ? 0 : (inCol >= config_.cols ? config_.cols - 1 : inCol);
	return impl_->axes_[
		static_cast<oa::Usize>(r) * static_cast<oa::Usize>(config_.cols)
		+ static_cast<oa::Usize>(c)];
}

void Figure::title(const char* inText)  { impl_->title_  = inText ? inText : ""; }
void Figure::xLabel(const char* inText) { impl_->xLabel_ = inText ? inText : ""; }
void Figure::yLabel(const char* inText) { impl_->yLabel_ = inText ? inText : ""; }

namespace {

static constexpr oa::I32 kTitleH          = 22;
static constexpr oa::I32 kCaptionH        = 20;
static constexpr oa::I32 kTitleGap        = 6;
static constexpr oa::I32 kAxesXLabelH     = 28;
static constexpr oa::I32 kAxesYLabelW     = 28;
static constexpr oa::I32 kFigureTitleH    = 38;
static constexpr oa::I32 kFigureXLabelH   = 30;
static constexpr oa::I32 kFigureYLabelW   = 30;

struct FigureFrame {
	oa::I32 x = 0;
	oa::I32 y = 0;
	oa::I32 w = 0;
	oa::I32 h = 0;
	oa::F32 scale = 0.0F;
};

FigureFrame resolveFigureFrame(
	const FigureConfig& inConfig,
	oa::U32 inW,
	oa::U32 inH) noexcept {
	if (inW == 0U or inH == 0U
		or inConfig.width == 0U or inConfig.height == 0U) {
		return {};
	}
	const oa::F32 figAspect = static_cast<oa::F32>(inConfig.width)
		/ static_cast<oa::F32>(inConfig.height);
	const oa::F32 winAspect = static_cast<oa::F32>(inW)
		/ static_cast<oa::F32>(inH);
	FigureFrame frame;
	if (winAspect > figAspect) {
		frame.h = static_cast<oa::I32>(inH);
		frame.w = static_cast<oa::I32>(static_cast<oa::F32>(inH) * figAspect);
	} else {
		frame.w = static_cast<oa::I32>(inW);
		frame.h = static_cast<oa::I32>(static_cast<oa::F32>(inW) / figAspect);
	}
	frame.x = (static_cast<oa::I32>(inW) - frame.w) / 2;
	frame.y = (static_cast<oa::I32>(inH) - frame.h) / 2;
	frame.scale = static_cast<oa::F32>(frame.w)
		/ static_cast<oa::F32>(inConfig.width);
	return frame;
}

oa::I32 scaledLayoutValue(oa::I32 inValue, oa::F32 inScale) noexcept {
	return static_cast<oa::I32>(static_cast<oa::F32>(inValue) * inScale);
}

struct CellLayout {
	Figure::Rect title;
	Figure::Rect yLabel;
	Figure::Rect plot;
	Figure::Rect xLabel;
	Figure::Rect caption;
};

CellLayout resolveCellLayout(
	Figure::Rect inCell,
	bool inHasTitle,
	bool inHasXLabel,
	bool inHasYLabel,
	bool inHasCaption,
	oa::F32 inScale) noexcept {
	const oa::I32 titleHeight = scaledLayoutValue(kTitleH, inScale);
	const oa::I32 titleGap = scaledLayoutValue(kTitleGap, inScale);
	const oa::I32 xLabelHeight = scaledLayoutValue(kAxesXLabelH, inScale);
	const oa::I32 yLabelWidth = scaledLayoutValue(kAxesYLabelW, inScale);
	const oa::I32 captionHeight = scaledLayoutValue(kCaptionH, inScale);
	const oa::I32 titleBand = inHasTitle ? titleHeight + titleGap : 0;
	const oa::I32 xLabelBand = inHasXLabel ? xLabelHeight : 0;
	const oa::I32 yLabelBand = inHasYLabel ? yLabelWidth : 0;
	const oa::I32 captionBand = inHasCaption ? captionHeight + titleGap : 0;
	CellLayout layout;
	layout.title = {inCell.x + yLabelBand, inCell.y,
		oa::max(0, inCell.w - 2 * yLabelBand), inHasTitle ? titleHeight : 0};
	layout.yLabel = {inCell.x, inCell.y + titleBand,
		inHasYLabel ? yLabelWidth : 0,
		oa::max(0, inCell.h - titleBand - xLabelBand - captionBand)};
	layout.plot = {inCell.x + yLabelBand, inCell.y + titleBand,
		oa::max(0, inCell.w - 2 * yLabelBand),
		oa::max(0, inCell.h - titleBand - xLabelBand - captionBand)};
	layout.xLabel = {layout.plot.x, layout.plot.y + layout.plot.h,
		layout.plot.w, inHasXLabel ? xLabelHeight : 0};
	layout.caption = {layout.plot.x,
		inCell.y + inCell.h - (inHasCaption ? captionHeight : 0),
		layout.plot.w, inHasCaption ? captionHeight : 0};
	return layout;
}

oa::UiStyle resolveFigureStyle(const FigureConfig& inConfig) {
	oa::UiStyle style = inConfig.theme == Theme::Light
		? oa::UiStyle::realmLight() : oa::UiStyle::realmDark();
	style.background = inConfig.background;
	return style;
}

oa::Color resolveOptionalColor(oa::Color inColor, oa::Color inFallback) {
	return inColor.a <= 0.0F ? inFallback : inColor;
}

struct DataBounds {
	oa::F32 xMin = oa::Limits<oa::F32>::infinity();
	oa::F32 xMax = -oa::Limits<oa::F32>::infinity();
	oa::F32 yMin = oa::Limits<oa::F32>::infinity();
	oa::F32 yMax = -oa::Limits<oa::F32>::infinity();

	void include(oa::F32 inX, oa::F32 inY) noexcept {
		if (not oa::isFinite(inX) or not oa::isFinite(inY)) return;
		xMin = oa::min(xMin, inX);
		xMax = oa::max(xMax, inX);
		yMin = oa::min(yMin, inY);
		yMax = oa::max(yMax, inY);
	}
	[[nodiscard]] bool isValid() const noexcept {
		return oa::isFinite(xMin) and oa::isFinite(xMax)
			and oa::isFinite(yMin) and oa::isFinite(yMax);
	}
	void expandDegenerate() noexcept {
		if (xMax <= xMin) {
			const oa::F32 margin = oa::max(1.0e-4F, oa::abs(xMin) * 0.05F);
			xMin -= margin;
			xMax += margin;
		}
		if (yMax <= yMin) {
			const oa::F32 margin = oa::max(1.0e-4F, oa::abs(yMin) * 0.05F);
			yMin -= margin;
			yMax += margin;
		}
	}
};

oa::Color seriesColor(const oa::UiStyle& inStyle, oa::U32 inIndex) {
	switch (inIndex % 8U) {
		case 0U: return inStyle.accent;
		case 1U: return inStyle.success;
		case 2U: return inStyle.warning;
		case 3U: return oa::Color::cyan();
		case 4U: return oa::Color::purple();
		case 5U: return oa::Color::pink();
		case 6U: return oa::Color::orange();
		default: return inStyle.error;
	}
}

} // namespace

Figure::Rect Figure::cellRect(oa::I32 inRow, oa::I32 inCol,
                              oa::U32 inW, oa::U32 inH) const noexcept {
	// Letterbox the figure inside (inW, inH) to preserve config_.width:height
	// aspect. matches matplotlib's "figure has a fixed canvas; window shows
	// it centred" behaviour — cells grow uniformly with the window without
	// the inter-cell gaps shrinking in proportion.
	const FigureFrame frame = resolveFigureFrame(config_, inW, inH);

	// pad / spacing scale with the figure render size so the proportions
	// the user configured are preserved across resizes.
	const oa::I32 pad = scaledLayoutValue(config_.padding, frame.scale);
	const oa::I32 hgap = scaledLayoutValue(config_.hSpacing, frame.scale);
	const oa::I32 vgap = scaledLayoutValue(config_.vSpacing, frame.scale);
	const oa::I32 titleH = impl_->title_.empty()
		? 0 : scaledLayoutValue(kFigureTitleH, frame.scale);
	const oa::I32 xLabelH = impl_->xLabel_.empty()
		? 0 : scaledLayoutValue(kFigureXLabelH, frame.scale);
	const oa::I32 yLabelW = impl_->yLabel_.empty()
		? 0 : scaledLayoutValue(kFigureYLabelW, frame.scale);
	const oa::I32 gridW = oa::max(0, frame.w - 2 * pad - 2 * yLabelW);
	const oa::I32 gridH = oa::max(0, frame.h - 2 * pad - titleH - xLabelH);
	const oa::I32 cellW = oa::max(
		0, (gridW - (config_.cols - 1) * hgap) / config_.cols);
	const oa::I32 cellH = oa::max(
		0, (gridH - (config_.rows - 1) * vgap) / config_.rows);
	return {
		.x = frame.x + pad + yLabelW + inCol * (cellW + hgap),
		.y = frame.y + pad + titleH + inRow * (cellH + vgap),
		.w = cellW,
		.h = cellH,
	};
}


// ─── GPU figure replay ───────────────────────────────────────────────────

void Figure::renderFrame(oa::U32 inWidth, oa::U32 inHeight, ::oa::Ui& inUi) {
	if (inWidth == 0U or inHeight == 0U) return;
	const FigureFrame frame = resolveFigureFrame(config_, inWidth, inHeight);
	if (frame.w <= 0 or frame.h <= 0 or frame.scale <= 0.0F) return;
	const oa::UiStyle& style = inUi.currentStyle();
	inUi.rect({frame.x, frame.y, frame.w, frame.h}, config_.background);

	const oa::I32 pad = scaledLayoutValue(config_.padding, frame.scale);
	const oa::I32 titleHeight = impl_->title_.empty()
		? 0 : scaledLayoutValue(kFigureTitleH, frame.scale);
	const oa::I32 xLabelHeight = impl_->xLabel_.empty()
		? 0 : scaledLayoutValue(kFigureXLabelH, frame.scale);
	const oa::I32 yLabelWidth = impl_->yLabel_.empty()
		? 0 : scaledLayoutValue(kFigureYLabelW, frame.scale);
	const Rect figureTitle{
		frame.x + pad, frame.y + pad,
		oa::max(0, frame.w - 2 * pad), titleHeight};
	const Rect figureYLabel{
		frame.x + pad, frame.y + pad + titleHeight, yLabelWidth,
		oa::max(0, frame.h - 2 * pad - titleHeight - xLabelHeight)};
	const Rect figureXLabel{
		frame.x + pad + yLabelWidth,
		frame.y + frame.h - pad - xLabelHeight,
		oa::max(0, frame.w - 2 * pad - 2 * yLabelWidth), xLabelHeight};
	const auto drawText = [&](oa::StringView inText, Rect inRect,
		oa::F32 inSize, oa::Color inColor,
		oa::UiTextDirection inDirection = oa::UiTextDirection::LeftToRight,
		oa::UiAlign inHorizontal = oa::UiAlign::Center) {
		if (inText.empty() or inRect.w <= 0 or inRect.h <= 0) return;
		inUi.textAt(inText, {inRect.x, inRect.y, inRect.w, inRect.h}, {
			.fontSize = oa::max(1.0F, inSize * frame.scale),
			.color = inColor,
			.horizontalAlign = inHorizontal,
			.verticalAlign = oa::UiAlign::Center,
			.direction = inDirection,
		});
	};
	drawText(impl_->title_, figureTitle, 18.0F,
		style.text);
	drawText(impl_->yLabel_, figureYLabel, 14.0F,
		style.textSecondary,
		oa::UiTextDirection::BottomToTop);
	drawText(impl_->xLabel_, figureXLabel, 14.0F,
		style.textSecondary);

	oa::I32 fixedImageSize = oa::Limits<oa::I32>::max();
	bool hasSquareImage = false;
	for (oa::I32 row = 0; row < rows(); ++row) {
		for (oa::I32 column = 0; column < cols(); ++column) {
			const Axes& axes = ax(row, column);
			if (not axes.image_.present or not axes.image_.tex.isValid()
				or axes.image_.tex.width() != axes.image_.tex.height()) continue;
			const CellLayout layout = resolveCellLayout(
				cellRect(row, column, inWidth, inHeight),
				axes.title_.present and not axes.title_.text.empty(),
				axes.xLabel_.present and not axes.xLabel_.text.empty(),
				axes.yLabel_.present and not axes.yLabel_.text.empty(),
				axes.caption_.present and not axes.caption_.text.empty(),
				frame.scale);
			const oa::I32 candidate = oa::min(layout.plot.w, layout.plot.h);
			if (candidate <= 0) continue;
			fixedImageSize = oa::min(fixedImageSize, candidate);
			hasSquareImage = true;
		}
	}

	for (oa::I32 row = 0; row < rows(); ++row) {
		for (oa::I32 column = 0; column < cols(); ++column) {
			const Axes& axes = ax(row, column);
			const bool hasTitle =
				axes.title_.present and not axes.title_.text.empty();
			const bool hasXLabel =
				axes.xLabel_.present and not axes.xLabel_.text.empty();
			const bool hasYLabel =
				axes.yLabel_.present and not axes.yLabel_.text.empty();
			const bool hasCaption =
				axes.caption_.present and not axes.caption_.text.empty();
			const CellLayout layout = resolveCellLayout(
				cellRect(row, column, inWidth, inHeight),
				hasTitle, hasXLabel, hasYLabel, hasCaption, frame.scale);
			drawText(axes.title_.text, layout.title, 14.0F,
				resolveOptionalColor(axes.title_.color, style.text));
			drawText(axes.yLabel_.text, layout.yLabel, 12.0F,
				resolveOptionalColor(axes.yLabel_.color, style.textSecondary),
				oa::UiTextDirection::BottomToTop);
			drawText(axes.xLabel_.text, layout.xLabel, 12.0F,
				resolveOptionalColor(axes.xLabel_.color, style.textSecondary));
			drawText(axes.caption_.text, layout.caption, 12.0F,
				resolveOptionalColor(axes.caption_.color, style.textMuted),
				oa::UiTextDirection::LeftToRight,
				oa::UiAlign::Start);
			if (layout.plot.w <= 0 or layout.plot.h <= 0) continue;

			char id[64];
			inUi.grid(
				{layout.plot.x, layout.plot.y, layout.plot.w, layout.plot.h}, {
					.origin = {
						static_cast<oa::F32>(layout.plot.x),
						static_cast<oa::F32>(layout.plot.y + layout.plot.h - 1),
					},
					.minorSpacing = {
						oa::max(4.0F, static_cast<oa::F32>(layout.plot.w - 1) / 100.0F),
						oa::max(4.0F, static_cast<oa::F32>(layout.plot.h - 1) / 100.0F),
					},
					.drawGrid = false,
					.drawAxes = false,
				});
			if (axes.heatmap_.present and not axes.heatmap_.v.empty()) {
				oa::F32 minimum = axes.heatmap_.style.vMin;
				oa::F32 maximum = axes.heatmap_.style.vMax;
				if (axes.heatmap_.style.autoScale) {
					minimum = oa::Limits<oa::F32>::infinity();
					maximum = -oa::Limits<oa::F32>::infinity();
					for (oa::F32 value : axes.heatmap_.v) {
						if (not oa::isFinite(value)) continue;
						minimum = oa::min(minimum, value);
						maximum = oa::max(maximum, value);
					}
				}
				if (oa::isFinite(minimum) and oa::isFinite(maximum)) {
					if (maximum <= minimum) maximum = minimum + 1.0F;
					::snprintf(id, sizeof(id), "figure-heat-%d-%d", row, column);
					inUi.beginPanel(id, {layout.plot.x, layout.plot.y,
						layout.plot.w, layout.plot.h});
					inUi.heatmap(id, axes.heatmap_.v.data(),
						axes.heatmap_.rows, axes.heatmap_.cols, {
							.vMin = minimum,
							.vMax = maximum,
							.colormap = axes.heatmap_.style.colormap,
							.showGrid = axes.heatmap_.style.showGrid,
						});
					inUi.endPanel();
				}
			} else if (axes.image_.present and axes.image_.tex.isValid()) {
				oa::I32 destinationWidth = layout.plot.w;
				oa::I32 destinationHeight = layout.plot.h;
				if (axes.image_.tex.width() == axes.image_.tex.height()
					and hasSquareImage) {
					destinationWidth = fixedImageSize;
					destinationHeight = fixedImageSize;
				} else {
					const oa::F32 aspect =
						static_cast<oa::F32>(axes.image_.tex.width())
						/ static_cast<oa::F32>(axes.image_.tex.height());
					destinationHeight = static_cast<oa::I32>(
						static_cast<oa::F32>(destinationWidth) / aspect);
					if (destinationHeight > layout.plot.h) {
						destinationHeight = layout.plot.h;
						destinationWidth = static_cast<oa::I32>(
							static_cast<oa::F32>(destinationHeight) * aspect);
					}
				}
				const oa::I32 destinationX = layout.plot.x
					+ (layout.plot.w - destinationWidth) / 2;
				const oa::I32 destinationY = layout.plot.y
					+ (layout.plot.h - destinationHeight) / 2;
				::snprintf(id, sizeof(id), "figure-image-%d-%d", row, column);
				inUi.beginPanel(id, {destinationX, destinationY,
					destinationWidth, destinationHeight});
				inUi.image(axes.image_.tex);
				inUi.endPanel();
			}

			if (not axes.artists_.empty()) {
				DataBounds bounds;
				if (axes.limits_.present) {
					bounds = {axes.limits_.xMin, axes.limits_.xMax,
						axes.limits_.yMin, axes.limits_.yMax};
				} else {
					for (const Axes::ArtistRef& artist : axes.artists_) {
						switch (artist.kind) {
							case Axes::ArtistKind::Line: {
								if (artist.index >= axes.lines_.size()) break;
								const auto& line = axes.lines_[artist.index];
								for (oa::Usize i = 0; i < line.y.size(); ++i) {
									const oa::F32 x = line.x.empty()
										? static_cast<oa::F32>(i) : line.x[i];
									bounds.include(x, line.y[i]);
								}
								break;
							}
							case Axes::ArtistKind::Scatter: {
								if (artist.index >= axes.scatters_.size()) break;
								const auto& scatter = axes.scatters_[artist.index];
								for (oa::Usize i = 0; i < scatter.y.size(); ++i) {
									bounds.include(scatter.x[i], scatter.y[i]);
								}
								break;
							}
							case Axes::ArtistKind::Bar: {
								if (artist.index >= axes.bars_.size()) break;
								const auto& bar = axes.bars_[artist.index];
								for (oa::Usize i = 0; i < bar.y.size(); ++i) {
									bounds.include(bar.x[i] - bar.width * 0.5F, 0.0F);
									bounds.include(bar.x[i] + bar.width * 0.5F, bar.y[i]);
								}
								break;
							}
						}
					}
					if (bounds.isValid()) bounds.expandDegenerate();
				}

				if (bounds.isValid() and bounds.xMax > bounds.xMin
					and bounds.yMax > bounds.yMin) {
					const auto mapX = [&](oa::F32 inX) {
						return static_cast<oa::F32>(layout.plot.x)
							+ (inX - bounds.xMin) / (bounds.xMax - bounds.xMin)
								* static_cast<oa::F32>(oa::max(0, layout.plot.w - 1));
					};
					const auto mapY = [&](oa::F32 inY) {
						return static_cast<oa::F32>(layout.plot.y)
							+ (1.0F - (inY - bounds.yMin)
								/ (bounds.yMax - bounds.yMin))
								* static_cast<oa::F32>(oa::max(0, layout.plot.h - 1));
					};
					::snprintf(id, sizeof(id), "figure-artists-%d-%d", row, column);
					inUi.beginPanel(id, {layout.plot.x, layout.plot.y,
						layout.plot.w, layout.plot.h});
					if (axes.showGrid_) {
						// Keep four to six square 100-unit cells on the shorter
						// dimension at any figure size or DPI. One pixel spacing
						// drives both axes, like a DCC UV grid; the longer dimension
						// simply exposes more cells. Both black reference axes stay
						// anchored to the visual center instead of data-space zero.
						const oa::F32 shortSide = static_cast<oa::F32>(oa::max(
							1, oa::min(layout.plot.w - 1, layout.plot.h - 1)));
						const oa::F32 targetMajorPixels = oa::max(1.0F,
							144.0F * frame.scale);
						const oa::I32 shortSideCells = oa::clamp(
							static_cast<oa::I32>(oa::round(
								shortSide / targetMajorPixels)), 4, 6);
						const oa::F32 minorSpacing = oa::max(1.0F,
							shortSide / static_cast<oa::F32>(shortSideCells * 10));
						inUi.grid(
							{layout.plot.x, layout.plot.y,
								layout.plot.w, layout.plot.h}, {
								.origin = {
									static_cast<oa::F32>(layout.plot.x)
										+ 0.5F * static_cast<oa::F32>(layout.plot.w - 1),
									static_cast<oa::F32>(layout.plot.y)
										+ 0.5F * static_cast<oa::F32>(layout.plot.h - 1),
								},
								.minorSpacing = {
									minorSpacing,
									minorSpacing,
								},
								.opacity = 0.65F,
								.fillBackground = false,
								.drawGrid = true,
								.drawAxes = true,
							});
					}

					oa::U32 seriesIndex = 0U;
					for (const Axes::ArtistRef& artist : axes.artists_) {
						const oa::Color palette = seriesColor(style, seriesIndex++);
						switch (artist.kind) {
							case Axes::ArtistKind::Line: {
								if (artist.index >= axes.lines_.size()) break;
								const auto& line = axes.lines_[artist.index];
								const oa::Color color = resolveOptionalColor(
									line.style.color, palette);
								if (not line.x.empty()) {
									::snprintf(id, sizeof(id),
										"figure-line-xy-%d-%d-%u", row, column, artist.index);
									inUi.plotLineXY(id, line.x.data(), line.y.data(),
										static_cast<oa::I32>(line.y.size()), {
											.color = color,
											.xMin = bounds.xMin,
											.xMax = bounds.xMax,
											.yMin = bounds.yMin,
											.yMax = bounds.yMax,
											.autoScale = false,
											.showGrid = false,
											.antialiasSamples = line.style.antialiasSamples,
											.lineWidth = line.style.width * frame.scale,
											.drawSurface = false,
										});
									break;
								}
								const oa::F32 implicitMaximum = line.y.empty()
									? 0.0F : static_cast<oa::F32>(line.y.size() - 1U);
								const bool direct = line.x.empty()
									and oa::abs(bounds.xMin) <= 1.0e-5F
									and oa::abs(bounds.xMax - implicitMaximum) <= 1.0e-5F;
								if (direct) {
									::snprintf(id, sizeof(id),
										"figure-line-%d-%d-%u", row, column, artist.index);
									inUi.plotLine(id, line.y.data(),
										static_cast<oa::I32>(line.y.size()), {
											.color = color,
											.yMin = bounds.yMin,
											.yMax = bounds.yMax,
											.autoScale = false,
											.showGrid = false,
											.antialiasSamples = line.style.antialiasSamples,
											.lineWidth = line.style.width * frame.scale,
											.drawSurface = false,
										});
									break;
								}
								if (line.y.size() == 1U) {
									const oa::F32 x = 0.0F;
									if (oa::isFinite(x) and oa::isFinite(line.y[0])) {
										inUi.rect({static_cast<oa::I32>(oa::round(mapX(x))) - 2,
											static_cast<oa::I32>(oa::round(mapY(line.y[0]))) - 2,
											5, 5}, color);
									}
									break;
								}
								for (oa::Usize i = 1U; i < line.y.size(); ++i) {
									const oa::F32 x0 = static_cast<oa::F32>(i - 1U);
									const oa::F32 x1 = static_cast<oa::F32>(i);
									if (not oa::isFinite(x0) or not oa::isFinite(x1)
										or not oa::isFinite(line.y[i - 1U])
										or not oa::isFinite(line.y[i])) continue;
									inUi.line({mapX(x0), mapY(line.y[i - 1U])},
										{mapX(x1), mapY(line.y[i])}, color,
										oa::clamp(line.style.width * frame.scale, 0.5F, 16.0F));
								}
								break;
							}
							case Axes::ArtistKind::Scatter: {
								if (artist.index >= axes.scatters_.size()) break;
								const auto& scatter = axes.scatters_[artist.index];
								const oa::Color color = resolveOptionalColor(
									scatter.style.color, palette);
								const oa::I32 radius = static_cast<oa::I32>(oa::round(
									oa::clamp(scatter.style.radius * frame.scale, 1.0F, 12.0F)));
								for (oa::Usize i = 0U; i < scatter.y.size(); ++i) {
									if (not oa::isFinite(scatter.x[i])
										or not oa::isFinite(scatter.y[i])) continue;
									inUi.rect({
										static_cast<oa::I32>(oa::round(mapX(scatter.x[i]))) - radius,
										static_cast<oa::I32>(oa::round(mapY(scatter.y[i]))) - radius,
										2 * radius + 1, 2 * radius + 1}, color);
								}
								break;
							}
							case Axes::ArtistKind::Bar: {
								if (artist.index >= axes.bars_.size()) break;
								const auto& bar = axes.bars_[artist.index];
								const oa::Color color = resolveOptionalColor(
									bar.style.color, palette).withAlpha(1.0F);
								for (oa::Usize i = 0U; i < bar.y.size(); ++i) {
									if (not oa::isFinite(bar.x[i])
										or not oa::isFinite(bar.y[i])) continue;
									const oa::F32 left = mapX(bar.x[i] - bar.width * 0.5F);
									const oa::F32 right = mapX(bar.x[i] + bar.width * 0.5F);
									const oa::F32 baseline = mapY(0.0F);
									const oa::F32 value = mapY(bar.y[i]);
									const oa::I32 x = static_cast<oa::I32>(oa::floor(oa::min(left, right)));
									const oa::I32 y = static_cast<oa::I32>(oa::floor(oa::min(baseline, value)));
									const oa::I32 w = oa::max(1, static_cast<oa::I32>(oa::ceil(oa::abs(right - left))));
									const oa::I32 h = oa::max(1, static_cast<oa::I32>(oa::ceil(oa::abs(value - baseline))));
									inUi.rect({x, y, w, h}, color);
								}
								break;
							}
						}
					}

					if (axes.showLegend_) {
						oa::U32 labeled = 0U;
						for (const Axes::ArtistRef& artist : axes.artists_) {
							if (artist.kind == Axes::ArtistKind::Line
								and artist.index < axes.lines_.size()
								and not axes.lines_[artist.index].style.label.empty()) ++labeled;
							if (artist.kind == Axes::ArtistKind::Scatter
								and artist.index < axes.scatters_.size()
								and not axes.scatters_[artist.index].style.label.empty()) ++labeled;
							if (artist.kind == Axes::ArtistKind::Bar
								and artist.index < axes.bars_.size()
								and not axes.bars_[artist.index].style.label.empty()) ++labeled;
						}
						if (labeled > 0U) {
							const oa::I32 rowHeight = oa::max(14, scaledLayoutValue(18, frame.scale));
							const oa::I32 legendWidth = oa::min(layout.plot.w - 8,
								oa::max(96, scaledLayoutValue(150, frame.scale)));
							const oa::I32 legendHeight = oa::min(layout.plot.h - 8,
								static_cast<oa::I32>(labeled) * rowHeight + 8);
							const oa::I32 legendX = layout.plot.x + layout.plot.w - legendWidth - 6;
							const oa::I32 legendY = layout.plot.y + 6;
							const oa::F32 legendRadius = oa::max(
								2.0F, 6.0F * frame.scale);
							inUi.rect({legendX, legendY, legendWidth, legendHeight},
								style.background.withAlpha(0.90F), legendRadius);
							inUi.rectOutline({legendX, legendY, legendWidth, legendHeight},
								style.borderStrong, 1U, legendRadius);
							oa::U32 labelIndex = 0U;
							oa::U32 paletteIndex = 0U;
							for (const Axes::ArtistRef& artist : axes.artists_) {
								oa::StringView label;
								oa::Color color = seriesColor(style, paletteIndex++);
								if (artist.kind == Axes::ArtistKind::Line
									and artist.index < axes.lines_.size()) {
									const auto& line = axes.lines_[artist.index];
									label = oa::StringView(line.style.label.data(), line.style.label.size());
									color = resolveOptionalColor(line.style.color, color);
								} else if (artist.kind == Axes::ArtistKind::Scatter
									and artist.index < axes.scatters_.size()) {
									const auto& scatter = axes.scatters_[artist.index];
									label = oa::StringView(scatter.style.label.data(), scatter.style.label.size());
									color = resolveOptionalColor(scatter.style.color, color);
								} else if (artist.kind == Axes::ArtistKind::Bar
									and artist.index < axes.bars_.size()) {
									const auto& bar = axes.bars_[artist.index];
									label = oa::StringView(bar.style.label.data(), bar.style.label.size());
									color = resolveOptionalColor(bar.style.color, color);
								}
								if (label.empty()) continue;
								const oa::I32 centerY = legendY + 4
									+ static_cast<oa::I32>(labelIndex) * rowHeight + rowHeight / 2;
								inUi.line({static_cast<oa::F32>(legendX + 8), static_cast<oa::F32>(centerY)},
									{static_cast<oa::F32>(legendX + 28), static_cast<oa::F32>(centerY)},
									color, 3.0F);
								inUi.textAt(label,
									{legendX + 34, legendY + 4 + static_cast<oa::I32>(labelIndex) * rowHeight,
										legendWidth - 40, rowHeight}, {
										.fontSize = oa::max(9.0F, 11.0F * frame.scale),
										.color = style.textSecondary,
										.horizontalAlign = oa::UiAlign::Start,
										.verticalAlign = oa::UiAlign::Center,
									});
								++labelIndex;
							}
						}
					}
					inUi.endPanel();
				}
			}
			if (axes.hasBorder_) {
				inUi.rectOutline(
					{layout.plot.x, layout.plot.y,
						layout.plot.w, layout.plot.h},
					axes.border_, 1U);
			}
		}
	}
}


// ─── show() — oa::Viewer live source ─────────────────────────────────────────

class Figure::ShowSource final : public oa::ViewerLiveSource {
public:
	Figure* fig = nullptr;
	[[nodiscard]] oa::ViewerLiveCapabilities capabilities() const noexcept override {
		return {};
	}

	oa::Status open(oa::Engine& inEngine) override {
		if (fig == nullptr) {
			return oa::Status::invalidArgument(
				"oa::plot::Figure::show: missing figure source");
		}
		return fig->prepareImageSources(inEngine);
	}
	oa::Status init(oa::InputSystem&, oa::Fn<void(bool)>) override {
		return fig != nullptr
			? oa::Status::ok()
			: oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::plot::Figure::show source is not open");
	}
	oa::Status update(oa::F32 inDeltaMs) override {
		return oa::isFinite(inDeltaMs) && inDeltaMs >= 0.0F
			? oa::Status::ok()
			: oa::Status::invalidArgument(
				"oa::plot::Figure::show requires a finite non-negative delta");
	}

	oa::Status render(
		oa::Ui& inUi,
		const oa::TextAtlas&,
		oa::U32 inWidth,
		oa::U32 inHeight) override {
		if (fig == nullptr) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::plot::Figure::show source is not open");
		}
		fig->renderFrame(inWidth, inHeight, inUi);
		return oa::Status::ok();
	}
	oa::Status close() override {
		fig = nullptr;
		return oa::Status::ok();
	}
};


oa::Status Figure::show() {
	ShowSource source;
	source.fig = this;

	const oa::String& title = impl_->title_.empty() ? config_.title : impl_->title_;
	oa::ViewerConfig config;
	config.mode = oa::ViewerMode::Live;
	config.liveSource = &source;
	config.title = title;
	config.width = config_.width;
	config.height = config_.height;
	config.showHelp = false;
	config.showStats = false;
	config.showTimeline = false;
	config.style = resolveFigureStyle(config_);
	oa::Viewer viewer(config);
	if (auto* context = oa::ExecutionSession::getActivePtr();
		context != nullptr and context->engine().hasGraphics()) {
		return viewer.run(context->engine());
	}
	return viewer.run();
}


oa::Status Figure::prepareImageSources(oa::Engine& inEngine) const {
	bool needsCompletion = false;
	for (const Axes& axes : impl_->axes_) {
		if (not axes.image_.present) {
			continue;
		}
		const oa::Texture& texture = axes.image_.tex;
		const oavk::Buffer* buffer = oa::TextureAccess::buffer(texture);
		const oa::U64 bytes = texture.width() > 0 and texture.height() > 0
			? static_cast<oa::U64>(texture.width())
				* static_cast<oa::U64>(texture.height()) * 4U
			: 0U;
		if (texture.isImageBacked() or buffer == nullptr
			or buffer->buffer == nullptr or bytes == 0U or buffer->size < bytes
			or oa::TextureAccess::engine(texture) != &inEngine
			or buffer->allocation == nullptr or buffer->aliasIdentity != nullptr
			or buffer->allocatorIdentity
				!= oa::EngineAllocatorAccess::get(inEngine).allocator) {
			return oa::Status::invalidArgument(
				"oa::plot::Figure imshow requires a valid non-aliased buffer texture owned by its engine");
		}
		needsCompletion = true;
	}
	if (needsCompletion) {
		return oa::ExecutionSession::forEngine(inEngine).submitAndWait();
	}
	return oa::Status::ok();
}

oa::Status Figure::renderRgba(
	oa::Engine& inEngine,
	oa::Vector<oa::U8>& outRgba) {
	if (config_.width == 0U or config_.height == 0U
		or config_.width > static_cast<oa::U32>(oa::Limits<oa::I32>::max())
		or config_.height > static_cast<oa::U32>(oa::Limits<oa::I32>::max())) {
		return oa::Status::invalidArgument(
			"oa::plot::Figure requires a non-zero signed-coordinate extent");
	}
	OA_RETURN_IF_ERROR(prepareImageSources(inEngine));
	oa::UiRenderConfig config;
	config.width_ = config_.width;
	config.height_ = config_.height;
	config.targetSlotCount_ = 1U;
	config.style_ = resolveFigureStyle(config_);
	auto created = oa::Renderer::create(inEngine, config);
	if (not created.isOk()) return created.getStatus();
	auto renderer = oa::move(*created);
	const auto closeWith = [&](const oa::Status& inStatus) {
		const oa::Status closeStatus = renderer->close();
		return inStatus.isOk() ? closeStatus : inStatus;
	};
	oa::Status status = renderer->beginFrame(0.0F);
	if (not status.isOk()) return closeWith(status);
	renderFrame(config_.width, config_.height, *renderer->ui());
	auto submitted = renderer->submitFrame();
	if (not submitted.isOk()) return closeWith(submitted.getStatus());
	auto readback = renderer->consumeReadback(*submitted);
	if (not readback.isOk()) return closeWith(readback.getStatus());
	outRgba.resize(readback->colorRgba8_.size());
	oa::memcpy(
		outRgba.data(), readback->colorRgba8_.data(),
		readback->colorRgba8_.size());
	return closeWith(oa::Status::ok());
}

oa::Status Figure::saveTo(oa::Engine& inEngine, const char* inPath) {
	if (inPath == nullptr or inPath[0] == '\0') {
		return oa::Status::invalidArgument(
			"oa::plot::Figure::saveTo requires a non-empty path");
	}
	oa::Vector<oa::U8> rgba;
	OA_RETURN_IF_ERROR(renderRgba(inEngine, rgba));
	OA_RETURN_IF_ERROR(oa::FnImage::saveRgbaFile(
		oa::Span<const oa::U8>(rgba.data(), rgba.size()),
		config_.width, config_.height, inPath));
	OaLogInfo(
		oa::LogComponent::Plot,
		"oa::plot::Figure::saveTo: {}x{} -> {}",
		config_.width, config_.height, inPath);
	return oa::Status::ok();
}

oa::Status Figure::saveTo(const char* inPath) {
	auto* context = oa::ExecutionSession::getActivePtr();
	if (context == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::plot::Figure::saveTo has no active engine context");
	}
	return saveTo(context->engine(), inPath);
}

oa::Result<oa::Image> Figure::render(oa::Engine& inEngine) {
	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	oa::Vector<oa::U8> rgba;
	OA_RETURN_IF_ERROR(renderRgba(inEngine, rgba));
	const oa::U64 pixelCount = static_cast<oa::U64>(config_.width)
		* static_cast<oa::U64>(config_.height);
	if (pixelCount == 0U
		or pixelCount > oa::Limits<oa::Usize>::max() / 4U
		or rgba.size() != static_cast<oa::Usize>(pixelCount * 4U)) {
		return oa::Status::error(
			oa::StatusCode::DataLoss,
			"oa::plot::Figure GPU readback has an inconsistent extent");
	}
	oa::Vector<oa::F32> planar(static_cast<oa::Usize>(pixelCount * 4U));
	constexpr oa::F32 normalizeU8 = 1.0F / 255.0F;
	for (oa::Usize channel = 0U; channel < 4U; ++channel) {
		for (oa::Usize pixel = 0U;
			pixel < static_cast<oa::Usize>(pixelCount); ++pixel) {
			planar[channel * static_cast<oa::Usize>(pixelCount) + pixel] =
				static_cast<oa::F32>(rgba[pixel * 4U + channel]) * normalizeU8;
		}
	}
	oa::Matrix matrix = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(planar.data()),
			planar.size() * sizeof(oa::F32)),
		oa::MatrixShape{
			1, 4,
			static_cast<oa::I64>(config_.height),
			static_cast<oa::I64>(config_.width)},
		oa::ScalarType::Float32);
	if (matrix.isEmpty()) {
		return oa::Status::error(
			oa::StatusCode::OutOfMemory,
			"oa::plot::Figure::render image upload failed");
	}
	return oa::Image(
		oa::move(matrix), oa::ImageLayout::Nchw, oa::ImageFormat::Rgba);
}

oa::Result<oa::Image> Figure::render() {
	auto* context = oa::ExecutionSession::getActivePtr();
	if (context == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::plot::Figure::render has no active engine context");
	}
	return render(context->engine());
}

}  // namespace oa::plot
