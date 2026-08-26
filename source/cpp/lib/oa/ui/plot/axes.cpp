// oa::plot::Axes — implementation. Each call records a tiny command struct;
// oa::plot::Figure replays them in show() / saveTo().

#include <oa/ui/plot/axes.h>

#include <oa/core/memory.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>

namespace oa::plot {

void Axes::imshow(const oa::Texture& inTex) {
	image_.tex     = inTex;
	image_.present = true;
	heatmap_.present = false;
}

void Axes::plot(oa::Span<const oa::F32> inY, const LineStyle& inStyle) {
	if (inY.empty()) return;
	LineCmd command;
	command.y.resize(inY.size());
	oa::memcpy(command.y.data(), inY.data(), inY.size() * sizeof(oa::F32));
	command.style = inStyle;
	lines_.pushBack(oa::move(command));
	artists_.pushBack({ArtistKind::Line,
		static_cast<oa::U32>(lines_.size() - 1U)});
}

void Axes::plot(oa::Span<const oa::F32> inX, oa::Span<const oa::F32> inY,
	const LineStyle& inStyle) {
	if (inX.empty() or inX.size() != inY.size()) return;
	LineCmd command;
	command.x.resize(inX.size());
	command.y.resize(inY.size());
	oa::memcpy(command.x.data(), inX.data(), inX.size() * sizeof(oa::F32));
	oa::memcpy(command.y.data(), inY.data(), inY.size() * sizeof(oa::F32));
	command.style = inStyle;
	lines_.pushBack(oa::move(command));
	artists_.pushBack({ArtistKind::Line,
		static_cast<oa::U32>(lines_.size() - 1U)});
}

void Axes::scatter(oa::Span<const oa::F32> inX, oa::Span<const oa::F32> inY,
	const ScatterStyle& inStyle) {
	if (inX.empty() or inX.size() != inY.size()) return;
	ScatterCmd command;
	command.x.resize(inX.size());
	command.y.resize(inY.size());
	oa::memcpy(command.x.data(), inX.data(), inX.size() * sizeof(oa::F32));
	oa::memcpy(command.y.data(), inY.data(), inY.size() * sizeof(oa::F32));
	command.style = inStyle;
	scatters_.pushBack(oa::move(command));
	artists_.pushBack({ArtistKind::Scatter,
		static_cast<oa::U32>(scatters_.size() - 1U)});
}

void Axes::bar(oa::Span<const oa::F32> inY, const BarStyle& inStyle) {
	if (inY.empty()) return;
	BarCmd command;
	command.x.resize(inY.size());
	command.y.resize(inY.size());
	for (oa::Usize i = 0; i < inY.size(); ++i) {
		command.x[i] = static_cast<oa::F32>(i);
		command.y[i] = inY[i];
	}
	command.style = inStyle;
	command.width = oa::clamp(1.0F - inStyle.gap, 0.05F, 1.0F);
	bars_.pushBack(oa::move(command));
	artists_.pushBack({ArtistKind::Bar,
		static_cast<oa::U32>(bars_.size() - 1U)});
}

void Axes::histogram(oa::Span<const oa::F32> inValues, oa::I32 inBins,
	const BarStyle& inStyle) {
	if (inValues.empty() or inBins <= 0) return;
	const oa::I32 bins = oa::min(inBins, 4096);
	oa::F32 minimum = oa::Limits<oa::F32>::infinity();
	oa::F32 maximum = -oa::Limits<oa::F32>::infinity();
	for (const oa::F32 value : inValues) {
		if (not oa::isFinite(value)) continue;
		minimum = oa::min(minimum, value);
		maximum = oa::max(maximum, value);
	}
	if (not oa::isFinite(minimum) or not oa::isFinite(maximum)) return;
	if (maximum <= minimum) {
		const oa::F32 margin = oa::max(1.0e-4F, oa::abs(minimum) * 0.05F);
		minimum -= margin;
		maximum += margin;
	}
	const oa::F32 binWidth = (maximum - minimum) / static_cast<oa::F32>(bins);
	BarCmd command;
	command.x.resize(static_cast<oa::Usize>(bins));
	command.y.resize(static_cast<oa::Usize>(bins));
	for (oa::I32 bin = 0; bin < bins; ++bin) {
		command.x[static_cast<oa::Usize>(bin)] = minimum
			+ (static_cast<oa::F32>(bin) + 0.5F) * binWidth;
	}
	for (const oa::F32 value : inValues) {
		if (not oa::isFinite(value)) continue;
		const oa::I32 bin = oa::clamp(
			static_cast<oa::I32>((value - minimum) / binWidth), 0, bins - 1);
		command.y[static_cast<oa::Usize>(bin)] += 1.0F;
	}
	command.style = inStyle;
	command.width = binWidth * oa::clamp(1.0F - inStyle.gap, 0.05F, 1.0F);
	bars_.pushBack(oa::move(command));
	artists_.pushBack({ArtistKind::Bar,
		static_cast<oa::U32>(bars_.size() - 1U)});
}

void Axes::limits(oa::F32 inXMin, oa::F32 inXMax, oa::F32 inYMin, oa::F32 inYMax) {
	const bool valid = oa::isFinite(inXMin) and oa::isFinite(inXMax)
		and oa::isFinite(inYMin) and oa::isFinite(inYMax)
		and inXMax > inXMin and inYMax > inYMin;
	limits_ = {inXMin, inXMax, inYMin, inYMax, valid};
}

void Axes::autoLimits() {
	limits_.present = false;
}

void Axes::grid(bool inVisible) {
	showGrid_ = inVisible;
}

void Axes::legend(bool inVisible) {
	showLegend_ = inVisible;
}

void Axes::heatmap(oa::Span<const oa::F32> inValues, oa::I32 inRows,
	oa::I32 inCols, const HeatmapStyle& inStyle) {
	const oa::U64 count = inRows > 0 and inCols > 0
		? static_cast<oa::U64>(inRows) * static_cast<oa::U64>(inCols) : 0U;
	const bool valid = count > 0U
		and count <= oa::Limits<oa::Usize>::max()
		and inValues.size() >= static_cast<oa::Usize>(count);
	heatmap_.v.resize(valid ? static_cast<oa::Usize>(count) : 0U);
	if (valid) {
		oa::memcpy(heatmap_.v.data(), inValues.data(),
			static_cast<oa::Usize>(count) * sizeof(oa::F32));
		heatmap_.rows = inRows;
		heatmap_.cols = inCols;
		heatmap_.style = inStyle;
		heatmap_.present = true;
		image_.present = false;
	} else {
		heatmap_.v.clear();
		heatmap_.present = false;
	}
}

void Axes::title(const char* inText, oa::Color inColor) {
	title_.text    = inText ? inText : "";
	title_.color   = inColor;
	title_.present = true;
}

void Axes::xLabel(const char* inText) {
	xLabel_.text    = inText ? inText : "";
	xLabel_.color   = {0.0F, 0.0F, 0.0F, 0.0F};
	xLabel_.present = true;
}

void Axes::yLabel(const char* inText) {
	yLabel_.text    = inText ? inText : "";
	yLabel_.color   = {0.0F, 0.0F, 0.0F, 0.0F};
	yLabel_.present = true;
}

void Axes::caption(const char* inText, oa::Color inColor) {
	caption_.text    = inText ? inText : "";
	caption_.color   = inColor;
	caption_.present = true;
}

void Axes::borderColor(oa::Color inColor) {
	border_    = inColor;
	hasBorder_ = true;
}

}  // namespace oa::plot
