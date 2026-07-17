// OaPlot::Axes — implementation. Each call records a tiny command struct;
// OaPlot::Figure replays them in Show() / SaveFig().

#include <Oa/Ui/Plot/Axes.h>

#include <cstring>

namespace OaPlot {

void Axes::Imshow(const OaTexture& InTex) {
	Image_.Tex     = InTex;
	Image_.Present = true;
}

void Axes::Plot(OaSpan<const OaF32> InY, const LineStyle& InStyle) {
	Line_.Y.Resize(InY.Size());
	if (InY.Size() > 0) {
		std::memcpy(Line_.Y.Data(), InY.Data(), InY.Size() * sizeof(OaF32));
	}
	Line_.Style   = InStyle;
	Line_.Present = true;
}

void Axes::Heatmap(OaSpan<const OaF32> InValues, OaI32 InRows,
	OaI32 InCols, const HeatmapStyle& InStyle) {
	const OaI64 count = InRows > 0 && InCols > 0
		? static_cast<OaI64>(InRows) * InCols : 0;
	Heatmap_.V.Resize(count);
	if (count > 0 && InValues.Size() >= static_cast<OaUsize>(count)) {
		std::memcpy(Heatmap_.V.Data(), InValues.Data(),
			static_cast<OaUsize>(count) * sizeof(OaF32));
		Heatmap_.Rows = InRows;
		Heatmap_.Cols = InCols;
		Heatmap_.Style = InStyle;
		Heatmap_.Present = true;
	} else {
		Heatmap_.V.Clear();
		Heatmap_.Present = false;
	}
}

void Axes::Title(const char* InText, OaColor InColor) {
	Title_.Text    = InText ? InText : "";
	Title_.Color   = InColor;
	Title_.Present = true;
}

void Axes::XLabel(const char* InText) {
	XLabel_.Text    = InText ? InText : "";
	XLabel_.Color   = {0.831F, 0.831F, 0.831F, 1.0F};  // TextSecondary
	XLabel_.Present = true;
}

void Axes::YLabel(const char* InText) {
	YLabel_.Text    = InText ? InText : "";
	YLabel_.Color   = {0.831F, 0.831F, 0.831F, 1.0F};
	YLabel_.Present = true;
}

void Axes::Caption(const char* InText, OaColor InColor) {
	Caption_.Text    = InText ? InText : "";
	Caption_.Color   = InColor;
	Caption_.Present = true;
}

void Axes::BorderColor(OaColor InColor) {
	Border_    = InColor;
	HasBorder_ = true;
}

}  // namespace OaPlot
