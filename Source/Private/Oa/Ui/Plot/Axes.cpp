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

void Axes::Bar(OaSpan<const OaF32> InValues, const BarStyle& InStyle) {
	Bar_.V.Resize(InValues.Size());
	if (InValues.Size() > 0) {
		std::memcpy(Bar_.V.Data(), InValues.Data(), InValues.Size() * sizeof(OaF32));
	}
	Bar_.Style   = InStyle;
	Bar_.Present = true;
}

void Axes::Scatter(OaSpan<const OaF32> InXs, OaSpan<const OaF32> InYs,
                   const ScatterStyle& InStyle) {
	const OaUsize n = InXs.Size() < InYs.Size() ? InXs.Size() : InYs.Size();
	Scatter_.Xs.Resize(n);
	Scatter_.Ys.Resize(n);
	if (n > 0) {
		std::memcpy(Scatter_.Xs.Data(), InXs.Data(), n * sizeof(OaF32));
		std::memcpy(Scatter_.Ys.Data(), InYs.Data(), n * sizeof(OaF32));
	}
	Scatter_.Style   = InStyle;
	Scatter_.Present = true;
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
