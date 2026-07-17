// OA Python bindings — compact plotting surface.
#include "../Binding.h"

#include <Oa/Ui/Plot/Plot.h>

void BindPlot(nb::module_& m) {
	nb::class_<OaPlot::FigureConfig>(m, "FigureConfig")
		.def(nb::init<>())
		.def_rw("Title", &OaPlot::FigureConfig::Title)
		.def_rw("Rows", &OaPlot::FigureConfig::Rows)
		.def_rw("Cols", &OaPlot::FigureConfig::Cols)
		.def_rw("Width", &OaPlot::FigureConfig::Width)
		.def_rw("Height", &OaPlot::FigureConfig::Height)
		.def_rw("HSpacing", &OaPlot::FigureConfig::HSpacing)
		.def_rw("VSpacing", &OaPlot::FigureConfig::VSpacing)
		.def_rw("Padding", &OaPlot::FigureConfig::Padding);

	nb::class_<OaPlot::HeatmapStyle>(m, "HeatmapStyle")
		.def(nb::init<>())
		.def_rw("VMin", &OaPlot::HeatmapStyle::VMin)
		.def_rw("VMax", &OaPlot::HeatmapStyle::VMax)
		.def_rw("Colormap", &OaPlot::HeatmapStyle::Colormap)
		.def_rw("AutoScale", &OaPlot::HeatmapStyle::AutoScale)
		.def_rw("ShowGrid", &OaPlot::HeatmapStyle::ShowGrid);

	nb::class_<OaPlot::Axes>(m, "Axes")
		.def("Plot", [](OaPlot::Axes& axes, const std::vector<OaF32>& values) {
			axes.Plot(OaSpan<const OaF32>(values.data(), values.size()));
		}, nb::arg("values"))
		.def("Heatmap", [](OaPlot::Axes& axes,
			const std::vector<OaF32>& values, OaI32 rows, OaI32 cols,
			const OaPlot::HeatmapStyle& style) {
			axes.Heatmap(OaSpan<const OaF32>(values.data(), values.size()),
				rows, cols, style);
		}, nb::arg("values"), nb::arg("rows"), nb::arg("cols"),
			nb::arg("style") = OaPlot::HeatmapStyle())
		.def("Title", [](OaPlot::Axes& axes, const char* text) {
			axes.Title(text);
		}, nb::arg("text"))
		.def("XLabel", &OaPlot::Axes::XLabel, nb::arg("text"))
		.def("YLabel", &OaPlot::Axes::YLabel, nb::arg("text"))
		.def("Caption", [](OaPlot::Axes& axes, const char* text) {
			axes.Caption(text);
		}, nb::arg("text"));

	nb::class_<OaPlot::Figure>(m, "Figure")
		.def(nb::init<const OaPlot::FigureConfig&>(),
			nb::arg("config") = OaPlot::FigureConfig())
		.def("Ax", &OaPlot::Figure::Ax, nb::arg("row"), nb::arg("col"),
			nb::rv_policy::reference_internal)
		.def("Title", &OaPlot::Figure::Title, nb::arg("text"))
		.def("XLabel", &OaPlot::Figure::XLabel, nb::arg("text"))
		.def("YLabel", &OaPlot::Figure::YLabel, nb::arg("text"))
		.def("SaveFig", [](OaPlot::Figure& figure, const char* path) {
			throw_if_error(figure.SaveFig(path));
		}, nb::arg("path"))
		.def("Show", [](OaPlot::Figure& figure) {
			throw_if_error(figure.Show());
		});
}
