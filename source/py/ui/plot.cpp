// OA Python bindings — compact plotting surface.
#include "../binding.h"

#include <oa/runtime/engine.h>
#include <oa/runtime/texture.h>
#include <oa/ui/plot/plot.h>

void bindPlot(nb::module_& m) {
	nb::enum_<oa::plot::Theme>(m, "Theme")
		.value("Dark", oa::plot::Theme::Dark)
		.value("Light", oa::plot::Theme::Light);

	nb::class_<oa::plot::FigureConfig>(m, "FigureConfig")
		.def(nb::init<>())
		.def_prop_rw("title",
			[](const oa::plot::FigureConfig& config) {
				return std::string(config.title.cStr());
			},
			[](oa::plot::FigureConfig& config, const std::string& title) {
				config.title = oa::String(title.c_str());
			})
		.def_rw("rows", &oa::plot::FigureConfig::rows)
		.def_rw("cols", &oa::plot::FigureConfig::cols)
		.def_rw("width", &oa::plot::FigureConfig::width)
		.def_rw("height", &oa::plot::FigureConfig::height)
		.def_rw("hSpacing", &oa::plot::FigureConfig::hSpacing)
		.def_rw("vSpacing", &oa::plot::FigureConfig::vSpacing)
		.def_rw("padding", &oa::plot::FigureConfig::padding)
		.def_rw("theme", &oa::plot::FigureConfig::theme)
		.def_prop_rw("backgroundRgba",
			[](const oa::plot::FigureConfig& config) {
				return config.background.toU32();
			},
			[](oa::plot::FigureConfig& config, oa::U32 rgba) {
				config.background = oa::Color::fromU32(rgba);
			});

	nb::class_<oa::plot::LineStyle>(m, "LineStyle")
		.def(nb::init<>())
		.def_rw("width", &oa::plot::LineStyle::width)
		.def_rw("antialiasSamples", &oa::plot::LineStyle::antialiasSamples)
		.def_prop_rw("colorRgba",
			[](const oa::plot::LineStyle& style) { return style.color.toU32(); },
			[](oa::plot::LineStyle& style, oa::U32 rgba) {
				style.color = oa::Color::fromU32(rgba);
			})
		.def_prop_rw("label",
			[](const oa::plot::LineStyle& style) {
				return std::string(style.label.cStr());
			},
			[](oa::plot::LineStyle& style, const std::string& label) {
				style.label = oa::String(label.c_str());
			});

	nb::class_<oa::plot::ScatterStyle>(m, "ScatterStyle")
		.def(nb::init<>())
		.def_rw("radius", &oa::plot::ScatterStyle::radius)
		.def_prop_rw("colorRgba",
			[](const oa::plot::ScatterStyle& style) { return style.color.toU32(); },
			[](oa::plot::ScatterStyle& style, oa::U32 rgba) {
				style.color = oa::Color::fromU32(rgba);
			})
		.def_prop_rw("label",
			[](const oa::plot::ScatterStyle& style) {
				return std::string(style.label.cStr());
			},
			[](oa::plot::ScatterStyle& style, const std::string& label) {
				style.label = oa::String(label.c_str());
			});

	nb::class_<oa::plot::BarStyle>(m, "BarStyle")
		.def(nb::init<>())
		.def_rw("gap", &oa::plot::BarStyle::gap)
		.def_prop_rw("colorRgba",
			[](const oa::plot::BarStyle& style) { return style.color.toU32(); },
			[](oa::plot::BarStyle& style, oa::U32 rgba) {
				style.color = oa::Color::fromU32(rgba);
			})
		.def_prop_rw("label",
			[](const oa::plot::BarStyle& style) {
				return std::string(style.label.cStr());
			},
			[](oa::plot::BarStyle& style, const std::string& label) {
				style.label = oa::String(label.c_str());
			});

	nb::class_<oa::plot::HeatmapStyle>(m, "HeatmapStyle")
		.def(nb::init<>())
		.def_rw("vMin", &oa::plot::HeatmapStyle::vMin)
		.def_rw("vMax", &oa::plot::HeatmapStyle::vMax)
		.def_rw("colormap", &oa::plot::HeatmapStyle::colormap)
		.def_rw("autoScale", &oa::plot::HeatmapStyle::autoScale)
		.def_rw("showGrid", &oa::plot::HeatmapStyle::showGrid);

	nb::class_<oa::plot::Axes>(m, "Axes")
		.def("imshow", [](oa::plot::Axes& axes, const oa::Image& image) {
			auto texture = oa::FnTexture::fromImage(pythonEngine(), image);
			if (texture.isError()) {
				throwIfError(texture.getStatus());
			}
			axes.imshow(*texture);
		}, nb::arg("image"))
		.def("plot", [](oa::plot::Axes& axes, const std::vector<oa::F32>& values,
			const oa::plot::LineStyle& style) {
			axes.plot(oa::Span<const oa::F32>(values.data(), values.size()), style);
		}, nb::arg("values"), nb::arg("style") = oa::plot::LineStyle())
		.def("plot", [](oa::plot::Axes& axes,
			const std::vector<oa::F32>& x, const std::vector<oa::F32>& y,
			const oa::plot::LineStyle& style) {
			axes.plot(oa::Span<const oa::F32>(x.data(), x.size()),
				oa::Span<const oa::F32>(y.data(), y.size()), style);
		}, nb::arg("x"), nb::arg("y"),
			nb::arg("style") = oa::plot::LineStyle())
		.def("scatter", [](oa::plot::Axes& axes,
			const std::vector<oa::F32>& x, const std::vector<oa::F32>& y,
			const oa::plot::ScatterStyle& style) {
			axes.scatter(oa::Span<const oa::F32>(x.data(), x.size()),
				oa::Span<const oa::F32>(y.data(), y.size()), style);
		}, nb::arg("x"), nb::arg("y"),
			nb::arg("style") = oa::plot::ScatterStyle())
		.def("bar", [](oa::plot::Axes& axes, const std::vector<oa::F32>& values,
			const oa::plot::BarStyle& style) {
			axes.bar(oa::Span<const oa::F32>(values.data(), values.size()), style);
		}, nb::arg("values"), nb::arg("style") = oa::plot::BarStyle())
		.def("histogram", [](oa::plot::Axes& axes,
			const std::vector<oa::F32>& values, oa::I32 bins,
			const oa::plot::BarStyle& style) {
			axes.histogram(oa::Span<const oa::F32>(values.data(), values.size()),
				bins, style);
		}, nb::arg("values"), nb::arg("bins") = 16,
			nb::arg("style") = oa::plot::BarStyle())
		.def("limits", &oa::plot::Axes::limits,
			nb::arg("xMin"), nb::arg("xMax"),
			nb::arg("yMin"), nb::arg("yMax"))
		.def("autoLimits", &oa::plot::Axes::autoLimits)
		.def("grid", &oa::plot::Axes::grid, nb::arg("visible") = true)
		.def("legend", &oa::plot::Axes::legend, nb::arg("visible") = true)
		.def("heatmap", [](oa::plot::Axes& axes,
			const std::vector<oa::F32>& values, oa::I32 rows, oa::I32 cols,
			const oa::plot::HeatmapStyle& style) {
			axes.heatmap(oa::Span<const oa::F32>(values.data(), values.size()),
				rows, cols, style);
		}, nb::arg("values"), nb::arg("rows"), nb::arg("cols"),
			nb::arg("style") = oa::plot::HeatmapStyle())
		.def("title", [](oa::plot::Axes& axes, const char* text, oa::U32 colorRgba) {
			axes.title(text, oa::Color::fromU32(colorRgba));
		}, nb::arg("text"), nb::arg("colorRgba") = 0U)
		.def("xLabel", &oa::plot::Axes::xLabel, nb::arg("text"))
		.def("yLabel", &oa::plot::Axes::yLabel, nb::arg("text"))
		.def("caption", [](oa::plot::Axes& axes, const char* text, oa::U32 colorRgba) {
			axes.caption(text, oa::Color::fromU32(colorRgba));
		}, nb::arg("text"), nb::arg("colorRgba") = 0U)
		.def("borderColor", [](oa::plot::Axes& axes, oa::U32 colorRgba) {
			axes.borderColor(oa::Color::fromU32(colorRgba));
		}, nb::arg("colorRgba"));

	nb::class_<oa::plot::Figure>(m, "Figure")
		.def(nb::init<const oa::plot::FigureConfig&>(),
			nb::arg("config") = oa::plot::FigureConfig())
		.def("ax", &oa::plot::Figure::ax, nb::arg("row"), nb::arg("col"),
			nb::rv_policy::reference_internal)
		.def("title", &oa::plot::Figure::title, nb::arg("text"))
		.def("xLabel", &oa::plot::Figure::xLabel, nb::arg("text"))
		.def("yLabel", &oa::plot::Figure::yLabel, nb::arg("text"))
		.def_prop_ro("rows", &oa::plot::Figure::rows)
		.def_prop_ro("cols", &oa::plot::Figure::cols)
		.def("saveTo", [](oa::plot::Figure& figure, const char* path) {
			throwIfError(figure.saveTo(pythonEngine(), path));
		}, nb::arg("path"))
		.def("render", [](oa::plot::Figure& figure) {
			auto result = figure.render(pythonEngine());
			if (result.isError()) {
				throwIfError(result.getStatus());
			}
			return oa::move(result).getValue();
		}, "Render the fixed-size plot as a semantic RGBA oa::Image.")
		.def("show", [](oa::plot::Figure& figure) {
			throwIfError(figure.show());
		});
}
