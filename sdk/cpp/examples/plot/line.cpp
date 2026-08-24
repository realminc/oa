// OA_DOC_BEGIN: plot-line
#include <oa/runtime/engine.h>
#include <oa/ui/plot/plot.h>

#include <array>
#include <cstdio>
#include <utility>

int main() {
	oa::EngineConfig engineConfig;
	engineConfig.appName = "ExamplePlotLine";
	engineConfig.presentationMode = oa::PresentationMode::Headless;

	auto created = oa::Engine::create(engineConfig);
	if (not created.isOk()) return 1;
	auto engine = std::move(created).getValue();

	oa::plot::Figure figure({
		.title = "training loss",
		.width = 360,
		.height = 240,
		.theme = oa::plot::Theme::Dark,
	});
	constexpr std::array<oa::F32, 6> loss{1.0F, 0.72F, 0.51F, 0.38F, 0.29F, 0.23F};
	figure.ax(0, 0).xLabel("step");
	figure.ax(0, 0).yLabel("loss");
	figure.ax(0, 0).plot(loss);

	auto rendered = figure.render(*engine);
	if (not rendered.isOk()) return 1;
	auto image = std::move(rendered).getValue();
	if (not image.validate() || image.width() != 360 || image.height() != 240) {
		return 1;
	}

	std::puts("rendered a 360x240 training-loss figure");
	return 0;
}
// OA_DOC_END: plot-line
