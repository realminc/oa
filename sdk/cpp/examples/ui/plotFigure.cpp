// OA_DOC_BEGIN: plot-line
#include <oa/oa.h>

OA_MAIN_MODE("ExamplePlotLine", argc > 1 and oa::StringView(argv[1]) == "--preview" ? oa::PresentationMode::Swapchain : oa::PresentationMode::Headless) {
	oa::plot::Figure figure({
		.title = "Training loss",
		.width = 960U,
		.height = 540U,
		.theme = oa::plot::Theme::Dark,
	});
	constexpr oa::Array<oa::F32, 6> figureValues{
		1.0F, 0.72F, 0.51F, 0.38F, 0.29F, 0.23F
	};
	figure.ax(0, 0).xLabel("step");
	figure.ax(0, 0).yLabel("loss");
	figure.ax(0, 0).plot(figureValues);

	auto renderedResult = figure.render(engine);
	if (not renderedResult.isOk()) return 1;
	auto rendered = oa::move(renderedResult).getValue();

	const oa::Path output = oa::Paths::var("example/ui/trainingLoss.jpg");
	if (not oa::Filesystem::createDirectories(output.parentPath()).isOk()) return 1;
	if (not oa::FnImage::saveFile(output, rendered, 92U).isOk()) return 1;

	if (not rendered.validate()) return 1;
	if (rendered.width() != 960 or rendered.height() != 540) return 1;
	if (rendered.format() != oa::ImageFormat::Rgba) return 1;
	if (not oa::Filesystem::isFile(output)) return 1;

	oa::print("Saved training-loss plot: {}", output.cStr());

	if (argc > 1 and oa::StringView(argv[1]) == "--preview") {
		oa::ViewerConfig previewConfig;
		previewConfig.mode = oa::ViewerMode::Image;
		previewConfig.title = "OA plot · training loss";
		previewConfig.width = 1280U;
		previewConfig.height = 646U;
		if (not oa::Viewer::preview(engine, oa::Paths::var("example/ui/trainingLoss.jpg").string(),
			previewConfig).isOk()) {
			return 1;
		}
	}
	return 0;
}
// OA_DOC_END: plot-line
