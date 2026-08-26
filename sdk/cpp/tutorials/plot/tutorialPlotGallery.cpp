// OA Plot walkthrough: nine deterministic diagnostic plots, two themes, one
// GPU compositor, and the same retained Figure replay for PNG and oa::Viewer.
//
// usage:
//   TutorialPlotGallery [--output DIR] [--headless]
//
// The two wireframe panels are deliberate 2D projections of scalar fields.
// They demonstrate explicit-X multi-series composition without claiming a
// general 3D scene, camera, depth, or surface-plot API.

#include <oa/core/filesystem.h>
#include <oa/core/paths.h>
#include <oa/runtime/engine.h>
#include <oa/ui/plot/plot.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr oa::F32 kPi = 3.14159265358979323846F;

oa::plot::LineStyle line(
	oa::Color inColor,
	const char* inLabel = "",
	oa::F32 inWidth = 1.35F,
	oa::U32 inAntialiasSamples = 4U) {
	return {
		.color = inColor,
		.label = inLabel,
		.width = inWidth,
		.antialiasSamples = inAntialiasSamples,
	};
}

oa::plot::ScatterStyle points(
	oa::Color inColor, const char* inLabel = "", oa::F32 inRadius = 3.0F) {
	return {.color = inColor, .label = inLabel, .radius = inRadius};
}

oa::plot::BarStyle bars(
	oa::Color inColor, const char* inLabel = "", oa::F32 inGap = 0.18F) {
	return {.color = inColor, .label = inLabel, .gap = inGap};
}

oa::Color gradient(oa::Color inA, oa::Color inB, oa::F32 inT) {
	const oa::F32 t = std::clamp(inT, 0.0F, 1.0F);
	return {
		inA.r + (inB.r - inA.r) * t,
		inA.g + (inB.g - inA.g) * t,
		inA.b + (inB.b - inA.b) * t,
		inA.a + (inB.a - inA.a) * t,
	};
}

// [oa-plot-intro-begin]
oa::plot::Figure introFigure() {
	oa::plot::Figure figure({
		.title = "OA Plot intro",
		.rows = 1,
		.cols = 2,
		.width = 1280U,
		.height = 560U,
		.hSpacing = 36,
		.padding = 34,
		.theme = oa::plot::Theme::Dark,
	});
	figure.title("One retained figure - C++ and Python parity");

	constexpr oa::Array<oa::F32, 8> steps{
		0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F};
	constexpr oa::Array<oa::F32, 8> train{
		1.00F, 0.78F, 0.61F, 0.48F, 0.37F, 0.29F, 0.23F, 0.19F};
	constexpr oa::Array<oa::F32, 8> validation{
		1.04F, 0.83F, 0.66F, 0.53F, 0.43F, 0.35F, 0.30F, 0.27F};
	constexpr oa::Array<oa::F32, 5> ideal{
		0.0F, 0.25F, 0.50F, 0.75F, 1.0F};
	constexpr oa::Array<oa::F32, 5> confidence{
		0.10F, 0.30F, 0.50F, 0.70F, 0.90F};
	constexpr oa::Array<oa::F32, 5> accuracy{
		0.08F, 0.34F, 0.47F, 0.74F, 0.88F};

	auto& curves = figure.ax(0, 0);
	curves.title("training curves");
	curves.xLabel("optimizer step");
	curves.yLabel("cross entropy");
	curves.limits(0.0F, 7.0F, 0.0F, 1.1F);
	curves.plot(steps, train,
		{.color = oa::Color::accent(), .label = "train", .width = 1.6F});
	curves.plot(steps, validation,
		{.color = oa::Color::success(), .label = "validation", .width = 1.6F});

	auto& quality = figure.ax(0, 1);
	quality.title("Calibration");
	quality.xLabel("confidence");
	quality.yLabel("observed accuracy");
	quality.limits(0.0F, 1.0F, 0.0F, 1.0F);
	quality.plot(ideal, ideal,
		{.color = oa::Color{0.565F, 0.565F, 0.565F, 1.0F},
		 .label = "ideal"});
	quality.scatter(confidence, accuracy,
		{.color = oa::Color::cyan(), .label = "model", .radius = 3.5F});
	return figure;
}
// [oa-plot-intro-end]

void trainingPlot(oa::plot::Axes& inAxes) {
	oa::Array<oa::F32, 72> train{};
	oa::Array<oa::F32, 72> validation{};
	for (oa::Usize i = 0U; i < train.size(); ++i) {
		const oa::F32 step = static_cast<oa::F32>(i);
		train[i] = 1.55F * std::exp(-step / 18.0F)
			+ 0.035F * std::sin(step * 0.52F) + 0.11F;
		validation[i] = 1.45F * std::exp(-step / 21.0F)
			+ 0.045F * std::sin(step * 0.39F + 0.8F) + 0.17F;
	}
	inAxes.title("training curves");
	inAxes.xLabel("optimizer step");
	inAxes.yLabel("cross entropy");
	inAxes.plot(train, line(oa::Color::accent(), "train"));
	inAxes.plot(validation, line(oa::Color::success(), "validation"));
}

void rocPlot(oa::plot::Axes& inAxes) {
	oa::Array<oa::F32, 33> rate{};
	oa::Array<oa::F32, 33> vision{};
	oa::Array<oa::F32, 33> fusion{};
	oa::Array<oa::F32, 33> chance{};
	for (oa::Usize i = 0U; i < rate.size(); ++i) {
		const oa::F32 x = static_cast<oa::F32>(i)
			/ static_cast<oa::F32>(rate.size() - 1U);
		rate[i] = x;
		vision[i] = std::clamp(1.0F - std::exp(-4.3F * x), 0.0F, 1.0F);
		fusion[i] = std::clamp(1.0F - std::exp(-6.7F * x), 0.0F, 1.0F);
		chance[i] = x;
	}
	inAxes.title("ROC - threshold sweep");
	inAxes.xLabel("false positive rate");
	inAxes.yLabel("true positive rate");
	inAxes.limits(0.0F, 1.0F, 0.0F, 1.0F);
	inAxes.plot(rate, vision, line(oa::Color::cyan(), "vision AUC 0.88"));
	inAxes.plot(rate, fusion, line(oa::Color::success(), "fusion AUC 0.93"));
	inAxes.plot(rate, chance,
		line(oa::Color{0.565F, 0.565F, 0.565F, 1.0F}, "chance"));
}

void precisionRecallPlot(oa::plot::Axes& inAxes) {
	oa::Array<oa::F32, 33> recall{};
	oa::Array<oa::F32, 33> vision{};
	oa::Array<oa::F32, 33> fusion{};
	for (oa::Usize i = 0U; i < recall.size(); ++i) {
		const oa::F32 x = static_cast<oa::F32>(i)
			/ static_cast<oa::F32>(recall.size() - 1U);
		recall[i] = x;
		vision[i] = std::clamp(0.98F - 0.40F * std::pow(x, 1.6F)
			- 0.025F * std::sin(x * 5.0F * kPi), 0.0F, 1.0F);
		fusion[i] = std::clamp(0.995F - 0.27F * std::pow(x, 2.1F)
			- 0.015F * std::sin(x * 4.0F * kPi), 0.0F, 1.0F);
	}
	inAxes.title("Precision-recall");
	inAxes.xLabel("recall");
	inAxes.yLabel("precision");
	inAxes.limits(0.0F, 1.0F, 0.0F, 1.0F);
	inAxes.plot(recall, vision, line(oa::Color::purple(), "vision AP 0.84"));
	inAxes.plot(recall, fusion, line(oa::Color::pink(), "fusion AP 0.91"));
}

void scoreHistogram(oa::plot::Axes& inAxes) {
	oa::Array<oa::F32, 192> scores{};
	for (oa::Usize i = 0U; i < scores.size(); ++i) {
		const oa::F32 phase = static_cast<oa::F32>(i);
		const oa::F32 cluster = i % 3U == 0U ? 0.28F : 0.74F;
		scores[i] = std::clamp(cluster
			+ 0.12F * std::sin(phase * 1.73F)
			+ 0.045F * std::cos(phase * 0.37F), 0.0F, 1.0F);
	}
	inAxes.title("confidence distribution");
	inAxes.xLabel("model score");
	inAxes.yLabel("samples");
	inAxes.histogram(scores, 20,
		bars(oa::Color::accent(), "validation scores", 0.20F));
}

void calibrationScatter(oa::plot::Axes& inAxes) {
	oa::Array<oa::F32, 24> confidence{};
	oa::Array<oa::F32, 24> accuracy{};
	oa::Array<oa::F32, 24> ideal{};
	for (oa::Usize i = 0U; i < confidence.size(); ++i) {
		const oa::F32 x = (static_cast<oa::F32>(i) + 0.5F)
			/ static_cast<oa::F32>(confidence.size());
		confidence[i] = x;
		accuracy[i] = std::clamp(x + 0.055F * std::sin(8.0F * kPi * x)
			- 0.025F, 0.0F, 1.0F);
		ideal[i] = x;
	}
	inAxes.title("Calibration");
	inAxes.xLabel("confidence");
	inAxes.yLabel("observed accuracy");
	inAxes.limits(0.0F, 1.0F, 0.0F, 1.0F);
	inAxes.plot(confidence, ideal,
		line(oa::Color{0.565F, 0.565F, 0.565F, 1.0F}, "ideal"));
	inAxes.scatter(confidence, accuracy,
		points(oa::Color::cyan(), "model", 3.5F));
}

void throughputBars(oa::plot::Axes& inAxes) {
	constexpr oa::Array<oa::F32, 6> throughput{
		0.42F, 0.58F, 0.71F, 0.86F, 0.79F, 0.94F};
	inAxes.title("normalized throughput");
	inAxes.xLabel("kernel route");
	inAxes.yLabel("relative peak");
	inAxes.limits(-0.5F, 5.5F, 0.0F, 1.0F);
	inAxes.bar(throughput,
		bars(oa::Color::warning(), "device routes", 0.24F));
}

void confusionHeatmap(oa::plot::Axes& inAxes) {
	constexpr oa::Array<oa::F32, 25> confusion{
		42.0F, 2.0F, 0.0F, 1.0F, 0.0F,
		3.0F, 37.0F, 2.0F, 0.0F, 1.0F,
		0.0F, 2.0F, 40.0F, 3.0F, 0.0F,
		1.0F, 0.0F, 2.0F, 38.0F, 2.0F,
		0.0F, 1.0F, 0.0F, 2.0F, 43.0F,
	};
	inAxes.title("confusion matrix");
	inAxes.xLabel("predicted class");
	inAxes.yLabel("reference class");
	inAxes.heatmap(confusion, 5, 5,
		{.colormap = 1U, .autoScale = true, .showGrid = true});
}

template <typename HeightFn>
void wireframe(
	oa::plot::Axes& inAxes,
	const char* inTitle,
	HeightFn&& inHeight,
	oa::Color inNear,
	oa::Color inFar) {
	constexpr oa::I32 lines = 13;
	constexpr oa::I32 samples = 29;
	for (oa::I32 row = 0; row < lines; ++row) {
		const oa::F32 v = -1.0F + 2.0F * static_cast<oa::F32>(row)
			/ static_cast<oa::F32>(lines - 1);
		oa::Array<oa::F32, samples> px{};
		oa::Array<oa::F32, samples> py{};
		for (oa::I32 column = 0; column < samples; ++column) {
			const oa::F32 u = -1.0F + 2.0F * static_cast<oa::F32>(column)
				/ static_cast<oa::F32>(samples - 1);
			const oa::F32 z = inHeight(u, v);
			px[static_cast<oa::Usize>(column)] = 0.50F + 0.34F * (u - v);
			py[static_cast<oa::Usize>(column)] = 0.58F
				- 0.16F * (u + v) - 0.23F * z;
		}
		const oa::F32 t = static_cast<oa::F32>(row)
			/ static_cast<oa::F32>(lines - 1);
		inAxes.plot(px, py,
			line(gradient(inFar, inNear, t), "", 1.15F, 8U));
	}
	for (oa::I32 column = 0; column < lines; ++column) {
		const oa::F32 u = -1.0F + 2.0F * static_cast<oa::F32>(column)
			/ static_cast<oa::F32>(lines - 1);
		oa::Array<oa::F32, samples> px{};
		oa::Array<oa::F32, samples> py{};
		for (oa::I32 row = 0; row < samples; ++row) {
			const oa::F32 v = -1.0F + 2.0F * static_cast<oa::F32>(row)
				/ static_cast<oa::F32>(samples - 1);
			const oa::F32 z = inHeight(u, v);
			px[static_cast<oa::Usize>(row)] = 0.50F + 0.34F * (u - v);
			py[static_cast<oa::Usize>(row)] = 0.58F
				- 0.16F * (u + v) - 0.23F * z;
		}
		const oa::F32 t = static_cast<oa::F32>(column)
			/ static_cast<oa::F32>(lines - 1);
		inAxes.plot(px, py,
			line(gradient(inFar, inNear, t).withAlpha(0.78F), "", 1.15F, 8U));
	}
	inAxes.title(inTitle);
	inAxes.limits(-0.20F, 1.20F, 0.02F, 1.08F);
	inAxes.grid(false);
	inAxes.legend(false);
}

void lossLandscape(oa::plot::Axes& inAxes) {
	wireframe(inAxes, "projected loss landscape",
		[](oa::F32 x, oa::F32 y) {
			const oa::F32 bowl = 0.42F * (x * x + 0.75F * y * y);
			return bowl + 0.18F * std::sin(2.7F * x) * std::cos(3.1F * y);
		}, oa::Color::cyan(), oa::Color::purple());
}

void gradientBasin(oa::plot::Axes& inAxes) {
	wireframe(inAxes, "projected optimizer basin",
		[](oa::F32 x, oa::F32 y) {
			const oa::F32 radius = std::sqrt(x * x + y * y);
			return 0.34F * radius + 0.17F * std::cos(9.0F * radius)
				* std::exp(-1.8F * radius);
		}, oa::Color::success(), oa::Color::warning());
}

oa::plot::Figure gallery(oa::plot::Theme inTheme) {
	oa::plot::Figure figure({
		.title = "OA Plot gallery",
		.rows = 3,
		.cols = 3,
		.width = 1600U,
		.height = 1050U,
		.hSpacing = 32,
		.vSpacing = 36,
		.padding = 34,
		.theme = inTheme,
	});
	figure.title("OA Plot - GPU-composed ML diagnostics");
	trainingPlot(figure.ax(0, 0));
	rocPlot(figure.ax(0, 1));
	precisionRecallPlot(figure.ax(0, 2));
	scoreHistogram(figure.ax(1, 0));
	calibrationScatter(figure.ax(1, 1));
	throughputBars(figure.ax(1, 2));
	confusionHeatmap(figure.ax(2, 0));
	lossLandscape(figure.ax(2, 1));
	gradientBasin(figure.ax(2, 2));
	return figure;
}

oa::plot::Figure evaluationFigure() {
	oa::plot::Figure figure({
		.title = "OA model evaluation",
		.rows = 1,
		.cols = 2,
		.width = 1280U,
		.height = 560U,
		.hSpacing = 36,
		.padding = 34,
	});
	figure.title("Model evaluation - ROC and precision-recall");
	rocPlot(figure.ax(0, 0));
	precisionRecallPlot(figure.ax(0, 1));
	return figure;
}

oa::plot::Figure diagnosticsFigure() {
	oa::plot::Figure figure({
		.title = "OA model diagnostics",
		.rows = 2,
		.cols = 2,
		.width = 1200U,
		.height = 900U,
		.hSpacing = 32,
		.vSpacing = 36,
		.padding = 34,
	});
	figure.title("training and evaluation diagnostics");
	trainingPlot(figure.ax(0, 0));
	scoreHistogram(figure.ax(0, 1));
	calibrationScatter(figure.ax(1, 0));
	confusionHeatmap(figure.ax(1, 1));
	return figure;
}

oa::plot::Figure landscapeFigure() {
	oa::plot::Figure figure({
		.title = "OA projected landscapes",
		.rows = 1,
		.cols = 2,
		.width = 1280U,
		.height = 560U,
		.hSpacing = 36,
		.padding = 34,
	});
	figure.title("scalar-field projections - explicit-X wireframes");
	lossLandscape(figure.ax(0, 0));
	gradientBasin(figure.ax(0, 1));
	return figure;
}

oa::Status saveFigure(
	oa::Engine& inEngine,
	oa::plot::Figure& inFigure,
	const oa::Path& inDirectory,
	const char* inName) {
	const oa::Path path = inDirectory / inName;
	OA_RETURN_IF_ERROR(inFigure.saveTo(inEngine, path.cStr()));
	std::printf("  %s\n", path.cStr());
	return oa::Status::ok();
}

void usage(const char* inProgram) {
	std::printf("usage: %s [--output DIR] [--headless]\n", inProgram);
}

} // namespace

int main(int argc, char** argv) {
	oa::Path output = oa::Paths::var("artifact/plot");
	bool show = true;
	for (int i = 1; i < argc; ++i) {
		if (std::strcmp(argv[i], "--headless") == 0) {
			show = false;
		} else if (std::strcmp(argv[i], "--output") == 0 and i + 1 < argc) {
			output = oa::Path(argv[++i]);
		} else if (std::strcmp(argv[i], "--help") == 0) {
			usage(argv[0]);
			return EXIT_SUCCESS;
		} else {
			usage(argv[0]);
			return EXIT_FAILURE;
		}
	}

	if (const oa::Status status = oa::Filesystem::createDirectories(output);
		status.isError()) {
		std::fprintf(stderr, "output directory failed: %s\n",
			status.toString().cStr());
		return EXIT_FAILURE;
	}

	oa::EngineConfig engineConfig;
	engineConfig.presentationMode = oa::PresentationMode::Headless;
	engineConfig.selectForThread = false;
	engineConfig.appName = "TutorialPlotGallery";
	auto engineResult = oa::Engine::create(engineConfig);
	if (not engineResult.isOk()) {
		std::fprintf(stderr, "Engine creation failed: %s\n",
			engineResult.getStatus().toString().cStr());
		return EXIT_FAILURE;
	}
	oa::Engine& engine = *engineResult.getValue();

	auto dark = gallery(oa::plot::Theme::Dark);
	auto light = gallery(oa::plot::Theme::Light);
	auto intro = introFigure();
	auto evaluation = evaluationFigure();
	auto diagnostics = diagnosticsFigure();
	auto landscapes = landscapeFigure();

	std::printf("OA Plot walkthrough artifacts:\n");
	if (saveFigure(engine, intro, output, "oa-plot-intro-dark.png").isError()
		or saveFigure(engine, dark, output, "oa-plot-gallery-dark.png").isError()
		or saveFigure(engine, light, output, "oa-plot-gallery-light.png").isError()
		or saveFigure(engine, evaluation, output, "oa-plot-evaluation-dark.png").isError()
		or saveFigure(engine, diagnostics, output, "oa-plot-diagnostics-dark.png").isError()
		or saveFigure(engine, landscapes, output, "oa-plot-landscapes-dark.png").isError()) {
		std::fprintf(stderr, "Plot walkthrough render failed\n");
		return EXIT_FAILURE;
	}

	if (show) {
		const oa::Status status = dark.show();
		if (status.isError()) {
			std::fprintf(stderr, "oa::Viewer failed: %s\n", status.toString().cStr());
			return EXIT_FAILURE;
		}
	}
	return EXIT_SUCCESS;
}
