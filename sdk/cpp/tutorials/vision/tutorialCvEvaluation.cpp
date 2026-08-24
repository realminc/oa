// OA Tutorial: compact computer-vision evaluation.
//
// Runs real GPU classification and detection metrics, reads back only the
// small evaluation artifacts, and saves a presentation-ready figure containing
// training curves, a confusion matrix, and mAP across IoU thresholds.

#include <oa/core/fnMatrix.h>
#include <oa/runtime/engine.h>
#include <oa/ui/plot/plot.h>
#include <oa/vision/fnDetection.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

[[nodiscard]] oa::Status submitAndWait(oa::Engine& inEngine) {
	auto submitted = inEngine.submit();
	if (not submitted.isOk()) return submitted.getStatus();
	return inEngine.wait(submitted.getValue());
}

template <typename T>
oa::Matrix matrixBytes(const std::vector<T>& inValues, oa::MatrixShape inShape,
	oa::ScalarType inType) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(inValues.data()),
			inValues.size() * sizeof(T)), inShape, inType);
}

oa::Matrix matrixF32(const std::vector<oa::F32>& inValues, oa::MatrixShape inShape) {
	return matrixBytes(inValues, inShape, oa::ScalarType::Float32);
}

oa::Matrix matrixI32(const std::vector<oa::I32>& inValues, oa::MatrixShape inShape) {
	return matrixBytes(inValues, inShape, oa::ScalarType::Int32);
}

template <typename T>
bool read(const oa::Matrix& inMatrix, std::vector<T>& out) {
	out.resize(static_cast<std::size_t>(inMatrix.numElements()));
	return oa::FnMatrix::copyToHost(inMatrix, out.data(),
		static_cast<oa::U64>(out.size() * sizeof(T))).isOk();
}

} // namespace

int main(int argc, char** argv) {
	const char* output = argc > 1 ? argv[1] : "/tmp/oa_cv_evaluation.png";
	oa::EngineConfig engineConfig;
	engineConfig.presentationMode = oa::PresentationMode::None;
	engineConfig.selectForThread = true;
	auto engineResult = oa::Engine::create(engineConfig);
	if (!engineResult.isOk()) {
		std::fprintf(stderr, "Engine creation failed: %s\n",
			engineResult.getStatus().toString().cStr());
		return EXIT_FAILURE;
	}

	// Classification confusion matrix: rows are references, columns predictions.
	auto predictedLabels = matrixI32(
		{0, 1, 2, 1, 0, 2, 2, 1, 0, 2, 1, 0}, {12});
	auto targetLabels = matrixI32(
		{0, 1, 2, 2, 0, 2, 1, 1, 0, 2, 1, 0}, {12});
	auto confusion = oa::FnDetection::confusionMatrix(
		predictedLabels, targetLabels, 3);

	// detection fixture spans two images and two classes. The duplicate class-0
	// prediction is a false positive; the remaining boxes are true positives.
	auto predictedBoxes = matrixF32({
		0.20F, 0.20F, 0.20F, 0.20F,
		0.20F, 0.20F, 0.20F, 0.20F,
		0.50F, 0.50F, 0.20F, 0.20F,
		0.80F, 0.80F, 0.20F, 0.20F}, {4, 4});
	auto predictedScores = matrixF32({0.90F, 0.80F, 0.70F, 0.60F}, {4});
	auto predictedClasses = matrixI32({0, 0, 1, 0}, {4});
	auto predictedImages = matrixI32({0, 0, 0, 1}, {4});
	auto targetBoxes = matrixF32({
		0.20F, 0.20F, 0.20F, 0.20F,
		0.80F, 0.80F, 0.20F, 0.20F,
		0.50F, 0.50F, 0.20F, 0.20F}, {3, 4});
	auto targetClasses = matrixI32({0, 0, 1}, {3});
	auto targetImages = matrixI32({0, 1, 0}, {3});
	auto thresholds = matrixF32({0.50F, 0.75F, 0.90F}, {3});
	auto detection = oa::FnDetection::evaluate(
		predictedBoxes, predictedScores, predictedClasses, predictedImages,
		targetBoxes, targetClasses, targetImages, thresholds, 2, 0.0F);

	auto& engine = *engineResult.getValue();
	if (auto status = submitAndWait(engine); status.isError()) {
		std::fprintf(stderr, "Evaluation submission failed: %s\n",
			status.toString().cStr());
		return EXIT_FAILURE;
	}

	std::vector<oa::U32> confusionU32;
	std::vector<oa::F32> perClass;
	std::vector<oa::F32> map;
	std::vector<oa::F32> meanMap;
	if (!read(confusion, confusionU32) || !read(detection.perClass, perClass)
		|| !read(detection.meanAveragePrecisionByThreshold, map)
		|| !read(detection.meanAveragePrecision, meanMap)) {
		std::fprintf(stderr, "Evaluation readback failed\n");
		return EXIT_FAILURE;
	}
	std::vector<oa::F32> confusionF32(confusionU32.begin(), confusionU32.end());

	const std::array<oa::F32, 10> trainLoss{
		1.20F, 0.94F, 0.78F, 0.65F, 0.55F,
		0.48F, 0.42F, 0.38F, 0.35F, 0.33F};
	const std::array<oa::F32, 10> valLoss{
		1.25F, 1.01F, 0.84F, 0.72F, 0.63F,
		0.57F, 0.52F, 0.49F, 0.47F, 0.46F};

	oa::plot::Figure figure({
		.title = "OA CV evaluation",
		.rows = 2,
		.cols = 2,
		.width = 960,
		.height = 640,
		.hSpacing = 18,
		.vSpacing = 18,
		.padding = 18,
		.background = {0.035F, 0.035F, 0.045F, 1.0F},
	});
	figure.ax(0, 0).title("training loss");
	figure.ax(0, 0).plot(trainLoss);
	figure.ax(0, 1).title("Validation loss");
	figure.ax(0, 1).plot(valLoss,
		{.color = {0.16F, 0.78F, 0.67F, 1.0F}});
	figure.ax(1, 0).title("Classification confusion");
	figure.ax(1, 0).heatmap(
		oa::Span<const oa::F32>(confusionF32.data(), confusionF32.size()), 3, 3,
		{.colormap = 1, .showGrid = true});
	figure.ax(1, 1).title("detection mAP by IoU");
	figure.ax(1, 1).plot(oa::Span<const oa::F32>(map.data(), map.size()),
		{.color = {0.96F, 0.64F, 0.20F, 1.0F}});

	if (auto status = figure.saveTo(output); status.isError()) {
		std::fprintf(stderr, "saveTo failed: %s\n", status.toString().cStr());
		return EXIT_FAILURE;
	}

	std::printf("OA CV evaluation\n");
	std::printf("  class 0: precision %.3f · recall %.3f · F1 %.3f · AP %.3f\n",
		perClass[0], perClass[1], perClass[2], perClass[3]);
	std::printf("  class 1: precision %.3f · recall %.3f · F1 %.3f · AP %.3f\n",
		perClass[4], perClass[5], perClass[6], perClass[7]);
	std::printf("  mAP@[.50:.90]: %.3f\n", meanMap[0]);
	std::printf("  figure: %s\n", output);
	return EXIT_SUCCESS;
}
