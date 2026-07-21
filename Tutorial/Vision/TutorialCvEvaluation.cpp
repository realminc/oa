// OA Tutorial: compact computer-vision evaluation.
//
// Runs real GPU classification and detection metrics, reads back only the
// small evaluation artifacts, and saves a presentation-ready figure containing
// training curves, a confusion matrix, and mAP across IoU thresholds.

#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Ui/Plot/Plot.h>
#include <Oa/Vision/FnDetection.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

template <typename T>
OaMatrix MatrixBytes(const std::vector<T>& InValues, OaMatrixShape InShape,
	OaScalarType InType) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(InValues.data()),
			InValues.size() * sizeof(T)), InShape, InType);
}

OaMatrix MatrixF32(const std::vector<OaF32>& InValues, OaMatrixShape InShape) {
	return MatrixBytes(InValues, InShape, OaScalarType::Float32);
}

OaMatrix MatrixI32(const std::vector<OaI32>& InValues, OaMatrixShape InShape) {
	return MatrixBytes(InValues, InShape, OaScalarType::Int32);
}

template <typename T>
bool Read(const OaMatrix& InMatrix, std::vector<T>& Out) {
	Out.resize(static_cast<std::size_t>(InMatrix.NumElements()));
	return OaFnMatrix::CopyToHost(InMatrix, Out.data(),
		static_cast<OaU64>(Out.size() * sizeof(T))).IsOk();
}

} // namespace

int main(int argc, char** argv) {
	const char* output = argc > 1 ? argv[1] : "/tmp/oa_cv_evaluation.png";
	OaEngineConfig engineConfig;
	engineConfig.PresentationMode = OaPresentationMode::None;
	engineConfig.RegisterAsGlobal = true;
	auto engineResult = OaEngine::Create(engineConfig);
	if (!engineResult.IsOk()) {
		std::fprintf(stderr, "Engine creation failed: %s\n",
			engineResult.GetStatus().ToString().c_str());
		return EXIT_FAILURE;
	}

	// Classification confusion matrix: rows are references, columns predictions.
	auto predictedLabels = MatrixI32(
		{0, 1, 2, 1, 0, 2, 2, 1, 0, 2, 1, 0}, {12});
	auto targetLabels = MatrixI32(
		{0, 1, 2, 2, 0, 2, 1, 1, 0, 2, 1, 0}, {12});
	auto confusion = OaFnDetection::ConfusionMatrix(
		predictedLabels, targetLabels, 3);

	// Detection fixture spans two images and two classes. The duplicate class-0
	// prediction is a false positive; the remaining boxes are true positives.
	auto predictedBoxes = MatrixF32({
		0.20F, 0.20F, 0.20F, 0.20F,
		0.20F, 0.20F, 0.20F, 0.20F,
		0.50F, 0.50F, 0.20F, 0.20F,
		0.80F, 0.80F, 0.20F, 0.20F}, {4, 4});
	auto predictedScores = MatrixF32({0.90F, 0.80F, 0.70F, 0.60F}, {4});
	auto predictedClasses = MatrixI32({0, 0, 1, 0}, {4});
	auto predictedImages = MatrixI32({0, 0, 0, 1}, {4});
	auto targetBoxes = MatrixF32({
		0.20F, 0.20F, 0.20F, 0.20F,
		0.80F, 0.80F, 0.20F, 0.20F,
		0.50F, 0.50F, 0.20F, 0.20F}, {3, 4});
	auto targetClasses = MatrixI32({0, 0, 1}, {3});
	auto targetImages = MatrixI32({0, 1, 0}, {3});
	auto thresholds = MatrixF32({0.50F, 0.75F, 0.90F}, {3});
	auto detection = OaFnDetection::Evaluate(
		predictedBoxes, predictedScores, predictedClasses, predictedImages,
		targetBoxes, targetClasses, targetImages, thresholds, 2, 0.0F);

	auto& context = OaContext::GetDefault();
	if (auto status = context.Execute(); status.IsError()) {
		std::fprintf(stderr, "Evaluation execute failed: %s\n",
			status.ToString().c_str());
		return EXIT_FAILURE;
	}
	if (auto status = context.Sync(); status.IsError()) {
		std::fprintf(stderr, "Evaluation sync failed: %s\n",
			status.ToString().c_str());
		return EXIT_FAILURE;
	}

	std::vector<OaU32> confusionU32;
	std::vector<OaF32> perClass;
	std::vector<OaF32> map;
	std::vector<OaF32> meanMap;
	if (!Read(confusion, confusionU32) || !Read(detection.PerClass, perClass)
		|| !Read(detection.MeanAveragePrecisionByThreshold, map)
		|| !Read(detection.MeanAveragePrecision, meanMap)) {
		std::fprintf(stderr, "Evaluation readback failed\n");
		return EXIT_FAILURE;
	}
	std::vector<OaF32> confusionF32(confusionU32.begin(), confusionU32.end());

	const std::array<OaF32, 10> trainLoss{
		1.20F, 0.94F, 0.78F, 0.65F, 0.55F,
		0.48F, 0.42F, 0.38F, 0.35F, 0.33F};
	const std::array<OaF32, 10> valLoss{
		1.25F, 1.01F, 0.84F, 0.72F, 0.63F,
		0.57F, 0.52F, 0.49F, 0.47F, 0.46F};

	OaPlot::Figure figure({
		.Title = "OA CV evaluation",
		.Rows = 2,
		.Cols = 2,
		.Width = 960,
		.Height = 640,
		.HSpacing = 18,
		.VSpacing = 18,
		.Padding = 18,
		.Background = {0.035F, 0.035F, 0.045F, 1.0F},
	});
	figure.Ax(0, 0).Title("Training loss");
	figure.Ax(0, 0).Plot(trainLoss);
	figure.Ax(0, 1).Title("Validation loss");
	figure.Ax(0, 1).Plot(valLoss,
		{.Color = {0.16F, 0.78F, 0.67F, 1.0F}});
	figure.Ax(1, 0).Title("Classification confusion");
	figure.Ax(1, 0).Heatmap(
		OaSpan<const OaF32>(confusionF32.data(), confusionF32.size()), 3, 3,
		{.Colormap = 1, .ShowGrid = true});
	figure.Ax(1, 1).Title("Detection mAP by IoU");
	figure.Ax(1, 1).Plot(OaSpan<const OaF32>(map.data(), map.size()),
		{.Color = {0.96F, 0.64F, 0.20F, 1.0F}});

	if (auto status = figure.SaveFig(output); status.IsError()) {
		std::fprintf(stderr, "SaveFig failed: %s\n", status.ToString().c_str());
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
