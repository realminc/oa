#include "../../oaTest.h"

#include <oa/vision/fnDetection.h>

#include <vector>

namespace {

oa::Matrix matrixF32(const std::vector<oa::F32>& inValues, oa::MatrixShape inShape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(inValues.data()),
			inValues.size() * sizeof(oa::F32)), inShape, oa::ScalarType::Float32);
}

oa::Matrix matrixI32(const std::vector<oa::I32>& inValues, oa::MatrixShape inShape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(inValues.data()),
			inValues.size() * sizeof(oa::I32)), inShape, oa::ScalarType::Int32);
}

oa::Matrix matrixU8(const std::vector<oa::U8>& inValues, oa::MatrixShape inShape) {
	return oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(inValues.data(), inValues.size()),
		inShape, oa::ScalarType::UInt8);
}

void submitAndWait() {
	auto& context = oa::ExecutionSession::getActive();
	ASSERT_TRUE(testSubmitAndWait(context).isOk());
}

template <typename T>
std::vector<T> read(const oa::Matrix& inMatrix) {
	std::vector<T> result(static_cast<oa::Usize>(inMatrix.numElements()));
	EXPECT_TRUE(oa::FnMatrix::copyToHost(inMatrix, result.data(),
		result.size() * sizeof(T)).isOk());
	return result;
}

} // namespace

TEST_F(VkEngineTestFixture, DetectionBoxIouMatchesCpuReference) {
	auto a = matrixF32({
		0.5F, 0.5F, 1.0F, 1.0F,
		0.0F, 0.0F, 2.0F, 2.0F}, {2, 4});
	auto b = matrixF32({
		0.5F, 0.5F, 1.0F, 1.0F,
		1.0F, 0.5F, 1.0F, 1.0F}, {2, 4});
	auto result = oa::FnDetection::boxIou(a, b);
	submitAndWait();
	auto values = read<oa::F32>(result);
	ASSERT_EQ(values.size(), 4U);
	EXPECT_NEAR(values[0], 1.0F, 1.0e-6F);
	EXPECT_NEAR(values[1], 1.0F / 3.0F, 1.0e-6F);
	EXPECT_NEAR(values[2], 0.25F, 1.0e-6F);
	EXPECT_NEAR(values[3], 1.0F / 9.0F, 1.0e-6F);
}

TEST_F(VkEngineTestFixture, DetectionNmsIsClassAwareAndDeterministic) {
	auto boxes = matrixF32({
		0.50F, 0.50F, 0.40F, 0.40F,
		0.51F, 0.50F, 0.40F, 0.40F,
		0.50F, 0.50F, 0.40F, 0.40F,
		0.10F, 0.10F, 0.10F, 0.10F}, {4, 4});
	auto scores = matrixF32({0.90F, 0.80F, 0.85F, 0.70F}, {4});
	auto classes = matrixI32({0, 0, 1, 0}, {4});
	oa::NmsConfig config;
	config.iouThreshold = 0.5F;
	config.maxDetections = 4;
	auto result = oa::FnDetection::nms(boxes, scores, classes, config);
	submitAndWait();
	ASSERT_TRUE(result.isValid());
	auto count = read<oa::U32>(result.count);
	auto indices = read<oa::I32>(result.indices);
	ASSERT_EQ(count[0], 3U);
	EXPECT_EQ(indices[0], 0);
	EXPECT_EQ(indices[1], 2);
	EXPECT_EQ(indices[2], 3);
}

TEST_F(VkEngineTestFixture, DetectionConfusionAndMaskCountsMatchCpu) {
	auto predicted = matrixI32({0, 1, 2, 1, -1}, {5});
	auto target = matrixI32({0, 2, 2, 1, 0}, {5});
	auto confusion = oa::FnDetection::confusionMatrix(predicted, target, 3);
	auto maskPredicted = matrixU8({1, 1, 0, 0, 3}, {5});
	auto maskTarget = matrixU8({1, 0, 1, 0, 1}, {5});
	auto counts = oa::FnDetection::binaryMaskCounts(maskPredicted, maskTarget);
	submitAndWait();
	EXPECT_EQ(read<oa::U32>(confusion), (std::vector<oa::U32>{
		1, 0, 0,
		0, 1, 0,
		0, 1, 1}));
	EXPECT_EQ(read<oa::U32>(counts), (std::vector<oa::U32>{2, 1, 1, 1}));
}

TEST_F(VkEngineTestFixture, DetectionMetricsMatchDatasetCpuOracle) {
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
	auto thresholds = matrixF32({0.50F, 0.75F}, {2});
	auto result = oa::FnDetection::evaluate(
		predictedBoxes, predictedScores, predictedClasses, predictedImages,
		targetBoxes, targetClasses, targetImages, thresholds, 2, 0.75F);
	submitAndWait();
	ASSERT_TRUE(result.isValid());
	EXPECT_EQ(read<oa::U32>(result.counts), (std::vector<oa::U32>{
		1, 1, 1,  0, 0, 1,
		1, 1, 1,  0, 0, 1}));
	const auto perClass = read<oa::F32>(result.perClass);
	ASSERT_EQ(perClass.size(), 16U);
	for (oa::U32 threshold = 0; threshold < 2; ++threshold) {
		const oa::U32 base = threshold * 8U;
		EXPECT_NEAR(perClass[base + 0], 0.5F, 1.0e-6F);
		EXPECT_NEAR(perClass[base + 1], 0.5F, 1.0e-6F);
		EXPECT_NEAR(perClass[base + 2], 0.5F, 1.0e-6F);
		EXPECT_NEAR(perClass[base + 3], 84.333333F / 101.0F, 1.0e-5F);
		EXPECT_NEAR(perClass[base + 4], 0.0F, 1.0e-6F);
		EXPECT_NEAR(perClass[base + 5], 0.0F, 1.0e-6F);
		EXPECT_NEAR(perClass[base + 6], 0.0F, 1.0e-6F);
		EXPECT_NEAR(perClass[base + 7], 1.0F, 1.0e-6F);
	}
	for (const oa::F32 map : read<oa::F32>(
		result.meanAveragePrecisionByThreshold)) {
		EXPECT_NEAR(map, 0.9174917F, 1.0e-5F);
	}
	EXPECT_NEAR(read<oa::F32>(result.meanAveragePrecision)[0],
		0.9174917F, 1.0e-5F);
}

TEST_F(VkEngineTestFixture, SegmentationMetricsMatchCpuOracle) {
	auto predicted = matrixI32({0, 1, 2, 1}, {2, 2});
	auto target = matrixI32({0, 2, 2, 1}, {2, 2});
	auto result = oa::FnDetection::evaluateSegmentation(predicted, target, 3);
	submitAndWait();
	ASSERT_TRUE(result.isValid());
	EXPECT_EQ(read<oa::U32>(result.confusion), (std::vector<oa::U32>{
		1, 0, 0,
		0, 1, 0,
		0, 1, 1}));
	const auto perClass = read<oa::F32>(result.perClass);
	ASSERT_EQ(perClass.size(), 12U);
	EXPECT_NEAR(perClass[0], 1.0F, 1.0e-6F);
	EXPECT_NEAR(perClass[1], 1.0F, 1.0e-6F);
	EXPECT_NEAR(perClass[2], 1.0F, 1.0e-6F);
	EXPECT_NEAR(perClass[3], 1.0F, 1.0e-6F);
	EXPECT_NEAR(perClass[4], 0.5F, 1.0e-6F);
	EXPECT_NEAR(perClass[5], 1.0F, 1.0e-6F);
	EXPECT_NEAR(perClass[6], 2.0F / 3.0F, 1.0e-6F);
	EXPECT_NEAR(perClass[7], 0.5F, 1.0e-6F);
	EXPECT_NEAR(perClass[8], 1.0F, 1.0e-6F);
	EXPECT_NEAR(perClass[9], 0.5F, 1.0e-6F);
	EXPECT_NEAR(perClass[10], 2.0F / 3.0F, 1.0e-6F);
	EXPECT_NEAR(perClass[11], 0.5F, 1.0e-6F);
	EXPECT_NEAR(read<oa::F32>(result.meanIou)[0], 2.0F / 3.0F, 1.0e-6F);
	EXPECT_NEAR(read<oa::F32>(result.pixelAccuracy)[0], 0.75F, 1.0e-6F);

	auto image = matrixF32({0, 0, 0, 0, 0, 0}, {1, 3, 1, 2});
	auto mask = matrixI32({0, 1}, {1, 1, 1, 2});
	auto palette = matrixF32({1, 0, 0,  0, 1, 0}, {2, 3});
	auto overlay = oa::FnImage::segmentationOverlay(image, mask, palette, 0.5F);
	submitAndWait();
	EXPECT_EQ(read<oa::F32>(overlay), (std::vector<oa::F32>{
		0.5F, 0.0F,
		0.0F, 0.5F,
		0.0F, 0.0F}));
}

TEST_F(VkEngineTestFixture, DetectionFunctionsRejectInvalidContracts) {
	auto invalidBoxes = matrixF32({0, 0, 1}, {1, 3});
	auto validBoxes = matrixF32({0, 0, 1, 1}, {1, 4});
	EXPECT_FALSE(oa::FnDetection::boxIou(invalidBoxes, validBoxes).hasStorage());
	EXPECT_FALSE(oa::FnDetection::nms(validBoxes,
		matrixF32({1}, {1}), matrixI32({0}, {1}),
		oa::NmsConfig{.iouThreshold = 2.0F}).isValid());
	EXPECT_FALSE(oa::FnDetection::evaluate(
		validBoxes, matrixF32({1}, {1}), matrixI32({0}, {1}),
		matrixI32({0}, {1}), validBoxes, matrixI32({0}, {1}),
		matrixI32({0}, {1}), matrixI32({0}, {1}), 1).isValid());
	EXPECT_FALSE(oa::FnDetection::evaluateSegmentation(
		matrixF32({0}, {1}), matrixF32({0}, {1}), 1).isValid());
}
