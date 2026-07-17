#include "../../OaTest.h"

#include <Oa/Vision/FnDetection.h>

#include <vector>

namespace {

OaMatrix MatrixF32(const std::vector<OaF32>& InValues, OaMatrixShape InShape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(InValues.data()),
			InValues.size() * sizeof(OaF32)), InShape, OaScalarType::Float32);
}

OaMatrix MatrixI32(const std::vector<OaI32>& InValues, OaMatrixShape InShape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(InValues.data()),
			InValues.size() * sizeof(OaI32)), InShape, OaScalarType::Int32);
}

OaMatrix MatrixU8(const std::vector<OaU8>& InValues, OaMatrixShape InShape) {
	return OaFnMatrix::FromBytes(OaSpan<const OaU8>(InValues.data(), InValues.size()),
		InShape, OaScalarType::UInt8);
}

void Sync() {
	auto& context = OaContext::GetDefault();
	ASSERT_TRUE(context.Execute().IsOk());
	ASSERT_TRUE(context.Sync().IsOk());
}

template <typename T>
std::vector<T> Read(const OaMatrix& InMatrix) {
	std::vector<T> result(static_cast<OaUsize>(InMatrix.NumElements()));
	EXPECT_TRUE(OaFnMatrix::CopyToHost(InMatrix, result.data(),
		result.size() * sizeof(T)).IsOk());
	return result;
}

} // namespace

TEST_F(OaVkEngineTestFixture, DetectionBoxIouMatchesCpuReference) {
	auto a = MatrixF32({
		0.5F, 0.5F, 1.0F, 1.0F,
		0.0F, 0.0F, 2.0F, 2.0F}, {2, 4});
	auto b = MatrixF32({
		0.5F, 0.5F, 1.0F, 1.0F,
		1.0F, 0.5F, 1.0F, 1.0F}, {2, 4});
	auto result = OaFnDetection::BoxIou(a, b);
	Sync();
	auto values = Read<OaF32>(result);
	ASSERT_EQ(values.size(), 4U);
	EXPECT_NEAR(values[0], 1.0F, 1.0e-6F);
	EXPECT_NEAR(values[1], 1.0F / 3.0F, 1.0e-6F);
	EXPECT_NEAR(values[2], 0.25F, 1.0e-6F);
	EXPECT_NEAR(values[3], 1.0F / 9.0F, 1.0e-6F);
}

TEST_F(OaVkEngineTestFixture, DetectionNmsIsClassAwareAndDeterministic) {
	auto boxes = MatrixF32({
		0.50F, 0.50F, 0.40F, 0.40F,
		0.51F, 0.50F, 0.40F, 0.40F,
		0.50F, 0.50F, 0.40F, 0.40F,
		0.10F, 0.10F, 0.10F, 0.10F}, {4, 4});
	auto scores = MatrixF32({0.90F, 0.80F, 0.85F, 0.70F}, {4});
	auto classes = MatrixI32({0, 0, 1, 0}, {4});
	OaNmsConfig config;
	config.IouThreshold = 0.5F;
	config.MaxDetections = 4;
	auto result = OaFnDetection::Nms(boxes, scores, classes, config);
	Sync();
	ASSERT_TRUE(result.IsValid());
	auto count = Read<OaU32>(result.Count);
	auto indices = Read<OaI32>(result.Indices);
	ASSERT_EQ(count[0], 3U);
	EXPECT_EQ(indices[0], 0);
	EXPECT_EQ(indices[1], 2);
	EXPECT_EQ(indices[2], 3);
}

TEST_F(OaVkEngineTestFixture, DetectionConfusionAndMaskCountsMatchCpu) {
	auto predicted = MatrixI32({0, 1, 2, 1, -1}, {5});
	auto target = MatrixI32({0, 2, 2, 1, 0}, {5});
	auto confusion = OaFnDetection::ConfusionMatrix(predicted, target, 3);
	auto maskPredicted = MatrixU8({1, 1, 0, 0, 3}, {5});
	auto maskTarget = MatrixU8({1, 0, 1, 0, 1}, {5});
	auto counts = OaFnDetection::BinaryMaskCounts(maskPredicted, maskTarget);
	Sync();
	EXPECT_EQ(Read<OaU32>(confusion), (std::vector<OaU32>{
		1, 0, 0,
		0, 1, 0,
		0, 1, 1}));
	EXPECT_EQ(Read<OaU32>(counts), (std::vector<OaU32>{2, 1, 1, 1}));
}

TEST_F(OaVkEngineTestFixture, DetectionMetricsMatchDatasetCpuOracle) {
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
	auto thresholds = MatrixF32({0.50F, 0.75F}, {2});
	auto result = OaFnDetection::Evaluate(
		predictedBoxes, predictedScores, predictedClasses, predictedImages,
		targetBoxes, targetClasses, targetImages, thresholds, 2, 0.75F);
	Sync();
	ASSERT_TRUE(result.IsValid());
	EXPECT_EQ(Read<OaU32>(result.Counts), (std::vector<OaU32>{
		1, 1, 1,  0, 0, 1,
		1, 1, 1,  0, 0, 1}));
	const auto perClass = Read<OaF32>(result.PerClass);
	ASSERT_EQ(perClass.size(), 16U);
	for (OaU32 threshold = 0; threshold < 2; ++threshold) {
		const OaU32 base = threshold * 8U;
		EXPECT_NEAR(perClass[base + 0], 0.5F, 1.0e-6F);
		EXPECT_NEAR(perClass[base + 1], 0.5F, 1.0e-6F);
		EXPECT_NEAR(perClass[base + 2], 0.5F, 1.0e-6F);
		EXPECT_NEAR(perClass[base + 3], 84.333333F / 101.0F, 1.0e-5F);
		EXPECT_NEAR(perClass[base + 4], 0.0F, 1.0e-6F);
		EXPECT_NEAR(perClass[base + 5], 0.0F, 1.0e-6F);
		EXPECT_NEAR(perClass[base + 6], 0.0F, 1.0e-6F);
		EXPECT_NEAR(perClass[base + 7], 1.0F, 1.0e-6F);
	}
	for (const OaF32 map : Read<OaF32>(
		result.MeanAveragePrecisionByThreshold)) {
		EXPECT_NEAR(map, 0.9174917F, 1.0e-5F);
	}
	EXPECT_NEAR(Read<OaF32>(result.MeanAveragePrecision)[0],
		0.9174917F, 1.0e-5F);
}

TEST_F(OaVkEngineTestFixture, SegmentationMetricsMatchCpuOracle) {
	auto predicted = MatrixI32({0, 1, 2, 1}, {2, 2});
	auto target = MatrixI32({0, 2, 2, 1}, {2, 2});
	auto result = OaFnDetection::EvaluateSegmentation(predicted, target, 3);
	Sync();
	ASSERT_TRUE(result.IsValid());
	EXPECT_EQ(Read<OaU32>(result.Confusion), (std::vector<OaU32>{
		1, 0, 0,
		0, 1, 0,
		0, 1, 1}));
	const auto perClass = Read<OaF32>(result.PerClass);
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
	EXPECT_NEAR(Read<OaF32>(result.MeanIou)[0], 2.0F / 3.0F, 1.0e-6F);
	EXPECT_NEAR(Read<OaF32>(result.PixelAccuracy)[0], 0.75F, 1.0e-6F);

	auto image = MatrixF32({0, 0, 0, 0, 0, 0}, {1, 3, 1, 2});
	auto mask = MatrixI32({0, 1}, {1, 1, 1, 2});
	auto palette = MatrixF32({1, 0, 0,  0, 1, 0}, {2, 3});
	auto overlay = OaFnImage::SegmentationOverlay(image, mask, palette, 0.5F);
	Sync();
	EXPECT_EQ(Read<OaF32>(overlay), (std::vector<OaF32>{
		0.5F, 0.0F,
		0.0F, 0.5F,
		0.0F, 0.0F}));
}

TEST_F(OaVkEngineTestFixture, DetectionFunctionsRejectInvalidContracts) {
	auto invalidBoxes = MatrixF32({0, 0, 1}, {1, 3});
	auto validBoxes = MatrixF32({0, 0, 1, 1}, {1, 4});
	EXPECT_FALSE(OaFnDetection::BoxIou(invalidBoxes, validBoxes).HasStorage());
	EXPECT_FALSE(OaFnDetection::Nms(validBoxes,
		MatrixF32({1}, {1}), MatrixI32({0}, {1}),
		OaNmsConfig{.IouThreshold = 2.0F}).IsValid());
	EXPECT_FALSE(OaFnDetection::Evaluate(
		validBoxes, MatrixF32({1}, {1}), MatrixI32({0}, {1}),
		MatrixI32({0}, {1}), validBoxes, MatrixI32({0}, {1}),
		MatrixI32({0}, {1}), MatrixI32({0}, {1}), 1).IsValid());
	EXPECT_FALSE(OaFnDetection::EvaluateSegmentation(
		MatrixF32({0}, {1}), MatrixF32({0}, {1}), 1).IsValid());
}
