// CPU-oracle and contract coverage for the public 50-operation oa::FnImage surface.

#include "../../../oaTest.h"
#include <oa/core/matrixAccess.h>
#include <oa/core/op.h>
#include <oa/runtime/executableGraph.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/semanticGraph.h>
#include <oa/vision.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

oa::Matrix makeMatrix(oa::MatrixShape inShape, const std::vector<float>& inValues)
{
	auto matrix = oa::FnMatrix::empty(inShape);
	for (oa::I64 i = 0; i < static_cast<oa::I64>(inValues.size()); ++i) {
		matrix.set(i, inValues[static_cast<oa::Usize>(i)]);
	}
	return matrix;
}

std::vector<float> read(const oa::Matrix& inMatrix)
{
	std::vector<float> result(static_cast<oa::Usize>(inMatrix.numElements()));
	for (oa::I64 i = 0; i < inMatrix.numElements(); ++i) {
		result[static_cast<oa::Usize>(i)] = inMatrix.at(i);
	}
	return result;
}

class Vision50Ops : public VkEngineTestFixture
{
protected:
	void materialize()
	{
		auto& context = oa::ExecutionSession::getActive();
		ASSERT_TRUE(testSubmitAndWait(context).isOk());
	}

	void expectNear(const oa::Matrix& inActual, const std::vector<float>& inExpected,
		float inTolerance = 1.0e-5F)
	{
		ASSERT_EQ(inActual.numElements(), static_cast<oa::I64>(inExpected.size()));
		const auto actual = read(inActual);
		for (oa::Usize i = 0; i < actual.size(); ++i) {
			EXPECT_NEAR(actual[i], inExpected[i], inTolerance) << "element " << i;
		}
	}
};

} // namespace

TEST_F(Vision50Ops, PointwiseIntensityFamilyMatchesCpu)
{
	auto input = makeMatrix({1, 1, 1, 5}, {-0.25F, 0.25F, 0.5F, 0.75F, 1.25F});
	auto binary = oa::FnImage::thresholdBinary(input, 0.5F, 2.0F);
	auto binaryInv = oa::FnImage::thresholdBinaryInv(input, 0.5F, 2.0F);
	auto truncate = oa::FnImage::thresholdTruncate(input, 0.5F);
	auto toZero = oa::FnImage::thresholdToZero(input, 0.5F);
	auto toZeroInv = oa::FnImage::thresholdToZeroInv(input, 0.5F);
	auto range = oa::FnImage::inRange(input, 0.25F, 0.75F, 3.0F);
	auto clamp = oa::FnImage::clamp(input, 0.0F, 1.0F);
	auto invert = oa::FnImage::invert(input, 1.0F);
	auto adjusted = oa::FnImage::brightnessContrast(input, 0.1F, 2.0F);
	auto gamma = oa::FnImage::gammaContrast(clamp, 2.0F, 1.0F);
	auto solar = oa::FnImage::solarize(clamp, 0.5F, 1.0F);
	auto poster = oa::FnImage::posterize(clamp, 3, 0.0F, 1.0F);
	materialize();

	expectNear(binary, {0, 0, 0, 2, 2});
	expectNear(binaryInv, {2, 2, 2, 0, 0});
	expectNear(truncate, {-0.25F, 0.25F, 0.5F, 0.5F, 0.5F});
	expectNear(toZero, {0, 0, 0, 0.75F, 1.25F});
	expectNear(toZeroInv, {-0.25F, 0.25F, 0.5F, 0, 0});
	expectNear(range, {0, 3, 3, 3, 0});
	expectNear(clamp, {0, 0.25F, 0.5F, 0.75F, 1});
	expectNear(invert, {1.25F, 0.75F, 0.5F, 0.25F, -0.25F});
	expectNear(adjusted, {-0.4F, 0.6F, 1.1F, 1.6F, 2.6F});
	expectNear(gamma, {0, 0.0625F, 0.25F, 0.5625F, 1});
	expectNear(solar, {0, 0.25F, 0.5F, 0.25F, 0});
	expectNear(poster, {0, 0.5F, 0.5F, 1, 1});
}

TEST_F(Vision50Ops, ColorCompositeAndNoiseFamilyMatchesContracts)
{
	auto rgb = makeMatrix({1, 3, 1, 2}, {1, 0, 0, 1, 0, 0});
	auto other = makeMatrix({1, 3, 1, 2}, {0, 1, 1, 0, 0.5F, 0.5F});
	auto mask = makeMatrix({1, 1, 1, 2}, {0, 1});
	auto transform = makeMatrix({3, 4}, {
		0, 1, 0, 0.1F,
		1, 0, 0, 0.2F,
		0, 0, 1, 0.3F});
	auto gray = oa::FnImage::grayscale(rgb);
	auto reordered = oa::FnImage::channelReorder(rgb, 2, 1, 0);
	auto blended = oa::FnImage::alphaBlend(rgb, other, 0.25F);
	auto composite = oa::FnImage::composite(rgb, other, mask);
	auto erased = oa::FnImage::erase(rgb, 1, 0, 1, 1, -1.0F);
	auto twisted = oa::FnImage::colorTwist(rgb, transform);
	auto gaussian = oa::FnImage::gaussianNoise(rgb, 0.25F, 0.0F, 7);
	auto saltPepperA = oa::FnImage::saltPepperNoise(rgb, 0.5F, 2.0F, -2.0F, 17);
	auto saltPepperB = oa::FnImage::saltPepperNoise(rgb, 0.5F, 2.0F, -2.0F, 17);
	materialize();

	expectNear(gray, {0.2126F, 0.7152F}, 1.0e-4F);
	expectNear(reordered, {0, 0, 0, 1, 1, 0});
	expectNear(blended, {0.75F, 0.25F, 0.25F, 0.75F, 0.125F, 0.125F});
	expectNear(composite, {1, 1, 0, 0, 0, 0.5F});
	expectNear(erased, {1, -1, 0, -1, 0, -1});
	expectNear(twisted, {0.1F, 1.1F, 1.2F, 0.2F, 0.3F, 0.3F});
	expectNear(gaussian, {1.25F, 0.25F, 0.25F, 1.25F, 0.25F, 0.25F});
	expectNear(saltPepperA, read(saltPepperB));
}

TEST_F(Vision50Ops, NeighborhoodAndComposedFiltersAreNumericallySound)
{
	auto impulse = makeMatrix({1, 1, 3, 3}, {0, 0, 0, 0, 1, 0, 0, 0, 0});
	auto constant = oa::FnMatrix::full({1, 1, 5, 5}, 0.4F);
	auto median = oa::FnImage::medianBlur(impulse, 3, oa::BorderMode::Replicate);
	auto bilateral = oa::FnImage::bilateralFilter(constant, 3, 0.1F, 1.0F);
	auto sharpen = oa::FnImage::sharpen(constant);
	auto unsharp = oa::FnImage::unsharpMask(constant, 1.0F, 2.0F, 3);
	auto topHat = oa::FnImage::morphologyTopHat(impulse, 3, 3,
		oa::BorderMode::Replicate);
	auto blackHat = oa::FnImage::morphologyBlackHat(impulse, 3, 3,
		oa::BorderMode::Replicate);
	auto adaptiveMean = oa::FnImage::adaptiveThresholdMean(impulse, 3, 0.0F, 1.0F,
		oa::BorderMode::Replicate);
	auto adaptiveGaussian = oa::FnImage::adaptiveThresholdGaussian(
		impulse, 3, 0.0F, 1.0F, 1.0F, oa::BorderMode::Replicate);
	materialize();

	expectNear(median, std::vector<float>(9, 0.0F));
	expectNear(bilateral, std::vector<float>(25, 0.4F), 2.0e-5F);
	expectNear(sharpen, std::vector<float>(25, 0.4F), 2.0e-5F);
	expectNear(unsharp, std::vector<float>(25, 0.4F), 2.0e-5F);
	EXPECT_NEAR(read(topHat)[4], 1.0F, 1.0e-5F);
	EXPECT_NEAR(read(blackHat)[4], 0.0F, 1.0e-5F);
	EXPECT_NEAR(read(adaptiveMean)[4], 1.0F, 1.0e-5F);
	EXPECT_NEAR(read(adaptiveGaussian)[4], 1.0F, 1.0e-5F);
}

TEST_F(Vision50Ops, PadCropRemapAndWarpsMatchExactCpuCoordinates)
{
	auto image = makeMatrix({1, 1, 2, 3}, {0, 1, 2, 3, 4, 5});
	auto map = makeMatrix({1, 2, 2, 3}, {
		0, 1, 2, 0, 1, 2,
		0, 0, 0, 1, 1, 1});
	auto affine = makeMatrix({2, 3}, {1, 0, 0, 0, 1, 0});
	auto perspective = makeMatrix({3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});
	auto padded = oa::FnImage::pad(image, 1, 1, 1, 1,
		oa::BorderMode::Constant, -1.0F);
	auto centered = oa::FnImage::centerCrop(padded, 3, 2);
	auto remapped = oa::FnImage::remap(image, map, oa::InterpolationMode::Nearest);
	auto warpedAffine = oa::FnImage::warpAffine(image, affine, 3, 2,
		oa::InterpolationMode::Nearest);
	auto warpedPerspective = oa::FnImage::warpPerspective(image, perspective, 3, 2,
		oa::InterpolationMode::Nearest);
	materialize();

	EXPECT_EQ(padded.getShape(), oa::MatrixShape({1, 1, 4, 5}));
	expectNear(padded, {
		-1, -1, -1, -1, -1,
		-1, 0, 1, 2, -1,
		-1, 3, 4, 5, -1,
		-1, -1, -1, -1, -1});
	expectNear(centered, {0, 1, 2, 3, 4, 5});
	expectNear(remapped, {0, 1, 2, 3, 4, 5});
	expectNear(warpedAffine, {0, 1, 2, 3, 4, 5});
	expectNear(warpedPerspective, {0, 1, 2, 3, 4, 5});
}

TEST_F(Vision50Ops, InvalidParametersAreNoOpContracts)
{
	auto input = oa::FnMatrix::full({1, 1, 3, 3}, 0.5F);
	EXPECT_EQ(oa::MatrixAccess::descriptor(oa::FnImage::posterize(input, 1)).buffer,
		oa::MatrixAccess::descriptor(input).buffer);
	EXPECT_EQ(oa::MatrixAccess::descriptor(oa::FnImage::medianBlur(input, 4)).buffer,
		oa::MatrixAccess::descriptor(input).buffer);
	EXPECT_EQ(oa::MatrixAccess::descriptor(oa::FnImage::morphologyTopHat(input, 2, 3)).buffer,
		oa::MatrixAccess::descriptor(input).buffer);
	EXPECT_EQ(oa::MatrixAccess::descriptor(oa::FnImage::centerCrop(input, 4, 2)).buffer,
		oa::MatrixAccess::descriptor(input).buffer);
}

TEST_F(Vision50Ops, ComposedOperationRetainsOneSchemaIdentity)
{
	auto& context = oa::ExecutionSession::getActive();
	auto input = oa::FnMatrix::full({1, 1, 2, 3}, 0.5F);
	context.clear();

	auto output = oa::FnImage::gaussianNoise(input, 0.25F, 0.5F, 7);
	(void)output;

	ASSERT_NE(context.semanticGraph(), nullptr);
	ASSERT_NE(context.graph(), nullptr);
	ASSERT_TRUE(context.semanticGraph()->validate().isOk());
	const auto operations = context.semanticGraph()->operations();
	ASSERT_EQ(operations.size(), 1U);
	EXPECT_EQ(operations[0].name,
		oa::detail::opRegistry::FnImage::gaussianNoise.name);
	EXPECT_EQ(operations[0].contractHash,
		oa::detail::opRegistry::FnImage::gaussianNoise.hash);
	ASSERT_EQ(operations[0].attributes.size(), 3U);
	EXPECT_EQ(operations[0].attributes[0].name, "mean");
	EXPECT_EQ(operations[0].attributes[0].floatVal, 0.25);
	EXPECT_EQ(operations[0].attributes[1].name, "stddev");
	EXPECT_EQ(operations[0].attributes[1].floatVal, 0.5);
	EXPECT_EQ(operations[0].attributes[2].name, "seed");
	EXPECT_EQ(operations[0].attributes[2].unsignedInteger, 7U);

	const auto nodes = context.graph()->nodes();
	ASSERT_EQ(nodes.size(), 2U);
	for (const auto& node : nodes) {
		EXPECT_EQ(node.operation,
			oa::detail::opRegistry::FnImage::gaussianNoise.name);
		EXPECT_EQ(node.opContractHash,
			oa::detail::opRegistry::FnImage::gaussianNoise.hash);
		ASSERT_EQ(node.semanticOps.size(), 1U);
		EXPECT_EQ(node.semanticOps[0], operations[0].id);
	}
	const auto analyzed = oa::analyzeSemanticLowering(
		*context.semanticGraph(), *context.graph());
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	EXPECT_EQ(analyzed.getValue().schemaOwnedNodeCount(), 2U);
	EXPECT_EQ(analyzed.getValue().compatibilityNodeCount(), 0U);
	EXPECT_EQ(analyzed.getValue().directOpCount(), 0U);
	EXPECT_EQ(analyzed.getValue().decomposedOpCount(), 1U);
	EXPECT_EQ(analyzed.getValue().maximumNodesPerOp(), 2U);
	materialize();
}

TEST_F(Vision50Ops, InvalidNoOpDoesNotRecordOrPoisonNextOperation)
{
	auto& context = oa::ExecutionSession::getActive();
	auto input = oa::FnMatrix::full({1, 1, 3, 3}, 0.5F);
	context.clear();

	auto invalid = oa::FnImage::medianBlur(input, 4);
	EXPECT_EQ(oa::MatrixAccess::descriptor(invalid).buffer,
		oa::MatrixAccess::descriptor(input).buffer);
	ASSERT_NE(context.semanticGraph(), nullptr);
	ASSERT_NE(context.graph(), nullptr);
	EXPECT_EQ(context.semanticGraph()->operationCount(), 0U);
	EXPECT_EQ(context.graph()->nodeCount(), 0U);

	auto valid = oa::FnImage::thresholdBinary(input, 0.25F, 1.0F);
	(void)valid;
	ASSERT_EQ(context.semanticGraph()->operationCount(), 1U);
	ASSERT_EQ(context.graph()->nodeCount(), 1U);
	EXPECT_EQ(context.semanticGraph()->operations()[0].name,
		oa::detail::opRegistry::FnImage::thresholdBinary.name);
	EXPECT_EQ(context.graph()->nodes()[0].semanticOps[0], 0U);
	materialize();
}
