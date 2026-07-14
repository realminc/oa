// CPU-oracle and contract coverage for the public 50-operation OaFnImage surface.

#include "../../../OaTest.h"
#include <Oa/Vision.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

OaMatrix MakeMatrix(OaMatrixShape InShape, const std::vector<float>& InValues)
{
	auto matrix = OaFnMatrix::Empty(InShape);
	for (OaI64 i = 0; i < static_cast<OaI64>(InValues.size()); ++i) {
		matrix.Set(i, InValues[static_cast<OaUsize>(i)]);
	}
	return matrix;
}

std::vector<float> Read(const OaMatrix& InMatrix)
{
	std::vector<float> result(static_cast<OaUsize>(InMatrix.NumElements()));
	for (OaI64 i = 0; i < InMatrix.NumElements(); ++i) {
		result[static_cast<OaUsize>(i)] = InMatrix.At(i);
	}
	return result;
}

class Vision50Ops : public OaVkEngineTestFixture
{
protected:
	void Materialize()
	{
		auto& context = OaContext::GetDefault();
		ASSERT_TRUE(context.Execute().IsOk());
		ASSERT_TRUE(context.Sync().IsOk());
	}

	void ExpectNear(const OaMatrix& InActual, const std::vector<float>& InExpected,
		float InTolerance = 1.0e-5F)
	{
		ASSERT_EQ(InActual.NumElements(), static_cast<OaI64>(InExpected.size()));
		const auto actual = Read(InActual);
		for (OaUsize i = 0; i < actual.size(); ++i) {
			EXPECT_NEAR(actual[i], InExpected[i], InTolerance) << "element " << i;
		}
	}
};

} // namespace

TEST_F(Vision50Ops, PointwiseIntensityFamilyMatchesCpu)
{
	auto input = MakeMatrix({1, 1, 1, 5}, {-0.25F, 0.25F, 0.5F, 0.75F, 1.25F});
	auto binary = OaFnImage::ThresholdBinary(input, 0.5F, 2.0F);
	auto binaryInv = OaFnImage::ThresholdBinaryInv(input, 0.5F, 2.0F);
	auto truncate = OaFnImage::ThresholdTruncate(input, 0.5F);
	auto toZero = OaFnImage::ThresholdToZero(input, 0.5F);
	auto toZeroInv = OaFnImage::ThresholdToZeroInv(input, 0.5F);
	auto range = OaFnImage::InRange(input, 0.25F, 0.75F, 3.0F);
	auto clamp = OaFnImage::Clamp(input, 0.0F, 1.0F);
	auto invert = OaFnImage::Invert(input, 1.0F);
	auto adjusted = OaFnImage::BrightnessContrast(input, 0.1F, 2.0F);
	auto gamma = OaFnImage::GammaContrast(clamp, 2.0F, 1.0F);
	auto solar = OaFnImage::Solarize(clamp, 0.5F, 1.0F);
	auto poster = OaFnImage::Posterize(clamp, 3, 0.0F, 1.0F);
	Materialize();

	ExpectNear(binary, {0, 0, 0, 2, 2});
	ExpectNear(binaryInv, {2, 2, 2, 0, 0});
	ExpectNear(truncate, {-0.25F, 0.25F, 0.5F, 0.5F, 0.5F});
	ExpectNear(toZero, {0, 0, 0, 0.75F, 1.25F});
	ExpectNear(toZeroInv, {-0.25F, 0.25F, 0.5F, 0, 0});
	ExpectNear(range, {0, 3, 3, 3, 0});
	ExpectNear(clamp, {0, 0.25F, 0.5F, 0.75F, 1});
	ExpectNear(invert, {1.25F, 0.75F, 0.5F, 0.25F, -0.25F});
	ExpectNear(adjusted, {-0.4F, 0.6F, 1.1F, 1.6F, 2.6F});
	ExpectNear(gamma, {0, 0.0625F, 0.25F, 0.5625F, 1});
	ExpectNear(solar, {0, 0.25F, 0.5F, 0.25F, 0});
	ExpectNear(poster, {0, 0.5F, 0.5F, 1, 1});
}

TEST_F(Vision50Ops, ColorCompositeAndNoiseFamilyMatchesContracts)
{
	auto rgb = MakeMatrix({1, 3, 1, 2}, {1, 0, 0, 1, 0, 0});
	auto other = MakeMatrix({1, 3, 1, 2}, {0, 1, 1, 0, 0.5F, 0.5F});
	auto mask = MakeMatrix({1, 1, 1, 2}, {0, 1});
	auto transform = MakeMatrix({3, 4}, {
		0, 1, 0, 0.1F,
		1, 0, 0, 0.2F,
		0, 0, 1, 0.3F});
	auto gray = OaFnImage::Grayscale(rgb);
	auto reordered = OaFnImage::ChannelReorder(rgb, 2, 1, 0);
	auto blended = OaFnImage::AlphaBlend(rgb, other, 0.25F);
	auto composite = OaFnImage::Composite(rgb, other, mask);
	auto erased = OaFnImage::Erase(rgb, 1, 0, 1, 1, -1.0F);
	auto twisted = OaFnImage::ColorTwist(rgb, transform);
	auto gaussian = OaFnImage::GaussianNoise(rgb, 0.25F, 0.0F, 7);
	auto saltPepperA = OaFnImage::SaltPepperNoise(rgb, 0.5F, 2.0F, -2.0F, 17);
	auto saltPepperB = OaFnImage::SaltPepperNoise(rgb, 0.5F, 2.0F, -2.0F, 17);
	Materialize();

	ExpectNear(gray, {0.2126F, 0.7152F}, 1.0e-4F);
	ExpectNear(reordered, {0, 0, 0, 1, 1, 0});
	ExpectNear(blended, {0.75F, 0.25F, 0.25F, 0.75F, 0.125F, 0.125F});
	ExpectNear(composite, {1, 1, 0, 0, 0, 0.5F});
	ExpectNear(erased, {1, -1, 0, -1, 0, -1});
	ExpectNear(twisted, {0.1F, 1.1F, 1.2F, 0.2F, 0.3F, 0.3F});
	ExpectNear(gaussian, {1.25F, 0.25F, 0.25F, 1.25F, 0.25F, 0.25F});
	ExpectNear(saltPepperA, Read(saltPepperB));
}

TEST_F(Vision50Ops, NeighborhoodAndComposedFiltersAreNumericallySound)
{
	auto impulse = MakeMatrix({1, 1, 3, 3}, {0, 0, 0, 0, 1, 0, 0, 0, 0});
	auto constant = OaFnMatrix::Full({1, 1, 5, 5}, 0.4F);
	auto median = OaFnImage::MedianBlur(impulse, 3, OaBorderMode::Replicate);
	auto bilateral = OaFnImage::BilateralFilter(constant, 3, 0.1F, 1.0F);
	auto sharpen = OaFnImage::Sharpen(constant);
	auto unsharp = OaFnImage::UnsharpMask(constant, 1.0F, 2.0F, 3);
	auto topHat = OaFnImage::MorphologyTopHat(impulse, 3, 3,
		OaBorderMode::Replicate);
	auto blackHat = OaFnImage::MorphologyBlackHat(impulse, 3, 3,
		OaBorderMode::Replicate);
	auto adaptiveMean = OaFnImage::AdaptiveThresholdMean(impulse, 3, 0.0F, 1.0F,
		OaBorderMode::Replicate);
	auto adaptiveGaussian = OaFnImage::AdaptiveThresholdGaussian(
		impulse, 3, 0.0F, 1.0F, 1.0F, OaBorderMode::Replicate);
	Materialize();

	ExpectNear(median, std::vector<float>(9, 0.0F));
	ExpectNear(bilateral, std::vector<float>(25, 0.4F), 2.0e-5F);
	ExpectNear(sharpen, std::vector<float>(25, 0.4F), 2.0e-5F);
	ExpectNear(unsharp, std::vector<float>(25, 0.4F), 2.0e-5F);
	EXPECT_NEAR(Read(topHat)[4], 1.0F, 1.0e-5F);
	EXPECT_NEAR(Read(blackHat)[4], 0.0F, 1.0e-5F);
	EXPECT_NEAR(Read(adaptiveMean)[4], 1.0F, 1.0e-5F);
	EXPECT_NEAR(Read(adaptiveGaussian)[4], 1.0F, 1.0e-5F);
}

TEST_F(Vision50Ops, PadCropRemapAndWarpsMatchExactCpuCoordinates)
{
	auto image = MakeMatrix({1, 1, 2, 3}, {0, 1, 2, 3, 4, 5});
	auto map = MakeMatrix({1, 2, 2, 3}, {
		0, 1, 2, 0, 1, 2,
		0, 0, 0, 1, 1, 1});
	auto affine = MakeMatrix({2, 3}, {1, 0, 0, 0, 1, 0});
	auto perspective = MakeMatrix({3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});
	auto padded = OaFnImage::Pad(image, 1, 1, 1, 1,
		OaBorderMode::Constant, -1.0F);
	auto centered = OaFnImage::CenterCrop(padded, 3, 2);
	auto remapped = OaFnImage::Remap(image, map, OaInterpolationMode::Nearest);
	auto warpedAffine = OaFnImage::WarpAffine(image, affine, 3, 2,
		OaInterpolationMode::Nearest);
	auto warpedPerspective = OaFnImage::WarpPerspective(image, perspective, 3, 2,
		OaInterpolationMode::Nearest);
	Materialize();

	EXPECT_EQ(padded.GetShape(), OaMatrixShape({1, 1, 4, 5}));
	ExpectNear(padded, {
		-1, -1, -1, -1, -1,
		-1, 0, 1, 2, -1,
		-1, 3, 4, 5, -1,
		-1, -1, -1, -1, -1});
	ExpectNear(centered, {0, 1, 2, 3, 4, 5});
	ExpectNear(remapped, {0, 1, 2, 3, 4, 5});
	ExpectNear(warpedAffine, {0, 1, 2, 3, 4, 5});
	ExpectNear(warpedPerspective, {0, 1, 2, 3, 4, 5});
}

TEST_F(Vision50Ops, InvalidParametersAreNoOpContracts)
{
	auto input = OaFnMatrix::Full({1, 1, 3, 3}, 0.5F);
	EXPECT_EQ(OaFnImage::Posterize(input, 1).GetVkBuffer().Buffer,
		input.GetVkBuffer().Buffer);
	EXPECT_EQ(OaFnImage::MedianBlur(input, 4).GetVkBuffer().Buffer,
		input.GetVkBuffer().Buffer);
	EXPECT_EQ(OaFnImage::MorphologyTopHat(input, 2, 3).GetVkBuffer().Buffer,
		input.GetVkBuffer().Buffer);
	EXPECT_EQ(OaFnImage::CenterCrop(input, 4, 2).GetVkBuffer().Buffer,
		input.GetVkBuffer().Buffer);
}
