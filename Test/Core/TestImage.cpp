// OaImage tests — Core media wrapper composed over OaMatrix.
// Phase 1: Core wrapper (requires Vulkan since OA is GPU-only).

#include "../OaTest.h"

#include <Oa/Core/Image.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Vision/FnImage.h>
#include <Oa/Ui/Image.h>

static void MaterializeVisionGraph() {
	auto& ctx = OaContext::GetDefault();
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
}

// ─── OaImageFormatChannels ───────────────────────────────────────────────────────

TEST(OaImage, FormatChannels) {
	EXPECT_EQ(OaImageFormatChannels(OaImageFormat::Gray), 1);
	EXPECT_EQ(OaImageFormatChannels(OaImageFormat::GrayAlpha), 2);
	EXPECT_EQ(OaImageFormatChannels(OaImageFormat::Rgb), 3);
	EXPECT_EQ(OaImageFormatChannels(OaImageFormat::Bgr), 3);
	EXPECT_EQ(OaImageFormatChannels(OaImageFormat::Rgba), 4);
	EXPECT_EQ(OaImageFormatChannels(OaImageFormat::Bgra), 4);
}

// ─── Construction from OaMatrix ───────────────────────────────────────────────────

TEST(OaImage, ConstructNchwRgb) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{1, 3, 224, 224});
	OaImage img(std::move(data), OaImageLayout::Nchw, OaImageFormat::Rgb);
	EXPECT_TRUE(img.Validate());
	EXPECT_EQ(img.Layout(), OaImageLayout::Nchw);
	EXPECT_EQ(img.Format(), OaImageFormat::Rgb);
	EXPECT_EQ(img.Width(), 224);
	EXPECT_EQ(img.Height(), 224);
	EXPECT_EQ(img.Channels(), 3);
	EXPECT_EQ(img.BatchSize(), 1);
}

TEST(OaImage, ConstructNhwcRgba) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{2, 224, 224, 4});
	OaImage img(std::move(data), OaImageLayout::Nhwc, OaImageFormat::Rgba);
	EXPECT_TRUE(img.Validate());
	EXPECT_EQ(img.Layout(), OaImageLayout::Nhwc);
	EXPECT_EQ(img.Format(), OaImageFormat::Rgba);
	EXPECT_EQ(img.Width(), 224);
	EXPECT_EQ(img.Height(), 224);
	EXPECT_EQ(img.Channels(), 4);
	EXPECT_EQ(img.BatchSize(), 2);
}

TEST(OaImage, ConstructChwBgr) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{3, 128, 128});
	OaImage img(std::move(data), OaImageLayout::Chw, OaImageFormat::Bgr);
	EXPECT_TRUE(img.Validate());
	EXPECT_EQ(img.Layout(), OaImageLayout::Chw);
	EXPECT_EQ(img.Format(), OaImageFormat::Bgr);
	EXPECT_EQ(img.Width(), 128);
	EXPECT_EQ(img.Height(), 128);
	EXPECT_EQ(img.Channels(), 3);
	EXPECT_EQ(img.BatchSize(), 1);
}

TEST(OaImage, ConstructHwcGray) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{64, 64, 1});
	OaImage img(std::move(data), OaImageLayout::Hwc, OaImageFormat::Gray);
	EXPECT_TRUE(img.Validate());
	EXPECT_EQ(img.Layout(), OaImageLayout::Hwc);
	EXPECT_EQ(img.Format(), OaImageFormat::Gray);
	EXPECT_EQ(img.Width(), 64);
	EXPECT_EQ(img.Height(), 64);
	EXPECT_EQ(img.Channels(), 1);
	EXPECT_EQ(img.BatchSize(), 1);
}

TEST(OaImage, ConstructHwGray) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{32, 32});
	OaImage img(std::move(data), OaImageLayout::Hw, OaImageFormat::Gray);
	EXPECT_TRUE(img.Validate());
	EXPECT_EQ(img.Layout(), OaImageLayout::Hw);
	EXPECT_EQ(img.Format(), OaImageFormat::Gray);
	EXPECT_EQ(img.Width(), 32);
	EXPECT_EQ(img.Height(), 32);
	EXPECT_EQ(img.Channels(), 1);  // Falls back to format
	EXPECT_EQ(img.BatchSize(), 1);
}

// ─── Invalid Shape/Layout Combinations ───────────────────────────────────────────

TEST(OaImage, InvalidRankNchw) {
	// Nchw requires rank 4
	auto data = OaFnMatrix::Zeros(OaMatrixShape{3, 224, 224});
	OaImage img(std::move(data), OaImageLayout::Nchw, OaImageFormat::Rgb);
	EXPECT_FALSE(img.Validate());
}

TEST(OaImage, InvalidRankHw) {
	// Hw requires rank 2
	auto data = OaFnMatrix::Zeros(OaMatrixShape{32, 32, 1});
	OaImage img(std::move(data), OaImageLayout::Hw, OaImageFormat::Gray);
	EXPECT_FALSE(img.Validate());
}

TEST(OaImage, InvalidChannelCount) {
	// Shape has 4 channels but format expects 3
	auto data = OaFnMatrix::Zeros(OaMatrixShape{1, 4, 224, 224});
	OaImage img(std::move(data), OaImageLayout::Nchw, OaImageFormat::Rgb);
	EXPECT_FALSE(img.Validate());
}

TEST(OaImage, InvalidChannelCountHwc) {
	// Shape has 3 channels but format expects 1
	auto data = OaFnMatrix::Zeros(OaMatrixShape{64, 64, 3});
	OaImage img(std::move(data), OaImageLayout::Hwc, OaImageFormat::Gray);
	EXPECT_FALSE(img.Validate());
}

// ─── AsMatrix Round-Trip ─────────────────────────────────────────────────────────

TEST(OaImage, AsMatrixRoundTrip) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{1, 3, 224, 224});
	OaImage img(std::move(data), OaImageLayout::Nchw, OaImageFormat::Rgb);

	// AsMatrix returns the same backing tensor
	const OaMatrix& mat = img.AsMatrix();
	EXPECT_EQ(mat.GetShape().Rank, 4);
	EXPECT_EQ(mat.GetShape()[0], 1);
	EXPECT_EQ(mat.GetShape()[1], 3);
	EXPECT_EQ(mat.GetShape()[2], 224);
	EXPECT_EQ(mat.GetShape()[3], 224);

	// Verify it's the same object (same address)
	EXPECT_EQ(&img.AsMatrix(), &mat);
}

TEST(OaImage, AsMatrixDtype) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{3, 64, 64});
	EXPECT_EQ(data.GetDtype(), OaScalarType::Float32);

	OaImage img(std::move(data), OaImageLayout::Chw, OaImageFormat::Rgb);
	EXPECT_EQ(img.GetDtype(), OaScalarType::Float32);
	EXPECT_EQ(img.AsMatrix().GetDtype(), OaScalarType::Float32);
}

// ─── Empty Image ─────────────────────────────────────────────────────────────────

TEST(OaImage, DefaultConstructed) {
	OaImage img;
	EXPECT_TRUE(img.IsEmpty());
	EXPECT_EQ(img.Width(), 0);
	EXPECT_EQ(img.Height(), 0);
	EXPECT_TRUE(img.Validate());  // Empty is trivially valid
}

// ─── Accessors for All Layouts ───────────────────────────────────────────────────

TEST(OaImage, AccessorsNchw) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{5, 3, 224, 224});
	OaImage img(std::move(data), OaImageLayout::Nchw, OaImageFormat::Rgb);
	EXPECT_EQ(img.BatchSize(), 5);
	EXPECT_EQ(img.Channels(), 3);
	EXPECT_EQ(img.Height(), 224);
	EXPECT_EQ(img.Width(), 224);
}

TEST(OaImage, AccessorsNhwc) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{3, 224, 224, 4});
	OaImage img(std::move(data), OaImageLayout::Nhwc, OaImageFormat::Rgba);
	EXPECT_EQ(img.BatchSize(), 3);
	EXPECT_EQ(img.Height(), 224);
	EXPECT_EQ(img.Width(), 224);
	EXPECT_EQ(img.Channels(), 4);
}

TEST(OaImage, AccessorsChw) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{1, 64, 64});
	OaImage img(std::move(data), OaImageLayout::Chw, OaImageFormat::Gray);
	EXPECT_EQ(img.BatchSize(), 1);
	EXPECT_EQ(img.Channels(), 1);
	EXPECT_EQ(img.Height(), 64);
	EXPECT_EQ(img.Width(), 64);
}

TEST(OaImage, AccessorsHwc) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{128, 128, 3});
	OaImage img(std::move(data), OaImageLayout::Hwc, OaImageFormat::Bgr);
	EXPECT_EQ(img.BatchSize(), 1);
	EXPECT_EQ(img.Height(), 128);
	EXPECT_EQ(img.Width(), 128);
	EXPECT_EQ(img.Channels(), 3);
}

TEST(OaImage, AccessorsHw) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{16, 16});
	OaImage img(std::move(data), OaImageLayout::Hw, OaImageFormat::Gray);
	EXPECT_EQ(img.BatchSize(), 1);
	EXPECT_EQ(img.Height(), 16);
	EXPECT_EQ(img.Width(), 16);
	EXPECT_EQ(img.Channels(), 1);  // From format, not shape
}

// ─── Format Variants ─────────────────────────────────────────────────────────────

TEST(OaImage, FormatGrayAlpha) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{2, 64, 64});
	OaImage img(std::move(data), OaImageLayout::Chw, OaImageFormat::GrayAlpha);
	EXPECT_TRUE(img.Validate());
	EXPECT_EQ(img.Channels(), 2);
}

TEST(OaImage, FormatBgra) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{1, 4, 224, 224});
	OaImage img(std::move(data), OaImageLayout::Nchw, OaImageFormat::Bgra);
	EXPECT_TRUE(img.Validate());
	EXPECT_EQ(img.Channels(), 4);
}

// ─── Phase 2: Vision Overloads ───────────────────────────────────────────────────

TEST(OaImage, VisionResize) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{1, 3, 224, 224});
	OaImage img(std::move(data), OaImageLayout::Nchw, OaImageFormat::Rgb);

	OaImage resized = OaFnImage::Resize(img, 128, 128);
	EXPECT_TRUE(resized.Validate());
	EXPECT_EQ(resized.Width(), 128);
	EXPECT_EQ(resized.Height(), 128);
	EXPECT_EQ(resized.Layout(), OaImageLayout::Nchw);
	EXPECT_EQ(resized.Format(), OaImageFormat::Rgb);
}

TEST(OaImage, VisionNormalize) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{1, 3, 224, 224});
	OaImage img(std::move(data), OaImageLayout::Nchw, OaImageFormat::Rgb);

	OaNormalizationParams params = {{0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f}};
	OaImage normalized = OaFnImage::Normalize(img, params);
	EXPECT_TRUE(normalized.Validate());
	EXPECT_EQ(normalized.Layout(), OaImageLayout::Nchw);
	EXPECT_EQ(normalized.Format(), OaImageFormat::Rgb);
}

TEST(OaImage, VisionConvertColorRgbToBgr) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{1, 3, 224, 224});
	OaImage img(std::move(data), OaImageLayout::Nchw, OaImageFormat::Rgb);

	OaImage converted = OaFnImage::ConvertColor(img, OaImageFormat::Bgr);
	EXPECT_TRUE(converted.Validate());
	EXPECT_EQ(converted.Format(), OaImageFormat::Bgr);
	EXPECT_EQ(converted.Layout(), OaImageLayout::Nchw);
}

TEST(OaImage, VisionConvertColorBgrToRgb) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{1, 3, 224, 224});
	OaImage img(std::move(data), OaImageLayout::Nchw, OaImageFormat::Bgr);

	OaImage converted = OaFnImage::ConvertColor(img, OaImageFormat::Rgb);
	EXPECT_TRUE(converted.Validate());
	EXPECT_EQ(converted.Format(), OaImageFormat::Rgb);
	EXPECT_EQ(converted.Layout(), OaImageLayout::Nchw);
}

TEST(OaImage, VisionConvertColorChwUsesChannelAxisZero) {
	auto data = OaFnMatrix::Empty(OaMatrixShape{3, 2, 2});
	for (OaI64 i = 0; i < 4; ++i) {
		data.Set(i, 1.0F);
		data.Set(4 + i, 2.0F);
		data.Set(8 + i, 3.0F);
	}
	OaImage img(std::move(data), OaImageLayout::Chw, OaImageFormat::Rgb);
	OaImage converted = OaFnImage::ConvertColor(img, OaImageFormat::Bgr);
	MaterializeVisionGraph();

	ASSERT_TRUE(converted.Validate());
	for (OaI64 i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(converted.AsMatrix().At(i), 3.0F);
		EXPECT_FLOAT_EQ(converted.AsMatrix().At(4 + i), 2.0F);
		EXPECT_FLOAT_EQ(converted.AsMatrix().At(8 + i), 1.0F);
	}
}

TEST(OaImage, VisionConvertColorNoOp) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{1, 3, 224, 224});
	OaImage img(std::move(data), OaImageLayout::Nchw, OaImageFormat::Rgb);

	// Same format should return the same image
	OaImage converted = OaFnImage::ConvertColor(img, OaImageFormat::Rgb);
	EXPECT_EQ(converted.Format(), OaImageFormat::Rgb);
}

// ─── Phase 3: Fused Preprocess ───────────────────────────────────────────────────

TEST(OaImage, FusedResizeNormalize) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{1, 3, 224, 224});
	OaImage img(std::move(data), OaImageLayout::Nchw, OaImageFormat::Rgb);

	OaNormalizationParams params = {{0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f}};
	OaImage result = OaFnImage::ResizeNormalize(img, 128, 128, params);
	MaterializeVisionGraph();
	EXPECT_TRUE(result.Validate());
	EXPECT_EQ(result.Width(), 128);
	EXPECT_EQ(result.Height(), 128);
	EXPECT_EQ(result.Layout(), OaImageLayout::Nchw);
	EXPECT_EQ(result.Format(), OaImageFormat::Rgb);
	EXPECT_NEAR(result.AsMatrix().At(0), -0.485F / 0.229F, 1.0e-5F);
	EXPECT_NEAR(result.AsMatrix().At(128 * 128), -0.456F / 0.224F, 1.0e-5F);
	EXPECT_NEAR(result.AsMatrix().At(2 * 128 * 128), -0.406F / 0.225F, 1.0e-5F);
}

TEST(OaImage, VisionResizeChwPreservesLayoutAndValues) {
	auto data = OaFnMatrix::Ones(OaMatrixShape{3, 4, 5});
	OaImage img(std::move(data), OaImageLayout::Chw, OaImageFormat::Rgb);
	OaImage result = OaFnImage::Resize(img, 3, 2);
	MaterializeVisionGraph();

	ASSERT_TRUE(result.Validate());
	EXPECT_EQ(result.AsMatrix().GetShape(), OaMatrixShape({3, 2, 3}));
	for (OaI64 i = 0; i < result.AsMatrix().NumElements(); ++i) {
		EXPECT_NEAR(result.AsMatrix().At(i), 1.0F, 1.0e-6F);
	}
}

// ─── Phase 4: UI Adapters ───────────────────────────────────────────────────────
// Note: Full UI adapter tests require actual OaTexture objects and Vulkan context.
// The bridge functions (OaTexture::ToImage, OaTexture::FromImage) are implemented
// in Source/Private/Oa/Ui/Image.cpp and can be tested in Vision UI tests.

// ─── Phase 5: OaImageBatch ───────────────────────────────────────────────────────

TEST(OaImageBatch, ConstructNchwRgb) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{4, 3, 224, 224});
	OaImageBatch batch(std::move(data), OaImageLayout::Nchw, OaImageFormat::Rgb);
	EXPECT_TRUE(batch.Validate());
	EXPECT_EQ(batch.Layout(), OaImageLayout::Nchw);
	EXPECT_EQ(batch.Format(), OaImageFormat::Rgb);
	EXPECT_EQ(batch.BatchSize(), 4);
	EXPECT_EQ(batch.Width(), 224);
	EXPECT_EQ(batch.Height(), 224);
	EXPECT_EQ(batch.Channels(), 3);
}

TEST(OaImageBatch, ConstructNhwcRgba) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{2, 64, 64, 4});
	OaImageBatch batch(std::move(data), OaImageLayout::Nhwc, OaImageFormat::Rgba);
	EXPECT_TRUE(batch.Validate());
	EXPECT_EQ(batch.Layout(), OaImageLayout::Nhwc);
	EXPECT_EQ(batch.Format(), OaImageFormat::Rgba);
	EXPECT_EQ(batch.BatchSize(), 2);
	EXPECT_EQ(batch.Width(), 64);
	EXPECT_EQ(batch.Height(), 64);
	EXPECT_EQ(batch.Channels(), 4);
}

TEST(OaImageBatch, InvalidRank) {
	// OaImageBatch only supports rank 4
	auto data = OaFnMatrix::Zeros(OaMatrixShape{3, 224, 224});
	OaImageBatch batch(std::move(data), OaImageLayout::Chw, OaImageFormat::Rgb);
	EXPECT_FALSE(batch.Validate());
}

TEST(OaImageBatch, InvalidLayout) {
	// OaImageBatch only supports batched layouts (Nchw or Nhwc)
	auto data = OaFnMatrix::Zeros(OaMatrixShape{4, 3, 224, 224});
	OaImageBatch batch(std::move(data), OaImageLayout::Chw, OaImageFormat::Rgb);
	EXPECT_FALSE(batch.Validate());
}

TEST(OaImageBatch, AsMatrixRoundTrip) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{4, 3, 224, 224});
	OaImageBatch batch(std::move(data), OaImageLayout::Nchw, OaImageFormat::Rgb);
	EXPECT_TRUE(batch.Validate());

	// Access underlying tensor
	const OaMatrix& mat = batch.AsMatrix();
	EXPECT_EQ(mat.GetShape().Rank, 4);
	EXPECT_EQ(mat.GetShape()[0], 4);
	EXPECT_EQ(mat.GetShape()[1], 3);
}

TEST(OaImageBatch, DefaultConstructed) {
	OaImageBatch batch;
	EXPECT_TRUE(batch.IsEmpty());
	EXPECT_EQ(batch.BatchSize(), 0);
}
