// oa::Image tests — Core media wrapper composed over oa::Matrix.
// phase 1: Core wrapper (requires vulkan since OA is GPU-only).

#include "../oaTest.h"

#include <oa/core/image.h>
#include <oa/core/fnMatrix.h>
#include <oa/vision/fnImage.h>
#include <oa/ui/image.h>

static void materializeVisionGraph() {
	auto& ctx = oa::ExecutionSession::getActive();
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
}

// ─── oa::imageFormatChannels ───────────────────────────────────────────────────────

TEST(Image, FormatChannels) {
	EXPECT_EQ(oa::imageFormatChannels(oa::ImageFormat::Gray), 1);
	EXPECT_EQ(oa::imageFormatChannels(oa::ImageFormat::GrayAlpha), 2);
	EXPECT_EQ(oa::imageFormatChannels(oa::ImageFormat::Rgb), 3);
	EXPECT_EQ(oa::imageFormatChannels(oa::ImageFormat::Bgr), 3);
	EXPECT_EQ(oa::imageFormatChannels(oa::ImageFormat::Rgba), 4);
	EXPECT_EQ(oa::imageFormatChannels(oa::ImageFormat::Bgra), 4);
}

// ─── Construction from oa::Matrix ───────────────────────────────────────────────────

TEST(Image, ConstructNchwRgb) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{1, 3, 224, 224});
	oa::Image img(std::move(data), oa::ImageLayout::Nchw, oa::ImageFormat::Rgb);
	EXPECT_TRUE(img.validate());
	EXPECT_EQ(img.layout(), oa::ImageLayout::Nchw);
	EXPECT_EQ(img.format(), oa::ImageFormat::Rgb);
	EXPECT_EQ(img.width(), 224);
	EXPECT_EQ(img.height(), 224);
	EXPECT_EQ(img.channels(), 3);
	EXPECT_EQ(img.batchSize(), 1);
}

TEST(Image, ConstructNhwcRgba) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{2, 224, 224, 4});
	oa::Image img(std::move(data), oa::ImageLayout::Nhwc, oa::ImageFormat::Rgba);
	EXPECT_TRUE(img.validate());
	EXPECT_EQ(img.layout(), oa::ImageLayout::Nhwc);
	EXPECT_EQ(img.format(), oa::ImageFormat::Rgba);
	EXPECT_EQ(img.width(), 224);
	EXPECT_EQ(img.height(), 224);
	EXPECT_EQ(img.channels(), 4);
	EXPECT_EQ(img.batchSize(), 2);
}

TEST(Image, ConstructChwBgr) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{3, 128, 128});
	oa::Image img(std::move(data), oa::ImageLayout::Chw, oa::ImageFormat::Bgr);
	EXPECT_TRUE(img.validate());
	EXPECT_EQ(img.layout(), oa::ImageLayout::Chw);
	EXPECT_EQ(img.format(), oa::ImageFormat::Bgr);
	EXPECT_EQ(img.width(), 128);
	EXPECT_EQ(img.height(), 128);
	EXPECT_EQ(img.channels(), 3);
	EXPECT_EQ(img.batchSize(), 1);
}

TEST(Image, ConstructHwcGray) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{64, 64, 1});
	oa::Image img(std::move(data), oa::ImageLayout::Hwc, oa::ImageFormat::Gray);
	EXPECT_TRUE(img.validate());
	EXPECT_EQ(img.layout(), oa::ImageLayout::Hwc);
	EXPECT_EQ(img.format(), oa::ImageFormat::Gray);
	EXPECT_EQ(img.width(), 64);
	EXPECT_EQ(img.height(), 64);
	EXPECT_EQ(img.channels(), 1);
	EXPECT_EQ(img.batchSize(), 1);
}

TEST(Image, ConstructHwGray) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{32, 32});
	oa::Image img(std::move(data), oa::ImageLayout::Hw, oa::ImageFormat::Gray);
	EXPECT_TRUE(img.validate());
	EXPECT_EQ(img.layout(), oa::ImageLayout::Hw);
	EXPECT_EQ(img.format(), oa::ImageFormat::Gray);
	EXPECT_EQ(img.width(), 32);
	EXPECT_EQ(img.height(), 32);
	EXPECT_EQ(img.channels(), 1);  // Falls back to format
	EXPECT_EQ(img.batchSize(), 1);
}

// ─── Invalid Shape/layout Combinations ───────────────────────────────────────────

TEST(Image, InvalidRankNchw) {
	// Nchw requires rank 4
	oa::Image img(
		oa::FnMatrix::zeros(oa::MatrixShape{1, 3, 224, 224}),
		oa::ImageLayout::Nchw,
		oa::ImageFormat::Rgb);
	img.asMatrix() = oa::FnMatrix::zeros(oa::MatrixShape{3, 224, 224});
	EXPECT_FALSE(img.validate());
}

TEST(Image, InvalidRankHw) {
	// Hw requires rank 2
	oa::Image img(
		oa::FnMatrix::zeros(oa::MatrixShape{32, 32}),
		oa::ImageLayout::Hw,
		oa::ImageFormat::Gray);
	img.asMatrix() = oa::FnMatrix::zeros(oa::MatrixShape{32, 32, 1});
	EXPECT_FALSE(img.validate());
}

TEST(Image, InvalidChannelCount) {
	// Shape has 4 channels but format expects 3
	oa::Image img(
		oa::FnMatrix::zeros(oa::MatrixShape{1, 3, 224, 224}),
		oa::ImageLayout::Nchw,
		oa::ImageFormat::Rgb);
	img.asMatrix() = oa::FnMatrix::zeros(oa::MatrixShape{1, 4, 224, 224});
	EXPECT_FALSE(img.validate());
}

TEST(Image, InvalidChannelCountHwc) {
	// Shape has 3 channels but format expects 1
	oa::Image img(
		oa::FnMatrix::zeros(oa::MatrixShape{64, 64, 1}),
		oa::ImageLayout::Hwc,
		oa::ImageFormat::Gray);
	img.asMatrix() = oa::FnMatrix::zeros(oa::MatrixShape{64, 64, 3});
	EXPECT_FALSE(img.validate());
}

// ─── asMatrix Round-Trip ─────────────────────────────────────────────────────────

TEST(Image, AsMatrixRoundTrip) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{1, 3, 224, 224});
	oa::Image img(std::move(data), oa::ImageLayout::Nchw, oa::ImageFormat::Rgb);

	// asMatrix returns the same backing tensor
	const oa::Matrix& mat = img.asMatrix();
	EXPECT_EQ(mat.getShape().rank, 4);
	EXPECT_EQ(mat.getShape()[0], 1);
	EXPECT_EQ(mat.getShape()[1], 3);
	EXPECT_EQ(mat.getShape()[2], 224);
	EXPECT_EQ(mat.getShape()[3], 224);

	// verify it's the same object (same address)
	EXPECT_EQ(&img.asMatrix(), &mat);
}

TEST(Image, AsMatrixDtype) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{3, 64, 64});
	EXPECT_EQ(data.getDtype(), oa::ScalarType::Float32);

	oa::Image img(std::move(data), oa::ImageLayout::Chw, oa::ImageFormat::Rgb);
	EXPECT_EQ(img.getDtype(), oa::ScalarType::Float32);
	EXPECT_EQ(img.asMatrix().getDtype(), oa::ScalarType::Float32);
}

// ─── Empty Image ─────────────────────────────────────────────────────────────────

TEST(Image, DefaultConstructed) {
	oa::Image img;
	EXPECT_TRUE(img.isEmpty());
	EXPECT_EQ(img.width(), 0);
	EXPECT_EQ(img.height(), 0);
	EXPECT_TRUE(img.validate());  // Empty is trivially valid
}

// ─── Accessors for All Layouts ───────────────────────────────────────────────────

TEST(Image, AccessorsNchw) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{5, 3, 224, 224});
	oa::Image img(std::move(data), oa::ImageLayout::Nchw, oa::ImageFormat::Rgb);
	EXPECT_EQ(img.batchSize(), 5);
	EXPECT_EQ(img.channels(), 3);
	EXPECT_EQ(img.height(), 224);
	EXPECT_EQ(img.width(), 224);
}

TEST(Image, AccessorsNhwc) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{3, 224, 224, 4});
	oa::Image img(std::move(data), oa::ImageLayout::Nhwc, oa::ImageFormat::Rgba);
	EXPECT_EQ(img.batchSize(), 3);
	EXPECT_EQ(img.height(), 224);
	EXPECT_EQ(img.width(), 224);
	EXPECT_EQ(img.channels(), 4);
}

TEST(Image, AccessorsChw) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{1, 64, 64});
	oa::Image img(std::move(data), oa::ImageLayout::Chw, oa::ImageFormat::Gray);
	EXPECT_EQ(img.batchSize(), 1);
	EXPECT_EQ(img.channels(), 1);
	EXPECT_EQ(img.height(), 64);
	EXPECT_EQ(img.width(), 64);
}

TEST(Image, AccessorsHwc) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{128, 128, 3});
	oa::Image img(std::move(data), oa::ImageLayout::Hwc, oa::ImageFormat::Bgr);
	EXPECT_EQ(img.batchSize(), 1);
	EXPECT_EQ(img.height(), 128);
	EXPECT_EQ(img.width(), 128);
	EXPECT_EQ(img.channels(), 3);
}

TEST(Image, AccessorsHw) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{16, 16});
	oa::Image img(std::move(data), oa::ImageLayout::Hw, oa::ImageFormat::Gray);
	EXPECT_EQ(img.batchSize(), 1);
	EXPECT_EQ(img.height(), 16);
	EXPECT_EQ(img.width(), 16);
	EXPECT_EQ(img.channels(), 1);  // From format, not shape
}

// ─── format Variants ─────────────────────────────────────────────────────────────

TEST(Image, FormatGrayAlpha) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{2, 64, 64});
	oa::Image img(std::move(data), oa::ImageLayout::Chw, oa::ImageFormat::GrayAlpha);
	EXPECT_TRUE(img.validate());
	EXPECT_EQ(img.channels(), 2);
}

TEST(Image, FormatBgra) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{1, 4, 224, 224});
	oa::Image img(std::move(data), oa::ImageLayout::Nchw, oa::ImageFormat::Bgra);
	EXPECT_TRUE(img.validate());
	EXPECT_EQ(img.channels(), 4);
}

// ─── phase 2: Vision Overloads ───────────────────────────────────────────────────

TEST(Image, VisionResize) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{1, 3, 224, 224});
	oa::Image img(std::move(data), oa::ImageLayout::Nchw, oa::ImageFormat::Rgb);

	oa::Image resized = oa::FnImage::resize(img, 128, 128);
	EXPECT_TRUE(resized.validate());
	EXPECT_EQ(resized.width(), 128);
	EXPECT_EQ(resized.height(), 128);
	EXPECT_EQ(resized.layout(), oa::ImageLayout::Nchw);
	EXPECT_EQ(resized.format(), oa::ImageFormat::Rgb);
}

TEST(Image, VisionNormalize) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{1, 3, 224, 224});
	oa::Image img(std::move(data), oa::ImageLayout::Nchw, oa::ImageFormat::Rgb);

	oa::NormalizationParams params = {{0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f}};
	oa::Image normalized = oa::FnImage::normalize(img, params);
	EXPECT_TRUE(normalized.validate());
	EXPECT_EQ(normalized.layout(), oa::ImageLayout::Nchw);
	EXPECT_EQ(normalized.format(), oa::ImageFormat::Rgb);
}

TEST(Image, VisionConvertColorRgbToBgr) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{1, 3, 224, 224});
	oa::Image img(std::move(data), oa::ImageLayout::Nchw, oa::ImageFormat::Rgb);

	oa::Image converted = oa::FnImage::convertColor(img, oa::ImageFormat::Bgr);
	EXPECT_TRUE(converted.validate());
	EXPECT_EQ(converted.format(), oa::ImageFormat::Bgr);
	EXPECT_EQ(converted.layout(), oa::ImageLayout::Nchw);
}

TEST(Image, VisionConvertColorBgrToRgb) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{1, 3, 224, 224});
	oa::Image img(std::move(data), oa::ImageLayout::Nchw, oa::ImageFormat::Bgr);

	oa::Image converted = oa::FnImage::convertColor(img, oa::ImageFormat::Rgb);
	EXPECT_TRUE(converted.validate());
	EXPECT_EQ(converted.format(), oa::ImageFormat::Rgb);
	EXPECT_EQ(converted.layout(), oa::ImageLayout::Nchw);
}

TEST(Image, VisionConvertColorChwUsesChannelAxisZero) {
	auto data = oa::FnMatrix::empty(oa::MatrixShape{3, 2, 2});
	for (oa::I64 i = 0; i < 4; ++i) {
		data.set(i, 1.0F);
		data.set(4 + i, 2.0F);
		data.set(8 + i, 3.0F);
	}
	oa::Image img(std::move(data), oa::ImageLayout::Chw, oa::ImageFormat::Rgb);
	oa::Image converted = oa::FnImage::convertColor(img, oa::ImageFormat::Bgr);
	materializeVisionGraph();

	ASSERT_TRUE(converted.validate());
	for (oa::I64 i = 0; i < 4; ++i) {
		EXPECT_FLOAT_EQ(converted.asMatrix().at(i), 3.0F);
		EXPECT_FLOAT_EQ(converted.asMatrix().at(4 + i), 2.0F);
		EXPECT_FLOAT_EQ(converted.asMatrix().at(8 + i), 1.0F);
	}
}

TEST(Image, VisionConvertColorNoOp) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{1, 3, 224, 224});
	oa::Image img(std::move(data), oa::ImageLayout::Nchw, oa::ImageFormat::Rgb);

	// Same format should return the same image
	oa::Image converted = oa::FnImage::convertColor(img, oa::ImageFormat::Rgb);
	EXPECT_EQ(converted.format(), oa::ImageFormat::Rgb);
}

// ─── phase 3: Fused Preprocess ───────────────────────────────────────────────────

TEST(Image, FusedResizeNormalize) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{1, 3, 224, 224});
	oa::Image img(std::move(data), oa::ImageLayout::Nchw, oa::ImageFormat::Rgb);

	oa::NormalizationParams params = {{0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f}};
	oa::Image result = oa::FnImage::resizeNormalize(img, 128, 128, params);
	materializeVisionGraph();
	EXPECT_TRUE(result.validate());
	EXPECT_EQ(result.width(), 128);
	EXPECT_EQ(result.height(), 128);
	EXPECT_EQ(result.layout(), oa::ImageLayout::Nchw);
	EXPECT_EQ(result.format(), oa::ImageFormat::Rgb);
	EXPECT_NEAR(result.asMatrix().at(0), -0.485F / 0.229F, 1.0e-5F);
	EXPECT_NEAR(result.asMatrix().at(128 * 128), -0.456F / 0.224F, 1.0e-5F);
	EXPECT_NEAR(result.asMatrix().at(2 * 128 * 128), -0.406F / 0.225F, 1.0e-5F);
}

TEST(Image, VisionResizeChwPreservesLayoutAndValues) {
	auto data = oa::FnMatrix::ones(oa::MatrixShape{3, 4, 5});
	oa::Image img(std::move(data), oa::ImageLayout::Chw, oa::ImageFormat::Rgb);
	oa::Image result = oa::FnImage::resize(img, 3, 2);
	materializeVisionGraph();

	ASSERT_TRUE(result.validate());
	EXPECT_EQ(result.asMatrix().getShape(), oa::MatrixShape({3, 2, 3}));
	for (oa::I64 i = 0; i < result.asMatrix().numElements(); ++i) {
		EXPECT_NEAR(result.asMatrix().at(i), 1.0F, 1.0e-6F);
	}
}

// Semantic-image to texture lowering has byte-exact layout/format coverage in
// TestUi; Core keeps only the value and image-operation contracts here.

// ─── phase 5: oa::ImageBatch ───────────────────────────────────────────────────────

TEST(ImageBatch, ConstructNchwRgb) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{4, 3, 224, 224});
	oa::ImageBatch batch(std::move(data), oa::ImageLayout::Nchw, oa::ImageFormat::Rgb);
	EXPECT_TRUE(batch.validate());
	EXPECT_EQ(batch.layout(), oa::ImageLayout::Nchw);
	EXPECT_EQ(batch.format(), oa::ImageFormat::Rgb);
	EXPECT_EQ(batch.batchSize(), 4);
	EXPECT_EQ(batch.width(), 224);
	EXPECT_EQ(batch.height(), 224);
	EXPECT_EQ(batch.channels(), 3);
}

TEST(ImageBatch, ConstructNhwcRgba) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{2, 64, 64, 4});
	oa::ImageBatch batch(std::move(data), oa::ImageLayout::Nhwc, oa::ImageFormat::Rgba);
	EXPECT_TRUE(batch.validate());
	EXPECT_EQ(batch.layout(), oa::ImageLayout::Nhwc);
	EXPECT_EQ(batch.format(), oa::ImageFormat::Rgba);
	EXPECT_EQ(batch.batchSize(), 2);
	EXPECT_EQ(batch.width(), 64);
	EXPECT_EQ(batch.height(), 64);
	EXPECT_EQ(batch.channels(), 4);
}

TEST(ImageBatch, InvalidRank) {
	// oa::ImageBatch only supports rank 4
	oa::ImageBatch batch(
		oa::FnMatrix::zeros(oa::MatrixShape{1, 3, 224, 224}),
		oa::ImageLayout::Nchw,
		oa::ImageFormat::Rgb);
	batch.asMatrix() = oa::FnMatrix::zeros(oa::MatrixShape{3, 224, 224});
	EXPECT_FALSE(batch.validate());
}

TEST(ImageBatch, InvalidLayout) {
	// oa::ImageBatch only supports batched layouts (Nchw or Nhwc)
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{3, 224, 224});
#if defined(NDEBUG)
	oa::ImageBatch batch(
		std::move(data),
		oa::ImageLayout::Chw,
		oa::ImageFormat::Rgb);
	EXPECT_FALSE(batch.validate());
#else
	EXPECT_DEATH(
		{
			oa::ImageBatch batch(
				std::move(data),
				oa::ImageLayout::Chw,
				oa::ImageFormat::Rgb);
		},
		"oa::ImageBatch: shape/layout/format mismatch");
#endif
}

TEST(ImageBatch, AsMatrixRoundTrip) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{4, 3, 224, 224});
	oa::ImageBatch batch(std::move(data), oa::ImageLayout::Nchw, oa::ImageFormat::Rgb);
	EXPECT_TRUE(batch.validate());

	// access underlying tensor
	const oa::Matrix& mat = batch.asMatrix();
	EXPECT_EQ(mat.getShape().rank, 4);
	EXPECT_EQ(mat.getShape()[0], 4);
	EXPECT_EQ(mat.getShape()[1], 3);
}

TEST(ImageBatch, DefaultConstructed) {
	oa::ImageBatch batch;
	EXPECT_TRUE(batch.isEmpty());
	EXPECT_EQ(batch.batchSize(), 0);
}
