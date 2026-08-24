// format-neutral oa::FnImage still-image codec contract tests.

#include "../../oaTest.h"

#include <oa/ui/image.h>
#include <oa/vision/fnImage.h>

static_assert(static_cast<oa::U8>(oa::ImageCodec::Auto) == 0U);
static_assert(static_cast<oa::U8>(oa::ImageCodec::Jpeg) == 1U);
static_assert(static_cast<oa::U8>(oa::ImageCodec::Png) == 2U);
static_assert(static_cast<oa::U8>(oa::ImageCodec::Webp) == 3U);
static_assert(static_cast<oa::U8>(oa::ImageCodec::Bmp) == 4U);
static_assert(static_cast<oa::U8>(oa::ImageCodec::Tga) == 5U);

namespace {

void expectDecodedPattern(const oa::Image& inImage, oa::I32 inChannels = 3)
{
	ASSERT_TRUE(inImage.validate());
	EXPECT_EQ(inImage.layout(), oa::ImageLayout::Nchw);
	EXPECT_EQ(inImage.format(),
		inChannels == 4 ? oa::ImageFormat::Rgba : oa::ImageFormat::Rgb);
	EXPECT_EQ(inImage.getDtype(), oa::ScalarType::Float32);
	EXPECT_EQ(inImage.width(), 320);
	EXPECT_EQ(inImage.height(), 180);
	EXPECT_EQ(inImage.channels(), inChannels);
	EXPECT_EQ(
		inImage.asMatrix().getShape(),
		oa::MatrixShape({1, inChannels, 180, 320}));

	oa::Vec<oa::F32> values;
	values.resize(static_cast<oa::Usize>(
		inImage.asMatrix().numElements()));
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		inImage.asMatrix(),
		values.data(),
		values.size() * sizeof(oa::F32)).isOk());
	oa::U32 nonZeroCount = 0U;
	oa::F32 minValue = 1.0F;
	oa::F32 maxValue = 0.0F;
	for (oa::F32 value : values) {
		EXPECT_GE(value, 0.0F);
		EXPECT_LE(value, 1.0F);
		nonZeroCount += value > 0.0F ? 1U : 0U;
		minValue = value < minValue ? value : minValue;
		maxValue = value > maxValue ? value : maxValue;
	}
	EXPECT_GT(nonZeroCount, 0U);
	EXPECT_GT(maxValue, minValue);
}

void expectRoundTrip(
	const oa::Image& inSource,
	oa::ImageCodec inCodec,
	oa::F32 inTolerance)
{
	auto encoded = oa::FnImage::encode(inSource, inCodec, 92U);
	ASSERT_TRUE(encoded.isOk()) << encoded.getStatus().toString();
	ASSERT_FALSE(encoded->empty());

	auto decoded = oa::FnImage::decodeMemory(
		oa::Span<const oa::U8>(encoded->data(), encoded->size()));
	ASSERT_TRUE(decoded.isOk()) << decoded.getStatus().toString();
	ASSERT_EQ(decoded->asMatrix().getShape(),
		inSource.asMatrix().getShape());

	oa::F64 absoluteError = 0.0;
	for (oa::I64 index = 0;
		index < inSource.asMatrix().numElements();
		++index) {
		absoluteError += std::abs(
			static_cast<oa::F64>(decoded->asMatrix().at(index))
			- static_cast<oa::F64>(inSource.asMatrix().at(index)));
	}
	const oa::F64 meanAbsoluteError = absoluteError
		/ static_cast<oa::F64>(inSource.asMatrix().numElements());
	EXPECT_LE(meanAbsoluteError, static_cast<oa::F64>(inTolerance));
}

} // namespace

TEST(ImageCodec, RejectsInvalidMemory)
{
	oa::Vec<oa::U8> invalid = {0x00U, 0x01U, 0x02U};
	auto result = oa::FnImage::decodeMemory(
		oa::Span<const oa::U8>(invalid.data(), invalid.size()));
	EXPECT_TRUE(result.isError());
	EXPECT_EQ(result.getStatus().getCode(), oa::StatusCode::FileCorrupt);
}

TEST(ImageCodec, RejectsMissingFile)
{
	auto result = oa::FnImage::decodeFile("nonexistent_file.jpg");
	EXPECT_TRUE(result.isError());
}

TEST(ImageCodec, ReportsBackendCapabilities)
{
	EXPECT_STREQ(oa::imageCodecToString(oa::ImageCodec::Webp), "webp");
	for (oa::ImageCodec codec : {
		oa::ImageCodec::Jpeg,
		oa::ImageCodec::Png,
		oa::ImageCodec::Bmp,
		oa::ImageCodec::Tga}) {
		EXPECT_TRUE(oa::FnImage::canDecode(codec));
		EXPECT_TRUE(oa::FnImage::canEncode(codec));
	}
	EXPECT_FALSE(oa::FnImage::canDecode(oa::ImageCodec::Auto));
	EXPECT_FALSE(oa::FnImage::canEncode(oa::ImageCodec::Auto));
	EXPECT_EQ(
		oa::FnImage::canDecode(oa::ImageCodec::Webp),
		oa::FnImage::canEncode(oa::ImageCodec::Webp));
}

TEST_F(VkEngineTestFixture, LoadsPngMemory)
{
	auto bytes = oa::Filesystem::readBinary(
		testAssetPath("image/visionTestPattern320x180.png"));
	ASSERT_TRUE(bytes.isOk()) << bytes.getStatus().toString();

	auto result = oa::FnImage::decodeMemory(
		oa::Span<const oa::U8>(bytes->data(), bytes->size()));
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	expectDecodedPattern(*result);
}

TEST_F(VkEngineTestFixture, LoadsJpegFile)
{
	auto result = oa::FnImage::decodeFile(
		testAssetPath("image/visionTestPattern320x180.jpg"));
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	expectDecodedPattern(*result);
}

TEST_F(VkEngineTestFixture, LoadsRgba)
{
	auto result = oa::FnImage::decodeFile(
		testAssetPath("image/visionTestPattern320x180.png"),
		oa::ImageFormat::Rgba);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	expectDecodedPattern(*result, 4);
}

TEST_F(VkEngineTestFixture, LosslessCodecMemoryRoundTrips)
{
	auto source = oa::FnImage::decodeFile(
		testAssetPath("image/visionTestPattern320x180.png"));
	ASSERT_TRUE(source.isOk()) << source.getStatus().toString();

	expectRoundTrip(*source, oa::ImageCodec::Png, 0.0F);
	expectRoundTrip(*source, oa::ImageCodec::Bmp, 0.0F);
	expectRoundTrip(*source, oa::ImageCodec::Tga, 0.0F);
}

TEST_F(VkEngineTestFixture, LossyCodecMemoryRoundTrips)
{
	auto source = oa::FnImage::decodeFile(
		testAssetPath("image/visionTestPattern320x180.png"));
	ASSERT_TRUE(source.isOk()) << source.getStatus().toString();

	expectRoundTrip(*source, oa::ImageCodec::Jpeg, 0.08F);
	if (oa::FnImage::canEncode(oa::ImageCodec::Webp)) {
		expectRoundTrip(*source, oa::ImageCodec::Webp, 0.08F);
	}
}

TEST_F(VkEngineTestFixture, SaveFileInfersCodec)
{
	auto source = oa::FnImage::decodeFile(
		testAssetPath("image/visionTestPattern320x180.png"));
	ASSERT_TRUE(source.isOk()) << source.getStatus().toString();

	const oa::Path path = oa::Paths::temp() / "oa_image_codec_roundtrip.png";
	ASSERT_TRUE(oa::FnImage::saveFile(path, *source).isOk());
	auto decoded = oa::FnImage::decodeFile(path);
	ASSERT_TRUE(decoded.isOk()) << decoded.getStatus().toString();
	expectDecodedPattern(*decoded);
	EXPECT_TRUE(oa::Filesystem::removeFile(path).isOk());
}

TEST_F(VkEngineTestFixture, SemanticTextureUsesDecodedWebp)
{
	if (not oa::FnImage::canEncode(oa::ImageCodec::Webp)) {
		GTEST_SKIP() << "libwebp is not available in this build";
	}
	auto source = oa::FnImage::decodeFile(
		testAssetPath("image/visionTestPattern320x180.png"));
	ASSERT_TRUE(source.isOk()) << source.getStatus().toString();

	const oa::Path path = oa::Paths::temp() / "oa_texture_codec.webp";
	ASSERT_TRUE(oa::FnImage::saveFile(path, *source).isOk());
	auto decoded = oa::FnImage::decodeFile(path, oa::ImageFormat::Rgba);
	ASSERT_TRUE(decoded.isOk()) << decoded.getStatus().toString();
	auto textureResult = oa::FnTexture::fromImage(rt(), *decoded);
	ASSERT_TRUE(textureResult.isOk())
		<< textureResult.getStatus().toString();
	EXPECT_EQ(textureResult->width(), 320);
	EXPECT_EQ(textureResult->height(), 180);
	EXPECT_TRUE(oa::Filesystem::removeFile(path).isOk());
}

TEST_F(VkEngineTestFixture, RejectsUnknownOutputExtension)
{
	auto source = oa::FnImage::decodeFile(
		testAssetPath("image/visionTestPattern320x180.png"));
	ASSERT_TRUE(source.isOk()) << source.getStatus().toString();

	const oa::Status status = oa::FnImage::saveFile(
		oa::Paths::temp() / "oa_image_codec.invalid",
		*source);
	EXPECT_EQ(status.getCode(), oa::StatusCode::InvalidArgument);
}
