// Format-neutral OaImageDecoder/OaImageEncoder contract tests.

#include "../../OaTest.h"

#include <Oa/Ui/Image.h>
#include <Oa/Vision/ImageDecoder.h>
#include <Oa/Vision/ImageEncoder.h>

namespace {

void ExpectDecodedPattern(const OaImage& InImage, OaI32 InChannels = 3)
{
	ASSERT_TRUE(InImage.Validate());
	EXPECT_EQ(InImage.Layout(), OaImageLayout::Nchw);
	EXPECT_EQ(InImage.Format(),
		InChannels == 4 ? OaImageFormat::Rgba : OaImageFormat::Rgb);
	EXPECT_EQ(InImage.GetDtype(), OaScalarType::Float32);
	EXPECT_EQ(InImage.Width(), 320);
	EXPECT_EQ(InImage.Height(), 180);
	EXPECT_EQ(InImage.Channels(), InChannels);
	EXPECT_EQ(
		InImage.AsMatrix().GetShape(),
		OaMatrixShape({1, InChannels, 180, 320}));

	OaVec<OaF32> values;
	values.Resize(static_cast<OaUsize>(
		InImage.AsMatrix().NumElements()));
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		InImage.AsMatrix(),
		values.Data(),
		values.Size() * sizeof(OaF32)).IsOk());
	OaU32 nonZeroCount = 0U;
	OaF32 minValue = 1.0F;
	OaF32 maxValue = 0.0F;
	for (OaF32 value : values) {
		EXPECT_GE(value, 0.0F);
		EXPECT_LE(value, 1.0F);
		nonZeroCount += value > 0.0F ? 1U : 0U;
		minValue = value < minValue ? value : minValue;
		maxValue = value > maxValue ? value : maxValue;
	}
	EXPECT_GT(nonZeroCount, 0U);
	EXPECT_GT(maxValue, minValue);
}

void ExpectRoundTrip(
	const OaImage& InSource,
	OaImageCodec InCodec,
	OaF32 InTolerance)
{
	auto encoded = OaImageEncoder::Encode(InSource, InCodec, 92U);
	ASSERT_TRUE(encoded.IsOk()) << encoded.GetStatus().ToString();
	ASSERT_FALSE(encoded->Empty());

	auto decoded = OaImageDecoder::LoadMemory(
		OaSpan<const OaU8>(encoded->Data(), encoded->Size()));
	ASSERT_TRUE(decoded.IsOk()) << decoded.GetStatus().ToString();
	ASSERT_EQ(decoded->AsMatrix().GetShape(),
		InSource.AsMatrix().GetShape());

	OaF64 absoluteError = 0.0;
	for (OaI64 index = 0;
		index < InSource.AsMatrix().NumElements();
		++index) {
		absoluteError += std::abs(
			static_cast<OaF64>(decoded->AsMatrix().At(index))
			- static_cast<OaF64>(InSource.AsMatrix().At(index)));
	}
	const OaF64 meanAbsoluteError = absoluteError
		/ static_cast<OaF64>(InSource.AsMatrix().NumElements());
	EXPECT_LE(meanAbsoluteError, static_cast<OaF64>(InTolerance));
}

} // namespace

TEST(ImageCodec, RejectsInvalidMemory)
{
	OaVec<OaU8> invalid = {0x00U, 0x01U, 0x02U};
	auto result = OaImageDecoder::LoadMemory(
		OaSpan<const OaU8>(invalid.Data(), invalid.Size()));
	EXPECT_TRUE(result.IsError());
	EXPECT_EQ(result.GetStatus().GetCode(), OaStatusCode::FileCorrupt);
}

TEST(ImageCodec, RejectsMissingFile)
{
	auto result = OaImageDecoder::LoadFile("nonexistent_file.jpg");
	EXPECT_TRUE(result.IsError());
}

TEST(ImageCodec, ReportsBackendCapabilities)
{
	for (OaImageCodec codec : {
		OaImageCodec::Jpeg,
		OaImageCodec::Png,
		OaImageCodec::Bmp,
		OaImageCodec::Tga}) {
		EXPECT_TRUE(OaImageDecoder::Supports(codec));
		EXPECT_TRUE(OaImageEncoder::Supports(codec));
	}
	EXPECT_FALSE(OaImageDecoder::Supports(OaImageCodec::Auto));
	EXPECT_FALSE(OaImageEncoder::Supports(OaImageCodec::Auto));
	EXPECT_EQ(
		OaImageDecoder::Supports(OaImageCodec::Webp),
		OaImageEncoder::Supports(OaImageCodec::Webp));
}

TEST_F(OaVkEngineTestFixture, LoadsPngMemory)
{
	auto bytes = OaFilesystem::ReadBinary(
		OaTestAssetPath("Image/VisionTestPattern320x180.png"));
	ASSERT_TRUE(bytes.IsOk()) << bytes.GetStatus().ToString();

	auto result = OaImageDecoder::LoadMemory(
		OaSpan<const OaU8>(bytes->Data(), bytes->Size()));
	ASSERT_TRUE(result.IsOk()) << result.GetStatus().ToString();
	ExpectDecodedPattern(*result);
}

TEST_F(OaVkEngineTestFixture, LoadsJpegFile)
{
	auto result = OaImageDecoder::LoadFile(
		OaTestAssetPath("Image/VisionTestPattern320x180.jpg"));
	ASSERT_TRUE(result.IsOk()) << result.GetStatus().ToString();
	ExpectDecodedPattern(*result);
}

TEST_F(OaVkEngineTestFixture, LoadsRgba)
{
	auto result = OaImageDecoder::LoadFile(
		OaTestAssetPath("Image/VisionTestPattern320x180.png"),
		OaImageFormat::Rgba);
	ASSERT_TRUE(result.IsOk()) << result.GetStatus().ToString();
	ExpectDecodedPattern(*result, 4);
}

TEST_F(OaVkEngineTestFixture, LosslessCodecMemoryRoundTrips)
{
	auto source = OaImageDecoder::LoadFile(
		OaTestAssetPath("Image/VisionTestPattern320x180.png"));
	ASSERT_TRUE(source.IsOk()) << source.GetStatus().ToString();

	ExpectRoundTrip(*source, OaImageCodec::Png, 0.0F);
	ExpectRoundTrip(*source, OaImageCodec::Bmp, 0.0F);
	ExpectRoundTrip(*source, OaImageCodec::Tga, 0.0F);
}

TEST_F(OaVkEngineTestFixture, LossyCodecMemoryRoundTrips)
{
	auto source = OaImageDecoder::LoadFile(
		OaTestAssetPath("Image/VisionTestPattern320x180.png"));
	ASSERT_TRUE(source.IsOk()) << source.GetStatus().ToString();

	ExpectRoundTrip(*source, OaImageCodec::Jpeg, 0.08F);
	if (OaImageEncoder::Supports(OaImageCodec::Webp)) {
		ExpectRoundTrip(*source, OaImageCodec::Webp, 0.08F);
	}
}

TEST_F(OaVkEngineTestFixture, SaveFileInfersCodec)
{
	auto source = OaImageDecoder::LoadFile(
		OaTestAssetPath("Image/VisionTestPattern320x180.png"));
	ASSERT_TRUE(source.IsOk()) << source.GetStatus().ToString();

	const OaPath path = OaPaths::Temp() / "oa_image_codec_roundtrip.png";
	ASSERT_TRUE(OaImageEncoder::SaveFile(path, *source).IsOk());
	auto decoded = OaImageDecoder::LoadFile(path);
	ASSERT_TRUE(decoded.IsOk()) << decoded.GetStatus().ToString();
	ExpectDecodedPattern(*decoded);
	EXPECT_TRUE(OaFilesystem::RemoveFile(path).IsOk());
}

TEST_F(OaVkEngineTestFixture, TextureLoadReusesWebpBackend)
{
	if (not OaImageEncoder::Supports(OaImageCodec::Webp)) {
		GTEST_SKIP() << "libwebp is not available in this build";
	}
	auto source = OaImageDecoder::LoadFile(
		OaTestAssetPath("Image/VisionTestPattern320x180.png"));
	ASSERT_TRUE(source.IsOk()) << source.GetStatus().ToString();

	const OaPath path = OaPaths::Temp() / "oa_texture_codec.webp";
	ASSERT_TRUE(OaImageEncoder::SaveFile(path, *source).IsOk());
	auto textureResult = OaTexture::LoadFile(Rt(), path.String());
	ASSERT_TRUE(textureResult.IsOk())
		<< textureResult.GetStatus().ToString();
	EXPECT_EQ(textureResult->Width, 320);
	EXPECT_EQ(textureResult->Height, 180);
	textureResult->Destroy(Rt());
	EXPECT_TRUE(OaFilesystem::RemoveFile(path).IsOk());
}

TEST_F(OaVkEngineTestFixture, RejectsUnknownOutputExtension)
{
	auto source = OaImageDecoder::LoadFile(
		OaTestAssetPath("Image/VisionTestPattern320x180.png"));
	ASSERT_TRUE(source.IsOk()) << source.GetStatus().ToString();

	const OaStatus status = OaImageEncoder::SaveFile(
		OaPaths::Temp() / "oa_image_codec.invalid",
		*source);
	EXPECT_EQ(status.GetCode(), OaStatusCode::InvalidArgument);
}
