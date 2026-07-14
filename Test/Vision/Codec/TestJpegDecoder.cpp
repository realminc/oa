// OaJpegDecoder tests - JPEG/PNG decode validation.

#include "../../OaTest.h"

#include <Oa/Core/FnMatrix.h>
#include <Oa/Vision/FnImage.h>
#include <Oa/Vision/JpegDecoder.h>

namespace {

OaString VisionAssetString(OaStringView InRelativePath)
{
	return OaTestAssetPath(InRelativePath).String();
}

void ExpectDecodedPattern(const OaJpegDecodeResult& InResult)
{
	ASSERT_EQ(InResult.Width, 320);
	ASSERT_EQ(InResult.Height, 180);
	ASSERT_EQ(InResult.Channels, 3);
	ASSERT_EQ(InResult.Pixels.Size(), static_cast<OaUsize>(320 * 180 * 3));

	OaU32 nonZeroCount = 0;
	OaU8 minValue = 255;
	OaU8 maxValue = 0;
	for (OaU8 value : InResult.Pixels) {
		nonZeroCount += value != 0 ? 1u : 0u;
		minValue = value < minValue ? value : minValue;
		maxValue = value > maxValue ? value : maxValue;
	}
	EXPECT_GT(nonZeroCount, 0u);
	EXPECT_GT(maxValue, minValue);
}

} // namespace

TEST(JpegDecoder, DecodeMemory_InvalidData)
{
	OaVec<OaU8> emptyData;
	auto result = OaJpegDecoder::Decode(OaSpan<const OaU8>(emptyData.Data(), emptyData.Size()));

	EXPECT_EQ(result.Width, 0);
	EXPECT_EQ(result.Height, 0);
	EXPECT_EQ(result.Channels, 0);
}

TEST(JpegDecoder, DecodeMemory_PngFixture)
{
	auto path = OaTestAssetPath("Image/VisionTestPattern320x180.png");
	auto bytes = OaFileIo::ReadBinary(path);
	ASSERT_TRUE(bytes.IsOk()) << bytes.GetStatus().ToString();

	auto result = OaJpegDecoder::Decode(OaSpan<const OaU8>(bytes->Data(), bytes->Size()));
	ExpectDecodedPattern(result);
}

TEST(JpegDecoder, DecodeFile_JpegFixture)
{
	OaString path = VisionAssetString("Image/VisionTestPattern320x180.jpg");
	auto result = OaJpegDecoder::DecodeFile(path);
	ExpectDecodedPattern(result);
}

TEST(JpegDecoder, DecodeFile_InvalidPath)
{
	auto result = OaJpegDecoder::DecodeFile("nonexistent_file.jpg");

	EXPECT_EQ(result.Width, 0);
	EXPECT_EQ(result.Height, 0);
	EXPECT_EQ(result.Channels, 0);
}

TEST_F(OaVkEngineTestFixture, DecodeToGpu_InvalidData)
{
	OaVec<OaU8> emptyData;
	auto tensor = OaJpegDecoder::DecodeToGpu(
		Rt(),
		OaSpan<const OaU8>(emptyData.Data(), emptyData.Size()),
		224,
		224,
		false);

	EXPECT_TRUE(tensor.IsEmpty());
}

TEST_F(OaVkEngineTestFixture, DecodeFileToGpu_InvalidPath)
{
	auto tensor = OaJpegDecoder::DecodeFileToGpu(
		Rt(),
		"nonexistent_file.jpg",
		224,
		224,
		false);

	EXPECT_TRUE(tensor.IsEmpty());
}

TEST_F(OaVkEngineTestFixture, DecodeFileToGpu_JpegFixture)
{
	OaString path = VisionAssetString("Image/VisionTestPattern320x180.jpg");
	auto tensor = OaJpegDecoder::DecodeFileToGpu(Rt(), path, 64, 64, false);

	ASSERT_FALSE(tensor.IsEmpty());
	OaExpectShape(tensor, {1, 3, 64, 64});
	OaExpectFinite(tensor);

	OaU32 nonZeroCount = 0;
	for (OaI64 i = 0; i < tensor.NumElements(); ++i) {
		const OaF32 value = tensor.At(i);
		EXPECT_GE(value, 0.0f);
		EXPECT_LE(value, 1.0f);
		nonZeroCount += value > 0.0f ? 1u : 0u;
	}
	EXPECT_GT(nonZeroCount, 0u);
}

TEST_F(OaVkEngineTestFixture, DecodeFileToGpu_NativePixelsMatchCpuDecode)
{
	OaString path = VisionAssetString("Image/VisionTestPattern320x180.png");
	auto decoded = OaJpegDecoder::DecodeFile(path);
	auto tensor = OaJpegDecoder::DecodeFileToGpu(Rt(), path, 0, 0, false);

	ASSERT_FALSE(tensor.IsEmpty());
	OaExpectShape(tensor, {1, 3, 180, 320});
	for (OaI32 c = 0; c < 3; ++c) {
		for (OaI32 y : {0, 17, 91, 179}) {
			for (OaI32 x : {0, 23, 177, 319}) {
				const OaI64 tensorIndex =
					static_cast<OaI64>(c) * 180 * 320 + y * 320 + x;
				const OaI64 pixelIndex =
					(static_cast<OaI64>(y) * 320 + x) * 3 + c;
				EXPECT_NEAR(tensor.At(tensorIndex),
					static_cast<OaF32>(decoded.Pixels[pixelIndex]) / 255.0F,
					1.0e-6F);
			}
		}
	}
}

TEST_F(OaVkEngineTestFixture, DecodeMemoryAndFileUseSameTensorContract)
{
	auto path = OaTestAssetPath("Image/VisionTestPattern320x180.png");
	auto bytes = OaFileIo::ReadBinary(path);
	ASSERT_TRUE(bytes.IsOk()) << bytes.GetStatus().ToString();
	auto fromMemory = OaJpegDecoder::DecodeToGpu(
		Rt(), OaSpan<const OaU8>(bytes->Data(), bytes->Size()), 0, 0, false);
	auto fromFile = OaJpegDecoder::DecodeFileToGpu(Rt(), path.String(), 0, 0, false);

	ASSERT_EQ(fromMemory.GetShape(), fromFile.GetShape());
	for (OaI64 i = 0; i < fromMemory.NumElements(); i += 997) {
		EXPECT_NEAR(fromMemory.At(i), fromFile.At(i), 1.0e-6F);
	}
}

TEST_F(OaVkEngineTestFixture, DecodeFileToGpu_NormalizedFixture)
{
	OaString path = VisionAssetString("Image/VisionTestPattern320x180.png");
	auto tensor = OaJpegDecoder::DecodeFileToGpu(Rt(), path, 32, 32, true);

	ASSERT_FALSE(tensor.IsEmpty());
	OaExpectShape(tensor, {1, 3, 32, 32});
	OaExpectFinite(tensor);

	OaF32 minValue = tensor.At(0);
	OaF32 maxValue = tensor.At(0);
	for (OaI64 i = 0; i < tensor.NumElements(); ++i) {
		const OaF32 value = tensor.At(i);
		minValue = value < minValue ? value : minValue;
		maxValue = value > maxValue ? value : maxValue;
	}
	EXPECT_LT(minValue, 0.0f);
	EXPECT_GT(maxValue, 1.0f);
}

TEST_F(OaVkEngineTestFixture, DecodeBatch_EmptyList)
{
	OaVec<OaString> paths;
	OaVec<OaMatrix> tensors;

	auto status = OaJpegDecoder::DecodeBatchToGpu(Rt(), paths, tensors, 224, 224);

	EXPECT_TRUE(status.IsOk());
	EXPECT_EQ(tensors.Size(), 0);
}

TEST_F(OaVkEngineTestFixture, DecodeBatch_InvalidPaths)
{
	OaVec<OaString> paths = {"missing1.jpg", "missing2.jpg"};
	OaVec<OaMatrix> matrices;

	auto status = OaJpegDecoder::DecodeBatchToGpu(Rt(), paths, matrices, 224, 224);

	EXPECT_FALSE(status.IsOk());
}

TEST_F(OaVkEngineTestFixture, DecodeBatch_Fixtures)
{
	OaVec<OaString> paths = {
		VisionAssetString("Image/VisionTestPattern320x180.jpg"),
		VisionAssetString("Image/VisionTestPattern320x180.png")
	};
	OaVec<OaMatrix> tensors;

	auto status = OaJpegDecoder::DecodeBatchToGpu(Rt(), paths, tensors, 48, 48);

	ASSERT_TRUE(status.IsOk()) << status.ToString();
	ASSERT_EQ(tensors.Size(), 2u);
	for (const OaMatrix& tensor : tensors) {
		ASSERT_FALSE(tensor.IsEmpty());
		OaExpectShape(tensor, {1, 3, 48, 48});
		OaExpectFinite(tensor);
	}
}
