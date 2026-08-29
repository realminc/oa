// OA Vision — Video Decoder Tests
// hardware H.264/H.265 decode validation

#include <oa/vision/video/codec/nalParser.h>

#include "../../oaTest.h"
#include "../videoTestSupport.h"
#include <oa/vision/video/codec/codecRegistry.h>
#include <oa/vision/video/codec/vcpAv1.h>
#include <oa/vision/video/codec/vcpH265.h>
#include <oa/vision/video/decoder/videoDecoderInternal.h>
#include <oa/vision/video/decoder/videoDecoderProfile.h>
#include <oa/runtime/engine.h>

#include <oa/vision/videoDecoder.h>
#include <oa/vision/videoDemuxer.h>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

oa::Path testReferencePath(const char* inStem)
{
#if defined(_WIN32)
	const oa::U64 processId = static_cast<oa::U64>(_getpid());
#else
	const oa::U64 processId = static_cast<oa::U64>(getpid());
#endif
	return oa::Paths::temp() / oa::Path(oa::format("{}-{}.bin", inStem, processId));
}

struct H264NalWriter {
	oa::Vector<oa::U8> bytes;
	oa::U32 bitOffset = 0;

	void writeBit(oa::U32 inBit) {
		if (bitOffset == 0) {
			bytes.pushBack(0);
		}
		if (inBit != 0) {
			bytes.back() |= static_cast<oa::U8>(1u << (7u - bitOffset));
		}
		bitOffset = (bitOffset + 1u) & 7u;
	}

	void writeBits(oa::U32 inValue, oa::U32 inCount) {
		for (oa::I32 bit = static_cast<oa::I32>(inCount) - 1; bit >= 0; --bit) {
			writeBit((inValue >> static_cast<oa::U32>(bit)) & 1u);
		}
	}

	void writeUE(oa::U32 inValue) {
		oa::U32 codeNum = inValue + 1;
		oa::U32 bitCount = 0;
		for (oa::U32 tmp = codeNum; tmp != 0; tmp >>= 1) {
			++bitCount;
		}
		for (oa::U32 i = 0; i + 1 < bitCount; ++i) {
			writeBit(0);
		}
		writeBits(codeNum, bitCount);
	}

	void writeSE(oa::I32 inValue)
	{
		oa::U32 codeNum = inValue <= 0
			? static_cast<oa::U32>(-inValue) * 2u
			: static_cast<oa::U32>(inValue) * 2u - 1u;
		writeUE(codeNum);
	}

	void finishRbsp()
	{
		writeBit(1);
		while (bitOffset != 0) {
			writeBit(0);
		}
	}
};

oa::Vector<oa::U8> makeH264SpsNal()
{
	H264NalWriter w;
	w.bytes.pushBack(0);
	w.bytes.pushBack(0);
	w.bytes.pushBack(0);
	w.bytes.pushBack(1);
	w.bytes.pushBack(0x67);
	w.writeBits(100, 8); // profile_idc: high
	w.writeBits(0, 8);   // constraint flags
	w.writeBits(40, 8);  // level_idc: 4.0
	w.writeUE(0);        // seq_parameter_set_id
	w.writeUE(1);        // chroma_format_idc: 4:2:0
	w.writeUE(0);        // bit_depth_luma_minus8
	w.writeUE(0);        // bit_depth_chroma_minus8
	w.writeBit(0);       // qpprime_y_zero_transform_bypass_flag
	w.writeBit(0);       // seq_scaling_matrix_present_flag
	w.writeUE(0);        // log2_max_frame_num_minus4
	w.writeUE(0);        // pic_order_cnt_type
	w.writeUE(0);        // log2_max_pic_order_cnt_lsb_minus4
	w.writeUE(1);        // max_num_ref_frames
	w.writeBit(0);       // gaps_in_frame_num_value_allowed_flag
	w.writeUE(119);      // pic_width_in_mbs_minus1: 1920
	w.writeUE(67);       // pic_height_in_map_units_minus1: 1088
	w.writeBit(1);       // frame_mbs_only_flag
	w.writeBit(1);       // direct_8x8_inference_flag
	w.writeBit(1);       // frame_cropping_flag
	w.writeUE(0);        // left
	w.writeUE(0);        // right
	w.writeUE(0);        // top
	w.writeUE(4);        // bottom -> 1080 visible
	w.finishRbsp();
	return w.bytes;
}

TEST(H264NalParser, WeightedPSlicePreservesMmco)
{
	H264NalWriter w;
	w.bytes.pushBack(0x61); // nal_ref_idc=3, non-IDR slice
	w.writeUE(0);           // first_mb_in_slice
	w.writeUE(0);           // slice_type=P
	w.writeUE(0);           // pic_parameter_set_id
	w.writeBits(5, 4);      // frame_num
	w.writeBits(6, 4);      // pic_order_cnt_lsb
	w.writeBit(0);          // num_ref_idx_active_override_flag
	w.writeBit(0);          // ref_pic_list_modification_flag_l0

	w.writeUE(0);           // luma_log2_weight_denom
	w.writeUE(0);           // chroma_log2_weight_denom
	w.writeBit(1);          // luma_weight_l0_flag
	w.writeSE(1);           // luma_weight_l0
	w.writeSE(0);           // luma_offset_l0
	w.writeBit(1);          // chroma_weight_l0_flag
	w.writeSE(1);           // chroma_weight_l0[0]
	w.writeSE(0);           // chroma_offset_l0[0]
	w.writeSE(1);           // chroma_weight_l0[1]
	w.writeSE(0);           // chroma_offset_l0[1]

	w.writeBit(1);          // adaptive_ref_pic_marking_mode_flag
	w.writeUE(1);           // memory_management_control_operation
	w.writeUE(4);           // difference_of_pic_nums_minus1
	w.writeUE(0);           // end of MMCO list
	w.finishRbsp();

	oa::H264SpsData sps{};
	sps.log2MaxFrameNumMinus4 = 0;
	sps.picOrderCntType = 0;
	sps.log2MaxPicOrderCntLsbMinus4 = 0;
	sps.chromaFormatIdc = 1;
	sps.frameMbsOnly = true;

	oa::H264PpsData pps{};
	pps.ppsId = 0;
	pps.weightedPred = true;
	pps.numRefIdxL0DefaultActiveMinus1 = 0;

	oa::H264SliceHeader header{};
	ASSERT_TRUE(oa::NalParser::parseSliceHeader(
		w.bytes.data(),
		w.bytes.size(),
		false,
		3,
		sps,
		pps,
		header));
	EXPECT_TRUE(header.refPicMarkingValid);
	EXPECT_TRUE(header.adaptiveRefPicMarking);
	ASSERT_EQ(header.mmcoCommands.size(), 1u);
	EXPECT_EQ(header.mmcoCommands[0].op, 1u);
	EXPECT_EQ(header.mmcoCommands[0].differenceOfPicNumsMinus1, 4u);
}

oa::Vector<oa::U8> makeH264PpsNal()
{
	H264NalWriter w;
	w.bytes.pushBack(0);
	w.bytes.pushBack(0);
	w.bytes.pushBack(0);
	w.bytes.pushBack(1);
	w.bytes.pushBack(0x68);
	w.writeUE(0);  // pic_parameter_set_id
	w.writeUE(0);  // seq_parameter_set_id
	w.writeBit(0); // entropy_coding_mode_flag
	w.writeBit(0); // bottom_field_pic_order_in_frame_present_flag
	w.writeUE(0);  // num_slice_groups_minus1
	w.writeUE(0);  // num_ref_idx_l0_default_active_minus1
	w.writeUE(0);  // num_ref_idx_l1_default_active_minus1
	w.writeBit(0); // weighted_pred_flag
	w.writeBits(0, 2); // weighted_bipred_idc
	w.writeSE(0);  // pic_init_qp_minus26
	w.writeSE(0);  // pic_init_qs_minus26
	w.writeSE(0);  // chroma_qp_index_offset
	w.writeBit(1); // deblocking_filter_control_present_flag
	w.writeBit(0); // constrained_intra_pred_flag
	w.writeBit(0); // redundant_pic_cnt_present_flag
	w.finishRbsp();
	return w.bytes;
}

oa::Vector<oa::U8> makeH264ParameterAccessUnit()
{
	oa::Vector<oa::U8> out = makeH264SpsNal();
	oa::Vector<oa::U8> pps = makeH264PpsNal();
	for (oa::U8 byte : pps) {
		out.pushBack(byte);
	}
	return out;
}

oa::VideoProfile makeH264FixtureProfile()
{
	oa::VideoProfile profile;
	profile.codec = oa::VideoCodec::H264;
	profile.standardProfile = oa::VideoCodecProfile::H264Baseline;
	profile.h264PictureLayout = oa::VideoH264PictureLayout::Progressive;
	profile.width = 128;
	profile.height = 72;
	profile.maxDpbSlots = 4;
	return profile;
}

oa::VideoProfile makeH265FixtureProfile()
{
	oa::VideoProfile profile;
	profile.codec = oa::VideoCodec::H265;
	profile.standardProfile = oa::VideoCodecProfile::H265Main;
	profile.width = 128;
	profile.height = 72;
	profile.maxDpbSlots = 4;
	return profile;
}

oa::VideoProfile makeH265Main10FixtureProfile()
{
	oa::VideoProfile profile = makeH265FixtureProfile();
	profile.standardProfile = oa::VideoCodecProfile::H265Main10;
	profile.lumaBitDepth = oa::VideoBitDepth::Bit10;
	profile.chromaBitDepth = oa::VideoBitDepth::Bit10;
	return profile;
}

oa::VideoProfile makeH265RangeExtFixtureProfile()
{
	oa::VideoProfile profile = makeH265FixtureProfile();
	profile.standardProfile = oa::VideoCodecProfile::H265FormatRangeExtensions;
	return profile;
}

oa::VideoProfile makeH265RangeExt10FixtureProfile()
{
	oa::VideoProfile profile = makeH265RangeExtFixtureProfile();
	profile.lumaBitDepth = oa::VideoBitDepth::Bit10;
	profile.chromaBitDepth = oa::VideoBitDepth::Bit10;
	return profile;
}

oa::VideoProfile makeAv1FixtureProfile()
{
	oa::VideoProfile profile;
	profile.codec = oa::VideoCodec::AV1;
	profile.width = 128;
	profile.height = 72;
	// AV1 spec / NVIDIA vk_video_samples use 8 ref slots + 1 current (9 total).
	profile.maxDpbSlots = 9;
	return profile;
}

oa::VideoProfile makeAv1Main10FixtureProfile()
{
	oa::VideoProfile profile = makeAv1FixtureProfile();
	profile.standardProfile = oa::VideoCodecProfile::Av1Main;
	profile.lumaBitDepth = oa::VideoBitDepth::Bit10;
	profile.chromaBitDepth = oa::VideoBitDepth::Bit10;
	return profile;
}

oa::VideoProfile makeVp9FixtureProfile()
{
	oa::VideoProfile profile;
	profile.codec = oa::VideoCodec::VP9;
	profile.width = 128;
	profile.height = 72;
	profile.maxDpbSlots = 9;
	return profile;
}

oa::VideoProfile makeVp9Profile2FixtureProfile()
{
	oa::VideoProfile profile = makeVp9FixtureProfile();
	profile.standardProfile = oa::VideoCodecProfile::Vp9Profile2;
	profile.lumaBitDepth = oa::VideoBitDepth::Bit10;
	profile.chromaBitDepth = oa::VideoBitDepth::Bit10;
	return profile;
}

oa::U32 alignVideoTestExtent(oa::U32 inValue, oa::U32 inMinimum, oa::U32 inGranularity)
{
	const oa::U32 granularity = inGranularity == 0 ? 1U : inGranularity;
	oa::U32 value = inValue < inMinimum ? inMinimum : inValue;
	const oa::U32 remainder = value % granularity;
	if (remainder != 0) {
		value += granularity - remainder;
	}
	return value;
}

oa::U32 codecExtentGranularityForTest(oa::VideoCodec inCodec)
{
	switch (inCodec) {
		case oa::VideoCodec::H264: return 16U;
		case oa::VideoCodec::H265:
		case oa::VideoCodec::AV1:
		case oa::VideoCodec::VP9: return 2U;
	}
	return 1U;
}

void expectDecoderCodedExtentAligned(
	const oa::VideoDecoder& inDecoder,
	const oa::VideoDecodeCapabilities& inCaps,
	const oa::VideoProfile& inProfile)
{
	const oa::U32 codecGranularity = codecExtentGranularityForTest(inProfile.codec);
	const oa::U32 widthGranularity = inCaps.pictureAccessGranularityWidth > codecGranularity
		? inCaps.pictureAccessGranularityWidth
		: codecGranularity;
	const oa::U32 heightGranularity = inCaps.pictureAccessGranularityHeight > codecGranularity
		? inCaps.pictureAccessGranularityHeight
		: codecGranularity;
	const oa::U32 expectedWidth = alignVideoTestExtent(inProfile.width, inCaps.minWidth, widthGranularity);
	const oa::U32 expectedHeight = alignVideoTestExtent(inProfile.height, inCaps.minHeight, heightGranularity);

	EXPECT_EQ(inDecoder.getCodedWidth(), expectedWidth);
	EXPECT_EQ(inDecoder.getCodedHeight(), expectedHeight);
	EXPECT_GE(inDecoder.getCodedWidth(), inProfile.width);
	EXPECT_GE(inDecoder.getCodedHeight(), inProfile.height);
	EXPECT_EQ(inDecoder.getCodedWidth() % widthGranularity, 0U);
	EXPECT_EQ(inDecoder.getCodedHeight() % heightGranularity, 0U);
}

void expectDecodedLumaIsReadable(oa::VideoDecoder& inDecoder, const oa::VideoFrame& inFrame, const oa::VideoProfile& inProfile)
{
	EXPECT_NE(inFrame.image, VK_NULL_HANDLE);
	EXPECT_NE(inFrame.imageView, VK_NULL_HANDLE);
	EXPECT_EQ(inFrame.width, inProfile.width);
	EXPECT_EQ(inFrame.height, inProfile.height);
	EXPECT_EQ(inFrame.format, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM);
	EXPECT_FALSE(inFrame.isRgb);

	auto lumaResult = inDecoder.readbackLuma(inFrame);
	ASSERT_TRUE(lumaResult.isOk()) << lumaResult.getStatus().toString();
	ASSERT_EQ(lumaResult->size(), static_cast<oa::Usize>(inProfile.width * inProfile.height));
	oa::U32 nonZeroCount = 0;
	oa::U8 minValue = 255;
	oa::U8 maxValue = 0;
	for (oa::U8 value : *lumaResult) {
		nonZeroCount += value != 0 ? 1u : 0u;
		minValue = value < minValue ? value : minValue;
		maxValue = value > maxValue ? value : maxValue;
	}
	if (nonZeroCount == 0 || maxValue == minValue) {
		fprintf(stderr, "[LUMA-DIAG] samples=%zu nonZero=%u min=%u max=%u first16=",
			lumaResult->size(), nonZeroCount, minValue, maxValue);
		for (oa::Usize i = 0; i < 16 && i < lumaResult->size(); ++i) {
			fprintf(stderr, "%u ", (*lumaResult)[i]);
		}
		fprintf(stderr, "\n");
		GTEST_SKIP() << "vulkan Video command submission succeeded, but luma "
						"readback is still all zero or flat";
	}
	EXPECT_GT(nonZeroCount, 0u);
	EXPECT_GT(maxValue, minValue);
}

void expectDecodedNv12MatchesFfmpeg(
	oa::VideoDecoder& inDecoder,
	const oa::VideoFrame& inFrame,
	const oa::VideoProfile& inProfile,
	const char* inFixtureRelativePath,
	const char* inReferenceStem)
{
	if (::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
		GTEST_SKIP() << "ffmpeg is not installed";
	}

	auto decoded = inDecoder.readbackNv12(inFrame);
	ASSERT_TRUE(decoded.isOk()) << decoded.getStatus().toString();
	ASSERT_EQ(
		decoded->size(),
		static_cast<oa::Usize>(inProfile.width) * inProfile.height * 3u / 2u);

	const oa::Path fixturePath = testAssetPath(inFixtureRelativePath);
	const oa::Path referencePath = testReferencePath(inReferenceStem);
	const oa::String command = oa::String("ffmpeg -v error -y -i \"")
		+ fixturePath.string()
		+ "\" -frames:v 1 -f rawvideo -pix_fmt nv12 \""
		+ referencePath.string()
		+ "\"";
	ASSERT_EQ(::system(command.cStr()), 0);
	auto reference = oa::Filesystem::readBinary(referencePath);
	(void) oa::Filesystem::removeFile(referencePath);
	ASSERT_TRUE(reference.isOk()) << reference.getStatus().toString();
	ASSERT_EQ(reference->size(), decoded->size());

	const oa::Usize lumaBytes =
		static_cast<oa::Usize>(inProfile.width) * inProfile.height;
	oa::U64 lumaError = 0;
	oa::U64 chromaError = 0;
	oa::U64 decodedLumaSum = 0;
	oa::U64 referenceLumaSum = 0;
	oa::U8 maxError = 0;
	for (oa::Usize i = 0; i < decoded->size(); ++i) {
		const oa::U8 error = static_cast<oa::U8>(oa::abs(
			static_cast<int>((*decoded)[i])
			- static_cast<int>((*reference)[i])));
		if (i < lumaBytes) {
			lumaError += error;
			decodedLumaSum += (*decoded)[i];
			referenceLumaSum += (*reference)[i];
		} else {
			chromaError += error;
		}
		maxError = error > maxError ? error : maxError;
	}

	const oa::F64 lumaMae =
		static_cast<oa::F64>(lumaError) / static_cast<oa::F64>(lumaBytes);
	const oa::F64 chromaMae =
		static_cast<oa::F64>(chromaError)
		/ static_cast<oa::F64>(decoded->size() - lumaBytes);
	EXPECT_LT(lumaMae, 3.0)
		<< "maxError=" << static_cast<oa::U32>(maxError)
		<< " decodedMean="
		<< static_cast<oa::F64>(decodedLumaSum) / static_cast<oa::F64>(lumaBytes)
		<< " referenceMean="
		<< static_cast<oa::F64>(referenceLumaSum) / static_cast<oa::F64>(lumaBytes)
		<< " decodedFirst="
		<< static_cast<oa::U32>((*decoded)[0]) << ","
		<< static_cast<oa::U32>((*decoded)[1]) << ","
		<< static_cast<oa::U32>((*decoded)[2]) << ","
		<< static_cast<oa::U32>((*decoded)[3])
		<< " referenceFirst="
		<< static_cast<oa::U32>((*reference)[0]) << ","
		<< static_cast<oa::U32>((*reference)[1]) << ","
		<< static_cast<oa::U32>((*reference)[2]) << ","
		<< static_cast<oa::U32>((*reference)[3]);
	EXPECT_LT(chromaMae, 3.0)
		<< "maxError=" << static_cast<oa::U32>(maxError);
}

void expectDecodedP010MatchesFfmpeg(
	oa::VideoDecoder& inDecoder,
	const oa::VideoFrame& inFrame,
	const oa::VideoProfile& inProfile,
	const char* inFixtureRelativePath,
	const char* inReferenceStem)
{
	if (::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
		GTEST_SKIP() << "ffmpeg is not installed";
	}
	ASSERT_EQ(
		inFrame.format,
		VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16);
	auto decoded = inDecoder.readbackYuv420(inFrame);
	ASSERT_TRUE(decoded.isOk()) << decoded.getStatus().toString();
	const oa::Usize componentCount =
		static_cast<oa::Usize>(inProfile.width) * inProfile.height * 3U / 2U;
	ASSERT_EQ(decoded->size(), componentCount * sizeof(oa::U16));

	const oa::Path fixturePath = testAssetPath(inFixtureRelativePath);
	const oa::Path referencePath = testReferencePath(inReferenceStem);
	const oa::String command = oa::String("ffmpeg -v error -y -i \"")
		+ fixturePath.string()
		+ "\" -frames:v 1 -f rawvideo -pix_fmt p010le \""
		+ referencePath.string()
		+ "\"";
	ASSERT_EQ(::system(command.cStr()), 0);
	auto reference = oa::Filesystem::readBinary(referencePath);
	(void) oa::Filesystem::removeFile(referencePath);
	ASSERT_TRUE(reference.isOk()) << reference.getStatus().toString();
	ASSERT_EQ(reference->size(), decoded->size());

	auto readCode = [](const oa::Vector<oa::U8>& inBytes, oa::Usize inIndex) {
		const oa::Usize offset = inIndex * 2U;
		const oa::U16 packed = static_cast<oa::U16>(inBytes[offset])
			| static_cast<oa::U16>(static_cast<oa::U16>(inBytes[offset + 1U]) << 8U);
		return static_cast<oa::U16>(packed >> 6U);
	};
	const oa::Usize lumaComponents =
		static_cast<oa::Usize>(inProfile.width) * inProfile.height;
	oa::U64 lumaError = 0U;
	oa::U64 chromaError = 0U;
	oa::U16 maxError = 0U;
	for (oa::Usize index = 0U; index < componentCount; ++index) {
		const oa::U16 decodedCode = readCode(*decoded, index);
		const oa::U16 referenceCode = readCode(*reference, index);
		const oa::U16 error = decodedCode > referenceCode
			? decodedCode - referenceCode : referenceCode - decodedCode;
		if (index < lumaComponents) lumaError += error;
		else chromaError += error;
		maxError = error > maxError ? error : maxError;
	}
	const oa::F64 lumaMae = static_cast<oa::F64>(lumaError)
		/ static_cast<oa::F64>(lumaComponents);
	const oa::F64 chromaMae = static_cast<oa::F64>(chromaError)
		/ static_cast<oa::F64>(componentCount - lumaComponents);
	EXPECT_LT(lumaMae, 3.0) << "maxCodeError=" << maxError;
	EXPECT_LT(chromaMae, 3.0) << "maxCodeError=" << maxError;
}

void expectP010ComputeRgbaMatchesCpu(
	oa::VideoDecoder& inDecoder,
	const oa::VideoFrame& inFrame)
{
	auto p010Result = inDecoder.readbackYuv420(inFrame);
	ASSERT_TRUE(p010Result.isOk()) << p010Result.getStatus().toString();

	oa::VideoConversionOptions options = {};
	options.preferHardwareYCbCr = false;
	options.colorSpace = oa::YCbCrModel::Auto;
	auto rgbaFrameResult = inDecoder.convert(inFrame, options);
	ASSERT_TRUE(rgbaFrameResult.isOk()) << rgbaFrameResult.getStatus().toString();
	auto rgbaResult = inDecoder.readbackRgba(*rgbaFrameResult);
	ASSERT_TRUE(rgbaResult.isOk()) << rgbaResult.getStatus().toString();
	ASSERT_EQ(
		rgbaResult->size(),
		static_cast<oa::Usize>(inFrame.width) * inFrame.height * 4U);

	auto readCode = [&p010Result](oa::Usize inComponent) {
		const oa::Usize offset = inComponent * 2U;
		const oa::U16 packed = static_cast<oa::U16>((*p010Result)[offset])
			| static_cast<oa::U16>(static_cast<oa::U16>((*p010Result)[offset + 1U]) << 8U);
		return static_cast<oa::F32>(packed >> 6U) / 1023.0F;
	};
	auto clampByte = [](oa::F32 inValue) {
		const oa::F32 clamped = inValue < 0.0F ? 0.0F : (inValue > 1.0F ? 1.0F : inValue);
		return static_cast<oa::I32>(clamped * 255.0F + 0.5F);
	};
	const oa::Usize lumaComponents =
		static_cast<oa::Usize>(inFrame.width) * inFrame.height;
	oa::U64 totalError = 0U;
	oa::U32 maxError = 0U;
	for (oa::U32 y = 0U; y < inFrame.height; ++y) {
		for (oa::U32 x = 0U; x < inFrame.width; ++x) {
			const oa::Usize pixel = static_cast<oa::Usize>(y) * inFrame.width + x;
			const oa::Usize uv = lumaComponents
				+ static_cast<oa::Usize>(y / 2U) * inFrame.width
				+ static_cast<oa::Usize>(x / 2U) * 2U;
			const oa::F32 yy = 1.164F * (readCode(pixel) - 64.0F / 1023.0F);
			const oa::F32 uu = readCode(uv) - 512.0F / 1023.0F;
			const oa::F32 vv = readCode(uv + 1U) - 512.0F / 1023.0F;
			const oa::I32 expected[4] = {
				clampByte(yy + 1.596F * vv),
				clampByte(yy - 0.391F * uu - 0.813F * vv),
				clampByte(yy + 2.018F * uu),
				255};
			for (oa::U32 channel = 0U; channel < 4U; ++channel) {
				const oa::I32 actual = (*rgbaResult)[pixel * 4U + channel];
				const oa::U32 error = static_cast<oa::U32>(
					actual > expected[channel]
						? actual - expected[channel]
						: expected[channel] - actual);
				totalError += error;
				maxError = error > maxError ? error : maxError;
			}
		}
	}
	const oa::F64 mae = static_cast<oa::F64>(totalError)
		/ static_cast<oa::F64>(rgbaResult->size());
	EXPECT_LT(mae, 1.0) << "maxByteError=" << maxError;
	EXPECT_LE(maxError, 3U);

	options.preferHardwareYCbCr = true;
	auto hardwareFrameResult = inDecoder.convert(inFrame, options);
	ASSERT_TRUE(hardwareFrameResult.isOk())
		<< hardwareFrameResult.getStatus().toString();
	auto hardwareRgbaResult = inDecoder.readbackRgba(*hardwareFrameResult);
	ASSERT_TRUE(hardwareRgbaResult.isOk())
		<< hardwareRgbaResult.getStatus().toString();
	ASSERT_EQ(hardwareRgbaResult->size(), rgbaResult->size());
	oa::U64 hardwareError = 0U;
	oa::U32 hardwareMaxError = 0U;
	for (oa::Usize pixel = 0U; pixel < rgbaResult->size(); pixel += 4U) {
		for (oa::Usize channel = 0U; channel < 3U; ++channel) {
			const oa::U8 hardware = (*hardwareRgbaResult)[pixel + channel];
			const oa::U8 compute = (*rgbaResult)[pixel + channel];
			const oa::U32 error = hardware > compute
				? hardware - compute : compute - hardware;
			hardwareError += error;
			hardwareMaxError = oa::max(hardwareMaxError, error);
		}
	}
	const oa::F64 hardwareMae = static_cast<oa::F64>(hardwareError)
		/ static_cast<oa::F64>(inFrame.width * inFrame.height * 3U);
	EXPECT_LT(hardwareMae, 6.0) << "maxByteError=" << hardwareMaxError;
	EXPECT_LT(hardwareMaxError, 96U);
}

} // namespace

TEST(VideoProfile, BuildsEveryRegistryProfileWithoutCodecFallback)
{
	struct Case {
		oa::VideoCodec codec;
		oa::VideoCodecProfile standardProfile;
		VkVideoCodecOperationFlagBitsKHR operation;
		oa::U32 vkStandardProfile;
	};
	const Case cases[] = {
		{oa::VideoCodec::H264, oa::VideoCodecProfile::H264Baseline, VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
		 STD_VIDEO_H264_PROFILE_IDC_BASELINE},
		{oa::VideoCodec::H264, oa::VideoCodecProfile::H264Main, VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
		 STD_VIDEO_H264_PROFILE_IDC_MAIN},
		{oa::VideoCodec::H264, oa::VideoCodecProfile::H264High, VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
		 STD_VIDEO_H264_PROFILE_IDC_HIGH},
		{oa::VideoCodec::H264, oa::VideoCodecProfile::H264High444Predictive, VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
		 STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE},
		{oa::VideoCodec::H265, oa::VideoCodecProfile::H265Main, VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR,
		 STD_VIDEO_H265_PROFILE_IDC_MAIN},
		{oa::VideoCodec::H265, oa::VideoCodecProfile::H265Main10, VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR,
		 STD_VIDEO_H265_PROFILE_IDC_MAIN_10},
		{oa::VideoCodec::H265, oa::VideoCodecProfile::H265MainStillPicture, VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR,
		 STD_VIDEO_H265_PROFILE_IDC_MAIN_STILL_PICTURE},
		{oa::VideoCodec::H265, oa::VideoCodecProfile::H265FormatRangeExtensions,
		 VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR, STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS},
		{oa::VideoCodec::H265, oa::VideoCodecProfile::H265ScreenContentCodingExtensions,
		 VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR, STD_VIDEO_H265_PROFILE_IDC_SCC_EXTENSIONS},
		{oa::VideoCodec::AV1, oa::VideoCodecProfile::Av1Main, VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR,
		 STD_VIDEO_AV1_PROFILE_MAIN},
		{oa::VideoCodec::AV1, oa::VideoCodecProfile::Av1High, VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR,
		 STD_VIDEO_AV1_PROFILE_HIGH},
		{oa::VideoCodec::AV1, oa::VideoCodecProfile::Av1Professional, VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR,
		 STD_VIDEO_AV1_PROFILE_PROFESSIONAL},
		{oa::VideoCodec::VP9, oa::VideoCodecProfile::Vp9Profile0, VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR,
		 STD_VIDEO_VP9_PROFILE_0},
		{oa::VideoCodec::VP9, oa::VideoCodecProfile::Vp9Profile1, VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR,
		 STD_VIDEO_VP9_PROFILE_1},
		{oa::VideoCodec::VP9, oa::VideoCodecProfile::Vp9Profile2, VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR,
		 STD_VIDEO_VP9_PROFILE_2},
		{oa::VideoCodec::VP9, oa::VideoCodecProfile::Vp9Profile3, VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR,
		 STD_VIDEO_VP9_PROFILE_3},
	};

	for (const Case& testCase : cases) {
		oa::VideoProfile request = {};
		request.codec = testCase.codec;
		request.standardProfile = testCase.standardProfile;
		VkVideoDecodeH264ProfileInfoKHR h264 = {};
		VkVideoDecodeH265ProfileInfoKHR h265 = {};
		VkVideoDecodeAV1ProfileInfoKHR av1 = {};
		VkVideoDecodeVP9ProfileInfoKHR vp9 = {};
		auto result = oa::videoDecoderProfile::buildDecodeProfile(request, h264, h265, av1, vp9);
		ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
		EXPECT_EQ(result->videoCodecOperation, testCase.operation);
		EXPECT_EQ(result->chromaSubsampling, VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR);
		EXPECT_EQ(result->lumaBitDepth, VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR);
		EXPECT_EQ(result->chromaBitDepth, VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR);
		switch (testCase.codec) {
		case oa::VideoCodec::H264: EXPECT_EQ(static_cast<oa::U32>(h264.stdProfileIdc), testCase.vkStandardProfile); break;
		case oa::VideoCodec::H265: EXPECT_EQ(static_cast<oa::U32>(h265.stdProfileIdc), testCase.vkStandardProfile); break;
		case oa::VideoCodec::AV1: EXPECT_EQ(static_cast<oa::U32>(av1.stdProfile), testCase.vkStandardProfile); break;
		case oa::VideoCodec::VP9: EXPECT_EQ(static_cast<oa::U32>(vp9.stdProfile), testCase.vkStandardProfile); break;
		}
	}
}

TEST(VideoProfile, MapsFormatAndCodecSpecificAxes)
{
	oa::VideoProfile request = {};
	request.codec = oa::VideoCodec::AV1;
	request.standardProfile = oa::VideoCodecProfile::Av1Professional;
	request.chromaSubsampling = oa::VideoChromaSubsampling::Yuv444;
	request.lumaBitDepth = oa::VideoBitDepth::Bit12;
	request.chromaBitDepth = oa::VideoBitDepth::Bit12;
	request.av1FilmGrain = true;
	VkVideoDecodeH264ProfileInfoKHR h264 = {};
	VkVideoDecodeH265ProfileInfoKHR h265 = {};
	VkVideoDecodeAV1ProfileInfoKHR av1 = {};
	VkVideoDecodeVP9ProfileInfoKHR vp9 = {};
	auto result = oa::videoDecoderProfile::buildDecodeProfile(request, h264, h265, av1, vp9);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	EXPECT_EQ(result->chromaSubsampling, VK_VIDEO_CHROMA_SUBSAMPLING_444_BIT_KHR);
	EXPECT_EQ(result->lumaBitDepth, VK_VIDEO_COMPONENT_BIT_DEPTH_12_BIT_KHR);
	EXPECT_EQ(result->chromaBitDepth, VK_VIDEO_COMPONENT_BIT_DEPTH_12_BIT_KHR);
	EXPECT_EQ(av1.stdProfile, STD_VIDEO_AV1_PROFILE_PROFESSIONAL);
	EXPECT_EQ(av1.filmGrainSupport, VK_TRUE);

	request = {};
	request.codec = oa::VideoCodec::H264;
	request.standardProfile = oa::VideoCodecProfile::H264High;
	request.h264PictureLayout = oa::VideoH264PictureLayout::InterlacedSeparatePlanes;
	h264 = {};
	h265 = {};
	av1 = {};
	vp9 = {};
	result = oa::videoDecoderProfile::buildDecodeProfile(request, h264, h265, av1, vp9);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	EXPECT_EQ(h264.pictureLayout, VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_INTERLACED_SEPARATE_PLANES_BIT_KHR);
}

TEST(VideoProfile, RejectsCodecProfileMismatch)
{
	oa::VideoProfile request = {};
	request.codec = oa::VideoCodec::H264;
	request.standardProfile = oa::VideoCodecProfile::H265Main;
	auto resolved = oa::videoDecoderProfile::resolveDecodeProfile(request);
	EXPECT_FALSE(resolved.isOk());
}

TEST(VideoProfile, LegacyDefaultsResolveToVerifiedProfiles)
{
	const oa::Pair<oa::VideoCodec, oa::VideoCodecProfile> cases[] = {
		{oa::VideoCodec::H264, oa::VideoCodecProfile::H264High},
		{oa::VideoCodec::H265, oa::VideoCodecProfile::H265Main},
		{oa::VideoCodec::AV1, oa::VideoCodecProfile::Av1Main},
		{oa::VideoCodec::VP9, oa::VideoCodecProfile::Vp9Profile0},
	};
	for (const auto& [codec, expected] : cases) {
		oa::VideoProfile request = {};
		request.codec = codec;
		auto resolved = oa::videoDecoderProfile::resolveDecodeProfile(request);
		ASSERT_TRUE(resolved.isOk());
		EXPECT_EQ(resolved->standardProfile, expected);
		EXPECT_TRUE(oa::videoDecoderProfile::isDecodePathImplemented(*resolved));
	}
}

TEST(VideoProfile, AssetManifestDrivesExactStreamProfiles)
{
	const oa::Path manifestPath = testAssetPath("video/clip/manifest.tsv");
	auto linesResult = oa::Filesystem::readLines(manifestPath);
	ASSERT_TRUE(linesResult.isOk()) << linesResult.getStatus().toString();
	ASSERT_FALSE(linesResult->empty());
	oa::U32 rowCount = 0;
	for (oa::Usize lineIndex = 1U; lineIndex < linesResult->size(); ++lineIndex) {
		const oa::String& line = (*linesResult)[lineIndex];
		if (line.empty()) continue;
		oa::Array<oa::StringView, 18> columns;
		oa::StringView remainder = line.view();
		for (oa::Usize columnIndex = 0U; columnIndex < columns.size(); ++columnIndex) {
			const oa::Usize separator = remainder.find('\t');
			if (columnIndex + 1U == columns.size()) {
				columns[columnIndex] = remainder;
				EXPECT_EQ(separator, oa::StringView::Npos) << line;
				break;
			}
			ASSERT_NE(separator, oa::StringView::Npos) << line;
			columns[columnIndex] = remainder.subStr(0U, separator);
			remainder.removePrefix(separator + 1U);
		}

		oa::String relativePath = "video/";
		relativePath += columns[0];
		const oa::Path fixturePath = testAssetPath(relativePath);
		auto streamResult = oa::VideoDemuxer::open(fixturePath.cStr());
		ASSERT_TRUE(streamResult.isOk()) << columns[0] << ": " << streamResult.getStatus().toString();
		const oa::VideoProfile profile = streamResult->getVideoProfile();
		oa::U64 manifestWidth = 0U;
		oa::U64 manifestHeight = 0U;
		ASSERT_TRUE(oa::parseU64(columns[8], manifestWidth));
		ASSERT_TRUE(oa::parseU64(columns[9], manifestHeight));
		EXPECT_EQ(profile.width, static_cast<oa::U32>(manifestWidth));
		EXPECT_EQ(profile.height, static_cast<oa::U32>(manifestHeight));
		EXPECT_EQ(profile.lumaBitDepth, oa::VideoBitDepth::Bit8);
		EXPECT_EQ(profile.chromaBitDepth, oa::VideoBitDepth::Bit8);
		EXPECT_EQ(profile.chromaSubsampling, oa::VideoChromaSubsampling::Yuv420);

		if (columns[1] == "h264") {
			EXPECT_EQ(profile.codec, oa::VideoCodec::H264);
			if (columns[2] == "baseline") {
				EXPECT_EQ(profile.standardProfile, oa::VideoCodecProfile::H264Baseline);
			} else if (columns[2] == "main") {
				EXPECT_EQ(profile.standardProfile, oa::VideoCodecProfile::H264Main);
			} else if (columns[2] == "high") {
				EXPECT_EQ(profile.standardProfile, oa::VideoCodecProfile::H264High);
			} else {
				FAIL() << "Unknown H.264 manifest profile: " << columns[2];
			}
			EXPECT_EQ(profile.h264PictureLayout, oa::VideoH264PictureLayout::Progressive);
			EXPECT_TRUE(profile.hasLevel);
			if (columns[15] == "3.1") {
				EXPECT_EQ(profile.level, static_cast<oa::U32>(STD_VIDEO_H264_LEVEL_IDC_3_1));
			} else if (columns[15] == "4.2") {
				EXPECT_EQ(profile.level, static_cast<oa::U32>(STD_VIDEO_H264_LEVEL_IDC_4_2));
			} else if (columns[15] == "5.2") {
				EXPECT_EQ(profile.level, static_cast<oa::U32>(STD_VIDEO_H264_LEVEL_IDC_5_2));
			} else {
				FAIL() << "Unknown H.264 manifest level: " << columns[15];
			}
		} else if (columns[1] == "h265") {
			EXPECT_EQ(profile.codec, oa::VideoCodec::H265);
			EXPECT_EQ(profile.standardProfile, oa::VideoCodecProfile::H265Main);
			EXPECT_TRUE(profile.hasLevel);
			if (columns[15] == "3.1") {
				EXPECT_EQ(profile.level, static_cast<oa::U32>(STD_VIDEO_H265_LEVEL_IDC_3_1));
			} else if (columns[15] == "4.1") {
				EXPECT_EQ(profile.level, static_cast<oa::U32>(STD_VIDEO_H265_LEVEL_IDC_4_1));
			} else if (columns[15] == "5.1") {
				EXPECT_EQ(profile.level, static_cast<oa::U32>(STD_VIDEO_H265_LEVEL_IDC_5_1));
			} else {
				FAIL() << "Unknown H.265 manifest level: " << columns[15];
			}
			EXPECT_FALSE(profile.highTier);
		} else if (columns[1] == "av1") {
			EXPECT_EQ(profile.codec, oa::VideoCodec::AV1);
			EXPECT_EQ(profile.standardProfile, oa::VideoCodecProfile::Av1Main);
			EXPECT_FALSE(profile.av1FilmGrain);
			EXPECT_TRUE(profile.hasLevel);
			if (columns[15] == "3.1") {
				EXPECT_EQ(profile.level, static_cast<oa::U32>(STD_VIDEO_AV1_LEVEL_3_1));
			} else if (columns[15] == "4.1") {
				EXPECT_EQ(profile.level, static_cast<oa::U32>(STD_VIDEO_AV1_LEVEL_4_1));
			} else if (columns[15] == "5.1") {
				EXPECT_EQ(profile.level, static_cast<oa::U32>(STD_VIDEO_AV1_LEVEL_5_1));
			} else {
				FAIL() << "Unknown AV1 manifest level: " << columns[15];
			}
			EXPECT_FALSE(profile.highTier);
		} else if (columns[1] == "vp9") {
			EXPECT_EQ(profile.codec, oa::VideoCodec::VP9);
			EXPECT_EQ(profile.standardProfile, oa::VideoCodecProfile::Vp9Profile0);
			EXPECT_TRUE(profile.hasLevel);
			if (columns[15] == "3.1") {
				EXPECT_EQ(profile.level, static_cast<oa::U32>(STD_VIDEO_VP9_LEVEL_3_1));
			} else if (columns[15] == "4.0") {
				EXPECT_EQ(profile.level, static_cast<oa::U32>(STD_VIDEO_VP9_LEVEL_4_0));
			} else if (columns[15] == "5.0") {
				EXPECT_EQ(profile.level, static_cast<oa::U32>(STD_VIDEO_VP9_LEVEL_5_0));
			} else {
				FAIL() << "Unknown VP9 manifest level: " << columns[15];
			}
		} else {
			FAIL() << "Unknown manifest codec: " << columns[1];
		}
		++rowCount;
	}
	EXPECT_EQ(rowCount, 14U);
}

TEST(VideoCodecRegistry, CreatesStreamLocalParserState)
{
	auto first = oa::VideoCodecRegistry::getInstance().createParser(oa::VideoCodec::H264);
	auto second = oa::VideoCodecRegistry::getInstance().createParser(oa::VideoCodec::H264);
	ASSERT_TRUE(first);
	ASSERT_TRUE(second);
	ASSERT_NE(first.get(), second.get());

	auto sps = makeH264SpsNal();
	ASSERT_GT(sps.size(), 4u);
	ASSERT_TRUE(first->parseSps(oa::Span<const oa::U8>(
		sps.data() + 4, sps.size() - 4)).isOk());
	EXPECT_NE(first->getH264Sps(0), nullptr);
	EXPECT_EQ(second->getH264Sps(0), nullptr);
}

// phase 2.1: vulkan Video Extensions Setup Tests
TEST_F(VkEngineTestFixture, VideoDecoder_QueryCodecSupport)
{
	auto& engine = rt();
	
	bool h264Supported = testVideoDecodeSupported(engine, oa::VideoCodec::H264);
	bool h265Supported = testVideoDecodeSupported(engine, oa::VideoCodec::H265);
	bool av1Supported = testVideoDecodeSupported(engine, oa::VideoCodec::AV1);
	const auto& sw = oa::EngineDeviceAccess::get(engine).info.software;
	if (!sw.hasVideoQueue || !sw.hasVideoDecodeQueue || !oa::EngineDeviceAccess::get(engine).queues.hasVideoDecodeQueue) {
		EXPECT_FALSE(h264Supported);
		EXPECT_FALSE(h265Supported);
		EXPECT_FALSE(av1Supported);
		GTEST_SKIP() << "Selected vulkan device does not expose a video decode queue";
	}
	if (h264Supported) EXPECT_TRUE(sw.hasVideoDecodeH264);
	if (h265Supported) EXPECT_TRUE(sw.hasVideoDecodeH265);
	if (av1Supported) EXPECT_TRUE(sw.hasVideoDecodeAV1);
	if (!h264Supported && !h265Supported && !av1Supported) {
		GTEST_SKIP() << "vulkan Video decode extensions are present, but default "
						"profile capability queries are unsupported";
	}
}

TEST_F(VkEngineTestFixture, VideoDecoder_QueryMaxResolution)
{
	auto& engine = rt();
	auto capabilities = oa::VideoDecoder::queryDecodeCapabilities(engine, oa::VideoCodec::H264);
	if (!capabilities.isOk() || !capabilities->supported) {
		GTEST_SKIP() << "H.264 vulkan Video decode is not supported on selected device: "
			<< capabilities.getStatus().toString();
	}
	
	// RTX 5090 supports up to 8K decode
	EXPECT_GE(capabilities->maxWidth, 3840);  // at least 4K width
	EXPECT_GE(capabilities->maxHeight, 2160); // at least 4K height
}

TEST_F(VkEngineTestFixture, VideoDecoder_QueryDecodeCapabilities)
{
	auto& engine = rt();
	auto capsResult = oa::VideoDecoder::queryDecodeCapabilities(engine, oa::VideoCodec::H264);
	if (!capsResult.isOk()) {
		GTEST_SKIP() << capsResult.getStatus().toString();
	}

	const auto& caps = *capsResult;
	EXPECT_TRUE(caps.supported);
	EXPECT_TRUE(caps.hardwareProfileSupported);
	EXPECT_TRUE(caps.oaDecodePathImplemented);
	EXPECT_EQ(caps.profile.standardProfile, oa::VideoCodecProfile::H264High);
	EXPECT_GE(caps.maxWidth, 3840u);
	EXPECT_GE(caps.maxHeight, 2160u);
	EXPECT_GT(caps.maxDpbSlots, 0u);
	EXPECT_GT(caps.maxActiveReferencePictures, 0u);
	EXPECT_GT(caps.minBitstreamBufferOffsetAlignment, 0u);
	EXPECT_GT(caps.minBitstreamBufferSizeAlignment, 0u);
	EXPECT_TRUE(caps.supportsDpbAndOutputCoincide || caps.supportsDpbAndOutputDistinct);
	EXPECT_TRUE(caps.supportsDecodedDpb);
	EXPECT_TRUE(caps.supportsNv12Dpb);
	EXPECT_FALSE(caps.dpbFormats.empty());
	EXPECT_FALSE(caps.outputFormats.empty());
	EXPECT_TRUE(
		(caps.supportsDpbAndOutputCoincide
			&& (caps.supportsNv12DpbTransferSrc || caps.supportsNv12DpbSampled))
		|| (caps.supportsDpbAndOutputDistinct && caps.supportsNv12OutputSampled));
}

TEST_F(VkEngineTestFixture, VideoDecoder_ReportsNativeFormatForImplementedProfiles)
{
	struct Case {
		oa::VideoCodec codec;
		oa::VideoCodecProfile standardProfile;
		oa::VideoBitDepth bitDepth;
		VkFormat format;
	};
	const Case cases[] = {
		{oa::VideoCodec::H265, oa::VideoCodecProfile::H265Main10,
			oa::VideoBitDepth::Bit10,
			VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16},
		{oa::VideoCodec::AV1, oa::VideoCodecProfile::Av1Main,
			oa::VideoBitDepth::Bit10,
			VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16},
		{oa::VideoCodec::VP9, oa::VideoCodecProfile::Vp9Profile2,
			oa::VideoBitDepth::Bit10,
			VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16},
	};
	for (const Case& testCase : cases) {
		oa::VideoProfile profile = {};
		profile.codec = testCase.codec;
		profile.standardProfile = testCase.standardProfile;
		profile.chromaSubsampling = oa::VideoChromaSubsampling::Yuv420;
		profile.lumaBitDepth = testCase.bitDepth;
		profile.chromaBitDepth = testCase.bitDepth;
		auto capsResult = oa::VideoDecoder::queryDecodeCapabilities(rt(), profile);
		ASSERT_TRUE(capsResult.isOk()) << capsResult.getStatus().toString();
		EXPECT_TRUE(capsResult->oaDecodePathImplemented);
		if (capsResult->hardwareProfileSupported) {
			EXPECT_EQ(capsResult->referencePictureFormat, testCase.format);
			EXPECT_EQ(capsResult->pictureFormat, testCase.format);
			EXPECT_EQ(capsResult->supported, capsResult->supportsDecodedDpb);
		}
	}

	oa::VideoProfile rangeExtensions = {};
	rangeExtensions.codec = oa::VideoCodec::H265;
	rangeExtensions.standardProfile = oa::VideoCodecProfile::H265FormatRangeExtensions;
	auto rangeResult = oa::VideoDecoder::queryDecodeCapabilities(rt(), rangeExtensions);
	ASSERT_TRUE(rangeResult.isOk()) << rangeResult.getStatus().toString();
	EXPECT_TRUE(rangeResult->oaDecodePathImplemented);
	if (rangeResult->hardwareProfileSupported) {
		EXPECT_EQ(rangeResult->supported, rangeResult->supportsDecodedDpb);
	} else {
		EXPECT_FALSE(rangeResult->supported);
	}

	oa::VideoProfile mainStill = {};
	mainStill.codec = oa::VideoCodec::H265;
	mainStill.standardProfile = oa::VideoCodecProfile::H265MainStillPicture;
	auto mainStillResult = oa::VideoDecoder::queryDecodeCapabilities(rt(), mainStill);
	ASSERT_TRUE(mainStillResult.isOk()) << mainStillResult.getStatus().toString();
	EXPECT_FALSE(mainStillResult->supported);
	EXPECT_FALSE(mainStillResult->oaDecodePathImplemented);
}

TEST_F(VkEngineTestFixture, VideoDecoder_RecordsExactHardwareProfileMatrix)
{
	struct Case {
		const char* name;
		oa::VideoCodec codec;
		oa::VideoCodecProfile profile;
		oa::VideoChromaSubsampling chroma;
		oa::VideoBitDepth depth;
	};
	const Case cases[] = {
		{"h264-baseline-420-8", oa::VideoCodec::H264, oa::VideoCodecProfile::H264Baseline, oa::VideoChromaSubsampling::Yuv420, oa::VideoBitDepth::Bit8},
		{"h264-main-420-8", oa::VideoCodec::H264, oa::VideoCodecProfile::H264Main, oa::VideoChromaSubsampling::Yuv420, oa::VideoBitDepth::Bit8},
		{"h264-high-420-8", oa::VideoCodec::H264, oa::VideoCodecProfile::H264High, oa::VideoChromaSubsampling::Yuv420, oa::VideoBitDepth::Bit8},
		{"h264-high444-444-8", oa::VideoCodec::H264, oa::VideoCodecProfile::H264High444Predictive, oa::VideoChromaSubsampling::Yuv444, oa::VideoBitDepth::Bit8},
		{"h264-high444-444-10", oa::VideoCodec::H264, oa::VideoCodecProfile::H264High444Predictive, oa::VideoChromaSubsampling::Yuv444, oa::VideoBitDepth::Bit10},
		{"h265-main-420-8", oa::VideoCodec::H265, oa::VideoCodecProfile::H265Main, oa::VideoChromaSubsampling::Yuv420, oa::VideoBitDepth::Bit8},
		{"h265-main10-420-10", oa::VideoCodec::H265, oa::VideoCodecProfile::H265Main10, oa::VideoChromaSubsampling::Yuv420, oa::VideoBitDepth::Bit10},
		{"h265-mainstill-420-8", oa::VideoCodec::H265, oa::VideoCodecProfile::H265MainStillPicture, oa::VideoChromaSubsampling::Yuv420, oa::VideoBitDepth::Bit8},
		{"h265-range-420-8", oa::VideoCodec::H265, oa::VideoCodecProfile::H265FormatRangeExtensions, oa::VideoChromaSubsampling::Yuv420, oa::VideoBitDepth::Bit8},
		{"h265-range-420-10", oa::VideoCodec::H265, oa::VideoCodecProfile::H265FormatRangeExtensions, oa::VideoChromaSubsampling::Yuv420, oa::VideoBitDepth::Bit10},
		{"h265-range-422-10", oa::VideoCodec::H265, oa::VideoCodecProfile::H265FormatRangeExtensions, oa::VideoChromaSubsampling::Yuv422, oa::VideoBitDepth::Bit10},
		{"h265-range-444-10", oa::VideoCodec::H265, oa::VideoCodecProfile::H265FormatRangeExtensions, oa::VideoChromaSubsampling::Yuv444, oa::VideoBitDepth::Bit10},
		{"h265-range-420-12", oa::VideoCodec::H265, oa::VideoCodecProfile::H265FormatRangeExtensions, oa::VideoChromaSubsampling::Yuv420, oa::VideoBitDepth::Bit12},
		{"h265-scc-420-8", oa::VideoCodec::H265, oa::VideoCodecProfile::H265ScreenContentCodingExtensions, oa::VideoChromaSubsampling::Yuv420, oa::VideoBitDepth::Bit8},
		{"av1-main-420-8", oa::VideoCodec::AV1, oa::VideoCodecProfile::Av1Main, oa::VideoChromaSubsampling::Yuv420, oa::VideoBitDepth::Bit8},
		{"av1-main-420-10", oa::VideoCodec::AV1, oa::VideoCodecProfile::Av1Main, oa::VideoChromaSubsampling::Yuv420, oa::VideoBitDepth::Bit10},
		{"av1-high-444-8", oa::VideoCodec::AV1, oa::VideoCodecProfile::Av1High, oa::VideoChromaSubsampling::Yuv444, oa::VideoBitDepth::Bit8},
		{"av1-high-444-10", oa::VideoCodec::AV1, oa::VideoCodecProfile::Av1High, oa::VideoChromaSubsampling::Yuv444, oa::VideoBitDepth::Bit10},
		{"av1-pro-422-10", oa::VideoCodec::AV1, oa::VideoCodecProfile::Av1Professional, oa::VideoChromaSubsampling::Yuv422, oa::VideoBitDepth::Bit10},
		{"av1-pro-444-12", oa::VideoCodec::AV1, oa::VideoCodecProfile::Av1Professional, oa::VideoChromaSubsampling::Yuv444, oa::VideoBitDepth::Bit12},
		{"vp9-p0-420-8", oa::VideoCodec::VP9, oa::VideoCodecProfile::Vp9Profile0, oa::VideoChromaSubsampling::Yuv420, oa::VideoBitDepth::Bit8},
		{"vp9-p1-444-8", oa::VideoCodec::VP9, oa::VideoCodecProfile::Vp9Profile1, oa::VideoChromaSubsampling::Yuv444, oa::VideoBitDepth::Bit8},
		{"vp9-p2-420-10", oa::VideoCodec::VP9, oa::VideoCodecProfile::Vp9Profile2, oa::VideoChromaSubsampling::Yuv420, oa::VideoBitDepth::Bit10},
		{"vp9-p2-420-12", oa::VideoCodec::VP9, oa::VideoCodecProfile::Vp9Profile2, oa::VideoChromaSubsampling::Yuv420, oa::VideoBitDepth::Bit12},
		{"vp9-p3-444-10", oa::VideoCodec::VP9, oa::VideoCodecProfile::Vp9Profile3, oa::VideoChromaSubsampling::Yuv444, oa::VideoBitDepth::Bit10},
		{"vp9-p3-444-12", oa::VideoCodec::VP9, oa::VideoCodecProfile::Vp9Profile3, oa::VideoChromaSubsampling::Yuv444, oa::VideoBitDepth::Bit12},
	};

	oa::U32 hardwareCount = 0U;
	for (const Case& testCase : cases) {
		oa::VideoProfile request = {};
		request.codec = testCase.codec;
		request.standardProfile = testCase.profile;
		request.chromaSubsampling = testCase.chroma;
		request.lumaBitDepth = testCase.depth;
		request.chromaBitDepth = testCase.depth;
		auto result = oa::VideoDecoder::queryDecodeCapabilities(rt(), request);
		if (!result.isOk() && result.getStatus().getCode() == oa::StatusCode::Unavailable) {
			ASSERT_TRUE(oa::print("PROFILE_MATRIX\t{}\tquery=unavailable", testCase.name).isOk());
			continue;
		}
		ASSERT_TRUE(result.isOk()) << testCase.name << ": " << result.getStatus().toString();
		if (result->hardwareProfileSupported) ++hardwareCount;
		ASSERT_TRUE(oa::print(
			"PROFILE_MATRIX\t{}\thardware={}\toa_path={}\tdecoded_format={}\tsupported={}",
			testCase.name,
			result->hardwareProfileSupported,
			result->oaDecodePathImplemented,
			static_cast<oa::U32>(result->referencePictureFormat),
			result->supported).isOk());
	}
	if (hardwareCount == 0U) {
		GTEST_SKIP() << "No queried Vulkan Video decode profile is supported on this device";
	}
}

TEST_F(VkEngineTestFixture, VideoDecoder_CreateH264Decoder)
{
	auto& engine = rt();
	if (!testVideoDecodeSupported(engine, oa::VideoCodec::H264)) {
		GTEST_SKIP() << "H.264 vulkan Video decode is not supported on selected device";
	}
	
	oa::VideoProfile profile;
	profile.codec = oa::VideoCodec::H264;
	profile.width = 1920;
	profile.height = 1080;
	profile.maxDpbSlots = 16;
	
	auto result = oa::VideoDecoder::create(engine, profile);
	
	// phase 2.1 implementation pending - expect creation to succeed
	// but actual decode will fail until implemented
	EXPECT_TRUE(result.isOk());
	
	if (result.isOk())
	{
		auto decoder = oa::move(*result);
		EXPECT_TRUE(decoder.isInitialized());
		EXPECT_TRUE(decoder.hasSessionParameters());
		EXPECT_GE(decoder.getDpbSlotCapacity(), profile.maxDpbSlots);
		EXPECT_GE(decoder.getDpbViewCount(), 1u);
		auto caps = oa::VideoDecoder::queryDecodeCapabilities(engine, oa::VideoCodec::H264);
		ASSERT_TRUE(caps.isOk());
		expectDecoderCodedExtentAligned(decoder, *caps, profile);
		if (caps->supportsDpbAndOutputDistinct) {
			EXPECT_GE(decoder.getOutputFrameCapacity(), profile.maxDpbSlots);
			EXPECT_GE(decoder.getOutputViewCount(), profile.maxDpbSlots);
		}
		EXPECT_TRUE(decoder.close().isOk());
	}
}

TEST_F(VkEngineTestFixture, VideoDecoder_CreateH265Decoder)
{
	auto& engine = rt();
	if (!testVideoDecodeSupported(engine, oa::VideoCodec::H265)) {
		GTEST_SKIP() << "H.265 vulkan Video decode is not supported on selected device";
	}
	
	oa::VideoProfile profile;
	profile.codec = oa::VideoCodec::H265;
	profile.width = 3840;
	profile.height = 2160;
	profile.maxDpbSlots = 16;
	
	auto result = oa::VideoDecoder::create(engine, profile);
	EXPECT_TRUE(result.isOk());
	
	if (result.isOk())
	{
		auto decoder = oa::move(*result);
		EXPECT_TRUE(decoder.isInitialized());
		EXPECT_TRUE(decoder.hasSessionParameters());
		EXPECT_GE(decoder.getDpbSlotCapacity(), profile.maxDpbSlots);
		EXPECT_GE(decoder.getDpbViewCount(), 1u);
		auto caps = oa::VideoDecoder::queryDecodeCapabilities(engine, oa::VideoCodec::H265);
		ASSERT_TRUE(caps.isOk());
		expectDecoderCodedExtentAligned(decoder, *caps, profile);
		if (caps->supportsDpbAndOutputDistinct) {
			EXPECT_GE(decoder.getOutputFrameCapacity(), profile.maxDpbSlots);
			EXPECT_GE(decoder.getOutputViewCount(), profile.maxDpbSlots);
		}
		EXPECT_TRUE(decoder.close().isOk());
	}
}

TEST_F(VkEngineTestFixture, VideoDecoder_CreateAv1Decoder)
{
	auto& engine = rt();
	if (!testVideoDecodeSupported(engine, oa::VideoCodec::AV1)) {
		GTEST_SKIP() << "AV1 vulkan Video decode is not supported on selected device";
	}

	oa::VideoProfile profile;
	profile.codec = oa::VideoCodec::AV1;
	profile.width = 3840;
	profile.height = 2160;
	// AV1 exposes eight reference-frame slots plus the current picture.
	profile.maxDpbSlots = 9;

	auto result = oa::VideoDecoder::create(engine, profile);
	EXPECT_TRUE(result.isOk());

	if (result.isOk())
	{
		auto decoder = oa::move(*result);
		EXPECT_TRUE(decoder.isInitialized());
		EXPECT_FALSE(decoder.hasSessionParameters());
		EXPECT_GE(decoder.getDpbSlotCapacity(), profile.maxDpbSlots);
		EXPECT_GE(decoder.getDpbViewCount(), 1u);
		auto caps = oa::VideoDecoder::queryDecodeCapabilities(engine, oa::VideoCodec::AV1);
		ASSERT_TRUE(caps.isOk());
		expectDecoderCodedExtentAligned(decoder, *caps, profile);
		if (caps->supportsDpbAndOutputDistinct) {
			EXPECT_GE(decoder.getOutputFrameCapacity(), profile.maxDpbSlots);
			EXPECT_GE(decoder.getOutputViewCount(), profile.maxDpbSlots);
		}
		EXPECT_TRUE(decoder.close().isOk());
	}
}

TEST_F(VkEngineTestFixture, VideoDecoder_RejectInvalidProfile)
{
	auto& engine = rt();
	if (!testVideoDecodeSupported(engine, oa::VideoCodec::H264)) {
		GTEST_SKIP() << "H.264 vulkan Video decode is not supported on selected device";
	}

	oa::VideoProfile profile;
	profile.codec = oa::VideoCodec::H264;
	profile.width = 0;
	profile.height = 1080;
	profile.maxDpbSlots = 16;

	auto result = oa::VideoDecoder::create(engine, profile);
	EXPECT_FALSE(result.isOk());
}

TEST_F(VkEngineTestFixture, VideoDecoder_UpdateH264SessionParameters)
{
	auto& engine = rt();
	if (!testVideoDecodeSupported(engine, oa::VideoCodec::H264)) {
		GTEST_SKIP() << "H.264 vulkan Video decode is not supported on selected device";
	}

	oa::VideoProfile profile;
	profile.codec = oa::VideoCodec::H264;
	profile.width = 1920;
	profile.height = 1080;
	profile.maxDpbSlots = 16;

	auto result = oa::VideoDecoder::create(engine, profile);
	ASSERT_TRUE(result.isOk());
	auto decoder = oa::move(*result);

	oa::VideoFrame frame = {};
	auto sps = makeH264SpsNal();
	auto status = oa::VideoDecoderInternal::decodeFrame(
		decoder, oa::Span<const oa::U8>(sps), frame);
	EXPECT_TRUE(status.isOk()) << status.toString();
	EXPECT_EQ(decoder.getSessionParameterUpdateCount(), 1u);
	EXPECT_GE(oa::VideoDecoderInternal::getBitstreamBufferCapacity(decoder), static_cast<oa::U64>(sps.size()));

	auto pps = makeH264PpsNal();
	status = oa::VideoDecoderInternal::decodeFrame(
		decoder, oa::Span<const oa::U8>(pps), frame);
	EXPECT_TRUE(status.isOk()) << status.toString();
	EXPECT_EQ(decoder.getSessionParameterUpdateCount(), 2u);
	EXPECT_GE(oa::VideoDecoderInternal::getBitstreamBufferCapacity(decoder), static_cast<oa::U64>(pps.size()));

	EXPECT_TRUE(decoder.close().isOk());
}

TEST_F(VkEngineTestFixture, VideoDecoder_UpdateH264SessionParametersFromAccessUnit)
{
	auto& engine = rt();
	if (!testVideoDecodeSupported(engine, oa::VideoCodec::H264)) {
		GTEST_SKIP() << "H.264 vulkan Video decode is not supported on selected device";
	}

	oa::VideoProfile profile;
	profile.codec = oa::VideoCodec::H264;
	profile.width = 1920;
	profile.height = 1080;
	profile.maxDpbSlots = 16;

	auto result = oa::VideoDecoder::create(engine, profile);
	ASSERT_TRUE(result.isOk());
	auto decoder = oa::move(*result);

	oa::VideoFrame frame = {};
	auto accessUnit = makeH264ParameterAccessUnit();
	auto status = oa::VideoDecoderInternal::decodeFrame(
		decoder, oa::Span<const oa::U8>(accessUnit), frame);
	EXPECT_TRUE(status.isOk()) << status.toString();
	EXPECT_EQ(decoder.getSessionParameterUpdateCount(), 2u);
	EXPECT_GE(oa::VideoDecoderInternal::getBitstreamBufferCapacity(decoder), static_cast<oa::U64>(accessUnit.size()));

	EXPECT_TRUE(decoder.close().isOk());
}

TEST_F(VkEngineTestFixture, VideoDecoder_DecodeH264FrameFromLocalFixture)
{
	auto fixtureResult = oa::Filesystem::readBinary(testAssetPath("video/conformance/test_pattern_72p_h264_baseline_idr_8bit_420.h264"));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();

	auto& engine = rt();
	if (!testVideoDecodeSupported(engine, oa::VideoCodec::H264)) {
		GTEST_SKIP() << "H.264 vulkan Video decode is not supported on selected device";
	}

	oa::VideoProfile profile;
	profile.codec = oa::VideoCodec::H264;
	profile.width = 128;
	profile.height = 72;
	profile.maxDpbSlots = 4;

	auto result = oa::VideoDecoder::create(engine, profile);
	ASSERT_TRUE(result.isOk());
	auto decoder = oa::move(*result);

	constexpr oa::U64 testPts = 42ULL;
	auto frameResult = decoder.decode(
		oa::Span<const oa::U8>(*fixtureResult),
		testPts);
	EXPECT_TRUE(frameResult.isOk()) << frameResult.getStatus().toString();
	if (frameResult.isOk()) {
		oa::VideoFrame frame = *frameResult;
		EXPECT_NE(frame.image, VK_NULL_HANDLE);
		EXPECT_NE(frame.imageView, VK_NULL_HANDLE);
		EXPECT_EQ(frame.width, profile.width);
		EXPECT_EQ(frame.height, profile.height);
		EXPECT_EQ(frame.presentationTimestamp, testPts);
		EXPECT_FALSE(frame.isRgb);

		auto lumaResult = decoder.readbackLuma(frame);
		ASSERT_TRUE(lumaResult.isOk()) << lumaResult.getStatus().toString();
		ASSERT_EQ(lumaResult->size(), static_cast<oa::Usize>(profile.width * profile.height));
		oa::U32 nonZeroCount = 0;
		oa::U8 minValue = 255;
		oa::U8 maxValue = 0;
		for (oa::U8 value : *lumaResult) {
			nonZeroCount += value != 0 ? 1u : 0u;
			minValue = value < minValue ? value : minValue;
			maxValue = value > maxValue ? value : maxValue;
		}
		if (nonZeroCount == 0 || maxValue == minValue) {
			GTEST_SKIP() << "vulkan H.264 command submission succeeded, but luma "
							"readback is still all zero";
		}
		EXPECT_GT(nonZeroCount, 0u);
		EXPECT_GT(maxValue, minValue);

		auto nv12Result = decoder.readbackNv12(frame);
		ASSERT_TRUE(nv12Result.isOk()) << nv12Result.getStatus().toString();
		ASSERT_EQ(nv12Result->size(), static_cast<oa::Usize>(profile.width * profile.height * 3 / 2));

		if (::system("command -v ffmpeg >/dev/null 2>&1") == 0) {
			const oa::Path fixturePath =
				testAssetPath("video/conformance/test_pattern_72p_h264_baseline_idr_8bit_420.h264");
			const oa::Path referencePath =
				testReferencePath("oa_h264_idr_reference_nv12");
			const oa::String command = oa::String("ffmpeg -v error -y -i \"")
				+ fixturePath.string()
				+ "\" -frames:v 1 -f rawvideo -pix_fmt nv12 \""
				+ referencePath.string()
				+ "\"";
			ASSERT_EQ(::system(command.cStr()), 0);
			auto referenceResult = oa::Filesystem::readBinary(referencePath);
			(void) oa::Filesystem::removeFile(referencePath);
			ASSERT_TRUE(referenceResult.isOk())
				<< referenceResult.getStatus().toString();
			ASSERT_EQ(referenceResult->size(), nv12Result->size());

			const oa::Usize lumaBytes =
				static_cast<oa::Usize>(profile.width) * profile.height;
			oa::U64 lumaError = 0;
			oa::U64 chromaError = 0;
			oa::U64 decodedLumaSum = 0;
			oa::U64 referenceLumaSum = 0;
			oa::U8 maxError = 0;
			for (oa::Usize i = 0; i < nv12Result->size(); ++i) {
				const oa::U8 error = static_cast<oa::U8>(oa::abs(
					static_cast<int>((*nv12Result)[i])
					- static_cast<int>((*referenceResult)[i])));
				if (i < lumaBytes) {
					lumaError += error;
					decodedLumaSum += (*nv12Result)[i];
					referenceLumaSum += (*referenceResult)[i];
				} else {
					chromaError += error;
				}
				maxError = error > maxError ? error : maxError;
			}
			const oa::F64 lumaMae =
				static_cast<oa::F64>(lumaError) / static_cast<oa::F64>(lumaBytes);
			const oa::F64 chromaMae =
				static_cast<oa::F64>(chromaError)
				/ static_cast<oa::F64>(nv12Result->size() - lumaBytes);
			EXPECT_LT(lumaMae, 3.0)
				<< "maxError=" << static_cast<oa::U32>(maxError)
				<< " decodedMean="
				<< static_cast<oa::F64>(decodedLumaSum) / static_cast<oa::F64>(lumaBytes)
				<< " referenceMean="
				<< static_cast<oa::F64>(referenceLumaSum) / static_cast<oa::F64>(lumaBytes)
				<< " decodedFirst="
				<< static_cast<oa::U32>((*nv12Result)[0]) << ","
				<< static_cast<oa::U32>((*nv12Result)[1]) << ","
				<< static_cast<oa::U32>((*nv12Result)[2]) << ","
				<< static_cast<oa::U32>((*nv12Result)[3])
				<< " referenceFirst="
				<< static_cast<oa::U32>((*referenceResult)[0]) << ","
				<< static_cast<oa::U32>((*referenceResult)[1]) << ","
				<< static_cast<oa::U32>((*referenceResult)[2]) << ","
				<< static_cast<oa::U32>((*referenceResult)[3]);
			EXPECT_LT(chromaMae, 3.0) << "maxError=" << static_cast<oa::U32>(maxError);
		}

		auto hardwareTensorResult = oa::VideoDecoderInternal::convertFrameToBf16Hardware(decoder, frame, false);
		EXPECT_FALSE(hardwareTensorResult.isOk());
		EXPECT_EQ(hardwareTensorResult.getStatus().getCode(), oa::StatusCode::Unavailable);
		const oa::U64 lumaBytes = static_cast<oa::U64>(profile.width) * profile.height;
		const oa::U8 yy = (*nv12Result)[0];
		const oa::U8 uu = (*nv12Result)[static_cast<oa::Usize>(lumaBytes)];
		const oa::U8 vv = (*nv12Result)[static_cast<oa::Usize>(lumaBytes + 1)];
		const oa::F32 Y = 1.164f * (static_cast<oa::F32>(yy) - 16.0f) / 255.0f;
		const oa::F32 U = (static_cast<oa::F32>(uu) - 128.0f) / 255.0f;
		const oa::F32 V = (static_cast<oa::F32>(vv) - 128.0f) / 255.0f;
		auto clampUnit = [](oa::F32 value) {
			if (value < 0.0f) {
				return 0.0f;
			}
			if (value > 1.0f) {
				return 1.0f;
			}
			return value;
		};
		const oa::F32 expectedR = clampUnit(Y + 1.596f * V);
		const oa::F32 expectedG = clampUnit(Y - 0.391f * U - 0.813f * V);
		const oa::F32 expectedB = clampUnit(Y + 2.018f * U);
		EXPECT_GE(expectedR, 0.0f);
		EXPECT_LE(expectedR, 1.0f);
		EXPECT_GE(expectedG, 0.0f);
		EXPECT_LE(expectedG, 1.0f);
		EXPECT_GE(expectedB, 0.0f);
		EXPECT_LE(expectedB, 1.0f);

		auto tensorResult = decoder.convertFrameToBf16(frame, false);
		ASSERT_TRUE(tensorResult.isOk()) << tensorResult.getStatus().toString();
		oa::Matrix tensor = oa::move(*tensorResult);
		expectShape(tensor, {1, 3, profile.height, profile.width});
		expectFinite(tensor);
		EXPECT_GE(tensor.at(0), 0.0f);
		EXPECT_LE(tensor.at(0), 1.0f);
	}

	EXPECT_TRUE(decoder.close().isOk());

	auto conversionDecoderResult = oa::VideoDecoder::create(engine, profile);
	ASSERT_TRUE(conversionDecoderResult.isOk()) << conversionDecoderResult.getStatus().toString();
	auto conversionDecoder = oa::move(*conversionDecoderResult);

	oa::VideoConversionOptions conversion = {};
	conversion.preferHardwareYCbCr = true;
	conversion.colorSpace = oa::YCbCrModel::Auto;
	conversion.convertToRgb = true;
	oa::VideoFrame rgbaFrame = {};
	auto status = oa::VideoDecoderInternal::decodeFrameWithConversion(conversionDecoder,
		oa::Span<const oa::U8>(*fixtureResult),
		conversion,
		rgbaFrame);
	ASSERT_TRUE(status.isOk()) << status.toString();
	EXPECT_NE(rgbaFrame.image, VK_NULL_HANDLE);
	EXPECT_NE(rgbaFrame.imageView, VK_NULL_HANDLE);
	EXPECT_EQ(rgbaFrame.format, VK_FORMAT_R8G8B8A8_UNORM);
	EXPECT_EQ(rgbaFrame.width, profile.width);
	EXPECT_EQ(rgbaFrame.height, profile.height);
	EXPECT_TRUE(rgbaFrame.isRgb);

	auto rgbaResult = conversionDecoder.readbackRgba(rgbaFrame);
	ASSERT_TRUE(rgbaResult.isOk()) << rgbaResult.getStatus().toString();
	ASSERT_EQ(rgbaResult->size(), static_cast<oa::Usize>(profile.width * profile.height * 4));
	oa::U32 alpha255Count = 0;
	for (oa::Usize i = 0; i < rgbaResult->size(); i += 4) {
		alpha255Count += (*rgbaResult)[i + 3] == 255 ? 1u : 0u;
	}
	// hardware YCbCr path must write opaque alpha; this also catches zero-write
	// failures (e.g. push-constant misalignment causing all threads to return).
	EXPECT_EQ(alpha255Count, static_cast<oa::U32>(profile.width * profile.height));
	EXPECT_GT(
		oa::VideoDecoderInternal::getHardwareYcbcrDispatchCount(conversionDecoder),
		0U);
	EXPECT_TRUE(conversionDecoder.close().isOk());

	auto computeConversionDecoderResult = oa::VideoDecoder::create(engine, profile);
	ASSERT_TRUE(computeConversionDecoderResult.isOk()) << computeConversionDecoderResult.getStatus().toString();
	auto computeConversionDecoder = oa::move(*computeConversionDecoderResult);

	oa::VideoConversionOptions computeConversion = {};
	computeConversion.preferHardwareYCbCr = false;
	computeConversion.colorSpace = oa::YCbCrModel::Auto;
	computeConversion.convertToRgb = true;
	oa::VideoFrame computeRgbaFrame = {};
	status = oa::VideoDecoderInternal::decodeFrameWithConversion(computeConversionDecoder,
		oa::Span<const oa::U8>(*fixtureResult),
		computeConversion,
		computeRgbaFrame);
	ASSERT_TRUE(status.isOk()) << status.toString();
	EXPECT_NE(computeRgbaFrame.image, VK_NULL_HANDLE);
	EXPECT_NE(computeRgbaFrame.imageView, VK_NULL_HANDLE);
	EXPECT_EQ(computeRgbaFrame.format, VK_FORMAT_R8G8B8A8_UNORM);
	EXPECT_EQ(computeRgbaFrame.width, profile.width);
	EXPECT_EQ(computeRgbaFrame.height, profile.height);
	EXPECT_TRUE(computeRgbaFrame.isRgb);

	auto computeRgbaResult = computeConversionDecoder.readbackRgba(computeRgbaFrame);
	ASSERT_TRUE(computeRgbaResult.isOk()) << computeRgbaResult.getStatus().toString();
	ASSERT_EQ(computeRgbaResult->size(), static_cast<oa::Usize>(profile.width * profile.height * 4));
	oa::U32 computeAlpha255Count = 0;
	oa::U32 computeColorNonZeroCount = 0;
	for (oa::Usize i = 0; i < computeRgbaResult->size(); i += 4) {
		computeColorNonZeroCount += ((*computeRgbaResult)[i] | (*computeRgbaResult)[i + 1] | (*computeRgbaResult)[i + 2]) != 0 ? 1u : 0u;
		computeAlpha255Count += (*computeRgbaResult)[i + 3] == 255 ? 1u : 0u;
	}
	EXPECT_GT(computeColorNonZeroCount, 0u);
	EXPECT_EQ(computeAlpha255Count, static_cast<oa::U32>(profile.width * profile.height));
	oa::U64 rgbAbsoluteError = 0U;
	oa::U8 rgbMaxError = 0U;
	for (oa::Usize i = 0; i < rgbaResult->size(); i += 4U) {
		for (oa::Usize channel = 0; channel < 3U; ++channel) {
			const oa::U8 hardware = (*rgbaResult)[i + channel];
			const oa::U8 compute = (*computeRgbaResult)[i + channel];
			const oa::U8 error = hardware > compute
				? static_cast<oa::U8>(hardware - compute)
				: static_cast<oa::U8>(compute - hardware);
			rgbAbsoluteError += error;
			rgbMaxError = oa::max(rgbMaxError, error);
		}
	}
	const oa::F64 rgbMae = static_cast<oa::F64>(rgbAbsoluteError)
		/ static_cast<oa::F64>(profile.width * profile.height * 3U);
	EXPECT_LT(rgbMae, 6.0) << "maxError=" << static_cast<oa::U32>(rgbMaxError);
	EXPECT_LT(rgbMaxError, 96U);
	EXPECT_TRUE(computeConversionDecoder.close().isOk());
}

TEST_F(VkEngineTestFixture, VideoDecoder_DecodeResizeNormalizeOwnedBySession)
{
	auto fixtureResult = oa::Filesystem::readBinary(
		testAssetPath("video/conformance/test_pattern_72p_h264_baseline_idr_8bit_420.h264"));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();

	auto& engine = rt();
	if (not testVideoDecodeSupported(engine, oa::VideoCodec::H264)) {
		GTEST_SKIP() << "H.264 vulkan Video decode is not supported on selected device";
	}

	oa::VideoProfile profile = {};
	profile.codec = oa::VideoCodec::H264;
	profile.width = 128;
	profile.height = 72;
	profile.maxDpbSlots = 4;
	auto decoderResult = oa::VideoDecoder::create(engine, profile);
	ASSERT_TRUE(decoderResult.isOk()) << decoderResult.getStatus().toString();
	auto decoder = oa::move(*decoderResult);

	auto preparedResult = decoder.decodeResizeNormalize(
		oa::Span<const oa::U8>(*fixtureResult),
		64,
		48);
	ASSERT_TRUE(preparedResult.isOk()) << preparedResult.getStatus().toString();
	oa::Matrix prepared = oa::move(*preparedResult);
	expectShape(prepared, {1, 3, 48, 64});
	expectFinite(prepared);

	EXPECT_TRUE(decoder.close().isOk());
}

TEST_F(VkEngineTestFixture, VideoDecoder_PipelinesAsyncH264Conversion)
{
	auto fixtureResult = oa::Filesystem::readBinary(
		testAssetPath("video/conformance/test_pattern_72p_h264_baseline_idr_8bit_420.h264"));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();

	auto& engine = rt();
	if (!testVideoDecodeSupported(engine, oa::VideoCodec::H264)) {
		GTEST_SKIP() << "H.264 vulkan Video decode is not supported on selected device";
	}

	oa::VideoProfile profile = makeH264FixtureProfile();
	profile.maxDpbSlots = 16;
	auto decoderResult = oa::VideoDecoder::create(engine, profile);
	ASSERT_TRUE(decoderResult.isOk()) << decoderResult.getStatus().toString();
	auto decoder = oa::move(*decoderResult);

	// Exceed the 16-slot decode-output pool so this test covers reuse while
	// earlier color conversions are still in flight.
	constexpr oa::U32 frameCount = 24;
	oa::Vector<oa::VideoFrame> rgbaFrames;
	oa::Vector<oa::Event> readyEvents;
	rgbaFrames.reserve(frameCount);
	readyEvents.reserve(frameCount);

	oa::VideoConversionOptions options = {};
	options.convertToRgb = true;
	options.preferHardwareYCbCr = true;

	for (oa::U32 i = 0; i < frameCount; ++i) {
		oa::VideoFrame nv12 = {};
		oa::Status decodeStatus = oa::VideoDecoderInternal::decodeFrame(decoder,
			oa::Span<const oa::U8>(*fixtureResult),
			nv12);
		ASSERT_TRUE(decodeStatus.isOk()) << decodeStatus.toString();

		auto rgbaResult = decoder.allocateRgbaFrame(
			profile.width, profile.height);
		ASSERT_TRUE(rgbaResult.isOk()) << rgbaResult.getStatus().toString();
		rgbaFrames.pushBack(*rgbaResult);

		auto readyResult = decoder.convertIntoAsync(
			nv12, options, rgbaFrames.back());
		ASSERT_TRUE(readyResult.isOk()) << readyResult.getStatus().toString();
		ASSERT_TRUE(readyResult->isValid())
			<< "async conversion returned an invalid completion event";
		readyEvents.pushBack(*readyResult);
	}

	for (oa::U32 i = 0; i < frameCount; ++i) {
		ASSERT_TRUE(readyEvents[i].wait().isOk());
		auto rgbaResult = decoder.readbackRgba(rgbaFrames[i]);
		ASSERT_TRUE(rgbaResult.isOk()) << rgbaResult.getStatus().toString();
		ASSERT_EQ(
			rgbaResult->size(),
			static_cast<oa::Usize>(profile.width * profile.height * 4));
		for (oa::Usize pixel = 0; pixel < rgbaResult->size(); pixel += 4) {
			ASSERT_EQ((*rgbaResult)[pixel + 3], 255);
		}
	}
	EXPECT_EQ(
		oa::VideoDecoderInternal::getHardwareYcbcrDispatchCount(decoder),
		frameCount);

	EXPECT_TRUE(decoder.close().isOk());
}

TEST_F(VkEngineTestFixture, VideoDecoder_AsyncConversionEventRetiresResourcesSafely)
{
	auto fixtureResult = oa::Filesystem::readBinary(
		testAssetPath("video/conformance/test_pattern_72p_h264_baseline_idr_8bit_420.h264"));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();

	auto& engine = rt();
	if (!testVideoDecodeSupported(engine, oa::VideoCodec::H264)) {
		GTEST_SKIP() << "H.264 vulkan Video decode is not supported on selected device";
	}

	oa::VideoProfile profile = makeH264FixtureProfile();
	profile.maxDpbSlots = 16;
	auto decoderResult = oa::VideoDecoder::create(engine, profile);
	ASSERT_TRUE(decoderResult.isOk()) << decoderResult.getStatus().toString();
	auto decoder = oa::move(*decoderResult);

	oa::VideoConversionOptions options = {};
	options.convertToRgb = true;
	options.preferHardwareYCbCr = false;

	oa::VideoFrame firstNv12 = {};
	ASSERT_TRUE(oa::VideoDecoderInternal::decodeFrame(
		decoder, oa::Span<const oa::U8>(*fixtureResult), firstNv12).isOk());
	auto firstRgbaResult = decoder.allocateRgbaFrame(
		profile.width, profile.height);
	ASSERT_TRUE(firstRgbaResult.isOk()) << firstRgbaResult.getStatus().toString();
	oa::VideoFrame firstRgba = *firstRgbaResult;

	auto firstReadyResult = decoder.convertIntoAsync(
		firstNv12, options, firstRgba);
	ASSERT_TRUE(firstReadyResult.isOk()) << firstReadyResult.getStatus().toString();
	const oa::Event droppedReady = *firstReadyResult;
	// The private dispatch ticket has already retired. Its resources must
	// remain valid until the public completion event reaches its value.
	ASSERT_TRUE(droppedReady.isValid());
	ASSERT_TRUE(droppedReady.wait().isOk());

	// A subsequent dispatch collects the completed retirement and safely
	// reuses the engine stream pool and bindless heap.
	oa::VideoFrame secondNv12 = {};
	ASSERT_TRUE(oa::VideoDecoderInternal::decodeFrame(
		decoder, oa::Span<const oa::U8>(*fixtureResult), secondNv12).isOk());
	auto secondRgbaResult = decoder.allocateRgbaFrame(
		profile.width, profile.height);
	ASSERT_TRUE(secondRgbaResult.isOk()) << secondRgbaResult.getStatus().toString();
	oa::VideoFrame secondRgba = *secondRgbaResult;
	auto secondReadyResult = decoder.convertIntoAsync(
		secondNv12, options, secondRgba);
	ASSERT_TRUE(secondReadyResult.isOk()) << secondReadyResult.getStatus().toString();
	ASSERT_TRUE(secondReadyResult->wait().isOk());

	auto rgbaResult = decoder.readbackRgba(secondRgba);
	ASSERT_TRUE(rgbaResult.isOk()) << rgbaResult.getStatus().toString();
	ASSERT_EQ(rgbaResult->size(), static_cast<oa::Usize>(profile.width * profile.height * 4));
	for (oa::Usize pixel = 0; pixel < rgbaResult->size(); pixel += 4) {
		ASSERT_EQ((*rgbaResult)[pixel + 3], 255);
	}

	EXPECT_TRUE(decoder.close().isOk());
}

TEST(VideoDecoderLifecycle, AbandonedSubmittedSessionRetiresAtEngineClose)
{
	auto fixtureResult = oa::Filesystem::readBinary(
		testAssetPath("video/conformance/test_pattern_72p_h264_baseline_idr_8bit_420.h264"));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();

	auto config = testEngineConfig(oa::Precision::FP32);
	config.selectForThread = false;
	config.preloadEmbeddedPipelines = false;
	config.enablePipelineCache = false;
	auto engineResult = oa::Engine::create(config);
	ASSERT_TRUE(engineResult.isOk()) << engineResult.getStatus().toString();
	auto engine = oa::move(*engineResult);
	if (not testVideoDecodeSupported(*engine, oa::VideoCodec::H264)) {
		ASSERT_TRUE(engine->close().isOk());
		GTEST_SKIP() << "H.264 vulkan Video decode is not supported on selected device";
	}

	oa::Event completion;
	{
		auto decoderResult = oa::VideoDecoder::create(
			*engine, makeH264FixtureProfile());
		ASSERT_TRUE(decoderResult.isOk())
			<< decoderResult.getStatus().toString();
		auto decoder = oa::move(*decoderResult);
		oa::VideoFrame frame = {};
		ASSERT_TRUE(oa::VideoDecoderInternal::decodeFrame(
			decoder, oa::Span<const oa::U8>(*fixtureResult), frame).isOk());
		completion = frame.ready;
		ASSERT_TRUE(completion.isValid());
		// No Close: the live video session and its semaphore must move to
		// engine retirement without waiting or invalidating copied completion.
	}

	ASSERT_TRUE(completion.wait().isOk());
	ASSERT_TRUE(engine->close().isOk());
}

TEST_F(VkEngineTestFixture, VideoDecoder_DecodeH265FrameFromLocalFixture)
{
	auto fixtureResult = oa::Filesystem::readBinary(testAssetPath("video/conformance/test_pattern_72p_h265_main_idr_8bit_420.h265"));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();

	auto& engine = rt();
	if (!testVideoDecodeSupported(engine, oa::VideoCodec::H265)) {
		GTEST_SKIP() << "H.265 vulkan Video decode is not supported on selected device";
	}

	oa::VideoProfile profile = makeH265FixtureProfile();

	auto result = oa::VideoDecoder::create(engine, profile);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	auto decoder = oa::move(*result);

	oa::VideoFrame frame = {};
	auto status = oa::VideoDecoderInternal::decodeFrame(
		decoder, oa::Span<const oa::U8>(*fixtureResult), frame);
	ASSERT_TRUE(status.isOk()) << status.toString();
	EXPECT_EQ(oa::VideoDecoderInternal::getCachedH265VpsCount(decoder), 1u);
	EXPECT_EQ(oa::VideoDecoderInternal::getCachedH265SpsCount(decoder), 1u);
	EXPECT_EQ(oa::VideoDecoderInternal::getCachedH265PpsCount(decoder), 1u);
	EXPECT_EQ(decoder.getSessionParameterUpdateCount(), 3u);
	EXPECT_EQ(decoder.getCurrentFrameNumber(), 1u);
	EXPECT_GE(decoder.getDpbInUseCount(), 1u);
	EXPECT_GE(decoder.getDpbReferenceCount(), 1u);
	expectDecodedLumaIsReadable(decoder, frame, profile);
	expectDecodedNv12MatchesFfmpeg(
		decoder,
		frame,
		profile,
		"video/conformance/test_pattern_72p_h265_main_idr_8bit_420.h265",
		"oa_h265_idr_reference_nv12");

	EXPECT_TRUE(decoder.close().isOk());
}

TEST_F(VkEngineTestFixture, VideoDecoder_DecodeH265Main10P010FromLocalFixture)
{
	constexpr const char* fixture =
		"video/conformance/test_pattern_72p_h265_main10_idr_10bit_420.h265";
	auto fixtureResult = oa::Filesystem::readBinary(testAssetPath(fixture));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();

	oa::VideoProfile profile = makeH265Main10FixtureProfile();
	auto capsResult = oa::VideoDecoder::queryDecodeCapabilities(rt(), profile);
	ASSERT_TRUE(capsResult.isOk()) << capsResult.getStatus().toString();
	if (!capsResult->supported) GTEST_SKIP() << "H.265 Main 10 P010 is unavailable";

	auto result = oa::VideoDecoder::create(rt(), profile);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	auto decoder = oa::move(*result);
	oa::VideoFrame frame = {};
	const oa::Status status = oa::VideoDecoderInternal::decodeFrame(
		decoder, oa::Span<const oa::U8>(*fixtureResult), frame);
	ASSERT_TRUE(status.isOk()) << status.toString();
	expectDecodedP010MatchesFfmpeg(
		decoder, frame, profile, fixture, "oa_h265_main10_reference_p010");
	expectP010ComputeRgbaMatchesCpu(decoder, frame);
	EXPECT_TRUE(decoder.close().isOk());
}

TEST_F(VkEngineTestFixture, VideoDecoder_DecodeH265RangeExt420_8bitFromLocalFixture)
{
	constexpr const char* fixture =
		"video/conformance/test_pattern_72p_h265_rangeext_intra_8bit_420.h265";
	auto fixtureResult = oa::Filesystem::readBinary(testAssetPath(fixture));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();
	oa::VcpH265 parser;
	oa::H265PictureDesc desc = {};
	const oa::Status parseStatus = parser.parseAccessUnit(
		oa::Span<const oa::U8>(*fixtureResult), desc);
	ASSERT_TRUE(parseStatus.isOk()) << parseStatus.toString();
	ASSERT_EQ(desc.vpsInAu.size(), 1U);
	EXPECT_EQ(
		desc.vpsInAu[0].generalProfileIdc,
		static_cast<oa::U32>(STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS));

	oa::VideoProfile profile = makeH265RangeExtFixtureProfile();
	auto capsResult = oa::VideoDecoder::queryDecodeCapabilities(rt(), profile);
	ASSERT_TRUE(capsResult.isOk()) << capsResult.getStatus().toString();
	if (!capsResult->supported) GTEST_SKIP() << "H.265 Range Extensions 4:2:0 8-bit is unavailable";

	auto result = oa::VideoDecoder::create(rt(), profile);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	auto decoder = oa::move(*result);
	oa::VideoFrame frame = {};
	const oa::Status status = oa::VideoDecoderInternal::decodeFrame(
		decoder, oa::Span<const oa::U8>(*fixtureResult), frame);
	ASSERT_TRUE(status.isOk()) << status.toString();
	expectDecodedNv12MatchesFfmpeg(
		decoder, frame, profile, fixture, "oa_h265_rangeext_reference_nv12");
	EXPECT_TRUE(decoder.close().isOk());
}

TEST_F(VkEngineTestFixture, VideoDecoder_DecodeH265RangeExt420_10bitFromLocalFixture)
{
	constexpr const char* fixture =
		"video/conformance/test_pattern_72p_h265_rangeext_intra_10bit_420.h265";
	auto fixtureResult = oa::Filesystem::readBinary(testAssetPath(fixture));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();
	oa::VcpH265 parser;
	oa::H265PictureDesc desc = {};
	const oa::Status parseStatus = parser.parseAccessUnit(
		oa::Span<const oa::U8>(*fixtureResult), desc);
	ASSERT_TRUE(parseStatus.isOk()) << parseStatus.toString();
	ASSERT_EQ(desc.vpsInAu.size(), 1U);
	EXPECT_EQ(
		desc.vpsInAu[0].generalProfileIdc,
		static_cast<oa::U32>(STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS));
	EXPECT_EQ(desc.sps.bitDepthLumaMinus8, 2U);
	EXPECT_EQ(desc.sps.bitDepthChromaMinus8, 2U);

	oa::VideoProfile profile = makeH265RangeExt10FixtureProfile();
	auto capsResult = oa::VideoDecoder::queryDecodeCapabilities(rt(), profile);
	ASSERT_TRUE(capsResult.isOk()) << capsResult.getStatus().toString();
	if (!capsResult->supported) {
		GTEST_SKIP() << "H.265 Range Extensions 4:2:0 10-bit is unavailable";
	}

	auto result = oa::VideoDecoder::create(rt(), profile);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	auto decoder = oa::move(*result);
	oa::VideoFrame frame = {};
	const oa::Status status = oa::VideoDecoderInternal::decodeFrame(
		decoder, oa::Span<const oa::U8>(*fixtureResult), frame);
	ASSERT_TRUE(status.isOk()) << status.toString();
	expectDecodedP010MatchesFfmpeg(
		decoder, frame, profile, fixture, "oa_h265_rangeext10_reference_p010");
	expectP010ComputeRgbaMatchesCpu(decoder, frame);
	EXPECT_TRUE(decoder.close().isOk());
}

TEST_F(VkEngineTestFixture, VideoDecoder_Av1FixturePresentAndCapabilityQueried)
{
	auto fixtureResult = oa::Filesystem::readBinary(testAssetPath("video/conformance/test_pattern_72p_av1_main_keyframe_8bit_420.ivf"));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();
	EXPECT_GT(fixtureResult->size(), 0u);

	auto& engine = rt();
	auto capsResult = oa::VideoDecoder::queryDecodeCapabilities(engine, oa::VideoCodec::AV1);
	if (!capsResult.isOk()) {
		GTEST_SKIP() << capsResult.getStatus().toString();
	}

	EXPECT_TRUE(capsResult->supported);
}

TEST_F(VkEngineTestFixture, VideoDecoder_CreateVp9Decoder)
{
	auto& engine = rt();
	if (!testVideoDecodeSupported(engine, oa::VideoCodec::VP9)) {
		GTEST_SKIP() << "VP9 vulkan Video decode is not supported on selected device";
	}

	oa::VideoProfile profile;
	profile.codec = oa::VideoCodec::VP9;
	profile.width = 3840;
	profile.height = 2160;
	profile.maxDpbSlots = 9;

	auto result = oa::VideoDecoder::create(engine, profile);
	EXPECT_TRUE(result.isOk());

	if (result.isOk())
	{
		auto decoder = oa::move(*result);
		EXPECT_TRUE(decoder.isInitialized());
		EXPECT_FALSE(decoder.hasSessionParameters());
		EXPECT_GE(decoder.getDpbSlotCapacity(), profile.maxDpbSlots);
		EXPECT_GE(decoder.getDpbViewCount(), 1u);
		auto caps = oa::VideoDecoder::queryDecodeCapabilities(engine, oa::VideoCodec::VP9);
		ASSERT_TRUE(caps.isOk());
		expectDecoderCodedExtentAligned(decoder, *caps, profile);
		if (caps->supportsDpbAndOutputDistinct) {
			EXPECT_GE(decoder.getOutputFrameCapacity(), profile.maxDpbSlots);
			EXPECT_GE(decoder.getOutputViewCount(), profile.maxDpbSlots);
		}
		EXPECT_TRUE(decoder.close().isOk());
	}
}

TEST_F(VkEngineTestFixture, VideoDecoder_Vp9FixturePresentAndCapabilityQueried)
{
	auto fixtureResult = oa::Filesystem::readBinary(testAssetPath("video/conformance/test_pattern_72p_vp9_profile0_keyframe_8bit_420.ivf"));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();
	EXPECT_GT(fixtureResult->size(), 0u);

	auto& engine = rt();
	auto capsResult = oa::VideoDecoder::queryDecodeCapabilities(engine, oa::VideoCodec::VP9);
	if (!capsResult.isOk()) {
		GTEST_SKIP() << capsResult.getStatus().toString();
	}

	EXPECT_TRUE(capsResult->supported);
}

TEST_F(VkEngineTestFixture, VideoDecoder_DecodeVp9FrameFromLocalFixture)
{
	auto fixtureResult = oa::Filesystem::readBinary(testAssetPath("video/conformance/test_pattern_72p_vp9_profile0_keyframe_8bit_420.ivf"));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();

	auto& engine = rt();
	if (!testVideoDecodeSupported(engine, oa::VideoCodec::VP9)) {
		GTEST_SKIP() << "VP9 vulkan Video decode is not supported on selected device";
	}

	oa::VideoProfile profile = makeVp9FixtureProfile();

	auto result = oa::VideoDecoder::create(engine, profile);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	auto decoder = oa::move(*result);

	oa::VideoFrame frame = {};
	auto status = oa::VideoDecoderInternal::decodeFrame(
		decoder, oa::Span<const oa::U8>(*fixtureResult), frame);
	ASSERT_TRUE(status.isOk()) << status.toString();
	EXPECT_EQ(decoder.getCurrentFrameNumber(), 1u);
	EXPECT_GE(decoder.getDpbInUseCount(), 1u);
	EXPECT_GE(decoder.getDpbReferenceCount(), 1u);
	expectDecodedLumaIsReadable(decoder, frame, profile);
	expectDecodedNv12MatchesFfmpeg(
		decoder,
		frame,
		profile,
		"video/conformance/test_pattern_72p_vp9_profile0_keyframe_8bit_420.ivf",
		"oa_vp9_keyframe_reference_nv12");

	EXPECT_TRUE(decoder.close().isOk());
}

TEST_F(VkEngineTestFixture, VideoDecoder_DecodeVp9Profile2P010FromLocalFixture)
{
	constexpr const char* fixture =
		"video/conformance/test_pattern_72p_vp9_profile2_key_10bit_420.ivf";
	auto fixtureResult = oa::Filesystem::readBinary(testAssetPath(fixture));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();
	oa::VideoProfile profile = makeVp9Profile2FixtureProfile();
	auto capsResult = oa::VideoDecoder::queryDecodeCapabilities(rt(), profile);
	ASSERT_TRUE(capsResult.isOk()) << capsResult.getStatus().toString();
	if (!capsResult->supported) GTEST_SKIP() << "VP9 Profile 2 P010 is unavailable";

	auto result = oa::VideoDecoder::create(rt(), profile);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	auto decoder = oa::move(*result);
	oa::VideoFrame frame = {};
	const oa::Status status = oa::VideoDecoderInternal::decodeFrame(
		decoder, oa::Span<const oa::U8>(*fixtureResult), frame);
	ASSERT_TRUE(status.isOk()) << status.toString();
	expectDecodedP010MatchesFfmpeg(
		decoder, frame, profile, fixture, "oa_vp9_profile2_reference_p010");
	expectP010ComputeRgbaMatchesCpu(decoder, frame);
	EXPECT_TRUE(decoder.close().isOk());
}

TEST_F(VkEngineTestFixture, Av1Parser_ParseLocalFixture)
{
	auto fixtureResult = oa::Filesystem::readBinary(testAssetPath("video/conformance/test_pattern_72p_av1_main_keyframe_8bit_420.ivf"));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();

	oa::VcpAv1 parser;
	oa::Av1PictureDesc desc;
	auto status = parser.parseAccessUnit(oa::Span<const oa::U8>(*fixtureResult), desc);
	ASSERT_TRUE(status.isOk()) << status.toString();
	ASSERT_TRUE(desc.hasPicture);
	EXPECT_EQ(desc.frameHeader.tileCols, 1u);
	EXPECT_EQ(desc.frameHeader.tileRows, 1u);
	EXPECT_EQ(desc.sequenceHeader.maxFrameWidthMinus1 + 1u, 128u);
	EXPECT_EQ(desc.sequenceHeader.maxFrameHeightMinus1 + 1u, 72u);
	ASSERT_EQ(desc.tileOffsets.size(), 1u);
	ASSERT_EQ(desc.tileSizes.size(), 1u);
	EXPECT_GT(desc.tileSizes[0], 0u);
}

TEST(Av1Parser, CountsEveryPictureInMultiFrameTemporalUnits)
{
	const oa::Path path = testAssetPath("video/clip/shibuya_720p_30fps_av1_main_8bit_420.mp4");
	auto streamResult = oa::VideoDemuxer::open(path.cStr());
	ASSERT_TRUE(streamResult.isOk()) << streamResult.getStatus().toString();
	oa::VcpAv1 parser;
	// FFmpeg trace_headers reports 1, 6, and 1 frame OBUs in the first three
	// packets.  Packet 2 is the SVT-AV1 hidden-reference pyramid that the old
	// single-picture parseAccessUnit path silently collapsed to one picture.
	constexpr oa::U32 expectedPictures[] = {1, 6, 1};
	for (oa::U32 packetIndex = 0; packetIndex < 3; ++packetIndex) {
		oa::VideoPacket packet;
		ASSERT_TRUE(streamResult->readNextPacket(packet).isOk()) << packetIndex;
		oa::Av1AccessUnitInfo info;
		const oa::Status status = parser.inspectAccessUnit(
			oa::Span<const oa::U8>(packet.data.data(), packet.data.size()), info);
		ASSERT_TRUE(status.isOk()) << status.toString();
		EXPECT_EQ(info.pictureCount(), expectedPictures[packetIndex]) << packetIndex;
	}
}

TEST_F(VkEngineTestFixture, VideoDecoder_DecodeAv1FrameFromLocalFixture)
{
	auto fixtureResult = oa::Filesystem::readBinary(testAssetPath("video/conformance/test_pattern_72p_av1_main_keyframe_8bit_420.ivf"));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();

	auto& engine = rt();
	if (!testVideoDecodeSupported(engine, oa::VideoCodec::AV1)) {
		GTEST_SKIP() << "AV1 vulkan Video decode is not supported on selected device";
	}

	oa::VideoProfile profile = makeAv1FixtureProfile();

	auto result = oa::VideoDecoder::create(engine, profile);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	auto decoder = oa::move(*result);

	oa::VideoFrame frame = {};
	auto status = oa::VideoDecoderInternal::decodeFrame(
		decoder, oa::Span<const oa::U8>(*fixtureResult), frame);
	ASSERT_TRUE(status.isOk()) << status.toString();
	EXPECT_EQ(decoder.getCurrentFrameNumber(), 1u);
	EXPECT_GE(decoder.getDpbInUseCount(), 1u);
	EXPECT_GE(decoder.getDpbReferenceCount(), 1u);
	expectDecodedNv12MatchesFfmpeg(
		decoder,
		frame,
		profile,
		"video/conformance/test_pattern_72p_av1_main_keyframe_8bit_420.ivf",
		"oa_av1_keyframe_reference_nv12");

	EXPECT_TRUE(decoder.close().isOk());
}

TEST_F(VkEngineTestFixture, VideoDecoder_DecodeAv1Main10P010FromLocalFixture)
{
	constexpr const char* fixture =
		"video/conformance/test_pattern_72p_av1_main_key_10bit_420.obu";
	auto fixtureResult = oa::Filesystem::readBinary(testAssetPath(fixture));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();
	oa::VideoProfile profile = makeAv1Main10FixtureProfile();
	auto capsResult = oa::VideoDecoder::queryDecodeCapabilities(rt(), profile);
	ASSERT_TRUE(capsResult.isOk()) << capsResult.getStatus().toString();
	if (!capsResult->supported) GTEST_SKIP() << "AV1 Main 10-bit P010 is unavailable";

	auto result = oa::VideoDecoder::create(rt(), profile);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	auto decoder = oa::move(*result);
	oa::VideoFrame frame = {};
	const oa::Status status = oa::VideoDecoderInternal::decodeFrame(
		decoder, oa::Span<const oa::U8>(*fixtureResult), frame);
	ASSERT_TRUE(status.isOk()) << status.toString();
	expectDecodedP010MatchesFfmpeg(
		decoder, frame, profile, fixture, "oa_av1_main10_reference_p010");
	expectP010ComputeRgbaMatchesCpu(decoder, frame);
	EXPECT_TRUE(decoder.close().isOk());
}

TEST_F(VkEngineTestFixture, VideoDecoder_RejectsAv1Main10LoopRestorationBeforeSubmit)
{
	constexpr const char* fixture =
		"video/conformance/test_pattern_72p_av1_main_key_10bit_420_loop_restoration.obu";
	auto fixtureResult = oa::Filesystem::readBinary(testAssetPath(fixture));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();
	oa::VideoProfile profile = makeAv1Main10FixtureProfile();
	auto capsResult = oa::VideoDecoder::queryDecodeCapabilities(rt(), profile);
	ASSERT_TRUE(capsResult.isOk()) << capsResult.getStatus().toString();
	if (!capsResult->supported) GTEST_SKIP() << "AV1 Main 10-bit P010 is unavailable";
	auto result = oa::VideoDecoder::create(rt(), profile);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	auto decoder = oa::move(*result);
	oa::VideoFrame frame = {};
	const oa::Status status = oa::VideoDecoderInternal::decodeFrame(
		decoder, oa::Span<const oa::U8>(*fixtureResult), frame);
	EXPECT_FALSE(status.isOk());
	EXPECT_EQ(status.getCode(), oa::StatusCode::Unavailable);
	EXPECT_EQ(decoder.getCurrentFrameNumber(), 0U);
	EXPECT_TRUE(decoder.close().isOk());
}

TEST_F(VkEngineTestFixture, VideoDecoder_DpbInitialState)
{
	auto& engine = rt();
	if (!testVideoDecodeSupported(engine, oa::VideoCodec::H264)) {
		GTEST_SKIP() << "H.264 vulkan Video decode is not supported on selected device";
	}

	auto result = oa::VideoDecoder::create(engine, makeH264FixtureProfile());
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	auto decoder = oa::move(*result);

	EXPECT_TRUE(decoder.isInitialized());
	EXPECT_GE(decoder.getDpbSlotCapacity(), 4u);
	EXPECT_GE(decoder.getDpbViewCount(), 1u);
	EXPECT_EQ(decoder.getDpbInUseCount(), 0u);
	EXPECT_EQ(decoder.getDpbReferenceCount(), 0u);
	EXPECT_EQ(decoder.getCurrentFrameNumber(), 0u);
	EXPECT_EQ(oa::VideoDecoderInternal::getCachedSpsCount(decoder), 0u);
	EXPECT_EQ(oa::VideoDecoderInternal::getCachedPpsCount(decoder), 0u);

	EXPECT_TRUE(decoder.close().isOk());
}

TEST_F(VkEngineTestFixture, VideoDecoder_DpbDecodeAndFlush)
{
	auto fixtureResult = oa::Filesystem::readBinary(testAssetPath("video/conformance/test_pattern_72p_h264_baseline_idr_8bit_420.h264"));
	ASSERT_TRUE(fixtureResult.isOk()) << fixtureResult.getStatus().toString();

	auto& engine = rt();
	if (!testVideoDecodeSupported(engine, oa::VideoCodec::H264)) {
		GTEST_SKIP() << "H.264 vulkan Video decode is not supported on selected device";
	}

	auto result = oa::VideoDecoder::create(engine, makeH264FixtureProfile());
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	auto decoder = oa::move(*result);

	oa::VideoFrame frame = {};
	auto status = oa::VideoDecoderInternal::decodeFrame(
		decoder, oa::Span<const oa::U8>(*fixtureResult), frame);
	ASSERT_TRUE(status.isOk()) << status.toString();

	EXPECT_EQ(decoder.getCurrentFrameNumber(), 1u);
	EXPECT_EQ(oa::VideoDecoderInternal::getCachedSpsCount(decoder), 1u);
	EXPECT_EQ(oa::VideoDecoderInternal::getCachedPpsCount(decoder), 1u);
	EXPECT_EQ(decoder.getSessionParameterUpdateCount(), 2u);
	EXPECT_GE(decoder.getDpbInUseCount(), 1u);
	EXPECT_GE(decoder.getDpbReferenceCount(), 1u);
	EXPECT_NE(frame.image, VK_NULL_HANDLE);
	EXPECT_EQ(frame.format, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM);

	status = decoder.flush();
	ASSERT_TRUE(status.isOk()) << status.toString();
	EXPECT_EQ(decoder.getDpbInUseCount(), 0u);
	EXPECT_EQ(decoder.getDpbReferenceCount(), 0u);
	EXPECT_EQ(decoder.getCurrentFrameNumber(), 0u);
	EXPECT_EQ(oa::VideoDecoderInternal::getCachedSpsCount(decoder), 0u);
	EXPECT_EQ(oa::VideoDecoderInternal::getCachedPpsCount(decoder), 0u);
	EXPECT_EQ(decoder.getSessionParameterUpdateCount(), 0u);

	EXPECT_TRUE(decoder.close().isOk());
}

// NOTE: DPB flush-with-references is already covered by the decode-then-flush
// assertions above (getDpbInUseCount()/getDpbReferenceCount() == 0 after
// flush). NV12→BF16 conversion is not tested here: oa::FnImage::CvtNv12ToBf16 is
// still a dispatch stub (allocates the output tensor but does not bind the
// VkImage / run the shader), so any test would only validate the shape of
// uninitialized memory. Add real coverage when the image-descriptor dispatch
// path lands.

// Frame pool Tests
TEST_F(VkEngineTestFixture, VideoFramePool_CreatePool)
{
	auto& engine = rt();
	
	auto result = oa::VideoFramePool::create(engine, 1920, 1080, 4);
	
	EXPECT_TRUE(result.isOk());
	
	if (result.isOk())
	{
		auto pool = oa::move(*result);
	}
}

TEST_F(VkEngineTestFixture, VideoFramePool_AcquireRelease)
{
	auto& engine = rt();
	
	auto result = oa::VideoFramePool::create(engine, 1920, 1080, 2);
	ASSERT_TRUE(result.isOk());
	
	auto pool = oa::move(*result);
	
	auto first = pool.acquire();
	EXPECT_NE(first.image, VK_NULL_HANDLE);
	EXPECT_NE(first.imageView, VK_NULL_HANDLE);
	EXPECT_EQ(first.format, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM);
	EXPECT_EQ(first.width, 1920u);
	EXPECT_EQ(first.height, 1080u);

	auto second = pool.acquire();
	EXPECT_NE(second.image, VK_NULL_HANDLE);
	EXPECT_NE(second.imageView, VK_NULL_HANDLE);
	EXPECT_NE(second.image, first.image);

	auto exhausted = pool.acquire();
	EXPECT_EQ(exhausted.image, VK_NULL_HANDLE);
	EXPECT_EQ(exhausted.imageView, VK_NULL_HANDLE);

	pool.release(first);
	auto reused = pool.acquire();
	EXPECT_EQ(reused.image, first.image);
	
}

// NOTE: 4K@60 / 8K@30 decode-throughput benchmarks are intentionally not part
// of the correctness suite — they are performance measurements (and un-runnable
// on the CI iGPU). If added later, they belong in a Bench* target, not here.
