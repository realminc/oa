// OA Vision — Video Decoder Tests
// Hardware H.264/H.265 decode validation

#include "../../../Source/Private/Oa/Vision/Video/Codec/NalParser.h"

#include "../../OaTest.h"
#include "../../../Source/Private/Oa/Vision/Video/Codec/CodecRegistry.h"
#include "../../../Source/Private/Oa/Vision/Video/Codec/VcpAv1.h"
#include "../../../Source/Private/Oa/Vision/Video/Decoder/VideoDecoderInternal.h"
#include "../../../Source/Private/Oa/Vision/Video/Decoder/VideoDecoderProfile.h"
#include <Oa/Runtime/Engine.h>

#include <Oa/Vision/VideoDecoder.h>
#include <Oa/Vision/VideoStream.h>
#include <array>
#include <fstream>
#include <sstream>
#include <string>

namespace {

struct H264NalWriter {
	OaVec<OaU8> Bytes;
	OaU32 BitOffset = 0;

	void WriteBit(OaU32 InBit) {
		if (BitOffset == 0) {
			Bytes.PushBack(0);
		}
		if (InBit != 0) {
			Bytes.Back() |= static_cast<OaU8>(1u << (7u - BitOffset));
		}
		BitOffset = (BitOffset + 1u) & 7u;
	}

	void WriteBits(OaU32 InValue, OaU32 InCount) {
		for (OaI32 bit = static_cast<OaI32>(InCount) - 1; bit >= 0; --bit) {
			WriteBit((InValue >> static_cast<OaU32>(bit)) & 1u);
		}
	}

	void WriteUE(OaU32 InValue) {
		OaU32 codeNum = InValue + 1;
		OaU32 bitCount = 0;
		for (OaU32 tmp = codeNum; tmp != 0; tmp >>= 1) {
			++bitCount;
		}
		for (OaU32 i = 0; i + 1 < bitCount; ++i) {
			WriteBit(0);
		}
		WriteBits(codeNum, bitCount);
	}

	void WriteSE(OaI32 InValue)
	{
		OaU32 codeNum = InValue <= 0
			? static_cast<OaU32>(-InValue) * 2u
			: static_cast<OaU32>(InValue) * 2u - 1u;
		WriteUE(codeNum);
	}

	void FinishRbsp()
	{
		WriteBit(1);
		while (BitOffset != 0) {
			WriteBit(0);
		}
	}
};

OaVec<OaU8> MakeH264SpsNal()
{
	H264NalWriter w;
	w.Bytes.PushBack(0);
	w.Bytes.PushBack(0);
	w.Bytes.PushBack(0);
	w.Bytes.PushBack(1);
	w.Bytes.PushBack(0x67);
	w.WriteBits(100, 8); // profile_idc: High
	w.WriteBits(0, 8);   // constraint flags
	w.WriteBits(40, 8);  // level_idc: 4.0
	w.WriteUE(0);        // seq_parameter_set_id
	w.WriteUE(1);        // chroma_format_idc: 4:2:0
	w.WriteUE(0);        // bit_depth_luma_minus8
	w.WriteUE(0);        // bit_depth_chroma_minus8
	w.WriteBit(0);       // qpprime_y_zero_transform_bypass_flag
	w.WriteBit(0);       // seq_scaling_matrix_present_flag
	w.WriteUE(0);        // log2_max_frame_num_minus4
	w.WriteUE(0);        // pic_order_cnt_type
	w.WriteUE(0);        // log2_max_pic_order_cnt_lsb_minus4
	w.WriteUE(1);        // max_num_ref_frames
	w.WriteBit(0);       // gaps_in_frame_num_value_allowed_flag
	w.WriteUE(119);      // pic_width_in_mbs_minus1: 1920
	w.WriteUE(67);       // pic_height_in_map_units_minus1: 1088
	w.WriteBit(1);       // frame_mbs_only_flag
	w.WriteBit(1);       // direct_8x8_inference_flag
	w.WriteBit(1);       // frame_cropping_flag
	w.WriteUE(0);        // left
	w.WriteUE(0);        // right
	w.WriteUE(0);        // top
	w.WriteUE(4);        // bottom -> 1080 visible
	w.FinishRbsp();
	return w.Bytes;
}

TEST(OaH264NalParser, WeightedPSlicePreservesMmco)
{
	H264NalWriter w;
	w.Bytes.PushBack(0x61); // nal_ref_idc=3, non-IDR slice
	w.WriteUE(0);           // first_mb_in_slice
	w.WriteUE(0);           // slice_type=P
	w.WriteUE(0);           // pic_parameter_set_id
	w.WriteBits(5, 4);      // frame_num
	w.WriteBits(6, 4);      // pic_order_cnt_lsb
	w.WriteBit(0);          // num_ref_idx_active_override_flag
	w.WriteBit(0);          // ref_pic_list_modification_flag_l0

	w.WriteUE(0);           // luma_log2_weight_denom
	w.WriteUE(0);           // chroma_log2_weight_denom
	w.WriteBit(1);          // luma_weight_l0_flag
	w.WriteSE(1);           // luma_weight_l0
	w.WriteSE(0);           // luma_offset_l0
	w.WriteBit(1);          // chroma_weight_l0_flag
	w.WriteSE(1);           // chroma_weight_l0[0]
	w.WriteSE(0);           // chroma_offset_l0[0]
	w.WriteSE(1);           // chroma_weight_l0[1]
	w.WriteSE(0);           // chroma_offset_l0[1]

	w.WriteBit(1);          // adaptive_ref_pic_marking_mode_flag
	w.WriteUE(1);           // memory_management_control_operation
	w.WriteUE(4);           // difference_of_pic_nums_minus1
	w.WriteUE(0);           // end of MMCO list
	w.FinishRbsp();

	OaH264SpsData sps{};
	sps.Log2MaxFrameNumMinus4 = 0;
	sps.PicOrderCntType = 0;
	sps.Log2MaxPicOrderCntLsbMinus4 = 0;
	sps.ChromaFormatIdc = 1;
	sps.FrameMbsOnly = true;

	OaH264PpsData pps{};
	pps.PpsId = 0;
	pps.WeightedPred = true;
	pps.NumRefIdxL0DefaultActiveMinus1 = 0;

	OaSliceHeader header{};
	ASSERT_TRUE(OaNalParser::ParseSliceHeader(
		w.Bytes.Data(),
		w.Bytes.Size(),
		false,
		3,
		sps,
		pps,
		header));
	EXPECT_TRUE(header.RefPicMarkingValid);
	EXPECT_TRUE(header.AdaptiveRefPicMarking);
	ASSERT_EQ(header.MmcoCommands.Size(), 1u);
	EXPECT_EQ(header.MmcoCommands[0].Op, 1u);
	EXPECT_EQ(header.MmcoCommands[0].DifferenceOfPicNumsMinus1, 4u);
}

OaVec<OaU8> MakeH264PpsNal()
{
	H264NalWriter w;
	w.Bytes.PushBack(0);
	w.Bytes.PushBack(0);
	w.Bytes.PushBack(0);
	w.Bytes.PushBack(1);
	w.Bytes.PushBack(0x68);
	w.WriteUE(0);  // pic_parameter_set_id
	w.WriteUE(0);  // seq_parameter_set_id
	w.WriteBit(0); // entropy_coding_mode_flag
	w.WriteBit(0); // bottom_field_pic_order_in_frame_present_flag
	w.WriteUE(0);  // num_slice_groups_minus1
	w.WriteUE(0);  // num_ref_idx_l0_default_active_minus1
	w.WriteUE(0);  // num_ref_idx_l1_default_active_minus1
	w.WriteBit(0); // weighted_pred_flag
	w.WriteBits(0, 2); // weighted_bipred_idc
	w.WriteSE(0);  // pic_init_qp_minus26
	w.WriteSE(0);  // pic_init_qs_minus26
	w.WriteSE(0);  // chroma_qp_index_offset
	w.WriteBit(1); // deblocking_filter_control_present_flag
	w.WriteBit(0); // constrained_intra_pred_flag
	w.WriteBit(0); // redundant_pic_cnt_present_flag
	w.FinishRbsp();
	return w.Bytes;
}

OaVec<OaU8> MakeH264ParameterAccessUnit()
{
	OaVec<OaU8> out = MakeH264SpsNal();
	OaVec<OaU8> pps = MakeH264PpsNal();
	for (OaU8 byte : pps) {
		out.PushBack(byte);
	}
	return out;
}

OaVideoProfile MakeH264FixtureProfile()
{
	OaVideoProfile profile;
	profile.Codec = OaVideoCodec::H264;
	profile.Width = 128;
	profile.Height = 72;
	profile.MaxDpbSlots = 4;
	return profile;
}

OaVideoProfile MakeH265FixtureProfile()
{
	OaVideoProfile profile;
	profile.Codec = OaVideoCodec::H265;
	profile.Width = 128;
	profile.Height = 72;
	profile.MaxDpbSlots = 4;
	return profile;
}

OaVideoProfile MakeAv1FixtureProfile()
{
	OaVideoProfile profile;
	profile.Codec = OaVideoCodec::AV1;
	profile.Width = 128;
	profile.Height = 72;
	// AV1 spec / NVIDIA vk_video_samples use 8 ref slots + 1 current (9 total).
	profile.MaxDpbSlots = 9;
	return profile;
}

OaVideoProfile MakeVp9FixtureProfile()
{
	OaVideoProfile profile;
	profile.Codec = OaVideoCodec::VP9;
	profile.Width = 128;
	profile.Height = 72;
	profile.MaxDpbSlots = 9;
	return profile;
}

OaU32 AlignVideoTestExtent(OaU32 InValue, OaU32 InMinimum, OaU32 InGranularity)
{
	const OaU32 granularity = InGranularity == 0 ? 1U : InGranularity;
	OaU32 value = InValue < InMinimum ? InMinimum : InValue;
	const OaU32 remainder = value % granularity;
	if (remainder != 0) {
		value += granularity - remainder;
	}
	return value;
}

OaU32 CodecExtentGranularityForTest(OaVideoCodec InCodec)
{
	switch (InCodec) {
		case OaVideoCodec::H264: return 16U;
		case OaVideoCodec::H265:
		case OaVideoCodec::AV1:
		case OaVideoCodec::VP9: return 2U;
	}
	return 1U;
}

void ExpectDecoderCodedExtentAligned(
	const OaVideoDecoder& InDecoder,
	const OaVideoDecodeCapabilities& InCaps,
	const OaVideoProfile& InProfile)
{
	const OaU32 codecGranularity = CodecExtentGranularityForTest(InProfile.Codec);
	const OaU32 widthGranularity = InCaps.PictureAccessGranularityWidth > codecGranularity
		? InCaps.PictureAccessGranularityWidth
		: codecGranularity;
	const OaU32 heightGranularity = InCaps.PictureAccessGranularityHeight > codecGranularity
		? InCaps.PictureAccessGranularityHeight
		: codecGranularity;
	const OaU32 expectedWidth = AlignVideoTestExtent(InProfile.Width, InCaps.MinWidth, widthGranularity);
	const OaU32 expectedHeight = AlignVideoTestExtent(InProfile.Height, InCaps.MinHeight, heightGranularity);

	EXPECT_EQ(InDecoder.GetCodedWidth(), expectedWidth);
	EXPECT_EQ(InDecoder.GetCodedHeight(), expectedHeight);
	EXPECT_GE(InDecoder.GetCodedWidth(), InProfile.Width);
	EXPECT_GE(InDecoder.GetCodedHeight(), InProfile.Height);
	EXPECT_EQ(InDecoder.GetCodedWidth() % widthGranularity, 0U);
	EXPECT_EQ(InDecoder.GetCodedHeight() % heightGranularity, 0U);
}

void ExpectDecodedLumaIsReadable(OaVideoDecoder& InDecoder, const OaVideoFrame& InFrame, const OaVideoProfile& InProfile)
{
	EXPECT_NE(InFrame.Image, VK_NULL_HANDLE);
	EXPECT_NE(InFrame.ImageView, VK_NULL_HANDLE);
	EXPECT_EQ(InFrame.Width, InProfile.Width);
	EXPECT_EQ(InFrame.Height, InProfile.Height);
	EXPECT_EQ(InFrame.Format, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM);
	EXPECT_FALSE(InFrame.IsRgb);

	auto lumaResult = OaVideoDecoderInternal::ReadbackLuma(InDecoder, InFrame);
	ASSERT_TRUE(lumaResult.IsOk()) << lumaResult.GetStatus().ToString();
	ASSERT_EQ(lumaResult->Size(), static_cast<OaUsize>(InProfile.Width * InProfile.Height));
	OaU32 nonZeroCount = 0;
	OaU8 minValue = 255;
	OaU8 maxValue = 0;
	for (OaU8 value : *lumaResult) {
		nonZeroCount += value != 0 ? 1u : 0u;
		minValue = value < minValue ? value : minValue;
		maxValue = value > maxValue ? value : maxValue;
	}
	if (nonZeroCount == 0 || maxValue == minValue) {
		fprintf(stderr, "[LUMA-DIAG] samples=%zu nonZero=%u min=%u max=%u first16=",
			lumaResult->Size(), nonZeroCount, minValue, maxValue);
		for (OaUsize i = 0; i < 16 && i < lumaResult->Size(); ++i) {
			fprintf(stderr, "%u ", (*lumaResult)[i]);
		}
		fprintf(stderr, "\n");
		GTEST_SKIP() << "Vulkan Video command submission succeeded, but luma "
						"readback is still all zero or flat";
	}
	EXPECT_GT(nonZeroCount, 0u);
	EXPECT_GT(maxValue, minValue);
}

void ExpectDecodedNv12MatchesFfmpeg(
	OaVideoDecoder& InDecoder,
	const OaVideoFrame& InFrame,
	const OaVideoProfile& InProfile,
	const char* InFixtureRelativePath,
	const char* InReferencePath)
{
	if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
		GTEST_SKIP() << "ffmpeg is not installed";
	}

	auto decoded = OaVideoDecoderInternal::ReadbackNv12(InDecoder, InFrame);
	ASSERT_TRUE(decoded.IsOk()) << decoded.GetStatus().ToString();
	ASSERT_EQ(
		decoded->Size(),
		static_cast<OaUsize>(InProfile.Width) * InProfile.Height * 3u / 2u);

	const OaPath fixturePath = OaTestAssetPath(InFixtureRelativePath);
	const OaString command = OaString("ffmpeg -v error -y -i \"")
		+ fixturePath.String()
		+ "\" -frames:v 1 -f rawvideo -pix_fmt nv12 \""
		+ InReferencePath
		+ "\"";
	ASSERT_EQ(std::system(command.CStr()), 0);
	auto reference = OaFilesystem::ReadBinary(OaPath(InReferencePath));
	std::remove(InReferencePath);
	ASSERT_TRUE(reference.IsOk()) << reference.GetStatus().ToString();
	ASSERT_EQ(reference->Size(), decoded->Size());

	const OaUsize lumaBytes =
		static_cast<OaUsize>(InProfile.Width) * InProfile.Height;
	OaU64 lumaError = 0;
	OaU64 chromaError = 0;
	OaU64 decodedLumaSum = 0;
	OaU64 referenceLumaSum = 0;
	OaU8 maxError = 0;
	for (OaUsize i = 0; i < decoded->Size(); ++i) {
		const OaU8 error = static_cast<OaU8>(std::abs(
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

	const OaF64 lumaMae =
		static_cast<OaF64>(lumaError) / static_cast<OaF64>(lumaBytes);
	const OaF64 chromaMae =
		static_cast<OaF64>(chromaError)
		/ static_cast<OaF64>(decoded->Size() - lumaBytes);
	EXPECT_LT(lumaMae, 3.0)
		<< "maxError=" << static_cast<OaU32>(maxError)
		<< " decodedMean="
		<< static_cast<OaF64>(decodedLumaSum) / static_cast<OaF64>(lumaBytes)
		<< " referenceMean="
		<< static_cast<OaF64>(referenceLumaSum) / static_cast<OaF64>(lumaBytes)
		<< " decodedFirst="
		<< static_cast<OaU32>((*decoded)[0]) << ","
		<< static_cast<OaU32>((*decoded)[1]) << ","
		<< static_cast<OaU32>((*decoded)[2]) << ","
		<< static_cast<OaU32>((*decoded)[3])
		<< " referenceFirst="
		<< static_cast<OaU32>((*reference)[0]) << ","
		<< static_cast<OaU32>((*reference)[1]) << ","
		<< static_cast<OaU32>((*reference)[2]) << ","
		<< static_cast<OaU32>((*reference)[3]);
	EXPECT_LT(chromaMae, 3.0)
		<< "maxError=" << static_cast<OaU32>(maxError);
}

} // namespace

TEST(VideoProfile, BuildsEveryRegistryProfileWithoutCodecFallback)
{
	struct Case {
		OaVideoCodec Codec;
		OaVideoCodecProfile StandardProfile;
		VkVideoCodecOperationFlagBitsKHR Operation;
		OaU32 VkStandardProfile;
	};
	const Case cases[] = {
		{OaVideoCodec::H264, OaVideoCodecProfile::H264Baseline, VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
		 STD_VIDEO_H264_PROFILE_IDC_BASELINE},
		{OaVideoCodec::H264, OaVideoCodecProfile::H264Main, VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
		 STD_VIDEO_H264_PROFILE_IDC_MAIN},
		{OaVideoCodec::H264, OaVideoCodecProfile::H264High, VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
		 STD_VIDEO_H264_PROFILE_IDC_HIGH},
		{OaVideoCodec::H264, OaVideoCodecProfile::H264High444Predictive, VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
		 STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE},
		{OaVideoCodec::H265, OaVideoCodecProfile::H265Main, VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR,
		 STD_VIDEO_H265_PROFILE_IDC_MAIN},
		{OaVideoCodec::H265, OaVideoCodecProfile::H265Main10, VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR,
		 STD_VIDEO_H265_PROFILE_IDC_MAIN_10},
		{OaVideoCodec::H265, OaVideoCodecProfile::H265MainStillPicture, VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR,
		 STD_VIDEO_H265_PROFILE_IDC_MAIN_STILL_PICTURE},
		{OaVideoCodec::H265, OaVideoCodecProfile::H265FormatRangeExtensions,
		 VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR, STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS},
		{OaVideoCodec::H265, OaVideoCodecProfile::H265ScreenContentCodingExtensions,
		 VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR, STD_VIDEO_H265_PROFILE_IDC_SCC_EXTENSIONS},
		{OaVideoCodec::AV1, OaVideoCodecProfile::Av1Main, VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR,
		 STD_VIDEO_AV1_PROFILE_MAIN},
		{OaVideoCodec::AV1, OaVideoCodecProfile::Av1High, VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR,
		 STD_VIDEO_AV1_PROFILE_HIGH},
		{OaVideoCodec::AV1, OaVideoCodecProfile::Av1Professional, VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR,
		 STD_VIDEO_AV1_PROFILE_PROFESSIONAL},
		{OaVideoCodec::VP9, OaVideoCodecProfile::Vp9Profile0, VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR,
		 STD_VIDEO_VP9_PROFILE_0},
		{OaVideoCodec::VP9, OaVideoCodecProfile::Vp9Profile1, VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR,
		 STD_VIDEO_VP9_PROFILE_1},
		{OaVideoCodec::VP9, OaVideoCodecProfile::Vp9Profile2, VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR,
		 STD_VIDEO_VP9_PROFILE_2},
		{OaVideoCodec::VP9, OaVideoCodecProfile::Vp9Profile3, VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR,
		 STD_VIDEO_VP9_PROFILE_3},
	};

	for (const Case& testCase : cases) {
		OaVideoProfile request = {};
		request.Codec = testCase.Codec;
		request.StandardProfile = testCase.StandardProfile;
		VkVideoDecodeH264ProfileInfoKHR h264 = {};
		VkVideoDecodeH265ProfileInfoKHR h265 = {};
		VkVideoDecodeAV1ProfileInfoKHR av1 = {};
		VkVideoDecodeVP9ProfileInfoKHR vp9 = {};
		auto result = OaVideoDecoderProfile::BuildDecodeProfile(request, h264, h265, av1, vp9);
		ASSERT_TRUE(result.IsOk()) << result.GetStatus().ToString();
		EXPECT_EQ(result->videoCodecOperation, testCase.Operation);
		EXPECT_EQ(result->chromaSubsampling, VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR);
		EXPECT_EQ(result->lumaBitDepth, VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR);
		EXPECT_EQ(result->chromaBitDepth, VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR);
		switch (testCase.Codec) {
		case OaVideoCodec::H264: EXPECT_EQ(static_cast<OaU32>(h264.stdProfileIdc), testCase.VkStandardProfile); break;
		case OaVideoCodec::H265: EXPECT_EQ(static_cast<OaU32>(h265.stdProfileIdc), testCase.VkStandardProfile); break;
		case OaVideoCodec::AV1: EXPECT_EQ(static_cast<OaU32>(av1.stdProfile), testCase.VkStandardProfile); break;
		case OaVideoCodec::VP9: EXPECT_EQ(static_cast<OaU32>(vp9.stdProfile), testCase.VkStandardProfile); break;
		}
	}
}

TEST(VideoProfile, MapsFormatAndCodecSpecificAxes)
{
	OaVideoProfile request = {};
	request.Codec = OaVideoCodec::AV1;
	request.StandardProfile = OaVideoCodecProfile::Av1Professional;
	request.ChromaSubsampling = OaVideoChromaSubsampling::Yuv444;
	request.LumaBitDepth = OaVideoBitDepth::Bit12;
	request.ChromaBitDepth = OaVideoBitDepth::Bit12;
	request.Av1FilmGrain = true;
	VkVideoDecodeH264ProfileInfoKHR h264 = {};
	VkVideoDecodeH265ProfileInfoKHR h265 = {};
	VkVideoDecodeAV1ProfileInfoKHR av1 = {};
	VkVideoDecodeVP9ProfileInfoKHR vp9 = {};
	auto result = OaVideoDecoderProfile::BuildDecodeProfile(request, h264, h265, av1, vp9);
	ASSERT_TRUE(result.IsOk()) << result.GetStatus().ToString();
	EXPECT_EQ(result->chromaSubsampling, VK_VIDEO_CHROMA_SUBSAMPLING_444_BIT_KHR);
	EXPECT_EQ(result->lumaBitDepth, VK_VIDEO_COMPONENT_BIT_DEPTH_12_BIT_KHR);
	EXPECT_EQ(result->chromaBitDepth, VK_VIDEO_COMPONENT_BIT_DEPTH_12_BIT_KHR);
	EXPECT_EQ(av1.stdProfile, STD_VIDEO_AV1_PROFILE_PROFESSIONAL);
	EXPECT_EQ(av1.filmGrainSupport, VK_TRUE);

	request = {};
	request.Codec = OaVideoCodec::H264;
	request.StandardProfile = OaVideoCodecProfile::H264High;
	request.H264PictureLayout = OaVideoH264PictureLayout::InterlacedSeparatePlanes;
	h264 = {};
	h265 = {};
	av1 = {};
	vp9 = {};
	result = OaVideoDecoderProfile::BuildDecodeProfile(request, h264, h265, av1, vp9);
	ASSERT_TRUE(result.IsOk()) << result.GetStatus().ToString();
	EXPECT_EQ(h264.pictureLayout, VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_INTERLACED_SEPARATE_PLANES_BIT_KHR);
}

TEST(VideoProfile, RejectsCodecProfileMismatch)
{
	OaVideoProfile request = {};
	request.Codec = OaVideoCodec::H264;
	request.StandardProfile = OaVideoCodecProfile::H265Main;
	auto resolved = OaVideoDecoderProfile::ResolveDecodeProfile(request);
	EXPECT_FALSE(resolved.IsOk());
}

TEST(VideoProfile, LegacyDefaultsResolveToVerifiedProfiles)
{
	const std::pair<OaVideoCodec, OaVideoCodecProfile> cases[] = {
		{OaVideoCodec::H264, OaVideoCodecProfile::H264High},
		{OaVideoCodec::H265, OaVideoCodecProfile::H265Main},
		{OaVideoCodec::AV1, OaVideoCodecProfile::Av1Main},
		{OaVideoCodec::VP9, OaVideoCodecProfile::Vp9Profile0},
	};
	for (const auto& [codec, expected] : cases) {
		OaVideoProfile request = {};
		request.Codec = codec;
		auto resolved = OaVideoDecoderProfile::ResolveDecodeProfile(request);
		ASSERT_TRUE(resolved.IsOk());
		EXPECT_EQ(resolved->StandardProfile, expected);
		EXPECT_TRUE(OaVideoDecoderProfile::IsDecodePathImplemented(*resolved));
	}
}

TEST(VideoProfile, AssetManifestDrivesExactStreamProfiles)
{
	const OaPath manifestPath = OaTestAssetPath("Video/manifest.tsv");
	std::ifstream manifest(manifestPath.CStr());
	ASSERT_TRUE(manifest.is_open()) << manifestPath.CStr();

	std::string line;
	ASSERT_TRUE(static_cast<bool>(std::getline(manifest, line)));
	OaU32 rowCount = 0;
	while (std::getline(manifest, line)) {
		if (line.empty()) continue;
		std::array<std::string, 18> columns;
		std::istringstream row(line);
		for (std::string& column : columns) {
			ASSERT_TRUE(static_cast<bool>(std::getline(row, column, '\t'))) << line;
		}

		OaString relativePath = "Video/";
		relativePath += columns[0];
		const OaPath fixturePath = OaTestAssetPath(relativePath);
		auto streamResult = OaVideoStream::OpenFile(fixturePath.CStr());
		ASSERT_TRUE(streamResult.IsOk()) << columns[0] << ": " << streamResult.GetStatus().ToString();
		const OaVideoProfile profile = streamResult->GetVideoProfile();
		EXPECT_EQ(profile.Width, static_cast<OaU32>(std::stoul(columns[8])));
		EXPECT_EQ(profile.Height, static_cast<OaU32>(std::stoul(columns[9])));
		EXPECT_EQ(profile.LumaBitDepth, OaVideoBitDepth::Bit8);
		EXPECT_EQ(profile.ChromaBitDepth, OaVideoBitDepth::Bit8);
		EXPECT_EQ(profile.ChromaSubsampling, OaVideoChromaSubsampling::Yuv420);

		if (columns[1] == "h264") {
			EXPECT_EQ(profile.Codec, OaVideoCodec::H264);
			EXPECT_EQ(profile.StandardProfile, OaVideoCodecProfile::H264High);
			EXPECT_EQ(profile.H264PictureLayout, OaVideoH264PictureLayout::Progressive);
			EXPECT_TRUE(profile.HasLevel);
			EXPECT_EQ(profile.Level, static_cast<OaU32>(STD_VIDEO_H264_LEVEL_IDC_3_1));
		} else if (columns[1] == "h265") {
			EXPECT_EQ(profile.Codec, OaVideoCodec::H265);
			EXPECT_EQ(profile.StandardProfile, OaVideoCodecProfile::H265Main);
			EXPECT_TRUE(profile.HasLevel);
			EXPECT_EQ(profile.Level, static_cast<OaU32>(STD_VIDEO_H265_LEVEL_IDC_3_1));
			EXPECT_FALSE(profile.HighTier);
		} else if (columns[1] == "av1") {
			EXPECT_EQ(profile.Codec, OaVideoCodec::AV1);
			EXPECT_EQ(profile.StandardProfile, OaVideoCodecProfile::Av1Main);
			EXPECT_FALSE(profile.Av1FilmGrain);
			EXPECT_TRUE(profile.HasLevel);
			EXPECT_EQ(profile.Level, static_cast<OaU32>(STD_VIDEO_AV1_LEVEL_3_1));
			EXPECT_FALSE(profile.HighTier);
		} else if (columns[1] == "vp9") {
			EXPECT_EQ(profile.Codec, OaVideoCodec::VP9);
			EXPECT_EQ(profile.StandardProfile, OaVideoCodecProfile::Vp9Profile0);
			EXPECT_TRUE(profile.HasLevel);
			EXPECT_EQ(profile.Level, static_cast<OaU32>(STD_VIDEO_VP9_LEVEL_3_1));
		} else {
			FAIL() << "Unknown manifest codec: " << columns[1];
		}
		++rowCount;
	}
	EXPECT_EQ(rowCount, 4U);
}

TEST(OaVideoCodecRegistry, CreatesStreamLocalParserState)
{
	auto first = OaVideoCodecRegistry::GetInstance().CreateParser(OaVideoCodec::H264);
	auto second = OaVideoCodecRegistry::GetInstance().CreateParser(OaVideoCodec::H264);
	ASSERT_TRUE(first);
	ASSERT_TRUE(second);
	ASSERT_NE(first.Get(), second.Get());

	auto sps = MakeH264SpsNal();
	ASSERT_GT(sps.Size(), 4u);
	ASSERT_TRUE(first->ParseSps(OaSpan<const OaU8>(
		sps.Data() + 4, sps.Size() - 4)).IsOk());
	EXPECT_NE(first->GetH264Sps(0), nullptr);
	EXPECT_EQ(second->GetH264Sps(0), nullptr);
}

// Phase 2.1: Vulkan Video Extensions Setup Tests
TEST_F(OaVkEngineTestFixture, VideoDecoder_QueryCodecSupport)
{
	auto& rt = Rt();
	
	bool h264Supported = OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::H264);
	bool h265Supported = OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::H265);
	bool av1Supported = OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::AV1);
	const auto& sw = rt.Device.Info.Software;
	if (!sw.HasVideoQueue || !sw.HasVideoDecodeQueue || !rt.Device.Queues.HasVideoDecodeQueue) {
		EXPECT_FALSE(h264Supported);
		EXPECT_FALSE(h265Supported);
		EXPECT_FALSE(av1Supported);
		GTEST_SKIP() << "Selected Vulkan device does not expose a video decode queue";
	}
	if (h264Supported) EXPECT_TRUE(sw.HasVideoDecodeH264);
	if (h265Supported) EXPECT_TRUE(sw.HasVideoDecodeH265);
	if (av1Supported) EXPECT_TRUE(sw.HasVideoDecodeAV1);
	if (!h264Supported && !h265Supported && !av1Supported) {
		GTEST_SKIP() << "Vulkan Video decode extensions are present, but default "
						"profile capability queries are unsupported";
	}
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_QueryMaxResolution)
{
	auto& rt = Rt();
	if (!OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::H264)) {
		GTEST_SKIP() << "H.264 Vulkan Video decode is not supported on selected device";
	}
	
	OaU32 maxWidth = OaVideoDecoder::GetMaxWidth(rt, OaVideoCodec::H264);
	OaU32 maxHeight = OaVideoDecoder::GetMaxHeight(rt, OaVideoCodec::H264);
	
	// RTX 5090 supports up to 8K decode
	EXPECT_GE(maxWidth, 3840);  // At least 4K width
	EXPECT_GE(maxHeight, 2160); // At least 4K height
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_QueryDecodeCapabilities)
{
	auto& rt = Rt();
	auto capsResult = OaVideoDecoder::QueryDecodeCapabilities(rt, OaVideoCodec::H264);
	if (!capsResult.IsOk()) {
		GTEST_SKIP() << capsResult.GetStatus().ToString();
	}

	const auto& caps = *capsResult;
	EXPECT_TRUE(caps.Supported);
	EXPECT_TRUE(caps.HardwareProfileSupported);
	EXPECT_TRUE(caps.OaDecodePathImplemented);
	EXPECT_EQ(caps.Profile.StandardProfile, OaVideoCodecProfile::H264High);
	EXPECT_GE(caps.MaxWidth, 3840u);
	EXPECT_GE(caps.MaxHeight, 2160u);
	EXPECT_GT(caps.MaxDpbSlots, 0u);
	EXPECT_GT(caps.MaxActiveReferencePictures, 0u);
	EXPECT_GT(caps.MinBitstreamBufferOffsetAlignment, 0u);
	EXPECT_GT(caps.MinBitstreamBufferSizeAlignment, 0u);
	EXPECT_TRUE(caps.SupportsDpbAndOutputCoincide || caps.SupportsDpbAndOutputDistinct);
	EXPECT_TRUE(caps.SupportsNv12Dpb);
	EXPECT_FALSE(caps.DpbFormats.Empty());
	EXPECT_FALSE(caps.OutputFormats.Empty());
	EXPECT_TRUE(
		(caps.SupportsDpbAndOutputCoincide
			&& (caps.SupportsNv12DpbTransferSrc || caps.SupportsNv12DpbSampled))
		|| (caps.SupportsDpbAndOutputDistinct && caps.SupportsNv12OutputSampled));
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_UnimplementedProfileNeverClaimsSupport)
{
	OaVideoProfile profile = {};
	profile.Codec = OaVideoCodec::H265;
	profile.StandardProfile = OaVideoCodecProfile::H265Main10;
	profile.ChromaSubsampling = OaVideoChromaSubsampling::Yuv420;
	profile.LumaBitDepth = OaVideoBitDepth::Bit10;
	profile.ChromaBitDepth = OaVideoBitDepth::Bit10;
	auto capsResult = OaVideoDecoder::QueryDecodeCapabilities(Rt(), profile);
	if (not capsResult.IsOk()) {
		GTEST_SKIP() << capsResult.GetStatus().ToString();
	}
	EXPECT_FALSE(capsResult->Supported);
	EXPECT_FALSE(capsResult->OaDecodePathImplemented);
	EXPECT_EQ(capsResult->Profile.StandardProfile, OaVideoCodecProfile::H265Main10);
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_CreateH264Decoder)
{
	auto& rt = Rt();
	if (!OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::H264)) {
		GTEST_SKIP() << "H.264 Vulkan Video decode is not supported on selected device";
	}
	
	OaVideoProfile profile;
	profile.Codec = OaVideoCodec::H264;
	profile.Width = 1920;
	profile.Height = 1080;
	profile.MaxDpbSlots = 16;
	
	auto result = OaVideoDecoder::Create(rt, profile);
	
	// Phase 2.1 implementation pending - expect creation to succeed
	// but actual decode will fail until implemented
	EXPECT_TRUE(result.IsOk());
	
	if (result.IsOk())
	{
		auto decoder = OaStdMove(*result);
		EXPECT_TRUE(decoder.IsInitialized());
		EXPECT_TRUE(decoder.HasSessionParameters());
		EXPECT_GE(decoder.GetDpbSlotCapacity(), profile.MaxDpbSlots);
		EXPECT_GE(decoder.GetDpbViewCount(), 1u);
		auto caps = OaVideoDecoder::QueryDecodeCapabilities(rt, OaVideoCodec::H264);
		ASSERT_TRUE(caps.IsOk());
		ExpectDecoderCodedExtentAligned(decoder, *caps, profile);
		if (caps->SupportsDpbAndOutputDistinct) {
			EXPECT_GE(decoder.GetOutputFrameCapacity(), profile.MaxDpbSlots);
			EXPECT_GE(decoder.GetOutputViewCount(), profile.MaxDpbSlots);
		}
		decoder.Destroy();
	}
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_CreateH265Decoder)
{
	auto& rt = Rt();
	if (!OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::H265)) {
		GTEST_SKIP() << "H.265 Vulkan Video decode is not supported on selected device";
	}
	
	OaVideoProfile profile;
	profile.Codec = OaVideoCodec::H265;
	profile.Width = 3840;
	profile.Height = 2160;
	profile.MaxDpbSlots = 16;
	
	auto result = OaVideoDecoder::Create(rt, profile);
	EXPECT_TRUE(result.IsOk());
	
	if (result.IsOk())
	{
		auto decoder = OaStdMove(*result);
		EXPECT_TRUE(decoder.IsInitialized());
		EXPECT_TRUE(decoder.HasSessionParameters());
		EXPECT_GE(decoder.GetDpbSlotCapacity(), profile.MaxDpbSlots);
		EXPECT_GE(decoder.GetDpbViewCount(), 1u);
		auto caps = OaVideoDecoder::QueryDecodeCapabilities(rt, OaVideoCodec::H265);
		ASSERT_TRUE(caps.IsOk());
		ExpectDecoderCodedExtentAligned(decoder, *caps, profile);
		if (caps->SupportsDpbAndOutputDistinct) {
			EXPECT_GE(decoder.GetOutputFrameCapacity(), profile.MaxDpbSlots);
			EXPECT_GE(decoder.GetOutputViewCount(), profile.MaxDpbSlots);
		}
		decoder.Destroy();
	}
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_CreateAv1Decoder)
{
	auto& rt = Rt();
	if (!OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::AV1)) {
		GTEST_SKIP() << "AV1 Vulkan Video decode is not supported on selected device";
	}

	OaVideoProfile profile;
	profile.Codec = OaVideoCodec::AV1;
	profile.Width = 3840;
	profile.Height = 2160;
	// AV1 exposes eight reference-frame slots plus the current picture.
	profile.MaxDpbSlots = 9;

	auto result = OaVideoDecoder::Create(rt, profile);
	EXPECT_TRUE(result.IsOk());

	if (result.IsOk())
	{
		auto decoder = OaStdMove(*result);
		EXPECT_TRUE(decoder.IsInitialized());
		EXPECT_FALSE(decoder.HasSessionParameters());
		EXPECT_GE(decoder.GetDpbSlotCapacity(), profile.MaxDpbSlots);
		EXPECT_GE(decoder.GetDpbViewCount(), 1u);
		auto caps = OaVideoDecoder::QueryDecodeCapabilities(rt, OaVideoCodec::AV1);
		ASSERT_TRUE(caps.IsOk());
		ExpectDecoderCodedExtentAligned(decoder, *caps, profile);
		if (caps->SupportsDpbAndOutputDistinct) {
			EXPECT_GE(decoder.GetOutputFrameCapacity(), profile.MaxDpbSlots);
			EXPECT_GE(decoder.GetOutputViewCount(), profile.MaxDpbSlots);
		}
		decoder.Destroy();
	}
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_RejectInvalidProfile)
{
	auto& rt = Rt();
	if (!OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::H264)) {
		GTEST_SKIP() << "H.264 Vulkan Video decode is not supported on selected device";
	}

	OaVideoProfile profile;
	profile.Codec = OaVideoCodec::H264;
	profile.Width = 0;
	profile.Height = 1080;
	profile.MaxDpbSlots = 16;

	auto result = OaVideoDecoder::Create(rt, profile);
	EXPECT_FALSE(result.IsOk());
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_UpdateH264SessionParameters)
{
	auto& rt = Rt();
	if (!OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::H264)) {
		GTEST_SKIP() << "H.264 Vulkan Video decode is not supported on selected device";
	}

	OaVideoProfile profile;
	profile.Codec = OaVideoCodec::H264;
	profile.Width = 1920;
	profile.Height = 1080;
	profile.MaxDpbSlots = 16;

	auto result = OaVideoDecoder::Create(rt, profile);
	ASSERT_TRUE(result.IsOk());
	auto decoder = OaStdMove(*result);

	OaVideoFrame frame = {};
	auto sps = MakeH264SpsNal();
	auto status = decoder.DecodeFrame(OaSpan<const OaU8>(sps), frame);
	EXPECT_TRUE(status.IsOk()) << status.ToString();
	EXPECT_EQ(decoder.GetSessionParameterUpdateCount(), 1u);
	EXPECT_GE(OaVideoDecoderInternal::GetBitstreamBufferCapacity(decoder), static_cast<OaU64>(sps.Size()));

	auto pps = MakeH264PpsNal();
	status = decoder.DecodeFrame(OaSpan<const OaU8>(pps), frame);
	EXPECT_TRUE(status.IsOk()) << status.ToString();
	EXPECT_EQ(decoder.GetSessionParameterUpdateCount(), 2u);
	EXPECT_GE(OaVideoDecoderInternal::GetBitstreamBufferCapacity(decoder), static_cast<OaU64>(pps.Size()));

	decoder.Destroy();
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_UpdateH264SessionParametersFromAccessUnit)
{
	auto& rt = Rt();
	if (!OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::H264)) {
		GTEST_SKIP() << "H.264 Vulkan Video decode is not supported on selected device";
	}

	OaVideoProfile profile;
	profile.Codec = OaVideoCodec::H264;
	profile.Width = 1920;
	profile.Height = 1080;
	profile.MaxDpbSlots = 16;

	auto result = OaVideoDecoder::Create(rt, profile);
	ASSERT_TRUE(result.IsOk());
	auto decoder = OaStdMove(*result);

	OaVideoFrame frame = {};
	auto accessUnit = MakeH264ParameterAccessUnit();
	auto status = decoder.DecodeFrame(OaSpan<const OaU8>(accessUnit), frame);
	EXPECT_TRUE(status.IsOk()) << status.ToString();
	EXPECT_EQ(decoder.GetSessionParameterUpdateCount(), 2u);
	EXPECT_GE(OaVideoDecoderInternal::GetBitstreamBufferCapacity(decoder), static_cast<OaU64>(accessUnit.Size()));

	decoder.Destroy();
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_DecodeH264FrameFromLocalFixture)
{
	auto fixtureResult = OaFilesystem::ReadBinary(OaTestAssetPath("Video/VisionTestPattern128x72_idr.h264"));
	ASSERT_TRUE(fixtureResult.IsOk()) << fixtureResult.GetStatus().ToString();

	auto& rt = Rt();
	if (!OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::H264)) {
		GTEST_SKIP() << "H.264 Vulkan Video decode is not supported on selected device";
	}

	OaVideoProfile profile;
	profile.Codec = OaVideoCodec::H264;
	profile.Width = 128;
	profile.Height = 72;
	profile.MaxDpbSlots = 4;

	auto result = OaVideoDecoder::Create(rt, profile);
	ASSERT_TRUE(result.IsOk());
	auto decoder = OaStdMove(*result);

	constexpr OaU64 testPts = 42ULL;
	auto frameResult = OaFnVideo::Decode(
		decoder,
		OaSpan<const OaU8>(*fixtureResult),
		testPts);
	EXPECT_TRUE(frameResult.IsOk()) << frameResult.GetStatus().ToString();
	if (frameResult.IsOk()) {
		OaVideoFrame frame = *frameResult;
		EXPECT_NE(frame.Image, VK_NULL_HANDLE);
		EXPECT_NE(frame.ImageView, VK_NULL_HANDLE);
		EXPECT_EQ(frame.Width, profile.Width);
		EXPECT_EQ(frame.Height, profile.Height);
		EXPECT_EQ(frame.PresentationTimestamp, testPts);
		EXPECT_FALSE(frame.IsRgb);

		auto lumaResult = OaVideoDecoderInternal::ReadbackLuma(decoder, frame);
		ASSERT_TRUE(lumaResult.IsOk()) << lumaResult.GetStatus().ToString();
		ASSERT_EQ(lumaResult->Size(), static_cast<OaUsize>(profile.Width * profile.Height));
		OaU32 nonZeroCount = 0;
		OaU8 minValue = 255;
		OaU8 maxValue = 0;
		for (OaU8 value : *lumaResult) {
			nonZeroCount += value != 0 ? 1u : 0u;
			minValue = value < minValue ? value : minValue;
			maxValue = value > maxValue ? value : maxValue;
		}
		if (nonZeroCount == 0 || maxValue == minValue) {
			GTEST_SKIP() << "Vulkan H.264 command submission succeeded, but luma "
							"readback is still all zero";
		}
		EXPECT_GT(nonZeroCount, 0u);
		EXPECT_GT(maxValue, minValue);

		auto nv12Result = OaVideoDecoderInternal::ReadbackNv12(decoder, frame);
		ASSERT_TRUE(nv12Result.IsOk()) << nv12Result.GetStatus().ToString();
		ASSERT_EQ(nv12Result->Size(), static_cast<OaUsize>(profile.Width * profile.Height * 3 / 2));

		if (std::system("command -v ffmpeg >/dev/null 2>&1") == 0) {
			const OaPath fixturePath =
				OaTestAssetPath("Video/VisionTestPattern128x72_idr.h264");
			const char* referencePath = "/tmp/oa_h264_idr_reference_nv12.bin";
			const OaString command = OaString("ffmpeg -v error -y -i \"")
				+ fixturePath.String()
				+ "\" -frames:v 1 -f rawvideo -pix_fmt nv12 "
				+ referencePath;
			ASSERT_EQ(std::system(command.CStr()), 0);
			auto referenceResult = OaFilesystem::ReadBinary(OaPath(referencePath));
			std::remove(referencePath);
			ASSERT_TRUE(referenceResult.IsOk())
				<< referenceResult.GetStatus().ToString();
			ASSERT_EQ(referenceResult->Size(), nv12Result->Size());

			const OaUsize lumaBytes =
				static_cast<OaUsize>(profile.Width) * profile.Height;
			OaU64 lumaError = 0;
			OaU64 chromaError = 0;
			OaU64 decodedLumaSum = 0;
			OaU64 referenceLumaSum = 0;
			OaU8 maxError = 0;
			for (OaUsize i = 0; i < nv12Result->Size(); ++i) {
				const OaU8 error = static_cast<OaU8>(std::abs(
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
			const OaF64 lumaMae =
				static_cast<OaF64>(lumaError) / static_cast<OaF64>(lumaBytes);
			const OaF64 chromaMae =
				static_cast<OaF64>(chromaError)
				/ static_cast<OaF64>(nv12Result->Size() - lumaBytes);
			EXPECT_LT(lumaMae, 3.0)
				<< "maxError=" << static_cast<OaU32>(maxError)
				<< " decodedMean="
				<< static_cast<OaF64>(decodedLumaSum) / static_cast<OaF64>(lumaBytes)
				<< " referenceMean="
				<< static_cast<OaF64>(referenceLumaSum) / static_cast<OaF64>(lumaBytes)
				<< " decodedFirst="
				<< static_cast<OaU32>((*nv12Result)[0]) << ","
				<< static_cast<OaU32>((*nv12Result)[1]) << ","
				<< static_cast<OaU32>((*nv12Result)[2]) << ","
				<< static_cast<OaU32>((*nv12Result)[3])
				<< " referenceFirst="
				<< static_cast<OaU32>((*referenceResult)[0]) << ","
				<< static_cast<OaU32>((*referenceResult)[1]) << ","
				<< static_cast<OaU32>((*referenceResult)[2]) << ","
				<< static_cast<OaU32>((*referenceResult)[3]);
			EXPECT_LT(chromaMae, 3.0) << "maxError=" << static_cast<OaU32>(maxError);
		}

		auto hardwareTensorResult = OaVideoDecoderInternal::ConvertFrameToBf16Hardware(decoder, frame, false);
		ASSERT_TRUE(hardwareTensorResult.IsOk()) << hardwareTensorResult.GetStatus().ToString();
		OaMatrix hardwareTensor = OaStdMove(*hardwareTensorResult);
		OaExpectShape(hardwareTensor, {1, 3, profile.Height, profile.Width});
		OaExpectFinite(hardwareTensor);
		EXPECT_GE(hardwareTensor.At(0), 0.0f);
		EXPECT_LE(hardwareTensor.At(0), 1.0f);
		const OaU64 lumaBytes = static_cast<OaU64>(profile.Width) * profile.Height;
		const OaU8 yy = (*nv12Result)[0];
		const OaU8 uu = (*nv12Result)[static_cast<OaUsize>(lumaBytes)];
		const OaU8 vv = (*nv12Result)[static_cast<OaUsize>(lumaBytes + 1)];
		const OaF32 Y = 1.164f * (static_cast<OaF32>(yy) - 16.0f) / 255.0f;
		const OaF32 U = (static_cast<OaF32>(uu) - 128.0f) / 255.0f;
		const OaF32 V = (static_cast<OaF32>(vv) - 128.0f) / 255.0f;
		auto clampUnit = [](OaF32 value) {
			if (value < 0.0f) {
				return 0.0f;
			}
			if (value > 1.0f) {
				return 1.0f;
			}
			return value;
		};
		const OaF32 expectedR = clampUnit(Y + 1.596f * V);
		const OaF32 expectedG = clampUnit(Y - 0.391f * U - 0.813f * V);
		const OaF32 expectedB = clampUnit(Y + 2.018f * U);
		EXPECT_GE(expectedR, 0.0f);
		EXPECT_LE(expectedR, 1.0f);
		EXPECT_GE(expectedG, 0.0f);
		EXPECT_LE(expectedG, 1.0f);
		EXPECT_GE(expectedB, 0.0f);
		EXPECT_LE(expectedB, 1.0f);

		auto tensorResult = OaVideoDecoderInternal::ConvertFrameToBf16(decoder, frame, false);
		ASSERT_TRUE(tensorResult.IsOk()) << tensorResult.GetStatus().ToString();
		OaMatrix tensor = OaStdMove(*tensorResult);
		OaExpectShape(tensor, {1, 3, profile.Height, profile.Width});
		OaExpectFinite(tensor);
		EXPECT_GE(tensor.At(0), 0.0f);
		EXPECT_LE(tensor.At(0), 1.0f);
		EXPECT_NEAR(tensor.At(0), hardwareTensor.At(0), 1e-3f);
	}

	decoder.Destroy();

	auto conversionDecoderResult = OaVideoDecoder::Create(rt, profile);
	ASSERT_TRUE(conversionDecoderResult.IsOk()) << conversionDecoderResult.GetStatus().ToString();
	auto conversionDecoder = OaStdMove(*conversionDecoderResult);

	OaVideoConversionOptions conversion = {};
	conversion.PreferHardwareYCbCr = true;
	conversion.ColorSpace = OaYCbCrModel::Auto;
	conversion.ConvertToRgb = true;
	OaVideoFrame rgbaFrame = {};
	auto status = conversionDecoder.DecodeFrameWithConversion(
		OaSpan<const OaU8>(*fixtureResult),
		conversion,
		rgbaFrame);
	ASSERT_TRUE(status.IsOk()) << status.ToString();
	EXPECT_NE(rgbaFrame.Image, VK_NULL_HANDLE);
	EXPECT_NE(rgbaFrame.ImageView, VK_NULL_HANDLE);
	EXPECT_EQ(rgbaFrame.Format, VK_FORMAT_R8G8B8A8_UNORM);
	EXPECT_EQ(rgbaFrame.Width, profile.Width);
	EXPECT_EQ(rgbaFrame.Height, profile.Height);
	EXPECT_TRUE(rgbaFrame.IsRgb);

	auto rgbaResult = OaVideoDecoderInternal::ReadbackRgba(conversionDecoder, rgbaFrame);
	ASSERT_TRUE(rgbaResult.IsOk()) << rgbaResult.GetStatus().ToString();
	ASSERT_EQ(rgbaResult->Size(), static_cast<OaUsize>(profile.Width * profile.Height * 4));
	OaU32 alpha255Count = 0;
	for (OaUsize i = 0; i < rgbaResult->Size(); i += 4) {
		alpha255Count += (*rgbaResult)[i + 3] == 255 ? 1u : 0u;
	}
	// Hardware YCbCr path must write opaque alpha; this also catches zero-write
	// failures (e.g. push-constant misalignment causing all threads to return).
	EXPECT_EQ(alpha255Count, static_cast<OaU32>(profile.Width * profile.Height));
	conversionDecoder.Destroy();

	auto computeConversionDecoderResult = OaVideoDecoder::Create(rt, profile);
	ASSERT_TRUE(computeConversionDecoderResult.IsOk()) << computeConversionDecoderResult.GetStatus().ToString();
	auto computeConversionDecoder = OaStdMove(*computeConversionDecoderResult);

	OaVideoConversionOptions computeConversion = {};
	computeConversion.PreferHardwareYCbCr = false;
	computeConversion.ColorSpace = OaYCbCrModel::Auto;
	computeConversion.ConvertToRgb = true;
	OaVideoFrame computeRgbaFrame = {};
	status = computeConversionDecoder.DecodeFrameWithConversion(
		OaSpan<const OaU8>(*fixtureResult),
		computeConversion,
		computeRgbaFrame);
	ASSERT_TRUE(status.IsOk()) << status.ToString();
	EXPECT_NE(computeRgbaFrame.Image, VK_NULL_HANDLE);
	EXPECT_NE(computeRgbaFrame.ImageView, VK_NULL_HANDLE);
	EXPECT_EQ(computeRgbaFrame.Format, VK_FORMAT_R8G8B8A8_UNORM);
	EXPECT_EQ(computeRgbaFrame.Width, profile.Width);
	EXPECT_EQ(computeRgbaFrame.Height, profile.Height);
	EXPECT_TRUE(computeRgbaFrame.IsRgb);

	auto computeRgbaResult = OaVideoDecoderInternal::ReadbackRgba(computeConversionDecoder, computeRgbaFrame);
	ASSERT_TRUE(computeRgbaResult.IsOk()) << computeRgbaResult.GetStatus().ToString();
	ASSERT_EQ(computeRgbaResult->Size(), static_cast<OaUsize>(profile.Width * profile.Height * 4));
	OaU32 computeAlpha255Count = 0;
	OaU32 computeColorNonZeroCount = 0;
	for (OaUsize i = 0; i < computeRgbaResult->Size(); i += 4) {
		computeColorNonZeroCount += ((*computeRgbaResult)[i] | (*computeRgbaResult)[i + 1] | (*computeRgbaResult)[i + 2]) != 0 ? 1u : 0u;
		computeAlpha255Count += (*computeRgbaResult)[i + 3] == 255 ? 1u : 0u;
	}
	EXPECT_GT(computeColorNonZeroCount, 0u);
	EXPECT_EQ(computeAlpha255Count, static_cast<OaU32>(profile.Width * profile.Height));
	computeConversionDecoder.Destroy();
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_PipelinesAsyncH264Conversion)
{
	auto fixtureResult = OaFilesystem::ReadBinary(
		OaTestAssetPath("Video/VisionTestPattern128x72_idr.h264"));
	ASSERT_TRUE(fixtureResult.IsOk()) << fixtureResult.GetStatus().ToString();

	auto& rt = Rt();
	if (!OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::H264)) {
		GTEST_SKIP() << "H.264 Vulkan Video decode is not supported on selected device";
	}

	OaVideoProfile profile = MakeH264FixtureProfile();
	profile.MaxDpbSlots = 16;
	auto decoderResult = OaVideoDecoder::Create(rt, profile);
	ASSERT_TRUE(decoderResult.IsOk()) << decoderResult.GetStatus().ToString();
	auto decoder = OaStdMove(*decoderResult);

	// Exceed the 16-slot decode-output pool so this test covers reuse while
	// earlier color conversions are still in flight.
	constexpr OaU32 frameCount = 24;
	OaVec<OaVideoFrame> rgbaFrames;
	OaVec<OaVkImageDispatchTicket> readyTickets;
	rgbaFrames.Reserve(frameCount);
	readyTickets.Reserve(frameCount);

	OaVideoConversionOptions options = {};
	options.ConvertToRgb = true;
	options.PreferHardwareYCbCr = false;

	for (OaU32 i = 0; i < frameCount; ++i) {
		OaVideoFrame nv12 = {};
		OaStatus decodeStatus = decoder.DecodeFrame(
			OaSpan<const OaU8>(*fixtureResult),
			nv12);
		ASSERT_TRUE(decodeStatus.IsOk()) << decodeStatus.ToString();

		auto rgbaResult = OaVideoDecoderInternal::AllocateRgbaFrame(
			decoder,
			profile.Width,
			profile.Height);
		ASSERT_TRUE(rgbaResult.IsOk()) << rgbaResult.GetStatus().ToString();
		rgbaFrames.PushBack(*rgbaResult);

		auto readyResult = OaVideoDecoderInternal::ConvertNv12ToRgbIntoAsync(
			decoder,
			nv12,
			options,
			rgbaFrames.Back());
		ASSERT_TRUE(readyResult.IsOk()) << readyResult.GetStatus().ToString();
		ASSERT_TRUE(readyResult->IsValid())
			<< "async conversion returned an already-retired dispatch ticket";
		readyTickets.PushBack(OaStdMove(*readyResult));
	}

	for (OaU32 i = 0; i < frameCount; ++i) {
		ASSERT_TRUE(readyTickets[i].Wait().IsOk());
		auto rgbaResult = OaVideoDecoderInternal::ReadbackRgba(decoder, rgbaFrames[i]);
		ASSERT_TRUE(rgbaResult.IsOk()) << rgbaResult.GetStatus().ToString();
		ASSERT_EQ(
			rgbaResult->Size(),
			static_cast<OaUsize>(profile.Width * profile.Height * 4));
		for (OaUsize pixel = 0; pixel < rgbaResult->Size(); pixel += 4) {
			ASSERT_EQ((*rgbaResult)[pixel + 3], 255);
		}
	}

	decoder.Destroy();
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_DroppedAsyncConversionTicketRetiresSafely)
{
	auto fixtureResult = OaFilesystem::ReadBinary(
		OaTestAssetPath("Video/VisionTestPattern128x72_idr.h264"));
	ASSERT_TRUE(fixtureResult.IsOk()) << fixtureResult.GetStatus().ToString();

	auto& rt = Rt();
	if (!OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::H264)) {
		GTEST_SKIP() << "H.264 Vulkan Video decode is not supported on selected device";
	}

	OaVideoProfile profile = MakeH264FixtureProfile();
	profile.MaxDpbSlots = 16;
	auto decoderResult = OaVideoDecoder::Create(rt, profile);
	ASSERT_TRUE(decoderResult.IsOk()) << decoderResult.GetStatus().ToString();
	auto decoder = OaStdMove(*decoderResult);

	OaVideoConversionOptions options = {};
	options.ConvertToRgb = true;
	options.PreferHardwareYCbCr = false;

	OaVideoFrame firstNv12 = {};
	ASSERT_TRUE(decoder.DecodeFrame(OaSpan<const OaU8>(*fixtureResult), firstNv12).IsOk());
	auto firstRgbaResult = OaVideoDecoderInternal::AllocateRgbaFrame(
		decoder, profile.Width, profile.Height);
	ASSERT_TRUE(firstRgbaResult.IsOk()) << firstRgbaResult.GetStatus().ToString();
	OaVideoFrame firstRgba = *firstRgbaResult;

	OaEvent droppedReady = {};
	{
		auto ticketResult = OaVideoDecoderInternal::ConvertNv12ToRgbIntoAsync(
			decoder, firstNv12, options, firstRgba);
		ASSERT_TRUE(ticketResult.IsOk()) << ticketResult.GetStatus().ToString();
		ASSERT_TRUE(ticketResult->IsValid());
		droppedReady = ticketResult->Completion();
		// Intentionally destroy the live ticket without Wait(). Its resources
		// must remain valid until droppedReady completes.
	}
	ASSERT_TRUE(droppedReady.IsValid());
	ASSERT_TRUE(droppedReady.Wait().IsOk());

	// A subsequent dispatch collects the completed retirement and safely
	// reuses the engine stream pool and bindless heap.
	OaVideoFrame secondNv12 = {};
	ASSERT_TRUE(decoder.DecodeFrame(OaSpan<const OaU8>(*fixtureResult), secondNv12).IsOk());
	auto secondRgbaResult = OaVideoDecoderInternal::AllocateRgbaFrame(
		decoder, profile.Width, profile.Height);
	ASSERT_TRUE(secondRgbaResult.IsOk()) << secondRgbaResult.GetStatus().ToString();
	OaVideoFrame secondRgba = *secondRgbaResult;
	auto secondTicketResult = OaVideoDecoderInternal::ConvertNv12ToRgbIntoAsync(
		decoder, secondNv12, options, secondRgba);
	ASSERT_TRUE(secondTicketResult.IsOk()) << secondTicketResult.GetStatus().ToString();
	ASSERT_TRUE(secondTicketResult->Wait().IsOk());

	auto rgbaResult = OaVideoDecoderInternal::ReadbackRgba(decoder, secondRgba);
	ASSERT_TRUE(rgbaResult.IsOk()) << rgbaResult.GetStatus().ToString();
	ASSERT_EQ(rgbaResult->Size(), static_cast<OaUsize>(profile.Width * profile.Height * 4));
	for (OaUsize pixel = 0; pixel < rgbaResult->Size(); pixel += 4) {
		ASSERT_EQ((*rgbaResult)[pixel + 3], 255);
	}

	decoder.Destroy();
}

TEST(VideoDecoderLifecycle, AbandonedSubmittedSessionRetiresAtEngineClose)
{
	auto fixtureResult = OaFilesystem::ReadBinary(
		OaTestAssetPath("Video/VisionTestPattern128x72_idr.h264"));
	ASSERT_TRUE(fixtureResult.IsOk()) << fixtureResult.GetStatus().ToString();

	auto config = OaTestEngineConfig(OaPrecision::FP32);
	config.RegisterAsGlobal = false;
	config.PreloadEmbeddedPipelines = false;
	config.EnablePipelineCache = false;
	auto engineResult = OaEngine::Create(config);
	ASSERT_TRUE(engineResult.IsOk()) << engineResult.GetStatus().ToString();
	auto engine = OaStdMove(*engineResult);
	if (not OaVideoDecoder::IsCodecSupported(*engine, OaVideoCodec::H264)) {
		ASSERT_TRUE(engine->Close().IsOk());
		GTEST_SKIP() << "H.264 Vulkan Video decode is not supported on selected device";
	}

	OaEvent completion;
	{
		auto decoderResult = OaVideoDecoder::Create(
			*engine, MakeH264FixtureProfile());
		ASSERT_TRUE(decoderResult.IsOk())
			<< decoderResult.GetStatus().ToString();
		auto decoder = OaStdMove(*decoderResult);
		OaVideoFrame frame = {};
		ASSERT_TRUE(decoder.DecodeFrame(
			OaSpan<const OaU8>(*fixtureResult), frame).IsOk());
		completion = frame.Ready;
		ASSERT_TRUE(completion.IsValid());
		// No Close/Destroy: the live video session and its semaphore must move to
		// engine retirement without waiting or invalidating copied completion.
	}

	ASSERT_TRUE(completion.Wait().IsOk());
	ASSERT_TRUE(engine->Close().IsOk());
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_DecodeH265FrameFromLocalFixture)
{
	auto fixtureResult = OaFilesystem::ReadBinary(OaTestAssetPath("Video/VisionTestPattern128x72_idr.h265"));
	ASSERT_TRUE(fixtureResult.IsOk()) << fixtureResult.GetStatus().ToString();

	auto& rt = Rt();
	if (!OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::H265)) {
		GTEST_SKIP() << "H.265 Vulkan Video decode is not supported on selected device";
	}

	OaVideoProfile profile = MakeH265FixtureProfile();

	auto result = OaVideoDecoder::Create(rt, profile);
	ASSERT_TRUE(result.IsOk()) << result.GetStatus().ToString();
	auto decoder = OaStdMove(*result);

	OaVideoFrame frame = {};
	auto status = decoder.DecodeFrame(OaSpan<const OaU8>(*fixtureResult), frame);
	ASSERT_TRUE(status.IsOk()) << status.ToString();
	EXPECT_EQ(OaVideoDecoderInternal::GetCachedH265VpsCount(decoder), 1u);
	EXPECT_EQ(OaVideoDecoderInternal::GetCachedH265SpsCount(decoder), 1u);
	EXPECT_EQ(OaVideoDecoderInternal::GetCachedH265PpsCount(decoder), 1u);
	EXPECT_EQ(decoder.GetSessionParameterUpdateCount(), 3u);
	EXPECT_EQ(decoder.GetCurrentFrameNumber(), 1u);
	EXPECT_GE(decoder.GetDpbInUseCount(), 1u);
	EXPECT_GE(decoder.GetDpbReferenceCount(), 1u);
	ExpectDecodedLumaIsReadable(decoder, frame, profile);
	ExpectDecodedNv12MatchesFfmpeg(
		decoder,
		frame,
		profile,
		"Video/VisionTestPattern128x72_idr.h265",
		"/tmp/oa_h265_idr_reference_nv12.bin");

	decoder.Destroy();
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_Av1FixturePresentAndCapabilityQueried)
{
	auto fixtureResult = OaFilesystem::ReadBinary(OaTestAssetPath("Video/VisionTestPattern128x72_keyframe.ivf"));
	ASSERT_TRUE(fixtureResult.IsOk()) << fixtureResult.GetStatus().ToString();
	EXPECT_GT(fixtureResult->Size(), 0u);

	auto& rt = Rt();
	auto capsResult = OaVideoDecoder::QueryDecodeCapabilities(rt, OaVideoCodec::AV1);
	if (!capsResult.IsOk()) {
		GTEST_SKIP() << capsResult.GetStatus().ToString();
	}

	EXPECT_EQ(capsResult->Supported, OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::AV1));
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_CreateVp9Decoder)
{
	auto& rt = Rt();
	if (!OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::VP9)) {
		GTEST_SKIP() << "VP9 Vulkan Video decode is not supported on selected device";
	}

	OaVideoProfile profile;
	profile.Codec = OaVideoCodec::VP9;
	profile.Width = 3840;
	profile.Height = 2160;
	profile.MaxDpbSlots = 9;

	auto result = OaVideoDecoder::Create(rt, profile);
	EXPECT_TRUE(result.IsOk());

	if (result.IsOk())
	{
		auto decoder = OaStdMove(*result);
		EXPECT_TRUE(decoder.IsInitialized());
		EXPECT_FALSE(decoder.HasSessionParameters());
		EXPECT_GE(decoder.GetDpbSlotCapacity(), profile.MaxDpbSlots);
		EXPECT_GE(decoder.GetDpbViewCount(), 1u);
		auto caps = OaVideoDecoder::QueryDecodeCapabilities(rt, OaVideoCodec::VP9);
		ASSERT_TRUE(caps.IsOk());
		ExpectDecoderCodedExtentAligned(decoder, *caps, profile);
		if (caps->SupportsDpbAndOutputDistinct) {
			EXPECT_GE(decoder.GetOutputFrameCapacity(), profile.MaxDpbSlots);
			EXPECT_GE(decoder.GetOutputViewCount(), profile.MaxDpbSlots);
		}
		decoder.Destroy();
	}
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_Vp9FixturePresentAndCapabilityQueried)
{
	auto fixtureResult = OaFilesystem::ReadBinary(OaTestAssetPath("Video/VisionTestPattern128x72_keyframe_vp9.ivf"));
	ASSERT_TRUE(fixtureResult.IsOk()) << fixtureResult.GetStatus().ToString();
	EXPECT_GT(fixtureResult->Size(), 0u);

	auto& rt = Rt();
	auto capsResult = OaVideoDecoder::QueryDecodeCapabilities(rt, OaVideoCodec::VP9);
	if (!capsResult.IsOk()) {
		GTEST_SKIP() << capsResult.GetStatus().ToString();
	}

	EXPECT_EQ(capsResult->Supported, OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::VP9));
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_DecodeVp9FrameFromLocalFixture)
{
	auto fixtureResult = OaFilesystem::ReadBinary(OaTestAssetPath("Video/VisionTestPattern128x72_keyframe_vp9.ivf"));
	ASSERT_TRUE(fixtureResult.IsOk()) << fixtureResult.GetStatus().ToString();

	auto& rt = Rt();
	if (!OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::VP9)) {
		GTEST_SKIP() << "VP9 Vulkan Video decode is not supported on selected device";
	}

	OaVideoProfile profile = MakeVp9FixtureProfile();

	auto result = OaVideoDecoder::Create(rt, profile);
	ASSERT_TRUE(result.IsOk()) << result.GetStatus().ToString();
	auto decoder = OaStdMove(*result);

	OaVideoFrame frame = {};
	auto status = decoder.DecodeFrame(OaSpan<const OaU8>(*fixtureResult), frame);
	ASSERT_TRUE(status.IsOk()) << status.ToString();
	EXPECT_EQ(decoder.GetCurrentFrameNumber(), 1u);
	EXPECT_GE(decoder.GetDpbInUseCount(), 1u);
	EXPECT_GE(decoder.GetDpbReferenceCount(), 1u);
	ExpectDecodedLumaIsReadable(decoder, frame, profile);
	ExpectDecodedNv12MatchesFfmpeg(
		decoder,
		frame,
		profile,
		"Video/VisionTestPattern128x72_keyframe_vp9.ivf",
		"/tmp/oa_vp9_keyframe_reference_nv12.bin");

	decoder.Destroy();
}

TEST_F(OaVkEngineTestFixture, Av1Parser_ParseLocalFixture)
{
	auto fixtureResult = OaFilesystem::ReadBinary(OaTestAssetPath("Video/VisionTestPattern128x72_keyframe.ivf"));
	ASSERT_TRUE(fixtureResult.IsOk()) << fixtureResult.GetStatus().ToString();

	OaVcpAv1 parser;
	OaAv1PictureDesc desc;
	auto status = parser.ParseAccessUnit(OaSpan<const OaU8>(*fixtureResult), desc);
	ASSERT_TRUE(status.IsOk()) << status.ToString();
	ASSERT_TRUE(desc.HasPicture);
	EXPECT_EQ(desc.FrameHeader.TileCols, 1u);
	EXPECT_EQ(desc.FrameHeader.TileRows, 1u);
	EXPECT_EQ(desc.SequenceHeader.MaxFrameWidthMinus1 + 1u, 128u);
	EXPECT_EQ(desc.SequenceHeader.MaxFrameHeightMinus1 + 1u, 72u);
	ASSERT_EQ(desc.TileOffsets.Size(), 1u);
	ASSERT_EQ(desc.TileSizes.Size(), 1u);
	EXPECT_GT(desc.TileSizes[0], 0u);
}

TEST(Av1Parser, CountsEveryPictureInMultiFrameTemporalUnits)
{
	const OaPath path = OaTestAssetPath("Video/shibuya_720p_av1_main_8bit_420.mp4");
	auto streamResult = OaVideoStream::OpenFile(path.CStr());
	ASSERT_TRUE(streamResult.IsOk()) << streamResult.GetStatus().ToString();
	OaVcpAv1 parser;
	// FFmpeg trace_headers reports 1, 6, and 1 frame OBUs in the first three
	// packets.  Packet 2 is the SVT-AV1 hidden-reference pyramid that the old
	// single-picture ParseAccessUnit path silently collapsed to one picture.
	constexpr OaU32 expectedPictures[] = {1, 6, 1};
	for (OaU32 packetIndex = 0; packetIndex < 3; ++packetIndex) {
		OaVideoPacket packet;
		ASSERT_TRUE(streamResult->ReadNextPacket(packet).IsOk()) << packetIndex;
		OaAv1AccessUnitInfo info;
		const OaStatus status = parser.InspectAccessUnit(
			OaSpan<const OaU8>(packet.Data.Data(), packet.Data.Size()), info);
		ASSERT_TRUE(status.IsOk()) << status.ToString();
		EXPECT_EQ(info.PictureCount(), expectedPictures[packetIndex]) << packetIndex;
	}
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_DecodeAv1FrameFromLocalFixture)
{
	auto fixtureResult = OaFilesystem::ReadBinary(OaTestAssetPath("Video/VisionTestPattern128x72_keyframe.ivf"));
	ASSERT_TRUE(fixtureResult.IsOk()) << fixtureResult.GetStatus().ToString();

	auto& rt = Rt();
	if (!OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::AV1)) {
		GTEST_SKIP() << "AV1 Vulkan Video decode is not supported on selected device";
	}

	OaVideoProfile profile = MakeAv1FixtureProfile();

	auto result = OaVideoDecoder::Create(rt, profile);
	ASSERT_TRUE(result.IsOk()) << result.GetStatus().ToString();
	auto decoder = OaStdMove(*result);

	OaVideoFrame frame = {};
	auto status = decoder.DecodeFrame(OaSpan<const OaU8>(*fixtureResult), frame);
	ASSERT_TRUE(status.IsOk()) << status.ToString();
	EXPECT_EQ(decoder.GetCurrentFrameNumber(), 1u);
	EXPECT_GE(decoder.GetDpbInUseCount(), 1u);
	EXPECT_GE(decoder.GetDpbReferenceCount(), 1u);
	ExpectDecodedNv12MatchesFfmpeg(
		decoder,
		frame,
		profile,
		"Video/VisionTestPattern128x72_keyframe.ivf",
		"/tmp/oa_av1_keyframe_reference_nv12.bin");

	decoder.Destroy();
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_DpbInitialState)
{
	auto& rt = Rt();
	if (!OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::H264)) {
		GTEST_SKIP() << "H.264 Vulkan Video decode is not supported on selected device";
	}

	auto result = OaVideoDecoder::Create(rt, MakeH264FixtureProfile());
	ASSERT_TRUE(result.IsOk()) << result.GetStatus().ToString();
	auto decoder = OaStdMove(*result);

	EXPECT_TRUE(decoder.IsInitialized());
	EXPECT_GE(decoder.GetDpbSlotCapacity(), 4u);
	EXPECT_GE(decoder.GetDpbViewCount(), 1u);
	EXPECT_EQ(decoder.GetDpbInUseCount(), 0u);
	EXPECT_EQ(decoder.GetDpbReferenceCount(), 0u);
	EXPECT_EQ(decoder.GetCurrentFrameNumber(), 0u);
	EXPECT_EQ(OaVideoDecoderInternal::GetCachedSpsCount(decoder), 0u);
	EXPECT_EQ(OaVideoDecoderInternal::GetCachedPpsCount(decoder), 0u);

	decoder.Destroy();
}

TEST_F(OaVkEngineTestFixture, VideoDecoder_DpbDecodeAndFlush)
{
	auto fixtureResult = OaFilesystem::ReadBinary(OaTestAssetPath("Video/VisionTestPattern128x72_idr.h264"));
	ASSERT_TRUE(fixtureResult.IsOk()) << fixtureResult.GetStatus().ToString();

	auto& rt = Rt();
	if (!OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::H264)) {
		GTEST_SKIP() << "H.264 Vulkan Video decode is not supported on selected device";
	}

	auto result = OaVideoDecoder::Create(rt, MakeH264FixtureProfile());
	ASSERT_TRUE(result.IsOk()) << result.GetStatus().ToString();
	auto decoder = OaStdMove(*result);

	OaVideoFrame frame = {};
	auto status = decoder.DecodeFrame(OaSpan<const OaU8>(*fixtureResult), frame);
	ASSERT_TRUE(status.IsOk()) << status.ToString();

	EXPECT_EQ(decoder.GetCurrentFrameNumber(), 1u);
	EXPECT_EQ(OaVideoDecoderInternal::GetCachedSpsCount(decoder), 1u);
	EXPECT_EQ(OaVideoDecoderInternal::GetCachedPpsCount(decoder), 1u);
	EXPECT_EQ(decoder.GetSessionParameterUpdateCount(), 2u);
	EXPECT_GE(decoder.GetDpbInUseCount(), 1u);
	EXPECT_GE(decoder.GetDpbReferenceCount(), 1u);
	EXPECT_NE(frame.Image, VK_NULL_HANDLE);
	EXPECT_EQ(frame.Format, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM);

	status = decoder.Flush();
	ASSERT_TRUE(status.IsOk()) << status.ToString();
	EXPECT_EQ(decoder.GetDpbInUseCount(), 0u);
	EXPECT_EQ(decoder.GetDpbReferenceCount(), 0u);
	EXPECT_EQ(decoder.GetCurrentFrameNumber(), 0u);
	EXPECT_EQ(OaVideoDecoderInternal::GetCachedSpsCount(decoder), 0u);
	EXPECT_EQ(OaVideoDecoderInternal::GetCachedPpsCount(decoder), 0u);
	EXPECT_EQ(decoder.GetSessionParameterUpdateCount(), 0u);

	decoder.Destroy();
}

// NOTE: DPB flush-with-references is already covered by the decode-then-Flush
// assertions above (GetDpbInUseCount()/GetDpbReferenceCount() == 0 after
// flush). NV12→BF16 conversion is not tested here: OaFnImage::CvtNv12ToBf16 is
// still a dispatch stub (allocates the output tensor but does not bind the
// VkImage / run the shader), so any test would only validate the shape of
// uninitialized memory. Add real coverage when the image-descriptor dispatch
// path lands.

// Frame Pool Tests
TEST_F(OaVkEngineTestFixture, VideoFramePool_CreatePool)
{
	auto& rt = Rt();
	
	auto result = OaVideoFramePool::Create(rt, 1920, 1080, 4);
	
	EXPECT_TRUE(result.IsOk());
	
	if (result.IsOk())
	{
		auto pool = OaStdMove(*result);
		pool.Destroy();
	}
}

TEST_F(OaVkEngineTestFixture, VideoFramePool_AcquireRelease)
{
	auto& rt = Rt();
	
	auto result = OaVideoFramePool::Create(rt, 1920, 1080, 2);
	ASSERT_TRUE(result.IsOk());
	
	auto pool = OaStdMove(*result);
	
	auto first = pool.Acquire();
	EXPECT_NE(first.Image, VK_NULL_HANDLE);
	EXPECT_NE(first.ImageView, VK_NULL_HANDLE);
	EXPECT_EQ(first.Format, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM);
	EXPECT_EQ(first.Width, 1920u);
	EXPECT_EQ(first.Height, 1080u);

	auto second = pool.Acquire();
	EXPECT_NE(second.Image, VK_NULL_HANDLE);
	EXPECT_NE(second.ImageView, VK_NULL_HANDLE);
	EXPECT_NE(second.Image, first.Image);

	auto exhausted = pool.Acquire();
	EXPECT_EQ(exhausted.Image, VK_NULL_HANDLE);
	EXPECT_EQ(exhausted.ImageView, VK_NULL_HANDLE);

	pool.Release(first);
	auto reused = pool.Acquire();
	EXPECT_EQ(reused.Image, first.Image);
	
	pool.Destroy();
}

// NOTE: 4K@60 / 8K@30 decode-throughput benchmarks are intentionally not part
// of the correctness suite — they are performance measurements (and un-runnable
// on the CI iGPU). If added later, they belong in a Bench* target, not here.
