// Shared exact VkVideoProfileInfoKHR builder for OaVideoDecoder.

#include "VideoDecoderProfile.h"

namespace {

bool ProfileMatchesCodec(
	OaVideoCodec InCodec, OaVideoCodecProfile InProfile)
{
	switch (InProfile) {
	case OaVideoCodecProfile::H264Baseline:
	case OaVideoCodecProfile::H264Main:
	case OaVideoCodecProfile::H264High:
	case OaVideoCodecProfile::H264High444Predictive: return InCodec == OaVideoCodec::H264;
	case OaVideoCodecProfile::H265Main:
	case OaVideoCodecProfile::H265Main10:
	case OaVideoCodecProfile::H265MainStillPicture:
	case OaVideoCodecProfile::H265FormatRangeExtensions:
	case OaVideoCodecProfile::H265ScreenContentCodingExtensions: return InCodec == OaVideoCodec::H265;
	case OaVideoCodecProfile::Av1Main:
	case OaVideoCodecProfile::Av1High:
	case OaVideoCodecProfile::Av1Professional: return InCodec == OaVideoCodec::AV1;
	case OaVideoCodecProfile::Vp9Profile0:
	case OaVideoCodecProfile::Vp9Profile1:
	case OaVideoCodecProfile::Vp9Profile2:
	case OaVideoCodecProfile::Vp9Profile3: return InCodec == OaVideoCodec::VP9;
	case OaVideoCodecProfile::Unspecified: return true;
	}
	return false;
}

OaVideoCodecProfile DefaultProfile(OaVideoCodec InCodec)
{
	switch (InCodec) {
	case OaVideoCodec::H264: return OaVideoCodecProfile::H264High;
	case OaVideoCodec::H265: return OaVideoCodecProfile::H265Main;
	case OaVideoCodec::AV1: return OaVideoCodecProfile::Av1Main;
	case OaVideoCodec::VP9: return OaVideoCodecProfile::Vp9Profile0;
	}
	return OaVideoCodecProfile::Unspecified;
}

OaResult<VkVideoChromaSubsamplingFlagsKHR> ToVkChroma(OaVideoChromaSubsampling InChroma)
{
	switch (InChroma) {
	case OaVideoChromaSubsampling::Monochrome: return VK_VIDEO_CHROMA_SUBSAMPLING_MONOCHROME_BIT_KHR;
	case OaVideoChromaSubsampling::Yuv420: return VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
	case OaVideoChromaSubsampling::Yuv422: return VK_VIDEO_CHROMA_SUBSAMPLING_422_BIT_KHR;
	case OaVideoChromaSubsampling::Yuv444: return VK_VIDEO_CHROMA_SUBSAMPLING_444_BIT_KHR;
	}
	return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid video chroma-subsampling value");
}

OaResult<VkVideoComponentBitDepthFlagsKHR> ToVkBitDepth(OaVideoBitDepth InDepth)
{
	switch (InDepth) {
	case OaVideoBitDepth::Bit8: return VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
	case OaVideoBitDepth::Bit10: return VK_VIDEO_COMPONENT_BIT_DEPTH_10_BIT_KHR;
	case OaVideoBitDepth::Bit12: return VK_VIDEO_COMPONENT_BIT_DEPTH_12_BIT_KHR;
	}
	return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid video component bit-depth value");
}

} // namespace

namespace OaVideoDecoderProfile {

OaResult<OaVideoProfile> ResolveDecodeProfile(const OaVideoProfile& InProfile)
{
	OaVideoProfile profile = InProfile;
	if (profile.StandardProfile == OaVideoCodecProfile::Unspecified) {
		profile.StandardProfile = DefaultProfile(profile.Codec);
	}
	if (profile.StandardProfile == OaVideoCodecProfile::Unspecified or
		not ProfileMatchesCodec(profile.Codec, profile.StandardProfile)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
							   "Video standard profile does not match the requested codec");
	}
	if (profile.Av1FilmGrain and profile.Codec != OaVideoCodec::AV1) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
							   "Film-grain support is only valid for AV1 decode profiles");
	}
	auto chromaResult = ToVkChroma(profile.ChromaSubsampling);
	if (not chromaResult.IsOk()) return chromaResult.GetStatus();
	auto lumaDepthResult = ToVkBitDepth(profile.LumaBitDepth);
	if (not lumaDepthResult.IsOk()) return lumaDepthResult.GetStatus();
	auto chromaDepthResult = ToVkBitDepth(profile.ChromaBitDepth);
	if (not chromaDepthResult.IsOk()) return chromaDepthResult.GetStatus();
	return profile;
}

bool IsDecodePathImplemented(const OaVideoProfile& InProfile)
{
	if (InProfile.ChromaSubsampling != OaVideoChromaSubsampling::Yuv420 or
		InProfile.LumaBitDepth != OaVideoBitDepth::Bit8 or InProfile.ChromaBitDepth != OaVideoBitDepth::Bit8) {
		return false;
	}
	switch (InProfile.StandardProfile) {
	case OaVideoCodecProfile::H264High:
		return InProfile.Codec == OaVideoCodec::H264 and
			   InProfile.H264PictureLayout == OaVideoH264PictureLayout::Progressive;
	case OaVideoCodecProfile::H265Main: return InProfile.Codec == OaVideoCodec::H265;
	case OaVideoCodecProfile::Av1Main: return InProfile.Codec == OaVideoCodec::AV1 and not InProfile.Av1FilmGrain;
	case OaVideoCodecProfile::Vp9Profile0: return InProfile.Codec == OaVideoCodec::VP9;
	default: return false;
	}
}

OaResult<VkVideoProfileInfoKHR> BuildDecodeProfile(const OaVideoProfile& InProfile,
	VkVideoDecodeH264ProfileInfoKHR& OutH264,
	VkVideoDecodeH265ProfileInfoKHR& OutH265,
	VkVideoDecodeAV1ProfileInfoKHR& OutAV1,
	VkVideoDecodeVP9ProfileInfoKHR& OutVp9)
{
	auto resolvedResult = ResolveDecodeProfile(InProfile);
	if (not resolvedResult.IsOk()) {
		return resolvedResult.GetStatus();
	}
	const OaVideoProfile& resolved = *resolvedResult;
	VkVideoProfileInfoKHR profile = {};
	profile.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR;
	switch (resolved.Codec) {
		case OaVideoCodec::H264:
			OutH264.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR;
		switch (resolved.StandardProfile) {
		case OaVideoCodecProfile::H264Baseline: OutH264.stdProfileIdc = STD_VIDEO_H264_PROFILE_IDC_BASELINE; break;
		case OaVideoCodecProfile::H264Main: OutH264.stdProfileIdc = STD_VIDEO_H264_PROFILE_IDC_MAIN; break;
		case OaVideoCodecProfile::H264High:
			OutH264.stdProfileIdc = STD_VIDEO_H264_PROFILE_IDC_HIGH; break;
		case OaVideoCodecProfile::H264High444Predictive:
			OutH264.stdProfileIdc = STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE;
			break;
		default: return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid H.264 standard profile");
		}
		switch (resolved.H264PictureLayout) {
		case OaVideoH264PictureLayout::Progressive:
			OutH264.pictureLayout = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR;
			break;
		case OaVideoH264PictureLayout::InterlacedInterleavedLines:
			OutH264.pictureLayout = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_INTERLACED_INTERLEAVED_LINES_BIT_KHR;
			break;
		case OaVideoH264PictureLayout::InterlacedSeparatePlanes:
			OutH264.pictureLayout = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_INTERLACED_SEPARATE_PLANES_BIT_KHR;
			break;
		}
			profile.pNext = &OutH264;
			profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
			break;
		case OaVideoCodec::H265:
			OutH265.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_INFO_KHR;
		switch (resolved.StandardProfile) {
		case OaVideoCodecProfile::H265Main:
			OutH265.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN; break;
		case OaVideoCodecProfile::H265Main10: OutH265.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN_10; break;
		case OaVideoCodecProfile::H265MainStillPicture:
			OutH265.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN_STILL_PICTURE;
			break;
		case OaVideoCodecProfile::H265FormatRangeExtensions:
			OutH265.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS;
			break;
		case OaVideoCodecProfile::H265ScreenContentCodingExtensions:
			OutH265.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_SCC_EXTENSIONS;
			break;
		default: return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid H.265 standard profile");
		}
			profile.pNext = &OutH265;
			profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR;
			break;
		case OaVideoCodec::AV1:
			OutAV1.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR;
		switch (resolved.StandardProfile) {
		case OaVideoCodecProfile::Av1Main:
			OutAV1.stdProfile = STD_VIDEO_AV1_PROFILE_MAIN; break;
		case OaVideoCodecProfile::Av1High: OutAV1.stdProfile = STD_VIDEO_AV1_PROFILE_HIGH; break;
		case OaVideoCodecProfile::Av1Professional: OutAV1.stdProfile = STD_VIDEO_AV1_PROFILE_PROFESSIONAL; break;
		default: return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 standard profile");
		}
			OutAV1.filmGrainSupport = resolved.Av1FilmGrain ? VK_TRUE : VK_FALSE;
			profile.pNext = &OutAV1;
			profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
			break;
		case OaVideoCodec::VP9:
			OutVp9.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PROFILE_INFO_KHR;
		switch (resolved.StandardProfile) {
		case OaVideoCodecProfile::Vp9Profile0:
			OutVp9.stdProfile = STD_VIDEO_VP9_PROFILE_0; break;
		case OaVideoCodecProfile::Vp9Profile1: OutVp9.stdProfile = STD_VIDEO_VP9_PROFILE_1; break;
		case OaVideoCodecProfile::Vp9Profile2: OutVp9.stdProfile = STD_VIDEO_VP9_PROFILE_2; break;
		case OaVideoCodecProfile::Vp9Profile3: OutVp9.stdProfile = STD_VIDEO_VP9_PROFILE_3; break;
		default: return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid VP9 standard profile");
		}
			profile.pNext = &OutVp9;
			profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR;
			break;
	}

	auto chromaResult = ToVkChroma(resolved.ChromaSubsampling);
	if (not chromaResult.IsOk()) return chromaResult.GetStatus();
	auto lumaDepthResult = ToVkBitDepth(resolved.LumaBitDepth);
	if (not lumaDepthResult.IsOk()) return lumaDepthResult.GetStatus();
	auto chromaDepthResult = ToVkBitDepth(resolved.ChromaBitDepth);
	if (not chromaDepthResult.IsOk()) return chromaDepthResult.GetStatus();
	profile.chromaSubsampling = *chromaResult;
	profile.lumaBitDepth      = *lumaDepthResult;
	profile.chromaBitDepth    = *chromaDepthResult;
	return profile;
}

} // namespace OaVideoDecoderProfile
