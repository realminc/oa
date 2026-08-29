// Shared exact VkVideoProfileInfoKHR builder for oa::VideoDecoder.

#include "videoDecoderProfile.h"

namespace {

bool profileMatchesCodec(
	oa::VideoCodec inCodec, oa::VideoCodecProfile inProfile)
{
	switch (inProfile) {
	case oa::VideoCodecProfile::H264Baseline:
	case oa::VideoCodecProfile::H264Main:
	case oa::VideoCodecProfile::H264High:
	case oa::VideoCodecProfile::H264High444Predictive: return inCodec == oa::VideoCodec::H264;
	case oa::VideoCodecProfile::H265Main:
	case oa::VideoCodecProfile::H265Main10:
	case oa::VideoCodecProfile::H265MainStillPicture:
	case oa::VideoCodecProfile::H265FormatRangeExtensions:
	case oa::VideoCodecProfile::H265ScreenContentCodingExtensions: return inCodec == oa::VideoCodec::H265;
	case oa::VideoCodecProfile::Av1Main:
	case oa::VideoCodecProfile::Av1High:
	case oa::VideoCodecProfile::Av1Professional: return inCodec == oa::VideoCodec::AV1;
	case oa::VideoCodecProfile::Vp9Profile0:
	case oa::VideoCodecProfile::Vp9Profile1:
	case oa::VideoCodecProfile::Vp9Profile2:
	case oa::VideoCodecProfile::Vp9Profile3: return inCodec == oa::VideoCodec::VP9;
	case oa::VideoCodecProfile::Unspecified: return true;
	}
	return false;
}

oa::VideoCodecProfile defaultProfile(oa::VideoCodec inCodec)
{
	switch (inCodec) {
	case oa::VideoCodec::H264: return oa::VideoCodecProfile::H264High;
	case oa::VideoCodec::H265: return oa::VideoCodecProfile::H265Main;
	case oa::VideoCodec::AV1: return oa::VideoCodecProfile::Av1Main;
	case oa::VideoCodec::VP9: return oa::VideoCodecProfile::Vp9Profile0;
	}
	return oa::VideoCodecProfile::Unspecified;
}

oa::Result<VkVideoChromaSubsamplingFlagsKHR> toVkChroma(oa::VideoChromaSubsampling inChroma)
{
	switch (inChroma) {
	case oa::VideoChromaSubsampling::Monochrome: return VK_VIDEO_CHROMA_SUBSAMPLING_MONOCHROME_BIT_KHR;
	case oa::VideoChromaSubsampling::Yuv420: return VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
	case oa::VideoChromaSubsampling::Yuv422: return VK_VIDEO_CHROMA_SUBSAMPLING_422_BIT_KHR;
	case oa::VideoChromaSubsampling::Yuv444: return VK_VIDEO_CHROMA_SUBSAMPLING_444_BIT_KHR;
	}
	return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid video chroma-subsampling value");
}

oa::Result<VkVideoComponentBitDepthFlagsKHR> toVkBitDepth(oa::VideoBitDepth inDepth)
{
	switch (inDepth) {
	case oa::VideoBitDepth::Bit8: return VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
	case oa::VideoBitDepth::Bit10: return VK_VIDEO_COMPONENT_BIT_DEPTH_10_BIT_KHR;
	case oa::VideoBitDepth::Bit12: return VK_VIDEO_COMPONENT_BIT_DEPTH_12_BIT_KHR;
	}
	return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid video component bit-depth value");
}

} // namespace

namespace oa::videoDecoderProfile {

oa::Result<oa::VideoProfile> resolveDecodeProfile(const oa::VideoProfile& inProfile)
{
	oa::VideoProfile profile = inProfile;
	if (profile.standardProfile == oa::VideoCodecProfile::Unspecified) {
		profile.standardProfile = defaultProfile(profile.codec);
	}
	if (profile.standardProfile == oa::VideoCodecProfile::Unspecified or
		not profileMatchesCodec(profile.codec, profile.standardProfile)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
							   "Video standard profile does not match the requested codec");
	}
	if (profile.av1FilmGrain and profile.codec != oa::VideoCodec::AV1) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
							   "Film-grain support is only valid for AV1 decode profiles");
	}
	auto chromaResult = toVkChroma(profile.chromaSubsampling);
	if (not chromaResult.isOk()) return chromaResult.getStatus();
	auto lumaDepthResult = toVkBitDepth(profile.lumaBitDepth);
	if (not lumaDepthResult.isOk()) return lumaDepthResult.getStatus();
	auto chromaDepthResult = toVkBitDepth(profile.chromaBitDepth);
	if (not chromaDepthResult.isOk()) return chromaDepthResult.getStatus();
	return profile;
}

bool isDecodePathImplemented(const oa::VideoProfile& inProfile)
{
	if (inProfile.chromaSubsampling != oa::VideoChromaSubsampling::Yuv420
		or inProfile.lumaBitDepth != inProfile.chromaBitDepth) {
		return false;
	}
	const bool is8Bit = inProfile.lumaBitDepth == oa::VideoBitDepth::Bit8;
	const bool is10Bit = inProfile.lumaBitDepth == oa::VideoBitDepth::Bit10;
	switch (inProfile.standardProfile) {
	case oa::VideoCodecProfile::H264Baseline:
	case oa::VideoCodecProfile::H264Main:
	case oa::VideoCodecProfile::H264High:
		return is8Bit and inProfile.codec == oa::VideoCodec::H264 and
			   inProfile.h264PictureLayout == oa::VideoH264PictureLayout::Progressive;
	case oa::VideoCodecProfile::H265Main:
		return is8Bit and inProfile.codec == oa::VideoCodec::H265;
	case oa::VideoCodecProfile::H265Main10:
		return is10Bit and inProfile.codec == oa::VideoCodec::H265;
	case oa::VideoCodecProfile::H265FormatRangeExtensions:
		// The qualified Range-Extensions subset is the Main-Intra-compatible
		// 8/10-bit 4:2:0 surface. Extension tools and every non-4:2:0 axis remain
		// closed until their Std structures and native formats are proven.
		return (is8Bit or is10Bit) and inProfile.codec == oa::VideoCodec::H265;
	case oa::VideoCodecProfile::Av1Main:
		return (is8Bit or is10Bit) and inProfile.codec == oa::VideoCodec::AV1
			and not inProfile.av1FilmGrain;
	case oa::VideoCodecProfile::Vp9Profile0:
		return is8Bit and inProfile.codec == oa::VideoCodec::VP9;
	case oa::VideoCodecProfile::Vp9Profile2:
		return is10Bit and inProfile.codec == oa::VideoCodec::VP9;
	default: return false;
	}
}

oa::Result<VkVideoProfileInfoKHR> buildDecodeProfile(const oa::VideoProfile& inProfile,
	VkVideoDecodeH264ProfileInfoKHR& outH264,
	VkVideoDecodeH265ProfileInfoKHR& outH265,
	VkVideoDecodeAV1ProfileInfoKHR& outAV1,
	VkVideoDecodeVP9ProfileInfoKHR& outVp9)
{
	auto resolvedResult = resolveDecodeProfile(inProfile);
	if (not resolvedResult.isOk()) {
		return resolvedResult.getStatus();
	}
	const oa::VideoProfile& resolved = *resolvedResult;
	VkVideoProfileInfoKHR profile = {};
	profile.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR;
	switch (resolved.codec) {
		case oa::VideoCodec::H264:
			outH264.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR;
		switch (resolved.standardProfile) {
		case oa::VideoCodecProfile::H264Baseline: outH264.stdProfileIdc = STD_VIDEO_H264_PROFILE_IDC_BASELINE; break;
		case oa::VideoCodecProfile::H264Main: outH264.stdProfileIdc = STD_VIDEO_H264_PROFILE_IDC_MAIN; break;
		case oa::VideoCodecProfile::H264High:
			outH264.stdProfileIdc = STD_VIDEO_H264_PROFILE_IDC_HIGH; break;
		case oa::VideoCodecProfile::H264High444Predictive:
			outH264.stdProfileIdc = STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE;
			break;
		default: return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid H.264 standard profile");
		}
		switch (resolved.h264PictureLayout) {
		case oa::VideoH264PictureLayout::Progressive:
			outH264.pictureLayout = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR;
			break;
		case oa::VideoH264PictureLayout::InterlacedInterleavedLines:
			outH264.pictureLayout = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_INTERLACED_INTERLEAVED_LINES_BIT_KHR;
			break;
		case oa::VideoH264PictureLayout::InterlacedSeparatePlanes:
			outH264.pictureLayout = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_INTERLACED_SEPARATE_PLANES_BIT_KHR;
			break;
		}
			profile.pNext = &outH264;
			profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
			break;
		case oa::VideoCodec::H265:
			outH265.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_INFO_KHR;
		switch (resolved.standardProfile) {
		case oa::VideoCodecProfile::H265Main:
			outH265.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN; break;
		case oa::VideoCodecProfile::H265Main10: outH265.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN_10; break;
		case oa::VideoCodecProfile::H265MainStillPicture:
			outH265.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN_STILL_PICTURE;
			break;
		case oa::VideoCodecProfile::H265FormatRangeExtensions:
			outH265.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS;
			break;
		case oa::VideoCodecProfile::H265ScreenContentCodingExtensions:
			outH265.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_SCC_EXTENSIONS;
			break;
		default: return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid H.265 standard profile");
		}
			profile.pNext = &outH265;
			profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR;
			break;
		case oa::VideoCodec::AV1:
			outAV1.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR;
		switch (resolved.standardProfile) {
		case oa::VideoCodecProfile::Av1Main:
			outAV1.stdProfile = STD_VIDEO_AV1_PROFILE_MAIN; break;
		case oa::VideoCodecProfile::Av1High: outAV1.stdProfile = STD_VIDEO_AV1_PROFILE_HIGH; break;
		case oa::VideoCodecProfile::Av1Professional: outAV1.stdProfile = STD_VIDEO_AV1_PROFILE_PROFESSIONAL; break;
		default: return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 standard profile");
		}
			outAV1.filmGrainSupport = resolved.av1FilmGrain ? VK_TRUE : VK_FALSE;
			profile.pNext = &outAV1;
			profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
			break;
		case oa::VideoCodec::VP9:
			outVp9.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PROFILE_INFO_KHR;
		switch (resolved.standardProfile) {
		case oa::VideoCodecProfile::Vp9Profile0:
			outVp9.stdProfile = STD_VIDEO_VP9_PROFILE_0; break;
		case oa::VideoCodecProfile::Vp9Profile1: outVp9.stdProfile = STD_VIDEO_VP9_PROFILE_1; break;
		case oa::VideoCodecProfile::Vp9Profile2: outVp9.stdProfile = STD_VIDEO_VP9_PROFILE_2; break;
		case oa::VideoCodecProfile::Vp9Profile3: outVp9.stdProfile = STD_VIDEO_VP9_PROFILE_3; break;
		default: return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid VP9 standard profile");
		}
			profile.pNext = &outVp9;
			profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR;
			break;
	}

	auto chromaResult = toVkChroma(resolved.chromaSubsampling);
	if (not chromaResult.isOk()) return chromaResult.getStatus();
	auto lumaDepthResult = toVkBitDepth(resolved.lumaBitDepth);
	if (not lumaDepthResult.isOk()) return lumaDepthResult.getStatus();
	auto chromaDepthResult = toVkBitDepth(resolved.chromaBitDepth);
	if (not chromaDepthResult.isOk()) return chromaDepthResult.getStatus();
	profile.chromaSubsampling = *chromaResult;
	profile.lumaBitDepth      = *lumaDepthResult;
	profile.chromaBitDepth    = *chromaDepthResult;
	return profile;
}

} // namespace oa::videoDecoderProfile
