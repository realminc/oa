// OA Vision — Video decoder session-parameter upload (H.264/H.265).
// DPB/bitstream helpers: Video/Decoder/codec/VideoDecoderCodecAccess.cpp
// vkCmdDecodeVideoKHR recording: Video/Decoder/codec/*/VideoDecoder*Record.cpp

#include <oa/vision/videoDecoder.h>
#include "videoDecoderImpl.h"
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/deviceAccess.h>
#include "../codec/vcpH264.h"
#include "../codec/vcpH265.h"

oa::Status oa::VideoDecoder::cacheSps(oa::U32 inSpsId, const oa::H264SpsData& inSps)
{
	impl_->spsCache.insert({inSpsId, inSps});
	return updateH264SessionParametersFromSps(inSps);
}

oa::Status oa::VideoDecoder::cachePps(oa::U32 inPpsId, const oa::H264PpsData& inPps)
{
	impl_->ppsCache.insert({inPpsId, inPps});
	return updateH264SessionParametersFromPps(inPps);
}

const oa::H264SpsData* oa::VideoDecoder::getSps(oa::U32 inSpsId) const
{
	auto it = impl_->spsCache.find(inSpsId);
	return (it != impl_->spsCache.end()) ? &it->second : nullptr;
}

const oa::H264PpsData* oa::VideoDecoder::getPps(oa::U32 inPpsId) const
{
	auto it = impl_->ppsCache.find(inPpsId);
	return (it != impl_->ppsCache.end()) ? &it->second : nullptr;
}

void oa::VideoDecoder::clearParameterSets()
{
	impl_->spsCache.clear();
	impl_->ppsCache.clear();
	impl_->h265VpsCache.clear();
	impl_->h265SpsCache.clear();
	impl_->h265PpsCache.clear();
}

oa::Status oa::VideoDecoder::updateH264SessionParametersFromSps(const oa::H264SpsData& inSps)
{
	if (impl_->profile.codec != oa::VideoCodec::H264 || impl_->sessionParameters.handle() == VK_NULL_HANDLE) {
		return oa::Status::ok();
	}
	if (inSps.spsId >= impl_->h264SpsUploaded.size()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "H.264 SPS id exceeds vulkan session parameter capacity");
	}
	if (impl_->h264SpsUploaded[inSps.spsId]) {
		return oa::Status::ok();
	}
	if (oa::VcpH264::toStdH264Level(inSps.levelIdc) == STD_VIDEO_H264_LEVEL_IDC_INVALID) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Unsupported H.264 level_idc");
	}
	if (inSps.chromaFormatIdc != 1 || inSps.bitDepthLumaMinus8 != 0 || inSps.bitDepthChromaMinus8 != 0) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Only H.264 8-bit 4:2:0 session parameters are supported");
	}
	if (!oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkUpdateVideoSessionParametersKHR) {
		return oa::Status::error("vkUpdateVideoSessionParametersKHR is not loaded");
	}

	StdVideoH264SequenceParameterSet stdSps = oa::VcpH264::toStdH264Sps(inSps);
	VkVideoDecodeH264SessionParametersAddInfoKHR addInfo = {};
	addInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR;
	addInfo.stdSPSCount = 1;
	addInfo.pStdSPSs = &stdSps;

	VkVideoSessionParametersUpdateInfoKHR updateInfo = {};
	updateInfo.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR;
	updateInfo.pNext = &addInfo;
	updateInfo.updateSequenceCount = impl_->sessionParameterUpdateCount + 1;

	auto& vkEngine = *impl_->engine;
	VkResult result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkUpdateVideoSessionParametersKHR(
		static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device),
		impl_->sessionParameters.handle(),
		&updateInfo);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkUpdateVideoSessionParametersKHR failed for H.264 SPS");
	}
	impl_->h264SpsUploaded[inSps.spsId] = true;
	impl_->sessionParameterUpdateCount = updateInfo.updateSequenceCount;
	return oa::Status::ok();
}

oa::Status oa::VideoDecoder::updateH264SessionParametersFromPps(const oa::H264PpsData& inPps)
{
	if (impl_->profile.codec != oa::VideoCodec::H264 || impl_->sessionParameters.handle() == VK_NULL_HANDLE) {
		return oa::Status::ok();
	}
	if (inPps.ppsId >= impl_->h264PpsUploaded.size() || inPps.spsId >= impl_->h264SpsUploaded.size()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "H.264 PPS id exceeds vulkan session parameter capacity");
	}
	if (impl_->h264PpsUploaded[inPps.ppsId]) {
		return oa::Status::ok();
	}
	if (!impl_->h264SpsUploaded[inPps.spsId]) {
		const oa::H264SpsData* sps = getSps(inPps.spsId);
		if (!sps) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "H.264 PPS references an unknown SPS");
		}
		OA_RETURN_IF_ERROR(updateH264SessionParametersFromSps(*sps));
	}
	if (!oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkUpdateVideoSessionParametersKHR) {
		return oa::Status::error("vkUpdateVideoSessionParametersKHR is not loaded");
	}

	StdVideoH264PictureParameterSet stdPps = oa::VcpH264::toStdH264Pps(inPps);
	VkVideoDecodeH264SessionParametersAddInfoKHR addInfo = {};
	addInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR;
	addInfo.stdPPSCount = 1;
	addInfo.pStdPPSs = &stdPps;

	VkVideoSessionParametersUpdateInfoKHR updateInfo = {};
	updateInfo.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR;
	updateInfo.pNext = &addInfo;
	updateInfo.updateSequenceCount = impl_->sessionParameterUpdateCount + 1;

	auto& vkEngine = *impl_->engine;
	VkResult result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkUpdateVideoSessionParametersKHR(
		static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device),
		impl_->sessionParameters.handle(),
		&updateInfo);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkUpdateVideoSessionParametersKHR failed for H.264 PPS");
	}
	impl_->h264PpsUploaded[inPps.ppsId] = true;
	impl_->sessionParameterUpdateCount = updateInfo.updateSequenceCount;
	return oa::Status::ok();
}

oa::Status oa::VideoDecoder::updateH265SessionParametersFromVps(const oa::H265VpsData& inVps)
{
	if (impl_->profile.codec != oa::VideoCodec::H265 || impl_->sessionParameters.handle() == VK_NULL_HANDLE) {
		return oa::Status::ok();
	}
	if (inVps.vpsId >= impl_->h265VpsUploaded.size()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "H.265 VPS id exceeds vulkan session parameter capacity");
	}
	if (impl_->h265VpsUploaded[inVps.vpsId]) {
		return oa::Status::ok();
	}
	if (oa::VcpH265::toStdH265Level(inVps.generalLevelIdc) == STD_VIDEO_H265_LEVEL_IDC_INVALID) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Unsupported H.265 general_level_idc");
	}
	if (!oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkUpdateVideoSessionParametersKHR) {
		return oa::Status::error("vkUpdateVideoSessionParametersKHR is not loaded");
	}

	StdVideoH265ProfileTierLevel profile = oa::VcpH265::toStdH265ProfileTierLevel(inVps);
	StdVideoH265VideoParameterSet stdVps = oa::VcpH265::toStdH265Vps(inVps, profile);
	VkVideoDecodeH265SessionParametersAddInfoKHR addInfo = {};
	addInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR;
	addInfo.stdVPSCount = 1;
	addInfo.pStdVPSs = &stdVps;

	VkVideoSessionParametersUpdateInfoKHR updateInfo = {};
	updateInfo.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR;
	updateInfo.pNext = &addInfo;
	updateInfo.updateSequenceCount = impl_->sessionParameterUpdateCount + 1;

	auto& vkEngine = *impl_->engine;
	VkResult result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkUpdateVideoSessionParametersKHR(
		static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device),
		impl_->sessionParameters.handle(),
		&updateInfo);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkUpdateVideoSessionParametersKHR failed for H.265 VPS");
	}
	impl_->h265VpsUploaded[inVps.vpsId] = true;
	impl_->sessionParameterUpdateCount = updateInfo.updateSequenceCount;
	return oa::Status::ok();
}

oa::Status oa::VideoDecoder::updateH265SessionParametersFromSps(const oa::H265SpsData& inSps)
{
	if (impl_->profile.codec != oa::VideoCodec::H265 || impl_->sessionParameters.handle() == VK_NULL_HANDLE) {
		return oa::Status::ok();
	}
	if (inSps.spsId >= impl_->h265SpsUploaded.size() || inSps.vpsId >= impl_->h265VpsUploaded.size()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "H.265 SPS id exceeds vulkan session parameter capacity");
	}
	if (impl_->h265SpsUploaded[inSps.spsId]) {
		return oa::Status::ok();
	}
	const oa::U32 expectedDepthMinus8 =
		static_cast<oa::U32>(impl_->profile.lumaBitDepth) - 8U;
	if (inSps.chromaFormatIdc != STD_VIDEO_H265_CHROMA_FORMAT_IDC_420
		or inSps.bitDepthLumaMinus8 != expectedDepthMinus8
		or inSps.bitDepthChromaMinus8 != expectedDepthMinus8) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"H.265 SPS chroma/bit-depth does not match the decoder profile");
	}
	auto vpsIt = impl_->h265VpsCache.find(inSps.vpsId);
	if (vpsIt == impl_->h265VpsCache.end()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "H.265 SPS references an unknown VPS");
	}
	OA_RETURN_IF_ERROR(updateH265SessionParametersFromVps(vpsIt->second));
	if (!oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkUpdateVideoSessionParametersKHR) {
		return oa::Status::error("vkUpdateVideoSessionParametersKHR is not loaded");
	}

	StdVideoH265ProfileTierLevel profile = oa::VcpH265::toStdH265ProfileTierLevel(vpsIt->second);
	StdVideoH265DecPicBufMgr dpb = oa::VcpH265::toStdH265DecPicBufMgr(inSps);
	StdVideoH265SequenceParameterSet stdSps = oa::VcpH265::toStdH265Sps(inSps, profile, dpb);
	VkVideoDecodeH265SessionParametersAddInfoKHR addInfo = {};
	addInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR;
	addInfo.stdSPSCount = 1;
	addInfo.pStdSPSs = &stdSps;

	VkVideoSessionParametersUpdateInfoKHR updateInfo = {};
	updateInfo.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR;
	updateInfo.pNext = &addInfo;
	updateInfo.updateSequenceCount = impl_->sessionParameterUpdateCount + 1;

	auto& vkEngine = *impl_->engine;
	VkResult result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkUpdateVideoSessionParametersKHR(
		static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device),
		impl_->sessionParameters.handle(),
		&updateInfo);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkUpdateVideoSessionParametersKHR failed for H.265 SPS");
	}
	impl_->h265SpsUploaded[inSps.spsId] = true;
	impl_->sessionParameterUpdateCount = updateInfo.updateSequenceCount;
	return oa::Status::ok();
}

oa::Status oa::VideoDecoder::updateH265SessionParametersFromPps(const oa::H265PpsData& inPps)
{
	if (impl_->profile.codec != oa::VideoCodec::H265 || impl_->sessionParameters.handle() == VK_NULL_HANDLE) {
		return oa::Status::ok();
	}
	if (inPps.ppsId >= impl_->h265PpsUploaded.size() || inPps.spsId >= impl_->h265SpsUploaded.size()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "H.265 PPS id exceeds vulkan session parameter capacity");
	}
	if (impl_->h265PpsUploaded[inPps.ppsId]) {
		return oa::Status::ok();
	}
	auto spsIt = impl_->h265SpsCache.find(inPps.spsId);
	if (spsIt == impl_->h265SpsCache.end()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "H.265 PPS references an unknown SPS");
	}
	if (!impl_->h265SpsUploaded[inPps.spsId]) {
		OA_RETURN_IF_ERROR(updateH265SessionParametersFromSps(spsIt->second));
	}
	if (!oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkUpdateVideoSessionParametersKHR) {
		return oa::Status::error("vkUpdateVideoSessionParametersKHR is not loaded");
	}

	StdVideoH265PictureParameterSet stdPps = oa::VcpH265::toStdH265Pps(inPps, spsIt->second);
	VkVideoDecodeH265SessionParametersAddInfoKHR addInfo = {};
	addInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR;
	addInfo.stdPPSCount = 1;
	addInfo.pStdPPSs = &stdPps;

	VkVideoSessionParametersUpdateInfoKHR updateInfo = {};
	updateInfo.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR;
	updateInfo.pNext = &addInfo;
	updateInfo.updateSequenceCount = impl_->sessionParameterUpdateCount + 1;

	auto& vkEngine = *impl_->engine;
	VkResult result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkUpdateVideoSessionParametersKHR(
		static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device),
		impl_->sessionParameters.handle(),
		&updateInfo);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkUpdateVideoSessionParametersKHR failed for H.265 PPS");
	}
	impl_->h265PpsUploaded[inPps.ppsId] = true;
	impl_->sessionParameterUpdateCount = updateInfo.updateSequenceCount;
	return oa::Status::ok();
}
