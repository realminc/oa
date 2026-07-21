// OA Vision — Video decoder session-parameter upload (H.264/H.265).
// DPB/bitstream helpers: FnVideo/Decoder/FnVideoDecoderShared.cpp
// vkCmdDecodeVideoKHR recording: FnVideo/Decoder/*/FnVideoDecoder*Record.cpp

#include <Oa/Vision/VideoDecoder.h>
#include <Oa/Runtime/Engine.h>
#include "../Codec/VcpH264.h"
#include "../Codec/VcpH265.h"

OaStatus OaVideoDecoder::CacheSps(OaU32 InSpsId, const OaH264SpsData& InSps)
{
	SpsCache_.Insert({InSpsId, InSps});
	return UpdateH264SessionParametersFromSps(InSps);
}

OaStatus OaVideoDecoder::CachePps(OaU32 InPpsId, const OaH264PpsData& InPps)
{
	PpsCache_.Insert({InPpsId, InPps});
	return UpdateH264SessionParametersFromPps(InPps);
}

const OaH264SpsData* OaVideoDecoder::GetSps(OaU32 InSpsId) const
{
	auto it = SpsCache_.Find(InSpsId);
	return (it != SpsCache_.End()) ? &it->second : nullptr;
}

const OaH264PpsData* OaVideoDecoder::GetPps(OaU32 InPpsId) const
{
	auto it = PpsCache_.Find(InPpsId);
	return (it != PpsCache_.End()) ? &it->second : nullptr;
}

void OaVideoDecoder::ClearParameterSets()
{
	SpsCache_.Clear();
	PpsCache_.Clear();
	H265VpsCache_.Clear();
	H265SpsCache_.Clear();
	H265PpsCache_.Clear();
}

OaStatus OaVideoDecoder::UpdateH264SessionParametersFromSps(const OaH264SpsData& InSps)
{
	if (Profile_.Codec != OaVideoCodec::H264 || SessionParams_.Handle() == VK_NULL_HANDLE) {
		return OaStatus::Ok();
	}
	if (InSps.SpsId >= H264SpsUploaded_.Size()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "H.264 SPS id exceeds Vulkan session parameter capacity");
	}
	if (H264SpsUploaded_[InSps.SpsId]) {
		return OaStatus::Ok();
	}
	if (OaVcpH264::ToStdH264Level(InSps.LevelIdc) == STD_VIDEO_H264_LEVEL_IDC_INVALID) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Unsupported H.264 level_idc");
	}
	if (InSps.ChromaFormatIdc != 1 || InSps.BitDepthLumaMinus8 != 0 || InSps.BitDepthChromaMinus8 != 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Only H.264 8-bit 4:2:0 session parameters are supported");
	}
	if (!vkUpdateVideoSessionParametersKHR) {
		return OaStatus::Error("vkUpdateVideoSessionParametersKHR is not loaded");
	}

	StdVideoH264SequenceParameterSet stdSps = OaVcpH264::ToStdH264Sps(InSps);
	VkVideoDecodeH264SessionParametersAddInfoKHR addInfo = {};
	addInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR;
	addInfo.stdSPSCount = 1;
	addInfo.pStdSPSs = &stdSps;

	VkVideoSessionParametersUpdateInfoKHR updateInfo = {};
	updateInfo.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR;
	updateInfo.pNext = &addInfo;
	updateInfo.updateSequenceCount = SessionParameterUpdateCount_ + 1;

	auto& vkEngine = *Rt_;
	VkResult result = vkUpdateVideoSessionParametersKHR(
		static_cast<VkDevice>(vkEngine.Device.Device),
		SessionParams_.Handle(),
		&updateInfo);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkUpdateVideoSessionParametersKHR failed for H.264 SPS");
	}
	H264SpsUploaded_[InSps.SpsId] = true;
	SessionParameterUpdateCount_ = updateInfo.updateSequenceCount;
	return OaStatus::Ok();
}

OaStatus OaVideoDecoder::UpdateH264SessionParametersFromPps(const OaH264PpsData& InPps)
{
	if (Profile_.Codec != OaVideoCodec::H264 || SessionParams_.Handle() == VK_NULL_HANDLE) {
		return OaStatus::Ok();
	}
	if (InPps.PpsId >= H264PpsUploaded_.Size() || InPps.SpsId >= H264SpsUploaded_.Size()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "H.264 PPS id exceeds Vulkan session parameter capacity");
	}
	if (H264PpsUploaded_[InPps.PpsId]) {
		return OaStatus::Ok();
	}
	if (!H264SpsUploaded_[InPps.SpsId]) {
		const OaH264SpsData* sps = GetSps(InPps.SpsId);
		if (!sps) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "H.264 PPS references an unknown SPS");
		}
		OA_RETURN_IF_ERROR(UpdateH264SessionParametersFromSps(*sps));
	}
	if (!vkUpdateVideoSessionParametersKHR) {
		return OaStatus::Error("vkUpdateVideoSessionParametersKHR is not loaded");
	}

	StdVideoH264PictureParameterSet stdPps = OaVcpH264::ToStdH264Pps(InPps);
	VkVideoDecodeH264SessionParametersAddInfoKHR addInfo = {};
	addInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR;
	addInfo.stdPPSCount = 1;
	addInfo.pStdPPSs = &stdPps;

	VkVideoSessionParametersUpdateInfoKHR updateInfo = {};
	updateInfo.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR;
	updateInfo.pNext = &addInfo;
	updateInfo.updateSequenceCount = SessionParameterUpdateCount_ + 1;

	auto& vkEngine = *Rt_;
	VkResult result = vkUpdateVideoSessionParametersKHR(
		static_cast<VkDevice>(vkEngine.Device.Device),
		SessionParams_.Handle(),
		&updateInfo);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkUpdateVideoSessionParametersKHR failed for H.264 PPS");
	}
	H264PpsUploaded_[InPps.PpsId] = true;
	SessionParameterUpdateCount_ = updateInfo.updateSequenceCount;
	return OaStatus::Ok();
}

OaStatus OaVideoDecoder::UpdateH265SessionParametersFromVps(const OaH265VpsData& InVps)
{
	if (Profile_.Codec != OaVideoCodec::H265 || SessionParams_.Handle() == VK_NULL_HANDLE) {
		return OaStatus::Ok();
	}
	if (InVps.VpsId >= H265VpsUploaded_.Size()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "H.265 VPS id exceeds Vulkan session parameter capacity");
	}
	if (H265VpsUploaded_[InVps.VpsId]) {
		return OaStatus::Ok();
	}
	if (OaVcpH265::ToStdH265Level(InVps.GeneralLevelIdc) == STD_VIDEO_H265_LEVEL_IDC_INVALID) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Unsupported H.265 general_level_idc");
	}
	if (!vkUpdateVideoSessionParametersKHR) {
		return OaStatus::Error("vkUpdateVideoSessionParametersKHR is not loaded");
	}

	StdVideoH265ProfileTierLevel profile = OaVcpH265::ToStdH265ProfileTierLevel(InVps);
	StdVideoH265VideoParameterSet stdVps = OaVcpH265::ToStdH265Vps(InVps, profile);
	VkVideoDecodeH265SessionParametersAddInfoKHR addInfo = {};
	addInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR;
	addInfo.stdVPSCount = 1;
	addInfo.pStdVPSs = &stdVps;

	VkVideoSessionParametersUpdateInfoKHR updateInfo = {};
	updateInfo.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR;
	updateInfo.pNext = &addInfo;
	updateInfo.updateSequenceCount = SessionParameterUpdateCount_ + 1;

	auto& vkEngine = *Rt_;
	VkResult result = vkUpdateVideoSessionParametersKHR(
		static_cast<VkDevice>(vkEngine.Device.Device),
		SessionParams_.Handle(),
		&updateInfo);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkUpdateVideoSessionParametersKHR failed for H.265 VPS");
	}
	H265VpsUploaded_[InVps.VpsId] = true;
	SessionParameterUpdateCount_ = updateInfo.updateSequenceCount;
	return OaStatus::Ok();
}

OaStatus OaVideoDecoder::UpdateH265SessionParametersFromSps(const OaH265SpsData& InSps)
{
	if (Profile_.Codec != OaVideoCodec::H265 || SessionParams_.Handle() == VK_NULL_HANDLE) {
		return OaStatus::Ok();
	}
	if (InSps.SpsId >= H265SpsUploaded_.Size() || InSps.VpsId >= H265VpsUploaded_.Size()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "H.265 SPS id exceeds Vulkan session parameter capacity");
	}
	if (H265SpsUploaded_[InSps.SpsId]) {
		return OaStatus::Ok();
	}
	if (InSps.ChromaFormatIdc != STD_VIDEO_H265_CHROMA_FORMAT_IDC_420 ||
		InSps.BitDepthLumaMinus8 != 0 ||
		InSps.BitDepthChromaMinus8 != 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Only H.265 8-bit 4:2:0 session parameters are supported");
	}
	auto vpsIt = H265VpsCache_.Find(InSps.VpsId);
	if (vpsIt == H265VpsCache_.End()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "H.265 SPS references an unknown VPS");
	}
	OA_RETURN_IF_ERROR(UpdateH265SessionParametersFromVps(vpsIt->second));
	if (!vkUpdateVideoSessionParametersKHR) {
		return OaStatus::Error("vkUpdateVideoSessionParametersKHR is not loaded");
	}

	StdVideoH265ProfileTierLevel profile = OaVcpH265::ToStdH265ProfileTierLevel(vpsIt->second);
	StdVideoH265DecPicBufMgr dpb = OaVcpH265::ToStdH265DecPicBufMgr(InSps);
	StdVideoH265SequenceParameterSet stdSps = OaVcpH265::ToStdH265Sps(InSps, profile, dpb);
	VkVideoDecodeH265SessionParametersAddInfoKHR addInfo = {};
	addInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR;
	addInfo.stdSPSCount = 1;
	addInfo.pStdSPSs = &stdSps;

	VkVideoSessionParametersUpdateInfoKHR updateInfo = {};
	updateInfo.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR;
	updateInfo.pNext = &addInfo;
	updateInfo.updateSequenceCount = SessionParameterUpdateCount_ + 1;

	auto& vkEngine = *Rt_;
	VkResult result = vkUpdateVideoSessionParametersKHR(
		static_cast<VkDevice>(vkEngine.Device.Device),
		SessionParams_.Handle(),
		&updateInfo);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkUpdateVideoSessionParametersKHR failed for H.265 SPS");
	}
	H265SpsUploaded_[InSps.SpsId] = true;
	SessionParameterUpdateCount_ = updateInfo.updateSequenceCount;
	return OaStatus::Ok();
}

OaStatus OaVideoDecoder::UpdateH265SessionParametersFromPps(const OaH265PpsData& InPps)
{
	if (Profile_.Codec != OaVideoCodec::H265 || SessionParams_.Handle() == VK_NULL_HANDLE) {
		return OaStatus::Ok();
	}
	if (InPps.PpsId >= H265PpsUploaded_.Size() || InPps.SpsId >= H265SpsUploaded_.Size()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "H.265 PPS id exceeds Vulkan session parameter capacity");
	}
	if (H265PpsUploaded_[InPps.PpsId]) {
		return OaStatus::Ok();
	}
	auto spsIt = H265SpsCache_.Find(InPps.SpsId);
	if (spsIt == H265SpsCache_.End()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "H.265 PPS references an unknown SPS");
	}
	if (!H265SpsUploaded_[InPps.SpsId]) {
		OA_RETURN_IF_ERROR(UpdateH265SessionParametersFromSps(spsIt->second));
	}
	if (!vkUpdateVideoSessionParametersKHR) {
		return OaStatus::Error("vkUpdateVideoSessionParametersKHR is not loaded");
	}

	StdVideoH265PictureParameterSet stdPps = OaVcpH265::ToStdH265Pps(InPps, spsIt->second);
	VkVideoDecodeH265SessionParametersAddInfoKHR addInfo = {};
	addInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR;
	addInfo.stdPPSCount = 1;
	addInfo.pStdPPSs = &stdPps;

	VkVideoSessionParametersUpdateInfoKHR updateInfo = {};
	updateInfo.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR;
	updateInfo.pNext = &addInfo;
	updateInfo.updateSequenceCount = SessionParameterUpdateCount_ + 1;

	auto& vkEngine = *Rt_;
	VkResult result = vkUpdateVideoSessionParametersKHR(
		static_cast<VkDevice>(vkEngine.Device.Device),
		SessionParams_.Handle(),
		&updateInfo);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkUpdateVideoSessionParametersKHR failed for H.265 PPS");
	}
	H265PpsUploaded_[InPps.PpsId] = true;
	SessionParameterUpdateCount_ = updateInfo.updateSequenceCount;
	return OaStatus::Ok();
}
