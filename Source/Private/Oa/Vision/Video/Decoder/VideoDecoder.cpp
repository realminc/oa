// OA Vision — Hardware Video Decoder Implementation
// VK_KHR_video_decode_h264 / h265 / av1
// Zero-copy: compressed bitstream → VkImage (NV12) → compute shader

#include <Oa/Vision/VideoDecoder.h>
#include <Oa/Vision/FnImage.h>
#include <Oa/Runtime/Engine.h>
#include <array>
#include <Oa/Runtime/Init.h>
#include <Oa/Runtime/ImageDispatch.h>
#include <Oa/Runtime/OaVma.h>
#include <Oa/Core/FnMatrix.h>
#include "../Codec/NalParser.h"
#include "../Codec/CodecRegistry.h"
#include "../Codec/VcpH265.h"
#include "../Codec/VcpAv1.h"
#include "VideoDecoderProfile.h"

static OaF32 ClampUnit(OaF32 InValue)
{
	if (InValue < 0.0f) return 0.0f;
	if (InValue > 1.0f) return 1.0f;
	return InValue;
}

static VkSamplerYcbcrModelConversion ToVkYcbcrModel(OaYCbCrModel InColorSpace, OaU32 InWidth, OaU32 InHeight)
{
	if (InColorSpace == OaYCbCrModel::BT2020) {
		return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020;
	}
	if (InColorSpace == OaYCbCrModel::BT709 || InWidth >= 1280 || InHeight >= 720) {
		return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
	}
	return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601;
}

static OaU32 ToVisionColorSpace(OaYCbCrModel InColorSpace, OaU32 InWidth, OaU32 InHeight)
{
	if (InColorSpace == OaYCbCrModel::BT2020) {
		return 2;
	}
	if (InColorSpace == OaYCbCrModel::BT709 || InWidth >= 1280 || InHeight >= 720) {
		return 1;
	}
	return 0;
}

static void AttachCodecCapabilities(
	OaVideoCodec InCodec,
	VkVideoCapabilitiesKHR& InOutCaps,
	VkVideoDecodeCapabilitiesKHR& OutDecode,
	VkVideoDecodeH264CapabilitiesKHR& OutH264,
	VkVideoDecodeH265CapabilitiesKHR& OutH265,
	VkVideoDecodeAV1CapabilitiesKHR& OutAV1,
	VkVideoDecodeVP9CapabilitiesKHR& OutVp9)
{
	OutDecode.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR;
	InOutCaps.pNext = &OutDecode;
	switch (InCodec) {
		case OaVideoCodec::H264:
			OutH264.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_KHR;
			OutDecode.pNext = &OutH264;
			break;
		case OaVideoCodec::H265:
			OutH265.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_CAPABILITIES_KHR;
			OutDecode.pNext = &OutH265;
			break;
		case OaVideoCodec::AV1:
			OutAV1.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_CAPABILITIES_KHR;
			OutDecode.pNext = &OutAV1;
			break;
		case OaVideoCodec::VP9:
			OutVp9.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_CAPABILITIES_KHR;
			OutDecode.pNext = &OutVp9;
			break;
	}
}

static VkVideoCodecOperationFlagBitsKHR ToVideoDecodeCodecOp(OaVideoCodec InCodec)
{
	switch (InCodec) {
		case OaVideoCodec::H264:
			return VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
		case OaVideoCodec::H265:
			return VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR;
		case OaVideoCodec::AV1:
			return VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
		case OaVideoCodec::VP9:
			return VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR;
	}
	return VK_VIDEO_CODEC_OPERATION_NONE_KHR;
}

static bool VideoDecodeQueueSupportsCodec(
	const OaVkQueues& InQueues,
	OaVideoCodec InCodec)
{
	const VkVideoCodecOperationFlagsKHR queueOps = InQueues.VideoDecodeCodecOps;
	const VkVideoCodecOperationFlagBitsKHR codecOp = ToVideoDecodeCodecOp(InCodec);
	return codecOp != VK_VIDEO_CODEC_OPERATION_NONE_KHR
		&& (queueOps == 0 || (queueOps & codecOp) != 0);
}

static bool HasFormatWithUsage(
	const OaVec<VkVideoFormatPropertiesKHR>& InFormats,
	VkFormat InFormat,
	VkImageUsageFlags InUsage)
{
	for (const auto& format : InFormats) {
		if (format.format == InFormat && (format.imageUsageFlags & InUsage) == InUsage) {
			return true;
		}
	}
	return false;
}

static const VkVideoFormatPropertiesKHR* FindFormatWithUsage(
	const OaVec<VkVideoFormatPropertiesKHR>& InFormats,
	VkFormat InFormat,
	VkImageUsageFlags InUsage)
{
	for (const auto& format : InFormats) {
		if (format.format == InFormat && (format.imageUsageFlags & InUsage) == InUsage) {
			return &format;
		}
	}
	return nullptr;
}

static OaStatus QueryVideoFormats(
	VkPhysicalDevice InPhys,
	const VkVideoProfileInfoKHR& InProfile,
	VkImageUsageFlags InUsage,
	OaVec<VkVideoFormatPropertiesKHR>& OutFormats)
{
	if (!vkGetPhysicalDeviceVideoFormatPropertiesKHR) {
		return OaStatus::Error("vkGetPhysicalDeviceVideoFormatPropertiesKHR is not loaded");
	}

	VkVideoProfileListInfoKHR profileList = {};
	profileList.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
	profileList.profileCount = 1;
	profileList.pProfiles = &InProfile;

	VkPhysicalDeviceVideoFormatInfoKHR formatInfo = {};
	formatInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR;
	formatInfo.pNext = &profileList;
	formatInfo.imageUsage = InUsage;

	OaU32 formatCount = 0;
	VkResult result = vkGetPhysicalDeviceVideoFormatPropertiesKHR(InPhys, &formatInfo, &formatCount, nullptr);
	if (result != VK_SUCCESS) {
		return OaStatus::Error("Failed to query Vulkan Video format count");
	}

	OutFormats.Resize(formatCount);
	for (auto& format : OutFormats) {
		format = {};
		format.sType = VK_STRUCTURE_TYPE_VIDEO_FORMAT_PROPERTIES_KHR;
	}

	if (formatCount == 0) {
		return OaStatus::Ok();
	}

	result = vkGetPhysicalDeviceVideoFormatPropertiesKHR(InPhys, &formatInfo, &formatCount, OutFormats.Data());
	if (result != VK_SUCCESS) {
		OutFormats.Resize(0);
		return OaStatus::Error("Failed to query Vulkan Video formats");
	}
	OutFormats.Resize(formatCount);
	return OaStatus::Ok();
}

static OaStatus CreateDecodeSessionParameters(
	VkDevice InDevice,
	OaVideoCodec InCodec,
	const OaVideoProfile& InProfile,
	VkVideoSessionKHR InSession,
	VkVideoSessionParametersKHR& OutParameters,
	const OaAv1SequenceHeaderInfo* InAv1SeqHeader = nullptr
) {
	if (!vkCreateVideoSessionParametersKHR) {
		return OaStatus::Error("vkCreateVideoSessionParametersKHR is not loaded");
	}

	VkVideoSessionParametersCreateInfoKHR createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR;
	createInfo.videoSession = InSession;
	createInfo.videoSessionParametersTemplate = VK_NULL_HANDLE;

	VkVideoDecodeH264SessionParametersCreateInfoKHR h264Info = {};
	VkVideoDecodeH265SessionParametersCreateInfoKHR h265Info = {};
	VkVideoDecodeAV1SessionParametersCreateInfoKHR av1Info = {};
	StdVideoAV1ColorConfig av1ColorConfig = {};
	StdVideoAV1TimingInfo av1TimingInfo = {};
	StdVideoAV1SequenceHeader av1SequenceHeader = {};

	auto bitCountMinusOne = [](OaU32 value) -> OaU8 {
		OaU32 bits = 0;
		for (OaU32 tmp = value > 0 ? value - 1u : 0u; tmp != 0; tmp >>= 1u) {
			++bits;
		}
		return static_cast<OaU8>(bits > 0 ? bits - 1u : 0u);
	};

	switch (InCodec) {
		case OaVideoCodec::H264:
			h264Info.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR;
			h264Info.maxStdSPSCount = 32;
			h264Info.maxStdPPSCount = 256;
			h264Info.pParametersAddInfo = nullptr;
			createInfo.pNext = &h264Info;
			break;
		case OaVideoCodec::H265:
			h265Info.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR;
			h265Info.maxStdVPSCount = 16;
			h265Info.maxStdSPSCount = 32;
			h265Info.maxStdPPSCount = 256;
			h265Info.pParametersAddInfo = nullptr;
			createInfo.pNext = &h265Info;
			break;
		case OaVideoCodec::AV1:
		{
			av1ColorConfig.BitDepth = 8;
			av1ColorConfig.subsampling_x = 1;
			av1ColorConfig.subsampling_y = 1;
			av1ColorConfig.color_primaries = STD_VIDEO_AV1_COLOR_PRIMARIES_UNSPECIFIED;
			av1ColorConfig.transfer_characteristics = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_UNSPECIFIED;
			av1ColorConfig.matrix_coefficients = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_UNSPECIFIED;
			av1ColorConfig.chroma_sample_position = STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_UNKNOWN;

			const OaAv1SequenceHeaderInfo* useSeq = InAv1SeqHeader ? InAv1SeqHeader : nullptr;
			av1SequenceHeader.seq_profile = STD_VIDEO_AV1_PROFILE_MAIN;
			if (useSeq) {
				av1SequenceHeader.seq_profile = useSeq->SeqProfile;
				av1ColorConfig = useSeq->ColorConfig;
				av1SequenceHeader.frame_width_bits_minus_1 = static_cast<uint8_t>(useSeq->FrameWidthBitsMinus1);
				av1SequenceHeader.frame_height_bits_minus_1 = static_cast<uint8_t>(useSeq->FrameHeightBitsMinus1);
				av1SequenceHeader.max_frame_width_minus_1 = static_cast<uint16_t>(useSeq->MaxFrameWidthMinus1);
				av1SequenceHeader.max_frame_height_minus_1 = static_cast<uint16_t>(useSeq->MaxFrameHeightMinus1);
				av1SequenceHeader.seq_force_screen_content_tools = static_cast<uint8_t>(useSeq->SeqForceScreenContentTools);
				av1SequenceHeader.seq_force_integer_mv = static_cast<uint8_t>(useSeq->SeqForceIntegerMv);
				av1SequenceHeader.flags.still_picture = useSeq->StillPicture ? 1 : 0;
				av1SequenceHeader.flags.reduced_still_picture_header = useSeq->ReducedStillPictureHeader ? 1 : 0;
				av1SequenceHeader.flags.use_128x128_superblock = useSeq->Use128x128Superblock ? 1 : 0;
				av1SequenceHeader.flags.enable_filter_intra = useSeq->EnableFilterIntra ? 1 : 0;
				av1SequenceHeader.flags.enable_intra_edge_filter = useSeq->EnableIntraEdgeFilter ? 1 : 0;
				av1SequenceHeader.flags.enable_interintra_compound = useSeq->EnableInterIntraCompound ? 1 : 0;
				av1SequenceHeader.flags.enable_masked_compound = useSeq->EnableMaskedCompound ? 1 : 0;
				av1SequenceHeader.flags.enable_warped_motion = useSeq->EnableWarpedMotion ? 1 : 0;
				av1SequenceHeader.flags.enable_dual_filter = useSeq->EnableDualFilter ? 1 : 0;
				av1SequenceHeader.flags.enable_order_hint = useSeq->EnableOrderHint ? 1 : 0;
				av1SequenceHeader.flags.enable_jnt_comp = useSeq->EnableJntComp ? 1 : 0;
				av1SequenceHeader.flags.enable_ref_frame_mvs = useSeq->EnableRefFrameMvs ? 1 : 0;
				av1SequenceHeader.flags.enable_superres = useSeq->EnableSuperres ? 1 : 0;
				av1SequenceHeader.flags.enable_cdef = useSeq->EnableCdef ? 1 : 0;
				av1SequenceHeader.flags.enable_restoration = useSeq->EnableRestoration ? 1 : 0;
				av1SequenceHeader.flags.film_grain_params_present = useSeq->FilmGrainParamsPresent ? 1 : 0;
				av1SequenceHeader.flags.frame_id_numbers_present_flag = useSeq->FrameIdNumbersPresent ? 1 : 0;
				av1SequenceHeader.flags.timing_info_present_flag = useSeq->TimingInfoPresent ? 1 : 0;
				av1SequenceHeader.flags.initial_display_delay_present_flag = useSeq->InitialDisplayDelayPresent ? 1 : 0;
				av1TimingInfo = useSeq->TimingInfo;
				av1SequenceHeader.delta_frame_id_length_minus_2 = useSeq->DeltaFrameIdLengthMinus2;
				av1SequenceHeader.additional_frame_id_length_minus_1 = useSeq->AdditionalFrameIdLengthMinus1;
				av1SequenceHeader.order_hint_bits_minus_1 = useSeq->EnableOrderHint ? static_cast<uint8_t>(useSeq->OrderHintBits > 0 ? useSeq->OrderHintBits - 1 : 0) : 0;
			} else {
				av1SequenceHeader.frame_width_bits_minus_1 = bitCountMinusOne(InProfile.Width);
				av1SequenceHeader.frame_height_bits_minus_1 = bitCountMinusOne(InProfile.Height);
				av1SequenceHeader.max_frame_width_minus_1 = static_cast<uint16_t>(InProfile.Width - 1u);
				av1SequenceHeader.max_frame_height_minus_1 = static_cast<uint16_t>(InProfile.Height - 1u);
				av1SequenceHeader.seq_force_integer_mv = STD_VIDEO_AV1_SELECT_INTEGER_MV;
				av1SequenceHeader.seq_force_screen_content_tools = STD_VIDEO_AV1_SELECT_SCREEN_CONTENT_TOOLS;
			}
			av1SequenceHeader.pColorConfig = &av1ColorConfig;
			av1SequenceHeader.pTimingInfo = av1SequenceHeader.flags.timing_info_present_flag
				? &av1TimingInfo : nullptr;

			OA_LOG_INFO(OaLogComponent::Core,
				"AV1 session params: profile=%u maxW=%u maxH=%u ohBits=%u "
				"cdef=%u lr=%u orderHint=%u refMvs=%u sb128=%u",
				av1SequenceHeader.seq_profile,
				av1SequenceHeader.max_frame_width_minus_1 + 1U,
				av1SequenceHeader.max_frame_height_minus_1 + 1U,
				av1SequenceHeader.order_hint_bits_minus_1,
				av1SequenceHeader.flags.enable_cdef,
				av1SequenceHeader.flags.enable_restoration,
				av1SequenceHeader.flags.enable_order_hint,
				av1SequenceHeader.flags.enable_ref_frame_mvs,
				av1SequenceHeader.flags.use_128x128_superblock);

			av1Info.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_SESSION_PARAMETERS_CREATE_INFO_KHR;
			av1Info.pStdSequenceHeader = &av1SequenceHeader;
			createInfo.pNext = &av1Info;
			break;
		}
	}

	VkResult result = vkCreateVideoSessionParametersKHR(InDevice, &createInfo, nullptr, &OutParameters);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkCreateVideoSessionParametersKHR failed");
	}
	return OaStatus::Ok();
}

OaStatus OaVideoDecoder::UpdateAv1SessionParametersFromSequenceHeader(const OaAv1SequenceHeaderInfo& InSeq) {
	if (Profile_.Codec != OaVideoCodec::AV1) {
		return OaStatus::Ok();
	}
	if (Av1SequenceHeaderUploaded_ && SessionParams_.Handle() != VK_NULL_HANDLE) {
		return OaStatus::Ok();
	}
	if (!Rt_) {
		return OaStatus::Error("Video decoder not fully initialized for AV1 params");
	}

	// Per NVIDIA vk_video_samples and Vulkan spec for AV1, the session parameters
	// object must be created with the actual StdVideoAV1SequenceHeader from the
	// bitstream (not a profile placeholder). Decoder Create() builds a placeholder
	// (nullptr seq header → default values) so the session can be constructed before
	// any bitstream arrives. Destroy the placeholder now and recreate with the real
	// header below.
	if (SessionParams_.Handle() != VK_NULL_HANDLE) {
		SessionParams_.Destroy();
	}

	auto& vkEngine = static_cast<OaComputeEngine&>(*Rt_);
	VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);
	VkVideoSessionParametersKHR newParams = VK_NULL_HANDLE;
	OaStatus st = CreateDecodeSessionParameters(
		device,
		Profile_.Codec,
		Profile_,
		Session_.Handle(),
		newParams,
		&InSeq);
	if (!st.IsOk()) {
		return st;
	}
	SessionParams_.Attach(vkEngine, newParams);
	Av1SequenceHeaderUploaded_ = true;
	// Real sequence header replaces the create-time placeholder — force a session
	// reset on the next decode so the driver picks up the new StdVideoAV1SequenceHeader.
	VideoSessionInitialized_ = false;
	return OaStatus::Ok();
}

static OaU32 AlignVideoExtent(OaU32 InValue, OaU32 InMinimum, OaU32 InGranularity) {
	const OaU32 granularity = InGranularity == 0 ? 1u : InGranularity;
	OaU32 value = InValue < InMinimum ? InMinimum : InValue;
	const OaU32 remainder = value % granularity;
	if (remainder != 0) {
		value += granularity - remainder;
	}
	return value;
}

static OaU32 GetCodecExtentGranularity(OaVideoCodec InCodec) {
	switch (InCodec) {
		case OaVideoCodec::H264:
			// H.264 coded dimensions are macroblock based.
			return 16U;
		case OaVideoCodec::H265:
		case OaVideoCodec::AV1:
		case OaVideoCodec::VP9:
			// 4:2:0 decode surfaces must be even-sized. H.265 CTB and AV1
			// superblock/tile geometry are carried in the bitstream, so do not
			// force larger codec block sizes here.
			return 2U;
	}
	return 1U;
}



OaVideoDecoder::OaVideoDecoder(OaVideoDecoder&& InOther) noexcept {
	MoveFrom(OaStdMove(InOther));
}

OaVideoDecoder& OaVideoDecoder::operator=(OaVideoDecoder&& InOther) noexcept {
	if (this != &InOther) {
		Destroy();
		MoveFrom(OaStdMove(InOther));
	}
	return *this;
}

OaVideoDecoder::~OaVideoDecoder()
{
	Destroy();
}

OaU32 OaVideoDecoder::GetDpbInUseCount() const noexcept
{
	OaU32 count = 0;
	for (const DpbSlot& slot : DpbSlots_) {
		count += slot.InUse ? 1u : 0u;
	}
	return count;
}

OaU32 OaVideoDecoder::GetDpbReferenceCount() const noexcept
{
	OaU32 count = 0;
	for (const DpbSlot& slot : DpbSlots_) {
		count += (slot.InUse && slot.IsReference) ? 1u : 0u;
	}
	return count;
}

void OaVideoDecoder::MoveFrom(OaVideoDecoder&& InOther) noexcept
{
	// OaVkVideo* wrappers are move-only and own their underlying Vk handles.
	Session_       = OaStdMove(InOther.Session_);
	SessionParams_ = OaStdMove(InOther.SessionParams_);
	Queue_         = OaStdMove(InOther.Queue_);
	CmdBuffers_    = InOther.CmdBuffers_;
	CmdFences_     = InOther.CmdFences_;
	CurrentCbIndex_ = InOther.CurrentCbIndex_;
	TimelineSem_   = OaStdMove(InOther.TimelineSem_);
	TimelineValue_ = InOther.TimelineValue_;
	for (OaU32 i = 0; i < kBitstreamRingSize; ++i) {
		BitstreamRing_[i].Buffer = OaStdMove(InOther.BitstreamRing_[i].Buffer);
		BitstreamRing_[i].Size = InOther.BitstreamRing_[i].Size;
		BitstreamRing_[i].UseValue = InOther.BitstreamRing_[i].UseValue;
		InOther.BitstreamRing_[i].Size = 0;
		InOther.BitstreamRing_[i].UseValue = 0;
	}
	CurrentBitstreamIndex_ = InOther.CurrentBitstreamIndex_;
	CodedWidth_  = InOther.CodedWidth_;
	CodedHeight_ = InOther.CodedHeight_;
	Dpb_             = OaStdMove(InOther.Dpb_);
	DpbImageLayouts_ = InOther.DpbImageLayouts_;
	OutputImages_       = OaStdMove(InOther.OutputImages_);
	OutputViews_        = OaStdMove(InOther.OutputViews_);
	OutputAllocations_  = OaStdMove(InOther.OutputAllocations_);
	OutputImageLayouts_ = InOther.OutputImageLayouts_;
	OutputReuseSemaphores_ = InOther.OutputReuseSemaphores_;
	OutputReuseValues_ = InOther.OutputReuseValues_;
	UseSampleStaging_   = InOther.UseSampleStaging_;
	CopySampleStagingOnVideoQueue_ = InOther.CopySampleStagingOnVideoQueue_;
	SampleImages_       = OaStdMove(InOther.SampleImages_);
	SampleYViews_       = OaStdMove(InOther.SampleYViews_);
	SampleUvViews_      = OaStdMove(InOther.SampleUvViews_);
	SampleAllocations_  = OaStdMove(InOther.SampleAllocations_);
	SampleImageLayouts_ = InOther.SampleImageLayouts_;
	RgbImages_          = OaStdMove(InOther.RgbImages_);
	RgbViews_           = OaStdMove(InOther.RgbViews_);
	RgbAllocations_     = OaStdMove(InOther.RgbAllocations_);
	RgbImageLayouts_    = OaStdMove(InOther.RgbImageLayouts_);
	Parser_       = OaStdMove(InOther.Parser_);
	SpsCache_     = OaStdMove(InOther.SpsCache_);
	PpsCache_     = OaStdMove(InOther.PpsCache_);
	H265VpsCache_ = OaStdMove(InOther.H265VpsCache_);
	H265SpsCache_ = OaStdMove(InOther.H265SpsCache_);
	H265PpsCache_ = OaStdMove(InOther.H265PpsCache_);
	DpbSlots_                    = InOther.DpbSlots_;
	DpbSlotCapacity_             = InOther.DpbSlotCapacity_;
	OutputFrameCapacity_         = InOther.OutputFrameCapacity_;
	SessionParameterUpdateCount_ = InOther.SessionParameterUpdateCount_;
	H264SpsUploaded_ = InOther.H264SpsUploaded_;
	H264PpsUploaded_ = InOther.H264PpsUploaded_;
	H265VpsUploaded_ = InOther.H265VpsUploaded_;
	H265SpsUploaded_ = InOther.H265SpsUploaded_;
	H265PpsUploaded_ = InOther.H265PpsUploaded_;
	Av1SequenceHeaderUploaded_ = InOther.Av1SequenceHeaderUploaded_;
	CurrentFrameNumber_        = InOther.CurrentFrameNumber_;
	LastAllocatedDpbSlot_      = InOther.LastAllocatedDpbSlot_;
	SlotDeviceActivated_       = InOther.SlotDeviceActivated_;
	VideoSessionInitialized_   = InOther.VideoSessionInitialized_;
	PrevPocLsb_                = InOther.PrevPocLsb_;
	PrevPocMsb_                = InOther.PrevPocMsb_;
	H265PrevPocLsb_            = InOther.H265PrevPocLsb_;
	H265PrevPocMsb_            = InOther.H265PrevPocMsb_;
	H265HasPrevPoc_            = InOther.H265HasPrevPoc_;
	CurrentH264FrameNum_       = InOther.CurrentH264FrameNum_;
	CurrentLog2MaxFrameNum_    = InOther.CurrentLog2MaxFrameNum_;
	Vp9BufferToDpbSlot_        = InOther.Vp9BufferToDpbSlot_;
	Vp9BufferExtents_          = InOther.Vp9BufferExtents_;
	Av1RefFrameToDpbSlot_      = InOther.Av1RefFrameToDpbSlot_;
	Av1DpbReferenceInfos_      = InOther.Av1DpbReferenceInfos_;
	YcbcrConversion_    = InOther.YcbcrConversion_;
	YcbcrSampler_       = InOther.YcbcrSampler_;
	YcbcrSamplerNearest_ = InOther.YcbcrSamplerNearest_;
	ConversionPipeline_ = InOther.ConversionPipeline_;
	ReusedRgbaIndex_  = InOther.ReusedRgbaIndex_;
	ReusedRgbaWidth_  = InOther.ReusedRgbaWidth_;
	ReusedRgbaHeight_ = InOther.ReusedRgbaHeight_;
	CachedNv12Image_  = InOther.CachedNv12Image_;
	CachedNv12YViews_ = InOther.CachedNv12YViews_;
	CachedNv12UvViews_= InOther.CachedNv12UvViews_;
	CachedNv12Sampler_= InOther.CachedNv12Sampler_;
	CachedNv12SamplerNearest_ = InOther.CachedNv12SamplerNearest_;
	Profile_ = InOther.Profile_;
	Rt_      = InOther.Rt_;

	InOther.CmdBuffers_.Fill(VK_NULL_HANDLE);
	InOther.CmdFences_.Fill(VK_NULL_HANDLE);
	InOther.CurrentCbIndex_    = 0;
	InOther.TimelineValue_   = 0;
	InOther.CurrentBitstreamIndex_ = 0;
	InOther.CodedWidth_  = 0;
	InOther.CodedHeight_ = 0;
	InOther.DpbSlotCapacity_ = 0;
	InOther.DpbImageLayouts_.Fill(VK_IMAGE_LAYOUT_UNDEFINED);
	InOther.OutputFrameCapacity_ = 0;
	InOther.UseSampleStaging_ = false;
	InOther.CopySampleStagingOnVideoQueue_ = false;
	InOther.SampleImageLayouts_.Fill(VK_IMAGE_LAYOUT_UNDEFINED);
	InOther.OutputImageLayouts_.Fill(VK_IMAGE_LAYOUT_UNDEFINED);
	InOther.OutputReuseSemaphores_.Fill(VK_NULL_HANDLE);
	InOther.OutputReuseValues_.Fill(0);
	InOther.RgbImageLayouts_.Clear();
	InOther.SessionParameterUpdateCount_ = 0;
	InOther.H264SpsUploaded_.Fill(false);
	InOther.H264PpsUploaded_.Fill(false);
	InOther.H265VpsUploaded_.Fill(false);
	InOther.H265SpsUploaded_.Fill(false);
	InOther.H265PpsUploaded_.Fill(false);
	InOther.Av1SequenceHeaderUploaded_ = false;
	InOther.CurrentFrameNumber_ = 0;
	InOther.LastAllocatedDpbSlot_ = -1;
	InOther.SlotDeviceActivated_.Fill(false);
	InOther.VideoSessionInitialized_ = false;
	InOther.PrevPocLsb_ = 0;
	InOther.PrevPocMsb_ = 0;
	InOther.H265PrevPocLsb_ = 0;
	InOther.H265PrevPocMsb_ = 0;
	InOther.H265HasPrevPoc_ = false;
	InOther.CurrentH264FrameNum_ = 0;
	InOther.CurrentLog2MaxFrameNum_ = 4;
	InOther.Vp9BufferToDpbSlot_.Fill(-1);
	InOther.Vp9BufferExtents_.Fill({0, 0});
	InOther.Av1RefFrameToDpbSlot_.Fill(-1);
	InOther.Av1DpbReferenceInfos_.Fill({});
	InOther.YcbcrConversion_ = VK_NULL_HANDLE;
	InOther.YcbcrSampler_ = VK_NULL_HANDLE;
	InOther.YcbcrSamplerNearest_ = VK_NULL_HANDLE;
	InOther.ConversionPipeline_ = VK_NULL_HANDLE;
	InOther.ReusedRgbaIndex_  = -1;
	InOther.ReusedRgbaWidth_  = 0;
	InOther.ReusedRgbaHeight_ = 0;
	InOther.CachedNv12Image_  = VK_NULL_HANDLE;
	InOther.CachedNv12YViews_.Fill(VK_NULL_HANDLE);
	InOther.CachedNv12UvViews_.Fill(VK_NULL_HANDLE);
	InOther.CachedNv12Sampler_ = VK_NULL_HANDLE;
	InOther.CachedNv12SamplerNearest_ = VK_NULL_HANDLE;
	InOther.Profile_ = {};
	InOther.Rt_ = nullptr;
}

OaResult<OaVideoDecodeCapabilities> OaVideoDecoder::QueryDecodeCapabilities(OaEngine& InRt, OaVideoCodec InCodec)
{
	auto& vkEngine = static_cast<OaComputeEngine&>(InRt);
	const auto& sw = vkEngine.Device.Info.Software;
	if (!sw.HasVideoQueue || !sw.HasVideoDecodeQueue || !vkEngine.Device.Queues.HasVideoDecodeQueue) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"Video decode not supported on this device: no queue family exposes "
			"VK_QUEUE_VIDEO_DECODE_BIT_KHR (the decode extensions may be advertised without "
			"a usable queue — on Intel this requires the xe kernel driver, not i915).");
	}
	if (InCodec == OaVideoCodec::H264 && !sw.HasVideoDecodeH264) {
		return OaStatus::Error(OaStatusCode::Unavailable, "VK_KHR_video_decode_h264 is not enabled");
	}
	if (InCodec == OaVideoCodec::H265 && !sw.HasVideoDecodeH265) {
		return OaStatus::Error(OaStatusCode::Unavailable, "VK_KHR_video_decode_h265 is not enabled");
	}
	if (InCodec == OaVideoCodec::AV1 && !sw.HasVideoDecodeAV1) {
		return OaStatus::Error(OaStatusCode::Unavailable, "VK_KHR_video_decode_av1 is not enabled");
	}
	if (InCodec == OaVideoCodec::VP9 && !sw.HasVideoDecodeVP9) {
		return OaStatus::Error(OaStatusCode::Unavailable, "VK_KHR_video_decode_vp9 is not enabled");
	}
	if (!VideoDecodeQueueSupportsCodec(vkEngine.Device.Queues, InCodec)) {
		return OaStatus::Error(
			OaStatusCode::Unavailable,
			"Video decode queue family does not support the requested codec");
	}
	if (!vkGetPhysicalDeviceVideoCapabilitiesKHR) {
		return OaStatus::Error("vkGetPhysicalDeviceVideoCapabilitiesKHR is not loaded");
	}

	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(vkEngine.Device.PhysicalDevice);

	VkVideoDecodeH264ProfileInfoKHR h264 = {};
	VkVideoDecodeH265ProfileInfoKHR h265 = {};
	VkVideoDecodeAV1ProfileInfoKHR av1 = {};
	VkVideoDecodeVP9ProfileInfoKHR vp9 = {};
	VkVideoProfileInfoKHR profile = OaVideoDecoderProfile::BuildDecodeProfile(InCodec, h264, h265, av1, vp9);
	VkVideoCapabilitiesKHR caps = {};
	caps.sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR;
	VkVideoDecodeCapabilitiesKHR decodeCaps = {};
	VkVideoDecodeH264CapabilitiesKHR h264Caps = {};
	VkVideoDecodeH265CapabilitiesKHR h265Caps = {};
	VkVideoDecodeAV1CapabilitiesKHR av1Caps = {};
	VkVideoDecodeVP9CapabilitiesKHR vp9Caps = {};
	AttachCodecCapabilities(InCodec, caps, decodeCaps, h264Caps, h265Caps, av1Caps, vp9Caps);
	VkResult result = vkGetPhysicalDeviceVideoCapabilitiesKHR(phys, &profile, &caps);
	if (result != VK_SUCCESS) {
		return OaStatus::Error("Vulkan Video decode profile is not supported");
	}

	OaVideoDecodeCapabilities out;
	out.Supported = true;
	out.SupportsDpbAndOutputCoincide = (decodeCaps.flags & VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR) != 0;
	out.SupportsDpbAndOutputDistinct = (decodeCaps.flags & VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_DISTINCT_BIT_KHR) != 0;
	out.MaxWidth = caps.maxCodedExtent.width;
	out.MaxHeight = caps.maxCodedExtent.height;
	out.MinWidth = caps.minCodedExtent.width;
	out.MinHeight = caps.minCodedExtent.height;
	out.PictureAccessGranularityWidth = caps.pictureAccessGranularity.width == 0 ? 1u : caps.pictureAccessGranularity.width;
	out.PictureAccessGranularityHeight = caps.pictureAccessGranularity.height == 0 ? 1u : caps.pictureAccessGranularity.height;
	out.MaxDpbSlots = caps.maxDpbSlots;
	out.MaxActiveReferencePictures = caps.maxActiveReferencePictures;
	out.MinBitstreamBufferOffsetAlignment = caps.minBitstreamBufferOffsetAlignment;
	out.MinBitstreamBufferSizeAlignment = caps.minBitstreamBufferSizeAlignment;
	out.DecodeFlags = decodeCaps.flags;
	out.StdHeaderVersion = caps.stdHeaderVersion;

	const VkImageUsageFlags dpbUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR;
	const VkImageUsageFlags dpbSampledUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR | VK_IMAGE_USAGE_SAMPLED_BIT;
	const VkImageUsageFlags dpbTransferUsage =
		VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR |
		VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR |
		VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	const VkImageUsageFlags outputSampledUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR | VK_IMAGE_USAGE_SAMPLED_BIT;

	OA_RETURN_IF_ERROR(QueryVideoFormats(phys, profile, dpbUsage, out.DpbFormats));
	out.SupportsNv12Dpb = HasFormatWithUsage(out.DpbFormats, out.ReferencePictureFormat, dpbUsage);

	OaVec<VkVideoFormatPropertiesKHR> dpbSampledFormats;
	OA_RETURN_IF_ERROR(QueryVideoFormats(phys, profile, dpbSampledUsage, dpbSampledFormats));
	out.SupportsNv12DpbSampled = HasFormatWithUsage(dpbSampledFormats, out.ReferencePictureFormat, dpbSampledUsage);

	OaVec<VkVideoFormatPropertiesKHR> dpbTransferFormats;
	OA_RETURN_IF_ERROR(QueryVideoFormats(phys, profile, dpbTransferUsage, dpbTransferFormats));
	out.SupportsNv12DpbTransferSrc =
		HasFormatWithUsage(dpbTransferFormats, out.ReferencePictureFormat, dpbTransferUsage);

	OA_RETURN_IF_ERROR(QueryVideoFormats(phys, profile, outputSampledUsage, out.OutputFormats));
	out.SupportsNv12OutputSampled = HasFormatWithUsage(out.OutputFormats, out.PictureFormat, outputSampledUsage);

	return out;
}

bool OaVideoDecoder::IsCodecSupported(OaEngine& InRt, OaVideoCodec InCodec)
{
	auto caps = QueryDecodeCapabilities(InRt, InCodec);
	return caps.IsOk() && caps->Supported;
}

OaU32 OaVideoDecoder::GetMaxWidth(OaEngine& InRt, OaVideoCodec InCodec)
{
	auto caps = QueryDecodeCapabilities(InRt, InCodec);
	return caps.IsOk() ? caps->MaxWidth : 0;
}

OaU32 OaVideoDecoder::GetMaxHeight(OaEngine& InRt, OaVideoCodec InCodec)
{
	auto caps = QueryDecodeCapabilities(InRt, InCodec);
	return caps.IsOk() ? caps->MaxHeight : 0;
}

// Create video decoder
OaResult<OaVideoDecoder> OaVideoDecoder::Create(
	OaEngine& InRt,
	const OaVideoProfile& InProfile)
{
	// Ensure codec parsers are registered
	OaVideoCodecRegistry::GetInstance().RegisterAllParsers();

	OaVideoDecoder decoder;
	decoder.Rt_ = &InRt;
	decoder.Profile_ = InProfile;
	decoder.Parser_ = OaVideoCodecRegistry::GetInstance().CreateParser(InProfile.Codec);
	if (!decoder.Parser_) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Unsupported video codec parser");
	}

	auto capsResult = QueryDecodeCapabilities(InRt, InProfile.Codec);
	if (!capsResult.IsOk()) {
		return capsResult.GetStatus();
	}
	const OaVideoDecodeCapabilities& caps = *capsResult;
	if (InProfile.Width == 0 || InProfile.Height == 0 || InProfile.Width > caps.MaxWidth || InProfile.Height > caps.MaxHeight) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Video decode extent is unsupported");
	}
	if (!caps.SupportsNv12Dpb) {
		return OaStatus::Error(OaStatusCode::Unavailable, "Vulkan Video decoder does not expose NV12 DPB support");
	}

	auto& vkEngine = static_cast<OaComputeEngine&>(InRt);

	VkVideoDecodeH264ProfileInfoKHR h264 = {};
	VkVideoDecodeH265ProfileInfoKHR h265 = {};
	VkVideoDecodeAV1ProfileInfoKHR av1 = {};
	VkVideoDecodeVP9ProfileInfoKHR vp9 = {};
	VkVideoProfileInfoKHR profile = OaVideoDecoderProfile::BuildDecodeProfile(InProfile.Codec, h264, h265, av1, vp9);

	// Calculate aligned coded extent
	const OaU32 codecGranularity = GetCodecExtentGranularity(InProfile.Codec);
	const OaU32 widthGranularity = caps.PictureAccessGranularityWidth > codecGranularity
		? caps.PictureAccessGranularityWidth
		: codecGranularity;
	const OaU32 heightGranularity = caps.PictureAccessGranularityHeight > codecGranularity
		? caps.PictureAccessGranularityHeight
		: codecGranularity;
	decoder.CodedWidth_ = AlignVideoExtent(InProfile.Width, caps.MinWidth, widthGranularity);
	decoder.CodedHeight_ = AlignVideoExtent(InProfile.Height, caps.MinHeight, heightGranularity);
	if (decoder.CodedWidth_ > caps.MaxWidth || decoder.CodedHeight_ > caps.MaxHeight) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Aligned video decode extent is unsupported");
	}

	// Calculate DPB slots
	constexpr OaU32 MaxTrackedDpbSlots = 16;
	const OaU32 requestedDpbSlots = InProfile.MaxDpbSlots == 0 ? caps.MaxDpbSlots : InProfile.MaxDpbSlots;
	const OaU32 driverDpbSlots = requestedDpbSlots > caps.MaxDpbSlots ? caps.MaxDpbSlots : requestedDpbSlots;
	const OaU32 maxDpbSlots = driverDpbSlots > MaxTrackedDpbSlots ? MaxTrackedDpbSlots : driverDpbSlots;
	if (maxDpbSlots == 0) {
		return OaStatus::Error(OaStatusCode::Unavailable, "Vulkan Video decoder reports zero DPB slots");
	}
	decoder.DpbSlotCapacity_ = maxDpbSlots;
	const OaU32 maxActiveReferences = maxDpbSlots < caps.MaxActiveReferencePictures
		? maxDpbSlots
		: caps.MaxActiveReferencePictures;
	const OaU32 finalMaxActiveReferences = maxActiveReferences == 0 && caps.MaxActiveReferencePictures > 0 ? 1 : maxActiveReferences;
	OA_LOG_INFO(OaLogComponent::Core,
		"VideoDecoder caps: MaxDpbSlots=%u MaxActiveRefs=%u "
		"DpbAndOutputCoincide=%d DpbAndOutputDistinct=%d "
		"DpbSampled=%d DpbTransferSrc=%d OutputSampled=%d",
		maxDpbSlots, finalMaxActiveReferences,
		static_cast<int>(caps.SupportsDpbAndOutputCoincide ? 1 : 0),
		static_cast<int>(caps.SupportsDpbAndOutputDistinct ? 1 : 0),
		static_cast<int>(caps.SupportsNv12DpbSampled ? 1 : 0),
		static_cast<int>(caps.SupportsNv12DpbTransferSrc ? 1 : 0),
		static_cast<int>(caps.SupportsNv12OutputSampled ? 1 : 0));

	VkExtent2D codedExtent = {decoder.CodedWidth_, decoder.CodedHeight_};

	// Create video session using OaVkVideoSession
	auto sessionResult = OaVkVideoSession::Create(
		vkEngine,
		profile,
		codedExtent,
		caps.PictureFormat,
		caps.ReferencePictureFormat,
		maxDpbSlots,
		finalMaxActiveReferences);
	if (!sessionResult.IsOk()) {
		return sessionResult.GetStatus();
	}
	decoder.Session_ = OaStdMove(sessionResult.GetValue());

	// Create video queue
	auto queueResult = OaVkVideoQueue::Create(vkEngine, OaVkVideoQueue::QueueType::Decode);
	if (!queueResult.IsOk()) {
		return queueResult.GetStatus();
	}
	decoder.Queue_ = OaStdMove(queueResult.GetValue());

	// Create bitstream buffer. We MUST attach the video profile via
	// VkVideoProfileListInfoKHR (otherwise VUID-vkCmdDecodeVideoKHR-pDecodeInfo-07135
	// fires every decode and the driver may produce undefined output, which
	// is what was causing the visible glitches).
	const OaU64 bitstreamSize = 4 * 1024 * 1024; // 4MB per in-flight access unit
	for (OaU32 i = 0; i < kBitstreamRingSize; ++i) {
		auto bitstreamResult = OaVkVideoBitstream::Create(
			vkEngine,
			bitstreamSize,
			OaVkVideoBitstream::Direction::Decoder,
			caps.MinBitstreamBufferOffsetAlignment == 0 ? 1 : caps.MinBitstreamBufferOffsetAlignment,
			caps.MinBitstreamBufferSizeAlignment == 0 ? 1 : caps.MinBitstreamBufferSizeAlignment,
			&profile);
		if (!bitstreamResult.IsOk()) {
			decoder.Destroy();
			return bitstreamResult.GetStatus();
		}
		decoder.BitstreamRing_[i].Buffer = OaStdMove(bitstreamResult.GetValue());
	}
	decoder.CurrentBitstreamIndex_ = 0;

	OaU32 queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(
		static_cast<VkPhysicalDevice>(vkEngine.Device.PhysicalDevice),
		&queueFamilyCount,
		nullptr);
	OaVec<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(
		static_cast<VkPhysicalDevice>(vkEngine.Device.PhysicalDevice),
		&queueFamilyCount,
		queueFamilyProperties.Data());
	const OaU32 videoQueueFamily = vkEngine.Device.Queues.VideoDecodeQueueFamily;
	const bool videoQueueSupportsTransfer =
		videoQueueFamily < queueFamilyCount
		and (queueFamilyProperties[videoQueueFamily].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0;

	// Create DPB using OaVkVideoDpb
	VkImageUsageFlags dpbUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR
		| VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR;

	// --- Resource path selection (capability-gated, topology-aware) ---
	//
	// Mirrors the GEMM capability-gating pattern: detect real caps + queue
	// topology, pick the fast path when it's genuinely available, fall back to
	// a safe path otherwise. The key rule: CoincidentFastStaging requires the
	// video queue to actually have TRANSFER_BIT so the DPB→staging copy can be
	// recorded in the same video submit. When the video queue lacks transfer
	// (common on iGPUs) AND is a different family than compute, cross-queue DPB
	// image copies device-loss Intel ANV — so we fall through to DistinctOutput
	// which avoids DPB copies entirely.
	//
	// See VideoDecoderDeviceCompatibility.md "Capability-Gated Architecture".

	const bool dedicatedVideoQueue =
		videoQueueFamily != vkEngine.Device.Queues.ComputeQueueFamily;

	// AV1 grey-frame fix (see OaVideoImplementationPlan.md §2):
	// AV1 reads the decoder *output* image, not the reconstructed DPB — the
	// reference decoders write the displayable picture to a distinct decode-output
	// even on coincide-capable drivers (film grain forces this), so sampling the DPB
	// yields a flat/grey frame. FFmpeg gives AV1 a distinct-output-capable setup
	// unconditionally (vulkan_decode.c:1341); Vulkan-Video-Samples' displayOut rule
	// reads decodeOut for AV1 (VkVideoDecoder.cpp:1054-1057). The film-grain seq flag
	// is not yet known at Create() time, so mirror FFmpeg and route ALL AV1 to the
	// DistinctComputeConvert path (which samples the decode-output) whenever distinct
	// output is available. Distinct is also the iGPU-safe path, so this cannot regress
	// the device-loss-avoidance behaviour below.
	// NEEDS GPU VERIFICATION: the grey frame reproduces only on NVIDIA coincide.
	const bool av1PreferDistinct =
		InProfile.Codec == OaVideoCodec::AV1
		and caps.SupportsDpbAndOutputDistinct
		and caps.SupportsNv12OutputSampled;

	// CoincidentFastStaging: only safe when the video queue has transfer
	// capability (copy stays on the same queue, no cross-family hazard).
	// When the video queue lacks transfer and is a different family, the
	// cross-queue DPB→staging copy has caused device-loss on Intel ANV.
	const bool useCoincideStaging =
		not av1PreferDistinct
		and caps.SupportsDpbAndOutputCoincide
		and caps.SupportsNv12DpbTransferSrc
		and videoQueueSupportsTransfer;

	// DistinctComputeConvert: decode into a distinct output image, then
	// NV12→RGBA via compute shader on the compute queue. Safe for iGPUs
	// because it avoids DPB image copies entirely — the distinct output
	// image is created with CONCURRENT sharing across video+compute families.
	const bool useDistinctOutput =
		not useCoincideStaging
		and caps.SupportsDpbAndOutputDistinct
		and caps.SupportsNv12OutputSampled;

	// CoincidentComputeStaging: copy the decoded DPB layer into an ordinary
	// NV12 image on the compute queue, then sample the staging image planes.
	// Prefer this over direct DPB sampling when TRANSFER_SRC is available.
	const bool useCoincideComputeStaging =
		not useCoincideStaging
		and not useDistinctOutput
		and caps.SupportsDpbAndOutputCoincide
		and caps.SupportsNv12DpbTransferSrc;

	// DirectCoincidentSampling: sample the DPB directly through a YCbCr
	// conversion sampler (COLOR_BIT aspect, no R8/R8G8 plane views). This is
	// only used when no transfer-based or distinct-output path is available.
	const bool useDirectCoincideSampling =
		not useCoincideStaging
		and not useDistinctOutput
		and not useCoincideComputeStaging
		and caps.SupportsDpbAndOutputCoincide
		and caps.SupportsNv12DpbSampled
		and HasHardwareYCbCrConversion(vkEngine);

	// Determine the resource path enum.
	OaVideoResourcePath resourcePath = OaVideoResourcePath::Unavailable;
	if (useCoincideStaging)              resourcePath = OaVideoResourcePath::CoincidentFastStaging;
	else if (useDistinctOutput)          resourcePath = OaVideoResourcePath::DistinctComputeConvert;
	else if (useCoincideComputeStaging)  resourcePath = OaVideoResourcePath::CoincidentComputeStaging;
	else if (useDirectCoincideSampling)  resourcePath = OaVideoResourcePath::DirectCoincidentSampling;

	if (resourcePath == OaVideoResourcePath::Unavailable) {
		decoder.Destroy();
		return OaStatus::Error(
			OaStatusCode::Unavailable,
			"Vulkan Video device exposes no usable decoded-frame path "
			"(requires coincident DPB transfer/sample or distinct sampled output)");
	}

	decoder.ResourcePath_ = resourcePath;

	if (useDirectCoincideSampling) {
		dpbUsage |= VK_IMAGE_USAGE_SAMPLED_BIT;
	}
	if (useCoincideStaging or useCoincideComputeStaging) {
		dpbUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	}

	OaVkVideoDpb::CreateInfo dpbInfo = {};
	dpbInfo.Profile = profile;
	dpbInfo.Format = caps.ReferencePictureFormat;
	dpbInfo.CodedExtent = codedExtent;
	dpbInfo.MaxDpbSlots = maxDpbSlots;
	dpbInfo.Usage = dpbUsage;

	auto dpbResult = OaVkVideoDpb::Create(vkEngine, dpbInfo);
	if (!dpbResult.IsOk()) {
		return dpbResult.GetStatus();
	}
	decoder.Dpb_ = OaStdMove(dpbResult.GetValue());

	// --- Create path-specific resources ---
	static constexpr std::array<const char*, 5> kResourcePathNames = {{
		"CoincidentFastStaging",
		"DistinctComputeConvert",
		"CoincidentComputeStaging",
		"DirectCoincidentSampling",
		"Unavailable",
	}};

	if (useCoincideStaging or useCoincideComputeStaging) {
		OaStatus stagingStatus = decoder.CreateSampleStagingImages(
			vkEngine,
			profile,
			codedExtent,
			maxDpbSlots);
		if (!stagingStatus.IsOk()) {
			decoder.Destroy();
			return stagingStatus;
		}
		decoder.UseSampleStaging_ = true;
		decoder.CopySampleStagingOnVideoQueue_ =
			useCoincideStaging and videoQueueSupportsTransfer;
	} else if (useDistinctOutput) {
		OaStatus outputStatus = decoder.CreateOutputImages(
			vkEngine,
			profile,
			caps.PictureFormat,
			codedExtent,
			maxDpbSlots);
		if (!outputStatus.IsOk()) {
			decoder.Destroy();
			return outputStatus;
		}
	}

	OA_LOG_INFO(OaLogComponent::Core,
		"VideoDecoder: resource path = %s (videoQF=%u computeQF=%u dedicated=%s transfer=%s "
		"coincide=%s distinct=%s)",
		kResourcePathNames[static_cast<OaU32>(resourcePath)],
		videoQueueFamily,
		vkEngine.Device.Queues.ComputeQueueFamily,
		dedicatedVideoQueue ? "yes" : "no",
		videoQueueSupportsTransfer ? "yes" : "no",
		caps.SupportsDpbAndOutputCoincide ? "yes" : "no",
		caps.SupportsDpbAndOutputDistinct ? "yes" : "no");

	// VP9 uses VK_NULL_HANDLE for videoSessionParameters at BeginCoding.
	// AV1 defers session parameters until the first sequence header OBU is parsed
	// (NVIDIA init_sequence pattern — no placeholder StdVideoAV1SequenceHeader).
	if (InProfile.Codec != OaVideoCodec::VP9 && InProfile.Codec != OaVideoCodec::AV1) {
		VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);
		VkVideoSessionParametersKHR rawParams = VK_NULL_HANDLE;
		OaStatus sessionParametersStatus = CreateDecodeSessionParameters(
			device,
			InProfile.Codec,
			InProfile,
			decoder.Session_.Handle(),
			rawParams,
			nullptr /* AV1 seq provided later on first decode */);
		if (!sessionParametersStatus.IsOk()) {
			decoder.Destroy();
			return sessionParametersStatus;
		}
		decoder.SessionParams_.Attach(vkEngine, rawParams);
	}

	// Allocate two command buffers so decode N and decode N+1 can be in
	// flight concurrently. The video queue pool owns the underlying memory;
	// the fence ring tracks GPU completion per slot.
	for (OaU32 i = 0; i < kCmdBufferCount; ++i) {
		auto cmdBufResult = decoder.Queue_.AllocateCommandBuffer();
		if (!cmdBufResult.IsOk()) {
			decoder.Destroy();
			return cmdBufResult.GetStatus();
		}
		decoder.CmdBuffers_[i] = cmdBufResult.GetValue();

		auto fenceResult = decoder.Queue_.AllocateFence();
		if (!fenceResult.IsOk()) {
			decoder.Destroy();
			return fenceResult.GetStatus();
		}
		decoder.CmdFences_[i] = fenceResult.GetValue();
	}

	// Timeline semaphore for GPU-side completion tracking.
	auto semResult = OaVkTimelineSemaphore::Create(vkEngine.Device, 0);
	if (!semResult.IsOk()) {
		decoder.Destroy();
		return semResult.GetStatus();
	}
	decoder.TimelineSem_ = OaStdMove(*semResult);
	decoder.TimelineValue_ = 0;
	for (OaI32& vp9Slot : decoder.Vp9BufferToDpbSlot_) {
		vp9Slot = -1;
	}
	decoder.Vp9BufferExtents_.Fill({0, 0});
	for (OaI32& av1Slot : decoder.Av1RefFrameToDpbSlot_) {
		av1Slot = -1;
	}

	return decoder;
}

// Decode one frame
// Flush decoder state
OaStatus OaVideoDecoder::Flush()
{
	if (Session_.Handle() == VK_NULL_HANDLE)
	{
		return OaStatus::Ok();
	}

	// Drain all in-flight GPU work before wiping DPB state.
	if (Rt_) {
		auto& vkEngine = static_cast<OaComputeEngine&>(*Rt_);
		VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);

		// DecodeFrame returns after async video-queue submit; wait fences
		// before the timeline or Flush can block while work is still in flight.
		for (OaU32 i = 0; i < kCmdBufferCount; ++i) {
			if (CmdFences_[i] != VK_NULL_HANDLE) {
				const VkResult fenceResult = vkWaitForFences(
					device, 1, &CmdFences_[i], VK_TRUE, UINT64_MAX);
				if (fenceResult != VK_SUCCESS) {
					return OaStatus::Error(
						OaStatusCode::VulkanError,
						"vkWaitForFences failed while flushing video decoder");
				}
			}
		}
		for (BitstreamSlot& slot : BitstreamRing_) {
			if (TimelineSem_.Semaphore != nullptr && slot.UseValue > 0) {
				OA_RETURN_IF_ERROR(TimelineSem_.Wait(vkEngine.Device, slot.UseValue));
			}
		}

		if (TimelineSem_.Semaphore != nullptr && TimelineValue_ > 0) {
			const OaU64 completed = TimelineSem_.GetValue(vkEngine.Device);
			if (completed < TimelineValue_) {
				OA_RETURN_IF_ERROR(TimelineSem_.Wait(vkEngine.Device, TimelineValue_));
			}
		}
	}

	// Drop reference flags on all DPB slots: once flushed, no slot is a
	// valid reference for a subsequent stream segment. We don't try to
	// re-emit late frames here — that's a player-level concern, and the
	// player can read frames via DecodeFrame() up to the EOS before
	// calling Flush().
	for (OaI32 i = 0; i < 16; ++i)
	{
		DpbSlots_[i].InUse = false;
		DpbSlots_[i].IsReference = false;
		DpbSlots_[i].PicOrderCnt = -1;
		DpbSlots_[i].FrameNumber = 0;
	}

	// Decode output and DPB layouts need resetting so the next segment
	// re-issues UNDEFINED → VIDEO_DECODE_DST transitions. Converted RGBA
	// images are persistent caller-facing resources and retain their actual
	// layout across a decoder flush.
	OutputImageLayouts_.Fill(VK_IMAGE_LAYOUT_UNDEFINED);
	DpbImageLayouts_.Fill(VK_IMAGE_LAYOUT_UNDEFINED);

	// Reset frame counter and pending decoded sizes.
	CurrentFrameNumber_ = 0;
	for (auto& slot : BitstreamRing_) {
		slot.Size = 0;
		slot.UseValue = 0;
	}
	CurrentBitstreamIndex_ = 0;
	OutputReuseSemaphores_.Fill(VK_NULL_HANDLE);
	OutputReuseValues_.Fill(0);
	PrevPocLsb_         = 0;
	PrevPocMsb_         = 0;
	H265PrevPocLsb_     = 0;
	H265PrevPocMsb_     = 0;
	H265HasPrevPoc_     = false;
	for (OaI32& vp9Slot : Vp9BufferToDpbSlot_) {
		vp9Slot = -1;
	}
	Vp9BufferExtents_.Fill({0, 0});
	for (OaI32& av1Slot : Av1RefFrameToDpbSlot_) {
		av1Slot = -1;
	}
	// Drop the "session initialized" flag — the next decode will re-issue
	// VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR before submitting any work.
	VideoSessionInitialized_ = false;
	SlotDeviceActivated_.Fill(false);

	// Clear the parameter-set caches. The next stream segment will re-upload
	// SPS/PPS/VPS from its own bitstream.
	ClearParameterSets();
	if (Parser_) {
		Parser_->ClearParameterSets();
	}
	H264SpsUploaded_.Fill(false);
	H264PpsUploaded_.Fill(false);
	H265VpsUploaded_.Fill(false);
	H265SpsUploaded_.Fill(false);
	H265PpsUploaded_.Fill(false);
	Av1SequenceHeaderUploaded_ = false;
	SessionParameterUpdateCount_ = 0;

	// Recreate session parameters object to clear Vulkan driver's internal state
	if (SessionParams_.Handle() != VK_NULL_HANDLE && Rt_) {
		SessionParams_.Destroy();
		auto& vkEngine = static_cast<OaComputeEngine&>(*Rt_);
		VkVideoSessionParametersKHR newParams = VK_NULL_HANDLE;
		VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);
		OaStatus createStatus = CreateDecodeSessionParameters(
			device,
			Profile_.Codec,
			Profile_,
			Session_.Handle(),
			newParams,
			nullptr /* AV1 real seq will be supplied on next DecodeFrame */);
		if (createStatus.IsOk()) {
			SessionParams_.SetHandle(newParams);
		}
	}

	return OaStatus::Ok();
}

OaStatus OaVideoDecoder::WaitForCompletion(OaU64 InTimeoutNs)
{
	if (TimelineSem_.Semaphore == nullptr || TimelineValue_ == 0) {
		return OaStatus::Ok();
	}
	if (!Rt_) {
		return OaStatus::Error("Video decoder not initialized");
	}
	return TimelineSem_.Wait(
		static_cast<OaComputeEngine&>(*Rt_).Device,
		TimelineValue_,
		InTimeoutNs);
}

void OaVideoDecoder::StampFrameReady(OaVideoFrame& OutFrame) const noexcept
{
	OutFrame.ReadySemaphore = &TimelineSem_;
	OutFrame.ReadyValue = TimelineValue_;
}

OaVideoDecoder::VideoCmdSlot OaVideoDecoder::AcquireVideoCmdSlot()
{
	VideoCmdSlot slot;
	slot.cb = CmdBuffers_[CurrentCbIndex_];
	slot.fence = CmdFences_[CurrentCbIndex_];
	if (Rt_ && slot.cb != VK_NULL_HANDLE && slot.fence != VK_NULL_HANDLE) {
		auto& vkEngine = static_cast<OaComputeEngine&>(*Rt_);
		VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);
		VkResult result = vkWaitForFences(
			device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
		if (result != VK_SUCCESS) {
			slot.Status = OaStatus::Error(
				OaStatusCode::VulkanError,
				"vkWaitForFences failed while acquiring video command buffer");
			return slot;
		}
		result = vkResetFences(device, 1, &slot.fence);
		if (result != VK_SUCCESS) {
			slot.Status = OaStatus::Error(
				OaStatusCode::VulkanError,
				"vkResetFences failed while acquiring video command buffer");
			return slot;
		}
		result = vkResetCommandBuffer(slot.cb, 0);
		if (result != VK_SUCCESS) {
			slot.Status = OaStatus::Error(
				OaStatusCode::VulkanError,
				"vkResetCommandBuffer failed while acquiring video command buffer");
		}
	}
	return slot;
}

void OaVideoDecoder::ReleaseVideoCmdSlot()
{
	CurrentCbIndex_ = (CurrentCbIndex_ + 1) % kCmdBufferCount;
}

// Destroy decoder
void OaVideoDecoder::Destroy()
{
	if (!Rt_)
	{
		return;
	}

	auto& vkEngine = static_cast<OaComputeEngine&>(*Rt_);
	VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);

	// Decode, sampled-read transitions, and DPB restores are submitted
	// asynchronously and all signal TimelineSem_. In particular, the DPB
	// restore is queued after the conversion ticket that callers wait on.
	// Destroying the semaphore, command buffers, or DPB after only waiting
	// for conversion can therefore free objects still referenced by the
	// video queue and eventually surface as VK_ERROR_DEVICE_LOST on an
	// unrelated graphics submit.
	if (TimelineSem_.Semaphore != nullptr && TimelineValue_ > 0) {
		const OaStatus waitStatus = TimelineSem_.Wait(
			vkEngine.Device,
			TimelineValue_);
		if (!waitStatus.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"VideoDecoder::Destroy: timeline drain failed: %s",
				waitStatus.ToString().c_str());
		}
	}

	// Drain per-CB fences that actually have a pending submit. After Flush()
	// fences are left signaled; an unsignaled fence with no in-flight work
	// must not be waited on (would block forever).
	for (OaU32 i = 0; i < kCmdBufferCount; ++i) {
		if (CmdFences_[i] == VK_NULL_HANDLE) {
			continue;
		}
		const VkResult fenceStatus = vkGetFenceStatus(device, CmdFences_[i]);
		if (fenceStatus == VK_NOT_READY) {
			vkWaitForFences(device, 1, &CmdFences_[i], VK_TRUE, UINT64_MAX);
		}
	}

	// Destroy fences and free command buffers before destroying the pool.
	for (OaU32 i = 0; i < kCmdBufferCount; ++i) {
		if (CmdFences_[i] != VK_NULL_HANDLE) {
			vkDestroyFence(device, CmdFences_[i], nullptr);
			CmdFences_[i] = VK_NULL_HANDLE;
		}
		if (CmdBuffers_[i] != VK_NULL_HANDLE && Queue_.GetCommandPool() != VK_NULL_HANDLE) {
			vkFreeCommandBuffers(device, Queue_.GetCommandPool(), 1, &CmdBuffers_[i]);
			CmdBuffers_[i] = VK_NULL_HANDLE;
		}
	}
	CurrentCbIndex_ = 0;

	TimelineSem_.Destroy(vkEngine.Device);
	TimelineValue_ = 0;
	OutputReuseSemaphores_.Fill(VK_NULL_HANDLE);
	OutputReuseValues_.Fill(0);

	// Destroy OaVkVideo layer objects
	Session_.Destroy();
	SessionParams_.Destroy();
	Queue_.Destroy();
	for (auto& slot : BitstreamRing_) {
		slot.Buffer.Destroy();
		slot.Size = 0;
		slot.UseValue = 0;
	}
	CurrentBitstreamIndex_ = 0;
	Dpb_.Destroy();

	if (ConversionPipeline_) {
		vkDestroyPipeline(device, ConversionPipeline_, nullptr);
		ConversionPipeline_ = VK_NULL_HANDLE;
	}
	if (YcbcrSampler_) {
		vkDestroySampler(device, YcbcrSampler_, nullptr);
		YcbcrSampler_ = VK_NULL_HANDLE;
	}
	if (YcbcrSamplerNearest_) {
		vkDestroySampler(device, YcbcrSamplerNearest_, nullptr);
		YcbcrSamplerNearest_ = VK_NULL_HANDLE;
	}
	if (YcbcrConversion_) {
		vkDestroySamplerYcbcrConversion(device, YcbcrConversion_, nullptr);
		YcbcrConversion_ = VK_NULL_HANDLE;
	}
	for (VkImageView& view : CachedNv12YViews_) {
		if (view) { vkDestroyImageView(device, view, nullptr); view = VK_NULL_HANDLE; }
	}
	for (VkImageView& view : CachedNv12UvViews_) {
		if (view) { vkDestroyImageView(device, view, nullptr); view = VK_NULL_HANDLE; }
	}
	if (CachedNv12Sampler_) {
		vkDestroySampler(device, CachedNv12Sampler_, nullptr);
		CachedNv12Sampler_ = VK_NULL_HANDLE;
	}
	if (CachedNv12SamplerNearest_) {
		vkDestroySampler(device, CachedNv12SamplerNearest_, nullptr);
		CachedNv12SamplerNearest_ = VK_NULL_HANDLE;
	}
	CachedNv12Image_ = VK_NULL_HANDLE;
	ReusedRgbaIndex_ = -1;
	ReusedRgbaWidth_ = 0;
	ReusedRgbaHeight_ = 0;

	// Destroy old-style resources (RGB images, output images - keeping for now)
	for (VkImageView view : OutputViews_) {
		if (view) {
			vkDestroyImageView(device, view, nullptr);
		}
	}
	OutputViews_.Clear();
	for (VkImageView view : RgbViews_) {
		if (view) {
			vkDestroyImageView(device, view, nullptr);
		}
	}
	RgbViews_.Clear();
	for (OaUsize i = 0; i < OutputImages_.Size(); ++i) {
		VkImage image = OutputImages_[i];
		void* allocation = i < OutputAllocations_.Size() ? OutputAllocations_[i] : nullptr;
		if (image && allocation) {
			OaVmaDestroyImage(
				static_cast<OaVmaAllocator>(vkEngine.Allocator.Allocator),
				image,
				static_cast<OaVmaAllocation>(allocation));
		}
	}
	OutputImages_.Clear();
	OutputAllocations_.Clear();
	OutputFrameCapacity_ = 0;
	OutputImageLayouts_.Fill(VK_IMAGE_LAYOUT_UNDEFINED);
	for (VkImageView view : SampleYViews_) {
		if (view) {
			vkDestroyImageView(device, view, nullptr);
		}
	}
	SampleYViews_.Clear();
	for (VkImageView view : SampleUvViews_) {
		if (view) {
			vkDestroyImageView(device, view, nullptr);
		}
	}
	SampleUvViews_.Clear();
	for (OaUsize i = 0; i < SampleImages_.Size(); ++i) {
		VkImage image = SampleImages_[i];
		void* allocation = i < SampleAllocations_.Size() ? SampleAllocations_[i] : nullptr;
		if (image && allocation) {
			OaVmaDestroyImage(
				static_cast<OaVmaAllocator>(vkEngine.Allocator.Allocator),
				image,
				static_cast<OaVmaAllocation>(allocation));
		}
	}
	SampleImages_.Clear();
	SampleAllocations_.Clear();
	SampleImageLayouts_.Fill(VK_IMAGE_LAYOUT_UNDEFINED);
	UseSampleStaging_ = false;
	CopySampleStagingOnVideoQueue_ = false;
	for (OaUsize i = 0; i < RgbImages_.Size(); ++i) {
		VkImage image = RgbImages_[i];
		void* allocation = i < RgbAllocations_.Size() ? RgbAllocations_[i] : nullptr;
		if (image && allocation) {
			OaVmaDestroyImage(
				static_cast<OaVmaAllocator>(vkEngine.Allocator.Allocator),
				image,
				static_cast<OaVmaAllocation>(allocation));
		}
	}
	RgbImages_.Clear();
	RgbAllocations_.Clear();
	RgbImageLayouts_.Clear();
	Parser_.Reset();

	// Vk session, parameters, queue (cmd pool), bitstream and DPB are owned
	// by their respective OaVkVideo wrappers above and were already torn
	// down by the .Destroy() calls at the top of this function.
	Rt_ = nullptr;
}
// ============================================================================
// Phase 2.4.1: DPB (Decoded Picture Buffer) Management
// ============================================================================

// Allocate a DPB slot for a new decoded frame
OaI32 OaVideoDecoder::AllocateDpbSlot()
{
	// Strategy: Find first unused slot, or evict oldest non-reference frame

	// 1. Try to find an unused slot — lowest free index first (matches the
	// v0.6.55 known-good allocator). A round-robin variant was tried during the
	// Phase C cleanup on the theory that spreading allocations across all 16
	// layers keeps host/device DPB association in sync longer; in practice it
	// reused layers in an order the H.264 reference/eviction bookkeeping did not
	// expect and reintroduced playback stutter/motion glitches. Deterministic
	// lowest-free reuse keeps the active reference set tight and predictable.
	for (OaI32 i = 0; i < 16; ++i)
	{
		if (!DpbSlots_[i].InUse)
		{
			DpbSlots_[i].InUse = true;
			DpbSlots_[i].FrameNumber = CurrentFrameNumber_;
			DpbSlots_[i].IsReference = false;
			LastAllocatedDpbSlot_ = i;
			return i;
		}
	}

	// 2. All slots in use - find oldest non-reference frame to evict
	OaI32 oldestSlot = -1;
	OaU32 oldestFrameNumber = CurrentFrameNumber_;

	for (OaI32 i = 0; i < 16; ++i)
	{
		if (!DpbSlots_[i].IsReference && DpbSlots_[i].FrameNumber < oldestFrameNumber)
		{
			oldestSlot = i;
			oldestFrameNumber = DpbSlots_[i].FrameNumber;
		}
	}

	// 3. If found non-reference slot, evict it
	if (oldestSlot >= 0)
	{
		DpbSlots_[oldestSlot].InUse = true;
		DpbSlots_[oldestSlot].FrameNumber = CurrentFrameNumber_;
		DpbSlots_[oldestSlot].IsReference = false;
		return oldestSlot;
	}

	// 4. All slots are reference frames - overflow
	return -1;
}
