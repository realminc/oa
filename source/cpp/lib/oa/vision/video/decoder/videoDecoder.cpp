// OA Vision — hardware Video Decoder Implementation
// VK_KHR_video_decode_h264 / h265 / av1
// Zero-copy: compressed bitstream → vkImage (NV12) → compute shader

#include <oa/vision/videoDecoder.h>
#include "videoDecoderImpl.h"
#include <oa/core/log.h>
#include <oa/vision/fnImage.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/eventAccess.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/deviceAccess.h>
#include <array>
#include <oa/runtime/init.h>
#include <oa/runtime/oaVma.h>
#include <oa/core/fnMatrix.h>
#include "oa/runtime/engine/borrowedServiceRetirement.h"
#include "../codec/nalParser.h"
#include "../codec/codecRegistry.h"
#include "../codec/vcpH265.h"
#include "../codec/vcpAv1.h"
#include "videoDecoderProfile.h"

static oa::F32 clampUnit(oa::F32 inValue)
{
	if (inValue < 0.0f) return 0.0f;
	if (inValue > 1.0f) return 1.0f;
	return inValue;
}

static oa::Status closeDecoderAfterCreateFailure(
	oa::VideoDecoder& inDecoder,
	const oa::Status& inCreateFailure)
{
	const oa::Status closeStatus = inDecoder.close();
	if (closeStatus.isOk()) return inCreateFailure;

	oa::String message = "video decoder creation failed: ";
	message += inCreateFailure.toString();
	message += "; cleanup also failed: ";
	message += closeStatus.toString();
	return oa::Status::error(closeStatus.getCode(), oa::move(message));
}

static VkSamplerYcbcrModelConversion toVkYcbcrModel(oa::YCbCrModel inColorSpace, oa::U32 inWidth, oa::U32 inHeight)
{
	if (inColorSpace == oa::YCbCrModel::BT2020) {
		return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020;
	}
	if (inColorSpace == oa::YCbCrModel::BT709 || inWidth >= 1280 || inHeight >= 720) {
		return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
	}
	return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601;
}

static oa::U32 toVisionColorSpace(oa::YCbCrModel inColorSpace, oa::U32 inWidth, oa::U32 inHeight)
{
	if (inColorSpace == oa::YCbCrModel::BT2020) {
		return 2;
	}
	if (inColorSpace == oa::YCbCrModel::BT709 || inWidth >= 1280 || inHeight >= 720) {
		return 1;
	}
	return 0;
}

static void attachCodecCapabilities(
	oa::VideoCodec inCodec,
	VkVideoCapabilitiesKHR& inOutCaps,
	VkVideoDecodeCapabilitiesKHR& outDecode,
	VkVideoDecodeH264CapabilitiesKHR& outH264,
	VkVideoDecodeH265CapabilitiesKHR& outH265,
	VkVideoDecodeAV1CapabilitiesKHR& outAV1,
	VkVideoDecodeVP9CapabilitiesKHR& outVp9)
{
	outDecode.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR;
	inOutCaps.pNext = &outDecode;
	switch (inCodec) {
		case oa::VideoCodec::H264:
			outH264.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_KHR;
			outDecode.pNext = &outH264;
			break;
		case oa::VideoCodec::H265:
			outH265.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_CAPABILITIES_KHR;
			outDecode.pNext = &outH265;
			break;
		case oa::VideoCodec::AV1:
			outAV1.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_CAPABILITIES_KHR;
			outDecode.pNext = &outAV1;
			break;
		case oa::VideoCodec::VP9:
			outVp9.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_CAPABILITIES_KHR;
			outDecode.pNext = &outVp9;
			break;
	}
}

static VkVideoCodecOperationFlagBitsKHR toVideoDecodeCodecOp(oa::VideoCodec inCodec)
{
	switch (inCodec) {
		case oa::VideoCodec::H264:
			return VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
		case oa::VideoCodec::H265:
			return VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR;
		case oa::VideoCodec::AV1:
			return VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
		case oa::VideoCodec::VP9:
			return VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR;
	}
	return VK_VIDEO_CODEC_OPERATION_NONE_KHR;
}

static bool videoDecodeQueueSupportsCodec(
	const oavk::Queues& inQueues,
	oa::VideoCodec inCodec)
{
	const VkVideoCodecOperationFlagsKHR queueOps = inQueues.videoDecodeCodecOps;
	const VkVideoCodecOperationFlagBitsKHR codecOp = toVideoDecodeCodecOp(inCodec);
	return codecOp != VK_VIDEO_CODEC_OPERATION_NONE_KHR
		&& (queueOps == 0 || (queueOps & codecOp) != 0);
}

static bool hasQualifiedIntelVp9LevelOverride(
	const oavk::Device& inDevice,
	const oa::VideoProfile& inProfile,
	const oa::VideoDecodeCapabilities& inCapabilities)
{
	// Mesa ANV currently writes the literal value 4 to VP9 maxLevel, which is
	// StdVideo level 3.0, even on this TGL device. OA's 1080p Profile-0 path was
	// independently matched against FFmpeg before level-aware admission landed.
	// Keep this exception bounded to that exact qualified device/subset; all
	// other devices and profiles remain governed by the reported maximum.
	return inDevice.info.hardware.vendorId == 0x8086U
		&& inDevice.info.hardware.deviceId == 0x9A49U
		&& inDevice.info.software.driverId
			== static_cast<oa::U32>(VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA)
		&& inCapabilities.maxLevel
			== static_cast<oa::U32>(STD_VIDEO_VP9_LEVEL_3_0)
		&& inProfile.codec == oa::VideoCodec::VP9
		&& inProfile.standardProfile == oa::VideoCodecProfile::Vp9Profile0
		&& inProfile.chromaSubsampling == oa::VideoChromaSubsampling::Yuv420
		&& inProfile.lumaBitDepth == oa::VideoBitDepth::Bit8
		&& inProfile.chromaBitDepth == oa::VideoBitDepth::Bit8
		&& inProfile.width <= 1920U
		&& inProfile.height <= 1080U
		&& inProfile.level
			<= static_cast<oa::U32>(STD_VIDEO_VP9_LEVEL_4_0);
}

static bool hasFormatWithUsage(
	const oa::Vec<VkVideoFormatPropertiesKHR>& inFormats,
	VkFormat inFormat,
	VkImageUsageFlags inUsage)
{
	for (const auto& format : inFormats) {
		if (format.format == inFormat && (format.imageUsageFlags & inUsage) == inUsage) {
			return true;
		}
	}
	return false;
}

static const VkVideoFormatPropertiesKHR* findFormatWithUsage(
	const oa::Vec<VkVideoFormatPropertiesKHR>& inFormats,
	VkFormat inFormat,
	VkImageUsageFlags inUsage)
{
	for (const auto& format : inFormats) {
		if (format.format == inFormat && (format.imageUsageFlags & inUsage) == inUsage) {
			return &format;
		}
	}
	return nullptr;
}

static oa::Status queryVideoFormats(
	const OaVkInstanceTable& inDispatch,
	VkPhysicalDevice inPhys,
	const VkVideoProfileInfoKHR& inProfile,
	VkImageUsageFlags inUsage,
	oa::Vec<VkVideoFormatPropertiesKHR>& outFormats)
{
	if (!inDispatch.vkGetPhysicalDeviceVideoFormatPropertiesKHR) {
		return oa::Status::error("vkGetPhysicalDeviceVideoFormatPropertiesKHR is not loaded");
	}

	VkVideoProfileListInfoKHR profileList = {};
	profileList.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
	profileList.profileCount = 1;
	profileList.pProfiles = &inProfile;

	VkPhysicalDeviceVideoFormatInfoKHR formatInfo = {};
	formatInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR;
	formatInfo.pNext = &profileList;
	formatInfo.imageUsage = inUsage;

	for (oa::U32 attempt = 0U; attempt < 3U; ++attempt) {
		oa::U32 formatCount = 0U;
		VkResult result = inDispatch.vkGetPhysicalDeviceVideoFormatPropertiesKHR(
			inPhys, &formatInfo, &formatCount, nullptr);
		if (result == VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR) {
			outFormats.clear();
			return oa::Status::ok();
		}
		if (result != VK_SUCCESS) {
			return oa::Status::error("Failed to query vulkan Video format count");
		}

		outFormats.resize(formatCount);
		for (auto& format : outFormats) {
			format = {};
			format.sType = VK_STRUCTURE_TYPE_VIDEO_FORMAT_PROPERTIES_KHR;
		}
		if (formatCount == 0U) {
			return oa::Status::ok();
		}

		result = inDispatch.vkGetPhysicalDeviceVideoFormatPropertiesKHR(
			inPhys, &formatInfo, &formatCount, outFormats.data());
		if (result == VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR) {
			outFormats.clear();
			return oa::Status::ok();
		}
		if (result == VK_INCOMPLETE) {
			continue;
		}
		if (result != VK_SUCCESS) {
			outFormats.clear();
			return oa::Status::error("Failed to query vulkan Video formats");
		}
		outFormats.resize(formatCount);
		return oa::Status::ok();
	}
	outFormats.clear();
	return oa::Status::error("vulkan Video format enumeration did not stabilize");
}

static oa::Status createDecodeSessionParameters(
	const oavk::Device& inDevice,
	oa::VideoCodec inCodec,
	const oa::VideoProfile& inProfile,
	VkVideoSessionKHR inSession,
	VkVideoSessionParametersKHR& outParameters,
	const oa::Av1SequenceHeaderInfo* inAv1SeqHeader = nullptr
) {
	if (!inDevice.deviceDispatch.vkCreateVideoSessionParametersKHR) {
		return oa::Status::error("vkCreateVideoSessionParametersKHR is not loaded");
	}

	VkVideoSessionParametersCreateInfoKHR createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR;
	createInfo.videoSession = inSession;
	createInfo.videoSessionParametersTemplate = VK_NULL_HANDLE;

	VkVideoDecodeH264SessionParametersCreateInfoKHR h264Info = {};
	VkVideoDecodeH265SessionParametersCreateInfoKHR h265Info = {};
	VkVideoDecodeAV1SessionParametersCreateInfoKHR av1Info = {};
	StdVideoAV1ColorConfig av1ColorConfig = {};
	StdVideoAV1TimingInfo av1TimingInfo = {};
	StdVideoAV1SequenceHeader av1SequenceHeader = {};

	auto bitCountMinusOne = [](oa::U32 value) -> oa::U8 {
		oa::U32 bits = 0;
		for (oa::U32 tmp = value > 0 ? value - 1u : 0u; tmp != 0; tmp >>= 1u) {
			++bits;
		}
		return static_cast<oa::U8>(bits > 0 ? bits - 1u : 0u);
	};

	switch (inCodec) {
		case oa::VideoCodec::H264:
			h264Info.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR;
			h264Info.maxStdSPSCount = 32;
			h264Info.maxStdPPSCount = 256;
			h264Info.pParametersAddInfo = nullptr;
			createInfo.pNext = &h264Info;
			break;
		case oa::VideoCodec::H265:
			h265Info.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR;
			h265Info.maxStdVPSCount = 16;
			h265Info.maxStdSPSCount = 32;
			h265Info.maxStdPPSCount = 256;
			h265Info.pParametersAddInfo = nullptr;
			createInfo.pNext = &h265Info;
			break;
		case oa::VideoCodec::AV1:
		{
			av1ColorConfig.BitDepth = 8;
			av1ColorConfig.subsampling_x = 1;
			av1ColorConfig.subsampling_y = 1;
			av1ColorConfig.color_primaries = STD_VIDEO_AV1_COLOR_PRIMARIES_UNSPECIFIED;
			av1ColorConfig.transfer_characteristics = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_UNSPECIFIED;
			av1ColorConfig.matrix_coefficients = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_UNSPECIFIED;
			av1ColorConfig.chroma_sample_position = STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_UNKNOWN;

			const oa::Av1SequenceHeaderInfo* useSeq = inAv1SeqHeader ? inAv1SeqHeader : nullptr;
			av1SequenceHeader.seq_profile = STD_VIDEO_AV1_PROFILE_MAIN;
			if (useSeq) {
				av1SequenceHeader.seq_profile = useSeq->seqProfile;
				av1ColorConfig = useSeq->colorConfig;
				av1SequenceHeader.frame_width_bits_minus_1 = static_cast<uint8_t>(useSeq->frameWidthBitsMinus1);
				av1SequenceHeader.frame_height_bits_minus_1 = static_cast<uint8_t>(useSeq->frameHeightBitsMinus1);
				av1SequenceHeader.max_frame_width_minus_1 = static_cast<uint16_t>(useSeq->maxFrameWidthMinus1);
				av1SequenceHeader.max_frame_height_minus_1 = static_cast<uint16_t>(useSeq->maxFrameHeightMinus1);
				av1SequenceHeader.seq_force_screen_content_tools = static_cast<uint8_t>(useSeq->seqForceScreenContentTools);
				av1SequenceHeader.seq_force_integer_mv = static_cast<uint8_t>(useSeq->seqForceIntegerMv);
				av1SequenceHeader.flags.still_picture = useSeq->stillPicture ? 1 : 0;
				av1SequenceHeader.flags.reduced_still_picture_header = useSeq->reducedStillPictureHeader ? 1 : 0;
				av1SequenceHeader.flags.use_128x128_superblock = useSeq->use128x128Superblock ? 1 : 0;
				av1SequenceHeader.flags.enable_filter_intra = useSeq->enableFilterIntra ? 1 : 0;
				av1SequenceHeader.flags.enable_intra_edge_filter = useSeq->enableIntraEdgeFilter ? 1 : 0;
				av1SequenceHeader.flags.enable_interintra_compound = useSeq->enableInterIntraCompound ? 1 : 0;
				av1SequenceHeader.flags.enable_masked_compound = useSeq->enableMaskedCompound ? 1 : 0;
				av1SequenceHeader.flags.enable_warped_motion = useSeq->enableWarpedMotion ? 1 : 0;
				av1SequenceHeader.flags.enable_dual_filter = useSeq->enableDualFilter ? 1 : 0;
				av1SequenceHeader.flags.enable_order_hint = useSeq->enableOrderHint ? 1 : 0;
				av1SequenceHeader.flags.enable_jnt_comp = useSeq->enableJntComp ? 1 : 0;
				av1SequenceHeader.flags.enable_ref_frame_mvs = useSeq->enableRefFrameMvs ? 1 : 0;
				av1SequenceHeader.flags.enable_superres = useSeq->enableSuperres ? 1 : 0;
				av1SequenceHeader.flags.enable_cdef = useSeq->enableCdef ? 1 : 0;
				av1SequenceHeader.flags.enable_restoration = useSeq->enableRestoration ? 1 : 0;
				av1SequenceHeader.flags.film_grain_params_present = useSeq->filmGrainParamsPresent ? 1 : 0;
				av1SequenceHeader.flags.frame_id_numbers_present_flag = useSeq->frameIdNumbersPresent ? 1 : 0;
				av1SequenceHeader.flags.timing_info_present_flag = useSeq->timingInfoPresent ? 1 : 0;
				av1SequenceHeader.flags.initial_display_delay_present_flag = useSeq->initialDisplayDelayPresent ? 1 : 0;
				av1TimingInfo = useSeq->timingInfo;
				av1SequenceHeader.delta_frame_id_length_minus_2 = useSeq->deltaFrameIdLengthMinus2;
				av1SequenceHeader.additional_frame_id_length_minus_1 = useSeq->additionalFrameIdLengthMinus1;
				av1SequenceHeader.order_hint_bits_minus_1 = useSeq->enableOrderHint ? static_cast<uint8_t>(useSeq->orderHintBits > 0 ? useSeq->orderHintBits - 1 : 0) : 0;
			} else {
				av1SequenceHeader.frame_width_bits_minus_1 = bitCountMinusOne(inProfile.width);
				av1SequenceHeader.frame_height_bits_minus_1 = bitCountMinusOne(inProfile.height);
				av1SequenceHeader.max_frame_width_minus_1 = static_cast<uint16_t>(inProfile.width - 1u);
				av1SequenceHeader.max_frame_height_minus_1 = static_cast<uint16_t>(inProfile.height - 1u);
				av1SequenceHeader.seq_force_integer_mv = STD_VIDEO_AV1_SELECT_INTEGER_MV;
				av1SequenceHeader.seq_force_screen_content_tools = STD_VIDEO_AV1_SELECT_SCREEN_CONTENT_TOOLS;
			}
			av1SequenceHeader.pColorConfig = &av1ColorConfig;
			av1SequenceHeader.pTimingInfo = av1SequenceHeader.flags.timing_info_present_flag
				? &av1TimingInfo : nullptr;

			OaLogInfo(oa::LogComponent::Video,
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

	VkResult result = inDevice.deviceDispatch.vkCreateVideoSessionParametersKHR(
		static_cast<VkDevice>(inDevice.device),
		&createInfo, nullptr, &outParameters);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkCreateVideoSessionParametersKHR failed");
	}
	return oa::Status::ok();
}

oa::Status oa::VideoDecoder::updateAv1SessionParametersFromSequenceHeader(const oa::Av1SequenceHeaderInfo& inSeq) {
	if (impl_->profile.codec != oa::VideoCodec::AV1) {
		return oa::Status::ok();
	}
	if (impl_->av1SequenceHeaderUploaded && impl_->sessionParameters.handle() != VK_NULL_HANDLE) {
		return oa::Status::ok();
	}
	if (!impl_->engine) {
		return oa::Status::error("Video decoder not fully initialized for AV1 params");
	}

	// Per NVIDIA vk_video_samples and vulkan spec for AV1, the session parameters
	// object must be created with the actual StdVideoAV1SequenceHeader from the
	// bitstream (not a profile placeholder). Decoder create() builds a placeholder
	// (nullptr seq header → default values) so the session can be constructed before
	// any bitstream arrives. destroy the placeholder now and recreate with the real
	// header below.
	if (impl_->sessionParameters.handle() != VK_NULL_HANDLE) {
		impl_->sessionParameters.destroy();
	}

	auto& vkEngine = *impl_->engine;
	VkVideoSessionParametersKHR newParams = VK_NULL_HANDLE;
	oa::Status st = createDecodeSessionParameters(
		oa::EngineDeviceAccess::get(vkEngine),
		impl_->profile.codec,
		impl_->profile,
		impl_->session.handle(),
		newParams,
		&inSeq);  // impl_->profile is PascalCase internal — not changed here
	if (!st.isOk()) {
		return st;
	}
	impl_->sessionParameters.attach(vkEngine, newParams);
	impl_->av1SequenceHeaderUploaded = true;
	// Real sequence header replaces the create-time placeholder — force a session
	// reset on the next decode so the driver picks up the new StdVideoAV1SequenceHeader.
	impl_->videoSessionInitialized = false;
	return oa::Status::ok();
}

static oa::U32 alignVideoExtent(oa::U32 inValue, oa::U32 inMinimum, oa::U32 inGranularity) {
	const oa::U32 granularity = inGranularity == 0 ? 1u : inGranularity;
	oa::U32 value = inValue < inMinimum ? inMinimum : inValue;
	const oa::U32 remainder = value % granularity;
	if (remainder != 0) {
		value += granularity - remainder;
	}
	return value;
}

static oa::U32 getCodecExtentGranularity(oa::VideoCodec inCodec) {
	switch (inCodec) {
		case oa::VideoCodec::H264:
			// H.264 coded dimensions are macroblock based.
			return 16U;
		case oa::VideoCodec::H265:
		case oa::VideoCodec::AV1:
		case oa::VideoCodec::VP9:
			// 4:2:0 decode surfaces must be even-sized. H.265 CTB and AV1
			// superblock/tile geometry are carried in the bitstream, so do not
			// force larger codec block sizes here.
			return 2U;
	}
	return 1U;
}


oa::VideoDecoder::Impl::~Impl() = default;


oa::VideoDecoder::VideoDecoder()
	: impl_(oa::makeUnique<Impl>()) {}


oa::VideoDecoder::VideoDecoder(oa::VideoDecoder&& inOther) noexcept
	: impl_(oa::move(inOther.impl_)) {}

oa::VideoDecoder& oa::VideoDecoder::operator=(oa::VideoDecoder&& inOther) noexcept {
	if (this != &inOther) {
		abandon_();
		moveFrom(oa::move(inOther));
	}
	return *this;
}

oa::VideoDecoder::~VideoDecoder()
{
	abandon_();
}

void oa::VideoDecoder::abandon_() noexcept {
	if (not impl_ or impl_->engine == nullptr) return;
	oa::Engine* engine = impl_->engine;
	auto retired = oa::makeUnique<oa::VideoDecoder>(oa::move(*this));
	oa::BorrowedServiceRetirement::retire(
		*engine,
		retired.release(),
		&oa::VideoDecoder::completeRetired_,
		&oa::VideoDecoder::releaseRetired_);
}

oa::Status oa::VideoDecoder::completeRetired_(void* inPayload) {
	auto* decoder = static_cast<oa::VideoDecoder*>(inPayload);
	return decoder ? decoder->close() : oa::Status::ok();
}

void oa::VideoDecoder::releaseRetired_(void* inPayload) {
	oa::UniquePtr<oa::VideoDecoder> decoder(
		static_cast<oa::VideoDecoder*>(inPayload));
}

bool oa::VideoDecoder::isInitialized() const noexcept
{
	return impl_ and impl_->session.handle() != VK_NULL_HANDLE;
}

oa::Engine* oa::VideoDecoder::getEngine() const noexcept
{
	return impl_ ? impl_->engine : nullptr;
}

bool oa::VideoDecoder::hasSessionParameters() const noexcept
{
	return impl_ and impl_->sessionParameters.handle() != VK_NULL_HANDLE;
}

oa::U32 oa::VideoDecoder::getSessionParameterUpdateCount() const noexcept
{
	return impl_ ? impl_->sessionParameterUpdateCount : 0U;
}

oa::U32 oa::VideoDecoder::getDpbSlotCapacity() const noexcept
{
	return impl_ ? impl_->dpbSlotCapacity : 0U;
}

oa::U32 oa::VideoDecoder::getOutputFrameCapacity() const noexcept
{
	return impl_ ? impl_->outputFrameCapacity : 0U;
}

oa::U32 oa::VideoDecoder::getOutputViewCount() const noexcept
{
	return impl_ ? static_cast<oa::U32>(impl_->outputViews.size()) : 0U;
}

oa::U32 oa::VideoDecoder::getDpbInUseCount() const noexcept
{
	if (not impl_) return 0U;
	oa::U32 count = 0;
	for (const DpbSlot& slot : impl_->dpbSlots) {
		count += slot.inUse ? 1u : 0u;
	}
	return count;
}

oa::U32 oa::VideoDecoder::getDpbReferenceCount() const noexcept
{
	if (not impl_) return 0U;
	oa::U32 count = 0;
	for (const DpbSlot& slot : impl_->dpbSlots) {
		count += (slot.inUse && slot.isReference) ? 1u : 0u;
	}
	return count;
}

oa::U64 oa::VideoDecoder::getCurrentFrameNumber() const noexcept
{
	return impl_ ? impl_->currentFrameNumber : 0U;
}

oa::U32 oa::VideoDecoder::getCodedWidth() const noexcept
{
	return impl_ ? impl_->codedWidth : 0U;
}

oa::U32 oa::VideoDecoder::getCodedHeight() const noexcept
{
	return impl_ ? impl_->codedHeight : 0U;
}

oa::VideoResourcePath oa::VideoDecoder::getResourcePath() const noexcept
{
	return impl_ ? impl_->resourcePath : oa::VideoResourcePath::Unavailable;
}

oa::U64 oa::VideoDecoder::getBitstreamBufferCapacity() const noexcept
{
	return impl_
		? impl_->bitstreamRing[impl_->currentBitstreamIndex].buffer.getCapacity()
		: 0U;
}

oa::U32 oa::VideoDecoder::getCachedSpsCount() const noexcept
{
	return impl_ ? static_cast<oa::U32>(impl_->spsCache.size()) : 0U;
}

oa::U32 oa::VideoDecoder::getCachedPpsCount() const noexcept
{
	return impl_ ? static_cast<oa::U32>(impl_->ppsCache.size()) : 0U;
}

oa::U32 oa::VideoDecoder::getCachedH265VpsCount() const noexcept
{
	return impl_ ? static_cast<oa::U32>(impl_->h265VpsCache.size()) : 0U;
}

oa::U32 oa::VideoDecoder::getCachedH265SpsCount() const noexcept
{
	return impl_ ? static_cast<oa::U32>(impl_->h265SpsCache.size()) : 0U;
}

oa::U32 oa::VideoDecoder::getCachedH265PpsCount() const noexcept
{
	return impl_ ? static_cast<oa::U32>(impl_->h265PpsCache.size()) : 0U;
}

void oa::VideoDecoder::moveFrom(oa::VideoDecoder&& inOther) noexcept
{
	impl_ = oa::move(inOther.impl_);
}

oa::Result<oa::VideoDecodeCapabilities> oa::VideoDecoder::queryDecodeCapabilities(
	oa::Engine& inRt,
	const oa::VideoProfile& inProfile)
{
	auto resolvedResult = oa::videoDecoderProfile::resolveDecodeProfile(inProfile);
	if (not resolvedResult.isOk()) {
		return resolvedResult.getStatus();
	}
	const oa::VideoProfile& resolved = *resolvedResult;
	const oa::VideoCodec codec = resolved.codec;
	auto& vkEngine = inRt;
	const auto& sw = oa::EngineDeviceAccess::get(vkEngine).info.software;
	if (!sw.hasVideoQueue || !sw.hasVideoDecodeQueue || !oa::EngineDeviceAccess::get(vkEngine).queues.hasVideoDecodeQueue) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"Video decode not supported on this device: no queue family exposes "
			"VK_QUEUE_VIDEO_DECODE_BIT_KHR (the decode extensions may be advertised without "
			"a usable queue — on Intel this requires the xe kernel driver, not i915).");
	}
	if (codec == oa::VideoCodec::H264 && !sw.hasVideoDecodeH264) {
		return oa::Status::error(oa::StatusCode::Unavailable, "VK_KHR_video_decode_h264 is not enabled");
	}
	if (codec == oa::VideoCodec::H265 && !sw.hasVideoDecodeH265) {
		return oa::Status::error(oa::StatusCode::Unavailable, "VK_KHR_video_decode_h265 is not enabled");
	}
	if (codec == oa::VideoCodec::AV1 && !sw.hasVideoDecodeAV1) {
		return oa::Status::error(oa::StatusCode::Unavailable, "VK_KHR_video_decode_av1 is not enabled");
	}
	if (codec == oa::VideoCodec::VP9 && !sw.hasVideoDecodeVP9) {
		return oa::Status::error(oa::StatusCode::Unavailable, "VK_KHR_video_decode_vp9 is not enabled");
	}
	if (!videoDecodeQueueSupportsCodec(oa::EngineDeviceAccess::get(vkEngine).queues, codec)) {
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			"Video decode queue family does not support the requested codec");
	}
	const auto& instanceDispatch =
		oa::EngineDeviceAccess::get(vkEngine).instanceDispatch;
	if (!instanceDispatch.vkGetPhysicalDeviceVideoCapabilitiesKHR) {
		return oa::Status::error("vkGetPhysicalDeviceVideoCapabilitiesKHR is not loaded");
	}

	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(oa::EngineDeviceAccess::get(vkEngine).physicalDevice);

	VkVideoDecodeH264ProfileInfoKHR h264 = {};
	VkVideoDecodeH265ProfileInfoKHR h265 = {};
	VkVideoDecodeAV1ProfileInfoKHR av1 = {};
	VkVideoDecodeVP9ProfileInfoKHR vp9 = {};
	auto vkProfileResult = oa::videoDecoderProfile::buildDecodeProfile(resolved, h264, h265, av1, vp9);
	if (not vkProfileResult.isOk()) {
		return vkProfileResult.getStatus();
	}
	const VkVideoProfileInfoKHR& profile = *vkProfileResult;
	VkVideoCapabilitiesKHR caps = {};
	caps.sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR;
	VkVideoDecodeCapabilitiesKHR decodeCaps = {};
	VkVideoDecodeH264CapabilitiesKHR h264Caps = {};
	VkVideoDecodeH265CapabilitiesKHR h265Caps = {};
	VkVideoDecodeAV1CapabilitiesKHR av1Caps = {};
	VkVideoDecodeVP9CapabilitiesKHR vp9Caps = {};
	attachCodecCapabilities(codec, caps, decodeCaps, h264Caps, h265Caps, av1Caps, vp9Caps);
	VkResult result = instanceDispatch.vkGetPhysicalDeviceVideoCapabilitiesKHR(
		phys, &profile, &caps);
	if (result != VK_SUCCESS) {
		if (result == VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR or
			result == VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR or
			result == VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR or
			result == VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR or
			result == VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR) {
			oa::VideoDecodeCapabilities unsupported = {};
			unsupported.profile = resolved;
			unsupported.oaDecodePathImplemented =
				oa::videoDecoderProfile::isDecodePathImplemented(resolved);
			return unsupported;
		}
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			"Failed to query vulkan Video decode capabilities");
	}

	oa::VideoDecodeCapabilities out;
	out.profile = resolved;
	out.hardwareProfileSupported = true;
	out.oaDecodePathImplemented = oa::videoDecoderProfile::isDecodePathImplemented(resolved);
	out.supportsDpbAndOutputCoincide = (decodeCaps.flags & VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR) != 0;
	out.supportsDpbAndOutputDistinct = (decodeCaps.flags & VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_DISTINCT_BIT_KHR) != 0;
	out.maxWidth = caps.maxCodedExtent.width;
	out.maxHeight = caps.maxCodedExtent.height;
	out.minWidth = caps.minCodedExtent.width;
	out.minHeight = caps.minCodedExtent.height;
	out.pictureAccessGranularityWidth = caps.pictureAccessGranularity.width == 0 ? 1u : caps.pictureAccessGranularity.width;
	out.pictureAccessGranularityHeight = caps.pictureAccessGranularity.height == 0 ? 1u : caps.pictureAccessGranularity.height;
	out.maxDpbSlots = caps.maxDpbSlots;
	out.maxActiveReferencePictures = caps.maxActiveReferencePictures;
	out.minBitstreamBufferOffsetAlignment = caps.minBitstreamBufferOffsetAlignment;
	out.minBitstreamBufferSizeAlignment = caps.minBitstreamBufferSizeAlignment;
	out.decodeFlags = decodeCaps.flags;
	out.stdHeaderVersion = caps.stdHeaderVersion;
	switch (codec) {
	case oa::VideoCodec::H264: out.maxLevel = static_cast<oa::U32>(h264Caps.maxLevelIdc); break;
	case oa::VideoCodec::H265: out.maxLevel = static_cast<oa::U32>(h265Caps.maxLevelIdc); break;
	case oa::VideoCodec::AV1: out.maxLevel = static_cast<oa::U32>(av1Caps.maxLevel); break;
	case oa::VideoCodec::VP9: out.maxLevel = static_cast<oa::U32>(vp9Caps.maxLevel); break;
	}

	const VkImageUsageFlags dpbUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR;
	const VkImageUsageFlags dpbSampledUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR | VK_IMAGE_USAGE_SAMPLED_BIT;
	const VkImageUsageFlags dpbTransferUsage =
		VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR |
		VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR |
		VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	const VkImageUsageFlags outputSampledUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR | VK_IMAGE_USAGE_SAMPLED_BIT;

	OA_RETURN_IF_ERROR(queryVideoFormats(
		instanceDispatch, phys, profile, dpbUsage, out.dpbFormats));
	if (out.oaDecodePathImplemented) {
		out.referencePictureFormat = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
		out.pictureFormat = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
	} else {
		if (not out.dpbFormats.empty()) {
			out.referencePictureFormat = out.dpbFormats[0].format;
		}
	}
	out.supportsNv12Dpb = hasFormatWithUsage(out.dpbFormats, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, dpbUsage);

	oa::Vec<VkVideoFormatPropertiesKHR> dpbSampledFormats;
	OA_RETURN_IF_ERROR(queryVideoFormats(
		instanceDispatch, phys, profile, dpbSampledUsage, dpbSampledFormats));
	out.supportsNv12DpbSampled = hasFormatWithUsage(dpbSampledFormats, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, dpbSampledUsage);

	oa::Vec<VkVideoFormatPropertiesKHR> dpbTransferFormats;
	OA_RETURN_IF_ERROR(queryVideoFormats(
		instanceDispatch, phys, profile, dpbTransferUsage, dpbTransferFormats));
	out.supportsNv12DpbTransferSrc =
		hasFormatWithUsage(dpbTransferFormats, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, dpbTransferUsage);

	OA_RETURN_IF_ERROR(queryVideoFormats(
		instanceDispatch, phys, profile, outputSampledUsage, out.outputFormats));
	if (out.pictureFormat == VK_FORMAT_UNDEFINED and not out.outputFormats.empty()) {
		out.pictureFormat = out.outputFormats[0].format;
	}
	out.supportsNv12OutputSampled = hasFormatWithUsage(out.outputFormats, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, outputSampledUsage);
	const bool levelWithinDeviceMaximum =
		not resolved.hasLevel or resolved.level <= out.maxLevel;
	const bool qualifiedLevelOverride =
		not levelWithinDeviceMaximum
		and hasQualifiedIntelVp9LevelOverride(
			oa::EngineDeviceAccess::get(vkEngine), resolved, out);
	const bool levelAccepted =
		levelWithinDeviceMaximum or qualifiedLevelOverride;
	OaLogDebug(oa::LogComponent::Video,
		"VideoDecoder exact profile: codec=%u profile=%u chroma=%u "
		"lumaDepth=%u chromaDepth=%u level=%u hasLevel=%d maxLevel=%u "
		"hardware=%d oaPath=%d nv12Dpb=%d levelAccepted=%d",
		static_cast<unsigned>(resolved.codec),
		static_cast<unsigned>(resolved.standardProfile),
		static_cast<unsigned>(resolved.chromaSubsampling),
		static_cast<unsigned>(resolved.lumaBitDepth),
		static_cast<unsigned>(resolved.chromaBitDepth),
		resolved.level,
		static_cast<int>(resolved.hasLevel),
		out.maxLevel,
		static_cast<int>(out.hardwareProfileSupported),
		static_cast<int>(out.oaDecodePathImplemented),
		static_cast<int>(out.supportsNv12Dpb),
		static_cast<int>(levelAccepted));
	out.supported =
		out.hardwareProfileSupported and out.oaDecodePathImplemented and out.supportsNv12Dpb and levelAccepted;

	return out;
}

oa::Result<oa::VideoDecodeCapabilities> oa::VideoDecoder::queryDecodeCapabilities(oa::Engine& inRt, oa::VideoCodec inCodec)
{
	oa::VideoProfile profile = {};
	profile.codec = inCodec;
	return queryDecodeCapabilities(inRt, profile);
}

// Create video decoder
oa::Result<oa::VideoDecoder> oa::VideoDecoder::create(
	oa::Engine& inRt,
	const oa::VideoProfile& inProfile)
{
	auto resolvedResult = oa::videoDecoderProfile::resolveDecodeProfile(inProfile);
	if (not resolvedResult.isOk()) {
		return resolvedResult.getStatus();
	}
	const oa::VideoProfile& resolved = *resolvedResult;
	// Ensure codec parsers are registered
	oa::VideoCodecRegistry::getInstance().registerAllParsers();

	oa::VideoDecoder decoder;
	decoder.impl_->engine = &inRt;
	decoder.impl_->profile = resolved;
	decoder.impl_->parser = oa::VideoCodecRegistry::getInstance().createParser(resolved.codec);
	if (!decoder.impl_->parser) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Unsupported video codec parser");
	}

	auto capsResult = queryDecodeCapabilities(inRt, resolved);
	if (!capsResult.isOk()) {
		return capsResult.getStatus();
	}
	const oa::VideoDecodeCapabilities& caps = *capsResult;
	if (not caps.hardwareProfileSupported) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"The selected device rejects the exact vulkan Video decode profile");
	}
	if (not caps.oaDecodePathImplemented) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"OA does not yet implement the exact requested video profile path");
	}
	if (not caps.supported) {
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			"The exact video profile, level, or required image format is unavailable");
	}
	if (resolved.hasLevel and resolved.level > caps.maxLevel
		and hasQualifiedIntelVp9LevelOverride(
			oa::EngineDeviceAccess::get(inRt), resolved, caps)) {
		OaLogInfo(oa::LogComponent::Video,
			"Using qualified Intel TGL VP9 level override: "
			"streamLevel=%u driverMaxLevel=%u extent=%ux%u",
			resolved.level,
			caps.maxLevel,
			resolved.width,
			resolved.height);
	}
	if (resolved.width == 0 || resolved.height == 0 or resolved.width > caps.maxWidth ||
		resolved.height > caps.maxHeight) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Video decode extent is unsupported");
	}
	if (!caps.supportsNv12Dpb) {
		return oa::Status::error(oa::StatusCode::Unavailable, "vulkan Video decoder does not expose NV12 DPB support");
	}

	auto& vkEngine = inRt;

	VkVideoDecodeH264ProfileInfoKHR h264 = {};
	VkVideoDecodeH265ProfileInfoKHR h265 = {};
	VkVideoDecodeAV1ProfileInfoKHR av1 = {};
	VkVideoDecodeVP9ProfileInfoKHR vp9 = {};
	auto vkProfileResult = oa::videoDecoderProfile::buildDecodeProfile(resolved, h264, h265, av1, vp9);
	if (not vkProfileResult.isOk()) {
		return vkProfileResult.getStatus();
	}
	const VkVideoProfileInfoKHR& profile = *vkProfileResult;

	// Calculate aligned coded extent
	const oa::U32 codecGranularity = getCodecExtentGranularity(resolved.codec);
	const oa::U32 widthGranularity = caps.pictureAccessGranularityWidth > codecGranularity
		? caps.pictureAccessGranularityWidth
		: codecGranularity;
	const oa::U32 heightGranularity = caps.pictureAccessGranularityHeight > codecGranularity
		? caps.pictureAccessGranularityHeight
		: codecGranularity;
	decoder.impl_->codedWidth = alignVideoExtent(resolved.width, caps.minWidth, widthGranularity);
	decoder.impl_->codedHeight = alignVideoExtent(resolved.height, caps.minHeight, heightGranularity);
	if (decoder.impl_->codedWidth > caps.maxWidth || decoder.impl_->codedHeight > caps.maxHeight) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Aligned video decode extent is unsupported");
	}

	// Calculate DPB slots
	constexpr oa::U32 MaxTrackedDpbSlots = 16;
	const oa::U32 requestedDpbSlots = resolved.maxDpbSlots == 0 ? caps.maxDpbSlots : resolved.maxDpbSlots;
	const oa::U32 driverDpbSlots = requestedDpbSlots > caps.maxDpbSlots ? caps.maxDpbSlots : requestedDpbSlots;
	const oa::U32 maxDpbSlots = driverDpbSlots > MaxTrackedDpbSlots ? MaxTrackedDpbSlots : driverDpbSlots;
	if (maxDpbSlots == 0) {
		return oa::Status::error(oa::StatusCode::Unavailable, "vulkan Video decoder reports zero DPB slots");
	}
	decoder.impl_->dpbSlotCapacity = maxDpbSlots;
	const oa::U32 maxActiveReferences = maxDpbSlots < caps.maxActiveReferencePictures
		? maxDpbSlots
		: caps.maxActiveReferencePictures;
	const oa::U32 finalMaxActiveReferences = maxActiveReferences == 0 && caps.maxActiveReferencePictures > 0 ? 1 : maxActiveReferences;
	OaLogInfo(oa::LogComponent::Video,
		"VideoDecoder caps: maxDpbSlots=%u MaxActiveRefs=%u "
		"DpbAndOutputCoincide=%d DpbAndOutputDistinct=%d "
		"DpbSampled=%d DpbTransferSrc=%d OutputSampled=%d",
		maxDpbSlots, finalMaxActiveReferences,
		static_cast<int>(caps.supportsDpbAndOutputCoincide ? 1 : 0),
		static_cast<int>(caps.supportsDpbAndOutputDistinct ? 1 : 0),
		static_cast<int>(caps.supportsNv12DpbSampled ? 1 : 0),
		static_cast<int>(caps.supportsNv12DpbTransferSrc ? 1 : 0),
		static_cast<int>(caps.supportsNv12OutputSampled ? 1 : 0));

	VkExtent2D codedExtent = {decoder.impl_->codedWidth, decoder.impl_->codedHeight};

	// Create video session using oavk::VideoSession
	auto sessionResult = oavk::VideoSession::create(
		vkEngine,
		profile,
		codedExtent,
		caps.pictureFormat,
		caps.referencePictureFormat,
		maxDpbSlots,
		finalMaxActiveReferences);
	if (!sessionResult.isOk()) {
		return sessionResult.getStatus();
	}
	decoder.impl_->session = oa::move(sessionResult.getValue());

	// Create video queue
	auto queueResult = oavk::VideoQueue::create(vkEngine, oavk::VideoQueue::QueueType::Decode);
	if (!queueResult.isOk()) {
		return queueResult.getStatus();
	}
	decoder.impl_->queue = oa::move(queueResult.getValue());

	// Create bitstream buffer. We MUST attach the video profile via
	// vkVideoProfileListInfoKHR (otherwise VUID-vkCmdDecodeVideoKHR-pDecodeInfo-07135
	// fires every decode and the driver may produce undefined output, which
	// is what was causing the visible glitches).
	const oa::U64 bitstreamSize = 4 * 1024 * 1024; // 4MB per in-flight access unit
	for (oa::U32 i = 0; i < kBitstreamRingSize; ++i) {
		auto bitstreamResult = oavk::VideoBitstream::create(
			vkEngine,
			bitstreamSize,
			oavk::VideoBitstream::Direction::Decoder,
			caps.minBitstreamBufferOffsetAlignment == 0 ? 1 : caps.minBitstreamBufferOffsetAlignment,
			caps.minBitstreamBufferSizeAlignment == 0 ? 1 : caps.minBitstreamBufferSizeAlignment,
			&profile);
		if (!bitstreamResult.isOk()) {
			return closeDecoderAfterCreateFailure(
				decoder, bitstreamResult.getStatus());
		}
		decoder.impl_->bitstreamRing[i].buffer = oa::move(bitstreamResult.getValue());
	}
	decoder.impl_->currentBitstreamIndex = 0;

	oa::U32 queueFamilyCount = 0;
	const auto& instanceDispatch =
		oa::EngineDeviceAccess::get(vkEngine).instanceDispatch;
	instanceDispatch.vkGetPhysicalDeviceQueueFamilyProperties(
		static_cast<VkPhysicalDevice>(oa::EngineDeviceAccess::get(vkEngine).physicalDevice),
		&queueFamilyCount,
		nullptr);
	oa::Vec<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
	instanceDispatch.vkGetPhysicalDeviceQueueFamilyProperties(
		static_cast<VkPhysicalDevice>(oa::EngineDeviceAccess::get(vkEngine).physicalDevice),
		&queueFamilyCount,
		queueFamilyProperties.data());
	const oa::U32 videoQueueFamily = oa::EngineDeviceAccess::get(vkEngine).queues.videoDecodeQueueFamily;
	const bool videoQueueSupportsTransfer =
		videoQueueFamily < queueFamilyCount
		and (queueFamilyProperties[videoQueueFamily].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0;

	// Create DPB using oavk::VideoDpb
	VkImageUsageFlags dpbUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR
		| VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR;

	// --- resource path selection (capability-gated, topology-aware) ---
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
	// See VideoDecoderDeviceCompatibility.md "Capability-Gated architecture".

	const bool dedicatedVideoQueue =
		videoQueueFamily != oa::EngineDeviceAccess::get(vkEngine).queues.computeQueueFamily;

	// AV1 grey-frame fix (see oaVideoImplementationPlan.md §2):
	// AV1 reads the decoder *output* image, not the reconstructed DPB — the
	// reference decoders write the displayable picture to a distinct decode-output
	// even on coincide-capable drivers (film grain forces this), so sampling the DPB
	// yields a flat/grey frame. FFmpeg gives AV1 a distinct-output-capable setup
	// unconditionally (vulkan_decode.c:1341); vulkan-Video-samples' displayOut rule
	// reads decodeOut for AV1 (VkVideoDecoder.cpp:1054-1057). The film-grain seq flag
	// is not yet known at create() time, so mirror FFmpeg and route ALL AV1 to the
	// DistinctComputeConvert path (which samples the decode-output) whenever distinct
	// output is available. Distinct is also the iGPU-safe path, so this cannot regress
	// the device-loss-avoidance behaviour below.
	// NEEDS GPU VERIFICATION: the grey frame reproduces only on NVIDIA coincide.
	const bool av1PreferDistinct =
		resolved.codec == oa::VideoCodec::AV1
		and caps.supportsDpbAndOutputDistinct
		and caps.supportsNv12OutputSampled;

	// CoincidentFastStaging: only safe when the video queue has transfer
	// capability (copy stays on the same queue, no cross-family hazard).
	// When the video queue lacks transfer and is a different family, the
	// cross-queue DPB→staging copy has caused device-loss on Intel ANV.
	const bool useCoincideStaging =
		not av1PreferDistinct
		and caps.supportsDpbAndOutputCoincide
		and caps.supportsNv12DpbTransferSrc
		and videoQueueSupportsTransfer;

	// DistinctComputeConvert: decode into a distinct output image, then
	// NV12→RGBA via compute shader on the compute queue. Safe for iGPUs
	// because it avoids DPB image copies entirely — the distinct output
	// image is created with CONCURRENT sharing across video+compute families.
	const bool useDistinctOutput =
		not useCoincideStaging
		and caps.supportsDpbAndOutputDistinct
		and caps.supportsNv12OutputSampled;

	// CoincidentComputeStaging: copy the decoded DPB layer into an ordinary
	// NV12 image on the compute queue, then sample the staging image planes.
	// Prefer this over direct DPB sampling when TRANSFER_SRC is available.
	const bool useCoincideComputeStaging =
		not useCoincideStaging
		and not useDistinctOutput
		and caps.supportsDpbAndOutputCoincide
		and caps.supportsNv12DpbTransferSrc;

	// DirectCoincidentSampling: sample the DPB directly through a YCbCr
	// conversion sampler (COLOR_BIT aspect, no R8/R8G8 plane views). This is
	// only used when no transfer-based or distinct-output path is available.
	const bool useDirectCoincideSampling =
		not useCoincideStaging
		and not useDistinctOutput
		and not useCoincideComputeStaging
		and caps.supportsDpbAndOutputCoincide
		and caps.supportsNv12DpbSampled
		and hasHardwareYCbCrConversion(vkEngine);

	// Determine the resource path enum.
	oa::VideoResourcePath resourcePath = oa::VideoResourcePath::Unavailable;
	if (useCoincideStaging)              resourcePath = oa::VideoResourcePath::CoincidentFastStaging;
	else if (useDistinctOutput)          resourcePath = oa::VideoResourcePath::DistinctComputeConvert;
	else if (useCoincideComputeStaging)  resourcePath = oa::VideoResourcePath::CoincidentComputeStaging;
	else if (useDirectCoincideSampling)  resourcePath = oa::VideoResourcePath::DirectCoincidentSampling;

	if (resourcePath == oa::VideoResourcePath::Unavailable) {
		return closeDecoderAfterCreateFailure(decoder, oa::Status::error(
			oa::StatusCode::Unavailable,
			"vulkan Video device exposes no usable decoded-frame path "
			"(requires coincident DPB transfer/sample or distinct sampled output)"));
	}

	decoder.impl_->resourcePath = resourcePath;

	if (useDirectCoincideSampling) {
		dpbUsage |= VK_IMAGE_USAGE_SAMPLED_BIT;
	}
	if (useCoincideStaging or useCoincideComputeStaging) {
		dpbUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	}

	oavk::VideoDpb::CreateInfo dpbInfo = {};
	dpbInfo.profile = profile;
	dpbInfo.format = caps.referencePictureFormat;
	dpbInfo.codedExtent = codedExtent;
	dpbInfo.maxDpbSlots = maxDpbSlots;
	dpbInfo.usage = dpbUsage;

	auto dpbResult = oavk::VideoDpb::create(vkEngine, dpbInfo);
	if (!dpbResult.isOk()) {
		return dpbResult.getStatus();
	}
	decoder.impl_->dpb = oa::move(dpbResult.getValue());

	// --- Create path-specific resources ---
	static constexpr std::array<const char*, 5> kResourcePathNames = {{
		"CoincidentFastStaging",
		"DistinctComputeConvert",
		"CoincidentComputeStaging",
		"DirectCoincidentSampling",
		"Unavailable",
	}};

	if (useCoincideStaging or useCoincideComputeStaging) {
		oa::Status stagingStatus = decoder.createSampleStagingImages(
			vkEngine,
			profile,
			codedExtent,
			maxDpbSlots);
		if (!stagingStatus.isOk()) {
			return closeDecoderAfterCreateFailure(decoder, stagingStatus);
		}
		decoder.impl_->useSampleStaging = true;
		decoder.impl_->copySampleStagingOnVideoQueue =
			useCoincideStaging and videoQueueSupportsTransfer;
	} else if (useDistinctOutput) {
		oa::Status outputStatus = decoder.createOutputImages(
			vkEngine,
			profile,
			caps.pictureFormat,
			codedExtent,
			maxDpbSlots);
		if (!outputStatus.isOk()) {
			return closeDecoderAfterCreateFailure(decoder, outputStatus);
		}
	}

	OaLogInfo(oa::LogComponent::Video,
		"VideoDecoder: resource path = %s (videoQF=%u computeQF=%u dedicated=%s transfer=%s "
		"coincide=%s distinct=%s)",
		kResourcePathNames[static_cast<oa::U32>(resourcePath)],
		videoQueueFamily,
		oa::EngineDeviceAccess::get(vkEngine).queues.computeQueueFamily,
		dedicatedVideoQueue ? "yes" : "no",
		videoQueueSupportsTransfer ? "yes" : "no",
		caps.supportsDpbAndOutputCoincide ? "yes" : "no",
		caps.supportsDpbAndOutputDistinct ? "yes" : "no");

	// VP9 uses VK_NULL_HANDLE for videoSessionParameters at BeginCoding.
	// AV1 defers session parameters until the first sequence header OBU is parsed
	// (NVIDIA init_sequence pattern — no placeholder StdVideoAV1SequenceHeader).
	if (inProfile.codec != oa::VideoCodec::VP9 && inProfile.codec != oa::VideoCodec::AV1) {
		VkVideoSessionParametersKHR rawParams = VK_NULL_HANDLE;
		oa::Status sessionParametersStatus = createDecodeSessionParameters(
			oa::EngineDeviceAccess::get(vkEngine),
			inProfile.codec,
			inProfile,
			decoder.impl_->session.handle(),
			rawParams,
			nullptr /* AV1 seq provided later on first decode */);
		if (!sessionParametersStatus.isOk()) {
			return closeDecoderAfterCreateFailure(
				decoder, sessionParametersStatus);
		}
		decoder.impl_->sessionParameters.attach(vkEngine, rawParams);
	}

	// allocate two command buffers so decode N and decode N+1 can be in
	// flight concurrently. The video queue pool owns the underlying memory;
	// the fence ring tracks GPU completion per slot.
	for (oa::U32 i = 0; i < kCmdBufferCount; ++i) {
		auto cmdBufResult = decoder.impl_->queue.allocateCommandBuffer();
		if (!cmdBufResult.isOk()) {
			return closeDecoderAfterCreateFailure(
				decoder, cmdBufResult.getStatus());
		}
		decoder.impl_->commandBuffers[i] = cmdBufResult.getValue();

		auto fenceResult = decoder.impl_->queue.allocateFence();
		if (!fenceResult.isOk()) {
			return closeDecoderAfterCreateFailure(
				decoder, fenceResult.getStatus());
		}
		decoder.impl_->commandFences[i] = fenceResult.getValue();
	}

	// Timeline semaphore for GPU-side completion tracking.
	auto semResult = oavk::TimelineSemaphore::create(oa::EngineDeviceAccess::get(vkEngine), 0);
	if (!semResult.isOk()) {
		return closeDecoderAfterCreateFailure(
			decoder, semResult.getStatus());
	}
	decoder.impl_->timelineSemaphore = oa::move(*semResult);
	decoder.impl_->timelineValue = 0;
	for (oa::I32& vp9Slot : decoder.impl_->vp9BufferToDpbSlot) {
		vp9Slot = -1;
	}
	decoder.impl_->vp9BufferExtents.fill({0, 0});
	for (oa::I32& av1Slot : decoder.impl_->av1RefFrameToDpbSlot) {
		av1Slot = -1;
	}

	return decoder;
}

// Decode one frame
// flush decoder state
oa::Status oa::VideoDecoder::flush()
{
	if (not impl_ or impl_->session.handle() == VK_NULL_HANDLE)
	{
		return oa::Status::ok();
	}

	// drain all in-flight GPU work before wiping DPB state.
	if (impl_->engine) {
		auto& vkEngine = *impl_->engine;
		VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device);

		// decodeFrame returns after async video-queue submit; wait fences
		// before the timeline or flush can block while work is still in flight.
		for (oa::U32 i = 0; i < kCmdBufferCount; ++i) {
			if (impl_->commandFences[i] != VK_NULL_HANDLE) {
				const VkResult fenceResult = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkWaitForFences(
					device, 1, &impl_->commandFences[i], VK_TRUE, UINT64_MAX);
				if (fenceResult != VK_SUCCESS) {
					return oa::Status::error(
						oa::StatusCode::VulkanError,
						"vkWaitForFences failed while flushing video decoder");
				}
			}
		}
		for (BitstreamSlot& slot : impl_->bitstreamRing) {
			if (impl_->timelineSemaphore.semaphore != nullptr && slot.useValue > 0) {
				OA_RETURN_IF_ERROR(impl_->timelineSemaphore.wait(oa::EngineDeviceAccess::get(vkEngine), slot.useValue));
			}
		}

		if (impl_->timelineSemaphore.semaphore != nullptr && impl_->timelineValue > 0) {
			const oa::U64 completed = impl_->timelineSemaphore.getValue(oa::EngineDeviceAccess::get(vkEngine));
			if (completed < impl_->timelineValue) {
				OA_RETURN_IF_ERROR(impl_->timelineSemaphore.wait(oa::EngineDeviceAccess::get(vkEngine), impl_->timelineValue));
			}
		}
	}

	// Drop reference flags on all DPB slots: once flushed, no slot is a
	// valid reference for a subsequent stream segment. We don't try to
	// re-emit late frames here — that's a player-level concern, and the
	// player can read frames via decodeFrame() up to the EOS before
	// calling flush().
	for (oa::I32 i = 0; i < 16; ++i)
	{
		impl_->dpbSlots[i].inUse = false;
		impl_->dpbSlots[i].isReference = false;
		impl_->dpbSlots[i].picOrderCnt = -1;
		impl_->dpbSlots[i].frameNumber = 0;
	}

	// Decode output and DPB layouts need resetting so the next segment
	// re-issues UNDEFINED → VIDEO_DECODE_DST transitions. Converted RGBA
	// images are persistent caller-facing resources and retain their actual
	// layout across a decoder flush.
	impl_->outputImageLayouts.fill(VK_IMAGE_LAYOUT_UNDEFINED);
	impl_->dpbImageLayouts.fill(VK_IMAGE_LAYOUT_UNDEFINED);

	// reset frame counter and pending decoded sizes.
	impl_->currentFrameNumber = 0;
	for (auto& slot : impl_->bitstreamRing) {
		slot.size = 0;
		slot.useValue = 0;
	}
	impl_->currentBitstreamIndex = 0;
	impl_->outputReuseSemaphores.fill(VK_NULL_HANDLE);
	impl_->outputReuseValues.fill(0);
	impl_->previousPocLsb         = 0;
	impl_->previousPocMsb         = 0;
	impl_->h265PreviousPocLsb     = 0;
	impl_->h265PreviousPocMsb     = 0;
	impl_->h265HasPreviousPoc     = false;
	for (oa::I32& vp9Slot : impl_->vp9BufferToDpbSlot) {
		vp9Slot = -1;
	}
	impl_->vp9BufferExtents.fill({0, 0});
	for (oa::I32& av1Slot : impl_->av1RefFrameToDpbSlot) {
		av1Slot = -1;
	}
	// Drop the "session initialized" flag — the next decode will re-issue
	// VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR before submitting any work.
	impl_->videoSessionInitialized = false;
	impl_->slotDeviceActivated.fill(false);

	// clear the parameter-set caches. The next stream segment will re-upload
	// SPS/PPS/VPS from its own bitstream.
	clearParameterSets();
	if (impl_->parser) {
		impl_->parser->clearParameterSets();
	}
	impl_->h264SpsUploaded.fill(false);
	impl_->h264PpsUploaded.fill(false);
	impl_->h265VpsUploaded.fill(false);
	impl_->h265SpsUploaded.fill(false);
	impl_->h265PpsUploaded.fill(false);
	impl_->av1SequenceHeaderUploaded = false;
	impl_->sessionParameterUpdateCount = 0;

	// Recreate session parameters object to clear vulkan driver's internal state
	if (impl_->sessionParameters.handle() != VK_NULL_HANDLE && impl_->engine) {
		impl_->sessionParameters.destroy();
		auto& vkEngine = *impl_->engine;
		VkVideoSessionParametersKHR newParams = VK_NULL_HANDLE;
		oa::Status createStatus = createDecodeSessionParameters(
			oa::EngineDeviceAccess::get(vkEngine),
			impl_->profile.codec,
			impl_->profile,
			impl_->session.handle(),
			newParams,
			nullptr /* AV1 real seq will be supplied on next decodeFrame */);
		if (createStatus.isOk()) {
			impl_->sessionParameters.attach(vkEngine, newParams);
		}
	}

	return oa::Status::ok();
}

oa::Status oa::VideoDecoder::waitForCompletion(oa::U64 inTimeoutNs)
{
	if (not impl_ or impl_->timelineSemaphore.semaphore == nullptr || impl_->timelineValue == 0) {
		return oa::Status::ok();
	}
	if (!impl_->engine) {
		return oa::Status::error("Video decoder not initialized");
	}
	return impl_->timelineSemaphore.wait(
		oa::EngineDeviceAccess::get(*impl_->engine),
		impl_->timelineValue,
		inTimeoutNs);
}

void oa::VideoDecoder::stampFrameReady(oa::VideoFrame& outFrame) const noexcept
{
	if (not impl_ or impl_->engine == nullptr || impl_->timelineSemaphore.semaphore == nullptr || impl_->timelineValue == 0U) {
		outFrame.ready = {};
		return;
	}
	outFrame.ready = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*impl_->engine),
		impl_->timelineSemaphore, impl_->timelineValue);
}

oa::VideoDecoder::VideoCmdSlot oa::VideoDecoder::acquireVideoCmdSlot()
{
	VideoCmdSlot slot;
	slot.cb = impl_->commandBuffers[impl_->currentCommandBufferIndex];
	slot.fence = impl_->commandFences[impl_->currentCommandBufferIndex];
	if (impl_->engine && slot.cb != VK_NULL_HANDLE && slot.fence != VK_NULL_HANDLE) {
		auto& vkEngine = *impl_->engine;
		VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device);
		VkResult result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkWaitForFences(
			device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
		if (result != VK_SUCCESS) {
			slot.status = oa::Status::error(
				oa::StatusCode::VulkanError,
				"vkWaitForFences failed while acquiring video command buffer");
			return slot;
		}
		result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkResetFences(device, 1, &slot.fence);
		if (result != VK_SUCCESS) {
			slot.status = oa::Status::error(
				oa::StatusCode::VulkanError,
				"vkResetFences failed while acquiring video command buffer");
			return slot;
		}
		result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkResetCommandBuffer(slot.cb, 0);
		if (result != VK_SUCCESS) {
			slot.status = oa::Status::error(
				oa::StatusCode::VulkanError,
				"vkResetCommandBuffer failed while acquiring video command buffer");
		}
	}
	return slot;
}

void oa::VideoDecoder::releaseVideoCmdSlot()
{
	impl_->currentCommandBufferIndex = (impl_->currentCommandBufferIndex + 1) % kCmdBufferCount;
}

// Close decoder
oa::Status oa::VideoDecoder::close()
{
	if (not impl_ or not impl_->engine)
	{
		return oa::Status::ok();
	}

	auto& vkEngine = *impl_->engine;
	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device);
	oa::Status firstError = oa::Status::ok();
	auto retainError = [&firstError](const oa::Status& inStatus) {
		if (firstError.isOk() and not inStatus.isOk()) firstError = inStatus;
	};

	// Decode, sampled-read transitions, and DPB restores are submitted
	// asynchronously and all signal impl_->timelineSemaphore. in particular, the DPB
	// restore is queued after the conversion ticket that callers wait on.
	// Destroying the semaphore, command buffers, or DPB after only waiting
	// for conversion can therefore free objects still referenced by the
	// video queue and eventually surface as VK_ERROR_DEVICE_LOST on an
	// unrelated graphics submit.
	if (impl_->timelineSemaphore.semaphore != nullptr && impl_->timelineValue > 0) {
		const oa::Status waitStatus = impl_->timelineSemaphore.wait(
			oa::EngineDeviceAccess::get(vkEngine),
			impl_->timelineValue);
		if (not waitStatus.isOk()) {
			retainError(waitStatus);
		}
	}

	// drain per-CB fences that actually have a pending submit. After flush()
	// fences are left signaled; an unsignaled fence with no in-flight work
	// must not be waited on (would block forever).
	for (oa::U32 i = 0; i < kCmdBufferCount; ++i) {
		if (impl_->commandFences[i] == VK_NULL_HANDLE) {
			continue;
		}
		const VkResult fenceStatus = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkGetFenceStatus(device, impl_->commandFences[i]);
		if (fenceStatus == VK_NOT_READY) {
			const VkResult waitResult = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkWaitForFences(
				device, 1, &impl_->commandFences[i], VK_TRUE, UINT64_MAX);
			if (waitResult != VK_SUCCESS) {
				retainError(oa::Status::error(
					oa::StatusCode::VulkanError,
					"video decoder command fence completion failed"));
			}
		} else if (fenceStatus != VK_SUCCESS) {
			retainError(oa::Status::error(
				oa::StatusCode::VulkanError,
				"video decoder command fence status query failed"));
		}
	}

	// destroy fences and free command buffers before destroying the pool.
	for (oa::U32 i = 0; i < kCmdBufferCount; ++i) {
		if (impl_->commandFences[i] != VK_NULL_HANDLE) {
			oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroyFence(device, impl_->commandFences[i], nullptr);
			impl_->commandFences[i] = VK_NULL_HANDLE;
		}
		if (impl_->commandBuffers[i] != VK_NULL_HANDLE && impl_->queue.getCommandPool() != VK_NULL_HANDLE) {
			oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkFreeCommandBuffers(device, impl_->queue.getCommandPool(), 1, &impl_->commandBuffers[i]);
			impl_->commandBuffers[i] = VK_NULL_HANDLE;
		}
	}
	impl_->currentCommandBufferIndex = 0;

	impl_->timelineSemaphore.destroy(oa::EngineDeviceAccess::get(vkEngine));
	impl_->timelineValue = 0;
	impl_->outputReuseSemaphores.fill(VK_NULL_HANDLE);
	impl_->outputReuseValues.fill(0);

	// Destroy oavk video-layer objects.
	impl_->session.destroy();
	impl_->sessionParameters.destroy();
	impl_->queue.destroy();
	for (auto& slot : impl_->bitstreamRing) {
		slot.buffer.destroy();
		slot.size = 0;
		slot.useValue = 0;
	}
	impl_->currentBitstreamIndex = 0;
	impl_->dpb.destroy();

	if (impl_->conversionPipeline) {
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroyPipeline(device, impl_->conversionPipeline, nullptr);
		impl_->conversionPipeline = VK_NULL_HANDLE;
	}
	if (impl_->ycbcrSampler) {
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroySampler(device, impl_->ycbcrSampler, nullptr);
		impl_->ycbcrSampler = VK_NULL_HANDLE;
	}
	if (impl_->ycbcrSamplerNearest) {
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroySampler(device, impl_->ycbcrSamplerNearest, nullptr);
		impl_->ycbcrSamplerNearest = VK_NULL_HANDLE;
	}
	if (impl_->ycbcrConversion) {
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroySamplerYcbcrConversion(device, impl_->ycbcrConversion, nullptr);
		impl_->ycbcrConversion = VK_NULL_HANDLE;
	}
	for (VkImageView& view : impl_->cachedNv12YViews) {
		if (view) { oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroyImageView(device, view, nullptr); view = VK_NULL_HANDLE; }
	}
	for (VkImageView& view : impl_->cachedNv12UvViews) {
		if (view) { oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroyImageView(device, view, nullptr); view = VK_NULL_HANDLE; }
	}
	if (impl_->cachedNv12Sampler) {
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroySampler(device, impl_->cachedNv12Sampler, nullptr);
		impl_->cachedNv12Sampler = VK_NULL_HANDLE;
	}
	if (impl_->cachedNv12SamplerNearest) {
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroySampler(device, impl_->cachedNv12SamplerNearest, nullptr);
		impl_->cachedNv12SamplerNearest = VK_NULL_HANDLE;
	}
	impl_->cachedNv12Image = VK_NULL_HANDLE;
	impl_->reusedRgbaIndex = -1;
	impl_->reusedRgbaWidth = 0;
	impl_->reusedRgbaHeight = 0;

	// destroy old-style resources (RGB images, output images - keeping for now)
	for (VkImageView view : impl_->outputViews) {
		if (view) {
			oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroyImageView(device, view, nullptr);
		}
	}
	impl_->outputViews.clear();
	for (VkImageView view : impl_->rgbViews) {
		if (view) {
			oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroyImageView(device, view, nullptr);
		}
	}
	impl_->rgbViews.clear();
	for (oa::Usize i = 0; i < impl_->outputImages.size(); ++i) {
		VkImage image = impl_->outputImages[i];
		void* allocation = i < impl_->outputAllocations.size() ? impl_->outputAllocations[i] : nullptr;
		if (image && allocation) {
			OaVmaDestroyImage(
				static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(vkEngine).allocator),
				image,
				static_cast<OaVmaAllocation>(allocation));
		}
	}
	impl_->outputImages.clear();
	impl_->outputAllocations.clear();
	impl_->outputFrameCapacity = 0;
	impl_->outputImageLayouts.fill(VK_IMAGE_LAYOUT_UNDEFINED);
	for (VkImageView view : impl_->sampleYViews) {
		if (view) {
			oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroyImageView(device, view, nullptr);
		}
	}
	impl_->sampleYViews.clear();
	for (VkImageView view : impl_->sampleUvViews) {
		if (view) {
			oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroyImageView(device, view, nullptr);
		}
	}
	impl_->sampleUvViews.clear();
	for (oa::Usize i = 0; i < impl_->sampleImages.size(); ++i) {
		VkImage image = impl_->sampleImages[i];
		void* allocation = i < impl_->sampleAllocations.size() ? impl_->sampleAllocations[i] : nullptr;
		if (image && allocation) {
			OaVmaDestroyImage(
				static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(vkEngine).allocator),
				image,
				static_cast<OaVmaAllocation>(allocation));
		}
	}
	impl_->sampleImages.clear();
	impl_->sampleAllocations.clear();
	impl_->sampleImageLayouts.fill(VK_IMAGE_LAYOUT_UNDEFINED);
	impl_->useSampleStaging = false;
	impl_->copySampleStagingOnVideoQueue = false;
	for (oa::Usize i = 0; i < impl_->rgbImages.size(); ++i) {
		VkImage image = impl_->rgbImages[i];
		void* allocation = i < impl_->rgbAllocations.size() ? impl_->rgbAllocations[i] : nullptr;
		if (image && allocation) {
			OaVmaDestroyImage(
				static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(vkEngine).allocator),
				image,
				static_cast<OaVmaAllocation>(allocation));
		}
	}
	impl_->rgbImages.clear();
	impl_->rgbAllocations.clear();
	impl_->rgbImageLayouts.clear();
	impl_->parser.reset();

	// Vk session, parameters, queue (cmd pool), bitstream and DPB are owned
	// by their respective low-level oavk video wrappers above.
	impl_->engine = nullptr;
	return firstError;
}
// ============================================================================
// phase 2.4.1: DPB (Decoded Picture Buffer) Management
// ============================================================================

// allocate a DPB slot for a new decoded frame
oa::I32 oa::VideoDecoder::allocateDpbSlot()
{
	// Strategy: find first unused slot, or evict oldest non-reference frame

	// 1. Try to find an unused slot — lowest free index first (matches the
	// v0.6.55 known-good allocator). A round-robin variant was tried during the
	// phase C cleanup on the theory that spreading allocations across all 16
	// layers keeps host/device DPB association in sync longer; in practice it
	// reused layers in an order the H.264 reference/eviction bookkeeping did not
	// expect and reintroduced playback stutter/motion glitches. Deterministic
	// lowest-free reuse keeps the active reference set tight and predictable.
	for (oa::I32 i = 0; i < 16; ++i)
	{
		if (!impl_->dpbSlots[i].inUse)
		{
			impl_->dpbSlots[i].inUse = true;
			impl_->dpbSlots[i].frameNumber = impl_->currentFrameNumber;
			impl_->dpbSlots[i].isReference = false;
			impl_->lastAllocatedDpbSlot = i;
			return i;
		}
	}

	// 2. All slots in use - find oldest non-reference frame to evict
	oa::I32 oldestSlot = -1;
	oa::U32 oldestFrameNumber = impl_->currentFrameNumber;

	for (oa::I32 i = 0; i < 16; ++i)
	{
		if (!impl_->dpbSlots[i].isReference && impl_->dpbSlots[i].frameNumber < oldestFrameNumber)
		{
			oldestSlot = i;
			oldestFrameNumber = impl_->dpbSlots[i].frameNumber;
		}
	}

	// 3. If found non-reference slot, evict it
	if (oldestSlot >= 0)
	{
		impl_->dpbSlots[oldestSlot].inUse = true;
		impl_->dpbSlots[oldestSlot].frameNumber = impl_->currentFrameNumber;
		impl_->dpbSlots[oldestSlot].isReference = false;
		return oldestSlot;
	}

	// 4. All slots are reference frames - overflow
	return -1;
}
