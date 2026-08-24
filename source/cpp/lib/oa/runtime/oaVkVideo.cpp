// OA Runtime — vulkan Video Core Implementation (layer 1)

#include <oa/runtime/oaVkVideo.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/deviceAccess.h>
#include <oa/runtime/oaVma.h>

namespace {

const OaVkInstanceTable& instanceVk(oa::Engine& inEngine)
{
	return oa::EngineDeviceAccess::get(inEngine).instanceDispatch;
}

const OaVkDeviceTable& deviceVk(oa::Engine& inEngine)
{
	return oa::EngineDeviceAccess::get(inEngine).deviceDispatch;
}

}  // namespace

namespace oavk {

// ============================================================================
// VideoFormat — capability query + format negotiation
// ============================================================================

oa::Status VideoFormat::queryCapabilities(
	class oa::Engine& inRt,
	const VkVideoProfileInfoKHR& inProfile,
	bool inIsEncode,
	VkVideoCapabilitiesKHR& outCapabilities)
{
	if (!instanceVk(inRt).vkGetPhysicalDeviceVideoCapabilitiesKHR) {
		return oa::Status::error("vkGetPhysicalDeviceVideoCapabilitiesKHR is not loaded");
	}

	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(oa::EngineDeviceAccess::get(inRt).physicalDevice);

	// attach codec-specific capabilities struct
	VkVideoDecodeCapabilitiesKHR decodeCaps = {};
	VkVideoEncodeCapabilitiesKHR encodeCaps = {};
	VkVideoDecodeH264CapabilitiesKHR h264DecodeCaps = {};
	VkVideoDecodeH265CapabilitiesKHR h265DecodeCaps = {};
	VkVideoDecodeAV1CapabilitiesKHR av1DecodeCaps = {};
	VkVideoDecodeVP9CapabilitiesKHR vp9DecodeCaps = {};
	VkVideoEncodeH264CapabilitiesKHR h264EncodeCaps = {};
	VkVideoEncodeH265CapabilitiesKHR h265EncodeCaps = {};
	VkVideoEncodeAV1CapabilitiesKHR av1EncodeCaps = {};

	outCapabilities = {};
	outCapabilities.sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR;

	if (inIsEncode) {
		encodeCaps.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_CAPABILITIES_KHR;
		outCapabilities.pNext = &encodeCaps;

		// attach codec-specific encode caps based on profile
		const VkVideoEncodeH264ProfileInfoKHR* h264Profile =
			reinterpret_cast<const VkVideoEncodeH264ProfileInfoKHR*>(inProfile.pNext);
		const VkVideoEncodeH265ProfileInfoKHR* h265Profile =
			reinterpret_cast<const VkVideoEncodeH265ProfileInfoKHR*>(inProfile.pNext);
		const VkVideoEncodeAV1ProfileInfoKHR* av1Profile =
			reinterpret_cast<const VkVideoEncodeAV1ProfileInfoKHR*>(inProfile.pNext);

		if (h264Profile && h264Profile->sType == VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PROFILE_INFO_KHR) {
			h264EncodeCaps.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_KHR;
			encodeCaps.pNext = &h264EncodeCaps;
		} else if (h265Profile && h265Profile->sType == VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_PROFILE_INFO_KHR) {
			h265EncodeCaps.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_CAPABILITIES_KHR;
			encodeCaps.pNext = &h265EncodeCaps;
		} else if (av1Profile && av1Profile->sType == VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_PROFILE_INFO_KHR) {
			av1EncodeCaps.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_CAPABILITIES_KHR;
			encodeCaps.pNext = &av1EncodeCaps;
		}
	} else {
		decodeCaps.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR;
		outCapabilities.pNext = &decodeCaps;

		// attach codec-specific decode caps based on profile
		const VkVideoDecodeH264ProfileInfoKHR* h264Profile =
			reinterpret_cast<const VkVideoDecodeH264ProfileInfoKHR*>(inProfile.pNext);
		const VkVideoDecodeH265ProfileInfoKHR* h265Profile =
			reinterpret_cast<const VkVideoDecodeH265ProfileInfoKHR*>(inProfile.pNext);
		const VkVideoDecodeAV1ProfileInfoKHR* av1Profile =
			reinterpret_cast<const VkVideoDecodeAV1ProfileInfoKHR*>(inProfile.pNext);
		const VkVideoDecodeVP9ProfileInfoKHR* vp9Profile =
			reinterpret_cast<const VkVideoDecodeVP9ProfileInfoKHR*>(inProfile.pNext);

		if (h264Profile && h264Profile->sType == VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR) {
			h264DecodeCaps.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_KHR;
			decodeCaps.pNext = &h264DecodeCaps;
		} else if (h265Profile && h265Profile->sType == VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_INFO_KHR) {
			h265DecodeCaps.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_CAPABILITIES_KHR;
			decodeCaps.pNext = &h265DecodeCaps;
		} else if (av1Profile && av1Profile->sType == VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR) {
			av1DecodeCaps.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_CAPABILITIES_KHR;
			decodeCaps.pNext = &av1DecodeCaps;
		} else if (vp9Profile && vp9Profile->sType == VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PROFILE_INFO_KHR) {
			vp9DecodeCaps.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_CAPABILITIES_KHR;
			decodeCaps.pNext = &vp9DecodeCaps;
		}
	}

	VkResult result = instanceVk(inRt).vkGetPhysicalDeviceVideoCapabilitiesKHR(
		phys, &inProfile, &outCapabilities);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkGetPhysicalDeviceVideoCapabilitiesKHR failed");
	}

	return oa::Status::ok();
}

oa::Status VideoFormat::queryFormats(
	class oa::Engine& inRt,
	const VkVideoProfileInfoKHR& inProfile,
	VkImageUsageFlags inUsage,
	oa::Vec<VkVideoFormatPropertiesKHR>& outFormats)
{
	if (!instanceVk(inRt).vkGetPhysicalDeviceVideoFormatPropertiesKHR) {
		return oa::Status::error("vkGetPhysicalDeviceVideoFormatPropertiesKHR is not loaded");
	}

	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(oa::EngineDeviceAccess::get(inRt).physicalDevice);

	VkVideoProfileListInfoKHR profileList = {};
	profileList.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
	profileList.profileCount = 1;
	profileList.pProfiles = &inProfile;

	VkPhysicalDeviceVideoFormatInfoKHR formatInfo = {};
	formatInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR;
	formatInfo.pNext = &profileList;
	formatInfo.imageUsage = inUsage;

	oa::U32 formatCount = 0;
	VkResult result = instanceVk(inRt).vkGetPhysicalDeviceVideoFormatPropertiesKHR(
		phys, &formatInfo, &formatCount, nullptr);
	if (result != VK_SUCCESS) {
		return oa::Status::error("Failed to query vulkan Video format count");
	}

	outFormats.resize(formatCount);
	for (auto& format : outFormats) {
		format = {};
		format.sType = VK_STRUCTURE_TYPE_VIDEO_FORMAT_PROPERTIES_KHR;
	}

	if (formatCount == 0) {
		return oa::Status::ok();
	}

	result = instanceVk(inRt).vkGetPhysicalDeviceVideoFormatPropertiesKHR(
		phys, &formatInfo, &formatCount, outFormats.data());
	if (result != VK_SUCCESS) {
		outFormats.resize(0);
		return oa::Status::error("Failed to query vulkan Video formats");
	}
	outFormats.resize(formatCount);
	return oa::Status::ok();
}

bool VideoFormat::hasFormatWithUsage(
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

const VkVideoFormatPropertiesKHR* VideoFormat::findFormatWithUsage(
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

// ============================================================================
// VideoSession — wraps VkVideoSessionKHR + memory bindings
// ============================================================================

oa::Result<VideoSession> VideoSession::create(
	class oa::Engine& inRt,
	const VkVideoProfileInfoKHR& inProfile,
	const VkExtent2D& inCodedExtent,
	VkFormat inPictureFormat,
	VkFormat inReferenceFormat,
	oa::U32 inMaxDpbSlots,
	oa::U32 inMaxActiveReferences,
	oa::U32 inEncodeQualityLevel)
{
	VideoSession session;
	session.rt_ = &inRt;
	session.codedExtent_ = inCodedExtent;
	session.maxDpbSlots_ = inMaxDpbSlots;
	session.maxActiveReferences_ = inMaxActiveReferences;

	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(inRt).device);

	// query capabilities to get std header version
	VkVideoCapabilitiesKHR caps = {};
	VkVideoDecodeCapabilitiesKHR decodeCaps = {};
	VkVideoEncodeCapabilitiesKHR encodeCaps = {};

	// Determine if encode or decode based on profile operation
	bool isEncode = (inProfile.videoCodecOperation & VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR) ||
	                (inProfile.videoCodecOperation & VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR) ||
	                (inProfile.videoCodecOperation & VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_BIT_KHR);

	oa::Status capsStatus = VideoFormat::queryCapabilities(inRt, inProfile, isEncode, caps);
	if (!capsStatus.isOk()) {
		return capsStatus;
	}

	// Create video session
	VkVideoSessionCreateInfoKHR sessionInfo = {};
	sessionInfo.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR;
	// Quality level belongs to video session *parameters* (and coding-control
	// updates), not VkVideoSessionCreateInfoKHR. Keep the argument here for API
	// compatibility; the encoder attaches it while creating its parameters.
	(void)inEncodeQualityLevel;
	sessionInfo.pVideoProfile = &inProfile;
	sessionInfo.pStdHeaderVersion = &caps.stdHeaderVersion;
	sessionInfo.queueFamilyIndex = isEncode
		? oa::EngineDeviceAccess::get(inRt).queues.videoEncodeQueueFamily
		: oa::EngineDeviceAccess::get(inRt).queues.videoDecodeQueueFamily;
	sessionInfo.maxCodedExtent = inCodedExtent;
	sessionInfo.maxDpbSlots = inMaxDpbSlots;
	sessionInfo.maxActiveReferencePictures = inMaxActiveReferences;
	sessionInfo.pictureFormat = inPictureFormat;
	sessionInfo.referencePictureFormat = inReferenceFormat;

	VkResult result = deviceVk(inRt).vkCreateVideoSessionKHR(
		device, &sessionInfo, nullptr, &session.session_);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkCreateVideoSessionKHR failed");
	}

	// get memory requirements
	oa::U32 requirementCount = 0;
	result = deviceVk(inRt).vkGetVideoSessionMemoryRequirementsKHR(
		device, session.session_, &requirementCount, nullptr);
	if (result != VK_SUCCESS) {
		session.destroy();
		return oa::Status::error(oa::StatusCode::VulkanError, "vkGetVideoSessionMemoryRequirementsKHR failed");
	}

	oa::Vec<VkVideoSessionMemoryRequirementsKHR> requirements(requirementCount);
	for (auto& requirement : requirements) {
		requirement = {};
		requirement.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR;
	}
	result = deviceVk(inRt).vkGetVideoSessionMemoryRequirementsKHR(
		device, session.session_, &requirementCount, requirements.data());
	if (result != VK_SUCCESS) {
		session.destroy();
		return oa::Status::error(oa::StatusCode::VulkanError, "vkGetVideoSessionMemoryRequirementsKHR failed");
	}
	requirements.resize(requirementCount);

	// allocate and bind memory
	oa::Vec<VkBindVideoSessionMemoryInfoKHR> bindInfos(requirementCount);
	for (oa::U32 i = 0; i < requirementCount; ++i) {
		const VkVideoSessionMemoryRequirementsKHR& requirement = requirements[i];
		OaVmaAllocationCreateInfo allocCreateInfo = {};
		allocCreateInfo.usage = OA_VMA_MEMORY_USAGE_GPU_ONLY;
		allocCreateInfo.memoryTypeBits = requirement.memoryRequirements.memoryTypeBits;

		OaVmaAllocation allocation = VK_NULL_HANDLE;
		OaVmaAllocationInfo allocInfo = {};
		result = OaVmaAllocateMemory(
			static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(inRt).allocator),
			&requirement.memoryRequirements,
			&allocCreateInfo,
			&allocation,
			&allocInfo);
		if (result != VK_SUCCESS) {
			session.destroy();
			return oa::Status::error(oa::StatusCode::OutOfMemory, "Video session memory allocation failed");
		}
		session.allocations_.pushBack(allocation);

		VkBindVideoSessionMemoryInfoKHR& bindInfo = bindInfos[i];
		bindInfo = {};
		bindInfo.sType = VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR;
		bindInfo.memoryBindIndex = requirement.memoryBindIndex;
		bindInfo.memory = allocInfo.deviceMemory;
		bindInfo.memoryOffset = allocInfo.offset;
		bindInfo.memorySize = allocInfo.size;
	}

	if (requirementCount > 0) {
		result = deviceVk(inRt).vkBindVideoSessionMemoryKHR(
			device, session.session_, requirementCount, bindInfos.data());
		if (result != VK_SUCCESS) {
			session.destroy();
			return oa::Status::error(oa::StatusCode::VulkanError, "vkBindVideoSessionMemoryKHR failed");
		}
	}

	return session;
}

VideoSession::VideoSession(VideoSession&& inOther) noexcept
{
	moveFrom(std::move(inOther));
}

VideoSession& VideoSession::operator=(VideoSession&& inOther) noexcept
{
	destroy();
	moveFrom(std::move(inOther));
	return *this;
}

VideoSession::~VideoSession()
{
	destroy();
}

void VideoSession::destroy()
{
	if (session_ != VK_NULL_HANDLE) {
		VkDevice device = rt_ ? static_cast<VkDevice>(oa::EngineDeviceAccess::get(*rt_).device) : VK_NULL_HANDLE;
		if (device != VK_NULL_HANDLE) {
			deviceVk(*rt_).vkDestroyVideoSessionKHR(device, session_, nullptr);
		}
		session_ = VK_NULL_HANDLE;
	}

	for (void* alloc : allocations_) {
		if (alloc != nullptr && rt_ != nullptr) {
			OaVmaFreeMemory(static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(*rt_).allocator),
			                 static_cast<OaVmaAllocation>(alloc));
		}
	}
	allocations_.clear();

	rt_ = nullptr;
	codedExtent_ = {0, 0};
	maxDpbSlots_ = 0;
	maxActiveReferences_ = 0;
}

void VideoSession::moveFrom(VideoSession&& inOther) noexcept
{
	rt_ = inOther.rt_;
	session_ = inOther.session_;
	allocations_ = std::move(inOther.allocations_);
	codedExtent_ = inOther.codedExtent_;
	maxDpbSlots_ = inOther.maxDpbSlots_;
	maxActiveReferences_ = inOther.maxActiveReferences_;

	inOther.rt_ = nullptr;
	inOther.session_ = VK_NULL_HANDLE;
	inOther.codedExtent_ = {0, 0};
	inOther.maxDpbSlots_ = 0;
	inOther.maxActiveReferences_ = 0;
}

// ============================================================================
// VideoParameters — wraps VkVideoSessionParametersKHR
// ============================================================================

oa::Result<VideoParameters> VideoParameters::create(
	class oa::Engine& inRt,
	VkVideoSessionKHR inSession,
	const VkVideoSessionParametersCreateInfoKHR& inCreateInfo)
{
	VideoParameters params;
	params.rt_ = &inRt;

	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(inRt).device);
	VkResult result = deviceVk(inRt).vkCreateVideoSessionParametersKHR(
		device, &inCreateInfo, nullptr, &params.params_);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkCreateVideoSessionParametersKHR failed");
	}

	return params;
}

VideoParameters::VideoParameters(VideoParameters&& inOther) noexcept
{
	moveFrom(std::move(inOther));
}

VideoParameters& VideoParameters::operator=(VideoParameters&& inOther) noexcept
{
	destroy();
	moveFrom(std::move(inOther));
	return *this;
}

VideoParameters::~VideoParameters()
{
	destroy();
}

void VideoParameters::destroy()
{
	if (params_ != VK_NULL_HANDLE) {
		VkDevice device = rt_ ? static_cast<VkDevice>(oa::EngineDeviceAccess::get(*rt_).device) : VK_NULL_HANDLE;
		if (device != VK_NULL_HANDLE) {
			deviceVk(*rt_).vkDestroyVideoSessionParametersKHR(device, params_, nullptr);
		}
		params_ = VK_NULL_HANDLE;
	}
	rt_ = nullptr;
}

void VideoParameters::moveFrom(VideoParameters&& inOther) noexcept
{
	rt_ = inOther.rt_;
	params_ = inOther.params_;
	inOther.rt_ = nullptr;
	inOther.params_ = VK_NULL_HANDLE;
}

// ============================================================================
// VideoDpb — wraps DPB array image + views
// ============================================================================

oa::Result<VideoDpb> VideoDpb::create(
	class oa::Engine& inRt,
	const CreateInfo& inInfo)
{
	VideoDpb dpb;
	dpb.rt_ = &inRt;
	dpb.slotCapacity_ = inInfo.maxDpbSlots;

	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(inRt).device);

	VkVideoProfileListInfoKHR profileList = {};
	profileList.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
	profileList.profileCount = 1;
	profileList.pProfiles = &inInfo.profile;

	VkImageCreateInfo imageInfo = {};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.pNext = &profileList;
	// Sampled video DPBs use a full multi-planar YCbCr view. They do not need
	// mutable single-plane views, and enabling MUTABLE_FORMAT can strip the
	// profile-derived video format features on restrictive drivers.
	imageInfo.flags = 0;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = inInfo.format;
	imageInfo.extent.width = inInfo.codedExtent.width;
	imageInfo.extent.height = inInfo.codedExtent.height;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = inInfo.maxDpbSlots;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = inInfo.usage;
	// The DPB image is written on the video-decode queue and sampled on the
	// compute queue (NV12→RGB convert). Without CONCURRENT sharing across
	// those two families, the compute reads land on whatever the previous
	// decode left in that layer instead of the just-decoded picture —
	// looking like every Nth B-frame "snaps back" to the IDR's luma.
	// VUID expectation: pQueueFamilyIndices must list every family that will
	// access the image. Graphics is included so blit/present paths can sample
	// it as well.
	const bool encodeProfile =
		(inInfo.profile.videoCodecOperation & (
			VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR
			| VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR
			| VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_BIT_KHR)) != 0;
	oa::U32 sharedFamilies[3] = {
		encodeProfile ? oa::EngineDeviceAccess::get(inRt).queues.videoEncodeQueueFamily
			: oa::EngineDeviceAccess::get(inRt).queues.videoDecodeQueueFamily,
		oa::EngineDeviceAccess::get(inRt).queues.computeQueueFamily,
		oa::EngineDeviceAccess::get(inRt).queues.graphicsQueueFamily,
	};
	oa::U32 sharedFamilyCount = 0;
	for (oa::U32 family : sharedFamilies) {
		if (family == oavk::EnumerationIndexUnset) {
			continue;
		}
		bool dup = false;
		for (oa::U32 i = 0; i < sharedFamilyCount; ++i) {
			dup = dup or sharedFamilies[i] == family;
		}
		if (not dup) {
			sharedFamilies[sharedFamilyCount++] = family;
		}
	}
	if (sharedFamilyCount > 1) {
		imageInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
		imageInfo.queueFamilyIndexCount = sharedFamilyCount;
		imageInfo.pQueueFamilyIndices = sharedFamilies;
	} else {
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	OaVmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = OA_VMA_MEMORY_USAGE_GPU_ONLY;

	OaVmaAllocation allocation = VK_NULL_HANDLE;
	VkResult result = OaVmaCreateImage(
		static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(inRt).allocator),
		&imageInfo,
		&allocInfo,
		&dpb.image_,
		&allocation,
		nullptr);
	dpb.allocation_ = allocation;
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to create vulkan Video DPB image array");
	}

	// A multiplane DPB view must not inherit SAMPLED/STORAGE usage that its
	// format does not expose. Restrict it to the exact video usages requested
	// by the caller, for either decode or encode.
	VkImageViewUsageCreateInfo videoOnlyUsage = {};
	videoOnlyUsage.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
	videoOnlyUsage.usage = inInfo.usage & (
		VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR
		| VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR
		| VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR);

	VkImageViewCreateInfo viewInfo = {};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.pNext = &videoOnlyUsage;
	viewInfo.image = dpb.image_;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
	viewInfo.format = inInfo.format;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = inInfo.maxDpbSlots;

	result = deviceVk(inRt).vkCreateImageView(device, &viewInfo, nullptr, &dpb.view_);
	if (result != VK_SUCCESS) {
		dpb.destroy();
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to create vulkan Video DPB image view");
	}
	return dpb;
}

VideoDpb::VideoDpb(VideoDpb&& inOther) noexcept
{
	moveFrom(std::move(inOther));
}

VideoDpb& VideoDpb::operator=(VideoDpb&& inOther) noexcept
{
	destroy();
	moveFrom(std::move(inOther));
	return *this;
}

VideoDpb::~VideoDpb()
{
	destroy();
}

void VideoDpb::destroy()
{
	VkDevice device = rt_ ? static_cast<VkDevice>(oa::EngineDeviceAccess::get(*rt_).device) : VK_NULL_HANDLE;

	if (view_ != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
		deviceVk(*rt_).vkDestroyImageView(device, view_, nullptr);
		view_ = VK_NULL_HANDLE;
	}
	if (image_ != VK_NULL_HANDLE && allocation_ != nullptr && rt_ != nullptr) {
		OaVmaDestroyImage(static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(*rt_).allocator),
		                  image_, static_cast<OaVmaAllocation>(allocation_));
		image_ = VK_NULL_HANDLE;
		allocation_ = nullptr;
	}

	rt_ = nullptr;
	slotCapacity_ = 0;
}

void VideoDpb::moveFrom(VideoDpb&& inOther) noexcept
{
	rt_ = inOther.rt_;
	image_ = inOther.image_;
	view_ = inOther.view_;
	allocation_ = inOther.allocation_;
	slotCapacity_ = inOther.slotCapacity_;

	inOther.rt_ = nullptr;
	inOther.image_ = VK_NULL_HANDLE;
	inOther.view_ = VK_NULL_HANDLE;
	inOther.allocation_ = nullptr;
	inOther.slotCapacity_ = 0;
}

// ============================================================================
// VideoBitstream — wraps VMA buffer + alignment helpers
// ============================================================================

oa::Result<VideoBitstream> VideoBitstream::create(
	class oa::Engine& inRt,
	oa::U64 inSize,
	Direction inDirection,
	oa::U64 inOffsetAlignment,
	oa::U64 inSizeAlignment,
	const VkVideoProfileInfoKHR* inProfile)
{
	VideoBitstream bitstream;
	bitstream.rt_ = &inRt;
	bitstream.capacity_ = inSize;
	bitstream.offsetAlignment_ = inOffsetAlignment;
	bitstream.sizeAlignment_ = inSizeAlignment;
	bitstream.direction_ = inDirection;

	VkVideoProfileListInfoKHR profileList = {};
	profileList.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
	profileList.profileCount = 1;
	profileList.pProfiles = inProfile;

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	if (inProfile != nullptr) {
		bufferInfo.pNext = &profileList;
	}
	bufferInfo.size = inSize;
	bufferInfo.usage = inDirection == Direction::Decoder
		? VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		: VK_BUFFER_USAGE_VIDEO_ENCODE_DST_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	OaVmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = inDirection == Direction::Decoder
		? OA_VMA_MEMORY_USAGE_CPU_TO_GPU
		: OA_VMA_MEMORY_USAGE_GPU_TO_CPU;
	allocInfo.flags = OA_VMA_ALLOCATION_CREATE_MAPPED_BIT;

	OaVmaAllocation allocation = VK_NULL_HANDLE;
	VkResult result = OaVmaCreateBuffer(
		static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(inRt).allocator),
		&bufferInfo,
		&allocInfo,
		&bitstream.buffer_,
		&allocation,
		nullptr);
	bitstream.allocation_ = allocation;
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to create video bitstream buffer");
	}

	// map the buffer
	result = OaVmaMapMemory(
		static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(inRt).allocator),
		static_cast<OaVmaAllocation>(bitstream.allocation_),
		&bitstream.mappedPtr_);
	if (result != VK_SUCCESS) {
		bitstream.destroy();
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to map video bitstream buffer");
	}

	return bitstream;
}

VideoBitstream::VideoBitstream(VideoBitstream&& inOther) noexcept
{
	moveFrom(std::move(inOther));
}

VideoBitstream& VideoBitstream::operator=(VideoBitstream&& inOther) noexcept
{
	destroy();
	moveFrom(std::move(inOther));
	return *this;
}

VideoBitstream::~VideoBitstream()
{
	destroy();
}

oa::Status VideoBitstream::resize(oa::U64 inNewSize)
{
	if (inNewSize <= capacity_) {
		return oa::Status::ok();
	}

	// For simplicity, destroy and recreate (could use vmaResizeBuffer in future)
	destroy();
	auto result = create(*rt_, inNewSize, direction_, offsetAlignment_, sizeAlignment_);
	if (!result.isOk()) {
		return result.getStatus();
	}
	*this = std::move(result.getValue());
	return oa::Status::ok();
}

void VideoBitstream::destroy()
{
	if (mappedPtr_ != nullptr && allocation_ != nullptr && rt_ != nullptr) {
		OaVmaUnmapMemory(static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(*rt_).allocator),
		               static_cast<OaVmaAllocation>(allocation_));
		mappedPtr_ = nullptr;
	}

	if (buffer_ != VK_NULL_HANDLE && allocation_ != nullptr && rt_ != nullptr) {
		OaVmaDestroyBuffer(static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(*rt_).allocator),
		                 buffer_, static_cast<OaVmaAllocation>(allocation_));
		buffer_ = VK_NULL_HANDLE;
		allocation_ = nullptr;
	}

	rt_ = nullptr;
	capacity_ = 0;
	offsetAlignment_ = 1;
	sizeAlignment_ = 1;
}

void VideoBitstream::moveFrom(VideoBitstream&& inOther) noexcept
{
	rt_ = inOther.rt_;
	buffer_ = inOther.buffer_;
	allocation_ = inOther.allocation_;
	mappedPtr_ = inOther.mappedPtr_;
	capacity_ = inOther.capacity_;
	offsetAlignment_ = inOther.offsetAlignment_;
	sizeAlignment_ = inOther.sizeAlignment_;
	direction_ = inOther.direction_;

	inOther.rt_ = nullptr;
	inOther.buffer_ = VK_NULL_HANDLE;
	inOther.allocation_ = nullptr;
	inOther.mappedPtr_ = nullptr;
	inOther.capacity_ = 0;
	inOther.offsetAlignment_ = 1;
	inOther.sizeAlignment_ = 1;
}

// ============================================================================
// VideoQueue — queue + command pool + fence pool
// ============================================================================

oa::Result<VideoQueue> VideoQueue::create(
	class oa::Engine& inRt,
	QueueType inType)
{
	VideoQueue queue;
	queue.rt_ = &inRt;
	queue.type_ = inType;

	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(inRt).device);

	// get queue family index and queue handle
	if (inType == QueueType::Decode) {
		queue.queueFamilyIndex_ = oa::EngineDeviceAccess::get(inRt).queues.videoDecodeQueueFamily;
		queue.queue_ = static_cast<VkQueue>(oa::EngineDeviceAccess::get(inRt).queues.videoDecodeQueue);
	} else {
		queue.queueFamilyIndex_ = oa::EngineDeviceAccess::get(inRt).queues.videoEncodeQueueFamily;
		queue.queue_ = static_cast<VkQueue>(oa::EngineDeviceAccess::get(inRt).queues.videoEncodeQueue);
	}

	if (queue.queue_ == VK_NULL_HANDLE) {
		return oa::Status::error(oa::StatusCode::Unavailable, "Video queue not available");
	}

	// Create command pool
	VkCommandPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = queue.queueFamilyIndex_;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	VkResult result = deviceVk(inRt).vkCreateCommandPool(
		device, &poolInfo, nullptr, &queue.cmdPool_);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to create video command pool");
	}

	return queue;
}

VideoQueue::VideoQueue(VideoQueue&& inOther) noexcept
{
	moveFrom(std::move(inOther));
}

VideoQueue& VideoQueue::operator=(VideoQueue&& inOther) noexcept
{
	destroy();
	moveFrom(std::move(inOther));
	return *this;
}

VideoQueue::~VideoQueue()
{
	destroy();
}

oa::Result<VkCommandBuffer> VideoQueue::allocateCommandBuffer()
{
	if (cmdPool_ == VK_NULL_HANDLE || rt_ == nullptr) {
		return oa::Status::error("command pool not initialized");
	}

	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(*rt_).device);

	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = cmdPool_;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
	VkResult result = deviceVk(*rt_).vkAllocateCommandBuffers(
		device, &allocInfo, &cmdBuffer);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to allocate command buffer");
	}

	return cmdBuffer;
}

oa::Result<VkFence> VideoQueue::allocateFence()
{
	if (rt_ == nullptr) {
		return oa::Status::error("Engine not initialized");
	}

	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(*rt_).device);

	VkFenceCreateInfo fenceInfo = {};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	VkFence fence = VK_NULL_HANDLE;
	VkResult result = deviceVk(*rt_).vkCreateFence(device, &fenceInfo, nullptr, &fence);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to create fence");
	}

	return fence;
}

void VideoQueue::destroy()
{
	VkDevice device = rt_ ? static_cast<VkDevice>(oa::EngineDeviceAccess::get(*rt_).device) : VK_NULL_HANDLE;

	if (cmdPool_ != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
		deviceVk(*rt_).vkDestroyCommandPool(device, cmdPool_, nullptr);
		cmdPool_ = VK_NULL_HANDLE;
	}

	rt_ = nullptr;
	queue_ = VK_NULL_HANDLE;
	queueFamilyIndex_ = 0;
}

void VideoQueue::moveFrom(VideoQueue&& inOther) noexcept
{
	rt_ = inOther.rt_;
	queue_ = inOther.queue_;
	queueFamilyIndex_ = inOther.queueFamilyIndex_;
	cmdPool_ = inOther.cmdPool_;
	type_ = inOther.type_;

	inOther.rt_ = nullptr;
	inOther.queue_ = VK_NULL_HANDLE;
	inOther.queueFamilyIndex_ = 0;
	inOther.cmdPool_ = VK_NULL_HANDLE;
}

} // namespace oavk
