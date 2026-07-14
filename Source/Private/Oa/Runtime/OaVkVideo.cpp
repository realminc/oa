// OA Runtime — Vulkan Video Core Implementation (Layer 1)

#include <Oa/Runtime/OaVkVideo.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/OaVma.h>

// ============================================================================
// OaVkVideoFormat — capability query + format negotiation
// ============================================================================

OaStatus OaVkVideoFormat::QueryCapabilities(
	class OaComputeEngine& InRt,
	const VkVideoProfileInfoKHR& InProfile,
	bool InIsEncode,
	VkVideoCapabilitiesKHR& OutCapabilities)
{
	if (!vkGetPhysicalDeviceVideoCapabilitiesKHR) {
		return OaStatus::Error("vkGetPhysicalDeviceVideoCapabilitiesKHR is not loaded");
	}

	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(InRt.Device.PhysicalDevice);

	// Attach codec-specific capabilities struct
	VkVideoDecodeCapabilitiesKHR decodeCaps = {};
	VkVideoEncodeCapabilitiesKHR encodeCaps = {};
	VkVideoDecodeH264CapabilitiesKHR h264DecodeCaps = {};
	VkVideoDecodeH265CapabilitiesKHR h265DecodeCaps = {};
	VkVideoDecodeAV1CapabilitiesKHR av1DecodeCaps = {};
	VkVideoDecodeVP9CapabilitiesKHR vp9DecodeCaps = {};
	VkVideoEncodeH264CapabilitiesKHR h264EncodeCaps = {};
	VkVideoEncodeH265CapabilitiesKHR h265EncodeCaps = {};
	VkVideoEncodeAV1CapabilitiesKHR av1EncodeCaps = {};

	OutCapabilities = {};
	OutCapabilities.sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR;

	if (InIsEncode) {
		encodeCaps.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_CAPABILITIES_KHR;
		OutCapabilities.pNext = &encodeCaps;

		// Attach codec-specific encode caps based on profile
		const VkVideoEncodeH264ProfileInfoKHR* h264Profile =
			reinterpret_cast<const VkVideoEncodeH264ProfileInfoKHR*>(InProfile.pNext);
		const VkVideoEncodeH265ProfileInfoKHR* h265Profile =
			reinterpret_cast<const VkVideoEncodeH265ProfileInfoKHR*>(InProfile.pNext);
		const VkVideoEncodeAV1ProfileInfoKHR* av1Profile =
			reinterpret_cast<const VkVideoEncodeAV1ProfileInfoKHR*>(InProfile.pNext);

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
		OutCapabilities.pNext = &decodeCaps;

		// Attach codec-specific decode caps based on profile
		const VkVideoDecodeH264ProfileInfoKHR* h264Profile =
			reinterpret_cast<const VkVideoDecodeH264ProfileInfoKHR*>(InProfile.pNext);
		const VkVideoDecodeH265ProfileInfoKHR* h265Profile =
			reinterpret_cast<const VkVideoDecodeH265ProfileInfoKHR*>(InProfile.pNext);
		const VkVideoDecodeAV1ProfileInfoKHR* av1Profile =
			reinterpret_cast<const VkVideoDecodeAV1ProfileInfoKHR*>(InProfile.pNext);
		const VkVideoDecodeVP9ProfileInfoKHR* vp9Profile =
			reinterpret_cast<const VkVideoDecodeVP9ProfileInfoKHR*>(InProfile.pNext);

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

	VkResult result = vkGetPhysicalDeviceVideoCapabilitiesKHR(phys, &InProfile, &OutCapabilities);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkGetPhysicalDeviceVideoCapabilitiesKHR failed");
	}

	return OaStatus::Ok();
}

OaStatus OaVkVideoFormat::QueryFormats(
	class OaComputeEngine& InRt,
	const VkVideoProfileInfoKHR& InProfile,
	VkImageUsageFlags InUsage,
	OaVec<VkVideoFormatPropertiesKHR>& OutFormats)
{
	if (!vkGetPhysicalDeviceVideoFormatPropertiesKHR) {
		return OaStatus::Error("vkGetPhysicalDeviceVideoFormatPropertiesKHR is not loaded");
	}

	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(InRt.Device.PhysicalDevice);

	VkVideoProfileListInfoKHR profileList = {};
	profileList.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
	profileList.profileCount = 1;
	profileList.pProfiles = &InProfile;

	VkPhysicalDeviceVideoFormatInfoKHR formatInfo = {};
	formatInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR;
	formatInfo.pNext = &profileList;
	formatInfo.imageUsage = InUsage;

	OaU32 formatCount = 0;
	VkResult result = vkGetPhysicalDeviceVideoFormatPropertiesKHR(phys, &formatInfo, &formatCount, nullptr);
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

	result = vkGetPhysicalDeviceVideoFormatPropertiesKHR(phys, &formatInfo, &formatCount, OutFormats.Data());
	if (result != VK_SUCCESS) {
		OutFormats.Resize(0);
		return OaStatus::Error("Failed to query Vulkan Video formats");
	}
	OutFormats.Resize(formatCount);
	return OaStatus::Ok();
}

bool OaVkVideoFormat::HasFormatWithUsage(
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

const VkVideoFormatPropertiesKHR* OaVkVideoFormat::FindFormatWithUsage(
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

// ============================================================================
// OaVkVideoSession — wraps VkVideoSessionKHR + memory bindings
// ============================================================================

OaResult<OaVkVideoSession> OaVkVideoSession::Create(
	class OaComputeEngine& InRt,
	const VkVideoProfileInfoKHR& InProfile,
	const VkExtent2D& InCodedExtent,
	VkFormat InPictureFormat,
	VkFormat InReferenceFormat,
	OaU32 InMaxDpbSlots,
	OaU32 InMaxActiveReferences,
	OaU32 InEncodeQualityLevel)
{
	OaVkVideoSession session;
	session.Rt_ = &InRt;
	session.CodedExtent_ = InCodedExtent;
	session.MaxDpbSlots_ = InMaxDpbSlots;
	session.MaxActiveReferences_ = InMaxActiveReferences;

	VkDevice device = static_cast<VkDevice>(InRt.Device.Device);

	// Query capabilities to get std header version
	VkVideoCapabilitiesKHR caps = {};
	VkVideoDecodeCapabilitiesKHR decodeCaps = {};
	VkVideoEncodeCapabilitiesKHR encodeCaps = {};
	
	// Determine if encode or decode based on profile operation
	bool isEncode = (InProfile.videoCodecOperation & VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR) ||
	                (InProfile.videoCodecOperation & VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR) ||
	                (InProfile.videoCodecOperation & VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_BIT_KHR);

	OaStatus capsStatus = OaVkVideoFormat::QueryCapabilities(InRt, InProfile, isEncode, caps);
	if (!capsStatus.IsOk()) {
		return capsStatus;
	}

	// Create video session
	VkVideoSessionCreateInfoKHR sessionInfo = {};
	sessionInfo.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR;
	// Quality level belongs to video session *parameters* (and coding-control
	// updates), not VkVideoSessionCreateInfoKHR. Keep the argument here for API
	// compatibility; the encoder attaches it while creating its parameters.
	(void)InEncodeQualityLevel;
	sessionInfo.pVideoProfile = &InProfile;
	sessionInfo.pStdHeaderVersion = &caps.stdHeaderVersion;
	sessionInfo.queueFamilyIndex = isEncode 
		? InRt.Device.Queues.VideoEncodeQueueFamily 
		: InRt.Device.Queues.VideoDecodeQueueFamily;
	sessionInfo.maxCodedExtent = InCodedExtent;
	sessionInfo.maxDpbSlots = InMaxDpbSlots;
	sessionInfo.maxActiveReferencePictures = InMaxActiveReferences;
	sessionInfo.pictureFormat = InPictureFormat;
	sessionInfo.referencePictureFormat = InReferenceFormat;

	VkResult result = vkCreateVideoSessionKHR(device, &sessionInfo, nullptr, &session.Session_);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkCreateVideoSessionKHR failed");
	}

	// Get memory requirements
	OaU32 requirementCount = 0;
	result = vkGetVideoSessionMemoryRequirementsKHR(device, session.Session_, &requirementCount, nullptr);
	if (result != VK_SUCCESS) {
		session.Destroy();
		return OaStatus::Error(OaStatusCode::VulkanError, "vkGetVideoSessionMemoryRequirementsKHR failed");
	}

	OaVec<VkVideoSessionMemoryRequirementsKHR> requirements(requirementCount);
	for (auto& requirement : requirements) {
		requirement = {};
		requirement.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR;
	}
	result = vkGetVideoSessionMemoryRequirementsKHR(device, session.Session_, &requirementCount, requirements.Data());
	if (result != VK_SUCCESS) {
		session.Destroy();
		return OaStatus::Error(OaStatusCode::VulkanError, "vkGetVideoSessionMemoryRequirementsKHR failed");
	}
	requirements.Resize(requirementCount);

	// Allocate and bind memory
	OaVec<VkBindVideoSessionMemoryInfoKHR> bindInfos(requirementCount);
	for (OaU32 i = 0; i < requirementCount; ++i) {
		const VkVideoSessionMemoryRequirementsKHR& requirement = requirements[i];
		OaVmaAllocationCreateInfo allocCreateInfo = {};
		allocCreateInfo.usage = OA_VMA_MEMORY_USAGE_GPU_ONLY;
		allocCreateInfo.memoryTypeBits = requirement.memoryRequirements.memoryTypeBits;

		OaVmaAllocation allocation = VK_NULL_HANDLE;
		OaVmaAllocationInfo allocInfo = {};
		result = OaVmaAllocateMemory(
			static_cast<OaVmaAllocator>(InRt.Allocator.Allocator),
			&requirement.memoryRequirements,
			&allocCreateInfo,
			&allocation,
			&allocInfo);
		if (result != VK_SUCCESS) {
			session.Destroy();
			return OaStatus::Error(OaStatusCode::OutOfMemory, "Video session memory allocation failed");
		}
		session.Allocations_.PushBack(allocation);

		VkBindVideoSessionMemoryInfoKHR& bindInfo = bindInfos[i];
		bindInfo = {};
		bindInfo.sType = VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR;
		bindInfo.memoryBindIndex = requirement.memoryBindIndex;
		bindInfo.memory = allocInfo.deviceMemory;
		bindInfo.memoryOffset = allocInfo.offset;
		bindInfo.memorySize = allocInfo.size;
	}

	if (requirementCount > 0) {
		result = vkBindVideoSessionMemoryKHR(device, session.Session_, requirementCount, bindInfos.Data());
		if (result != VK_SUCCESS) {
			session.Destroy();
			return OaStatus::Error(OaStatusCode::VulkanError, "vkBindVideoSessionMemoryKHR failed");
		}
	}

	return session;
}

OaVkVideoSession::OaVkVideoSession(OaVkVideoSession&& InOther) noexcept
{
	MoveFrom(std::move(InOther));
}

OaVkVideoSession& OaVkVideoSession::operator=(OaVkVideoSession&& InOther) noexcept
{
	Destroy();
	MoveFrom(std::move(InOther));
	return *this;
}

OaVkVideoSession::~OaVkVideoSession()
{
	Destroy();
}

void OaVkVideoSession::Destroy()
{
	if (Session_ != VK_NULL_HANDLE) {
		VkDevice device = Rt_ ? static_cast<VkDevice>(Rt_->Device.Device) : VK_NULL_HANDLE;
		if (device != VK_NULL_HANDLE) {
			vkDestroyVideoSessionKHR(device, Session_, nullptr);
		}
		Session_ = VK_NULL_HANDLE;
	}

	for (void* alloc : Allocations_) {
		if (alloc != nullptr && Rt_ != nullptr) {
			OaVmaFreeMemory(static_cast<OaVmaAllocator>(Rt_->Allocator.Allocator), 
			                 static_cast<OaVmaAllocation>(alloc));
		}
	}
	Allocations_.Clear();

	Rt_ = nullptr;
	CodedExtent_ = {0, 0};
	MaxDpbSlots_ = 0;
	MaxActiveReferences_ = 0;
}

void OaVkVideoSession::MoveFrom(OaVkVideoSession&& InOther) noexcept
{
	Rt_ = InOther.Rt_;
	Session_ = InOther.Session_;
	Allocations_ = std::move(InOther.Allocations_);
	CodedExtent_ = InOther.CodedExtent_;
	MaxDpbSlots_ = InOther.MaxDpbSlots_;
	MaxActiveReferences_ = InOther.MaxActiveReferences_;

	InOther.Rt_ = nullptr;
	InOther.Session_ = VK_NULL_HANDLE;
	InOther.CodedExtent_ = {0, 0};
	InOther.MaxDpbSlots_ = 0;
	InOther.MaxActiveReferences_ = 0;
}

// ============================================================================
// OaVkVideoParameters — wraps VkVideoSessionParametersKHR
// ============================================================================

OaResult<OaVkVideoParameters> OaVkVideoParameters::Create(
	class OaComputeEngine& InRt,
	VkVideoSessionKHR InSession,
	const VkVideoSessionParametersCreateInfoKHR& InCreateInfo)
{
	OaVkVideoParameters params;
	params.Rt_ = &InRt;

	VkDevice device = static_cast<VkDevice>(InRt.Device.Device);
	VkResult result = vkCreateVideoSessionParametersKHR(device, &InCreateInfo, nullptr, &params.Params_);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkCreateVideoSessionParametersKHR failed");
	}

	return params;
}

OaVkVideoParameters::OaVkVideoParameters(OaVkVideoParameters&& InOther) noexcept
{
	MoveFrom(std::move(InOther));
}

OaVkVideoParameters& OaVkVideoParameters::operator=(OaVkVideoParameters&& InOther) noexcept
{
	Destroy();
	MoveFrom(std::move(InOther));
	return *this;
}

OaVkVideoParameters::~OaVkVideoParameters()
{
	Destroy();
}

void OaVkVideoParameters::Destroy()
{
	if (Params_ != VK_NULL_HANDLE) {
		VkDevice device = Rt_ ? static_cast<VkDevice>(Rt_->Device.Device) : VK_NULL_HANDLE;
		if (device != VK_NULL_HANDLE) {
			vkDestroyVideoSessionParametersKHR(device, Params_, nullptr);
		}
		Params_ = VK_NULL_HANDLE;
	}
	Rt_ = nullptr;
}

void OaVkVideoParameters::MoveFrom(OaVkVideoParameters&& InOther) noexcept
{
	Rt_ = InOther.Rt_;
	Params_ = InOther.Params_;
	InOther.Rt_ = nullptr;
	InOther.Params_ = VK_NULL_HANDLE;
}

// ============================================================================
// OaVkVideoDpb — wraps DPB array image + views
// ============================================================================

OaResult<OaVkVideoDpb> OaVkVideoDpb::Create(
	class OaComputeEngine& InRt,
	const CreateInfo& InInfo)
{
	OaVkVideoDpb dpb;
	dpb.Rt_ = &InRt;
	dpb.SlotCapacity_ = InInfo.MaxDpbSlots;

	VkDevice device = static_cast<VkDevice>(InRt.Device.Device);

	VkVideoProfileListInfoKHR profileList = {};
	profileList.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
	profileList.profileCount = 1;
	profileList.pProfiles = &InInfo.Profile;

	VkImageCreateInfo imageInfo = {};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.pNext = &profileList;
	// Sampled video DPBs use a full multi-planar YCbCr view. They do not need
	// mutable single-plane views, and enabling MUTABLE_FORMAT can strip the
	// profile-derived video format features on restrictive drivers.
	imageInfo.flags = 0;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = InInfo.Format;
	imageInfo.extent.width = InInfo.CodedExtent.width;
	imageInfo.extent.height = InInfo.CodedExtent.height;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = InInfo.MaxDpbSlots;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = InInfo.Usage;
	// The DPB image is written on the video-decode queue and sampled on the
	// compute queue (NV12→RGB convert). Without CONCURRENT sharing across
	// those two families, the compute reads land on whatever the previous
	// decode left in that layer instead of the just-decoded picture —
	// looking like every Nth B-frame "snaps back" to the IDR's luma.
	// VUID expectation: pQueueFamilyIndices must list every family that will
	// access the image. Graphics is included so blit/present paths can sample
	// it as well.
	const bool encodeProfile =
		(InInfo.Profile.videoCodecOperation & (
			VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR
			| VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR
			| VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_BIT_KHR)) != 0;
	OaU32 sharedFamilies[3] = {
		encodeProfile ? InRt.Device.Queues.VideoEncodeQueueFamily
			: InRt.Device.Queues.VideoDecodeQueueFamily,
		InRt.Device.Queues.ComputeQueueFamily,
		InRt.Device.Queues.GraphicsQueueFamily,
	};
	OaU32 sharedFamilyCount = 0;
	for (OaU32 family : sharedFamilies) {
		if (family == OaVkEnumerationIndexUnset) {
			continue;
		}
		bool dup = false;
		for (OaU32 i = 0; i < sharedFamilyCount; ++i) {
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
		static_cast<OaVmaAllocator>(InRt.Allocator.Allocator),
		&imageInfo,
		&allocInfo,
		&dpb.Image_,
		&allocation,
		nullptr);
	dpb.Allocation_ = allocation;
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to create Vulkan Video DPB image array");
	}

	// A multiplane DPB view must not inherit SAMPLED/STORAGE usage that its
	// format does not expose. Restrict it to the exact video usages requested
	// by the caller, for either decode or encode.
	VkImageViewUsageCreateInfo videoOnlyUsage = {};
	videoOnlyUsage.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
	videoOnlyUsage.usage = InInfo.Usage & (
		VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR
		| VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR
		| VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR);

	VkImageViewCreateInfo viewInfo = {};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.pNext = &videoOnlyUsage;
	viewInfo.image = dpb.Image_;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
	viewInfo.format = InInfo.Format;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = InInfo.MaxDpbSlots;

	result = vkCreateImageView(device, &viewInfo, nullptr, &dpb.View_);
	if (result != VK_SUCCESS) {
		dpb.Destroy();
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to create Vulkan Video DPB image view");
	}
	return dpb;
}

OaVkVideoDpb::OaVkVideoDpb(OaVkVideoDpb&& InOther) noexcept
{
	MoveFrom(std::move(InOther));
}

OaVkVideoDpb& OaVkVideoDpb::operator=(OaVkVideoDpb&& InOther) noexcept
{
	Destroy();
	MoveFrom(std::move(InOther));
	return *this;
}

OaVkVideoDpb::~OaVkVideoDpb()
{
	Destroy();
}

void OaVkVideoDpb::Destroy()
{
	VkDevice device = Rt_ ? static_cast<VkDevice>(Rt_->Device.Device) : VK_NULL_HANDLE;

	if (View_ != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
		vkDestroyImageView(device, View_, nullptr);
		View_ = VK_NULL_HANDLE;
	}
	if (Image_ != VK_NULL_HANDLE && Allocation_ != nullptr && Rt_ != nullptr) {
		OaVmaDestroyImage(static_cast<OaVmaAllocator>(Rt_->Allocator.Allocator), 
		                  Image_, static_cast<OaVmaAllocation>(Allocation_));
		Image_ = VK_NULL_HANDLE;
		Allocation_ = nullptr;
	}

	Rt_ = nullptr;
	SlotCapacity_ = 0;
}

void OaVkVideoDpb::MoveFrom(OaVkVideoDpb&& InOther) noexcept
{
	Rt_ = InOther.Rt_;
	Image_ = InOther.Image_;
	View_ = InOther.View_;
	Allocation_ = InOther.Allocation_;
	SlotCapacity_ = InOther.SlotCapacity_;

	InOther.Rt_ = nullptr;
	InOther.Image_ = VK_NULL_HANDLE;
	InOther.View_ = VK_NULL_HANDLE;
	InOther.Allocation_ = nullptr;
	InOther.SlotCapacity_ = 0;
}

// ============================================================================
// OaVkVideoBitstream — wraps VMA buffer + alignment helpers
// ============================================================================

OaResult<OaVkVideoBitstream> OaVkVideoBitstream::Create(
	class OaComputeEngine& InRt,
	OaU64 InSize,
	Direction InDirection,
	OaU64 InOffsetAlignment,
	OaU64 InSizeAlignment,
	const VkVideoProfileInfoKHR* InProfile)
{
	OaVkVideoBitstream bitstream;
	bitstream.Rt_ = &InRt;
	bitstream.Capacity_ = InSize;
	bitstream.OffsetAlignment_ = InOffsetAlignment;
	bitstream.SizeAlignment_ = InSizeAlignment;
	bitstream.Direction_ = InDirection;

	VkVideoProfileListInfoKHR profileList = {};
	profileList.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
	profileList.profileCount = 1;
	profileList.pProfiles = InProfile;

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	if (InProfile != nullptr) {
		bufferInfo.pNext = &profileList;
	}
	bufferInfo.size = InSize;
	bufferInfo.usage = InDirection == Direction::Decoder
		? VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		: VK_BUFFER_USAGE_VIDEO_ENCODE_DST_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	OaVmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = InDirection == Direction::Decoder 
		? OA_VMA_MEMORY_USAGE_CPU_TO_GPU 
		: OA_VMA_MEMORY_USAGE_GPU_TO_CPU;
	allocInfo.flags = OA_VMA_ALLOCATION_CREATE_MAPPED_BIT;

	OaVmaAllocation allocation = VK_NULL_HANDLE;
	VkResult result = OaVmaCreateBuffer(
		static_cast<OaVmaAllocator>(InRt.Allocator.Allocator),
		&bufferInfo,
		&allocInfo,
		&bitstream.Buffer_,
		&allocation,
		nullptr);
	bitstream.Allocation_ = allocation;
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to create video bitstream buffer");
	}

	// Map the buffer
	result = OaVmaMapMemory(
		static_cast<OaVmaAllocator>(InRt.Allocator.Allocator),
		static_cast<OaVmaAllocation>(bitstream.Allocation_),
		&bitstream.MappedPtr_);
	if (result != VK_SUCCESS) {
		bitstream.Destroy();
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to map video bitstream buffer");
	}

	return bitstream;
}

OaVkVideoBitstream::OaVkVideoBitstream(OaVkVideoBitstream&& InOther) noexcept
{
	MoveFrom(std::move(InOther));
}

OaVkVideoBitstream& OaVkVideoBitstream::operator=(OaVkVideoBitstream&& InOther) noexcept
{
	Destroy();
	MoveFrom(std::move(InOther));
	return *this;
}

OaVkVideoBitstream::~OaVkVideoBitstream()
{
	Destroy();
}

OaStatus OaVkVideoBitstream::Resize(OaU64 InNewSize)
{
	if (InNewSize <= Capacity_) {
		return OaStatus::Ok();
	}

	// For simplicity, destroy and recreate (could use vmaResizeBuffer in future)
	Destroy();
	auto result = Create(*Rt_, InNewSize, Direction_, OffsetAlignment_, SizeAlignment_);
	if (!result.IsOk()) {
		return result.GetStatus();
	}
	*this = std::move(result.GetValue());
	return OaStatus::Ok();
}

void OaVkVideoBitstream::Destroy()
{
	if (MappedPtr_ != nullptr && Allocation_ != nullptr && Rt_ != nullptr) {
		OaVmaUnmapMemory(static_cast<OaVmaAllocator>(Rt_->Allocator.Allocator), 
		               static_cast<OaVmaAllocation>(Allocation_));
		MappedPtr_ = nullptr;
	}

	if (Buffer_ != VK_NULL_HANDLE && Allocation_ != nullptr && Rt_ != nullptr) {
		OaVmaDestroyBuffer(static_cast<OaVmaAllocator>(Rt_->Allocator.Allocator), 
		                 Buffer_, static_cast<OaVmaAllocation>(Allocation_));
		Buffer_ = VK_NULL_HANDLE;
		Allocation_ = nullptr;
	}

	Rt_ = nullptr;
	Capacity_ = 0;
	OffsetAlignment_ = 1;
	SizeAlignment_ = 1;
}

void OaVkVideoBitstream::MoveFrom(OaVkVideoBitstream&& InOther) noexcept
{
	Rt_ = InOther.Rt_;
	Buffer_ = InOther.Buffer_;
	Allocation_ = InOther.Allocation_;
	MappedPtr_ = InOther.MappedPtr_;
	Capacity_ = InOther.Capacity_;
	OffsetAlignment_ = InOther.OffsetAlignment_;
	SizeAlignment_ = InOther.SizeAlignment_;
	Direction_ = InOther.Direction_;

	InOther.Rt_ = nullptr;
	InOther.Buffer_ = VK_NULL_HANDLE;
	InOther.Allocation_ = nullptr;
	InOther.MappedPtr_ = nullptr;
	InOther.Capacity_ = 0;
	InOther.OffsetAlignment_ = 1;
	InOther.SizeAlignment_ = 1;
}

// ============================================================================
// OaVkVideoQueue — queue + command pool + fence pool
// ============================================================================

OaResult<OaVkVideoQueue> OaVkVideoQueue::Create(
	class OaComputeEngine& InRt,
	QueueType InType)
{
	OaVkVideoQueue queue;
	queue.Rt_ = &InRt;
	queue.Type_ = InType;

	VkDevice device = static_cast<VkDevice>(InRt.Device.Device);

	// Get queue family index and queue handle
	if (InType == QueueType::Decode) {
		queue.QueueFamilyIndex_ = InRt.Device.Queues.VideoDecodeQueueFamily;
		queue.Queue_ = static_cast<VkQueue>(InRt.Device.Queues.VideoDecodeQueue);
	} else {
		queue.QueueFamilyIndex_ = InRt.Device.Queues.VideoEncodeQueueFamily;
		queue.Queue_ = static_cast<VkQueue>(InRt.Device.Queues.VideoEncodeQueue);
	}

	if (queue.Queue_ == VK_NULL_HANDLE) {
		return OaStatus::Error(OaStatusCode::Unavailable, "Video queue not available");
	}

	// Create command pool
	VkCommandPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = queue.QueueFamilyIndex_;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	VkResult result = vkCreateCommandPool(device, &poolInfo, nullptr, &queue.CmdPool_);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to create video command pool");
	}

	return queue;
}

OaVkVideoQueue::OaVkVideoQueue(OaVkVideoQueue&& InOther) noexcept
{
	MoveFrom(std::move(InOther));
}

OaVkVideoQueue& OaVkVideoQueue::operator=(OaVkVideoQueue&& InOther) noexcept
{
	Destroy();
	MoveFrom(std::move(InOther));
	return *this;
}

OaVkVideoQueue::~OaVkVideoQueue()
{
	Destroy();
}

OaResult<VkCommandBuffer> OaVkVideoQueue::AllocateCommandBuffer()
{
	if (CmdPool_ == VK_NULL_HANDLE || Rt_ == nullptr) {
		return OaStatus::Error("Command pool not initialized");
	}

	VkDevice device = static_cast<VkDevice>(Rt_->Device.Device);

	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = CmdPool_;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
	VkResult result = vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to allocate command buffer");
	}

	return cmdBuffer;
}

OaResult<VkFence> OaVkVideoQueue::AllocateFence()
{
	if (Rt_ == nullptr) {
		return OaStatus::Error("Engine not initialized");
	}

	VkDevice device = static_cast<VkDevice>(Rt_->Device.Device);

	VkFenceCreateInfo fenceInfo = {};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	VkFence fence = VK_NULL_HANDLE;
	VkResult result = vkCreateFence(device, &fenceInfo, nullptr, &fence);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to create fence");
	}

	return fence;
}

void OaVkVideoQueue::Destroy()
{
	VkDevice device = Rt_ ? static_cast<VkDevice>(Rt_->Device.Device) : VK_NULL_HANDLE;

	if (CmdPool_ != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
		vkDestroyCommandPool(device, CmdPool_, nullptr);
		CmdPool_ = VK_NULL_HANDLE;
	}

	Rt_ = nullptr;
	Queue_ = VK_NULL_HANDLE;
	QueueFamilyIndex_ = 0;
}

void OaVkVideoQueue::MoveFrom(OaVkVideoQueue&& InOther) noexcept
{
	Rt_ = InOther.Rt_;
	Queue_ = InOther.Queue_;
	QueueFamilyIndex_ = InOther.QueueFamilyIndex_;
	CmdPool_ = InOther.CmdPool_;
	Type_ = InOther.Type_;

	InOther.Rt_ = nullptr;
	InOther.Queue_ = VK_NULL_HANDLE;
	InOther.QueueFamilyIndex_ = 0;
	InOther.CmdPool_ = VK_NULL_HANDLE;
}
