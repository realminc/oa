#include <Oa/Runtime/ExternalMemory.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Device.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Runtime/OaVma.h>
#include <Oa/Core/Log.h>

#include <utility>

#ifdef __linux__
#include <unistd.h>
#endif

namespace {

bool HasEnabledExtension(const OaVkDevice& InDevice, OaStringView InName) {
	for (const auto& extension : InDevice.Info.Software.EnabledDeviceExtensions) {
		if (extension == InName) return true;
	}
	return false;
}

} // namespace

OaExternalBuffer::OaExternalBuffer(OaExternalBuffer&& InOther) noexcept
	: Fd(InOther.Fd), Size(InOther.Size), SourceNode(InOther.SourceNode) {
	InOther.Fd = -1;
	InOther.Size = 0;
	InOther.SourceNode = 0;
}

OaExternalBuffer& OaExternalBuffer::operator=(OaExternalBuffer&& InOther) noexcept {
	if (this != &InOther) {
		Close();
		Fd = InOther.Fd;
		Size = InOther.Size;
		SourceNode = InOther.SourceNode;
		InOther.Fd = -1;
		InOther.Size = 0;
		InOther.SourceNode = 0;
	}
	return *this;
}

OaExternalBuffer::~OaExternalBuffer() { Close(); }

void OaExternalBuffer::Close() {
#ifdef __linux__
	if (Fd >= 0) {
		::close(Fd);
		Fd = -1;
	}
#endif
}

OaBool OaCanShareOpaqueFd(const OaVkDevice& InSrc, const OaVkDevice& InDst) {
#ifdef __linux__
	if (!InSrc.Info.Software.HasExternalMemoryFd || !InDst.Info.Software.HasExternalMemoryFd) return false;
	// Same vendor required for compatible memory types
	return InSrc.Info.Hardware.VendorId == InDst.Info.Hardware.VendorId;
#else
	(void)InSrc;
	(void)InDst;
	return false;
#endif
}

OaResult<OaExternalBuffer> OaExportBufferFd(
	const OaVkDevice& InDevice,
	OaVma& InAllocator,
	const OaVkBuffer& InBuf,
	OaU32 InSourceNode
) {
#ifdef __linux__
	if (!InDevice.Info.Software.HasExternalMemoryFd) {
		return OaStatus::Unimplemented("OaExportBufferFd: device lacks VK_KHR_external_memory_fd");
	}
	if (!InBuf.Allocation) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "buffer has no allocation");
	}

	OaVmaAllocationInfo allocInfo{};
	OaVmaGetAllocationInfo(
		static_cast<OaVmaAllocator>(InAllocator.Allocator),
		static_cast<OaVmaAllocation>(InBuf.Allocation),
		&allocInfo
	);
	VkDeviceMemory devMem = static_cast<VkDeviceMemory>(allocInfo.deviceMemory);
	if (!devMem) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaVma allocation has no VkDeviceMemory (virtual allocation?)"
		);
	}

	VkMemoryGetFdInfoKHR getFdInfo{};
	getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
	getFdInfo.memory = devMem;
	getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

	int fd = -1;
	auto pfn = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
		vkGetDeviceProcAddr(static_cast<VkDevice>(InDevice.Device), "vkGetMemoryFdKHR"));
	if (!pfn) {
		return OaStatus::Unimplemented("vkGetMemoryFdKHR not available");
	}

	VkResult r = pfn(static_cast<VkDevice>(InDevice.Device), &getFdInfo, &fd);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkGetMemoryFdKHR failed");
	}

	OaExternalBuffer ext;
	ext.Fd = fd;
	ext.Size = InBuf.Size;
	ext.SourceNode = InSourceNode;
	return std::move(ext);
#else
	(void)InDevice;
	(void)InAllocator;
	(void)InBuf;
	(void)InSourceNode;
	return OaStatus::Unimplemented("OaExportBufferFd: not supported on this platform");
#endif
}

OaResult<OaVkBuffer> OaImportBufferFd(
	const OaVkDevice& InDevice,
	OaVma& InAllocator,
	OaExternalBuffer&& InExt)
{
#ifdef __linux__
	if (!InDevice.Info.Software.HasExternalMemoryFd) {
		return OaStatus::Unimplemented("OaImportBufferFd: device lacks VK_KHR_external_memory_fd");
	}
	if (InExt.Fd < 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "invalid fd");
	}

	// Create buffer with external memory
	VkExternalMemoryBufferCreateInfo extBufInfo{};
	extBufInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
	extBufInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

	VkBufferCreateInfo bufCI{};
	bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufCI.pNext = &extBufInfo;
	bufCI.size = InExt.Size;
	bufCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |	VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkBuffer buffer = VK_NULL_HANDLE;
	VkResult r = vkCreateBuffer(static_cast<VkDevice>(InDevice.Device), &bufCI, nullptr, &buffer);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkCreateBuffer for import failed");
	}

	VkMemoryRequirements memReqs{};
	vkGetBufferMemoryRequirements(static_cast<VkDevice>(InDevice.Device), buffer, &memReqs);

	VkImportMemoryFdInfoKHR importFdInfo{};
	importFdInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
	importFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
	importFdInfo.fd = InExt.Fd;

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.pNext = &importFdInfo;
	allocInfo.allocationSize = memReqs.size;

	// Find a compatible memory type (device-local preferred)
	VkPhysicalDeviceMemoryProperties memProps{};
	vkGetPhysicalDeviceMemoryProperties(
		static_cast<VkPhysicalDevice>(InDevice.PhysicalDevice), &memProps
	);

	OaU32 memTypeIdx = UINT32_MAX;
	for (OaU32 i = 0; i < memProps.memoryTypeCount; ++i) {
		if (memReqs.memoryTypeBits & (1u << i)) {
			memTypeIdx = i;
			if (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
				break;
		}
	}
	if (memTypeIdx == UINT32_MAX) {
		vkDestroyBuffer(static_cast<VkDevice>(InDevice.Device), buffer, nullptr);
		return OaStatus::Error(OaStatusCode::VulkanError, "no compatible memory type for import");
	}
	allocInfo.memoryTypeIndex = memTypeIdx;

	VkDeviceMemory memory = VK_NULL_HANDLE;
	r = vkAllocateMemory(static_cast<VkDevice>(InDevice.Device), &allocInfo, nullptr, &memory);
	if (r != VK_SUCCESS) {
		vkDestroyBuffer(static_cast<VkDevice>(InDevice.Device), buffer, nullptr);
		return OaStatus::Error(OaStatusCode::VulkanError, "vkAllocateMemory for import failed");
	}
	// Vulkan owns and closes an imported OPAQUE_FD after successful allocation.
	// Clear the RAII wrapper so it cannot close the same descriptor again.
	InExt.Fd = -1;

	r = vkBindBufferMemory(static_cast<VkDevice>(InDevice.Device), buffer, memory, 0);
	if (r != VK_SUCCESS) {
		vkFreeMemory(static_cast<VkDevice>(InDevice.Device), memory, nullptr);
		vkDestroyBuffer(static_cast<VkDevice>(InDevice.Device), buffer, nullptr);
		return OaStatus::Error(OaStatusCode::VulkanError, "vkBindBufferMemory for import failed");
	}

	OaVkBuffer result;
	result.Buffer = buffer;
	result.Allocation = memory;
	result.Size = InExt.Size;
	result.Capacity = InExt.Size;
	result.MappedPtr = nullptr;
	result.Flags = OA_VK_BUFFER_FLAG_IMPORTED;
	result.Placement = OaMemoryPlacement::DeviceLocal;
	result.NodeIndex = 0;

	(void)InAllocator;
	return result;
#else
	(void)InDevice;
	(void)InAllocator;
	(void)InExt;
	return OaStatus::Unimplemented("OaImportBufferFd: not supported on this platform");
#endif
}

OaResult<OaVkBuffer> OaImportBufferFd(
	const OaVkDevice& InDevice,
	OaVma& InAllocator,
	const OaExternalBuffer& InExt)
{
#ifdef __linux__
	if (!InExt.IsValid()) {
		return OaStatus::InvalidArgument("OaImportBufferFd: invalid fd");
	}
	OaExternalBuffer duplicate;
	duplicate.Fd = ::dup(InExt.Fd);
	if (duplicate.Fd < 0) {
		return OaStatus::Error(OaStatusCode::ResourceExhausted,
			"OaImportBufferFd: could not duplicate fd");
	}
	duplicate.Size = InExt.Size;
	duplicate.SourceNode = InExt.SourceNode;
	return OaImportBufferFd(InDevice, InAllocator, std::move(duplicate));
#else
	(void)InDevice;
	(void)InAllocator;
	(void)InExt;
	return OaStatus::Unimplemented("OaImportBufferFd: not supported on this platform");
#endif
}

OaImportedDmaBufImage::OaImportedDmaBufImage(
	OaImportedDmaBufImage&& InOther) noexcept
	: Engine_(InOther.Engine_)
	, Image_(InOther.Image_)
	, View_(InOther.View_)
	, Memory_(InOther.Memory_)
	, Format_(InOther.Format_)
	, Width_(InOther.Width_)
	, Height_(InOther.Height_)
{
	InOther.Engine_ = nullptr;
	InOther.Image_ = VK_NULL_HANDLE;
	InOther.View_ = VK_NULL_HANDLE;
	InOther.Memory_ = VK_NULL_HANDLE;
}

OaImportedDmaBufImage& OaImportedDmaBufImage::operator=(
	OaImportedDmaBufImage&& InOther) noexcept
{
	if (this != &InOther) {
		Destroy();
		Engine_ = InOther.Engine_;
		Image_ = InOther.Image_;
		View_ = InOther.View_;
		Memory_ = InOther.Memory_;
		Format_ = InOther.Format_;
		Width_ = InOther.Width_;
		Height_ = InOther.Height_;
		InOther.Engine_ = nullptr;
		InOther.Image_ = VK_NULL_HANDLE;
		InOther.View_ = VK_NULL_HANDLE;
		InOther.Memory_ = VK_NULL_HANDLE;
	}
	return *this;
}

OaImportedDmaBufImage::~OaImportedDmaBufImage() { Destroy(); }

OaResult<OaImportedDmaBufImage> OaImportedDmaBufImage::Import(
	OaComputeEngine& InEngine, const OaDmaBufImageDesc& InDesc)
{
#if !defined(__linux__) || !defined(VK_EXT_external_memory_dma_buf) || !defined(VK_EXT_image_drm_format_modifier)
	(void)InEngine;
	(void)InDesc;
	return OaStatus::Unimplemented("DMA-BUF image import requires Linux Vulkan DRM-modifier support");
#else
	if (InDesc.Fd < 0 or InDesc.Width == 0U or InDesc.Height == 0U
		or InDesc.Format == VK_FORMAT_UNDEFINED or InDesc.RowPitch == 0U) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"DMA-BUF image import requires fd, extent, format and row pitch");
	}
	auto& deviceInfo = InEngine.Device;
	if (not HasEnabledExtension(deviceInfo, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME)
		or not HasEnabledExtension(deviceInfo, VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME)
		or not HasEnabledExtension(deviceInfo, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME)) {
		return OaStatus::Unimplemented(
			"Device lacks enabled DMA-BUF, DRM-modifier or foreign-queue support");
	}
	if (vkGetMemoryFdPropertiesKHR == nullptr) {
		return OaStatus::Unimplemented("vkGetMemoryFdPropertiesKHR is unavailable");
	}

	VkPhysicalDeviceExternalImageFormatInfo externalInfo = {};
	externalInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
	externalInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
	VkPhysicalDeviceImageDrmFormatModifierInfoEXT modifierInfo = {};
	modifierInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT;
	modifierInfo.pNext = &externalInfo;
	modifierInfo.drmFormatModifier = InDesc.Modifier;
	modifierInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VkPhysicalDeviceImageFormatInfo2 formatInfo = {};
	formatInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
	formatInfo.pNext = &modifierInfo;
	formatInfo.format = InDesc.Format;
	formatInfo.type = VK_IMAGE_TYPE_2D;
	formatInfo.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
	formatInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
	VkExternalImageFormatProperties externalProperties = {};
	externalProperties.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;
	VkImageFormatProperties2 formatProperties = {};
	formatProperties.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
	formatProperties.pNext = &externalProperties;
	VkResult result = vkGetPhysicalDeviceImageFormatProperties2(
		static_cast<VkPhysicalDevice>(deviceInfo.PhysicalDevice),
		&formatInfo, &formatProperties);
	if (result != VK_SUCCESS
		or (externalProperties.externalMemoryProperties.externalMemoryFeatures
			& VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) == 0U) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"DMA-BUF format/modifier is not importable by this Vulkan device");
	}

	VkSubresourceLayout planeLayout = {};
	planeLayout.offset = InDesc.Offset;
	planeLayout.rowPitch = InDesc.RowPitch;
	VkImageDrmFormatModifierExplicitCreateInfoEXT explicitModifier = {};
	explicitModifier.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT;
	explicitModifier.drmFormatModifier = InDesc.Modifier;
	explicitModifier.drmFormatModifierPlaneCount = 1U;
	explicitModifier.pPlaneLayouts = &planeLayout;
	VkExternalMemoryImageCreateInfo externalCreate = {};
	externalCreate.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
	externalCreate.pNext = &explicitModifier;
	externalCreate.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
	VkImageCreateInfo imageInfo = {};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.pNext = &externalCreate;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = InDesc.Format;
	imageInfo.extent = { InDesc.Width, InDesc.Height, 1U };
	imageInfo.mipLevels = 1U;
	imageInfo.arrayLayers = 1U;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
	imageInfo.usage = formatInfo.usage;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	OaImportedDmaBufImage imported;
	imported.Engine_ = &InEngine;
	result = vkCreateImage(
		static_cast<VkDevice>(deviceInfo.Device), &imageInfo, nullptr, &imported.Image_);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError,
			"vkCreateImage for DMA-BUF import failed");
	}

	const int importedFd = ::dup(InDesc.Fd);
	if (importedFd < 0) {
		imported.Destroy();
		return OaStatus::Error(OaStatusCode::ResourceExhausted,
			"Could not duplicate producer DMA-BUF fd");
	}
	VkMemoryFdPropertiesKHR fdProperties = {};
	fdProperties.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
	result = vkGetMemoryFdPropertiesKHR(
		static_cast<VkDevice>(deviceInfo.Device),
		VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
		importedFd, &fdProperties);
	if (result != VK_SUCCESS) {
		::close(importedFd);
		imported.Destroy();
		return OaStatus::Error(OaStatusCode::VulkanError,
			"vkGetMemoryFdPropertiesKHR for DMA-BUF failed");
	}
	VkMemoryRequirements requirements = {};
	vkGetImageMemoryRequirements(
		static_cast<VkDevice>(deviceInfo.Device), imported.Image_, &requirements);
	const OaU32 compatibleTypes = requirements.memoryTypeBits & fdProperties.memoryTypeBits;
	OaU32 memoryType = UINT32_MAX;
	for (OaU32 idx = 0U; idx < 32U; ++idx) {
		if ((compatibleTypes & (1U << idx)) != 0U) { memoryType = idx; break; }
	}
	if (memoryType == UINT32_MAX) {
		::close(importedFd);
		imported.Destroy();
		return OaStatus::Error(OaStatusCode::Unavailable,
			"DMA-BUF has no compatible Vulkan memory type");
	}

	VkMemoryDedicatedAllocateInfo dedicated = {};
	dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
	dedicated.image = imported.Image_;
	VkImportMemoryFdInfoKHR importInfo = {};
	importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
	importInfo.pNext = &dedicated;
	importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
	importInfo.fd = importedFd;
	VkMemoryAllocateInfo allocationInfo = {};
	allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocationInfo.pNext = &importInfo;
	allocationInfo.allocationSize = requirements.size;
	allocationInfo.memoryTypeIndex = memoryType;
	result = vkAllocateMemory(static_cast<VkDevice>(deviceInfo.Device),
		&allocationInfo, nullptr, &imported.Memory_);
	if (result != VK_SUCCESS) {
		::close(importedFd); // Vulkan consumes it only after successful import.
		imported.Destroy();
		return OaStatus::Error(OaStatusCode::VulkanError,
			"vkAllocateMemory for DMA-BUF import failed");
	}
	result = vkBindImageMemory(static_cast<VkDevice>(deviceInfo.Device),
		imported.Image_, imported.Memory_, 0U);
	if (result != VK_SUCCESS) {
		imported.Destroy();
		return OaStatus::Error(OaStatusCode::VulkanError,
			"vkBindImageMemory for DMA-BUF import failed");
	}
	VkImageViewCreateInfo viewInfo = {};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = imported.Image_;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = InDesc.Format;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.levelCount = 1U;
	viewInfo.subresourceRange.layerCount = 1U;
	result = vkCreateImageView(static_cast<VkDevice>(deviceInfo.Device),
		&viewInfo, nullptr, &imported.View_);
	if (result != VK_SUCCESS) {
		imported.Destroy();
		return OaStatus::Error(OaStatusCode::VulkanError,
			"vkCreateImageView for DMA-BUF import failed");
	}
	imported.Format_ = InDesc.Format;
	imported.Width_ = InDesc.Width;
	imported.Height_ = InDesc.Height;
	return imported;
#endif
}

void OaImportedDmaBufImage::Destroy() {
	if (Engine_ != nullptr) {
		VkDevice device = static_cast<VkDevice>(Engine_->Device.Device);
		if (View_ != VK_NULL_HANDLE) vkDestroyImageView(device, View_, nullptr);
		if (Image_ != VK_NULL_HANDLE) vkDestroyImage(device, Image_, nullptr);
		if (Memory_ != VK_NULL_HANDLE) vkFreeMemory(device, Memory_, nullptr);
	}
	Engine_ = nullptr;
	Image_ = VK_NULL_HANDLE;
	View_ = VK_NULL_HANDLE;
	Memory_ = VK_NULL_HANDLE;
	Format_ = VK_FORMAT_UNDEFINED;
	Width_ = 0U;
	Height_ = 0U;
}
