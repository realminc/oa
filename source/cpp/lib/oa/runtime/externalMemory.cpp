#include <oa/runtime/externalMemory.h>
#include "engine/deviceAccess.h"

// Linux media-capture interop; this is not a distributed transport.
#include <oa/runtime/device.h>
#include <oa/runtime/engine.h>
#include <vkl/vkl.h>

#ifdef __linux__
#include <unistd.h>
#endif

namespace {

bool hasEnabledExtension(const oavk::Device& inDevice, oa::StringView inName) {
	for (const auto& extension : inDevice.info.software.enabledDeviceExtensions) {
		if (extension == inName) return true;
	}
	return false;
}

} // namespace

oa::ImportedDmaBufImage::ImportedDmaBufImage(
	oa::ImportedDmaBufImage&& inOther) noexcept
	: engine_(inOther.engine_)
	, image_(inOther.image_)
	, view_(inOther.view_)
	, memory_(inOther.memory_)
	, format_(inOther.format_)
	, width_(inOther.width_)
	, height_(inOther.height_)
{
	inOther.engine_ = nullptr;
	inOther.image_ = VK_NULL_HANDLE;
	inOther.view_ = VK_NULL_HANDLE;
	inOther.memory_ = VK_NULL_HANDLE;
}

oa::ImportedDmaBufImage& oa::ImportedDmaBufImage::operator=(
	oa::ImportedDmaBufImage&& inOther) noexcept
{
	if (this != &inOther) {
		reset_();
		engine_ = inOther.engine_;
		image_ = inOther.image_;
		view_ = inOther.view_;
		memory_ = inOther.memory_;
		format_ = inOther.format_;
		width_ = inOther.width_;
		height_ = inOther.height_;
		inOther.engine_ = nullptr;
		inOther.image_ = VK_NULL_HANDLE;
		inOther.view_ = VK_NULL_HANDLE;
		inOther.memory_ = VK_NULL_HANDLE;
	}
	return *this;
}

oa::ImportedDmaBufImage::~ImportedDmaBufImage() { reset_(); }

oa::Result<oa::ImportedDmaBufImage> oa::ImportedDmaBufImage::import(
	oa::Engine& inEngine, const oa::DmaBufImageDesc& inDesc)
{
#if !defined(__linux__) || !defined(VK_EXT_external_memory_dma_buf) || !defined(VK_EXT_image_drm_format_modifier)
	(void)inEngine;
	(void)inDesc;
	return oa::Status::unimplemented("DMA-BUF image import requires Linux vulkan DRM-modifier support");
#else
	if (inDesc.fd < 0 or inDesc.width == 0U or inDesc.height == 0U
		or inDesc.format == VK_FORMAT_UNDEFINED or inDesc.rowPitch == 0U) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"DMA-BUF image import requires fd, extent, format and row pitch");
	}
	auto& deviceInfo = oa::EngineDeviceAccess::get(inEngine);
	if (not hasEnabledExtension(deviceInfo, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME)
		or not hasEnabledExtension(deviceInfo, VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME)
		or not hasEnabledExtension(deviceInfo, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME)) {
		return oa::Status::unimplemented(
			"Device lacks enabled DMA-BUF, DRM-modifier or foreign-queue support");
	}
	if (deviceInfo.deviceDispatch.vkGetMemoryFdPropertiesKHR == nullptr) {
		return oa::Status::unimplemented("vkGetMemoryFdPropertiesKHR is unavailable");
	}

	VkPhysicalDeviceExternalImageFormatInfo externalInfo = {};
	externalInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
	externalInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
	VkPhysicalDeviceImageDrmFormatModifierInfoEXT modifierInfo = {};
	modifierInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT;
	modifierInfo.pNext = &externalInfo;
	modifierInfo.drmFormatModifier = inDesc.modifier;
	modifierInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VkPhysicalDeviceImageFormatInfo2 formatInfo = {};
	formatInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
	formatInfo.pNext = &modifierInfo;
	formatInfo.format = inDesc.format;
	formatInfo.type = VK_IMAGE_TYPE_2D;
	formatInfo.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
	formatInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
	VkExternalImageFormatProperties externalProperties = {};
	externalProperties.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;
	VkImageFormatProperties2 formatProperties = {};
	formatProperties.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
	formatProperties.pNext = &externalProperties;
	VkResult result = deviceInfo.instanceDispatch.vkGetPhysicalDeviceImageFormatProperties2(
		static_cast<VkPhysicalDevice>(deviceInfo.physicalDevice),
		&formatInfo, &formatProperties);
	if (result != VK_SUCCESS
		or (externalProperties.externalMemoryProperties.externalMemoryFeatures
			& VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) == 0U) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"DMA-BUF format/modifier is not importable by this vulkan device");
	}

	VkSubresourceLayout planeLayout = {};
	planeLayout.offset = inDesc.offset;
	planeLayout.rowPitch = inDesc.rowPitch;
	VkImageDrmFormatModifierExplicitCreateInfoEXT explicitModifier = {};
	explicitModifier.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT;
	explicitModifier.drmFormatModifier = inDesc.modifier;
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
	imageInfo.format = inDesc.format;
	imageInfo.extent = { inDesc.width, inDesc.height, 1U };
	imageInfo.mipLevels = 1U;
	imageInfo.arrayLayers = 1U;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
	imageInfo.usage = formatInfo.usage;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	oa::ImportedDmaBufImage imported;
	imported.engine_ = &inEngine;
	result = deviceInfo.deviceDispatch.vkCreateImage(
		static_cast<VkDevice>(deviceInfo.device), &imageInfo, nullptr, &imported.image_);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			"vkCreateImage for DMA-BUF import failed");
	}

	const int importedFd = ::dup(inDesc.fd);
	if (importedFd < 0) {
		imported.reset_();
		return oa::Status::error(oa::StatusCode::ResourceExhausted,
			"Could not duplicate producer DMA-BUF fd");
	}
	VkMemoryFdPropertiesKHR fdProperties = {};
	fdProperties.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
	result = deviceInfo.deviceDispatch.vkGetMemoryFdPropertiesKHR(
		static_cast<VkDevice>(deviceInfo.device),
		VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
		importedFd, &fdProperties);
	if (result != VK_SUCCESS) {
		::close(importedFd);
		imported.reset_();
		return oa::Status::error(oa::StatusCode::VulkanError,
			"vkGetMemoryFdPropertiesKHR for DMA-BUF failed");
	}
	VkMemoryRequirements requirements = {};
	deviceInfo.deviceDispatch.vkGetImageMemoryRequirements(
		static_cast<VkDevice>(deviceInfo.device), imported.image_, &requirements);
	const oa::U32 compatibleTypes = requirements.memoryTypeBits & fdProperties.memoryTypeBits;
	oa::U32 memoryType = UINT32_MAX;
	for (oa::U32 idx = 0U; idx < 32U; ++idx) {
		if ((compatibleTypes & (1U << idx)) != 0U) { memoryType = idx; break; }
	}
	if (memoryType == UINT32_MAX) {
		::close(importedFd);
		imported.reset_();
		return oa::Status::error(oa::StatusCode::Unavailable,
			"DMA-BUF has no compatible vulkan memory type");
	}

	VkMemoryDedicatedAllocateInfo dedicated = {};
	dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
	dedicated.image = imported.image_;
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
	result = deviceInfo.deviceDispatch.vkAllocateMemory(
		static_cast<VkDevice>(deviceInfo.device),
		&allocationInfo, nullptr, &imported.memory_);
	if (result != VK_SUCCESS) {
		::close(importedFd); // vulkan consumes it only after successful import.
		imported.reset_();
		return oa::Status::error(oa::StatusCode::VulkanError,
			"vkAllocateMemory for DMA-BUF import failed");
	}
	result = deviceInfo.deviceDispatch.vkBindImageMemory(
		static_cast<VkDevice>(deviceInfo.device),
		imported.image_, imported.memory_, 0U);
	if (result != VK_SUCCESS) {
		imported.reset_();
		return oa::Status::error(oa::StatusCode::VulkanError,
			"vkBindImageMemory for DMA-BUF import failed");
	}
	VkImageViewCreateInfo viewInfo = {};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = imported.image_;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = inDesc.format;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.levelCount = 1U;
	viewInfo.subresourceRange.layerCount = 1U;
	result = deviceInfo.deviceDispatch.vkCreateImageView(
		static_cast<VkDevice>(deviceInfo.device),
		&viewInfo, nullptr, &imported.view_);
	if (result != VK_SUCCESS) {
		imported.reset_();
		return oa::Status::error(oa::StatusCode::VulkanError,
			"vkCreateImageView for DMA-BUF import failed");
	}
	imported.format_ = inDesc.format;
	imported.width_ = inDesc.width;
	imported.height_ = inDesc.height;
	return imported;
#endif
}

void oa::ImportedDmaBufImage::reset_() noexcept {
	if (engine_ != nullptr) {
		const auto& deviceInfo = oa::EngineDeviceAccess::get(*engine_);
		VkDevice device = static_cast<VkDevice>(deviceInfo.device);
		if (view_ != VK_NULL_HANDLE) {
			deviceInfo.deviceDispatch.vkDestroyImageView(device, view_, nullptr);
		}
		if (image_ != VK_NULL_HANDLE) {
			deviceInfo.deviceDispatch.vkDestroyImage(device, image_, nullptr);
		}
		if (memory_ != VK_NULL_HANDLE) {
			deviceInfo.deviceDispatch.vkFreeMemory(device, memory_, nullptr);
		}
	}
	engine_ = nullptr;
	image_ = VK_NULL_HANDLE;
	view_ = VK_NULL_HANDLE;
	memory_ = VK_NULL_HANDLE;
	format_ = VK_FORMAT_UNDEFINED;
	width_ = 0U;
	height_ = 0U;
}
