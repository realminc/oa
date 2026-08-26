#include "rendererInternal.h"

#include <oa/core/log.h>
#include <oa/render/fnMesh.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/eventAccess.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/deviceAccess.h>
#include <oa/runtime/graphicsStream.h>
#include <oa/runtime/oaVma.h>
#include <oa/runtime/spirv.h>
#include <oa/runtime/stream.h>

#include "../runtime/textureAccess.h"

#include <stddef.h>

namespace {

constexpr VkFormat ColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat DepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr oa::U32 MaxTargetSlots = 4U;

struct RenderPushConstants {
	oa::vlm::Mat4 ViewProjection;
	oa::vlm::Vec4 LightDirectionAmbient;
};
static_assert(sizeof(RenderPushConstants) == 80U);

enum class RenderSlotState : oa::U8 {
	Free,
	Recording,
	Submitted,
	Retired,
};

struct RenderTarget {
	VkImage colorImage = VK_NULL_HANDLE;
	VkImageView colorView = VK_NULL_HANDLE;
	OaVmaAllocation colorAllocation = VK_NULL_HANDLE;
	VkImageLayout colorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VkImage colorMsaaImage = VK_NULL_HANDLE;
	VkImageView colorMsaaView = VK_NULL_HANDLE;
	OaVmaAllocation colorMsaaAllocation = VK_NULL_HANDLE;
	VkImageLayout colorMsaaLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VkImage depthImage = VK_NULL_HANDLE;
	VkImageView depthView = VK_NULL_HANDLE;
	OaVmaAllocation depthAllocation = VK_NULL_HANDLE;
	VkImageLayout depthLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VkImage depthMsaaImage = VK_NULL_HANDLE;
	VkImageView depthMsaaView = VK_NULL_HANDLE;
	OaVmaAllocation depthMsaaAllocation = VK_NULL_HANDLE;
	VkImageLayout depthMsaaLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	oavk::Buffer colorReadback;
	oavk::Buffer depthReadback;
};

struct RenderSlot {
	RenderSlotState state = RenderSlotState::Free;
	oa::U64 generation = 0U;
	RenderTarget target;
	oavk::Buffer vertexBuffer;
	oavk::Buffer indexBuffer;
	oa::U32 indexCount = 0U;
	oa::Optional<oa::GraphicsStreamLease> streamLease;
	oa::Event producer;
	oa::Event consumer;
};

[[nodiscard]] oa::U64 nextGeneration(oa::U64 inGeneration) noexcept {
	++inGeneration;
	return inGeneration == 0U ? 1U : inGeneration;
}

[[nodiscard]] bool checkedMultiply(
	oa::U64 inA, oa::U64 inB, oa::U64& outResult) noexcept {
	if (inA != 0U && inB > oa::Limits<oa::U64>::max() / inA) {
		return false;
	}
	outResult = inA * inB;
	return true;
}

[[nodiscard]] VkSampleCountFlagBits toVkSampleCount(
	oa::U32 inSampleCount) noexcept {
	switch (inSampleCount) {
		case 1U: return VK_SAMPLE_COUNT_1_BIT;
		case 2U: return VK_SAMPLE_COUNT_2_BIT;
		case 4U: return VK_SAMPLE_COUNT_4_BIT;
		case 8U: return VK_SAMPLE_COUNT_8_BIT;
		case 16U: return VK_SAMPLE_COUNT_16_BIT;
		case 32U: return VK_SAMPLE_COUNT_32_BIT;
		case 64U: return VK_SAMPLE_COUNT_64_BIT;
		default: return static_cast<VkSampleCountFlagBits>(0U);
	}
}

[[nodiscard]] bool isFiniteVlm(const oa::vlm::Mat4& inMatrix) noexcept {
	for (oa::U32 row = 0U; row < 4U; ++row) {
		for (oa::U32 column = 0U; column < 4U; ++column) {
			if (not oa::isFinite(inMatrix.m[row][column])) return false;
		}
	}
	return true;
}

[[nodiscard]] bool isFiniteVlm(const oa::vlm::Vec3& inVector) noexcept {
	return oa::isFinite(inVector.x)
		and oa::isFinite(inVector.y)
		and oa::isFinite(inVector.z);
}

[[nodiscard]] oa::Status validateTargetExtent(
	oa::Engine& inEngine,
	const OaVkInstanceTable& inInstanceTable,
	oa::U32 inWidth,
	oa::U32 inHeight,
	VkSampleCountFlagBits inSampleCount) {
	if (inWidth == 0U or inHeight == 0U) {
		return oa::Status::invalidArgument(
			"oa::Renderer target dimensions must be non-zero");
	}
	const VkPhysicalDevice physicalDevice =
		static_cast<VkPhysicalDevice>(oa::EngineDeviceAccess::get(inEngine).physicalDevice);
	if (physicalDevice == VK_NULL_HANDLE
		or inInstanceTable.vkGetPhysicalDeviceProperties == nullptr
		or inInstanceTable.vkGetPhysicalDeviceImageFormatProperties == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer requires physical-device capability queries");
	}

	VkPhysicalDeviceProperties properties{};
	inInstanceTable.vkGetPhysicalDeviceProperties(physicalDevice, &properties);
	const auto& limits = properties.limits;
	if (inSampleCount == 0U) {
		return oa::Status::invalidArgument(
			"oa::Renderer sample count must be one of 1, 2, 4, 8, 16, 32, or 64");
	}
	const VkSampleCountFlags framebufferSamples =
		limits.framebufferColorSampleCounts
		& limits.framebufferDepthSampleCounts;
	if ((framebufferSamples & inSampleCount) == 0U) {
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			"oa::Renderer sample count is unsupported by the color/depth framebuffer limits");
	}
	if (inWidth > limits.maxImageDimension2D
		or inHeight > limits.maxImageDimension2D
		or inWidth > limits.maxFramebufferWidth
		or inHeight > limits.maxFramebufferHeight
		or inWidth > limits.maxViewportDimensions[0]
		or inHeight > limits.maxViewportDimensions[1]
		or static_cast<oa::F32>(inWidth) > limits.viewportBoundsRange[1]
		or static_cast<oa::F32>(inHeight) > limits.viewportBoundsRange[1]) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::Renderer target exceeds queried image, framebuffer, or viewport limits");
	}
	oa::U64 pixelCount = 0U;
	oa::U64 readbackBytes = 0U;
	if (not checkedMultiply(inWidth, inHeight, pixelCount)
		or not checkedMultiply(pixelCount, 4U, readbackBytes)) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::Renderer readback size overflows");
	}
	const oa::U32 apiVersion =
		oa::EngineDeviceAccess::get(inEngine).info.software.apiVersionPacked;
	const bool core13 = VK_API_VERSION_MAJOR(apiVersion) > 1U
		or (VK_API_VERSION_MAJOR(apiVersion) == 1U
			and VK_API_VERSION_MINOR(apiVersion) >= 3U);
	if (core13 or inSampleCount != VK_SAMPLE_COUNT_1_BIT) {
		if (inInstanceTable.vkGetPhysicalDeviceProperties2 == nullptr) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::Renderer requires physical-device properties2 queries");
		}
		VkPhysicalDeviceMaintenance4Properties maintenance4{};
		maintenance4.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES;
		VkPhysicalDeviceDepthStencilResolveProperties depthResolve{};
		depthResolve.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES;
		if (inSampleCount != VK_SAMPLE_COUNT_1_BIT) {
			depthResolve.pNext = core13 ? &maintenance4 : nullptr;
		}
		VkPhysicalDeviceProperties2 properties2{};
		properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		properties2.pNext = inSampleCount != VK_SAMPLE_COUNT_1_BIT
			? static_cast<void*>(&depthResolve)
			: static_cast<void*>(&maintenance4);
		inInstanceTable.vkGetPhysicalDeviceProperties2(
			physicalDevice, &properties2);
		if (core13 and readbackBytes > maintenance4.maxBufferSize) {
			return oa::Status::error(
				oa::StatusCode::OutOfRange,
				"oa::Renderer readback buffer exceeds the queried maintenance4 limit");
		}
		// The public readback exposes the nearest depth represented by each pixel.
		// MIN preserves that contract for the renderer's LESS depth comparison.
		if (inSampleCount != VK_SAMPLE_COUNT_1_BIT
			and (depthResolve.supportedDepthResolveModes
				& VK_RESOLVE_MODE_MIN_BIT) == 0U) {
			return oa::Status::error(
				oa::StatusCode::Unavailable,
				"oa::Renderer multisampling requires MIN depth resolve support");
		}
	}

	struct ImageRequirement {
		VkFormat format;
		VkImageUsageFlags usage;
		VkSampleCountFlagBits samples;
	};
	const ImageRequirement requirements[4] = {
		{
			ColorFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
				| VK_IMAGE_USAGE_TRANSFER_SRC_BIT
				| VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_SAMPLE_COUNT_1_BIT,
		},
		{
			DepthFormat,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
				| VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
			VK_SAMPLE_COUNT_1_BIT,
		},
		{
			ColorFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			inSampleCount,
		},
		{
			DepthFormat,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			inSampleCount,
		},
	};
	const oa::U32 requirementCount =
		inSampleCount == VK_SAMPLE_COUNT_1_BIT ? 2U : 4U;
	for (oa::U32 index = 0U; index < requirementCount; ++index) {
		const ImageRequirement& requirement = requirements[index];
		VkImageFormatProperties imageProperties{};
		const VkResult result =
			inInstanceTable.vkGetPhysicalDeviceImageFormatProperties(
				physicalDevice,
				requirement.format,
				VK_IMAGE_TYPE_2D,
				VK_IMAGE_TILING_OPTIMAL,
				requirement.usage,
				0U,
				&imageProperties
			);
		if (result == VK_ERROR_FORMAT_NOT_SUPPORTED) {
			return oa::Status::error(
				oa::StatusCode::Unavailable,"oa::Renderer attachment/readback image tuple is unsupported");
		}
		if (result != VK_SUCCESS) {
			return oa::Status::error(
				oa::StatusCode::VulkanError,
				"oa::Renderer image-format capability query failed");
		}
		if (inWidth > imageProperties.maxExtent.width
			or inHeight > imageProperties.maxExtent.height
			or imageProperties.maxMipLevels < 1U
			or imageProperties.maxArrayLayers < 1U) {
			return oa::Status::error(
				oa::StatusCode::OutOfRange,
				"oa::Renderer target exceeds the exact image tuple limits");
		}
		if ((imageProperties.sampleCounts & requirement.samples) == 0U) {
			return oa::Status::error(
				oa::StatusCode::Unavailable,
				"oa::Renderer sample count is unsupported by an exact attachment image tuple");
		}
	}
	return oa::Status::ok();
}

[[nodiscard]] VkShaderModule createShaderModule(
	const OaVkDeviceTable& inDeviceTable,
	VkDevice inDevice,
	const oavk::SpirvEntry& inSpirv) noexcept {
	VkShaderModuleCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	info.codeSize = inSpirv.size;
	info.pCode = reinterpret_cast<const oa::U32*>(inSpirv.data);
	VkShaderModule module = VK_NULL_HANDLE;
	return inDeviceTable.vkCreateShaderModule(
		inDevice, &info, nullptr, &module) == VK_SUCCESS
		? module : VK_NULL_HANDLE;
}

[[nodiscard]] oa::Result<oavk::Buffer> createMappedVertexOrIndexBuffer(
	oa::Engine& inEngine,
	oa::U64 inSize,
	VkBufferUsageFlags inUsage) {
	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = inSize;
	bufferInfo.usage = inUsage;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	OaVmaAllocationCreateInfo allocationInfo{};
	allocationInfo.usage = OA_VMA_MEMORY_USAGE_CPU_TO_GPU;
	allocationInfo.flags = OA_VMA_ALLOCATION_CREATE_MAPPED_BIT
		| OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

	VkBuffer buffer = VK_NULL_HANDLE;
	OaVmaAllocation allocation = VK_NULL_HANDLE;
	OaVmaAllocationInfo allocationResult{};
	if (OaVmaCreateBuffer(
			static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(inEngine).allocator),
			&bufferInfo, &allocationInfo, &buffer, &allocation,
			&allocationResult) != VK_SUCCESS) {
		return oa::Status::error(
			oa::StatusCode::OutOfMemory,
			"oa::Renderer mapped geometry allocation failed");
	}

	oavk::Buffer result;
	result.buffer = buffer;
	result.allocation = allocation;
	result.allocatorIdentity = oa::EngineAllocatorAccess::get(inEngine).allocator;
	result.size = inSize;
	result.capacity = inSize;
	result.mappedPtr = allocationResult.pMappedData;
	result.placement = oa::MemoryPlacement::HostUpload;
	(void)result.observeMutationVersion();
	return result;
}

void destroyTarget(
	oa::Engine& inEngine,
	const OaVkDeviceTable& inDeviceTable,
	RenderTarget& inOutTarget) noexcept {
	const VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(inEngine).device);
	if (inOutTarget.colorView != VK_NULL_HANDLE) {
		inDeviceTable.vkDestroyImageView(
			device, inOutTarget.colorView, nullptr);
	}
	if (inOutTarget.colorMsaaView != VK_NULL_HANDLE) {
		inDeviceTable.vkDestroyImageView(
			device, inOutTarget.colorMsaaView, nullptr);
	}
	if (inOutTarget.depthView != VK_NULL_HANDLE) {
		inDeviceTable.vkDestroyImageView(
			device, inOutTarget.depthView, nullptr);
	}
	if (inOutTarget.depthMsaaView != VK_NULL_HANDLE) {
		inDeviceTable.vkDestroyImageView(
			device, inOutTarget.depthMsaaView, nullptr);
	}
	if (inOutTarget.colorImage != VK_NULL_HANDLE) {
		OaVmaDestroyImage(
			static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(inEngine).allocator),
			inOutTarget.colorImage, inOutTarget.colorAllocation);
	}
	if (inOutTarget.depthImage != VK_NULL_HANDLE) {
		OaVmaDestroyImage(
			static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(inEngine).allocator),
			inOutTarget.depthImage, inOutTarget.depthAllocation);
	}
	if (inOutTarget.colorMsaaImage != VK_NULL_HANDLE) {
		OaVmaDestroyImage(
			static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(inEngine).allocator),
			inOutTarget.colorMsaaImage, inOutTarget.colorMsaaAllocation);
	}
	if (inOutTarget.depthMsaaImage != VK_NULL_HANDLE) {
		OaVmaDestroyImage(
			static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(inEngine).allocator),
			inOutTarget.depthMsaaImage, inOutTarget.depthMsaaAllocation);
	}
	oa::EngineAllocatorAccess::get(inEngine).free(inOutTarget.colorReadback);
	oa::EngineAllocatorAccess::get(inEngine).free(inOutTarget.depthReadback);
	inOutTarget = {};
}

[[nodiscard]] oa::Status createImage(
	oa::Engine& inEngine,
	const OaVkDeviceTable& inDeviceTable,
	oa::U32 inWidth,
	oa::U32 inHeight,
	VkFormat inFormat,
	VkImageUsageFlags inUsage,
	VkImageAspectFlags inAspect,
	VkSampleCountFlagBits inSampleCount,
	VkImage& outImage,
	VkImageView& outView,
	OaVmaAllocation& outAllocation) {
	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = inFormat;
	imageInfo.extent = {inWidth, inHeight, 1U};
	imageInfo.mipLevels = 1U;
	imageInfo.arrayLayers = 1U;
	imageInfo.samples = inSampleCount;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = inUsage;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	OaVmaAllocationCreateInfo allocationInfo{};
	allocationInfo.usage = OA_VMA_MEMORY_USAGE_GPU_ONLY;
	if (OaVmaCreateImage(
			static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(inEngine).allocator),
			&imageInfo, &allocationInfo, &outImage, &outAllocation,
			nullptr) != VK_SUCCESS) {
		return oa::Status::error(
			oa::StatusCode::OutOfMemory,
			"oa::Renderer target image allocation failed");
	}

	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = outImage;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = inFormat;
	viewInfo.subresourceRange.aspectMask = inAspect;
	viewInfo.subresourceRange.levelCount = 1U;
	viewInfo.subresourceRange.layerCount = 1U;
	if (inDeviceTable.vkCreateImageView(
			static_cast<VkDevice>(oa::EngineDeviceAccess::get(inEngine).device),
			&viewInfo, nullptr, &outView) != VK_SUCCESS) {
		OaVmaDestroyImage(
			static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(inEngine).allocator),
			outImage, outAllocation);
		outImage = VK_NULL_HANDLE;
		outAllocation = VK_NULL_HANDLE;
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			"oa::Renderer image-view creation failed");
	}
	return oa::Status::ok();
}

[[nodiscard]] oa::Status createTarget(
	oa::Engine& inEngine,
	const OaVkDeviceTable& inDeviceTable,
	oa::U32 inWidth,
	oa::U32 inHeight,
	VkSampleCountFlagBits inSampleCount,
	RenderTarget& outTarget) {
	oa::U64 pixelCount = 0U;
	oa::U64 colorBytes = 0U;
	oa::U64 depthBytes = 0U;
	if (not checkedMultiply(inWidth, inHeight, pixelCount)
		or not checkedMultiply(pixelCount, 4U, colorBytes)
		or not checkedMultiply(pixelCount, sizeof(oa::F32), depthBytes)) {
		return oa::Status::invalidArgument(
			"oa::Renderer target size overflows");
	}

	oa::Status status = createImage(
		inEngine, inDeviceTable, inWidth, inHeight, ColorFormat,
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
			| VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT,
		VK_SAMPLE_COUNT_1_BIT,
		outTarget.colorImage, outTarget.colorView,
		outTarget.colorAllocation);
	if (not status.isOk()) return status;
	status = createImage(
		inEngine, inDeviceTable, inWidth, inHeight, DepthFormat,
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
			| VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		VK_IMAGE_ASPECT_DEPTH_BIT,
		VK_SAMPLE_COUNT_1_BIT,
		outTarget.depthImage, outTarget.depthView,
		outTarget.depthAllocation);
	if (not status.isOk()) {
		destroyTarget(inEngine, inDeviceTable, outTarget);
		return status;
	}
	if (inSampleCount != VK_SAMPLE_COUNT_1_BIT) {
		status = createImage(
			inEngine, inDeviceTable, inWidth, inHeight, ColorFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			VK_IMAGE_ASPECT_COLOR_BIT,
			inSampleCount,
			outTarget.colorMsaaImage, outTarget.colorMsaaView,
			outTarget.colorMsaaAllocation);
		if (not status.isOk()) {
			destroyTarget(inEngine, inDeviceTable, outTarget);
			return status;
		}
		status = createImage(
			inEngine, inDeviceTable, inWidth, inHeight, DepthFormat,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			VK_IMAGE_ASPECT_DEPTH_BIT,
			inSampleCount,
			outTarget.depthMsaaImage, outTarget.depthMsaaView,
			outTarget.depthMsaaAllocation);
		if (not status.isOk()) {
			destroyTarget(inEngine, inDeviceTable, outTarget);
			return status;
		}
	}
	auto colorReadback = oa::EngineAllocatorAccess::get(inEngine).allocHostReadback(colorBytes);
	if (not colorReadback.isOk()) {
		destroyTarget(inEngine, inDeviceTable, outTarget);
		return colorReadback.getStatus();
	}
	outTarget.colorReadback = oa::move(*colorReadback);
	auto depthReadback = oa::EngineAllocatorAccess::get(inEngine).allocHostReadback(depthBytes);
	if (not depthReadback.isOk()) {
		destroyTarget(inEngine, inDeviceTable, outTarget);
		return depthReadback.getStatus();
	}
	outTarget.depthReadback = oa::move(*depthReadback);
	return oa::Status::ok();
}

} // namespace

class oa::Renderer::MeshImpl final : public oa::Renderer::Impl {
public:
	OaVkInstanceTable InstanceTable{};
	OaVkDeviceTable DeviceTable{};
	oa::RendererConfig renderConfig;
	oa::Vec<RenderSlot> slots;
	oa::U32 activeSlot = oa::Limits<oa::U32>::max();
	oa::U64 TargetGeneration = 1U;
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;

	[[nodiscard]] oa::Status initialize(
		oa::Engine& inEngine,
		const oa::RendererConfig& inRenderConfig);
	[[nodiscard]] oa::Status validateCapabilities() const;
	[[nodiscard]] oa::Status createPipeline();
	[[nodiscard]] oa::Status createSlots();
	[[nodiscard]] oa::Status writeFrameGeometry(
		RenderSlot& inSlot,
		const oa::MeshData& inMesh);
	[[nodiscard]] oa::Status recordFrame(
		RenderSlot& inSlot,
		const oa::CameraState& inCamera);
	void recordReadback(
		VkCommandBuffer inCommandBuffer,
		RenderTarget& inTarget) const;
	[[nodiscard]] oa::Status validateFrame(
		const oa::RenderFrame& inFrame,
		RenderSlotState inRequiredState,
		RenderSlot*& outSlot);
	[[nodiscard]] oa::Status collectRetired();
	void destroyAll() noexcept;

	[[nodiscard]] oa::Status beginMeshFrame(
		const oa::MeshData& inMesh,
		const oa::CameraState& inCamera) override;
	[[nodiscard]] oa::Result<oa::RenderFrame> submitFrame(
		oa::Span<const oa::Event> inDependencies) override;
	[[nodiscard]] oa::Status cancelFrame() override;
	[[nodiscard]] oa::Result<oa::RenderReadback> consumeReadback(
		const oa::RenderFrame& inFrame) override;
	[[nodiscard]] oa::Status markConsumed(
		const oa::RenderFrame& inFrame,
		const oa::Event& inConsumer) override;
	[[nodiscard]] oa::Status abandonFrame(
		const oa::RenderFrame& inFrame) override;
	[[nodiscard]] oa::Status collect() override;
	[[nodiscard]] oa::Status resize(oa::U32 inWidth, oa::U32 inHeight) override;
	[[nodiscard]] oa::Status close() override;
	[[nodiscard]] bool prepareNonWaitingRetirement() noexcept override;
	[[nodiscard]] oa::Status cleanupWithoutSubmission() override;
	[[nodiscard]] oa::Status completeRetired() override;
};

oa::Status oa::Renderer::MeshImpl::validateCapabilities() const {
	if (engine == nullptr or not engine->isReady()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer requires a ready engine");
	}
	if (not engine->hasGraphics()
		or oa::EngineDeviceAccess::get(*engine).queues.graphicsQueue == nullptr
		or oa::EngineDeviceAccess::get(*engine).queues.graphicsQueueFamily
			== oavk::EnumerationIndexUnset) {
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			"oa::Renderer requires a graphics-capable engine");
	}
	if (DeviceTable.vkCmdBeginRendering == nullptr
		or DeviceTable.vkCmdEndRendering == nullptr
		or DeviceTable.vkCmdPipelineBarrier2 == nullptr
		or DeviceTable.vkCmdCopyImageToBuffer == nullptr
		or DeviceTable.vkCreateGraphicsPipelines == nullptr) {
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			"oa::Renderer requires dynamic rendering, synchronization2, and image copy commands");
	}

	OA_RETURN_IF_ERROR(validateTargetExtent(
		*engine, InstanceTable, renderConfig.width_, renderConfig.height_,
		toVkSampleCount(renderConfig.sampleCount_)));
	VkPhysicalDeviceProperties properties{};
	InstanceTable.vkGetPhysicalDeviceProperties(
		static_cast<VkPhysicalDevice>(oa::EngineDeviceAccess::get(*engine).physicalDevice),
		&properties);
	if (properties.limits.maxPushConstantsSize < sizeof(RenderPushConstants)) {
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			"oa::Renderer requires 80 bytes of push constants");
	}

	return oa::Status::ok();
}

oa::Status oa::Renderer::MeshImpl::createPipeline() {
	const oavk::SpirvEntry* vertexSpirv =
		oavk::findSpirv("VertexColorLit.vert");
	const oavk::SpirvEntry* fragmentSpirv =
		oavk::findSpirv("VertexColorLit.frag");
	if (vertexSpirv == nullptr or fragmentSpirv == nullptr) {
		return oa::Status::error(
			oa::StatusCode::NotFound,
			"oa::Renderer: vertex-color shaders are unavailable");
	}
	const VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(*engine).device);
	VkShaderModule vertexModule =
		createShaderModule(DeviceTable, device, *vertexSpirv);
	VkShaderModule fragmentModule =
		createShaderModule(DeviceTable, device, *fragmentSpirv);
	if (vertexModule == VK_NULL_HANDLE or fragmentModule == VK_NULL_HANDLE) {
		if (vertexModule != VK_NULL_HANDLE) {
			DeviceTable.vkDestroyShaderModule(
				device, vertexModule, nullptr);
		}
		if (fragmentModule != VK_NULL_HANDLE) {
			DeviceTable.vkDestroyShaderModule(
				device, fragmentModule, nullptr);
		}
		return oa::Status::error(
			oa::StatusCode::PipelineError,
			"oa::Renderer: shader-module creation failed");
	}

	VkPushConstantRange pushRange{};
	pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	pushRange.offset = 0U;
	pushRange.size = sizeof(RenderPushConstants);
	VkPipelineLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layoutInfo.pushConstantRangeCount = 1U;
	layoutInfo.pPushConstantRanges = &pushRange;
	if (DeviceTable.vkCreatePipelineLayout(
			device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
		DeviceTable.vkDestroyShaderModule(device, vertexModule, nullptr);
		DeviceTable.vkDestroyShaderModule(device, fragmentModule, nullptr);
		return oa::Status::error(
			oa::StatusCode::PipelineError,
			"oa::Renderer: pipeline-layout creation failed");
	}

	VkPipelineShaderStageCreateInfo stages[2]{};
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vertexModule;
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = fragmentModule;
	stages[1].pName = "main";

	VkVertexInputBindingDescription binding{};
	binding.binding = 0U;
	binding.stride = sizeof(oa::MeshVertex);
	binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	VkVertexInputAttributeDescription attributes[3]{};
	attributes[0] = {
		0U, 0U, VK_FORMAT_R32G32B32_SFLOAT,
		static_cast<oa::U32>(offsetof(oa::MeshVertex, position))};
	attributes[1] = {
		1U, 0U, VK_FORMAT_R32G32B32_SFLOAT,
		static_cast<oa::U32>(offsetof(oa::MeshVertex, normal))};
	attributes[2] = {
		2U, 0U, VK_FORMAT_R32G32B32A32_SFLOAT,
		static_cast<oa::U32>(offsetof(oa::MeshVertex, color))};
	VkPipelineVertexInputStateCreateInfo vertexInput{};
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInput.vertexBindingDescriptionCount = 1U;
	vertexInput.pVertexBindingDescriptions = &binding;
	vertexInput.vertexAttributeDescriptionCount = 3U;
	vertexInput.pVertexAttributeDescriptions = attributes;

	VkPipelineInputAssemblyStateCreateInfo assembly{};
	assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	VkPipelineViewportStateCreateInfo viewport{};
	viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport.viewportCount = 1U;
	viewport.scissorCount = 1U;
	VkPipelineRasterizationStateCreateInfo raster{};
	raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	raster.polygonMode = VK_POLYGON_MODE_FILL;
	raster.cullMode = VK_CULL_MODE_NONE;
	raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	raster.lineWidth = 1.0F;
	VkPipelineMultisampleStateCreateInfo multisample{};
	multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample.rasterizationSamples =
		toVkSampleCount(renderConfig.sampleCount_);
	VkPipelineDepthStencilStateCreateInfo depthStencil{};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_TRUE;
	depthStencil.depthWriteEnable = VK_TRUE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencil.minDepthBounds = 0.0F;
	depthStencil.maxDepthBounds = 1.0F;
	VkPipelineColorBlendAttachmentState blendAttachment{};
	blendAttachment.colorWriteMask =
		VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
		| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	VkPipelineColorBlendStateCreateInfo blend{};
	blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	blend.attachmentCount = 1U;
	blend.pAttachments = &blendAttachment;
	const VkDynamicState dynamicStates[2] = {
		VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamic{};
	dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic.dynamicStateCount = 2U;
	dynamic.pDynamicStates = dynamicStates;

	VkPipelineRenderingCreateInfo rendering{};
	rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
	rendering.colorAttachmentCount = 1U;
	rendering.pColorAttachmentFormats = &ColorFormat;
	rendering.depthAttachmentFormat = DepthFormat;
	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.pNext = &rendering;
	pipelineInfo.stageCount = 2U;
	pipelineInfo.pStages = stages;
	pipelineInfo.pVertexInputState = &vertexInput;
	pipelineInfo.pInputAssemblyState = &assembly;
	pipelineInfo.pViewportState = &viewport;
	pipelineInfo.pRasterizationState = &raster;
	pipelineInfo.pMultisampleState = &multisample;
	pipelineInfo.pDepthStencilState = &depthStencil;
	pipelineInfo.pColorBlendState = &blend;
	pipelineInfo.pDynamicState = &dynamic;
	pipelineInfo.layout = pipelineLayout;
	const VkResult pipelineResult = DeviceTable.vkCreateGraphicsPipelines(
		device, VK_NULL_HANDLE, 1U, &pipelineInfo, nullptr, &pipeline);
	DeviceTable.vkDestroyShaderModule(device, vertexModule, nullptr);
	DeviceTable.vkDestroyShaderModule(device, fragmentModule, nullptr);
	if (pipelineResult != VK_SUCCESS) {
		return oa::Status::error(
			oa::StatusCode::PipelineError,
			"oa::Renderer: graphics-pipeline creation failed");
	}
	return oa::Status::ok();
}

oa::Status oa::Renderer::MeshImpl::createSlots() {
	const oa::U64 vertexBytes =
		static_cast<oa::U64>(renderConfig.maxVertexCount_)
			* sizeof(oa::MeshVertex);
	const oa::U64 indexBytes =
		static_cast<oa::U64>(renderConfig.maxIndexCount_) * sizeof(oa::U32);

	slots.reserve(renderConfig.targetSlotCount_);
	for (oa::U32 index = 0U; index < renderConfig.targetSlotCount_; ++index) {
		slots.emplaceBack();
		RenderSlot& slot = slots.back();
		auto vertices = createMappedVertexOrIndexBuffer(
			*engine, vertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		if (not vertices.isOk()) return vertices.getStatus();
		slot.vertexBuffer = oa::move(*vertices);
		auto indices = createMappedVertexOrIndexBuffer(
			*engine, indexBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		if (not indices.isOk()) return indices.getStatus();
		slot.indexBuffer = oa::move(*indices);
		const oa::Status targetStatus = createTarget(
			*engine, DeviceTable, renderConfig.width_, renderConfig.height_,
			toVkSampleCount(renderConfig.sampleCount_),
			slot.target);
		if (not targetStatus.isOk()) return targetStatus;
	}
	return oa::Status::ok();
}

oa::Status oa::Renderer::MeshImpl::initialize(
	oa::Engine& inEngine,
	const oa::RendererConfig& inRenderConfig) {
	engine = &inEngine;
	renderConfig = inRenderConfig;
	if (renderConfig.width_ == 0U or renderConfig.height_ == 0U) {
		return oa::Status::invalidArgument(
			"oa::Renderer: target dimensions must be non-zero");
	}
	if (renderConfig.targetSlotCount_ == 0U
		or renderConfig.targetSlotCount_ > MaxTargetSlots) {
		return oa::Status::invalidArgument(
			"oa::Renderer: target slot count must be in [1, 4]");
	}
	if (renderConfig.maxVertexCount_ == 0U
		or renderConfig.maxIndexCount_ == 0U) {
		return oa::Status::invalidArgument(
			"oa::Renderer: mesh capacities must be non-zero");
	}
	if (toVkSampleCount(renderConfig.sampleCount_) == 0U) {
		return oa::Status::invalidArgument(
			"oa::Renderer: sample count must be one of 1, 2, 4, 8, 16, 32, or 64");
	}
	if (not oa::isFinite(renderConfig.clearColor_.x)
		or not oa::isFinite(renderConfig.clearColor_.y)
		or not oa::isFinite(renderConfig.clearColor_.z)
		or not oa::isFinite(renderConfig.clearColor_.w)
		or not isFiniteVlm(renderConfig.lightDirection_)
		or not oa::isFinite(renderConfig.ambientLight_)
		or renderConfig.ambientLight_ < 0.0F
		or renderConfig.ambientLight_ > 1.0F) {
		return oa::Status::invalidArgument(
			"oa::Renderer: clear color and lighting must be finite");
	}
	if (not inEngine.isReady()
		or oa::EngineDeviceAccess::get(inEngine).instance == nullptr
		or oa::EngineDeviceAccess::get(inEngine).device == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer requires a ready engine");
	}
	const auto& device = oa::EngineDeviceAccess::get(inEngine);
	InstanceTable = device.instanceDispatch;
	DeviceTable = device.deviceDispatch;
	OA_RETURN_IF_ERROR(validateCapabilities());
	OA_RETURN_IF_ERROR(createPipeline());
	return createSlots();
}

oa::Status oa::Renderer::MeshImpl::writeFrameGeometry(
	RenderSlot& inSlot,
	const oa::MeshData& inMesh) {
	if (inMesh.vertices.empty()
		or inMesh.indices.empty()
		or (inMesh.indices.size() % 3U) != 0U) {
		return oa::Status::invalidArgument(
			"oa::Renderer: expected a non-empty indexed triangle-list mesh");
	}
	if (inMesh.vertices.size() > renderConfig.maxVertexCount_
		or inMesh.indices.size() > renderConfig.maxIndexCount_) {
		return oa::Status::error(
			oa::StatusCode::ResourceExhausted,
			"oa::Renderer: mesh exceeds the configured bounded slot capacity");
	}
	for (const oa::MeshVertex& vertex : inMesh.vertices) {
		if (not isFiniteVlm(vertex.position)
			or not isFiniteVlm(vertex.normal)
			or not oa::isFinite(vertex.color.x)
			or not oa::isFinite(vertex.color.y)
			or not oa::isFinite(vertex.color.z)
			or not oa::isFinite(vertex.color.w)) {
			return oa::Status::invalidArgument(
				"oa::Renderer: mesh contains non-finite vertex data");
		}
	}
	for (oa::U32 index : inMesh.indices) {
		if (index >= inMesh.vertices.size()) {
			return oa::Status::invalidArgument(
				"oa::Renderer: mesh index is outside the vertex snapshot");
		}
	}
	inSlot.indexCount = static_cast<oa::U32>(inMesh.indices.size());
	const oa::U64 vertexBytes =
		static_cast<oa::U64>(inMesh.vertices.size()) * sizeof(oa::MeshVertex);
	const oa::U64 indexBytes =
		static_cast<oa::U64>(inMesh.indices.size()) * sizeof(oa::U32);
	oa::memcpy(
		inSlot.vertexBuffer.mappedPtr,
		inMesh.vertices.data(),
		static_cast<oa::Usize>(vertexBytes));
	oa::memcpy(
		inSlot.indexBuffer.mappedPtr,
		inMesh.indices.data(),
		static_cast<oa::Usize>(indexBytes));
	if (not oa::EngineAllocatorAccess::get(*engine).flushHostBuffer(
			inSlot.vertexBuffer, 0U, vertexBytes)
		or not oa::EngineAllocatorAccess::get(*engine).flushHostBuffer(
			inSlot.indexBuffer, 0U, indexBytes)) {
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			"oa::Renderer: mapped geometry flush failed");
	}
	return oa::Status::ok();
}

oa::Status oa::Renderer::MeshImpl::recordFrame(
	RenderSlot& inSlot,
	const oa::CameraState& inCamera) {
	if (not isFiniteVlm(oa::FnCamera::getViewProjectionMatrix(inCamera))) {
		return oa::Status::invalidArgument(
			"oa::Renderer camera matrix must be finite");
	}
	if (not inSlot.streamLease.hasValue()) {
		return oa::Status::error(
			oa::StatusCode::Internal,
			"oa::Renderer has no active graphics lease");
	}
	oavk::Stream* stream = inSlot.streamLease->getStream();
	if (stream == nullptr or stream->commandBuffer == nullptr) {
		return oa::Status::error(
			oa::StatusCode::Internal,
			"oa::Renderer graphics encoder is unavailable");
	}
	const VkCommandBuffer commandBuffer =
		static_cast<VkCommandBuffer>(stream->commandBuffer);

	VkMemoryBarrier2 hostToGeometry{};
	hostToGeometry.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
	hostToGeometry.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
	hostToGeometry.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
	hostToGeometry.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
	hostToGeometry.dstAccessMask =
		VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT;

	const bool multisampled = renderConfig.sampleCount_ != 1U;
	VkImageMemoryBarrier2 toAttachments[4]{};
	toAttachments[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	toAttachments[0].srcStageMask =
		inSlot.target.colorLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
			? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_2_NONE;
	toAttachments[0].srcAccessMask =
		inSlot.target.colorLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
			? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE;
	toAttachments[0].dstStageMask =
		VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
	toAttachments[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
	toAttachments[0].oldLayout = inSlot.target.colorLayout;
	toAttachments[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	toAttachments[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toAttachments[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toAttachments[0].image = inSlot.target.colorImage;
	toAttachments[0].subresourceRange = {
		VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
	toAttachments[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	toAttachments[1].srcStageMask =
		inSlot.target.depthLayout
			== VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
			? VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
				| VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
				| VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
			: VK_PIPELINE_STAGE_2_NONE;
	toAttachments[1].srcAccessMask =
		inSlot.target.depthLayout
			== VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
			? VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
				| VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
				| VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
			: VK_ACCESS_2_NONE;
	toAttachments[1].dstStageMask =
		VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
		| VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
		| VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
	toAttachments[1].dstAccessMask =
		VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
		| VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
		| VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	toAttachments[1].oldLayout = inSlot.target.depthLayout;
	toAttachments[1].newLayout =
		VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	toAttachments[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toAttachments[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toAttachments[1].image = inSlot.target.depthImage;
	toAttachments[1].subresourceRange = {
		VK_IMAGE_ASPECT_DEPTH_BIT, 0U, 1U, 0U, 1U};
	oa::U32 attachmentBarrierCount = 2U;
	if (multisampled) {
		toAttachments[2].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		toAttachments[2].srcStageMask =
			inSlot.target.colorMsaaLayout
				== VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
			? VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
			: VK_PIPELINE_STAGE_2_NONE;
		toAttachments[2].srcAccessMask =
			inSlot.target.colorMsaaLayout
				== VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
			? VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
			: VK_ACCESS_2_NONE;
		toAttachments[2].dstStageMask =
			VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
		toAttachments[2].dstAccessMask =
			VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
		toAttachments[2].oldLayout = inSlot.target.colorMsaaLayout;
		toAttachments[2].newLayout =
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		toAttachments[2].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		toAttachments[2].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		toAttachments[2].image = inSlot.target.colorMsaaImage;
		toAttachments[2].subresourceRange = {
			VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};

		toAttachments[3].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		toAttachments[3].srcStageMask =
			inSlot.target.depthMsaaLayout
				== VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
			? VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
				| VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
			: VK_PIPELINE_STAGE_2_NONE;
		toAttachments[3].srcAccessMask =
			inSlot.target.depthMsaaLayout
				== VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
			? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
			: VK_ACCESS_2_NONE;
		toAttachments[3].dstStageMask =
			VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
			| VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
		toAttachments[3].dstAccessMask =
			VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
			| VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		toAttachments[3].oldLayout = inSlot.target.depthMsaaLayout;
		toAttachments[3].newLayout =
			VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		toAttachments[3].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		toAttachments[3].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		toAttachments[3].image = inSlot.target.depthMsaaImage;
		toAttachments[3].subresourceRange = {
			VK_IMAGE_ASPECT_DEPTH_BIT, 0U, 1U, 0U, 1U};
		attachmentBarrierCount = 4U;
	}
	VkDependencyInfo attachmentDependency{};
	attachmentDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	attachmentDependency.memoryBarrierCount = 1U;
	attachmentDependency.pMemoryBarriers = &hostToGeometry;
	attachmentDependency.imageMemoryBarrierCount = attachmentBarrierCount;
	attachmentDependency.pImageMemoryBarriers = toAttachments;
	DeviceTable.vkCmdPipelineBarrier2(
		commandBuffer, &attachmentDependency);

	VkClearValue colorClear{};
	colorClear.color.float32[0] = renderConfig.clearColor_.x;
	colorClear.color.float32[1] = renderConfig.clearColor_.y;
	colorClear.color.float32[2] = renderConfig.clearColor_.z;
	colorClear.color.float32[3] = renderConfig.clearColor_.w;
	VkClearValue depthClear{};
	depthClear.depthStencil = {1.0F, 0U};
	VkRenderingAttachmentInfo colorAttachment{};
	colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
	colorAttachment.imageView = multisampled
		? inSlot.target.colorMsaaView : inSlot.target.colorView;
	colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	colorAttachment.resolveMode = multisampled
		? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE;
	colorAttachment.resolveImageView = multisampled
		? inSlot.target.colorView : VK_NULL_HANDLE;
	colorAttachment.resolveImageLayout = multisampled
		? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		: VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.storeOp = multisampled
		? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.clearValue = colorClear;
	VkRenderingAttachmentInfo depthAttachment{};
	depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
	depthAttachment.imageView = multisampled
		? inSlot.target.depthMsaaView : inSlot.target.depthView;
	depthAttachment.imageLayout =
		VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	depthAttachment.resolveMode = multisampled
		? VK_RESOLVE_MODE_MIN_BIT : VK_RESOLVE_MODE_NONE;
	depthAttachment.resolveImageView = multisampled
		? inSlot.target.depthView : VK_NULL_HANDLE;
	depthAttachment.resolveImageLayout = multisampled
		? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
		: VK_IMAGE_LAYOUT_UNDEFINED;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.storeOp = multisampled
		? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
	depthAttachment.clearValue = depthClear;
	VkRenderingInfo rendering{};
	rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
	rendering.renderArea = {{0, 0}, {renderConfig.width_, renderConfig.height_}};
	rendering.layerCount = 1U;
	rendering.colorAttachmentCount = 1U;
	rendering.pColorAttachments = &colorAttachment;
	rendering.pDepthAttachment = &depthAttachment;
	DeviceTable.vkCmdBeginRendering(commandBuffer, &rendering);

	const VkViewport viewport{
		0.0F,
		static_cast<oa::F32>(renderConfig.height_),
		static_cast<oa::F32>(renderConfig.width_),
		-static_cast<oa::F32>(renderConfig.height_),
		0.0F,
		1.0F,
	};
	const VkRect2D scissor{{0, 0}, {renderConfig.width_, renderConfig.height_}};
	DeviceTable.vkCmdSetViewport(commandBuffer, 0U, 1U, &viewport);
	DeviceTable.vkCmdSetScissor(commandBuffer, 0U, 1U, &scissor);
	DeviceTable.vkCmdBindPipeline(
		commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	const VkBuffer vertexBuffer =
		static_cast<VkBuffer>(inSlot.vertexBuffer.buffer);
	const VkDeviceSize vertexOffset = 0U;
	DeviceTable.vkCmdBindVertexBuffers(
		commandBuffer, 0U, 1U, &vertexBuffer, &vertexOffset);
	DeviceTable.vkCmdBindIndexBuffer(
		commandBuffer,
		static_cast<VkBuffer>(inSlot.indexBuffer.buffer),
		0U, VK_INDEX_TYPE_UINT32);
	const RenderPushConstants push{
		oa::FnCamera::getViewProjectionMatrix(inCamera),
		{
			renderConfig.lightDirection_.x,
			renderConfig.lightDirection_.y,
			renderConfig.lightDirection_.z,
			renderConfig.ambientLight_,
		},
	};
	DeviceTable.vkCmdPushConstants(
		commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
		0U, sizeof(push), &push);
	DeviceTable.vkCmdDrawIndexed(
		commandBuffer, inSlot.indexCount, 1U, 0U, 0, 0U);
	DeviceTable.vkCmdEndRendering(commandBuffer);

	VkImageMemoryBarrier2 colorForSampling{};
	colorForSampling.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	colorForSampling.srcStageMask =
		VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
	colorForSampling.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
	colorForSampling.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	colorForSampling.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
	colorForSampling.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	colorForSampling.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	colorForSampling.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	colorForSampling.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	colorForSampling.image = inSlot.target.colorImage;
	colorForSampling.subresourceRange = {
		VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
	VkDependencyInfo hostDependency{};
	hostDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	hostDependency.imageMemoryBarrierCount = 1U;
	hostDependency.pImageMemoryBarriers = &colorForSampling;
	DeviceTable.vkCmdPipelineBarrier2(commandBuffer, &hostDependency);
	return oa::Status::ok();
}

void oa::Renderer::MeshImpl::recordReadback(
	VkCommandBuffer inCommandBuffer,
	RenderTarget& inTarget) const {
	VkImageMemoryBarrier2 toCopies[2]{};
	toCopies[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	toCopies[0].srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	toCopies[0].srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
	toCopies[0].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
	toCopies[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	toCopies[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	toCopies[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	toCopies[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toCopies[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toCopies[0].image = inTarget.colorImage;
	toCopies[0].subresourceRange = {
		VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
	toCopies[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	toCopies[1].srcStageMask =
		VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
		| VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
		| VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
	toCopies[1].srcAccessMask =
		VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
		| VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
		| VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	toCopies[1].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
	toCopies[1].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	toCopies[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	toCopies[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	toCopies[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toCopies[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toCopies[1].image = inTarget.depthImage;
	toCopies[1].subresourceRange = {
		VK_IMAGE_ASPECT_DEPTH_BIT, 0U, 1U, 0U, 1U};

	VkBufferMemoryBarrier2 hostRelease[2]{};
	hostRelease[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
	hostRelease[0].srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
	hostRelease[0].srcAccessMask = VK_ACCESS_2_HOST_READ_BIT;
	hostRelease[0].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
	hostRelease[0].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	hostRelease[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	hostRelease[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	hostRelease[0].buffer = static_cast<VkBuffer>(inTarget.colorReadback.buffer);
	hostRelease[0].offset = 0U;
	hostRelease[0].size = VK_WHOLE_SIZE;
	hostRelease[1] = hostRelease[0];
	hostRelease[1].buffer = static_cast<VkBuffer>(inTarget.depthReadback.buffer);

	VkDependencyInfo copyDependency{};
	copyDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	copyDependency.bufferMemoryBarrierCount = 2U;
	copyDependency.pBufferMemoryBarriers = hostRelease;
	copyDependency.imageMemoryBarrierCount = 2U;
	copyDependency.pImageMemoryBarriers = toCopies;
	DeviceTable.vkCmdPipelineBarrier2(inCommandBuffer, &copyDependency);

	VkBufferImageCopy colorCopy{};
	colorCopy.imageSubresource = {
		VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, 1U};
	colorCopy.imageExtent = {renderConfig.width_, renderConfig.height_, 1U};
	DeviceTable.vkCmdCopyImageToBuffer(
		inCommandBuffer, inTarget.colorImage,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		static_cast<VkBuffer>(inTarget.colorReadback.buffer),
		1U, &colorCopy);
	VkBufferImageCopy depthCopy{};
	depthCopy.imageSubresource = {
		VK_IMAGE_ASPECT_DEPTH_BIT, 0U, 0U, 1U};
	depthCopy.imageExtent = {renderConfig.width_, renderConfig.height_, 1U};
	DeviceTable.vkCmdCopyImageToBuffer(
		inCommandBuffer, inTarget.depthImage,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		static_cast<VkBuffer>(inTarget.depthReadback.buffer),
		1U, &depthCopy);

	VkBufferMemoryBarrier2 toHost[2]{};
	toHost[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
	toHost[0].srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
	toHost[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	toHost[0].dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
	toHost[0].dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
	toHost[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toHost[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toHost[0].buffer = static_cast<VkBuffer>(inTarget.colorReadback.buffer);
	toHost[0].offset = 0U;
	toHost[0].size = VK_WHOLE_SIZE;
	toHost[1] = toHost[0];
	toHost[1].buffer = static_cast<VkBuffer>(inTarget.depthReadback.buffer);

	VkImageMemoryBarrier2 restore[2]{};
	restore[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	restore[0].srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
	restore[0].srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	restore[0].dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	restore[0].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
	restore[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	restore[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	restore[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	restore[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	restore[0].image = inTarget.colorImage;
	restore[0].subresourceRange = {
		VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
	restore[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	restore[1].srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
	restore[1].srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	restore[1].dstStageMask =
		VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
		| VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
		| VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
	restore[1].dstAccessMask =
		VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
		| VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
		| VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	restore[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	restore[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	restore[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	restore[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	restore[1].image = inTarget.depthImage;
	restore[1].subresourceRange = {
		VK_IMAGE_ASPECT_DEPTH_BIT, 0U, 1U, 0U, 1U};

	VkDependencyInfo hostDependency{};
	hostDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	hostDependency.bufferMemoryBarrierCount = 2U;
	hostDependency.pBufferMemoryBarriers = toHost;
	hostDependency.imageMemoryBarrierCount = 2U;
	hostDependency.pImageMemoryBarriers = restore;
	DeviceTable.vkCmdPipelineBarrier2(inCommandBuffer, &hostDependency);
}

oa::Status oa::Renderer::MeshImpl::validateFrame(
	const oa::RenderFrame& inFrame,
	RenderSlotState inRequiredState,
	RenderSlot*& outSlot) {
	if (inFrame.sourceKind_ != oa::RenderFrame::SourceKind::Mesh) {
		return oa::Status::invalidArgument(
			"oa::Renderer frame was produced by a different render mode");
	}
	outSlot = nullptr;
	if (closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	if (inFrame.targetGeneration_ != TargetGeneration
		or inFrame.slot_ >= slots.size()
		or inFrame.width_ != renderConfig.width_
		or inFrame.height_ != renderConfig.height_) {
		return oa::Status::invalidArgument(
			"oa::Renderer frame has stale or forged target metadata");
	}
	RenderSlot& slot = slots[inFrame.slot_];
	if (slot.generation != inFrame.slotGeneration_
		or slot.state != inRequiredState
		or oa::TextureAccess::engine(inFrame.color_) != engine
		or oa::TextureAccess::image(inFrame.color_)
			!= slot.target.colorImage
		or oa::TextureAccess::view(inFrame.color_)
			!= slot.target.colorView
		or oa::TextureAccess::layout(inFrame.color_)
			!= VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		or not engine->ownsEvent(inFrame.producer_)
		or not slot.producer.isSameCompletion(inFrame.producer_)) {
		return oa::Status::invalidArgument(
			"oa::Renderer frame is stale or not the exact live completion");
	}
	outSlot = &slot;
	return oa::Status::ok();
}

oa::Status oa::Renderer::MeshImpl::collectRetired() {
	if (closed) return oa::Status::ok();
	for (RenderSlot& slot : slots) {
		if (slot.state != RenderSlotState::Retired) continue;
		if (not slot.producer.isValid()) {
			return oa::Status::error(
				oa::StatusCode::Internal,
				"oa::Renderer retired target lost its producer event");
		}
		if (not slot.producer.isComplete()
			or (slot.consumer.isValid() and not slot.consumer.isComplete())) {
			continue;
		}
		// The counter query above is only a non-blocking observation. Complete
		// the exact semaphore wait dependency before making mapped geometry and
		// readback buffers host-reusable. Because the observed value is already
		// reached, this wait does not stall for outstanding device work.
		OA_RETURN_IF_ERROR(slot.producer.wait());
		if (slot.consumer.isValid()) {
			OA_RETURN_IF_ERROR(slot.consumer.wait());
		}
		slot.producer = {};
		slot.consumer = {};
		slot.state = RenderSlotState::Free;
	}
	return oa::Status::ok();
}

void oa::Renderer::MeshImpl::destroyAll() noexcept {
	if (engine == nullptr or oa::EngineDeviceAccess::get(*engine).device == nullptr) return;
	for (RenderSlot& slot : slots) {
		destroyTarget(*engine, DeviceTable, slot.target);
		oa::EngineAllocatorAccess::get(*engine).free(slot.vertexBuffer);
		oa::EngineAllocatorAccess::get(*engine).free(slot.indexBuffer);
		slot.streamLease.reset();
		slot.producer = {};
		slot.consumer = {};
		slot.state = RenderSlotState::Free;
	}
	slots.clear();
	const VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(*engine).device);
	if (pipeline != VK_NULL_HANDLE) {
		DeviceTable.vkDestroyPipeline(device, pipeline, nullptr);
		pipeline = VK_NULL_HANDLE;
	}
	if (pipelineLayout != VK_NULL_HANDLE) {
		DeviceTable.vkDestroyPipelineLayout(
			device, pipelineLayout, nullptr);
		pipelineLayout = VK_NULL_HANDLE;
	}
}

bool oa::Renderer::MeshImpl::prepareNonWaitingRetirement() noexcept {
	if (closed) return false;
	bool hasSubmission = false;
	if (activeSlot < slots.size()) {
		RenderSlot& slot = slots[activeSlot];
		if (slot.streamLease.hasValue()) {
			const oa::Status status = slot.streamLease->cancel();
			if (not status.isOk()) {
				(void)slot.streamLease->close();
			}
			slot.streamLease.reset();
		}
		slot.state = RenderSlotState::Free;
		slot.producer = {};
		slot.consumer = {};
	}
	activeSlot = oa::Limits<oa::U32>::max();
	for (RenderSlot& slot : slots) {
		if (slot.state == RenderSlotState::Submitted) {
			if (slot.streamLease.hasValue()) {
				(void)slot.streamLease->close();
				slot.streamLease.reset();
			}
			slot.state = RenderSlotState::Retired;
			hasSubmission = true;
		} else if (slot.state == RenderSlotState::Retired) {
			hasSubmission = true;
		}
	}
	return hasSubmission;
}

oa::Status oa::Renderer::MeshImpl::cleanupWithoutSubmission() {
	destroyAll();
	closed = true;
	return oa::Status::ok();
}

oa::Status oa::Renderer::MeshImpl::completeRetired() {
	oa::Status firstError = oa::Status::ok();
	for (RenderSlot& slot : slots) {
		if (slot.state != RenderSlotState::Retired) continue;
		const oa::Status producerStatus = slot.producer.wait();
		oa::Status consumerStatus = oa::Status::ok();
		if (producerStatus.isOk() and slot.consumer.isValid()) {
			consumerStatus = slot.consumer.wait();
		}
		if (not producerStatus.isOk() and firstError.isOk()) {
			firstError = producerStatus;
		} else if (not consumerStatus.isOk() and firstError.isOk()) {
			firstError = consumerStatus;
		}
		if (producerStatus.isOk() and consumerStatus.isOk()) {
			slot.producer = {};
			slot.consumer = {};
			slot.state = RenderSlotState::Free;
		}
	}
	if (firstError.isOk()) {
		destroyAll();
		closed = true;
	}
	return firstError;
}

oa::Result<oa::UniquePtr<oa::Renderer>>
oa::Renderer::create(
	oa::Engine& inEngine,
	const oa::RendererConfig& inRenderConfig) {
	switch (inRenderConfig.mode_) {
		case oa::RendererMode::Rasterization:
			break;
		case oa::RendererMode::RayTracing:
			return oa::Status::error(
				oa::StatusCode::Unavailable,
				"oa::Renderer ray-tracing mode is not implemented");
		default:
			return oa::Status::invalidArgument(
				"oa::Renderer mode is invalid");
	}
	oa::UniquePtr<oa::Renderer> session(new oa::Renderer());
	auto* impl = new MeshImpl();
	const oa::Status status = impl->initialize(inEngine, inRenderConfig);
	if (not status.isOk()) {
		(void)impl->cleanupWithoutSubmission();
		delete impl;
		return status;
	}
	session->impl_ = oa::UniquePtr<Impl>(impl);
	return oa::move(session);
}

oa::Status oa::Renderer::MeshImpl::beginMeshFrame(
	const oa::MeshData& inMesh,
	const oa::CameraState& inCamera) {
	if (closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	if (activeSlot < slots.size()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer already has an active recording");
	}
	OA_RETURN_IF_ERROR(collectRetired());
	oa::U32 slotIndex = static_cast<oa::U32>(slots.size());
	for (oa::U32 index = 0U; index < slots.size(); ++index) {
		if (slots[index].state == RenderSlotState::Free) {
			slotIndex = index;
			break;
		}
	}
	if (slotIndex == slots.size()) {
		return oa::Status::error(
			oa::StatusCode::ResourceExhausted,
			"oa::Renderer target ring is full; consume or abandon an older frame");
	}

	RenderSlot& slot = slots[slotIndex];
	slot.generation = nextGeneration(slot.generation);
	slot.state = RenderSlotState::Recording;
	slot.producer = {};
	slot.consumer = {};
	auto lease = oa::GraphicsStreamLease::acquire(*engine);
	if (not lease.isOk()) {
		slot.state = RenderSlotState::Free;
		return lease.getStatus();
	}
	slot.streamLease.emplace(oa::move(*lease));
	oa::Status status = writeFrameGeometry(slot, inMesh);
	if (status.isOk()) status = recordFrame(slot, inCamera);
	if (not status.isOk()) {
		const oa::Status cancelStatus = slot.streamLease->cancel();
		if (not cancelStatus.isOk()) (void)slot.streamLease->close();
		slot.streamLease.reset();
		slot.state = RenderSlotState::Free;
		return status;
	}
	activeSlot = slotIndex;
	return oa::Status::ok();
}

oa::Result<oa::RenderFrame>
oa::Renderer::MeshImpl::submitFrame(
	oa::Span<const oa::Event> inDependencies) {
	if (closed
		or activeSlot >= slots.size()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer has no active recording to submit");
	}
	RenderSlot& slot = slots[activeSlot];
	if (slot.state != RenderSlotState::Recording
		or not slot.streamLease.hasValue()) {
		return oa::Status::error(
			oa::StatusCode::Internal,
			"oa::Renderer active target lost its graphics recording");
	}
	auto submission = slot.streamLease->submit(inDependencies);
	if (not submission.isOk()) {
		// Dependency provenance failures occur before queue submission, so the
		// exact recording stays active and cancelFrame() remains valid. A queue
		// submission failure is reset by the engine and is closed here.
		const oa::StatusCode code = submission.getStatus().getCode();
		if (code != oa::StatusCode::InvalidArgument
			and code != oa::StatusCode::FailedPrecondition) {
			(void)slot.streamLease->close();
			slot.streamLease.reset();
			slot.state = RenderSlotState::Free;
			activeSlot = oa::Limits<oa::U32>::max();
		}
		return submission.getStatus();
	}
	slot.producer = *submission;
	slot.consumer = {};
	slot.target.colorLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	slot.target.depthLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	if (renderConfig.sampleCount_ != 1U) {
		slot.target.colorMsaaLayout =
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		slot.target.depthMsaaLayout =
			VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	}
	slot.state = RenderSlotState::Submitted;
	const oa::U32 submittedSlot = activeSlot;
	activeSlot = oa::Limits<oa::U32>::max();
	oa::RenderFrame frame;
	frame.slot_ = submittedSlot;
	frame.slotGeneration_ = slot.generation;
	frame.targetGeneration_ = TargetGeneration;
	frame.width_ = renderConfig.width_;
	frame.height_ = renderConfig.height_;
	frame.color_ = oa::TextureAccess::fromBorrowedImage(
		*engine,
		slot.target.colorImage,
		slot.target.colorView,
		ColorFormat,
		slot.target.colorLayout,
		static_cast<oa::I32>(frame.width_),
		static_cast<oa::I32>(frame.height_));
	frame.producer_ = slot.producer;
	frame.sourceKind_ = oa::RenderFrame::SourceKind::Mesh;
	return frame;
}

oa::Status oa::Renderer::MeshImpl::cancelFrame() {
	if (closed
		or activeSlot >= slots.size()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer has no active recording to cancel");
	}
	RenderSlot& slot = slots[activeSlot];
	oa::Status status = slot.streamLease.hasValue()
		? slot.streamLease->cancel()
		: oa::Status::error(
			oa::StatusCode::Internal,
			"oa::Renderer active target has no graphics lease");
	if (not status.isOk() and slot.streamLease.hasValue()) {
		(void)slot.streamLease->close();
	}
	slot.streamLease.reset();
	slot.state = RenderSlotState::Free;
	slot.producer = {};
	slot.consumer = {};
	activeSlot = oa::Limits<oa::U32>::max();
	return status;
}

oa::Result<oa::RenderReadback>
oa::Renderer::MeshImpl::consumeReadback(
	const oa::RenderFrame& inFrame) {
	if (closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	RenderSlot* slot = nullptr;
	const oa::Status validation = validateFrame(
		inFrame, RenderSlotState::Submitted, slot);
	if (not validation.isOk()) return validation;
	if (not slot->streamLease.hasValue()) {
		return oa::Status::error(
			oa::StatusCode::Internal,
			"oa::Renderer submitted target lost its graphics lease");
	}

	auto readbackLeaseResult = oa::GraphicsStreamLease::acquire(*engine);
	if (not readbackLeaseResult.isOk()) {
		return readbackLeaseResult.getStatus();
	}
	oa::GraphicsStreamLease readbackLease = oa::move(*readbackLeaseResult);
	oavk::Stream* readbackStream = readbackLease.getStream();
	if (readbackStream == nullptr or readbackStream->commandBuffer == nullptr) {
		return oa::Status::error(
			oa::StatusCode::Internal,
			"oa::Renderer readback graphics encoder is unavailable");
	}
	recordReadback(
		static_cast<VkCommandBuffer>(readbackStream->commandBuffer),
		slot->target);
	auto readbackSubmission = readbackLease.submit(
		oa::Span<const oa::Event>(&inFrame.producer_, 1U));
	if (not readbackSubmission.isOk()) {
		return readbackSubmission.getStatus();
	}
	const oa::Event readbackCompletion = *readbackSubmission;
	const oa::Status readbackWait = readbackCompletion.wait();
	if (not readbackWait.isOk()) {
		// The submitted copy consumes the target even if waiting reports a
		// device failure. Retire both leases against their exact events so no
		// caller can reuse the images underneath the in-flight command buffer.
		(void)readbackLease.close();
		(void)slot->streamLease->close();
		slot->streamLease.reset();
		slot->consumer = readbackCompletion;
		slot->state = RenderSlotState::Retired;
		return readbackWait;
	}
	const oa::Status readbackRecycle =
		readbackLease.recycle(readbackCompletion);
	if (not readbackRecycle.isOk()) return readbackRecycle;
	oa::U64 pixelCount = 0U;
	oa::U64 colorBytes = 0U;
	oa::U64 depthBytes = 0U;
	if (not checkedMultiply(
			renderConfig.width_, renderConfig.height_, pixelCount)
		or not checkedMultiply(pixelCount, 4U, colorBytes)
		or not checkedMultiply(pixelCount, sizeof(oa::F32), depthBytes)) {
		return oa::Status::error(
			oa::StatusCode::Internal,
			"oa::Renderer validated target extent overflowed readback size");
	}
	if (not oa::EngineAllocatorAccess::get(*engine).invalidateHostBuffer(
			slot->target.colorReadback, 0U, colorBytes)
		or not oa::EngineAllocatorAccess::get(*engine).invalidateHostBuffer(
			slot->target.depthReadback, 0U, depthBytes)) {
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			"oa::Renderer target readback invalidate failed");
	}

	oa::RenderReadback readback;
	readback.width_ = renderConfig.width_;
	readback.height_ = renderConfig.height_;
	readback.colorRgba8_.resize(static_cast<oa::Usize>(colorBytes));
	readback.depth32_.resize(static_cast<oa::Usize>(pixelCount));
	oa::memcpy(
		readback.colorRgba8_.data(),
		slot->target.colorReadback.mappedPtr,
		static_cast<oa::Usize>(colorBytes));
	oa::memcpy(
		readback.depth32_.data(),
		slot->target.depthReadback.mappedPtr,
		static_cast<oa::Usize>(depthBytes));

	const oa::Status recycleStatus =
		slot->streamLease->recycle(inFrame.producer_);
	if (not recycleStatus.isOk()) return recycleStatus;
	slot->streamLease.reset();
	slot->producer = {};
	slot->consumer = {};
	slot->state = RenderSlotState::Free;
	return readback;
}

oa::Status oa::Renderer::MeshImpl::markConsumed(
	const oa::RenderFrame& inFrame,
	const oa::Event& inConsumer) {
	if (closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	RenderSlot* slot = nullptr;
	OA_RETURN_IF_ERROR(validateFrame(
		inFrame, RenderSlotState::Submitted, slot));
	if (not engine->ownsEvent(inConsumer)
		or not inConsumer.hasQueueFamily()) {
		return oa::Status::invalidArgument(
			"oa::Renderer consumer must be an exact event from this engine");
	}
	if (inConsumer.queueFamily()
		!= oa::EngineDeviceAccess::get(*engine).queues.graphicsQueueFamily) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer cross-family consumption requires an explicit image ownership transfer");
	}
	if (inConsumer.isSameCompletion(slot->producer)) {
		return oa::Status::invalidArgument(
			"oa::Renderer consumer completion cannot alias its producer completion");
	}
	const void* producerSemaphore =
		oa::EventAccess::semaphoreHandle(slot->producer);
	const void* consumerSemaphore =
		oa::EventAccess::semaphoreHandle(inConsumer);
	if (producerSemaphore != nullptr and consumerSemaphore != nullptr
		and producerSemaphore == consumerSemaphore
		and inConsumer.value() <= slot->producer.value()) {
		return oa::Status::invalidArgument(
			"oa::Renderer consumer completion must follow its producer on a shared timeline");
	}

	oa::Status closeStatus = oa::Status::ok();
	if (slot->streamLease.hasValue()) {
		closeStatus = slot->streamLease->close();
		slot->streamLease.reset();
	}
	// register before returning even if graphics-stream retirement reports a
	// failure. A failed registration path must never make a sampled target
	// reusable while the exact consumer event is still outstanding.
	slot->consumer = inConsumer;
	slot->state = RenderSlotState::Retired;
	return closeStatus;
}

oa::Status oa::Renderer::MeshImpl::abandonFrame(
	const oa::RenderFrame& inFrame) {
	if (closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	RenderSlot* slot = nullptr;
	OA_RETURN_IF_ERROR(validateFrame(
		inFrame, RenderSlotState::Submitted, slot));
	oa::Status closeStatus = oa::Status::ok();
	if (slot->streamLease.hasValue()) {
		closeStatus = slot->streamLease->close();
		slot->streamLease.reset();
	}
	// abandon is strictly non-waiting even if the producer already appears
	// complete. collect() performs the exact, already-satisfied semaphore wait
	// before this target's mapped buffers become reusable by the host.
	slot->consumer = {};
	slot->state = RenderSlotState::Retired;
	return closeStatus;
}

oa::Status oa::Renderer::MeshImpl::collect() {
	if (closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	return collectRetired();
}

oa::Status oa::Renderer::MeshImpl::resize(
	oa::U32 inWidth, oa::U32 inHeight) {
	if (closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	if (inWidth == 0U or inHeight == 0U) {
		return oa::Status::invalidArgument(
			"oa::Renderer target dimensions must be non-zero");
	}
	OA_RETURN_IF_ERROR(collectRetired());
	if (activeSlot < slots.size()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer resize requires cancelling the active recording");
	}
	for (const RenderSlot& slot : slots) {
		if (slot.state != RenderSlotState::Free) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::Renderer resize requires every old-generation frame to be consumed or retired");
		}
	}
	OA_RETURN_IF_ERROR(validateTargetExtent(
		*engine, InstanceTable, inWidth, inHeight,
		toVkSampleCount(renderConfig.sampleCount_)));
	if (inWidth == renderConfig.width_
		and inHeight == renderConfig.height_) {
		return oa::Status::ok();
	}

	// allocate a full replacement generation first. A Busy or allocation-failure
	// path does not mutate old resources, layouts, dimensions, or generations.
	oa::Vec<RenderTarget> replacements(slots.size());
	for (oa::Usize index = 0U; index < replacements.size(); ++index) {
		const oa::Status status = createTarget(
			*engine, DeviceTable,
			inWidth, inHeight,
			toVkSampleCount(renderConfig.sampleCount_), replacements[index]);
		if (not status.isOk()) {
			for (RenderTarget& target : replacements) {
				destroyTarget(
					*engine, DeviceTable, target);
			}
			return status;
		}
	}
	for (oa::Usize index = 0U; index < slots.size(); ++index) {
		destroyTarget(
			*engine, DeviceTable,
			slots[index].target);
		slots[index].target = replacements[index];
		replacements[index] = {};
		slots[index].generation =
			nextGeneration(slots[index].generation);
	}
	renderConfig.width_ = inWidth;
	renderConfig.height_ = inHeight;
	TargetGeneration = nextGeneration(TargetGeneration);
	return oa::Status::ok();
}

oa::Status oa::Renderer::MeshImpl::close() {
	if (closed) return oa::Status::ok();
	if (activeSlot < slots.size()) {
		const oa::Status cancelStatus = cancelFrame();
		if (not cancelStatus.isOk()) return cancelStatus;
	}
	for (RenderSlot& slot : slots) {
		if (slot.state == RenderSlotState::Submitted) {
			const oa::Status waitStatus = slot.producer.wait();
			if (not waitStatus.isOk()) return waitStatus;
			if (not slot.streamLease.hasValue()) {
				return oa::Status::error(
					oa::StatusCode::Internal,
					"oa::Renderer submitted target lost its graphics lease during Close");
			}
			const oa::Status releaseStatus =
				slot.streamLease->recycle(slot.producer);
			if (not releaseStatus.isOk()) return releaseStatus;
			slot.streamLease.reset();
		} else if (slot.state == RenderSlotState::Retired) {
			const oa::Status waitStatus = slot.producer.wait();
			if (not waitStatus.isOk()) return waitStatus;
			if (slot.consumer.isValid()) {
				const oa::Status consumerStatus = slot.consumer.wait();
				if (not consumerStatus.isOk()) return consumerStatus;
			}
		}
		slot.producer = {};
		slot.consumer = {};
		slot.state = RenderSlotState::Free;
	}
	destroyAll();
	closed = true;
	return oa::Status::ok();
}
