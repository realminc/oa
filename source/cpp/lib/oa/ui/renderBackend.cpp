#include <oa/runtime/engine.h>
#include <oa/runtime/eventAccess.h>
#include <vkl/vkl.h>
#include "../render/rendererInternal.h"

#include <oa/core/log.h>
#include <oa/core/std/memory.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/optional.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/bindlessAccess.h>
#include <oa/runtime/engine/deviceAccess.h>
#include <oa/runtime/engine/submissionAccess.h>
#include <oa/runtime/graphicsStream.h>
#include <vma/vma.hpp>
#include <oa/runtime/stream.h>
#include <oa/ui/text.h>
#include <oa/ui/ui.h>
#include <oa/ui/renderConfig.h>
#include <oa/vision/fnImage.h>

#include "../runtime/textureAccess.h"

namespace {

constexpr VkFormat TargetFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr oa::U32 MaxTargetSlots = 4U;
constexpr oa::U32 NoActiveSlot = oa::Limits<oa::U32>::max();

enum class UiSlotState : oa::U8 {
	Free,
	Recording,
	Submitted,
	Retired,
};

enum class UiSubmissionRoute : oa::U8 {
	None,
	Compute,
	Graphics,
};

struct UiTarget {
	VkImage image = VK_NULL_HANDLE;
	VkImageView view = VK_NULL_HANDLE;
	vma::Allocation allocation = VK_NULL_HANDLE;
	VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
	oa::U32 bindlessIndex = OA_BINDLESS_INVALID;
	oavk::Buffer readback;
};

struct UiSlot {
	UiSlotState state = UiSlotState::Free;
	UiSubmissionRoute route = UiSubmissionRoute::None;
	oa::U64 generation = 0U;
	UiTarget target;
	oavk::Stream* computeStream = nullptr;
	oa::Optional<oa::GraphicsStreamLease> graphicsLease;
	oa::Event producer;
	oa::Event consumer;
};

[[nodiscard]] oa::U64 nextGeneration(oa::U64 inGeneration) noexcept {
	++inGeneration;
	return inGeneration == 0U ? 1U : inGeneration;
}

[[nodiscard]] bool checkedMultiply(
	oa::U64 inA,
	oa::U64 inB,
	oa::U64& outResult) noexcept {
	if (inA != 0U && inB > oa::Limits<oa::U64>::max() / inA) {
		return false;
	}
	outResult = inA * inB;
	return true;
}

[[nodiscard]] oa::Status validateExtent(
	oa::Engine& inEngine,
	const VklInstanceTable& inInstanceTable,
	oa::U32 inWidth,
	oa::U32 inHeight) {
	if (inWidth == 0U || inHeight == 0U) {
		return oa::Status::invalidArgument(
			"oa::Renderer target dimensions must be non-zero");
	}
	if (inWidth > static_cast<oa::U32>(oa::Limits<oa::I32>::max())
		|| inHeight > static_cast<oa::U32>(oa::Limits<oa::I32>::max())) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::Renderer target dimensions exceed signed UI coordinates");
	}
	const VkPhysicalDevice physicalDevice = static_cast<VkPhysicalDevice>(
		oa::EngineDeviceAccess::get(inEngine).physicalDevice);
	if (physicalDevice == VK_NULL_HANDLE
		|| inInstanceTable.vkGetPhysicalDeviceProperties == nullptr
		|| inInstanceTable.vkGetPhysicalDeviceImageFormatProperties == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer requires physical-device capability queries");
	}
	VkPhysicalDeviceProperties properties{};
	inInstanceTable.vkGetPhysicalDeviceProperties(physicalDevice, &properties);
	if (inWidth > properties.limits.maxImageDimension2D
		|| inHeight > properties.limits.maxImageDimension2D) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::Renderer target exceeds maxImageDimension2D");
	}
	oa::U64 pixels = 0U;
	oa::U64 bytes = 0U;
	if (!checkedMultiply(inWidth, inHeight, pixels)
		|| !checkedMultiply(pixels, 4U, bytes)
		|| bytes > static_cast<oa::U64>(oa::Limits<oa::Usize>::max())) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::Renderer readback size overflows host address space");
	}
	const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT
		| VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
		| VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	VkImageFormatProperties imageProperties{};
	const VkResult formatResult =
		inInstanceTable.vkGetPhysicalDeviceImageFormatProperties(
			physicalDevice,
			TargetFormat,
			VK_IMAGE_TYPE_2D,
			VK_IMAGE_TILING_OPTIMAL,
			usage,
			0U,
			&imageProperties);
	if (formatResult == VK_ERROR_FORMAT_NOT_SUPPORTED) {
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			"oa::Renderer requires RGBA8 UNORM storage, sampling, and transfer support");
	}
	if (formatResult != VK_SUCCESS) {
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			"oa::Renderer image-format capability query failed");
	}
	if (inWidth > imageProperties.maxExtent.width
		|| inHeight > imageProperties.maxExtent.height) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::Renderer target exceeds queried RGBA8 image extent");
	}
	return oa::Status::ok();
}

void destroyTarget(oa::Engine& inEngine, UiTarget& inOutTarget) noexcept {
	if (inOutTarget.bindlessIndex != OA_BINDLESS_INVALID) {
		oa::EngineBindlessAccess::get(inEngine).deregisterStorageImage(
			inOutTarget.bindlessIndex);
	}
	const VkDevice device = static_cast<VkDevice>(
		oa::EngineDeviceAccess::get(inEngine).device);
	if (inOutTarget.view != VK_NULL_HANDLE) {
		oa::EngineDeviceAccess::get(inEngine).deviceDispatch.vkDestroyImageView(
			device, inOutTarget.view, nullptr);
	}
	if (inOutTarget.image != VK_NULL_HANDLE) {
		vma::destroyImage(
			static_cast<vma::Allocator>(
				oa::EngineAllocatorAccess::get(inEngine).allocator),
			inOutTarget.image,
			inOutTarget.allocation);
	}
	oa::EngineAllocatorAccess::get(inEngine).free(inOutTarget.readback);
	inOutTarget = {};
}

[[nodiscard]] oa::Status createTarget(
	oa::Engine& inEngine,
	oa::U32 inWidth,
	oa::U32 inHeight,
	UiTarget& outTarget) {
	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = TargetFormat;
	imageInfo.extent = {inWidth, inHeight, 1U};
	imageInfo.mipLevels = 1U;
	imageInfo.arrayLayers = 1U;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT
		| VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
		| VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	const auto& queues = oa::EngineDeviceAccess::get(inEngine).queues;
	const oa::U32 families[2] = {
		queues.computeQueueFamily,
		queues.graphicsQueueFamily,
	};
	if (queues.graphicsQueue != nullptr
		&& queues.graphicsQueueFamily != oavk::EnumerationIndexUnset
		&& queues.graphicsQueueFamily != queues.computeQueueFamily) {
		imageInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
		imageInfo.queueFamilyIndexCount = 2U;
		imageInfo.pQueueFamilyIndices = families;
	} else {
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}

	vma::AllocationCreateInfo allocationInfo{};
	allocationInfo.usage = vma::memoryUsageGpuOnly;
	if (vma::createImage(
		static_cast<vma::Allocator>(
			oa::EngineAllocatorAccess::get(inEngine).allocator),
		&imageInfo,
		&allocationInfo,
		&outTarget.image,
		&outTarget.allocation,
		nullptr) != VK_SUCCESS) {
		return oa::Status::error(
			oa::StatusCode::OutOfMemory,
			"oa::Renderer target image allocation failed");
	}

	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = outTarget.image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = TargetFormat;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.levelCount = 1U;
	viewInfo.subresourceRange.layerCount = 1U;
	if (oa::EngineDeviceAccess::get(inEngine).deviceDispatch.vkCreateImageView(
		static_cast<VkDevice>(oa::EngineDeviceAccess::get(inEngine).device),
		&viewInfo,
		nullptr,
		&outTarget.view) != VK_SUCCESS) {
		destroyTarget(inEngine, outTarget);
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			"oa::Renderer target image-view creation failed");
	}

	outTarget.bindlessIndex = oa::EngineBindlessAccess::get(inEngine)
		.registerStorageImage(
			oa::EngineDeviceAccess::get(inEngine),
			outTarget.view,
			VK_IMAGE_LAYOUT_GENERAL);
	if (outTarget.bindlessIndex == OA_BINDLESS_INVALID) {
		destroyTarget(inEngine, outTarget);
		return oa::Status::error(
			oa::StatusCode::ResourceExhausted,
			"oa::Renderer storage-image descriptor heap is full");
	}

	oa::U64 pixels = 0U;
	oa::U64 bytes = 0U;
	if (!checkedMultiply(inWidth, inHeight, pixels)
		|| !checkedMultiply(pixels, 4U, bytes)) {
		destroyTarget(inEngine, outTarget);
		return oa::Status::error(
			oa::StatusCode::Internal,
			"oa::Renderer validated extent overflowed readback size");
	}
	auto readback = oa::EngineAllocatorAccess::get(inEngine)
		.allocHostReadback(bytes);
	if (!readback.isOk()) {
		destroyTarget(inEngine, outTarget);
		return readback.getStatus();
	}
	outTarget.readback = oa::move(*readback);
	return oa::Status::ok();
}

void recordTargetBegin(
	const VklDeviceTable& inDispatch,
	VkCommandBuffer inCommandBuffer,
	UiTarget& inTarget,
	const oa::Color& inClearColor) {
	VkImageSubresourceRange range{};
	range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	range.levelCount = 1U;
	range.layerCount = 1U;

	VkImageMemoryBarrier2 reuse{};
	reuse.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	reuse.srcStageMask = inTarget.layout == VK_IMAGE_LAYOUT_UNDEFINED
		? VK_PIPELINE_STAGE_2_NONE
		: VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	reuse.srcAccessMask = inTarget.layout == VK_IMAGE_LAYOUT_UNDEFINED
		? VK_ACCESS_2_NONE
		: VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
	reuse.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	reuse.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	reuse.oldLayout = inTarget.layout;
	reuse.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	reuse.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	reuse.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	reuse.image = inTarget.image;
	reuse.subresourceRange = range;
	VkDependencyInfo dependency{};
	dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	dependency.imageMemoryBarrierCount = 1U;
	dependency.pImageMemoryBarriers = &reuse;
	inDispatch.vkCmdPipelineBarrier2(inCommandBuffer, &dependency);

	VkClearColorValue clear{};
	clear.float32[0] = inClearColor.r;
	clear.float32[1] = inClearColor.g;
	clear.float32[2] = inClearColor.b;
	clear.float32[3] = inClearColor.a;
	inDispatch.vkCmdClearColorImage(
		inCommandBuffer,
		inTarget.image,
		VK_IMAGE_LAYOUT_GENERAL,
		&clear,
		1U,
		&range);

	VkImageMemoryBarrier2 clearToUi{};
	clearToUi.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	clearToUi.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	clearToUi.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	clearToUi.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
	clearToUi.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
		| VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
	clearToUi.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	clearToUi.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	clearToUi.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	clearToUi.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	clearToUi.image = inTarget.image;
	clearToUi.subresourceRange = range;
	dependency.pImageMemoryBarriers = &clearToUi;
	inDispatch.vkCmdPipelineBarrier2(inCommandBuffer, &dependency);
}

void recordTargetFinish(
	const VklDeviceTable& inDispatch,
	VkCommandBuffer inCommandBuffer,
	const UiTarget& inTarget) {
	VkImageMemoryBarrier2 output{};
	output.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	output.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT
		| VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
	output.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT
		| VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
	output.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	output.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
	output.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	output.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	output.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	output.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	output.image = inTarget.image;
	output.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	output.subresourceRange.levelCount = 1U;
	output.subresourceRange.layerCount = 1U;
	VkDependencyInfo dependency{};
	dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	dependency.imageMemoryBarrierCount = 1U;
	dependency.pImageMemoryBarriers = &output;
	inDispatch.vkCmdPipelineBarrier2(inCommandBuffer, &dependency);
}

void recordReadback(
	const VklDeviceTable& inDispatch,
	VkCommandBuffer inCommandBuffer,
	const UiTarget& inTarget,
	oa::U32 inWidth,
	oa::U32 inHeight) {
	VkImageMemoryBarrier2 toCopy{};
	toCopy.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	toCopy.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	toCopy.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
	toCopy.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	toCopy.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	toCopy.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	toCopy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	toCopy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toCopy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toCopy.image = inTarget.image;
	toCopy.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	toCopy.subresourceRange.levelCount = 1U;
	toCopy.subresourceRange.layerCount = 1U;
	VkDependencyInfo dependency{};
	dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	dependency.imageMemoryBarrierCount = 1U;
	dependency.pImageMemoryBarriers = &toCopy;
	inDispatch.vkCmdPipelineBarrier2(inCommandBuffer, &dependency);

	VkBufferImageCopy copy{};
	copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copy.imageSubresource.layerCount = 1U;
	copy.imageExtent = {inWidth, inHeight, 1U};
	inDispatch.vkCmdCopyImageToBuffer(
		inCommandBuffer,
		inTarget.image,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		static_cast<VkBuffer>(inTarget.readback.buffer),
		1U,
		&copy);

	VkImageMemoryBarrier2 restore{};
	restore.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	restore.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	restore.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	restore.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	restore.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT
		| VK_ACCESS_2_MEMORY_WRITE_BIT;
	restore.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	restore.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	restore.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	restore.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	restore.image = inTarget.image;
	restore.subresourceRange = toCopy.subresourceRange;
	VkBufferMemoryBarrier2 host{};
	host.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
	host.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	host.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	host.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
	host.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
	host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	host.buffer = static_cast<VkBuffer>(inTarget.readback.buffer);
	host.offset = 0U;
	host.size = VK_WHOLE_SIZE;
	dependency.imageMemoryBarrierCount = 1U;
	dependency.pImageMemoryBarriers = &restore;
	dependency.bufferMemoryBarrierCount = 1U;
	dependency.pBufferMemoryBarriers = &host;
	inDispatch.vkCmdPipelineBarrier2(inCommandBuffer, &dependency);
}

} // namespace

class oa::Renderer::UiImpl final : public oa::Renderer::Impl {
public:
	VklInstanceTable instanceTable_{};
	oa::UiRenderConfig config_;
	oa::TextAtlas textAtlas;
	oa::Ui uiSession_;
	oa::Vector<UiSlot> slots;
	oa::U32 activeSlot = NoActiveSlot;
	oa::U64 targetGeneration_ = 1U;

	[[nodiscard]] oa::Status initialize(
		oa::Engine& inEngine,
		const oa::UiRenderConfig& inConfig);
	[[nodiscard]] oa::Status createSlots();
	[[nodiscard]] oa::Status validateDependencies(
		oa::Span<const oa::Event> inDependencies,
		UiSubmissionRoute& outRoute) const;
	[[nodiscard]] oa::Status acquireRecording(
		UiSlot& inSlot,
		UiSubmissionRoute inRoute,
		VkCommandBuffer& outCommandBuffer);
	[[nodiscard]] oa::Status cancelRecording(UiSlot& inSlot);
	[[nodiscard]] oa::Status releaseProducer(UiSlot& inSlot);
	[[nodiscard]] oa::Status validateFrame(
		const oa::RenderFrame& inFrame,
		UiSlotState inRequiredState,
		UiSlot*& outSlot);
	[[nodiscard]] oa::Status collectRetired();
	[[nodiscard]] oa::Status closeUi();
	void destroyAll() noexcept;

	[[nodiscard]] oa::Status beginUiFrame(
		oa::F32 inDeltaMs,
		oa::F32 inContentScale) override;
	[[nodiscard]] oa::Ui* ui() noexcept override { return &uiSession_; }
	[[nodiscard]] const oa::Ui* ui() const noexcept override { return &uiSession_; }
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

oa::Status oa::Renderer::UiImpl::initialize(
	oa::Engine& inEngine,
	const oa::UiRenderConfig& inConfig) {
	engine = &inEngine;
	config_ = inConfig;
	if (config_.targetSlotCount_ == 0U
		|| config_.targetSlotCount_ > MaxTargetSlots) {
		return oa::Status::invalidArgument(
			"oa::Renderer target slot count must be in [1, 4]");
	}
	OA_RETURN_IF_ERROR(config_.style_.validate());
	if (!inEngine.isReady()
		|| oa::EngineDeviceAccess::get(inEngine).instance == nullptr
		|| oa::EngineDeviceAccess::get(inEngine).device == nullptr
		|| oa::EngineDeviceAccess::get(inEngine).queues.computeQueue == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer requires a ready compute engine");
	}
	vklLoadInstanceTable(
		&instanceTable_,
		static_cast<VkInstance>(
			oa::EngineDeviceAccess::get(inEngine).instance));
	OA_RETURN_IF_ERROR(validateExtent(
		inEngine, instanceTable_, config_.width_, config_.height_));
	OA_RETURN_IF_ERROR(uiSession_.init(inEngine, config_.style_));
	OA_RETURN_IF_ERROR(uiSession_.initBlit(nullptr));
	OA_RETURN_IF_ERROR(textAtlas.init(inEngine));
	OA_RETURN_IF_ERROR(uiSession_.bindTextAtlas(textAtlas));
	return createSlots();
}

oa::Status oa::Renderer::UiImpl::createSlots() {
	slots.reserve(config_.targetSlotCount_);
	for (oa::U32 index = 0U; index < config_.targetSlotCount_; ++index) {
		slots.emplaceBack();
		const oa::Status status = createTarget(
			*engine, config_.width_, config_.height_, slots.back().target);
		if (!status.isOk()) return status;
	}
	return oa::Status::ok();
}

oa::Status oa::Renderer::UiImpl::validateDependencies(
	oa::Span<const oa::Event> inDependencies,
	UiSubmissionRoute& outRoute) const {
	const auto& queues = oa::EngineDeviceAccess::get(*engine).queues;
	outRoute = UiSubmissionRoute::Compute;
	oa::U32 selectedFamily = queues.computeQueueFamily;
	bool hasSelectedFamily = false;
	for (const oa::Event& dependency : inDependencies) {
		if (!engine->ownsEvent(dependency) || !dependency.hasQueueFamily()) {
			return oa::Status::invalidArgument(
				"oa::Renderer dependencies must be exact events from its engine");
		}
		const oa::U32 family = dependency.queueFamily();
		const bool compute = family == queues.computeQueueFamily;
		const bool graphics = queues.graphicsQueue != nullptr
			&& queues.graphicsQueueFamily != oavk::EnumerationIndexUnset
			&& family == queues.graphicsQueueFamily;
		if (!compute && !graphics) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::Renderer dependency queue family is outside its compute/graphics target sharing set");
		}
		if (hasSelectedFamily && family != selectedFamily) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::Renderer cannot compose mixed queue-family producers without explicit source ownership lowering");
		}
		selectedFamily = family;
		hasSelectedFamily = true;
		outRoute = graphics && !compute
			? UiSubmissionRoute::Graphics
			: UiSubmissionRoute::Compute;
	}
	return oa::Status::ok();
}

oa::Status oa::Renderer::UiImpl::acquireRecording(
	UiSlot& inSlot,
	UiSubmissionRoute inRoute,
	VkCommandBuffer& outCommandBuffer) {
	inSlot.route = inRoute;
	if (inRoute == UiSubmissionRoute::Graphics) {
		auto lease = oa::GraphicsStreamLease::acquire(*engine);
		if (!lease.isOk()) return lease.getStatus();
		inSlot.graphicsLease.emplace(oa::move(*lease));
		oavk::Stream* stream = inSlot.graphicsLease->getStream();
		if (stream == nullptr) {
			(void)inSlot.graphicsLease->cancel();
			inSlot.graphicsLease.reset();
			return oa::Status::error(
				oa::StatusCode::Internal,
				"oa::Renderer graphics lease lost its recording stream");
		}
		outCommandBuffer = static_cast<VkCommandBuffer>(stream->commandBuffer);
		return oa::Status::ok();
	}
	inSlot.computeStream = oa::EngineSubmissionAccess::acquireStream(*engine);
	if (inSlot.computeStream == nullptr) {
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			"oa::Renderer failed to acquire a compute stream");
	}
	const oa::Status beginStatus = inSlot.computeStream->begin(
		oa::EngineDeviceAccess::get(*engine));
	if (!beginStatus.isOk()) {
		oa::EngineSubmissionAccess::releaseStream(
			*engine, inSlot.computeStream);
		inSlot.computeStream = nullptr;
		return beginStatus;
	}
	outCommandBuffer = static_cast<VkCommandBuffer>(
		inSlot.computeStream->commandBuffer);
	return oa::Status::ok();
}

oa::Status oa::Renderer::UiImpl::cancelRecording(UiSlot& inSlot) {
	oa::Status status = oa::Status::ok();
	if (inSlot.graphicsLease.hasValue()) {
		status = inSlot.graphicsLease->cancel();
		if (!status.isOk()) (void)inSlot.graphicsLease->close();
		inSlot.graphicsLease.reset();
	}
	if (inSlot.computeStream != nullptr) {
		status = inSlot.computeStream->resetUnsubmitted(
			oa::EngineDeviceAccess::get(*engine));
		oa::EngineSubmissionAccess::releaseStream(
			*engine, inSlot.computeStream);
		inSlot.computeStream = nullptr;
	}
	inSlot.route = UiSubmissionRoute::None;
	return status;
}

oa::Status oa::Renderer::UiImpl::releaseProducer(UiSlot& inSlot) {
	if (inSlot.computeStream != nullptr) {
		OA_RETURN_IF_ERROR(inSlot.computeStream->synchronize(
			oa::EngineDeviceAccess::get(*engine)));
		oa::EngineSubmissionAccess::releaseStream(
			*engine, inSlot.computeStream);
		inSlot.computeStream = nullptr;
	}
	if (inSlot.graphicsLease.hasValue()) {
		OA_RETURN_IF_ERROR(inSlot.graphicsLease->recycle(inSlot.producer));
		inSlot.graphicsLease.reset();
	}
	inSlot.route = UiSubmissionRoute::None;
	return oa::Status::ok();
}

oa::Status oa::Renderer::UiImpl::validateFrame(
	const oa::RenderFrame& inFrame,
	UiSlotState inRequiredState,
	UiSlot*& outSlot) {
	if (inFrame.sourceKind_ != oa::RenderFrame::SourceKind::Ui) {
		return oa::Status::invalidArgument(
			"oa::Renderer frame was produced by a different render mode");
	}
	if (closed || inFrame.targetGeneration_ != targetGeneration_
		|| inFrame.width_ != config_.width_
		|| inFrame.height_ != config_.height_
		|| inFrame.slot_ >= slots.size()) {
		return oa::Status::invalidArgument(
			"oa::Renderer frame belongs to a stale target generation");
	}
	UiSlot& slot = slots[inFrame.slot_];
	if (slot.state != inRequiredState
		|| slot.generation != inFrame.slotGeneration_
		|| !inFrame.color_.isImageBacked()
		|| oa::TextureAccess::engine(inFrame.color_) != engine
		|| oa::TextureAccess::image(inFrame.color_) != slot.target.image
		|| oa::TextureAccess::view(inFrame.color_) != slot.target.view
		|| oa::TextureAccess::format(inFrame.color_) != TargetFormat
		|| oa::TextureAccess::layout(inFrame.color_) != VK_IMAGE_LAYOUT_GENERAL
		|| !engine->ownsEvent(inFrame.producer_)
		|| !slot.producer.isSameCompletion(inFrame.producer_)) {
		return oa::Status::invalidArgument(
			"oa::Renderer frame is stale or not the exact live completion");
	}
	outSlot = &slot;
	return oa::Status::ok();
}

oa::Status oa::Renderer::UiImpl::collectRetired() {
	if (closed) return oa::Status::ok();
	for (UiSlot& slot : slots) {
		if (slot.state != UiSlotState::Retired) continue;
		if (!slot.producer.isValid()) {
			return oa::Status::error(
				oa::StatusCode::Internal,
				"oa::Renderer retired target lost its producer event");
		}
		if (!slot.producer.isComplete()
			|| (slot.consumer.isValid() && !slot.consumer.isComplete())) {
			continue;
		}
		OA_RETURN_IF_ERROR(slot.producer.wait());
		if (slot.consumer.isValid()) OA_RETURN_IF_ERROR(slot.consumer.wait());
		OA_RETURN_IF_ERROR(releaseProducer(slot));
		slot.producer = {};
		slot.consumer = {};
		slot.state = UiSlotState::Free;
	}
	return oa::Status::ok();
}

oa::Status oa::Renderer::UiImpl::closeUi() {
	return uiSession_.close();
}

void oa::Renderer::UiImpl::destroyAll() noexcept {
	if (engine == nullptr
		|| oa::EngineDeviceAccess::get(*engine).device == nullptr) return;
	for (UiSlot& slot : slots) {
		destroyTarget(*engine, slot.target);
		slot.computeStream = nullptr;
		slot.graphicsLease.reset();
		slot.producer = {};
		slot.consumer = {};
		slot.route = UiSubmissionRoute::None;
		slot.state = UiSlotState::Free;
	}
	slots.clear();
	textAtlas = {};
}

bool oa::Renderer::UiImpl::prepareNonWaitingRetirement() noexcept {
	if (closed) return false;
	if (activeSlot < slots.size()) {
		uiSession_.endFrame();
		UiSlot& slot = slots[activeSlot];
		if (slot.computeStream != nullptr || slot.graphicsLease.hasValue()) {
			(void)cancelRecording(slot);
		}
		slot.state = UiSlotState::Free;
		slot.producer = {};
		slot.consumer = {};
	}
	activeSlot = NoActiveSlot;
	bool hasSubmission = false;
	for (UiSlot& slot : slots) {
		if (slot.state == UiSlotState::Submitted) {
			if (slot.graphicsLease.hasValue()) {
				(void)slot.graphicsLease->close();
				slot.graphicsLease.reset();
			}
			slot.state = UiSlotState::Retired;
			hasSubmission = true;
		} else if (slot.state == UiSlotState::Retired) {
			hasSubmission = true;
		}
	}
	return hasSubmission;
}

oa::Status oa::Renderer::UiImpl::cleanupWithoutSubmission() {
	OA_RETURN_IF_ERROR(closeUi());
	destroyAll();
	closed = true;
	return oa::Status::ok();
}

oa::Status oa::Renderer::UiImpl::completeRetired() {
	for (UiSlot& slot : slots) {
		if (slot.state != UiSlotState::Retired) continue;
		OA_RETURN_IF_ERROR(slot.producer.wait());
		if (slot.consumer.isValid()) OA_RETURN_IF_ERROR(slot.consumer.wait());
		OA_RETURN_IF_ERROR(releaseProducer(slot));
		slot.producer = {};
		slot.consumer = {};
		slot.state = UiSlotState::Free;
	}
	OA_RETURN_IF_ERROR(closeUi());
	destroyAll();
	closed = true;
	return oa::Status::ok();
}

oa::Result<oa::UniquePtr<oa::Renderer>> oa::Renderer::create(
	oa::Engine& inEngine,
	const oa::UiRenderConfig& inConfig) {
	oa::UniquePtr<oa::Renderer> renderer(new oa::Renderer());
	auto* impl = new UiImpl();
	const oa::Status status = impl->initialize(inEngine, inConfig);
	if (!status.isOk()) {
		(void)impl->cleanupWithoutSubmission();
		delete impl;
		return status;
	}
	renderer->impl_ = oa::UniquePtr<Impl>(impl);
	return oa::move(renderer);
}

oa::Status oa::Renderer::UiImpl::beginUiFrame(
	oa::F32 inDeltaMs,
	oa::F32 inContentScale) {
	if (closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	if (activeSlot < slots.size()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer already has an active frame");
	}
	OA_RETURN_IF_ERROR(collectRetired());
	oa::U32 slotIndex = static_cast<oa::U32>(slots.size());
	for (oa::U32 index = 0U; index < slots.size(); ++index) {
		if (slots[index].state == UiSlotState::Free) {
			slotIndex = index;
			break;
		}
	}
	if (slotIndex == slots.size()) {
		return oa::Status::error(
			oa::StatusCode::ResourceExhausted,
			"oa::Renderer target ring is full; consume or abandon an older frame");
	}
	UiSlot& slot = slots[slotIndex];
	slot.generation = nextGeneration(slot.generation);
	slot.state = UiSlotState::Recording;
	slot.route = UiSubmissionRoute::None;
	slot.producer = {};
	slot.consumer = {};
	activeSlot = slotIndex;
	uiSession_.beginFrame(
		inDeltaMs,
		{0, 0, static_cast<oa::I32>(config_.width_),
			static_cast<oa::I32>(config_.height_)},
		inContentScale);
	return oa::Status::ok();
}

oa::Result<oa::RenderFrame> oa::Renderer::UiImpl::submitFrame(
	oa::Span<const oa::Event> inDependencies) {
	if (closed
		|| activeSlot >= slots.size()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer has no active frame to submit");
	}
	UiSlot& slot = slots[activeSlot];
	UiSubmissionRoute route = UiSubmissionRoute::None;
	oa::Status status = validateDependencies(inDependencies, route);
	if (!status.isOk()) return status;

	VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
	status = acquireRecording(slot, route, commandBuffer);
	if (status.isOk()) {
		recordTargetBegin(
			oa::EngineDeviceAccess::get(*engine).deviceDispatch,
			commandBuffer, slot.target, config_.style_.background);
		status = uiSession_.recordRender(
			commandBuffer, slot.target.bindlessIndex);
		if (status.isOk()) {
			recordTargetFinish(
				oa::EngineDeviceAccess::get(*engine).deviceDispatch,
				commandBuffer, slot.target);
		}
	}
	uiSession_.endFrame();
	if (!status.isOk()) {
		(void)cancelRecording(slot);
		slot.state = UiSlotState::Free;
		activeSlot = NoActiveSlot;
		return status;
	}

	oa::Result<oa::Event> submitted = oa::Status::error(
		oa::StatusCode::Internal,
		"oa::Renderer submission route was not selected");
	if (route == UiSubmissionRoute::Graphics) {
		submitted = slot.graphicsLease->submit(inDependencies);
	} else {
		oa::Vector<oavk::TimelineWait> waits;
		waits.reserve(inDependencies.size());
		for (const oa::Event& dependency : inDependencies) {
			waits.pushBack(oa::EventAccess::timelineWait(dependency));
		}
		status = waits.empty()
			? slot.computeStream->submit(*engine)
			: slot.computeStream->submitWithDependencies(
				*engine,
				oa::Span<const oavk::TimelineWait>(waits.data(), waits.size()));
		if (status.isOk()) {
			submitted = slot.computeStream->completion(
				oa::EngineDeviceAccess::get(*engine));
		} else {
			submitted = status;
		}
	}
	if (!submitted.isOk()) {
		if (slot.computeStream != nullptr) {
			(void)slot.computeStream->resetUnsubmitted(
				oa::EngineDeviceAccess::get(*engine));
			oa::EngineSubmissionAccess::releaseStream(
				*engine, slot.computeStream);
			slot.computeStream = nullptr;
		}
		if (slot.graphicsLease.hasValue()) {
			(void)slot.graphicsLease->close();
			slot.graphicsLease.reset();
		}
		slot.route = UiSubmissionRoute::None;
		slot.state = UiSlotState::Free;
		activeSlot = NoActiveSlot;
		return submitted.getStatus();
	}
	if (!submitted->isValid()) {
		if (slot.computeStream != nullptr) {
			(void)slot.computeStream->synchronize(
				oa::EngineDeviceAccess::get(*engine));
			oa::EngineSubmissionAccess::releaseStream(
				*engine, slot.computeStream);
			slot.computeStream = nullptr;
		}
		if (slot.graphicsLease.hasValue()) {
			(void)slot.graphicsLease->close();
			slot.graphicsLease.reset();
		}
		slot.route = UiSubmissionRoute::None;
		slot.state = UiSlotState::Free;
		activeSlot = NoActiveSlot;
		return oa::Status::error(
			oa::StatusCode::Internal,
			"oa::Renderer submission produced no exact completion event");
	}

	slot.producer = *submitted;
	slot.target.layout = VK_IMAGE_LAYOUT_GENERAL;
	slot.state = UiSlotState::Submitted;
	const oa::U32 submittedSlot = activeSlot;
	activeSlot = NoActiveSlot;
	status = uiSession_.markFrameSubmitted(slot.producer);
	if (!status.isOk()) {
		if (slot.graphicsLease.hasValue()) {
			(void)slot.graphicsLease->close();
			slot.graphicsLease.reset();
		}
		slot.state = UiSlotState::Retired;
		return status;
	}

	oa::RenderFrame frame;
	frame.slot_ = submittedSlot;
	frame.slotGeneration_ = slot.generation;
	frame.targetGeneration_ = targetGeneration_;
	frame.width_ = config_.width_;
	frame.height_ = config_.height_;
	frame.color_ = oa::TextureAccess::fromBorrowedImage(
		*engine,
		slot.target.image,
		slot.target.view,
		TargetFormat,
		VK_IMAGE_LAYOUT_GENERAL,
		static_cast<oa::I32>(frame.width_),
		static_cast<oa::I32>(frame.height_));
	frame.producer_ = slot.producer;
	frame.sourceKind_ = oa::RenderFrame::SourceKind::Ui;
	return frame;
}

oa::Status oa::Renderer::UiImpl::cancelFrame() {
	if (closed
		|| activeSlot >= slots.size()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer has no active frame to cancel");
	}
	UiSlot& slot = slots[activeSlot];
	uiSession_.endFrame();
	oa::Status status = oa::Status::ok();
	if (slot.computeStream != nullptr || slot.graphicsLease.hasValue()) {
		status = cancelRecording(slot);
	}
	slot.state = UiSlotState::Free;
	slot.route = UiSubmissionRoute::None;
	slot.producer = {};
	slot.consumer = {};
	activeSlot = NoActiveSlot;
	return status;
}

oa::Result<oa::RenderReadback> oa::Renderer::UiImpl::consumeReadback(
	const oa::RenderFrame& inFrame) {
	if (closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	UiSlot* slot = nullptr;
	oa::Status status = validateFrame(
		inFrame, UiSlotState::Submitted, slot);
	if (!status.isOk()) return status;

	oavk::Stream* readbackStream =
		oa::EngineSubmissionAccess::acquireStream(*engine);
	if (readbackStream == nullptr) {
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			"oa::Renderer failed to acquire its readback stream");
	}
	oa::Event readbackCompletion;
	status = readbackStream->begin(
		oa::EngineDeviceAccess::get(*engine));
	if (status.isOk()) {
		recordReadback(
			oa::EngineDeviceAccess::get(*engine).deviceDispatch,
			static_cast<VkCommandBuffer>(readbackStream->commandBuffer),
			slot->target,
			config_.width_,
			config_.height_);
		const oavk::TimelineWait wait =
			oa::EventAccess::timelineWait(inFrame.producer_);
		status = readbackStream->submitWithDependencies(
			*engine,
			oa::Span<const oavk::TimelineWait>(&wait, 1U));
		if (status.isOk()) {
			readbackCompletion = readbackStream->completion(
				oa::EngineDeviceAccess::get(*engine));
			status = readbackStream->synchronize(
				oa::EngineDeviceAccess::get(*engine));
		}
	}
	if (!status.isOk() && !readbackStream->submitted) {
		(void)readbackStream->resetUnsubmitted(
			oa::EngineDeviceAccess::get(*engine));
	}
	oa::EngineSubmissionAccess::releaseStream(
		*engine, readbackStream);
	if (!status.isOk()) {
		// A submitted readback is also an image consumer. Preserve its exact
		// completion even when the synchronous wait reports a device failure, so
		// abandonFrame cannot make the target reusable underneath that copy.
		if (readbackCompletion.isValid()) {
			if (slot->graphicsLease.hasValue()) {
				(void)slot->graphicsLease->close();
				slot->graphicsLease.reset();
			}
			slot->consumer = readbackCompletion;
			slot->state = UiSlotState::Retired;
		}
		return status;
	}

	oa::U64 pixels = 0U;
	oa::U64 bytes = 0U;
	if (!checkedMultiply(
		config_.width_, config_.height_, pixels)
		|| !checkedMultiply(pixels, 4U, bytes)) {
		return oa::Status::error(
			oa::StatusCode::Internal,
			"oa::Renderer validated target overflowed readback size");
	}
	if (!oa::EngineAllocatorAccess::get(*engine).invalidateHostBuffer(
		slot->target.readback, 0U, bytes)) {
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			"oa::Renderer readback invalidate failed");
	}
	oa::RenderReadback readback;
	readback.width_ = config_.width_;
	readback.height_ = config_.height_;
	readback.colorRgba8_.resize(static_cast<oa::Usize>(bytes));
	oa::memcpy(
		readback.colorRgba8_.data(),
		slot->target.readback.mappedPtr,
		static_cast<oa::Usize>(bytes));
	OA_RETURN_IF_ERROR(releaseProducer(*slot));
	slot->producer = {};
	slot->consumer = {};
	slot->state = UiSlotState::Free;
	return readback;
}

oa::Status oa::Renderer::UiImpl::markConsumed(
	const oa::RenderFrame& inFrame,
	const oa::Event& inConsumer) {
	if (closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	UiSlot* slot = nullptr;
	OA_RETURN_IF_ERROR(validateFrame(
		inFrame, UiSlotState::Submitted, slot));
	if (!engine->ownsEvent(inConsumer)
		|| !inConsumer.hasQueueFamily()) {
		return oa::Status::invalidArgument(
			"oa::Renderer consumer must be an exact event from this engine");
	}
	const auto& queues = oa::EngineDeviceAccess::get(*engine).queues;
	const bool compute = inConsumer.queueFamily()
		== queues.computeQueueFamily;
	const bool graphics = queues.graphicsQueue != nullptr
		&& queues.graphicsQueueFamily != oavk::EnumerationIndexUnset
		&& inConsumer.queueFamily() == queues.graphicsQueueFamily;
	if (!compute && !graphics) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer consumer queue family is outside the target sharing set");
	}
	if (inConsumer.isSameCompletion(slot->producer)) {
		return oa::Status::invalidArgument(
			"oa::Renderer consumer completion cannot alias its producer completion");
	}
	const void* producerSemaphore =
		oa::EventAccess::semaphoreHandle(slot->producer);
	const void* consumerSemaphore =
		oa::EventAccess::semaphoreHandle(inConsumer);
	if (producerSemaphore != nullptr && consumerSemaphore != nullptr
		&& producerSemaphore == consumerSemaphore
		&& inConsumer.value() <= slot->producer.value()) {
		return oa::Status::invalidArgument(
			"oa::Renderer consumer completion must follow its producer on a shared timeline");
	}
	oa::Status closeStatus = oa::Status::ok();
	if (slot->graphicsLease.hasValue()) {
		closeStatus = slot->graphicsLease->close();
		slot->graphicsLease.reset();
	}
	slot->consumer = inConsumer;
	slot->state = UiSlotState::Retired;
	return closeStatus;
}

oa::Status oa::Renderer::UiImpl::abandonFrame(const oa::RenderFrame& inFrame) {
	if (closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	UiSlot* slot = nullptr;
	OA_RETURN_IF_ERROR(validateFrame(
		inFrame, UiSlotState::Submitted, slot));
	oa::Status closeStatus = oa::Status::ok();
	if (slot->graphicsLease.hasValue()) {
		closeStatus = slot->graphicsLease->close();
		slot->graphicsLease.reset();
	}
	slot->consumer = {};
	slot->state = UiSlotState::Retired;
	return closeStatus;
}

oa::Status oa::Renderer::UiImpl::collect() {
	if (closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	return collectRetired();
}

oa::Status oa::Renderer::UiImpl::resize(oa::U32 inWidth, oa::U32 inHeight) {
	if (closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	OA_RETURN_IF_ERROR(collectRetired());
	if (activeSlot < slots.size()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer resize requires cancelling the active frame");
	}
	for (const UiSlot& slot : slots) {
		if (slot.state != UiSlotState::Free) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::Renderer resize requires every old-generation frame to be consumed or retired");
		}
	}
	OA_RETURN_IF_ERROR(validateExtent(
		*engine, instanceTable_, inWidth, inHeight));
	if (inWidth == config_.width_
		&& inHeight == config_.height_) return oa::Status::ok();

	oa::Vector<UiTarget> replacements(slots.size());
	for (UiTarget& target : replacements) {
		const oa::Status status = createTarget(
			*engine, inWidth, inHeight, target);
		if (!status.isOk()) {
			for (UiTarget& replacement : replacements) {
				destroyTarget(*engine, replacement);
			}
			return status;
		}
	}
	for (oa::Usize index = 0U; index < slots.size(); ++index) {
		destroyTarget(*engine, slots[index].target);
		slots[index].target = replacements[index];
		replacements[index] = {};
		slots[index].generation = nextGeneration(
			slots[index].generation);
	}
	config_.width_ = inWidth;
	config_.height_ = inHeight;
	targetGeneration_ = nextGeneration(targetGeneration_);
	return oa::Status::ok();
}

oa::Status oa::Renderer::UiImpl::close() {
	if (closed) return oa::Status::ok();
	if (activeSlot < slots.size()) {
		OA_RETURN_IF_ERROR(cancelFrame());
	}
	for (UiSlot& slot : slots) {
		if (slot.state == UiSlotState::Submitted
			|| slot.state == UiSlotState::Retired) {
			OA_RETURN_IF_ERROR(slot.producer.wait());
			if (slot.consumer.isValid()) {
				OA_RETURN_IF_ERROR(slot.consumer.wait());
			}
			OA_RETURN_IF_ERROR(releaseProducer(slot));
		}
		slot.producer = {};
		slot.consumer = {};
		slot.state = UiSlotState::Free;
	}
	// UI owns completion-tracked pipelines/descriptors and borrows textAtlas.
	// release it while every producer semaphore and the atlas are still live.
	OA_RETURN_IF_ERROR(closeUi());
	destroyAll();
	closed = true;
	return oa::Status::ok();
}

oa::Status oa::Renderer::saveTo(
	const oa::RenderFrame& inFrame,
	oa::StringView inPath,
	oa::U32 inQuality) {
	auto readback = consumeReadback(inFrame);
	if (!readback.isOk()) return readback.getStatus();
	return oa::FnImage::saveRgbaFile(
		oa::Span<const oa::U8>(
			readback->colorRgba8_.data(),
			readback->colorRgba8_.size()),
		readback->width_,
		readback->height_,
		inPath,
		inQuality);
}
