#include <oa/runtime/stream.h>
#include <oa/core/std/memory.h>
#include <oa/core/std/utility.h>
#include <oa/core/log.h>
#include <oa/core/validation.h>
#include <oa/runtime/bindless.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/deviceAccess.h>
#include <oa/runtime/device.h>
#include <oa/runtime/pipeline.h>
#include <vkl/vkl.h>
#include <oa/runtime/engine/bindlessAccess.h>
#include <oa/runtime/engine/pipelineAccess.h>
#include <oa/runtime/engine/submissionAccess.h>
#include "engine/engineAccess.h"
#include "descriptorValidation.h"
#include "dispatchValidation.h"
#include "storageDtype.h"

// ─── move Semantics ────────────────────────────────────────────────────────────

oavk::Stream::Stream(oavk::Stream&& inOther) noexcept
	: commandPool(inOther.commandPool)
	, commandBuffer(inOther.commandBuffer)
	, deviceDispatch(inOther.deviceDispatch)
	, timelineSem(oa::move(inOther.timelineSem))
	, timelineValue(inOther.timelineValue)
	, pendingPools(oa::move(inOther.pendingPools))
	, queue(inOther.queue)
	, queueFamily(inOther.queueFamily)
	, recording(inOther.recording)
	, submitted(inOther.submitted)
	, suppressAutoBarrier(inOther.suppressAutoBarrier)
{
	inOther.commandPool = nullptr;
	inOther.commandBuffer = nullptr;
	inOther.deviceDispatch = nullptr;
	inOther.timelineSem.semaphore = nullptr;
	inOther.timelineValue = 0;
	inOther.queue = nullptr;
	inOther.recording = false;
	inOther.submitted = false;
	inOther.suppressAutoBarrier = false;
}

oavk::Stream& oavk::Stream::operator=(oavk::Stream&& inOther) noexcept {
	if (this != &inOther) {
		commandPool = inOther.commandPool;
		commandBuffer = inOther.commandBuffer;
		deviceDispatch = inOther.deviceDispatch;
		timelineSem = oa::move(inOther.timelineSem);
		timelineValue = inOther.timelineValue;
		pendingPools = oa::move(inOther.pendingPools);
		queue = inOther.queue;
		queueFamily = inOther.queueFamily;
		recording = inOther.recording;
		submitted = inOther.submitted;
		suppressAutoBarrier = inOther.suppressAutoBarrier;
		inOther.commandPool = nullptr;
		inOther.commandBuffer = nullptr;
		inOther.deviceDispatch = nullptr;
		inOther.timelineSem.semaphore = nullptr;
		inOther.timelineValue = 0;
		inOther.queue = nullptr;
		inOther.recording = false;
		inOther.submitted = false;
		inOther.suppressAutoBarrier = false;
	}
	return *this;
}

// ─── Lifecycle ─────────────────────────────────────────────────────────────────

oa::Result<oavk::Stream> oavk::Stream::create(
	const oavk::Device& inDevice, oa::U32 inQueueFamily, void* inQueue)
{
	VkDevice dev = static_cast<VkDevice>(inDevice.device);

	VkCommandPoolCreateInfo cpCI{};
	cpCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cpCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	cpCI.queueFamilyIndex = inQueueFamily;

	VkCommandPool pool = VK_NULL_HANDLE;
	VkResult r = inDevice.deviceDispatch.vkCreateCommandPool(dev, &cpCI, nullptr, &pool);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "stream: vkCreateCommandPool failed");
	}

	VkCommandBufferAllocateInfo cbAI{};
	cbAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cbAI.commandPool = pool;
	cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cbAI.commandBufferCount = 1;

	VkCommandBuffer cb = VK_NULL_HANDLE;
	r = inDevice.deviceDispatch.vkAllocateCommandBuffers(dev, &cbAI, &cb);
	if (r != VK_SUCCESS) {
		inDevice.deviceDispatch.vkDestroyCommandPool(dev, pool, nullptr);
		return oa::Status::error(oa::StatusCode::VulkanError, "stream: vkAllocateCommandBuffers failed");
	}

	auto semRes = oavk::TimelineSemaphore::create(inDevice, 0);
	if (!semRes) {
		inDevice.deviceDispatch.vkDestroyCommandPool(dev, pool, nullptr);
		return semRes.getStatus();
	}

	oavk::Stream s;
	s.commandPool = pool;
	s.commandBuffer = cb;
	s.deviceDispatch = &inDevice.deviceDispatch;
	s.timelineSem = oa::move(*semRes);
	s.timelineValue = 0;
	s.queue = inQueue;
	s.queueFamily = inQueueFamily;
	return s;
}

oa::Result<oavk::Stream> oavk::Stream::createCompute(const oavk::Device& inDevice) {
	return create(inDevice, inDevice.queues.computeQueueFamily, inDevice.queues.computeQueue);
}

void oavk::Stream::destroy(const oavk::Device& inDevice) {
	VkDevice dev = static_cast<VkDevice>(inDevice.device);
	if (submitted) {
		if (auto s = timelineSem.wait(inDevice, timelineValue); !s.isOk()) {
			OaLogError(oa::LogComponent::Runtime, "stream::destroy: timeline wait failed: {}", s.getMessage().cStr());
		}
	}
	for (void* pool : pendingPools) {
		deviceDispatch->vkDestroyDescriptorPool(dev, static_cast<VkDescriptorPool>(pool), nullptr);
	}
	pendingPools.clear();
	timelineSem.destroy(inDevice);
	if (commandPool) {
		// Explicitly free command buffer before destroying pool (vulkan validation requirement)
		if (commandBuffer) {
			VkCommandBuffer cb = static_cast<VkCommandBuffer>(commandBuffer);
			deviceDispatch->vkFreeCommandBuffers(dev, static_cast<VkCommandPool>(commandPool), 1, &cb);
			commandBuffer = nullptr;
		}
		deviceDispatch->vkDestroyCommandPool(dev, static_cast<VkCommandPool>(commandPool), nullptr);
		commandPool = nullptr;
	}
	deviceDispatch = nullptr;
}

oa::Status oavk::Stream::resetUnsubmitted(const oavk::Device&) {
	if (submitted) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"stream: cannot reset a submitted command buffer before completion");
	}
	if (!commandBuffer) {
		recording = false;
		return oa::Status::ok();
	}

	const VkResult result = deviceDispatch->vkResetCommandBuffer(
		static_cast<VkCommandBuffer>(commandBuffer), 0);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			"stream: vkResetCommandBuffer failed while cancelling unsubmitted work");
	}
	recording = false;
	return oa::Status::ok();
}

// ─── recording ─────────────────────────────────────────────────────────────────

oa::Status oavk::Stream::begin(const oavk::Device& inDevice) {
	VkDevice dev = static_cast<VkDevice>(inDevice.device);

	if (submitted) {
		OA_RETURN_IF_ERROR(timelineSem.wait(inDevice, timelineValue));
		submitted = false;
	}

	for (void* pool : pendingPools) {
		deviceDispatch->vkDestroyDescriptorPool(dev, static_cast<VkDescriptorPool>(pool), nullptr);
	}
	pendingPools.clear();

	VkResult r = deviceDispatch->vkResetCommandBuffer(
		static_cast<VkCommandBuffer>(commandBuffer), 0);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "stream: vkResetCommandBuffer failed");
	}

	VkCommandBufferBeginInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	r = deviceDispatch->vkBeginCommandBuffer(static_cast<VkCommandBuffer>(commandBuffer), &bi);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "stream: vkBeginCommandBuffer failed");
	}

	recording = true;
	return oa::Status::ok();
}

// Legacy descriptor allocation removed - bindless is now mandatory for performance.
// The per-dispatch descriptor pool allocation was a major CPU bottleneck (30-40% overhead).

// Heuristic guard against the single most common bindless-dispatch mistake:
// duplicating the auto-prepended buffer indices inside the host push struct.
//
// The bindless path prepends one u32 per buffer (its bindlessIndex) ahead of
// the caller's push (see the packing loops below). If the caller *also* placed
// those indices at the front of their C++ push struct — e.g. by stuffing
// heapSlot() values in to "match the shader" — the header doubles them and
// every scalar param after them is read from the wrong byte offset on the GPU.
// This silently corrupts results (it caused the GRU fused-kernel bug).
//
// Signature: the first numBufs u32s of the user push exactly equal the buffer
// bindless indices, in order, and the assembled push size disagrees with the
// shader's reflected declaration. Requires >=2 buffers. ALWAYS compiled via
// OA_VALIDATE_PUSH_NO_BUFFER_INDICES below (NOT gated on OA_VALIDATE): this
// catches silent result corruption, so it must run in the shipped binary too.
[[maybe_unused]] static bool pushDuplicatesBufferIndices(
	oa::Span<oavk::Buffer> inBufs,
	const void* inPush,
	oa::U32 inPushSize,
	oa::StringView inPipeline) noexcept
{
	const oa::U32 numBufs = static_cast<oa::U32>(inBufs.size());
	if (numBufs < 2 || inPush == nullptr || inPushSize < numBufs * sizeof(oa::U32)) {
		return false;
	}
	const oa::U32* words = static_cast<const oa::U32*>(inPush);
	for (oa::U32 i = 0; i < numBufs; ++i) {
		if (words[i] != inBufs[i].bindlessIndex) {
			return false;
		}
	}

	// Dimensions and other scalar parameters are small integers too. A valid
	// push can therefore begin with values that coincidentally equal the current
	// bindless slots (for example B=1,C=3). Disambiguate that case with the
	// shader's reflected push-block size: a correct call exactly fills the
	// declared block after the runtime prepends its index header, whereas a call
	// that duplicated the indices is larger by numBufs*sizeof(u32).
	const oa::String pipeline(inPipeline);
	const oa::U32 declared = oavk::spirvPushConstantBlockSizeByName(pipeline.cStr());
	const oa::U32 assembled = numBufs * sizeof(oa::U32) + inPushSize;
	return declared == 0U || assembled != declared;
}

// Convention reminder for the error below; keeps the message DRY across the
// three bindless packers. ALWAYS compiled (not OA_VALIDATE, which is debug-only per
// oa::Validation.md §4.1): this catches silent result corruption — the GRU fused-kernel
// bug class — so it must fire in the shipped release binary too. On a match we refuse
// the dispatch (return error) rather than let corrupt data through. Reflection
// distinguishes duplicated indices from ordinary scalar values that happen to equal
// the current bindless slots.
#define OA_VALIDATE_PUSH_NO_BUFFER_INDICES(bufs_, push_, pushSize_, pipeline_)          \
	do {                                                                               \
		if (pushDuplicatesBufferIndices(                                           \
				(bufs_), (push_), (pushSize_), (pipeline_))) {                            \
			OaLogError(oa::LogComponent::Runtime,                                         \
				"Dispatch '{}': host push constants begin with the buffer bindless "   \
				"indices. The bindless path auto-prepends those; do NOT put "          \
				"heapSlot()/buffer indices in the C++ push struct — pass scalar "      \
				"params only, after the indices. Refusing dispatch to avoid silent "   \
				"corruption.", oa::String(pipeline_).cStr());                              \
			return oa::Status::error(oa::StatusCode::InvalidArgument,                      \
				"bindless push duplicates auto-prepended buffer indices");             \
		}                                                                              \
	} while (0)

oa::Status oavk::Stream::recordDispatch(
	oa::Engine& inRt, oa::StringView inPipeline,
	oa::Span<oavk::Buffer> inBufs, const void* inPush, oa::U32 inPushSize,
	oa::ScalarType inStorageDtype,
	oa::U32 inGroupsX, oa::U32 inGroupsY, oa::U32 inGroupsZ)
{
	auto dtype = oavk::resolveStorageDtypeSpecConstant(inStorageDtype);
	if (not dtype.isOk()) return dtype.getStatus();
	oa::ComputeDispatchDesc desc;
	desc.kernel = inPipeline;
	desc.buffers = inBufs;
	desc.pushData = inPush;
	desc.pushSize = inPushSize;
	desc.dtype = dtype.getValue();
	desc.groupsX = inGroupsX;
	desc.groupsY = inGroupsY;
	desc.groupsZ = inGroupsZ;
	return recordDispatchDesc(inRt, desc);
}

oa::Status oavk::Stream::recordDispatchDesc(
	oa::Engine& inRt, const oa::ComputeDispatchDesc& inDesc)
{
	if (inDesc.dtype > 1U) {
		return oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"stream: dispatch descriptor storage dtype must be 0 or 1");
	}
	if (inDesc.indirect) {
		OA_RETURN_IF_ERROR(oavk::validateIndirectComputeDispatch(
			inDesc.indirectBuffer,
			inDesc.indirectOffset,
			oa::EngineAllocatorAccess::get(inRt).allocator));
	} else {
		OA_RETURN_IF_ERROR(oavk::validateDirectComputeDispatch(
			oa::EngineDeviceAccess::get(inRt), inDesc.groupsX, inDesc.groupsY, inDesc.groupsZ));
	}
	auto& pipeline = oa::EnginePipelineAccess::get(inRt).getPipeline(
		inDesc.kernel, inDesc.dtype);
	if (not pipeline.pipeline) {
		OaLogError(oa::LogComponent::Runtime,
			"recordDispatchDesc: pipeline not found: {} (dtype={})",
			inDesc.kernel,
			inDesc.dtype);
		return oa::Status::error("stream: pipeline not found: " + oa::String(inDesc.kernel));
	}
	if (not pipeline.bindless) {
		return oa::Status::error(oa::StatusCode::PipelineError,
			"stream: pipeline must use bindless (legacy descriptor path removed for performance)");
	}
	OA_RETURN_IF_ERROR(oavk::validateStorageBufferDescriptors(
		oa::EngineDeviceAccess::get(inRt),
		inDesc.buffers,
		true,
		oa::EngineAllocatorAccess::get(inRt).allocator));
	VkCommandBuffer cb = static_cast<VkCommandBuffer>(commandBuffer);
	deviceDispatch->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
		static_cast<VkPipeline>(pipeline.pipeline));
	VkDescriptorSet ds = static_cast<VkDescriptorSet>(
		oa::EngineBindlessAccess::get(inRt).descriptorSet);
	deviceDispatch->vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
		static_cast<VkPipelineLayout>(pipeline.pipelineLayout),
		0, 1, &ds, 0, nullptr);

	const oa::U32 numBufs = static_cast<oa::U32>(inDesc.buffers.size());
	if (not oavk::bindlessPushFits(numBufs, inDesc.pushSize)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"stream: bindless push exceeds oavk::OA_VK_MAX_PUSH_CONSTANT_BYTES "
			"(buffer index header + user push)");
	}
	OA_VALIDATE_PUSH_NO_BUFFER_INDICES(
		inDesc.buffers, inDesc.pushData, inDesc.pushSize, inDesc.kernel);
	const oa::U32 headerBytes = numBufs * sizeof(oa::U32);
	const oa::U32 totalPush = headerBytes + inDesc.pushSize;
	alignas(16) oa::U8 pushBuf[oavk::OA_VK_MAX_PUSH_CONSTANT_BYTES] = {};
	auto* indices = reinterpret_cast<oa::U32*>(pushBuf);
	for (oa::U32 i = 0; i < numBufs; ++i) {
		indices[i] = inDesc.buffers[i].bindlessIndex;
	}
	if (inDesc.pushData and inDesc.pushSize > 0) {
		oa::memcpy(pushBuf + headerBytes, inDesc.pushData, inDesc.pushSize);
	}
	deviceDispatch->vkCmdPushConstants(cb,
		static_cast<VkPipelineLayout>(pipeline.pipelineLayout),
		VK_SHADER_STAGE_COMPUTE_BIT, 0, totalPush, pushBuf);
	if (inDesc.indirect) {
		deviceDispatch->vkCmdDispatchIndirect(cb,
			static_cast<VkBuffer>(inDesc.indirectBuffer.buffer), inDesc.indirectOffset);
	} else {
		deviceDispatch->vkCmdDispatch(cb, inDesc.groupsX, inDesc.groupsY, inDesc.groupsZ);
	}
	return oa::Status::ok();
}

oa::Status oavk::Stream::recordDispatchIndirect(
	oa::Engine& inRt, oa::StringView inPipeline,
	oa::Span<oavk::Buffer> inBufs, const void* inPush, oa::U32 inPushSize,
	oa::ScalarType inStorageDtype,
	const oavk::Buffer& inIndirectBuffer, oa::U64 inOffset
) {
	auto dtype = oavk::resolveStorageDtypeSpecConstant(inStorageDtype);
	if (not dtype.isOk()) return dtype.getStatus();
	oa::ComputeDispatchDesc desc;
	desc.kernel = inPipeline;
	desc.buffers = inBufs;
	desc.pushData = inPush;
	desc.pushSize = inPushSize;
	desc.dtype = dtype.getValue();
	desc.indirectBuffer = inIndirectBuffer;
	desc.indirectOffset = inOffset;
	desc.indirect = true;
	return recordDispatchDesc(inRt, desc);
}

oa::Status oavk::Stream::record(
	oa::Engine& inRt, oa::StringView inPipeline,
	oa::Span<oavk::Buffer> inBufs, const void* inPush, oa::U32 inPushSize,
	oa::ScalarType inStorageDtype,
	oa::U32 inGroupsX, oa::U32 inGroupsY, oa::U32 inGroupsZ
) {
	OA_RETURN_IF_ERROR(recordDispatch(
		inRt, inPipeline, inBufs, inPush, inPushSize,
		inStorageDtype,
		inGroupsX, inGroupsY, inGroupsZ));
	recordBufferBarrier();
	return oa::Status::ok();
}

void oavk::Stream::recordCopyBuffer(const oavk::Buffer& inSrc, const oavk::Buffer& inDst, oa::U64 inSize) {
	const oavk::BufferCopyRegion region{.size = inSize};
	recordCopyBufferRegions(
		inSrc, inDst, oa::Span<const oavk::BufferCopyRegion>(&region, 1));
}

void oavk::Stream::recordCopyBufferRegions(
	const oavk::Buffer& inSrc,
	const oavk::Buffer& inDst,
	oa::Span<const oavk::BufferCopyRegion> inRegions)
{
	if (inRegions.empty()) return;
	oa::Vector<VkBufferCopy> regions;
	regions.reserve(inRegions.size());
	for (const oavk::BufferCopyRegion& region : inRegions) {
		if (region.size == 0) continue;
		regions.pushBack(VkBufferCopy{
			.srcOffset = region.srcOffset,
			.dstOffset = region.dstOffset,
			.size = region.size,
		});
	}
	if (regions.empty()) return;
	deviceDispatch->vkCmdCopyBuffer(
		static_cast<VkCommandBuffer>(commandBuffer),
		static_cast<VkBuffer>(inSrc.buffer), static_cast<VkBuffer>(inDst.buffer),
		static_cast<oa::U32>(regions.size()), regions.data());
}

void oavk::Stream::recordTransferReadBarrier(
	const oavk::Buffer& inSrc,
	oa::U64 inOffset,
	oa::U64 inSize)
{
	if (inSrc.aliasIdentity != nullptr) {
		VkMemoryBarrier2 aliasBarrier = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
			.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
		};
		VkDependencyInfo aliasDependency = {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.pNext = nullptr,
			.dependencyFlags = 0,
			.memoryBarrierCount = 1,
			.pMemoryBarriers = &aliasBarrier,
			.bufferMemoryBarrierCount = 0,
			.pBufferMemoryBarriers = nullptr,
			.imageMemoryBarrierCount = 0,
			.pImageMemoryBarriers = nullptr,
		};
		deviceDispatch->vkCmdPipelineBarrier2(
			static_cast<VkCommandBuffer>(commandBuffer), &aliasDependency);
		return;
	}

	VkBufferMemoryBarrier2 barrier = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
		.pNext = nullptr,
		.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
		.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer = static_cast<VkBuffer>(inSrc.buffer),
		.offset = inOffset,
		.size = inSize,
	};
	VkDependencyInfo dependency = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.pNext = nullptr,
		.dependencyFlags = 0,
		.memoryBarrierCount = 0,
		.pMemoryBarriers = nullptr,
		.bufferMemoryBarrierCount = 1,
		.pBufferMemoryBarriers = &barrier,
		.imageMemoryBarrierCount = 0,
		.pImageMemoryBarriers = nullptr,
	};
	deviceDispatch->vkCmdPipelineBarrier2(
		static_cast<VkCommandBuffer>(commandBuffer), &dependency);
}

void oavk::Stream::recordTransferWriteBarrier(
	const oavk::Buffer& inDst,
	oa::U64 inOffset,
	oa::U64 inSize)
{
	if (inDst.aliasIdentity != nullptr) {
		VkMemoryBarrier2 aliasBarrier = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
			.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
				| VK_PIPELINE_STAGE_2_HOST_BIT,
			.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT
				| VK_ACCESS_2_MEMORY_WRITE_BIT
				| VK_ACCESS_2_HOST_READ_BIT,
		};
		VkDependencyInfo aliasDependency = {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.pNext = nullptr,
			.dependencyFlags = 0,
			.memoryBarrierCount = 1,
			.pMemoryBarriers = &aliasBarrier,
			.bufferMemoryBarrierCount = 0,
			.pBufferMemoryBarriers = nullptr,
			.imageMemoryBarrierCount = 0,
			.pImageMemoryBarriers = nullptr,
		};
		deviceDispatch->vkCmdPipelineBarrier2(
			static_cast<VkCommandBuffer>(commandBuffer), &aliasDependency);
		return;
	}

	VkBufferMemoryBarrier2 barrier = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
		.pNext = nullptr,
		.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
		.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
			| VK_PIPELINE_STAGE_2_HOST_BIT,
		.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT
			| VK_ACCESS_2_MEMORY_WRITE_BIT
			| VK_ACCESS_2_HOST_READ_BIT,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer = static_cast<VkBuffer>(inDst.buffer),
		.offset = inOffset,
		.size = inSize,
	};

	VkDependencyInfo dependency = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.pNext = nullptr,
		.dependencyFlags = 0,
		.memoryBarrierCount = 0,
		.pMemoryBarriers = nullptr,
		.bufferMemoryBarrierCount = 1,
		.pBufferMemoryBarriers = &barrier,
		.imageMemoryBarrierCount = 0,
		.pImageMemoryBarriers = nullptr,
	};
	deviceDispatch->vkCmdPipelineBarrier2(
		static_cast<VkCommandBuffer>(commandBuffer), &dependency);
}

void oavk::Stream::recordBufferBarrier() {
	if (suppressAutoBarrier) return;
	VkMemoryBarrier2 barrier = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_HOST_BIT,
		.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_HOST_READ_BIT,
	};

	VkDependencyInfo dep = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.memoryBarrierCount = 1,
		.pMemoryBarriers = &barrier,
	};

	deviceDispatch->vkCmdPipelineBarrier2(
		static_cast<VkCommandBuffer>(commandBuffer), &dep);
}

void oavk::Stream::recordHostReadbackBarrier() {
	VkMemoryBarrier2 barrier = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
		.pNext = nullptr,
		.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
		.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
	};

	VkDependencyInfo dep = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.pNext = nullptr,
		.dependencyFlags = 0,
		.memoryBarrierCount = 1,
		.pMemoryBarriers = &barrier,
		.bufferMemoryBarrierCount = 0,
		.pBufferMemoryBarriers = nullptr,
		.imageMemoryBarrierCount = 0,
		.pImageMemoryBarriers = nullptr,
	};

	deviceDispatch->vkCmdPipelineBarrier2(
		static_cast<VkCommandBuffer>(commandBuffer), &dep);
}

void oavk::Stream::recordBufferMemoryBarriers(const oavk::Buffer* inBufs, oa::U32 inCount) {
	if (inCount == 0) return;

	constexpr oa::U32 kStackMax = 16;
	VkBufferMemoryBarrier2 stackBarriers[kStackMax];
	oa::Vector<VkBufferMemoryBarrier2> heapBarriers;
	VkBufferMemoryBarrier2* barriers = stackBarriers;
	if (inCount > kStackMax) {
		heapBarriers.resize(inCount);
		barriers = heapBarriers.data();
	}

	for (oa::U32 i = 0; i < inCount; ++i) {
		barriers[i] = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.buffer = static_cast<VkBuffer>(inBufs[i].buffer),
			.offset = 0,
			.size = VK_WHOLE_SIZE,
		};
	}

	VkDependencyInfo dep = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.bufferMemoryBarrierCount = inCount,
		.pBufferMemoryBarriers = barriers,
	};

	deviceDispatch->vkCmdPipelineBarrier2(
		static_cast<VkCommandBuffer>(commandBuffer), &dep);
}

// ─── Submission ────────────────────────────────────────────────────────────────
oa::Status oavk::Stream::submit(oa::Engine& inRt) {
	VkResult r = deviceDispatch->vkEndCommandBuffer(static_cast<VkCommandBuffer>(commandBuffer));
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "stream: vkEndCommandBuffer failed");
	}
	recording = false;

	++timelineValue;

	VkSemaphore sem = static_cast<VkSemaphore>(timelineSem.semaphore);
	VkTimelineSemaphoreSubmitInfo tsInfo = {
		.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
		.signalSemaphoreValueCount = 1,
		.pSignalSemaphoreValues = &timelineValue,
	};

	VkCommandBuffer cb = static_cast<VkCommandBuffer>(commandBuffer);
	VkSubmitInfo si = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = &tsInfo,
		.commandBufferCount = 1,
		.pCommandBuffers = &cb,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &sem,
	};

	OA_RETURN_IF_ERROR(oa::EngineAccess(inRt).submitToQueue(queue, &si, nullptr));

	submitted = true;
	return oa::Status::ok();
}

oa::Status oavk::Stream::submitWithDependency(
	oa::Engine& inRt,
	const oavk::TimelineSemaphore& inWaitSem,
	oa::U64 inWaitValue
) {
	const oavk::TimelineWait wait{inWaitSem.semaphore, inWaitValue};
	return submitWithDependencies(
		inRt,
		oa::Span<const oavk::TimelineWait>(&wait, 1));
}

oa::Status oavk::Stream::submitWithDependencies(
	oa::Engine& inRt,
	oa::Span<const oavk::TimelineWait> inWaits
) {
	oa::Vector<VkSemaphore> waitSemaphores;
	oa::Vector<oa::U64> waitValues;
	for (const oavk::TimelineWait& wait : inWaits) {
		if (wait.semaphore == nullptr || wait.value == 0) continue;
		const VkSemaphore semaphore = static_cast<VkSemaphore>(wait.semaphore);
		bool merged = false;
		for (oa::Usize i = 0; i < waitSemaphores.size(); ++i) {
			if (waitSemaphores[i] == semaphore) {
				if (wait.value > waitValues[i]) waitValues[i] = wait.value;
				merged = true;
				break;
			}
		}
		if (!merged) {
			waitSemaphores.pushBack(semaphore);
			waitValues.pushBack(wait.value);
		}
	}
	if (waitSemaphores.empty()) {
		return submit(inRt);
	}

	VkResult r = deviceDispatch->vkEndCommandBuffer(static_cast<VkCommandBuffer>(commandBuffer));
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "stream: vkEndCommandBuffer failed");
	}
	recording = false;

	++timelineValue;

	VkSemaphore signalSem = static_cast<VkSemaphore>(timelineSem.semaphore);

	VkCommandBuffer cb = static_cast<VkCommandBuffer>(commandBuffer);
	oa::Vector<VkSemaphoreSubmitInfo> waitInfos;
	for (oa::Usize waitIdx = 0; waitIdx < waitSemaphores.size(); ++waitIdx) {
		VkSemaphoreSubmitInfo waitInfo = {};
		waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
		waitInfo.semaphore = waitSemaphores[waitIdx];
		waitInfo.value = waitValues[waitIdx];
		waitInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		waitInfos.pushBack(waitInfo);
	}
	VkSemaphoreSubmitInfo signalInfo = {};
	signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
	signalInfo.semaphore = signalSem;
	signalInfo.value = timelineValue;
	signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	VkCommandBufferSubmitInfo commandInfo = {};
	commandInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
	commandInfo.commandBuffer = cb;
	VkSubmitInfo2 submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
	submitInfo.waitSemaphoreInfoCount = static_cast<oa::U32>(waitInfos.size());
	submitInfo.pWaitSemaphoreInfos = waitInfos.data();
	submitInfo.commandBufferInfoCount = 1;
	submitInfo.pCommandBufferInfos = &commandInfo;
	submitInfo.signalSemaphoreInfoCount = 1;
	submitInfo.pSignalSemaphoreInfos = &signalInfo;
	OA_RETURN_IF_ERROR(oa::EngineAccess(inRt).submitToQueue2(queue, &submitInfo));

	submitted = true;
	return oa::Status::ok();
}

oa::Status oavk::Stream::synchronize(const oavk::Device& inDevice) {
	if (!submitted) return oa::Status::ok();
	// A timeline counter query is only a non-blocking completion observation.
	// Establish the signal-to-host wait dependency explicitly even when the
	// counter has already advanced; the wait returns immediately in that case.
	auto status = timelineSem.wait(inDevice, timelineValue);
	if (status.isOk()) {
		submitted = false;
	}
	return status;
}

oa::Status oavk::Stream::submitAndWait(oa::Engine& inRt) {
	OA_RETURN_IF_ERROR(submit(inRt));
	return synchronize(oa::EngineDeviceAccess::get(inRt));
}

oa::Bool oavk::Stream::isComplete(const oavk::Device& inDevice) const {
	if (!submitted) return true;
	return timelineSem.getValue(inDevice) >= timelineValue;
}

// ─── Single-Shot ───────────────────────────────────────────────────────────────

oa::Status oavk::Stream::runOnce(
	oa::Engine& inRt, oa::StringView inPipeline,
	oa::Span<oavk::Buffer> inBufs, const void* inPush, oa::U32 inPushSize,
	oa::ScalarType inStorageDtype,
	oa::U32 inGroupsX, oa::U32 inGroupsY, oa::U32 inGroupsZ)
{
	auto dtype = oavk::resolveStorageDtypeSpecConstant(inStorageDtype);
	if (not dtype.isOk()) return dtype.getStatus();
	oavk::Stream* stream = oa::EngineSubmissionAccess::acquireStream(inRt);
	if (!stream) {
		return oa::Status::error(oa::StatusCode::VulkanError, "stream: failed to acquire from pool");
	}

	oa::Status status = stream->begin(oa::EngineDeviceAccess::get(inRt));
	if (not status.isOk()) {
		oa::EngineSubmissionAccess::releaseStream(inRt, stream);
		return status;
	}
	status = stream->record(
		inRt, inPipeline, inBufs, inPush, inPushSize,
		inStorageDtype,
		inGroupsX, inGroupsY, inGroupsZ);
	if (not status.isOk()) {
		const auto reset = stream->resetUnsubmitted(oa::EngineDeviceAccess::get(inRt));
		if (not reset.isOk()) {
			OaLogError(oa::LogComponent::Runtime,
				"stream: failed to cancel rejected one-shot dispatch: {}",
				reset.getMessage().cStr());
		}
		oa::EngineSubmissionAccess::releaseStream(inRt, stream);
		return status;
	}
	status = stream->submitAndWait(inRt);
	oa::EngineSubmissionAccess::releaseStream(inRt, stream);
	return status;
}
