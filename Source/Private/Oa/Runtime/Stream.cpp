#include <cstring>

#include <Oa/Runtime/Stream.h>
#include <Oa/Core/Validation.h>
#include <Oa/Runtime/Bindless.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Device.h>
#include <Oa/Runtime/Pipeline.h>
#include <Oa/Runtime/OaVk.h>

// ─── Move Semantics ────────────────────────────────────────────────────────────

OaVkStream::OaVkStream(OaVkStream&& InOther) noexcept
	: CommandPool(InOther.CommandPool)
	, CommandBuffer(InOther.CommandBuffer)
	, TimelineSem(std::move(InOther.TimelineSem))
	, TimelineValue(InOther.TimelineValue)
	, PendingPools(std::move(InOther.PendingPools))
	, Queue(InOther.Queue)
	, QueueFamily(InOther.QueueFamily)
	, Recording(InOther.Recording)
	, Submitted(InOther.Submitted)
	, SuppressAutoBarrier(InOther.SuppressAutoBarrier)
	, MeshNodeIndex(InOther.MeshNodeIndex)
{
	InOther.CommandPool = nullptr;
	InOther.CommandBuffer = nullptr;
	InOther.TimelineSem.Semaphore = nullptr;
	InOther.TimelineValue = 0;
	InOther.Queue = nullptr;
	InOther.Recording = false;
	InOther.Submitted = false;
	InOther.SuppressAutoBarrier = false;
	InOther.MeshNodeIndex = 0;
}

OaVkStream& OaVkStream::operator=(OaVkStream&& InOther) noexcept {
	if (this != &InOther) {
		CommandPool = InOther.CommandPool;
		CommandBuffer = InOther.CommandBuffer;
		TimelineSem = std::move(InOther.TimelineSem);
		TimelineValue = InOther.TimelineValue;
		PendingPools = std::move(InOther.PendingPools);
		Queue = InOther.Queue;
		QueueFamily = InOther.QueueFamily;
		Recording = InOther.Recording;
		Submitted = InOther.Submitted;
		SuppressAutoBarrier = InOther.SuppressAutoBarrier;
		MeshNodeIndex = InOther.MeshNodeIndex;
		InOther.CommandPool = nullptr;
		InOther.CommandBuffer = nullptr;
		InOther.TimelineSem.Semaphore = nullptr;
		InOther.TimelineValue = 0;
		InOther.Queue = nullptr;
		InOther.Recording = false;
		InOther.Submitted = false;
		InOther.SuppressAutoBarrier = false;
		InOther.MeshNodeIndex = 0;
	}
	return *this;
}

// ─── Lifecycle ─────────────────────────────────────────────────────────────────

OaResult<OaVkStream> OaVkStream::Create(
	const OaVkDevice& InDevice, OaU32 InQueueFamily, void* InQueue)
{
	VkDevice dev = static_cast<VkDevice>(InDevice.Device);

	VkCommandPoolCreateInfo cpCI{};
	cpCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cpCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	cpCI.queueFamilyIndex = InQueueFamily;

	VkCommandPool pool = VK_NULL_HANDLE;
	VkResult r = vkCreateCommandPool(dev, &cpCI, nullptr, &pool);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "stream: vkCreateCommandPool failed");
	}

	VkCommandBufferAllocateInfo cbAI{};
	cbAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cbAI.commandPool = pool;
	cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cbAI.commandBufferCount = 1;

	VkCommandBuffer cb = VK_NULL_HANDLE;
	r = vkAllocateCommandBuffers(dev, &cbAI, &cb);
	if (r != VK_SUCCESS) {
		vkDestroyCommandPool(dev, pool, nullptr);
		return OaStatus::Error(OaStatusCode::VulkanError, "stream: vkAllocateCommandBuffers failed");
	}

	auto semRes = OaVkTimelineSemaphore::Create(InDevice, 0);
	if (!semRes) {
		vkDestroyCommandPool(dev, pool, nullptr);
		return semRes.GetStatus();
	}

	OaVkStream s;
	s.CommandPool = pool;
	s.CommandBuffer = cb;
	s.TimelineSem = std::move(*semRes);
	s.TimelineValue = 0;
	s.Queue = InQueue;
	s.QueueFamily = InQueueFamily;
	return s;
}

OaResult<OaVkStream> OaVkStream::CreateCompute(const OaVkDevice& InDevice) {
	return Create(InDevice, InDevice.Queues.ComputeQueueFamily, InDevice.Queues.ComputeQueue);
}

void OaVkStream::Destroy(const OaVkDevice& InDevice) {
	VkDevice dev = static_cast<VkDevice>(InDevice.Device);
	if (Submitted) {
		if (auto s = TimelineSem.Wait(InDevice, TimelineValue); !s.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core, "Stream::Destroy: timeline wait failed: %s", s.GetMessage().c_str());
		}
	}
	for (void* pool : PendingPools) {
		vkDestroyDescriptorPool(dev, static_cast<VkDescriptorPool>(pool), nullptr);
	}
	PendingPools.Clear();
	TimelineSem.Destroy(InDevice);
	if (CommandPool) {
		// Explicitly free command buffer before destroying pool (Vulkan validation requirement)
		if (CommandBuffer) {
			VkCommandBuffer cb = static_cast<VkCommandBuffer>(CommandBuffer);
			vkFreeCommandBuffers(dev, static_cast<VkCommandPool>(CommandPool), 1, &cb);
			CommandBuffer = nullptr;
		}
		vkDestroyCommandPool(dev, static_cast<VkCommandPool>(CommandPool), nullptr);
		CommandPool = nullptr;
	}
}

// ─── Recording ─────────────────────────────────────────────────────────────────

OaStatus OaVkStream::Begin(const OaVkDevice& InDevice) {
	VkDevice dev = static_cast<VkDevice>(InDevice.Device);

	if (Submitted) {
		OA_RETURN_IF_ERROR(TimelineSem.Wait(InDevice, TimelineValue));
		Submitted = false;
	}

	for (void* pool : PendingPools) {
		vkDestroyDescriptorPool(dev, static_cast<VkDescriptorPool>(pool), nullptr);
	}
	PendingPools.Clear();

	VkResult r = vkResetCommandBuffer(
		static_cast<VkCommandBuffer>(CommandBuffer), 0);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "stream: vkResetCommandBuffer failed");
	}

	VkCommandBufferBeginInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	r = vkBeginCommandBuffer(static_cast<VkCommandBuffer>(CommandBuffer), &bi);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "stream: vkBeginCommandBuffer failed");
	}

	Recording = true;
	return OaStatus::Ok();
}

// Legacy descriptor allocation removed - bindless is now mandatory for performance.
// The per-dispatch descriptor pool allocation was a major CPU bottleneck (30-40% overhead).

// Heuristic guard against the single most common bindless-dispatch mistake:
// duplicating the auto-prepended buffer indices inside the host push struct.
//
// The bindless path prepends one u32 per buffer (its BindlessIndex) ahead of
// the caller's push (see the packing loops below). If the caller *also* placed
// those indices at the front of their C++ push struct — e.g. by stuffing
// HeapSlot() values in to "match the shader" — the header doubles them and
// every scalar param after them is read from the wrong byte offset on the GPU.
// This silently corrupts results (it caused the GRU fused-kernel bug).
//
// Signature: the first numBufs u32s of the user push exactly equal the buffer
// bindless indices, in order, and the assembled push size disagrees with the
// shader's reflected declaration. Requires >=2 buffers. ALWAYS compiled via
// OA_VALIDATE_PUSH_NO_BUFFER_INDICES below (NOT gated on OA_VALIDATE): this
// catches silent result corruption, so it must run in the shipped binary too.
[[maybe_unused]] static bool OaVkPushDuplicatesBufferIndices(
	OaSpan<OaVkBuffer> InBufs,
	const void* InPush,
	OaU32 InPushSize,
	OaStringView InPipeline) noexcept
{
	const OaU32 numBufs = static_cast<OaU32>(InBufs.size());
	if (numBufs < 2 || InPush == nullptr || InPushSize < numBufs * sizeof(OaU32)) {
		return false;
	}
	const OaU32* words = static_cast<const OaU32*>(InPush);
	for (OaU32 i = 0; i < numBufs; ++i) {
		if (words[i] != InBufs[i].BindlessIndex) {
			return false;
		}
	}

	// Dimensions and other scalar parameters are small integers too. A valid
	// push can therefore begin with values that coincidentally equal the current
	// bindless slots (for example B=1,C=3). Disambiguate that case with the
	// shader's reflected push-block size: a correct call exactly fills the
	// declared block after the runtime prepends its index header, whereas a call
	// that duplicated the indices is larger by numBufs*sizeof(u32).
	const OaString pipeline(InPipeline);
	const OaU32 declared = OaSpvPushConstantBlockSizeByName(pipeline.c_str());
	const OaU32 assembled = numBufs * sizeof(OaU32) + InPushSize;
	return declared == 0U || assembled != declared;
}

// Convention reminder for the error below; keeps the message DRY across the
// three bindless packers. ALWAYS compiled (not OA_VALIDATE, which is debug-only per
// OaValidation.md §4.1): this catches silent result corruption — the GRU fused-kernel
// bug class — so it must fire in the shipped Release binary too. On a match we refuse
// the dispatch (return error) rather than let corrupt data through. Reflection
// distinguishes duplicated indices from ordinary scalar values that happen to equal
// the current bindless slots.
#define OA_VALIDATE_PUSH_NO_BUFFER_INDICES(bufs_, push_, pushSize_, pipeline_)          \
	do {                                                                               \
		if (OaVkPushDuplicatesBufferIndices(                                           \
				(bufs_), (push_), (pushSize_), (pipeline_))) {                            \
			OA_LOG_ERROR(OaLogComponent::Core,                                         \
				"Dispatch '%s': host push constants begin with the buffer bindless "   \
				"indices. The bindless path auto-prepends those; do NOT put "          \
				"HeapSlot()/buffer indices in the C++ push struct — pass scalar "      \
				"params only, after the indices. Refusing dispatch to avoid silent "   \
				"corruption.", OaString(pipeline_).c_str());                              \
			return OaStatus::Error(OaStatusCode::InvalidArgument,                      \
				"bindless push duplicates auto-prepended buffer indices");             \
		}                                                                              \
	} while (0)

OaStatus OaVkStream::RecordDispatch(
	OaComputeEngine& InRt, OaStringView InPipeline,
	OaSpan<OaVkBuffer> InBufs, const void* InPush, OaU32 InPushSize,
	OaU32 InGroupsX, OaU32 InGroupsY, OaU32 InGroupsZ)
{
	OaComputeDispatchDesc desc;
	desc.Kernel = InPipeline;
	desc.Buffers = InBufs;
	desc.PushData = InPush;
	desc.PushSize = InPushSize;
	desc.Dtype = InRt.DtypeSpecConstant();
	desc.GroupsX = InGroupsX;
	desc.GroupsY = InGroupsY;
	desc.GroupsZ = InGroupsZ;
	return RecordDispatchDesc(InRt, desc);
}

OaStatus OaVkStream::RecordDispatchDesc(
	OaComputeEngine& InRt, const OaComputeDispatchDesc& InDesc)
{
	auto* target = InDesc.NodeIndex == 0 ? nullptr : InRt.GetNode(InDesc.NodeIndex);
	if (InDesc.NodeIndex != 0 and target == nullptr) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"stream: dispatch descriptor has an invalid node index");
	}
	auto& pipeline = target
		? target->Pipelines.GetPipeline(InDesc.Kernel, InDesc.Dtype)
		: InRt.Pipelines.GetPipeline(InDesc.Kernel, InDesc.Dtype);
	if (not pipeline.Pipeline) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"RecordDispatchDesc: pipeline not found: %.*s (dtype=%u node=%u)",
			static_cast<int>(InDesc.Kernel.Size()), InDesc.Kernel.Data(),
			InDesc.Dtype, InDesc.NodeIndex);
		return OaStatus::Error("stream: pipeline not found: " + OaString(InDesc.Kernel));
	}
	if (not pipeline.Bindless) {
		return OaStatus::Error(OaStatusCode::PipelineError,
			"stream: pipeline must use bindless (legacy descriptor path removed for performance)");
	}
	for (OaU32 i = 0; i < static_cast<OaU32>(InDesc.Buffers.Size()); ++i) {
		if (InDesc.Buffers[i].NodeIndex != InDesc.NodeIndex) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"stream: dispatch buffer node does not match descriptor node");
		}
	}
	if (InDesc.Indirect) {
		if (not InDesc.IndirectBuffer.Buffer
			or (InDesc.IndirectOffset & 3ULL) != 0
			or InDesc.IndirectOffset + 3ULL * sizeof(OaU32) > InDesc.IndirectBuffer.Size)
		{
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"stream: invalid indirect dispatch buffer or offset");
		}
	}

	VkCommandBuffer cb = static_cast<VkCommandBuffer>(CommandBuffer);
	vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
		static_cast<VkPipeline>(pipeline.Pipeline));
	auto& bindless = target ? target->Bindless : InRt.Bindless;
	VkDescriptorSet ds = static_cast<VkDescriptorSet>(bindless.DescriptorSet);
	vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
		static_cast<VkPipelineLayout>(pipeline.PipelineLayout),
		0, 1, &ds, 0, nullptr);

	const OaU32 numBufs = static_cast<OaU32>(InDesc.Buffers.Size());
	if (not OaVkBindlessPushFits(numBufs, InDesc.PushSize)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"stream: bindless push exceeds OA_VK_MAX_PUSH_CONSTANT_BYTES "
			"(buffer index header + user push)");
	}
	OA_VALIDATE_PUSH_NO_BUFFER_INDICES(
		InDesc.Buffers, InDesc.PushData, InDesc.PushSize, InDesc.Kernel);
	const OaU32 headerBytes = numBufs * sizeof(OaU32);
	const OaU32 totalPush = headerBytes + InDesc.PushSize;
	alignas(16) OaU8 pushBuf[OA_VK_MAX_PUSH_CONSTANT_BYTES] = {};
	auto* indices = reinterpret_cast<OaU32*>(pushBuf);
	for (OaU32 i = 0; i < numBufs; ++i) {
		indices[i] = InDesc.Buffers[i].BindlessIndex;
	}
	if (InDesc.PushData and InDesc.PushSize > 0) {
		std::memcpy(pushBuf + headerBytes, InDesc.PushData, InDesc.PushSize);
	}
	vkCmdPushConstants(cb,
		static_cast<VkPipelineLayout>(pipeline.PipelineLayout),
		VK_SHADER_STAGE_COMPUTE_BIT, 0, totalPush, pushBuf);
	if (InDesc.Indirect) {
		vkCmdDispatchIndirect(cb,
			static_cast<VkBuffer>(InDesc.IndirectBuffer.Buffer), InDesc.IndirectOffset);
	} else {
		vkCmdDispatch(cb, InDesc.GroupsX, InDesc.GroupsY, InDesc.GroupsZ);
	}
	return OaStatus::Ok();
}

OaStatus OaVkStream::RecordDispatchOnNode(
	OaComputeEngine& InRt, OaU32 InNodeIndex,
	OaStringView InPipeline,
	OaSpan<OaVkBuffer> InBufs, const void* InPush, OaU32 InPushSize,
	OaU32 InGroupsX, OaU32 InGroupsY, OaU32 InGroupsZ
){
	OaComputeDispatchDesc desc;
	desc.Kernel = InPipeline;
	desc.Buffers = InBufs;
	desc.PushData = InPush;
	desc.PushSize = InPushSize;
	desc.Dtype = InRt.DtypeSpecConstant();
	desc.GroupsX = InGroupsX;
	desc.GroupsY = InGroupsY;
	desc.GroupsZ = InGroupsZ;
	desc.NodeIndex = InNodeIndex;
	return RecordDispatchDesc(InRt, desc);
}

OaStatus OaVkStream::RecordDispatchIndirect(
	OaComputeEngine& InRt, OaStringView InPipeline,
	OaSpan<OaVkBuffer> InBufs, const void* InPush, OaU32 InPushSize,
	const OaVkBuffer& InIndirectBuffer, OaU64 InOffset
) {
	OaComputeDispatchDesc desc;
	desc.Kernel = InPipeline;
	desc.Buffers = InBufs;
	desc.PushData = InPush;
	desc.PushSize = InPushSize;
	desc.Dtype = InRt.DtypeSpecConstant();
	desc.IndirectBuffer = InIndirectBuffer;
	desc.IndirectOffset = InOffset;
	desc.Indirect = true;
	return RecordDispatchDesc(InRt, desc);
}

OaStatus OaVkStream::Record(
	OaComputeEngine& InRt, OaStringView InPipeline,
	OaSpan<OaVkBuffer> InBufs, const void* InPush, OaU32 InPushSize,
	OaU32 InGroupsX, OaU32 InGroupsY, OaU32 InGroupsZ
) {
	OA_RETURN_IF_ERROR(RecordDispatch(
		InRt, InPipeline, InBufs, InPush, InPushSize,
		InGroupsX, InGroupsY, InGroupsZ));
	RecordBufferBarrier();
	return OaStatus::Ok();
}

void OaVkStream::RecordCopyBuffer(const OaVkBuffer& InSrc, const OaVkBuffer& InDst, OaU64 InSize) {
	VkBufferCopy region = {
		.size = InSize,
	};

	vkCmdCopyBuffer(
		static_cast<VkCommandBuffer>(CommandBuffer),
		static_cast<VkBuffer>(InSrc.Buffer), static_cast<VkBuffer>(InDst.Buffer),
		1, &region
	);
}

void OaVkStream::RecordBufferBarrier() {
	if (SuppressAutoBarrier) return;
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

	vkCmdPipelineBarrier2(
		static_cast<VkCommandBuffer>(CommandBuffer), &dep);
}

void OaVkStream::RecordHostReadbackBarrier() {
	VkMemoryBarrier2 barrier = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
		.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
	};

	VkDependencyInfo dep = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.memoryBarrierCount = 1,
		.pMemoryBarriers = &barrier,
	};

	vkCmdPipelineBarrier2(
		static_cast<VkCommandBuffer>(CommandBuffer), &dep);
}

void OaVkStream::RecordBufferMemoryBarriers(const OaVkBuffer* InBufs, OaU32 InCount) {
	if (InCount == 0) return;

	constexpr OaU32 kStackMax = 16;
	VkBufferMemoryBarrier2 stackBarriers[kStackMax];
	OaVec<VkBufferMemoryBarrier2> heapBarriers;
	VkBufferMemoryBarrier2* barriers = stackBarriers;
	if (InCount > kStackMax) {
		heapBarriers.resize(InCount);
		barriers = heapBarriers.data();
	}

	for (OaU32 i = 0; i < InCount; ++i) {
		barriers[i] = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.buffer = static_cast<VkBuffer>(InBufs[i].Buffer),
			.offset = 0,
			.size = VK_WHOLE_SIZE,
		};
	}

	VkDependencyInfo dep = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.bufferMemoryBarrierCount = InCount,
		.pBufferMemoryBarriers = barriers,
	};

	vkCmdPipelineBarrier2(
		static_cast<VkCommandBuffer>(CommandBuffer), &dep);
}

// ─── Submission ────────────────────────────────────────────────────────────────
OaStatus OaVkStream::Submit(OaComputeEngine& InRt, OaBool InDispatchAlreadyLoadedForNode) {
	VkResult r = vkEndCommandBuffer(static_cast<VkCommandBuffer>(CommandBuffer));
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "stream: vkEndCommandBuffer failed");
	}
	Recording = false;

	++TimelineValue;

	VkSemaphore sem = static_cast<VkSemaphore>(TimelineSem.Semaphore);
	VkTimelineSemaphoreSubmitInfo tsInfo = {
		.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
		.signalSemaphoreValueCount = 1,
		.pSignalSemaphoreValues = &TimelineValue,
	};

	VkCommandBuffer cb = static_cast<VkCommandBuffer>(CommandBuffer);
	VkSubmitInfo si = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = &tsInfo,
		.commandBufferCount = 1,
		.pCommandBuffers = &cb,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &sem,
	};

	if (MeshNodeIndex != 0 && InRt.IsMultiDevice()) {
		OA_RETURN_IF_ERROR(InRt.SubmitToNodeQueue(MeshNodeIndex, Queue, &si, nullptr,
			InDispatchAlreadyLoadedForNode));
	} else {
		OA_RETURN_IF_ERROR(InRt.SubmitToQueue(Queue, &si, nullptr));
	}

	Submitted = true;
	return OaStatus::Ok();
}

OaStatus OaVkStream::SubmitWithDependency(
	OaComputeEngine& InRt,
	const OaVkTimelineSemaphore& InWaitSem,
	OaU64 InWaitValue,
	OaBool InDispatchAlreadyLoadedForNode
) {
	const OaVkTimelineWait wait{&InWaitSem, InWaitValue};
	return SubmitWithDependencies(
		InRt,
		OaSpan<const OaVkTimelineWait>(&wait, 1),
		InDispatchAlreadyLoadedForNode);
}

OaStatus OaVkStream::SubmitWithDependencies(
	OaComputeEngine& InRt,
	OaSpan<const OaVkTimelineWait> InWaits,
	OaBool InDispatchAlreadyLoadedForNode
) {
	OaVec<VkSemaphore> waitSemaphores;
	OaVec<OaU64> waitValues;
	OaVec<VkPipelineStageFlags> waitStages;
	for (const OaVkTimelineWait& wait : InWaits) {
		if (wait.Semaphore == nullptr || wait.Semaphore->Semaphore == nullptr
			|| wait.Value == 0) continue;
		const VkSemaphore semaphore = static_cast<VkSemaphore>(wait.Semaphore->Semaphore);
		bool merged = false;
		for (OaUsize i = 0; i < waitSemaphores.Size(); ++i) {
			if (waitSemaphores[i] == semaphore) {
				if (wait.Value > waitValues[i]) waitValues[i] = wait.Value;
				merged = true;
				break;
			}
		}
		if (!merged) {
			waitSemaphores.PushBack(semaphore);
			waitValues.PushBack(wait.Value);
			waitStages.PushBack(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
		}
	}
	if (waitSemaphores.Empty()) {
		return Submit(InRt, InDispatchAlreadyLoadedForNode);
	}

	VkResult r = vkEndCommandBuffer(static_cast<VkCommandBuffer>(CommandBuffer));
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "stream: vkEndCommandBuffer failed");
	}
	Recording = false;

	++TimelineValue;

	VkSemaphore signalSem = static_cast<VkSemaphore>(TimelineSem.Semaphore);

	VkTimelineSemaphoreSubmitInfo tsInfo = {
		.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
		.waitSemaphoreValueCount = static_cast<OaU32>(waitValues.Size()),
		.pWaitSemaphoreValues = waitValues.Data(),
		.signalSemaphoreValueCount = 1,
		.pSignalSemaphoreValues = &TimelineValue,
	};

	VkCommandBuffer cb = static_cast<VkCommandBuffer>(CommandBuffer);
	VkSubmitInfo si = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = &tsInfo,
		.waitSemaphoreCount = static_cast<OaU32>(waitSemaphores.Size()),
		.pWaitSemaphores = waitSemaphores.Data(),
		.pWaitDstStageMask = waitStages.Data(),
		.commandBufferCount = 1,
		.pCommandBuffers = &cb,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &signalSem,
	};

	if (MeshNodeIndex != 0 && InRt.IsMultiDevice()) {
		OA_RETURN_IF_ERROR(InRt.SubmitToNodeQueue(MeshNodeIndex, Queue, &si, nullptr,
			InDispatchAlreadyLoadedForNode));
	} else {
		OA_RETURN_IF_ERROR(InRt.SubmitToQueue(Queue, &si, nullptr));
	}

	Submitted = true;
	return OaStatus::Ok();
}

OaStatus OaVkStream::Synchronize(const OaVkDevice& InDevice) {
	if (!Submitted) return OaStatus::Ok();
	// Fast path: poll once before blocking. For small GPU workloads the
	// GPU may have already completed — skip the vkWaitSemaphores syscall.
	if (TimelineSem.GetValue(InDevice) >= TimelineValue) {
		Submitted = false;
		return OaStatus::Ok();
	}
	auto status = TimelineSem.Wait(InDevice, TimelineValue);
	if (status.IsOk()) {
		Submitted = false;
	}
	return status;
}

OaStatus OaVkStream::SubmitAndWait(OaComputeEngine& InRt, OaBool InDispatchAlreadyLoadedForNode) {
	OA_RETURN_IF_ERROR(Submit(InRt, InDispatchAlreadyLoadedForNode));
	const OaVkDevice* syncDev = &InRt.Device;
	if (MeshNodeIndex != 0 && InRt.IsMultiDevice()) {
		if (auto* n = InRt.GetNode(MeshNodeIndex)) {
			syncDev = &n->Device;
		}
	}
	return Synchronize(*syncDev);
}

OaBool OaVkStream::IsComplete(const OaVkDevice& InDevice) const {
	if (!Submitted) return true;
	return TimelineSem.GetValue(InDevice) >= TimelineValue;
}

// ─── Single-Shot ───────────────────────────────────────────────────────────────

OaStatus OaVkStream::RunOnce(
	OaComputeEngine& InRt, OaStringView InPipeline,
	OaSpan<OaVkBuffer> InBufs, const void* InPush, OaU32 InPushSize,
	OaU32 InGroupsX, OaU32 InGroupsY, OaU32 InGroupsZ)
{
	OaVkStream* stream = InRt.AcquireStream();
	if (!stream) {
		return OaStatus::Error(OaStatusCode::VulkanError, "stream: failed to acquire from pool");
	}

	OA_RETURN_IF_ERROR(stream->Begin(InRt.Device));
	OA_RETURN_IF_ERROR(stream->Record(
		InRt, InPipeline, InBufs, InPush, InPushSize,
		InGroupsX, InGroupsY, InGroupsZ));
	OaStatus status = stream->SubmitAndWait(InRt);
	InRt.ReleaseStream(stream);
	return status;
}
