#include <oa/runtime/executableGraph.h>
#include <oa/core/jsonWriter.h>
#include <oa/runtime/eventAccess.h>
#include "engine/deviceAccess.h"
#include "engine/engineAccess.h"
#include <oa/core/log.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/resourceAccess.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/bindless.h>
#include <oa/runtime/stream.h>
#include <oa/runtime/pipeline.h>
#include <oa/runtime/oaVk.h>
#include <oa/runtime/engine/bindlessAccess.h>
#include <oa/runtime/engine/deviceAccess.h>
#include <oa/runtime/engine/pipelineAccess.h>
#include <oa/runtime/engine/submissionAccess.h>
#include <oa/core/envFlag.h>
#include <oa/core/matrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/core/validation.h>
#include "descriptorValidation.h"
#include "dispatchValidation.h"

struct GraphBufferState {
	VkPipelineStageFlags2 stageMask = 0;
	VkAccessFlags2 accessMask = 0;
	bool writes = false;
	void* buffer = nullptr;
	oa::QueueHint queue = oa::QueueHint::Compute;
	oa::Bool mixedQueues = false;
	oa::U32 firstNode = 0;
	oa::U32 lastNode = 0;
};

enum class GraphHazard : oa::U8 {
	Raw,
	War,
	Waw,
};

struct GraphBarrierDebug {
	GraphHazard hazard = GraphHazard::Raw;
	oa::U32 sourceFirstNode = 0;
	oa::U32 sourceLastNode = 0;
	oa::U32 destinationNode = 0;
	void* sourceBuffer = nullptr;
	void* destinationBuffer = nullptr;
	void* hazardIdentity = nullptr;
	oa::U64 bytes = 0;
	oa::QueueHint sourceQueue = oa::QueueHint::Compute;
	oa::Bool sourceQueuesMixed = false;
	oa::QueueHint destinationQueue = oa::QueueHint::Compute;
	VkPipelineStageFlags2 sourceStages = 0;
	VkAccessFlags2 sourceAccess = 0;
	VkPipelineStageFlags2 destinationStages = 0;
	VkAccessFlags2 destinationAccess = 0;
	oa::Bool alias = false;
};

static oa::U32 computeNodeBarriers(
	const oa::ComputeNode& inNode,
	oa::U32 inNodeIndex,
	const oa::HashMap<void*, GraphBufferState>& inState,
	oa::Vec<VkBufferMemoryBarrier2>& outBufferBarriers,
	oa::Vec<VkMemoryBarrier2>& outAliasBarriers,
	oa::Vec<GraphBarrierDebug>* outBufferDebug = nullptr,
	oa::Vec<GraphBarrierDebug>* outAliasDebug = nullptr
);
static oa::U32 pruneRedundantWarBarriers(
	oa::Vec<VkBufferMemoryBarrier2>& inOutBarriers,
	oa::Vec<GraphBarrierDebug>* inOutDebug = nullptr
);
static void updateBufferStates(
	const oa::ComputeNode& inNode,
	oa::U32 inNodeIndex,
	oa::HashMap<void*, GraphBufferState>& inOutState
);
static void recordFinalBarrier(
	const OaVkDeviceTable& inDispatch,
	VkCommandBuffer inCb,
	oa::Bool inRequired);
static oa::Status allocGraphDescriptorSet(
	const oavk::Device& inDevice,
	oa::ComputePipeline& inPipeline,
	oa::Span<oavk::Buffer> inBuffers,
	void** outPool,
	void** outSet
);
static oa::ComputeNode makeComputeNode(const oa::ComputeDispatchDesc& inDesc);
static oa::ComputeDispatchDesc makeDispatchDesc(oa::ComputeNode& inNode);

static const char* bufferAccessName(oa::BufferAccess inAccess) {
	switch (inAccess) {
	case oa::BufferAccess::Read: return "read";
	case oa::BufferAccess::Write: return "write";
	case oa::BufferAccess::ReadWrite: return "read_write";
	}
	return "unknown";
}

static const char* queueHintName(oa::QueueHint inQueue) {
	switch (inQueue) {
	case oa::QueueHint::Compute: return "compute";
	case oa::QueueHint::AsyncCompute: return "async_compute";
	case oa::QueueHint::Transfer: return "transfer";
	}
	return "unknown";
}

static const char* kernelSelectionName(oa::KernelSelectionKind inSelection) {
	switch (inSelection) {
	case oa::KernelSelectionKind::Unspecified: return "unspecified";
	case oa::KernelSelectionKind::Direct: return "direct";
	case oa::KernelSelectionKind::PrecisionFallback: return "precision_fallback";
	case oa::KernelSelectionKind::LayoutFallback: return "layout_fallback";
	case oa::KernelSelectionKind::NaiveFallback: return "naive_fallback";
	}
	return "unknown";
}

static const char* hazardName(GraphHazard inHazard) {
	switch (inHazard) {
	case GraphHazard::Raw: return "read_after_write";
	case GraphHazard::War: return "write_after_read";
	case GraphHazard::Waw: return "write_after_write";
	}
	return "unknown";
}

static void writeJsonString(oa::internal::JsonWriter& out, oa::StringView inValue) {
	out.writeString(inValue);
}

static void writeHexId(oa::internal::JsonWriter& out, oa::U64 inValue) {
	out.writeHexId(inValue);
}

static void writeStageNames(oa::internal::JsonWriter& out, VkPipelineStageFlags2 inMask) {
	out << '[';
	bool separator = false;
	auto write = [&](VkPipelineStageFlags2 bit, const char* name) {
		if ((inMask & bit) == 0) return;
		if (separator) out << ", ";
		writeJsonString(out, name);
		separator = true;
	};
	write(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, "compute_shader");
	write(VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, "draw_indirect");
	write(VK_PIPELINE_STAGE_2_HOST_BIT, "host");
	out << ']';
}

static void writeAccessNames(oa::internal::JsonWriter& out, VkAccessFlags2 inMask) {
	out << '[';
	bool separator = false;
	auto write = [&](VkAccessFlags2 bit, const char* name) {
		if ((inMask & bit) == 0) return;
		if (separator) out << ", ";
		writeJsonString(out, name);
		separator = true;
	};
	write(VK_ACCESS_2_SHADER_STORAGE_READ_BIT, "shader_storage_read");
	write(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, "shader_storage_write");
	write(VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, "indirect_command_read");
	write(VK_ACCESS_2_HOST_READ_BIT, "host_read");
	out << ']';
}

struct GraphDebugHashes {
	oa::U64 topology = 14695981039346656037ULL;
	oa::U64 resources = 14695981039346656037ULL;
	oa::U64 push = 14695981039346656037ULL;
};

static GraphDebugHashes executableGraphDebugHashes(
	oa::Span<const oa::ComputeNode> inNodes,
	oa::Bool inHostReadbackRequired)
{
	GraphDebugHashes result;
	const auto append = [](oa::U64& inOutHash, const void* inData, oa::U64 inBytes) {
		const auto* bytes = static_cast<const oa::U8*>(inData);
		for (oa::U64 i = 0; i < inBytes; ++i) {
			inOutHash ^= bytes[i];
			inOutHash *= 1099511628211ULL;
		}
	};
	const oa::U32 count = static_cast<oa::U32>(inNodes.size());
	append(result.topology, &count, sizeof(count));
	append(result.topology, &inHostReadbackRequired, sizeof(inHostReadbackRequired));
	for (const auto& node : inNodes) {
		append(result.topology, node.operation.data(), node.operation.size());
		const oa::U32 semanticOpCount =
			static_cast<oa::U32>(node.semanticOps.size());
		append(result.topology, &semanticOpCount,
			sizeof(semanticOpCount));
		for (const auto semanticOp : node.semanticOps) {
			append(result.topology, &semanticOp, sizeof(semanticOp));
		}
		append(result.topology, &node.implementationId, sizeof(node.implementationId));
		append(result.topology, &node.opContractHash,
			sizeof(node.opContractHash));
		append(result.topology, &node.problemContractHash,
			sizeof(node.problemContractHash));
		append(result.topology, &node.kernelContentHash, sizeof(node.kernelContentHash));
		append(result.topology, &node.kernelSelection, sizeof(node.kernelSelection));
		append(result.topology, node.shader.data(), node.shader.size());
		append(result.topology, &node.dtype, sizeof(node.dtype));
		append(result.topology, &node.groupsX, sizeof(node.groupsX));
		append(result.topology, &node.groupsY, sizeof(node.groupsY));
		append(result.topology, &node.groupsZ, sizeof(node.groupsZ));
		append(result.topology, &node.indirect, sizeof(node.indirect));
		if (node.indirect) {
			append(result.resources, &node.indirectBuffer.buffer,
				sizeof(node.indirectBuffer.buffer));
			append(result.resources, &node.indirectBuffer.bindlessIndex,
				sizeof(node.indirectBuffer.bindlessIndex));
			append(result.resources, &node.indirectBuffer.size,
				sizeof(node.indirectBuffer.size));
			append(result.resources, &node.indirectBuffer.flags,
				sizeof(node.indirectBuffer.flags));
			append(result.resources, &node.indirectBuffer.allocatorIdentity,
				sizeof(node.indirectBuffer.allocatorIdentity));
			append(result.topology, &node.indirectOffset,
				sizeof(node.indirectOffset));
		}
		append(result.topology, &node.queue, sizeof(node.queue));
		const oa::U32 bufferCount = static_cast<oa::U32>(node.buffers.size());
		append(result.topology, &bufferCount, sizeof(bufferCount));
		for (oa::U32 i = 0; i < bufferCount; ++i) {
			append(result.topology, &node.access[i], sizeof(node.access[i]));
			append(result.resources, &node.buffers[i].buffer, sizeof(node.buffers[i].buffer));
			append(result.resources, &node.buffers[i].bindlessIndex, sizeof(node.buffers[i].bindlessIndex));
			append(result.resources, &node.buffers[i].size, sizeof(node.buffers[i].size));
		}
		append(result.push, &node.pushSize, sizeof(node.pushSize));
		append(result.push, node.pushData, node.pushSize);
	}
	return result;
}

void oa::ExecutableGraph::add(const oa::ComputeDispatchDesc& inDesc) {
	if (inDesc.access.size() != inDesc.buffers.size()) {
		OaLogError(oa::LogComponent::Compute,
			"oa::ExecutableGraph::add '%.*s': access=%zu buffers=%zu",
			static_cast<int>(inDesc.kernel.size()), inDesc.kernel.data(),
			inDesc.access.size(), inDesc.buffers.size());
		return;
	}
	if (not inDesc.bufferOwners.empty()
		and inDesc.bufferOwners.size() != inDesc.buffers.size())
	{
		OaLogError(oa::LogComponent::Compute,
			"oa::ExecutableGraph::add '%.*s': owners=%zu buffers=%zu",
			static_cast<int>(inDesc.kernel.size()), inDesc.kernel.data(),
			inDesc.bufferOwners.size(), inDesc.buffers.size());
		return;
	}
	if (inDesc.pushSize > oavk::OA_VK_MAX_PUSH_CONSTANT_BYTES
		or (inDesc.pushSize != 0U and inDesc.pushData == nullptr))
	{
		OaLogError(oa::LogComponent::Compute,
			"oa::ExecutableGraph::add '%.*s': invalid push payload size=%u",
			static_cast<int>(inDesc.kernel.size()), inDesc.kernel.data(),
			inDesc.pushSize);
		return;
	}
	for (oa::U32 operation = 0;
		operation < inDesc.semanticOps.size(); ++operation)
	{
		if (inDesc.semanticOps[operation]
			== oa::invalidSemanticOpId)
		{
			OaLogError(oa::LogComponent::Compute,
				"oa::ExecutableGraph::add '%.*s': invalid semantic provenance",
				static_cast<int>(inDesc.kernel.size()), inDesc.kernel.data());
			return;
		}
		for (oa::U32 previous = 0; previous < operation; ++previous) {
			if (inDesc.semanticOps[previous]
				!= inDesc.semanticOps[operation])
			{
				continue;
			}
			OaLogError(oa::LogComponent::Compute,
				"oa::ExecutableGraph::add '%.*s': duplicate semantic provenance",
				static_cast<int>(inDesc.kernel.size()), inDesc.kernel.data());
			return;
		}
	}
	nodes_.pushBack(makeComputeNode(inDesc));
}

void oa::ExecutableGraph::add(
	oa::StringView inShader,
	oa::Span<oavk::Buffer> inBuffers,
	oa::Span<oa::BufferAccess> inAccess,
	const void* inPush, oa::U32 inPushSize,
	oa::U32 inGroupsX, oa::U32 inGroupsY, oa::U32 inGroupsZ
) {
	add(inShader, inBuffers, oa::Span<oa::SharedPtr<oavk::Buffer>>{}, inAccess,
		inPush, inPushSize, inGroupsX, inGroupsY, inGroupsZ);
}

void oa::ExecutableGraph::add(
	oa::StringView inShader,
	oa::Span<oavk::Buffer> inBuffers,
	oa::Span<oa::SharedPtr<oavk::Buffer>> inBufferOwners,
	oa::Span<oa::BufferAccess> inAccess,
	const void* inPush, oa::U32 inPushSize,
	oa::U32 inGroupsX, oa::U32 inGroupsY, oa::U32 inGroupsZ
) {
	oa::ComputeDispatchDesc desc;
	desc.kernel = inShader;
	desc.buffers = inBuffers;
	desc.bufferOwners = inBufferOwners;
	desc.access = inAccess;
	desc.pushData = inPush;
	desc.pushSize = inPushSize;
	desc.groupsX = inGroupsX;
	desc.groupsY = inGroupsY;
	desc.groupsZ = inGroupsZ;
	add(desc);
}

void oa::ExecutableGraph::add(
	oa::StringView inShader,
	oa::Span<oavk::Buffer> inBuffers,
	oa::Span<oa::BufferAccess> inAccess,
	const void* inPush, oa::U32 inPushSize,
	oa::U32 inGroupsX, oa::U32 inGroupsY, oa::U32 inGroupsZ,
	oa::QueueHint inQueue
) {
	add(inShader, inBuffers, oa::Span<oa::SharedPtr<oavk::Buffer>>{}, inAccess,
		inPush, inPushSize, inGroupsX, inGroupsY, inGroupsZ, inQueue);
}

void oa::ExecutableGraph::add(
	oa::StringView inShader,
	oa::Span<oavk::Buffer> inBuffers,
	oa::Span<oa::SharedPtr<oavk::Buffer>> inBufferOwners,
	oa::Span<oa::BufferAccess> inAccess,
	const void* inPush, oa::U32 inPushSize,
	oa::U32 inGroupsX, oa::U32 inGroupsY, oa::U32 inGroupsZ,
	oa::QueueHint inQueue
) {
	oa::ComputeDispatchDesc desc;
	desc.kernel = inShader;
	desc.buffers = inBuffers;
	desc.bufferOwners = inBufferOwners;
	desc.access = inAccess;
	desc.pushData = inPush;
	desc.pushSize = inPushSize;
	desc.groupsX = inGroupsX;
	desc.groupsY = inGroupsY;
	desc.groupsZ = inGroupsZ;
	desc.queue = inQueue;
	add(desc);
}

void oa::ExecutableGraph::addIndirect(
	oa::StringView inShader,
	oa::Span<oavk::Buffer> inBuffers,
	oa::Span<oa::BufferAccess> inAccess,
	const void* inPush, oa::U32 inPushSize,
	const oavk::Buffer& inIndirectBuffer, oa::U64 inOffset
) {
	addIndirect(inShader, inBuffers, oa::Span<oa::SharedPtr<oavk::Buffer>>{},
		inAccess, inPush, inPushSize, inIndirectBuffer, inOffset);
}

void oa::ExecutableGraph::addIndirect(
	oa::StringView inShader,
	oa::Span<oavk::Buffer> inBuffers,
	oa::Span<oa::SharedPtr<oavk::Buffer>> inBufferOwners,
	oa::Span<oa::BufferAccess> inAccess,
	const void* inPush, oa::U32 inPushSize,
	const oavk::Buffer& inIndirectBuffer, oa::U64 inOffset
) {
	oa::ComputeDispatchDesc desc;
	desc.kernel = inShader;
	desc.buffers = inBuffers;
	desc.bufferOwners = inBufferOwners;
	desc.access = inAccess;
	desc.pushData = inPush;
	desc.pushSize = inPushSize;
	desc.indirectBuffer = inIndirectBuffer;
	desc.indirectOffset = inOffset;
	desc.indirect = true;
	add(desc);
}

static oa::ComputeNode makeComputeNode(const oa::ComputeDispatchDesc& inDesc) {
	oa::ComputeNode node;
	node.operation = oa::String(inDesc.operation);
	node.semanticOps.assign(
		inDesc.semanticOps.begin(), inDesc.semanticOps.end());
	node.implementationId = inDesc.implementationId;
	node.opContractHash = inDesc.opContractHash;
	node.problemContractHash = inDesc.problemContractHash;
	node.kernelContentHash = inDesc.kernelContentHash;
	node.kernelSelection = inDesc.kernelSelection;
	node.shader = oa::String(inDesc.kernel);
	node.buffers.assign(inDesc.buffers.begin(), inDesc.buffers.end());
	node.bufferOwners.assign(inDesc.bufferOwners.begin(), inDesc.bufferOwners.end());
	node.access.assign(inDesc.access.begin(), inDesc.access.end());
	node.dtype = inDesc.dtype;
	node.groupsX = inDesc.groupsX;
	node.groupsY = inDesc.groupsY;
	node.groupsZ = inDesc.groupsZ;
	node.indirectBuffer = inDesc.indirectBuffer;
	node.indirectOffset = inDesc.indirectOffset;
	node.indirect = inDesc.indirect;
	node.queue = inDesc.queue;
	if (inDesc.pushData and inDesc.pushSize > 0) {
		oa::memcpy(node.pushData, inDesc.pushData, inDesc.pushSize);
		node.pushSize = inDesc.pushSize;
	}
	return node;
}

static oa::ComputeDispatchDesc makeDispatchDesc(oa::ComputeNode& inNode) {
	oa::ComputeDispatchDesc desc;
	desc.operation = inNode.operation;
	desc.semanticOps = oa::Span<const oa::U32>(
		inNode.semanticOps.data(), inNode.semanticOps.size());
	desc.implementationId = inNode.implementationId;
	desc.opContractHash = inNode.opContractHash;
	desc.problemContractHash = inNode.problemContractHash;
	desc.kernelContentHash = inNode.kernelContentHash;
	desc.kernelSelection = inNode.kernelSelection;
	desc.kernel = inNode.shader;
	desc.buffers = inNode.buffers.span();
	desc.bufferOwners = inNode.bufferOwners.span();
	desc.access = inNode.access.span();
	desc.pushData = inNode.pushSize > 0 ? inNode.pushData : nullptr;
	desc.pushSize = inNode.pushSize;
	desc.dtype = inNode.dtype;
	desc.groupsX = inNode.groupsX;
	desc.groupsY = inNode.groupsY;
	desc.groupsZ = inNode.groupsZ;
	desc.indirectBuffer = inNode.indirectBuffer;
	desc.indirectOffset = inNode.indirectOffset;
	desc.indirect = inNode.indirect;
	desc.queue = inNode.queue;
	return desc;
}

static VkAccessFlags2 shaderAccessMask(oa::BufferAccess inAccess) {
	if (inAccess == oa::BufferAccess::Read) {
		return VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
	}
	if (inAccess == oa::BufferAccess::Write) {
		return VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
	}
	return VK_ACCESS_2_SHADER_STORAGE_READ_BIT
		| VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
}

// Shared hazard computation used by execute and compile. Read-after-read is the
// only reuse requiring no dependency. RAW/WAW use memory dependencies; WAR uses
// a pure execution dependency (zero access masks), as required by vulkan.
static oa::U32 computeNodeBarriers(
	const oa::ComputeNode& inNode,
	oa::U32 inNodeIndex,
	const oa::HashMap<void*, GraphBufferState>& inState,
	oa::Vec<VkBufferMemoryBarrier2>& outBufferBarriers,
	oa::Vec<VkMemoryBarrier2>& outAliasBarriers,
	oa::Vec<GraphBarrierDebug>* outBufferDebug,
	oa::Vec<GraphBarrierDebug>* outAliasDebug)
{
	outBufferBarriers.clear();
	outAliasBarriers.clear();
	if (outBufferDebug) outBufferDebug->clear();
	if (outAliasDebug) outAliasDebug->clear();
	auto emit = [&](const oavk::Buffer& inBuffer, VkPipelineStageFlags2 inStageMask,
		VkAccessFlags2 inAccessMask, bool inWrites) {
		auto it = inState.find(inBuffer.synchronizationIdentity());
		if (it == inState.end()) return;
		const GraphBufferState& previous = it->second;
		if (!previous.writes && !inWrites) return;
		GraphBarrierDebug debug;
		debug.hazard = previous.writes
			? (inWrites ? GraphHazard::Waw : GraphHazard::Raw)
			: GraphHazard::War;
		debug.sourceFirstNode = previous.firstNode;
		debug.sourceLastNode = previous.lastNode;
		debug.destinationNode = inNodeIndex;
		debug.sourceBuffer = previous.buffer;
		debug.destinationBuffer = inBuffer.buffer;
		debug.hazardIdentity = inBuffer.synchronizationIdentity();
		debug.bytes = inBuffer.size;
		debug.sourceQueue = previous.queue;
		debug.sourceQueuesMixed = previous.mixedQueues;
		debug.destinationQueue = inNode.queue;
		debug.sourceStages = previous.stageMask;
		debug.sourceAccess = previous.writes ? previous.accessMask : 0;
		debug.destinationStages = inStageMask;
		debug.destinationAccess = previous.writes ? inAccessMask : 0;

		if (previous.buffer != inBuffer.buffer) {
			// Buffer barriers scope accesses through one VkBuffer handle. aliases
			// are distinct handles over the same memory and require a global
			// dependency at the lifetime hand-off.
			VkMemoryBarrier2 bar{};
			bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
			bar.srcStageMask = previous.stageMask;
			bar.srcAccessMask = previous.writes ? previous.accessMask : 0;
			bar.dstStageMask = inStageMask;
			bar.dstAccessMask = previous.writes ? inAccessMask : 0;
			outAliasBarriers.pushBack(bar);
			debug.alias = true;
			if (outAliasDebug) outAliasDebug->pushBack(debug);
			return;
		}
		VkBufferMemoryBarrier2 bar{};
		bar.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
		bar.srcStageMask = previous.stageMask;
		bar.srcAccessMask = previous.writes ? previous.accessMask : 0;
		bar.dstStageMask = inStageMask;
		bar.dstAccessMask = previous.writes ? inAccessMask : 0;
		bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bar.buffer = static_cast<::VkBuffer>(inBuffer.buffer);
		bar.offset = 0;
		bar.size = VK_WHOLE_SIZE;
		outBufferBarriers.pushBack(bar);
		if (outBufferDebug) outBufferDebug->pushBack(debug);
	};

	// Merge duplicate declarations in-place without allocating a temporary
	// access vector. Node fan-in is small, so this quadratic scan is cheaper
	// than heap traffic in the graph planner's per-step hot path.
	for (oa::U32 i = 0; i < static_cast<oa::U32>(inNode.buffers.size()); ++i) {
		void* handle = inNode.buffers[i].buffer;
		if (!handle) continue;
		bool alreadyProcessed = false;
		for (oa::U32 p = 0; p < i; ++p) {
			if (inNode.buffers[p].buffer == handle) {
				alreadyProcessed = true;
				break;
			}
		}
		if (alreadyProcessed) continue;

		VkPipelineStageFlags2 stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		VkAccessFlags2 accessMask = 0;
		bool writes = false;
		for (oa::U32 j = i; j < static_cast<oa::U32>(inNode.buffers.size()); ++j) {
			if (inNode.buffers[j].buffer != handle) continue;
			accessMask |= shaderAccessMask(inNode.access[j]);
			writes = writes || inNode.access[j] != oa::BufferAccess::Read;
		}
		if (inNode.indirect && inNode.indirectBuffer.buffer == handle) {
			stageMask |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
			accessMask |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
		}
		emit(inNode.buffers[i], stageMask, accessMask, writes);
	}

	if (inNode.indirect && inNode.indirectBuffer.buffer) {
		bool includedByRegularAccess = false;
		for (const auto& buffer : inNode.buffers) {
			if (buffer.buffer == inNode.indirectBuffer.buffer) {
				includedByRegularAccess = true;
				break;
			}
		}
		if (!includedByRegularAccess) {
			emit(inNode.indirectBuffer,
				VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
				VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, false);
		}
	}
	return static_cast<oa::U32>(outBufferBarriers.size() + outAliasBarriers.size());
}

// An execution dependency orders pipeline stages, not individual buffers.  If
// a barrier batch already contains a RAW/WAW memory dependency whose source
// and destination stage scopes cover a WAR dependency, the WAR contributes no
// additional ordering and can be omitted.  This is common at optimizer
// boundaries: gradient RAW already orders the backward dispatch (which also
// reads the weight) before the optimizer dispatch (which writes the weight).
// A standalone WAR is retained.
static oa::U32 pruneRedundantWarBarriers(
	oa::Vec<VkBufferMemoryBarrier2>& inOutBarriers,
	oa::Vec<GraphBarrierDebug>* inOutDebug)
{
	VkPipelineStageFlags2 orderedSrcStages = 0;
	VkPipelineStageFlags2 orderedDstStages = 0;
	for (const auto& bar : inOutBarriers) {
		if (bar.srcAccessMask == 0 && bar.dstAccessMask == 0) continue;
		orderedSrcStages |= bar.srcStageMask;
		orderedDstStages |= bar.dstStageMask;
	}

	oa::Usize write = 0;
	oa::U32 removed = 0;
	for (oa::Usize read = 0; read < inOutBarriers.size(); ++read) {
		const auto& bar = inOutBarriers[read];
		const bool isWar = bar.srcAccessMask == 0 && bar.dstAccessMask == 0;
		const bool stageOrderingCovered =
			(bar.srcStageMask & ~orderedSrcStages) == 0
			&& (bar.dstStageMask & ~orderedDstStages) == 0;
		if (isWar && stageOrderingCovered) {
			++removed;
			continue;
		}
		if (write != read) {
			inOutBarriers[write] = bar;
			if (inOutDebug) (*inOutDebug)[write] = (*inOutDebug)[read];
		}
		++write;
	}
	inOutBarriers.resize(write);
	if (inOutDebug) inOutDebug->resize(write);
	return removed;
}

static void updateBufferStates(
	const oa::ComputeNode& inNode,
	oa::U32 inNodeIndex,
	oa::HashMap<void*, GraphBufferState>& inOutState)
{
	auto update = [&](const oavk::Buffer& inBuffer, VkPipelineStageFlags2 inStageMask,
		VkAccessFlags2 inAccessMask, bool inWrites) {
		GraphBufferState next;
		next.stageMask = inStageMask;
		next.accessMask = inAccessMask;
		next.writes = inWrites;
		next.buffer = inBuffer.buffer;
		next.queue = inNode.queue;
		next.mixedQueues = false;
		next.firstNode = inNodeIndex;
		next.lastNode = inNodeIndex;

		auto it = inOutState.find(inBuffer.synchronizationIdentity());
		if (it == inOutState.end()) {
			inOutState.emplace(inBuffer.synchronizationIdentity(), next);
		} else if (!it->second.writes && !next.writes) {
			// Preserve every outstanding read domain for a later WAR dependency.
			it->second.stageMask |= next.stageMask;
			it->second.accessMask |= next.accessMask;
			it->second.mixedQueues = it->second.mixedQueues
				or it->second.queue != next.queue;
			it->second.lastNode = inNodeIndex;
		} else {
			it->second = next;
		}
	};

	for (oa::U32 i = 0; i < static_cast<oa::U32>(inNode.buffers.size()); ++i) {
		void* handle = inNode.buffers[i].buffer;
		if (!handle) continue;
		bool alreadyProcessed = false;
		for (oa::U32 p = 0; p < i; ++p) {
			if (inNode.buffers[p].buffer == handle) {
				alreadyProcessed = true;
				break;
			}
		}
		if (alreadyProcessed) continue;

		VkPipelineStageFlags2 stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		VkAccessFlags2 accessMask = 0;
		bool writes = false;
		for (oa::U32 j = i; j < static_cast<oa::U32>(inNode.buffers.size()); ++j) {
			if (inNode.buffers[j].buffer != handle) continue;
			accessMask |= shaderAccessMask(inNode.access[j]);
			writes = writes || inNode.access[j] != oa::BufferAccess::Read;
		}
		if (inNode.indirect && inNode.indirectBuffer.buffer == handle) {
			stageMask |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
			accessMask |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
		}
		update(inNode.buffers[i], stageMask, accessMask, writes);
	}

	if (inNode.indirect && inNode.indirectBuffer.buffer) {
		bool includedByRegularAccess = false;
		for (const auto& buffer : inNode.buffers) {
			if (buffer.buffer == inNode.indirectBuffer.buffer) {
				includedByRegularAccess = true;
				break;
			}
		}
		if (!includedByRegularAccess) {
			update(inNode.indirectBuffer,
				VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
				VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, false);
		}
	}
}

static void recordFinalBarrier(
	const OaVkDeviceTable& inDispatch,
	VkCommandBuffer inCb,
	oa::Bool inRequired)
{
	if (not inRequired) return;
	// graph submissions may be followed by a host wait and direct mapped-buffer
	// read. Keep the required compute -> host visibility edge. TRANSFER is not a
	// graph-final consumer; transfer work in a later submission is synchronized
	// by its semaphore dependency and must declare its own destination stage.
	VkMemoryBarrier2 finalBar{};
	finalBar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
	finalBar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
	finalBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
	finalBar.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
	finalBar.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;

	VkDependencyInfo finalDep{};
	finalDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	finalDep.memoryBarrierCount = 1;
	finalDep.pMemoryBarriers = &finalBar;
	inDispatch.vkCmdPipelineBarrier2(inCb, &finalDep);
}

// allocate a per-dispatch descriptor pool + set. Mirrors AllocStreamDescriptorSet.
static oa::Status allocGraphDescriptorSet(
	const oavk::Device& inDevice,
	oa::ComputePipeline& inPipeline,
	oa::Span<oavk::Buffer> inBuffers,
	void** outPool,
	void** outSet
) {
	OA_RETURN_IF_ERROR(oavk::validateStorageBufferDescriptors(
		inDevice, inBuffers));
	VkDevice dev = static_cast<VkDevice>(inDevice.device);

	oa::U32 numDesc = inPipeline.numBindings;
	if (numDesc < static_cast<oa::U32>(inBuffers.size())) {
		numDesc = static_cast<oa::U32>(inBuffers.size());
	}
	if (numDesc == 0) numDesc = 1;

	VkDescriptorPoolSize poolSize{};
	poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSize.descriptorCount = numDesc;

	VkDescriptorPoolCreateInfo dpCI{};
	dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpCI.maxSets = 1;
	dpCI.poolSizeCount = 1;
	dpCI.pPoolSizes = &poolSize;

	VkDescriptorPool pool = VK_NULL_HANDLE;
	VkResult r = inDevice.deviceDispatch.vkCreateDescriptorPool(
		dev, &dpCI, nullptr, &pool);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::PipelineError,
			"graph: vkCreateDescriptorPool failed");
	}

	VkDescriptorSetLayout dsl = static_cast<VkDescriptorSetLayout>(inPipeline.descriptorSetLayout);
	VkDescriptorSetAllocateInfo dsAI{};
	dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAI.descriptorPool = pool;
	dsAI.descriptorSetCount = 1;
	dsAI.pSetLayouts = &dsl;

	VkDescriptorSet ds = VK_NULL_HANDLE;
	r = inDevice.deviceDispatch.vkAllocateDescriptorSets(dev, &dsAI, &ds);
	if (r != VK_SUCCESS) {
		inDevice.deviceDispatch.vkDestroyDescriptorPool(dev, pool, nullptr);
		return oa::Status::error(oa::StatusCode::PipelineError,
			"graph: vkAllocateDescriptorSets failed");
	}

	for (oa::U32 i = 0; i < static_cast<oa::U32>(inBuffers.size()); ++i) {
		VkDescriptorBufferInfo bufInfo{};
		bufInfo.buffer = static_cast<::VkBuffer>(inBuffers[i].buffer);
		bufInfo.offset = 0;
		bufInfo.range = inBuffers[i].descriptorRange();

		VkWriteDescriptorSet write{};
		write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write.dstSet = ds;
		write.dstBinding = i;
		write.descriptorCount = 1;
		write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		write.pBufferInfo = &bufInfo;

		inDevice.deviceDispatch.vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
	}

	*outPool = pool;
	*outSet = ds;
	return oa::Status::ok();
}

// ─── phase 1: One-shot execution ──────────────────────────────────────────────

oa::Status oa::ExecutableGraph::execute(oa::Engine& inRt) {
	if (nodes_.empty()) return oa::Status::ok();

	oavk::Stream* stream = oa::EngineSubmissionAccess::acquireStream(inRt);
	if (!stream) {
		return oa::Status::error(oa::StatusCode::VulkanError, "graph: failed to acquire stream");
	}

	OA_RETURN_IF_ERROR(stream->begin(oa::EngineDeviceAccess::get(inRt)));

	oa::HashMap<void*, GraphBufferState> bufferStates;
	bufferStates.reserve(nodes_.size() * 4U);
	oa::Vec<VkBufferMemoryBarrier2> barriers;
	oa::Vec<VkMemoryBarrier2> aliasBarriers;
	barriers.reserve(8);

	for (oa::U32 nodeIdx = 0; nodeIdx < nodes_.size(); ++nodeIdx) {
		auto& node = nodes_[nodeIdx];
		computeNodeBarriers(node, nodeIdx, bufferStates, barriers, aliasBarriers);
		pruneRedundantWarBarriers(barriers);

		if (!barriers.empty() || !aliasBarriers.empty()) {
			VkDependencyInfo dep{};
			dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
			dep.memoryBarrierCount = static_cast<oa::U32>(aliasBarriers.size());
			dep.pMemoryBarriers = aliasBarriers.data();
			dep.bufferMemoryBarrierCount = static_cast<oa::U32>(barriers.size());
			dep.pBufferMemoryBarriers = barriers.data();
			oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdPipelineBarrier2(
				static_cast<VkCommandBuffer>(stream->commandBuffer), &dep);
		}

		OA_RETURN_IF_ERROR(stream->recordDispatchDesc(inRt, makeDispatchDesc(node)));

		updateBufferStates(node, nodeIdx, bufferStates);
	}

	recordFinalBarrier(
		oa::EngineDeviceAccess::get(inRt).deviceDispatch,
		static_cast<VkCommandBuffer>(stream->commandBuffer),
		hostReadbackRequired_);

	oa::Status status = stream->submitAndWait(inRt);
	oa::EngineSubmissionAccess::releaseStream(inRt, stream);
	return status;
}

// ─── Multi-queue Execution ────────────────────────────────────────────────

oa::Status oa::ExecutableGraph::executeMultiQueue(oa::Engine& inRt) {
	if (nodes_.empty()) return oa::Status::ok();

	if (!oa::EngineDeviceAccess::get(inRt).queues.hasAsyncCompute) {
		return execute(inRt);
	}

	oavk::Stream* computeStream = oa::EngineSubmissionAccess::acquireStream(inRt);
	oa::EngineAccess engineAccess(inRt);
	oavk::Stream* asyncStream = engineAccess.acquireAsyncStream();
	if (!computeStream || !asyncStream) {
		if (computeStream) oa::EngineSubmissionAccess::releaseStream(inRt, computeStream);
		if (asyncStream) engineAccess.releaseAsyncStream(asyncStream);
		return execute(inRt);
	}

	OA_RETURN_IF_ERROR(computeStream->begin(oa::EngineDeviceAccess::get(inRt)));
	OA_RETURN_IF_ERROR(asyncStream->begin(oa::EngineDeviceAccess::get(inRt)));

	oa::HashMap<void*, GraphBufferState> bufferStates;
	bufferStates.reserve(nodes_.size() * 4U);
	oa::Vec<VkBufferMemoryBarrier2> barriers;
	oa::Vec<VkMemoryBarrier2> aliasBarriers;
	barriers.reserve(8);

	// Track the last-submitted stream for cross-queue dependency
	oavk::Stream* prevStream = nullptr;
	oa::U64 prevTimelineValue = 0;

	oa::QueueHint currentQueue = oa::QueueHint::Compute;
	for (oa::U32 i = 0; i < static_cast<oa::U32>(nodes_.size()); ++i) {
		auto& node = nodes_[i];
		oa::QueueHint nodeQueue = node.queue;
		// Transfer collapses to compute unless separate transfer queue exists
		if (nodeQueue == oa::QueueHint::Transfer) nodeQueue = oa::QueueHint::Compute;

		if (nodeQueue != currentQueue) {
			oavk::Stream* outgoing = (currentQueue == oa::QueueHint::AsyncCompute)
				? asyncStream : computeStream;

			recordFinalBarrier(
				oa::EngineDeviceAccess::get(inRt).deviceDispatch,
				static_cast<VkCommandBuffer>(outgoing->commandBuffer),
				hostReadbackRequired_);
			OA_RETURN_IF_ERROR(outgoing->submit(inRt));

			// Track the dependency for the incoming stream
			prevStream = outgoing;
			prevTimelineValue = outgoing->timelineValue;

			oavk::Stream* incoming = (nodeQueue == oa::QueueHint::AsyncCompute)
				? asyncStream : computeStream;
			OA_RETURN_IF_ERROR(incoming->begin(oa::EngineDeviceAccess::get(inRt)));

			currentQueue = nodeQueue;
		}

		oavk::Stream* activeStream = (currentQueue == oa::QueueHint::AsyncCompute)
			? asyncStream : computeStream;

		computeNodeBarriers(node, i, bufferStates, barriers, aliasBarriers);
		pruneRedundantWarBarriers(barriers);
		if (!barriers.empty() || !aliasBarriers.empty()) {
			VkDependencyInfo dep{};
			dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
			dep.memoryBarrierCount = static_cast<oa::U32>(aliasBarriers.size());
			dep.pMemoryBarriers = aliasBarriers.data();
			dep.bufferMemoryBarrierCount = static_cast<oa::U32>(barriers.size());
			dep.pBufferMemoryBarriers = barriers.data();
			oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdPipelineBarrier2(
				static_cast<VkCommandBuffer>(activeStream->commandBuffer), &dep);
		}

		OA_RETURN_IF_ERROR(activeStream->recordDispatchDesc(
			inRt, makeDispatchDesc(node)));

		updateBufferStates(node, i, bufferStates);
	}

	// submit final batch with cross-queue dependency if needed
	oavk::Stream* finalStream = (currentQueue == oa::QueueHint::AsyncCompute)
		? asyncStream : computeStream;
	recordFinalBarrier(
		oa::EngineDeviceAccess::get(inRt).deviceDispatch,
		static_cast<VkCommandBuffer>(finalStream->commandBuffer),
		hostReadbackRequired_);

	oa::Status status;
	if (prevStream && prevStream != finalStream) {
		// GPU-side wait: final stream waits on previous stream's semaphore
		OA_RETURN_IF_ERROR(finalStream->submitWithDependency(
			inRt, prevStream->timelineSem, prevTimelineValue));
		status = finalStream->synchronize(oa::EngineDeviceAccess::get(inRt));
	} else {
		status = finalStream->submitAndWait(inRt);
	}

	// Ensure all streams are complete before releasing
	if (computeStream->submitted) {
		if (auto s = computeStream->synchronize(oa::EngineDeviceAccess::get(inRt)); !s.isOk()) {
			OaLogError(oa::LogComponent::Compute, "graph execute: compute stream sync failed: %s", s.getMessage().cStr());
		}
	}
	if (asyncStream->submitted) {
		if (auto s = asyncStream->synchronize(oa::EngineDeviceAccess::get(inRt)); !s.isOk()) {
			OaLogError(oa::LogComponent::Compute, "graph execute: async stream sync failed: %s", s.getMessage().cStr());
		}
	}

	oa::EngineSubmissionAccess::releaseStream(inRt, computeStream);
	engineAccess.releaseAsyncStream(asyncStream);
	return status;
}

// ─── phase 2: compile & replay ────────────────────────────────────────────────

oa::U64 oa::ExecutableGraph::computeNodeHash() const {
	oa::U64 hash = 14695981039346656037ULL;  // FNV-1a 64-bit offset basis
	auto fnv = [&](const void* inData, oa::U64 inBytes) {
		const auto* p = static_cast<const oa::U8*>(inData);
		for (oa::U64 i = 0; i < inBytes; ++i) {
			hash ^= p[i];
			hash *= 1099511628211ULL;  // FNV-1a 64-bit prime
		}
	};

	// Node count
	const oa::U32 n = static_cast<oa::U32>(nodes_.size());
	fnv(&n, sizeof(n));
	fnv(&hostReadbackRequired_, sizeof(hostReadbackRequired_));
	fnv(&replayTimingEnabled_, sizeof(replayTimingEnabled_));

	for (const auto& node : nodes_) {
		// shader name
		fnv(node.shader.data(), node.shader.size());
		// dtype
		fnv(&node.dtype, sizeof(node.dtype));
		// Buffer identity and access declarations. The VkBuffer handles are
		// baked into synchronization barriers, while bindless indices are baked
		// into push constants; both therefore participate in cache identity.
		const oa::U32 nBufs = static_cast<oa::U32>(node.buffers.size());
		fnv(&nBufs, sizeof(nBufs));
		for (oa::U32 i = 0; i < nBufs; ++i) {
			const auto& buf = node.buffers[i];
			fnv(&buf.buffer, sizeof(buf.buffer));
			fnv(&buf.bindlessIndex, sizeof(buf.bindlessIndex));
			fnv(&buf.size, sizeof(buf.size));
			fnv(&node.access[i], sizeof(node.access[i]));
		}
		// Push constants (includes buffer index header for bindless + user push)
		fnv(&node.pushSize, sizeof(node.pushSize));
		fnv(node.pushData, node.pushSize);
		// Dispatch dimensions
		fnv(&node.groupsX, sizeof(node.groupsX));
		fnv(&node.groupsY, sizeof(node.groupsY));
		fnv(&node.groupsZ, sizeof(node.groupsZ));
		// indirect dispatch flag
		fnv(&node.indirect, sizeof(node.indirect));
		if (node.indirect) {
			fnv(&node.indirectBuffer.buffer, sizeof(node.indirectBuffer.buffer));
			fnv(&node.indirectBuffer.bindlessIndex,
				sizeof(node.indirectBuffer.bindlessIndex));
			fnv(&node.indirectBuffer.size, sizeof(node.indirectBuffer.size));
			// Admission depends on creation-time indirect usage and allocator
			// ownership. Keep both in cache identity so a descriptor whose
			// metadata changed cannot reuse a previously recorded command buffer
			// before the compile preflight sees it.
			fnv(&node.indirectBuffer.flags, sizeof(node.indirectBuffer.flags));
			fnv(&node.indirectBuffer.allocatorIdentity,
				sizeof(node.indirectBuffer.allocatorIdentity));
			fnv(&node.indirectOffset, sizeof(node.indirectOffset));
		}
		fnv(&node.queue, sizeof(node.queue));
		const oa::U32 semanticOpCount =
			static_cast<oa::U32>(node.semanticOps.size());
		fnv(&semanticOpCount, sizeof(semanticOpCount));
		for (const auto semanticOp : node.semanticOps) {
			fnv(&semanticOp, sizeof(semanticOp));
		}
	}

	return hash;
}

oa::Status oa::ExecutableGraph::compile(oa::Engine& inRt) {
	lastCompileReused_ = false;
	if (nodes_.empty()) {
		compiled_ = true;
		barrierCount_ = 0;
		return oa::Status::ok();
	}
	OA_RETURN_IF_ERROR(bindEngine_(inRt));

	// compile-once-replay-many: if the node list hash matches the last
	// compilation, the existing secondary CB is still valid. Skip
	// vkResetCommandBuffer + re-recording entirely — just mark compiled.
	// Also skip rebuilding the primary CB — it still wraps the same secondary.
	const oa::U64 nodeHash = computeNodeHash();
	static const oa::Bool logGraphCacheMisses =
		oa::EnvFlag::isSet("OA_LOG_GRAPH_CACHE_MISSES");
	if (logGraphCacheMisses) {
		static thread_local oa::HashMap<const oa::ExecutableGraph*, GraphDebugHashes> previous;
		const auto current = executableGraphDebugHashes(
			oa::Span<const oa::ComputeNode>(nodes_.data(), nodes_.size()),
			hostReadbackRequired_);
		auto it = previous.find(this);
		if (it != previous.end()) {
			OaLogInfo(oa::LogComponent::Compute,
				"graph cache identity: topology=%s resources=%s push=%s",
				it->second.topology == current.topology ? "same" : "changed",
				it->second.resources == current.resources ? "same" : "changed",
				it->second.push == current.push ? "same" : "changed");
		}
		if (it != previous.end()) {
			it->second = current;
		} else {
			previous.emplace(this, current);
		}
	}
	if (secondaryCb_ and nodeHash == lastCompileHash_) {
		compiled_ = true;
		lastCompileReused_ = true;
		return oa::Status::ok();
	}

	// Batch only the variants required by this graph. This preserves lazy startup
	// while using the same bounded parallel loading and compact summary reporting
	// as eager engine preload.
	oa::Vec<oa::PipelineVariantRequest> pipelineRequests;
	pipelineRequests.reserve(nodes_.size());
	for (const auto& node : nodes_) {
		pipelineRequests.pushBack({.name = node.shader, .dtype = node.dtype});
	}
	OA_RETURN_IF_ERROR(
		oa::EnginePipelineAccess::get(inRt).ensurePipelinesOnDemand(
			oa::Span<const oa::PipelineVariantRequest>(
				pipelineRequests.data(), pipelineRequests.size())));

	// validate the complete selected-engine descriptor set before resetting or
	// beginning a command buffer.
	for (const auto& node : nodes_) {
		auto& pipeline =
			oa::EnginePipelineAccess::get(inRt).getPipeline(node.shader, node.dtype);
		if (not pipeline.pipeline) {
			return oa::Status::error(
				"graph compile: pipeline not found: " + node.shader);
		}
		if (node.indirect) {
			OA_RETURN_IF_ERROR(oavk::validateIndirectComputeDispatch(
				node.indirectBuffer,
				node.indirectOffset,
				oa::EngineAllocatorAccess::get(inRt).allocator));
		} else {
			OA_RETURN_IF_ERROR(oavk::validateDirectComputeDispatch(
				oa::EngineDeviceAccess::get(inRt),
				node.groupsX, node.groupsY, node.groupsZ));
		}
		OA_RETURN_IF_ERROR(oavk::validateStorageBufferDescriptors(
			oa::EngineDeviceAccess::get(inRt),
			node.buffers.span(),
			pipeline.bindless,
			oa::EngineAllocatorAccess::get(inRt).allocator));
		if (pipeline.bindless and
			not oavk::bindlessPushFits(
				static_cast<oa::U32>(node.buffers.size()), node.pushSize))
		{
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"graph compile: bindless push exceeds "
				"oavk::OA_VK_MAX_PUSH_CONSTANT_BYTES "
				"(buffer index header + user push)");
		}
	}

	VkDevice dev = static_cast<VkDevice>(oa::EngineDeviceAccess::get(inRt).device);
	queueFamily_ = oa::EngineDeviceAccess::get(inRt).queues.computeQueueFamily;
	if (replayTimingEnabled_ and not replayTimestamp_.pool) {
		auto timestamp = oavk::Timestamp::create(inRt, 2);
		if (not timestamp) return timestamp.getStatus();
		replayTimestamp_ = oa::move(timestamp).getValue();
	}

	// Reuse existing command pool + secondary CB if available. This avoids
	// vkCreateCommandPool + vkAllocateCommandBuffers + vkDestroyCommandPool
	// + vkFreeCommandBuffers per completed submission — saves ~0.05ms.
	if (secondaryPool_ and secondaryCb_) {
		// Free descriptor pools from previous compilation (non-bindless only).
		for (void* pool : descriptorPools_) {
			oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkDestroyDescriptorPool(dev, static_cast<VkDescriptorPool>(pool), nullptr);
		}
		descriptorPools_.clear();

		// reset the secondary CB for re-recording.
		VkResult r = oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkResetCommandBuffer(
			static_cast<VkCommandBuffer>(secondaryCb_), 0);
		if (r != VK_SUCCESS) {
			// reset failed — fall back to full recreate.
			invalidateDevice_(oa::EngineDeviceAccess::get(inRt));
		}
		compiledBufferOwners_.clear();
		// compiled_ is already false (set by clearNodes or Invalidate)
	}

	if (!secondaryPool_) {
		// Create a dedicated command pool for the secondary CB.
		VkCommandPoolCreateInfo cpCI{};
		cpCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cpCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		cpCI.queueFamilyIndex = queueFamily_;

		VkCommandPool pool = VK_NULL_HANDLE;
		VkResult r = oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCreateCommandPool(dev, &cpCI, nullptr, &pool);
		if (r != VK_SUCCESS) {
			return oa::Status::error(oa::StatusCode::VulkanError,
				"graph compile: vkCreateCommandPool failed");
		}
		secondaryPool_ = pool;
	}

	if (!secondaryCb_) {
		// allocate secondary command buffer.
		VkCommandBufferAllocateInfo cbAI{};
		cbAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cbAI.commandPool = static_cast<VkCommandPool>(secondaryPool_);
		cbAI.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
		cbAI.commandBufferCount = 1;

		VkCommandBuffer scb = VK_NULL_HANDLE;
		VkResult r = oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkAllocateCommandBuffers(dev, &cbAI, &scb);
		if (r != VK_SUCCESS) {
			oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkDestroyCommandPool(dev, static_cast<VkCommandPool>(secondaryPool_), nullptr);
			secondaryPool_ = nullptr;
			return oa::Status::error(oa::StatusCode::VulkanError,
				"graph compile: vkAllocateCommandBuffers failed");
		}
		secondaryCb_ = scb;
	}

	// Begin secondary CB — simultaneous use allows replay while previous is in-flight
	VkCommandBufferInheritanceInfo inhInfo{};
	inhInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
	beginInfo.pInheritanceInfo = &inhInfo;

	VkResult r = oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkBeginCommandBuffer(static_cast<VkCommandBuffer>(secondaryCb_), &beginInfo);
	if (r != VK_SUCCESS) {
		invalidateDevice_(oa::EngineDeviceAccess::get(inRt));
		return oa::Status::error(oa::StatusCode::VulkanError,
			"graph compile: vkBeginCommandBuffer failed");
	}

	// Record all dispatches with minimal barriers into the secondary CB
	oa::HashMap<void*, GraphBufferState> bufferStates;
	bufferStates.reserve(nodes_.size() * 4U);
	oa::Vec<VkBufferMemoryBarrier2> barriers;
	oa::Vec<VkMemoryBarrier2> aliasBarriers;
	barriers.reserve(8);
	barrierCount_ = 0;
	warBarrierCount_ = 0;
	indirectBarrierCount_ = 0;
	aliasBarrierCount_ = 0;

	VkPipeline boundPipeline = VK_NULL_HANDLE;
	bool bindlessDescriptorBound = false;
	for (oa::U32 nodeIdx = 0; nodeIdx < nodes_.size(); ++nodeIdx) {
		auto& node = nodes_[nodeIdx];
		computeNodeBarriers(node, nodeIdx, bufferStates, barriers, aliasBarriers);
		pruneRedundantWarBarriers(barriers);
		barrierCount_ += static_cast<oa::U32>(barriers.size() + aliasBarriers.size());
		aliasBarrierCount_ += static_cast<oa::U32>(aliasBarriers.size());
		for (const auto& bar : barriers) {
			if (bar.srcAccessMask == 0 && bar.dstAccessMask == 0) {
				++warBarrierCount_;
			}
			if ((bar.dstStageMask & VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT) != 0
				&& (bar.dstAccessMask & VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT) != 0)
			{
				++indirectBarrierCount_;
			}
		}
		for (const auto& bar : aliasBarriers) {
			if (bar.srcAccessMask == 0 && bar.dstAccessMask == 0) {
				++warBarrierCount_;
			}
			if ((bar.dstStageMask & VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT) != 0
				&& (bar.dstAccessMask & VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT) != 0)
			{
				++indirectBarrierCount_;
			}
		}

		if (!barriers.empty() || !aliasBarriers.empty()) {
			VkDependencyInfo dep{};
			dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
			dep.memoryBarrierCount = static_cast<oa::U32>(aliasBarriers.size());
			dep.pMemoryBarriers = aliasBarriers.data();
			dep.bufferMemoryBarrierCount = static_cast<oa::U32>(barriers.size());
			dep.pBufferMemoryBarriers = barriers.data();
			oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdPipelineBarrier2(static_cast<VkCommandBuffer>(secondaryCb_), &dep);
		}

		// Resolve pipeline — DTYPE from the node (derived from its operand tensors).
			auto& pipeline = oa::EnginePipelineAccess::get(inRt).getPipeline(
				node.shader, node.dtype);
			if (!pipeline.pipeline) {
				invalidateDevice_(oa::EngineDeviceAccess::get(inRt));
				return oa::Status::error("graph compile: pipeline not found: " + node.shader);
			}
			void* dsPool = nullptr;
			void* dsSet = nullptr;
			if (!pipeline.bindless) {
				oa::Status dsStatus = allocGraphDescriptorSet(
					oa::EngineDeviceAccess::get(inRt), pipeline, node.buffers.span(), &dsPool, &dsSet);
				if (!dsStatus.isOk()) {
					invalidateDevice_(oa::EngineDeviceAccess::get(inRt));
					return dsStatus;
				}
				descriptorPools_.pushBack(dsPool);
			}

			const VkPipeline vkPipeline = static_cast<VkPipeline>(pipeline.pipeline);
			if (vkPipeline != boundPipeline) {
				oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdBindPipeline(static_cast<VkCommandBuffer>(secondaryCb_),
					VK_PIPELINE_BIND_POINT_COMPUTE, vkPipeline);
				boundPipeline = vkPipeline;
			}

			if (pipeline.bindless) {
				// Every bindless compute pipeline uses the engine-owned common
				// pipeline layout. Descriptor bindings remain valid across
				// compatible pipeline changes, so bind the global heap once.
				if (!bindlessDescriptorBound) {
					VkDescriptorSet bds =
						static_cast<VkDescriptorSet>(
							oa::EngineBindlessAccess::get(inRt).descriptorSet);
					oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdBindDescriptorSets(
						static_cast<VkCommandBuffer>(secondaryCb_),
						VK_PIPELINE_BIND_POINT_COMPUTE,
						static_cast<VkPipelineLayout>(pipeline.pipelineLayout),
						0, 1, &bds, 0, nullptr);
					bindlessDescriptorBound = true;
				}

				oa::U32 numBufs = static_cast<oa::U32>(node.buffers.size());
				oa::U32 headerBytes = numBufs * sizeof(oa::U32);
				oa::U32 totalPush = headerBytes + node.pushSize;
				alignas(16) oa::U8 pushBuf[oavk::OA_VK_MAX_PUSH_CONSTANT_BYTES] = {};
				oa::U32* indices = reinterpret_cast<oa::U32*>(pushBuf);
				for (oa::U32 b = 0; b < numBufs; ++b) {
					indices[b] = node.buffers[b].bindlessIndex;
				}
				if (node.pushSize > 0) {
					oa::memcpy(pushBuf + headerBytes, node.pushData, node.pushSize);
				}
				oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdPushConstants(static_cast<VkCommandBuffer>(secondaryCb_),
					static_cast<VkPipelineLayout>(pipeline.pipelineLayout),
					VK_SHADER_STAGE_COMPUTE_BIT, 0, totalPush, pushBuf);
			} else {
				bindlessDescriptorBound = false;
				VkDescriptorSet ds = static_cast<VkDescriptorSet>(dsSet);
				oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdBindDescriptorSets(static_cast<VkCommandBuffer>(secondaryCb_), VK_PIPELINE_BIND_POINT_COMPUTE,
					static_cast<VkPipelineLayout>(pipeline.pipelineLayout),
					0, 1, &ds, 0, nullptr);

				if (node.pushSize > 0) {
					oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdPushConstants(static_cast<VkCommandBuffer>(secondaryCb_),
						static_cast<VkPipelineLayout>(pipeline.pipelineLayout),
						VK_SHADER_STAGE_COMPUTE_BIT, 0, node.pushSize, node.pushData);
				}
			}

			if (node.indirect) {
				oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdDispatchIndirect(static_cast<VkCommandBuffer>(secondaryCb_),
					static_cast<::VkBuffer>(node.indirectBuffer.buffer),
					node.indirectOffset);
			} else {
				oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdDispatch(static_cast<VkCommandBuffer>(secondaryCb_), node.groupsX, node.groupsY, node.groupsZ);
			}
		updateBufferStates(node, nodeIdx, bufferStates);
	}

	recordFinalBarrier(
		oa::EngineDeviceAccess::get(inRt).deviceDispatch,
		static_cast<VkCommandBuffer>(secondaryCb_),
		hostReadbackRequired_);

	r = oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkEndCommandBuffer(static_cast<VkCommandBuffer>(secondaryCb_));
	if (r != VK_SUCCESS) {
		invalidateDevice_(oa::EngineDeviceAccess::get(inRt));
		return oa::Status::error(oa::StatusCode::VulkanError,
			"graph compile: vkEndCommandBuffer failed");
	}

	// Opt-in compile-time synchronization summary.
	if (oa::EnvFlag::isSet("OA_LOG_BARRIERS")) {
		const oa::U32 n = static_cast<oa::U32>(nodes_.size());
		OaLogInfo(oa::LogComponent::Compute,
			"oa::ExecutableGraph::compile: nodes=%u barriers=%u war=%u indirect=%u alias=%u",
			n, barrierCount_, warBarrierCount_, indirectBarrierCount_,
			aliasBarrierCount_);
	}

	// Build a pre-recorded primary CB that wraps the secondary. This lets
	// replay() skip stream acquire + Begin + vkCmdExecuteCommands entirely
	// — just submit the primary directly with a dedicated timeline semaphore.
	if (primaryCb_ and primaryPool_) {
		VkCommandBuffer pcb = static_cast<VkCommandBuffer>(primaryCb_);
		oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkResetCommandBuffer(pcb, 0);
	} else {
		VkCommandPoolCreateInfo ppCI{};
		ppCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		ppCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		ppCI.queueFamilyIndex = queueFamily_;
		VkResult pr = oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCreateCommandPool(dev, &ppCI, nullptr,
			reinterpret_cast<VkCommandPool*>(&primaryPool_));
		if (pr != VK_SUCCESS) {
			return oa::Status::error(oa::StatusCode::VulkanError,
				"graph compile: vkCreateCommandPool (primary) failed");
		}
		VkCommandBufferAllocateInfo cbAI{};
		cbAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cbAI.commandPool = static_cast<VkCommandPool>(primaryPool_);
		cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cbAI.commandBufferCount = 1;
		pr = oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkAllocateCommandBuffers(dev, &cbAI,
			reinterpret_cast<VkCommandBuffer*>(&primaryCb_));
		if (pr != VK_SUCCESS) {
			oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkDestroyCommandPool(dev,
				static_cast<VkCommandPool>(primaryPool_), nullptr);
			primaryPool_ = nullptr;
			return oa::Status::error(oa::StatusCode::VulkanError,
				"graph compile: vkAllocateCommandBuffers (primary) failed");
		}
		// Create the dedicated timeline semaphore for cached replay.
		if (not replayTimelineSem_.semaphore) {
			auto semRes = oavk::TimelineSemaphore::create(oa::EngineDeviceAccess::get(inRt));
			if (not semRes) {
				return oa::Status::error(oa::StatusCode::VulkanError,
					"graph compile: timeline semaphore creation failed");
			}
			replayTimelineSem_ = oa::move(semRes).getValue();
		}
	}
	{
		VkCommandBuffer pcb = static_cast<VkCommandBuffer>(primaryCb_);
		VkCommandBufferBeginInfo bi{};
		bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		bi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		VkResult pr = oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkBeginCommandBuffer(pcb, &bi);
		if (pr != VK_SUCCESS) {
			return oa::Status::error(oa::StatusCode::VulkanError,
				"graph compile: vkBeginCommandBuffer (primary) failed");
		}
		if (replayTimingEnabled_) {
			oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdResetQueryPool(pcb,
				static_cast<VkQueryPool>(replayTimestamp_.pool), 0, 2);
			oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdWriteTimestamp2(pcb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
				static_cast<VkQueryPool>(replayTimestamp_.pool), 0);
			replayTimestamp_.writeIndex = 2;
		}
		// replay is re-entrant: parameters, optimizer state, recurrent state, and
		// GPU-built plans written by replay N may be consumed by replay N+1. Put
		// the external memory dependency in the reusable primary wrapper so every
		// replay begins with an explicit device/host-write -> compute/indirect-read
		// boundary instead of relying on cache behavior across submissions.
		VkMemoryBarrier2 reentry{};
		reentry.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
		reentry.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
			| VK_PIPELINE_STAGE_2_HOST_BIT;
		reentry.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT
			| VK_ACCESS_2_HOST_WRITE_BIT;
		reentry.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
			| VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
		reentry.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
			| VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
			| VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
		VkDependencyInfo reentryDep{};
		reentryDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		reentryDep.memoryBarrierCount = 1;
		reentryDep.pMemoryBarriers = &reentry;
		oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdPipelineBarrier2(pcb, &reentryDep);

		VkCommandBuffer scb = static_cast<VkCommandBuffer>(secondaryCb_);
		oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdExecuteCommands(pcb, 1, &scb);
		if (replayTimingEnabled_) {
			oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdWriteTimestamp2(pcb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
				static_cast<VkQueryPool>(replayTimestamp_.pool), 1);
		}
		pr = oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkEndCommandBuffer(pcb);
		if (pr != VK_SUCCESS) {
			return oa::Status::error(oa::StatusCode::VulkanError,
				"graph compile: vkEndCommandBuffer (primary) failed");
		}
	}

	compiled_ = true;
	lastCompileHash_ = nodeHash;
	compiledBufferOwners_.clear();
	for (const auto& node : nodes_) {
		for (const auto& owner : node.bufferOwners) {
			if (not owner) continue;
			oa::Bool found = false;
			for (const auto& existing : compiledBufferOwners_) {
				if (existing.get() == owner.get()) {
					found = true;
					break;
				}
			}
			if (not found) compiledBufferOwners_.pushBack(owner);
		}
	}
	return oa::Status::ok();
}

oa::Status oa::ExecutableGraph::replay(oa::Engine& inRt) {
	if (nodes_.empty()) return oa::Status::ok();
	if (owner_ != &inRt) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"graph replay engine does not own the compiled graph");
	}
	if (!compiled_ || !secondaryCb_) {
		return oa::Status::error("graph replay: not compiled — call compile() first");
	}
	if (replayTimingEnabled_ and replayTimestampReadValue_ < replayTimelineValue_) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"timed graph replay is single-flight; wait before resubmitting");
	}

	// Fast path: if we have a pre-built primary CB (from compile), submit
	// it directly with our dedicated timeline semaphore. Skips stream pool
	// acquire + Begin + vkCmdExecuteCommands + stream release entirely.
	// The primary CB uses SIMULTANEOUS_USE_BIT so it can be resubmitted
	// without re-recording.
	//
	// Non-blocking: submit and return immediately. Same-queue submissions
	// are implicitly ordered by the vulkan spec — the GPU executes them in
	// submission order without host-side waits. The caller must call
	// waitForPendingReplay() (or sync()) before reading results.
	if (primaryCb_ and primaryPool_ and replayTimelineSem_.semaphore) {
		++replayTimelineValue_;
		VkSemaphore sem = static_cast<VkSemaphore>(replayTimelineSem_.semaphore);
		VkCommandBuffer pcb = static_cast<VkCommandBuffer>(primaryCb_);

		// Use vkQueueSubmit2 (vulkan 1.3) — thinner driver path than
		// vkQueueSubmit, avoids VkSubmitInfo→internal conversion.
		VkCommandBufferSubmitInfo cbSI{};
		cbSI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
		cbSI.commandBuffer = pcb;

		VkSemaphoreSubmitInfo signalSI{};
		signalSI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
		signalSI.semaphore = sem;
		signalSI.value = replayTimelineValue_;
		signalSI.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

		VkSubmitInfo2 si2{};
		si2.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
		si2.commandBufferInfoCount = 1;
		si2.pCommandBufferInfos = &cbSI;
		si2.signalSemaphoreInfoCount = 1;
		si2.pSignalSemaphoreInfos = &signalSI;

		OA_RETURN_IF_ERROR(oa::EngineAccess(inRt).submitToQueue2(
			oa::EngineDeviceAccess::get(inRt).queues.computeQueue, &si2));
		return oa::Status::ok();
	}

	// Slow path: acquire stream, record secondary into primary, submit+wait.
	oavk::Stream* stream = oa::EngineSubmissionAccess::acquireStream(inRt);
	if (!stream) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			"graph replay: failed to acquire stream");
	}

	OA_RETURN_IF_ERROR(stream->begin(oa::EngineDeviceAccess::get(inRt)));

	VkCommandBuffer primary = static_cast<VkCommandBuffer>(stream->commandBuffer);
	VkCommandBuffer secondary = static_cast<VkCommandBuffer>(secondaryCb_);
	oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdExecuteCommands(primary, 1, &secondary);

	oa::Status status = stream->submitAndWait(inRt);
	oa::EngineSubmissionAccess::releaseStream(inRt, stream);
	return status;
}

oa::Status oa::ExecutableGraph::waitForPendingReplay(const oa::Engine& inEngine) {
	if (owner_ != nullptr and owner_ != &inEngine) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"graph replay wait engine does not own the compiled graph");
	}
	return waitForPendingReplayDevice_(oa::EngineDeviceAccess::get(inEngine));
}

oa::Status oa::ExecutableGraph::waitForPendingReplayDevice_(const oavk::Device& inDevice) {
	if (replayTimelineValue_ == 0 or not replayTimelineSem_.semaphore) {
		return oa::Status::ok();
	}
	OA_RETURN_IF_ERROR(replayTimelineSem_.wait(inDevice, replayTimelineValue_));
	if (replayTimingEnabled_ and replayTimestampReadValue_ < replayTimelineValue_) {
		OA_RETURN_IF_ERROR(replayTimestamp_.readbackDevice_(inDevice));
		lastReplayGpuMs_ = replayTimestamp_.elapsedMs(0, 1);
		replayTimestampReadValue_ = replayTimelineValue_;
	}
	return oa::Status::ok();
}

oa::Result<oa::Event> oa::ExecutableGraph::replayAsync(oa::Engine& inRt)
{
	OA_RETURN_IF_ERROR(replay(inRt));
	return lastCompletion(inRt);
}

oa::Event oa::ExecutableGraph::lastCompletion(const oa::Engine& inEngine) const
{
	if (owner_ != nullptr and owner_ != &inEngine) return {};
	return lastCompletionDevice_(oa::EngineDeviceAccess::get(inEngine));
}

oa::Event oa::ExecutableGraph::lastCompletionDevice_(const oavk::Device& inDevice) const
{
	return replayTimelineValue_ != 0U && replayTimelineSem_.semaphore
		? oa::EventAccess::create(
			inDevice, replayTimelineSem_, replayTimelineValue_, queueFamily_)
		: oa::Event();
}

oa::Status oa::ExecutableGraph::recordReplay(oa::Engine& inRt, void* inPrimaryCommandBuffer) const {
	(void)inRt;
	if (nodes_.empty()) return oa::Status::ok();
	if (!compiled_ || !secondaryCb_) {
		return oa::Status::error("graph record replay: not compiled — call compile() first");
	}
	VkCommandBuffer primary = static_cast<VkCommandBuffer>(inPrimaryCommandBuffer);
	VkCommandBuffer secondary = static_cast<VkCommandBuffer>(secondaryCb_);
	oa::EngineDeviceAccess::get(inRt).deviceDispatch.vkCmdExecuteCommands(primary, 1, &secondary);
	return oa::Status::ok();
}

// ─── phase 3: memory aliasing analysis ────────────────────────────────────────

oa::Vec<oa::BufferLifetime> oa::ExecutableGraph::computeLifetimes() const {
	// map: VkBuffer handle -> (first_access, last_access, size)
	struct LifetimeEntry {
		oa::U64 size = 0;
		oa::U32 first = UINT32_MAX;
		oa::U32 last = 0;
		oa::U32 order = 0;
	};
	oa::HashMap<void*, LifetimeEntry> map;
	oa::U32 nextOrder = 0;

	for (oa::U32 i = 0; i < static_cast<oa::U32>(nodes_.size()); ++i) {
		auto& node = nodes_[i];
		for (oa::U32 j = 0; j < static_cast<oa::U32>(node.buffers.size()); ++j) {
			void* handle = node.buffers[j].buffer;
			if (!handle) continue;
			auto emplaceResult = map.emplace(handle, LifetimeEntry{.order = nextOrder});
			if (emplaceResult.second) ++nextOrder;
			LifetimeEntry& entry = emplaceResult.first->second;
			if (i < entry.first) entry.first = i;
			if (i > entry.last) entry.last = i;
			if (node.buffers[j].size > entry.size) entry.size = node.buffers[j].size;
		}
		if (node.indirect && node.indirectBuffer.buffer) {
			void* handle = node.indirectBuffer.buffer;
			auto emplaceResult = map.emplace(handle, LifetimeEntry{.order = nextOrder});
			if (emplaceResult.second) ++nextOrder;
			LifetimeEntry& entry = emplaceResult.first->second;
			if (i < entry.first) entry.first = i;
			if (i > entry.last) entry.last = i;
			if (node.indirectBuffer.size > entry.size) {
				entry.size = node.indirectBuffer.size;
			}
		}
	}

	oa::Vec<oa::BufferLifetime> result;
	result.reserve(static_cast<oa::U32>(map.size()));
	for (auto& [handle, entry] : map) {
		oa::BufferLifetime lt;
		lt.buffer = handle;
		lt.size = entry.size;
		lt.firstAccess = entry.first;
		lt.lastAccess = entry.last;
		lt.resourceOrder = entry.order;
		result.pushBack(lt);
	}

	oa::sort(result.begin(), result.end(),
		[](const oa::BufferLifetime& a, const oa::BufferLifetime& b) {
			if (a.firstAccess != b.firstAccess) return a.firstAccess < b.firstAccess;
			return a.resourceOrder < b.resourceOrder;
		});

	return result;
}

oa::Vec<oa::AliasGroup> oa::ExecutableGraph::computeAliasGroups() const {
	auto lifetimes = computeLifetimes();
	oa::Vec<oa::AliasGroup> groups;

	// Greedy interval coloring: assign each buffer to the first group
	// where it doesn't overlap with any existing member.
	for (auto& lt : lifetimes) {
		bool placed = false;
		for (auto& group : groups) {
			bool overlaps = false;
			for (auto& member : group.members) {
				if (lt.firstAccess <= member.lastAccess
					&& lt.lastAccess >= member.firstAccess)
				{
					overlaps = true;
					break;
				}
			}
			if (!overlaps) {
				group.members.pushBack(lt);
				if (lt.size > group.requiredSize)
					group.requiredSize = lt.size;
				placed = true;
				break;
			}
		}
		if (!placed) {
			oa::AliasGroup newGroup;
			newGroup.members.pushBack(lt);
			newGroup.requiredSize = lt.size;
			groups.pushBack(oa::move(newGroup));
		}
	}

	return groups;
}

oa::Status oa::ExecutableGraph::materializeAliases(
	oa::Engine& inRt, oa::Span<oa::Matrix*> inEligible) {
	return materializeAliases(inRt, inEligible, {});
}

oa::Status oa::ExecutableGraph::materializeAliases(
	oa::Engine& inRt,
	oa::Span<oa::Matrix*> inEligible,
	oa::Span<const oa::U32> inPermittedAdditionalOwners)
{
	if (compiled_ or not aliasOwners_.empty()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"graph aliases must be materialized exactly once before compilation");
	}
	if (inEligible.empty()) return oa::Status::ok();
	OA_RETURN_IF_ERROR(bindEngine_(inRt));
	if (not inPermittedAdditionalOwners.empty()
		and inPermittedAdditionalOwners.size() != inEligible.size())
	{
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"graph alias retained-owner counts must match the eligible resources");
	}

	struct EligibleMatrix {
		oa::Matrix* matrix = nullptr;
		void* handle = nullptr;
		oa::MemoryPlacement placement = oa::MemoryPlacement::Auto;
	};
	oa::Vec<EligibleMatrix> eligible;
	eligible.reserve(inEligible.size());
	for (auto* matrix : inEligible) {
		if (matrix == nullptr
			or not oa::MatrixAccess::storageOwner(*matrix)
			or matrix->byteOffset() != 0U)
		{
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"graph alias transient must be an allocated base matrix");
		}
		const auto& storage = oa::MatrixAccess::storageOwner(*matrix);
		for (const auto& existing : eligible) {
			if (existing.handle == storage->buffer) {
				return oa::Status::error(oa::StatusCode::InvalidArgument,
					"graph alias transient list contains duplicate storage");
			}
		}
		EligibleMatrix entry;
		entry.matrix = matrix;
		entry.handle = storage->buffer;
		entry.placement = storage->placement;
		eligible.pushBack(entry);
	}

	auto isEligible = [&](void* handle) {
		for (const auto& entry : eligible) {
			if (entry.handle == handle) return true;
		}
		return false;
	};
	auto placementOf = [&](void* handle) {
		for (const auto& entry : eligible) {
			if (entry.handle == handle) return entry.placement;
		}
		return oa::MemoryPlacement::Auto;
	};
	struct PlacementAliasGroup {
		oa::AliasGroup alias;
		oa::MemoryPlacement placement = oa::MemoryPlacement::Auto;
	};
	auto lifetimes = computeLifetimes();
	oa::Vec<PlacementAliasGroup> groups;
	for (const auto& lt : lifetimes) {
		if (not isEligible(lt.buffer)) continue;
		const auto placement = placementOf(lt.buffer);
		bool placed = false;
		for (auto& group : groups) {
			if (group.placement != placement) continue;
			bool overlaps = false;
			for (const auto& member : group.alias.members) {
				if (lt.firstAccess <= member.lastAccess and lt.lastAccess >= member.firstAccess) {
					overlaps = true; break;
				}
			}
			if (not overlaps) {
				group.alias.members.pushBack(lt);
				group.alias.requiredSize = oa::max(group.alias.requiredSize, lt.size);
				placed = true; break;
			}
		}
		if (not placed) {
			PlacementAliasGroup group;
			group.placement = placement;
			group.alias.members.pushBack(lt);
			group.alias.requiredSize = lt.size;
			groups.pushBack(oa::move(group));
		}
	}

	// Exclusive ownership is the safety boundary: if another matrix/view retains
	// an eligible allocation, rebinding only the supplied matrix would leave two
	// logical values backed by different physical storage.
	for (const auto& group : groups) {
		if (group.alias.members.size() < 2U) continue;
		for (const auto& member : group.alias.members) {
			oa::Matrix* matrix = nullptr;
			oa::U32 eligibleIndex = 0;
			for (const auto& entry : eligible) {
				if (entry.handle == member.buffer) {
					matrix = entry.matrix;
					break;
				}
				++eligibleIndex;
			}
			if (matrix == nullptr or eligibleIndex >= eligible.size()) {
				return oa::Status::error(oa::StatusCode::Internal,
					"graph alias group contains an unregistered transient");
			}
			const auto& storage = oa::MatrixAccess::storageOwner(*matrix);
			oa::U32 graphOwners = 0;
			for (const auto& node : nodes_) {
				for (const auto& owner : node.bufferOwners) {
					if (owner and owner.get() == storage.get()) ++graphOwners;
				}
			}
			const oa::U32 permitted = inPermittedAdditionalOwners.empty()
				? 0U : inPermittedAdditionalOwners[eligibleIndex];
			if (storage.useCount()
				!= static_cast<long>(graphOwners + permitted + 1U))
			{
				return oa::Status::error(oa::StatusCode::FailedPrecondition,
					"graph alias transient still has an external matrix/view owner");
			}
		}
	}

	auto ownBacking = [&](oavk::Buffer&& buffer) {
		return oa::EngineAccess(inRt).adoptBufferLease(oa::move(buffer));
	};
	auto ownView = [&](oavk::Buffer&& buffer, oa::SharedPtr<oavk::Buffer> backing) {
		return oa::EngineAccess(inRt).adoptBufferLease(
			oa::move(buffer), oa::move(backing));
	};

	oa::HashMap<void*, oa::SharedPtr<oavk::Buffer>> replacements;
	oa::U64 originalBytes = 0, arenaBytes = 0;
	for (const auto& group : groups) {
		if (group.alias.members.size() < 2U) continue;
		auto placement = group.placement == oa::MemoryPlacement::Auto
			? oa::EngineResourceAccess::defaultMatrixPlacement(inRt) : group.placement;
		auto backingResult = oa::EngineAllocatorAccess::get(inRt).allocAliased(
			group.alias.requiredSize, placement);
		if (not backingResult.isOk()) {
			destroyAliasArena(); return backingResult.getStatus();
		}
		auto backingBuffer = oa::move(backingResult.getValue());
		if (oa::EngineBindlessAccess::registerBuffer(inRt, backingBuffer) == OA_BINDLESS_INVALID) {
			oa::EngineAllocatorAccess::get(inRt).free(backingBuffer);
			destroyAliasArena();
			return oa::Status::error(oa::StatusCode::ResourceExhausted,
				"graph alias backing bindless registration failed");
		}
		auto backing = ownBacking(oa::move(backingBuffer));
		if (not backing) {
			destroyAliasArena();
			return oa::Status::error(oa::StatusCode::FailedPrecondition,
				"graph alias backing lease unavailable during engine shutdown");
		}
		aliasOwners_.pushBack(backing);
		replacements.emplace(group.alias.members[0].buffer, backing);
		for (const auto& member : group.alias.members) originalBytes += member.size;
		arenaBytes += group.alias.requiredSize;
		for (oa::U32 memberIdx = 1; memberIdx < group.alias.members.size(); ++memberIdx) {
			auto aliasResult = oa::EngineAllocatorAccess::get(inRt).createAliasingBuffer(
				*backing, group.alias.members[memberIdx].size);
			if (not aliasResult.isOk()) { destroyAliasArena(); return aliasResult.getStatus(); }
			auto alias = oa::move(aliasResult.getValue());
			if (oa::EngineBindlessAccess::registerBuffer(inRt, alias) == OA_BINDLESS_INVALID) {
				oa::EngineAllocatorAccess::get(inRt).freeAlias(alias);
				destroyAliasArena();
				return oa::Status::error(oa::StatusCode::ResourceExhausted,
					"graph alias view bindless registration failed");
			}
			auto owner = ownView(oa::move(alias), backing);
			if (not owner) {
				destroyAliasArena();
				return oa::Status::error(oa::StatusCode::FailedPrecondition,
					"graph alias view lease unavailable during engine shutdown");
			}
			replacements.emplace(group.alias.members[memberIdx].buffer, owner);
			aliasOwners_.pushBack(oa::move(owner));
		}
	}
	for (auto& node : nodes_) {
		if (node.bufferOwners.empty() and not replacements.empty()) {
			node.bufferOwners.resize(node.buffers.size());
		}
		for (oa::U32 i = 0; i < node.buffers.size(); ++i) {
			auto& buffer = node.buffers[i];
			auto found = replacements.find(buffer.buffer);
			if (found != replacements.end()) {
				buffer = *found->second;
				node.bufferOwners[i] = found->second;
			}
		}
		if (node.indirect) {
			auto found = replacements.find(node.indirectBuffer.buffer);
			if (found != replacements.end()) node.indirectBuffer = *found->second;
		}
	}
	for (const auto& entry : eligible) {
		auto found = replacements.find(entry.handle);
		if (found == replacements.end()) continue;
		// These allocations are being retired into an alias arena. Bypass the
		// general host-visible reuse cache so materialization reduces physical
		// allocation bytes on unified GPUs as well as discrete GPUs.
		auto& storage = oa::MatrixAccess::storageOwner(*entry.matrix);
		storage->flags |= OA_VK_BUFFER_FLAG_TRANSIENT;
		storage = found->second;
		oa::MatrixAccess::hostOwner(*entry.matrix).reset();
		oa::MatrixAccess::syncDescriptor(*entry.matrix);
	}
	materializedAliasSavings_ = originalBytes > arenaBytes ? originalBytes - arenaBytes : 0U;
	return oa::Status::ok();
}

void oa::ExecutableGraph::destroyAliasArena() {
	aliasOwners_.clear();
	materializedAliasSavings_ = 0;
}

// ─── Queries ──────────────────────────────────────────────────────────────────

oa::GraphStats oa::ExecutableGraph::getStats() const {
	oa::GraphStats stats;
	stats.dispatchCount = static_cast<oa::U32>(nodes_.size());
	stats.barrierCount = barrierCount_;
	stats.descriptorSetCount = static_cast<oa::U32>(descriptorPools_.size());
	stats.warBarrierCount = warBarrierCount_;
	stats.indirectBarrierCount = indirectBarrierCount_;
	stats.aliasBarrierCount = aliasBarrierCount_;
	stats.hostBarrierCount = hostReadbackRequired_ && not nodes_.empty() ? 1U : 0U;
	for (const auto& node : nodes_) {
		if (node.kernelSelection == oa::KernelSelectionKind::Unspecified) continue;
		++stats.kernelSelectionCount;
		switch (node.kernelSelection) {
		case oa::KernelSelectionKind::PrecisionFallback:
			++stats.kernelFallbackCount;
			++stats.precisionFallbackCount;
			break;
		case oa::KernelSelectionKind::LayoutFallback:
			++stats.kernelFallbackCount;
			++stats.layoutFallbackCount;
			break;
		case oa::KernelSelectionKind::NaiveFallback:
			++stats.kernelFallbackCount;
			++stats.naiveFallbackCount;
			break;
		case oa::KernelSelectionKind::Unspecified:
		case oa::KernelSelectionKind::Direct:
			break;
		}
	}

	// Compute total buffer bytes and alias savings
	auto lifetimes = computeLifetimes();
	for (auto& lt : lifetimes) {
		stats.totalBufferBytes += lt.size;
	}

	auto groups = computeAliasGroups();
	oa::U64 aliasedTotal = 0;
	for (auto& g : groups) {
		aliasedTotal += g.requiredSize;
	}
	if (stats.totalBufferBytes > aliasedTotal) {
		stats.potentialAliasSavings = stats.totalBufferBytes - aliasedTotal;
	}

	return stats;
}

oa::String oa::ExecutableGraph::debugReportJson(oa::StringView inName) const {
	const auto stats = getStats();
	const auto lifetimes = computeLifetimes();
	const auto aliasGroups = computeAliasGroups();
	oa::HashMap<void*, oa::U32> resourceIds;
	for (const auto& lifetime : lifetimes) {
		resourceIds.emplace(lifetime.buffer, lifetime.resourceOrder);
	}
	oa::HashMap<void*, oa::U32> hazardDomainIds;
	oa::HashMap<void*, oa::U32> resourceHazardDomains;
	oa::U32 nextHazardDomain = 0;
	auto registerHazardDomain = [&](const oavk::Buffer& buffer) {
		if (not buffer.buffer) return;
		void* identity = buffer.synchronizationIdentity();
		auto [domain, inserted] = hazardDomainIds.emplace(identity, nextHazardDomain);
		if (inserted) ++nextHazardDomain;
		resourceHazardDomains.emplace(buffer.buffer, domain->second);
	};
	for (const auto& node : nodes_) {
		for (const auto& buffer : node.buffers) registerHazardDomain(buffer);
		if (node.indirect) registerHazardDomain(node.indirectBuffer);
	}

	oa::HashMap<void*, GraphBufferState> bufferStates;
	oa::Vec<VkBufferMemoryBarrier2> bufferBarriers;
	oa::Vec<VkMemoryBarrier2> aliasBarriers;
	oa::Vec<GraphBarrierDebug> bufferDebug;
	oa::Vec<GraphBarrierDebug> aliasDebug;
	oa::Vec<GraphBarrierDebug> plannedBarriers;
	for (oa::U32 nodeIdx = 0; nodeIdx < nodes_.size(); ++nodeIdx) {
		const auto& node = nodes_[nodeIdx];
		computeNodeBarriers(node, nodeIdx, bufferStates, bufferBarriers,
			aliasBarriers, &bufferDebug, &aliasDebug);
		pruneRedundantWarBarriers(bufferBarriers, &bufferDebug);
		for (const auto& debug : bufferDebug) plannedBarriers.pushBack(debug);
		for (const auto& debug : aliasDebug) plannedBarriers.pushBack(debug);
		updateBufferStates(node, nodeIdx, bufferStates);
	}

	oa::internal::JsonWriter out;
	out << "{\n  \"schema\": \"oa.execution_graph.v3\",\n  \"name\": ";
	writeJsonString(out, inName);
	out << ",\n  \"compiled\": " << (compiled_ ? "true" : "false")
		<< ",\n  \"completion\": {\"submitted\": "
		<< (replayTimelineValue_ != 0U ? "true" : "false")
		<< ", \"timeline_value\": " << replayTimelineValue_ << "},\n"
		<< "  \"stats\": {\"dispatches\": " << stats.dispatchCount
		<< ", \"barriers\": " << stats.barrierCount
		<< ", \"war_barriers\": " << stats.warBarrierCount
		<< ", \"indirect_barriers\": " << stats.indirectBarrierCount
		<< ", \"alias_barriers\": " << stats.aliasBarrierCount
		<< ", \"host_barriers\": " << stats.hostBarrierCount
		<< ", \"descriptor_sets\": " << stats.descriptorSetCount
		<< ", \"kernel_selections\": " << stats.kernelSelectionCount
		<< ", \"kernel_fallbacks\": " << stats.kernelFallbackCount
		<< ", \"precision_fallbacks\": " << stats.precisionFallbackCount
		<< ", \"layout_fallbacks\": " << stats.layoutFallbackCount
		<< ", \"naive_fallbacks\": " << stats.naiveFallbackCount
		<< ", \"buffer_bytes\": " << stats.totalBufferBytes
		<< ", \"potential_alias_savings\": " << stats.potentialAliasSavings
		<< ", \"materialized_alias_savings\": " << materializedAliasSavings_
		<< "},\n  \"resources\": [";

	for (oa::U32 i = 0; i < lifetimes.size(); ++i) {
		const auto& lifetime = lifetimes[i];
		out << (i == 0 ? "\n" : ",\n")
			<< "    {\"id\": " << lifetime.resourceOrder
			<< ", \"hazard_domain\": "
			<< resourceHazardDomains.at(lifetime.buffer)
			<< ", \"bytes\": " << lifetime.size
			<< ", \"first_node\": " << lifetime.firstAccess
			<< ", \"last_node\": " << lifetime.lastAccess << "}";
	}
	if (not lifetimes.empty()) out << '\n';
	out << "  ],\n  \"alias_groups\": [";
	for (oa::U32 i = 0; i < aliasGroups.size(); ++i) {
		const auto& group = aliasGroups[i];
		out << (i == 0 ? "\n" : ",\n")
			<< "    {\"required_bytes\": " << group.requiredSize << ", \"resources\": [";
		for (oa::U32 j = 0; j < group.members.size(); ++j) {
			const auto found = resourceIds.find(group.members[j].buffer);
			if (j != 0) out << ", ";
			out << (found != resourceIds.end() ? found->second : UINT32_MAX);
		}
		out << "]}";
	}
	if (not aliasGroups.empty()) out << '\n';
	out << "  ],\n  \"barriers\": [";
	for (oa::U32 i = 0; i < plannedBarriers.size(); ++i) {
		const auto& barrier = plannedBarriers[i];
		out << (i == 0 ? "\n" : ",\n")
			<< "    {\"reason\": \"" << hazardName(barrier.hazard)
			<< "\", \"scope\": \""
			<< (barrier.alias ? "memory_alias" : "buffer")
			<< "\", \"source_nodes\": [" << barrier.sourceFirstNode
			<< ", " << barrier.sourceLastNode << "]"
			<< ", \"destination_node\": " << barrier.destinationNode
			<< ", \"source_resource\": ";
		auto sourceResource = resourceIds.find(barrier.sourceBuffer);
		if (sourceResource == resourceIds.end()) out << "null";
		else out << sourceResource->second;
		out << ", \"destination_resource\": ";
		auto destinationResource = resourceIds.find(barrier.destinationBuffer);
		if (destinationResource == resourceIds.end()) out << "null";
		else out << destinationResource->second;
		out << ", \"hazard_domain\": ";
		auto hazardDomain = hazardDomainIds.find(barrier.hazardIdentity);
		if (hazardDomain == hazardDomainIds.end()) out << "null";
		else out << hazardDomain->second;
		out << ", \"range\": {\"offset\": 0, \"bytes\": "
			<< barrier.bytes << "}"
			<< ", \"source_queue\": \""
			<< (barrier.sourceQueuesMixed ? "mixed" : queueHintName(barrier.sourceQueue))
			<< "\", \"destination_queue\": \""
			<< queueHintName(barrier.destinationQueue) << "\""
			<< ", \"cross_queue\": "
			<< (barrier.sourceQueuesMixed
				or barrier.sourceQueue != barrier.destinationQueue ? "true" : "false")
			<< ", \"ownership_transfer\": false"
			<< ", \"source_stage_mask\": ";
		writeHexId(out, barrier.sourceStages);
		out << ", \"source_stages\": ";
		writeStageNames(out, barrier.sourceStages);
		out << ", \"source_access_mask\": ";
		writeHexId(out, barrier.sourceAccess);
		out << ", \"source_accesses\": ";
		writeAccessNames(out, barrier.sourceAccess);
		out << ", \"destination_stage_mask\": ";
		writeHexId(out, barrier.destinationStages);
		out << ", \"destination_stages\": ";
		writeStageNames(out, barrier.destinationStages);
		out << ", \"destination_access_mask\": ";
		writeHexId(out, barrier.destinationAccess);
		out << ", \"destination_accesses\": ";
		writeAccessNames(out, barrier.destinationAccess);
		out << '}';
	}
	if (hostReadbackRequired_ and not nodes_.empty()) {
		out << (plannedBarriers.empty() ? "\n" : ",\n")
			<< "    {\"reason\": \"host_readback\", \"scope\": \"memory\""
			<< ", \"source_nodes\": [" << (nodes_.size() - 1U) << ", "
			<< (nodes_.size() - 1U) << "]"
			<< ", \"destination_node\": null, \"source_resource\": null"
			<< ", \"destination_resource\": null, \"hazard_domain\": null"
			<< ", \"range\": null, \"source_queue\": \"compute\""
			<< ", \"destination_queue\": \"host\", \"cross_queue\": true"
			<< ", \"ownership_transfer\": false, \"source_stage_mask\": ";
		writeHexId(out, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
		out << ", \"source_stages\": ";
		writeStageNames(out, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
		out << ", \"source_access_mask\": ";
		writeHexId(out, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
		out << ", \"source_accesses\": ";
		writeAccessNames(out, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
		out << ", \"destination_stage_mask\": ";
		writeHexId(out, VK_PIPELINE_STAGE_2_HOST_BIT);
		out << ", \"destination_stages\": ";
		writeStageNames(out, VK_PIPELINE_STAGE_2_HOST_BIT);
		out << ", \"destination_access_mask\": ";
		writeHexId(out, VK_ACCESS_2_HOST_READ_BIT);
		out << ", \"destination_accesses\": ";
		writeAccessNames(out, VK_ACCESS_2_HOST_READ_BIT);
		out << '}';
	}
	if (not plannedBarriers.empty() or (hostReadbackRequired_ and not nodes_.empty())) {
		out << '\n';
	}
	out << "  ],\n  \"nodes\": [";

	for (oa::U32 i = 0; i < nodes_.size(); ++i) {
		const auto& node = nodes_[i];
		out << (i == 0 ? "\n" : ",\n") << "    {\"index\": " << i
			<< ", \"operation\": ";
		if (node.operation.empty()) out << "null";
		else writeJsonString(out, node.operation);
		out << ", \"semantic_operations\": [";
		for (oa::U32 operation = 0;
			operation < node.semanticOps.size(); ++operation)
		{
			if (operation != 0U) out << ", ";
			out << node.semanticOps[operation];
		}
		out << ']';
		out << ", \"implementation_id\": ";
		if (node.implementationId == 0U) out << "null";
		else writeHexId(out, node.implementationId);
		out << ", \"operation_contract_hash\": ";
		if (node.opContractHash == 0U) out << "null";
		else writeHexId(out, node.opContractHash);
		out << ", \"problem_contract_hash\": ";
		if (node.problemContractHash == 0U) out << "null";
		else writeHexId(out, node.problemContractHash);
		out << ", \"kernel_content_hash\": ";
		if (node.kernelContentHash == 0U) out << "null";
		else writeHexId(out, node.kernelContentHash);
		out << ", \"kernel_selection\": ";
		if (node.kernelSelection == oa::KernelSelectionKind::Unspecified) out << "null";
		else writeJsonString(out, kernelSelectionName(node.kernelSelection));
		out << ", \"kernel\": ";
		writeJsonString(out, node.shader);
		out << ", \"dtype_class\": " << node.dtype
			<< ", \"groups\": [" << node.groupsX << ", " << node.groupsY
			<< ", " << node.groupsZ << "]"
			<< ", \"queue\": \"" << queueHintName(node.queue) << "\""
			<< ", \"indirect\": " << (node.indirect ? "true" : "false")
			<< ", \"effects\": [";
		for (oa::U32 j = 0; j < node.buffers.size(); ++j) {
			if (j != 0) out << ", ";
			out << "{\"resource\": ";
			const auto found = resourceIds.find(node.buffers[j].buffer);
			if (found == resourceIds.end()) out << "null";
			else out << found->second;
			out << ", \"access\": \"" << bufferAccessName(node.access[j]) << "\"}";
		}
		out << "]}";
	}
	if (not nodes_.empty()) out << '\n';
	out << "  ]\n}\n";
	return out.take();
}

// ─── Lifecycle ────────────────────────────────────────────────────────────────

oa::ExecutableGraph::~ExecutableGraph() {
	release_();
}

oa::Status oa::ExecutableGraph::bindEngine_(oa::Engine& inEngine) {
	if (owner_ != nullptr and owner_ != &inEngine) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"compute graph resources cannot cross engine ownership");
	}
	owner_ = &inEngine;
	return oa::Status::ok();
}

bool oa::ExecutableGraph::hasDeviceState_() const noexcept {
	return secondaryPool_ != nullptr or secondaryCb_ != nullptr
		or primaryPool_ != nullptr or primaryCb_ != nullptr
		or replayTimelineSem_.semaphore != nullptr
		or replayTimestamp_.isInitialized()
		or not descriptorPools_.empty();
}

void oa::ExecutableGraph::swapState_(oa::ExecutableGraph& inOther) noexcept {
	oa::swapValues(nodes_, inOther.nodes_);
	oa::swapValues(owner_, inOther.owner_);
	oa::swapValues(secondaryPool_, inOther.secondaryPool_);
	oa::swapValues(secondaryCb_, inOther.secondaryCb_);
	oa::swapValues(primaryPool_, inOther.primaryPool_);
	oa::swapValues(primaryCb_, inOther.primaryCb_);
	oa::swapValues(replayTimelineSem_, inOther.replayTimelineSem_);
	oa::swapValues(replayTimelineValue_, inOther.replayTimelineValue_);
	oa::swapValues(replayTimestamp_, inOther.replayTimestamp_);
	oa::swapValues(replayTimestampReadValue_, inOther.replayTimestampReadValue_);
	oa::swapValues(lastReplayGpuMs_, inOther.lastReplayGpuMs_);
	oa::swapValues(replayTimingEnabled_, inOther.replayTimingEnabled_);
	oa::swapValues(descriptorPools_, inOther.descriptorPools_);
	oa::swapValues(queueFamily_, inOther.queueFamily_);
	oa::swapValues(compiled_, inOther.compiled_);
	oa::swapValues(lastCompileHash_, inOther.lastCompileHash_);
	oa::swapValues(lastCompileReused_, inOther.lastCompileReused_);
	oa::swapValues(compiledBufferOwners_, inOther.compiledBufferOwners_);
	oa::swapValues(barrierCount_, inOther.barrierCount_);
	oa::swapValues(warBarrierCount_, inOther.warBarrierCount_);
	oa::swapValues(indirectBarrierCount_, inOther.indirectBarrierCount_);
	oa::swapValues(aliasBarrierCount_, inOther.aliasBarrierCount_);
	oa::swapValues(hostReadbackRequired_, inOther.hostReadbackRequired_);
	oa::swapValues(aliasOwners_, inOther.aliasOwners_);
	oa::swapValues(materializedAliasSavings_, inOther.materializedAliasSavings_);
}

void oa::ExecutableGraph::release_() noexcept {
	if (not hasDeviceState_()) {
		destroyAliasArena();
		owner_ = nullptr;
		return;
	}

	oa::Engine* owner = owner_;
	if (owner == nullptr) {
		OaLogError(oa::LogComponent::Compute,
			"compute graph lost its engine owner; device resources retained");
		return;
	}

	const auto state = owner->getState();
	if (state != oa::EngineState::Empty and state != oa::EngineState::Destroyed) {
		const auto& device = oa::EngineDeviceAccess::get(*owner);
		const auto completion = lastCompletionDevice_(device);
		if (completion.isValid() and not completion.isComplete()
			and state != oa::EngineState::Destroying)
		{
			oa::UniquePtr<oa::ExecutableGraph> retired(new oa::ExecutableGraph());
			retired->swapState_(*this);
			oa::EngineAccess(*owner).retireExecutionPlan(oa::move(retired));
			return;
		}
		if (not completion.isValid() or completion.isComplete()) {
			destroyDevice_(device);
			return;
		}
	}

	// The device is already gone, or shutdown is already committed and a late
	// external graph still names unfinished work. Never wait from destruction;
	// device teardown owns the last-resort vulkan reclamation in this state.
	secondaryPool_ = nullptr;
	secondaryCb_ = nullptr;
	primaryPool_ = nullptr;
	primaryCb_ = nullptr;
	replayTimelineSem_.semaphore = nullptr;
	replayTimelineValue_ = 0;
	replayTimestamp_ = {};
	descriptorPools_.clear();
	compiledBufferOwners_.clear();
	destroyAliasArena();
	owner_ = nullptr;
}

oa::Status oa::ExecutableGraph::copyNodesFrom(const oa::ExecutableGraph& inSource) {
	if (this == &inSource) {
		return oa::Status::invalidArgument(
			"oa::ExecutableGraph::copyNodesFrom cannot copy a graph onto itself");
	}
	if (compiled_ or secondaryPool_ != nullptr or primaryPool_ != nullptr
		or not descriptorPools_.empty() or replayTimelineSem_.semaphore)
	{
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::ExecutableGraph::copyNodesFrom requires a fresh or destroyed destination graph");
	}
	nodes_ = inSource.nodes_;
	hostReadbackRequired_ = inSource.hostReadbackRequired_;
	replayTimingEnabled_ = inSource.replayTimingEnabled_;
	lastCompileHash_ = 0;
	lastCompileReused_ = false;
	barrierCount_ = 0;
	warBarrierCount_ = 0;
	indirectBarrierCount_ = 0;
	aliasBarrierCount_ = 0;
	return oa::Status::ok();
}

void oa::ExecutableGraph::invalidate(const oa::Engine& inEngine) {
	if (owner_ != nullptr and owner_ != &inEngine) return;
	invalidateDevice_(oa::EngineDeviceAccess::get(inEngine));
}

void oa::ExecutableGraph::invalidateDevice_(const oavk::Device& inDevice) {
	VkDevice dev = static_cast<VkDevice>(inDevice.device);

	for (void* pool : descriptorPools_) {
		inDevice.deviceDispatch.vkDestroyDescriptorPool(
			dev, static_cast<VkDescriptorPool>(pool), nullptr);
	}
	descriptorPools_.clear();

	if (secondaryPool_) {
		// Explicitly free secondary command buffer before destroying pool (vulkan validation requirement)
		if (secondaryCb_) {
			VkCommandBuffer scb = static_cast<VkCommandBuffer>(secondaryCb_);
			inDevice.deviceDispatch.vkFreeCommandBuffers(
				dev, static_cast<VkCommandPool>(secondaryPool_), 1, &scb);
			secondaryCb_ = nullptr;
		}
		inDevice.deviceDispatch.vkDestroyCommandPool(
			dev, static_cast<VkCommandPool>(secondaryPool_), nullptr);
		secondaryPool_ = nullptr;
	}

	if (primaryPool_) {
		if (primaryCb_) {
			VkCommandBuffer pcb = static_cast<VkCommandBuffer>(primaryCb_);
			inDevice.deviceDispatch.vkFreeCommandBuffers(
				dev, static_cast<VkCommandPool>(primaryPool_), 1, &pcb);
			primaryCb_ = nullptr;
		}
		inDevice.deviceDispatch.vkDestroyCommandPool(
			dev, static_cast<VkCommandPool>(primaryPool_), nullptr);
		primaryPool_ = nullptr;
	}

	if (replayTimelineSem_.semaphore) {
		replayTimelineSem_.destroy(inDevice);
	}
	replayTimelineValue_ = 0;
	if (replayTimestamp_.pool) replayTimestamp_.destroyDevice_(inDevice);
	replayTimestampReadValue_ = 0;
	lastReplayGpuMs_ = 0.0;

	compiled_ = false;
	lastCompileHash_ = 0;
	lastCompileReused_ = false;
	compiledBufferOwners_.clear();
	barrierCount_ = 0;
	warBarrierCount_ = 0;
	indirectBarrierCount_ = 0;
	aliasBarrierCount_ = 0;
}

void oa::ExecutableGraph::reset() {
	if (hasDeviceState_()) {
		OaLogError(oa::LogComponent::Compute,
			"compute graph reset() requires its owning engine while device state exists");
		return;
	}
	destroyAliasArena();
	owner_ = nullptr;
	nodes_.clear();
}

void oa::ExecutableGraph::reset(const oa::Engine& inEngine) {
	if (owner_ != nullptr and owner_ != &inEngine) return;
	resetDevice_(oa::EngineDeviceAccess::get(inEngine));
}

void oa::ExecutableGraph::resetDevice_(const oavk::Device& inDevice) {
	invalidateDevice_(inDevice);
	destroyAliasArena();
	nodes_.clear();
	owner_ = nullptr;
}

void oa::ExecutableGraph::clearNodes() {
	nodes_.clear();
	compiled_ = false;
	lastCompileReused_ = false;
	barrierCount_ = 0;
	warBarrierCount_ = 0;
	indirectBarrierCount_ = 0;
	aliasBarrierCount_ = 0;
	// Keep secondaryPool_, secondaryCb_ for reuse in next compile().
	// descriptorPools_ are cleaned up at the start of compile().
}

void oa::ExecutableGraph::releaseCompletedBufferOwners() {
	compiledBufferOwners_.clear();
}

void oa::ExecutableGraph::destroyDevice_(const oavk::Device& inDevice) {
	invalidateDevice_(inDevice);
	destroyAliasArena();
	nodes_.clear();
	nodes_.shrinkToFit();
	owner_ = nullptr;
}
