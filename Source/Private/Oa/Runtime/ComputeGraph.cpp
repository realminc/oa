#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Bindless.h>
#include <Oa/Runtime/Topology.h>
#include <Oa/Runtime/Scheduler.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Runtime/Pipeline.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Validation.h>
#include <Oa/Runtime/Gemm/Router.h>

#include <algorithm>
#include <cstring>

struct OaGraphBufferState {
	VkPipelineStageFlags2 StageMask = 0;
	VkAccessFlags2 AccessMask = 0;
	bool Writes = false;
};

static OaU32 ComputeNodeBarriers(
	const OaComputeNode& InNode,
	const OaHashMap<void*, OaGraphBufferState>& InState,
	OaVec<VkBufferMemoryBarrier2>& OutBarriers
);
static OaU32 PruneRedundantWarBarriers(
	OaVec<VkBufferMemoryBarrier2>& InOutBarriers
);
static void UpdateBufferStates(
	const OaComputeNode& InNode,
	OaHashMap<void*, OaGraphBufferState>& InOutState
);
static void RecordFinalBarrier(VkCommandBuffer InCb);
static OaStatus AllocGraphDescriptorSet(
	const OaVkDevice& InDevice,
	OaComputePipeline& InPipeline,
	OaSpan<OaVkBuffer> InBuffers,
	void** OutPool,
	void** OutSet
);
static OaComputeNode MakeComputeNode(
	OaStringView InShader,
	OaSpan<OaVkBuffer> InBuffers,
	OaSpan<OaSharedPtr<OaVkBuffer>> InBufferOwners,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush, OaU32 InPushSize,
	OaU32 InGroupsX, OaU32 InGroupsY, OaU32 InGroupsZ
);

void OaComputeGraph::Add(
	OaStringView InShader,
	OaSpan<OaVkBuffer> InBuffers,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush, OaU32 InPushSize,
	OaU32 InGroupsX, OaU32 InGroupsY, OaU32 InGroupsZ
) {
	Add(InShader, InBuffers, OaSpan<OaSharedPtr<OaVkBuffer>>{}, InAccess,
		InPush, InPushSize, InGroupsX, InGroupsY, InGroupsZ);
}

void OaComputeGraph::Add(
	OaStringView InShader,
	OaSpan<OaVkBuffer> InBuffers,
	OaSpan<OaSharedPtr<OaVkBuffer>> InBufferOwners,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush, OaU32 InPushSize,
	OaU32 InGroupsX, OaU32 InGroupsY, OaU32 InGroupsZ
) {
	Nodes_.PushBack(MakeComputeNode(
		InShader, InBuffers, InBufferOwners, InAccess,
		InPush, InPushSize, InGroupsX, InGroupsY, InGroupsZ));
}

void OaComputeGraph::Add(
	OaStringView InShader,
	OaSpan<OaVkBuffer> InBuffers,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush, OaU32 InPushSize,
	OaU32 InGroupsX, OaU32 InGroupsY, OaU32 InGroupsZ,
	OaQueueHint InQueue
) {
	Add(InShader, InBuffers, OaSpan<OaSharedPtr<OaVkBuffer>>{}, InAccess,
		InPush, InPushSize, InGroupsX, InGroupsY, InGroupsZ, InQueue);
}

void OaComputeGraph::Add(
	OaStringView InShader,
	OaSpan<OaVkBuffer> InBuffers,
	OaSpan<OaSharedPtr<OaVkBuffer>> InBufferOwners,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush, OaU32 InPushSize,
	OaU32 InGroupsX, OaU32 InGroupsY, OaU32 InGroupsZ,
	OaQueueHint InQueue
) {
	OaComputeNode node = MakeComputeNode(
		InShader, InBuffers, InBufferOwners, InAccess,
		InPush, InPushSize, InGroupsX, InGroupsY, InGroupsZ);
	node.Queue = InQueue;
	Nodes_.PushBack(std::move(node));
}

void OaComputeGraph::Add(
	OaStringView InShader,
	OaSpan<OaVkBuffer> InBuffers,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush, OaU32 InPushSize,
	OaU32 InGroupsX, OaU32 InGroupsY, OaU32 InGroupsZ,
	OaU32 InNodeIndex
) {
	Add(InShader, InBuffers, OaSpan<OaSharedPtr<OaVkBuffer>>{}, InAccess,
		InPush, InPushSize, InGroupsX, InGroupsY, InGroupsZ, InNodeIndex);
}

void OaComputeGraph::Add(
	OaStringView InShader,
	OaSpan<OaVkBuffer> InBuffers,
	OaSpan<OaSharedPtr<OaVkBuffer>> InBufferOwners,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush, OaU32 InPushSize,
	OaU32 InGroupsX, OaU32 InGroupsY, OaU32 InGroupsZ,
	OaU32 InNodeIndex
) {
	OaComputeNode node = MakeComputeNode(
		InShader, InBuffers, InBufferOwners, InAccess,
		InPush, InPushSize, InGroupsX, InGroupsY, InGroupsZ);
	node.NodeIndex = InNodeIndex;
	Nodes_.PushBack(std::move(node));
}

void OaComputeGraph::AddIndirect(
	OaStringView InShader,
	OaSpan<OaVkBuffer> InBuffers,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush, OaU32 InPushSize,
	const OaVkBuffer& InIndirectBuffer, OaU64 InOffset
) {
	AddIndirect(InShader, InBuffers, OaSpan<OaSharedPtr<OaVkBuffer>>{},
		InAccess, InPush, InPushSize, InIndirectBuffer, InOffset);
}

void OaComputeGraph::AddIndirect(
	OaStringView InShader,
	OaSpan<OaVkBuffer> InBuffers,
	OaSpan<OaSharedPtr<OaVkBuffer>> InBufferOwners,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush, OaU32 InPushSize,
	const OaVkBuffer& InIndirectBuffer, OaU64 InOffset
) {
	OaComputeNode node = MakeComputeNode(
		InShader, InBuffers, InBufferOwners, InAccess,
		InPush, InPushSize, 1, 1, 1);
	node.IndirectBuffer = InIndirectBuffer;
	node.IndirectOffset = InOffset;
	node.Indirect = true;
	Nodes_.PushBack(std::move(node));
}

static OaComputeNode MakeComputeNode(
	OaStringView InShader,
	OaSpan<OaVkBuffer> InBuffers,
	OaSpan<OaSharedPtr<OaVkBuffer>> InBufferOwners,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush, OaU32 InPushSize,
	OaU32 InGroupsX, OaU32 InGroupsY, OaU32 InGroupsZ
) {
	OaComputeNode node;
	node.Shader = OaString(InShader);
	node.Buffers.Assign(InBuffers.begin(), InBuffers.end());
	node.BufferOwners.Assign(InBufferOwners.begin(), InBufferOwners.end());
	node.Access.Assign(InAccess.begin(), InAccess.end());
	node.GroupsX = InGroupsX;
	node.GroupsY = InGroupsY;
	node.GroupsZ = InGroupsZ;
	if (InPush && InPushSize > 0) {
		OaU32 clampedSize = InPushSize > OA_VK_MAX_PUSH_CONSTANT_BYTES
			? OA_VK_MAX_PUSH_CONSTANT_BYTES
			: InPushSize;
		std::memcpy(node.PushData, InPush, clampedSize);
		node.PushSize = clampedSize;
	}
	return node;
}

// ─── AddGemm - GEMM kernel selection for graph API ───────────────────────────

void OaComputeGraph::AddGemm(
	OaComputeEngine& InRt,
	OaVkBuffer InA,
	OaVkBuffer InB,
	OaVkBuffer OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
) {
	const auto route = OaGemmRouter::Select(InRt, InM, InN, InK, OaGemmPrecision::Auto);

	if (route.Path == OaGemmPath::CoopVec) {
		// CoopVec GEMV: different push constant layout
		struct PushCoopVec { OaU32 N; OaU32 K; };
		PushCoopVec push{ InN, InK };
		
		OaVkBuffer bufs[] = { InA, InB, OutC };
		OaBufferAccess access[] = { OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write };
		
		Add(route.KernelName, bufs, access, &push, sizeof(push), route.Gx, route.Gy, 1);
		return;
	}
	
	// Standard path
	struct Push { OaU32 M; OaU32 N; OaU32 K; };
	Push push{ InM, InN, InK };

	OaVkBuffer bufs[] = { InA, InB, OutC };
	OaBufferAccess access[] = { OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write };

	Add(route.KernelName, bufs, access, &push, sizeof(push), route.Gx, route.Gy, 1);
}

OaStatus OaComputeGraph::ExecuteDistributed(OaComputeEngine& InRt) {
	if (!InRt.IsMultiDevice()) {
		return Execute(InRt);
	}

	if (Nodes_.Empty()) return OaStatus::Ok();

	OaU32 meshNodeCount = InRt.DeviceCount();

	// Group compute nodes by their target device
	OaVec<OaVec<OaU32>> nodeGroups(meshNodeCount);
	for (OaU32 i = 0; i < static_cast<OaU32>(Nodes_.Size()); ++i) {
		OaU32 target = Nodes_[i].NodeIndex;
		if (target >= meshNodeCount) target = 0;
		nodeGroups[target].PushBack(i);
	}

	// Acquire a stream per device that has work
	struct PerDeviceBatch {
		OaU32 NodeIdx;
		OaVkStream* Stream;
		bool HasWork;
	};
	OaVec<PerDeviceBatch> batches(meshNodeCount);
	for (OaU32 d = 0; d < meshNodeCount; ++d) {
		batches[d].NodeIdx = d;
		batches[d].Stream = nullptr;
		batches[d].HasWork = !nodeGroups[d].Empty();
		if (batches[d].HasWork) {
			batches[d].Stream = InRt.AcquireStreamOn(d);
			if (!batches[d].Stream) {
				return OaStatus::Error(OaStatusCode::VulkanError,
					"distributed exec: failed to acquire stream on node");
			}
		}
	}

	// Record dispatches per device. Each device gets its own command buffer
	// with independent barriers.
	for (OaU32 d = 0; d < meshNodeCount; ++d) {
		if (!batches[d].HasWork) continue;

		auto* devNode = InRt.GetNode(d);
		if (!devNode) continue;

		OaVkStream* stream = batches[d].Stream;

		// For non-primary devices, temporarily switch global function pointers
		if (d != 0) {
			OaSpinlockGuard guard(*InRt.GetMesh()->DeviceLoadLock);
			OaVkLoadDevice(static_cast<VkDevice>(devNode->Device.Device));
			OA_RETURN_IF_ERROR(stream->Begin(devNode->Device));
			OaVkLoadDevice(static_cast<VkDevice>(InRt.Device.Device));
		} else {
			OA_RETURN_IF_ERROR(stream->Begin(devNode->Device));
		}

		OaHashMap<void*, OaGraphBufferState> bufferStates;
		OaVec<VkBufferMemoryBarrier2> barriers;
		barriers.Reserve(8);

		for (OaU32 idx : nodeGroups[d]) {
			auto& node = Nodes_[idx];
			ComputeNodeBarriers(node, bufferStates, barriers);
			PruneRedundantWarBarriers(barriers);

			if (!barriers.Empty()) {
				VkDependencyInfo dep{};
				dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
				dep.bufferMemoryBarrierCount = static_cast<OaU32>(barriers.Size());
				dep.pBufferMemoryBarriers = barriers.Data();
				vkCmdPipelineBarrier2(
					static_cast<VkCommandBuffer>(stream->CommandBuffer), &dep);
			}

			// Resolve pipeline from the target node's pipeline registry. DTYPE comes from
			// the node (derived from its operand tensors at record time), not a global mode.
			auto& pipeline = (d == 0)
				? InRt.Pipelines.GetPipeline(node.Shader, node.Dtype)
				: devNode->Pipelines.GetPipeline(node.Shader, node.Dtype);

			if (!pipeline.Pipeline) {
				return OaStatus::Error("distributed exec: pipeline not found: " + node.Shader);
			}

			VkCommandBuffer cb = static_cast<VkCommandBuffer>(stream->CommandBuffer);

			void* dsPool = nullptr;
			void* dsSet = nullptr;
			if (!pipeline.Bindless) {
				OaStatus dsStatus = AllocGraphDescriptorSet(
					devNode->Device, pipeline, node.Buffers.Span(), &dsPool, &dsSet);
				if (!dsStatus.IsOk()) return dsStatus;
				DescriptorPools_.PushBack(dsPool);
			}

			vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(pipeline.Pipeline));

			if (pipeline.Bindless) {
				auto& bindless = (d == 0) ? InRt.Bindless : devNode->Bindless;
				VkDescriptorSet bds = static_cast<VkDescriptorSet>(bindless.DescriptorSet);
				vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
					static_cast<VkPipelineLayout>(pipeline.PipelineLayout),
					0, 1, &bds, 0, nullptr);

				OaU32 numBufs = static_cast<OaU32>(node.Buffers.Size());
				if (!OaVkBindlessPushFits(numBufs, node.PushSize)) {
					return OaStatus::Error(OaStatusCode::InvalidArgument,
						"distributed exec: bindless push exceeds OA_VK_MAX_PUSH_CONSTANT_BYTES "
						"(buffer index header + user push)");
				}
				OaU32 headerBytes = numBufs * sizeof(OaU32);
				OaU32 totalPush = headerBytes + node.PushSize;
				alignas(16) OaU8 pushBuf[OA_VK_MAX_PUSH_CONSTANT_BYTES] = {};
				OaU32* indices = reinterpret_cast<OaU32*>(pushBuf);
				for (OaU32 b = 0; b < numBufs; ++b) {
					indices[b] = node.Buffers[b].BindlessIndex;
				}
				if (node.PushSize > 0) {
					std::memcpy(pushBuf + headerBytes, node.PushData, node.PushSize);
				}
				vkCmdPushConstants(cb,
					static_cast<VkPipelineLayout>(pipeline.PipelineLayout),
					VK_SHADER_STAGE_COMPUTE_BIT, 0, totalPush, pushBuf);
			} else {
				VkDescriptorSet ds = static_cast<VkDescriptorSet>(dsSet);
				vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
					static_cast<VkPipelineLayout>(pipeline.PipelineLayout),
					0, 1, &ds, 0, nullptr);

				if (node.PushSize > 0) {
					vkCmdPushConstants(cb,
						static_cast<VkPipelineLayout>(pipeline.PipelineLayout),
						VK_SHADER_STAGE_COMPUTE_BIT, 0, node.PushSize, node.PushData);
				}
			}

			if (node.Indirect) {
				vkCmdDispatchIndirect(cb,
					static_cast<VkBuffer>(node.IndirectBuffer.Buffer),
					node.IndirectOffset);
			} else {
				vkCmdDispatch(cb, node.GroupsX, node.GroupsY, node.GroupsZ);
			}

			UpdateBufferStates(node, bufferStates);
		}

		RecordFinalBarrier(static_cast<VkCommandBuffer>(stream->CommandBuffer));
	}

	// Submit all device batches and track fences locally.
	OaVec<VkFence> fences(meshNodeCount, VK_NULL_HANDLE);
	OaVec<VkDevice> fenceDevs(meshNodeCount, VK_NULL_HANDLE);

	for (OaU32 d = 0; d < meshNodeCount; ++d) {
		if (!batches[d].HasWork) continue;
		auto* stream = batches[d].Stream;

		VkCommandBuffer cb = static_cast<VkCommandBuffer>(stream->CommandBuffer);
		auto* devNode = InRt.GetNode(d);
		if (!devNode) continue;

		VkDevice dev = static_cast<VkDevice>(devNode->Device.Device);
		fenceDevs[d] = dev;

		VkFenceCreateInfo fenceCI{};
		fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cb;

		if (d != 0) {
			OaSpinlockGuard guard(*InRt.GetMesh()->DeviceLoadLock);
			OaVkLoadDevice(dev);
			vkEndCommandBuffer(cb);
			vkCreateFence(dev, &fenceCI, nullptr, &fences[d]);
			vkQueueSubmit(
				static_cast<VkQueue>(devNode->Device.Queues.ComputeQueue),
				1, &submitInfo, fences[d]);
			OaVkLoadDevice(static_cast<VkDevice>(InRt.Device.Device));
		} else {
			vkEndCommandBuffer(cb);
			vkCreateFence(dev, &fenceCI, nullptr, &fences[d]);
			if (auto s = InRt.SubmitToQueue(devNode->Device.Queues.ComputeQueue, &submitInfo, fences[d]); !s.IsOk()) {
				OA_LOG_ERROR(OaLogComponent::Core, "Graph Execute: SubmitToQueue node %u failed: %s", d, s.GetMessage().c_str());
			}
		}
	}

	// Wait for all device batches to complete (host-side fence sync)
	for (OaU32 d = 0; d < meshNodeCount; ++d) {
		if (fences[d] == VK_NULL_HANDLE) continue;

		if (d != 0) {
			OaSpinlockGuard guard(*InRt.GetMesh()->DeviceLoadLock);
			OaVkLoadDevice(fenceDevs[d]);
			vkWaitForFences(fenceDevs[d], 1, &fences[d], VK_TRUE, UINT64_MAX);
			vkDestroyFence(fenceDevs[d], fences[d], nullptr);
			OaVkLoadDevice(static_cast<VkDevice>(InRt.Device.Device));
		} else {
			vkWaitForFences(fenceDevs[d], 1, &fences[d], VK_TRUE, UINT64_MAX);
			vkDestroyFence(fenceDevs[d], fences[d], nullptr);
		}
	}

	// Release all streams
	for (OaU32 d = 0; d < meshNodeCount; ++d) {
		if (batches[d].Stream) {
			InRt.ReleaseStreamOn(d, batches[d].Stream);
		}
	}

	return OaStatus::Ok();
}

void OaComputeGraph::MapToMesh(OaDeviceMesh& InMesh, OaScheduler& InScheduler) {
	(void)InMesh;
	for (auto& node : Nodes_) {
		OaDispatchHint hint;
		hint.Class = OaComputeClass::Any;
		if (node.NodeIndex != 0) {
			hint.PreferNode = node.NodeIndex;
		}
		auto* target = InScheduler.Route(hint);
		node.NodeIndex = target ? target->Index : 0;
	}
}

static VkAccessFlags2 ShaderAccessMask(OaBufferAccess InAccess) {
	if (InAccess == OaBufferAccess::Read) {
		return VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
	}
	if (InAccess == OaBufferAccess::Write) {
		return VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
	}
	return VK_ACCESS_2_SHADER_STORAGE_READ_BIT
		| VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
}

// Shared hazard computation used by Execute and Compile. Read-after-read is the
// only reuse requiring no dependency. RAW/WAW use memory dependencies; WAR uses
// a pure execution dependency (zero access masks), as required by Vulkan.
static OaU32 ComputeNodeBarriers(
	const OaComputeNode& InNode,
	const OaHashMap<void*, OaGraphBufferState>& InState,
	OaVec<VkBufferMemoryBarrier2>& OutBarriers)
{
	OutBarriers.Clear();
	auto emit = [&](void* InBuffer, VkPipelineStageFlags2 InStageMask,
		VkAccessFlags2 InAccessMask, bool InWrites) {
		auto it = InState.Find(InBuffer);
		if (it == InState.End()) return;
		const OaGraphBufferState& previous = it->second;
		if (!previous.Writes && !InWrites) return;

		VkBufferMemoryBarrier2 bar{};
		bar.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
		bar.srcStageMask = previous.StageMask;
		bar.srcAccessMask = previous.Writes ? previous.AccessMask : 0;
		bar.dstStageMask = InStageMask;
		bar.dstAccessMask = previous.Writes ? InAccessMask : 0;
		bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bar.buffer = static_cast<VkBuffer>(InBuffer);
		bar.offset = 0;
		bar.size = VK_WHOLE_SIZE;
		OutBarriers.PushBack(bar);
	};

	// Merge duplicate declarations in-place without allocating a temporary
	// access vector. Node fan-in is small, so this quadratic scan is cheaper
	// than heap traffic in the graph planner's per-step hot path.
	for (OaU32 i = 0; i < static_cast<OaU32>(InNode.Buffers.Size()); ++i) {
		void* handle = InNode.Buffers[i].Buffer;
		if (!handle) continue;
		bool alreadyProcessed = false;
		for (OaU32 p = 0; p < i; ++p) {
			if (InNode.Buffers[p].Buffer == handle) {
				alreadyProcessed = true;
				break;
			}
		}
		if (alreadyProcessed) continue;

		VkPipelineStageFlags2 stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		VkAccessFlags2 accessMask = 0;
		bool writes = false;
		for (OaU32 j = i; j < static_cast<OaU32>(InNode.Buffers.Size()); ++j) {
			if (InNode.Buffers[j].Buffer != handle) continue;
			accessMask |= ShaderAccessMask(InNode.Access[j]);
			writes = writes || InNode.Access[j] != OaBufferAccess::Read;
		}
		if (InNode.Indirect && InNode.IndirectBuffer.Buffer == handle) {
			stageMask |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
			accessMask |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
		}
		emit(handle, stageMask, accessMask, writes);
	}

	if (InNode.Indirect && InNode.IndirectBuffer.Buffer) {
		bool includedByRegularAccess = false;
		for (const auto& buffer : InNode.Buffers) {
			if (buffer.Buffer == InNode.IndirectBuffer.Buffer) {
				includedByRegularAccess = true;
				break;
			}
		}
		if (!includedByRegularAccess) {
			emit(InNode.IndirectBuffer.Buffer,
				VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
				VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, false);
		}
	}
	return static_cast<OaU32>(OutBarriers.Size());
}

// An execution dependency orders pipeline stages, not individual buffers.  If
// a barrier batch already contains a RAW/WAW memory dependency whose source
// and destination stage scopes cover a WAR dependency, the WAR contributes no
// additional ordering and can be omitted.  This is common at optimizer
// boundaries: gradient RAW already orders the backward dispatch (which also
// reads the weight) before the optimizer dispatch (which writes the weight).
// A standalone WAR is retained.
static OaU32 PruneRedundantWarBarriers(
	OaVec<VkBufferMemoryBarrier2>& InOutBarriers)
{
	VkPipelineStageFlags2 orderedSrcStages = 0;
	VkPipelineStageFlags2 orderedDstStages = 0;
	for (const auto& bar : InOutBarriers) {
		if (bar.srcAccessMask == 0 && bar.dstAccessMask == 0) continue;
		orderedSrcStages |= bar.srcStageMask;
		orderedDstStages |= bar.dstStageMask;
	}

	OaUsize write = 0;
	OaU32 removed = 0;
	for (OaUsize read = 0; read < InOutBarriers.Size(); ++read) {
		const auto& bar = InOutBarriers[read];
		const bool isWar = bar.srcAccessMask == 0 && bar.dstAccessMask == 0;
		const bool stageOrderingCovered =
			(bar.srcStageMask & ~orderedSrcStages) == 0
			&& (bar.dstStageMask & ~orderedDstStages) == 0;
		if (isWar && stageOrderingCovered) {
			++removed;
			continue;
		}
		if (write != read) InOutBarriers[write] = bar;
		++write;
	}
	InOutBarriers.Resize(write);
	return removed;
}

// PR-3 item G (OaLlamaCppVulkanLessons.md §12) — barrier-coalescing groups.
//
// Two consecutive nodes "conflict" iff they share a buffer with at least one
// non-Read access (RAW, WAR, WAW hazard). Non-conflicting consecutive nodes
// can be dispatched between a SINGLE barrier (covering the union of their
// incoming deps from earlier groups) instead of one barrier each.
//
// Returns true iff the two nodes conflict.
static bool NodesConflict(const OaComputeNode& InA, const OaComputeNode& InB) {
	for (OaU32 i = 0; i < static_cast<OaU32>(InA.Buffers.Size()); ++i) {
		void* a = InA.Buffers[i].Buffer;
		if (!a) continue;
		for (OaU32 j = 0; j < static_cast<OaU32>(InB.Buffers.Size()); ++j) {
			if (a != InB.Buffers[j].Buffer) continue;
			if (InA.Access[i] != OaBufferAccess::Read
				|| InB.Access[j] != OaBufferAccess::Read)
			{
				return true;
			}
		}
		if (InB.Indirect && a == InB.IndirectBuffer.Buffer
			&& InA.Access[i] != OaBufferAccess::Read)
		{
			return true;
		}
	}
	if (InA.Indirect && InA.IndirectBuffer.Buffer) {
		for (OaU32 j = 0; j < static_cast<OaU32>(InB.Buffers.Size()); ++j) {
			if (InA.IndirectBuffer.Buffer == InB.Buffers[j].Buffer
				&& InB.Access[j] != OaBufferAccess::Read)
			{
				return true;
			}
		}
	}
	return false;
}

// PR-3 G v2 — look-ahead REORDER. The actual Bolz 2026 mechanism.
//
// Builds a barrier-group by scanning forward up to InWindow nodes and pulling
// every non-conflicting independent node into the group. Critically, this
// SKIPS blocked nodes (instead of stopping at them) — so node j+1 may join the
// group even if j cannot.
//
// "Independent" means: every node in [InStart, candidate) that is NOT yet in
// the schedule (used[k]==false) has no dataflow conflict with candidate.
// (Already-scheduled deps are by definition resolved before this group runs.)
//
// On return, OutGroup contains the indices to dispatch IN ORDER. InOutUsed is
// updated to mark each included index. Caller must advance past used[] until
// the next unused entry.
//
// Indirect-dispatch nodes and queue-hint heterogeneity are treated as hard
// boundaries — same as v1.
static void BuildBarrierGroup(
	const OaVec<OaComputeNode>& InNodes,
	OaVec<bool>& InOutUsed,
	OaU32 InStart, OaU32 InWindow,
	OaVec<OaU32>& OutGroup)
{
	OutGroup.Clear();
	const OaU32 N = static_cast<OaU32>(InNodes.Size());
	if (InStart >= N || InOutUsed[InStart]) return;

	// Always include the seed.
	OutGroup.PushBack(InStart);
	InOutUsed[InStart] = true;

	// Indirect dispatch is a hard scheduling boundary on both sides.
	if (InNodes[InStart].Indirect) return;

	// Window-1 means seed-only (= OA_DISABLE_GRAPH_OPTIMIZE behavior).
	if (InWindow <= 1) return;

	const OaU32 end = std::min(InStart + InWindow, N);
	for (OaU32 j = InStart + 1; j < end; ++j) {
		if (InOutUsed[j]) continue;
		// Hard barrier boundaries.
		if (InNodes[j].Indirect) break;
		if (InNodes[j].Queue != InNodes[InStart].Queue) continue;

		// Does j depend on any not-yet-scheduled node in [InStart, j)?
		// A "dep" exists iff some unused k in that range conflicts with j.
		bool depUnsatisfied = false;
		for (OaU32 k = InStart; k < j; ++k) {
			if (InOutUsed[k]) continue;
			if (NodesConflict(InNodes[k], InNodes[j])) {
				depUnsatisfied = true;
				break;
			}
		}
		if (depUnsatisfied) continue;

		// Does j conflict with any node already IN the group?
		bool conflictsWithGroup = false;
		for (OaU32 g : OutGroup) {
			if (NodesConflict(InNodes[g], InNodes[j])) {
				conflictsWithGroup = true;
				break;
			}
		}
		if (conflictsWithGroup) continue;

		OutGroup.PushBack(j);
		InOutUsed[j] = true;
	}
}

static void UpdateBufferStates(
	const OaComputeNode& InNode,
	OaHashMap<void*, OaGraphBufferState>& InOutState)
{
	auto update = [&](void* InBuffer, VkPipelineStageFlags2 InStageMask,
		VkAccessFlags2 InAccessMask, bool InWrites) {
		OaGraphBufferState next;
		next.StageMask = InStageMask;
		next.AccessMask = InAccessMask;
		next.Writes = InWrites;

		auto it = InOutState.Find(InBuffer);
		if (it == InOutState.End()) {
			InOutState.Emplace(InBuffer, next);
		} else if (!it->second.Writes && !next.Writes) {
			// Preserve every outstanding read domain for a later WAR dependency.
			it->second.StageMask |= next.StageMask;
			it->second.AccessMask |= next.AccessMask;
		} else {
			it->second = next;
		}
	};

	for (OaU32 i = 0; i < static_cast<OaU32>(InNode.Buffers.Size()); ++i) {
		void* handle = InNode.Buffers[i].Buffer;
		if (!handle) continue;
		bool alreadyProcessed = false;
		for (OaU32 p = 0; p < i; ++p) {
			if (InNode.Buffers[p].Buffer == handle) {
				alreadyProcessed = true;
				break;
			}
		}
		if (alreadyProcessed) continue;

		VkPipelineStageFlags2 stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		VkAccessFlags2 accessMask = 0;
		bool writes = false;
		for (OaU32 j = i; j < static_cast<OaU32>(InNode.Buffers.Size()); ++j) {
			if (InNode.Buffers[j].Buffer != handle) continue;
			accessMask |= ShaderAccessMask(InNode.Access[j]);
			writes = writes || InNode.Access[j] != OaBufferAccess::Read;
		}
		if (InNode.Indirect && InNode.IndirectBuffer.Buffer == handle) {
			stageMask |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
			accessMask |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
		}
		update(handle, stageMask, accessMask, writes);
	}

	if (InNode.Indirect && InNode.IndirectBuffer.Buffer) {
		bool includedByRegularAccess = false;
		for (const auto& buffer : InNode.Buffers) {
			if (buffer.Buffer == InNode.IndirectBuffer.Buffer) {
				includedByRegularAccess = true;
				break;
			}
		}
		if (!includedByRegularAccess) {
			update(InNode.IndirectBuffer.Buffer,
				VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
				VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, false);
		}
	}
}

static void RecordFinalBarrier(VkCommandBuffer InCb) {
	// Graph submissions may be followed by a host wait and direct mapped-buffer
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
	vkCmdPipelineBarrier2(InCb, &finalDep);
}

// Allocate a per-dispatch descriptor pool + set. Mirrors AllocStreamDescriptorSet.
static OaStatus AllocGraphDescriptorSet(
	const OaVkDevice& InDevice,
	OaComputePipeline& InPipeline,
	OaSpan<OaVkBuffer> InBuffers,
	void** OutPool,
	void** OutSet
) {
	VkDevice dev = static_cast<VkDevice>(InDevice.Device);

	OaU32 numDesc = InPipeline.NumBindings;
	if (numDesc < static_cast<OaU32>(InBuffers.size())) {
		numDesc = static_cast<OaU32>(InBuffers.size());
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
	VkResult r = vkCreateDescriptorPool(dev, &dpCI, nullptr, &pool);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::PipelineError,
			"graph: vkCreateDescriptorPool failed");
	}

	VkDescriptorSetLayout dsl = static_cast<VkDescriptorSetLayout>(InPipeline.DescriptorSetLayout);
	VkDescriptorSetAllocateInfo dsAI{};
	dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAI.descriptorPool = pool;
	dsAI.descriptorSetCount = 1;
	dsAI.pSetLayouts = &dsl;

	VkDescriptorSet ds = VK_NULL_HANDLE;
	r = vkAllocateDescriptorSets(dev, &dsAI, &ds);
	if (r != VK_SUCCESS) {
		vkDestroyDescriptorPool(dev, pool, nullptr);
		return OaStatus::Error(OaStatusCode::PipelineError,
			"graph: vkAllocateDescriptorSets failed");
	}

	for (OaU32 i = 0; i < static_cast<OaU32>(InBuffers.size()); ++i) {
		VkDescriptorBufferInfo bufInfo{};
		bufInfo.buffer = static_cast<VkBuffer>(InBuffers[i].Buffer);
		bufInfo.offset = 0;
		bufInfo.range = InBuffers[i].Size;

		VkWriteDescriptorSet write{};
		write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write.dstSet = ds;
		write.dstBinding = i;
		write.descriptorCount = 1;
		write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		write.pBufferInfo = &bufInfo;

		vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
	}

	*OutPool = pool;
	*OutSet = ds;
	return OaStatus::Ok();
}

// ─── Phase 1: One-shot execution ──────────────────────────────────────────────

OaStatus OaComputeGraph::Execute(OaComputeEngine& InRt) {
	if (Nodes_.Empty()) return OaStatus::Ok();

	OaVkStream* stream = InRt.AcquireStream();
	if (!stream) {
		return OaStatus::Error(OaStatusCode::VulkanError, "graph: failed to acquire stream");
	}

	OA_RETURN_IF_ERROR(stream->Begin(InRt.Device));

	OaHashMap<void*, OaGraphBufferState> bufferStates;
	OaVec<VkBufferMemoryBarrier2> barriers;
	barriers.Reserve(8);

	// PR-3 item G v2: look-ahead REORDER + barrier-coalescing.
	// BuildBarrierGroup pulls non-conflicting nodes adjacent across blocked
	// nodes (within a 20-node window). Within a group: ONE coalesced barrier,
	// dispatch in group order, deferred resource-state update.
	const OaU32 kGroupWindow =
		OaEnvFlag::IsSet("OA_DISABLE_GRAPH_OPTIMIZE") ? 1U : 20U;
	OaVec<bool> usedNode(Nodes_.Size(), false);
	OaVec<OaU32> group;
	OaU32 first = 0;
	while (first < static_cast<OaU32>(Nodes_.Size())) {
		while (first < static_cast<OaU32>(Nodes_.Size()) && usedNode[first]) ++first;
		if (first >= static_cast<OaU32>(Nodes_.Size())) break;

		BuildBarrierGroup(Nodes_, usedNode, first, kGroupWindow, group);

		OaVec<VkBufferMemoryBarrier2> groupBarriers;
		OaHashMap<void*, OaU32> seenBufsInGroup;
		for (OaU32 idx : group) {
			ComputeNodeBarriers(Nodes_[idx], bufferStates, barriers);
			for (const auto& bar : barriers) {
				void* bufKey = static_cast<void*>(bar.buffer);
				if (seenBufsInGroup.Find(bufKey) == seenBufsInGroup.End()) {
					groupBarriers.PushBack(bar);
					seenBufsInGroup.Emplace(bufKey, 1U);
				}
			}
		}
		PruneRedundantWarBarriers(groupBarriers);

		if (!groupBarriers.Empty()) {
			VkDependencyInfo dep{};
			dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
			dep.bufferMemoryBarrierCount = static_cast<OaU32>(groupBarriers.Size());
			dep.pBufferMemoryBarriers = groupBarriers.Data();
			vkCmdPipelineBarrier2(
				static_cast<VkCommandBuffer>(stream->CommandBuffer), &dep);
		}

		for (OaU32 idx : group) {
			auto& node = Nodes_[idx];
			if (node.Indirect) {
				OA_RETURN_IF_ERROR(stream->RecordDispatchIndirect(
					InRt, node.Shader, node.Buffers.Span(),
					node.PushSize > 0 ? node.PushData : nullptr,
					node.PushSize,
					node.IndirectBuffer, node.IndirectOffset));
			} else {
				OA_RETURN_IF_ERROR(stream->RecordDispatch(
					InRt, node.Shader, node.Buffers.Span(),
					node.PushSize > 0 ? node.PushData : nullptr,
					node.PushSize,
					node.GroupsX, node.GroupsY, node.GroupsZ));
			}
		}

		for (OaU32 idx : group) {
			UpdateBufferStates(Nodes_[idx], bufferStates);
		}
	}

	RecordFinalBarrier(static_cast<VkCommandBuffer>(stream->CommandBuffer));

	OaStatus status = stream->SubmitAndWait(InRt);
	InRt.ReleaseStream(stream);
	return status;
}

// ─── Multi-Queue Execution ────────────────────────────────────────────────

OaStatus OaComputeGraph::ExecuteMultiQueue(OaComputeEngine& InRt) {
	if (Nodes_.Empty()) return OaStatus::Ok();

	if (!InRt.HasAsyncCompute()) return Execute(InRt);

	OaVkStream* computeStream = InRt.AcquireStream();
	OaVkStream* asyncStream = InRt.AcquireAsyncStream();
	if (!computeStream || !asyncStream) {
		if (computeStream) InRt.ReleaseStream(computeStream);
		if (asyncStream) InRt.ReleaseAsyncStream(asyncStream);
		return Execute(InRt);
	}

	OA_RETURN_IF_ERROR(computeStream->Begin(InRt.Device));
	OA_RETURN_IF_ERROR(asyncStream->Begin(InRt.Device));

	OaHashMap<void*, OaGraphBufferState> bufferStates;
	OaVec<VkBufferMemoryBarrier2> barriers;
	barriers.Reserve(8);

	// Track the last-submitted stream for cross-queue dependency
	OaVkStream* prevStream = nullptr;
	OaU64 prevTimelineValue = 0;

	OaQueueHint currentQueue = OaQueueHint::Compute;
	for (OaU32 i = 0; i < static_cast<OaU32>(Nodes_.Size()); ++i) {
		auto& node = Nodes_[i];
		OaQueueHint nodeQueue = node.Queue;
		// Transfer collapses to compute unless separate transfer queue exists
		if (nodeQueue == OaQueueHint::Transfer) nodeQueue = OaQueueHint::Compute;

		if (nodeQueue != currentQueue) {
			OaVkStream* outgoing = (currentQueue == OaQueueHint::AsyncCompute)
				? asyncStream : computeStream;

			RecordFinalBarrier(static_cast<VkCommandBuffer>(outgoing->CommandBuffer));
			OA_RETURN_IF_ERROR(outgoing->Submit(InRt));

			// Track the dependency for the incoming stream
			prevStream = outgoing;
			prevTimelineValue = outgoing->TimelineValue;

			OaVkStream* incoming = (nodeQueue == OaQueueHint::AsyncCompute)
				? asyncStream : computeStream;
			OA_RETURN_IF_ERROR(incoming->Begin(InRt.Device));

			currentQueue = nodeQueue;
		}

		OaVkStream* activeStream = (currentQueue == OaQueueHint::AsyncCompute)
			? asyncStream : computeStream;

		ComputeNodeBarriers(node, bufferStates, barriers);
		PruneRedundantWarBarriers(barriers);
		if (!barriers.Empty()) {
			VkDependencyInfo dep{};
			dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
			dep.bufferMemoryBarrierCount = static_cast<OaU32>(barriers.Size());
			dep.pBufferMemoryBarriers = barriers.Data();
			vkCmdPipelineBarrier2(
				static_cast<VkCommandBuffer>(activeStream->CommandBuffer), &dep);
		}

		if (node.Indirect) {
			OA_RETURN_IF_ERROR(activeStream->RecordDispatchIndirect(
				InRt, node.Shader, node.Buffers.Span(),
				node.PushSize > 0 ? node.PushData : nullptr,
				node.PushSize, node.IndirectBuffer, node.IndirectOffset));
		} else {
			OA_RETURN_IF_ERROR(activeStream->RecordDispatch(
				InRt, node.Shader, node.Buffers.Span(),
				node.PushSize > 0 ? node.PushData : nullptr,
				node.PushSize,
				node.GroupsX, node.GroupsY, node.GroupsZ));
		}

		UpdateBufferStates(node, bufferStates);
	}

	// Submit final batch with cross-queue dependency if needed
	OaVkStream* finalStream = (currentQueue == OaQueueHint::AsyncCompute)
		? asyncStream : computeStream;
	RecordFinalBarrier(static_cast<VkCommandBuffer>(finalStream->CommandBuffer));

	OaStatus status;
	if (prevStream && prevStream != finalStream) {
		// GPU-side wait: final stream waits on previous stream's semaphore
		OA_RETURN_IF_ERROR(finalStream->SubmitWithDependency(
			InRt, prevStream->TimelineSem, prevTimelineValue));
		status = finalStream->Synchronize(InRt.Device);
	} else {
		status = finalStream->SubmitAndWait(InRt);
	}

	// Ensure all streams are complete before releasing
	if (computeStream->Submitted) {
		if (auto s = computeStream->Synchronize(InRt.Device); !s.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core, "Graph Execute: compute stream sync failed: %s", s.GetMessage().c_str());
		}
	}
	if (asyncStream->Submitted) {
		if (auto s = asyncStream->Synchronize(InRt.Device); !s.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core, "Graph Execute: async stream sync failed: %s", s.GetMessage().c_str());
		}
	}

	InRt.ReleaseStream(computeStream);
	InRt.ReleaseAsyncStream(asyncStream);
	return status;
}

// ─── Phase 2: Compile & Replay ────────────────────────────────────────────────

OaU64 OaComputeGraph::ComputeNodeHash() const {
	OaU64 hash = 14695981039346656037ULL;  // FNV-1a 64-bit offset basis
	auto fnv = [&](const void* InData, OaU64 InBytes) {
		const auto* p = static_cast<const OaU8*>(InData);
		for (OaU64 i = 0; i < InBytes; ++i) {
			hash ^= p[i];
			hash *= 1099511628211ULL;  // FNV-1a 64-bit prime
		}
	};

	// Node count
	const OaU32 n = static_cast<OaU32>(Nodes_.Size());
	fnv(&n, sizeof(n));

	for (const auto& node : Nodes_) {
		// Shader name
		fnv(node.Shader.Data(), node.Shader.Size());
		// Dtype
		fnv(&node.Dtype, sizeof(node.Dtype));
		// Buffer identity and access declarations. The VkBuffer handles are
		// baked into synchronization barriers, while bindless indices are baked
		// into push constants; both therefore participate in cache identity.
		const OaU32 nBufs = static_cast<OaU32>(node.Buffers.Size());
		fnv(&nBufs, sizeof(nBufs));
		for (OaU32 i = 0; i < nBufs; ++i) {
			const auto& buf = node.Buffers[i];
			fnv(&buf.Buffer, sizeof(buf.Buffer));
			fnv(&buf.BindlessIndex, sizeof(buf.BindlessIndex));
			fnv(&buf.Size, sizeof(buf.Size));
			fnv(&node.Access[i], sizeof(node.Access[i]));
		}
		// Push constants (includes buffer index header for bindless + user push)
		fnv(&node.PushSize, sizeof(node.PushSize));
		fnv(node.PushData, node.PushSize);
		// Dispatch dimensions
		fnv(&node.GroupsX, sizeof(node.GroupsX));
		fnv(&node.GroupsY, sizeof(node.GroupsY));
		fnv(&node.GroupsZ, sizeof(node.GroupsZ));
		// Indirect dispatch flag
		fnv(&node.Indirect, sizeof(node.Indirect));
		if (node.Indirect) {
			fnv(&node.IndirectBuffer.Buffer, sizeof(node.IndirectBuffer.Buffer));
			fnv(&node.IndirectBuffer.BindlessIndex,
				sizeof(node.IndirectBuffer.BindlessIndex));
			fnv(&node.IndirectBuffer.Size, sizeof(node.IndirectBuffer.Size));
			fnv(&node.IndirectOffset, sizeof(node.IndirectOffset));
		}
		fnv(&node.Queue, sizeof(node.Queue));
		fnv(&node.NodeIndex, sizeof(node.NodeIndex));
	}

	return hash;
}

OaStatus OaComputeGraph::Compile(OaComputeEngine& InRt) {
	if (Nodes_.Empty()) {
		Compiled_ = true;
		BarrierCount_ = 0;
		return OaStatus::Ok();
	}

	// Compile-once-replay-many: if the node list hash matches the last
	// compilation, the existing secondary CB is still valid. Skip
	// vkResetCommandBuffer + re-recording entirely — just mark compiled.
	// Also skip rebuilding the primary CB — it still wraps the same secondary.
	const OaU64 nodeHash = ComputeNodeHash();
	if (SecondaryCb_ and nodeHash == LastCompileHash_) {
		Compiled_ = true;
		return OaStatus::Ok();
	}

	VkDevice dev = static_cast<VkDevice>(InRt.Device.Device);
	QueueFamily_ = InRt.Device.Queues.ComputeQueueFamily;

	// Reuse existing command pool + secondary CB if available. This avoids
	// vkCreateCommandPool + vkAllocateCommandBuffers + vkDestroyCommandPool
	// + vkFreeCommandBuffers per Execute() call — saves ~0.05ms.
	if (SecondaryPool_ and SecondaryCb_) {
		// Free descriptor pools from previous compilation (non-bindless only).
		for (void* pool : DescriptorPools_) {
			vkDestroyDescriptorPool(dev, static_cast<VkDescriptorPool>(pool), nullptr);
		}
		DescriptorPools_.Clear();

		// Reset the secondary CB for re-recording.
		VkResult r = vkResetCommandBuffer(
			static_cast<VkCommandBuffer>(SecondaryCb_), 0);
		if (r != VK_SUCCESS) {
			// Reset failed — fall back to full recreate.
			Invalidate(InRt.Device);
		}
		// Compiled_ is already false (set by ClearNodes or Invalidate)
	}

	if (!SecondaryPool_) {
		// Create a dedicated command pool for the secondary CB.
		VkCommandPoolCreateInfo cpCI{};
		cpCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cpCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		cpCI.queueFamilyIndex = QueueFamily_;

		VkCommandPool pool = VK_NULL_HANDLE;
		VkResult r = vkCreateCommandPool(dev, &cpCI, nullptr, &pool);
		if (r != VK_SUCCESS) {
			return OaStatus::Error(OaStatusCode::VulkanError,
				"graph compile: vkCreateCommandPool failed");
		}
		SecondaryPool_ = pool;
	}

	if (!SecondaryCb_) {
		// Allocate secondary command buffer.
		VkCommandBufferAllocateInfo cbAI{};
		cbAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cbAI.commandPool = static_cast<VkCommandPool>(SecondaryPool_);
		cbAI.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
		cbAI.commandBufferCount = 1;

		VkCommandBuffer scb = VK_NULL_HANDLE;
		VkResult r = vkAllocateCommandBuffers(dev, &cbAI, &scb);
		if (r != VK_SUCCESS) {
			vkDestroyCommandPool(dev, static_cast<VkCommandPool>(SecondaryPool_), nullptr);
			SecondaryPool_ = nullptr;
			return OaStatus::Error(OaStatusCode::VulkanError,
				"graph compile: vkAllocateCommandBuffers failed");
		}
		SecondaryCb_ = scb;
	}

	// Begin secondary CB — simultaneous use allows replay while previous is in-flight
	VkCommandBufferInheritanceInfo inhInfo{};
	inhInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
	beginInfo.pInheritanceInfo = &inhInfo;

	VkResult r = vkBeginCommandBuffer(static_cast<VkCommandBuffer>(SecondaryCb_), &beginInfo);
	if (r != VK_SUCCESS) {
		Invalidate(InRt.Device);
		return OaStatus::Error(OaStatusCode::VulkanError,
			"graph compile: vkBeginCommandBuffer failed");
	}

	// Record all dispatches with minimal barriers into the secondary CB
	OaHashMap<void*, OaGraphBufferState> bufferStates;
	OaVec<VkBufferMemoryBarrier2> barriers;
	barriers.Reserve(8);
	BarrierCount_ = 0;
	GroupCount_   = 0;
	MaxGroupSize_ = 0;
	ReorderCount_ = 0;
	WarBarrierCount_ = 0;
	IndirectBarrierCount_ = 0;

	// PR-3 item G v2 (OaLlamaCppVulkanLessons.md §12): look-ahead REORDER +
	// barrier coalescing. BuildBarrierGroup pulls non-conflicting nodes
	// adjacent across blocked nodes (within a 20-node window). Within a group:
	// ONE coalesced barrier, dispatch in group order, deferred resource-state update.
	const OaU32 kGroupWindow =
		OaEnvFlag::IsSet("OA_DISABLE_GRAPH_OPTIMIZE") ? 1U : 20U;
	OaVec<bool> usedNode(Nodes_.Size(), false);
	OaVec<OaU32> group;
	VkPipeline boundPipeline = VK_NULL_HANDLE;
	bool bindlessDescriptorBound = false;
	OaU32 first = 0;
	while (first < static_cast<OaU32>(Nodes_.Size())) {
		while (first < static_cast<OaU32>(Nodes_.Size()) && usedNode[first]) ++first;
		if (first >= static_cast<OaU32>(Nodes_.Size())) break;

		BuildBarrierGroup(Nodes_, usedNode, first, kGroupWindow, group);

		// Stats: every node added past the seed AND not at its natural-order
		// position counts as a "reorder event". Seed itself doesn't count.
		// Track by checking if group[g] > first + g (gap means it was pulled).
		++GroupCount_;
		const OaU32 gsz = static_cast<OaU32>(group.Size());
		if (gsz > MaxGroupSize_) MaxGroupSize_ = gsz;
		for (OaU32 g = 1; g < gsz; ++g) {
			if (group[g] > first + g) ++ReorderCount_;
		}

		// ── Coalesce incoming barriers across ALL nodes in the group.
		OaVec<VkBufferMemoryBarrier2> groupBarriers;
		OaHashMap<void*, OaU32> seenBufsInGroup;
		for (OaU32 idx : group) {
			ComputeNodeBarriers(Nodes_[idx], bufferStates, barriers);
			for (const auto& bar : barriers) {
				void* bufKey = static_cast<void*>(bar.buffer);
				if (seenBufsInGroup.Find(bufKey) == seenBufsInGroup.End()) {
					groupBarriers.PushBack(bar);
					seenBufsInGroup.Emplace(bufKey, 1U);
				}
			}
		}
		PruneRedundantWarBarriers(groupBarriers);
		BarrierCount_ += static_cast<OaU32>(groupBarriers.Size());
		for (const auto& bar : groupBarriers) {
			if (bar.srcAccessMask == 0 && bar.dstAccessMask == 0) {
				++WarBarrierCount_;
			}
			if ((bar.dstStageMask & VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT) != 0
				&& (bar.dstAccessMask & VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT) != 0)
			{
				++IndirectBarrierCount_;
			}
		}

		if (!groupBarriers.Empty()) {
			VkDependencyInfo dep{};
			dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
			dep.bufferMemoryBarrierCount = static_cast<OaU32>(groupBarriers.Size());
			dep.pBufferMemoryBarriers = groupBarriers.Data();
			vkCmdPipelineBarrier2(static_cast<VkCommandBuffer>(SecondaryCb_), &dep);
		}

		// ── Dispatch each node in the group WITHOUT inter-node barriers.
		for (OaU32 idx : group) {
			auto& node = Nodes_[idx];

			// Resolve pipeline — DTYPE from the node (derived from its operand tensors).
			auto& pipeline = InRt.Pipelines.GetPipeline(node.Shader, node.Dtype);
			if (!pipeline.Pipeline) {
				Invalidate(InRt.Device);
				return OaStatus::Error("graph compile: pipeline not found: " + node.Shader);
			}

			void* dsPool = nullptr;
			void* dsSet = nullptr;
			if (!pipeline.Bindless) {
				OaStatus dsStatus = AllocGraphDescriptorSet(
					InRt.Device, pipeline, node.Buffers.Span(), &dsPool, &dsSet);
				if (!dsStatus.IsOk()) {
					Invalidate(InRt.Device);
					return dsStatus;
				}
				DescriptorPools_.PushBack(dsPool);
			}

			const VkPipeline vkPipeline = static_cast<VkPipeline>(pipeline.Pipeline);
			if (vkPipeline != boundPipeline) {
				vkCmdBindPipeline(static_cast<VkCommandBuffer>(SecondaryCb_),
					VK_PIPELINE_BIND_POINT_COMPUTE, vkPipeline);
				boundPipeline = vkPipeline;
			}

			if (pipeline.Bindless) {
				// Every bindless compute pipeline uses the engine-owned common
				// pipeline layout. Descriptor bindings remain valid across
				// compatible pipeline changes, so bind the global heap once.
				if (!bindlessDescriptorBound) {
					VkDescriptorSet bds =
						static_cast<VkDescriptorSet>(InRt.Bindless.DescriptorSet);
					vkCmdBindDescriptorSets(
						static_cast<VkCommandBuffer>(SecondaryCb_),
						VK_PIPELINE_BIND_POINT_COMPUTE,
						static_cast<VkPipelineLayout>(pipeline.PipelineLayout),
						0, 1, &bds, 0, nullptr);
					bindlessDescriptorBound = true;
				}

				OaU32 numBufs = static_cast<OaU32>(node.Buffers.Size());
				if (!OaVkBindlessPushFits(numBufs, node.PushSize)) {
					Invalidate(InRt.Device);
					return OaStatus::Error(OaStatusCode::InvalidArgument,
						"graph compile: bindless push exceeds OA_VK_MAX_PUSH_CONSTANT_BYTES "
						"(buffer index header + user push)");
				}
				OaU32 headerBytes = numBufs * sizeof(OaU32);
				OaU32 totalPush = headerBytes + node.PushSize;
				alignas(16) OaU8 pushBuf[OA_VK_MAX_PUSH_CONSTANT_BYTES] = {};
				OaU32* indices = reinterpret_cast<OaU32*>(pushBuf);
				for (OaU32 b = 0; b < numBufs; ++b) {
					indices[b] = node.Buffers[b].BindlessIndex;
				}
				if (node.PushSize > 0) {
					std::memcpy(pushBuf + headerBytes, node.PushData, node.PushSize);
				}
				vkCmdPushConstants(static_cast<VkCommandBuffer>(SecondaryCb_),
					static_cast<VkPipelineLayout>(pipeline.PipelineLayout),
					VK_SHADER_STAGE_COMPUTE_BIT, 0, totalPush, pushBuf);
			} else {
				bindlessDescriptorBound = false;
				VkDescriptorSet ds = static_cast<VkDescriptorSet>(dsSet);
				vkCmdBindDescriptorSets(static_cast<VkCommandBuffer>(SecondaryCb_), VK_PIPELINE_BIND_POINT_COMPUTE,
					static_cast<VkPipelineLayout>(pipeline.PipelineLayout),
					0, 1, &ds, 0, nullptr);

				if (node.PushSize > 0) {
					vkCmdPushConstants(static_cast<VkCommandBuffer>(SecondaryCb_),
						static_cast<VkPipelineLayout>(pipeline.PipelineLayout),
						VK_SHADER_STAGE_COMPUTE_BIT, 0, node.PushSize, node.PushData);
				}
			}

			if (node.Indirect) {
				vkCmdDispatchIndirect(static_cast<VkCommandBuffer>(SecondaryCb_),
					static_cast<VkBuffer>(node.IndirectBuffer.Buffer),
					node.IndirectOffset);
			} else {
				vkCmdDispatch(static_cast<VkCommandBuffer>(SecondaryCb_), node.GroupsX, node.GroupsY, node.GroupsZ);
			}
		}

		// ── Update resource state for the ENTIRE group AFTER all dispatches.
		// (Within a group nodes are non-conflicting, so the relative order of
		// these updates is semantically irrelevant.)
		for (OaU32 idx : group) {
			UpdateBufferStates(Nodes_[idx], bufferStates);
		}
	}

	RecordFinalBarrier(static_cast<VkCommandBuffer>(SecondaryCb_));

	r = vkEndCommandBuffer(static_cast<VkCommandBuffer>(SecondaryCb_));
	if (r != VK_SUCCESS) {
		Invalidate(InRt.Device);
		return OaStatus::Error(OaStatusCode::VulkanError,
			"graph compile: vkEndCommandBuffer failed");
	}

	// PR-3 G v3: opt-in compile-time summary log.
	if (OaEnvFlag::IsSet("OA_LOG_BARRIERS")) {
		const OaU32 n = static_cast<OaU32>(Nodes_.Size());
		const float avgGroupSize =
			GroupCount_ > 0
				? static_cast<float>(n) / static_cast<float>(GroupCount_)
				: 0.0F;
		OA_LOG_INFO(OaLogComponent::Core,
			"OaComputeGraph::Compile: nodes=%u barriers=%u groups=%u "
			"max_group=%u avg_group=%.2f reorder_events=%u war=%u indirect=%u "
			"(coalescing %s)",
			n, BarrierCount_, GroupCount_, MaxGroupSize_, avgGroupSize,
			ReorderCount_, WarBarrierCount_, IndirectBarrierCount_,
			GroupCount_ < n ? "ACTIVE" : "off / no opportunity");
	}

	// Build a pre-recorded primary CB that wraps the secondary. This lets
	// Replay() skip stream acquire + Begin + vkCmdExecuteCommands entirely
	// — just submit the primary directly with a dedicated timeline semaphore.
	if (PrimaryCb_ and PrimaryPool_) {
		VkCommandBuffer pcb = static_cast<VkCommandBuffer>(PrimaryCb_);
		vkResetCommandBuffer(pcb, 0);
	} else {
		VkCommandPoolCreateInfo ppCI{};
		ppCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		ppCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		ppCI.queueFamilyIndex = QueueFamily_;
		VkResult pr = vkCreateCommandPool(dev, &ppCI, nullptr,
			reinterpret_cast<VkCommandPool*>(&PrimaryPool_));
		if (pr != VK_SUCCESS) {
			return OaStatus::Error(OaStatusCode::VulkanError,
				"graph compile: vkCreateCommandPool (primary) failed");
		}
		VkCommandBufferAllocateInfo cbAI{};
		cbAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cbAI.commandPool = static_cast<VkCommandPool>(PrimaryPool_);
		cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cbAI.commandBufferCount = 1;
		pr = vkAllocateCommandBuffers(dev, &cbAI,
			reinterpret_cast<VkCommandBuffer*>(&PrimaryCb_));
		if (pr != VK_SUCCESS) {
			vkDestroyCommandPool(dev,
				static_cast<VkCommandPool>(PrimaryPool_), nullptr);
			PrimaryPool_ = nullptr;
			return OaStatus::Error(OaStatusCode::VulkanError,
				"graph compile: vkAllocateCommandBuffers (primary) failed");
		}
		// Create the dedicated timeline semaphore for cached replay.
		if (not ReplayTimelineSem_.Semaphore) {
			auto semRes = OaVkTimelineSemaphore::Create(InRt.Device);
			if (not semRes) {
				return OaStatus::Error(OaStatusCode::VulkanError,
					"graph compile: timeline semaphore creation failed");
			}
			ReplayTimelineSem_ = std::move(semRes).GetValue();
		}
	}
	{
		VkCommandBuffer pcb = static_cast<VkCommandBuffer>(PrimaryCb_);
		VkCommandBufferBeginInfo bi{};
		bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		bi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		VkResult pr = vkBeginCommandBuffer(pcb, &bi);
		if (pr != VK_SUCCESS) {
			return OaStatus::Error(OaStatusCode::VulkanError,
				"graph compile: vkBeginCommandBuffer (primary) failed");
		}
		VkCommandBuffer scb = static_cast<VkCommandBuffer>(SecondaryCb_);
		vkCmdExecuteCommands(pcb, 1, &scb);
		pr = vkEndCommandBuffer(pcb);
		if (pr != VK_SUCCESS) {
			return OaStatus::Error(OaStatusCode::VulkanError,
				"graph compile: vkEndCommandBuffer (primary) failed");
		}
	}

	Compiled_ = true;
	LastCompileHash_ = nodeHash;
	return OaStatus::Ok();
}

OaStatus OaComputeGraph::Replay(OaComputeEngine& InRt) {
	if (Nodes_.Empty()) return OaStatus::Ok();
	if (!Compiled_ || !SecondaryCb_) {
		return OaStatus::Error("graph replay: not compiled — call Compile() first");
	}

	// Fast path: if we have a pre-built primary CB (from Compile), submit
	// it directly with our dedicated timeline semaphore. Skips stream pool
	// acquire + Begin + vkCmdExecuteCommands + stream release entirely.
	// The primary CB uses SIMULTANEOUS_USE_BIT so it can be resubmitted
	// without re-recording.
	//
	// Non-blocking: submit and return immediately. Same-queue submissions
	// are implicitly ordered by the Vulkan spec — the GPU executes them in
	// submission order without host-side waits. The caller must call
	// WaitForPendingReplay() (or Sync()) before reading results.
	if (PrimaryCb_ and PrimaryPool_ and ReplayTimelineSem_.Semaphore) {
		++ReplayTimelineValue_;
		VkSemaphore sem = static_cast<VkSemaphore>(ReplayTimelineSem_.Semaphore);
		VkCommandBuffer pcb = static_cast<VkCommandBuffer>(PrimaryCb_);

		// Use vkQueueSubmit2 (Vulkan 1.3) — thinner driver path than
		// vkQueueSubmit, avoids VkSubmitInfo→internal conversion.
		VkCommandBufferSubmitInfo cbSI{};
		cbSI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
		cbSI.commandBuffer = pcb;

		VkSemaphoreSubmitInfo signalSI{};
		signalSI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
		signalSI.semaphore = sem;
		signalSI.value = ReplayTimelineValue_;
		signalSI.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

		VkSubmitInfo2 si2{};
		si2.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
		si2.commandBufferInfoCount = 1;
		si2.pCommandBufferInfos = &cbSI;
		si2.signalSemaphoreInfoCount = 1;
		si2.pSignalSemaphoreInfos = &signalSI;

		OA_RETURN_IF_ERROR(InRt.SubmitToQueue2(
			InRt.Device.Queues.ComputeQueue, &si2));
		return OaStatus::Ok();
	}

	// Slow path: acquire stream, record secondary into primary, submit+wait.
	OaVkStream* stream = InRt.AcquireStream();
	if (!stream) {
		return OaStatus::Error(OaStatusCode::VulkanError,
			"graph replay: failed to acquire stream");
	}

	OA_RETURN_IF_ERROR(stream->Begin(InRt.Device));

	VkCommandBuffer primary = static_cast<VkCommandBuffer>(stream->CommandBuffer);
	VkCommandBuffer secondary = static_cast<VkCommandBuffer>(SecondaryCb_);
	vkCmdExecuteCommands(primary, 1, &secondary);

	OaStatus status = stream->SubmitAndWait(InRt);
	InRt.ReleaseStream(stream);
	return status;
}

OaStatus OaComputeGraph::WaitForPendingReplay(const OaVkDevice& InDevice) {
	if (ReplayTimelineValue_ == 0 or not ReplayTimelineSem_.Semaphore) {
		return OaStatus::Ok();
	}
	return ReplayTimelineSem_.Wait(InDevice, ReplayTimelineValue_);
}

OaResult<OaCompletionToken> OaComputeGraph::ReplayAsync(OaComputeEngine& InRt)
{
	OA_RETURN_IF_ERROR(Replay(InRt));
	return LastCompletion(InRt.Device);
}

OaCompletionToken OaComputeGraph::LastCompletion(const OaVkDevice& InDevice) const
{
	return ReplayTimelineValue_ != 0U && ReplayTimelineSem_.Semaphore
		? OaCompletionToken(InDevice, ReplayTimelineSem_, ReplayTimelineValue_)
		: OaCompletionToken();
}

OaStatus OaComputeGraph::RecordReplay(OaComputeEngine& InRt, void* InPrimaryCommandBuffer) const {
	(void)InRt;
	if (Nodes_.Empty()) return OaStatus::Ok();
	if (!Compiled_ || !SecondaryCb_) {
		return OaStatus::Error("graph record replay: not compiled — call Compile() first");
	}
	VkCommandBuffer primary = static_cast<VkCommandBuffer>(InPrimaryCommandBuffer);
	VkCommandBuffer secondary = static_cast<VkCommandBuffer>(SecondaryCb_);
	vkCmdExecuteCommands(primary, 1, &secondary);
	return OaStatus::Ok();
}

// ─── Phase 3: Memory aliasing analysis ────────────────────────────────────────

OaVec<OaBufferLifetime> OaComputeGraph::ComputeLifetimes() const {
	// Map: VkBuffer handle -> (first_access, last_access, size)
	struct LifetimeEntry {
		OaU64 Size = 0;
		OaU32 First = UINT32_MAX;
		OaU32 Last = 0;
	};
	OaHashMap<void*, LifetimeEntry> map;

	for (OaU32 i = 0; i < static_cast<OaU32>(Nodes_.Size()); ++i) {
		auto& node = Nodes_[i];
		for (OaU32 j = 0; j < static_cast<OaU32>(node.Buffers.Size()); ++j) {
			void* handle = node.Buffers[j].Buffer;
			if (!handle) continue;
			auto emplaceResult = map.Emplace(handle, LifetimeEntry{});
			LifetimeEntry& entry = emplaceResult.first->second;
			if (i < entry.First) entry.First = i;
			if (i > entry.Last) entry.Last = i;
			if (node.Buffers[j].Size > entry.Size) entry.Size = node.Buffers[j].Size;
		}
		if (node.Indirect && node.IndirectBuffer.Buffer) {
			void* handle = node.IndirectBuffer.Buffer;
			auto emplaceResult = map.Emplace(handle, LifetimeEntry{});
			LifetimeEntry& entry = emplaceResult.first->second;
			if (i < entry.First) entry.First = i;
			if (i > entry.Last) entry.Last = i;
			if (node.IndirectBuffer.Size > entry.Size) {
				entry.Size = node.IndirectBuffer.Size;
			}
		}
	}

	OaVec<OaBufferLifetime> result;
	result.Reserve(static_cast<OaU32>(map.Size()));
	for (auto& [handle, entry] : map) {
		OaBufferLifetime lt;
		lt.Buffer = handle;
		lt.Size = entry.Size;
		lt.FirstAccess = entry.First;
		lt.LastAccess = entry.Last;
		result.PushBack(lt);
	}

	std::sort(result.Begin(), result.End(),
		[](const OaBufferLifetime& a, const OaBufferLifetime& b) {
			return a.FirstAccess < b.FirstAccess;
		});

	return result;
}

OaVec<OaAliasGroup> OaComputeGraph::ComputeAliasGroups() const {
	auto lifetimes = ComputeLifetimes();
	OaVec<OaAliasGroup> groups;

	// Greedy interval coloring: assign each buffer to the first group
	// where it doesn't overlap with any existing member.
	for (auto& lt : lifetimes) {
		bool placed = false;
		for (auto& group : groups) {
			bool overlaps = false;
			for (auto& member : group.Members) {
				if (lt.FirstAccess <= member.LastAccess
					&& lt.LastAccess >= member.FirstAccess)
				{
					overlaps = true;
					break;
				}
			}
			if (!overlaps) {
				group.Members.PushBack(lt);
				if (lt.Size > group.RequiredSize)
					group.RequiredSize = lt.Size;
				placed = true;
				break;
			}
		}
		if (!placed) {
			OaAliasGroup newGroup;
			newGroup.Members.PushBack(lt);
			newGroup.RequiredSize = lt.Size;
			groups.PushBack(std::move(newGroup));
		}
	}

	return groups;
}

// ─── Queries ──────────────────────────────────────────────────────────────────

OaGraphStats OaComputeGraph::GetStats() const {
	OaGraphStats stats;
	stats.DispatchCount = static_cast<OaU32>(Nodes_.Size());
	stats.BarrierCount = BarrierCount_;
	stats.DescriptorSetCount = static_cast<OaU32>(DescriptorPools_.Size());
	// PR-3 G v3 — graph-optimizer observability fields.
	stats.GroupCount    = GroupCount_;
	stats.MaxGroupSize  = MaxGroupSize_;
	stats.ReorderCount  = ReorderCount_;
	stats.NodeCountStat = static_cast<OaU32>(Nodes_.Size());
	stats.WarBarrierCount = WarBarrierCount_;
	stats.IndirectBarrierCount = IndirectBarrierCount_;

	// Compute total buffer bytes and alias savings
	auto lifetimes = ComputeLifetimes();
	for (auto& lt : lifetimes) {
		stats.TotalBufferBytes += lt.Size;
	}

	auto groups = ComputeAliasGroups();
	OaU64 aliasedTotal = 0;
	for (auto& g : groups) {
		aliasedTotal += g.RequiredSize;
	}
	if (stats.TotalBufferBytes > aliasedTotal) {
		stats.PotentialAliasSavings = stats.TotalBufferBytes - aliasedTotal;
	}

	return stats;
}

// ─── Lifecycle ────────────────────────────────────────────────────────────────

void OaComputeGraph::Invalidate(const OaVkDevice& InDevice) {
	VkDevice dev = static_cast<VkDevice>(InDevice.Device);

	for (void* pool : DescriptorPools_) {
		vkDestroyDescriptorPool(dev, static_cast<VkDescriptorPool>(pool), nullptr);
	}
	DescriptorPools_.Clear();

	if (SecondaryPool_) {
		// Explicitly free secondary command buffer before destroying pool (Vulkan validation requirement)
		if (SecondaryCb_) {
			VkCommandBuffer scb = static_cast<VkCommandBuffer>(SecondaryCb_);
			vkFreeCommandBuffers(dev, static_cast<VkCommandPool>(SecondaryPool_), 1, &scb);
			SecondaryCb_ = nullptr;
		}
		vkDestroyCommandPool(dev, static_cast<VkCommandPool>(SecondaryPool_), nullptr);
		SecondaryPool_ = nullptr;
	}

	if (PrimaryPool_) {
		if (PrimaryCb_) {
			VkCommandBuffer pcb = static_cast<VkCommandBuffer>(PrimaryCb_);
			vkFreeCommandBuffers(dev, static_cast<VkCommandPool>(PrimaryPool_), 1, &pcb);
			PrimaryCb_ = nullptr;
		}
		vkDestroyCommandPool(dev, static_cast<VkCommandPool>(PrimaryPool_), nullptr);
		PrimaryPool_ = nullptr;
	}

	if (ReplayTimelineSem_.Semaphore) {
		ReplayTimelineSem_.Destroy(InDevice);
	}
	ReplayTimelineValue_ = 0;

	Compiled_ = false;
	LastCompileHash_ = 0;
	BarrierCount_ = 0;
	WarBarrierCount_ = 0;
	IndirectBarrierCount_ = 0;
}

void OaComputeGraph::Reset() {
	Nodes_.Clear();
	// Compiled state leaks if not destroyed — use Reset(device) instead
}

void OaComputeGraph::Reset(const OaVkDevice& InDevice) {
	Invalidate(InDevice);
	Nodes_.Clear();
}

void OaComputeGraph::ClearNodes() {
	Nodes_.Clear();
	Compiled_ = false;
	BarrierCount_ = 0;
	WarBarrierCount_ = 0;
	IndirectBarrierCount_ = 0;
	// Keep SecondaryPool_, SecondaryCb_ for reuse in next Compile().
	// DescriptorPools_ are cleaned up at the start of Compile().
}

void OaComputeGraph::Destroy(const OaVkDevice& InDevice) {
	Invalidate(InDevice);
	Nodes_.Clear();
	Nodes_.ShrinkToFit();
}
