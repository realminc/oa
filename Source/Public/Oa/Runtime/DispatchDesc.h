#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Allocator.h>

// Queue placement is execution metadata, not operation semantics. A planner or
// scheduler may annotate a dispatch; the graph and engine consume the result.
enum class OaQueueHint : OaU8 {
	Compute,
	AsyncCompute,
	Transfer,
};

// Non-owning description of one compute dispatch. Recorders copy this data into
// their owning IR immediately, so stack arrays and push structures are valid.
//
// This is the lowering boundary between semantic operation layers
// (OaFnMatrix, OaFnAudio, planners) and runtime execution. It deliberately has
// no MatMul, Linear, audio, vision, or autograd knowledge.
class OaComputeDispatchDesc {
public:
	// Optional semantic provenance. These fields identify what the dispatch
	// implements without teaching the executable graph domain-specific behavior.
	// A zero id/hash means the lowering did not provide that metadata yet.
	OaStringView Operation;
	OaU64 ImplementationId = 0;
	OaU64 OperationContractHash = 0;
	OaU64 KernelContentHash = 0;
	OaStringView Kernel;
	OaSpan<OaVkBuffer> Buffers;
	OaSpan<OaSharedPtr<OaVkBuffer>> BufferOwners;
	OaSpan<OaBufferAccess> Access;
	const void* PushData = nullptr;
	OaU32 PushSize = 0;
	OaU32 Dtype = 0;
	OaU32 GroupsX = 1;
	OaU32 GroupsY = 1;
	OaU32 GroupsZ = 1;
	OaVkBuffer IndirectBuffer;
	OaU64 IndirectOffset = 0;
	OaBool Indirect = false;
	OaQueueHint Queue = OaQueueHint::Compute;
	OaU32 NodeIndex = 0;
};
