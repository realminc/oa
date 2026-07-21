#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/SemanticGraphFwd.h>

class OaMatrix;

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
	// Optional semantic provenance. Multiple owners describe one fused executable
	// node without teaching the executable graph domain-specific behavior. The
	// recorder copies the span before returning.
	OaStringView Operation;
	OaSpan<const OaSemanticOperationId> SemanticOperations;
	OaU64 ImplementationId = 0;
	OaU64 OperationContractHash = 0;
	// Shape/layout/precision-specific lowering identity. Unlike the semantic
	// operation contract, this may vary between invocations of the same op.
	OaU64 ProblemContractHash = 0;
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

// Matrix-aware lowering request. Context resolves semantic storage handles into
// raw Vulkan buffers and owning references, then immediately records the
// resulting OaComputeDispatchDesc. The nested dispatch must not provide raw
// Buffers/BufferOwners or indirect fields: there is one source for each.
class OaMatrixDispatchDesc {
public:
	OaComputeDispatchDesc Dispatch;
	OaSpan<const OaMatrix* const> Matrices;
	const OaMatrix* IndirectArgs = nullptr;
	OaU64 IndirectOffset = 0;
};
