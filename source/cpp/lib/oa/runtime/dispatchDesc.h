#pragma once

#include <oa/core/types.h>
#include <oa/core/bufferAccess.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/executionStats.h>
#include <oa/runtime/semanticGraphFwd.h>

namespace oa {

class Matrix;

// Private queue placement is execution metadata, not operation semantics. A planner or
// scheduler may annotate a dispatch; the graph and engine consume the result.
enum class QueueHint : U8 {
	Compute,
	AsyncCompute,
	Transfer,
};

// Non-owning description of one compute dispatch. Recorders copy this data into
// their owning IR immediately, so stack arrays and push structures are valid.
//
// This is the lowering boundary between semantic operation layers
// (oa::FnMatrix, oa::FnAudio, planners) and runtime execution. It deliberately has
// no MatMul, Linear, audio, vision, or autograd knowledge.
class ComputeDispatchDesc {
public:
	// Optional semantic provenance. Multiple owners describe one fused executable
	// node without teaching the executable graph domain-specific behavior. The
	// recorder copies the span before returning.
	oa::StringView operation;
	oa::Span<const oa::U32> semanticOps;
	oa::U64 implementationId = 0;
	oa::U64 opContractHash = 0;
	// Shape/layout/precision-specific lowering identity. Unlike the semantic
	// operation contract, this may vary between invocations of the same op.
	oa::U64 problemContractHash = 0;
	oa::U64 kernelContentHash = 0;
	oa::KernelSelectionKind kernelSelection = oa::KernelSelectionKind::Unspecified;
	oa::StringView kernel;
	oa::Span<oavk::Buffer> buffers;
	oa::Span<oa::SharedPtr<oavk::Buffer>> bufferOwners;
	oa::Span<oa::BufferAccess> access;
	const void* pushData = nullptr;
	oa::U32 pushSize = 0;
	oa::U32 dtype = 0;
	oa::U32 groupsX = 1;
	oa::U32 groupsY = 1;
	oa::U32 groupsZ = 1;
	oavk::Buffer indirectBuffer;
	oa::U64 indirectOffset = 0;
	oa::Bool indirect = false;
	QueueHint queue = QueueHint::Compute;
};

// Matrix-aware lowering request. context resolves semantic storage handles into
// raw vulkan buffers and owning references, then immediately records the
// resulting oa::ComputeDispatchDesc. The nested dispatch must not provide raw
// buffers/bufferOwners or indirect fields: there is one source for each.
class MatrixDispatchDesc {
public:
	ComputeDispatchDesc dispatch;
	oa::Span<const oa::Matrix* const> matrices;
	const oa::Matrix* indirectArgs = nullptr;
	oa::U64 indirectOffset = 0;
};

} // namespace oa
