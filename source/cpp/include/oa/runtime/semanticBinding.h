#pragma once

#include <oa/core/device.h>
#include <oa/core/matrixShape.h>
#include <oa/core/types.h>
#include <oa/runtime/semanticGraphFwd.h>

namespace oa {

constexpr oa::U32 invalidSemanticResourceId = UINT32_MAX;

// Deterministic bridge from a handle-free semantic value to one captured
// physical-resource identity. resource IDs are assigned by first semantic
// appearance; vulkan handles and host addresses never enter this descriptor.
class SemanticStorageBinding {
public:
	oa::U32 value = oa::invalidSemanticValueId;
	oa::U32 resource = oa::invalidSemanticResourceId;
	oa::U64 byteOffset = 0;
	oa::MatrixShape shape{};
	oa::Array<oa::I64, OA_MAX_TENSOR_DIMS> strides{};
	oa::ScalarType dtype = oa::ScalarType::Float32;
	oa::Bool semanticExternal = false;
	oa::Bool stableReplayInput = false;
	oa::Bool observedOutput = false;
};

// handle-free liveness contract for one captured physical resource. resources
// that have not yet gained schema-v2 semantic values still appear here, which
// lets compatibility operations protect loss/readback outputs during migration.
class CapturedResourceDesc {
public:
	oa::U32 resource = oa::invalidSemanticResourceId;
	oa::Bool semanticExternal = false;
	oa::Bool stableReplayInput = false;
	oa::Bool stableTransient = false;
	oa::Bool observedOutput = false;
	oa::Bool hasLifetime = false;
	oa::Bool aliasCandidate = false;
	oa::Bool aliasMaterialized = false;
	oa::MemoryPlacement placement = oa::MemoryPlacement::Auto;
	oa::U64 byteSize = 0;
	oa::U32 firstAccess = 0;
	oa::U32 lastAccess = 0;
	// Strong references retained by the source execution session while capture
	// transactionally compiles a replacement graph. The captured resource-table
	// reference itself is excluded. This count lets materialization reject any
	// unaccounted user matrix/view or autograd owner without exposing pointers.
	oa::U32 captureRetainedOwnerCount = 0;

	[[nodiscard]] oa::Bool isExternallyLive() const noexcept {
		return semanticExternal or stableReplayInput or observedOutput;
	}
};

} // namespace oa
