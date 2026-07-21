#pragma once

#include <Oa/Core/MatrixShape.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/SemanticGraphFwd.h>

using OaSemanticResourceId = OaU32;
constexpr OaSemanticResourceId OaInvalidSemanticResourceId = UINT32_MAX;

// Deterministic bridge from a handle-free semantic value to one captured
// physical-resource identity. Resource IDs are assigned by first semantic
// appearance; Vulkan handles and host addresses never enter this descriptor.
class OaSemanticStorageBinding {
public:
	OaSemanticValueId Value = OaInvalidSemanticValueId;
	OaSemanticResourceId Resource = OaInvalidSemanticResourceId;
	OaU64 ByteOffset = 0;
	OaMatrixShape Shape{};
	OaArray<OaI64, OA_MAX_TENSOR_DIMS> Strides{};
	OaScalarType Dtype = OaScalarType::Float32;
	OaBool SemanticExternal = false;
	OaBool StableReplayInput = false;
	OaBool ObservedOutput = false;
};

// Handle-free liveness contract for one captured physical resource. Resources
// that have not yet gained schema-v2 semantic values still appear here, which
// lets compatibility operations protect loss/readback outputs during migration.
class OaCapturedResourceDesc {
public:
	OaSemanticResourceId Resource = OaInvalidSemanticResourceId;
	OaBool SemanticExternal = false;
	OaBool StableReplayInput = false;
	OaBool StableTransient = false;
	OaBool ObservedOutput = false;
	OaBool HasLifetime = false;
	OaBool AliasCandidate = false;
	OaBool AliasMaterialized = false;
	OaMemoryPlacement Placement = OaMemoryPlacement::Auto;
	OaU64 ByteSize = 0;
	OaU32 FirstAccess = 0;
	OaU32 LastAccess = 0;
	// Strong references retained by the source execution session while capture
	// transactionally compiles a replacement graph. The captured resource-table
	// reference itself is excluded. This count lets materialization reject any
	// unaccounted user matrix/view or autograd owner without exposing pointers.
	OaU32 CaptureRetainedOwnerCount = 0;

	[[nodiscard]] OaBool IsExternallyLive() const noexcept {
		return SemanticExternal or StableReplayInput or ObservedOutput;
	}
};
