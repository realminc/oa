// OaScheduler — experimental static mesh routing
//
// Routes Heavy work to Primary and Light work to Auxiliary. It has no measured
// transfer-cost/load model and must not be described as optimal or supported
// heterogeneous execution.
//
// Existing code dispatches to the primary device by default. Experimental
// graph mapping may query this router with an explicit static hint.

#pragma once

#include <Oa/Core/Types.h>

class OaDeviceMesh;
class OaDeviceNode;

static constexpr OaU32 OA_NODE_AUTO = UINT32_MAX;

// Workload classification for scheduler routing.
enum class OaComputeClass : OaU8 {
	Heavy,     // Large matmul, convolution, NTT — route to Primary
	Light,     // Norms, activations, reductions — can go to Auxiliary
	Transfer,  // Data movement — route to transfer queue/device
	Any,       // No preference — scheduler decides based on load
};

[[nodiscard]] constexpr OaStringView OaComputeClassName(OaComputeClass InClass) noexcept {
	switch (InClass) {
		case OaComputeClass::Heavy:    return "Heavy";
		case OaComputeClass::Light:    return "Light";
		case OaComputeClass::Transfer: return "Transfer";
		case OaComputeClass::Any:      return "Any";
		default:                       return "Unknown";
	}
}

// Static routing hint for experimental graph mapping or direct scheduler query.
class OaDispatchHint {
public:
	OaComputeClass Class = OaComputeClass::Any;
	OaU32 PreferNode = OA_NODE_AUTO;  // Explicit node override
};

// Static experimental router. Future lowering must be capability-queried,
// measured, dependency-aware, and private policy.
class OaScheduler {
public:
	OaDeviceMesh* Mesh = nullptr;

	[[nodiscard]] static OaScheduler Create(OaDeviceMesh& InMesh);

	// Route a dispatch to the best node for this workload class.
	[[nodiscard]] OaDeviceNode* Route(const OaDispatchHint& InHint);
	[[nodiscard]] OaDeviceNode* Route(OaComputeClass InClass);
};
