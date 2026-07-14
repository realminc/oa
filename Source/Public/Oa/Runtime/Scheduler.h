// OaScheduler — Affinity-based dispatch routing
//
// Routes compute dispatches to the best device in the mesh based on
// workload classification (heavy/light/transfer) and device profiles.
//
// Existing code dispatches to the primary device by default.
// Opt-in hints enable multi-device routing.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/Sync.h>

class OaDeviceMesh;
class OaDeviceNode;
class OaVkDevice;

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

// Dispatch routing hint — passed to Run() or graph construction.
class OaDispatchHint {
public:
	OaComputeClass Class = OaComputeClass::Any;
	OaU32 PreferNode = OA_NODE_AUTO;  // Explicit node override
	OaBool Async = false;              // Don't block caller
};

// Handle for async dispatches — owns a timeline semaphore + target value.
// Move-only: prevents double-destroy of the underlying VkSemaphore.
class OaDispatchTicket {
public:
	OaVkTimelineSemaphore Semaphore;
	OaU64 Value = 0;
	OaU32 NodeIndex = 0;

	OaDispatchTicket() = default;
	OaDispatchTicket(const OaDispatchTicket&) = delete;
	OaDispatchTicket& operator=(const OaDispatchTicket&) = delete;
	OaDispatchTicket(OaDispatchTicket&& InOther) noexcept
		: Semaphore(InOther.Semaphore), Value(InOther.Value), NodeIndex(InOther.NodeIndex) {
		InOther.Semaphore.Semaphore = nullptr;
		InOther.Value = 0;
	}
	OaDispatchTicket& operator=(OaDispatchTicket&& InOther) noexcept {
		if (this != &InOther) {
			Semaphore = InOther.Semaphore;
			Value = InOther.Value;
			NodeIndex = InOther.NodeIndex;
			InOther.Semaphore.Semaphore = nullptr;
			InOther.Value = 0;
		}
		return *this;
	}

	// Blocks until the async operation completes, then destroys the semaphore.
	[[nodiscard]] OaStatus Wait(const OaVkDevice& InDevice, OaU64 InTimeoutNs = UINT64_MAX);
	[[nodiscard]] OaBool IsComplete(const OaVkDevice& InDevice) const;
};

// Dispatch router — picks the best device node for a given workload.
// Stub: always returns Primary. Future: load-aware, profile-driven.
class OaScheduler {
public:
	OaDeviceMesh* Mesh = nullptr;

	[[nodiscard]] static OaScheduler Create(OaDeviceMesh& InMesh);

	// Route a dispatch to the best node for this workload class.
	[[nodiscard]] OaDeviceNode* Route(const OaDispatchHint& InHint);
	[[nodiscard]] OaDeviceNode* Route(OaComputeClass InClass);
};
