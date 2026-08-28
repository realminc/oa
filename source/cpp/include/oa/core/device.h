// OA CORE - Device Abstraction
//
// vulkan-native device model. No CUDA. No ROCm. No Metal.
// All GPU compute goes through vulkan 1.4. software vulkan (lavapipe) for CI.
//
// semantics (do not conflate):
// - DeviceType::Host — no vulkan tensor placement; placeholder when no engine / unknown host.
//   Not the same as a vulkan "CPU" physical device.
// - DeviceType::VkCpu — vulkan VK_PHYSICAL_DEVICE_TYPE_CPU (OAV ICD, lavapipe, SwiftShader, etc.).
//   Real VkPhysicalDevice; tensors and dispatch use the normal vulkan path.
// - DeviceType::VkVirtualGpu — VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU (SR-IOV, hypervisor passthrough, etc.).
// - DeviceType::VkOther — VK_PHYSICAL_DEVICE_TYPE_OTHER (unclassified vulkan implementation).
// vulkan VkPhysicalDeviceType is fully covered by the Vk* variants above plus host (non-vulkan).
//
// Device::index is a logical placement index. The current local engine
// exposes only index 0; distributed rank and placement metadata remain values
// owned by a future distributed session.
#pragma once

#include <oa/core/types.h>

namespace oa {

// DEVICE TYPE
enum class DeviceType : oa::U8 {
	Host         = 0,   // No vulkan tensor device / unknown host (not VkCpu)
	VkDiscrete   = 1,   // VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	VkIntegrated = 2,   // VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
	VkCpu        = 3,   // VK_PHYSICAL_DEVICE_TYPE_CPU
	VkVirtualGpu = 4,   // VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU
	VkOther      = 5,   // VK_PHYSICAL_DEVICE_TYPE_OTHER
};

[[nodiscard]] constexpr oa::StringView deviceTypeName(DeviceType inType) noexcept {
	switch (inType) {
		case DeviceType::Host:         return "Host";
		case DeviceType::VkDiscrete:   return "VkDiscrete";
		case DeviceType::VkIntegrated: return "VkIntegrated";
		case DeviceType::VkCpu:        return "VkCpu";
		case DeviceType::VkVirtualGpu: return "VkVirtualGpu";
		case DeviceType::VkOther:      return "VkOther";
		default:                       return "Unknown";
	}
}

[[nodiscard]] constexpr bool isVulkanDevice(DeviceType inType) noexcept {
	return inType == DeviceType::VkDiscrete || inType == DeviceType::VkIntegrated ||
		inType == DeviceType::VkCpu || inType == DeviceType::VkVirtualGpu ||
		inType == DeviceType::VkOther;
}


class Device {
public:
	// Logical device kind + placement index (see the header comment).

	// Data, class members.
	DeviceType type = DeviceType::Host;
	oa::I32 index = 0;

	// Constructors.
	constexpr Device() = default;
	constexpr Device(DeviceType inType, oa::I32 inIndex = 0) noexcept
		: type(inType)
		, index(inIndex)
	{}

	// Methods.
	// Non-vulkan sentinel only (no VkDevice). Not "CPU execution": DeviceType::VkCpu is still isVulkan().
	[[nodiscard]] constexpr bool isHost() const noexcept { return type == DeviceType::Host; }
	[[nodiscard]] constexpr bool isVulkan() const noexcept { return isVulkanDevice(type); }
	// Any vulkan physical device kind (discrete, integrated, VkCpu/OAV, virtual, other). Not hardware "GPU" only.
	[[nodiscard]] constexpr bool isGpu() const noexcept { return isVulkan(); }

	// Operators.
	[[nodiscard]] constexpr bool operator==(const Device& inOther) const noexcept {
		return type == inOther.type && index == inOther.index;
	}
	[[nodiscard]] constexpr bool operator!=(const Device& inOther) const noexcept {
		return !(*this == inOther);
	}
};

inline constexpr Device HostDevice{DeviceType::Host, 0};

// MEMORY LOCATION
//
// Logical placement for buffers (not a silicon map). Discrete vs SoC:
// - Discrete GPU (e.g. dGPU in a Strix-class laptop): Device = separate VRAM; Host = system RAM;
//   Shared = CPU-mapped VRAM (ReBAR / Smart access memory / large BAR) — same VRAM bytes, two views.
// - Unified memory / UMA soCs (Apple Silicon, Snapdragon X Elite, many iGPUs): one DRAM pool for
//   CPU + GPU + NPU. vulkan often exposes heaps with HOST_VISIBLE and DEVICE_LOCAL on the same
//   memory — treat that as shared (one pool, visible to both). Pure pageable CPU malloc with no
//   device mapping stays Host. Device means "allocated as device-local / GPU-primary" even when
//   it is physically the same DRAM as the CPU (allocator hint + visibility flags), not a second chip.
// NPU-only carve-outs without a separate vulkan heap are not a fourth category here until modeled.
enum class MemoryLocation : oa::U8 {
	Host   = 0,   // CPU-owned pageable RAM; not a device-mapped GPU resource
	Device = 1,   // Device-local (discrete VRAM, or GPU-primary allocation on UMA)
	Shared = 2,   // Host-visible device memory: ReBAR/SAM, or UMA pool visible to CPU + GPU
};

[[nodiscard]] constexpr oa::StringView memoryLocationName(MemoryLocation inLoc) noexcept {
	switch (inLoc) {
		case MemoryLocation::Host:   return "Host";
		case MemoryLocation::Device: return "Device";
		case MemoryLocation::Shared: return "Shared";
		default:                       return "Unknown";
	}
}

// allocation intent carried by semantic values and engine requests.
// This is distinct from MemoryLocation: Auto is unresolved policy, while
// HostUpload and HostReadback name transfer direction. The Runtime allocator
// resolves the intent against the selected device and records the result on
// its private physical-storage descriptor.
enum class MemoryPlacement : oa::U8 {
	Auto = 0,
	DeviceLocal,
	HostUpload,
	HostReadback,
	Unified,
};

// memory capacity for value-only device descriptors. VkCpu reports host physical
// RAM because that identity is process-independent. Other vulkan descriptors do
// not identify a live allocator or physical device and return zero; query the
// exact oa::Engine for live vulkan budget/usage.
class MemoryUsage {
public:
	oa::U64 totalBytes = 0;
	oa::U64 freeBytes = 0;
	oa::U64 usedBytes = 0;
	oa::F64 usedPercent = 0.0;
};

[[nodiscard]] MemoryUsage getMemoryUsage(Device inDevice = HostDevice);

} // namespace oa
