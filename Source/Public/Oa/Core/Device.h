// OA CORE - Device Abstraction
//
// Vulkan-native device model. No CUDA. No ROCm. No Metal.
// All GPU compute goes through Vulkan 1.4. Software Vulkan (lavapipe) for CI.
//
// Semantics (do not conflate):
// - OaDeviceType::Host — no Vulkan tensor placement; placeholder when no engine / unknown host.
//   Not the same as a Vulkan "CPU" physical device.
// - OaDeviceType::VkCpu — Vulkan VK_PHYSICAL_DEVICE_TYPE_CPU (OAV ICD, lavapipe, SwiftShader, etc.).
//   Real VkPhysicalDevice; tensors and dispatch use the normal Vulkan path.
// - OaDeviceType::VkVirtualGpu — VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU (SR-IOV, hypervisor passthrough, etc.).
// - OaDeviceType::VkOther — VK_PHYSICAL_DEVICE_TYPE_OTHER (unclassified Vulkan implementation).
// Vulkan VkPhysicalDeviceType is fully covered by the Vk* variants above plus Host (non-Vulkan).
//
// OaDevice::Index — when Type is Vulkan, this is the mesh node index (OaDeviceMesh); 0 is primary.
#pragma once

#include <Oa/Core/Types.h>

// DEVICE TYPE
enum class OaDeviceType : OaU8 {
	Host         = 0,   // No Vulkan tensor device / unknown host (not VkCpu)
	VkDiscrete   = 1,   // VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	VkIntegrated = 2,   // VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
	VkCpu        = 3,   // VK_PHYSICAL_DEVICE_TYPE_CPU
	VkVirtualGpu = 4,   // VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU
	VkOther      = 5,   // VK_PHYSICAL_DEVICE_TYPE_OTHER
};

[[nodiscard]] constexpr OaStringView OaDeviceTypeName(OaDeviceType InType) noexcept {
	switch (InType) {
		case OaDeviceType::Host:         return "Host";
		case OaDeviceType::VkDiscrete:   return "VkDiscrete";
		case OaDeviceType::VkIntegrated: return "VkIntegrated";
		case OaDeviceType::VkCpu:        return "VkCpu";
		case OaDeviceType::VkVirtualGpu: return "VkVirtualGpu";
		case OaDeviceType::VkOther:      return "VkOther";
		default:                         return "Unknown";
	}
}

[[nodiscard]] constexpr bool OaIsVulkanDevice(OaDeviceType InType) noexcept {
	return InType == OaDeviceType::VkDiscrete || InType == OaDeviceType::VkIntegrated ||
		InType == OaDeviceType::VkCpu || InType == OaDeviceType::VkVirtualGpu ||
		InType == OaDeviceType::VkOther;
}


class OaDevice {
public:
	// Logical device id: mesh node kind + index (see header comment on OaDeviceType / Index).

	// Data, class members.
	OaDeviceType Type = OaDeviceType::Host;
	OaI32 Index = 0;

	// Constructors.
	constexpr OaDevice() = default;
	constexpr OaDevice(OaDeviceType InType, OaI32 InIndex = 0) noexcept
		: Type(InType)
		, Index(InIndex)
	{}

	// Methods.
	// Non-Vulkan sentinel only (no VkDevice). Not "CPU execution": OaDeviceType::VkCpu is still IsVulkan().
	[[nodiscard]] constexpr bool IsHost() const noexcept { return Type == OaDeviceType::Host; }
	[[nodiscard]] constexpr bool IsVulkan() const noexcept { return OaIsVulkanDevice(Type); }
	// Any Vulkan physical device kind (discrete, integrated, VkCpu/OAV, virtual, other). Not hardware "GPU" only.
	[[nodiscard]] constexpr bool IsGpu() const noexcept { return IsVulkan(); }

	// Operators.
	[[nodiscard]] constexpr bool operator==(const OaDevice& InOther) const noexcept {
		return Type == InOther.Type && Index == InOther.Index;
	}
	[[nodiscard]] constexpr bool operator!=(const OaDevice& InOther) const noexcept {
		return !(*this == InOther);
	}
};

inline constexpr OaDevice OA_DEVICE_HOST{OaDeviceType::Host, 0};

// MEMORY LOCATION
//
// Logical placement for buffers (not a silicon map). Discrete vs SoC:
// - Discrete GPU (e.g. dGPU in a Strix-class laptop): Device = separate VRAM; Host = system RAM;
//   Shared = CPU-mapped VRAM (ReBAR / Smart Access Memory / large BAR) — same VRAM bytes, two views.
// - Unified memory / UMA SoCs (Apple Silicon, Snapdragon X Elite, many iGPUs): one DRAM pool for
//   CPU + GPU + NPU. Vulkan often exposes heaps with HOST_VISIBLE and DEVICE_LOCAL on the same
//   memory — treat that as Shared (one pool, visible to both). Pure pageable CPU malloc with no
//   device mapping stays Host. Device means "allocated as device-local / GPU-primary" even when
//   it is physically the same DRAM as the CPU (allocator hint + visibility flags), not a second chip.
// NPU-only carve-outs without a separate Vulkan heap are not a fourth category here until modeled.
enum class OaMemoryLocation : OaU8 {
	Host   = 0,   // CPU-owned pageable RAM; not a device-mapped GPU resource
	Device = 1,   // Device-local (discrete VRAM, or GPU-primary allocation on UMA)
	Shared = 2,   // Host-visible device memory: ReBAR/SAM, or UMA pool visible to CPU + GPU
};

[[nodiscard]] constexpr OaStringView OaMemoryLocationName(OaMemoryLocation InLoc) noexcept {
	switch (InLoc) {
		case OaMemoryLocation::Host:   return "Host";
		case OaMemoryLocation::Device: return "Device";
		case OaMemoryLocation::Shared: return "Shared";
		default:                       return "Unknown";
	}
}

// Device memory budget for ML / tuning (runtime TU when OaComputeEngine is linked).
// VkDiscrete / VkIntegrated / VkVirtualGpu: device-local VRAM capacity from Vulkan heaps.
// VkCpu: host physical RAM (CPU path uses system memory, not VRAM).
// VkOther (incl. future NPU-class devices): not queryable here — zeros.
class OaMemoryUsage {
public:
	OaU64 TotalBytes = 0;
	OaU64 FreeBytes = 0;
	OaU64 UsedBytes = 0;
	OaF64 UsedPercent = 0.0;
};

[[nodiscard]] OaMemoryUsage OaGetMemoryUsage(OaDevice InDevice = OA_DEVICE_HOST);
