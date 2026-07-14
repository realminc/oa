// OaDeviceMesh — Vulkan Heterogeneous Compute Mesh
//
// Multi-device topology: enumerate, profile, classify, and connect all GPUs.
// Supports any mix of vendors (NVIDIA + Intel + AMD + software).
// GPU big.LITTLE: Primary (heavy compute) + Auxiliary (light ops) pipelining.
//
// Layer 1 of the multi-GPU stack:
//   topology.h  — hardware abstraction (this file)
//   scheduler.h — dispatch routing
//   collective.h — distributed primitives (AllReduce, Broadcast)
//   cluster.h   — multi-node stubs

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Thread.h>
#include <Oa/Runtime/Device.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Pipeline.h>
#include <Oa/Runtime/Bindless.h>
#include <Oa/Runtime/Stream.h>

class OaEngineConfig;

// Device role within the mesh — assigned by profiling at init.
enum class OaDeviceRole : OaU8 {
	Primary,     // Heaviest compute capability (largest VRAM, most CUs)
	Auxiliary,   // Secondary compute (iGPU, smaller dGPU)
	Transfer,    // DMA engine only (future)
};

[[nodiscard]] constexpr OaStringView OaDeviceRoleName(OaDeviceRole InRole) noexcept {
	switch (InRole) {
		case OaDeviceRole::Primary:   return "Primary";
		case OaDeviceRole::Auxiliary: return "Auxiliary";
		case OaDeviceRole::Transfer:  return "Transfer";
		default:                      return "Unknown";
	}
}

// Interconnect topology between two devices.
enum class OaInterconnect : OaU8 {
	HostMemory,  // Staging through CPU RAM (PCIe, universal)
	PciE,        // Direct PCIe peer-to-peer (same IOMMU group)
	NvLink,      // NVIDIA NVLink (high-bandwidth, same vendor)
	Xgmi,        // AMD xGMI / Infinity Fabric (MI250X, same vendor)
	SharedRam,   // Shared system memory (iGPU + dGPU on same die/SoC)
};

[[nodiscard]] constexpr OaStringView OaInterconnectName(OaInterconnect InType) noexcept {
	switch (InType) {
		case OaInterconnect::HostMemory: return "HostMemory";
		case OaInterconnect::PciE:       return "PCIe";
		case OaInterconnect::NvLink:     return "NVLink";
		case OaInterconnect::Xgmi:       return "xGMI";
		case OaInterconnect::SharedRam:  return "SharedRAM";
		default:                         return "Unknown";
	}
}

// Benchmarked device capabilities — filled by micro-benchmark or device properties.
class OaDeviceProfile {
public:
	OaF64 PeakTflopsF32 = 0.0;     // Estimated FP32 TFLOPS
	OaF64 MemBandwidthGBs = 0.0;   // Estimated memory bandwidth (GB/s)
	OaU64 VramBytes = 0;            // Total VRAM (device-local heap)
	OaU64 SharedMemBytes = 0;       // Shared/local memory per workgroup

	// Relative scores (primary = 1.0, others scaled proportionally)
	OaF32 MatmulScore = 1.0f;      // Relative matmul throughput
	OaF32 ReductionScore = 1.0f;   // Relative reduction throughput
	OaF32 TransferScore = 1.0f;    // Host-device transfer bandwidth score
};

// One GPU in the mesh — owns its own Vulkan device context.
// In single-device mode, there is one node and the engine aliases it.
class OaDeviceNode {
public:
	OaVkDevice Device;
	OaVma Allocator;
	OaPipelineRegistry Pipelines;
	OaBindlessHeap Bindless;

	OaDeviceRole Role = OaDeviceRole::Primary;
	OaDeviceProfile Profile;
	OaU32 Index = 0;

	// Per-node stream pool
	OaVec<OaUniquePtr<OaVkStream>> StreamPool;
	OaVec<OaU32> FreeStack;
	OaUniquePtr<OaSpinlock> StreamLock;

	OaDeviceNode();
	~OaDeviceNode() = default;
	// Allow move for EmplaceBack (will use in-place construction, not actual move)
	OaDeviceNode(OaDeviceNode&&) noexcept = default;
	OaDeviceNode& operator=(OaDeviceNode&&) noexcept = default;
	OaDeviceNode(const OaDeviceNode&) = delete;
	OaDeviceNode& operator=(const OaDeviceNode&) = delete;

	void Destroy();

	OaVkStream* AcquireStream();
	void ReleaseStream(OaVkStream* InStream);
};

// Transport mode for cross-device data movement.
enum class OaTransport : OaU8 {
	HostStaging,  // CPU memcpy through host-visible mapped pointers (universal)
	DmaBuf,       // Linux DMA-BUF fd export/import (zero-copy, same-vendor)
};

[[nodiscard]] constexpr OaStringView OaTransportName(OaTransport InMode) noexcept {
	switch (InMode) {
		case OaTransport::HostStaging: return "HostStaging";
		case OaTransport::DmaBuf:      return "DmaBuf";
		default:                       return "Unknown";
	}
}

// Transfer link between two mesh nodes.
class OaTransferLink {
public:
	OaU32 SrcNode = 0;
	OaU32 DstNode = 0;
	OaF64 BandwidthGBs = 0.0;        // Measured or estimated transfer bandwidth
	OaBool SharedMemory = false;      // Devices share system memory (laptop iGPU)
	OaBool PeerToPeer = false;        // Same vendor, direct DMA possible
	OaInterconnect Topology = OaInterconnect::HostMemory;
	OaTransport BestTransport = OaTransport::HostStaging;
	OaBool DmaBufCapable = false;
};

// Multi-device topology — all GPUs + their interconnections.
// Create() enumerates physical devices, profiles them, and builds the mesh.
// In single-device mode: one node, zero links, Primary pointer set.
class OaDeviceMesh {
public:
	OaVec<OaDeviceNode> Nodes;
	OaVec<OaTransferLink> Links;
	OaDeviceNode* Primary = nullptr;

	// Serializes OaVkLoadDevice switching for non-primary operations
	OaUniquePtr<OaSpinlock> DeviceLoadLock;

	OaDeviceMesh();
	~OaDeviceMesh() = default;
	// Allow move for return value optimization (RVO/NRVO)
	// Move constructor will never actually be called due to copy elision in C++17
	OaDeviceMesh(OaDeviceMesh&&) noexcept = default;
	OaDeviceMesh& operator=(OaDeviceMesh&&) noexcept = default;
	OaDeviceMesh(const OaDeviceMesh&) = delete;
	OaDeviceMesh& operator=(const OaDeviceMesh&) = delete;

	[[nodiscard]] static OaResult<OaDeviceMesh> Create(const OaEngineConfig& InConfig);
	void Destroy();

	[[nodiscard]] OaDeviceNode* GetNode(OaU32 InIndex);
	[[nodiscard]] const OaDeviceNode* GetNode(OaU32 InIndex) const;
	[[nodiscard]] OaDeviceNode* GetByRole(OaDeviceRole InRole);
	[[nodiscard]] OaU32 NodeCount() const { return static_cast<OaU32>(Nodes.Size()); }

	// Cross-device buffer copy via host-visible staging (mapped pointer memcpy).
	[[nodiscard]] OaStatus CopyBuffer(
		OaU32 InSrcNode, const OaVkBuffer& InSrc,
		OaU32 InDstNode, OaVkBuffer& InDst,
		OaU64 InSize);

	// Look up transfer path between two nodes. Returns nullptr if no link.
	[[nodiscard]] const OaTransferLink* FindLink(OaU32 InSrc, OaU32 InDst) const;

	// NxN bandwidth matrix (GB/s). [i][j] = bandwidth from node i to node j.
	// Diagonal = device-local bandwidth. Off-diagonal = inter-device.
	[[nodiscard]] OaVec<OaVec<OaF64>> BandwidthMatrix() const;

	// Scratch buffer for collective operations on the primary node.
	// Lazy-allocated on first collective call, freed in Destroy().
	OaVkBuffer ScratchBuf;
	OaU64 ScratchSize = 0;

	// Ensure scratch buffer is at least InSize bytes on the primary node.
	[[nodiscard]] OaStatus EnsureScratch(OaU64 InSize);

	void PrintTopology() const;
};
