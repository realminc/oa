#include <Oa/Core/Log.h>
#include <Oa/Runtime/Topology.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Device.h>
#include <Oa/Runtime/ExternalMemory.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Runtime/Instance.h>
#include <Oa/Core/Memory.h>

#include <algorithm>


// ─── Device Profiling (PCI-ID table + heuristics in Runtime/Device/) ─
static void ProfileDevice(OaDeviceNode& InOutNode) {
	auto& dev = InOutNode.Device;
	auto& prof = InOutNode.Profile;

	prof.VramBytes = dev.Info.Hardware.VramBytes;
	prof.PeakTflopsF32 = OaVkEstimatePeakTflopsF32Ex(
		dev.Info.Hardware.VendorId, dev.Info.Hardware.DeviceId, dev.Info.Hardware.DeviceType,
		dev.Info.Hardware.VramBytes
	);
	prof.MemBandwidthGBs = OaVkEstimateMemBandwidthGbpsEx(
		dev.Info.Hardware.VendorId, dev.Info.Hardware.DeviceId, dev.Info.Hardware.DeviceType,
		dev.Info.Hardware.VramBytes
	);

	// SharedMemBytes from device limits (queried via physical device properties)
	VkPhysicalDeviceProperties props{};
	vkGetPhysicalDeviceProperties(static_cast<VkPhysicalDevice>(dev.PhysicalDevice), &props);
	prof.SharedMemBytes = props.limits.maxComputeSharedMemorySize;
}

static void NormalizeProfiles(OaVec<OaDeviceNode>& InOutNodes, OaU32 InPrimaryIdx) {
	if (InOutNodes.Empty()) return;
	auto& primary = InOutNodes[InPrimaryIdx].Profile;
	OaF64 pTflops = primary.PeakTflopsF32;
	OaF64 pBw = primary.MemBandwidthGBs;
	if (pTflops <= 0.0) pTflops = 1.0;
	if (pBw <= 0.0) pBw = 1.0;

	for (auto& node : InOutNodes) {
		node.Profile.MatmulScore = static_cast<OaF32>(node.Profile.PeakTflopsF32 / pTflops);
		node.Profile.ReductionScore = static_cast<OaF32>(node.Profile.MemBandwidthGBs / pBw);
		node.Profile.TransferScore = node.Profile.ReductionScore;
	}
}

// ─── Topology Detection ────────────────────────────────────────────────────
static OaInterconnect ClassifyInterconnect(const OaVkDevice& InA, const OaVkDevice& InB) {
	bool aIntegrated = (InA.Info.Hardware.DeviceType == OaDeviceType::VkIntegrated);
	bool bIntegrated = (InB.Info.Hardware.DeviceType == OaDeviceType::VkIntegrated);
	bool aDiscreteLike =
		InA.Info.Hardware.DeviceType == OaDeviceType::VkDiscrete ||
		InA.Info.Hardware.DeviceType == OaDeviceType::VkVirtualGpu;
	bool bDiscreteLike =
		InB.Info.Hardware.DeviceType == OaDeviceType::VkDiscrete ||
		InB.Info.Hardware.DeviceType == OaDeviceType::VkVirtualGpu;

	// iGPU + dGPU (or virtual passthrough GPU) on same machine share system memory
	if ((aIntegrated && bDiscreteLike) || (aDiscreteLike && bIntegrated)) {
		return OaInterconnect::SharedRam;
	}

	// Same vendor discrete / virtual GPUs — might have high-bandwidth link
	if (aDiscreteLike && bDiscreteLike && InA.Info.Hardware.VendorId == InB.Info.Hardware.VendorId) {
		if (InA.Info.Hardware.VendorId == 0x10DE) return OaInterconnect::PciE;  // NVLink detection requires NVML
		if (InA.Info.Hardware.VendorId == 0x1002) return OaInterconnect::PciE;  // xGMI detection requires rocm-smi
	}

	return OaInterconnect::HostMemory;
}

static OaF64 EstimateLinkBandwidth(OaInterconnect InType, const OaVkDevice& InA, const OaVkDevice& InB) {
	switch (InType) {
		case OaInterconnect::SharedRam:
			return 50.0;  // DDR5 dual-channel typical
		case OaInterconnect::PciE:
			// SAM/ReBAR on both endpoints → lower latency, estimate higher effective BW
			if (InA.Info.Hardware.HasSAM && InB.Info.Hardware.HasSAM) return 28.0;
			return 25.0;  // PCIe 4.0 x16 typical
		case OaInterconnect::NvLink:
			return 300.0;  // NVLink 3.0 typical
		case OaInterconnect::Xgmi:
			return 200.0;  // xGMI typical
		case OaInterconnect::HostMemory:
		default:
			return 12.0;  // Host staging limited by map/unmap overhead
	}
}

static void BuildTransferLinks(OaDeviceMesh& InOutMesh) {
	OaU32 n = InOutMesh.NodeCount();
	for (OaU32 i = 0; i < n; ++i) {
		for (OaU32 j = i + 1; j < n; ++j) {
			auto& a = InOutMesh.Nodes[i];
			auto& b = InOutMesh.Nodes[j];

			OaInterconnect topo = ClassifyInterconnect(a.Device, b.Device);
			OaF64 bw = EstimateLinkBandwidth(topo, a.Device, b.Device);
			bool shared = (topo == OaInterconnect::SharedRam);
			const bool aDiscreteLike = a.Device.Info.Hardware.DeviceType == OaDeviceType::VkDiscrete ||
				a.Device.Info.Hardware.DeviceType == OaDeviceType::VkVirtualGpu;
			const bool bDiscreteLike = b.Device.Info.Hardware.DeviceType == OaDeviceType::VkDiscrete ||
				b.Device.Info.Hardware.DeviceType == OaDeviceType::VkVirtualGpu;
			bool p2p = (a.Device.Info.Hardware.VendorId == b.Device.Info.Hardware.VendorId) &&
				aDiscreteLike && bDiscreteLike;

			#ifdef OA_ANDROID_ML
			bool dmaBuf = false;
			#else
			bool dmaBuf = OaCanUseDmaBuf(a.Device, b.Device);
			#endif

			OaTransferLink link;
			link.SrcNode = i;
			link.DstNode = j;
			link.BandwidthGBs = bw;
			link.SharedMemory = shared;
			link.PeerToPeer = p2p;
			link.Topology = topo;
			link.DmaBufCapable = dmaBuf;
			link.BestTransport = dmaBuf ? OaTransport::DmaBuf : OaTransport::HostStaging;
			InOutMesh.Links.PushBack(link);
		}
	}
}

// ─── OaDeviceNode ──────────────────────────────────────────────────────────

void OaDeviceNode::Destroy() {
	for (auto& s : StreamPool) s->Destroy(Device);
	StreamPool.Clear();
	FreeStack.Clear();
	Bindless.Destroy(Device);
	Pipelines.Destroy(Device);
	Allocator.Destroy();
	Device.Destroy();
}

OaDeviceNode::OaDeviceNode() : StreamLock(OaMakeUniquePtr<OaSpinlock>()) {}

OaVkStream* OaDeviceNode::AcquireStream() {
	OaSpinlockGuard guard(*StreamLock);
	OaVkStream* raw = nullptr;
	if (!FreeStack.Empty()) {
		OaU32 idx = FreeStack.Back();
		FreeStack.PopBack();
		raw = StreamPool[idx].get();
	} else {
		auto res = OaVkStream::CreateCompute(Device);
		if (!res) return nullptr;
		auto ptr = OaMakeUniquePtr<OaVkStream>(std::move(*res));
		raw = ptr.get();
		StreamPool.PushBack(std::move(ptr));
	}
	raw->MeshNodeIndex = Index;
	return raw;
}

void OaDeviceNode::ReleaseStream(OaVkStream* InStream) {
	OaSpinlockGuard guard(*StreamLock);
	for (OaU32 i = 0; i < StreamPool.Size(); ++i) {
		if (StreamPool[i].get() == InStream) {
			FreeStack.PushBack(i);
			return;
		}
	}
}

// ─── OaDeviceMesh ──────────────────────────────────────────────────────────

OaDeviceMesh::OaDeviceMesh() : DeviceLoadLock(OaMakeUniquePtr<OaSpinlock>()) {}

OaResult<OaDeviceMesh> OaDeviceMesh::Create(const OaEngineConfig& InConfig) {
	OaVec<const char*> instanceExtraPtrs;
	for (const auto& ext : InConfig.InstanceExtraExtensions) {
		if (!ext.Empty()) {
			instanceExtraPtrs.PushBack(ext.CStr());
		}
	}
	OaSpan<const char* const> extraSpan(instanceExtraPtrs.Data(), instanceExtraPtrs.Size());

	auto instResult = OaVkInstance::CreateInstance(
		OaStringView(InConfig.AppName),
		InConfig.AppVersion,
		InConfig.EnableValidation,
		extraSpan);
	if (!instResult.IsOk()) {
		return OaResult<OaDeviceMesh>(instResult.GetStatus());
	}
	VkInstance instance = instResult.GetValue();
	OaVkLoadInstance(instance);

	OaU32 devCount = 0;
	vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
	if (devCount == 0) {
		OaVkInstance::DestroyInstance(instance);
		return OaStatus::Error(OaStatusCode::DeviceNotFound, "no Vulkan physical devices");
	}

	OaVec<VkPhysicalDevice> physDevices(devCount);
	vkEnumeratePhysicalDevices(instance, &devCount, physDevices.Data());

	// Score all devices — primary is the one with highest score for preferred type
	OaDeviceType pref = OaDeviceType::VkDiscrete;
	switch (InConfig.DevicePref) {
		case OaDevicePreference::Integrated: 
			pref = OaDeviceType::VkIntegrated;
			break;
		case OaDevicePreference::Discrete:
			pref = OaDeviceType::VkDiscrete;
			break;
		case OaDevicePreference::Cpu:
			pref = OaDeviceType::VkCpu;
			break;
		default:
			break;
	}

	OaVkLogPhysicalDeviceSurvey(
		devCount,
		reinterpret_cast<void* const*>(physDevices.Data()),
		pref);

	struct ScoredDevice {
		VkPhysicalDevice PhysDev;
		VkPhysicalDeviceProperties Props;
		OaU64 Score;
		OaU32 EnumIndex = 0;
	};
	OaVec<ScoredDevice> scored;
	OaU32 maxDevices = std::min(devCount, InConfig.MaxDevices);

	if (!InConfig.MeshVulkanIndices.Empty()) {
		OaU32 listed = static_cast<OaU32>(InConfig.MeshVulkanIndices.Size());
		maxDevices = std::min(maxDevices, listed);
		for (OaU32 k = 0; k < maxDevices; ++k) {
			OaU32 ei = InConfig.MeshVulkanIndices[k];
			if (ei >= devCount) {
				OaVkInstance::DestroyInstance(instance);
				return OaStatus::Error(OaStatusCode::DeviceNotFound,
					"mesh_indices vulkan index out of range (see survey log)");
			}
			ScoredDevice sd;
			sd.PhysDev = physDevices[ei];
			sd.EnumIndex = ei;
			vkGetPhysicalDeviceProperties(sd.PhysDev, &sd.Props);
			sd.Score = OaVkPhysicalDeviceRate(sd.PhysDev, pref);
			scored.PushBack(sd);
		}
		OA_LOG_INFO(OaLogComponent::Core,
			"Mesh: explicit Vulkan index list (%u device(s), primary = first listed)",
			static_cast<unsigned>(scored.Size()));
		for (OaU32 j = 0; j < scored.Size(); ++j) {
			OA_LOG_INFO(OaLogComponent::Core,
				"  mesh[%u] <- Vulkan[%u] %s | PickScore=%s",
				static_cast<unsigned>(j),
				static_cast<unsigned>(scored[j].EnumIndex),
				scored[j].Props.deviceName,
				OaFormatNumberU64(scored[j].Score).c_str()
			);
		}
	} else {
		maxDevices = std::min(devCount, InConfig.MaxDevices);
		for (OaU32 i = 0; i < devCount; ++i) {
			ScoredDevice sd;
			sd.PhysDev = physDevices[i];
			sd.EnumIndex = i;
			vkGetPhysicalDeviceProperties(sd.PhysDev, &sd.Props);
			sd.Score = OaVkPhysicalDeviceRate(sd.PhysDev, pref);
			scored.PushBack(sd);
		}

		// Sort descending by score (primary first)
		std::sort(scored.Begin(), scored.End(),
			[](const ScoredDevice& a, const ScoredDevice& b) { return a.Score > b.Score; });

		OA_LOG_INFO(OaLogComponent::Core,
			"Mesh: attaching to top %u device(s) by PickScore (primary = highest)",
			static_cast<unsigned>(maxDevices));
		for (OaU32 j = 0; j < maxDevices && j < scored.Size(); ++j) {
			OA_LOG_INFO(OaLogComponent::Core,
				"  mesh[%u] <- Vulkan[%u] %s | PickScore=%s",
				static_cast<unsigned>(j),
				static_cast<unsigned>(scored[j].EnumIndex),
				scored[j].Props.deviceName,
				OaFormatNumberU64(scored[j].Score).c_str());
		}
	}

	OaDeviceMesh mesh;
	OaString cacheDir = InConfig.EnablePipelineCache ? OaString(InConfig.PipelineCacheDir) : OaString{};

	for (OaU32 i = 0; i < maxDevices; ++i) {
		auto devResult = OaVkDevice::CreateFromPhysical(
			instance,
			scored[i].PhysDev,
			InConfig.EnableValidation,
			scored[i].Score,
			scored[i].EnumIndex);
		if (!devResult.IsOk()) {
			OA_LOG_WARN(OaLogComponent::Core,
				"Skipping device %s — %s",
				scored[i].Props.deviceName,
				devResult.GetStatus().GetMessage().c_str());
			continue;
		}

		// Reserve space in vector to avoid reallocation (OaDeviceNode is not movable)
		if (mesh.Nodes.Empty()) {
			mesh.Nodes.Reserve(InConfig.MaxDevices);
		}

		// Construct node in-place since OaDeviceNode is not movable
		mesh.Nodes.EmplaceBack();
		OaDeviceNode& node = mesh.Nodes.Back();
		
		node.Device = std::move(devResult.GetValue());
		node.Index = static_cast<OaU32>(mesh.Nodes.Size() - 1);
		node.Device.SetMeshLogicalIndex(static_cast<OaI32>(node.Index));

		// First node owns the VkInstance. Per-device setup must use that VkDevice's dispatch:
		// OaVk* globals come from OaVkLoadDevice(); only loading the first GPU breaks aux nodes.
		if (node.Index == 0) {
			node.Device.OwnsInstance = true;
		}
		OaVkLoadDevice(static_cast<VkDevice>(node.Device.Device));

		// Create per-device allocator
		auto allocResult = OaVma::Create(node.Device);
		if (!allocResult.IsOk()) {
			OA_LOG_WARN(OaLogComponent::Core,
				"Skipping device %s — allocator failed: %s",
				node.Device.Info.Hardware.DeviceName.c_str(),
				allocResult.GetStatus().GetMessage().c_str());
			node.Device.Destroy();
			mesh.Nodes.PopBack();  // Remove the failed node
			continue;
		}
		node.Allocator = std::move(allocResult.GetValue());

		// Create per-device bindless heap
		auto bindlessResult = OaBindlessHeap::Create(node.Device);
		if (bindlessResult.IsOk()) {
			node.Bindless = std::move(bindlessResult.GetValue());
		} else {
			OA_LOG_WARN(OaLogComponent::Core,
				"Device %s: bindless heap failed — per-dispatch descriptors only",
				node.Device.Info.Hardware.DeviceName.c_str());
		}

		// Pipeline cache blobs are not portable across different physical devices (vendor / ISA).
		// Sharing one file on a mesh (e.g. dGPU cache fed into iGPU vkCreatePipelineCache) can fault the driver.
		OaString nodeCacheDir = cacheDir;
		if (!cacheDir.empty()) {
			nodeCacheDir += "/vk_";
			nodeCacheDir += OaToString(scored[i].EnumIndex);
		}

		// Create per-device pipeline registry
		auto pipeStatus = node.Pipelines.Init(
			node.Device, nodeCacheDir, node.Bindless.PipelineLayout);
		if (!pipeStatus.IsOk()) {
			OA_LOG_WARN(OaLogComponent::Core,
				"Skipping device %s — pipeline init failed: %s",
				node.Device.Info.Hardware.DeviceName.c_str(),
				pipeStatus.GetMessage().c_str());
			node.Bindless.Destroy(node.Device);
			node.Allocator.Destroy();
			node.Device.Destroy();
			mesh.Nodes.PopBack();  // Remove the failed node
			continue;
		}

		// Profile the device
		ProfileDevice(node);

		// Assign role: first (highest score) = Primary, rest = Auxiliary
		node.Role = (node.Index == 0) ? OaDeviceRole::Primary : OaDeviceRole::Auxiliary;
	}

	if (mesh.Nodes.Empty()) {
		OaVkInstance::DestroyInstance(instance);
		return OaStatus::Error(OaStatusCode::DeviceNotFound,
			"no usable Vulkan devices for mesh");
	}

	mesh.Primary = &mesh.Nodes[0];

	OA_LOG_INFO(OaLogComponent::Core,
		"Mesh: primary %s (Vulkan index %u, PickScore %s)",
		mesh.Primary->Device.Info.Hardware.DeviceName.c_str(),
		static_cast<unsigned>(mesh.Primary->Device.Info.Hardware.EnumerationIndex),
		OaFormatNumberU64(mesh.Primary->Device.Info.Hardware.PickRating).c_str());

	NormalizeProfiles(mesh.Nodes, 0);

	// Build transfer links between all device pairs
	BuildTransferLinks(mesh);

	mesh.PrintTopology();

	// Engine and single-device code paths use global OaVk* symbols — pin them to primary.
	OaVkLoadDevice(static_cast<VkDevice>(mesh.Primary->Device.Device));

	// OaDeviceMesh is now movable - return by move
	return std::move(mesh);
}

OaStatus OaDeviceMesh::EnsureScratch(OaU64 InSize) {
	if (ScratchSize >= InSize) return OaStatus::Ok();
	if (!Primary) return OaStatus::Error(OaStatusCode::InvalidArgument, "no primary device");

	if (ScratchBuf.Buffer) {
		Primary->Allocator.Free(ScratchBuf);
		ScratchBuf = {};
		ScratchSize = 0;
	}

	auto result = Primary->Allocator.AllocHostVisible(InSize);
	if (!result.IsOk()) return result.GetStatus();
	ScratchBuf = std::move(result.GetValue());
	ScratchSize = InSize;
	return OaStatus::Ok();
}

void OaDeviceMesh::Destroy() {
	if (ScratchBuf.Buffer && Primary) {
		OaSpinlockGuard scratchGuard(*DeviceLoadLock);
		OaVkLoadDevice(static_cast<VkDevice>(Primary->Device.Device));
		Primary->Allocator.Free(ScratchBuf);
		ScratchBuf = {};
		ScratchSize = 0;
	}

	// Defer OaVkInstance::DestroyInstance until after every VkDevice from that instance is destroyed.
	VkInstance instanceToDestroy = VK_NULL_HANDLE;
	for (auto& node : Nodes) {
		if (node.Device.OwnsInstance) {
			instanceToDestroy = static_cast<VkInstance>(node.Device.Instance);
			node.Device.OwnsInstance = false;
			break;
		}
	}

	for (auto& node : Nodes) {
		OaSpinlockGuard guard(*DeviceLoadLock);
		if (node.Device.Device) {
			OaVkLoadDevice(static_cast<VkDevice>(node.Device.Device));
		}
		node.Destroy();
	}
	Nodes.Clear();
	Links.Clear();
	Primary = nullptr;

	if (instanceToDestroy != VK_NULL_HANDLE) {
		OaVkInstance::DestroyInstance(instanceToDestroy);
	}
}

OaDeviceNode* OaDeviceMesh::GetNode(OaU32 InIndex) {
	if (InIndex < Nodes.Size()) return &Nodes[InIndex];
	return nullptr;
}

const OaDeviceNode* OaDeviceMesh::GetNode(OaU32 InIndex) const {
	if (InIndex < Nodes.Size()) return &Nodes[InIndex];
	return nullptr;
}

OaDeviceNode* OaDeviceMesh::GetByRole(OaDeviceRole InRole) {
	for (auto& node : Nodes) {
		if (node.Role == InRole) return &node;
	}
	return nullptr;
}

OaStatus OaDeviceMesh::CopyBuffer(
	OaU32 InSrcNode, const OaVkBuffer& InSrc,
	OaU32 InDstNode, OaVkBuffer& InDst,
	OaU64 InSize
) {
	(void)InSrcNode;
	(void)InDstNode;
	if (!InSrc.MappedPtr || !InDst.MappedPtr) {
		return OaStatus::Error(
			OaStatusCode::InvalidArgument,
			"cross-device copy requires host-visible buffers (MappedPtr must be valid)"
		);
	}
	OaMemcpy(InDst.MappedPtr, InSrc.MappedPtr, InSize);
	return OaStatus::Ok();
}

const OaTransferLink* OaDeviceMesh::FindLink(OaU32 InSrc, OaU32 InDst) const {
	for (const auto& link : Links) {
		if (link.SrcNode == InSrc && link.DstNode == InDst) {
			return &link;
		}
		if (link.SrcNode == InDst && link.DstNode == InSrc) {
			return &link;
		}
	}
	return nullptr;
}

OaVec<OaVec<OaF64>> OaDeviceMesh::BandwidthMatrix() const {
	OaU32 n = NodeCount();
	OaVec<OaVec<OaF64>> mat(n, OaVec<OaF64>(n, 0.0));

	for (OaU32 i = 0; i < n; ++i) {
		mat[i][i] = Nodes[i].Profile.MemBandwidthGBs;
	}

	for (const auto& link : Links) {
		if (link.SrcNode < n && link.DstNode < n) {
			mat[link.SrcNode][link.DstNode] = link.BandwidthGBs;
			mat[link.DstNode][link.SrcNode] = link.BandwidthGBs;
		}
	}

	return mat;
}

void OaDeviceMesh::PrintTopology() const {
	OA_LOG_INFO(OaLogComponent::Core, "OaDeviceMesh: %u node(s), %zu link(s)", NodeCount(), Links.Size());
	for (const auto& node : Nodes) {
		OA_LOG_INFO(OaLogComponent::Core, "  [%u] %s (%.*s) — %.*s — VRAM %.1f GB — %.1f TFLOPS",
			node.Index,
			node.Device.Info.Hardware.DeviceName.c_str(),
			static_cast<int>(OaDeviceRoleName(node.Role).size()), OaDeviceRoleName(node.Role).data(),
			static_cast<int>(OaDeviceTypeName(node.Device.Info.Hardware.DeviceType).size()), OaDeviceTypeName(node.Device.Info.Hardware.DeviceType).data(),
			static_cast<OaF64>(node.Profile.VramBytes) / (1024.0 * 1024.0 * 1024.0),
			node.Profile.PeakTflopsF32
		);
	}
	for (const auto& link : Links) {
		OA_LOG_INFO(OaLogComponent::Core, "  %u <-> %u via %.*s (%.1f GB/s) transport=%.*s%s%s%s",
			link.SrcNode, link.DstNode,
			static_cast<int>(OaInterconnectName(link.Topology).size()), OaInterconnectName(link.Topology).data(),
			link.BandwidthGBs,
			static_cast<int>(OaTransportName(link.BestTransport).size()), OaTransportName(link.BestTransport).data(),
			link.PeerToPeer ? " [P2P]" : "",
			link.SharedMemory ? " [shared]" : "",
			link.DmaBufCapable ? " [DMA-BUF]" : ""
		);
	}
}
