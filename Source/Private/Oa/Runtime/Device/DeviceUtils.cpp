#include <Oa/Runtime/Device.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Runtime/Init.h>     // OaVkVendorId*, OaCoopMatTrust forward decl
#include <Oa/Core/EnvFlag.h>

#include <algorithm>
#include <cstring>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif


OaDeviceType OaVkMapPhysicalType(VkPhysicalDeviceType InType) {
	switch (InType) {
		case VK_PHYSICAL_DEVICE_TYPE_OTHER:
			return OaDeviceType::VkOther;
		case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
			return OaDeviceType::VkIntegrated;
		case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
			return OaDeviceType::VkDiscrete;
		case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
			return OaDeviceType::VkVirtualGpu;
		case VK_PHYSICAL_DEVICE_TYPE_CPU:
			return OaDeviceType::VkCpu;
		default:
			return OaDeviceType::VkOther;
	}
}


const char* OaVkPhysicalTypeLabel(VkPhysicalDeviceType InType) {
	switch (InType) {
		case VK_PHYSICAL_DEVICE_TYPE_OTHER:
			return "Other";
		case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
			return "Integrated GPU";
		case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
			return "Discrete GPU";
		case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
			return "Virtual GPU";
		case VK_PHYSICAL_DEVICE_TYPE_CPU:
			return "CPU";
		default:
			return "Other";
	}
}


const char* OaVkVendorLabel(OaU32 InVendorId) {
	switch (InVendorId) {
		case 0x1002: return "AMD";
		case 0x10DE: return "NVIDIA";
		case 0x8086: return "Intel";
		case 0x13B5: return "ARM";
		case 0x5143: return "Qualcomm";
		case 0x1010: return "ImgTec";
		case 0x1AE0: return "Google";
		case 0x106B: return "Apple";
		default: return "Unknown";
	}
}


OaString OaVkFormatDriverVersion(OaU32 InVersion) {
	return OaString(
		OaToString(static_cast<OaU32>(VK_VERSION_MAJOR(InVersion)))
		+ "." + OaToString(static_cast<OaU32>(VK_VERSION_MINOR(InVersion)))
		+ "." + OaToString(static_cast<OaU32>(VK_VERSION_PATCH(InVersion)))
	);
}


const char* OaVkDriverIdLabel(OaU32 InDriverId) {
	switch (InDriverId) {
		case VK_DRIVER_ID_AMD_PROPRIETARY: {}
			return "AMD_PROPRIETARY";
		case VK_DRIVER_ID_AMD_OPEN_SOURCE:
			return "AMD_OPEN_SOURCE";
		case VK_DRIVER_ID_MESA_RADV:
			return "MESA_RADV";
		case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
			return "NVIDIA_PROPRIETARY";
		case VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS:
			return "INTEL_PROPRIETARY_WINDOWS";
		case VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA:
			return "INTEL_OPEN_SOURCE_MESA";
		case VK_DRIVER_ID_IMAGINATION_PROPRIETARY:
			return "IMAGINATION_PROPRIETARY";
		case VK_DRIVER_ID_QUALCOMM_PROPRIETARY:
			return "QUALCOMM_PROPRIETARY";
		case VK_DRIVER_ID_ARM_PROPRIETARY:
			return "ARM_PROPRIETARY";
		case VK_DRIVER_ID_GOOGLE_SWIFTSHADER:
			return "GOOGLE_SWIFTSHADER";
		case VK_DRIVER_ID_GGP_PROPRIETARY:
			return "GGP_PROPRIETARY";
		case VK_DRIVER_ID_BROADCOM_PROPRIETARY:
			return "BROADCOM_PROPRIETARY";
		case VK_DRIVER_ID_MESA_LLVMPIPE:
			return "MESA_LLVMPIPE";
		case VK_DRIVER_ID_MOLTENVK:
			return "MOLTENVK";
		case VK_DRIVER_ID_COREAVI_PROPRIETARY:
			return "COREAVI_PROPRIETARY";
		case VK_DRIVER_ID_JUICE_PROPRIETARY:
			return "JUICE_PROPRIETARY";
		case VK_DRIVER_ID_VERISILICON_PROPRIETARY:
			return "VERISILICON_PROPRIETARY";
		case VK_DRIVER_ID_MESA_TURNIP:
			return "MESA_TURNIP";
		case VK_DRIVER_ID_MESA_V3DV:
			return "MESA_V3DV";
		case VK_DRIVER_ID_MESA_PANVK:
			return "MESA_PANVK";
		case VK_DRIVER_ID_SAMSUNG_PROPRIETARY:
			return "SAMSUNG_PROPRIETARY";
		case VK_DRIVER_ID_MESA_VENUS:
			return "MESA_VENUS";
		case VK_DRIVER_ID_MESA_DOZEN:
			return "MESA_DOZEN";
		case VK_DRIVER_ID_MESA_NVK:
			return "MESA_NVK";
		case VK_DRIVER_ID_IMAGINATION_OPEN_SOURCE_MESA:
			return "IMAGINATION_OPEN_SOURCE_MESA";
		case VK_DRIVER_ID_MESA_HONEYKRISP:
			return "MESA_HONEYKRISP";
		case VK_DRIVER_ID_VULKAN_SC_EMULATION_ON_VULKAN:
			return "VULKAN_SC_EMULATION_ON_VULKAN";
		case VK_DRIVER_ID_MESA_KOSMICKRISP:
			return "MESA_KOSMICKRISP";
		default:
			return "UNKNOWN";
	}
}


static OaF64 OaVkEstimateMemBandwidthGbpsHeuristic(
	OaU32 InVendorId,
	OaDeviceType InType,
	OaU64 InLocalHeapBytes)
{
	OaF64 vramGb = static_cast<OaF64>(InLocalHeapBytes) / (1024.0 * 1024.0 * 1024.0);

	if (InType == OaDeviceType::VkCpu) {
		return 20.0;
	}

	if (InType == OaDeviceType::VkIntegrated) {
		if (InVendorId == 0x106B) {
			return 130.0;
		}
		if (InVendorId == 0x1002) {
			if (vramGb >= 48.0) {
				return 200.0;
			}
			if (vramGb >= 32.0) {
				return 180.0;
			}
			if (vramGb >= 16.0) {
				return 120.0;
			}
			return 70.0;
		}
		if (InVendorId == 0x8086) {
			return 55.0;
		}
		return 50.0;
	}

	switch (InVendorId) {
		case 0x10DE:
			if (vramGb >= 80.0) {
				return 4500.0;
			}
			if (vramGb >= 48.0) {
				return 2000.0;
			}
			if (vramGb >= 24.0) {
				return 1200.0;
			}
			if (vramGb >= 16.0) {
				return 700.0;
			}
			if (vramGb >= 10.0) {
				return 500.0;
			}
			return 300.0;
		case 0x1002:
			if (vramGb >= 96.0) {
				return 3600.0;
			}
			if (vramGb >= 64.0) {
				return 1700.0;
			}
			if (vramGb >= 24.0) {
				return 900.0;
			}
			if (vramGb >= 16.0) {
				return 600.0;
			}
			return 400.0;
		case 0x8086:
			if (vramGb >= 48.0) {
				return 800.0;
			}
			return 500.0;
		default:
			if (vramGb >= 48.0) {
				return 1500.0;
			}
			return 200.0;
	}
}


static OaF64 OaVkEstimatePeakTflopsF32Heuristic(
	OaU32 InVendorId,
	OaDeviceType InType,
	OaU64 InLocalHeapBytes)
{
	OaF64 vramGb =
		static_cast<OaF64>(InLocalHeapBytes) / (1024.0 * 1024.0 * 1024.0);

	if (InType == OaDeviceType::VkCpu) {
		return 0.1;
	}

	if (InType == OaDeviceType::VkIntegrated) {
		if (InVendorId == 0x106B) {
			return vramGb >= 64.0 ? 18.0 : (vramGb >= 32.0 ? 12.0 : 6.0);
		}
		if (InVendorId == 0x1002) {
			if (vramGb >= 64.0) {
				return 35.0;
			}
			if (vramGb >= 32.0) {
				return 25.0;
			}
			if (vramGb >= 16.0) {
				return 15.0;
			}
			return 4.0;
		}
		if (InVendorId == 0x8086) {
			return 2.5;
		}
		return 1.5;
	}

	switch (InVendorId) {
		case 0x10DE:
			if (vramGb >= 80.0) {
				return 9000.0;
			}
			if (vramGb >= 48.0) {
				return 120.0;
			}
			if (vramGb >= 20.0) {
				return 90.0;
			}
			if (vramGb >= 14.0) {
				return 60.0;
			}
			if (vramGb >= 10.0) {
				return 35.0;
			}
			if (vramGb >= 6.0) {
				return 15.0;
			}
			return 8.0;
		case 0x1002:
			if (vramGb >= 96.0) {
				return 4000.0;
			}
			if (vramGb >= 64.0) {
				return 200.0;
			}
			if (vramGb >= 20.0) {
				return 60.0;
			}
			if (vramGb >= 14.0) {
				return 45.0;
			}
			if (vramGb >= 10.0) {
				return 25.0;
			}
			return 10.0;
		case 0x8086:
			if (vramGb >= 14.0) {
				return 20.0;
			}
			return 10.0;
		default:
			if (vramGb >= 48.0) {
				return 150.0;
			}
			return 5.0;
	}
}


// Curated PCI vendor:device → mem BW / FP32 peaks (datasheets, vulkan.gpuinfo.org).
// Sorted by (VendorId, DeviceId) for binary search. Extend when pick_rating skews on new silicon.

struct OaVkGpuSpecEntry {
	OaU32 VendorId = 0;
	OaU32 DeviceId = 0;
	OaF32 MemBandwidthGbps = 0.0f;
	OaF32 PeakFp32Tflops = 0.0f;
	OaU32 NumSMs = 0;  // Streaming multiprocessors (NVIDIA) / Compute units (AMD)
};


static OaU64 OaVkGpuSpecSortKey(OaU32 InVendorId, OaU32 InDeviceId) {
	return (static_cast<OaU64>(InVendorId) << 32) | static_cast<OaU64>(InDeviceId);
}


static const OaVkGpuSpecEntry OaVkGpuSpecTable[] = {
	{0x1002, 0x744C, 960.0f, 61.0f, 96},   // RX 7900 XTX (96 CUs)
	{0x1002, 0x7480, 864.0f, 52.0f, 84},   // RX 7900 XT (84 CUs, approx.)
	{0x8086, 0x56A0, 560.0f, 17.0f, 32},   // Arc A770 16 GB (32 Xe-cores, approx.)
	{0x8086, 0x56A1, 512.0f, 14.0f, 28},   // Arc A750 (28 Xe-cores, approx.)
	{0x10DE, 0x2684, 1008.0f, 82.6f, 128}, // GeForce RTX 4090 (128 SMs, AD102)
	{0x10DE, 0x2704, 1008.0f, 82.6f, 128}, // RTX 4090 D / close AD102 variants
	{0x10DE, 0x2805, 720.0f, 42.0f, 58},   // RTX 4080 Laptop (58 SMs, approx.)
	{0x10DE, 0x2820, 504.0f, 32.5f, 46},   // RTX 4080 Laptop (alternate ID, 46 SMs approx.)
	{0x10DE, 0x2B85, 896.0f, 32.0f, 170},  // RTX 5090 Laptop (GB202, 170 SMs — Blackwell)
	{0x10DE, 0x2B8C, 896.0f, 32.0f, 170},  // RTX 5090 Laptop (alternate PCI ID)
	{0x10DE, 0x2C18, 896.0f, 32.0f, 170},  // RTX 5090 Mobile / Max-Q (170 SMs)
	{0x10DE, 0x2C58, 896.0f, 32.0f, 170},  // RTX 5090 Mobile / Max-Q (variant)
};


static bool OaVkGpuSpecLookup(OaU32 InVendorId, OaU32 InDeviceId, OaVkGpuSpecEntry& OutEntry) {
	const OaU64 key = OaVkGpuSpecSortKey(InVendorId, InDeviceId);
	OaU32 lo = 0;
	OaU32 hi = static_cast<OaU32>(
		sizeof(OaVkGpuSpecTable) / sizeof(OaVkGpuSpecTable[0]));
	while (lo < hi) {
		const OaU32 mid = (lo + hi) >> 1;
		const OaVkGpuSpecEntry& row = OaVkGpuSpecTable[mid];
		const OaU64 midKey = OaVkGpuSpecSortKey(row.VendorId, row.DeviceId);
		if (midKey < key) {
			lo = mid + 1;
		} else if (midKey > key) {
			hi = mid;
		} else {
			OutEntry = row;
			return true;
		}
	}
	return false;
}


OaF64 OaVkEstimateMemBandwidthGbpsEx(
	OaU32 InVendorId,
	OaU32 InDeviceId,
	OaDeviceType InType,
	OaU64 InLocalHeapBytes)
{
	if (InDeviceId != OaVkPciDeviceIdUnknown) {
		OaVkGpuSpecEntry row{};
		if (OaVkGpuSpecLookup(InVendorId, InDeviceId, row)) {
			return static_cast<OaF64>(row.MemBandwidthGbps);
		}
	}
	return OaVkEstimateMemBandwidthGbpsHeuristic(
		InVendorId, InType, InLocalHeapBytes);
}


OaF64 OaVkEstimatePeakTflopsF32Ex(
	OaU32 InVendorId,
	OaU32 InDeviceId,
	OaDeviceType InType,
	OaU64 InLocalHeapBytes)
{
	if (InDeviceId != OaVkPciDeviceIdUnknown) {
		OaVkGpuSpecEntry row{};
		if (OaVkGpuSpecLookup(InVendorId, InDeviceId, row)) {
			return static_cast<OaF64>(row.PeakFp32Tflops);
		}
	}
	return OaVkEstimatePeakTflopsF32Heuristic(
		InVendorId, InType, InLocalHeapBytes);
}


OaU32 OaVkEstimateNumSMsEx(
	OaU32 InVendorId,
	OaU32 InDeviceId,
	OaDeviceType InType,
	OaU64 InLocalHeapBytes)
{
	// Try exact lookup first
	if (InDeviceId != OaVkPciDeviceIdUnknown) {
		OaVkGpuSpecEntry row{};
		if (OaVkGpuSpecLookup(InVendorId, InDeviceId, row)) {
			if (row.NumSMs > 0) {
				return row.NumSMs;
			}
		}
	}

	// Fallback heuristic based on VRAM and vendor
	const OaU64 vramGb = InLocalHeapBytes / (1024 * 1024 * 1024);
	
	if (InVendorId == 0x10DE) {  // NVIDIA
		// Rough heuristic: ~8-12 SMs per GB for modern GPUs
		if (InType == OaDeviceType::VkDiscrete) {
			if (vramGb >= 20) return 128;  // High-end (4090, 5090 class)
			if (vramGb >= 12) return 80;   // Mid-high (4080 class)
			if (vramGb >= 8) return 60;    // Mid (4070 class)
			return 46;                     // Entry (4060 class)
		}
		return 32;  // Integrated/mobile fallback
	}
	
	if (InVendorId == 0x1002) {  // AMD
		// CUs: ~60-96 for RDNA3
		if (InType == OaDeviceType::VkDiscrete) {
			if (vramGb >= 20) return 96;   // 7900 XTX class
			if (vramGb >= 16) return 84;   // 7900 XT class
			if (vramGb >= 12) return 60;   // 7800 XT class
			return 48;                     // 7700 XT class
		}
		return 12;  // Integrated fallback
	}
	
	if (InVendorId == 0x8086) {  // Intel
		// Xe-cores: ~16-32 for Arc
		if (InType == OaDeviceType::VkDiscrete) {
			if (vramGb >= 12) return 32;   // A770 class
			return 28;                     // A750 class
		}
		return 16;  // Integrated fallback
	}
	
	// Unknown vendor: conservative estimate
	return 32;
}


OaF64 OaVkEstimateMemBandwidthGbps(
	OaU32 InVendorId,
	OaDeviceType InType,
	OaU64 InLocalHeapBytes)
{
	return OaVkEstimateMemBandwidthGbpsEx(
		InVendorId, OaVkPciDeviceIdUnknown, InType, InLocalHeapBytes);
}


OaF64 OaVkEstimatePeakTflopsF32(
	OaU32 InVendorId,
	OaDeviceType InType,
	OaU64 InLocalHeapBytes)
{
	return OaVkEstimatePeakTflopsF32Ex(
		InVendorId, OaVkPciDeviceIdUnknown, InType, InLocalHeapBytes);
}


OaU64 OaVkPhysicalDevicePickScore(
	OaDeviceType InType,
	OaU64 InLocalHeapBytes,
	OaDeviceType InPreferred,
	OaU32 InDeviceId)
{
	OaU32 tierRank = 1;
	if (InType == InPreferred) {
		tierRank = 4;
	} else if (InType == OaDeviceType::VkDiscrete || InType == OaDeviceType::VkVirtualGpu) {
		tierRank = 3;
	} else if (InType == OaDeviceType::VkIntegrated || InType == OaDeviceType::VkOther) {
		tierRank = 2;
	}
	const OaU64 tierPart = static_cast<OaU64>(tierRank) << 60;
	const OaU64 heapMb = InLocalHeapBytes / (1024 * 1024);
	const OaU64 heapPart = std::min(heapMb, (OaU64{1} << 20) - OaU64{1}) << 12;
	const OaU64 idPart = static_cast<OaU64>(InDeviceId) & 0xFFFu;
	return tierPart + heapPart + idPart;
}

static bool OaVkPhysHasExtensionImpl(VkPhysicalDevice InPhys, const char* InName) {
	OaU32 extCount = 0;
	vkEnumerateDeviceExtensionProperties(InPhys, nullptr, &extCount, nullptr);
	if (extCount == 0) {
		return false;
	}
	OaVec<VkExtensionProperties> extensions(extCount);
	vkEnumerateDeviceExtensionProperties(InPhys, nullptr, &extCount, extensions.Data());
	for (OaU32 i = 0; i < extCount; ++i) {
		if (strcmp(extensions[i].extensionName, InName) == 0) {
			return true;
		}
	}
	return false;
}


OaU32 OaVkCountComputeQueueSlots(void* InPhysicalDevice) {
	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(InPhysicalDevice);
	if (!phys || !vkGetPhysicalDeviceQueueFamilyProperties) {
		return 0;
	}
	OaU32 qfCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, nullptr);
	if (qfCount == 0) {
		return 0;
	}
	OaVec<VkQueueFamilyProperties> qfProps(qfCount);
	vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, qfProps.Data());
	OaU32 computeSlots = 0;
	for (OaU32 i = 0; i < qfCount; ++i) {
		if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			computeSlots += qfProps[i].queueCount;
		}
	}
	return computeSlots;
}


OaU64 OaVkPhysicalDeviceRate(void* InPhysicalDevice, OaDeviceType InPreferred) {
	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(InPhysicalDevice);
	if (!phys || !vkGetPhysicalDeviceQueueFamilyProperties) {
		return 0;
	}

	VkPhysicalDeviceProperties props{};
	vkGetPhysicalDeviceProperties(phys, &props);

	OaDeviceType oaType = OaVkMapPhysicalType(props.deviceType);
	const OaU64 localBytes = OaVkPhysicalDeviceLocalHeapBytes(phys);
	const OaU64 heapMb = localBytes / (1024 * 1024);

	const OaU32 computeSlots = OaVkCountComputeQueueSlots(phys);
	if (computeSlots == 0) {
		return 0;
	}

	OaU64 score = 0;

	if (oaType == InPreferred) {
		score += 50'000'000ULL;
	}

	switch (props.deviceType) {
		case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: {
			score += 8'000'000ULL;
			break;
		}
		case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: {
			score += 4'000'000ULL;
			break;
		}
		case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: {
			score += 3'000'000ULL;
			break;
		}
		case VK_PHYSICAL_DEVICE_TYPE_CPU: {
			score += 1'000'000ULL;
			break;
		}
		default: {
			score += 2'000'000ULL;
			break;
		}
	}

	const OaU64 heapCap = static_cast<OaU64>(524288);
	const OaU64 heapPart = heapMb < heapCap ? heapMb : heapCap;
	score += heapPart;

	const auto& lim = props.limits;
	score += static_cast<OaU64>(lim.maxComputeWorkGroupInvocations) * 2000ULL;
	score += static_cast<OaU64>(lim.maxComputeWorkGroupSize[0]) * 100ULL;
	score += static_cast<OaU64>(computeSlots) * 50'000ULL;

	const OaU32 sharedKib = lim.maxComputeSharedMemorySize / 1024u;
	score += static_cast<OaU64>(std::min(sharedKib, 256u)) * 1000ULL;

	OaU32 subgroupSize = 0;
	if (vkGetPhysicalDeviceProperties2) {
		VkPhysicalDeviceSubgroupProperties subgroupProps = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
		};
		VkPhysicalDeviceProperties2 props2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
			.pNext = &subgroupProps,
		};
		vkGetPhysicalDeviceProperties2(phys, &props2);
		subgroupSize = subgroupProps.subgroupSize;
	}
	if (subgroupSize >= 64u) {
		score += 80'000ULL;
	} else if (subgroupSize >= 32u) {
		score += 50'000ULL;
	} else if (subgroupSize >= 16u) {
		score += 25'000ULL;
	}

	const bool hasCoopExt = OaVkPhysHasExtensionImpl(phys, "VK_KHR_cooperative_matrix");
	const bool hasPipelineLib = OaVkPhysHasExtensionImpl(phys, "VK_KHR_pipeline_library");
	const bool hasExtMem = OaVkPhysHasExtensionImpl(phys, "VK_KHR_external_memory");
	const bool hasExtMemFd = OaVkPhysHasExtensionImpl(phys, "VK_KHR_external_memory_fd");
	const bool hasDeviceGeneratedCommands = OaVkPhysHasExtensionImpl(phys, "VK_EXT_device_generated_commands");
	const bool hasVideoQueue = OaVkPhysHasExtensionImpl(phys, "VK_KHR_video_queue");
	const bool hasVideoDecodeQueue = OaVkPhysHasExtensionImpl(phys, "VK_KHR_video_decode_queue");
	const bool hasVideoDecodeH264 = OaVkPhysHasExtensionImpl(phys, "VK_KHR_video_decode_h264");
	const bool hasVideoDecodeH265 = OaVkPhysHasExtensionImpl(phys, "VK_KHR_video_decode_h265");
	const bool hasVideoDecodeAV1 = OaVkPhysHasExtensionImpl(phys, "VK_KHR_video_decode_av1");
	const bool hasVideoEncodeQueue = OaVkPhysHasExtensionImpl(phys, "VK_KHR_video_encode_queue");
	const bool hasVideoEncodeH264 = OaVkPhysHasExtensionImpl(phys, "VK_KHR_video_encode_h264");
	const bool hasVideoEncodeH265 = OaVkPhysHasExtensionImpl(phys, "VK_KHR_video_encode_h265");
	const bool hasSamplerYcbcr = OaVkPhysHasExtensionImpl(phys, "VK_KHR_sampler_ycbcr_conversion");

	VkPhysicalDeviceFeatures2 f2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
	};
	VkPhysicalDeviceVulkan11Features f11 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
	};
	VkPhysicalDeviceVulkan12Features f12 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
	};
	VkPhysicalDeviceVulkan13Features f13 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
	};
	VkPhysicalDeviceCooperativeMatrixFeaturesKHR fCoop = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
	};

	f2.pNext = &f11;
	f11.pNext = &f12;
	f12.pNext = &f13;
	if (hasCoopExt) {
		f13.pNext = &fCoop;
	} else {
		f13.pNext = nullptr;
	}

	if (vkGetPhysicalDeviceFeatures2) {
		vkGetPhysicalDeviceFeatures2(phys, &f2);
		if (f2.features.shaderInt64) {
			score += 40'000ULL;
		}
		if (f11.storageBuffer16BitAccess) {
			score += 15'000ULL;
		}
		if (f12.timelineSemaphore) {
			score += 100'000ULL;
		}
		if (f12.bufferDeviceAddress) {
			score += 40'000ULL;
		}
		if (f12.runtimeDescriptorArray) {
			score += 60'000ULL;
		}
		if (f12.shaderFloat16) {
			score += 20'000ULL;
		}
		if (f13.synchronization2) {
			score += 100'000ULL;
		}
		if (hasCoopExt && fCoop.cooperativeMatrix) {
			score += 200'000ULL;
		}
	}

	if (hasPipelineLib) {
		score += 30'000ULL;
	}
	if (hasExtMem && hasExtMemFd) {
		score += 20'000ULL;
	}
	if (hasVideoQueue) {
		score += 25'000ULL;
	}
	if (hasVideoQueue && hasVideoDecodeQueue) {
		score += 60'000ULL;
		if (hasVideoDecodeH264) score += 20'000ULL;
		if (hasVideoDecodeH265) score += 25'000ULL;
		if (hasVideoDecodeAV1) score += 30'000ULL;
	}
	if (hasVideoQueue && hasVideoEncodeQueue) {
		score += 50'000ULL;
		if (hasVideoEncodeH264) score += 20'000ULL;
		if (hasVideoEncodeH265) score += 25'000ULL;
	}
	if (hasSamplerYcbcr) {
		score += 15'000ULL;
	}

	const OaF64 bw = OaVkEstimateMemBandwidthGbpsEx(
		props.vendorID, props.deviceID, oaType, localBytes
	);
	score += static_cast<OaU64>(std::min(bw * 500.0, 500'000.0));

	const OaF64 tflops = OaVkEstimatePeakTflopsF32Ex(
		props.vendorID, props.deviceID, oaType, localBytes
	);
	score += static_cast<OaU64>(std::min(tflops * 8000.0, 2'000'000.0));

	score <<= 16;
	score |= static_cast<OaU64>(props.deviceID & 0xFFFFu);
	return score;
}


static OaU64 OaQueryHostRamTotalBytes() {
#ifdef __linux__
	const long pageCount = sysconf(_SC_PHYS_PAGES);
	const long pageSize = sysconf(_SC_PAGE_SIZE);
	if (pageCount < 0 || pageSize < 0) {
		return 0;
	}
	return static_cast<OaU64>(pageCount) * static_cast<OaU64>(pageSize);
#elif defined(_WIN32)
	MEMORYSTATUSEX state{};
	state.dwLength = sizeof(state);
	if (!GlobalMemoryStatusEx(&state)) {
		return 0;
	}
	return static_cast<OaU64>(state.ullTotalPhys);
#else
	return 0;
#endif
}


static OaMemoryUsage OaMakeMemoryUsageTotalOnly(OaU64 InTotalBytes) {
	OaMemoryUsage out{};
	if (InTotalBytes == 0) {
		return out;
	}
	out.TotalBytes = InTotalBytes;
	out.FreeBytes = InTotalBytes;
	out.UsedBytes = 0;
	out.UsedPercent = 0.0;
	return out;
}


OaMemoryUsage OaGetMemoryUsage(OaDevice InDevice) {
	if (!OaIsVulkanDevice(InDevice.Type)) {
		return OaMemoryUsage{};
	}

	// CPU Vulkan (OAV, etc.): tensors live in host RAM — report system RAM, not device-local heaps.
	if (InDevice.Type == OaDeviceType::VkCpu) {
		return OaMakeMemoryUsageTotalOnly(OaQueryHostRamTotalBytes());
	}

	// Unclassified / NPU-style devices: no stable device memory budget through this API.
	if (InDevice.Type == OaDeviceType::VkOther) {
		return OaMemoryUsage{};
	}

	OaEngine* rt = OaEngine::GetGlobal();
	if (!rt) {
		return OaMemoryUsage{};
	}

	OaU64 totalBytes = 0;

	if (rt->IsMultiDevice()) {
		if (InDevice.Index < 0) {
			return OaMemoryUsage{};
		}
		const OaU32 nodeIdx = static_cast<OaU32>(InDevice.Index);
		OaDeviceNode* node = rt->GetNode(nodeIdx);
		if (!node) {
			return OaMemoryUsage{};
		}
		if (node->Device.Info.Hardware.DeviceType != InDevice.Type) {
			return OaMemoryUsage{};
		}
		totalBytes = node->Device.Info.Hardware.VramBytes;
	} else {
		if (InDevice.Index != 0) {
			return OaMemoryUsage{};
		}
		if (rt->Device.Info.Hardware.DeviceType != InDevice.Type) {
			return OaMemoryUsage{};
		}
		totalBytes = rt->Device.Info.Hardware.VramBytes;
	}

	return OaMakeMemoryUsageTotalOnly(totalBytes);
}


// ─────────────────────────────────────────────────────────────────────────────
// OaCoopMatTrust — vendor/arch/driver gate for KHR_cooperative_matrix.
//
// This policy mirrors the capability and driver checks used by llama.cpp's
// ggml_vk_khr_cooperative_matrix_support trust table.
//
// Today the OA Vulkan init code at OaVkRefineCooperativeMatrixCapability()
// already gates HasCoopMatrix on a usable 16x16x16 shape being reported.
// This function adds the VENDOR layer on top: known-bad driver/hardware combos
// that lie about CoopMat support are blacklisted, OR the user opts in with
// OA_FORCE_COOPMAT=1.
//
// Returns true iff CoopMat is trustworthy on this (vendor, deviceId, driverId)
// triple. Caller is expected to AND this with the actual shape/feature gates.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Intel Xe2-architecture device IDs (Battlemage + Lunar Lake). The earlier
// Arc series (A380, A770 etc., device ID prefix 0x56xx) report coopmat support
// but regress below FP32 — they are EXCLUDED from this list.
[[nodiscard]] bool OaVkIsIntelXe2DeviceId(OaU32 InDeviceId) {
	// Battlemage (Xe2): 0xE2xx range (Pro B50, B570, B580, Pro B60, etc.)
	if ((InDeviceId & 0xFF00U) == 0xE200U) {
		return true;
	}
	// Lunar Lake integrated Xe2 — TODO when device IDs are stable.
	return false;
}

// AMD RDNA3-architecture device IDs (Navi 31/32/33: RX 7900 series, RX 7800/7700/7600).
// Pre-RDNA3 AMD reports KHR_cooperative_matrix support via the proprietary
// driver but miscompiles. RADV (Mesa open-source) is trustworthy on all generations.
[[nodiscard]] bool OaVkIsAmdRdna3DeviceId(OaU32 InDeviceId) {
	// Navi 3x: 0x744C (RX 7900 XTX/XT), 0x7448 (W7900), 0x7470 (RX 7700/7800),
	//          0x7480 (RX 7600), 0x7483 (RX 7600M), etc. Range roughly 0x7440-0x74FF.
	if (InDeviceId >= 0x7440U && InDeviceId <= 0x74FFU) {
		return true;
	}
	return false;
}

// Vulkan VkDriverId values relevant to vendor-trust gating.
inline constexpr OaU32 OaVkDriverIdAmdProprietary  = 1U;
inline constexpr OaU32 OaVkDriverIdAmdOpenSource   = 2U;
inline constexpr OaU32 OaVkDriverIdMesaRadv        = 3U;

} // namespace

void OaVkLogCoopMatShapes(const OaVkCoopMatShapes& InShapes, const char* InIndent) {
	const char* p = InIndent != nullptr ? InIndent : "";
	OA_LOG_INFO(OaLogComponent::Core,
		"%sCooperativeMatrix shapes (%u total enumerated):",
		p, InShapes.TotalShapesEnumerated);
	if (InShapes.Fp16AccFp32.Available) {
		OA_LOG_INFO(OaLogComponent::Core,
			"%s  Fp16AccFp32:  M=%u N=%u K=%u",
			p, InShapes.Fp16AccFp32.M, InShapes.Fp16AccFp32.N, InShapes.Fp16AccFp32.K);
	}
	if (InShapes.Fp16AccFp16.Available) {
		OA_LOG_INFO(OaLogComponent::Core,
			"%s  Fp16AccFp16:  M=%u N=%u K=%u",
			p, InShapes.Fp16AccFp16.M, InShapes.Fp16AccFp16.N, InShapes.Fp16AccFp16.K);
	}
	if (InShapes.Bf16AccFp32.Available) {
		OA_LOG_INFO(OaLogComponent::Core,
			"%s  Bf16AccFp32:  M=%u N=%u K=%u scope=subgroup",
			p, InShapes.Bf16AccFp32.M, InShapes.Bf16AccFp32.N, InShapes.Bf16AccFp32.K);
	}
	if (InShapes.Bf16AccFp32Workgroup.Available) {
		OA_LOG_INFO(OaLogComponent::Core,
			"%s  Bf16AccFp32Wg: M=%u N=%u K=%u scope=workgroup",
			p, InShapes.Bf16AccFp32Workgroup.M, InShapes.Bf16AccFp32Workgroup.N, InShapes.Bf16AccFp32Workgroup.K);
	}
	if (InShapes.Int8AccInt32.Available) {
		OA_LOG_INFO(OaLogComponent::Core,
			"%s  Int8AccInt32: M=%u N=%u K=%u",
			p, InShapes.Int8AccInt32.M, InShapes.Int8AccInt32.N, InShapes.Int8AccInt32.K);
	}
	OA_LOG_INFO(OaLogComponent::Core,
		"%s  Has16x16x16_Fp32Acc=%s  Has16x16x16_Fp16Acc=%s",
		p,
		InShapes.Has16x16x16_Fp32Acc ? "Y" : "N",
		InShapes.Has16x16x16_Fp16Acc ? "Y" : "N");
}

bool OaCoopMatTrust(OaU32 InVendorId, OaU32 InDeviceId, OaU32 InDriverId) {
	// Explicit override for CI / new-hardware testing.
	if (OaEnvFlag::IsSet("OA_FORCE_COOPMAT")) {
		return true;
	}

	switch (InVendorId) {
		case OaVkVendorIdNvidia:
			// NVIDIA pre-Turing has no CoopMat at all (extension absent), so by
			// the time we get here Turing+ is implied. Trusted.
			return true;

		case OaVkVendorIdAmd: {
			// RADV (open-source Mesa) is trustworthy on all generations.
			if (InDriverId == OaVkDriverIdMesaRadv) {
				return true;
			}
			// AMD proprietary / OPEN_SOURCE driver: trust only RDNA3+.
			const bool isAmdBlob =
				InDriverId == OaVkDriverIdAmdProprietary ||
				InDriverId == OaVkDriverIdAmdOpenSource;
			if (isAmdBlob) {
				return OaVkIsAmdRdna3DeviceId(InDeviceId);
			}
			// Unknown AMD driver: trust by default (probably a new Mesa fork).
			return true;
		}

		case OaVkVendorIdIntel:
			// Intel pre-Xe2 reports support but regresses. Only Xe2+ trusted.
			return OaVkIsIntelXe2DeviceId(InDeviceId);

		default:
			// ARM Mali, Imagination, Qualcomm, lavapipe, etc. — assume the
			// driver tells the truth. Override with OA_DISABLE_COOPMAT if not.
			return true;
	}
}

bool OaBf16Trust(OaU32 InVendorId, OaU32 InDeviceId, OaU32 InDriverId) {
	// Explicit override for CI / new-hardware testing.
	if (OaEnvFlag::IsSet("OA_FORCE_BF16")) {
		return true;
	}

	// Same vendor/arch reasoning as OaCoopMatTrust: a driver may advertise
	// shaderBFloat16Type but miscompile bf16 arithmetic (Intel pre-Xe2 Mesa/ANV,
	// AMD pre-RDNA3 proprietary blob). NVIDIA Turing+ and Mesa RADV are trustworthy.
	switch (InVendorId) {
		case OaVkVendorIdNvidia:
			return true;

		case OaVkVendorIdAmd: {
			if (InDriverId == OaVkDriverIdMesaRadv) {
				return true;
			}
			const bool isAmdBlob =
				InDriverId == OaVkDriverIdAmdProprietary ||
				InDriverId == OaVkDriverIdAmdOpenSource;
			if (isAmdBlob) {
				return OaVkIsAmdRdna3DeviceId(InDeviceId);
			}
			return true;
		}

		case OaVkVendorIdIntel:
			// Native bf16 unverified on Intel pre-Xe2 (same ANV backend that
			// miscompiles CoopMat). Only Xe2+ trusted.
			return OaVkIsIntelXe2DeviceId(InDeviceId);

		default:
			return true;
	}
}
