#include <oa/runtime/device.h>
#include <oa/core/log.h>
#include <oa/runtime/oaVk.h>
#include <oa/runtime/init.h>     // oavk::VendorId*, oavk::coopMatTrust forward decl
#include <oa/core/envFlag.h>

#include <oa/core/std/cString.h>

#ifdef __linux__
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif


oa::DeviceType oavk::mapPhysicalType(VkPhysicalDeviceType inType) {
	switch (inType) {
		case VK_PHYSICAL_DEVICE_TYPE_OTHER:
			return oa::DeviceType::VkOther;
		case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
			return oa::DeviceType::VkIntegrated;
		case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
			return oa::DeviceType::VkDiscrete;
		case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
			return oa::DeviceType::VkVirtualGpu;
		case VK_PHYSICAL_DEVICE_TYPE_CPU:
			return oa::DeviceType::VkCpu;
		default:
			return oa::DeviceType::VkOther;
	}
}


const char* oavk::physicalTypeLabel(VkPhysicalDeviceType inType) {
	switch (inType) {
		case VK_PHYSICAL_DEVICE_TYPE_OTHER:
			return "Other";
		case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
			return "Integrated GPU";
		case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
			return "Discrete GPU";
		case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
			return "virtual GPU";
		case VK_PHYSICAL_DEVICE_TYPE_CPU:
			return "CPU";
		default:
			return "Other";
	}
}


const char* oavk::vendorLabel(oa::U32 inVendorId) {
	switch (inVendorId) {
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


oa::String oavk::formatDriverVersion(oa::U32 inVersion) {
	return oa::String(
		oa::toString(static_cast<oa::U32>(VK_VERSION_MAJOR(inVersion)))
		+ "." + oa::toString(static_cast<oa::U32>(VK_VERSION_MINOR(inVersion)))
		+ "." + oa::toString(static_cast<oa::U32>(VK_VERSION_PATCH(inVersion)))
	);
}


const char* oavk::driverIdLabel(oa::U32 inDriverId) {
	switch (inDriverId) {
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


static oa::F64 estimateMemBandwidthGbpsHeuristic(
	oa::U32 inVendorId,
	oa::DeviceType inType,
	oa::U64 inLocalHeapBytes)
{
	oa::F64 vramGb = static_cast<oa::F64>(inLocalHeapBytes) / (1024.0 * 1024.0 * 1024.0);

	if (inType == oa::DeviceType::VkCpu) {
		return 20.0;
	}

	if (inType == oa::DeviceType::VkIntegrated) {
		if (inVendorId == 0x106B) {
			return 130.0;
		}
		if (inVendorId == 0x1002) {
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
		if (inVendorId == 0x8086) {
			return 55.0;
		}
		return 50.0;
	}

	switch (inVendorId) {
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


static oa::F64 estimatePeakTflopsF32Heuristic(
	oa::U32 inVendorId,
	oa::DeviceType inType,
	oa::U64 inLocalHeapBytes)
{
	oa::F64 vramGb =
		static_cast<oa::F64>(inLocalHeapBytes) / (1024.0 * 1024.0 * 1024.0);

	if (inType == oa::DeviceType::VkCpu) {
		return 0.1;
	}

	if (inType == oa::DeviceType::VkIntegrated) {
		if (inVendorId == 0x106B) {
			return vramGb >= 64.0 ? 18.0 : (vramGb >= 32.0 ? 12.0 : 6.0);
		}
		if (inVendorId == 0x1002) {
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
		if (inVendorId == 0x8086) {
			return 2.5;
		}
		return 1.5;
	}

	switch (inVendorId) {
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
// Sorted by (vendorId, deviceId) for binary search. extend when pick_rating skews on new silicon.

struct GpuSpecEntry {
	oa::U32 vendorId = 0;
	oa::U32 deviceId = 0;
	oa::F32 memBandwidthGbps = 0.0f;
	oa::F32 peakFp32Tflops = 0.0f;
	oa::U32 numSMs = 0;  // streaming multiprocessors (NVIDIA) / Compute units (AMD)
};


static oa::U64 gpuSpecSortKey(oa::U32 inVendorId, oa::U32 inDeviceId) {
	return (static_cast<oa::U64>(inVendorId) << 32) | static_cast<oa::U64>(inDeviceId);
}


static const GpuSpecEntry GpuSpecTable[] = {
	{0x1002, 0x744C, 960.0f, 61.0f, 96},   // RX 7900 XTX (96 CUs)
	{0x1002, 0x7480, 864.0f, 52.0f, 84},   // RX 7900 XT (84 CUs, approx.)
	{0x8086, 0x56A0, 560.0f, 17.0f, 32},   // Arc A770 16 GB (32 Xe-cores, approx.)
	{0x8086, 0x56A1, 512.0f, 14.0f, 28},   // Arc A750 (28 Xe-cores, approx.)
	{0x10DE, 0x2684, 1008.0f, 82.6f, 128}, // GeForce RTX 4090 (128 SMs, AD102)
	{0x10DE, 0x2704, 1008.0f, 82.6f, 128}, // RTX 4090 D / close AD102 variants
	{0x10DE, 0x2805, 720.0f, 42.0f, 58},   // RTX 4080 laptop (58 SMs, approx.)
	{0x10DE, 0x2820, 504.0f, 32.5f, 46},   // RTX 4080 laptop (alternate ID, 46 SMs approx.)
	{0x10DE, 0x2B85, 896.0f, 32.0f, 170},  // RTX 5090 laptop (GB202, 170 SMs — Blackwell)
	{0x10DE, 0x2B8C, 896.0f, 32.0f, 170},  // RTX 5090 laptop (alternate PCI ID)
	{0x10DE, 0x2C18, 896.0f, 32.0f, 170},  // RTX 5090 Mobile / Max-Q (170 SMs)
	{0x10DE, 0x2C58, 896.0f, 32.0f, 170},  // RTX 5090 Mobile / Max-Q (variant)
};


static bool gpuSpecLookup(oa::U32 inVendorId, oa::U32 inDeviceId, GpuSpecEntry& outEntry) {
	const oa::U64 key = gpuSpecSortKey(inVendorId, inDeviceId);
	oa::U32 lo = 0;
	oa::U32 hi = static_cast<oa::U32>(
		sizeof(GpuSpecTable) / sizeof(GpuSpecTable[0]));
	while (lo < hi) {
		const oa::U32 mid = (lo + hi) >> 1;
		const GpuSpecEntry& row = GpuSpecTable[mid];
		const oa::U64 midKey = gpuSpecSortKey(row.vendorId, row.deviceId);
		if (midKey < key) {
			lo = mid + 1;
		} else if (midKey > key) {
			hi = mid;
		} else {
			outEntry = row;
			return true;
		}
	}
	return false;
}


oa::F64 oavk::estimateMemBandwidthGbpsForDevice(
	oa::U32 inVendorId,
	oa::U32 inDeviceId,
	oa::DeviceType inType,
	oa::U64 inLocalHeapBytes)
{
	if (inDeviceId != oavk::PciDeviceIdUnknown) {
		GpuSpecEntry row{};
		if (gpuSpecLookup(inVendorId, inDeviceId, row)) {
			return static_cast<oa::F64>(row.memBandwidthGbps);
		}
	}
	return estimateMemBandwidthGbpsHeuristic(
		inVendorId, inType, inLocalHeapBytes);
}


oa::F64 oavk::estimatePeakTflopsF32ForDevice(
	oa::U32 inVendorId,
	oa::U32 inDeviceId,
	oa::DeviceType inType,
	oa::U64 inLocalHeapBytes)
{
	if (inDeviceId != oavk::PciDeviceIdUnknown) {
		GpuSpecEntry row{};
		if (gpuSpecLookup(inVendorId, inDeviceId, row)) {
			return static_cast<oa::F64>(row.peakFp32Tflops);
		}
	}
	return estimatePeakTflopsF32Heuristic(
		inVendorId, inType, inLocalHeapBytes);
}


oa::U32 oavk::estimateNumSmsForDevice(
	oa::U32 inVendorId,
	oa::U32 inDeviceId,
	oa::DeviceType inType,
	oa::U64 inLocalHeapBytes)
{
	// Try exact lookup first
	if (inDeviceId != oavk::PciDeviceIdUnknown) {
		GpuSpecEntry row{};
		if (gpuSpecLookup(inVendorId, inDeviceId, row)) {
			if (row.numSMs > 0) {
				return row.numSMs;
			}
		}
	}

	// Fallback heuristic based on VRAM and vendor
	const oa::U64 vramGb = inLocalHeapBytes / (1024 * 1024 * 1024);
	
	if (inVendorId == 0x10DE) {  // NVIDIA
		// Rough heuristic: ~8-12 SMs per GB for modern GPUs
		if (inType == oa::DeviceType::VkDiscrete) {
			if (vramGb >= 20) return 128;  // high-end (4090, 5090 class)
			if (vramGb >= 12) return 80;   // mid-high (4080 class)
			if (vramGb >= 8) return 60;    // mid (4070 class)
			return 46;                     // Entry (4060 class)
		}
		return 32;  // Integrated/mobile fallback
	}
	
	if (inVendorId == 0x1002) {  // AMD
		// CUs: ~60-96 for RDNA3
		if (inType == oa::DeviceType::VkDiscrete) {
			if (vramGb >= 20) return 96;   // 7900 XTX class
			if (vramGb >= 16) return 84;   // 7900 XT class
			if (vramGb >= 12) return 60;   // 7800 XT class
			return 48;                     // 7700 XT class
		}
		return 12;  // Integrated fallback
	}
	
	if (inVendorId == 0x8086) {  // Intel
		// Xe-cores: ~16-32 for Arc
		if (inType == oa::DeviceType::VkDiscrete) {
			if (vramGb >= 12) return 32;   // A770 class
			return 28;                     // A750 class
		}
		return 16;  // Integrated fallback
	}
	
	// Unknown vendor: conservative estimate
	return 32;
}


oa::F64 oavk::estimateMemBandwidthGbps(
	oa::U32 inVendorId,
	oa::DeviceType inType,
	oa::U64 inLocalHeapBytes)
{
	return oavk::estimateMemBandwidthGbpsForDevice(
		inVendorId, oavk::PciDeviceIdUnknown, inType, inLocalHeapBytes);
}


oa::F64 oavk::estimatePeakTflopsF32(
	oa::U32 inVendorId,
	oa::DeviceType inType,
	oa::U64 inLocalHeapBytes)
{
	return oavk::estimatePeakTflopsF32ForDevice(
		inVendorId, oavk::PciDeviceIdUnknown, inType, inLocalHeapBytes);
}


oa::U64 oavk::physicalDevicePickScore(
	oa::DeviceType inType,
	oa::U64 inLocalHeapBytes,
	oa::DeviceType inPreferred,
	oa::U32 inDeviceId)
{
	oa::U32 tierRank = 1;
	if (inType == inPreferred) {
		tierRank = 4;
	} else if (inType == oa::DeviceType::VkDiscrete || inType == oa::DeviceType::VkVirtualGpu) {
		tierRank = 3;
	} else if (inType == oa::DeviceType::VkIntegrated || inType == oa::DeviceType::VkOther) {
		tierRank = 2;
	}
	const oa::U64 tierPart = static_cast<oa::U64>(tierRank) << 60;
	const oa::U64 heapMb = inLocalHeapBytes / (1024 * 1024);
	const oa::U64 heapPart = oa::min(heapMb, (oa::U64{1} << 20) - oa::U64{1}) << 12;
	const oa::U64 idPart = static_cast<oa::U64>(inDeviceId) & 0xFFFu;
	return tierPart + heapPart + idPart;
}

static bool physicalDeviceHasExtension(
	const OaVkInstanceTable& inDispatch,
	VkPhysicalDevice inPhys,
	const char* inName)
{
	oa::U32 extCount = 0;
	inDispatch.vkEnumerateDeviceExtensionProperties(
		inPhys, nullptr, &extCount, nullptr);
	if (extCount == 0) {
		return false;
	}
	oa::Vec<VkExtensionProperties> extensions(extCount);
	inDispatch.vkEnumerateDeviceExtensionProperties(
		inPhys, nullptr, &extCount, extensions.data());
	for (oa::U32 i = 0; i < extCount; ++i) {
		if (oa::strcmp(extensions[i].extensionName, inName) == 0) {
			return true;
		}
	}
	return false;
}


oa::U32 oavk::countComputeQueueSlots(
	const OaVkInstanceTable& inDispatch,
	void* inPhysicalDevice)
{
	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(inPhysicalDevice);
	if (!phys || !inDispatch.vkGetPhysicalDeviceQueueFamilyProperties) {
		return 0;
	}
	oa::U32 qfCount = 0;
	inDispatch.vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, nullptr);
	if (qfCount == 0) {
		return 0;
	}
	oa::Vec<VkQueueFamilyProperties> qfProps(qfCount);
	inDispatch.vkGetPhysicalDeviceQueueFamilyProperties(
		phys, &qfCount, qfProps.data());
	oa::U32 computeSlots = 0;
	for (oa::U32 i = 0; i < qfCount; ++i) {
		if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			computeSlots += qfProps[i].queueCount;
		}
	}
	return computeSlots;
}


oa::U64 oavk::physicalDeviceRate(
	const OaVkInstanceTable& inDispatch,
	void* inPhysicalDevice,
	oa::DeviceType inPreferred)
{
	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(inPhysicalDevice);
	if (!phys || !inDispatch.vkGetPhysicalDeviceQueueFamilyProperties) {
		return 0;
	}

	VkPhysicalDeviceProperties props{};
	inDispatch.vkGetPhysicalDeviceProperties(phys, &props);

	oa::DeviceType oaType = oavk::mapPhysicalType(props.deviceType);
	const oa::U64 localBytes = oavk::physicalDeviceLocalHeapBytes(inDispatch, phys);
	const oa::U64 heapMb = localBytes / (1024 * 1024);

	const oa::U32 computeSlots = oavk::countComputeQueueSlots(inDispatch, phys);
	if (computeSlots == 0) {
		return 0;
	}

	oa::U64 score = 0;

	if (oaType == inPreferred) {
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

	const oa::U64 heapCap = static_cast<oa::U64>(524288);
	const oa::U64 heapPart = heapMb < heapCap ? heapMb : heapCap;
	score += heapPart;

	const auto& lim = props.limits;
	score += static_cast<oa::U64>(lim.maxComputeWorkGroupInvocations) * 2000ULL;
	score += static_cast<oa::U64>(lim.maxComputeWorkGroupSize[0]) * 100ULL;
	score += static_cast<oa::U64>(computeSlots) * 50'000ULL;

	const oa::U32 sharedKib = lim.maxComputeSharedMemorySize / 1024u;
	score += static_cast<oa::U64>(oa::min(sharedKib, 256u)) * 1000ULL;

	oa::U32 subgroupSize = 0;
	if (inDispatch.vkGetPhysicalDeviceProperties2) {
		VkPhysicalDeviceSubgroupProperties subgroupProps = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
		};
		VkPhysicalDeviceProperties2 props2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
			.pNext = &subgroupProps,
		};
		inDispatch.vkGetPhysicalDeviceProperties2(phys, &props2);
		subgroupSize = subgroupProps.subgroupSize;
	}
	if (subgroupSize >= 64u) {
		score += 80'000ULL;
	} else if (subgroupSize >= 32u) {
		score += 50'000ULL;
	} else if (subgroupSize >= 16u) {
		score += 25'000ULL;
	}

	const bool hasCoopExt = physicalDeviceHasExtension(inDispatch, phys, "VK_KHR_cooperative_matrix");
	const bool hasPipelineLib = physicalDeviceHasExtension(inDispatch, phys, "VK_KHR_pipeline_library");
	const bool hasExtMem = physicalDeviceHasExtension(inDispatch, phys, "VK_KHR_external_memory");
	const bool hasExtMemFd = physicalDeviceHasExtension(inDispatch, phys, "VK_KHR_external_memory_fd");
	const bool hasDeviceGeneratedCommands = physicalDeviceHasExtension(inDispatch, phys, "VK_EXT_device_generated_commands");
	const bool hasVideoQueue = physicalDeviceHasExtension(inDispatch, phys, "VK_KHR_video_queue");
	const bool hasVideoDecodeQueue = physicalDeviceHasExtension(inDispatch, phys, "VK_KHR_video_decode_queue");
	const bool hasVideoDecodeH264 = physicalDeviceHasExtension(inDispatch, phys, "VK_KHR_video_decode_h264");
	const bool hasVideoDecodeH265 = physicalDeviceHasExtension(inDispatch, phys, "VK_KHR_video_decode_h265");
	const bool hasVideoDecodeAV1 = physicalDeviceHasExtension(inDispatch, phys, "VK_KHR_video_decode_av1");
	const bool hasVideoEncodeQueue = physicalDeviceHasExtension(inDispatch, phys, "VK_KHR_video_encode_queue");
	const bool hasVideoEncodeH264 = physicalDeviceHasExtension(inDispatch, phys, "VK_KHR_video_encode_h264");
	const bool hasVideoEncodeH265 = physicalDeviceHasExtension(inDispatch, phys, "VK_KHR_video_encode_h265");
	const bool hasSamplerYcbcr = physicalDeviceHasExtension(inDispatch, phys, "VK_KHR_sampler_ycbcr_conversion");

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

	if (inDispatch.vkGetPhysicalDeviceFeatures2) {
		inDispatch.vkGetPhysicalDeviceFeatures2(phys, &f2);
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

	const oa::F64 bw = oavk::estimateMemBandwidthGbpsForDevice(
		props.vendorID, props.deviceID, oaType, localBytes
	);
	score += static_cast<oa::U64>(oa::min(bw * 500.0, 500'000.0));

	const oa::F64 tflops = oavk::estimatePeakTflopsF32ForDevice(
		props.vendorID, props.deviceID, oaType, localBytes
	);
	score += static_cast<oa::U64>(oa::min(tflops * 8000.0, 2'000'000.0));

	score <<= 16;
	score |= static_cast<oa::U64>(props.deviceID & 0xFFFFu);
	return score;
}


static oa::U64 queryHostRamTotalBytes() {
#ifdef __linux__
	const long pageCount = sysconf(_SC_PHYS_PAGES);
	const long pageSize = sysconf(_SC_PAGE_SIZE);
	if (pageCount < 0 || pageSize < 0) {
		return 0;
	}
	return static_cast<oa::U64>(pageCount) * static_cast<oa::U64>(pageSize);
#elif defined(_WIN32)
	MEMORYSTATUSEX state{};
	state.dwLength = sizeof(state);
	if (!globalMemoryStatusEx(&state)) {
		return 0;
	}
	return static_cast<oa::U64>(state.ullTotalPhys);
#else
	return 0;
#endif
}


static oa::MemoryUsage makeMemoryUsageTotalOnly(oa::U64 inTotalBytes) {
	oa::MemoryUsage out{};
	if (inTotalBytes == 0) {
		return out;
	}
	out.totalBytes = inTotalBytes;
	out.freeBytes = inTotalBytes;
	out.usedBytes = 0;
	out.usedPercent = 0.0;
	return out;
}


oa::MemoryUsage oa::getMemoryUsage(oa::Device inDevice) {
	if (!oa::isVulkanDevice(inDevice.type)) {
		return oa::MemoryUsage{};
	}

	// CPU vulkan (OAV, etc.): tensors live in host RAM — report system RAM, not device-local heaps.
	if (inDevice.type == oa::DeviceType::VkCpu) {
		return makeMemoryUsageTotalOnly(queryHostRamTotalBytes());
	}

	// A logical vulkan value does not identify an engine, allocator, or physical
	// enumeration index. Live budget and usage belong to oa::Engine.
	return oa::MemoryUsage{};
}


// ─────────────────────────────────────────────────────────────────────────────
// oavk::coopMatTrust — vendor/arch/driver gate for KHR_cooperative_matrix.
//
// This policy mirrors the capability and driver checks used by llama.cpp's
// ggml_vk_khr_cooperative_matrix_support trust table.
//
// Today the OA vulkan init code at ML feature-module cooperative-matrix discovery
// already gates hasCoopMatrix on a usable 16x16x16 shape being reported.
// This function adds the VENDOR layer on top: known-bad driver/hardware combos
// that lie about CoopMat support are blacklisted, OR the user opts in with
// OA_FORCE_COOPMAT=1.
//
// Returns true iff CoopMat is trustworthy on this (vendor, deviceId, driverId)
// triple. Caller is expected to AND this with the actual shape/feature gates.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Intel Xe2-architecture device iDs (Battlemage + Lunar Lake). The earlier
// Arc series (A380, A770 etc., device ID prefix 0x56xx) report coopmat support
// but regress below FP32 — they are EXCLUDED from this list.
[[nodiscard]] bool isIntelXe2DeviceId(oa::U32 inDeviceId) {
	// battlemage (Xe2): 0xE2xx range (Pro B50, B570, B580, Pro B60, etc.)
	if ((inDeviceId & 0xFF00U) == 0xE200U) {
		return true;
	}
	// Lunar Lake integrated Xe2 — TODO when device IDs are stable.
	return false;
}

// AMD RDNA3-architecture device iDs (Navi 31/32/33: RX 7900 series, RX 7800/7700/7600).
// Pre-RDNA3 AMD reports KHR_cooperative_matrix support via the proprietary
// driver but miscompiles. RADV (Mesa open-source) is trustworthy on all generations.
[[nodiscard]] bool isAmdRdna3DeviceId(oa::U32 inDeviceId) {
	// Navi 3x: 0x744C (RX 7900 XTX/XT), 0x7448 (W7900), 0x7470 (RX 7700/7800),
	//          0x7480 (RX 7600), 0x7483 (RX 7600M), etc. range roughly 0x7440-0x74FF.
	if (inDeviceId >= 0x7440U && inDeviceId <= 0x74FFU) {
		return true;
	}
	return false;
}

// vulkan VkDriverId values relevant to vendor-trust gating.
inline constexpr oa::U32 DriverIdAmdProprietary  = 1U;
inline constexpr oa::U32 DriverIdAmdOpenSource   = 2U;
inline constexpr oa::U32 DriverIdMesaRadv        = 3U;

} // namespace

void oavk::logCoopMatShapes(const oavk::CoopMatShapes& inShapes, const char* inIndent) {
	const char* p = inIndent != nullptr ? inIndent : "";
	OaLogInfo(oa::LogComponent::Runtime,
		"%sCooperativeMatrix shapes (%u total enumerated):",
		p, inShapes.totalShapesEnumerated);
	if (inShapes.fp16AccFp32.available) {
		OaLogInfo(oa::LogComponent::Runtime,
			"%s  fp16AccFp32:  M=%u N=%u K=%u",
			p, inShapes.fp16AccFp32.m, inShapes.fp16AccFp32.n, inShapes.fp16AccFp32.k);
	}
	if (inShapes.fp16AccFp16.available) {
		OaLogInfo(oa::LogComponent::Runtime,
			"%s  fp16AccFp16:  M=%u N=%u K=%u",
			p, inShapes.fp16AccFp16.m, inShapes.fp16AccFp16.n, inShapes.fp16AccFp16.k);
	}
	if (inShapes.bf16AccFp32.available) {
		OaLogInfo(oa::LogComponent::Runtime,
			"%s  bf16AccFp32:  M=%u N=%u K=%u scope=subgroup",
			p, inShapes.bf16AccFp32.m, inShapes.bf16AccFp32.n, inShapes.bf16AccFp32.k);
	}
	if (inShapes.bf16AccFp32Workgroup.available) {
		OaLogInfo(oa::LogComponent::Runtime,
			"%s  Bf16AccFp32Wg: M=%u N=%u K=%u scope=workgroup",
			p, inShapes.bf16AccFp32Workgroup.m, inShapes.bf16AccFp32Workgroup.n, inShapes.bf16AccFp32Workgroup.k);
	}
	if (inShapes.int8AccInt32.available) {
		OaLogInfo(oa::LogComponent::Runtime,
			"%s  int8AccInt32: M=%u N=%u K=%u",
			p, inShapes.int8AccInt32.m, inShapes.int8AccInt32.n, inShapes.int8AccInt32.k);
	}
	OaLogInfo(oa::LogComponent::Runtime,
		"%s  has16x16x16_Fp32Acc=%s  has16x16x16_Fp16Acc=%s",
		p,
		inShapes.has16x16x16_Fp32Acc ? "Y" : "N",
		inShapes.has16x16x16_Fp16Acc ? "Y" : "N");
}

bool oavk::coopMatTrust(oa::U32 inVendorId, oa::U32 inDeviceId, oa::U32 inDriverId) {
	// Explicit override for CI / new-hardware testing.
	if (oa::EnvFlag::isSet("OA_FORCE_COOPMAT")) {
		return true;
	}

	switch (inVendorId) {
		case oavk::VendorIdNvidia:
			// NVIDIA pre-Turing has no CoopMat at all (extension absent), so by
			// the time we get here Turing+ is implied. Trusted.
			return true;

		case oavk::VendorIdAmd: {
			// RADV (open-source Mesa) is trustworthy on all generations.
			if (inDriverId == DriverIdMesaRadv) {
				return true;
			}
			// AMD proprietary / OPEN_SOURCE driver: trust only RDNA3+.
			const bool isAmdBlob =
				inDriverId == DriverIdAmdProprietary ||
				inDriverId == DriverIdAmdOpenSource;
			if (isAmdBlob) {
				return isAmdRdna3DeviceId(inDeviceId);
			}
			// Unknown AMD driver: trust by default (probably a new Mesa fork).
			return true;
		}

		case oavk::VendorIdIntel:
			// Intel pre-Xe2 reports support but regresses. Only Xe2+ trusted.
			return isIntelXe2DeviceId(inDeviceId);

		default:
			// ARM Mali, Imagination, Qualcomm, lavapipe, etc. — assume the
			// driver tells the truth. Override with OA_DISABLE_COOPMAT if not.
			return true;
	}
}

bool oavk::bf16Trust(oa::U32 inVendorId, oa::U32 inDeviceId, oa::U32 inDriverId) {
	// Explicit override for CI / new-hardware testing.
	if (oa::EnvFlag::isSet("OA_FORCE_BF16")) {
		return true;
	}

	// Same vendor/arch reasoning as oavk::coopMatTrust: a driver may advertise
	// shaderBFloat16Type but miscompile bf16 arithmetic (Intel pre-Xe2 Mesa/ANV,
	// AMD pre-RDNA3 proprietary blob). NVIDIA Turing+ and Mesa RADV are trustworthy.
	switch (inVendorId) {
		case oavk::VendorIdNvidia:
			return true;

		case oavk::VendorIdAmd: {
			if (inDriverId == DriverIdMesaRadv) {
				return true;
			}
			const bool isAmdBlob =
				inDriverId == DriverIdAmdProprietary ||
				inDriverId == DriverIdAmdOpenSource;
			if (isAmdBlob) {
				return isAmdRdna3DeviceId(inDeviceId);
			}
			return true;
		}

		case oavk::VendorIdIntel:
			// Native bf16 unverified on Intel pre-xe2 (same ANV backend that
			// miscompiles CoopMat). Only Xe2+ trusted.
			return isIntelXe2DeviceId(inDeviceId);

		default:
			return true;
	}
}
