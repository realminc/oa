#include <Oa/Core/Log.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Runtime/Device.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Runtime/Instance.h>
#include <Oa/Runtime/Bindless.h>
#include "DeviceBuilder.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>


// VRAM: MiB below 1 GiB; otherwise whole GB / TB / PB (binary base, "GB" label like retail GPUs).
static OaString OaVkFormatCapacityBytesHuman(OaU64 InBytes) {
	char buf[72];
	const OaU64 kib = 1024ULL;
	const OaU64 mib = kib * kib;
	const OaU64 gib = mib * kib;
	const OaU64 tib = gib * kib;
	const OaU64 pib = tib * kib;
	if (InBytes < gib) {
		OaU64 mibVal = (InBytes + mib / 2) / mib;
		if (mibVal == 0) {
			mibVal = 1;
		}
		std::snprintf(
			buf, sizeof(buf), "%llu MiB", static_cast<unsigned long long>(mibVal));
		return OaString(buf);
	}
	if (InBytes < tib) {
		const double gigabytes =
			static_cast<double>(InBytes) / static_cast<double>(gib);
		std::snprintf(buf, sizeof(buf), "%.0f GB", std::round(gigabytes));
		return OaString(buf);
	}
	if (InBytes < pib) {
		const double terabytes =
			static_cast<double>(InBytes) / static_cast<double>(tib);
		std::snprintf(buf, sizeof(buf), "%.2f TB", terabytes);
		return OaString(buf);
	}
	const double petabytes = static_cast<double>(InBytes) / static_cast<double>(pib);
	std::snprintf(buf, sizeof(buf), "%.2f PB", petabytes);
	return OaString(buf);
}


// EstMemBandwidthGbps is gigabytes per second (see Device.h).
static OaString OaVkFormatMemoryBandwidthHuman(OaF64 InGigabytesPerSecond) {
	char buf[72];
	if (InGigabytesPerSecond <= 0.0) {
		return OaString("0 MB/s");
	}
	if (InGigabytesPerSecond < 1.0) {
		const double mibPerSec = InGigabytesPerSecond * 1024.0;
		std::snprintf(buf, sizeof(buf), "%.0f MB/s", mibPerSec);
		return OaString(buf);
	}
	if (InGigabytesPerSecond < 1024.0) {
		std::snprintf(buf, sizeof(buf), "%.0f GB/s", InGigabytesPerSecond);
		return OaString(buf);
	}
	std::snprintf(buf, sizeof(buf), "%.2f TB/s", InGigabytesPerSecond / 1024.0);
	return OaString(buf);
}


static OaString OaVkFormatPeakTflopsHuman(OaF64 InTflops) {
	char buf[72];
	if (InTflops <= 0.0) {
		return OaString("0 TFLOPS");
	}
	if (InTflops < 1000.0) {
		std::snprintf(buf, sizeof(buf), "%.2f TFLOPS", InTflops);
		return OaString(buf);
	}
	if (InTflops < 1.0e6) {
		std::snprintf(buf, sizeof(buf), "%.2f PFLOPS", InTflops / 1000.0);
		return OaString(buf);
	}
	std::snprintf(buf, sizeof(buf), "%.2f EFLOPS", InTflops / 1.0e6);
	return OaString(buf);
}

enum class OaDeviceInfoLogMode : OaU8 {
	Compact,
	Full,
	Off,
};

static bool OaAsciiEqualsIgnoreCase(const char* InA, const char* InB) {
	if (InA == nullptr || InB == nullptr) {
		return InA == InB;
	}
	while (*InA != '\0' && *InB != '\0') {
		const char a = static_cast<char>(std::tolower(static_cast<unsigned char>(*InA)));
		const char b = static_cast<char>(std::tolower(static_cast<unsigned char>(*InB)));
		if (a != b) {
			return false;
		}
		++InA;
		++InB;
	}
	return *InA == '\0' && *InB == '\0';
}

static OaDeviceInfoLogMode OaDefaultDeviceInfoLogMode() {
	const OaString env = OaEnvFlag::GetString("OA_LOG_DEVICE_INIT", "");
	if (!env.empty()) {
		const char* value = env.c_str();
		if (OaAsciiEqualsIgnoreCase(value, "full") ||
		    OaAsciiEqualsIgnoreCase(value, "debug") ||
		    OaAsciiEqualsIgnoreCase(value, "verbose") ||
		    OaAsciiEqualsIgnoreCase(value, "1") ||
		    OaAsciiEqualsIgnoreCase(value, "true") ||
		    OaAsciiEqualsIgnoreCase(value, "on")) {
			return OaDeviceInfoLogMode::Full;
		}
		if (OaAsciiEqualsIgnoreCase(value, "compact") ||
		    OaAsciiEqualsIgnoreCase(value, "minimal") ||
		    OaAsciiEqualsIgnoreCase(value, "min")) {
			return OaDeviceInfoLogMode::Compact;
		}
		if (OaAsciiEqualsIgnoreCase(value, "off") ||
		    OaAsciiEqualsIgnoreCase(value, "none") ||
		    OaAsciiEqualsIgnoreCase(value, "0") ||
		    OaAsciiEqualsIgnoreCase(value, "false") ||
		    OaAsciiEqualsIgnoreCase(value, "no")) {
			return OaDeviceInfoLogMode::Off;
		}
	}
#ifdef NDEBUG
	return OaDeviceInfoLogMode::Compact;
#else
	return OaDeviceInfoLogMode::Full;
#endif
}


OaU64 OaVkPhysicalDeviceLocalHeapBytes(void* InPhysicalDevice) {
	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(InPhysicalDevice);
	VkPhysicalDeviceMemoryProperties mp{};
	vkGetPhysicalDeviceMemoryProperties(phys, &mp);
	OaU64 sum = 0;
	for (OaU32 i = 0; i < mp.memoryHeapCount; ++i) {
		if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
			sum += mp.memoryHeaps[i].size;
		}
	}
	return sum;
}


void OaVkLogPhysicalDeviceSurvey(OaU32 InCount,	void* const* InPhysicalDevices,	OaDeviceType InPreferred) {
	if (!InPhysicalDevices || InCount == 0) {
		return;
	}
	if (OaDefaultDeviceInfoLogMode() != OaDeviceInfoLogMode::Full) {
		return;
	}
	OA_LOG_INFO(OaLogComponent::Core,
		"Vulkan: enumerating physical devices (found %u)",
		static_cast<unsigned>(InCount));
	for (OaU32 i = 0; i < InCount; ++i) {
		VkPhysicalDevice phys =
			static_cast<VkPhysicalDevice>(InPhysicalDevices[i]);
		if (!phys) {
			continue;
		}
		VkPhysicalDeviceProperties props{};
		vkGetPhysicalDeviceProperties(phys, &props);
		const OaU64 localBytes = OaVkPhysicalDeviceLocalHeapBytes(phys);
		const OaU64 rating = OaVkPhysicalDeviceRate(phys, InPreferred);
		const OaU32 slots = OaVkCountComputeQueueSlots(phys);
		const OaDeviceType oaType = OaVkMapPhysicalType(props.deviceType);
		const OaF64 bw = OaVkEstimateMemBandwidthGbpsEx(
			props.vendorID, props.deviceID, oaType, localBytes);
		const OaF64 tflops = OaVkEstimatePeakTflopsF32Ex(
			props.vendorID, props.deviceID, oaType, localBytes);
		OA_LOG_INFO(OaLogComponent::Core,
			"  [%u] %s | PickScore=%s | vram=%s MiB | compute_queue_slots=%u | est_bw=%.0f GB/s | est_fp32=%.1f TFLOPS | %s",
			static_cast<unsigned>(i),
			props.deviceName,
			OaFormatNumberU64(rating).c_str(),
			OaFormatNumberU64(localBytes / (1024ull * 1024ull)).c_str(),
			static_cast<unsigned>(slots),
			bw,
			tflops,
			OaVkPhysicalTypeLabel(props.deviceType)
		);
	}
}

// Extension probing and feature detection now handled by DeviceBuilder + feature modules
// ------------------------------------------------------------
// OaVkPlanDeviceQueues  — PATCHED (non-static for DeviceBuilder)
// ------------------------------------------------------------
OaStatus OaVkPlanDeviceQueues(
	VkPhysicalDevice  InPhys,
	VkSurfaceKHR      InSurface,
	OaVkQueuePlan&    OutPlan,
	bool              InHintNeedsPresentation)
{
	OaU32 qfCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(InPhys, &qfCount, nullptr);
	OaVec<VkQueueFamilyProperties> qfProps(qfCount);
	vkGetPhysicalDeviceQueueFamilyProperties(InPhys, &qfCount, qfProps.Data());

	OutPlan.PriorityBacking.Clear();
	OutPlan.QueueCIs.Clear();
	OutPlan.GraphicsQF             = UINT32_MAX;
	OutPlan.PresentQF              = UINT32_MAX;
	OutPlan.WantsSurfacePresentation = false;
	OutPlan.ComputeSlots           = 0;
	OutPlan.DedicatedTransferSlots = 0;

	for (OaU32 idx = 0; idx < qfCount; ++idx) {
		const VkQueueFlags qflags = qfProps[idx].queueFlags;
		if (qflags & VK_QUEUE_COMPUTE_BIT)
			OutPlan.ComputeSlots += qfProps[idx].queueCount;
		if ((qflags & VK_QUEUE_TRANSFER_BIT) && !(qflags & VK_QUEUE_COMPUTE_BIT))
			OutPlan.DedicatedTransferSlots += qfProps[idx].queueCount;
	}

	const bool wantSurface  = (InSurface != VK_NULL_HANDLE);
	// "wantGraphics" covers the real-surface path AND the hint-only path.
	const bool wantGraphics = wantSurface || InHintNeedsPresentation;

	OaVec<VkBool32> presentSupport;
	if (wantSurface) {
		if (!vkGetPhysicalDeviceSurfaceSupportKHR) {
			return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				"vkGetPhysicalDeviceSurfaceSupportKHR unavailable (load instance WSI)");
		}
		presentSupport.Resize(qfCount);
		for (OaU32 idx = 0; idx < qfCount; ++idx) {
			VkResult sr = vkGetPhysicalDeviceSurfaceSupportKHR(
				InPhys, idx, InSurface, &presentSupport[idx]);
			if (sr != VK_SUCCESS) presentSupport[idx] = VK_FALSE;
		}
	}

	OutPlan.ComputeQF          = UINT32_MAX;
	OutPlan.AsyncComputeQF     = UINT32_MAX;
	OutPlan.TransferQF         = UINT32_MAX;
	OutPlan.ComputeHasMultiQueue = false;

	if (wantGraphics) {
		if (wantSurface) {
			// Original surface path: need graphics + present on the same family.
			OaU32 graphicsPresent = UINT32_MAX;
			for (OaU32 idx = 0; idx < qfCount; ++idx) {
				const VkQueueFlags fl = qfProps[idx].queueFlags;
				if ((fl & VK_QUEUE_GRAPHICS_BIT) && presentSupport[idx] == VK_TRUE) {
					graphicsPresent = idx;
					break;
				}
			}
			if (graphicsPresent == UINT32_MAX) {
				return OaStatus::Error(
					OaStatusCode::DeviceNotFound,
					"no queue family with graphics + surface present support");
			}
			OutPlan.GraphicsQF = graphicsPresent;
			OutPlan.PresentQF  = graphicsPresent;
		} else {
			// Hint-only path: find any graphics-capable family (no surface check).
			// Present support is verified later in OaGraphicsEngine::InitPresentation.
			OaU32 gfxFamily = UINT32_MAX;
			for (OaU32 idx = 0; idx < qfCount; ++idx) {
				if (qfProps[idx].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
					gfxFamily = idx;
					break;
				}
			}
			if (gfxFamily == UINT32_MAX) {
				return OaStatus::Error(
					OaStatusCode::DeviceNotFound,
					"InHintNeedsPresentation=true but device has no VK_QUEUE_GRAPHICS_BIT family");
			}
			OutPlan.GraphicsQF = gfxFamily;
			OutPlan.PresentQF  = gfxFamily;  // confirmed in InitPresentation
		}

		OutPlan.WantsSurfacePresentation = true;

		if (qfProps[OutPlan.GraphicsQF].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			OutPlan.ComputeQF = OutPlan.GraphicsQF;
			if (qfProps[OutPlan.GraphicsQF].queueCount >= 2)
				OutPlan.ComputeHasMultiQueue = true;
		} else {
			for (OaU32 idx = 0; idx < qfCount; ++idx) {
				if (qfProps[idx].queueFlags & VK_QUEUE_COMPUTE_BIT) {
					OutPlan.ComputeQF = idx;
					if (qfProps[idx].queueCount >= 2)
						OutPlan.ComputeHasMultiQueue = true;
					break;
				}
			}
			if (OutPlan.ComputeQF == UINT32_MAX)
				return OaStatus::Error(OaStatusCode::DeviceNotFound, "no compute queue family");
		}
	} else {
		// Pure compute path (original).
		for (OaU32 idx = 0; idx < qfCount; idx++) {
			if ((qfProps[idx].queueFlags & VK_QUEUE_COMPUTE_BIT) && OutPlan.ComputeQF == UINT32_MAX) {
				OutPlan.ComputeQF = idx;
				if (qfProps[idx].queueCount >= 2)
					OutPlan.ComputeHasMultiQueue = true;
			}
			if ((qfProps[idx].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
				!(qfProps[idx].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
				idx != OutPlan.ComputeQF && OutPlan.AsyncComputeQF == UINT32_MAX)
			{
				OutPlan.AsyncComputeQF = idx;
			}
			if ((qfProps[idx].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
				!(qfProps[idx].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
				OutPlan.TransferQF == UINT32_MAX)
			{
				OutPlan.TransferQF = idx;
			}
		}
	}

	if (OutPlan.ComputeQF == UINT32_MAX)
		return OaStatus::Error(OaStatusCode::DeviceNotFound, "no compute queue family");

	// Async + dedicated-transfer scan (same for both graphics and compute paths).
	for (OaU32 idx = 0; idx < qfCount; idx++) {
		if ((qfProps[idx].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
			!(qfProps[idx].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
			idx != OutPlan.ComputeQF && OutPlan.AsyncComputeQF == UINT32_MAX)
		{
			OutPlan.AsyncComputeQF = idx;
		}
		if ((qfProps[idx].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
			!(qfProps[idx].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
			OutPlan.TransferQF == UINT32_MAX)
		{
			OutPlan.TransferQF = idx;
		}
	}

	if (OutPlan.TransferQF == UINT32_MAX)
		OutPlan.TransferQF = OutPlan.ComputeQF;

	// --- Video queue selection with per-family codec cross-check (Gap 1) ---
	// A family can advertise VK_QUEUE_VIDEO_DECODE_BIT_KHR but not support a
	// particular codec (e.g. AV1). Chain VkQueueFamilyVideoPropertiesKHR to
	// query videoCodecOperations per family and pick the broadest. Store the
	// ops on the plan so decoder Create() can verify codec support.
	{
		OaU32 qf2Count = qfCount;
		OaVec<VkQueueFamilyProperties2> qfProps2(qf2Count);
		OaVec<VkQueueFamilyVideoPropertiesKHR> videoProps(qf2Count);
		for (OaU32 idx = 0; idx < qf2Count; ++idx) {
			videoProps[idx] = {};
			videoProps[idx].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR;
			videoProps[idx].pNext = nullptr;
			videoProps[idx].videoCodecOperations = 0;
			qfProps2[idx] = {};
			qfProps2[idx].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
			qfProps2[idx].pNext = &videoProps[idx];
		}
		const bool hasQueueFamilyProperties2 = vkGetPhysicalDeviceQueueFamilyProperties2 != nullptr;
		if (hasQueueFamilyProperties2) {
			vkGetPhysicalDeviceQueueFamilyProperties2(InPhys, &qf2Count, qfProps2.Data());
		}

		// Pick video decode family: prefer the family with the broadest codec
		// coverage. Ties broken by first-match (stable behavior).
		OaU32 bestDecodeQF = UINT32_MAX;
		OaU32 bestDecodePopcount = 0;
		OaU32 bestEncodeQF = UINT32_MAX;
		OaU32 bestEncodePopcount = 0;
		for (OaU32 idx = 0; idx < qf2Count; ++idx) {
			const VkQueueFlags qflags = hasQueueFamilyProperties2
				? qfProps2[idx].queueFamilyProperties.queueFlags
				: qfProps[idx].queueFlags;
			if ((qflags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) != 0) {
				const auto ops = videoProps[idx].videoCodecOperations;
				auto popcount = static_cast<OaU32>(__builtin_popcount(
					static_cast<unsigned>(ops)));
				if (bestDecodeQF == UINT32_MAX || popcount > bestDecodePopcount) {
					bestDecodeQF = idx;
					bestDecodePopcount = popcount;
					OutPlan.VideoDecodeCodecOps = ops;
				}
			}
			if ((qflags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR) != 0) {
				const auto ops = videoProps[idx].videoCodecOperations;
				auto popcount = static_cast<OaU32>(__builtin_popcount(
					static_cast<unsigned>(ops)));
				if (bestEncodeQF == UINT32_MAX || popcount > bestEncodePopcount) {
					bestEncodeQF = idx;
					bestEncodePopcount = popcount;
					OutPlan.VideoEncodeCodecOps = ops;
				}
			}
		}
		OutPlan.VideoDecodeQF = bestDecodeQF;
		OutPlan.VideoEncodeQF = bestEncodeQF;
	}

	OutPlan.HasAsync         = (OutPlan.AsyncComputeQF != UINT32_MAX) || OutPlan.ComputeHasMultiQueue;
	OutPlan.MainComputeCount = (OutPlan.ComputeHasMultiQueue && OutPlan.AsyncComputeQF == UINT32_MAX) ? 2 : 1;

	OaVec<OaU32> need;
	need.Resize(qfCount);
	for (OaU32 idx = 0; idx < qfCount; ++idx) need[idx] = 0;

	auto bump = [&](OaU32 fam, OaU32 cnt) {
		if (fam == UINT32_MAX || fam >= qfCount || cnt == 0) return;
		const OaU32 cap  = qfProps[fam].queueCount;
		const OaU32 want = cnt > cap ? cap : cnt;
		if (want > need[fam]) need[fam] = want;
	};

	bump(OutPlan.ComputeQF, OutPlan.MainComputeCount);
	if (OutPlan.AsyncComputeQF != UINT32_MAX && OutPlan.AsyncComputeQF != OutPlan.ComputeQF)
		bump(OutPlan.AsyncComputeQF, 1);
	if (OutPlan.TransferQF != OutPlan.ComputeQF && OutPlan.TransferQF != OutPlan.AsyncComputeQF)
		bump(OutPlan.TransferQF, 1);
	if (OutPlan.WantsSurfacePresentation && OutPlan.GraphicsQF != UINT32_MAX &&
		OutPlan.GraphicsQF != OutPlan.ComputeQF)
	{
		bump(OutPlan.GraphicsQF, 1);
	}
	if (OutPlan.VideoDecodeQF != UINT32_MAX)
		bump(OutPlan.VideoDecodeQF, 1);
	if (OutPlan.VideoEncodeQF != UINT32_MAX)
		bump(OutPlan.VideoEncodeQF, 1);

	for (OaU32 fi = 0; fi < qfCount; ++fi) {
		if (need[fi] == 0) continue;
		const OaU32 baseIdx = OutPlan.PriorityBacking.Size();
		for (OaU32 q = 0; q < need[fi]; ++q) {
			OaF32 pri = 1.0f;
			if (fi == OutPlan.ComputeQF && need[fi] >= 2 && q > 0) pri = 0.5f;
			OutPlan.PriorityBacking.PushBack(pri);
		}
		VkDeviceQueueCreateInfo ci{};
		ci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		ci.queueFamilyIndex = fi;
		ci.queueCount       = need[fi];
		ci.pQueuePriorities = OutPlan.PriorityBacking.Data() + baseIdx;
		OutPlan.QueueCIs.PushBack(ci);
	}

	if (OutPlan.QueueCIs.Empty())
		return OaStatus::Error(OaStatusCode::DeviceNotFound, "queue create info list empty");

	return OaStatus::Ok();
}

// Feature querying, refinement, and extension collection now handled by feature modules
// (CoreFeatures, MlFeatures, VisionFeatures, AudioFeatures, RenderFeatures)
// Old monolithic functions removed - see feature modules for new implementation

// ------------------------------------------------------------
// OaVkDevice::CreateFromPhysical — Refactored to use DeviceBuilder
// ------------------------------------------------------------
OaResult<OaVkDevice> OaVkDevice::CreateFromPhysical(
	void*  InInstance,
	void*  InPhysicalDevice,
	OaBool InEnableValidation,
	OaU64  InPickRating,
	OaU32  InEnumerationIndex,
	void*  InSurface,
	OaBool InHintNeedsPresentation)
{
	VkInstance       instance = static_cast<VkInstance>(InInstance);
	VkPhysicalDevice bestPhys = static_cast<VkPhysicalDevice>(InPhysicalDevice);
	VkSurfaceKHR     surface  = static_cast<VkSurfaceKHR>(InSurface);

	// Use DeviceBuilder with all features for backward compatibility
	OaVkDeviceBuilder builder;
	builder.WithAllFeatures();

	// Build base device directly (no derived class complications).
	// Thread InHintNeedsPresentation through so the queue planner picks a
	// graphics-capable family when the caller intends to attach a surface
	// later (Step 3c.5 fix — previously the hint dropped here, leaving
	// Queues.GraphicsQueue unset and breaking OaGraphicsEngine::
	// InitPresentation under the new ctx-mediated path).
	auto deviceResult = builder.BuildBase(
		instance, bestPhys, InEnableValidation, InHintNeedsPresentation);
	if (!deviceResult.IsOk()) {
		return deviceResult.GetStatus();
	}

	OaVkDevice dev = std::move(deviceResult.GetValue());

	// Fill in legacy fields that DeviceBuilder doesn't set
	dev.Info.Hardware.PickRating = InPickRating;
	dev.Info.Hardware.EnumerationIndex = InEnumerationIndex;
	dev.Type = dev.Info.Hardware.DeviceType;
	dev.Index = 0;

	return dev;
}


// ------------------------------------------------------------
// OaVkDevice::Create  — PATCHED (added InHintNeedsPresentation)
// ------------------------------------------------------------
OaResult<OaVkDevice> OaVkDevice::Create(
	OaStringView               InAppName,
	OaBool                     InEnableValidation,
	OaDeviceType               InPreferred,
	OaU32                      InForceEnumerationIndex,
	OaU32                      InAppVersionPatch,
	OaSpan<const char* const>  InInstanceExtraExtensions,
	OaBool                     InHintNeedsPresentation
) {
	auto instRes = OaVkInstance::CreateInstance(
		InAppName, InAppVersionPatch, InEnableValidation, InInstanceExtraExtensions);
	if (!instRes.IsOk()) return OaResult<OaVkDevice>(instRes.GetStatus());

	VkInstance instance = std::move(instRes).GetValue();
	OaVkLoadInstance(instance);

	OaU32 devCount = 0;
	vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
	if (devCount == 0) {
		OaVkInstance::DestroyInstance(instance);
		return OaStatus::Error(OaStatusCode::DeviceNotFound, "no Vulkan physical devices");
	}

	OaVec<VkPhysicalDevice> physDevices(devCount);
	vkEnumeratePhysicalDevices(instance, &devCount, physDevices.Data());

	OaVkLogPhysicalDeviceSurvey(
		devCount,
		reinterpret_cast<void* const*>(physDevices.Data()),
		InPreferred
	);

	OaU32            bestIdx   = 0;
	VkPhysicalDevice bestPhys  = physDevices[0];
	OaU64            bestScore = 0;

	if (InForceEnumerationIndex != OaVkEnumerationIndexUnset) {
		if (InForceEnumerationIndex >= devCount) {
			OaVkInstance::DestroyInstance(instance);
			return OaStatus::Error(OaStatusCode::DeviceNotFound,
				"vulkan_index out of range (use survey log indices)");
		}
		bestIdx   = InForceEnumerationIndex;
		bestPhys  = physDevices[bestIdx];
		bestScore = OaVkPhysicalDeviceRate(bestPhys, InPreferred);
		if (OaDefaultDeviceInfoLogMode() == OaDeviceInfoLogMode::Full) {
			OA_LOG_INFO(OaLogComponent::Core,
				"Vulkan: using forced enumeration index %u",
				static_cast<unsigned>(bestIdx));
		}
	} else {
		bestScore = OaVkPhysicalDeviceRate(bestPhys, InPreferred);
		for (OaU32 i = 1; i < devCount; ++i) {
			const OaU64 score = OaVkPhysicalDeviceRate(physDevices[i], InPreferred);
			if (score > bestScore) {
				bestPhys  = physDevices[i];
				bestScore = score;
				bestIdx   = i;
			}
		}
	}

	VkPhysicalDeviceProperties pickProps{};
	vkGetPhysicalDeviceProperties(bestPhys, &pickProps);
	if (OaDefaultDeviceInfoLogMode() == OaDeviceInfoLogMode::Full) {
		OA_LOG_INFO(OaLogComponent::Core,
			"Vulkan: selected physical device [%u] %s (PickScore=%s)",
			static_cast<unsigned>(bestIdx),
			pickProps.deviceName,
			OaFormatNumberU64(bestScore).c_str()
		);
	}

	// Pass InHintNeedsPresentation; InSurface is null here (surface comes after instance).
	auto result = CreateFromPhysical(
		instance, bestPhys, InEnableValidation,
		bestScore, bestIdx,
		/*InSurface=*/nullptr,
		InHintNeedsPresentation
	);   // <-- threaded through

	if (!result.IsOk()) {
		OaVkInstance::DestroyInstance(instance);
		return OaResult<OaVkDevice>(result.GetStatus());
	}

	auto dev         = std::move(result.GetValue());
	dev.OwnsInstance = true;
	OaVkLoadDevice(static_cast<VkDevice>(dev.Device));
	return dev;
}


void OaVkDevice::SetMeshLogicalIndex(OaI32 InMeshIndex) noexcept {
	Index = InMeshIndex;
	Type = Info.Hardware.DeviceType;
}


void OaVkDevice::Destroy() {
	if (Device) {
		vkDestroyDevice(static_cast<VkDevice>(Device), nullptr);
		Device = nullptr;
	}
	if (Instance && OwnsInstance) {
		OaVkInstance::DestroyInstance(static_cast<VkInstance>(Instance));
	}
	Instance = nullptr;
	PhysicalDevice = nullptr;
	Queues.ComputeQueue = nullptr;
	Queues.AsyncComputeQueue = nullptr;
	Queues.TransferQueue = nullptr;
	Queues.GraphicsQueue = nullptr;
	Queues.PresentQueue = nullptr;
	Queues.GraphicsQueueFamily = OaVkEnumerationIndexUnset;
	Queues.PresentQueueFamily = OaVkEnumerationIndexUnset;
	Queues.HasPresentation = false;
	Type = OaDeviceType::Host;
	Index = 0;
}


void OaVkDevice::PrintInfo() const {
	switch (OaDefaultDeviceInfoLogMode()) {
		case OaDeviceInfoLogMode::Full:
			PrintInfoDetailed();
			break;
		case OaDeviceInfoLogMode::Compact:
			PrintInfoCompact();
			break;
		case OaDeviceInfoLogMode::Off:
			break;
	}
}

void OaVkDevice::PrintInfoCompact() const {
	const auto& hw = Info.Hardware;
	const auto& sw = Info.Software;
	const OaU32 index = hw.EnumerationIndex != OaVkEnumerationIndexUnset
		? hw.EnumerationIndex
		: static_cast<OaU32>(Index < 0 ? 0 : Index);
	const OaString vram = OaVkFormatCapacityBytesHuman(hw.VramBytes);
	OA_LOG_INFO(OaLogComponent::Core,
		"OaComputeDevice (%u): %s, Vulkan %s, %s",
		static_cast<unsigned>(index),
		hw.DeviceName.c_str(),
		sw.ApiVersion.c_str(),
		vram.c_str());
	// One-line video-decode capability, even in compact mode, so an operator immediately
	// sees why hardware decode is unavailable. Distinguish the two failure shapes and add
	// the actionable Intel hint: Mesa ANV only exposes Vulkan-Video (extensions AND a
	// VK_QUEUE_VIDEO_DECODE_BIT_KHR queue) under the xe KMD — under i915 the hardware VCS
	// engines exist but nothing is advertised. See PrintInfoDetailed for the full breakdown.
	if (!Queues.HasVideoDecodeQueue) {
		const bool isIntel = (hw.VendorId == 0x8086u);
		const char* intelHint = isIntel
			? " (Intel: hardware decode requires the xe kernel driver; i915 exposes no Vulkan video)"
			: "";
		if (sw.HasVideoDecodeQueue) {
			OA_LOG_WARN(OaLogComponent::Core,
				"OaComputeDevice (%u): video decode NOT usable — extensions advertised but no "
				"VK_QUEUE_VIDEO_DECODE_BIT_KHR queue%s.",
				static_cast<unsigned>(index), intelHint);
		} else {
			OA_LOG_INFO(OaLogComponent::Core,
				"OaComputeDevice (%u): video decode not supported (no VK_KHR_video_decode_queue)%s.",
				static_cast<unsigned>(index), intelHint);
		}
	}
}

void OaVkDevice::PrintInfoDetailed() const {
	const auto& hw = Info.Hardware;
	const auto& sw = Info.Software;
	const char* typeName = OaVkPhysicalTypeLabel(
		hw.DeviceType == OaDeviceType::VkDiscrete ? VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU :
		hw.DeviceType == OaDeviceType::VkIntegrated ? VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU :
		VK_PHYSICAL_DEVICE_TYPE_CPU
	);

	OA_LOG_INFO(OaLogComponent::Core, "OaVkDevice(");
	if (hw.EnumerationIndex != OaVkEnumerationIndexUnset) {
		OA_LOG_INFO(OaLogComponent::Core, "  DeviceIndex:    %u", hw.EnumerationIndex);
	} else {
		OA_LOG_INFO(OaLogComponent::Core, "  DeviceIndex:    (unset)");
	}
	OA_LOG_INFO(OaLogComponent::Core, "  Vendor:         %s", hw.VendorName.c_str());
	OA_LOG_INFO(OaLogComponent::Core, "  Device:         %s", hw.DeviceName.c_str());
	OA_LOG_INFO(OaLogComponent::Core, "  DeviceType:     %s", typeName);
	OA_LOG_INFO(OaLogComponent::Core, "  VendorId:       %s (%u)",
		OaFormatHexU32(hw.VendorId).c_str(), static_cast<unsigned>(hw.VendorId)
	);
	OA_LOG_INFO(OaLogComponent::Core, "  DeviceId:       %s (%u)",
		OaFormatHexU32(hw.DeviceId).c_str(), static_cast<unsigned>(hw.DeviceId)
	);
	OA_LOG_INFO(OaLogComponent::Core, "  Vram:           %s",
		OaVkFormatCapacityBytesHuman(hw.VramBytes).c_str()
	);
	OA_LOG_INFO(OaLogComponent::Core, "  ApiVersion:     Vulkan %s", sw.ApiVersion.c_str());
	// VendorDriverVersion = VkPhysicalDeviceProperties::driverVersion (vendor U32, shown M.N.P).
	// DriverInfoString = VkPhysicalDeviceDriverProperties::driverInfo (separate vendor metadata).
	OA_LOG_INFO(OaLogComponent::Core, "  Driver(");
	OA_LOG_INFO(OaLogComponent::Core, "    Provider:              %s", sw.DriverName.c_str());
	OA_LOG_INFO(OaLogComponent::Core, "    VendorDriverVersion:   %s", sw.DriverVersion.c_str());
	if (!sw.DriverInfo.empty()) {
		OA_LOG_INFO(OaLogComponent::Core, "    DriverInfoString:      %s", sw.DriverInfo.c_str());
	}
	OA_LOG_INFO(OaLogComponent::Core, "    DriverId:              %s (%s)",
		OaFormatHexU32(sw.DriverId).c_str(), OaVkDriverIdLabel(sw.DriverId)
	);
	OA_LOG_INFO(OaLogComponent::Core, "  )");
	// OA_LOG_INFO(OaLogComponent::Core, "  PickScore:       %s (sort key for default GPU; higher=better)",
	OA_LOG_INFO(OaLogComponent::Core, "  PickScore:       %s", OaFormatNumberU64(hw.PickRating).c_str());
	OA_LOG_INFO(OaLogComponent::Core, "  Queues(");
	OA_LOG_INFO(OaLogComponent::Core, "    ComputeSlots: %u", Queues.ComputeQueueSlotCount);
	OA_LOG_INFO(OaLogComponent::Core, "    TransferSlots: %u", Queues.DedicatedTransferQueueSlotCount);
	if (Queues.HasVideoDecodeQueue) {
		OA_LOG_INFO(OaLogComponent::Core,
			"    VideoDecode: enabled (family=%u codecOps=0x%08X)",
			static_cast<unsigned>(Queues.VideoDecodeQueueFamily),
			static_cast<unsigned>(Queues.VideoDecodeCodecOps));
	}
	if (Queues.HasVideoEncodeQueue) {
		OA_LOG_INFO(OaLogComponent::Core,
			"    VideoEncode: enabled (family=%u codecOps=0x%08X)",
			static_cast<unsigned>(Queues.VideoEncodeQueueFamily),
			static_cast<unsigned>(Queues.VideoEncodeCodecOps));
	}
	if (Queues.HasPresentation) {
		OA_LOG_INFO(OaLogComponent::Core,
			"    Presentation: enabled (graphicsFamily=%u presentFamily=%u)",
			static_cast<unsigned>(Queues.GraphicsQueueFamily),
			static_cast<unsigned>(Queues.PresentQueueFamily));
	} else {
		OA_LOG_INFO(OaLogComponent::Core, "    Presentation: disabled");
	}
	OA_LOG_INFO(OaLogComponent::Core, "  )");
	OA_LOG_INFO(OaLogComponent::Core, "  ThroughputEstimate(");
	OA_LOG_INFO(OaLogComponent::Core, "    MemoryBandwidth: %s",
		OaVkFormatMemoryBandwidthHuman(hw.EstMemBandwidthGbps).c_str()
	);
	OA_LOG_INFO(OaLogComponent::Core, "    PeakPerformance(");
	OA_LOG_INFO(OaLogComponent::Core, "      Float32:  %s",
		OaVkFormatPeakTflopsHuman(hw.EstPeakTflopsF32).c_str()
	);
	if (sw.ShaderBfloat16ExtensionEnabled && sw.ShaderBfloat16TypeEnabled) {
		const OaF64 estBf16Tflops = hw.EstPeakTflopsF32 * 2.0;
		OA_LOG_INFO(OaLogComponent::Core, "      BFloat16: %s",
			OaVkFormatPeakTflopsHuman(estBf16Tflops).c_str()
		);
	}
	OA_LOG_INFO(OaLogComponent::Core, "    )");
	OA_LOG_INFO(OaLogComponent::Core, "  )");
	OA_LOG_INFO(OaLogComponent::Core, "  ComputeLimits(");
	OA_LOG_INFO(OaLogComponent::Core, "    SubgroupSize: %u", hw.SubgroupSize);
	OA_LOG_INFO(
		OaLogComponent::Core,
		"    MaxComputeWorkGroupSize: %u (MaxInvocations: %u)",
		hw.MaxComputeWorkGroupSize, hw.MaxComputeWorkGroupInvocations
	);
	OA_LOG_INFO(OaLogComponent::Core, "  )");
	OA_LOG_INFO(OaLogComponent::Core, "  DescriptorLimits(");
	OA_LOG_INFO(OaLogComponent::Core, "    maxPerStageDescriptorUpdateAfterBindStorageBuffers: %u",
		hw.MaxPerStageDescriptorUpdateAfterBindStorageBuffers);
	OA_LOG_INFO(OaLogComponent::Core, "    maxPerStageDescriptorUpdateAfterBindSampledImages: %u",
		hw.MaxPerStageDescriptorUpdateAfterBindSampledImages);
	OA_LOG_INFO(OaLogComponent::Core, "    maxPerStageDescriptorUpdateAfterBindSamplers: %u",
		hw.MaxPerStageDescriptorUpdateAfterBindSamplers);
	OA_LOG_INFO(OaLogComponent::Core, "  )");
	OA_LOG_INFO(OaLogComponent::Core, "  BindlessCapacity(");
	OA_LOG_INFO(OaLogComponent::Core, "    StorageBuffers: %u%s", OA_BINDLESS_CAPACITY,
		OA_BINDLESS_CAPACITY < hw.MaxPerStageDescriptorUpdateAfterBindStorageBuffers ? " (capped)" : "");
	OA_LOG_INFO(OaLogComponent::Core, "    SampledImages: %u%s", OA_BINDLESS_IMAGE_CAPACITY,
		OA_BINDLESS_IMAGE_CAPACITY < hw.MaxPerStageDescriptorUpdateAfterBindSampledImages ? " (capped)" : "");
	OA_LOG_INFO(OaLogComponent::Core, "    Samplers: %u%s", OA_BINDLESS_SAMPLER_CAPACITY,
		OA_BINDLESS_SAMPLER_CAPACITY < hw.MaxPerStageDescriptorUpdateAfterBindSamplers ? " (capped)" : "");
	OA_LOG_INFO(OaLogComponent::Core, "  )");
	OA_LOG_INFO(OaLogComponent::Core, "  Capabilities(");

	// Core/Infrastructure extensions
	OaString coreExts;
	for (const auto& ext : sw.EnabledDeviceExtensions) {
		if (ext == "VK_KHR_pipeline_library" ||
		    ext == "VK_KHR_maintenance5" ||
		    ext == "VK_KHR_buffer_device_address" ||
		    ext == "VK_EXT_device_generated_commands") {
			if (!coreExts.empty()) coreExts += ", ";
			coreExts += ext;
		}
	}
	if (!coreExts.empty()) {
		OA_LOG_INFO(OaLogComponent::Core, "    Core: %s", coreExts.c_str());
	}

	// Memory extensions
	if (hw.HasSAM) {
		OA_LOG_INFO(OaLogComponent::Core, "    SmartAccessMemory: VK_KHR_external_memory, VK_KHR_external_memory_fd");
	}

	// Shader precision capabilities (non-GEMM related)
	if (sw.Has16BitStorage) {
		OA_LOG_INFO(OaLogComponent::Core, "    StorageBuffer16Bit: VK_KHR_16bit_storage");
	}
	if (sw.ShaderFloat16Enabled) {
		OA_LOG_INFO(OaLogComponent::Core, "    ShaderFloat16: VK_KHR_shader_float16_int8");
	}
	// Note: bf16 Type and int8 DotProduct are shown in Precision summary below
	if (sw.ShaderBfloat16DotProductEnabled) {
		OA_LOG_INFO(OaLogComponent::Core, "    ShaderBfloat16DotProduct: VK_KHR_shader_bfloat16");
	}

	// Cooperative matrix capabilities
	if (sw.HasCooperativeMatrix) {
		OA_LOG_INFO(OaLogComponent::Core, "    CooperativeMatrix: VK_KHR_cooperative_matrix");
		if (OaEnvFlag::IsSet("OA_LOG_COOPMAT_SHAPES")) {
			OaVkLogCoopMatShapes(sw.CoopMatShapes, "      ");
		}
	}
	if (sw.HasCooperativeMatrix2) {
		OA_LOG_INFO(OaLogComponent::Core, "    CooperativeMatrix2: VK_NV_cooperative_matrix2");
	}
	if (sw.HasCooperativeVector) {
		OA_LOG_INFO(OaLogComponent::Core, "    CooperativeVector: VK_NV_cooperative_vector");
	}
	if (sw.HasCooperativeMatrixDecodeVector) {
		OA_LOG_INFO(OaLogComponent::Core, "    CooperativeMatrixDecodeVector: VK_NV_cooperative_matrix_decode_vector");
	}

	// Video decode capability. Report the ACTUAL usable state, not just extension
	// advertisement: a driver can advertise VK_KHR_video_decode_queue while exposing
	// no queue family with VK_QUEUE_VIDEO_DECODE_BIT_KHR (e.g. Intel TGL/ADL under the
	// i915 KMD — hardware VCS engines exist but ANV only wires Vulkan-Video queues under
	// the xe KMD). Without such a queue, vkCmdDecodeVideoKHR cannot be submitted anywhere,
	// so decode is unavailable regardless of the advertised extension. ffmpeg's Vulkan
	// hwaccel fails the same way ("Device does not support the VK_KHR_video_decode_queue
	// extension!") and falls back to software.
	if (Queues.HasVideoDecodeQueue) {
		OaString decodeExts = "VK_KHR_video_queue, VK_KHR_video_decode_queue";
		if (sw.HasVideoDecodeH264) decodeExts += ", VK_KHR_video_decode_h264";
		if (sw.HasVideoDecodeH265) decodeExts += ", VK_KHR_video_decode_h265";
		if (sw.HasVideoDecodeAV1) decodeExts += ", VK_KHR_video_decode_av1";
		if (sw.HasVideoDecodeVP9) decodeExts += ", VK_KHR_video_decode_vp9";
		OA_LOG_INFO(OaLogComponent::Core, "    VideoDecode: %s", decodeExts.c_str());
	} else if (sw.HasVideoDecodeQueue) {
		OA_LOG_WARN(OaLogComponent::Core,
			"    VideoDecode: NOT SUPPORTED on this device — video-decode extensions are "
			"advertised but no queue family exposes VK_QUEUE_VIDEO_DECODE_BIT_KHR "
			"(on Intel this needs the xe kernel driver; i915 exposes no Vulkan video queue). "
			"Hardware video decode is unavailable; falling back to non-accelerated paths.");
	} else {
		OA_LOG_INFO(OaLogComponent::Core,
			"    VideoDecode: not supported on this device (no VK_KHR_video_decode_queue).");
	}

	// Video encode capability — same extension-vs-queue distinction as decode above.
	if (Queues.HasVideoEncodeQueue) {
		OaString encodeExts = "VK_KHR_video_queue, VK_KHR_video_encode_queue";
		if (sw.HasVideoEncodeH264) encodeExts += ", VK_KHR_video_encode_h264";
		if (sw.HasVideoEncodeH265) encodeExts += ", VK_KHR_video_encode_h265";
		if (sw.HasVideoEncodeAV1) encodeExts += ", VK_KHR_video_encode_av1";
		OA_LOG_INFO(OaLogComponent::Core, "    VideoEncode: %s", encodeExts.c_str());
	} else if (sw.HasVideoEncodeQueue) {
		OA_LOG_WARN(OaLogComponent::Core,
			"    VideoEncode: NOT SUPPORTED on this device — video-encode extensions are "
			"advertised but no queue family exposes VK_QUEUE_VIDEO_ENCODE_BIT_KHR "
			"(on Intel this needs the xe kernel driver; i915 exposes no Vulkan video queue).");
	} else {
		OA_LOG_INFO(OaLogComponent::Core,
			"    VideoEncode: not supported on this device (no VK_KHR_video_encode_queue).");
	}

	// Hardware color conversion (YCbCr for video decode/encode)
	if (sw.HasSamplerYcbcrConversion) {
		OA_LOG_INFO(OaLogComponent::Core, "    HardwareColorConversion: VK_KHR_sampler_ycbcr_conversion");
	}

	// Swapchain for presentation
	for (const auto& ext : sw.EnabledDeviceExtensions) {
		if (ext == "VK_KHR_swapchain") {
			OA_LOG_INFO(OaLogComponent::Core, "    Swapchain: VK_KHR_swapchain");
			break;
		}
	}

	// Precision summary with extensions - shows what GEMM routing will use
	// Order: fp32 (baseline), bf16, fp16, int8
	OaString precisions = "fp32";
	if (sw.ShaderBfloat16TypeEnabled) {
		precisions += ", bf16 (VK_KHR_shader_bfloat16, VK_KHR_16bit_storage)";
	}
	if (sw.ShaderFloat16Enabled) {
		precisions += ", fp16 (VK_KHR_shader_float16_int8)";
	}
	if (sw.ShaderIntegerDotProductEnabled) {
		precisions += ", int8 (VK_KHR_shader_integer_dot_product)";
	}
	OA_LOG_INFO(OaLogComponent::Core, "    Precision: %s", precisions.c_str());

	OA_LOG_INFO(OaLogComponent::Core, "  )");

	OA_LOG_INFO(OaLogComponent::Core, ")");
}


void OaVkDevice::LogShaderPrecisionCaps() const {
	const auto& sw = Info.Software;
	OA_LOG_INFO(OaLogComponent::Core,
		"Shader precision caps: 16b_storage=%u shader_float16=%u khr_shader_bfloat16=%u "
		"(type=%u dot=%u coopmat=%u) int_dot_product=%u",
		sw.Has16BitStorage ? 1u : 0u,
		sw.ShaderFloat16Enabled ? 1u : 0u,
		sw.ShaderBfloat16ExtensionEnabled ? 1u : 0u,
		sw.ShaderBfloat16TypeEnabled ? 1u : 0u,
		sw.ShaderBfloat16DotProductEnabled ? 1u : 0u,
		sw.ShaderBfloat16CooperativeMatrixEnabled ? 1u : 0u,
		sw.ShaderIntegerDotProductEnabled ? 1u : 0u
	);
}
