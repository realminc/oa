#include <oa/core/log.h>
#include <oa/core/envFlag.h>
#include <oa/runtime/device.h>
#include <vkl/vkl.h>
#include <oa/runtime/instance.h>
#include <oa/runtime/bindless.h>
#include "deviceBuilder.h"

#include <stdio.h>


// VRAM: MiB below 1 GiB; otherwise whole GB / TB / PB (binary base, "GB" label like retail GPUs).
static oa::String formatCapacityBytesHuman(oa::U64 inBytes) {
	char buf[72];
	const oa::U64 kib = 1024ULL;
	const oa::U64 mib = kib * kib;
	const oa::U64 gib = mib * kib;
	const oa::U64 tib = gib * kib;
	const oa::U64 pib = tib * kib;
	if (inBytes < gib) {
		oa::U64 mibVal = (inBytes + mib / 2) / mib;
		if (mibVal == 0) {
			mibVal = 1;
		}
		::snprintf(
			buf, sizeof(buf), "%llu MiB", static_cast<unsigned long long>(mibVal));
		return oa::String(buf);
	}
	if (inBytes < tib) {
		const double gigabytes =
			static_cast<double>(inBytes) / static_cast<double>(gib);
		::snprintf(buf, sizeof(buf), "%.0f GB", oa::round(gigabytes));
		return oa::String(buf);
	}
	if (inBytes < pib) {
		const double terabytes =
			static_cast<double>(inBytes) / static_cast<double>(tib);
		::snprintf(buf, sizeof(buf), "%.2f TB", terabytes);
		return oa::String(buf);
	}
	const double petabytes = static_cast<double>(inBytes) / static_cast<double>(pib);
	::snprintf(buf, sizeof(buf), "%.2f PB", petabytes);
	return oa::String(buf);
}


// estMemBandwidthGbps is gigabytes per second (see device.h).
static oa::String formatMemoryBandwidthHuman(oa::F64 inGigabytesPerSecond) {
	char buf[72];
	if (inGigabytesPerSecond <= 0.0) {
		return oa::String("0 MB/s");
	}
	if (inGigabytesPerSecond < 1.0) {
		const double mibPerSec = inGigabytesPerSecond * 1024.0;
		::snprintf(buf, sizeof(buf), "%.0f MB/s", mibPerSec);
		return oa::String(buf);
	}
	if (inGigabytesPerSecond < 1024.0) {
		::snprintf(buf, sizeof(buf), "%.0f GB/s", inGigabytesPerSecond);
		return oa::String(buf);
	}
	::snprintf(buf, sizeof(buf), "%.2f TB/s", inGigabytesPerSecond / 1024.0);
	return oa::String(buf);
}


static oa::String formatPeakTflopsHuman(oa::F64 inTflops) {
	char buf[72];
	if (inTflops <= 0.0) {
		return oa::String("0 TFLOPS");
	}
	if (inTflops < 1000.0) {
		::snprintf(buf, sizeof(buf), "%.2f TFLOPS", inTflops);
		return oa::String(buf);
	}
	if (inTflops < 1.0e6) {
		::snprintf(buf, sizeof(buf), "%.2f PFLOPS", inTflops / 1000.0);
		return oa::String(buf);
	}
	::snprintf(buf, sizeof(buf), "%.2f EFLOPS", inTflops / 1.0e6);
	return oa::String(buf);
}

enum class DeviceInfoLogMode : oa::U8 {
	Compact,
	Full,
	Off,
};

static bool asciiEqualsIgnoreCase(const char* inA, const char* inB) {
	if (inA == nullptr || inB == nullptr) {
		return inA == inB;
	}
	while (*inA != '\0' && *inB != '\0') {
		const auto lowerAscii = [](char inValue) noexcept {
			return inValue >= 'A' && inValue <= 'Z'
				? static_cast<char>(inValue + ('a' - 'A'))
				: inValue;
		};
		const char a = lowerAscii(*inA);
		const char b = lowerAscii(*inB);
		if (a != b) {
			return false;
		}
		++inA;
		++inB;
	}
	return *inA == '\0' && *inB == '\0';
}

static DeviceInfoLogMode defaultDeviceInfoLogMode() {
	const oa::String env = oa::EnvFlag::getString("OA_LOG_DEVICE_INIT", "");
	if (!env.empty()) {
		const char* value = env.cStr();
		if (asciiEqualsIgnoreCase(value, "full") ||
		    asciiEqualsIgnoreCase(value, "debug") ||
		    asciiEqualsIgnoreCase(value, "verbose") ||
		    asciiEqualsIgnoreCase(value, "1") ||
		    asciiEqualsIgnoreCase(value, "true") ||
		    asciiEqualsIgnoreCase(value, "on")) {
			return DeviceInfoLogMode::Full;
		}
		if (asciiEqualsIgnoreCase(value, "compact") ||
		    asciiEqualsIgnoreCase(value, "minimal") ||
		    asciiEqualsIgnoreCase(value, "min")) {
			return DeviceInfoLogMode::Compact;
		}
		if (asciiEqualsIgnoreCase(value, "off") ||
		    asciiEqualsIgnoreCase(value, "none") ||
		    asciiEqualsIgnoreCase(value, "0") ||
		    asciiEqualsIgnoreCase(value, "false") ||
		    asciiEqualsIgnoreCase(value, "no")) {
			return DeviceInfoLogMode::Off;
		}
	}
#ifdef NDEBUG
	return DeviceInfoLogMode::Compact;
#else
	return DeviceInfoLogMode::Full;
#endif
}


oa::U64 oavk::physicalDeviceLocalHeapBytes(
	const VklInstanceTable& inDispatch,
	void* inPhysicalDevice)
{
	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(inPhysicalDevice);
	VkPhysicalDeviceMemoryProperties mp{};
	inDispatch.vkGetPhysicalDeviceMemoryProperties(phys, &mp);
	oa::U64 sum = 0;
	for (oa::U32 i = 0; i < mp.memoryHeapCount; ++i) {
		if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
			sum += mp.memoryHeaps[i].size;
		}
	}
	return sum;
}


void oavk::logPhysicalDeviceSurvey(
	const VklInstanceTable& inDispatch,
	oa::U32 inCount,
	void* const* inPhysicalDevices,
	oa::DeviceType inPreferred)
{
	if (!inPhysicalDevices || inCount == 0) {
		return;
	}
	if (defaultDeviceInfoLogMode() != DeviceInfoLogMode::Full) {
		return;
	}
	OaLogInfo(oa::LogComponent::Runtime,
		"vulkan: enumerating physical devices (found %u)",
		static_cast<unsigned>(inCount));
	for (oa::U32 i = 0; i < inCount; ++i) {
		VkPhysicalDevice phys =
			static_cast<VkPhysicalDevice>(inPhysicalDevices[i]);
		if (!phys) {
			continue;
		}
		VkPhysicalDeviceProperties props{};
		inDispatch.vkGetPhysicalDeviceProperties(phys, &props);
		const oa::U64 localBytes = oavk::physicalDeviceLocalHeapBytes(inDispatch, phys);
		const oa::U64 rating = oavk::physicalDeviceRate(inDispatch, phys, inPreferred);
		const oa::U32 slots = oavk::countComputeQueueSlots(inDispatch, phys);
		const oa::DeviceType oaType = oavk::mapPhysicalType(props.deviceType);
		const oa::F64 bw = oavk::estimateMemBandwidthGbpsForDevice(
			props.vendorID, props.deviceID, oaType, localBytes);
		const oa::F64 tflops = oavk::estimatePeakTflopsF32ForDevice(
			props.vendorID, props.deviceID, oaType, localBytes);
		OaLogInfo(oa::LogComponent::Runtime,
			"  [%u] %s | PickScore=%s | vram=%s MiB | compute_queue_slots=%u | est_bw=%.0f GB/s | est_fp32=%.1f TFLOPS | %s",
			static_cast<unsigned>(i),
			props.deviceName,
			oa::formatNumberU64(rating).cStr(),
			oa::formatNumberU64(localBytes / (1024ull * 1024ull)).cStr(),
			static_cast<unsigned>(slots),
			bw,
			tflops,
			oavk::physicalTypeLabel(props.deviceType)
		);
	}
}

// extension probing and feature detection now handled by DeviceBuilder + feature modules
// ------------------------------------------------------------
// oavk::planDeviceQueues — shared by the device builders
// ------------------------------------------------------------
oa::Status oavk::planDeviceQueues(
	const VklInstanceTable& inDispatch,
	VkPhysicalDevice  inPhys,
	VkSurfaceKHR      inSurface,
	oavk::QueuePlan&    outPlan,
	bool              inNeedsGraphics)
{
	oa::U32 qfCount = 0;
	inDispatch.vkGetPhysicalDeviceQueueFamilyProperties(inPhys, &qfCount, nullptr);
	oa::Vector<VkQueueFamilyProperties> qfProps(qfCount);
	inDispatch.vkGetPhysicalDeviceQueueFamilyProperties(
		inPhys, &qfCount, qfProps.data());

	outPlan.priorityBacking.clear();
	outPlan.queueCIs.clear();
	outPlan.graphicsQF             = UINT32_MAX;
	outPlan.presentQF              = UINT32_MAX;
	outPlan.wantsGraphics          = false;
	outPlan.computeSlots           = 0;
	outPlan.dedicatedTransferSlots = 0;

	for (oa::U32 idx = 0; idx < qfCount; ++idx) {
		const VkQueueFlags qflags = qfProps[idx].queueFlags;
		if (qflags & VK_QUEUE_COMPUTE_BIT)
			outPlan.computeSlots += qfProps[idx].queueCount;
		if ((qflags & VK_QUEUE_TRANSFER_BIT) && !(qflags & VK_QUEUE_COMPUTE_BIT))
			outPlan.dedicatedTransferSlots += qfProps[idx].queueCount;
	}

	const bool wantSurface  = (inSurface != VK_NULL_HANDLE);
	const bool wantGraphics = wantSurface or inNeedsGraphics;

	oa::Vector<VkBool32> presentSupport;
	if (wantSurface) {
		if (!inDispatch.vkGetPhysicalDeviceSurfaceSupportKHR) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"vkGetPhysicalDeviceSurfaceSupportKHR unavailable (load instance WSI)");
		}
		presentSupport.resize(qfCount);
		for (oa::U32 idx = 0; idx < qfCount; ++idx) {
			VkResult sr = inDispatch.vkGetPhysicalDeviceSurfaceSupportKHR(
				inPhys, idx, inSurface, &presentSupport[idx]);
			if (sr != VK_SUCCESS) presentSupport[idx] = VK_FALSE;
		}
	}

	outPlan.computeQF          = UINT32_MAX;
	outPlan.asyncComputeQF     = UINT32_MAX;
	outPlan.transferQF         = UINT32_MAX;
	outPlan.computeHasMultiQueue = false;

	if (wantGraphics) {
		if (wantSurface) {
			// Original surface path: need graphics + present on the same family.
			oa::U32 graphicsPresent = UINT32_MAX;
			for (oa::U32 idx = 0; idx < qfCount; ++idx) {
				const VkQueueFlags fl = qfProps[idx].queueFlags;
				if ((fl & VK_QUEUE_GRAPHICS_BIT) && presentSupport[idx] == VK_TRUE) {
					graphicsPresent = idx;
					break;
				}
			}
			if (graphicsPresent == UINT32_MAX) {
				return oa::Status::error(
					oa::StatusCode::DeviceNotFound,
					"no queue family with graphics + surface present support");
			}
			outPlan.graphicsQF = graphicsPresent;
			outPlan.presentQF  = graphicsPresent;
		} else {
			// Hint-only path: find any graphics-capable family (no surface check).
			// Present support is verified later in oa::Presenter::initPresentation.
			oa::U32 gfxFamily = UINT32_MAX;
			for (oa::U32 idx = 0; idx < qfCount; ++idx) {
				if (qfProps[idx].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
					gfxFamily = idx;
					break;
				}
			}
			if (gfxFamily == UINT32_MAX) {
				return oa::Status::error(
					oa::StatusCode::DeviceNotFound,
					"graphics requested but device has no VK_QUEUE_GRAPHICS_BIT family");
			}
			outPlan.graphicsQF = gfxFamily;
		}

		outPlan.wantsGraphics = true;

		if (qfProps[outPlan.graphicsQF].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			outPlan.computeQF = outPlan.graphicsQF;
			if (qfProps[outPlan.graphicsQF].queueCount >= 2)
				outPlan.computeHasMultiQueue = true;
		} else {
			for (oa::U32 idx = 0; idx < qfCount; ++idx) {
				if (qfProps[idx].queueFlags & VK_QUEUE_COMPUTE_BIT) {
					outPlan.computeQF = idx;
					if (qfProps[idx].queueCount >= 2)
						outPlan.computeHasMultiQueue = true;
					break;
				}
			}
			if (outPlan.computeQF == UINT32_MAX)
				return oa::Status::error(oa::StatusCode::DeviceNotFound, "no compute queue family");
		}
	} else {
		// Pure compute path (original).
		for (oa::U32 idx = 0; idx < qfCount; idx++) {
			if ((qfProps[idx].queueFlags & VK_QUEUE_COMPUTE_BIT) && outPlan.computeQF == UINT32_MAX) {
				outPlan.computeQF = idx;
				if (qfProps[idx].queueCount >= 2)
					outPlan.computeHasMultiQueue = true;
			}
			if ((qfProps[idx].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
				!(qfProps[idx].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
				idx != outPlan.computeQF && outPlan.asyncComputeQF == UINT32_MAX)
			{
				outPlan.asyncComputeQF = idx;
			}
			if ((qfProps[idx].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
				!(qfProps[idx].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
				outPlan.transferQF == UINT32_MAX)
			{
				outPlan.transferQF = idx;
			}
		}
	}

	if (outPlan.computeQF == UINT32_MAX)
		return oa::Status::error(oa::StatusCode::DeviceNotFound, "no compute queue family");

	// Async + dedicated-transfer scan (same for both graphics and compute paths).
	for (oa::U32 idx = 0; idx < qfCount; idx++) {
		if ((qfProps[idx].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
			!(qfProps[idx].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
			idx != outPlan.computeQF && outPlan.asyncComputeQF == UINT32_MAX)
		{
			outPlan.asyncComputeQF = idx;
		}
		if ((qfProps[idx].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
			!(qfProps[idx].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
			outPlan.transferQF == UINT32_MAX)
		{
			outPlan.transferQF = idx;
		}
	}

	if (outPlan.transferQF == UINT32_MAX)
		outPlan.transferQF = outPlan.computeQF;

	// --- Video queue selection with per-family codec cross-check (gap 1) ---
	// A family can advertise VK_QUEUE_VIDEO_DECODE_BIT_KHR but not support a
	// particular codec (e.g. AV1). Chain VkQueueFamilyVideoPropertiesKHR to
	// query videoCodecOperations per family and pick the broadest. store the
	// ops on the plan so decoder create() can verify codec support.
	{
		oa::U32 qf2Count = qfCount;
		oa::Vector<VkQueueFamilyProperties2> qfProps2(qf2Count);
		oa::Vector<VkQueueFamilyVideoPropertiesKHR> videoProps(qf2Count);
		for (oa::U32 idx = 0; idx < qf2Count; ++idx) {
			videoProps[idx] = {};
			videoProps[idx].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR;
			videoProps[idx].pNext = nullptr;
			videoProps[idx].videoCodecOperations = 0;
			qfProps2[idx] = {};
			qfProps2[idx].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
			qfProps2[idx].pNext = &videoProps[idx];
		}
		const bool hasQueueFamilyProperties2 =
			inDispatch.vkGetPhysicalDeviceQueueFamilyProperties2 != nullptr;
		if (hasQueueFamilyProperties2) {
			inDispatch.vkGetPhysicalDeviceQueueFamilyProperties2(
				inPhys, &qf2Count, qfProps2.data());
		}

		// Pick video decode family: prefer the family with the broadest codec
		// coverage. Ties broken by first-match (stable behavior).
		oa::U32 bestDecodeQF = UINT32_MAX;
		oa::U32 bestDecodePopcount = 0;
		oa::U32 bestEncodeQF = UINT32_MAX;
		oa::U32 bestEncodePopcount = 0;
		for (oa::U32 idx = 0; idx < qf2Count; ++idx) {
			const VkQueueFlags qflags = hasQueueFamilyProperties2
				? qfProps2[idx].queueFamilyProperties.queueFlags
				: qfProps[idx].queueFlags;
			if ((qflags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) != 0) {
				const auto ops = videoProps[idx].videoCodecOperations;
				auto popcount = static_cast<oa::U32>(__builtin_popcount(
					static_cast<unsigned>(ops)));
				if (bestDecodeQF == UINT32_MAX || popcount > bestDecodePopcount) {
					bestDecodeQF = idx;
					bestDecodePopcount = popcount;
					outPlan.videoDecodeCodecOps = ops;
				}
			}
			if ((qflags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR) != 0) {
				const auto ops = videoProps[idx].videoCodecOperations;
				auto popcount = static_cast<oa::U32>(__builtin_popcount(
					static_cast<unsigned>(ops)));
				if (bestEncodeQF == UINT32_MAX || popcount > bestEncodePopcount) {
					bestEncodeQF = idx;
					bestEncodePopcount = popcount;
					outPlan.videoEncodeCodecOps = ops;
				}
			}
		}
		outPlan.videoDecodeQF = bestDecodeQF;
		outPlan.videoEncodeQF = bestEncodeQF;
	}

	outPlan.hasAsync         = (outPlan.asyncComputeQF != UINT32_MAX) || outPlan.computeHasMultiQueue;
	outPlan.mainComputeCount = (outPlan.computeHasMultiQueue && outPlan.asyncComputeQF == UINT32_MAX) ? 2 : 1;

	oa::Vector<oa::U32> need;
	need.resize(qfCount);
	for (oa::U32 idx = 0; idx < qfCount; ++idx) need[idx] = 0;

	auto bump = [&](oa::U32 fam, oa::U32 cnt) {
		if (fam == UINT32_MAX || fam >= qfCount || cnt == 0) return;
		const oa::U32 cap  = qfProps[fam].queueCount;
		const oa::U32 want = cnt > cap ? cap : cnt;
		if (want > need[fam]) need[fam] = want;
	};

	bump(outPlan.computeQF, outPlan.mainComputeCount);
	if (outPlan.asyncComputeQF != UINT32_MAX && outPlan.asyncComputeQF != outPlan.computeQF)
		bump(outPlan.asyncComputeQF, 1);
	if (outPlan.transferQF != outPlan.computeQF && outPlan.transferQF != outPlan.asyncComputeQF)
		bump(outPlan.transferQF, 1);
	if (outPlan.wantsGraphics and outPlan.graphicsQF != UINT32_MAX and
		outPlan.graphicsQF != outPlan.computeQF)
	{
		bump(outPlan.graphicsQF, 1);
	}
	if (outPlan.videoDecodeQF != UINT32_MAX)
		bump(outPlan.videoDecodeQF, 1);
	if (outPlan.videoEncodeQF != UINT32_MAX)
		bump(outPlan.videoEncodeQF, 1);

	for (oa::U32 fi = 0; fi < qfCount; ++fi) {
		if (need[fi] == 0) continue;
		const oa::U32 baseIdx = outPlan.priorityBacking.size();
		for (oa::U32 q = 0; q < need[fi]; ++q) {
			oa::F32 pri = 1.0f;
			if (fi == outPlan.computeQF && need[fi] >= 2 && q > 0) pri = 0.5f;
			outPlan.priorityBacking.pushBack(pri);
		}
		VkDeviceQueueCreateInfo ci{};
		ci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		ci.queueFamilyIndex = fi;
		ci.queueCount       = need[fi];
		ci.pQueuePriorities = outPlan.priorityBacking.data() + baseIdx;
		outPlan.queueCIs.pushBack(ci);
	}

	if (outPlan.queueCIs.empty())
		return oa::Status::error(oa::StatusCode::DeviceNotFound, "queue create info list empty");

	return oa::Status::ok();
}

// Feature querying, refinement, and extension collection now handled by feature modules
// (CoreFeatures, MlFeatures, VisionFeatures, AudioFeatures, RenderFeatures)
// Old monolithic functions removed - see feature modules for new implementation

// ------------------------------------------------------------
// oavk::Device::CreateFromPhysical — Refactored to use DeviceBuilder
// ------------------------------------------------------------
oa::Result<oavk::Device> oavk::Device::createFromPhysical(
	void*  inInstance,
	void*  inPhysicalDevice,
	oa::Bool inEnableValidation,
	oa::U64  inPickRating,
	oa::U32  inEnumerationIndex,
	void*  inSurface,
	oa::Bool inHintNeedsPresentation,
	oa::Bool inHintNeedsGraphics)
{
	VkInstance       instance = static_cast<VkInstance>(inInstance);
	VkPhysicalDevice bestPhys = static_cast<VkPhysicalDevice>(inPhysicalDevice);
	(void)inSurface;

	// Use DeviceBuilder with all features for backward compatibility
	oavk::DeviceBuilder builder;
	builder.withAllFeatures();

	// Build the base device directly. The two hints intentionally separate WSI
	// extension admission from graphics-queue selection. The legacy inSurface
	// path remains handled by BuildRender; making this base path surface-aware
	// is tracked separately because it requires a live-surface conformance test.
	auto deviceResult = builder.buildBase(
		instance, bestPhys, inEnableValidation,
		inHintNeedsPresentation, inHintNeedsGraphics);
	if (!deviceResult.isOk()) {
		return deviceResult.getStatus();
	}

	oavk::Device dev = oa::move(deviceResult.getValue());

	// Fill in legacy fields that DeviceBuilder doesn't set
	dev.info.hardware.pickRating = inPickRating;
	dev.info.hardware.enumerationIndex = inEnumerationIndex;
	dev.type = dev.info.hardware.deviceType;
	dev.index = 0;

	return dev;
}


// ------------------------------------------------------------
// oavk::Device::Create
// ------------------------------------------------------------
oa::Result<oavk::Device> oavk::Device::create(
	oa::StringView               inAppName,
	oa::Bool                     inEnableValidation,
	oa::DeviceType               inPreferred,
	oa::U32                      inForceEnumerationIndex,
	oa::U32                      inAppVersionPatch,
	oa::Span<const char* const>  inInstanceExtraExtensions,
	oa::Bool                     inHintNeedsPresentation,
	oa::Bool                     inHintNeedsGraphics,
	PFN_vkGetInstanceProcAddr    inCustomLoader
) {
	auto instRes = oavk::Instance::createInstance(
		inAppName, inAppVersionPatch, inEnableValidation,
		inInstanceExtraExtensions, inHintNeedsPresentation, inCustomLoader);
	if (!instRes.isOk()) return oa::Result<oavk::Device>(instRes.getStatus());

	VkInstance instance = oa::move(instRes).getValue();
	VklInstanceTable instanceDispatch{};
	vklLoadInstanceTable(&instanceDispatch, instance);
	if (instanceDispatch.vkEnumeratePhysicalDevices == nullptr) {
		oavk::Instance::destroyInstance(instanceDispatch, instance);
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"vulkan instance dispatch is incomplete");
	}

	oa::U32 devCount = 0;
	instanceDispatch.vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
	if (devCount == 0) {
		oavk::Instance::destroyInstance(instanceDispatch, instance);
		return oa::Status::error(oa::StatusCode::DeviceNotFound, "no vulkan physical devices");
	}

	oa::Vector<VkPhysicalDevice> physDevices(devCount);
	instanceDispatch.vkEnumeratePhysicalDevices(
		instance, &devCount, physDevices.data());

	oavk::logPhysicalDeviceSurvey(
		instanceDispatch,
		devCount,
		reinterpret_cast<void* const*>(physDevices.data()),
		inPreferred
	);

	oa::U32            bestIdx   = 0;
	VkPhysicalDevice bestPhys  = physDevices[0];
	oa::U64            bestScore = 0;

	if (inForceEnumerationIndex != oavk::EnumerationIndexUnset) {
		if (inForceEnumerationIndex >= devCount) {
			oavk::Instance::destroyInstance(instanceDispatch, instance);
			return oa::Status::error(oa::StatusCode::DeviceNotFound,
				"vulkan_index out of range (use survey log indices)");
		}
		bestIdx   = inForceEnumerationIndex;
		bestPhys  = physDevices[bestIdx];
		bestScore = oavk::physicalDeviceRate(
			instanceDispatch, bestPhys, inPreferred);
		if (defaultDeviceInfoLogMode() == DeviceInfoLogMode::Full) {
			OaLogInfo(oa::LogComponent::Runtime,
				"vulkan: using forced enumeration index %u",
				static_cast<unsigned>(bestIdx));
		}
	} else {
		bestScore = oavk::physicalDeviceRate(
			instanceDispatch, bestPhys, inPreferred);
		for (oa::U32 i = 1; i < devCount; ++i) {
			const oa::U64 score = oavk::physicalDeviceRate(
				instanceDispatch, physDevices[i], inPreferred);
			if (score > bestScore) {
				bestPhys  = physDevices[i];
				bestScore = score;
				bestIdx   = i;
			}
		}
	}

	VkPhysicalDeviceProperties pickProps{};
	instanceDispatch.vkGetPhysicalDeviceProperties(bestPhys, &pickProps);
	if (defaultDeviceInfoLogMode() == DeviceInfoLogMode::Full) {
		OaLogInfo(oa::LogComponent::Runtime,
			"vulkan: selected physical device [%u] %s (PickScore=%s)",
			static_cast<unsigned>(bestIdx),
			pickProps.deviceName,
			oa::formatNumberU64(bestScore).cStr()
		);
	}

	// inSurface is null here because platform surface creation follows instance
	// creation. Presentation and headless-graphics intent remain distinct.
	auto result = createFromPhysical(
		instance, bestPhys, inEnableValidation,
		bestScore, bestIdx,
		/*inSurface=*/nullptr,
		inHintNeedsPresentation,
		inHintNeedsGraphics
	);

	if (!result.isOk()) {
		oavk::Instance::destroyInstance(instanceDispatch, instance);
		return oa::Result<oavk::Device>(result.getStatus());
	}

	auto dev         = oa::move(result.getValue());
	dev.ownsInstance = true;
	return dev;
}


void oavk::Device::destroy() {
	if (device) {
		deviceDispatch.vkDestroyDevice(static_cast<VkDevice>(device), nullptr);
		device = nullptr;
	}
	if (instance && ownsInstance) {
		oavk::Instance::destroyInstance(
			instanceDispatch, static_cast<VkInstance>(instance));
	}
	instance = nullptr;
	physicalDevice = nullptr;
	queues.computeQueue = nullptr;
	queues.asyncComputeQueue = nullptr;
	queues.transferQueue = nullptr;
	queues.graphicsQueue = nullptr;
	queues.presentQueue = nullptr;
	queues.graphicsQueueFamily = oavk::EnumerationIndexUnset;
	queues.presentQueueFamily = oavk::EnumerationIndexUnset;
	queues.hasPresentation = false;
	type = oa::DeviceType::Host;
	index = 0;
	instanceDispatch = {};
	deviceDispatch = {};
}


void oavk::Device::printInfo() const {
	switch (defaultDeviceInfoLogMode()) {
		case DeviceInfoLogMode::Full:
			printInfoDetailed();
			break;
		case DeviceInfoLogMode::Compact:
			printInfoCompact();
			break;
		case DeviceInfoLogMode::Off:
			break;
	}
}

void oavk::Device::printInfoCompact() const {
	const auto& hw = info.hardware;
	const auto& sw = info.software;
	const oa::U32 selectedIndex = hw.enumerationIndex != oavk::EnumerationIndexUnset
		? hw.enumerationIndex
		: static_cast<oa::U32>(this->index < 0 ? 0 : this->index);
	const oa::String vram = formatCapacityBytesHuman(hw.vramBytes);
	OaLogInfo(oa::LogComponent::Runtime,
		"ComputeDevice (%u): %s, vulkan %s, %s",
		static_cast<unsigned>(selectedIndex),
		hw.deviceName.cStr(),
		sw.apiVersion.cStr(),
		vram.cStr());
	// One-line video-decode capability, even in compact mode, so an operator immediately
	// sees why hardware decode is unavailable. Distinguish the two failure shapes and add
	// the actionable Intel hint: Mesa ANV only exposes vulkan-Video (extensions AND a
	// VK_QUEUE_VIDEO_DECODE_BIT_KHR queue) under the xe KMD — under i915 the hardware VCS
	// engines exist but nothing is advertised. See printInfoDetailed for the full breakdown.
	if (!queues.hasVideoDecodeQueue) {
		const bool isIntel = (hw.vendorId == 0x8086u);
		const char* intelHint = isIntel
			? " (Intel: hardware decode requires the xe kernel driver; i915 exposes no vulkan video)"
			: "";
		if (sw.hasVideoDecodeQueue) {
			OaLogWarn(oa::LogComponent::Runtime,
				"ComputeDevice (%u): video decode NOT usable — extensions advertised but no "
				"VK_QUEUE_VIDEO_DECODE_BIT_KHR queue%s.",
				static_cast<unsigned>(index), intelHint);
		} else {
			OaLogInfo(oa::LogComponent::Runtime,
				"ComputeDevice (%u): video decode not supported (no VK_KHR_video_decode_queue)%s.",
				static_cast<unsigned>(index), intelHint);
		}
	}
}

void oavk::Device::printInfoDetailed() const {
	const auto& hw = info.hardware;
	const auto& sw = info.software;
	const char* typeName = oavk::physicalTypeLabel(
		hw.deviceType == oa::DeviceType::VkDiscrete ? VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU :
		hw.deviceType == oa::DeviceType::VkIntegrated ? VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU :
		VK_PHYSICAL_DEVICE_TYPE_CPU
	);

	OaLogInfo(oa::LogComponent::Runtime, "oavk::Device(");
	if (hw.enumerationIndex != oavk::EnumerationIndexUnset) {
		OaLogInfo(oa::LogComponent::Runtime, "  deviceIndex:    %u", hw.enumerationIndex);
	} else {
		OaLogInfo(oa::LogComponent::Runtime, "  deviceIndex:    (unset)");
	}
	OaLogInfo(oa::LogComponent::Runtime, "  Vendor:         %s", hw.vendorName.cStr());
	OaLogInfo(oa::LogComponent::Runtime, "  device:         %s", hw.deviceName.cStr());
	OaLogInfo(oa::LogComponent::Runtime, "  deviceType:     %s", typeName);
	OaLogInfo(oa::LogComponent::Runtime, "  vendorId:       %s (%u)",
		oa::formatHexU32(hw.vendorId).cStr(), static_cast<unsigned>(hw.vendorId)
	);
	OaLogInfo(oa::LogComponent::Runtime, "  deviceId:       %s (%u)",
		oa::formatHexU32(hw.deviceId).cStr(), static_cast<unsigned>(hw.deviceId)
	);
	OaLogInfo(oa::LogComponent::Runtime, "  Vram:           %s",
		formatCapacityBytesHuman(hw.vramBytes).cStr()
	);
	OaLogInfo(oa::LogComponent::Runtime, "  apiVersion:     vulkan %s", sw.apiVersion.cStr());
	// VendorDriverVersion = VkPhysicalDeviceProperties::driverVersion (vendor U32, shown M.n.p).
	// DriverInfoString = VkPhysicalDeviceDriverProperties::driverInfo (separate vendor metadata).
	OaLogInfo(oa::LogComponent::Runtime, "  driver(");
	OaLogInfo(oa::LogComponent::Runtime, "    Provider:              %s", sw.driverName.cStr());
	OaLogInfo(oa::LogComponent::Runtime, "    VendorDriverVersion:   %s", sw.driverVersion.cStr());
	if (!sw.driverInfo.empty()) {
		OaLogInfo(oa::LogComponent::Runtime, "    DriverInfoString:      %s", sw.driverInfo.cStr());
	}
	OaLogInfo(oa::LogComponent::Runtime, "    driverId:              %s (%s)",
		oa::formatHexU32(sw.driverId).cStr(), oavk::driverIdLabel(sw.driverId)
	);
	OaLogInfo(oa::LogComponent::Runtime, "  )");
	// OaLogInfo(oa::LogComponent::Runtime, "  PickScore:       %s (sort key for default GPU; higher=better)",
	OaLogInfo(oa::LogComponent::Runtime, "  PickScore:       %s", oa::formatNumberU64(hw.pickRating).cStr());
	OaLogInfo(oa::LogComponent::Runtime, "  queues(");
	OaLogInfo(oa::LogComponent::Runtime, "    computeSlots: %u", queues.computeQueueSlotCount);
	OaLogInfo(oa::LogComponent::Runtime, "    TransferSlots: %u", queues.dedicatedTransferQueueSlotCount);
	if (queues.hasVideoDecodeQueue) {
		OaLogInfo(oa::LogComponent::Runtime,
			"    VideoDecode: enabled (family=%u codecOps=0x%08X)",
			static_cast<unsigned>(queues.videoDecodeQueueFamily),
			static_cast<unsigned>(queues.videoDecodeCodecOps));
	}
	if (queues.hasVideoEncodeQueue) {
		OaLogInfo(oa::LogComponent::Runtime,
			"    VideoEncode: enabled (family=%u codecOps=0x%08X)",
			static_cast<unsigned>(queues.videoEncodeQueueFamily),
			static_cast<unsigned>(queues.videoEncodeCodecOps));
	}
	if (queues.hasPresentation) {
		OaLogInfo(oa::LogComponent::Runtime,
			"    Presentation: enabled (graphicsFamily=%u presentFamily=%u)",
			static_cast<unsigned>(queues.graphicsQueueFamily),
			static_cast<unsigned>(queues.presentQueueFamily));
	} else {
		OaLogInfo(oa::LogComponent::Runtime, "    Presentation: disabled");
	}
	OaLogInfo(oa::LogComponent::Runtime, "  )");
	OaLogInfo(oa::LogComponent::Runtime, "  throughputEstimate(");
	OaLogInfo(oa::LogComponent::Runtime, "    MemoryBandwidth: %s",
		formatMemoryBandwidthHuman(hw.estMemBandwidthGbps).cStr()
	);
	OaLogInfo(oa::LogComponent::Runtime, "    peakPerformance(");
	OaLogInfo(oa::LogComponent::Runtime, "      Float32:  %s",
		formatPeakTflopsHuman(hw.estPeakTflopsF32).cStr()
	);
	if (sw.shaderBfloat16ExtensionEnabled && sw.shaderBfloat16TypeEnabled) {
		const oa::F64 estBf16Tflops = hw.estPeakTflopsF32 * 2.0;
		OaLogInfo(oa::LogComponent::Runtime, "      BFloat16: %s",
			formatPeakTflopsHuman(estBf16Tflops).cStr()
		);
	}
	OaLogInfo(oa::LogComponent::Runtime, "    )");
	OaLogInfo(oa::LogComponent::Runtime, "  )");
	OaLogInfo(oa::LogComponent::Runtime, "  computeLimits(");
	OaLogInfo(oa::LogComponent::Runtime, "    subgroupSize: %u", hw.subgroupSize);
	OaLogInfo(
		oa::LogComponent::Runtime,
		"    MaxComputeWorkGroupSizeX: %u (MaxInvocations: %u)",
		hw.maxComputeWorkGroupSize, hw.maxComputeWorkGroupInvocations
	);
	OaLogInfo(oa::LogComponent::Runtime,
		"    maxComputeWorkGroupCount: [%u, %u, %u]",
		hw.maxComputeWorkGroupCountX,
		hw.maxComputeWorkGroupCountY,
		hw.maxComputeWorkGroupCountZ);
	OaLogInfo(oa::LogComponent::Runtime, "    MaxComputeSharedMemory: %u bytes",
		hw.maxComputeSharedMemoryBytes);
	OaLogInfo(oa::LogComponent::Runtime, "    maxStorageBufferRange: %s bytes",
		oa::formatNumberU64(hw.maxStorageBufferRangeBytes).cStr());
	OaLogInfo(oa::LogComponent::Runtime, "  )");
	OaLogInfo(oa::LogComponent::Runtime, "  descriptorLimits(");
	OaLogInfo(oa::LogComponent::Runtime, "    maxPerStageDescriptorUpdateAfterBindStorageBuffers: %u",
		hw.maxPerStageDescriptorUpdateAfterBindStorageBuffers);
	OaLogInfo(oa::LogComponent::Runtime, "    maxPerStageDescriptorUpdateAfterBindSampledImages: %u",
		hw.maxPerStageDescriptorUpdateAfterBindSampledImages);
	OaLogInfo(oa::LogComponent::Runtime, "    maxPerStageDescriptorUpdateAfterBindSamplers: %u",
		hw.maxPerStageDescriptorUpdateAfterBindSamplers);
	OaLogInfo(oa::LogComponent::Runtime, "  )");
	OaLogInfo(oa::LogComponent::Runtime, "  bindlessCapacity(");
	OaLogInfo(oa::LogComponent::Runtime, "    StorageBuffers: %u%s", hw.bindlessBufferCapacity,
		hw.bindlessBufferCapacity < hw.maxPerStageDescriptorUpdateAfterBindStorageBuffers ? " (capped)" : "");
	OaLogInfo(oa::LogComponent::Runtime, "    SampledImages: %u%s", hw.bindlessImageCapacity,
		hw.bindlessImageCapacity < hw.maxPerStageDescriptorUpdateAfterBindSampledImages ? " (capped)" : "");
	OaLogInfo(oa::LogComponent::Runtime, "    samplers: %u%s", hw.bindlessSamplerCapacity,
		hw.bindlessSamplerCapacity < hw.maxPerStageDescriptorUpdateAfterBindSamplers ? " (capped)" : "");
	OaLogInfo(oa::LogComponent::Runtime, "  )");
	OaLogInfo(oa::LogComponent::Runtime, "  capabilities(");

	// Core/Infrastructure extensions
	oa::String coreExts;
	for (const auto& ext : sw.enabledDeviceExtensions) {
		if (ext == "VK_KHR_pipeline_library" ||
		    ext == "VK_KHR_maintenance5" ||
		    ext == "VK_KHR_buffer_device_address" ||
		    ext == "VK_EXT_device_generated_commands") {
			if (!coreExts.empty()) coreExts += ", ";
			coreExts += ext;
		}
	}
	if (!coreExts.empty()) {
		OaLogInfo(oa::LogComponent::Runtime, "    Core: %s", coreExts.cStr());
	}

	// memory extensions
	if (hw.hasSAM) {
		OaLogInfo(oa::LogComponent::Runtime, "    SmartAccessMemory: VK_KHR_external_memory, VK_KHR_external_memory_fd");
	}

	// shader precision capabilities (non-GEMM related)
	if (sw.has16BitStorage) {
		OaLogInfo(oa::LogComponent::Runtime, "    StorageBuffer16Bit: VK_KHR_16bit_storage");
	}
	if (sw.shaderFloat16Enabled) {
		OaLogInfo(oa::LogComponent::Runtime, "    shaderFloat16: VK_KHR_shader_float16_int8");
	}
	if (sw.shaderFloat64Enabled) {
		OaLogInfo(oa::LogComponent::Runtime, "    shaderFloat64: vulkan core feature");
	}
	// Note: bf16 type and int8 DotProduct are shown in Precision summary below
	if (sw.shaderBfloat16DotProductEnabled) {
		OaLogInfo(oa::LogComponent::Runtime, "    ShaderBfloat16DotProduct: VK_KHR_shader_bfloat16");
	}

	// Cooperative matrix capabilities
	if (sw.hasCooperativeMatrix) {
		OaLogInfo(oa::LogComponent::Runtime, "    cooperativeMatrix: VK_KHR_cooperative_matrix");
		if (oa::EnvFlag::isSet("OA_LOG_COOPMAT_SHAPES")) {
			oavk::logCoopMatShapes(sw.coopMatShapes, "      ");
		}
	}
	if (sw.hasCooperativeMatrix2) {
		OaLogInfo(oa::LogComponent::Runtime, "    CooperativeMatrix2: VK_NV_cooperative_matrix2");
	}
	if (sw.hasCooperativeVector) {
		OaLogInfo(oa::LogComponent::Runtime, "    cooperativeVector: VK_NV_cooperative_vector");
	}
	if (sw.hasCooperativeMatrixDecodeVector) {
		OaLogInfo(oa::LogComponent::Runtime, "    cooperativeMatrixDecodeVector: VK_NV_cooperative_matrix_decode_vector");
	}

	// Video decode capability. Report the ACTUAL usable state, not just extension
	// advertisement: a driver can advertise VK_KHR_video_decode_queue while exposing
	// no queue family with VK_QUEUE_VIDEO_DECODE_BIT_KHR (e.g. Intel TGL/ADL under the
	// i915 KMD — hardware VCS engines exist but ANV only wires vulkan-Video queues under
	// the xe KMD). Without such a queue, vkCmdDecodeVideoKHR cannot be submitted anywhere,
	// so decode is unavailable regardless of the advertised extension. ffmpeg's vulkan
	// hwaccel fails the same way ("device does not support the VK_KHR_video_decode_queue
	// extension!") and falls back to software.
	if (queues.hasVideoDecodeQueue) {
		oa::String decodeExts = "VK_KHR_video_queue, VK_KHR_video_decode_queue";
		if (sw.hasVideoDecodeH264) decodeExts += ", VK_KHR_video_decode_h264";
		if (sw.hasVideoDecodeH265) decodeExts += ", VK_KHR_video_decode_h265";
		if (sw.hasVideoDecodeAV1) decodeExts += ", VK_KHR_video_decode_av1";
		if (sw.hasVideoDecodeVP9) decodeExts += ", VK_KHR_video_decode_vp9";
		OaLogInfo(oa::LogComponent::Runtime, "    VideoDecode: %s", decodeExts.cStr());
	} else if (sw.hasVideoDecodeQueue) {
		OaLogWarn(oa::LogComponent::Runtime,
			"    VideoDecode: NOT SUPPORTED on this device — video-decode extensions are "
			"advertised but no queue family exposes VK_QUEUE_VIDEO_DECODE_BIT_KHR "
			"(on Intel this needs the xe kernel driver; i915 exposes no vulkan video queue). "
			"hardware video decode is unavailable; falling back to non-accelerated paths.");
	} else {
		OaLogInfo(oa::LogComponent::Runtime,
			"    VideoDecode: not supported on this device (no VK_KHR_video_decode_queue).");
	}

	// Video encode capability — same extension-vs-queue distinction as decode above.
	if (queues.hasVideoEncodeQueue) {
		oa::String encodeExts = "VK_KHR_video_queue, VK_KHR_video_encode_queue";
		if (sw.hasVideoEncodeH264) encodeExts += ", VK_KHR_video_encode_h264";
		if (sw.hasVideoEncodeH265) encodeExts += ", VK_KHR_video_encode_h265";
		if (sw.hasVideoEncodeAV1) encodeExts += ", VK_KHR_video_encode_av1";
		OaLogInfo(oa::LogComponent::Runtime, "    VideoEncode: %s", encodeExts.cStr());
	} else if (sw.hasVideoEncodeQueue) {
		OaLogWarn(oa::LogComponent::Runtime,
			"    VideoEncode: NOT SUPPORTED on this device — video-encode extensions are "
			"advertised but no queue family exposes VK_QUEUE_VIDEO_ENCODE_BIT_KHR "
			"(on Intel this needs the xe kernel driver; i915 exposes no vulkan video queue).");
	} else {
		OaLogInfo(oa::LogComponent::Runtime,
			"    VideoEncode: not supported on this device (no VK_KHR_video_encode_queue).");
	}

	// hardware color conversion (YCbCr for video decode/encode)
	if (sw.hasSamplerYcbcrConversion) {
		OaLogInfo(oa::LogComponent::Runtime, "    HardwareColorConversion: VK_KHR_sampler_ycbcr_conversion");
	}

	// Swapchain for presentation
	for (const auto& ext : sw.enabledDeviceExtensions) {
		if (ext == "VK_KHR_swapchain") {
			OaLogInfo(oa::LogComponent::Runtime, "    Swapchain: VK_KHR_swapchain");
			break;
		}
	}

	// Precision summary with extensions - shows what GEMM routing will use
	// hardware vocabulary. Engine policies remain narrower and fail closed.
	oa::String precisions = "fp32";
	if (sw.shaderBfloat16TypeEnabled) {
		precisions += ", bf16 (VK_KHR_shader_bfloat16, VK_KHR_16bit_storage)";
	}
	if (sw.shaderFloat16Enabled) {
		precisions += ", fp16 (VK_KHR_shader_float16_int8)";
	}
	if (sw.shaderIntegerDotProductEnabled) {
		precisions += ", int8 (VK_KHR_shader_integer_dot_product)";
	}
	if (sw.shaderFloat64Enabled) {
		precisions += ", fp64 (shaderFloat64; kernels not yet admitted)";
	}
	OaLogInfo(oa::LogComponent::Runtime, "    Precision: %s", precisions.cStr());

	OaLogInfo(oa::LogComponent::Runtime, "  )");

	OaLogInfo(oa::LogComponent::Runtime, ")");
}


void oavk::Device::logShaderPrecisionCaps() const {
	const auto& sw = info.software;
	OaLogInfo(oa::LogComponent::Runtime,
		"shader precision caps: 16b_storage=%u shader_float16=%u shader_float64=%u khr_shader_bfloat16=%u "
		"(type=%u dot=%u coopmat=%u) int_dot_product=%u",
		sw.has16BitStorage ? 1u : 0u,
		sw.shaderFloat16Enabled ? 1u : 0u,
		sw.shaderFloat64Enabled ? 1u : 0u,
		sw.shaderBfloat16ExtensionEnabled ? 1u : 0u,
		sw.shaderBfloat16TypeEnabled ? 1u : 0u,
		sw.shaderBfloat16DotProductEnabled ? 1u : 0u,
		sw.shaderBfloat16CooperativeMatrixEnabled ? 1u : 0u,
		sw.shaderIntegerDotProductEnabled ? 1u : 0u
	);
}
