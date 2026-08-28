// Private physical vulkan device heuristics: local VRAM, pick score, bandwidth/TFLOPS estimates,
// combined rating, and enumeration survey logging. Used before oavk::Device exists.

#pragma once

#include <oa/core/device.h>
#include <oa/core/types.h>

struct VklInstanceTable;

namespace oavk {

void logPhysicalDeviceSurvey(
	const VklInstanceTable& inDispatch,
	oa::U32 inCount,
	void* const* inPhysicalDevices,
	oa::DeviceType inPreferred
);

[[nodiscard]] oa::U64 physicalDeviceLocalHeapBytes(
	const VklInstanceTable& inDispatch,
	void* inPhysicalDevice);

[[nodiscard]] oa::U64 physicalDevicePickScore(
	oa::DeviceType inType,
	oa::U64 inLocalHeapBytes,
	oa::DeviceType inPreferred,
	oa::U32 inDeviceId
);

[[nodiscard]] oa::F64 estimateMemBandwidthGbpsForDevice(
	oa::U32 inVendorId,
	oa::U32 inDeviceId,
	oa::DeviceType inType,
	oa::U64 inLocalHeapBytes
);

[[nodiscard]] oa::F64 estimatePeakTflopsF32ForDevice(
	oa::U32 inVendorId,
	oa::U32 inDeviceId,
	oa::DeviceType inType,
	oa::U64 inLocalHeapBytes
);

[[nodiscard]] oa::U32 estimateNumSmsForDevice(
	oa::U32 inVendorId,
	oa::U32 inDeviceId,
	oa::DeviceType inType,
	oa::U64 inLocalHeapBytes
);

[[nodiscard]] oa::F64 estimateMemBandwidthGbps(
	oa::U32 inVendorId,
	oa::DeviceType inType,
	oa::U64 inLocalHeapBytes
);

[[nodiscard]] oa::F64 estimatePeakTflopsF32(
	oa::U32 inVendorId,
	oa::DeviceType inType,
	oa::U64 inLocalHeapBytes
);

[[nodiscard]] oa::U64 physicalDeviceRate(
	const VklInstanceTable& inDispatch,
	void* inPhysicalDevice,
	oa::DeviceType inPreferred
);

} // namespace oavk
