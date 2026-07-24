// Physical Vulkan device heuristics: local VRAM, pick score, bandwidth/TFLOPS estimates,
// combined rating, and enumeration survey logging. Used before OaVkDevice exists.

#pragma once

#include <Oa/Core/Device.h>
#include <Oa/Core/Types.h>

void OaVkLogPhysicalDeviceSurvey(
	OaU32 InCount,
	void* const* InPhysicalDevices,
	OaDeviceType InPreferred
);

[[nodiscard]] OaU64 OaVkPhysicalDeviceLocalHeapBytes(void* InPhysicalDevice);

[[nodiscard]] OaU64 OaVkPhysicalDevicePickScore(
	OaDeviceType InType,
	OaU64 InLocalHeapBytes,
	OaDeviceType InPreferred,
	OaU32 InDeviceId
);

[[nodiscard]] OaF64 OaVkEstimateMemBandwidthGbpsEx(
	OaU32 InVendorId,
	OaU32 InDeviceId,
	OaDeviceType InType,
	OaU64 InLocalHeapBytes
);

[[nodiscard]] OaF64 OaVkEstimatePeakTflopsF32Ex(
	OaU32 InVendorId,
	OaU32 InDeviceId,
	OaDeviceType InType,
	OaU64 InLocalHeapBytes
);

[[nodiscard]] OaU32 OaVkEstimateNumSMsEx(
	OaU32 InVendorId,
	OaU32 InDeviceId,
	OaDeviceType InType,
	OaU64 InLocalHeapBytes
);

[[nodiscard]] OaF64 OaVkEstimateMemBandwidthGbps(
	OaU32 InVendorId,
	OaDeviceType InType,
	OaU64 InLocalHeapBytes
);

[[nodiscard]] OaF64 OaVkEstimatePeakTflopsF32(
	OaU32 InVendorId,
	OaDeviceType InType,
	OaU64 InLocalHeapBytes
);

[[nodiscard]] OaU64 OaVkPhysicalDeviceRate(void* InPhysicalDevice, OaDeviceType InPreferred);
