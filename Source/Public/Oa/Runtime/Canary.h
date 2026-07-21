// OA Runtime — deterministic device-admission canary.
//
// This is sampled evidence, not proof that arbitrary hardware arithmetic is
// correct. Expected values are computed independently on the host; a mismatch
// fails closed with DATA_LOSS and leaves a structured report for attribution.

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>

class OaEngine;

class OaDeviceCanaryCheck {
public:
	OaString Name;
	OaBool Passed = false;
	OaBool Exact = false;
	OaU32 SampleCount = 0;
	OaU64 ExpectedHash = 0;
	OaU64 ActualHash = 0;
	OaF64 MaxAbsoluteError = 0.0;
	OaF64 Tolerance = 0.0;
};

class OaDeviceCanaryReport {
public:
	OaString DeviceName;
	OaString VendorName;
	OaString DriverName;
	OaString DriverVersion;
	OaString ApiVersion;
	OaVec<OaDeviceCanaryCheck> Checks;

	[[nodiscard]] OaBool Passed() const noexcept;
	[[nodiscard]] OaString DebugReportJson() const;
};

class OaDeviceCanary {
public:
	// Requires an otherwise idle engine context. On arithmetic or transport
	// disagreement OutReport is populated and DATA_LOSS is returned. Vulkan,
	// allocation, and readback failures preserve their original status code.
	[[nodiscard]] static OaStatus Run(
		OaEngine& InEngine,
		OaDeviceCanaryReport& OutReport);
};
