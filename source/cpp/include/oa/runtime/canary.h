// OA Runtime — deterministic device-admission canary.
//
// This is sampled evidence, not proof that arbitrary hardware arithmetic is
// correct. expected values are computed independently on the host; a mismatch
// fails closed with DATA_LOSS and leaves a structured report for attribution.

#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>

namespace oa {

class Engine;

class DeviceCanaryCheck {
public:
	oa::String name;
	oa::Bool passed = false;
	oa::Bool exact = false;
	oa::U32 sampleCount = 0;
	oa::U64 expectedHash = 0;
	oa::U64 actualHash = 0;
	oa::F64 maxAbsoluteError = 0.0;
	oa::F64 tolerance = 0.0;
};

class DeviceCanaryReport {
public:
	oa::String deviceName;
	oa::String vendorName;
	oa::String driverName;
	oa::String driverVersion;
	oa::String apiVersion;
	oa::Vector<DeviceCanaryCheck> checks;

	[[nodiscard]] oa::Bool passed() const noexcept;
	[[nodiscard]] oa::String debugReportJson() const;
};

class DeviceCanary {
public:
	// Requires an otherwise idle engine context. On arithmetic or transport
	// disagreement outReport is populated and DATA_LOSS is returned. vulkan,
	// allocation, and readback failures preserve their original status code.
	[[nodiscard]] static oa::Status run(
		oa::Engine& inEngine,
		DeviceCanaryReport& outReport);
};

} // namespace oa
