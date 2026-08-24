#pragma once

#include <oa/core/types.h>

namespace oa {

// Host clock paired with the vulkan device timestamp domain. Host timestamps
// are nanoseconds in the corresponding POSIX clock domain; device timestamps
// are ticks and must be scaled by deviceNanosecondsPerTick.
enum class HostClockDomain : oa::U8 {
	Monotonic,
	MonotonicRaw,
};

// One quasi-simultaneous host/device clock sample. maximumDeviationNanoseconds
// is the vulkan implementation's uncertainty bound for the pair, not an OA
// estimate. Consumers should recalibrate long-running traces periodically.
class ClockCalibration {
public:
	oa::U64 deviceTimestampTicks = 0;
	oa::U64 hostTimestampNanoseconds = 0;
	oa::U64 maximumDeviationNanoseconds = 0;
	oa::F64 deviceNanosecondsPerTick = 0.0;
	oa::U32 deviceTimestampValidBits = 0;
	oa::HostClockDomain hostClockDomain = oa::HostClockDomain::Monotonic;

	// map a query-pool timestamp into the calibrated host clock. The signed
	// modular delta handles vulkan timestamp wrap for queues exposing fewer
	// than 64 valid bits. values farther than half a device-clock period from
	// this sample are inherently ambiguous and require a newer calibration.
	[[nodiscard]] oa::F64 hostNanosecondsForDeviceTimestamp(
		oa::U64 inDeviceTimestampTicks) const noexcept
	{
		if (deviceTimestampValidBits == 0U
			or deviceTimestampValidBits > 64U
			or deviceNanosecondsPerTick <= 0.0)
		{
			return 0.0;
		}

		oa::I64 signedDelta = 0;
		if (deviceTimestampValidBits == 64U) {
			signedDelta = static_cast<oa::I64>(
				inDeviceTimestampTicks - deviceTimestampTicks);
		} else {
			const oa::U64 modulus = oa::U64{1} << deviceTimestampValidBits;
			const oa::U64 mask = modulus - 1U;
			const oa::U64 delta =
				(inDeviceTimestampTicks - deviceTimestampTicks) & mask;
			signedDelta = delta >= (modulus >> 1U)
				? -static_cast<oa::I64>(modulus - delta)
				: static_cast<oa::I64>(delta);
		}
		return static_cast<oa::F64>(hostTimestampNanoseconds)
			+ static_cast<oa::F64>(signedDelta) * deviceNanosecondsPerTick;
	}
};

} // namespace oa
