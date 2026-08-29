#pragma once

// OA-owned clock values. Platform clock APIs remain in the compiled core
// implementation; no hosted C++ time type or native handle crosses this
// boundary.

#define OA_TYPES_H_SKIP_REST
#include <oa/core/types.h>
#undef OA_TYPES_H_SKIP_REST

#include <oa/core/std/assert.h>
#include <oa/core/std/limits.h>

namespace oa {

namespace detail {

[[nodiscard]] constexpr oa::I64 checkedI64Multiply(
	oa::I64 inLeft,
	oa::I64 inRight,
	const char* inMessage
) noexcept {
	constexpr oa::I64 minimum = oa::Limits<oa::I64>::min();
	constexpr oa::I64 maximum = oa::Limits<oa::I64>::max();

	if (inLeft > 0) {
		if (inRight > 0) {
			OA_REQUIRE_MSG(inLeft <= maximum / inRight, inMessage);
		} else if (inRight < 0) {
			OA_REQUIRE_MSG(inRight >= minimum / inLeft, inMessage);
		}
	} else if (inLeft < 0) {
		if (inRight > 0) {
			OA_REQUIRE_MSG(inLeft >= minimum / inRight, inMessage);
		} else if (inRight < 0) {
			OA_REQUIRE_MSG(inLeft >= maximum / inRight, inMessage);
		}
	}
	return inLeft * inRight;
}

[[nodiscard]] constexpr oa::I64 checkedI64Add(
	oa::I64 inLeft,
	oa::I64 inRight,
	const char* inMessage
) noexcept {
	constexpr oa::I64 minimum = oa::Limits<oa::I64>::min();
	constexpr oa::I64 maximum = oa::Limits<oa::I64>::max();
	if (inRight > 0) {
		OA_REQUIRE_MSG(inLeft <= maximum - inRight, inMessage);
	} else if (inRight < 0) {
		OA_REQUIRE_MSG(inLeft >= minimum - inRight, inMessage);
	}
	return inLeft + inRight;
}

[[nodiscard]] constexpr oa::I64 checkedI64Subtract(
	oa::I64 inLeft,
	oa::I64 inRight,
	const char* inMessage
) noexcept {
	constexpr oa::I64 minimum = oa::Limits<oa::I64>::min();
	constexpr oa::I64 maximum = oa::Limits<oa::I64>::max();
	if (inRight > 0) {
		OA_REQUIRE_MSG(inLeft >= minimum + inRight, inMessage);
	} else if (inRight < 0) {
		OA_REQUIRE_MSG(inLeft <= maximum + inRight, inMessage);
	}
	return inLeft - inRight;
}

[[nodiscard]] constexpr oa::I64 checkedSecondsToNanoseconds(
	oa::F64 inSeconds,
	const char* inMessage
) noexcept {
	OA_REQUIRE_MSG(
		inSeconds == inSeconds
			and inSeconds >= oa::Limits<oa::F64>::lowest()
			and inSeconds <= oa::Limits<oa::F64>::max(),
		"time conversion requires a finite value"
	);
	const oa::F64 nanoseconds = inSeconds * 1'000'000'000.0;
	constexpr oa::F64 minimum = -0x1.0p63;
	constexpr oa::F64 maximumExclusive = 0x1.0p63;
	OA_REQUIRE_MSG(
		nanoseconds >= minimum and nanoseconds < maximumExclusive,
		inMessage);
	return static_cast<oa::I64>(nanoseconds);
}

} // namespace detail

class Duration {
public:
	constexpr Duration() noexcept = default;
	constexpr explicit Duration(oa::I64 inNanoseconds) noexcept
		: nanoseconds_(inNanoseconds) {}

	[[nodiscard]] static constexpr Duration fromSeconds(oa::I64 inSeconds) noexcept {
		return Duration(oa::detail::checkedI64Multiply(
			inSeconds, 1'000'000'000LL, "Duration seconds overflow"));
	}
	[[nodiscard]] static constexpr Duration fromMilliseconds(oa::I64 inMilliseconds) noexcept {
		return Duration(oa::detail::checkedI64Multiply(
			inMilliseconds, 1'000'000LL, "Duration milliseconds overflow"));
	}
	[[nodiscard]] static constexpr Duration fromMicroseconds(oa::I64 inMicroseconds) noexcept {
		return Duration(oa::detail::checkedI64Multiply(
			inMicroseconds, 1'000LL, "Duration microseconds overflow"));
	}
	[[nodiscard]] static constexpr Duration fromNanoseconds(oa::I64 inNanoseconds) noexcept {
		return Duration(inNanoseconds);
	}
	[[nodiscard]] static constexpr Duration fromDouble(oa::F64 inSeconds) noexcept {
		return Duration(oa::detail::checkedSecondsToNanoseconds(
			inSeconds, "Duration floating-point seconds overflow"));
	}

	[[nodiscard]] constexpr oa::I64 nanoseconds() const noexcept { return nanoseconds_; }
	[[nodiscard]] constexpr oa::I64 microseconds() const noexcept { return nanoseconds_ / 1'000LL; }
	[[nodiscard]] constexpr oa::I64 milliseconds() const noexcept { return nanoseconds_ / 1'000'000LL; }
	[[nodiscard]] constexpr oa::F64 toSeconds() const noexcept {
		return static_cast<oa::F64>(nanoseconds_) / 1'000'000'000.0;
	}
	[[nodiscard]] constexpr oa::F64 toMilliseconds() const noexcept {
		return static_cast<oa::F64>(nanoseconds_) / 1'000'000.0;
	}
	[[nodiscard]] constexpr oa::F64 toMicroseconds() const noexcept {
		return static_cast<oa::F64>(nanoseconds_) / 1'000.0;
	}

	[[nodiscard]] constexpr Duration operator+(Duration inOther) const noexcept {
		return Duration(oa::detail::checkedI64Add(
			nanoseconds_, inOther.nanoseconds_, "Duration addition overflow"));
	}
	[[nodiscard]] constexpr Duration operator-(Duration inOther) const noexcept {
		return Duration(oa::detail::checkedI64Subtract(
			nanoseconds_, inOther.nanoseconds_, "Duration subtraction overflow"));
	}
	constexpr Duration& operator+=(Duration inOther) noexcept {
		nanoseconds_ = oa::detail::checkedI64Add(
			nanoseconds_, inOther.nanoseconds_, "Duration addition overflow");
		return *this;
	}
	constexpr Duration& operator-=(Duration inOther) noexcept {
		nanoseconds_ = oa::detail::checkedI64Subtract(
			nanoseconds_, inOther.nanoseconds_, "Duration subtraction overflow");
		return *this;
	}

	[[nodiscard]] constexpr bool operator==(Duration inOther) const noexcept {
		return nanoseconds_ == inOther.nanoseconds_;
	}
	[[nodiscard]] constexpr bool operator<(Duration inOther) const noexcept {
		return nanoseconds_ < inOther.nanoseconds_;
	}
	[[nodiscard]] constexpr bool operator<=(Duration inOther) const noexcept {
		return nanoseconds_ <= inOther.nanoseconds_;
	}
	[[nodiscard]] constexpr bool operator>(Duration inOther) const noexcept {
		return nanoseconds_ > inOther.nanoseconds_;
	}
	[[nodiscard]] constexpr bool operator>=(Duration inOther) const noexcept {
		return nanoseconds_ >= inOther.nanoseconds_;
	}

private:
	oa::I64 nanoseconds_ = 0;
};

class SteadyTimePoint {
public:
	constexpr SteadyTimePoint() noexcept = default;
	constexpr explicit SteadyTimePoint(oa::I64 inNanoseconds) noexcept
		: nanoseconds_(inNanoseconds) {}

	[[nodiscard]] constexpr oa::I64 nanosecondsSinceEpoch() const noexcept {
		return nanoseconds_;
	}
	[[nodiscard]] constexpr Duration operator-(SteadyTimePoint inOther) const noexcept {
		return Duration::fromNanoseconds(oa::detail::checkedI64Subtract(
			nanoseconds_, inOther.nanoseconds_, "SteadyTimePoint difference overflow"));
	}
	[[nodiscard]] constexpr SteadyTimePoint operator+(Duration inDuration) const noexcept {
		return SteadyTimePoint(oa::detail::checkedI64Add(
			nanoseconds_, inDuration.nanoseconds(), "SteadyTimePoint addition overflow"));
	}
	[[nodiscard]] constexpr SteadyTimePoint operator-(Duration inDuration) const noexcept {
		return SteadyTimePoint(oa::detail::checkedI64Subtract(
			nanoseconds_, inDuration.nanoseconds(), "SteadyTimePoint subtraction overflow"));
	}
	constexpr SteadyTimePoint& operator+=(Duration inDuration) noexcept {
		nanoseconds_ = oa::detail::checkedI64Add(
			nanoseconds_, inDuration.nanoseconds(), "SteadyTimePoint addition overflow");
		return *this;
	}
	constexpr SteadyTimePoint& operator-=(Duration inDuration) noexcept {
		nanoseconds_ = oa::detail::checkedI64Subtract(
			nanoseconds_, inDuration.nanoseconds(), "SteadyTimePoint subtraction overflow");
		return *this;
	}

	[[nodiscard]] constexpr bool operator==(SteadyTimePoint inOther) const noexcept {
		return nanoseconds_ == inOther.nanoseconds_;
	}
	[[nodiscard]] constexpr bool operator<(SteadyTimePoint inOther) const noexcept {
		return nanoseconds_ < inOther.nanoseconds_;
	}
	[[nodiscard]] constexpr bool operator<=(SteadyTimePoint inOther) const noexcept {
		return nanoseconds_ <= inOther.nanoseconds_;
	}
	[[nodiscard]] constexpr bool operator>(SteadyTimePoint inOther) const noexcept {
		return nanoseconds_ > inOther.nanoseconds_;
	}
	[[nodiscard]] constexpr bool operator>=(SteadyTimePoint inOther) const noexcept {
		return nanoseconds_ >= inOther.nanoseconds_;
	}

private:
	oa::I64 nanoseconds_ = 0;
};

class SystemTimePoint {
public:
	constexpr SystemTimePoint() noexcept = default;
	constexpr explicit SystemTimePoint(oa::I64 inNanoseconds) noexcept
		: nanoseconds_(inNanoseconds) {}

	[[nodiscard]] constexpr oa::I64 nanosecondsSinceEpoch() const noexcept {
		return nanoseconds_;
	}
	[[nodiscard]] constexpr Duration operator-(SystemTimePoint inOther) const noexcept {
		return Duration::fromNanoseconds(oa::detail::checkedI64Subtract(
			nanoseconds_, inOther.nanoseconds_, "SystemTimePoint difference overflow"));
	}
	[[nodiscard]] constexpr SystemTimePoint operator+(Duration inDuration) const noexcept {
		return SystemTimePoint(oa::detail::checkedI64Add(
			nanoseconds_, inDuration.nanoseconds(), "SystemTimePoint addition overflow"));
	}
	[[nodiscard]] constexpr SystemTimePoint operator-(Duration inDuration) const noexcept {
		return SystemTimePoint(oa::detail::checkedI64Subtract(
			nanoseconds_, inDuration.nanoseconds(), "SystemTimePoint subtraction overflow"));
	}
	constexpr SystemTimePoint& operator+=(Duration inDuration) noexcept {
		nanoseconds_ = oa::detail::checkedI64Add(
			nanoseconds_, inDuration.nanoseconds(), "SystemTimePoint addition overflow");
		return *this;
	}
	constexpr SystemTimePoint& operator-=(Duration inDuration) noexcept {
		nanoseconds_ = oa::detail::checkedI64Subtract(
			nanoseconds_, inDuration.nanoseconds(), "SystemTimePoint subtraction overflow");
		return *this;
	}

	[[nodiscard]] constexpr bool operator==(SystemTimePoint inOther) const noexcept {
		return nanoseconds_ == inOther.nanoseconds_;
	}
	[[nodiscard]] constexpr bool operator<(SystemTimePoint inOther) const noexcept {
		return nanoseconds_ < inOther.nanoseconds_;
	}
	[[nodiscard]] constexpr bool operator<=(SystemTimePoint inOther) const noexcept {
		return nanoseconds_ <= inOther.nanoseconds_;
	}
	[[nodiscard]] constexpr bool operator>(SystemTimePoint inOther) const noexcept {
		return nanoseconds_ > inOther.nanoseconds_;
	}
	[[nodiscard]] constexpr bool operator>=(SystemTimePoint inOther) const noexcept {
		return nanoseconds_ >= inOther.nanoseconds_;
	}

private:
	oa::I64 nanoseconds_ = 0;
};

class SteadyClock {
public:
	using TimePoint = SteadyTimePoint;
	using Duration = oa::Duration;

	[[nodiscard]] static SteadyTimePoint now() noexcept;
};

class SystemClock {
public:
	using TimePoint = SystemTimePoint;
	using Duration = oa::Duration;

	[[nodiscard]] static SystemTimePoint now() noexcept;
};

using SteadyDuration = Duration;
using SystemDuration = Duration;
using HighResolutionClock = SteadyClock;
using HighResolutionTimePoint = SteadyTimePoint;
using HighResolutionDuration = Duration;

[[nodiscard]] inline SteadyTimePoint steadyNow() noexcept { return SteadyClock::now(); }
[[nodiscard]] inline SystemTimePoint systemNow() noexcept { return SystemClock::now(); }
[[nodiscard]] inline HighResolutionTimePoint highResolutionNow() noexcept {
	return HighResolutionClock::now();
}

[[nodiscard]] constexpr oa::F64 durationToSeconds(Duration inDuration) noexcept {
	return inDuration.toSeconds();
}

[[nodiscard]] constexpr oa::F64 durationToMilliseconds(Duration inDuration) noexcept {
	return inDuration.toMilliseconds();
}

[[nodiscard]] constexpr oa::F64 durationToMicroseconds(Duration inDuration) noexcept {
	return inDuration.toMicroseconds();
}

} // namespace oa
