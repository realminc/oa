#pragma once

// OA-owned clock values. Platform clock APIs remain in the compiled core
// implementation; no hosted C++ time type or native handle crosses this
// boundary.

#define OA_TYPES_H_SKIP_REST
#include <oa/core/types.h>
#undef OA_TYPES_H_SKIP_REST

namespace oa {

class Duration {
public:
	constexpr Duration() noexcept = default;
	constexpr explicit Duration(oa::I64 inNanoseconds) noexcept
		: nanoseconds_(inNanoseconds) {}

	[[nodiscard]] static constexpr Duration fromSeconds(oa::I64 inSeconds) noexcept {
		return Duration(inSeconds * 1'000'000'000LL);
	}
	[[nodiscard]] static constexpr Duration fromMilliseconds(oa::I64 inMilliseconds) noexcept {
		return Duration(inMilliseconds * 1'000'000LL);
	}
	[[nodiscard]] static constexpr Duration fromMicroseconds(oa::I64 inMicroseconds) noexcept {
		return Duration(inMicroseconds * 1'000LL);
	}
	[[nodiscard]] static constexpr Duration fromNanoseconds(oa::I64 inNanoseconds) noexcept {
		return Duration(inNanoseconds);
	}
	[[nodiscard]] static constexpr Duration fromDouble(oa::F64 inSeconds) noexcept {
		return Duration(static_cast<oa::I64>(inSeconds * 1'000'000'000.0));
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
		return Duration(nanoseconds_ + inOther.nanoseconds_);
	}
	[[nodiscard]] constexpr Duration operator-(Duration inOther) const noexcept {
		return Duration(nanoseconds_ - inOther.nanoseconds_);
	}
	constexpr Duration& operator+=(Duration inOther) noexcept {
		nanoseconds_ += inOther.nanoseconds_;
		return *this;
	}
	constexpr Duration& operator-=(Duration inOther) noexcept {
		nanoseconds_ -= inOther.nanoseconds_;
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
		return Duration::fromNanoseconds(nanoseconds_ - inOther.nanoseconds_);
	}
	[[nodiscard]] constexpr SteadyTimePoint operator+(Duration inDuration) const noexcept {
		return SteadyTimePoint(nanoseconds_ + inDuration.nanoseconds());
	}
	[[nodiscard]] constexpr SteadyTimePoint operator-(Duration inDuration) const noexcept {
		return SteadyTimePoint(nanoseconds_ - inDuration.nanoseconds());
	}
	constexpr SteadyTimePoint& operator+=(Duration inDuration) noexcept {
		nanoseconds_ += inDuration.nanoseconds();
		return *this;
	}
	constexpr SteadyTimePoint& operator-=(Duration inDuration) noexcept {
		nanoseconds_ -= inDuration.nanoseconds();
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
		return Duration::fromNanoseconds(nanoseconds_ - inOther.nanoseconds_);
	}
	[[nodiscard]] constexpr SystemTimePoint operator+(Duration inDuration) const noexcept {
		return SystemTimePoint(nanoseconds_ + inDuration.nanoseconds());
	}
	[[nodiscard]] constexpr SystemTimePoint operator-(Duration inDuration) const noexcept {
		return SystemTimePoint(nanoseconds_ - inDuration.nanoseconds());
	}
	constexpr SystemTimePoint& operator+=(Duration inDuration) noexcept {
		nanoseconds_ += inDuration.nanoseconds();
		return *this;
	}
	constexpr SystemTimePoint& operator-=(Duration inDuration) noexcept {
		nanoseconds_ -= inDuration.nanoseconds();
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
