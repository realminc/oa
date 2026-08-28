// OA Core — Timestamps, Duration & Datetime
// Nanosecond-precision timestamps. Stopwatch, scoped timing, human-readable datetime.

#pragma once

#include <oa/core/types.h>
#include <oa/core/std/chrono.h>

namespace oa {

class Timestamp {
public:
	constexpr Timestamp() noexcept : nanos_(0) {}
	constexpr explicit Timestamp(oa::I64 inNanos) noexcept : nanos_(inNanos) {}
	explicit constexpr Timestamp(oa::SteadyTimePoint inTime) noexcept
		: nanos_(inTime.nanosecondsSinceEpoch()) {}

	[[nodiscard]] static Timestamp now() noexcept { return Timestamp(oa::steadyNow()); }
	[[nodiscard]] static constexpr Timestamp fromSeconds(oa::I64 inSeconds) noexcept {
		return Timestamp(oa::Duration::fromSeconds(inSeconds).nanoseconds());
	}
	[[nodiscard]] static constexpr Timestamp fromMilliseconds(oa::I64 inMillis) noexcept {
		return Timestamp(oa::Duration::fromMilliseconds(inMillis).nanoseconds());
	}
	[[nodiscard]] static constexpr Timestamp fromMicroseconds(oa::I64 inMicros) noexcept {
		return Timestamp(oa::Duration::fromMicroseconds(inMicros).nanoseconds());
	}
	[[nodiscard]] static constexpr Timestamp fromNanoseconds(oa::I64 inNanos) noexcept { return Timestamp(inNanos); }
	[[nodiscard]] static constexpr Timestamp fromDouble(oa::F64 inSeconds) noexcept {
		return Timestamp(oa::Duration::fromDouble(inSeconds).nanoseconds());
	}
	[[nodiscard]] static constexpr Timestamp zero() noexcept { return Timestamp(0); }

	[[nodiscard]] constexpr oa::I64 nanos() const noexcept { return nanos_; }
	[[nodiscard]] constexpr oa::I64 micros() const noexcept { return nanos_ / 1'000LL; }
	[[nodiscard]] constexpr oa::I64 millis() const noexcept { return nanos_ / 1'000'000LL; }
	[[nodiscard]] constexpr oa::I64 secs() const noexcept { return nanos_ / 1'000'000'000LL; }
	[[nodiscard]] constexpr oa::F64 toSeconds() const noexcept { return static_cast<oa::F64>(nanos_) / 1e9; }
	[[nodiscard]] constexpr oa::F64 toMs() const noexcept { return static_cast<oa::F64>(nanos_) / 1e6; }

	[[nodiscard]] constexpr Timestamp operator+(Timestamp inOther) const noexcept {
		return Timestamp(oa::detail::checkedI64Add(
			nanos_, inOther.nanos_, "Timestamp addition overflow"));
	}
	[[nodiscard]] constexpr Timestamp operator-(Timestamp inOther) const noexcept {
		return Timestamp(oa::detail::checkedI64Subtract(
			nanos_, inOther.nanos_, "Timestamp subtraction overflow"));
	}
	constexpr Timestamp& operator+=(Timestamp inOther) noexcept {
		nanos_ = oa::detail::checkedI64Add(
			nanos_, inOther.nanos_, "Timestamp addition overflow");
		return *this;
	}
	constexpr Timestamp& operator-=(Timestamp inOther) noexcept {
		nanos_ = oa::detail::checkedI64Subtract(
			nanos_, inOther.nanos_, "Timestamp subtraction overflow");
		return *this;
	}

	[[nodiscard]] constexpr bool operator==(Timestamp inOther) const noexcept { return nanos_ == inOther.nanos_; }
	[[nodiscard]] constexpr bool operator!=(Timestamp inOther) const noexcept { return nanos_ != inOther.nanos_; }
	[[nodiscard]] constexpr bool operator<(Timestamp inOther) const noexcept { return nanos_ < inOther.nanos_; }
	[[nodiscard]] constexpr bool operator<=(Timestamp inOther) const noexcept { return nanos_ <= inOther.nanos_; }
	[[nodiscard]] constexpr bool operator>(Timestamp inOther) const noexcept { return nanos_ > inOther.nanos_; }
	[[nodiscard]] constexpr bool operator>=(Timestamp inOther) const noexcept { return nanos_ >= inOther.nanos_; }

	[[nodiscard]] constexpr bool isValid() const noexcept { return nanos_ > 0; }
	[[nodiscard]] constexpr explicit operator bool() const noexcept { return isValid(); }

private:
	oa::I64 nanos_;
};

[[nodiscard]] inline Timestamp now() noexcept { return Timestamp::now(); }

class ScopedTimer {
public:
	explicit ScopedTimer(oa::F64& outSeconds) : outSeconds_(outSeconds), start_(now()) {}
	~ScopedTimer() { outSeconds_ = (now() - start_).toSeconds(); }
	ScopedTimer(const ScopedTimer&) = delete;
	ScopedTimer& operator=(const ScopedTimer&) = delete;
private:
	oa::F64& outSeconds_;
	Timestamp start_;
};

class Stopwatch {
public:
	Stopwatch() = default;
	void start() { start_ = now(); running_ = true; }
	void stop() { if (running_) { elapsed_ += (now() - start_); running_ = false; } }
	void reset() { elapsed_ = Timestamp::zero(); running_ = false; }
	void restart() { reset(); start(); }
	[[nodiscard]] Timestamp elapsed() const { return running_ ? elapsed_ + (now() - start_) : elapsed_; }
	[[nodiscard]] oa::F64 elapsedSec() const { return elapsed().toSeconds(); }
	[[nodiscard]] oa::F64 elapsedMs() const { return elapsed().toMs(); }
	[[nodiscard]] bool isRunning() const { return running_; }
private:
	Timestamp start_;
	Timestamp elapsed_;
	bool running_ = false;
};

// Datetime — human-readable UTC date/time over the system clock.
// For logs, debugging, display. NEVER for consensus or deterministic math.
class Datetime {
public:
	Datetime() noexcept = default;
	explicit constexpr Datetime(oa::SystemTimePoint inTime) noexcept : time_(inTime) {}

	Datetime(
		oa::I32 inYear,
		oa::I32 inMonth,
		oa::I32 inDay,
		oa::I32 inHour = 0,
		oa::I32 inMinute = 0,
		oa::I32 inSecond = 0
	);

	[[nodiscard]] static Datetime now() noexcept;
	[[nodiscard]] static constexpr Datetime fromUnixSeconds(oa::I64 inSeconds) noexcept {
		return Datetime(oa::SystemTimePoint(
			oa::Duration::fromSeconds(inSeconds).nanoseconds()));
	}
	[[nodiscard]] static constexpr Datetime fromUnixNanoseconds(oa::I64 inNanoseconds) noexcept {
		return Datetime(oa::SystemTimePoint(inNanoseconds));
	}
	[[nodiscard]] static constexpr Datetime fromDouble(oa::F64 inSeconds) noexcept {
		return fromUnixNanoseconds(oa::Duration::fromDouble(inSeconds).nanoseconds());
	}

	[[nodiscard]] oa::SystemTimePoint getTimePoint() const noexcept { return time_; }
	[[nodiscard]] oa::I64 unixNanoseconds() const noexcept {
		return time_.nanosecondsSinceEpoch();
	}
	[[nodiscard]] oa::I64 unixSeconds() const noexcept {
		return unixNanoseconds() / 1'000'000'000LL;
	}
	[[nodiscard]] oa::F64 toSeconds() const noexcept {
		return static_cast<oa::F64>(unixNanoseconds()) / 1'000'000'000.0;
	}

	[[nodiscard]] oa::I32 year() const;
	[[nodiscard]] oa::I32 month() const;
	[[nodiscard]] oa::I32 day() const;
	[[nodiscard]] oa::I32 hour() const;
	[[nodiscard]] oa::I32 minute() const;
	[[nodiscard]] oa::I32 second() const;
	[[nodiscard]] oa::I32 microsecond() const {
		oa::I64 subsecond = unixNanoseconds() % 1'000'000'000LL;
		if (subsecond < 0) subsecond += 1'000'000'000LL;
		return static_cast<oa::I32>(subsecond / 1'000LL);
	}
	[[nodiscard]] oa::I32 dayOfWeek() const;
	[[nodiscard]] oa::I32 dayOfYear() const;

	[[nodiscard]] oa::String format(const char* inFmt = "%Y-%m-%dT%H:%M:%SZ") const;
	[[nodiscard]] oa::String toIso() const { return format("%Y-%m-%dT%H:%M:%SZ"); }
	[[nodiscard]] oa::String toDate() const { return format("%Y-%m-%d"); }
	[[nodiscard]] oa::String toTime() const { return format("%H:%M:%S"); }
	[[nodiscard]] oa::String toIsoMicro() const;

	[[nodiscard]] Datetime add(oa::Duration inDuration) const noexcept {
		return Datetime(time_ + inDuration);
	}
	[[nodiscard]] Datetime subtract(oa::Duration inDuration) const noexcept {
		return Datetime(time_ - inDuration);
	}
	[[nodiscard]] oa::Duration diff(const Datetime& inOther) const noexcept {
		return time_ - inOther.time_;
	}

	[[nodiscard]] bool operator==(const Datetime& inOther) const noexcept { return time_ == inOther.time_; }
	[[nodiscard]] bool operator!=(const Datetime& inOther) const noexcept { return not (*this == inOther); }
	[[nodiscard]] bool operator<(const Datetime& inOther) const noexcept { return time_ < inOther.time_; }
	[[nodiscard]] bool operator<=(const Datetime& inOther) const noexcept { return time_ <= inOther.time_; }
	[[nodiscard]] bool operator>(const Datetime& inOther) const noexcept { return time_ > inOther.time_; }
	[[nodiscard]] bool operator>=(const Datetime& inOther) const noexcept { return time_ >= inOther.time_; }

	[[nodiscard]] bool isValid() const noexcept { return unixNanoseconds() > 0; }

private:
	oa::SystemTimePoint time_;
};

[[nodiscard]] inline Datetime datetimeNow() noexcept { return Datetime::now(); }
[[nodiscard]] inline oa::String formatTimestamp(oa::SystemTimePoint inTime) {
	return Datetime(inTime).toIsoMicro();
}

// Smart human-readable duration: "3s", "45s", "2m 30s", "1h 15m", "2d 6h"
[[nodiscard]] inline oa::String formatDuration(oa::F64 inSeconds) {
	OA_REQUIRE_MSG(
		inSeconds == inSeconds
			and inSeconds >= oa::Limits<oa::F64>::lowest()
			and inSeconds <= oa::Limits<oa::F64>::max(),
		"formatDuration requires a finite value");
	if (inSeconds < 0.0) inSeconds = 0.0;
	const oa::F64 rounded = inSeconds + 0.5;
	OA_REQUIRE_MSG(rounded < 0x1.0p63, "formatDuration seconds overflow");
	const oa::I64 total = static_cast<oa::I64>(rounded);
	if (total < 60) {
		oa::String out;
		out += toString(total);
		out += "s";
		return out;
	}
	const oa::I64 sec = total % 60;
	const oa::I64 min = (total / 60) % 60;
	const oa::I64 hr  = (total / 3600) % 24;
	const oa::I64 day = total / 86400;
	oa::String out;
	if (day > 0) {
		out += toString(day);
		out += "d";
		if (hr > 0) {
			out += " ";
			out += toString(hr);
			out += "h";
		}
	} else if (hr > 0) {
		out += toString(hr);
		out += "h";
		if (min > 0) {
			out += " ";
			out += toString(min);
			out += "m";
		}
	} else {
		out += toString(min);
		out += "m";
		if (sec > 0) {
			out += " ";
			out += toString(sec);
			out += "s";
		}
	}
	return out;
}

// parse duration string: "30s", "5m", "20m", "2h", "1d", "1w", "1mo", "1y"
// Bare number (no suffix) treated as seconds. Returns 0 on parse failure.
[[nodiscard]] inline oa::F64 parseDuration(oa::StringView inStr) {
	if (inStr.empty()) return 0.0;
	oa::Usize i = 0;
	while (i < inStr.size() && (inStr[i] == ' ' || inStr[i] == '\t')) ++i;
	oa::F64 num = 0.0;
	bool hasDigit = false;
	while (i < inStr.size() && inStr[i] >= '0' && inStr[i] <= '9') {
		num = num * 10.0 + (inStr[i] - '0');
		hasDigit = true;
		++i;
	}
	if (i < inStr.size() and inStr[i] == '.') {
		++i;
		oa::F64 frac = 0.1;
		while (i < inStr.size() && inStr[i] >= '0' && inStr[i] <= '9') {
			num += (inStr[i] - '0') * frac;
			frac *= 0.1;
			++i;
		}
	}
	if (!hasDigit) return 0.0;
	while (i < inStr.size() && (inStr[i] == ' ' || inStr[i] == '\t')) ++i;
	if (i >= inStr.size()) return num;
	oa::String suffix;
	while (i < inStr.size() && inStr[i] != ' ' && inStr[i] != '\t') {
		const char character = inStr[i];
		suffix += character >= 'A' and character <= 'Z'
			? static_cast<char>(character - 'A' + 'a') : character;
		++i;
	}
	if (suffix == "s" || suffix == "sec") return num;
	if (suffix == "m" || suffix == "min") return num * 60.0;
	if (suffix == "h" || suffix == "hr") return num * 3600.0;
	if (suffix == "d" || suffix == "day" || suffix == "days") return num * 86400.0;
	if (suffix == "w" || suffix == "wk" || suffix == "week" || suffix == "weeks") return num * 604800.0;
	if (suffix == "mo" || suffix == "month" || suffix == "months") return num * 2592000.0;
	if (suffix == "y" || suffix == "yr" || suffix == "year" || suffix == "years") return num * 31536000.0;
	return 0.0;
}

} // namespace oa
