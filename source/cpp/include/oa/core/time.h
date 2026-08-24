// OA Core — Timestamps, Duration & Datetime
// Nanosecond-precision timestamps. Stopwatch, scoped timing, human-readable datetime.

#pragma once

#include <oa/core/types.h>
#include <oa/core/std/chrono.h>
#include <cctype>
#include <cstdio>
#include <ctime>

namespace oa {

[[nodiscard]] inline std::time_t timeGm(std::tm* inTm) {
#if defined(_WIN32)
	return _mkgmtime(inTm);
#else
	return timegm(inTm);
#endif
}

[[nodiscard]] inline std::tm gmTime(std::time_t inTime) {
	std::tm tm{};
#if defined(_WIN32)
	gmtime_s(&tm, &inTime);
#else
	gmtime_r(&inTime, &tm);
#endif
	return tm;
}

class Timestamp {
public:
	using Clock     = oa::SteadyClock;
	using Duration  = std::chrono::nanoseconds;
	using TimePoint = std::chrono::time_point<Clock, Duration>;

	constexpr Timestamp() noexcept : nanos_(0) {}
	constexpr explicit Timestamp(oa::I64 inNanos) noexcept : nanos_(inNanos) {}
	explicit Timestamp(TimePoint inTp) noexcept
		: nanos_(std::chrono::duration_cast<Duration>(inTp.time_since_epoch()).count()) {}

	[[nodiscard]] static Timestamp now() noexcept { return Timestamp(Clock::now()); }
	[[nodiscard]] static constexpr Timestamp fromSeconds(oa::I64 inSeconds) noexcept { return Timestamp(inSeconds * 1'000'000'000LL); }
	[[nodiscard]] static constexpr Timestamp fromMilliseconds(oa::I64 inMillis) noexcept { return Timestamp(inMillis * 1'000'000LL); }
	[[nodiscard]] static constexpr Timestamp fromMicroseconds(oa::I64 inMicros) noexcept { return Timestamp(inMicros * 1'000LL); }
	[[nodiscard]] static constexpr Timestamp fromNanoseconds(oa::I64 inNanos) noexcept { return Timestamp(inNanos); }
	[[nodiscard]] static constexpr Timestamp fromDouble(oa::F64 inSeconds) noexcept { return Timestamp(static_cast<oa::I64>(inSeconds * 1'000'000'000.0)); }
	[[nodiscard]] static constexpr Timestamp zero() noexcept { return Timestamp(0); }

	[[nodiscard]] constexpr oa::I64 nanos() const noexcept { return nanos_; }
	[[nodiscard]] constexpr oa::I64 micros() const noexcept { return nanos_ / 1'000LL; }
	[[nodiscard]] constexpr oa::I64 millis() const noexcept { return nanos_ / 1'000'000LL; }
	[[nodiscard]] constexpr oa::I64 secs() const noexcept { return nanos_ / 1'000'000'000LL; }
	[[nodiscard]] constexpr oa::F64 toSeconds() const noexcept { return static_cast<oa::F64>(nanos_) / 1e9; }
	[[nodiscard]] constexpr oa::F64 toMs() const noexcept { return static_cast<oa::F64>(nanos_) / 1e6; }

	[[nodiscard]] constexpr Timestamp operator+(Timestamp inOther) const noexcept { return Timestamp(nanos_ + inOther.nanos_); }
	[[nodiscard]] constexpr Timestamp operator-(Timestamp inOther) const noexcept { return Timestamp(nanos_ - inOther.nanos_); }
	constexpr Timestamp& operator+=(Timestamp inOther) noexcept { nanos_ += inOther.nanos_; return *this; }
	constexpr Timestamp& operator-=(Timestamp inOther) noexcept { nanos_ -= inOther.nanos_; return *this; }

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

// Datetime — human-readable date/time over Timestamp.
// For logs, debugging, display. NEVER for consensus or deterministic math.
class Datetime {
public:
	Datetime() noexcept : ts_() {}
	explicit Datetime(Timestamp inTs) noexcept : ts_(inTs) {}

	Datetime(oa::I32 inYear, oa::I32 inMonth, oa::I32 inDay, oa::I32 inHour = 0, oa::I32 inMinute = 0, oa::I32 inSecond = 0) {
		std::tm tm = {};
		tm.tm_year = inYear - 1900;
		tm.tm_mon = inMonth - 1;
		tm.tm_mday = inDay;
		tm.tm_hour = inHour;
		tm.tm_min = inMinute;
		tm.tm_sec = inSecond;
		tm.tm_isdst = 0;
		std::time_t t = timeGm(&tm);
		ts_ = Timestamp::fromSeconds(static_cast<oa::I64>(t));
	}

	[[nodiscard]] static Datetime now() noexcept { return Datetime(Timestamp::now()); }
	[[nodiscard]] static Datetime fromTimestamp(Timestamp inTs) noexcept { return Datetime(inTs); }
	[[nodiscard]] static Datetime fromUnixSeconds(oa::I64 inSeconds) noexcept { return Datetime(Timestamp::fromSeconds(inSeconds)); }
	[[nodiscard]] static Datetime fromDouble(oa::F64 inSeconds) noexcept { return Datetime(Timestamp::fromDouble(inSeconds)); }

	[[nodiscard]] Timestamp getTimestamp() const noexcept { return ts_; }
	[[nodiscard]] oa::I64 unixSeconds() const noexcept { return ts_.secs(); }
	[[nodiscard]] oa::F64 toSeconds() const noexcept { return ts_.toSeconds(); }

	[[nodiscard]] oa::I32 year() const { return toTm().tm_year + 1900; }
	[[nodiscard]] oa::I32 month() const { return toTm().tm_mon + 1; }
	[[nodiscard]] oa::I32 day() const { return toTm().tm_mday; }
	[[nodiscard]] oa::I32 hour() const { return toTm().tm_hour; }
	[[nodiscard]] oa::I32 minute() const { return toTm().tm_min; }
	[[nodiscard]] oa::I32 second() const { return toTm().tm_sec; }
	[[nodiscard]] oa::I32 microsecond() const {
		return static_cast<oa::I32>((ts_.nanos() % 1'000'000'000) / 1'000);
	}
	[[nodiscard]] oa::I32 dayOfWeek() const { return toTm().tm_wday; }
	[[nodiscard]] oa::I32 dayOfYear() const { return toTm().tm_yday + 1; }

	[[nodiscard]] oa::String format(const char* inFmt = "%Y-%m-%dT%H:%M:%SZ") const {
		std::tm tm = toTm();
		char buf[128];
		const size_t n = std::strftime(buf, sizeof(buf), inFmt, &tm);
		if (n == 0) {
			return oa::String();
		}
		return oa::String(oa::StringView(buf, n));
	}
	[[nodiscard]] oa::String toIso() const { return format("%Y-%m-%dT%H:%M:%SZ"); }
	[[nodiscard]] oa::String toDate() const { return format("%Y-%m-%d"); }
	[[nodiscard]] oa::String toTime() const { return format("%H:%M:%S"); }
	[[nodiscard]] oa::String toIsoMicro() const {
		std::tm tm = toTm();
		char buf[64];
		const size_t n = std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
		if (n == 0) {
			return oa::String();
		}
		oa::String out(oa::StringView(buf, n));
		char tail[16];
		if (std::snprintf(tail, sizeof(tail), ".%06dZ", static_cast<int>(microsecond())) > 0) {
			out += tail;
		}
		return out;
	}

	[[nodiscard]] Datetime add(Timestamp inDuration) const noexcept { return Datetime(ts_ + inDuration); }
	[[nodiscard]] Datetime subtract(Timestamp inDuration) const noexcept { return Datetime(ts_ - inDuration); }
	[[nodiscard]] Timestamp diff(const Datetime& inOther) const noexcept { return ts_ - inOther.ts_; }

	[[nodiscard]] bool operator==(const Datetime& inOther) const noexcept { return ts_ == inOther.ts_; }
	[[nodiscard]] bool operator!=(const Datetime& inOther) const noexcept { return ts_ != inOther.ts_; }
	[[nodiscard]] bool operator<(const Datetime& inOther) const noexcept { return ts_ < inOther.ts_; }
	[[nodiscard]] bool operator<=(const Datetime& inOther) const noexcept { return ts_ <= inOther.ts_; }
	[[nodiscard]] bool operator>(const Datetime& inOther) const noexcept { return ts_ > inOther.ts_; }
	[[nodiscard]] bool operator>=(const Datetime& inOther) const noexcept { return ts_ >= inOther.ts_; }

	[[nodiscard]] bool isValid() const noexcept { return ts_.isValid(); }

private:
	Timestamp ts_;

	[[nodiscard]] std::tm toTm() const {
		std::time_t t = static_cast<std::time_t>(ts_.secs());
		return gmTime(t);
	}
};

[[nodiscard]] inline Datetime datetimeNow() noexcept { return Datetime::now(); }
[[nodiscard]] inline oa::String formatTimestamp(Timestamp inTs) { return Datetime(inTs).toIsoMicro(); }

// Smart human-readable duration: "3s", "45s", "2m 30s", "1h 15m", "2d 6h"
[[nodiscard]] inline oa::String formatDuration(oa::F64 inSeconds) {
	if (inSeconds < 0.0) inSeconds = 0.0;
	const oa::I64 total = static_cast<oa::I64>(inSeconds + 0.5);
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
	size_t i = 0;
	while (i < inStr.size() && (inStr[i] == ' ' || inStr[i] == '\t')) ++i;
	oa::F64 num = 0.0;
	bool hasDigit = false;
	while (i < inStr.size() && inStr[i] >= '0' && inStr[i] <= '9') {
		num = num * 10.0 + (inStr[i] - '0');
		hasDigit = true;
		++i;
	}
	if (inStr[i] == '.') {
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
		suffix += static_cast<char>(std::tolower(static_cast<unsigned char>(inStr[i])));
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
