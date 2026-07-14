// OA Core — Timestamps, Duration & Datetime
// Nanosecond-precision timestamps. Stopwatch, scoped timing, human-readable datetime.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Std/Chrono.h>
#include <cctype>
#include <cstdio>
#include <ctime>

[[nodiscard]] inline std::time_t OaTimeGm(std::tm* InTm) {
#if defined(_WIN32)
	return _mkgmtime(InTm);
#else
	return timegm(InTm);
#endif
}

[[nodiscard]] inline std::tm OaGmTime(std::time_t InTime) {
	std::tm tm{};
#if defined(_WIN32)
	gmtime_s(&tm, &InTime);
#else
	gmtime_r(&InTime, &tm);
#endif
	return tm;
}

class OaTimestamp {
public:
	using Clock     = OaSteadyClock;
	using Duration  = std::chrono::nanoseconds;
	using TimePoint = std::chrono::time_point<Clock, Duration>;

	constexpr OaTimestamp() noexcept : Nanos_(0) {}
	constexpr explicit OaTimestamp(OaI64 InNanos) noexcept : Nanos_(InNanos) {}
	explicit OaTimestamp(TimePoint InTp) noexcept
		: Nanos_(std::chrono::duration_cast<Duration>(InTp.time_since_epoch()).count()) {}

	[[nodiscard]] static OaTimestamp Now() noexcept { return OaTimestamp(Clock::now()); }
	[[nodiscard]] static constexpr OaTimestamp FromSeconds(OaI64 InSeconds) noexcept { return OaTimestamp(InSeconds * 1'000'000'000LL); }
	[[nodiscard]] static constexpr OaTimestamp FromMilliseconds(OaI64 InMillis) noexcept { return OaTimestamp(InMillis * 1'000'000LL); }
	[[nodiscard]] static constexpr OaTimestamp FromMicroseconds(OaI64 InMicros) noexcept { return OaTimestamp(InMicros * 1'000LL); }
	[[nodiscard]] static constexpr OaTimestamp FromNanoseconds(OaI64 InNanos) noexcept { return OaTimestamp(InNanos); }
	[[nodiscard]] static constexpr OaTimestamp FromDouble(OaF64 InSeconds) noexcept { return OaTimestamp(static_cast<OaI64>(InSeconds * 1'000'000'000.0)); }
	[[nodiscard]] static constexpr OaTimestamp Zero() noexcept { return OaTimestamp(0); }

	[[nodiscard]] constexpr OaI64 Nanos() const noexcept { return Nanos_; }
	[[nodiscard]] constexpr OaI64 Micros() const noexcept { return Nanos_ / 1'000LL; }
	[[nodiscard]] constexpr OaI64 Millis() const noexcept { return Nanos_ / 1'000'000LL; }
	[[nodiscard]] constexpr OaI64 Secs() const noexcept { return Nanos_ / 1'000'000'000LL; }
	[[nodiscard]] constexpr OaF64 ToSeconds() const noexcept { return static_cast<OaF64>(Nanos_) / 1e9; }
	[[nodiscard]] constexpr OaF64 ToMs() const noexcept { return static_cast<OaF64>(Nanos_) / 1e6; }

	[[nodiscard]] constexpr OaTimestamp operator+(OaTimestamp InOther) const noexcept { return OaTimestamp(Nanos_ + InOther.Nanos_); }
	[[nodiscard]] constexpr OaTimestamp operator-(OaTimestamp InOther) const noexcept { return OaTimestamp(Nanos_ - InOther.Nanos_); }
	constexpr OaTimestamp& operator+=(OaTimestamp InOther) noexcept { Nanos_ += InOther.Nanos_; return *this; }
	constexpr OaTimestamp& operator-=(OaTimestamp InOther) noexcept { Nanos_ -= InOther.Nanos_; return *this; }

	[[nodiscard]] constexpr bool operator==(OaTimestamp InOther) const noexcept { return Nanos_ == InOther.Nanos_; }
	[[nodiscard]] constexpr bool operator!=(OaTimestamp InOther) const noexcept { return Nanos_ != InOther.Nanos_; }
	[[nodiscard]] constexpr bool operator<(OaTimestamp InOther) const noexcept { return Nanos_ < InOther.Nanos_; }
	[[nodiscard]] constexpr bool operator<=(OaTimestamp InOther) const noexcept { return Nanos_ <= InOther.Nanos_; }
	[[nodiscard]] constexpr bool operator>(OaTimestamp InOther) const noexcept { return Nanos_ > InOther.Nanos_; }
	[[nodiscard]] constexpr bool operator>=(OaTimestamp InOther) const noexcept { return Nanos_ >= InOther.Nanos_; }

	[[nodiscard]] constexpr bool IsValid() const noexcept { return Nanos_ > 0; }
	[[nodiscard]] constexpr explicit operator bool() const noexcept { return IsValid(); }

private:
	OaI64 Nanos_;
};

[[nodiscard]] inline OaTimestamp OaNow() noexcept { return OaTimestamp::Now(); }

class OaScopedTimer {
public:
	explicit OaScopedTimer(OaF64& OutSeconds) : OutSeconds_(OutSeconds), Start_(OaNow()) {}
	~OaScopedTimer() { OutSeconds_ = (OaNow() - Start_).ToSeconds(); }
	OaScopedTimer(const OaScopedTimer&) = delete;
	OaScopedTimer& operator=(const OaScopedTimer&) = delete;
private:
	OaF64& OutSeconds_;
	OaTimestamp Start_;
};

class OaStopwatch {
public:
	OaStopwatch() = default;
	void Start() { Start_ = OaNow(); Running_ = true; }
	void Stop() { if (Running_) { Elapsed_ += (OaNow() - Start_); Running_ = false; } }
	void Reset() { Elapsed_ = OaTimestamp::Zero(); Running_ = false; }
	void Restart() { Reset(); Start(); }
	[[nodiscard]] OaTimestamp Elapsed() const { return Running_ ? Elapsed_ + (OaNow() - Start_) : Elapsed_; }
	[[nodiscard]] OaF64 ElapsedSec() const { return Elapsed().ToSeconds(); }
	[[nodiscard]] OaF64 ElapsedMs() const { return Elapsed().ToMs(); }
	[[nodiscard]] bool IsRunning() const { return Running_; }
private:
	OaTimestamp Start_;
	OaTimestamp Elapsed_;
	bool Running_ = false;
};

// OaDatetime — human-readable date/time over OaTimestamp.
// For logs, debugging, display. NEVER for consensus or deterministic math.
class OaDatetime {
public:
	OaDatetime() noexcept : Ts_() {}
	explicit OaDatetime(OaTimestamp InTs) noexcept : Ts_(InTs) {}

	OaDatetime(OaI32 InYear, OaI32 InMonth, OaI32 InDay, OaI32 InHour = 0, OaI32 InMinute = 0, OaI32 InSecond = 0) {
		std::tm tm = {};
		tm.tm_year = InYear - 1900;
		tm.tm_mon = InMonth - 1;
		tm.tm_mday = InDay;
		tm.tm_hour = InHour;
		tm.tm_min = InMinute;
		tm.tm_sec = InSecond;
		tm.tm_isdst = 0;
		std::time_t t = OaTimeGm(&tm);
		Ts_ = OaTimestamp::FromSeconds(static_cast<OaI64>(t));
	}

	[[nodiscard]] static OaDatetime Now() noexcept { return OaDatetime(OaTimestamp::Now()); }
	[[nodiscard]] static OaDatetime FromTimestamp(OaTimestamp InTs) noexcept { return OaDatetime(InTs); }
	[[nodiscard]] static OaDatetime FromUnixSeconds(OaI64 InSeconds) noexcept { return OaDatetime(OaTimestamp::FromSeconds(InSeconds)); }
	[[nodiscard]] static OaDatetime FromDouble(OaF64 InSeconds) noexcept { return OaDatetime(OaTimestamp::FromDouble(InSeconds)); }

	[[nodiscard]] OaTimestamp GetTimestamp() const noexcept { return Ts_; }
	[[nodiscard]] OaI64 UnixSeconds() const noexcept { return Ts_.Secs(); }
	[[nodiscard]] OaF64 ToSeconds() const noexcept { return Ts_.ToSeconds(); }

	[[nodiscard]] OaI32 Year() const { return ToTm().tm_year + 1900; }
	[[nodiscard]] OaI32 Month() const { return ToTm().tm_mon + 1; }
	[[nodiscard]] OaI32 Day() const { return ToTm().tm_mday; }
	[[nodiscard]] OaI32 Hour() const { return ToTm().tm_hour; }
	[[nodiscard]] OaI32 Minute() const { return ToTm().tm_min; }
	[[nodiscard]] OaI32 Second() const { return ToTm().tm_sec; }
	[[nodiscard]] OaI32 Microsecond() const {
		return static_cast<OaI32>((Ts_.Nanos() % 1'000'000'000) / 1'000);
	}
	[[nodiscard]] OaI32 DayOfWeek() const { return ToTm().tm_wday; }
	[[nodiscard]] OaI32 DayOfYear() const { return ToTm().tm_yday + 1; }

	[[nodiscard]] OaString Format(const char* InFmt = "%Y-%m-%dT%H:%M:%SZ") const {
		std::tm tm = ToTm();
		char buf[128];
		const size_t n = std::strftime(buf, sizeof(buf), InFmt, &tm);
		if (n == 0) {
			return OaString();
		}
		return OaString(OaStringView(buf, n));
	}
	[[nodiscard]] OaString ToIso() const { return Format("%Y-%m-%dT%H:%M:%SZ"); }
	[[nodiscard]] OaString ToDate() const { return Format("%Y-%m-%d"); }
	[[nodiscard]] OaString ToTime() const { return Format("%H:%M:%S"); }
	[[nodiscard]] OaString ToIsoMicro() const {
		std::tm tm = ToTm();
		char buf[64];
		const size_t n = std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
		if (n == 0) {
			return OaString();
		}
		OaString out(OaStringView(buf, n));
		char tail[16];
		if (std::snprintf(tail, sizeof(tail), ".%06dZ", static_cast<int>(Microsecond())) > 0) {
			out += tail;
		}
		return out;
	}

	[[nodiscard]] OaDatetime Add(OaTimestamp InDuration) const noexcept { return OaDatetime(Ts_ + InDuration); }
	[[nodiscard]] OaDatetime Subtract(OaTimestamp InDuration) const noexcept { return OaDatetime(Ts_ - InDuration); }
	[[nodiscard]] OaTimestamp Diff(const OaDatetime& InOther) const noexcept { return Ts_ - InOther.Ts_; }

	[[nodiscard]] bool operator==(const OaDatetime& InOther) const noexcept { return Ts_ == InOther.Ts_; }
	[[nodiscard]] bool operator!=(const OaDatetime& InOther) const noexcept { return Ts_ != InOther.Ts_; }
	[[nodiscard]] bool operator<(const OaDatetime& InOther) const noexcept { return Ts_ < InOther.Ts_; }
	[[nodiscard]] bool operator<=(const OaDatetime& InOther) const noexcept { return Ts_ <= InOther.Ts_; }
	[[nodiscard]] bool operator>(const OaDatetime& InOther) const noexcept { return Ts_ > InOther.Ts_; }
	[[nodiscard]] bool operator>=(const OaDatetime& InOther) const noexcept { return Ts_ >= InOther.Ts_; }

	[[nodiscard]] bool IsValid() const noexcept { return Ts_.IsValid(); }

private:
	OaTimestamp Ts_;

	[[nodiscard]] std::tm ToTm() const {
		std::time_t t = static_cast<std::time_t>(Ts_.Secs());
		return OaGmTime(t);
	}
};

[[nodiscard]] inline OaDatetime OaDatetimeNow() noexcept { return OaDatetime::Now(); }
[[nodiscard]] inline OaString OaFormatTimestamp(OaTimestamp InTs) { return OaDatetime(InTs).ToIsoMicro(); }

// Smart human-readable duration: "3s", "45s", "2m 30s", "1h 15m", "2d 6h"
[[nodiscard]] inline OaString OaFormatDuration(OaF64 InSeconds) {
	if (InSeconds < 0.0) InSeconds = 0.0;
	const OaI64 total = static_cast<OaI64>(InSeconds + 0.5);
	if (total < 60) {
		OaString out;
		out += OaToString(total);
		out += "s";
		return out;
	}
	const OaI64 sec = total % 60;
	const OaI64 min = (total / 60) % 60;
	const OaI64 hr  = (total / 3600) % 24;
	const OaI64 day = total / 86400;
	OaString out;
	if (day > 0) {
		out += OaToString(day);
		out += "d";
		if (hr > 0) {
			out += " ";
			out += OaToString(hr);
			out += "h";
		}
	} else if (hr > 0) {
		out += OaToString(hr);
		out += "h";
		if (min > 0) {
			out += " ";
			out += OaToString(min);
			out += "m";
		}
	} else {
		out += OaToString(min);
		out += "m";
		if (sec > 0) {
			out += " ";
			out += OaToString(sec);
			out += "s";
		}
	}
	return out;
}

// Parse duration string: "30s", "5m", "20m", "2h", "1d", "1w", "1mo", "1y"
// Bare number (no suffix) treated as seconds. Returns 0 on parse failure.
[[nodiscard]] inline OaF64 OaParseDuration(OaStringView InStr) {
	if (InStr.empty()) return 0.0;
	size_t i = 0;
	while (i < InStr.size() && (InStr[i] == ' ' || InStr[i] == '\t')) ++i;
	OaF64 num = 0.0;
	bool hasDigit = false;
	while (i < InStr.size() && InStr[i] >= '0' && InStr[i] <= '9') {
		num = num * 10.0 + (InStr[i] - '0');
		hasDigit = true;
		++i;
	}
	if (InStr[i] == '.') {
		++i;
		OaF64 frac = 0.1;
		while (i < InStr.size() && InStr[i] >= '0' && InStr[i] <= '9') {
			num += (InStr[i] - '0') * frac;
			frac *= 0.1;
			++i;
		}
	}
	if (!hasDigit) return 0.0;
	while (i < InStr.size() && (InStr[i] == ' ' || InStr[i] == '\t')) ++i;
	if (i >= InStr.size()) return num;
	OaString suffix;
	while (i < InStr.size() && InStr[i] != ' ' && InStr[i] != '\t') {
		suffix += static_cast<char>(std::tolower(static_cast<unsigned char>(InStr[i])));
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
