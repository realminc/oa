#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/core/filesystem.h>
#include <oa/core/time.h>

#include <stdarg.h>
#include <stdio.h>

namespace oa {

enum class LogLevel : oa::U8 {
	Trace = 0,
	Debug = 1,
	Info  = 2,
	Warn  = 3,
	Error = 4,
	Fatal = 5,
	Off   = 6
};

// A component is a small value, not a closed framework enum or a mutable
// registry entry. OA supplies neutral built-ins through Builtin; downstream
// libraries define their own named constexpr values from one-to-four-character
// tags without modifying OA or acquiring process-global state.
//
// namespace ChainLog {
// inline constexpr LogComponent Consensus{"CONS"};
// }
class LogComponent {
public:
	enum Builtin : oa::U8 {
		Core,
		Runtime,
		Engine,
		Compute,
		Ml,
		Data,
		Vision,
		Video,
		Audio,
		Render,
		Ui,
		Plot,
		Animation,
		Network,
		Crypto,
		Python,
		App,
		Mcp,
	};

	// Intentionally implicit so existing LogComponent::Core-style built-ins
	// remain ordinary component arguments rather than a second public type.
	constexpr LogComponent(Builtin inBuiltin) noexcept {
		switch (inBuiltin) {
			case Core:      assign_("CORE", 4U); break;
			case Runtime:   assign_("RT",   2U); break;
			case Engine:    assign_("ENGN", 4U); break;
			case Compute:   assign_("COMP", 4U); break;
			case Ml:        assign_("ML",   2U); break;
			case Data:      assign_("DATA", 4U); break;
			case Vision:    assign_("VISN", 4U); break;
			case Video:     assign_("VID",  3U); break;
			case Audio:     assign_("AUD",  3U); break;
			case Render:    assign_("RNDR", 4U); break;
			case Ui:        assign_("UI",   2U); break;
			case Plot:      assign_("PLOT", 4U); break;
			case Animation: assign_("ANIM", 4U); break;
			case Network:   assign_("NET",  3U); break;
			case Crypto:    assign_("CRYP", 4U); break;
			case Python:    assign_("PY",   2U); break;
			case App:       assign_("APP",  3U); break;
			case Mcp:       assign_("MCP",  3U); break;
			default:        assign_("????", 4U); break;
		}
	}

	template <oa::Usize N>
	constexpr explicit LogComponent(const char (&inTag)[N]) noexcept {
		static_assert(N >= 2U and N <= 5U,
			"LogComponent tags must contain one to four characters");
		assign_(inTag, N - 1U);
	}

	[[nodiscard]] constexpr const char* cStr() const noexcept { return tag_; }

private:
	constexpr void assign_(const char* inTag, oa::Usize inLength) noexcept {
		oa::Usize index = 0U;
		for (; index < inLength and index < 4U; ++index) tag_[index] = inTag[index];
		for (; index < 4U; ++index) tag_[index] = ' ';
		tag_[4] = '\0';
	}

	char tag_[5]{};
};

// Stateful log output is an explicitly owned host session. oa::Engine creates one
// from these options and selects it for the thread that owns the engine. The
// Log* macros use that selected session and fall back to stderr before an
// engine exists or on a thread that has not selected one.
class LogOptions {
public:
	oa::String directory;
	oa::String prefix = "oa";
#ifdef NDEBUG
	LogLevel minimumLevel = LogLevel::Info;
#else
	LogLevel minimumLevel = LogLevel::Debug;
#endif
	oa::Bool consoleOutput = true;
	oa::Bool fileOutput = false;
};

class LogAccess;
class LogImpl;

class Log {
public:
	[[nodiscard]] static oa::Result<oa::UniquePtr<Log>> create(const LogOptions& inOptions = {});
	~Log();

	// writes are serialized and safe from any thread. Write never throws; an I/O
	// failure is returned to explicit callers and retained for flush/Close.
	[[nodiscard]] oa::Status write(
		LogLevel inLevel,
		LogComponent inComponent,
		const char* inFormat,
		...
	);
	[[nodiscard]] oa::Status writeV(
		LogLevel inLevel,
		LogComponent inComponent,
		const char* inFormat,
		va_list inArgs
	);
	[[nodiscard]] oa::Status flush();
	[[nodiscard]] oa::Status close();

	void setLevel(LogLevel inLevel) noexcept;
	[[nodiscard]] LogLevel getLevel() const noexcept;
	[[nodiscard]] oa::String getLogPath() const;
	[[nodiscard]] oa::Bool isOpen() const noexcept;

	Log(const Log&) = delete;
	Log& operator=(const Log&) = delete;
	Log(Log&&) noexcept = delete;
	Log& operator=(Log&&) noexcept = delete;

private:
	friend class LogAccess;
	explicit Log(oa::SharedPtr<LogImpl> inImpl) noexcept;
	oa::SharedPtr<LogImpl> impl_;
};

// Macro-compatible dispatch without process-global logger ownership. calls made
// before engine creation deliberately use the synchronous stderr fallback.
void logWrite(LogLevel inLevel, LogComponent inComponent, const char* inFormat, ...);

#define OaLogInfo(component, ...) ::oa::logWrite(::oa::LogLevel::Info, component, __VA_ARGS__)
#define OaLogWarn(component, ...) ::oa::logWrite(::oa::LogLevel::Warn, component, __VA_ARGS__)
#define OaLogError(component, ...) ::oa::logWrite(::oa::LogLevel::Error, component, __VA_ARGS__)
#define OaLogFatal(component, ...) ::oa::logWrite(::oa::LogLevel::Fatal, component, __VA_ARGS__)

#ifdef NDEBUG
	#define OaLogTrace(component, ...) ((void)0)
	#define OaLogDebug(component, ...) ((void)0)
#else
	#define OaLogTrace(component, ...) ::oa::logWrite(::oa::LogLevel::Trace, component, __VA_ARGS__)
	#define OaLogDebug(component, ...) ::oa::logWrite(::oa::LogLevel::Debug, component, __VA_ARGS__)
#endif

#define OA_CLI(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#define OA_CLI_RAW(...) do { fprintf(stderr, __VA_ARGS__); } while(0)

// format integer with comma-separated thousands (e.g. 9521568 → "9,521,568")
[[nodiscard]] inline oa::String formatNumber(oa::I64 inN) {
	if (inN < 0) {
		return oa::String("-") + formatNumber(-inN);
	}
	if (inN < 1000) {
		char smallBuf[16];
		if (::snprintf(smallBuf, sizeof(smallBuf), "%lld", static_cast<long long>(inN)) <= 0) {
			return oa::String();
		}
		return oa::String(smallBuf);
	}
	char buf[32];
	if (inN < 1000000) {
		snprintf(buf, sizeof(buf), "%lld,%03lld",
			(long long)(inN / 1000), (long long)(inN % 1000)
		);
	} else if (inN < 1000000000) {
		snprintf(buf, sizeof(buf), "%lld,%03lld,%03lld",
			(long long)(inN / 1000000), (long long)((inN / 1000) % 1000),
			(long long)(inN % 1000)
		);
	} else {
		snprintf(buf, sizeof(buf), "%lld,%03lld,%03lld,%03lld",
			(long long)(inN / 1000000000), (long long)((inN / 1000000) % 1000),
			(long long)((inN / 1000) % 1000), (long long)(inN % 1000)
		);
	}
	return oa::String(buf);
}

// Uppercase hex with 0x prefix (PCI / vulkan id style, no leading-zero padding).
[[nodiscard]] inline oa::String formatHexU32(oa::U32 inVal) {
	char buf[16];
	if (::snprintf(
			buf, sizeof(buf), "0x%X", static_cast<unsigned>(inVal)) <= 0) {
		return oa::String("0x0");
	}
	return oa::String(buf);
}

// Comma-separated decimal for oa::U64 (full range; for log readability only).
[[nodiscard]] inline oa::String formatNumberU64(oa::U64 inN) {
	if (inN == 0) {
		return oa::String("0");
	}
	char stack[32];
	int stackTop = 0;
	oa::U64 work = inN;
	while (work > 0) {
		stack[stackTop++] = static_cast<char>('0' + static_cast<int>(work % 10u));
		work /= 10u;
	}
	oa::String out;
	for (int idx = stackTop - 1; idx >= 0; --idx) {
		out += stack[idx];
		if (idx > 0 && idx % 3 == 0) {
			out += ',';
		}
	}
	return out;
}

// log a right-aligned summary line: "  (label): type(dims)          1,234"
// width 60 between left text and right value. Used by model summaries and device info.
inline void logSummary(LogComponent inComp, const char* inLeft, oa::I64 inParams) {
	auto val = formatNumber(inParams);
	oa::I32 pad = 60 - static_cast<oa::I32>(oa::strlen(inLeft))
		- static_cast<oa::I32>(val.size());
	if (pad < 1) pad = 1;
	OaLogInfo(inComp, "%s%*s%s", inLeft, pad, "", val.cStr());
}

inline void logSummary(LogComponent inComp, const char* inLeft, const char* inRight) {
	oa::I32 pad = 60 - static_cast<oa::I32>(oa::strlen(inLeft))
		- static_cast<oa::I32>(oa::strlen(inRight));
	if (pad < 1) pad = 1;
	OaLogInfo(inComp, "%s%*s%s", inLeft, pad, "", inRight);
}

class LogMetrics {
public:
	// Structured metrics to JSONL. For numerical time-series (loss, tok/s, etc.).
	// Writes, flush, and close are serialized. Configuration and path access
	// belong to the owning thread before concurrent producers start.

	// Constructors.
	LogMetrics() = default;
	explicit LogMetrics(const oa::String& inLogDir) { open(inLogDir); }

	// Destructors.
	~LogMetrics() { close(); }

	// Methods.
	oa::Status open(const oa::String& inLogDir) {
		oa::ScopedLock<oa::Mutex> lock(mutex_);
		flushUnlocked_();
		logDir_ = inLogDir;
		(void)oa::Filesystem::createDirectories(oa::Path(logDir_));
		eventsPath_ = logDir_ + "/events.jsonl";
		startTime_ = oa::now();
		isOpen_ = true;
		return oa::Status::ok();
	}

	void close() {
		oa::ScopedLock<oa::Mutex> lock(mutex_);
		flushUnlocked_();
		isOpen_.store(false, oa::MemoryOrder::Release);
	}

	[[nodiscard]] bool isOpen() const {
		return isOpen_.load(oa::MemoryOrder::Acquire);
	}
	[[nodiscard]] const oa::String& getLogDir() const { return logDir_; }

	void logScalar(const oa::String& inTag, oa::I64 inStep, oa::F64 inValue) {
		if (!isOpen()) return;
		oa::ScopedLock<oa::Mutex> lock(mutex_);
		if (!isOpen_.load(oa::MemoryOrder::Relaxed)) return;

		oa::F64 wallTime = (oa::now() - startTime_).toSeconds();

		char buf[256];
		snprintf(buf, sizeof(buf),
			R"({"tag":"%s","step":%lld,"value":%.6g,"wall_time":%.3f})",
			inTag.cStr(), static_cast<long long>(inStep), inValue, wallTime);

		buffer_ += buf;
		buffer_ += '\n';
		++bufferCount_;

		if (bufferCount_ >= flushInterval_) {
			flushUnlocked_();
		}
	}

	void logScalars(const oa::String& inTag, oa::I64 inStep, const oa::HashMap<oa::String, oa::F64>& inValues) {
		for (const auto& [name, value] : inValues) {
			logScalar(inTag + "/" + name, inStep, value);
		}
	}

	void flush() {
		oa::ScopedLock<oa::Mutex> lock(mutex_);
		flushUnlocked_();
	}

	void setFlushInterval(oa::I32 inN) {
		oa::ScopedLock<oa::Mutex> lock(mutex_);
		flushInterval_ = inN;
	}

	// Operators.
	LogMetrics(const LogMetrics&) = delete;
	LogMetrics& operator=(const LogMetrics&) = delete;
	LogMetrics(LogMetrics&&) = delete;
	LogMetrics& operator=(LogMetrics&&) = delete;

private:
	void flushUnlocked_() {
		if (buffer_.empty()) {
			return;
		}
		(void)oa::Filesystem::appendText(oa::Path(eventsPath_), buffer_);
		buffer_.clear();
		bufferCount_ = 0;
	}
	oa::String logDir_;
	oa::String eventsPath_;
	oa::Timestamp startTime_;
	oa::Mutex mutex_;
	oa::Atomic<bool> isOpen_{false};
	oa::String buffer_;
	oa::I32 bufferCount_ = 0;
	oa::I32 flushInterval_ = 16;
};

} // namespace oa
