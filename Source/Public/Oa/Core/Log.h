#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/FileIo.h>
#include <Oa/Core/Time.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <mutex>

enum class OaLogLevel : OaU8 {
	Trace = 0,
	Debug = 1,
	Info  = 2,
	Warn  = 3,
	Error = 4,
	Fatal = 5,
	Off   = 6
};

enum class OaLogComponent : OaU8 {
	Core, ML, Crypto, Chain, Network, Storage, Api, App,
	// Extended components for realm-chain
	Consensus, Exchange, Execution, Mempool, Validator, P2P
};

[[nodiscard]] inline const char* OaGetComponentName(OaLogComponent InC) {
	switch (InC) {
		case OaLogComponent::Core:      return "CORE";
		case OaLogComponent::ML:        return "ML  ";
		case OaLogComponent::Crypto:    return "CRYP";
		case OaLogComponent::Chain:     return "CAIN";
		case OaLogComponent::Network:   return "NET ";
		case OaLogComponent::Storage:   return "STOR";
		case OaLogComponent::Api:       return "API ";
		case OaLogComponent::App:       return "APP ";
		case OaLogComponent::Consensus: return "CONS";
		case OaLogComponent::Exchange:  return "EXCH";
		case OaLogComponent::Execution: return "EXEC";
		case OaLogComponent::Mempool:   return "MEMP";
		case OaLogComponent::Validator: return "VALD";
		case OaLogComponent::P2P:       return "P2P ";
		default:                        return "????";
	}
}

class OaLog {
public:
	// OaLog class.

	// Methods.
	static OaLog& Instance() { static OaLog instance; return instance; }

	void Init(
		const OaString& InLogPath = "",
		OaLogLevel InMinLevel = OaLogLevel::Info,
		OaBool InConsoleOutput = true,
		const OaString& InPrefix = "oa"
	) {
		std::lock_guard<std::mutex> lock(Mutex_);
		MinLevel_ = InMinLevel;
		ConsoleOutput_ = InConsoleOutput;
		if (!InLogPath.empty() && InLogPath != "none") {
			try {
				auto now = std::chrono::system_clock::now();
				auto time = std::chrono::system_clock::to_time_t(now);
				char filename[256];
				std::strftime(filename, sizeof(filename), "%Y%m%d", std::localtime(&time));
				LogPath_ = InLogPath + "/" + InPrefix + "_" + filename + ".log";
				LogFile_.open(LogPath_.StdStr(), std::ios::app);
				if (LogFile_.is_open()) {
					LogFile_ << "═══════════════════════════════════════════════════════════════\n";
					LogFile_ << "  OA LOG - Started at " << std::ctime(&time);
					LogFile_ << "═══════════════════════════════════════════════════════════════\n\n";
					LogFile_.flush();
				}
			} catch (...) {}
		}
		Initialized_ = true;
	}

	void SetLevel(OaLogLevel InLevel) { MinLevel_ = InLevel; }
	[[nodiscard]] OaLogLevel GetLevel() const { return MinLevel_; }

	void Log(OaLogLevel InLevel, OaLogComponent InComponent, const char* InFmt, ...) {
		if (InLevel < MinLevel_) {
			return;
		}
		std::lock_guard<std::mutex> lock(Mutex_);
		auto now = std::chrono::system_clock::now();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
		auto time = std::chrono::system_clock::to_time_t(now);
		char timestamp[32];
		std::strftime(timestamp, sizeof(timestamp), "%H:%M:%S", std::localtime(&time));
		char msg[4096];
		va_list args;
		va_start(args, InFmt);
		vsnprintf(msg, sizeof(msg), InFmt, args);
		va_end(args);
		const char* levelStr;
		const char* color;
		switch (InLevel) {
			case OaLogLevel::Trace: levelStr = "TRACE"; color = "\033[90m"; break;
			case OaLogLevel::Debug: levelStr = "DEBUG"; color = "\033[36m"; break;
			case OaLogLevel::Info:  levelStr = "INFO "; color = "\033[32m"; break;
			case OaLogLevel::Warn:  levelStr = "WARN "; color = "\033[33m"; break;
			case OaLogLevel::Error: levelStr = "ERROR"; color = "\033[31m"; break;
			case OaLogLevel::Fatal: levelStr = "FATAL"; color = "\033[35m"; break;
			default:                levelStr = "?????"; color = "";         break;
		}
		if (LogFile_.is_open()) {
			LogFile_ << timestamp << "." << std::setfill('0') << std::setw(3) << ms.count()
							 << " [" << levelStr << "] [" << OaGetComponentName(InComponent) << "] " << msg << "\n";
			LogFile_.flush();
		}
		if (ConsoleOutput_) {
			fprintf(stderr, "%s%s.%03ld [%s] [%s] %s\033[0m\n", color, timestamp, static_cast<long>(ms.count()), levelStr, OaGetComponentName(InComponent), msg);
		}
	}

	[[nodiscard]] const OaString& GetLogPath() const { return LogPath_; }

private:
	// Data, class members.
	std::mutex Mutex_;
	std::ofstream LogFile_;
	OaString LogPath_;
	OaLogLevel MinLevel_ = OaLogLevel::Info;  // default level is info
	OaBool ConsoleOutput_ = true;  // default is to output to console
	OaBool Initialized_ = false;  // default is not initialized

	// Constructors.
	OaLog() = default;

	// Destructors.
	~OaLog() {
		if (LogFile_.is_open()) {
			LogFile_.close(); 
		}
	}
};

#define OA_LOG_INIT(Path, Level, Console) OaLog::Instance().Init(Path, Level, Console)
#define OA_LOG_INIT_PREFIX(Path, Level, Console, Prefix) OaLog::Instance().Init(Path, Level, Console, Prefix)
#define OA_LOG_SET_LEVEL(Level) OaLog::Instance().SetLevel(Level)

#define OA_LOG_INFO(Component, ...) OaLog::Instance().Log(OaLogLevel::Info, Component, __VA_ARGS__)
#define OA_LOG_WARN(Component, ...) OaLog::Instance().Log(OaLogLevel::Warn, Component, __VA_ARGS__)
#define OA_LOG_ERROR(Component, ...) OaLog::Instance().Log(OaLogLevel::Error, Component, __VA_ARGS__)
#define OA_LOG_FATAL(Component, ...) OaLog::Instance().Log(OaLogLevel::Fatal, Component, __VA_ARGS__)

#ifdef NDEBUG
	#define OA_LOG_TRACE(Component, ...) ((void)0)
	#define OA_LOG_DEBUG(Component, ...) ((void)0)
#else
	#define OA_LOG_TRACE(Component, ...) OaLog::Instance().Log(OaLogLevel::Trace, Component, __VA_ARGS__)
	#define OA_LOG_DEBUG(Component, ...) OaLog::Instance().Log(OaLogLevel::Debug, Component, __VA_ARGS__)
#endif

#define OA_CLI(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#define OA_CLI_RAW(...) do { fprintf(stderr, __VA_ARGS__); } while(0)

// Format integer with comma-separated thousands (e.g. 9521568 → "9,521,568")
[[nodiscard]] inline OaString OaFormatNumber(OaI64 InN) {
	if (InN < 0) {
		return OaString("-") + OaFormatNumber(-InN);
	}
	if (InN < 1000) {
		char smallBuf[16];
		if (std::snprintf(smallBuf, sizeof(smallBuf), "%lld", static_cast<long long>(InN)) <= 0) {
			return OaString();
		}
		return OaString(smallBuf);
	}
	char buf[32];
	if (InN < 1000000) {
		snprintf(buf, sizeof(buf), "%lld,%03lld",
			(long long)(InN / 1000), (long long)(InN % 1000)
		);
	} else if (InN < 1000000000) {
		snprintf(buf, sizeof(buf), "%lld,%03lld,%03lld",
			(long long)(InN / 1000000), (long long)((InN / 1000) % 1000),
			(long long)(InN % 1000)
		);
	} else {
		snprintf(buf, sizeof(buf), "%lld,%03lld,%03lld,%03lld",
			(long long)(InN / 1000000000), (long long)((InN / 1000000) % 1000),
			(long long)((InN / 1000) % 1000), (long long)(InN % 1000)
		);
	}
	return OaString(buf);
}

// Uppercase hex with 0x prefix (PCI / Vulkan id style, no leading-zero padding).
[[nodiscard]] inline OaString OaFormatHexU32(OaU32 InVal) {
	char buf[16];
	if (std::snprintf(
			buf, sizeof(buf), "0x%X", static_cast<unsigned>(InVal)) <= 0) {
		return OaString("0x0");
	}
	return OaString(buf);
}

// Comma-separated decimal for OaU64 (full range; for log readability only).
[[nodiscard]] inline OaString OaFormatNumberU64(OaU64 InN) {
	if (InN == 0) {
		return OaString("0");
	}
	char stack[32];
	int stackTop = 0;
	OaU64 work = InN;
	while (work > 0) {
		stack[stackTop++] = static_cast<char>('0' + static_cast<int>(work % 10u));
		work /= 10u;
	}
	OaString out;
	for (int idx = stackTop - 1; idx >= 0; --idx) {
		out += stack[idx];
		if (idx > 0 && idx % 3 == 0) {
			out += ',';
		}
	}
	return out;
}

// Log a right-aligned summary line: "  (label): Type(dims)          1,234"
// Width 60 between left text and right value. Used by model summaries and device info.
inline void OaLogSummary(OaLogComponent InComp, const char* InLeft, OaI64 InParams) {
	auto val = OaFormatNumber(InParams);
	OaI32 pad = 60 - static_cast<OaI32>(strlen(InLeft)) - static_cast<OaI32>(val.size());
	if (pad < 1) pad = 1;
	OA_LOG_INFO(InComp, "%s%*s%s", InLeft, pad, "", val.c_str());
}

inline void OaLogSummary(OaLogComponent InComp, const char* InLeft, const char* InRight) {
	OaI32 pad = 60 - static_cast<OaI32>(strlen(InLeft)) - static_cast<OaI32>(strlen(InRight));
	if (pad < 1) pad = 1;
	OA_LOG_INFO(InComp, "%s%*s%s", InLeft, pad, "", InRight);
}

class OaLogMetrics {
public:
	// Structured metrics to JSONL. For numerical time-series (loss, tok/s, etc.).
	// Thread-safe. Reusable across ML training, backtesting, benchmarks.

	// Constructors.
	OaLogMetrics() = default;
	explicit OaLogMetrics(const OaString& InLogDir) { Open(InLogDir); }

	// Destructors.
	~OaLogMetrics() { Close(); }
	
	// Methods.
	OaStatus Open(const OaString& InLogDir) {
		LogDir_ = InLogDir;
		(void)OaFileIo::CreateDirectories(OaPath(LogDir_));
		EventsPath_ = LogDir_ + "/events.jsonl";
		StartTime_ = OaNow();
		IsOpen_ = true;
		return OaStatus::Ok();
	}

	void Close() {
		std::lock_guard<std::mutex> lock(Mutex_);
		Flush();
		IsOpen_ = false;
	}

	[[nodiscard]] bool IsOpen() const { return IsOpen_; }
	[[nodiscard]] const OaString& GetLogDir() const { return LogDir_; }

	void LogScalar(const OaString& InTag, OaI64 InStep, OaF64 InValue) {
		if (!IsOpen_) return;
		std::lock_guard<std::mutex> lock(Mutex_);

		OaF64 wallTime = (OaNow() - StartTime_).ToSeconds();

		char buf[256];
		snprintf(buf, sizeof(buf),
			R"({"tag":"%s","step":%lld,"value":%.6g,"wall_time":%.3f})",
			InTag.c_str(), static_cast<long long>(InStep), InValue, wallTime);

		Buffer_ += buf;
		Buffer_ += '\n';
		++BufferCount_;

		if (BufferCount_ >= FlushInterval_) {
			Flush();
		}
	}

	void LogScalars(const OaString& InTag, OaI64 InStep, const OaHashMap<OaString, OaF64>& InValues) {
		for (const auto& [name, value] : InValues) {
			LogScalar(InTag + "/" + name, InStep, value);
		}
	}

	void Flush() {
		if (Buffer_.empty()) {
			return;
		}
		(void)OaFileIo::AppendText(OaPath(EventsPath_), Buffer_);
		Buffer_.clear();
		BufferCount_ = 0;
	}

	void SetFlushInterval(OaI32 InN) { FlushInterval_ = InN; }

	// Operators.
	OaLogMetrics(const OaLogMetrics&) = delete;
	OaLogMetrics& operator=(const OaLogMetrics&) = delete;
	OaLogMetrics(OaLogMetrics&&) = delete;
	OaLogMetrics& operator=(OaLogMetrics&&) = delete;

private:
	OaString LogDir_;
	OaString EventsPath_;
	OaTimestamp StartTime_;
	std::mutex Mutex_;
	bool IsOpen_ = false;
	OaString Buffer_;
	OaI32 BufferCount_ = 0;
	OaI32 FlushInterval_ = 16;
};

using OaTrainingLogger = OaLogMetrics;

