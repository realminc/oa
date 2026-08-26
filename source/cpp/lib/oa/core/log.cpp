#include <oa/core/log.h>
#include "logAccess.h"

#include <stdio.h>
#include <time.h>

namespace {

// main-thread TLS is destroyed before ordinary static storage. A static
// oa::Engine may therefore close after its selected logger weak reference has
// already been destroyed. Keep a trivially destructible availability flag so
// shutdown-time logging can fall back to stderr without touching dead TLS.
thread_local oa::Bool threadLogSelectionAvailable = true;

struct ThreadLogSelection {
	oa::WeakPtr<oa::LogImpl> state;

	~ThreadLogSelection() {
		threadLogSelectionAvailable = false;
	}
};

thread_local ThreadLogSelection selectedLog;

[[nodiscard]] const char* levelName(oa::LogLevel inLevel) noexcept {
	switch (inLevel) {
		case oa::LogLevel::Trace: return "TRACE";
		case oa::LogLevel::Debug: return "DEBUG";
		case oa::LogLevel::Info:  return "INFO ";
		case oa::LogLevel::Warn:  return "WARN ";
		case oa::LogLevel::Error: return "ERROR";
		case oa::LogLevel::Fatal: return "FATAL";
		default:                return "?????";
	}
}

[[nodiscard]] const char* levelColor(oa::LogLevel inLevel) noexcept {
	switch (inLevel) {
		case oa::LogLevel::Trace: return "\033[90m";
		case oa::LogLevel::Debug: return "\033[36m";
		case oa::LogLevel::Info:  return "\033[32m";
		case oa::LogLevel::Warn:  return "\033[33m";
		case oa::LogLevel::Error: return "\033[31m";
		case oa::LogLevel::Fatal: return "\033[35m";
		default:                return "";
	}
}

[[nodiscard]] bool localTime(::time_t inTime, ::tm& outTime) noexcept {
#if defined(_WIN32)
	return localtime_s(&outTime, &inTime) == 0;
#else
	return localtime_r(&inTime, &outTime) != nullptr;
#endif
}

void formatLogTimestamp(char* outText, oa::Usize inCapacity, long& outMillis) noexcept {
	const oa::I64 epochNanoseconds = oa::systemNow().nanosecondsSinceEpoch();
	outMillis = static_cast<long>((epochNanoseconds / 1'000'000LL) % 1000LL);
	const ::time_t wall = static_cast<::time_t>(epochNanoseconds / 1'000'000'000LL);
	::tm local{};
	if (localTime(wall, local)) {
		(void)::strftime(outText, inCapacity, "%H:%M:%S", &local);
	} else if (inCapacity != 0U) {
		(void)::snprintf(outText, inCapacity, "00:00:00");
	}
}

void fallbackWrite(
	oa::LogLevel inLevel,
	oa::LogComponent inComponent,
	const char* inFormat,
	va_list inArgs) noexcept
{
	char message[4096]{};
	if (inFormat != nullptr) {
		(void)::vsnprintf(message, sizeof(message), inFormat, inArgs);
	}
	char timestamp[32]{};
	long millis = 0;
	formatLogTimestamp(timestamp, sizeof(timestamp), millis);
	(void)::fprintf(stderr, "%s%s.%03ld [%s] [%s] %s\033[0m\n",
		levelColor(inLevel), timestamp, millis, levelName(inLevel),
		inComponent.cStr(), message);
}

} // namespace

namespace oa {

class LogImpl {
public:
	oa::Mutex mutex;
	::FILE* file = nullptr;
	oa::String path;
	oa::Atomic<oa::U8> minimumLevel{
		static_cast<oa::U8>(oa::LogLevel::Info)};
	oa::Bool consoleOutput = true;
	oa::Bool open = true;
	oa::Status firstError = oa::Status::ok();

	[[nodiscard]] oa::Status writeV(
		oa::LogLevel inLevel,
		oa::LogComponent inComponent,
		const char* inFormat,
		va_list inArgs)
	{
		if (static_cast<oa::U8>(inLevel)
			< minimumLevel.load(oa::MemoryOrder::Relaxed)
			or inLevel == oa::LogLevel::Off) {
			return oa::Status::ok();
		}
		char message[4096]{};
		if (inFormat != nullptr) {
			(void)::vsnprintf(message, sizeof(message), inFormat, inArgs);
		}
		char timestamp[32]{};
		long millis = 0;
		formatLogTimestamp(timestamp, sizeof(timestamp), millis);

		oa::ScopedLock<oa::Mutex> lock(mutex);
		if (not open) {
			return oa::Status::error(oa::StatusCode::FailedPrecondition,
				"oa::Log::write called after Close");
		}
		if (file != nullptr) {
			if (::fprintf(file, "%s.%03ld [%s] [%s] %s\n",
				timestamp, millis, levelName(inLevel), inComponent.cStr(), message) < 0
				and firstError.isOk()) {
				firstError = oa::Status::error(oa::StatusCode::Internal,
					"log file write failed");
			}
		}
		if (consoleOutput) {
			(void)::fprintf(stderr, "%s%s.%03ld [%s] [%s] %s\033[0m\n",
				levelColor(inLevel), timestamp, millis, levelName(inLevel),
				inComponent.cStr(), message);
		}
		return firstError;
	}
};

} // namespace oa

oa::Log::Log(oa::SharedPtr<oa::LogImpl> inImpl) noexcept
	: impl_(oa::move(inImpl)) {}

oa::Log::~Log() {
	(void)close();
}

oa::Result<oa::UniquePtr<oa::Log>> oa::Log::create(const oa::LogOptions& inOptions) {
	auto impl = oa::makeShared<oa::LogImpl>();
	impl->minimumLevel.store(
		static_cast<oa::U8>(inOptions.minimumLevel), oa::MemoryOrder::Relaxed);
	impl->consoleOutput = inOptions.consoleOutput;

	if (inOptions.fileOutput) {
		if (inOptions.directory.empty()) {
			return oa::Status::invalidArgument(
				"oa::Log file output requires a directory");
		}
		const oa::Path directory(inOptions.directory);
		if (const oa::Status status = oa::Filesystem::createDirectories(directory);
			status.isError()) {
			return status;
		}

		const ::time_t wall = static_cast<::time_t>(
			oa::systemNow().nanosecondsSinceEpoch() / 1'000'000'000LL);
		::tm local{};
		char date[16] = "unknown";
		if (localTime(wall, local)) {
			(void)::strftime(date, sizeof(date), "%Y%m%d", &local);
		}
		oa::String filename = inOptions.prefix.empty() ? oa::String("oa") : inOptions.prefix;
		filename += "_";
		filename += date;
		filename += ".log";
		impl->path = (directory / filename.view()).string();
		impl->file = ::fopen(impl->path.cStr(), "a");
		if (impl->file == nullptr) {
			return oa::Status::error(oa::StatusCode::PermissionError,
				oa::String("cannot open log file: ") + impl->path);
		}
	}

	return oa::UniquePtr<oa::Log>(new oa::Log(oa::move(impl)));
}

oa::Status oa::Log::write(
	oa::LogLevel inLevel,
	oa::LogComponent inComponent,
	const char* inFormat,
	...)
{
	va_list args;
	va_start(args, inFormat);
	const oa::Status status = writeV(inLevel, inComponent, inFormat, args);
	va_end(args);
	return status;
}

oa::Status oa::Log::writeV(
	oa::LogLevel inLevel,
	oa::LogComponent inComponent,
	const char* inFormat,
	va_list inArgs)
{
	if (not impl_) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::Log has no state");
	}
	return impl_->writeV(inLevel, inComponent, inFormat, inArgs);
}

oa::Status oa::Log::flush() {
	if (not impl_) return oa::Status::ok();
	oa::ScopedLock<oa::Mutex> lock(impl_->mutex);
	if (impl_->file != nullptr) {
		if (::fflush(impl_->file) != 0 and impl_->firstError.isOk()) {
			impl_->firstError = oa::Status::error(oa::StatusCode::Internal,
				"log file flush failed");
		}
	}
	return impl_->firstError;
}

oa::Status oa::Log::close() {
	if (not impl_) return oa::Status::ok();
	oa::ScopedLock<oa::Mutex> lock(impl_->mutex);
	if (not impl_->open) return impl_->firstError;
	if (impl_->file != nullptr) {
		if (::fflush(impl_->file) != 0 and impl_->firstError.isOk()) {
			impl_->firstError = oa::Status::error(oa::StatusCode::Internal,
				"log file flush failed during Close");
		}
		if (::fclose(impl_->file) != 0 and impl_->firstError.isOk()) {
			impl_->firstError = oa::Status::error(oa::StatusCode::Internal,
				"log file close failed");
		}
		impl_->file = nullptr;
	}
	impl_->open = false;
	return impl_->firstError;
}

void oa::Log::setLevel(oa::LogLevel inLevel) noexcept {
	if (impl_) {
		impl_->minimumLevel.store(
			static_cast<oa::U8>(inLevel), oa::MemoryOrder::Relaxed);
	}
}

oa::LogLevel oa::Log::getLevel() const noexcept {
	return impl_
		? static_cast<oa::LogLevel>(
			impl_->minimumLevel.load(oa::MemoryOrder::Relaxed))
		: oa::LogLevel::Off;
}

oa::String oa::Log::getLogPath() const {
	if (not impl_) return {};
	oa::ScopedLock<oa::Mutex> lock(impl_->mutex);
	return impl_->path;
}

oa::Bool oa::Log::isOpen() const noexcept {
	if (not impl_) return false;
	oa::ScopedLock<oa::Mutex> lock(impl_->mutex);
	return impl_->open;
}

oa::LogSelection oa::LogAccess::select(oa::Log* inLog) noexcept {
	if (not threadLogSelectionAvailable) return {};
	oa::LogSelection previous{.state = selectedLog.state};
	selectedLog.state = inLog
		? oa::WeakPtr<oa::LogImpl>(inLog->impl_)
		: oa::WeakPtr<oa::LogImpl>{};
	return previous;
}

void oa::LogAccess::restore(const oa::LogSelection& inSelection) noexcept {
	if (threadLogSelectionAvailable) selectedLog.state = inSelection.state;
}

void oa::LogAccess::restoreIfCurrent(
	oa::Log* inExpected,
	const oa::LogSelection& inSelection) noexcept
{
	if (not threadLogSelectionAvailable) return;
	const auto current = selectedLog.state.lock();
	if (inExpected != nullptr and current.get() == inExpected->impl_.get()) {
		restore(inSelection);
	}
}

oa::SharedPtr<oa::LogImpl> oa::LogAccess::current() noexcept {
	return threadLogSelectionAvailable
		? selectedLog.state.lock()
		: oa::SharedPtr<oa::LogImpl>{};
}

oa::LogSelection oa::LogAccess::currentSelection() noexcept {
	return threadLogSelectionAvailable
		? oa::LogSelection{.state = selectedLog.state}
		: oa::LogSelection{};
}

void oa::logWrite(
	oa::LogLevel inLevel,
	oa::LogComponent inComponent,
	const char* inFormat,
	...)
{
	va_list args;
	va_start(args, inFormat);
	if (auto impl = oa::LogAccess::current()) {
		(void)impl->writeV(inLevel, inComponent, inFormat, args);
	} else {
		fallbackWrite(inLevel, inComponent, inFormat, args);
	}
	va_end(args);
}
