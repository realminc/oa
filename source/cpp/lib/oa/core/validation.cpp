#include <oa/core/validation.h>
#include <oa/core/log.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

// ─────────────────────────────────────────────────────────────────────────────
// Internal state
// ─────────────────────────────────────────────────────────────────────────────

namespace {

struct ValidationState {
#ifndef NDEBUG
	oa::Atomic<oa::Bool> enabled{true};
	oa::Atomic<oa::U8> minSev{
		static_cast<oa::U8>(oa::ValidationSeverity::Verbose)};
#else
	oa::Atomic<oa::Bool> enabled{false};
	oa::Atomic<oa::U8> minSev{
		static_cast<oa::U8>(oa::ValidationSeverity::Warning)};
#endif
	oa::Atomic<oa::Validation::Callback> callback{nullptr};

#ifndef NDEBUG
	oa::Mutex counterMutex;
	oa::HashMap<oa::String, oa::U64> counters;
#endif
};

ValidationState& state() {
	static ValidationState s;
	return s;
}

#ifndef NDEBUG
void incrementCounter(ValidationState& inState, const char* inName) {
	oa::ScopedLock<oa::Mutex> lock(inState.counterMutex);
	oa::String name(inName);
	auto counter = inState.counters.find(name);
	if (counter == inState.counters.end()) {
		inState.counters.emplace(oa::move(name), oa::U64(1));
	} else {
		++counter->second;
	}
}
#endif

// Severity → oa::LogLevel mapping
oa::LogLevel sevToLogLevel(oa::ValidationSeverity inSev) {
	switch (inSev) {
		case oa::ValidationSeverity::Verbose: return oa::LogLevel::Trace;
		case oa::ValidationSeverity::Info:    return oa::LogLevel::Debug;
		case oa::ValidationSeverity::Warning: return oa::LogLevel::Warn;
		case oa::ValidationSeverity::Error:   return oa::LogLevel::Error;
		case oa::ValidationSeverity::Fatal:   return oa::LogLevel::Fatal;
		default:                            return oa::LogLevel::Error;
	}
}

// oa::StatusCode to use for Error/Fatal returns
oa::StatusCode sevToStatusCode(oa::ValidationSeverity inSev) {
	switch (inSev) {
		case oa::ValidationSeverity::Fatal: return oa::StatusCode::Internal;
		default:                          return oa::StatusCode::InvalidArgument;
	}
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// oa::Validation — Enable / Severity / Callback
// ─────────────────────────────────────────────────────────────────────────────

void oa::Validation::enable(bool inEnable) {
	state().enabled.store(inEnable, oa::MemoryOrder::Relaxed);
}

bool oa::Validation::isEnabled() {
	return state().enabled.load(oa::MemoryOrder::Relaxed);
}

void oa::Validation::initFromEnv() {
	// OA_VALIDATION=0 disables even in debug; =1 enables in release.
	const char* envEnable = ::getenv("OA_VALIDATION");
	if (envEnable != nullptr) {
		enable(oa::strcmp(envEnable, "0") != 0);
	}

	// OA_VALIDATION_SEVERITY=verbose|info|warning|error|fatal
	const char* envSev = ::getenv("OA_VALIDATION_SEVERITY");
	if (envSev != nullptr) {
		oa::ValidationSeverity sev = oa::ValidationSeverity::Verbose;
		if (oa::strcmp(envSev, "info") == 0) sev = oa::ValidationSeverity::Info;
		if (oa::strcmp(envSev, "warning") == 0) sev = oa::ValidationSeverity::Warning;
		if (oa::strcmp(envSev, "error") == 0) sev = oa::ValidationSeverity::Error;
		if (oa::strcmp(envSev, "fatal") == 0) sev = oa::ValidationSeverity::Fatal;
		setMinSeverity(sev);
	}
}

void oa::Validation::setMinSeverity(oa::ValidationSeverity inSev) {
	state().minSev.store(
		static_cast<oa::U8>(inSev), oa::MemoryOrder::Relaxed);
}

oa::ValidationSeverity oa::Validation::getMinSeverity() {
	return static_cast<oa::ValidationSeverity>(
		state().minSev.load(oa::MemoryOrder::Relaxed));
}

void oa::Validation::setCallback(Callback inCb) {
	state().callback.store(inCb, oa::MemoryOrder::Relaxed);
}

// ─────────────────────────────────────────────────────────────────────────────
// oa::Validation::Report
// ─────────────────────────────────────────────────────────────────────────────

oa::Status oa::Validation::report(
	oa::ValidationSeverity inSev,
	oa::LogComponent       inComp,
	const char*          inFmt, ...)
{
	const auto minSev = static_cast<oa::ValidationSeverity>(
		state().minSev.load(oa::MemoryOrder::Relaxed));
	if (static_cast<oa::U8>(inSev) < static_cast<oa::U8>(minSev)) {
		return oa::Status::ok();
	}

	oa::Array<char, 1024> buf{};
	va_list args;
	va_start(args, inFmt);
	::vsnprintf(buf.data(), buf.size(), inFmt, args);
	va_end(args);

	const auto cb = state().callback.load(oa::MemoryOrder::Relaxed);
	if (cb != nullptr) {
		cb(inSev, inComp, buf.data());
	} else {
		oa::logWrite(sevToLogLevel(inSev), inComp,
			"[oa::Validation::%s] %s", oa::validationSeverityName(inSev), buf.data());
	}

	if (static_cast<oa::U8>(inSev) >= static_cast<oa::U8>(oa::ValidationSeverity::Error)) {
		return oa::Status::error(sevToStatusCode(inSev), oa::String(buf.data()));
	}
	return oa::Status::ok();
}

// ─────────────────────────────────────────────────────────────────────────────
// Debug counters (compiled in debug builds only)
// ─────────────────────────────────────────────────────────────────────────────

#ifndef NDEBUG

void oa::Validation::incrCounter(const char* inName) {
	incrementCounter(state(), inName);
}

void oa::Validation::incrCounterNamed(const char* inName) {
	incrementCounter(state(), inName);
}

oa::U64 oa::Validation::getCounter(const char* inName) {
	auto& st = state();
	oa::ScopedLock<oa::Mutex> lock(st.counterMutex);
	auto it = st.counters.find(inName);
	return it != st.counters.end() ? it->second : oa::U64(0);
}

void oa::Validation::resetCounters() {
	auto& st = state();
	oa::ScopedLock<oa::Mutex> lock(st.counterMutex);
	st.counters.clear();
}

void oa::Validation::dumpCounters(oa::LogComponent inComp) {
	auto& st = state();
	oa::ScopedLock<oa::Mutex> lock(st.counterMutex);
	if (st.counters.empty()) {
		OaLogDebug(inComp, "oa::Validation::dumpCounters: (no counters recorded)");
		return;
	}
	OaLogDebug(inComp, "DebugSession(");
	for (const auto& [name, value] : st.counters) {
		if (value > 0) {
			OaLogDebug(inComp, "  {:<36} {}", name.cStr(),
				static_cast<unsigned long long>(value));
		}
	}
	OaLogDebug(inComp, ")");
}

#else

// release stubs — always return 0 / no-op.
void  oa::Validation::incrCounter(const char* /*inName*/)        {}
void  oa::Validation::incrCounterNamed(const char* /*inName*/)   {}
oa::U64 oa::Validation::getCounter(const char* /*inName*/)         { return oa::U64(0); }
void  oa::Validation::resetCounters()                            {}
void  oa::Validation::dumpCounters(oa::LogComponent /*inComp*/)    {}

#endif
