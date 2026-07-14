#include <Oa/Core/Validation.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifndef NDEBUG
#include <mutex>
#include <unordered_map>
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Internal state
// ─────────────────────────────────────────────────────────────────────────────

namespace {

struct OaValidationState {
#ifndef NDEBUG
	std::atomic<bool>                       Enabled{true};
	std::atomic<OaU8>                       MinSev{static_cast<OaU8>(OaValidationSeverity::Verbose)};
#else
	std::atomic<bool>                       Enabled{false};
	std::atomic<OaU8>                       MinSev{static_cast<OaU8>(OaValidationSeverity::Warning)};
#endif
	std::atomic<OaValidation::Callback>     Cb{nullptr};

#ifndef NDEBUG
	// Counter storage — debug only
	std::mutex                              CounterMtx;
	std::unordered_map<std::string, OaU64> Counters;
#endif
};

OaValidationState& State() {
	static OaValidationState s;
	return s;
}

// Severity → OaLogLevel mapping
OaLogLevel SevToLogLevel(OaValidationSeverity InSev) {
	switch (InSev) {
		case OaValidationSeverity::Verbose: return OaLogLevel::Trace;
		case OaValidationSeverity::Info:    return OaLogLevel::Debug;
		case OaValidationSeverity::Warning: return OaLogLevel::Warn;
		case OaValidationSeverity::Error:   return OaLogLevel::Error;
		case OaValidationSeverity::Fatal:   return OaLogLevel::Fatal;
		default:                            return OaLogLevel::Error;
	}
}

// OaStatusCode to use for Error/Fatal returns
OaStatusCode SevToStatusCode(OaValidationSeverity InSev) {
	switch (InSev) {
		case OaValidationSeverity::Fatal: return OaStatusCode::Internal;
		default:                          return OaStatusCode::InvalidArgument;
	}
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// OaValidation — Enable / Severity / Callback
// ─────────────────────────────────────────────────────────────────────────────

void OaValidation::Enable(bool InEnable) {
	State().Enabled.store(InEnable, std::memory_order_relaxed);
}

bool OaValidation::IsEnabled() {
	return State().Enabled.load(std::memory_order_relaxed);
}

void OaValidation::InitFromEnv() {
	// OA_VALIDATION=0 disables even in debug; =1 enables in release.
	const char* envEnable = std::getenv("OA_VALIDATION");
	if (envEnable != nullptr) {
		Enable(std::strcmp(envEnable, "0") != 0);
	}

	// OA_VALIDATION_SEVERITY=verbose|info|warning|error|fatal
	const char* envSev = std::getenv("OA_VALIDATION_SEVERITY");
	if (envSev != nullptr) {
		OaValidationSeverity sev = OaValidationSeverity::Verbose;
		if (std::strcmp(envSev, "info")    == 0) { sev = OaValidationSeverity::Info; }
		if (std::strcmp(envSev, "warning") == 0) { sev = OaValidationSeverity::Warning; }
		if (std::strcmp(envSev, "error")   == 0) { sev = OaValidationSeverity::Error; }
		if (std::strcmp(envSev, "fatal")   == 0) { sev = OaValidationSeverity::Fatal; }
		SetMinSeverity(sev);
	}
}

void OaValidation::SetMinSeverity(OaValidationSeverity InSev) {
	State().MinSev.store(static_cast<OaU8>(InSev), std::memory_order_relaxed);
}

OaValidationSeverity OaValidation::GetMinSeverity() {
	return static_cast<OaValidationSeverity>(
		State().MinSev.load(std::memory_order_relaxed));
}

void OaValidation::SetCallback(Callback InCb) {
	State().Cb.store(InCb, std::memory_order_relaxed);
}

// ─────────────────────────────────────────────────────────────────────────────
// OaValidation::Report
// ─────────────────────────────────────────────────────────────────────────────

OaStatus OaValidation::Report(
	OaValidationSeverity InSev,
	OaLogComponent       InComp,
	const char*          InFmt, ...)
{
	const auto minSev = static_cast<OaValidationSeverity>(
		State().MinSev.load(std::memory_order_relaxed));
	if (static_cast<OaU8>(InSev) < static_cast<OaU8>(minSev)) {
		return OaStatus::Ok();
	}

	OaArray<char, 1024> buf{};
	va_list args;
	va_start(args, InFmt);
	std::vsnprintf(buf.data(), buf.size(), InFmt, args);
	va_end(args);

	const auto cb = State().Cb.load(std::memory_order_relaxed);
	if (cb != nullptr) {
		cb(InSev, InComp, buf.data());
	} else {
		OaLog::Instance().Log(SevToLogLevel(InSev), InComp,
			"[OaValidation::%s] %s", OaValidationSeverityName(InSev), buf.data());
	}

	if (static_cast<OaU8>(InSev) >= static_cast<OaU8>(OaValidationSeverity::Error)) {
		return OaStatus::Error(SevToStatusCode(InSev), OaString(buf.data()));
	}
	return OaStatus::Ok();
}

// ─────────────────────────────────────────────────────────────────────────────
// Debug counters (compiled in debug builds only)
// ─────────────────────────────────────────────────────────────────────────────

#ifndef NDEBUG

void OaValidation::IncrCounter(const char* InName) {
	auto& st = State();
	std::lock_guard<std::mutex> lock(st.CounterMtx);
	st.Counters[InName]++;
}

void OaValidation::IncrCounterNamed(const char* InName) {
	auto& st = State();
	std::lock_guard<std::mutex> lock(st.CounterMtx);
	st.Counters[InName]++;
}

OaU64 OaValidation::GetCounter(const char* InName) {
	auto& st = State();
	std::lock_guard<std::mutex> lock(st.CounterMtx);
	auto it = st.Counters.find(InName);
	return it != st.Counters.end() ? it->second : OaU64(0);
}

void OaValidation::ResetCounters() {
	auto& st = State();
	std::lock_guard<std::mutex> lock(st.CounterMtx);
	st.Counters.clear();
}

void OaValidation::DumpCounters(OaLogComponent InComp) {
	auto& st = State();
	std::lock_guard<std::mutex> lock(st.CounterMtx);
	if (st.Counters.empty()) {
		OA_LOG_DEBUG(InComp, "OaValidation::DumpCounters: (no counters recorded)");
		return;
	}
	OA_LOG_DEBUG(InComp, "OaDebugSession(");
	for (const auto& [name, value] : st.Counters) {
		if (value > 0) {
			OA_LOG_DEBUG(InComp, "  %-36s %llu", name.c_str(), static_cast<unsigned long long>(value));
		}
	}
	OA_LOG_DEBUG(InComp, ")");
}

#else

// Release stubs — always return 0 / no-op.
void  OaValidation::IncrCounter(const char* /*InName*/)        {}
void  OaValidation::IncrCounterNamed(const char* /*InName*/)   {}
OaU64 OaValidation::GetCounter(const char* /*InName*/)         { return OaU64(0); }
void  OaValidation::ResetCounters()                            {}
void  OaValidation::DumpCounters(OaLogComponent /*InComp*/)    {}

#endif
