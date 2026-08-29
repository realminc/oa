// OA Validation — vulkan-style debug layer for Realm compute.
//
// phase 1: macro infrastructure + Validation class.
//
// Build model:
//   !NDEBUG (debug)              : OA_VALIDATE always active; counters compiled in
//   NDEBUG  (release)            : OA_VALIDATE compiled out — zero binary size
//   NDEBUG + OA_ENABLE_VALIDATION: OA_VALIDATE compiled in, runtime-gated by isEnabled()
//
// OaLogDebug and OA_DEBUG_COUNTER_INC are always compiled out in NDEBUG.

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/core/log.h>

namespace oa {

// ─────────────────────────────────────────────────────────────────────────────
// Severity
// ─────────────────────────────────────────────────────────────────────────────

enum class ValidationSeverity : oa::U8 {
	Verbose = 0,   // internal state dumps, opt-in
	Info    = 1,   // routing decisions, kernel selection, shape reports
	Warning = 2,   // performance issues (naive fallback, misaligned tiles)
	Error   = 3,   // contract violations producing wrong results
	Fatal   = 4,   // invariant broken, cannot continue safely
};

[[nodiscard]] constexpr const char* validationSeverityName(ValidationSeverity inSev) noexcept {
	switch (inSev) {
		case ValidationSeverity::Verbose: return "VERBOSE";
		case ValidationSeverity::Info:    return "INFO";
		case ValidationSeverity::Warning: return "WARNING";
		case ValidationSeverity::Error:   return "ERROR";
		case ValidationSeverity::Fatal:   return "FATAL";
		default:                          return "?";
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation — global validation controller
// ─────────────────────────────────────────────────────────────────────────────

class Validation {
public:
	using Callback = void(*)(ValidationSeverity, oa::LogComponent, const char*);

	// Enable / disable. Debug: on by default. release (OA_ENABLE_VALIDATION): off by default.
	// call initFromEnv() once at startup to apply OA_VALIDATION / OA_VALIDATION_SEVERITY.
	static void enable(bool inEnable = true);
	[[nodiscard]] static bool isEnabled();

	// Read OA_VALIDATION and OA_VALIDATION_SEVERITY from the process environment.
	// call once before any OA_VALIDATE macro fires (e.g. Engine::create).
	static void initFromEnv();

	// Override severity filter (default: Verbose in debug, warning in release).
	static void setMinSeverity(ValidationSeverity inSev);
	[[nodiscard]] static ValidationSeverity getMinSeverity();

	// Optional custom callback; default writes via oa::Log*.
	static void setCallback(Callback inCb);

	// Internal: called by OA_VALIDATE macros. Formats message, calls callback.
	// Returns oa::Status::error on Error/Fatal severity; ok on lower severity.
	[[nodiscard]] static oa::Status report(
		ValidationSeverity inSev,
		oa::LogComponent   inComp,
		const char*        inFmt, ...) __attribute__((format(printf, 3, 4)));

	// ── Debug counter API (debug builds only; always returns 0 in release) ──

	// Increment a named counter (thread-safe, lock-free on the hot path).
	static void incrCounter(const char* inName);

	// Increment using a runtime string variable (for kernel names, etc.).
	static void incrCounterNamed(const char* inName);

	// Returns current counter value (0 if never incremented or in release).
	[[nodiscard]] static oa::U64 getCounter(const char* inName);

	// reset all counters to 0.
	static void resetCounters();

	// log all non-zero counters at Debug level.
	static void dumpCounters(oa::LogComponent inComp = oa::LogComponent::Core);
};

} // namespace oa

// ─────────────────────────────────────────────────────────────────────────────
// Macro: OA_VALIDATE — core validation primitive
// ─────────────────────────────────────────────────────────────────────────────
//
// usage (in functions returning oa::Status):
//   OA_VALIDATE(M > 0, oa::ValidationSeverity::Error, oa::LogComponent::Core,
//       "Gemm: M must be > 0, got %u", M);
//
// - Error / Fatal severity: logs the message and returns oa::Status::error.
// - Warning / Info / Verbose: logs the message, does not return.
// - Fatal: also calls OA_ASSERT(false) after logging.

#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)

#define OA_VALIDATE(cond_, sev_, comp_, ...)                                      \
	do {                                                                          \
		if (oa::Validation::isEnabled() and not (cond_)) {                        \
			auto _oa_val_st = oa::Validation::report((sev_), (comp_), __VA_ARGS__); \
			if (static_cast<oa::U8>(sev_) >= static_cast<oa::U8>(oa::ValidationSeverity::Fatal)) { \
				OA_ASSERT(false);                                                 \
			}                                                                     \
			if (static_cast<oa::U8>(sev_) >= static_cast<oa::U8>(oa::ValidationSeverity::Error)) { \
				return _oa_val_st;                                                \
			}                                                                     \
		}                                                                         \
	} while (0)

#else

#define OA_VALIDATE(cond_, sev_, comp_, ...) ((void)0)

#endif

// ─────────────────────────────────────────────────────────────────────────────
// Convenience validation macros (all forward to OA_VALIDATE)
// ─────────────────────────────────────────────────────────────────────────────

// Null pointer check.
#define OA_VALIDATE_NOT_NULL(ptr_, ctx_)                                         \
	OA_VALIDATE((ptr_) != nullptr, oa::ValidationSeverity::Error,                \
		oa::LogComponent::Core, "%s: null pointer", (ctx_))

// Array / tensor index bounds (index < limit).
#define OA_VALIDATE_BOUNDS(idx_, limit_, ctx_)                                   \
	OA_VALIDATE(static_cast<oa::I64>(idx_) >= 0 and                                \
		static_cast<oa::I64>(idx_) < static_cast<oa::I64>(limit_),                   \
		oa::ValidationSeverity::Error, oa::LogComponent::Core,                      \
		"%s: index %lld out of bounds [0, %lld)", (ctx_),                         \
		static_cast<oa::I64>(idx_), static_cast<oa::I64>(limit_))

// alignment check (value % align == 0).
#define OA_VALIDATE_ALIGNMENT(val_, align_, ctx_)                                \
	OA_VALIDATE((static_cast<oa::U64>(val_) % static_cast<oa::U64>(align_)) == 0,   \
		oa::ValidationSeverity::Error, oa::LogComponent::Core,                      \
		"%s: value %llu not aligned to %llu bytes", (ctx_),                       \
		static_cast<oa::U64>(val_), static_cast<oa::U64>(align_))

// Push constant size: actual must exactly equal declared shader struct size.
#define OA_VALIDATE_PUSH_SIZE(actual_, declared_, kernel_)                       \
	OA_VALIDATE((actual_) == (declared_), oa::ValidationSeverity::Error,          \
		oa::LogComponent::Compute,                                                   \
		"Dispatch '%s': push constant size %zu != declared %zu bytes",             \
		(kernel_), static_cast<oa::Usize>(actual_), static_cast<oa::Usize>(declared_))

// Buffer count check.
#define OA_VALIDATE_BUFFER_COUNT(actual_, expected_, kernel_)                    \
	OA_VALIDATE((actual_) == (expected_), oa::ValidationSeverity::Error,          \
		oa::LogComponent::Compute,                                                   \
		"Dispatch '%s': buffer count %zu != expected %zu",                         \
		(kernel_), static_cast<oa::Usize>(actual_), static_cast<oa::Usize>(expected_))

// Matrix dtype match (requires getDtype() and oa::scalarTypeName()).
#define OA_VALIDATE_DTYPE(a_, b_, ctx_)                                          \
	OA_VALIDATE((a_).getDtype() == (b_).getDtype(),                              \
		oa::ValidationSeverity::Error, oa::LogComponent::Compute,                   \
		"%s: dtype mismatch — %s vs %s", (ctx_),                                 \
		oa::scalarTypeName((a_).getDtype()).data(),                                  \
		oa::scalarTypeName((b_).getDtype()).data())

// exact dtype check (requires getDtype() and oa::scalarTypeName()).
#define OA_VALIDATE_DTYPE_EXACT(mat_, expected_, ctx_)                           \
	OA_VALIDATE((mat_).getDtype() == (expected_),                                 \
		oa::ValidationSeverity::Error, oa::LogComponent::Compute,                   \
		"%s: expected dtype %s, got %s", (ctx_),                                  \
		oa::scalarTypeName(expected_).data(),                                        \
		oa::scalarTypeName((mat_).getDtype()).data())

// Matrix multiplication shape compatibility: A is [M,K], B is [N,K] (weight-transposed).
// checks inA.size(-1) == inB.size(-1).
#define OA_VALIDATE_SHAPE_COMPAT(a_, b_, op_)                                    \
	OA_VALIDATE((a_).rank() >= 2 and (b_).rank() == 2 and                        \
		(a_).size(-1) == (b_).size(-1),                                           \
		oa::ValidationSeverity::Error, oa::LogComponent::Compute,                   \
		"%s: shape mismatch — A.k=%lld B.k=%lld (A rank=%d B rank=%d)", (op_),   \
		(a_).rank() >= 2 ? (a_).size(-1) : -1LL,                                 \
		(b_).rank() >= 2 ? (b_).size(-1) : -1LL,                                 \
		(a_).rank(), (b_).rank())

// Heap slot registration check (requires heapSlot() >= 0 and isOnDevice()).
#define OA_VALIDATE_HEAP_SLOT(buf_, ctx_)                                        \
	OA_VALIDATE((buf_).heapSlot() >= 0, oa::ValidationSeverity::Error,           \
		oa::LogComponent::Runtime,                                                  \
		"%s: buffer not registered with bindless heap (heapSlot=%d)", (ctx_),     \
		(buf_).heapSlot())

// ─────────────────────────────────────────────────────────────────────────────
// OA_WARN_PERF — performance warning (no return; warning severity)
// ─────────────────────────────────────────────────────────────────────────────

#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)

#define OA_WARN_PERF(cond_, ...)                                                 \
	do {                                                                         \
		if (oa::Validation::isEnabled() and (cond_)) {                           \
			(void)oa::Validation::report(oa::ValidationSeverity::Warning,        \
				oa::LogComponent::Core, __VA_ARGS__);                              \
		}                                                                        \
	} while (0)

#else

#define OA_WARN_PERF(cond_, ...) ((void)0)

#endif

// ─────────────────────────────────────────────────────────────────────────────
// OA_DEBUG_COUNTER_INC / OA_DEBUG_COUNTER_GET — always compiled out in release
// ─────────────────────────────────────────────────────────────────────────────

#ifndef NDEBUG

#define OA_DEBUG_COUNTER_INC(name_)          oa::Validation::incrCounter(#name_)
#define OA_DEBUG_COUNTER_INC_NAMED(strExpr_) oa::Validation::incrCounterNamed(strExpr_)
#define OA_DEBUG_COUNTER_GET(name_)          oa::Validation::getCounter(#name_)

#else

#define OA_DEBUG_COUNTER_INC(name_)          ((void)0)
#define OA_DEBUG_COUNTER_INC_NAMED(strExpr_) ((void)0)
#define OA_DEBUG_COUNTER_GET(name_)          (oa::U64(0))

#endif
