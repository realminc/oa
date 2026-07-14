#pragma once

// OaValidation — Vulkan-style debug layer for Realm compute.
//
// Phase 1: macro infrastructure + OaValidation class.
// Design: oa/Docs/OaValidation.md
//
// Build model:
//   !NDEBUG (debug)              : OA_VALIDATE always active; counters compiled in
//   NDEBUG  (release)            : OA_VALIDATE compiled out — zero binary size
//   NDEBUG + OA_ENABLE_VALIDATION: OA_VALIDATE compiled in, runtime-gated by IsEnabled()
//
// OA_LOG_DEBUG and OA_DEBUG_COUNTER_INC are always compiled out in NDEBUG.

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Log.h>

// ─────────────────────────────────────────────────────────────────────────────
// Severity
// ─────────────────────────────────────────────────────────────────────────────

enum class OaValidationSeverity : OaU8 {
	Verbose = 0,   // internal state dumps, opt-in
	Info    = 1,   // routing decisions, kernel selection, shape reports
	Warning = 2,   // performance issues (Naive fallback, misaligned tiles)
	Error   = 3,   // contract violations producing wrong results
	Fatal   = 4,   // invariant broken, cannot continue safely
};

[[nodiscard]] constexpr const char* OaValidationSeverityName(OaValidationSeverity InSev) noexcept {
	switch (InSev) {
		case OaValidationSeverity::Verbose: return "VERBOSE";
		case OaValidationSeverity::Info:    return "INFO";
		case OaValidationSeverity::Warning: return "WARNING";
		case OaValidationSeverity::Error:   return "ERROR";
		case OaValidationSeverity::Fatal:   return "FATAL";
		default:                            return "?";
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// OaValidation — global validation controller
// ─────────────────────────────────────────────────────────────────────────────

class OaValidation {
public:
	using Callback = void(*)(OaValidationSeverity, OaLogComponent, const char*);

	// Enable / disable. Debug: on by default. Release (OA_ENABLE_VALIDATION): off by default.
	// Call InitFromEnv() once at startup to apply OA_VALIDATION / OA_VALIDATION_SEVERITY.
	static void Enable(bool InEnable = true);
	[[nodiscard]] static bool IsEnabled();

	// Read OA_VALIDATION and OA_VALIDATION_SEVERITY from the process environment.
	// Call once before any OA_VALIDATE macro fires (e.g. OaEngine::Create).
	static void InitFromEnv();

	// Override severity filter (default: Verbose in debug, Warning in release).
	static void SetMinSeverity(OaValidationSeverity InSev);
	[[nodiscard]] static OaValidationSeverity GetMinSeverity();

	// Optional custom callback; default writes via OA_LOG_*.
	static void SetCallback(Callback InCb);

	// Internal: called by OA_VALIDATE macros. Formats message, calls callback.
	// Returns OaStatus::Error on Error/Fatal severity; Ok on lower severity.
	[[nodiscard]] static OaStatus Report(
		OaValidationSeverity InSev,
		OaLogComponent       InComp,
		const char*          InFmt, ...) __attribute__((format(printf, 3, 4)));

	// ── Debug counter API (debug builds only; always returns 0 in release) ──

	// Increment a named counter (thread-safe, lock-free on the hot path).
	static void IncrCounter(const char* InName);

	// Increment using a runtime string variable (for kernel names, etc.).
	static void IncrCounterNamed(const char* InName);

	// Returns current counter value (0 if never incremented or in release).
	[[nodiscard]] static OaU64 GetCounter(const char* InName);

	// Reset all counters to 0.
	static void ResetCounters();

	// Log all non-zero counters at Debug level.
	static void DumpCounters(OaLogComponent InComp = OaLogComponent::Core);
};

// ─────────────────────────────────────────────────────────────────────────────
// Macro: OA_VALIDATE — core validation primitive
// ─────────────────────────────────────────────────────────────────────────────
//
// Usage (in functions returning OaStatus):
//   OA_VALIDATE(M > 0, OaValidationSeverity::Error, OaLogComponent::Core,
//       "Gemm: M must be > 0, got %u", M);
//
// - Error / Fatal severity: logs the message and returns OaStatus::Error.
// - Warning / Info / Verbose: logs the message, does not return.
// - Fatal: also calls OA_ASSERT(false) after logging.

#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)

#define OA_VALIDATE(cond_, sev_, comp_, ...)                                      \
	do {                                                                          \
		if (OaValidation::IsEnabled() and not (cond_)) {                          \
			auto _oa_val_st = OaValidation::Report((sev_), (comp_), __VA_ARGS__); \
			if (static_cast<OaU8>(sev_) >= static_cast<OaU8>(OaValidationSeverity::Fatal)) { \
				OA_ASSERT(false);                                                 \
			}                                                                     \
			if (static_cast<OaU8>(sev_) >= static_cast<OaU8>(OaValidationSeverity::Error)) { \
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
	OA_VALIDATE((ptr_) != nullptr, OaValidationSeverity::Error,                  \
		OaLogComponent::Core, "%s: null pointer", (ctx_))

// Array / tensor index bounds (index < limit).
#define OA_VALIDATE_BOUNDS(idx_, limit_, ctx_)                                   \
	OA_VALIDATE(static_cast<OaI64>(idx_) >= 0 and                                \
		static_cast<OaI64>(idx_) < static_cast<OaI64>(limit_),                   \
		OaValidationSeverity::Error, OaLogComponent::Core,                        \
		"%s: index %lld out of bounds [0, %lld)", (ctx_),                         \
		static_cast<OaI64>(idx_), static_cast<OaI64>(limit_))

// Alignment check (value % align == 0).
#define OA_VALIDATE_ALIGNMENT(val_, align_, ctx_)                                \
	OA_VALIDATE((static_cast<OaU64>(val_) % static_cast<OaU64>(align_)) == 0,   \
		OaValidationSeverity::Error, OaLogComponent::Core,                        \
		"%s: value %llu not aligned to %llu bytes", (ctx_),                       \
		static_cast<OaU64>(val_), static_cast<OaU64>(align_))

// Push constant size: actual must exactly equal declared shader struct size.
#define OA_VALIDATE_PUSH_SIZE(actual_, declared_, kernel_)                       \
	OA_VALIDATE((actual_) == (declared_), OaValidationSeverity::Error,            \
		OaLogComponent::Core,                                                      \
		"Dispatch '%s': push constant size %zu != declared %zu bytes",             \
		(kernel_), static_cast<OaUsize>(actual_), static_cast<OaUsize>(declared_))

// Buffer count check.
#define OA_VALIDATE_BUFFER_COUNT(actual_, expected_, kernel_)                    \
	OA_VALIDATE((actual_) == (expected_), OaValidationSeverity::Error,            \
		OaLogComponent::Core,                                                      \
		"Dispatch '%s': buffer count %zu != expected %zu",                         \
		(kernel_), static_cast<OaUsize>(actual_), static_cast<OaUsize>(expected_))

// Node index range check for multi-device dispatch.
#define OA_VALIDATE_NODE_INDEX(nodeIdx_, nodeCount_, ctx_)                       \
	OA_VALIDATE((nodeIdx_) < (nodeCount_), OaValidationSeverity::Error,           \
		OaLogComponent::Core,                                                      \
		"%s: nodeIndex=%u >= DeviceCount=%u", (ctx_),                             \
		static_cast<OaU32>(nodeIdx_), static_cast<OaU32>(nodeCount_))

// Matrix dtype match (requires GetDtype() and OaScalarTypeName()).
#define OA_VALIDATE_DTYPE(a_, b_, ctx_)                                          \
	OA_VALIDATE((a_).GetDtype() == (b_).GetDtype(),                              \
		OaValidationSeverity::Error, OaLogComponent::ML,                          \
		"%s: dtype mismatch — %s vs %s", (ctx_),                                 \
		OaScalarTypeName((a_).GetDtype()).data(),                                  \
		OaScalarTypeName((b_).GetDtype()).data())

// Exact dtype check (requires GetDtype() and OaScalarTypeName()).
#define OA_VALIDATE_DTYPE_EXACT(mat_, expected_, ctx_)                           \
	OA_VALIDATE((mat_).GetDtype() == (expected_),                                 \
		OaValidationSeverity::Error, OaLogComponent::ML,                          \
		"%s: expected dtype %s, got %s", (ctx_),                                  \
		OaScalarTypeName(expected_).data(),                                        \
		OaScalarTypeName((mat_).GetDtype()).data())

// Matrix multiplication shape compatibility: A is [M,K], B is [N,K] (weight-transposed).
// Checks InA.Size(-1) == InB.Size(-1).
#define OA_VALIDATE_SHAPE_COMPAT(a_, b_, op_)                                    \
	OA_VALIDATE((a_).Rank() >= 2 and (b_).Rank() == 2 and                        \
		(a_).Size(-1) == (b_).Size(-1),                                           \
		OaValidationSeverity::Error, OaLogComponent::ML,                          \
		"%s: shape mismatch — A.K=%lld B.K=%lld (A rank=%d B rank=%d)", (op_),   \
		(a_).Rank() >= 2 ? (a_).Size(-1) : -1LL,                                 \
		(b_).Rank() >= 2 ? (b_).Size(-1) : -1LL,                                 \
		(a_).Rank(), (b_).Rank())

// Heap slot registration check (requires HeapSlot() >= 0 and IsOnDevice()).
#define OA_VALIDATE_HEAP_SLOT(buf_, ctx_)                                        \
	OA_VALIDATE((buf_).HeapSlot() >= 0, OaValidationSeverity::Error,             \
		OaLogComponent::Core,                                                     \
		"%s: buffer not registered with bindless heap (HeapSlot=%d)", (ctx_),     \
		(buf_).HeapSlot())

// ─────────────────────────────────────────────────────────────────────────────
// OA_WARN_PERF — performance warning (no return; Warning severity)
// ─────────────────────────────────────────────────────────────────────────────
//
// Fires at Warning severity when cond is true. Does not return from caller.
// Active in debug and when OA_ENABLE_VALIDATION or OA_VALIDATION_PERF is set.

#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)

#define OA_WARN_PERF(cond_, ...)                                                 \
	do {                                                                         \
		if (OaValidation::IsEnabled() and (cond_)) {                             \
			(void)OaValidation::Report(OaValidationSeverity::Warning,            \
				OaLogComponent::Core, __VA_ARGS__);                              \
		}                                                                        \
	} while (0)

#else

#define OA_WARN_PERF(cond_, ...) ((void)0)

#endif

// ─────────────────────────────────────────────────────────────────────────────
// OA_DEBUG_COUNTER_INC / OA_DEBUG_COUNTER_GET — always compiled out in release
// ─────────────────────────────────────────────────────────────────────────────

#ifndef NDEBUG

// Increment a compile-time-named counter. Name becomes a string literal.
#define OA_DEBUG_COUNTER_INC(name_) OaValidation::IncrCounter(#name_)

// Increment using a runtime string expression (e.g., kernel name variable).
#define OA_DEBUG_COUNTER_INC_NAMED(strExpr_) OaValidation::IncrCounterNamed(strExpr_)

// Read counter value (returns OaU64).
#define OA_DEBUG_COUNTER_GET(name_) OaValidation::GetCounter(#name_)

#else

#define OA_DEBUG_COUNTER_INC(name_)          ((void)0)
#define OA_DEBUG_COUNTER_INC_NAMED(strExpr_) ((void)0)
#define OA_DEBUG_COUNTER_GET(name_)          (OaU64(0))

#endif
