#include <gtest/gtest.h>
#include <Oa/Core/Validation.h>

// OA_VALIDATION_ACTIVE: true when OA_VALIDATE macros are compiled in.
// In release without OA_ENABLE_VALIDATION, macros are ((void)0) — violations return Ok.
#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
static constexpr bool kValidationActive = true;
#else
static constexpr bool kValidationActive = false;
#endif

// Helper: a function that uses OA_VALIDATE and returns OaStatus
static OaStatus CheckPositive(OaI32 InVal) {
	OA_VALIDATE(InVal > 0, OaValidationSeverity::Error, OaLogComponent::Core,
		"CheckPositive: value must be > 0, got %d", InVal);
	return OaStatus::Ok();
}

static OaStatus CheckBounds(OaI32 InIdx, OaI32 InLimit) {
	OA_VALIDATE_BOUNDS(InIdx, InLimit, "CheckBounds");
	return OaStatus::Ok();
}

static OaStatus CheckNull(const int* InPtr) {
	OA_VALIDATE_NOT_NULL(InPtr, "CheckNull");
	return OaStatus::Ok();
}

static OaStatus CheckAlignment(OaU64 InOffset) {
	OA_VALIDATE_ALIGNMENT(InOffset, 16, "CheckAlignment");
	return OaStatus::Ok();
}

static OaStatus CheckPushSize(OaUsize InActual) {
	OA_VALIDATE_PUSH_SIZE(InActual, 16, "TestKernel");
	return OaStatus::Ok();
}

// ─────────────────────────────────────────────────────────────────────────────
// Basic enable / disable
// ─────────────────────────────────────────────────────────────────────────────

TEST(OaValidation, EnableDisable) {
	OaValidation::Enable(true);
	EXPECT_TRUE(OaValidation::IsEnabled());
	OaValidation::Enable(false);
	EXPECT_FALSE(OaValidation::IsEnabled());
	OaValidation::Enable(true);  // restore for subsequent tests
}

TEST(OaValidation, SeverityFilter) {
	OaValidation::Enable(true);
	OaValidation::SetMinSeverity(OaValidationSeverity::Warning);
	EXPECT_EQ(OaValidation::GetMinSeverity(), OaValidationSeverity::Warning);
	OaValidation::SetMinSeverity(OaValidationSeverity::Verbose);  // restore
}

// ─────────────────────────────────────────────────────────────────────────────
// OA_VALIDATE — Error fires on violation, Ok on success
// ─────────────────────────────────────────────────────────────────────────────

TEST(OaValidation, ValidateOkOnPass) {
	OaValidation::Enable(true);
	auto st = CheckPositive(5);
	EXPECT_TRUE(st.IsOk());
}

TEST(OaValidation, ValidateErrorOnViolation) {
	OaValidation::Enable(true);
	auto st = CheckPositive(-1);
	if (kValidationActive) {
		EXPECT_TRUE(st.IsError());
		EXPECT_EQ(st.GetCode(), OaStatusCode::InvalidArgument);
		EXPECT_FALSE(st.GetMessage().empty());
	} else {
		EXPECT_TRUE(st.IsOk());  // macro compiled out in release
	}
}

TEST(OaValidation, ValidatePassthroughWhenDisabled) {
	// When validation is disabled, OA_VALIDATE should not fire.
#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
	OaValidation::Enable(false);
	auto st = CheckPositive(-99);   // would fire if enabled
	EXPECT_TRUE(st.IsOk());         // disabled → no check → Ok
	OaValidation::Enable(true);
#else
	// In pure release without OA_ENABLE_VALIDATION, macro is ((void)0).
	SUCCEED();
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience macros
// ─────────────────────────────────────────────────────────────────────────────

TEST(OaValidation, ValidateBoundsPass) {
	OaValidation::Enable(true);
	EXPECT_TRUE(CheckBounds(0, 10).IsOk());
	EXPECT_TRUE(CheckBounds(9, 10).IsOk());
}

TEST(OaValidation, ValidateBoundsFail) {
	OaValidation::Enable(true);
	if (kValidationActive) {
		EXPECT_TRUE(CheckBounds(10, 10).IsError());
		EXPECT_TRUE(CheckBounds(-1, 10).IsError());
	} else {
		SUCCEED();
	}
}

TEST(OaValidation, ValidateNotNullPass) {
	OaValidation::Enable(true);
	int x = 0;
	EXPECT_TRUE(CheckNull(&x).IsOk());
}

TEST(OaValidation, ValidateNotNullFail) {
	OaValidation::Enable(true);
	if (kValidationActive) {
		EXPECT_TRUE(CheckNull(nullptr).IsError());
	} else {
		SUCCEED();
	}
}

TEST(OaValidation, ValidateAlignmentPass) {
	OaValidation::Enable(true);
	EXPECT_TRUE(CheckAlignment(0).IsOk());
	EXPECT_TRUE(CheckAlignment(16).IsOk());
	EXPECT_TRUE(CheckAlignment(256).IsOk());
}

TEST(OaValidation, ValidateAlignmentFail) {
	OaValidation::Enable(true);
	if (kValidationActive) {
		EXPECT_TRUE(CheckAlignment(1).IsError());
		EXPECT_TRUE(CheckAlignment(15).IsError());
		EXPECT_TRUE(CheckAlignment(17).IsError());
	} else {
		SUCCEED();
	}
}

TEST(OaValidation, ValidatePushSizePass) {
	OaValidation::Enable(true);
	EXPECT_TRUE(CheckPushSize(16).IsOk());
}

TEST(OaValidation, ValidatePushSizeFail) {
	OaValidation::Enable(true);
	if (kValidationActive) {
		EXPECT_TRUE(CheckPushSize(8).IsError());
		EXPECT_TRUE(CheckPushSize(32).IsError());
	} else {
		SUCCEED();
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom callback
// ─────────────────────────────────────────────────────────────────────────────

TEST(OaValidation, CustomCallback) {
	OaValidation::Enable(true);

	OaString capturedMsg;

	OaValidation::SetCallback([](OaValidationSeverity InSev, OaLogComponent, const char* InMsg) {
		// Store in test-local statics (callback is a plain fn pointer)
		static OaValidationSeverity* pSev = nullptr;
		static OaString* pMsg = nullptr;
		if (InSev == OaValidationSeverity::Verbose and InMsg[0] == '\0') {
			// Special sentinel: initialize pointers
			return;
		}
		if (pSev) { *pSev = InSev; }
		if (pMsg) { *pMsg = InMsg; }
	});

	// Plain callback test via Report directly.
	// We can't capture via lambda since SetCallback takes a raw fn ptr.
	// Use Report directly and check that it doesn't crash.
	auto st = OaValidation::Report(
		OaValidationSeverity::Warning, OaLogComponent::Core, "test warning %d", 42);
	EXPECT_TRUE(st.IsOk());  // Warning doesn't return error

	auto st2 = OaValidation::Report(
		OaValidationSeverity::Error, OaLogComponent::Core, "test error %s", "oops");
	EXPECT_TRUE(st2.IsError());

	OaValidation::SetCallback(nullptr);  // restore
}

// ─────────────────────────────────────────────────────────────────────────────
// Debug counters (only meaningful in debug builds)
// ─────────────────────────────────────────────────────────────────────────────

TEST(OaValidation, DebugCounters) {
	OaValidation::ResetCounters();

	OA_DEBUG_COUNTER_INC(test_counter_a);
	OA_DEBUG_COUNTER_INC(test_counter_a);
	OA_DEBUG_COUNTER_INC(test_counter_b);

#ifndef NDEBUG
	EXPECT_EQ(OaValidation::GetCounter("test_counter_a"), OaU64(2));
	EXPECT_EQ(OaValidation::GetCounter("test_counter_b"), OaU64(1));
	EXPECT_EQ(OaValidation::GetCounter("test_counter_missing"), OaU64(0));
#else
	// In release, GetCounter always returns 0
	EXPECT_EQ(OA_DEBUG_COUNTER_GET(test_counter_a), OaU64(0));
#endif

	OaValidation::ResetCounters();

#ifndef NDEBUG
	EXPECT_EQ(OaValidation::GetCounter("test_counter_a"), OaU64(0));
#endif
}

TEST(OaValidation, DebugCounterNamedIncr) {
	OaValidation::ResetCounters();

	OA_DEBUG_COUNTER_INC_NAMED("GemmCmSgBf16");
	OA_DEBUG_COUNTER_INC_NAMED("GemmCmSgBf16");
	OA_DEBUG_COUNTER_INC_NAMED("GemmCmSgBf16");

#ifndef NDEBUG
	EXPECT_EQ(OaValidation::GetCounter("GemmCmSgBf16"), OaU64(3));
#else
	SUCCEED();
#endif
	OaValidation::ResetCounters();
}

TEST(OaValidation, DumpCountersDoesNotCrash) {
	OA_DEBUG_COUNTER_INC(dispatch_count);
	OA_DEBUG_COUNTER_INC(dispatch_count);
	OaValidation::DumpCounters(OaLogComponent::Core);
	OaValidation::ResetCounters();
}

// ─────────────────────────────────────────────────────────────────────────────
// InitFromEnv (smoke: doesn't crash)
// ─────────────────────────────────────────────────────────────────────────────

TEST(OaValidation, InitFromEnvDoesNotCrash) {
	OaValidation::InitFromEnv();
	OaValidation::Enable(true);  // restore after env may have set it
}
