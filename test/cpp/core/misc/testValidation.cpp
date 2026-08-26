#include <gtest/gtest.h>
#include <oa/core/validation.h>

#include <thread>
#include <vector>

// OA_VALIDATION_ACTIVE: true when OA_VALIDATE macros are compiled in.
// in release without OA_ENABLE_VALIDATION, macros are ((void)0) — violations return ok.
#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
static constexpr bool kValidationActive = true;
#else
static constexpr bool kValidationActive = false;
#endif

// Helper: a function that uses OA_VALIDATE and returns oa::Status
static oa::Status checkPositive(oa::I32 inVal) {
	OA_VALIDATE(inVal > 0, oa::ValidationSeverity::Error, oa::LogComponent::Core,
		"CheckPositive: value must be > 0, got %d", inVal);
	return oa::Status::ok();
}

static oa::Status checkBounds(oa::I32 inIdx, oa::I32 inLimit) {
	OA_VALIDATE_BOUNDS(inIdx, inLimit, "CheckBounds");
	return oa::Status::ok();
}

static oa::Status checkNull(const int* inPtr) {
	OA_VALIDATE_NOT_NULL(inPtr, "CheckNull");
	return oa::Status::ok();
}

static oa::Status checkAlignment(oa::U64 inOffset) {
	OA_VALIDATE_ALIGNMENT(inOffset, 16, "CheckAlignment");
	return oa::Status::ok();
}

static oa::Status checkPushSize(oa::Usize inActual) {
	OA_VALIDATE_PUSH_SIZE(inActual, 16, "TestKernel");
	return oa::Status::ok();
}

// ─────────────────────────────────────────────────────────────────────────────
// Basic enable / disable
// ─────────────────────────────────────────────────────────────────────────────

TEST(Validation, EnableDisable) {
	oa::Validation::enable(true);
	EXPECT_TRUE(oa::Validation::isEnabled());
	oa::Validation::enable(false);
	EXPECT_FALSE(oa::Validation::isEnabled());
	oa::Validation::enable(true);  // restore for subsequent tests
}

TEST(Validation, SeverityFilter) {
	oa::Validation::enable(true);
	oa::Validation::setMinSeverity(oa::ValidationSeverity::Warning);
	EXPECT_EQ(oa::Validation::getMinSeverity(), oa::ValidationSeverity::Warning);
	oa::Validation::setMinSeverity(oa::ValidationSeverity::Verbose);  // restore
}

// ─────────────────────────────────────────────────────────────────────────────
// OA_VALIDATE — Error fires on violation, ok on success
// ─────────────────────────────────────────────────────────────────────────────

TEST(Validation, ValidateOkOnPass) {
	oa::Validation::enable(true);
	auto st = checkPositive(5);
	EXPECT_TRUE(st.isOk());
}

TEST(Validation, ValidateErrorOnViolation) {
	oa::Validation::enable(true);
	auto st = checkPositive(-1);
	if (kValidationActive) {
		EXPECT_TRUE(st.isError());
		EXPECT_EQ(st.getCode(), oa::StatusCode::InvalidArgument);
		EXPECT_FALSE(st.getMessage().empty());
	} else {
		EXPECT_TRUE(st.isOk());  // macro compiled out in release
	}
}

TEST(Validation, ValidatePassthroughWhenDisabled) {
	// When validation is disabled, OA_VALIDATE should not fire.
#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
	oa::Validation::enable(false);
	auto st = checkPositive(-99);   // would fire if enabled
	EXPECT_TRUE(st.isOk());         // disabled → no check → ok
	oa::Validation::enable(true);
#else
	// in pure release without OA_ENABLE_VALIDATION, macro is ((void)0).
	SUCCEED();
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience macros
// ─────────────────────────────────────────────────────────────────────────────

TEST(Validation, ValidateBoundsPass) {
	oa::Validation::enable(true);
	EXPECT_TRUE(checkBounds(0, 10).isOk());
	EXPECT_TRUE(checkBounds(9, 10).isOk());
}

TEST(Validation, ValidateBoundsFail) {
	oa::Validation::enable(true);
	if (kValidationActive) {
		EXPECT_TRUE(checkBounds(10, 10).isError());
		EXPECT_TRUE(checkBounds(-1, 10).isError());
	} else {
		SUCCEED();
	}
}

TEST(Validation, ValidateNotNullPass) {
	oa::Validation::enable(true);
	int x = 0;
	EXPECT_TRUE(checkNull(&x).isOk());
}

TEST(Validation, ValidateNotNullFail) {
	oa::Validation::enable(true);
	if (kValidationActive) {
		EXPECT_TRUE(checkNull(nullptr).isError());
	} else {
		SUCCEED();
	}
}

TEST(Validation, ValidateAlignmentPass) {
	oa::Validation::enable(true);
	EXPECT_TRUE(checkAlignment(0).isOk());
	EXPECT_TRUE(checkAlignment(16).isOk());
	EXPECT_TRUE(checkAlignment(256).isOk());
}

TEST(Validation, ValidateAlignmentFail) {
	oa::Validation::enable(true);
	if (kValidationActive) {
		EXPECT_TRUE(checkAlignment(1).isError());
		EXPECT_TRUE(checkAlignment(15).isError());
		EXPECT_TRUE(checkAlignment(17).isError());
	} else {
		SUCCEED();
	}
}

TEST(Validation, ValidatePushSizePass) {
	oa::Validation::enable(true);
	EXPECT_TRUE(checkPushSize(16).isOk());
}

TEST(Validation, ValidatePushSizeFail) {
	oa::Validation::enable(true);
	if (kValidationActive) {
		EXPECT_TRUE(checkPushSize(8).isError());
		EXPECT_TRUE(checkPushSize(32).isError());
	} else {
		SUCCEED();
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom callback
// ─────────────────────────────────────────────────────────────────────────────

TEST(Validation, CustomCallback) {
	oa::Validation::enable(true);

	oa::String capturedMsg;

	oa::Validation::setCallback([](oa::ValidationSeverity inSev, oa::LogComponent, const char* inMsg) {
		// store in test-local statics (callback is a plain fn pointer)
		static oa::ValidationSeverity* pSev = nullptr;
		static oa::String* pMsg = nullptr;
		if (inSev == oa::ValidationSeverity::Verbose and inMsg[0] == '\0') {
			// Special sentinel: initialize pointers
			return;
		}
		if (pSev) { *pSev = inSev; }
		if (pMsg) { *pMsg = inMsg; }
	});

	// Plain callback test via Report directly.
	// We can't capture via lambda since setCallback takes a raw fn ptr.
	// Use Report directly and check that it doesn't crash.
	auto st = oa::Validation::report(
		oa::ValidationSeverity::Warning, oa::LogComponent::Core, "test warning %d", 42);
	EXPECT_TRUE(st.isOk());  // warning doesn't return error

	auto st2 = oa::Validation::report(
		oa::ValidationSeverity::Error, oa::LogComponent::Core, "test error %s", "oops");
	EXPECT_TRUE(st2.isError());

	oa::Validation::setCallback(nullptr);  // restore
}

// ─────────────────────────────────────────────────────────────────────────────
// Debug counters (only meaningful in debug builds)
// ─────────────────────────────────────────────────────────────────────────────

TEST(Validation, DebugCounters) {
	oa::Validation::resetCounters();

	OA_DEBUG_COUNTER_INC(test_counter_a);
	OA_DEBUG_COUNTER_INC(test_counter_a);
	OA_DEBUG_COUNTER_INC(test_counter_b);

#ifndef NDEBUG
	EXPECT_EQ(oa::Validation::getCounter("test_counter_a"), oa::U64(2));
	EXPECT_EQ(oa::Validation::getCounter("test_counter_b"), oa::U64(1));
	EXPECT_EQ(oa::Validation::getCounter("test_counter_missing"), oa::U64(0));
#else
	// in release, GetCounter always returns 0
	EXPECT_EQ(OA_DEBUG_COUNTER_GET(test_counter_a), oa::U64(0));
#endif

	oa::Validation::resetCounters();

#ifndef NDEBUG
	EXPECT_EQ(oa::Validation::getCounter("test_counter_a"), oa::U64(0));
#endif
}

TEST(Validation, DebugCounterNamedIncr) {
	oa::Validation::resetCounters();

	OA_DEBUG_COUNTER_INC_NAMED("GemmCmSgBf16");
	OA_DEBUG_COUNTER_INC_NAMED("GemmCmSgBf16");
	OA_DEBUG_COUNTER_INC_NAMED("GemmCmSgBf16");

#ifndef NDEBUG
	EXPECT_EQ(oa::Validation::getCounter("GemmCmSgBf16"), oa::U64(3));
#else
	SUCCEED();
#endif
	oa::Validation::resetCounters();
}

TEST(Validation, DebugCounterSerializesWriters) {
#ifndef NDEBUG
	oa::Validation::resetCounters();
	std::vector<std::thread> writers;
	for (oa::I32 thread = 0; thread < 4; ++thread) {
		writers.emplace_back([] {
			for (oa::I32 value = 0; value < 1000; ++value)
				oa::Validation::incrCounterNamed("concurrent_counter");
		});
	}
	for (auto& writer : writers) writer.join();
	EXPECT_EQ(oa::Validation::getCounter("concurrent_counter"), oa::U64(4000));
	oa::Validation::resetCounters();
#else
	SUCCEED();
#endif
}

TEST(Validation, DumpCountersDoesNotCrash) {
	OA_DEBUG_COUNTER_INC(dispatch_count);
	OA_DEBUG_COUNTER_INC(dispatch_count);
	oa::Validation::dumpCounters(oa::LogComponent::Core);
	oa::Validation::resetCounters();
}

// ─────────────────────────────────────────────────────────────────────────────
// initFromEnv (smoke: doesn't crash)
// ─────────────────────────────────────────────────────────────────────────────

TEST(Validation, InitFromEnvDoesNotCrash) {
	oa::Validation::initFromEnv();
	oa::Validation::enable(true);  // restore after env may have set it
}
