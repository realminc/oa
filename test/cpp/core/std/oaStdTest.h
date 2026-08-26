#pragma once

// OaStd tests — shared preamble: OaTest.h + full OaStd bundle (<Oa/Core/Std.h>).
// Parity logging matches Test/Ml/TestSimpleLlm.cpp style: stderr, indented lines.
// Run with: ctest -R TestOaStd -V or ./bin/release/test/core/std/testOaStd
// (plain ctest hides child stderr for passing tests unless -V.)

#include "../../oaTest.h"
#include <oa/core/std.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

// Current GTest case: "  [oastd] Suite.name" (call once per TEST before detail lines).
static inline void stdEchoCurrentTest() {
	const auto* ti = ::testing::UnitTest::GetInstance()->current_test_info();
	if (ti) {
		fprintf(stderr, "  [oastd] %s.%s\n", ti->test_suite_name(), ti->name());
	} else {
		fprintf(stderr, "  [oastd] (unknown test)\n");
	}
}

template <typename Rep, typename Period>
inline double stdWallMs(std::chrono::duration<Rep, Period> inD) {
	return std::chrono::duration<double, std::milli>(inD).count();
}

inline double stdWallMs(oa::Duration inDuration) {
	return inDuration.toMilliseconds();
}

// Oa/std_time > 1 means Oa wall clock is slower (more ms than std).
static inline void stdReportCompareMsLines(
	const char* inOaOp, double inOaMs, const char* inStdOp, double inStdMs) {
	const double oaPerStd = (inStdMs > 1e-12) ? (inOaMs / inStdMs) : 0.0;
	const double stdPerOa = (inOaMs > 1e-12) ? (inStdMs / inOaMs) : 0.0;
	const char* faster = (inOaMs < inStdMs) ? "OaStd" : ((inStdMs < inOaMs) ? "std" : "tie");
	fprintf(stderr,
		"    %s %.3f ms  %s %.3f ms  faster=%s  Oa/std_time=%.2fx  std/Oa_time=%.2fx\n",
		inOaOp, inOaMs, inStdOp, inStdMs, faster, oaPerStd, stdPerOa);
	fflush(stderr);
}

static inline void stdReportCompareMs(
	const char* inOaOp, double inOaMs, const char* inStdOp, double inStdMs) {
	stdEchoCurrentTest();
	stdReportCompareMsLines(inOaOp, inOaMs, inStdOp, inStdMs);
}

static inline void stdReportCompareUs(
	const char* inOaOp, long long inOaUs, const char* inStdOp, long long inStdUs) {
	stdReportCompareMs(inOaOp, static_cast<double>(inOaUs) / 1000.0, inStdOp,
		static_cast<double>(inStdUs) / 1000.0);
}

static inline void stdReportOaMsOnly(const char* inOaOp, double inOaMs) {
	stdEchoCurrentTest();
	fprintf(stderr, "    %s %.3f ms  (no std:: counterpart timed here)\n", inOaOp, inOaMs);
	fflush(stderr);
}

// After timing Oa work from inT0→inT1 and std work from inT1→inT2 (same clock).
template <typename TimePoint>
static inline void stdReportCompareSequentialRuns(
	const char* inOaLabel,
	TimePoint inT0,
	TimePoint inT1,
	const char* inStdLabel,
	TimePoint inT2) {
	stdReportCompareMs(inOaLabel, stdWallMs(inT1 - inT0), inStdLabel, stdWallMs(inT2 - inT1));
}

static inline void stdExpectGotFloat(const char* inCtx, double inExpected, double inGot) {
	fprintf(stderr, "    %s: expected=%.6f got=%.6f\n", inCtx, inExpected, inGot);
	fflush(stderr);
}

static inline void stdExpectGotInt(const char* inCtx, long long inExpected, long long inGot) {
	fprintf(stderr, "    %s: expected=%lld got=%lld\n", inCtx,
		static_cast<long long>(inExpected), static_cast<long long>(inGot));
	fflush(stderr);
}

static inline void stdExpectGotSize(const char* inCtx, std::size_t inExpected, std::size_t inGot) {
	fprintf(stderr, "    %s: expected=%zu got=%zu\n", inCtx, inExpected, inGot);
	fflush(stderr);
}

// Legacy names — single [oastd] header, then label + compare line.
static inline void stdLogParityUs(
	const char* inLabel, long long inOaUs, long long inStdUs, std::size_t inN) {
	stdEchoCurrentTest();
	fprintf(stderr, "    %s  n=%zu\n", inLabel, inN);
	stdReportCompareMsLines("OaStd", static_cast<double>(inOaUs) / 1000.0, "std",
		static_cast<double>(inStdUs) / 1000.0);
}

static inline void stdLogParityMs(
	const char* inLabel, long long inOaMs, long long inStdMs, std::size_t inN) {
	stdEchoCurrentTest();
	fprintf(stderr, "    %s  n=%zu\n", inLabel, inN);
	stdReportCompareMsLines("OaStd", static_cast<double>(inOaMs), "std",
		static_cast<double>(inStdMs));
}

static inline void stdLogExpectedGotSize(
	const char* inCtx, std::size_t inExpected, std::size_t inGot) {
	stdExpectGotSize(inCtx, inExpected, inGot);
}

static inline void stdLogExpectedGotInt(const char* inCtx, long long inExpected, long long inGot) {
	stdExpectGotInt(inCtx, inExpected, inGot);
}
