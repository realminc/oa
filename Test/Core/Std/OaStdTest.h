#pragma once

// OaStd tests — shared preamble: OaTest.h + full OaStd bundle (<Oa/Core/Std.h>).
// Parity logging matches Test/Ml/TestSimpleLlm.cpp style: stderr, indented lines.
// Run with:  ctest -R test_oastd -V   or   ./bin/release/Test/test_oastd
// (plain ctest hides child stderr for passing tests unless -V.)

#include "../../OaTest.h"
#include <Oa/Core/Std.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

// Current GTest case: "  [oastd] Suite.Name" (call once per TEST before detail lines).
static inline void OaStdEchoCurrentTest() {
	const auto* ti = ::testing::UnitTest::GetInstance()->current_test_info();
	if (ti) {
		fprintf(stderr, "  [oastd] %s.%s\n", ti->test_suite_name(), ti->name());
	} else {
		fprintf(stderr, "  [oastd] (unknown test)\n");
	}
}

template <typename Rep, typename Period>
inline double OaStdWallMs(std::chrono::duration<Rep, Period> InD) {
	return OaChronoToMilli(InD);
}

// Oa/std_time > 1 means Oa wall clock is slower (more ms than std).
static inline void OaStdReportCompareMsLines(
	const char* InOaOp, double InOaMs, const char* InStdOp, double InStdMs) {
	const double oaPerStd = (InStdMs > 1e-12) ? (InOaMs / InStdMs) : 0.0;
	const double stdPerOa = (InOaMs > 1e-12) ? (InStdMs / InOaMs) : 0.0;
	const char* faster = (InOaMs < InStdMs) ? "OaStd" : ((InStdMs < InOaMs) ? "std" : "tie");
	fprintf(stderr,
		"    %s %.3f ms  %s %.3f ms  faster=%s  Oa/std_time=%.2fx  std/Oa_time=%.2fx\n",
		InOaOp, InOaMs, InStdOp, InStdMs, faster, oaPerStd, stdPerOa);
	fflush(stderr);
}

static inline void OaStdReportCompareMs(
	const char* InOaOp, double InOaMs, const char* InStdOp, double InStdMs) {
	OaStdEchoCurrentTest();
	OaStdReportCompareMsLines(InOaOp, InOaMs, InStdOp, InStdMs);
}

static inline void OaStdReportCompareUs(
	const char* InOaOp, long long InOaUs, const char* InStdOp, long long InStdUs) {
	OaStdReportCompareMs(InOaOp, static_cast<double>(InOaUs) / 1000.0, InStdOp,
		static_cast<double>(InStdUs) / 1000.0);
}

static inline void OaStdReportOaMsOnly(const char* InOaOp, double InOaMs) {
	OaStdEchoCurrentTest();
	fprintf(stderr, "    %s %.3f ms  (no std:: counterpart timed here)\n", InOaOp, InOaMs);
	fflush(stderr);
}

// After timing Oa work from InT0→InT1 and std work from InT1→InT2 (same clock).
template <typename Clock>
static inline void OaStdReportCompareSequentialRuns(
	const char* InOaLabel,
	std::chrono::time_point<Clock> InT0,
	std::chrono::time_point<Clock> InT1,
	const char* InStdLabel,
	std::chrono::time_point<Clock> InT2) {
	OaStdReportCompareMs(InOaLabel, OaStdWallMs(InT1 - InT0), InStdLabel, OaStdWallMs(InT2 - InT1));
}

static inline void OaStdExpectGotFloat(const char* InCtx, double InExpected, double InGot) {
	fprintf(stderr, "    %s: expected=%.6f got=%.6f\n", InCtx, InExpected, InGot);
	fflush(stderr);
}

static inline void OaStdExpectGotInt(const char* InCtx, long long InExpected, long long InGot) {
	fprintf(stderr, "    %s: expected=%lld got=%lld\n", InCtx,
		static_cast<long long>(InExpected), static_cast<long long>(InGot));
	fflush(stderr);
}

static inline void OaStdExpectGotSize(const char* InCtx, std::size_t InExpected, std::size_t InGot) {
	fprintf(stderr, "    %s: expected=%zu got=%zu\n", InCtx, InExpected, InGot);
	fflush(stderr);
}

// Legacy names — single [oastd] header, then label + compare line.
static inline void OaStdLogParityUs(
	const char* InLabel, long long InOaUs, long long InStdUs, std::size_t InN) {
	OaStdEchoCurrentTest();
	fprintf(stderr, "    %s  n=%zu\n", InLabel, InN);
	OaStdReportCompareMsLines("OaStd", static_cast<double>(InOaUs) / 1000.0, "std",
		static_cast<double>(InStdUs) / 1000.0);
}

static inline void OaStdLogParityMs(
	const char* InLabel, long long InOaMs, long long InStdMs, std::size_t InN) {
	OaStdEchoCurrentTest();
	fprintf(stderr, "    %s  n=%zu\n", InLabel, InN);
	OaStdReportCompareMsLines("OaStd", static_cast<double>(InOaMs), "std",
		static_cast<double>(InStdMs));
}

static inline void OaStdLogExpectedGotSize(
	const char* InCtx, std::size_t InExpected, std::size_t InGot) {
	OaStdExpectGotSize(InCtx, InExpected, InGot);
}

static inline void OaStdLogExpectedGotInt(const char* InCtx, long long InExpected, long long InGot) {
	OaStdExpectGotInt(InCtx, InExpected, InGot);
}
