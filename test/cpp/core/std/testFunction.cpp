#include "oaStdTest.h"

#include <functional>

TEST(Fn, Call) {
	oa::Fn<int(int)> fn = [](int x) { return x * 2; };
	EXPECT_EQ(fn(21), 42);
}

TEST(Fn, EmptyAndSwap) {
	oa::Fn<int()> a;
	oa::Fn<int()> b = [] { return 1; };
	EXPECT_TRUE(a.empty());
	EXPECT_FALSE(b.empty());
	EXPECT_THROW((void)a(), std::bad_function_call);
	a.swap(b);
	EXPECT_FALSE(a.empty());
	EXPECT_TRUE(b.empty());
	EXPECT_THROW((void)b(), std::bad_function_call);
}

TEST(Fn, StdFn) {
	oa::Fn<int()> fn = [] { return 99; };
	EXPECT_EQ(fn.stdFn()(), 99);
}

TEST(StdFnVsStd, SameLambdaResultAsStdFunction) {
	const auto lam = [](int x) { return x * x + 1; };
	oa::Fn<int(int)> oa(lam);
	std::function<int(int)> st(lam);
	stdEchoCurrentTest();
	stdExpectGotInt("function(7)", static_cast<long long>(st(7)), static_cast<long long>(oa(7)));
	EXPECT_EQ(oa(7), st(7));
}

TEST(StdFnVsStd, TimedInvokeWallUs) {
	constexpr int kCalls = 300'000;
	oa::Fn<int(int)> oa = [](int x) { return x ^ (x >> 2); };
	std::function<int(int)> st = [](int x) { return x ^ (x >> 2); };
	volatile long long sinkOa = 0;
	const auto t0 = oa::highResolutionNow();
	for (int i = 0; i < kCalls; ++i) {
		sinkOa += oa(i);
	}
	const auto t1 = oa::highResolutionNow();
	volatile long long sinkSt = 0;
	for (int i = 0; i < kCalls; ++i) {
		sinkSt += st(i);
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Fn::operator() x300k", t0, t1, "std::function::operator() x300k", t2);
	stdExpectGotInt("function invoke sum", static_cast<long long>(sinkSt), static_cast<long long>(sinkOa));
	EXPECT_EQ(sinkOa, sinkSt);
}
