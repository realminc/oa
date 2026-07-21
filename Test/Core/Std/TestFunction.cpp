#include "OaStdTest.h"

#include <functional>

TEST(OaStdFn, Call) {
	OaStdFn<int(int)> fn = [](int x) { return x * 2; };
	EXPECT_EQ(fn(21), 42);
}

TEST(OaStdFn, EmptyAndSwap) {
	OaStdFn<int()> a;
	OaStdFn<int()> b = [] { return 1; };
	EXPECT_TRUE(a.Empty());
	EXPECT_FALSE(b.Empty());
	EXPECT_THROW((void)a(), std::bad_function_call);
	a.Swap(b);
	EXPECT_FALSE(a.Empty());
	EXPECT_TRUE(b.Empty());
	EXPECT_THROW((void)b(), std::bad_function_call);
}

TEST(OaStdFn, StdFn) {
	OaStdFn<int()> fn = [] { return 99; };
	EXPECT_EQ(fn.StdFn()(), 99);
}

TEST(OaStdFnVsStd, SameLambdaResultAsStdFunction) {
	const auto lam = [](int x) { return x * x + 1; };
	OaStdFn<int(int)> oa(lam);
	std::function<int(int)> st(lam);
	OaStdEchoCurrentTest();
	OaStdExpectGotInt("function(7)", static_cast<long long>(st(7)), static_cast<long long>(oa(7)));
	EXPECT_EQ(oa(7), st(7));
}

TEST(OaStdFnVsStd, TimedInvokeWallUs) {
	constexpr int kCalls = 300'000;
	OaStdFn<int(int)> oa = [](int x) { return x ^ (x >> 2); };
	std::function<int(int)> st = [](int x) { return x ^ (x >> 2); };
	volatile long long sinkOa = 0;
	const auto t0 = OaHighResolutionNow();
	for (int i = 0; i < kCalls; ++i) {
		sinkOa += oa(i);
	}
	const auto t1 = OaHighResolutionNow();
	volatile long long sinkSt = 0;
	for (int i = 0; i < kCalls; ++i) {
		sinkSt += st(i);
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdFn::operator() x300k", t0, t1, "std::function::operator() x300k", t2);
	OaStdExpectGotInt("function invoke sum", static_cast<long long>(sinkSt), static_cast<long long>(sinkOa));
	EXPECT_EQ(sinkOa, sinkSt);
}
