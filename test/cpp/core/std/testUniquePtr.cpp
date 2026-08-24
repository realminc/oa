#include "oaStdTest.h"

#include <memory>

TEST(UniquePtr, MakeUnique) {
	auto p = oa::makeUnique<int>(33);
	ASSERT_TRUE(static_cast<bool>(p));
	EXPECT_EQ(*p, 33);
}

TEST(UniquePtr, MoveAndReset) {
	auto a = oa::makeUnique<int>(7);
	oa::UniquePtr<int> b = std::move(a);
	EXPECT_FALSE(static_cast<bool>(a));
	ASSERT_TRUE(static_cast<bool>(b));
	EXPECT_EQ(*b, 7);
	b.reset();
	EXPECT_FALSE(static_cast<bool>(b));
}

TEST(UniquePtr, StdPtrRvalue) {
	auto p = oa::makeUnique<int>(99);
	auto s = std::move(p).stdPtr();
	ASSERT_TRUE(s);
	EXPECT_EQ(*s, 99);
}

TEST(StdUniquePtrVsStd, DerefMatchesParallelStdUniquePtr) {
	auto oa = oa::makeUnique<int>(1001);
	std::unique_ptr<int> st = std::make_unique<int>(1001);
	stdEchoCurrentTest();
	stdExpectGotInt("unique_ptr *", static_cast<long long>(*st), static_cast<long long>(*oa));
	EXPECT_EQ(*oa, *st);
}

TEST(StdUniquePtrVsStd, TimedMakeResetWallUs) {
	constexpr int kIters = 80'000;
	const auto t0 = oa::highResolutionNow();
	for (int i = 0; i < kIters; ++i) {
		auto p = oa::makeUnique<int>(i);
		(void)*p;
		p.reset();
	}
	const auto t1 = oa::highResolutionNow();
	for (int i = 0; i < kIters; ++i) {
		auto p = std::make_unique<int>(i);
		(void)*p;
		p.reset();
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::makeUnique+Reset x80k", t0, t1, "std::make_unique+reset x80k", t2);
}
