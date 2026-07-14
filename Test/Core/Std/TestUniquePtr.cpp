#include "OaStdTest.h"

#include <memory>

TEST(OaStdUniquePtr, MakeUnique) {
	auto p = OaStdMakeUnique<int>(33);
	ASSERT_TRUE(static_cast<bool>(p));
	EXPECT_EQ(*p, 33);
}

TEST(OaStdUniquePtr, MoveAndReset) {
	auto a = OaStdMakeUnique<int>(7);
	OaStdUniquePtr<int> b = std::move(a);
	EXPECT_FALSE(static_cast<bool>(a));
	ASSERT_TRUE(static_cast<bool>(b));
	EXPECT_EQ(*b, 7);
	b.Reset();
	EXPECT_FALSE(static_cast<bool>(b));
}

TEST(OaStdUniquePtr, StdPtrRvalue) {
	auto p = OaStdMakeUnique<int>(99);
	auto s = std::move(p).StdPtr();
	ASSERT_TRUE(s);
	EXPECT_EQ(*s, 99);
}

TEST(OaStdUniquePtrVsStd, DerefMatchesParallelStdUniquePtr) {
	auto oa = OaStdMakeUnique<int>(1001);
	std::unique_ptr<int> st = std::make_unique<int>(1001);
	OaStdEchoCurrentTest();
	OaStdExpectGotInt("unique_ptr *", static_cast<long long>(*st), static_cast<long long>(*oa));
	EXPECT_EQ(*oa, *st);
}

TEST(OaStdUniquePtrVsStd, TimedMakeResetWallUs) {
	constexpr int kIters = 80'000;
	const auto t0 = OaHighResolutionNow();
	for (int i = 0; i < kIters; ++i) {
		auto p = OaStdMakeUnique<int>(i);
		(void)*p;
		p.Reset();
	}
	const auto t1 = OaHighResolutionNow();
	for (int i = 0; i < kIters; ++i) {
		auto p = std::make_unique<int>(i);
		(void)*p;
		p.reset();
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdMakeUnique+Reset x80k", t0, t1, "std::make_unique+reset x80k", t2);
}
