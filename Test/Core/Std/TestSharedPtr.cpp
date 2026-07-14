#include "OaStdTest.h"

#include <memory>

TEST(OaStdSharedPtr, MakeShared) {
	auto p = OaStdMakeShared<int>(44);
	ASSERT_TRUE(static_cast<bool>(p));
	EXPECT_EQ(*p, 44);
	EXPECT_EQ(p.UseCount(), 1);
}

TEST(OaStdSharedPtr, CopySharesUseCount) {
	auto a = OaStdMakeShared<int>(7);
	OaStdSharedPtr<int> b = a;
	EXPECT_EQ(a.UseCount(), 2);
	EXPECT_EQ(b.UseCount(), 2);
	EXPECT_EQ(*a, 7);
	EXPECT_EQ(*b, 7);
}

TEST(OaStdSharedPtr, MoveLeavesEmpty) {
	auto a = OaStdMakeShared<int>(3);
	OaStdSharedPtr<int> b = std::move(a);
	EXPECT_FALSE(static_cast<bool>(a));
	EXPECT_TRUE(static_cast<bool>(b));
	EXPECT_EQ(*b, 3);
	EXPECT_EQ(b.UseCount(), 1);
}

TEST(OaStdSharedPtr, Reset) {
	auto p = OaStdMakeShared<int>(99);
	p.Reset();
	EXPECT_FALSE(p);
	EXPECT_EQ(p.UseCount(), 0);
}

TEST(OaStdSharedPtr, CustomDeleter) {
	static int calls = 0;
	struct D {
		void operator()(int* InP) {
			++calls;
			delete InP;
		}
	};
	calls = 0;
	{
		OaStdSharedPtr<int> p(new int(5), D{});
		EXPECT_EQ(*p, 5);
	}
	EXPECT_EQ(calls, 1);
}

TEST(OaStdWeakPtr, FromSharedTracksUseCount) {
	auto s = OaStdMakeShared<int>(11);
	OaStdWeakPtr<int> w(s);
	EXPECT_FALSE(w.Expired());
	EXPECT_EQ(w.UseCount(), 1);
	EXPECT_EQ(s.UseCount(), 1);
}

TEST(OaStdWeakPtr, LockWhileAlive) {
	auto s = OaStdMakeShared<int>(7);
	OaStdWeakPtr<int> w(s);
	auto l = w.Lock();
	ASSERT_TRUE(static_cast<bool>(l));
	EXPECT_EQ(*l, 7);
	EXPECT_EQ(s.UseCount(), 2);
}

TEST(OaStdWeakPtr, ExpiredAfterLastSharedDestroyed) {
	OaStdWeakPtr<int> w;
	{
		auto s = OaStdMakeShared<int>(3);
		w = OaStdWeakPtr<int>(s);
		EXPECT_FALSE(w.Expired());
	}
	EXPECT_TRUE(w.Expired());
	EXPECT_FALSE(static_cast<bool>(w.Lock()));
}

TEST(OaStdWeakPtr, CopySharesControlBlock) {
	auto s = OaStdMakeShared<int>(9);
	OaStdWeakPtr<int> a(s);
	OaStdWeakPtr<int> b = a;
	EXPECT_FALSE(a.Expired());
	EXPECT_FALSE(b.Expired());
	s.Reset();
	EXPECT_TRUE(a.Expired());
	EXPECT_TRUE(b.Expired());
}

TEST(OaStdSharedPtrVsStd, UseCountAfterCopyMatchesPattern) {
	auto oa = OaStdMakeShared<int>(55);
	std::shared_ptr<int> st = std::make_shared<int>(55);
	OaStdSharedPtr<int> oa2 = oa;
	std::shared_ptr<int> st2 = st;
	OaStdEchoCurrentTest();
	OaStdExpectGotInt("shared use_count Oa", 2, static_cast<long long>(oa.UseCount()));
	OaStdExpectGotInt("shared use_count std", 2, static_cast<long long>(st.use_count()));
	EXPECT_EQ(oa.UseCount(), 2u);
	EXPECT_EQ(st.use_count(), 2u);
	EXPECT_EQ(*oa, *st);
}

TEST(OaStdSharedPtrVsStd, TimedMakeSharedWallUs) {
	constexpr int kIters = 50'000;
	const auto t0 = OaHighResolutionNow();
	for (int i = 0; i < kIters; ++i) {
		auto p = OaStdMakeShared<int>(i);
		(void)p.UseCount();
		p.Reset();
	}
	const auto t1 = OaHighResolutionNow();
	for (int i = 0; i < kIters; ++i) {
		auto p = std::make_shared<int>(i);
		(void)p.use_count();
		p.reset();
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdMakeShared+Reset x50k", t0, t1, "std::make_shared+reset x50k", t2);
}
