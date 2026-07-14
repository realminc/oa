#include "OaStdTest.h"

#include <type_traits>
#include <variant>

TEST(OaStdVariant, IndexAndGet) {
	OaStdVariant<int, float> v{3.14f};
	EXPECT_EQ(v.Index(), 1U);
	EXPECT_FLOAT_EQ(v.Get<float>(), 3.14f);

	OaStdVariant<int, float> oa{1.0f};
	std::variant<int, float> st{1.0f};
	constexpr int kLoops = 300'000;
	const auto t0 = OaHighResolutionNow();
	volatile float sinkOa = 0;
	for (int i = 0; i < kLoops; ++i) {
		(void)i;
		sinkOa += oa.Get<float>();
	}
	const auto t1 = OaHighResolutionNow();
	volatile float sinkSt = 0;
	for (int i = 0; i < kLoops; ++i) {
		(void)i;
		sinkSt += std::get<float>(st);
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdVariant::Get<float> x300k", t0, t1, "std::get<float> x300k", t2);
	OaStdExpectGotFloat("variant float sum (scaled)", sinkSt / 1000.f, sinkOa / 1000.f);
	EXPECT_FLOAT_EQ(sinkOa, sinkSt);
}

TEST(OaStdVariant, HoldsAlternativeAndVisit) {
	OaStdVariant<int, const char*> v{7};
	ASSERT_TRUE(v.HoldsAlternative<int>());
	int sum = 0;
	v.Visit([&sum](auto&& x) {
		using U = std::decay_t<decltype(x)>;
		if constexpr (std::is_same_v<U, int>) {
			sum += x;
		}
	});
	EXPECT_EQ(sum, 7);

	OaStdVariant<int, float> oa{42};
	std::variant<int, float> st{42};
	constexpr int kVisits = 200'000;
	const auto t0 = OaHighResolutionNow();
	volatile int sinkOa = 0;
	for (int i = 0; i < kVisits; ++i) {
		oa.Visit([&sinkOa](auto&& x) {
			using U = std::decay_t<decltype(x)>;
			if constexpr (std::is_same_v<U, int>) {
				sinkOa += x;
			}
		});
	}
	const auto t1 = OaHighResolutionNow();
	volatile int sinkSt = 0;
	for (int i = 0; i < kVisits; ++i) {
		sinkSt += std::visit([](auto&& x) { return static_cast<int>(x); }, st);
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdVariant::Visit x200k", t0, t1, "std::visit x200k", t2);
	const long long expectSum = static_cast<long long>(42) * kVisits;
	OaStdExpectGotInt("visit accumulated sum", expectSum, static_cast<long long>(sinkOa));
	EXPECT_EQ(sinkOa, sinkSt);
	EXPECT_EQ(sinkOa, expectSum);
}

TEST(OaStdVariant, EmplaceAndSwap) {
	OaStdVariant<int, float> a{1};
	OaStdVariant<int, float> b{2.5f};
	EXPECT_TRUE(a.HoldsAlternative<int>());
	a.Emplace<float>(1.5f);
	EXPECT_TRUE(a.HoldsAlternative<float>());
	EXPECT_TRUE(b.HoldsAlternative<float>());
	a.Swap(b);
	EXPECT_FLOAT_EQ(a.Get<float>(), 2.5f);
	EXPECT_FLOAT_EQ(b.Get<float>(), 1.5f);

	constexpr int kSwaps = 500'000;
	const auto t0 = OaHighResolutionNow();
	for (int i = 0; i < kSwaps; ++i) {
		OaStdVariant<int, float> x{1};
		OaStdVariant<int, float> y{2.0f};
		x.Swap(y);
		(void)x.Index();
	}
	const auto t1 = OaHighResolutionNow();
	for (int i = 0; i < kSwaps; ++i) {
		std::variant<int, float> x{1};
		std::variant<int, float> y{2.0f};
		std::swap(x, y);
		(void)x.index();
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdVariant::Swap x500k", t0, t1, "std::swap(variant) x500k", t2);
}

TEST(OaStdVariantVsStd, StdVariantMatchesStdVariantState) {
	OaStdVariant<int, double> oa{3.141592653589793};
	std::variant<int, double> st{3.141592653589793};
	auto conv = oa.StdVariant();
	OaStdEchoCurrentTest();
	OaStdExpectGotInt("variant index std", static_cast<long long>(st.index()),
		static_cast<long long>(conv.index()));
	EXPECT_EQ(conv.index(), st.index());
	EXPECT_DOUBLE_EQ(std::get<double>(conv), std::get<double>(st));
	oa.Emplace<int>(-7);
	st.emplace<int>(-7);
	conv = oa.StdVariant();
	EXPECT_EQ(conv.index(), st.index());
	EXPECT_EQ(std::get<int>(conv), std::get<int>(st));
}

TEST(OaStdVariantVsStd, VisitSumMatchesStdVisit) {
	OaStdVariant<int, float> oa{2.5f};
	std::variant<int, float> st{2.5f};
	int sumOa = 0;
	oa.Visit([&sumOa](auto&& x) {
		using U = std::decay_t<decltype(x)>;
		if constexpr (std::is_same_v<U, int>) {
			sumOa += x;
		} else {
			sumOa += static_cast<int>(x);
		}
	});
	int sumSt = std::visit([](auto&& x) { return static_cast<int>(x); }, st);
	OaStdEchoCurrentTest();
	OaStdExpectGotInt("variant visit sum", static_cast<long long>(sumSt), static_cast<long long>(sumOa));
	EXPECT_EQ(sumOa, sumSt);
}
