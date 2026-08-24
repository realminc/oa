#include "oaStdTest.h"

#include <type_traits>
#include <variant>

TEST(Variant, IndexAndGet) {
	oa::Variant<int, float> v{3.14f};
	EXPECT_EQ(v.index(), 1U);
	EXPECT_FLOAT_EQ(v.get<float>(), 3.14f);

	oa::Variant<int, float> oa{1.0f};
	std::variant<int, float> st{1.0f};
	constexpr int kLoops = 300'000;
	const auto t0 = oa::highResolutionNow();
	volatile float sinkOa = 0;
	for (int i = 0; i < kLoops; ++i) {
		(void)i;
		sinkOa += oa.get<float>();
	}
	const auto t1 = oa::highResolutionNow();
	volatile float sinkSt = 0;
	for (int i = 0; i < kLoops; ++i) {
		(void)i;
		sinkSt += std::get<float>(st);
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Variant::get<float> x300k", t0, t1, "std::get<float> x300k", t2);
	stdExpectGotFloat("variant float sum (scaled)", sinkSt / 1000.f, sinkOa / 1000.f);
	EXPECT_FLOAT_EQ(sinkOa, sinkSt);
}

TEST(Variant, HoldsAlternativeAndVisit) {
	oa::Variant<int, const char*> v{7};
	ASSERT_TRUE(v.holdsAlternative<int>());
	int sum = 0;
	v.visit([&sum](auto&& x) {
		using U = std::decay_t<decltype(x)>;
		if constexpr (std::is_same_v<U, int>) {
			sum += x;
		}
	});
	EXPECT_EQ(sum, 7);

	oa::Variant<int, float> oa{42};
	std::variant<int, float> st{42};
	constexpr int kVisits = 200'000;
	const auto t0 = oa::highResolutionNow();
	volatile int sinkOa = 0;
	for (int i = 0; i < kVisits; ++i) {
		oa.visit([&sinkOa](auto&& x) {
			using U = std::decay_t<decltype(x)>;
			if constexpr (std::is_same_v<U, int>) {
				sinkOa += x;
			}
		});
	}
	const auto t1 = oa::highResolutionNow();
	volatile int sinkSt = 0;
	for (int i = 0; i < kVisits; ++i) {
		sinkSt += std::visit([](auto&& x) { return static_cast<int>(x); }, st);
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Variant::visit x200k", t0, t1, "std::visit x200k", t2);
	const long long expectSum = static_cast<long long>(42) * kVisits;
	stdExpectGotInt("visit accumulated sum", expectSum, static_cast<long long>(sinkOa));
	EXPECT_EQ(sinkOa, sinkSt);
	EXPECT_EQ(sinkOa, expectSum);
}

TEST(Variant, EmplaceAndSwap) {
	oa::Variant<int, float> a{1};
	oa::Variant<int, float> b{2.5f};
	EXPECT_TRUE(a.holdsAlternative<int>());
	a.emplace<float>(1.5f);
	EXPECT_TRUE(a.holdsAlternative<float>());
	EXPECT_TRUE(b.holdsAlternative<float>());
	a.swap(b);
	EXPECT_FLOAT_EQ(a.get<float>(), 2.5f);
	EXPECT_FLOAT_EQ(b.get<float>(), 1.5f);

	constexpr int kSwaps = 500'000;
	const auto t0 = oa::highResolutionNow();
	for (int i = 0; i < kSwaps; ++i) {
		oa::Variant<int, float> x{1};
		oa::Variant<int, float> y{2.0f};
		x.swap(y);
		(void)x.index();
	}
	const auto t1 = oa::highResolutionNow();
	for (int i = 0; i < kSwaps; ++i) {
		std::variant<int, float> x{1};
		std::variant<int, float> y{2.0f};
		std::swap(x, y);
		(void)x.index();
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Variant::swap x500k", t0, t1, "std::swap(variant) x500k", t2);
}

TEST(StdVariantVsStd, StdVariantMatchesStdVariantState) {
	oa::Variant<int, double> oa{3.141592653589793};
	std::variant<int, double> st{3.141592653589793};
	auto conv = oa.stdVariant();
	stdEchoCurrentTest();
	stdExpectGotInt("variant index std", static_cast<long long>(st.index()),
		static_cast<long long>(conv.index()));
	EXPECT_EQ(conv.index(), st.index());
	EXPECT_DOUBLE_EQ(std::get<double>(conv), std::get<double>(st));
	oa.emplace<int>(-7);
	st.emplace<int>(-7);
	conv = oa.stdVariant();
	EXPECT_EQ(conv.index(), st.index());
	EXPECT_EQ(std::get<int>(conv), std::get<int>(st));
}

TEST(StdVariantVsStd, VisitSumMatchesStdVisit) {
	oa::Variant<int, float> oa{2.5f};
	std::variant<int, float> st{2.5f};
	int sumOa = 0;
	oa.visit([&sumOa](auto&& x) {
		using U = std::decay_t<decltype(x)>;
		if constexpr (std::is_same_v<U, int>) {
			sumOa += x;
		} else {
			sumOa += static_cast<int>(x);
		}
	});
	int sumSt = std::visit([](auto&& x) { return static_cast<int>(x); }, st);
	stdEchoCurrentTest();
	stdExpectGotInt("variant visit sum", static_cast<long long>(sumSt), static_cast<long long>(sumOa));
	EXPECT_EQ(sumOa, sumSt);
}
