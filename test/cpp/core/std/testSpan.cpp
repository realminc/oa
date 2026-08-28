#include "oaStdTest.h"


TEST(Span, FromArray) {
	oa::Array<int, 3> data;
	data[0] = 1;
	data[1] = 2;
	data[2] = 3;
	oa::Span<int> sp(data);
	ASSERT_EQ(sp.size(), 3U);
	EXPECT_EQ(sp[1], 2);
}

TEST(Span, FromVecPreservesConstness) {
	oa::Vector<int> data{1, 2, 3};
	oa::Span<int> mutableSpan(data);
	oa::Span<const int> constSpan(data);
	const oa::Vector<int>& constData = data;
	oa::Span<const int> constSourceSpan(constData);

	mutableSpan[1] = 7;
	EXPECT_EQ(data[1], 7);
	EXPECT_EQ(constSpan.data(), data.data());
	EXPECT_EQ(constSourceSpan.size(), data.size());
}

TEST(Span, StdSpanParity) {
	std::array<int, 4> data = {10, 20, 30, 40};
	oa::Span<int> oa(data.data(), data.size());
	std::span<int> st(data.data(), data.size());
	EXPECT_EQ(oa.data(), st.data());
	EXPECT_EQ(oa.size(), st.size());
}

TEST(Span, FirstSubSpanMatchesStd) {
	std::array<int, 5> data = {1, 2, 3, 4, 5};
	oa::Span<int> oa(data.data(), data.size());
	std::span<int> st(data.data(), data.size());
	auto oaF = oa.first(3);
	auto stF = st.first<3>();
	EXPECT_EQ(oaF.size(), stF.size());
	EXPECT_EQ(oaF[0], stF[0]);
	auto oaS = oa.subSpan(1, oa::Span<int>::DynamicExtent);
	auto stS = st.subspan(1);
	EXPECT_EQ(oaS.size(), stS.size());
	EXPECT_EQ(oaS[0], stS[0]);
}

TEST(Span, InvalidRangesRejectContract) {
	int value = 1;
	oa::Span<int> span(&value, 1);
	EXPECT_DEATH((void)oa::Span<int>(nullptr, 1),
		"OA contract failed: inPtr != nullptr \\|\\| inCount == 0");
	EXPECT_DEATH((void)span.first(2), "OA contract failed: inCount <= size_");
	EXPECT_DEATH((void)span.subSpan(2), "OA contract failed: inOffset <= size_");
	EXPECT_DEATH((void)span.subSpan(0, 2),
		"OA contract failed: inCount == DynamicExtent \\|\\| ext <= rem");
	EXPECT_DEATH(
		(void)oa::Span<int>(&value, static_cast<oa::Usize>(-1)),
		"OA contract failed: inCount <= static_cast<size_type>\\(-1\\) / sizeof\\(T\\)");
}

TEST(Span, EmptyAccessRejectsContractWithoutNullArithmetic) {
	oa::Span<int> span;
	EXPECT_EQ(span.begin(), nullptr);
	EXPECT_EQ(span.end(), nullptr);
	EXPECT_EQ(span.subSpan(0).data(), nullptr);
	EXPECT_DEATH((void)span.front(), "OA contract failed: !empty\\(\\)");
	EXPECT_DEATH((void)span.back(), "OA contract failed: !empty\\(\\)");
}

TEST(Span, SubSpanMicrobenchVsStd) {
	std::array<int, 256> data{};
	for (std::size_t i = 0; i < data.size(); ++i) {
		data[i] = static_cast<int>(i);
	}
	oa::Span<int> oa(data.data(), data.size());
	std::span<int> st(data.data(), data.size());
	constexpr int kLoops = 200'000;

	const auto t0 = oa::highResolutionNow();
	volatile int sink = 0;
	for (int n = 0; n < kLoops; ++n) {
		const std::size_t off = static_cast<std::size_t>(n) & 63U;
		auto s = oa.subSpan(off, 32);
		sink += s[0] + s[31];
	}
	const auto t1 = oa::highResolutionNow();
	for (int n = 0; n < kLoops; ++n) {
		const std::size_t off = static_cast<std::size_t>(n) & 63U;
		auto s = st.subspan(off, 32);
		sink += s[0] + s[31];
	}
	const auto t2 = oa::highResolutionNow();

	stdReportCompareSequentialRuns(
		"oa::Span::subSpan x200k", t0, t1, "std::span::subspan x200k", t2);
	EXPECT_NE(sink, 0);
}
