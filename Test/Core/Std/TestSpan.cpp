#include "OaStdTest.h"


TEST(OaStdSpan, FromArray) {
	std::array<int, 3> data = {1, 2, 3};
	OaStdSpan<int> sp(data);
	ASSERT_EQ(sp.Size(), 3U);
	EXPECT_EQ(sp[1], 2);
}

TEST(OaStdSpan, StdSpanParity) {
	std::array<int, 4> data = {10, 20, 30, 40};
	OaStdSpan<int> oa(data);
	std::span<int> st(data.data(), data.size());
	EXPECT_EQ(oa.StdSpan().data(), st.data());
	EXPECT_EQ(oa.StdSpan().size(), st.size());
}

TEST(OaStdSpan, FirstSubSpanMatchesStd) {
	std::array<int, 5> data = {1, 2, 3, 4, 5};
	OaStdSpan<int> oa(data);
	std::span<int> st(data.data(), data.size());
	auto oaF = oa.First(3);
	auto stF = st.first<3>();
	EXPECT_EQ(oaF.StdSpan().size(), stF.size());
	EXPECT_EQ(oaF[0], stF[0]);
	auto oaS = oa.SubSpan(1, OaStdSpan<int>::DynamicExtent);
	auto stS = st.subspan(1);
	EXPECT_EQ(oaS.Size(), stS.size());
	EXPECT_EQ(oaS[0], stS[0]);
}

TEST(OaStdSpan, SubSpanMicrobenchVsStd) {
	std::array<int, 256> data{};
	for (std::size_t i = 0; i < data.size(); ++i) {
		data[i] = static_cast<int>(i);
	}
	OaStdSpan<int> oa(data);
	std::span<int> st(data.data(), data.size());
	constexpr int kLoops = 200'000;

	const auto t0 = OaHighResolutionNow();
	volatile int sink = 0;
	for (int n = 0; n < kLoops; ++n) {
		const std::size_t off = static_cast<std::size_t>(n) & 63U;
		auto s = oa.SubSpan(off, 32);
		sink += s[0] + s[31];
	}
	const auto t1 = OaHighResolutionNow();
	for (int n = 0; n < kLoops; ++n) {
		const std::size_t off = static_cast<std::size_t>(n) & 63U;
		auto s = st.subspan(off, 32);
		sink += s[0] + s[31];
	}
	const auto t2 = OaHighResolutionNow();

	OaStdReportCompareSequentialRuns(
		"OaStdSpan::SubSpan x200k", t0, t1, "std::span::subspan x200k", t2);
	EXPECT_NE(sink, 0);
}
