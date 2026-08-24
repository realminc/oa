#include "oaStdTest.h"

#include <stdexcept>
#include <string_view>

TEST(StringView, SubStr) {
	oa::StringView v("hello");
	EXPECT_EQ(v.size(), 5U);
	oa::StringView sub = v.subStr(1, 3);
	EXPECT_EQ(sub.stdView(), "ell");
}

TEST(StringView, SubStrMatchesStdStringView) {
	const char lit[] = "hello";
	oa::StringView oa(lit);
	std::string_view st(lit);
	auto oaSub = oa.subStr(1, 3);
	auto stSub = st.substr(1, 3);
	stdEchoCurrentTest();
	stdExpectGotSize("string_view substr len", stSub.size(), oaSub.size());
	EXPECT_EQ(oaSub.stdView(), stSub);
	EXPECT_EQ(oa.subStr(2).stdView(), st.substr(2));
}

TEST(StringView, CompareEqualsMatchStd) {
	oa::StringView a("abc");
	oa::StringView b("abd");
	std::string_view sa("abc");
	std::string_view sb("abd");
	EXPECT_EQ(a.equals(b), (sa == sb));
	EXPECT_EQ(a.compare(b) < 0, (sa < sb));
	EXPECT_TRUE(a.equals(oa::StringView("abc")));
}

TEST(StringView, AtThrowsOutOfRange) {
	oa::StringView v("x");
	EXPECT_EQ(v.at(0), 'x');
	EXPECT_THROW((void)v.at(1), std::out_of_range);
}

TEST(StringView, NullPtrIsEmpty) {
	oa::StringView v(nullptr);
	EXPECT_TRUE(v.empty());
	EXPECT_EQ(v.data(), nullptr);
}

TEST(StringView, SubStrCompareMicrobenchVsStd) {
	static constexpr const char kLit[] =
		"realm.software oastd string_view substr compare parity bench";
	oa::StringView oa(kLit);
	std::string_view st(kLit);
	constexpr int kLoops = 500'000;
	oa::StringView needle("realm");
	std::string_view needleSt("realm");

	const auto t0 = oa::highResolutionNow();
	volatile int sink = 0;
	for (int n = 0; n < kLoops; ++n) {
		const std::size_t a = static_cast<std::size_t>(n) & 15U;
		const std::size_t b = 8 + ((static_cast<std::size_t>(n) >> 2) & 7U);
		sink += static_cast<int>(oa.subStr(a, b).size());
		sink += oa.subStr(a, b).compare(needle);
	}
	const auto t1 = oa::highResolutionNow();
	for (int n = 0; n < kLoops; ++n) {
		const std::size_t a = static_cast<std::size_t>(n) & 15U;
		const std::size_t b = 8 + ((static_cast<std::size_t>(n) >> 2) & 7U);
		sink += static_cast<int>(st.substr(a, b).size());
		sink += st.substr(a, b).compare(needleSt);
	}
	const auto t2 = oa::highResolutionNow();

	stdReportCompareSequentialRuns(
		"oa::StringView SubStr+Compare x500k", t0, t1,
		"std::string_view substr+compare x500k", t2);
	EXPECT_NE(sink, 0);
}
