#include "OaStdTest.h"

#include <stdexcept>
#include <string_view>

TEST(OaStdStringView, SubStr) {
	OaStdStringView v("hello");
	EXPECT_EQ(v.Size(), 5U);
	OaStdStringView sub = v.SubStr(1, 3);
	EXPECT_EQ(sub.StdView(), "ell");
}

TEST(OaStdStringView, SubStrMatchesStdStringView) {
	const char lit[] = "hello";
	OaStdStringView oa(lit);
	std::string_view st(lit);
	auto oaSub = oa.SubStr(1, 3);
	auto stSub = st.substr(1, 3);
	OaStdEchoCurrentTest();
	OaStdExpectGotSize("string_view substr len", stSub.size(), oaSub.Size());
	EXPECT_EQ(oaSub.StdView(), stSub);
	EXPECT_EQ(oa.SubStr(2).StdView(), st.substr(2));
}

TEST(OaStdStringView, CompareEqualsMatchStd) {
	OaStdStringView a("abc");
	OaStdStringView b("abd");
	std::string_view sa("abc");
	std::string_view sb("abd");
	EXPECT_EQ(a.Equals(b), (sa == sb));
	EXPECT_EQ(a.Compare(b) < 0, (sa < sb));
	EXPECT_TRUE(a.Equals(OaStdStringView("abc")));
}

TEST(OaStdStringView, AtThrowsOutOfRange) {
	OaStdStringView v("x");
	EXPECT_EQ(v.At(0), 'x');
	EXPECT_THROW((void)v.At(1), std::out_of_range);
}

TEST(OaStdStringView, NullPtrIsEmpty) {
	OaStdStringView v(nullptr);
	EXPECT_TRUE(v.Empty());
	EXPECT_EQ(v.Data(), nullptr);
}

TEST(OaStdStringView, SubStrCompareMicrobenchVsStd) {
	static constexpr const char kLit[] =
		"realm.software oastd string_view substr compare parity bench";
	OaStdStringView oa(kLit);
	std::string_view st(kLit);
	constexpr int kLoops = 500'000;
	OaStdStringView needle("realm");
	std::string_view needleSt("realm");

	const auto t0 = OaHighResolutionNow();
	volatile int sink = 0;
	for (int n = 0; n < kLoops; ++n) {
		const std::size_t a = static_cast<std::size_t>(n) & 15U;
		const std::size_t b = 8 + ((static_cast<std::size_t>(n) >> 2) & 7U);
		sink += static_cast<int>(oa.SubStr(a, b).Size());
		sink += oa.SubStr(a, b).Compare(needle);
	}
	const auto t1 = OaHighResolutionNow();
	for (int n = 0; n < kLoops; ++n) {
		const std::size_t a = static_cast<std::size_t>(n) & 15U;
		const std::size_t b = 8 + ((static_cast<std::size_t>(n) >> 2) & 7U);
		sink += static_cast<int>(st.substr(a, b).size());
		sink += st.substr(a, b).compare(needleSt);
	}
	const auto t2 = OaHighResolutionNow();

	OaStdReportCompareSequentialRuns(
		"OaStdStringView SubStr+Compare x500k", t0, t1,
		"std::string_view substr+compare x500k", t2);
	EXPECT_NE(sink, 0);
}
