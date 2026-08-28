#include "oaStdTest.h"

TEST(StringView, RejectsNonemptyNullRange) {
	EXPECT_DEATH((static_cast<void>(oa::StringView(nullptr, 1))),
		"OA contract failed: ins != nullptr \\|\\| inlen == 0");
}

TEST(StringView, FindRejectsNullWithNonzeroCount) {
	const oa::StringView value("abc");
	EXPECT_DEATH(static_cast<void>(value.find(nullptr, 0, 1)),
		"OA contract failed: ins != nullptr \\|\\| incount == 0");
	EXPECT_EQ(value.find(nullptr, 0, 0), 0U);
}

#include <string_view>

TEST(StringView, SubStr) {
	oa::StringView v("hello");
	EXPECT_EQ(v.size(), 5U);
	oa::StringView sub = v.subStr(1, 3);
	EXPECT_EQ(sub, "ell");
}

TEST(StringView, SubStrMatchesStdStringView) {
	const char lit[] = "hello";
	oa::StringView oa(lit);
	std::string_view st(lit);
	auto oaSub = oa.subStr(1, 3);
	auto stSub = st.substr(1, 3);
	stdEchoCurrentTest();
	stdExpectGotSize("string_view substr len", stSub.size(), oaSub.size());
	EXPECT_EQ(oaSub, oa::StringView(stSub.data(), stSub.size()));
	auto stTail = st.substr(2);
	EXPECT_EQ(oa.subStr(2), oa::StringView(stTail.data(), stTail.size()));
}

TEST(StringView, ReverseFindMatchesStdStringView) {
	oa::StringView value("host:port:tail");
	std::string_view standard("host:port:tail");
	EXPECT_EQ(value.rfind(':'), standard.rfind(':'));
	EXPECT_EQ(value.rfind(':', 7U), standard.rfind(':', 7U));
	EXPECT_EQ(value.rfind('x'), oa::StringView::Npos);
	EXPECT_EQ(oa::StringView{}.rfind('x'), oa::StringView::Npos);
}

TEST(StringView, CharacterFindMatchesStdAcrossBoundsAndByteValues) {
	const char bytes[] = {'a', '\0', static_cast<char>(0xFF), 'a'};
	const oa::StringView value(bytes, sizeof(bytes));
	const std::string_view standard(bytes, sizeof(bytes));
	const char needles[] = {'a', '\0', static_cast<char>(0xFF), 'z'};
	for (const char needle : needles) {
		for (oa::Usize position = 0; position <= sizeof(bytes) + 2U; ++position) {
			EXPECT_EQ(value.find(needle, position), standard.find(needle, position))
				<< "position=" << position
				<< " needle=" << static_cast<unsigned int>(
					static_cast<unsigned char>(needle));
		}
	}
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

TEST(StringView, AtRejectsOutOfRange) {
	oa::StringView v("x");
	EXPECT_EQ(v.at(0), 'x');
	EXPECT_DEATH((void)v.at(1), "OA contract failed: inidx < len_");
}

TEST(StringView, SubStrRejectsOutOfRange) {
	oa::StringView view("x");
	EXPECT_DEATH((void)view.subStr(2), "OA contract failed: inpos <= len_");
}

TEST(StringView, NullPtrIsEmpty) {
	oa::StringView v(nullptr);
	EXPECT_TRUE(v.empty());
	EXPECT_EQ(v.data(), nullptr);
	v.removePrefix(0);
	EXPECT_EQ(v.data(), nullptr);
}

TEST(StringView, NativeEmptyHashMatchesEmptyText) {
	const oa::StringView value;
	EXPECT_EQ(oa::KeyHash<oa::StringView>{}(value), oa::hashTextKey(""));
}

TEST(StringView, EmptyElementAccessRejectsContract) {
	oa::StringView view;
	EXPECT_DEATH(static_cast<void>(view.front()), "OA contract failed: !empty\\(\\)");
	EXPECT_DEATH(static_cast<void>(view.back()), "OA contract failed: !empty\\(\\)");
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
