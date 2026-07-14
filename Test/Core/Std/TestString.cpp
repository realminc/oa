#include "OaStdTest.h"

#include <random>
#include <string>

TEST(OaStdString, AppendView) {
	OaStdString s("ab");
	s.Append(OaStdStringView("cd"));
	EXPECT_EQ(s.StdStr(), "abcd");
}

TEST(OaStdString, SsoMaxThenHeap) {
	std::string s22(22, 'x');
	OaStdString a(s22.c_str());
	EXPECT_EQ(a.Size(), 22U);
	EXPECT_EQ(a.StdStr(), s22);
	a.PushBack('y');
	EXPECT_EQ(a.Size(), 23U);
	EXPECT_EQ(a.StdStr(), s22 + 'y');
}

TEST(OaStdString, ResizeShrinkToSso) {
	OaStdString s("01234567890123456789012");
	ASSERT_EQ(s.Size(), 23U);
	s.Resize(3);
	EXPECT_EQ(s.StdStr(), "012");
}

TEST(OaStdString, ParityWithStdString) {
	const char* lit = "realm";
	OaStdString oa(lit);
	std::string st(lit);
	EXPECT_EQ(oa.StdStr(), st);
	EXPECT_TRUE(oa.Equals(OaStdString(st.c_str())));
}

TEST(OaStdString, AtThrowsOutOfRange) {
	OaStdString s("a");
	EXPECT_THROW(static_cast<void>(s.At(1)), std::out_of_range);
}

TEST(OaStdString, MoveIsEmptySource) {
	OaStdString a("hello");
	OaStdString b(std::move(a));
	EXPECT_EQ(b.StdStr(), "hello");
	EXPECT_TRUE(a.Empty());
}

TEST(OaStdStringVsStd, AppendSameBytesAsStdString) {
	OaStdString oa;
	std::string st;
	std::minstd_rand rng(0xABCD1234u);
	for (int i = 0; i < 2000; ++i) {
		const char c = static_cast<char>('a' + (static_cast<int>(rng()) % 26));
		oa.PushBack(c);
		st.push_back(c);
	}
	OaStdEchoCurrentTest();
	OaStdExpectGotSize("string length", st.size(), oa.Size());
	EXPECT_EQ(oa.StdStr(), st);
}

TEST(OaStdStringVsStd, TimedPushBackWallUs) {
	constexpr int kIters = 100'000;
	auto runOa = [] {
		OaStdString s;
		s.Reserve(128);
		std::minstd_rand rng(0x51DEu);
		for (int i = 0; i < kIters; ++i) {
			s.PushBack(static_cast<char>('0' + (static_cast<int>(rng()) % 10)));
		}
		return s.Size();
	};
	auto runStd = [] {
		std::string s;
		s.reserve(128);
		std::minstd_rand rng(0x51DEu);
		for (int i = 0; i < kIters; ++i) {
			s.push_back(static_cast<char>('0' + (static_cast<int>(rng()) % 10)));
		}
		return s.size();
	};
	const auto t0 = OaHighResolutionNow();
	const auto szOa = runOa();
	const auto t1 = OaHighResolutionNow();
	const auto szSt = runStd();
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdString::PushBack x100k", t0, t1, "std::string::push_back x100k", t2);
	OaStdExpectGotSize("string final size", szSt, szOa);
	EXPECT_EQ(szOa, szSt);
}
