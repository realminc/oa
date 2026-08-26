#include "oaStdTest.h"

#include <random>
#include <string>
#include <type_traits>

static_assert(not std::is_constructible_v<oa::String, std::string>);

TEST(String, AppendView) {
	oa::String s("ab");
	s.append(oa::StringView("cd"));
	EXPECT_EQ(testStdString(s), "abcd");
}

TEST(String, SsoMaxThenHeap) {
	std::string s22(22, 'x');
	oa::String a(s22.c_str());
	EXPECT_EQ(a.size(), 22U);
	EXPECT_EQ(testStdString(a), s22);
	a.pushBack('y');
	EXPECT_EQ(a.size(), 23U);
	EXPECT_EQ(testStdString(a), s22 + 'y');
}

TEST(String, ReservedHeapAppendKeepsStorageAndTerminator) {
	oa::String string;
	string.reserve(64);
	const char* const storage = string.data();
	for (oa::Usize index = 0; index < 64; ++index) {
		string.pushBack(static_cast<char>('a' + index % 26));
	}

	EXPECT_EQ(string.data(), storage);
	EXPECT_EQ(string.size(), 64U);
	EXPECT_EQ(string.capacity(), 64U);
	EXPECT_EQ(string.cStr()[64], '\0');
}

TEST(String, ResizeShrinkToSso) {
	oa::String s("01234567890123456789012");
	ASSERT_EQ(s.size(), 23U);
	s.resize(3);
	EXPECT_EQ(testStdString(s), "012");
}

TEST(String, ParityWithStdString) {
	const char* lit = "realm";
	oa::String oa(lit);
	std::string st(lit);
	EXPECT_EQ(testStdString(oa), st);
	EXPECT_TRUE(oa.equals(oa::String(st.c_str())));
}

TEST(String, AtRejectsOutOfRange) {
	oa::String s("a");
	EXPECT_DEATH(static_cast<void>(s.at(1)), "OA contract failed: inIdx < size\\(\\)");
}

TEST(String, EmptyElementAccessRejectsContract) {
	oa::String string;
	EXPECT_DEATH(static_cast<void>(string.front()), "OA contract failed: !empty\\(\\)");
	EXPECT_DEATH(static_cast<void>(string.back()), "OA contract failed: !empty\\(\\)");
	EXPECT_DEATH(string.popBack(), "OA contract failed: !empty\\(\\)");
}

TEST(String, MoveIsEmptySource) {
	oa::String a("hello");
	oa::String b(std::move(a));
	EXPECT_EQ(testStdString(b), "hello");
	EXPECT_TRUE(a.empty());
}

TEST(String, MoveCoversEverySsoLengthAndHeap) {
	for (std::size_t length = 0; length <= oa::String::SsoCap + 1; ++length) {
		const std::string expected(length, static_cast<char>('a' + length % 26));

		oa::String constructorSource(expected.data(), expected.size());
		oa::String constructed(std::move(constructorSource));
		EXPECT_EQ(testStdString(constructed), expected) << "length=" << length;
		EXPECT_TRUE(constructorSource.empty()) << "length=" << length;

		oa::String assignmentSource(expected.data(), expected.size());
		oa::String assigned("existing heap storage that must be released");
		assigned = std::move(assignmentSource);
		EXPECT_EQ(testStdString(assigned), expected) << "length=" << length;
		EXPECT_TRUE(assignmentSource.empty()) << "length=" << length;
	}
}

TEST(StdStringVsStd, AppendSameBytesAsStdString) {
	oa::String oa;
	std::string st;
	std::minstd_rand rng(0xABCD1234u);
	for (int i = 0; i < 2000; ++i) {
		const char c = static_cast<char>('a' + (static_cast<int>(rng()) % 26));
		oa.pushBack(c);
		st.push_back(c);
	}
	stdEchoCurrentTest();
	stdExpectGotSize("string length", st.size(), oa.size());
	EXPECT_EQ(testStdString(oa), st);
}

TEST(StdStringVsStd, TimedPushBackWallUs) {
	constexpr int kIters = 100'000;
	auto runOa = [] {
		oa::String s;
		s.reserve(128);
		std::minstd_rand rng(0x51DEu);
		for (int i = 0; i < kIters; ++i) {
			s.pushBack(static_cast<char>('0' + (static_cast<int>(rng()) % 10)));
		}
		return s.size();
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
	const auto t0 = oa::highResolutionNow();
	const auto szOa = runOa();
	const auto t1 = oa::highResolutionNow();
	const auto szSt = runStd();
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::String::pushBack x100k", t0, t1, "std::string::push_back x100k", t2);
	stdExpectGotSize("string final size", szSt, szOa);
	EXPECT_EQ(szOa, szSt);
}
