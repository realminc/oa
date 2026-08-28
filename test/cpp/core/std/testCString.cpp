#include "oaStdTest.h"

#include <oa/core/std/cString.h>

#include <cstring>

static int sign(int inX) {
	return (inX > 0) - (inX < 0);
}

TEST(StdCString, Strlen) {
	EXPECT_EQ(oa::strlen(""), 0u);
	EXPECT_EQ(oa::strlen("hello"), 5u);
	EXPECT_EQ(oa::strlen(nullptr), 0u);
	EXPECT_EQ(oa::strlen("hello world"), std::strlen("hello world"));
}

TEST(StdCString, NullContractsAreExplicit) {
	EXPECT_EQ(oa::strncmp(nullptr, nullptr, 0), 0);
	EXPECT_DEATH(
		static_cast<void>(oa::strcmp(nullptr, "")),
		"strcmp requires a left C string");
	EXPECT_DEATH(
		static_cast<void>(oa::strcmp("", nullptr)),
		"strcmp requires a right C string");
	EXPECT_DEATH(
		static_cast<void>(oa::strncmp(nullptr, "x", 1)),
		"strncmp requires a left byte range");
	EXPECT_DEATH(
		static_cast<void>(oa::strncmp("x", nullptr, 1)),
		"strncmp requires a right byte range");
	EXPECT_DEATH(
		static_cast<void>(oa::strchr(nullptr, 'x')),
		"strchr requires a C string");
}

TEST(StdCString, StrcmpSignParityWithStd) {
	const char* pairs[][2] = {
		{"a", "a"}, {"a", "b"}, {"b", "a"},
		{"abc", "abd"}, {"abc", "ab"}, {"ab", "abc"}, {"", "a"}, {"a", ""},
	};
	for (auto& p : pairs) {
		EXPECT_EQ(sign(oa::strcmp(p[0], p[1])), sign(std::strcmp(p[0], p[1])))
			<< "'" << p[0] << "' vs '" << p[1] << "'";
	}
}

TEST(StdCString, Strncmp) {
	EXPECT_EQ(oa::strncmp("abcXX", "abcYY", 3), 0);
	EXPECT_NE(oa::strncmp("abcXX", "abcYY", 4), 0);
	EXPECT_EQ(oa::strncmp("abc", "abc", 10), 0);  // stops at terminator
}

TEST(StdCString, Strchr) {
	const char* s = "hello";
	EXPECT_EQ(oa::strchr(s, 'l'), s + 2);
	EXPECT_EQ(oa::strchr(s, 'z'), nullptr);
	EXPECT_EQ(oa::strchr(s, '\0'), s + 5);  // finds the terminator, like strchr
}

TEST(StdCString, DifferentialLengthsAlignmentsAndHighBytes) {
	constexpr oa::Usize lengths[] = {
		0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128,
	};
	constexpr char needles[] = {
		'\0', 'a', 'w', static_cast<char>(0x80), static_cast<char>(0xFF),
	};

	for (oa::Usize offset = 0; offset < 16; ++offset) {
		for (const oa::Usize length : lengths) {
			char leftStorage[160]{};
			char rightStorage[160]{};
			char* const left = leftStorage + offset;
			char* const right = rightStorage + offset;
			for (oa::Usize index = 0; index < length; ++index) {
				const unsigned char byte = static_cast<unsigned char>(
					((index * 73U + offset * 29U) % 255U) + 1U);
				left[index] = static_cast<char>(byte);
				right[index] = static_cast<char>(byte);
			}
			left[length] = '\0';
			right[length] = '\0';

			EXPECT_EQ(oa::strlen(left), std::strlen(left));
			EXPECT_EQ(sign(oa::strcmp(left, right)), sign(std::strcmp(left, right)));
			for (const oa::Usize count : {
				oa::Usize{0}, length / 2U, length, length + 1U,
			}) {
				EXPECT_EQ(
					sign(oa::strncmp(left, right, count)),
					sign(std::strncmp(left, right, count)));
			}
			for (const char needle : needles) {
				const char* const oaFound = oa::strchr(left, needle);
				const char* const stdFound = std::strchr(
					left, static_cast<unsigned char>(needle));
				const oa::Isize oaOffset = oaFound == nullptr
					? -1 : static_cast<oa::Isize>(oaFound - left);
				const oa::Isize stdOffset = stdFound == nullptr
					? -1 : static_cast<oa::Isize>(stdFound - left);
				EXPECT_EQ(oaOffset, stdOffset);
			}

			if (length != 0) {
				const oa::Usize mismatch = length / 2U;
				right[mismatch] = left[mismatch] == static_cast<char>(1)
					? static_cast<char>(2) : static_cast<char>(1);
				EXPECT_EQ(sign(oa::strcmp(left, right)), sign(std::strcmp(left, right)));
				for (const oa::Usize count : {
					mismatch, mismatch + 1U, length + 1U,
				}) {
					EXPECT_EQ(
						sign(oa::strncmp(left, right, count)),
						sign(std::strncmp(left, right, count)));
				}
			}
		}
	}
}

TEST(StdCString, Memcmp) {
	EXPECT_EQ(oa::memcmp("abc", "abc", 3), 0);
	EXPECT_EQ(sign(oa::memcmp("abc", "abd", 3)), sign(std::memcmp("abc", "abd", 3)));
	EXPECT_EQ(sign(oa::memcmp("abd", "abc", 3)), sign(std::memcmp("abd", "abc", 3)));
}
