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

TEST(StdCString, Memcmp) {
	EXPECT_EQ(oa::memcmp("abc", "abc", 3), 0);
	EXPECT_EQ(sign(oa::memcmp("abc", "abd", 3)), sign(std::memcmp("abc", "abd", 3)));
	EXPECT_EQ(sign(oa::memcmp("abd", "abc", 3)), sign(std::memcmp("abd", "abc", 3)));
}
