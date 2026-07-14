#include "OaStdTest.h"

#include <Oa/Core/Std/CString.h>

#include <cstring>

static int Sign(int InX) {
	return (InX > 0) - (InX < 0);
}

TEST(OaStdCString, Strlen) {
	EXPECT_EQ(OaStdStrlen(""), 0u);
	EXPECT_EQ(OaStdStrlen("hello"), 5u);
	EXPECT_EQ(OaStdStrlen(nullptr), 0u);
	EXPECT_EQ(OaStdStrlen("hello world"), std::strlen("hello world"));
}

TEST(OaStdCString, StrcmpSignParityWithStd) {
	const char* pairs[][2] = {
		{"a", "a"}, {"a", "b"}, {"b", "a"},
		{"abc", "abd"}, {"abc", "ab"}, {"ab", "abc"}, {"", "a"}, {"a", ""},
	};
	for (auto& p : pairs) {
		EXPECT_EQ(Sign(OaStdStrcmp(p[0], p[1])), Sign(std::strcmp(p[0], p[1])))
			<< "'" << p[0] << "' vs '" << p[1] << "'";
	}
}

TEST(OaStdCString, Strncmp) {
	EXPECT_EQ(OaStdStrncmp("abcXX", "abcYY", 3), 0);
	EXPECT_NE(OaStdStrncmp("abcXX", "abcYY", 4), 0);
	EXPECT_EQ(OaStdStrncmp("abc", "abc", 10), 0);  // stops at terminator
}

TEST(OaStdCString, Strchr) {
	const char* s = "hello";
	EXPECT_EQ(OaStdStrchr(s, 'l'), s + 2);
	EXPECT_EQ(OaStdStrchr(s, 'z'), nullptr);
	EXPECT_EQ(OaStdStrchr(s, '\0'), s + 5);  // finds the terminator, like strchr
}

TEST(OaStdCString, Memcmp) {
	EXPECT_EQ(OaStdMemcmp("abc", "abc", 3), 0);
	EXPECT_EQ(Sign(OaStdMemcmp("abc", "abd", 3)), Sign(std::memcmp("abc", "abd", 3)));
	EXPECT_EQ(Sign(OaStdMemcmp("abd", "abc", 3)), Sign(std::memcmp("abd", "abc", 3)));
}
