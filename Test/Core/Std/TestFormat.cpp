#include "OaStdTest.h"

#include <Oa/Core/Std/Format.h>

#include <cstring>
#include <string>

TEST(OaFormat, IntegerToString) {
	EXPECT_STREQ(OaToString(OaU32{0}).CStr(), "0");
	EXPECT_STREQ(OaToString(OaU32{4294967295u}).CStr(), "4294967295");
	EXPECT_STREQ(OaToString(OaI64{-9223372036854775807LL}).CStr(), "-9223372036854775807");
}

TEST(OaFormat, FloatToString) {
	EXPECT_STREQ(OaToString(0.0).CStr(), "0");
	EXPECT_STREQ(OaToString(1.5).CStr(), "1.5");
	EXPECT_STREQ(OaToString(1.5F).CStr(), "1.5");   // float overload
	EXPECT_STREQ(OaToString(100.0).CStr(), "100");  // %g stays compact
	EXPECT_STREQ(OaToString(-2.25).CStr(), "-2.25");
}

TEST(OaFormat, Basic) {
	OaString s = OaFormat("%s=%d (%.2f)", "x", 42, 3.14159);
	EXPECT_STREQ(s.CStr(), "x=42 (3.14)");
	EXPECT_EQ(s.Size(), std::strlen("x=42 (3.14)"));
}

TEST(OaFormat, Empty) {
	OaString s = OaFormat("%s", "");
	EXPECT_EQ(s.Size(), 0u);
	EXPECT_STREQ(s.CStr(), "");
}

TEST(OaFormat, LongExceedsStackBuffer) {
	// > 256 chars exercises the heap fallback path.
	std::string big(1000, 'a');
	OaString s = OaFormat("%s", big.c_str());
	EXPECT_EQ(s.Size(), 1000u);
	EXPECT_STREQ(s.CStr(), big.c_str());
}
