#include "oaStdTest.h"

#include <oa/core/std/format.h>

#include <cstring>
#include <string>

TEST(format, IntegerToString) {
	EXPECT_STREQ(oa::toString(oa::U32{0}).cStr(), "0");
	EXPECT_STREQ(oa::toString(oa::U32{4294967295u}).cStr(), "4294967295");
	EXPECT_STREQ(oa::toString(oa::I64{-9223372036854775807LL}).cStr(), "-9223372036854775807");
}

TEST(format, FloatToString) {
	EXPECT_STREQ(oa::toString(0.0).cStr(), "0");
	EXPECT_STREQ(oa::toString(1.5).cStr(), "1.5");
	EXPECT_STREQ(oa::toString(1.5F).cStr(), "1.5");   // float overload
	EXPECT_STREQ(oa::toString(100.0).cStr(), "100");  // %g stays compact
	EXPECT_STREQ(oa::toString(-2.25).cStr(), "-2.25");
}

TEST(format, Basic) {
	oa::String s = oa::format("%s=%d (%.2f)", "x", 42, 3.14159);
	EXPECT_STREQ(s.cStr(), "x=42 (3.14)");
	EXPECT_EQ(s.size(), std::strlen("x=42 (3.14)"));
}

TEST(format, Empty) {
	oa::String s = oa::format("%s", "");
	EXPECT_EQ(s.size(), 0u);
	EXPECT_STREQ(s.cStr(), "");
}

TEST(format, LongExceedsStackBuffer) {
	// > 256 chars exercises the heap fallback path.
	std::string big(1000, 'a');
	oa::String s = oa::format("%s", big.c_str());
	EXPECT_EQ(s.size(), 1000u);
	EXPECT_STREQ(s.cStr(), big.c_str());
}
