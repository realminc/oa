#include "oaStdTest.h"

#include <oa/core/std/assert.h>

TEST(StdAssert, BuildModeContract) {
	int evaluations = 0;
	OA_ASSERT(++evaluations == 1);
#ifdef NDEBUG
	EXPECT_EQ(evaluations, 0);
#else
	EXPECT_EQ(evaluations, 1);
#endif
}

TEST(StdAssert, RequireAlwaysEvaluates) {
	int evaluations = 0;
	OA_REQUIRE(++evaluations == 1);
	EXPECT_EQ(evaluations, 1);
}

TEST(StdAssert, RequireReportsExpressionAndTerminates) {
	EXPECT_DEATH(OA_REQUIRE(false), "OA contract failed: false");
}

TEST(StdAssert, RequireMessageReportsAndTerminates) {
	EXPECT_DEATH(OA_REQUIRE_MSG(false, "explicit contract message"),
		"OA contract failed: false: explicit contract message");
}

#ifndef NDEBUG
TEST(StdAssert, FailureReportsExpressionAndTerminates) {
	EXPECT_DEATH(OA_ASSERT(false), "OA assertion failed: false");
}
#endif
