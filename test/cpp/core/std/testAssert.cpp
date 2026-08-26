#include "oaStdTest.h"

#include <oa/core/assert.h>

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

#ifndef NDEBUG
TEST(StdAssert, FailureReportsExpressionAndTerminates) {
	EXPECT_DEATH(OA_ASSERT(false), "OA assertion failed: false");
}
#endif
