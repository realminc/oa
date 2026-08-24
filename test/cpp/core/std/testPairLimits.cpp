#include "oaStdTest.h"

#include <oa/core/std/pair.h>
#include <oa/core/std/limits.h>

#include <limits>

TEST(Pair, Basic) {
	oa::Pair<int, float> p(3, 1.5F);
	EXPECT_EQ(p.first, 3);
	EXPECT_EQ(p.second, 1.5F);

	auto q = oa::makePair(3, 1.5F);
	EXPECT_TRUE(p == q);

	oa::Pair<int, float> r(4, 1.5F);
	EXPECT_TRUE(p != r);
}

TEST(Pair, DefaultValueInit) {
	oa::Pair<int, int> p;
	EXPECT_EQ(p.first, 0);
	EXPECT_EQ(p.second, 0);
}

TEST(Limits, MatchesStdNumericLimits) {
	EXPECT_EQ(oa::Limits<int>::max(), std::numeric_limits<int>::max());
	EXPECT_EQ(oa::Limits<int>::min(), std::numeric_limits<int>::min());
	EXPECT_EQ(oa::Limits<float>::lowest(), std::numeric_limits<float>::lowest());
	EXPECT_EQ(oa::Limits<double>::epsilon(), std::numeric_limits<double>::epsilon());
	EXPECT_TRUE(oa::Limits<int>::isSigned);
	EXPECT_TRUE(oa::Limits<int>::isInteger);
	EXPECT_FALSE(oa::Limits<float>::isInteger);
	EXPECT_TRUE(oa::Limits<float>::hasNaN);
}
