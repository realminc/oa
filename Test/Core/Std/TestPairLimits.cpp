#include "OaStdTest.h"

#include <Oa/Core/Std/Pair.h>
#include <Oa/Core/Std/Limits.h>

#include <limits>

TEST(OaStdPair, Basic) {
	OaStdPair<int, float> p(3, 1.5F);
	EXPECT_EQ(p.First, 3);
	EXPECT_EQ(p.Second, 1.5F);

	auto q = OaStdMakePair(3, 1.5F);
	EXPECT_TRUE(p == q);

	OaStdPair<int, float> r(4, 1.5F);
	EXPECT_TRUE(p != r);
}

TEST(OaStdPair, DefaultValueInit) {
	OaStdPair<int, int> p;
	EXPECT_EQ(p.First, 0);
	EXPECT_EQ(p.Second, 0);
}

TEST(OaStdLimits, MatchesStdNumericLimits) {
	EXPECT_EQ(OaStdLimits<int>::Max(), std::numeric_limits<int>::max());
	EXPECT_EQ(OaStdLimits<int>::Min(), std::numeric_limits<int>::min());
	EXPECT_EQ(OaStdLimits<float>::Lowest(), std::numeric_limits<float>::lowest());
	EXPECT_EQ(OaStdLimits<double>::Epsilon(), std::numeric_limits<double>::epsilon());
	EXPECT_TRUE(OaStdLimits<int>::IsSigned);
	EXPECT_TRUE(OaStdLimits<int>::IsInteger);
	EXPECT_FALSE(OaStdLimits<float>::IsInteger);
	EXPECT_TRUE(OaStdLimits<float>::HasNaN);
}
