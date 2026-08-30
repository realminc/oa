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
	EXPECT_EQ(oa::Limits<signed char>::lowest(), std::numeric_limits<signed char>::lowest());
	EXPECT_EQ(oa::Limits<unsigned char>::max(), std::numeric_limits<unsigned char>::max());
	EXPECT_EQ(oa::Limits<short>::min(), std::numeric_limits<short>::min());
	EXPECT_EQ(oa::Limits<int>::max(), std::numeric_limits<int>::max());
	EXPECT_EQ(oa::Limits<int>::min(), std::numeric_limits<int>::min());
	EXPECT_EQ(oa::Limits<unsigned int>::max(), std::numeric_limits<unsigned int>::max());
	EXPECT_EQ(oa::Limits<long>::digits, std::numeric_limits<long>::digits);
	EXPECT_EQ(oa::Limits<unsigned long long>::digits10,
		std::numeric_limits<unsigned long long>::digits10);
	EXPECT_EQ(oa::Limits<float>::min(), std::numeric_limits<float>::min());
	EXPECT_EQ(oa::Limits<float>::lowest(), std::numeric_limits<float>::lowest());
	EXPECT_EQ(oa::Limits<float>::max(), std::numeric_limits<float>::max());
	EXPECT_EQ(oa::Limits<float>::epsilon(), std::numeric_limits<float>::epsilon());
	EXPECT_EQ(oa::Limits<double>::epsilon(), std::numeric_limits<double>::epsilon());
	EXPECT_EQ(oa::Limits<double>::digits, std::numeric_limits<double>::digits);
	EXPECT_EQ(oa::Limits<double>::digits10, std::numeric_limits<double>::digits10);
	EXPECT_TRUE(oa::Limits<float>::infinity() > oa::Limits<float>::max());
	EXPECT_TRUE(oa::Limits<float>::quietNaN() != oa::Limits<float>::quietNaN());
	EXPECT_TRUE(oa::Limits<int>::isSigned);
	static_assert(oa::IsSignedV<int>);
	static_assert(not oa::IsSignedV<unsigned int>);
	static_assert(not oa::IsSignedV<float>);
	EXPECT_TRUE(oa::Limits<int>::isInteger);
	EXPECT_FALSE(oa::Limits<float>::isInteger);
	EXPECT_TRUE(oa::Limits<float>::hasNaN);
}
