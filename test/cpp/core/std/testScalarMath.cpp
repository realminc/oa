#include "oaStdTest.h"

#include <oa/core/std/scalarMath.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

template<typename T>
void expectNearRelative(T inActual, T inExpected, T inTolerance) {
	const T scale = std::max<T>(T(1), std::abs(inExpected));
	EXPECT_LE(std::abs(inActual - inExpected), inTolerance * scale);
}

} // namespace

TEST(StdScalarMath, FloatParity) {
	constexpr float tolerance = 8.0F * std::numeric_limits<float>::epsilon();
	for (float value : {-3.25F, -1.0F, -0.25F, 0.0F, 0.25F, 1.0F, 3.25F}) {
		expectNearRelative(oa::abs(value), std::abs(value), tolerance);
		expectNearRelative(oa::floor(value), std::floor(value), tolerance);
		expectNearRelative(oa::ceil(value), std::ceil(value), tolerance);
		expectNearRelative(oa::round(value), std::round(value), tolerance);
		EXPECT_EQ(oa::lround(value), std::lround(value));
		expectNearRelative(oa::trunc(value), std::trunc(value), tolerance);
		expectNearRelative(oa::sin(value), std::sin(value), tolerance);
		expectNearRelative(oa::cos(value), std::cos(value), tolerance);
		expectNearRelative(oa::tan(value), std::tan(value), tolerance);
		expectNearRelative(oa::tanh(value), std::tanh(value), tolerance);
		expectNearRelative(oa::cbrt(value), std::cbrt(value), tolerance);
	}

	for (float value : {0.125F, 0.5F, 1.0F, 2.0F, 8.0F}) {
		expectNearRelative(oa::sqrt(value), std::sqrt(value), tolerance);
		expectNearRelative(oa::exp(value), std::exp(value), tolerance);
		expectNearRelative(oa::exp2(value), std::exp2(value), tolerance);
		expectNearRelative(oa::log(value), std::log(value), tolerance);
		expectNearRelative(oa::log2(value), std::log2(value), tolerance);
		expectNearRelative(oa::log10(value), std::log10(value), tolerance);
	}

	expectNearRelative(oa::pow(1.75F, 2.25F), std::pow(1.75F, 2.25F), tolerance);
	expectNearRelative(oa::atan2(0.75F, -0.5F), std::atan2(0.75F, -0.5F), tolerance);
	expectNearRelative(oa::fmod(7.25F, 2.0F), std::fmod(7.25F, 2.0F), tolerance);
	expectNearRelative(oa::fmin(7.25F, 2.0F), std::fmin(7.25F, 2.0F), tolerance);
	expectNearRelative(oa::fmax(7.25F, 2.0F), std::fmax(7.25F, 2.0F), tolerance);
	expectNearRelative(oa::copySign(2.5F, -1.0F), std::copysign(2.5F, -1.0F), tolerance);
}

TEST(StdScalarMath, DoubleParity) {
	constexpr double tolerance = 16.0 * std::numeric_limits<double>::epsilon();
	for (double value : {-0.9, -0.25, 0.0, 0.25, 0.9}) {
		expectNearRelative(oa::asin(value), std::asin(value), tolerance);
		expectNearRelative(oa::acos(value), std::acos(value), tolerance);
		expectNearRelative(oa::atan(value), std::atan(value), tolerance);
	}
	expectNearRelative(oa::pow(2.125, -3.5), std::pow(2.125, -3.5), tolerance);
	expectNearRelative(oa::atan2(-0.25, 0.75), std::atan2(-0.25, 0.75), tolerance);
	expectNearRelative(oa::copySign(-2.5, 1.0), std::copysign(-2.5, 1.0), tolerance);
}

TEST(StdScalarMath, ExceptionalClassificationParity) {
	const float infinity = std::numeric_limits<float>::infinity();
	const float nan = std::numeric_limits<float>::quiet_NaN();

	EXPECT_EQ(oa::isFinite(1.0F), std::isfinite(1.0F));
	EXPECT_EQ(oa::isFinite(infinity), std::isfinite(infinity));
	EXPECT_EQ(oa::isInf(infinity), std::isinf(infinity));
	EXPECT_EQ(oa::isNan(nan), std::isnan(nan));
	EXPECT_EQ(oa::fmin(nan, 2.0F), std::fmin(nan, 2.0F));
	EXPECT_EQ(oa::fmax(nan, 2.0F), std::fmax(nan, 2.0F));
	EXPECT_TRUE(oa::isNan(oa::sqrt(-1.0F)));
	EXPECT_TRUE(oa::isInf(oa::log(0.0F)));
}

TEST(StdScalarMath, IntegralAbsParity) {
	for (int value : {-1234567, -1, 0, 1, 1234567}) {
		EXPECT_EQ(oa::abs(value), std::abs(value));
	}
	for (long long value : {-9'876'543'210LL, -1LL, 0LL, 1LL, 9'876'543'210LL}) {
		EXPECT_EQ(oa::abs(value), std::abs(value));
	}
}
