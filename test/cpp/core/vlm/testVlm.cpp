#include <gtest/gtest.h>

#include <oa/core/vlm.h>

#include <cmath>
#include <limits>
#include <type_traits>

namespace {

template <typename T>
oa::vlm::detail::Vec4<T> transformOracle(
	const oa::vlm::detail::Vec4<T>& inValue,
	const oa::vlm::detail::Mat4<T>& inMatrix
) {
	return {
		inValue.x * inMatrix.m[0][0] + inValue.y * inMatrix.m[1][0]
			+ inValue.z * inMatrix.m[2][0] + inValue.w * inMatrix.m[3][0],
		inValue.x * inMatrix.m[0][1] + inValue.y * inMatrix.m[1][1]
			+ inValue.z * inMatrix.m[2][1] + inValue.w * inMatrix.m[3][1],
		inValue.x * inMatrix.m[0][2] + inValue.y * inMatrix.m[1][2]
			+ inValue.z * inMatrix.m[2][2] + inValue.w * inMatrix.m[3][2],
		inValue.x * inMatrix.m[0][3] + inValue.y * inMatrix.m[1][3]
			+ inValue.z * inMatrix.m[2][3] + inValue.w * inMatrix.m[3][3],
	};
}

static_assert(std::is_standard_layout_v<oa::vlm::Vec2>);
static_assert(std::is_standard_layout_v<oa::vlm::Vec3>);
static_assert(std::is_standard_layout_v<oa::vlm::Vec4>);
static_assert(std::is_standard_layout_v<oa::vlm::Quat>);
static_assert(std::is_standard_layout_v<oa::vlm::Mat4>);
static_assert(std::is_trivially_copyable_v<oa::vlm::Vec2>);
static_assert(std::is_trivially_copyable_v<oa::vlm::Vec3>);
static_assert(std::is_trivially_copyable_v<oa::vlm::Vec4>);
static_assert(std::is_trivially_copyable_v<oa::vlm::Quat>);
static_assert(std::is_trivially_copyable_v<oa::vlm::Mat4>);
static_assert(sizeof(oa::vlm::Vec2) == sizeof(oa::F32) * 2);
static_assert(sizeof(oa::vlm::Vec3) == sizeof(oa::F32) * 3);
static_assert(sizeof(oa::vlm::Vec4) == sizeof(oa::F32) * 4);
static_assert(sizeof(oa::vlm::Quat) == sizeof(oa::F32) * 4);
static_assert(sizeof(oa::vlm::Mat4) == sizeof(oa::F32) * 16);
static_assert(sizeof(oa::vlm::DVec3) == sizeof(oa::F64) * 3);
static_assert(sizeof(oa::vlm::DQuat) == sizeof(oa::F64) * 4);
static_assert(sizeof(oa::vlm::DMat4) == sizeof(oa::F64) * 16);

} // namespace

TEST(VlmAbi, PackedValueContractsRemainStable) {
	EXPECT_EQ(alignof(oa::vlm::Vec3), alignof(oa::F32));
	EXPECT_EQ(alignof(oa::vlm::DVec3), alignof(oa::F64));
	EXPECT_EQ(oa::vlm::Quat::identity(), (oa::vlm::Quat{0.0F, 0.0F, 0.0F, 1.0F}));
}

TEST(VlmConstants, PreserveFloatAndDoublePrecision) {
	EXPECT_FLOAT_EQ(oa::vlm::degrees(oa::vlm::Pi<oa::F32>), 180.0F);
	EXPECT_DOUBLE_EQ(oa::vlm::degrees(oa::vlm::Pi<oa::F64>), 180.0);
	EXPECT_LT(std::abs(oa::vlm::Pi<oa::F64> - std::acos(-1.0)), 1.0e-15);
	EXPECT_LT(oa::vlm::InverseTolerance<oa::F64>, oa::vlm::Tolerance<oa::F64>);
}

TEST(VlmVector, OperatorsUseNamedAuthority) {
	const oa::vlm::Vec3 a{1.0F, 2.0F, 3.0F};
	const oa::vlm::Vec3 b{4.0F, 5.0F, 6.0F};
	EXPECT_EQ(a + b, oa::vlm::add(a, b));
	EXPECT_EQ(a - b, oa::vlm::sub(a, b));
	EXPECT_EQ(-a, oa::vlm::scale(a, -1.0F));
	EXPECT_EQ(a * 2.0F, oa::vlm::scale(a, 2.0F));
	EXPECT_EQ(2.0F * a, oa::vlm::scale(a, 2.0F));
	EXPECT_EQ(a / 2.0F, oa::vlm::divide(a, 2.0F));

	oa::vlm::Vec3 compound = a;
	compound += b;
	compound -= b;
	compound *= 4.0F;
	compound /= 2.0F;
	EXPECT_EQ(compound, oa::vlm::scale(a, 2.0F));
}

TEST(VlmVector, GeometryEssentialsMatchIndependentValues) {
	const oa::vlm::Vec3 x{1.0F, 0.0F, 0.0F};
	const oa::vlm::Vec3 y{0.0F, 1.0F, 0.0F};
	EXPECT_EQ(oa::vlm::cross(x, y), (oa::vlm::Vec3{0.0F, 0.0F, 1.0F}));
	EXPECT_FLOAT_EQ(oa::vlm::dot(x, y), 0.0F);
	EXPECT_FLOAT_EQ(oa::vlm::distanceSquared(x, y), 2.0F);
	EXPECT_NEAR(oa::vlm::distance(x, y), std::sqrt(2.0F), 1.0e-6F);
	EXPECT_EQ(
		oa::vlm::reflect({1.0F, -1.0F, 0.0F}, y),
		(oa::vlm::Vec3{1.0F, 1.0F, 0.0F}));
	EXPECT_EQ(
		oa::vlm::faceForward(y, {0.0F, -1.0F, 0.0F}, y), y);
	EXPECT_EQ(
		oa::vlm::faceForward(y, {0.0F, 1.0F, 0.0F}, y), -y);
	EXPECT_EQ(
		oa::vlm::refract({1.0F, 0.0F, 0.0F}, y, 2.0F), oa::vlm::Vec3{});
}

TEST(VlmVector, ComponentUtilitiesAreComponentWise) {
	const oa::vlm::Vec3 value{-3.0F, 9.0F, 2.0F};
	EXPECT_EQ(oa::vlm::abs(value), (oa::vlm::Vec3{3.0F, 9.0F, 2.0F}));
	EXPECT_EQ(
		oa::vlm::min(value, {0.0F, 8.0F, 4.0F}),
		(oa::vlm::Vec3{-3.0F, 8.0F, 2.0F}));
	EXPECT_EQ(
		oa::vlm::max(value, {0.0F, 8.0F, 4.0F}),
		(oa::vlm::Vec3{0.0F, 9.0F, 4.0F}));
	EXPECT_EQ(
		oa::vlm::clamp(value, {-1.0F, 0.0F, 0.0F}, {1.0F, 5.0F, 3.0F}),
		(oa::vlm::Vec3{-1.0F, 5.0F, 2.0F}));
	EXPECT_FLOAT_EQ(oa::vlm::componentMin(value), -3.0F);
	EXPECT_FLOAT_EQ(oa::vlm::componentMax(value), 9.0F);
}

TEST(VlmVector, DegenerateNormalizationIsFiniteAndDeterministic) {
	EXPECT_EQ(oa::vlm::normalize(oa::vlm::Vec3{}), oa::vlm::Vec3{});
	const oa::F32 infinity = std::numeric_limits<oa::F32>::infinity();
	EXPECT_EQ(oa::vlm::normalize(oa::vlm::Vec3{infinity, 0.0F, 0.0F}), oa::vlm::Vec3{});
	EXPECT_FALSE(oa::vlm::approximatelyEqual(infinity, infinity));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(1000.0F, 1000.005F));
	EXPECT_FALSE(oa::vlm::approximatelyEqual(1.0F, 1.1F));
}

TEST(VlmQuaternion, OperatorsUseNamedAuthority) {
	const oa::vlm::Quat a{1.0F, 2.0F, 3.0F, 4.0F};
	const oa::vlm::Quat b{-2.0F, 1.0F, 0.5F, 3.0F};
	EXPECT_EQ(a + b, oa::vlm::add(a, b));
	EXPECT_EQ(a * b, oa::vlm::quaternionMul(a, b));
	EXPECT_EQ(-a, oa::vlm::scale(a, -1.0F));
}

TEST(VlmQuaternion, CheckedInverseRejectsDegenerateInput) {
	const oa::vlm::Quat input{0.2F, -0.3F, 0.4F, 0.5F};
	oa::vlm::Quat inverse{};
	ASSERT_TRUE(oa::vlm::tryInverse(input, inverse));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		input * inverse, oa::vlm::Quat::identity(), 1.0e-6F, 1.0e-6F));

	const oa::vlm::Quat sentinel{1.0F, 2.0F, 3.0F, 4.0F};
	inverse = sentinel;
	EXPECT_FALSE(oa::vlm::tryInverse(oa::vlm::Quat{0.0F, 0.0F, 0.0F, 0.0F}, inverse));
	EXPECT_EQ(inverse, sentinel);
}

TEST(VlmQuaternion, InterpolationUsesShortestUnitPath) {
	const oa::vlm::Quat a = oa::vlm::Quat::identity();
	const oa::vlm::Quat b = oa::vlm::Quat::fromAxisAngle(
		{0.0F, 0.0F, 1.0F}, oa::vlm::radians(120.0F));
	const oa::vlm::Vec3 expected{0.5F, 0.8660254F, 0.0F};
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		oa::vlm::slerp(a, b, 0.5F).rotate({1.0F, 0.0F, 0.0F}),
		expected, 2.0e-5F, 2.0e-5F));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		oa::vlm::nlerp(a, -b, 0.5F).rotate({1.0F, 0.0F, 0.0F}),
		expected, 2.0e-5F, 2.0e-5F));
}

TEST(VlmQuaternion, MatrixRoundTripPreservesRotation) {
	const oa::vlm::Quat input = oa::vlm::Quat::fromAxisAngle(
		{1.0F, 2.0F, -3.0F}, 1.234F);
	const oa::vlm::Quat output = oa::vlm::quaternionFromMatrix(
		oa::vlm::quaternionToMatrix(input));
	EXPECT_NEAR(std::abs(oa::vlm::quaternionDot(input, output)), 1.0F, 1.0e-5F);
}

TEST(VlmQuaternion, MatrixAndQuaternionRotationAgreeAcrossFiniteInputs) {
	for (oa::I32 index = 1; index <= 256; ++index) {
		const oa::F32 value = static_cast<oa::F32>(index);
		const oa::vlm::Vec3 axis = oa::vlm::normalize(
			oa::vlm::Vec3{value * 0.31F, value * -0.17F + 1.0F, value * 0.07F - 2.0F});
		const oa::vlm::Quat rotation = oa::vlm::Quat::fromAxisAngle(
			axis, std::fmod(value * 0.113F, oa::vlm::Pi<oa::F32> * 2.0F));
		const oa::vlm::Vec3 input{value * 0.5F, 3.0F - value * 0.01F, -2.0F};
		EXPECT_TRUE(oa::vlm::approximatelyEqual(
			rotation.rotate(input),
			oa::vlm::transformDirection(input, oa::vlm::quaternionToMatrix(rotation)),
			2.0e-4F, 2.0e-5F));
	}
}

TEST(VlmMatrix, OperatorsAndRowVectorTransformUseNamedAuthority) {
	const oa::vlm::Mat4 a = oa::vlm::translation(
		oa::vlm::Vec3{1.0F, 2.0F, 3.0F});
	const oa::vlm::Mat4 b = oa::vlm::scaleMatrix(
		oa::vlm::Vec3{2.0F, 3.0F, 4.0F});
	EXPECT_EQ(a + b, oa::vlm::add(a, b));
	EXPECT_EQ(a * b, oa::vlm::matrixMul(a, b));
	EXPECT_EQ(-a, oa::vlm::scale(a, -1.0F));

	const oa::vlm::Vec4 value{3.0F, 4.0F, 5.0F, 1.0F};
	EXPECT_EQ(value * a, oa::vlm::transform(value, a));
	EXPECT_EQ(value * a, transformOracle(value, a));
}

TEST(VlmMatrix, ComposeTrsHasExplicitRowVectorOrder) {
	const oa::vlm::Quat rotation = oa::vlm::Quat::fromAxisAngle(
		{0.0F, 0.0F, 1.0F}, oa::vlm::Pi<oa::F32> * 0.5F);
	const oa::vlm::Mat4 trs = oa::vlm::composeTrs(
		{10.0F, 20.0F, 30.0F}, rotation, {2.0F, 3.0F, 4.0F});
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		oa::vlm::transformPoint({1.0F, 0.0F, 0.0F}, trs),
		oa::vlm::Vec3{10.0F, 22.0F, 30.0F}));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		oa::vlm::transformDirection({1.0F, 0.0F, 0.0F}, trs),
		oa::vlm::Vec3{0.0F, 2.0F, 0.0F}));
}

TEST(VlmMatrix, CheckedInverseAndDeterminantCoverAffineTransforms) {
	const oa::vlm::Mat4 matrix = oa::vlm::composeTrs(
		{7.0F, -11.0F, 13.0F},
		oa::vlm::Quat::fromAxisAngle({2.0F, 1.0F, -3.0F}, 0.71F),
		{2.0F, 3.0F, 4.0F});
	EXPECT_NEAR(oa::vlm::determinant(matrix), 24.0F, 2.0e-5F);

	oa::vlm::Mat4 inverse{};
	ASSERT_TRUE(oa::vlm::tryInverse(matrix, inverse));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		matrix * inverse, oa::vlm::Mat4::identity(), 2.0e-5F, 2.0e-5F));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		inverse * matrix, oa::vlm::Mat4::identity(), 2.0e-5F, 2.0e-5F));

	const oa::vlm::Mat4 singular = oa::vlm::scaleMatrix(
		oa::vlm::Vec3{1.0F, 0.0F, 1.0F});
	EXPECT_EQ(oa::vlm::determinant(singular), 0.0F);

	oa::vlm::Mat4 nonFinite = oa::vlm::Mat4::identity();
	nonFinite.m[0][0] = std::numeric_limits<oa::F32>::infinity();
	EXPECT_TRUE(std::isnan(oa::vlm::determinant(nonFinite)));
}

TEST(VlmMatrix, CheckedInverseSatisfiesDeterministicTrsProperties) {
	for (oa::I32 index = 1; index <= 256; ++index) {
		const oa::F32 value = static_cast<oa::F32>(index);
		const oa::vlm::Mat4 matrix = oa::vlm::composeTrs(
			{value * 11.0F, value * -7.0F, value * 0.125F},
			oa::vlm::Quat::fromAxisAngle(
				oa::vlm::normalize(oa::vlm::Vec3{value, value + 1.0F, value - 3.0F}),
				std::fmod(value * 0.071F, oa::vlm::Pi<oa::F32> * 2.0F)),
			{0.25F + value * 0.01F, 0.5F + value * 0.02F, 1.0F + value * 0.03F});
		oa::vlm::Mat4 inverse{};
		ASSERT_TRUE(oa::vlm::tryInverse(matrix, inverse));
		EXPECT_TRUE(oa::vlm::approximatelyEqual(
			matrix * inverse, oa::vlm::Mat4::identity(), 2.0e-3F, 2.0e-4F));
	}
}

TEST(VlmMatrix, CheckedInverseDoesNotModifyOutputOnFailure) {
	const oa::vlm::Mat4 singular = oa::vlm::scaleMatrix(
		oa::vlm::Vec3{1.0F, 0.0F, 1.0F});
	const oa::vlm::Mat4 sentinel = oa::vlm::translation(
		oa::vlm::Vec3{9.0F, 8.0F, 7.0F});
	oa::vlm::Mat4 output = sentinel;
	EXPECT_FALSE(oa::vlm::tryInverse(singular, output));
	EXPECT_EQ(output, sentinel);
}

TEST(VlmMatrix, NormalTransformUsesInverseTranspose) {
	const oa::vlm::Mat4 matrix = oa::vlm::scaleMatrix(
		oa::vlm::Vec3{2.0F, 4.0F, 8.0F});
	oa::vlm::Vec3 normal{};
	ASSERT_TRUE(oa::vlm::tryTransformNormal(
		oa::vlm::normalize(oa::vlm::Vec3{1.0F, 1.0F, 0.0F}), matrix, normal));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		normal, oa::vlm::normalize(oa::vlm::Vec3{0.5F, 0.25F, 0.0F})));
}

TEST(VlmProjection, PerspectiveUsesVulkanDepthRange) {
	const oa::F32 nearPlane = 0.1F;
	const oa::F32 farPlane = 100.0F;
	const oa::vlm::Mat4 projection = oa::vlm::perspective(
		60.0F, 16.0F / 9.0F, nearPlane, farPlane);
	oa::vlm::Vec3 nearNdc{};
	oa::vlm::Vec3 farNdc{};
	ASSERT_TRUE(oa::vlm::tryProjectPoint({0.0F, 0.0F, -nearPlane}, projection, nearNdc));
	ASSERT_TRUE(oa::vlm::tryProjectPoint({0.0F, 0.0F, -farPlane}, projection, farNdc));
	EXPECT_NEAR(nearNdc.z, 0.0F, 1.0e-5F);
	EXPECT_NEAR(farNdc.z, 1.0F, 1.0e-5F);
}

TEST(VlmProjection, OrthographicUsesVulkanDepthRange) {
	const oa::vlm::Mat4 projection = oa::vlm::orthographic(
		1920.0F, 1080.0F, -1.0F, 1.0F);
	EXPECT_NEAR(
		oa::vlm::transformPoint({0.0F, 0.0F, 1.0F}, projection).z,
		0.0F, 1.0e-6F);
	EXPECT_NEAR(
		oa::vlm::transformPoint({0.0F, 0.0F, -1.0F}, projection).z,
		1.0F, 1.0e-6F);
}

TEST(VlmDouble, TrsInverseRetainsDoublePrecision) {
	const oa::vlm::DMat4 matrix = oa::vlm::composeTrs(
		oa::vlm::DVec3{1.0e8, -3.0e7, 2.0e6},
		oa::vlm::DQuat::fromAxisAngle({1.0, 2.0, 3.0}, 0.123456789012345),
		oa::vlm::DVec3{0.125, 128.0, 3.5}
	);
	oa::vlm::DMat4 inverse{};
	ASSERT_TRUE(oa::vlm::tryInverse(matrix, inverse));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		matrix * inverse, oa::vlm::DMat4::identity(), 2.0e-8, 2.0e-12));
}
