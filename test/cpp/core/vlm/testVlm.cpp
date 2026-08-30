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
static_assert(std::is_standard_layout_v<oa::vlm::Mat3>);
static_assert(std::is_standard_layout_v<oa::vlm::Mat4>);
static_assert(std::is_trivially_copyable_v<oa::vlm::Vec2>);
static_assert(std::is_trivially_copyable_v<oa::vlm::Vec3>);
static_assert(std::is_trivially_copyable_v<oa::vlm::Vec4>);
static_assert(std::is_trivially_copyable_v<oa::vlm::Quat>);
static_assert(std::is_trivially_copyable_v<oa::vlm::Mat3>);
static_assert(std::is_trivially_copyable_v<oa::vlm::Mat4>);
static_assert(sizeof(oa::vlm::Vec2) == sizeof(oa::F32) * 2);
static_assert(sizeof(oa::vlm::Vec3) == sizeof(oa::F32) * 3);
static_assert(sizeof(oa::vlm::Vec4) == sizeof(oa::F32) * 4);
static_assert(sizeof(oa::vlm::Quat) == sizeof(oa::F32) * 4);
static_assert(sizeof(oa::vlm::Mat3) == sizeof(oa::F32) * 9);
static_assert(sizeof(oa::vlm::Mat4) == sizeof(oa::F32) * 16);
static_assert(sizeof(oa::vlm::DVec3) == sizeof(oa::F64) * 3);
static_assert(sizeof(oa::vlm::DQuat) == sizeof(oa::F64) * 4);
static_assert(sizeof(oa::vlm::DMat3) == sizeof(oa::F64) * 9);
static_assert(sizeof(oa::vlm::DMat4) == sizeof(oa::F64) * 16);

template <typename T>
constexpr T hostTolerance() {
	return std::is_same_v<T, oa::F32> ? T(5.0e-4F) : T(2.0e-10);
}

template <typename T>
constexpr T quaternionTolerance() {
	return std::is_same_v<T, oa::F32> ? T(5.0e-5F) : T(2.0e-11);
}

template <typename T>
bool equivalentRotation(
	const oa::vlm::detail::Quat<T>& inA,
	const oa::vlm::detail::Quat<T>& inB,
	T inTolerance = quaternionTolerance<T>()) {
	return std::abs(oa::vlm::quaternionDot(inA, inB)) >= T(1) - inTolerance;
}

template <typename T>
void verifyNormalizationAndAxisAngleContracts() {
	using Quat = oa::vlm::detail::Quat<T>;
	using Vec3 = oa::vlm::detail::Vec3<T>;
	const Vec3 vectorSentinel{T(7), T(8), T(9)};
	Vec3 vector = vectorSentinel;
	EXPECT_FALSE(oa::vlm::tryNormalize(Vec3{}, vector));
	EXPECT_EQ(vector, vectorSentinel);
	const T infinity = std::numeric_limits<T>::infinity();
	EXPECT_FALSE(oa::vlm::tryNormalize(Vec3{infinity, T(0), T(0)}, vector));
	EXPECT_EQ(vector, vectorSentinel);

	const T large = std::numeric_limits<T>::max() / T(16);
	vector = {large, -large / T(2), large / T(4)};
	ASSERT_TRUE(oa::vlm::tryNormalize(vector, vector));
	EXPECT_NEAR(oa::vlm::length(vector), T(1), hostTolerance<T>());
	const T small = std::numeric_limits<T>::min();
	vector = {small, -small, small};
	ASSERT_TRUE(oa::vlm::tryNormalize(vector, vector));
	EXPECT_NEAR(oa::vlm::length(vector), T(1), hostTolerance<T>());

	const Quat quaternionSentinel{T(1), T(2), T(3), T(4)};
	Quat quaternion = quaternionSentinel;
	EXPECT_FALSE(oa::vlm::tryNormalize(Quat{T(0), T(0), T(0), T(0)}, quaternion));
	EXPECT_EQ(quaternion, quaternionSentinel);
	quaternion = {large, -large / T(2), large / T(4), large / T(8)};
	ASSERT_TRUE(oa::vlm::tryNormalize(quaternion, quaternion));
	EXPECT_NEAR(quaternion.norm(), T(1), hostTolerance<T>());

	Quat axisAngle = quaternionSentinel;
	EXPECT_FALSE(oa::vlm::tryQuaternionFromAxisAngle(Vec3{}, T(1), axisAngle));
	EXPECT_EQ(axisAngle, quaternionSentinel);
	EXPECT_FALSE(oa::vlm::tryQuaternionFromAxisAngle(
		Vec3{T(1), T(0), T(0)}, infinity, axisAngle));
	EXPECT_EQ(axisAngle, quaternionSentinel);
	ASSERT_TRUE(oa::vlm::tryQuaternionFromAxisAngle(
		Vec3{large, large / T(2), -large / T(4)}, T(0.75), axisAngle));
	EXPECT_NEAR(axisAngle.norm(), T(1), hostTolerance<T>());

	const T extreme = std::numeric_limits<T>::max() / T(2);
	Vec3 projection = vectorSentinel;
	ASSERT_TRUE(oa::vlm::tryProjectVector(
		Vec3{extreme, extreme, T(0)},
		Vec3{extreme, T(0), T(0)}, projection));
	EXPECT_EQ(projection, (Vec3{extreme, T(0), T(0)}));
	T angle = T(-1);
	ASSERT_TRUE(oa::vlm::tryAngleBetween(
		Vec3{extreme, extreme, T(0)},
		Vec3{extreme, -extreme, T(0)}, angle));
	EXPECT_NEAR(angle, oa::vlm::Pi<T> * T(0.5), hostTolerance<T>());
	Quat fromTo = quaternionSentinel;
	ASSERT_TRUE(oa::vlm::tryQuaternionFromTo(
		Vec3{extreme, extreme, extreme},
		Vec3{extreme, extreme, extreme}, fromTo));
	EXPECT_EQ(fromTo, Quat::identity());
}

template <typename T>
void verifyInverseScaleAliasAndFailureContracts() {
	using Mat3 = oa::vlm::detail::Mat3<T>;
	using Mat4 = oa::vlm::detail::Mat4<T>;
	using Quat = oa::vlm::detail::Quat<T>;
	using Vec3 = oa::vlm::detail::Vec3<T>;
	const T smallScale = std::is_same_v<T, oa::F32> ? T(1.0e-12F) : T(1.0e-120);
	const T largeScale = T(1) / smallScale;
	const Quat rotation = oa::vlm::quaternionFromEulerRadians(
		Vec3{T(0.37), T(-0.51), T(0.83)}, oa::vlm::EulerOrder::Yzx);
	const Mat3 base3 = oa::vlm::linearPart(oa::vlm::composeTrs(
		Vec3{}, rotation, Vec3{T(2), T(3), T(4)}));
	for (const T scale : {smallScale, largeScale}) {
		Mat3 matrix = base3 * scale;
		ASSERT_TRUE(oa::vlm::tryInverse(matrix, matrix));
		EXPECT_TRUE(oa::vlm::approximatelyEqual(
			(base3 * scale) * matrix, Mat3::identity(),
			hostTolerance<T>(), hostTolerance<T>()));
	}

	const Mat4 base4 = oa::vlm::composeTrs(
		Vec3{T(11), T(-7), T(3)}, rotation, Vec3{T(2), T(3), T(4)});
	for (const T scale : {smallScale, largeScale}) {
		Mat4 matrix = base4 * scale;
		ASSERT_TRUE(oa::vlm::tryInverse(matrix, matrix));
		EXPECT_TRUE(oa::vlm::approximatelyEqual(
			(base4 * scale) * matrix, Mat4::identity(),
			hostTolerance<T>() * T(4), hostTolerance<T>() * T(4)));
	}

	Quat quaternion{largeScale, -largeScale / T(2), largeScale / T(4), largeScale / T(8)};
	const Quat original = quaternion;
	ASSERT_TRUE(oa::vlm::tryInverse(quaternion, quaternion));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		original * quaternion, Quat::identity(),
		hostTolerance<T>(), hostTolerance<T>()));

	Mat3 nearSingular = Mat3::identity();
	nearSingular.m[1][1] = oa::vlm::InverseTolerance<T> * T(0.25);
	const Mat3 sentinel3 = oa::vlm::scale(Mat3::identity(), T(7));
	Mat3 output3 = sentinel3;
	EXPECT_FALSE(oa::vlm::tryInverse(nearSingular, output3));
	EXPECT_EQ(output3, sentinel3);
	EXPECT_FALSE(oa::vlm::tryInverse(
		Mat3::identity(), output3, -oa::vlm::InverseTolerance<T>));
	EXPECT_EQ(output3, sentinel3);

	const Mat4 sentinel4 = oa::vlm::translation(Vec3{T(7), T(8), T(9)});
	Mat4 output4 = sentinel4;
	EXPECT_FALSE(oa::vlm::tryInverse(
		Mat4::identity(), output4, std::numeric_limits<T>::quiet_NaN()));
	EXPECT_EQ(output4, sentinel4);
}

template <typename T>
void verifyEulerGimbalBoundaries() {
	using Vec3 = oa::vlm::detail::Vec3<T>;
	const oa::vlm::EulerOrder orders[] = {
		oa::vlm::EulerOrder::Xyz, oa::vlm::EulerOrder::Xzy,
		oa::vlm::EulerOrder::Yxz, oa::vlm::EulerOrder::Yzx,
		oa::vlm::EulerOrder::Zxy, oa::vlm::EulerOrder::Zyx,
	};
	const oa::Usize lockedComponent[] = {1U, 2U, 0U, 2U, 0U, 1U};
	const T nearLockDelta = std::is_same_v<T, oa::F32> ? T(1.0e-3F) : T(1.0e-8);
	for (oa::Usize orderIndex = 0; orderIndex < 6U; ++orderIndex) {
		for (const T sign : {T(-1), T(1)}) {
			for (const T delta : {T(0), nearLockDelta}) {
				Vec3 angles{T(0.37), T(-0.42), T(0.61)};
				angles.at(lockedComponent[orderIndex]) =
					sign * (oa::vlm::Pi<T> * T(0.5) - delta);
				const auto rotation = oa::vlm::quaternionFromEulerRadians(
					angles, orders[orderIndex]);
				const Vec3 recovered = oa::vlm::quaternionToEulerRadians(
					rotation, orders[orderIndex]);
				const auto roundTrip = oa::vlm::quaternionFromEulerRadians(
					recovered, orders[orderIndex]);
				EXPECT_TRUE(equivalentRotation(rotation, roundTrip))
					<< "order=" << orderIndex << " sign=" << sign
					<< " delta=" << delta;
			}
		}
	}
}

template <typename T>
void verifyAffineProperties() {
	using Affine = oa::vlm::detail::AffineDecomposition<T>;
	using Mat4 = oa::vlm::detail::Mat4<T>;
	using Vec3 = oa::vlm::detail::Vec3<T>;
	for (oa::I32 index = 1; index <= 192; ++index) {
		const T value = static_cast<T>(index);
		Affine input{};
		input.translation = {
			value * T(0.31), value * T(-0.17), value * T(0.07)};
		input.rotation = oa::vlm::quaternionFromEulerRadians(
			Vec3{value * T(0.013), value * T(-0.017), value * T(0.019)},
			static_cast<oa::vlm::EulerOrder>((index - 1) % 6));
		input.scale = {
			(index % 2 == 0 ? T(-1) : T(1)) * (T(0.25) + value * T(0.01)),
			(index % 3 == 0 ? T(-1) : T(1)) * (T(0.5) + value * T(0.007)),
			T(0.75) + value * T(0.005)};
		input.shear = {
			T((index % 11) - 5) * T(0.017),
			T((index % 13) - 6) * T(0.013),
			T((index % 7) - 3) * T(0.019)};
		input.reflected = input.scale.x * input.scale.y * input.scale.z < T(0);
		const Mat4 matrix = oa::vlm::composeAffine(input);
		Affine output{};
		ASSERT_TRUE(oa::vlm::tryDecomposeAffine(matrix, output)) << index;
		EXPECT_EQ(output.reflected, input.reflected) << index;
		EXPECT_TRUE(oa::vlm::approximatelyEqual(
			matrix, oa::vlm::composeAffine(output),
			hostTolerance<T>() * T(4), hostTolerance<T>() * T(4))) << index;
	}

	Mat4 singular = Mat4::identity();
	singular.m[1][1] = T(0);
	Affine sentinel{};
	sentinel.translation = {T(7), T(8), T(9)};
	Affine output = sentinel;
	EXPECT_FALSE(oa::vlm::tryDecomposeAffine(singular, output));
	EXPECT_EQ(output.translation, sentinel.translation);
}

template <typename T>
void verifyProjectionAndViewportProperties() {
	using Mat4 = oa::vlm::detail::Mat4<T>;
	using Vec3 = oa::vlm::detail::Vec3<T>;
	using Viewport = oa::vlm::detail::Viewport<T>;
	constexpr T nearPlane = T(0.125);
	constexpr T farPlane = T(4096);
	Mat4 projection{};
	ASSERT_TRUE(oa::vlm::tryPerspective(
		T(67), T(16) / T(9), nearPlane, farPlane, projection));
	Vec3 nearNdc{};
	Vec3 farNdc{};
	ASSERT_TRUE(oa::vlm::tryProjectPoint(
		Vec3{T(0), T(0), -nearPlane}, projection, nearNdc));
	ASSERT_TRUE(oa::vlm::tryProjectPoint(
		Vec3{T(0), T(0), -farPlane}, projection, farNdc));
	EXPECT_NEAR(nearNdc.z, T(0), hostTolerance<T>());
	EXPECT_NEAR(farNdc.z, T(1), hostTolerance<T>());

	ASSERT_TRUE(oa::vlm::tryPerspectiveReverseZ(
		T(67), T(16) / T(9), nearPlane, farPlane, projection));
	ASSERT_TRUE(oa::vlm::tryProjectPoint(
		Vec3{T(0), T(0), -nearPlane}, projection, nearNdc));
	ASSERT_TRUE(oa::vlm::tryProjectPoint(
		Vec3{T(0), T(0), -farPlane}, projection, farNdc));
	EXPECT_NEAR(nearNdc.z, T(1), hostTolerance<T>());
	EXPECT_NEAR(farNdc.z, T(0), hostTolerance<T>());

	const Viewport viewports[] = {
		{T(0), T(0), T(1), T(1), T(0), T(1)},
		{T(13), T(17), T(1921), T(1081), T(0.125), T(0.875)},
	};
	const Mat4 forward = oa::vlm::perspective(
		T(67), T(16) / T(9), nearPlane, farPlane);
	for (const Viewport& viewport : viewports) {
		Vec3 point{T(0.25), T(-0.5), T(-3)};
		const Vec3 original = point;
		ASSERT_TRUE(oa::vlm::tryProjectToViewport(
			point, forward, viewport, point));
		ASSERT_TRUE(oa::vlm::tryUnprojectFromViewport(
			point, forward, viewport, point));
		EXPECT_TRUE(oa::vlm::approximatelyEqual(
			point, original, hostTolerance<T>() * T(8), hostTolerance<T>() * T(8)));
	}

	const Vec3 sentinel{T(7), T(8), T(9)};
	Vec3 output = sentinel;
	EXPECT_FALSE(oa::vlm::tryProjectPoint(
		Vec3{}, Mat4::identity(), output,
		std::numeric_limits<T>::quiet_NaN()));
	EXPECT_EQ(output, sentinel);
}

template <typename T>
void verifyCheckedFailurePreservation() {
	using Affine = oa::vlm::detail::AffineDecomposition<T>;
	using Mat3 = oa::vlm::detail::Mat3<T>;
	using Mat4 = oa::vlm::detail::Mat4<T>;
	using Quat = oa::vlm::detail::Quat<T>;
	using Trs = oa::vlm::detail::TrsDecomposition<T>;
	using Vec2 = oa::vlm::detail::Vec2<T>;
	using Vec3 = oa::vlm::detail::Vec3<T>;
	using Vec4 = oa::vlm::detail::Vec4<T>;
	using Viewport = oa::vlm::detail::Viewport<T>;
	const T infinity = std::numeric_limits<T>::infinity();
	const T nan = std::numeric_limits<T>::quiet_NaN();

	const Vec2 sentinel2{T(7), T(8)};
	Vec2 output2 = sentinel2;
	EXPECT_FALSE(oa::vlm::tryNormalize(Vec2{}, output2));
	EXPECT_EQ(output2, sentinel2);
	const Vec4 sentinel4{T(7), T(8), T(9), T(10)};
	Vec4 output4 = sentinel4;
	EXPECT_FALSE(oa::vlm::tryNormalize(Vec4{}, output4));
	EXPECT_EQ(output4, sentinel4);

	const Vec3 sentinel3{T(7), T(8), T(9)};
	Vec3 output3 = sentinel3;
	EXPECT_FALSE(oa::vlm::tryProjectVector(
		Vec3{T(1), T(2), T(3)}, Vec3{}, output3));
	EXPECT_EQ(output3, sentinel3);
	EXPECT_FALSE(oa::vlm::tryRejectVector(
		Vec3{T(1), T(2), T(3)}, Vec3{}, output3));
	EXPECT_EQ(output3, sentinel3);
	EXPECT_FALSE(oa::vlm::tryPerpendicular(Vec3{}, output3));
	EXPECT_EQ(output3, sentinel3);
	T scalarOutput = T(7);
	EXPECT_FALSE(oa::vlm::tryAngleBetween(Vec3{}, sentinel3, scalarOutput));
	EXPECT_EQ(scalarOutput, T(7));
	EXPECT_FALSE(oa::vlm::trySignedAngleBetween(
		sentinel3, sentinel3, Vec3{}, scalarOutput));
	EXPECT_EQ(scalarOutput, T(7));

	const Quat quaternionSentinel{T(1), T(2), T(3), T(4)};
	Quat quaternionOutput = quaternionSentinel;
	EXPECT_FALSE(oa::vlm::tryNormalize(
		Quat{T(0), T(0), T(0), T(0)}, quaternionOutput));
	EXPECT_EQ(quaternionOutput, quaternionSentinel);
	EXPECT_FALSE(oa::vlm::tryQuaternionFromAxisAngle(
		Vec3{}, T(1), quaternionOutput));
	EXPECT_EQ(quaternionOutput, quaternionSentinel);
	EXPECT_FALSE(oa::vlm::tryInverse(
		Quat{T(0), T(0), T(0), T(0)}, quaternionOutput));
	EXPECT_EQ(quaternionOutput, quaternionSentinel);
	EXPECT_FALSE(oa::vlm::tryQuaternionFromTo(
		Vec3{}, sentinel3, quaternionOutput));
	EXPECT_EQ(quaternionOutput, quaternionSentinel);
	EXPECT_FALSE(oa::vlm::tryLookRotation(
		Vec3{T(0), T(1), T(0)}, Vec3{T(0), T(1), T(0)}, quaternionOutput));
	EXPECT_EQ(quaternionOutput, quaternionSentinel);

	const Mat3 matrixSentinel3 = oa::vlm::scale(Mat3::identity(), T(7));
	Mat3 matrixOutput3 = matrixSentinel3;
	EXPECT_FALSE(oa::vlm::tryInverse(Mat3{}, matrixOutput3));
	EXPECT_EQ(matrixOutput3, matrixSentinel3);
	EXPECT_FALSE(oa::vlm::tryOrthonormalBasis(
		Vec3{}, sentinel3, matrixOutput3));
	EXPECT_EQ(matrixOutput3, matrixSentinel3);
	EXPECT_FALSE(oa::vlm::tryQuaternionFromRotationMatrix(
		Mat3{}, quaternionOutput));
	EXPECT_EQ(quaternionOutput, quaternionSentinel);
	EXPECT_FALSE(oa::vlm::tryTransformNormal(
		Vec3{}, Mat3::identity(), output3));
	EXPECT_EQ(output3, sentinel3);

	const Mat4 matrixSentinel4 = oa::vlm::translation(sentinel3);
	Mat4 matrixOutput4 = matrixSentinel4;
	EXPECT_FALSE(oa::vlm::tryInverse(Mat4{}, matrixOutput4));
	EXPECT_EQ(matrixOutput4, matrixSentinel4);
	Mat4 nonAffine = Mat4::identity();
	nonAffine.m[0][3] = T(1);
	EXPECT_FALSE(oa::vlm::tryAffineInverse(nonAffine, matrixOutput4));
	EXPECT_EQ(matrixOutput4, matrixSentinel4);
	EXPECT_FALSE(oa::vlm::tryNormalMatrix(nonAffine, matrixOutput3));
	EXPECT_EQ(matrixOutput3, matrixSentinel3);
	EXPECT_FALSE(oa::vlm::tryQuaternionFromRotationMatrix(
		nonAffine, quaternionOutput));
	EXPECT_EQ(quaternionOutput, quaternionSentinel);
	EXPECT_FALSE(oa::vlm::tryTransformNormal(
		Vec3{}, Mat4::identity(), output3));
	EXPECT_EQ(output3, sentinel3);
	EXPECT_FALSE(oa::vlm::tryViewFromPose(
		Vec3{}, Quat{T(0), T(0), T(0), T(0)}, matrixOutput4));
	EXPECT_EQ(matrixOutput4, matrixSentinel4);
	EXPECT_FALSE(oa::vlm::tryLookAt(
		Vec3{}, Vec3{}, Vec3{T(0), T(1), T(0)}, matrixOutput4));
	EXPECT_EQ(matrixOutput4, matrixSentinel4);
	const T extreme = std::numeric_limits<T>::max();
	EXPECT_FALSE(oa::vlm::tryLookAt(
		Vec3{extreme, extreme, extreme}, Vec3{},
		Vec3{T(0), T(1), T(0)}, matrixOutput4));
	EXPECT_EQ(matrixOutput4, matrixSentinel4);

	Affine affineSentinel{};
	affineSentinel.translation = sentinel3;
	Affine affineOutput = affineSentinel;
	EXPECT_FALSE(oa::vlm::tryDecomposeAffine(nonAffine, affineOutput));
	EXPECT_EQ(affineOutput.translation, affineSentinel.translation);
	Trs trsSentinel{};
	trsSentinel.translation = sentinel3;
	Trs trsOutput = trsSentinel;
	EXPECT_FALSE(oa::vlm::tryDecomposeTrs(nonAffine, trsOutput));
	EXPECT_EQ(trsOutput.translation, trsSentinel.translation);

	EXPECT_FALSE(oa::vlm::tryPerspectiveOffCenter(
		T(0), T(0), T(-1), T(1), T(0.1), T(10), matrixOutput4));
	EXPECT_EQ(matrixOutput4, matrixSentinel4);
	EXPECT_FALSE(oa::vlm::tryPerspective(
		T(60), T(0), T(0.1), T(10), matrixOutput4));
	EXPECT_EQ(matrixOutput4, matrixSentinel4);
	EXPECT_FALSE(oa::vlm::tryPerspectiveShifted(
		T(60), T(1), T(0.1), T(10), Vec2{infinity, T(0)}, matrixOutput4));
	EXPECT_EQ(matrixOutput4, matrixSentinel4);
	EXPECT_FALSE(oa::vlm::tryOrthographicOffCenter(
		T(0), T(0), T(-1), T(1), T(-1), T(1), matrixOutput4));
	EXPECT_EQ(matrixOutput4, matrixSentinel4);
	EXPECT_FALSE(oa::vlm::tryOrthographic(
		T(0), T(1), T(-1), T(1), T(1), matrixOutput4));
	EXPECT_EQ(matrixOutput4, matrixSentinel4);
	EXPECT_FALSE(oa::vlm::tryOrthographicShifted(
		T(1), T(1), T(-1), T(1), T(1), Vec2{nan, T(0)}, matrixOutput4));
	EXPECT_EQ(matrixOutput4, matrixSentinel4);
	EXPECT_FALSE(oa::vlm::tryPerspectiveReverseZ(
		T(60), T(1), T(1), T(1), matrixOutput4));
	EXPECT_EQ(matrixOutput4, matrixSentinel4);
	EXPECT_FALSE(oa::vlm::tryPerspectiveReverseZInfinite(
		T(60), T(1), T(0), matrixOutput4));
	EXPECT_EQ(matrixOutput4, matrixSentinel4);

	EXPECT_FALSE(oa::vlm::tryProjectPoint(
		Vec3{}, Mat4{}, output3));
	EXPECT_EQ(output3, sentinel3);
	EXPECT_FALSE(oa::vlm::tryProjectToViewport(
		Vec3{}, Mat4::identity(), Viewport{T(0), T(0), T(0), T(1)}, output3));
	EXPECT_EQ(output3, sentinel3);
	EXPECT_FALSE(oa::vlm::tryUnprojectFromViewport(
		Vec3{}, Mat4::identity(), Viewport{T(0), T(0), T(1), T(0)}, output3));
	EXPECT_EQ(output3, sentinel3);

	Vec3 alias{T(1), T(2), T(3)};
	ASSERT_TRUE(oa::vlm::tryProjectVector(
		alias, Vec3{T(1), T(0), T(0)}, alias));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		alias, Vec3{T(1), T(0), T(0)}, hostTolerance<T>(), hostTolerance<T>()));
	alias = {T(1), T(2), T(3)};
	ASSERT_TRUE(oa::vlm::tryRejectVector(
		alias, Vec3{T(1), T(0), T(0)}, alias));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		alias, Vec3{T(0), T(2), T(3)}, hostTolerance<T>(), hostTolerance<T>()));
	alias = {T(2), T(3), T(4)};
	ASSERT_TRUE(oa::vlm::tryPerpendicular(alias, alias));
	EXPECT_NEAR(oa::vlm::dot(alias, Vec3{T(2), T(3), T(4)}), T(0), hostTolerance<T>());
	alias = {T(1), T(1), T(0)};
	ASSERT_TRUE(oa::vlm::tryTransformNormal(
		alias, Mat3::identity(), alias));
	EXPECT_NEAR(oa::vlm::length(alias), T(1), hostTolerance<T>());
}

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

TEST(VlmVector, DirectionHelpersRejectDegenerateInputs) {
	oa::vlm::Vec3 projection{};
	ASSERT_TRUE(oa::vlm::tryProjectVector(
		{2.0F, 3.0F, 4.0F}, {1.0F, 0.0F, 0.0F}, projection));
	EXPECT_EQ(projection, (oa::vlm::Vec3{2.0F, 0.0F, 0.0F}));
	oa::vlm::Vec3 perpendicular{};
	ASSERT_TRUE(oa::vlm::tryPerpendicular(
		{2.0F, -3.0F, 4.0F}, perpendicular));
	EXPECT_NEAR(
		oa::vlm::dot(perpendicular, {2.0F, -3.0F, 4.0F}),
		0.0F, 1.0e-6F);
	oa::F32 signedAngle = 0.0F;
	ASSERT_TRUE(oa::vlm::trySignedAngleBetween(
		{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F},
		{0.0F, 0.0F, 1.0F}, signedAngle));
	EXPECT_NEAR(signedAngle, oa::vlm::Pi<oa::F32> * 0.5F, 1.0e-6F);
	const oa::vlm::Vec3 sentinel{7.0F, 8.0F, 9.0F};
	perpendicular = sentinel;
	EXPECT_FALSE(oa::vlm::tryPerpendicular({}, perpendicular));
	EXPECT_EQ(perpendicular, sentinel);
}

TEST(VlmAbi, CheckedComponentAccessFailsClosed) {
	oa::vlm::Vec3 vector{1.0F, 2.0F, 3.0F};
	vector.at(1U) = 7.0F;
	EXPECT_EQ(vector.y, 7.0F);
	oa::vlm::Mat3 matrix = oa::vlm::Mat3::identity();
	matrix.at(1U, 2U) = 5.0F;
	EXPECT_EQ(matrix.m[1][2], 5.0F);
	EXPECT_DEATH((void)vector.at(3U), "");
	EXPECT_DEATH((void)matrix.at(3U, 0U), "");
}

TEST(VlmVector, DegenerateNormalizationFailsClosed) {
	const oa::vlm::Vec3 sentinel{7.0F, 8.0F, 9.0F};
	oa::vlm::Vec3 output = sentinel;
	EXPECT_FALSE(oa::vlm::tryNormalize(oa::vlm::Vec3{}, output));
	EXPECT_EQ(output, sentinel);
	const oa::F32 infinity = std::numeric_limits<oa::F32>::infinity();
	EXPECT_FALSE(oa::vlm::tryNormalize(
		oa::vlm::Vec3{infinity, 0.0F, 0.0F}, output));
	EXPECT_EQ(output, sentinel);
	EXPECT_DEATH((void)oa::vlm::normalize(oa::vlm::Vec3{}), "");
	EXPECT_FALSE(oa::vlm::approximatelyEqual(infinity, infinity));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(1000.0F, 1000.005F));
	EXPECT_FALSE(oa::vlm::approximatelyEqual(1.0F, 1.1F));
}

TEST(VlmContract, NormalizationAndAxisAngleCoverFloatAndDoubleExtremes) {
	verifyNormalizationAndAxisAngleContracts<oa::F32>();
	verifyNormalizationAndAxisAngleContracts<oa::F64>();
	EXPECT_DEATH((void)oa::vlm::Vec3{}.normalized(), "");
	EXPECT_DEATH((void)oa::vlm::Quat{}.fromAxisAngle({}, 1.0F), "");
}

TEST(VlmContract, InverseScaleAliasAndFailureCoverFloatAndDouble) {
	verifyInverseScaleAliasAndFailureContracts<oa::F32>();
	verifyInverseScaleAliasAndFailureContracts<oa::F64>();
}

TEST(VlmContract, EveryEulerOrderCoversBothGimbalBoundaries) {
	verifyEulerGimbalBoundaries<oa::F32>();
	verifyEulerGimbalBoundaries<oa::F64>();
	const auto invalid = static_cast<oa::vlm::EulerOrder>(255U);
	EXPECT_DEATH(
		(void)oa::vlm::quaternionFromEulerRadians(oa::vlm::Vec3{}, invalid),
		"");
	EXPECT_DEATH(
		(void)oa::vlm::quaternionToEulerRadians(oa::vlm::Quat{}, invalid),
		"");
}

TEST(VlmContract, AffinePropertiesCoverFloatAndDouble) {
	verifyAffineProperties<oa::F32>();
	verifyAffineProperties<oa::F64>();
}

TEST(VlmContract, ProjectionViewportPropertiesCoverFloatAndDouble) {
	verifyProjectionAndViewportProperties<oa::F32>();
	verifyProjectionAndViewportProperties<oa::F64>();
}

TEST(VlmContract, CheckedFailuresPreserveOutputsAndAliases) {
	verifyCheckedFailurePreservation<oa::F32>();
	verifyCheckedFailurePreservation<oa::F64>();
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
	oa::vlm::Quat output{};
	ASSERT_TRUE(oa::vlm::tryQuaternionFromRotationMatrix(
		oa::vlm::quaternionToMatrix(input), output));
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

TEST(VlmQuaternion, NamedEulerOrdersRoundTripRotation) {
	const oa::vlm::EulerOrder orders[] = {
		oa::vlm::EulerOrder::Xyz,
		oa::vlm::EulerOrder::Xzy,
		oa::vlm::EulerOrder::Yxz,
		oa::vlm::EulerOrder::Yzx,
		oa::vlm::EulerOrder::Zxy,
		oa::vlm::EulerOrder::Zyx,
	};
	for (const oa::vlm::EulerOrder order : orders) {
		const oa::vlm::Quat rotation = oa::vlm::quaternionFromEulerDegrees(
			oa::vlm::Vec3{23.0F, -31.0F, 47.0F}, order);
		const oa::vlm::Vec3 recovered = oa::vlm::quaternionToEulerDegrees(
			rotation, order);
		const oa::vlm::Quat roundTrip = oa::vlm::quaternionFromEulerDegrees(
			recovered, order);
		EXPECT_NEAR(
			std::abs(oa::vlm::quaternionDot(rotation, roundTrip)),
			1.0F, 2.0e-5F);
	}
}

TEST(VlmQuaternion, FromToAndLookRotationHandleCanonicalAxes) {
	oa::vlm::Quat fromTo{};
	ASSERT_TRUE(oa::vlm::tryQuaternionFromTo(
		{1.0F, 0.0F, 0.0F}, {-1.0F, 0.0F, 0.0F}, fromTo));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		fromTo.rotate({1.0F, 0.0F, 0.0F}),
		oa::vlm::Vec3{-1.0F, 0.0F, 0.0F}, 2.0e-5F, 2.0e-5F));
	EXPECT_NEAR(oa::vlm::quaternionAngle(fromTo), oa::vlm::Pi<oa::F32>, 2.0e-5F);
	EXPECT_NEAR(oa::vlm::length(oa::vlm::quaternionAxis(fromTo)), 1.0F, 1.0e-5F);

	oa::vlm::Quat look{};
	ASSERT_TRUE(oa::vlm::tryLookRotation(
		{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, look));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		look.rotate({0.0F, 0.0F, -1.0F}),
		oa::vlm::Vec3{1.0F, 0.0F, 0.0F}, 2.0e-5F, 2.0e-5F));
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
	oa::vlm::Mat4 triangular{};
	triangular.m[0][0] = 2.0F;
	triangular.m[0][1] = 1.0F;
	triangular.m[0][2] = 3.0F;
	triangular.m[0][3] = 4.0F;
	triangular.m[1][1] = 3.0F;
	triangular.m[1][2] = 2.0F;
	triangular.m[1][3] = 1.0F;
	triangular.m[2][2] = 4.0F;
	triangular.m[2][3] = 2.0F;
	triangular.m[3][3] = 5.0F;
	EXPECT_EQ(oa::vlm::determinant(triangular), 120.0F);

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

TEST(VlmMatrix, Mat3InverseAndNormalMatrixUseRowVectorConvention) {
	const oa::vlm::Mat4 transform = oa::vlm::composeTrs(
		{3.0F, 4.0F, 5.0F},
		oa::vlm::Quat::fromAxisAngle({0.0F, 0.0F, 1.0F}, 0.4F),
		{2.0F, 3.0F, 4.0F});
	oa::vlm::Mat3 normalMatrix{};
	ASSERT_TRUE(oa::vlm::tryNormalMatrix(transform, normalMatrix));
	oa::vlm::Mat3 inverse{};
	ASSERT_TRUE(oa::vlm::tryInverse(oa::vlm::linearPart(transform), inverse));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		normalMatrix, oa::vlm::transpose(inverse)));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		oa::vlm::linearPart(transform) * inverse,
		oa::vlm::Mat3::identity(), 2.0e-5F, 2.0e-5F));
}

TEST(VlmMatrix, ExactTrsDecompositionPreservesReflectionAndRejectsShear) {
	const oa::vlm::Mat4 matrix = oa::vlm::composeTrs(
		{7.0F, -2.0F, 11.0F},
		oa::vlm::quaternionFromEulerDegrees(
			oa::vlm::Vec3{20.0F, -30.0F, 40.0F},
			oa::vlm::EulerOrder::Yxz),
		{-2.0F, 3.0F, 4.0F});
	oa::vlm::TrsDecomposition decomposition{};
	ASSERT_TRUE(oa::vlm::tryDecomposeTrs(matrix, decomposition));
	EXPECT_TRUE(decomposition.reflected);
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		matrix,
		oa::vlm::composeTrs(
			decomposition.translation,
			decomposition.rotation,
			decomposition.scale),
		2.0e-5F, 2.0e-5F));

	oa::vlm::Mat4 shear = oa::vlm::Mat4::identity();
	shear.m[1][0] = 0.25F;
	const oa::vlm::TrsDecomposition sentinel = decomposition;
	EXPECT_FALSE(oa::vlm::tryDecomposeTrs(shear, decomposition));
	EXPECT_EQ(decomposition.translation, sentinel.translation);
	EXPECT_EQ(decomposition.rotation, sentinel.rotation);
	EXPECT_EQ(decomposition.scale, sentinel.scale);
}

TEST(VlmMatrix, AffineDecompositionPreservesShearAndReflection) {
	oa::vlm::AffineDecomposition input{};
	input.translation = {5.0F, -7.0F, 11.0F};
	input.rotation = oa::vlm::quaternionFromEulerDegrees(
		oa::vlm::Vec3{19.0F, -27.0F, 41.0F},
		oa::vlm::EulerOrder::Zxy);
	input.scale = {-2.0F, 3.0F, 4.0F};
	input.shear = {0.2F, -0.15F, 0.35F};
	input.reflected = true;
	const oa::vlm::Mat4 matrix = oa::vlm::composeAffine(input);
	oa::vlm::AffineDecomposition output{};
	ASSERT_TRUE(oa::vlm::tryDecomposeAffine(matrix, output));
	EXPECT_TRUE(output.reflected);
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		matrix, oa::vlm::composeAffine(output), 3.0e-5F, 3.0e-5F));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		input.shear, output.shear, 2.0e-5F, 2.0e-5F));
	oa::vlm::TrsDecomposition trs{};
	EXPECT_FALSE(oa::vlm::tryDecomposeTrs(matrix, trs));
}

TEST(VlmMatrix, MatrixQueriesSeparateReflectionFromRotation) {
	const oa::vlm::Mat3 reflection = oa::vlm::linearPart(
		oa::vlm::scaleMatrix(oa::vlm::Vec3{-1.0F, 1.0F, 1.0F}));
	EXPECT_TRUE(reflection.isFinite());
	EXPECT_TRUE(oa::vlm::isOrthonormal(reflection));
	EXPECT_FALSE(oa::vlm::isProperRotation(reflection));
	EXPECT_TRUE(oa::vlm::isIdentity(oa::vlm::Mat4::identity()));
	oa::vlm::Quat output{1.0F, 2.0F, 3.0F, 4.0F};
	const oa::vlm::Quat sentinel = output;
	EXPECT_FALSE(oa::vlm::tryQuaternionFromRotationMatrix(
		reflection, output));
	EXPECT_EQ(output, sentinel);
}

TEST(VlmMatrix, ViewFromPoseIsInverseOfCameraWorldTransform) {
	const oa::vlm::Vec3 position{3.0F, 5.0F, -7.0F};
	const oa::vlm::Quat rotation = oa::vlm::quaternionFromEulerDegrees(
		oa::vlm::Vec3{15.0F, 25.0F, -35.0F},
		oa::vlm::EulerOrder::Xyz);
	const oa::vlm::Mat4 world = oa::vlm::composeTrs(
		position, rotation, {1.0F, 1.0F, 1.0F});
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		world * oa::vlm::viewFromPose(position, rotation),
		oa::vlm::Mat4::identity(), 2.0e-5F, 2.0e-5F));
	oa::vlm::Mat4 sentinel = world;
	EXPECT_FALSE(oa::vlm::tryViewFromPose(
		position, {0.0F, 0.0F, 0.0F, 0.0F}, sentinel));
	EXPECT_EQ(sentinel, world);
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

TEST(VlmProjection, CheckedBuildersPreserveOutputOnInvalidInput) {
	const oa::vlm::Mat4 sentinel = oa::vlm::translation(
		oa::vlm::Vec3{1.0F, 2.0F, 3.0F});
	oa::vlm::Mat4 output = sentinel;
	EXPECT_FALSE(oa::vlm::tryPerspective(60.0F, 0.0F, 0.1F, 100.0F, output));
	EXPECT_EQ(output, sentinel);
	EXPECT_FALSE(oa::vlm::tryOrthographic(
		1920.0F, 0.0F, -1.0F, 1.0F, 1.0F, output));
	EXPECT_EQ(output, sentinel);
	EXPECT_FALSE(oa::vlm::tryLookAt(
		{}, {}, {0.0F, 1.0F, 0.0F}, output));
	EXPECT_EQ(output, sentinel);
	const oa::F32 maximum = std::numeric_limits<oa::F32>::max();
	EXPECT_FALSE(oa::vlm::tryPerspectiveReverseZ(
		60.0F, 1.0F, maximum * 0.5F, maximum, output));
	EXPECT_EQ(output, sentinel);
	EXPECT_FALSE(oa::vlm::tryPerspectiveOffCenter(
		-maximum, maximum, -1.0F, 1.0F, 0.1F, 100.0F, output));
	EXPECT_EQ(output, sentinel);
}

TEST(VlmProjection, OffCenterFrustumMapsNearEdgesExactly) {
	oa::vlm::Mat4 projection{};
	ASSERT_TRUE(oa::vlm::tryPerspectiveOffCenter(
		-0.2F, 0.4F, -0.1F, 0.3F, 0.5F, 20.0F, projection));
	oa::vlm::Vec3 lowerLeft{};
	oa::vlm::Vec3 upperRight{};
	ASSERT_TRUE(oa::vlm::tryProjectPoint(
		{-0.2F, -0.1F, -0.5F}, projection, lowerLeft));
	ASSERT_TRUE(oa::vlm::tryProjectPoint(
		{0.4F, 0.3F, -0.5F}, projection, upperRight));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		lowerLeft, oa::vlm::Vec3{-1.0F, -1.0F, 0.0F}));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		upperRight, oa::vlm::Vec3{1.0F, 1.0F, 0.0F}));
}

TEST(VlmProjection, ReverseZFiniteAndInfiniteMapNearToOne) {
	constexpr oa::F32 nearPlane = 0.25F;
	constexpr oa::F32 farPlane = 1000.0F;
	oa::vlm::Mat4 finite{};
	ASSERT_TRUE(oa::vlm::tryPerspectiveReverseZ(
		70.0F, 16.0F / 9.0F, nearPlane, farPlane, finite));
	oa::vlm::Vec3 nearNdc{};
	oa::vlm::Vec3 farNdc{};
	ASSERT_TRUE(oa::vlm::tryProjectPoint(
		{0.0F, 0.0F, -nearPlane}, finite, nearNdc));
	ASSERT_TRUE(oa::vlm::tryProjectPoint(
		{0.0F, 0.0F, -farPlane}, finite, farNdc));
	EXPECT_NEAR(nearNdc.z, 1.0F, 1.0e-6F);
	EXPECT_NEAR(farNdc.z, 0.0F, 1.0e-6F);

	oa::vlm::Mat4 infinite{};
	ASSERT_TRUE(oa::vlm::tryPerspectiveReverseZInfinite(
		70.0F, 16.0F / 9.0F, nearPlane, infinite));
	ASSERT_TRUE(oa::vlm::tryProjectPoint(
		{0.0F, 0.0F, -nearPlane}, infinite, nearNdc));
	ASSERT_TRUE(oa::vlm::tryProjectPoint(
		{0.0F, 0.0F, -1.0e7F}, infinite, farNdc));
	EXPECT_NEAR(nearNdc.z, 1.0F, 1.0e-6F);
	EXPECT_NEAR(farNdc.z, 0.0F, 1.0e-6F);
}

TEST(VlmProjection, TopOriginViewportRoundTripIsStable) {
	const oa::vlm::Mat4 viewProjection = oa::vlm::viewFromPose(
		{2.0F, 3.0F, 4.0F},
		oa::vlm::quaternionFromEulerDegrees(
			oa::vlm::Vec3{10.0F, -20.0F, 5.0F},
			oa::vlm::EulerOrder::Xyz))
		* oa::vlm::perspective(65.0F, 1921.0F / 1081.0F, 0.1F, 500.0F);
	const oa::vlm::Viewport viewport{
		13.0F, 17.0F, 1921.0F, 1081.0F, 0.0F, 1.0F};
	const oa::vlm::Vec3 point{1.0F, 2.0F, -5.0F};
	oa::vlm::Vec3 projected{};
	oa::vlm::Vec3 unprojected{};
	ASSERT_TRUE(oa::vlm::tryProjectToViewport(
		point, viewProjection, viewport, projected));
	ASSERT_TRUE(oa::vlm::tryUnprojectFromViewport(
		projected, viewProjection, viewport, unprojected));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		point, unprojected, 2.0e-4F, 2.0e-5F));
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
