#include "../../oaTest.h"

#include <oa/core/fnTransform.h>

static_assert(not __is_constructible(oa::FnTransform));

namespace {
void expectVecNear(const oa::vlm::Vec3& A, const oa::vlm::Vec3& B, float eps = 1e-4f) {
	EXPECT_NEAR(A.x, B.x, eps);
	EXPECT_NEAR(A.y, B.y, eps);
	EXPECT_NEAR(A.z, B.z, eps);
}

void expectMatrixNear(
	const oa::vlm::Mat4& inActual,
	const oa::vlm::Mat4& inExpected,
	oa::F32 inTolerance = 1.0e-4F) {
	for (oa::U32 column = 0U; column < 4U; ++column) {
		for (oa::U32 row = 0U; row < 4U; ++row) {
			EXPECT_NEAR(
				inActual.m[column][row],
				inExpected.m[column][row],
				inTolerance);
		}
	}
}
} // namespace

TEST(Transform, MemberAndFunctionSetOperateOnOneLockedValue) {
	oa::Transform member;
	member.setPosition({2.0F, 3.0F, 4.0F});
	member.setRotationDegrees(
		{11.0F, 22.0F, 33.0F}, oa::vlm::RotationOrder::Zyx);
	member.setScale({0.5F, 2.0F, 3.0F});
	member.setShear({0.1F, -0.2F, 0.3F});

	oa::Transform functional;
	oa::FnTransform::setPosition(functional, {2.0F, 3.0F, 4.0F});
	oa::FnTransform::setRotationDegrees(
		functional, {11.0F, 22.0F, 33.0F}, oa::vlm::RotationOrder::Zyx);
	oa::FnTransform::setScale(functional, {0.5F, 2.0F, 3.0F});
	oa::FnTransform::setShear(functional, {0.1F, -0.2F, 0.3F});

	expectMatrixNear(member.getMatrix(), oa::FnTransform::getMatrix(functional));
}

TEST(Transform, AffineMatrixRoundTripPreservesShearAndReflection) {
	oa::vlm::AffineDecomposition input{};
	input.translation = {4.0F, -3.0F, 2.0F};
	input.rotation = oa::vlm::quaternionFromEulerDegrees(
		oa::vlm::Vec3{17.0F, -29.0F, 53.0F},
		oa::vlm::RotationOrder::Yzx);
	input.scale = {-2.0F, 3.0F, 0.5F};
	input.shear = {0.2F, -0.35F, 0.15F};
	const oa::vlm::Mat4 matrix = oa::vlm::composeAffine(input);

	oa::Transform transform;
	const oa::Status status = transform.setMatrix(matrix);
	ASSERT_TRUE(status.isOk()) << status.toString().cStr();
	expectMatrixNear(transform.getMatrix(), matrix, 3.0e-5F);
}

TEST(Transform, InvalidMatrixLeavesValueUnchanged) {
	oa::Transform transform{{1.0F, 2.0F, 3.0F}};
	const oa::vlm::Mat4 before = transform.getMatrix();
	const oa::vlm::Mat4 singular =
		oa::vlm::scaleMatrix(oa::vlm::Vec3{1.0F, 0.0F, 1.0F});

	EXPECT_EQ(
		transform.setMatrix(singular).getCode(),
		oa::StatusCode::InvalidArgument);
	expectMatrixNear(transform.getMatrix(), before);
}

// FnTransform matrix composition agrees with the direct
// quaternion rotate-scale-translate path. This pins the matrix/quat convention.
TEST(Transform, TrsMatchesQuaternionPath) {
	oa::Transform t;
	t.setPosition({1.0F, 2.0F, 3.0F});
	t.setRotation(oa::vlm::quaternionFromEulerDegrees(
		oa::vlm::Vec3{15.0F, 30.0F, 45.0F}, oa::vlm::RotationOrder::Xyz));
	t.setScale({2.0F, 0.5F, 1.5F});

	const oa::vlm::Vec3 p = { 1.0f, -2.0f, 0.5f };
	const oa::vlm::Vec3 viaMatrix =
		oa::vlm::transformPoint(p, oa::FnTransform::getMatrix(t));
	const oa::vlm::Vec3& scale = t.getScale();
	const oa::vlm::Vec3 scaled = {
		p.x * scale.x, p.y * scale.y, p.z * scale.z};
	const oa::vlm::Vec3 viaQuat = oa::vlm::add(
		oa::vlm::rotateVector(t.getRotation(), scaled), t.getPosition());
	expectVecNear(viaMatrix, viaQuat);
}

// Maya XYZ Euler: +90° about Z maps +X → +Y.
TEST(Transform, EulerXyzZ90) {
	const oa::vlm::Quat q = oa::vlm::quaternionFromEulerDegrees(
		oa::vlm::Vec3{0.0F, 0.0F, 90.0F}, oa::vlm::RotationOrder::Xyz);
	expectVecNear(oa::vlm::rotateVector(q, { 1.0f, 0.0f, 0.0f }), { 0.0f, 1.0f, 0.0f });
}

// Identity rotation/scale → only the translation applies.
TEST(Transform, IdentityIsTranslate) {
	oa::Transform t;
	t.setPosition({5.0F, -2.0F, 7.0F});
	expectVecNear(
		oa::vlm::transformPoint(
			{3.0F, 4.0F, 5.0F}, oa::FnTransform::getMatrix(t)),
		{8.0F, 2.0F, 12.0F});
}

// Conjugate inverts a unit quaternion; multiplication composes.
TEST(Transform, QuatConjAndMul) {
	const oa::vlm::Quat q = oa::vlm::quaternionFromEulerDegrees(
		oa::vlm::Vec3{10.0F, 20.0F, 30.0F}, oa::vlm::RotationOrder::Xyz);
	const oa::vlm::Quat id = q * q.conjugate();
	EXPECT_NEAR(id.x, 0.0f, 1e-5f);
	EXPECT_NEAR(id.y, 0.0f, 1e-5f);
	EXPECT_NEAR(id.z, 0.0f, 1e-5f);
	EXPECT_NEAR(std::abs(id.w), 1.0f, 1e-5f);
}

TEST(Transform, SixDRotationRoundTripPreservesQuaternion) {
	const oa::vlm::Quat input = oa::vlm::quaternionFromEulerDegrees(
		oa::vlm::Vec3{31.0F, -47.0F, 83.0F},
		oa::vlm::RotationOrder::Xyz).normalized();
	oa::F32 rotation[6] = {};
	oa::FnTransform::quaternionToSixD(input, rotation);
	const oa::vlm::Quat output =
		oa::FnTransform::quaternionFromSixD(rotation).normalized();
	const oa::F32 alignment = input.x * output.x + input.y * output.y
		+ input.z * output.z + input.w * output.w;
	EXPECT_NEAR(std::abs(alignment), 1.0F, 1e-5F);
}
