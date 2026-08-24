#include "../../oaTest.h"

#include <core/joint.h>

namespace {
void expectVecNear(const oa::vlm::Vec3& A, const oa::vlm::Vec3& B, float eps = 1e-4f) {
	EXPECT_NEAR(A.x, B.x, eps);
	EXPECT_NEAR(A.y, B.y, eps);
	EXPECT_NEAR(A.z, B.z, eps);
}
} // namespace

// localMatrix (row-vector S·R·T) + oa::transformPoint agree with the direct
// quaternion rotate-scale-translate path. This pins the matrix/quat convention.
TEST(Transform, TrsMatchesQuaternionPath) {
	oa::Transform t;
	t.translate = { 1.0f, 2.0f, 3.0f };
	t.rotate    = oa::eulerXyzDegToQuat({ 15.0f, 30.0f, 45.0f });
	t.scale     = { 2.0f, 0.5f, 1.5f };

	const oa::vlm::Vec3 p = { 1.0f, -2.0f, 0.5f };
	const oa::vlm::Vec3 viaMatrix = oa::transformPoint(t.localMatrix(), p);
	const oa::vlm::Vec3 scaled = { p.x * t.scale.x, p.y * t.scale.y, p.z * t.scale.z };
	const oa::vlm::Vec3 viaQuat = oa::vlm::add(oa::vlm::rotateVector(t.rotate, scaled), t.translate);
	expectVecNear(viaMatrix, viaQuat);
}

// Maya XYZ Euler: +90° about Z maps +X → +Y.
TEST(Transform, EulerXyzZ90) {
	const oa::vlm::Quat q = oa::eulerXyzDegToQuat({ 0.0f, 0.0f, 90.0f });
	expectVecNear(oa::vlm::rotateVector(q, { 1.0f, 0.0f, 0.0f }), { 0.0f, 1.0f, 0.0f });
}

// Identity rotation/scale → only the translation applies.
TEST(Transform, IdentityIsTranslate) {
	oa::Transform t;
	t.translate = { 5.0f, -2.0f, 7.0f };
	expectVecNear(oa::transformPoint(t.localMatrix(), { 3.0f, 4.0f, 5.0f }), { 8.0f, 2.0f, 12.0f });
}

// Conjugate inverts a unit quaternion; multiplication composes.
TEST(Transform, QuatConjAndMul) {
	const oa::vlm::Quat q  = oa::eulerXyzDegToQuat({ 10.0f, 20.0f, 30.0f });
	const oa::vlm::Quat id = q * q.conjugate();
	EXPECT_NEAR(id.x, 0.0f, 1e-5f);
	EXPECT_NEAR(id.y, 0.0f, 1e-5f);
	EXPECT_NEAR(id.z, 0.0f, 1e-5f);
	EXPECT_NEAR(std::abs(id.w), 1.0f, 1e-5f);
}

TEST(Transform, SixDRotationRoundTripPreservesQuaternion) {
	const oa::vlm::Quat input = oa::eulerXyzDegToQuat(
		{31.0F, -47.0F, 83.0F}).normalized();
	oa::F32 rotation[6] = {};
	oa::quaternionToSixD(input, rotation);
	const oa::vlm::Quat output = oa::quaternionFromSixD(rotation).normalized();
	const oa::F32 alignment = input.x * output.x + input.y * output.y
		+ input.z * output.z + input.w * output.w;
	EXPECT_NEAR(std::abs(alignment), 1.0F, 1e-5F);
}

// oa::Joint folds jointOrient in: orientedRotation = jointOrient ⊗ rotate.
TEST(Transform, JointOrientCompose) {
	oa::Joint j;
	j.jointOrient = oa::eulerXyzDegToQuat({ 0.0f, 0.0f, 90.0f });
	j.rotate      = { 0.0f, 0.0f, 0.0f, 1.0f };
	// rotate identity ⇒ effective is just the orient: +X → +Y.
	expectVecNear(oa::vlm::rotateVector(j.orientedRotation(), { 1.0f, 0.0f, 0.0f }),
	              { 0.0f, 1.0f, 0.0f });
	// Another 90° animated on top ⇒ 180° total: +X → −X.
	j.rotate = oa::eulerXyzDegToQuat({ 0.0f, 0.0f, 90.0f });
	expectVecNear(oa::vlm::rotateVector(j.orientedRotation(), { 1.0f, 0.0f, 0.0f }),
	              { -1.0f, 0.0f, 0.0f });
}
