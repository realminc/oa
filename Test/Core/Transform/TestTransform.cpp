#include "../../OaTest.h"

#include <Core/Joint.h>

namespace {
void ExpectVecNear(const VlmVec3& A, const VlmVec3& B, float Eps = 1e-4f) {
	EXPECT_NEAR(A.X, B.X, Eps);
	EXPECT_NEAR(A.Y, B.Y, Eps);
	EXPECT_NEAR(A.Z, B.Z, Eps);
}
} // namespace

// LocalMatrix (row-vector S·R·T) + OaTransformPoint agree with the direct
// quaternion rotate-scale-translate path. This pins the matrix/quat convention.
TEST(Transform, TrsMatchesQuaternionPath) {
	OaTransform t;
	t.Translate = { 1.0f, 2.0f, 3.0f };
	t.Rotate    = OaEulerXyzDegToQuat({ 15.0f, 30.0f, 45.0f });
	t.Scale     = { 2.0f, 0.5f, 1.5f };

	const VlmVec3 p = { 1.0f, -2.0f, 0.5f };
	const VlmVec3 viaMatrix = OaTransformPoint(t.LocalMatrix(), p);
	const VlmVec3 scaled = { p.X * t.Scale.X, p.Y * t.Scale.Y, p.Z * t.Scale.Z };
	const VlmVec3 viaQuat = Vlm::Add(Vlm::RotateVector(t.Rotate, scaled), t.Translate);
	ExpectVecNear(viaMatrix, viaQuat);
}

// Maya XYZ Euler: +90° about Z maps +X → +Y.
TEST(Transform, EulerXyzZ90) {
	const VlmQuat q = OaEulerXyzDegToQuat({ 0.0f, 0.0f, 90.0f });
	ExpectVecNear(Vlm::RotateVector(q, { 1.0f, 0.0f, 0.0f }), { 0.0f, 1.0f, 0.0f });
}

// Identity rotation/scale → only the translation applies.
TEST(Transform, IdentityIsTranslate) {
	OaTransform t;
	t.Translate = { 5.0f, -2.0f, 7.0f };
	ExpectVecNear(OaTransformPoint(t.LocalMatrix(), { 3.0f, 4.0f, 5.0f }), { 8.0f, 2.0f, 12.0f });
}

// Conjugate inverts a unit quat; VlmQuatMul composes.
TEST(Transform, QuatConjAndMul) {
	const VlmQuat q  = OaEulerXyzDegToQuat({ 10.0f, 20.0f, 30.0f });
	const VlmQuat id = VlmQuatMul(q, VlmQuatConjugate(q));
	EXPECT_NEAR(id.X, 0.0f, 1e-5f);
	EXPECT_NEAR(id.Y, 0.0f, 1e-5f);
	EXPECT_NEAR(id.Z, 0.0f, 1e-5f);
	EXPECT_NEAR(std::abs(id.W), 1.0f, 1e-5f);
}

// OaJoint folds JointOrient in: OrientedRotation = JointOrient ⊗ Rotate.
TEST(Transform, JointOrientCompose) {
	OaJoint j;
	j.JointOrient = OaEulerXyzDegToQuat({ 0.0f, 0.0f, 90.0f });
	j.Rotate      = { 0.0f, 0.0f, 0.0f, 1.0f };
	// Rotate identity ⇒ effective is just the orient: +X → +Y.
	ExpectVecNear(Vlm::RotateVector(j.OrientedRotation(), { 1.0f, 0.0f, 0.0f }),
	              { 0.0f, 1.0f, 0.0f });
	// Another 90° animated on top ⇒ 180° total: +X → −X.
	j.Rotate = OaEulerXyzDegToQuat({ 0.0f, 0.0f, 90.0f });
	ExpectVecNear(Vlm::RotateVector(j.OrientedRotation(), { 1.0f, 0.0f, 0.0f }),
	              { -1.0f, 0.0f, 0.0f });
}
