#pragma once

// OaTransform — a TRS transform node (Maya `MTransform` / `MFnTransform` analog).
//
// Row-vector convention throughout: a point maps as `p' = p · M`, with the
// translation living in row 3 of the matrix — matching the rest of the engine
// math (`Vlm::Translation` / `LookAt`). The 3×3 rotation block of a
// `LocalMatrix` is therefore the *transpose* of `Vlm::QuaternionToMatrix`
// (which returns the column-vector form used by the 6D rotation code). Use the
// `OaTransformPoint` / `OaTrsMatrix` helpers here so callers never have to track
// that distinction.

#include <Oa/Core/Vlm.h>

// Hamilton product `a ⊗ b` — the rotation that applies `b` first, then `a`
// (VlmQuat is stored (x, y, z, w)). Mirrors the local helper in PosePack so the
// whole 3D stack composes quaternions the same way.
[[nodiscard]] VlmQuat VlmQuatMul(const VlmQuat& A, const VlmQuat& B) noexcept;

// Normalize a quaternion; returns identity if degenerate.
[[nodiscard]] VlmQuat VlmQuatNormalize(const VlmQuat& Q) noexcept;

// Conjugate (= inverse for a unit quaternion).
[[nodiscard]] inline VlmQuat VlmQuatConjugate(const VlmQuat& Q) noexcept {
	return { -Q.X, -Q.Y, -Q.Z, Q.W };
}

// Build a quaternion from Maya-style XYZ Euler angles in degrees: the rotation
// applies X first, then Y, then Z (Maya's default `rotateOrder`). This is how
// rtg `rotateX/Y/Z` values bake into rest orientations.
[[nodiscard]] VlmQuat OaEulerXyzDegToQuat(const VlmVec3& InDegXyz) noexcept;

// Compose a TRS local matrix (row-vector S·R·T): a point is scaled, rotated by
// `Rot`, then translated by `Trans`. Consistent with `OaTransformPoint`.
[[nodiscard]] VlmMat4 OaTrsMatrix(const VlmVec3& InScale, const VlmQuat& InRot, const VlmVec3& InTrans) noexcept;

// Transform a point by a row-vector matrix: `p' = p · M` (applies the 3×3 then
// adds row 3). The correct extractor for forward-kinematics chains built with
// `OaTrsMatrix` + `Vlm::MatrixMul`.
[[nodiscard]] VlmVec3 OaTransformPoint(const VlmMat4& InM, const VlmVec3& InP) noexcept;

struct OaTransform {
	VlmVec3 Translate = { 0.0f, 0.0f, 0.0f };
	VlmQuat Rotate    = { 0.0f, 0.0f, 0.0f, 1.0f };
	VlmVec3 Scale     = { 1.0f, 1.0f, 1.0f };

	// The rotation this node imparts to its children. A plain transform just
	// returns `Rotate`; `OaJoint` folds in its `JointOrient`.
	[[nodiscard]] VlmQuat OrientedRotation() const noexcept { return Rotate; }

	// Local matrix mapping this node's space into its parent's space.
	[[nodiscard]] VlmMat4 LocalMatrix() const noexcept {
		return OaTrsMatrix(Scale, OrientedRotation(), Translate);
	}
};
