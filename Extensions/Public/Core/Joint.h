#pragma once

// OaJoint — a skeletal joint transform (Maya `MFnIkJoint` analog).
//
// Extends OaTransform with a fixed `JointOrient`: the joint's rest orientation,
// against which the animated `Rotate` is a delta. In Maya terms the local matrix
// is `S · Rotate · JointOrient · Translate` (rotate applied first, then the
// orient) — and `JointOrient` is exactly the rotational part of the Maya-2020+
// `offsetParentMatrix`, surfaced here via `OffsetParentMatrix()`.
//
// Forward kinematics over a chain of OaJoints is cleanest in quaternion form
// (see OaSkeleton::RestWorld), but `LocalMatrix()` is provided for the
// matrix-composition path and for callers that want the UE/Maya offset-parent
// representation.

#include <Core/Transform.h>

struct OaJoint : OaTransform {
	VlmQuat JointOrient = { 0.0f, 0.0f, 0.0f, 1.0f };

	// Effective rotation imparted to children/points: apply the animated Rotate
	// first, then the fixed JointOrient  ⇒  JointOrient ⊗ Rotate.
	[[nodiscard]] VlmQuat OrientedRotation() const noexcept {
		return VlmQuatMul(JointOrient, Rotate);
	}

	[[nodiscard]] VlmMat4 LocalMatrix() const noexcept {
		return OaTrsMatrix(Scale, OrientedRotation(), Translate);
	}

	// The fixed rest orientation as a matrix — the rotational part of Maya's
	// `offsetParentMatrix` (row-vector convention, no translation/scale).
	[[nodiscard]] VlmMat4 OffsetParentMatrix() const noexcept {
		return OaTrsMatrix({ 1.0f, 1.0f, 1.0f }, JointOrient, { 0.0f, 0.0f, 0.0f });
	}
};
