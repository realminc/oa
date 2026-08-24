#pragma once

// Joint — a skeletal joint transform (Maya `MFnIkJoint` analog).
//
// Extends Transform with a fixed `jointOrient`: the joint's rest orientation,
// against which the animated `rotate` is a delta. in Maya terms the local matrix
// is `S · rotate · jointOrient · translate` (rotate applied first, then the
// orient) — and `jointOrient` is exactly the rotational part of the Maya-2020+
// `offsetParentMatrix`, surfaced here via `offsetParentMatrix()`.
//
// forward kinematics over a chain of OaJoints is cleanest in quaternion form
// (see Skeleton::restWorld), but `localMatrix()` is provided for the
// matrix-composition path and for callers that want the UE/Maya offset-parent
// representation.

#include <core/transform.h>

namespace oa {

struct Joint : Transform {
	oa::vlm::Quat jointOrient = { 0.0f, 0.0f, 0.0f, 1.0f };

	// Effective rotation imparted to children/points: apply the animated rotate
	// first, then the fixed jointOrient  ⇒  jointOrient ⊗ rotate.
	[[nodiscard]] oa::vlm::Quat orientedRotation() const noexcept {
		return jointOrient * rotate;
	}

	[[nodiscard]] oa::vlm::Mat4 localMatrix() const noexcept {
		return trsMatrix(scale, orientedRotation(), translate);
	}

	// The fixed rest orientation as a matrix — the rotational part of Maya's
	// `offsetParentMatrix` (row-vector convention, no translation/scale).
	[[nodiscard]] oa::vlm::Mat4 offsetParentMatrix() const noexcept {
		return trsMatrix({ 1.0f, 1.0f, 1.0f }, jointOrient, { 0.0f, 0.0f, 0.0f });
	}
};

}  // namespace oa
