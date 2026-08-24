#pragma once

// Transform — a TRS transform node (Maya `MTransform` / `MFnTransform` analog).
//
// Row-vector convention throughout: a point maps as `p' = p · M`, with the
// translation living in row 3 of the matrix — matching the rest of the engine
// math. VLM quaternion, TRS, camera, and shader transforms share this one
// convention; callers never transpose or flip axes between them.

#include <oa/core/vlm.h>

namespace oa {

// Build a quaternion from Maya-style XYZ Euler angles in degrees: the rotation
// applies X first, then Y, then Z (Maya's default `rotateOrder`). This is how
// rtg `rotateX/Y/Z` values bake into rest orientations.
[[nodiscard]] oa::vlm::Quat eulerXyzDegToQuat(const oa::vlm::Vec3& inDegXyz) noexcept;

// Encode/decode the first two columns of the conventional column-vector
// rotation matrix. VLM stores the same basis as rows because its matrices use
// row vectors; these helpers own that representation boundary.
void quaternionToSixD(
	const oa::vlm::Quat& inQuaternion,
	oa::F32 outRotation[6]) noexcept;
[[nodiscard]] oa::vlm::Quat quaternionFromSixD(
	const oa::F32 inRotation[6]) noexcept;

// Compose a TRS local matrix (row-vector S·R·T): a point is scaled, rotated by
// `Rot`, then translated by `Trans`. Consistent with `transformPoint`.
[[nodiscard]] oa::vlm::Mat4 trsMatrix(
	const oa::vlm::Vec3& inScale,
	const oa::vlm::Quat& inRot,
	const oa::vlm::Vec3& inTrans) noexcept;

// Transform a point by a row-vector matrix: `p' = p · M` (applies the 3×3 then
// adds row 3). The correct extractor for forward-kinematics chains built with
// `trsMatrix` + `oa::vlm::matrixMul`.
[[nodiscard]] oa::vlm::Vec3 transformPoint(
	const oa::vlm::Mat4& inM,
	const oa::vlm::Vec3& inP
) noexcept;

struct Transform {
	oa::vlm::Vec3 translate = { 0.0f, 0.0f, 0.0f };
	oa::vlm::Quat rotate    = { 0.0f, 0.0f, 0.0f, 1.0f };
	oa::vlm::Vec3 scale     = { 1.0f, 1.0f, 1.0f };

	// The rotation this node imparts to its children. A plain transform just
	// returns `rotate`; `Joint` folds in its `jointOrient`.
	[[nodiscard]] oa::vlm::Quat orientedRotation() const noexcept { return rotate; }

	// local matrix mapping this node's space into its parent's space.
	[[nodiscard]] oa::vlm::Mat4 localMatrix() const noexcept {
		return trsMatrix(scale, orientedRotation(), translate);
	}
};

}  // namespace oa
