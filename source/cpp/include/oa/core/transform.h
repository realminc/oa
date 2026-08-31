// oa::Transform — semantic local spatial transform value.
//
// Transform owns affine placement only. Cameras, meshes, lights, joints, and
// other scene semantics compose it instead of duplicating spatial state.

#pragma once

#include <oa/core/status.h>
#include <oa/core/vlm.h>

namespace oa {

class FnTransform;

class Transform {
public:
	Transform() = default;
	Transform(
		const oa::vlm::Vec3& inTranslation,
		const oa::vlm::Quat& inRotation = oa::vlm::Quat::identity(),
		const oa::vlm::Vec3& inScale = {1.0F, 1.0F, 1.0F},
		const oa::vlm::Vec3& inShear = {0.0F, 0.0F, 0.0F}
	);

	void setPosition(const oa::vlm::Vec3& inPosition);
	void setRotation(const oa::vlm::Quat& inRotation);
	void setScale(const oa::vlm::Vec3& inScale);
	void setShear(const oa::vlm::Vec3& inShear);

	[[nodiscard]] const oa::vlm::Vec3& getPosition() const noexcept;
	[[nodiscard]] const oa::vlm::Quat& getRotation() const noexcept;
	[[nodiscard]] const oa::vlm::Vec3& getScale() const noexcept;
	[[nodiscard]] const oa::vlm::Vec3& getShear() const noexcept;

	[[nodiscard]] oa::vlm::Vec3 getForward() const noexcept;
	[[nodiscard]] oa::vlm::Vec3 getRight() const noexcept;
	[[nodiscard]] oa::vlm::Vec3 getUp() const noexcept;

	void lookAt(
		const oa::vlm::Vec3& inTarget,
		const oa::vlm::Vec3& inUp = {0.0F, 1.0F, 0.0F}
	);
	void setRotationDegrees(
		const oa::vlm::Vec3& inDegrees,
		oa::vlm::RotationOrder inOrder = oa::vlm::RotationOrder::Xyz
	);
	[[nodiscard]] oa::vlm::Vec3 getRotationDegrees(
		oa::vlm::RotationOrder inOrder = oa::vlm::RotationOrder::Xyz
	) const noexcept;
	void panLocal(oa::F32 inRight, oa::F32 inUp, oa::F32 inForward);

	[[nodiscard]] oa::vlm::Mat4 getMatrix() const noexcept;
	[[nodiscard]] oa::Status setMatrix(const oa::vlm::Mat4& inMatrix) noexcept;

private:
	friend class FnTransform;

	oa::vlm::Vec3 translation_ = {0.0F, 0.0F, 0.0F};
	oa::vlm::Quat rotation_ = oa::vlm::Quat::identity();
	oa::vlm::Vec3 scale_ = {1.0F, 1.0F, 1.0F};
	// Dimensionless row-basis shear factors: XY, XZ, and YZ.
	oa::vlm::Vec3 shear_ = {0.0F, 0.0F, 0.0F};
};

} // namespace oa
