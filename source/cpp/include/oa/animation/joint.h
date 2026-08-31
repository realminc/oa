// oa::Joint — locked skeletal-joint transform value.
//
// The matrix contract follows Maya's joint transform semantics while retaining
// OA composition: scale/shear, scale orientation (rotateAxis), animated
// rotation, joint orientation, optional parent-scale compensation, translation.

#pragma once

#include <oa/core/transform.h>

namespace oa {

class FnJoint;

struct JointDegreesOfFreedom {
	bool x = true;
	bool y = true;
	bool z = true;
};

struct JointRotationLimits {
	oa::vlm::Vec3 minimumDegrees{};
	oa::vlm::Vec3 maximumDegrees{};
	JointDegreesOfFreedom minimumEnabled{false, false, false};
	JointDegreesOfFreedom maximumEnabled{false, false, false};
};

class Joint {
public:
	Joint() = default;
	explicit Joint(const oa::Transform& inTransform);

	[[nodiscard]] oa::Transform& getTransform() noexcept;
	[[nodiscard]] const oa::Transform& getTransform() const noexcept;

	void setOrientation(const oa::vlm::Quat& inOrientation);
	[[nodiscard]] const oa::vlm::Quat& getOrientation() const noexcept;
	void setScaleOrientation(const oa::vlm::Quat& inOrientation);
	[[nodiscard]] const oa::vlm::Quat& getScaleOrientation() const noexcept;

	void setSegmentScaleCompensate(bool inEnabled) noexcept;
	[[nodiscard]] bool getSegmentScaleCompensate() const noexcept;
	void setRotationOrder(oa::vlm::RotationOrder inOrder);
	[[nodiscard]] oa::vlm::RotationOrder getRotationOrder() const noexcept;

	void setDegreesOfFreedom(const JointDegreesOfFreedom& inDegreesOfFreedom) noexcept;
	[[nodiscard]] const JointDegreesOfFreedom& getDegreesOfFreedom() const noexcept;
	void setRotationLimits(const JointRotationLimits& inLimits);
	[[nodiscard]] const JointRotationLimits& getRotationLimits() const noexcept;
	void setPreferredAngles(const oa::vlm::Vec3& inDegrees);
	[[nodiscard]] const oa::vlm::Vec3& getPreferredAngles() const noexcept;
	void setStiffness(const oa::vlm::Vec3& inStiffness);
	[[nodiscard]] const oa::vlm::Vec3& getStiffness() const noexcept;

	[[nodiscard]] oa::vlm::Quat getOrientedRotation() const noexcept;
	[[nodiscard]] oa::vlm::Vec3 clampRotationDegrees(
		const oa::vlm::Vec3& inDegrees) const noexcept;
	[[nodiscard]] oa::vlm::Mat4 getMatrix(
		const oa::vlm::Vec3& inParentScale = {1.0F, 1.0F, 1.0F}
	) const noexcept;

private:
	friend class FnJoint;

	oa::Transform transform_;
	oa::vlm::Quat orientation_ = oa::vlm::Quat::identity();
	oa::vlm::Quat scaleOrientation_ = oa::vlm::Quat::identity();
	bool segmentScaleCompensate_ = true;
	oa::vlm::RotationOrder rotationOrder_ = oa::vlm::RotationOrder::Xyz;
	JointDegreesOfFreedom degreesOfFreedom_{};
	JointRotationLimits rotationLimits_{};
	oa::vlm::Vec3 preferredAngles_{};
	oa::vlm::Vec3 stiffness_{};
};

} // namespace oa
