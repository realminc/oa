// oa::FnJoint — stateless operations on oa::Joint values.

#pragma once

#include <oa/animation/joint.h>
#include <oa/core/status.h>

namespace oa {

class FnJoint final {
public:
	FnJoint() = delete;

	[[nodiscard]] static oa::Status validate(const oa::Joint& inJoint);

	[[nodiscard]] static oa::Transform& getTransform(oa::Joint& inJoint) noexcept;
	[[nodiscard]] static const oa::Transform& getTransform(
		const oa::Joint& inJoint) noexcept;

	static void setOrientation(
		oa::Joint& inJoint,
		const oa::vlm::Quat& inOrientation
	);
	[[nodiscard]] static const oa::vlm::Quat& getOrientation(
		const oa::Joint& inJoint) noexcept;
	static void setScaleOrientation(
		oa::Joint& inJoint,
		const oa::vlm::Quat& inOrientation
	);
	[[nodiscard]] static const oa::vlm::Quat& getScaleOrientation(
		const oa::Joint& inJoint) noexcept;

	static void setSegmentScaleCompensate(
		oa::Joint& inJoint,
		bool inEnabled
	) noexcept;
	[[nodiscard]] static bool getSegmentScaleCompensate(
		const oa::Joint& inJoint) noexcept;
	static void setRotationOrder(
		oa::Joint& inJoint,
		oa::vlm::RotationOrder inOrder
	);
	[[nodiscard]] static oa::vlm::RotationOrder getRotationOrder(
		const oa::Joint& inJoint) noexcept;

	static void setDegreesOfFreedom(
		oa::Joint& inJoint,
		const oa::JointDegreesOfFreedom& inDegreesOfFreedom
	) noexcept;
	[[nodiscard]] static const oa::JointDegreesOfFreedom& getDegreesOfFreedom(
		const oa::Joint& inJoint) noexcept;
	static void setRotationLimits(
		oa::Joint& inJoint,
		const oa::JointRotationLimits& inLimits
	);
	[[nodiscard]] static const oa::JointRotationLimits& getRotationLimits(
		const oa::Joint& inJoint) noexcept;
	static void setPreferredAngles(
		oa::Joint& inJoint,
		const oa::vlm::Vec3& inDegrees
	);
	[[nodiscard]] static const oa::vlm::Vec3& getPreferredAngles(
		const oa::Joint& inJoint) noexcept;
	static void setStiffness(
		oa::Joint& inJoint,
		const oa::vlm::Vec3& inStiffness
	);
	[[nodiscard]] static const oa::vlm::Vec3& getStiffness(
		const oa::Joint& inJoint) noexcept;

	[[nodiscard]] static oa::vlm::Quat getOrientedRotation(
		const oa::Joint& inJoint) noexcept;
	[[nodiscard]] static oa::vlm::Vec3 clampRotationDegrees(
		const oa::Joint& inJoint,
		const oa::vlm::Vec3& inDegrees
	) noexcept;
	[[nodiscard]] static oa::vlm::Mat4 getMatrix(
		const oa::Joint& inJoint,
		const oa::vlm::Vec3& inParentScale = {1.0F, 1.0F, 1.0F}
	) noexcept;
};

} // namespace oa
