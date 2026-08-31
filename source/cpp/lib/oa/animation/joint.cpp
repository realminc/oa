#include <oa/animation/fnJoint.h>

namespace oa {

Joint::Joint(const oa::Transform& inTransform) : transform_(inTransform) {}

oa::Transform& Joint::getTransform() noexcept {
	return oa::FnJoint::getTransform(*this);
}

const oa::Transform& Joint::getTransform() const noexcept {
	return oa::FnJoint::getTransform(*this);
}

void Joint::setOrientation(const oa::vlm::Quat& inOrientation) {
	oa::FnJoint::setOrientation(*this, inOrientation);
}

const oa::vlm::Quat& Joint::getOrientation() const noexcept {
	return oa::FnJoint::getOrientation(*this);
}

void Joint::setScaleOrientation(const oa::vlm::Quat& inOrientation) {
	oa::FnJoint::setScaleOrientation(*this, inOrientation);
}

const oa::vlm::Quat& Joint::getScaleOrientation() const noexcept {
	return oa::FnJoint::getScaleOrientation(*this);
}

void Joint::setSegmentScaleCompensate(bool inEnabled) noexcept {
	oa::FnJoint::setSegmentScaleCompensate(*this, inEnabled);
}

bool Joint::getSegmentScaleCompensate() const noexcept {
	return oa::FnJoint::getSegmentScaleCompensate(*this);
}

void Joint::setRotationOrder(oa::vlm::RotationOrder inOrder) {
	oa::FnJoint::setRotationOrder(*this, inOrder);
}

oa::vlm::RotationOrder Joint::getRotationOrder() const noexcept {
	return oa::FnJoint::getRotationOrder(*this);
}

void Joint::setDegreesOfFreedom(
	const JointDegreesOfFreedom& inDegreesOfFreedom) noexcept {
	oa::FnJoint::setDegreesOfFreedom(*this, inDegreesOfFreedom);
}

const JointDegreesOfFreedom& Joint::getDegreesOfFreedom() const noexcept {
	return oa::FnJoint::getDegreesOfFreedom(*this);
}

void Joint::setRotationLimits(const JointRotationLimits& inLimits) {
	oa::FnJoint::setRotationLimits(*this, inLimits);
}

const JointRotationLimits& Joint::getRotationLimits() const noexcept {
	return oa::FnJoint::getRotationLimits(*this);
}

void Joint::setPreferredAngles(const oa::vlm::Vec3& inDegrees) {
	oa::FnJoint::setPreferredAngles(*this, inDegrees);
}

const oa::vlm::Vec3& Joint::getPreferredAngles() const noexcept {
	return oa::FnJoint::getPreferredAngles(*this);
}

void Joint::setStiffness(const oa::vlm::Vec3& inStiffness) {
	oa::FnJoint::setStiffness(*this, inStiffness);
}

const oa::vlm::Vec3& Joint::getStiffness() const noexcept {
	return oa::FnJoint::getStiffness(*this);
}

oa::vlm::Quat Joint::getOrientedRotation() const noexcept {
	return oa::FnJoint::getOrientedRotation(*this);
}

oa::vlm::Vec3 Joint::clampRotationDegrees(
	const oa::vlm::Vec3& inDegrees) const noexcept {
	return oa::FnJoint::clampRotationDegrees(*this, inDegrees);
}

oa::vlm::Mat4 Joint::getMatrix(
	const oa::vlm::Vec3& inParentScale) const noexcept {
	return oa::FnJoint::getMatrix(*this, inParentScale);
}

} // namespace oa
