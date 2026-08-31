#include <oa/core/fnTransform.h>

namespace oa {

Transform::Transform(
	const oa::vlm::Vec3& inTranslation,
	const oa::vlm::Quat& inRotation,
	const oa::vlm::Vec3& inScale,
	const oa::vlm::Vec3& inShear) {
	oa::FnTransform::setPosition(*this, inTranslation);
	oa::FnTransform::setRotation(*this, inRotation);
	oa::FnTransform::setScale(*this, inScale);
	oa::FnTransform::setShear(*this, inShear);
}

void Transform::setPosition(const oa::vlm::Vec3& inPosition) {
	oa::FnTransform::setPosition(*this, inPosition);
}

void Transform::setRotation(const oa::vlm::Quat& inRotation) {
	oa::FnTransform::setRotation(*this, inRotation);
}

void Transform::setScale(const oa::vlm::Vec3& inScale) {
	oa::FnTransform::setScale(*this, inScale);
}

void Transform::setShear(const oa::vlm::Vec3& inShear) {
	oa::FnTransform::setShear(*this, inShear);
}

const oa::vlm::Vec3& Transform::getPosition() const noexcept {
	return oa::FnTransform::getPosition(*this);
}

const oa::vlm::Quat& Transform::getRotation() const noexcept {
	return oa::FnTransform::getRotation(*this);
}

const oa::vlm::Vec3& Transform::getScale() const noexcept {
	return oa::FnTransform::getScale(*this);
}

const oa::vlm::Vec3& Transform::getShear() const noexcept {
	return oa::FnTransform::getShear(*this);
}

oa::vlm::Vec3 Transform::getForward() const noexcept {
	return oa::FnTransform::getForward(*this);
}

oa::vlm::Vec3 Transform::getRight() const noexcept {
	return oa::FnTransform::getRight(*this);
}

oa::vlm::Vec3 Transform::getUp() const noexcept {
	return oa::FnTransform::getUp(*this);
}

void Transform::lookAt(
	const oa::vlm::Vec3& inTarget,
	const oa::vlm::Vec3& inUp) {
	oa::FnTransform::lookAt(*this, inTarget, inUp);
}

void Transform::setRotationDegrees(
	const oa::vlm::Vec3& inDegrees,
	oa::vlm::RotationOrder inOrder) {
	oa::FnTransform::setRotationDegrees(*this, inDegrees, inOrder);
}

oa::vlm::Vec3 Transform::getRotationDegrees(
	oa::vlm::RotationOrder inOrder) const noexcept {
	return oa::FnTransform::getRotationDegrees(*this, inOrder);
}

void Transform::panLocal(
	oa::F32 inRight,
	oa::F32 inUp,
	oa::F32 inForward) {
	oa::FnTransform::panLocal(*this, inRight, inUp, inForward);
}

oa::vlm::Mat4 Transform::getMatrix() const noexcept {
	return oa::FnTransform::getMatrix(*this);
}

oa::Status Transform::setMatrix(const oa::vlm::Mat4& inMatrix) noexcept {
	return oa::FnTransform::setMatrix(*this, inMatrix);
}

} // namespace oa
