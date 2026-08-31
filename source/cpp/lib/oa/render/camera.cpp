#include <oa/render/fnCamera.h>

namespace oa {

Camera::Camera(
	const oa::vlm::Vec3& inPosition,
	const oa::vlm::Vec3& inTarget,
	oa::F32 inFovY,
	oa::F32 inAspect,
	oa::F32 inNear,
	oa::F32 inFar) {
	oa::FnCamera::initPerspective(
		*this, inPosition, inTarget, inFovY, inAspect, inNear, inFar);
}

Camera::Camera(
	oa::F32 inWidth,
	oa::F32 inHeight,
	oa::F32 inNear,
	oa::F32 inFar) {
	oa::FnCamera::initOrthographic(*this, inWidth, inHeight, inNear, inFar);
}

oa::Transform& Camera::getTransform() noexcept { return transform_; }

const oa::Transform& Camera::getTransform() const noexcept { return transform_; }

void Camera::setPerspective(
	oa::F32 inFovY,
	oa::F32 inAspect,
	oa::F32 inNear,
	oa::F32 inFar) {
	oa::FnCamera::setPerspective(*this, inFovY, inAspect, inNear, inFar);
}

void Camera::setOrthographic(
	oa::F32 inWidth,
	oa::F32 inHeight,
	oa::F32 inNear,
	oa::F32 inFar) {
	oa::FnCamera::setOrthographic(*this, inWidth, inHeight, inNear, inFar);
}

void Camera::fitToWindow(oa::F32 inWindowWidth, oa::F32 inWindowHeight) {
	oa::FnCamera::fitToWindow(*this, inWindowWidth, inWindowHeight);
}

void Camera::setAspectRatio(oa::F32 inAspect) {
	oa::FnCamera::setAspectRatio(*this, inAspect);
}

CameraProjection Camera::getProjectionType() const noexcept {
	return oa::FnCamera::getProjectionType(*this);
}

oa::F32 Camera::getAspectRatio() const noexcept {
	return oa::FnCamera::getAspectRatio(*this);
}

oa::vlm::Vec2 Camera::getOrthographicSize() const noexcept {
	return oa::FnCamera::getOrthographicSize(*this);
}

oa::vlm::Mat4 Camera::getViewMatrix() const noexcept {
	return oa::FnCamera::getViewMatrix(*this);
}

oa::vlm::Mat4 Camera::getProjectionMatrix() const noexcept {
	return oa::FnCamera::getProjectionMatrix(*this);
}

oa::vlm::Mat4 Camera::getViewProjectionMatrix() const noexcept {
	return oa::FnCamera::getViewProjectionMatrix(*this);
}

void Camera::setNearFar(oa::F32 inNear, oa::F32 inFar) {
	oa::FnCamera::setNearFar(*this, inNear, inFar);
}

oa::F32 Camera::getNear() const noexcept { return oa::FnCamera::getNear(*this); }

oa::F32 Camera::getFar() const noexcept { return oa::FnCamera::getFar(*this); }

void Camera::setFovY(oa::F32 inFovY) {
	oa::FnCamera::setFovY(*this, inFovY);
}

oa::F32 Camera::getFovY() const noexcept { return oa::FnCamera::getFovY(*this); }

void Camera::setZoom(oa::F32 inZoom) {
	oa::FnCamera::setZoom(*this, inZoom);
}

oa::F32 Camera::getZoom() const noexcept { return oa::FnCamera::getZoom(*this); }

void Camera::setFocalLength(oa::F32 inFocalLengthMm) {
	oa::FnCamera::setFocalLength(*this, inFocalLengthMm);
}

oa::F32 Camera::getFocalLength() const noexcept {
	return oa::FnCamera::getFocalLength(*this);
}

void Camera::setSensorHeight(oa::F32 inSensorHeightMm) {
	oa::FnCamera::setSensorHeight(*this, inSensorHeightMm);
}

oa::F32 Camera::getSensorHeight() const noexcept {
	return oa::FnCamera::getSensorHeight(*this);
}

oa::F32 Camera::getEffectiveFovY() const noexcept {
	return oa::FnCamera::getEffectiveFovY(*this);
}

void Camera::setOffset(oa::F32 inOffsetX, oa::F32 inOffsetY) {
	oa::FnCamera::setOffset(*this, inOffsetX, inOffsetY);
}

oa::vlm::Vec2 Camera::getOffset() const noexcept {
	return oa::FnCamera::getOffset(*this);
}

void Camera::setOrbitTarget(const oa::vlm::Vec3& inTarget) {
	oa::FnCamera::setOrbitTarget(*this, inTarget);
}

void Camera::setOrbitDistance(oa::F32 inDistance) {
	oa::FnCamera::setOrbitDistance(*this, inDistance);
}

void Camera::orbitYawPitch(oa::F32 inYawDelta, oa::F32 inPitchDelta) {
	oa::FnCamera::orbitYawPitch(*this, inYawDelta, inPitchDelta);
}

void Camera::orbitSetYawPitch(oa::F32 inYaw, oa::F32 inPitch) {
	oa::FnCamera::orbitSetYawPitch(*this, inYaw, inPitch);
}

} // namespace oa
