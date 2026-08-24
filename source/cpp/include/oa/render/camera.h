// oa::Camera — class wrapper around oa::FnCamera.
//
// OOP wrapper for the functional camera API. All methods delegate to oa::FnCamera.
// For the functional namespace, see <oa/render/fnCamera.h>.
//
// usage (3D perspective):
//   oa::Camera camera({0.0f, 2.0f, 5.0f});  // pos, target, fov, aspect, near, far
//   oa::vlm::Mat4 view = camera.getViewMatrix();
//
// usage (2D orthographic for image viewing):
//   oa::Camera camera(imageWidth, imageHeight);  // width, height, near, far
//   camera.fitToWindow(windowWidth, windowHeight);

#pragma once

#include <oa/render/fnCamera.h>

namespace oa {

class Camera {
public:
	// Default constructor — perspective camera at (0,0,5) looking at origin
	Camera() = default;

	// 3D perspective camera constructor
	explicit Camera(
		const oa::vlm::Vec3& inPosition,
		const oa::vlm::Vec3& inTarget    = {0.0f, 0.0f, 0.0f},
		oa::F32         inFovY       = 60.0f,
		oa::F32         inAspect     = 1280.0f / 720.0f,
		oa::F32         inNear       = 0.1f,
		oa::F32         inFar        = 10000.0f
	) {
		oa::FnCamera::initPerspective(state_, inPosition, inTarget, inFovY, inAspect, inNear, inFar);
	}

	// 2D orthographic camera constructor (for image/video viewing)
	explicit Camera(
		oa::F32 inWidth,
		oa::F32 inHeight,
		oa::F32 inNear = 0.0f,
		oa::F32 inFar  = 100.0f
	) {
		oa::FnCamera::initOrthographic(state_, inWidth, inHeight, inNear, inFar);
	}

	~Camera() = default;

	// Transform
	void setPosition(const oa::vlm::Vec3& inPosition) {
		oa::FnCamera::setPosition(state_, inPosition);
	}
	void setRotation(const oa::vlm::Quat& inRotation) {
		oa::FnCamera::setRotation(state_, inRotation);
	}
	void lookAt(const oa::vlm::Vec3& inTarget, const oa::vlm::Vec3& inUp = {0.0f, 1.0f, 0.0f}) {
		oa::FnCamera::lookAt(state_, inTarget, inUp);
	}

	[[nodiscard]] const oa::vlm::Vec3& getPosition() const noexcept {
		return oa::FnCamera::getPosition(state_);
	}
	[[nodiscard]] const oa::vlm::Quat& getRotation() const noexcept {
		return oa::FnCamera::getRotation(state_);
	}
	[[nodiscard]] oa::vlm::Vec3 getForward() const noexcept {
		return oa::FnCamera::getForward(state_);
	}
	[[nodiscard]] oa::vlm::Vec3 getRight() const noexcept { return oa::FnCamera::getRight(state_); }
	[[nodiscard]] oa::vlm::Vec3 getUp() const noexcept { return oa::FnCamera::getUp(state_); }

	// projection (3D perspective)
	void setPerspective(
		oa::F32 inFovY,
		oa::F32 inAspect,
		oa::F32 inNear,
		oa::F32 inFar
	) {
		oa::FnCamera::setPerspective(state_, inFovY, inAspect, inNear, inFar);
	}

	// projection (2D orthographic)
	void setOrthographic(
		oa::F32 inWidth,
		oa::F32 inHeight,
		oa::F32 inNear = 0.0f,
		oa::F32 inFar  = 1.0f
	) {
		oa::FnCamera::setOrthographic(state_, inWidth, inHeight, inNear, inFar);
	}

	// viewport aspect ratio helper
	void fitToWindow(oa::F32 inWindowWidth, oa::F32 inWindowHeight) {
		oa::FnCamera::fitToWindow(state_, inWindowWidth, inWindowHeight);
	}
	void setAspectRatio(oa::F32 inAspect) { oa::FnCamera::setAspectRatio(state_, inAspect); }

	[[nodiscard]] CameraProjection getProjectionType() const noexcept {
		return oa::FnCamera::getProjectionType(state_);
	}

	// matrices
	[[nodiscard]] oa::vlm::Mat4 getViewMatrix() const noexcept {
		return oa::FnCamera::getViewMatrix(state_);
	}
	[[nodiscard]] oa::vlm::Mat4 getProjectionMatrix() const noexcept {
		return oa::FnCamera::getProjectionMatrix(state_);
	}
	[[nodiscard]] oa::vlm::Mat4 getViewProjectionMatrix() const noexcept {
		return oa::FnCamera::getViewProjectionMatrix(state_);
	}

	// Frustum
	void setNearFar(oa::F32 inNear, oa::F32 inFar) { oa::FnCamera::setNearFar(state_, inNear, inFar); }
	[[nodiscard]] oa::F32 getNear() const noexcept { return oa::FnCamera::getNear(state_); }
	[[nodiscard]] oa::F32 getFar() const noexcept { return oa::FnCamera::getFar(state_); }

	// FOV (perspective only)
	void setFovY(oa::F32 inFovY) { oa::FnCamera::setFovY(state_, inFovY); }
	[[nodiscard]] oa::F32 getFovY() const noexcept { return oa::FnCamera::getFovY(state_); }

	// zoom (orthographic only)
	void setZoom(oa::F32 inZoom) { oa::FnCamera::setZoom(state_, inZoom); }
	[[nodiscard]] oa::F32 getZoom() const noexcept { return oa::FnCamera::getZoom(state_); }

	// Focal length / sensor (photography-style perspective)
	void setFocalLength(oa::F32 inFocalLengthMm) { oa::FnCamera::setFocalLength(state_, inFocalLengthMm); }
	[[nodiscard]] oa::F32 getFocalLength() const noexcept { return oa::FnCamera::getFocalLength(state_); }
	void setSensorHeight(oa::F32 inSensorHeightMm) { oa::FnCamera::setSensorHeight(state_, inSensorHeightMm); }
	[[nodiscard]] oa::F32 getSensorHeight() const noexcept { return oa::FnCamera::getSensorHeight(state_); }

	// Effective FOV
	[[nodiscard]] oa::F32 getEffectiveFovY() const noexcept { return oa::FnCamera::getEffectiveFovY(state_); }

	// Screen offset (lens shift / 2D pan)
	void setOffset(oa::F32 inOffsetX, oa::F32 inOffsetY) { oa::FnCamera::setOffset(state_, inOffsetX, inOffsetY); }
	[[nodiscard]] oa::vlm::Vec2 getOffset() const noexcept { return oa::FnCamera::getOffset(state_); }

	// Orbit controls (rotate around target)
	void setOrbitTarget(const oa::vlm::Vec3& inTarget) {
		oa::FnCamera::setOrbitTarget(state_, inTarget);
	}
	void setOrbitDistance(oa::F32 inDistance) { oa::FnCamera::setOrbitDistance(state_, inDistance); }
	void orbitYawPitch(oa::F32 inYawDelta, oa::F32 inPitchDelta) { oa::FnCamera::orbitYawPitch(state_, inYawDelta, inPitchDelta); }
	void orbitSetYawPitch(oa::F32 inYaw, oa::F32 inPitch) { oa::FnCamera::orbitSetYawPitch(state_, inYaw, inPitch); }

	// Euler angles (yaw/pitch/roll in degrees)
	void setYawPitchRoll(oa::F32 inYawDeg, oa::F32 inPitchDeg, oa::F32 inRollDeg) {
		oa::FnCamera::setYawPitchRoll(state_, inYawDeg, inPitchDeg, inRollDeg);
	}
	[[nodiscard]] oa::vlm::Vec3 getYawPitchRoll() const noexcept {
		return oa::FnCamera::getYawPitchRoll(state_);
	}

	// camera-space movement (pan in camera local axes)
	void panLocal(oa::F32 inRight, oa::F32 inUp, oa::F32 inForward) {
		oa::FnCamera::panLocal(state_, inRight, inUp, inForward);
	}

	// Direct state access (for advanced use)
	[[nodiscard]] CameraState& getState() noexcept { return state_; }
	[[nodiscard]] const CameraState& getState() const noexcept { return state_; }

private:
	CameraState state_;
};

} // namespace oa
