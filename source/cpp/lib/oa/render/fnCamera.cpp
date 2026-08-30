// oa::Camera implementation

#include <oa/render/camera.h>
#include <oa/core/std/scalarMath.h>
#include <oa/core/vlm.h>

namespace oa {

namespace FnCamera {

// ─── Factory / init ──────────────────────────────────────────────────────

void initPerspective(
	oa::CameraState& inState,
	const oa::vlm::Vec3&  inPosition,
	const oa::vlm::Vec3&  inTarget,
	oa::F32          inFovY,
	oa::F32          inAspect,
	oa::F32          inNear,
	oa::F32          inFar
) {
	inState.position   = inPosition;
	inState.rotation   = {0.0f, 0.0f, 0.0f, 1.0f};
	inState.projection = oa::CameraProjection::Perspective;
	inState.fovY       = inFovY;
	inState.aspect     = inAspect;
	inState.near       = inNear;
	inState.far        = inFar;
	inState.focalLength = 0.0f;
	inState.offsetX    = 0.0f;
	inState.offsetY    = 0.0f;
	inState.useOrbit   = false;
	lookAt(inState, inTarget);
}

void initOrthographic(
	oa::CameraState& inState,
	oa::F32          inWidth,
	oa::F32          inHeight,
	oa::F32          inNear,
	oa::F32          inFar
) {
	inState.position    = {0.0f, 0.0f, 1.0f};
	inState.rotation    = {0.0f, 0.0f, 0.0f, 1.0f};
	inState.projection  = oa::CameraProjection::Orthographic;
	inState.orthoWidth  = inWidth;
	inState.orthoHeight = inHeight;
	inState.near        = inNear;
	inState.far         = inFar;
	inState.zoom        = 1.0f;
	inState.offsetX     = 0.0f;
	inState.offsetY     = 0.0f;
	inState.useOrbit    = false;
}

// ─── Transform ─────────────────────────────────────────────────────────────

void setPosition(oa::CameraState& inState, const oa::vlm::Vec3& inPosition) {
	inState.position = inPosition;
}

void setRotation(oa::CameraState& inState, const oa::vlm::Quat& inRotation) {
	inState.rotation = inRotation;
}

void lookAt(oa::CameraState& inState, const oa::vlm::Vec3& inTarget, const oa::vlm::Vec3& inUp) {
	inState.rotation = oa::vlm::lookRotation(
		inTarget - inState.position, inUp);
}

const oa::vlm::Vec3& getPosition(const oa::CameraState& inState) noexcept {
	return inState.position;
}

const oa::vlm::Quat& getRotation(const oa::CameraState& inState) noexcept {
	return inState.rotation;
}

oa::vlm::Vec3 getForward(const oa::CameraState& inState) noexcept {
	return oa::vlm::rotateVector(inState.rotation, {0.0f, 0.0f, -1.0f});
}

oa::vlm::Vec3 getRight(const oa::CameraState& inState) noexcept {
	return oa::vlm::rotateVector(inState.rotation, {1.0f, 0.0f, 0.0f});
}

oa::vlm::Vec3 getUp(const oa::CameraState& inState) noexcept {
	return oa::vlm::rotateVector(inState.rotation, {0.0f, 1.0f, 0.0f});
}

// ─── projection ───────────────────────────────────────────────────────────

void setPerspective(
	oa::CameraState& inState,
	oa::F32 inFovY,
	oa::F32 inAspect,
	oa::F32 inNear,
	oa::F32 inFar
) {
	inState.projection = oa::CameraProjection::Perspective;
	inState.fovY = inFovY;
	inState.aspect = inAspect;
	inState.near = inNear;
	inState.far = inFar;
}

void setOrthographic(
	oa::CameraState& inState,
	oa::F32 inWidth,
	oa::F32 inHeight,
	oa::F32 inNear,
	oa::F32 inFar
) {
	inState.projection = oa::CameraProjection::Orthographic;
	inState.orthoWidth = inWidth;
	inState.orthoHeight = inHeight;
	inState.near = inNear;
	inState.far = inFar;
}

void fitToWindow(oa::CameraState& inState, oa::F32 inWindowWidth, oa::F32 inWindowHeight) {
	if (inState.projection == oa::CameraProjection::Orthographic) {
		// fit orthographic view to window while maintaining aspect ratio
		oa::F32 windowAspect = inWindowWidth / inWindowHeight;
		oa::F32 contentAspect = inState.orthoWidth / inState.orthoHeight;

		if (windowAspect > contentAspect) {
			// Window is wider - fit height
			inState.orthoWidth = inState.orthoHeight * windowAspect;
		} else {
			// Window is taller - fit width
			inState.orthoHeight = inState.orthoWidth / windowAspect;
		}
	} else {
		// Update aspect ratio for perspective
		inState.aspect = inWindowWidth / inWindowHeight;
	}
}

void setAspectRatio(oa::CameraState& inState, oa::F32 inAspect) {
	if (inState.projection == oa::CameraProjection::Perspective) {
		inState.aspect = inAspect;
	}
}

oa::CameraProjection getProjectionType(const oa::CameraState& inState) noexcept {
	return inState.projection;
}

// ─── matrices ─────────────────────────────────────────────────────────────

oa::vlm::Mat4 getViewMatrix(const oa::CameraState& inState) noexcept {
	return oa::vlm::viewFromPose(inState.position, inState.rotation);
}

oa::vlm::Mat4 getProjectionMatrix(const oa::CameraState& inState) noexcept {
	if (inState.projection == oa::CameraProjection::Perspective) {
		return oa::vlm::perspectiveShifted(
			getEffectiveFovY(inState), inState.aspect,
			inState.near, inState.far,
			{inState.offsetX, inState.offsetY});
	}
	return oa::vlm::orthographicShifted(
		inState.orthoWidth, inState.orthoHeight,
		inState.near, inState.far, inState.zoom,
		{inState.offsetX, inState.offsetY});
}

oa::vlm::Mat4 getViewProjectionMatrix(const oa::CameraState& inState) noexcept {
	// OA shaders multiply row vectors: mul(position, matrix).
	return oa::vlm::matrixMul(getViewMatrix(inState), getProjectionMatrix(inState));
}

// ─── Frustum ─────────────────────────────────────────────────────────────

void setNearFar(oa::CameraState& inState, oa::F32 inNear, oa::F32 inFar) {
	inState.near = inNear;
	inState.far = inFar;
}

oa::F32 getNear(const oa::CameraState& inState) noexcept {
	return inState.near;
}

oa::F32 getFar(const oa::CameraState& inState) noexcept {
	return inState.far;
}

// ─── FOV ─────────────────────────────────────────────────────────────────

void setFovY(oa::CameraState& inState, oa::F32 inFovY) {
	inState.fovY = inFovY;
}

oa::F32 getFovY(const oa::CameraState& inState) noexcept {
	return inState.fovY;
}

// ─── zoom ────────────────────────────────────────────────────────────────

void setZoom(oa::CameraState& inState, oa::F32 inZoom) {
	inState.zoom = inZoom;
}

oa::F32 getZoom(const oa::CameraState& inState) noexcept {
	return inState.zoom;
}

// ─── Focal length / Sensor ─────────────────────────────────────────────────

void setFocalLength(oa::CameraState& inState, oa::F32 inFocalLengthMm) {
	inState.focalLength = inFocalLengthMm;
}

oa::F32 getFocalLength(const oa::CameraState& inState) noexcept {
	return inState.focalLength;
}

void setSensorHeight(oa::CameraState& inState, oa::F32 inSensorHeightMm) {
	inState.sensorHeight = inSensorHeightMm;
}

oa::F32 getSensorHeight(const oa::CameraState& inState) noexcept {
	return inState.sensorHeight;
}

oa::F32 getEffectiveFovY(const oa::CameraState& inState) noexcept {
	if (inState.focalLength > 0.0f && inState.sensorHeight > 0.0f) {
		// FOV = 2 * arctan(sensorHeight / (2 * focalLength))
		return 2.0f * oa::atan(inState.sensorHeight / (2.0f * inState.focalLength)) * 180.0f / 3.14159265358979323846f;
	}
	return inState.fovY;
}

// ─── Screen offset ───────────────────────────────────────────────────────

void setOffset(oa::CameraState& inState, oa::F32 inOffsetX, oa::F32 inOffsetY) {
	inState.offsetX = inOffsetX;
	inState.offsetY = inOffsetY;
}

oa::vlm::Vec2 getOffset(const oa::CameraState& inState) noexcept {
	return {inState.offsetX, inState.offsetY};
}

// ─── Orbit Controls ──────────────────────────────────────────────────────

void setOrbitTarget(oa::CameraState& inState, const oa::vlm::Vec3& inTarget) {
	inState.orbitTarget = inTarget;
	inState.useOrbit = true;
}

void setOrbitDistance(oa::CameraState& inState, oa::F32 inDistance) {
	// Reposition camera along the forward vector from target
	oa::vlm::Vec3 forward = getForward(inState);
	inState.position.x = inState.orbitTarget.x - forward.x * inDistance;
	inState.position.y = inState.orbitTarget.y - forward.y * inDistance;
	inState.position.z = inState.orbitTarget.z - forward.z * inDistance;
}

void orbitYawPitch(oa::CameraState& inState, oa::F32 inYawDelta, oa::F32 inPitchDelta) {
	oa::vlm::Vec3 offset = oa::vlm::sub(inState.position, inState.orbitTarget);
	oa::vlm::Vec3 sph = oa::vlm::cartesianToSpherical(offset);
	if (sph.z < oa::vlm::Tolerance<oa::F32>) return;

	sph.x += oa::vlm::radians(inYawDelta);     // yaw
	sph.y += oa::vlm::radians(inPitchDelta);  // pitch

	// Clamp pitch
	oa::F32 pitchLimit = oa::vlm::radians(89.0F);
	if (sph.y > pitchLimit) sph.y = pitchLimit;
	if (sph.y < -pitchLimit) sph.y = -pitchLimit;

	inState.position = oa::vlm::add(
		inState.orbitTarget,
		oa::vlm::sphericalToCartesian(sph.x, sph.y, sph.z));
	lookAt(inState, inState.orbitTarget);
}

void orbitSetYawPitch(oa::CameraState& inState, oa::F32 inYaw, oa::F32 inPitch) {
	oa::vlm::Vec3 offset = oa::vlm::sub(inState.position, inState.orbitTarget);
	oa::vlm::Vec3 sph = oa::vlm::cartesianToSpherical(offset);
	if (sph.z < oa::vlm::Tolerance<oa::F32>) return;

	inState.position = oa::vlm::add(
		inState.orbitTarget,
		oa::vlm::sphericalToCartesian(
			oa::vlm::radians(inYaw),
			oa::vlm::radians(inPitch),
			sph.z)
	);
	lookAt(inState, inState.orbitTarget);
}

// ─── Euler Angles ─────────────────────────────────────────────────────────

void setYawPitchRoll(oa::CameraState& inState, oa::F32 inYawDeg, oa::F32 inPitchDeg, oa::F32 inRollDeg) {
	inState.rotation = oa::vlm::quaternionFromEuler(inYawDeg, inPitchDeg, inRollDeg);
}

oa::vlm::Vec3 getYawPitchRoll(const oa::CameraState& inState) noexcept {
	return oa::vlm::quaternionToEuler(inState.rotation);
}

// ─── camera-Space movement ───────────────────────────────────────────────

void panLocal(oa::CameraState& inState, oa::F32 inRight, oa::F32 inUp, oa::F32 inForward) {
	oa::vlm::Vec3 delta = oa::vlm::add(
		oa::vlm::add(
			oa::vlm::scale(getRight(inState), inRight),
			oa::vlm::scale(getUp(inState), inUp)
		),
		oa::vlm::scale(getForward(inState), inForward)
	);
	inState.position = oa::vlm::add(inState.position, delta);
}

} // namespace FnCamera

} // namespace oa
