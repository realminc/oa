// oa::FnCamera — functional camera operations.
//
// Stateless camera math. Operates on oa::CameraState POD.
// For the OOP wrapper, see <oa/render/camera.h>.
//
// usage:
//   oa::CameraState state;
//   oa::FnCamera::setPosition(state, {0.0f, 2.0f, 5.0f});
//   oa::FnCamera::lookAt(state, {0.0f, 0.0f, 0.0f});
//   oa::FnCamera::setPerspective(state, 60.0f, 16.0f / 9.0f, 0.1f, 100.0f);
//   oa::vlm::Mat4 view = oa::FnCamera::getViewMatrix(state);

#pragma once

#include <oa/core/types.h>
#include <oa/core/vlm.h>

namespace oa {

// camera projection type
enum class CameraProjection : oa::U8 {
	Perspective,   // 3D perspective projection
	Orthographic,  // 2D orthographic projection
};

// camera state (POD for functional API)
struct CameraState {
	oa::vlm::Vec3 position = {0.0f, 0.0f, 5.0f};
	oa::vlm::Quat rotation = {0.0f, 0.0f, 0.0f, 1.0f};  // Identity
	CameraProjection projection = CameraProjection::Perspective;

	// Perspective params
	oa::F32 fovY   = 60.0f;
	oa::F32 aspect = 1280.0f / 720.0f;
	oa::F32 near   = 0.1f;
	oa::F32 far    = 100.0f;

	// Photography-style focal length (mm) and sensor (mm)
	// focalLength and sensorHeight provide an alternative way to specify FOV.
	//   FOV = 2 * arctan(sensorHeight / (2 * focalLength))
	// When focalLength > 0 it overrides fovY; otherwise fovY is used.
	oa::F32 focalLength  = 0.0f;   // 0 = disabled, use fovY instead
	oa::F32 sensorHeight = 36.0f;  // Default: full-frame 35mm sensor height

	// Orthographic params
	oa::F32 orthoWidth  = 1.0f;
	oa::F32 orthoHeight = 1.0f;
	oa::F32 zoom        = 1.0f;

	// Screen offset (lens shift / 2D pan in normalized device coords)
	// Applied to projection matrix for panning without moving the camera.
	oa::F32 offsetX = 0.0f;
	oa::F32 offsetY = 0.0f;

	// Orbit target (for orbit mode; when non-zero, camera rotates around this point)
	oa::vlm::Vec3 orbitTarget = {0.0f, 0.0f, 0.0f};
	bool useOrbit = false;
};

namespace FnCamera {

// ─── Factory / init ────────────────────────────────────────────────────────

// initialize as 3D perspective camera (convenience for default setup)
void initPerspective(
	oa::CameraState& inState,
	const oa::vlm::Vec3& inPosition = {0.0f, 2.0f, 5.0f},
	const oa::vlm::Vec3& inTarget   = {0.0f, 0.0f, 0.0f},
	oa::F32          inFovY         = 60.0f,
	oa::F32          inAspect       = 1280.0f / 720.0f,
	oa::F32          inNear         = 0.1f,
	oa::F32          inFar          = 100.0f
);

// initialize as 2D orthographic camera (convenience for image/video viewing)
void initOrthographic(
	oa::CameraState& inState,
	oa::F32          inWidth        = 1.0f,
	oa::F32          inHeight       = 1.0f,
	oa::F32          inNear         = 0.0f,
	oa::F32          inFar          = 1.0f
);

// ─── Transform ─────────────────────────────────────────────────────────────

void setPosition(oa::CameraState& inState, const oa::vlm::Vec3& inPosition);
void setRotation(oa::CameraState& inState, const oa::vlm::Quat& inRotation);
void lookAt(
	oa::CameraState& inState,
	const oa::vlm::Vec3& inTarget,
	const oa::vlm::Vec3& inUp = {0.0f, 1.0f, 0.0f}
);

[[nodiscard]] const oa::vlm::Vec3& getPosition(const oa::CameraState& inState) noexcept;
[[nodiscard]] const oa::vlm::Quat& getRotation(const oa::CameraState& inState) noexcept;
[[nodiscard]] oa::vlm::Vec3 getForward(const oa::CameraState& inState) noexcept;
[[nodiscard]] oa::vlm::Vec3 getRight(const oa::CameraState& inState) noexcept;
[[nodiscard]] oa::vlm::Vec3 getUp(const oa::CameraState& inState) noexcept;

// ─── projection ───────────────────────────────────────────────────────────

void setPerspective(
	oa::CameraState& inState,
	oa::F32 inFovY,
	oa::F32 inAspect,
	oa::F32 inNear,
	oa::F32 inFar
);

void setOrthographic(
	oa::CameraState& inState,
	oa::F32 inWidth,
	oa::F32 inHeight,
	oa::F32 inNear = 0.0f,
	oa::F32 inFar  = 1.0f
);

void fitToWindow(oa::CameraState& inState, oa::F32 inWindowWidth, oa::F32 inWindowHeight);
void setAspectRatio(oa::CameraState& inState, oa::F32 inAspect);

[[nodiscard]] oa::CameraProjection getProjectionType(const oa::CameraState& inState) noexcept;

// ─── matrices ─────────────────────────────────────────────────────────────

[[nodiscard]] oa::vlm::Mat4 getViewMatrix(const oa::CameraState& inState) noexcept;
[[nodiscard]] oa::vlm::Mat4 getProjectionMatrix(const oa::CameraState& inState) noexcept;
[[nodiscard]] oa::vlm::Mat4 getViewProjectionMatrix(const oa::CameraState& inState) noexcept;

// ─── Frustum ─────────────────────────────────────────────────────────────

void setNearFar(oa::CameraState& inState, oa::F32 inNear, oa::F32 inFar);
[[nodiscard]] oa::F32 getNear(const oa::CameraState& inState) noexcept;
[[nodiscard]] oa::F32 getFar(const oa::CameraState& inState) noexcept;

// ─── FOV ─────────────────────────────────────────────────────────────────

void setFovY(oa::CameraState& inState, oa::F32 inFovY);
[[nodiscard]] oa::F32 getFovY(const oa::CameraState& inState) noexcept;

// ─── zoom ────────────────────────────────────────────────────────────────

void setZoom(oa::CameraState& inState, oa::F32 inZoom);
[[nodiscard]] oa::F32 getZoom(const oa::CameraState& inState) noexcept;

// ─── Focal length / Sensor ─────────────────────────────────────────────────

void setFocalLength(oa::CameraState& inState, oa::F32 inFocalLengthMm);
[[nodiscard]] oa::F32 getFocalLength(const oa::CameraState& inState) noexcept;
void setSensorHeight(oa::CameraState& inState, oa::F32 inSensorHeightMm);
[[nodiscard]] oa::F32 getSensorHeight(const oa::CameraState& inState) noexcept;

// Effective FOV (respects focal length override)
[[nodiscard]] oa::F32 getEffectiveFovY(const oa::CameraState& inState) noexcept;

// ─── Screen offset ───────────────────────────────────────────────────────

void setOffset(oa::CameraState& inState, oa::F32 inOffsetX, oa::F32 inOffsetY);
[[nodiscard]] oa::vlm::Vec2 getOffset(const oa::CameraState& inState) noexcept;

// ─── Orbit Controls ──────────────────────────────────────────────────────

void setOrbitTarget(oa::CameraState& inState, const oa::vlm::Vec3& inTarget);
void setOrbitDistance(oa::CameraState& inState, oa::F32 inDistance);
void orbitYawPitch(oa::CameraState& inState, oa::F32 inYawDelta, oa::F32 inPitchDelta);
void orbitSetYawPitch(oa::CameraState& inState, oa::F32 inYaw, oa::F32 inPitch);

// ─── Euler Angles ─────────────────────────────────────────────────────────

void setYawPitchRoll(oa::CameraState& inState, oa::F32 inYawDeg, oa::F32 inPitchDeg, oa::F32 inRollDeg);
[[nodiscard]] oa::vlm::Vec3 getYawPitchRoll(const oa::CameraState& inState) noexcept;

// ─── camera-Space movement ───────────────────────────────────────────────

void panLocal(oa::CameraState& inState, oa::F32 inRight, oa::F32 inUp, oa::F32 inForward);

} // namespace FnCamera

} // namespace oa
