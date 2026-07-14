// OaFnCamera — Functional camera operations.
//
// Stateless camera math. Operates on OaCameraState POD.
// For the OOP wrapper, see <Oa/Render/Camera.h>.
//
// Usage:
//   OaCameraState state;
//   OaFnCamera::SetPosition(state, {0.0f, 2.0f, 5.0f});
//   OaFnCamera::LookAt(state, {0.0f, 0.0f, 0.0f});
//   OaFnCamera::SetPerspective(state, 60.0f, 16.0f / 9.0f, 0.1f, 100.0f);
//   VlmMat4 view = OaFnCamera::GetViewMatrix(state);

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Vlm.h>

// Camera projection type
enum class OaCameraProjection : OaU8 {
	Perspective,   // 3D perspective projection
	Orthographic,  // 2D orthographic projection
};

// Camera state (POD for functional API)
struct OaCameraState {
	VlmVec3            Position   = {0.0f, 0.0f, 5.0f};
	VlmQuat            Rotation   = {0.0f, 0.0f, 0.0f, 1.0f};  // Identity
	OaCameraProjection Projection = OaCameraProjection::Perspective;

	// Perspective params
	OaF32 FovY   = 60.0f;
	OaF32 Aspect = 1280.0f / 720.0f;
	OaF32 Near   = 0.1f;
	OaF32 Far    = 100.0f;

	// Photography-style focal length (mm) and sensor (mm)
	// FocalLength and SensorHeight provide an alternative way to specify FOV.
	//   FOV = 2 * arctan(sensorHeight / (2 * focalLength))
	// When FocalLength > 0 it overrides FovY; otherwise FovY is used.
	OaF32 FocalLength  = 0.0f;   // 0 = disabled, use FovY instead
	OaF32 SensorHeight = 36.0f;  // Default: full-frame 35mm sensor height

	// Orthographic params
	OaF32 OrthoWidth  = 1.0f;
	OaF32 OrthoHeight = 1.0f;
	OaF32 Zoom         = 1.0f;

	// Screen offset (lens shift / 2D pan in normalized device coords)
	// Applied to projection matrix for panning without moving the camera.
	OaF32 OffsetX = 0.0f;
	OaF32 OffsetY = 0.0f;

	// Orbit target (for orbit mode; when non-zero, camera rotates around this point)
	VlmVec3 OrbitTarget = {0.0f, 0.0f, 0.0f};
	bool   UseOrbit     = false;
};

namespace OaFnCamera {

// ─── Factory / Init ────────────────────────────────────────────────────────

// Initialize as 3D perspective camera (convenience for default setup)
void InitPerspective(
	OaCameraState& InState,
	const VlmVec3&  InPosition    = {0.0f, 2.0f, 5.0f},
	const VlmVec3&  InTarget       = {0.0f, 0.0f, 0.0f},
	OaF32          InFovY         = 60.0f,
	OaF32          InAspect       = 1280.0f / 720.0f,
	OaF32          InNear         = 0.1f,
	OaF32          InFar          = 100.0f
);

// Initialize as 2D orthographic camera (convenience for image/video viewing)
void InitOrthographic(
	OaCameraState& InState,
	OaF32          InWidth        = 1.0f,
	OaF32          InHeight       = 1.0f,
	OaF32          InNear         = 0.0f,
	OaF32          InFar          = 1.0f
);

// ─── Transform ─────────────────────────────────────────────────────────────

void SetPosition(OaCameraState& InState, const VlmVec3& InPosition);
void SetRotation(OaCameraState& InState, const VlmQuat& InRotation);
void LookAt(OaCameraState& InState, const VlmVec3& InTarget, const VlmVec3& InUp = {0.0f, 1.0f, 0.0f});

[[nodiscard]] const VlmVec3& GetPosition(const OaCameraState& InState) noexcept;
[[nodiscard]] const VlmQuat& GetRotation(const OaCameraState& InState) noexcept;
[[nodiscard]] VlmVec3 GetForward(const OaCameraState& InState) noexcept;
[[nodiscard]] VlmVec3 GetRight(const OaCameraState& InState) noexcept;
[[nodiscard]] VlmVec3 GetUp(const OaCameraState& InState) noexcept;

// ─── Projection ───────────────────────────────────────────────────────────

void SetPerspective(
	OaCameraState& InState,
	OaF32 InFovY,
	OaF32 InAspect,
	OaF32 InNear,
	OaF32 InFar
);

void SetOrthographic(
	OaCameraState& InState,
	OaF32 InWidth,
	OaF32 InHeight,
	OaF32 InNear = 0.0f,
	OaF32 InFar  = 1.0f
);

void FitToWindow(OaCameraState& InState, OaF32 InWindowWidth, OaF32 InWindowHeight);
void SetAspectRatio(OaCameraState& InState, OaF32 InAspect);

[[nodiscard]] OaCameraProjection GetProjectionType(const OaCameraState& InState) noexcept;

// ─── Matrices ─────────────────────────────────────────────────────────────

[[nodiscard]] VlmMat4 GetViewMatrix(const OaCameraState& InState) noexcept;
[[nodiscard]] VlmMat4 GetProjectionMatrix(const OaCameraState& InState) noexcept;
[[nodiscard]] VlmMat4 GetViewProjectionMatrix(const OaCameraState& InState) noexcept;

// ─── Frustum ─────────────────────────────────────────────────────────────

void SetNearFar(OaCameraState& InState, OaF32 InNear, OaF32 InFar);
[[nodiscard]] OaF32 GetNear(const OaCameraState& InState) noexcept;
[[nodiscard]] OaF32 GetFar(const OaCameraState& InState) noexcept;

// ─── FOV ─────────────────────────────────────────────────────────────────

void SetFovY(OaCameraState& InState, OaF32 InFovY);
[[nodiscard]] OaF32 GetFovY(const OaCameraState& InState) noexcept;

// ─── Zoom ────────────────────────────────────────────────────────────────

void SetZoom(OaCameraState& InState, OaF32 InZoom);
[[nodiscard]] OaF32 GetZoom(const OaCameraState& InState) noexcept;

// ─── Focal Length / Sensor ─────────────────────────────────────────────────

void SetFocalLength(OaCameraState& InState, OaF32 InFocalLengthMm);
[[nodiscard]] OaF32 GetFocalLength(const OaCameraState& InState) noexcept;
void SetSensorHeight(OaCameraState& InState, OaF32 InSensorHeightMm);
[[nodiscard]] OaF32 GetSensorHeight(const OaCameraState& InState) noexcept;

// Effective FOV (respects focal length override)
[[nodiscard]] OaF32 GetEffectiveFovY(const OaCameraState& InState) noexcept;

// ─── Screen Offset ───────────────────────────────────────────────────────

void SetOffset(OaCameraState& InState, OaF32 InOffsetX, OaF32 InOffsetY);
[[nodiscard]] VlmVec2 GetOffset(const OaCameraState& InState) noexcept;

// ─── Orbit Controls ──────────────────────────────────────────────────────

void SetOrbitTarget(OaCameraState& InState, const VlmVec3& InTarget);
void SetOrbitDistance(OaCameraState& InState, OaF32 InDistance);
void OrbitYawPitch(OaCameraState& InState, OaF32 InYawDelta, OaF32 InPitchDelta);
void OrbitSetYawPitch(OaCameraState& InState, OaF32 InYaw, OaF32 InPitch);

// ─── Euler Angles ─────────────────────────────────────────────────────────

void SetYawPitchRoll(OaCameraState& InState, OaF32 InYawDeg, OaF32 InPitchDeg, OaF32 InRollDeg);
[[nodiscard]] VlmVec3 GetYawPitchRoll(const OaCameraState& InState) noexcept;

// ─── Camera-Space Movement ───────────────────────────────────────────────

void PanLocal(OaCameraState& InState, OaF32 InRight, OaF32 InUp, OaF32 InForward);

} // namespace OaFnCamera
