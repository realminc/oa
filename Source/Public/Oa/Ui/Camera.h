// OaCamera — Class wrapper around OaFnCamera.
//
// OOP wrapper for the functional camera API. All methods delegate to OaFnCamera.
// For the functional namespace, see <Oa/Render/FnCamera.h>.
//
// Usage (3D perspective):
//   OaCamera camera({0.0f, 2.0f, 5.0f});  // pos, target, fov, aspect, near, far
//   VlmMat4 view = camera.GetViewMatrix();
//
// Usage (2D orthographic for image viewing):
//   OaCamera camera(imageWidth, imageHeight);  // width, height, near, far
//   camera.FitToWindow(windowWidth, windowHeight);

#pragma once

#include <Oa/Ui/FnCamera.h>

// ─── OaCamera (Class Wrapper) ─────────────────────────────────────────────────

class OaCamera {
public:
	// Default constructor — perspective camera at (0,0,5) looking at origin
	OaCamera() = default;

	// 3D perspective camera constructor
	explicit OaCamera(
		const VlmVec3& InPosition,
		const VlmVec3& InTarget     = {0.0f, 0.0f, 0.0f},
		OaF32         InFovY       = 60.0f,
		OaF32         InAspect     = 1280.0f / 720.0f,
		OaF32         InNear      = 0.1f,
		OaF32         InFar       = 10000.0f
	) {
		OaFnCamera::InitPerspective(State_, InPosition, InTarget, InFovY, InAspect, InNear, InFar);
	}

	// 2D orthographic camera constructor (for image/video viewing)
	explicit OaCamera(
		OaF32 InWidth,
		OaF32 InHeight,
		OaF32 InNear = 0.0f,
		OaF32 InFar  = 100.0f
	) {
		OaFnCamera::InitOrthographic(State_, InWidth, InHeight, InNear, InFar);
	}

	~OaCamera() = default;

	// Transform
	void SetPosition(const VlmVec3& InPosition) { OaFnCamera::SetPosition(State_, InPosition); }
	void SetRotation(const VlmQuat& InRotation) { OaFnCamera::SetRotation(State_, InRotation); }
	void LookAt(const VlmVec3& InTarget, const VlmVec3& InUp = {0.0f, 1.0f, 0.0f}) {
		OaFnCamera::LookAt(State_, InTarget, InUp);
	}

	[[nodiscard]] const VlmVec3& GetPosition() const noexcept { return OaFnCamera::GetPosition(State_); }
	[[nodiscard]] const VlmQuat& GetRotation() const noexcept { return OaFnCamera::GetRotation(State_); }
	[[nodiscard]] VlmVec3 GetForward() const noexcept { return OaFnCamera::GetForward(State_); }
	[[nodiscard]] VlmVec3 GetRight() const noexcept { return OaFnCamera::GetRight(State_); }
	[[nodiscard]] VlmVec3 GetUp() const noexcept { return OaFnCamera::GetUp(State_); }

	// Projection (3D perspective)
	void SetPerspective(
		OaF32 InFovY,
		OaF32 InAspect,
		OaF32 InNear,
		OaF32 InFar
	) {
		OaFnCamera::SetPerspective(State_, InFovY, InAspect, InNear, InFar);
	}

	// Projection (2D orthographic)
	void SetOrthographic(
		OaF32 InWidth,
		OaF32 InHeight,
		OaF32 InNear = 0.0f,
		OaF32 InFar  = 1.0f
	) {
		OaFnCamera::SetOrthographic(State_, InWidth, InHeight, InNear, InFar);
	}

	// Viewport aspect ratio helper
	void FitToWindow(OaF32 InWindowWidth, OaF32 InWindowHeight) {
		OaFnCamera::FitToWindow(State_, InWindowWidth, InWindowHeight);
	}
	void SetAspectRatio(OaF32 InAspect) { OaFnCamera::SetAspectRatio(State_, InAspect); }

	[[nodiscard]] OaCameraProjection GetProjectionType() const noexcept {
		return OaFnCamera::GetProjectionType(State_);
	}

	// Matrices
	[[nodiscard]] VlmMat4 GetViewMatrix() const noexcept { return OaFnCamera::GetViewMatrix(State_); }
	[[nodiscard]] VlmMat4 GetProjectionMatrix() const noexcept { return OaFnCamera::GetProjectionMatrix(State_); }
	[[nodiscard]] VlmMat4 GetViewProjectionMatrix() const noexcept {
		return OaFnCamera::GetViewProjectionMatrix(State_);
	}

	// Frustum
	void SetNearFar(OaF32 InNear, OaF32 InFar) { OaFnCamera::SetNearFar(State_, InNear, InFar); }
	[[nodiscard]] OaF32 GetNear() const noexcept { return OaFnCamera::GetNear(State_); }
	[[nodiscard]] OaF32 GetFar() const noexcept { return OaFnCamera::GetFar(State_); }

	// FOV (perspective only)
	void SetFovY(OaF32 InFovY) { OaFnCamera::SetFovY(State_, InFovY); }
	[[nodiscard]] OaF32 GetFovY() const noexcept { return OaFnCamera::GetFovY(State_); }

	// Zoom (orthographic only)
	void SetZoom(OaF32 InZoom) { OaFnCamera::SetZoom(State_, InZoom); }
	[[nodiscard]] OaF32 GetZoom() const noexcept { return OaFnCamera::GetZoom(State_); }

	// Focal length / sensor (photography-style perspective)
	void SetFocalLength(OaF32 InFocalLengthMm) { OaFnCamera::SetFocalLength(State_, InFocalLengthMm); }
	[[nodiscard]] OaF32 GetFocalLength() const noexcept { return OaFnCamera::GetFocalLength(State_); }
	void SetSensorHeight(OaF32 InSensorHeightMm) { OaFnCamera::SetSensorHeight(State_, InSensorHeightMm); }
	[[nodiscard]] OaF32 GetSensorHeight() const noexcept { return OaFnCamera::GetSensorHeight(State_); }

	// Effective FOV
	[[nodiscard]] OaF32 GetEffectiveFovY() const noexcept { return OaFnCamera::GetEffectiveFovY(State_); }

	// Screen offset (lens shift / 2D pan)
	void SetOffset(OaF32 InOffsetX, OaF32 InOffsetY) { OaFnCamera::SetOffset(State_, InOffsetX, InOffsetY); }
	[[nodiscard]] VlmVec2 GetOffset() const noexcept { return OaFnCamera::GetOffset(State_); }

	// Orbit controls (rotate around target)
	void SetOrbitTarget(const VlmVec3& InTarget) { OaFnCamera::SetOrbitTarget(State_, InTarget); }
	void SetOrbitDistance(OaF32 InDistance) { OaFnCamera::SetOrbitDistance(State_, InDistance); }
	void OrbitYawPitch(OaF32 InYawDelta, OaF32 InPitchDelta) { OaFnCamera::OrbitYawPitch(State_, InYawDelta, InPitchDelta); }
	void OrbitSetYawPitch(OaF32 InYaw, OaF32 InPitch) { OaFnCamera::OrbitSetYawPitch(State_, InYaw, InPitch); }

	// Euler angles (yaw/pitch/roll in degrees)
	void SetYawPitchRoll(OaF32 InYawDeg, OaF32 InPitchDeg, OaF32 InRollDeg) {
		OaFnCamera::SetYawPitchRoll(State_, InYawDeg, InPitchDeg, InRollDeg);
	}
	[[nodiscard]] VlmVec3 GetYawPitchRoll() const noexcept { return OaFnCamera::GetYawPitchRoll(State_); }

	// Camera-space movement (pan in camera local axes)
	void PanLocal(OaF32 InRight, OaF32 InUp, OaF32 InForward) {
		OaFnCamera::PanLocal(State_, InRight, InUp, InForward);
	}

	// Direct state access (for advanced use)
	[[nodiscard]] OaCameraState& GetState() noexcept { return State_; }
	[[nodiscard]] const OaCameraState& GetState() const noexcept { return State_; }

private:
	OaCameraState State_;
};
