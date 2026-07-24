// OaCamera implementation

#include <Oa/Render/Camera.h>
#include <Oa/Core/Vlm.h>
#include <cmath>

namespace OaFnCamera {

// ─── Factory / Init ──────────────────────────────────────────────────────

void InitPerspective(
	OaCameraState& InState,
	const VlmVec3&  InPosition,
	const VlmVec3&  InTarget,
	OaF32          InFovY,
	OaF32          InAspect,
	OaF32          InNear,
	OaF32          InFar
) {
	InState.Position   = InPosition;
	InState.Rotation   = {0.0f, 0.0f, 0.0f, 1.0f};
	InState.Projection = OaCameraProjection::Perspective;
	InState.FovY       = InFovY;
	InState.Aspect     = InAspect;
	InState.Near       = InNear;
	InState.Far        = InFar;
	InState.FocalLength = 0.0f;
	InState.OffsetX    = 0.0f;
	InState.OffsetY    = 0.0f;
	InState.UseOrbit   = false;
	LookAt(InState, InTarget);
}

void InitOrthographic(
	OaCameraState& InState,
	OaF32          InWidth,
	OaF32          InHeight,
	OaF32          InNear,
	OaF32          InFar
) {
	InState.Position    = {0.0f, 0.0f, 1.0f};
	InState.Rotation    = {0.0f, 0.0f, 0.0f, 1.0f};
	InState.Projection  = OaCameraProjection::Orthographic;
	InState.OrthoWidth  = InWidth;
	InState.OrthoHeight = InHeight;
	InState.Near        = InNear;
	InState.Far         = InFar;
	InState.Zoom        = 1.0f;
	InState.OffsetX     = 0.0f;
	InState.OffsetY     = 0.0f;
	InState.UseOrbit    = false;
}

// ─── Transform ─────────────────────────────────────────────────────────────

void SetPosition(OaCameraState& InState, const VlmVec3& InPosition) {
	InState.Position = InPosition;
}

void SetRotation(OaCameraState& InState, const VlmQuat& InRotation) {
	InState.Rotation = InRotation;
}

void LookAt(OaCameraState& InState, const VlmVec3& InTarget, const VlmVec3& InUp) {
	// Use shared LookAt to build view matrix, then extract rotation quaternion
	VlmMat4 view = Vlm::LookAt(InState.Position, InTarget, InUp);

	// Extract rotation quaternion from the upper 3x3 of the view matrix
	// (view matrix is the inverse of camera transform; we need the inverse of that)
	OaF32 m00 = view.M[0][0], m01 = view.M[0][1], m02 = view.M[0][2];
	OaF32 m10 = view.M[1][0], m11 = view.M[1][1], m12 = view.M[1][2];
	OaF32 m20 = view.M[2][0], m21 = view.M[2][1], m22 = view.M[2][2];

	// View matrix is R^T (transpose of rotation), so camera rotation is R.
	// Extract quaternion from rotation matrix (right-handed, Y-up).
	OaF32 tr = m00 + m11 + m22;
	if (tr > 0.0f) {
		OaF32 s = std::sqrt(tr + 1.0f) * 2.0f;
		InState.Rotation.W = 0.25f * s;
		InState.Rotation.X = (m21 - m12) / s;
		InState.Rotation.Y = (m02 - m20) / s;
		InState.Rotation.Z = (m10 - m01) / s;
	} else if (m00 > m11 && m00 > m22) {
		OaF32 s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
		InState.Rotation.W = (m21 - m12) / s;
		InState.Rotation.X = 0.25f * s;
		InState.Rotation.Y = (m01 + m10) / s;
		InState.Rotation.Z = (m02 + m20) / s;
	} else if (m11 > m22) {
		OaF32 s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
		InState.Rotation.W = (m02 - m20) / s;
		InState.Rotation.X = (m01 + m10) / s;
		InState.Rotation.Y = 0.25f * s;
		InState.Rotation.Z = (m12 + m21) / s;
	} else {
		OaF32 s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
		InState.Rotation.W = (m10 - m01) / s;
		InState.Rotation.X = (m02 + m20) / s;
		InState.Rotation.Y = (m12 + m21) / s;
		InState.Rotation.Z = 0.25f * s;
	}
}

const VlmVec3& GetPosition(const OaCameraState& InState) noexcept {
	return InState.Position;
}

const VlmQuat& GetRotation(const OaCameraState& InState) noexcept {
	return InState.Rotation;
}

VlmVec3 GetForward(const OaCameraState& InState) noexcept {
	return Vlm::RotateVector(InState.Rotation, {0.0f, 0.0f, -1.0f});
}

VlmVec3 GetRight(const OaCameraState& InState) noexcept {
	return Vlm::RotateVector(InState.Rotation, {1.0f, 0.0f, 0.0f});
}

VlmVec3 GetUp(const OaCameraState& InState) noexcept {
	return Vlm::RotateVector(InState.Rotation, {0.0f, 1.0f, 0.0f});
}

// ─── Projection ───────────────────────────────────────────────────────────

void SetPerspective(
	OaCameraState& InState,
	OaF32 InFovY,
	OaF32 InAspect,
	OaF32 InNear,
	OaF32 InFar
) {
	InState.Projection = OaCameraProjection::Perspective;
	InState.FovY = InFovY;
	InState.Aspect = InAspect;
	InState.Near = InNear;
	InState.Far = InFar;
}

void SetOrthographic(
	OaCameraState& InState,
	OaF32 InWidth,
	OaF32 InHeight,
	OaF32 InNear,
	OaF32 InFar
) {
	InState.Projection = OaCameraProjection::Orthographic;
	InState.OrthoWidth = InWidth;
	InState.OrthoHeight = InHeight;
	InState.Near = InNear;
	InState.Far = InFar;
}

void FitToWindow(OaCameraState& InState, OaF32 InWindowWidth, OaF32 InWindowHeight) {
	if (InState.Projection == OaCameraProjection::Orthographic) {
		// Fit orthographic view to window while maintaining aspect ratio
		OaF32 windowAspect = InWindowWidth / InWindowHeight;
		OaF32 contentAspect = InState.OrthoWidth / InState.OrthoHeight;

		if (windowAspect > contentAspect) {
			// Window is wider - fit height
			InState.OrthoWidth = InState.OrthoHeight * windowAspect;
		} else {
			// Window is taller - fit width
			InState.OrthoHeight = InState.OrthoWidth / windowAspect;
		}
	} else {
		// Update aspect ratio for perspective
		InState.Aspect = InWindowWidth / InWindowHeight;
	}
}

void SetAspectRatio(OaCameraState& InState, OaF32 InAspect) {
	if (InState.Projection == OaCameraProjection::Perspective) {
		InState.Aspect = InAspect;
	}
}

OaCameraProjection GetProjectionType(const OaCameraState& InState) noexcept {
	return InState.Projection;
}

// ─── Matrices ─────────────────────────────────────────────────────────────

VlmMat4 GetViewMatrix(const OaCameraState& InState) noexcept {
	// Camera rotation is the world-space camera basis. Row-vector view space
	// therefore uses that basis directly, with -position * rotation in the
	// translation row. Transposing the basis here produces a mixed-convention
	// matrix whose translation belongs to a different camera.
	VlmMat4 rot = Vlm::QuaternionToMatrix(InState.Rotation);
	VlmMat4 view = rot;
	view.M[3][0] = -(
		InState.Position.X * rot.M[0][0]
		+ InState.Position.Y * rot.M[1][0]
		+ InState.Position.Z * rot.M[2][0]);
	view.M[3][1] = -(
		InState.Position.X * rot.M[0][1]
		+ InState.Position.Y * rot.M[1][1]
		+ InState.Position.Z * rot.M[2][1]);
	view.M[3][2] = -(
		InState.Position.X * rot.M[0][2]
		+ InState.Position.Y * rot.M[1][2]
		+ InState.Position.Z * rot.M[2][2]);

	return view;
}

VlmMat4 GetProjectionMatrix(const OaCameraState& InState) noexcept {
	VlmMat4 proj = {};

	if (InState.Projection == OaCameraProjection::Perspective) {
		OaF32 fovY = GetEffectiveFovY(InState);
		proj = Vlm::Perspective(fovY, InState.Aspect, InState.Near, InState.Far);
		// Apply screen offset (lens shift)
		proj.M[2][0] = -InState.OffsetX;
		proj.M[2][1] = -InState.OffsetY;
	} else {
		proj = Vlm::Orthographic(InState.OrthoWidth, InState.OrthoHeight, InState.Near, InState.Far, InState.Zoom);
		// Apply screen offset (2D pan)
		proj.M[3][0] = -InState.OffsetX;
		proj.M[3][1] = -InState.OffsetY;
	}

	return proj;
}

VlmMat4 GetViewProjectionMatrix(const OaCameraState& InState) noexcept {
	// OA shaders multiply row vectors: mul(position, matrix).
	return Vlm::MatrixMul(GetViewMatrix(InState), GetProjectionMatrix(InState));
}

// ─── Frustum ─────────────────────────────────────────────────────────────

void SetNearFar(OaCameraState& InState, OaF32 InNear, OaF32 InFar) {
	InState.Near = InNear;
	InState.Far = InFar;
}

OaF32 GetNear(const OaCameraState& InState) noexcept {
	return InState.Near;
}

OaF32 GetFar(const OaCameraState& InState) noexcept {
	return InState.Far;
}

// ─── FOV ─────────────────────────────────────────────────────────────────

void SetFovY(OaCameraState& InState, OaF32 InFovY) {
	InState.FovY = InFovY;
}

OaF32 GetFovY(const OaCameraState& InState) noexcept {
	return InState.FovY;
}

// ─── Zoom ────────────────────────────────────────────────────────────────

void SetZoom(OaCameraState& InState, OaF32 InZoom) {
	InState.Zoom = InZoom;
}

OaF32 GetZoom(const OaCameraState& InState) noexcept {
	return InState.Zoom;
}

// ─── Focal Length / Sensor ─────────────────────────────────────────────────

void SetFocalLength(OaCameraState& InState, OaF32 InFocalLengthMm) {
	InState.FocalLength = InFocalLengthMm;
}

OaF32 GetFocalLength(const OaCameraState& InState) noexcept {
	return InState.FocalLength;
}

void SetSensorHeight(OaCameraState& InState, OaF32 InSensorHeightMm) {
	InState.SensorHeight = InSensorHeightMm;
}

OaF32 GetSensorHeight(const OaCameraState& InState) noexcept {
	return InState.SensorHeight;
}

OaF32 GetEffectiveFovY(const OaCameraState& InState) noexcept {
	if (InState.FocalLength > 0.0f && InState.SensorHeight > 0.0f) {
		// FOV = 2 * arctan(sensorHeight / (2 * focalLength))
		return 2.0f * std::atan(InState.SensorHeight / (2.0f * InState.FocalLength)) * 180.0f / 3.14159265358979323846f;
	}
	return InState.FovY;
}

// ─── Screen Offset ───────────────────────────────────────────────────────

void SetOffset(OaCameraState& InState, OaF32 InOffsetX, OaF32 InOffsetY) {
	InState.OffsetX = InOffsetX;
	InState.OffsetY = InOffsetY;
}

VlmVec2 GetOffset(const OaCameraState& InState) noexcept {
	return {InState.OffsetX, InState.OffsetY};
}

// ─── Orbit Controls ──────────────────────────────────────────────────────

void SetOrbitTarget(OaCameraState& InState, const VlmVec3& InTarget) {
	InState.OrbitTarget = InTarget;
	InState.UseOrbit = true;
}

void SetOrbitDistance(OaCameraState& InState, OaF32 InDistance) {
	// Reposition camera along the forward vector from target
	VlmVec3 forward = GetForward(InState);
	InState.Position.X = InState.OrbitTarget.X - forward.X * InDistance;
	InState.Position.Y = InState.OrbitTarget.Y - forward.Y * InDistance;
	InState.Position.Z = InState.OrbitTarget.Z - forward.Z * InDistance;
}

void OrbitYawPitch(OaCameraState& InState, OaF32 InYawDelta, OaF32 InPitchDelta) {
	VlmVec3 offset = Vlm::Sub(InState.Position, InState.OrbitTarget);
	VlmVec3 sph = Vlm::CartesianToSpherical(offset);
	if (sph.Z < Vlm::EPSILON) return;

	sph.X += InYawDelta * Vlm::PI / 180.0f;     // yaw
	sph.Y += InPitchDelta * Vlm::PI / 180.0f;  // pitch

	// Clamp pitch
	OaF32 pitchLimit = 89.0f * Vlm::PI / 180.0f;
	if (sph.Y > pitchLimit) sph.Y = pitchLimit;
	if (sph.Y < -pitchLimit) sph.Y = -pitchLimit;

	InState.Position = Vlm::Add(InState.OrbitTarget, Vlm::SphericalToCartesian(sph.X, sph.Y, sph.Z));
	LookAt(InState, InState.OrbitTarget);
}

void OrbitSetYawPitch(OaCameraState& InState, OaF32 InYaw, OaF32 InPitch) {
	VlmVec3 offset = Vlm::Sub(InState.Position, InState.OrbitTarget);
	VlmVec3 sph = Vlm::CartesianToSpherical(offset);
	if (sph.Z < Vlm::EPSILON) return;

	InState.Position = Vlm::Add(
		InState.OrbitTarget,
		Vlm::SphericalToCartesian(InYaw * Vlm::PI / 180.0f, InPitch * Vlm::PI / 180.0f, sph.Z)
	);
	LookAt(InState, InState.OrbitTarget);
}

// ─── Euler Angles ─────────────────────────────────────────────────────────

void SetYawPitchRoll(OaCameraState& InState, OaF32 InYawDeg, OaF32 InPitchDeg, OaF32 InRollDeg) {
	InState.Rotation = Vlm::QuaternionFromEuler(InYawDeg, InPitchDeg, InRollDeg);
}

VlmVec3 GetYawPitchRoll(const OaCameraState& InState) noexcept {
	return Vlm::QuaternionToEuler(InState.Rotation);
}

// ─── Camera-Space Movement ───────────────────────────────────────────────

void PanLocal(OaCameraState& InState, OaF32 InRight, OaF32 InUp, OaF32 InForward) {
	VlmVec3 delta = Vlm::Add(
		Vlm::Add(
			Vlm::Scale(GetRight(InState), InRight),
			Vlm::Scale(GetUp(InState), InUp)
		),
		Vlm::Scale(GetForward(InState), InForward)
	);
	InState.Position = Vlm::Add(InState.Position, delta);
}

} // namespace OaFnCamera
