// oa::Camera — locked semantic camera value.
//
// Camera composes one oa::Transform for placement and owns only camera lens and
// viewport-controller state beyond it. Generic spatial behavior belongs to
// Transform/FnTransform; camera-specific behavior belongs to FnCamera.

#pragma once

#include <oa/core/transform.h>
#include <oa/core/types.h>
#include <oa/core/vlm.h>

namespace oa {

enum class CameraProjection : oa::U8 {
	Perspective,
	Orthographic,
};

class FnCamera;

class Camera {
public:
	// Perspective camera at (0, 0, 5), looking along -Z.
	Camera() = default;
	Camera(
		const oa::vlm::Vec3& inPosition,
		const oa::vlm::Vec3& inTarget = {0.0F, 0.0F, 0.0F},
		oa::F32 inFovY = 60.0F,
		oa::F32 inAspect = 1280.0F / 720.0F,
		oa::F32 inNear = 0.1F,
		oa::F32 inFar = 10000.0F
	);
	// Orthographic camera for image and video viewing.
	Camera(
		oa::F32 inWidth,
		oa::F32 inHeight,
		oa::F32 inNear = 0.0F,
		oa::F32 inFar = 100.0F
	);

	[[nodiscard]] oa::Transform& getTransform() noexcept;
	[[nodiscard]] const oa::Transform& getTransform() const noexcept;

	void setPerspective(
		oa::F32 inFovY,
		oa::F32 inAspect,
		oa::F32 inNear,
		oa::F32 inFar
	);
	void setOrthographic(
		oa::F32 inWidth,
		oa::F32 inHeight,
		oa::F32 inNear = 0.0F,
		oa::F32 inFar = 1.0F
	);
	void fitToWindow(oa::F32 inWindowWidth, oa::F32 inWindowHeight);
	void setAspectRatio(oa::F32 inAspect);

	[[nodiscard]] CameraProjection getProjectionType() const noexcept;
	[[nodiscard]] oa::F32 getAspectRatio() const noexcept;
	[[nodiscard]] oa::vlm::Vec2 getOrthographicSize() const noexcept;

	[[nodiscard]] oa::vlm::Mat4 getViewMatrix() const noexcept;
	[[nodiscard]] oa::vlm::Mat4 getProjectionMatrix() const noexcept;
	[[nodiscard]] oa::vlm::Mat4 getViewProjectionMatrix() const noexcept;

	void setNearFar(oa::F32 inNear, oa::F32 inFar);
	[[nodiscard]] oa::F32 getNear() const noexcept;
	[[nodiscard]] oa::F32 getFar() const noexcept;
	void setFovY(oa::F32 inFovY);
	[[nodiscard]] oa::F32 getFovY() const noexcept;
	void setZoom(oa::F32 inZoom);
	[[nodiscard]] oa::F32 getZoom() const noexcept;
	void setFocalLength(oa::F32 inFocalLengthMm);
	[[nodiscard]] oa::F32 getFocalLength() const noexcept;
	void setSensorHeight(oa::F32 inSensorHeightMm);
	[[nodiscard]] oa::F32 getSensorHeight() const noexcept;
	[[nodiscard]] oa::F32 getEffectiveFovY() const noexcept;
	void setOffset(oa::F32 inOffsetX, oa::F32 inOffsetY);
	[[nodiscard]] oa::vlm::Vec2 getOffset() const noexcept;

	void setOrbitTarget(const oa::vlm::Vec3& inTarget);
	void setOrbitDistance(oa::F32 inDistance);
	void orbitYawPitch(oa::F32 inYawDelta, oa::F32 inPitchDelta);
	void orbitSetYawPitch(oa::F32 inYaw, oa::F32 inPitch);

private:
	friend class FnCamera;

	oa::Transform transform_{{0.0F, 0.0F, 5.0F}};
	CameraProjection projection_ = CameraProjection::Perspective;
	oa::F32 fovY_ = 60.0F;
	oa::F32 aspect_ = 1280.0F / 720.0F;
	oa::F32 near_ = 0.1F;
	oa::F32 far_ = 100.0F;
	oa::F32 focalLength_ = 0.0F;
	oa::F32 sensorHeight_ = 36.0F;
	oa::F32 orthoWidth_ = 1.0F;
	oa::F32 orthoHeight_ = 1.0F;
	oa::F32 zoom_ = 1.0F;
	oa::F32 offsetX_ = 0.0F;
	oa::F32 offsetY_ = 0.0F;
	oa::vlm::Vec3 orbitTarget_ = {0.0F, 0.0F, 0.0F};
};

} // namespace oa
