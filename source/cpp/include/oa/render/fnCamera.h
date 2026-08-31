// oa::FnCamera — stateless camera-specific operations.

#pragma once

#include <oa/render/camera.h>

namespace oa {

class FnCamera final {
public:
	FnCamera() = delete;

	static void initPerspective(
		oa::Camera& inCamera,
		const oa::vlm::Vec3& inPosition = {0.0F, 2.0F, 5.0F},
		const oa::vlm::Vec3& inTarget = {0.0F, 0.0F, 0.0F},
		oa::F32 inFovY = 60.0F,
		oa::F32 inAspect = 1280.0F / 720.0F,
		oa::F32 inNear = 0.1F,
		oa::F32 inFar = 10000.0F
	);
	static void initOrthographic(
		oa::Camera& inCamera,
		oa::F32 inWidth = 1.0F,
		oa::F32 inHeight = 1.0F,
		oa::F32 inNear = 0.0F,
		oa::F32 inFar = 1.0F
	);

	static void setPerspective(
		oa::Camera& inCamera,
		oa::F32 inFovY,
		oa::F32 inAspect,
		oa::F32 inNear,
		oa::F32 inFar
	);
	static void setOrthographic(
		oa::Camera& inCamera,
		oa::F32 inWidth,
		oa::F32 inHeight,
		oa::F32 inNear = 0.0F,
		oa::F32 inFar = 1.0F
	);
	static void fitToWindow(
		oa::Camera& inCamera,
		oa::F32 inWindowWidth,
		oa::F32 inWindowHeight
	);
	static void setAspectRatio(oa::Camera& inCamera, oa::F32 inAspect);

	[[nodiscard]] static oa::CameraProjection getProjectionType(const oa::Camera& inCamera) noexcept;
	[[nodiscard]] static oa::F32 getAspectRatio(const oa::Camera& inCamera) noexcept;
	[[nodiscard]] static oa::vlm::Vec2 getOrthographicSize(const oa::Camera& inCamera) noexcept;
	[[nodiscard]] static oa::vlm::Mat4 getViewMatrix(const oa::Camera& inCamera) noexcept;
	[[nodiscard]] static oa::vlm::Mat4 getProjectionMatrix(const oa::Camera& inCamera) noexcept;
	[[nodiscard]] static oa::vlm::Mat4 getViewProjectionMatrix(const oa::Camera& inCamera) noexcept;

	static void setNearFar(oa::Camera& inCamera, oa::F32 inNear, oa::F32 inFar);
	[[nodiscard]] static oa::F32 getNear(const oa::Camera& inCamera) noexcept;
	[[nodiscard]] static oa::F32 getFar(const oa::Camera& inCamera) noexcept;
	static void setFovY(oa::Camera& inCamera, oa::F32 inFovY);
	[[nodiscard]] static oa::F32 getFovY(const oa::Camera& inCamera) noexcept;
	static void setZoom(oa::Camera& inCamera, oa::F32 inZoom);
	[[nodiscard]] static oa::F32 getZoom(const oa::Camera& inCamera) noexcept;
	static void setFocalLength(oa::Camera& inCamera, oa::F32 inFocalLengthMm);
	[[nodiscard]] static oa::F32 getFocalLength(const oa::Camera& inCamera) noexcept;
	static void setSensorHeight(oa::Camera& inCamera, oa::F32 inSensorHeightMm);
	[[nodiscard]] static oa::F32 getSensorHeight(const oa::Camera& inCamera) noexcept;
	[[nodiscard]] static oa::F32 getEffectiveFovY(const oa::Camera& inCamera) noexcept;
	static void setOffset(
		oa::Camera& inCamera,
		oa::F32 inOffsetX,
		oa::F32 inOffsetY
	);
	[[nodiscard]] static oa::vlm::Vec2 getOffset(const oa::Camera& inCamera) noexcept;

	static void setOrbitTarget(oa::Camera& inCamera, const oa::vlm::Vec3& inTarget);
	static void setOrbitDistance(oa::Camera& inCamera, oa::F32 inDistance);
	static void orbitYawPitch(
		oa::Camera& inCamera,
		oa::F32 inYawDelta,
		oa::F32 inPitchDelta
	);
	static void orbitSetYawPitch(
		oa::Camera& inCamera,
		oa::F32 inYaw,
		oa::F32 inPitch
	);
};

} // namespace oa
