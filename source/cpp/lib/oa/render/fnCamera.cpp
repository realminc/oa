#include <oa/render/fnCamera.h>

#include <oa/core/fnTransform.h>
#include <oa/core/std/assert.h>
#include <oa/core/std/scalarMath.h>
#include <oa/core/vlm.h>

namespace oa {

void FnCamera::initPerspective(
	oa::Camera& inCamera,
	const oa::vlm::Vec3& inPosition,
	const oa::vlm::Vec3& inTarget,
	oa::F32 inFovY,
	oa::F32 inAspect,
	oa::F32 inNear,
	oa::F32 inFar) {
	inCamera = oa::Camera{};
	oa::FnTransform::setPosition(inCamera.transform_, inPosition);
	oa::FnTransform::lookAt(inCamera.transform_, inTarget);
	setPerspective(inCamera, inFovY, inAspect, inNear, inFar);
}

void FnCamera::initOrthographic(
	oa::Camera& inCamera,
	oa::F32 inWidth,
	oa::F32 inHeight,
	oa::F32 inNear,
	oa::F32 inFar) {
	inCamera = oa::Camera{};
	oa::FnTransform::setPosition(
		inCamera.transform_, {0.0F, 0.0F, 1.0F});
	setOrthographic(inCamera, inWidth, inHeight, inNear, inFar);
}

void FnCamera::setPerspective(
	oa::Camera& inCamera,
	oa::F32 inFovY,
	oa::F32 inAspect,
	oa::F32 inNear,
	oa::F32 inFar) {
	OA_REQUIRE_MSG(
		oa::isFinite(inFovY) and inFovY > 0.0F and inFovY < 180.0F,
		"Camera perspective field of view must be finite and inside (0, 180)");
	OA_REQUIRE_MSG(
		oa::isFinite(inAspect) and inAspect > 0.0F,
		"Camera perspective aspect ratio must be finite and positive");
	OA_REQUIRE_MSG(
		oa::isFinite(inNear) and oa::isFinite(inFar)
			and inNear > 0.0F and inFar > inNear,
		"Camera perspective near/far planes are invalid");
	inCamera.projection_ = oa::CameraProjection::Perspective;
	inCamera.fovY_ = inFovY;
	inCamera.aspect_ = inAspect;
	inCamera.near_ = inNear;
	inCamera.far_ = inFar;
}

void FnCamera::setOrthographic(
	oa::Camera& inCamera,
	oa::F32 inWidth,
	oa::F32 inHeight,
	oa::F32 inNear,
	oa::F32 inFar) {
	OA_REQUIRE_MSG(
		oa::isFinite(inWidth) and oa::isFinite(inHeight)
			and inWidth > 0.0F and inHeight > 0.0F,
		"Camera orthographic extent must be finite and positive");
	OA_REQUIRE_MSG(
		oa::isFinite(inNear) and oa::isFinite(inFar) and inFar > inNear,
		"Camera orthographic near/far planes are invalid");
	inCamera.projection_ = oa::CameraProjection::Orthographic;
	inCamera.orthoWidth_ = inWidth;
	inCamera.orthoHeight_ = inHeight;
	inCamera.near_ = inNear;
	inCamera.far_ = inFar;
}

void FnCamera::fitToWindow(
	oa::Camera& inCamera,
	oa::F32 inWindowWidth,
	oa::F32 inWindowHeight) {
	OA_REQUIRE_MSG(
		oa::isFinite(inWindowWidth) and oa::isFinite(inWindowHeight)
			and inWindowWidth > 0.0F and inWindowHeight > 0.0F,
		"Camera window extent must be finite and positive");
	if (inCamera.projection_ == oa::CameraProjection::Orthographic) {
		const oa::F32 windowAspect = inWindowWidth / inWindowHeight;
		const oa::F32 contentAspect =
			inCamera.orthoWidth_ / inCamera.orthoHeight_;
		if (windowAspect > contentAspect) {
			inCamera.orthoWidth_ = inCamera.orthoHeight_ * windowAspect;
		} else {
			inCamera.orthoHeight_ = inCamera.orthoWidth_ / windowAspect;
		}
		return;
	}
	inCamera.aspect_ = inWindowWidth / inWindowHeight;
}

void FnCamera::setAspectRatio(oa::Camera& inCamera, oa::F32 inAspect) {
	OA_REQUIRE_MSG(
		oa::isFinite(inAspect) and inAspect > 0.0F,
		"Camera aspect ratio must be finite and positive");
	if (inCamera.projection_ == oa::CameraProjection::Perspective) {
		inCamera.aspect_ = inAspect;
	}
}

oa::CameraProjection FnCamera::getProjectionType(
	const oa::Camera& inCamera) noexcept {
	return inCamera.projection_;
}

oa::F32 FnCamera::getAspectRatio(const oa::Camera& inCamera) noexcept {
	return inCamera.aspect_;
}

oa::vlm::Vec2 FnCamera::getOrthographicSize(
	const oa::Camera& inCamera) noexcept {
	return {inCamera.orthoWidth_, inCamera.orthoHeight_};
}

oa::vlm::Mat4 FnCamera::getViewMatrix(const oa::Camera& inCamera) noexcept {
	return oa::vlm::viewFromPose(
		oa::FnTransform::getPosition(inCamera.transform_),
		oa::FnTransform::getRotation(inCamera.transform_));
}

oa::vlm::Mat4 FnCamera::getProjectionMatrix(
	const oa::Camera& inCamera) noexcept {
	if (inCamera.projection_ == oa::CameraProjection::Perspective) {
		return oa::vlm::perspectiveShifted(
			getEffectiveFovY(inCamera),
			inCamera.aspect_,
			inCamera.near_,
			inCamera.far_,
			{inCamera.offsetX_, inCamera.offsetY_});
	}
	return oa::vlm::orthographicShifted(
		inCamera.orthoWidth_,
		inCamera.orthoHeight_,
		inCamera.near_,
		inCamera.far_,
		inCamera.zoom_,
		{inCamera.offsetX_, inCamera.offsetY_});
}

oa::vlm::Mat4 FnCamera::getViewProjectionMatrix(
	const oa::Camera& inCamera) noexcept {
	return oa::vlm::matrixMul(
		getViewMatrix(inCamera), getProjectionMatrix(inCamera));
}

void FnCamera::setNearFar(
	oa::Camera& inCamera,
	oa::F32 inNear,
	oa::F32 inFar) {
	if (inCamera.projection_ == oa::CameraProjection::Perspective) {
		setPerspective(
			inCamera, inCamera.fovY_, inCamera.aspect_, inNear, inFar);
		return;
	}
	setOrthographic(
		inCamera,
		inCamera.orthoWidth_,
		inCamera.orthoHeight_,
		inNear,
		inFar);
}

oa::F32 FnCamera::getNear(const oa::Camera& inCamera) noexcept {
	return inCamera.near_;
}

oa::F32 FnCamera::getFar(const oa::Camera& inCamera) noexcept {
	return inCamera.far_;
}

void FnCamera::setFovY(oa::Camera& inCamera, oa::F32 inFovY) {
	setPerspective(
		inCamera, inFovY, inCamera.aspect_, inCamera.near_, inCamera.far_);
}

oa::F32 FnCamera::getFovY(const oa::Camera& inCamera) noexcept {
	return inCamera.fovY_;
}

void FnCamera::setZoom(oa::Camera& inCamera, oa::F32 inZoom) {
	OA_REQUIRE_MSG(
		oa::isFinite(inZoom) and inZoom > 0.0F,
		"Camera zoom must be finite and positive");
	inCamera.zoom_ = inZoom;
}

oa::F32 FnCamera::getZoom(const oa::Camera& inCamera) noexcept {
	return inCamera.zoom_;
}

void FnCamera::setFocalLength(
	oa::Camera& inCamera,
	oa::F32 inFocalLengthMm) {
	OA_REQUIRE_MSG(
		oa::isFinite(inFocalLengthMm) and inFocalLengthMm >= 0.0F,
		"Camera focal length must be finite and non-negative");
	inCamera.focalLength_ = inFocalLengthMm;
}

oa::F32 FnCamera::getFocalLength(const oa::Camera& inCamera) noexcept {
	return inCamera.focalLength_;
}

void FnCamera::setSensorHeight(
	oa::Camera& inCamera,
	oa::F32 inSensorHeightMm) {
	OA_REQUIRE_MSG(
		oa::isFinite(inSensorHeightMm) and inSensorHeightMm > 0.0F,
		"Camera sensor height must be finite and positive");
	inCamera.sensorHeight_ = inSensorHeightMm;
}

oa::F32 FnCamera::getSensorHeight(const oa::Camera& inCamera) noexcept {
	return inCamera.sensorHeight_;
}

oa::F32 FnCamera::getEffectiveFovY(const oa::Camera& inCamera) noexcept {
	const oa::F32 focalLength = inCamera.focalLength_;
	const oa::F32 sensorHeight = inCamera.sensorHeight_;
	if (focalLength > 0.0F and sensorHeight > 0.0F) {
		return 2.0F * oa::atan(sensorHeight / (2.0F * focalLength))
			* 180.0F / oa::vlm::Pi<oa::F32>;
	}
	return inCamera.fovY_;
}

void FnCamera::setOffset(
	oa::Camera& inCamera,
	oa::F32 inOffsetX,
	oa::F32 inOffsetY) {
	OA_REQUIRE_MSG(
		oa::isFinite(inOffsetX) and oa::isFinite(inOffsetY),
		"Camera lens offset must be finite");
	inCamera.offsetX_ = inOffsetX;
	inCamera.offsetY_ = inOffsetY;
}

oa::vlm::Vec2 FnCamera::getOffset(const oa::Camera& inCamera) noexcept {
	return {inCamera.offsetX_, inCamera.offsetY_};
}

void FnCamera::setOrbitTarget(
	oa::Camera& inCamera,
	const oa::vlm::Vec3& inTarget) {
	OA_REQUIRE_MSG(inTarget.isFinite(), "Camera orbit target must be finite");
	inCamera.orbitTarget_ = inTarget;
}

void FnCamera::setOrbitDistance(
	oa::Camera& inCamera,
	oa::F32 inDistance) {
	OA_REQUIRE_MSG(
		oa::isFinite(inDistance) and inDistance > 0.0F,
		"Camera orbit distance must be finite and positive");
	oa::FnTransform::setPosition(
		inCamera.transform_,
		inCamera.orbitTarget_
			- oa::FnTransform::getForward(inCamera.transform_) * inDistance);
}

void FnCamera::orbitYawPitch(
	oa::Camera& inCamera,
	oa::F32 inYawDelta,
	oa::F32 inPitchDelta) {
	OA_REQUIRE_MSG(
		oa::isFinite(inYawDelta) and oa::isFinite(inPitchDelta),
		"Camera orbit deltas must be finite");
	oa::vlm::Vec3 spherical = oa::vlm::cartesianToSpherical(
		oa::FnTransform::getPosition(inCamera.transform_)
			- inCamera.orbitTarget_);
	if (spherical.z < oa::vlm::Tolerance<oa::F32>) return;
	spherical.x += oa::vlm::radians(inYawDelta);
	spherical.y += oa::vlm::radians(inPitchDelta);
	const oa::F32 pitchLimit = oa::vlm::radians(89.0F);
	spherical.y = oa::clamp(spherical.y, -pitchLimit, pitchLimit);
	oa::FnTransform::setPosition(
		inCamera.transform_,
		inCamera.orbitTarget_
			+ oa::vlm::sphericalToCartesian(
				spherical.x, spherical.y, spherical.z));
	oa::FnTransform::lookAt(inCamera.transform_, inCamera.orbitTarget_);
}

void FnCamera::orbitSetYawPitch(
	oa::Camera& inCamera,
	oa::F32 inYaw,
	oa::F32 inPitch) {
	OA_REQUIRE_MSG(
		oa::isFinite(inYaw) and oa::isFinite(inPitch),
		"Camera orbit angles must be finite");
	const oa::vlm::Vec3 spherical = oa::vlm::cartesianToSpherical(
		oa::FnTransform::getPosition(inCamera.transform_)
			- inCamera.orbitTarget_);
	if (spherical.z < oa::vlm::Tolerance<oa::F32>) return;
	oa::FnTransform::setPosition(
		inCamera.transform_,
		inCamera.orbitTarget_
			+ oa::vlm::sphericalToCartesian(
				oa::vlm::radians(inYaw),
				oa::vlm::radians(inPitch),
				spherical.z));
	oa::FnTransform::lookAt(inCamera.transform_, inCamera.orbitTarget_);
}

} // namespace oa
