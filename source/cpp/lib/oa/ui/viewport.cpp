// oa::Viewport — passive validated view description.

#include <oa/ui/viewport.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

[[nodiscard]] bool isKnownMode(oa::ViewportMode inMode) noexcept {
	return static_cast<oa::U8>(inMode)
		<= static_cast<oa::U8>(oa::ViewportMode::CameraStream);
}

[[nodiscard]] bool isFinitePositive(oa::F32 inValue) noexcept {
	return std::isfinite(inValue) and inValue > 0.0F;
}

} // namespace

oa::Status oa::Viewport::setMode(oa::ViewportMode inMode) {
	if (not isKnownMode(inMode)) {
		return oa::Status::invalidArgument("oa::Viewport::setMode received an unknown mode");
	}
	mode_ = inMode;
	return oa::Status::ok();
}

oa::Status oa::Viewport::setViewport(const oa::ViewportDesc& inViewport) {
	if (not inViewport.isValid()) {
		return oa::Status::invalidArgument(
			"oa::Viewport::setViewport requires positive dimensions and a finite [0,1] depth range");
	}
	viewport_ = inViewport;
	return oa::Status::ok();
}

oa::Status oa::Viewport::setScissor(const oa::ScissorDesc& inScissor) {
	if (not inScissor.isValid()) {
		return oa::Status::invalidArgument(
			"oa::Viewport::setScissor requires positive dimensions");
	}
	scissor_ = inScissor;
	useScissor_ = true;
	return oa::Status::ok();
}

oa::Status oa::Viewport::setClearColor(const oa::vlm::Vec4& inColor) {
	if (not std::isfinite(inColor.x) or not std::isfinite(inColor.y)
		or not std::isfinite(inColor.z) or not std::isfinite(inColor.w)) {
		return oa::Status::invalidArgument(
			"oa::Viewport::setClearColor requires finite components");
	}
	clearColor_ = inColor;
	return oa::Status::ok();
}

oa::Status oa::Viewport::setClearDepth(oa::F32 inDepth) {
	if (not std::isfinite(inDepth) or inDepth < 0.0F or inDepth > 1.0F) {
		return oa::Status::invalidArgument(
			"oa::Viewport::setClearDepth requires a finite value in [0,1]");
	}
	clearDepth_ = inDepth;
	return oa::Status::ok();
}

oa::Status oa::Viewport::setupImage2D(
	oa::F32 inImageWidth,
	oa::F32 inImageHeight,
	const oa::Texture* inTarget,
	const oa::ViewportDesc& inViewport) {
	if (not isFinitePositive(inImageWidth) or not isFinitePositive(inImageHeight)) {
		return oa::Status::invalidArgument(
			"oa::Viewport::setupImage2D requires finite positive image dimensions");
	}
	if (inTarget == nullptr or not inTarget->isValid()) {
		return oa::Status::invalidArgument(
			"oa::Viewport::setupImage2D requires a valid render target");
	}
	if (not inViewport.isValid()) {
		return oa::Status::invalidArgument(
			"oa::Viewport::setupImage2D requires a valid viewport");
	}

	oa::Camera camera(inImageWidth, inImageHeight);
	camera.setOrthographic(inImageWidth, inImageHeight);
	internalCamera_ = camera;
	mode_ = oa::ViewportMode::Image2D;
	camera_ = &internalCamera_;
	target_ = inTarget;
	viewport_ = inViewport;
	scissor_ = {inViewport.x, inViewport.y, inViewport.width, inViewport.height};
	useScissor_ = false;
	return oa::Status::ok();
}

oa::Status oa::Viewport::setupImageAspectFit(
	oa::F32 inImageWidth,
	oa::F32 inImageHeight,
	oa::F32 inWindowWidth,
	oa::F32 inWindowHeight,
	const oa::Texture* inTarget) {
	if (not isFinitePositive(inImageWidth) or not isFinitePositive(inImageHeight)
		or not isFinitePositive(inWindowWidth) or not isFinitePositive(inWindowHeight)) {
		return oa::Status::invalidArgument(
			"oa::Viewport::setupImageAspectFit requires finite positive image and window dimensions");
	}
	if (inTarget == nullptr or not inTarget->isValid()) {
		return oa::Status::invalidArgument(
			"oa::Viewport::setupImageAspectFit requires a valid render target");
	}
	const oa::F64 maxPixel = std::numeric_limits<oa::I32>::max();
	if (inWindowWidth > maxPixel or inWindowHeight > maxPixel) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::Viewport::setupImageAspectFit window exceeds signed pixel coordinates");
	}

	const oa::F64 imageAspect = static_cast<oa::F64>(inImageWidth) / inImageHeight;
	const oa::F64 windowAspect = static_cast<oa::F64>(inWindowWidth) / inWindowHeight;
	oa::F64 displayWidth = 0.0;
	oa::F64 displayHeight = 0.0;
	if (imageAspect > windowAspect) {
		displayWidth = inWindowWidth;
		displayHeight = static_cast<oa::F64>(inWindowWidth) / imageAspect;
	} else {
		displayHeight = inWindowHeight;
		displayWidth = static_cast<oa::F64>(inWindowHeight) * imageAspect;
	}
	const oa::F64 offsetX = (static_cast<oa::F64>(inWindowWidth) - displayWidth) * 0.5;
	const oa::F64 offsetY = (static_cast<oa::F64>(inWindowHeight) - displayHeight) * 0.5;
	if (not std::isfinite(displayWidth) or not std::isfinite(displayHeight)
		or not std::isfinite(offsetX) or not std::isfinite(offsetY)
		or displayWidth < 0.5 or displayHeight < 0.5
		or displayWidth > maxPixel or displayHeight > maxPixel
		or offsetX < 0.0 or offsetY < 0.0
		or offsetX > maxPixel or offsetY > maxPixel) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::Viewport::setupImageAspectFit result cannot be represented as a positive pixel rectangle");
	}

	const oa::ViewportDesc viewport{
		.x = static_cast<oa::I32>(std::llround(offsetX)),
		.y = static_cast<oa::I32>(std::llround(offsetY)),
		.width = static_cast<oa::I32>(std::llround(displayWidth)),
		.height = static_cast<oa::I32>(std::llround(displayHeight)),
	};
	return setupImage2D(inImageWidth, inImageHeight, inTarget, viewport);
}
