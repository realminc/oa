// oa::Navigation — 2D viewer camera and screen-space navigation.

#include <oa/ui/navigation.h>
#include <oa/core/log.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>
#include <oa/ui/input.h>
#include <oa/ui/platformInput.h>

namespace {

[[nodiscard]] bool finiteVec3(const oa::vlm::Vec3& inValue) noexcept {
	return oa::isFinite(inValue.x) and oa::isFinite(inValue.y)
		and oa::isFinite(inValue.z);
}

[[nodiscard]] bool fitsFloat(oa::F64 inValue) noexcept {
	return oa::isFinite(inValue)
		and oa::abs(inValue) <= oa::Limits<oa::F32>::max();
}

[[nodiscard]] oa::Status navigationRangeError(oa::StringView inOperation) {
	oa::String message(inOperation);
	message += " produced navigation state outside the finite float range";
	return oa::Status::error(oa::StatusCode::OutOfRange, oa::move(message));
}

void reportShortcutFailure(const oa::Status& inStatus) {
	if (inStatus.isOk()) return;
	OaLogError(oa::LogComponent::Ui,
		"viewport navigation shortcut failed: %s", inStatus.toString().cStr());
}

} // namespace

oa::Status oa::registerViewportShortcuts(
	oa::InputSystem& inInput,
	oa::Navigation& inNav,
	const oa::NavigationShortcuts& inKeys) {
	OA_RETURN_IF_ERROR(inNav.validate());
	inInput.registerAction({
		.name = "nav_zoomin",
		.binding = {.key = inKeys.zoomIn},
		.allowRepeat = true,
		.callback = [&inNav] { reportShortcutFailure(inNav.keyboardZoomIn()); },
	});
	inInput.registerAction({
		.name = "nav_zoomout",
		.binding = {.key = inKeys.zoomOut},
		.allowRepeat = true,
		.callback = [&inNav] { reportShortcutFailure(inNav.keyboardZoomOut()); },
	});
	inInput.registerAction({
		.name = "nav_zoomfit",
		.binding = {.key = inKeys.zoomFit},
		.callback = [&inNav] { reportShortcutFailure(inNav.keyboardFitToWindow()); },
	});
	inInput.registerAction({
		.name = "nav_zoomfit_alt",
		.binding = {.key = inKeys.zoomFitAlt},
		.callback = [&inNav] { reportShortcutFailure(inNav.keyboardFitToWindow()); },
	});
	inInput.registerAction({
		.name = "nav_zoom100",
		.binding = {.key = inKeys.zoom100},
		.callback = [&inNav] { reportShortcutFailure(inNav.keyboardZoomTo100()); },
	});
	inInput.registerAction({
		.name = "nav_panup",
		.binding = {.key = inKeys.panUp},
		.allowRepeat = true,
		.callback = [&inNav] { reportShortcutFailure(inNav.keyboardPan(0.0F, 1.0F)); },
	});
	inInput.registerAction({
		.name = "nav_pandown",
		.binding = {.key = inKeys.panDown},
		.allowRepeat = true,
		.callback = [&inNav] { reportShortcutFailure(inNav.keyboardPan(0.0F, -1.0F)); },
	});
	inInput.registerAction({
		.name = "nav_panleft",
		.binding = {.key = inKeys.panLeft},
		.allowRepeat = true,
		.callback = [&inNav] { reportShortcutFailure(inNav.keyboardPan(1.0F, 0.0F)); },
	});
	inInput.registerAction({
		.name = "nav_panright",
		.binding = {.key = inKeys.panRight},
		.allowRepeat = true,
		.callback = [&inNav] { reportShortcutFailure(inNav.keyboardPan(-1.0F, 0.0F)); },
	});
	return oa::Status::ok();
}

oa::Navigation::StateSnapshot oa::Navigation::snapshot() const noexcept {
	return {
		.contentW = contentW_,
		.contentH = contentH_,
		.windowW = windowW_,
		.windowH = windowH_,
		.movement = movement_,
		.animStartMovement = animStartMovement_,
		.targetMovement = targetMovement_,
		.zoom = zoom_,
		.panX = panX_,
		.panY = panY_,
		.animTimeMs = animTimeMs_,
		.animDurationMs = animDurationMs_,
		.isPanDragging = isPanDragging_,
		.isRmbZooming = isRmbZooming_,
		.isPinching = isPinching_,
		.lastMouseX = lastMouseX_,
		.lastMouseY = lastMouseY_,
		.pinchBeginZ = pinchBeginZ_,
		.pinchAccumScale = pinchAccumScale_,
	};
}

void oa::Navigation::restore(const StateSnapshot& inState) noexcept {
	contentW_ = inState.contentW;
	contentH_ = inState.contentH;
	windowW_ = inState.windowW;
	windowH_ = inState.windowH;
	movement_ = inState.movement;
	animStartMovement_ = inState.animStartMovement;
	targetMovement_ = inState.targetMovement;
	zoom_ = inState.zoom;
	panX_ = inState.panX;
	panY_ = inState.panY;
	animTimeMs_ = inState.animTimeMs;
	animDurationMs_ = inState.animDurationMs;
	isPanDragging_ = inState.isPanDragging;
	isRmbZooming_ = inState.isRmbZooming;
	isPinching_ = inState.isPinching;
	lastMouseX_ = inState.lastMouseX;
	lastMouseY_ = inState.lastMouseY;
	pinchBeginZ_ = inState.pinchBeginZ;
	pinchAccumScale_ = inState.pinchAccumScale;
}

oa::Status oa::Navigation::validate() const {
	const bool configValid = oa::isFinite(config_.panLimit)
		and config_.panLimit >= 0.0F
		and oa::isFinite(config_.zoomMinZ) and config_.zoomMinZ > 0.0F
		and config_.zoomMinZ <= refZ_
		and static_cast<oa::F64>(refZ_) / config_.zoomMinZ <= oa::Limits<oa::F32>::max()
		and oa::isFinite(config_.maxZoomOutZ)
		and config_.maxZoomOutZ >= refZ_
		and config_.maxZoomOutZ >= config_.zoomMinZ
		and oa::isFinite(config_.mouseWheelSensitivity)
		and config_.mouseWheelSensitivity >= 0.0F
		and oa::isFinite(config_.wheelPanScale) and config_.wheelPanScale >= 0.0F
		and oa::isFinite(config_.ctrlScrollDollyScale)
		and config_.ctrlScrollDollyScale >= 0.0F
		and oa::isFinite(config_.rmbZoomDragScale)
		and config_.rmbZoomDragScale >= 0.0F
		and oa::isFinite(config_.keyboardPanStep)
		and config_.keyboardPanStep >= 0.0F
		and oa::isFinite(config_.touchpadPanScale)
		and config_.touchpadPanScale >= 0.0F
		and oa::isFinite(config_.pinchGestureScale)
		and config_.pinchGestureScale >= 0.0F
		and oa::isFinite(config_.keyboardZoomStep)
		and config_.keyboardZoomStep > 1.0F
		and oa::isFinite(config_.animationDurationMs)
		and config_.animationDurationMs >= 0.0F;
	if (not configValid) {
		return oa::Status::invalidArgument(
			"oa::Navigation requires finite non-negative controls, zoomMinZ <= 1 <= maxZoomOutZ, and keyboardZoomStep > 1");
	}
	const bool stateValid = oa::isFinite(contentW_) and contentW_ > 0.0F
		and oa::isFinite(contentH_) and contentH_ > 0.0F
		and oa::isFinite(windowW_) and windowW_ > 0.0F
		and oa::isFinite(windowH_) and windowH_ > 0.0F
		and finiteVec3(movement_) and finiteVec3(animStartMovement_)
		and finiteVec3(targetMovement_)
		and movement_.z >= minZ() and movement_.z <= maxZ()
		and targetMovement_.z >= minZ() and targetMovement_.z <= maxZ()
		and oa::isFinite(zoom_) and zoom_ > 0.0F
		and oa::isFinite(panX_) and oa::isFinite(panY_)
		and oa::isFinite(animTimeMs_) and animTimeMs_ >= 0.0F
		and oa::isFinite(animDurationMs_) and animDurationMs_ >= 0.0F
		and oa::isFinite(lastMouseX_) and oa::isFinite(lastMouseY_)
		and oa::isFinite(pinchBeginZ_) and pinchBeginZ_ > 0.0F
		and oa::isFinite(pinchAccumScale_) and pinchAccumScale_ > 0.0F;
	if (not stateValid) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Navigation state is outside its finite admitted range");
	}
	return oa::Status::ok();
}

oa::Status oa::Navigation::refreshZoom() {
	const oa::F32 z = oa::clamp(movement_.z, minZ(), maxZ());
	const oa::F64 zoom = static_cast<oa::F64>(refZ_) / z;
	if (not fitsFloat(zoom) or zoom <= 0.0) {
		return navigationRangeError("oa::Navigation::refreshZoom");
	}
	zoom_ = static_cast<oa::F32>(zoom);
	return oa::Status::ok();
}

oa::Status oa::Navigation::syncPanelFromMovement() {
	const oa::F32 z = oa::clamp(movement_.z, minZ(), maxZ());
	const oa::F64 zoom = static_cast<oa::F64>(refZ_) / z;
	const oa::F64 pixelsPerUnit = static_cast<oa::F64>(windowH_) / (2.0 * z);
	const oa::F64 displayW = static_cast<oa::F64>(contentW_) * zoom;
	const oa::F64 displayH = static_cast<oa::F64>(contentH_) * zoom;
	const oa::F64 panX = (static_cast<oa::F64>(windowW_) - displayW) * 0.5
		+ static_cast<oa::F64>(movement_.x) * pixelsPerUnit;
	const oa::F64 panY = (static_cast<oa::F64>(windowH_) - displayH) * 0.5
		- static_cast<oa::F64>(movement_.y) * pixelsPerUnit;
	if (not fitsFloat(zoom) or zoom <= 0.0
		or not fitsFloat(panX) or not fitsFloat(panY)) {
		return navigationRangeError("oa::Navigation::syncPanelFromMovement");
	}
	zoom_ = static_cast<oa::F32>(zoom);
	panX_ = static_cast<oa::F32>(panX);
	panY_ = static_cast<oa::F32>(panY);
	return oa::Status::ok();
}

oa::Status oa::Navigation::syncMovementFromPanel() {
	const oa::F32 z = oa::clamp(movement_.z, minZ(), maxZ());
	const oa::F64 zoom = static_cast<oa::F64>(refZ_) / z;
	const oa::F64 pixelsPerUnit = static_cast<oa::F64>(windowH_) / (2.0 * z);
	if (not oa::isFinite(pixelsPerUnit) or pixelsPerUnit <= 0.0) {
		return navigationRangeError("oa::Navigation::syncMovementFromPanel");
	}
	const oa::F64 movementX = (static_cast<oa::F64>(panX_)
		- (static_cast<oa::F64>(windowW_) - contentW_ * zoom) * 0.5)
		/ pixelsPerUnit;
	const oa::F64 movementY = ((static_cast<oa::F64>(windowH_)
		- contentH_ * zoom) * 0.5 - panY_) / pixelsPerUnit;
	if (not fitsFloat(movementX) or not fitsFloat(movementY)) {
		return navigationRangeError("oa::Navigation::syncMovementFromPanel");
	}
	movement_.x = static_cast<oa::F32>(movementX);
	movement_.y = static_cast<oa::F32>(movementY);
	movement_.z = z;
	zoom_ = static_cast<oa::F32>(zoom);
	return oa::Status::ok();
}

oa::Status oa::Navigation::clampMovement() {
	const StateSnapshot previous = snapshot();
	movement_.x = oa::clamp(movement_.x, -config_.panLimit, config_.panLimit);
	movement_.y = oa::clamp(movement_.y, -config_.panLimit, config_.panLimit);
	movement_.z = oa::clamp(movement_.z, minZ(), maxZ());
	targetMovement_ = movement_;
	const oa::Status status = syncPanelFromMovement();
	if (not status.isOk()) restore(previous);
	return status;
}

oa::Status oa::Navigation::clampPanel() {
	const StateSnapshot previous = snapshot();
	OA_RETURN_IF_ERROR(refreshZoom());
	const oa::F64 imageW = static_cast<oa::F64>(contentW_) * zoom_;
	const oa::F64 imageH = static_cast<oa::F64>(contentH_) * zoom_;
	const oa::F64 minDim = oa::min<oa::F64>(windowW_, windowH_);
	const oa::F64 margin = oa::min(minDim * 0.1, 100.0);
	const oa::F64 panX = oa::clamp<oa::F64>(panX_, -imageW + margin, windowW_ - margin);
	const oa::F64 panY = oa::clamp<oa::F64>(panY_, -imageH + margin, windowH_ - margin);
	if (not fitsFloat(panX) or not fitsFloat(panY)) {
		restore(previous);
		return navigationRangeError("oa::Navigation::clampPanel");
	}
	panX_ = static_cast<oa::F32>(panX);
	panY_ = static_cast<oa::F32>(panY);
	const oa::Status status = syncMovementFromPanel();
	if (not status.isOk()) {
		restore(previous);
		return status;
	}
	targetMovement_ = movement_;
	return oa::Status::ok();
}

oa::Status oa::Navigation::setContentSize(oa::F32 inWidth, oa::F32 inHeight) {
	OA_RETURN_IF_ERROR(validate());
	if (not oa::isFinite(inWidth) or not oa::isFinite(inHeight)
		or inWidth <= 0.0F or inHeight <= 0.0F) {
		return oa::Status::invalidArgument(
			"oa::Navigation::setContentSize requires finite positive dimensions");
	}
	const StateSnapshot previous = snapshot();
	contentW_ = inWidth;
	contentH_ = inHeight;
	const oa::Status status = syncPanelFromMovement();
	if (not status.isOk()) restore(previous);
	return status;
}

oa::Status oa::Navigation::setWindowSize(oa::F32 inWidth, oa::F32 inHeight) {
	OA_RETURN_IF_ERROR(validate());
	if (not oa::isFinite(inWidth) or not oa::isFinite(inHeight)
		or inWidth <= 0.0F or inHeight <= 0.0F) {
		return oa::Status::invalidArgument(
			"oa::Navigation::setWindowSize requires finite positive dimensions");
	}
	const StateSnapshot previous = snapshot();
	windowW_ = inWidth;
	windowH_ = inHeight;
	const oa::Status status = syncPanelFromMovement();
	if (not status.isOk()) restore(previous);
	return status;
}

oa::Status oa::Navigation::fitToWindow(bool inAnimate) {
	OA_RETURN_IF_ERROR(validate());
	const oa::F64 fitZoom = oa::min(
		static_cast<oa::F64>(windowW_) / contentW_,
		static_cast<oa::F64>(windowH_) / contentH_);
	if (not oa::isFinite(fitZoom) or fitZoom <= 0.0) {
		return navigationRangeError("oa::Navigation::fitToWindow");
	}
	const StateSnapshot previous = snapshot();
	targetMovement_ = {
		0.0F,
		0.0F,
		static_cast<oa::F32>(oa::clamp<oa::F64>(
			static_cast<oa::F64>(refZ_) / fitZoom, minZ(), maxZ())),
	};
	const oa::Status status = beginAnimation(inAnimate, config_.animationDurationMs);
	if (not status.isOk()) restore(previous);
	return status;
}

oa::Status oa::Navigation::dollyWheel(oa::F32 inScrollDelta) {
	OA_RETURN_IF_ERROR(validate());
	if (not oa::isFinite(inScrollDelta)) {
		return oa::Status::invalidArgument(
			"oa::Navigation wheel delta must be finite");
	}
	if (oa::abs(inScrollDelta) < 0.0001F) return oa::Status::ok();
	const oa::F64 nextZ = oa::clamp<oa::F64>(
		static_cast<oa::F64>(movement_.z)
			* (1.0 - static_cast<oa::F64>(inScrollDelta)
				* config_.mouseWheelSensitivity),
		minZ(), maxZ());
	if (not fitsFloat(nextZ)) return navigationRangeError("oa::Navigation::dollyWheel");
	const StateSnapshot previous = snapshot();
	movement_.z = static_cast<oa::F32>(nextZ);
	targetMovement_ = movement_;
	animStartMovement_ = movement_;
	animTimeMs_ = animDurationMs_;
	const oa::Status status = syncPanelFromMovement();
	if (not status.isOk()) restore(previous);
	return status;
}

oa::Status oa::Navigation::rmbZoomDrag(oa::F32 inDeltaX) {
	OA_RETURN_IF_ERROR(validate());
	if (not oa::isFinite(inDeltaX)) {
		return oa::Status::invalidArgument(
			"oa::Navigation RMB zoom delta must be finite");
	}
	if (oa::abs(inDeltaX) < 0.0001F) return oa::Status::ok();
	const oa::F64 delta = oa::abs(static_cast<oa::F64>(inDeltaX))
		* config_.rmbZoomDragScale;
	const oa::F64 candidate = inDeltaX > 0.0F
		? static_cast<oa::F64>(movement_.z) / (1.0 + delta)
		: static_cast<oa::F64>(movement_.z) * (1.0 + delta);
	const oa::F64 nextZ = oa::clamp(candidate, static_cast<oa::F64>(minZ()),
		static_cast<oa::F64>(maxZ()));
	if (not fitsFloat(nextZ)) return navigationRangeError("oa::Navigation::rmbZoomDrag");
	const StateSnapshot previous = snapshot();
	movement_.z = static_cast<oa::F32>(nextZ);
	targetMovement_ = movement_;
	animStartMovement_ = movement_;
	animTimeMs_ = animDurationMs_;
	const oa::Status status = syncPanelFromMovement();
	if (not status.isOk()) restore(previous);
	return status;
}

oa::Status oa::Navigation::zoomAt(
	oa::F32 inZoom,
	oa::F32 inAnchorX,
	oa::F32 inAnchorY,
	bool inAnimate) {
	OA_RETURN_IF_ERROR(validate());
	if (not oa::isFinite(inZoom) or inZoom <= 0.0F
		or not oa::isFinite(inAnchorX) or not oa::isFinite(inAnchorY)) {
		return oa::Status::invalidArgument(
			"oa::Navigation::zoomTo requires finite positive zoom and finite anchor coordinates");
	}
	const StateSnapshot previous = snapshot();
	oa::Status status = syncPanelFromMovement();
	if (not status.isOk()) {
		restore(previous);
		return status;
	}
	const oa::F64 oldZoom = zoom_;
	const oa::F64 targetZoom = oa::clamp<oa::F64>(
		inZoom,
		static_cast<oa::F64>(refZ_) / maxZ(),
		static_cast<oa::F64>(refZ_) / minZ());
	if (oa::abs(targetZoom - oldZoom) < 0.0001) return oa::Status::ok();
	const oa::F64 ratio = targetZoom / oldZoom;
	const oa::F64 newPanX = inAnchorX - (static_cast<oa::F64>(inAnchorX) - panX_) * ratio;
	const oa::F64 newPanY = inAnchorY - (static_cast<oa::F64>(inAnchorY) - panY_) * ratio;
	const oa::F64 targetZ = oa::clamp<oa::F64>(
		static_cast<oa::F64>(refZ_) / targetZoom, minZ(), maxZ());
	const oa::F64 admittedZoom = static_cast<oa::F64>(refZ_) / targetZ;
	const oa::F64 pixelsPerUnit = static_cast<oa::F64>(windowH_) / (2.0 * targetZ);
	const oa::F64 targetX = (newPanX
		- (static_cast<oa::F64>(windowW_) - contentW_ * admittedZoom) * 0.5)
		/ pixelsPerUnit;
	const oa::F64 targetY = ((static_cast<oa::F64>(windowH_)
		- contentH_ * admittedZoom) * 0.5 - newPanY) / pixelsPerUnit;
	if (not fitsFloat(targetX) or not fitsFloat(targetY) or not fitsFloat(targetZ)) {
		restore(previous);
		return navigationRangeError("oa::Navigation::zoomTo");
	}
	targetMovement_ = {
		static_cast<oa::F32>(targetX),
		static_cast<oa::F32>(targetY),
		static_cast<oa::F32>(targetZ),
	};
	if (not inAnimate) {
		movement_ = targetMovement_;
		status = syncPanelFromMovement();
		if (status.isOk()) status = clampPanel();
		if (not status.isOk()) {
			restore(previous);
		} else {
			animStartMovement_ = movement_;
			animTimeMs_ = animDurationMs_;
		}
		return status;
	}

	// Admit the target through the same panel clamp used by direct zoom, then
	// restore the current frame and animate from its fixed starting point.
	const oa::vlm::Vec3 requestedTarget = targetMovement_;
	movement_ = requestedTarget;
	status = syncPanelFromMovement();
	if (status.isOk()) status = clampPanel();
	const oa::vlm::Vec3 admittedTarget = movement_;
	restore(previous);
	if (not status.isOk()) return status;
	targetMovement_ = admittedTarget;
	return beginAnimation(true, config_.animationDurationMs);
}

oa::Status oa::Navigation::zoomTo(
	oa::F32 inZoom,
	oa::F32 inAnchorX,
	oa::F32 inAnchorY,
	bool inAnimate) {
	return zoomAt(inZoom, inAnchorX, inAnchorY, inAnimate);
}

oa::Status oa::Navigation::panScreenBy(oa::F32 inDx, oa::F32 inDy) {
	OA_RETURN_IF_ERROR(validate());
	if (not oa::isFinite(inDx) or not oa::isFinite(inDy)) {
		return oa::Status::invalidArgument(
			"oa::Navigation::panBy requires finite screen deltas");
	}
	if (inDx == 0.0F and inDy == 0.0F) return oa::Status::ok();
	const oa::F64 nextX = static_cast<oa::F64>(panX_) + inDx;
	const oa::F64 nextY = static_cast<oa::F64>(panY_) + inDy;
	if (not fitsFloat(nextX) or not fitsFloat(nextY)) {
		return navigationRangeError("oa::Navigation::panBy");
	}
	const StateSnapshot previous = snapshot();
	panX_ = static_cast<oa::F32>(nextX);
	panY_ = static_cast<oa::F32>(nextY);
	const oa::Status status = clampPanel();
	if (not status.isOk()) {
		restore(previous);
	} else {
		animStartMovement_ = movement_;
		animTimeMs_ = animDurationMs_;
	}
	return status;
}

oa::Status oa::Navigation::panBy(oa::F32 inDx, oa::F32 inDy) {
	return panScreenBy(inDx, inDy);
}

oa::Status oa::Navigation::keyboardPan(oa::F32 inDirX, oa::F32 inDirY) {
	OA_RETURN_IF_ERROR(validate());
	if (not oa::isFinite(inDirX) or not oa::isFinite(inDirY)) {
		return oa::Status::invalidArgument(
			"oa::Navigation::keyboardPan requires finite directions");
	}
	const oa::F64 nextX = static_cast<oa::F64>(movement_.x)
		+ static_cast<oa::F64>(inDirX) * config_.keyboardPanStep;
	const oa::F64 nextY = static_cast<oa::F64>(movement_.y)
		- static_cast<oa::F64>(inDirY) * config_.keyboardPanStep;
	if (not fitsFloat(nextX) or not fitsFloat(nextY)) {
		return navigationRangeError("oa::Navigation::keyboardPan");
	}
	const StateSnapshot previous = snapshot();
	movement_.x = static_cast<oa::F32>(nextX);
	movement_.y = static_cast<oa::F32>(nextY);
	const oa::Status status = clampMovement();
	if (not status.isOk()) {
		restore(previous);
	} else {
		animStartMovement_ = movement_;
		animTimeMs_ = animDurationMs_;
	}
	return status;
}

oa::Status oa::Navigation::keyboardZoomBy(oa::F32 inFactor) {
	OA_RETURN_IF_ERROR(validate());
	if (not oa::isFinite(inFactor) or inFactor <= 0.0F) {
		return oa::Status::invalidArgument(
			"oa::Navigation keyboard zoom factor must be finite and positive");
	}
	OA_RETURN_IF_ERROR(syncPanelFromMovement());
	const oa::F32 targetZoom = oa::clamp(
		zoom_ * inFactor,
		refZ_ / maxZ(),
		refZ_ / minZ());
	const oa::F32 centerX = panX_ + contentW_ * zoom_ * 0.5F;
	const oa::F32 centerY = panY_ + contentH_ * zoom_ * 0.5F;
	return zoomAt(targetZoom, centerX, centerY, false);
}

oa::Status oa::Navigation::keyboardZoomIn() {
	return keyboardZoomBy(config_.keyboardZoomStep);
}

oa::Status oa::Navigation::keyboardZoomOut() {
	return keyboardZoomBy(1.0F / config_.keyboardZoomStep);
}

oa::Status oa::Navigation::keyboardFitToWindow() {
	return fitToWindow(false);
}

oa::Status oa::Navigation::keyboardZoomTo100() {
	OA_RETURN_IF_ERROR(validate());
	const StateSnapshot previous = snapshot();
	targetMovement_ = {0.0F, 0.0F, refZ_};
	const oa::Status status = beginAnimation(true, config_.animationDurationMs);
	if (not status.isOk()) restore(previous);
	return status;
}

oa::Status oa::Navigation::keyboardReset() {
	return keyboardFitToWindow();
}

void oa::Navigation::beginPanDrag(oa::F32 inMouseX, oa::F32 inMouseY) noexcept {
	isPanDragging_ = true;
	lastMouseX_ = inMouseX;
	lastMouseY_ = inMouseY;
	animStartMovement_ = movement_;
	animTimeMs_ = animDurationMs_;
}

oa::Status oa::Navigation::endPanDrag() {
	const StateSnapshot previous = snapshot();
	isPanDragging_ = false;
	oa::Status status = syncMovementFromPanel();
	if (status.isOk()) status = clampPanel();
	if (not status.isOk()) restore(previous);
	return status;
}

oa::Status oa::Navigation::updatePanDrag(const oa::UiEvent& inEvent) {
	const oa::F32 dx = inEvent.mouseDX != 0.0F
		? inEvent.mouseDX : inEvent.mouseX - lastMouseX_;
	const oa::F32 dy = inEvent.mouseDY != 0.0F
		? inEvent.mouseDY : inEvent.mouseY - lastMouseY_;
	if (not oa::isFinite(dx) or not oa::isFinite(dy)
		or not oa::isFinite(inEvent.mouseX) or not oa::isFinite(inEvent.mouseY)) {
		return oa::Status::invalidArgument(
			"oa::Navigation pan event requires finite coordinates and deltas");
	}
	if (dx == 0.0F and dy == 0.0F) return oa::Status::ok();
	const oa::F64 nextX = static_cast<oa::F64>(panX_) + dx;
	const oa::F64 nextY = static_cast<oa::F64>(panY_) + dy;
	if (not fitsFloat(nextX) or not fitsFloat(nextY)) {
		return navigationRangeError("oa::Navigation pan drag");
	}
	panX_ = static_cast<oa::F32>(nextX);
	panY_ = static_cast<oa::F32>(nextY);
	lastMouseX_ = inEvent.mouseX;
	lastMouseY_ = inEvent.mouseY;
	animStartMovement_ = movement_;
	animTimeMs_ = animDurationMs_;
	return oa::Status::ok();
}

void oa::Navigation::beginRmbZoom(oa::F32 inMouseX, oa::F32 inMouseY) noexcept {
	isRmbZooming_ = true;
	lastMouseX_ = inMouseX;
	lastMouseY_ = inMouseY;
	animStartMovement_ = movement_;
	animTimeMs_ = animDurationMs_;
}

oa::Status oa::Navigation::endRmbZoom() {
	const StateSnapshot previous = snapshot();
	isRmbZooming_ = false;
	const oa::Status status = syncPanelFromMovement();
	if (not status.isOk()) restore(previous);
	return status;
}

oa::F32 oa::Navigation::easeOutCubic(oa::F32 inT) noexcept {
	const oa::F32 t = 1.0F - inT;
	return 1.0F - t * t * t;
}

oa::Status oa::Navigation::beginAnimation(bool inAnimate, oa::F32 inDurationMs) {
	if (not oa::isFinite(inDurationMs) or inDurationMs < 0.0F) {
		return oa::Status::invalidArgument(
			"oa::Navigation animation duration must be finite and non-negative");
	}
	animDurationMs_ = inDurationMs;
	animStartMovement_ = movement_;
	if (not inAnimate or inDurationMs == 0.0F) {
		movement_ = targetMovement_;
		animStartMovement_ = movement_;
		animTimeMs_ = animDurationMs_;
		return syncPanelFromMovement();
	}
	animTimeMs_ = 0.0F;
	return oa::Status::ok();
}

oa::Status oa::Navigation::handleScroll(const oa::UiEvent& inEvent) {
	if (isPanDragging_ or isRmbZooming_ or isPinching_) return oa::Status::ok();
	if (not oa::isFinite(inEvent.scrollX) or not oa::isFinite(inEvent.scrollY)) {
		return oa::Status::invalidArgument(
			"oa::Navigation scroll event requires finite deltas");
	}
	oa::UiScrollGesture gesture = inEvent.scrollGesture;
	if (gesture == oa::UiScrollGesture::None) gesture = oa::input::classifyScroll(inEvent);
	if (gesture == oa::UiScrollGesture::None) return oa::Status::ok();
	switch (gesture) {
		case oa::UiScrollGesture::PinchScroll: {
			const oa::F32 tick = inEvent.integerScrollY != 0
				? static_cast<oa::F32>(inEvent.integerScrollY) : inEvent.scrollY;
			if (oa::abs(tick) < 0.0001F) return oa::Status::ok();
			const oa::F32 delta = oa::clamp(
				tick * 120.0F * config_.ctrlScrollDollyScale,
				-360.0F, 360.0F);
			return dollyWheel(delta);
		}
		case oa::UiScrollGesture::MouseWheel: {
			const oa::F32 sx = inEvent.integerScrollX != 0
				? static_cast<oa::F32>(inEvent.integerScrollX) : inEvent.scrollX;
			const oa::F32 sy = inEvent.integerScrollY != 0
				? static_cast<oa::F32>(inEvent.integerScrollY) : inEvent.scrollY;
			return panScreenBy(-sx * config_.wheelPanScale,
				sy * config_.wheelPanScale);
		}
		case oa::UiScrollGesture::TouchpadPan:
			return panScreenBy(
				-inEvent.scrollX * config_.touchpadPanScale,
				inEvent.scrollY * config_.touchpadPanScale);
		default:
			return oa::Status::ok();
	}
}

oa::Status oa::Navigation::handlePinch(const oa::UiEvent& inEvent) {
	if (isPanDragging_ or isRmbZooming_) return oa::Status::ok();
	if (inEvent.pinchPhase == oa::UiPinchPhase::Update
		and (not oa::isFinite(inEvent.gestureScale)
			or inEvent.gestureScale <= 0.0F)) {
		return oa::Status::invalidArgument(
			"oa::Navigation pinch scale must be finite and positive");
	}
	const oa::F32 cx = windowW_ * 0.5F;
	const oa::F32 cy = windowH_ * 0.5F;
	switch (inEvent.pinchPhase) {
		case oa::UiPinchPhase::Begin:
			isPinching_ = true;
			pinchBeginZ_ = movement_.z;
			pinchAccumScale_ = 1.0F;
			break;
		case oa::UiPinchPhase::Update: {
			const StateSnapshot previous = snapshot();
			if (not isPinching_) {
				pinchBeginZ_ = movement_.z;
				pinchAccumScale_ = 1.0F;
				isPinching_ = true;
			}
			const oa::F64 damped = oa::pow(
				static_cast<oa::F64>(inEvent.gestureScale),
				static_cast<oa::F64>(config_.pinchGestureScale));
			const oa::F64 accumulated = static_cast<oa::F64>(pinchAccumScale_) * damped;
			const oa::F64 targetZoom = (static_cast<oa::F64>(refZ_) / pinchBeginZ_)
				* accumulated;
			if (not fitsFloat(accumulated) or accumulated <= 0.0
				or not fitsFloat(targetZoom) or targetZoom <= 0.0) {
				restore(previous);
				return navigationRangeError("oa::Navigation pinch");
			}
			pinchAccumScale_ = static_cast<oa::F32>(accumulated);
			const oa::Status status = zoomAt(
				static_cast<oa::F32>(targetZoom), cx, cy, false);
			if (not status.isOk()) restore(previous);
			return status;
		}
		case oa::UiPinchPhase::End:
			isPinching_ = false;
			break;
		default:
			break;
	}
	return oa::Status::ok();
}

oa::Result<bool> oa::Navigation::handleEvent(const oa::UiEvent& inEvent) {
	OA_RETURN_IF_ERROR(validate());
	switch (inEvent.type) {
		case oa::UiEventType::MouseScroll:
			OA_RETURN_IF_ERROR(handleScroll(inEvent));
			return true;
		case oa::UiEventType::Pinch:
			OA_RETURN_IF_ERROR(handlePinch(inEvent));
			return true;
		case oa::UiEventType::MouseDown:
			if (inEvent.button == 1 or inEvent.button == 2 or inEvent.button == 3) {
				if (not oa::isFinite(inEvent.mouseX) or not oa::isFinite(inEvent.mouseY)) {
					return oa::Status::invalidArgument(
						"oa::Navigation pointer-down coordinates must be finite");
				}
				if (inEvent.button == 3) beginRmbZoom(inEvent.mouseX, inEvent.mouseY);
				else beginPanDrag(inEvent.mouseX, inEvent.mouseY);
				if (capturePointer_) capturePointer_(true);
				return true;
			}
			break;
		case oa::UiEventType::MouseUp:
			if ((inEvent.button == 1 or inEvent.button == 2) and isPanDragging_) {
				OA_RETURN_IF_ERROR(endPanDrag());
				if (capturePointer_ and not isRmbZooming_) capturePointer_(false);
				return true;
			}
			if (inEvent.button == 3 and isRmbZooming_) {
				OA_RETURN_IF_ERROR(endRmbZoom());
				if (capturePointer_ and not isPanDragging_) capturePointer_(false);
				return true;
			}
			break;
		case oa::UiEventType::MouseMove:
			if (isPanDragging_) {
				OA_RETURN_IF_ERROR(updatePanDrag(inEvent));
				return true;
			}
			if (isRmbZooming_) {
				const oa::F32 dx = inEvent.mouseDX != 0.0F
					? inEvent.mouseDX : inEvent.mouseX - lastMouseX_;
				if (not oa::isFinite(dx) or not oa::isFinite(inEvent.mouseX)
					or not oa::isFinite(inEvent.mouseY)) {
					return oa::Status::invalidArgument(
						"oa::Navigation RMB zoom event requires finite coordinates and delta");
				}
				OA_RETURN_IF_ERROR(rmbZoomDrag(dx));
				lastMouseX_ = inEvent.mouseX;
				lastMouseY_ = inEvent.mouseY;
				return true;
			}
			break;
		case oa::UiEventType::WindowResize:
			// Zero-sized resize events represent a minimized native window. They do
			// not alter the last usable navigation viewport.
			if (inEvent.windowW <= 0 or inEvent.windowH <= 0) return true;
			{
				const StateSnapshot previous = snapshot();
				oa::Status status = setWindowSize(
				static_cast<oa::F32>(inEvent.windowW),
				static_cast<oa::F32>(inEvent.windowH));
				if (status.isOk()) status = clampPanel();
				if (not status.isOk()) {
					restore(previous);
					return status;
				}
			}
			return true;
		default:
			break;
	}
	return false;
}

oa::Status oa::Navigation::update(oa::F32 inDeltaMs) {
	OA_RETURN_IF_ERROR(validate());
	if (not oa::isFinite(inDeltaMs) or inDeltaMs < 0.0F) {
		return oa::Status::invalidArgument(
			"oa::Navigation::update requires a finite non-negative delta");
	}
	if (animTimeMs_ >= animDurationMs_) return oa::Status::ok();
	const StateSnapshot previous = snapshot();
	const oa::F64 time = oa::min<oa::F64>(
		static_cast<oa::F64>(animTimeMs_) + inDeltaMs,
		animDurationMs_);
	const oa::F32 t = animDurationMs_ > 0.0F
		? static_cast<oa::F32>(time / animDurationMs_) : 1.0F;
	const oa::F32 eased = easeOutCubic(t);
	movement_.x = animStartMovement_.x
		+ (targetMovement_.x - animStartMovement_.x) * eased;
	movement_.y = animStartMovement_.y
		+ (targetMovement_.y - animStartMovement_.y) * eased;
	movement_.z = animStartMovement_.z
		+ (targetMovement_.z - animStartMovement_.z) * eased;
	animTimeMs_ = static_cast<oa::F32>(time);
	if (animTimeMs_ >= animDurationMs_) {
		movement_ = targetMovement_;
		animStartMovement_ = movement_;
		animTimeMs_ = animDurationMs_;
	}
	const oa::Status status = syncPanelFromMovement();
	if (not status.isOk()) restore(previous);
	return status;
}
