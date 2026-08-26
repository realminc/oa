#include <oa/ui/canvas.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>

namespace {

[[nodiscard]] bool isFiniteVec2(oa::vlm::Vec2 inValue) noexcept {
	return oa::isFinite(inValue.x) and oa::isFinite(inValue.y);
}

[[nodiscard]] bool isValidState(
	const oa::NodeCanvasState& inState,
	bool inRequirePositiveView) noexcept {
	return isFiniteVec2(inState.pan) and isFiniteVec2(inState.viewSize)
		and oa::isFinite(inState.zoom)
		and inState.zoom >= oa::NodeCanvas::kZoomMin
		and inState.zoom <= oa::NodeCanvas::kZoomMax
		and (inRequirePositiveView
			? inState.viewSize.x > 0.0F and inState.viewSize.y > 0.0F
			: inState.viewSize.x >= 0.0F and inState.viewSize.y >= 0.0F);
}

[[nodiscard]] oa::Status invalidCanvasState() {
	return oa::Status::error(
		oa::StatusCode::FailedPrecondition,
		"oa::NodeCanvas requires finite pan/view state and an admitted zoom");
}

} // namespace

oa::vlm::Vec2 oa::NodeCanvas::worldToScreen(oa::vlm::Vec2 inWorld) const noexcept {
	return {
		(inWorld.x - state_.pan.x) * state_.zoom + state_.viewSize.x * 0.5F,
		(inWorld.y - state_.pan.y) * state_.zoom + state_.viewSize.y * 0.5F,
	};
}

oa::vlm::Vec2 oa::NodeCanvas::screenToWorld(oa::vlm::Vec2 inScreen) const noexcept {
	return {
		(inScreen.x - state_.viewSize.x * 0.5F) / state_.zoom + state_.pan.x,
		(inScreen.y - state_.viewSize.y * 0.5F) / state_.zoom + state_.pan.y,
	};
}

oa::WorldAabb oa::NodeCanvas::visibleWorldRect() const noexcept {
	const oa::vlm::Vec2 half = {
		state_.viewSize.x * 0.5F / state_.zoom,
		state_.viewSize.y * 0.5F / state_.zoom};
	return {state_.pan - half, state_.pan + half};
}

oa::Result<oa::Optional<oa::NodeCanvasHit>> oa::NodeCanvas::hitTest(
	oa::vlm::Vec2 inScreen,
	oa::Span<const oa::NodeCanvasHitItem> inItems) const {
	if (not isValidState(state_, true)) return invalidCanvasState();
	if (not isFiniteVec2(inScreen)) {
		return oa::Status::invalidArgument(
			"oa::NodeCanvas::hitTest requires a finite screen point");
	}
	for (const oa::NodeCanvasHitItem& item : inItems) {
		if (item.id == 0U or not item.bounds.isValid()) {
			return oa::Status::invalidArgument(
				"oa::NodeCanvas::hitTest requires non-zero IDs and valid bounds");
		}
	}
	const oa::vlm::Vec2 world = screenToWorld(inScreen);
	if (not isFiniteVec2(world)) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::NodeCanvas::hitTest produced a non-finite world point");
	}
	oa::Optional<oa::NodeCanvasHit> hit;
	for (oa::Usize index = 0U; index < inItems.size(); ++index) {
		const oa::NodeCanvasHitItem& item = inItems[index];
		if (not item.enabled or not item.bounds.contains(world)) continue;
		if (not hit.hasValue() or item.layer >= hit->layer) {
			hit = oa::NodeCanvasHit{
				.id = item.id,
				.index = index,
				.layer = item.layer,
				.worldPoint = world,
			};
		}
	}
	return hit;
}

oa::Result<oa::NodeCanvasGrid> oa::NodeCanvas::grid(
	oa::F32 inMinimumScreenSpacing,
	oa::F32 inBaseWorldStep,
	oa::U32 inMajorEvery,
	oa::U32 inSuperMajorEvery) const {
	if (not isValidState(state_, true)) return invalidCanvasState();
	if (not oa::isFinite(inMinimumScreenSpacing)
		or not oa::isFinite(inBaseWorldStep)
		or inMinimumScreenSpacing < 4.0F or inMinimumScreenSpacing > 512.0F
		or inBaseWorldStep <= 0.0F or inMajorEvery < 2U or inMajorEvery > 100U
		or inSuperMajorEvery < inMajorEvery * 2U
		or inSuperMajorEvery > 10000U
		or inSuperMajorEvery % inMajorEvery != 0U) {
		return oa::Status::invalidArgument(
			"oa::NodeCanvas::grid requires finite spacing, a positive base step, and divisible major/super-major tiers");
	}
	const oa::F64 ratio = static_cast<oa::F64>(inMinimumScreenSpacing)
		/ (static_cast<oa::F64>(inBaseWorldStep) * state_.zoom);
	// Preserve the Maya-style decimal hierarchy: 10 -> 100 -> 1000. Zooming
	// out promotes the visible minor tier by whole decades; zooming in never
	// invents a sub-base grid.
	const oa::F64 exponentValue = oa::max(0.0, oa::ceil(oa::log10(ratio)));
	if (not oa::isFinite(exponentValue)
		or exponentValue < static_cast<oa::F64>(oa::Limits<int>::min())
		or exponentValue > static_cast<oa::F64>(oa::Limits<int>::max())) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::NodeCanvas::grid spacing exponent is out of range");
	}
	const oa::F64 worldStep = static_cast<oa::F64>(inBaseWorldStep)
		* oa::pow(10.0, exponentValue);
	const oa::F64 screenStep = worldStep * state_.zoom;
	const oa::vlm::Vec2 origin = worldToScreen({0.0F, 0.0F});
	if (not oa::isFinite(worldStep) or not oa::isFinite(screenStep)
		or worldStep <= 0.0 or screenStep <= 0.0
		or worldStep > oa::Limits<oa::F32>::max()
		or screenStep > oa::Limits<oa::F32>::max()
		or not isFiniteVec2(origin)) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::NodeCanvas::grid produced non-finite metrics");
	}
	return oa::NodeCanvasGrid{
		.minorWorldStep = static_cast<oa::F32>(worldStep),
		.minorScreenStep = static_cast<oa::F32>(screenStep),
		.originScreen = origin,
		.majorEvery = inMajorEvery,
		.superMajorEvery = inSuperMajorEvery,
	};
}

oa::Status oa::NodeCanvas::setState(const oa::NodeCanvasState& inState) {
	if (not isValidState(inState, true)) {
		return oa::Status::invalidArgument(
			"oa::NodeCanvas::setState requires finite pan, positive view size, and zoom in range");
	}
	state_ = inState;
	animStart_ = inState;
	animTarget_ = inState;
	animating_ = false;
	animT_ = 0.0F;
	return oa::Status::ok();
}

oa::Status oa::NodeCanvas::setViewSize(oa::F32 inW, oa::F32 inH) {
	if (not oa::isFinite(inW) or not oa::isFinite(inH)
		or inW <= 0.0F or inH <= 0.0F) {
		return oa::Status::invalidArgument(
			"oa::NodeCanvas::setViewSize requires finite positive dimensions");
	}
	state_.viewSize = {inW, inH};
	animStart_.viewSize = state_.viewSize;
	animTarget_.viewSize = state_.viewSize;
	animating_ = false;
	animT_ = 0.0F;
	return oa::Status::ok();
}

oa::Status oa::NodeCanvas::pan(oa::vlm::Vec2 inDeltaScreen) {
	if (not isValidState(state_, false)) return invalidCanvasState();
	if (not isFiniteVec2(inDeltaScreen)) {
		return oa::Status::invalidArgument(
			"oa::NodeCanvas::pan requires a finite screen delta");
	}
	const oa::F64 nextX = static_cast<oa::F64>(state_.pan.x)
		- static_cast<oa::F64>(inDeltaScreen.x) / state_.zoom;
	const oa::F64 nextY = static_cast<oa::F64>(state_.pan.y)
		- static_cast<oa::F64>(inDeltaScreen.y) / state_.zoom;
	if (not oa::isFinite(nextX) or not oa::isFinite(nextY)
		or oa::abs(nextX) > oa::Limits<oa::F32>::max()
		or oa::abs(nextY) > oa::Limits<oa::F32>::max()) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::NodeCanvas::pan result is out of range");
	}
	state_.pan = {static_cast<oa::F32>(nextX), static_cast<oa::F32>(nextY)};
	animating_ = false;
	animT_ = 0.0F;
	return oa::Status::ok();
}

oa::Status oa::NodeCanvas::zoomAt(oa::F32 inFactor, oa::vlm::Vec2 inFocusScreen) {
	if (not isValidState(state_, false)) return invalidCanvasState();
	if (not oa::isFinite(inFactor) or inFactor <= 0.0F
		or not isFiniteVec2(inFocusScreen)) {
		return oa::Status::invalidArgument(
			"oa::NodeCanvas::zoomAt requires a finite positive factor and focus");
	}
	const oa::NodeCanvasState previous = state_;
	const oa::vlm::Vec2 worldBefore = screenToWorld(inFocusScreen);
	if (not isFiniteVec2(worldBefore)) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::NodeCanvas::zoomAt focus is outside the finite world range");
	}
	const oa::F32 newZoom = oa::clamp(state_.zoom * inFactor, kZoomMin, kZoomMax);
	state_.zoom = newZoom;
	const oa::vlm::Vec2 worldAfter = screenToWorld(inFocusScreen);
	state_.pan = state_.pan - (worldAfter - worldBefore);
	if (not isValidState(state_, false)) {
		state_ = previous;
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::NodeCanvas::zoomAt result is out of range");
	}
	animating_ = false;
	animT_ = 0.0F;
	return oa::Status::ok();
}

oa::Status oa::NodeCanvas::fitToView(
	const oa::WorldAabb& inBounds,
	oa::F32 inPaddingPixels) {
	if (not isValidState(state_, true)) return invalidCanvasState();
	if (not inBounds.isValid() or not oa::isFinite(inPaddingPixels)
		or inPaddingPixels < 0.0F
		or inPaddingPixels * 2.0F >= state_.viewSize.x
		or inPaddingPixels * 2.0F >= state_.viewSize.y) {
		return oa::Status::invalidArgument(
			"oa::NodeCanvas::fitToView requires valid bounds and usable padding");
	}
	const oa::F64 extentX = oa::max<oa::F64>(
		static_cast<oa::F64>(inBounds.max.x) - inBounds.min.x, 1.0);
	const oa::F64 extentY = oa::max<oa::F64>(
		static_cast<oa::F64>(inBounds.max.y) - inBounds.min.y, 1.0);
	oa::NodeCanvasState target = state_;
	target.pan = {
		static_cast<oa::F32>((static_cast<oa::F64>(inBounds.min.x)
			+ inBounds.max.x) * 0.5),
		static_cast<oa::F32>((static_cast<oa::F64>(inBounds.min.y)
			+ inBounds.max.y) * 0.5),
	};
	target.zoom = oa::clamp(static_cast<oa::F32>(oa::min(
		(state_.viewSize.x - inPaddingPixels * 2.0F) / extentX,
		(state_.viewSize.y - inPaddingPixels * 2.0F) / extentY)),
		kZoomMin, kZoomMax);
	if (not isValidState(target, true)) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::NodeCanvas::fitToView produced an invalid target");
	}
	animStart_ = state_;
	animTarget_ = target;
	animating_ = true;
	animT_ = 0.0F;
	return oa::Status::ok();
}

oa::Status oa::NodeCanvas::stepAnimation(oa::F32 inDeltaMs) {
	if (not oa::isFinite(inDeltaMs) or inDeltaMs < 0.0F) {
		return oa::Status::invalidArgument(
			"oa::NodeCanvas::stepAnimation requires a finite non-negative delta");
	}
	if (!animating_) return oa::Status::ok();
	animT_ += inDeltaMs * 0.005F;
	if (animT_ >= 1.0F) {
		state_.pan  = animTarget_.pan;
		state_.zoom = animTarget_.zoom;
		animating_ = false;
		animT_ = 1.0F;
		return oa::Status::ok();
	}
	const oa::F32 t = animT_ * animT_ * (3.0F - 2.0F * animT_);
	state_.pan.x = animStart_.pan.x
		+ (animTarget_.pan.x - animStart_.pan.x) * t;
	state_.pan.y = animStart_.pan.y
		+ (animTarget_.pan.y - animStart_.pan.y) * t;
	state_.zoom = animStart_.zoom
		+ (animTarget_.zoom - animStart_.zoom) * t;
	return oa::Status::ok();
}
