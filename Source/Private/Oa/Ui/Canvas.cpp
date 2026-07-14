#include <Oa/Ui/Canvas.h>
#include <cmath>
#include <algorithm>

VlmVec2 OaNodeCanvas::WorldToScreen(VlmVec2 InWorld) const noexcept {
	return {
		(InWorld.X - State_.Pan.X) * State_.Zoom + State_.ViewSize.X * 0.5F,
		(InWorld.Y - State_.Pan.Y) * State_.Zoom + State_.ViewSize.Y * 0.5F,
	};
}

VlmVec2 OaNodeCanvas::ScreenToWorld(VlmVec2 InScreen) const noexcept {
	return {
		(InScreen.X - State_.ViewSize.X * 0.5F) / State_.Zoom + State_.Pan.X,
		(InScreen.Y - State_.ViewSize.Y * 0.5F) / State_.Zoom + State_.Pan.Y,
	};
}

OaWorldAABB OaNodeCanvas::VisibleWorldRect() const noexcept {
	VlmVec2 half = {State_.ViewSize.X * 0.5F / State_.Zoom, State_.ViewSize.Y * 0.5F / State_.Zoom};
	return {State_.Pan - half, State_.Pan + half};
}

void OaNodeCanvas::Pan(VlmVec2 InDeltaScreen) noexcept {
	State_.Pan = State_.Pan - (InDeltaScreen / State_.Zoom);
}

void OaNodeCanvas::ZoomAt(OaF32 InFactor, VlmVec2 InFocusScreen) noexcept {
	VlmVec2 worldBefore = ScreenToWorld(InFocusScreen);
	OaF32 newZoom = std::clamp(State_.Zoom * InFactor, kZoomMin, kZoomMax);
	State_.Zoom = newZoom;
	VlmVec2 worldAfter = ScreenToWorld(InFocusScreen);
	State_.Pan = State_.Pan - (worldAfter - worldBefore);
}

void OaNodeCanvas::FitToView(const OaWorldAABB& InBounds) noexcept {
	AnimTarget_.Pan = (InBounds.Min + InBounds.Max) * 0.5F;
	OaF32 scaleX = State_.ViewSize.X / (InBounds.Max.X - InBounds.Min.X + 1.0F);
	OaF32 scaleY = State_.ViewSize.Y / (InBounds.Max.Y - InBounds.Min.Y + 1.0F);
	AnimTarget_.Zoom = std::clamp(std::min(scaleX, scaleY) * 0.9F, kZoomMin, kZoomMax);
	Animating_ = true;
	AnimT_ = 0.0F;
}

void OaNodeCanvas::StepAnimation(OaF32 InDeltaMs) noexcept {
	if (!Animating_) return;
	AnimT_ += InDeltaMs * 0.005F;
	if (AnimT_ >= 1.0F) {
		State_.Pan  = AnimTarget_.Pan;
		State_.Zoom = AnimTarget_.Zoom;
		Animating_ = false;
		return;
	}
	OaF32 t = AnimT_ * AnimT_ * (3.0F - 2.0F * AnimT_);
	State_.Pan.X = State_.Pan.X + (AnimTarget_.Pan.X - State_.Pan.X) * t;
	State_.Pan.Y = State_.Pan.Y + (AnimTarget_.Pan.Y - State_.Pan.Y) * t;
	State_.Zoom  = State_.Zoom  + (AnimTarget_.Zoom  - State_.Zoom)  * t;
}
