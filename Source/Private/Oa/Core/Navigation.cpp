// OaNavigation — VulkanEngine plane camera + v0.6.55 screen pan.

#include <Oa/Core/Navigation.h>
#include <Oa/Core/Input.h>
#include <Oa/Ui/Camera.h>
#include <Oa/Ui/Input.h>

#include <cmath>


void RegisterViewportShortcuts(OaInputSystem& InInput,
                             OaNavigation& InNav,
                             const OaNavigationShortcuts& InKeys) {
	InInput.RegisterAction({.Name = "nav_zoomin", .Binding = {.Key = InKeys.ZoomIn},
		.AllowRepeat = true,
		.Callback = [&InNav] { InNav.KeyboardZoomIn(); }});
	InInput.RegisterAction({.Name = "nav_zoomout", .Binding = {.Key = InKeys.ZoomOut},
		.AllowRepeat = true,
		.Callback = [&InNav] { InNav.KeyboardZoomOut(); }});
	InInput.RegisterAction({.Name = "nav_zoomfit", .Binding = {.Key = InKeys.ZoomFit},
		.Callback = [&InNav] { InNav.KeyboardFitToWindow(); }});
	InInput.RegisterAction({.Name = "nav_zoomfit_alt", .Binding = {.Key = InKeys.ZoomFitAlt},
		.Callback = [&InNav] { InNav.KeyboardFitToWindow(); }});
	InInput.RegisterAction({.Name = "nav_zoom100", .Binding = {.Key = InKeys.Zoom100},
		.Callback = [&InNav] { InNav.KeyboardZoomTo100(); }});
	InInput.RegisterAction({.Name = "nav_panup", .Binding = {.Key = InKeys.PanUp},
		.AllowRepeat = true,
		.Callback = [&InNav] { InNav.KeyboardPan(0.0F, 1.0F); }});
	InInput.RegisterAction({.Name = "nav_pandown", .Binding = {.Key = InKeys.PanDown},
		.AllowRepeat = true,
		.Callback = [&InNav] { InNav.KeyboardPan(0.0F, -1.0F); }});
	InInput.RegisterAction({.Name = "nav_panleft", .Binding = {.Key = InKeys.PanLeft},
		.AllowRepeat = true,
		.Callback = [&InNav] { InNav.KeyboardPan(1.0F, 0.0F); }});
	InInput.RegisterAction({.Name = "nav_panright", .Binding = {.Key = InKeys.PanRight},
		.AllowRepeat = true,
		.Callback = [&InNav] { InNav.KeyboardPan(-1.0F, 0.0F); }});
}


void OaNavigation::RefreshZoom() noexcept {
	const OaF32 z = OaStdClamp(Movement_.Z, MinZ(), MaxZ());
	Zoom_ = RefZ_ / z;
}

void OaNavigation::SyncPanelFromMovement() noexcept {
	RefreshZoom();
	const OaF32 pxPerUnit = WindowH_ / (2.0F * OaStdClamp(Movement_.Z, MinZ(), MaxZ()));
	const OaF32 displayW = ContentW_ * Zoom_;
	const OaF32 displayH = ContentH_ * Zoom_;
	PanX_ = (WindowW_ - displayW) * 0.5F + Movement_.X * pxPerUnit;
	PanY_ = (WindowH_ - displayH) * 0.5F - Movement_.Y * pxPerUnit;
}

void OaNavigation::SyncMovementFromPanel() noexcept {
	RefreshZoom();
	const OaF32 z = OaStdClamp(Movement_.Z, MinZ(), MaxZ());
	const OaF32 pxPerUnit = WindowH_ / (2.0F * z);
	if (pxPerUnit > 0.001F) {
		Movement_.X = (PanX_ - (WindowW_ - ContentW_ * Zoom_) * 0.5F) / pxPerUnit;
		Movement_.Y = ((WindowH_ - ContentH_ * Zoom_) * 0.5F - PanY_) / pxPerUnit;
	}
}

void OaNavigation::UpdateCamera(OaCamera& InCamera) const noexcept {
	const OaF32 z = (Movement_.Z > 0.001F) ? Movement_.Z : 0.001F;
	InCamera.SetPosition({Movement_.X, Movement_.Y, 1.0F});
	InCamera.SetZoom(RefZ_ / z);
	InCamera.SetOffset(0.0F, 0.0F);
}

void OaNavigation::ClampMovement() noexcept {
	Movement_.X = OaStdClamp(Movement_.X, -Config_.PanLimit, Config_.PanLimit);
	Movement_.Y = OaStdClamp(Movement_.Y, -Config_.PanLimit, Config_.PanLimit);
	Movement_.Z = OaStdClamp(Movement_.Z, MinZ(), MaxZ());
	TargetMovement_ = Movement_;
	SyncPanelFromMovement();
}

void OaNavigation::ClampPanel() noexcept {
	RefreshZoom();

	const OaF32 imgW = ContentW_ * Zoom_;
	const OaF32 imgH = ContentH_ * Zoom_;
	const OaF32 minDim = (WindowW_ < WindowH_) ? WindowW_ : WindowH_;
	const OaF32 margin = (minDim * 0.1F < 100.0F) ? minDim * 0.1F : 100.0F;

	PanX_ = OaStdClamp(PanX_, -imgW + margin, WindowW_ - margin);
	PanY_ = OaStdClamp(PanY_, -imgH + margin, WindowH_ - margin);
	SyncMovementFromPanel();
	TargetMovement_ = Movement_;
}

void OaNavigation::SetContentSize(OaF32 InWidth, OaF32 InHeight) noexcept {
	ContentW_ = (InWidth  > 0.0F) ? InWidth  : 1.0F;
	ContentH_ = (InHeight > 0.0F) ? InHeight : 1.0F;
	SyncPanelFromMovement();
}

void OaNavigation::SetWindowSize(OaF32 InWidth, OaF32 InHeight) noexcept {
	WindowW_ = (InWidth  > 0.0F) ? InWidth  : 1.0F;
	WindowH_ = (InHeight > 0.0F) ? InHeight : 1.0F;
	SyncPanelFromMovement();
}

void OaNavigation::FitToWindow(bool InAnimate) noexcept {
	const OaF32 scaleX = WindowW_ / ContentW_;
	const OaF32 scaleY = WindowH_ / ContentH_;
	const OaF32 fitZoom = (scaleX < scaleY) ? scaleX : scaleY;

	TargetMovement_ = {0.0F, 0.0F, OaStdClamp(RefZ_ / fitZoom, MinZ(), MaxZ())};

	if (!InAnimate) {
		Movement_ = TargetMovement_;
		AnimTimeMs_ = Config_.AnimationDurationMs;
		SyncPanelFromMovement();
	} else {
		BeginAnimation(true, Config_.AnimationDurationMs);
	}
}

void OaNavigation::DollyWheel(OaF32 InScrollDelta) noexcept {
	if (std::fabs(InScrollDelta) < 0.0001F) { return; }
	Movement_.Z *= (1.0F - InScrollDelta * Config_.MouseWheelSensitivity);
	Movement_.Z = OaStdClamp(Movement_.Z, MinZ(), MaxZ());
	TargetMovement_ = Movement_;
	SyncPanelFromMovement();
	AnimTimeMs_ = Config_.AnimationDurationMs;
}

void OaNavigation::RmbZoomDrag(OaF32 InDeltaX) noexcept {
	if (std::fabs(InDeltaX) < 0.0001F) { return; }
	const OaF32 delta = std::fabs(InDeltaX) * Config_.RmbZoomDragScale;
	if (InDeltaX > 0.0F) {
		Movement_.Z /= (1.0F + delta);
	} else {
		Movement_.Z *= (1.0F + delta);
	}
	Movement_.Z = OaStdClamp(Movement_.Z, MinZ(), MaxZ());
	TargetMovement_ = Movement_;
	SyncPanelFromMovement();
	AnimTimeMs_ = Config_.AnimationDurationMs;
}

void OaNavigation::ZoomAt(OaF32 InZoom, OaF32 InAnchorX, OaF32 InAnchorY, bool InAnimate) noexcept {
	SyncPanelFromMovement();
	const OaF32 oldZoom = Zoom_;
	const OaF32 targetZoom = OaStdClamp(InZoom, RefZ_ / MaxZ(), RefZ_ / MinZ());
	if (std::fabs(targetZoom - oldZoom) < 0.0001F) { return; }

	const OaF32 ratio = targetZoom / oldZoom;
	const OaF32 newPanX = InAnchorX - (InAnchorX - PanX_) * ratio;
	const OaF32 newPanY = InAnchorY - (InAnchorY - PanY_) * ratio;

	TargetMovement_.Z = OaStdClamp(RefZ_ / targetZoom, MinZ(), MaxZ());
	const OaF32 newZoom = RefZ_ / TargetMovement_.Z;
	const OaF32 pxPerUnit = WindowH_ / (2.0F * TargetMovement_.Z);
	TargetMovement_.X = (newPanX - (WindowW_ - ContentW_ * newZoom) * 0.5F) / pxPerUnit;
	TargetMovement_.Y = ((WindowH_ - ContentH_ * newZoom) * 0.5F - newPanY) / pxPerUnit;

	if (!InAnimate) {
		Movement_ = TargetMovement_;
		Movement_.Z = OaStdClamp(Movement_.Z, MinZ(), MaxZ());
		TargetMovement_ = Movement_;
		AnimTimeMs_ = Config_.AnimationDurationMs;
		SyncPanelFromMovement();
		ClampPanel();
	} else {
		BeginAnimation(true, Config_.AnimationDurationMs);
	}
}

void OaNavigation::ZoomTo(OaF32 InZoom, OaF32 InAnchorX, OaF32 InAnchorY, bool InAnimate) noexcept {
	ZoomAt(InZoom, InAnchorX, InAnchorY, InAnimate);
}

void OaNavigation::PanScreenBy(OaF32 InDx, OaF32 InDy) noexcept {
	if (InDx == 0.0F && InDy == 0.0F) { return; }
	RefreshZoom();
	PanX_ += InDx;
	PanY_ += InDy;
	ClampPanel();
	AnimTimeMs_ = Config_.AnimationDurationMs;
}

void OaNavigation::PanBy(OaF32 InDx, OaF32 InDy) noexcept {
	PanScreenBy(InDx, InDy);
}

void OaNavigation::KeyboardPan(OaF32 InDirX, OaF32 InDirY) noexcept {
	Movement_.X += InDirX * Config_.KeyboardPanStep;
	Movement_.Y -= InDirY * Config_.KeyboardPanStep;
	TargetMovement_ = Movement_;
	ClampMovement();
	AnimTimeMs_ = Config_.AnimationDurationMs;
}

void OaNavigation::KeyboardZoomBy(OaF32 InFactor) noexcept {
	SyncPanelFromMovement();
	const OaF32 oldZ = Movement_.Z;
	const OaF32 newZ = OaStdClamp(oldZ / InFactor, MinZ(), MaxZ());
	if (std::fabs(newZ - oldZ) < 0.0001F) { return; }

	const OaF32 oldZoom = RefZ_ / oldZ;
	const OaF32 cx = PanX_ + ContentW_ * oldZoom * 0.5F;
	const OaF32 cy = PanY_ + ContentH_ * oldZoom * 0.5F;

	Movement_.Z = newZ;
	const OaF32 newZoom = RefZ_ / newZ;
	const OaF32 newPanX = cx - ContentW_ * newZoom * 0.5F;
	const OaF32 newPanY = cy - ContentH_ * newZoom * 0.5F;
	const OaF32 pxPerUnit = WindowH_ / (2.0F * newZ);
	Movement_.X = (newPanX - (WindowW_ - ContentW_ * newZoom) * 0.5F) / pxPerUnit;
	Movement_.Y = ((WindowH_ - ContentH_ * newZoom) * 0.5F - newPanY) / pxPerUnit;
	TargetMovement_ = Movement_;
	SyncPanelFromMovement();
	ClampPanel();
	AnimTimeMs_ = Config_.AnimationDurationMs;
}

void OaNavigation::KeyboardZoomIn() noexcept {
	KeyboardZoomBy(Config_.KeyboardZoomStep);
}

void OaNavigation::KeyboardZoomOut() noexcept {
	KeyboardZoomBy(1.0F / Config_.KeyboardZoomStep);
}

void OaNavigation::KeyboardFitToWindow() noexcept {
	AnimTimeMs_ = Config_.AnimationDurationMs;
	FitToWindow(false);
}

void OaNavigation::KeyboardZoomTo100() noexcept {
	// 100% native frame size, centered in viewport (same origin as FitToWindow).
	TargetMovement_ = {0.0F, 0.0F, RefZ_};
	BeginAnimation(true, Config_.AnimationDurationMs);
}

void OaNavigation::KeyboardReset() noexcept {
	KeyboardFitToWindow();
}

void OaNavigation::BeginPanDrag(OaF32 InMouseX, OaF32 InMouseY) noexcept {
	IsPanDragging_ = true;
	LastMouseX_ = InMouseX;
	LastMouseY_ = InMouseY;
	AnimTimeMs_ = Config_.AnimationDurationMs;
}

void OaNavigation::EndPanDrag() noexcept {
	IsPanDragging_ = false;
	SyncMovementFromPanel();
	TargetMovement_ = Movement_;
	ClampPanel();
}

void OaNavigation::UpdatePanDrag(const OaUiEvent& InEvent) noexcept {
	// Native 1:1 pixel pan — raw relative deltas (FPS-style), position fallback at edge.
	const OaF32 dx = (InEvent.MouseDX != 0.0F)
		? InEvent.MouseDX
		: (InEvent.MouseX - LastMouseX_);
	const OaF32 dy = (InEvent.MouseDY != 0.0F)
		? InEvent.MouseDY
		: (InEvent.MouseY - LastMouseY_);
	if (dx == 0.0F && dy == 0.0F) { return; }

	PanX_ += dx;
	PanY_ += dy;
	LastMouseX_ = InEvent.MouseX;
	LastMouseY_ = InEvent.MouseY;
	AnimTimeMs_ = Config_.AnimationDurationMs;
}

void OaNavigation::BeginRmbZoom(OaF32 InMouseX, OaF32 InMouseY) noexcept {
	IsRmbZooming_ = true;
	LastMouseX_ = InMouseX;
	LastMouseY_ = InMouseY;
	AnimTimeMs_ = Config_.AnimationDurationMs;
}

void OaNavigation::EndRmbZoom() noexcept {
	IsRmbZooming_ = false;
	SyncPanelFromMovement();
}

OaF32 OaNavigation::EaseInQuad(OaF32 InT) noexcept {
	return InT * InT;
}

OaF32 OaNavigation::EaseOutCubic(OaF32 InT) noexcept {
	const OaF32 t = 1.0F - InT;
	return 1.0F - (t * t * t);
}

void OaNavigation::BeginAnimation(bool InAnimate, OaF32 InDurationMs) noexcept {
	AnimDurationMs_ = InDurationMs;
	if (!InAnimate) {
		Movement_ = TargetMovement_;
		AnimTimeMs_ = AnimDurationMs_;
		SyncPanelFromMovement();
	} else {
		AnimTimeMs_ = 0.0F;
	}
}

void OaNavigation::HandleScroll(const OaUiEvent& InEvent) noexcept {
	if (IsPanDragging_ || IsRmbZooming_ || IsPinching_) { return; }

	OuiScrollGesture gesture = InEvent.ScrollGesture;
	if (gesture == OuiScrollGesture::None) {
		gesture = OaInput::ClassifyScroll(InEvent);
	}
	if (gesture == OuiScrollGesture::None) { return; }

	switch (gesture) {
		case OuiScrollGesture::PinchScroll: {
			const OaF32 tick = (InEvent.IntegerScrollY != 0)
				? static_cast<OaF32>(InEvent.IntegerScrollY)
				: InEvent.ScrollY;
			if (std::fabs(tick) < 0.0001F) { break; }
			OaF32 scrollDelta = tick * 120.0F * Config_.CtrlScrollDollyScale;
			scrollDelta = OaStdClamp(scrollDelta, -360.0F, 360.0F);
			DollyWheel(scrollDelta);
			break;
		}
		case OuiScrollGesture::MouseWheel: {
			const OaF32 sx = (InEvent.IntegerScrollX != 0)
				? static_cast<OaF32>(InEvent.IntegerScrollX)
				: InEvent.ScrollX;
			const OaF32 sy = (InEvent.IntegerScrollY != 0)
				? static_cast<OaF32>(InEvent.IntegerScrollY)
				: InEvent.ScrollY;
			PanScreenBy(-sx * Config_.WheelPanScale, sy * Config_.WheelPanScale);
			break;
		}
		case OuiScrollGesture::TouchpadPan:
			PanScreenBy(-InEvent.ScrollX * Config_.TouchpadPanScale,
			            InEvent.ScrollY * Config_.TouchpadPanScale);
			break;
		default:
			break;
	}
}

void OaNavigation::HandlePinch(const OaUiEvent& InEvent) noexcept {
	if (IsPanDragging_ || IsRmbZooming_) { return; }

	const OaF32 cx = WindowW_ * 0.5F;
	const OaF32 cy = WindowH_ * 0.5F;

	switch (InEvent.PinchPhase) {
		case OuiPinchPhase::Begin:
			IsPinching_ = true;
			PinchBeginZ_ = Movement_.Z;
			PinchAccumScale_ = 1.0F;
			break;
		case OuiPinchPhase::Update: {
			if (!IsPinching_) {
				PinchBeginZ_ = Movement_.Z;
				PinchAccumScale_ = 1.0F;
				IsPinching_ = true;
			}
			const OaF32 damped = std::pow(InEvent.GestureScale, Config_.PinchGestureScale);
			PinchAccumScale_ *= damped;
			const OaF32 targetZoom = (RefZ_ / PinchBeginZ_) * PinchAccumScale_;
			ZoomAt(targetZoom, cx, cy, false);
			break;
		}
		case OuiPinchPhase::End:
			IsPinching_ = false;
			break;
		default:
			break;
	}
}

bool OaNavigation::HandleEvent(const OaUiEvent& InEvent) noexcept {
	switch (InEvent.Type) {
		case OuiEventType::MouseScroll:
			HandleScroll(InEvent);
			return true;
		case OuiEventType::Pinch:
			HandlePinch(InEvent);
			return true;
		case OuiEventType::MouseDown:
			if (InEvent.Button == 1 || InEvent.Button == 2) {
				BeginPanDrag(InEvent.MouseX, InEvent.MouseY);
				if (CapturePointer_) { CapturePointer_(true); }
				return true;
			}
			if (InEvent.Button == 3) {
				BeginRmbZoom(InEvent.MouseX, InEvent.MouseY);
				if (CapturePointer_) { CapturePointer_(true); }
				return true;
			}
			break;
		case OuiEventType::MouseUp:
			if ((InEvent.Button == 1 || InEvent.Button == 2) && IsPanDragging_) {
				EndPanDrag();
				if (CapturePointer_ && !IsRmbZooming_) { CapturePointer_(false); }
				return true;
			}
			if (InEvent.Button == 3 && IsRmbZooming_) {
				EndRmbZoom();
				if (CapturePointer_ && !IsPanDragging_) { CapturePointer_(false); }
				return true;
			}
			break;
		case OuiEventType::MouseMove:
			if (IsPanDragging_) {
				UpdatePanDrag(InEvent);
				return true;
			}
			if (IsRmbZooming_) {
				const OaF32 dx = (InEvent.MouseDX != 0.0F)
					? InEvent.MouseDX
					: (InEvent.MouseX - LastMouseX_);
				RmbZoomDrag(dx);
				LastMouseX_ = InEvent.MouseX;
				LastMouseY_ = InEvent.MouseY;
				return true;
			}
			break;
		case OuiEventType::WindowResize:
			SetWindowSize(static_cast<OaF32>(InEvent.WindowW),
			              static_cast<OaF32>(InEvent.WindowH));
			ClampPanel();
			return true;
		default:
			break;
	}
	return false;
}

void OaNavigation::Update(OaF32 InDeltaMs) noexcept {
	// Idle — do not resync panel from movement every frame (causes cursor drift).
	if (AnimTimeMs_ >= AnimDurationMs_) {
		return;
	}

	AnimTimeMs_ += InDeltaMs;
	const OaF32 t = (AnimTimeMs_ < AnimDurationMs_)
		? (AnimTimeMs_ / AnimDurationMs_)
		: 1.0F;
	const OaF32 eased = IsResetAnim_ ? EaseInQuad(t) : EaseOutCubic(t);

	Movement_.X = Movement_.X + (TargetMovement_.X - Movement_.X) * eased;
	Movement_.Y = Movement_.Y + (TargetMovement_.Y - Movement_.Y) * eased;
	Movement_.Z = Movement_.Z + (TargetMovement_.Z - Movement_.Z) * eased;

	if (AnimTimeMs_ >= AnimDurationMs_) {
		Movement_ = TargetMovement_;
		IsResetAnim_ = false;
		AnimTimeMs_ = AnimDurationMs_;
	}
	SyncPanelFromMovement();
}
