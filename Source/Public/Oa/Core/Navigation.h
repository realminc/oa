// OaNavigation — VulkanEngine 2D plane camera (InputHandler + ortho UBO).
//
// Mouse:
//   LMB / MMB drag  → 1:1 screen pan
//   RMB drag        → dolly zoom (horizontal → dolly Z)
//   Wheel vertical  → pan up/down (web scroll)
//   Wheel horizontal (MX side) → pan left/right
//   Ctrl + wheel    → dolly Z in/out
//
// Touchpad:
//   2-finger scroll → pan
//   Pinch           → zoom at viewport center

#pragma once

#include <Oa/Core/Vlm.h>
#include <Oa/Core/Types.h>
#include <Oa/Ui/Event.h>


struct OaPlaneCamera {
	VlmVec3 Position{0.0F, 0.0F, 1.0F};
};

struct OaNavigationConfig {
	OaF32 PanLimit              = 1.0F;
	OaF32 ZoomMinZ              = 0.15F;
	OaF32 ZoomMaxZ              = 1.15F;
	OaF32 MouseWheelSensitivity = 0.001F;  // Ctrl+scroll dolly

	OaF32 WheelPanScale         = 50.0F;   // main / side wheel → pan pixels per line
	OaF32 CtrlScrollDollyScale  = 0.35F;   // Ctrl+scroll dolly multiplier (higher = faster)
	OaF32 RmbZoomDragScale      = 0.005F;  // RMB horizontal drag → dolly Z
	OaF32 KeyboardPanStep       = 0.5F;
	OaF32 ResetAnimationMs      = 100.0F;

	OaF32 MaxZoomOutZ = 100.0F;

	OaF32 TouchpadPanScale  = 40.0F;
	OaF32 PinchGestureScale = 0.09F;
	OaF32 PinchScrollScale  = 0.10F;

	OaF32 KeyboardZoomStep = 1.05F;

	OaF32 AnimationDurationMs = 200.0F;
};

class OaCamera;

class OaNavigation {
public:
	explicit OaNavigation(const OaNavigationConfig& InConfig = {}) : Config_(InConfig) {}

	void SetContentSize(OaF32 InWidth, OaF32 InHeight) noexcept;
	void SetWindowSize(OaF32 InWidth, OaF32 InHeight) noexcept;

	void FitToWindow(bool InAnimate = false) noexcept;
	void ZoomTo(OaF32 InZoom, OaF32 InAnchorX, OaF32 InAnchorY, bool InAnimate = true) noexcept;
	void PanBy(OaF32 InDx, OaF32 InDy) noexcept;

	void KeyboardPan(OaF32 InDirX, OaF32 InDirY) noexcept;
	void KeyboardZoomIn() noexcept;
	void KeyboardZoomOut() noexcept;
	void KeyboardFitToWindow() noexcept;
	void KeyboardZoomTo100() noexcept;
	void KeyboardReset() noexcept;

	[[nodiscard]] bool HandleEvent(const OaUiEvent& InEvent) noexcept;
	void Update(OaF32 InDeltaMs) noexcept;

	[[nodiscard]] const VlmVec3& Movement() const noexcept { return Movement_; }
	[[nodiscard]] OaPlaneCamera GetPlaneCamera() const noexcept {
		return OaPlaneCamera{Movement_};
	}

	void UpdateCamera(OaCamera& InCamera) const noexcept;

	[[nodiscard]] OaF32 Zoom() const noexcept { return Zoom_; }
	[[nodiscard]] OaF32 PanX() const noexcept { return PanX_; }
	[[nodiscard]] OaF32 PanY() const noexcept { return PanY_; }
	[[nodiscard]] bool  IsPanning() const noexcept { return IsPanDragging_; }
	[[nodiscard]] bool  IsPinching() const noexcept { return IsPinching_; }

	// Wire to the owning viewer's pointer-capture callback for border tracking.
	using CaptureFn = OaFunc<void(bool)>;
	void SetCapturePointer(CaptureFn InFn) { CapturePointer_ = OaStdMove(InFn); }

private:
	static constexpr OaF32 RefZ_ = 1.0F;

	OaNavigationConfig Config_;

	OaF32 ContentW_ = 1.0F;
	OaF32 ContentH_ = 1.0F;
	OaF32 WindowW_  = 1.0F;
	OaF32 WindowH_  = 1.0F;

	VlmVec3 Movement_{0.0F, 0.0F, RefZ_};
	VlmVec3 TargetMovement_{0.0F, 0.0F, RefZ_};

	OaF32 Zoom_ = 1.0F;
	OaF32 PanX_ = 0.0F;
	OaF32 PanY_ = 0.0F;

	OaF32 AnimTimeMs_     = 200.0F;
	OaF32 AnimDurationMs_ = 200.0F;
	bool  IsResetAnim_    = false;

	bool  IsPanDragging_   = false;
	bool  IsRmbZooming_    = false;
	bool  IsPinching_      = false;
	OaF32 LastMouseX_      = 0.0F;
	OaF32 LastMouseY_      = 0.0F;

	OaF32 PinchBeginZ_     = RefZ_;
	OaF32 PinchAccumScale_ = 1.0F;

	CaptureFn CapturePointer_;

	[[nodiscard]] OaF32 MinZ() const noexcept { return Config_.ZoomMinZ; }
	[[nodiscard]] OaF32 MaxZ() const noexcept { return Config_.MaxZoomOutZ; }

	void RefreshZoom() noexcept;
	void SyncPanelFromMovement() noexcept;
	void SyncMovementFromPanel() noexcept;
	void PanScreenBy(OaF32 InDx, OaF32 InDy) noexcept;
	void DollyWheel(OaF32 InScrollDelta) noexcept;
	void RmbZoomDrag(OaF32 InDeltaX) noexcept;
	void ZoomAt(OaF32 InZoom, OaF32 InAnchorX, OaF32 InAnchorY, bool InAnimate) noexcept;
	void ClampMovement() noexcept;
	void ClampPanel() noexcept;
	void BeginPanDrag(OaF32 InMouseX, OaF32 InMouseY) noexcept;
	void EndPanDrag() noexcept;
	void UpdatePanDrag(const OaUiEvent& InEvent) noexcept;
	void BeginRmbZoom(OaF32 InMouseX, OaF32 InMouseY) noexcept;
	void EndRmbZoom() noexcept;
	void BeginAnimation(bool InAnimate, OaF32 InDurationMs) noexcept;
	static OaF32 EaseInQuad(OaF32 InT) noexcept;
	static OaF32 EaseOutCubic(OaF32 InT) noexcept;

	void HandleScroll(const OaUiEvent& InEvent) noexcept;
	void HandlePinch(const OaUiEvent& InEvent) noexcept;
	void KeyboardZoomBy(OaF32 InFactor) noexcept;
};

// ─── Default viewport / viewer keyboard bindings ─────────────────────────────

class OaInputSystem;

struct OaNavigationShortcuts {
	OuiKey ZoomIn     = OuiKey::Equals;
	OuiKey ZoomOut    = OuiKey::Minus;
	OuiKey ZoomFit    = OuiKey::Num0;
	OuiKey ZoomFitAlt = OuiKey::F;
	OuiKey Zoom100    = OuiKey::Num9;
	OuiKey PanUp      = OuiKey::Kp8;
	OuiKey PanDown    = OuiKey::Kp2;
	OuiKey PanLeft    = OuiKey::Kp4;
	OuiKey PanRight   = OuiKey::Kp6;
};

[[nodiscard]] inline constexpr const char* OaNavigationHelpLine() noexcept {
	return "  pan=LMB/MMB/wheel/side/2-finger  zoom=RMB/Ctrl+wheel/pinch  fit=0/F  9=100%";
}

void RegisterViewportShortcuts(OaInputSystem& InInput,
                               OaNavigation& InNav,
                               const OaNavigationShortcuts& InKeys = {});
