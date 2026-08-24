// Navigation — 2D viewer navigation state and input mapping.
//
// mouse:
//   LMB / MMB drag  → 1:1 screen pan
//   RMB drag        → dolly zoom (horizontal → dolly Z)
//   wheel vertical  → pan up/down (web scroll)
//   wheel horizontal (MX side) → pan left/right
//   ctrl + wheel    → dolly Z in/out
//
// Touchpad:
//   2-finger scroll → pan
//   Pinch           → zoom at viewport center

#pragma once

#include <oa/core/status.h>
#include <oa/core/vlm.h>
#include <oa/core/types.h>
#include <oa/ui/event.h>


namespace oa {

struct PlaneCamera {
	oa::vlm::Vec3 position{0.0F, 0.0F, 1.0F};
};

struct NavigationConfig {
	oa::F32 panLimit              = 1.0F;
	oa::F32 zoomMinZ              = 0.15F;
	oa::F32 maxZoomOutZ           = 100.0F;
	oa::F32 mouseWheelSensitivity = 0.001F;  // ctrl+scroll dolly

	oa::F32 wheelPanScale         = 50.0F;   // main / side wheel → pan pixels per line
	oa::F32 ctrlScrollDollyScale  = 0.35F;   // ctrl+scroll dolly multiplier (higher = faster)
	oa::F32 rmbZoomDragScale      = 0.005F;  // RMB horizontal drag → dolly Z
	oa::F32 keyboardPanStep       = 0.5F;

	oa::F32 touchpadPanScale  = 40.0F;
	oa::F32 pinchGestureScale = 0.09F;

	oa::F32 keyboardZoomStep = 1.05F;

	oa::F32 animationDurationMs = 200.0F;
};

class Navigation {
public:
	explicit Navigation(const NavigationConfig& inConfig = {}) : config_(inConfig) {}

	// Configuration and direct mutations are result-bearing. Rejected input
	// never changes the current navigation state.
	[[nodiscard]] oa::Status validate() const;
	[[nodiscard]] oa::Status setContentSize(oa::F32 inWidth, oa::F32 inHeight);
	[[nodiscard]] oa::Status setWindowSize(oa::F32 inWidth, oa::F32 inHeight);

	[[nodiscard]] oa::Status fitToWindow(bool inAnimate = false);
	[[nodiscard]] oa::Status zoomTo(
		oa::F32 inZoom,
		oa::F32 inAnchorX,
		oa::F32 inAnchorY,
		bool inAnimate = true);
	[[nodiscard]] oa::Status panBy(oa::F32 inDx, oa::F32 inDy);

	[[nodiscard]] oa::Status keyboardPan(oa::F32 inDirX, oa::F32 inDirY);
	[[nodiscard]] oa::Status keyboardZoomIn();
	[[nodiscard]] oa::Status keyboardZoomOut();
	[[nodiscard]] oa::Status keyboardFitToWindow();
	[[nodiscard]] oa::Status keyboardZoomTo100();
	[[nodiscard]] oa::Status keyboardReset();

	[[nodiscard]] oa::Result<bool> handleEvent(const UiEvent& inEvent);
	[[nodiscard]] oa::Status update(oa::F32 inDeltaMs);

	[[nodiscard]] const oa::vlm::Vec3& movement() const noexcept { return movement_; }
	[[nodiscard]] PlaneCamera getPlaneCamera() const noexcept {
		return PlaneCamera{movement_};
	}

	[[nodiscard]] oa::F32 zoom() const noexcept { return zoom_; }
	[[nodiscard]] oa::F32 panX() const noexcept { return panX_; }
	[[nodiscard]] oa::F32 panY() const noexcept { return panY_; }
	[[nodiscard]] bool  isPanning() const noexcept { return isPanDragging_; }
	[[nodiscard]] bool  isPinching() const noexcept { return isPinching_; }
	[[nodiscard]] bool  isAnimating() const noexcept {
		return animTimeMs_ < animDurationMs_;
	}

	// Wire to the owning viewer's pointer-capture callback for border tracking.
	using CaptureFn = oa::Fn<void(bool)>;
	void setCapturePointer(CaptureFn inFn) { capturePointer_ = oa::move(inFn); }

private:
	static constexpr oa::F32 refZ_ = 1.0F;
	struct StateSnapshot {
		oa::F32 contentW;
		oa::F32 contentH;
		oa::F32 windowW;
		oa::F32 windowH;
		oa::vlm::Vec3 movement;
		oa::vlm::Vec3 animStartMovement;
		oa::vlm::Vec3 targetMovement;
		oa::F32 zoom;
		oa::F32 panX;
		oa::F32 panY;
		oa::F32 animTimeMs;
		oa::F32 animDurationMs;
		bool isPanDragging;
		bool isRmbZooming;
		bool isPinching;
		oa::F32 lastMouseX;
		oa::F32 lastMouseY;
		oa::F32 pinchBeginZ;
		oa::F32 pinchAccumScale;
	};

	NavigationConfig config_;

	oa::F32 contentW_ = 1.0F;
	oa::F32 contentH_ = 1.0F;
	oa::F32 windowW_  = 1.0F;
	oa::F32 windowH_  = 1.0F;

	oa::vlm::Vec3 movement_{0.0F, 0.0F, refZ_};
	oa::vlm::Vec3 animStartMovement_{0.0F, 0.0F, refZ_};
	oa::vlm::Vec3 targetMovement_{0.0F, 0.0F, refZ_};

	oa::F32 zoom_ = 1.0F;
	oa::F32 panX_ = 0.0F;
	oa::F32 panY_ = 0.0F;

	oa::F32 animTimeMs_     = 200.0F;
	oa::F32 animDurationMs_ = 200.0F;

	bool  isPanDragging_   = false;
	bool  isRmbZooming_    = false;
	bool  isPinching_      = false;
	oa::F32 lastMouseX_      = 0.0F;
	oa::F32 lastMouseY_      = 0.0F;

	oa::F32 pinchBeginZ_     = refZ_;
	oa::F32 pinchAccumScale_ = 1.0F;

	CaptureFn capturePointer_;

	[[nodiscard]] oa::F32 minZ() const noexcept { return config_.zoomMinZ; }
	[[nodiscard]] oa::F32 maxZ() const noexcept { return config_.maxZoomOutZ; }
	[[nodiscard]] StateSnapshot snapshot() const noexcept;
	void restore(const StateSnapshot& inState) noexcept;

	[[nodiscard]] oa::Status refreshZoom();
	[[nodiscard]] oa::Status syncPanelFromMovement();
	[[nodiscard]] oa::Status syncMovementFromPanel();
	[[nodiscard]] oa::Status panScreenBy(oa::F32 inDx, oa::F32 inDy);
	[[nodiscard]] oa::Status dollyWheel(oa::F32 inScrollDelta);
	[[nodiscard]] oa::Status rmbZoomDrag(oa::F32 inDeltaX);
	[[nodiscard]] oa::Status zoomAt(
		oa::F32 inZoom,
		oa::F32 inAnchorX,
		oa::F32 inAnchorY,
		bool inAnimate);
	[[nodiscard]] oa::Status clampMovement();
	[[nodiscard]] oa::Status clampPanel();
	void beginPanDrag(oa::F32 inMouseX, oa::F32 inMouseY) noexcept;
	[[nodiscard]] oa::Status endPanDrag();
	[[nodiscard]] oa::Status updatePanDrag(const UiEvent& inEvent);
	void beginRmbZoom(oa::F32 inMouseX, oa::F32 inMouseY) noexcept;
	[[nodiscard]] oa::Status endRmbZoom();
	[[nodiscard]] oa::Status beginAnimation(bool inAnimate, oa::F32 inDurationMs);
	static oa::F32 easeOutCubic(oa::F32 inT) noexcept;

	[[nodiscard]] oa::Status handleScroll(const UiEvent& inEvent);
	[[nodiscard]] oa::Status handlePinch(const UiEvent& inEvent);
	[[nodiscard]] oa::Status keyboardZoomBy(oa::F32 inFactor);
};

// ─── Default viewport / viewer keyboard bindings ─────────────────────────────

class InputSystem;

struct NavigationShortcuts {
	UiKey zoomIn     = UiKey::Equals;
	UiKey zoomOut    = UiKey::Minus;
	UiKey zoomFit    = UiKey::Num0;
	UiKey zoomFitAlt = UiKey::F;
	UiKey zoom100    = UiKey::Num9;
	UiKey panUp      = UiKey::Kp8;
	UiKey panDown    = UiKey::Kp2;
	UiKey panLeft    = UiKey::Kp4;
	UiKey panRight   = UiKey::Kp6;
};

[[nodiscard]] inline constexpr const char* navigationHelpLine() noexcept {
	return "  pan=LMB/MMB/wheel/side/2-finger  zoom=RMB/ctrl+wheel/pinch  fit=0/F  9=100%";
}

[[nodiscard]] oa::Status registerViewportShortcuts(
	InputSystem& inInput,
	Navigation& inNav,
	const NavigationShortcuts& inKeys = {});

}  // namespace oa
