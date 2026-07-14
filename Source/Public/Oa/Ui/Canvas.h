// OaUi — Node canvas: infinite 2D space with pan, zoom, and hit testing.
//
// Coordinate spaces:
//   World space  — the logical graph layout (float, origin at canvas center)
//   Screen space — window pixels (origin top-left, Y down)
//
// Spatial index: flat 64×64 grid for O(1) hit testing on sparse graphs.

#pragma once

#include <Oa/Core/Vlm.h>


// ─── OaPixelRect ──────────────────────────────────────────────────────────────

struct OaPixelRect {
	OaI32 X = 0;
	OaI32 Y = 0;
	OaI32 W = 0;
	OaI32 H = 0;

	[[nodiscard]] constexpr bool Contains(OaF32 InPx, OaF32 InPy) const noexcept {
		return InPx >= static_cast<OaF32>(X) and InPx < static_cast<OaF32>(X + W)
		   and InPy >= static_cast<OaF32>(Y) and InPy < static_cast<OaF32>(Y + H);
	}

	[[nodiscard]] constexpr bool Intersects(const OaPixelRect& InOther) const noexcept {
		return X < (InOther.X + InOther.W) and (X + W) > InOther.X
		   and Y < (InOther.Y + InOther.H) and (Y + H) > InOther.Y;
	}
};


// ─── OaWorldAABB ──────────────────────────────────────────────────────────────

struct OaWorldAABB {
	VlmVec2 Min;
	VlmVec2 Max;

	[[nodiscard]] constexpr bool Contains(VlmVec2 InP) const noexcept {
		return InP.X >= Min.X and InP.X <= Max.X
		   and InP.Y >= Min.Y and InP.Y <= Max.Y;
	}
};


// ─── OaNodeCanvas ─────────────────────────────────────────────────────────────

struct OaNodeCanvasState {
	VlmVec2 Pan       = {0.0F, 0.0F};  // world-space origin of viewport
	OaF32  Zoom      = 1.0F;           // pixels per world unit
	VlmVec2 ViewSize  = {0.0F, 0.0F};  // viewport dimensions (screen pixels)
};

class OaNodeCanvas {
public:
	OaNodeCanvas() = default;

	// ── Coordinate transforms ─────────────────────────────────────────────────

	[[nodiscard]] VlmVec2 WorldToScreen(VlmVec2 InWorld) const noexcept;
	[[nodiscard]] VlmVec2 ScreenToWorld(VlmVec2 InScreen) const noexcept;
	[[nodiscard]] OaWorldAABB VisibleWorldRect() const noexcept;

	// ── Input: update Pan/Zoom from raw deltas ────────────────────────────────

	void Pan(VlmVec2 InDeltaScreen) noexcept;
	// Zoom centered on InFocusScreen (screen pixel).
	void ZoomAt(OaF32 InFactor, VlmVec2 InFocusScreen) noexcept;
	// Animate Pan/Zoom to fit all nodes.  Call every frame until IsAnimating().
	void FitToView(const OaWorldAABB& InBounds) noexcept;
	[[nodiscard]] bool IsAnimating() const noexcept { return Animating_; }
	void StepAnimation(OaF32 InDeltaMs) noexcept;

	void SetViewSize(OaF32 InW, OaF32 InH) noexcept {
		State_.ViewSize = {InW, InH};
	}

	[[nodiscard]] const OaNodeCanvasState& State() const noexcept { return State_; }

	static constexpr OaF32 kZoomMin = 0.05F;
	static constexpr OaF32 kZoomMax = 8.0F;

private:
	OaNodeCanvasState State_;
	OaNodeCanvasState AnimTarget_;
	bool              Animating_ = false;
	OaF32             AnimT_     = 0.0F;
};
