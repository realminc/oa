// Ui — caller-owned canvas navigation, coordinate conversion, and hit tests.
//
// Coordinate spaces:
//   World space  — the logical graph layout (float, origin at canvas center)
//   Screen space — canvas-local pixels (origin top-left, Y down)
//
// OA owns no graph, node storage, selection, or document policy. Hit-test
// geometry is borrowed for the duration of each query.

#pragma once

#include <oa/core/vlm.h>
#include <oa/core/status.h>
#include <oa/core/std/scalarMath.h>


// ─── PixelRect ──────────────────────────────────────────────────────────────

namespace oa {

struct PixelRect {
	oa::I32 x = 0;
	oa::I32 y = 0;
	oa::I32 w = 0;
	oa::I32 h = 0;

	[[nodiscard]] constexpr bool isValid() const noexcept {
		return w > 0 and h > 0;
	}

	[[nodiscard]] constexpr bool contains(oa::F32 inPx, oa::F32 inPy) const noexcept {
		return isValid()
			and static_cast<oa::F64>(inPx) >= static_cast<oa::F64>(x)
			and static_cast<oa::F64>(inPx)
				< static_cast<oa::F64>(static_cast<oa::I64>(x) + w)
			and static_cast<oa::F64>(inPy) >= static_cast<oa::F64>(y)
			and static_cast<oa::F64>(inPy)
				< static_cast<oa::F64>(static_cast<oa::I64>(y) + h);
	}

	[[nodiscard]] constexpr bool intersects(const PixelRect& inOther) const noexcept {
		return isValid() and inOther.isValid()
			and static_cast<oa::I64>(x)
				< static_cast<oa::I64>(inOther.x) + inOther.w
			and static_cast<oa::I64>(x) + w
				> static_cast<oa::I64>(inOther.x)
			and static_cast<oa::I64>(y)
				< static_cast<oa::I64>(inOther.y) + inOther.h
			and static_cast<oa::I64>(y) + h
				> static_cast<oa::I64>(inOther.y);
	}
};


// ─── WorldAabb ──────────────────────────────────────────────────────────────

struct WorldAabb {
	oa::vlm::Vec2 min;
	oa::vlm::Vec2 max;

	[[nodiscard]] bool isValid() const noexcept {
		return oa::isFinite(min.x) and oa::isFinite(min.y)
			and oa::isFinite(max.x) and oa::isFinite(max.y)
			and min.x <= max.x and min.y <= max.y;
	}

	[[nodiscard]] bool contains(oa::vlm::Vec2 inP) const noexcept {
		return isValid() and oa::isFinite(inP.x) and oa::isFinite(inP.y)
			and inP.x >= min.x and inP.x <= max.x
		   and inP.y >= min.y and inP.y <= max.y;
	}

	[[nodiscard]] bool intersects(const WorldAabb& inOther) const noexcept {
		return isValid() and inOther.isValid()
			and min.x <= inOther.max.x and max.x >= inOther.min.x
			and min.y <= inOther.max.y and max.y >= inOther.min.y;
	}
};


// ─── Caller-borrowed hit-test records ────────────────────────────────────────

struct NodeCanvasHitItem {
	oa::U64 id = 0U;
	WorldAabb bounds;
	oa::I32 layer = 0;
	bool enabled = true;
};

struct NodeCanvasHit {
	oa::U64 id = 0U;
	oa::Usize index = 0U;
	oa::I32 layer = 0;
	oa::vlm::Vec2 worldPoint;
};

struct NodeCanvasGrid {
	oa::F32 minorWorldStep = 0.0F;
	oa::F32 minorScreenStep = 0.0F;
	oa::vlm::Vec2 originScreen;
	oa::U32 majorEvery = 10U;
	oa::U32 superMajorEvery = 100U;
};


// ─── NodeCanvas ─────────────────────────────────────────────────────────────

struct NodeCanvasState {
	oa::vlm::Vec2 pan       = {0.0F, 0.0F};  // world-space origin of viewport
	oa::F32  zoom      = 1.0F;           // pixels per world unit
	oa::vlm::Vec2 viewSize  = {0.0F, 0.0F};  // viewport dimensions (screen pixels)
};

class NodeCanvas {
public:
	NodeCanvas() = default;

	// ── Coordinate transforms ─────────────────────────────────────────────────

	[[nodiscard]] oa::vlm::Vec2 worldToScreen(oa::vlm::Vec2 inWorld) const noexcept;
	[[nodiscard]] oa::vlm::Vec2 screenToWorld(oa::vlm::Vec2 inScreen) const noexcept;
	[[nodiscard]] WorldAabb visibleWorldRect() const noexcept;
	[[nodiscard]] oa::Result<oa::Optional<NodeCanvasHit>> hitTest(
		oa::vlm::Vec2 inScreen,
		oa::Span<const NodeCanvasHitItem> inItems) const;
	[[nodiscard]] oa::Result<NodeCanvasGrid> grid(
		oa::F32 inMinimumScreenSpacing = 8.0F,
		oa::F32 inBaseWorldStep = 10.0F,
		oa::U32 inMajorEvery = 10U,
		oa::U32 inSuperMajorEvery = 100U) const;

	// ── input: update pan/zoom from raw deltas ────────────────────────────────

	[[nodiscard]] oa::Status setState(const NodeCanvasState& inState);
	[[nodiscard]] oa::Status pan(oa::vlm::Vec2 inDeltaScreen);
	// zoom centered on inFocusScreen (screen pixel).
	[[nodiscard]] oa::Status zoomAt(oa::F32 inFactor, oa::vlm::Vec2 inFocusScreen);
	// Animate pan/zoom to fit all nodes.  call every frame until isAnimating().
	[[nodiscard]] oa::Status fitToView(
		const WorldAabb& inBounds,
		oa::F32 inPaddingPixels = 32.0F);
	[[nodiscard]] bool isAnimating() const noexcept { return animating_; }
	[[nodiscard]] oa::Status stepAnimation(oa::F32 inDeltaMs);

	[[nodiscard]] oa::Status setViewSize(oa::F32 inW, oa::F32 inH);

	[[nodiscard]] const NodeCanvasState& state() const noexcept { return state_; }

	static constexpr oa::F32 kZoomMin = 0.05F;
	static constexpr oa::F32 kZoomMax = 8.0F;

private:
	NodeCanvasState state_;
	NodeCanvasState animStart_;
	NodeCanvasState animTarget_;
	bool              animating_ = false;
	oa::F32             animT_     = 0.0F;
};

}  // namespace oa
