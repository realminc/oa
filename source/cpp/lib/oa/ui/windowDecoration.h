// Private geometry contract for Viewer client-side window decoration.
//
// Coordinates are SDL logical window pixels. The platform callback maps these
// regions to native compositor move/resize operations; rendering applies the
// window's logical-to-pixel scale separately.

#pragma once

#include <oa/core/types.h>
#include <oa/core/std/scalarMath.h>


namespace oa {

enum class WindowDecorationControl : oa::U8 {
	None = 0,
	Minimize,
	Maximize,
	Close,
};

enum class WindowDecorationHit : oa::U8 {
	Normal = 0,
	Draggable,
	ResizeTopLeft,
	ResizeTop,
	ResizeTopRight,
	ResizeRight,
	ResizeBottomRight,
	ResizeBottom,
	ResizeBottomLeft,
	ResizeLeft,
};

struct WindowDecorationMetrics {
	oa::I32 titleHeight = 36;
	oa::I32 controlWidth = 46;
	oa::I32 resizeBorder = 7;
	oa::I32 resizeCorner = 12;
};

[[nodiscard]] inline oa::I32 windowLogicalSizeForPixels(
	oa::U32 inPixels,
	oa::F32 inPixelScale) noexcept {
	const oa::F32 scale = oa::isFinite(inPixelScale) and inPixelScale > 0.001F
		? inPixelScale : 1.0F;
	return oa::max(
		1,
		static_cast<oa::I32>(oa::ceil(
			static_cast<oa::F32>(inPixels) / scale)));
}

[[nodiscard]] constexpr WindowDecorationControl
windowDecorationControlAt(
	oa::I32 inX,
	oa::I32 inY,
	oa::I32 inWindowWidth,
	const WindowDecorationMetrics& inMetrics = {}) noexcept {
	if (inWindowWidth <= 0 or inX < 0 or inX >= inWindowWidth
		or inY < 0 or inY >= inMetrics.titleHeight) {
		return WindowDecorationControl::None;
	}
	const oa::I32 controlsBegin = inWindowWidth - inMetrics.controlWidth * 3;
	if (inX < controlsBegin) return WindowDecorationControl::None;
	const oa::I32 control = (inX - controlsBegin) / inMetrics.controlWidth;
	if (control == 0) return WindowDecorationControl::Minimize;
	if (control == 1) return WindowDecorationControl::Maximize;
	return WindowDecorationControl::Close;
}

[[nodiscard]] constexpr WindowDecorationHit windowDecorationHitTest(
	oa::I32 inX,
	oa::I32 inY,
	oa::I32 inWindowWidth,
	oa::I32 inWindowHeight,
	bool inResizable,
	bool inMaximized,
	const WindowDecorationMetrics& inMetrics = {}) noexcept {
	if (inWindowWidth <= 0 or inWindowHeight <= 0
		or inX < 0 or inY < 0
		or inX >= inWindowWidth or inY >= inWindowHeight) {
		return WindowDecorationHit::Normal;
	}

	if (inResizable and not inMaximized) {
		const bool left = inX < inMetrics.resizeBorder;
		const bool right = inX >= inWindowWidth - inMetrics.resizeBorder;
		const bool top = inY < inMetrics.resizeBorder;
		const bool bottom = inY >= inWindowHeight - inMetrics.resizeBorder;
		const bool cornerLeft = inX < inMetrics.resizeCorner;
		const bool cornerRight =
			inX >= inWindowWidth - inMetrics.resizeCorner;
		const bool cornerTop = inY < inMetrics.resizeCorner;
		const bool cornerBottom =
			inY >= inWindowHeight - inMetrics.resizeCorner;

		if (cornerLeft and cornerTop) {
			return WindowDecorationHit::ResizeTopLeft;
		}
		if (cornerRight and cornerTop) {
			return WindowDecorationHit::ResizeTopRight;
		}
		if (cornerRight and cornerBottom) {
			return WindowDecorationHit::ResizeBottomRight;
		}
		if (cornerLeft and cornerBottom) {
			return WindowDecorationHit::ResizeBottomLeft;
		}
		if (top) return WindowDecorationHit::ResizeTop;
		if (right) return WindowDecorationHit::ResizeRight;
		if (bottom) return WindowDecorationHit::ResizeBottom;
		if (left) return WindowDecorationHit::ResizeLeft;
	}

	if (inY < inMetrics.titleHeight
		and windowDecorationControlAt(
			inX, inY, inWindowWidth, inMetrics)
			== WindowDecorationControl::None) {
		return WindowDecorationHit::Draggable;
	}
	return WindowDecorationHit::Normal;
}

}  // namespace oa
