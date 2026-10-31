//! Client-frame geometry in SDL logical pixels.
//! Donor: oacpp/source/cpp/lib/oa/ui/windowDecoration.h (mechanical Rust port).

use sdl3::video::HitTestResult;

pub(crate) const TITLE_HEIGHT: i32 = 43;
pub(crate) const BUTTON_SIZE: i32 = 34;
pub(crate) const OUTER_MARGIN: i32 = 6;
pub(crate) const MENU_SPACING: i32 = 6;
pub(crate) const CONTROL_SPACING: i32 = 3;
pub(crate) const CORNER_RADIUS: f32 = 15.0;

pub(crate) fn control_at(x: i32, y: i32, width: i32) -> bool {
	if width <= 0 || x < 0 || x >= width || !(OUTER_MARGIN..OUTER_MARGIN + BUTTON_SIZE).contains(&y) {
		return false;
	}
	if (OUTER_MARGIN..OUTER_MARGIN + BUTTON_SIZE).contains(&x) {
		return true;
	}
	let close = width - OUTER_MARGIN - BUTTON_SIZE;
	let maximize = close - CONTROL_SPACING - BUTTON_SIZE;
	let minimize = maximize - CONTROL_SPACING - BUTTON_SIZE;
	let menu = minimize - MENU_SPACING - BUTTON_SIZE;
	[menu, minimize, maximize, close]
		.iter()
		.any(|begin| (*begin..*begin + BUTTON_SIZE).contains(&x))
}

pub(crate) fn hit_test(
	x: i32,
	y: i32,
	width: i32,
	height: i32,
	resizable: bool,
	maximized: bool,
) -> HitTestResult {
	use HitTestResult::*;
	if width <= 0 || height <= 0 || x < 0 || y < 0 || x >= width || y >= height {
		return Normal;
	}
	if resizable && !maximized {
		let left = x < 7;
		let right = x >= width - 7;
		let top = y < 7;
		let bottom = y >= height - 7;
		let corner_left = x < 12;
		let corner_right = x >= width - 12;
		let corner_top = y < 12;
		let corner_bottom = y >= height - 12;
		if corner_left && corner_top {
			return ResizeTopLeft;
		}
		if corner_right && corner_top {
			return ResizeTopRight;
		}
		if corner_right && corner_bottom {
			return ResizeBottomRight;
		}
		if corner_left && corner_bottom {
			return ResizeBottomLeft;
		}
		if top {
			return ResizeTop;
		}
		if right {
			return ResizeRight;
		}
		if bottom {
			return ResizeBottom;
		}
		if left {
			return ResizeLeft;
		}
	}
	if y < TITLE_HEIGHT && !control_at(x, y, width) {
		Draggable
	} else {
		Normal
	}
}

#[cfg(test)]
#[path = "../../../../test/rs/runtime/window_decoration_unit.rs"]
mod tests;
