//! Geometric symbolic icons. Donor: `oa::Ui::iconButton`, ui.cpp 4722–4857.
//! Mechanical command adaptation; coordinates, proportions and stroke formulas
//! remain the donor's. Icons never depend on font/Unicode glyph availability.

use super::{
	command::{UiCmd, UiCmdList},
	types::{PixelRect, UiIcon},
};
use crate::Color;

pub(super) fn draw(
	commands: &mut UiCmdList,
	rect: PixelRect,
	bounds: PixelRect,
	kind: UiIcon,
	scale: f32,
	window_control: bool,
	colors: [Color; 2],
) {
	let [icon, transport] = colors;
	let cx = rect.x as f32 + rect.w as f32 * 0.5;
	let cy = rect.y as f32 + rect.h as f32 * 0.5;
	let nominal: f32 = if window_control {
		8.0
	} else if matches!(kind, UiIcon::Play | UiIcon::Pause) {
		32.0
	} else {
		16.0
	};
	let span = (0.5 * rect.w.min(rect.h) as f32 - scale * 2.0)
		.max(3.0)
		.min(nominal * scale * 0.5)
		.max(3.0);
	let thickness = (0.055 * rect.w.min(rect.h) as f32).max(1.5);
	macro_rules! line {
		($x0:expr, $y0:expr, $x1:expr, $y1:expr) => {
			commands.push(UiCmd::Line {
				x0: $x0,
				y0: $y0,
				x1: $x1,
				y1: $y1,
				color: icon,
				thickness,
				bounds,
			});
		};
	}
	macro_rules! outline {
		($x:expr,$y:expr,$w:expr,$h:expr,$radius:expr) => {
			commands.push(UiCmd::RectOutline {
				rect: PixelRect::new($x, $y, $w, $h),
				color: icon,
				thickness: (thickness as u32).max(1),
				clip: bounds,
				corner_radius: $radius,
			});
		};
	}
	match kind {
		UiIcon::Previous | UiIcon::Next => {
			let direction = if kind == UiIcon::Previous { -1.0 } else { 1.0 };
			let tail = cx - direction * span;
			commands.push(UiCmd::Triangle {
				points: [
					[tail, cy - span],
					[cx + direction * span, cy],
					[tail, cy + span],
				],
				color: transport,
				bounds,
			});
		}
		UiIcon::Play => {
			let left = cx - span * 0.62;
			commands.push(UiCmd::Triangle {
				points: [[left, cy - span], [cx + span * 0.82, cy], [left, cy + span]],
				color: transport,
				bounds,
			});
		}
		UiIcon::Pause => {
			let width = (thickness * 1.35).round().max(2.0) as u32;
			let height = (span * 2.0).round().max(4.0) as u32;
			let gap = width.max(2) as i32;
			let top = cy.round() as i32 - height as i32 / 2;
			for x in [
				cx.round() as i32 - gap - width as i32,
				cx.round() as i32 + gap,
			] {
				commands.push(UiCmd::Rect {
					rect: PixelRect::new(x, top, width, height),
					color: transport,
					clip: bounds,
					corner_radius: 0.0,
				});
			}
		}
		UiIcon::Menu => {
			for row in -1..=1 {
				let y = cy + row as f32 * span * 0.72;
				line!(cx - span, y, cx + span, y);
			}
		}
		UiIcon::Minimize => {
			line!(cx - span, cy + span * 0.58, cx + span, cy + span * 0.58);
		}
		UiIcon::Maximize => {
			let size = (span * 2.0).round().max(4.0) as u32;
			outline!(
				cx.round() as i32 - size as i32 / 2,
				cy.round() as i32 - size as i32 / 2,
				size,
				size,
				0.0
			);
		}
		UiIcon::Restore => {
			let size = (span * 1.35).round().max(4.0) as u32;
			let offset = (size / 4).max(2) as i32;
			let x = cx.round() as i32 - size as i32 / 2;
			let y = cy.round() as i32 - size as i32 / 2;
			outline!(x + offset, y - offset, size, size, 0.0);
			outline!(x - offset, y + offset, size, size, 0.0);
		}
		UiIcon::Close => {
			line!(cx - span, cy - span, cx + span, cy + span);
			line!(cx + span, cy - span, cx - span, cy + span);
		}
		UiIcon::Volume | UiIcon::Muted => {
			let left = cx - span;
			let right = cx - span * 0.42;
			let cone = cx + span * 0.18;
			line!(left, cy - span * 0.32, right, cy - span * 0.32);
			line!(left, cy - span * 0.32, left, cy + span * 0.32);
			line!(left, cy + span * 0.32, right, cy + span * 0.32);
			line!(right, cy - span * 0.32, cone, cy - span);
			line!(cone, cy - span, cone, cy + span);
			line!(cone, cy + span, right, cy + span * 0.32);
			if kind == UiIcon::Volume {
				line!(cx + span * 0.46, cy - span * 0.48, cx + span * 0.78, cy);
				line!(cx + span * 0.78, cy, cx + span * 0.46, cy + span * 0.48);
			} else {
				line!(
					cx + span * 0.38,
					cy - span * 0.50,
					cx + span,
					cy + span * 0.50
				);
				line!(
					cx + span,
					cy - span * 0.50,
					cx + span * 0.38,
					cy + span * 0.50
				);
			}
		}
		UiIcon::Settings => {
			let outer = span * 0.92;
			let inner = span * 0.68;
			line!(cx - outer, cy, cx - inner, cy);
			line!(cx + inner, cy, cx + outer, cy);
			line!(cx, cy - outer, cx, cy - inner);
			line!(cx, cy + inner, cx, cy + outer);
			let outer = outer * 0.72;
			let diagonal = inner * 0.72;
			line!(cx - outer, cy - outer, cx - diagonal, cy - diagonal);
			line!(cx + diagonal, cy + diagonal, cx + outer, cy + outer);
			line!(cx + outer, cy - outer, cx + diagonal, cy - diagonal);
			line!(cx - diagonal, cy + diagonal, cx - outer, cy + outer);
			let diameter = (inner * 1.45).round().max(4.0) as u32;
			outline!(
				cx.round() as i32 - diameter as i32 / 2,
				cy.round() as i32 - diameter as i32 / 2,
				diameter,
				diameter,
				diameter as f32 * 0.5
			);
		}
		UiIcon::Fullscreen | UiIcon::ExitFullscreen => {
			let extent = if kind == UiIcon::Fullscreen {
				span
			} else {
				span * 0.72
			};
			let leg = (span * 0.62).max(2.0);
			let left = cx - extent;
			let right = cx + extent;
			let top = cy - extent;
			let bottom = cy + extent;
			line!(left, top + leg, left, top);
			line!(left, top, left + leg, top);
			line!(right - leg, top, right, top);
			line!(right, top, right, top + leg);
			line!(right, bottom - leg, right, bottom);
			line!(right, bottom, right - leg, bottom);
			line!(left + leg, bottom, left, bottom);
			line!(left, bottom, left, bottom - leg);
		}
	}
}
