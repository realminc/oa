//! Panel layout state for the UI compositor.
//!
//! Tracks the current cursor position inside the active panel or row so that
//! widget calls can position themselves without explicit coordinates.
//!
//! Donor reference: `oa::Ui` panel/row/cursor state in `ui.cpp`

use super::types::{PixelRect, UiDirection, UiLayout, UiSizing};

/// One active panel scope (pushed by `begin_panel`, popped by `end_panel`).
#[derive(Clone, Debug)]
pub(super) struct PanelScope {
	/// Outer bounds of this panel (physical pixels).
	pub(super) outer: PixelRect,
	/// Inner content area after padding.
	pub(super) inner: PixelRect,
	pub(super) layout: UiLayout,
	/// Running cursor inside `inner`.
	pub(super) cursor: [i32; 2],
	/// Width of the last-placed widget (used by row layout).
	pub(super) row_width: u32,
	/// Effective clip inherited from the parent or a scroll viewport.
	pub(super) clip_override: Option<PixelRect>,
	/// Present only for a scroll-panel scope; coordinates stay in screen space.
	pub(super) scroll_viewport: Option<PixelRect>,
	pub(super) scroll_offset_y: i32,
}

impl PanelScope {
	pub(super) fn new(rect: PixelRect, layout: UiLayout) -> Self {
		let pad = &layout.padding;
		let ix = rect.x + pad.left as i32;
		let iy = rect.y + pad.top as i32;
		let iw = (rect.w as f32 - pad.left - pad.right).max(0.0) as u32;
		let ih = (rect.h as f32 - pad.top - pad.bottom).max(0.0) as u32;
		let inner = PixelRect {
			x: ix,
			y: iy,
			w: iw,
			h: ih,
		};
		Self {
			outer: rect,
			inner,
			cursor: [ix, iy],
			layout,
			row_width: 0,
			clip_override: None,
			scroll_viewport: None,
			scroll_offset_y: 0,
		}
	}

	/// Advance the cursor by `extent` pixels in the flow direction and return
	/// the allocated rect for the next widget.
	pub(super) fn allocate(&mut self, w: u32, h: u32) -> PixelRect {
		let rect = PixelRect {
			x: self.cursor[0],
			y: self.cursor[1],
			w,
			h,
		};
		match self.layout.direction {
			UiDirection::Column => {
				self.cursor[1] += h as i32 + self.layout.gap as i32;
			}
			UiDirection::Row => {
				self.cursor[0] += w as i32 + self.layout.gap as i32;
				if h > self.row_width {
					self.row_width = h;
				}
			}
		}
		rect
	}

	/// Width of the available content area.
	pub(super) fn content_width(&self) -> u32 {
		self.inner.w
	}

	/// Clip rect for children — intersection of viewport and inner.
	pub(super) fn clip(&self) -> PixelRect {
		self.clip_override.unwrap_or(self.outer).clip(&self.inner)
	}
}

/// Stack of panel scopes for nested layout.
#[derive(Default)]
pub(super) struct LayoutStack {
	pub(super) panels: Vec<PanelScope>,
}

impl LayoutStack {
	pub(super) fn push(&mut self, scope: PanelScope) {
		self.panels.push(scope);
	}

	pub(super) fn pop(&mut self) {
		self.panels.pop();
	}

	/// Borrow the innermost panel, if any.
	pub(super) fn top(&self) -> Option<&PanelScope> {
		self.panels.last()
	}

	/// Mutably borrow the innermost panel, if any.
	pub(super) fn top_mut(&mut self) -> Option<&mut PanelScope> {
		self.panels.last_mut()
	}
}

// ── Sizing helpers ────────────────────────────────────────────────────────────

/// Resolve `UiSizing` for a widget width given the parent content width.
pub(super) fn resolve_width(sizing: &UiSizing, parent_content_width: u32) -> u32 {
	match sizing {
		UiSizing::Fill => parent_content_width,
		UiSizing::Fixed(px) => *px as u32,
		UiSizing::Hug => 0, // caller provides explicit content width
	}
}

/// Standard widget height for panels without an explicit height.
pub(super) fn default_item_height(padding: f32) -> u32 {
	(padding * 4.0 + 14.0) as u32
}

/// Emit a single-pixel separator's rect in the flow direction.
pub(super) fn separator_rect(scope: &PanelScope) -> PixelRect {
	match scope.layout.direction {
		UiDirection::Column => PixelRect {
			x: scope.inner.x,
			y: scope.cursor[1],
			w: scope.inner.w,
			h: 1,
		},
		UiDirection::Row => PixelRect {
			x: scope.cursor[0],
			y: scope.inner.y,
			w: 1,
			h: scope.inner.h,
		},
	}
}
