//! 2-D viewer navigation (pan / zoom) state machine.
//!
//! Donor: `oa/ui/navigation.h`, `oa/ui/navigation.cpp`.
//!
//! The camera lives in "content space": +X right, +Y down, Z = zoom level.
//! `pan_x` / `pan_y` are screen-pixel offsets from the viewport origin to
//! the top-left corner of the content image.  `zoom` is the scale factor
//! applied to the content before those offsets.

use crate::ui::types::{UiEvent, UiEventKind, UiKey, UiPinchPhase, UiScrollGesture};
use crate::{Error, Result};

// ── Config ────────────────────────────────────────────────────────────────────

/// Navigation tuning knobs. Donor: `NavigationConfig`.
#[derive(Clone, Debug)]
pub struct NavigationConfig {
	pub zoom_min: f32,
	pub zoom_max: f32,
	pub mouse_wheel_sensitivity: f32,
	pub wheel_pan_scale: f32,
	pub ctrl_scroll_dolly_scale: f32,
	pub rmb_zoom_drag_scale: f32,
	pub keyboard_pan_step: f32,
	pub touchpad_pan_scale: f32,
	pub pinch_gesture_scale: f32,
	pub keyboard_zoom_step: f32,
	pub animation_duration_ms: f32,
}

impl Default for NavigationConfig {
	fn default() -> Self {
		Self {
			zoom_min: 0.01,
			zoom_max: 100.0,
			mouse_wheel_sensitivity: 0.001,
			wheel_pan_scale: 50.0,
			ctrl_scroll_dolly_scale: 0.35,
			rmb_zoom_drag_scale: 0.005,
			keyboard_pan_step: 0.5,
			touchpad_pan_scale: 40.0,
			pinch_gesture_scale: 0.09,
			keyboard_zoom_step: 1.05,
			animation_duration_ms: 200.0,
		}
	}
}

// ── Navigation ────────────────────────────────────────────────────────────────

/// 2-D viewport navigation session. Donor: `oa::Navigation`.
pub struct Navigation {
	config: NavigationConfig,
	// Current state (screen pixels).
	pan_x: f32,
	pan_y: f32,
	zoom: f32,
	// Content and viewport dimensions.
	content_w: f32,
	content_h: f32,
	viewport_w: f32,
	viewport_h: f32,
	// Drag state.
	dragging_pan: bool,
	last_drag_x: f32,
	last_drag_y: f32,
	dragging_rmb: bool,
	last_rmb_x: f32,
	// Animation.
	animating: bool,
	anim_elapsed_ms: f32,
	anim_duration_ms: f32,
	anim_start_pan_x: f32,
	anim_start_pan_y: f32,
	anim_start_zoom: f32,
	anim_target_pan_x: f32,
	anim_target_pan_y: f32,
	anim_target_zoom: f32,
	// Pinch state.
	pinch_active: bool,
	pinch_start_zoom: f32,
}

impl Navigation {
	/// Create with default config.
	pub fn new() -> Self {
		Self::with_config(NavigationConfig::default())
	}

	pub fn with_config(config: NavigationConfig) -> Self {
		Self {
			config,
			pan_x: 0.0,
			pan_y: 0.0,
			zoom: 1.0,
			content_w: 0.0,
			content_h: 0.0,
			viewport_w: 0.0,
			viewport_h: 0.0,
			dragging_pan: false,
			last_drag_x: 0.0,
			last_drag_y: 0.0,
			dragging_rmb: false,
			last_rmb_x: 0.0,
			animating: false,
			anim_elapsed_ms: 0.0,
			anim_duration_ms: 0.0,
			anim_start_pan_x: 0.0,
			anim_start_pan_y: 0.0,
			anim_start_zoom: 1.0,
			anim_target_pan_x: 0.0,
			anim_target_pan_y: 0.0,
			anim_target_zoom: 1.0,
			pinch_active: false,
			pinch_start_zoom: 1.0,
		}
	}

	/// Set the content (source image) size.
	pub fn set_content_size(&mut self, w: f32, h: f32) -> Result<()> {
		if !w.is_finite() || !h.is_finite() || w < 0.0 || h < 0.0 {
			return Err(Error::invalid_argument(
				"Navigation content size must be finite and non-negative",
			));
		}
		self.content_w = w;
		self.content_h = h;
		Ok(())
	}

	/// Set the viewport (window) size.
	pub fn set_viewport_size(&mut self, w: f32, h: f32) -> Result<()> {
		if !w.is_finite() || !h.is_finite() || w < 0.0 || h < 0.0 {
			return Err(Error::invalid_argument(
				"Navigation viewport size must be finite and non-negative",
			));
		}
		self.viewport_w = w;
		self.viewport_h = h;
		Ok(())
	}

	// ── Read-back ─────────────────────────────────────────────────────────────

	pub fn pan_x(&self) -> f32 {
		self.pan_x
	}
	pub fn pan_y(&self) -> f32 {
		self.pan_y
	}
	pub fn zoom(&self) -> f32 {
		self.zoom
	}

	// ── Mutations ─────────────────────────────────────────────────────────────

	/// Fit content to the viewport with no animation.
	pub fn fit_to_window(&mut self) -> Result<()> {
		self.fit_to_window_animated(false)
	}

	pub fn fit_to_window_animated(&mut self, animate: bool) -> Result<()> {
		if self.content_w <= 0.0
			|| self.content_h <= 0.0
			|| self.viewport_w <= 0.0
			|| self.viewport_h <= 0.0
		{
			return Ok(());
		}
		let scale = (self.viewport_w / self.content_w)
			.min(self.viewport_h / self.content_h)
			.max(self.config.zoom_min);
		let target_pan_x = (self.viewport_w - self.content_w * scale) * 0.5;
		let target_pan_y = (self.viewport_h - self.content_h * scale) * 0.5;
		if animate {
			self.begin_animation(target_pan_x, target_pan_y, scale);
		} else {
			self.pan_x = target_pan_x;
			self.pan_y = target_pan_y;
			self.zoom = scale;
			self.animating = false;
		}
		Ok(())
	}

	/// Zoom to 1:1 pixel (100%).
	pub fn zoom_to_100(&mut self) {
		let center_x = self.viewport_w * 0.5;
		let center_y = self.viewport_h * 0.5;
		self.zoom_at(1.0, center_x, center_y, false);
	}

	/// Pan by screen-pixel delta.
	pub fn pan_by(&mut self, dx: f32, dy: f32) {
		self.pan_x += dx;
		self.pan_y += dy;
		self.animating = false;
	}

	/// Keyboard directional pan.
	pub fn keyboard_pan(&mut self, dir_x: f32, dir_y: f32) {
		let step = self.config.keyboard_pan_step * self.viewport_w.max(1.0);
		self.pan_by(dir_x * step, dir_y * step);
	}

	/// Keyboard zoom in.
	pub fn keyboard_zoom_in(&mut self) {
		let center_x = self.viewport_w * 0.5;
		let center_y = self.viewport_h * 0.5;
		let new_zoom = (self.zoom * self.config.keyboard_zoom_step)
			.clamp(self.config.zoom_min, self.config.zoom_max);
		self.zoom_at(new_zoom, center_x, center_y, false);
	}

	/// Keyboard zoom out.
	pub fn keyboard_zoom_out(&mut self) {
		let center_x = self.viewport_w * 0.5;
		let center_y = self.viewport_h * 0.5;
		let new_zoom = (self.zoom / self.config.keyboard_zoom_step)
			.clamp(self.config.zoom_min, self.config.zoom_max);
		self.zoom_at(new_zoom, center_x, center_y, false);
	}

	// ── Per-frame ─────────────────────────────────────────────────────────────

	/// Advance animation by `delta_ms` milliseconds.
	pub fn update(&mut self, delta_ms: f32) {
		if !self.animating {
			return;
		}
		self.anim_elapsed_ms = (self.anim_elapsed_ms + delta_ms).min(self.anim_duration_ms);
		let t = if self.anim_duration_ms > 0.0 {
			Self::ease_out_cubic(self.anim_elapsed_ms / self.anim_duration_ms)
		} else {
			1.0
		};
		self.pan_x = lerp(self.anim_start_pan_x, self.anim_target_pan_x, t);
		self.pan_y = lerp(self.anim_start_pan_y, self.anim_target_pan_y, t);
		self.zoom = lerp(self.anim_start_zoom, self.anim_target_zoom, t);
		if self.anim_elapsed_ms >= self.anim_duration_ms {
			self.animating = false;
		}
	}

	/// Route one UiEvent to the navigation logic.
	///
	/// Returns `true` when the event was consumed.
	pub fn handle_event(&mut self, ev: &UiEvent) -> bool {
		match ev.kind {
			UiEventKind::MouseDown => {
				if ev.button == 1 || ev.button == 2 {
					// LMB or MMB → begin pan drag.
					self.dragging_pan = true;
					self.last_drag_x = ev.mouse_x;
					self.last_drag_y = ev.mouse_y;
					self.animating = false;
					return true;
				}
				if ev.button == 3 {
					// RMB → begin zoom drag.
					self.dragging_rmb = true;
					self.last_rmb_x = ev.mouse_x;
					self.animating = false;
					return true;
				}
			}
			UiEventKind::MouseUp => {
				if ev.button == 1 || ev.button == 2 {
					self.dragging_pan = false;
					return false;
				}
				if ev.button == 3 {
					self.dragging_rmb = false;
					return false;
				}
			}
			UiEventKind::MouseMove => {
				if self.dragging_pan {
					let dx = ev.mouse_x - self.last_drag_x;
					let dy = ev.mouse_y - self.last_drag_y;
					self.last_drag_x = ev.mouse_x;
					self.last_drag_y = ev.mouse_y;
					self.pan_by(dx, dy);
					return true;
				}
				if self.dragging_rmb {
					let dx = ev.mouse_x - self.last_rmb_x;
					self.last_rmb_x = ev.mouse_x;
					let factor = 1.0 + dx * self.config.rmb_zoom_drag_scale;
					let new_zoom = (self.zoom * factor).clamp(self.config.zoom_min, self.config.zoom_max);
					let cx = self.viewport_w * 0.5;
					let cy = self.viewport_h * 0.5;
					self.zoom_at(new_zoom, cx, cy, false);
					return true;
				}
			}
			UiEventKind::MouseScroll => {
				match ev.scroll_gesture {
					UiScrollGesture::MouseWheel => {
						if ev.ctrl() {
							// ctrl+wheel → zoom at cursor.
							let delta = ev.scroll_y * self.config.ctrl_scroll_dolly_scale;
							let factor = 1.0 + delta;
							let new_zoom = (self.zoom * factor).clamp(self.config.zoom_min, self.config.zoom_max);
							self.zoom_at(new_zoom, ev.mouse_x, ev.mouse_y, false);
						} else {
							// wheel → pan.
							let step = self.config.wheel_pan_scale;
							self.pan_by(-ev.scroll_x * step, -ev.scroll_y * step);
						}
						return true;
					}
					UiScrollGesture::TouchpadPan => {
						let scale = self.config.touchpad_pan_scale;
						self.pan_by(ev.scroll_x * scale, ev.scroll_y * scale);
						return true;
					}
					UiScrollGesture::PinchScroll => {
						let delta = ev.scroll_y * self.config.ctrl_scroll_dolly_scale;
						let factor = 1.0 + delta;
						let new_zoom = (self.zoom * factor).clamp(self.config.zoom_min, self.config.zoom_max);
						self.zoom_at(new_zoom, ev.mouse_x, ev.mouse_y, false);
						return true;
					}
					UiScrollGesture::None => {}
				}
			}
			UiEventKind::Pinch => {
				match ev.pinch_phase {
					UiPinchPhase::Begin => {
						self.pinch_active = true;
						self.pinch_start_zoom = self.zoom;
					}
					UiPinchPhase::Update => {
						if self.pinch_active {
							let scale = 1.0 + (ev.gesture_scale - 1.0) * self.config.pinch_gesture_scale;
							let new_zoom =
								(self.pinch_start_zoom * scale).clamp(self.config.zoom_min, self.config.zoom_max);
							let cx = self.viewport_w * 0.5;
							let cy = self.viewport_h * 0.5;
							self.zoom_at(new_zoom, cx, cy, false);
						}
					}
					UiPinchPhase::End | UiPinchPhase::None => {
						self.pinch_active = false;
					}
				}
				return true;
			}
			UiEventKind::KeyDown => {
				if ev.key == UiKey::Equals && !ev.ctrl() {
					self.keyboard_zoom_in();
					return true;
				}
				if ev.key == UiKey::Minus && !ev.ctrl() {
					self.keyboard_zoom_out();
					return true;
				}
				if ev.key == UiKey::Num0 || ev.key == UiKey::F {
					let _ = self.fit_to_window();
					return true;
				}
				if ev.key == UiKey::Num9 {
					self.zoom_to_100();
					return true;
				}
				if ev.key == UiKey::Kp8 {
					self.keyboard_pan(0.0, -1.0);
					return true;
				}
				if ev.key == UiKey::Kp2 {
					self.keyboard_pan(0.0, 1.0);
					return true;
				}
				if ev.key == UiKey::Kp4 {
					self.keyboard_pan(-1.0, 0.0);
					return true;
				}
				if ev.key == UiKey::Kp6 {
					self.keyboard_pan(1.0, 0.0);
					return true;
				}
			}
			_ => {}
		}
		false
	}

	// ── Private ───────────────────────────────────────────────────────────────

	fn zoom_at(&mut self, new_zoom: f32, anchor_x: f32, anchor_y: f32, animate: bool) {
		let clamped = new_zoom.clamp(self.config.zoom_min, self.config.zoom_max);
		let ratio = clamped / self.zoom;
		let new_pan_x = anchor_x - (anchor_x - self.pan_x) * ratio;
		let new_pan_y = anchor_y - (anchor_y - self.pan_y) * ratio;
		if animate {
			self.begin_animation(new_pan_x, new_pan_y, clamped);
		} else {
			self.pan_x = new_pan_x;
			self.pan_y = new_pan_y;
			self.zoom = clamped;
			self.animating = false;
		}
	}

	fn begin_animation(&mut self, target_pan_x: f32, target_pan_y: f32, target_zoom: f32) {
		self.anim_start_pan_x = self.pan_x;
		self.anim_start_pan_y = self.pan_y;
		self.anim_start_zoom = self.zoom;
		self.anim_target_pan_x = target_pan_x;
		self.anim_target_pan_y = target_pan_y;
		self.anim_target_zoom = target_zoom;
		self.anim_elapsed_ms = 0.0;
		self.anim_duration_ms = self.config.animation_duration_ms;
		self.animating = true;
	}

	fn ease_out_cubic(t: f32) -> f32 {
		let t = t.clamp(0.0, 1.0);
		1.0 - (1.0 - t).powi(3)
	}
}

impl Default for Navigation {
	fn default() -> Self {
		Self::new()
	}
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
	a + (b - a) * t
}
