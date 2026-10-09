//! SDL3 → `UiEvent` translation.
//!
//! Donor: `convertSdlEvent` in `oa/ui/viewerApplication.cpp`.
//!
//! The pixel-scale factor is queried once from the window on each call so DPI
//! changes (high-density monitors, live rescale) are reflected immediately.
//! Pointer coordinates are always returned in physical pixels to match the
//! compositor's pixel-space contract.

use sdl3::event::Event as SdlEvent;
use sdl3::keyboard::Scancode;

use super::types::{
	UI_MOD_ALT, UI_MOD_CTRL, UI_MOD_SHIFT, UI_MOD_SUPER, UiEvent, UiEventKind, UiKey, UiModifiers,
	UiPinchPhase, classify_scroll,
};

// ── Modifier helpers ──────────────────────────────────────────────────────────

fn sdl_modifiers(sdl_mods: sdl3::keyboard::Mod) -> UiModifiers {
	let mut m = 0u32;
	if sdl_mods.intersects(sdl3::keyboard::Mod::LSHIFTMOD | sdl3::keyboard::Mod::RSHIFTMOD) {
		m |= UI_MOD_SHIFT;
	}
	if sdl_mods.intersects(sdl3::keyboard::Mod::LCTRLMOD | sdl3::keyboard::Mod::RCTRLMOD) {
		m |= UI_MOD_CTRL;
	}
	if sdl_mods.intersects(sdl3::keyboard::Mod::LALTMOD | sdl3::keyboard::Mod::RALTMOD) {
		m |= UI_MOD_ALT;
	}
	if sdl_mods.intersects(sdl3::keyboard::Mod::LGUIMOD | sdl3::keyboard::Mod::RGUIMOD) {
		m |= UI_MOD_SUPER;
	}
	m
}

fn scancode_to_key(sc: Scancode) -> UiKey {
	UiKey::from_sdl_scancode(sc as u32)
}

// ── Pixel-scale helpers ───────────────────────────────────────────────────────

/// Compute logical→physical pixel scale for one window.
///
/// Returns `(scale_x, scale_y)`.  Both default to `1.0` when the window is
/// zero-size or the query fails.
pub fn window_pixel_scale(window: &sdl3::video::Window) -> (f32, f32) {
	let (lw, lh) = window.size();
	let (pw, ph) = window.size_in_pixels();
	let sx = if lw > 0 { pw as f32 / lw as f32 } else { 1.0 };
	let sy = if lh > 0 { ph as f32 / lh as f32 } else { 1.0 };
	(sx, sy)
}

// ── SDL event conversion ──────────────────────────────────────────────────────

/// Convert one SDL3 event into a `UiEvent`.
///
/// `pixel_scale_x` / `pixel_scale_y` convert logical SDL coordinates to
/// physical pixels. Obtain them with `window_pixel_scale` before calling this.
/// Returns `None` for SDL events that have no UI representation (timers, audio
/// device changes, etc.).
pub fn convert_sdl_event(ev: &SdlEvent, pixel_scale_x: f32, pixel_scale_y: f32) -> Option<UiEvent> {
	let px = |x: f32| x * pixel_scale_x;
	let py = |y: f32| y * pixel_scale_y;

	match ev {
		// ── Pointer ──────────────────────────────────────────────────────────
		SdlEvent::MouseMotion {
			x, y, xrel, yrel, ..
		} => Some(UiEvent {
			kind: UiEventKind::MouseMove,
			mouse_x: px(*x),
			mouse_y: py(*y),
			mouse_dx: px(*xrel),
			mouse_dy: py(*yrel),
			gesture_scale: 1.0,
			text_selection_start: -1,
			text_selection_length: -1,
			modifiers: current_modifiers(),
			..Default::default()
		}),
		SdlEvent::MouseButtonDown {
			x,
			y,
			mouse_btn,
			clicks,
			..
		} => {
			let button = sdl_button_id(*mouse_btn);
			Some(UiEvent {
				kind: UiEventKind::MouseDown,
				mouse_x: px(*x),
				mouse_y: py(*y),
				button,
				click_count: *clicks as i32,
				gesture_scale: 1.0,
				text_selection_start: -1,
				text_selection_length: -1,
				modifiers: current_modifiers(),
				..Default::default()
			})
		}
		SdlEvent::MouseButtonUp {
			x, y, mouse_btn, ..
		} => {
			let button = sdl_button_id(*mouse_btn);
			Some(UiEvent {
				kind: UiEventKind::MouseUp,
				mouse_x: px(*x),
				mouse_y: py(*y),
				button,
				gesture_scale: 1.0,
				text_selection_start: -1,
				text_selection_length: -1,
				modifiers: current_modifiers(),
				..Default::default()
			})
		}
		SdlEvent::MouseWheel {
			x,
			y,
			integer_x,
			integer_y,
			mouse_x,
			mouse_y,
			timestamp,
			..
		} => {
			let mods = current_modifiers();
			let gesture = classify_scroll(*integer_x, *integer_y, mods);
			Some(UiEvent {
				kind: UiEventKind::MouseScroll,
				mouse_x: px(*mouse_x),
				mouse_y: py(*mouse_y),
				scroll_x: *x,
				scroll_y: *y,
				integer_scroll_x: *integer_x,
				integer_scroll_y: *integer_y,
				timestamp_ns: *timestamp,
				scroll_gesture: gesture,
				gesture_scale: 1.0,
				text_selection_start: -1,
				text_selection_length: -1,
				modifiers: mods,
				..Default::default()
			})
		}
		// ── Keyboard ─────────────────────────────────────────────────────────
		SdlEvent::KeyDown {
			scancode,
			keymod,
			repeat,
			..
		} => {
			let key = scancode.map(scancode_to_key).unwrap_or(UiKey::Unknown);
			let mods = sdl_modifiers(*keymod);
			Some(UiEvent {
				kind: UiEventKind::KeyDown,
				key,
				modifiers: mods,
				key_repeat: *repeat,
				gesture_scale: 1.0,
				text_selection_start: -1,
				text_selection_length: -1,
				..Default::default()
			})
		}
		SdlEvent::KeyUp {
			scancode, keymod, ..
		} => {
			let key = scancode.map(scancode_to_key).unwrap_or(UiKey::Unknown);
			let mods = sdl_modifiers(*keymod);
			Some(UiEvent {
				kind: UiEventKind::KeyUp,
				key,
				modifiers: mods,
				gesture_scale: 1.0,
				text_selection_start: -1,
				text_selection_length: -1,
				..Default::default()
			})
		}
		SdlEvent::TextInput { text, .. } => Some(UiEvent {
			kind: UiEventKind::KeyChar,
			text: text.clone(),
			gesture_scale: 1.0,
			text_selection_start: -1,
			text_selection_length: -1,
			modifiers: current_modifiers(),
			..Default::default()
		}),
		SdlEvent::TextEditing {
			text,
			start,
			length,
			..
		} => Some(UiEvent {
			kind: UiEventKind::TextEditing,
			text: text.clone(),
			text_selection_start: *start,
			text_selection_length: *length,
			gesture_scale: 1.0,
			modifiers: current_modifiers(),
			..Default::default()
		}),
		// ── Window ────────────────────────────────────────────────────────────
		SdlEvent::Window {
			win_event: sdl3::event::WindowEvent::PixelSizeChanged(w, h),
			..
		} => Some(UiEvent {
			kind: UiEventKind::WindowResize,
			window_w: *w,
			window_h: *h,
			gesture_scale: 1.0,
			text_selection_start: -1,
			text_selection_length: -1,
			..Default::default()
		}),
		SdlEvent::Window {
			win_event: sdl3::event::WindowEvent::Resized(w, h),
			..
		} => {
			// Resized fires in logical pixels; convert to physical.
			let pw = (*w as f32 * pixel_scale_x) as i32;
			let ph = (*h as f32 * pixel_scale_y) as i32;
			Some(UiEvent {
				kind: UiEventKind::WindowResize,
				window_w: pw,
				window_h: ph,
				gesture_scale: 1.0,
				text_selection_start: -1,
				text_selection_length: -1,
				..Default::default()
			})
		}
		SdlEvent::Window {
			win_event: sdl3::event::WindowEvent::MouseEnter,
			..
		} => Some(UiEvent {
			kind: UiEventKind::MouseEnter,
			gesture_scale: 1.0,
			text_selection_start: -1,
			text_selection_length: -1,
			..Default::default()
		}),
		SdlEvent::Window {
			win_event: sdl3::event::WindowEvent::MouseLeave,
			..
		} => Some(UiEvent {
			kind: UiEventKind::MouseLeave,
			gesture_scale: 1.0,
			text_selection_start: -1,
			text_selection_length: -1,
			..Default::default()
		}),
		SdlEvent::Window {
			win_event: sdl3::event::WindowEvent::FocusGained,
			..
		} => Some(UiEvent {
			kind: UiEventKind::WindowFocus,
			gesture_scale: 1.0,
			text_selection_start: -1,
			text_selection_length: -1,
			..Default::default()
		}),
		SdlEvent::Window {
			win_event: sdl3::event::WindowEvent::FocusLost,
			..
		} => Some(UiEvent {
			kind: UiEventKind::WindowBlur,
			gesture_scale: 1.0,
			text_selection_start: -1,
			text_selection_length: -1,
			..Default::default()
		}),
		// ── Pinch ─────────────────────────────────────────────────────────────
		// SDL3 pinch events are exposed through MultiGesture; map them directly.
		SdlEvent::MultiGesture {
			d_dist,
			num_fingers,
			..
		} if *num_fingers >= 2 => {
			// A magnitude change > threshold signals a pinch update.
			let phase = if d_dist.abs() > 0.001 {
				UiPinchPhase::Update
			} else {
				UiPinchPhase::Begin
			};
			Some(UiEvent {
				kind: UiEventKind::Pinch,
				pinch_phase: phase,
				gesture_scale: 1.0 + d_dist,
				text_selection_start: -1,
				text_selection_length: -1,
				modifiers: current_modifiers(),
				..Default::default()
			})
		}
		_ => None,
	}
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn sdl_button_id(btn: sdl3::mouse::MouseButton) -> i32 {
	match btn {
		sdl3::mouse::MouseButton::Left => 1,
		sdl3::mouse::MouseButton::Middle => 2,
		sdl3::mouse::MouseButton::Right => 3,
		sdl3::mouse::MouseButton::X1 => 4,
		sdl3::mouse::MouseButton::X2 => 5,
		sdl3::mouse::MouseButton::Unknown => 0,
	}
}

/// Read the current SDL modifier state.
fn current_modifiers() -> UiModifiers {
	// SAFETY: SDL_GetModState is safe to call from any thread while SDL is initialised.
	let raw = unsafe { sdl3::sys::keyboard::SDL_GetModState() };
	// sdl3::keyboard::Mod is a bitflags wrapping the SDL3 keymod integer.
	let mods = sdl3::keyboard::Mod::from_bits_truncate(raw.0);
	sdl_modifiers(mods)
}
