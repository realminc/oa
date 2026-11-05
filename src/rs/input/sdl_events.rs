//! SDL3 → `InputEvent` translation.
//!
//! Donor: `convertSdlEvent` in `oa/ui/viewerApplication.cpp`.
//!
//! The pixel-scale factor is queried once from the window on each call so DPI
//! changes (high-density monitors, live rescale) are reflected immediately.
//! Pointer coordinates are always returned in physical pixels to match the
//! compositor's pixel-space contract.

pub use super::types::{
	INPUT_MOD_ALT, INPUT_MOD_CTRL, INPUT_MOD_SHIFT, INPUT_MOD_SUPER, InputEvent, InputEventKind,
	InputModifiers, Key, PinchPhase, ScrollGesture,
};
use sdl3::event::Event as SdlEvent;
use sdl3::keyboard::Scancode;
use sdl3::sys::events::{
	SDL_EVENT_PINCH_BEGIN, SDL_EVENT_PINCH_END, SDL_EVENT_PINCH_UPDATE, SDL_Event,
};

// ── Modifier helpers ──────────────────────────────────────────────────────────

fn sdl_modifiers(sdl_mods: sdl3::keyboard::Mod) -> InputModifiers {
	let mut m = 0u32;
	if sdl_mods.intersects(sdl3::keyboard::Mod::LSHIFTMOD | sdl3::keyboard::Mod::RSHIFTMOD) {
		m |= INPUT_MOD_SHIFT;
	}
	if sdl_mods.intersects(sdl3::keyboard::Mod::LCTRLMOD | sdl3::keyboard::Mod::RCTRLMOD) {
		m |= INPUT_MOD_CTRL;
	}
	if sdl_mods.intersects(sdl3::keyboard::Mod::LALTMOD | sdl3::keyboard::Mod::RALTMOD) {
		m |= INPUT_MOD_ALT;
	}
	if sdl_mods.intersects(sdl3::keyboard::Mod::LGUIMOD | sdl3::keyboard::Mod::RGUIMOD) {
		m |= INPUT_MOD_SUPER;
	}
	m
}

fn scancode_to_key(sc: Scancode) -> Key {
	Key::new_from_sdl_scancode(sc as u32)
}

// ── Scroll classification ─────────────────────────────────────────────────────

/// Classify a scroll event. Stateless: `scroll_x`/`scroll_y` arrive in
/// normalized traditional units (`+y` = away/up, `+x` = right) because the
/// conversion arm negates `SDL_MOUSEWHEEL_FLIPPED` payloads, so gesture kind
/// only selects semantics and sensitivity. `ctrl+scroll` reports
/// `PinchScroll` — Wayland compositors without pointer-gesture support map
/// pinch that way.
fn scroll_gesture(ev: &InputEvent) -> ScrollGesture {
	if ev.scroll_x.abs() < 0.001 && ev.scroll_y.abs() < 0.001 {
		return ScrollGesture::None;
	}
	if ev.ctrl() {
		return ScrollGesture::PinchScroll;
	}
	if ev.integer_scroll_x != 0 || ev.integer_scroll_y != 0 {
		return ScrollGesture::MouseWheel;
	}
	ScrollGesture::TouchpadPan
}

// ── Pinch gesture bridge ──────────────────────────────────────────────────────
//
// The sdl3 safe `Event` enum predates `SDL_EVENT_PINCH_*`: SDL 3.4 delivers
// Wayland pointer-gesture pinches as `Event::Unknown` and drops the scale. An
// SDL event watch preserves the payload in arrival order, and the next
// `Unknown` of a matching type is mapped back to `InputEvent::Pinch`. The watch
// runs on the thread that pumps events, so the queue is thread-local.

thread_local! {
	static PINCH_QUEUE: std::cell::RefCell<std::collections::VecDeque<(u32, f32)>> =
		const { std::cell::RefCell::new(std::collections::VecDeque::new()) };
}

fn is_pinch_type(ty: u32) -> bool {
	ty == SDL_EVENT_PINCH_BEGIN.0 || ty == SDL_EVENT_PINCH_UPDATE.0 || ty == SDL_EVENT_PINCH_END.0
}

unsafe extern "C" fn pinch_watch_filter(
	_userdata: *mut std::ffi::c_void,
	event: *mut SDL_Event,
) -> bool {
	// SAFETY: SDL passes a valid event; `pinch` is read only for pinch types.
	let ty = unsafe { (*event).r#type };
	if is_pinch_type(ty) {
		let scale = unsafe { (*event).pinch.scale };
		PINCH_QUEUE.with(|q| q.borrow_mut().push_back((ty, scale)));
	}
	true
}

fn install_pinch_watch() {
	static ONCE: std::sync::Once = std::sync::Once::new();
	ONCE.call_once(|| {
		// SAFETY: process-lifetime watch; the filter is a static function.
		unsafe {
			sdl3::sys::events::SDL_AddEventWatch(Some(pinch_watch_filter), std::ptr::null_mut());
		}
	});
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

/// Convert one SDL3 event into a `InputEvent`.
///
/// `pixel_scale_x` / `pixel_scale_y` convert logical SDL coordinates to
/// physical pixels. Obtain them with `window_pixel_scale` before calling this.
/// Returns `None` for SDL events that have no UI representation (timers, audio
/// device changes, etc.).
pub fn convert_sdl_event(
	ev: &SdlEvent,
	pixel_scale_x: f32,
	pixel_scale_y: f32,
) -> Option<InputEvent> {
	install_pinch_watch();
	let px = |x: f32| x * pixel_scale_x;
	let py = |y: f32| y * pixel_scale_y;

	match ev {
		// ── Pointer ──────────────────────────────────────────────────────────
		SdlEvent::MouseMotion {
			x, y, xrel, yrel, ..
		} => Some(InputEvent {
			kind: InputEventKind::MouseMove,
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
			Some(InputEvent {
				kind: InputEventKind::MouseDown,
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
			Some(InputEvent {
				kind: InputEventKind::MouseUp,
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
			direction,
			mouse_x,
			mouse_y,
			timestamp,
			..
		} => {
			// Normalize to traditional units: `+y` = away/up, `+x` = right.
			// SDL already negates the deltas when the platform uses natural
			// scrolling (direction == Flipped), so recover the physical
			// gesture direction here and let each consumer pick its own
			// content-vs-view semantics.
			let flipped = matches!(direction, sdl3::mouse::MouseWheelDirection::Flipped);
			let (scroll_x, scroll_y) = if flipped { (-*x, -*y) } else { (*x, *y) };
			let (integer_scroll_x, integer_scroll_y) = if flipped {
				(-*integer_x, -*integer_y)
			} else {
				(*integer_x, *integer_y)
			};
			let mods = current_modifiers();
			let mut event = InputEvent {
				kind: InputEventKind::MouseScroll,
				mouse_x: px(*mouse_x),
				mouse_y: py(*mouse_y),
				scroll_x,
				scroll_y,
				integer_scroll_x,
				integer_scroll_y,
				timestamp_ns: *timestamp,
				scroll_gesture: ScrollGesture::None,
				gesture_scale: 1.0,
				text_selection_start: -1,
				text_selection_length: -1,
				modifiers: mods,
				..Default::default()
			};
			event.scroll_gesture = scroll_gesture(&event);
			Some(event)
		}
		// ── Keyboard ─────────────────────────────────────────────────────────
		SdlEvent::KeyDown {
			scancode,
			keymod,
			repeat,
			..
		} => {
			let key = scancode.map(scancode_to_key).unwrap_or(Key::Unknown);
			let mods = sdl_modifiers(*keymod);
			Some(InputEvent {
				kind: InputEventKind::KeyDown,
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
			let key = scancode.map(scancode_to_key).unwrap_or(Key::Unknown);
			let mods = sdl_modifiers(*keymod);
			Some(InputEvent {
				kind: InputEventKind::KeyUp,
				key,
				modifiers: mods,
				gesture_scale: 1.0,
				text_selection_start: -1,
				text_selection_length: -1,
				..Default::default()
			})
		}
		SdlEvent::TextInput { text, .. } => Some(InputEvent {
			kind: InputEventKind::KeyChar,
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
		} => Some(InputEvent {
			kind: InputEventKind::TextEditing,
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
		} => Some(InputEvent {
			kind: InputEventKind::WindowResize,
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
			Some(InputEvent {
				kind: InputEventKind::WindowResize,
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
		} => Some(InputEvent {
			kind: InputEventKind::MouseEnter,
			gesture_scale: 1.0,
			text_selection_start: -1,
			text_selection_length: -1,
			..Default::default()
		}),
		SdlEvent::Window {
			win_event: sdl3::event::WindowEvent::MouseLeave,
			..
		} => Some(InputEvent {
			kind: InputEventKind::MouseLeave,
			gesture_scale: 1.0,
			text_selection_start: -1,
			text_selection_length: -1,
			..Default::default()
		}),
		SdlEvent::Window {
			win_event: sdl3::event::WindowEvent::FocusGained,
			..
		} => Some(InputEvent {
			kind: InputEventKind::WindowFocus,
			gesture_scale: 1.0,
			text_selection_start: -1,
			text_selection_length: -1,
			..Default::default()
		}),
		SdlEvent::Window {
			win_event: sdl3::event::WindowEvent::FocusLost,
			..
		} => Some(InputEvent {
			kind: InputEventKind::WindowBlur,
			gesture_scale: 1.0,
			text_selection_start: -1,
			text_selection_length: -1,
			..Default::default()
		}),
		// ── Pinch (X11/older backends) ────────────────────────────────────────
		// MultiGesture carries only a per-event distance delta normalized by
		// screen size; there is no phase signal. Emit incremental Updates —
		// `1.0 + d_dist` is the ratio since the previous event.
		SdlEvent::MultiGesture {
			d_dist,
			num_fingers,
			..
		} if *num_fingers >= 2 => Some(InputEvent {
			kind: InputEventKind::Pinch,
			pinch_phase: PinchPhase::Update,
			gesture_scale: 1.0 + d_dist,
			text_selection_start: -1,
			text_selection_length: -1,
			modifiers: current_modifiers(),
			..Default::default()
		}),
		// ── Pinch gestures (Wayland pointer_gestures) ────────────────────────
		// Arrive as `Unknown` in the safe enum; the event watch recovers the
		// scale in queue order.
		SdlEvent::Unknown { type_, .. } if is_pinch_type(*type_) => {
			let scale = PINCH_QUEUE
				.with(|q| {
					let mut q = q.borrow_mut();
					match q.front() {
						Some((ty, _)) if ty == type_ => q.pop_front().map(|(_, s)| s),
						_ => None,
					}
				})
				.unwrap_or(1.0);
			let phase = if *type_ == SDL_EVENT_PINCH_BEGIN.0 {
				PinchPhase::Begin
			} else if *type_ == SDL_EVENT_PINCH_END.0 {
				PinchPhase::End
			} else {
				PinchPhase::Update
			};
			Some(InputEvent {
				kind: InputEventKind::Pinch,
				pinch_phase: phase,
				gesture_scale: scale,
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
fn current_modifiers() -> InputModifiers {
	// SAFETY: SDL_GetModState is safe to call from any thread while SDL is initialised.
	let raw = unsafe { sdl3::sys::keyboard::SDL_GetModState() };
	// sdl3::keyboard::Mod is a bitflags wrapping the SDL3 keymod integer.
	let mods = sdl3::keyboard::Mod::from_bits_truncate(raw.0);
	sdl_modifiers(mods)
}
