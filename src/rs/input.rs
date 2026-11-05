//! `oa::input` — platform-neutral input layer.
//!
//! Donor: `oa/ui/platformInput.h` and `oa/ui/platformInput.cpp` (SDL3 backend),
//! `oa/ui/input.h` and `oa/ui/input.cpp` (action registry).
//!
//! The module owns everything between the platform and consumers:
//!
//! - [`sdl_events`]: SDL3 event translation (pixel scale, wheel-direction
//!   normalization, pinch-gesture bridging) into [`InputEvent`].
//! - Per-frame snapshots: [`process_event`], [`update`], [`down`], [`press`],
//!   [`release`], [`keyboard`], [`mouse`], [`pointer_position`].
//! - [`InputSystem`]: named key-action registry with context dispatch.
//!
//! Per-frame usage:
//!
//! ```ignore
//! oa::input::initialize();
//! // each frame, before clearing deltas:
//! for event in sdl_events { oa::input::process_event(&event); }
//! oa::input::update();
//! let mouse = oa::input::mouse();
//! oa::input::clear_for_next_frame();
//! ```
//!
//! State is thread-local, matching the donor's effectively main-thread SDL
//! event pump: each thread that pumps events owns its own snapshot.

use std::cell::{Cell, RefCell};

use sdl3::event::Event as SdlEvent;

mod actions;
pub mod navigation;
pub mod sdl_events;
#[path = "input/type.gen.rs"]
mod type_gen;
#[path = "input/type.rs"]
pub mod types;

pub use actions::{InputContext, InputSystem, KeyAction, KeyBinding};
pub use navigation::{Navigation, NavigationConfig};
pub use types::{
	INPUT_MOD_ALT, INPUT_MOD_CTRL, INPUT_MOD_NONE, INPUT_MOD_SHIFT, INPUT_MOD_SUPER, InputEvent,
	InputEventKind, InputModifiers, InputState, Key, MotionSpeed, PinchPhase, ScrollGesture,
};

/// Stable input-button identity for held/rising/falling queries.
///
/// Donor: `oa::input::Button`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Button {
	#[default]
	None,
	MouseLeft,
	MouseMiddle,
	MouseRight,
	KeyUp,
	KeyDown,
	KeyLeft,
	KeyRight,
	KeySpace,
	KeyEscape,
	KeyEnter,
}

/// Per-frame keyboard held-state snapshot, indexed by SDL scancode.
///
/// Donor: `oa::input::KeyboardState`.
#[derive(Clone, Debug)]
pub struct KeyboardState {
	pub keys: [bool; 256],
}

impl Default for KeyboardState {
	fn default() -> Self {
		Self { keys: [false; 256] }
	}
}

impl KeyboardState {
	/// Return whether `key` is currently held.
	#[must_use]
	pub fn down(&self, key: Key) -> bool {
		self.keys.get(key as usize).copied().unwrap_or(false)
	}
}

/// Per-frame pointer-state snapshot.
///
/// Donor: `oa::input::MouseState`.
#[derive(Clone, Debug, Default)]
pub struct MouseState {
	pub x: f32,
	pub y: f32,
	pub dx: f32,
	pub dy: f32,
	pub scroll_x: f32,
	pub scroll_y: f32,
	pub pressure: f32,
	pub left: bool,
	pub middle: bool,
	pub right: bool,
}

struct FrameState {
	keyboard: KeyboardState,
	mouse: MouseState,
	prev_keys: [bool; 256],
	press_keys: [bool; 256],
	release_keys: [bool; 256],
	prev_left: bool,
	prev_middle: bool,
	prev_right: bool,
	press_left: bool,
	press_middle: bool,
	press_right: bool,
	release_left: bool,
	release_middle: bool,
	release_right: bool,
	pointer_x: f32,
	pointer_y: f32,
}

impl Default for FrameState {
	fn default() -> Self {
		Self {
			keyboard: KeyboardState::default(),
			mouse: MouseState::default(),
			prev_keys: [false; 256],
			press_keys: [false; 256],
			release_keys: [false; 256],
			prev_left: false,
			prev_middle: false,
			prev_right: false,
			press_left: false,
			press_middle: false,
			press_right: false,
			release_left: false,
			release_middle: false,
			release_right: false,
			pointer_x: 0.0,
			pointer_y: 0.0,
		}
	}
}

impl FrameState {
	fn update_pointer(&mut self, x: f32, y: f32) {
		self.pointer_x = x;
		self.pointer_y = y;
		self.mouse.x = x;
		self.mouse.y = y;
	}
}

thread_local! {
	static FRAME_STATE: RefCell<FrameState> = RefCell::new(FrameState::default());
	static INITIALIZED: Cell<bool> = const { Cell::new(false) };
}

/// Initialize the input backend. Idempotent.
///
/// Prefers native pinch gestures and precise wheel deltas on Wayland; the
/// touchpad stays pointer/wheel input rather than raw touch.
pub fn initialize() {
	INITIALIZED.with(|flag| {
		if flag.get() {
			return;
		}
		// SAFETY: constant name/value pointers outlive the call.
		unsafe {
			sdl3::sys::hints::SDL_SetHint(
				sdl3::sys::hints::SDL_HINT_TRACKPAD_IS_TOUCH_ONLY,
				c"0".as_ptr(),
			)
		};
		flag.set(true);
	});
}

/// Reset all input snapshots.
pub fn shutdown() {
	FRAME_STATE.with(|state| *state.borrow_mut() = FrameState::default());
	INITIALIZED.with(|flag| flag.set(false));
}

/// Feed one raw platform event into the per-frame snapshot.
///
/// Mouse motion/buttons/wheel fold into the pointer state; key down/up sets
/// the scancode-indexed held bits. Press/release edges are derived by
/// [`update`].
///
/// Donor: `oa::input::processEvent`.
pub fn process_event(event: &SdlEvent) {
	FRAME_STATE.with(|state| {
		let state = &mut *state.borrow_mut();
		match event {
			SdlEvent::MouseMotion {
				x, y, xrel, yrel, ..
			} => {
				state.mouse.dx += xrel;
				state.mouse.dy += yrel;
				state.update_pointer(*x, *y);
			}
			SdlEvent::MouseButtonDown {
				x, y, mouse_btn, ..
			} => {
				state.update_pointer(*x, *y);
				match mouse_button_index(*mouse_btn) {
					1 => state.mouse.left = true,
					2 => state.mouse.middle = true,
					3 => state.mouse.right = true,
					_ => {}
				}
			}
			SdlEvent::MouseButtonUp {
				x, y, mouse_btn, ..
			} => {
				state.update_pointer(*x, *y);
				match mouse_button_index(*mouse_btn) {
					1 => state.mouse.left = false,
					2 => state.mouse.middle = false,
					3 => state.mouse.right = false,
					_ => {}
				}
			}
			SdlEvent::MouseWheel {
				x,
				y,
				mouse_x,
				mouse_y,
				..
			} => {
				state.mouse.scroll_x += x;
				state.mouse.scroll_y += y;
				state.update_pointer(*mouse_x, *mouse_y);
			}
			SdlEvent::KeyDown { scancode, .. } => {
				if let Some(index) = scancode_index(*scancode) {
					state.keyboard.keys[index] = true;
				}
			}
			SdlEvent::KeyUp { scancode, .. } => {
				if let Some(index) = scancode_index(*scancode) {
					state.keyboard.keys[index] = false;
				}
			}
			_ => {}
		}
	});
}

fn mouse_button_index(button: sdl3::mouse::MouseButton) -> i32 {
	match button {
		sdl3::mouse::MouseButton::Left => 1,
		sdl3::mouse::MouseButton::Middle => 2,
		sdl3::mouse::MouseButton::Right => 3,
		sdl3::mouse::MouseButton::X1 => 4,
		sdl3::mouse::MouseButton::X2 => 5,
		sdl3::mouse::MouseButton::Unknown => 0,
	}
}

fn scancode_index(scancode: Option<sdl3::keyboard::Scancode>) -> Option<usize> {
	let index = scancode? as i32;
	(0..256).contains(&index).then_some(index as usize)
}

/// Fold queued events into held/rising/falling edges.
///
/// Call once per frame after all [`process_event`] calls and before reading
/// [`press`]/[`release`].
pub fn update() {
	FRAME_STATE.with(|state| {
		let state = &mut *state.borrow_mut();
		for index in 0..256 {
			state.press_keys[index] = state.keyboard.keys[index] && !state.prev_keys[index];
			state.release_keys[index] = !state.keyboard.keys[index] && state.prev_keys[index];
			state.prev_keys[index] = state.keyboard.keys[index];
		}
		state.press_left = state.mouse.left && !state.prev_left;
		state.press_middle = state.mouse.middle && !state.prev_middle;
		state.press_right = state.mouse.right && !state.prev_right;
		state.release_left = !state.mouse.left && state.prev_left;
		state.release_middle = !state.mouse.middle && state.prev_middle;
		state.release_right = !state.mouse.right && state.prev_right;
		state.prev_left = state.mouse.left;
		state.prev_middle = state.mouse.middle;
		state.prev_right = state.mouse.right;
	});
}

/// Clear per-frame deltas (scroll, motion delta, press/release edges).
pub fn clear_for_next_frame() {
	FRAME_STATE.with(|state| {
		let state = &mut *state.borrow_mut();
		state.mouse.dx = 0.0;
		state.mouse.dy = 0.0;
		state.mouse.scroll_x = 0.0;
		state.mouse.scroll_y = 0.0;
		state.press_left = false;
		state.press_middle = false;
		state.press_right = false;
		state.release_left = false;
		state.release_middle = false;
		state.release_right = false;
		state.press_keys.fill(false);
		state.release_keys.fill(false);
	});
}

/// Access the current keyboard held-state snapshot.
pub fn keyboard<R>(read: impl FnOnce(&KeyboardState) -> R) -> R {
	FRAME_STATE.with(|state| read(&state.borrow().keyboard))
}

/// Access the current pointer-state snapshot.
pub fn mouse<R>(read: impl FnOnce(&MouseState) -> R) -> R {
	FRAME_STATE.with(|state| read(&state.borrow().mouse))
}

/// Last known pointer position, updated on motion, buttons, and wheel.
#[must_use]
pub fn pointer_position() -> (f32, f32) {
	FRAME_STATE.with(|state| {
		let state = state.borrow();
		(state.pointer_x, state.pointer_y)
	})
}

fn button_scancode(button: Button) -> Option<usize> {
	let key = match button {
		Button::KeyUp => Key::Up,
		Button::KeyDown => Key::Down,
		Button::KeyLeft => Key::Left,
		Button::KeyRight => Key::Right,
		Button::KeySpace => Key::Space,
		Button::KeyEscape => Key::Escape,
		Button::KeyEnter => Key::Return,
		_ => return None,
	};
	Some(key as usize)
}

enum ButtonEdge {
	Mouse(usize),
	Key(usize),
}

fn button_edge(button: Button) -> Option<ButtonEdge> {
	match button {
		Button::MouseLeft => Some(ButtonEdge::Mouse(0)),
		Button::MouseMiddle => Some(ButtonEdge::Mouse(1)),
		Button::MouseRight => Some(ButtonEdge::Mouse(2)),
		_ => button_scancode(button).map(ButtonEdge::Key),
	}
}

/// Return whether `button` is currently held.
#[must_use]
pub fn down(button: Button) -> bool {
	FRAME_STATE.with(|state| {
		let state = state.borrow();
		match button {
			Button::MouseLeft => state.mouse.left,
			Button::MouseMiddle => state.mouse.middle,
			Button::MouseRight => state.mouse.right,
			_ => button_scancode(button).is_some_and(|index| state.keyboard.keys[index]),
		}
	})
}

/// Return whether `button` went down since the last [`update`].
///
/// Remains valid until [`clear_for_next_frame`].
#[must_use]
pub fn press(button: Button) -> bool {
	FRAME_STATE.with(|state| {
		let state = state.borrow();
		match button_edge(button) {
			Some(ButtonEdge::Mouse(0)) => state.press_left,
			Some(ButtonEdge::Mouse(1)) => state.press_middle,
			Some(ButtonEdge::Mouse(2)) => state.press_right,
			Some(ButtonEdge::Key(index)) => state.press_keys[index],
			_ => false,
		}
	})
}

/// Return whether `button` went up since the last [`update`].
///
/// Remains valid until [`clear_for_next_frame`].
#[must_use]
pub fn release(button: Button) -> bool {
	FRAME_STATE.with(|state| {
		let state = state.borrow();
		match button_edge(button) {
			Some(ButtonEdge::Mouse(0)) => state.release_left,
			Some(ButtonEdge::Mouse(1)) => state.release_middle,
			Some(ButtonEdge::Mouse(2)) => state.release_right,
			Some(ButtonEdge::Key(index)) => state.release_keys[index],
			_ => false,
		}
	})
}
