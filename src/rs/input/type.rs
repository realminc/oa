//! Platform-neutral input vocabulary.
//!
//! Donor: `oa/ui/event.h` (`UiEvent`, `UiKey`, modifiers, gestures) and the
//! shared motion-timing preset. These records carry no widget semantics so
//! `oa::input` stays usable outside the UI layer.

// ── Key ─────────────────────────────────────────────────────────────────────
// Values match SDL3 scancodes. Donor: `oa/ui/event.h`

/// Platform-neutral keyboard key identifier (SDL3 scancode values).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Key {
	#[default]
	Unknown = 0,
	A = 4,
	B = 5,
	C = 6,
	D = 7,
	E = 8,
	F = 9,
	G = 10,
	H = 11,
	I = 12,
	J = 13,
	K = 14,
	L = 15,
	M = 16,
	N = 17,
	O = 18,
	P = 19,
	Q = 20,
	R = 21,
	S = 22,
	T = 23,
	U = 24,
	V = 25,
	W = 26,
	X = 27,
	Y = 28,
	Z = 29,
	Num1 = 30,
	Num2 = 31,
	Num3 = 32,
	Num4 = 33,
	Num5 = 34,
	Num6 = 35,
	Num7 = 36,
	Num8 = 37,
	Num9 = 38,
	Num0 = 39,
	Return = 40,
	Escape = 41,
	Backspace = 42,
	Tab = 43,
	Space = 44,
	Minus = 45,
	Equals = 46,
	Comma = 54,
	Period = 55,
	Slash = 56,
	F1 = 58,
	F2 = 59,
	F3 = 60,
	F4 = 61,
	F5 = 62,
	F6 = 63,
	F7 = 64,
	F8 = 65,
	F9 = 66,
	F10 = 67,
	F11 = 68,
	F12 = 69,
	Home = 74,
	Delete = 76,
	End = 77,
	Right = 79,
	Left = 80,
	Down = 81,
	Up = 82,
	KpEnter = 88,
	Kp1 = 89,
	Kp2 = 90,
	Kp3 = 91,
	Kp4 = 92,
	Kp5 = 93,
	Kp6 = 94,
	Kp7 = 95,
	Kp8 = 96,
	Kp9 = 97,
	Kp0 = 98,
}

impl Key {
	/// Try to convert an SDL3 scancode integer to a `Key`.
	pub fn new_from_sdl_scancode(scancode: u32) -> Self {
		match scancode {
			4 => Self::A,
			5 => Self::B,
			6 => Self::C,
			7 => Self::D,
			8 => Self::E,
			9 => Self::F,
			10 => Self::G,
			11 => Self::H,
			12 => Self::I,
			13 => Self::J,
			14 => Self::K,
			15 => Self::L,
			16 => Self::M,
			17 => Self::N,
			18 => Self::O,
			19 => Self::P,
			20 => Self::Q,
			21 => Self::R,
			22 => Self::S,
			23 => Self::T,
			24 => Self::U,
			25 => Self::V,
			26 => Self::W,
			27 => Self::X,
			28 => Self::Y,
			29 => Self::Z,
			30 => Self::Num1,
			31 => Self::Num2,
			32 => Self::Num3,
			33 => Self::Num4,
			34 => Self::Num5,
			35 => Self::Num6,
			36 => Self::Num7,
			37 => Self::Num8,
			38 => Self::Num9,
			39 => Self::Num0,
			40 => Self::Return,
			41 => Self::Escape,
			42 => Self::Backspace,
			43 => Self::Tab,
			44 => Self::Space,
			45 => Self::Minus,
			46 => Self::Equals,
			54 => Self::Comma,
			55 => Self::Period,
			56 => Self::Slash,
			58 => Self::F1,
			59 => Self::F2,
			60 => Self::F3,
			61 => Self::F4,
			62 => Self::F5,
			63 => Self::F6,
			64 => Self::F7,
			65 => Self::F8,
			66 => Self::F9,
			67 => Self::F10,
			68 => Self::F11,
			69 => Self::F12,
			74 => Self::Home,
			76 => Self::Delete,
			77 => Self::End,
			79 => Self::Right,
			80 => Self::Left,
			81 => Self::Down,
			82 => Self::Up,
			88 => Self::KpEnter,
			89 => Self::Kp1,
			90 => Self::Kp2,
			91 => Self::Kp3,
			92 => Self::Kp4,
			93 => Self::Kp5,
			94 => Self::Kp6,
			95 => Self::Kp7,
			96 => Self::Kp8,
			97 => Self::Kp9,
			98 => Self::Kp0,
			_ => Self::Unknown,
		}
	}

	/// True when key is a navigation arrow.
	pub fn is_arrow(self) -> bool {
		matches!(self, Self::Left | Self::Right | Self::Up | Self::Down)
	}
}

// ── InputModifiers ───────────────────────────────────────────────────────────────

/// Modifier key bitmask (matches donor `UiModifier*` constants).
pub type InputModifiers = u32;
pub const INPUT_MOD_NONE: InputModifiers = 0;
pub const INPUT_MOD_SHIFT: InputModifiers = 1 << 0;
pub const INPUT_MOD_CTRL: InputModifiers = 1 << 1;
pub const INPUT_MOD_ALT: InputModifiers = 1 << 2;
pub const INPUT_MOD_SUPER: InputModifiers = 1 << 3;

// ── ScrollGesture ───────────────────────────────────────────────────────────

/// Classified scroll gesture — maps to donor `UiScrollGesture`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScrollGesture {
	#[default]
	None,
	/// Discrete mouse-wheel notch → zoom-at-cursor.
	MouseWheel,
	/// Smooth two-finger trackpad pan.
	TouchpadPan,
	/// Ctrl + smooth scroll (Wayland pinch fallback).
	PinchScroll,
}

// ── PinchPhase ─────────────────────────────────────────────────────────────

/// Phase of a multi-touch pinch gesture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PinchPhase {
	#[default]
	None,
	Begin,
	Update,
	End,
}

// ── InputEvent ───────────────────────────────────────────────────────────────────

/// Platform-neutral input event. Donor: `oa/ui/event.h` `UiEvent`.
#[derive(Clone, Debug, Default)]
pub struct InputEvent {
	pub kind: InputEventKind,
	/// Physical pixel pointer position.
	pub mouse_x: f32,
	pub mouse_y: f32,
	/// Pointer delta from previous position.
	pub mouse_dx: f32,
	pub mouse_dy: f32,
	/// 1 = left, 2 = middle, 3 = right.
	pub button: i32,
	pub click_count: i32,
	/// Smooth scroll deltas (SDL3 wheel.x/y).
	pub scroll_x: f32,
	pub scroll_y: f32,
	/// Accumulated discrete wheel ticks (SDL3 integer_x/y).
	pub integer_scroll_x: i32,
	pub integer_scroll_y: i32,
	/// SDL3 timestamp in nanoseconds (for burst detection).
	pub timestamp_ns: u64,
	pub scroll_gesture: ScrollGesture,
	/// Multi-touch pinch — scale relative to Begin (1.0 = no change).
	pub gesture_scale: f32,
	pub pinch_phase: PinchPhase,
	pub key: Key,
	pub modifiers: InputModifiers,
	pub key_repeat: bool,
	/// Committed UTF-8 text from platform IME.
	pub text: String,
	pub text_selection_start: i32,
	pub text_selection_length: i32,
	/// Resized window pixel dimensions.
	pub window_w: i32,
	pub window_h: i32,
}

impl InputEvent {
	pub fn ctrl(&self) -> bool {
		self.modifiers & INPUT_MOD_CTRL != 0
	}
	pub fn shift(&self) -> bool {
		self.modifiers & INPUT_MOD_SHIFT != 0
	}
	pub fn alt(&self) -> bool {
		self.modifiers & INPUT_MOD_ALT != 0
	}

	/// Construct a pointer-move event.
	pub fn mouse_move(x: f32, y: f32, modifiers: InputModifiers) -> Self {
		Self {
			kind: InputEventKind::MouseMove,
			mouse_x: x,
			mouse_y: y,
			modifiers,
			gesture_scale: 1.0,
			text_selection_start: -1,
			text_selection_length: -1,
			..Default::default()
		}
	}
	/// Construct a pointer-down event.
	pub fn mouse_down(
		x: f32,
		y: f32,
		button: i32,
		click_count: i32,
		modifiers: InputModifiers,
	) -> Self {
		Self {
			kind: InputEventKind::MouseDown,
			mouse_x: x,
			mouse_y: y,
			button,
			click_count,
			modifiers,
			gesture_scale: 1.0,
			text_selection_start: -1,
			text_selection_length: -1,
			..Default::default()
		}
	}
	/// Construct a pointer-up event.
	pub fn mouse_up(x: f32, y: f32, button: i32, modifiers: InputModifiers) -> Self {
		Self {
			kind: InputEventKind::MouseUp,
			mouse_x: x,
			mouse_y: y,
			button,
			modifiers,
			gesture_scale: 1.0,
			text_selection_start: -1,
			text_selection_length: -1,
			..Default::default()
		}
	}
	/// Construct a key-down event.
	pub fn key_down(key: Key, modifiers: InputModifiers, repeat: bool) -> Self {
		Self {
			kind: InputEventKind::KeyDown,
			key,
			modifiers,
			key_repeat: repeat,
			gesture_scale: 1.0,
			text_selection_start: -1,
			text_selection_length: -1,
			..Default::default()
		}
	}
	/// Construct a key-up event.
	pub fn key_up(key: Key, modifiers: InputModifiers) -> Self {
		Self {
			kind: InputEventKind::KeyUp,
			key,
			modifiers,
			gesture_scale: 1.0,
			text_selection_start: -1,
			text_selection_length: -1,
			..Default::default()
		}
	}
}

/// Discriminant for [`InputEvent`]. Donor: `UiEventType`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InputEventKind {
	#[default]
	None,
	MouseMove,
	MouseDown,
	MouseUp,
	MouseScroll,
	KeyDown,
	KeyUp,
	/// Committed text from platform IME.
	KeyChar,
	/// Window pixel-size changed.
	WindowResize,
	WindowClose,
	WindowFocus,
	WindowBlur,
	/// Multi-touch pinch.
	Pinch,
	/// IME pre-edit text update.
	TextEditing,
	MouseEnter,
	MouseLeave,
}

// ── InputState ─────────────────────────────────────────────────────────────

/// Stable per-frame pointer and modifier snapshot. Donor: `oa/ui/event.h`.
#[derive(Clone, Debug, Default)]
pub struct InputState {
	pub mouse_x: f32,
	pub mouse_y: f32,
	pub mouse_dx: f32,
	pub mouse_dy: f32,
	pub l_button: bool,
	pub m_button: bool,
	pub r_button: bool,
	/// True on the exact frame the left button transitioned down.
	pub l_pressed: bool,
	/// True on the exact frame the left button transitioned up.
	pub l_released: bool,
	pub l_click_count: i32,
	pub scroll_x: f32,
	pub scroll_y: f32,
	pub modifiers: InputModifiers,
	// ── backwards-compat field aliases ──────────────────────────────────────
	/// Alias for `l_button` (legacy API used by current widget code).
	pub pointer_down: bool,
	pub pointer_x: f32,
	pub pointer_y: f32,
	pub shift: bool,
	pub ctrl: bool,
}

impl InputState {
	pub const EMPTY: InputState = InputState {
		mouse_x: 0.0,
		mouse_y: 0.0,
		mouse_dx: 0.0,
		mouse_dy: 0.0,
		l_button: false,
		m_button: false,
		r_button: false,
		l_pressed: false,
		l_released: false,
		l_click_count: 0,
		scroll_x: 0.0,
		scroll_y: 0.0,
		modifiers: INPUT_MOD_NONE,
		pointer_down: false,
		pointer_x: 0.0,
		pointer_y: 0.0,
		shift: false,
		ctrl: false,
	};
}

/// Shared donor timing policy for widget and viewport transitions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MotionSpeed {
	#[default]
	Normal,
	Fast,
	Fastest,
	Instant,
}

impl MotionSpeed {
	pub const fn duration_ms(self, normal: f32) -> f32 {
		match self {
			Self::Normal => normal,
			Self::Fast => normal * 0.65,
			Self::Fastest => normal * 0.35,
			Self::Instant => 0.0,
		}
	}

	pub const fn next(self) -> Self {
		match self {
			Self::Normal => Self::Fast,
			Self::Fast => Self::Fastest,
			Self::Fastest => Self::Instant,
			Self::Instant => Self::Normal,
		}
	}
}
