//! UI types: PixelRect, layout primitives, UiStyle, UiEvent, UiInputState.
//!
//! Donor references: `oa/ui/canvas.h`, `oa/ui/style.h`, `oa/ui/event.h`

// ── PixelRect ─────────────────────────────────────────────────────────────────

/// Axis-aligned pixel rectangle in physical screen coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PixelRect {
	pub x: i32,
	pub y: i32,
	pub w: u32,
	pub h: u32,
}

impl PixelRect {
	pub const fn new(x: i32, y: i32, w: u32, h: u32) -> Self {
		Self { x, y, w, h }
	}

	/// True when the rectangle has non-zero area.
	pub const fn is_non_empty(&self) -> bool {
		self.w > 0 && self.h > 0
	}

	/// True when `(px, py)` is inside this rectangle.
	pub fn contains(&self, px: f32, py: f32) -> bool {
		px >= self.x as f32
			&& py >= self.y as f32
			&& px < (self.x + self.w as i32) as f32
			&& py < (self.y + self.h as i32) as f32
	}

	/// Return the bounding clip for `child` intersected with `self`.
	pub fn clip(&self, child: &PixelRect) -> PixelRect {
		let x0 = self.x.max(child.x);
		let y0 = self.y.max(child.y);
		let x1 = (self.x + self.w as i32).min(child.x + child.w as i32);
		let y1 = (self.y + self.h as i32).min(child.y + child.h as i32);
		if x1 <= x0 || y1 <= y0 {
			return PixelRect::default();
		}
		PixelRect {
			x: x0,
			y: y0,
			w: (x1 - x0) as u32,
			h: (y1 - y0) as u32,
		}
	}
}

use crate::Color;

// ── UiStyle ───────────────────────────────────────────────────────────────────

/// Theme colors and metrics used by all widgets.
#[derive(Clone, Debug)]
pub struct UiStyle {
	/// Panel / window background.
	pub background: Color,
	/// Widget surface (button, slider track).
	pub surface: Color,
	/// Active/hovered widget accent.
	pub accent: Color,
	/// Primary text color.
	pub text: Color,
	/// Secondary/muted text.
	pub text_secondary: Color,
	/// Widget border.
	pub border: Color,
	/// Widget corner radius in pixels.
	pub corner_radius: f32,
	/// Inner content padding in pixels.
	pub padding: f32,
	/// Base font size in pixels.
	pub font_size: f32,
}

impl Default for UiStyle {
	/// Realm dark theme — mirrors the donor default.
	fn default() -> Self {
		Self {
			background: Color::new(0.082, 0.082, 0.090, 1.0),
			surface: Color::new(0.141, 0.141, 0.161, 1.0),
			accent: Color::new(0.388, 0.400, 0.945, 1.0),
			text: Color::new(0.961, 0.961, 0.961, 1.0),
			text_secondary: Color::new(0.600, 0.600, 0.620, 1.0),
			border: Color::new(0.250, 0.250, 0.280, 1.0),
			corner_radius: 4.0,
			padding: 8.0,
			font_size: 14.0,
		}
	}
}

impl UiStyle {
	/// Viewer dark theme — slightly deeper background than the default.
	pub fn viewer_dark() -> Self {
		Self {
			background: Color::new(0.055, 0.055, 0.060, 1.0),
			..Self::default()
		}
	}
}

// ── Layout primitives ─────────────────────────────────────────────────────────

/// Sizing policy for one layout axis.
#[derive(Clone, Copy, Debug, Default)]
pub enum UiSizing {
	/// Expand to fill available parent space.
	#[default]
	Fill,
	/// Fixed pixel size.
	Fixed(f32),
	/// Shrink-wrap content.
	Hug,
}

/// Per-side padding in pixels.
#[derive(Clone, Copy, Debug, Default)]
pub struct UiEdge {
	pub top: f32,
	pub right: f32,
	pub bottom: f32,
	pub left: f32,
}

impl UiEdge {
	pub const fn all(v: f32) -> Self {
		Self {
			top: v,
			right: v,
			bottom: v,
			left: v,
		}
	}
}

/// Flow direction for a panel or row.
#[derive(Clone, Copy, Debug, Default)]
pub enum UiDirection {
	#[default]
	Column,
	Row,
}

/// Alignment of children along the cross-axis.
#[derive(Clone, Copy, Debug, Default)]
pub enum UiAlign {
	Start,
	Center,
	End,
	#[default]
	Stretch,
}

/// Layout descriptor passed to `begin_panel`.
#[derive(Clone, Debug)]
pub struct UiLayout {
	pub direction: UiDirection,
	pub align: UiAlign,
	pub gap: f32,
	pub padding: UiEdge,
	pub width: UiSizing,
	pub height: UiSizing,
}

impl Default for UiLayout {
	fn default() -> Self {
		Self {
			direction: UiDirection::Column,
			align: UiAlign::Stretch,
			gap: 4.0,
			padding: UiEdge::all(8.0),
			width: UiSizing::Fill,
			height: UiSizing::Hug,
		}
	}
}

// ── UiKey ─────────────────────────────────────────────────────────────────────
// Values match SDL3 scancodes. Donor: `oa/ui/event.h`

/// Platform-neutral keyboard key identifier (SDL3 scancode values).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum UiKey {
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

impl UiKey {
	/// Try to convert an SDL3 scancode integer to a `UiKey`.
	pub fn from_sdl_scancode(scancode: u32) -> Self {
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

// ── UiModifiers ───────────────────────────────────────────────────────────────

/// Modifier key bitmask (matches donor `UiModifier*` constants).
pub type UiModifiers = u32;
pub const UI_MOD_NONE: UiModifiers = 0;
pub const UI_MOD_SHIFT: UiModifiers = 1 << 0;
pub const UI_MOD_CTRL: UiModifiers = 1 << 1;
pub const UI_MOD_ALT: UiModifiers = 1 << 2;
pub const UI_MOD_SUPER: UiModifiers = 1 << 3;

// ── UiScrollGesture ───────────────────────────────────────────────────────────

/// Classified scroll gesture — maps to donor `UiScrollGesture`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiScrollGesture {
	#[default]
	None,
	/// Discrete mouse-wheel notch → zoom-at-cursor.
	MouseWheel,
	/// Smooth two-finger trackpad pan.
	TouchpadPan,
	/// Ctrl + smooth scroll (Wayland pinch fallback).
	PinchScroll,
}

/// Classify an SDL3 wheel event into a [`UiScrollGesture`].
///
/// Heuristic: integer_x/y non-zero → discrete wheel; ctrl held → pinch
/// fallback; otherwise smooth trackpad pan.
pub fn classify_scroll(integer_x: i32, integer_y: i32, modifiers: UiModifiers) -> UiScrollGesture {
	if modifiers & UI_MOD_CTRL != 0 {
		return UiScrollGesture::PinchScroll;
	}
	if integer_x != 0 || integer_y != 0 {
		UiScrollGesture::MouseWheel
	} else {
		UiScrollGesture::TouchpadPan
	}
}

// ── UiPinchPhase ─────────────────────────────────────────────────────────────

/// Phase of a multi-touch pinch gesture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiPinchPhase {
	#[default]
	None,
	Begin,
	Update,
	End,
}

// ── UiEvent ───────────────────────────────────────────────────────────────────

/// Platform-neutral UI event. Donor: `oa/ui/event.h` `UiEvent`.
#[derive(Clone, Debug, Default)]
pub struct UiEvent {
	pub kind: UiEventKind,
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
	pub scroll_gesture: UiScrollGesture,
	/// Multi-touch pinch — scale relative to Begin (1.0 = no change).
	pub gesture_scale: f32,
	pub pinch_phase: UiPinchPhase,
	pub key: UiKey,
	pub modifiers: UiModifiers,
	pub key_repeat: bool,
	/// Committed UTF-8 text from platform IME.
	pub text: String,
	pub text_selection_start: i32,
	pub text_selection_length: i32,
	/// Resized window pixel dimensions.
	pub window_w: i32,
	pub window_h: i32,
}

impl UiEvent {
	pub fn ctrl(&self) -> bool {
		self.modifiers & UI_MOD_CTRL != 0
	}
	pub fn shift(&self) -> bool {
		self.modifiers & UI_MOD_SHIFT != 0
	}
	pub fn alt(&self) -> bool {
		self.modifiers & UI_MOD_ALT != 0
	}

	/// Construct a pointer-move event.
	pub fn mouse_move(x: f32, y: f32, modifiers: UiModifiers) -> Self {
		Self {
			kind: UiEventKind::MouseMove,
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
	pub fn mouse_down(x: f32, y: f32, button: i32, click_count: i32, modifiers: UiModifiers) -> Self {
		Self {
			kind: UiEventKind::MouseDown,
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
	pub fn mouse_up(x: f32, y: f32, button: i32, modifiers: UiModifiers) -> Self {
		Self {
			kind: UiEventKind::MouseUp,
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
	pub fn key_down(key: UiKey, modifiers: UiModifiers, repeat: bool) -> Self {
		Self {
			kind: UiEventKind::KeyDown,
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
	pub fn key_up(key: UiKey, modifiers: UiModifiers) -> Self {
		Self {
			kind: UiEventKind::KeyUp,
			key,
			modifiers,
			gesture_scale: 1.0,
			text_selection_start: -1,
			text_selection_length: -1,
			..Default::default()
		}
	}
}

/// Discriminant for [`UiEvent`]. Donor: `UiEventType`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiEventKind {
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

// ── UiInputState ─────────────────────────────────────────────────────────────

/// Stable per-frame pointer and modifier snapshot. Donor: `oa/ui/event.h`.
#[derive(Clone, Debug, Default)]
pub struct UiInputState {
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
	pub modifiers: UiModifiers,
	// ── backwards-compat field aliases ──────────────────────────────────────
	/// Alias for `l_button` (legacy API used by current widget code).
	pub pointer_down: bool,
	pub pointer_x: f32,
	pub pointer_y: f32,
	pub shift: bool,
	pub ctrl: bool,
}

impl UiInputState {
	pub const EMPTY: UiInputState = UiInputState {
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
		modifiers: UI_MOD_NONE,
		pointer_down: false,
		pointer_x: 0.0,
		pointer_y: 0.0,
		shift: false,
		ctrl: false,
	};
}

// ── Tab bar ───────────────────────────────────────────────────────────────────

/// A single tab entry passed to [`crate::ui::Ui::tab_bar`].
///
/// Donor: `oa::UiTabItem`.
#[derive(Clone, Debug, Default)]
pub struct UiTabItem {
	/// Stable string identifier (not displayed).
	pub id: String,
	/// Visible tab label.
	pub label: String,
	/// Dirty indicator shown next to the label.
	pub dirty: bool,
	/// Whether the tab shows a close button.
	pub closable: bool,
	/// Whether the tab is enabled (greyed out when false).
	pub enabled: bool,
}

impl UiTabItem {
	pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
		Self {
			id: id.into(),
			label: label.into(),
			closable: true,
			enabled: true,
			dirty: false,
		}
	}
}

/// Caller-owned tab bar presentation state.
///
/// Donor: `oa::UiTabBarState`.
#[derive(Clone, Debug, Default)]
pub struct UiTabBarState {
	/// Index of the currently selected tab, or -1 for none.
	pub selected: i32,
	/// Index of the first visible tab (for overflow scrolling).
	pub first_visible: i32,
}

/// Return value from [`crate::ui::Ui::tab_bar`].
///
/// Donor: `oa::UiTabBarResult`.
#[derive(Clone, Debug, Default)]
pub struct UiTabBarResult {
	/// Full tab-bar rectangle.
	pub bar: PixelRect,
	/// Rectangle spanning visible tab buttons only.
	pub tabs: PixelRect,
	/// Index of the first visible tab after this call.
	pub first_visible: i32,
	/// One past the last visible tab.
	pub one_past_last: i32,
	/// Index of the tab that was clicked, or -1.
	pub activated_index: i32,
	/// Index of the tab whose close button was clicked, or -1.
	pub close_requested_index: i32,
	/// Whether `state.selected` changed.
	pub selection_changed: bool,
}

// ── Tree row ──────────────────────────────────────────────────────────────────

/// Configuration for one [`crate::ui::Ui::tree_row`] call.
///
/// Donor: `oa::UiTreeRowConfig`.
#[derive(Clone, Debug, Default)]
pub struct UiTreeRowConfig {
	/// Nesting depth (0 = root).
	pub depth: i32,
	/// Pixels per indent level.
	pub indent: i32,
	/// Width reserved for the disclosure arrow.
	pub disclosure_width: i32,
	/// True if this item has children.
	pub has_children: bool,
	/// True if this item is currently expanded.
	pub open: bool,
	/// True if this item is selected.
	pub selected: bool,
	/// False to render greyed out.
	pub enabled: bool,
}

impl UiTreeRowConfig {
	pub const fn new() -> Self {
		Self {
			depth: 0,
			indent: 16,
			disclosure_width: 18,
			has_children: false,
			open: false,
			selected: false,
			enabled: true,
		}
	}
}

/// Return value from [`crate::ui::Ui::tree_row`].
///
/// Donor: `oa::UiTreeRowResult`.
#[derive(Clone, Debug, Default)]
pub struct UiTreeRowResult {
	/// Full row rectangle.
	pub row: PixelRect,
	/// Disclosure-arrow rectangle.
	pub disclosure: PixelRect,
	/// Label rectangle.
	pub label: PixelRect,
	/// True if the row was activated (clicked).
	pub activated: bool,
	/// True if the open/close state toggled.
	pub open_changed: bool,
	/// Current open state after this call.
	pub open: bool,
}

// ── Node canvas ───────────────────────────────────────────────────────────────

/// Caller-owned view state for a node-graph canvas.
///
/// Donor: `oa::NodeCanvasState`.
#[derive(Clone, Debug)]
pub struct NodeCanvasState {
	/// World-space pan offset (origin of the viewport in world coords).
	pub pan: [f32; 2],
	/// Zoom factor (pixels per world unit).
	pub zoom: f32,
	/// Viewport dimensions in screen pixels.
	pub view_size: [f32; 2],
}

impl Default for NodeCanvasState {
	fn default() -> Self {
		Self {
			pan: [0.0, 0.0],
			zoom: 1.0,
			view_size: [0.0, 0.0],
		}
	}
}

/// Node-graph canvas navigation and coordinate mapping.
///
/// Caller-owned; passed to [`crate::ui::Ui::node_canvas_grid`] to render the background
/// grid.  Coordinate transforms are provided for callers that need to project
/// node world positions into screen pixels.
///
/// Donor: `oa::NodeCanvas`.
#[derive(Clone, Debug, Default)]
pub struct NodeCanvas {
	state: NodeCanvasState,
}

impl NodeCanvas {
	pub fn new() -> Self {
		Self::default()
	}

	/// Set the canvas view size (called once when the viewport is known).
	pub fn set_view_size(&mut self, w: f32, h: f32) -> crate::Result<()> {
		self.state.view_size = [w, h];
		Ok(())
	}

	/// Access the current canvas state.
	pub fn state(&self) -> &NodeCanvasState {
		&self.state
	}

	/// Convert a world-space position to screen-space pixels.
	pub fn world_to_screen(&self, world: [f32; 2]) -> [f32; 2] {
		let cx = self.state.view_size[0] * 0.5;
		let cy = self.state.view_size[1] * 0.5;
		[
			cx + (world[0] - self.state.pan[0]) * self.state.zoom,
			cy + (world[1] - self.state.pan[1]) * self.state.zoom,
		]
	}

	/// Convert screen-space pixels to world-space coordinates.
	pub fn screen_to_world(&self, screen: [f32; 2]) -> [f32; 2] {
		let cx = self.state.view_size[0] * 0.5;
		let cy = self.state.view_size[1] * 0.5;
		let inv = if self.state.zoom.abs() > 1e-9 {
			1.0 / self.state.zoom
		} else {
			1.0
		};
		[
			self.state.pan[0] + (screen[0] - cx) * inv,
			self.state.pan[1] + (screen[1] - cy) * inv,
		]
	}

	/// Pan by a screen-space delta.
	pub fn pan(&mut self, delta: [f32; 2]) {
		let inv = if self.state.zoom.abs() > 1e-9 {
			1.0 / self.state.zoom
		} else {
			1.0
		};
		self.state.pan[0] -= delta[0] * inv;
		self.state.pan[1] -= delta[1] * inv;
	}
}

// ── UiScrollConfig / UiScrollRegion ─────────────────────────────────────────

/// Configuration for a scrollable panel.
///
/// Donor: `oa::UiScrollConfig`.
#[derive(Clone, Debug)]
pub struct UiScrollConfig {
	/// Pixels scrolled per mouse-wheel notch.
	pub wheel_step: f32,
	/// Width of the scrollbar track in pixels.
	pub scrollbar_width: f32,
	/// Gap between content and scrollbar track.
	pub scrollbar_gap: f32,
	/// Whether to show the scrollbar.
	pub show_scrollbar: bool,
}

impl Default for UiScrollConfig {
	fn default() -> Self {
		Self {
			wheel_step: 40.0,
			scrollbar_width: 6.0,
			scrollbar_gap: 2.0,
			show_scrollbar: true,
		}
	}
}

/// Return value from [`crate::ui::Ui::begin_scroll_panel`].
///
/// Donor: `oa::UiScrollRegion`.
#[derive(Clone, Debug, Default)]
pub struct UiScrollRegion {
	/// Visible viewport rectangle.
	pub viewport: PixelRect,
	/// Content area rectangle (may extend below viewport).
	pub content: PixelRect,
	/// Current vertical scroll offset in pixels.
	pub offset_y: i32,
	/// Maximum reachable vertical offset.
	pub max_offset_y: i32,
}

// ── UiVirtualRange ───────────────────────────────────────────────────────────

/// Half-open row range produced by [`crate::ui::Ui::virtual_rows`].
///
/// Donor: `oa::UiVirtualRange`.
#[derive(Clone, Copy, Debug, Default)]
pub struct UiVirtualRange {
	/// Index of the first row to render.
	pub first: i32,
	/// One past the last row to render.
	pub one_past_last: i32,
}

impl UiVirtualRange {
	/// True when no rows are in range.
	pub fn is_empty(self) -> bool {
		self.one_past_last <= self.first
	}
}

// ── UiSplitConfig / UiSplitRegion ────────────────────────────────────────────

/// Configuration for a two-pane split view.
///
/// Donor: `oa::UiSplitConfig`.
#[derive(Clone, Debug)]
pub struct UiSplitConfig {
	/// Flow direction: Column = top/bottom split, Row = left/right split.
	pub direction: UiDirection,
	/// Width of the drag handle in pixels.
	pub handle_size: f32,
	/// Minimum size of the first pane in pixels.
	pub minimum_first: f32,
	/// Minimum size of the second pane in pixels.
	pub minimum_second: f32,
	/// Pixels moved per arrow-key press.
	pub keyboard_step: f32,
}

impl Default for UiSplitConfig {
	fn default() -> Self {
		Self {
			direction: super::types::UiDirection::Row,
			handle_size: 4.0,
			minimum_first: 60.0,
			minimum_second: 60.0,
			keyboard_step: 8.0,
		}
	}
}

/// Return value from [`crate::ui::Ui::split_pane`].
///
/// Donor: `oa::UiSplitRegion`.
#[derive(Clone, Debug, Default)]
pub struct UiSplitRegion {
	/// Bounds of the first (top or left) pane.
	pub first: PixelRect,
	/// Bounds of the drag handle.
	pub handle: PixelRect,
	/// Bounds of the second (bottom or right) pane.
	pub second: PixelRect,
	/// Whether the ratio changed this frame.
	pub changed: bool,
}

// ── UiPropertyRowConfig / UiPropertyRegion ───────────────────────────────────

/// Configuration for a labeled property row.
///
/// Donor: `oa::UiPropertyRowConfig`.
#[derive(Clone, Debug)]
pub struct UiPropertyRowConfig {
	/// Fraction of the row width reserved for the label (0–1).
	pub label_fraction: f32,
	/// Gap between label and value columns in pixels.
	pub gap: f32,
	/// Horizontal padding inside each column.
	pub padding_x: f32,
	/// Draw an alternating background on even rows.
	pub alternate: bool,
}

impl Default for UiPropertyRowConfig {
	fn default() -> Self {
		Self {
			label_fraction: 0.40,
			gap: 4.0,
			padding_x: 4.0,
			alternate: false,
		}
	}
}

/// Return value from [`crate::ui::Ui::property_row`].
///
/// Donor: `oa::UiPropertyRegion`.
#[derive(Clone, Debug, Default)]
pub struct UiPropertyRegion {
	/// Full row rectangle.
	pub row: PixelRect,
	/// Label sub-rectangle.
	pub label: PixelRect,
	/// Value sub-rectangle.
	pub value: PixelRect,
}

// ── UiPopupConfig ────────────────────────────────────────────────────────────

/// Configuration for a floating popup overlay.
///
/// Donor: `oa::UiPopupConfig`.
#[derive(Clone, Debug)]
pub struct UiPopupConfig {
	/// Popup width in pixels.
	pub width: f32,
	/// Popup height in pixels (0 = hug content).
	pub height: f32,
	/// Gap between anchor and popup in pixels.
	pub gap: f32,
	/// Padding inside the popup panel.
	pub padding: f32,
}

impl Default for UiPopupConfig {
	fn default() -> Self {
		Self {
			width: 160.0,
			height: 0.0,
			gap: 2.0,
			padding: 6.0,
		}
	}
}

// ── UiDropdownConfig ─────────────────────────────────────────────────────────

/// Configuration for a dropdown widget.
///
/// Donor: `oa::UiDropdownConfig`.
#[derive(Clone, Debug)]
pub struct UiDropdownConfig {
	/// Maximum number of visible items before scrolling.
	pub max_visible_items: i32,
	/// Popup width override (0 = match widget width).
	pub popup_width: f32,
	/// Gap between widget and popup.
	pub popup_gap: f32,
}

impl Default for UiDropdownConfig {
	fn default() -> Self {
		Self {
			max_visible_items: 8,
			popup_width: 0.0,
			popup_gap: 2.0,
		}
	}
}

// ── UiTooltipConfig ──────────────────────────────────────────────────────────

/// Configuration for a hover tooltip.
///
/// Donor: `oa::UiTooltipConfig`.
#[derive(Clone, Debug)]
pub struct UiTooltipConfig {
	/// Milliseconds to wait before showing the tooltip.
	pub delay_ms: f32,
	/// Maximum tooltip width in pixels.
	pub max_width: f32,
	/// Gap between cursor and tooltip box.
	pub gap: f32,
	/// Padding inside the tooltip panel.
	pub padding: f32,
}

impl Default for UiTooltipConfig {
	fn default() -> Self {
		Self {
			delay_ms: 400.0,
			max_width: 240.0,
			gap: 8.0,
			padding: 6.0,
		}
	}
}

// ── UiHeatmapConfig ──────────────────────────────────────────────────────────

/// Configuration for a Ui-level inline heatmap.
///
/// Donor: `oa::UiHeatmapConfig`.
#[derive(Clone, Debug)]
pub struct UiHeatmapConfig {
	/// Number of data rows.
	pub rows: i32,
	/// Number of data columns.
	pub cols: i32,
	/// Minimum value mapped to the cold color.
	pub v_min: f32,
	/// Maximum value mapped to the hot color.
	pub v_max: f32,
	/// Colormap index (matches [`crate::plot::axes::HeatmapStyle`] colormap).
	pub colormap: i32,
	/// Auto-scale range from data when true.
	pub auto_scale: bool,
	/// Draw grid lines between cells.
	pub show_grid: bool,
}

impl Default for UiHeatmapConfig {
	fn default() -> Self {
		Self {
			rows: 1,
			cols: 1,
			v_min: 0.0,
			v_max: 1.0,
			colormap: 1,
			auto_scale: true,
			show_grid: false,
		}
	}
}

// ── UiGridConfig ─────────────────────────────────────────────────────────────

/// Configuration for a regular axis-aligned background grid.
///
/// Donor: `oa::UiGridConfig`.
#[derive(Clone, Debug)]
pub struct UiGridConfig {
	/// Pixels between minor grid lines.
	pub minor_spacing: f32,
	/// Number of minor lines between major lines.
	pub major_every: i32,
	/// Number of major lines between super-major lines.
	pub super_major_every: i32,
	/// Thickness of minor lines in pixels.
	pub minor_thickness: f32,
	/// Thickness of major lines in pixels.
	pub major_thickness: f32,
	/// Thickness of super-major lines in pixels.
	pub super_major_thickness: f32,
	/// Thickness of the axis lines.
	pub axis_thickness: f32,
	/// Fill panel background before drawing grid.
	pub fill_background: bool,
	/// Draw axis lines through origin.
	pub draw_axes: bool,
}

impl Default for UiGridConfig {
	fn default() -> Self {
		Self {
			minor_spacing: 20.0,
			major_every: 5,
			super_major_every: 10,
			minor_thickness: 0.5,
			major_thickness: 1.0,
			super_major_thickness: 1.5,
			axis_thickness: 1.5,
			fill_background: true,
			draw_axes: true,
		}
	}
}

// ── UiIcon ───────────────────────────────────────────────────────────────────

/// Named icon for icon-button widgets.
///
/// Donor: `oa::UiIcon`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiIcon {
	#[default]
	Previous,
	Next,
	Play,
	Pause,
	Menu,
	Minimize,
	Maximize,
	Restore,
	Close,
	Fullscreen,
	ExitFullscreen,
	Volume,
	Muted,
	Settings,
}

/// Display style for an icon button.
///
/// Donor: `oa::UiIconButtonStyle`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiIconButtonStyle {
	/// Semi-transparent overlay (e.g. video transport controls).
	#[default]
	Overlay,
	/// Inline header button.
	Header,
	/// Window chrome control (min/max/close).
	WindowControl,
}

/// Direction for a chevron button.
///
/// Donor: `oa::UiChevronDirection`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiChevronDirection {
	#[default]
	Previous,
	Next,
}

// ── UiAccessibility ──────────────────────────────────────────────────────────

/// Accessibility role for a widget.
///
/// Donor: `oa::UiAccessibilityRole`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiAccessibilityRole {
	#[default]
	Button,
	Checkbox,
	Slider,
	TextField,
	ComboBox,
	MenuItem,
	Tab,
	TreeItem,
	Splitter,
	Timeline,
}

/// Opaque accessibility snapshot — a flat list of node descriptors.
///
/// Donor: `oa::Ui::accessibilitySnapshot` return.
#[derive(Clone, Debug, Default)]
pub struct UiAccessibilitySnapshot {
	/// Number of nodes in the snapshot.
	pub node_count: usize,
}
