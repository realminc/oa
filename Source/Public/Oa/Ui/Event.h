// OaUi — Input events and per-frame input state.
//
// Events are polled per-frame from the windowing layer (SDL3).
// OaUiInputState is the stable per-frame snapshot; widgets read it instead of
// consuming the event queue directly.

#pragma once

#include <Oa/Core/Types.h>


// ─── Modifier mask ────────────────────────────────────────────────────────────

static constexpr OaU32 OUI_MOD_NONE  = 0U;
static constexpr OaU32 OUI_MOD_SHIFT = 1U << 0;
static constexpr OaU32 OUI_MOD_CTRL  = 1U << 1;
static constexpr OaU32 OUI_MOD_ALT   = 1U << 2;
static constexpr OaU32 OUI_MOD_SUPER = 1U << 3;  // Cmd on macOS


// ─── OuiKey ───────────────────────────────────────────────────────────────────
// SDL3 scancode subset.  Values deliberately match SDL_Scancode to allow
// direct cast; do not rely on the numeric values in switch logic.

enum class OuiKey : OaU8 {
	Unknown   =  0,
	A         =  4,
	B         =  5,
	C         =  6,
	D         =  7,
	E         =  8,
	F         =  9,
	G         = 10,
	H         = 11,
	I         = 12,
	J         = 13,
	K         = 14,
	L         = 15,
	M         = 16,
	N         = 17,
	O         = 18,
	P         = 19,
	Q         = 20,
	R         = 21,
	S         = 22,
	T         = 23,
	U         = 24,
	V         = 25,
	W         = 26,
	X         = 27,
	Y         = 28,
	Z         = 29,
	Num1      = 30,
	Num2      = 31,
	Num3      = 32,
	Num4      = 33,
	Num5      = 34,
	Num6      = 35,
	Num7      = 36,
	Num8      = 37,
	Num9      = 38,
	Num0      = 39,
	Return    = 40,
	Escape    = 41,
	Backspace = 42,
	Tab       = 43,
	Space     = 44,
	Minus     = 45,
	Equals    = 46,
	Comma     = 54,
	Period    = 55,
	Slash     = 56,
	Delete    = 76,
	Right     = 79,
	Left      = 80,
	Down      = 81,
	Up        = 82,
	F1        = 58,
	F2        = 59,
	F3        = 60,
	F4        = 61,
	F5        = 62,
	F6        = 63,
	F7        = 64,
	F8        = 65,
	F9        = 66,
	F10       = 67,
	F11       = 68,
	F12       = 69,
	KpEnter   = 88,
	Kp1       = 89,
	Kp2       = 90,
	Kp3       = 91,
	Kp4       = 92,
	Kp5       = 93,
	Kp6       = 94,
	Kp7       = 95,
	Kp8       = 96,
	Kp9       = 97,
	Kp0       = 98,
};


// ─── OuiEventType ─────────────────────────────────────────────────────────────

enum class OuiEventType : OaU8 {
	None         = 0,
	MouseMove    = 1,
	MouseDown    = 2,
	MouseUp      = 3,
	MouseScroll  = 4,
	KeyDown      = 5,
	KeyUp        = 6,
	KeyChar      = 7,
	WindowResize = 8,
	WindowClose  = 9,
	WindowFocus  = 10,
	WindowBlur   = 11,
	Pinch        = 12,
};


// ─── OuiScrollGesture ───────────────────────────────────────────────────────
// Classified once in the input layer (SDL wheel heuristics). Navigation maps
// each gesture to its own sensitivity — wheel zoom, touchpad pan, pinch zoom.

enum class OuiScrollGesture : OaU8 {
	None        = 0,
	MouseWheel  = 1,  // discrete notch → zoom at cursor
	TouchpadPan = 2,  // smooth two-finger scroll → pan only
	PinchScroll = 3,  // ctrl+smooth scroll (Wayland pinch fallback)
};


// ─── OuiPinchPhase ────────────────────────────────────────────────────────────

enum class OuiPinchPhase : OaU8 {
	None   = 0,
	Begin  = 1,
	Update = 2,
	End    = 3,
};


// ─── OaUiEvent ─────────────────────────────────────────────────────────────────

struct OaUiEvent {
	OuiEventType Type     = OuiEventType::None;

	// Mouse / scroll
	OaF32  MouseX    = 0.0F;
	OaF32  MouseY    = 0.0F;
	OaF32  MouseDX   = 0.0F;
	OaF32  MouseDY   = 0.0F;
	OaI32  Button    = 0;       // 1=left 2=middle 3=right
	OaF32  ScrollX   = 0.0F;
	OaF32  ScrollY   = 0.0F;
	OaF32  GestureScale = 1.0F;
	OuiPinchPhase PinchPhase = OuiPinchPhase::None;
	OaI32  IntegerScrollX = 0;  // Accumulated wheel ticks (SDL3 integer_x/y)
	OaI32  IntegerScrollY = 0;  // 0 on trackpad smooth scroll, ±1 on wheel click
	OaU64  TimestampNs   = 0;   // SDL event timestamp (scroll burst classification)
	OuiScrollGesture ScrollGesture = OuiScrollGesture::None;

	// Keyboard
	OuiKey Key       = OuiKey::Unknown;
	OaU32  Modifiers = OUI_MOD_NONE;
	bool   KeyRepeat = false;
	OaU32  Codepoint = 0;       // KeyChar: Unicode codepoint

	// Window (resize)
	OaI32  WindowW   = 0;
	OaI32  WindowH   = 0;

	[[nodiscard]] bool Ctrl()  const noexcept { return (Modifiers & OUI_MOD_CTRL)  != 0U; }
	[[nodiscard]] bool Shift() const noexcept { return (Modifiers & OUI_MOD_SHIFT) != 0U; }
	[[nodiscard]] bool Alt()   const noexcept { return (Modifiers & OUI_MOD_ALT)   != 0U; }
};


// ─── OaUiInputState ────────────────────────────────────────────────────────────
// Stable per-frame snapshot updated by OaUi::BeginFrame before widget dispatch.

struct OaUiInputState {
	OaF32 MouseX  = 0.0F;
	OaF32 MouseY  = 0.0F;
	OaF32 MouseDX = 0.0F;   // delta from previous frame
	OaF32 MouseDY = 0.0F;

	bool  LButton = false;
	bool  MButton = false;
	bool  RButton = false;
	bool  LPressed = false;
	bool  LReleased = false;

	OaF32 ScrollX = 0.0F;   // accumulated this frame
	OaF32 ScrollY = 0.0F;

	OaU32 Modifiers = OUI_MOD_NONE;

	// Focus / hover / active widget IDs (0 = none)
	OaU32 FocusId  = 0;
	OaU32 HoverId  = 0;
	OaU32 ActiveId = 0;

	[[nodiscard]] bool IsMouseOver(OaF32 InX, OaF32 InY, OaF32 InW, OaF32 InH) const noexcept {
		return MouseX >= InX and MouseX < (InX + InW)
		   and MouseY >= InY and MouseY < (InY + InH);
	}
};
