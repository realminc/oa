// Ui — input events and per-frame input state.
//
// Events are polled per-frame from the windowing layer (SDL3).
// UiInputState is the stable per-frame snapshot; widgets read it instead of
// consuming the event queue directly.

#pragma once

#include <oa/core/types.h>


// ─── modifier mask ────────────────────────────────────────────────────────────

namespace oa {

inline constexpr oa::U32 UiModifierNone  = 0U;
inline constexpr oa::U32 UiModifierShift = 1U << 0;
inline constexpr oa::U32 UiModifierCtrl  = 1U << 1;
inline constexpr oa::U32 UiModifierAlt   = 1U << 2;
inline constexpr oa::U32 UiModifierSuper = 1U << 3;  // Cmd on macOS


// ─── UiKey ───────────────────────────────────────────────────────────────────
// SDL3 scancode subset.  values deliberately match SDL_Scancode to allow
// direct cast; do not rely on the numeric values in switch logic.

enum class UiKey : oa::U8 {
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
	Home      = 74,
	Delete    = 76,
	End       = 77,
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


// ─── UiEventType ─────────────────────────────────────────────────────────────

enum class UiEventType : oa::U8 {
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
	TextEditing  = 13,
};


// ─── UiScrollGesture ───────────────────────────────────────────────────────
// Classified once in the input layer (SDL wheel heuristics). Navigation maps
// each gesture to its own sensitivity — wheel zoom, touchpad pan, pinch zoom.

enum class UiScrollGesture : oa::U8 {
	None        = 0,
	MouseWheel  = 1,  // discrete notch → zoom at cursor
	TouchpadPan = 2,  // smooth two-finger scroll → pan only
	PinchScroll = 3,  // ctrl+smooth scroll (Wayland pinch fallback)
};


// ─── UiPinchPhase ────────────────────────────────────────────────────────────

enum class UiPinchPhase : oa::U8 {
	None   = 0,
	Begin  = 1,
	Update = 2,
	End    = 3,
};


// ─── UiEvent ─────────────────────────────────────────────────────────────────

struct UiEvent {
	UiEventType type     = UiEventType::None;

	// mouse / scroll
	oa::F32  mouseX    = 0.0F;
	oa::F32  mouseY    = 0.0F;
	oa::F32  mouseDX   = 0.0F;
	oa::F32  mouseDY   = 0.0F;
	oa::I32  button    = 0;       // 1=left 2=middle 3=right
	oa::I32  clickCount = 0;      // 1=single, 2=word, 3+=all for left press
	oa::F32  scrollX   = 0.0F;
	oa::F32  scrollY   = 0.0F;
	oa::F32  gestureScale = 1.0F;
	UiPinchPhase pinchPhase = UiPinchPhase::None;
	oa::I32  integerScrollX = 0;  // Accumulated wheel ticks (SDL3 integer_x/y)
	oa::I32  integerScrollY = 0;  // 0 on trackpad smooth scroll, ±1 on wheel click
	oa::U64  timestampNs   = 0;   // SDL event timestamp (scroll burst classification)
	UiScrollGesture scrollGesture = UiScrollGesture::None;

	// keyboard
	UiKey key       = UiKey::Unknown;
	oa::U32  modifiers = UiModifierNone;
	bool   keyRepeat = false;
	// KeyChar carries committed UTF-8 text from the platform input method.
	// codepoint remains available for embedders that emit one scalar at a time;
	// text takes precedence when both are present.
	oa::U32  codepoint = 0;
	oa::String text;
	// TextEditing carries the platform IME pre-edit string. Selection offsets
	// count Unicode scalars within text, matching SDL3's UTF-8-character
	// contract; -1 means the input method did not provide the value.
	oa::I32 textSelectionStart = -1;
	oa::I32 textSelectionLength = -1;

	// Window (resize)
	oa::I32  windowW   = 0;
	oa::I32  windowH   = 0;

	[[nodiscard]] bool ctrl()  const noexcept { return (modifiers & UiModifierCtrl)  != 0U; }
	[[nodiscard]] bool shift() const noexcept { return (modifiers & UiModifierShift) != 0U; }
	[[nodiscard]] bool alt()   const noexcept { return (modifiers & UiModifierAlt)   != 0U; }
};


// ─── UiInputState ────────────────────────────────────────────────────────────
// Stable per-frame snapshot updated by Ui::beginFrame before widget dispatch.

struct UiInputState {
	oa::F32 mouseX  = 0.0F;
	oa::F32 mouseY  = 0.0F;
	oa::F32 mouseDX = 0.0F;   // delta from previous frame
	oa::F32 mouseDY = 0.0F;

	bool  lButton = false;
	bool  mButton = false;
	bool  rButton = false;
	bool  lPressed = false;
	bool  lReleased = false;
	oa::I32 lClickCount = 0;

	oa::F32 scrollX = 0.0F;   // accumulated this frame
	oa::F32 scrollY = 0.0F;

	oa::U32 modifiers = UiModifierNone;

	// Focus / hover / active widget iDs (0 = none)
	oa::U32 focusId  = 0;
	oa::U32 hoverId  = 0;
	oa::U32 activeId = 0;

	[[nodiscard]] bool isMouseOver(oa::F32 inX, oa::F32 inY, oa::F32 inW, oa::F32 inH) const noexcept {
		return mouseX >= inX and mouseX < (inX + inW)
		   and mouseY >= inY and mouseY < (inY + inH);
	}
};

}  // namespace oa
