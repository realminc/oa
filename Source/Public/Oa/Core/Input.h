// OaInput — platform-neutral input state + gesture classification.
//
// Low-level device layer (WickedEngine wiInput / wiSDLInput analogue).
// SDL3 backend lives in Input.cpp; public headers stay SDL-free.
//
// Ui/Input.h (OaInputSystem) is the high-level key-action registry on top.
//
// Usage:
//   OaInput::Initialize();
//   // each frame, before clearing deltas:
//   for (auto& e : sdlEvents) OaInput::ProcessEvent(e);
//   OaInput::Update();
//   const auto& mouse = OaInput::GetMouse();
//   OaInput::ClearForNextFrame();

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Ui/Event.h>


namespace OaInput {

// ─── Scroll / gesture classification ─────────────────────────────────────────
// SDL scroll bursts: integer ticks → mouse wheel pan; rapid small deltas → touchpad pan;
// ctrl+scroll → dolly Z.

using OaScrollGesture = OuiScrollGesture;

[[nodiscard]] OaScrollGesture ClassifyScroll(const OaUiEvent& InEvent) noexcept;

// ─── Button enum (stable order; extend for gamepad later) ────────────────────

enum class Button : OaU16 {
	None = 0,

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
};

// ─── Per-device state snapshots ──────────────────────────────────────────────

struct KeyboardState {
	bool Keys[256] = {};
};

struct MouseState {
	OaF32 X = 0.0F;
	OaF32 Y = 0.0F;
	OaF32 DX = 0.0F;
	OaF32 DY = 0.0F;
	OaF32 ScrollX = 0.0F;
	OaF32 ScrollY = 0.0F;
	OaF32 Pressure = 1.0F;
	bool  Left = false;
	bool  Middle = false;
	bool  Right = false;
};

// ─── Lifecycle ─────────────────────────────────────────────────────────────────

void Initialize();
void Shutdown();

// Feed raw platform events (SDL_Event in Input.cpp).
void ProcessEvent(const void* InPlatformEvent);

// End-of-poll: fold queued events into state snapshots.
void Update();

// Clear per-frame deltas (scroll, motion delta, press edges).
void ClearForNextFrame();

[[nodiscard]] const KeyboardState& GetKeyboard() noexcept;
[[nodiscard]] const MouseState&    GetMouse() noexcept;

[[nodiscard]] bool Down(Button InButton) noexcept;
[[nodiscard]] bool Press(Button InButton) noexcept;
[[nodiscard]] bool Release(Button InButton) noexcept;

// Last known pointer position (updated on motion, buttons, wheel).
[[nodiscard]] OaF32 PointerX() noexcept;
[[nodiscard]] OaF32 PointerY() noexcept;

}  // namespace OaInput