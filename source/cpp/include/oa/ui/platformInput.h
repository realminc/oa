// oa::input — platform-neutral input snapshots and gesture classification.
//
// The SDL3 backend lives in PlatformInput.cpp; public headers stay SDL-free.
//
// <oa/ui/input.h> (oa::InputSystem) is the high-level key-action registry on top.
//
// usage:
//   oa::input::initialize();
//   // each frame, before clearing deltas:
//   for (auto& e : sdlEvents) oa::input::processEvent(e);
//   oa::input::update();
//   const auto& mouse = oa::input::getMouse();
//   oa::input::clearForNextFrame();

#pragma once

#include <oa/core/types.h>
#include <oa/ui/event.h>


namespace oa::input {

// ─── Scroll / gesture classification ─────────────────────────────────────────
// SDL scroll bursts: integer ticks → mouse wheel pan; rapid small deltas → touchpad pan;
// ctrl+scroll → dolly Z.

[[nodiscard]] oa::UiScrollGesture classifyScroll(const oa::UiEvent& inEvent) noexcept;

// ─── Button enum (stable order; extend for gamepad later) ────────────────────

enum class Button : oa::U16 {
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
	bool keys[256] = {};
};

struct MouseState {
	oa::F32 x = 0.0F;
	oa::F32 y = 0.0F;
	oa::F32 dX = 0.0F;
	oa::F32 dY = 0.0F;
	oa::F32 scrollX = 0.0F;
	oa::F32 scrollY = 0.0F;
	oa::F32 pressure = 1.0F;
	bool  left = false;
	bool  middle = false;
	bool  right = false;
};

// ─── Lifecycle ─────────────────────────────────────────────────────────────────

void initialize();
void shutdown();

// Feed raw platform events (SDL_Event in input.cpp).
void processEvent(const void* inPlatformEvent);

// End-of-poll: fold queued events into state snapshots.
void update();

// clear per-frame deltas (scroll, motion delta, press edges).
void clearForNextFrame();

[[nodiscard]] const KeyboardState& getKeyboard() noexcept;
[[nodiscard]] const MouseState&    getMouse() noexcept;

// mouse and the declared keyboard buttons all report held/rising/falling state.
// Press/release remain valid until ClearForNextFrame.
[[nodiscard]] bool down(Button inButton) noexcept;
[[nodiscard]] bool press(Button inButton) noexcept;
[[nodiscard]] bool release(Button inButton) noexcept;

// Last known pointer position (updated on motion, buttons, wheel).
[[nodiscard]] oa::F32 pointerX() noexcept;
[[nodiscard]] oa::F32 pointerY() noexcept;

}  // namespace oa::input
