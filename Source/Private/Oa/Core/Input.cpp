// OaInput — SDL3 backend.

#include <Oa/Core/Input.h>

#include <SDL3/SDL.h>

#include <cmath>


namespace OaInput {

namespace {

struct FrameState {
	KeyboardState Keyboard{};
	MouseState    Mouse{};

	bool PrevLeft   = false;
	bool PrevMiddle = false;
	bool PrevRight  = false;

	bool PressLeft   = false;
	bool PressMiddle = false;
	bool PressRight  = false;

	bool ReleaseLeft   = false;
	bool ReleaseMiddle = false;
	bool ReleaseRight  = false;

	OaF32 PointerX = 0.0F;
	OaF32 PointerY = 0.0F;
};

// Scroll bursts: trackpads emit rapid small deltas; mouse wheels are sparse detents.
struct ScrollBurst {
	OaU64 firstNs     = 0;
	OaU64 lastNs      = 0;
	OaU32 eventCount  = 0;
	OaF32 peakAbsY    = 0.0F;
	OaF32 peakAbsX    = 0.0F;
	bool  sawIntegerY = false;
	bool  sawIntegerX = false;
	OaU8  locked      = 0;  // OuiScrollGesture, 0 = unlocked
};

FrameState   GState{};
ScrollBurst  GScrollBurst{};
bool         GInitialized = false;

constexpr OaU64 kBurstGapNs = 150'000'000ULL;  // 150 ms idle ends a gesture

void UpdatePointer(OaF32 InX, OaF32 InY) noexcept {
	GState.PointerX = InX;
	GState.PointerY = InY;
	GState.Mouse.X = InX;
	GState.Mouse.Y = InY;
}

OaU64 ScrollNowNs(const OaUiEvent& InEvent) noexcept {
	return (InEvent.TimestampNs != 0ULL) ? InEvent.TimestampNs : SDL_GetTicksNS();
}

bool LooksLikeWheelDetent(OaF32 InAbs) noexcept {
	if (InAbs < 0.9F || InAbs > 3.5F) { return false; }
	const OaF32 nearest = std::round(InAbs);
	return std::fabs(InAbs - nearest) < 0.12F;
}

void ResetScrollBurst() noexcept {
	GScrollBurst = {};
}

OaScrollGesture ClassifyBurst(const OaUiEvent& InEvent) noexcept {
	const OaU64 nowNs = ScrollNowNs(InEvent);
	const OaF32 absX  = std::fabs(InEvent.ScrollX);
	const OaF32 absY  = std::fabs(InEvent.ScrollY);

	if (GScrollBurst.lastNs != 0ULL && nowNs - GScrollBurst.lastNs > kBurstGapNs) {
		ResetScrollBurst();
	}

	if (GScrollBurst.eventCount == 0U) {
		GScrollBurst.firstNs = nowNs;
	}
	GScrollBurst.lastNs = nowNs;
	++GScrollBurst.eventCount;
	if (absY > GScrollBurst.peakAbsY) { GScrollBurst.peakAbsY = absY; }
	if (absX > GScrollBurst.peakAbsX) { GScrollBurst.peakAbsX = absX; }
	if (InEvent.IntegerScrollY != 0) { GScrollBurst.sawIntegerY = true; }
	if (InEvent.IntegerScrollX != 0) { GScrollBurst.sawIntegerX = true; }

	if (GScrollBurst.locked != 0U) {
		return static_cast<OuiScrollGesture>(GScrollBurst.locked);
	}

	const bool verticalOnly   = absX < 0.001F;
	const bool hasHorizontal  = absX > 0.001F;
	OaScrollGesture guess     = OaScrollGesture::TouchpadPan;

	// Physical wheel: integer tick on any axis (MX Master side wheel = integer_x).
	if (GScrollBurst.sawIntegerY || GScrollBurst.sawIntegerX) {
		guess = OaScrollGesture::MouseWheel;
	} else if (hasHorizontal) {
		if (GScrollBurst.eventCount >= 2U) {
			guess = OaScrollGesture::TouchpadPan;
		} else if (LooksLikeWheelDetent(absX)) {
			guess = OaScrollGesture::MouseWheel;
		} else {
			guess = OaScrollGesture::TouchpadPan;
		}
	} else if (GScrollBurst.eventCount >= 2U) {
		guess = OaScrollGesture::TouchpadPan;
	} else if (verticalOnly && LooksLikeWheelDetent(absY)) {
		guess = OaScrollGesture::MouseWheel;
	} else if (absY < 0.55F) {
		guess = OaScrollGesture::TouchpadPan;
	}

	const bool lockNow = GScrollBurst.sawIntegerY
		|| GScrollBurst.sawIntegerX
		|| GScrollBurst.eventCount >= 2U
		|| (guess == OaScrollGesture::MouseWheel
		 && (LooksLikeWheelDetent(GScrollBurst.peakAbsY)
		  || LooksLikeWheelDetent(GScrollBurst.peakAbsX)));
	if (lockNow) {
		GScrollBurst.locked = static_cast<OaU8>(guess);
	}

	return guess;
}

}  // namespace

OaScrollGesture ClassifyScroll(const OaUiEvent& InEvent) noexcept {
	if (InEvent.Type != OuiEventType::MouseScroll) {
		return OaScrollGesture::None;
	}

	// Wayland trackpad pinch is often synthesized as ctrl+smooth scroll.
	if (InEvent.Ctrl()
	 && (std::fabs(InEvent.ScrollX) > 0.001F || std::fabs(InEvent.ScrollY) > 0.001F)) {
		ResetScrollBurst();
		return OaScrollGesture::PinchScroll;
	}

	if (std::fabs(InEvent.ScrollX) < 0.001F && std::fabs(InEvent.ScrollY) < 0.001F) {
		return OaScrollGesture::None;
	}

	return ClassifyBurst(InEvent);
}

void Initialize() {
	if (GInitialized) { return; }
	// Prefer native pinch/touch on Wayland; still handle ctrl+scroll fallback.
	SDL_SetHint(SDL_HINT_TRACKPAD_IS_TOUCH_ONLY, "0");
	GInitialized = true;
}

void Shutdown() {
	GState = {};
	ResetScrollBurst();
	GInitialized = false;
}

void ProcessEvent(const void* InPlatformEvent) {
	if (InPlatformEvent == nullptr) { return; }
	const auto& ev = *static_cast<const SDL_Event*>(InPlatformEvent);

	switch (ev.type) {
		case SDL_EVENT_MOUSE_MOTION:
			GState.Mouse.DX += ev.motion.xrel;
			GState.Mouse.DY += ev.motion.yrel;
			UpdatePointer(ev.motion.x, ev.motion.y);
			break;
		case SDL_EVENT_MOUSE_BUTTON_DOWN:
			UpdatePointer(ev.button.x, ev.button.y);
			switch (ev.button.button) {
				case SDL_BUTTON_LEFT:   GState.Mouse.Left = true; break;
				case SDL_BUTTON_MIDDLE: GState.Mouse.Middle = true; break;
				case SDL_BUTTON_RIGHT:  GState.Mouse.Right = true; break;
				default: break;
			}
			break;
		case SDL_EVENT_MOUSE_BUTTON_UP:
			UpdatePointer(ev.button.x, ev.button.y);
			switch (ev.button.button) {
				case SDL_BUTTON_LEFT:   GState.Mouse.Left = false; break;
				case SDL_BUTTON_MIDDLE: GState.Mouse.Middle = false; break;
				case SDL_BUTTON_RIGHT:  GState.Mouse.Right = false; break;
				default: break;
			}
			break;
		case SDL_EVENT_MOUSE_WHEEL:
			GState.Mouse.ScrollX += ev.wheel.x;
			GState.Mouse.ScrollY += ev.wheel.y;
			UpdatePointer(ev.wheel.mouse_x, ev.wheel.mouse_y);
			break;
		case SDL_EVENT_KEY_DOWN: {
			const int idx = static_cast<int>(ev.key.scancode);
			if (idx >= 0 && idx < 256) {
				GState.Keyboard.Keys[idx] = true;
			}
			break;
		}
		case SDL_EVENT_KEY_UP: {
			const int idx = static_cast<int>(ev.key.scancode);
			if (idx >= 0 && idx < 256) {
				GState.Keyboard.Keys[idx] = false;
			}
			break;
		}
		default:
			break;
	}
}

void Update() {
	GState.PressLeft   = GState.Mouse.Left   && !GState.PrevLeft;
	GState.PressMiddle = GState.Mouse.Middle && !GState.PrevMiddle;
	GState.PressRight  = GState.Mouse.Right  && !GState.PrevRight;

	GState.ReleaseLeft   = !GState.Mouse.Left   && GState.PrevLeft;
	GState.ReleaseMiddle = !GState.Mouse.Middle && GState.PrevMiddle;
	GState.ReleaseRight  = !GState.Mouse.Right  && GState.PrevRight;

	GState.PrevLeft   = GState.Mouse.Left;
	GState.PrevMiddle = GState.Mouse.Middle;
	GState.PrevRight  = GState.Mouse.Right;
}

void ClearForNextFrame() {
	GState.Mouse.DX = 0.0F;
	GState.Mouse.DY = 0.0F;
	GState.Mouse.ScrollX = 0.0F;
	GState.Mouse.ScrollY = 0.0F;
	GState.PressLeft = GState.PressMiddle = GState.PressRight = false;
	GState.ReleaseLeft = GState.ReleaseMiddle = GState.ReleaseRight = false;
}

const KeyboardState& GetKeyboard() noexcept { return GState.Keyboard; }
const MouseState&    GetMouse() noexcept { return GState.Mouse; }

OaF32 PointerX() noexcept { return GState.PointerX; }
OaF32 PointerY() noexcept { return GState.PointerY; }

bool Down(Button InButton) noexcept {
	switch (InButton) {
		case Button::MouseLeft:   return GState.Mouse.Left;
		case Button::MouseMiddle: return GState.Mouse.Middle;
		case Button::MouseRight:  return GState.Mouse.Right;
		default: return false;
	}
}

bool Press(Button InButton) noexcept {
	switch (InButton) {
		case Button::MouseLeft:   return GState.PressLeft;
		case Button::MouseMiddle: return GState.PressMiddle;
		case Button::MouseRight:  return GState.PressRight;
		default: return false;
	}
}

bool Release(Button InButton) noexcept {
	switch (InButton) {
		case Button::MouseLeft:   return GState.ReleaseLeft;
		case Button::MouseMiddle: return GState.ReleaseMiddle;
		case Button::MouseRight:  return GState.ReleaseRight;
		default: return false;
	}
}

}  // namespace OaInput