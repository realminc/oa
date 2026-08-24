// oa::input — SDL3 backend.

#include <oa/ui/platformInput.h>

#include <SDL3/SDL.h>

#include <cmath>
#include <iterator>


namespace oa::input {

namespace {

struct FrameState {
	KeyboardState keyboard{};
	MouseState    mouse{};
	bool prevKeys[256] = {};
	bool pressKeys[256] = {};
	bool releaseKeys[256] = {};

	bool prevLeft   = false;
	bool prevMiddle = false;
	bool prevRight  = false;

	bool pressLeft   = false;
	bool pressMiddle = false;
	bool pressRight  = false;

	bool releaseLeft   = false;
	bool releaseMiddle = false;
	bool releaseRight  = false;

	oa::F32 pointerX = 0.0F;
	oa::F32 pointerY = 0.0F;
};

// Scroll bursts: trackpads emit rapid small deltas; mouse wheels are sparse detents.
struct ScrollBurst {
	oa::U64 firstNs     = 0;
	oa::U64 lastNs      = 0;
	oa::U32 eventCount  = 0;
	oa::F32 peakAbsY    = 0.0F;
	oa::F32 peakAbsX    = 0.0F;
	bool  sawIntegerY = false;
	bool  sawIntegerX = false;
	oa::U8  locked      = 0;  // oa::UiScrollGesture, 0 = unlocked
};

FrameState   GState{};
ScrollBurst  GScrollBurst{};
bool         GInitialized = false;

constexpr oa::U64 kBurstGapNs = 150'000'000ULL;  // 150 ms idle ends a gesture

void updatePointer(oa::F32 inX, oa::F32 inY) noexcept {
	GState.pointerX = inX;
	GState.pointerY = inY;
	GState.mouse.x = inX;
	GState.mouse.y = inY;
}

oa::U64 scrollNowNs(const oa::UiEvent& inEvent) noexcept {
	return (inEvent.timestampNs != 0ULL) ? inEvent.timestampNs : SDL_GetTicksNS();
}

bool looksLikeWheelDetent(oa::F32 inAbs) noexcept {
	if (inAbs < 0.9F || inAbs > 3.5F) { return false; }
	const oa::F32 nearest = std::round(inAbs);
	return std::fabs(inAbs - nearest) < 0.12F;
}

void resetScrollBurst() noexcept {
	GScrollBurst = {};
}

oa::I32 buttonScancode(Button inButton) noexcept {
	switch (inButton) {
		case Button::KeyUp: return static_cast<oa::I32>(oa::UiKey::Up);
		case Button::KeyDown: return static_cast<oa::I32>(oa::UiKey::Down);
		case Button::KeyLeft: return static_cast<oa::I32>(oa::UiKey::Left);
		case Button::KeyRight: return static_cast<oa::I32>(oa::UiKey::Right);
		case Button::KeySpace: return static_cast<oa::I32>(oa::UiKey::Space);
		case Button::KeyEscape: return static_cast<oa::I32>(oa::UiKey::Escape);
		case Button::KeyEnter: return static_cast<oa::I32>(oa::UiKey::Return);
		default: return -1;
	}
}

oa::UiScrollGesture classifyBurst(const oa::UiEvent& inEvent) noexcept {
	const oa::U64 nowNs = scrollNowNs(inEvent);
	const oa::F32 absX  = std::fabs(inEvent.scrollX);
	const oa::F32 absY  = std::fabs(inEvent.scrollY);

	if (GScrollBurst.lastNs != 0ULL && nowNs - GScrollBurst.lastNs > kBurstGapNs) {
		resetScrollBurst();
	}

	if (GScrollBurst.eventCount == 0U) {
		GScrollBurst.firstNs = nowNs;
	}
	GScrollBurst.lastNs = nowNs;
	++GScrollBurst.eventCount;
	if (absY > GScrollBurst.peakAbsY) { GScrollBurst.peakAbsY = absY; }
	if (absX > GScrollBurst.peakAbsX) { GScrollBurst.peakAbsX = absX; }
	if (inEvent.integerScrollY != 0) { GScrollBurst.sawIntegerY = true; }
	if (inEvent.integerScrollX != 0) { GScrollBurst.sawIntegerX = true; }

	if (GScrollBurst.locked != 0U) {
		return static_cast<oa::UiScrollGesture>(GScrollBurst.locked);
	}

	const bool verticalOnly   = absX < 0.001F;
	const bool hasHorizontal  = absX > 0.001F;
	oa::UiScrollGesture guess     = oa::UiScrollGesture::TouchpadPan;

	// Physical wheel: integer tick on any axis (MX Master side wheel = integer_x).
	if (GScrollBurst.sawIntegerY || GScrollBurst.sawIntegerX) {
		guess = oa::UiScrollGesture::MouseWheel;
	} else if (hasHorizontal) {
		if (GScrollBurst.eventCount >= 2U) {
			guess = oa::UiScrollGesture::TouchpadPan;
		} else if (looksLikeWheelDetent(absX)) {
			guess = oa::UiScrollGesture::MouseWheel;
		} else {
			guess = oa::UiScrollGesture::TouchpadPan;
		}
	} else if (GScrollBurst.eventCount >= 2U) {
		guess = oa::UiScrollGesture::TouchpadPan;
	} else if (verticalOnly && looksLikeWheelDetent(absY)) {
		guess = oa::UiScrollGesture::MouseWheel;
	} else if (absY < 0.55F) {
		guess = oa::UiScrollGesture::TouchpadPan;
	}

	const bool lockNow = GScrollBurst.sawIntegerY
		|| GScrollBurst.sawIntegerX
		|| GScrollBurst.eventCount >= 2U
		|| (guess == oa::UiScrollGesture::MouseWheel
		 && (looksLikeWheelDetent(GScrollBurst.peakAbsY)
		  || looksLikeWheelDetent(GScrollBurst.peakAbsX)));
	if (lockNow) {
		GScrollBurst.locked = static_cast<oa::U8>(guess);
	}

	return guess;
}

}  // namespace

oa::UiScrollGesture classifyScroll(const oa::UiEvent& inEvent) noexcept {
	if (inEvent.type != oa::UiEventType::MouseScroll) {
		return oa::UiScrollGesture::None;
	}

	// Wayland trackpad pinch is often synthesized as ctrl+smooth scroll.
	if (inEvent.ctrl()
	 && (std::fabs(inEvent.scrollX) > 0.001F || std::fabs(inEvent.scrollY) > 0.001F)) {
		resetScrollBurst();
		return oa::UiScrollGesture::PinchScroll;
	}

	if (std::fabs(inEvent.scrollX) < 0.001F && std::fabs(inEvent.scrollY) < 0.001F) {
		return oa::UiScrollGesture::None;
	}

	return classifyBurst(inEvent);
}

void initialize() {
	if (GInitialized) { return; }
	// Prefer native pinch/touch on Wayland; still handle ctrl+scroll fallback.
	SDL_SetHint(SDL_HINT_TRACKPAD_IS_TOUCH_ONLY, "0");
	GInitialized = true;
}

void shutdown() {
	GState = {};
	resetScrollBurst();
	GInitialized = false;
}

void processEvent(const void* inPlatformEvent) {
	if (inPlatformEvent == nullptr) { return; }
	const auto& ev = *static_cast<const SDL_Event*>(inPlatformEvent);

	switch (ev.type) {
		case SDL_EVENT_MOUSE_MOTION:
			GState.mouse.dX += ev.motion.xrel;
			GState.mouse.dY += ev.motion.yrel;
			updatePointer(ev.motion.x, ev.motion.y);
			break;
		case SDL_EVENT_MOUSE_BUTTON_DOWN:
			updatePointer(ev.button.x, ev.button.y);
			switch (ev.button.button) {
				case SDL_BUTTON_LEFT:   GState.mouse.left = true; break;
				case SDL_BUTTON_MIDDLE: GState.mouse.middle = true; break;
				case SDL_BUTTON_RIGHT:  GState.mouse.right = true; break;
				default: break;
			}
			break;
		case SDL_EVENT_MOUSE_BUTTON_UP:
			updatePointer(ev.button.x, ev.button.y);
			switch (ev.button.button) {
				case SDL_BUTTON_LEFT:   GState.mouse.left = false; break;
				case SDL_BUTTON_MIDDLE: GState.mouse.middle = false; break;
				case SDL_BUTTON_RIGHT:  GState.mouse.right = false; break;
				default: break;
			}
			break;
		case SDL_EVENT_MOUSE_WHEEL:
			GState.mouse.scrollX += ev.wheel.x;
			GState.mouse.scrollY += ev.wheel.y;
			updatePointer(ev.wheel.mouse_x, ev.wheel.mouse_y);
			break;
		case SDL_EVENT_KEY_DOWN: {
			const int idx = static_cast<int>(ev.key.scancode);
			if (idx >= 0 && idx < 256) {
				GState.keyboard.keys[idx] = true;
			}
			break;
		}
		case SDL_EVENT_KEY_UP: {
			const int idx = static_cast<int>(ev.key.scancode);
			if (idx >= 0 && idx < 256) {
				GState.keyboard.keys[idx] = false;
			}
			break;
		}
		default:
			break;
	}
}

void update() {
	for (oa::Usize index = 0U; index < std::size(GState.keyboard.keys); ++index) {
		GState.pressKeys[index] = GState.keyboard.keys[index]
			&& not GState.prevKeys[index];
		GState.releaseKeys[index] = not GState.keyboard.keys[index]
			&& GState.prevKeys[index];
		GState.prevKeys[index] = GState.keyboard.keys[index];
	}
	GState.pressLeft   = GState.mouse.left   && !GState.prevLeft;
	GState.pressMiddle = GState.mouse.middle && !GState.prevMiddle;
	GState.pressRight  = GState.mouse.right  && !GState.prevRight;

	GState.releaseLeft   = !GState.mouse.left   && GState.prevLeft;
	GState.releaseMiddle = !GState.mouse.middle && GState.prevMiddle;
	GState.releaseRight  = !GState.mouse.right  && GState.prevRight;

	GState.prevLeft   = GState.mouse.left;
	GState.prevMiddle = GState.mouse.middle;
	GState.prevRight  = GState.mouse.right;
}

void clearForNextFrame() {
	GState.mouse.dX = 0.0F;
	GState.mouse.dY = 0.0F;
	GState.mouse.scrollX = 0.0F;
	GState.mouse.scrollY = 0.0F;
	GState.pressLeft = GState.pressMiddle = GState.pressRight = false;
	GState.releaseLeft = GState.releaseMiddle = GState.releaseRight = false;
	for (oa::Usize index = 0U; index < std::size(GState.pressKeys); ++index) {
		GState.pressKeys[index] = false;
		GState.releaseKeys[index] = false;
	}
}

const KeyboardState& getKeyboard() noexcept { return GState.keyboard; }
const MouseState&    getMouse() noexcept { return GState.mouse; }

oa::F32 pointerX() noexcept { return GState.pointerX; }
oa::F32 pointerY() noexcept { return GState.pointerY; }

bool down(Button inButton) noexcept {
	switch (inButton) {
		case Button::MouseLeft:   return GState.mouse.left;
		case Button::MouseMiddle: return GState.mouse.middle;
		case Button::MouseRight:  return GState.mouse.right;
		default: {
			const oa::I32 scancode = buttonScancode(inButton);
			return scancode >= 0 && scancode < 256
				&& GState.keyboard.keys[scancode];
		}
	}
}

bool press(Button inButton) noexcept {
	switch (inButton) {
		case Button::MouseLeft:   return GState.pressLeft;
		case Button::MouseMiddle: return GState.pressMiddle;
		case Button::MouseRight:  return GState.pressRight;
		default: {
			const oa::I32 scancode = buttonScancode(inButton);
			return scancode >= 0 && scancode < 256
				&& GState.pressKeys[scancode];
		}
	}
}

bool release(Button inButton) noexcept {
	switch (inButton) {
		case Button::MouseLeft:   return GState.releaseLeft;
		case Button::MouseMiddle: return GState.releaseMiddle;
		case Button::MouseRight:  return GState.releaseRight;
		default: {
			const oa::I32 scancode = buttonScancode(inButton);
			return scancode >= 0 && scancode < 256
				&& GState.releaseKeys[scancode];
		}
	}
}

}  // namespace oa::input
