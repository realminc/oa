// OA UI motion policy — one speed contract for widgets and viewport navigation.

#pragma once

#include <oa/core/types.h>

namespace oa {

enum class UiMotionSpeed : oa::U8 {
	Normal = 0,
	Fast,
	Fastest,
	Instant,
};

[[nodiscard]] constexpr bool isValidUiMotionSpeed(
	oa::UiMotionSpeed inSpeed) noexcept {
	switch (inSpeed) {
		case oa::UiMotionSpeed::Normal:
		case oa::UiMotionSpeed::Fast:
		case oa::UiMotionSpeed::Fastest:
		case oa::UiMotionSpeed::Instant:
			return true;
	}
	return false;
}

[[nodiscard]] constexpr oa::F32 uiMotionDurationMs(
	oa::F32 inNormalDurationMs,
	oa::UiMotionSpeed inSpeed) noexcept {
	switch (inSpeed) {
		case oa::UiMotionSpeed::Normal: return inNormalDurationMs;
		case oa::UiMotionSpeed::Fast: return inNormalDurationMs * 0.65F;
		case oa::UiMotionSpeed::Fastest: return inNormalDurationMs * 0.35F;
		case oa::UiMotionSpeed::Instant: return 0.0F;
	}
	return 0.0F;
}

[[nodiscard]] constexpr const char* uiMotionSpeedName(
	oa::UiMotionSpeed inSpeed) noexcept {
	switch (inSpeed) {
		case oa::UiMotionSpeed::Normal: return "Normal";
		case oa::UiMotionSpeed::Fast: return "Fast";
		case oa::UiMotionSpeed::Fastest: return "Fastest";
		case oa::UiMotionSpeed::Instant: return "Instant";
	}
	return "Invalid";
}

[[nodiscard]] constexpr oa::UiMotionSpeed nextUiMotionSpeed(
	oa::UiMotionSpeed inSpeed) noexcept {
	switch (inSpeed) {
		case oa::UiMotionSpeed::Normal: return oa::UiMotionSpeed::Fast;
		case oa::UiMotionSpeed::Fast: return oa::UiMotionSpeed::Fastest;
		case oa::UiMotionSpeed::Fastest: return oa::UiMotionSpeed::Instant;
		case oa::UiMotionSpeed::Instant: return oa::UiMotionSpeed::Normal;
	}
	return oa::UiMotionSpeed::Normal;
}

} // namespace oa
