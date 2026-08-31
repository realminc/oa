// oa::Keyframe — one sparse animation-curve sample.

#pragma once

#include <oa/core/std/chrono.h>
#include <oa/core/std/typeTraits.h>
#include <oa/core/vlm.h>

namespace oa {

enum class AnimInterpolation : oa::U8 {
	Step,
	Linear,
	CubicSpline,
};

[[nodiscard]] constexpr bool isValidAnimInterpolation(
	oa::AnimInterpolation inInterpolation) noexcept {
	switch (inInterpolation) {
		case oa::AnimInterpolation::Step:
		case oa::AnimInterpolation::Linear:
		case oa::AnimInterpolation::CubicSpline: return true;
	}
	return false;
}

template<typename T>
inline constexpr bool isAnimCurveValueV =
	oa::isSameV<T, oa::F32>
	or oa::isSameV<T, oa::vlm::Vec3>
	or oa::isSameV<T, oa::vlm::Quat>;

namespace detail {

template<typename T>
[[nodiscard]] constexpr T animCurveZero() noexcept {
	if constexpr (oa::isSameV<T, oa::vlm::Quat>) {
		return {0.0F, 0.0F, 0.0F, 0.0F};
	}
	return {};
}

} // namespace detail

template<typename T>
struct Keyframe {
	static_assert(
		oa::isAnimCurveValueV<T>,
		"Keyframe supports F32, vlm::Vec3, and vlm::Quat values");

	oa::Duration time{};
	T value{};
	T inTangent = oa::detail::animCurveZero<T>();
	T outTangent = oa::detail::animCurveZero<T>();
};

} // namespace oa
