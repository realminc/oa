// oa::FnColor — Color conversion and utility functions.
//
// Provides the implemented arithmetic utilities used by oa::Color.
// Pattern: oa::Color value + oa::FnColor stateless operations.
// oa::Color operators delegate to the same operation authority as oa::Matrix.

#pragma once

#include <oa/core/color.h>
#include <oa/core/types.h>

namespace oa {

namespace FnColor {

// ─── Color operations ─────────────────────────────────────────────────────────────

/// Linear interpolation between two colors
[[nodiscard]] constexpr oa::Color lerp(const oa::Color& inA, const oa::Color& inB, oa::F32 inT) noexcept {
	return {
		inA.r + ((inB.r - inA.r) * inT),
		inA.g + ((inB.g - inA.g) * inT),
		inA.b + ((inB.b - inA.b) * inT),
		inA.a + ((inB.a - inA.a) * inT),
	};
}

/// component-wise addition
[[nodiscard]] constexpr oa::Color add(const oa::Color& inA, const oa::Color& inB) noexcept {
	return {inA.r + inB.r, inA.g + inB.g, inA.b + inB.b, inA.a + inB.a};
}

/// component-wise subtraction
[[nodiscard]] constexpr oa::Color sub(const oa::Color& inA, const oa::Color& inB) noexcept {
	return {inA.r - inB.r, inA.g - inB.g, inA.b - inB.b, inA.a - inB.a};
}

/// component-wise multiplication (color modulation)
[[nodiscard]] constexpr oa::Color mul(const oa::Color& inA, const oa::Color& inB) noexcept {
	return {inA.r * inB.r, inA.g * inB.g, inA.b * inB.b, inA.a * inB.a};
}

/// scalar multiplication
[[nodiscard]] constexpr oa::Color scale(const oa::Color& inColor, oa::F32 inScalar) noexcept {
	return {inColor.r * inScalar, inColor.g * inScalar, inColor.b * inScalar, inColor.a * inScalar};
}

/// scalar division
[[nodiscard]] constexpr oa::Color div(const oa::Color& inColor, oa::F32 inScalar) noexcept {
	return {inColor.r / inScalar, inColor.g / inScalar, inColor.b / inScalar, inColor.a / inScalar};
}

/// Clamp color components to [0, 1] range
[[nodiscard]] constexpr oa::Color clamp(const oa::Color& inColor) noexcept {
	auto clamp01 = [](oa::F32 v) -> oa::F32 {
		return v < 0.0F ? 0.0F : (v > 1.0F ? 1.0F : v);
	};
	return {clamp01(inColor.r), clamp01(inColor.g), clamp01(inColor.b), clamp01(inColor.a)};
}

} // namespace FnColor

} // namespace oa
