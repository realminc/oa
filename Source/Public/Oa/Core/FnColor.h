// OaFnColor — Color conversion and utility functions.
//
// Provides color space conversions (HSV, HSL, hex string parsing),
// factory functions for palette colors, and utility operations.
// Pattern: OaColor (public class) + OaFnColor (namespace functions).
// OaColor operators wrap these functions, similar to OaMatrix/OaFnMatrix.

#pragma once

#include <Oa/Core/Color.h>
#include <Oa/Core/Types.h>
#include <string>

namespace OaFnColor {

// ─── Color operations ─────────────────────────────────────────────────────────────

/// Linear interpolation between two colors
[[nodiscard]] constexpr OaColor Lerp(const OaColor& InA, const OaColor& InB, OaF32 InT) noexcept {
	return {
		InA.R + ((InB.R - InA.R) * InT),
		InA.G + ((InB.G - InA.G) * InT),
		InA.B + ((InB.B - InA.B) * InT),
		InA.A + ((InB.A - InA.A) * InT),
	};
}

/// Component-wise addition
[[nodiscard]] constexpr OaColor Add(const OaColor& InA, const OaColor& InB) noexcept {
	return {InA.R + InB.R, InA.G + InB.G, InA.B + InB.B, InA.A + InB.A};
}

/// Component-wise subtraction
[[nodiscard]] constexpr OaColor Sub(const OaColor& InA, const OaColor& InB) noexcept {
	return {InA.R - InB.R, InA.G - InB.G, InA.B - InB.B, InA.A - InB.A};
}

/// Component-wise multiplication (color modulation)
[[nodiscard]] constexpr OaColor Mul(const OaColor& InA, const OaColor& InB) noexcept {
	return {InA.R * InB.R, InA.G * InB.G, InA.B * InB.B, InA.A * InB.A};
}

/// Scalar multiplication
[[nodiscard]] constexpr OaColor Scale(const OaColor& InColor, OaF32 InScalar) noexcept {
	return {InColor.R * InScalar, InColor.G * InScalar, InColor.B * InScalar, InColor.A * InScalar};
}

/// Scalar division
[[nodiscard]] constexpr OaColor Div(const OaColor& InColor, OaF32 InScalar) noexcept {
	return {InColor.R / InScalar, InColor.G / InScalar, InColor.B / InScalar, InColor.A / InScalar};
}

/// Clamp color components to [0, 1] range
[[nodiscard]] constexpr OaColor Clamp(const OaColor& InColor) noexcept {
	auto clamp01 = [](OaF32 v) -> OaF32 {
		return v < 0.0F ? 0.0F : (v > 1.0F ? 1.0F : v);
	};
	return {clamp01(InColor.R), clamp01(InColor.G), clamp01(InColor.B), clamp01(InColor.A)};
}

// ─── HSV conversion ───────────────────────────────────────────────────────────────

struct Hsv {
	OaF32 H = 0.0F;  // [0, 360)
	OaF32 S = 0.0F;  // [0, 1]
	OaF32 V = 0.0F;  // [0, 1]
};

[[nodiscard]] Hsv RgbToHsv(const OaColor& InRgb) noexcept;
[[nodiscard]] OaColor HsvToRgb(const Hsv& InHsv) noexcept;

// ─── Hex string parsing ───────────────────────────────────────────────────────────

/// Parse hex string "#RRGGBB" or "#RRGGBBAA" to OaColor
[[nodiscard]] OaColor FromHex(const char* InHex) noexcept;
[[nodiscard]] OaColor FromHex(const std::string& InHex) noexcept;

/// Convert OaColor to hex string "#RRGGBB" or "#RRGGBBAA"
[[nodiscard]] std::string ToHex(const OaColor& InColor, bool IncludeAlpha = false) noexcept;

} // namespace OaFnColor
