// OaColor — RGBA color type used throughout the codebase.
//
// Provides RGBA float color with pack/unpack utilities, lerp, and alpha modification.
// Used in UI, plotting, animation, detection overlays, and visualization.
//
// Pattern: OaColor (public class) + OaFnColor (namespace functions).
// Operators wrap OaFnColor functions, similar to OaMatrix/OaFnMatrix.

#pragma once

#include <cmath>
#include <Oa/Core/Types.h>

class OaColor {
public:
	// Data members.
	OaF32 R = 0.0F;
	OaF32 G = 0.0F;
	OaF32 B = 0.0F;
	OaF32 A = 1.0F;

	// Constructors.
	constexpr OaColor() = default;
	constexpr OaColor(OaF32 InR, OaF32 InG, OaF32 InB, OaF32 InA = 1.0F)
		: R(InR), G(InG), B(InB), A(InA)
  {}

	// Destructors.
	~OaColor() = default;

	// Methods.
	
	/// Pack to 0xRRGGBBAA u32
	[[nodiscard]] OaU32 ToU32() const noexcept {
		auto to8 = [](OaF32 v) -> OaU32 {
			v = v < 0.0F ? 0.0F : v;
			v = v > 1.0F ? 1.0F : v;
			return static_cast<OaU32>(std::lroundf(v * 255.0F));
		};
		return (to8(R) << 24) | (to8(G) << 16) | (to8(B) << 8) | to8(A);
	}

	/// Unpack from 0xRRGGBBAA u32
	[[nodiscard]] static constexpr OaColor FromU32(OaU32 InRgba) noexcept {
		return {
			static_cast<OaF32>((InRgba >> 24) & 0xFFU) / 255.0F,
			static_cast<OaF32>((InRgba >> 16) & 0xFFU) / 255.0F,
			static_cast<OaF32>((InRgba >>  8) & 0xFFU) / 255.0F,
			static_cast<OaF32>( InRgba        & 0xFFU) / 255.0F,
		};
	}

	/// Return a copy with modified alpha channel
	[[nodiscard]] constexpr OaColor WithAlpha(OaF32 InA) const noexcept {
		return {R, G, B, InA};
	}

	/// Linear interpolation between this color and another
	[[nodiscard]] OaColor Lerp(const OaColor& InOther, OaF32 InT) const;

	// ─── Palette factory functions (Realm Design System) ─────────────────────────
	
	[[nodiscard]] static constexpr OaColor Accent()      noexcept { return {0.388F, 0.400F, 0.945F, 1.0F}; }
	[[nodiscard]] static constexpr OaColor AccentHover() noexcept { return {0.506F, 0.549F, 0.973F, 1.0F}; }
	[[nodiscard]] static constexpr OaColor Success()     noexcept { return {0.188F, 0.820F, 0.345F, 1.0F}; }
	[[nodiscard]] static constexpr OaColor Warning()     noexcept { return {0.961F, 0.620F, 0.043F, 1.0F}; }
	[[nodiscard]] static constexpr OaColor Error()       noexcept { return {1.000F, 0.271F, 0.227F, 1.0F}; }
	[[nodiscard]] static constexpr OaColor Orange()      noexcept { return {1.000F, 0.420F, 0.208F, 1.0F}; }
	[[nodiscard]] static constexpr OaColor Purple()      noexcept { return {0.659F, 0.333F, 0.969F, 1.0F}; }
	[[nodiscard]] static constexpr OaColor Cyan()        noexcept { return {0.133F, 0.827F, 0.933F, 1.0F}; }
	[[nodiscard]] static constexpr OaColor Pink()        noexcept { return {0.925F, 0.282F, 0.600F, 1.0F}; }
	[[nodiscard]] static constexpr OaColor Yellow()      noexcept { return {0.961F, 0.620F, 0.043F, 1.0F}; }
	[[nodiscard]] static constexpr OaColor TextPrimary() noexcept { return {0.961F, 0.961F, 0.961F, 1.0F}; }

	// Operators.
	OaColor(const OaColor&) = default;
	OaColor& operator=(const OaColor&) = default;

	OaColor(OaColor&&) noexcept = default;
	OaColor& operator=(OaColor&&) noexcept = default;

	/// Equality comparison
	[[nodiscard]] constexpr bool operator==(const OaColor& InOther) const noexcept {
		return R == InOther.R and G == InOther.G and B == InOther.B and A == InOther.A;
	}

	/// Inequality comparison
	[[nodiscard]] constexpr bool operator!=(const OaColor& InOther) const noexcept {
		return !(*this == InOther);
	}

	/// Component-wise addition
	[[nodiscard]] OaColor operator+(const OaColor& InOther) const;

	/// Component-wise subtraction
	[[nodiscard]] OaColor operator-(const OaColor& InOther) const;

	/// Component-wise multiplication (modulation)
	[[nodiscard]] OaColor operator*(const OaColor& InOther) const;

	/// Scalar multiplication
	[[nodiscard]] OaColor operator*(OaF32 InScalar) const;

	/// Scalar division
	[[nodiscard]] OaColor operator/(OaF32 InScalar) const;

	/// Component-wise addition assignment
	OaColor& operator+=(const OaColor& InOther);

	/// Component-wise subtraction assignment
	OaColor& operator-=(const OaColor& InOther);

	/// Component-wise multiplication assignment
	OaColor& operator*=(const OaColor& InOther);

	/// Scalar multiplication assignment
	OaColor& operator*=(OaF32 InScalar);

	/// Scalar division assignment
	OaColor& operator/=(OaF32 InScalar);
};

/// Scalar multiplication (scalar on left side)
[[nodiscard]] inline OaColor operator*(OaF32 InScalar, const OaColor& InColor) {
	return InColor * InScalar;
}
