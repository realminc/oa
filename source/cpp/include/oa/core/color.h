// Color — RGBA color type used throughout the codebase.
//
// Provides RGBA float color with pack/unpack utilities, lerp, and alpha modification.
// Used in UI, plotting, animation, detection overlays, and visualization.

#pragma once

#include <oa/core/types.h>
#include <oa/core/std/scalarMath.h>

namespace oa {

class Color {
public:
	oa::F32 r = 0.0F;
	oa::F32 g = 0.0F;
	oa::F32 b = 0.0F;
	oa::F32 a = 1.0F;

	constexpr Color() = default;
	constexpr Color(oa::F32 inR, oa::F32 inG, oa::F32 inB, oa::F32 inA = 1.0F)
		: r(inR), g(inG), b(inB), a(inA)
	{}

	~Color() = default;

	/// Pack to 0xRRGGBBAA u32
	[[nodiscard]] oa::U32 toU32() const noexcept {
		auto to8 = [](oa::F32 v) -> oa::U32 {
			v = v < 0.0F ? 0.0F : v;
			v = v > 1.0F ? 1.0F : v;
			return static_cast<oa::U32>(oa::round(v * 255.0F));
		};
		return (to8(r) << 24) | (to8(g) << 16) | (to8(b) << 8) | to8(a);
	}

	/// Unpack from 0xRRGGBBAA u32
	[[nodiscard]] static constexpr Color fromU32(oa::U32 inRgba) noexcept {
		return {
			static_cast<oa::F32>((inRgba >> 24) & 0xFFU) / 255.0F,
			static_cast<oa::F32>((inRgba >> 16) & 0xFFU) / 255.0F,
			static_cast<oa::F32>((inRgba >>  8) & 0xFFU) / 255.0F,
			static_cast<oa::F32>( inRgba        & 0xFFU) / 255.0F,
		};
	}

	/// Return a copy with modified alpha channel
	[[nodiscard]] constexpr Color withAlpha(oa::F32 inA) const noexcept {
		return {r, g, b, inA};
	}

	/// Linear interpolation between this color and another
	[[nodiscard]] Color lerp(const Color& inOther, oa::F32 inT) const;

	// ─── Palette factory functions (Realm Design System) ──────────────────────

	[[nodiscard]] static constexpr Color accent()      noexcept { return {0.388F, 0.400F, 0.945F, 1.0F}; }
	[[nodiscard]] static constexpr Color accentHover() noexcept { return {0.506F, 0.549F, 0.973F, 1.0F}; }
	[[nodiscard]] static constexpr Color success()     noexcept { return {0.188F, 0.820F, 0.345F, 1.0F}; }
	[[nodiscard]] static constexpr Color warning()     noexcept { return {0.961F, 0.620F, 0.043F, 1.0F}; }
	[[nodiscard]] static constexpr Color error()       noexcept { return {1.000F, 0.271F, 0.227F, 1.0F}; }
	[[nodiscard]] static constexpr Color orange()      noexcept { return {1.000F, 0.420F, 0.208F, 1.0F}; }
	[[nodiscard]] static constexpr Color purple()      noexcept { return {0.659F, 0.333F, 0.969F, 1.0F}; }
	[[nodiscard]] static constexpr Color cyan()        noexcept { return {0.133F, 0.827F, 0.933F, 1.0F}; }
	[[nodiscard]] static constexpr Color pink()        noexcept { return {0.925F, 0.282F, 0.600F, 1.0F}; }
	[[nodiscard]] static constexpr Color yellow()      noexcept { return {0.961F, 0.620F, 0.043F, 1.0F}; }
	[[nodiscard]] static constexpr Color textPrimary() noexcept { return {0.961F, 0.961F, 0.961F, 1.0F}; }

	Color(const Color&) = default;
	Color& operator=(const Color&) = default;
	Color(Color&&) noexcept = default;
	Color& operator=(Color&&) noexcept = default;

	[[nodiscard]] constexpr bool operator==(const Color& inOther) const noexcept {
		return r == inOther.r and g == inOther.g and b == inOther.b and a == inOther.a;
	}
	[[nodiscard]] constexpr bool operator!=(const Color& inOther) const noexcept {
		return !(*this == inOther);
	}

	[[nodiscard]] Color operator+(const Color& inOther) const;
	[[nodiscard]] Color operator-(const Color& inOther) const;
	[[nodiscard]] Color operator*(const Color& inOther) const;
	[[nodiscard]] Color operator*(oa::F32 inScalar) const;
	[[nodiscard]] Color operator/(oa::F32 inScalar) const;
	Color& operator+=(const Color& inOther);
	Color& operator-=(const Color& inOther);
	Color& operator*=(const Color& inOther);
	Color& operator*=(oa::F32 inScalar);
	Color& operator/=(oa::F32 inScalar);
};

[[nodiscard]] inline Color operator*(oa::F32 inScalar, const Color& inColor) {
	return inColor * inScalar;
}

// Legacy alias — remove once call sites are migrated.

} // namespace oa
