// oa::Fixed<Decimals> — Fixed-Point Arithmetic
//
// Byte-stable deterministic fixed-point math for financial calculations.
// No floating-point rounding errors. Cross-platform identical results.

#pragma once

#include <oa/core/types.h>

namespace oa {

template<oa::I32 Decimals>
class Fixed {
public:
	// FIXED-POINT CLASS

	// Data, class members.
	oa::I64 raw;

	// Constructors.
	OA_HOST_DEVICE constexpr Fixed()
		: raw(0)
	{}
	OA_HOST_DEVICE constexpr explicit Fixed(oa::I64 inRaw)
		: raw(inRaw)
	{}

	// Methods.
	static constexpr oa::I64 kScale = []() constexpr {
		oa::I64 s = 1;
		for (oa::I32 i = 0; i < Decimals; ++i) s *= 10;
		return s;
	}();
	[[nodiscard]] OA_HOST_DEVICE static constexpr Fixed fromRaw(oa::I64 inRaw) { return Fixed(inRaw); }
	[[nodiscard]] OA_HOST_DEVICE static constexpr Fixed fromInt(oa::I64 inValue) { return Fixed(inValue * kScale); }
	[[nodiscard]] OA_HOST_DEVICE static constexpr Fixed fromDouble(oa::F64 inValue) {
		return Fixed(static_cast<oa::I64>(inValue * static_cast<oa::F64>(kScale) + (inValue >= 0 ? 0.5 : -0.5)));
	}
	[[nodiscard]] OA_HOST_DEVICE static constexpr Fixed fromFloat(oa::F32 inValue) { return fromDouble(static_cast<oa::F64>(inValue)); }
	[[nodiscard]] OA_HOST_DEVICE static constexpr Fixed zero() { return Fixed(0); }
	[[nodiscard]] OA_HOST_DEVICE static constexpr Fixed one() { return Fixed(kScale); }
	[[nodiscard]] OA_HOST_DEVICE static constexpr Fixed max() { return Fixed(oa::I64Max); }
	[[nodiscard]] OA_HOST_DEVICE static constexpr Fixed min() { return Fixed(oa::I64Min); }

	[[nodiscard]] OA_HOST_DEVICE constexpr oa::F64 toDouble() const { return static_cast<oa::F64>(raw) / static_cast<oa::F64>(kScale); }
	[[nodiscard]] OA_HOST_DEVICE constexpr oa::F32 toFloat() const { return static_cast<oa::F32>(toDouble()); }
	[[nodiscard]] OA_HOST_DEVICE constexpr oa::I64 toInt() const { return raw / kScale; }
	[[nodiscard]] OA_HOST_DEVICE constexpr oa::I64 getRaw() const { return raw; }

	[[nodiscard]] OA_HOST_DEVICE constexpr bool isZero() const { return raw == 0; }
	[[nodiscard]] OA_HOST_DEVICE constexpr bool isPositive() const { return raw > 0; }
	[[nodiscard]] OA_HOST_DEVICE constexpr bool isNegative() const { return raw < 0; }

	[[nodiscard]] OA_HOST_DEVICE constexpr Fixed applyPercent(Fixed inPercent) const { return *this * inPercent; }
	[[nodiscard]] OA_HOST_DEVICE static constexpr Fixed fromPercent(oa::I64 inPercent) { return Fixed::fromRaw(inPercent * (kScale / 100)); }
	[[nodiscard]] OA_HOST_DEVICE static constexpr Fixed fromBps(oa::I64 inBps) { return Fixed::fromRaw(inBps * (kScale / 10000)); }

	[[nodiscard]] OA_HOST_DEVICE constexpr Fixed abs() const { return Fixed(raw >= 0 ? raw : -raw); }
	[[nodiscard]] OA_HOST_DEVICE constexpr Fixed minWith(Fixed inOther) const { return Fixed(raw < inOther.raw ? raw : inOther.raw); }
	[[nodiscard]] OA_HOST_DEVICE constexpr Fixed maxWith(Fixed inOther) const { return Fixed(raw > inOther.raw ? raw : inOther.raw); }
	[[nodiscard]] OA_HOST_DEVICE constexpr Fixed clamp(Fixed inLo, Fixed inHi) const {
		if (raw < inLo.raw) {
			return inLo;
		}
		if (raw > inHi.raw) {
			return inHi;
		}
		return *this;
	}

	[[nodiscard]] oa::String toString() const {
		oa::String result;
		oa::I64 absRaw = raw >= 0 ? raw : -raw;
		oa::I64 intPart = absRaw / kScale;
		oa::I64 fracPart = absRaw % kScale;
		if (raw < 0) {
			result += '-';
		}
		result += oa::toString(static_cast<oa::I64>(intPart));
		result += '.';
		oa::String fracStr(oa::toString(static_cast<oa::I64>(fracPart)));
		for (oa::I32 i = 0; i < Decimals - static_cast<oa::I32>(fracStr.size()); ++i) {
			result += '0';
		}
		result += fracStr;
		return result;
	}

	// Operators.
	[[nodiscard]] OA_HOST_DEVICE constexpr Fixed operator+(Fixed inOther) const { return Fixed(raw + inOther.raw); }
	[[nodiscard]] OA_HOST_DEVICE constexpr Fixed operator-(Fixed inOther) const { return Fixed(raw - inOther.raw); }
	[[nodiscard]] OA_HOST_DEVICE constexpr Fixed operator-() const { return Fixed(-raw); }

	[[nodiscard]] OA_HOST_DEVICE constexpr Fixed operator*(Fixed inOther) const {
#ifdef OA_COMPILER_MSVC
		return Fixed((raw * inOther.raw) / kScale);
#else
		oa::I128 result = static_cast<oa::I128>(raw) * static_cast<oa::I128>(inOther.raw);
		return Fixed(static_cast<oa::I64>(result / kScale));
#endif
	}

	[[nodiscard]] OA_HOST_DEVICE constexpr Fixed operator/(Fixed inOther) const {
		if (inOther.raw == 0) return Fixed(0);
#ifdef OA_COMPILER_MSVC
		return Fixed((raw * kScale) / inOther.raw);
#else
		oa::I128 scaled = static_cast<oa::I128>(raw) * static_cast<oa::I128>(kScale);
		return Fixed(static_cast<oa::I64>(scaled / inOther.raw));
#endif
	}

	OA_HOST_DEVICE constexpr Fixed& operator+=(Fixed inOther) { raw += inOther.raw; return *this; }
	OA_HOST_DEVICE constexpr Fixed& operator-=(Fixed inOther) { raw -= inOther.raw; return *this; }
	OA_HOST_DEVICE constexpr Fixed& operator*=(Fixed inOther) { *this = *this * inOther; return *this; }
	OA_HOST_DEVICE constexpr Fixed& operator/=(Fixed inOther) { *this = *this / inOther; return *this; }

	[[nodiscard]] OA_HOST_DEVICE constexpr Fixed operator*(oa::I64 inScalar) const { return Fixed(raw * inScalar); }
	[[nodiscard]] OA_HOST_DEVICE constexpr Fixed operator/(oa::I64 inScalar) const { return inScalar == 0 ? Fixed(0) : Fixed(raw / inScalar); }

	[[nodiscard]] OA_HOST_DEVICE constexpr bool operator==(Fixed inOther) const { return raw == inOther.raw; }
	[[nodiscard]] OA_HOST_DEVICE constexpr bool operator!=(Fixed inOther) const { return raw != inOther.raw; }
	[[nodiscard]] OA_HOST_DEVICE constexpr bool operator<(Fixed inOther) const { return raw < inOther.raw; }
	[[nodiscard]] OA_HOST_DEVICE constexpr bool operator<=(Fixed inOther) const { return raw <= inOther.raw; }
	[[nodiscard]] OA_HOST_DEVICE constexpr bool operator>(Fixed inOther) const { return raw > inOther.raw; }
	[[nodiscard]] OA_HOST_DEVICE constexpr bool operator>=(Fixed inOther) const { return raw >= inOther.raw; }

};

// TYPE ALIASES
using Price   = Fixed<8>;  // Price: 8 decimals
using Qty     = Fixed<8>;  // Quantity: 8 decimals
using Balance = Fixed<8>;  // Balance: 8 decimals
using Rate    = Fixed<8>;  // Rate: 8 decimals
using Percent = Fixed<6>;  // Percent: 6 decimals

inline constexpr Rate kRateBps     = Rate::fromRaw(10000);     // 1 basis point
inline constexpr Rate kRatePercent = Rate::fromRaw(1000000);   // 1 percent

inline constexpr Rate OA_RATE_BPS     = kRateBps;
inline constexpr Rate OA_RATE_PERCENT = kRatePercent;

} // namespace oa
