// OA CORE - Fixed-Point Arithmetic
//
// Byte-stable deterministic fixed-point math for financial calculations.
// No floating-point rounding errors. Cross-platform identical results.

#pragma once

#include <Oa/Core/Types.h>

template<OaI32 Decimals>
class OaFixed {
public:
	// FIXED-POINT CLASS

	// Data, class members.
	OaI64 Raw;

	// Constructors.
	OA_HOST_DEVICE constexpr OaFixed()
		: Raw(0)
	{}
	OA_HOST_DEVICE constexpr explicit OaFixed(OaI64 InRaw)
		: Raw(InRaw)
	{}
	
	// Methods.
	static constexpr OaI64 SCALE = []() constexpr {
		OaI64 S = 1;
		for (OaI32 i = 0; i < Decimals; ++i) S *= 10;
		return S;
	}();
	[[nodiscard]] OA_HOST_DEVICE static constexpr OaFixed FromRaw(OaI64 InRaw) { return OaFixed(InRaw); }
	[[nodiscard]] OA_HOST_DEVICE static constexpr OaFixed FromInt(OaI64 InValue) { return OaFixed(InValue * SCALE); }
	[[nodiscard]] OA_HOST_DEVICE static constexpr OaFixed FromDouble(OaF64 InValue) {
		return OaFixed(static_cast<OaI64>(InValue * static_cast<OaF64>(SCALE) + (InValue >= 0 ? 0.5 : -0.5)));
	}
	[[nodiscard]] OA_HOST_DEVICE static constexpr OaFixed FromFloat(OaF32 InValue) { return FromDouble(static_cast<OaF64>(InValue)); }
	[[nodiscard]] OA_HOST_DEVICE static constexpr OaFixed Zero() { return OaFixed(0); }
	[[nodiscard]] OA_HOST_DEVICE static constexpr OaFixed One() { return OaFixed(SCALE); }
	[[nodiscard]] OA_HOST_DEVICE static constexpr OaFixed Max() { return OaFixed(OA_I64_MAX); }
	[[nodiscard]] OA_HOST_DEVICE static constexpr OaFixed Min() { return OaFixed(OA_I64_MIN); }
	
	[[nodiscard]] OA_HOST_DEVICE constexpr OaF64 ToDouble() const { return static_cast<OaF64>(Raw) / static_cast<OaF64>(SCALE); }
	[[nodiscard]] OA_HOST_DEVICE constexpr OaF32 ToFloat() const { return static_cast<OaF32>(ToDouble()); }
	[[nodiscard]] OA_HOST_DEVICE constexpr OaI64 ToInt() const { return Raw / SCALE; }
	[[nodiscard]] OA_HOST_DEVICE constexpr OaI64 GetRaw() const { return Raw; }

	[[nodiscard]] OA_HOST_DEVICE constexpr bool IsZero() const { return Raw == 0; }
	[[nodiscard]] OA_HOST_DEVICE constexpr bool IsPositive() const { return Raw > 0; }
	[[nodiscard]] OA_HOST_DEVICE constexpr bool IsNegative() const { return Raw < 0; }

	[[nodiscard]] OA_HOST_DEVICE constexpr OaFixed ApplyPercent(OaFixed InPercent) const { return *this * InPercent; }
	[[nodiscard]] OA_HOST_DEVICE static constexpr OaFixed FromPercent(OaI64 InPercent) { return OaFixed::FromRaw(InPercent * (SCALE / 100)); }
	[[nodiscard]] OA_HOST_DEVICE static constexpr OaFixed FromBps(OaI64 InBps) { return OaFixed::FromRaw(InBps * (SCALE / 10000)); }

	[[nodiscard]] OA_HOST_DEVICE constexpr OaFixed Abs() const { return OaFixed(Raw >= 0 ? Raw : -Raw); }
	[[nodiscard]] OA_HOST_DEVICE constexpr OaFixed Min(OaFixed InOther) const { return OaFixed(Raw < InOther.Raw ? Raw : InOther.Raw); }
	[[nodiscard]] OA_HOST_DEVICE constexpr OaFixed Max(OaFixed InOther) const { return OaFixed(Raw > InOther.Raw ? Raw : InOther.Raw); }
	[[nodiscard]] OA_HOST_DEVICE constexpr OaFixed Clamp(OaFixed InLo, OaFixed InHi) const {
		if (Raw < InLo.Raw) {
			return InLo;
		}
		if (Raw > InHi.Raw) {
			return InHi;
		}
		return *this;
	}

	[[nodiscard]] OaString ToString() const {
		OaString Result;
		OaI64 AbsRaw = Raw >= 0 ? Raw : -Raw;
		OaI64 IntPart = AbsRaw / SCALE;
		OaI64 FracPart = AbsRaw % SCALE;
		if (Raw < 0) {
			Result += '-';
		}
		Result += OaToString(static_cast<OaI64>(IntPart));
		Result += '.';
		OaString FracStr(OaToString(static_cast<OaI64>(FracPart)));
		for (OaI32 i = 0; i < Decimals - static_cast<OaI32>(FracStr.Size()); ++i) {
			Result += '0';
		}
		Result += FracStr;
		return Result;
	}

	// Operators.
	[[nodiscard]] OA_HOST_DEVICE constexpr OaFixed operator+(OaFixed InOther) const { return OaFixed(Raw + InOther.Raw); }
	[[nodiscard]] OA_HOST_DEVICE constexpr OaFixed operator-(OaFixed InOther) const { return OaFixed(Raw - InOther.Raw); }
	[[nodiscard]] OA_HOST_DEVICE constexpr OaFixed operator-() const { return OaFixed(-Raw); }

	[[nodiscard]] OA_HOST_DEVICE constexpr OaFixed operator*(OaFixed InOther) const {
#ifdef OA_COMPILER_MSVC
		return OaFixed((Raw * InOther.Raw) / SCALE);
#else
		OaI128 Result = static_cast<OaI128>(Raw) * static_cast<OaI128>(InOther.Raw);
		return OaFixed(static_cast<OaI64>(Result / SCALE));
#endif
	}

	[[nodiscard]] OA_HOST_DEVICE constexpr OaFixed operator/(OaFixed InOther) const {
		if (InOther.Raw == 0) return OaFixed(0);
#ifdef OA_COMPILER_MSVC
		return OaFixed((Raw * SCALE) / InOther.Raw);
#else
		OaI128 Scaled = static_cast<OaI128>(Raw) * static_cast<OaI128>(SCALE);
		return OaFixed(static_cast<OaI64>(Scaled / InOther.Raw));
#endif
	}
	
	OA_HOST_DEVICE constexpr OaFixed& operator+=(OaFixed InOther) { Raw += InOther.Raw; return *this; }
	OA_HOST_DEVICE constexpr OaFixed& operator-=(OaFixed InOther) { Raw -= InOther.Raw; return *this; }
	OA_HOST_DEVICE constexpr OaFixed& operator*=(OaFixed InOther) { *this = *this * InOther; return *this; }
	OA_HOST_DEVICE constexpr OaFixed& operator/=(OaFixed InOther) { *this = *this / InOther; return *this; }
	
	[[nodiscard]] OA_HOST_DEVICE constexpr OaFixed operator*(OaI64 InScalar) const { return OaFixed(Raw * InScalar); }
	[[nodiscard]] OA_HOST_DEVICE constexpr OaFixed operator/(OaI64 InScalar) const { return InScalar == 0 ? OaFixed(0) : OaFixed(Raw / InScalar); }
	
	[[nodiscard]] OA_HOST_DEVICE constexpr bool operator==(OaFixed InOther) const { return Raw == InOther.Raw; }
	[[nodiscard]] OA_HOST_DEVICE constexpr bool operator!=(OaFixed InOther) const { return Raw != InOther.Raw; }
	[[nodiscard]] OA_HOST_DEVICE constexpr bool operator<(OaFixed InOther) const { return Raw < InOther.Raw; }
	[[nodiscard]] OA_HOST_DEVICE constexpr bool operator<=(OaFixed InOther) const { return Raw <= InOther.Raw; }
	[[nodiscard]] OA_HOST_DEVICE constexpr bool operator>(OaFixed InOther) const { return Raw > InOther.Raw; }
	[[nodiscard]] OA_HOST_DEVICE constexpr bool operator>=(OaFixed InOther) const { return Raw >= InOther.Raw; }
	
};

// TYPE ALIASES
using OaPrice   = OaFixed<8>;  // Price: 8 decimals
using OaQty     = OaFixed<8>;  // Quantity: 8 decimals
using OaBalance = OaFixed<8>;  // Balance: 8 decimals
using OaRate    = OaFixed<8>;  // Rate: 8 decimals
using OaPercent = OaFixed<6>;  // Percent: 6 decimals

inline constexpr OaRate OA_RATE_BPS     = OaRate::FromRaw(10000);     // 1 basis point
inline constexpr OaRate OA_RATE_PERCENT = OaRate::FromRaw(1000000);   // 1 percent
