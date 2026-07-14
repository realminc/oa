// OA SIMD — Portable vector math via Google Highway
//
// Highway wraps runtime CPU detection for math/vector ops.
// Memory ops (memcpy, memset, memzero, memequal) are in memory.h.
//
// Usage:
//   #include <Oa/Core/Simd.h>
//   OaSimd::DotF32(a, b, n);
//   OaSimd::ScaleF32(data, scale, n);
//   OaSimd::AddF32(data, other, n);

#pragma once

#include <Oa/Core/Types.h>


class OaSimd {
public:
	// Oa vector math wrapper, currently via Google Highway.

  // Methods.
	[[nodiscard]] static OaF32 DotF32(const OaF32* InA, const OaF32* InB, OaI64 InN);
	static void ScaleF32(OaF32* InOut, OaF32 InScale, OaI64 InN);
	static void AddF32(OaF32* InOut, const OaF32* InB, OaI64 InN);
	static void SubF32(OaF32* InOut, const OaF32* InB, OaI64 InN);
	static void MulF32(OaF32* InOut, const OaF32* InB, OaI64 InN);
	static void DivF32(OaF32* InOut, const OaF32* InB, OaI64 InN);
	static void NegF32(OaF32* InOut, OaI64 InN);
};
