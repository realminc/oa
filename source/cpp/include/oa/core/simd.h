// OA SIMD — Portable vector math via Google Highway
//
// Highway wraps runtime CPU detection for math/vector ops.
// memory ops (memcpy, memset, memzero, memequal) are in memory.h.
//
// usage:
//   #include <oa/core/simd.h>
//   Simd::dotF32(a, b, n);
//   Simd::scaleF32(data, scale, n);
//   Simd::addF32(data, other, n);

#pragma once

#include <oa/core/types.h>

namespace oa {

class Simd {
public:
	// Oa vector math wrapper, currently via Google Highway.

  // Methods.
	[[nodiscard]] static oa::F32 dotF32(const oa::F32* inA, const oa::F32* inB, oa::I64 inN);
	static void scaleF32(oa::F32* inOut, oa::F32 inScale, oa::I64 inN);
	static void addF32(oa::F32* inOut, const oa::F32* inB, oa::I64 inN);
	static void subF32(oa::F32* inOut, const oa::F32* inB, oa::I64 inN);
	static void mulF32(oa::F32* inOut, const oa::F32* inB, oa::I64 inN);
	static void divF32(oa::F32* inOut, const oa::F32* inB, oa::I64 inN);
	static void negF32(oa::F32* inOut, oa::I64 inN);
};

} // namespace oa
