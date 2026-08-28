// OA SIMD — Portable compiled batch math via xsimd
//
// Memory operations are owned by <oa/core/std/memory.h>.
//
// usage:
//   #include <oa/core/simd.h>
//   oa::FnSimd::dotF32(a, b, n);
//   oa::FnSimd::scaleF32(data, scale, n);
//   oa::FnSimd::addF32(data, other, n);

#pragma once

#include <oa/core/types.h>

namespace oa {

namespace FnSimd {

	// OA's backend-neutral array math. xsimd is private to the compiled source;
	// callers never inherit its types, alignment, or headers.
	[[nodiscard]] oa::F32 dotF32(const oa::F32* inA, const oa::F32* inB, oa::I64 inN);
	void scaleF32(oa::F32* inOut, oa::F32 inScale, oa::I64 inN);
	void addF32(oa::F32* inOut, const oa::F32* inB, oa::I64 inN);
	void subF32(oa::F32* inOut, const oa::F32* inB, oa::I64 inN);
	void mulF32(oa::F32* inOut, const oa::F32* inB, oa::I64 inN);
	void divF32(oa::F32* inOut, const oa::F32* inB, oa::I64 inN);
	void negF32(oa::F32* inOut, oa::I64 inN);

} // namespace FnSimd

} // namespace oa
