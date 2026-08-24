#include <oa/core/simd.h>

#include <xsimd/xsimd.hpp>

namespace {

using F32Batch = xsimd::batch<oa::F32>;
constexpr oa::I64 kF32Lanes = static_cast<oa::I64>(F32Batch::size);

} // namespace

namespace oa {

namespace FnSimd {

oa::F32 dotF32(
	const oa::F32* inA,
	const oa::F32* inB,
	oa::I64 inN
) {
	F32Batch sum(0.0F);
	oa::I64 index = 0;
	for (; index + kF32Lanes <= inN; index += kF32Lanes) {
		const F32Batch a = F32Batch::load_unaligned(inA + index);
		const F32Batch b = F32Batch::load_unaligned(inB + index);
		sum = xsimd::fma(a, b, sum);
	}
	oa::F32 result = xsimd::reduce_add(sum);
	for (; index < inN; ++index) result += inA[index] * inB[index];
	return result;
}

void scaleF32(
	oa::F32* inOut,
	oa::F32 inScale,
	oa::I64 inN
) {
	const F32Batch scale(inScale);
	oa::I64 index = 0;
	for (; index + kF32Lanes <= inN; index += kF32Lanes) {
		const F32Batch value = F32Batch::load_unaligned(inOut + index);
		(value * scale).store_unaligned(inOut + index);
	}
	for (; index < inN; ++index) inOut[index] *= inScale;
}

void addF32(
	oa::F32* inOut,
	const oa::F32* inB,
	oa::I64 inN
) {
	oa::I64 index = 0;
	for (; index + kF32Lanes <= inN; index += kF32Lanes) {
		const F32Batch a = F32Batch::load_unaligned(inOut + index);
		const F32Batch b = F32Batch::load_unaligned(inB + index);
		(a + b).store_unaligned(inOut + index);
	}
	for (; index < inN; ++index) inOut[index] += inB[index];
}

void subF32(
	oa::F32* inOut,
	const oa::F32* inB,
	oa::I64 inN
) {
	oa::I64 index = 0;
	for (; index + kF32Lanes <= inN; index += kF32Lanes) {
		const F32Batch a = F32Batch::load_unaligned(inOut + index);
		const F32Batch b = F32Batch::load_unaligned(inB + index);
		(a - b).store_unaligned(inOut + index);
	}
	for (; index < inN; ++index) inOut[index] -= inB[index];
}

void mulF32(
	oa::F32* inOut,
	const oa::F32* inB,
	oa::I64 inN
) {
	oa::I64 index = 0;
	for (; index + kF32Lanes <= inN; index += kF32Lanes) {
		const F32Batch a = F32Batch::load_unaligned(inOut + index);
		const F32Batch b = F32Batch::load_unaligned(inB + index);
		(a * b).store_unaligned(inOut + index);
	}
	for (; index < inN; ++index) inOut[index] *= inB[index];
}

void divF32(
	oa::F32* inOut,
	const oa::F32* inB,
	oa::I64 inN
) {
	oa::I64 index = 0;
	for (; index + kF32Lanes <= inN; index += kF32Lanes) {
		const F32Batch a = F32Batch::load_unaligned(inOut + index);
		const F32Batch b = F32Batch::load_unaligned(inB + index);
		(a / b).store_unaligned(inOut + index);
	}
	for (; index < inN; ++index) inOut[index] /= inB[index];
}

void negF32(oa::F32* inOut, oa::I64 inN) {
	oa::I64 index = 0;
	for (; index + kF32Lanes <= inN; index += kF32Lanes) {
		const F32Batch value = F32Batch::load_unaligned(inOut + index);
		(-value).store_unaligned(inOut + index);
	}
	for (; index < inN; ++index) inOut[index] = -inOut[index];
}

} // namespace FnSimd

} // namespace oa
