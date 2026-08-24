#include <hwy/highway.h>

#include <oa/core/simd.h>

namespace hn = hwy::HWY_NAMESPACE;

oa::F32 oa::Simd::dotF32(const oa::F32* inA, const oa::F32* inB, oa::I64 inN) {
	const hn::ScalableTag<oa::F32> d;
	const oa::I64 N = static_cast<oa::I64>(hn::Lanes(d));
	auto sum = hn::Zero(d);

	oa::I64 i = 0;
	for (; i + N <= inN; i += N) {
		auto va = hn::LoadU(d, inA + i);
		auto vb = hn::LoadU(d, inB + i);
		sum = hn::MulAdd(va, vb, sum);
	}

	oa::F32 result = hn::ReduceSum(d, sum);
	for (; i < inN; ++i) {
		result += inA[i] * inB[i];
	}
	return result;
}

void oa::Simd::scaleF32(oa::F32* inOut, oa::F32 inScale, oa::I64 inN) {
	const hn::ScalableTag<oa::F32> d;
	const oa::I64 N = static_cast<oa::I64>(hn::Lanes(d));
	const auto scale = hn::Set(d, inScale);

	oa::I64 i = 0;
	for (; i + N <= inN; i += N) {
		auto v = hn::LoadU(d, inOut + i);
		hn::StoreU(hn::Mul(v, scale), d, inOut + i);
	}
	for (; i < inN; ++i) {
		inOut[i] *= inScale;
	}
}

void oa::Simd::addF32(oa::F32* inOut, const oa::F32* inB, oa::I64 inN) {
	const hn::ScalableTag<oa::F32> d;
	const oa::I64 N = static_cast<oa::I64>(hn::Lanes(d));

	oa::I64 i = 0;
	for (; i + N <= inN; i += N) {
		auto va = hn::LoadU(d, inOut + i);
		auto vb = hn::LoadU(d, inB + i);
		hn::StoreU(hn::Add(va, vb), d, inOut + i);
	}
	for (; i < inN; ++i) {
		inOut[i] += inB[i];
	}
}

void oa::Simd::subF32(oa::F32* inOut, const oa::F32* inB, oa::I64 inN) {
	const hn::ScalableTag<oa::F32> d;
	const oa::I64 N = static_cast<oa::I64>(hn::Lanes(d));

	oa::I64 i = 0;
	for (; i + N <= inN; i += N) {
		auto va = hn::LoadU(d, inOut + i);
		auto vb = hn::LoadU(d, inB + i);
		hn::StoreU(hn::Sub(va, vb), d, inOut + i);
	}
	for (; i < inN; ++i) {
		inOut[i] -= inB[i];
	}
}

void oa::Simd::mulF32(oa::F32* inOut, const oa::F32* inB, oa::I64 inN) {
	const hn::ScalableTag<oa::F32> d;
	const oa::I64 N = static_cast<oa::I64>(hn::Lanes(d));

	oa::I64 i = 0;
	for (; i + N <= inN; i += N) {
		auto va = hn::LoadU(d, inOut + i);
		auto vb = hn::LoadU(d, inB + i);
		hn::StoreU(hn::Mul(va, vb), d, inOut + i);
	}
	for (; i < inN; ++i) {
		inOut[i] *= inB[i];
	}
}

void oa::Simd::divF32(oa::F32* inOut, const oa::F32* inB, oa::I64 inN) {
	const hn::ScalableTag<oa::F32> d;
	const oa::I64 N = static_cast<oa::I64>(hn::Lanes(d));

	oa::I64 i = 0;
	for (; i + N <= inN; i += N) {
		auto va = hn::LoadU(d, inOut + i);
		auto vb = hn::LoadU(d, inB + i);
		hn::StoreU(hn::Div(va, vb), d, inOut + i);
	}
	for (; i < inN; ++i) {
		inOut[i] /= inB[i];
	}
}

void oa::Simd::negF32(oa::F32* inOut, oa::I64 inN) {
	const hn::ScalableTag<oa::F32> d;
	const oa::I64 N = static_cast<oa::I64>(hn::Lanes(d));
	const auto zero = hn::Zero(d);

	oa::I64 i = 0;
	for (; i + N <= inN; i += N) {
		auto v = hn::LoadU(d, inOut + i);
		hn::StoreU(hn::Sub(zero, v), d, inOut + i);
	}
	for (; i < inN; ++i) {
		inOut[i] = -inOut[i];
	}
}
