#include <hwy/highway.h>

#include <Oa/Core/Simd.h>

namespace hn = hwy::HWY_NAMESPACE;

OaF32 OaSimd::DotF32(const OaF32* InA, const OaF32* InB, OaI64 InN) {
	const hn::ScalableTag<OaF32> d;
	const OaI64 N = static_cast<OaI64>(hn::Lanes(d));
	auto sum = hn::Zero(d);

	OaI64 i = 0;
	for (; i + N <= InN; i += N) {
		auto va = hn::LoadU(d, InA + i);
		auto vb = hn::LoadU(d, InB + i);
		sum = hn::MulAdd(va, vb, sum);
	}

	OaF32 result = hn::ReduceSum(d, sum);
	for (; i < InN; ++i) {
		result += InA[i] * InB[i];
	}
	return result;
}

void OaSimd::ScaleF32(OaF32* InOut, OaF32 InScale, OaI64 InN) {
	const hn::ScalableTag<OaF32> d;
	const OaI64 N = static_cast<OaI64>(hn::Lanes(d));
	const auto scale = hn::Set(d, InScale);

	OaI64 i = 0;
	for (; i + N <= InN; i += N) {
		auto v = hn::LoadU(d, InOut + i);
		hn::StoreU(hn::Mul(v, scale), d, InOut + i);
	}
	for (; i < InN; ++i) {
		InOut[i] *= InScale;
	}
}

void OaSimd::AddF32(OaF32* InOut, const OaF32* InB, OaI64 InN) {
	const hn::ScalableTag<OaF32> d;
	const OaI64 N = static_cast<OaI64>(hn::Lanes(d));

	OaI64 i = 0;
	for (; i + N <= InN; i += N) {
		auto va = hn::LoadU(d, InOut + i);
		auto vb = hn::LoadU(d, InB + i);
		hn::StoreU(hn::Add(va, vb), d, InOut + i);
	}
	for (; i < InN; ++i) {
		InOut[i] += InB[i];
	}
}

void OaSimd::SubF32(OaF32* InOut, const OaF32* InB, OaI64 InN) {
	const hn::ScalableTag<OaF32> d;
	const OaI64 N = static_cast<OaI64>(hn::Lanes(d));

	OaI64 i = 0;
	for (; i + N <= InN; i += N) {
		auto va = hn::LoadU(d, InOut + i);
		auto vb = hn::LoadU(d, InB + i);
		hn::StoreU(hn::Sub(va, vb), d, InOut + i);
	}
	for (; i < InN; ++i) {
		InOut[i] -= InB[i];
	}
}

void OaSimd::MulF32(OaF32* InOut, const OaF32* InB, OaI64 InN) {
	const hn::ScalableTag<OaF32> d;
	const OaI64 N = static_cast<OaI64>(hn::Lanes(d));

	OaI64 i = 0;
	for (; i + N <= InN; i += N) {
		auto va = hn::LoadU(d, InOut + i);
		auto vb = hn::LoadU(d, InB + i);
		hn::StoreU(hn::Mul(va, vb), d, InOut + i);
	}
	for (; i < InN; ++i) {
		InOut[i] *= InB[i];
	}
}

void OaSimd::DivF32(OaF32* InOut, const OaF32* InB, OaI64 InN) {
	const hn::ScalableTag<OaF32> d;
	const OaI64 N = static_cast<OaI64>(hn::Lanes(d));

	OaI64 i = 0;
	for (; i + N <= InN; i += N) {
		auto va = hn::LoadU(d, InOut + i);
		auto vb = hn::LoadU(d, InB + i);
		hn::StoreU(hn::Div(va, vb), d, InOut + i);
	}
	for (; i < InN; ++i) {
		InOut[i] /= InB[i];
	}
}

void OaSimd::NegF32(OaF32* InOut, OaI64 InN) {
	const hn::ScalableTag<OaF32> d;
	const OaI64 N = static_cast<OaI64>(hn::Lanes(d));
	const auto zero = hn::Zero(d);

	OaI64 i = 0;
	for (; i + N <= InN; i += N) {
		auto v = hn::LoadU(d, InOut + i);
		hn::StoreU(hn::Sub(zero, v), d, InOut + i);
	}
	for (; i < InN; ++i) {
		InOut[i] = -InOut[i];
	}
}
