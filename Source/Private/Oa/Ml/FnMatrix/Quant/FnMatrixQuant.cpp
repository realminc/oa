// OaFnMatrix — Quantization operations (INT8 and Q4)
//
// QuantizeInt8, DequantizeInt8, ComputeScaleInt8, GemmInt8
// QuantizeQ4_0, DequantizeQ4_0, ComputeScaleQ4_0

#include <Oa/Ml/FnMatrix.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/Validation.h>

#include <cassert>

static OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }

// ─── QuantizeQ4_0 (Quantize FP32/BF16 to Q4_0 (llama.cpp format)) ────

OaMatrix OaFnMatrix::QuantizeQ4_0(const OaMatrix& InInput, const OaMatrix& InScale) {
	auto& ctx = OaContext::GetDefault();
	OaI64 count = InInput.NumElements();
	OaI64 numBlocks = DivCeil(static_cast<OaU32>(count), 32);

	assert(InScale.NumElements() == numBlocks && "Scale must have one value per block");

	// Q4_0 packs 2 values per byte
	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{numBlocks * 16}, OaScalarType::UInt8);

	struct {
		OaU32 InputIdx;
		OaU32 OutputIdx;
		OaU32 ScaleIdx;
		OaU32 Count;
	} push{0, 0, 0, static_cast<OaU32>(count)};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write, OaBufferAccess::Read};
	ctx.Add("Quantize/QuantizeQ4_0", {&InInput, &out, &InScale}, access, &push, sizeof(push), DivCeil(static_cast<OaU32>(numBlocks * 32), 256));

	return out;
}

// ─── DequantizeQ4_0 (Dequantize Q4_0 to FP32/BF16 (llama.cpp format)) ────

OaMatrix OaFnMatrix::DequantizeQ4_0(const OaMatrix& InInput, const OaMatrix& InScale, OaI64 InCount) {
	auto& ctx = OaContext::GetDefault();
	OaI64 numBlocks = DivCeil(static_cast<OaU32>(InCount), 32);

	assert(InScale.NumElements() == numBlocks && "Scale must have one value per block");
	assert(InInput.NumElements() == numBlocks * 16 && "Input must be packed Q4_0");

	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{InCount}, OaScalarType::Float32);

	struct {
		OaU32 InputIdx;
		OaU32 OutputIdx;
		OaU32 ScaleIdx;
		OaU32 Count;
	} push{0, 0, 0, static_cast<OaU32>(InCount)};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write, OaBufferAccess::Read};
	ctx.Add("Quantize/DequantizeQ4_0", {&InInput, &out, &InScale}, access, &push, sizeof(push), DivCeil(static_cast<OaU32>(numBlocks * 32), 256));

	return out;
}

// ─── ComputeScaleQ4_0 (Compute per-block scale for Q4_0 quantization) ────

OaMatrix OaFnMatrix::ComputeScaleQ4_0(const OaMatrix& InInput) {
	auto& ctx = OaContext::GetDefault();
	OaI64 count = InInput.NumElements();
	OaI64 numBlocks = DivCeil(static_cast<OaU32>(count), 32);

	OaMatrix scale = OaFnMatrix::Empty(OaMatrixShape{numBlocks}, OaScalarType::Float32);

	struct {
		OaU32 InputIdx;
		OaU32 ScaleIdx;
		OaU32 Count;
	} push{0, 0, static_cast<OaU32>(count)};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Quantize/ComputeScaleQ4_0", {&InInput, &scale}, access, &push, sizeof(push), DivCeil(static_cast<OaU32>(numBlocks * 32), 256));

	return scale;
}
