// OaFnMatrix — RNG operations (GPU-native Philox PRNG).
//
// Philox4x32-10 counter-based PRNG for parallel, reproducible random number generation.
// - PhiloxUniform: Uniform distribution [low, high)
// - PhiloxNormal: Normal distribution N(mean, stddev)
//
// These are the low-level GPU RNG primitives. High-level initialization functions
// (Rand, RandN, RandXavier, etc.) are in FnMatrixAlloc.cpp and use these primitives.

#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>

#include <algorithm>
#include <random>

static OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }

// Shared host-side seed source for all Philox calls made with InSeed == 0.
// Seeded from std::random_device by default (per-run non-deterministic);
// OaFnMatrix::SetRngSeed() reseeds it for reproducible init/training.
static std::mt19937& OaRngState() {
	static thread_local std::mt19937 rng(std::random_device{}());
	return rng;
}

void OaFnMatrix::SetRngSeed(OaU64 InSeed) {
	OaRngState().seed(static_cast<std::mt19937::result_type>(InSeed));
}

// PhiloxUniform: Generate uniform random floats in [low, high)
OaMatrix OaFnMatrix::PhiloxUniform(const OaMatrix& InA, OaF32 InLow, OaF32 InHigh, OaU64 InSeed) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InA.NumElements());
	OaMatrix out = OaFnMatrix::Empty(InA.Shape_, InA.Dtype_);
	
	// Generate seed if not provided
	OaU32 seed = static_cast<OaU32>(InSeed);
	if (InSeed == 0) {
		seed = OaRngState()();
	}

	// Push constants: out_idx, count, seed, offset
	struct { OaU32 Count; OaU32 Seed; OaU32 Offset; } push{.Count = n, .Seed = seed, .Offset = 0};
	OaBufferAccess access[] = {OaBufferAccess::Write};
	
	// Dispatch: each thread generates 4 floats, so workgroups = (count / 1024)
	OaU32 workgroups = DivCeil(n, 1024);
	ctx.Add("PhiloxUniform", {&out}, access, &push, sizeof(push), workgroups);
	
	// Scale from [0, 1) to [low, high): out = out * (high - low) + low
	if (InLow != 0.0F || InHigh != 1.0F) {
		OaF32 scale = InHigh - InLow;
		out = OaFnMatrix::Scale(out, scale);
		if (InLow != 0.0F) {
			auto offset_tensor = OaFnMatrix::Full(out.Shape_, InLow, out.Dtype_);
			out = OaFnMatrix::Add(out, offset_tensor);
		}
	}
	
	return out;
}

// PhiloxNormal: Generate normal random floats N(mean, stddev)
OaMatrix OaFnMatrix::PhiloxNormal(const OaMatrix& InA, OaF32 InMean, OaF32 InStddev, OaU64 InSeed) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InA.NumElements());
	OaMatrix out = OaFnMatrix::Empty(InA.Shape_, InA.Dtype_);
	
	// Generate seed if not provided
	OaU32 seed = static_cast<OaU32>(InSeed);
	if (InSeed == 0) {
		seed = OaRngState()();
	}

	// Push constants: out_idx, count, seed, offset, mean, stddev
	struct { OaU32 Count; OaU32 Seed; OaU32 Offset; OaF32 Mean; OaF32 Stddev; } push{
		.Count = n, .Seed = seed, .Offset = 0, .Mean = InMean, .Stddev = InStddev
	};
	OaBufferAccess access[] = {OaBufferAccess::Write};
	
	// Dispatch: each thread generates 2 floats, so workgroups = (count / 512)
	OaU32 workgroups = DivCeil(n, 512);
	ctx.Add("PhiloxNormal", {&out}, access, &push, sizeof(push), workgroups);
	
	return out;
}

OaMatrix OaFnMatrix::Dropout(const OaMatrix& InA, OaF32 InP, OaU64 InSeed) {
	if (InP < 0.0F or InP >= 1.0F) {
		OA_LOG_ERROR(OaLogComponent::ML, "Dropout: probability must be in [0,1), got %g", InP);
		return {};
	}
	if (InP == 0.0F) return InA;
	auto uniform = PhiloxUniform(InA, 0.0F, 1.0F, InSeed);
	auto keep = GreaterEqual(uniform, InP);
	auto scaledKeep = Scale(keep, 1.0F / (1.0F - InP));
	return Mul(InA, scaledKeep);
}

OaMatrix OaFnMatrix::SampleLogits(const OaMatrix& InLogits, OaF32 InTemperature,
	OaI32 InTopK, OaF32 InTopP, OaU64 InSeed) {
	if ((InLogits.Rank() != 1 and InLogits.Rank() != 2) or
		(InLogits.GetDtype() != OaScalarType::Float32 and
		 InLogits.GetDtype() != OaScalarType::BFloat16)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"SampleLogits: expected Float32/BFloat16 [V] or [R,V]");
		return {};
	}
	const OaI32 vocab = static_cast<OaI32>(InLogits.Size(InLogits.Rank() - 1));
	const OaI32 rows = InLogits.Rank() == 2 ? static_cast<OaI32>(InLogits.Size(0)) : 1;
	if (vocab <= 0) {
		OA_LOG_ERROR(OaLogComponent::ML, "SampleLogits: vocabulary must be positive");
		return {};
	}
	const OaI32 candidates = InTopK > 0 ? std::min(InTopK, vocab) : vocab;
	if (InTemperature <= 0.0F) {
		auto top = TopK(InLogits, 1);
		return top.Indices.IsEmpty() ? OaMatrix{} : Reshape(top.Indices, OaMatrixShape{rows});
	}

	auto randomShape = OaMatrixShape{rows};
	auto randomBase = Empty(randomShape, OaScalarType::Float32);
	auto random = PhiloxUniform(randomBase, 0.0F, 1.0F, InSeed);
	auto out = Empty(randomShape, OaScalarType::Int32);
	const OaF32 topP = std::max(1.0e-7F, std::min(InTopP, 1.0F));
	if (InTopK <= 0 and topP >= 1.0F) {
		struct { OaU32 Rows, Vocab; OaF32 Temperature; } push{
			static_cast<OaU32>(rows), static_cast<OaU32>(vocab), InTemperature};
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
		auto& ctx = OaContext::GetDefault();
		ctx.Add("SampleDenseLogits", {&InLogits, &random, &out}, access, &push, sizeof(push),
			DivCeil(static_cast<OaU32>(rows), 256));
		return out;
	}
	if (candidates > 1024) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"SampleLogits: top-k/top-p candidate count %d exceeds GPU limit 1024", candidates);
		return {};
	}
	auto top = TopK(InLogits, candidates);
	if (top.Indices.IsEmpty()) return {};
	struct { OaU32 Rows, K; OaF32 Temperature, TopP; } push{
		static_cast<OaU32>(rows), static_cast<OaU32>(candidates),
		InTemperature, topP};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("SampleSortedLogits", {&top.Values, &top.Indices, &random, &out},
		access, &push, sizeof(push), DivCeil(static_cast<OaU32>(rows), 256));
	return out;
}
