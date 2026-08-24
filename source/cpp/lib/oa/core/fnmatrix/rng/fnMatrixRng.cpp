// oa::FnMatrix — RNG operations (GPU-native Philox PRNG).
//
// Philox4x32-10 counter-based PRNG for parallel, reproducible random number generation.
// - PhiloxUniform: Uniform distribution [low, high)
// - PhiloxNormal: Normal distribution N(mean, stddev)
//
// These are the low-level GPU RNG primitives. high-level initialization functions
// (Rand, RandN, RandXavier, etc.) are in FnMatrixAlloc.cpp and use these primitives.

#include <oa/core/autograd.h>
#include <oa/core/log.h>
#include <oa/core/matrix.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/op.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/core/bufferAccess.h>
#include <oa/runtime/executionSession.h>

#include <algorithm>
#include <random>

static oa::U32 divCeil(oa::U32 inA, oa::U32 inB) { return (inA + inB - 1) / inB; }

// Shared host-side seed source for all Philox calls made with inSeed == 0.
// Seeded from std::random_device by default (per-run non-deterministic);
// oa::FnMatrix::setRngSeed() reseeds it for reproducible init/training.
static std::mt19937_64& rngState() {
	static thread_local std::mt19937_64 rng(std::random_device{}());
	return rng;
}

static oa::U64 resolveRngSeed(oa::U64 inSeed) {
	return inSeed == 0 ? rngState()() : inSeed;
}

void oa::FnMatrix::setRngSeed(oa::U64 inSeed) {
	rngState().seed(inSeed);
}

// PhiloxUniform: generate uniform random floats in [low, high)
oa::Matrix oa::FnMatrix::philoxUniform(const oa::Matrix& inA, oa::F32 inLow, oa::F32 inHigh, oa::U64 inSeed) {
	if (not (inLow < inHigh)) {
		OaLogError(oa::LogComponent::Compute,
			"PhiloxUniform: low must be smaller than high");
		return {};
	}
	auto& ctx = oa::ExecutionSession::getActive();
	oa::U32 n = static_cast<oa::U32>(inA.numElements());
	oa::Matrix out = oa::FnMatrix::empty(inA.getShape(), inA.getDtype());
	
	const oa::U64 seed = resolveRngSeed(inSeed);
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::philoxUniform, {&inA}, {&out},
		{
			oa::OpAttribute::fromFloat("low", inLow),
			oa::OpAttribute::fromFloat("high", inHigh),
			oa::OpAttribute::fromUnsignedInteger("seed", seed),
		});
	if (not semantic.isOk()) return {};

	struct Push {
		oa::U32 count;
		oa::U32 seedLow;
		oa::U32 seedHigh;
		oa::U32 offset;
		oa::F32 low;
		oa::F32 high;
	} push{
		.count = n,
		.seedLow = static_cast<oa::U32>(seed),
		.seedHigh = static_cast<oa::U32>(seed >> 32U),
		.offset = 0,
		.low = inLow,
		.high = inHigh,
	};
	oa::BufferAccess access[] = {oa::BufferAccess::Write};
	
	// Dispatch: each thread generates 4 floats, so workgroups = (count / 1024)
	oa::U32 workgroups = divCeil(n, 1024);
	ctx.add( "PhiloxUniform", {&out}, access, &push, sizeof(push), workgroups,
		1, 1, oa::detail::opRegistry::FnMatrix::philoxUniform.name, 0,
		oa::detail::opRegistry::FnMatrix::philoxUniform.hash, 0, 0,
		semantic.getValue());
	
	return out;
}

// PhiloxNormal: generate normal random floats N(mean, stddev)
oa::Matrix oa::FnMatrix::philoxNormal(const oa::Matrix& inA, oa::F32 inMean, oa::F32 inStddev, oa::U64 inSeed) {
	if (inStddev < 0.0F) {
		OaLogError(oa::LogComponent::Compute,
			"PhiloxNormal: stddev must be non-negative");
		return {};
	}
	auto& ctx = oa::ExecutionSession::getActive();
	oa::U32 n = static_cast<oa::U32>(inA.numElements());
	oa::Matrix out = oa::FnMatrix::empty(inA.getShape(), inA.getDtype());
	
	const oa::U64 seed = resolveRngSeed(inSeed);
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::philoxNormal, {&inA}, {&out},
		{
			oa::OpAttribute::fromFloat("mean", inMean),
			oa::OpAttribute::fromFloat("stddev", inStddev),
			oa::OpAttribute::fromUnsignedInteger("seed", seed),
		});
	if (not semantic.isOk()) return {};

	struct Push {
		oa::U32 count;
		oa::U32 seedLow;
		oa::U32 seedHigh;
		oa::U32 offset;
		oa::F32 mean;
		oa::F32 stddev;
	} push{
		.count = n,
		.seedLow = static_cast<oa::U32>(seed),
		.seedHigh = static_cast<oa::U32>(seed >> 32U),
		.offset = 0,
		.mean = inMean,
		.stddev = inStddev,
	};
	oa::BufferAccess access[] = {oa::BufferAccess::Write};
	
	// Dispatch: each thread generates 2 floats, so workgroups = (count / 512)
	oa::U32 workgroups = divCeil(n, 512);
	ctx.add( "PhiloxNormal", {&out}, access, &push, sizeof(push), workgroups,
		1, 1, oa::detail::opRegistry::FnMatrix::philoxNormal.name, 0,
		oa::detail::opRegistry::FnMatrix::philoxNormal.hash, 0, 0,
		semantic.getValue());
	
	return out;
}

oa::Matrix oa::FnMatrix::dropout(const oa::Matrix& inA, oa::F32 inP, oa::U64 inSeed) {
	if (inP < 0.0F or inP >= 1.0F) {
		OaLogError(oa::LogComponent::Compute, "Dropout: probability must be in [0,1), got %g", inP);
		return {};
	}
	if (inP == 0.0F) return inA;
	const oa::U64 seed = resolveRngSeed(inSeed);
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	auto uniform = philoxUniform(inA, 0.0F, 1.0F, seed);
	auto keep = greaterEqual(uniform, inP);
	auto scaledKeep = scale(keep, 1.0F / (1.0F - inP));
	auto out = mul(inA, scaledKeep);
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::dropout,
		{&inA}, {&out},
		{
			oa::OpAttribute::fromFloat("probability", inP),
			oa::OpAttribute::fromUnsignedInteger("seed", seed),
		});
	if (not semantic.isOk()) return {};
	if (auto grad = out.getGradFn()) {
		if (not oa::FnAutograd::attachSemantic(
			grad, semantic.getValue()).isOk())
		{
			return {};
		}
	}
	return out;
}

oa::Matrix oa::FnMatrix::sampleLogits(const oa::Matrix& inLogits, oa::F32 inTemperature,
	oa::I32 inTopK, oa::F32 inTopP, oa::U64 inSeed) {
	if ((inLogits.rank() != 1 and inLogits.rank() != 2) or
		(inLogits.getDtype() != oa::ScalarType::Float32 and
		 inLogits.getDtype() != oa::ScalarType::BFloat16)) {
		OaLogError(oa::LogComponent::Compute,
			"SampleLogits: expected Float32/BFloat16 [V] or [R,V]");
		return {};
	}
	const oa::I32 vocab = static_cast<oa::I32>(inLogits.size(inLogits.rank() - 1));
	const oa::I32 rows = inLogits.rank() == 2 ? static_cast<oa::I32>(inLogits.size(0)) : 1;
	if (vocab <= 0) {
		OaLogError(oa::LogComponent::Compute, "SampleLogits: vocabulary must be positive");
		return {};
	}
	const oa::I32 candidates = inTopK > 0 ? std::min(inTopK, vocab) : vocab;
	const oa::F32 topP = std::max(1.0e-7F, std::min(inTopP, 1.0F));
	const oa::U64 seed = inTemperature > 0.0F
		? resolveRngSeed(inSeed)
		: inSeed;
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const auto commitResult = [&](oa::Matrix inResult) -> oa::Matrix {
		const auto status = lowering.commit(
			oa::detail::opRegistry::FnMatrix::sampleLogits,
			{&inLogits}, {&inResult},
			{
				oa::OpAttribute::fromFloat(
					"temperature", inTemperature),
				oa::OpAttribute::fromSignedInteger(
					"topK", inTopK),
				oa::OpAttribute::fromFloat("topP", topP),
				oa::OpAttribute::fromUnsignedInteger(
					"seed", seed),
			});
		return status.isOk() ? inResult : oa::Matrix{};
	};
	if (inTemperature <= 0.0F) {
		auto top = oa::FnMatrix::topK(inLogits, 1);
		if (top.indices.isEmpty()) return {};
		return commitResult(top.indices.reshape(oa::MatrixShape{rows}));
	}

	auto randomShape = oa::MatrixShape{rows};
	auto randomBase = oa::FnMatrix::empty(randomShape, oa::ScalarType::Float32);
	auto random = oa::FnMatrix::philoxUniform(randomBase, 0.0F, 1.0F, seed);
	auto out = oa::FnMatrix::empty(randomShape, oa::ScalarType::Int32);
	if (inTopK <= 0 and topP >= 1.0F) {
		struct { oa::U32 rows, Vocab; oa::F32 temperature; } push{
			static_cast<oa::U32>(rows), static_cast<oa::U32>(vocab), inTemperature};
		oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
		ctx.add( "SampleDenseLogits", {&inLogits, &random, &out}, access, &push, sizeof(push),
			divCeil(static_cast<oa::U32>(rows), 256));
		return commitResult(out);
	}
	if (candidates > 1024) {
		OaLogError(oa::LogComponent::Compute,
			"SampleLogits: top-k/top-p candidate count %d exceeds GPU limit 1024", candidates);
		return {};
	}
	auto top = oa::FnMatrix::topK(inLogits, candidates);
	if (top.indices.isEmpty()) return {};
	struct { oa::U32 rows, K; oa::F32 temperature, topP; } push{
		static_cast<oa::U32>(rows), static_cast<oa::U32>(candidates),
		inTemperature, topP};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "SampleSortedLogits", {&top.values, &top.indices, &random, &out},
		access, &push, sizeof(push), divCeil(static_cast<oa::U32>(rows), 256));
	return commitResult(out);
}
