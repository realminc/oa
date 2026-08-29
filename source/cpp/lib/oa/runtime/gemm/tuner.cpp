// oa::GemmTuner Implementation
// Runtime GEMM kernel benchmarking and cache population
//
// BENCHMARKS THE REAL PRODUCTION PATH:
//   oa::Matrix + Runtime-owned oa::GemmGraphLowering
//   -> immutable oa::MatmulPlan -> private context dispatch recording
//   -> the exact raw or fused generated candidate
//
// high-level operations add semantic/autograd provenance before entering this
// same executable lowering path. The tuner intentionally measures only the
// executable candidate it is selecting.

#include <oa/runtime/gemm/tuner.h>
#include <oa/runtime/gemm/router.h>
#include <oa/runtime/gemm/graphLowering.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include "routeCache.h"
#include "engineRouteCacheAccess.h"
#include "../engine/deviceAccess.h"
#include <oa/runtime/matmulTypes.h>
#include <oa/runtime/timer.h>
#include <oa/core/log.h>

static constexpr oa::F32 kInvalidMeasurementMs = 1e9F;

// Candidate entry for benchmarking
struct Candidate {
	const oa::MatmulVariant* variant;
	const char* name;
};

class ScopedCandidateSession {
public:
	ScopedCandidateSession(
		oa::Engine& inRt)
		: previous_(oa::ExecutionSession::getActivePtr())
		, session_(oa::makeUnique<oa::ExecutionSession>(&inRt))
	{
		oa::ExecutionSession::setActive(session_.get());
	}

	~ScopedCandidateSession() {
		oa::ExecutionSession::setActive(previous_);
	}

	[[nodiscard]] oa::ExecutionSession& session() const { return *session_; }

private:
	oa::ExecutionSession* previous_ = nullptr;
	oa::UniquePtr<oa::ExecutionSession> session_;
};

bool checkedProduct(oa::U32 inA, oa::U32 inB, oa::Usize& outProduct) {
	if (inA == 0U or inB == 0U) return false;
	const oa::Usize a = static_cast<oa::Usize>(inA);
	const oa::Usize b = static_cast<oa::Usize>(inB);
	if (a > oa::Limits<oa::Usize>::max() / b) return false;
	outProduct = a * b;
	return true;
}

oa::F32 applyEpilogue(oa::F32 inValue, oa::GemmEpilogue inEpilogue) {
	switch (inEpilogue) {
		case oa::GemmEpilogue::None:
		case oa::GemmEpilogue::Bias:
			return inValue;
		case oa::GemmEpilogue::BiasRelu:
			return oa::max(0.0F, inValue);
		case oa::GemmEpilogue::BiasGelu: {
			const oa::F32 x3 = inValue * inValue * inValue;
			return 0.5F * inValue * (1.0F + oa::tanh(
				0.7978845608F * (inValue + 0.044715F * x3)));
		}
		case oa::GemmEpilogue::BiasSilu:
			return inValue / (1.0F + oa::exp(-inValue));
		case oa::GemmEpilogue::SiluDual:
			break;
	}
	return oa::Limits<oa::F32>::quietNaN();
}

oa::Status validateCandidate(
	oa::ExecutionSession& inContext,
	const oa::MatmulProblem& inProblem,
	oa::U64 inVariant,
	const oa::GemmTunerShape& inShape)
{
	oa::Usize aCount = 0U;
	oa::Usize bCount = 0U;
	oa::Usize outputCount = 0U;
	if (not checkedProduct(inProblem.m, inProblem.k, aCount)
		or not checkedProduct(inProblem.n, inProblem.k, bCount)
		or not checkedProduct(inProblem.m, inProblem.n, outputCount))
	{
		return oa::Status::invalidArgument(
			"oa::GemmTuner: numerical validation received an invalid shape");
	}

	oa::Vector<oa::F32> aData(aCount);
	oa::Vector<oa::F32> bData(bCount);
	oa::Vector<oa::F32> biasData(
		inProblem.epilogue == oa::GemmEpilogue::None ? 0U : inProblem.n);
	for (oa::Usize i = 0U; i < aData.size(); ++i) {
		aData[i] = static_cast<oa::F32>(
			static_cast<oa::I32>((i % 29U) * 17U % 29U) - 14) * 0.03125F;
	}
	for (oa::Usize i = 0U; i < bData.size(); ++i) {
		bData[i] = static_cast<oa::F32>(
			static_cast<oa::I32>((i % 23U) * 11U % 23U) - 11) * 0.025F;
	}
	for (oa::Usize i = 0U; i < biasData.size(); ++i) {
		biasData[i] = static_cast<oa::F32>(
			static_cast<oa::I32>((i % 13U) * 7U % 13U) - 6) * 0.02F;
	}

	auto a = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(aData.data()),
			aData.size() * sizeof(oa::F32)),
		oa::MatrixShape{inProblem.m, inProblem.k}, oa::ScalarType::Float32);
	auto b = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(bData.data()),
			bData.size() * sizeof(oa::F32)),
		oa::MatrixShape{inProblem.n, inProblem.k}, oa::ScalarType::Float32);
	oa::Matrix bias;
	if (not biasData.empty()) {
		bias = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(biasData.data()),
				biasData.size() * sizeof(oa::F32)),
			oa::MatrixShape{inProblem.n}, oa::ScalarType::Float32);
	}
	auto output = oa::FnMatrix::empty(
		oa::MatrixShape{inProblem.m, inProblem.n}, oa::ScalarType::Float32);
	if (a.isEmpty() or b.isEmpty() or output.isEmpty()
		or (inProblem.epilogue != oa::GemmEpilogue::None and bias.isEmpty()))
	{
		return oa::Status::error(oa::StatusCode::ResourceExhausted,
			"oa::GemmTuner: numerical validation allocation failed");
	}

	const oa::Status recorded = oa::GemmGraphLowering::record(inContext, {
		.a = &a,
		.b = &b,
		.bias = inProblem.epilogue == oa::GemmEpilogue::None ? nullptr : &bias,
		.c = &output,
		.m = inProblem.m,
		.n = inProblem.n,
		.k = inProblem.k,
		.precision = oa::MatMulPrecision::Fp32,
		.epilogue = inProblem.epilogue,
		.preference = {.requiredVariant = inVariant},
		.operation = "oa::GemmTuner::numericalValidation",
	});
	if (not recorded.isOk()) return recorded;
	auto submitted = inContext.submit();
	if (not submitted.isOk()) return submitted.getStatus();
	OA_RETURN_IF_ERROR(inContext.wait(submitted.getValue()));

	oa::Vector<oa::F32> outputData(outputCount);
	OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
		output, outputData.data(), outputData.size() * sizeof(oa::F32)));
	return oa::GemmTuner::validateNumericalOutput(
		inShape,
		oa::Span<const oa::F32>(aData.data(), aData.size()),
		oa::Span<const oa::F32>(bData.data(), bData.size()),
		oa::Span<const oa::F32>(biasData.data(), biasData.size()),
		oa::Span<const oa::F32>(outputData.data(), outputData.size()));
}

// Collect candidates for one exact production contract whose caps are
// satisfied on the current device.
static oa::Vector<Candidate> collectCandidates(
	const oa::Engine& inRt,
	const oa::MatmulProblem& inProblem)
{
	oa::Vector<Candidate> candidates;

	for (const auto& v : oa::matmulRegistry::all()) {
		if (v.epilogue != inProblem.epilogue) continue;
		// Skip Naive for non-trivial sizes (always slower, wastes time)
		if (v.kernel == oa::GemmKernel::Naive && inProblem.m * inProblem.n >= 64) {
			continue;
		}
		// GemmStrided is the universal correctness route for non-canonical views
		// and batches. BenchmarkShape constructs one canonical contiguous problem,
		// where generated tiled candidates are legal and the scalar strided route
		// cannot become the required fallback. Keeping it in this sweep made large
		// product-shape tuning spend most of its time measuring a route that cannot
		// win; arbitrary-layout planning still selects it through the normal router.
		if (v.kernel == oa::GemmKernel::StridedFp32
			or v.kernel == oa::GemmKernel::StridedTiledFp32) {
			continue;
		}
		if (!oa::GemmRouter::isVariantLegal(inRt, v, inProblem)) continue;
		candidates.pushBack({&v, v.kernelName});
	}

	return candidates;
}

// Benchmark a single candidate through Runtime's production executable
// lowering path. high-level semantic/autograd recording is deliberately
// excluded from a kernel-selection measurement.
//
// Returns average GPU execution time per iteration (ms).
static oa::F32 benchmarkCandidate(
	oa::Engine& inRt,
	oa::ExecutionSession& inContext,
	const oa::MatmulProblem& inProblem,
	oa::U64 inVariant,
	oa::U32 inWarmIterations,
	oa::U32 inBenchIterations)
{
	if (inBenchIterations == 0U) {
		return kInvalidMeasurementMs;
	}

	// The cached problem below is explicitly FP32. Keep the actual storage
	// contract independent of the process-wide default weight dtype.
	oa::Matrix a = oa::FnMatrix::rand(
		oa::MatrixShape{inProblem.m, inProblem.k}, oa::ScalarType::Float32);
	oa::Matrix b = oa::FnMatrix::rand(
		oa::MatrixShape{inProblem.n, inProblem.k}, oa::ScalarType::Float32);
	oa::Matrix bias;
	if (inProblem.epilogue != oa::GemmEpilogue::None) {
		bias = oa::FnMatrix::rand(
			oa::MatrixShape{inProblem.n}, oa::ScalarType::Float32);
	}

	oa::Timer timer;
	if (not timer.init(inRt, "gemm_tuner_candidate").isOk()) {
		return kInvalidMeasurementMs;
	}

	auto recordSubmitAndWait = [&](oa::Timer* inTimer = nullptr) {
		oa::Matrix c = oa::FnMatrix::empty(
			oa::MatrixShape{inProblem.m, inProblem.n}, oa::ScalarType::Float32);
		if (c.isEmpty()) return false;
		const oa::Status recorded = oa::GemmGraphLowering::record(inContext, {
			.a = &a,
			.b = &b,
			.bias = inProblem.epilogue == oa::GemmEpilogue::None ? nullptr : &bias,
			.c = &c,
			.m = inProblem.m,
			.n = inProblem.n,
			.k = inProblem.k,
			.precision = oa::MatMulPrecision::Auto,
			.epilogue = inProblem.epilogue,
			.preference = {.requiredVariant = inVariant},
			.operation = "oa::GemmTuner::candidate",
		});
		if (not recorded.isOk()) return false;
		auto submitted = inContext.submit(inTimer);
		return submitted.isOk()
			and inContext.wait(submitted.getValue()).isOk();
	};

	// Warmup: let the pipeline and clocks settle. The explicit completion wait
	// is outside the GPU timestamp interval and does not contaminate samples.
	for (oa::U32 i = 0; i < inWarmIterations; ++i) {
		if (not recordSubmitAndWait()) {
			return kInvalidMeasurementMs;
		}
	}

	// Benchmark GPU execution with timestamp queries. CPU recording, allocation,
	// submission, and fence wait are intentionally outside the reported value.
	oa::F64 totalGpuMs = 0.0;
	bool timingValid = true;
	for (oa::U32 i = 0; i < inBenchIterations; ++i) {
		if (not recordSubmitAndWait(&timer)) {
			timingValid = false;
			break;
		}
		auto committed = timer.commit(inRt);
		if (not committed.isOk()) {
			timingValid = false;
			break;
		}
		const oa::F64 sampleMs = committed.getValue();
		if (not oa::isFinite(sampleMs) or sampleMs <= 0.0) {
			timingValid = false;
			break;
		}
		totalGpuMs += sampleMs;
	}

	return timingValid
		? static_cast<oa::F32>(totalGpuMs / inBenchIterations)
		: kInvalidMeasurementMs;
}

// Default shapes to benchmark (training + inference patterns)
static const oa::GemmTunerShape kDefaultShapes[] = {
	// training shapes (batch × seq_len × hidden_dim patterns)
	{64, 256, 128, "train_tiny"},        // Atom config forward
	{256, 256, 256, "train_atom"},       // Atom config standard
	{512, 512, 512, "train_small"},      // Small model
	{1024, 1024, 1024, "train_base"},    // base model
	{1024, 32, 32, "nlp_qkv_bias", oa::GemmEpilogue::Bias},
	{1024, 64, 32, "nlp_ffn1_bias_gelu", oa::GemmEpilogue::BiasGelu},
	{1024, 32, 64, "nlp_ffn2_bias", oa::GemmEpilogue::Bias},
	{4096, 384, 384, "alm_qkv"},         // OA ALM Q/K/V projection
	{4096, 1536, 384, "alm_ffn1"},       // OA ALM FFN expansion
	{4096, 384, 1536, "alm_ffn2"},       // OA ALM FFN contraction
	
	// Irregular shapes (tall-skinny and short-wide occupancy cases)
	{1000, 256, 512, "irregular_tall"},  // >25% tail waste
	{500, 512, 256, "irregular_wide"},
	{750, 1024, 512, "irregular_med"},
	
	// Inference decode (M=1, CoopVec candidates)
	{1, 256, 256, "decode_tiny"},
	{1, 512, 512, "decode_small"},
	{1, 1024, 1024, "decode_base"},
	{1, 2048, 2048, "decode_large"},
	
	// Batch inference (small batch)
	{4, 1024, 1024, "batch4_base"},
	{8, 512, 512, "batch8_small"},
	{16, 256, 256, "batch16_tiny"},
};

oa::Span<const oa::GemmTunerShape> oa::GemmTuner::getDefaultShapes() {
	return oa::Span<const oa::GemmTunerShape>(kDefaultShapes, sizeof(kDefaultShapes) / sizeof(kDefaultShapes[0]));
}

oa::String oa::GemmTuner::getCachePath(const oa::Engine& inRt) {
	(void)inRt;
	return oa::String(oa::GemmRouteCache::DefaultPath);
}

oa::Status oa::GemmTuner::loadCache(oa::Engine& inRt) {
	auto* routeCache = oa::GemmRouteCacheAccess::get(inRt);
	if (routeCache == nullptr) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::GemmTuner: route cache is not initialized");
	}
	oa::String path = getCachePath(inRt);
	if (routeCache->load(path.data())) {
		OaLogInfo(oa::LogComponent::Compute, "oa::GemmTuner: Loaded route cache from '{}'",
			path.data());
	} else {
		OaLogInfo(oa::LogComponent::Compute,
			"oa::GemmTuner: No compatible route cache at '{}'; benchmark required",
			path.data());
	}
	return oa::Status::ok(); // Missing or stale cache is not an error.
}

oa::Status oa::GemmTuner::saveCache(const oa::Engine& inRt) {
	const auto* routeCache = oa::GemmRouteCacheAccess::get(inRt);
	if (routeCache == nullptr) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::GemmTuner: route cache is not initialized");
	}
	oa::String path = getCachePath(inRt);
	if (not routeCache->save(path.data())) {
		return oa::Status::error(oa::StatusCode::Internal,
			"oa::GemmTuner: failed to save route cache to " + path);
	}
	return oa::Status::ok();
}

oa::Status oa::GemmTuner::validateNumericalOutput(
	const oa::GemmTunerShape& inShape,
	oa::Span<const oa::F32> inA,
	oa::Span<const oa::F32> inB,
	oa::Span<const oa::F32> inBias,
	oa::Span<const oa::F32> inOutput)
{
	size_t aCount = 0U;
	size_t bCount = 0U;
	size_t outputCount = 0U;
	if (not checkedProduct(inShape.m, inShape.k, aCount)
		or not checkedProduct(inShape.n, inShape.k, bCount)
		or not checkedProduct(inShape.m, inShape.n, outputCount))
	{
		return oa::Status::invalidArgument(
			"oa::GemmTuner: numerical oracle requires non-zero bounded dimensions");
	}
	if (inA.size() != aCount or inB.size() != bCount
		or inOutput.size() != outputCount)
	{
		return oa::Status::invalidArgument(
			"oa::GemmTuner: numerical oracle buffer sizes do not match the shape");
	}
	if (inShape.epilogue == oa::GemmEpilogue::SiluDual) {
		return oa::Status::invalidArgument(
			"oa::GemmTuner: dual-output SiLU requires a two-output validation contract");
	}
	const bool hasBias = inShape.epilogue != oa::GemmEpilogue::None;
	if ((hasBias and inBias.size() != inShape.n)
		or (not hasBias and not inBias.empty()))
	{
		return oa::Status::invalidArgument(
			"oa::GemmTuner: numerical oracle bias contract does not match the epilogue");
	}

	auto validateAt = [&](oa::U32 row, oa::U32 col) -> oa::Status {
		oa::F32 expected = hasBias ? inBias[col] : 0.0F;
		for (oa::U32 k = 0U; k < inShape.k; ++k) {
			expected += inA[static_cast<size_t>(row) * inShape.k + k]
				* inB[static_cast<size_t>(col) * inShape.k + k];
		}
		expected = applyEpilogue(expected, inShape.epilogue);
		const oa::F32 actual = inOutput[static_cast<size_t>(row) * inShape.n + col];
		const oa::F32 tolerance = 2.0e-4F
			+ 4.0e-6F * static_cast<oa::F32>(inShape.k)
			+ 2.0e-4F * oa::abs(expected);
		if (not oa::isFinite(expected) or not oa::isFinite(actual)
			or oa::abs(actual - expected) > tolerance)
		{
			return oa::Status::error(oa::StatusCode::DataLoss,
				oa::String("oa::GemmTuner: numerical validation failed at row=")
				+ oa::toString(row) + " col=" + oa::toString(col)
				+ " expected=" + oa::toString(expected)
				+ " actual=" + oa::toString(actual)
				+ " tolerance=" + oa::toString(tolerance));
		}
		return oa::Status::ok();
	};

	constexpr size_t kFullValidationOutputLimit = 4096U;
	if (outputCount <= kFullValidationOutputLimit) {
		for (oa::U32 row = 0U; row < inShape.m; ++row) {
			for (oa::U32 col = 0U; col < inShape.n; ++col) {
				OA_RETURN_IF_ERROR(validateAt(row, col));
			}
		}
		return oa::Status::ok();
	}

	// Always probe tile boundaries and tails, then add reproducible interior
	// coordinates. This keeps validation bounded for product-sized tune shapes
	// while covering the regions most likely to expose dispatch/tail defects.
	const oa::U32 rows[] = {
		0U, oa::min(1U, inShape.m - 1U), inShape.m / 2U,
		inShape.m > 1U ? inShape.m - 2U : 0U, inShape.m - 1U,
	};
	const oa::U32 cols[] = {
		0U, oa::min(1U, inShape.n - 1U), inShape.n / 2U,
		inShape.n > 1U ? inShape.n - 2U : 0U, inShape.n - 1U,
	};
	for (const oa::U32 row : rows) {
		for (const oa::U32 col : cols) {
			OA_RETURN_IF_ERROR(validateAt(row, col));
		}
	}
	oa::U32 state = 0x9e3779b9U ^ inShape.m ^ (inShape.n << 7U)
		^ (inShape.k << 15U) ^ static_cast<oa::U32>(inShape.epilogue);
	for (oa::U32 i = 0U; i < 256U; ++i) {
		state = state * 1664525U + 1013904223U;
		const oa::U32 row = state % inShape.m;
		state = state * 1664525U + 1013904223U;
		const oa::U32 col = state % inShape.n;
		OA_RETURN_IF_ERROR(validateAt(row, col));
	}
	return oa::Status::ok();
}

oa::Status oa::GemmTuner::benchmarkShape(
	oa::Engine& inRt,
	const oa::GemmTunerShape& inShape,
	oa::U32 inWarmIterations,
	oa::U32 inBenchIterations,
	oa::GemmTunerResult& outResult)
{
	if (inBenchIterations == 0U) {
		return oa::Status::invalidArgument("GEMM tuner requires at least one benchmark iteration");
	}

	const oa::U32 M = inShape.m;
	const oa::U32 N = inShape.n;
	const oa::U32 K = inShape.k;
	auto problem = oa::GemmRouter::problemForRaw(
		M, N, K,
		oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	// Runtime executable lowering uses the training-capable contract. Candidate
	// legality and the published cache key must describe that exact path or the
	// measured winner cannot be replayed.
	problem.epilogue = inShape.epilogue;
	problem.training = true;
	problem.precisionHint = oa::GemmPrecision::Auto;

	// Collect candidates for this shape
	auto candidates = collectCandidates(inRt, problem);
	if (candidates.size() == 0) {
		return oa::Status::error("No candidate kernels available for this shape/device");
	}

	// Keep tuning isolated from the engine's ordinary recorder, but reuse one
	// execution context for every candidate block and the winner oracle. Each
	// sample explicitly submits and consumes its exact event, so the context is
	// clean before the next candidate without repeated construction churn.
	ScopedCandidateSession tuningScope(inRt);

	// Benchmark each candidate through Runtime's production executable path.
	// This exercises the same router, immutable plan, recorder, and kernel path
	// used after public operations have recorded their semantic provenance.
	oa::F32 bestMs = 1e9f;
	oa::GemmKernel bestKernel = oa::GemmKernel::Auto;
	oa::U64 bestVariant = oa::invalidMatmulVariantId;
	const char* bestName = "";
	oa::F32 bestP95Ms = kInvalidMeasurementMs;
	oa::U32 bestSampleCount = 0U;

	// Benchmark in alternating forward/reverse blocks. Running every sample for
	// candidate A before candidate B systematically favors whichever candidate
	// sees the more favorable clock/thermal state, especially on an iGPU. Four
	// short blocks preserve the requested approximate sample count while making
	// each candidate appear early and late in the sweep. The median block mean
	// rejects a single scheduler/clock excursion without hiding stable changes.
	const oa::U32 blockCount = oa::min<oa::U32>(4U, inBenchIterations);
	const oa::U32 iterationsPerBlock =
		(inBenchIterations + blockCount - 1U) / blockCount;
	oa::Vector<oa::Vector<oa::F32>> blockMeans(candidates.size());
	for (oa::U32 block = 0; block < blockCount; ++block) {
		for (oa::U32 order = 0; order < candidates.size(); ++order) {
			const oa::U32 candidateIdx = (block & 1U) == 0U
				? order
				: static_cast<oa::U32>(candidates.size() - 1U - order);
			const auto& candidate = candidates[candidateIdx];
			blockMeans[candidateIdx].pushBack(benchmarkCandidate(
				inRt, tuningScope.session(), problem,
				candidate.variant->id, inWarmIterations, iterationsPerBlock));
		}
	}

	for (oa::U32 candidateIdx = 0; candidateIdx < candidates.size(); ++candidateIdx) {
		const auto& cand = candidates[candidateIdx];
		auto samples = blockMeans[candidateIdx];
		samples.erase(
			oa::removeIf(samples.begin(), samples.end(), [](oa::F32 sample) {
				return not oa::isFinite(sample) or sample <= 0.0F or
					sample >= kInvalidMeasurementMs;
			}),
			samples.end());
		if (samples.empty()) {
			OaLogWarn(oa::LogComponent::Compute,
				"  oa::GemmTuner candidate {}: rejected (no valid GPU timing blocks)",
				cand.name);
			continue;
		}
		oa::sort(samples.begin(), samples.end());
		const oa::U32 middle = static_cast<oa::U32>(samples.size() / 2U);
		const oa::F32 ms = (samples.size() & 1U) != 0U
			? samples[middle]
			: 0.5F * (samples[middle - 1U] + samples[middle]);
		const auto p95Index = static_cast<oa::Usize>(oa::ceil(
			0.95 * static_cast<double>(samples.size()))) - 1U;
		const oa::F32 p95Ms = samples[oa::min(p95Index, samples.size() - 1U)];
		outResult.rankedCandidates.pushBack({
			.variant = cand.variant->id,
			.kernel = cand.variant->kernel,
			.name = cand.name,
			.medianTimeMs = ms,
			.p95TimeMs = p95Ms,
			.sampleCount = static_cast<oa::U32>(samples.size()),
		});

		oa::F64 flops = 2.0 * M * N * K;
		oa::F32 gflops = static_cast<oa::F32>((flops / (ms * 1e-3f)) / 1e9);

		OaLogInfo(oa::LogComponent::Compute,
			"  oa::GemmTuner candidate {}: {:.4f} ms ({:.1f} GFLOP/s, median of {} blocks)",
			cand.name, ms, gflops, static_cast<oa::U32>(samples.size()));

		if (ms < bestMs) {
			bestMs = ms;
			bestKernel = cand.variant->kernel;
			bestVariant = cand.variant->id;
			bestName = cand.name;
			bestP95Ms = p95Ms;
			bestSampleCount = static_cast<oa::U32>(samples.size());
		}
	}
	oa::sort(outResult.rankedCandidates.begin(), outResult.rankedCandidates.end(),
		[](const oa::GemmTunerCandidateResult& a, const oa::GemmTunerCandidateResult& b) {
			return a.medianTimeMs < b.medianTimeMs;
		});

	if (bestVariant == oa::invalidMatmulVariantId or not oa::isFinite(bestMs) or
		bestMs <= 0.0F or bestMs >= kInvalidMeasurementMs) {
		return oa::Status::error("oa::GemmTuner: every legal candidate failed GPU timing");
	}

	// Timing alone never authorizes route publication. Re-run the exact winner
	// on deterministic data and compare it with the independent host oracle.
	// A mismatch leaves the prior route cache untouched.
	const oa::Status numerical = validateCandidate(
		tuningScope.session(), problem, bestVariant, inShape);
	if (not numerical.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"oa::GemmTuner rejected winner {} for {}x{}x{}: {}",
			bestName, M, N, K, numerical.getMessage().data());
		return numerical;
	}

	// Populate route cache with measured winner
	if (auto* routeCache = oa::GemmRouteCacheAccess::get(inRt);
		routeCache and bestVariant != oa::invalidMatmulVariantId) {
		const oa::RouteCacheKey key = oa::GemmRouter::cacheKey(inRt, problem);

		routeCache->publish(
			key, bestVariant, bestMs, bestP95Ms, bestSampleCount);

		OaLogInfo(oa::LogComponent::Compute,
			"oa::GemmTuner winner for {}x{}x{} epilogue={}: {} ({:.4f} ms) -> route cache",
			M, N, K, static_cast<oa::U32>(problem.epilogue), bestName, bestMs);
	}

	// Calculate GFLOPS for winner
	oa::F64 flops = 2.0 * M * N * K;
	oa::F32 bestGflops = static_cast<oa::F32>((flops / (bestMs * 1e-3f)) / 1e9);

	// Fill result
	outResult.shape = inShape;
	outResult.bestVariant = bestVariant;
	outResult.bestKernel = bestKernel;
	outResult.bestTimeMs = bestMs;
	outResult.bestGflops = bestGflops;

	return oa::Status::ok();
}

oa::Status oa::GemmTuner::run(
	oa::Engine& inRt,
	oa::U32 inWarmIterations,
	oa::U32 inBenchIterations)
{
	OaLogInfo(oa::LogComponent::Compute, "oa::GemmTuner: Starting benchmark suite...");
	
	const auto& hw = oa::EngineDeviceAccess::get(inRt).info.hardware;
	OaLogInfo(oa::LogComponent::Compute, "  GPU: {}", hw.deviceName.data());
	OaLogInfo(oa::LogComponent::Compute, "  numSMs: {}", hw.numSMs);
	OaLogInfo(oa::LogComponent::Compute, "  VRAM: {:.2f} GB",
		static_cast<oa::F64>(hw.vramBytes) / (1024.0 * 1024.0 * 1024.0));
	
	auto shapes = getDefaultShapes();
	oa::Vector<oa::GemmTunerResult> results;
	results.reserve(shapes.size());
	
	for (oa::U32 i = 0; i < shapes.size(); ++i) {
		const auto& shape = shapes[i];
		
		oa::GemmTunerResult result;
		oa::Status status = benchmarkShape(inRt, shape, inWarmIterations, inBenchIterations, result);
		
		if (status.isOk()) {
			results.pushBack(result);
			
			OaLogInfo(oa::LogComponent::Compute,
			            "  [{}] {}x{}x{}: {:.3f} ms, {:.1f} GFLOPS",
			            shape.name, shape.m, shape.n, shape.k,
			            result.bestTimeMs, result.bestGflops);
		} else {
			OaLogWarn(oa::LogComponent::Compute,
			            "  [{}] {}x{}x{}: FAILED - {}",
			            shape.name, shape.m, shape.n, shape.k,
			            status.getMessage().data());
		}
	}
	
	// Persist cache to disk
	if (oa::GemmRouteCacheAccess::get(inRt)) {
		oa::Status saveStatus = saveCache(inRt);
		if (not saveStatus.isOk()) {
			return saveStatus;
		}
		OaLogInfo(oa::LogComponent::Compute, "oa::GemmTuner: Saved route cache to {}",
			oa::GemmRouteCache::DefaultPath);
	}

	OaLogInfo(oa::LogComponent::Compute,
	            "oa::GemmTuner: Benchmark complete ({} shapes tested)",
	            static_cast<oa::U32>(results.size()));

	return oa::Status::ok();
}
