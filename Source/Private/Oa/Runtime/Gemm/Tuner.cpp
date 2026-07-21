// OaGemmTuner Implementation
// Runtime GEMM kernel benchmarking and cache population
//
// BENCHMARKS THE REAL PRODUCTION PATH:
//   OaMatrix + OaFnMatrix::{MatMulNt,Linear,LinearRelu,LinearGelu}
//   -> OaContext::Record(OaComputeDispatchDesc) -> immutable OaMatmulPlan
//   -> the exact raw or fused generated candidate
//
// This is the SAME path TutorialCoreMatMulIntro uses.

#include <Oa/Runtime/Gemm/Tuner.h>
#include <Oa/Runtime/Gemm/Router.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/FnMatrix.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/GemmRouteCache.h>
#include <Oa/Runtime/MatmulTypes.h>
#include <Oa/Runtime/GpuTimer.h>
#include <Oa/Core/Log.h>

#include <algorithm>
#include <cmath>
#include <vector>

static constexpr OaF32 kInvalidMeasurementMs = 1e9F;

// Candidate entry for benchmarking
struct Candidate {
	const OaMatmulVariant* Variant;
	const char* Name;
};

// Collect candidates for one exact production contract whose caps are
// satisfied on the current device.
static OaVec<Candidate> CollectCandidates(
	const OaEngine& InRt,
	const OaMatmulProblem& InProblem)
{
	OaVec<Candidate> candidates;

	for (const auto& v : OaMatmulRegistry::All()) {
		if (v.Epilogue != InProblem.Epilogue) continue;
		// Skip Naive for non-trivial sizes (always slower, wastes time)
		if (v.Kernel == OaGemmKernel::Naive && InProblem.M * InProblem.N >= 64) {
			continue;
		}
		if (!OaGemmRouter::IsVariantLegal(InRt, v, InProblem)) continue;
		candidates.PushBack({&v, v.KernelName});
	}

	return candidates;
}

// Benchmark a single candidate through the REAL production path.
// Creates OaMatrix objects, uses the public OaFnMatrix operation matching the
// requested epilogue, and measures the submitted graph with GPU timestamps.
//
// Returns average GPU execution time per iteration (ms).
static OaF32 BenchmarkCandidate(
	OaEngine& InRt,
	const OaMatmulProblem& InProblem,
	OaMatmulVariantId InVariant,
	OaU32 InWarmIterations,
	OaU32 InBenchIterations)
{
	if (InBenchIterations == 0U) {
		return kInvalidMeasurementMs;
	}

	// Create an isolated temporary context backed by this engine. Engine
	// initialization establishes a default context; preserve it so running the
	// tuner cannot invalidate the caller's subsequent work or test teardown.
	OaContext* previousCtx = &OaContext::GetDefault();
	OaContext* ctx = OaContext::Create(&InRt);
	OaContext::SetDefault(ctx);

	// Force the router to pick this specific kernel for the shape.
	// OaFnMatrix::MatMulNt -> OaGemmGraphLowering -> OaGemmRouter::Plan
	// will see the ForceKernel entry and route accordingly.
	OaGemmRouter::ForceVariant(InProblem.M, InProblem.N, InProblem.K, InVariant);

	// Create input matrices. B is [N,K] transposed (OA convention).
	OaMatrix a = OaFnMatrix::Rand(OaMatrixShape{InProblem.M, InProblem.K});
	OaMatrix b = OaFnMatrix::Rand(OaMatrixShape{InProblem.N, InProblem.K});
	OaMatrix bias;
	if (InProblem.Epilogue != OaGemmEpilogue::None) {
		bias = OaFnMatrix::Rand(OaMatrixShape{InProblem.N});
	}
	auto run = [&]() -> OaMatrix {
		switch (InProblem.Epilogue) {
			case OaGemmEpilogue::None: return OaFnMatrix::MatMulNt(a, b);
			case OaGemmEpilogue::Bias: return OaFnMatrix::Linear(a, b, bias);
			case OaGemmEpilogue::BiasRelu: return OaFnMatrix::LinearRelu(a, b, bias);
			case OaGemmEpilogue::BiasGelu: return OaFnMatrix::LinearGelu(a, b, bias);
			default: return {};
		}
	};

	OaGpuTimer timer;
	if (not timer.Init(InRt, "gemm_tuner_candidate").IsOk()) {
		OaGemmRouter::ClearForced();
		OaContext::SetDefault(previousCtx);
		delete ctx;
		return kInvalidMeasurementMs;
	}

	auto submitAndWait = [&](OaGpuTimer* InTimer = nullptr) {
		auto submitted = ctx->Submit(InTimer);
		return submitted.IsOk() and ctx->Wait(submitted.GetValue()).IsOk();
	};

	// Warmup: let the pipeline and clocks settle. The explicit completion wait
	// is outside the GPU timestamp interval and does not contaminate samples.
	for (OaU32 i = 0; i < InWarmIterations; ++i) {
		OaMatrix c = run();
		if (not submitAndWait()) {
			timer.Destroy(InRt.Device);
			OaGemmRouter::ClearForced();
			OaContext::SetDefault(previousCtx);
			delete ctx;
			return kInvalidMeasurementMs;
		}
	}

	// Benchmark GPU execution with timestamp queries. CPU recording, allocation,
	// submission, and fence wait are intentionally outside the reported value.
	OaF64 totalGpuMs = 0.0;
	bool timingValid = true;
	for (OaU32 i = 0; i < InBenchIterations; ++i) {
		OaMatrix c = run();
		if (not submitAndWait(&timer)) {
			timingValid = false;
			break;
		}
		const OaF64 sampleMs = timer.ReadbackMs(InRt.Device);
		if (not std::isfinite(sampleMs) or sampleMs <= 0.0) {
			timingValid = false;
			break;
		}
		totalGpuMs += sampleMs;
	}

	// Cleanup
	timer.Destroy(InRt.Device);
	OaGemmRouter::ClearForced();
	OaContext::SetDefault(previousCtx);
	delete ctx;

	return timingValid
		? static_cast<OaF32>(totalGpuMs / InBenchIterations)
		: kInvalidMeasurementMs;
}

// Default shapes to benchmark (training + inference patterns)
static const OaGemmTunerShape kDefaultShapes[] = {
	// Training shapes (batch × seq_len × hidden_dim patterns)
	{64, 256, 128, "train_tiny"},        // Atom config forward
	{256, 256, 256, "train_atom"},       // Atom config standard
	{512, 512, 512, "train_small"},      // Small model
	{1024, 1024, 1024, "train_base"},    // Base model
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

OaSpan<const OaGemmTunerShape> OaGemmTuner::GetDefaultShapes() {
	return OaSpan<const OaGemmTunerShape>(kDefaultShapes, sizeof(kDefaultShapes) / sizeof(kDefaultShapes[0]));
}

OaString OaGemmTuner::GetCachePath(const OaEngine& InRt) {
	(void)InRt;
	return OaString(OaGemmRouteCache::DefaultPath);
}

OaStatus OaGemmTuner::LoadCache(OaEngine& InRt) {
	if (InRt.GemmRouteCache == nullptr) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaGemmTuner: route cache is not initialized");
	}
	OaString path = GetCachePath(InRt);
	if (InRt.GemmRouteCache->Load(path.Data())) {
		OA_LOG_INFO(OaLogComponent::Core, "OaGemmTuner: Loaded route cache from '%s'",
			path.Data());
	} else {
		OA_LOG_INFO(OaLogComponent::Core,
			"OaGemmTuner: No compatible route cache at '%s'; benchmark required",
			path.Data());
	}
	return OaStatus::Ok(); // Missing or stale cache is not an error.
}

OaStatus OaGemmTuner::SaveCache(const OaEngine& InRt) {
	if (InRt.GemmRouteCache == nullptr) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaGemmTuner: route cache is not initialized");
	}
	OaString path = GetCachePath(InRt);
	if (not InRt.GemmRouteCache->Save(path.Data())) {
		return OaStatus::Error(OaStatusCode::Internal,
			"OaGemmTuner: failed to save route cache to " + path);
	}
	return OaStatus::Ok();
}

OaStatus OaGemmTuner::BenchmarkShape(
	OaEngine& InRt,
	const OaGemmTunerShape& InShape,
	OaU32 InWarmIterations,
	OaU32 InBenchIterations,
	OaGemmTunerResult& OutResult)
{
	if (InBenchIterations == 0U) {
		return OaStatus::InvalidArgument("GEMM tuner requires at least one benchmark iteration");
	}

	const OaU32 M = InShape.M;
	const OaU32 N = InShape.N;
	const OaU32 K = InShape.K;
	auto problem = OaGemmRouter::ProblemForRaw(
		M, N, K,
		OaStoragePrecision::Fp32, OaStoragePrecision::Fp32, true);
	// OaFnMatrix lowering uses the training-capable contract. Candidate legality
	// and the published cache key must describe that exact path or the measured
	// winner cannot be replayed.
	problem.Epilogue = InShape.Epilogue;
	problem.Training = true;
	problem.PrecisionHint = OaGemmPrecision::Auto;

	// Collect candidates for this shape
	auto candidates = CollectCandidates(InRt, problem);
	if (candidates.Size() == 0) {
		return OaStatus::Error("No candidate kernels available for this shape/device");
	}

	// Benchmark each candidate through the REAL production path
	// (OaContext + OaMatrix + OaFnMatrix::MatMulNt). This exercises the same
	// router + kernel path applications use.
	OaF32 bestMs = 1e9f;
	OaGemmKernel bestKernel = OaGemmKernel::Auto;
	OaMatmulVariantId bestVariant = OaInvalidMatmulVariantId;
	const char* bestName = "";
	OaF32 bestP95Ms = kInvalidMeasurementMs;
	OaU32 bestSampleCount = 0U;

	// Benchmark in alternating forward/reverse blocks. Running every sample for
	// candidate A before candidate B systematically favors whichever candidate
	// sees the more favorable clock/thermal state, especially on an iGPU. Four
	// short blocks preserve the requested approximate sample count while making
	// each candidate appear early and late in the sweep. The median block mean
	// rejects a single scheduler/clock excursion without hiding stable changes.
	const OaU32 blockCount = std::min<OaU32>(4U, InBenchIterations);
	const OaU32 iterationsPerBlock =
		(InBenchIterations + blockCount - 1U) / blockCount;
	std::vector<std::vector<OaF32>> blockMeans(candidates.Size());
	for (OaU32 block = 0; block < blockCount; ++block) {
		for (OaU32 order = 0; order < candidates.Size(); ++order) {
			const OaU32 candidateIdx = (block & 1U) == 0U
				? order
				: static_cast<OaU32>(candidates.Size() - 1U - order);
			const auto& candidate = candidates[candidateIdx];
			blockMeans[candidateIdx].push_back(BenchmarkCandidate(
				InRt, problem,
				candidate.Variant->Id, InWarmIterations, iterationsPerBlock));
		}
	}

	for (OaU32 candidateIdx = 0; candidateIdx < candidates.Size(); ++candidateIdx) {
		const auto& cand = candidates[candidateIdx];
		auto samples = blockMeans[candidateIdx];
		samples.erase(
			std::remove_if(samples.begin(), samples.end(), [](OaF32 sample) {
				return not std::isfinite(sample) or sample <= 0.0F or
					sample >= kInvalidMeasurementMs;
			}),
			samples.end());
		if (samples.empty()) {
			OA_LOG_WARN(OaLogComponent::Core,
				"  OaGemmTuner candidate %s: rejected (no valid GPU timing blocks)",
				cand.Name);
			continue;
		}
		std::sort(samples.begin(), samples.end());
		const OaU32 middle = static_cast<OaU32>(samples.size() / 2U);
		const OaF32 ms = (samples.size() & 1U) != 0U
			? samples[middle]
			: 0.5F * (samples[middle - 1U] + samples[middle]);
		const auto p95Index = static_cast<size_t>(std::ceil(
			0.95 * static_cast<double>(samples.size()))) - 1U;
		const OaF32 p95Ms = samples[std::min(p95Index, samples.size() - 1U)];
		OutResult.RankedCandidates.PushBack({
			.Variant = cand.Variant->Id,
			.Kernel = cand.Variant->Kernel,
			.Name = cand.Name,
			.MedianTimeMs = ms,
			.P95TimeMs = p95Ms,
			.SampleCount = static_cast<OaU32>(samples.size()),
		});

		OaF64 flops = 2.0 * M * N * K;
		OaF32 gflops = static_cast<OaF32>((flops / (ms * 1e-3f)) / 1e9);

		OA_LOG_INFO(OaLogComponent::Core,
			"  OaGemmTuner candidate %s: %.4f ms (%.1f GFLOP/s, median of %u blocks)",
			cand.Name, ms, gflops, static_cast<OaU32>(samples.size()));

		if (ms < bestMs) {
			bestMs = ms;
			bestKernel = cand.Variant->Kernel;
			bestVariant = cand.Variant->Id;
			bestName = cand.Name;
			bestP95Ms = p95Ms;
			bestSampleCount = static_cast<OaU32>(samples.size());
		}
	}
	std::sort(OutResult.RankedCandidates.begin(), OutResult.RankedCandidates.end(),
		[](const OaGemmTunerCandidateResult& a, const OaGemmTunerCandidateResult& b) {
			return a.MedianTimeMs < b.MedianTimeMs;
		});

	if (bestVariant == OaInvalidMatmulVariantId or not std::isfinite(bestMs) or
		bestMs <= 0.0F or bestMs >= kInvalidMeasurementMs) {
		return OaStatus::Error("OaGemmTuner: every legal candidate failed GPU timing");
	}

	// Populate route cache with measured winner
	if (InRt.GemmRouteCache && bestVariant != OaInvalidMatmulVariantId) {
		const OaRouteCacheKey key = OaGemmRouter::CacheKey(InRt, problem);

		static OaU64 sGlobalStep = 0;
		InRt.GemmRouteCache->Update(
			key, bestVariant, bestMs, bestP95Ms, bestSampleCount, ++sGlobalStep);

		OA_LOG_INFO(OaLogComponent::Core,
			"OaGemmTuner winner for %ux%ux%u epilogue=%u: %s (%.4f ms) -> route cache",
			M, N, K, static_cast<OaU32>(problem.Epilogue), bestName, bestMs);
	}

	// Calculate GFLOPS for winner
	OaF64 flops = 2.0 * M * N * K;
	OaF32 bestGflops = static_cast<OaF32>((flops / (bestMs * 1e-3f)) / 1e9);

	// Fill result
	OutResult.Shape = InShape;
	OutResult.BestVariant = bestVariant;
	OutResult.BestKernel = bestKernel;
	OutResult.BestTimeMs = bestMs;
	OutResult.BestGflops = bestGflops;

	return OaStatus::Ok();
}

OaStatus OaGemmTuner::Run(
	OaEngine& InRt,
	OaU32 InWarmIterations,
	OaU32 InBenchIterations)
{
	OA_LOG_INFO(OaLogComponent::Core, "OaGemmTuner: Starting benchmark suite...");
	
	const auto& hw = InRt.Device.Info.Hardware;
	OA_LOG_INFO(OaLogComponent::Core, "  GPU: %s", hw.DeviceName.Data());
	OA_LOG_INFO(OaLogComponent::Core, "  NumSMs: %u", hw.NumSMs);
	OA_LOG_INFO(OaLogComponent::Core, "  VRAM: %.2f GB", hw.VramBytes / (1024.0 * 1024.0 * 1024.0));
	
	auto shapes = GetDefaultShapes();
	OaVec<OaGemmTunerResult> results;
	results.Reserve(shapes.Size());
	
	for (OaU32 i = 0; i < shapes.Size(); ++i) {
		const auto& shape = shapes[i];
		
		OaGemmTunerResult result;
		OaStatus status = BenchmarkShape(InRt, shape, InWarmIterations, InBenchIterations, result);
		
		if (status.IsOk()) {
			results.PushBack(result);
			
			OA_LOG_INFO(OaLogComponent::Core, 
			            "  [{}] {}x{}x{}: {:.3f} ms, {:.1f} GFLOPS",
			            shape.Name, shape.M, shape.N, shape.K,
			            result.BestTimeMs, result.BestGflops);
		} else {
			OA_LOG_WARN(OaLogComponent::Core,
			            "  [%s] %ux%ux%u: FAILED - %s",
			            shape.Name, shape.M, shape.N, shape.K,
			            status.GetMessage().Data());
		}
	}
	
	// Persist cache to disk
	if (InRt.GemmRouteCache) {
		OaStatus saveStatus = SaveCache(InRt);
		if (not saveStatus.IsOk()) {
			return saveStatus;
		}
		OA_LOG_INFO(OaLogComponent::Core, "OaGemmTuner: Saved route cache to %s",
			OaGemmRouteCache::DefaultPath);
	}

	OA_LOG_INFO(OaLogComponent::Core,
	            "OaGemmTuner: Benchmark complete (%u shapes tested)",
	            static_cast<OaU32>(results.Size()));

	return OaStatus::Ok();
}
