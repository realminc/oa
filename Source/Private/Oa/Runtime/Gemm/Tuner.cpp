// OaGemmTuner Implementation
// Runtime GEMM kernel benchmarking and cache population
//
// BENCHMARKS THE REAL PRODUCTION PATH:
//   OaMatrix + OaFnMatrix::MatMulNt -> OaContext::AddMatMul(OaMatrix&) ->
//   OaGemmRouter::Select -> GemmCmSgBf16 / GemmCmWgBf16 (KHR CoopMat)
//
// This is the SAME path TutorialCoreMatMulIntro uses.

#include <Oa/Runtime/Gemm/Tuner.h>
#include <Oa/Runtime/Gemm/Router.h>
#include <Oa/Runtime/Gemm/Cache.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/GemmRouteCache.h>
#include <Oa/Runtime/MatmulTypes.h>
#include <Oa/Runtime/GpuTimer.h>
#include <Oa/Core/Log.h>

// Candidate entry for benchmarking
struct Candidate {
	const OaMatmulVariant* Variant;
	OaGemmKernel Kernel;
	const char* Name;
};

// Collect all raw-GEMM candidates (no bias, no activation) whose caps are
// satisfied on the current device.
static OaVec<Candidate> CollectCandidates(
	const OaComputeEngine& InRt,
	OaU32 InM, OaU32 InN, OaU32 InK)
{
	OaVec<Candidate> candidates;
	const OaU64 caps = InRt.GemmCapsMask();

	for (const auto& v : OaMatmulRegistry::All()) {
		// Skip fused-op kernels. Fused kernels reuse raw kernel enums in the
		// registry, so we filter by name prefix instead of enum value.
		{
			OaStringView name(v.KernelName);
			if (name.find("Bias") != OaStringView::Npos ||
			    name.find("Silu") != OaStringView::Npos ||
			    name.find("SwiGlu") != OaStringView::Npos) {
				continue;
			}
		}
		// Skip persistent variants — they need special grid logic and
		// the OaContext path doesn't handle persistent dispatch yet.
		if (v.Persistent) {
			continue;
		}
		// Skip Naive for non-trivial sizes (always slower, wastes time)
		if (v.Kernel == OaGemmKernel::Naive && InM * InN >= 64) {
			continue;
		}
		// CoopVec only for M==1
		if (v.Path == OaGemmPath::CoopVec && InM != 1) {
			continue;
		}
		// StreamK needs 2-pass overhead; skip for microbenchmarking
		if (v.Path == OaGemmPath::StreamK) {
			continue;
		}
		// Cap check
		if (!OaMatmulRegistry::CapsSatisfy(caps, v.RequiredCapsMask)) {
			continue;
		}
		candidates.PushBack({&v, v.Kernel, v.KernelName});
	}

	return candidates;
}

// Benchmark a single candidate through the REAL production path.
// Creates OaMatrix objects, uses OaFnMatrix::MatMulNt (which routes through
// OaContext::AddMatMul), and measures the submitted graph with GPU timestamps.
//
// Returns average GPU execution time per iteration (ms).
static OaF32 BenchmarkCandidate(
	OaComputeEngine& InRt,
	OaU32 InM, OaU32 InN, OaU32 InK,
	OaGemmKernel InKernel,
	OaU32 InWarmIterations,
	OaU32 InBenchIterations)
{
	if (InBenchIterations == 0U) {
		return 1e9F;
	}

	// Create a temporary context backed by this engine.
	// NOTE: We do NOT try to save/restore the old default via GetDefault()
	// because the caller may not have one (e.g. TestGemmKernels).
	OaContext* ctx = OaContext::Create(&InRt);
	OaContext::SetDefault(ctx);

	// Force the router to pick this specific kernel for the shape.
	// OaFnMatrix::MatMulNt -> OaContext::AddMatMul -> OaGemmRouter::Select
	// will see the ForceKernel entry and route accordingly.
	OaGemmRouter::ForceKernel(InM, InN, InK, InKernel);

	// Create input matrices. B is [N,K] transposed (OA convention).
	OaMatrix a = OaFnMatrix::Rand(OaMatrixShape{InM, InK});
	OaMatrix b = OaFnMatrix::Rand(OaMatrixShape{InN, InK});

	OaGpuTimer timer;
	if (not timer.Init(InRt, "gemm_tuner_candidate").IsOk()) {
		OaGemmRouter::ClearForced();
		OaContext::SetDefault(nullptr);
		delete ctx;
		return 1e9F;
	}

	// Warmup: let the pipeline and clocks settle. ExecuteAsync submits the exact
	// production graph path but does not contaminate the measured samples.
	for (OaU32 i = 0; i < InWarmIterations; ++i) {
		OaMatrix c = OaFnMatrix::MatMulNt(a, b);
		if (not ctx->ExecuteAsync().IsOk()) {
			timer.Destroy(InRt.Device);
			OaGemmRouter::ClearForced();
			OaContext::SetDefault(nullptr);
			delete ctx;
			return 1e9F;
		}
	}

	// Benchmark GPU execution with timestamp queries. CPU recording, allocation,
	// submission, and fence wait are intentionally outside the reported value.
	OaF64 totalGpuMs = 0.0;
	for (OaU32 i = 0; i < InBenchIterations; ++i) {
		OaMatrix c = OaFnMatrix::MatMulNt(a, b);
		if (not ctx->ExecuteAsync(&timer).IsOk()) {
			totalGpuMs = 1e9;
			break;
		}
		const OaF64 sampleMs = timer.ReadbackMs(InRt.Device);
		if (sampleMs <= 0.0) {
			totalGpuMs = 1e9;
			break;
		}
		totalGpuMs += sampleMs;
	}

	// Cleanup
	timer.Destroy(InRt.Device);
	OaGemmRouter::ClearForced();
	OaContext::SetDefault(nullptr);
	delete ctx;

	return static_cast<OaF32>(totalGpuMs / InBenchIterations);
}

// Default shapes to benchmark (training + inference patterns)
static const OaGemmTunerShape kDefaultShapes[] = {
	// Training shapes (batch × seq_len × hidden_dim patterns)
	{64, 256, 128, "train_tiny"},        // Atom config forward
	{256, 256, 256, "train_atom"},       // Atom config standard
	{512, 512, 512, "train_small"},      // Small model
	{1024, 1024, 1024, "train_base"},    // Base model
	
	// Irregular shapes (tall-skinny, Stream-K candidates)
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

OaString OaGemmTuner::GetCachePath(const OaComputeEngine& InRt) {
	(void)InRt;
	return OaString(OaGemmRouteCache::DefaultPath);
}

OaStatus OaGemmTuner::LoadCache(OaComputeEngine& InRt) {
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

OaStatus OaGemmTuner::SaveCache(const OaComputeEngine& InRt) {
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
	OaComputeEngine& InRt,
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

	// Collect candidates for this shape
	auto candidates = CollectCandidates(InRt, M, N, K);
	if (candidates.Size() == 0) {
		return OaStatus::Error("No candidate kernels available for this shape/device");
	}

	// Benchmark each candidate through the REAL production path
	// (OaContext + OaMatrix + OaFnMatrix::MatMulNt). This exercises the same
	// router + kernel path applications use.
	OaF32 bestMs = 1e9f;
	OaGemmKernel bestKernel = OaGemmKernel::Auto;
	const char* bestName = "";

	for (const auto& cand : candidates) {
		OaF32 ms = BenchmarkCandidate(
			InRt, M, N, K,
			cand.Kernel, InWarmIterations, InBenchIterations);

		OaF64 flops = 2.0 * M * N * K;
		OaF32 gflops = static_cast<OaF32>((flops / (ms * 1e-3f)) / 1e9);

		OA_LOG_INFO(OaLogComponent::Core,
			"  OaGemmTuner candidate %s: %.4f ms (%.1f GFLOP/s)",
			cand.Name, ms, gflops);

		if (ms < bestMs) {
			bestMs = ms;
			bestKernel = cand.Kernel;
			bestName = cand.Name;
		}
	}

	// Populate route cache with measured winner
	if (InRt.GemmRouteCache && bestKernel != OaGemmKernel::Auto) {
		OaRouteCacheKey key{};
		key.VendorId = InRt.Device.Info.Hardware.VendorId;
		key.DeviceId = InRt.Device.Info.Hardware.DeviceId;
		key.DriverId = InRt.Device.Info.Software.DriverId;
		OaU64 driverHash = 0xcbf29ce484222325ULL;
		for (const char* p = InRt.Device.Info.Software.DriverVersion.c_str(); *p != '\0'; ++p) {
			driverHash ^= static_cast<OaU8>(*p);
			driverHash *= 0x100000001b3ULL;
		}
		key.DriverVersionHash = driverHash;
		key.ShaderBuildId = OaMatmulRegistry::BuildId();
		key.Variant  = OaGemmKernel::Auto;  // cache key for Auto precision
		key.M = M;
		key.N = N;
		key.K = K;
		key.APrecision = OaGemmPrecision::Auto;
		key.BPrecision = OaGemmPrecision::Auto;
		key.Epilogue = OaGemmEpilogue::None;
		key.Training = false;
		key.UseTMA = InRt.IsBlackwell();

		static OaU64 sGlobalStep = 0;
		InRt.GemmRouteCache->Update(key, bestKernel, bestMs, ++sGlobalStep);

		OA_LOG_INFO(OaLogComponent::Core,
			"OaGemmTuner winner for %ux%ux%u: %s (%.4f ms) -> route cache",
			M, N, K, bestName, bestMs);
	}

	// Calculate GFLOPS for winner
	OaF64 flops = 2.0 * M * N * K;
	OaF32 bestGflops = static_cast<OaF32>((flops / (bestMs * 1e-3f)) / 1e9);

	// Fill result
	OutResult.Shape = InShape;
	OutResult.BestKernel = static_cast<OaU8>(bestKernel);
	OutResult.BestTimeMs = bestMs;
	OutResult.BestGflops = bestGflops;

	return OaStatus::Ok();
}

OaStatus OaGemmTuner::Run(
	OaComputeEngine& InRt,
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
