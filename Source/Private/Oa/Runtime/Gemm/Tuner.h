// OaGemmTuner — Runtime GEMM kernel benchmarking and cache population
// Measures actual GEMM performance on the current GPU and persists route hints.
// Caches versioned route results to var/gemm_route_cache.bin.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/GemmTypes.h>

// Forward declarations
class OaComputeEngine;

// Shape to benchmark (common training/inference patterns)
struct OaGemmTunerShape {
	OaU32 M, N, K;
	const char* Name;  // e.g. "train_256x1024x512", "decode_1x1024x1024"
	OaGemmEpilogue Epilogue = OaGemmEpilogue::None;
};

struct OaGemmTunerCandidateResult {
	OaMatmulVariantId Variant = OaInvalidMatmulVariantId;
	OaGemmKernel Kernel = OaGemmKernel::Auto;
	const char* Name = "";
	OaF32 MedianTimeMs = 0.0F;
	OaF32 P95TimeMs = 0.0F;
	OaU32 SampleCount = 0;
};

// Benchmark result for a single shape
struct OaGemmTunerResult {
	OaGemmTunerShape Shape;
	OaMatmulVariantId BestVariant;
	OaGemmKernel BestKernel;
	OaF32 BestTimeMs;    // Best median block-mean GPU time in milliseconds
	OaF32 BestGflops;    // Best GFLOPS achieved
	// Fastest first. This is deliberately the complete legal ranking rather
	// than only the winner, so diagnostics and future policy can compare a
	// stable second choice without re-running an exhaustive tune.
	OaVec<OaGemmTunerCandidateResult> RankedCandidates;
};

// OaGemmTuner — benchmark harness
class OaGemmTuner {
public:
	// Run benchmarks for common shapes and populate cache
	// InWarmIterations: warmup iterations per kernel (default: 3)
	// InBenchIterations: benchmark iterations per kernel (default: 10)
	static OaStatus Run(
		OaComputeEngine& InRt,
		OaU32 InWarmIterations = 3,
		OaU32 InBenchIterations = 10);
	
	// Get default shape list (training + inference patterns)
	static OaSpan<const OaGemmTunerShape> GetDefaultShapes();
	
	// Benchmark a single shape across all available kernels
	static OaStatus BenchmarkShape(
		OaComputeEngine& InRt,
		const OaGemmTunerShape& InShape,
		OaU32 InWarmIterations,
		OaU32 InBenchIterations,
		OaGemmTunerResult& OutResult);
	
	// Get the process-wide route-cache path. Device and driver identity are
	// fields in every key rather than separate filename conventions.
	static OaString GetCachePath(const OaComputeEngine& InRt);
	
	// Load cache from disk (if exists)
	static OaStatus LoadCache(OaComputeEngine& InRt);
	
	// Save cache to disk
	static OaStatus SaveCache(const OaComputeEngine& InRt);
};
