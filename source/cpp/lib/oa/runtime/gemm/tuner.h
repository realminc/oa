// oa::GemmTuner — Runtime GEMM kernel benchmarking and cache population
// Measures actual GEMM performance on the current GPU and persists route hints.
// Caches versioned route results to var/gemm_route_cache.bin.

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/runtime/gemmTypes.h>

// forward declarations
namespace oa { class Engine; }

namespace oa {

// Shape to benchmark (common training/inference patterns)
struct GemmTunerShape {
	oa::U32 m, n, k;
	const char* name;  // e.g. "train_256x1024x512", "decode_1x1024x1024"
	oa::GemmEpilogue epilogue = oa::GemmEpilogue::None;
};

struct GemmTunerCandidateResult {
	oa::U64 variant = oa::invalidMatmulVariantId;
	oa::GemmKernel kernel = oa::GemmKernel::Auto;
	const char* name = "";
	oa::F32 medianTimeMs = 0.0F;
	oa::F32 p95TimeMs = 0.0F;
	oa::U32 sampleCount = 0;
};

// Benchmark result for a single shape
struct GemmTunerResult {
	oa::GemmTunerShape shape;
	oa::U64 bestVariant;
	oa::GemmKernel bestKernel;
	oa::F32 bestTimeMs;    // Best median block-mean GPU time in milliseconds
	oa::F32 bestGflops;    // Best GFLOPS achieved
	// Fastest first. This is deliberately the complete legal ranking rather
	// than only the winner, so diagnostics and future policy can compare a
	// stable second choice without re-running an exhaustive tune.
	oa::Vector<oa::GemmTunerCandidateResult> rankedCandidates;
};

// oa::GemmTuner — benchmark harness
class GemmTuner {
public:
	// run benchmarks for common shapes and populate cache
	// inWarmIterations: warmup iterations per kernel (default: 3)
	// inBenchIterations: benchmark iterations per kernel (default: 10)
	static oa::Status run(
		oa::Engine& inRt,
		oa::U32 inWarmIterations = 3,
		oa::U32 inBenchIterations = 10);
	
	// get default shape list (training + inference patterns)
	static oa::Span<const oa::GemmTunerShape> getDefaultShapes();
	
	// Benchmark a single shape across all available kernels
	static oa::Status benchmarkShape(
		oa::Engine& inRt,
		const oa::GemmTunerShape& inShape,
		oa::U32 inWarmIterations,
		oa::U32 inBenchIterations,
		oa::GemmTunerResult& outResult);

	// Independent host oracle used by the publish transaction. Large problems
	// validate a deterministic set of tile edges and interior coordinates;
	// small problems validate every output. Exposed only through this private
	// Runtime header so the rejection path can be tested without a deliberately
	// broken production shader.
	static oa::Status validateNumericalOutput(
		const oa::GemmTunerShape& inShape,
		oa::Span<const oa::F32> inA,
		oa::Span<const oa::F32> inB,
		oa::Span<const oa::F32> inBias,
		oa::Span<const oa::F32> inOutput);
	
	// get the process-wide route-cache path. Device and driver identity are
	// fields in every key rather than separate filename conventions.
	static oa::String getCachePath(const oa::Engine& inRt);
	
	// load cache from disk (if exists)
	static oa::Status loadCache(oa::Engine& inRt);
	
	// save cache to disk
	static oa::Status saveCache(const oa::Engine& inRt);
};

} // namespace oa
