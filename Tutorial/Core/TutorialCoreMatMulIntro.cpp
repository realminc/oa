// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial: Core MatMul Intro — the HPC entry point
// Level 1 Core API — OaMatrix + OaFnMatrix::MatMulNt / Linear + OaContext
// ═══════════════════════════════════════════════════════════════════════════
//
// This is the base HPC/Core tutorial. Where the Fashion-MNIST classifier shows
// the ML stack (OaModule, autograd, optimizers), this one shows the layer
// everything else is built on: tensors, one matrix multiply, and the execution
// context. The public surface is intentionally tiny.
//
//   Concept                          OA C++
//   ─────────────────────────────    ─────────────────────────────────────
//   make a tensor (on the GPU)       OaFnMatrix::Rand / Full / Zeros
//   C = A @ B^T                      OaFnMatrix::MatMulNt(a, b)
//   y = x @ W^T + b                  OaFnMatrix::Linear(x, w, bias)
//   run it on the GPU                OaContext::Scope / Execute() + Sync()
//   read it back                     OaFnMatrix::CopyToHost(c, ...)
//
// GPU-only: inputs are generated on-device (Philox RNG via Rand, or the Fill
// kernel via Full). Nothing is uploaded from the host. The only host-side data
// movement is reading results/inputs *back* to build the CPU reference for the
// correctness check — validation, not initialization.
//
// Shape convention (matches Test/Core/Matrix/TestGemmKernels.cpp):
//   A is [M, K], B is stored transposed as [N, K], result C is [M, N], so
//   C[m,n] = sum_k A[m,k] * B[n,k].
//
// The user never calls private GEMM dispatch, OaGemmRouter, or a fused-kernel
// name. The context records semantic MatMul/Linear nodes and the runtime
// selects the kernel (naive / tiled / BF16 CoopMat) from shape + device caps.
// If routing improves, this tutorial gets faster without source changes.
//
// What it demonstrates, in order:
//   1. Basic syntax — one tiny matmul, printed.
//   2. Correctness — random inputs vs a CPU reference across shape archetypes
//      (square, tall-skinny, short-wide, irregular, GEMV, ML layer shapes).
//   3. Performance — wall time + GFLOP/s for the same archetypes at scale.
//   4. Linear — fused x @ W^T + bias with a CPU check.
// ═══════════════════════════════════════════════════════════════════════════

#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/Types.h>
#include <Oa/Ml/FnMatrix.h>          // OaFnMatrix::Linear (bias-broadcasting)
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/RuntimeGlobal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

// ─── Host helpers (validation only) ───────────────────────────────────────

// Read a device tensor back to host. Used purely to build the CPU reference
// and to inspect results — never to initialize GPU data.
static std::vector<float> ReadHost(const OaMatrix& InMat) {
	std::vector<float> h(static_cast<size_t>(InMat.NumElements()));
	(void)OaFnMatrix::CopyToHost(InMat, h.data(), h.size() * sizeof(float));
	return h;
}

// CPU reference: C[m,n] = sum_k A[m,k] * B[n,k]  (B is [N,K], OA convention).
static std::vector<float> CpuGemm(
	const std::vector<float>& InA,
	const std::vector<float>& InB,
	OaU32 InM, OaU32 InN, OaU32 InK)
{
	std::vector<float> c(static_cast<size_t>(InM) * InN, 0.0f);
	for (OaU32 m = 0; m < InM; ++m) {
		for (OaU32 n = 0; n < InN; ++n) {
			float sum = 0.0f;
			for (OaU32 k = 0; k < InK; ++k) {
				sum += InA[m * InK + k] * InB[n * InK + k];
			}
			c[m * InN + n] = sum;
		}
	}
	return c;
}

// Magnitude-normalized error: max element error divided by the largest
// reference magnitude. Robust for random signed data, where individual output
// entries can land near zero and would make per-element relative error explode
// even when the result is numerically fine.
static double MaxRelError(const std::vector<float>& InRef, const std::vector<float>& InGpu) {
	double maxErr = 0.0;
	double maxRef = 1e-6;
	for (size_t i = 0; i < InRef.size(); ++i) {
		maxErr = std::max(maxErr, std::abs(static_cast<double>(InRef[i]) - InGpu[i]));
		maxRef = std::max(maxRef, std::abs(static_cast<double>(InRef[i])));
	}
	return maxErr / maxRef;
}

// ─── Section 1: Basic syntax ────────────────────────────────────────────────

static void DemoBasicSyntax() {
	std::printf("\n── 1. Basic syntax ───────────────────────────────────────────\n");
	std::printf("   A[2,3] (all 1) @ B[2,3]^T (all 2)  ->  C[2,2], each = 3*2 = 6\n\n");

	auto a = OaFnMatrix::Full(OaMatrixShape{2, 3}, 1.0f);
	auto b = OaFnMatrix::Full(OaMatrixShape{2, 3}, 2.0f);

	OaMatrix c;
	{
		OaContext::Scope scope(OaContext::GetDefault());
		c = OaFnMatrix::MatMulNt(a, b);   // records into the default context
	}                                   // scope exit: Execute() + Sync()

	std::vector<float> host(2 * 2);
	(void)OaFnMatrix::CopyToHost(c, host.data(), host.size() * sizeof(float));
	std::printf("   C = [ %.1f %.1f ; %.1f %.1f ]\n", host[0], host[1], host[2], host[3]);
}

// ─── Section 2: Correctness vs CPU ────────────────────────────────────────────

struct ShapeCase {
	const char* Name;
	OaU32 M, N, K;
};

static bool RunCorrectness(const ShapeCase& InCase, double InTol) {
	// Inputs generated ON THE GPU (Philox uniform RNG) — no host upload.
	auto a = OaFnMatrix::Rand(OaMatrixShape{InCase.M, InCase.K});
	auto b = OaFnMatrix::Rand(OaMatrixShape{InCase.N, InCase.K});

	// Read inputs back only to build the CPU reference (validation).
	const auto hA = ReadHost(a);
	const auto hB = ReadHost(b);

	OaMatrix c;
	{
		OaContext::Scope scope(OaContext::GetDefault());
		c = OaFnMatrix::MatMulNt(a, b);
	}

	const auto gpu = ReadHost(c);
	const auto ref = CpuGemm(hA, hB, InCase.M, InCase.N, InCase.K);
	const double err = MaxRelError(ref, gpu);
	const bool ok = err < InTol;

	std::printf("   %-16s [%5u,%5u,%5u]   norm_err=%.2e   tol=%.1e   %s\n",
		InCase.Name, InCase.M, InCase.N, InCase.K, err, InTol, ok ? "ok" : "FAIL");
	return ok;
}

// ─── Section 3: Performance ────────────────────────────────────────────────

// One-precision benchmark summary. Wall p50 is the primary throughput metric;
// p95 exposes jitter/throttling instead of hiding it behind a best-of-N sample.
struct PerfResult { double P50Ms; double P95Ms; double Gflops; };

static PerfResult SummarizeSamples(std::vector<double> InSamples, double InGflop) {
	if (InSamples.empty()) return {};
	std::sort(InSamples.begin(), InSamples.end());
	const auto percentile = [&](double p) {
		const size_t idx = static_cast<size_t>(std::ceil(p * InSamples.size())) - 1;
		return InSamples[std::min(idx, InSamples.size() - 1)];
	};
	const double p50 = percentile(0.50);
	const double p95 = percentile(0.95);
	return {p50, p95, p50 > 0.0 ? InGflop / (p50 / 1000.0) : 0.0};
}

static PerfResult BenchOne(
	const ShapeCase& InCase,
	OaContextMatMulPrecision InPrecision,
	int InWarmup,
	int InIters)
{
	auto& ctx = OaContext::GetDefault();

	auto a = OaFnMatrix::Rand(OaMatrixShape{InCase.M, InCase.K});
	auto b = OaFnMatrix::Rand(OaMatrixShape{InCase.N, InCase.K});
	(void)ctx.Execute();
	(void)ctx.Sync();

	auto once = [&]() {
		OaMatrix c = OaFnMatrix::MatMulNt(a, b, InPrecision);
		(void)ctx.Execute();
		(void)ctx.Sync();
	};

	for (int i = 0; i < InWarmup; ++i) { once(); }

	std::vector<double> samples;
	samples.reserve(static_cast<size_t>(InIters));
	for (int i = 0; i < InIters; ++i) {
		const auto t0 = std::chrono::high_resolution_clock::now();
		once();
		const auto t1 = std::chrono::high_resolution_clock::now();
		const double iterMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
		samples.push_back(iterMs);
	}
	const double gflop  = 2.0 * InCase.M * InCase.N * InCase.K / 1e9;
	return SummarizeSamples(std::move(samples), gflop);
}

// Pre-allocated bench: reuses the same output buffer across iterations,
// pre-alloc dC and reuse for all iters (standard GEMM-benchmark practice).
// This isolates kernel + dispatch overhead from per-call buffer allocation.
static PerfResult BenchOnePreAlloc(
	const ShapeCase& InCase,
	OaContextMatMulPrecision InPrecision,
	int InWarmup,
	int InIters)
{
	auto& ctx = OaContext::GetDefault();

	auto a = OaFnMatrix::Rand(OaMatrixShape{InCase.M, InCase.K});
	auto b = OaFnMatrix::Rand(OaMatrixShape{InCase.N, InCase.K});
	auto c = OaFnMatrix::Empty(OaMatrixShape{InCase.M, InCase.N}, a.Dtype_);
	(void)ctx.Execute();
	(void)ctx.Sync();

	auto once = [&]() {
		ctx.AddMatMul(a, b, c,
			static_cast<OaU32>(InCase.M),
			static_cast<OaU32>(InCase.N),
			static_cast<OaU32>(InCase.K),
			InPrecision);
		(void)ctx.Execute();
		(void)ctx.Sync();
	};

	for (int i = 0; i < InWarmup; ++i) { once(); }

	std::vector<double> samples;
	samples.reserve(static_cast<size_t>(InIters));
	for (int i = 0; i < InIters; ++i) {
		const auto t0 = std::chrono::high_resolution_clock::now();
		once();
		const auto t1 = std::chrono::high_resolution_clock::now();
		const double iterMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
		samples.push_back(iterMs);
	}
	const double gflop  = 2.0 * InCase.M * InCase.N * InCase.K / 1e9;
	return SummarizeSamples(std::move(samples), gflop);
}

// Two-precision bench. Reports Auto (BF16 CoopMat on RTX 5090) and Fp32
// (Tiled scalar fallback) side-by-side so the user can see the tensor-core
// vs FP32 gap on the same shape with the same public API. A sister
// benchmark does the same across the two OA precision paths.
static void RunPerf(const ShapeCase& InCase, int InWarmup, int InIters) {
	const auto autoR = BenchOne(InCase, OaContextMatMulPrecision::Auto, InWarmup, InIters);
	const auto fp32R = BenchOne(InCase, OaContextMatMulPrecision::Fp32, InWarmup, InIters);

	std::printf("   %-16s [%5u,%5u,%5u]   Auto p50 %7.3f p95 %7.3f ms %8.1f GFLOP/s   |   Fp32 p50 %7.3f p95 %7.3f ms %8.1f GFLOP/s\n",
		InCase.Name, InCase.M, InCase.N, InCase.K,
		autoR.P50Ms, autoR.P95Ms, autoR.Gflops, fp32R.P50Ms, fp32R.P95Ms, fp32R.Gflops);
}

// Pre-allocated bench: shows kernel + dispatch overhead without per-call
// buffer allocation, the standard pre-alloc benchmark pattern.
static void RunPerfPreAlloc(const ShapeCase& InCase, int InWarmup, int InIters) {
	const auto autoR = BenchOnePreAlloc(InCase, OaContextMatMulPrecision::Auto, InWarmup, InIters);
	const auto fp32R = BenchOnePreAlloc(InCase, OaContextMatMulPrecision::Fp32, InWarmup, InIters);

	std::printf("   %-16s [%5u,%5u,%5u]   Auto p50 %7.3f p95 %7.3f ms %8.1f GFLOP/s   |   Fp32 p50 %7.3f p95 %7.3f ms %8.1f GFLOP/s\n",
		InCase.Name, InCase.M, InCase.N, InCase.K,
		autoR.P50Ms, autoR.P95Ms, autoR.Gflops, fp32R.P50Ms, fp32R.P95Ms, fp32R.Gflops);
}

// Pipelined bench: N submits without waiting, then one sync at the end.
// Shows amortized per-op cost when the host doesn't block between ops.
// Uses Compile() once + N×Replay() (non-blocking) + 1×WaitForPendingReplay().
// This bypasses Execute()'s ClearNodes() which would force a wait between
// submits (descriptor pools must survive until GPU consumes them).
static PerfResult BenchOnePipelined(
	const ShapeCase& InCase,
	OaContextMatMulPrecision InPrecision,
	int InPipelineDepth,
	int InWarmup,
	int InIters)
{
	auto& ctx = OaContext::GetDefault();

	auto a = OaFnMatrix::Rand(OaMatrixShape{InCase.M, InCase.K});
	auto b = OaFnMatrix::Rand(OaMatrixShape{InCase.N, InCase.K});
	auto c = OaFnMatrix::Empty(OaMatrixShape{InCase.M, InCase.N}, a.Dtype_);
	(void)ctx.Execute();
	(void)ctx.Sync();

	auto* rt = ctx.GetRuntime();
	auto* graph = ctx.Graph();

	auto oncePipeline = [&]() {
		// Record one matmul into the graph and compile.
		ctx.AddMatMul(a, b, c,
			static_cast<OaU32>(InCase.M),
			static_cast<OaU32>(InCase.N),
			static_cast<OaU32>(InCase.K),
			InPrecision);
		(void)graph->Compile(*rt);
		// Submit N times without waiting — same-queue ordering ensures
		// GPU executes in submission order. The pre-recorded primary CB
		// with SIMULTANEOUS_USE_BIT is resubmitted each time.
		for (int i = 0; i < InPipelineDepth; ++i) {
			(void)graph->Replay(*rt);
		}
		// One wait for all N submissions.
		(void)graph->WaitForPendingReplay(rt->Device);
		// Clear for next iteration.
		graph->ClearNodes();
	};

	for (int i = 0; i < InWarmup; ++i) { oncePipeline(); }

	std::vector<double> samples;
	samples.reserve(static_cast<size_t>(InIters));
	for (int i = 0; i < InIters; ++i) {
		const auto t0 = std::chrono::high_resolution_clock::now();
		oncePipeline();
		const auto t1 = std::chrono::high_resolution_clock::now();
		const double iterMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
		samples.push_back(iterMs / InPipelineDepth);
	}
	const double gflop  = 2.0 * InCase.M * InCase.N * InCase.K / 1e9;
	return SummarizeSamples(std::move(samples), gflop);
}

static void RunPerfPipelined(const ShapeCase& InCase, int InPipelineDepth, int InWarmup, int InIters) {
	const auto autoR = BenchOnePipelined(InCase, OaContextMatMulPrecision::Auto, InPipelineDepth, InWarmup, InIters);

	std::printf("   %-16s [%5u,%5u,%5u]   ×%d p50 %7.3f p95 %7.3f ms/op %8.1f GFLOP/s\n",
		InCase.Name, InCase.M, InCase.N, InCase.K,
		InPipelineDepth, autoR.P50Ms, autoR.P95Ms, autoR.Gflops);
}

// ─── Section 3b: Batch Dispatch ────────────────────────────────────────────

static PerfResult BenchBatchOne(
	const ShapeCase& InCase,
	OaContextMatMulPrecision InPrecision,
	int InBatchSize,
	int InWarmup,
	int InIters)
{
	auto& ctx = OaContext::GetDefault();

	OaVec<OaMatrix> aVec;
	OaVec<OaMatrix> bVec;
	for (int i = 0; i < InBatchSize; ++i) {
		aVec.PushBack(OaFnMatrix::Rand(OaMatrixShape{InCase.M, InCase.K}));
		bVec.PushBack(OaFnMatrix::Rand(OaMatrixShape{InCase.N, InCase.K}));
	}
	(void)ctx.Execute();
	(void)ctx.Sync();

	auto onceBatch = [&]() {
		OaVec<OaMatrix> results;
		for (int i = 0; i < InBatchSize; ++i) {
			results.PushBack(OaFnMatrix::MatMulNt(aVec[i], bVec[i], InPrecision));
		}
		(void)ctx.Execute();
		(void)ctx.Sync();
	};

	for (int i = 0; i < InWarmup; ++i) { onceBatch(); }

	std::vector<double> samples;
	samples.reserve(static_cast<size_t>(InIters));
	for (int i = 0; i < InIters; ++i) {
		const auto t0 = std::chrono::high_resolution_clock::now();
		onceBatch();
		const auto t1 = std::chrono::high_resolution_clock::now();
		const double iterMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
		samples.push_back(iterMs);
	}
	const double totalGflop = static_cast<double>(InBatchSize) *
		(2.0 * InCase.M * InCase.N * InCase.K / 1e9);
	return SummarizeSamples(std::move(samples), totalGflop);
}

static void RunBatchPerf(const ShapeCase& InCase, int InBatchSize, int InWarmup, int InIters) {
	const auto autoR = BenchBatchOne(InCase, OaContextMatMulPrecision::Auto, InBatchSize, InWarmup, InIters);
	const auto fp32R = BenchBatchOne(InCase, OaContextMatMulPrecision::Fp32, InBatchSize, InWarmup, InIters);

	std::printf("   %-16s [%5u,%5u,%5u] ×%d   Auto p50 %7.3f p95 %7.3f ms total, %7.3f ms/op %8.1f GFLOP/s   |   Fp32 p50 %7.3f p95 %7.3f ms total, %7.3f ms/op %8.1f GFLOP/s\n",
		InCase.Name, InCase.M, InCase.N, InCase.K, InBatchSize,
		autoR.P50Ms, autoR.P95Ms, autoR.P50Ms / InBatchSize, autoR.Gflops,
		fp32R.P50Ms, fp32R.P95Ms, fp32R.P50Ms / InBatchSize, fp32R.Gflops);
}

// ─── Section 5: Autotuner Benchmark Grid ────────────────────────────────────
//
// Expanded shape matrix for OaGemmRouter autotuning. Covers the full space the
// router cares about: square tiles, tall/skinny, GEMV, small/large K, and
// realistic LLM layer shapes. Gated by OA_AUTOTUNE_BENCH so the tutorial
// stays fast by default.
//
// Output is CSV: shape,m,n,k,precision,p50_ms,p95_ms,gflops
//
// Typical run:
//   OA_AUTOTUNE_BENCH=1 ./TutorialCoreMatMulIntro

struct AutotuneRow {
	const char* Name;
	OaU32 M, N, K;
};

static const AutotuneRow kAutotune[] = {
	// Small squares (warm-up / correctness baselines)
	{"sq-64",         64,    64,    64   },
	{"sq-128",        128,   128,   128  },
	{"sq-256",        256,   256,   256  },
	{"sq-512",        512,   512,   512  },
	// Medium squares
	{"sq-1024",       1024,  1024,  1024 },
	{"sq-2048",       2048,  2048,  2048 },
	// Large squares (persistent-kernel territory)
	{"sq-4096",       4096,  4096,  4096 },
	{"sq-8192",       8192,  8192,  8192 },
	// Tall-skinny (M >> N)
	{"ts-4096x128",   4096,  128,   1024 },
	{"ts-8192x128",   8192,  128,   1024 },
	{"ts-2048x64",    2048,  64,    512  },
	{"ts-16384x256",  16384, 256,   4096 },
	// Short-wide (N >> M)
	{"sw-128x4096",   128,   4096,  1024 },
	{"sw-128x8192",   128,   8192,  1024 },
	{"sw-64x2048",    64,    2048,  512  },
	// GEMV (M = 1)
	{"gemv-1x256",    1,     256,   256  },
	{"gemv-1x4096",   1,     4096,  4096 },
	{"gemv-1x8192",   1,     8192,  4096 },
	{"gemv-1x32000",  1,     32000, 4096 },
	// Small K (K-tile edge cases: K < tile size)
	{"sk-2048",       2048,  2048,  64   },
	{"sk-4096",       4096,  4096,  128  },
	{"sk-8192",       8192,  8192,  256  },
	// Large K (K-dominant)
	{"lk-512",        512,   512,   4096 },
	{"lk-1024",       1024,  1024,  8192 },
	{"lk-2048",       2048,  2048,  16384},
	// LLM-like layer shapes
	{"llm-emb",       128,   4096,  4096 },   // embedding projection
	{"llm-attn-qkv",  128,   12288, 4096 },  // QKV fused (3*h, d_model)
	{"llm-attn-out",  128,   4096,  4096 },  // attention output projection
	{"llm-ffn-up",    128,   11008, 4096 },  // FFN up-projection (LLaMA)
	{"llm-ffn-down",  128,   4096,  11008 }, // FFN down-projection
	{"llm-logits",    128,   32000, 4096 },  // final logits
	// Irregular / non-power-of-2
	{"irr-100",       100,   130,   77   },
	{"irr-333",       333,   777,   111  },
	{"irr-1920",      1920,  1080,  256  },
};

static void RunAutotuneGrid(int InWarmup, int InIters) {
	std::printf("\n── 5. Autotuner Benchmark Grid (pre-allocated, wall-time) ───\n");
	std::printf("   (triggered by OA_AUTOTUNE_BENCH=1)\n");
	std::printf("   Method: std::chrono around Execute()+Sync(), one op per timing sample.\n");
	std::printf("   Matches vk_cooperative_matrix_perf wall-time measurement.\n\n");
	std::printf("shape,m,n,k,precision,p50_ms,p95_ms,gflops\n");

	const OaContextMatMulPrecision precisions[] = {
		OaContextMatMulPrecision::Auto,
		OaContextMatMulPrecision::Bf16,
		OaContextMatMulPrecision::Fp32,
	};
	const char* precNames[] = {"Auto", "Bf16", "Fp32"};

	for (const auto& row : kAutotune) {
		for (size_t p = 0; p < std::size(precisions); ++p) {
			const auto r = BenchOnePreAlloc({row.Name, row.M, row.N, row.K},
			                        precisions[p], InWarmup, InIters);
			std::printf("%s,%u,%u,%u,%s,%.4f,%.4f,%.1f\n",
			            row.Name, row.M, row.N, row.K, precNames[p], r.P50Ms, r.P95Ms, r.Gflops);
		}
	}
}

// ─── Section 4: Linear (x @ W^T + bias) ──────────────────────────────────────

static bool RunLinear(double InTol) {
	const OaU32 M = 8, N = 4, K = 16;   // [batch, out, in]
	// Generated on the GPU: x, weight, bias.
	auto x = OaFnMatrix::Rand(OaMatrixShape{M, K});
	auto w = OaFnMatrix::Rand(OaMatrixShape{N, K});
	auto bias = OaFnMatrix::Rand(OaMatrixShape{N});

	const auto hX = ReadHost(x);
	const auto hW = ReadHost(w);
	const auto hBias = ReadHost(bias);

	OaMatrix y;
	{
		OaContext::Scope scope(OaContext::GetDefault());
		// Linear fuses MatMul + bias-broadcast add into one record. Plain
		// OaFnMatrix::Add is element-wise (no [M,N] + [N] broadcasting) so
		// pairing it with MatMul here would compare against garbage rows;
		// Linear is the right public surface for the y = x @ W^T + bias
		// shape and is what the tutorial header advertises.
		y = OaFnMatrix::Linear(x, w, bias);
	}

	const auto gpu = ReadHost(y);
	auto ref = CpuGemm(hX, hW, M, N, K);
	for (OaU32 m = 0; m < M; ++m)
		for (OaU32 n = 0; n < N; ++n)
			ref[m * N + n] += hBias[n];

	const double err = MaxRelError(ref, gpu);
	const bool ok = err < InTol;
	std::printf("   %-16s [%5u,%5u,%5u]   norm_err=%.2e   tol=%.1e   %s\n",
		"linear+bias", M, N, K, err, InTol, ok ? "ok" : "FAIL");
	return ok;
}

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
	// Device selection via OA_DEVICE so the same binary can run on the discrete
	// GPU, an integrated GPU, or a specific enumeration index:
	//   OA_DEVICE=discrete    (default)  — pick the discrete GPU (e.g. RTX 5090)
	//   OA_DEVICE=integrated             — pick the iGPU (e.g. Intel) to verify
	//                                      the FP32 fallback path with no CoopMat
	//   OA_DEVICE=index:N                — force Vulkan enumeration index N
	OaEngineConfig cfg;
	cfg.AppName = "TutorialCoreMatMulIntro";
	if (const char* dev = std::getenv("OA_DEVICE")) {
		if (std::strcmp(dev, "integrated") == 0 || std::strcmp(dev, "igpu") == 0) {
			cfg.DevicePref = OaDevicePreference::Integrated;
		} else if (std::strcmp(dev, "cpu") == 0) {
			cfg.DevicePref = OaDevicePreference::Cpu;
		} else if (std::strncmp(dev, "index:", 6) == 0) {
			cfg.DevicePref = OaDevicePreference::ByIndex;
			cfg.DeviceIndex = static_cast<OaU32>(std::atoi(dev + 6));
		}
	}

	auto engine = OaComputeEngine::Create(cfg);
	if (!engine.IsOk()) {
		// No Vulkan device available — skip cleanly (exit 0), don't fail CI.
		std::printf("[skip] No OaComputeEngine: %s\n",
			engine.GetStatus().GetMessage().c_str());
		return 0;
	}

	OaComputeEngine& rt = *engine.GetValue();
	OaRuntimeGlobal::SetRuntime(&rt);

	// BF16 CoopMat tensor-core paths trade a little precision for throughput,
	// so the correctness tolerance widens when the router will pick them.
	const bool bf16 = rt.Device.Info.Software.ShaderBfloat16CooperativeMatrixEnabled;
	const double tol = bf16 ? 3e-2 : 1e-4;

	std::printf("\n");
	std::printf("╔═══════════════════════════════════════════════════════════════╗\n");
	std::printf("║              OA Core MatMul Intro — C = A @ B^T               ║\n");
	std::printf("╚═══════════════════════════════════════════════════════════════╝\n");
	std::printf("   GPU            : %s\n", rt.Device.Info.Hardware.DeviceName.c_str());
	std::printf("   BF16 CoopMat   : %s (correctness tol = %.0e)\n",
		bf16 ? "yes" : "no", tol);

	bool ok = true;

	DemoBasicSyntax();

	// Shape archetypes: every GEMM falls into one of these aspect ratios.
	const ShapeCase kCorrectness[] = {
		{"tiny-square",   8,    8,    8   },
		{"square",        128,  128,  128 },
		{"tall-skinny",   512,  64,   128 },
		{"short-wide",    64,   512,  128 },
		{"irregular",     100,  130,  77  },
		{"gemv-decode",   1,    256,  256 },
		{"mnist-hidden",  64,   128,  784 },
		{"mnist-logits",  64,   10,   128 },
	};

	std::printf("\n── 2. Correctness vs CPU reference ───────────────────────────\n");
	for (const auto& c : kCorrectness) ok &= RunCorrectness(c, tol);

	std::printf("\n── 2b. Linear: y = x @ W^T + bias ────────────────────────────\n");
	ok &= RunLinear(tol);

	// Same archetypes scaled up — these exercise the throughput kernels.
	const ShapeCase kPerf[] = {
		{"square-512",    512,  512,  512  },
		{"square-1024",   1024, 1024, 1024 },
		{"square-2048",   2048, 2048, 2048 },
		{"tall-skinny",   4096, 128,  1024 },
		{"short-wide",    128,  4096, 1024 },
		{"gemv-decode",   1,    4096, 4096 },
	};

	std::printf("\n── 3. Performance (public OaFnMatrix::MatMulNt path) ───────────\n");

	// GPU clock warmup: run a large matmul to boost GPU clocks before timing.
	// Without this, the first few shapes measure at lower clock and show high variance.
	{
		auto& warmupCtx = OaContext::GetDefault();
		auto wa = OaFnMatrix::Rand(OaMatrixShape{2048, 2048});
		auto wb = OaFnMatrix::Rand(OaMatrixShape{2048, 2048});
		for (int i = 0; i < 10; ++i) {
			OaMatrix wc = OaFnMatrix::MatMulNt(wa, wb);
			(void)warmupCtx.Execute();
			(void)warmupCtx.Sync();
		}
	}

	for (const auto& c : kPerf) RunPerf(c, /*warmup=*/5, /*iters=*/20);

	// Pre-allocated benchmark: reuses output buffer across iterations,
	// standard pre-alloc pattern. Isolates kernel + dispatch overhead
	// from per-call buffer allocation in the public MatMulNt path.
	std::printf("\n── 3a. Pre-Allocated (kernel + dispatch, no per-call alloc) ────\n");
	for (const auto& c : kPerf) RunPerfPreAlloc(c, /*warmup=*/5, /*iters=*/20);

	// Pipelined benchmark: N submits without waiting, then one sync.
	// Shows amortized per-op cost with non-blocking Replay() + deferred Sync().
	std::printf("\n── 3a2. Pipelined (N submits + 1 sync, fire-and-forget) ──────\n");
	for (const auto& c : kPerf) RunPerfPipelined(c, /*depth=*/8, /*warmup=*/3, /*iters=*/10);

	// ── ML layer shapes: the GEMMs the Transformer / FFN / ALM actually issue.
	// A[M,K] @ B[N,K]^T. NLP suite: D=32, DFF=64, rows=B*S=64*16=1024.
	// ALM prior: D=384, DFF=1536, rows=B*S=32*128=4096. Far smaller than the
	// square peak shapes, so a big single-vs-pipelined gap here means submit-bound
	// — exactly the regime QKV/SwiGLU fusion targets by cutting dispatch count.
	const ShapeCase kMlShapes[] = {
		{"nlp-qkv",   1024, 32,   32   },
		{"nlp-ffn1",  1024, 64,   32   },
		{"nlp-ffn2",  1024, 32,   64   },
		{"alm-qkv",   4096, 384,  384  },
		{"alm-ffn1",  4096, 1536, 384  },
		{"alm-ffn2",  4096, 384,  1536 },
	};
	std::printf("\n── 3d. ML layer shapes — single dispatch (public MatMulNt) ────\n");
	for (const auto& c : kMlShapes) RunPerf(c, /*warmup=*/5, /*iters=*/20);
	std::printf("\n── 3d2. ML layer shapes — pipelined ×8 (amortized submit) ─────\n");
	for (const auto& c : kMlShapes) RunPerfPipelined(c, /*depth=*/8, /*warmup=*/3, /*iters=*/10);

	// Batch dispatch demonstration: multiple ops in single execute
	std::printf("\n── 3b. Batch Dispatch (multiple ops, single submit) ──────────\n");
	const ShapeCase kBatch[] = {
		{"square-512",    512,  512,  512  },
		{"square-1024",   1024, 1024, 1024 },
		{"square-2048",   2048, 2048, 2048 },
		{"tall-skinny",   2048, 128,  1024 },
	};
	for (const auto& c : kBatch) {
		RunBatchPerf(c, /*batchSize=*/4, /*warmup=*/3, /*iters=*/10);
	}
	std::printf("\n── 3c. Batch Dispatch ×8 (larger batch, single submit) ──────────\n");
	for (const auto& c : kBatch) {
		RunBatchPerf(c, /*batchSize=*/8, /*warmup=*/3, /*iters=*/10);
	}

	// Extended autotuner grid (OA_AUTOTUNE_BENCH=1)
	if (std::getenv("OA_AUTOTUNE_BENCH")) {
		RunAutotuneGrid(/*warmup=*/3, /*iters=*/15);
	}

	auto sync = OaContext::GetDefault().Sync();
	if (!sync.IsOk()) {
		std::printf("Context sync failed: %s\n", sync.GetMessage().c_str());
		ok = false;
	}

	std::printf("\n%s\n\n", ok
		? "All correctness checks passed."
		: "Some correctness checks FAILED — see rows marked FAIL above.");
	return ok ? 0 : 2;
}
