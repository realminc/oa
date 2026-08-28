// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial: Core MatMul Intro — the HPC entry point
// level 1 Core API — oa::Matrix + oa::FnMatrix::matMulNt / Linear + oa::Engine
// ═══════════════════════════════════════════════════════════════════════════
//
// This is the base HPC/Core tutorial. Where the Fashion-MNIST classifier shows
// the ML stack (oa::Module, autograd, optimizers), this one shows the layer
// everything else is built on: tensors, one matrix multiply, and the engine
// submission boundary. The public surface is intentionally tiny.
//
//   Concept                          OA C++
//   ─────────────────────────────    ─────────────────────────────────────
//   make a tensor (on the GPU)       oa::FnMatrix::rand / Full / Zeros
//   C = A @ B^T                      oa::FnMatrix::matMulNt(a, b)
//   y = x @ W^T + b                  oa::FnMatrix::linear(x, w, bias)
//   compile reusable work            engine.capture(...)
//   run it on the GPU                engine.submit(plan) / event.wait()
//   read it back                     oa::FnMatrix::copyToHost(c, ...)
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
// The user never calls private GEMM dispatch, oa::GemmRouter, or a fused-kernel
// name. Engine capture records semantic MatMul/Linear nodes and the runtime
// selects the kernel (naive / tiled / BF16 CoopMat) from shape + device caps.
// If routing improves, this tutorial gets faster without source changes.
//
// what it demonstrates, in order:
//   1. Basic syntax — one tiny matmul, printed.
//   2. Correctness — random inputs vs a CPU reference across shape archetypes
//      (square, tall-skinny, short-wide, irregular, GEMV, ML layer shapes).
//   3. Performance — wall time + GFLOP/s for the same archetypes at scale.
//   4. Linear — fused x @ W^T + bias with a CPU check.
// ═══════════════════════════════════════════════════════════════════════════

#include <oa/core/fnMatrix.h>
#include <oa/core/matrix.h>
#include <oa/core/types.h>
#include <oa/ml/fnMatrix.h>          // oa::FnMatrix::linear (bias-broadcasting)
#include <oa/runtime/engine.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

[[nodiscard]] static oa::Status submitAndWait(oa::Engine& inEngine) {
	auto submitted = inEngine.submit();
	if (not submitted.isOk()) return submitted.getStatus();
	return inEngine.wait(submitted.getValue());
}

// ─── Host helpers (validation only) ───────────────────────────────────────

// Read a device tensor back to host. Used purely to build the CPU reference
// and to inspect results — never to initialize GPU data.
static std::vector<float> readHost(const oa::Matrix& inMat) {
	std::vector<float> h(static_cast<size_t>(inMat.numElements()));
	(void)oa::FnMatrix::copyToHost(inMat, h.data(), h.size() * sizeof(float));
	return h;
}

// CPU reference: C[m,n] = sum_k A[m,k] * B[n,k]  (B is [N,K], OA convention).
static std::vector<float> cpuGemm(
	const std::vector<float>& inA,
	const std::vector<float>& inB,
	oa::U32 inM, oa::U32 inN, oa::U32 inK)
{
	std::vector<float> c(static_cast<size_t>(inM) * inN, 0.0f);
	for (oa::U32 m = 0; m < inM; ++m) {
		for (oa::U32 n = 0; n < inN; ++n) {
			float sum = 0.0f;
			for (oa::U32 k = 0; k < inK; ++k) {
				sum += inA[m * inK + k] * inB[n * inK + k];
			}
			c[m * inN + n] = sum;
		}
	}
	return c;
}

// Magnitude-normalized error: max element error divided by the largest
// reference magnitude. Robust for random signed data, where individual output
// entries can land near zero and would make per-element relative error explode
// even when the result is numerically fine.
static double maxRelError(const std::vector<float>& inRef, const std::vector<float>& inGpu) {
	double maxErr = 0.0;
	double maxRef = 1e-6;
	for (size_t i = 0; i < inRef.size(); ++i) {
		maxErr = std::max(maxErr, std::abs(static_cast<double>(inRef[i]) - inGpu[i]));
		maxRef = std::max(maxRef, std::abs(static_cast<double>(inRef[i])));
	}
	return maxErr / maxRef;
}

// ─── Section 1: Basic syntax ────────────────────────────────────────────────

static bool demoBasicSyntax(oa::Engine& inEngine) {
	std::printf("\n── 1. Basic syntax ───────────────────────────────────────────\n");
	std::printf("   A[2,3] (all 1) @ B[2,3]^T (all 2)  ->  C[2,2], each = 3*2 = 6\n\n");

	auto a = oa::FnMatrix::full(oa::MatrixShape{2, 3}, 1.0f);
	auto b = oa::FnMatrix::full(oa::MatrixShape{2, 3}, 2.0f);

	auto c = oa::FnMatrix::matMulNt(a, b);
	if (not submitAndWait(inEngine).isOk()) {
		std::fprintf(stderr, "MatMul submission failed\n");
		return false;
	}

	std::vector<float> host(2 * 2);
	(void)oa::FnMatrix::copyToHost(c, host.data(), host.size() * sizeof(float));
	std::printf("   C = [ %.1f %.1f ; %.1f %.1f ]\n", host[0], host[1], host[2], host[3]);
	return true;
}

// ─── Section 2: Correctness vs CPU ────────────────────────────────────────────

struct ShapeCase {
	const char* name;
	oa::U32 m, n, k;
};

static bool runCorrectness(
	oa::Engine& inEngine, const ShapeCase& inCase, double inTol) {
	// Inputs generated ON THE GPU (Philox uniform RNG) — no host upload.
	auto a = oa::FnMatrix::rand(oa::MatrixShape{inCase.m, inCase.k});
	auto b = oa::FnMatrix::rand(oa::MatrixShape{inCase.n, inCase.k});

	if (not submitAndWait(inEngine).isOk()) {
		std::fprintf(stderr, "input submission failed for %s\n", inCase.name);
		return false;
	}

	// Read inputs back only to build the CPU reference (validation).
	const auto hA = readHost(a);
	const auto hB = readHost(b);

	auto c = oa::FnMatrix::matMulNt(a, b);
	if (not submitAndWait(inEngine).isOk()) {
		std::fprintf(stderr, "MatMul submission failed for %s\n", inCase.name);
		return false;
	}

	const auto gpu = readHost(c);
	const auto ref = cpuGemm(hA, hB, inCase.m, inCase.n, inCase.k);
	const double err = maxRelError(ref, gpu);
	const bool ok = err < inTol;

	std::printf("   %-16s [%5u,%5u,%5u]   norm_err=%.2e   tol=%.1e   %s\n",
		inCase.name, inCase.m, inCase.n, inCase.k, err, inTol, ok ? "ok" : "FAIL");
	return ok;
}

// ─── Section 3: Performance ────────────────────────────────────────────────

// One-precision benchmark summary. Wall p50 is the primary throughput metric;
// p95 exposes jitter/throttling instead of hiding it behind a best-of-N sample.
struct PerfResult { double p50Ms; double p95Ms; double gflops; };

static PerfResult summarizeSamples(std::vector<double> inSamples, double inGflop) {
	if (inSamples.empty()) return {};
	std::sort(inSamples.begin(), inSamples.end());
	const auto percentile = [&](double p) {
		const size_t idx = static_cast<size_t>(std::ceil(p * inSamples.size())) - 1;
		return inSamples[std::min(idx, inSamples.size() - 1)];
	};
	const double p50 = percentile(0.50);
	const double p95 = percentile(0.95);
	return {p50, p95, p50 > 0.0 ? inGflop / (p50 / 1000.0) : 0.0};
}

static PerfResult benchOne(
	oa::Engine& inEngine,
	const ShapeCase& inCase,
	oa::MatMulPrecision inPrecision,
	int inWarmup,
	int inIters)
{
	auto a = oa::FnMatrix::rand(oa::MatrixShape{inCase.m, inCase.k});
	auto b = oa::FnMatrix::rand(oa::MatrixShape{inCase.n, inCase.k});
	if (not submitAndWait(inEngine).isOk()) return {};

	oa::Matrix c;
	auto captured = inEngine.capture([&]() {
		c = oa::FnMatrix::matMulNt(a, b, inPrecision);
	});
	if (not captured.isOk()) return {};
	auto plan = oa::move(captured).getValue();

	auto once = [&]() -> bool {
		auto submitted = inEngine.submit(plan);
		if (not submitted.isOk()) return false;
		return inEngine.wait(submitted.getValue()).isOk();
	};

	for (int i = 0; i < inWarmup; ++i) {
		if (not once()) return {};
	}

	std::vector<double> samples;
	samples.reserve(static_cast<size_t>(inIters));
	for (int i = 0; i < inIters; ++i) {
		const auto t0 = std::chrono::high_resolution_clock::now();
		if (not once()) return {};
		const auto t1 = std::chrono::high_resolution_clock::now();
		const double iterMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
		samples.push_back(iterMs);
	}
	const double gflop  = 2.0 * inCase.m * inCase.n * inCase.k / 1e9;
	return summarizeSamples(std::move(samples), gflop);
}

// Two-precision bench. Reports auto (BF16 CoopMat on RTX 5090) and Fp32
// (Tiled scalar fallback) side-by-side so the user can see the tensor-core
// vs FP32 gap on the same shape with the same public API. A sister
// benchmark does the same across the two OA precision paths.
static void runPerf(
	oa::Engine& inEngine, const ShapeCase& inCase, int inWarmup, int inIters) {
	const auto autoR = benchOne(
		inEngine, inCase, oa::MatMulPrecision::Auto, inWarmup, inIters);
	const auto fp32R = benchOne(
		inEngine, inCase, oa::MatMulPrecision::Fp32, inWarmup, inIters);

	std::printf("   %-16s [%5u,%5u,%5u]   Auto p50 %7.3f p95 %7.3f ms %8.1f GFLOP/s   |   Fp32 p50 %7.3f p95 %7.3f ms %8.1f GFLOP/s\n",
		inCase.name, inCase.m, inCase.n, inCase.k,
		autoR.p50Ms, autoR.p95Ms, autoR.gflops, fp32R.p50Ms, fp32R.p95Ms, fp32R.gflops);
}

// Pipelined bench: N submits without waiting, then one sync at the end.
// Shows amortized per-op cost using only the reusable public execution plan.
static PerfResult benchOnePipelined(
	oa::Engine& inEngine,
	const ShapeCase& inCase,
	oa::MatMulPrecision inPrecision,
	int inPipelineDepth,
	int inWarmup,
	int inIters)
{
	auto a = oa::FnMatrix::rand(oa::MatrixShape{inCase.m, inCase.k});
	auto b = oa::FnMatrix::rand(oa::MatrixShape{inCase.n, inCase.k});
	if (not submitAndWait(inEngine).isOk()) return {};

	oa::Matrix c;
	auto captured = inEngine.capture([&]() {
		c = oa::FnMatrix::matMulNt(a, b, inPrecision);
	});
	if (not captured.isOk()) return {};
	auto plan = oa::move(captured).getValue();

	auto oncePipeline = [&]() -> bool {
		std::vector<oa::Event> events;
		events.reserve(static_cast<size_t>(inPipelineDepth));
		for (int i = 0; i < inPipelineDepth; ++i) {
			auto submitted = inEngine.submit(plan);
			if (not submitted.isOk()) return false;
			events.push_back(submitted.getValue());
		}
		for (const auto& event : events) {
			if (not inEngine.wait(event).isOk()) return false;
		}
		return true;
	};

	for (int i = 0; i < inWarmup; ++i) {
		if (not oncePipeline()) return {};
	}

	std::vector<double> samples;
	samples.reserve(static_cast<size_t>(inIters));
	for (int i = 0; i < inIters; ++i) {
		const auto t0 = std::chrono::high_resolution_clock::now();
		if (not oncePipeline()) return {};
		const auto t1 = std::chrono::high_resolution_clock::now();
		const double iterMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
		samples.push_back(iterMs / inPipelineDepth);
	}
	const double gflop  = 2.0 * inCase.m * inCase.n * inCase.k / 1e9;
	return summarizeSamples(std::move(samples), gflop);
}

static void runPerfPipelined(
	oa::Engine& inEngine, const ShapeCase& inCase,
	int inPipelineDepth, int inWarmup, int inIters) {
	const auto autoR = benchOnePipelined(
		inEngine, inCase, oa::MatMulPrecision::Auto,
		inPipelineDepth, inWarmup, inIters);

	std::printf("   %-16s [%5u,%5u,%5u]   ×%d p50 %7.3f p95 %7.3f ms/op %8.1f GFLOP/s\n",
		inCase.name, inCase.m, inCase.n, inCase.k,
		inPipelineDepth, autoR.p50Ms, autoR.p95Ms, autoR.gflops);
}

// ─── Section 3b: Batch Dispatch ────────────────────────────────────────────

static PerfResult benchBatchOne(
	oa::Engine& inEngine,
	const ShapeCase& inCase,
	oa::MatMulPrecision inPrecision,
	int inBatchSize,
	int inWarmup,
	int inIters)
{
	oa::Vector<oa::Matrix> aVec;
	oa::Vector<oa::Matrix> bVec;
	for (int i = 0; i < inBatchSize; ++i) {
		aVec.pushBack(oa::FnMatrix::rand(oa::MatrixShape{inCase.m, inCase.k}));
		bVec.pushBack(oa::FnMatrix::rand(oa::MatrixShape{inCase.n, inCase.k}));
	}
	if (not submitAndWait(inEngine).isOk()) return {};

	oa::Vector<oa::Matrix> results;
	auto captured = inEngine.capture([&]() {
		for (int i = 0; i < inBatchSize; ++i) {
			results.pushBack(oa::FnMatrix::matMulNt(aVec[i], bVec[i], inPrecision));
		}
	});
	if (not captured.isOk()) return {};
	auto plan = oa::move(captured).getValue();

	auto onceBatch = [&]() -> bool {
		auto submitted = inEngine.submit(plan);
		if (not submitted.isOk()) return false;
		return inEngine.wait(submitted.getValue()).isOk();
	};

	for (int i = 0; i < inWarmup; ++i) {
		if (not onceBatch()) return {};
	}

	std::vector<double> samples;
	samples.reserve(static_cast<size_t>(inIters));
	for (int i = 0; i < inIters; ++i) {
		const auto t0 = std::chrono::high_resolution_clock::now();
		if (not onceBatch()) return {};
		const auto t1 = std::chrono::high_resolution_clock::now();
		const double iterMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
		samples.push_back(iterMs);
	}
	const double totalGflop = static_cast<double>(inBatchSize) *
		(2.0 * inCase.m * inCase.n * inCase.k / 1e9);
	return summarizeSamples(std::move(samples), totalGflop);
}

static void runBatchPerf(
	oa::Engine& inEngine, const ShapeCase& inCase,
	int inBatchSize, int inWarmup, int inIters) {
	const auto autoR = benchBatchOne(
		inEngine, inCase, oa::MatMulPrecision::Auto,
		inBatchSize, inWarmup, inIters);
	const auto fp32R = benchBatchOne(
		inEngine, inCase, oa::MatMulPrecision::Fp32,
		inBatchSize, inWarmup, inIters);

	std::printf("   %-16s [%5u,%5u,%5u] ×%d   Auto p50 %7.3f p95 %7.3f ms total, %7.3f ms/op %8.1f GFLOP/s   |   Fp32 p50 %7.3f p95 %7.3f ms total, %7.3f ms/op %8.1f GFLOP/s\n",
		inCase.name, inCase.m, inCase.n, inCase.k, inBatchSize,
		autoR.p50Ms, autoR.p95Ms, autoR.p50Ms / inBatchSize, autoR.gflops,
		fp32R.p50Ms, fp32R.p95Ms, fp32R.p50Ms / inBatchSize, fp32R.gflops);
}

// ─── Section 5: Autotuner Benchmark grid ────────────────────────────────────
//
// Expanded shape matrix for oa::GemmRouter autotuning. Covers the full space the
// router cares about: square tiles, tall/skinny, GEMV, small/large K, and
// realistic LLM layer shapes. Gated by OA_AUTOTUNE_BENCH so the tutorial
// stays fast by default.
//
// output is CSV: shape,m,n,k,precision,p50_ms,p95_ms,gflops
//
// Typical run:
//   OA_AUTOTUNE_BENCH=1 ./TutorialCoreMatMulIntro

struct AutotuneRow {
	const char* name;
	oa::U32 m, n, k;
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

static void runAutotuneGrid(
	oa::Engine& inEngine, int inWarmup, int inIters) {
	std::printf("\n── 5. Autotuner Benchmark grid (captured plan, wall-time) ────\n");
	std::printf("   (triggered by OA_AUTOTUNE_BENCH=1)\n");
	std::printf("   method: std::chrono around submit()+wait(event), one op per timing sample.\n");
	std::printf("   matches vk_cooperative_matrix_perf wall-time measurement.\n\n");
	std::printf("shape,m,n,k,precision,p50_ms,p95_ms,gflops\n");

	const oa::MatMulPrecision precisions[] = {
		oa::MatMulPrecision::Auto,
		oa::MatMulPrecision::Bf16,
		oa::MatMulPrecision::Fp32,
	};
	const char* precNames[] = {"Auto", "Bf16", "Fp32"};

	for (const auto& row : kAutotune) {
		for (size_t p = 0; p < std::size(precisions); ++p) {
			const auto r = benchOne(
				inEngine, {row.name, row.m, row.n, row.k},
				precisions[p], inWarmup, inIters);
			std::printf("%s,%u,%u,%u,%s,%.4f,%.4f,%.1f\n",
			            row.name, row.m, row.n, row.k, precNames[p], r.p50Ms, r.p95Ms, r.gflops);
		}
	}
}

// ─── Section 4: Linear (x @ W^T + bias) ──────────────────────────────────────

static bool runLinear(oa::Engine& inEngine, double inTol) {
	const oa::U32 kM = 8, kN = 4, kK = 16;   // [batch, out, in]
	// generated on the GPU: x, weight, bias.
	auto x = oa::FnMatrix::rand(oa::MatrixShape{kM, kK});
	auto w = oa::FnMatrix::rand(oa::MatrixShape{kN, kK});
	auto bias = oa::FnMatrix::rand(oa::MatrixShape{kN});

	const auto hX = readHost(x);
	const auto hW = readHost(w);
	const auto hBias = readHost(bias);

	oa::Matrix y;
	auto captured = inEngine.capture([&]() {
		// Linear fuses MatMul + bias-broadcast add into one record. Plain
		// oa::FnMatrix::add is element-wise (no [M,N] + [N] broadcasting) so
		// pairing it with MatMul here would compare against garbage rows;
		// Linear is the right public surface for the y = x @ W^T + bias
		// shape and is what the tutorial header advertises.
		y = oa::FnMatrix::linear(x, w, bias);
	});
	if (not captured.isOk()) {
		std::fprintf(
			stderr, "Linear capture failed: %s\n",
			captured.getStatus().getMessage().cStr());
		return false;
	}
	auto plan = oa::move(captured).getValue();
	auto submitted = inEngine.submit(plan);
	if (not submitted.isOk()) {
		std::fprintf(
			stderr, "Linear submission failed: %s\n",
			submitted.getStatus().getMessage().cStr());
		return false;
	}
	const auto completed = submitted.getValue().wait();
	if (not completed.isOk()) {
		std::fprintf(
			stderr, "Linear wait failed: %s\n",
			completed.getMessage().cStr());
		return false;
	}

	const auto gpu = readHost(y);
	auto ref = cpuGemm(hX, hW, kM, kN, kK);
	for (oa::U32 m = 0; m < kM; ++m)
		for (oa::U32 n = 0; n < kN; ++n)
			ref[m * kN + n] += hBias[n];

	const double err = maxRelError(ref, gpu);
	const bool ok = err < inTol;
	std::printf("   %-16s [%5u,%5u,%5u]   norm_err=%.2e   tol=%.1e   %s\n",
		"linear+bias", kM, kN, kK, err, inTol, ok ? "ok" : "FAIL");
	return ok;
}

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
	// Device selection via OA_DEVICE so the same binary can run on the discrete
	// GPU, an integrated GPU, or a specific enumeration index:
	//   OA_DEVICE=discrete    (default)  — pick the discrete GPU (e.g. RTX 5090)
	//   OA_DEVICE=integrated             — pick the iGPU (e.g. Intel) to verify
	//                                      the FP32 fallback path with no CoopMat
	//   OA_DEVICE=index:N                — force vulkan enumeration index N
	oa::EngineConfig cfg;
	cfg.appName = "TutorialCoreMatMulIntro";
	if (const char* dev = std::getenv("OA_DEVICE")) {
		if (std::strcmp(dev, "integrated") == 0 || std::strcmp(dev, "igpu") == 0) {
			cfg.devicePref = oa::DevicePreference::Integrated;
		} else if (std::strcmp(dev, "cpu") == 0) {
			cfg.devicePref = oa::DevicePreference::Cpu;
		} else if (std::strncmp(dev, "index:", 6) == 0) {
			cfg.devicePref = oa::DevicePreference::ByIndex;
			cfg.deviceIndex = static_cast<oa::U32>(std::atoi(dev + 6));
		}
	}

	auto engine = oa::Engine::create(cfg);
	if (!engine.isOk()) {
		// No vulkan device available — skip cleanly (exit 0), don't fail CI.
		std::printf("[skip] No oa::Engine: %s\n",
			engine.getStatus().getMessage().cStr());
		return 0;
	}

	oa::Engine& rt = *engine.getValue();

	// Numeric tolerance follows the requested public precision contract, not a
	// vendor capability or the router's private implementation choice.
	const bool bf16 = rt.getPrecision() == oa::Precision::BF16;
	const double tol = bf16 ? 3e-2 : 1e-4;

	std::printf("\n");
	std::printf("╔═══════════════════════════════════════════════════════════════╗\n");
	std::printf("║              OA Core MatMul Intro — C = A @ B^T               ║\n");
	std::printf("╚═══════════════════════════════════════════════════════════════╝\n");
	std::printf("   GPU            : %.*s\n",
		static_cast<int>(rt.deviceName().size()), rt.deviceName().data());
	std::printf("   Precision      : %s (correctness tol = %.0e)\n",
		bf16 ? "BF16" : "FP32", tol);

	bool ok = true;

	ok &= demoBasicSyntax(rt);

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
	for (const auto& c : kCorrectness) ok &= runCorrectness(rt, c, tol);

	std::printf("\n── 2b. Linear: y = x @ W^T + bias ────────────────────────────\n");
	ok &= runLinear(rt, tol);

	// Same archetypes scaled up — these exercise the throughput kernels.
	const ShapeCase kPerf[] = {
		{"square-512",    512,  512,  512  },
		{"square-1024",   1024, 1024, 1024 },
		{"square-2048",   2048, 2048, 2048 },
		{"tall-skinny",   4096, 128,  1024 },
		{"short-wide",    128,  4096, 1024 },
		{"gemv-decode",   1,    4096, 4096 },
	};

	std::printf("\n── 3. performance (captured public MatMulNt plan) ─────────────\n");
	for (const auto& c : kPerf) {
		runPerf(rt, c, /*warmup=*/5, /*iters=*/20);
	}

	// Pipelined benchmark: N submits without waiting, then one sync.
	// Shows amortized per-op cost through repeated public plan submission.
	std::printf("\n── 3a. pipelined (public reusable plan) ───────────────────────\n");
	for (const auto& c : kPerf) {
		runPerfPipelined(
			rt, c, /*depth=*/8, /*warmup=*/3, /*iters=*/10);
	}

	// ── ML layer shapes: the GEMMs the Transformer / FFN / ALM actually issue.
	// A[M,K] @ B[N,K]^T. NLP suite: D=32, DFF=64, rows=B*S=64*16=1024.
	// ALM prior: D=384, DFF=1536, rows=B*S=32*128=4096. far smaller than the
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
	std::printf("\n── 3b. ML layer shapes — captured public MatMulNt ─────────────\n");
	for (const auto& c : kMlShapes) {
		runPerf(rt, c, /*warmup=*/5, /*iters=*/20);
	}
	std::printf("\n── 3b2. ML layer shapes — pipelined ×8 ────────────────────────\n");
	for (const auto& c : kMlShapes) {
		runPerfPipelined(
			rt, c, /*depth=*/8, /*warmup=*/3, /*iters=*/10);
	}

	// Batch dispatch demonstration: multiple ops in single execute
	std::printf("\n── 3c. Batch capture (multiple ops, one reusable plan) ────────\n");
	const ShapeCase kBatch[] = {
		{"square-512",    512,  512,  512  },
		{"square-1024",   1024, 1024, 1024 },
		{"square-2048",   2048, 2048, 2048 },
		{"tall-skinny",   2048, 128,  1024 },
	};
	for (const auto& c : kBatch) {
		runBatchPerf(
			rt, c, /*batchSize=*/4, /*warmup=*/3, /*iters=*/10);
	}
	std::printf("\n── 3c2. Batch capture ×8 ──────────────────────────────────────\n");
	for (const auto& c : kBatch) {
		runBatchPerf(
			rt, c, /*batchSize=*/8, /*warmup=*/3, /*iters=*/10);
	}

	// Extended autotuner grid (OA_AUTOTUNE_BENCH=1)
	if (std::getenv("OA_AUTOTUNE_BENCH")) {
		runAutotuneGrid(rt, /*warmup=*/3, /*iters=*/15);
	}

	std::printf("\n%s\n\n", ok
		? "All correctness checks passed."
		: "Some correctness checks FAILED — see rows marked FAIL above.");
	return ok ? 0 : 2;
}
