// Correctness + perf harness for GemmCmSgBf16 — the tuned KHR cooperative-matrix
// GEMM that reads FP32 masters, stages bf16 tiles in shared memory, and uses
// 16×16 CoopMat1 fragments. No pack, no mirror. Fused epilogue variants
// (bias, bias+relu, bias+gelu, silu) share the same core.

#include "../oaTest.h"
#include <oa/runtime/dispatch.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/bindlessAccess.h>
#include <oa/runtime/engine/pipelineAccess.h>
#include <oa/runtime/gemm/engineGemmAccess.h>
#include <oa/runtime/matmulTypes.h>

#include <chrono>
#include <cstring>
#include <random>
#include <vector>

namespace {

// The correctness tests below build FP32 input buffers and validate the DTYPE=0
// (fp32-master) pipeline. Under the BF16 engine (OA_TEST_BF16=1) every dispatch
// selects the DTYPE=1 native-bf16 pipeline, which reinterprets those fp32 bytes
// as packed bf16 → garbage (inf/1e37 norm_err). Skip them under bf16; the native
// bf16 path is covered by nativeBf16Perf (which skips under the fp32 engine).
#define OA_SKIP_IF_BF16_ENGINE(rt)                                                  \
	do {                                                                            \
		if ((rt).getPrecision() == oa::Precision::BF16) {                             \
			GTEST_SKIP() << "fp32-master correctness test skipped under BF16 "      \
			                "engine — see NativeBf16Perf for the bf16 path";        \
		}                                                                           \
	} while (0)

oavk::Buffer makeBuf(oa::Engine& rt, size_t elems) {
	auto res = oa::EngineAllocatorAccess::get(rt).allocHostVisible(elems * sizeof(oa::F32));
	oavk::Buffer buf = *res;
	EXPECT_NE(oa::EngineBindlessAccess::registerBuffer(rt, buf), OA_BINDLESS_INVALID);
	return buf;
}

// tile = the kernel's output-tile edge (workgroups = ceil(M/tile) x ceil(N/tile)).
// SMEM kernels use 128; GemmCmSgBf16 uses 128; GemmCmWgBf16 uses 64. Passing the wrong tile
// under-dispatches → partial output (norm_err ~1) AND inflates TFLOP/s ~ (right/wrong)^2.
float runAndCheck(oa::Engine& rt, oa::U32 M, oa::U32 N, oa::U32 K, const char* kernel, oa::U32 tile = 128u) {
	oavk::Buffer bufA = makeBuf(rt, static_cast<size_t>(M) * K);
	oavk::Buffer bufB = makeBuf(rt, static_cast<size_t>(N) * K);
	oavk::Buffer bufOut = makeBuf(rt, static_cast<size_t>(M) * N);

	auto* aPtr = static_cast<oa::F32*>(bufA.mappedPtr);
	auto* bPtr = static_cast<oa::F32*>(bufB.mappedPtr);
	auto* outPtr = static_cast<oa::F32*>(bufOut.mappedPtr);

	std::mt19937 rng(7);
	std::uniform_real_distribution<oa::F32> dist(-1.0F, 1.0F);
	for (oa::U32 i = 0; i < M * K; ++i) aPtr[i] = dist(rng);
	for (oa::U32 i = 0; i < N * K; ++i) bPtr[i] = dist(rng);
	std::memset(outPtr, 0, static_cast<size_t>(M) * N * sizeof(oa::F32));

	struct { oa::U32 M, N, K; } push = {M, N, K};
	oavk::Buffer bufs[] = {bufA, bufB, bufOut};
	oa::Status st = oavk::Dispatch::run(rt, kernel, bufs, &push, sizeof(push),
		oa::ScalarType::Float32,
		(M + tile - 1u) / tile, (N + tile - 1u) / tile, 1u);
	EXPECT_TRUE(st.isOk()) << st.toString();

	// CPU reference (bf16 round-to-nearest on inputs).
	auto toBf16Rtn = [](oa::F32 v) -> oa::F32 {
		oa::U32 bits; std::memcpy(&bits, &v, sizeof(bits));
		oa::U32 r = (bits + 0x7FFFU + ((bits >> 16U) & 1U)) & 0xFFFF0000U;
		oa::F32 f; std::memcpy(&f, &r, sizeof(f)); return f;
	};
	std::vector<oa::F32> ah(static_cast<size_t>(M) * K), bh(static_cast<size_t>(N) * K);
	for (oa::U32 i = 0; i < M * K; ++i) ah[i] = toBf16Rtn(aPtr[i]);
	for (oa::U32 i = 0; i < N * K; ++i) bh[i] = toBf16Rtn(bPtr[i]);

	float maxErr = 0.0F, maxAbs = 0.0F;
	for (oa::U32 m = 0; m < M; ++m)
		for (oa::U32 n = 0; n < N; ++n) {
			float s = 0.0F;
			for (oa::U32 k = 0; k < K; ++k) s += ah[m * K + k] * bh[n * K + k];
			maxErr = std::max(maxErr, std::abs(outPtr[m * N + n] - s));
			maxAbs = std::max(maxAbs, std::abs(s));
		}

	oa::EngineBindlessAccess::deregisterBuffer(rt, bufA); oa::EngineBindlessAccess::deregisterBuffer(rt, bufB); oa::EngineBindlessAccess::deregisterBuffer(rt, bufOut);
	oa::EngineAllocatorAccess::get(rt).free(bufA); oa::EngineAllocatorAccess::get(rt).free(bufB); oa::EngineAllocatorAccess::get(rt).free(bufOut);
	return maxAbs > 0.0F ? maxErr / maxAbs : 0.0F;
}

float runAndCheckFused(oa::Engine& rt, oa::U32 M, oa::U32 N, oa::U32 K, const char* kernel, oa::U32 tile = 128u) {
	oavk::Buffer bufA = makeBuf(rt, static_cast<size_t>(M) * K);
	oavk::Buffer bufB = makeBuf(rt, static_cast<size_t>(N) * K);
	oavk::Buffer bufBias = makeBuf(rt, static_cast<size_t>(N));
	oavk::Buffer bufOut = makeBuf(rt, static_cast<size_t>(M) * N);

	auto* aPtr = static_cast<oa::F32*>(bufA.mappedPtr);
	auto* bPtr = static_cast<oa::F32*>(bufB.mappedPtr);
	auto* biasPtr = static_cast<oa::F32*>(bufBias.mappedPtr);
	auto* outPtr = static_cast<oa::F32*>(bufOut.mappedPtr);

	std::mt19937 rng(7);
	std::uniform_real_distribution<oa::F32> dist(-1.0F, 1.0F);
	for (oa::U32 i = 0; i < M * K; ++i) aPtr[i] = dist(rng);
	for (oa::U32 i = 0; i < N * K; ++i) bPtr[i] = dist(rng);
	for (oa::U32 i = 0; i < N; ++i) biasPtr[i] = dist(rng);
	std::memset(outPtr, 0, static_cast<size_t>(M) * N * sizeof(oa::F32));

	struct { oa::U32 M, N, K; } push = {M, N, K};
	oavk::Buffer bufs[] = {bufA, bufB, bufBias, bufOut};
	oa::Status st = oavk::Dispatch::run(rt, kernel, bufs, &push, sizeof(push),
		oa::ScalarType::Float32,
		(M + tile - 1u) / tile, (N + tile - 1u) / tile, 1u);
	EXPECT_TRUE(st.isOk()) << st.toString();

	auto toBf16Rtn = [](oa::F32 v) -> oa::F32 {
		oa::U32 bits; std::memcpy(&bits, &v, sizeof(bits));
		oa::U32 r = (bits + 0x7FFFU + ((bits >> 16U) & 1U)) & 0xFFFF0000U;
		oa::F32 f; std::memcpy(&f, &r, sizeof(f)); return f;
	};
	std::vector<oa::F32> ah(static_cast<size_t>(M) * K), bh(static_cast<size_t>(N) * K);
	for (oa::U32 i = 0; i < M * K; ++i) ah[i] = toBf16Rtn(aPtr[i]);
	for (oa::U32 i = 0; i < N * K; ++i) bh[i] = toBf16Rtn(bPtr[i]);

	float maxErr = 0.0F, maxAbs = 0.0F;
	for (oa::U32 m = 0; m < M; ++m)
		for (oa::U32 n = 0; n < N; ++n) {
			float s = 0.0F;
			for (oa::U32 k = 0; k < K; ++k) s += ah[m * K + k] * bh[n * K + k];
			s += biasPtr[n];
			float ref = s;
			if (std::strcmp(kernel, "GemmBiasReluCmSgBf16") == 0 or std::strcmp(kernel, "GemmBiasReluCmWgBf16") == 0) ref = std::max(0.0F, s);
			else if (std::strcmp(kernel, "GemmBiasGeluCmSgBf16") == 0 or std::strcmp(kernel, "GemmBiasGeluCmWgBf16") == 0) {
				float x3 = s * s * s;
				float inner = 0.7978845608F * (s + 0.044715F * x3);
				ref = 0.5F * s * (1.0F + std::tanh(inner));
			}
			else if (std::strcmp(kernel, "GemmBiasSiluCmSgBf16") == 0 or std::strcmp(kernel, "GemmBiasSiluCmWgBf16") == 0) {
				ref = s / (1.0F + std::exp(-s));
			}
			maxErr = std::max(maxErr, std::abs(outPtr[m * N + n] - ref));
			maxAbs = std::max(maxAbs, std::abs(ref));
		}

	oa::EngineBindlessAccess::deregisterBuffer(rt, bufA); oa::EngineBindlessAccess::deregisterBuffer(rt, bufB); oa::EngineBindlessAccess::deregisterBuffer(rt, bufBias); oa::EngineBindlessAccess::deregisterBuffer(rt, bufOut);
	oa::EngineAllocatorAccess::get(rt).free(bufA); oa::EngineAllocatorAccess::get(rt).free(bufB); oa::EngineAllocatorAccess::get(rt).free(bufBias); oa::EngineAllocatorAccess::get(rt).free(bufOut);
	return maxAbs > 0.0F ? maxErr / maxAbs : 0.0F;
}

float runAndCheckSilu(oa::Engine& rt, oa::U32 M, oa::U32 N, oa::U32 K, const char* kernel = "GemmSiluCmSgBf16", oa::U32 tile = 128u) {
	oavk::Buffer bufA = makeBuf(rt, static_cast<size_t>(M) * K);
	oavk::Buffer bufB = makeBuf(rt, static_cast<size_t>(N) * K);
	oavk::Buffer bufPre = makeBuf(rt, static_cast<size_t>(M) * N);
	oavk::Buffer bufAct = makeBuf(rt, static_cast<size_t>(M) * N);

	auto* aPtr = static_cast<oa::F32*>(bufA.mappedPtr);
	auto* bPtr = static_cast<oa::F32*>(bufB.mappedPtr);
	auto* prePtr = static_cast<oa::F32*>(bufPre.mappedPtr);
	auto* actPtr = static_cast<oa::F32*>(bufAct.mappedPtr);

	std::mt19937 rng(7);
	std::uniform_real_distribution<oa::F32> dist(-1.0F, 1.0F);
	for (oa::U32 i = 0; i < M * K; ++i) aPtr[i] = dist(rng);
	for (oa::U32 i = 0; i < N * K; ++i) bPtr[i] = dist(rng);
	std::memset(prePtr, 0, static_cast<size_t>(M) * N * sizeof(oa::F32));
	std::memset(actPtr, 0, static_cast<size_t>(M) * N * sizeof(oa::F32));

	struct { oa::U32 M, N, K; } push = {M, N, K};
	oavk::Buffer bufs[] = {bufA, bufB, bufPre, bufAct};
	oa::Status st = oavk::Dispatch::run(rt, kernel, bufs, &push, sizeof(push),
		oa::ScalarType::Float32,
		(M + tile - 1u) / tile, (N + tile - 1u) / tile, 1u);
	EXPECT_TRUE(st.isOk()) << st.toString();

	auto toBf16Rtn = [](oa::F32 v) -> oa::F32 {
		oa::U32 bits; std::memcpy(&bits, &v, sizeof(bits));
		oa::U32 r = (bits + 0x7FFFU + ((bits >> 16U) & 1U)) & 0xFFFF0000U;
		oa::F32 f; std::memcpy(&f, &r, sizeof(f)); return f;
	};
	auto silu = [](oa::F32 x) {
		return x / (1.0F + std::exp(-x));
	};
	std::vector<oa::F32> ah(static_cast<size_t>(M) * K), bh(static_cast<size_t>(N) * K);
	for (oa::U32 i = 0; i < M * K; ++i) ah[i] = toBf16Rtn(aPtr[i]);
	for (oa::U32 i = 0; i < N * K; ++i) bh[i] = toBf16Rtn(bPtr[i]);

	float maxErr = 0.0F, maxAbs = 0.0F;
	for (oa::U32 m = 0; m < M; ++m)
		for (oa::U32 n = 0; n < N; ++n) {
			float s = 0.0F;
			for (oa::U32 k = 0; k < K; ++k) s += ah[m * K + k] * bh[n * K + k];
			maxErr = std::max(maxErr, std::abs(prePtr[m * N + n] - s));
			maxAbs = std::max(maxAbs, std::abs(s));
			maxErr = std::max(maxErr, std::abs(actPtr[m * N + n] - silu(s)));
			maxAbs = std::max(maxAbs, std::abs(silu(s)));
		}

	oa::EngineBindlessAccess::deregisterBuffer(rt, bufA); oa::EngineBindlessAccess::deregisterBuffer(rt, bufB); oa::EngineBindlessAccess::deregisterBuffer(rt, bufPre); oa::EngineBindlessAccess::deregisterBuffer(rt, bufAct);
	oa::EngineAllocatorAccess::get(rt).free(bufA); oa::EngineAllocatorAccess::get(rt).free(bufB); oa::EngineAllocatorAccess::get(rt).free(bufPre); oa::EngineAllocatorAccess::get(rt).free(bufAct);
	return maxAbs > 0.0F ? maxErr / maxAbs : 0.0F;
}

double benchTflops(oa::Engine& rt, oa::U32 M, oa::U32 N, oa::U32 K, oa::U32 iters, const char* kernel) {
	oavk::Buffer bufA = makeBuf(rt, static_cast<size_t>(M) * K);
	oavk::Buffer bufB = makeBuf(rt, static_cast<size_t>(N) * K);
	oavk::Buffer bufOut = makeBuf(rt, static_cast<size_t>(M) * N);
	std::memset(bufA.mappedPtr, 0, static_cast<size_t>(M) * K * sizeof(oa::F32));
	std::memset(bufB.mappedPtr, 0, static_cast<size_t>(N) * K * sizeof(oa::F32));

	struct { oa::U32 M, N, K; } push = {M, N, K};
	oavk::Buffer bufs[] = {bufA, bufB, bufOut};
	auto run = [&]() {
		(void)oavk::Dispatch::run(rt, kernel, bufs, &push, sizeof(push),
			oa::ScalarType::Float32,
			(M + 127u) / 128u, (N + 127u) / 128u, 1u);
	};
	for (oa::U32 i = 0; i < 10; ++i) run();  // warmup
	auto t0 = std::chrono::high_resolution_clock::now();
	for (oa::U32 i = 0; i < iters; ++i) run();
	auto t1 = std::chrono::high_resolution_clock::now();

	double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
	double gflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e9;

	oa::EngineBindlessAccess::deregisterBuffer(rt, bufA); oa::EngineBindlessAccess::deregisterBuffer(rt, bufB); oa::EngineBindlessAccess::deregisterBuffer(rt, bufOut);
	oa::EngineAllocatorAccess::get(rt).free(bufA); oa::EngineAllocatorAccess::get(rt).free(bufB); oa::EngineAllocatorAccess::get(rt).free(bufOut);
	return gflops / 1000.0;  // TFLOP/s
}

// Batched throughput — the honest measurement. The per-dispatch oavk::Dispatch::run
// does a full submit+WAIT roundtrip, so its wall time is ~90% CPU/sync overhead
// (see BenchTflops above). Here we record `iters` dispatches into ONE batch and
// submit+wait ONCE, amortizing that overhead ~iters-fold, so wall time tracks GPU
// kernel throughput. A buffer barrier between dispatches serializes them (WAW on
// bufOut) so the GPU actually runs `iters` full GEMMs, not overlapped/elided ones.
double benchTflopsBatched(oa::Engine& rt, oa::U32 M, oa::U32 N, oa::U32 K, oa::U32 iters, const char* kernel, oa::U32 tile = 128u) {
	oavk::Buffer bufA = makeBuf(rt, static_cast<size_t>(M) * K);
	oavk::Buffer bufB = makeBuf(rt, static_cast<size_t>(N) * K);
	oavk::Buffer bufOut = makeBuf(rt, static_cast<size_t>(M) * N);
	std::memset(bufA.mappedPtr, 0, static_cast<size_t>(M) * K * sizeof(oa::F32));
	std::memset(bufB.mappedPtr, 0, static_cast<size_t>(N) * K * sizeof(oa::F32));

	struct { oa::U32 M, N, K; } push = {M, N, K};
	oavk::Buffer bufs[] = {bufA, bufB, bufOut};
	const oa::U32 gx = (M + tile - 1u) / tile, gy = (N + tile - 1u) / tile;

	auto runBatch = [&](oa::U32 n) -> double {
		auto batchRes = oavk::Dispatch::beginBatch(rt);
		if (!batchRes.isOk()) return 0.0;
		oavk::Batch batch = batchRes.getValue();
		auto t0 = std::chrono::high_resolution_clock::now();
		for (oa::U32 i = 0; i < n; ++i) {
			(void)oavk::Dispatch::record(
				batch, rt, kernel, bufs, &push, sizeof(push),
				oa::ScalarType::Float32, gx, gy, 1u);
		}
		(void)oavk::Dispatch::flush(batch, rt);  // single submit+wait for all n
		auto t1 = std::chrono::high_resolution_clock::now();
		return std::chrono::duration<double, std::milli>(t1 - t0).count();
	};

	runBatch(10);  // warmup
	double best = 1e30;
	for (int rep = 0; rep < 3; ++rep) {
		double ms = runBatch(iters);
		if (ms > 0.0) best = std::min(best, ms / iters);
	}

	oa::EngineBindlessAccess::deregisterBuffer(rt, bufA); oa::EngineBindlessAccess::deregisterBuffer(rt, bufB); oa::EngineBindlessAccess::deregisterBuffer(rt, bufOut);
	oa::EngineAllocatorAccess::get(rt).free(bufA); oa::EngineAllocatorAccess::get(rt).free(bufB); oa::EngineAllocatorAccess::get(rt).free(bufOut);
	return (2.0 * M * N * K) / (best * 1e-3) / 1e12;  // TFLOP/s from best ms/dispatch
}

oavk::Buffer makeBufBf16(oa::Engine& rt, size_t elems) {
	auto res = oa::EngineAllocatorAccess::get(rt).allocHostVisible(elems * 2u);  // 2 bytes / bf16
	oavk::Buffer buf = *res;
	EXPECT_NE(oa::EngineBindlessAccess::registerBuffer(rt, buf), OA_BINDLESS_INVALID);
	return buf;
}

static inline oa::U16 packBf16(oa::F32 v) {
	oa::U32 b; std::memcpy(&b, &v, 4);
	return static_cast<oa::U16>((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);  // round-to-nearest-even
}
static inline oa::F32 unpackBf16(oa::U16 h) {
	oa::U32 b = static_cast<oa::U32>(h) << 16; oa::F32 f; std::memcpy(&f, &b, 4); return f;
}

// Native-bf16 perf + self-validating correctness. Inputs are 2-byte bf16 buffers;
// the raw GemmCmSgBf16 store path writes fp32 output. REQUIRES the engine in BF16
// precision (run with OA_TEST_BF16=1) so the dispatch selects the DTYPE=1 pipeline
// variant (128-bit uint4 native-bf16 staging). The norm_err check is the guard: if
// the DTYPE=1 variant did NOT engage and the fp32 path read our bf16 buffer as
// float4, the result is garbage (norm_err ~1), so a passing check proves the native
// path actually ran.
double benchBf16(oa::Engine& rt, oa::U32 M, oa::U32 N, oa::U32 K, oa::U32 iters,
                 const char* kernel, oa::U32 tile, float* outNormErr) {
	oavk::Buffer bufA = makeBufBf16(rt, static_cast<size_t>(M) * K);
	oavk::Buffer bufB = makeBufBf16(rt, static_cast<size_t>(N) * K);
	oavk::Buffer bufOut = makeBuf(rt, static_cast<size_t>(M) * N);  // fp32 out (CoopMat store)

	auto* aPtr = static_cast<oa::U16*>(bufA.mappedPtr);
	auto* bPtr = static_cast<oa::U16*>(bufB.mappedPtr);
	auto* outPtr = static_cast<oa::F32*>(bufOut.mappedPtr);

	std::mt19937 rng(7);
	std::uniform_real_distribution<oa::F32> dist(-1.0F, 1.0F);
	std::vector<oa::F32> ah(static_cast<size_t>(M) * K), bh(static_cast<size_t>(N) * K);
	for (oa::U32 i = 0; i < M * K; ++i) { oa::U16 p = packBf16(dist(rng)); aPtr[i] = p; ah[i] = unpackBf16(p); }
	for (oa::U32 i = 0; i < N * K; ++i) { oa::U16 p = packBf16(dist(rng)); bPtr[i] = p; bh[i] = unpackBf16(p); }
	std::memset(outPtr, 0, static_cast<size_t>(M) * N * sizeof(oa::F32));

	struct { oa::U32 M, N, K; } push = {M, N, K};
	oavk::Buffer bufs[] = {bufA, bufB, bufOut};
	const oa::U32 gx = (M + tile - 1u) / tile, gy = (N + tile - 1u) / tile;

	(void)oavk::Dispatch::run(
		rt, kernel, bufs, &push, sizeof(push),
		oa::ScalarType::BFloat16, gx, gy, 1u);
	float maxErr = 0.0F, maxAbs = 0.0F;
	for (oa::U32 m = 0; m < M; ++m)
		for (oa::U32 n = 0; n < N; ++n) {
			float s = 0.0F;
			for (oa::U32 k = 0; k < K; ++k) s += ah[m * K + k] * bh[n * K + k];
			maxErr = std::max(maxErr, std::abs(outPtr[m * N + n] - s));
			maxAbs = std::max(maxAbs, std::abs(s));
		}
	if (outNormErr) *outNormErr = maxAbs > 0.0F ? maxErr / maxAbs : 0.0F;

	auto runBatch = [&](oa::U32 n) -> double {
		auto batchRes = oavk::Dispatch::beginBatch(rt);
		if (!batchRes.isOk()) return 0.0;
		oavk::Batch batch = batchRes.getValue();
		auto t0 = std::chrono::high_resolution_clock::now();
		for (oa::U32 i = 0; i < n; ++i)
			(void)oavk::Dispatch::record(
				batch, rt, kernel, bufs, &push, sizeof(push),
				oa::ScalarType::BFloat16, gx, gy, 1u);
		(void)oavk::Dispatch::flush(batch, rt);
		auto t1 = std::chrono::high_resolution_clock::now();
		return std::chrono::duration<double, std::milli>(t1 - t0).count();
	};
	runBatch(10);
	double best = 1e30;
	for (int rep = 0; rep < 3; ++rep) { double ms = runBatch(iters); if (ms > 0.0) best = std::min(best, ms / iters); }

	oa::EngineBindlessAccess::deregisterBuffer(rt, bufA); oa::EngineBindlessAccess::deregisterBuffer(rt, bufB); oa::EngineBindlessAccess::deregisterBuffer(rt, bufOut);
	oa::EngineAllocatorAccess::get(rt).free(bufA); oa::EngineAllocatorAccess::get(rt).free(bufB); oa::EngineAllocatorAccess::get(rt).free(bufOut);
	return (2.0 * M * N * K) / (best * 1e-3) / 1e12;
}

} // namespace

// Native-bf16 (128-bit uint4 staging) vs the fp32-master path. run with OA_TEST_BF16=1
// to engage the DTYPE=1 pipeline variant; skips under the fp32 engine.
TEST(GemmCmSgBf16, NativeBf16Perf) {
	ASSERT_TRUE(vkTestEngineOk());
	oa::Engine& rt = testEngine();
	if (rt.getPrecision() != oa::Precision::BF16) {
		OaLogInfo(oa::LogComponent::Compute, "NativeBf16Perf: engine is FP32 — set OA_TEST_BF16=1 to bench native bf16. Skipping.");
		GTEST_SKIP();
	}
	for (auto [M, N, K] : std::vector<std::array<oa::U32, 3>>{
			{512, 512, 512}, {1024, 1024, 1024}, {2048, 2048, 2048}}) {
		float eSg = 0.0F, eWg = 0.0F;
		double sg = benchBf16(rt, M, N, K, 100, "GemmCmSgBf16", 128u, &eSg);
		double wg = benchBf16(rt, M, N, K, 100, "GemmCmWgBf16", 64u, &eWg);
		EXPECT_LT(eSg, 3e-2F) << "gemmCmSgBf16 (native bf16) " << M << "x" << N << "x" << K << " norm_err=" << eSg;
		EXPECT_LT(eWg, 3e-2F) << "gemmCmWgBf16 (native bf16) " << M << "x" << N << "x" << K << " norm_err=" << eWg;
		OaLogInfo(oa::LogComponent::Compute,
			"NATIVE-BF16 {}x{}x{} : CmSg {:.1f} (err {:.1e})  CmWg {:.1f} (err {:.1e}) TFLOP/s", M, N, K, sg, eSg, wg, eWg);
	}
}

TEST(GemmCmSgBf16, FusedEpilogueCorrectness) {
	ASSERT_TRUE(vkTestEngineOk());
	oa::Engine& rt = testEngine();
	if (!oa::EngineDeviceAccess::get(rt).info.software.hasCooperativeMatrix2) GTEST_SKIP();
	OA_SKIP_IF_BF16_ENGINE(rt);
	for (auto [M, N, K] : std::vector<std::array<oa::U32, 3>>{{128, 128, 128}, {256, 256, 256}, {64, 128, 784}}) {
		for (const char* kernel : {"GemmBiasCmSgBf16", "GemmBiasReluCmSgBf16", "GemmBiasGeluCmSgBf16", "GemmBiasSiluCmSgBf16"}) {
			float e = runAndCheckFused(rt, M, N, K, kernel, 128u);
			EXPECT_LT(e, 3e-2F) << kernel << " " << M << "x" << N << "x" << K << " norm_err=" << e;
			OaLogInfo(oa::LogComponent::Compute, "{} {}x{}x{} norm_err={:.2e}", kernel, M, N, K, e);
		}
	}
}

TEST(GemmCmSgBf16, SiluDualOutputCorrectness) {
	ASSERT_TRUE(vkTestEngineOk());
	oa::Engine& rt = testEngine();
	if (!oa::EngineDeviceAccess::get(rt).info.software.hasCooperativeMatrix2) GTEST_SKIP();
	OA_SKIP_IF_BF16_ENGINE(rt);
	for (auto [M, N, K] : std::vector<std::array<oa::U32, 3>>{{128, 128, 128}, {256, 256, 256}, {64, 128, 784}}) {
		float e = runAndCheckSilu(rt, M, N, K, "GemmSiluCmSgBf16", 128u);
		EXPECT_LT(e, 3e-2F) << "GemmSiluCmSgBf16 " << M << "x" << N << "x" << K << " norm_err=" << e;
		OaLogInfo(oa::LogComponent::Compute, "GemmSiluCmSgBf16 {}x{}x{} norm_err={:.2e}", M, N, K, e);
	}
}

TEST(GemmCmSgBf16, WorkgroupScopeCorrectness) {
	ASSERT_TRUE(vkTestEngineOk());
	oa::Engine& rt = testEngine();
	const bool hasWg = (oa::EngineGemmAccess::capsMask(rt)
		& oa::kCapCoopMat1WorkgroupBf16) != 0;
	OaLogInfo(oa::LogComponent::Compute, "workgroup-scope BF16 CoopMat available: {}", hasWg ? "yes" : "no");
	if (!hasWg) GTEST_SKIP();
	OA_SKIP_IF_BF16_ENGINE(rt);
	for (auto [M, N, K] : std::vector<std::array<oa::U32, 3>>{{128, 128, 128}, {256, 256, 256}, {64, 128, 784}}) {
		float e = runAndCheck(rt, M, N, K, "GemmCmWgBf16", 64u);
		EXPECT_LT(e, 3e-2F) << "GemmCmWgBf16 " << M << "x" << N << "x" << K << " norm_err=" << e;
		OaLogInfo(oa::LogComponent::Compute, "GemmCmWgBf16 {}x{}x{} norm_err={:.2e}", M, N, K, e);
	}
	for (auto [M, N, K] : std::vector<std::array<oa::U32, 3>>{{128, 128, 128}, {256, 256, 256}, {64, 128, 784}}) {
		for (const char* kernel : {"GemmBiasCmWgBf16", "GemmBiasReluCmWgBf16", "GemmBiasGeluCmWgBf16", "GemmBiasSiluCmWgBf16"}) {
			float e = runAndCheckFused(rt, M, N, K, kernel, 64u);
			EXPECT_LT(e, 3e-2F) << kernel << " " << M << "x" << N << "x" << K << " norm_err=" << e;
			OaLogInfo(oa::LogComponent::Compute, "{} {}x{}x{} norm_err={:.2e}", kernel, M, N, K, e);
		}
		float e = runAndCheckSilu(rt, M, N, K, "GemmSiluCmWgBf16", 64u);
		EXPECT_LT(e, 3e-2F) << "GemmSiluCmWgBf16 " << M << "x" << N << "x" << K << " norm_err=" << e;
		OaLogInfo(oa::LogComponent::Compute, "GemmSiluCmWgBf16 {}x{}x{} norm_err={:.2e}", M, N, K, e);
	}
}

TEST(GemmCmSgBf16, WorkgroupScopePerf) {
	ASSERT_TRUE(vkTestEngineOk());
	oa::Engine& rt = testEngine();
	const bool hasWg = (oa::EngineGemmAccess::capsMask(rt)
		& oa::kCapCoopMat1WorkgroupBf16) != 0;
	if (!hasWg) GTEST_SKIP();
	for (auto [M, N, K] : std::vector<std::array<oa::U32, 3>>{
			{64, 128, 784}, {128, 128, 256}, {256, 256, 256}, {384, 1536, 512},
			{512, 512, 512}, {768, 768, 768}, {1024, 1024, 1024}, {2048, 2048, 2048}}) {
		double sub = benchTflopsBatched(rt, M, N, K, 100, "GemmCmSgBf16", 128u);
		double wg  = benchTflopsBatched(rt, M, N, K, 100, "GemmCmWgBf16", 64u);
		OaLogInfo(oa::LogComponent::Compute,
			"WG xover {}x{}x{} : GemmCmSgBf16 {:.1f}  GemmCmWgBf16 {:.1f} TFLOP/s  (WG/sub = {:.0f}%)",
			M, N, K, sub, wg, 100.0 * wg / sub);
	}
}

// Crossover sweep: gemmCmSgBf16 (128x128 tile, 256 threads) vs GemmCmWgBf16
// (64x64 tile, 256 threads). GPU-TIMESTAMPED (not wall) so the result reflects
// real kernel throughput, not per-dispatch CPU overhead.
TEST(GemmCmSgBf16, CrossoverVsCoopMat1) {
	ASSERT_TRUE(vkTestEngineOk());
	oa::Engine& rt = testEngine();
	if (!oa::EngineDeviceAccess::get(rt).info.software.hasCooperativeMatrix) GTEST_SKIP();
	OA_SKIP_IF_BF16_ENGINE(rt);
	// Correctness of GemmCmSgBf16 on aligned shapes (M%16==0, N%16==0).
	// GemmCmSgBf16 uses a 128x128 output tile and 16x16 CoopMat fragments; the
	// direct-to-global store is safe only when each fragment is fully in-bounds.
	// Tail shapes are routed to the fp32 Tiled fallback in production.
	for (auto [M, N, K] : std::vector<std::array<oa::U32, 3>>{{128, 128, 128}, {256, 256, 256}, {64, 128, 784}, {160, 96, 100}}) {
		float e = runAndCheck(rt, M, N, K, "GemmCmSgBf16", 128u);
		EXPECT_LT(e, 3e-2F) << "GemmCmSgBf16 " << M << "x" << N << "x" << K << " norm_err=" << e;
		OaLogInfo(oa::LogComponent::Compute, "GemmCmSgBf16 {}x{}x{} norm_err={:.2e}", M, N, K, e);
	}
	for (auto [M, N, K] : std::vector<std::array<oa::U32, 3>>{
			{64, 128, 784}, {128, 128, 256}, {256, 256, 256}, {384, 1536, 512},
			{512, 512, 512}, {768, 768, 768}, {1024, 1024, 1024}, {2048, 2048, 2048}}) {
		double nu   = benchTflopsBatched(rt, M, N, K, 100, "GemmCmSgBf16", 128u);
		OaLogInfo(oa::LogComponent::Compute,
			"GemmCmSgBf16 {}x{}x{} : {:.1f} TFLOP/s", M, N, K, nu);
	}
}

// ── Regression tests for the SMEM-overflow bug (commit 1398f5e) ──────────────
// The fused epilogue variants (GemmBiasCmSgBf16 etc.) used smOut[BM*BN] = 64KB,
// pushing total SMEM to 100KB — exceeding the 48KB device limit. pipeline
// creation deferred silently; first dispatch hung. These tests guard against
// that class of failure.

// 1. pipeline smoke: verify all fused CmSg/CmWg variant pipelines were created
//    successfully during engine init. A missing pipeline means the shader
//    exceeded device limits (SMEM, register file, etc.) and would hang on
//    first dispatch.
TEST(GemmCmSgBf16, FusedVariantPipelinesExist) {
	ASSERT_TRUE(vkTestEngineOk());
	oa::Engine& rt = testEngine();
	if (!oa::EngineDeviceAccess::get(rt).info.software.hasCooperativeMatrix) GTEST_SKIP();

	static const char* kFusedKernels[] = {
		"GemmBiasCmSgBf16", "GemmBiasReluCmSgBf16", "GemmBiasGeluCmSgBf16", "GemmSiluCmSgBf16", "GemmBiasSiluCmSgBf16",
		"GemmBiasCmWgBf16", "GemmBiasReluCmWgBf16", "GemmBiasGeluCmWgBf16", "GemmSiluCmWgBf16", "GemmBiasSiluCmWgBf16",
	};
	for (const char* name : kFusedKernels) {
		auto& pipe = oa::EnginePipelineAccess::get(rt).getPipeline(name, 0U);
		EXPECT_NE(pipe.pipeline, nullptr)
			<< "pipeline '" << name << "' is null — shader likely exceeds device SMEM/register limits";
	}
}

// 2. Small-N fused correctness: NLP tutorials use shapes like M=1024, N=32, K=32
//    (addLinear routes these to GemmBiasCmSgBf16 when N%16==0). The 128x128 tile
//    means N=32 dispatches with a partial tail tile. verify correctness.
TEST(GemmCmSgBf16, FusedSmallNCorrectness) {
	ASSERT_TRUE(vkTestEngineOk());
	oa::Engine& rt = testEngine();
	if (!oa::EngineDeviceAccess::get(rt).info.software.hasCooperativeMatrix2) GTEST_SKIP();
	OA_SKIP_IF_BF16_ENGINE(rt);

	// NLP-scale shapes: large M, small N (divisible by 16 but < 128 tile).
	for (auto [M, N, K] : std::vector<std::array<oa::U32, 3>>{
			{1024, 32, 32}, {512, 64, 32}, {1024, 32, 64}, {256, 16, 16}}) {
		for (const char* kernel : {"GemmBiasCmSgBf16", "GemmBiasReluCmSgBf16", "GemmBiasGeluCmSgBf16", "GemmBiasSiluCmSgBf16"}) {
			float e = runAndCheckFused(rt, M, N, K, kernel, 128u);
			EXPECT_LT(e, 3e-2F) << kernel << " " << M << "x" << N << "x" << K << " norm_err=" << e;
			OaLogInfo(oa::LogComponent::Compute, "{} small-N {}x{}x{} norm_err={:.2e}", kernel, M, N, K, e);
		}
	}
}

// 3. Wg fused small-N: same regression check for the workgroup-scope variants
//    (64x64 tile, 34KB SMEM — fits, but still verify tail-tile correctness).
TEST(GemmCmSgBf16, WgFusedSmallNCorrectness) {
	ASSERT_TRUE(vkTestEngineOk());
	oa::Engine& rt = testEngine();
	const bool hasWg = (oa::EngineGemmAccess::capsMask(rt)
		& oa::kCapCoopMat1WorkgroupBf16) != 0;
	if (!hasWg) GTEST_SKIP();
	OA_SKIP_IF_BF16_ENGINE(rt);

	for (auto [M, N, K] : std::vector<std::array<oa::U32, 3>>{
			{1024, 32, 32}, {512, 64, 32}, {256, 32, 64}}) {
		for (const char* kernel : {"GemmBiasCmWgBf16", "GemmBiasReluCmWgBf16", "GemmBiasGeluCmWgBf16", "GemmBiasSiluCmWgBf16"}) {
			float e = runAndCheckFused(rt, M, N, K, kernel, 64u);
			EXPECT_LT(e, 3e-2F) << kernel << " " << M << "x" << N << "x" << K << " norm_err=" << e;
			OaLogInfo(oa::LogComponent::Compute, "{} small-N {}x{}x{} norm_err={:.2e}", kernel, M, N, K, e);
		}
	}
}
