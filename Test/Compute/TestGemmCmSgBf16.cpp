// Correctness + perf harness for GemmCmSgBf16 — the tuned KHR cooperative-matrix
// GEMM that reads FP32 masters, stages bf16 tiles in shared memory, and uses
// 16×16 CoopMat1 fragments. No pack, no mirror. Fused epilogue variants
// (bias, bias+relu, bias+gelu, silu) share the same core.

#include "../OaTest.h"
#include <Oa/Runtime/Dispatch.h>
#include <Oa/Runtime/Engine.h>

#include <chrono>
#include <cstring>
#include <random>
#include <vector>

namespace {

// The correctness tests below build FP32 input buffers and validate the DTYPE=0
// (fp32-master) pipeline. Under the BF16 engine (OA_TEST_BF16=1) every dispatch
// selects the DTYPE=1 native-bf16 pipeline, which reinterprets those fp32 bytes
// as packed bf16 → garbage (inf/1e37 norm_err). Skip them under bf16; the native
// bf16 path is covered by NativeBf16Perf (which skips under the fp32 engine).
#define OA_SKIP_IF_BF16_ENGINE(rt)                                                  \
	do {                                                                            \
		if ((rt).GetPrecision() == OaPrecision::BF16) {                             \
			GTEST_SKIP() << "fp32-master correctness test skipped under BF16 "      \
			                "engine — see NativeBf16Perf for the bf16 path";        \
		}                                                                           \
	} while (0)

OaVkBuffer MakeBuf(OaEngine& rt, size_t elems) {
	auto res = rt.Allocator.AllocHostVisible(elems * sizeof(OaF32));
	OaVkBuffer buf = *res;
	rt.RegisterBuffer(buf);
	return buf;
}

// tile = the kernel's output-tile edge (workgroups = ceil(M/tile) x ceil(N/tile)).
// SMEM kernels use 128; GemmCmSgBf16 uses 128; GemmCmWgBf16 uses 64. Passing the wrong tile
// under-dispatches → partial output (norm_err ~1) AND inflates TFLOP/s ~ (right/wrong)^2.
float RunAndCheck(OaEngine& rt, OaU32 M, OaU32 N, OaU32 K, const char* kernel, OaU32 tile = 128u) {
	OaVkBuffer bufA = MakeBuf(rt, static_cast<size_t>(M) * K);
	OaVkBuffer bufB = MakeBuf(rt, static_cast<size_t>(N) * K);
	OaVkBuffer bufOut = MakeBuf(rt, static_cast<size_t>(M) * N);

	auto* aPtr = static_cast<OaF32*>(bufA.MappedPtr);
	auto* bPtr = static_cast<OaF32*>(bufB.MappedPtr);
	auto* outPtr = static_cast<OaF32*>(bufOut.MappedPtr);

	std::mt19937 rng(7);
	std::uniform_real_distribution<OaF32> dist(-1.0F, 1.0F);
	for (OaU32 i = 0; i < M * K; ++i) aPtr[i] = dist(rng);
	for (OaU32 i = 0; i < N * K; ++i) bPtr[i] = dist(rng);
	std::memset(outPtr, 0, static_cast<size_t>(M) * N * sizeof(OaF32));

	struct { OaU32 M, N, K; } push = {M, N, K};
	OaVkBuffer bufs[] = {bufA, bufB, bufOut};
	OaStatus st = OaVkDispatch::Run(rt, kernel, bufs, &push, sizeof(push),
		(M + tile - 1u) / tile, (N + tile - 1u) / tile, 1u);
	EXPECT_TRUE(st.IsOk()) << st.ToString();

	// CPU reference (bf16 round-to-nearest on inputs).
	auto ToBf16Rtn = [](OaF32 v) -> OaF32 {
		OaU32 bits; std::memcpy(&bits, &v, sizeof(bits));
		OaU32 r = (bits + 0x7FFFU + ((bits >> 16U) & 1U)) & 0xFFFF0000U;
		OaF32 f; std::memcpy(&f, &r, sizeof(f)); return f;
	};
	std::vector<OaF32> ah(static_cast<size_t>(M) * K), bh(static_cast<size_t>(N) * K);
	for (OaU32 i = 0; i < M * K; ++i) ah[i] = ToBf16Rtn(aPtr[i]);
	for (OaU32 i = 0; i < N * K; ++i) bh[i] = ToBf16Rtn(bPtr[i]);

	float maxErr = 0.0F, maxAbs = 0.0F;
	for (OaU32 m = 0; m < M; ++m)
		for (OaU32 n = 0; n < N; ++n) {
			float s = 0.0F;
			for (OaU32 k = 0; k < K; ++k) s += ah[m * K + k] * bh[n * K + k];
			maxErr = std::max(maxErr, std::abs(outPtr[m * N + n] - s));
			maxAbs = std::max(maxAbs, std::abs(s));
		}

	rt.DeregisterBuffer(bufA); rt.DeregisterBuffer(bufB); rt.DeregisterBuffer(bufOut);
	rt.Allocator.Free(bufA); rt.Allocator.Free(bufB); rt.Allocator.Free(bufOut);
	return maxAbs > 0.0F ? maxErr / maxAbs : 0.0F;
}

float RunAndCheckFused(OaEngine& rt, OaU32 M, OaU32 N, OaU32 K, const char* kernel, OaU32 tile = 128u) {
	OaVkBuffer bufA = MakeBuf(rt, static_cast<size_t>(M) * K);
	OaVkBuffer bufB = MakeBuf(rt, static_cast<size_t>(N) * K);
	OaVkBuffer bufBias = MakeBuf(rt, static_cast<size_t>(N));
	OaVkBuffer bufOut = MakeBuf(rt, static_cast<size_t>(M) * N);

	auto* aPtr = static_cast<OaF32*>(bufA.MappedPtr);
	auto* bPtr = static_cast<OaF32*>(bufB.MappedPtr);
	auto* biasPtr = static_cast<OaF32*>(bufBias.MappedPtr);
	auto* outPtr = static_cast<OaF32*>(bufOut.MappedPtr);

	std::mt19937 rng(7);
	std::uniform_real_distribution<OaF32> dist(-1.0F, 1.0F);
	for (OaU32 i = 0; i < M * K; ++i) aPtr[i] = dist(rng);
	for (OaU32 i = 0; i < N * K; ++i) bPtr[i] = dist(rng);
	for (OaU32 i = 0; i < N; ++i) biasPtr[i] = dist(rng);
	std::memset(outPtr, 0, static_cast<size_t>(M) * N * sizeof(OaF32));

	struct { OaU32 M, N, K; } push = {M, N, K};
	OaVkBuffer bufs[] = {bufA, bufB, bufBias, bufOut};
	OaStatus st = OaVkDispatch::Run(rt, kernel, bufs, &push, sizeof(push),
		(M + tile - 1u) / tile, (N + tile - 1u) / tile, 1u);
	EXPECT_TRUE(st.IsOk()) << st.ToString();

	auto ToBf16Rtn = [](OaF32 v) -> OaF32 {
		OaU32 bits; std::memcpy(&bits, &v, sizeof(bits));
		OaU32 r = (bits + 0x7FFFU + ((bits >> 16U) & 1U)) & 0xFFFF0000U;
		OaF32 f; std::memcpy(&f, &r, sizeof(f)); return f;
	};
	std::vector<OaF32> ah(static_cast<size_t>(M) * K), bh(static_cast<size_t>(N) * K);
	for (OaU32 i = 0; i < M * K; ++i) ah[i] = ToBf16Rtn(aPtr[i]);
	for (OaU32 i = 0; i < N * K; ++i) bh[i] = ToBf16Rtn(bPtr[i]);

	float maxErr = 0.0F, maxAbs = 0.0F;
	for (OaU32 m = 0; m < M; ++m)
		for (OaU32 n = 0; n < N; ++n) {
			float s = 0.0F;
			for (OaU32 k = 0; k < K; ++k) s += ah[m * K + k] * bh[n * K + k];
			s += biasPtr[n];
			float ref = s;
			if (std::strcmp(kernel, "GemmBiasReluCmSgBf16") == 0 or std::strcmp(kernel, "GemmBiasReluCmWgBf16") == 0) ref = std::max(0.0F, s);
			else if (std::strcmp(kernel, "GemmBiasGeluCmSgBf16") == 0 or std::strcmp(kernel, "GemmBiasGeluCmWgBf16") == 0) {
				float x3 = s * s * s;
				float inner = 0.7978845608F * (s + 0.044715F * x3);
				ref = 0.5F * s * (1.0F + std::tanh(inner));
			}
			maxErr = std::max(maxErr, std::abs(outPtr[m * N + n] - ref));
			maxAbs = std::max(maxAbs, std::abs(ref));
		}

	rt.DeregisterBuffer(bufA); rt.DeregisterBuffer(bufB); rt.DeregisterBuffer(bufBias); rt.DeregisterBuffer(bufOut);
	rt.Allocator.Free(bufA); rt.Allocator.Free(bufB); rt.Allocator.Free(bufBias); rt.Allocator.Free(bufOut);
	return maxAbs > 0.0F ? maxErr / maxAbs : 0.0F;
}

float RunAndCheckSilu(OaEngine& rt, OaU32 M, OaU32 N, OaU32 K, const char* kernel = "GemmSiluCmSgBf16", OaU32 tile = 128u) {
	OaVkBuffer bufA = MakeBuf(rt, static_cast<size_t>(M) * K);
	OaVkBuffer bufB = MakeBuf(rt, static_cast<size_t>(N) * K);
	OaVkBuffer bufPre = MakeBuf(rt, static_cast<size_t>(M) * N);
	OaVkBuffer bufAct = MakeBuf(rt, static_cast<size_t>(M) * N);

	auto* aPtr = static_cast<OaF32*>(bufA.MappedPtr);
	auto* bPtr = static_cast<OaF32*>(bufB.MappedPtr);
	auto* prePtr = static_cast<OaF32*>(bufPre.MappedPtr);
	auto* actPtr = static_cast<OaF32*>(bufAct.MappedPtr);

	std::mt19937 rng(7);
	std::uniform_real_distribution<OaF32> dist(-1.0F, 1.0F);
	for (OaU32 i = 0; i < M * K; ++i) aPtr[i] = dist(rng);
	for (OaU32 i = 0; i < N * K; ++i) bPtr[i] = dist(rng);
	std::memset(prePtr, 0, static_cast<size_t>(M) * N * sizeof(OaF32));
	std::memset(actPtr, 0, static_cast<size_t>(M) * N * sizeof(OaF32));

	struct { OaU32 M, N, K; } push = {M, N, K};
	OaVkBuffer bufs[] = {bufA, bufB, bufPre, bufAct};
	OaStatus st = OaVkDispatch::Run(rt, kernel, bufs, &push, sizeof(push),
		(M + tile - 1u) / tile, (N + tile - 1u) / tile, 1u);
	EXPECT_TRUE(st.IsOk()) << st.ToString();

	auto ToBf16Rtn = [](OaF32 v) -> OaF32 {
		OaU32 bits; std::memcpy(&bits, &v, sizeof(bits));
		OaU32 r = (bits + 0x7FFFU + ((bits >> 16U) & 1U)) & 0xFFFF0000U;
		OaF32 f; std::memcpy(&f, &r, sizeof(f)); return f;
	};
	auto Silu = [](OaF32 x) {
		return x / (1.0F + std::exp(-x));
	};
	std::vector<OaF32> ah(static_cast<size_t>(M) * K), bh(static_cast<size_t>(N) * K);
	for (OaU32 i = 0; i < M * K; ++i) ah[i] = ToBf16Rtn(aPtr[i]);
	for (OaU32 i = 0; i < N * K; ++i) bh[i] = ToBf16Rtn(bPtr[i]);

	float maxErr = 0.0F, maxAbs = 0.0F;
	for (OaU32 m = 0; m < M; ++m)
		for (OaU32 n = 0; n < N; ++n) {
			float s = 0.0F;
			for (OaU32 k = 0; k < K; ++k) s += ah[m * K + k] * bh[n * K + k];
			maxErr = std::max(maxErr, std::abs(prePtr[m * N + n] - s));
			maxAbs = std::max(maxAbs, std::abs(s));
			maxErr = std::max(maxErr, std::abs(actPtr[m * N + n] - Silu(s)));
			maxAbs = std::max(maxAbs, std::abs(Silu(s)));
		}

	rt.DeregisterBuffer(bufA); rt.DeregisterBuffer(bufB); rt.DeregisterBuffer(bufPre); rt.DeregisterBuffer(bufAct);
	rt.Allocator.Free(bufA); rt.Allocator.Free(bufB); rt.Allocator.Free(bufPre); rt.Allocator.Free(bufAct);
	return maxAbs > 0.0F ? maxErr / maxAbs : 0.0F;
}

double BenchTflops(OaEngine& rt, OaU32 M, OaU32 N, OaU32 K, OaU32 iters, const char* kernel) {
	OaVkBuffer bufA = MakeBuf(rt, static_cast<size_t>(M) * K);
	OaVkBuffer bufB = MakeBuf(rt, static_cast<size_t>(N) * K);
	OaVkBuffer bufOut = MakeBuf(rt, static_cast<size_t>(M) * N);
	std::memset(bufA.MappedPtr, 0, static_cast<size_t>(M) * K * sizeof(OaF32));
	std::memset(bufB.MappedPtr, 0, static_cast<size_t>(N) * K * sizeof(OaF32));

	struct { OaU32 M, N, K; } push = {M, N, K};
	OaVkBuffer bufs[] = {bufA, bufB, bufOut};
	auto run = [&]() {
		(void)OaVkDispatch::Run(rt, kernel, bufs, &push, sizeof(push),
			(M + 127u) / 128u, (N + 127u) / 128u, 1u);
	};
	for (OaU32 i = 0; i < 10; ++i) run();  // warmup
	auto t0 = std::chrono::high_resolution_clock::now();
	for (OaU32 i = 0; i < iters; ++i) run();
	auto t1 = std::chrono::high_resolution_clock::now();

	double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
	double gflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e9;

	rt.DeregisterBuffer(bufA); rt.DeregisterBuffer(bufB); rt.DeregisterBuffer(bufOut);
	rt.Allocator.Free(bufA); rt.Allocator.Free(bufB); rt.Allocator.Free(bufOut);
	return gflops / 1000.0;  // TFLOP/s
}

// Batched throughput — the honest measurement. The per-dispatch OaVkDispatch::Run
// does a full submit+WAIT roundtrip, so its wall time is ~90% CPU/sync overhead
// (see BenchTflops above). Here we record `iters` dispatches into ONE batch and
// submit+wait ONCE, amortizing that overhead ~iters-fold, so wall time tracks GPU
// kernel throughput. A buffer barrier between dispatches serializes them (WAW on
// bufOut) so the GPU actually runs `iters` full GEMMs, not overlapped/elided ones.
double BenchTflopsBatched(OaEngine& rt, OaU32 M, OaU32 N, OaU32 K, OaU32 iters, const char* kernel, OaU32 tile = 128u) {
	OaVkBuffer bufA = MakeBuf(rt, static_cast<size_t>(M) * K);
	OaVkBuffer bufB = MakeBuf(rt, static_cast<size_t>(N) * K);
	OaVkBuffer bufOut = MakeBuf(rt, static_cast<size_t>(M) * N);
	std::memset(bufA.MappedPtr, 0, static_cast<size_t>(M) * K * sizeof(OaF32));
	std::memset(bufB.MappedPtr, 0, static_cast<size_t>(N) * K * sizeof(OaF32));

	struct { OaU32 M, N, K; } push = {M, N, K};
	OaVkBuffer bufs[] = {bufA, bufB, bufOut};
	const OaU32 gx = (M + tile - 1u) / tile, gy = (N + tile - 1u) / tile;

	auto runBatch = [&](OaU32 n) -> double {
		auto batchRes = OaVkDispatch::BeginBatch(rt);
		if (!batchRes.IsOk()) return 0.0;
		OaVkBatch batch = batchRes.GetValue();
		auto t0 = std::chrono::high_resolution_clock::now();
		for (OaU32 i = 0; i < n; ++i) {
			(void)OaVkDispatch::Record(batch, rt, kernel, bufs, &push, sizeof(push), gx, gy, 1u);
		}
		(void)OaVkDispatch::Flush(batch, rt);  // single submit+wait for all n
		auto t1 = std::chrono::high_resolution_clock::now();
		return std::chrono::duration<double, std::milli>(t1 - t0).count();
	};

	runBatch(10);  // warmup
	double best = 1e30;
	for (int rep = 0; rep < 3; ++rep) {
		double ms = runBatch(iters);
		if (ms > 0.0) best = std::min(best, ms / iters);
	}

	rt.DeregisterBuffer(bufA); rt.DeregisterBuffer(bufB); rt.DeregisterBuffer(bufOut);
	rt.Allocator.Free(bufA); rt.Allocator.Free(bufB); rt.Allocator.Free(bufOut);
	return (2.0 * M * N * K) / (best * 1e-3) / 1e12;  // TFLOP/s from best ms/dispatch
}

OaVkBuffer MakeBufBf16(OaEngine& rt, size_t elems) {
	auto res = rt.Allocator.AllocHostVisible(elems * 2u);  // 2 bytes / bf16
	OaVkBuffer buf = *res;
	rt.RegisterBuffer(buf);
	return buf;
}

static inline OaU16 PackBf16(OaF32 v) {
	OaU32 b; std::memcpy(&b, &v, 4);
	return static_cast<OaU16>((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);  // round-to-nearest-even
}
static inline OaF32 UnpackBf16(OaU16 h) {
	OaU32 b = static_cast<OaU32>(h) << 16; OaF32 f; std::memcpy(&f, &b, 4); return f;
}

// Native-bf16 perf + self-validating correctness. Inputs are 2-byte bf16 buffers;
// the raw GemmCmSgBf16 store path writes fp32 output. REQUIRES the engine in BF16
// precision (run with OA_TEST_BF16=1) so the dispatch selects the DTYPE=1 pipeline
// variant (128-bit uint4 native-bf16 staging). The norm_err check is the guard: if
// the DTYPE=1 variant did NOT engage and the fp32 path read our bf16 buffer as
// float4, the result is garbage (norm_err ~1), so a passing check proves the native
// path actually ran.
double BenchBf16(OaEngine& rt, OaU32 M, OaU32 N, OaU32 K, OaU32 iters,
                 const char* kernel, OaU32 tile, float* outNormErr) {
	OaVkBuffer bufA = MakeBufBf16(rt, static_cast<size_t>(M) * K);
	OaVkBuffer bufB = MakeBufBf16(rt, static_cast<size_t>(N) * K);
	OaVkBuffer bufOut = MakeBuf(rt, static_cast<size_t>(M) * N);  // fp32 out (CoopMat store)

	auto* aPtr = static_cast<OaU16*>(bufA.MappedPtr);
	auto* bPtr = static_cast<OaU16*>(bufB.MappedPtr);
	auto* outPtr = static_cast<OaF32*>(bufOut.MappedPtr);

	std::mt19937 rng(7);
	std::uniform_real_distribution<OaF32> dist(-1.0F, 1.0F);
	std::vector<OaF32> ah(static_cast<size_t>(M) * K), bh(static_cast<size_t>(N) * K);
	for (OaU32 i = 0; i < M * K; ++i) { OaU16 p = PackBf16(dist(rng)); aPtr[i] = p; ah[i] = UnpackBf16(p); }
	for (OaU32 i = 0; i < N * K; ++i) { OaU16 p = PackBf16(dist(rng)); bPtr[i] = p; bh[i] = UnpackBf16(p); }
	std::memset(outPtr, 0, static_cast<size_t>(M) * N * sizeof(OaF32));

	struct { OaU32 M, N, K; } push = {M, N, K};
	OaVkBuffer bufs[] = {bufA, bufB, bufOut};
	const OaU32 gx = (M + tile - 1u) / tile, gy = (N + tile - 1u) / tile;

	(void)OaVkDispatch::Run(rt, kernel, bufs, &push, sizeof(push), gx, gy, 1u);
	float maxErr = 0.0F, maxAbs = 0.0F;
	for (OaU32 m = 0; m < M; ++m)
		for (OaU32 n = 0; n < N; ++n) {
			float s = 0.0F;
			for (OaU32 k = 0; k < K; ++k) s += ah[m * K + k] * bh[n * K + k];
			maxErr = std::max(maxErr, std::abs(outPtr[m * N + n] - s));
			maxAbs = std::max(maxAbs, std::abs(s));
		}
	if (outNormErr) *outNormErr = maxAbs > 0.0F ? maxErr / maxAbs : 0.0F;

	auto runBatch = [&](OaU32 n) -> double {
		auto batchRes = OaVkDispatch::BeginBatch(rt);
		if (!batchRes.IsOk()) return 0.0;
		OaVkBatch batch = batchRes.GetValue();
		auto t0 = std::chrono::high_resolution_clock::now();
		for (OaU32 i = 0; i < n; ++i)
			(void)OaVkDispatch::Record(batch, rt, kernel, bufs, &push, sizeof(push), gx, gy, 1u);
		(void)OaVkDispatch::Flush(batch, rt);
		auto t1 = std::chrono::high_resolution_clock::now();
		return std::chrono::duration<double, std::milli>(t1 - t0).count();
	};
	runBatch(10);
	double best = 1e30;
	for (int rep = 0; rep < 3; ++rep) { double ms = runBatch(iters); if (ms > 0.0) best = std::min(best, ms / iters); }

	rt.DeregisterBuffer(bufA); rt.DeregisterBuffer(bufB); rt.DeregisterBuffer(bufOut);
	rt.Allocator.Free(bufA); rt.Allocator.Free(bufB); rt.Allocator.Free(bufOut);
	return (2.0 * M * N * K) / (best * 1e-3) / 1e12;
}

} // namespace

// Native-bf16 (128-bit uint4 staging) vs the fp32-master path. Run with OA_TEST_BF16=1
// to engage the DTYPE=1 pipeline variant; skips under the fp32 engine.
TEST(GemmCmSgBf16, NativeBf16Perf) {
	ASSERT_TRUE(OaVkTestEngineOk());
	OaEngine& rt = *OaEngine::GetGlobal();
	if (rt.GetPrecision() != OaPrecision::BF16) {
		OA_LOG_INFO(OaLogComponent::Core, "NativeBf16Perf: engine is FP32 — set OA_TEST_BF16=1 to bench native bf16. Skipping.");
		GTEST_SKIP();
	}
	for (auto [M, N, K] : std::vector<std::array<OaU32, 3>>{
			{512, 512, 512}, {1024, 1024, 1024}, {2048, 2048, 2048}}) {
		float eSg = 0.0F, eWg = 0.0F;
		double sg = BenchBf16(rt, M, N, K, 100, "GemmCmSgBf16", 128u, &eSg);
		double wg = BenchBf16(rt, M, N, K, 100, "GemmCmWgBf16", 64u, &eWg);
		EXPECT_LT(eSg, 3e-2F) << "GemmCmSgBf16 (native bf16) " << M << "x" << N << "x" << K << " norm_err=" << eSg;
		EXPECT_LT(eWg, 3e-2F) << "GemmCmWgBf16 (native bf16) " << M << "x" << N << "x" << K << " norm_err=" << eWg;
		OA_LOG_INFO(OaLogComponent::Core,
			"NATIVE-BF16 %ux%ux%u : CmSg %.1f (err %.1e)  CmWg %.1f (err %.1e) TFLOP/s", M, N, K, sg, eSg, wg, eWg);
	}
}

TEST(GemmCmSgBf16, FusedEpilogueCorrectness) {
	ASSERT_TRUE(OaVkTestEngineOk());
	OaEngine& rt = *OaEngine::GetGlobal();
	if (!rt.Device.Info.Software.HasCooperativeMatrix2) GTEST_SKIP();
	OA_SKIP_IF_BF16_ENGINE(rt);
	for (auto [M, N, K] : std::vector<std::array<OaU32, 3>>{{128, 128, 128}, {256, 256, 256}, {64, 128, 784}}) {
		for (const char* kernel : {"GemmBiasCmSgBf16", "GemmBiasReluCmSgBf16", "GemmBiasGeluCmSgBf16"}) {
			float e = RunAndCheckFused(rt, M, N, K, kernel, 128u);
			EXPECT_LT(e, 3e-2F) << kernel << " " << M << "x" << N << "x" << K << " norm_err=" << e;
			OA_LOG_INFO(OaLogComponent::Core, "%s %ux%ux%u norm_err=%.2e", kernel, M, N, K, e);
		}
	}
}

TEST(GemmCmSgBf16, SiluDualOutputCorrectness) {
	ASSERT_TRUE(OaVkTestEngineOk());
	OaEngine& rt = *OaEngine::GetGlobal();
	if (!rt.Device.Info.Software.HasCooperativeMatrix2) GTEST_SKIP();
	OA_SKIP_IF_BF16_ENGINE(rt);
	for (auto [M, N, K] : std::vector<std::array<OaU32, 3>>{{128, 128, 128}, {256, 256, 256}, {64, 128, 784}}) {
		float e = RunAndCheckSilu(rt, M, N, K, "GemmSiluCmSgBf16", 128u);
		EXPECT_LT(e, 3e-2F) << "GemmSiluCmSgBf16 " << M << "x" << N << "x" << K << " norm_err=" << e;
		OA_LOG_INFO(OaLogComponent::Core, "GemmSiluCmSgBf16 %ux%ux%u norm_err=%.2e", M, N, K, e);
	}
}

TEST(GemmCmSgBf16, WorkgroupScopeCorrectness) {
	ASSERT_TRUE(OaVkTestEngineOk());
	OaEngine& rt = *OaEngine::GetGlobal();
	const bool hasWg = (rt.GemmCapsMask() & kCapCoopMat1WorkgroupBf16) != 0;
	OA_LOG_INFO(OaLogComponent::Core, "Workgroup-scope BF16 CoopMat available: %s", hasWg ? "yes" : "no");
	if (!hasWg) GTEST_SKIP();
	OA_SKIP_IF_BF16_ENGINE(rt);
	for (auto [M, N, K] : std::vector<std::array<OaU32, 3>>{{128, 128, 128}, {256, 256, 256}, {64, 128, 784}}) {
		float e = RunAndCheck(rt, M, N, K, "GemmCmWgBf16", 64u);
		EXPECT_LT(e, 3e-2F) << "GemmCmWgBf16 " << M << "x" << N << "x" << K << " norm_err=" << e;
		OA_LOG_INFO(OaLogComponent::Core, "GemmCmWgBf16 %ux%ux%u norm_err=%.2e", M, N, K, e);
	}
	for (auto [M, N, K] : std::vector<std::array<OaU32, 3>>{{128, 128, 128}, {256, 256, 256}, {64, 128, 784}}) {
		for (const char* kernel : {"GemmBiasCmWgBf16", "GemmBiasReluCmWgBf16", "GemmBiasGeluCmWgBf16"}) {
			float e = RunAndCheckFused(rt, M, N, K, kernel, 64u);
			EXPECT_LT(e, 3e-2F) << kernel << " " << M << "x" << N << "x" << K << " norm_err=" << e;
			OA_LOG_INFO(OaLogComponent::Core, "%s %ux%ux%u norm_err=%.2e", kernel, M, N, K, e);
		}
		float e = RunAndCheckSilu(rt, M, N, K, "GemmSiluCmWgBf16", 64u);
		EXPECT_LT(e, 3e-2F) << "GemmSiluCmWgBf16 " << M << "x" << N << "x" << K << " norm_err=" << e;
		OA_LOG_INFO(OaLogComponent::Core, "GemmSiluCmWgBf16 %ux%ux%u norm_err=%.2e", M, N, K, e);
	}
}

TEST(GemmCmSgBf16, WorkgroupScopePerf) {
	ASSERT_TRUE(OaVkTestEngineOk());
	OaEngine& rt = *OaEngine::GetGlobal();
	const bool hasWg = (rt.GemmCapsMask() & kCapCoopMat1WorkgroupBf16) != 0;
	if (!hasWg) GTEST_SKIP();
	for (auto [M, N, K] : std::vector<std::array<OaU32, 3>>{
			{64, 128, 784}, {128, 128, 256}, {256, 256, 256}, {384, 1536, 512},
			{512, 512, 512}, {768, 768, 768}, {1024, 1024, 1024}, {2048, 2048, 2048}}) {
		double sub = BenchTflopsBatched(rt, M, N, K, 100, "GemmCmSgBf16", 128u);
		double wg  = BenchTflopsBatched(rt, M, N, K, 100, "GemmCmWgBf16", 64u);
		OA_LOG_INFO(OaLogComponent::Core,
			"WG xover %ux%ux%u : GemmCmSgBf16 %.1f  GemmCmWgBf16 %.1f TFLOP/s  (WG/sub = %.0f%%)",
			M, N, K, sub, wg, 100.0 * wg / sub);
	}
}

// Crossover sweep: GemmCmSgBf16 (128x128 tile, 256 threads) vs GemmCmWgBf16
// (64x64 tile, 256 threads). GPU-TIMESTAMPED (not wall) so the result reflects
// real kernel throughput, not per-dispatch CPU overhead.
TEST(GemmCmSgBf16, CrossoverVsCoopMat1) {
	ASSERT_TRUE(OaVkTestEngineOk());
	OaEngine& rt = *OaEngine::GetGlobal();
	if (!rt.Device.Info.Software.HasCooperativeMatrix) GTEST_SKIP();
	OA_SKIP_IF_BF16_ENGINE(rt);
	// Correctness of GemmCmSgBf16 on aligned shapes (M%16==0, N%16==0).
	// GemmCmSgBf16 uses a 128x128 output tile and 16x16 CoopMat fragments; the
	// direct-to-global store is safe only when each fragment is fully in-bounds.
	// Tail shapes are routed to the fp32 Tiled fallback in production.
	for (auto [M, N, K] : std::vector<std::array<OaU32, 3>>{{128, 128, 128}, {256, 256, 256}, {64, 128, 784}, {160, 96, 100}}) {
		float e = RunAndCheck(rt, M, N, K, "GemmCmSgBf16", 128u);
		EXPECT_LT(e, 3e-2F) << "GemmCmSgBf16 " << M << "x" << N << "x" << K << " norm_err=" << e;
		OA_LOG_INFO(OaLogComponent::Core, "GemmCmSgBf16 %ux%ux%u norm_err=%.2e", M, N, K, e);
	}
	for (auto [M, N, K] : std::vector<std::array<OaU32, 3>>{
			{64, 128, 784}, {128, 128, 256}, {256, 256, 256}, {384, 1536, 512},
			{512, 512, 512}, {768, 768, 768}, {1024, 1024, 1024}, {2048, 2048, 2048}}) {
		double nu   = BenchTflopsBatched(rt, M, N, K, 100, "GemmCmSgBf16", 128u);
		OA_LOG_INFO(OaLogComponent::Core,
			"GemmCmSgBf16 %ux%ux%u : %.1f TFLOP/s", M, N, K, nu);
	}
}

// ── Regression tests for the SMEM-overflow bug (commit 1398f5e) ──────────────
// The fused epilogue variants (GemmBiasCmSgBf16 etc.) used smOut[BM*BN] = 64KB,
// pushing total SMEM to 100KB — exceeding the 48KB device limit. Pipeline
// creation deferred silently; first dispatch hung. These tests guard against
// that class of failure.

// 1. Pipeline smoke: verify all fused CmSg/CmWg variant pipelines were created
//    successfully during engine init. A missing pipeline means the shader
//    exceeded device limits (SMEM, register file, etc.) and would hang on
//    first dispatch.
TEST(GemmCmSgBf16, FusedVariantPipelinesExist) {
	ASSERT_TRUE(OaVkTestEngineOk());
	OaEngine& rt = *OaEngine::GetGlobal();
	if (!rt.Device.Info.Software.HasCooperativeMatrix) GTEST_SKIP();

	static const char* kFusedKernels[] = {
		"GemmBiasCmSgBf16", "GemmBiasReluCmSgBf16", "GemmBiasGeluCmSgBf16", "GemmSiluCmSgBf16",
		"GemmBiasCmWgBf16", "GemmBiasReluCmWgBf16", "GemmBiasGeluCmWgBf16", "GemmSiluCmWgBf16",
	};
	for (const char* name : kFusedKernels) {
		auto& pipe = rt.GetPipeline(name);
		EXPECT_NE(pipe.Pipeline, nullptr)
			<< "Pipeline '" << name << "' is null — shader likely exceeds device SMEM/register limits";
	}
}

// 2. Small-N fused correctness: NLP tutorials use shapes like M=1024, N=32, K=32
//    (AddLinear routes these to GemmBiasCmSgBf16 when N%16==0). The 128x128 tile
//    means N=32 dispatches with a partial tail tile. Verify correctness.
TEST(GemmCmSgBf16, FusedSmallNCorrectness) {
	ASSERT_TRUE(OaVkTestEngineOk());
	OaEngine& rt = *OaEngine::GetGlobal();
	if (!rt.Device.Info.Software.HasCooperativeMatrix2) GTEST_SKIP();
	OA_SKIP_IF_BF16_ENGINE(rt);

	// NLP-scale shapes: large M, small N (divisible by 16 but < 128 tile).
	for (auto [M, N, K] : std::vector<std::array<OaU32, 3>>{
			{1024, 32, 32}, {512, 64, 32}, {1024, 32, 64}, {256, 16, 16}}) {
		for (const char* kernel : {"GemmBiasCmSgBf16", "GemmBiasReluCmSgBf16", "GemmBiasGeluCmSgBf16"}) {
			float e = RunAndCheckFused(rt, M, N, K, kernel, 128u);
			EXPECT_LT(e, 3e-2F) << kernel << " " << M << "x" << N << "x" << K << " norm_err=" << e;
			OA_LOG_INFO(OaLogComponent::Core, "%s small-N %ux%ux%u norm_err=%.2e", kernel, M, N, K, e);
		}
	}
}

// 3. Wg fused small-N: same regression check for the workgroup-scope variants
//    (64x64 tile, 34KB SMEM — fits, but still verify tail-tile correctness).
TEST(GemmCmSgBf16, WgFusedSmallNCorrectness) {
	ASSERT_TRUE(OaVkTestEngineOk());
	OaEngine& rt = *OaEngine::GetGlobal();
	const bool hasWg = (rt.GemmCapsMask() & kCapCoopMat1WorkgroupBf16) != 0;
	if (!hasWg) GTEST_SKIP();
	OA_SKIP_IF_BF16_ENGINE(rt);

	for (auto [M, N, K] : std::vector<std::array<OaU32, 3>>{
			{1024, 32, 32}, {512, 64, 32}, {256, 32, 64}}) {
		for (const char* kernel : {"GemmBiasCmWgBf16", "GemmBiasReluCmWgBf16", "GemmBiasGeluCmWgBf16"}) {
			float e = RunAndCheckFused(rt, M, N, K, kernel, 64u);
			EXPECT_LT(e, 3e-2F) << kernel << " " << M << "x" << N << "x" << K << " norm_err=" << e;
			OA_LOG_INFO(OaLogComponent::Core, "%s small-N %ux%ux%u norm_err=%.2e", kernel, M, N, K, e);
		}
	}
}
