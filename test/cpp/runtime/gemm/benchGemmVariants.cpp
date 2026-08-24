// Controlled raw-GEMM variant benchmark.
//
// This is deliberately not registered with ctest. It uses GPU timestamps on
// the production oa::FnMatrix::matMulNt path and exists for kernel-selection
// experiments, not pass/fail CI. run multiple fresh processes before changing
// routing policy; integrated-GPU clocks and thermals can move during a sweep.

#include "../../oaTest.h"

#include <oa/core/fnMatrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/timer.h>
#include <oa/runtime/timerAccess.h>
#include <oa/runtime/dispatch.h>
#include <oa/runtime/gemm/dispatch.h>
#include <oa/runtime/gemm/router.h>
#include <oa/runtime/gemm/tuner.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

oa::U32 envIterations(const char* inName, oa::U32 inDefault) {
	const char* text = std::getenv(inName);
	if (text == nullptr or *text == '\0') return inDefault;
	char* end = nullptr;
	const unsigned long value = std::strtoul(text, &end, 10);
	if (end == text or *end != '\0' or value == 0UL or value > 1000UL) return inDefault;
	return static_cast<oa::U32>(value);
}

oa::F64 median(std::vector<oa::F64> inSamples) {
	std::sort(inSamples.begin(), inSamples.end());
	const size_t middle = inSamples.size() / 2U;
	return (inSamples.size() & 1U) != 0U
		? inSamples[middle]
		: 0.5 * (inSamples[middle - 1U] + inSamples[middle]);
}

oa::Result<oa::F64> measurePlan(
	oa::Engine& inEngine,
	oa::Timer& inTimer,
	const oa::MatmulPlan& inPlan,
	const oa::MatmulProblem& inProblem,
	oa::Span<oavk::Buffer> inBuffers,
	oa::U32 inDispatches)
{
	auto batchResult = oavk::Dispatch::beginBatch(inEngine);
	if (not batchResult.isOk()) return batchResult.getStatus();
	auto batch = batchResult.getValue();
	OA_RETURN_IF_ERROR(oa::TimerAccess::beginDevice(inTimer, batch.stream));
	for (oa::U32 i = 0U; i < inDispatches; ++i) {
		const auto status = oa::GemmDispatch::recordPlan(
			batch, inEngine, inPlan, inProblem, inBuffers);
		if (not status.isOk()) return status;
	}
	OA_RETURN_IF_ERROR(oa::TimerAccess::endDevice(inTimer, batch.stream));
	const auto flushed = oavk::Dispatch::flush(batch, inEngine);
	if (not flushed.isOk()) {
		oa::TimerAccess::cancelDevice(inTimer);
		return flushed;
	}
	OA_RETURN_IF_ERROR(oa::TimerAccess::completeSynchronously(inTimer));
	auto elapsed = inTimer.commit(inEngine);
	if (not elapsed.isOk()) return elapsed.getStatus();
	const oa::F64 elapsedMs = elapsed.getValue();
	if (not std::isfinite(elapsedMs) or elapsedMs <= 0.0) {
		return oa::Status::error("strided GEMM benchmark returned an invalid GPU timestamp");
	}
	return elapsedMs / static_cast<oa::F64>(inDispatches);
}

} // namespace

TEST(BenchGemmVariants, CanonicalNlpQkv) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& engine = testEngine();
	const oa::GemmTunerShape shape{1024U, 32U, 32U, "nlp_qkv"};
	oa::GemmTunerResult result{};
	const oa::Status status = oa::GemmTuner::benchmarkShape(
		engine, shape,
		envIterations("OA_BENCH_GEMM_WARMUP", 3U),
		envIterations("OA_BENCH_GEMM_ITERS", 10U),
		result);
	ASSERT_TRUE(status.isOk()) << status.getMessage().data();
	auto problem = oa::GemmRouter::problemForRaw(
		shape.m, shape.n, shape.k,
		oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.training = true;
	problem.precisionHint = oa::GemmPrecision::Auto;
	EXPECT_EQ(oa::GemmRouter::select(engine, problem).variant, result.bestVariant);
	std::printf(
		"OABENCH gemm.nlp_qkv p50_ms=%.6f gflops=%.3f variant=%llu\n",
		result.bestTimeMs, result.bestGflops,
		static_cast<unsigned long long>(result.bestVariant));
}

TEST(BenchGemmVariants, ProductionShapes) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& engine = testEngine();
	const oa::U32 warmup = envIterations("OA_BENCH_GEMM_WARMUP", 5U);
	const oa::U32 iterations = envIterations("OA_BENCH_GEMM_ITERS", 30U);

	const oa::GemmTunerShape shapes[] = {
		{1024U, 32U, 32U, "nlp_qkv"},
		{1024U, 32U, 32U, "nlp_qkv_bias", oa::GemmEpilogue::Bias},
		{1024U, 64U, 32U, "nlp_ffn1"},
		{1024U, 64U, 32U, "nlp_ffn1_bias_gelu", oa::GemmEpilogue::BiasGelu},
		{1024U, 32U, 64U, "nlp_ffn2_bias", oa::GemmEpilogue::Bias},
		{4096U, 384U, 384U, "alm_qkv"},
		{4096U, 384U, 384U, "alm_qkv_bias", oa::GemmEpilogue::Bias},
		{4096U, 1536U, 384U, "alm_ffn1"},
		{4096U, 1536U, 384U, "alm_ffn1_bias_gelu", oa::GemmEpilogue::BiasGelu},
		{4096U, 384U, 1536U, "alm_ffn2"},
	};

	for (const auto& shape : shapes) {
		oa::GemmTunerResult result{};
		const oa::Status status = oa::GemmTuner::benchmarkShape(
			engine, shape, warmup, iterations, result);
		ASSERT_TRUE(status.isOk()) << status.getMessage().data();
		auto problem = oa::GemmRouter::problemForRaw(
			shape.m, shape.n, shape.k,
			oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
		problem.training = true;
		problem.precisionHint = oa::GemmPrecision::Auto;
		problem.epilogue = shape.epilogue;
		EXPECT_EQ(oa::GemmRouter::select(engine, problem).variant, result.bestVariant)
			<< "measured winner was not replayed for " << shape.name;
	}
}

TEST(BenchGemmVariants, PortableSmallMShapes) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& engine = testEngine();
	const oa::U32 warmup = envIterations("OA_BENCH_GEMM_WARMUP", 5U);
	const oa::U32 iterations = envIterations("OA_BENCH_GEMM_ITERS", 40U);
	const oa::GemmTunerShape shapes[] = {
		{1U, 256U, 256U, "decode_m1_256"},
		{1U, 512U, 512U, "decode_m1_512"},
		{4U, 256U, 256U, "decode_m4_256"},
		{8U, 512U, 512U, "decode_m8_512"},
		{1U, 512U, 512U, "decode_m1_512_bias", oa::GemmEpilogue::Bias},
	};
	for (const auto& shape : shapes) {
		oa::GemmTunerResult result{};
		const oa::Status status = oa::GemmTuner::benchmarkShape(
			engine, shape, warmup, iterations, result);
		ASSERT_TRUE(status.isOk()) << status.getMessage().data();
		std::printf(
			"OABENCH gemm.%s p50_ms=%.6f gflops=%.3f variant=%llu\n",
			shape.name, result.bestTimeMs, result.bestGflops,
			static_cast<unsigned long long>(result.bestVariant));
	}
}

TEST(BenchGemmVariants, StridedBatchScalarVsTiled) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& engine = testEngine();
	const oa::U32 batchCount = envIterations("OA_BENCH_GEMM_BATCH", 8U);
	const oa::U32 M = envIterations("OA_BENCH_GEMM_M", 128U);
	const oa::U32 N = envIterations("OA_BENCH_GEMM_N", 128U);
	const oa::U32 K = envIterations("OA_BENCH_GEMM_K", 128U);
	const oa::U32 warmup = envIterations("OA_BENCH_GEMM_WARMUP", 3U);
	const oa::U32 samples = envIterations("OA_BENCH_GEMM_ITERS", 9U);
	const oa::U32 dispatches = envIterations("OA_BENCH_GEMM_DISPATCHES", 5U);

	const size_t aElements = static_cast<size_t>(batchCount) * M * K;
	const size_t bElements = static_cast<size_t>(batchCount) * N * K;
	const size_t cElements = static_cast<size_t>(batchCount) * M * N;
	std::vector<oa::F32> aData(aElements, 0.001F);
	std::vector<oa::F32> bData(bElements, 0.002F);
	std::vector<oa::F32> cData(cElements, -1.0F);
	auto a = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(aData.data()),
			aData.size() * sizeof(oa::F32)),
		oa::MatrixShape{static_cast<oa::I64>(aElements)});
	auto b = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(bData.data()),
			bData.size() * sizeof(oa::F32)),
		oa::MatrixShape{static_cast<oa::I64>(bElements)});
	auto cScalar = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(cData.data()),
			cData.size() * sizeof(oa::F32)),
		oa::MatrixShape{static_cast<oa::I64>(cElements)});
	auto cTiled = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(cData.data()),
			cData.size() * sizeof(oa::F32)),
		oa::MatrixShape{static_cast<oa::I64>(cElements)});
	ASSERT_TRUE(a.hasStorage() and b.hasStorage()
		and cScalar.hasStorage() and cTiled.hasStorage());

	auto problem = oa::GemmRouter::problemForRaw(
		M, N, K, oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.batchCount = batchCount;
	problem.a.batchStride = M * K;
	problem.b.batchStride = N * K;
	problem.c.batchStride = M * N;
	problem.precisionHint = oa::GemmPrecision::Fp32;
	oa::MatmulPreference heuristicOnly{};
	heuristicOnly.useMeasuredCache = false;

	heuristicOnly.requiredVariant = oa::matmulVariantIdFromName("GemmStrided");
	const auto scalarPlan = oa::GemmRouter::plan(engine, problem, heuristicOnly);
	heuristicOnly.requiredVariant = oa::matmulVariantIdFromName("GemmStridedTiled");
	const auto tiledPlan = oa::GemmRouter::plan(engine, problem, heuristicOnly);
	ASSERT_TRUE(static_cast<bool>(scalarPlan));
	ASSERT_TRUE(static_cast<bool>(tiledPlan));
	ASSERT_EQ(scalarPlan.kernel, oa::GemmKernel::StridedFp32);
	ASSERT_EQ(tiledPlan.kernel, oa::GemmKernel::StridedTiledFp32);

	oavk::Buffer scalarBuffers[] = {oa::MatrixAccess::descriptor(a),
		oa::MatrixAccess::descriptor(b), oa::MatrixAccess::descriptor(cScalar)};
	oavk::Buffer tiledBuffers[] = {oa::MatrixAccess::descriptor(a),
		oa::MatrixAccess::descriptor(b), oa::MatrixAccess::descriptor(cTiled)};
	oa::Timer timer;
	ASSERT_TRUE(timer.init(engine, "gemm_strided_batch_pair").isOk());
	auto run = [&](const oa::MatmulPlan& plan, oa::Span<oavk::Buffer> buffers) {
		auto measured = measurePlan(
			engine, timer, plan, problem, buffers, dispatches);
		EXPECT_TRUE(measured.isOk())
			<< measured.getStatus().getMessage().data();
		return measured.isOk() ? measured.getValue() : 0.0;
	};
	for (oa::U32 i = 0U; i < warmup; ++i) {
		(void)run(scalarPlan, scalarBuffers);
		(void)run(tiledPlan, tiledBuffers);
	}

	std::vector<oa::F64> scalarMs;
	std::vector<oa::F64> tiledMs;
	std::vector<oa::F64> pairedGain;
	scalarMs.reserve(samples);
	tiledMs.reserve(samples);
	pairedGain.reserve(samples);
	for (oa::U32 i = 0U; i < samples; ++i) {
		oa::F64 scalar = 0.0;
		oa::F64 tiled = 0.0;
		if ((i & 1U) == 0U) {
			scalar = run(scalarPlan, scalarBuffers);
			tiled = run(tiledPlan, tiledBuffers);
		} else {
			tiled = run(tiledPlan, tiledBuffers);
			scalar = run(scalarPlan, scalarBuffers);
		}
		ASSERT_GT(scalar, 0.0);
		ASSERT_GT(tiled, 0.0);
		scalarMs.push_back(scalar);
		tiledMs.push_back(tiled);
		pairedGain.push_back(100.0 * (scalar - tiled) / scalar);
	}

	std::vector<oa::F32> scalarOutput(cElements);
	std::vector<oa::F32> tiledOutput(cElements);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(cScalar, scalarOutput.data(),
		scalarOutput.size() * sizeof(oa::F32)).isOk());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(cTiled, tiledOutput.data(),
		tiledOutput.size() * sizeof(oa::F32)).isOk());
	const oa::F32 expected = static_cast<oa::F32>(K) * 0.001F * 0.002F;
	EXPECT_NEAR(scalarOutput.front(), expected, 1.0e-6F);
	EXPECT_NEAR(scalarOutput.back(), expected, 1.0e-6F);
	EXPECT_NEAR(tiledOutput.front(), expected, 1.0e-6F);
	EXPECT_NEAR(tiledOutput.back(), expected, 1.0e-6F);

	std::printf(
		"OABENCH gemm.strided_batch batch=%u m=%u n=%u k=%u "
		"scalar_p50_ms=%.6f tiled_p50_ms=%.6f "
		"paired_gain_p50_percent=%.3f\n",
		batchCount, M, N, K, median(scalarMs), median(tiledMs),
		median(pairedGain));
}
