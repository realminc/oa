// Controlled fused-quantized MatMulNt benchmark.
//
// This executable is deliberately not registered with CTest. It compares the
// direct Q4/Q8 plane consumer with the honest existing alternative on the same
// commit: DequantizeQ*_0 followed by the production MatMulNt router. run it in
// fresh processes through the canonical benchmark tooling before making route
// or performance claims.

#include "../../oaTest.h"

#include <oa/core/fnMatrix.h>
#include <oa/core/perfStat.h>
#include <oa/ml/fnMatrix.h>
#include <oa/ml/fnmatrix/quant/fnMatrixQuantInternal.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/timer.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

oa::U32 envU32(const char* inName, oa::U32 inDefault) {
	const char* text = std::getenv(inName);
	if (text == nullptr or *text == '\0') return inDefault;
	char* end = nullptr;
	const unsigned long value = std::strtoul(text, &end, 10);
	if (end == text or *end != '\0' or value == 0UL
		or value > std::numeric_limits<oa::U32>::max())
	{
		return inDefault;
	}
	return static_cast<oa::U32>(value);
}

oa::Matrix fromFloat32(
	const std::vector<oa::F32>& inValues,
	oa::MatrixShape inShape)
{
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(inValues.data()),
			inValues.size() * sizeof(oa::F32)),
		inShape,
		oa::ScalarType::Float32);
}

template<typename Enqueue>
oa::F64 measure(
	oa::Engine& inEngine,
	const char* inName,
	oa::U32 inWarmup,
	oa::U32 inIterations,
	Enqueue&& inEnqueue)
{
	oa::Timer timer;
	EXPECT_TRUE(timer.init(inEngine, inName).isOk());
	oa::PerfStat samples(inName, inIterations, inWarmup);
	auto& context = oa::ExecutionSession::getActive();
	for (oa::U32 iteration = 0; iteration < inWarmup + inIterations; ++iteration) {
		inEnqueue();
		auto submitted = context.submit(&timer);
		EXPECT_TRUE(submitted.isOk());
		if (not submitted.isOk()) return 0.0;
		EXPECT_TRUE(context.wait(submitted.getValue()).isOk());
		const oa::F64 elapsed = *timer.commit(inEngine);
		EXPECT_TRUE(std::isfinite(elapsed) and elapsed > 0.0);
		if (not std::isfinite(elapsed) or elapsed <= 0.0) return 0.0;
		samples.push(elapsed);
	}
	EXPECT_TRUE(samples.isReady());
	return samples.isReady() ? samples.p50() : 0.0;
}

void expectNear(const oa::Matrix& inA, const oa::Matrix& inB) {
	std::vector<oa::F32> a(static_cast<std::size_t>(inA.numElements()));
	std::vector<oa::F32> b(static_cast<std::size_t>(inB.numElements()));
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		inA, a.data(), a.size() * sizeof(oa::F32)).isOk());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		inB, b.data(), b.size() * sizeof(oa::F32)).isOk());
	ASSERT_EQ(a.size(), b.size());
	for (std::size_t i = 0; i < a.size(); ++i) {
		const oa::F32 tolerance = 2.0e-4F * std::max(1.0F, std::abs(b[i]));
		EXPECT_NEAR(a[i], b[i], tolerance) << "element " << i;
	}
}

} // namespace

TEST(BenchFnMatrixQuant, FusedVersusExpandedSmallM) {
	if (not vkTestEngineOk()) GTEST_SKIP();
	const oa::U32 m = envU32("OA_BENCH_QUANT_M", 1U);
	const oa::U32 n = envU32("OA_BENCH_QUANT_N", 512U);
	const oa::U32 k = envU32("OA_BENCH_QUANT_K", 512U);
	const oa::U32 warmup = envU32("OA_BENCH_QUANT_WARMUP", 5U);
	const oa::U32 iterations = envU32("OA_BENCH_QUANT_ITERS", 21U);

	std::vector<oa::F32> input(static_cast<std::size_t>(m) * k);
	std::vector<oa::F32> weight(static_cast<std::size_t>(n) * k);
	for (std::size_t i = 0; i < input.size(); ++i) {
		input[i] = std::sin(static_cast<oa::F32>(i) * 0.017F) * 0.5F;
	}
	for (std::size_t i = 0; i < weight.size(); ++i) {
		weight[i] = std::cos(static_cast<oa::F32>(i) * 0.0031F) * 0.25F;
	}
	const auto inputMatrix = fromFloat32(
		input, oa::MatrixShape{static_cast<oa::I64>(m), static_cast<oa::I64>(k)});
	const auto weightMatrix = fromFloat32(
		weight, oa::MatrixShape{static_cast<oa::I64>(n), static_cast<oa::I64>(k)});
	ASSERT_TRUE(inputMatrix.hasStorage() and weightMatrix.hasStorage());

	const auto q4Scale = oa::FnMatrix::computeScaleQ4(weightMatrix);
	const auto q4Payload = oa::FnMatrix::quantizeQ4(weightMatrix, q4Scale);
	const auto q8Scale = oa::FnMatrix::computeScaleQ8(weightMatrix);
	const auto q8Payload = oa::FnMatrix::quantizeQ8(weightMatrix, q8Scale);
	ASSERT_TRUE(testSubmitAndWait(oa::ExecutionSession::getActive()).isOk());

	auto q4ExpandedWeight = oa::FnMatrix::dequantizeQ4(
		q4Payload, q4Scale, static_cast<oa::I64>(n) * k)
		.reshape(oa::MatrixShape{static_cast<oa::I64>(n), static_cast<oa::I64>(k)});
	auto q4Expanded = oa::FnMatrix::matMulNt(inputMatrix, q4ExpandedWeight);
	auto q4Fused = oa::FnMatrix::matMulNtQ4(inputMatrix, q4Payload, q4Scale, n);
	auto q8ExpandedWeight = oa::FnMatrix::dequantizeQ8(
		q8Payload, q8Scale, static_cast<oa::I64>(n) * k)
		.reshape(oa::MatrixShape{static_cast<oa::I64>(n), static_cast<oa::I64>(k)});
	auto q8Expanded = oa::FnMatrix::matMulNt(inputMatrix, q8ExpandedWeight);
	auto q8Fused = oa::FnMatrix::matMulNtQ8(inputMatrix, q8Payload, q8Scale, n);
	ASSERT_TRUE(testSubmitAndWait(oa::ExecutionSession::getActive()).isOk());
	expectNear(q4Fused, q4Expanded);
	expectNear(q8Fused, q8Expanded);

	oa::Vec<oa::Matrix> keepAlive;
	auto q4ExpandedMs = measure(
		testEngine(), "q4_expanded", warmup, iterations, [&] {
			auto expanded = oa::FnMatrix::dequantizeQ4(
				q4Payload, q4Scale, static_cast<oa::I64>(n) * k);
			auto output = oa::FnMatrix::matMulNt(
				inputMatrix,
				expanded.reshape(oa::MatrixShape{
					static_cast<oa::I64>(n), static_cast<oa::I64>(k)}));
			keepAlive = {expanded, output};
		});
	auto q4FusedMs = measure(
		testEngine(), "q4_fused", warmup, iterations, [&] {
			keepAlive = {oa::FnMatrix::matMulNtQ4(
				inputMatrix, q4Payload, q4Scale, n)};
		});
	auto q8ExpandedMs = measure(
		testEngine(), "q8_expanded", warmup, iterations, [&] {
			auto expanded = oa::FnMatrix::dequantizeQ8(
				q8Payload, q8Scale, static_cast<oa::I64>(n) * k);
			auto output = oa::FnMatrix::matMulNt(
				inputMatrix,
				expanded.reshape(oa::MatrixShape{
					static_cast<oa::I64>(n), static_cast<oa::I64>(k)}));
			keepAlive = {expanded, output};
		});
	auto q8FusedMs = measure(
		testEngine(), "q8_fused", warmup, iterations, [&] {
			keepAlive = {oa::FnMatrix::matMulNtQ8(
				inputMatrix, q8Payload, q8Scale, n)};
		});

	ASSERT_GT(q4ExpandedMs, 0.0);
	ASSERT_GT(q4FusedMs, 0.0);
	ASSERT_GT(q8ExpandedMs, 0.0);
	ASSERT_GT(q8FusedMs, 0.0);
	std::printf(
		"OABENCH quant.q4 M=%u N=%u K=%u expanded_p50_ms=%.6f "
		"fused_p50_ms=%.6f gain_pct=%.3f\n",
		m, n, k, q4ExpandedMs, q4FusedMs,
		100.0 * (q4ExpandedMs - q4FusedMs) / q4ExpandedMs);
	std::printf(
		"OABENCH quant.q8 M=%u N=%u K=%u expanded_p50_ms=%.6f "
		"fused_p50_ms=%.6f gain_pct=%.3f\n",
		m, n, k, q8ExpandedMs, q8FusedMs,
		100.0 * (q8ExpandedMs - q8FusedMs) / q8ExpandedMs);
}
