// Empirical current-device roofline probes.
//
// These benchmarks report delivered throughput through OA's production Copy
// and FP32 GEMM routes. Copy GB/s counts one algorithmic read plus one write;
// it is not a hardware-counter claim about DRAM traffic. run each test through
// Tools/Diagnostics/oabench.py for the canonical fresh-process distribution.

#include "../../oaTest.h"

#include <oa/core/fnMatrix.h>
#include <oa/core/perfStat.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/executionStats.h>
#include <oa/runtime/timer.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

oa::U32 envU32(
	const char* inName, oa::U32 inDefault, oa::U32 inMinimum, oa::U32 inMaximum)
{
	const char* text = std::getenv(inName);
	if (text == nullptr or *text == '\0') return inDefault;
	char* end = nullptr;
	const unsigned long value = std::strtoul(text, &end, 10);
	if (end == text or *end != '\0' or value < inMinimum or value > inMaximum) {
		return inDefault;
	}
	return static_cast<oa::U32>(value);
}

oa::F64 median(std::vector<oa::F64> inSamples) {
	std::sort(inSamples.begin(), inSamples.end());
	const size_t middle = inSamples.size() / 2U;
	return (inSamples.size() & 1U) != 0U
		? inSamples[middle]
		: 0.5 * (inSamples[middle - 1U] + inSamples[middle]);
}

void submitAndWait(oa::Engine& inEngine, oa::Timer* inTimer = nullptr) {
	auto submitted = inEngine.submit(inTimer);
	ASSERT_TRUE(submitted.isOk()) << submitted.getStatus().getMessage().data();
	ASSERT_TRUE(inEngine.wait(submitted.getValue()).isOk());
}

void printClock(const oa::Engine& inEngine) {
	const auto calibration = inEngine.calibrateClock(16U);
	if (not calibration.isOk()) {
		std::printf(" clock_calibrated=0 clock_status=%s",
			calibration.getStatus().getCodeName().data());
		return;
	}
	std::printf(
		" clock_calibrated=1 clock_domain=%s clock_max_deviation_ns=%llu"
		" timestamp_period_ns=%.7f timestamp_valid_bits=%u",
		calibration->hostClockDomain == oa::HostClockDomain::MonotonicRaw
			? "monotonic_raw" : "monotonic",
		static_cast<unsigned long long>(
			calibration->maximumDeviationNanoseconds),
		calibration->deviceNanosecondsPerTick,
		calibration->deviceTimestampValidBits);
}

void printCounters(const oa::ExecutionStats& inStats) {
	std::printf(
		" dispatches=%u submissions=%u graphs=%u barriers=%u"
		" boundary_barriers=%u host_barriers=%u descriptor_sets=%u"
		" referenced_bytes=%llu",
		inStats.dispatchCount,
		inStats.submissionCount,
		inStats.graphCount,
		inStats.intraGraphBarrierCount,
		inStats.boundaryBarrierCount,
		inStats.hostBarrierCount,
		inStats.descriptorSetCount,
		static_cast<unsigned long long>(inStats.referencedBufferBytes));
}

} // namespace

TEST(BenchDeviceRoofline, CopyBandwidth) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& engine = testEngine();
	const oa::U32 copyMiB = envU32("OA_ROOFLINE_COPY_MIB", 128U, 8U, 512U);
	const oa::U32 warmup = envU32("OA_ROOFLINE_WARMUP", 3U, 1U, 100U);
	const oa::U32 samples = envU32("OA_ROOFLINE_SAMPLES", 9U, 3U, 100U);
	const oa::U64 bytes = static_cast<oa::U64>(copyMiB) * 1024ULL * 1024ULL;
	const oa::U64 elements = bytes / sizeof(oa::F32);

	auto source = oa::FnMatrix::full(
		oa::MatrixShape{static_cast<oa::I64>(elements)}, 1.25,
		oa::ScalarType::Float32);
	ASSERT_TRUE(source.hasStorage());
	submitAndWait(engine);

	for (oa::U32 i = 0; i < warmup; ++i) {
		auto output = oa::FnMatrix::copy(source);
		ASSERT_TRUE(output.hasStorage());
		submitAndWait(engine);
	}

	oa::Timer timer;
	ASSERT_TRUE(timer.init(engine, "device_roofline_copy").isOk());
	std::vector<oa::F64> milliseconds;
	milliseconds.reserve(samples);
	oa::Matrix lastOutput;
	oa::ExecutionStats lastStats{};
	for (oa::U32 i = 0; i < samples; ++i) {
		auto output = oa::FnMatrix::copy(source);
		ASSERT_TRUE(output.hasStorage());
		submitAndWait(engine, &timer);
		const oa::F64 elapsed = *timer.commit(engine);
		ASSERT_TRUE(std::isfinite(elapsed));
		ASSERT_GT(elapsed, 0.0);
		milliseconds.push_back(elapsed);
		lastStats = engine.lastExecutionStats();
		lastOutput = oa::move(output);
	}

	ASSERT_NEAR(lastOutput.at(0), 1.25F, 1e-6F);
	ASSERT_NEAR(lastOutput.at(static_cast<oa::I64>(elements - 1U)), 1.25F, 1e-6F);
	const oa::F64 p50Ms = median(milliseconds);
	const oa::F64 effectiveGbps =
		(2.0 * static_cast<oa::F64>(bytes)) / (p50Ms * 1.0e6);
	std::printf(
		"OAROOFLINE schema=oa.device_roofline_probe.v1 probe=copy"
		" bytes=%llu p50_ms=%.6f effective_gbps=%.6f",
		static_cast<unsigned long long>(bytes), p50Ms, effectiveGbps);
	printClock(engine);
	printCounters(lastStats);
	std::printf("\n");
}

TEST(BenchDeviceRoofline, Fp32Compute) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& engine = testEngine();
	const oa::U32 n = envU32("OA_ROOFLINE_GEMM_N", 2048U, 128U, 4096U);
	const oa::U32 warmup = envU32("OA_ROOFLINE_WARMUP", 3U, 1U, 100U);
	const oa::U32 samples = envU32("OA_ROOFLINE_SAMPLES", 9U, 3U, 100U);
	auto a = oa::FnMatrix::full(
		oa::MatrixShape{n, n}, 0.001, oa::ScalarType::Float32);
	auto b = oa::FnMatrix::full(
		oa::MatrixShape{n, n}, 0.002, oa::ScalarType::Float32);
	ASSERT_TRUE(a.hasStorage() and b.hasStorage());
	submitAndWait(engine);

	for (oa::U32 i = 0; i < warmup; ++i) {
		auto output = oa::FnMatrix::matMulNt(a, b, oa::MatMulPrecision::Fp32);
		ASSERT_TRUE(output.hasStorage());
		submitAndWait(engine);
	}

	oa::Timer timer;
	ASSERT_TRUE(timer.init(engine, "device_roofline_fp32_gemm").isOk());
	std::vector<oa::F64> milliseconds;
	milliseconds.reserve(samples);
	oa::Matrix lastOutput;
	oa::ExecutionStats lastStats{};
	for (oa::U32 i = 0; i < samples; ++i) {
		auto output = oa::FnMatrix::matMulNt(a, b, oa::MatMulPrecision::Fp32);
		ASSERT_TRUE(output.hasStorage());
		submitAndWait(engine, &timer);
		const oa::F64 elapsed = *timer.commit(engine);
		ASSERT_TRUE(std::isfinite(elapsed));
		ASSERT_GT(elapsed, 0.0);
		milliseconds.push_back(elapsed);
		lastStats = engine.lastExecutionStats();
		lastOutput = oa::move(output);
	}

	const oa::F64 expected = static_cast<oa::F64>(n) * 0.001 * 0.002;
	ASSERT_NEAR(lastOutput.at(0), expected, expected * 0.01 + 1e-5);
	ASSERT_NEAR(lastOutput.at(
		static_cast<oa::I64>(n) * static_cast<oa::I64>(n) - 1),
		expected, expected * 0.01 + 1e-5);
	const oa::F64 p50Ms = median(milliseconds);
	const oa::F64 operations = 2.0 * static_cast<oa::F64>(n)
		* static_cast<oa::F64>(n) * static_cast<oa::F64>(n);
	const oa::F64 gflops = operations / (p50Ms * 1.0e6);
	std::printf(
		"OAROOFLINE schema=oa.device_roofline_probe.v1 probe=fp32_gemm"
		" m=%u n=%u k=%u p50_ms=%.6f gflops=%.6f",
		n, n, n, p50Ms, gflops);
	printClock(engine);
	printCounters(lastStats);
	std::printf("\n");
}
