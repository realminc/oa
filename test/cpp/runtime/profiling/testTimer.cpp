// Test: oa::Timer + oa::PerfStat — GPU timestamp variance and statistical correctness

#include "../../oaTest.h"
#include <oa/runtime/timer.h>
#include <oa/core/perfStat.h>
#include <oa/core/perfStore.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/stream.h>
#include <oa/runtime/dispatch.h>
#include <oa/core/fnMatrix.h>

#include <ctime>

// ─── oa::PerfStat unit tests (no GPU) ──────────────────────────────────────────

TEST(PerfStat, WarmupDiscards) {
    oa::PerfStat stat("test", 100, 5);
    for (oa::I32 i = 0; i < 5; ++i) {
        stat.push(999.0);
    }
    EXPECT_FALSE(stat.isReady());
    EXPECT_EQ(stat.count(), 5U);

    stat.push(1.0);
    EXPECT_TRUE(stat.isReady());
    EXPECT_NEAR(stat.mean(), 1.0, 1e-9);
}

TEST(PerfStat, RollingWindowEviction) {
    oa::PerfStat stat("test", 4, 0);
    stat.push(1.0);
    stat.push(2.0);
    stat.push(3.0);
    stat.push(4.0);
    EXPECT_NEAR(stat.mean(), 2.5, 1e-9);

    stat.push(5.0);  // evicts 1.0
    EXPECT_NEAR(stat.mean(), 3.5, 1e-9);
}

TEST(PerfStat, Percentiles) {
    oa::PerfStat stat("test", 100, 0);
    for (oa::I32 i = 1; i <= 100; ++i) {
        stat.push(static_cast<oa::F64>(i));
    }
    EXPECT_TRUE(stat.isReady());
    EXPECT_NEAR(stat.p50(), 50.0, 2.0);   // median ≈ 50
    EXPECT_NEAR(stat.p95(), 95.0, 2.0);   // 95th percentile ≈ 95
    EXPECT_NEAR(stat.min(), 1.0,  1e-9);
    EXPECT_NEAR(stat.max(), 100.0, 1e-9);
}

TEST(PerfStat, StddevKnownValues) {
    oa::PerfStat stat("test", 10, 0);
    // Push identical values — stddev should be 0
    for (oa::I32 i = 0; i < 10; ++i) {
        stat.push(5.0);
    }
    EXPECT_NEAR(stat.stddev(), 0.0, 1e-9);
    EXPECT_NEAR(stat.mean(), 5.0, 1e-9);
}

TEST(PerfStat, reset) {
    oa::PerfStat stat("test", 10, 0);
    for (oa::I32 i = 0; i < 5; ++i) {
        stat.push(1.0);
    }
    EXPECT_TRUE(stat.isReady());
    stat.reset();
    EXPECT_FALSE(stat.isReady());
    EXPECT_EQ(stat.count(), 0U);
}

TEST(ClockCalibration, MapsWrappedDeviceTimestamps) {
    oa::ClockCalibration calibration{};
    calibration.deviceTimestampTicks = (oa::U64{1} << 36U) - 4U;
    calibration.hostTimestampNanoseconds = 1000U;
    calibration.deviceNanosecondsPerTick = 2.0;
    calibration.deviceTimestampValidBits = 36U;

    EXPECT_DOUBLE_EQ(
        calibration.hostNanosecondsForDeviceTimestamp(3U), 1014.0);
    EXPECT_DOUBLE_EQ(
        calibration.hostNanosecondsForDeviceTimestamp(
            (oa::U64{1} << 36U) - 9U),
        990.0);
}

TEST(GpuTimestamp, ComputesElapsedTimeAcrossValidBitWrap) {
    oavk::Timestamp timestamp;
    timestamp.writeIndex = 2U;
    timestamp.validBits = 36U;
    timestamp.nanosPerTick = 2.0;
    timestamp.results.resize(2U);
    timestamp.results[0] = (oa::U64{1} << 36U) - 4U;
    timestamp.results[1] = 3U;
    EXPECT_DOUBLE_EQ(timestamp.elapsedNs(0U, 1U), 14.0);
}

// ─── GPU timer tests (requires vulkan) ───────────────────────────────────────

TEST(DeviceTimer, InitAndMoveRelease) {
    if (not vkTestEngineOk()) { GTEST_SKIP(); }
    auto& rt = testEngine();

    oa::Timer timer;
    auto status = timer.init(rt, "test_init");
    EXPECT_TRUE(status.isOk());
    EXPECT_TRUE(timer.isInitialized());
    EXPECT_FALSE(timer.isPending());

	timer = oa::Timer{};
	EXPECT_FALSE(timer.isInitialized());
}

TEST(DeviceTimer, CalibratesDeviceAndHostClocks) {
    if (not vkTestEngineOk()) { GTEST_SKIP(); }
    auto& rt = testEngine();

    const auto calibration = rt.calibrateClock(16U);
    if (not rt.supportsClockCalibration()) {
        ASSERT_FALSE(calibration.isOk());
        EXPECT_EQ(calibration.getStatus().getCode(), oa::StatusCode::Unavailable);
        return;
    }
    ASSERT_TRUE(calibration.isOk())
        << calibration.getStatus().getMessage().data();
    EXPECT_GT(calibration->deviceNanosecondsPerTick, 0.0);
    EXPECT_GE(calibration->deviceTimestampValidBits, 36U);
    EXPECT_LE(calibration->deviceTimestampValidBits, 64U);
    EXPECT_GT(calibration->maximumDeviationNanoseconds, 0U);
    EXPECT_DOUBLE_EQ(
        calibration->hostNanosecondsForDeviceTimestamp(
            calibration->deviceTimestampTicks),
        static_cast<oa::F64>(calibration->hostTimestampNanoseconds));

    timespec now{};
    const clockid_t clock =
        calibration->hostClockDomain == oa::HostClockDomain::MonotonicRaw
        ? CLOCK_MONOTONIC_RAW : CLOCK_MONOTONIC;
    ASSERT_EQ(clock_gettime(clock, &now), 0);
    const oa::U64 nowNanoseconds = static_cast<oa::U64>(now.tv_sec) * 1000000000ULL
        + static_cast<oa::U64>(now.tv_nsec);
    EXPECT_GE(nowNanoseconds, calibration->hostTimestampNanoseconds);
    EXPECT_LT(nowNanoseconds - calibration->hostTimestampNanoseconds,
        1000000000ULL);
}

TEST(DeviceTimer, MeasuresKernelTime) {
    if (not vkTestEngineOk()) { GTEST_SKIP(); }
    auto& rt = testEngine();

    oa::Timer timer;
    ASSERT_TRUE(timer.init(rt, "fill_test").isOk());

    // time a small Fill kernel. oa::FnMatrix::fill records into oa::ExecutionSession (it does
    // not dispatch immediately), so the kernel must be timed via the context's
    // submitted batch — wrapping timer.begin/End around a bare Fill would
    // time an empty batch.
    auto& ctx = oa::ExecutionSession::getActive();
    auto tensor = oa::FnMatrix::empty(oa::MatrixShape{1024 * 1024}, oa::ScalarType::Float32);

    oa::FnMatrix::fillInPlace(tensor, 1.0F);
    auto submitted = ctx.submit(&timer);
    ASSERT_TRUE(submitted.isOk());
    ASSERT_TRUE(ctx.wait(submitted.getValue()).isOk());

    auto committed = timer.commit(rt);
    ASSERT_TRUE(committed.isOk()) << committed.getStatus().toString();
    oa::F64 gpuMs = committed.getValue();
    EXPECT_GT(gpuMs, 0.0);
    EXPECT_LT(gpuMs, 100.0);  // 1M float fill should be well under 100ms

}

// Variance test: run the same kernel 120 times and discard 20 warmup samples.
// Use a percentile spread rather than mean/stddev: one OS/driver preemption is
// a real timing outlier on an integrated GPU but does not invalidate the other
// 99 timestamp pairs.
TEST(DeviceTimer, LowVarianceOverWindow) {
    if (not vkTestEngineOk()) { GTEST_SKIP(); }
    auto& rt = testEngine();

    oa::Timer timer;
    ASSERT_TRUE(timer.init(rt, "variance_test").isOk());

    auto& ctx = oa::ExecutionSession::getActive();
    auto tensor = oa::FnMatrix::empty(oa::MatrixShape{4 * 1024 * 1024}, oa::ScalarType::Float32);

    oa::PerfStat stat("fill_4m", 100, 20);

    for (oa::I32 i = 0; i < 120; ++i) {
        // Fill records into the context; submit(&timer) dispatches the
        // recorded graph inside the timed batch (timer.begin → run → timer.end).
        oa::FnMatrix::fillInPlace(tensor, static_cast<oa::F32>(i));
        auto submitted = ctx.submit(&timer);
        ASSERT_TRUE(submitted.isOk());
        ASSERT_TRUE(ctx.wait(submitted.getValue()).isOk());

        auto committed = timer.commit(rt);
        ASSERT_TRUE(committed.isOk()) << committed.getStatus().toString();
        oa::F64 gpuMs = committed.getValue();
        stat.push(gpuMs);
    }

    ASSERT_TRUE(stat.isReady());
    const oa::F64 median = stat.p50();
    const oa::F64 p95 = stat.p95();
    const oa::F64 p95Ratio = p95 / median;
    printf("  fill_4m: median=%.4fms  p95=%.4fms  p95/median=%.2fx  max=%.4fms\n",
        median, p95, p95Ratio, stat.max());

    // The timer itself must always be sane: every sample positive, maximum bounded.
    // This catches a genuinely broken timestamp path (zeros, NaN, wild garbage)
    // on any device class.
    EXPECT_TRUE(std::isfinite(stat.min()));
    EXPECT_TRUE(std::isfinite(median));
    EXPECT_TRUE(std::isfinite(p95));
    EXPECT_TRUE(std::isfinite(stat.max()));
    EXPECT_GT(stat.min(), 0.0) << "GPU timer produced a non-positive sample";
    EXPECT_LT(stat.max(), 100.0) << "4M float fill timestamp is implausibly large";

    // Variance remains device-class-dependent. Discrete GPUs hold a stable clock;
    // integrated GPUs share power and memory bandwidth with the host. The p95
    // bound still covers 95 samples while excluding the few preemption outliers
    // that made coefficient-of-variation depend on unrelated system scheduling.
    const oa::DeviceType dt = rt.deviceType();
    const bool isDiscrete = (dt == oa::DeviceType::VkDiscrete || dt == oa::DeviceType::VkVirtualGpu);
    if (isDiscrete) {
        EXPECT_LT(p95Ratio, 1.25) << "GPU timer p95 spread > 25% on discrete GPU";
    } else {
        EXPECT_LT(p95Ratio, 4.0) << "GPU timer p95 spread implausibly high on an integrated GPU";
    }

}

// ─── oa::Timer (unified) tests ──────────────────────────────────────────────────

TEST(TimerTest, RejectsIncompleteHostRegions) {
	oa::Timer timer(oa::TimerDomain::Host, "host_contract");
	EXPECT_EQ(timer.commit().getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);
	ASSERT_TRUE(timer.begin().isOk());
	EXPECT_EQ(timer.begin().getCode(), oa::StatusCode::FailedPrecondition);
	EXPECT_EQ(timer.commit().getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);
	ASSERT_TRUE(timer.end().isOk());
	EXPECT_EQ(timer.end().getCode(), oa::StatusCode::FailedPrecondition);
	ASSERT_TRUE(timer.commit().isOk());
	EXPECT_EQ(timer.commit().getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);
}

TEST(TimerTest, RejectsUnsubmittedDeviceRegion) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& engine = testEngine();
	oa::Timer timer(oa::TimerDomain::Device, "device_contract");
	ASSERT_TRUE(timer.init(engine).isOk());
	EXPECT_EQ(timer.commit(engine).getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);
}

TEST(TimerTest, CpuMode) {
    oa::Timer timer(oa::TimerDomain::Host, "cpu_test");
    EXPECT_EQ(timer.domain(), oa::TimerDomain::Host);

    ASSERT_TRUE(timer.begin().isOk());
    // Small busy wait (~1ms)
    oa::Timestamp t0 = oa::Timestamp::now();
    while ((oa::Timestamp::now() - t0).toMs() < 1.0) {}
    ASSERT_TRUE(timer.end().isOk());

    auto committed = timer.commit(64.0);
    ASSERT_TRUE(committed.isOk());
    oa::F64 ms = committed.getValue();
    EXPECT_GT(ms, 0.5);
    EXPECT_LT(ms, 100.0);
    EXPECT_GT(timer.throughput(), 0.0);
}

TEST(TimerTest, GpuMode) {
    if (not vkTestEngineOk()) { GTEST_SKIP(); }
    auto& rt = testEngine();

    oa::Timer timer(oa::TimerDomain::Device, "gpu_unified_test");
    ASSERT_TRUE(timer.init(rt).isOk());

    auto tensor = oa::FnMatrix::empty(oa::MatrixShape{1024 * 1024}, oa::ScalarType::Float32);

    oa::FnMatrix::fillInPlace(tensor, 2.0F);
    auto& ctx = oa::ExecutionSession::getActive();
    auto submitted = ctx.submit(&timer);
    ASSERT_TRUE(submitted.isOk());
    ASSERT_TRUE(ctx.wait(submitted.getValue()).isOk());

    auto committed = timer.commit(rt, 1024.0 * 1024.0);
    ASSERT_TRUE(committed.isOk());
    oa::F64 ms = committed.getValue();
    EXPECT_GT(ms, 0.0);
    EXPECT_LT(ms, 100.0);
    EXPECT_GT(timer.throughput(), 0.0);

}

// ─── oa::PerfStore tests ────────────────────────────────────────────────────────

TEST(PerfStore, AppendAndFindLatest) {
    oa::PerfStore store;
    // Use a temp path to avoid polluting the real perf store
    oa::Path tmpPath = oa::Paths::var("perf") / "OaPerfTest_tmp.dat";
    static_cast<void>(oa::Filesystem::removeFile(tmpPath));

    auto loadStatus = store.load(tmpPath.cStr());
    EXPECT_TRUE(loadStatus.isOk());
    EXPECT_EQ(store.recordCount(), 0U);

    oa::PerfRecord rec{};
    rec.timestampNs = 1000000000LL;
    oa::memcpy(rec.gpuName,    "TestGPU", 7);
    oa::memcpy(rec.metricName, "test.metric", 11);
    rec.mean        = 42.0;
    rec.sampleCount = 100;

    ASSERT_TRUE(store.append(rec).isOk());
    EXPECT_EQ(store.recordCount(), 1U);

    const oa::PerfRecord* found = store.findLatest("TestGPU", "test.metric");
    ASSERT_NE(found, nullptr);
    EXPECT_NEAR(found->mean, 42.0, 1e-9);

    // Reload from disk and verify persistence
    oa::PerfStore store2;
    ASSERT_TRUE(store2.load(tmpPath.cStr()).isOk());
    EXPECT_EQ(store2.recordCount(), 1U);
    const oa::PerfRecord* found2 = store2.findLatest("TestGPU", "test.metric");
    ASSERT_NE(found2, nullptr);
    EXPECT_NEAR(found2->mean, 42.0, 1e-9);

    static_cast<void>(oa::Filesystem::removeFile(tmpPath));
}
