// OA CORE - Performance statistics Accumulator
//
// Rolling window mean/stddev/percentiles. No GPU dependency — pure CPU math.
// Warmup period discards the first N samples (GPU boost clock / JIT pipeline caches).
// After warmup, accumulates over a sliding window of configurable size.
//
// usage:
//   PerfStat stat("training_step_ms", 200, 20);
//   stat.push(gpuMs);
//   if (stat.isReady()) {
//       printf("mean=%.3fms p95=%.3fms\n", stat.mean(), stat.p95());
//   }

#pragma once

#include <oa/core/types.h>

namespace oa {

class PerfStat {
public:
    explicit PerfStat(
        const char* inName   = "",
        oa::U32       inWindow = 200,
        oa::U32       inWarmup = 20
    );

    void push(oa::F64 inValue);

    // True once warmup is done and at least one sample is in the window.
    [[nodiscard]] bool isReady() const;

    // Full window statistics (valid if isReady()).
    [[nodiscard]] oa::F64 mean()   const;
    [[nodiscard]] oa::F64 stddev() const;
    [[nodiscard]] oa::F64 min()    const;
    [[nodiscard]] oa::F64 max()    const;
    [[nodiscard]] oa::F64 p50()    const;
    [[nodiscard]] oa::F64 p95()    const;
    [[nodiscard]] oa::F64 p99()    const;
    [[nodiscard]] oa::F64 last()   const;
    [[nodiscard]] oa::U64 count()  const { return totalCount_; }

    void reset();

    [[nodiscard]] const char* getName() const { return name_; }

private:
    const char*  name_;
    oa::U32        window_;
    oa::U32        warmup_;

    oa::U64        totalCount_ = 0;
    oa::Vec<oa::F64> ring_;
    oa::U32        head_   = 0;
    oa::U32        filled_ = 0;
    oa::F64        sum_    = 0.0;
    oa::F64        sumSq_  = 0.0;
    oa::F64        lastVal_ = 0.0;

    mutable bool         dirty_     = true;
    mutable oa::Vec<oa::F64> sortedBuf_;

    void ensureSorted() const;
    [[nodiscard]] oa::F64 percentile(oa::F64 inP) const;
};

} // namespace oa
