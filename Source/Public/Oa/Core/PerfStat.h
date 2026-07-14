// OA CORE - Performance Statistics Accumulator
//
// Rolling window mean/stddev/percentiles. No GPU dependency — pure CPU math.
// Warmup period discards the first N samples (GPU boost clock / JIT pipeline caches).
// After warmup, accumulates over a sliding window of configurable size.
//
// Usage:
//   OaPerfStat stat("training_step_ms", 200, 20);
//   stat.Push(gpuMs);
//   if (stat.IsReady()) {
//       printf("mean=%.3fms p95=%.3fms\n", stat.Mean(), stat.P95());
//   }

#pragma once

#include <Oa/Core/Types.h>

class OaPerfStat {
public:
    explicit OaPerfStat(
        const char* InName   = "",
        OaU32       InWindow = 200,
        OaU32       InWarmup = 20
    );

    void Push(OaF64 InValue);

    // True once warmup is done and at least one sample is in the window.
    [[nodiscard]] bool IsReady() const;

    // Full window statistics (valid if IsReady()).
    [[nodiscard]] OaF64 Mean()   const;
    [[nodiscard]] OaF64 Stddev() const;
    [[nodiscard]] OaF64 Min()    const;
    [[nodiscard]] OaF64 Max()    const;
    [[nodiscard]] OaF64 P50()    const;
    [[nodiscard]] OaF64 P95()    const;
    [[nodiscard]] OaF64 P99()    const;
    [[nodiscard]] OaF64 Last()   const;
    [[nodiscard]] OaU64 Count()  const { return TotalCount_; }

    void Reset();

    [[nodiscard]] const char* GetName() const { return Name_; }

private:
    const char*  Name_;
    OaU32        Window_;
    OaU32        Warmup_;

    OaU64        TotalCount_ = 0;
    OaVec<OaF64> Ring_;
    OaU32        Head_   = 0;
    OaU32        Filled_ = 0;
    OaF64        Sum_    = 0.0;
    OaF64        SumSq_  = 0.0;
    OaF64        LastVal_ = 0.0;

    mutable bool         Dirty_     = true;
    mutable OaVec<OaF64> SortedBuf_;

    void EnsureSorted() const;
    [[nodiscard]] OaF64 Percentile(OaF64 InP) const;
};
