// OA CORE - Persistent Performance Record Store
//
// Appends benchmark results to a binary file (var/perf/OaPerf.dat).
// Each entry stores mean/stddev/percentiles keyed by (gpu_name, metric_name).
// Enables run-to-run regression detection.
//
// Usage:
//   OaPerfStore store;
//   store.Load();   // or Load("custom/path/OaPerf.dat")
//   OaPerfRecord rec = ...;
//   store.Append(rec);
//   store.PrintComparison("mnist.step.gpu_ms", rec, rt.Device);

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>

// Binary on-disk format (fixed-size, no padding concerns):
//   File header: OaPerfFileHeader (32 bytes)
//   Entries:     N × OaPerfRecord (fixed size per entry)
struct OaPerfRecord {
    OaI64 TimestampNs   = 0;
    char  GpuName[64]   = {};
    char  MetricName[64] = {};
    OaU64 SampleCount   = 0;
    OaF64 Mean          = 0.0;
    OaF64 Stddev        = 0.0;
    OaF64 Min           = 0.0;
    OaF64 Max           = 0.0;
    OaF64 P50           = 0.0;
    OaF64 P95           = 0.0;
    OaF64 P99           = 0.0;
};

class OaPerfStore {
public:
    // Load records from disk. Creates empty store if file not found.
    [[nodiscard]] OaStatus Load(const char* InPath = nullptr);

    // Append a record and flush to disk.
    [[nodiscard]] OaStatus Append(const OaPerfRecord& InRecord);

    // Find the most recent record matching (gpu_name, metric_name).
    [[nodiscard]] const OaPerfRecord* FindLatest(
        const char* InGpuName,
        const char* InMetricName
    ) const;

    // Print "current vs previous" comparison line to stdout.
    void PrintComparison(
        const char* InMetricName,
        const OaPerfRecord& InCurrent,
        const char* InGpuName
    ) const;

    [[nodiscard]] OaUsize RecordCount() const { return Records_.Size(); }

private:
    OaString            Path_;
    OaVec<OaPerfRecord> Records_;

    [[nodiscard]] OaStatus Flush() const;
};
