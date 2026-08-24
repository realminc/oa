// OA CORE - persistent Performance Record store

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>

namespace oa {

struct PerfRecord {
	oa::I64 timestampNs   = 0;
	char  gpuName[64]   = {};
	char  metricName[64] = {};
	oa::U64 sampleCount   = 0;
	oa::F64 mean          = 0.0;
	oa::F64 stddev        = 0.0;
	oa::F64 min           = 0.0;
	oa::F64 max           = 0.0;
	oa::F64 p50           = 0.0;
	oa::F64 p95           = 0.0;
	oa::F64 p99           = 0.0;
};

class PerfStore {
public:
	[[nodiscard]] oa::Status load(const char* inPath = nullptr);
	[[nodiscard]] oa::Status append(const PerfRecord& inRecord);
	[[nodiscard]] const PerfRecord* findLatest(
		const char* inGpuName,
		const char* inMetricName
	) const;
	void printComparison(
		const char* inMetricName,
		const PerfRecord& inCurrent,
		const char* inGpuName
	) const;
	[[nodiscard]] oa::Usize recordCount() const { return records_.size(); }

private:
	oa::String           path_;
	oa::Vec<PerfRecord>  records_;
	[[nodiscard]] oa::Status flush_() const;
};

// Legacy aliases — remove once call sites are migrated.

} // namespace oa
