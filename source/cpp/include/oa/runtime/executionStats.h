#pragma once

#include <oa/core/types.h>

namespace oa {

// lowering records this only when it made an explicit implementation choice.
// Unspecified nodes are ordinary fixed dispatches, not inferred successes.
enum class KernelSelectionKind : oa::U8 {
	Unspecified,
	Direct,
	PrecisionFallback,
	LayoutFallback,
	NaiveFallback,
};

// Stable execution telemetry exposed after an explicit engine boundary.
// Mutable graph, recorder, and execution-session state remain private.
class ExecutionStats {
public:
	oa::U32 nodeCount = 0;
	oa::U32 dispatchCount = 0;
	oa::U32 graphCount = 0;
	oa::U32 submissionCount = 0;
	oa::U32 compileCacheHits = 0;
	oa::U32 intraGraphBarrierCount = 0;
	oa::U32 boundaryBarrierCount = 0;
	oa::U32 hostBarrierCount = 0;
	oa::U32 warBarrierCount = 0;
	oa::U32 indirectBarrierCount = 0;
	oa::U32 aliasBarrierCount = 0;
	oa::U32 descriptorSetCount = 0;
	oa::U32 kernelSelectionCount = 0;
	oa::U32 kernelFallbackCount = 0;
	oa::U32 precisionFallbackCount = 0;
	oa::U32 layoutFallbackCount = 0;
	oa::U32 naiveFallbackCount = 0;
	oa::U64 referencedBufferBytes = 0;
	oa::F64 compileMs = 0.0;
	oa::F64 recordMs = 0.0;
	oa::F64 submitMs = 0.0;
	oa::F64 waitMs = 0.0;

	[[nodiscard]] oa::F64 cpuMs() const noexcept {
		return compileMs + recordMs + submitMs + waitMs;
	}
};

// Stable summary of one compiled executable graph. The graph and its mutable
// nodes remain private; training diagnostics expose only this value snapshot.
class GraphStats {
public:
	oa::U32 dispatchCount = 0;
	oa::U32 barrierCount = 0;
	oa::U32 descriptorSetCount = 0;
	oa::U64 totalBufferBytes = 0;
	oa::U64 potentialAliasSavings = 0;
	oa::U32 warBarrierCount = 0;
	oa::U32 indirectBarrierCount = 0;
	oa::U32 aliasBarrierCount = 0;
	oa::U32 hostBarrierCount = 0;
	oa::U32 kernelSelectionCount = 0;
	oa::U32 kernelFallbackCount = 0;
	oa::U32 precisionFallbackCount = 0;
	oa::U32 layoutFallbackCount = 0;
	oa::U32 naiveFallbackCount = 0;
};

} // namespace oa
