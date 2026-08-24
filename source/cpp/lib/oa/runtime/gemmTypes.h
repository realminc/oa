#pragma once

#include <oa/core/types.h>
#include <oa/runtime/type.h>

namespace oa {

// Private Runtime GEMM routing/cache data types. Public callers reach GEMM
// through oa::FnMatrix::matMulNt / Linear.

// oa::GemmKernel, oa::GemmPath, and oa::GemmPrecision are schema-owned by
// Runtime/type.h. Their explicit values are serialized route-cache identities:
// retired kernel IDs remain gaps and new generated families must append a new
// identity rather than reusing one.

// Logical post-GEMM contract. Keep this explicit in routing/cache keys: a
// boolean "has activation" cannot distinguish ReLU, GELU, SiLU, residual, or
// dual-output training variants and can therefore replay the wrong pipeline.
enum class GemmEpilogue : U8 {
	None = 0,
	Bias,
	BiasRelu,
	BiasGelu,
	BiasSilu,
	SiluDual,
};

// Stable identity for one compiled matmul contract. The generator will derive
// this from the canonical variant name; it is deliberately distinct from
// oa::GemmKernel because several tile/epilogue variants may share one family.
inline constexpr U64 invalidMatmulVariantId = 0U;

[[nodiscard]] constexpr U64 matmulVariantIdFromName(const char* inName) {
	U64 hash = 0xcbf29ce484222325ULL;
	for (const char* p = inName; p != nullptr and *p != '\0'; ++p) {
		hash ^= static_cast<oa::U8>(*p);
		hash *= 0x100000001b3ULL;
	}
	return hash;
}

// One picked variant for a problem. kernelName is the registry dispatch
// string; kernel is the coarse route family; Path and actualPrec describe the
// dispatch shape; gx/gy is the workgroup grid.
struct GemmRouteResult {
	oa::U64 variant = oa::invalidMatmulVariantId;
	const char*      kernelName = nullptr;
	oa::GemmKernel     kernel = oa::GemmKernel::Auto;
	oa::GemmPath       path = oa::GemmPath::Standard;
	oa::GemmPrecision  actualPrec = oa::GemmPrecision::Auto;
	oa::U32            gx = 0;
	oa::U32            gy = 0;
	oa::U32            gz = 1;
};

// route cache key for learning per-device variant selection policy
struct RouteCacheKey {
	oa::U32            vendorId;
	oa::U64            deviceId;
	oa::U32            driverId;
	oa::U64            driverVersionHash;
	oa::U64            shaderBuildId;
	// exact dimensions are intentional. Log2 buckets allowed an aligned tuned
	// winner to be replayed for an unaligned shape in the same bucket.
	oa::U32            m;
	oa::U32            n;
	oa::U32            k;
	oa::U32            batchCount = 1;
	oa::U32            aOffset = 0, aRowStride = 1, aColStride = 1, aBatchStride = 0;
	oa::U32            bOffset = 0, bRowStride = 1, bColStride = 1, bBatchStride = 0;
	oa::U32            cOffset = 0, cRowStride = 1, cColStride = 1, cBatchStride = 0;
	oa::GemmPrecision  aPrecision;
	oa::GemmPrecision  bPrecision;
	oa::GemmPrecision  outputPrecision;
	oa::GemmPrecision  requestedPrecision;
	oa::GemmEpilogue   epilogue;
	bool             aContiguous;
	bool             bContiguous;
	bool             bTransposed;
	bool             requiresPreActivation;
	bool             training;

	bool operator==(const RouteCacheKey& inOther) const noexcept {
		return vendorId == inOther.vendorId
			&& deviceId == inOther.deviceId
			&& driverId == inOther.driverId
			&& driverVersionHash == inOther.driverVersionHash
			&& shaderBuildId == inOther.shaderBuildId
			&& m == inOther.m
			&& n == inOther.n
			&& k == inOther.k
			&& batchCount == inOther.batchCount
			&& aOffset == inOther.aOffset && aRowStride == inOther.aRowStride
			&& aColStride == inOther.aColStride && aBatchStride == inOther.aBatchStride
			&& bOffset == inOther.bOffset && bRowStride == inOther.bRowStride
			&& bColStride == inOther.bColStride && bBatchStride == inOther.bBatchStride
			&& cOffset == inOther.cOffset && cRowStride == inOther.cRowStride
			&& cColStride == inOther.cColStride && cBatchStride == inOther.cBatchStride
			&& aPrecision == inOther.aPrecision
			&& bPrecision == inOther.bPrecision
			&& outputPrecision == inOther.outputPrecision
			&& requestedPrecision == inOther.requestedPrecision
			&& epilogue == inOther.epilogue
			&& aContiguous == inOther.aContiguous
			&& bContiguous == inOther.bContiguous
			&& bTransposed == inOther.bTransposed
			&& requiresPreActivation == inOther.requiresPreActivation
			&& training == inOther.training;
	}
};

struct RouteCacheKeyHash {
	oa::U64 operator()(const RouteCacheKey& inKey) const noexcept {
		oa::U64 h = 0xcbf29ce484222325ULL;
		auto mix = [&](oa::U64 v) {
			h ^= v;
			h *= 0x100000001b3ULL;
		};
		auto mix32 = [&](oa::U32 v) {
			mix(v);
		};
		mix32(inKey.vendorId);
		mix(inKey.deviceId);
		mix32(inKey.driverId);
		mix(inKey.driverVersionHash);
		mix(inKey.shaderBuildId);
		mix32(inKey.m);
		mix32(inKey.n);
		mix32(inKey.k);
		mix32(inKey.batchCount);
		mix32(inKey.aOffset); mix32(inKey.aRowStride); mix32(inKey.aColStride); mix32(inKey.aBatchStride);
		mix32(inKey.bOffset); mix32(inKey.bRowStride); mix32(inKey.bColStride); mix32(inKey.bBatchStride);
		mix32(inKey.cOffset); mix32(inKey.cRowStride); mix32(inKey.cColStride); mix32(inKey.cBatchStride);
		mix32(static_cast<oa::U32>(inKey.aPrecision));
		mix32(static_cast<oa::U32>(inKey.bPrecision));
		mix32(static_cast<oa::U32>(inKey.outputPrecision));
		mix32(static_cast<oa::U32>(inKey.requestedPrecision));
		mix32(static_cast<oa::U32>(inKey.epilogue));
		mix32(inKey.aContiguous ? 1 : 0);
		mix32(inKey.bContiguous ? 1 : 0);
		mix32(inKey.bTransposed ? 1 : 0);
		mix32(inKey.requiresPreActivation ? 1 : 0);
		mix32(inKey.training ? 1 : 0);
		return h;
	}
};

// route cache value storing measured profitability
struct RouteCacheValue {
	oa::U64 winnerVariant;
	float        medianGpuTimeMs;
	float        p95GpuTimeMs;
	oa::U32        sampleCount;
	oa::U64        lastUpdatedStep;
};

} // namespace oa
