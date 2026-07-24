#pragma once

#include <Oa/Core/Types.h>

// Runtime GEMM routing/cache data types. High-level users should reach GEMM
// through OaFnMatrix::MatMulNt / Linear.

// bf16 is the sole tensor-core input dtype. fp16 was removed: its 5-bit
// exponent (range ±15) collapses the SSM scan and is unsafe for training,
// whereas bf16 keeps fp32's 8-bit exponent range. Enum values are left with
// gaps where the fp16 variants were removed so the surviving bf16 values keep
// their serialized identity in the route-cache contract.
enum class OaGemmKernel : OaU8 {
	Auto              = 0,
	// IDs 1-2 retired: were CoopMatBf16, CoopMatBf16Sk (old CoopMat1 BF16).
	// Replaced by GemmCmSgBf16 / GemmCmWgBf16 (KHR CoopMat, IDs 11-12).
	// ID 3 (CoopMat2Fused) and 14-40 (CoopMat2 variants) also retired.
	TiledFp32         = 4,
	Naive             = 5,
	CoopVec           = 6,
	// ID 7 retired: was FusedSilu (early fp32 tiled fusion attempt, never routed).
	// ID 8 retired: was FusedSwiGlu (dead code, Ffn uses separate MatMul+Swiglu).
	// ID 9 retired: was FusedSk (StreamK path removed).
	// ID 10 retired: was CoopMatSmallMBf16 (old CoopMat1 small-M).
	// Tuned KHR CoopMat GEMM: 128x128 tile, double-buffered, register-tiled.
	GemmCmSgBf16        = 11,
	// Workgroup-scope KHR CoopMat GEMM: 32x32x16 fragments, 64x64 tile.
	GemmCmWgBf16        = 12,
	// Correctness-complete arbitrary-stride / strided-batch fallback. Tuned
	// contiguous families remain preferred when their stricter contract holds.
	StridedFp32         = 13,
};

// Dispatch style used by OaGemmRouter. Standard = tiled (Gx × Gy grid),
// CoopVec = M=1 GEMV path.
enum class OaGemmPath : OaU8 {
	Standard,
	CoopVec,
};

// Precision requested by the caller. Auto lets the router pick by device
// capability + shape; explicit values skip CoopMat paths that don't match.
enum class OaGemmPrecision : OaU8 {
	Auto = 0,
	Fp32 = 1,
	Bf16 = 2,
};

// Logical post-GEMM contract. Keep this explicit in routing/cache keys: a
// boolean "has activation" cannot distinguish ReLU, GELU, SiLU, residual, or
// dual-output training variants and can therefore replay the wrong pipeline.
enum class OaGemmEpilogue : OaU8 {
	None = 0,
	Bias,
	BiasRelu,
	BiasGelu,
	BiasSilu,
	SiluDual,
};

// Stable identity for one compiled matmul contract. The generator will derive
// this from the canonical variant name; it is deliberately distinct from
// OaGemmKernel because several tile/epilogue variants may share one family.
using OaMatmulVariantId = OaU64;
constexpr OaMatmulVariantId OaInvalidMatmulVariantId = 0U;

[[nodiscard]] constexpr OaMatmulVariantId OaMatmulVariantIdFromName(const char* InName) {
	OaMatmulVariantId hash = 0xcbf29ce484222325ULL;
	for (const char* p = InName; p != nullptr and *p != '\0'; ++p) {
		hash ^= static_cast<OaU8>(*p);
		hash *= 0x100000001b3ULL;
	}
	return hash;
}

// One picked variant for a problem. KernelName is the registry dispatch
// string; Kernel is the coarse route family; Path and ActualPrec describe the
// dispatch shape; Gx/Gy is the workgroup grid.
struct OaGemmRouteResult {
	OaMatmulVariantId Variant = OaInvalidMatmulVariantId;
	const char*      KernelName = nullptr;
	OaGemmKernel     Kernel = OaGemmKernel::Auto;
	OaGemmPath       Path = OaGemmPath::Standard;
	OaGemmPrecision  ActualPrec = OaGemmPrecision::Auto;
	OaU32            Gx = 0;
	OaU32            Gy = 0;
	OaU32            Gz = 1;
};

// Route cache key for learning per-device variant selection policy
struct OaRouteCacheKey {
	OaU32            VendorId;
	OaU64            DeviceId;
	OaU32            DriverId;
	OaU64            DriverVersionHash;
	OaU64            ShaderBuildId;
	// Exact dimensions are intentional. Log2 buckets allowed an aligned tuned
	// winner to be replayed for an unaligned shape in the same bucket.
	OaU32            M;
	OaU32            N;
	OaU32            K;
	OaU32            BatchCount = 1;
	OaU32            AOffset = 0, ARowStride = 1, AColStride = 1, ABatchStride = 0;
	OaU32            BOffset = 0, BRowStride = 1, BColStride = 1, BBatchStride = 0;
	OaU32            COffset = 0, CRowStride = 1, CColStride = 1, CBatchStride = 0;
	OaGemmPrecision  APrecision;
	OaGemmPrecision  BPrecision;
	OaGemmPrecision  OutputPrecision;
	OaGemmPrecision  RequestedPrecision;
	OaGemmEpilogue   Epilogue;
	bool             AContiguous;
	bool             BContiguous;
	bool             BTransposed;
	bool             RequiresPreActivation;
	bool             Training;

	bool operator==(const OaRouteCacheKey& InOther) const noexcept {
		return VendorId == InOther.VendorId
			&& DeviceId == InOther.DeviceId
			&& DriverId == InOther.DriverId
			&& DriverVersionHash == InOther.DriverVersionHash
			&& ShaderBuildId == InOther.ShaderBuildId
			&& M == InOther.M
			&& N == InOther.N
			&& K == InOther.K
			&& BatchCount == InOther.BatchCount
			&& AOffset == InOther.AOffset && ARowStride == InOther.ARowStride
			&& AColStride == InOther.AColStride && ABatchStride == InOther.ABatchStride
			&& BOffset == InOther.BOffset && BRowStride == InOther.BRowStride
			&& BColStride == InOther.BColStride && BBatchStride == InOther.BBatchStride
			&& COffset == InOther.COffset && CRowStride == InOther.CRowStride
			&& CColStride == InOther.CColStride && CBatchStride == InOther.CBatchStride
			&& APrecision == InOther.APrecision
			&& BPrecision == InOther.BPrecision
			&& OutputPrecision == InOther.OutputPrecision
			&& RequestedPrecision == InOther.RequestedPrecision
			&& Epilogue == InOther.Epilogue
			&& AContiguous == InOther.AContiguous
			&& BContiguous == InOther.BContiguous
			&& BTransposed == InOther.BTransposed
			&& RequiresPreActivation == InOther.RequiresPreActivation
			&& Training == InOther.Training;
	}
};

struct OaRouteCacheKeyHash {
	OaU64 operator()(const OaRouteCacheKey& InKey) const noexcept {
		OaU64 h = 0xcbf29ce484222325ULL;
		auto mix = [&](OaU64 v) {
			h ^= v;
			h *= 0x100000001b3ULL;
		};
		auto mix32 = [&](OaU32 v) {
			mix(v);
		};
		mix32(InKey.VendorId);
		mix(InKey.DeviceId);
		mix32(InKey.DriverId);
		mix(InKey.DriverVersionHash);
		mix(InKey.ShaderBuildId);
		mix32(InKey.M);
		mix32(InKey.N);
		mix32(InKey.K);
		mix32(InKey.BatchCount);
		mix32(InKey.AOffset); mix32(InKey.ARowStride); mix32(InKey.AColStride); mix32(InKey.ABatchStride);
		mix32(InKey.BOffset); mix32(InKey.BRowStride); mix32(InKey.BColStride); mix32(InKey.BBatchStride);
		mix32(InKey.COffset); mix32(InKey.CRowStride); mix32(InKey.CColStride); mix32(InKey.CBatchStride);
		mix32(static_cast<OaU32>(InKey.APrecision));
		mix32(static_cast<OaU32>(InKey.BPrecision));
		mix32(static_cast<OaU32>(InKey.OutputPrecision));
		mix32(static_cast<OaU32>(InKey.RequestedPrecision));
		mix32(static_cast<OaU32>(InKey.Epilogue));
		mix32(InKey.AContiguous ? 1 : 0);
		mix32(InKey.BContiguous ? 1 : 0);
		mix32(InKey.BTransposed ? 1 : 0);
		mix32(InKey.RequiresPreActivation ? 1 : 0);
		mix32(InKey.Training ? 1 : 0);
		return h;
	}
};

// Route cache value storing measured profitability
struct OaRouteCacheValue {
	OaMatmulVariantId WinnerVariant;
	float        MedianGpuTimeMs;
	float        P95GpuTimeMs;
	OaU32        SampleCount;
	OaU64        LastUpdatedStep;
};
