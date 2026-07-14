#pragma once

#include <Oa/Core/Types.h>

// Runtime GEMM routing/cache data types. Public runtime headers use these for
// pipeline-cache storage; high-level users should reach GEMM through
// OaFnMatrix::MatMulNt / Linear.

// bf16 is the sole tensor-core input dtype. fp16 was removed: its 5-bit
// exponent (range ±15) collapses the SSM scan and is unsafe for training,
// whereas bf16 keeps fp32's 8-bit exponent range. Enum values are left with
// gaps where the fp16 variants were removed so the surviving bf16 values keep
// their serialized identity (route cache / pipeline cache).
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
	GemmCmSgBf16          = 11,
	// Workgroup-scope KHR CoopMat GEMM: 32x32x16 fragments, 64x64 tile.
	GemmCmWgBf16        = 12,
};

// Dispatch style used by OaGemmRouter. Standard = tiled (Gx × Gy grid),
// StreamK = two-pass split-K with a reduce pass, CoopVec = M=1 GEMV path.
enum class OaGemmPath : OaU8 {
	Standard,
	StreamK,
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

// One picked variant for a problem. KernelName is the registry dispatch
// string; Kernel is the legacy enum kept for pipeline-cache storage; Path
// and ActualPrec describe the dispatch shape; Gx/Gy is the workgroup grid.
struct OaGemmRouteResult {
	const char*      KernelName;
	OaGemmKernel     Kernel;
	OaGemmPath       Path;
	OaGemmPrecision  ActualPrec;
	OaU32            Gx;
	OaU32            Gy;
};

struct OaGemmShapeKey {
	OaU32 M, N, K;

	bool operator==(const OaGemmShapeKey& InOther) const noexcept {
		return M == InOther.M && N == InOther.N && K == InOther.K;
	}
};

struct OaGemmShapeKeyHash {
	OaU64 operator()(const OaGemmShapeKey& InKey) const noexcept {
		OaU64 h = 0xcbf29ce484222325ULL;
		auto mix = [&](OaU32 v) {
			h ^= v;
			h *= 0x100000001b3ULL;
		};
		mix(InKey.M);
		mix(InKey.N);
		mix(InKey.K);
		return h;
	}
};

// Route cache key for learning per-device variant selection policy
struct OaRouteCacheKey {
	OaU32            VendorId;
	OaU64            DeviceId;
	OaU32            DriverId;
	OaU64            DriverVersionHash;
	OaU64            ShaderBuildId;
	OaGemmKernel     Variant;
	// Exact dimensions are intentional. Log2 buckets allowed an aligned tuned
	// winner to be replayed for an unaligned shape in the same bucket.
	OaU32            M;
	OaU32            N;
	OaU32            K;
	OaGemmPrecision  APrecision;
	OaGemmPrecision  BPrecision;
	OaGemmEpilogue   Epilogue;
	bool             Training;
	bool             UseTMA;  // TMA optimization flag (Blackwell-specific)

	bool operator==(const OaRouteCacheKey& InOther) const noexcept {
		return VendorId == InOther.VendorId
			&& DeviceId == InOther.DeviceId
			&& DriverId == InOther.DriverId
			&& DriverVersionHash == InOther.DriverVersionHash
			&& ShaderBuildId == InOther.ShaderBuildId
			&& Variant == InOther.Variant
			&& M == InOther.M
			&& N == InOther.N
			&& K == InOther.K
			&& APrecision == InOther.APrecision
			&& BPrecision == InOther.BPrecision
			&& Epilogue == InOther.Epilogue
			&& Training == InOther.Training
			&& UseTMA == InOther.UseTMA;
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
		mix32(static_cast<OaU32>(InKey.Variant));
		mix32(InKey.M);
		mix32(InKey.N);
		mix32(InKey.K);
		mix32(static_cast<OaU32>(InKey.APrecision));
		mix32(static_cast<OaU32>(InKey.BPrecision));
		mix32(static_cast<OaU32>(InKey.Epilogue));
		mix32(InKey.Training ? 1 : 0);
		mix32(InKey.UseTMA ? 1 : 0);
		return h;
	}
};

// Route cache value storing measured profitability
struct OaRouteCacheValue {
	OaGemmKernel Winner;
	float        MedianGpuTimeMs;
	float        P95GpuTimeMs;
	OaU32        SampleCount;
	OaU64        LastUpdatedStep;
};
