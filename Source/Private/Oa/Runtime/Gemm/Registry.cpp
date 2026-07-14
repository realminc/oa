// OaMatmulRegistry — constexpr variant table + cap mask synthesis.
//
// Registry-driven OaGemmRouter metadata. The table mirrors every kernel that currently
// ships through OaGemmRouter::Select so the rewritten Select can pick from
// the table without changing behavior. R2 replaces ComputeCapsMask's reads
// of legacy boolean fields with a real device capability query; R3 has the
// generator write this file directly from the compiled SPV lattice.

#include <Oa/Runtime/MatmulTypes.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Device.h>

#include <algorithm>

namespace OaMatmulRegistry {

namespace {

// One entry per kernel that the OaPipelineRegistry currently ships.
// Fields in declaration order match OaMatmulVariant:
//   KernelName, Kernel, Path,
//   APrecision, BPrecision, OutputPrecision, AccumulatorPrecision,
//   TileM, TileN, TileK, WorkgroupInvocations,
//   RequiresAligned, RequiresTransposedB, SupportsBias, SupportsActivation,
//   DualOutput, SharedMemoryBytes, RequiredCapsMask.

constexpr OaMatmulVariant kVariants[] = {
	// ── Raw GEMM (no bias, no activation) ────────────────────────────────────

	// Tuned KHR CoopMat GEMM — 128x128 tile, FP32 acc, double-buffered.
	{"GemmCmSgBf16",
	 OaGemmKernel::GemmCmSgBf16, OaGemmPath::Standard,
	 OaStoragePrecision::Bf16, OaStoragePrecision::Bf16,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 128, 128, 64, 256,
	 false, false, false, false, false, 0,
	 kCapCoopMat1Khr | kCapCoopMat1Bf16Input | kCapCoopMat1Fp32Acc},

	// Workgroup-scope KHR CoopMat GEMM — 32x32x16 fragments, 64x64 tile.
	{"GemmCmWgBf16",
	 OaGemmKernel::GemmCmWgBf16, OaGemmPath::Standard,
	 OaStoragePrecision::Bf16, OaStoragePrecision::Bf16,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 64, 64, 64, 256,
	 false, false, false, false, false, 0,
	 kCapCoopMat1Khr | kCapCoopMat1WorkgroupBf16 | kCapCoopMat1Fp32Acc},

	// Tiled FP32 fallback — 64x64.
	{"GemmTiled",
	 OaGemmKernel::TiledFp32, OaGemmPath::Standard,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 64, 64, 16, 256,
	 false, false, false, false, false, 0,
	 kCapTiledFp32},

	// Scalar fallback.
	{"GemmNaive",
	 OaGemmKernel::Naive, OaGemmPath::Standard,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 1, 1, 1, 256,
	 false, false, false, false, false, 0,
	 kCapNaiveFp32},

	// CoopVec GEMV (M==1 decode path).
	{"GemmCoopVec",
	 OaGemmKernel::CoopVec, OaGemmPath::CoopVec,
	 OaStoragePrecision::Bf16, OaStoragePrecision::Bf16,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 1, 16, 16, 32,
	 false, false, false, false, false, 0,
	 kCapCoopVec},

	// ── Tiled fused fallbacks (always available) ─────────────────────────────
	{"GemmBiasTiled",
	 OaGemmKernel::TiledFp32, OaGemmPath::Standard,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 64, 64, 16, 256,
	 false, false, true, false, false, 0,
	 kCapTiledFp32},

	{"GemmBiasReluTiled",
	 OaGemmKernel::TiledFp32, OaGemmPath::Standard,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 64, 64, 16, 256,
	 false, false, true, true, false, 0,
	 kCapTiledFp32},

	{"GemmBiasGeluTiled",
	 OaGemmKernel::TiledFp32, OaGemmPath::Standard,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 64, 64, 16, 256,
	 false, false, true, true, false, 0,
	 kCapTiledFp32},

};

constexpr OaU64 kVariantCount = sizeof(kVariants) / sizeof(kVariants[0]);

} // namespace

OaSpan<const OaMatmulVariant> All() {
	return {kVariants, static_cast<OaUsize>(kVariantCount)};
}

OaU64 BuildId() {
	OaU64 h = 0xcbf29ce484222325ULL;
	auto mix = [&](OaU64 v) { h ^= v; h *= 0x100000001b3ULL; };
	for (const auto& v : kVariants) {
		for (const char* p = v.KernelName; *p != '\0'; ++p) {
			mix(static_cast<OaU8>(*p));
		}
		mix(static_cast<OaU8>(v.Kernel));
		mix(static_cast<OaU8>(v.Path));
		mix(static_cast<OaU8>(v.APrecision));
		mix(static_cast<OaU8>(v.BPrecision));
		mix(static_cast<OaU8>(v.OutputPrecision));
		mix(static_cast<OaU8>(v.AccumulatorPrecision));
		mix(v.TileM); mix(v.TileN); mix(v.TileK); mix(v.WorkgroupInvocations);
		mix(v.RequiresAligned); mix(v.RequiresTransposedB);
		mix(v.SupportsBias); mix(v.SupportsActivation); mix(v.DualOutput);
		mix(v.SharedMemoryBytes); mix(v.RequiredCapsMask); mix(v.Persistent);
	}
	return h;
}

// DivCeil + ScoreVariant inline (duplicated from Router.cpp's anonymous
// namespace to keep the registry self-contained). If this surface grows
// it gets hoisted into a shared internal header.
namespace {
inline OaU32 DivCeilLocal(OaU32 InA, OaU32 InB) {
	return (InA + InB - 1U) / InB;
}
}

float ScoreVariant(const OaMatmulVariant& InVariant,
                    OaU32 InM, OaU32 InN, OaU32 InCores) {
	if (InVariant.TileM == 0U or InVariant.TileN == 0U or InCores == 0U) {
		return 0.0F;
	}
	const OaU32 tilesM = DivCeilLocal(InM, InVariant.TileM);
	const OaU32 tilesN = DivCeilLocal(InN, InVariant.TileN);
	const OaU32 total  = tilesM * tilesN;
	const float cores  = static_cast<float>(InCores);
	const float fill   = std::min(1.0F, static_cast<float>(total) / cores);
	const float spill  = std::max(0.0F,
		(static_cast<float>(total) - (4.0F * cores)) / (4.0F * cores));
	return fill - (0.25F * spill);
}

// String-prefix match that accepts either an exact match or a name where the
// next char after InBaseName is '_' (the suffix convention used by codegen
// variants like "GemmBiasReluCmSgBf16_32x32").
[[nodiscard]] static bool NameMatchesPrefix(const char* InName, const char* InBaseName) {
	OaUsize i = 0;
	while (InBaseName[i] != '\0') {
		if (InName[i] != InBaseName[i]) {
			return false;
		}
		++i;
	}
	return InName[i] == '\0' or InName[i] == '_';
}

const OaMatmulVariant* PickFusedByPrefix(
	const OaComputeEngine& InRt,
	const char*              InBaseName,
	OaU32                    InM,
	OaU32                    InN)
{
	const OaU64 caps  = InRt.GemmCapsMask();
	const OaU32 cores = InRt.Device.Info.Hardware.NumSMs;

	const OaMatmulVariant* best = nullptr;
	float bestScore = -1.0F;
	for (const auto& v : kVariants) {
		if (not NameMatchesPrefix(v.KernelName, InBaseName)) {
			continue;
		}
		if (not CapsSatisfy(caps, v.RequiredCapsMask)) {
			continue;
		}
		const float score = ScoreVariant(v, InM, InN, cores);
		if (score > bestScore) {
			bestScore = score;
			best = &v;
		}
	}
	return best;
}

OaU64 RequiredCapsMaskForShaderName(const char* InName) {
	for (const auto& v : kVariants) {
		if (NameMatchesPrefix(InName, v.KernelName)) {
			return v.RequiredCapsMask;
		}
	}
	return 0;
}

OaU64 ComputeCapsMask(const OaComputeEngine& InRt) {
	const auto& sw     = InRt.Device.Info.Software;
	const auto& hw     = InRt.Device.Info.Hardware;
	const auto& shapes = sw.CoopMatShapes;

	OaU64 mask = kCapTiledFp32 | kCapNaiveFp32;

	// Vendor/driver trust: a driver may enumerate a cooperative-matrix shape it
	// cannot actually compile (Intel pre-Xe2 on Mesa/ANV, AMD pre-RDNA3 blob).
	// Withhold every CoopMat cap on an untrusted device so no CoopMat pipeline is
	// ever built — belt-and-suspenders alongside DeviceBuilder clearing
	// HasCooperativeMatrix, in case that flag leaks through a code path. Routing
	// falls back to fp32. Overridable via OA_FORCE_COOPMAT=1 (inside the trust fn).
	const bool coopMatTrusted = OaCoopMatTrust(hw.VendorId, hw.DeviceId, sw.DriverId);

	// CoopMat1 KHR — only set the bit when the device actually reported a
	// usable 16×16×16 FP32-acc subgroup-scope shape. RefineCooperative-
	// MatrixCapability clears HasCooperativeMatrix below that threshold, so
	// honoring it here keeps the cap mask aligned with the shape table.
	if (coopMatTrusted and sw.HasCooperativeMatrix and shapes.Has16x16x16_Fp32Acc) {
		mask |= kCapCoopMat1Khr | kCapCoopMat1Fp32Acc;
		if (shapes.Bf16AccFp32.Available
			and sw.ShaderBfloat16CooperativeMatrixEnabled
			and sw.ShaderBfloat16TypeEnabled)
		{
			mask |= kCapCoopMat1Bf16Input;
		}
		if (shapes.Bf16AccFp32Workgroup.Available
		    and sw.ShaderBfloat16CooperativeMatrixEnabled
		    and sw.ShaderBfloat16TypeEnabled)
		{
			mask |= kCapCoopMat1WorkgroupBf16;
		}
		// fp16 input removed: bf16 is the only 16-bit CoopMat1 input dtype.
	}

	// NVIDIA exposes workgroup-scope KHR cooperative matrix via the NV CoopMat2
	// feature bit, even when the KHR enumeration does not list the workgroup
	// shape. The GemmCmWgBf16 kernels use KHR CoopMat with MemoryScope.Workgroup.
	if (coopMatTrusted and sw.HasCooperativeMatrix2 and shapes.CoopMat2Supported
	    and shapes.CoopMat2WorkgroupScope
	    and sw.ShaderBfloat16CooperativeMatrixEnabled
	    and sw.ShaderBfloat16TypeEnabled)
	{
		mask |= kCapCoopMat1WorkgroupBf16;
	}

	if (sw.HasCooperativeVector) {
		mask |= kCapCoopVec;
	}

	return mask;
}

} // namespace OaMatmulRegistry
