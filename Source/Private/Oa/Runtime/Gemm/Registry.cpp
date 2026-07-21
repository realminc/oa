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
#include <Oa/Runtime/Spirv.h>

#include "OaTileMetadata.gen.h"
#include "OaTileBf16Metadata.gen.h"

#include <array>

namespace OaMatmulRegistry {

namespace {

// One entry per kernel that the OaPipelineRegistry currently ships.
// Fields in declaration order match OaMatmulVariant:
//   Id, KernelName, Kernel, Path, Epilogue,
//   APrecision, BPrecision, OutputPrecision, AccumulatorPrecision,
//   TileM, TileN, TileK, WorkgroupInvocations,
//   RequiresAligned, RequiresTransposedB, SupportsBias, SupportsActivation,
//   DualOutput, SharedMemoryBytes, RequiredCapsMask.

constexpr OaMatmulVariant kVariants[] = {
	// ── Raw GEMM (no bias, no activation) ────────────────────────────────────

	#include "OaTileBf16Variants.gen.inc"

	#include "OaTileFp32Variants.gen.inc"

	// Scalar fallback.
	{OaMatmulVariantIdFromName("GemmNaive"), "GemmNaive",
	 OaGemmKernel::Naive, OaGemmPath::Standard, OaGemmEpilogue::None,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 1, 1, 1, 256,
	 false, true, false, false, false, 0,
	 kCapNaiveFp32},

	// Universal correctness route for non-canonical views and strided batches.
	{OaMatmulVariantIdFromName("GemmStrided"), "GemmStrided",
	 OaGemmKernel::StridedFp32, OaGemmPath::Standard, OaGemmEpilogue::None,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 1, 1, 1, 256,
	 false, false, false, false, false, 0,
	 kCapNaiveFp32, true, true},

	// CoopVec GEMV (M==1 decode path).
	{OaMatmulVariantIdFromName("GemmCoopVec"), "GemmCoopVec",
	 OaGemmKernel::CoopVec, OaGemmPath::CoopVec, OaGemmEpilogue::None,
	 OaStoragePrecision::Bf16, OaStoragePrecision::Bf16,
	 OaStoragePrecision::Fp32, OaStoragePrecision::Fp32,
	 1, 16, 16, 32,
	 false, true, false, false, false, 0,
	 kCapCoopVec},

};

constexpr OaU64 kVariantCount = sizeof(kVariants) / sizeof(kVariants[0]);

} // namespace

OaSpan<const OaMatmulVariant> All() {
	return {kVariants, static_cast<OaUsize>(kVariantCount)};
}

const OaMatmulVariant* Find(OaMatmulVariantId InId) {
	for (const auto& variant : kVariants) {
		if (variant.Id == InId) {
			return &variant;
		}
	}
	return nullptr;
}

OaU64 BuildId() {
	OaU64 h = 0xcbf29ce484222325ULL;
	auto mix = [&](OaU64 v) { h ^= v; h *= 0x100000001b3ULL; };
	mix(OaTileFp32SchemaVersion);
	mix(OaTileGeneratorAbi);
	mix(OaTileFp32SchemaHash);
	mix(OaTileBf16SchemaVersion);
	mix(OaTileBf16GeneratorAbi);
	mix(OaTileBf16SchemaHash);
	for (const auto& v : kVariants) {
		mix(v.Id);
		for (const char* p = v.KernelName; *p != '\0'; ++p) {
			mix(static_cast<OaU8>(*p));
		}
		mix(static_cast<OaU8>(v.Kernel));
		mix(static_cast<OaU8>(v.Path));
		mix(static_cast<OaU8>(v.Epilogue));
		mix(static_cast<OaU8>(v.APrecision));
		mix(static_cast<OaU8>(v.BPrecision));
		mix(static_cast<OaU8>(v.OutputPrecision));
		mix(static_cast<OaU8>(v.AccumulatorPrecision));
		mix(v.TileM); mix(v.TileN); mix(v.TileK); mix(v.WorkgroupInvocations);
		mix(v.RequiresAligned); mix(v.RequiresTransposedB);
		mix(v.SupportsBias); mix(v.SupportsActivation); mix(v.DualOutput);
		mix(v.SupportsArbitraryLayout); mix(v.SupportsBatch);
		mix(v.SharedMemoryBytes); mix(v.RequiredCapsMask);
	}
	return h;
}

OaU64 ShaderBuildId() {
	static const OaU64 buildId = [] {
		OaU64 h = 0xcbf29ce484222325ULL;
		auto mix = [&](OaU64 v) { h ^= v; h *= 0x100000001b3ULL; };
		mix(BuildId());
		for (const auto& variant : kVariants) {
			mix(variant.Id);
			mix(ShaderContentHash(variant.Id));
		}
		return h;
	}();
	return buildId;
}

OaU64 ShaderContentHash(OaMatmulVariantId InId) {
	static const auto hashes = [] {
		std::array<OaU64, kVariantCount> result{};
		for (OaUsize i = 0; i < kVariantCount; ++i) {
			const OaSpvEntry* spv = OaSpvFindAny(kVariants[i].KernelName);
			result[i] = spv != nullptr
				? OaSpvContentHash(spv->Data, spv->Size)
				: 0U;
		}
		return result;
	}();
	for (OaUsize i = 0; i < kVariantCount; ++i) {
		if (kVariants[i].Id == InId) return hashes[i];
	}
	return 0U;
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

OaU64 RequiredCapsMaskForShaderName(const char* InName) {
	for (const auto& v : kVariants) {
		if (NameMatchesPrefix(InName, v.KernelName)) {
			return v.RequiredCapsMask;
		}
	}
	return 0;
}

OaU64 ComputeCapsMask(const OaEngine& InRt) {
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
