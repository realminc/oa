// oa::matmulRegistry — constexpr variant table + cap mask synthesis.
//
// registry-driven oa::GemmRouter metadata. The table mirrors every kernel that currently
// ships through oa::GemmRouter::Select so the rewritten Select can pick from
// the table without changing behavior. R2 replaces ComputeCapsMask's reads
// of legacy boolean fields with a real device capability query; R3 has the
// generator write this file directly from the compiled SPV lattice.

#include <oa/runtime/matmulTypes.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/device.h>
#include <oa/runtime/spirv.h>
#include "../engine/deviceAccess.h"

#include "gen/tileBaseMetadata.h"
#include "gen/tileMetadata.h"
#include "gen/tileSmallMMetadata.h"
#include "gen/tileBf16Metadata.h"

#include <array>

namespace oa::matmulRegistry {

namespace {

// One entry per kernel that the oa::PipelineRegistry currently ships.
// fields in declaration order match oa::MatmulVariant:
//   id, kernelName, kernel, Path, epilogue,
//   aPrecision, bPrecision, outputPrecision, accumulatorPrecision,
//   tileM, tileN, tileK, workgroupInvocations,
//   requiresAligned, requiresTransposedB, supportsBias, supportsActivation,
//   dualOutput, sharedMemoryBytes, requiredCapsMask.

constexpr oa::MatmulVariant kVariants[] = {
	// ── Raw GEMM (no bias, no activation) ────────────────────────────────────

	#include "gen/tileBf16Variants.inc"

	#include "gen/tileFp32Variants.inc"

	#include "gen/tileSmallMVariants.inc"

	#include "gen/tileBaseVariants.inc"

};

constexpr oa::U64 kVariantCount = sizeof(kVariants) / sizeof(kVariants[0]);

} // namespace

oa::Span<const oa::MatmulVariant> all() {
	return {kVariants, static_cast<oa::Usize>(kVariantCount)};
}

const oa::MatmulVariant* find(oa::U64 inId) {
	for (const auto& variant : kVariants) {
		if (variant.id == inId) {
			return &variant;
		}
	}
	return nullptr;
}

oa::U64 buildId() {
	oa::U64 h = 0xcbf29ce484222325ULL;
	auto mix = [&](oa::U64 v) { h ^= v; h *= 0x100000001b3ULL; };
	mix(oa::tileBaseSchemaVersion);
	mix(oa::tileBaseGeneratorAbi);
	mix(oa::tileBaseSchemaHash);
	mix(oa::tileFp32SchemaVersion);
	mix(oa::tileGeneratorAbi);
	mix(oa::tileFp32SchemaHash);
	mix(oa::tileSmallMSchemaVersion);
	mix(oa::tileSmallMGeneratorAbi);
	mix(oa::tileSmallMSchemaHash);
	mix(oa::tileBf16SchemaVersion);
	mix(oa::tileBf16GeneratorAbi);
	mix(oa::tileBf16SchemaHash);
	for (const auto& v : kVariants) {
		mix(v.id);
		for (const char* p = v.kernelName; *p != '\0'; ++p) {
			mix(static_cast<oa::U8>(*p));
		}
		mix(static_cast<oa::U8>(v.kernel));
		mix(static_cast<oa::U8>(v.path));
		mix(static_cast<oa::U8>(v.epilogue));
		mix(static_cast<oa::U8>(v.aPrecision));
		mix(static_cast<oa::U8>(v.bPrecision));
		mix(static_cast<oa::U8>(v.outputPrecision));
		mix(static_cast<oa::U8>(v.accumulatorPrecision));
		mix(v.tileM); mix(v.tileN); mix(v.tileK); mix(v.workgroupInvocations);
		mix(v.requiresAligned); mix(v.requiresTransposedB);
		mix(v.supportsBias); mix(v.supportsActivation); mix(v.dualOutput);
		mix(v.supportsArbitraryLayout); mix(v.supportsBatch);
		mix(v.maxM);
		mix(v.sharedMemoryBytes); mix(v.requiredCapsMask);
	}
	return h;
}

oa::U64 shaderBuildId() {
	static const oa::U64 cachedBuildId = [] {
		oa::U64 h = 0xcbf29ce484222325ULL;
		auto mix = [&](oa::U64 v) { h ^= v; h *= 0x100000001b3ULL; };
		mix(buildId());
		for (const auto& variant : kVariants) {
			mix(variant.id);
			mix(shaderContentHash(variant.id));
		}
		return h;
	}();
	return cachedBuildId;
}

oa::U64 shaderContentHash(oa::U64 inId) {
	for (oa::Usize i = 0; i < kVariantCount; ++i) {
		if (kVariants[i].id != inId) continue;
		const oavk::SpirvEntry* spv = oavk::findSpirv(kVariants[i].kernelName);
		return spv != nullptr ? spv->contentHash : 0U;
	}
	return 0U;
}

const oa::MatmulVariant* findByShaderName(const char* inName) {
	if (inName == nullptr) return nullptr;
	for (const auto& v : kVariants) {
		oa::Usize index = 0;
		while (inName[index] != '\0' and inName[index] == v.kernelName[index]) {
			++index;
		}
		if (inName[index] == '\0' and v.kernelName[index] == '\0') {
			return &v;
		}
	}
	return nullptr;
}

oa::U64 computeCapsMask(const oa::Engine& inRt) {
	const auto& device = oa::EngineDeviceAccess::get(inRt);
	const auto& sw     = device.info.software;
	const auto& hw     = device.info.hardware;
	const auto& shapes = sw.coopMatShapes;

	oa::U64 mask = kCapTiledFp32 | kCapNaiveFp32;

	// Vendor/driver trust: a driver may enumerate a cooperative-matrix shape it
	// cannot actually compile (Intel pre-Xe2 on Mesa/ANV, AMD pre-RDNA3 blob).
	// Withhold every CoopMat cap on an untrusted device so no CoopMat pipeline is
	// ever built — belt-and-suspenders alongside DeviceBuilder clearing
	// hasCooperativeMatrix, in case that flag leaks through a code path. Routing
	// falls back to fp32. Overridable via OA_FORCE_COOPMAT=1 (inside the trust fn).
	const bool coopMatTrusted = oavk::coopMatTrust(hw.vendorId, hw.deviceId, sw.driverId);

	// CoopMat1 KHR — only set the bit when the device actually reported a
	// usable 16×16×16 FP32-acc subgroup-scope shape. RefineCooperative-
	// MatrixCapability clears hasCooperativeMatrix below that threshold, so
	// honoring it here keeps the cap mask aligned with the shape table.
	if (coopMatTrusted and sw.hasCooperativeMatrix and shapes.has16x16x16_Fp32Acc) {
		mask |= kCapCoopMat1Khr | kCapCoopMat1Fp32Acc;
		if (shapes.bf16AccFp32.available
			and sw.shaderBfloat16CooperativeMatrixEnabled
			and sw.shaderBfloat16TypeEnabled)
		{
			mask |= kCapCoopMat1Bf16Input;
		}
		if (shapes.bf16AccFp32Workgroup.available
		    and sw.shaderBfloat16CooperativeMatrixEnabled
		    and sw.shaderBfloat16TypeEnabled)
		{
			mask |= kCapCoopMat1WorkgroupBf16;
		}
		// fp16 input removed: bf16 is the only 16-bit CoopMat1 input dtype.
	}

	// NVIDIA exposes workgroup-scope KHR cooperative matrix via the NV CoopMat2
	// feature bit, even when the KHR enumeration does not list the workgroup
	// shape. The GemmCmWgBf16 kernels use KHR CoopMat with MemoryScope.workgroup.
	if (coopMatTrusted and sw.hasCooperativeMatrix2 and shapes.coopMat2Supported
	    and shapes.coopMat2WorkgroupScope
	    and sw.shaderBfloat16CooperativeMatrixEnabled
	    and sw.shaderBfloat16TypeEnabled)
	{
		mask |= kCapCoopMat1WorkgroupBf16;
	}

	if (sw.hasCooperativeVector) {
		mask |= kCapCoopVec;
	}

	return mask;
}

} // namespace oa::matmulRegistry
