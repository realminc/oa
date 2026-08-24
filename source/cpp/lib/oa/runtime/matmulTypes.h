#pragma once

// Private oa::MatmulProblem / oa::MatmulVariant registry-driven routing inputs.
//
// oa::GemmRouter::Select used to be 400 lines of switch ladders
// over hardcoded (precision, hasCoopMat, hasBf16) triplets. This header
// introduces the declarative replacement:
//
//   - oa::StoragePrecision  — lifted from MixedPrecisionDesign.md §3.2 so the
//                            mixed-precision contract and the matmul router
//                            speak the same vocabulary.
//   - oa::MatmulProblem     — full problem description: shape, dtype pair,
//                            layout, mirror availability, training-vs-inference.
//   - oa::MatmulVariant     — registry row: kernel id, required caps, tile
//                            geometry, contract flags (bias, activation).
//   - oa::CapBit            — capability bitset populated from the actual
//                            device (R2) and AND-checked against each
//                            variant's requiredCapsMask.
//
// Reference impl: ggml-vulkan's vk_matmul_pipeline_struct + ggml_vk_guess_
// matmul_pipeline. See ggml/src/ggml-vulkan/{vulkan-shaders/vulkan-shaders-
// gen.cpp,ggml-vulkan.cpp}.

#include <oa/core/types.h>
#include <oa/runtime/gemmTypes.h>

namespace oa {

// ─────────────────────────────────────────────────────────────────────────────
// storage precision — single vocabulary across mixed-precision + matmul router
// ─────────────────────────────────────────────────────────────────────────────

// oa::StoragePrecision is schema-owned by Runtime/type.h through GemmTypes.h.
// BF16 is the only 16-bit tensor-core storage precision; FP16 was removed
// because its five-bit exponent is unsafe for the training workloads. Value 1
// stays retired so Bf16 preserves its serialized route-cache identity.

// ─────────────────────────────────────────────────────────────────────────────
// Capability bits — populated at device init (R2), AND-checked per variant
// ─────────────────────────────────────────────────────────────────────────────
//
// Bit ordering is stable across runs — variant tables reference these by
// numeric bit position in serialized form. New bits append to the end.

enum CapBit : U64 {
	// coopMat1 (KHR)
	kCapCoopMat1Khr        = 1ULL << 0,
	kCapCoopMat1Bf16Input  = 1ULL << 1,
	kCapCoopMat1Bf16Acc    = 1ULL << 2,
	kCapCoopMat1Fp16Input  = 1ULL << 3,
	kCapCoopMat1Fp32Acc    = 1ULL << 4,
	kCapCoopMat1WorkgroupBf16 = 1ULL << 14,  // 32x32x16 workgroup-scope BF16 input

	// coopVec (NV)
	kCapCoopVec            = 1ULL << 11,

	// Fallbacks always available
	kCapTiledFp32          = 1ULL << 12,
	kCapNaiveFp32          = 1ULL << 13,

	// 5-10, 14 reserved (retired CoopMat2 bits)
	// 15..63 reserved for future extension.
};

// ─────────────────────────────────────────────────────────────────────────────
// oa::MatmulProblem — full problem description handed to the router
// ─────────────────────────────────────────────────────────────────────────────
//
// Caller-built. Trivially copyable so route benchmarks can stash it in a
// hash table without oa::Matrix fixtures. Routes shouldn't peek at oa::Matrix
// directly during selection.

struct MatmulLayout {
	oa::U32 offset = 0;
	oa::U32 rowStride = 0;
	oa::U32 colStride = 1;
	oa::U32 batchStride = 0;

	[[nodiscard]] bool operator==(const MatmulLayout&) const noexcept = default;
};

struct MatmulProblem {
	oa::U32 m = 0;
	oa::U32 n = 0;
	oa::U32 k = 0;
	oa::U32 batchCount = 1;

	// Logical A[batch,M,K], B[batch,N,K], C[batch,M,N] address contracts.
	// Existing OA weights are B=[N,K], hence bTransposed remains true for the
	// tuned path even though these explicit strides describe physical storage.
	oa::MatmulLayout a{};
	oa::MatmulLayout b{};
	oa::MatmulLayout c{};

	oa::StoragePrecision aMaster          = oa::StoragePrecision::Fp32;
	oa::StoragePrecision bMaster          = oa::StoragePrecision::Fp32;
	oa::StoragePrecision requestedOutput  = oa::StoragePrecision::Fp32;

	// layout flags. bTransposed=true means B is stored [N,K] (the OA weight
	// convention) and the chosen kernel must read it transposed (either via
	// a pack step or a tensorView in the shader).
	bool aContiguous   = true;
	bool bContiguous   = true;
	bool bTransposed   = true;

	// epilogue / contract flags.
	oa::GemmEpilogue epilogue        = oa::GemmEpilogue::None;
	bool requiresPreActivation     = false;  // dual-output (Silu fwd, etc.)

	// Routing policy hints.
	bool training                  = true;
	oa::GemmPrecision precisionHint  = oa::GemmPrecision::Auto;
};

// Selection policy is intentionally smaller than the mathematical problem.
// Changing a preference may change the chosen implementation, but never the
// result contract. current kernels are deterministic and workspace-free; the
// remaining fields reserve explicit gates for generated split-K, prepacked and
// persistent variants instead of hiding those choices in router heuristics.
struct MatmulPreference {
	oa::U64 maxWorkspaceBytes       = 0;
	// Private diagnostic/tuning constraint. When set, planning succeeds only
	// when this exact generated variant is legal for the complete problem and
	// live device contract. This keeps candidate isolation local to one plan;
	// it must never be represented by process-global router state.
	oa::U64 requiredVariant = oa::invalidMatmulVariantId;
	bool useMeasuredCache         = true;
	bool requireDeterministic     = true;
	bool allowInputDownconversion = false;
	bool allowWeightPrepack       = false;
	bool allowPersistent          = false;
};

struct MatmulDispatchShape {
	oa::U32 x = 0;
	oa::U32 y = 0;
	oa::U32 z = 1;
};

// Immutable result of planning one exact problem on one device contract.
// kernelName points into the process-lifetime registry and is therefore valid
// until shutdown. Plans are runtime objects, not serialized pointers; cache
// persistence continues to use the stable Variant identity.
struct MatmulPlan {
	oa::U64 variant = oa::invalidMatmulVariantId;
	const char* kernelName = nullptr;
	oa::GemmKernel kernel = oa::GemmKernel::Auto;
	oa::GemmPath path = oa::GemmPath::Standard;
	oa::GemmPrecision actualPrecision = oa::GemmPrecision::Auto;
	oa::MatmulDispatchShape grid{};
	oa::U64 workspaceBytes = 0;
	oa::U64 problemContractHash = 0;
	oa::U64 deviceContractHash = 0;
	oa::U64 registryBuildId = 0;
	oa::U64 shaderContentHash = 0;

	[[nodiscard]] explicit operator bool() const noexcept {
		return variant != oa::invalidMatmulVariantId and kernelName != nullptr;
	}
};

// ─────────────────────────────────────────────────────────────────────────────
// oa::MatmulVariant — one row in the routing registry
// ─────────────────────────────────────────────────────────────────────────────
//
// Variants are the cross product of (kernel template, tile geometry, dtype
// pair, alignment, activation fusion). Today they live in a constexpr table
// in source/cpp/lib/oa/Runtime/Gemm/registry.cpp; R3 will replace that with
// a generator-emitted table that matches the SPIR-V actually compiled.

struct MatmulVariant {
	oa::U64 id;                 // stable generated variant identity
	const char*       kernelName;          // dispatch string in oa::PipelineRegistry
	oa::GemmKernel      kernel;              // coarse family used by routing policy
	oa::GemmPath        path           = oa::GemmPath::Standard;
	oa::GemmEpilogue    epilogue       = oa::GemmEpilogue::None;

	oa::StoragePrecision aPrecision;
	oa::StoragePrecision bPrecision;
	oa::StoragePrecision outputPrecision;
	oa::StoragePrecision accumulatorPrecision;

	// tile geometry. workgroup grid is built as
	// (divCeil(M, tileM), divCeil(N, tileN), 1) by default.
	oa::U32 tileM = 0;
	oa::U32 tileN = 0;
	oa::U32 tileK = 0;
	oa::U32 workgroupInvocations = 0;  // local_size_x of the shader

	// Legality flags.
	bool requiresAligned        = false;  // M and K must divide tile dims
	bool requiresTransposedB    = false;  // B must be [N,K] layout
	bool supportsBias           = false;  // derived contract metadata for tooling
	bool supportsActivation     = false;  // derived contract metadata for tooling
	bool dualOutput             = false;  // emits pre-activation alongside post

	// resource limits used by the occupancy heuristic (R4).
	oa::U32 sharedMemoryBytes     = 0;

	// AND-checked against the device cap mask at routing time.
	oa::U64 requiredCapsMask      = 0;
	bool supportsArbitraryLayout = false;
	bool supportsBatch           = false;
	// Zero means unbounded. Specialized schedules use an explicit upper bound
	// so forced variants, tuning, and normal routing share one legality rule.
	oa::U32 maxM                    = 0;

};

// ─────────────────────────────────────────────────────────────────────────────
// registry access — implemented in registry.cpp
// ─────────────────────────────────────────────────────────────────────────────

class Engine;

namespace matmulRegistry {

// Span over the static variant table. Stable for the program lifetime.
[[nodiscard]] oa::Span<const oa::MatmulVariant> all();

// exact stable-ID lookup. Returns nullptr for unknown or retired variants.
[[nodiscard]] const oa::MatmulVariant* find(oa::U64 inId);

// Stable hash of the registered raw/fused variant contracts. Persisted route
// entries are ignored when this changes, preventing shader/registry updates
// from replaying stale winners.
[[nodiscard]] oa::U64 buildId();

// Stable aggregate of the exact embedded SPIR-V bytes referenced by the
// registry. Unlike buildId(), this changes when compiler output changes while
// schema and launch metadata stay identical.
[[nodiscard]] oa::U64 shaderBuildId();

// exact build-generated embedded SPIR-V identity for one registered
// implementation. Hot plan validation reads immutable registry metadata; it
// does not take a name-cache mutex or rescan shader bytes.
[[nodiscard]] oa::U64 shaderContentHash(oa::U64 inId);

// Device cap mask built from oavk::Device software info. R1 derives this
// from the existing boolean fields; R2 reads the cap table directly.
[[nodiscard]] oa::U64 computeCapsMask(const oa::Engine& inRt);

// True if every bit in inRequired is set in inAvailable.
[[nodiscard]] inline bool capsSatisfy(oa::U64 inAvailable, oa::U64 inRequired) {
	return (inAvailable & inRequired) == inRequired;
}

// exact shader-identity lookup. Returns nullptr for null, unknown, or
// non-matmul names; callers must not infer capability policy from name syntax.
[[nodiscard]] const oa::MatmulVariant* findByShaderName(const char* inName);

} // namespace matmulRegistry

} // namespace oa
