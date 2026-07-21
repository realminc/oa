#pragma once

// OaMatmulProblem / OaMatmulVariant — registry-driven GEMM routing inputs.
//
// OaGemmRouter::Select used to be 400 lines of switch ladders
// over hardcoded (precision, hasCoopMat, hasBf16) triplets. This header
// introduces the declarative replacement:
//
//   - OaStoragePrecision  — lifted from MixedPrecisionDesign.md §3.2 so the
//                            mixed-precision contract and the matmul router
//                            speak the same vocabulary.
//   - OaMatmulProblem     — full problem description: shape, dtype pair,
//                            layout, mirror availability, training-vs-inference.
//   - OaMatmulVariant     — registry row: kernel id, required caps, tile
//                            geometry, contract flags (bias, activation).
//   - OaCapBit            — capability bitset populated from the actual
//                            device (R2) and AND-checked against each
//                            variant's RequiredCapsMask.
//
// Reference impl: ggml-vulkan's vk_matmul_pipeline_struct + ggml_vk_guess_
// matmul_pipeline. See ggml/src/ggml-vulkan/{vulkan-shaders/vulkan-shaders-
// gen.cpp,ggml-vulkan.cpp}.

#include <Oa/Core/Types.h>
#include <Oa/Runtime/GemmTypes.h>

// ─────────────────────────────────────────────────────────────────────────────
// Storage precision — single vocabulary across mixed-precision + matmul router
// ─────────────────────────────────────────────────────────────────────────────

// bf16 is the only 16-bit tensor-core storage precision; fp16 was removed
// (5-bit exponent unsafe for training). Value 1 kept as a gap so Bf16 retains
// its serialized identity.
enum class OaStoragePrecision : OaU8 {
	Fp32 = 0,
	Bf16 = 2,
};

// ─────────────────────────────────────────────────────────────────────────────
// Capability bits — populated at device init (R2), AND-checked per variant
// ─────────────────────────────────────────────────────────────────────────────
//
// Bit ordering is stable across runs — variant tables reference these by
// numeric bit position in serialized form. New bits append to the end.

enum OaCapBit : OaU64 {
	// CoopMat1 (KHR)
	kCapCoopMat1Khr        = 1ULL << 0,
	kCapCoopMat1Bf16Input  = 1ULL << 1,
	kCapCoopMat1Bf16Acc    = 1ULL << 2,
	kCapCoopMat1Fp16Input  = 1ULL << 3,
	kCapCoopMat1Fp32Acc    = 1ULL << 4,
	kCapCoopMat1WorkgroupBf16 = 1ULL << 14,  // 32x32x16 workgroup-scope BF16 input

	// CoopVec (NV)
	kCapCoopVec            = 1ULL << 11,

	// Fallbacks always available
	kCapTiledFp32          = 1ULL << 12,
	kCapNaiveFp32          = 1ULL << 13,

	// 5-10, 14 reserved (retired CoopMat2 bits)
	// 15..63 reserved for future extension.
};

// ─────────────────────────────────────────────────────────────────────────────
// OaMatmulProblem — full problem description handed to the router
// ─────────────────────────────────────────────────────────────────────────────
//
// Caller-built. Trivially copyable so route benchmarks can stash it in a
// hash table without OaMatrix fixtures. Routes shouldn't peek at OaMatrix
// directly during selection.

struct OaMatmulLayout {
	OaU32 Offset = 0;
	OaU32 RowStride = 0;
	OaU32 ColStride = 1;
	OaU32 BatchStride = 0;

	[[nodiscard]] bool operator==(const OaMatmulLayout&) const noexcept = default;
};

struct OaMatmulProblem {
	OaU32 M = 0;
	OaU32 N = 0;
	OaU32 K = 0;
	OaU32 BatchCount = 1;

	// Logical A[batch,M,K], B[batch,N,K], C[batch,M,N] address contracts.
	// Existing OA weights are B=[N,K], hence BTransposed remains true for the
	// tuned path even though these explicit strides describe physical storage.
	OaMatmulLayout A{};
	OaMatmulLayout B{};
	OaMatmulLayout C{};

	OaStoragePrecision AMaster          = OaStoragePrecision::Fp32;
	OaStoragePrecision BMaster          = OaStoragePrecision::Fp32;
	OaStoragePrecision RequestedOutput  = OaStoragePrecision::Fp32;

	// Layout flags. BTransposed=true means B is stored [N,K] (the OA weight
	// convention) and the chosen kernel must read it transposed (either via
	// a pack step or a tensorView in the shader).
	bool AContiguous   = true;
	bool BContiguous   = true;
	bool BTransposed   = true;

	// Epilogue / contract flags.
	OaGemmEpilogue Epilogue        = OaGemmEpilogue::None;
	bool RequiresPreActivation     = false;  // dual-output (Silu fwd, etc.)

	// Routing policy hints.
	bool Training                  = true;
	OaGemmPrecision PrecisionHint  = OaGemmPrecision::Auto;
};

// Selection policy is intentionally smaller than the mathematical problem.
// Changing a preference may change the chosen implementation, but never the
// result contract. Current kernels are deterministic and workspace-free; the
// remaining fields reserve explicit gates for generated split-K, prepacked and
// persistent variants instead of hiding those choices in router heuristics.
struct OaMatmulPreference {
	OaU64 MaxWorkspaceBytes       = 0;
	bool UseMeasuredCache         = true;
	bool RequireDeterministic     = true;
	bool AllowInputDownconversion = false;
	bool AllowWeightPrepack       = false;
	bool AllowPersistent          = false;
};

struct OaMatmulDispatchShape {
	OaU32 X = 0;
	OaU32 Y = 0;
	OaU32 Z = 1;
};

// Immutable result of planning one exact problem on one device contract.
// KernelName points into the process-lifetime registry and is therefore valid
// until shutdown. Plans are runtime objects, not serialized pointers; cache
// persistence continues to use the stable Variant identity.
struct OaMatmulPlan {
	OaMatmulVariantId Variant = OaInvalidMatmulVariantId;
	const char* KernelName = nullptr;
	OaGemmKernel Kernel = OaGemmKernel::Auto;
	OaGemmPath Path = OaGemmPath::Standard;
	OaGemmPrecision ActualPrecision = OaGemmPrecision::Auto;
	OaMatmulDispatchShape Grid{};
	OaU64 WorkspaceBytes = 0;
	OaU64 ProblemContractHash = 0;
	OaU64 DeviceContractHash = 0;
	OaU64 RegistryBuildId = 0;
	OaU64 ShaderContentHash = 0;

	[[nodiscard]] explicit operator bool() const noexcept {
		return Variant != OaInvalidMatmulVariantId and KernelName != nullptr;
	}
};

// ─────────────────────────────────────────────────────────────────────────────
// OaMatmulVariant — one row in the routing registry
// ─────────────────────────────────────────────────────────────────────────────
//
// Variants are the cross product of (kernel template, tile geometry, dtype
// pair, alignment, activation fusion). Today they live in a constexpr table
// in Source/Private/Oa/Runtime/Gemm/Registry.cpp; R3 will replace that with
// a generator-emitted table that matches the SPIR-V actually compiled.

struct OaMatmulVariant {
	OaMatmulVariantId Id;                 // stable generated variant identity
	const char*       KernelName;          // dispatch string in OaPipelineRegistry
	OaGemmKernel      Kernel;              // coarse family used by routing policy
	OaGemmPath        Path           = OaGemmPath::Standard;
	OaGemmEpilogue    Epilogue       = OaGemmEpilogue::None;

	OaStoragePrecision APrecision;
	OaStoragePrecision BPrecision;
	OaStoragePrecision OutputPrecision;
	OaStoragePrecision AccumulatorPrecision;

	// Tile geometry. Workgroup grid is built as
	// (DivCeil(M, TileM), DivCeil(N, TileN), 1) by default.
	OaU32 TileM = 0;
	OaU32 TileN = 0;
	OaU32 TileK = 0;
	OaU32 WorkgroupInvocations = 0;  // local_size_x of the shader

	// Legality flags.
	bool RequiresAligned        = false;  // M and K must divide tile dims
	bool RequiresTransposedB    = false;  // B must be [N,K] layout
	bool SupportsBias           = false;  // derived contract metadata for tooling
	bool SupportsActivation     = false;  // derived contract metadata for tooling
	bool DualOutput             = false;  // emits pre-activation alongside post

	// Resource limits used by the occupancy heuristic (R4).
	OaU32 SharedMemoryBytes     = 0;

	// AND-checked against the device cap mask at routing time.
	OaU64 RequiredCapsMask      = 0;
	bool SupportsArbitraryLayout = false;
	bool SupportsBatch           = false;

};

// ─────────────────────────────────────────────────────────────────────────────
// Registry access — implemented in Registry.cpp
// ─────────────────────────────────────────────────────────────────────────────

class OaEngine;

namespace OaMatmulRegistry {

// Span over the static variant table. Stable for the program lifetime.
[[nodiscard]] OaSpan<const OaMatmulVariant> All();

// Exact stable-ID lookup. Returns nullptr for unknown or retired variants.
[[nodiscard]] const OaMatmulVariant* Find(OaMatmulVariantId InId);

// Stable hash of the registered raw/fused variant contracts. Persisted route
// entries are ignored when this changes, preventing shader/registry updates
// from replaying stale winners.
[[nodiscard]] OaU64 BuildId();

// Stable aggregate of the exact embedded SPIR-V bytes referenced by the
// registry. Unlike BuildId(), this changes when compiler output changes while
// schema and launch metadata stay identical.
[[nodiscard]] OaU64 ShaderBuildId();

// Exact embedded SPIR-V identity for one registered implementation. The
// registry memoizes these hashes once so hot plan validation does not take a
// name-cache mutex or rescan shader bytes.
[[nodiscard]] OaU64 ShaderContentHash(OaMatmulVariantId InId);

// Device cap mask built from OaVkDevice software info. R1 derives this
// from the existing boolean fields; R2 reads the cap table directly.
[[nodiscard]] OaU64 ComputeCapsMask(const OaEngine& InRt);

// True if every bit in InRequired is set in InAvailable.
[[nodiscard]] inline bool CapsSatisfy(OaU64 InAvailable, OaU64 InRequired) {
	return (InAvailable & InRequired) == InRequired;
}

// Return the RequiredCapsMask for the first registry variant whose KernelName
// matches InName exactly or is a prefix followed by '_' (the suffix convention
// used by codegen variants like "GemmBiasReluCmSgBf16_32x32"). Returns 0 when
// no variant matches, which lets callers fall back to broader gating.
[[nodiscard]] OaU64 RequiredCapsMaskForShaderName(const char* InName);

} // namespace OaMatmulRegistry
