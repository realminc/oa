#pragma once

// OaMatmulProblem / OaMatmulVariant — registry-driven GEMM routing inputs.
//
// See Docs/Rewrite/ThisIsTheKey/OaGemmRouterRewrite.md for the full design.
// Short version: OaGemmRouter::Select used to be 400 lines of switch ladders
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
	// 15..63 reserved for future extension (StreamK, decode-vector, etc.)
};

// ─────────────────────────────────────────────────────────────────────────────
// OaMatmulProblem — full problem description handed to the router
// ─────────────────────────────────────────────────────────────────────────────
//
// Caller-built. Trivially copyable so route benchmarks can stash it in a
// hash table without OaMatrix fixtures. Routes shouldn't peek at OaMatrix
// directly during selection.

struct OaMatmulProblem {
	OaU32 M = 0;
	OaU32 N = 0;
	OaU32 K = 0;

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

// ─────────────────────────────────────────────────────────────────────────────
// OaMatmulVariant — one row in the routing registry
// ─────────────────────────────────────────────────────────────────────────────
//
// Variants are the cross product of (kernel template, tile geometry, dtype
// pair, alignment, activation fusion). Today they live in a constexpr table
// in Source/Private/Oa/Runtime/Gemm/Registry.cpp; R3 will replace that with
// a generator-emitted table that matches the SPIR-V actually compiled.

struct OaMatmulVariant {
	const char*       KernelName;          // dispatch string in OaPipelineRegistry
	OaGemmKernel      Kernel;              // legacy enum kept for pipeline-cache key
	OaGemmPath        Path           = OaGemmPath::Standard;

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
	bool SupportsBias           = false;
	bool SupportsActivation     = false;
	bool DualOutput             = false;  // emits pre-activation alongside post

	// Resource limits used by the occupancy heuristic (R4).
	OaU32 SharedMemoryBytes     = 0;

	// AND-checked against the device cap mask at routing time.
	OaU64 RequiredCapsMask      = 0;

	// Persistent kernel pattern — when true, the router dispatches numSMs
	// workgroups (1D) and the shader walks tiles in a strided loop. When
	// false (default), the router dispatches (DivCeil(M,TileM),
	// DivCeil(N,TileN)) and each workgroup handles one tile. Persistent
	// amortizes per-tile dispatch overhead on shapes with many tiles.
	bool Persistent             = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// Registry access — implemented in Registry.cpp
// ─────────────────────────────────────────────────────────────────────────────

class OaComputeEngine;

namespace OaMatmulRegistry {

// Span over the static variant table. Stable for the program lifetime.
[[nodiscard]] OaSpan<const OaMatmulVariant> All();

// Stable hash of the registered raw/fused variant contracts. Persisted route
// entries are ignored when this changes, preventing shader/registry updates
// from replaying stale winners.
[[nodiscard]] OaU64 BuildId();

// Device cap mask built from OaVkDevice software info. R1 derives this
// from the existing boolean fields; R2 reads the cap table directly.
[[nodiscard]] OaU64 ComputeCapsMask(const OaComputeEngine& InRt);

// True if every bit in InRequired is set in InAvailable.
[[nodiscard]] inline bool CapsSatisfy(OaU64 InAvailable, OaU64 InRequired) {
	return (InAvailable & InRequired) == InRequired;
}

// Occupancy heuristic — fill - 0.25*spill against the device's shader-core
// count. Higher score = better fit for (M, N) on this device. See
// Source/Private/Oa/Runtime/Gemm/Router.cpp::ScoreVariant for the original
// definition and the ggml citation.
[[nodiscard]] float ScoreVariant(const OaMatmulVariant& InVariant,
                                  OaU32 InM, OaU32 InN, OaU32 InCores);

// Pick the best registry variant whose name starts with InBaseName (so
// "GemmBiasReluCmSgBf16" matches the subgroup-scope row), gated by the
// engine's cap mask and scored against (M, N). Returns nullptr when no
// candidate satisfies caps. Used by OaContext fused-op dispatch to defer
// tile choice to the registry.
[[nodiscard]] const OaMatmulVariant* PickFusedByPrefix(
	const OaComputeEngine& InRt,
	const char*              InBaseName,
	OaU32                    InM,
	OaU32                    InN);

// Return the RequiredCapsMask for the first registry variant whose KernelName
// matches InName exactly or is a prefix followed by '_' (the suffix convention
// used by codegen variants like "GemmBiasReluCmSgBf16_32x32"). Returns 0 when
// no variant matches, which lets callers fall back to broader gating.
[[nodiscard]] OaU64 RequiredCapsMaskForShaderName(const char* InName);

} // namespace OaMatmulRegistry
