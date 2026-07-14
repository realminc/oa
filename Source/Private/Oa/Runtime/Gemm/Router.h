#pragma once

// OaGemmRouter — unified GEMM kernel selection for Blas.cpp and Graph.cpp.
//
// Single source of truth for routing logic.
// Previously duplicated as OaGemmRouterSelect (static in Blas.cpp) and
// OaGemmRouterSelectLocal (anonymous namespace in Graph.cpp) — both removed.
//
// Design: oa/Docs/OaValidation.md §2

#include <Oa/Core/Types.h>
#include <Oa/Runtime/GemmTypes.h>
#include <Oa/Runtime/MatmulTypes.h>

// ─────────────────────────────────────────────────────────────────────────────
// Forward declarations
// ─────────────────────────────────────────────────────────────────────────────

class OaComputeEngine;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

// OaGemmPath, OaGemmPrecision, OaGemmRouteResult moved to GemmTypes.h
// (public) so OaMatmulProblem / OaMatmulVariant can reference them without
// dragging the private Router header. They remain unchanged in shape.

// ─────────────────────────────────────────────────────────────────────────────
// OaGemmRouter
// ─────────────────────────────────────────────────────────────────────────────

class OaGemmRouter {
public:
	// Primary entry called by GEMM dispatch and OaComputeGraph.
	// InPrec=Auto: use device caps + shape heuristics (full BF16/CoopVec path).
	// InPrec=Fp32: skip all CoopMat paths, use TiledFp32 or Naive (graph-safe).
	// InPrec=Bf16/Fp16: require that precision; falls back to Naive if unavailable.
	[[nodiscard]] static OaGemmRouteResult Select(
		const OaComputeEngine& InRt,
		OaU32                    InM,
		OaU32                    InN,
		OaU32                    InK,
		OaGemmPrecision          InPrec = OaGemmPrecision::Auto
	);

	// Canonical R5 entrypoint: route from a full OaMatmulProblem. The problem
	// struct carries shape + master dtypes + mirror availability + epilogue
	// flags, so callers don't have to re-derive these from OaMatrix at every
	// dispatch site. Equivalent to Select(InRt, M, N, K, PrecisionHint) for
	// the raw-GEMM path today; fused-op routes (R5 follow-ups) will consume
	// the bias/activation/mirror flags directly.
	[[nodiscard]] static OaGemmRouteResult Select(
		const OaComputeEngine& InRt,
		const OaMatmulProblem&   InProblem
	);

	// Build a baseline OaMatmulProblem for a forward linear from a matrix's
	// runtime state. Bias / activation / training flags set by caller.
	[[nodiscard]] static OaMatmulProblem ProblemForRaw(
		OaU32 InM, OaU32 InN, OaU32 InK,
		OaStoragePrecision InAMaster,
		OaStoragePrecision InBMaster,
		bool  InBTransposed
	);

	// Force a specific kernel for a shape (overrides heuristic + cache).
	// Use for benchmarking or correctness isolation. Thread-safe.
	static void ForceKernel(OaU32 InM, OaU32 InN, OaU32 InK, OaGemmKernel InKernel);
	static void ClearForced();

	// Query whether a precision tier is available on this device.
	[[nodiscard]] static bool PrecisionAvailable(
		const OaComputeEngine& InRt,
		OaGemmPrecision        InPrec
	);

	// Shape predicate for the tuned KHR CoopMat GEMM route (GemmCmSgBf16).
	// 128×128 tile, register-tiled, double-buffered. Gates on the portable
	// CoopMat1 triplet (kCapCoopMat1Khr + Bf16Input + Fp32Acc) so it qualifies
	// on AMD RDNA3.5/Strix as well as NVIDIA. Used only by the raw MatMul route
	// in Select; the fused-activation path uses the dedicated GemmBiasCmSgBf16 /
	// GemmBiasReluCmSgBf16 / GemmBiasGeluCmSgBf16 / GemmSiluCmSgBf16 dispatch functions.
	[[nodiscard]] static bool IsGemmCmSgBf16Suitable(
		const OaComputeEngine& InRt,
		OaU32                    InM,
		OaU32                    InN,
		OaU32                    InK
	);

	// Shape predicate for the workgroup-scope KHR CoopMat GEMM route (GemmCmWgBf16).
	// 32x32x16 fragments, 64×64 tile. Gates on the workgroup-scope BF16 shape
	// reported by the device; used only by the raw MatMul route in Select.
	[[nodiscard]] static bool IsGemmCmWgBf16Suitable(
		const OaComputeEngine& InRt,
		OaU32                    InM,
		OaU32                    InN,
		OaU32                    InK
	);
};
