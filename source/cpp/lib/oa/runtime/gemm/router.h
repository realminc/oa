#pragma once

// oa::GemmRouter — unified GEMM kernel selection for Blas.cpp and graph.cpp.
//
// Single source of truth for routing logic.
// Previously duplicated as separate static selectors in Blas.cpp and graph.cpp;
// both superseded routes have been removed.
//
#include <oa/core/types.h>
#include <oa/runtime/executionStats.h>
#include <oa/runtime/gemmTypes.h>
#include <oa/runtime/matmulTypes.h>

// ─────────────────────────────────────────────────────────────────────────────
// forward declarations
// ─────────────────────────────────────────────────────────────────────────────

namespace oa { class Engine; }

namespace oa {

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

// oa::GemmPath, oa::GemmPrecision, oa::GemmRouteResult moved to GemmTypes.h
// (public) so oa::MatmulProblem / oa::MatmulVariant can reference them without
// dragging the private Router header. They remain unchanged in shape.

// ─────────────────────────────────────────────────────────────────────────────
// oa::GemmRouter
// ─────────────────────────────────────────────────────────────────────────────

class GemmRouter {
public:
	// Canonical planning boundary. Selection consumes the complete mathematical
	// problem, device contract and implementation preferences exactly once.
	// Execution may replay the returned immutable plan without rerunning cache
	// lookup or heuristics.
	[[nodiscard]] static oa::MatmulPlan plan(
		const oa::Engine& inRt,
		const oa::MatmulProblem& inProblem,
		oa::MatmulPreference inPreference = {}
	);

	// Reject stale, cross-device or contract-mismatched plans before execution.
	[[nodiscard]] static bool validatePlan(
		const oa::Engine& inRt,
		const oa::MatmulPlan& inPlan,
		const oa::MatmulProblem& inProblem
	);

	// One authority shared by machine-readable logs and executable-graph
	// telemetry. Unspecified is never returned for a valid GEMM plan.
	[[nodiscard]] static oa::KernelSelectionKind classifySelection(
		const oa::MatmulPlan& inPlan,
		const oa::MatmulProblem& inProblem);

	// primary entry called by GEMM dispatch and oa::ExecutableGraph.
	// inPrec=Auto: use device caps + shape heuristics (full BF16/CoopVec path).
	// inPrec=Fp32: skip all CoopMat paths, use TiledFp32 or naive (graph-safe).
	// inPrec=Bf16/Fp16: require that precision; falls back to Naive if unavailable.
	[[nodiscard]] static oa::GemmRouteResult select(
		const oa::Engine& inRt,
		oa::U32                    inM,
		oa::U32                    inN,
		oa::U32                    inK,
		oa::GemmPrecision          inPrec = oa::GemmPrecision::Auto
	);

	// Canonical R5 entrypoint: route from a full oa::MatmulProblem. The problem
	// struct carries shape + master dtypes + mirror availability + epilogue
	// flags, so callers don't have to re-derive these from oa::Matrix at every
	// dispatch site. Equivalent to select(inRt, M, N, K, precisionHint) for
	// the raw-GEMM path today; fused-op routes (R5 follow-ups) will consume
	// the bias/activation/mirror flags directly.
	[[nodiscard]] static oa::GemmRouteResult select(
		const oa::Engine& inRt,
		const oa::MatmulProblem&   inProblem
	);

	// Build a baseline oa::MatmulProblem for a forward linear from a matrix's
	// runtime state. Bias / activation / training flags set by caller.
	[[nodiscard]] static oa::MatmulProblem problemForRaw(
		oa::U32 inM, oa::U32 inN, oa::U32 inK,
		oa::StoragePrecision inAMaster,
		oa::StoragePrecision inBMaster,
		bool  inBTransposed
	);

	// Canonical persisted-cache contract and registry legality predicate.
	// The tuner calls these rather than duplicating routing policy.
	[[nodiscard]] static oa::RouteCacheKey cacheKey(
		const oa::Engine& inRt,
		const oa::MatmulProblem& inProblem);
	[[nodiscard]] static bool isVariantLegal(
		const oa::Engine& inRt,
		const oa::MatmulVariant& inVariant,
		const oa::MatmulProblem& inProblem);

	// query whether a precision tier is available on this device.
	[[nodiscard]] static bool precisionAvailable(
		const oa::Engine& inRt,
		oa::GemmPrecision        inPrec
	);

	// Shape predicate for the tuned KHR CoopMat GEMM route (GemmCmSgBf16).
	// 128×128 tile, register-tiled, double-buffered. Gates on the portable
	// CoopMat1 triplet (kCapCoopMat1Khr + Bf16Input + Fp32Acc) so it qualifies
	// on AMD RDNA3.5/Strix as well as NVIDIA. Used only by the raw MatMul route
	// in Select; the fused-activation path uses the dedicated GemmBiasCmSgBf16 /
	// GemmBiasReluCmSgBf16 / GemmBiasGeluCmSgBf16 / GemmSiluCmSgBf16 dispatch functions.
	[[nodiscard]] static bool isGemmCmSgBf16Suitable(
		const oa::Engine& inRt,
		oa::U32                    inM,
		oa::U32                    inN,
		oa::U32                    inK
	);

	// Shape predicate for the workgroup-scope KHR CoopMat GEMM route (GemmCmWgBf16).
	// 32x32x16 fragments, 64×64 tile. Gates on the workgroup-scope BF16 shape
	// reported by the device; used only by the raw MatMul route in Select.
	[[nodiscard]] static bool isGemmCmWgBf16Suitable(
		const oa::Engine& inRt,
		oa::U32                    inM,
		oa::U32                    inN,
		oa::U32                    inK
	);
};

} // namespace oa
