// ============================================================================
// OaGemmDispatch - private GEMM dispatch facade with automatic kernel selection.
//
// Do not call this from user-facing code.
// OaFnMatrix::MatMulNt() and OaFnMatrix::Linear() record semantic context ops.
// ============================================================================

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/MatmulTypes.h>

class OaComputeEngine;
class OaVkBuffer;
class OaVkBatch;

class OaGemmDispatch {
public:
	// Initialize: detect capabilities, select kernel paths. Idempotent.
	[[nodiscard]] static OaStatus Init(OaComputeEngine& InRt);

	// Execute an already-selected immutable plan. These entrypoints perform no
	// heuristic or route-cache lookup; they only validate the exact contract and
	// lower it to one dispatch.
	[[nodiscard]] static OaStatus ExecutePlan(
		OaComputeEngine& InRt,
		const OaMatmulPlan& InPlan,
		const OaMatmulProblem& InProblem,
		OaSpan<OaVkBuffer> InBuffers
	);
	[[nodiscard]] static OaStatus RecordPlan(
		OaVkBatch& InBatch,
		OaComputeEngine& InRt,
		const OaMatmulPlan& InPlan,
		const OaMatmulProblem& InProblem,
		OaSpan<OaVkBuffer> InBuffers
	);

	// Standard GEMM: C = A @ B^T  (B stored transposed — OA convention).
	// OaGemmRouter selects kernel from (M, N, K, dtype, device caps).
	[[nodiscard]] static OaStatus Gemm(
		OaComputeEngine& InRt,
		OaVkBuffer         InA,
		OaVkBuffer         InB,
		OaVkBuffer         OutC,
		OaU32              InM,
		OaU32              InN,
		OaU32              InK
	);

	// Batch-aware GEMM: records into an existing batch command buffer.
	[[nodiscard]] static OaStatus GemmRecord(
		OaVkBatch&         InBatch,
		OaComputeEngine& InRt,
		OaVkBuffer         InA,
		OaVkBuffer         InB,
		OaVkBuffer         OutC,
		OaU32              InM,
		OaU32              InN,
		OaU32              InK
	);

	// GEMM with BF16 output cast epilogue.
	[[nodiscard]] static OaStatus GemmCmSgBf16Out(
		OaComputeEngine& InRt,
		OaVkBuffer         InA,
		OaVkBuffer         InB,
		OaVkBuffer         OutC,
		OaU32              InM,
		OaU32              InN,
		OaU32              InK
	);

	// Tiled transpose: out[j, i] = in[i, j].
	[[nodiscard]] static OaStatus Transpose(
		OaComputeEngine& InRt,
		OaVkBuffer         InX,
		OaVkBuffer         OutY,
		OaU32              InRows,
		OaU32              InCols
	);

	// Fused GEMM + Bias: out = A @ B^T + bias — single dispatch.
	[[nodiscard]] static OaStatus GemmBias(
		OaComputeEngine& InRt,
		OaVkBuffer         InA,
		OaVkBuffer         InB,
		OaVkBuffer         InBias,
		OaVkBuffer         OutC,
		OaU32              InM,
		OaU32              InN,
		OaU32              InK
	);

	// Fused GEMM + Bias + ReLU: out = max(0, A @ B^T + bias) — single dispatch.
	[[nodiscard]] static OaStatus GemmBiasRelu(
		OaComputeEngine& InRt,
		OaVkBuffer         InA,
		OaVkBuffer         InB,
		OaVkBuffer         InBias,
		OaVkBuffer         OutC,
		OaU32              InM,
		OaU32              InN,
		OaU32              InK
	);

	// Fused GEMM + Bias + GELU: out = GELU(A @ B^T + bias) — single dispatch.
	[[nodiscard]] static OaStatus GemmBiasGelu(
		OaComputeEngine& InRt,
		OaVkBuffer         InA,
		OaVkBuffer         InB,
		OaVkBuffer         InBias,
		OaVkBuffer         OutC,
		OaU32              InM,
		OaU32              InN,
		OaU32              InK
	);

	// Fused GEMM + SiLU: pre = A @ B^T,  act = SiLU(pre) — single dispatch (BF16 CoopMat).
	[[nodiscard]] static OaStatus GemmSiluCoopMatBf16(
		OaComputeEngine& InRt,
		OaVkBuffer         InA,
		OaVkBuffer         InB,
		OaVkBuffer         OutPre,
		OaVkBuffer         OutAct,
		OaU32              InM,
		OaU32              InN,
		OaU32              InK
	);

	// Element-wise SiLU on first half, multiply with second half
	[[nodiscard]] static OaStatus SiluMul(
		OaComputeEngine& InRt,
		OaVkBuffer         InFused,
		OaVkBuffer         OutY,
		OaU32              InBatchSize,
		OaU32              InIntermediateSize
	);

	// Element-wise GELU on second half, multiply with first half (Gemma3)
	[[nodiscard]] static OaStatus Geglu(
		OaComputeEngine& InRt,
		OaVkBuffer         InFused,
		OaVkBuffer         OutY,
		OaU32              InBatchSize,
		OaU32              InIntermediateSize
	);

};
