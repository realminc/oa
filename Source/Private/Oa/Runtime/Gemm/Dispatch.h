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

class OaEngine;
class OaVkBuffer;
class OaVkBatch;

// Concrete kernel launch produced from one already-validated immutable plan.
// BufferOrder maps kernel binding slots to mathematical problem buffers. This
// is not a second planner: it is the single ABI lowering shared by eager,
// batch, and graph execution.
struct OaMatmulKernelLaunch {
	static constexpr OaU32 MaxBuffers = 4U;
	static constexpr OaU32 MaxPushBytes = 64U;

	const char* KernelName = nullptr;
	OaMatmulDispatchShape Grid{};
	OaU8 BufferOrder[MaxBuffers] = {0U, 1U, 2U, 3U};
	OaU32 BufferCount = 0U;
	OaU8 PushData[MaxPushBytes]{};
	OaU32 PushSize = 0U;
};

class OaGemmDispatch {
public:
	// Initialize: detect capabilities, select kernel paths. Idempotent.
	[[nodiscard]] static OaStatus Init(OaEngine& InRt);

	// Pure ABI lowering for a plan already accepted by OaGemmRouter::ValidatePlan.
	// Callers must validate against the active engine immediately before using
	// the returned launch. Keeping this step pure makes every execution frontend
	// consume identical push constants and buffer ordering.
	[[nodiscard]] static OaResult<OaMatmulKernelLaunch> DescribeValidatedPlan(
		const OaMatmulPlan& InPlan,
		const OaMatmulProblem& InProblem
	);

	// Execute an already-selected immutable plan. These entrypoints perform no
	// heuristic or route-cache lookup; they only validate the exact contract and
	// lower it to one dispatch.
	[[nodiscard]] static OaStatus ExecutePlan(
		OaEngine& InRt,
		const OaMatmulPlan& InPlan,
		const OaMatmulProblem& InProblem,
		OaSpan<OaVkBuffer> InBuffers
	);
	[[nodiscard]] static OaStatus RecordPlan(
		OaVkBatch& InBatch,
		OaEngine& InRt,
		const OaMatmulPlan& InPlan,
		const OaMatmulProblem& InProblem,
		OaSpan<OaVkBuffer> InBuffers
	);

	// Standard GEMM: C = A @ B^T  (B stored transposed — OA convention).
	// OaGemmRouter selects kernel from (M, N, K, dtype, device caps).
	[[nodiscard]] static OaStatus Gemm(
		OaEngine& InRt,
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
		OaEngine& InRt,
		OaVkBuffer         InA,
		OaVkBuffer         InB,
		OaVkBuffer         OutC,
		OaU32              InM,
		OaU32              InN,
		OaU32              InK
	);

	// GEMM with BF16 output cast epilogue.
	[[nodiscard]] static OaStatus GemmCmSgBf16Out(
		OaEngine& InRt,
		OaVkBuffer         InA,
		OaVkBuffer         InB,
		OaVkBuffer         OutC,
		OaU32              InM,
		OaU32              InN,
		OaU32              InK
	);

	// Tiled transpose: out[j, i] = in[i, j].
	[[nodiscard]] static OaStatus Transpose(
		OaEngine& InRt,
		OaVkBuffer         InX,
		OaVkBuffer         OutY,
		OaU32              InRows,
		OaU32              InCols
	);

	// Fused GEMM + Bias: out = A @ B^T + bias — single dispatch.
	[[nodiscard]] static OaStatus GemmBias(
		OaEngine& InRt,
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
		OaEngine& InRt,
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
		OaEngine& InRt,
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
		OaEngine& InRt,
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
		OaEngine& InRt,
		OaVkBuffer         InFused,
		OaVkBuffer         OutY,
		OaU32              InBatchSize,
		OaU32              InIntermediateSize
	);

	// Element-wise GELU on second half, multiply with first half (Gemma3)
	[[nodiscard]] static OaStatus Geglu(
		OaEngine& InRt,
		OaVkBuffer         InFused,
		OaVkBuffer         OutY,
		OaU32              InBatchSize,
		OaU32              InIntermediateSize
	);

};
