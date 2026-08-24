// ============================================================================
// oa::GemmDispatch - private GEMM dispatch facade with automatic kernel selection.
//
// Do not call this from user-facing code.
// oa::FnMatrix::matMulNt() and oa::FnMatrix::linear() record semantic context ops.
// ============================================================================

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/runtime/matmulTypes.h>

namespace oa { class Engine; }
namespace oavk {
class Batch;
class Buffer;
}

namespace oa {

// Concrete kernel launch produced from one already-validated immutable plan.
// bufferOrder maps kernel binding slots to mathematical problem buffers. This
// is not a second planner: it is the single ABI lowering shared by eager,
// batch, and graph execution.
struct MatmulKernelLaunch {
	static constexpr oa::U32 MaxBuffers = 4U;
	static constexpr oa::U32 MaxPushBytes = 64U;

	const char* kernelName = nullptr;
	oa::MatmulDispatchShape grid{};
	oa::U8 bufferOrder[MaxBuffers] = {0U, 1U, 2U, 3U};
	oa::U32 bufferCount = 0U;
	oa::U8 pushData[MaxPushBytes]{};
	oa::U32 pushSize = 0U;
};

class GemmDispatch {
public:
	// initialize: detect capabilities, select kernel paths. idempotent.
	[[nodiscard]] static oa::Status init(oa::Engine& inRt);

	// Pure ABI lowering for a plan already accepted by oa::GemmRouter::ValidatePlan.
	// Callers must validate against the active engine immediately before using
	// the returned launch. Keeping this step pure makes every execution frontend
	// consume identical push constants and buffer ordering.
	[[nodiscard]] static oa::Result<oa::MatmulKernelLaunch> describeValidatedPlan(
		const oa::MatmulPlan& inPlan,
		const oa::MatmulProblem& inProblem
	);

	// execute an already-selected immutable plan. These entrypoints perform no
	// heuristic or route-cache lookup; they only validate the exact contract and
	// lower it to one dispatch.
	[[nodiscard]] static oa::Status executePlan(
		oa::Engine& inRt,
		const oa::MatmulPlan& inPlan,
		const oa::MatmulProblem& inProblem,
		oa::Span<oavk::Buffer> inBuffers
	);
	[[nodiscard]] static oa::Status recordPlan(
		oavk::Batch& inBatch,
		oa::Engine& inRt,
		const oa::MatmulPlan& inPlan,
		const oa::MatmulProblem& inProblem,
		oa::Span<oavk::Buffer> inBuffers
	);

	// Standard GEMM: C = A @ B^T  (B stored transposed — OA convention).
	// oa::GemmRouter selects kernel from (M, N, K, dtype, device caps).
	[[nodiscard]] static oa::Status gemm(
		oa::Engine& inRt,
		oavk::Buffer         inA,
		oavk::Buffer         inB,
		oavk::Buffer         outC,
		oa::U32              inM,
		oa::U32              inN,
		oa::U32              inK
	);

	// Batch-aware GEMM: records into an existing batch command buffer.
	[[nodiscard]] static oa::Status gemmRecord(
		oavk::Batch&         inBatch,
		oa::Engine& inRt,
		oavk::Buffer         inA,
		oavk::Buffer         inB,
		oavk::Buffer         outC,
		oa::U32              inM,
		oa::U32              inN,
		oa::U32              inK
	);

	// GEMM with BF16 output cast epilogue.
	[[nodiscard]] static oa::Status gemmCmSgBf16Out(
		oa::Engine& inRt,
		oavk::Buffer         inA,
		oavk::Buffer         inB,
		oavk::Buffer         outC,
		oa::U32              inM,
		oa::U32              inN,
		oa::U32              inK
	);

	// Tiled transpose: out[j, i] = in[i, j].
	[[nodiscard]] static oa::Status transpose(
		oa::Engine& inRt,
		oavk::Buffer         inX,
		oavk::Buffer         outY,
		oa::U32              inRows,
		oa::U32              inCols
	);

	// Fused GEMM + Bias: out = A @ B^T + bias — single dispatch.
	[[nodiscard]] static oa::Status gemmBias(
		oa::Engine& inRt,
		oavk::Buffer         inA,
		oavk::Buffer         inB,
		oavk::Buffer         inBias,
		oavk::Buffer         outC,
		oa::U32              inM,
		oa::U32              inN,
		oa::U32              inK
	);

	// Fused GEMM + Bias + ReLU: out = max(0, A @ B^T + bias) — single dispatch.
	[[nodiscard]] static oa::Status gemmBiasRelu(
		oa::Engine& inRt,
		oavk::Buffer         inA,
		oavk::Buffer         inB,
		oavk::Buffer         inBias,
		oavk::Buffer         outC,
		oa::U32              inM,
		oa::U32              inN,
		oa::U32              inK
	);

	// Fused GEMM + Bias + GELU: out = GELU(A @ B^T + bias) — single dispatch.
	[[nodiscard]] static oa::Status gemmBiasGelu(
		oa::Engine& inRt,
		oavk::Buffer         inA,
		oavk::Buffer         inB,
		oavk::Buffer         inBias,
		oavk::Buffer         outC,
		oa::U32              inM,
		oa::U32              inN,
		oa::U32              inK
	);

	// Fused GEMM + SiLU: pre = A @ B^T,  act = siLU(pre) — single dispatch (BF16 CoopMat).
	[[nodiscard]] static oa::Status gemmSiluCoopMatBf16(
		oa::Engine& inRt,
		oavk::Buffer         inA,
		oavk::Buffer         inB,
		oavk::Buffer         outPre,
		oavk::Buffer         outAct,
		oa::U32              inM,
		oa::U32              inN,
		oa::U32              inK
	);

	// Element-wise SiLU on first half, multiply with second half
	[[nodiscard]] static oa::Status siluMul(
		oa::Engine& inRt,
		oavk::Buffer         inFused,
		oavk::Buffer         outY,
		oa::U32              inBatchSize,
		oa::U32              inIntermediateSize
	);

	// Element-wise GELU on second half, multiply with first half (Gemma3)
	[[nodiscard]] static oa::Status geglu(
		oa::Engine& inRt,
		oavk::Buffer         inFused,
		oavk::Buffer         outY,
		oa::U32              inBatchSize,
		oa::U32              inIntermediateSize
	);

};

} // namespace oa
