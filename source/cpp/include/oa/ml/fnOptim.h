// OA ML - Optimizer operations
//
// Stateless optimizer step functions for training.
// Records through the active engine's private recorder (clean API level 1).
// One overload per op — no graph /
// executor variants. Mutates param / momentum / m / v in-place.
//
// design (mirrors oa::FnMatrix / oa::FnLoss):
//   oa::FnMatrix   → forward operations (Relu, Matmul, Add, …)
//   oa::FnLoss     → loss functions (CrossEntropy, Mse, …)
//   oa::FnOptim    → Optimizer steps (AdamWStep, SgdStep, …)

#pragma once

#include <oa/core/matrix.h>

namespace oa {

namespace FnOptim {

struct AdamWParamSet {
	oa::Matrix* param;
	oa::Matrix* m;
	oa::Matrix* v;
	const oa::Matrix* grad;
};

/// advance the mutable replay state immediately before replay-safe AdamW
/// updates. The host seeds state[0] with the previous completed step; this
/// one-thread GPU op makes each replay consume a distinct optimizer step.
void adamWAdvanceGraphState(oa::Matrix& inOutState);

/// AdamW optimizer step: param -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * param)
void adamWStep(
	oa::Matrix& inOutParam,
	oa::Matrix& inOutM,
	oa::Matrix& inOutV,
	const oa::Matrix& inGrad,
	oa::F32 inLr,
	oa::F32 inBeta1,
	oa::F32 inBeta2,
	oa::F32 inEps,
	oa::F32 inWeightDecay,
	oa::I32 inStep
);

/// replay-safe AdamW step. Mutable optimizer scalars are read from inState:
/// uint32[6] = {step, bitcast(lr), bitcast(beta1), bitcast(beta2),
/// bitcast(eps), bitcast(weight_decay)}. Only Count remains in push constants.
void adamWStepGraph(
	oa::Matrix& inOutParam,
	oa::Matrix& inOutM,
	oa::Matrix& inOutV,
	const oa::Matrix& inGrad,
	const oa::Matrix& inState
);

/// AdamW optimizer step for multiple parameter tensors.
/// Currently fuses the common four-tensor case into one dispatch and falls back
/// to AdamWStep for other counts.
void adamWStepMany(
	oa::Span<const AdamWParamSet> inParams,
	oa::F32 inLr,
	oa::F32 inBeta1,
	oa::F32 inBeta2,
	oa::F32 inEps,
	oa::F32 inWeightDecay,
	oa::I32 inStep
);

/// replay-safe fused four-parameter AdamW step using the same state contract as
/// AdamWStepGraph. Other parameter counts fall back to per-parameter graph ops.
void adamWStepManyGraph(
	oa::Span<const AdamWParamSet> inParams,
	const oa::Matrix& inState
);

/// SGD optimizer step with optional momentum and weight decay.
/// Pass an empty inOutMomentum (or inMomentum == 0) for plain SGD.
void sgdStep(
	oa::Matrix& inOutParam,
	oa::Matrix& inOutMomentum,
	const oa::Matrix& inGrad,
	oa::F32 inLr,
	oa::F32 inMomentum,
	oa::F32 inWeightDecay
);

/// Muon optimizer step: Nesterov momentum, NS5 orthogonalization, and
/// Moonshot 0.2*sqrt(max dimension) update scaling.
/// Rank-2 matrices use the GPU NS5 path. Other ranks use the fused GPU momentum
/// path. The operation never executes a CPU update or another optimizer.
/// https://kellerjordan.github.io/posts/muon/  https://arxiv.org/html/2502.16982v1
void muonStep(
	oa::Matrix& inOutParam,
	oa::Matrix& inOutMomentum,
	const oa::Matrix& inGrad,
	oa::F32 inLr,
	oa::F32 inBeta,
	oa::F32 inWeightDecay,
	oa::F32 inEps,
	oa::I32 inNS5Iterations
);

/// GPU-only gradient norm clipping (torch.nn.utils.clip_grad_norm_ equivalent).
/// Records two dispatches through the active engine — no CPU/GPU sync required.
///
/// inGrads     : span of grad matrices (max 16; skip empty ones automatically)
/// inMaxNorm   : clipping threshold (e.g. 1.0f)
/// inOutParams : persistent uint[17] scratch buffer (alloc once, reuse every step)
/// inOutPartials: persistent float[16] scratch buffer (alloc once, reuse every step)
///
/// Both scratch buffers must be allocated with oa::FnMatrix::Zeros before first use.
void clipGradNorm(
    oa::Span<oa::Matrix*> inGrads,
    oa::F32 inMaxNorm,
    oa::Matrix& inOutParams,
    oa::Matrix& inOutPartials
);

} // namespace FnOptim

} // namespace oa
