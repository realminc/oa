// OA ML - Optimizer Operations
//
// Stateless optimizer step functions for training.
// Records into OaContext (clean api lvl1). One overload per op — no graph /
// executor variants. Mutates param / momentum / m / v in-place.
//
// Design (mirrors OaFnMatrix / OaFnLoss):
//   OaFnMatrix   → Forward operations (Relu, Matmul, Add, …)
//   OaFnLoss     → Loss functions (CrossEntropy, Mse, …)
//   OaFnOptim    → Optimizer steps (AdamWStep, SgdStep, …)

#pragma once

#include <Oa/Core/Matrix.h>

namespace OaFnOptim {

struct OaAdamWParamSet {
	OaMatrix* Param;
	OaMatrix* M;
	OaMatrix* V;
	const OaMatrix* Grad;
};

/// Advance the mutable replay state immediately before replay-safe AdamW
/// updates. The host seeds state[0] with the previous completed step; this
/// one-thread GPU op makes each replay consume a distinct optimizer step.
void AdamWAdvanceGraphState(OaMatrix& InOutState);

/// AdamW optimizer step: param -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * param)
void AdamWStep(
	OaMatrix& InOutParam,
	OaMatrix& InOutM,
	OaMatrix& InOutV,
	const OaMatrix& InGrad,
	OaF32 InLr,
	OaF32 InBeta1,
	OaF32 InBeta2,
	OaF32 InEps,
	OaF32 InWeightDecay,
	OaI32 InStep
);

/// Replay-safe AdamW step. Mutable optimizer scalars are read from InState:
/// uint32[6] = {step, bitcast(lr), bitcast(beta1), bitcast(beta2),
/// bitcast(eps), bitcast(weight_decay)}. Only Count remains in push constants.
void AdamWStepGraph(
	OaMatrix& InOutParam,
	OaMatrix& InOutM,
	OaMatrix& InOutV,
	const OaMatrix& InGrad,
	const OaMatrix& InState
);

/// AdamW optimizer step for multiple parameter tensors.
/// Currently fuses the common four-tensor case into one dispatch and falls back
/// to AdamWStep for other counts.
void AdamWStepMany(
	OaSpan<const OaAdamWParamSet> InParams,
	OaF32 InLr,
	OaF32 InBeta1,
	OaF32 InBeta2,
	OaF32 InEps,
	OaF32 InWeightDecay,
	OaI32 InStep
);

/// Replay-safe fused four-parameter AdamW step using the same state contract as
/// AdamWStepGraph. Other parameter counts fall back to per-parameter graph ops.
void AdamWStepManyGraph(
	OaSpan<const OaAdamWParamSet> InParams,
	const OaMatrix& InState
);

/// SGD optimizer step with optional momentum and weight decay.
/// Pass an empty InOutMomentum (or InMomentum == 0) for plain SGD.
void SgdStep(
	OaMatrix& InOutParam,
	OaMatrix& InOutMomentum,
	const OaMatrix& InGrad,
	OaF32 InLr,
	OaF32 InMomentum,
	OaF32 InWeightDecay
);

/// Muon optimizer step (OaMuonRef: Nesterov + NS5 + Moonshot 0.2*sqrt(max dim) scaling).
/// 2D matrices: GPU NS5 via OaFnMatrix MatMul pipeline (same as AdamW — records into OaContext).
/// 1D: MuonVector GPU kernel. Route embed/head/1D to AdamW via OaOptimizerComposite.
/// https://kellerjordan.github.io/posts/muon/  https://arxiv.org/html/2502.16982v1
void MuonStep(
	OaMatrix& InOutParam,
	OaMatrix& InOutMomentum,
	const OaMatrix& InGrad,
	OaF32 InLr,
	OaF32 InBeta,
	OaF32 InWeightDecay,
	OaF32 InEps,
	OaI32 InNS5Iterations
);

/// GPU-only gradient norm clipping (torch.nn.utils.clip_grad_norm_ equivalent).
/// Records two dispatches into the active OaContext — no CPU/GPU sync required.
///
/// InGrads     : span of grad matrices (max 16; skip empty ones automatically)
/// InMaxNorm   : clipping threshold (e.g. 1.0f)
/// InOutParams : persistent uint[17] scratch buffer (alloc once, reuse every step)
/// InOutPartials: persistent float[16] scratch buffer (alloc once, reuse every step)
///
/// Both scratch buffers must be allocated with OaFnMatrix::Zeros before first use.
void ClipGradNorm(
    OaSpan<OaMatrix*> InGrads,
    OaF32 InMaxNorm,
    OaMatrix& InOutParams,
    OaMatrix& InOutPartials
);

} // namespace OaFnOptim
