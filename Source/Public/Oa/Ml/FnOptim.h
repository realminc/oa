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
