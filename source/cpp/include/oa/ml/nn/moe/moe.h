// Moe — Mixture of experts (level 1 API)
//
// MoE routes each token to top-k experts dynamically
// architecture: Router → topK selection → Expert FFNs → Weighted combine
//
// Used in GPT-4, GPT-5, Mixtral, etc.
//
// stage 0 additions (anti-collapse + specialization, all inert at defaults):
//   • route-utilization telemetry (MoeRouteStats)
//   • aux-loss-free load balancing (DeepSeek-V3 per-expert routing bias)
//   • optional switch aux loss + router z-loss (differentiable, opt-in)
//   • shared always-on experts (DeepSeekMoE)
// The dense-oracle forward is unchanged when balancing bias is zero, shared
// experts are zero, and both loss coefficients are zero — so the finite-diff
// gradchecks that pin correctness cannot regress.

#pragma once

#include <oa/ml/module.h>

namespace oa {

class Linear;
class RmsNorm;
class Swiglu;

// route-utilization telemetry from the last forward. Read after execute+Sync,
// exactly like lastSelectionMask(). This is the instrument that makes expert
// collapse observable — with no balancing, a MoE can silently route everything
// to one expert (harmless while the oracle runs every expert densely, fatal the
// moment the sparse executor lands).
struct MoeRouteStats {
	oa::Vec<oa::F32> loadFraction;   // [E] fraction of (token×slot) assignments; sums to 1
	oa::Vec<oa::F32> meanProb;       // [E] mean router softmax probability per expert
	oa::F32 entropy = 0.0F;        // normalized load entropy in [0,1]; 1 = perfectly balanced
	oa::F32 maxLoadRatio = 0.0F;   // E·max_e load; 1 = balanced, →E = full collapse
	oa::I32 deadExperts = 0;       // experts receiving zero tokens
};

class Moe : public oa::Module {
public:
	Moe() = default;
	Moe(oa::I32 inDModel, oa::I32 inDFF, oa::I32 inNumExperts, oa::I32 inExpertsPerToken,
		oa::F32 inRmsEps = 1e-5f, oa::I32 inNumSharedExperts = 0);

	void init(oa::I32 inDModel, oa::I32 inDFF, oa::I32 inNumExperts, oa::I32 inExpertsPerToken,
		oa::F32 inRmsEps = 1e-5f, oa::I32 inNumSharedExperts = 0);

	// forward: x[T,D] → moe_out[T,D]
	oa::Matrix forward(const oa::Matrix& inX) override;

	[[nodiscard]] oa::I32 dModel() const { return dModel_; }
	[[nodiscard]] oa::I32 dFfn() const { return dFF_; }
	[[nodiscard]] oa::I32 numExperts() const { return numExperts_; }
	[[nodiscard]] oa::I32 expertsPerToken() const { return expertsPerToken_; }
	[[nodiscard]] oa::I32 numSharedExperts() const { return numSharedExperts_; }
	[[nodiscard]] const oa::Matrix& lastSelectionMask() const { return lastSelectionMask_; }
	void setSparseExecution(bool inEnabled) { sparseExecution_ = inEnabled; }
	[[nodiscard]] bool sparseExecution() const { return sparseExecution_; }

	// ── route telemetry (read after execute+Sync) ──────────────────────────────
	[[nodiscard]] MoeRouteStats routeStats() const;

	// ── Aux-loss-free load balancing (DeepSeek-V3) ─────────────────────────────
	// A per-expert bias added to the routing logits for the top-k SELECTION
	// decision only (never into the gate magnitude, so it does not distort the
	// weighted combine and produces no gradient). After each optimizer step, call
	// updateRoutingBias() to nudge under-loaded experts up and over-loaded ones
	// down. The update and next forward remain in the deferred GPU graph.
	void setBalanceRate(oa::F32 inGamma) { balanceGamma_ = inGamma > 0.0F ? inGamma : 0.0F; }
	[[nodiscard]] oa::F32 balanceRate() const { return balanceGamma_; }
	void updateRoutingBias();  // once per training step; queues a GPU update
	[[nodiscard]] oa::F32 routingBias(oa::I32 inExpert) const;

	// ── Optional differentiable balancing losses (opt-in, default off) ─────────
	// Switch/GShard aux loss  α·E·Σ_e f_e·p_e  (f = hard load fraction, constant;
	// P = mean router prob, differentiable) plus router z-loss  β·mean(LSE²).
	// Add auxLoss() to the task loss before backward. Both coefficients 0 ⇒
	// auxLoss() is a 0 scalar recorded on the tape with no gradient effect.
	void setAuxLossAlpha(oa::F32 inAlpha) { auxAlpha_ = inAlpha > 0.0F ? inAlpha : 0.0F; }
	void setRouterZLossBeta(oa::F32 inBeta) { zBeta_ = inBeta > 0.0F ? inBeta : 0.0F; }
	[[nodiscard]] const oa::Matrix& auxLoss() const { return lastAuxLoss_; }

private:
	// GPU-reduced per-expert load fraction of the last forward. Explicit telemetry
	// synchronizes and reads only E reduced scalars, never the [T,E] route mask.
	[[nodiscard]] oa::Vec<oa::F32> lastLoadFraction() const;
	[[nodiscard]] oa::Matrix denseExpertDelta(const oa::Matrix& inNormed,
		const oa::Matrix& inGate) const;
	[[nodiscard]] oa::Matrix sparseExpertDelta(const oa::Matrix& inNormed,
		const oa::Matrix& inRouteGate, const oa::Matrix& inTopKIndices) const;
	[[nodiscard]] const oa::Matrix& expertGateUpWeight() const { return params_[0].data; }
	[[nodiscard]] const oa::Matrix& expertGateUpBias()   const { return params_[1].data; }
	[[nodiscard]] const oa::Matrix& expertDownWeight()   const { return params_[2].data; }
	[[nodiscard]] const oa::Matrix& expertDownBias()     const { return params_[3].data; }

	oa::I32 dModel_ = 0;
	oa::I32 dFF_ = 0;
	oa::I32 numExperts_ = 0;
	oa::I32 expertsPerToken_ = 0;
	oa::I32 numSharedExperts_ = 0;
	oa::F32 rmsEps_ = 1e-5f;
	bool sparseExecution_ = true;

	// One shared pre-norm feeds both router and experts.
	oa::SharedPtr<oa::RmsNorm> norm_;

	// Router: projects normalized input to expert logits.
	oa::SharedPtr<oa::Linear> router_;  // [D, numExperts]

	// Routed experts are native stacked parameters:
	//   gate_up_weight [E,2*DFF,D], gate_up_bias [E,2*DFF]
	//   down_weight    [E,D,DFF],   down_bias    [E,D]
	// Grouped sparse compute consumes these tensors directly. The former layout
	// registered E child modules and rebuilt these same stacks with four Concat
	// operations every forward, then split all four gradients again in backward.
	// Shared always-on experts (DeepSeekMoE): applied unconditionally, no gate.
	oa::Vec<oa::SharedPtr<oa::Swiglu>> sharedExperts_;

	// Aux-loss-free balancing state (device buffer + host policy scalar).
	oa::Matrix routingBias_;        // persistent [1,E] selection bias, gradient-free
	oa::F32 balanceGamma_ = 0.0f;   // nudge rate; opt-in, 0 disables

	// Differentiable balancing coefficients (opt-in).
	oa::F32 auxAlpha_ = 0.0f;
	oa::F32 zBeta_ = 0.0f;

	oa::Matrix lastSelectionMask_;  // [T,E] 0/1 membership
	oa::Matrix lastGateProbs_;      // [T,E] router softmax (telemetry)
	oa::Matrix lastGate_;           // sparse [T,K] route weights or dense-oracle [T,E] gate
	oa::Matrix lastAuxLoss_;        // scalar, on the tape (0 when disabled)
	oa::Matrix zeroAuxLoss_;        // stable scalar returned when aux losses are disabled
};

} // namespace oa
