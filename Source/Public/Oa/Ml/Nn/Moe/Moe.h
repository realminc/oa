// OaMoE — Mixture of Experts (Level 1 API)
//
// MoE routes each token to top-k experts dynamically
// Architecture: Router → TopK selection → Expert FFNs → Weighted combine
//
// Used in GPT-4, GPT-5, Mixtral, etc.
//
// Stage 0 additions (anti-collapse + specialization, all inert at defaults):
//   • route-utilization telemetry (OaMoeRouteStats)
//   • aux-loss-free load balancing (DeepSeek-V3 per-expert routing bias)
//   • optional switch aux loss + router z-loss (differentiable, opt-in)
//   • shared always-on experts (DeepSeekMoE)
// The dense-oracle forward is unchanged when balancing bias is zero, shared
// experts are zero, and both loss coefficients are zero — so the finite-diff
// gradchecks that pin correctness cannot regress.

#pragma once

#include <Oa/Ml/Module.h>

class OaLinear;
class OaRmsNorm;

// One expert MLP. Normalization and the residual connection belong to the MoE
// layer, not to each expert: selected experts all consume the same normalized
// token and their weighted deltas are added to the residual exactly once.
class OaMoeExpert : public OaModule {
public:
	OaMoeExpert() = default;
	OaMoeExpert(OaI32 InDModel, OaI32 InDFF);

	void Init(OaI32 InDModel, OaI32 InDFF);
	OaMatrix Forward(const OaMatrix& InX) override;
	[[nodiscard]] const OaMatrix& GateUpWeight() const;
	[[nodiscard]] const OaMatrix& GateUpBias() const;
	[[nodiscard]] const OaMatrix& DownWeight() const;
	[[nodiscard]] const OaMatrix& DownBias() const;

private:
	OaI32 DFF_ = 0;
	OaSharedPtr<OaLinear> GateUp_;
	OaSharedPtr<OaLinear> Down_;
};

// Route-utilization telemetry from the last Forward. Read after Execute+Sync,
// exactly like LastSelectionMask(). This is the instrument that makes expert
// collapse observable — with no balancing, a MoE can silently route everything
// to one expert (harmless while the oracle runs every expert densely, fatal the
// moment the sparse executor lands).
struct OaMoeRouteStats {
	OaVec<OaF32> LoadFraction;   // [E] fraction of (token×slot) assignments; sums to 1
	OaVec<OaF32> MeanProb;       // [E] mean router softmax probability per expert
	OaF32 Entropy = 0.0F;        // normalized load entropy in [0,1]; 1 = perfectly balanced
	OaF32 MaxLoadRatio = 0.0F;   // E·max_e load; 1 = balanced, →E = full collapse
	OaI32 DeadExperts = 0;       // experts receiving zero tokens
};

class OaMoE : public OaModule {
public:
	OaMoE() = default;
	OaMoE(OaI32 InDModel, OaI32 InDFF, OaI32 InNumExperts, OaI32 InExpertsPerToken,
		OaF32 InRmsEps = 1e-5f, OaI32 InNumSharedExperts = 0);

	void Init(OaI32 InDModel, OaI32 InDFF, OaI32 InNumExperts, OaI32 InExpertsPerToken,
		OaF32 InRmsEps = 1e-5f, OaI32 InNumSharedExperts = 0);

	// Forward: x[T,D] → moe_out[T,D]
	OaMatrix Forward(const OaMatrix& InX) override;

	[[nodiscard]] OaI32 DModel() const { return DModel_; }
	[[nodiscard]] OaI32 DFF() const { return DFF_; }
	[[nodiscard]] OaI32 NumExperts() const { return NumExperts_; }
	[[nodiscard]] OaI32 ExpertsPerToken() const { return ExpertsPerToken_; }
	[[nodiscard]] OaI32 NumSharedExperts() const { return NumSharedExperts_; }
	[[nodiscard]] const OaMatrix& LastSelectionMask() const { return LastSelectionMask_; }
	void SetSparseExecution(bool InEnabled) { SparseExecution_ = InEnabled; }
	[[nodiscard]] bool SparseExecution() const { return SparseExecution_; }

	// ── Route telemetry (read after Execute+Sync) ──────────────────────────────
	[[nodiscard]] OaMoeRouteStats RouteStats() const;

	// ── Aux-loss-free load balancing (DeepSeek-V3) ─────────────────────────────
	// A per-expert bias added to the routing logits for the top-k SELECTION
	// decision only (never into the gate magnitude, so it does not distort the
	// weighted combine and produces no gradient). After each optimizer step, call
	// UpdateRoutingBias() to nudge under-loaded experts up and over-loaded ones
	// down. The update and next forward remain in the deferred GPU graph.
	void SetBalanceRate(OaF32 InGamma) { BalanceGamma_ = InGamma > 0.0F ? InGamma : 0.0F; }
	[[nodiscard]] OaF32 BalanceRate() const { return BalanceGamma_; }
	void UpdateRoutingBias();  // once per training step; queues a GPU update
	[[nodiscard]] OaF32 RoutingBias(OaI32 InExpert) const;

	// ── Optional differentiable balancing losses (opt-in, default off) ─────────
	// Switch/GShard aux loss  α·E·Σ_e f_e·P_e  (f = hard load fraction, constant;
	// P = mean router prob, differentiable) plus router z-loss  β·mean(LSE²).
	// Add AuxLoss() to the task loss before Backward. Both coefficients 0 ⇒
	// AuxLoss() is a 0 scalar recorded on the tape with no gradient effect.
	void SetAuxLossAlpha(OaF32 InAlpha) { AuxAlpha_ = InAlpha > 0.0F ? InAlpha : 0.0F; }
	void SetRouterZLossBeta(OaF32 InBeta) { ZBeta_ = InBeta > 0.0F ? InBeta : 0.0F; }
	[[nodiscard]] const OaMatrix& AuxLoss() const { return LastAuxLoss_; }

private:
	// GPU-reduced per-expert load fraction of the last Forward. Explicit telemetry
	// synchronizes and reads only E reduced scalars, never the [T,E] route mask.
	[[nodiscard]] OaVec<OaF32> LastLoadFraction() const;
	[[nodiscard]] OaMatrix DenseExpertDelta(const OaMatrix& InNormed,
		const OaMatrix& InGate) const;
	[[nodiscard]] OaMatrix SparseExpertDelta(const OaMatrix& InNormed,
		const OaMatrix& InRouteGate, const OaMatrix& InTopKIndices) const;
	[[nodiscard]] const OaMatrix& ExpertGateUpWeight() const { return Params_[0].Data; }
	[[nodiscard]] const OaMatrix& ExpertGateUpBias() const { return Params_[1].Data; }
	[[nodiscard]] const OaMatrix& ExpertDownWeight() const { return Params_[2].Data; }
	[[nodiscard]] const OaMatrix& ExpertDownBias() const { return Params_[3].Data; }

	OaI32 DModel_ = 0;
	OaI32 DFF_ = 0;
	OaI32 NumExperts_ = 0;
	OaI32 ExpertsPerToken_ = 0;
	OaI32 NumSharedExperts_ = 0;
	OaF32 RmsEps_ = 1e-5f;
	bool SparseExecution_ = true;

	// One shared pre-norm feeds both router and experts.
	OaSharedPtr<OaRmsNorm> Norm_;

	// Router: projects normalized input to expert logits.
	OaSharedPtr<OaLinear> Router_;  // [D, NumExperts]

	// Routed experts are native stacked parameters:
	//   gate_up_weight [E,2*DFF,D], gate_up_bias [E,2*DFF]
	//   down_weight    [E,D,DFF],   down_bias    [E,D]
	// Grouped sparse compute consumes these tensors directly. The former layout
	// registered E child modules and rebuilt these same stacks with four Concat
	// operations every forward, then split all four gradients again in backward.
	// Shared always-on experts (DeepSeekMoE): applied unconditionally, no gate.
	OaVec<OaSharedPtr<OaMoeExpert>> SharedExperts_;

	// Aux-loss-free balancing state (device buffer + host policy scalar).
	OaMatrix RoutingBias_;        // persistent [1,E] selection bias, gradient-free
	OaF32 BalanceGamma_ = 0.0f;   // nudge rate; opt-in, 0 disables

	// Differentiable balancing coefficients (opt-in).
	OaF32 AuxAlpha_ = 0.0f;
	OaF32 ZBeta_ = 0.0f;

	OaMatrix LastSelectionMask_;  // [T,E] 0/1 membership
	OaMatrix LastGateProbs_;      // [T,E] router softmax (telemetry)
	OaMatrix LastGate_;           // sparse [T,K] route weights or dense-oracle [T,E] gate
	OaMatrix LastAuxLoss_;        // scalar, on the tape (0 when disabled)
	OaMatrix ZeroAuxLoss_;        // stable scalar returned when aux losses are disabled
};
