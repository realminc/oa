// OA ML - Optimizers
//
// PyTorch torch.optim equivalent.
// Step() applies gradients, ZeroGrad() clears them.

#pragma once

#include <Oa/Ml/Module.h>

class OamModel;

// BASE OPTIMIZER

class OaOptimizer {
public:
	virtual ~OaOptimizer() = default;

	/// Apply accumulated gradients to parameters
	virtual void Step() = 0;

	/// Describe optimizer step for compiled graph mode
	/// Default: falls back to immediate Step() (works but not optimal)
	/// Override to add operations to InGraph for better performance
	virtual void DescribeStep(class OaComputeGraph& InGraph, class OaComputeEngine& InRt) {
		(void)InGraph;
		(void)InRt;
		Step();  // Fallback to immediate dispatch
	}

	/// Zero all gradients
	virtual void ZeroGrad() = 0;

	/// Get/set learning rate
	virtual void SetLr(OaF32 InLr) { Lr_ = InLr; }
	OaF32 Lr() const { return Lr_; }
	[[nodiscard]] OaF32 GetLr() const { return Lr_; }

	/// Current step count
	[[nodiscard]] OaU64 GetStep() const { return Step_; }

	// Persistence — write/read optimizer state (moments, step count, hyperparams)
	// into an OamModel section. Default no-op so SGD/Adam compile until they implement.
	// Combined with OaModule::Save/Load(path, optimizer) for one-call full checkpoints.
	virtual void SaveTo(OamModel& OutOam) const { (void)OutOam; }
	virtual void LoadFrom(const OamModel& InOam) { (void)InOam; }

protected:
	OaVec<OaParameter*> Params_;
	OaF32 Lr_ = 1e-3f;
	OaU64 Step_ = 0;

	// ── Mixed-precision master weights (shared by every optimizer) ──────────────
	// For each low-precision (bf16/fp16) parameter, keep an fp32 master accumulator.
	// The optimizer math runs entirely in fp32 on the master (grads are upcast); the
	// bf16 weight the forward pass reads is re-derived as round(master) each step.
	// fp32 params keep an empty Master_ slot and update in place. This gives every
	// optimizer bf16-correct convergence with ZERO precision-aware optimizer kernels
	// — the only precision logic is a Cast at the boundary. See OaPrecisionDtype.md §5.
	OaVec<OaMatrix> Master_;      // fp32 master per param; empty slot ⇒ param is fp32
	OaVec<OaMatrix> GradF32_;     // fp32 upcast-grad scratch per param
	bool MixedReady_ = false;     // Master_/GradF32_ sized
	bool MasterSeeded_ = false;   // masters seeded from current weights (clear on load)

	// Call once at the top of Step() (after ++Step_): allocates masters for
	// low-precision params and seeds them from the current weights on first use.
	void PrepareMasters();
	// True if param InIdx is low-precision and therefore has an fp32 master.
	[[nodiscard]] bool HasMaster(OaUsize InIdx) const;
	// The buffer the optimizer math should update (fp32 master if low-precision).
	[[nodiscard]] OaMatrix& MasterOrData(OaUsize InIdx);
	// The grad the optimizer math should read (fp32 upcast for low-precision params,
	// recorded into GradF32_[InIdx]; else InGrad unchanged).
	[[nodiscard]] OaMatrix MasterGrad(OaUsize InIdx, const OaMatrix& InGrad);
	// Re-derive the bf16 weight from its fp32 master (no-op for fp32 params).
	void WritebackMaster(OaUsize InIdx);
	// Clear the seeded flag so masters re-init from weights after a checkpoint load.
	void ResetMasterSeed() { MasterSeeded_ = false; }
};

// NO-OP — for callers that pass an OaOptimizer& but actually update params
// themselves (e.g. hand-rolled training tutorials wiring OaItTraining purely
// for its cadence + callbacks). Step() and ZeroGrad() are intentional no-ops.

class OaOptimizerNoOp : public OaOptimizer {
public:
	void Step() override {}
	void ZeroGrad() override {}
};

// COMPOSITE — run multiple optimizers on disjoint param sets (e.g. Muon + AdamW).
// Not a fused "mixed optimizer": each child owns its params and state independently.

class OaOptimizerComposite : public OaOptimizer {
public:
	void Add(OaUniquePtr<OaOptimizer> InOptimizer);

	/// Record base LRs for schedule scaling (AdamW schedule × Muon proportionally).
	void InitDualLr(OaF32 InMuonLr, OaF32 InAdamWLr);

	void SetLr(OaF32 InLr) override;
	void Step() override;
	void ZeroGrad() override;
	void SaveTo(OamModel& OutOam) const override;
	void LoadFrom(const OamModel& InOam) override;

private:
	OaVec<OaUniquePtr<OaOptimizer>> Children_;
	OaF32 BaseMuonLr_ = 2e-2f;
	OaF32 BaseAdamWLr_ = 3e-4f;
};

// SGD

class OaSGD : public OaOptimizer {
public:
	OaSGD(OaVec<OaParameter>& InParams, OaF32 InLr = 1e-2f, OaF32 InMomentum = 0.0f, OaF32 InWeightDecay = 0.0f)
		: Momentum_(InMomentum), WeightDecay_(InWeightDecay) {
		Lr_ = InLr;
		for (auto& p : InParams) if (p.RequiresGrad) Params_.PushBack(&p);
	}
	OaSGD(OaSpan<OaParameter*> InParamPtrs, OaF32 InLr = 1e-2f, OaF32 InMomentum = 0.0f, OaF32 InWeightDecay = 0.0f)
		: Momentum_(InMomentum), WeightDecay_(InWeightDecay) {
		Lr_ = InLr;
		for (auto* p : InParamPtrs) if (p && p->RequiresGrad) Params_.PushBack(p);
	}
	OaSGD(OaVec<OaParameter*>& InParamPtrs, OaF32 InLr = 1e-2f, OaF32 InMomentum = 0.0f, OaF32 InWeightDecay = 0.0f)
		: Momentum_(InMomentum), WeightDecay_(InWeightDecay) {
		Lr_ = InLr;
		for (auto* p : InParamPtrs) if (p && p->RequiresGrad) Params_.PushBack(p);
	}

	void Step() override;
	void ZeroGrad() override;
	void SaveTo(OamModel& OutOam) const override;
	void LoadFrom(const OamModel& InOam) override;

private:
	OaF32 Momentum_;
	OaF32 WeightDecay_;
};

// ADAM

class OaAdam : public OaOptimizer {
public:
	OaAdam(OaVec<OaParameter>& InParams, OaF32 InLr = 1e-3f, OaF32 InBeta1 = 0.9f, OaF32 InBeta2 = 0.999f, OaF32 InEps = 1e-8f)
		: Beta1_(InBeta1), Beta2_(InBeta2), Eps_(InEps) {
		Lr_ = InLr;
		for (auto& p : InParams) if (p.RequiresGrad) Params_.PushBack(&p);
	}
	OaAdam(OaSpan<OaParameter*> InParamPtrs, OaF32 InLr = 1e-3f, OaF32 InBeta1 = 0.9f, OaF32 InBeta2 = 0.999f, OaF32 InEps = 1e-8f)
		: Beta1_(InBeta1), Beta2_(InBeta2), Eps_(InEps) {
		Lr_ = InLr;
		for (auto* p : InParamPtrs) if (p && p->RequiresGrad) Params_.PushBack(p);
	}
	OaAdam(OaVec<OaParameter*>& InParamPtrs, OaF32 InLr = 1e-3f, OaF32 InBeta1 = 0.9f, OaF32 InBeta2 = 0.999f, OaF32 InEps = 1e-8f)
		: Beta1_(InBeta1), Beta2_(InBeta2), Eps_(InEps) {
		Lr_ = InLr;
		for (auto* p : InParamPtrs) if (p && p->RequiresGrad) Params_.PushBack(p);
	}

	void Step() override;
	void ZeroGrad() override;
	void SaveTo(OamModel& OutOam) const override;
	void LoadFrom(const OamModel& InOam) override;

private:
	OaF32 Beta1_, Beta2_, Eps_;
	OaVec<OaMatrix> M_;  // First moment estimates
	OaVec<OaMatrix> V_;  // Second moment estimates
};

// ADAMW (decoupled weight decay — preferred for transformers)

class OaAdamW : public OaOptimizer {
public:
	OaAdamW(OaVec<OaParameter>& InParams, OaF32 InLr = 1e-3f, OaF32 InBeta1 = 0.9f, OaF32 InBeta2 = 0.999f,
		OaF32 InEps = 1e-8f, OaF32 InWeightDecay = 0.01f)
		: Beta1_(InBeta1), Beta2_(InBeta2), Eps_(InEps), WeightDecay_(InWeightDecay) {
		Lr_ = InLr;
		for (auto& p : InParams) if (p.RequiresGrad) Params_.PushBack(&p);
	}

	// Composite modules (e.g. OaGpt2): root Params_ is often empty; pass pointers from subtree.
	OaAdamW(OaSpan<OaParameter*> InParamPtrs, OaF32 InLr = 1e-3f, OaF32 InBeta1 = 0.9f, OaF32 InBeta2 = 0.999f,
		OaF32 InEps = 1e-8f, OaF32 InWeightDecay = 0.01f)
		: Beta1_(InBeta1), Beta2_(InBeta2), Eps_(InEps), WeightDecay_(InWeightDecay) {
		Lr_ = InLr;
		for (OaParameter* paramPtr : InParamPtrs) {
			if (paramPtr && paramPtr->RequiresGrad) Params_.PushBack(paramPtr);
		}
	}
	OaAdamW(OaVec<OaParameter*>& InParamPtrs, OaF32 InLr = 1e-3f, OaF32 InBeta1 = 0.9f, OaF32 InBeta2 = 0.999f,
		OaF32 InEps = 1e-8f, OaF32 InWeightDecay = 0.01f)
		: Beta1_(InBeta1), Beta2_(InBeta2), Eps_(InEps), WeightDecay_(InWeightDecay) {
		Lr_ = InLr;
		for (auto* p : InParamPtrs) if (p && p->RequiresGrad) Params_.PushBack(p);
	}

	void Step() override;
	void ZeroGrad() override;
	void SaveTo(OamModel& OutOam) const override;
	void LoadFrom(const OamModel& InOam) override;

private:
	OaF32 Beta1_, Beta2_, Eps_, WeightDecay_;
	OaVec<OaMatrix> M_;
	OaVec<OaMatrix> V_;
};

// MUON — Momentum + Newton-Schulz5 Orthogonalization Optimizer
// https://arxiv.org/abs/2502.16982  (Moonshot "Muon is Scalable...") + https://kellerjordan.github.io/posts/muon/
// Official usage: 2D hidden-layer matrices get momentum followed by NS5 ortho (tuned quintic coeffs).
// Embeddings, final heads, vectors/scalars/biases stay on AdamW (or equivalent) for best results.
// Benefits: ~50% less optimizer state VRAM vs AdamW (single momentum buffer), norm-preserving updates.
// 2D path: GPU NS5 (MatMul + MuonNesterov/MuonApply). 1D: MuonVector. Pair with OaAdamW via OaOptimizerComposite.

class OaMuon : public OaOptimizer {
public:
	OaMuon(OaVec<OaParameter>& InParams, OaF32 InLr = 1e-3f, OaF32 InBeta = 0.95f,
		OaF32 InWeightDecay = 0.1f, OaF32 InEps = 1e-8f, OaI32 InNS5Iterations = 5)
		: Beta_(InBeta), WeightDecay_(InWeightDecay), Eps_(InEps), NS5Iterations_(InNS5Iterations) {
		Lr_ = InLr;
		for (auto& p : InParams) if (p.RequiresGrad) Params_.PushBack(&p);
	}

	OaMuon(OaSpan<OaParameter*> InParamPtrs, OaF32 InLr = 1e-3f, OaF32 InBeta = 0.95f,
		OaF32 InWeightDecay = 0.1f, OaF32 InEps = 1e-8f, OaI32 InNS5Iterations = 5)
		: Beta_(InBeta), WeightDecay_(InWeightDecay), Eps_(InEps), NS5Iterations_(InNS5Iterations) {
		Lr_ = InLr;
		for (OaParameter* paramPtr : InParamPtrs) {
			if (paramPtr && paramPtr->RequiresGrad) Params_.PushBack(paramPtr);
		}
	}
	OaMuon(OaVec<OaParameter*>& InParamPtrs, OaF32 InLr = 1e-3f, OaF32 InBeta = 0.95f,
		OaF32 InWeightDecay = 0.1f, OaF32 InEps = 1e-8f, OaI32 InNS5Iterations = 5)
		: Beta_(InBeta), WeightDecay_(InWeightDecay), Eps_(InEps), NS5Iterations_(InNS5Iterations) {
		Lr_ = InLr;
		for (auto* p : InParamPtrs) if (p && p->RequiresGrad) Params_.PushBack(p);
	}

	void Step() override;
	void ZeroGrad() override;
	void SaveTo(OamModel& OutOam) const override;
	void LoadFrom(const OamModel& InOam) override;

private:
	OaF32 Beta_, WeightDecay_, Eps_;
	OaI32 NS5Iterations_;
	OaVec<OaMatrix> Momentum_;  // First moment (momentum buffer)
};
