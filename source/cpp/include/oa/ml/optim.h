// oa:: ML Optimizers
//
// PyTorch torch.optim equivalent.
// step() applies gradients, zeroGrad() clears them.

#pragma once

#include <oa/ml/module.h>

namespace oa {

class Engine;
class ModelFile;

// ─── base Optimizer ──────────────────────────────────────────────────────────

class Optimizer {
public:
	virtual ~Optimizer() = default;

	/// apply accumulated gradients to parameters.
	virtual void step() = 0;

	/// Zero all gradients.
	virtual void zeroGrad() = 0;

	/// get/set learning rate.
	virtual void setLr(oa::F32 inLr) { lr_ = inLr; }
	[[nodiscard]] oa::F32 lr() const { return lr_; }
	[[nodiscard]] oa::F32 getLr() const { return lr_; }

	/// current step count.
	[[nodiscard]] oa::U64 getStep() const { return step_; }

	// A compiled training program executes the already-recorded optimizer kernels
	// without calling step() again. Keep the host-visible logical step aligned so
	// schedules and checkpoints observe the same state as eager execution.
	virtual void notifyProgramReplay(oa::U64 inCount = 1) { step_ += inCount; }

	// Persistence — write/read optimizer state (moments, step count, hyperparams)
	// into a ModelFile section. Default no-op so SGD/Adam compile until they implement.
	[[nodiscard]] virtual oa::Status saveTo(Engine& inEngine, ModelFile& outFile) const {
		(void)inEngine; (void)outFile; return oa::Status::ok();
	}
	[[nodiscard]] virtual oa::Status validateLoad(const ModelFile& inFile) const {
		(void)inFile; return oa::Status::ok();
	}
	[[nodiscard]] virtual oa::Status loadFrom(Engine& inEngine, const ModelFile& inFile) {
		(void)inEngine; (void)inFile; return oa::Status::ok();
	}

protected:
	oa::Vector<Parameter*> params_;
	oa::F32  lr_    = 1e-3f;
	oa::U64  step_  = 0;

	// ── Mixed-precision master weights (shared by every optimizer) ──────────────
	// For each admitted low-precision (bf16) parameter, keep an fp32 master accumulator.
	// The optimizer math runs entirely in fp32 on the master (grads are upcast); the
	// bf16 weight the forward pass reads is re-derived as round(master) each step.
	// fp32 params keep an empty master_ slot and update in place.
	oa::Vector<Matrix> master_;       // fp32 master per param; empty slot ⇒ param is fp32
	oa::Vector<Matrix> gradF32_;      // fp32 upcast-grad scratch per param
	bool mixedReady_   = false;  // master_/gradF32_ sized
	bool masterSeeded_ = false;  // masters seeded from current weights (clear on load)

	// call once at the top of step() (after ++step_): allocates masters for
	// low-precision params and seeds them from the current weights on first use.
	void prepareMasters();
	[[nodiscard]] bool    hasMaster(oa::Usize inIdx) const;
	[[nodiscard]] Matrix& masterOrData(oa::Usize inIdx);
	[[nodiscard]] Matrix  masterGrad(oa::Usize inIdx, const Matrix& inGrad);
	void writebackMaster(oa::Usize inIdx);
	void resetMasterSeed() { masterSeeded_ = false; }
};

// ─── No-op optimizer ─────────────────────────────────────────────────────────
// For callers that pass an Optimizer& but update params themselves
// (e.g. hand-rolled training tutorials using ItTraining purely for cadence + callbacks).

class OptimizerNoOp : public Optimizer {
public:
	void step() override {}
	void zeroGrad() override {}
	void notifyProgramReplay(oa::U64 inCount = 1) override { (void)inCount; }
};

// ─── SGD ─────────────────────────────────────────────────────────────────────

class Sgd : public Optimizer {
public:
	Sgd(oa::Vector<Parameter>& inParams, oa::F32 inLr = 1e-2f, oa::F32 inMomentum = 0.0f, oa::F32 inWeightDecay = 0.0f)
		: momentum_(inMomentum), weightDecay_(inWeightDecay) {
		lr_ = inLr;
		for (auto& p : inParams) if (p.requiresGrad) params_.pushBack(&p);
	}
	Sgd(oa::Span<Parameter*> inParamPtrs, oa::F32 inLr = 1e-2f, oa::F32 inMomentum = 0.0f, oa::F32 inWeightDecay = 0.0f)
		: momentum_(inMomentum), weightDecay_(inWeightDecay) {
		lr_ = inLr;
		for (auto* p : inParamPtrs) if (p && p->requiresGrad) params_.pushBack(p);
	}
	Sgd(oa::Vector<Parameter*>& inParamPtrs, oa::F32 inLr = 1e-2f, oa::F32 inMomentum = 0.0f, oa::F32 inWeightDecay = 0.0f)
		: momentum_(inMomentum), weightDecay_(inWeightDecay) {
		lr_ = inLr;
		for (auto* p : inParamPtrs) if (p && p->requiresGrad) params_.pushBack(p);
	}

	void step() override;
	void zeroGrad() override;
	[[nodiscard]] oa::Status saveTo(Engine& inEngine, ModelFile& outFile) const override;
	[[nodiscard]] oa::Status validateLoad(const ModelFile& inFile) const override;
	[[nodiscard]] oa::Status loadFrom(Engine& inEngine, const ModelFile& inFile) override;

private:
	oa::F32 momentum_;
	oa::F32 weightDecay_;
};

// ─── Adam ────────────────────────────────────────────────────────────────────

class Adam : public Optimizer {
public:
	Adam(oa::Vector<Parameter>& inParams, oa::F32 inLr = 1e-3f, oa::F32 inBeta1 = 0.9f, oa::F32 inBeta2 = 0.999f, oa::F32 inEps = 1e-8f)
		: beta1_(inBeta1), beta2_(inBeta2), eps_(inEps) {
		lr_ = inLr;
		for (auto& p : inParams) if (p.requiresGrad) params_.pushBack(&p);
	}
	Adam(oa::Span<Parameter*> inParamPtrs, oa::F32 inLr = 1e-3f, oa::F32 inBeta1 = 0.9f, oa::F32 inBeta2 = 0.999f, oa::F32 inEps = 1e-8f)
		: beta1_(inBeta1), beta2_(inBeta2), eps_(inEps) {
		lr_ = inLr;
		for (auto* p : inParamPtrs) if (p && p->requiresGrad) params_.pushBack(p);
	}
	Adam(oa::Vector<Parameter*>& inParamPtrs, oa::F32 inLr = 1e-3f, oa::F32 inBeta1 = 0.9f, oa::F32 inBeta2 = 0.999f, oa::F32 inEps = 1e-8f)
		: beta1_(inBeta1), beta2_(inBeta2), eps_(inEps) {
		lr_ = inLr;
		for (auto* p : inParamPtrs) if (p && p->requiresGrad) params_.pushBack(p);
	}

	void step() override;
	void zeroGrad() override;
	[[nodiscard]] oa::Status saveTo(Engine& inEngine, ModelFile& outFile) const override;
	[[nodiscard]] oa::Status validateLoad(const ModelFile& inFile) const override;
	[[nodiscard]] oa::Status loadFrom(Engine& inEngine, const ModelFile& inFile) override;

private:
	oa::F32 beta1_, beta2_, eps_;
	oa::Vector<Matrix> m_;  // first moment estimates
	oa::Vector<Matrix> v_;  // second moment estimates
};

// ─── AdamW ───────────────────────────────────────────────────────────────────
// Decoupled weight decay — preferred for transformers.

class AdamW : public Optimizer {
public:
	AdamW(oa::Vector<Parameter>& inParams, oa::F32 inLr = 1e-3f, oa::F32 inBeta1 = 0.9f, oa::F32 inBeta2 = 0.999f,
		oa::F32 inEps = 1e-8f, oa::F32 inWeightDecay = 0.01f)
		: beta1_(inBeta1), beta2_(inBeta2), eps_(inEps), weightDecay_(inWeightDecay) {
		lr_ = inLr;
		for (auto& p : inParams) if (p.requiresGrad) params_.pushBack(&p);
	}
	// Composite modules: root params_ is often empty; pass pointers from subtree.
	AdamW(oa::Span<Parameter*> inParamPtrs, oa::F32 inLr = 1e-3f, oa::F32 inBeta1 = 0.9f, oa::F32 inBeta2 = 0.999f,
		oa::F32 inEps = 1e-8f, oa::F32 inWeightDecay = 0.01f)
		: beta1_(inBeta1), beta2_(inBeta2), eps_(inEps), weightDecay_(inWeightDecay) {
		lr_ = inLr;
		for (Parameter* p : inParamPtrs) if (p && p->requiresGrad) params_.pushBack(p);
	}
	AdamW(oa::Vector<Parameter*>& inParamPtrs, oa::F32 inLr = 1e-3f, oa::F32 inBeta1 = 0.9f, oa::F32 inBeta2 = 0.999f,
		oa::F32 inEps = 1e-8f, oa::F32 inWeightDecay = 0.01f)
		: beta1_(inBeta1), beta2_(inBeta2), eps_(inEps), weightDecay_(inWeightDecay) {
		lr_ = inLr;
		for (auto* p : inParamPtrs) if (p && p->requiresGrad) params_.pushBack(p);
	}

	void step() override;
	void setLr(oa::F32 inLr) override;
	void zeroGrad() override;
	[[nodiscard]] oa::Status saveTo(Engine& inEngine, ModelFile& outFile) const override;
	[[nodiscard]] oa::Status validateLoad(const ModelFile& inFile) const override;
	[[nodiscard]] oa::Status loadFrom(Engine& inEngine, const ModelFile& inFile) override;

private:
	oa::F32 beta1_, beta2_, eps_, weightDecay_;
	oa::Vector<Matrix> m_;
	oa::Vector<Matrix> v_;
	Matrix        graphState_;         // uint32[6], GPU-stepped replay state
	Engine*       graphStateEngine_ = nullptr;
};

// ─── Muon ────────────────────────────────────────────────────────────────────
// GPU momentum + Newton-Schulz5 orthogonalization optimizer.
// https://github.com/KellerJordan/Muon + https://arxiv.org/abs/2502.16982
// Rank-2 parameters use the orthogonalized Muon update. Other ranks use the
// optimizer's fused GPU momentum update. Muon owns exactly the parameter set
// supplied by the caller and never delegates to another optimizer.

class Muon : public Optimizer {
public:
	Muon(oa::Vector<Parameter>& inParams, oa::F32 inLr = 1e-3f, oa::F32 inBeta = 0.95f,
		oa::F32 inWeightDecay = 0.1f, oa::F32 inEps = 1e-7f, oa::I32 inNs5Iterations = 5)
		: beta_(inBeta), weightDecay_(inWeightDecay), eps_(inEps), ns5Iterations_(inNs5Iterations) {
		lr_ = inLr;
		for (auto& p : inParams) if (p.requiresGrad) params_.pushBack(&p);
	}
	Muon(oa::Span<Parameter*> inParamPtrs, oa::F32 inLr = 1e-3f, oa::F32 inBeta = 0.95f,
		oa::F32 inWeightDecay = 0.1f, oa::F32 inEps = 1e-7f, oa::I32 inNs5Iterations = 5)
		: beta_(inBeta), weightDecay_(inWeightDecay), eps_(inEps), ns5Iterations_(inNs5Iterations) {
		lr_ = inLr;
		for (Parameter* p : inParamPtrs) if (p && p->requiresGrad) params_.pushBack(p);
	}
	Muon(oa::Vector<Parameter*>& inParamPtrs, oa::F32 inLr = 1e-3f, oa::F32 inBeta = 0.95f,
		oa::F32 inWeightDecay = 0.1f, oa::F32 inEps = 1e-7f, oa::I32 inNs5Iterations = 5)
		: beta_(inBeta), weightDecay_(inWeightDecay), eps_(inEps), ns5Iterations_(inNs5Iterations) {
		lr_ = inLr;
		for (auto* p : inParamPtrs) if (p && p->requiresGrad) params_.pushBack(p);
	}

	void step() override;
	void zeroGrad() override;
	[[nodiscard]] oa::Status saveTo(Engine& inEngine, ModelFile& outFile) const override;
	[[nodiscard]] oa::Status validateLoad(const ModelFile& inFile) const override;
	[[nodiscard]] oa::Status loadFrom(Engine& inEngine, const ModelFile& inFile) override;

private:
	oa::F32 beta_, weightDecay_, eps_;
	oa::I32 ns5Iterations_;
	oa::Vector<Matrix> momentum_;
};

} // namespace oa
