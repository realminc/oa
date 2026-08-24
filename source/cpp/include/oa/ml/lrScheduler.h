// OA ML — Learning Rate Schedulers
//
// Provides various learning rate scheduling strategies for training.
// All schedulers inherit from LRScheduler and are used via callback
// during training.

#pragma once

#include <oa/core/types.h>

namespace oa {

class Optimizer;

// ─── LRScheduler (base) ──────────────────────────────────────────────────────

class LRScheduler {
public:
	virtual ~LRScheduler() = default;
	virtual oa::F32 getLr(oa::U64 inStep) const = 0;
	void apply(oa::Optimizer& inOptimizer) const;
};

// ─── CosineScheduler ─────────────────────────────────────────────────────────
//
// Cosine annealing from maxLr to minLr over totalSteps.

class CosineScheduler : public LRScheduler {
public:
	CosineScheduler(oa::F32 inMaxLr, oa::F32 inMinLr, oa::U64 inTotalSteps)
		: maxLr_(inMaxLr), minLr_(inMinLr), totalSteps_(inTotalSteps) {}

	[[nodiscard]] oa::F32 getLr(oa::U64 inStep) const override;

private:
	oa::F32 maxLr_, minLr_;
	oa::U64 totalSteps_;
};

// ─── WarmupScheduler ─────────────────────────────────────────────────────────
//
// Linear warmup to targetLr over warmupSteps, then delegate to after scheduler.

class WarmupScheduler : public LRScheduler {
public:
	WarmupScheduler(oa::F32 inTargetLr, oa::U64 inWarmupSteps, oa::SharedPtr<LRScheduler> inAfter = nullptr)
		: targetLr_(inTargetLr), warmupSteps_(inWarmupSteps), after_(std::move(inAfter))
	{}

	[[nodiscard]] oa::F32 getLr(oa::U64 inStep) const override;

private:
	oa::F32 targetLr_;
	oa::U64 warmupSteps_;
	oa::SharedPtr<LRScheduler> after_;
};

// ─── OneCycleScheduler ───────────────────────────────────────────────────────
//
// Smith's 1cycle policy: warmup ramp to maxLr, then cosine anneal to near-zero.
// initialLr = maxLr/divFactor. finalLr = initialLr/finalDivFactor.

class OneCycleScheduler : public LRScheduler {
public:
	OneCycleScheduler(oa::F32 inMaxLr, oa::U64 inTotalSteps, oa::F32 inPctStart = 0.3f, oa::F32 inDivFactor = 25.0f, oa::F32 inFinalDivFactor = 1e4f)
		: maxLr_(inMaxLr), totalSteps_(inTotalSteps), pctStart_(inPctStart),
		  divFactor_(inDivFactor), finalDivFactor_(inFinalDivFactor) {}

	[[nodiscard]] oa::F32 getLr(oa::U64 inStep) const override;

private:
	oa::F32 maxLr_;
	oa::U64 totalSteps_;
	oa::F32 pctStart_, divFactor_, finalDivFactor_;
};

// ─── CyclicScheduler ─────────────────────────────────────────────────────────
//
// Cyclic LR: oscillates between baseLr and maxLr.
// Triangular = constant amplitude. Triangular2 = halving amplitude per cycle.
// ExpRange = exponential decay per iteration.

enum class CyclicMode : oa::U8 { Triangular, Triangular2, ExpRange };

class CyclicScheduler : public LRScheduler {
public:
	CyclicScheduler(oa::F32 inBaseLr, oa::F32 inMaxLr, oa::U64 inStepSizeUp,
		CyclicMode inMode = CyclicMode::Triangular, oa::F32 inGamma = 1.0f)
		: baseLr_(inBaseLr), maxLr_(inMaxLr), stepSizeUp_(inStepSizeUp),
		  mode_(inMode), gamma_(inGamma) {}

	[[nodiscard]] oa::F32 getLr(oa::U64 inStep) const override;

private:
	oa::F32 baseLr_, maxLr_;
	oa::U64 stepSizeUp_;
	CyclicMode mode_;
	oa::F32 gamma_;
};

// ─── CosineWarmRestartsScheduler ─────────────────────────────────────────────
//
// SGDR: cosine annealing with periodic warm restarts.
// Period starts at t0 steps, multiplied by tMult after each restart.

class CosineWarmRestartsScheduler : public LRScheduler {
public:
	CosineWarmRestartsScheduler(oa::F32 inMaxLr, oa::U64 inT0, oa::U64 inTMult = 1, oa::F32 inEtaMin = 0.0f)
		: maxLr_(inMaxLr), t0_(inT0), tMult_(inTMult), etaMin_(inEtaMin) {}

	[[nodiscard]] oa::F32 getLr(oa::U64 inStep) const override;

private:
	oa::F32 maxLr_;
	oa::U64 t0_, tMult_;
	oa::F32 etaMin_;
};

// ─── ReduceOnPlateauScheduler ────────────────────────────────────────────────
//
// Metric-driven: drop LR by factor when metric stalls for patience steps.
// call step(metric) each epoch. getLr() returns current LR (ignores inStep).

enum class PlateauMode : oa::U8 { Min, Max };

class ReduceOnPlateauScheduler : public LRScheduler {
public:
	ReduceOnPlateauScheduler(oa::F32 inInitialLr, oa::F32 inFactor = 0.1f,
		oa::U64 inPatience = 10, oa::F32 inThreshold = 1e-4f,
		oa::F32 inMinLr = 0.0f, PlateauMode inMode = PlateauMode::Min)
		: currentLr_(inInitialLr), factor_(inFactor), threshold_(inThreshold),
		  minLr_(inMinLr), best_(inMode == PlateauMode::Min ? 1e30f : -1e30f),
		  patience_(inPatience), mode_(inMode) {}

	void step(oa::F32 inMetric);
	[[nodiscard]] oa::F32 getLr(oa::U64 inStep) const override;

private:
	oa::F32 currentLr_, factor_, threshold_, minLr_, best_;
	oa::U64 patience_, numBadEpochs_ = 0;
	PlateauMode mode_;
};

// ─── SequentialScheduler ─────────────────────────────────────────────────────
//
// Chain N schedulers at milestones. milestones[i] = step at which scheduler i+1 starts.
// Each sub-scheduler receives step relative to its own start.

class SequentialScheduler : public LRScheduler {
public:
	SequentialScheduler(oa::Vec<oa::SharedPtr<LRScheduler>> inSchedulers, oa::Vec<oa::U64> inMilestones)
		: schedulers_(std::move(inSchedulers)), milestones_(std::move(inMilestones)) {}

	[[nodiscard]] oa::F32 getLr(oa::U64 inStep) const override;

private:
	oa::Vec<oa::SharedPtr<LRScheduler>> schedulers_;
	oa::Vec<oa::U64> milestones_;
};

// ─── LinearWarmupCosineScheduler ─────────────────────────────────────────────
//
// Convenience scheduler: linear warmup to targetLr, then cosine annealing to minLr.
// Composes WarmupScheduler + CosineScheduler.

class LinearWarmupCosineScheduler : public LRScheduler {
public:
	LinearWarmupCosineScheduler(oa::I32 inWarmupSteps, oa::I32 inTotalSteps, oa::F32 inMaxLr, oa::F32 inMinLr);

	[[nodiscard]] oa::F32 getLr(oa::U64 inStep) const override;

private:
	oa::SharedPtr<LRScheduler> inner_;
};

} // namespace oa
