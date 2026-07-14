// Oa Ml — Learning Rate Schedulers
//
// Provides various learning rate scheduling strategies for training.
// All schedulers inherit from OaLRScheduler and are used via OaCbLrScheduler
// callback during training.

#pragma once

#include <Oa/Core/Types.h>

// Forward declaration
class OaOptimizer;

// ─── OaLRScheduler (Base) ─────────────────────────────────────────────────────

class OaLRScheduler {
public:
	virtual ~OaLRScheduler() = default;
	virtual OaF32 GetLr(OaU64 InStep) const = 0;
	void Apply(OaOptimizer& InOptimizer) const;
};

// ─── OaCosineScheduler ─────────────────────────────────────────────────────────
//
// Cosine annealing from MaxLr to MinLr over TotalSteps.

class OaCosineScheduler : public OaLRScheduler {
public:
	OaCosineScheduler(OaF32 InMaxLr, OaF32 InMinLr, OaU64 InTotalSteps)
		: MaxLr_(InMaxLr), MinLr_(InMinLr), TotalSteps_(InTotalSteps) {}

	[[nodiscard]] OaF32 GetLr(OaU64 InStep) const override;

private:
	OaF32 MaxLr_, MinLr_;
	OaU64 TotalSteps_;
};

// ─── OaWarmupScheduler ─────────────────────────────────────────────────────────
//
// Linear warmup to TargetLr over WarmupSteps, then delegate to After scheduler.

class OaWarmupScheduler : public OaLRScheduler {
public:
	OaWarmupScheduler(OaF32 InTargetLr, OaU64 InWarmupSteps, OaSharedPtr<OaLRScheduler> InAfter = nullptr)
		: TargetLr_(InTargetLr), WarmupSteps_(InWarmupSteps), After_(std::move(InAfter))
	{}

	[[nodiscard]] OaF32 GetLr(OaU64 InStep) const override;

private:
	OaF32 TargetLr_;
	OaU64 WarmupSteps_;
	OaSharedPtr<OaLRScheduler> After_;
};

// ─── OaOneCycleScheduler ───────────────────────────────────────────────────────
//
// Smith's 1cycle policy: warmup ramp to MaxLr, then cosine anneal to near-zero.
// InitialLr = MaxLr/DivFactor. FinalLr = InitialLr/FinalDivFactor.

class OaOneCycleScheduler : public OaLRScheduler {
public:
	OaOneCycleScheduler(OaF32 InMaxLr, OaU64 InTotalSteps, OaF32 InPctStart = 0.3f, OaF32 InDivFactor = 25.0f, OaF32 InFinalDivFactor = 1e4f)
		: MaxLr_(InMaxLr), TotalSteps_(InTotalSteps), PctStart_(InPctStart),
		  DivFactor_(InDivFactor), FinalDivFactor_(InFinalDivFactor) {}

	[[nodiscard]] OaF32 GetLr(OaU64 InStep) const override;

private:
	OaF32 MaxLr_;
	OaU64 TotalSteps_;
	OaF32 PctStart_, DivFactor_, FinalDivFactor_;
};

// ─── OaCyclicScheduler ─────────────────────────────────────────────────────────
//
// Cyclic LR: oscillates between BaseLr and MaxLr.
// Triangular = constant amplitude. Triangular2 = halving amplitude per cycle.
// ExpRange = exponential decay per iteration.

enum class OaCyclicMode : OaU8 { Triangular, Triangular2, ExpRange };

class OaCyclicScheduler : public OaLRScheduler {
public:
	OaCyclicScheduler(OaF32 InBaseLr, OaF32 InMaxLr, OaU64 InStepSizeUp,
		OaCyclicMode InMode = OaCyclicMode::Triangular, OaF32 InGamma = 1.0f)
		: BaseLr_(InBaseLr), MaxLr_(InMaxLr), StepSizeUp_(InStepSizeUp),
		  Mode_(InMode), Gamma_(InGamma) {}

	[[nodiscard]] OaF32 GetLr(OaU64 InStep) const override;

private:
	OaF32 BaseLr_, MaxLr_;
	OaU64 StepSizeUp_;
	OaCyclicMode Mode_;
	OaF32 Gamma_;
};

// ─── OaCosineWarmRestartsScheduler ────────────────────────────────────────────
//
// SGDR: cosine annealing with periodic warm restarts.
// Period starts at T0 steps, multiplied by TMult after each restart.

class OaCosineWarmRestartsScheduler : public OaLRScheduler {
public:
	OaCosineWarmRestartsScheduler(OaF32 InMaxLr, OaU64 InT0, OaU64 InTMult = 1, OaF32 InEtaMin = 0.0f)
		: MaxLr_(InMaxLr), T0_(InT0), TMult_(InTMult), EtaMin_(InEtaMin) {}

	[[nodiscard]] OaF32 GetLr(OaU64 InStep) const override;

private:
	OaF32 MaxLr_;
	OaU64 T0_, TMult_;
	OaF32 EtaMin_;
};

// ─── OaReduceOnPlateauScheduler ─────────────────────────────────────────────────
//
// Metric-driven: drop LR by Factor when metric stalls for Patience steps.
// Call Step(metric) each epoch. GetLr() returns current LR (ignores InStep).

enum class OaPlateauMode : OaU8 { Min, Max };

class OaReduceOnPlateauScheduler : public OaLRScheduler {
public:
	OaReduceOnPlateauScheduler(OaF32 InInitialLr, OaF32 InFactor = 0.1f,
		OaU64 InPatience = 10, OaF32 InThreshold = 1e-4f,
		OaF32 InMinLr = 0.0f, OaPlateauMode InMode = OaPlateauMode::Min)
		: CurrentLr_(InInitialLr), Factor_(InFactor), Threshold_(InThreshold),
		  MinLr_(InMinLr), Best_(InMode == OaPlateauMode::Min ? 1e30f : -1e30f),
		  Patience_(InPatience), Mode_(InMode) {}

	void Step(OaF32 InMetric);
	[[nodiscard]] OaF32 GetLr(OaU64 InStep) const override;

private:
	OaF32 CurrentLr_, Factor_, Threshold_, MinLr_, Best_;
	OaU64 Patience_, NumBadEpochs_ = 0;
	OaPlateauMode Mode_;
};

// ─── OaSequentialScheduler ────────────────────────────────────────────────────
//
// Chain N schedulers at milestones. Milestones[i] = step at which scheduler i+1 starts.
// Each sub-scheduler receives step relative to its own start.

class OaSequentialScheduler : public OaLRScheduler {
public:
	OaSequentialScheduler(OaVec<OaSharedPtr<OaLRScheduler>> InSchedulers, OaVec<OaU64> InMilestones)
		: Schedulers_(std::move(InSchedulers)), Milestones_(std::move(InMilestones)) {}

	[[nodiscard]] OaF32 GetLr(OaU64 InStep) const override;

private:
	OaVec<OaSharedPtr<OaLRScheduler>> Schedulers_;
	OaVec<OaU64> Milestones_;
};

// ─── OaLinearWarmupCosineScheduler ─────────────────────────────────────────────
//
// Convenience scheduler: linear warmup to TargetLr, then cosine annealing to MinLr.
// Composes OaWarmupScheduler + OaCosineScheduler.

class OaLinearWarmupCosineScheduler : public OaLRScheduler {
public:
	OaLinearWarmupCosineScheduler(OaI32 InWarmupSteps, OaI32 InTotalSteps, OaF32 InMaxLr, OaF32 InMinLr);

	[[nodiscard]] OaF32 GetLr(OaU64 InStep) const override;

private:
	OaSharedPtr<OaLRScheduler> Inner_;
};
