// oa::ItTraining — exact training lifecycle iterator
//
// Lifecycle-managed optimizer step shared by supervised and reinforcement
// learning coordinators. The direct loop remains the canonical surface:
//
//   // flat while-loop — canonical pattern.
//   oa::CbProgressBar bar;
//   oa::ItTraining iter(engine, opt, oa::ItTrainingConfig{
//     .totalSteps    = kSteps,
//     .stepsPerEpoch = kBatchesPerEpoch,
//     .batchSize     = kBatch,
//     .callbacks     = {&bar},
//   });
//   while (!iter.isDone()) {
//     sampler.nextBatch(x, y);
//     auto logits = model->forward(x);
//     auto loss   = oa::FnLoss::crossEntropy(logits, y);
//     model->backward(dLog);
//     iter.next(loss);
//   }
//   (void)iter.finish();

#pragma once

#include <oa/core/callback.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/chrono.h>
#include <oa/core/std/function.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/vec.h>
#include <oa/core/iterator.h>
#include <oa/core/matrix.h>
#include <oa/core/status.h>
#include <oa/runtime/timer.h>

namespace oa { class ExecutionSession; }

namespace oa {

class Optimizer;
class Engine;
class TrainingProgram;
class TrainingSession;
class ItTraining;
class CbTraining;
class Metric;

struct ItTrainingConfig {
	// total step budget. 0 = open-ended (isDone() never returns true via totalSteps).
	oa::I64 totalSteps      = 0;
	// batches per epoch. 0 = single-epoch mode (epoch boundary never fires).
	oa::I64 stepsPerEpoch   = 0;
	// Variable-length epoch schedule: epochSteps[i] = steps in epoch i+1.
	// Overrides stepsPerEpoch when non-empty. totalSteps is forced to the sum.
	oa::Vec<oa::I64> epochSteps;
	oa::I32 batchSize       = 1;
	// Optional sequence work per sample (for throughput reporting).
	oa::I32 sequenceLength  = 0;
	oa::String sequenceUnit = "token";
	oa::F64 sourceUnitsPerSample = 0.0;
	oa::String sourceUnit   = "byte";
	oa::String timerName    = "training_step";
	oa::Bool enableGpuTiming = true;
	// Stateful metrics updated once after each completed step. Non-owning.
	oa::Vec<Metric*> metrics;
	// callbacks registered at construction. Non-owning.
	oa::Vec<CbTraining*> callbacks;
	// Optional fixed-shape program for capture/replay. Non-owning.
	TrainingProgram* program = nullptr;
};

struct GpuTimingStats {
	oa::I64 count    = 0;
	oa::F64 meanMs   = 0.0;
	oa::F64 minMs    = 0.0;
	oa::F64 medianMs = 0.0;
	oa::F64 p95Ms    = 0.0;
	oa::F64 lastMs   = 0.0;
};

struct TrainingPhaseStats {
	oa::I64 count          = 0;
	oa::F64 totalMs        = 0.0;
	oa::F64 bodyMs         = 0.0;
	oa::F64 optimizerMs    = 0.0;
	oa::F64 compileMs      = 0.0;
	oa::F64 recordMs       = 0.0;
	oa::F64 submitMs       = 0.0;
	oa::F64 waitMs         = 0.0;
	oa::F64 scalarMetricMs = 0.0;
	oa::F64 callbackMs     = 0.0;

	[[nodiscard]] oa::F64 mean(oa::F64 inSumMs) const {
		return count > 0 ? inSumMs / static_cast<oa::F64>(count) : 0.0;
	}
	[[nodiscard]] oa::F64 accountedMs() const {
		return bodyMs + optimizerMs + compileMs + recordMs + submitMs + waitMs + scalarMetricMs + callbackMs;
	}
};

class ItTraining : public oa::Iterator {
public:
	ItTraining(Engine& inEngine, Optimizer& inOpt, ItTrainingConfig inCfg = {});
	~ItTraining() override;

	// ─── oa::Iterator interface ─────────────────────────────────────────────
	[[nodiscard]] bool isDone() const override;
	void next() override;
	void next(const Matrix& inLoss);
	void reset() override;
	[[nodiscard]] oa::I64 index() const override { return index_; }

	// ─── training step (lambda sugar) ─────────────────────────────────────
	void step(const oa::Fn<void()>& inOpFn);
	void step(
		const oa::Fn<void()>& inPrepareFn,
		const oa::Fn<void()>& inRecordFn);

	void recordLoss(const Matrix& inLoss);
	void recordAccuracy(oa::F32 inAcc) { liveAccuracy_ = inAcc; }
	void recordSourceUnits(oa::I64 inUnits) { pendingSourceUnits_ = oa::max<oa::I64>(inUnits, 0); }

	[[nodiscard]] oa::Status finish();
	[[nodiscard]] oa::Status requestProgramRecapture();
	void attachSession(TrainingSession* inSession) { session_ = inSession; }
	[[nodiscard]] TrainingSession* session() const { return session_; }

	void requestStop() { stopRequested_ = true; }
	[[nodiscard]] bool stopRequested() const { return stopRequested_; }
	[[nodiscard]] const oa::Status& lastStatus() const { return lastStatus_; }

	// ─── callbacks ────────────────────────────────────────────────────────
	void addCallback(CbTraining* inCallback);
	void addMetric(Metric* inMetric);

	// ─── State ────────────────────────────────────────────────────────────
	[[nodiscard]] oa::I64 stepCount()           const { return index_; }
	[[nodiscard]] oa::I64 totalSteps()          const { return cfg_.totalSteps; }
	[[nodiscard]] oa::I64 epoch()               const;
	[[nodiscard]] oa::I64 stepInEpoch()         const;
	[[nodiscard]] oa::I64 totalEpochs()         const;
	[[nodiscard]] bool  isEpochBoundary()     const;
	[[nodiscard]] bool  isLastStep()          const;
	[[nodiscard]] oa::I64 stepsInCurrentEpoch() const;

	[[nodiscard]] oa::F32 lastLoss()            const { return lastLoss_; }
	[[nodiscard]] oa::F64 lastGpuMs()           const { return lastGpuMs_; }
	[[nodiscard]] oa::I64 lastLossStep()        const { return lastLossStep_; }
	[[nodiscard]] oa::I64 lastGpuTimeStep()     const { return lastGpuTimeStep_; }
	[[nodiscard]] bool  hasLossSample()       const { return lastLossStep_ == index_; }
	[[nodiscard]] bool  hasGpuTimeSample()    const { return lastGpuTimeStep_ == index_; }
	[[nodiscard]] GpuTimingStats gpuTimingStats() const;
	[[nodiscard]] const TrainingPhaseStats& trainingPhaseStats() const {
		return trainingPhaseStats_;
	}

	[[nodiscard]] oa::F32 liveAccuracy()          const { return liveAccuracy_; }
	[[nodiscard]] oa::F64 gpuSamplesPerSecond()    const;
	[[nodiscard]] oa::F64 wallSamplesPerSecond()   const;
	[[nodiscard]] oa::F64 gpuUnitsPerSecond()      const;
	[[nodiscard]] oa::F64 wallUnitsPerSecond()     const;
	[[nodiscard]] oa::F64 gpuSourceUnitsPerSecond()  const;
	[[nodiscard]] oa::F64 wallSourceUnitsPerSecond() const;
	[[nodiscard]] oa::F64 epochSourceUnitsPerSecond() const;
	[[nodiscard]] oa::F64 wallMsPerStep()          const;
	[[nodiscard]] oa::I64 totalSamples()           const { return totalSamples_; }
	[[nodiscard]] oa::I64 totalUnits()             const { return totalUnits_; }
	[[nodiscard]] oa::I64 totalSourceUnits()       const { return totalSourceUnits_; }
	[[nodiscard]] oa::F64 elapsedSeconds()         const;

	void excludeWallTime(oa::F64 inSeconds);

	[[nodiscard]] oa::F64 epochLossSum()          const { return epochLossSum_; }
	[[nodiscard]] oa::I64 epochLossCount()        const { return epochLossCount_; }
	[[nodiscard]] oa::F64 epochSampledMeanLoss()  const;
	[[nodiscard]] oa::F64 epochMeanLoss()         const;
	[[nodiscard]] oa::F64 trainingMeanLoss()      const;
	[[nodiscard]] oa::F64 epochSeconds()          const;

	[[nodiscard]] Optimizer& optimizer() const { return opt_; }
	[[nodiscard]] const ItTrainingConfig& cfg() const { return cfg_; }
	[[nodiscard]] bool hasEpochs() const {
		return cfg_.stepsPerEpoch > 0 || !cfg_.epochSteps.empty();
	}

private:
	void fireTrainBegin();
	void fireEpochBegin();
	void fireStepEnd();
	void fireEpochEnd();
	void fireTrainEnd();
	bool captureCallbackStatus_(CbTraining& inCallback, const char* inPhase);
	void resetMetrics_();
	void updateMetrics_();
	void closeStableResourceFrame_();
	void advanceIfNeeded_();
	[[nodiscard]] oa::I64 epochIndexForStep_(oa::I64 inStep) const;

	Optimizer&           opt_;
	oa::ExecutionSession* executionSession_ = nullptr;
	Engine*              rt_                 = nullptr;
	ItTrainingConfig     cfg_;
	oa::Vec<oa::I64>       epochOffsets_;
	bool                 stopRequested_        = false;
	oa::Status             lastStatus_           = oa::Status::ok();
	oa::I64                totalSamples_         = 0;
	oa::I64                totalUnits_           = 0;
	oa::I64                totalSourceUnits_     = 0;
	oa::I64                gpuTimedSourceUnits_  = 0;
	oa::I64                pendingSourceUnits_   = -1;
	oa::I64                lastStepSourceUnits_  = 0;
	oa::I64                epochSourceUnits_     = 0;
	Matrix               pendingLoss_;
	Matrix               programLoss_;
	oa::F32                liveAccuracy_         = oa::Limits<oa::F32>::quietNaN();
	oa::F32                lastLoss_             = 0.0F;
	oa::F64                lastGpuMs_            = 0.0;
	oa::I64                lastLossStep_         = 0;
	oa::I64                lastGpuTimeStep_      = 0;
	oa::Vec<oa::F64>       gpuTimingSamples_;
	oa::F64                gpuTimingSumMs_       = 0.0;
	oa::Timer              timer_;
	bool                 timerReady_           = false;
	bool                 trainBeginFired_      = false;
	mutable bool         bodyPending_          = false;
	oa::I64                lastEpochFired_       = -1;
	oa::F64                epochLossSum_         = 0.0;
	oa::I64                epochLossCount_       = 0;
	oa::F64                trainingLossSum_      = 0.0;
	oa::I64                trainingLossCount_    = 0;
	oa::HighResolutionTimePoint t0_;
	oa::HighResolutionTimePoint epochT0_;
	oa::HighResolutionTimePoint lastStepT_;
	oa::HighResolutionTimePoint phaseBodyT0_;
	TrainingPhaseStats   trainingPhaseStats_;
	bool                 trainingPhaseTiming_ = false;
	bool                 phaseBodyStarted_    = false;
	bool                 stableResourceFrameOpen_ = false;
	bool                 programCaptureDisabled_  = false;
	bool                 programReportWritten_    = false;
	oa::Vec<Metric*>       metrics_;
	oa::Vec<CbTraining*>   callbacks_;
	TrainingSession* session_ = nullptr;
};

// ─── CbTraining ──────────────────────────────────────────────────────────────
//
// training-specific callback base class. Subclass and attach via addCallback().
// All hooks have default no-op implementations.

class CbTraining : public oa::Callback {
public:
	~CbTraining() override = default;

	virtual void onTrainBegin(ItTraining& inIter) { (void)inIter; }
	virtual void onEpochBegin(ItTraining& inIter) { (void)inIter; }
	virtual void onStepEnd(ItTraining& inIter)    { (void)inIter; }
	virtual void onEpochEnd(ItTraining& inIter)   { (void)inIter; }
	virtual void onTrainEnd(ItTraining& inIter)   { (void)inIter; }

	[[nodiscard]] virtual oa::Status getStatus() const { return oa::Status::ok(); }

protected:
	CbTraining() = default;
};

} // namespace oa
