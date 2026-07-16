// OaItTraining — exact training lifecycle iterator
//
// Lifecycle-managed training step. Sits between the bare for-loop and a future
// OaTrainer::Fit. Two usage styles, both equivalent:
//
//   // tqdm/Keras flat while-loop — the canonical pattern.
//   OaCbProgressBar bar;
//   OaItTraining iter(*opt, OaItTrainingConfig{
//     .TotalSteps    = kSteps,
//     .StepsPerEpoch = kBatchesPerEpoch,
//     .BatchSize     = kBatch,
//     .Callbacks     = {&bar},
//   });
//   while (not iter.IsDone()) {
//     sampler.NextBatch(x, y);
//     auto logits = model->Forward(x);
//     auto loss   = OaFnLoss::CrossEntropy(logits, y);
//     auto dLog   = OaFnLoss::CrossEntropyBwd(logits, y);
//     model->Backward(dLog);
//     iter.Next(loss);     // records loss, performs one optimizer step, and advances.
//   }
//   (void)iter.Finish();
//
//   // Lambda style — equivalent sugar for the same flow.
//   while (not iter.IsDone()) {
//     iter.Step([&]{ /* forward / loss / backward */ });
//   }
//
// Every optimizer step is one complete, observable unit: forward, backward,
// optimizer, submit, synchronization, exact loss, timing, then callbacks.
// Console redraws and checkpoint writes may be throttled by their callbacks;
// metric correctness never depends on an output cadence.
//
// Phase 3 implicit-autograd will eliminate the explicit RecordLoss + Backward
// inside the body; only `model->Forward(x);` + loss call will remain.

#pragma once

#include <Oa/Core/Callback.h>
#include <Oa/Core/Iterator.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/GpuTimer.h>

#include <chrono>
#include <algorithm>
#include <functional>
#include <limits>
#include <vector>

class OaOptimizer;
class OaTrainingProgram;
class OaItTraining;
class OaCbTraining;
class OaMetric;

class OaItTrainingConfig {
public:
	// Total step budget. 0 = open-ended (IsDone() never returns true via TotalSteps).
	OaI64 TotalSteps      = 0;
	// Batches per epoch. 0 = single-epoch mode (epoch boundary never fires).
	// When set, Epoch() and StepInEpoch() are meaningful and OnEpoch* callbacks
	// fire at boundaries.
	OaI64 StepsPerEpoch   = 0;
	// Variable-length epoch schedule: EpochSteps[i] = steps in epoch i+1.
	// Overrides StepsPerEpoch when non-empty (multi-phase training where phases
	// have different epoch counts/lengths). TotalSteps is forced to the sum.
	std::vector<OaI64> EpochSteps;
	// Work completed by one optimizer step. A training step always contains a
	// batch; the smallest valid batch is one sample.
	OaI32 BatchSize       = 1;
	// Optional sequence work per sample. When >0, throughput is reported both
	// as sample/s and as <SequenceUnit>/s. Examples: 16 + "token" for an LM,
	// 64 + "frame" for motion training.
	OaI32 SequenceLength  = 0;
	OaString SequenceUnit = "token";
	// Optional source-domain work represented by one sample. This is separate
	// from SequenceLength because tokenizers change how many source bytes each
	// token covers. Fixed-width byte models set this to SequenceLength; variable
	// tokenizers call RecordSourceUnits() with the exact whole-batch count.
	OaF64 SourceUnitsPerSample = 0.0;
	OaString SourceUnit = "byte";
	// Optional GPU timer label for nsys / perf store integration.
	const char* TimerName = "training_step";
	// Disable only for a deliberately uninstrumented latency experiment. Eager
	// and captured paths otherwise report the same complete-step GPU interval.
	OaBool EnableGpuTiming = true;
	// Stateful metrics updated exactly once after each completed step and reset
	// at epoch boundaries. Non-owning; caller controls lifetime.
	std::vector<OaMetric*> Metrics;
	// Callbacks registered at construction. Non-owning — caller controls
	// lifetime (typically declared on the stack above the iterator). Equivalent
	// to calling AddCallback(&cb) for each pointer in registration order.
	std::vector<OaCbTraining*> Callbacks;
	// Optional fixed-shape forward + backward + optimizer program. Step one runs
	// eagerly to create lazy state, step two is captured from a stable resource
	// frame, and following steps replay one pre-recorded Vulkan plan. Unsupported
	// operations fall back to eager execution for the rest of the run.
	// Non-owning; the program must outlive this iterator.
	OaTrainingProgram* Program = nullptr;
};

struct OaGpuTimingStats {
	OaI64 Count = 0;
	OaF64 MeanMs = 0.0;
	OaF64 MinMs = 0.0;
	OaF64 MedianMs = 0.0;
	OaF64 P95Ms = 0.0;
	OaF64 LastMs = 0.0;
};

// Opt-in host/runtime decomposition of a complete training step. Enable with
// OA_LOG_TRAINING_PHASES=1. The normal training path does not take per-phase
// timestamps when disabled.
struct OaTrainingPhaseStats {
	OaI64 Count = 0;
	OaF64 TotalMs = 0.0;
	OaF64 BodyMs = 0.0;
	OaF64 OptimizerMs = 0.0;
	OaF64 CompileMs = 0.0;
	OaF64 RecordMs = 0.0;
	OaF64 SubmitMs = 0.0;
	OaF64 WaitMs = 0.0;
	OaF64 ScalarMetricMs = 0.0;
	OaF64 CallbackMs = 0.0;

	[[nodiscard]] OaF64 Mean(OaF64 InSumMs) const {
		return Count > 0 ? InSumMs / static_cast<OaF64>(Count) : 0.0;
	}

	[[nodiscard]] OaF64 AccountedMs() const {
		return BodyMs + OptimizerMs + CompileMs + RecordMs + SubmitMs + WaitMs
			+ ScalarMetricMs + CallbackMs;
	}
};

class OaItTraining : public OaIterator {
public:
	OaItTraining(OaOptimizer& InOpt, OaItTrainingConfig InCfg = {});
	~OaItTraining() override;

	// ─── OaIterator interface ─────────────────────────────────────────────
	// IsDone() also lazily advances Index_ (on first call AND between Next()
	// calls) and fires OnTrainBegin / OnEpochBegin via const_cast. This lets
	// the natural `while (!iter.IsDone()) { body; iter.Next(); }` pattern
	// work without a separate Begin() method — the body for step N runs with
	// Index_ already equal to N.
	[[nodiscard]] bool IsDone() const override;
	// Finalize the current step: records opt.Step(), submits and synchronizes it,
	// reads exact loss + GPU time, fires OnStepEnd, then OnEpochEnd if this was
	// the last step of an epoch. Does NOT advance Index_ — that happens lazily in
	// the next IsDone() call so the body for step N always sees Index_=N.
	void Next() override;
	// Convenience overload: tag the loss matrix then run Next(). Lets the
	// body read like `iter.Next(loss);` instead of two calls.
	void Next(const OaMatrix& InLoss);
	void Reset() override;
	[[nodiscard]] OaI64 Index() const override { return Index_; }

	// ─── Training step (lambda sugar) ─────────────────────────────────────
	// Equivalent to: if (IsDone()) return; InOpFn(); Next();
	// Body should record forward + loss-grad + backward into the active
	// OaContext, and call RecordLoss for the scalar loss matrix when desired.
	// With Program enabled this overload deliberately replays the same captured
	// input after capture; use the two-lambda overload for changing batches.
	void Step(const std::function<void()>& InOpFn);
	// Fixed-shape capture with changing input. InPrepareFn always runs and may
	// refill already allocated/mapped input matrices, but it must not append graph
	// nodes after capture. InRecordFn records zero-grad + forward + backward and
	// loss only for eager warm-up/capture; it is skipped on replay steps.
	void Step(
		const std::function<void()>& InPrepareFn,
		const std::function<void()>& InRecordFn);

	// Tag the scalar loss matrix from inside InOpFn. It is read exactly after the
	// step completes and is available to every OnStepEnd callback.
	void RecordLoss(const OaMatrix& InLoss);

	// Record a precomputed accuracy fraction in [0, 1]. CPU-side scalar:
	// caller computes accuracy however they want (typically at epoch
	// boundaries via EvalAccuracy) and stuffs the value here for
	// LiveAccuracy() / progress-bar display. Unset (=NaN) means "no value".
	void RecordAccuracy(OaF32 InAcc) { LiveAccuracy_ = InAcc; }
	// Override source-domain work for the current step (whole batch). Useful for
	// variable-length tokenization where each fixed token window covers a
	// different number of bytes. Must be called after IsDone() and before Next().
	void RecordSourceUnits(OaI64 InUnits) { PendingSourceUnits_ = std::max<OaI64>(InUnits, 0); }

	// Drain any pending submitted-but-not-synced batches. Call once after the
	// last Step() so the final commands complete before teardown.
	[[nodiscard]] OaStatus Finish();
	// Explicitly invalidate a captured fixed-shape program. The next Step()
	// records and captures the new shape/topology; optimizer state is preserved.
	[[nodiscard]] OaStatus RequestProgramRecapture();

	// Cooperative stop (Keras `model.stop_training = True`). Callbacks (early
	// stopping, time budget) call this; the next IsDone() returns true and the
	// caller's while-loop exits normally — Finish() still runs OnTrainEnd.
	void RequestStop() { StopRequested_ = true; }
	[[nodiscard]] bool StopRequested() const { return StopRequested_; }
	// Execution status for the most recently attempted step. A failed async
	// submit/flush/sync requests a cooperative stop and preserves the concrete
	// runtime error here so embedders can surface it without parsing logs.
	[[nodiscard]] const OaStatus& LastStatus() const { return LastStatus_; }

	// ─── Callbacks ────────────────────────────────────────────────────────
	// Attach a callback; OaItTraining calls its hooks at train/epoch/step bounds.
	// Order of registration = order of invocation. Callbacks are non-owning.
	void AddCallback(OaCbTraining* InCallback);
	void AddMetric(OaMetric* InMetric);

	// ─── State ────────────────────────────────────────────────────────────
	[[nodiscard]] OaI64 StepCount()           const { return Index_; }
	[[nodiscard]] OaI64 TotalSteps()          const { return Cfg_.TotalSteps; }
	[[nodiscard]] OaI64 Epoch()               const;
	[[nodiscard]] OaI64 StepInEpoch()         const;
	[[nodiscard]] OaI64 TotalEpochs()         const;
	[[nodiscard]] bool  IsEpochBoundary()     const;  // last step of an epoch
	[[nodiscard]] bool  IsLastStep()          const;  // Index == TotalSteps (during Step body)
	// Steps in the epoch currently running (varies per epoch with EpochSteps).
	[[nodiscard]] OaI64 StepsInCurrentEpoch() const;

	// ── Most recently completed step ──
	[[nodiscard]] OaF32 LastLoss()            const { return LastLoss_; }
	[[nodiscard]] OaF64 LastGpuMs()           const { return LastGpuMs_; }
	[[nodiscard]] OaI64 LastLossStep()        const { return LastLossStep_; }
	[[nodiscard]] OaI64 LastGpuTimeStep()     const { return LastGpuTimeStep_; }
	[[nodiscard]] bool  HasLossSample()        const { return LastLossStep_ == Index_; }
	[[nodiscard]] bool  HasGpuTimeSample()     const { return LastGpuTimeStep_ == Index_; }
	[[nodiscard]] OaGpuTimingStats GpuTimingStats() const;
	[[nodiscard]] const OaTrainingPhaseStats& TrainingPhaseStats() const {
		return TrainingPhaseStats_;
	}

	// Compatibility alias: completed steps are synchronized, so live == last.
	[[nodiscard]] OaF32 LiveLoss()            const { return LastLoss_; }

	// Last RecordAccuracy() value, or NaN if never set. CPU-side scalar; no
	// async / mailbox plumbing because accuracy is usually a low-rate metric.
	[[nodiscard]] OaF32 LiveAccuracy()        const { return LiveAccuracy_; }
	[[nodiscard]] OaF64 GpuSamplesPerSecond() const;
	[[nodiscard]] OaF64 WallSamplesPerSecond() const;
	[[nodiscard]] OaF64 GpuUnitsPerSecond() const;
	[[nodiscard]] OaF64 WallUnitsPerSecond() const;
	[[nodiscard]] OaF64 GpuSourceUnitsPerSecond() const;
	[[nodiscard]] OaF64 WallSourceUnitsPerSecond() const;
	[[nodiscard]] OaF64 EpochSourceUnitsPerSecond() const;
	[[nodiscard]] OaF64 WallMsPerStep() const;
	[[nodiscard]] OaI64 TotalSamples()        const { return TotalSamples_; }
	[[nodiscard]] OaI64 TotalUnits()          const { return TotalUnits_; }
	[[nodiscard]] OaI64 TotalSourceUnits()    const { return TotalSourceUnits_; }
	[[nodiscard]] OaF64 ElapsedSeconds()      const;

	// Remove non-training work performed synchronously by a callback (validation,
	// external evaluation) from wall/epoch/metric timing. The callback reports its
	// own duration; training latency and throughput remain training-only.
	void ExcludeWallTime(OaF64 InSeconds);

	// ── Epoch-running averages (reset on each OnEpochBegin) ──
	[[nodiscard]] OaF64 EpochLossSum()        const { return EpochLossSum_; }
	[[nodiscard]] OaI64 EpochLossCount()      const { return EpochLossCount_; }
	[[nodiscard]] OaF64 EpochSampledMeanLoss() const;
	[[nodiscard]] OaF64 EpochMeanLoss()       const;
	[[nodiscard]] OaF64 TrainingMeanLoss()    const;
	[[nodiscard]] OaF64 EpochSeconds()        const;

	// Read-only access to the optimizer (for callbacks that need to query/set LR)
	[[nodiscard]] OaOptimizer& Optimizer()    const { return Opt_; }

	// Read-only access to the config (so callbacks can introspect StepsPerEpoch, etc.)
	[[nodiscard]] const OaItTrainingConfig& Cfg() const { return Cfg_; }
	[[nodiscard]] bool HasEpochs()             const { return Cfg_.StepsPerEpoch > 0 or not Cfg_.EpochSteps.empty(); }
	[[nodiscard]] bool HasStepsPerEpoch()      const { return HasEpochs(); }  // legacy alias

private:
	void FireTrainBegin();
	void FireEpochBegin();
	void FireStepEnd();
	void FireEpochEnd();
	void FireTrainEnd();
	void ResetMetrics_();
	void UpdateMetrics_();
	void CloseStableResourceFrame_();

	// Lazy advance: fires TrainBegin (first call), advances Index_ to the
	// step the user is about to run, fires EpochBegin if we crossed into a
	// new epoch. Called from IsDone() (via const_cast) and Step() — no-op
	// if the body for the current step hasn't run yet.
	void AdvanceIfNeeded_();

	// Epoch index (0-based) for a 1-based step, honoring EpochSteps. Clamped
	// to the last epoch for the final IsDone() probe past TotalSteps.
	[[nodiscard]] OaI64 EpochIndexForStep_(OaI64 InStep) const;

	OaOptimizer&         Opt_;
	OaComputeEngine*     Rt_                = nullptr;
	OaItTrainingConfig   Cfg_;
	// Prefix sums of EpochSteps: EpochOffsets_[i] = first step of epoch i+1
	// minus one (i.e. cumulative steps after epoch i). Empty when uniform.
	std::vector<OaI64>   EpochOffsets_;
	bool                 StopRequested_        = false;
	OaStatus             LastStatus_           = OaStatus::Ok();
	// Index_ inherited from OaIterator; equals the current step number
	// during body execution and OnStepEnd / OnEpochEnd callbacks (1-based).
	OaI64                TotalSamples_         = 0;
	OaI64                TotalUnits_           = 0;
	OaI64                TotalSourceUnits_     = 0;
	OaI64                GpuTimedSourceUnits_  = 0;
	OaI64                PendingSourceUnits_   = -1;
	OaI64                LastStepSourceUnits_  = 0;
	OaI64                EpochSourceUnits_     = 0;
	OaMatrix             PendingLoss_;
	OaMatrix             ProgramLoss_;
	OaF32                LiveAccuracy_         = std::numeric_limits<OaF32>::quiet_NaN();
	OaF32                LastLoss_             = 0.0F;
	OaF64                LastGpuMs_            = 0.0;
	OaI64                LastLossStep_         = 0;
	OaI64                LastGpuTimeStep_      = 0;
	std::vector<OaF64>   GpuTimingSamples_;
	OaF64                GpuTimingSumMs_        = 0.0;
	OaGpuTimer           Timer_;
	bool                 TimerReady_           = false;
	bool                 TrainBeginFired_      = false;
	mutable bool         BodyPending_          = false;  // true between AdvanceIfNeeded_ and Next
	OaI64                LastEpochFired_       = -1;  // -1 = no epoch begun yet
	OaF64                EpochLossSum_         = 0.0;
	OaI64                EpochLossCount_       = 0;
	OaF64                TrainingLossSum_      = 0.0;
	OaI64                TrainingLossCount_    = 0;
	std::chrono::high_resolution_clock::time_point T0_;
	std::chrono::high_resolution_clock::time_point EpochT0_;
	std::chrono::high_resolution_clock::time_point LastStepT_;
	std::chrono::high_resolution_clock::time_point PhaseBodyT0_;
	OaTrainingPhaseStats TrainingPhaseStats_;
	bool                 TrainingPhaseTiming_ = false;
	bool                 PhaseBodyStarted_ = false;
	bool                 StableResourceFrameOpen_ = false;
	bool                 ProgramCaptureDisabled_ = false;
	std::vector<OaMetric*> Metrics_;
	std::vector<OaCbTraining*> Callbacks_;
};

// ─── OaCbTraining ──────────────────────────────────────────────────────────
//
// Training-specific callback base class. Extends OaCallback with training
// lifecycle hooks (epochs, steps, metrics). Subclass and attach via AddCallback.
// All hooks have default no-op implementations so you only override what you need.

class OaCbTraining : public OaCallback {
public:
	~OaCbTraining() override = default;

	// Fired once before the first Step(). Use for resource allocation,
	// header prints, opening files, etc.
	virtual void OnTrainBegin(OaItTraining& InIter) { (void)InIter; }

	// Fired before the first Step() of each epoch (only when StepsPerEpoch>0).
	virtual void OnEpochBegin(OaItTraining& InIter) { (void)InIter; }

	// Fired after each completed Step(). Loss and timing describe that exact step.
	virtual void OnStepEnd(OaItTraining& InIter) { (void)InIter; }

	// Fired after the last Step() of each epoch (only when StepsPerEpoch>0).
	// EpochMeanLoss() / EpochSeconds() report the just-finished epoch.
	virtual void OnEpochEnd(OaItTraining& InIter) { (void)InIter; }

	// Fired once after Finish(). Use for summary prints, file flush, etc.
	virtual void OnTrainEnd(OaItTraining& InIter) { (void)InIter; }

protected:
	OaCbTraining() = default;
};
