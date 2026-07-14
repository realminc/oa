// Oa Ml — built-in training callbacks (Keras-style)
//
// Built-in callbacks cover the common Keras model.fit() use cases. Attach via
// OaItTraining config or AddCallback(&cb). All are stateful, header-only
// where possible, and don't allocate per-step.
//
// Naming convention: OaCb<Name> (e.g. OaCbProgressBar, OaCbCheckpoint)
// to match the OaFn* / OaIt* / OaCb* pattern across the codebase.

#pragma once

#include <Oa/Core/Callback.h>
#include <Oa/Ml/ItTraining.h>
#include <Oa/Ml/Checkpoint.h>
#include <Oa/Ml/Config.h>
#include <Oa/Ml/LrScheduler.h>
#include <Oa/Ml/Optim.h>
#include <Oa/Ml/Metric.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <vector>

// ─── OaCbProgressBar ──────────────────────────────────────────────────────
//
// tqdm/Keras-hybrid progress bar with `█`+`░` and rolling metrics:
//
//   Epoch 1/5
//    938/938 |██████████| 0.65s · 0.7 ms/step · 1.26M sample/s · accuracy: 0.9091 · loss: 0.2914
//   Epoch 2/5
//    938/938 |██████████| 0.59s · 0.6 ms/step · 1.31M sample/s · accuracy: 0.9134 · loss: 0.1822
//
// Mid-epoch:
//   468/938 |█████░░░░░| 0.32s · 0.7 ms/step · 1.25M sample/s · loss: 0.3128
//
// Per-step updates rewrite the same line via `\r`; epoch end leaves the final
// line and starts a new one. Uses LiveLoss() (lazy host-coherent read) so per-
// step refresh never forces a Sync. Accuracy comes from RecordAccuracy(); if
// the caller hasn't set it (NaN), the field is omitted. Wall time per step
// and workload throughput are shown instead of GPU time. Latency and rates are
// derived from the iterator's single workload definition; they are not metrics
// that callers must register separately.

class OaCbProgressBar : public OaCbTraining {
public:
	explicit OaCbProgressBar(OaI32 InBarWidth = 10) : BarWidth_(InBarWidth) {}

	// Add a metric to render in the progress bar. Non-owning pointer.
	void AddMetric(OaMetric* InMetric) {
		Metrics_.push_back(InMetric);
	}

	// Suppress the "Epoch N/M" header line — used with OaCbPhase, which prints
	// phase-relative epoch headers instead.
	void SetShowEpochHeader(bool InShow) { ShowEpochHeader_ = InShow; }

	void OnTrainBegin(OaItTraining& InIter) override {
		(void)InIter;
		LastPrintT_ = std::chrono::steady_clock::now();
	}

	void OnEpochBegin(OaItTraining& InIter) override {
		if (ShowEpochHeader_ and InIter.TotalEpochs() > 0) {
			std::printf("Epoch %lld/%lld\n",
				static_cast<long long>(InIter.Epoch()),
				static_cast<long long>(InIter.TotalEpochs()));
		}
	}

	void OnStepEnd(OaItTraining& InIter) override {
		// EpochEnd owns the boundary redraw/newline; do not render it twice.
		if (InIter.IsEpochBoundary()) return;
		// Throttle to ~30 redraws/sec so printf doesn't slow training down.
		auto now = std::chrono::steady_clock::now();
		const OaF64 sinceMs = std::chrono::duration<double, std::milli>(now - LastPrintT_).count();
		const bool isBoundary = InIter.IsEpochBoundary() or InIter.IsLastStep();
		if (not isBoundary and sinceMs < 33.0) return;
		LastPrintT_ = now;

		Render(InIter, /*InFinalize=*/isBoundary);
		if (isBoundary) {
			LastFinalizedStep_ = InIter.StepCount();
			std::fputc('\n', stdout);
			std::fflush(stdout);
		}
	}

	void OnEpochEnd(OaItTraining& InIter) override {
		Render(InIter, /*InFinalize=*/true);
		LastFinalizedStep_ = InIter.StepCount();
		std::fputc('\n', stdout);
		std::fflush(stdout);
	}

	void OnTrainEnd(OaItTraining& InIter) override {
		if (LastFinalizedStep_ == InIter.StepCount()) return;
		if (InIter.TotalEpochs() == 0 or (InIter.Epoch() == InIter.TotalEpochs() and not InIter.IsEpochBoundary())) {
			Render(InIter, /*InFinalize=*/true);
			std::fputc('\n', stdout);
		}
	}

private:
	// Format any per-second rate with compact SI suffixes.
	static const char* FormatRate(OaF64 InRate, char* OutBuf, size_t InBufSize) {
		if (InRate >= 1e6) {
			std::snprintf(OutBuf, InBufSize, "%.2fM", InRate / 1e6);
		} else if (InRate >= 1e3) {
			std::snprintf(OutBuf, InBufSize, "%.2fK", InRate / 1e3);
		} else {
			std::snprintf(OutBuf, InBufSize, "%.0f", InRate);
		}
		return OutBuf;
	}

	static void RenderRate(OaF64 InRate, const char* InUnit) {
		char buf[32];
		std::printf("%s %s/s", FormatRate(InRate, buf, sizeof(buf)), InUnit);
	}

	// tqdm-style 8-level partial-fill chars (1/8 .. 8/8 of a cell).
	static const char* PartialChar(OaI32 InEighths) {
		switch (InEighths) {
			case 0:  return "";    // empty — use ░ instead
			case 1:  return "▏";
			case 2:  return "▎";
			case 3:  return "▍";
			case 4:  return "▌";
			case 5:  return "▋";
			case 6:  return "▊";
			case 7:  return "▉";
			default: return "█";  // 8/8 — full block
		}
	}

	void Render(const OaItTraining& InIter, bool InFinalize) const {
		const OaI64 stepNow   = InIter.HasEpochs() ? InIter.StepInEpoch() : InIter.StepCount();
		const OaI64 stepTotal = InIter.HasEpochs() ? InIter.StepsInCurrentEpoch() : InIter.TotalSteps();
		const OaI64 width     = static_cast<OaI64>(BarWidth_);

		std::printf("\r  ");
		if (stepTotal > 0) {
			std::printf("%lld/%lld ", static_cast<long long>(stepNow), static_cast<long long>(stepTotal));
			// Smooth fill: progress in eighths of a cell across the bar.
			const OaI64 totalEighths = (stepNow * width * 8 + (stepTotal / 2)) / stepTotal;
			const OaI64 fullCells    = totalEighths / 8;
			const OaI32 partial      = static_cast<OaI32>(totalEighths % 8);
			std::printf("|");
			for (OaI64 i = 0; i < fullCells and i < width; ++i) std::printf("█");
			if (fullCells < width and partial > 0) {
				std::printf("%s", PartialChar(partial));
				for (OaI64 i = fullCells + 1; i < width; ++i) std::printf("░");
			} else {
				for (OaI64 i = fullCells; i < width; ++i) std::printf("░");
			}
			std::printf("| ");
		} else {
			std::printf("step %lld ", static_cast<long long>(stepNow));
		}

		const OaF64 elapsed = InIter.HasEpochs() ? InIter.EpochSeconds() : InIter.ElapsedSeconds();
		const OaF64 completedSteps = static_cast<OaF64>(stepNow);
		const OaF64 msPerStep = completedSteps > 0.0
			? elapsed * 1000.0 / completedSteps : 0.0;
		const OaF64 samplesPerSecond = elapsed > 0.0
			? completedSteps * InIter.Cfg().BatchSize / elapsed : 0.0;
		std::printf("%.2fs · %.2f ms/step · ", elapsed, msPerStep);
		RenderRate(samplesPerSecond, "sample");
		if (InIter.Cfg().SequenceLength > 0) {
			std::printf(" · ");
			RenderRate(samplesPerSecond * InIter.Cfg().SequenceLength,
				InIter.Cfg().SequenceUnit.CStr());
		}
		if (InIter.TotalSourceUnits() > 0) {
			std::printf(" · ");
			RenderRate(InIter.HasEpochs() ? InIter.EpochSourceUnitsPerSecond()
				: InIter.WallSourceUnitsPerSecond(), InIter.Cfg().SourceUnit.CStr());
		}

		// Render custom metrics
		if (!Metrics_.empty()) {
			std::printf(" · ");
		}
		for (size_t i = 0; i < Metrics_.size(); ++i) {
			char metricBuf[64];
			const OaI32 written = Metrics_[i]->Render(metricBuf, sizeof(metricBuf), false);
			if (written > 0) {
				std::printf("%s", metricBuf);
				// Add separator between metrics, not after the last one
				if (i < Metrics_.size() - 1) {
					std::printf(" · ");
				}
			}
		}

		// Trim trailing spaces so partial redraws don't leave stale chars.
		std::printf("        ");
		std::fflush(stdout);
		(void)InFinalize;
	}

	OaI32 BarWidth_;
	bool  ShowEpochHeader_ = true;
	std::chrono::steady_clock::time_point LastPrintT_;
	OaI64 LastFinalizedStep_ = 0;
	std::vector<OaMetric*> Metrics_;
};

// ─── OaCbValidation ───────────────────────────────────────────────────────
//
// Runs inference-only validation after every epoch. Step-only runs can provide
// an interval and are always evaluated once at train end. The evaluator owns
// batching/model semantics and returns the sample-weighted mean loss. Validation
// time is excluded from training throughput and printed separately. Register
// this callback before checkpoint and early-stopping callbacks, then pass
// MetricPtr() to both.

struct OaValidationResult {
	OaF64 Loss = std::numeric_limits<OaF64>::quiet_NaN();
	OaI64 Batches = 0;
	OaI64 Samples = 0;
};

class OaValidationMetric final : public OaMetricLoss {
public:
	explicit OaValidationMetric(OaString InName) : OaMetricLoss(std::move(InName)) {}
	[[nodiscard]] OaF64 Result() const override {
		return Count() > 0 ? OaMetricLoss::Result()
			: std::numeric_limits<OaF64>::quiet_NaN();
	}
};

class OaCbValidation : public OaCbTraining {
public:
	using EvalFn = std::function<OaValidationResult(OaItTraining&)>;

	explicit OaCbValidation(EvalFn InEval, OaString InMetricName = "val_loss",
		OaI64 InStepInterval = 0)
		: Eval_(std::move(InEval)), Metric_(std::move(InMetricName)),
		  StepInterval_(InStepInterval) {}

	void OnStepEnd(OaItTraining& InIter) override {
		if (not InIter.HasEpochs() and StepInterval_ > 0
			and InIter.StepCount() % StepInterval_ == 0) Run(InIter);
	}
	void OnEpochEnd(OaItTraining& InIter) override { Run(InIter); }
	void OnTrainEnd(OaItTraining& InIter) override {
		if (not InIter.HasEpochs() and LastEvalStep_ != InIter.StepCount()) Run(InIter);
	}

	[[nodiscard]] OaMetricLoss* MetricPtr() { return &Metric_; }
	[[nodiscard]] const OaMetricLoss& Metric() const { return Metric_; }
	[[nodiscard]] const OaValidationResult& LastResult() const { return LastResult_; }
	[[nodiscard]] OaF64 LastSeconds() const { return LastSeconds_; }

private:
	void Run(OaItTraining& InIter) {
		const auto begin = std::chrono::steady_clock::now();
		LastResult_ = Eval_(InIter);
		LastSeconds_ = std::chrono::duration<OaF64>(std::chrono::steady_clock::now() - begin).count();
		InIter.ExcludeWallTime(LastSeconds_);
		LastEvalStep_ = InIter.StepCount();

		Metric_.Reset();
		if (std::isfinite(LastResult_.Loss) and LastResult_.Batches > 0) {
			Metric_.Update(static_cast<OaF32>(LastResult_.Loss));
			std::printf("Validation: %s %.6f · %lld batches · %lld samples · %.2fs\n",
				Metric_.Name(), LastResult_.Loss,
				static_cast<long long>(LastResult_.Batches),
				static_cast<long long>(LastResult_.Samples), LastSeconds_);
		} else {
			std::printf("Validation: %s n/a · %.2fs\n", Metric_.Name(), LastSeconds_);
		}
		std::fflush(stdout);
	}

	EvalFn Eval_;
	OaValidationMetric Metric_;
	OaValidationResult LastResult_;
	OaF64 LastSeconds_ = 0.0;
	OaI64 LastEvalStep_ = -1;
	OaI64 StepInterval_ = 0;
};

// ─── OaCbCheckpoint ───────────────────────────────────────────────────────
//
// Keras ModelCheckpoint(save_freq="epoch") + EarlyStopping's
// restore_best_weights, on top of OaCheckpointManager. Every epoch end writes
// a resumable rotating checkpoint. Pass SaveEvery > 0 to add mid-epoch
// checkpoints every N completed optimizer steps. The master model is updated
// only on improvement. At each epoch end it prints a TF-style mini summary:
//
//   Epoch 3: cross_entropy improved from 0.4056 to 0.3486 — saving model
//   Epoch 4: cross_entropy did not improve from 0.3486
//
// At train end, if RestoreBest is set and the best epoch wasn't the last one,
// the best checkpoint is loaded back into model + optimizer — you always walk
// away with the best weights, not whatever the final (possibly degraded)
// epoch produced.
//
// Monitored value: InMetric->Result() when provided (e.g. a val_loss metric),
// otherwise the epoch mean train loss. Better/worse direction comes from the
// manager's LowerIsBetter config.

class OaCbCheckpoint : public OaCbTraining {
public:
	OaCbCheckpoint(OaCheckpointManager& InMgr, OaModule& InModel,
	               OaOptimizer& InOpt, OaI64 InSaveEvery = 0,
	               OaMetric* InMetric = nullptr, bool InRestoreBest = true,
	               bool InVerbose = true)
		: Mgr_(InMgr), Model_(InModel), Opt_(InOpt), SaveEvery_(InSaveEvery),
		  Metric_(InMetric), RestoreBest_(InRestoreBest), Verbose_(InVerbose) {}

	void OnStepEnd(OaItTraining& InIter) override {
		if (SaveEvery_ <= 0 or InIter.IsEpochBoundary()) return;
		if ((InIter.StepCount() % SaveEvery_) != 0) return;
		// Use the exact current-step loss (not epoch mean) for step checkpoints.
		const OaF64 metric = static_cast<OaF64>(InIter.LastLoss());
		if (Verbose_) {
			const long long step = static_cast<long long>(InIter.StepCount());
			// Leading \n breaks out of the progress bar's \r line so the
			// message isn't immediately overwritten by the next bar redraw.
			std::printf("\nStep %lld: loss = %.6f — saving resumable checkpoint\n", step, metric);
			std::fflush(stdout);
		}
		(void)Mgr_.SaveIncremental(Model_, Opt_, static_cast<OaU64>(InIter.StepCount()),
			metric, "loss");
	}

	void OnEpochEnd(OaItTraining& InIter) override {
		const OaF64 metric = GetMetric(InIter);
		const bool improved = Mgr_.IsBetter(metric);
		const OaF64 prevBest = Mgr_.GetBestMetric();
		if (Verbose_) {
			const long long epoch = static_cast<long long>(InIter.Epoch());
			// Leading \n breaks out of the progress bar's \r line.
			if (improved and not HaveBest_) {
				std::printf("\nEpoch %lld: %s = %.6f — saving model\n",
					epoch, MetricName(), metric);
			} else if (improved) {
				std::printf("\nEpoch %lld: %s improved from %.6f to %.6f — saving model\n",
					epoch, MetricName(), prevBest, metric);
			} else {
				std::printf("\nEpoch %lld: %s did not improve from %.6f\n",
					epoch, MetricName(), prevBest);
			}
			std::fflush(stdout);
		}
		if (improved) {
			HaveBest_  = true;
			BestEpoch_ = InIter.Epoch();
		}
		LastEpoch_ = InIter.Epoch();
		// Every epoch produces a resumable checkpoint. The manager updates the
		// master/best model only when the monitored metric improves.
		(void)Mgr_.MaybeSave(Model_, Opt_, static_cast<OaU64>(InIter.StepCount()),
			metric, /*InForce=*/true);
	}

	void OnTrainEnd(OaItTraining& InIter) override {
		// Step-only training (no epochs): save once at the end.
		if (not InIter.HasEpochs()) {
			(void)Mgr_.MaybeSave(Model_, Opt_, static_cast<OaU64>(InIter.StepCount()),
				GetMetric(InIter));
			return;
		}
		if (not RestoreBest_) {
			// A completed epoch (including a partial final epoch) was already saved
			// by OnEpochEnd. Only an early stop inside an epoch needs a final save.
			if (not InIter.IsEpochBoundary()) {
				(void)Mgr_.MaybeSave(Model_, Opt_, static_cast<OaU64>(InIter.StepCount()),
					GetMetric(InIter), /*InForce=*/true);
			}
			return;
		}
		if (not HaveBest_) return;
		// Even when the last completed epoch was the best one, an interruption
		// inside the following epoch has already changed the live weights.  Only
		// skip the reload when training ended exactly at that best boundary.
		if (InIter.IsEpochBoundary() and BestEpoch_ == LastEpoch_) return;
		auto status = Mgr_.LoadBestInto(Model_, Opt_);
		if (status.IsOk()) {
			std::printf("Restoring model weights from the end of the best epoch: %lld (%s %.6f)\n",
				static_cast<long long>(BestEpoch_), MetricName(), Mgr_.GetBestMetric());
		} else {
			std::printf("Restore best weights failed: %s\n", status.GetMessage().c_str());
		}
	}

	[[nodiscard]] OaI64 BestEpoch() const { return BestEpoch_; }

private:
	[[nodiscard]] const char* MetricName() const {
		// Explicit metric wins; otherwise use the manager's configured name
		// (the loss name from YAML) — never hardcode "loss".
		return Metric_ ? Metric_->Name() : Mgr_.MetricName().c_str();
	}

	OaF64 GetMetric(OaItTraining& InIter) const {
		if (Metric_) {
			return Metric_->Result();
		}
		// Fallback to epoch mean loss if no metric specified
		return InIter.EpochMeanLoss();
	}

	OaCheckpointManager& Mgr_;
	OaModule&            Model_;
	OaOptimizer&         Opt_;
	OaI64                SaveEvery_;
	OaMetric*            Metric_;
	bool                 RestoreBest_;
	bool                 Verbose_;
	bool                 HaveBest_  = false;
	OaI64                BestEpoch_ = 0;
	OaI64                LastEpoch_ = 0;
};

// ─── OaCbEarlyStop ────────────────────────────────────────────────────────
//
// Keras EarlyStopping. Calls iter.RequestStop() once the monitored value has
// not improved for `patience` epochs — the training while-loop exits on the
// next IsDone(). Pair with OaCbCheckpoint(RestoreBest) to also get
// restore_best_weights semantics. Patience: number of epochs without
// improvement to tolerate. MinDelta: minimum change to count as improvement.
// Mode: Min for loss (lower is better), Max for accuracy (higher is better).
// If InMetric is provided, uses that metric's value; otherwise epoch mean loss.

enum class OaEarlyStopMode : OaU8 { Min, Max };

class OaCbEarlyStop : public OaCbTraining {
public:
	OaCbEarlyStop(OaI64 InPatience = 5, OaF64 InMinDelta = 1e-4,
		OaEarlyStopMode InMode = OaEarlyStopMode::Min, OaMetric* InMetric = nullptr)
		: Patience_(InPatience), MinDelta_(InMinDelta), Mode_(InMode), Metric_(InMetric) {}

	void OnEpochEnd(OaItTraining& InIter) override {
		const OaF64 metric = GetMetric(InIter);
		const bool improved = Mode_ == OaEarlyStopMode::Min
			? metric < BestMetric_ - MinDelta_
			: metric > BestMetric_ + MinDelta_;

		if (improved) {
			BestMetric_ = metric;
			BadEpochs_  = 0;
			return;
		}
		++BadEpochs_;
		if (BadEpochs_ >= Patience_) {
			std::printf("Epoch %lld: early stopping — %s did not improve for %lld epochs (best %.6f)\n",
				static_cast<long long>(InIter.Epoch()), MetricName(),
				static_cast<long long>(BadEpochs_), BestMetric_);
			Stop_ = true;
			InIter.RequestStop();
		}
	}

	[[nodiscard]] bool ShouldStop() const { return Stop_; }

private:
	[[nodiscard]] const char* MetricName() const {
		return Metric_ ? Metric_->Name() : "loss";
	}

	OaF64 GetMetric(OaItTraining& InIter) const {
		if (Metric_) {
			return Metric_->Result();
		}
		// Fallback to epoch mean loss if no metric specified
		return InIter.EpochMeanLoss();
	}

	OaI64 Patience_;
	OaF64 MinDelta_;
	OaEarlyStopMode Mode_;
	OaMetric* Metric_;
	OaF64 BestMetric_ = Mode_ == OaEarlyStopMode::Min
		? std::numeric_limits<OaF64>::max()
		: std::numeric_limits<OaF64>::lowest();
	OaI64 BadEpochs_ = 0;
	bool  Stop_      = false;
};

// ─── OaLrSchedulerCallback ────────────────────────────────────────────────
//
// Applies any OaLRScheduler to the optimizer at each step. Use OaCosineScheduler,
// OaOneCycleScheduler, OaWarmupScheduler, etc. from Oa/Ml/Optim.h.

class OaCbLrScheduler : public OaCbTraining {
public:
	OaCbLrScheduler(OaLRScheduler& InScheduler, OaOptimizer& InOpt)
		: Scheduler_(InScheduler), Opt_(InOpt) {}

	void OnStepEnd(OaItTraining& InIter) override {
		// The completed step already consumed the current LR; install the value for
		// the next one. Schedulers use one-based training-step semantics.
		const OaF32 lr = Scheduler_.GetLr(static_cast<OaU64>(InIter.StepCount() + 1));
		Opt_.SetLr(lr);
	}

private:
	OaLRScheduler& Scheduler_;
	OaOptimizer&   Opt_;
};

// ─── OaCsvLoggerCallback ──────────────────────────────────────────────────
//
// Appends one exact row per completed optimizer step. Rate columns use explicit
// units; there is no logging cadence hidden inside the metric lifecycle.

class OaCbCsvLogger : public OaCbTraining {
public:
	explicit OaCbCsvLogger(const OaString& InPath) : Path_(InPath) {}

	void OnTrainBegin(OaItTraining& InIter) override {
		(void)InIter;
		File_.open(Path_.c_str(), std::ios::out | std::ios::trunc);
		if (File_.is_open()) {
			File_ << "epoch,step,batch_size,sequence_length,sequence_unit,source_unit,loss,gpu_ms,wall_ms_per_step,"
			         "wall_samples_per_second,gpu_samples_per_second,wall_units_per_second,gpu_units_per_second,"
			         "wall_source_units_per_second,gpu_source_units_per_second\n";
			File_.flush();
		}
	}

	void OnStepEnd(OaItTraining& InIter) override {
		if (not File_.is_open()) return;
		File_ << InIter.Epoch() << ","
		      << InIter.StepCount() << ","
		      << InIter.Cfg().BatchSize << ","
		      << InIter.Cfg().SequenceLength << ","
		      << InIter.Cfg().SequenceUnit.CStr() << ","
		      << InIter.Cfg().SourceUnit.CStr() << ","
		      << InIter.LastLoss() << ","
		      << InIter.LastGpuMs() << ","
		      << InIter.WallMsPerStep() << ","
		      << InIter.WallSamplesPerSecond() << ","
		      << InIter.GpuSamplesPerSecond() << ","
		      << InIter.WallUnitsPerSecond() << ","
		      << InIter.GpuUnitsPerSecond() << ","
		      << InIter.WallSourceUnitsPerSecond() << ","
		      << InIter.GpuSourceUnitsPerSecond() << "\n";
	}

	void OnEpochEnd(OaItTraining&) override {
		if (File_.is_open()) File_.flush();
	}

	void OnTrainEnd(OaItTraining& InIter) override {
		(void)InIter;
		if (File_.is_open()) File_.close();
	}

private:
	OaString      Path_;
	std::ofstream File_;
};

// ─── OaCbSummary ─────────────────────────────────────────────────────────
//
// Prints a final training summary at OnTrainEnd: loss, wall latency/throughput,
// GPU mean/p50/p95, CPU overhead, and total duration. Optionally tracks
// initial loss for comparison. Example output:
//
//   Summary:
//     Loss: initial 2.6558 · final 0.2965 · mean 0.4812
//     Wall: 0.07 ms/step · 943.68K sample/s · 60.40M token/s
//     GPU: mean 0.051 ms/step · p50 0.049 · p95 0.061 · 1.26M sample/s
//     Run: 0.32s · 4686 steps · batch 64 · sequence 64 token/sample

class OaCbSummary : public OaCbTraining {
public:
	explicit OaCbSummary(bool InTrackInitialLoss = true) : TrackInitialLoss_(InTrackInitialLoss) {}
	void SetValidationMetric(const OaMetric* InMetric) { ValidationMetric_ = InMetric; }

	void OnStepEnd(OaItTraining& InIter) override {
		if (TrackInitialLoss_ && !InitialLossRecorded_ && InIter.StepCount() == 1) {
			InitialLoss_ = InIter.LastLoss();
			InitialLossRecorded_ = true;
		}
	}

	void OnTrainEnd(OaItTraining& InIter) override {
		const OaF64 totalSec = InIter.ElapsedSeconds();
		const OaF64 wallSps = totalSec > 0.0
			? static_cast<OaF64>(InIter.TotalSamples()) / totalSec : 0.0;
		const OaF64 wallUps = totalSec > 0.0
			? static_cast<OaF64>(InIter.TotalUnits()) / totalSec : 0.0;
		const OaF64 wallMsPerStep = InIter.StepCount() > 0
			? totalSec * 1000.0 / static_cast<OaF64>(InIter.StepCount()) : 0.0;

		const OaGpuTimingStats gpuStats = InIter.GpuTimingStats();
		const OaF64 gpuMs = gpuStats.MeanMs;
		const OaF64 gpuSps = gpuMs > 0.0
			? static_cast<OaF64>(InIter.Cfg().BatchSize) / (gpuMs / 1000.0) : 0.0;
		const OaF64 gpuUps = gpuSps * static_cast<OaF64>(InIter.Cfg().SequenceLength);
		const OaF64 rawCpuOverhead = gpuSps > 0.0 ? 100.0 * (1.0 - wallSps / gpuSps) : 0.0;
		const OaF64 cpuOverhead = rawCpuOverhead < 0.0 ? 0.0
			: (rawCpuOverhead > 100.0 ? 100.0 : rawCpuOverhead);

		char spsBuf[32];
		auto FormatSps = [](OaF64 sps, char* buf, size_t sz) -> const char* {
			if (sps >= 1e6) std::snprintf(buf, sz, "%.2fM", sps / 1e6);
			else if (sps >= 1e3) std::snprintf(buf, sz, "%.2fK", sps / 1e3);
			else std::snprintf(buf, sz, "%.0f", sps);
			return buf;
		};

		const OaF64 finalLoss = static_cast<OaF64>(InIter.LastLoss());
		const OaF64 meanLoss = InIter.TrainingMeanLoss();
		std::printf("\nSummary:\n");
		if (InIter.LastLossStep() == 0) {
			std::printf("  Loss: n/a (no loss recorded)\n");
		} else if (TrackInitialLoss_ && InitialLossRecorded_) {
			std::printf("  Loss: initial %.6f · final %.6f · mean %.6f\n",
				static_cast<double>(InitialLoss_), finalLoss, meanLoss);
		} else {
			std::printf("  Loss: final %.6f · mean %.6f\n", finalLoss, meanLoss);
		}
		if (ValidationMetric_ and std::isfinite(ValidationMetric_->Result())) {
			std::printf("  Validation: %s %.6f\n", ValidationMetric_->Name(),
				ValidationMetric_->Result());
		}
		std::printf("  Wall: %.2f ms/step · %s sample/s",
			wallMsPerStep, FormatSps(wallSps, spsBuf, sizeof(spsBuf)));
		if (InIter.Cfg().SequenceLength > 0) {
			std::printf(" · %s %s/s", FormatSps(wallUps, spsBuf, sizeof(spsBuf)),
				InIter.Cfg().SequenceUnit.CStr());
		}
		if (InIter.TotalSourceUnits() > 0) {
			std::printf(" · %s %s/s",
				FormatSps(InIter.WallSourceUnitsPerSecond(), spsBuf, sizeof(spsBuf)),
				InIter.Cfg().SourceUnit.CStr());
		}
		std::fputc('\n', stdout);
			if (InIter.LastGpuTimeStep() == 0) {
				std::printf("  GPU: n/a (timer unavailable)\n");
			} else {
				std::printf("  GPU: mean %.3f ms/step · p50 %.3f · p95 %.3f · %s sample/s",
					gpuMs, gpuStats.MedianMs, gpuStats.P95Ms,
					FormatSps(gpuSps, spsBuf, sizeof(spsBuf)));
				if (InIter.Cfg().SequenceLength > 0) {
					std::printf(" · %s %s/s", FormatSps(gpuUps, spsBuf, sizeof(spsBuf)),
						InIter.Cfg().SequenceUnit.CStr());
				}
				if (InIter.TotalSourceUnits() > 0) {
					std::printf(" · %s %s/s",
						FormatSps(InIter.GpuSourceUnitsPerSecond(), spsBuf, sizeof(spsBuf)),
						InIter.Cfg().SourceUnit.CStr());
				}
				std::printf(" · CPU overhead %.0f%%\n", cpuOverhead);
			}
		std::printf("  Run: %.2fs · %lld steps · batch %d",
			totalSec, static_cast<long long>(InIter.StepCount()), InIter.Cfg().BatchSize);
		if (InIter.Cfg().SequenceLength > 0) {
			std::printf(" · sequence %d %s/sample", InIter.Cfg().SequenceLength,
				InIter.Cfg().SequenceUnit.CStr());
		}
		std::fputc('\n', stdout);
	}

private:
	bool TrackInitialLoss_;
	bool InitialLossRecorded_ = false;
	OaF32 InitialLoss_ = 0.0F;
	const OaMetric* ValidationMetric_ = nullptr;
};

// ─── OaCbPhase ─────────────────────────────────────────────────────────────
//
// Multi-phase training callback. Phases are consecutive ranges of epochs —
// build the iterator with OaItTrainingConfig::EpochSteps so epoch boundaries
// follow the phase schedule, then register one AddPhase() per phase in order.
//
// Prints a schedule preview at train begin and a phase banner + phase-relative
// epoch headers (disable the progress bar's own header via SetShowEpochHeader):
//
//   Phase schedule:
//     1. warmup — 1 epoch (2000 steps)
//     2. main   — 10 epochs (20000 steps)
//
//   Phase 1/2 — warmup · 1 epoch × 2000 steps
//   Epoch 1/1
//    2000/2000 |██████████| ...
//
// The OnPhaseBegin hook fires on entering each phase (including the first) —
// use it to swap LR schedulers, change datasets, etc.

class OaCbPhase : public OaCbTraining {
public:
	struct Phase {
		OaString Id;
		OaI64 Epochs = 0;
		OaI64 Steps  = 0;  // total steps across the phase's epochs
	};
	using PhaseBeginFn = std::function<void(OaI32 InPhaseIdx, const Phase& InPhase)>;

	void AddPhase(OaString InId, OaI64 InEpochs, OaI64 InSteps) {
		Phases_.push_back({.Id = std::move(InId), .Epochs = InEpochs, .Steps = InSteps});
	}

	void SetOnPhaseBegin(PhaseBeginFn InFn) { OnPhaseBegin_ = std::move(InFn); }

	[[nodiscard]] OaI32 CurrentPhase() const { return CurrentPhase_; }

	void OnTrainBegin(OaItTraining& InIter) override {
		(void)InIter;
		if (Phases_.empty()) return;
		std::printf("Phase schedule:\n");
		for (size_t i = 0; i < Phases_.size(); ++i) {
			const Phase& p = Phases_[i];
			std::printf("  %zu. %-8s — %lld epoch%s (%lld steps)\n",
				i + 1, p.Id.c_str(), static_cast<long long>(p.Epochs),
				p.Epochs == 1 ? "" : "s", static_cast<long long>(p.Steps));
		}
	}

	void OnEpochBegin(OaItTraining& InIter) override {
		if (Phases_.empty()) return;
		const OaI64 epoch = InIter.Epoch();  // 1-based global epoch

		// Map global epoch -> (phase, epoch within phase).
		OaI64 epochsBefore = 0;
		OaI32 phaseIdx = static_cast<OaI32>(Phases_.size()) - 1;
		for (size_t i = 0; i < Phases_.size(); ++i) {
			if (epoch <= epochsBefore + Phases_[i].Epochs) {
				phaseIdx = static_cast<OaI32>(i);
				break;
			}
			epochsBefore += Phases_[i].Epochs;
		}
		const Phase& phase = Phases_[static_cast<size_t>(phaseIdx)];

		if (phaseIdx != CurrentPhase_) {
			CurrentPhase_ = phaseIdx;
			const OaI64 stepsPerEpoch = phase.Epochs > 0 ? phase.Steps / phase.Epochs : phase.Steps;
			std::printf("\nPhase %d/%zu — %s · %lld epoch%s × %lld steps\n",
				phaseIdx + 1, Phases_.size(), phase.Id.c_str(),
				static_cast<long long>(phase.Epochs), phase.Epochs == 1 ? "" : "s",
				static_cast<long long>(stepsPerEpoch));
			if (OnPhaseBegin_) OnPhaseBegin_(phaseIdx, phase);
		}

		std::printf("Epoch %lld/%lld\n",
			static_cast<long long>(epoch - epochsBefore),
			static_cast<long long>(phase.Epochs));
	}

private:
	std::vector<Phase> Phases_;
	PhaseBeginFn       OnPhaseBegin_;
	OaI32              CurrentPhase_ = -1;
};
