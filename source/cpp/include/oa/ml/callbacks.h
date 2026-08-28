// OA Ml — built-in training callbacks (Keras-style)
//
// Built-in callbacks cover the common keras model.fit() use cases. attach via
// oa::ItTraining config or addCallback(&cb). All are stateful, header-only
// where possible, and don't allocate per-step.
//
// Stateful callback types use the `Cb` prefix (for example,
// oa::CbProgressBar and oa::CbCheckpoint). Stateless operations remain in
// schema-owned oa::Fn* namespaces.

#pragma once

#include <oa/core/callback.h>
#include <oa/ml/itTraining.h>
#include <oa/ml/checkpoint.h>
#include <oa/ml/config.h>
#include <oa/ml/lrScheduler.h>
#include <oa/ml/optim.h>
#include <oa/ml/metric.h>

#include <stdio.h>

namespace oa {

// ─── oa::CbProgressBar ──────────────────────────────────────────────────────
//
// tqdm/keras-hybrid progress bar with `█`+`░` and rolling metrics:
//
//   epoch 1/5
//    938/938 |██████████| 0.65s · 0.7 ms/step · 1.26M sample/s · accuracy: 0.9091 · loss: 0.2914
//   epoch 2/5
//    938/938 |██████████| 0.59s · 0.6 ms/step · 1.31M sample/s · accuracy: 0.9134 · loss: 0.1822
//
// mid-epoch:
//   468/938 |█████░░░░░| 0.32s · 0.7 ms/step · 1.25M sample/s · loss: 0.3128
//
// Per-step updates rewrite the same line via `\r`; epoch end leaves the final
// line and starts a new one. Uses lastLoss() so per-
// step refresh never forces a Sync. Accuracy comes from recordAccuracy(); if
// the caller hasn't set it (NaN), the field is omitted. Wall time per step
// and workload throughput are shown instead of GPU time. Latency and rates are
// derived from the iterator's single workload definition; they are not metrics
// that callers must register separately.

class CbProgressBar : public oa::CbTraining {
public:
	explicit CbProgressBar(oa::I32 inBarWidth = 10) : barWidth_(inBarWidth) {}

	// Add a metric to render in the progress bar. Non-owning pointer.
	void addMetric(oa::Metric* inMetric) {
		metrics_.pushBack(inMetric);
	}

	// Suppress the "epoch N/M" header line — used with oa::CbPhase, which prints
	// phase-relative epoch headers instead.
	void setShowEpochHeader(bool inShow) { showEpochHeader_ = inShow; }

	void onTrainBegin(oa::ItTraining& inIter) override {
		(void)inIter;
		lastPrintT_ = oa::steadyNow();
	}

	void onEpochBegin(oa::ItTraining& inIter) override {
		if (showEpochHeader_ and inIter.totalEpochs() > 0) {
			::printf("epoch %lld/%lld\n",
				static_cast<long long>(inIter.epoch()),
				static_cast<long long>(inIter.totalEpochs()));
		}
	}

	void onStepEnd(oa::ItTraining& inIter) override {
		// EpochEnd owns the boundary redraw/newline; do not render it twice.
		if (inIter.isEpochBoundary()) return;
		// Throttle to ~30 redraws/sec so printf doesn't slow training down.
		auto now = oa::steadyNow();
		const oa::F64 sinceMs = (now - lastPrintT_).toMilliseconds();
		const bool isBoundary = inIter.isEpochBoundary() or inIter.isLastStep();
		if (not isBoundary and sinceMs < 33.0) return;
		lastPrintT_ = now;

		render(inIter, /*inFinalize=*/isBoundary);
		if (isBoundary) {
			lastFinalizedStep_ = inIter.stepCount();
			::fputc('\n', stdout);
			::fflush(stdout);
		}
	}

	void onEpochEnd(oa::ItTraining& inIter) override {
		render(inIter, /*inFinalize=*/true);
		lastFinalizedStep_ = inIter.stepCount();
		::fputc('\n', stdout);
		::fflush(stdout);
	}

	void onTrainEnd(oa::ItTraining& inIter) override {
		if (lastFinalizedStep_ == inIter.stepCount()) return;
		if (inIter.totalEpochs() == 0 or (inIter.epoch() == inIter.totalEpochs() and not inIter.isEpochBoundary())) {
			render(inIter, /*inFinalize=*/true);
			::fputc('\n', stdout);
		}
	}

private:
	// format any per-second rate with compact SI suffixes.
	static const char* formatRate(oa::F64 inRate, char* outBuf, oa::Usize inBufSize) {
		if (inRate >= 1e6) {
			::snprintf(outBuf, inBufSize, "%.2fM", inRate / 1e6);
		} else if (inRate >= 1e3) {
			::snprintf(outBuf, inBufSize, "%.2fK", inRate / 1e3);
		} else {
			::snprintf(outBuf, inBufSize, "%.0f", inRate);
		}
		return outBuf;
	}

	static void renderRate(oa::F64 inRate, const char* inUnit) {
		char buf[32];
		::printf("%s %s/s", formatRate(inRate, buf, sizeof(buf)), inUnit);
	}

	// tqdm-style 8-level partial-fill chars (1/8 .. 8/8 of a cell).
	static const char* partialChar(oa::I32 inEighths) {
		switch (inEighths) {
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

	void render(const oa::ItTraining& inIter, bool inFinalize) const {
		const oa::I64 stepNow   = inIter.hasEpochs() ? inIter.stepInEpoch() : inIter.stepCount();
		const oa::I64 stepTotal = inIter.hasEpochs() ? inIter.stepsInCurrentEpoch() : inIter.totalSteps();
		const oa::I64 width     = static_cast<oa::I64>(barWidth_);

		::printf("\r  ");
		if (stepTotal > 0) {
			::printf("%lld/%lld ", static_cast<long long>(stepNow), static_cast<long long>(stepTotal));
			// Smooth fill: progress in eighths of a cell across the bar.
			const oa::I64 totalEighths = (stepNow * width * 8 + (stepTotal / 2)) / stepTotal;
			const oa::I64 fullCells    = totalEighths / 8;
			const oa::I32 partial      = static_cast<oa::I32>(totalEighths % 8);
			::printf("|");
			for (oa::I64 i = 0; i < fullCells and i < width; ++i) ::printf("█");
			if (fullCells < width and partial > 0) {
				::printf("%s", partialChar(partial));
				for (oa::I64 i = fullCells + 1; i < width; ++i) ::printf("░");
			} else {
				for (oa::I64 i = fullCells; i < width; ++i) ::printf("░");
			}
			::printf("| ");
		} else {
			::printf("step %lld ", static_cast<long long>(stepNow));
		}

		const oa::F64 elapsed = inIter.hasEpochs() ? inIter.epochSeconds() : inIter.elapsedSeconds();
		const oa::F64 completedSteps = static_cast<oa::F64>(stepNow);
		const oa::F64 msPerStep = completedSteps > 0.0
			? elapsed * 1000.0 / completedSteps : 0.0;
		const oa::F64 samplesPerSecond = elapsed > 0.0
			? completedSteps * inIter.cfg().batchSize / elapsed : 0.0;
		::printf("%.2fs · %.2f ms/step · ", elapsed, msPerStep);
		renderRate(samplesPerSecond, "sample");
		if (inIter.cfg().sequenceLength > 0) {
			::printf(" · ");
			renderRate(samplesPerSecond * inIter.cfg().sequenceLength,
				inIter.cfg().sequenceUnit.cStr());
		}
		if (inIter.totalSourceUnits() > 0) {
			::printf(" · ");
			renderRate(inIter.hasEpochs() ? inIter.epochSourceUnitsPerSecond()
				: inIter.wallSourceUnitsPerSecond(), inIter.cfg().sourceUnit.cStr());
		}

		// Render custom metrics
		if (!metrics_.empty()) {
			::printf(" · ");
		}
		for (oa::Usize i = 0; i < metrics_.size(); ++i) {
			char metricBuf[64];
			const oa::I32 written = metrics_[i]->render(metricBuf, sizeof(metricBuf), false);
			if (written > 0) {
				::printf("%s", metricBuf);
				// Add separator between metrics, not after the last one
				if (i < metrics_.size() - 1) {
					::printf(" · ");
				}
			}
		}

		// Trim trailing spaces so partial redraws don't leave stale chars.
		::printf("        ");
		::fflush(stdout);
		(void)inFinalize;
	}

	oa::I32 barWidth_;
	bool  showEpochHeader_ = true;
	oa::SteadyTimePoint lastPrintT_;
	oa::I64 lastFinalizedStep_ = 0;
	oa::Vector<oa::Metric*> metrics_;
};

// ─── oa::CbValidation ───────────────────────────────────────────────────────
//
// Runs inference-only validation after every epoch. step-only runs can provide
// an interval and are always evaluated once at train end. The evaluator owns
// batching/model semantics and returns the sample-weighted mean loss. Validation
// time is excluded from training throughput and printed separately. register
// this callback before checkpoint and early-stopping callbacks, then pass
// metricPtr() to both.

struct ValidationResult {
	oa::F64 loss = oa::Limits<oa::F64>::quietNaN();
	oa::I64 batches = 0;
	oa::I64 samples = 0;
};

class ValidationMetric final : public oa::MetricLoss {
public:
	explicit ValidationMetric(oa::String inName) : oa::MetricLoss(oa::move(inName)) {}
	[[nodiscard]] oa::F64 result() const override {
		return count() > 0 ? oa::MetricLoss::result()
			: oa::Limits<oa::F64>::quietNaN();
	}
};

class CbValidation : public oa::CbTraining {
public:
	using EvalFn = oa::Fn<ValidationResult(oa::ItTraining&)>;

	explicit CbValidation(EvalFn inEval, oa::String inMetricName = "val_loss",
		oa::I64 inStepInterval = 0)
		: eval_(oa::move(inEval)), metric_(oa::move(inMetricName)),
		  stepInterval_(inStepInterval) {}

	void onStepEnd(oa::ItTraining& inIter) override {
		if (not inIter.hasEpochs() and stepInterval_ > 0
			and inIter.stepCount() % stepInterval_ == 0) run(inIter);
	}
	void onEpochEnd(oa::ItTraining& inIter) override { run(inIter); }
	void onTrainEnd(oa::ItTraining& inIter) override {
		if (not inIter.hasEpochs() and lastEvalStep_ != inIter.stepCount()) run(inIter);
	}

	[[nodiscard]] oa::MetricLoss* metricPtr() { return &metric_; }
	[[nodiscard]] const oa::MetricLoss& metric() const { return metric_; }
	[[nodiscard]] const ValidationResult& lastResult() const { return lastResult_; }
	[[nodiscard]] oa::F64 lastSeconds() const { return lastSeconds_; }

private:
	void run(oa::ItTraining& inIter) {
		const auto begin = oa::steadyNow();
		lastResult_ = eval_(inIter);
		lastSeconds_ = (oa::steadyNow() - begin).toSeconds();
		inIter.excludeWallTime(lastSeconds_);
		lastEvalStep_ = inIter.stepCount();

		metric_.reset();
		if (oa::isFinite(lastResult_.loss) and lastResult_.batches > 0) {
			metric_.update(static_cast<oa::F32>(lastResult_.loss));
			::printf("Validation: %s %.6f · %lld batches · %lld samples · %.2fs\n",
				metric_.name(), lastResult_.loss,
				static_cast<long long>(lastResult_.batches),
				static_cast<long long>(lastResult_.samples), lastSeconds_);
		} else {
			::printf("Validation: %s n/a · %.2fs\n", metric_.name(), lastSeconds_);
		}
		::fflush(stdout);
	}

	EvalFn eval_;
	ValidationMetric metric_;
	ValidationResult lastResult_;
	oa::F64 lastSeconds_ = 0.0;
	oa::I64 lastEvalStep_ = -1;
	oa::I64 stepInterval_ = 0;
};

// ─── oa::CbCheckpoint ───────────────────────────────────────────────────────
//
// keras modelCheckpoint(save_freq="epoch") + EarlyStopping's
// restore_best_weights, on top of oa::CheckpointManager. Every epoch end writes
// a resumable rotating checkpoint. Pass SaveEvery > 0 to add mid-epoch
// checkpoints every N completed optimizer steps. The master model is updated
// only on improvement. at each epoch end it prints a TF-style mini summary:
//
//   epoch 3: cross_entropy improved from 0.4056 to 0.3486 — saving model
//   epoch 4: cross_entropy did not improve from 0.3486
//
// at train end, if RestoreBest is set and the best epoch wasn't the last one,
// the best checkpoint is loaded back into model + optimizer — you always walk
// away with the best weights, not whatever the final (possibly degraded)
// epoch produced.
//
// Monitored value: inMetric->result() when provided (e.g. a val_loss metric),
// otherwise the epoch mean train loss. Better/worse direction comes from the
// manager's lowerIsBetter config.

class CbCheckpoint : public oa::CbTraining {
public:
	CbCheckpoint(oa::CheckpointManager& inMgr, oa::Module& inModel,
	               oa::Optimizer& inOpt, oa::I64 inSaveEvery = 0,
	               oa::Metric* inMetric = nullptr, bool inRestoreBest = true,
	               bool inVerbose = true)
		: mgr_(inMgr), model_(inModel), opt_(inOpt), saveEvery_(inSaveEvery),
		  metric_(inMetric), restoreBest_(inRestoreBest), verbose_(inVerbose) {}

	[[nodiscard]] oa::Status getStatus() const override { return status_; }

	void onStepEnd(oa::ItTraining& inIter) override {
		if (not status_.isOk()) return;
		if (saveEvery_ <= 0 or inIter.isEpochBoundary()) return;
		if ((inIter.stepCount() % saveEvery_) != 0) return;
		// Use the exact current-step loss (not epoch mean) for step checkpoints.
		const oa::F64 metric = static_cast<oa::F64>(inIter.lastLoss());
		if (verbose_) {
			const long long step = static_cast<long long>(inIter.stepCount());
			// Leading \n breaks out of the progress bar's \r line so the
			// message isn't immediately overwritten by the next bar redraw.
			::printf("\nStep %lld: loss = %.6f — saving resumable checkpoint\n", step, metric);
			::fflush(stdout);
		}
		status_ = mgr_.saveIncremental(model_, opt_,
			static_cast<oa::U64>(inIter.stepCount()), metric, "loss");
	}

	void onEpochEnd(oa::ItTraining& inIter) override {
		if (not status_.isOk()) return;
		const oa::F64 metric = getMetric(inIter);
		const bool improved = mgr_.isBetter(metric);
		const oa::F64 prevBest = mgr_.bestMetric();
		if (verbose_) {
			const long long epoch = static_cast<long long>(inIter.epoch());
			// Leading \n breaks out of the progress bar's \r line.
			if (improved and not haveBest_) {
				::printf("\nEpoch %lld: %s = %.6f — saving model\n",
					epoch, metricName(), metric);
			} else if (improved) {
				::printf("\nEpoch %lld: %s improved from %.6f to %.6f — saving model\n",
					epoch, metricName(), prevBest, metric);
			} else {
				::printf("\nEpoch %lld: %s did not improve from %.6f\n",
					epoch, metricName(), prevBest);
			}
			::fflush(stdout);
		}
		lastEpoch_ = inIter.epoch();
		// Every epoch produces a resumable checkpoint. The manager updates the
		// master/best model only when the monitored metric improves.
		status_ = mgr_.maybeSave(model_, opt_,
			static_cast<oa::U64>(inIter.stepCount()), metric, /*inForce=*/true);
		if (status_.isOk() and improved) {
			haveBest_  = true;
			bestEpoch_ = inIter.epoch();
		}
	}

	void onTrainEnd(oa::ItTraining& inIter) override {
		if (not status_.isOk()) return;
		// step-only training (no epochs): save once at the end.
		if (not inIter.hasEpochs()) {
			status_ = mgr_.maybeSave(model_, opt_,
				static_cast<oa::U64>(inIter.stepCount()), getMetric(inIter));
			return;
		}
		if (not restoreBest_) {
			// A completed epoch (including a partial final epoch) was already saved
			// by onEpochEnd. Only an early stop inside an epoch needs a final save.
			if (not inIter.isEpochBoundary()) {
				status_ = mgr_.maybeSave(model_, opt_,
					static_cast<oa::U64>(inIter.stepCount()), getMetric(inIter),
					/*inForce=*/true);
			}
			return;
		}
		if (not haveBest_) return;
		// Even when the last completed epoch was the best one, an interruption
		// inside the following epoch has already changed the live weights.  Only
		// skip the reload when training ended exactly at that best boundary.
		if (inIter.isEpochBoundary() and bestEpoch_ == lastEpoch_) return;
		auto status = mgr_.loadBestInto(model_, opt_);
		if (status.isOk()) {
			::printf("Restoring model weights from the end of the best epoch: %lld (%s %.6f)\n",
				static_cast<long long>(bestEpoch_), metricName(), mgr_.bestMetric());
		} else {
			status_ = status;
			::printf("Restore best weights failed: %s\n", status.getMessage().cStr());
		}
	}

	[[nodiscard]] oa::I64 bestEpoch() const { return bestEpoch_; }

private:
	[[nodiscard]] const char* metricName() const {
		// Explicit metric wins; otherwise use the manager's configured name
		// (the loss name from YAML) — never hardcode "loss".
		return metric_ ? metric_->name() : mgr_.metricName().cStr();
	}

	oa::F64 getMetric(oa::ItTraining& inIter) const {
		if (metric_) {
			return metric_->result();
		}
		// Fallback to epoch mean loss if no metric specified
		return inIter.epochMeanLoss();
	}

	oa::CheckpointManager& mgr_;
	oa::Module&            model_;
	oa::Optimizer&         opt_;
	oa::I64                saveEvery_;
	oa::Metric*            metric_;
	bool                 restoreBest_;
	bool                 verbose_;
	oa::Status             status_ = oa::Status::ok();
	bool                 haveBest_  = false;
	oa::I64                bestEpoch_ = 0;
	oa::I64                lastEpoch_ = 0;
};

// ─── oa::CbEarlyStop ────────────────────────────────────────────────────────
//
// keras EarlyStopping. calls iter.requestStop() once the monitored value has
// not improved for `patience` epochs — the training while-loop exits on the
// next isDone(). Pair with oa::CbCheckpoint(RestoreBest) to also get
// restore_best_weights semantics. Patience: number of epochs without
// improvement to tolerate. MinDelta: minimum change to count as improvement.
// Mode: Min for loss (lower is better), Max for accuracy (higher is better).
// If inMetric is provided, uses that metric's value; otherwise epoch mean loss.

enum class EarlyStopMode : oa::U8 { Min, Max };

class CbEarlyStop : public oa::CbTraining {
public:
	CbEarlyStop(oa::I64 inPatience = 5, oa::F64 inMinDelta = 1e-4,
		EarlyStopMode inMode = EarlyStopMode::Min, oa::Metric* inMetric = nullptr)
		: patience_(inPatience), minDelta_(inMinDelta), mode_(inMode), metric_(inMetric) {}

	void onEpochEnd(oa::ItTraining& inIter) override {
		const oa::F64 metric = getMetric(inIter);
		const bool improved = mode_ == EarlyStopMode::Min
			? metric < bestMetric_ - minDelta_
			: metric > bestMetric_ + minDelta_;

		if (improved) {
			bestMetric_ = metric;
			badEpochs_  = 0;
			return;
		}
		++badEpochs_;
		if (badEpochs_ >= patience_) {
			::printf("epoch %lld: early stopping — %s did not improve for %lld epochs (best %.6f)\n",
				static_cast<long long>(inIter.epoch()), metricName(),
				static_cast<long long>(badEpochs_), bestMetric_);
			stop_ = true;
			inIter.requestStop();
		}
	}

	[[nodiscard]] bool shouldStop() const { return stop_; }

private:
	[[nodiscard]] const char* metricName() const {
		return metric_ ? metric_->name() : "loss";
	}

	oa::F64 getMetric(oa::ItTraining& inIter) const {
		if (metric_) {
			return metric_->result();
		}
		// Fallback to epoch mean loss if no metric specified
		return inIter.epochMeanLoss();
	}

	oa::I64 patience_;
	oa::F64 minDelta_;
	EarlyStopMode mode_;
	oa::Metric* metric_;
	oa::F64 bestMetric_ = mode_ == EarlyStopMode::Min
		? oa::Limits<oa::F64>::max()
		: oa::Limits<oa::F64>::lowest();
	oa::I64 badEpochs_ = 0;
	bool  stop_      = false;
};

// ─── oa::LrSchedulerCallback ────────────────────────────────────────────────
//
// Applies any oa::LRScheduler to the optimizer at each step. Use oa::CosineScheduler,
// oa::OneCycleScheduler, oa::WarmupScheduler, etc. from <oa/ml/optim.h>.

class CbLrScheduler : public oa::CbTraining {
public:
	CbLrScheduler(oa::LRScheduler& inScheduler, oa::Optimizer& inOpt)
		: scheduler_(inScheduler), opt_(inOpt) {}

	void onStepEnd(oa::ItTraining& inIter) override {
		// The completed step already consumed the current LR; install the value for
		// the next one. Schedulers use one-based training-step semantics.
		const oa::F32 lr = scheduler_.getLr(static_cast<oa::U64>(inIter.stepCount() + 1));
		opt_.setLr(lr);
	}

private:
	oa::LRScheduler& scheduler_;
	oa::Optimizer&   opt_;
};

// ─── oa::CsvLoggerCallback ──────────────────────────────────────────────────
//
// Appends one exact row per completed optimizer step. Rate columns use explicit
// units; there is no logging cadence hidden inside the metric lifecycle.

class CbCsvLogger : public oa::CbTraining {
public:
	explicit CbCsvLogger(const oa::String& inPath);
	~CbCsvLogger() override;

	void onTrainBegin(oa::ItTraining& inIter) override;
	void onStepEnd(oa::ItTraining& inIter) override;
	void onEpochEnd(oa::ItTraining& inIter) override;
	void onTrainEnd(oa::ItTraining& inIter) override;

private:
	class Impl;
	oa::UniquePtr<Impl> impl_;
};

// ─── oa::CbSummary ─────────────────────────────────────────────────────────
//
// Prints a final training summary at onTrainEnd: loss, wall latency/throughput,
// GPU mean/p50/p95, the wall-to-GPU timing gap, and total duration. Optionally tracks
// initial loss for comparison. example output:
//
//   summary:
//     loss: initial 2.6558 · final 0.2965 · mean 0.4812
//     Wall: 0.07 ms/step · 943.68K sample/s · 60.40M token/s
//     GPU: mean 0.051 ms/step · p50 0.049 · p95 0.061 · 1.26M sample/s
//     run: 0.32s · 4686 steps · batch 64 · sequence 64 token/sample

class CbSummary : public oa::CbTraining {
public:
	explicit CbSummary(bool inTrackInitialLoss = true) : trackInitialLoss_(inTrackInitialLoss) {}
	void setValidationMetric(const oa::Metric* inMetric) { validationMetric_ = inMetric; }

	void onStepEnd(oa::ItTraining& inIter) override {
		if (trackInitialLoss_ && !initialLossRecorded_ && inIter.stepCount() == 1) {
			initialLoss_ = inIter.lastLoss();
			initialLossRecorded_ = true;
		}
	}

	void onTrainEnd(oa::ItTraining& inIter) override {
		const oa::F64 totalSec = inIter.elapsedSeconds();
		const oa::F64 wallSps = totalSec > 0.0
			? static_cast<oa::F64>(inIter.totalSamples()) / totalSec : 0.0;
		const oa::F64 wallUps = totalSec > 0.0
			? static_cast<oa::F64>(inIter.totalUnits()) / totalSec : 0.0;
		const oa::F64 wallMsPerStep = inIter.stepCount() > 0
			? totalSec * 1000.0 / static_cast<oa::F64>(inIter.stepCount()) : 0.0;

		const oa::GpuTimingStats gpuStats = inIter.gpuTimingStats();
		const oa::F64 gpuMs = gpuStats.meanMs;
		const oa::F64 gpuSps = gpuMs > 0.0
			? static_cast<oa::F64>(inIter.cfg().batchSize) / (gpuMs / 1000.0) : 0.0;
		const oa::F64 gpuUps = gpuSps * static_cast<oa::F64>(inIter.cfg().sequenceLength);
		const oa::F64 rawWallGpuGap = gpuSps > 0.0 ? 100.0 * (1.0 - wallSps / gpuSps) : 0.0;
		const oa::F64 wallGpuGap = rawWallGpuGap < 0.0 ? 0.0
			: (rawWallGpuGap > 100.0 ? 100.0 : rawWallGpuGap);

		char spsBuf[32];
		auto formatSps = [](oa::F64 sps, char* buf, oa::Usize sz) -> const char* {
			if (sps >= 1e6) ::snprintf(buf, sz, "%.2fM", sps / 1e6);
			else if (sps >= 1e3) ::snprintf(buf, sz, "%.2fK", sps / 1e3);
			else ::snprintf(buf, sz, "%.0f", sps);
			return buf;
		};

		const oa::F64 finalLoss = static_cast<oa::F64>(inIter.lastLoss());
		const oa::F64 meanLoss = inIter.trainingMeanLoss();
		::printf("\nSummary:\n");
		if (inIter.lastLossStep() == 0) {
			::printf("  loss: n/a (no loss recorded)\n");
		} else if (trackInitialLoss_ && initialLossRecorded_) {
			::printf("  loss: initial %.6f · final %.6f · mean %.6f\n",
				static_cast<double>(initialLoss_), finalLoss, meanLoss);
		} else {
			::printf("  loss: final %.6f · mean %.6f\n", finalLoss, meanLoss);
		}
		if (validationMetric_ and oa::isFinite(validationMetric_->result())) {
			::printf("  Validation: %s %.6f\n", validationMetric_->name(),
				validationMetric_->result());
		}
		::printf("  Wall: %.2f ms/step · %s sample/s",
			wallMsPerStep, formatSps(wallSps, spsBuf, sizeof(spsBuf)));
		if (inIter.cfg().sequenceLength > 0) {
			::printf(" · %s %s/s", formatSps(wallUps, spsBuf, sizeof(spsBuf)),
				inIter.cfg().sequenceUnit.cStr());
		}
		if (inIter.totalSourceUnits() > 0) {
			::printf(" · %s %s/s",
				formatSps(inIter.wallSourceUnitsPerSecond(), spsBuf, sizeof(spsBuf)),
				inIter.cfg().sourceUnit.cStr());
		}
		::fputc('\n', stdout);
			if (inIter.lastGpuTimeStep() == 0) {
				::printf("  GPU: n/a (timer unavailable)\n");
			} else {
				::printf("  GPU: mean %.3f ms/step · p50 %.3f · p95 %.3f · %s sample/s",
					gpuMs, gpuStats.medianMs, gpuStats.p95Ms,
					formatSps(gpuSps, spsBuf, sizeof(spsBuf)));
				if (inIter.cfg().sequenceLength > 0) {
					::printf(" · %s %s/s", formatSps(gpuUps, spsBuf, sizeof(spsBuf)),
						inIter.cfg().sequenceUnit.cStr());
				}
				if (inIter.totalSourceUnits() > 0) {
					::printf(" · %s %s/s",
						formatSps(inIter.gpuSourceUnitsPerSecond(), spsBuf, sizeof(spsBuf)),
						inIter.cfg().sourceUnit.cStr());
				}
				::printf(" · wall-GPU gap %.0f%%\n", wallGpuGap);
			}
		::printf("  run: %.2fs · %lld steps · batch %d",
			totalSec, static_cast<long long>(inIter.stepCount()), inIter.cfg().batchSize);
		if (inIter.cfg().sequenceLength > 0) {
			::printf(" · sequence %d %s/sample", inIter.cfg().sequenceLength,
				inIter.cfg().sequenceUnit.cStr());
		}
		::fputc('\n', stdout);
	}

private:
	bool trackInitialLoss_;
	bool initialLossRecorded_ = false;
	oa::F32 initialLoss_ = 0.0F;
	const oa::Metric* validationMetric_ = nullptr;
};

// ─── oa::CbPhase ─────────────────────────────────────────────────────────────
//
// Multi-phase training callback. phases are consecutive ranges of epochs —
// build the iterator with oa::ItTrainingConfig::epochSteps so epoch boundaries
// follow the phase schedule, then register one addPhase() per phase in order.
//
// Prints a schedule preview at train begin and a phase banner + phase-relative
// epoch headers (disable the progress bar's own header via SetShowEpochHeader):
//
//   phase schedule:
//     1. warmup — 1 epoch (2000 steps)
//     2. main   — 10 epochs (20000 steps)
//
//   phase 1/2 — warmup · 1 epoch × 2000 steps
//   epoch 1/1
//    2000/2000 |██████████| ...
//
// The OnPhaseBegin hook fires on entering each phase (including the first) —
// use it to swap LR schedulers, change datasets, etc.

class CbPhase : public oa::CbTraining {
public:
	struct Phase {
		oa::String id;
		oa::I64 epochs = 0;
		oa::I64 steps  = 0;  // total steps across the phase's epochs
	};
	using PhaseBeginFn = oa::Fn<void(oa::I32 inPhaseIdx, const Phase& inPhase)>;

	void addPhase(oa::String inId, oa::I64 inEpochs, oa::I64 inSteps) {
		phases_.pushBack({.id = oa::move(inId), .epochs = inEpochs, .steps = inSteps});
	}

	void setOnPhaseBegin(PhaseBeginFn inFn) { onPhaseBegin_ = oa::move(inFn); }

	[[nodiscard]] oa::I32 currentPhase() const { return currentPhase_; }

	void onTrainBegin(oa::ItTraining& inIter) override {
		(void)inIter;
		if (phases_.empty()) return;
		::printf("phase schedule:\n");
		for (oa::Usize i = 0; i < phases_.size(); ++i) {
			const Phase& p = phases_[i];
			::printf("  %zu. %-8s — %lld epoch%s (%lld steps)\n",
				i + 1, p.id.cStr(), static_cast<long long>(p.epochs),
				p.epochs == 1 ? "" : "s", static_cast<long long>(p.steps));
		}
	}

	void onEpochBegin(oa::ItTraining& inIter) override {
		if (phases_.empty()) return;
		const oa::I64 epoch = inIter.epoch();  // 1-based global epoch

		// map global epoch -> (phase, epoch within phase).
		oa::I64 epochsBefore = 0;
		oa::I32 phaseIdx = static_cast<oa::I32>(phases_.size()) - 1;
		for (oa::Usize i = 0; i < phases_.size(); ++i) {
			if (epoch <= epochsBefore + phases_[i].epochs) {
				phaseIdx = static_cast<oa::I32>(i);
				break;
			}
			epochsBefore += phases_[i].epochs;
		}
		const Phase& phase = phases_[static_cast<oa::Usize>(phaseIdx)];

		if (phaseIdx != currentPhase_) {
			currentPhase_ = phaseIdx;
			const oa::I64 stepsPerEpoch = phase.epochs > 0 ? phase.steps / phase.epochs : phase.steps;
			::printf("\nPhase %d/%zu — %s · %lld epoch%s × %lld steps\n",
				phaseIdx + 1, phases_.size(), phase.id.cStr(),
				static_cast<long long>(phase.epochs), phase.epochs == 1 ? "" : "s",
				static_cast<long long>(stepsPerEpoch));
			if (onPhaseBegin_) onPhaseBegin_(phaseIdx, phase);
		}

		::printf("epoch %lld/%lld\n",
			static_cast<long long>(epoch - epochsBefore),
			static_cast<long long>(phase.epochs));
	}

private:
	oa::Vector<Phase> phases_;
	PhaseBeginFn       onPhaseBegin_;
	oa::I32              currentPhase_ = -1;
};

} // namespace oa
