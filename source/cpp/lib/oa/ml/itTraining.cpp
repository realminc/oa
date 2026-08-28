// oa::ItTraining — implementation

#include <oa/ml/itTraining.h>
#include <oa/ml/module.h>
#include <oa/ml/metric.h>
#include <oa/ml/optim.h>
#include <oa/ml/trainingProgram.h>
#include <oa/ml/trainingSession.h>
#include <oa/core/envFlag.h>
#include <oa/core/filesystem.h>
#include <oa/core/paths.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/deviceAccess.h>

oa::ItTraining::ItTraining(
	oa::Engine& inEngine,
	oa::Optimizer& inOpt,
	oa::ItTrainingConfig inCfg)
	: opt_(inOpt), cfg_(oa::move(inCfg)
) {
	cfg_.batchSize = oa::max(cfg_.batchSize, 1);
	cfg_.sequenceLength = oa::max(cfg_.sequenceLength, 0);
	cfg_.sourceUnitsPerSample = oa::max(cfg_.sourceUnitsPerSample, 0.0);
	// Variable epoch schedule: prefix sums, and the schedule owns totalSteps.
	if (not cfg_.epochSteps.empty()) {
		epochOffsets_.reserve(cfg_.epochSteps.size());
		oa::I64 sum = 0;
		for (const oa::I64 steps : cfg_.epochSteps) {
			sum += oa::max<oa::I64>(steps, 1);
			epochOffsets_.pushBack(sum);
		}
		cfg_.totalSteps = sum;
	}

	auto& session = oa::ExecutionSession::forEngine(inEngine);
	executionSession_ = &session;
	rt_ = &inEngine;

	if (cfg_.enableGpuTiming) {
		auto status = timer_.init(inEngine, cfg_.timerName.cStr());
		timerReady_ = status.isOk();
		if (not timerReady_) {
			OaLogWarn(oa::LogComponent::Ml,
				"oa::ItTraining: GPU timer init failed (%s); GPU timing disabled",
				status.getMessage().cStr());
		}
	}

	t0_      = oa::highResolutionNow();
	epochT0_ = t0_;
	lastStepT_ = t0_;
	trainingPhaseTiming_ = oa::EnvFlag::isSet("OA_LOG_TRAINING_PHASES");
	metrics_ = cfg_.metrics;

	// register config-time callbacks. addCallback can still be used after
	// construction; these are simply equivalent to calling it in a loop.
	for (auto* cb : cfg_.callbacks) {
		if (cb != nullptr) callbacks_.pushBack(cb);
	}
}

oa::ItTraining::~ItTraining() {
	closeStableResourceFrame_();
}

// ─── oa::Iterator interface ─────────────────────────────────────────────────

// Lazy advance: invoked from isDone() and step(). idempotent — no-op while
// the body for the current index_ is pending. Fires TrainBegin on first call
// and EpochBegin whenever we cross into a new epoch.
void oa::ItTraining::advanceIfNeeded_() {
	if (not trainBeginFired_) {
		trainBeginFired_ = true;
		fireTrainBegin();
		if (stopRequested_) return;
	}
	if (bodyPending_) return;
	bodyPending_ = true;
	++index_;
	// Skip EpochBegin if we just advanced past the last step — that final
	// advance is the isDone() probe that exits the while loop; the next
	// epoch never starts.
	if (cfg_.totalSteps > 0 and index_ > cfg_.totalSteps) return;
	if (hasEpochs()) {
		const oa::I64 currentEpoch = epoch();
		if (currentEpoch != lastEpochFired_) {
			epochLossSum_   = 0.0;
			epochLossCount_ = 0;
			epochSourceUnits_ = 0;
			epochT0_        = oa::highResolutionNow();
			lastEpochFired_ = currentEpoch;
			fireEpochBegin();
			if (stopRequested_) return;
		}
	}
	// step one is the warm-up: optimizers and autograd may lazily create
	// persistent state. Starting at step two, fixed-shape training allocations
	// are assigned stable context slots so the recorded graph can reuse exact
	// buffer identities on every following step.
	if (index_ > 1) {
		executionSession_->beginStableResourceFrame();
		stableResourceFrameOpen_ = true;
	}
	if (trainingPhaseTiming_) {
		phaseBodyT0_ = oa::highResolutionNow();
		phaseBodyStarted_ = true;
	}
}

bool oa::ItTraining::isDone() const {
	// Cooperative stop: don't advance into (or fire onEpochBegin for) a step
	// that will never run.
	if (stopRequested_) return true;
	if (cfg_.totalSteps > 0 and not bodyPending_ and index_ >= cfg_.totalSteps) {
		return true;
	}
	const_cast<oa::ItTraining*>(this)->advanceIfNeeded_();
	return stopRequested_;
}

void oa::ItTraining::reset() {
	closeStableResourceFrame_();
	if (cfg_.program != nullptr) {
		auto status = cfg_.program->reset();
		if (not status.isOk()) {
			OaLogWarn(oa::LogComponent::Ml,
				"oa::ItTraining::reset: training program reset failed: %s",
				status.getMessage().cStr());
		}
	}
	index_           = 0;
	totalSamples_    = 0;
	totalUnits_      = 0;
	totalSourceUnits_ = 0;
	gpuTimedSourceUnits_ = 0;
	pendingSourceUnits_ = -1;
	lastStepSourceUnits_ = 0;
	epochSourceUnits_ = 0;
	lastLoss_        = 0.0F;
	lastGpuMs_       = 0.0;
	lastLossStep_    = 0;
	lastGpuTimeStep_ = 0;
	gpuTimingSamples_.clear();
	gpuTimingSumMs_  = 0.0;
	liveAccuracy_    = oa::Limits<oa::F32>::quietNaN();
	stopRequested_   = false;
	lastStatus_      = oa::Status::ok();
	trainBeginFired_ = false;
	bodyPending_     = false;
	lastEpochFired_  = -1;
	epochLossSum_    = 0.0;
	epochLossCount_  = 0;
	trainingLossSum_ = 0.0;
	trainingLossCount_ = 0;
	t0_              = oa::highResolutionNow();
	epochT0_         = t0_;
	lastStepT_       = t0_;
	trainingPhaseStats_ = {};
	phaseBodyStarted_ = false;
	pendingLoss_     = oa::Matrix{};
	programLoss_     = oa::Matrix{};
	programCaptureDisabled_ = false;
	programReportWritten_ = false;
	resetMetrics_();
	if (session_ != nullptr) session_->onReset(*this);
}

bool oa::ItTraining::isEpochBoundary() const {
	if (index_ <= 0) return false;
	if (hasEpochs() and isLastStep()) return true;
	if (not epochOffsets_.empty()) {
		return oa::binarySearch(epochOffsets_.begin(), epochOffsets_.end(), index_);
	}
	if (cfg_.stepsPerEpoch <= 0) return false;
	return (index_ % cfg_.stepsPerEpoch) == 0;
}

bool oa::ItTraining::isLastStep() const {
	if (cfg_.totalSteps <= 0) return false;
	return index_ == cfg_.totalSteps;
}

// ─── epoch accessors ──────────────────────────────────────────────────────

oa::I64 oa::ItTraining::epochIndexForStep_(oa::I64 inStep) const {
	// first offset >= inStep is the epoch containing that step.
	auto it = oa::lowerBound(epochOffsets_.begin(), epochOffsets_.end(), inStep);
	if (it == epochOffsets_.end()) return static_cast<oa::I64>(epochOffsets_.size()) - 1;
	return static_cast<oa::I64>(it - epochOffsets_.begin());
}

oa::I64 oa::ItTraining::epoch() const {
	if (index_ == 0) return 0;
	if (not epochOffsets_.empty()) {
		return epochIndexForStep_(index_) + 1;
	}
	if (cfg_.stepsPerEpoch <= 0) return 0;
	return ((index_ - 1) / cfg_.stepsPerEpoch) + 1;
}

oa::I64 oa::ItTraining::stepInEpoch() const {
	if (not epochOffsets_.empty()) {
		if (index_ == 0) return 0;
		const oa::I64 epochIdx = epochIndexForStep_(index_);
		const oa::I64 begin = epochIdx == 0 ? 0 : epochOffsets_[static_cast<oa::Usize>(epochIdx - 1)];
		return index_ - begin;
	}
	if (cfg_.stepsPerEpoch <= 0) return index_;
	const oa::I64 mod = index_ % cfg_.stepsPerEpoch;
	return mod == 0 ? cfg_.stepsPerEpoch : mod;
}

oa::I64 oa::ItTraining::stepsInCurrentEpoch() const {
	if (not cfg_.epochSteps.empty()) {
		const oa::I64 epochIdx = index_ == 0 ? 0 : epochIndexForStep_(index_);
		return cfg_.epochSteps[static_cast<oa::Usize>(epochIdx)];
	}
	if (cfg_.stepsPerEpoch <= 0) return cfg_.totalSteps;
	if (cfg_.totalSteps > 0 and epoch() == totalEpochs()) {
		const oa::I64 completedBefore = (epoch() - 1) * cfg_.stepsPerEpoch;
		return oa::min(cfg_.stepsPerEpoch, cfg_.totalSteps - completedBefore);
	}
	return cfg_.stepsPerEpoch;
}

oa::I64 oa::ItTraining::totalEpochs() const {
	if (not cfg_.epochSteps.empty()) return static_cast<oa::I64>(cfg_.epochSteps.size());
	if (cfg_.stepsPerEpoch <= 0 or cfg_.totalSteps <= 0) return 0;
	return (cfg_.totalSteps + cfg_.stepsPerEpoch - 1) / cfg_.stepsPerEpoch;
}

// ─── loss tagging ─────────────────────────────────────────────────────────

void oa::ItTraining::recordLoss(const oa::Matrix& inLoss) {
	pendingLoss_ = inLoss;
}

// ─── step / Next ──────────────────────────────────────────────────────────

void oa::ItTraining::next() {
	// Body for step index_ has just run — finalize: record opt step, push the
	// recorded ops through execution, then fire StepEnd/EpochEnd. index_ stays
	// at the current step number; the next isDone() advances lazily.
	using Clock = oa::HighResolutionClock;
	const auto elapsedMs = [](oa::HighResolutionTimePoint inBegin,
		oa::HighResolutionTimePoint inEnd) -> oa::F64 {
		return (inEnd - inBegin).toMilliseconds();
	};
	const auto stepT0 = trainingPhaseTiming_ and phaseBodyStarted_
		? phaseBodyT0_ : Clock::TimePoint{};
	auto& session = *executionSession_;
	const auto failStep = [&](const oa::Status& inStatus, const char* inAction) {
		closeStableResourceFrame_();
		lastStatus_ = inStatus;
		OaLogError(oa::LogComponent::Ml,
			"oa::ItTraining: %s failed at step %lld: %s",
			inAction, static_cast<long long>(index_), inStatus.getMessage().cStr());
		stopRequested_ = true;
		phaseBodyStarted_ = false;
		bodyPending_ = false;
	};
	const oa::Bool replayExisting = cfg_.program != nullptr
		and cfg_.program->isCaptured();
	if (replayExisting and session.nodeCount() != 0) {
		auto status = oa::Status::error(oa::StatusCode::FailedPrecondition,
			"recorded new graph nodes after capture; use step(prepare, record) so "
			"only mapped input preparation runs on replay steps");
		session.clear();
		failStep(status, "program replay");
		return;
	}

	const auto optimizerT0 = trainingPhaseTiming_ ? Clock::now() : Clock::TimePoint{};
	if (trainingPhaseTiming_ and phaseBodyStarted_) {
		trainingPhaseStats_.bodyMs += elapsedMs(phaseBodyT0_, optimizerT0);
	}

	if (not replayExisting) opt_.step();
	const auto optimizerT1 = trainingPhaseTiming_ ? Clock::now() : Clock::TimePoint{};
	if (trainingPhaseTiming_) {
		trainingPhaseStats_.optimizerMs += elapsedMs(optimizerT0, optimizerT1);
	}

	oa::Bool usedProgram = false;
	if (cfg_.program != nullptr and not programCaptureDisabled_
		and (replayExisting or index_ > 1))
	{
		if (not replayExisting) {
			const auto captureT0 = Clock::now();
			oa::Vector<const oa::Matrix*> observedOutputs;
			if (pendingLoss_.hasStorage()) observedOutputs.pushBack(&pendingLoss_);
			oa::TrainingProgramOptions programOptions;
			programOptions.enableGpuTiming = timerReady_;
			programOptions.observedOutputs = {
				observedOutputs.data(), observedOutputs.size()};
			auto captureStatus = cfg_.program->capture(
				session.engine(), programOptions);
			const auto captureT1 = Clock::now();
			if (trainingPhaseTiming_) {
				trainingPhaseStats_.compileMs += elapsedMs(captureT0, captureT1);
			}
			if (not captureStatus.isOk()) {
				// capture is an optimization. The source graph is deliberately left
				// intact on rejection so this and all later steps remain correct eager
				// execution rather than turning an unsupported op into a hard failure.
				programCaptureDisabled_ = true;
				OaLogWarn(oa::LogComponent::Ml,
					"oa::ItTraining: static capture unavailable at step %lld (%s); "
					"continuing eagerly",
					static_cast<long long>(index_), captureStatus.getMessage().cStr());
			} else {
				usedProgram = true;
				programLoss_ = pendingLoss_;
				// Drop eager warm-up timing: captured samples use the timestamp bracket
				// embedded around the complete reusable device program.
				gpuTimingSamples_.clear();
				gpuTimingSumMs_ = 0.0;
				gpuTimedSourceUnits_ = 0;
				lastGpuMs_ = 0.0;
				lastGpuTimeStep_ = 0;
			}
		} else {
			usedProgram = true;
		}
	}

	if (usedProgram) {
		const auto submitT0 = Clock::now();
		auto replayStatus = cfg_.program->replay();
		const auto submitT1 = Clock::now();
		if (trainingPhaseTiming_) {
			trainingPhaseStats_.submitMs += elapsedMs(submitT0, submitT1);
		}
		if (not replayStatus.isOk()) {
			failStep(replayStatus, "training program submit");
			return;
		}
		const auto waitT0 = Clock::now();
		auto waitStatus = cfg_.program->wait();
		const auto waitT1 = Clock::now();
		if (trainingPhaseTiming_) {
			trainingPhaseStats_.waitMs += elapsedMs(waitT0, waitT1);
		}
		if (not waitStatus.isOk()) {
			failStep(waitStatus, "training program wait");
			return;
		}
		// Emit evidence only after the first completed replay. The report then
		// contains the actual submission/timeline completion event, rather than a
		// compiled-but-never-submitted placeholder.
		if (not programReportWritten_) {
			const oa::String reportSetting = oa::EnvFlag::getString("OA_GRAPH_REPORT");
			if (not reportSetting.empty()) {
				const oa::Path reportPath = reportSetting == "1"
					? oa::Paths::var("report") / "training_graph.json"
					: oa::Path(reportSetting);
				const auto parent = reportPath.parentPath();
				auto reportStatus = parent.empty()
					? oa::Status::ok() : oa::Filesystem::createDirectories(parent);
				if (reportStatus.isOk()) {
					const oa::String report = cfg_.program->debugReportJson("TrainingStep");
					reportStatus = oa::Filesystem::writeText(reportPath, report);
				}
				if (reportStatus.isOk()) {
					OaLogInfo(oa::LogComponent::Ml,
						"training graph report: %s", reportPath.string().cStr());
				} else {
					OaLogWarn(oa::LogComponent::Ml,
						"training graph report failed: %s",
						reportStatus.getMessage().cStr());
				}
				programReportWritten_ = true;
			}
		}
		if (replayExisting) opt_.notifyProgramReplay();
	} else {
		// A legal step may be host-only (for example oa::OptimizerNoOp in callback
		// lifecycle tests). submit() deliberately rejects an empty recording, so
		// only create a GPU event when this step actually recorded device work.
		if (session.nodeCount() != 0U) {
			oa::Timer* timer = timerReady_ ? &timer_ : nullptr;
			auto submitted = session.submit(timer);
			if (not submitted.isOk()) {
				failStep(submitted.getStatus(), "training step submit");
				return;
			}
			auto waitStatus = session.wait(submitted.getValue());
			if (not waitStatus.isOk()) {
				failStep(waitStatus, "training step wait");
				return;
			}
		}
		if (trainingPhaseTiming_) {
			const auto& runtime = session.lastExecutionStats();
			trainingPhaseStats_.compileMs += runtime.compileMs;
			trainingPhaseStats_.recordMs += runtime.recordMs;
			trainingPhaseStats_.submitMs += runtime.submitMs;
			trainingPhaseStats_.waitMs += runtime.waitMs;
		}
	}

	totalSamples_ += cfg_.batchSize;
	if (cfg_.sequenceLength > 0) {
		totalUnits_ += static_cast<oa::I64>(cfg_.batchSize) * cfg_.sequenceLength;
	}
	lastStepSourceUnits_ = pendingSourceUnits_ >= 0
		? pendingSourceUnits_
		: static_cast<oa::I64>(oa::round(
			static_cast<oa::F64>(cfg_.batchSize) * cfg_.sourceUnitsPerSample));
	totalSourceUnits_ += lastStepSourceUnits_;
	epochSourceUnits_ += lastStepSourceUnits_;

	// The step is complete, so the scalar is exact and safe to read. Every step
	// contributes exactly once to running and epoch metrics.
	const auto scalarMetricT0 = trainingPhaseTiming_ ? Clock::now() : Clock::TimePoint{};
	const oa::Matrix& completedLoss = usedProgram and programLoss_.hasStorage()
		? programLoss_ : pendingLoss_;
	if (completedLoss.hasStorage() and completedLoss.numElements() > 0) {
		lastLoss_      = completedLoss.at(0);
		lastLossStep_  = index_;
		epochLossSum_ += static_cast<oa::F64>(lastLoss_);
		++epochLossCount_;
		trainingLossSum_ += static_cast<oa::F64>(lastLoss_);
		++trainingLossCount_;
	}
	if (timerReady_ and usedProgram) {
		lastGpuMs_ = cfg_.program->lastGpuMs();
		if (lastGpuMs_ > 0.0) {
			lastGpuTimeStep_ = index_;
			gpuTimingSamples_.pushBack(lastGpuMs_);
			gpuTimingSumMs_ += lastGpuMs_;
			gpuTimedSourceUnits_ += lastStepSourceUnits_;
		}
	} else if (timerReady_) {
		auto elapsed = timer_.commit(session.engine());
		if (elapsed.isOk()) {
			lastGpuMs_ = elapsed.getValue();
		} else {
			timerReady_ = false;
			lastGpuMs_ = 0.0;
			OaLogWarn(oa::LogComponent::Ml,
				"oa::ItTraining: device timer commit failed (%s); timing disabled",
				elapsed.getStatus().getMessage().cStr());
		}
		if (lastGpuMs_ > 0.0) {
			lastGpuTimeStep_ = index_;
			gpuTimingSamples_.pushBack(lastGpuMs_);
			gpuTimingSumMs_ += lastGpuMs_;
			gpuTimedSourceUnits_ += lastStepSourceUnits_;
		}
	}
	updateMetrics_();
	lastStatus_ = oa::Status::ok();
	const auto scalarMetricT1 = trainingPhaseTiming_ ? Clock::now() : Clock::TimePoint{};
	if (trainingPhaseTiming_) {
		trainingPhaseStats_.scalarMetricMs += elapsedMs(scalarMetricT0, scalarMetricT1);
	}
	closeStableResourceFrame_();

	const auto callbackT0 = trainingPhaseTiming_ ? Clock::now() : Clock::TimePoint{};
	fireStepEnd();

	if (lastStatus_.isOk() and isEpochBoundary()) {
		fireEpochEnd();
	}
	if (trainingPhaseTiming_) {
		const auto callbackT1 = Clock::now();
		trainingPhaseStats_.callbackMs += elapsedMs(callbackT0, callbackT1);
		if (phaseBodyStarted_) trainingPhaseStats_.totalMs += elapsedMs(stepT0, callbackT1);
		++trainingPhaseStats_.count;
	}

	pendingLoss_ = oa::Matrix{};
	pendingSourceUnits_ = -1;
	phaseBodyStarted_ = false;
	bodyPending_ = false;  // ready for next isDone() to advance index_
	if (session_ != nullptr) session_->onStepCompleted(*this);
}

void oa::ItTraining::next(const oa::Matrix& inLoss) {
	recordLoss(inLoss);
	next();
}

void oa::ItTraining::step(const oa::Fn<void()>& inOpFn) {
	if (isDone()) return;  // also handles lazy advanceIfNeeded_
	const oa::Bool recordsStep = cfg_.program == nullptr
		or not cfg_.program->isCaptured();
	if (recordsStep) inOpFn();
	if (stableResourceFrameOpen_ and recordsStep) {
		// This overload provides no prepare/record boundary. Retain every stable
		// slot rather than guessing which allocations escape the captured step.
		executionSession_->sealAllStableResourcesExternal();
	}
	next();
}

void oa::ItTraining::step(
	const oa::Fn<void()>& inPrepareFn,
	const oa::Fn<void()>& inRecordFn)
{
	if (isDone()) return;  // also opens the stable frame before preparation
	const oa::Bool recordsStep = cfg_.program == nullptr
		or not cfg_.program->isCaptured();
	inPrepareFn();
	if (stableResourceFrameOpen_ and recordsStep) {
		executionSession_->sealStableResourceInputs();
	}
	if (recordsStep) inRecordFn();
	next();
}

oa::Status oa::ItTraining::finish() {
	closeStableResourceFrame_();
	if (cfg_.program != nullptr and cfg_.program->isCaptured()) {
		auto status = cfg_.program->wait();
		if (not status.isOk()) {
			if (session_ != nullptr) session_->onFinished(status, *this);
			return status;
		}
	}
	if (not lastStatus_.isOk()) {
		if (session_ != nullptr) session_->onFinished(lastStatus_, *this);
		return lastStatus_;
	}

	fireTrainEnd();
	if (not lastStatus_.isOk()) {
		if (session_ != nullptr) session_->onFinished(lastStatus_, *this);
		return lastStatus_;
	}
	if (trainingPhaseTiming_ and trainingPhaseStats_.count > 0) {
		const auto& s = trainingPhaseStats_;
		const oa::F64 total = s.mean(s.totalMs);
		const oa::F64 accounted = s.mean(s.accountedMs());
		OaLogInfo(oa::LogComponent::Ml,
			"training phases: steps=%lld total=%.3f ms/step body=%.3f optimizer=%.3f "
			"compile=%.3f record=%.3f submit=%.3f wait=%.3f scalar_metric=%.3f "
			"callbacks=%.3f unaccounted=%.3f",
			static_cast<long long>(s.count), total, s.mean(s.bodyMs),
			s.mean(s.optimizerMs), s.mean(s.compileMs), s.mean(s.recordMs),
			s.mean(s.submitMs), s.mean(s.waitMs), s.mean(s.scalarMetricMs),
			s.mean(s.callbackMs), oa::max<oa::F64>(total - accounted, 0.0));
	}
	auto status = oa::Status::ok();
	if (session_ != nullptr) session_->onFinished(status, *this);
	return status;
}

oa::Status oa::ItTraining::requestProgramRecapture() {
	if (cfg_.program == nullptr) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::ItTraining::requestProgramRecapture requires a configured program");
	}
	OA_RETURN_IF_ERROR(cfg_.program->reset());
	programCaptureDisabled_ = false;
	programLoss_ = oa::Matrix{};
	programReportWritten_ = false;
	return oa::Status::ok();
}

void oa::ItTraining::closeStableResourceFrame_() {
	if (not stableResourceFrameOpen_) return;
	executionSession_->endStableResourceFrame();
	stableResourceFrameOpen_ = false;
}

// ─── throughput accessors ─────────────────────────────────────────────────

oa::F64 oa::ItTraining::gpuSamplesPerSecond() const {
	if (gpuTimingSamples_.empty()) return 0.0;
	const oa::F64 meanMs = gpuTimingSumMs_ / static_cast<oa::F64>(gpuTimingSamples_.size());
	return meanMs > 0.0 ? static_cast<oa::F64>(cfg_.batchSize) / (meanMs / 1000.0) : 0.0;
}

oa::GpuTimingStats oa::ItTraining::gpuTimingStats() const {
	oa::GpuTimingStats stats;
	if (gpuTimingSamples_.empty()) return stats;

	oa::Vector<oa::F64> sorted = gpuTimingSamples_;
	oa::sort(sorted.begin(), sorted.end());

	const auto percentile = [&sorted](oa::F64 p) -> oa::F64 {
		if (sorted.empty()) return 0.0;
		const oa::F64 pos = p * static_cast<oa::F64>(sorted.size() - 1);
		const auto lo = static_cast<oa::Usize>(pos);
		const auto hi = oa::min(lo + 1, sorted.size() - 1);
		const oa::F64 t = pos - static_cast<oa::F64>(lo);
		return sorted[lo] * (1.0 - t) + sorted[hi] * t;
	};

	stats.count = static_cast<oa::I64>(gpuTimingSamples_.size());
	stats.meanMs = gpuTimingSumMs_ / static_cast<oa::F64>(gpuTimingSamples_.size());
	stats.minMs = sorted.front();
	stats.medianMs = percentile(0.50);
	stats.p95Ms = percentile(0.95);
	stats.lastMs = gpuTimingSamples_.back();
	return stats;
}

oa::F64 oa::ItTraining::elapsedSeconds() const {
	auto now = oa::highResolutionNow();
	return (now - t0_).toSeconds();
}

void oa::ItTraining::excludeWallTime(oa::F64 inSeconds) {
	if (inSeconds <= 0.0) return;
	const auto excluded = oa::Duration::fromDouble(inSeconds);
	t0_ += excluded;
	epochT0_ += excluded;
	lastStepT_ += excluded;
}

oa::F64 oa::ItTraining::wallSamplesPerSecond() const {
	const oa::F64 elapsed = elapsedSeconds();
	if (elapsed <= 0.0 or totalSamples_ == 0) return 0.0;
	return static_cast<oa::F64>(totalSamples_) / elapsed;
}

oa::F64 oa::ItTraining::gpuUnitsPerSecond() const {
	return cfg_.sequenceLength > 0
		? gpuSamplesPerSecond() * static_cast<oa::F64>(cfg_.sequenceLength) : 0.0;
}

oa::F64 oa::ItTraining::wallUnitsPerSecond() const {
	const oa::F64 elapsed = elapsedSeconds();
	return elapsed > 0.0 ? static_cast<oa::F64>(totalUnits_) / elapsed : 0.0;
}

oa::F64 oa::ItTraining::gpuSourceUnitsPerSecond() const {
	if (gpuTimingSumMs_ <= 0.0 or gpuTimedSourceUnits_ <= 0) return 0.0;
	return static_cast<oa::F64>(gpuTimedSourceUnits_) / (gpuTimingSumMs_ / 1000.0);
}

oa::F64 oa::ItTraining::wallSourceUnitsPerSecond() const {
	const oa::F64 elapsed = elapsedSeconds();
	return elapsed > 0.0 ? static_cast<oa::F64>(totalSourceUnits_) / elapsed : 0.0;
}

oa::F64 oa::ItTraining::epochSourceUnitsPerSecond() const {
	const oa::F64 elapsed = epochSeconds();
	return elapsed > 0.0 ? static_cast<oa::F64>(epochSourceUnits_) / elapsed : 0.0;
}

oa::F64 oa::ItTraining::wallMsPerStep() const {
	return index_ > 0 ? elapsedSeconds() * 1000.0 / static_cast<oa::F64>(index_) : 0.0;
}

oa::F64 oa::ItTraining::epochSampledMeanLoss() const {
	if (epochLossCount_ == 0) return 0.0;
	return epochLossSum_ / static_cast<oa::F64>(epochLossCount_);
}

oa::F64 oa::ItTraining::epochMeanLoss() const {
	return epochSampledMeanLoss();
}

oa::F64 oa::ItTraining::trainingMeanLoss() const {
	return trainingLossCount_ > 0
		? trainingLossSum_ / static_cast<oa::F64>(trainingLossCount_) : 0.0;
}

oa::F64 oa::ItTraining::epochSeconds() const {
	auto now = oa::highResolutionNow();
	return (now - epochT0_).toSeconds();
}

// ─── Callback dispatch ────────────────────────────────────────────────────

void oa::ItTraining::addCallback(oa::CbTraining* inCallback) {
	if (inCallback != nullptr) callbacks_.pushBack(inCallback);
}

void oa::ItTraining::addMetric(oa::Metric* inMetric) {
	if (inMetric != nullptr) metrics_.pushBack(inMetric);
}

void oa::ItTraining::resetMetrics_() {
	for (auto* metric : metrics_) metric->reset();
	lastStepT_ = oa::highResolutionNow();
}

void oa::ItTraining::updateMetrics_() {
	const auto now = oa::highResolutionNow();
	const oa::F64 seconds = (now - lastStepT_).toSeconds();
	for (auto* metric : metrics_) {
		metric->updateStep(lastLoss_, hasLossSample(), seconds, 1);
	}
	lastStepT_ = now;
}

void oa::ItTraining::fireTrainBegin() {
	// training wall time begins at the first iteration, excluding construction,
	// model/header setup, and time spent waiting before the loop starts.
	t0_ = oa::highResolutionNow();
	epochT0_ = t0_;
	resetMetrics_();
	for (auto* cb : callbacks_) {
		cb->onTrainBegin(*this);
		if (not captureCallbackStatus_(*cb, "onTrainBegin")) break;
	}
}

void oa::ItTraining::fireEpochBegin() {
	resetMetrics_();
	for (auto* cb : callbacks_) {
		cb->onEpochBegin(*this);
		if (not captureCallbackStatus_(*cb, "onEpochBegin")) break;
	}
}

void oa::ItTraining::fireStepEnd() {
	for (auto* cb : callbacks_) {
		cb->onStepEnd(*this);
		if (not captureCallbackStatus_(*cb, "onStepEnd")) break;
	}
}

void oa::ItTraining::fireEpochEnd() {
	for (auto* cb : callbacks_) {
		cb->onEpochEnd(*this);
		if (not captureCallbackStatus_(*cb, "onEpochEnd")) break;
	}
}

void oa::ItTraining::fireTrainEnd() {
	for (auto* cb : callbacks_) {
		cb->onTrainEnd(*this);
		if (not captureCallbackStatus_(*cb, "onTrainEnd")) break;
	}
}

bool oa::ItTraining::captureCallbackStatus_(
	oa::CbTraining& inCallback, const char* inPhase)
{
	const oa::Status status = inCallback.getStatus();
	if (status.isOk()) return true;
	if (lastStatus_.isOk()) lastStatus_ = status;
	stopRequested_ = true;
	OaLogError(oa::LogComponent::Ml,
		"oa::ItTraining: callback %s failed at step %lld: %s",
		inPhase, static_cast<long long>(index_), status.getMessage().cStr());
	return false;
}
