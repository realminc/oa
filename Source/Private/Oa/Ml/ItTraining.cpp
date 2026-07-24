// OaItTraining — implementation

#include <Oa/Ml/ItTraining.h>
#include <Oa/Ml/Module.h>
#include <Oa/Ml/Metric.h>
#include <Oa/Ml/Optim.h>
#include <Oa/Ml/TrainingProgram.h>
#include <Oa/Ml/TrainingSession.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Filesystem.h>
#include <Oa/Core/Paths.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/ExecutionMemory.h>

#include <algorithm>

OaItTraining::OaItTraining(OaOptimizer& InOpt, OaItTrainingConfig InCfg)
	: Opt_(InOpt), Cfg_(std::move(InCfg))
{
	Cfg_.BatchSize = std::max(Cfg_.BatchSize, 1);
	Cfg_.SequenceLength = std::max(Cfg_.SequenceLength, 0);
	Cfg_.SourceUnitsPerSample = std::max(Cfg_.SourceUnitsPerSample, 0.0);
	// Variable epoch schedule: prefix sums, and the schedule owns TotalSteps.
	if (not Cfg_.EpochSteps.empty()) {
		EpochOffsets_.reserve(Cfg_.EpochSteps.size());
		OaI64 sum = 0;
		for (const OaI64 steps : Cfg_.EpochSteps) {
			sum += std::max<OaI64>(steps, 1);
			EpochOffsets_.push_back(sum);
		}
		Cfg_.TotalSteps = sum;
	}

	auto& ctx = OaContext::GetDefault();
	auto* rt = ctx.GetEngine();
	Rt_ = rt;

	if (rt != nullptr and Cfg_.EnableGpuTiming) {
		auto status = Timer_.Init(*rt, Cfg_.TimerName.c_str());
		TimerReady_ = status.IsOk();
		if (not TimerReady_) {
			OA_LOG_WARN(OaLogComponent::ML,
				"OaItTraining: GPU timer init failed (%s); GPU timing disabled",
				status.GetMessage().c_str());
		}
	}

	T0_      = std::chrono::high_resolution_clock::now();
	EpochT0_ = T0_;
	LastStepT_ = T0_;
	TrainingPhaseTiming_ = OaEnvFlag::IsSet("OA_LOG_TRAINING_PHASES");
	Metrics_ = Cfg_.Metrics;

	// Register config-time callbacks. AddCallback can still be used after
	// construction; these are simply equivalent to calling it in a loop.
	for (auto* cb : Cfg_.Callbacks) {
		if (cb != nullptr) Callbacks_.push_back(cb);
	}
}

OaItTraining::~OaItTraining() {
	CloseStableResourceFrame_();
	if (TimerReady_ and Rt_ != nullptr) {
		Timer_.Destroy(Rt_->Device);
	}
}

// ─── OaIterator interface ─────────────────────────────────────────────────

// Lazy advance: invoked from IsDone() and Step(). Idempotent — no-op while
// the body for the current Index_ is pending. Fires TrainBegin on first call
// and EpochBegin whenever we cross into a new epoch.
void OaItTraining::AdvanceIfNeeded_() {
	if (not TrainBeginFired_) {
		TrainBeginFired_ = true;
		FireTrainBegin();
		if (StopRequested_) return;
	}
	if (BodyPending_) return;
	BodyPending_ = true;
	++Index_;
	// Skip EpochBegin if we just advanced past the last step — that final
	// advance is the IsDone() probe that exits the while loop; the next
	// epoch never starts.
	if (Cfg_.TotalSteps > 0 and Index_ > Cfg_.TotalSteps) return;
	if (HasEpochs()) {
		const OaI64 currentEpoch = Epoch();
		if (currentEpoch != LastEpochFired_) {
			EpochLossSum_   = 0.0;
			EpochLossCount_ = 0;
			EpochSourceUnits_ = 0;
			EpochT0_        = std::chrono::high_resolution_clock::now();
			LastEpochFired_ = currentEpoch;
			FireEpochBegin();
			if (StopRequested_) return;
		}
	}
	// Step one is the warm-up: optimizers and autograd may lazily create
	// persistent state. Starting at step two, fixed-shape training allocations
	// are assigned stable context slots so the recorded graph can reuse exact
	// buffer identities on every following step.
	if (Index_ > 1) {
		OaExecutionMemory::BeginStableFrame(OaContext::GetDefault());
		StableResourceFrameOpen_ = true;
	}
	if (TrainingPhaseTiming_) {
		PhaseBodyT0_ = std::chrono::high_resolution_clock::now();
		PhaseBodyStarted_ = true;
	}
}

bool OaItTraining::IsDone() const {
	// Cooperative stop: don't advance into (or fire OnEpochBegin for) a step
	// that will never run.
	if (StopRequested_) return true;
	if (Cfg_.TotalSteps > 0 and not BodyPending_ and Index_ >= Cfg_.TotalSteps) {
		return true;
	}
	const_cast<OaItTraining*>(this)->AdvanceIfNeeded_();
	return StopRequested_;
}

void OaItTraining::Reset() {
	CloseStableResourceFrame_();
	if (Cfg_.Program != nullptr) {
		auto status = Cfg_.Program->Reset();
		if (not status.IsOk()) {
			OA_LOG_WARN(OaLogComponent::ML,
				"OaItTraining::Reset: training program reset failed: %s",
				status.GetMessage().c_str());
		}
	}
	Index_           = 0;
	TotalSamples_    = 0;
	TotalUnits_      = 0;
	TotalSourceUnits_ = 0;
	GpuTimedSourceUnits_ = 0;
	PendingSourceUnits_ = -1;
	LastStepSourceUnits_ = 0;
	EpochSourceUnits_ = 0;
	LastLoss_        = 0.0F;
	LastGpuMs_       = 0.0;
	LastLossStep_    = 0;
	LastGpuTimeStep_ = 0;
	GpuTimingSamples_.clear();
	GpuTimingSumMs_  = 0.0;
	LiveAccuracy_    = std::numeric_limits<OaF32>::quiet_NaN();
	StopRequested_   = false;
	LastStatus_      = OaStatus::Ok();
	TrainBeginFired_ = false;
	BodyPending_     = false;
	LastEpochFired_  = -1;
	EpochLossSum_    = 0.0;
	EpochLossCount_  = 0;
	TrainingLossSum_ = 0.0;
	TrainingLossCount_ = 0;
	T0_              = std::chrono::high_resolution_clock::now();
	EpochT0_         = T0_;
	LastStepT_       = T0_;
	TrainingPhaseStats_ = {};
	PhaseBodyStarted_ = false;
	PendingLoss_     = OaMatrix{};
	ProgramLoss_     = OaMatrix{};
	ProgramCaptureDisabled_ = false;
	ProgramReportWritten_ = false;
	ResetMetrics_();
	if (Session_ != nullptr) Session_->OnReset(*this);
}

bool OaItTraining::IsEpochBoundary() const {
	if (Index_ <= 0) return false;
	if (HasEpochs() and IsLastStep()) return true;
	if (not EpochOffsets_.empty()) {
		return std::binary_search(EpochOffsets_.begin(), EpochOffsets_.end(), Index_);
	}
	if (Cfg_.StepsPerEpoch <= 0) return false;
	return (Index_ % Cfg_.StepsPerEpoch) == 0;
}

bool OaItTraining::IsLastStep() const {
	if (Cfg_.TotalSteps <= 0) return false;
	return Index_ == Cfg_.TotalSteps;
}

// ─── Epoch accessors ──────────────────────────────────────────────────────

OaI64 OaItTraining::EpochIndexForStep_(OaI64 InStep) const {
	// First offset >= InStep is the epoch containing that step.
	auto it = std::lower_bound(EpochOffsets_.begin(), EpochOffsets_.end(), InStep);
	if (it == EpochOffsets_.end()) return static_cast<OaI64>(EpochOffsets_.size()) - 1;
	return static_cast<OaI64>(it - EpochOffsets_.begin());
}

OaI64 OaItTraining::Epoch() const {
	if (Index_ == 0) return 0;
	if (not EpochOffsets_.empty()) {
		return EpochIndexForStep_(Index_) + 1;
	}
	if (Cfg_.StepsPerEpoch <= 0) return 0;
	return ((Index_ - 1) / Cfg_.StepsPerEpoch) + 1;
}

OaI64 OaItTraining::StepInEpoch() const {
	if (not EpochOffsets_.empty()) {
		if (Index_ == 0) return 0;
		const OaI64 epochIdx = EpochIndexForStep_(Index_);
		const OaI64 begin = epochIdx == 0 ? 0 : EpochOffsets_[static_cast<size_t>(epochIdx - 1)];
		return Index_ - begin;
	}
	if (Cfg_.StepsPerEpoch <= 0) return Index_;
	const OaI64 mod = Index_ % Cfg_.StepsPerEpoch;
	return mod == 0 ? Cfg_.StepsPerEpoch : mod;
}

OaI64 OaItTraining::StepsInCurrentEpoch() const {
	if (not Cfg_.EpochSteps.empty()) {
		const OaI64 epochIdx = Index_ == 0 ? 0 : EpochIndexForStep_(Index_);
		return Cfg_.EpochSteps[static_cast<size_t>(epochIdx)];
	}
	if (Cfg_.StepsPerEpoch <= 0) return Cfg_.TotalSteps;
	if (Cfg_.TotalSteps > 0 and Epoch() == TotalEpochs()) {
		const OaI64 completedBefore = (Epoch() - 1) * Cfg_.StepsPerEpoch;
		return std::min(Cfg_.StepsPerEpoch, Cfg_.TotalSteps - completedBefore);
	}
	return Cfg_.StepsPerEpoch;
}

OaI64 OaItTraining::TotalEpochs() const {
	if (not Cfg_.EpochSteps.empty()) return static_cast<OaI64>(Cfg_.EpochSteps.size());
	if (Cfg_.StepsPerEpoch <= 0 or Cfg_.TotalSteps <= 0) return 0;
	return (Cfg_.TotalSteps + Cfg_.StepsPerEpoch - 1) / Cfg_.StepsPerEpoch;
}

// ─── Loss tagging ─────────────────────────────────────────────────────────

void OaItTraining::RecordLoss(const OaMatrix& InLoss) {
	PendingLoss_ = InLoss;
}

// ─── Step / Next ──────────────────────────────────────────────────────────

void OaItTraining::Next() {
	// Body for step Index_ has just run — finalize: record opt step, push the
	// recorded ops through execution, then fire StepEnd/EpochEnd. Index_ stays
	// at the current step number; the next IsDone() advances lazily.
	using Clock = std::chrono::high_resolution_clock;
	const auto elapsedMs = [](const Clock::time_point& InBegin,
		const Clock::time_point& InEnd) -> OaF64 {
		return std::chrono::duration<OaF64, std::milli>(InEnd - InBegin).count();
	};
	const auto stepT0 = TrainingPhaseTiming_ and PhaseBodyStarted_
		? PhaseBodyT0_ : Clock::time_point{};
	auto& ctx = OaContext::GetDefault();
	const auto failStep = [&](const OaStatus& InStatus, const char* InAction) {
		CloseStableResourceFrame_();
		LastStatus_ = InStatus;
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaItTraining: %s failed at step %lld: %s",
			InAction, static_cast<long long>(Index_), InStatus.GetMessage().c_str());
		StopRequested_ = true;
		PhaseBodyStarted_ = false;
		BodyPending_ = false;
	};
	const OaBool replayExisting = Cfg_.Program != nullptr
		and Cfg_.Program->IsCaptured();
	if (replayExisting and ctx.NodeCount() != 0) {
		auto status = OaStatus::Error(OaStatusCode::FailedPrecondition,
			"recorded new graph nodes after capture; use Step(prepare, record) so "
			"only mapped input preparation runs on replay steps");
		ctx.Clear();
		failStep(status, "program replay");
		return;
	}

	const auto optimizerT0 = TrainingPhaseTiming_ ? Clock::now() : Clock::time_point{};
	if (TrainingPhaseTiming_ and PhaseBodyStarted_) {
		TrainingPhaseStats_.BodyMs += elapsedMs(PhaseBodyT0_, optimizerT0);
	}

	if (not replayExisting) Opt_.Step();
	const auto optimizerT1 = TrainingPhaseTiming_ ? Clock::now() : Clock::time_point{};
	if (TrainingPhaseTiming_) {
		TrainingPhaseStats_.OptimizerMs += elapsedMs(optimizerT0, optimizerT1);
	}

	OaBool usedProgram = false;
	if (Cfg_.Program != nullptr and not ProgramCaptureDisabled_
		and (replayExisting or Index_ > 1))
	{
		if (not replayExisting) {
			const auto captureT0 = Clock::now();
			OaVec<const OaMatrix*> observedOutputs;
			if (PendingLoss_.HasStorage()) observedOutputs.PushBack(&PendingLoss_);
			OaTrainingProgramOptions programOptions;
			programOptions.EnableGpuTiming = TimerReady_;
			programOptions.ObservedOutputs = {
				observedOutputs.Data(), observedOutputs.Size()};
			auto captureStatus = Cfg_.Program->Capture(ctx, programOptions);
			const auto captureT1 = Clock::now();
			if (TrainingPhaseTiming_) {
				TrainingPhaseStats_.CompileMs += elapsedMs(captureT0, captureT1);
			}
			if (not captureStatus.IsOk()) {
				// Capture is an optimization. The source graph is deliberately left
				// intact on rejection so this and all later steps remain correct eager
				// execution rather than turning an unsupported op into a hard failure.
				ProgramCaptureDisabled_ = true;
				OA_LOG_WARN(OaLogComponent::ML,
					"OaItTraining: static capture unavailable at step %lld (%s); "
					"continuing eagerly",
					static_cast<long long>(Index_), captureStatus.GetMessage().c_str());
			} else {
				usedProgram = true;
				ProgramLoss_ = PendingLoss_;
				// Drop eager warm-up timing: captured samples use the timestamp bracket
				// embedded around the complete reusable device program.
				GpuTimingSamples_.clear();
				GpuTimingSumMs_ = 0.0;
				GpuTimedSourceUnits_ = 0;
				LastGpuMs_ = 0.0;
				LastGpuTimeStep_ = 0;
			}
		} else {
			usedProgram = true;
		}
	}

	if (usedProgram) {
		const auto submitT0 = Clock::now();
		auto replayStatus = Cfg_.Program->Replay();
		const auto submitT1 = Clock::now();
		if (TrainingPhaseTiming_) {
			TrainingPhaseStats_.SubmitMs += elapsedMs(submitT0, submitT1);
		}
		if (not replayStatus.IsOk()) {
			failStep(replayStatus, "training program submit");
			return;
		}
		const auto waitT0 = Clock::now();
		auto waitStatus = Cfg_.Program->Wait();
		const auto waitT1 = Clock::now();
		if (TrainingPhaseTiming_) {
			TrainingPhaseStats_.WaitMs += elapsedMs(waitT0, waitT1);
		}
		if (not waitStatus.IsOk()) {
			failStep(waitStatus, "training program wait");
			return;
		}
		// Emit evidence only after the first completed replay. The report then
		// contains the actual submission/timeline completion token, rather than a
		// compiled-but-never-submitted placeholder.
		if (not ProgramReportWritten_) {
			const OaString reportSetting = OaEnvFlag::GetString("OA_GRAPH_REPORT");
			if (not reportSetting.Empty()) {
				const OaPath reportPath = reportSetting == "1"
					? OaPaths::Var("report") / "training_graph.json"
					: OaPath(reportSetting.StdStr());
				const auto parent = reportPath.ParentPath();
				auto reportStatus = parent.Empty()
					? OaStatus::Ok() : OaFilesystem::CreateDirectories(parent);
				if (reportStatus.IsOk()) {
					const OaString report = Cfg_.Program->DebugReportJson("TrainingStep");
					reportStatus = OaFilesystem::WriteText(reportPath, report);
				}
				if (reportStatus.IsOk()) {
					OA_LOG_INFO(OaLogComponent::ML,
						"Training graph report: %s", reportPath.String().c_str());
				} else {
					OA_LOG_WARN(OaLogComponent::ML,
						"Training graph report failed: %s",
						reportStatus.GetMessage().c_str());
				}
				ProgramReportWritten_ = true;
			}
		}
		if (replayExisting) Opt_.NotifyProgramReplay();
	} else {
		// A legal step may be host-only (for example OaOptimizerNoOp in callback
		// lifecycle tests). Submit() deliberately rejects an empty recording, so
		// only create a GPU event when this step actually recorded device work.
		if (ctx.NodeCount() != 0U) {
			OaGpuTimer* timer = TimerReady_ ? &Timer_ : nullptr;
			auto submitted = ctx.Submit(timer);
			if (not submitted.IsOk()) {
				failStep(submitted.GetStatus(), "training step submit");
				return;
			}
			auto waitStatus = ctx.Wait(submitted.GetValue());
			if (not waitStatus.IsOk()) {
				failStep(waitStatus, "training step wait");
				return;
			}
		}
		if (TrainingPhaseTiming_) {
			const auto& runtime = ctx.LastExecutionStats();
			TrainingPhaseStats_.CompileMs += runtime.CompileMs;
			TrainingPhaseStats_.RecordMs += runtime.RecordMs;
			TrainingPhaseStats_.SubmitMs += runtime.SubmitMs;
			TrainingPhaseStats_.WaitMs += runtime.WaitMs;
		}
	}

	TotalSamples_ += Cfg_.BatchSize;
	if (Cfg_.SequenceLength > 0) {
		TotalUnits_ += static_cast<OaI64>(Cfg_.BatchSize) * Cfg_.SequenceLength;
	}
	LastStepSourceUnits_ = PendingSourceUnits_ >= 0
		? PendingSourceUnits_
		: static_cast<OaI64>(std::llround(
			static_cast<OaF64>(Cfg_.BatchSize) * Cfg_.SourceUnitsPerSample));
	TotalSourceUnits_ += LastStepSourceUnits_;
	EpochSourceUnits_ += LastStepSourceUnits_;

	// The step is complete, so the scalar is exact and safe to read. Every step
	// contributes exactly once to running and epoch metrics.
	const auto scalarMetricT0 = TrainingPhaseTiming_ ? Clock::now() : Clock::time_point{};
	const OaMatrix& completedLoss = usedProgram and ProgramLoss_.HasStorage()
		? ProgramLoss_ : PendingLoss_;
	if (completedLoss.HasStorage() and completedLoss.NumElements() > 0) {
		LastLoss_      = completedLoss.At(0);
		LastLossStep_  = Index_;
		EpochLossSum_ += static_cast<OaF64>(LastLoss_);
		++EpochLossCount_;
		TrainingLossSum_ += static_cast<OaF64>(LastLoss_);
		++TrainingLossCount_;
	}
	if (TimerReady_ and usedProgram) {
		LastGpuMs_ = Cfg_.Program->LastGpuMs();
		if (LastGpuMs_ > 0.0) {
			LastGpuTimeStep_ = Index_;
			GpuTimingSamples_.push_back(LastGpuMs_);
			GpuTimingSumMs_ += LastGpuMs_;
			GpuTimedSourceUnits_ += LastStepSourceUnits_;
		}
	} else if (TimerReady_) {
		auto* rt = ctx.GetEngine();
		if (rt != nullptr) {
			LastGpuMs_ = Timer_.ReadbackMs(rt->Device);
			if (LastGpuMs_ > 0.0) {
				LastGpuTimeStep_ = Index_;
				GpuTimingSamples_.push_back(LastGpuMs_);
				GpuTimingSumMs_ += LastGpuMs_;
				GpuTimedSourceUnits_ += LastStepSourceUnits_;
			}
		}
	}
	UpdateMetrics_();
	LastStatus_ = OaStatus::Ok();
	const auto scalarMetricT1 = TrainingPhaseTiming_ ? Clock::now() : Clock::time_point{};
	if (TrainingPhaseTiming_) {
		TrainingPhaseStats_.ScalarMetricMs += elapsedMs(scalarMetricT0, scalarMetricT1);
	}
	CloseStableResourceFrame_();

	const auto callbackT0 = TrainingPhaseTiming_ ? Clock::now() : Clock::time_point{};
	FireStepEnd();

	if (LastStatus_.IsOk() and IsEpochBoundary()) {
		FireEpochEnd();
	}
	if (TrainingPhaseTiming_) {
		const auto callbackT1 = Clock::now();
		TrainingPhaseStats_.CallbackMs += elapsedMs(callbackT0, callbackT1);
		if (PhaseBodyStarted_) TrainingPhaseStats_.TotalMs += elapsedMs(stepT0, callbackT1);
		++TrainingPhaseStats_.Count;
	}

	PendingLoss_ = OaMatrix{};
	PendingSourceUnits_ = -1;
	PhaseBodyStarted_ = false;
	BodyPending_ = false;  // ready for next IsDone() to advance Index_
	if (Session_ != nullptr) Session_->OnStepCompleted(*this);
}

void OaItTraining::Next(const OaMatrix& InLoss) {
	RecordLoss(InLoss);
	Next();
}

void OaItTraining::Step(const std::function<void()>& InOpFn) {
	if (IsDone()) return;  // also handles lazy AdvanceIfNeeded_
	const OaBool recordsStep = Cfg_.Program == nullptr
		or not Cfg_.Program->IsCaptured();
	if (recordsStep) InOpFn();
	if (StableResourceFrameOpen_ and recordsStep) {
		// This overload provides no prepare/record boundary. Retain every stable
		// slot rather than guessing which allocations escape the captured step.
		OaExecutionMemory::SealAllStableResourcesExternal(
			OaContext::GetDefault());
	}
	Next();
}

void OaItTraining::Step(
	const std::function<void()>& InPrepareFn,
	const std::function<void()>& InRecordFn)
{
	if (IsDone()) return;  // also opens the stable frame before preparation
	const OaBool recordsStep = Cfg_.Program == nullptr
		or not Cfg_.Program->IsCaptured();
	InPrepareFn();
	if (StableResourceFrameOpen_ and recordsStep) {
		OaExecutionMemory::SealStableInputs(OaContext::GetDefault());
	}
	if (recordsStep) InRecordFn();
	Next();
}

OaStatus OaItTraining::Finish() {
	CloseStableResourceFrame_();
	if (Cfg_.Program != nullptr and Cfg_.Program->IsCaptured()) {
		auto status = Cfg_.Program->Wait();
		if (not status.IsOk()) {
			if (Session_ != nullptr) Session_->OnFinished(status, *this);
			return status;
		}
	}
	auto& ctx = OaContext::GetDefault();
	auto syncStatus = ctx.Sync();
	if (not syncStatus.IsOk()) {
		if (Session_ != nullptr) Session_->OnFinished(syncStatus, *this);
		return syncStatus;
	}
	if (not LastStatus_.IsOk()) {
		if (Session_ != nullptr) Session_->OnFinished(LastStatus_, *this);
		return LastStatus_;
	}

	FireTrainEnd();
	if (not LastStatus_.IsOk()) {
		if (Session_ != nullptr) Session_->OnFinished(LastStatus_, *this);
		return LastStatus_;
	}
	if (TrainingPhaseTiming_ and TrainingPhaseStats_.Count > 0) {
		const auto& s = TrainingPhaseStats_;
		const OaF64 total = s.Mean(s.TotalMs);
		const OaF64 accounted = s.Mean(s.AccountedMs());
		OA_LOG_INFO(OaLogComponent::ML,
			"Training phases: steps=%lld total=%.3f ms/step body=%.3f optimizer=%.3f "
			"compile=%.3f record=%.3f submit=%.3f wait=%.3f scalar_metric=%.3f "
			"callbacks=%.3f unaccounted=%.3f",
			static_cast<long long>(s.Count), total, s.Mean(s.BodyMs),
			s.Mean(s.OptimizerMs), s.Mean(s.CompileMs), s.Mean(s.RecordMs),
			s.Mean(s.SubmitMs), s.Mean(s.WaitMs), s.Mean(s.ScalarMetricMs),
			s.Mean(s.CallbackMs), std::max<OaF64>(total - accounted, 0.0));
	}
	auto status = OaStatus::Ok();
	if (Session_ != nullptr) Session_->OnFinished(status, *this);
	return status;
}

OaStatus OaItTraining::RequestProgramRecapture() {
	if (Cfg_.Program == nullptr) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaItTraining::RequestProgramRecapture requires a configured program");
	}
	OA_RETURN_IF_ERROR(Cfg_.Program->Reset());
	ProgramCaptureDisabled_ = false;
	ProgramLoss_ = OaMatrix{};
	ProgramReportWritten_ = false;
	return OaStatus::Ok();
}

void OaItTraining::CloseStableResourceFrame_() {
	if (not StableResourceFrameOpen_) return;
	OaExecutionMemory::EndStableFrame(OaContext::GetDefault());
	StableResourceFrameOpen_ = false;
}

// ─── Throughput accessors ─────────────────────────────────────────────────

OaF64 OaItTraining::GpuSamplesPerSecond() const {
	if (GpuTimingSamples_.empty()) return 0.0;
	const OaF64 meanMs = GpuTimingSumMs_ / static_cast<OaF64>(GpuTimingSamples_.size());
	return meanMs > 0.0 ? static_cast<OaF64>(Cfg_.BatchSize) / (meanMs / 1000.0) : 0.0;
}

OaGpuTimingStats OaItTraining::GpuTimingStats() const {
	OaGpuTimingStats stats;
	if (GpuTimingSamples_.empty()) return stats;

	std::vector<OaF64> sorted = GpuTimingSamples_;
	std::sort(sorted.begin(), sorted.end());

	const auto percentile = [&sorted](OaF64 p) -> OaF64 {
		if (sorted.empty()) return 0.0;
		const OaF64 pos = p * static_cast<OaF64>(sorted.size() - 1);
		const auto lo = static_cast<size_t>(pos);
		const auto hi = std::min(lo + 1, sorted.size() - 1);
		const OaF64 t = pos - static_cast<OaF64>(lo);
		return sorted[lo] * (1.0 - t) + sorted[hi] * t;
	};

	stats.Count = static_cast<OaI64>(GpuTimingSamples_.size());
	stats.MeanMs = GpuTimingSumMs_ / static_cast<OaF64>(GpuTimingSamples_.size());
	stats.MinMs = sorted.front();
	stats.MedianMs = percentile(0.50);
	stats.P95Ms = percentile(0.95);
	stats.LastMs = GpuTimingSamples_.back();
	return stats;
}

OaF64 OaItTraining::ElapsedSeconds() const {
	auto now = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double>(now - T0_).count();
}

void OaItTraining::ExcludeWallTime(OaF64 InSeconds) {
	if (InSeconds <= 0.0) return;
	const auto excluded = std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(
		std::chrono::duration<OaF64>(InSeconds));
	T0_ += excluded;
	EpochT0_ += excluded;
	LastStepT_ += excluded;
}

OaF64 OaItTraining::WallSamplesPerSecond() const {
	const OaF64 elapsed = ElapsedSeconds();
	if (elapsed <= 0.0 or TotalSamples_ == 0) return 0.0;
	return static_cast<OaF64>(TotalSamples_) / elapsed;
}

OaF64 OaItTraining::GpuUnitsPerSecond() const {
	return Cfg_.SequenceLength > 0
		? GpuSamplesPerSecond() * static_cast<OaF64>(Cfg_.SequenceLength) : 0.0;
}

OaF64 OaItTraining::WallUnitsPerSecond() const {
	const OaF64 elapsed = ElapsedSeconds();
	return elapsed > 0.0 ? static_cast<OaF64>(TotalUnits_) / elapsed : 0.0;
}

OaF64 OaItTraining::GpuSourceUnitsPerSecond() const {
	if (GpuTimingSumMs_ <= 0.0 or GpuTimedSourceUnits_ <= 0) return 0.0;
	return static_cast<OaF64>(GpuTimedSourceUnits_) / (GpuTimingSumMs_ / 1000.0);
}

OaF64 OaItTraining::WallSourceUnitsPerSecond() const {
	const OaF64 elapsed = ElapsedSeconds();
	return elapsed > 0.0 ? static_cast<OaF64>(TotalSourceUnits_) / elapsed : 0.0;
}

OaF64 OaItTraining::EpochSourceUnitsPerSecond() const {
	const OaF64 elapsed = EpochSeconds();
	return elapsed > 0.0 ? static_cast<OaF64>(EpochSourceUnits_) / elapsed : 0.0;
}

OaF64 OaItTraining::WallMsPerStep() const {
	return Index_ > 0 ? ElapsedSeconds() * 1000.0 / static_cast<OaF64>(Index_) : 0.0;
}

OaF64 OaItTraining::EpochSampledMeanLoss() const {
	if (EpochLossCount_ == 0) return 0.0;
	return EpochLossSum_ / static_cast<OaF64>(EpochLossCount_);
}

OaF64 OaItTraining::EpochMeanLoss() const {
	return EpochSampledMeanLoss();
}

OaF64 OaItTraining::TrainingMeanLoss() const {
	return TrainingLossCount_ > 0
		? TrainingLossSum_ / static_cast<OaF64>(TrainingLossCount_) : 0.0;
}

OaF64 OaItTraining::EpochSeconds() const {
	auto now = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double>(now - EpochT0_).count();
}

// ─── Callback dispatch ────────────────────────────────────────────────────

void OaItTraining::AddCallback(OaCbTraining* InCallback) {
	if (InCallback != nullptr) Callbacks_.push_back(InCallback);
}

void OaItTraining::AddMetric(OaMetric* InMetric) {
	if (InMetric != nullptr) Metrics_.push_back(InMetric);
}

void OaItTraining::ResetMetrics_() {
	for (auto* metric : Metrics_) metric->Reset();
	LastStepT_ = std::chrono::high_resolution_clock::now();
}

void OaItTraining::UpdateMetrics_() {
	const auto now = std::chrono::high_resolution_clock::now();
	const OaF64 seconds = std::chrono::duration<double>(now - LastStepT_).count();
	for (auto* metric : Metrics_) {
		metric->UpdateStep(LastLoss_, HasLossSample(), seconds, 1);
	}
	LastStepT_ = now;
}

void OaItTraining::FireTrainBegin() {
	// Training wall time begins at the first iteration, excluding construction,
	// model/header setup, and time spent waiting before the loop starts.
	T0_ = std::chrono::high_resolution_clock::now();
	EpochT0_ = T0_;
	ResetMetrics_();
	for (auto* cb : Callbacks_) {
		cb->OnTrainBegin(*this);
		if (not CaptureCallbackStatus_(*cb, "OnTrainBegin")) break;
	}
}

void OaItTraining::FireEpochBegin() {
	ResetMetrics_();
	for (auto* cb : Callbacks_) {
		cb->OnEpochBegin(*this);
		if (not CaptureCallbackStatus_(*cb, "OnEpochBegin")) break;
	}
}

void OaItTraining::FireStepEnd() {
	for (auto* cb : Callbacks_) {
		cb->OnStepEnd(*this);
		if (not CaptureCallbackStatus_(*cb, "OnStepEnd")) break;
	}
}

void OaItTraining::FireEpochEnd() {
	for (auto* cb : Callbacks_) {
		cb->OnEpochEnd(*this);
		if (not CaptureCallbackStatus_(*cb, "OnEpochEnd")) break;
	}
}

void OaItTraining::FireTrainEnd() {
	for (auto* cb : Callbacks_) {
		cb->OnTrainEnd(*this);
		if (not CaptureCallbackStatus_(*cb, "OnTrainEnd")) break;
	}
}

bool OaItTraining::CaptureCallbackStatus_(
	OaCbTraining& InCallback, const char* InPhase)
{
	const OaStatus status = InCallback.GetStatus();
	if (status.IsOk()) return true;
	if (LastStatus_.IsOk()) LastStatus_ = status;
	StopRequested_ = true;
	OA_LOG_ERROR(OaLogComponent::ML,
		"OaItTraining: callback %s failed at step %lld: %s",
		InPhase, static_cast<long long>(Index_), status.GetMessage().CStr());
	return false;
}
