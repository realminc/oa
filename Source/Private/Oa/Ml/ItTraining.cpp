// OaItTraining — implementation

#include <Oa/Ml/ItTraining.h>
#include <Oa/Ml/Module.h>
#include <Oa/Ml/Metric.h>
#include <Oa/Ml/Optim.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>

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
	auto* rt = ctx.GetRuntime();
	Rt_ = rt;

	if (rt != nullptr) {
		auto status = Timer_.Init(*rt, Cfg_.TimerName);
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
	Metrics_ = Cfg_.Metrics;

	// Register config-time callbacks. AddCallback can still be used after
	// construction; these are simply equivalent to calling it in a loop.
	for (auto* cb : Cfg_.Callbacks) {
		if (cb != nullptr) Callbacks_.push_back(cb);
	}
}

OaItTraining::~OaItTraining() {
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
		}
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
	return false;
}

void OaItTraining::Reset() {
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
	PendingLoss_     = OaMatrix{};
	ResetMetrics_();
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
	Opt_.Step();

	auto& ctx     = OaContext::GetDefault();
	OaGpuTimer* timer = TimerReady_ ? &Timer_ : nullptr;
	auto execStatus = ctx.ExecuteInAsyncBatch(timer);
	if (not execStatus.IsOk()) {
		LastStatus_ = execStatus;
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaItTraining: ExecuteInAsyncBatch failed at step %lld: %s",
			static_cast<long long>(Index_), execStatus.GetMessage().c_str());
		StopRequested_ = true;
		BodyPending_ = false;
		return;
	}

	auto flushStatus = ctx.FlushAsyncBatch();
	if (not flushStatus.IsOk()) {
		LastStatus_ = flushStatus;
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaItTraining: FlushAsyncBatch failed at step %lld: %s",
			static_cast<long long>(Index_), flushStatus.GetMessage().c_str());
		StopRequested_ = true;
		BodyPending_ = false;
		return;
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

	auto syncStatus = ctx.Sync();
	if (not syncStatus.IsOk()) {
		LastStatus_ = syncStatus;
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaItTraining: Sync failed at step %lld: %s",
			static_cast<long long>(Index_), syncStatus.GetMessage().c_str());
		StopRequested_ = true;
		BodyPending_ = false;
		return;
	}

	// The step is complete, so the scalar is exact and safe to read. Every step
	// contributes exactly once to running and epoch metrics.
	if (PendingLoss_.HasStorage() and PendingLoss_.NumElements() > 0) {
		LastLoss_      = PendingLoss_.At(0);
		LastLossStep_  = Index_;
		EpochLossSum_ += static_cast<OaF64>(LastLoss_);
		++EpochLossCount_;
		TrainingLossSum_ += static_cast<OaF64>(LastLoss_);
		++TrainingLossCount_;
	}
	if (TimerReady_) {
		auto* rt = ctx.GetRuntime();
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

	FireStepEnd();

	if (IsEpochBoundary()) {
		FireEpochEnd();
	}

	PendingLoss_ = OaMatrix{};
	PendingSourceUnits_ = -1;
	BodyPending_ = false;  // ready for next IsDone() to advance Index_
}

void OaItTraining::Next(const OaMatrix& InLoss) {
	RecordLoss(InLoss);
	Next();
}

void OaItTraining::Step(const std::function<void()>& InOpFn) {
	if (IsDone()) return;  // also handles lazy AdvanceIfNeeded_
	InOpFn();
	Next();
}

OaStatus OaItTraining::Finish() {
	auto& ctx = OaContext::GetDefault();
	auto flushStatus = ctx.FlushAsyncBatch();
	if (not flushStatus.IsOk()) return flushStatus;
	auto syncStatus = ctx.Sync();
	if (not syncStatus.IsOk()) return syncStatus;

	FireTrainEnd();
	return OaStatus::Ok();
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
	for (auto* cb : Callbacks_) cb->OnTrainBegin(*this);
}

void OaItTraining::FireEpochBegin() {
	ResetMetrics_();
	for (auto* cb : Callbacks_) cb->OnEpochBegin(*this);
}

void OaItTraining::FireStepEnd() {
	for (auto* cb : Callbacks_) cb->OnStepEnd(*this);
}

void OaItTraining::FireEpochEnd() {
	for (auto* cb : Callbacks_) cb->OnEpochEnd(*this);
}

void OaItTraining::FireTrainEnd() {
	for (auto* cb : Callbacks_) cb->OnTrainEnd(*this);
}
