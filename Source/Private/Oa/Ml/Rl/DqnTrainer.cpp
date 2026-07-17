#include <Oa/Ml/Rl/DqnTrainer.h>

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/Optim.h>
#include <Oa/Ml/TrainingSession.h>
#include <Oa/Runtime/Context.h>

namespace {

OaStatus CopyModel(OaModule& InSource, OaModule& InTarget) {
	auto source = InSource.AllNamedParameterPtrs();
	auto target = InTarget.AllNamedParameterPtrs();
	if (source.Size() != target.Size()) {
		return OaStatus::Error(OaStatusCode::ShapeMismatch,
			"DQN online and target modules have different parameter counts");
	}
	for (OaUsize index = 0; index < source.Size(); ++index) {
		if (source[index].Path != target[index].Path
			|| source[index].Param == nullptr || target[index].Param == nullptr
			|| source[index].Param->Data.GetShape()
				!= target[index].Param->Data.GetShape()
			|| source[index].Param->Data.GetDtype()
				!= target[index].Param->Data.GetDtype()) {
			return OaStatus::Error(OaStatusCode::ShapeMismatch,
				"DQN online and target module schemas do not match");
		}
		target[index].Param->Data = OaFnMatrix::Copy(source[index].Param->Data);
		target[index].Param->Data.SetRequiresGrad(false);
		target[index].Param->RequiresGrad = false;
	}
	return OaStatus::Ok();
}

} // namespace

struct OaDqnTrainer::Impl {
	OaModule& Online;
	OaModule& Target;
	OaOptimizer& Optimizer;
	OaRlReplayBuffer& Replay;
	OaDqnTrainerConfig Config;
	OaItTraining Training;
	OaDqnTrainerMetrics Metrics;

	Impl(OaModule& InOnline, OaModule& InTarget, OaOptimizer& InOptimizer,
		OaRlReplayBuffer& InReplay, const OaDqnTrainerConfig& InConfig)
		: Online(InOnline), Target(InTarget), Optimizer(InOptimizer), Replay(InReplay)
		, Config(InConfig)
		, Training(InOptimizer, OaItTrainingConfig{
			.TotalSteps = static_cast<OaI64>(InConfig.Updates),
			.BatchSize = static_cast<OaI32>(InConfig.BatchSize),
			.TimerName = "dqn_update",
		}) {}
};

OaDqnTrainer::OaDqnTrainer(OaUniquePtr<Impl> InImpl)
	: Impl_(OaStdMove(InImpl)) {}

OaDqnTrainer::~OaDqnTrainer() = default;

OaResult<OaUniquePtr<OaDqnTrainer>> OaDqnTrainer::Create(
	OaModule& InOnline,
	OaModule& InTarget,
	OaOptimizer& InOptimizer,
	OaRlReplayBuffer& InReplay,
	const OaDqnTrainerConfig& InConfig) {
	const auto& replay = InReplay.Config();
	if (InConfig.Updates == 0 || InConfig.BatchSize == 0
		|| InConfig.TargetUpdateInterval == 0
		|| InConfig.ObservationShape != replay.ObservationShape
		|| replay.ActionShape.Rank != 0
		|| replay.ActionDtype != OaScalarType::Int32) {
		return OaStatus::InvalidArgument(
			"OaDqnTrainer expects positive update settings and scalar Int32 replay actions");
	}
	auto impl = OaMakeUniquePtr<Impl>(
		InOnline, InTarget, InOptimizer, InReplay, InConfig);
	OA_RETURN_IF_ERROR(CopyModel(InOnline, InTarget));
	OA_RETURN_IF_ERROR(OaContext::GetDefault().Execute());
	OA_RETURN_IF_ERROR(OaContext::GetDefault().Sync());
	return OaUniquePtr<OaDqnTrainer>(new OaDqnTrainer(OaStdMove(impl)));
}

OaStatus OaDqnTrainer::Update() {
	if (!Impl_) return OaStatus::Error(OaStatusCode::FailedPrecondition,
		"OaDqnTrainer is empty");
	auto& impl = *Impl_;
	if (impl.Training.StopRequested()) return impl.Training.Finish();
	if (impl.Training.StepCount() >= impl.Training.TotalSteps()) return OaStatus::Ok();
	const bool mayBegin = impl.Training.Session() != nullptr
		? impl.Training.Session()->TryBeginStep()
		: !impl.Training.IsDone();
	if (!mayBegin) {
		return impl.Training.StopRequested()
			? impl.Training.Finish() : OaStatus::Ok();
	}
	if (impl.Replay.Size() < impl.Config.BatchSize) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaDqnTrainer replay does not contain one complete batch");
	}
	auto sampled = impl.Replay.Sample(impl.Config.BatchSize,
		impl.Config.Seed + static_cast<OaU64>(impl.Training.Index()));
	if (sampled.IsError()) return sampled.GetStatus();
	const OaI64 observationElements = impl.Config.ObservationShape.NumElements();
	const OaMatrix observation = OaFnMatrix::Reshape(sampled->Observation,
		{static_cast<OaI64>(impl.Config.BatchSize), observationElements});
	const OaMatrix nextObservation = OaFnMatrix::Reshape(
		sampled->NextObservation,
		{static_cast<OaI64>(impl.Config.BatchSize), observationElements});
	OaMatrix nextQ;
	{
		OaGradNo noGrad;
		nextQ = impl.Target.Forward(nextObservation);
	}
	impl.Optimizer.ZeroGrad();
	OaGradientTape tape;
	const OaMatrix q = impl.Online.Forward(observation);
	const OaDqnLossResult loss = OaFnLoss::Dqn(
		q, sampled->Action, sampled->Reward, nextQ,
		sampled->Terminated, sampled->Truncated, impl.Config.Loss);
	if (!loss.IsValid()) return OaStatus::Error(OaStatusCode::FailedPrecondition,
		"OaDqnTrainer loss construction failed");
	tape.Backward(loss.Loss);
	impl.Training.Next(loss.Loss);
	if (impl.Training.LastStatus().IsError()) return impl.Training.LastStatus();
	impl.Metrics = {
		.Update = static_cast<OaU64>(impl.Training.Index()),
		.Loss = impl.Training.LastLoss(),
	};
	if (impl.Metrics.Update % impl.Config.TargetUpdateInterval == 0) {
		OA_RETURN_IF_ERROR(SyncTarget());
	}
	if (impl.Training.StepCount() >= impl.Training.TotalSteps()) {
		return impl.Training.Finish();
	}
	return OaStatus::Ok();
}

OaStatus OaDqnTrainer::SyncTarget() {
	if (!Impl_) return OaStatus::Error(OaStatusCode::FailedPrecondition,
		"OaDqnTrainer is empty");
	OA_RETURN_IF_ERROR(CopyModel(Impl_->Online, Impl_->Target));
	OA_RETURN_IF_ERROR(OaContext::GetDefault().Execute());
	return OaContext::GetDefault().Sync();
}

bool OaDqnTrainer::IsDone() const noexcept {
	return Impl_ && (Impl_->Training.StopRequested()
		|| Impl_->Training.StepCount() >= Impl_->Training.TotalSteps());
}

const OaDqnTrainerMetrics& OaDqnTrainer::Metrics() const noexcept {
	return Impl_->Metrics;
}

OaItTraining& OaDqnTrainer::TrainingLoop() noexcept {
	return Impl_->Training;
}

const OaItTraining& OaDqnTrainer::TrainingLoop() const noexcept {
	return Impl_->Training;
}

OaStatus OaDqnTrainer::Save(const OaString& InPath) const {
	if (!Impl_) return OaStatus::Error(OaStatusCode::FailedPrecondition,
		"OaDqnTrainer is empty");
	return Impl_->Online.Save(InPath, Impl_->Optimizer);
}

OaStatus OaDqnTrainer::Load(const OaString& InPath) {
	if (!Impl_) return OaStatus::Error(OaStatusCode::FailedPrecondition,
		"OaDqnTrainer is empty");
	OA_RETURN_IF_ERROR(Impl_->Online.Load(InPath, Impl_->Optimizer));
	return SyncTarget();
}
