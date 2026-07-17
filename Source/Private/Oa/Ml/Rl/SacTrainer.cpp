#include <Oa/Ml/Rl/SacTrainer.h>

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/Optim.h>
#include <Oa/Ml/Rl/Policy.h>
#include <Oa/Ml/TrainingSession.h>
#include <Oa/Runtime/Context.h>

#include <cmath>

namespace {

OaStatus CopyModel(OaModule& InSource, OaModule& InTarget) {
	auto source = InSource.AllNamedParameterPtrs();
	auto target = InTarget.AllNamedParameterPtrs();
	if (source.Size() != target.Size()) return OaStatus::Error(
		OaStatusCode::ShapeMismatch, "SAC critic schemas differ");
	for (OaUsize index = 0; index < source.Size(); ++index) {
		if (source[index].Path != target[index].Path
			|| source[index].Param == nullptr || target[index].Param == nullptr
			|| source[index].Param->Data.GetShape()
				!= target[index].Param->Data.GetShape()
			|| source[index].Param->Data.GetDtype()
				!= target[index].Param->Data.GetDtype()) {
			return OaStatus::Error(OaStatusCode::ShapeMismatch,
				"SAC critic schemas differ");
		}
		target[index].Param->Data = OaFnMatrix::Copy(source[index].Param->Data);
		target[index].Param->Data.SetRequiresGrad(false);
		target[index].Param->RequiresGrad = false;
	}
	return OaStatus::Ok();
}

OaMatrix VectorQ(const OaMatrix& InQ, OaU32 InBatch) {
	if (InQ.GetDtype() != OaScalarType::Float32) return {};
	if (InQ.GetShape() == OaMatrixShape{static_cast<OaI64>(InBatch)}) return InQ;
	if (InQ.GetShape() == OaMatrixShape{static_cast<OaI64>(InBatch), 1}) {
		return OaFnMatrix::Reshape(InQ, {static_cast<OaI64>(InBatch)});
	}
	return {};
}

OaMatrix CriticInput(const OaMatrix& InObservation, const OaMatrix& InAction) {
	OaMatrix parts[] = {InObservation, InAction};
	return OaFnMatrix::Concat(OaSpan<OaMatrix>(parts, 2), 1);
}

OaRlContinuousPolicyResult ActorPolicy(
	OaModule& InActor,
	const OaMatrix& InObservation,
	OaU32 InBatch,
	OaU32 InActionDimensions,
	OaF32 InMinimum,
	OaF32 InMaximum,
	OaU64 InSeed) {
	const OaMatrix output = InActor.Forward(InObservation);
	if (output.GetShape() != OaMatrixShape{
		static_cast<OaI64>(InBatch),
		static_cast<OaI64>(2U * InActionDimensions)}
		|| output.GetDtype() != OaScalarType::Float32) return {};
	OaI64 sizes[] = {
		static_cast<OaI64>(InActionDimensions),
		static_cast<OaI64>(InActionDimensions)};
	auto split = OaFnMatrix::Split(output, OaSpan<OaI64>(sizes, 2), 1);
	if (split.Size() != 2) return {};
	const OaMatrix value = OaFnMatrix::Zeros(
		{static_cast<OaI64>(InBatch)}, OaScalarType::Float32);
	return OaFnRl::SampleTanhNormalPolicy(
		split[0], split[1], value, InMinimum, InMaximum, InSeed);
}

} // namespace

struct OaSacTrainer::Impl {
	OaModule& Actor;
	OaModule& Critic1;
	OaModule& Critic2;
	OaModule& TargetCritic1;
	OaModule& TargetCritic2;
	OaOptimizer& ActorOptimizer;
	OaOptimizer& CriticOptimizer;
	OaRlReplayBuffer& Replay;
	OaSacTrainerConfig Config;
	OaItTraining CriticTraining;
	OaItTraining ActorTraining;
	OaSacTrainerMetrics Metrics;

	Impl(OaModule& InActor, OaModule& InCritic1, OaModule& InCritic2,
		OaModule& InTargetCritic1, OaModule& InTargetCritic2,
		OaOptimizer& InActorOptimizer, OaOptimizer& InCriticOptimizer,
		OaRlReplayBuffer& InReplay, const OaSacTrainerConfig& InConfig)
		: Actor(InActor), Critic1(InCritic1), Critic2(InCritic2)
		, TargetCritic1(InTargetCritic1), TargetCritic2(InTargetCritic2)
		, ActorOptimizer(InActorOptimizer), CriticOptimizer(InCriticOptimizer)
		, Replay(InReplay), Config(InConfig)
		, CriticTraining(InCriticOptimizer, OaItTrainingConfig{
			.TotalSteps = static_cast<OaI64>(InConfig.Updates),
			.BatchSize = static_cast<OaI32>(InConfig.BatchSize),
			.TimerName = "sac_critic_update",
		})
		, ActorTraining(InActorOptimizer, OaItTrainingConfig{
			.TotalSteps = static_cast<OaI64>(InConfig.Updates),
			.BatchSize = static_cast<OaI32>(InConfig.BatchSize),
			.TimerName = "sac_actor_update",
		}) {}
};

OaSacTrainer::OaSacTrainer(OaUniquePtr<Impl> InImpl)
	: Impl_(OaStdMove(InImpl)) {}

OaSacTrainer::~OaSacTrainer() = default;

OaResult<OaUniquePtr<OaSacTrainer>> OaSacTrainer::Create(
	OaModule& InActor, OaModule& InCritic1, OaModule& InCritic2,
	OaModule& InTargetCritic1, OaModule& InTargetCritic2,
	OaOptimizer& InActorOptimizer, OaOptimizer& InCriticOptimizer,
	OaRlReplayBuffer& InReplay, const OaSacTrainerConfig& InConfig) {
	const auto& replay = InReplay.Config();
	if (InConfig.Updates == 0 || InConfig.BatchSize == 0
		|| InConfig.ActionDimensions == 0 || InConfig.TargetUpdateInterval == 0
		|| InConfig.ObservationShape != replay.ObservationShape
		|| replay.ActionDtype != OaScalarType::Float32
		|| replay.ActionShape != OaMatrixShape{
			static_cast<OaI64>(InConfig.ActionDimensions)}
		|| !std::isfinite(InConfig.ActionMinimum)
		|| !std::isfinite(InConfig.ActionMaximum)
		|| InConfig.ActionMinimum >= InConfig.ActionMaximum) {
		return OaStatus::InvalidArgument(
			"OaSacTrainer configuration does not match continuous replay storage");
	}
	auto impl = OaMakeUniquePtr<Impl>(InActor, InCritic1, InCritic2,
		InTargetCritic1, InTargetCritic2, InActorOptimizer,
		InCriticOptimizer, InReplay, InConfig);
	OA_RETURN_IF_ERROR(CopyModel(InCritic1, InTargetCritic1));
	OA_RETURN_IF_ERROR(CopyModel(InCritic2, InTargetCritic2));
	OA_RETURN_IF_ERROR(OaContext::GetDefault().Execute());
	OA_RETURN_IF_ERROR(OaContext::GetDefault().Sync());
	return OaUniquePtr<OaSacTrainer>(new OaSacTrainer(OaStdMove(impl)));
}

OaStatus OaSacTrainer::Update() {
	if (!Impl_) return OaStatus::Error(OaStatusCode::FailedPrecondition,
		"OaSacTrainer is empty");
	auto& impl = *Impl_;
	if (impl.CriticTraining.StopRequested()) {
		OA_RETURN_IF_ERROR(impl.CriticTraining.Finish());
		impl.ActorTraining.RequestStop();
		return impl.ActorTraining.Finish();
	}
	if (IsDone()) return OaStatus::Ok();
	const bool mayBegin = impl.CriticTraining.Session() != nullptr
		? impl.CriticTraining.Session()->TryBeginStep()
		: !impl.CriticTraining.IsDone();
	if (!mayBegin) {
		if (!impl.CriticTraining.StopRequested()) return OaStatus::Ok();
		OA_RETURN_IF_ERROR(impl.CriticTraining.Finish());
		impl.ActorTraining.RequestStop();
		return impl.ActorTraining.Finish();
	}
	if (impl.Replay.Size() < impl.Config.BatchSize) return OaStatus::Error(
		OaStatusCode::FailedPrecondition,
		"OaSacTrainer replay does not contain one complete batch");
	auto sampled = impl.Replay.Sample(impl.Config.BatchSize,
		impl.Config.Seed + impl.Metrics.Update + 1U);
	if (sampled.IsError()) return sampled.GetStatus();
	const OaI64 observationElements = impl.Config.ObservationShape.NumElements();
	const OaMatrix observation = OaFnMatrix::Reshape(sampled->Observation,
		{static_cast<OaI64>(impl.Config.BatchSize), observationElements});
	const OaMatrix nextObservation = OaFnMatrix::Reshape(
		sampled->NextObservation,
		{static_cast<OaI64>(impl.Config.BatchSize), observationElements});

	OaRlContinuousPolicyResult nextPolicy;
	OaMatrix nextQ1;
	OaMatrix nextQ2;
	{
		OaGradNo noGrad;
		nextPolicy = ActorPolicy(impl.Actor, nextObservation,
			impl.Config.BatchSize, impl.Config.ActionDimensions,
			impl.Config.ActionMinimum, impl.Config.ActionMaximum,
			impl.Config.Seed + 0x100000001ULL + impl.Metrics.Update);
		if (!nextPolicy.IsValid()) return OaStatus::Error(
			OaStatusCode::ShapeMismatch,
			"OaSacTrainer actor must return [B,2*action-dim]");
		const OaMatrix input = CriticInput(nextObservation, nextPolicy.Action);
		nextQ1 = VectorQ(impl.TargetCritic1.Forward(input), impl.Config.BatchSize);
		nextQ2 = VectorQ(impl.TargetCritic2.Forward(input), impl.Config.BatchSize);
	}
	if (nextQ1.IsEmpty() || nextQ2.IsEmpty()) return OaStatus::Error(
		OaStatusCode::ShapeMismatch,
		"OaSacTrainer critics must return [B] or [B,1]");

	impl.CriticOptimizer.ZeroGrad();
	OaGradientTape criticTape;
	const OaMatrix storedInput = CriticInput(observation, sampled->Action);
	const OaMatrix q1 = VectorQ(
		impl.Critic1.Forward(storedInput), impl.Config.BatchSize);
	const OaMatrix q2 = VectorQ(
		impl.Critic2.Forward(storedInput), impl.Config.BatchSize);
	const OaSacCriticLossResult criticLoss = OaFnLoss::SacCritic(
		q1, q2, sampled->Reward, nextQ1, nextQ2,
		nextPolicy.LogProbability, sampled->Terminated,
		sampled->Truncated, impl.Config.Loss);
	if (!criticLoss.IsValid()) return OaStatus::Error(
		OaStatusCode::FailedPrecondition, "OaSacTrainer critic loss failed");
	criticTape.Backward(criticLoss.TotalLoss);
	impl.CriticTraining.Next(criticLoss.TotalLoss);
	if (impl.CriticTraining.LastStatus().IsError()) {
		return impl.CriticTraining.LastStatus();
	}
	impl.Metrics.CriticLoss = impl.CriticTraining.LastLoss();

	if (impl.ActorTraining.IsDone()) return OaStatus::Error(
		OaStatusCode::FailedPrecondition,
		"OaSacTrainer actor iterator completed before the critic iterator");
	impl.ActorOptimizer.ZeroGrad();
	impl.CriticOptimizer.ZeroGrad();
	OaGradientTape actorTape;
	const OaRlContinuousPolicyResult policy = ActorPolicy(
		impl.Actor, observation, impl.Config.BatchSize,
		impl.Config.ActionDimensions, impl.Config.ActionMinimum,
		impl.Config.ActionMaximum,
		impl.Config.Seed + 0x200000001ULL + impl.Metrics.Update);
	if (!policy.IsValid()) return OaStatus::Error(
		OaStatusCode::FailedPrecondition, "OaSacTrainer actor policy failed");
	const OaMatrix policyInput = CriticInput(observation, policy.Action);
	const OaMatrix actorQ1 = VectorQ(
		impl.Critic1.Forward(policyInput), impl.Config.BatchSize);
	const OaMatrix actorQ2 = VectorQ(
		impl.Critic2.Forward(policyInput), impl.Config.BatchSize);
	const OaMatrix actorLoss = OaFnLoss::SacActor(
		actorQ1, actorQ2, policy.LogProbability,
		impl.Config.Loss.EntropyCoefficient);
	if (actorLoss.IsEmpty()) return OaStatus::Error(
		OaStatusCode::FailedPrecondition, "OaSacTrainer actor loss failed");
	actorTape.Backward(actorLoss);
	impl.ActorTraining.Next(actorLoss);
	if (impl.ActorTraining.LastStatus().IsError()) {
		return impl.ActorTraining.LastStatus();
	}
	impl.Metrics.ActorLoss = impl.ActorTraining.LastLoss();
	impl.Metrics.Update = static_cast<OaU64>(impl.CriticTraining.StepCount());
	if (impl.Metrics.Update % impl.Config.TargetUpdateInterval == 0) {
		OA_RETURN_IF_ERROR(SyncTargets());
	}
	if (impl.Metrics.Update >= impl.Config.Updates) {
		OA_RETURN_IF_ERROR(impl.CriticTraining.Finish());
		OA_RETURN_IF_ERROR(impl.ActorTraining.Finish());
	}
	return OaStatus::Ok();
}

OaStatus OaSacTrainer::SyncTargets() {
	if (!Impl_) return OaStatus::Error(OaStatusCode::FailedPrecondition,
		"OaSacTrainer is empty");
	OA_RETURN_IF_ERROR(CopyModel(Impl_->Critic1, Impl_->TargetCritic1));
	OA_RETURN_IF_ERROR(CopyModel(Impl_->Critic2, Impl_->TargetCritic2));
	OA_RETURN_IF_ERROR(OaContext::GetDefault().Execute());
	return OaContext::GetDefault().Sync();
}

bool OaSacTrainer::IsDone() const noexcept {
	return Impl_ && (Impl_->CriticTraining.StopRequested()
		|| Impl_->CriticTraining.StepCount() >= Impl_->CriticTraining.TotalSteps());
}

const OaSacTrainerMetrics& OaSacTrainer::Metrics() const noexcept {
	return Impl_->Metrics;
}

OaItTraining& OaSacTrainer::TrainingLoop() noexcept {
	return Impl_->CriticTraining;
}

const OaItTraining& OaSacTrainer::TrainingLoop() const noexcept {
	return Impl_->CriticTraining;
}

OaItTraining& OaSacTrainer::ActorTrainingLoop() noexcept {
	return Impl_->ActorTraining;
}

const OaItTraining& OaSacTrainer::ActorTrainingLoop() const noexcept {
	return Impl_->ActorTraining;
}
