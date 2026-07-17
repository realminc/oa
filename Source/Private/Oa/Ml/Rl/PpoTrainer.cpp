#include <Oa/Ml/Rl/PpoTrainer.h>

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/Optim.h>
#include <Oa/Ml/TrainingSession.h>
#include <Oa/Runtime/Context.h>

#include <cmath>

namespace {

OaI64 ObservationElements(const OaMatrixShape& InShape) {
	return InShape.NumElements();
}

bool ValidConfig(const OaPpoTrainerConfig& InConfig) {
	return InConfig.Rollouts > 0 && InConfig.Horizon > 0
		&& InConfig.Environments > 0 && InConfig.UpdateEpochs > 0
		&& InConfig.ObservationShape.Rank > 0
		&& ObservationElements(InConfig.ObservationShape) > 0;
}

} // namespace

struct OaPpoTrainer::Impl {
	OaRlActorCritic& Model;
	OaOptimizer& Optimizer;
	OaPpoTrainerConfig Config;
	OaRlRolloutBuffer Rollout;
	OaItRlTraining Training;
	OaPpoTrainerMetrics Metrics;
	OaU64 ActionIndex = 0;
	bool Collecting = false;

	Impl(OaRlActorCritic& InModel, OaOptimizer& InOptimizer,
		const OaPpoTrainerConfig& InConfig, OaRlRolloutBuffer&& InRollout)
		: Model(InModel)
		, Optimizer(InOptimizer)
		, Config(InConfig)
		, Rollout(OaStdMove(InRollout))
		, Training(InOptimizer, OaItRlTrainingConfig{
			.Rollouts = InConfig.Rollouts,
			.Horizon = InConfig.Horizon,
			.Environments = InConfig.Environments,
			.UpdateEpochs = InConfig.UpdateEpochs,
			.TimerName = "ppo_update",
		}) {}
};

OaPpoTrainer::OaPpoTrainer(OaUniquePtr<Impl> InImpl)
	: Impl_(OaStdMove(InImpl)) {}

OaPpoTrainer::~OaPpoTrainer() = default;

OaResult<OaUniquePtr<OaPpoTrainer>> OaPpoTrainer::Create(
	OaRlActorCritic& InModel,
	OaOptimizer& InOptimizer,
	const OaPpoTrainerConfig& InConfig) {
	if (!ValidConfig(InConfig)) {
		return OaStatus::InvalidArgument(
			"OaPpoTrainer expects non-zero rollout dimensions and a non-empty observation shape");
	}
	auto rollout = OaRlRolloutBuffer::Create(OaRlRolloutConfig{
		.Time = InConfig.Horizon,
		.Environments = InConfig.Environments,
		.ObservationShape = InConfig.ObservationShape,
	});
	if (rollout.IsError()) return rollout.GetStatus();
	auto impl = OaMakeUniquePtr<Impl>(
		InModel, InOptimizer, InConfig, OaStdMove(*rollout));
	if (!impl->Training.IsValid()) return impl->Training.LastStatus();
	OA_RETURN_IF_ERROR(OaContext::GetDefault().Execute());
	OA_RETURN_IF_ERROR(OaContext::GetDefault().Sync());
	return OaUniquePtr<OaPpoTrainer>(new OaPpoTrainer(OaStdMove(impl)));
}

OaStatus OaPpoTrainer::BeginCollection() {
	if (!Impl_ || Impl_->Collecting
		|| Impl_->Training.Phase() != OaRlTrainingPhase::Collect) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaPpoTrainer cannot begin collection in the current phase");
	}
	OA_RETURN_IF_ERROR(Impl_->Training.BeginRollout(Impl_->Rollout));
	Impl_->Collecting = true;
	return OaStatus::Ok();
}

OaRlPolicyResult OaPpoTrainer::Act(const OaMatrix& InObservation) {
	if (!Impl_ || !Impl_->Collecting) return {};
	OaGradNo noGrad;
	const OaI64 observationElements = ObservationElements(
		Impl_->Config.ObservationShape);
	const OaMatrix flat = OaFnMatrix::Reshape(InObservation,
		{static_cast<OaI64>(Impl_->Config.Environments), observationElements});
	const OaRlActorCriticOutput network = Impl_->Model.Evaluate(flat);
	if (!network.IsValid()) return {};
	return OaFnRl::SampleCategoricalPolicy(
		network.Logits, network.Value,
		Impl_->Config.Seed + ++Impl_->ActionIndex);
}

OaStatus OaPpoTrainer::Observe(
	const OaMatrix& InObservation,
	const OaMatrix& InNextObservation,
	const OaMatrix& InReward,
	const OaMatrix& InTerminated,
	const OaMatrix& InTruncated,
	const OaRlPolicyResult& InPolicy) {
	if (!Impl_ || !Impl_->Collecting || !InPolicy.IsValid()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaPpoTrainer::Observe requires active collection and a valid policy result");
	}
	OaGradNo noGrad;
	const OaI64 observationElements = ObservationElements(
		Impl_->Config.ObservationShape);
	const OaMatrix nextFlat = OaFnMatrix::Reshape(InNextObservation,
		{static_cast<OaI64>(Impl_->Config.Environments), observationElements});
	const OaRlActorCriticOutput next = Impl_->Model.Evaluate(nextFlat);
	if (!next.IsValid()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaPpoTrainer next-value evaluation failed");
	}
	return Impl_->Rollout.Append(OaRlTransition{
		.Observation = InObservation,
		.Action = InPolicy.Action,
		.Reward = InReward,
		.Value = InPolicy.Value,
		.NextValue = next.Value,
		.LogProbability = InPolicy.LogProbability,
		.Terminated = InTerminated,
		.Truncated = InTruncated,
	});
}

OaStatus OaPpoTrainer::EndCollection() {
	if (!Impl_ || !Impl_->Collecting) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaPpoTrainer collection is not active");
	}
	OA_RETURN_IF_ERROR(Impl_->Training.FinalizeRollout(
		Impl_->Rollout, Impl_->Config.Gae));
	Impl_->Collecting = false;
	return OaStatus::Ok();
}

OaStatus OaPpoTrainer::Update() {
	if (!Impl_ || Impl_->Training.Phase() != OaRlTrainingPhase::Update) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaPpoTrainer cannot update in the current phase");
	}
	if (!Impl_->Training.BeginUpdate()) {
		auto* session = Impl_->Training.UpdateLoop().Session();
		if (Impl_->Training.UpdateLoop().StopRequested()) {
			return Impl_->Training.Finish();
		}
		if (session != nullptr && session->State() == OaTrainingState::Paused) {
			return OaStatus::Ok();
		}
		return Impl_->Training.LastStatus().IsError()
			? Impl_->Training.LastStatus()
			: OaStatus::Error(OaStatusCode::FailedPrecondition,
				"OaPpoTrainer update did not begin");
	}
	auto& impl = *Impl_;
	const OaI64 batch = static_cast<OaI64>(impl.Config.Environments)
		* impl.Config.Horizon;
	const OaI64 observationElements = ObservationElements(
		impl.Config.ObservationShape);
	const OaMatrix observation = OaFnMatrix::Reshape(
		impl.Rollout.Batch().Observation, {batch, observationElements});
	const OaMatrix action = OaFnMatrix::Reshape(
		impl.Rollout.Batch().Action, {batch});
	const OaMatrix oldLogProbability = OaFnMatrix::Reshape(
		impl.Rollout.Batch().OldLogProbability, {batch});
	const OaMatrix advantage = OaFnMatrix::Reshape(
		impl.Rollout.Batch().Advantage, {batch});
	const OaMatrix targetReturn = OaFnMatrix::Reshape(
		impl.Rollout.Batch().Return, {batch});

	impl.Optimizer.ZeroGrad();
	OaGradientTape tape;
	const OaRlActorCriticOutput network = impl.Model.Evaluate(observation);
	const OaRlPolicyResult policy = OaFnRl::EvaluateCategoricalPolicy(
		network.Logits, action, network.Value);
	if (!policy.IsValid()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaPpoTrainer action re-evaluation failed");
	}
	const OaMatrix normalizedAdvantage = OaFnRl::NormalizeAdvantages(advantage);
	const OaPpoLossResult loss = OaFnLoss::Ppo(
		policy.LogProbability, oldLogProbability, normalizedAdvantage,
		policy.Value, targetReturn, policy.Entropy, impl.Config.Loss);
	if (!loss.IsValid()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaPpoTrainer loss construction failed");
	}
	tape.Backward(loss.TotalLoss);
	OA_RETURN_IF_ERROR(impl.Training.NextUpdate(loss.TotalLoss));
	impl.Metrics = {
		.Rollout = impl.Training.RolloutIndex(),
		.UpdateEpoch = impl.Training.UpdateEpoch(),
		.TotalLoss = impl.Training.UpdateLoop().LastLoss(),
		.PolicyLoss = loss.PolicyLoss.Item(),
		.ValueLoss = loss.ValueLoss.Item(),
		.Entropy = loss.Entropy.Item(),
	};
	if (impl.Training.IsDone()) {
		OA_RETURN_IF_ERROR(impl.Training.Finish());
	}
	return OaStatus::Ok();
}

bool OaPpoTrainer::IsValid() const noexcept {
	return Impl_ && Impl_->Training.IsValid();
}

bool OaPpoTrainer::IsDone() const noexcept {
	return Impl_ && Impl_->Training.IsDone();
}

bool OaPpoTrainer::NeedsCollection() const noexcept {
	return Impl_ && !Impl_->Collecting
		&& Impl_->Training.Phase() == OaRlTrainingPhase::Collect;
}

OaRlTrainingPhase OaPpoTrainer::Phase() const noexcept {
	return Impl_ ? Impl_->Training.Phase() : OaRlTrainingPhase::Complete;
}

const OaPpoTrainerConfig& OaPpoTrainer::Config() const noexcept {
	return Impl_->Config;
}

const OaPpoTrainerMetrics& OaPpoTrainer::Metrics() const noexcept {
	return Impl_->Metrics;
}

const OaRlRolloutBatch& OaPpoTrainer::Batch() const noexcept {
	return Impl_->Rollout.Batch();
}

OaItTraining& OaPpoTrainer::TrainingLoop() noexcept {
	return Impl_->Training.UpdateLoop();
}

const OaItTraining& OaPpoTrainer::TrainingLoop() const noexcept {
	return Impl_->Training.UpdateLoop();
}

OaStatus OaPpoTrainer::Save(const OaString& InPath) const {
	if (!Impl_) return OaStatus::Error(OaStatusCode::FailedPrecondition,
		"OaPpoTrainer is empty");
	return Impl_->Model.Save(InPath, Impl_->Optimizer);
}

OaStatus OaPpoTrainer::Load(const OaString& InPath) {
	if (!Impl_) return OaStatus::Error(OaStatusCode::FailedPrecondition,
		"OaPpoTrainer is empty");
	return Impl_->Model.Load(InPath, Impl_->Optimizer);
}
