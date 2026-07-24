#include <Oa/Ml/Rl/Collector.h>

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Autograd.h>

OaResult<OaRlCollector> OaRlCollector::Create(
	OaRlEnvironment& InEnvironment,
	OaRlActorCritic& InModel,
	const OaRlCollectorConfig& InConfig) {
	if (InConfig.Horizon == 0 || InEnvironment.Environments() == 0) {
		return OaStatus::InvalidArgument(
			"OaRlCollector expects a non-empty vector environment and horizon");
	}
	OA_RETURN_IF_ERROR(InEnvironment.Spec().ValidateDefinition());
	if (InEnvironment.Spec().Action.Kind != OaRlSpaceKind::Discrete) {
		return OaStatus::InvalidArgument(
			"OaRlCollector currently requires a discrete action space");
	}
	OaRlCollector result;
	result.Environment_ = &InEnvironment;
	result.Model_ = &InModel;
	result.Config_ = InConfig;
	return result;
}

OaResult<OaEvent> OaRlCollector::Collect(OaRlRolloutBuffer& InOutRollout) {
	if (Environment_ == nullptr || Model_ == nullptr || !InOutRollout.IsValid()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaRlCollector::Collect requires valid environment, model and rollout");
	}
	const OaU32 environments = Environment_->Environments();
	const auto& rolloutConfig = InOutRollout.Config();
	if (rolloutConfig.Time != Config_.Horizon
		|| rolloutConfig.Environments != environments
		|| rolloutConfig.ObservationShape != Environment_->Spec().Observation.Shape) {
		return OaStatus::Error(OaStatusCode::ShapeMismatch,
			"OaRlCollector rollout does not match environment/horizon schema");
	}
	const OaI64 observationElements =
		Environment_->Spec().Observation.ElementsPerEnvironment();
	const OaStatus recorded = Environment_->RecordCommands([&]() -> OaStatus {
		InOutRollout.Reset();
		if (Metrics_.Collections == 0U) {
			OA_RETURN_IF_ERROR(Environment_->ResetEnvironment(Config_.Seed));
		}
		OaGradNo noGrad;
		for (OaU32 step = 0; step < Config_.Horizon; ++step) {
			const OaMatrix observation = Environment_->Observation();
			const OaMatrix flat = OaFnMatrix::Reshape(observation,
				{static_cast<OaI64>(environments), observationElements});
			const OaRlActorCriticOutput network = Model_->Evaluate(flat);
			if (!network.IsValid()) return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				"OaRlCollector actor/critic evaluation failed");
			const OaRlPolicyResult policy = OaFnRl::SampleCategoricalPolicy(
				network.Logits, network.Value, Config_.Seed + ++ActionIndex_);
			if (!policy.IsValid()) return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				"OaRlCollector policy sampling failed");
			auto transition = Environment_->StepEnvironment(policy.Action);
			if (transition.IsError()) return transition.GetStatus();
			const OaMatrix nextFlat = OaFnMatrix::Reshape(
				transition->NextObservation,
				{static_cast<OaI64>(environments), observationElements});
			const OaRlActorCriticOutput next = Model_->Evaluate(nextFlat);
			if (!next.IsValid()) return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				"OaRlCollector next-value evaluation failed");
			OA_RETURN_IF_ERROR(InOutRollout.Append(OaRlTransition{
				.Observation = transition->Observation,
				.Action = policy.Action,
				.Reward = transition->Reward,
				.Value = policy.Value,
				.NextValue = next.Value,
				.LogProbability = policy.LogProbability,
				.Terminated = transition->Terminated,
				.Truncated = transition->Truncated,
			}));
			OA_RETURN_IF_ERROR(Environment_->ResetCompleted());
		}
		return InOutRollout.Finalize(Config_.Gae);
	});
	if (recorded.IsError()) return recorded;
	auto completion = Environment_->Submit();
	if (completion.IsError()) return completion.GetStatus();
	++Metrics_.Collections;
	Metrics_.EnvironmentSteps += Config_.Horizon;
	Metrics_.Transitions += static_cast<OaU64>(Config_.Horizon) * environments;
	return *completion;
}
