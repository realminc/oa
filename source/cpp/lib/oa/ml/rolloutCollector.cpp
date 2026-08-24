#include <oa/ml/rolloutCollector.h>

#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>

oa::Result<oa::RolloutCollector> oa::RolloutCollector::create(
	oa::Environment& inEnvironment,
	oa::ActorCritic& inModel,
	const oa::RolloutCollectorConfig& inConfig) {
	if (inConfig.horizon == 0 || inEnvironment.environments() == 0) {
		return oa::Status::invalidArgument(
			"oa::RolloutCollector expects a non-empty vector environment and horizon");
	}
	OA_RETURN_IF_ERROR(inEnvironment.spec().validateDefinition());
	if (inEnvironment.spec().action.kind != oa::EnvironmentSpaceKind::Discrete) {
		return oa::Status::invalidArgument(
			"oa::RolloutCollector currently requires a discrete action space");
	}
	oa::RolloutCollector result;
	result.environment_ = &inEnvironment;
	result.model_ = &inModel;
	result.config_ = inConfig;
	return result;
}

oa::Result<oa::Event> oa::RolloutCollector::collect(oa::RolloutBuffer& inOutRollout) {
	if (environment_ == nullptr || model_ == nullptr || !inOutRollout.isValid()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::RolloutCollector::collect requires valid environment, model and rollout");
	}
	const oa::U32 environments = environment_->environments();
	const auto& rolloutConfig = inOutRollout.config();
	if (rolloutConfig.time != config_.horizon
		|| rolloutConfig.environments != environments
		|| rolloutConfig.observationShape != environment_->spec().observation.shape) {
		return oa::Status::error(oa::StatusCode::ShapeMismatch,
			"oa::RolloutCollector rollout does not match environment/horizon schema");
	}
	const oa::I64 observationElements =
		environment_->spec().observation.elementsPerEnvironment();
	const oa::Status recorded = environment_->recordCommands([&]() -> oa::Status {
		inOutRollout.reset();
		if (metrics_.collections == 0U) {
			OA_RETURN_IF_ERROR(environment_->reset(config_.seed));
		}
		oa::GradNo noGrad;
		for (oa::U32 step = 0; step < config_.horizon; ++step) {
			const oa::Matrix observation = environment_->observation();
			const oa::Matrix flat = oa::FnMatrix::reshape(observation,
				{static_cast<oa::I64>(environments), observationElements});
			const oa::ActorCriticOutput network = model_->evaluate(flat);
			if (!network.isValid()) return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::RolloutCollector actor/critic evaluation failed");
			const oa::PolicyResult policy = oa::FnPolicy::sampleCategorical(
				network.logits, network.value, config_.seed + ++actionIndex_);
			if (!policy.isValid()) return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::RolloutCollector policy sampling failed");
			auto transition = environment_->step(policy.action);
			if (transition.isError()) return transition.getStatus();
			const oa::Matrix nextFlat = oa::FnMatrix::reshape(
				transition->nextObservation,
				{static_cast<oa::I64>(environments), observationElements});
			const oa::ActorCriticOutput next = model_->evaluate(nextFlat);
			if (!next.isValid()) return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::RolloutCollector next-value evaluation failed");
			OA_RETURN_IF_ERROR(inOutRollout.append(oa::RolloutTransition{
				.observation = transition->observation,
				.action = policy.action,
				.reward = transition->reward,
				.value = policy.value,
				.nextValue = next.value,
				.logProbability = policy.logProbability,
				.terminated = transition->terminated,
				.truncated = transition->truncated,
			}));
			OA_RETURN_IF_ERROR(environment_->resetCompleted());
		}
		return inOutRollout.finalize(config_.gae);
	});
	if (recorded.isError()) return recorded;
	auto completion = environment_->submit();
	if (completion.isError()) return completion.getStatus();
	++metrics_.collections;
	metrics_.environmentSteps += config_.horizon;
	metrics_.transitions += static_cast<oa::U64>(config_.horizon) * environments;
	return *completion;
}
