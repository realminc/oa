#include <oa/ml/actorCritic.h>

#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>

oa::Result<oa::UniquePtr<oa::CategoricalActorCritic>>
oa::CategoricalActorCritic::create(
	const oa::CategoricalActorCriticConfig& inConfig) {
	if (inConfig.observationSize <= 0 || inConfig.actionCount <= 1
		|| inConfig.hiddenSize <= 0) {
		return oa::Status::invalidArgument(
			"oa::CategoricalActorCritic expects positive observation/hidden sizes and at least two actions");
	}
	return oa::UniquePtr<oa::CategoricalActorCritic>(
		new oa::CategoricalActorCritic(inConfig));
}

oa::CategoricalActorCritic::CategoricalActorCritic(
	const oa::CategoricalActorCriticConfig& inConfig)
	: config_(inConfig) {
	policy0_ = oa::makeShared<oa::Linear>(config_.observationSize, config_.hiddenSize);
	policy0_->setActivation(oa::Activation::Relu);
	policy1_ = oa::makeShared<oa::Linear>(config_.hiddenSize, config_.hiddenSize);
	policy1_->setActivation(oa::Activation::Relu);
	policy_ = oa::makeShared<oa::Linear>(config_.hiddenSize, config_.actionCount);
	value0_ = oa::makeShared<oa::Linear>(config_.observationSize, config_.hiddenSize);
	value0_->setActivation(oa::Activation::Relu);
	value1_ = oa::makeShared<oa::Linear>(config_.hiddenSize, config_.hiddenSize);
	value1_->setActivation(oa::Activation::Relu);
	value_ = oa::makeShared<oa::Linear>(config_.hiddenSize, 1);
	registerModule("policy.0", policy0_);
	registerModule("policy.1", policy1_);
	registerModule("policy", policy_);
	registerModule("value.0", value0_);
	registerModule("value.1", value1_);
	registerModule("value", value_);
}

oa::ActorCriticOutput oa::CategoricalActorCritic::evaluate(
	const oa::Matrix& inObservation) {
	if (inObservation.isEmpty() || inObservation.getDtype() != oa::ScalarType::Float32
		|| inObservation.rank() != 2
		|| inObservation.size(1) != config_.observationSize) {
		OaLogError(oa::LogComponent::Ml,
			"oa::CategoricalActorCritic expects FP32 observations [batch,%d]",
			config_.observationSize);
		return {};
	}
	const oa::Matrix policyHidden = policy1_->forward(
		policy0_->forward(inObservation));
	const oa::Matrix valueHidden = value1_->forward(
		value0_->forward(inObservation));
	return {
		.logits = policy_->forward(policyHidden),
		.value = oa::FnMatrix::reshape(
			value_->forward(valueHidden), {inObservation.size(0)}),
	};
}
