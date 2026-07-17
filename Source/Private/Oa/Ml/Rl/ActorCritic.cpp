#include <Oa/Ml/Rl/ActorCritic.h>

#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>

OaResult<OaUniquePtr<OaCategoricalActorCritic>>
OaCategoricalActorCritic::Create(
	const OaCategoricalActorCriticConfig& InConfig) {
	if (InConfig.ObservationSize <= 0 || InConfig.ActionCount <= 1
		|| InConfig.HiddenSize <= 0) {
		return OaStatus::InvalidArgument(
			"OaCategoricalActorCritic expects positive observation/hidden sizes and at least two actions");
	}
	return OaUniquePtr<OaCategoricalActorCritic>(
		new OaCategoricalActorCritic(InConfig));
}

OaCategoricalActorCritic::OaCategoricalActorCritic(
	const OaCategoricalActorCriticConfig& InConfig)
	: Config_(InConfig) {
	Policy0_ = OaMakeSharedPtr<OaLinear>(Config_.ObservationSize, Config_.HiddenSize);
	Policy0_->SetActivation(OaActivation::Relu);
	Policy1_ = OaMakeSharedPtr<OaLinear>(Config_.HiddenSize, Config_.HiddenSize);
	Policy1_->SetActivation(OaActivation::Relu);
	Policy_ = OaMakeSharedPtr<OaLinear>(Config_.HiddenSize, Config_.ActionCount);
	Value0_ = OaMakeSharedPtr<OaLinear>(Config_.ObservationSize, Config_.HiddenSize);
	Value0_->SetActivation(OaActivation::Relu);
	Value1_ = OaMakeSharedPtr<OaLinear>(Config_.HiddenSize, Config_.HiddenSize);
	Value1_->SetActivation(OaActivation::Relu);
	Value_ = OaMakeSharedPtr<OaLinear>(Config_.HiddenSize, 1);
	RegisterModule("policy.0", Policy0_);
	RegisterModule("policy.1", Policy1_);
	RegisterModule("policy", Policy_);
	RegisterModule("value.0", Value0_);
	RegisterModule("value.1", Value1_);
	RegisterModule("value", Value_);
}

OaRlActorCriticOutput OaCategoricalActorCritic::Evaluate(
	const OaMatrix& InObservation) {
	if (InObservation.IsEmpty() || InObservation.GetDtype() != OaScalarType::Float32
		|| InObservation.Rank() != 2
		|| InObservation.Size(1) != Config_.ObservationSize) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaCategoricalActorCritic expects FP32 observations [batch,%d]",
			Config_.ObservationSize);
		return {};
	}
	const OaMatrix policyHidden = Policy1_->Forward(
		Policy0_->Forward(InObservation));
	const OaMatrix valueHidden = Value1_->Forward(
		Value0_->Forward(InObservation));
	return {
		.Logits = Policy_->Forward(policyHidden),
		.Value = OaFnMatrix::Reshape(
			Value_->Forward(valueHidden), {InObservation.Size(0)}),
	};
}
