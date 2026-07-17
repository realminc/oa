#pragma once

#include <Oa/Ml/Nn.h>

struct OaRlActorCriticOutput {
	OaMatrix Logits;
	OaMatrix Value;

	[[nodiscard]] bool IsValid() const noexcept {
		return !Logits.IsEmpty() && !Value.IsEmpty();
	}
};

// Environment-neutral discrete actor/critic contract. Custom policies only
// need to implement Evaluate; PPO collection, loss construction and updates do
// not depend on the concrete network architecture.
class OaRlActorCritic : public OaModule {
public:
	OaMatrix Forward(const OaMatrix& InObservation) final {
		return Evaluate(InObservation).Logits;
	}

	[[nodiscard]] virtual OaRlActorCriticOutput Evaluate(const OaMatrix& InObservation) = 0;
};

struct OaCategoricalActorCriticConfig {
	OaI32 ObservationSize = 0;
	OaI32 ActionCount = 0;
	OaI32 HiddenSize = 64;
};

// Compact default MLP used by discrete-control PPO. It is a convenience model,
// not a restriction: callers can supply any OaRlActorCritic implementation.
class OaCategoricalActorCritic final : public OaRlActorCritic {
public:
	[[nodiscard]] static OaResult<OaUniquePtr<OaCategoricalActorCritic>> Create(const OaCategoricalActorCriticConfig& InConfig);

	[[nodiscard]] OaRlActorCriticOutput Evaluate(const OaMatrix& InObservation) override;
	[[nodiscard]] const OaCategoricalActorCriticConfig& Config() const noexcept {
		return Config_;
	}

private:
	explicit OaCategoricalActorCritic(const OaCategoricalActorCriticConfig& InConfig);

	OaCategoricalActorCriticConfig Config_;
	OaSharedPtr<OaLinear> Policy0_;
	OaSharedPtr<OaLinear> Policy1_;
	OaSharedPtr<OaLinear> Policy_;
	OaSharedPtr<OaLinear> Value0_;
	OaSharedPtr<OaLinear> Value1_;
	OaSharedPtr<OaLinear> Value_;
};
