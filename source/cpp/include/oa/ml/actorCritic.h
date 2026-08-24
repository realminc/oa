#pragma once

#include <oa/ml/nn.h>

namespace oa {

struct ActorCriticOutput {
	oa::Matrix logits;
	oa::Matrix value;

	[[nodiscard]] bool isValid() const noexcept {
		return !logits.isEmpty() && !value.isEmpty();
	}
};

// environment-neutral discrete actor/critic contract. Custom policies only
// need to implement evaluate; PPO collection, loss construction and updates do
// not depend on the concrete network architecture.
class ActorCritic : public oa::Module {
public:
	oa::Matrix forward(const oa::Matrix& inObservation) final {
		return evaluate(inObservation).logits;
	}

	[[nodiscard]] virtual ActorCriticOutput evaluate(const oa::Matrix& inObservation) = 0;
};

struct CategoricalActorCriticConfig {
	oa::I32 observationSize = 0;
	oa::I32 actionCount = 0;
	oa::I32 hiddenSize = 64;
};

// Compact default MLP used by discrete-control PPO. It is a convenience model,
// not a restriction: callers can supply any ActorCritic implementation.
class CategoricalActorCritic final : public ActorCritic {
public:
	[[nodiscard]] static oa::Result<oa::UniquePtr<CategoricalActorCritic>> create(const CategoricalActorCriticConfig& inConfig);

	[[nodiscard]] ActorCriticOutput evaluate(const oa::Matrix& inObservation) override;
	[[nodiscard]] const CategoricalActorCriticConfig& config() const noexcept {
		return config_;
	}

private:
	explicit CategoricalActorCritic(const CategoricalActorCriticConfig& inConfig);

	CategoricalActorCriticConfig config_;
	oa::SharedPtr<oa::Linear> policy0_;
	oa::SharedPtr<oa::Linear> policy1_;
	oa::SharedPtr<oa::Linear> policy_;
	oa::SharedPtr<oa::Linear> value0_;
	oa::SharedPtr<oa::Linear> value1_;
	oa::SharedPtr<oa::Linear> value_;
};

} // namespace oa
