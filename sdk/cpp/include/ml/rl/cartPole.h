#pragma once

#include <oa/core/matrix.h>
#include <oa/core/status.h>
#include <oa/ml/environment.h>

namespace oa {

struct CartPoleConfig {
	static constexpr oa::U32 dynamicsVersion = 1;

	oa::U32 environments = 1;
	oa::U32 maxEpisodeSteps = 500;
	oa::U64 seed = 1;
	oa::F32 gravity = 9.8F;
	oa::F32 cartMass = 1.0F;
	oa::F32 poleMass = 0.1F;
	oa::F32 halfPoleLength = 0.5F;
	oa::F32 forceMagnitude = 10.0F;
	oa::F32 timeStep = 0.02F;
	oa::F32 positionThreshold = 2.4F;
	oa::F32 angleThresholdRadians = 0.2094395102F;

	// Stable, versioned identity of every field that changes step behavior.
	// Lane count is shape-derived and reset seed deliberately is not dynamics.
	[[nodiscard]] oa::U64 dynamicsIdentity() const noexcept;
};

struct CartPoleStep {
	oa::Matrix observation;     // pre-action FP32 [E,4]
	oa::Matrix nextObservation; // post-action FP32 [E,4]
	oa::Matrix reward;          // FP32 [E]
	oa::Matrix terminated;      // UInt8 [E]
	oa::Matrix truncated;       // UInt8 [E]
	oa::Matrix done;            // UInt8 [E]

	[[nodiscard]] bool isValid() const noexcept;
};

// SDK-owned vectorized CartPole workload. It exercises the reusable
// oa::Environment contract without adding task-specific code to liboa.
class CartPole : public oa::Environment {
public:
	using oa::Environment::reset;

	CartPole(const CartPole&) = delete;
	CartPole& operator=(const CartPole&) = delete;
	CartPole(CartPole&&) noexcept = default;
	CartPole& operator=(CartPole&&) noexcept = default;

	[[nodiscard]] static oa::Result<CartPole> create(
		oa::Engine& inEngine,
		const CartPoleConfig& inConfig);

	// Full deterministic reset. Repeating reset with the same configured seed
	// produces the same initial states.
	[[nodiscard]] oa::Status reset();
	// Resets only lanes marked done by the preceding step. call after preserving
	// the terminal transition in the rollout buffer.
	[[nodiscard]] oa::Status resetDone();

	// Records one standard CartPole Euler integration step. actions are Int32
	// [E], with 0 = left and 1 = right.
	[[nodiscard]] oa::Result<CartPoleStep> step(
		const oa::Matrix& inAction);

	[[nodiscard]] bool isValid() const noexcept;
	[[nodiscard]] const oa::Matrix& observation() const noexcept override { return state_; }
	[[nodiscard]] const oa::Matrix& done() const noexcept { return done_; }
	[[nodiscard]] const oa::Matrix& episodeSteps() const noexcept { return episodeSteps_; }
	[[nodiscard]] const oa::Matrix& episodeIndex() const noexcept { return episodeIndex_; }
	[[nodiscard]] const CartPoleConfig& config() const noexcept { return config_; }
	[[nodiscard]] const oa::EnvironmentSpec& spec() const noexcept override { return spec_; }
	[[nodiscard]] oa::U32 environments() const noexcept override {
		return config_.environments;
	}
private:
	explicit CartPole(oa::Engine& inEngine);
	[[nodiscard]] oa::Status recordReset_(bool inOnlyDone);
	[[nodiscard]] oa::Result<CartPoleStep> recordDetailedStep_(
		const oa::Matrix& inAction);
	[[nodiscard]] oa::U64 effectiveSeed_() const noexcept;

protected:
	[[nodiscard]] oa::Status recordReset_(oa::U64 inSeed) override;
	[[nodiscard]] oa::Result<oa::EnvironmentTransition>
		recordStep_(const oa::Matrix& inAction) override;
	[[nodiscard]] oa::Status recordResetCompleted_() override;
	void commitRecordedState_() noexcept override;
	void rollbackRecordedState_() noexcept override;

private:
	CartPoleConfig config_;
	oa::EnvironmentSpec spec_;
	oa::Matrix state_;
	oa::Matrix transitionObservation_;
	oa::Matrix reward_;
	oa::Matrix terminated_;
	oa::Matrix truncated_;
	oa::Matrix done_;
	oa::Matrix episodeSteps_;
	oa::Matrix episodeIndex_;
	oa::U64 pendingSeed_ = 0;
	bool hasPendingSeed_ = false;
	bool hasCommittedState_ = false;
	bool hasPendingFullReset_ = false;
};

} // namespace oa
