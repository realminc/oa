#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Ml/Rl/Environment.h>

struct OaTutorialCartPoleConfig {
	OaU32 Environments = 1;
	OaU32 MaxEpisodeSteps = 500;
	OaU64 Seed = 1;
	OaF32 Gravity = 9.8F;
	OaF32 CartMass = 1.0F;
	OaF32 PoleMass = 0.1F;
	OaF32 HalfPoleLength = 0.5F;
	OaF32 ForceMagnitude = 10.0F;
	OaF32 TimeStep = 0.02F;
	OaF32 PositionThreshold = 2.4F;
	OaF32 AngleThresholdRadians = 0.2094395102F;
};

struct OaTutorialCartPoleStep {
	OaMatrix Observation;     // pre-action FP32 [E,4]
	OaMatrix NextObservation; // post-action FP32 [E,4]
	OaMatrix Reward;          // FP32 [E]
	OaMatrix Terminated;      // UInt8 [E]
	OaMatrix Truncated;       // UInt8 [E]
	OaMatrix Done;            // UInt8 [E]

	[[nodiscard]] bool IsValid() const noexcept;
};

// Tutorial-local vector environment. It deliberately does not establish a
// public OaEnv hierarchy: CartPole is the acceptance fixture used to discover
// that contract.
class OaTutorialCartPole : public OaRlEnvironment {
public:
	OaTutorialCartPole() = default;
	OaTutorialCartPole(const OaTutorialCartPole&) = delete;
	OaTutorialCartPole& operator=(const OaTutorialCartPole&) = delete;
	OaTutorialCartPole(OaTutorialCartPole&&) noexcept = default;
	OaTutorialCartPole& operator=(OaTutorialCartPole&&) noexcept = default;

	[[nodiscard]] static OaResult<OaTutorialCartPole> Create(
		const OaTutorialCartPoleConfig& InConfig);

	// Full deterministic reset. Repeating Reset with the same configured seed
	// produces the same initial states.
	void Reset();
	// Resets only lanes marked Done by the preceding Step. Call after preserving
	// the terminal transition in the rollout buffer.
	void ResetDone();

	// Records one standard CartPole Euler integration step. Actions are Int32
	// [E], with 0 = left and 1 = right.
	[[nodiscard]] OaResult<OaTutorialCartPoleStep> Step(
		const OaMatrix& InAction);

	[[nodiscard]] bool IsValid() const noexcept;
	[[nodiscard]] const OaMatrix& Observation() const noexcept override { return State_; }
	[[nodiscard]] const OaMatrix& Done() const noexcept { return Done_; }
	[[nodiscard]] const OaMatrix& EpisodeSteps() const noexcept { return EpisodeSteps_; }
	[[nodiscard]] const OaMatrix& EpisodeIndex() const noexcept { return EpisodeIndex_; }
	[[nodiscard]] const OaTutorialCartPoleConfig& Config() const noexcept { return Config_; }
	[[nodiscard]] const OaRlEnvironmentSpec& Spec() const noexcept override { return Spec_; }
	[[nodiscard]] OaU32 Environments() const noexcept override {
		return Config_.Environments;
	}
	[[nodiscard]] OaStatus ResetEnvironment(OaU64 InSeed) override;
	[[nodiscard]] OaResult<OaRlEnvironmentTransition> StepEnvironment(
		const OaMatrix& InAction) override;
	[[nodiscard]] OaStatus ResetCompleted() override;

private:
	void RecordReset_(bool InOnlyDone);

	OaTutorialCartPoleConfig Config_;
	OaRlEnvironmentSpec Spec_;
	OaMatrix State_;
	OaMatrix TransitionObservation_;
	OaMatrix Reward_;
	OaMatrix Terminated_;
	OaMatrix Truncated_;
	OaMatrix Done_;
	OaMatrix EpisodeSteps_;
	OaMatrix EpisodeIndex_;
};
