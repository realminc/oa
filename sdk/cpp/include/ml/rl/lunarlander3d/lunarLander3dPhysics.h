#pragma once

#include <ml/rl/lunarlander3d/lunarTerrain.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace oa {

class LunarSupportSphere {
public:
	oa::vlm::DVec3 bodyOffset_;
	double radius_ = 0.1;

	constexpr LunarSupportSphere() noexcept = default;
	constexpr LunarSupportSphere(
		const oa::vlm::DVec3& inBodyOffset,
		double inRadius) noexcept
		: bodyOffset_(inBodyOffset), radius_(inRadius) {}
};

class LunarLander3dConfig {
public:
	std::uint32_t environmentVersion_ = kLunarEnvironmentVersion;
	std::uint32_t physicsVersion_ = kLunarPhysicsVersion;
	std::uint32_t observationVersion_ = kLunarObservationVersion;
	std::uint32_t rewardVersion_ = kLunarRewardVersion;
	LunarTerrainConfig terrain_;

	double policyTimeStep_ = 1.0 / 60.0;
	std::uint32_t physicsSubsteps_ = 4U;
	std::uint32_t contactIterations_ = 4U;
	double gravity_ = 1.62;
	double mass_ = 1200.0;
	oa::vlm::DVec3 diagonalInertia_{900.0, 800.0, 900.0};
	double mainThrust_ = 4200.0;
	double attitudeTorque_ = 900.0;
	double fuelCapacity_ = 100.0;
	double mainFuelRate_ = 2.0;
	double attitudeFuelRate_ = 0.35;

	double restitution_ = 0.05;
	double friction_ = 0.65;
	double contactSlop_ = 0.01;
	double penetrationCorrectionFraction_ = 0.6;
	double maxPositionCorrectionPerContact_ = 0.03;
	double maxContactImpulse_ = 6000.0;
	double maxBiasSpeed_ = 1.0;

	double taskMinimumY_ = -2.0;
	double taskMaximumY_ = 40.0;
	double safeLinearSpeed_ = 0.55;
	double safeAngularSpeed_ = 0.25;
	double safeTiltRadians_ = 0.18;
	double hardFootImpactSpeed_ = 1.2;
	std::uint32_t safeDwellSteps_ = 20U;
	std::uint32_t maxEpisodeSteps_ = 1200U;

	double positionObservationScale_ = 16.0;
	double velocityObservationScale_ = 4.0;
	double angularVelocityObservationScale_ = 2.0;
	double terrainClearanceObservationScale_ = 8.0;
	double footClearanceObservationScale_ = 2.0;
	double terrainProbeSpacing_ = 1.5;

	double rewardGamma_ = 0.99;
	double positionPotentialWeight_ = 1.0;
	double velocityPotentialWeight_ = 0.6;
	double tiltPotentialWeight_ = 0.5;
	double angularPotentialWeight_ = 0.25;
	double mainFuelCostWeight_ = 0.04;
	double attitudeFuelCostWeight_ = 0.02;
	double softFootContactReward_ = 0.02;
	double stableDwellReward_ = 0.05;
	double successReward_ = 100.0;
	double failurePenalty_ = -100.0;

	std::array<LunarSupportSphere, 3U> bodySupports_ = {{
		{oa::vlm::DVec3(0.0, 0.25, 0.0), 0.50},
		{oa::vlm::DVec3(0.0, -0.15, 0.0), 0.50},
		{oa::vlm::DVec3(0.0, 0.65, 0.0), 0.38},
	}};

	std::array<LunarSupportSphere, 4U> footSupports_ = {{
		{oa::vlm::DVec3(-0.85, -1.0, -0.85), 0.15},
		{oa::vlm::DVec3(0.85, -1.0, -0.85), 0.15},
		{oa::vlm::DVec3(0.85, -1.0, 0.85), 0.15},
		{oa::vlm::DVec3(-0.85, -1.0, 0.85), 0.15},
	}};

	[[nodiscard]] std::string validationError() const;
	[[nodiscard]] std::uint64_t contractFingerprint() const noexcept;
};

class LunarLander3dState {
public:
	oa::vlm::DVec3 position_{0.0, 6.0, 0.0};
	oa::vlm::DVec3 linearVelocity_;
	oa::vlm::DQuat orientation_;
	oa::vlm::DVec3 angularVelocityBody_;
	double fuel_ = 100.0;
	LunarAction lastAction_ = LunarAction::Coast;
	double mainThrottle_ = 0.0;
	oa::vlm::DVec3 attitudeCommandBody_;
	std::array<bool, 3U> bodyContacts_{};
	std::array<double, 3U> bodyContactImpulses_{};
	std::array<bool, 4U> footContacts_{};
	std::array<bool, 4U> feetOnPad_{};
	std::array<double, 4U> footContactImpulses_{};
	// Each foot may earn the soft-contact reward once per episode. reset clears
	// these latches; losing and regaining contact does not create another award.
	std::array<bool, 4U> footContactRewarded_{};
	std::uint32_t episodeStep_ = 0U;
	std::uint32_t stableDwell_ = 0U;
	bool terminated_ = false;
	bool truncated_ = false;
	LunarEndReason endReason_ = LunarEndReason::None;
	double episodeReturn_ = 0.0;

	[[nodiscard]] bool isFinite() const noexcept;
};

class LunarContactDiagnostics {
public:
	double maximumPenetration_ = 0.0;
	double maximumNormalImpulse_ = 0.0;
	double maximumFrictionImpulse_ = 0.0;
	double maximumFootClosingSpeed_ = 0.0;
	double totalPositionCorrection_ = 0.0;
	std::uint32_t contactCount_ = 0U;
	bool bodyContactOccurred_ = false;
	bool footContactOccurred_ = false;
	bool bounded_ = true;

	[[nodiscard]] bool isFinite() const noexcept;
};

class LunarPhysicsResult {
public:
	bool valid_ = true;
	std::string error_;
	double mainFuelUsed_ = 0.0;
	double attitudeFuelUsed_ = 0.0;
	LunarContactDiagnostics contact_;
};

class LunarRewardTerms {
public:
	double potentialBefore_ = 0.0;
	double potentialAfter_ = 0.0;
	double shaping_ = 0.0;
	double mainFuelCost_ = 0.0;
	double attitudeFuelCost_ = 0.0;
	double softFootContact_ = 0.0;
	double stableDwell_ = 0.0;
	double terminal_ = 0.0;
	double total_ = 0.0;

	[[nodiscard]] bool isFinite() const noexcept;
	[[nodiscard]] double sum() const noexcept;
};

class LunarTransition {
public:
	bool valid_ = false;
	std::string error_;
	std::array<float, kLunarObservationSize> observation_{};
	double reward_ = 0.0;
	bool terminated_ = false;
	bool truncated_ = false;
	LunarEndReason endReason_ = LunarEndReason::None;
	LunarRewardTerms rewardTerms_;
	LunarContactDiagnostics contact_;
};

namespace FnLunarLander {
	// Semi-implicit Euler uses exactly physicsSubsteps_. Each contact iteration
	// visits the three body supports in index order, then the four feet in index
	// order. Impulses and position corrections are clamped by the configuration.
	[[nodiscard]] LunarPhysicsResult integrate(
		const LunarLander3dConfig& inConfig,
		const LunarTerrain& inTerrain,
		LunarAction inAction,
		LunarLander3dState& inOutState
	);
	[[nodiscard]] std::array<float, kLunarObservationSize>
	observe(
		const LunarLander3dConfig& inConfig,
		const LunarTerrain& inTerrain,
		const LunarLander3dState& inState
	) noexcept;
	[[nodiscard]] double potential(
		const LunarLander3dConfig& inConfig,
		const LunarLander3dState& inState
	) noexcept;
	[[nodiscard]] oa::vlm::DVec3 supportWorldCenter(
		const LunarLander3dState& inState,
		const LunarSupportSphere& inSupport
	) noexcept;
} // namespace FnLunarLander

} // namespace oa
