#pragma once

#include "LunarTerrain.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

class OaLunarSupportSphere {
public:
	OaLunarVec3 BodyOffset_;
	double Radius_ = 0.1;

	constexpr OaLunarSupportSphere() noexcept = default;
	constexpr OaLunarSupportSphere(
		const OaLunarVec3& InBodyOffset,
		double InRadius) noexcept
		: BodyOffset_(InBodyOffset), Radius_(InRadius) {}
};

class OaLunarLander3dConfig {
public:
	std::uint32_t EnvironmentVersion_ = OA_LUNAR_ENVIRONMENT_VERSION;
	std::uint32_t PhysicsVersion_ = OA_LUNAR_PHYSICS_VERSION;
	std::uint32_t ObservationVersion_ = OA_LUNAR_OBSERVATION_VERSION;
	std::uint32_t RewardVersion_ = OA_LUNAR_REWARD_VERSION;
	OaLunarTerrainConfig Terrain_;

	double PolicyTimeStep_ = 1.0 / 60.0;
	std::uint32_t PhysicsSubsteps_ = 4U;
	std::uint32_t ContactIterations_ = 4U;
	double Gravity_ = 1.62;
	double Mass_ = 1200.0;
	OaLunarVec3 DiagonalInertia_{900.0, 800.0, 900.0};
	double MainThrust_ = 4200.0;
	double AttitudeTorque_ = 900.0;
	double FuelCapacity_ = 100.0;
	double MainFuelRate_ = 2.0;
	double AttitudeFuelRate_ = 0.35;

	double Restitution_ = 0.05;
	double Friction_ = 0.65;
	double ContactSlop_ = 0.01;
	double PenetrationCorrectionFraction_ = 0.6;
	double MaxPositionCorrectionPerContact_ = 0.03;
	double MaxContactImpulse_ = 6000.0;
	double MaxBiasSpeed_ = 1.0;

	double TaskMinimumY_ = -2.0;
	double TaskMaximumY_ = 40.0;
	double SafeLinearSpeed_ = 0.55;
	double SafeAngularSpeed_ = 0.25;
	double SafeTiltRadians_ = 0.18;
	double HardFootImpactSpeed_ = 1.2;
	std::uint32_t SafeDwellSteps_ = 20U;
	std::uint32_t MaxEpisodeSteps_ = 1200U;

	double PositionObservationScale_ = 16.0;
	double VelocityObservationScale_ = 4.0;
	double AngularVelocityObservationScale_ = 2.0;
	double TerrainClearanceObservationScale_ = 8.0;
	double FootClearanceObservationScale_ = 2.0;
	double TerrainProbeSpacing_ = 1.5;

	double RewardGamma_ = 0.99;
	double PositionPotentialWeight_ = 1.0;
	double VelocityPotentialWeight_ = 0.6;
	double TiltPotentialWeight_ = 0.5;
	double AngularPotentialWeight_ = 0.25;
	double MainFuelCostWeight_ = 0.04;
	double AttitudeFuelCostWeight_ = 0.02;
	double SoftFootContactReward_ = 0.02;
	double StableDwellReward_ = 0.05;
	double SuccessReward_ = 100.0;
	double FailurePenalty_ = -100.0;

	std::array<OaLunarSupportSphere, 3U> BodySupports_ = {{
		{OaLunarVec3(0.0, 0.25, 0.0), 0.50},
		{OaLunarVec3(0.0, -0.15, 0.0), 0.50},
		{OaLunarVec3(0.0, 0.65, 0.0), 0.38},
	}};

	std::array<OaLunarSupportSphere, 4U> FootSupports_ = {{
		{OaLunarVec3(-0.85, -1.0, -0.85), 0.15},
		{OaLunarVec3(0.85, -1.0, -0.85), 0.15},
		{OaLunarVec3(0.85, -1.0, 0.85), 0.15},
		{OaLunarVec3(-0.85, -1.0, 0.85), 0.15},
	}};

	[[nodiscard]] std::string ValidationError() const;
	[[nodiscard]] std::uint64_t ContractFingerprint() const noexcept;
};

class OaLunarLander3dState {
public:
	OaLunarVec3 Position_{0.0, 6.0, 0.0};
	OaLunarVec3 LinearVelocity_;
	OaLunarQuat Orientation_;
	OaLunarVec3 AngularVelocityBody_;
	double Fuel_ = 100.0;
	OaLunarAction LastAction_ = OaLunarAction::Coast;
	double MainThrottle_ = 0.0;
	OaLunarVec3 AttitudeCommandBody_;
	std::array<bool, 3U> BodyContacts_{};
	std::array<double, 3U> BodyContactImpulses_{};
	std::array<bool, 4U> FootContacts_{};
	std::array<bool, 4U> FeetOnPad_{};
	std::array<double, 4U> FootContactImpulses_{};
	// Each foot may earn the soft-contact reward once per episode. Reset clears
	// these latches; losing and regaining contact does not create another award.
	std::array<bool, 4U> FootContactRewarded_{};
	std::uint32_t EpisodeStep_ = 0U;
	std::uint32_t StableDwell_ = 0U;
	bool Terminated_ = false;
	bool Truncated_ = false;
	OaLunarEndReason EndReason_ = OaLunarEndReason::None;
	double EpisodeReturn_ = 0.0;

	[[nodiscard]] bool IsFinite() const noexcept;
};

class OaLunarContactDiagnostics {
public:
	double MaximumPenetration_ = 0.0;
	double MaximumNormalImpulse_ = 0.0;
	double MaximumFrictionImpulse_ = 0.0;
	double MaximumFootClosingSpeed_ = 0.0;
	double TotalPositionCorrection_ = 0.0;
	std::uint32_t ContactCount_ = 0U;
	bool BodyContactOccurred_ = false;
	bool FootContactOccurred_ = false;
	bool Bounded_ = true;

	[[nodiscard]] bool IsFinite() const noexcept;
};

class OaLunarPhysicsResult {
public:
	bool Valid_ = true;
	std::string Error_;
	double MainFuelUsed_ = 0.0;
	double AttitudeFuelUsed_ = 0.0;
	OaLunarContactDiagnostics Contact_;
};

class OaLunarRewardTerms {
public:
	double PotentialBefore_ = 0.0;
	double PotentialAfter_ = 0.0;
	double Shaping_ = 0.0;
	double MainFuelCost_ = 0.0;
	double AttitudeFuelCost_ = 0.0;
	double SoftFootContact_ = 0.0;
	double StableDwell_ = 0.0;
	double Terminal_ = 0.0;
	double Total_ = 0.0;

	[[nodiscard]] bool IsFinite() const noexcept;
	[[nodiscard]] double Sum() const noexcept;
};

class OaLunarTransition {
public:
	bool Valid_ = false;
	std::string Error_;
	std::array<float, OA_LUNAR_OBSERVATION_SIZE> Observation_{};
	double Reward_ = 0.0;
	bool Terminated_ = false;
	bool Truncated_ = false;
	OaLunarEndReason EndReason_ = OaLunarEndReason::None;
	OaLunarRewardTerms RewardTerms_;
	OaLunarContactDiagnostics Contact_;
};

class OaLunarScalarPhysics {
public:
	// Semi-implicit Euler uses exactly PhysicsSubsteps_. Each contact iteration
	// visits the three body supports in index order, then the four feet in index
	// order. Impulses and position corrections are clamped by the configuration.
	[[nodiscard]] static OaLunarPhysicsResult Integrate(
		const OaLunarLander3dConfig& InConfig,
		const OaLunarTerrain& InTerrain,
		OaLunarAction InAction,
		OaLunarLander3dState& InOutState
	);
	[[nodiscard]] static std::array<float, OA_LUNAR_OBSERVATION_SIZE> Observe(
		const OaLunarLander3dConfig& InConfig,
		const OaLunarTerrain& InTerrain,
		const OaLunarLander3dState& InState
	) noexcept;
	[[nodiscard]] static double Potential(
		const OaLunarLander3dConfig& InConfig,
		const OaLunarLander3dState& InState
	) noexcept;
	[[nodiscard]] static OaLunarVec3 SupportWorldCenter(
		const OaLunarLander3dState& InState,
		const OaLunarSupportSphere& InSupport
	) noexcept;
};
