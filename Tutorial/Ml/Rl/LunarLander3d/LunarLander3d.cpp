#include "LunarLander3d.h"

#include <algorithm>
#include <cmath>
#include <utility>

static bool OaLunarAllFootContacts(
	const OaLunarLander3dState& InState) noexcept {
	for (const bool contact : InState.FootContacts_) {
		if (not contact) return false;
	}
	return true;
}

static bool OaLunarAllFeetOnPad(
	const OaLunarLander3dState& InState) noexcept {
	for (const bool onPad : InState.FeetOnPad_) {
		if (not onPad) return false;
	}
	return true;
}

static bool OaLunarAnyBodyContact(
	const OaLunarLander3dState& InState) noexcept {
	for (const bool contact : InState.BodyContacts_) {
		if (contact) return true;
	}
	return false;
}

static bool OaLunarInstantaneouslySafe(
	const OaLunarLander3dConfig& InConfig,
	const OaLunarLander3dState& InState) noexcept {
	const OaLunarVec3 bodyUp = InState.Orientation_.Rotate({0.0, 1.0, 0.0});
	const double tilt = std::acos(std::clamp(bodyUp.ComponentY_, -1.0, 1.0));
	return OaLunarAllFootContacts(InState)
		and OaLunarAllFeetOnPad(InState)
		and not OaLunarAnyBodyContact(InState)
		and InState.LinearVelocity_.Length() <= InConfig.SafeLinearSpeed_
		and InState.AngularVelocityBody_.Length() <= InConfig.SafeAngularSpeed_
		and tilt <= InConfig.SafeTiltRadians_;
}

OaLunarAction OaLunarScriptedLandingAction(
	const OaLunarLander3dConfig& InConfig,
	const OaLunarLander3dState& InState) noexcept {
	double uprightFootContactHeight = 0.0;
	for (const OaLunarSupportSphere& foot : InConfig.FootSupports_) {
		uprightFootContactHeight = std::max(
			uprightFootContactHeight,
			foot.Radius_ - foot.BodyOffset_.ComponentY_);
	}
	const double footClearance = std::max(
		0.0, InState.Position_.ComponentY_ - uprightFootContactHeight);
	const double desiredDescent = -std::clamp(
		0.48 * std::sqrt(footClearance), 0.24, 0.72);
	if (InState.LinearVelocity_.ComponentY_ < desiredDescent) {
		return OaLunarAction::MainEngine;
	}

	const OaLunarVec3 bodyUp = InState.Orientation_.Rotate({0.0, 1.0, 0.0});
	const double guidanceBlend = std::clamp(footClearance, 0.0, 1.0);
	constexpr double lateralPositionGain = 0.055;
	constexpr double lateralVelocityGain = 0.30;
	constexpr double maximumGuidanceTilt = 0.12;
	const double targetBodyUpX = guidanceBlend * std::clamp(
		-lateralPositionGain * InState.Position_.ComponentX_
			- lateralVelocityGain * InState.LinearVelocity_.ComponentX_,
		-maximumGuidanceTilt, maximumGuidanceTilt);
	const double targetBodyUpZ = guidanceBlend * std::clamp(
		-lateralPositionGain * InState.Position_.ComponentZ_
			- lateralVelocityGain * InState.LinearVelocity_.ComponentZ_,
		-maximumGuidanceTilt, maximumGuidanceTilt);
	constexpr double proportionalGain = 8.0;
	constexpr double dampingGain = 3.0;
	constexpr double commandDeadZone = 0.025;
	const double pitchCommand = -proportionalGain
		* (bodyUp.ComponentZ_ - targetBodyUpZ)
		- dampingGain * InState.AngularVelocityBody_.ComponentX_;
	const double rollCommand = proportionalGain
		* (bodyUp.ComponentX_ - targetBodyUpX)
		- dampingGain * InState.AngularVelocityBody_.ComponentZ_;
	if (std::max(std::abs(pitchCommand), std::abs(rollCommand))
		<= commandDeadZone) {
		return OaLunarAction::Coast;
	}
	if (std::abs(pitchCommand) >= std::abs(rollCommand)) {
		return pitchCommand > 0.0
			? OaLunarAction::PitchPositive
			: OaLunarAction::PitchNegative;
	}
	return rollCommand > 0.0
		? OaLunarAction::RollPositive
		: OaLunarAction::RollNegative;
}

OaLunarScalarEnvironment OaLunarScalarEnvironment::Invalid_(
	const OaLunarLander3dConfig& InConfig,
	const OaLunarEpisodeManifest& InManifest,
	const OaLunarTerrain& InTerrain,
	std::string InError) {
	OaLunarScalarEnvironment environment;
	environment.Config_ = InConfig;
	environment.Manifest_ = InManifest;
	environment.Terrain_ = InTerrain;
	environment.Error_ = std::move(InError);
	return environment;
}

OaLunarScalarEnvironment OaLunarScalarEnvironment::CreateFlat(
	const OaLunarLander3dConfig& InConfig,
	const OaLunarEpisodeManifest& InManifest) {
	return CreateWithTerrain(
		InConfig, InManifest, OaLunarTerrain::CreateFlat(InConfig.Terrain_));
}

OaLunarScalarEnvironment OaLunarScalarEnvironment::CreateSeeded(
	const OaLunarLander3dConfig& InConfig,
	const OaLunarEpisodeManifest& InManifest) {
	return CreateWithTerrain(
		InConfig, InManifest,
		OaLunarTerrain::CreateSeeded(InConfig.Terrain_, InManifest));
}

OaLunarScalarEnvironment OaLunarScalarEnvironment::CreateWithTerrain(
	const OaLunarLander3dConfig& InConfig,
	const OaLunarEpisodeManifest& InManifest,
	const OaLunarTerrain& InTerrain) {
	const std::string configError = InConfig.ValidationError();
	if (not configError.empty()) {
		return Invalid_(InConfig, InManifest, InTerrain, configError);
	}
	const std::string manifestError = InManifest.ValidationError();
	if (not manifestError.empty()) {
		return Invalid_(InConfig, InManifest, InTerrain, manifestError);
	}
	if (InManifest.EnvironmentVersion_ != InConfig.EnvironmentVersion_
		or InManifest.PhysicsVersion_ != InConfig.PhysicsVersion_
		or InManifest.ObservationVersion_ != InConfig.ObservationVersion_
		or InManifest.RewardVersion_ != InConfig.RewardVersion_
		or InManifest.TerrainVersion_ != OA_LUNAR_TERRAIN_VERSION) {
		return Invalid_(
			InConfig, InManifest, InTerrain,
			"lunar manifest versions do not match the environment configuration");
	}
	if (InManifest.ConfigFingerprint_ != InConfig.ContractFingerprint()) {
		return Invalid_(
			InConfig, InManifest, InTerrain,
			"lunar manifest configuration fingerprint does not match");
	}
	if (not InTerrain.IsValid()) {
		return Invalid_(InConfig, InManifest, InTerrain, InTerrain.Error());
	}
	if (InTerrain.Config() != InConfig.Terrain_) {
		return Invalid_(
			InConfig, InManifest, InTerrain,
			"lunar terrain configuration does not match the environment");
	}

	OaLunarScalarEnvironment environment;
	environment.Config_ = InConfig;
	environment.Manifest_ = InManifest;
	environment.Terrain_ = InTerrain;
	environment.Error_.clear();
	if (not environment.Reset()) {
		environment.Error_ = "lunar deterministic reset produced an invalid state";
	}
	return environment;
}

OaLunarLander3dState OaLunarScalarEnvironment::SpawnState_() const noexcept {
	OaLunarLander3dState state;
	const double padRange = Config_.Terrain_.PadHalfExtent_ * 0.35;
	state.Position_.ComponentX_ = (
		Manifest_.Sample01(OaLunarRandomPurpose::Spawn, 0U) * 2.0 - 1.0)
		* padRange;
	state.Position_.ComponentY_ = 5.0
		+ Manifest_.Sample01(OaLunarRandomPurpose::Spawn, 1U) * 2.0;
	state.Position_.ComponentZ_ = (
		Manifest_.Sample01(OaLunarRandomPurpose::Spawn, 2U) * 2.0 - 1.0)
		* padRange;
	state.LinearVelocity_.ComponentX_ = (
		Manifest_.Sample01(OaLunarRandomPurpose::Spawn, 3U) * 2.0 - 1.0)
		* 0.12;
	state.LinearVelocity_.ComponentY_ = -0.1
		- Manifest_.Sample01(OaLunarRandomPurpose::Spawn, 4U) * 0.2;
	state.LinearVelocity_.ComponentZ_ = (
		Manifest_.Sample01(OaLunarRandomPurpose::Spawn, 5U) * 2.0 - 1.0)
		* 0.12;
	const double pitch = (
		Manifest_.Sample01(OaLunarRandomPurpose::Spawn, 6U) * 2.0 - 1.0)
		* 0.03;
	const double roll = (
		Manifest_.Sample01(OaLunarRandomPurpose::Spawn, 7U) * 2.0 - 1.0)
		* 0.03;
	const double yaw = (
		Manifest_.Sample01(OaLunarRandomPurpose::Spawn, 8U) * 2.0 - 1.0)
		* 0.08;
	state.Orientation_ = (
		OaLunarQuat::FromAxisAngle({0.0, 1.0, 0.0}, yaw)
		* OaLunarQuat::FromAxisAngle({1.0, 0.0, 0.0}, pitch)
		* OaLunarQuat::FromAxisAngle({0.0, 0.0, 1.0}, roll)).Normalized();
	state.Fuel_ = Config_.FuelCapacity_;
	return state;
}

bool OaLunarScalarEnvironment::Reset() noexcept {
	if (not Error_.empty()) return false;
	State_ = SpawnState_();
	return State_.IsFinite();
}

bool OaLunarScalarEnvironment::SetState(
	const OaLunarLander3dState& InState) noexcept {
	if (not IsValid() or not InState.IsFinite()
		or InState.Fuel_ < 0.0 or InState.Fuel_ > Config_.FuelCapacity_
		or (InState.Terminated_ and InState.Truncated_)
		or ((InState.Terminated_ or InState.Truncated_)
			and InState.EndReason_ == OaLunarEndReason::None)
		or (not InState.Terminated_ and not InState.Truncated_
			and InState.EndReason_ != OaLunarEndReason::None)) {
		return false;
	}
	State_ = InState;
	State_.Orientation_ = State_.Orientation_.Normalized();
	return true;
}

std::array<float, OA_LUNAR_OBSERVATION_SIZE>
OaLunarScalarEnvironment::Observation() const noexcept {
	return OaLunarScalarPhysics::Observe(Config_, Terrain_, State_);
}

OaLunarTransition OaLunarScalarEnvironment::Step(
	std::uint32_t InAction,
	bool InExternalStop) {
	OaLunarTransition transition;
	if (not IsValid()) {
		transition.Error_ = Error_;
		return transition;
	}
	if (State_.Terminated_ or State_.Truncated_) {
		transition.Error_ = "lunar episode has ended; reset is required";
		return transition;
	}
	if (not OaLunarActionIsValid(InAction)) {
		// Batched device execution cannot reject one lane without rejecting the
		// whole submission. Invalid actions therefore consume one transition and
		// terminate only that lane with the configured failure penalty.
		++State_.EpisodeStep_;
		State_.Terminated_ = true;
		State_.EndReason_ = OaLunarEndReason::InvalidAction;
		State_.EpisodeReturn_ += Config_.FailurePenalty_;
		transition.Valid_ = true;
		transition.Observation_ = Observation();
		transition.Reward_ = Config_.FailurePenalty_;
		transition.Terminated_ = true;
		transition.EndReason_ = State_.EndReason_;
		transition.RewardTerms_.Terminal_ = Config_.FailurePenalty_;
		transition.RewardTerms_.Total_ = Config_.FailurePenalty_;
		return transition;
	}
	if (InExternalStop) {
		State_.Truncated_ = true;
		State_.EndReason_ = OaLunarEndReason::ExternalStop;
		transition.Valid_ = true;
		transition.Observation_ = Observation();
		transition.Truncated_ = true;
		transition.EndReason_ = State_.EndReason_;
		return transition;
	}

	const double potentialBefore = OaLunarScalarPhysics::Potential(
		Config_, State_);
	OaLunarPhysicsResult physics = OaLunarScalarPhysics::Integrate(
		Config_, Terrain_, static_cast<OaLunarAction>(InAction), State_);
	++State_.EpisodeStep_;
	bool instantaneousSafe = false;
	if (not physics.Valid_) {
		State_.Terminated_ = true;
		State_.EndReason_ = OaLunarEndReason::NumericalFailure;
	} else if (not State_.IsFinite()) {
		State_.Terminated_ = true;
		State_.EndReason_ = OaLunarEndReason::NumericalFailure;
	} else if (not Terrain_.Contains(State_.Position_.ComponentX_, State_.Position_.ComponentZ_)
		or State_.Position_.ComponentY_ < Config_.TaskMinimumY_
		or State_.Position_.ComponentY_ > Config_.TaskMaximumY_) {
		State_.Terminated_ = true;
		State_.EndReason_ = OaLunarEndReason::OutOfBounds;
	} else if (physics.Contact_.BodyContactOccurred_
		or OaLunarAnyBodyContact(State_)) {
		State_.Terminated_ = true;
		State_.EndReason_ = OaLunarEndReason::BodyImpact;
	} else if (physics.Contact_.MaximumFootClosingSpeed_
		> Config_.HardFootImpactSpeed_) {
		State_.Terminated_ = true;
		State_.EndReason_ = OaLunarEndReason::HardFootImpact;
	} else {
		instantaneousSafe = OaLunarInstantaneouslySafe(Config_, State_);
		State_.StableDwell_ = instantaneousSafe
			? State_.StableDwell_ + 1U : 0U;
		if (State_.StableDwell_ >= Config_.SafeDwellSteps_) {
			State_.Terminated_ = true;
			State_.EndReason_ = OaLunarEndReason::SafeLanding;
		}
	}
	if (not State_.Terminated_
		and State_.EpisodeStep_ >= Config_.MaxEpisodeSteps_) {
		State_.Truncated_ = true;
		State_.EndReason_ = OaLunarEndReason::TimeLimit;
	}

	OaLunarRewardTerms reward;
	reward.PotentialBefore_ = potentialBefore;
	reward.PotentialAfter_ = OaLunarScalarPhysics::Potential(Config_, State_);
	const double effectivePotentialAfter = State_.Terminated_
		? 0.0 : reward.PotentialAfter_;
	reward.Shaping_ = Config_.RewardGamma_ * effectivePotentialAfter
		- reward.PotentialBefore_;
	reward.MainFuelCost_ = -Config_.MainFuelCostWeight_ * physics.MainFuelUsed_;
	reward.AttitudeFuelCost_ = -Config_.AttitudeFuelCostWeight_
		* physics.AttitudeFuelUsed_;
	if (physics.Contact_.MaximumFootClosingSpeed_
		<= Config_.HardFootImpactSpeed_) {
		for (std::size_t footIndex = 0U;
			footIndex < State_.FootContacts_.size();
			++footIndex) {
			if (State_.FootContacts_[footIndex]
				and not State_.FootContactRewarded_[footIndex]) {
				reward.SoftFootContact_ += Config_.SoftFootContactReward_;
				State_.FootContactRewarded_[footIndex] = true;
			}
		}
	}
	reward.StableDwell_ = instantaneousSafe
		? Config_.StableDwellReward_ : 0.0;
	if (State_.EndReason_ == OaLunarEndReason::SafeLanding) {
		reward.Terminal_ = Config_.SuccessReward_;
	} else if (State_.Terminated_) {
		reward.Terminal_ = Config_.FailurePenalty_;
	}
	reward.Total_ = reward.Sum();
	State_.EpisodeReturn_ += reward.Total_;

	transition.Valid_ = true;
	transition.Error_ = physics.Valid_ ? std::string{} : physics.Error_;
	transition.Observation_ = Observation();
	transition.Reward_ = reward.Total_;
	transition.Terminated_ = State_.Terminated_;
	transition.Truncated_ = State_.Truncated_;
	transition.EndReason_ = State_.EndReason_;
	transition.RewardTerms_ = reward;
	transition.Contact_ = physics.Contact_;
	return transition;
}
