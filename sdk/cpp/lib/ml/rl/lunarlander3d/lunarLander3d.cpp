#include <ml/rl/lunarlander3d/lunarLander3d.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace oa {

static bool lunarAllFootContacts(
	const LunarLander3dState& inState) noexcept {
	for (const bool contact : inState.footContacts_) {
		if (not contact) return false;
	}
	return true;
}

static bool lunarAllFeetOnPad(
	const LunarLander3dState& inState) noexcept {
	for (const bool onPad : inState.feetOnPad_) {
		if (not onPad) return false;
	}
	return true;
}

static bool lunarAnyBodyContact(
	const LunarLander3dState& inState) noexcept {
	for (const bool contact : inState.bodyContacts_) {
		if (contact) return true;
	}
	return false;
}

static bool lunarInstantaneouslySafe(
	const LunarLander3dConfig& inConfig,
	const LunarLander3dState& inState) noexcept {
	const oa::vlm::DVec3 bodyUp = inState.orientation_.rotate({0.0, 1.0, 0.0});
	const double tilt = std::acos(std::clamp(bodyUp.y, -1.0, 1.0));
	return lunarAllFootContacts(inState)
		and lunarAllFeetOnPad(inState)
		and not lunarAnyBodyContact(inState)
		and inState.linearVelocity_.length() <= inConfig.safeLinearSpeed_
		and inState.angularVelocityBody_.length() <= inConfig.safeAngularSpeed_
		and tilt <= inConfig.safeTiltRadians_;
}

LunarAction lunarScriptedLandingAction(
	const LunarLander3dConfig& inConfig,
	const LunarLander3dState& inState) noexcept {
	double uprightFootContactHeight = 0.0;
	for (const LunarSupportSphere& foot : inConfig.footSupports_) {
		uprightFootContactHeight = std::max(
			uprightFootContactHeight,
			foot.radius_ - foot.bodyOffset_.y);
	}
	const double footClearance = std::max(
		0.0, inState.position_.y - uprightFootContactHeight);
	const double desiredDescent = -std::clamp(
		0.48 * std::sqrt(footClearance), 0.24, 0.72);
	if (inState.linearVelocity_.y < desiredDescent) {
		return LunarAction::MainEngine;
	}

	const oa::vlm::DVec3 bodyUp = inState.orientation_.rotate({0.0, 1.0, 0.0});
	const double guidanceBlend = std::clamp(footClearance, 0.0, 1.0);
	constexpr double lateralPositionGain = 0.055;
	constexpr double lateralVelocityGain = 0.30;
	constexpr double maximumGuidanceTilt = 0.12;
	const double targetBodyUpX = guidanceBlend * std::clamp(
		-lateralPositionGain * inState.position_.x
			- lateralVelocityGain * inState.linearVelocity_.x,
		-maximumGuidanceTilt, maximumGuidanceTilt);
	const double targetBodyUpZ = guidanceBlend * std::clamp(
		-lateralPositionGain * inState.position_.z
			- lateralVelocityGain * inState.linearVelocity_.z,
		-maximumGuidanceTilt, maximumGuidanceTilt);
	constexpr double proportionalGain = 8.0;
	constexpr double dampingGain = 3.0;
	constexpr double commandDeadZone = 0.025;
	const double pitchCommand = -proportionalGain
		* (bodyUp.z - targetBodyUpZ)
		- dampingGain * inState.angularVelocityBody_.x;
	const double rollCommand = proportionalGain
		* (bodyUp.x - targetBodyUpX)
		- dampingGain * inState.angularVelocityBody_.z;
	if (std::max(std::abs(pitchCommand), std::abs(rollCommand))
		<= commandDeadZone) {
		return LunarAction::Coast;
	}
	if (std::abs(pitchCommand) >= std::abs(rollCommand)) {
		return pitchCommand > 0.0
			? LunarAction::PitchPositive
			: LunarAction::PitchNegative;
	}
	return rollCommand > 0.0
		? LunarAction::RollPositive
		: LunarAction::RollNegative;
}

LunarScalarEnvironment LunarScalarEnvironment::invalid_(
	const LunarLander3dConfig& inConfig,
	const LunarEpisodeManifest& inManifest,
	const LunarTerrain& inTerrain,
	std::string inError) {
	LunarScalarEnvironment environment;
	environment.config_ = inConfig;
	environment.manifest_ = inManifest;
	environment.terrain_ = inTerrain;
	environment.error_ = std::move(inError);
	return environment;
}

LunarScalarEnvironment LunarScalarEnvironment::createFlat(
	const LunarLander3dConfig& inConfig,
	const LunarEpisodeManifest& inManifest) {
	return createWithTerrain(
		inConfig, inManifest, LunarTerrain::createFlat(inConfig.terrain_));
}

LunarScalarEnvironment LunarScalarEnvironment::createSeeded(
	const LunarLander3dConfig& inConfig,
	const LunarEpisodeManifest& inManifest) {
	return createWithTerrain(
		inConfig, inManifest,
		LunarTerrain::createSeeded(inConfig.terrain_, inManifest));
}

LunarScalarEnvironment LunarScalarEnvironment::createWithTerrain(
	const LunarLander3dConfig& inConfig,
	const LunarEpisodeManifest& inManifest,
	const LunarTerrain& inTerrain) {
	const std::string configError = inConfig.validationError();
	if (not configError.empty()) {
		return invalid_(inConfig, inManifest, inTerrain, configError);
	}
	const std::string manifestError = inManifest.validationError();
	if (not manifestError.empty()) {
		return invalid_(inConfig, inManifest, inTerrain, manifestError);
	}
	if (inManifest.environmentVersion_ != inConfig.environmentVersion_
		or inManifest.physicsVersion_ != inConfig.physicsVersion_
		or inManifest.observationVersion_ != inConfig.observationVersion_
		or inManifest.rewardVersion_ != inConfig.rewardVersion_
		or inManifest.terrainVersion_ != kLunarTerrainVersion) {
		return invalid_(
			inConfig, inManifest, inTerrain,
			"lunar manifest versions do not match the environment configuration");
	}
	if (inManifest.configFingerprint_ != inConfig.contractFingerprint()) {
		return invalid_(
			inConfig, inManifest, inTerrain,
			"lunar manifest configuration fingerprint does not match");
	}
	if (not inTerrain.isValid()) {
		return invalid_(inConfig, inManifest, inTerrain, inTerrain.error());
	}
	if (inTerrain.config() != inConfig.terrain_) {
		return invalid_(
			inConfig, inManifest, inTerrain,
			"lunar terrain configuration does not match the environment");
	}

	LunarScalarEnvironment environment;
	environment.config_ = inConfig;
	environment.manifest_ = inManifest;
	environment.terrain_ = inTerrain;
	environment.error_.clear();
	if (not environment.reset()) {
		environment.error_ = "lunar deterministic reset produced an invalid state";
	}
	return environment;
}

LunarLander3dState LunarScalarEnvironment::spawnState_() const noexcept {
	LunarLander3dState state;
	const double padRange = config_.terrain_.padHalfExtent_ * 0.35;
	state.position_.x = (
		manifest_.sample01(LunarRandomPurpose::Spawn, 0U) * 2.0 - 1.0)
		* padRange;
	state.position_.y = 5.0
		+ manifest_.sample01(LunarRandomPurpose::Spawn, 1U) * 2.0;
	state.position_.z = (
		manifest_.sample01(LunarRandomPurpose::Spawn, 2U) * 2.0 - 1.0)
		* padRange;
	state.linearVelocity_.x = (
		manifest_.sample01(LunarRandomPurpose::Spawn, 3U) * 2.0 - 1.0)
		* 0.12;
	state.linearVelocity_.y = -0.1
		- manifest_.sample01(LunarRandomPurpose::Spawn, 4U) * 0.2;
	state.linearVelocity_.z = (
		manifest_.sample01(LunarRandomPurpose::Spawn, 5U) * 2.0 - 1.0)
		* 0.12;
	const double pitch = (
		manifest_.sample01(LunarRandomPurpose::Spawn, 6U) * 2.0 - 1.0)
		* 0.03;
	const double roll = (
		manifest_.sample01(LunarRandomPurpose::Spawn, 7U) * 2.0 - 1.0)
		* 0.03;
	const double yaw = (
		manifest_.sample01(LunarRandomPurpose::Spawn, 8U) * 2.0 - 1.0)
		* 0.08;
	state.orientation_ = (
		oa::vlm::DQuat::fromAxisAngle({0.0, 1.0, 0.0}, yaw)
		* oa::vlm::DQuat::fromAxisAngle({1.0, 0.0, 0.0}, pitch)
		* oa::vlm::DQuat::fromAxisAngle({0.0, 0.0, 1.0}, roll)).normalized();
	state.fuel_ = config_.fuelCapacity_;
	return state;
}

bool LunarScalarEnvironment::reset() noexcept {
	if (not error_.empty()) return false;
	state_ = spawnState_();
	return state_.isFinite();
}

bool LunarScalarEnvironment::setState(
	const LunarLander3dState& inState) noexcept {
	if (not isValid() or not inState.isFinite()
		or inState.fuel_ < 0.0 or inState.fuel_ > config_.fuelCapacity_
		or (inState.terminated_ and inState.truncated_)
		or ((inState.terminated_ or inState.truncated_)
			and inState.endReason_ == LunarEndReason::None)
		or (not inState.terminated_ and not inState.truncated_
			and inState.endReason_ != LunarEndReason::None)) {
		return false;
	}
	state_ = inState;
	state_.orientation_ = state_.orientation_.normalized();
	return true;
}

std::array<float, kLunarObservationSize>
LunarScalarEnvironment::observation() const noexcept {
	return FnLunarLander::observe(config_, terrain_, state_);
}

LunarTransition LunarScalarEnvironment::step(
	std::uint32_t inAction,
	bool inExternalStop) {
	LunarTransition transition;
	if (not isValid()) {
		transition.error_ = error_;
		return transition;
	}
	if (state_.terminated_ or state_.truncated_) {
		transition.error_ = "lunar episode has ended; reset is required";
		return transition;
	}
	if (not lunarActionIsValid(inAction)) {
		// Batched device execution cannot reject one lane without rejecting the
		// whole submission. Invalid actions therefore consume one transition and
		// terminate only that lane with the configured failure penalty.
		++state_.episodeStep_;
		state_.terminated_ = true;
		state_.endReason_ = LunarEndReason::InvalidAction;
		state_.episodeReturn_ += config_.failurePenalty_;
		transition.valid_ = true;
		transition.observation_ = observation();
		transition.reward_ = config_.failurePenalty_;
		transition.terminated_ = true;
		transition.endReason_ = state_.endReason_;
		transition.rewardTerms_.terminal_ = config_.failurePenalty_;
		transition.rewardTerms_.total_ = config_.failurePenalty_;
		return transition;
	}
	if (inExternalStop) {
		state_.truncated_ = true;
		state_.endReason_ = LunarEndReason::ExternalStop;
		transition.valid_ = true;
		transition.observation_ = observation();
		transition.truncated_ = true;
		transition.endReason_ = state_.endReason_;
		return transition;
	}

	const double potentialBefore = FnLunarLander::potential(
		config_, state_);
	LunarPhysicsResult physics = FnLunarLander::integrate(
		config_, terrain_, static_cast<LunarAction>(inAction), state_);
	++state_.episodeStep_;
	bool instantaneousSafe = false;
	if (not physics.valid_) {
		state_.terminated_ = true;
		state_.endReason_ = LunarEndReason::NumericalFailure;
	} else if (not state_.isFinite()) {
		state_.terminated_ = true;
		state_.endReason_ = LunarEndReason::NumericalFailure;
	} else if (not terrain_.contains(state_.position_.x, state_.position_.z)
		or state_.position_.y < config_.taskMinimumY_
		or state_.position_.y > config_.taskMaximumY_) {
		state_.terminated_ = true;
		state_.endReason_ = LunarEndReason::OutOfBounds;
	} else if (physics.contact_.bodyContactOccurred_
		or lunarAnyBodyContact(state_)) {
		state_.terminated_ = true;
		state_.endReason_ = LunarEndReason::BodyImpact;
	} else if (physics.contact_.maximumFootClosingSpeed_
		> config_.hardFootImpactSpeed_) {
		state_.terminated_ = true;
		state_.endReason_ = LunarEndReason::HardFootImpact;
	} else {
		instantaneousSafe = lunarInstantaneouslySafe(config_, state_);
		state_.stableDwell_ = instantaneousSafe
			? state_.stableDwell_ + 1U : 0U;
		if (state_.stableDwell_ >= config_.safeDwellSteps_) {
			state_.terminated_ = true;
			state_.endReason_ = LunarEndReason::SafeLanding;
		}
	}
	if (not state_.terminated_
		and state_.episodeStep_ >= config_.maxEpisodeSteps_) {
		state_.truncated_ = true;
		state_.endReason_ = LunarEndReason::TimeLimit;
	}

	LunarRewardTerms reward;
	reward.potentialBefore_ = potentialBefore;
	reward.potentialAfter_ = FnLunarLander::potential(config_, state_);
	const double effectivePotentialAfter = state_.terminated_
		? 0.0 : reward.potentialAfter_;
	reward.shaping_ = config_.rewardGamma_ * effectivePotentialAfter
		- reward.potentialBefore_;
	reward.mainFuelCost_ = -config_.mainFuelCostWeight_ * physics.mainFuelUsed_;
	reward.attitudeFuelCost_ = -config_.attitudeFuelCostWeight_
		* physics.attitudeFuelUsed_;
	if (physics.contact_.maximumFootClosingSpeed_
		<= config_.hardFootImpactSpeed_) {
		for (std::size_t footIndex = 0U;
			footIndex < state_.footContacts_.size();
			++footIndex) {
			if (state_.footContacts_[footIndex]
				and not state_.footContactRewarded_[footIndex]) {
				reward.softFootContact_ += config_.softFootContactReward_;
				state_.footContactRewarded_[footIndex] = true;
			}
		}
	}
	reward.stableDwell_ = instantaneousSafe
		? config_.stableDwellReward_ : 0.0;
	if (state_.endReason_ == LunarEndReason::SafeLanding) {
		reward.terminal_ = config_.successReward_;
	} else if (state_.terminated_) {
		reward.terminal_ = config_.failurePenalty_;
	}
	reward.total_ = reward.sum();
	state_.episodeReturn_ += reward.total_;

	transition.valid_ = true;
	transition.error_ = physics.valid_ ? std::string{} : physics.error_;
	transition.observation_ = observation();
	transition.reward_ = reward.total_;
	transition.terminated_ = state_.terminated_;
	transition.truncated_ = state_.truncated_;
	transition.endReason_ = state_.endReason_;
	transition.rewardTerms_ = reward;
	transition.contact_ = physics.contact_;
	return transition;
}

} // namespace oa
