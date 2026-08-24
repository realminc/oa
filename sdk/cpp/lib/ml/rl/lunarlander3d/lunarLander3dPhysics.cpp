#include <ml/rl/lunarlander3d/lunarLander3dPhysics.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace oa {

static bool lunarFinitePositive(double inValue) noexcept {
	return std::isfinite(inValue) and inValue > 0.0;
}

static std::uint64_t lunarFingerprintAdd(
	std::uint64_t inFingerprint,
	std::uint64_t inValue) noexcept {
	inFingerprint ^= inValue + 0x9e3779b97f4a7c15ULL
		+ (inFingerprint << 6U) + (inFingerprint >> 2U);
	inFingerprint ^= inFingerprint >> 30U;
	inFingerprint *= 0xbf58476d1ce4e5b9ULL;
	inFingerprint ^= inFingerprint >> 27U;
	inFingerprint *= 0x94d049bb133111ebULL;
	return inFingerprint ^ (inFingerprint >> 31U);
}

static std::uint64_t lunarFingerprintDouble(double inValue) noexcept {
	const double canonical = inValue == 0.0 ? 0.0 : inValue;
	return std::bit_cast<std::uint64_t>(canonical);
}

static bool lunarFiniteNonNegative(double inValue) noexcept {
	return std::isfinite(inValue) and inValue >= 0.0;
}

static bool lunarSupportIsValid(
	const LunarSupportSphere& inSupport) noexcept {
	return inSupport.bodyOffset_.isFinite()
		and lunarFinitePositive(inSupport.radius_);
}

static double lunarClampUnit(double inValue) noexcept {
	return std::clamp(inValue, -1.0, 1.0);
}

static float lunarNormalizedFloat(
	double inValue,
	double inScale) noexcept {
	if (not std::isfinite(inValue) or not lunarFinitePositive(inScale)) {
		return 0.0F;
	}
	return static_cast<float>(lunarClampUnit(inValue / inScale));
}

static oa::vlm::DVec3 lunarComponentDivide(
	const oa::vlm::DVec3& inNumerator,
	const oa::vlm::DVec3& inDenominator) noexcept {
	return {
		inNumerator.x / inDenominator.x,
		inNumerator.y / inDenominator.y,
		inNumerator.z / inDenominator.z,
	};
}

static oa::vlm::DVec3 lunarComponentMultiply(
	const oa::vlm::DVec3& inLeft,
	const oa::vlm::DVec3& inRight) noexcept {
	return {
		inLeft.x * inRight.x,
		inLeft.y * inRight.y,
		inLeft.z * inRight.z,
	};
}

std::string LunarLander3dConfig::validationError() const {
	if (environmentVersion_ != kLunarEnvironmentVersion
		or physicsVersion_ != kLunarPhysicsVersion
		or observationVersion_ != kLunarObservationVersion
		or rewardVersion_ != kLunarRewardVersion) {
		return "unsupported lunar lander contract version";
	}
	const std::string terrainError = terrain_.validationError();
	if (not terrainError.empty()) {
		return terrainError;
	}
	if (not lunarFinitePositive(policyTimeStep_)
		or policyTimeStep_ > 0.1) {
		return "lunar policy time step must be finite and in (0, 0.1]";
	}
	if (physicsSubsteps_ == 0U or physicsSubsteps_ > 64U) {
		return "lunar physics substeps must be in [1, 64]";
	}
	if (contactIterations_ == 0U or contactIterations_ > 16U) {
		return "lunar contact iterations must be in [1, 16]";
	}
	if (not lunarFiniteNonNegative(gravity_)
		or not lunarFinitePositive(mass_)
		or not lunarFinitePositive(diagonalInertia_.x)
		or not lunarFinitePositive(diagonalInertia_.y)
		or not lunarFinitePositive(diagonalInertia_.z)) {
		return "lunar gravity, mass, and diagonal inertia are invalid";
	}
	if (not lunarFiniteNonNegative(mainThrust_)
		or not lunarFiniteNonNegative(attitudeTorque_)
		or not lunarFinitePositive(fuelCapacity_)
		or not lunarFiniteNonNegative(mainFuelRate_)
		or not lunarFiniteNonNegative(attitudeFuelRate_)) {
		return "lunar actuator or fuel configuration is invalid";
	}
	if ((mainThrust_ > 0.0 and mainFuelRate_ <= 0.0)
		or (attitudeTorque_ > 0.0 and attitudeFuelRate_ <= 0.0)) {
		return "lunar enabled actuators require a positive fuel rate";
	}
	if (not lunarFiniteNonNegative(restitution_) or restitution_ > 1.0
		or not lunarFiniteNonNegative(friction_)
		or not lunarFiniteNonNegative(contactSlop_)
		or not lunarFiniteNonNegative(penetrationCorrectionFraction_)
		or penetrationCorrectionFraction_ > 1.0
		or not lunarFiniteNonNegative(maxPositionCorrectionPerContact_)
		or not lunarFinitePositive(maxContactImpulse_)
		or not lunarFiniteNonNegative(maxBiasSpeed_)) {
		return "lunar contact parameters are invalid";
	}
	if (not std::isfinite(taskMinimumY_) or not std::isfinite(taskMaximumY_)
		or taskMinimumY_ >= taskMaximumY_) {
		return "lunar task vertical bounds are invalid";
	}
	if (not lunarFiniteNonNegative(safeLinearSpeed_)
		or not lunarFiniteNonNegative(safeAngularSpeed_)
		or not lunarFiniteNonNegative(safeTiltRadians_)
		or safeTiltRadians_ > 1.5707963267948966
		or not lunarFiniteNonNegative(hardFootImpactSpeed_)
		or safeDwellSteps_ == 0U or maxEpisodeSteps_ == 0U) {
		return "lunar terminal thresholds are invalid";
	}
	if (not lunarFinitePositive(positionObservationScale_)
		or not lunarFinitePositive(velocityObservationScale_)
		or not lunarFinitePositive(angularVelocityObservationScale_)
		or not lunarFinitePositive(terrainClearanceObservationScale_)
		or not lunarFinitePositive(footClearanceObservationScale_)
		or not lunarFinitePositive(terrainProbeSpacing_)) {
		return "lunar observation scales are invalid";
	}
	if (not std::isfinite(rewardGamma_) or rewardGamma_ < 0.0
		or rewardGamma_ > 1.0
		or not lunarFiniteNonNegative(positionPotentialWeight_)
		or not lunarFiniteNonNegative(velocityPotentialWeight_)
		or not lunarFiniteNonNegative(tiltPotentialWeight_)
		or not lunarFiniteNonNegative(angularPotentialWeight_)
		or not lunarFiniteNonNegative(mainFuelCostWeight_)
		or not lunarFiniteNonNegative(attitudeFuelCostWeight_)
		or not lunarFiniteNonNegative(softFootContactReward_)
		or not lunarFiniteNonNegative(stableDwellReward_)
		or not std::isfinite(successReward_)
		or successReward_ < 0.0
		or not std::isfinite(failurePenalty_)
		or failurePenalty_ > 0.0) {
		return "lunar reward parameters are invalid";
	}
	for (const LunarSupportSphere& support : bodySupports_) {
		if (not lunarSupportIsValid(support)) {
			return "lunar body support sphere is invalid";
		}
	}
	for (const LunarSupportSphere& support : footSupports_) {
		if (not lunarSupportIsValid(support)) {
			return "lunar foot support sphere is invalid";
		}
	}
	return {};
}

std::uint64_t LunarLander3dConfig::contractFingerprint() const noexcept {
	std::uint64_t fingerprint = 0x4f414c554e434631ULL;
	auto addInteger = [&fingerprint](std::uint64_t inValue) {
		fingerprint = lunarFingerprintAdd(fingerprint, inValue);
	};
	auto addDouble = [&fingerprint](double inValue) {
		fingerprint = lunarFingerprintAdd(
			fingerprint, lunarFingerprintDouble(inValue));
	};
	addInteger(environmentVersion_);
	addInteger(physicsVersion_);
	addInteger(observationVersion_);
	addInteger(rewardVersion_);
	addInteger(kLunarTerrainVersion);
	addInteger(terrain_.cellsX_);
	addInteger(terrain_.cellsZ_);
	addDouble(terrain_.cellSize_);
	addDouble(terrain_.maxAbsHeight_);
	addDouble(terrain_.maxSlope_);
	addDouble(terrain_.padHalfExtent_);
	addDouble(terrain_.padTransitionWidth_);
	addDouble(policyTimeStep_);
	addInteger(physicsSubsteps_);
	addInteger(contactIterations_);
	addDouble(gravity_);
	addDouble(mass_);
	addDouble(diagonalInertia_.x);
	addDouble(diagonalInertia_.y);
	addDouble(diagonalInertia_.z);
	addDouble(mainThrust_);
	addDouble(attitudeTorque_);
	addDouble(fuelCapacity_);
	addDouble(mainFuelRate_);
	addDouble(attitudeFuelRate_);
	addDouble(restitution_);
	addDouble(friction_);
	addDouble(contactSlop_);
	addDouble(penetrationCorrectionFraction_);
	addDouble(maxPositionCorrectionPerContact_);
	addDouble(maxContactImpulse_);
	addDouble(maxBiasSpeed_);
	addDouble(taskMinimumY_);
	addDouble(taskMaximumY_);
	addDouble(safeLinearSpeed_);
	addDouble(safeAngularSpeed_);
	addDouble(safeTiltRadians_);
	addDouble(hardFootImpactSpeed_);
	addInteger(safeDwellSteps_);
	addInteger(maxEpisodeSteps_);
	addDouble(positionObservationScale_);
	addDouble(velocityObservationScale_);
	addDouble(angularVelocityObservationScale_);
	addDouble(terrainClearanceObservationScale_);
	addDouble(footClearanceObservationScale_);
	addDouble(terrainProbeSpacing_);
	addDouble(rewardGamma_);
	addDouble(positionPotentialWeight_);
	addDouble(velocityPotentialWeight_);
	addDouble(tiltPotentialWeight_);
	addDouble(angularPotentialWeight_);
	addDouble(mainFuelCostWeight_);
	addDouble(attitudeFuelCostWeight_);
	addDouble(softFootContactReward_);
	addDouble(stableDwellReward_);
	addDouble(successReward_);
	addDouble(failurePenalty_);
	for (const LunarSupportSphere& support : bodySupports_) {
		addDouble(support.bodyOffset_.x);
		addDouble(support.bodyOffset_.y);
		addDouble(support.bodyOffset_.z);
		addDouble(support.radius_);
	}
	for (const LunarSupportSphere& support : footSupports_) {
		addDouble(support.bodyOffset_.x);
		addDouble(support.bodyOffset_.y);
		addDouble(support.bodyOffset_.z);
		addDouble(support.radius_);
	}
	return fingerprint;
}

bool LunarLander3dState::isFinite() const noexcept {
	const double orientationNormSquared = orientation_.normSquared();
	if (not position_.isFinite() or not linearVelocity_.isFinite()
		or not orientation_.isFinite()
		or not std::isfinite(orientationNormSquared)
		or orientationNormSquared <= 1.0e-20
		or not angularVelocityBody_.isFinite()
		or not std::isfinite(fuel_) or not std::isfinite(mainThrottle_)
		or not attitudeCommandBody_.isFinite()
		or not std::isfinite(episodeReturn_)) {
		return false;
	}
	for (const double impulse : bodyContactImpulses_) {
		if (not std::isfinite(impulse)) return false;
	}
	for (const double impulse : footContactImpulses_) {
		if (not std::isfinite(impulse)) return false;
	}
	return true;
}

bool LunarContactDiagnostics::isFinite() const noexcept {
	return std::isfinite(maximumPenetration_)
		and std::isfinite(maximumNormalImpulse_)
		and std::isfinite(maximumFrictionImpulse_)
		and std::isfinite(maximumFootClosingSpeed_)
		and std::isfinite(totalPositionCorrection_);
}

bool LunarRewardTerms::isFinite() const noexcept {
	return std::isfinite(potentialBefore_) and std::isfinite(potentialAfter_)
		and std::isfinite(shaping_) and std::isfinite(mainFuelCost_)
		and std::isfinite(attitudeFuelCost_) and std::isfinite(softFootContact_)
		and std::isfinite(stableDwell_) and std::isfinite(terminal_)
		and std::isfinite(total_);
}

double LunarRewardTerms::sum() const noexcept {
	return shaping_ + mainFuelCost_ + attitudeFuelCost_
		+ softFootContact_ + stableDwell_ + terminal_;
}

oa::vlm::DVec3 FnLunarLander::supportWorldCenter(
	const LunarLander3dState& inState,
	const LunarSupportSphere& inSupport) noexcept {
	return inState.position_ + inState.orientation_.rotate(inSupport.bodyOffset_);
}

static double lunarEffectiveInverseMass(
	const LunarLander3dConfig& inConfig,
	const LunarLander3dState& inState,
	const oa::vlm::DVec3& inLeverWorld,
	const oa::vlm::DVec3& inDirectionWorld) noexcept {
	const oa::vlm::DVec3 rotationalWorld = oa::vlm::cross(
		inLeverWorld, inDirectionWorld);
	const oa::vlm::DVec3 rotationalBody = inState.orientation_.inverseRotate(
		rotationalWorld);
	const oa::vlm::DVec3 inverseInertiaApplied = lunarComponentDivide(
		rotationalBody, inConfig.diagonalInertia_);
	return 1.0 / inConfig.mass_
		+ oa::vlm::dot(rotationalBody, inverseInertiaApplied);
}

static void lunarApplyImpulse(
	const LunarLander3dConfig& inConfig,
	LunarLander3dState& inOutState,
	const oa::vlm::DVec3& inLeverWorld,
	const oa::vlm::DVec3& inImpulseWorld) noexcept {
	inOutState.linearVelocity_ += inImpulseWorld / inConfig.mass_;
	const oa::vlm::DVec3 angularImpulseWorld = oa::vlm::cross(
		inLeverWorld, inImpulseWorld);
	const oa::vlm::DVec3 angularImpulseBody =
		inOutState.orientation_.inverseRotate(angularImpulseWorld);
	inOutState.angularVelocityBody_ += lunarComponentDivide(
		angularImpulseBody, inConfig.diagonalInertia_);
}

static void lunarResolveSupport(
	const LunarLander3dConfig& inConfig,
	const LunarTerrain& inTerrain,
	double inSubstepTime,
	const LunarSupportSphere& inSupport,
	bool inIsFoot,
	std::size_t inSupportIndex,
	LunarLander3dState& inOutState,
	LunarContactDiagnostics& inOutDiagnostics) noexcept {
	const oa::vlm::DVec3 supportCenter = FnLunarLander::supportWorldCenter(
		inOutState, inSupport);
	const LunarTerrainSample terrain = inTerrain.query(
		supportCenter.x, supportCenter.z);
	if (not terrain.inBounds_) return;
	const double separation = supportCenter.y - inSupport.radius_
		- terrain.height_;
	if (separation >= 0.0) return;

	const double penetration = -separation;
	inOutDiagnostics.maximumPenetration_ = std::max(
		inOutDiagnostics.maximumPenetration_, penetration);
	++inOutDiagnostics.contactCount_;
	const oa::vlm::DVec3 leverWorld = supportCenter - inOutState.position_;
	const oa::vlm::DVec3 angularVelocityWorld =
		inOutState.orientation_.rotate(inOutState.angularVelocityBody_);
	oa::vlm::DVec3 pointVelocity = inOutState.linearVelocity_
		+ oa::vlm::cross(angularVelocityWorld, leverWorld);
	const double normalVelocity = oa::vlm::dot(pointVelocity, terrain.normal_);
	const double closingSpeed = std::max(0.0, -normalVelocity);
	if (inIsFoot) {
		inOutDiagnostics.footContactOccurred_ = true;
		inOutDiagnostics.maximumFootClosingSpeed_ = std::max(
			inOutDiagnostics.maximumFootClosingSpeed_, closingSpeed);
	}
	const double biasSpeed = std::min(
		inConfig.maxBiasSpeed_,
		penetration * inConfig.penetrationCorrectionFraction_ / inSubstepTime);
	const double targetDeltaSpeed = std::max(
		0.0, -(1.0 + inConfig.restitution_) * normalVelocity + biasSpeed);
	const double normalInverseMass = lunarEffectiveInverseMass(
		inConfig, inOutState, leverWorld, terrain.normal_);
	double normalImpulse = targetDeltaSpeed / normalInverseMass;
	normalImpulse = std::clamp(
		normalImpulse, 0.0, inConfig.maxContactImpulse_);
	lunarApplyImpulse(
		inConfig, inOutState, leverWorld, terrain.normal_ * normalImpulse);
	inOutDiagnostics.maximumNormalImpulse_ = std::max(
		inOutDiagnostics.maximumNormalImpulse_, normalImpulse);

	const oa::vlm::DVec3 updatedAngularVelocityWorld =
		inOutState.orientation_.rotate(inOutState.angularVelocityBody_);
	pointVelocity = inOutState.linearVelocity_
		+ oa::vlm::cross(updatedAngularVelocityWorld, leverWorld);
	const oa::vlm::DVec3 tangentVelocity = pointVelocity
		- terrain.normal_ * oa::vlm::dot(pointVelocity, terrain.normal_);
	const double tangentSpeed = tangentVelocity.length();
	double frictionImpulse = 0.0;
	if (tangentSpeed > 1.0e-12) {
		const oa::vlm::DVec3 tangentDirection = -tangentVelocity / tangentSpeed;
		const double tangentInverseMass = lunarEffectiveInverseMass(
			inConfig, inOutState, leverWorld, tangentDirection);
		frictionImpulse = std::min(
			tangentSpeed / tangentInverseMass,
			inConfig.friction_ * normalImpulse);
		frictionImpulse = std::clamp(
			frictionImpulse, 0.0, inConfig.maxContactImpulse_);
		lunarApplyImpulse(
			inConfig, inOutState, leverWorld,
			tangentDirection * frictionImpulse);
	}
	inOutDiagnostics.maximumFrictionImpulse_ = std::max(
		inOutDiagnostics.maximumFrictionImpulse_, frictionImpulse);

	const double correction = std::min(
		inConfig.maxPositionCorrectionPerContact_,
		penetration * inConfig.penetrationCorrectionFraction_);
	inOutState.position_ += terrain.normal_ * correction;
	inOutDiagnostics.totalPositionCorrection_ += correction;
	if (inIsFoot) {
		inOutState.footContacts_[inSupportIndex] = true;
		inOutState.footContactImpulses_[inSupportIndex] += normalImpulse;
	} else {
		inOutDiagnostics.bodyContactOccurred_ = true;
		inOutState.bodyContacts_[inSupportIndex] = true;
		inOutState.bodyContactImpulses_[inSupportIndex] += normalImpulse;
	}
}

static void lunarRefreshSupportContacts(
	const LunarLander3dConfig& inConfig,
	const LunarTerrain& inTerrain,
	LunarLander3dState& inOutState) noexcept {
	for (std::size_t index = 0U; index < inConfig.bodySupports_.size(); ++index) {
		const LunarSupportSphere& support = inConfig.bodySupports_[index];
		const oa::vlm::DVec3 center = FnLunarLander::supportWorldCenter(
			inOutState, support);
		const LunarTerrainSample terrain = inTerrain.query(center.x, center.z);
		inOutState.bodyContacts_[index] = terrain.inBounds_
			and center.y - support.radius_ - terrain.height_
				<= inConfig.contactSlop_;
	}
	for (std::size_t index = 0U; index < inConfig.footSupports_.size(); ++index) {
		const LunarSupportSphere& support = inConfig.footSupports_[index];
		const oa::vlm::DVec3 center = FnLunarLander::supportWorldCenter(
			inOutState, support);
		const LunarTerrainSample terrain = inTerrain.query(center.x, center.z);
		inOutState.footContacts_[index] = terrain.inBounds_
			and center.y - support.radius_ - terrain.height_
				<= inConfig.contactSlop_;
		inOutState.feetOnPad_[index] = terrain.inBounds_
			and inTerrain.isOnPad(center.x, center.z);
	}
}

static oa::vlm::DVec3 lunarActionTorque(
	LunarAction inAction,
	double inMagnitude) noexcept {
	switch (inAction) {
		case LunarAction::PitchPositive: return {inMagnitude, 0.0, 0.0};
		case LunarAction::PitchNegative: return {-inMagnitude, 0.0, 0.0};
		case LunarAction::RollPositive: return {0.0, 0.0, inMagnitude};
		case LunarAction::RollNegative: return {0.0, 0.0, -inMagnitude};
		case LunarAction::YawPositive: return {0.0, inMagnitude, 0.0};
		case LunarAction::YawNegative: return {0.0, -inMagnitude, 0.0};
		case LunarAction::Coast:
		case LunarAction::MainEngine: return {};
	}
	return {};
}

LunarPhysicsResult FnLunarLander::integrate(
	const LunarLander3dConfig& inConfig,
	const LunarTerrain& inTerrain,
	LunarAction inAction,
	LunarLander3dState& inOutState) {
	LunarPhysicsResult result;
	const std::string configError = inConfig.validationError();
	if (not configError.empty()) {
		result.valid_ = false;
		result.error_ = configError;
		return result;
	}
	if (not inTerrain.isValid()) {
		result.valid_ = false;
		result.error_ = inTerrain.error();
		return result;
	}
	if (inTerrain.config() != inConfig.terrain_) {
		result.valid_ = false;
		result.error_ =
			"lunar integration terrain does not match the versioned configuration";
		return result;
	}
	if (not lunarActionIsValid(static_cast<std::uint32_t>(inAction))) {
		result.valid_ = false;
		result.error_ = "lunar integration action is outside discrete(8)";
		return result;
	}
	if (not inOutState.isFinite() or inOutState.fuel_ < 0.0
		or inOutState.fuel_ > inConfig.fuelCapacity_) {
		result.valid_ = false;
		result.error_ = "lunar integration received an invalid state";
		return result;
	}

	inOutState.lastAction_ = inAction;
	inOutState.mainThrottle_ = 0.0;
	inOutState.attitudeCommandBody_ = {};
	inOutState.bodyContacts_.fill(false);
	inOutState.bodyContactImpulses_.fill(0.0);
	inOutState.footContacts_.fill(false);
	inOutState.feetOnPad_.fill(false);
	inOutState.footContactImpulses_.fill(0.0);
	const double substepTime = inConfig.policyTimeStep_
		/ static_cast<double>(inConfig.physicsSubsteps_);

	for (std::uint32_t substep = 0U;
		substep < inConfig.physicsSubsteps_;
		++substep) {
		double mainActivation = 0.0;
		oa::vlm::DVec3 attitudeTorque;
		if (inAction == LunarAction::MainEngine) {
			const double requestedFuel = inConfig.mainFuelRate_ * substepTime;
			mainActivation = requestedFuel > 0.0
				? std::min(1.0, inOutState.fuel_ / requestedFuel)
				: 1.0;
			const double fuelUsed = requestedFuel * mainActivation;
			inOutState.fuel_ -= fuelUsed;
			result.mainFuelUsed_ += fuelUsed;
		} else if (inAction != LunarAction::Coast) {
			const double requestedFuel = inConfig.attitudeFuelRate_ * substepTime;
			const double activation = requestedFuel > 0.0
				? std::min(1.0, inOutState.fuel_ / requestedFuel)
				: 1.0;
			const double fuelUsed = requestedFuel * activation;
			inOutState.fuel_ -= fuelUsed;
			result.attitudeFuelUsed_ += fuelUsed;
			attitudeTorque = lunarActionTorque(
				inAction, inConfig.attitudeTorque_ * activation);
		}
		inOutState.fuel_ = std::max(0.0, inOutState.fuel_);
		inOutState.mainThrottle_ = mainActivation;
		inOutState.attitudeCommandBody_ = attitudeTorque;

		const oa::vlm::DVec3 thrustWorld = inOutState.orientation_.rotate(
			oa::vlm::DVec3(0.0, inConfig.mainThrust_ * mainActivation, 0.0));
		const oa::vlm::DVec3 linearAcceleration = thrustWorld / inConfig.mass_
			+ oa::vlm::DVec3(0.0, -inConfig.gravity_, 0.0);
		inOutState.linearVelocity_ += linearAcceleration * substepTime;
		inOutState.position_ += inOutState.linearVelocity_ * substepTime;

		const oa::vlm::DVec3 angularMomentum = lunarComponentMultiply(
			inConfig.diagonalInertia_, inOutState.angularVelocityBody_);
		const oa::vlm::DVec3 gyroscopicTorque = oa::vlm::cross(
			inOutState.angularVelocityBody_, angularMomentum);
		const oa::vlm::DVec3 angularAcceleration = lunarComponentDivide(
			attitudeTorque - gyroscopicTorque, inConfig.diagonalInertia_);
		inOutState.angularVelocityBody_ += angularAcceleration * substepTime;
		const oa::vlm::DQuat angularQuaternion{
			inOutState.angularVelocityBody_.x,
			inOutState.angularVelocityBody_.y,
			inOutState.angularVelocityBody_.z,
			0.0};
		const oa::vlm::DQuat derivative =
			(inOutState.orientation_ * angularQuaternion) * 0.5;
		inOutState.orientation_ =
			(inOutState.orientation_ + derivative * substepTime).normalized();

		if (not inOutState.isFinite()) {
			result.valid_ = false;
			result.error_ = "lunar integration produced a non-finite state";
			return result;
		}

		for (std::uint32_t iteration = 0U;
			iteration < inConfig.contactIterations_;
			++iteration) {
			for (std::size_t supportIndex = 0U;
				supportIndex < inConfig.bodySupports_.size();
				++supportIndex) {
				lunarResolveSupport(
					inConfig, inTerrain, substepTime,
					inConfig.bodySupports_[supportIndex], false,
					supportIndex, inOutState, result.contact_);
			}
			for (std::size_t supportIndex = 0U;
				supportIndex < inConfig.footSupports_.size();
				++supportIndex) {
				lunarResolveSupport(
					inConfig, inTerrain, substepTime,
					inConfig.footSupports_[supportIndex], true,
					supportIndex, inOutState, result.contact_);
			}
		}
		if (not inOutState.isFinite() or not result.contact_.isFinite()) {
			result.valid_ = false;
			result.error_ = "lunar contact resolution produced a non-finite state";
			return result;
		}
	}
	lunarRefreshSupportContacts(inConfig, inTerrain, inOutState);
	const double maximumCorrection = static_cast<double>(
		inConfig.physicsSubsteps_ * inConfig.contactIterations_)
		* static_cast<double>(
			inConfig.bodySupports_.size() + inConfig.footSupports_.size())
		* inConfig.maxPositionCorrectionPerContact_;
	result.contact_.bounded_ = result.contact_.maximumNormalImpulse_
		<= inConfig.maxContactImpulse_
		and result.contact_.maximumFrictionImpulse_
		<= inConfig.maxContactImpulse_
		and result.contact_.totalPositionCorrection_
		<= maximumCorrection + 1.0e-12;
	return result;
}

std::array<float, kLunarObservationSize>
FnLunarLander::observe(
	const LunarLander3dConfig& inConfig,
	const LunarTerrain& inTerrain,
	const LunarLander3dState& inState) noexcept {
	std::array<float, kLunarObservationSize> observation{};
	if (not inState.isFinite() or not inTerrain.isValid()) {
		return observation;
	}
	std::size_t offset = 0U;
	observation[offset++] = lunarNormalizedFloat(
		inState.position_.x, inConfig.positionObservationScale_);
	observation[offset++] = lunarNormalizedFloat(
		inState.position_.y, inConfig.positionObservationScale_);
	observation[offset++] = lunarNormalizedFloat(
		inState.position_.z, inConfig.positionObservationScale_);
	observation[offset++] = lunarNormalizedFloat(
		inState.linearVelocity_.x, inConfig.velocityObservationScale_);
	observation[offset++] = lunarNormalizedFloat(
		inState.linearVelocity_.y, inConfig.velocityObservationScale_);
	observation[offset++] = lunarNormalizedFloat(
		inState.linearVelocity_.z, inConfig.velocityObservationScale_);

	const oa::vlm::DVec3 bodyUp = inState.orientation_.rotate({0.0, 1.0, 0.0});
	const oa::vlm::DVec3 bodyForward = inState.orientation_.rotate({0.0, 0.0, -1.0});
	observation[offset++] = static_cast<float>(lunarClampUnit(bodyUp.x));
	observation[offset++] = static_cast<float>(lunarClampUnit(bodyUp.y));
	observation[offset++] = static_cast<float>(lunarClampUnit(bodyUp.z));
	observation[offset++] = static_cast<float>(lunarClampUnit(bodyForward.x));
	observation[offset++] = static_cast<float>(lunarClampUnit(bodyForward.y));
	observation[offset++] = static_cast<float>(lunarClampUnit(bodyForward.z));
	observation[offset++] = lunarNormalizedFloat(
		inState.angularVelocityBody_.x,
		inConfig.angularVelocityObservationScale_);
	observation[offset++] = lunarNormalizedFloat(
		inState.angularVelocityBody_.y,
		inConfig.angularVelocityObservationScale_);
	observation[offset++] = lunarNormalizedFloat(
		inState.angularVelocityBody_.z,
		inConfig.angularVelocityObservationScale_);

	for (std::int32_t probeZ = -1; probeZ <= 1; ++probeZ) {
		for (std::int32_t probeX = -1; probeX <= 1; ++probeX) {
			const double positionX = inState.position_.x
				+ static_cast<double>(probeX) * inConfig.terrainProbeSpacing_;
			const double positionZ = inState.position_.z
				+ static_cast<double>(probeZ) * inConfig.terrainProbeSpacing_;
			const LunarTerrainSample terrain = inTerrain.query(
				positionX, positionZ);
			const double clearance = terrain.inBounds_
				? inState.position_.y - terrain.height_
				: inConfig.terrainClearanceObservationScale_;
			observation[offset++] = lunarNormalizedFloat(
				clearance, inConfig.terrainClearanceObservationScale_);
		}
	}
	for (std::size_t footIndex = 0U;
		footIndex < inConfig.footSupports_.size();
		++footIndex) {
		const LunarSupportSphere& support = inConfig.footSupports_[footIndex];
		const oa::vlm::DVec3 center = supportWorldCenter(
			inState, support);
		const LunarTerrainSample terrain = inTerrain.query(center.x, center.z);
		const double clearance = terrain.inBounds_
			? center.y - support.radius_ - terrain.height_
			: inConfig.footClearanceObservationScale_;
		observation[offset++] = lunarNormalizedFloat(
			clearance, inConfig.footClearanceObservationScale_);
	}
	for (const bool contact : inState.footContacts_) {
		observation[offset++] = contact ? 1.0F : 0.0F;
	}
	observation[offset++] = static_cast<float>(std::clamp(
		inState.fuel_ / inConfig.fuelCapacity_, 0.0, 1.0));
	return observation;
}

double FnLunarLander::potential(
	const LunarLander3dConfig& inConfig,
	const LunarLander3dState& inState) noexcept {
	if (not inState.isFinite()) return 0.0;
	const double positionCost = std::min(
		1.0, inState.position_.length() / inConfig.positionObservationScale_);
	const double velocityCost = std::min(
		1.0, inState.linearVelocity_.length()
			/ inConfig.velocityObservationScale_);
	const oa::vlm::DVec3 bodyUp = inState.orientation_.rotate({0.0, 1.0, 0.0});
	const double tiltCost = std::clamp((1.0 - bodyUp.y) * 0.5, 0.0, 1.0);
	const double angularCost = std::min(
		1.0, inState.angularVelocityBody_.length()
			/ inConfig.angularVelocityObservationScale_);
	return -(
		inConfig.positionPotentialWeight_ * positionCost
		+ inConfig.velocityPotentialWeight_ * velocityCost
		+ inConfig.tiltPotentialWeight_ * tiltCost
		+ inConfig.angularPotentialWeight_ * angularCost);
}

} // namespace oa
