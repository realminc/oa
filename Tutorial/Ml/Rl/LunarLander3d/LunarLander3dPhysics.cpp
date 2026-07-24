#include "LunarLander3dPhysics.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

static bool OaLunarFinitePositive(double InValue) noexcept {
	return std::isfinite(InValue) and InValue > 0.0;
}

static std::uint64_t OaLunarFingerprintAdd(
	std::uint64_t InFingerprint,
	std::uint64_t InValue) noexcept {
	InFingerprint ^= InValue + 0x9e3779b97f4a7c15ULL
		+ (InFingerprint << 6U) + (InFingerprint >> 2U);
	InFingerprint ^= InFingerprint >> 30U;
	InFingerprint *= 0xbf58476d1ce4e5b9ULL;
	InFingerprint ^= InFingerprint >> 27U;
	InFingerprint *= 0x94d049bb133111ebULL;
	return InFingerprint ^ (InFingerprint >> 31U);
}

static std::uint64_t OaLunarFingerprintDouble(double InValue) noexcept {
	const double canonical = InValue == 0.0 ? 0.0 : InValue;
	return std::bit_cast<std::uint64_t>(canonical);
}

static bool OaLunarFiniteNonNegative(double InValue) noexcept {
	return std::isfinite(InValue) and InValue >= 0.0;
}

static bool OaLunarSupportIsValid(
	const OaLunarSupportSphere& InSupport) noexcept {
	return InSupport.BodyOffset_.IsFinite()
		and OaLunarFinitePositive(InSupport.Radius_);
}

static double OaLunarClampUnit(double InValue) noexcept {
	return std::clamp(InValue, -1.0, 1.0);
}

static float OaLunarNormalizedFloat(
	double InValue,
	double InScale) noexcept {
	if (not std::isfinite(InValue) or not OaLunarFinitePositive(InScale)) {
		return 0.0F;
	}
	return static_cast<float>(OaLunarClampUnit(InValue / InScale));
}

static OaLunarVec3 OaLunarComponentDivide(
	const OaLunarVec3& InNumerator,
	const OaLunarVec3& InDenominator) noexcept {
	return {
		InNumerator.ComponentX_ / InDenominator.ComponentX_,
		InNumerator.ComponentY_ / InDenominator.ComponentY_,
		InNumerator.ComponentZ_ / InDenominator.ComponentZ_,
	};
}

static OaLunarVec3 OaLunarComponentMultiply(
	const OaLunarVec3& InLeft,
	const OaLunarVec3& InRight) noexcept {
	return {
		InLeft.ComponentX_ * InRight.ComponentX_,
		InLeft.ComponentY_ * InRight.ComponentY_,
		InLeft.ComponentZ_ * InRight.ComponentZ_,
	};
}

std::string OaLunarLander3dConfig::ValidationError() const {
	if (EnvironmentVersion_ != OA_LUNAR_ENVIRONMENT_VERSION
		or PhysicsVersion_ != OA_LUNAR_PHYSICS_VERSION
		or ObservationVersion_ != OA_LUNAR_OBSERVATION_VERSION
		or RewardVersion_ != OA_LUNAR_REWARD_VERSION) {
		return "unsupported lunar lander contract version";
	}
	const std::string terrainError = Terrain_.ValidationError();
	if (not terrainError.empty()) {
		return terrainError;
	}
	if (not OaLunarFinitePositive(PolicyTimeStep_)
		or PolicyTimeStep_ > 0.1) {
		return "lunar policy time step must be finite and in (0, 0.1]";
	}
	if (PhysicsSubsteps_ == 0U or PhysicsSubsteps_ > 64U) {
		return "lunar physics substeps must be in [1, 64]";
	}
	if (ContactIterations_ == 0U or ContactIterations_ > 16U) {
		return "lunar contact iterations must be in [1, 16]";
	}
	if (not OaLunarFiniteNonNegative(Gravity_)
		or not OaLunarFinitePositive(Mass_)
		or not OaLunarFinitePositive(DiagonalInertia_.ComponentX_)
		or not OaLunarFinitePositive(DiagonalInertia_.ComponentY_)
		or not OaLunarFinitePositive(DiagonalInertia_.ComponentZ_)) {
		return "lunar gravity, mass, and diagonal inertia are invalid";
	}
	if (not OaLunarFiniteNonNegative(MainThrust_)
		or not OaLunarFiniteNonNegative(AttitudeTorque_)
		or not OaLunarFinitePositive(FuelCapacity_)
		or not OaLunarFiniteNonNegative(MainFuelRate_)
		or not OaLunarFiniteNonNegative(AttitudeFuelRate_)) {
		return "lunar actuator or fuel configuration is invalid";
	}
	if ((MainThrust_ > 0.0 and MainFuelRate_ <= 0.0)
		or (AttitudeTorque_ > 0.0 and AttitudeFuelRate_ <= 0.0)) {
		return "lunar enabled actuators require a positive fuel rate";
	}
	if (not OaLunarFiniteNonNegative(Restitution_) or Restitution_ > 1.0
		or not OaLunarFiniteNonNegative(Friction_)
		or not OaLunarFiniteNonNegative(ContactSlop_)
		or not OaLunarFiniteNonNegative(PenetrationCorrectionFraction_)
		or PenetrationCorrectionFraction_ > 1.0
		or not OaLunarFiniteNonNegative(MaxPositionCorrectionPerContact_)
		or not OaLunarFinitePositive(MaxContactImpulse_)
		or not OaLunarFiniteNonNegative(MaxBiasSpeed_)) {
		return "lunar contact parameters are invalid";
	}
	if (not std::isfinite(TaskMinimumY_) or not std::isfinite(TaskMaximumY_)
		or TaskMinimumY_ >= TaskMaximumY_) {
		return "lunar task vertical bounds are invalid";
	}
	if (not OaLunarFiniteNonNegative(SafeLinearSpeed_)
		or not OaLunarFiniteNonNegative(SafeAngularSpeed_)
		or not OaLunarFiniteNonNegative(SafeTiltRadians_)
		or SafeTiltRadians_ > 1.5707963267948966
		or not OaLunarFiniteNonNegative(HardFootImpactSpeed_)
		or SafeDwellSteps_ == 0U or MaxEpisodeSteps_ == 0U) {
		return "lunar terminal thresholds are invalid";
	}
	if (not OaLunarFinitePositive(PositionObservationScale_)
		or not OaLunarFinitePositive(VelocityObservationScale_)
		or not OaLunarFinitePositive(AngularVelocityObservationScale_)
		or not OaLunarFinitePositive(TerrainClearanceObservationScale_)
		or not OaLunarFinitePositive(FootClearanceObservationScale_)
		or not OaLunarFinitePositive(TerrainProbeSpacing_)) {
		return "lunar observation scales are invalid";
	}
	if (not std::isfinite(RewardGamma_) or RewardGamma_ < 0.0
		or RewardGamma_ > 1.0
		or not OaLunarFiniteNonNegative(PositionPotentialWeight_)
		or not OaLunarFiniteNonNegative(VelocityPotentialWeight_)
		or not OaLunarFiniteNonNegative(TiltPotentialWeight_)
		or not OaLunarFiniteNonNegative(AngularPotentialWeight_)
		or not OaLunarFiniteNonNegative(MainFuelCostWeight_)
		or not OaLunarFiniteNonNegative(AttitudeFuelCostWeight_)
		or not OaLunarFiniteNonNegative(SoftFootContactReward_)
		or not OaLunarFiniteNonNegative(StableDwellReward_)
		or not std::isfinite(SuccessReward_)
		or SuccessReward_ < 0.0
		or not std::isfinite(FailurePenalty_)
		or FailurePenalty_ > 0.0) {
		return "lunar reward parameters are invalid";
	}
	for (const OaLunarSupportSphere& support : BodySupports_) {
		if (not OaLunarSupportIsValid(support)) {
			return "lunar body support sphere is invalid";
		}
	}
	for (const OaLunarSupportSphere& support : FootSupports_) {
		if (not OaLunarSupportIsValid(support)) {
			return "lunar foot support sphere is invalid";
		}
	}
	return {};
}

std::uint64_t OaLunarLander3dConfig::ContractFingerprint() const noexcept {
	std::uint64_t fingerprint = 0x4f414c554e434631ULL;
	auto addInteger = [&fingerprint](std::uint64_t InValue) {
		fingerprint = OaLunarFingerprintAdd(fingerprint, InValue);
	};
	auto addDouble = [&fingerprint](double InValue) {
		fingerprint = OaLunarFingerprintAdd(
			fingerprint, OaLunarFingerprintDouble(InValue));
	};
	addInteger(EnvironmentVersion_);
	addInteger(PhysicsVersion_);
	addInteger(ObservationVersion_);
	addInteger(RewardVersion_);
	addInteger(OA_LUNAR_TERRAIN_VERSION);
	addInteger(Terrain_.CellsX_);
	addInteger(Terrain_.CellsZ_);
	addDouble(Terrain_.CellSize_);
	addDouble(Terrain_.MaxAbsHeight_);
	addDouble(Terrain_.MaxSlope_);
	addDouble(Terrain_.PadHalfExtent_);
	addDouble(Terrain_.PadTransitionWidth_);
	addDouble(PolicyTimeStep_);
	addInteger(PhysicsSubsteps_);
	addInteger(ContactIterations_);
	addDouble(Gravity_);
	addDouble(Mass_);
	addDouble(DiagonalInertia_.ComponentX_);
	addDouble(DiagonalInertia_.ComponentY_);
	addDouble(DiagonalInertia_.ComponentZ_);
	addDouble(MainThrust_);
	addDouble(AttitudeTorque_);
	addDouble(FuelCapacity_);
	addDouble(MainFuelRate_);
	addDouble(AttitudeFuelRate_);
	addDouble(Restitution_);
	addDouble(Friction_);
	addDouble(ContactSlop_);
	addDouble(PenetrationCorrectionFraction_);
	addDouble(MaxPositionCorrectionPerContact_);
	addDouble(MaxContactImpulse_);
	addDouble(MaxBiasSpeed_);
	addDouble(TaskMinimumY_);
	addDouble(TaskMaximumY_);
	addDouble(SafeLinearSpeed_);
	addDouble(SafeAngularSpeed_);
	addDouble(SafeTiltRadians_);
	addDouble(HardFootImpactSpeed_);
	addInteger(SafeDwellSteps_);
	addInteger(MaxEpisodeSteps_);
	addDouble(PositionObservationScale_);
	addDouble(VelocityObservationScale_);
	addDouble(AngularVelocityObservationScale_);
	addDouble(TerrainClearanceObservationScale_);
	addDouble(FootClearanceObservationScale_);
	addDouble(TerrainProbeSpacing_);
	addDouble(RewardGamma_);
	addDouble(PositionPotentialWeight_);
	addDouble(VelocityPotentialWeight_);
	addDouble(TiltPotentialWeight_);
	addDouble(AngularPotentialWeight_);
	addDouble(MainFuelCostWeight_);
	addDouble(AttitudeFuelCostWeight_);
	addDouble(SoftFootContactReward_);
	addDouble(StableDwellReward_);
	addDouble(SuccessReward_);
	addDouble(FailurePenalty_);
	for (const OaLunarSupportSphere& support : BodySupports_) {
		addDouble(support.BodyOffset_.ComponentX_);
		addDouble(support.BodyOffset_.ComponentY_);
		addDouble(support.BodyOffset_.ComponentZ_);
		addDouble(support.Radius_);
	}
	for (const OaLunarSupportSphere& support : FootSupports_) {
		addDouble(support.BodyOffset_.ComponentX_);
		addDouble(support.BodyOffset_.ComponentY_);
		addDouble(support.BodyOffset_.ComponentZ_);
		addDouble(support.Radius_);
	}
	return fingerprint;
}

bool OaLunarLander3dState::IsFinite() const noexcept {
	const double orientationNormSquared = Orientation_.NormSquared();
	if (not Position_.IsFinite() or not LinearVelocity_.IsFinite()
		or not Orientation_.IsFinite()
		or not std::isfinite(orientationNormSquared)
		or orientationNormSquared <= 1.0e-20
		or not AngularVelocityBody_.IsFinite()
		or not std::isfinite(Fuel_) or not std::isfinite(MainThrottle_)
		or not AttitudeCommandBody_.IsFinite()
		or not std::isfinite(EpisodeReturn_)) {
		return false;
	}
	for (const double impulse : BodyContactImpulses_) {
		if (not std::isfinite(impulse)) return false;
	}
	for (const double impulse : FootContactImpulses_) {
		if (not std::isfinite(impulse)) return false;
	}
	return true;
}

bool OaLunarContactDiagnostics::IsFinite() const noexcept {
	return std::isfinite(MaximumPenetration_)
		and std::isfinite(MaximumNormalImpulse_)
		and std::isfinite(MaximumFrictionImpulse_)
		and std::isfinite(MaximumFootClosingSpeed_)
		and std::isfinite(TotalPositionCorrection_);
}

bool OaLunarRewardTerms::IsFinite() const noexcept {
	return std::isfinite(PotentialBefore_) and std::isfinite(PotentialAfter_)
		and std::isfinite(Shaping_) and std::isfinite(MainFuelCost_)
		and std::isfinite(AttitudeFuelCost_) and std::isfinite(SoftFootContact_)
		and std::isfinite(StableDwell_) and std::isfinite(Terminal_)
		and std::isfinite(Total_);
}

double OaLunarRewardTerms::Sum() const noexcept {
	return Shaping_ + MainFuelCost_ + AttitudeFuelCost_
		+ SoftFootContact_ + StableDwell_ + Terminal_;
}

OaLunarVec3 OaLunarScalarPhysics::SupportWorldCenter(
	const OaLunarLander3dState& InState,
	const OaLunarSupportSphere& InSupport) noexcept {
	return InState.Position_ + InState.Orientation_.Rotate(InSupport.BodyOffset_);
}

static double OaLunarEffectiveInverseMass(
	const OaLunarLander3dConfig& InConfig,
	const OaLunarLander3dState& InState,
	const OaLunarVec3& InLeverWorld,
	const OaLunarVec3& InDirectionWorld) noexcept {
	const OaLunarVec3 rotationalWorld = OaLunarCross(
		InLeverWorld, InDirectionWorld);
	const OaLunarVec3 rotationalBody = InState.Orientation_.InverseRotate(
		rotationalWorld);
	const OaLunarVec3 inverseInertiaApplied = OaLunarComponentDivide(
		rotationalBody, InConfig.DiagonalInertia_);
	return 1.0 / InConfig.Mass_
		+ OaLunarDot(rotationalBody, inverseInertiaApplied);
}

static void OaLunarApplyImpulse(
	const OaLunarLander3dConfig& InConfig,
	OaLunarLander3dState& InOutState,
	const OaLunarVec3& InLeverWorld,
	const OaLunarVec3& InImpulseWorld) noexcept {
	InOutState.LinearVelocity_ += InImpulseWorld / InConfig.Mass_;
	const OaLunarVec3 angularImpulseWorld = OaLunarCross(
		InLeverWorld, InImpulseWorld);
	const OaLunarVec3 angularImpulseBody =
		InOutState.Orientation_.InverseRotate(angularImpulseWorld);
	InOutState.AngularVelocityBody_ += OaLunarComponentDivide(
		angularImpulseBody, InConfig.DiagonalInertia_);
}

static void OaLunarResolveSupport(
	const OaLunarLander3dConfig& InConfig,
	const OaLunarTerrain& InTerrain,
	double InSubstepTime,
	const OaLunarSupportSphere& InSupport,
	bool InIsFoot,
	std::size_t InSupportIndex,
	OaLunarLander3dState& InOutState,
	OaLunarContactDiagnostics& InOutDiagnostics) noexcept {
	const OaLunarVec3 supportCenter = OaLunarScalarPhysics::SupportWorldCenter(
		InOutState, InSupport);
	const OaLunarTerrainSample terrain = InTerrain.Query(
		supportCenter.ComponentX_, supportCenter.ComponentZ_);
	if (not terrain.InBounds_) return;
	const double separation = supportCenter.ComponentY_ - InSupport.Radius_
		- terrain.Height_;
	if (separation >= 0.0) return;

	const double penetration = -separation;
	InOutDiagnostics.MaximumPenetration_ = std::max(
		InOutDiagnostics.MaximumPenetration_, penetration);
	++InOutDiagnostics.ContactCount_;
	const OaLunarVec3 leverWorld = supportCenter - InOutState.Position_;
	const OaLunarVec3 angularVelocityWorld =
		InOutState.Orientation_.Rotate(InOutState.AngularVelocityBody_);
	OaLunarVec3 pointVelocity = InOutState.LinearVelocity_
		+ OaLunarCross(angularVelocityWorld, leverWorld);
	const double normalVelocity = OaLunarDot(pointVelocity, terrain.Normal_);
	const double closingSpeed = std::max(0.0, -normalVelocity);
	if (InIsFoot) {
		InOutDiagnostics.FootContactOccurred_ = true;
		InOutDiagnostics.MaximumFootClosingSpeed_ = std::max(
			InOutDiagnostics.MaximumFootClosingSpeed_, closingSpeed);
	}
	const double biasSpeed = std::min(
		InConfig.MaxBiasSpeed_,
		penetration * InConfig.PenetrationCorrectionFraction_ / InSubstepTime);
	const double targetDeltaSpeed = std::max(
		0.0, -(1.0 + InConfig.Restitution_) * normalVelocity + biasSpeed);
	const double normalInverseMass = OaLunarEffectiveInverseMass(
		InConfig, InOutState, leverWorld, terrain.Normal_);
	double normalImpulse = targetDeltaSpeed / normalInverseMass;
	normalImpulse = std::clamp(
		normalImpulse, 0.0, InConfig.MaxContactImpulse_);
	OaLunarApplyImpulse(
		InConfig, InOutState, leverWorld, terrain.Normal_ * normalImpulse);
	InOutDiagnostics.MaximumNormalImpulse_ = std::max(
		InOutDiagnostics.MaximumNormalImpulse_, normalImpulse);

	const OaLunarVec3 updatedAngularVelocityWorld =
		InOutState.Orientation_.Rotate(InOutState.AngularVelocityBody_);
	pointVelocity = InOutState.LinearVelocity_
		+ OaLunarCross(updatedAngularVelocityWorld, leverWorld);
	const OaLunarVec3 tangentVelocity = pointVelocity
		- terrain.Normal_ * OaLunarDot(pointVelocity, terrain.Normal_);
	const double tangentSpeed = tangentVelocity.Length();
	double frictionImpulse = 0.0;
	if (tangentSpeed > 1.0e-12) {
		const OaLunarVec3 tangentDirection = -tangentVelocity / tangentSpeed;
		const double tangentInverseMass = OaLunarEffectiveInverseMass(
			InConfig, InOutState, leverWorld, tangentDirection);
		frictionImpulse = std::min(
			tangentSpeed / tangentInverseMass,
			InConfig.Friction_ * normalImpulse);
		frictionImpulse = std::clamp(
			frictionImpulse, 0.0, InConfig.MaxContactImpulse_);
		OaLunarApplyImpulse(
			InConfig, InOutState, leverWorld,
			tangentDirection * frictionImpulse);
	}
	InOutDiagnostics.MaximumFrictionImpulse_ = std::max(
		InOutDiagnostics.MaximumFrictionImpulse_, frictionImpulse);

	const double correction = std::min(
		InConfig.MaxPositionCorrectionPerContact_,
		penetration * InConfig.PenetrationCorrectionFraction_);
	InOutState.Position_ += terrain.Normal_ * correction;
	InOutDiagnostics.TotalPositionCorrection_ += correction;
	if (InIsFoot) {
		InOutState.FootContacts_[InSupportIndex] = true;
		InOutState.FootContactImpulses_[InSupportIndex] += normalImpulse;
	} else {
		InOutDiagnostics.BodyContactOccurred_ = true;
		InOutState.BodyContacts_[InSupportIndex] = true;
		InOutState.BodyContactImpulses_[InSupportIndex] += normalImpulse;
	}
}

static void OaLunarRefreshSupportContacts(
	const OaLunarLander3dConfig& InConfig,
	const OaLunarTerrain& InTerrain,
	OaLunarLander3dState& InOutState) noexcept {
	for (std::size_t index = 0U; index < InConfig.BodySupports_.size(); ++index) {
		const OaLunarSupportSphere& support = InConfig.BodySupports_[index];
		const OaLunarVec3 center = OaLunarScalarPhysics::SupportWorldCenter(
			InOutState, support);
		const OaLunarTerrainSample terrain = InTerrain.Query(center.ComponentX_, center.ComponentZ_);
		InOutState.BodyContacts_[index] = terrain.InBounds_
			and center.ComponentY_ - support.Radius_ - terrain.Height_
				<= InConfig.ContactSlop_;
	}
	for (std::size_t index = 0U; index < InConfig.FootSupports_.size(); ++index) {
		const OaLunarSupportSphere& support = InConfig.FootSupports_[index];
		const OaLunarVec3 center = OaLunarScalarPhysics::SupportWorldCenter(
			InOutState, support);
		const OaLunarTerrainSample terrain = InTerrain.Query(center.ComponentX_, center.ComponentZ_);
		InOutState.FootContacts_[index] = terrain.InBounds_
			and center.ComponentY_ - support.Radius_ - terrain.Height_
				<= InConfig.ContactSlop_;
		InOutState.FeetOnPad_[index] = terrain.InBounds_
			and InTerrain.IsOnPad(center.ComponentX_, center.ComponentZ_);
	}
}

static OaLunarVec3 OaLunarActionTorque(
	OaLunarAction InAction,
	double InMagnitude) noexcept {
	switch (InAction) {
		case OaLunarAction::PitchPositive: return {InMagnitude, 0.0, 0.0};
		case OaLunarAction::PitchNegative: return {-InMagnitude, 0.0, 0.0};
		case OaLunarAction::RollPositive: return {0.0, 0.0, InMagnitude};
		case OaLunarAction::RollNegative: return {0.0, 0.0, -InMagnitude};
		case OaLunarAction::YawPositive: return {0.0, InMagnitude, 0.0};
		case OaLunarAction::YawNegative: return {0.0, -InMagnitude, 0.0};
		case OaLunarAction::Coast:
		case OaLunarAction::MainEngine: return {};
	}
	return {};
}

OaLunarPhysicsResult OaLunarScalarPhysics::Integrate(
	const OaLunarLander3dConfig& InConfig,
	const OaLunarTerrain& InTerrain,
	OaLunarAction InAction,
	OaLunarLander3dState& InOutState) {
	OaLunarPhysicsResult result;
	const std::string configError = InConfig.ValidationError();
	if (not configError.empty()) {
		result.Valid_ = false;
		result.Error_ = configError;
		return result;
	}
	if (not InTerrain.IsValid()) {
		result.Valid_ = false;
		result.Error_ = InTerrain.Error();
		return result;
	}
	if (InTerrain.Config() != InConfig.Terrain_) {
		result.Valid_ = false;
		result.Error_ =
			"lunar integration terrain does not match the versioned configuration";
		return result;
	}
	if (not OaLunarActionIsValid(static_cast<std::uint32_t>(InAction))) {
		result.Valid_ = false;
		result.Error_ = "lunar integration action is outside Discrete(8)";
		return result;
	}
	if (not InOutState.IsFinite() or InOutState.Fuel_ < 0.0
		or InOutState.Fuel_ > InConfig.FuelCapacity_) {
		result.Valid_ = false;
		result.Error_ = "lunar integration received an invalid state";
		return result;
	}

	InOutState.LastAction_ = InAction;
	InOutState.MainThrottle_ = 0.0;
	InOutState.AttitudeCommandBody_ = {};
	InOutState.BodyContacts_.fill(false);
	InOutState.BodyContactImpulses_.fill(0.0);
	InOutState.FootContacts_.fill(false);
	InOutState.FeetOnPad_.fill(false);
	InOutState.FootContactImpulses_.fill(0.0);
	const double substepTime = InConfig.PolicyTimeStep_
		/ static_cast<double>(InConfig.PhysicsSubsteps_);

	for (std::uint32_t substep = 0U;
		substep < InConfig.PhysicsSubsteps_;
		++substep) {
		double mainActivation = 0.0;
		OaLunarVec3 attitudeTorque;
		if (InAction == OaLunarAction::MainEngine) {
			const double requestedFuel = InConfig.MainFuelRate_ * substepTime;
			mainActivation = requestedFuel > 0.0
				? std::min(1.0, InOutState.Fuel_ / requestedFuel)
				: 1.0;
			const double fuelUsed = requestedFuel * mainActivation;
			InOutState.Fuel_ -= fuelUsed;
			result.MainFuelUsed_ += fuelUsed;
		} else if (InAction != OaLunarAction::Coast) {
			const double requestedFuel = InConfig.AttitudeFuelRate_ * substepTime;
			const double activation = requestedFuel > 0.0
				? std::min(1.0, InOutState.Fuel_ / requestedFuel)
				: 1.0;
			const double fuelUsed = requestedFuel * activation;
			InOutState.Fuel_ -= fuelUsed;
			result.AttitudeFuelUsed_ += fuelUsed;
			attitudeTorque = OaLunarActionTorque(
				InAction, InConfig.AttitudeTorque_ * activation);
		}
		InOutState.Fuel_ = std::max(0.0, InOutState.Fuel_);
		InOutState.MainThrottle_ = mainActivation;
		InOutState.AttitudeCommandBody_ = attitudeTorque;

		const OaLunarVec3 thrustWorld = InOutState.Orientation_.Rotate(
			OaLunarVec3(0.0, InConfig.MainThrust_ * mainActivation, 0.0));
		const OaLunarVec3 linearAcceleration = thrustWorld / InConfig.Mass_
			+ OaLunarVec3(0.0, -InConfig.Gravity_, 0.0);
		InOutState.LinearVelocity_ += linearAcceleration * substepTime;
		InOutState.Position_ += InOutState.LinearVelocity_ * substepTime;

		const OaLunarVec3 angularMomentum = OaLunarComponentMultiply(
			InConfig.DiagonalInertia_, InOutState.AngularVelocityBody_);
		const OaLunarVec3 gyroscopicTorque = OaLunarCross(
			InOutState.AngularVelocityBody_, angularMomentum);
		const OaLunarVec3 angularAcceleration = OaLunarComponentDivide(
			attitudeTorque - gyroscopicTorque, InConfig.DiagonalInertia_);
		InOutState.AngularVelocityBody_ += angularAcceleration * substepTime;
		const OaLunarQuat angularQuaternion(
			0.0,
			InOutState.AngularVelocityBody_.ComponentX_,
			InOutState.AngularVelocityBody_.ComponentY_,
			InOutState.AngularVelocityBody_.ComponentZ_);
		const OaLunarQuat derivative =
			(InOutState.Orientation_ * angularQuaternion) * 0.5;
		InOutState.Orientation_ =
			(InOutState.Orientation_ + derivative * substepTime).Normalized();

		if (not InOutState.IsFinite()) {
			result.Valid_ = false;
			result.Error_ = "lunar integration produced a non-finite state";
			return result;
		}

		for (std::uint32_t iteration = 0U;
			iteration < InConfig.ContactIterations_;
			++iteration) {
			for (std::size_t supportIndex = 0U;
				supportIndex < InConfig.BodySupports_.size();
				++supportIndex) {
				OaLunarResolveSupport(
					InConfig, InTerrain, substepTime,
					InConfig.BodySupports_[supportIndex], false,
					supportIndex, InOutState, result.Contact_);
			}
			for (std::size_t supportIndex = 0U;
				supportIndex < InConfig.FootSupports_.size();
				++supportIndex) {
				OaLunarResolveSupport(
					InConfig, InTerrain, substepTime,
					InConfig.FootSupports_[supportIndex], true,
					supportIndex, InOutState, result.Contact_);
			}
		}
		if (not InOutState.IsFinite() or not result.Contact_.IsFinite()) {
			result.Valid_ = false;
			result.Error_ = "lunar contact resolution produced a non-finite state";
			return result;
		}
	}
	OaLunarRefreshSupportContacts(InConfig, InTerrain, InOutState);
	const double maximumCorrection = static_cast<double>(
		InConfig.PhysicsSubsteps_ * InConfig.ContactIterations_)
		* static_cast<double>(
			InConfig.BodySupports_.size() + InConfig.FootSupports_.size())
		* InConfig.MaxPositionCorrectionPerContact_;
	result.Contact_.Bounded_ = result.Contact_.MaximumNormalImpulse_
		<= InConfig.MaxContactImpulse_
		and result.Contact_.MaximumFrictionImpulse_
		<= InConfig.MaxContactImpulse_
		and result.Contact_.TotalPositionCorrection_
		<= maximumCorrection + 1.0e-12;
	return result;
}

std::array<float, OA_LUNAR_OBSERVATION_SIZE> OaLunarScalarPhysics::Observe(
	const OaLunarLander3dConfig& InConfig,
	const OaLunarTerrain& InTerrain,
	const OaLunarLander3dState& InState) noexcept {
	std::array<float, OA_LUNAR_OBSERVATION_SIZE> observation{};
	if (not InState.IsFinite() or not InTerrain.IsValid()) {
		return observation;
	}
	std::size_t offset = 0U;
	observation[offset++] = OaLunarNormalizedFloat(
		InState.Position_.ComponentX_, InConfig.PositionObservationScale_);
	observation[offset++] = OaLunarNormalizedFloat(
		InState.Position_.ComponentY_, InConfig.PositionObservationScale_);
	observation[offset++] = OaLunarNormalizedFloat(
		InState.Position_.ComponentZ_, InConfig.PositionObservationScale_);
	observation[offset++] = OaLunarNormalizedFloat(
		InState.LinearVelocity_.ComponentX_, InConfig.VelocityObservationScale_);
	observation[offset++] = OaLunarNormalizedFloat(
		InState.LinearVelocity_.ComponentY_, InConfig.VelocityObservationScale_);
	observation[offset++] = OaLunarNormalizedFloat(
		InState.LinearVelocity_.ComponentZ_, InConfig.VelocityObservationScale_);

	const OaLunarVec3 bodyUp = InState.Orientation_.Rotate({0.0, 1.0, 0.0});
	const OaLunarVec3 bodyForward = InState.Orientation_.Rotate({0.0, 0.0, -1.0});
	observation[offset++] = static_cast<float>(OaLunarClampUnit(bodyUp.ComponentX_));
	observation[offset++] = static_cast<float>(OaLunarClampUnit(bodyUp.ComponentY_));
	observation[offset++] = static_cast<float>(OaLunarClampUnit(bodyUp.ComponentZ_));
	observation[offset++] = static_cast<float>(OaLunarClampUnit(bodyForward.ComponentX_));
	observation[offset++] = static_cast<float>(OaLunarClampUnit(bodyForward.ComponentY_));
	observation[offset++] = static_cast<float>(OaLunarClampUnit(bodyForward.ComponentZ_));
	observation[offset++] = OaLunarNormalizedFloat(
		InState.AngularVelocityBody_.ComponentX_,
		InConfig.AngularVelocityObservationScale_);
	observation[offset++] = OaLunarNormalizedFloat(
		InState.AngularVelocityBody_.ComponentY_,
		InConfig.AngularVelocityObservationScale_);
	observation[offset++] = OaLunarNormalizedFloat(
		InState.AngularVelocityBody_.ComponentZ_,
		InConfig.AngularVelocityObservationScale_);

	for (std::int32_t probeZ = -1; probeZ <= 1; ++probeZ) {
		for (std::int32_t probeX = -1; probeX <= 1; ++probeX) {
			const double positionX = InState.Position_.ComponentX_
				+ static_cast<double>(probeX) * InConfig.TerrainProbeSpacing_;
			const double positionZ = InState.Position_.ComponentZ_
				+ static_cast<double>(probeZ) * InConfig.TerrainProbeSpacing_;
			const OaLunarTerrainSample terrain = InTerrain.Query(
				positionX, positionZ);
			const double clearance = terrain.InBounds_
				? InState.Position_.ComponentY_ - terrain.Height_
				: InConfig.TerrainClearanceObservationScale_;
			observation[offset++] = OaLunarNormalizedFloat(
				clearance, InConfig.TerrainClearanceObservationScale_);
		}
	}
	for (std::size_t footIndex = 0U;
		footIndex < InConfig.FootSupports_.size();
		++footIndex) {
		const OaLunarSupportSphere& support = InConfig.FootSupports_[footIndex];
		const OaLunarVec3 center = SupportWorldCenter(InState, support);
		const OaLunarTerrainSample terrain = InTerrain.Query(center.ComponentX_, center.ComponentZ_);
		const double clearance = terrain.InBounds_
			? center.ComponentY_ - support.Radius_ - terrain.Height_
			: InConfig.FootClearanceObservationScale_;
		observation[offset++] = OaLunarNormalizedFloat(
			clearance, InConfig.FootClearanceObservationScale_);
	}
	for (const bool contact : InState.FootContacts_) {
		observation[offset++] = contact ? 1.0F : 0.0F;
	}
	observation[offset++] = static_cast<float>(std::clamp(
		InState.Fuel_ / InConfig.FuelCapacity_, 0.0, 1.0));
	return observation;
}

double OaLunarScalarPhysics::Potential(
	const OaLunarLander3dConfig& InConfig,
	const OaLunarLander3dState& InState) noexcept {
	if (not InState.IsFinite()) return 0.0;
	const double positionCost = std::min(
		1.0, InState.Position_.Length() / InConfig.PositionObservationScale_);
	const double velocityCost = std::min(
		1.0, InState.LinearVelocity_.Length()
			/ InConfig.VelocityObservationScale_);
	const OaLunarVec3 bodyUp = InState.Orientation_.Rotate({0.0, 1.0, 0.0});
	const double tiltCost = std::clamp((1.0 - bodyUp.ComponentY_) * 0.5, 0.0, 1.0);
	const double angularCost = std::min(
		1.0, InState.AngularVelocityBody_.Length()
			/ InConfig.AngularVelocityObservationScale_);
	return -(
		InConfig.PositionPotentialWeight_ * positionCost
		+ InConfig.VelocityPotentialWeight_ * velocityCost
		+ InConfig.TiltPotentialWeight_ * tiltCost
		+ InConfig.AngularPotentialWeight_ * angularCost);
}
