#include <ml/rl/lunarlander3d/lunarLander3dVector.h>

#include <oa/core/bufferAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/op.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/deviceAccess.h>
#include <oa/runtime/oaVk.h>

#include <ml/rl/environmentKernelPack.h>
#include <ml/rl/gen/environmentOpRegistry.h>
#include <oa/ml/environmentExecution.h>
#include <oa/runtime/dispatchValidation.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace oa {

enum class LunarVectorConfigF32Index : oa::U32 {
	PolicyTimeStep,
	Gravity,
	Mass,
	InertiaX,
	InertiaY,
	InertiaZ,
	MainThrust,
	AttitudeTorque,
	FuelCapacity,
	MainFuelRate,
	AttitudeFuelRate,
	Restitution,
	Friction,
	ContactSlop,
	PenetrationCorrectionFraction,
	MaxPositionCorrectionPerContact,
	MaxContactImpulse,
	MaxBiasSpeed,
	TaskMinimumY,
	TaskMaximumY,
	SafeLinearSpeed,
	SafeAngularSpeed,
	SafeTiltRadians,
	HardFootImpactSpeed,
	PositionObservationScale,
	VelocityObservationScale,
	AngularVelocityObservationScale,
	TerrainClearanceObservationScale,
	FootClearanceObservationScale,
	TerrainProbeSpacing,
	RewardGamma,
	PositionPotentialWeight,
	VelocityPotentialWeight,
	TiltPotentialWeight,
	AngularPotentialWeight,
	MainFuelCostWeight,
	AttitudeFuelCostWeight,
	SoftFootContactReward,
	StableDwellReward,
	SuccessReward,
	FailurePenalty,
	TerrainCellSize,
	TerrainMinimumX,
	TerrainMaximumX,
	TerrainMinimumZ,
	TerrainMaximumZ,
	PadHalfExtent,
	SupportBase,
};

static constexpr oa::U32 kLunarVectorConfigF32Count = 75U;
static constexpr oa::U32 kLunarVectorConfigU32Count = 17U;
static constexpr oa::U32 kLunarVectorStateF32Width = 32U;
static constexpr oa::U32 kLunarVectorStateU32Width = 16U;
static constexpr oa::U32 kLunarVectorStateLinearVelocity = 3U;
static constexpr oa::U32 kLunarVectorStateAngularVelocity = 10U;
static constexpr oa::U32 kLunarVectorStateFuel = 13U;
static constexpr oa::U32 kLunarVectorStateFootImpulses = 21U;
static constexpr oa::U32 kLunarVectorStateEpisodeReturn = 25U;
static constexpr oa::U32 kLunarVectorStateEpisodeStep = 5U;
static constexpr oa::U32 kLunarVectorStateTerminated = 7U;
static constexpr oa::U32 kLunarVectorStateTruncated = 8U;
static constexpr oa::U32 kLunarVectorStateEndReason = 9U;

static oa::Usize lunarVectorConfigIndex(
	LunarVectorConfigF32Index inIndex) noexcept {
	return static_cast<oa::Usize>(inIndex);
}

static oa::Result<oa::Vec<oa::F32>> lunarVectorSerializeConfigF32(
	const LunarLander3dConfig& inConfig,
	const LunarTerrain& inTerrain) {
	oa::Vec<oa::F32> values(kLunarVectorConfigF32Count, 0.0F);
	bool losesNonzero = false;
	bool requiresSubnormal = false;
	auto setValue = [&values, &losesNonzero, &requiresSubnormal](
		oa::Usize inIndex, double inValue) {
		const oa::F32 converted = static_cast<oa::F32>(inValue);
		values[inIndex] = converted;
		losesNonzero = losesNonzero
			or (inValue != 0.0 and converted == 0.0F);
		requiresSubnormal = requiresSubnormal
			or (converted != 0.0F
				and std::fpclassify(converted) == FP_SUBNORMAL);
	};
	auto set = [&setValue](
		LunarVectorConfigF32Index inIndex, double inValue) {
		setValue(lunarVectorConfigIndex(inIndex), inValue);
	};
	set(LunarVectorConfigF32Index::PolicyTimeStep, inConfig.policyTimeStep_);
	set(LunarVectorConfigF32Index::Gravity, inConfig.gravity_);
	set(LunarVectorConfigF32Index::Mass, inConfig.mass_);
	set(LunarVectorConfigF32Index::InertiaX,
		inConfig.diagonalInertia_.x);
	set(LunarVectorConfigF32Index::InertiaY,
		inConfig.diagonalInertia_.y);
	set(LunarVectorConfigF32Index::InertiaZ,
		inConfig.diagonalInertia_.z);
	set(LunarVectorConfigF32Index::MainThrust, inConfig.mainThrust_);
	set(LunarVectorConfigF32Index::AttitudeTorque, inConfig.attitudeTorque_);
	set(LunarVectorConfigF32Index::FuelCapacity, inConfig.fuelCapacity_);
	set(LunarVectorConfigF32Index::MainFuelRate, inConfig.mainFuelRate_);
	set(LunarVectorConfigF32Index::AttitudeFuelRate,
		inConfig.attitudeFuelRate_);
	set(LunarVectorConfigF32Index::Restitution, inConfig.restitution_);
	set(LunarVectorConfigF32Index::Friction, inConfig.friction_);
	set(LunarVectorConfigF32Index::ContactSlop, inConfig.contactSlop_);
	set(LunarVectorConfigF32Index::PenetrationCorrectionFraction,
		inConfig.penetrationCorrectionFraction_);
	set(LunarVectorConfigF32Index::MaxPositionCorrectionPerContact,
		inConfig.maxPositionCorrectionPerContact_);
	set(LunarVectorConfigF32Index::MaxContactImpulse,
		inConfig.maxContactImpulse_);
	set(LunarVectorConfigF32Index::MaxBiasSpeed, inConfig.maxBiasSpeed_);
	set(LunarVectorConfigF32Index::TaskMinimumY, inConfig.taskMinimumY_);
	set(LunarVectorConfigF32Index::TaskMaximumY, inConfig.taskMaximumY_);
	set(LunarVectorConfigF32Index::SafeLinearSpeed,
		inConfig.safeLinearSpeed_);
	set(LunarVectorConfigF32Index::SafeAngularSpeed,
		inConfig.safeAngularSpeed_);
	set(LunarVectorConfigF32Index::SafeTiltRadians,
		inConfig.safeTiltRadians_);
	set(LunarVectorConfigF32Index::HardFootImpactSpeed,
		inConfig.hardFootImpactSpeed_);
	set(LunarVectorConfigF32Index::PositionObservationScale,
		inConfig.positionObservationScale_);
	set(LunarVectorConfigF32Index::VelocityObservationScale,
		inConfig.velocityObservationScale_);
	set(LunarVectorConfigF32Index::AngularVelocityObservationScale,
		inConfig.angularVelocityObservationScale_);
	set(LunarVectorConfigF32Index::TerrainClearanceObservationScale,
		inConfig.terrainClearanceObservationScale_);
	set(LunarVectorConfigF32Index::FootClearanceObservationScale,
		inConfig.footClearanceObservationScale_);
	set(LunarVectorConfigF32Index::TerrainProbeSpacing,
		inConfig.terrainProbeSpacing_);
	set(LunarVectorConfigF32Index::RewardGamma, inConfig.rewardGamma_);
	set(LunarVectorConfigF32Index::PositionPotentialWeight,
		inConfig.positionPotentialWeight_);
	set(LunarVectorConfigF32Index::VelocityPotentialWeight,
		inConfig.velocityPotentialWeight_);
	set(LunarVectorConfigF32Index::TiltPotentialWeight,
		inConfig.tiltPotentialWeight_);
	set(LunarVectorConfigF32Index::AngularPotentialWeight,
		inConfig.angularPotentialWeight_);
	set(LunarVectorConfigF32Index::MainFuelCostWeight,
		inConfig.mainFuelCostWeight_);
	set(LunarVectorConfigF32Index::AttitudeFuelCostWeight,
		inConfig.attitudeFuelCostWeight_);
	set(LunarVectorConfigF32Index::SoftFootContactReward,
		inConfig.softFootContactReward_);
	set(LunarVectorConfigF32Index::StableDwellReward,
		inConfig.stableDwellReward_);
	set(LunarVectorConfigF32Index::SuccessReward, inConfig.successReward_);
	set(LunarVectorConfigF32Index::FailurePenalty, inConfig.failurePenalty_);
	set(LunarVectorConfigF32Index::TerrainCellSize,
		inConfig.terrain_.cellSize_);
	set(LunarVectorConfigF32Index::TerrainMinimumX, inTerrain.minX());
	set(LunarVectorConfigF32Index::TerrainMaximumX, inTerrain.maxX());
	set(LunarVectorConfigF32Index::TerrainMinimumZ, inTerrain.minZ());
	set(LunarVectorConfigF32Index::TerrainMaximumZ, inTerrain.maxZ());
	set(LunarVectorConfigF32Index::PadHalfExtent,
		inConfig.terrain_.padHalfExtent_);

	oa::U32 supportOffset = static_cast<oa::U32>(
		LunarVectorConfigF32Index::SupportBase);
	auto appendSupport = [&setValue, &supportOffset](
		const LunarSupportSphere& inSupport) {
		setValue(supportOffset++, inSupport.bodyOffset_.x);
		setValue(supportOffset++, inSupport.bodyOffset_.y);
		setValue(supportOffset++, inSupport.bodyOffset_.z);
		setValue(supportOffset++, inSupport.radius_);
	};
	for (const LunarSupportSphere& support : inConfig.bodySupports_) {
		appendSupport(support);
	}
	for (const LunarSupportSphere& support : inConfig.footSupports_) {
		appendSupport(support);
	}
	if (losesNonzero or requiresSubnormal) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"Lunar Lander 3D configuration contains a value that underflows or requires FP32 subnormal preservation");
	}
	return values;
}

static oa::Vec<oa::U32> lunarVectorSerializeConfigU32(
	const LunarLander3dConfig& inConfig) {
	oa::Vec<oa::U32> values(kLunarVectorConfigU32Count, 0U);
	const oa::U64 fingerprint = inConfig.contractFingerprint();
	values[0] = kLunarVectorConfigLayoutVersion;
	values[1] = inConfig.environmentVersion_;
	values[2] = kLunarRandomVersion;
	values[3] = kLunarTerrainVersion;
	values[4] = inConfig.physicsVersion_;
	values[5] = inConfig.observationVersion_;
	values[6] = inConfig.rewardVersion_;
	values[7] = inConfig.physicsSubsteps_;
	values[8] = inConfig.contactIterations_;
	values[9] = inConfig.safeDwellSteps_;
	values[10] = inConfig.maxEpisodeSteps_;
	values[11] = inConfig.terrain_.cellsX_;
	values[12] = inConfig.terrain_.cellsZ_;
	values[13] = static_cast<oa::U32>(fingerprint);
	values[14] = static_cast<oa::U32>(fingerprint >> 32U);
	values[15] = kLunarVectorStateF32Width;
	values[16] = kLunarVectorStateU32Width;
	return values;
}

static bool lunarVectorAllFinite(const oa::Vec<oa::F32>& inValues) noexcept {
	for (const oa::F32 value : inValues) {
		if (not std::isfinite(value)) return false;
	}
	return true;
}

static oa::Status lunarVectorValidateSerializedConfig(
	const oa::Vec<oa::F32>& inValues,
	const LunarLander3dConfig& inConfig) {
	if (inValues.size() != kLunarVectorConfigF32Count
		or not lunarVectorAllFinite(inValues)) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"Lunar Lander 3D configuration cannot be represented as finite FP32");
	}
	const auto get = [&inValues](LunarVectorConfigF32Index inIndex) {
		return inValues[lunarVectorConfigIndex(inIndex)];
	};
	const auto safeDenominator = [&get](
		LunarVectorConfigF32Index inIndex) {
		const oa::F32 value = get(inIndex);
		return value > 0.0F and std::isfinite(1.0F / value);
	};
	const bool positiveDenominators =
		safeDenominator(LunarVectorConfigF32Index::PolicyTimeStep)
		and safeDenominator(LunarVectorConfigF32Index::Mass)
		and safeDenominator(LunarVectorConfigF32Index::InertiaX)
		and safeDenominator(LunarVectorConfigF32Index::InertiaY)
		and safeDenominator(LunarVectorConfigF32Index::InertiaZ)
		and safeDenominator(LunarVectorConfigF32Index::FuelCapacity)
		and get(LunarVectorConfigF32Index::MaxContactImpulse) > 0.0F
		and safeDenominator(
			LunarVectorConfigF32Index::PositionObservationScale)
		and safeDenominator(
			LunarVectorConfigF32Index::VelocityObservationScale)
		and safeDenominator(
			LunarVectorConfigF32Index::AngularVelocityObservationScale)
		and safeDenominator(
			LunarVectorConfigF32Index::TerrainClearanceObservationScale)
		and safeDenominator(
			LunarVectorConfigF32Index::FootClearanceObservationScale)
		and get(LunarVectorConfigF32Index::TerrainProbeSpacing) > 0.0F
		and safeDenominator(LunarVectorConfigF32Index::TerrainCellSize);
	const bool orderedBounds =
		get(LunarVectorConfigF32Index::TaskMinimumY)
			< get(LunarVectorConfigF32Index::TaskMaximumY)
		and get(LunarVectorConfigF32Index::TerrainMinimumX)
			< get(LunarVectorConfigF32Index::TerrainMaximumX)
		and get(LunarVectorConfigF32Index::TerrainMinimumZ)
			< get(LunarVectorConfigF32Index::TerrainMaximumZ);
	const oa::F32 substepTime =
		get(LunarVectorConfigF32Index::PolicyTimeStep)
		/ static_cast<oa::F32>(inConfig.physicsSubsteps_);
	const bool representableSubstep = substepTime > 0.0F
		and std::isfinite(1.0F / substepTime)
		and std::fpclassify(substepTime) != FP_SUBNORMAL;
	const oa::F32 fuelCapacity = get(
		LunarVectorConfigF32Index::FuelCapacity);
	const auto representableFuelDebit = [substepTime, fuelCapacity](
		oa::F32 inFuelRate) {
		if (inFuelRate == 0.0F) return true;
		const oa::F32 debit = inFuelRate * substepTime;
		return debit > 0.0F and std::isfinite(debit)
			and std::fpclassify(debit) != FP_SUBNORMAL
			and fuelCapacity - debit != fuelCapacity;
	};
	const bool representableFuelRates = representableSubstep
		and representableFuelDebit(get(
			LunarVectorConfigF32Index::MainFuelRate))
		and representableFuelDebit(get(
			LunarVectorConfigF32Index::AttitudeFuelRate));
	bool representableSupportRadii = true;
	oa::U32 supportOffset = static_cast<oa::U32>(
		LunarVectorConfigF32Index::SupportBase);
	for (oa::U32 support = 0U; support < 7U; ++support) {
		representableSupportRadii = representableSupportRadii
			and inValues[supportOffset + support * 4U + 3U] > 0.0F;
	}
	const double potentialMagnitude =
		static_cast<double>(get(
			LunarVectorConfigF32Index::PositionPotentialWeight))
		+ static_cast<double>(get(
			LunarVectorConfigF32Index::VelocityPotentialWeight))
		+ static_cast<double>(get(
			LunarVectorConfigF32Index::TiltPotentialWeight))
		+ static_cast<double>(get(
			LunarVectorConfigF32Index::AngularPotentialWeight));
	const double fuelCostMagnitude =
		static_cast<double>(get(LunarVectorConfigF32Index::FuelCapacity))
		* (static_cast<double>(get(
			LunarVectorConfigF32Index::MainFuelCostWeight))
			+ static_cast<double>(get(
				LunarVectorConfigF32Index::AttitudeFuelCostWeight)));
	const double terminalMagnitude = std::max(
		static_cast<double>(get(LunarVectorConfigF32Index::SuccessReward)),
		std::abs(static_cast<double>(get(
			LunarVectorConfigF32Index::FailurePenalty))));
	const double maximumRewardMagnitude =
		(1.0 + static_cast<double>(get(
			LunarVectorConfigF32Index::RewardGamma))) * potentialMagnitude
		+ fuelCostMagnitude
		+ 4.0 * static_cast<double>(get(
			LunarVectorConfigF32Index::SoftFootContactReward))
		+ static_cast<double>(get(
			LunarVectorConfigF32Index::StableDwellReward))
		+ terminalMagnitude;
	const double maximumEpisodeReturnMagnitude = maximumRewardMagnitude
		* static_cast<double>(inConfig.maxEpisodeSteps_);
	const double maximumFp32 =
		static_cast<double>(std::numeric_limits<oa::F32>::max());
	const bool boundedRewards = std::isfinite(maximumRewardMagnitude)
		and maximumRewardMagnitude <= maximumFp32
		and std::isfinite(maximumEpisodeReturnMagnitude)
		and maximumEpisodeReturnMagnitude <= maximumFp32;
	if (not positiveDenominators or not orderedBounds
		or not representableFuelRates or not representableSupportRadii
		or not boundedRewards) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"Lunar Lander 3D configuration loses required FP32 relationships or reward bounds");
	}
	return oa::Status::ok();
}

static oa::Status lunarVectorValidateDeviceLimits(
	oa::Engine& inEngine,
	oa::U32 inEnvironments,
	oa::U64 inTerrainVertices) {
	if (not oa::EngineDeviceAccess::get(inEngine).instance or not oa::EngineDeviceAccess::get(inEngine).physicalDevice) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D requires queried vulkan device limits");
	}
	OaVkInstanceTable instanceTable{};
	oaVkLoadInstanceTable(
		&instanceTable,
		static_cast<VkInstance>(oa::EngineDeviceAccess::get(inEngine).instance));
	if (not instanceTable.vkGetPhysicalDeviceProperties
		or not instanceTable.vkGetPhysicalDeviceFeatures2) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D requires vulkan instance capability queries");
	}
	const oa::U32 apiVersion = oa::EngineDeviceAccess::get(inEngine).info.software.apiVersionPacked;
	const bool core12 = VK_API_VERSION_MAJOR(apiVersion) > 1U
		or (VK_API_VERSION_MAJOR(apiVersion) == 1U
			and VK_API_VERSION_MINOR(apiVersion) >= 2U);
	if (not core12) {
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			"Lunar Lander 3D UInt8 transition outputs require vulkan 1.2 storage-8 capabilities enabled by oa::Engine");
	}
	VkPhysicalDeviceVulkan12Features features12{};
	features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	VkPhysicalDeviceFeatures2 features2{};
	features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	features2.pNext = &features12;
	instanceTable.vkGetPhysicalDeviceFeatures2(
		static_cast<VkPhysicalDevice>(oa::EngineDeviceAccess::get(inEngine).physicalDevice), &features2);
	if (features12.shaderInt8 != VK_TRUE
		or features12.uniformAndStorageBuffer8BitAccess != VK_TRUE) {
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			"Lunar Lander 3D UInt8 transition outputs require shaderInt8 and uniformAndStorageBuffer8BitAccess");
	}
	VkPhysicalDeviceProperties properties{};
	instanceTable.vkGetPhysicalDeviceProperties(
		static_cast<VkPhysicalDevice>(oa::EngineDeviceAccess::get(inEngine).physicalDevice),
		&properties);
	if (properties.limits.maxComputeWorkGroupInvocations < 256U
		or properties.limits.maxComputeWorkGroupSize[0] < 256U) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D requires a queried 256-thread compute workgroup");
	}
	const oa::U64 groupsX = 1U + (static_cast<oa::U64>(inEnvironments) - 1U) / 256U;
	OA_RETURN_IF_ERROR(oavk::validateDirectComputeDispatch(
		oa::EngineDeviceAccess::get(inEngine), static_cast<oa::U32>(groupsX), 1U, 1U));
	const oa::U64 maximumBufferBytes = std::min<oa::U64>(
		properties.limits.maxStorageBufferRange,
		std::numeric_limits<oa::U32>::max());
	auto validateBuffer = [maximumBufferBytes](
		oa::U64 inElements, oa::U64 inElementBytes, oa::StringView inName) -> oa::Status {
		if (inElements > std::numeric_limits<oa::U64>::max() / inElementBytes
			or inElements * inElementBytes > maximumBufferBytes) {
			return oa::Status::error(
				oa::StatusCode::OutOfRange,
				oa::String(inName) + " exceeds this device's queried storage-buffer range");
		}
		return oa::Status::ok();
	};
	OA_RETURN_IF_ERROR(validateBuffer(
		static_cast<oa::U64>(inEnvironments) * kLunarVectorStateF32Width,
		sizeof(oa::F32), "Lunar Lander 3D FP32 state"));
	OA_RETURN_IF_ERROR(validateBuffer(
		static_cast<oa::U64>(inEnvironments) * kLunarVectorStateU32Width,
		sizeof(oa::U32), "Lunar Lander 3D UInt32 state"));
	OA_RETURN_IF_ERROR(validateBuffer(
		static_cast<oa::U64>(inEnvironments) * kLunarObservationSize,
		sizeof(oa::F32), "Lunar Lander 3D observation"));
	OA_RETURN_IF_ERROR(validateBuffer(
		inTerrainVertices, sizeof(oa::F32), "Lunar Lander 3D terrain"));
	OA_RETURN_IF_ERROR(validateBuffer(
		inEnvironments, sizeof(oa::U32), "Lunar Lander 3D lane vector"));
	return oa::Status::ok();
}

static oa::Matrix lunarVectorFromF32(const oa::Vec<oa::F32>& inValues) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(inValues.data()),
			inValues.size() * sizeof(oa::F32)),
		{static_cast<oa::I64>(inValues.size())}, oa::ScalarType::Float32);
}

static oa::Matrix lunarVectorFromU32(const oa::Vec<oa::U32>& inValues) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(inValues.data()),
			inValues.size() * sizeof(oa::U32)),
		{static_cast<oa::I64>(inValues.size())}, oa::ScalarType::UInt32);
}

bool LunarLander3dVectorStep::isValid() const noexcept {
	return not observation_.isEmpty() and not nextObservation_.isEmpty()
		and not reward_.isEmpty() and not terminated_.isEmpty()
		and not truncated_.isEmpty() and not endReason_.isEmpty();
}

bool LunarLander3dEpisodeTelemetry::isFinite() const noexcept {
	return std::isfinite(episodeReturn_) and std::isfinite(fuelRemaining_)
		and std::isfinite(terminalLinearSpeed_)
		and std::isfinite(terminalAngularSpeed_)
		and std::isfinite(maximumFootImpulse_);
}

LunarLander3dVector::LunarLander3dVector(oa::Engine& inEngine)
	: oa::Environment(inEngine) {}

oa::U64 LunarLander3dVector::effectiveSeed_() const noexcept {
	return hasPendingSeed_ ? pendingSeed_ : config_.seed_;
}

oa::Result<LunarLander3dVector> LunarLander3dVector::createFlat(
	oa::Engine& inEngine,
	const LunarLander3dVectorConfig& inConfig) {
	if (not inEngine.isReady()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D requires a ready engine");
	}
	OA_RETURN_IF_ERROR(oa::ensureEnvironmentKernelPack(inEngine));
	if (inConfig.environments_ == 0U) {
		return oa::Status::invalidArgument(
			"Lunar Lander 3D requires at least one environment lane");
	}
	const oa::String configError(inConfig.environment_.validationError());
	if (not configError.empty()) return oa::Status::invalidArgument(configError);
	const LunarTerrain terrain = LunarTerrain::createFlat(
		inConfig.environment_.terrain_);
	if (not terrain.isValid()) {
		return oa::Status::invalidArgument(oa::String(terrain.error()));
	}
	const oa::U64 terrainVertices = static_cast<oa::U64>(terrain.heights().size());
	auto serializedConfig = lunarVectorSerializeConfigF32(
		inConfig.environment_, terrain);
	if (serializedConfig.isError()) return serializedConfig.getStatus();
	oa::Vec<oa::F32> configF32 = oa::move(serializedConfig).getValue();
	OA_RETURN_IF_ERROR(lunarVectorValidateSerializedConfig(
		configF32, inConfig.environment_));
	OA_RETURN_IF_ERROR(lunarVectorValidateDeviceLimits(
		inEngine, inConfig.environments_, terrainVertices));
	const oa::Vec<oa::U32> configU32 = lunarVectorSerializeConfigU32(
		inConfig.environment_);
	oa::Vec<oa::F32> terrainF32(
		static_cast<oa::Usize>(terrainVertices), 0.0F);
	for (oa::Usize index = 0; index < terrainF32.size(); ++index) {
		terrainF32[index] = static_cast<oa::F32>(terrain.heights()[index]);
	}
	const oa::Vec<oa::U8> noExternalStop(inConfig.environments_, 0U);

	const oa::MatrixShape stateF32Shape{
		static_cast<oa::I64>(inConfig.environments_),
		static_cast<oa::I64>(kLunarVectorStateF32Width)};
	const oa::MatrixShape stateU32Shape{
		static_cast<oa::I64>(inConfig.environments_),
		static_cast<oa::I64>(kLunarVectorStateU32Width)};
	const oa::MatrixShape observationShape{
		static_cast<oa::I64>(inConfig.environments_),
		static_cast<oa::I64>(kLunarObservationSize)};
	const oa::MatrixShape vectorShape{
		static_cast<oa::I64>(inConfig.environments_)};

	LunarLander3dVector result(inEngine);
	if (not result.isOpen()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D could not open its execution session");
	}
	result.config_ = inConfig;
	result.spec_ = {
		.observation = oa::EnvironmentSpace::box(
			"observation", {static_cast<oa::I64>(kLunarObservationSize)},
			oa::ScalarType::Float32, -1.0, 1.0),
		.action = oa::EnvironmentSpace::discrete("action", 8),
		.reward = oa::EnvironmentSpace::box("reward", {}, oa::ScalarType::Float32),
		.terminated = oa::EnvironmentSpace::binary("terminated"),
		.truncated = oa::EnvironmentSpace::binary("truncated"),
	};
	OA_RETURN_IF_ERROR(result.spec_.validateDefinition());
	const oa::Status initialized = result.recordCommands([&]() -> oa::Status {
		result.configF32_ = lunarVectorFromF32(configF32);
		result.configU32_ = lunarVectorFromU32(configU32);
		result.terrainF32_ = lunarVectorFromF32(terrainF32);
		result.stateF32_ = oa::FnMatrix::empty(
			stateF32Shape, oa::ScalarType::Float32);
		result.stateU32_ = oa::FnMatrix::empty(
			stateU32Shape, oa::ScalarType::UInt32);
		result.observation_ = oa::FnMatrix::empty(
			observationShape, oa::ScalarType::Float32);
		result.transitionObservation_ = oa::FnMatrix::empty(
			observationShape, oa::ScalarType::Float32);
		result.reward_ = oa::FnMatrix::empty(
			vectorShape, oa::ScalarType::Float32);
		result.terminated_ = oa::FnMatrix::empty(
			vectorShape, oa::ScalarType::UInt8);
		result.truncated_ = oa::FnMatrix::empty(
			vectorShape, oa::ScalarType::UInt8);
		result.endReason_ = oa::FnMatrix::empty(
			vectorShape, oa::ScalarType::UInt32);
		// FromBytes completes this upload before returning, and the member owns
		// the input storage across deferred submission and transaction rollback.
		result.noExternalStop_ = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(
				noExternalStop.data(), noExternalStop.size()),
			vectorShape, oa::ScalarType::UInt8);
		if (not result.isValid()) {
			return oa::Status::error(
				oa::StatusCode::OutOfMemory,
				"Lunar Lander 3D could not allocate bounded device storage");
		}
		return result.recordReset_(false);
	});
	if (initialized.isError()) return initialized;
	return result;
}

bool LunarLander3dVector::isValid() const noexcept {
	const oa::MatrixShape stateF32Shape{
		static_cast<oa::I64>(config_.environments_),
		static_cast<oa::I64>(kLunarVectorStateF32Width)};
	const oa::MatrixShape stateU32Shape{
		static_cast<oa::I64>(config_.environments_),
		static_cast<oa::I64>(kLunarVectorStateU32Width)};
	const oa::MatrixShape observationShape{
		static_cast<oa::I64>(config_.environments_),
		static_cast<oa::I64>(kLunarObservationSize)};
	const oa::MatrixShape vectorShape{
		static_cast<oa::I64>(config_.environments_)};
	const oa::MatrixShape configF32Shape{
		static_cast<oa::I64>(kLunarVectorConfigF32Count)};
	const oa::MatrixShape configU32Shape{
		static_cast<oa::I64>(kLunarVectorConfigU32Count)};
	const oa::I64 terrainVertices =
		(static_cast<oa::I64>(config_.environment_.terrain_.cellsX_) + 1)
		* (static_cast<oa::I64>(config_.environment_.terrain_.cellsZ_) + 1);
	const auto matches = [](const oa::Matrix& inMatrix,
		const oa::MatrixShape& inShape, oa::ScalarType inDtype) {
		return not inMatrix.isEmpty() and inMatrix.getShape() == inShape
			and inMatrix.getDtype() == inDtype;
	};
	return matches(configF32_, configF32Shape, oa::ScalarType::Float32)
		and matches(configU32_, configU32Shape, oa::ScalarType::UInt32)
		and matches(terrainF32_, {terrainVertices}, oa::ScalarType::Float32)
		and matches(stateF32_, stateF32Shape, oa::ScalarType::Float32)
		and matches(stateU32_, stateU32Shape, oa::ScalarType::UInt32)
		and matches(observation_, observationShape, oa::ScalarType::Float32)
		and matches(
			transitionObservation_, observationShape, oa::ScalarType::Float32)
		and matches(reward_, vectorShape, oa::ScalarType::Float32)
		and matches(terminated_, vectorShape, oa::ScalarType::UInt8)
		and matches(truncated_, vectorShape, oa::ScalarType::UInt8)
		and matches(endReason_, vectorShape, oa::ScalarType::UInt32)
		and matches(noExternalStop_, vectorShape, oa::ScalarType::UInt8);
}

oa::Result<oa::Vec<LunarLander3dEpisodeTelemetry>>
LunarLander3dVector::copyEpisodeTelemetry() const {
	if (not isValid() or not isOpen()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D telemetry requires a valid open environment");
	}
	if (hasActiveRecording()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D telemetry requires submitting and waiting for, or cancelling, the active recording");
	}
	if (hasPendingEvent()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D telemetry requires waiting on the exact submitted event");
	}
	const oa::Usize environments = static_cast<oa::Usize>(config_.environments_);
	oa::Vec<oa::F32> stateF32(
		environments * kLunarVectorStateF32Width, 0.0F);
	oa::Vec<oa::U32> stateU32(
		environments * kLunarVectorStateU32Width, 0U);
	{
		// CopyToHost resolves its runtime through the selected context. Select
		// this environment's private execution context so a borrowed non-default
		// engine can never read its buffers through an ambient compatibility
		// context.
		oa::EnvironmentRecordingScope scope(*this);
		OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
			stateF32_, stateF32.data(),
			static_cast<oa::U64>(stateF32.size() * sizeof(oa::F32))));
		OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
			stateU32_, stateU32.data(),
			static_cast<oa::U64>(stateU32.size() * sizeof(oa::U32))));
	}

	oa::Vec<LunarLander3dEpisodeTelemetry> result;
	result.reserve(environments);
	for (oa::Usize lane = 0U; lane < environments; ++lane) {
		const oa::Usize f32Base = lane * kLunarVectorStateF32Width;
		const oa::Usize u32Base = lane * kLunarVectorStateU32Width;
		const auto vectorLength = [&stateF32, f32Base](oa::U32 inOffset) {
			const oa::F64 x = stateF32[f32Base + inOffset];
			const oa::F64 y = stateF32[f32Base + inOffset + 1U];
			const oa::F64 z = stateF32[f32Base + inOffset + 2U];
			return static_cast<oa::F32>(std::sqrt(x * x + y * y + z * z));
		};
		oa::F32 maximumFootImpulse = 0.0F;
		for (oa::U32 foot = 0U; foot < 4U; ++foot) {
			maximumFootImpulse = std::max(
				maximumFootImpulse,
				stateF32[f32Base + kLunarVectorStateFootImpulses + foot]);
		}
		const oa::U32 rawEndReason =
			stateU32[u32Base + kLunarVectorStateEndReason];
		const oa::U32 rawTerminated =
			stateU32[u32Base + kLunarVectorStateTerminated];
		const oa::U32 rawTruncated =
			stateU32[u32Base + kLunarVectorStateTruncated];
		const bool completed = rawTerminated != 0U or rawTruncated != 0U;
		const bool truncatedByReason = rawEndReason == static_cast<oa::U32>(
			LunarEndReason::TimeLimit)
			or rawEndReason == static_cast<oa::U32>(
				LunarEndReason::ExternalStop);
		const bool terminatedByReason = rawEndReason != static_cast<oa::U32>(
			LunarEndReason::None) and not truncatedByReason;
		if (rawEndReason > static_cast<oa::U32>(LunarEndReason::InvalidAction)
			or rawTerminated > 1U or rawTruncated > 1U
			or (rawTerminated != 0U and rawTruncated != 0U)
			or completed != (rawEndReason != static_cast<oa::U32>(
				LunarEndReason::None))
			or (rawTerminated != 0U) != terminatedByReason
			or (rawTruncated != 0U) != truncatedByReason) {
			return oa::Status::error(
				oa::StatusCode::DataLoss,
				"Lunar Lander 3D telemetry contains invalid terminal state");
		}
		LunarLander3dEpisodeTelemetry telemetry{
			.episodeReturn_ = stateF32[
				f32Base + kLunarVectorStateEpisodeReturn],
			.fuelRemaining_ = stateF32[
				f32Base + kLunarVectorStateFuel],
			.terminalLinearSpeed_ = vectorLength(
				kLunarVectorStateLinearVelocity),
			.terminalAngularSpeed_ = vectorLength(
				kLunarVectorStateAngularVelocity),
			.maximumFootImpulse_ = maximumFootImpulse,
			.episodeStep_ = stateU32[
				u32Base + kLunarVectorStateEpisodeStep],
			.terminated_ = rawTerminated != 0U,
			.truncated_ = rawTruncated != 0U,
			.endReason_ = static_cast<LunarEndReason>(rawEndReason),
		};
		if (not telemetry.isFinite() or telemetry.fuelRemaining_ < 0.0F
			or telemetry.terminalLinearSpeed_ < 0.0F
			or telemetry.terminalAngularSpeed_ < 0.0F
			or telemetry.maximumFootImpulse_ < 0.0F) {
			return oa::Status::error(
				oa::StatusCode::DataLoss,
				"Lunar Lander 3D telemetry contains a non-finite value");
		}
		result.pushBack(telemetry);
	}
	return result;
}

oa::Status LunarLander3dVector::recordReset_(bool inOnlyCompleted) {
	if (not isValid()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D reset requires a valid environment");
	}
	if (inOnlyCompleted
		and not hasCommittedState_
		and not hasPendingFullReset_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D completed reset requires submitted state or an earlier full reset in this transaction");
	}
	const oa::U64 seed = effectiveSeed_();
	if (not inOnlyCompleted) hasPendingFullReset_ = true;
	class Push {
	public:
		oa::U32 environments;
		oa::U32 seedLow;
		oa::U32 seedHigh;
		oa::U32 onlyCompleted;
	};
	const Push push{
		.environments = config_.environments_,
		.seedLow = static_cast<oa::U32>(seed),
		.seedHigh = static_cast<oa::U32>(seed >> 32U),
		.onlyCompleted = inOnlyCompleted ? 1U : 0U,
	};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
		oa::BufferAccess::ReadWrite,
		oa::BufferAccess::Write,
		oa::BufferAccess::Write,
	};
	auto& context = oa::EnvironmentExecutionAccess::session(*this);
	const auto semantic = context.recordOp(
		oa::detail::opRegistry::FnEnvironment::lunarLander3dReset,
		{&configF32_, &configU32_, &terrainF32_, &stateF32_, &stateU32_,
		 &observation_, &endReason_},
		{&stateF32_, &stateU32_, &observation_, &endReason_},
		{
			oa::OpAttribute::fromUnsignedInteger(
				"environmentVersion", config_.environment_.environmentVersion_),
			oa::OpAttribute::fromUnsignedInteger(
				"stateLayoutVersion", kLunarVectorStateLayoutVersion),
			oa::OpAttribute::fromUnsignedInteger("seed", seed),
			oa::OpAttribute::fromBoolean(
				"onlyCompleted", inOnlyCompleted),
		});
	if (semantic.isError()) return semantic.getStatus();
	context.add(
		"RlLunarLander3dReset",
		{&configF32_, &configU32_, &terrainF32_, &stateF32_, &stateU32_,
		 &observation_, &endReason_},
		access, &push, sizeof(push),
		1U + (config_.environments_ - 1U) / 256U, 1U, 1U,
		oa::detail::opRegistry::FnEnvironment::lunarLander3dReset.name, 0,
		oa::detail::opRegistry::FnEnvironment::lunarLander3dReset.hash, 0, 0,
		semantic.getValue());
	return oa::Status::ok();
}

oa::Status LunarLander3dVector::reset() {
	return oa::Environment::reset(effectiveSeed_());
}

oa::Status LunarLander3dVector::resetDone() {
	return resetCompleted();
}

oa::Status LunarLander3dVector::recordReset_(oa::U64 inSeed) {
	pendingSeed_ = inSeed;
	hasPendingSeed_ = true;
	return recordReset_(false);
}

oa::Status LunarLander3dVector::recordResetCompleted_() {
	return recordReset_(true);
}

oa::Result<oa::EnvironmentTransition>
LunarLander3dVector::recordStep_(const oa::Matrix& inAction) {
	auto step = recordStep_(inAction, noExternalStop_);
	if (step.isError()) return step.getStatus();
	return oa::EnvironmentTransition{
		.observation = step->observation_,
		.nextObservation = step->nextObservation_,
		.reward = step->reward_,
		.terminated = step->terminated_,
		.truncated = step->truncated_,
	};
}

oa::Result<LunarLander3dVectorStep> LunarLander3dVector::step(
	const oa::Matrix& inAction) {
	auto transition = oa::Environment::step(inAction);
	if (transition.isError()) return transition.getStatus();
	return LunarLander3dVectorStep{
		.observation_ = transition->observation,
		.nextObservation_ = transition->nextObservation,
		.reward_ = transition->reward,
		.terminated_ = transition->terminated,
		.truncated_ = transition->truncated,
		.endReason_ = endReason_,
	};
}

oa::Result<LunarLander3dVectorStep> LunarLander3dVector::step(
	const oa::Matrix& inAction,
	const oa::Matrix& inExternalStop) {
	LunarLander3dVectorStep step;
	const oa::Status recorded = recordCommands([&]() -> oa::Status {
		auto result = recordStep_(inAction, inExternalStop);
		if (result.isError()) return result.getStatus();
		step = oa::move(*result);
		return oa::Status::ok();
	});
	if (recorded.isError()) return recorded;
	return step;
}

oa::Result<LunarLander3dVectorStep> LunarLander3dVector::recordStep_(
	const oa::Matrix& inAction,
	const oa::Matrix& inExternalStop) {
	if (not isValid()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D step requires a valid environment");
	}
	if (not hasCommittedState_ and not hasPendingFullReset_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D step requires a submitted state or an earlier full reset in this transaction");
	}
	OA_RETURN_IF_ERROR(spec_.validateAction(inAction, config_.environments_));
	OA_RETURN_IF_ERROR(oa::EnvironmentSpace::binary("external_stop").validateMatrix(
		inExternalStop, config_.environments_));

	class Push {
	public:
		oa::U32 environments;
	};
	const Push push{.environments = config_.environments_};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::ReadWrite,
		oa::BufferAccess::ReadWrite,
		oa::BufferAccess::Write,
		oa::BufferAccess::ReadWrite,
		oa::BufferAccess::Write,
		oa::BufferAccess::Write,
		oa::BufferAccess::Write,
		oa::BufferAccess::Write,
	};
	auto& context = oa::EnvironmentExecutionAccess::session(*this);
	const LunarLander3dConfig& config = config_.environment_;
	const auto semantic = context.recordOp(
		oa::detail::opRegistry::FnEnvironment::lunarLander3dStep,
		{&inAction, &inExternalStop, &configF32_, &configU32_, &terrainF32_,
		 &stateF32_, &stateU32_, &observation_},
		{&stateF32_, &stateU32_, &transitionObservation_, &observation_,
		 &reward_, &terminated_, &truncated_, &endReason_},
		{
			oa::OpAttribute::fromUnsignedInteger(
				"environmentVersion", config.environmentVersion_),
			oa::OpAttribute::fromUnsignedInteger(
				"physicsVersion", config.physicsVersion_),
			oa::OpAttribute::fromUnsignedInteger(
				"observationVersion", config.observationVersion_),
			oa::OpAttribute::fromUnsignedInteger(
				"rewardVersion", config.rewardVersion_),
			oa::OpAttribute::fromUnsignedInteger(
				"stateLayoutVersion", kLunarVectorStateLayoutVersion),
			oa::OpAttribute::fromUnsignedInteger(
				"configIdentity", config.contractFingerprint()),
			oa::OpAttribute::fromUnsignedInteger(
				"maxEpisodeSteps", config.maxEpisodeSteps_),
			oa::OpAttribute::fromFloat(
				"failurePenalty", config.failurePenalty_),
		});
	if (semantic.isError()) return semantic.getStatus();
	context.add(
		"RlLunarLander3dStep",
		{&inAction, &inExternalStop, &configF32_, &configU32_, &terrainF32_,
		 &stateF32_, &stateU32_, &transitionObservation_, &observation_,
		 &reward_, &terminated_, &truncated_, &endReason_},
		access, &push, sizeof(push),
		1U + (config_.environments_ - 1U) / 256U, 1U, 1U,
		oa::detail::opRegistry::FnEnvironment::lunarLander3dStep.name, 0,
		oa::detail::opRegistry::FnEnvironment::lunarLander3dStep.hash, 0, 0,
		semantic.getValue());
	return LunarLander3dVectorStep{
		.observation_ = transitionObservation_,
		.nextObservation_ = observation_,
		.reward_ = reward_,
		.terminated_ = terminated_,
		.truncated_ = truncated_,
		.endReason_ = endReason_,
	};
}

void LunarLander3dVector::commitRecordedState_() noexcept {
	if (hasPendingSeed_) config_.seed_ = pendingSeed_;
	if (hasPendingFullReset_) hasCommittedState_ = true;
	hasPendingSeed_ = false;
	hasPendingFullReset_ = false;
}

void LunarLander3dVector::rollbackRecordedState_() noexcept {
	hasPendingSeed_ = false;
	hasPendingFullReset_ = false;
}

} // namespace oa
