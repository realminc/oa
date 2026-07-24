#include "LunarLander3dVector.h"

#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Operation.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/OaVk.h>

#include "../../../../Source/Private/Oa/Core/OperationRegistry.gen.h"
#include "../../../../Source/Private/Oa/Ml/Rl/EnvironmentExecution.h"

#include <algorithm>
#include <cmath>
#include <limits>

enum class OaLunarVectorConfigF32Index : OaU32 {
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

static constexpr OaU32 OA_LUNAR_VECTOR_CONFIG_F32_COUNT = 75U;
static constexpr OaU32 OA_LUNAR_VECTOR_CONFIG_U32_COUNT = 17U;
static constexpr OaU32 OA_LUNAR_VECTOR_STATE_F32_WIDTH = 32U;
static constexpr OaU32 OA_LUNAR_VECTOR_STATE_U32_WIDTH = 16U;
static constexpr OaU32 OA_LUNAR_VECTOR_STATE_LINEAR_VELOCITY = 3U;
static constexpr OaU32 OA_LUNAR_VECTOR_STATE_ANGULAR_VELOCITY = 10U;
static constexpr OaU32 OA_LUNAR_VECTOR_STATE_FUEL = 13U;
static constexpr OaU32 OA_LUNAR_VECTOR_STATE_FOOT_IMPULSES = 21U;
static constexpr OaU32 OA_LUNAR_VECTOR_STATE_EPISODE_RETURN = 25U;
static constexpr OaU32 OA_LUNAR_VECTOR_STATE_EPISODE_STEP = 5U;
static constexpr OaU32 OA_LUNAR_VECTOR_STATE_TERMINATED = 7U;
static constexpr OaU32 OA_LUNAR_VECTOR_STATE_TRUNCATED = 8U;
static constexpr OaU32 OA_LUNAR_VECTOR_STATE_END_REASON = 9U;

static OaUsize OaLunarVectorConfigIndex(
	OaLunarVectorConfigF32Index InIndex) noexcept {
	return static_cast<OaUsize>(InIndex);
}

static OaResult<OaVec<OaF32>> OaLunarVectorSerializeConfigF32(
	const OaLunarLander3dConfig& InConfig,
	const OaLunarTerrain& InTerrain) {
	OaVec<OaF32> values(OA_LUNAR_VECTOR_CONFIG_F32_COUNT, 0.0F);
	bool losesNonzero = false;
	bool requiresSubnormal = false;
	auto setValue = [&values, &losesNonzero, &requiresSubnormal](
		OaUsize InIndex, double InValue) {
		const OaF32 converted = static_cast<OaF32>(InValue);
		values[InIndex] = converted;
		losesNonzero = losesNonzero
			or (InValue != 0.0 and converted == 0.0F);
		requiresSubnormal = requiresSubnormal
			or (converted != 0.0F
				and std::fpclassify(converted) == FP_SUBNORMAL);
	};
	auto set = [&setValue](
		OaLunarVectorConfigF32Index InIndex, double InValue) {
		setValue(OaLunarVectorConfigIndex(InIndex), InValue);
	};
	set(OaLunarVectorConfigF32Index::PolicyTimeStep, InConfig.PolicyTimeStep_);
	set(OaLunarVectorConfigF32Index::Gravity, InConfig.Gravity_);
	set(OaLunarVectorConfigF32Index::Mass, InConfig.Mass_);
	set(OaLunarVectorConfigF32Index::InertiaX,
		InConfig.DiagonalInertia_.ComponentX_);
	set(OaLunarVectorConfigF32Index::InertiaY,
		InConfig.DiagonalInertia_.ComponentY_);
	set(OaLunarVectorConfigF32Index::InertiaZ,
		InConfig.DiagonalInertia_.ComponentZ_);
	set(OaLunarVectorConfigF32Index::MainThrust, InConfig.MainThrust_);
	set(OaLunarVectorConfigF32Index::AttitudeTorque, InConfig.AttitudeTorque_);
	set(OaLunarVectorConfigF32Index::FuelCapacity, InConfig.FuelCapacity_);
	set(OaLunarVectorConfigF32Index::MainFuelRate, InConfig.MainFuelRate_);
	set(OaLunarVectorConfigF32Index::AttitudeFuelRate,
		InConfig.AttitudeFuelRate_);
	set(OaLunarVectorConfigF32Index::Restitution, InConfig.Restitution_);
	set(OaLunarVectorConfigF32Index::Friction, InConfig.Friction_);
	set(OaLunarVectorConfigF32Index::ContactSlop, InConfig.ContactSlop_);
	set(OaLunarVectorConfigF32Index::PenetrationCorrectionFraction,
		InConfig.PenetrationCorrectionFraction_);
	set(OaLunarVectorConfigF32Index::MaxPositionCorrectionPerContact,
		InConfig.MaxPositionCorrectionPerContact_);
	set(OaLunarVectorConfigF32Index::MaxContactImpulse,
		InConfig.MaxContactImpulse_);
	set(OaLunarVectorConfigF32Index::MaxBiasSpeed, InConfig.MaxBiasSpeed_);
	set(OaLunarVectorConfigF32Index::TaskMinimumY, InConfig.TaskMinimumY_);
	set(OaLunarVectorConfigF32Index::TaskMaximumY, InConfig.TaskMaximumY_);
	set(OaLunarVectorConfigF32Index::SafeLinearSpeed,
		InConfig.SafeLinearSpeed_);
	set(OaLunarVectorConfigF32Index::SafeAngularSpeed,
		InConfig.SafeAngularSpeed_);
	set(OaLunarVectorConfigF32Index::SafeTiltRadians,
		InConfig.SafeTiltRadians_);
	set(OaLunarVectorConfigF32Index::HardFootImpactSpeed,
		InConfig.HardFootImpactSpeed_);
	set(OaLunarVectorConfigF32Index::PositionObservationScale,
		InConfig.PositionObservationScale_);
	set(OaLunarVectorConfigF32Index::VelocityObservationScale,
		InConfig.VelocityObservationScale_);
	set(OaLunarVectorConfigF32Index::AngularVelocityObservationScale,
		InConfig.AngularVelocityObservationScale_);
	set(OaLunarVectorConfigF32Index::TerrainClearanceObservationScale,
		InConfig.TerrainClearanceObservationScale_);
	set(OaLunarVectorConfigF32Index::FootClearanceObservationScale,
		InConfig.FootClearanceObservationScale_);
	set(OaLunarVectorConfigF32Index::TerrainProbeSpacing,
		InConfig.TerrainProbeSpacing_);
	set(OaLunarVectorConfigF32Index::RewardGamma, InConfig.RewardGamma_);
	set(OaLunarVectorConfigF32Index::PositionPotentialWeight,
		InConfig.PositionPotentialWeight_);
	set(OaLunarVectorConfigF32Index::VelocityPotentialWeight,
		InConfig.VelocityPotentialWeight_);
	set(OaLunarVectorConfigF32Index::TiltPotentialWeight,
		InConfig.TiltPotentialWeight_);
	set(OaLunarVectorConfigF32Index::AngularPotentialWeight,
		InConfig.AngularPotentialWeight_);
	set(OaLunarVectorConfigF32Index::MainFuelCostWeight,
		InConfig.MainFuelCostWeight_);
	set(OaLunarVectorConfigF32Index::AttitudeFuelCostWeight,
		InConfig.AttitudeFuelCostWeight_);
	set(OaLunarVectorConfigF32Index::SoftFootContactReward,
		InConfig.SoftFootContactReward_);
	set(OaLunarVectorConfigF32Index::StableDwellReward,
		InConfig.StableDwellReward_);
	set(OaLunarVectorConfigF32Index::SuccessReward, InConfig.SuccessReward_);
	set(OaLunarVectorConfigF32Index::FailurePenalty, InConfig.FailurePenalty_);
	set(OaLunarVectorConfigF32Index::TerrainCellSize,
		InConfig.Terrain_.CellSize_);
	set(OaLunarVectorConfigF32Index::TerrainMinimumX, InTerrain.MinX());
	set(OaLunarVectorConfigF32Index::TerrainMaximumX, InTerrain.MaxX());
	set(OaLunarVectorConfigF32Index::TerrainMinimumZ, InTerrain.MinZ());
	set(OaLunarVectorConfigF32Index::TerrainMaximumZ, InTerrain.MaxZ());
	set(OaLunarVectorConfigF32Index::PadHalfExtent,
		InConfig.Terrain_.PadHalfExtent_);

	OaU32 supportOffset = static_cast<OaU32>(
		OaLunarVectorConfigF32Index::SupportBase);
	auto appendSupport = [&setValue, &supportOffset](
		const OaLunarSupportSphere& InSupport) {
		setValue(supportOffset++, InSupport.BodyOffset_.ComponentX_);
		setValue(supportOffset++, InSupport.BodyOffset_.ComponentY_);
		setValue(supportOffset++, InSupport.BodyOffset_.ComponentZ_);
		setValue(supportOffset++, InSupport.Radius_);
	};
	for (const OaLunarSupportSphere& support : InConfig.BodySupports_) {
		appendSupport(support);
	}
	for (const OaLunarSupportSphere& support : InConfig.FootSupports_) {
		appendSupport(support);
	}
	if (losesNonzero or requiresSubnormal) {
		return OaStatus::Error(
			OaStatusCode::OutOfRange,
			"Lunar Lander 3D configuration contains a value that underflows or requires FP32 subnormal preservation");
	}
	return values;
}

static OaVec<OaU32> OaLunarVectorSerializeConfigU32(
	const OaLunarLander3dConfig& InConfig) {
	OaVec<OaU32> values(OA_LUNAR_VECTOR_CONFIG_U32_COUNT, 0U);
	const OaU64 fingerprint = InConfig.ContractFingerprint();
	values[0] = OA_LUNAR_VECTOR_CONFIG_LAYOUT_VERSION;
	values[1] = InConfig.EnvironmentVersion_;
	values[2] = OA_LUNAR_RANDOM_VERSION;
	values[3] = OA_LUNAR_TERRAIN_VERSION;
	values[4] = InConfig.PhysicsVersion_;
	values[5] = InConfig.ObservationVersion_;
	values[6] = InConfig.RewardVersion_;
	values[7] = InConfig.PhysicsSubsteps_;
	values[8] = InConfig.ContactIterations_;
	values[9] = InConfig.SafeDwellSteps_;
	values[10] = InConfig.MaxEpisodeSteps_;
	values[11] = InConfig.Terrain_.CellsX_;
	values[12] = InConfig.Terrain_.CellsZ_;
	values[13] = static_cast<OaU32>(fingerprint);
	values[14] = static_cast<OaU32>(fingerprint >> 32U);
	values[15] = OA_LUNAR_VECTOR_STATE_F32_WIDTH;
	values[16] = OA_LUNAR_VECTOR_STATE_U32_WIDTH;
	return values;
}

static bool OaLunarVectorAllFinite(const OaVec<OaF32>& InValues) noexcept {
	for (const OaF32 value : InValues) {
		if (not std::isfinite(value)) return false;
	}
	return true;
}

static OaStatus OaLunarVectorValidateSerializedConfig(
	const OaVec<OaF32>& InValues,
	const OaLunarLander3dConfig& InConfig) {
	if (InValues.Size() != OA_LUNAR_VECTOR_CONFIG_F32_COUNT
		or not OaLunarVectorAllFinite(InValues)) {
		return OaStatus::Error(
			OaStatusCode::OutOfRange,
			"Lunar Lander 3D configuration cannot be represented as finite FP32");
	}
	const auto get = [&InValues](OaLunarVectorConfigF32Index InIndex) {
		return InValues[OaLunarVectorConfigIndex(InIndex)];
	};
	const auto safeDenominator = [&get](
		OaLunarVectorConfigF32Index InIndex) {
		const OaF32 value = get(InIndex);
		return value > 0.0F and std::isfinite(1.0F / value);
	};
	const bool positiveDenominators =
		safeDenominator(OaLunarVectorConfigF32Index::PolicyTimeStep)
		and safeDenominator(OaLunarVectorConfigF32Index::Mass)
		and safeDenominator(OaLunarVectorConfigF32Index::InertiaX)
		and safeDenominator(OaLunarVectorConfigF32Index::InertiaY)
		and safeDenominator(OaLunarVectorConfigF32Index::InertiaZ)
		and safeDenominator(OaLunarVectorConfigF32Index::FuelCapacity)
		and get(OaLunarVectorConfigF32Index::MaxContactImpulse) > 0.0F
		and safeDenominator(
			OaLunarVectorConfigF32Index::PositionObservationScale)
		and safeDenominator(
			OaLunarVectorConfigF32Index::VelocityObservationScale)
		and safeDenominator(
			OaLunarVectorConfigF32Index::AngularVelocityObservationScale)
		and safeDenominator(
			OaLunarVectorConfigF32Index::TerrainClearanceObservationScale)
		and safeDenominator(
			OaLunarVectorConfigF32Index::FootClearanceObservationScale)
		and get(OaLunarVectorConfigF32Index::TerrainProbeSpacing) > 0.0F
		and safeDenominator(OaLunarVectorConfigF32Index::TerrainCellSize);
	const bool orderedBounds =
		get(OaLunarVectorConfigF32Index::TaskMinimumY)
			< get(OaLunarVectorConfigF32Index::TaskMaximumY)
		and get(OaLunarVectorConfigF32Index::TerrainMinimumX)
			< get(OaLunarVectorConfigF32Index::TerrainMaximumX)
		and get(OaLunarVectorConfigF32Index::TerrainMinimumZ)
			< get(OaLunarVectorConfigF32Index::TerrainMaximumZ);
	const OaF32 substepTime =
		get(OaLunarVectorConfigF32Index::PolicyTimeStep)
		/ static_cast<OaF32>(InConfig.PhysicsSubsteps_);
	const bool representableSubstep = substepTime > 0.0F
		and std::isfinite(1.0F / substepTime)
		and std::fpclassify(substepTime) != FP_SUBNORMAL;
	const OaF32 fuelCapacity = get(
		OaLunarVectorConfigF32Index::FuelCapacity);
	const auto representableFuelDebit = [substepTime, fuelCapacity](
		OaF32 InFuelRate) {
		if (InFuelRate == 0.0F) return true;
		const OaF32 debit = InFuelRate * substepTime;
		return debit > 0.0F and std::isfinite(debit)
			and std::fpclassify(debit) != FP_SUBNORMAL
			and fuelCapacity - debit != fuelCapacity;
	};
	const bool representableFuelRates = representableSubstep
		and representableFuelDebit(get(
			OaLunarVectorConfigF32Index::MainFuelRate))
		and representableFuelDebit(get(
			OaLunarVectorConfigF32Index::AttitudeFuelRate));
	bool representableSupportRadii = true;
	OaU32 supportOffset = static_cast<OaU32>(
		OaLunarVectorConfigF32Index::SupportBase);
	for (OaU32 support = 0U; support < 7U; ++support) {
		representableSupportRadii = representableSupportRadii
			and InValues[supportOffset + support * 4U + 3U] > 0.0F;
	}
	const double potentialMagnitude =
		static_cast<double>(get(
			OaLunarVectorConfigF32Index::PositionPotentialWeight))
		+ static_cast<double>(get(
			OaLunarVectorConfigF32Index::VelocityPotentialWeight))
		+ static_cast<double>(get(
			OaLunarVectorConfigF32Index::TiltPotentialWeight))
		+ static_cast<double>(get(
			OaLunarVectorConfigF32Index::AngularPotentialWeight));
	const double fuelCostMagnitude =
		static_cast<double>(get(OaLunarVectorConfigF32Index::FuelCapacity))
		* (static_cast<double>(get(
			OaLunarVectorConfigF32Index::MainFuelCostWeight))
			+ static_cast<double>(get(
				OaLunarVectorConfigF32Index::AttitudeFuelCostWeight)));
	const double terminalMagnitude = std::max(
		static_cast<double>(get(OaLunarVectorConfigF32Index::SuccessReward)),
		std::abs(static_cast<double>(get(
			OaLunarVectorConfigF32Index::FailurePenalty))));
	const double maximumRewardMagnitude =
		(1.0 + static_cast<double>(get(
			OaLunarVectorConfigF32Index::RewardGamma))) * potentialMagnitude
		+ fuelCostMagnitude
		+ 4.0 * static_cast<double>(get(
			OaLunarVectorConfigF32Index::SoftFootContactReward))
		+ static_cast<double>(get(
			OaLunarVectorConfigF32Index::StableDwellReward))
		+ terminalMagnitude;
	const double maximumEpisodeReturnMagnitude = maximumRewardMagnitude
		* static_cast<double>(InConfig.MaxEpisodeSteps_);
	const double maximumFp32 =
		static_cast<double>(std::numeric_limits<OaF32>::max());
	const bool boundedRewards = std::isfinite(maximumRewardMagnitude)
		and maximumRewardMagnitude <= maximumFp32
		and std::isfinite(maximumEpisodeReturnMagnitude)
		and maximumEpisodeReturnMagnitude <= maximumFp32;
	if (not positiveDenominators or not orderedBounds
		or not representableFuelRates or not representableSupportRadii
		or not boundedRewards) {
		return OaStatus::Error(
			OaStatusCode::OutOfRange,
			"Lunar Lander 3D configuration loses required FP32 relationships or reward bounds");
	}
	return OaStatus::Ok();
}

static OaStatus OaLunarVectorValidateDeviceLimits(
	OaEngine& InEngine,
	OaU32 InEnvironments,
	OaU64 InTerrainVertices) {
	if (not InEngine.Device.Instance or not InEngine.Device.PhysicalDevice) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D requires queried Vulkan device limits");
	}
	OaVkInstanceTable instanceTable{};
	OaVkLoadInstanceTable(
		&instanceTable,
		static_cast<VkInstance>(InEngine.Device.Instance));
	if (not instanceTable.vkGetPhysicalDeviceProperties
		or not instanceTable.vkGetPhysicalDeviceFeatures2) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D requires Vulkan instance capability queries");
	}
	const OaU32 apiVersion = InEngine.Device.Info.Software.ApiVersionPacked;
	const bool core12 = VK_API_VERSION_MAJOR(apiVersion) > 1U
		or (VK_API_VERSION_MAJOR(apiVersion) == 1U
			and VK_API_VERSION_MINOR(apiVersion) >= 2U);
	if (not core12) {
		return OaStatus::Error(
			OaStatusCode::Unavailable,
			"Lunar Lander 3D UInt8 transition outputs require Vulkan 1.2 storage-8 capabilities enabled by OaEngine");
	}
	VkPhysicalDeviceVulkan12Features features12{};
	features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	VkPhysicalDeviceFeatures2 features2{};
	features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	features2.pNext = &features12;
	instanceTable.vkGetPhysicalDeviceFeatures2(
		static_cast<VkPhysicalDevice>(InEngine.Device.PhysicalDevice), &features2);
	if (features12.shaderInt8 != VK_TRUE
		or features12.uniformAndStorageBuffer8BitAccess != VK_TRUE) {
		return OaStatus::Error(
			OaStatusCode::Unavailable,
			"Lunar Lander 3D UInt8 transition outputs require shaderInt8 and uniformAndStorageBuffer8BitAccess");
	}
	VkPhysicalDeviceProperties properties{};
	instanceTable.vkGetPhysicalDeviceProperties(
		static_cast<VkPhysicalDevice>(InEngine.Device.PhysicalDevice),
		&properties);
	if (properties.limits.maxComputeWorkGroupInvocations < 256U
		or properties.limits.maxComputeWorkGroupSize[0] < 256U) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D requires a queried 256-thread compute workgroup");
	}
	const OaU64 groupsX = 1U + (static_cast<OaU64>(InEnvironments) - 1U) / 256U;
	if (groupsX > properties.limits.maxComputeWorkGroupCount[0]) {
		return OaStatus::Error(
			OaStatusCode::OutOfRange,
			"Lunar Lander 3D lane count exceeds this device's queried dispatch-X limit");
	}
	const OaU64 maximumBufferBytes = std::min<OaU64>(
		properties.limits.maxStorageBufferRange,
		std::numeric_limits<OaU32>::max());
	auto validateBuffer = [maximumBufferBytes](
		OaU64 InElements, OaU64 InElementBytes, OaStringView InName) -> OaStatus {
		if (InElements > std::numeric_limits<OaU64>::max() / InElementBytes
			or InElements * InElementBytes > maximumBufferBytes) {
			return OaStatus::Error(
				OaStatusCode::OutOfRange,
				OaString(InName) + " exceeds this device's queried storage-buffer range");
		}
		return OaStatus::Ok();
	};
	OA_RETURN_IF_ERROR(validateBuffer(
		static_cast<OaU64>(InEnvironments) * OA_LUNAR_VECTOR_STATE_F32_WIDTH,
		sizeof(OaF32), "Lunar Lander 3D FP32 state"));
	OA_RETURN_IF_ERROR(validateBuffer(
		static_cast<OaU64>(InEnvironments) * OA_LUNAR_VECTOR_STATE_U32_WIDTH,
		sizeof(OaU32), "Lunar Lander 3D UInt32 state"));
	OA_RETURN_IF_ERROR(validateBuffer(
		static_cast<OaU64>(InEnvironments) * OA_LUNAR_OBSERVATION_SIZE,
		sizeof(OaF32), "Lunar Lander 3D observation"));
	OA_RETURN_IF_ERROR(validateBuffer(
		InTerrainVertices, sizeof(OaF32), "Lunar Lander 3D terrain"));
	OA_RETURN_IF_ERROR(validateBuffer(
		InEnvironments, sizeof(OaU32), "Lunar Lander 3D lane vector"));
	return OaStatus::Ok();
}

static OaMatrix OaLunarVectorFromF32(const OaVec<OaF32>& InValues) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(InValues.Data()),
			InValues.Size() * sizeof(OaF32)),
		{static_cast<OaI64>(InValues.Size())}, OaScalarType::Float32);
}

static OaMatrix OaLunarVectorFromU32(const OaVec<OaU32>& InValues) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(InValues.Data()),
			InValues.Size() * sizeof(OaU32)),
		{static_cast<OaI64>(InValues.Size())}, OaScalarType::UInt32);
}

bool OaLunarLander3dVectorStep::IsValid() const noexcept {
	return not Observation_.IsEmpty() and not NextObservation_.IsEmpty()
		and not Reward_.IsEmpty() and not Terminated_.IsEmpty()
		and not Truncated_.IsEmpty() and not EndReason_.IsEmpty();
}

bool OaLunarLander3dEpisodeTelemetry::IsFinite() const noexcept {
	return std::isfinite(EpisodeReturn_) and std::isfinite(FuelRemaining_)
		and std::isfinite(TerminalLinearSpeed_)
		and std::isfinite(TerminalAngularSpeed_)
		and std::isfinite(MaximumFootImpulse_);
}

OaLunarLander3dVector::OaLunarLander3dVector(OaEngine& InEngine)
	: OaRlEnvironment(InEngine) {}

OaU64 OaLunarLander3dVector::EffectiveSeed_() const noexcept {
	return HasPendingSeed_ ? PendingSeed_ : Config_.Seed_;
}

OaResult<OaLunarLander3dVector> OaLunarLander3dVector::CreateFlat(
	OaEngine& InEngine,
	const OaLunarLander3dVectorConfig& InConfig) {
	if (not InEngine.IsReady()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D requires a ready engine");
	}
	if (InConfig.Environments_ == 0U) {
		return OaStatus::InvalidArgument(
			"Lunar Lander 3D requires at least one environment lane");
	}
	const OaString configError(InConfig.Environment_.ValidationError());
	if (not configError.Empty()) return OaStatus::InvalidArgument(configError);
	const OaLunarTerrain terrain = OaLunarTerrain::CreateFlat(
		InConfig.Environment_.Terrain_);
	if (not terrain.IsValid()) {
		return OaStatus::InvalidArgument(OaString(terrain.Error()));
	}
	const OaU64 terrainVertices = static_cast<OaU64>(terrain.Heights().size());
	auto serializedConfig = OaLunarVectorSerializeConfigF32(
		InConfig.Environment_, terrain);
	if (serializedConfig.IsError()) return serializedConfig.GetStatus();
	OaVec<OaF32> configF32 = OaStdMove(serializedConfig).GetValue();
	OA_RETURN_IF_ERROR(OaLunarVectorValidateSerializedConfig(
		configF32, InConfig.Environment_));
	OA_RETURN_IF_ERROR(OaLunarVectorValidateDeviceLimits(
		InEngine, InConfig.Environments_, terrainVertices));
	const OaVec<OaU32> configU32 = OaLunarVectorSerializeConfigU32(
		InConfig.Environment_);
	OaVec<OaF32> terrainF32(
		static_cast<OaUsize>(terrainVertices), 0.0F);
	for (OaUsize index = 0; index < terrainF32.Size(); ++index) {
		terrainF32[index] = static_cast<OaF32>(terrain.Heights()[index]);
	}
	const OaVec<OaU8> noExternalStop(InConfig.Environments_, 0U);

	const OaMatrixShape stateF32Shape{
		static_cast<OaI64>(InConfig.Environments_),
		static_cast<OaI64>(OA_LUNAR_VECTOR_STATE_F32_WIDTH)};
	const OaMatrixShape stateU32Shape{
		static_cast<OaI64>(InConfig.Environments_),
		static_cast<OaI64>(OA_LUNAR_VECTOR_STATE_U32_WIDTH)};
	const OaMatrixShape observationShape{
		static_cast<OaI64>(InConfig.Environments_),
		static_cast<OaI64>(OA_LUNAR_OBSERVATION_SIZE)};
	const OaMatrixShape vectorShape{
		static_cast<OaI64>(InConfig.Environments_)};

	OaLunarLander3dVector result(InEngine);
	if (not result.IsOpen()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D could not open its execution session");
	}
	result.Config_ = InConfig;
	result.Spec_ = {
		.Observation = OaRlFieldSpec::Box(
			"observation", {static_cast<OaI64>(OA_LUNAR_OBSERVATION_SIZE)},
			OaScalarType::Float32, -1.0, 1.0),
		.Action = OaRlFieldSpec::Discrete("action", 8),
		.Reward = OaRlFieldSpec::Box("reward", {}, OaScalarType::Float32),
		.Terminated = OaRlFieldSpec::Binary("terminated"),
		.Truncated = OaRlFieldSpec::Binary("truncated"),
	};
	OA_RETURN_IF_ERROR(result.Spec_.ValidateDefinition());
	const OaStatus initialized = result.RecordCommands([&]() -> OaStatus {
		result.ConfigF32_ = OaLunarVectorFromF32(configF32);
		result.ConfigU32_ = OaLunarVectorFromU32(configU32);
		result.TerrainF32_ = OaLunarVectorFromF32(terrainF32);
		result.StateF32_ = OaFnMatrix::Empty(
			stateF32Shape, OaScalarType::Float32);
		result.StateU32_ = OaFnMatrix::Empty(
			stateU32Shape, OaScalarType::UInt32);
		result.Observation_ = OaFnMatrix::Empty(
			observationShape, OaScalarType::Float32);
		result.TransitionObservation_ = OaFnMatrix::Empty(
			observationShape, OaScalarType::Float32);
		result.Reward_ = OaFnMatrix::Empty(
			vectorShape, OaScalarType::Float32);
		result.Terminated_ = OaFnMatrix::Empty(
			vectorShape, OaScalarType::UInt8);
		result.Truncated_ = OaFnMatrix::Empty(
			vectorShape, OaScalarType::UInt8);
		result.EndReason_ = OaFnMatrix::Empty(
			vectorShape, OaScalarType::UInt32);
		// FromBytes completes this upload before returning, and the member owns
		// the input storage across deferred submission and transaction rollback.
		result.NoExternalStop_ = OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(
				noExternalStop.Data(), noExternalStop.Size()),
			vectorShape, OaScalarType::UInt8);
		if (not result.IsValid()) {
			return OaStatus::Error(
				OaStatusCode::OutOfMemory,
				"Lunar Lander 3D could not allocate bounded device storage");
		}
		return result.RecordReset_(false);
	});
	if (initialized.IsError()) return initialized;
	return result;
}

bool OaLunarLander3dVector::IsValid() const noexcept {
	const OaMatrixShape stateF32Shape{
		static_cast<OaI64>(Config_.Environments_),
		static_cast<OaI64>(OA_LUNAR_VECTOR_STATE_F32_WIDTH)};
	const OaMatrixShape stateU32Shape{
		static_cast<OaI64>(Config_.Environments_),
		static_cast<OaI64>(OA_LUNAR_VECTOR_STATE_U32_WIDTH)};
	const OaMatrixShape observationShape{
		static_cast<OaI64>(Config_.Environments_),
		static_cast<OaI64>(OA_LUNAR_OBSERVATION_SIZE)};
	const OaMatrixShape vectorShape{
		static_cast<OaI64>(Config_.Environments_)};
	const OaMatrixShape configF32Shape{
		static_cast<OaI64>(OA_LUNAR_VECTOR_CONFIG_F32_COUNT)};
	const OaMatrixShape configU32Shape{
		static_cast<OaI64>(OA_LUNAR_VECTOR_CONFIG_U32_COUNT)};
	const OaI64 terrainVertices =
		(static_cast<OaI64>(Config_.Environment_.Terrain_.CellsX_) + 1)
		* (static_cast<OaI64>(Config_.Environment_.Terrain_.CellsZ_) + 1);
	const auto matches = [](const OaMatrix& InMatrix,
		const OaMatrixShape& InShape, OaScalarType InDtype) {
		return not InMatrix.IsEmpty() and InMatrix.GetShape() == InShape
			and InMatrix.GetDtype() == InDtype;
	};
	return matches(ConfigF32_, configF32Shape, OaScalarType::Float32)
		and matches(ConfigU32_, configU32Shape, OaScalarType::UInt32)
		and matches(TerrainF32_, {terrainVertices}, OaScalarType::Float32)
		and matches(StateF32_, stateF32Shape, OaScalarType::Float32)
		and matches(StateU32_, stateU32Shape, OaScalarType::UInt32)
		and matches(Observation_, observationShape, OaScalarType::Float32)
		and matches(
			TransitionObservation_, observationShape, OaScalarType::Float32)
		and matches(Reward_, vectorShape, OaScalarType::Float32)
		and matches(Terminated_, vectorShape, OaScalarType::UInt8)
		and matches(Truncated_, vectorShape, OaScalarType::UInt8)
		and matches(EndReason_, vectorShape, OaScalarType::UInt32)
		and matches(NoExternalStop_, vectorShape, OaScalarType::UInt8);
}

OaResult<OaVec<OaLunarLander3dEpisodeTelemetry>>
OaLunarLander3dVector::CopyEpisodeTelemetry() const {
	if (not IsValid() or not IsOpen()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D telemetry requires a valid open environment");
	}
	if (HasActiveRecording()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D telemetry requires submitting and waiting for, or cancelling, the active recording");
	}
	if (HasPendingEvent()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D telemetry requires waiting on the exact submitted event");
	}
	const OaUsize environments = static_cast<OaUsize>(Config_.Environments_);
	OaVec<OaF32> stateF32(
		environments * OA_LUNAR_VECTOR_STATE_F32_WIDTH, 0.0F);
	OaVec<OaU32> stateU32(
		environments * OA_LUNAR_VECTOR_STATE_U32_WIDTH, 0U);
	{
		// CopyToHost resolves its runtime through the selected context. Select
		// this environment's private execution context so a borrowed non-global
		// engine can never read its buffers through an ambient compatibility
		// context.
		OaRlEnvironmentRecordingScope scope(*this);
		OA_RETURN_IF_ERROR(OaFnMatrix::CopyToHost(
			StateF32_, stateF32.Data(),
			static_cast<OaU64>(stateF32.Size() * sizeof(OaF32))));
		OA_RETURN_IF_ERROR(OaFnMatrix::CopyToHost(
			StateU32_, stateU32.Data(),
			static_cast<OaU64>(stateU32.Size() * sizeof(OaU32))));
	}

	OaVec<OaLunarLander3dEpisodeTelemetry> result;
	result.Reserve(environments);
	for (OaUsize lane = 0U; lane < environments; ++lane) {
		const OaUsize f32Base = lane * OA_LUNAR_VECTOR_STATE_F32_WIDTH;
		const OaUsize u32Base = lane * OA_LUNAR_VECTOR_STATE_U32_WIDTH;
		const auto vectorLength = [&stateF32, f32Base](OaU32 InOffset) {
			const OaF64 x = stateF32[f32Base + InOffset];
			const OaF64 y = stateF32[f32Base + InOffset + 1U];
			const OaF64 z = stateF32[f32Base + InOffset + 2U];
			return static_cast<OaF32>(std::sqrt(x * x + y * y + z * z));
		};
		OaF32 maximumFootImpulse = 0.0F;
		for (OaU32 foot = 0U; foot < 4U; ++foot) {
			maximumFootImpulse = std::max(
				maximumFootImpulse,
				stateF32[f32Base + OA_LUNAR_VECTOR_STATE_FOOT_IMPULSES + foot]);
		}
		const OaU32 rawEndReason =
			stateU32[u32Base + OA_LUNAR_VECTOR_STATE_END_REASON];
		const OaU32 rawTerminated =
			stateU32[u32Base + OA_LUNAR_VECTOR_STATE_TERMINATED];
		const OaU32 rawTruncated =
			stateU32[u32Base + OA_LUNAR_VECTOR_STATE_TRUNCATED];
		const bool completed = rawTerminated != 0U or rawTruncated != 0U;
		const bool truncatedByReason = rawEndReason == static_cast<OaU32>(
			OaLunarEndReason::TimeLimit)
			or rawEndReason == static_cast<OaU32>(
				OaLunarEndReason::ExternalStop);
		const bool terminatedByReason = rawEndReason != static_cast<OaU32>(
			OaLunarEndReason::None) and not truncatedByReason;
		if (rawEndReason > static_cast<OaU32>(OaLunarEndReason::InvalidAction)
			or rawTerminated > 1U or rawTruncated > 1U
			or (rawTerminated != 0U and rawTruncated != 0U)
			or completed != (rawEndReason != static_cast<OaU32>(
				OaLunarEndReason::None))
			or (rawTerminated != 0U) != terminatedByReason
			or (rawTruncated != 0U) != truncatedByReason) {
			return OaStatus::Error(
				OaStatusCode::DataLoss,
				"Lunar Lander 3D telemetry contains invalid terminal state");
		}
		OaLunarLander3dEpisodeTelemetry telemetry{
			.EpisodeReturn_ = stateF32[
				f32Base + OA_LUNAR_VECTOR_STATE_EPISODE_RETURN],
			.FuelRemaining_ = stateF32[
				f32Base + OA_LUNAR_VECTOR_STATE_FUEL],
			.TerminalLinearSpeed_ = vectorLength(
				OA_LUNAR_VECTOR_STATE_LINEAR_VELOCITY),
			.TerminalAngularSpeed_ = vectorLength(
				OA_LUNAR_VECTOR_STATE_ANGULAR_VELOCITY),
			.MaximumFootImpulse_ = maximumFootImpulse,
			.EpisodeStep_ = stateU32[
				u32Base + OA_LUNAR_VECTOR_STATE_EPISODE_STEP],
			.Terminated_ = rawTerminated != 0U,
			.Truncated_ = rawTruncated != 0U,
			.EndReason_ = static_cast<OaLunarEndReason>(rawEndReason),
		};
		if (not telemetry.IsFinite() or telemetry.FuelRemaining_ < 0.0F
			or telemetry.TerminalLinearSpeed_ < 0.0F
			or telemetry.TerminalAngularSpeed_ < 0.0F
			or telemetry.MaximumFootImpulse_ < 0.0F) {
			return OaStatus::Error(
				OaStatusCode::DataLoss,
				"Lunar Lander 3D telemetry contains a non-finite value");
		}
		result.PushBack(telemetry);
	}
	return result;
}

OaStatus OaLunarLander3dVector::RecordReset_(bool InOnlyCompleted) {
	if (not IsValid()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D reset requires a valid environment");
	}
	if (InOnlyCompleted
		and not HasCommittedState_
		and not HasPendingFullReset_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D completed reset requires submitted state or an earlier full reset in this transaction");
	}
	const OaU64 seed = EffectiveSeed_();
	if (not InOnlyCompleted) HasPendingFullReset_ = true;
	class Push {
	public:
		OaU32 Environments;
		OaU32 SeedLow;
		OaU32 SeedHigh;
		OaU32 OnlyCompleted;
	};
	const Push push{
		.Environments = Config_.Environments_,
		.SeedLow = static_cast<OaU32>(seed),
		.SeedHigh = static_cast<OaU32>(seed >> 32U),
		.OnlyCompleted = InOnlyCompleted ? 1U : 0U,
	};
	OaBufferAccess access[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Write,
		OaBufferAccess::ReadWrite,
		OaBufferAccess::Write,
		OaBufferAccess::Write,
	};
	auto& context = OaRlEnvironmentExecutionAccess::Context(*this);
	const auto semantic = context.RecordOperation(
		OaOperationRegistry::LunarLander3dReset,
		{&ConfigF32_, &ConfigU32_, &TerrainF32_, &StateF32_, &StateU32_,
		 &Observation_, &EndReason_},
		{&StateF32_, &StateU32_, &Observation_, &EndReason_},
		{
			OaOperationAttribute::FromUnsignedInteger(
				"EnvironmentVersion", Config_.Environment_.EnvironmentVersion_),
			OaOperationAttribute::FromUnsignedInteger(
				"StateLayoutVersion", OA_LUNAR_VECTOR_STATE_LAYOUT_VERSION),
			OaOperationAttribute::FromUnsignedInteger("Seed", seed),
			OaOperationAttribute::FromBoolean(
				"OnlyCompleted", InOnlyCompleted),
		});
	if (semantic.IsError()) return semantic.GetStatus();
	context.Add(
		"RlLunarLander3dReset",
		{&ConfigF32_, &ConfigU32_, &TerrainF32_, &StateF32_, &StateU32_,
		 &Observation_, &EndReason_},
		access, &push, sizeof(push),
		1U + (Config_.Environments_ - 1U) / 256U, 1U, 1U,
		OaOperationRegistry::LunarLander3dReset.Name, 0,
		OaOperationRegistry::LunarLander3dReset.Hash, 0, 0,
		semantic.GetValue());
	return OaStatus::Ok();
}

OaStatus OaLunarLander3dVector::Reset() {
	return ResetEnvironment(EffectiveSeed_());
}

OaStatus OaLunarLander3dVector::ResetDone() {
	return ResetCompleted();
}

OaStatus OaLunarLander3dVector::RecordResetEnvironment_(OaU64 InSeed) {
	PendingSeed_ = InSeed;
	HasPendingSeed_ = true;
	return RecordReset_(false);
}

OaStatus OaLunarLander3dVector::RecordResetCompleted_() {
	return RecordReset_(true);
}

OaResult<OaRlEnvironmentTransition>
OaLunarLander3dVector::RecordStepEnvironment_(const OaMatrix& InAction) {
	auto step = RecordStep_(InAction, NoExternalStop_);
	if (step.IsError()) return step.GetStatus();
	return OaRlEnvironmentTransition{
		.Observation = step->Observation_,
		.NextObservation = step->NextObservation_,
		.Reward = step->Reward_,
		.Terminated = step->Terminated_,
		.Truncated = step->Truncated_,
	};
}

OaResult<OaLunarLander3dVectorStep> OaLunarLander3dVector::Step(
	const OaMatrix& InAction) {
	auto transition = StepEnvironment(InAction);
	if (transition.IsError()) return transition.GetStatus();
	return OaLunarLander3dVectorStep{
		.Observation_ = transition->Observation,
		.NextObservation_ = transition->NextObservation,
		.Reward_ = transition->Reward,
		.Terminated_ = transition->Terminated,
		.Truncated_ = transition->Truncated,
		.EndReason_ = EndReason_,
	};
}

OaResult<OaLunarLander3dVectorStep> OaLunarLander3dVector::Step(
	const OaMatrix& InAction,
	const OaMatrix& InExternalStop) {
	OaLunarLander3dVectorStep step;
	const OaStatus recorded = RecordCommands([&]() -> OaStatus {
		auto result = RecordStep_(InAction, InExternalStop);
		if (result.IsError()) return result.GetStatus();
		step = OaStdMove(*result);
		return OaStatus::Ok();
	});
	if (recorded.IsError()) return recorded;
	return step;
}

OaResult<OaLunarLander3dVectorStep> OaLunarLander3dVector::RecordStep_(
	const OaMatrix& InAction,
	const OaMatrix& InExternalStop) {
	if (not IsValid()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D step requires a valid environment");
	}
	if (not HasCommittedState_ and not HasPendingFullReset_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D step requires a submitted state or an earlier full reset in this transaction");
	}
	OA_RETURN_IF_ERROR(Spec_.ValidateAction(InAction, Config_.Environments_));
	OA_RETURN_IF_ERROR(OaRlFieldSpec::Binary("external_stop").ValidateMatrix(
		InExternalStop, Config_.Environments_));

	class Push {
	public:
		OaU32 Environments;
	};
	const Push push{.Environments = Config_.Environments_};
	OaBufferAccess access[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::ReadWrite,
		OaBufferAccess::ReadWrite,
		OaBufferAccess::Write,
		OaBufferAccess::ReadWrite,
		OaBufferAccess::Write,
		OaBufferAccess::Write,
		OaBufferAccess::Write,
		OaBufferAccess::Write,
	};
	auto& context = OaRlEnvironmentExecutionAccess::Context(*this);
	const OaLunarLander3dConfig& config = Config_.Environment_;
	const auto semantic = context.RecordOperation(
		OaOperationRegistry::LunarLander3dStep,
		{&InAction, &InExternalStop, &ConfigF32_, &ConfigU32_, &TerrainF32_,
		 &StateF32_, &StateU32_, &Observation_},
		{&StateF32_, &StateU32_, &TransitionObservation_, &Observation_,
		 &Reward_, &Terminated_, &Truncated_, &EndReason_},
		{
			OaOperationAttribute::FromUnsignedInteger(
				"EnvironmentVersion", config.EnvironmentVersion_),
			OaOperationAttribute::FromUnsignedInteger(
				"PhysicsVersion", config.PhysicsVersion_),
			OaOperationAttribute::FromUnsignedInteger(
				"ObservationVersion", config.ObservationVersion_),
			OaOperationAttribute::FromUnsignedInteger(
				"RewardVersion", config.RewardVersion_),
			OaOperationAttribute::FromUnsignedInteger(
				"StateLayoutVersion", OA_LUNAR_VECTOR_STATE_LAYOUT_VERSION),
			OaOperationAttribute::FromUnsignedInteger(
				"ConfigIdentity", config.ContractFingerprint()),
			OaOperationAttribute::FromUnsignedInteger(
				"MaxEpisodeSteps", config.MaxEpisodeSteps_),
			OaOperationAttribute::FromFloat(
				"FailurePenalty", config.FailurePenalty_),
		});
	if (semantic.IsError()) return semantic.GetStatus();
	context.Add(
		"RlLunarLander3dStep",
		{&InAction, &InExternalStop, &ConfigF32_, &ConfigU32_, &TerrainF32_,
		 &StateF32_, &StateU32_, &TransitionObservation_, &Observation_,
		 &Reward_, &Terminated_, &Truncated_, &EndReason_},
		access, &push, sizeof(push),
		1U + (Config_.Environments_ - 1U) / 256U, 1U, 1U,
		OaOperationRegistry::LunarLander3dStep.Name, 0,
		OaOperationRegistry::LunarLander3dStep.Hash, 0, 0,
		semantic.GetValue());
	return OaLunarLander3dVectorStep{
		.Observation_ = TransitionObservation_,
		.NextObservation_ = Observation_,
		.Reward_ = Reward_,
		.Terminated_ = Terminated_,
		.Truncated_ = Truncated_,
		.EndReason_ = EndReason_,
	};
}

void OaLunarLander3dVector::CommitRecordedState_() noexcept {
	if (HasPendingSeed_) Config_.Seed_ = PendingSeed_;
	if (HasPendingFullReset_) HasCommittedState_ = true;
	HasPendingSeed_ = false;
	HasPendingFullReset_ = false;
}

void OaLunarLander3dVector::RollbackRecordedState_() noexcept {
	HasPendingSeed_ = false;
	HasPendingFullReset_ = false;
}
