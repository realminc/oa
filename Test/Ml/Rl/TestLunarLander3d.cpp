#include "LunarLander3d.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <vector>

class OaLunarTraceDigest {
public:
	void AddU64(std::uint64_t InValue) noexcept {
		for (std::uint32_t byteIndex = 0U; byteIndex < 8U; ++byteIndex) {
			const std::uint8_t byte = static_cast<std::uint8_t>(
				(InValue >> (byteIndex * 8U)) & 0xffU);
			Hash_ ^= byte;
			Hash_ *= 1099511628211ULL;
		}
	}
	void AddU32(std::uint32_t InValue) noexcept { AddU64(InValue); }
	void AddBool(bool InValue) noexcept { AddU64(InValue ? 1U : 0U); }
	void AddDouble(double InValue) noexcept {
		// Freeze the numerical trace, not optimizer-specific FP encodings.
		AddU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(
			std::llround(InValue * 1.0e9))));
	}
	void AddFloat(float InValue) noexcept {
		AddU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(
			std::llround(static_cast<double>(InValue) * 1.0e6))));
	}
	[[nodiscard]] std::uint64_t Value() const noexcept { return Hash_; }

private:
	std::uint64_t Hash_ = 1469598103934665603ULL;
};

static void OaLunarDigestVector(
	OaLunarTraceDigest& InOutDigest,
	const OaLunarVec3& InVector) noexcept {
	InOutDigest.AddDouble(InVector.ComponentX_);
	InOutDigest.AddDouble(InVector.ComponentY_);
	InOutDigest.AddDouble(InVector.ComponentZ_);
}

static void OaLunarDigestQuaternion(
	OaLunarTraceDigest& InOutDigest,
	const OaLunarQuat& InQuaternion) noexcept {
	InOutDigest.AddDouble(InQuaternion.Scalar_);
	InOutDigest.AddDouble(InQuaternion.ComponentX_);
	InOutDigest.AddDouble(InQuaternion.ComponentY_);
	InOutDigest.AddDouble(InQuaternion.ComponentZ_);
}

static void OaLunarDigestManifest(
	OaLunarTraceDigest& InOutDigest,
	const OaLunarEpisodeManifest& InManifest) noexcept {
	InOutDigest.AddU32(InManifest.EnvironmentVersion_);
	InOutDigest.AddU32(InManifest.RandomVersion_);
	InOutDigest.AddU32(InManifest.TerrainVersion_);
	InOutDigest.AddU32(InManifest.PhysicsVersion_);
	InOutDigest.AddU32(InManifest.ObservationVersion_);
	InOutDigest.AddU32(InManifest.RewardVersion_);
	InOutDigest.AddU64(InManifest.ConfigFingerprint_);
	InOutDigest.AddU64(InManifest.BaseSeed_);
	InOutDigest.AddU32(InManifest.EnvironmentLane_);
	InOutDigest.AddU64(InManifest.EpisodeIndex_);
	InOutDigest.AddU64(InManifest.TerrainSeed_);
	InOutDigest.AddU64(InManifest.SpawnSeed_);
	InOutDigest.AddU64(InManifest.DomainSeed_);
}

static void OaLunarDigestState(
	OaLunarTraceDigest& InOutDigest,
	const OaLunarLander3dState& InState) noexcept {
	OaLunarDigestVector(InOutDigest, InState.Position_);
	OaLunarDigestVector(InOutDigest, InState.LinearVelocity_);
	OaLunarDigestQuaternion(InOutDigest, InState.Orientation_);
	OaLunarDigestVector(InOutDigest, InState.AngularVelocityBody_);
	InOutDigest.AddDouble(InState.Fuel_);
	InOutDigest.AddU32(static_cast<std::uint32_t>(InState.LastAction_));
	InOutDigest.AddDouble(InState.MainThrottle_);
	OaLunarDigestVector(InOutDigest, InState.AttitudeCommandBody_);
	for (const bool contact : InState.BodyContacts_) {
		InOutDigest.AddBool(contact);
	}
	for (const double impulse : InState.BodyContactImpulses_) {
		InOutDigest.AddDouble(impulse);
	}
	for (const bool contact : InState.FootContacts_) {
		InOutDigest.AddBool(contact);
	}
	for (const bool onPad : InState.FeetOnPad_) {
		InOutDigest.AddBool(onPad);
	}
	for (const double impulse : InState.FootContactImpulses_) {
		InOutDigest.AddDouble(impulse);
	}
	for (const bool rewarded : InState.FootContactRewarded_) {
		InOutDigest.AddBool(rewarded);
	}
	InOutDigest.AddU32(InState.EpisodeStep_);
	InOutDigest.AddU32(InState.StableDwell_);
	InOutDigest.AddBool(InState.Terminated_);
	InOutDigest.AddBool(InState.Truncated_);
	InOutDigest.AddU32(static_cast<std::uint32_t>(InState.EndReason_));
	InOutDigest.AddDouble(InState.EpisodeReturn_);
}

static void OaLunarDigestReward(
	OaLunarTraceDigest& InOutDigest,
	const OaLunarRewardTerms& InReward) noexcept {
	InOutDigest.AddDouble(InReward.PotentialBefore_);
	InOutDigest.AddDouble(InReward.PotentialAfter_);
	InOutDigest.AddDouble(InReward.Shaping_);
	InOutDigest.AddDouble(InReward.MainFuelCost_);
	InOutDigest.AddDouble(InReward.AttitudeFuelCost_);
	InOutDigest.AddDouble(InReward.SoftFootContact_);
	InOutDigest.AddDouble(InReward.StableDwell_);
	InOutDigest.AddDouble(InReward.Terminal_);
	InOutDigest.AddDouble(InReward.Total_);
}

static OaLunarEpisodeManifest OaLunarTestManifest(
	std::uint64_t InSeed = 0x123456789abcdef0ULL,
	std::uint32_t InLane = 3U,
	std::uint64_t InEpisode = 7U) {
	const OaLunarLander3dConfig config;
	return OaLunarEpisodeManifest::Derive(
		InSeed, InLane, InEpisode, config.ContractFingerprint());
}

static OaLunarEpisodeManifest OaLunarTestManifestForConfig(
	const OaLunarLander3dConfig& InConfig,
	std::uint64_t InSeed = 0x123456789abcdef0ULL,
	std::uint32_t InLane = 3U,
	std::uint64_t InEpisode = 7U) {
	return OaLunarEpisodeManifest::DeriveVersioned(
		InSeed, InLane, InEpisode,
		InConfig.EnvironmentVersion_, OA_LUNAR_TERRAIN_VERSION,
		InConfig.PhysicsVersion_, InConfig.ObservationVersion_,
		InConfig.RewardVersion_, InConfig.ContractFingerprint());
}

static OaLunarLander3dState OaLunarTestFlightState(
	const OaLunarLander3dConfig& InConfig,
	double InHeight = 20.0) {
	OaLunarLander3dState state;
	state.Position_ = {0.0, InHeight, 0.0};
	state.Fuel_ = InConfig.FuelCapacity_;
	return state;
}

static double OaLunarTerrainSlope(
	const OaLunarTerrainSample& InSample) {
	return std::hypot(InSample.Normal_.ComponentX_, InSample.Normal_.ComponentZ_)
		/ InSample.Normal_.ComponentY_;
}

TEST(LunarLander3d, RejectsInvalidConfigurationTerrainAndState) {
	OaLunarLander3dConfig config;
	EXPECT_TRUE(config.ValidationError().empty());
	config.Mass_ = 0.0;
	EXPECT_FALSE(config.ValidationError().empty());
	config = {};
	config.PhysicsSubsteps_ = 0U;
	EXPECT_FALSE(config.ValidationError().empty());
	config = {};
	config.Terrain_.CellsX_ = 0U;
	EXPECT_FALSE(config.ValidationError().empty());
	config = {};
	config.SuccessReward_ = -1.0;
	EXPECT_FALSE(config.ValidationError().empty());
	config = {};
	config.FailurePenalty_ = 1.0;
	EXPECT_FALSE(config.ValidationError().empty());

	config = {};
	const OaLunarEpisodeManifest manifest = OaLunarTestManifestForConfig(config);
	const OaLunarTerrain flat = OaLunarTerrain::CreateFlat(config.Terrain_);
	ASSERT_TRUE(flat.IsValid()) << flat.Error();
	auto environment = OaLunarScalarEnvironment::CreateWithTerrain(
		config, manifest, flat);
	ASSERT_TRUE(environment.IsValid()) << environment.Error();
	const OaLunarLander3dState before = environment.State();
	const OaLunarTransition invalidAction = environment.Step(8U);
	ASSERT_TRUE(invalidAction.Valid_) << invalidAction.Error_;
	EXPECT_TRUE(invalidAction.Terminated_);
	EXPECT_FALSE(invalidAction.Truncated_);
	EXPECT_EQ(invalidAction.EndReason_, OaLunarEndReason::InvalidAction);
	EXPECT_DOUBLE_EQ(invalidAction.Reward_, config.FailurePenalty_);
	EXPECT_DOUBLE_EQ(
		invalidAction.RewardTerms_.Terminal_, config.FailurePenalty_);
	EXPECT_DOUBLE_EQ(
		invalidAction.RewardTerms_.Total_, config.FailurePenalty_);
	EXPECT_EQ(environment.State().Position_, before.Position_);
	EXPECT_EQ(environment.State().EpisodeStep_, before.EpisodeStep_ + 1U);
	EXPECT_EQ(environment.State().EndReason_, OaLunarEndReason::InvalidAction);

	OaLunarLander3dState invalidState = before;
	invalidState.Fuel_ = config.FuelCapacity_ + 1.0;
	EXPECT_FALSE(environment.SetState(invalidState));
	invalidState = before;
	invalidState.Terminated_ = true;
	EXPECT_FALSE(environment.SetState(invalidState));

	OaLunarLander3dState directState = OaLunarTestFlightState(config);
	const OaLunarPhysicsResult invalidEnum = OaLunarScalarPhysics::Integrate(
		config, flat, static_cast<OaLunarAction>(99U), directState);
	EXPECT_FALSE(invalidEnum.Valid_);

	OaLunarLander3dConfig mismatchedConfig = config;
	mismatchedConfig.Terrain_.MaxSlope_ = 0.2;
	directState = OaLunarTestFlightState(mismatchedConfig);
	const OaLunarPhysicsResult mismatchedTerrain =
		OaLunarScalarPhysics::Integrate(
			mismatchedConfig, flat, OaLunarAction::Coast, directState);
	EXPECT_FALSE(mismatchedTerrain.Valid_);
	EXPECT_FALSE(OaLunarScalarEnvironment::CreateWithTerrain(
		mismatchedConfig, manifest, flat).IsValid());
	OaLunarLander3dConfig changedReward = config;
	changedReward.SuccessReward_ += 1.0;
	const OaLunarTerrain changedRewardFlat = OaLunarTerrain::CreateFlat(
		changedReward.Terrain_);
	ASSERT_TRUE(changedRewardFlat.IsValid());
	EXPECT_FALSE(OaLunarScalarEnvironment::CreateWithTerrain(
		changedReward, manifest, changedRewardFlat).IsValid());
	OaLunarEpisodeManifest tamperedFingerprint = manifest;
	tamperedFingerprint.ConfigFingerprint_ ^= 1U;
	EXPECT_FALSE(OaLunarScalarEnvironment::CreateWithTerrain(
		config, tamperedFingerprint, flat).IsValid());
}

TEST(LunarLander3d, ManifestDerivationIsVersionedDeterministicAndTamperEvident) {
	const OaLunarEpisodeManifest first = OaLunarTestManifest();
	const OaLunarEpisodeManifest repeated = OaLunarTestManifest();
	EXPECT_EQ(first, repeated);
	EXPECT_TRUE(first.ValidationError().empty());
	EXPECT_NE(first.TerrainSeed_, first.SpawnSeed_);
	EXPECT_NE(first.SpawnSeed_, first.DomainSeed_);
	EXPECT_DOUBLE_EQ(
		first.Sample01(OaLunarRandomPurpose::Spawn, 11U),
		repeated.Sample01(OaLunarRandomPurpose::Spawn, 11U));

	EXPECT_NE(OaLunarTestManifest(first.BaseSeed_, 4U, first.EpisodeIndex_), first);
	EXPECT_NE(OaLunarTestManifest(
		first.BaseSeed_, first.EnvironmentLane_, first.EpisodeIndex_ + 1U), first);

	OaLunarEpisodeManifest tamperedSeed = first;
	tamperedSeed.SpawnSeed_ ^= 1U;
	EXPECT_FALSE(tamperedSeed.ValidationError().empty());
	OaLunarEpisodeManifest tamperedPhysics = first;
	++tamperedPhysics.PhysicsVersion_;
	EXPECT_FALSE(tamperedPhysics.ValidationError().empty());
	OaLunarEpisodeManifest tamperedObservation = first;
	++tamperedObservation.ObservationVersion_;
	EXPECT_FALSE(tamperedObservation.ValidationError().empty());
	OaLunarEpisodeManifest tamperedReward = first;
	++tamperedReward.RewardVersion_;
	EXPECT_FALSE(tamperedReward.ValidationError().empty());

	EXPECT_EQ(first.TerrainSeed_, 0x87b6d5a424c9738bULL);
	EXPECT_EQ(first.SpawnSeed_, 0xdb0f56e47eab3480ULL);
	EXPECT_EQ(first.DomainSeed_, 0x2ce2d29d6f0762e7ULL);
}

TEST(LunarLander3d, SeededTerrainIsDeterministicBoundedAndPadFlat) {
	const OaLunarTerrainConfig config;
	const OaLunarEpisodeManifest manifest = OaLunarTestManifest();
	const OaLunarTerrain first = OaLunarTerrain::CreateSeeded(config, manifest);
	const OaLunarTerrain repeated = OaLunarTerrain::CreateSeeded(config, manifest);
	ASSERT_TRUE(first.IsValid()) << first.Error();
	ASSERT_TRUE(repeated.IsValid()) << repeated.Error();
	EXPECT_EQ(first.Heights(), repeated.Heights());
	const OaLunarTerrain changed = OaLunarTerrain::CreateSeeded(
		config, OaLunarEpisodeManifest::Derive(
			manifest.BaseSeed_ + 1U,
			manifest.EnvironmentLane_, manifest.EpisodeIndex_,
			manifest.ConfigFingerprint_));
	ASSERT_TRUE(changed.IsValid()) << changed.Error();
	EXPECT_NE(first.Heights(), changed.Heights());

	for (std::uint32_t vertexZ = 0U; vertexZ <= config.CellsZ_; ++vertexZ) {
		for (std::uint32_t vertexX = 0U; vertexX <= config.CellsX_; ++vertexX) {
			EXPECT_LE(std::abs(first.VertexHeight(vertexX, vertexZ)),
				config.MaxAbsHeight_ + 1.0e-12);
		}
	}
	for (std::uint32_t cellZ = 0U; cellZ < config.CellsZ_; ++cellZ) {
		for (std::uint32_t cellX = 0U; cellX < config.CellsX_; ++cellX) {
			const double baseX = first.MinX()
				+ static_cast<double>(cellX) * config.CellSize_;
			const double baseZ = first.MinZ()
				+ static_cast<double>(cellZ) * config.CellSize_;
			const OaLunarTerrainSample lower = first.Query(
				baseX + 0.75 * config.CellSize_,
				baseZ + 0.25 * config.CellSize_);
			const OaLunarTerrainSample upper = first.Query(
				baseX + 0.25 * config.CellSize_,
				baseZ + 0.75 * config.CellSize_);
			ASSERT_TRUE(lower.InBounds_);
			ASSERT_TRUE(upper.InBounds_);
			EXPECT_LE(OaLunarTerrainSlope(lower), config.MaxSlope_ + 1.0e-12);
			EXPECT_LE(OaLunarTerrainSlope(upper), config.MaxSlope_ + 1.0e-12);
		}
	}

	for (const double positionX : {-3.0, -1.0, 0.0, 1.0, 3.0}) {
		for (const double positionZ : {-3.0, -1.0, 0.0, 1.0, 3.0}) {
			const OaLunarTerrainSample pad = first.Query(positionX, positionZ);
			ASSERT_TRUE(pad.InBounds_);
			EXPECT_DOUBLE_EQ(pad.Height_, 0.0);
			EXPECT_DOUBLE_EQ(pad.Normal_.ComponentX_, 0.0);
			EXPECT_DOUBLE_EQ(pad.Normal_.ComponentY_, 1.0);
			EXPECT_DOUBLE_EQ(pad.Normal_.ComponentZ_, 0.0);
			EXPECT_TRUE(first.IsOnPad(positionX, positionZ));
		}
	}
}

TEST(LunarLander3d, RecordedActionTraceHasFrozenSameBuildDigest) {
	OaLunarLander3dConfig config;
	config.MaxEpisodeSteps_ = 96U;
	const OaLunarEpisodeManifest manifest =
		OaLunarTestManifestForConfig(config);
	auto environment = OaLunarScalarEnvironment::CreateSeeded(config, manifest);
	ASSERT_TRUE(environment.IsValid()) << environment.Error();
	OaLunarTraceDigest digest;
	OaLunarDigestManifest(digest, manifest);
	for (const double height : environment.Terrain().Heights()) {
		digest.AddDouble(height);
	}
	OaLunarDigestState(digest, environment.State());
	const std::array<OaLunarAction, 8U> recordedActions = {
		OaLunarAction::Coast,
		OaLunarAction::MainEngine,
		OaLunarAction::PitchPositive,
		OaLunarAction::PitchNegative,
		OaLunarAction::RollPositive,
		OaLunarAction::RollNegative,
		OaLunarAction::YawPositive,
		OaLunarAction::YawNegative,
	};
	for (std::uint32_t step = 0U; step < config.MaxEpisodeSteps_; ++step) {
		const OaLunarAction action = recordedActions[step % recordedActions.size()];
		digest.AddU32(static_cast<std::uint32_t>(action));
		const OaLunarTransition transition = environment.Step(
			static_cast<std::uint32_t>(action));
		ASSERT_TRUE(transition.Valid_) << "step=" << step << " " << transition.Error_;
		OaLunarDigestState(digest, environment.State());
		OaLunarDigestReward(digest, transition.RewardTerms_);
		digest.AddDouble(transition.Reward_);
		digest.AddBool(transition.Terminated_);
		digest.AddBool(transition.Truncated_);
		digest.AddU32(static_cast<std::uint32_t>(transition.EndReason_));
		for (const float observation : transition.Observation_) {
			digest.AddFloat(observation);
		}
	}
	EXPECT_FALSE(environment.State().Terminated_);
	EXPECT_TRUE(environment.State().Truncated_);
	EXPECT_EQ(environment.State().EndReason_, OaLunarEndReason::TimeLimit);
	EXPECT_EQ(digest.Value(), 0xa85ea7bfd77f9ab5ULL);
}

TEST(LunarLander3d, TerrainFreezesVerticesEdgesDiagonalsAndNormals) {
	OaLunarTerrainConfig config;
	config.CellsX_ = 2U;
	config.CellsZ_ = 2U;
	config.CellSize_ = 1.0;
	config.MaxAbsHeight_ = 10.0;
	config.MaxSlope_ = 20.0;
	config.PadHalfExtent_ = 0.0;
	config.PadTransitionWidth_ = 0.0;
	const std::vector<double> heights = {
		0.0, 1.0, 4.0,
		2.0, 4.0, 8.0,
		3.0, 6.0, 9.0,
	};
	const OaLunarTerrain terrain = OaLunarTerrain::CreateFromHeights(
		config, heights);
	ASSERT_TRUE(terrain.IsValid()) << terrain.Error();

	const OaLunarTerrainSample vertex = terrain.Query(0.0, 0.0);
	ASSERT_TRUE(vertex.InBounds_);
	EXPECT_EQ(vertex.CellX_, 1U);
	EXPECT_EQ(vertex.CellZ_, 1U);
	EXPECT_DOUBLE_EQ(vertex.LocalX_, 0.0);
	EXPECT_DOUBLE_EQ(vertex.LocalZ_, 0.0);
	EXPECT_DOUBLE_EQ(vertex.Height_, 4.0);

	const OaLunarTerrainSample edge = terrain.Query(0.0, -0.5);
	ASSERT_TRUE(edge.InBounds_);
	EXPECT_EQ(edge.CellX_, 1U);
	EXPECT_EQ(edge.CellZ_, 0U);
	EXPECT_EQ(edge.Triangle_, OaLunarTerrainTriangle::UpperLeft);
	EXPECT_DOUBLE_EQ(edge.Height_, 2.5);

	const OaLunarTerrainSample diagonal = terrain.Query(-0.5, -0.5);
	ASSERT_TRUE(diagonal.InBounds_);
	EXPECT_EQ(diagonal.Triangle_, OaLunarTerrainTriangle::LowerRight);
	EXPECT_DOUBLE_EQ(diagonal.Height_, 2.0);
	const OaLunarTerrainSample lower = terrain.Query(-0.25, -0.75);
	const OaLunarTerrainSample upper = terrain.Query(-0.75, -0.25);
	ASSERT_TRUE(lower.InBounds_);
	ASSERT_TRUE(upper.InBounds_);
	EXPECT_DOUBLE_EQ(lower.Height_, 1.5);
	EXPECT_DOUBLE_EQ(upper.Height_, 2.0);
	const double lowerNormalScale = 1.0 / std::sqrt(11.0);
	EXPECT_NEAR(lower.Normal_.ComponentX_, -lowerNormalScale, 1.0e-15);
	EXPECT_NEAR(lower.Normal_.ComponentY_, lowerNormalScale, 1.0e-15);
	EXPECT_NEAR(lower.Normal_.ComponentZ_, -3.0 * lowerNormalScale, 1.0e-15);
	EXPECT_NEAR(upper.Normal_.ComponentX_, -2.0 / 3.0, 1.0e-15);
	EXPECT_NEAR(upper.Normal_.ComponentY_, 1.0 / 3.0, 1.0e-15);
	EXPECT_NEAR(upper.Normal_.ComponentZ_, -2.0 / 3.0, 1.0e-15);

	const OaLunarTerrainSample maximum = terrain.Query(1.0, 1.0);
	ASSERT_TRUE(maximum.InBounds_);
	EXPECT_EQ(maximum.CellX_, 1U);
	EXPECT_EQ(maximum.CellZ_, 1U);
	EXPECT_DOUBLE_EQ(maximum.LocalX_, 1.0);
	EXPECT_DOUBLE_EQ(maximum.LocalZ_, 1.0);
	EXPECT_DOUBLE_EQ(maximum.Height_, 9.0);
	EXPECT_FALSE(terrain.Query(terrain.MinX() - 0.01, 0.0).InBounds_);
	EXPECT_FALSE(terrain.Query(0.0, terrain.MaxZ() + 0.01).InBounds_);
}

TEST(LunarLander3d, FreeFallMatchesSemiImplicitAnalyticSolution) {
	OaLunarLander3dConfig config;
	const auto manifest = OaLunarTestManifestForConfig(config);
	auto environment = OaLunarScalarEnvironment::CreateFlat(config, manifest);
	ASSERT_TRUE(environment.IsValid()) << environment.Error();
	OaLunarLander3dState state = OaLunarTestFlightState(config);
	state.LinearVelocity_ = {0.25, 1.0, -0.5};
	ASSERT_TRUE(environment.SetState(state));
	const OaLunarTransition transition = environment.Step(
		static_cast<std::uint32_t>(OaLunarAction::Coast));
	ASSERT_TRUE(transition.Valid_) << transition.Error_;
	ASSERT_FALSE(transition.Terminated_);
	ASSERT_FALSE(transition.Truncated_);

	const double substep = config.PolicyTimeStep_
		/ static_cast<double>(config.PhysicsSubsteps_);
	const double substepCount = static_cast<double>(config.PhysicsSubsteps_);
	const double expectedY = state.Position_.ComponentY_
		+ state.LinearVelocity_.ComponentY_ * config.PolicyTimeStep_
		- config.Gravity_ * substep * substep
			* substepCount * (substepCount + 1.0) * 0.5;
	EXPECT_NEAR(environment.State().Position_.ComponentX_,
		state.Position_.ComponentX_ + state.LinearVelocity_.ComponentX_ * config.PolicyTimeStep_,
		1.0e-14);
	EXPECT_NEAR(environment.State().Position_.ComponentY_, expectedY, 1.0e-14);
	EXPECT_NEAR(environment.State().Position_.ComponentZ_,
		state.Position_.ComponentZ_ + state.LinearVelocity_.ComponentZ_ * config.PolicyTimeStep_,
		1.0e-14);
	EXPECT_NEAR(environment.State().LinearVelocity_.ComponentY_,
		state.LinearVelocity_.ComponentY_ - config.Gravity_ * config.PolicyTimeStep_,
		1.0e-14);
}

TEST(LunarLander3d, MainThrustAndFuelAreIsolated) {
	OaLunarLander3dConfig config;
	config.Gravity_ = 0.0;
	auto environment = OaLunarScalarEnvironment::CreateFlat(
		config, OaLunarTestManifestForConfig(config));
	ASSERT_TRUE(environment.IsValid()) << environment.Error();
	const OaLunarLander3dState state = OaLunarTestFlightState(config);
	ASSERT_TRUE(environment.SetState(state));
	const OaLunarTransition transition = environment.Step(
		static_cast<std::uint32_t>(OaLunarAction::MainEngine));
	ASSERT_TRUE(transition.Valid_) << transition.Error_;
	const double acceleration = config.MainThrust_ / config.Mass_;
	const double substep = config.PolicyTimeStep_
		/ static_cast<double>(config.PhysicsSubsteps_);
	const double substepCount = static_cast<double>(config.PhysicsSubsteps_);
	EXPECT_NEAR(environment.State().LinearVelocity_.ComponentY_,
		acceleration * config.PolicyTimeStep_, 1.0e-14);
	EXPECT_NEAR(environment.State().Position_.ComponentY_,
		state.Position_.ComponentY_ + acceleration * substep * substep
			* substepCount * (substepCount + 1.0) * 0.5,
		1.0e-14);
	EXPECT_DOUBLE_EQ(environment.State().LinearVelocity_.ComponentX_, 0.0);
	EXPECT_DOUBLE_EQ(environment.State().LinearVelocity_.ComponentZ_, 0.0);
	EXPECT_NEAR(environment.State().Fuel_,
		config.FuelCapacity_ - config.MainFuelRate_ * config.PolicyTimeStep_,
		1.0e-12);
	EXPECT_DOUBLE_EQ(environment.State().AngularVelocityBody_.LengthSquared(), 0.0);
}

TEST(LunarLander3d, AttitudeTorqueIsIsolated) {
	OaLunarLander3dConfig config;
	config.Gravity_ = 0.0;
	auto environment = OaLunarScalarEnvironment::CreateFlat(
		config, OaLunarTestManifestForConfig(config));
	ASSERT_TRUE(environment.IsValid()) << environment.Error();
	const OaLunarLander3dState state = OaLunarTestFlightState(config);
	ASSERT_TRUE(environment.SetState(state));
	const OaLunarTransition transition = environment.Step(
		static_cast<std::uint32_t>(OaLunarAction::PitchPositive));
	ASSERT_TRUE(transition.Valid_) << transition.Error_;
	EXPECT_NEAR(environment.State().AngularVelocityBody_.ComponentX_,
		config.AttitudeTorque_ / config.DiagonalInertia_.ComponentX_
			* config.PolicyTimeStep_, 1.0e-14);
	EXPECT_DOUBLE_EQ(environment.State().AngularVelocityBody_.ComponentY_, 0.0);
	EXPECT_DOUBLE_EQ(environment.State().AngularVelocityBody_.ComponentZ_, 0.0);
	EXPECT_DOUBLE_EQ(environment.State().LinearVelocity_.LengthSquared(), 0.0);
	EXPECT_NEAR(environment.State().Fuel_,
		config.FuelCapacity_
			- config.AttitudeFuelRate_ * config.PolicyTimeStep_,
		1.0e-14);
	EXPECT_NEAR(environment.State().Orientation_.Norm(), 1.0, 1.0e-15);
}

TEST(LunarLander3d, QuaternionRenormalizationSurvivesLongTorqueTrace) {
	OaLunarLander3dConfig config;
	config.Gravity_ = 0.0;
	config.MaxEpisodeSteps_ = 2000U;
	auto environment = OaLunarScalarEnvironment::CreateFlat(
		config, OaLunarTestManifestForConfig(config));
	ASSERT_TRUE(environment.IsValid()) << environment.Error();
	ASSERT_TRUE(environment.SetState(OaLunarTestFlightState(config)));
	const std::array<OaLunarAction, 6U> actions = {
		OaLunarAction::PitchPositive, OaLunarAction::RollNegative,
		OaLunarAction::YawPositive, OaLunarAction::PitchNegative,
		OaLunarAction::RollPositive, OaLunarAction::YawNegative,
	};
	for (std::uint32_t step = 0U; step < 600U; ++step) {
		const OaLunarTransition transition = environment.Step(
			static_cast<std::uint32_t>(actions[step % actions.size()]));
		ASSERT_TRUE(transition.Valid_) << "step=" << step << " " << transition.Error_;
		ASSERT_FALSE(transition.Terminated_) << "step=" << step;
		ASSERT_FALSE(transition.Truncated_) << "step=" << step;
		EXPECT_NEAR(environment.State().Orientation_.Norm(), 1.0, 2.0e-15);
	}
}

TEST(LunarLander3d, FuelDepletionClampsThrustWithoutChangingMass) {
	OaLunarLander3dConfig config;
	config.Gravity_ = 0.0;
	config.FuelCapacity_ = 0.01;
	auto environment = OaLunarScalarEnvironment::CreateFlat(
		config, OaLunarTestManifestForConfig(config));
	ASSERT_TRUE(environment.IsValid()) << environment.Error();
	ASSERT_TRUE(environment.SetState(OaLunarTestFlightState(config)));
	const OaLunarTransition first = environment.Step(
		static_cast<std::uint32_t>(OaLunarAction::MainEngine));
	ASSERT_TRUE(first.Valid_) << first.Error_;
	const double burnDuration = config.FuelCapacity_ / config.MainFuelRate_;
	const double expectedVelocity = config.MainThrust_ / config.Mass_
		* burnDuration;
	EXPECT_DOUBLE_EQ(environment.State().Fuel_, 0.0);
	EXPECT_NEAR(environment.State().LinearVelocity_.ComponentY_, expectedVelocity, 1.0e-14);
	const double velocityBefore = environment.State().LinearVelocity_.ComponentY_;
	const OaLunarTransition second = environment.Step(
		static_cast<std::uint32_t>(OaLunarAction::MainEngine));
	ASSERT_TRUE(second.Valid_) << second.Error_;
	EXPECT_DOUBLE_EQ(environment.State().Fuel_, 0.0);
	EXPECT_DOUBLE_EQ(environment.State().LinearVelocity_.ComponentY_, velocityBefore);
}

TEST(LunarLander3d, ObservationHasFrozenThirtyThreeValueLayout) {
	OaLunarLander3dConfig config;
	auto environment = OaLunarScalarEnvironment::CreateFlat(
		config, OaLunarTestManifestForConfig(config));
	ASSERT_TRUE(environment.IsValid()) << environment.Error();
	OaLunarLander3dState state = OaLunarTestFlightState(config, 6.0);
	state.Position_.ComponentX_ = 2.0;
	state.Position_.ComponentZ_ = -3.0;
	state.LinearVelocity_ = {0.4, -0.8, 1.2};
	state.AngularVelocityBody_ = {0.1, -0.2, 0.3};
	state.Fuel_ = config.FuelCapacity_ * 0.5;
	state.FootContacts_ = {true, false, true, false};
	ASSERT_TRUE(environment.SetState(state));
	const auto observation = environment.Observation();
	EXPECT_EQ(observation.size(), 33U);
	for (const float value : observation) {
		EXPECT_TRUE(std::isfinite(value));
		EXPECT_GE(value, -1.0F);
		EXPECT_LE(value, 1.0F);
	}
	EXPECT_FLOAT_EQ(observation[0], static_cast<float>(
		state.Position_.ComponentX_ / config.PositionObservationScale_));
	EXPECT_FLOAT_EQ(observation[6], 0.0F);
	EXPECT_FLOAT_EQ(observation[7], 1.0F);
	EXPECT_FLOAT_EQ(observation[8], 0.0F);
	EXPECT_FLOAT_EQ(observation[9], 0.0F);
	EXPECT_FLOAT_EQ(observation[10], 0.0F);
	EXPECT_FLOAT_EQ(observation[11], -1.0F);
	EXPECT_FLOAT_EQ(observation[28], 1.0F);
	EXPECT_FLOAT_EQ(observation[29], 0.0F);
	EXPECT_FLOAT_EQ(observation[30], 1.0F);
	EXPECT_FLOAT_EQ(observation[31], 0.0F);
	EXPECT_FLOAT_EQ(observation[32], 0.5F);
}

TEST(LunarLander3d, FlatContactIsFiniteFixedIterationAndBounded) {
	OaLunarLander3dConfig config;
	config.HardFootImpactSpeed_ = 10.0;
	config.SafeDwellSteps_ = 1000U;
	auto environment = OaLunarScalarEnvironment::CreateFlat(
		config, OaLunarTestManifestForConfig(config));
	ASSERT_TRUE(environment.IsValid()) << environment.Error();
	OaLunarLander3dState state = OaLunarTestFlightState(config, 1.10);
	state.LinearVelocity_.ComponentY_ = -0.1;
	ASSERT_TRUE(environment.SetState(state));
	const OaLunarTransition transition = environment.Step(
		static_cast<std::uint32_t>(OaLunarAction::Coast));
	ASSERT_TRUE(transition.Valid_) << transition.Error_;
	EXPECT_TRUE(environment.State().IsFinite());
	EXPECT_TRUE(transition.Contact_.IsFinite());
	EXPECT_TRUE(transition.Contact_.Bounded_);
	EXPECT_TRUE(transition.Contact_.FootContactOccurred_);
	EXPECT_GT(transition.Contact_.ContactCount_, 0U);
	EXPECT_LE(transition.Contact_.MaximumNormalImpulse_, config.MaxContactImpulse_);
	EXPECT_LE(transition.Contact_.MaximumFrictionImpulse_, config.MaxContactImpulse_);
	const double maximumCorrection = static_cast<double>(
		config.PhysicsSubsteps_ * config.ContactIterations_)
		* static_cast<double>(
			config.BodySupports_.size() + config.FootSupports_.size())
		* config.MaxPositionCorrectionPerContact_;
	EXPECT_LE(transition.Contact_.TotalPositionCorrection_,
		maximumCorrection + 1.0e-12);
}

TEST(LunarLander3d, SlopeContactIsFiniteAndBounded) {
	OaLunarLander3dConfig config;
	config.HardFootImpactSpeed_ = 10.0;
	config.SafeDwellSteps_ = 1000U;
	std::vector<double> heights;
	heights.reserve(static_cast<std::size_t>(config.Terrain_.CellsX_ + 1U)
		* static_cast<std::size_t>(config.Terrain_.CellsZ_ + 1U));
	for (std::uint32_t vertexZ = 0U;
		vertexZ <= config.Terrain_.CellsZ_;
		++vertexZ) {
		for (std::uint32_t vertexX = 0U;
			vertexX <= config.Terrain_.CellsX_;
			++vertexX) {
			const double positionX = -static_cast<double>(config.Terrain_.CellsX_)
				* config.Terrain_.CellSize_ * 0.5
				+ static_cast<double>(vertexX) * config.Terrain_.CellSize_;
			heights.push_back(positionX * 0.08);
		}
	}
	const OaLunarTerrain slope = OaLunarTerrain::CreateFromHeights(
		config.Terrain_, heights);
	ASSERT_TRUE(slope.IsValid()) << slope.Error();
	auto environment = OaLunarScalarEnvironment::CreateWithTerrain(
		config, OaLunarTestManifestForConfig(config), slope);
	ASSERT_TRUE(environment.IsValid()) << environment.Error();
	OaLunarLander3dState state = OaLunarTestFlightState(config, 1.20);
	state.LinearVelocity_.ComponentY_ = -0.1;
	ASSERT_TRUE(environment.SetState(state));
	const OaLunarTransition transition = environment.Step(
		static_cast<std::uint32_t>(OaLunarAction::Coast));
	ASSERT_TRUE(transition.Valid_) << transition.Error_;
	EXPECT_TRUE(environment.State().IsFinite());
	EXPECT_TRUE(transition.Contact_.IsFinite());
	EXPECT_TRUE(transition.Contact_.Bounded_);
	EXPECT_TRUE(transition.Contact_.FootContactOccurred_);
	EXPECT_LE(transition.Contact_.MaximumNormalImpulse_, config.MaxContactImpulse_);
}

TEST(LunarLander3d, EndReasonsKeepTerminationAndTruncationDistinct) {
	OaLunarLander3dConfig successConfig;
	successConfig.Gravity_ = 0.0;
	successConfig.SafeDwellSteps_ = 2U;
	auto success = OaLunarScalarEnvironment::CreateFlat(
		successConfig, OaLunarTestManifestForConfig(successConfig));
	ASSERT_TRUE(success.IsValid()) << success.Error();
	ASSERT_TRUE(success.SetState(OaLunarTestFlightState(successConfig, 1.15)));
	const OaLunarTransition dwell = success.Step(
		static_cast<std::uint32_t>(OaLunarAction::Coast));
	ASSERT_TRUE(dwell.Valid_);
	EXPECT_FALSE(dwell.Terminated_);
	EXPECT_FALSE(dwell.Truncated_);
	const OaLunarTransition landed = success.Step(
		static_cast<std::uint32_t>(OaLunarAction::Coast));
	ASSERT_TRUE(landed.Valid_);
	EXPECT_TRUE(landed.Terminated_);
	EXPECT_FALSE(landed.Truncated_);
	EXPECT_EQ(landed.EndReason_, OaLunarEndReason::SafeLanding);
	EXPECT_FALSE(success.Step(0U).Valid_);

	OaLunarLander3dConfig impactConfig;
	impactConfig.Gravity_ = 0.0;
	const OaLunarEpisodeManifest impactManifest =
		OaLunarTestManifestForConfig(impactConfig);
	auto bodyImpact = OaLunarScalarEnvironment::CreateFlat(
		impactConfig, impactManifest);
	ASSERT_TRUE(bodyImpact.IsValid());
	ASSERT_TRUE(bodyImpact.SetState(OaLunarTestFlightState(impactConfig, 0.60)));
	const OaLunarTransition body = bodyImpact.Step(0U);
	ASSERT_TRUE(body.Valid_);
	EXPECT_TRUE(body.Terminated_);
	EXPECT_FALSE(body.Truncated_);
	EXPECT_EQ(body.EndReason_, OaLunarEndReason::BodyImpact);

	auto hardImpact = OaLunarScalarEnvironment::CreateFlat(
		impactConfig, impactManifest);
	ASSERT_TRUE(hardImpact.IsValid());
	OaLunarLander3dState hardState = OaLunarTestFlightState(impactConfig, 1.17);
	hardState.LinearVelocity_.ComponentY_ = -3.0;
	ASSERT_TRUE(hardImpact.SetState(hardState));
	const OaLunarTransition hard = hardImpact.Step(0U);
	ASSERT_TRUE(hard.Valid_);
	EXPECT_TRUE(hard.Terminated_);
	EXPECT_FALSE(hard.Truncated_);
	EXPECT_EQ(hard.EndReason_, OaLunarEndReason::HardFootImpact);

	auto outOfBounds = OaLunarScalarEnvironment::CreateFlat(
		impactConfig, impactManifest);
	ASSERT_TRUE(outOfBounds.IsValid());
	OaLunarLander3dState outside = OaLunarTestFlightState(impactConfig);
	outside.Position_.ComponentX_ = outOfBounds.Terrain().MaxX() + 0.5;
	ASSERT_TRUE(outOfBounds.SetState(outside));
	const OaLunarTransition bounds = outOfBounds.Step(0U);
	ASSERT_TRUE(bounds.Valid_);
	EXPECT_TRUE(bounds.Terminated_);
	EXPECT_FALSE(bounds.Truncated_);
	EXPECT_EQ(bounds.EndReason_, OaLunarEndReason::OutOfBounds);

	OaLunarLander3dConfig horizonConfig;
	horizonConfig.Gravity_ = 0.0;
	horizonConfig.MaxEpisodeSteps_ = 1U;
	auto horizon = OaLunarScalarEnvironment::CreateFlat(
		horizonConfig, OaLunarTestManifestForConfig(horizonConfig));
	ASSERT_TRUE(horizon.IsValid());
	ASSERT_TRUE(horizon.SetState(OaLunarTestFlightState(horizonConfig)));
	const OaLunarTransition timeout = horizon.Step(0U);
	ASSERT_TRUE(timeout.Valid_);
	EXPECT_FALSE(timeout.Terminated_);
	EXPECT_TRUE(timeout.Truncated_);
	EXPECT_EQ(timeout.EndReason_, OaLunarEndReason::TimeLimit);
	EXPECT_FALSE(horizon.Step(0U).Valid_);

	auto stopped = OaLunarScalarEnvironment::CreateFlat(
		impactConfig, impactManifest);
	ASSERT_TRUE(stopped.IsValid());
	const OaLunarTransition external = stopped.Step(0U, true);
	ASSERT_TRUE(external.Valid_);
	EXPECT_FALSE(external.Terminated_);
	EXPECT_TRUE(external.Truncated_);
	EXPECT_EQ(external.EndReason_, OaLunarEndReason::ExternalStop);

	auto numerical = OaLunarScalarEnvironment::CreateFlat(
		impactConfig, impactManifest);
	ASSERT_TRUE(numerical.IsValid());
	OaLunarLander3dState huge = OaLunarTestFlightState(impactConfig);
	huge.Position_.ComponentY_ = std::numeric_limits<double>::max();
	huge.LinearVelocity_.ComponentY_ = std::numeric_limits<double>::max();
	ASSERT_TRUE(numerical.SetState(huge));
	const OaLunarTransition failed = numerical.Step(0U);
	ASSERT_TRUE(failed.Valid_);
	EXPECT_TRUE(failed.Terminated_);
	EXPECT_FALSE(failed.Truncated_);
	EXPECT_EQ(failed.EndReason_, OaLunarEndReason::NumericalFailure);
	EXPECT_TRUE(std::isfinite(failed.Reward_));
}

TEST(LunarLander3d, RewardDiagnosticsSumAndFreezeTerminalPotentialHandling) {
	OaLunarLander3dConfig config;
	config.Gravity_ = 0.0;
	auto flight = OaLunarScalarEnvironment::CreateFlat(
		config, OaLunarTestManifestForConfig(config));
	ASSERT_TRUE(flight.IsValid());
	ASSERT_TRUE(flight.SetState(OaLunarTestFlightState(config)));
	const OaLunarTransition main = flight.Step(
		static_cast<std::uint32_t>(OaLunarAction::MainEngine));
	ASSERT_TRUE(main.Valid_);
	EXPECT_TRUE(main.RewardTerms_.IsFinite());
	EXPECT_DOUBLE_EQ(main.RewardTerms_.Total_, main.RewardTerms_.Sum());
	EXPECT_DOUBLE_EQ(main.Reward_, main.RewardTerms_.Total_);
	EXPECT_LT(main.RewardTerms_.MainFuelCost_, 0.0);
	EXPECT_DOUBLE_EQ(main.RewardTerms_.AttitudeFuelCost_, 0.0);

	auto failure = OaLunarScalarEnvironment::CreateFlat(
		config, OaLunarTestManifestForConfig(config));
	ASSERT_TRUE(failure.IsValid());
	ASSERT_TRUE(failure.SetState(OaLunarTestFlightState(config, 0.60)));
	const OaLunarTransition terminal = failure.Step(0U);
	ASSERT_TRUE(terminal.Valid_);
	ASSERT_TRUE(terminal.Terminated_);
	EXPECT_DOUBLE_EQ(terminal.RewardTerms_.Shaping_,
		-terminal.RewardTerms_.PotentialBefore_);
	EXPECT_DOUBLE_EQ(terminal.RewardTerms_.Terminal_, config.FailurePenalty_);
	EXPECT_DOUBLE_EQ(terminal.RewardTerms_.Total_, terminal.RewardTerms_.Sum());

	OaLunarLander3dConfig horizonConfig = config;
	horizonConfig.MaxEpisodeSteps_ = 1U;
	auto horizon = OaLunarScalarEnvironment::CreateFlat(
		horizonConfig, OaLunarTestManifestForConfig(horizonConfig));
	ASSERT_TRUE(horizon.IsValid());
	ASSERT_TRUE(horizon.SetState(OaLunarTestFlightState(horizonConfig)));
	const OaLunarTransition truncated = horizon.Step(0U);
	ASSERT_TRUE(truncated.Valid_);
	ASSERT_TRUE(truncated.Truncated_);
	EXPECT_NEAR(truncated.RewardTerms_.Shaping_,
		horizonConfig.RewardGamma_ * truncated.RewardTerms_.PotentialAfter_
			- truncated.RewardTerms_.PotentialBefore_,
		1.0e-15);
	EXPECT_DOUBLE_EQ(truncated.RewardTerms_.Terminal_, 0.0);
}

TEST(LunarLander3d, PotentialShapingCannotFarmHoverCyclesOrDelayedTermination) {
	OaLunarLander3dConfig config;
	config.Gravity_ = 0.0;
	config.MaxEpisodeSteps_ = 1000U;
	auto hover = OaLunarScalarEnvironment::CreateFlat(
		config, OaLunarTestManifestForConfig(config));
	ASSERT_TRUE(hover.IsValid()) << hover.Error();
	ASSERT_TRUE(hover.SetState(OaLunarTestFlightState(config, 10.0)));
	const double fixedPotential = OaLunarScalarPhysics::Potential(
		config, hover.State());
	double discountedHoverShaping = 0.0;
	double discount = 1.0;
	constexpr std::uint32_t hoverSteps = 120U;
	for (std::uint32_t step = 0U; step < hoverSteps; ++step) {
		const OaLunarTransition transition = hover.Step(0U);
		ASSERT_TRUE(transition.Valid_);
		ASSERT_FALSE(transition.Terminated_);
		ASSERT_FALSE(transition.Truncated_);
		EXPECT_DOUBLE_EQ(transition.RewardTerms_.MainFuelCost_, 0.0);
		EXPECT_DOUBLE_EQ(transition.RewardTerms_.AttitudeFuelCost_, 0.0);
		EXPECT_DOUBLE_EQ(transition.RewardTerms_.SoftFootContact_, 0.0);
		EXPECT_DOUBLE_EQ(transition.RewardTerms_.StableDwell_, 0.0);
		EXPECT_DOUBLE_EQ(transition.RewardTerms_.Terminal_, 0.0);
		discountedHoverShaping += discount * transition.RewardTerms_.Shaping_;
		discount *= config.RewardGamma_;
	}
	const double expectedHoverShaping = fixedPotential
		* (std::pow(config.RewardGamma_, static_cast<double>(hoverSteps)) - 1.0);
	EXPECT_NEAR(discountedHoverShaping, expectedHoverShaping, 2.0e-14);
	EXPECT_LE(discountedHoverShaping, -fixedPotential + 1.0e-14);

	OaLunarLander3dState cycleStateA = OaLunarTestFlightState(config, 9.0);
	cycleStateA.Position_.ComponentX_ = 1.0;
	OaLunarLander3dState cycleStateB = OaLunarTestFlightState(config, 7.0);
	cycleStateB.Position_.ComponentX_ = -1.0;
	cycleStateB.LinearVelocity_.ComponentZ_ = 0.5;
	const double potentialA = OaLunarScalarPhysics::Potential(config, cycleStateA);
	const double potentialB = OaLunarScalarPhysics::Potential(config, cycleStateB);
	const double shapingAB = config.RewardGamma_ * potentialB - potentialA;
	const double shapingBA = config.RewardGamma_ * potentialA - potentialB;
	const double discountedCycle = shapingAB + config.RewardGamma_ * shapingBA;
	EXPECT_NEAR(discountedCycle,
		(std::pow(config.RewardGamma_, 2.0) - 1.0) * potentialA,
		1.0e-15);

	for (const std::uint32_t delaySteps : {0U, 1U, 17U, 120U}) {
		double discountedDelayedShaping = 0.0;
		double delayedDiscount = 1.0;
		for (std::uint32_t step = 0U; step < delaySteps; ++step) {
			discountedDelayedShaping += delayedDiscount
				* (config.RewardGamma_ * fixedPotential - fixedPotential);
			delayedDiscount *= config.RewardGamma_;
		}
		discountedDelayedShaping += delayedDiscount * (-fixedPotential);
		EXPECT_NEAR(discountedDelayedShaping, -fixedPotential, 2.0e-14)
			<< "delay_steps=" << delaySteps;
	}

	auto terminal = OaLunarScalarEnvironment::CreateFlat(
		config, OaLunarTestManifestForConfig(config));
	ASSERT_TRUE(terminal.IsValid());
	ASSERT_TRUE(terminal.SetState(OaLunarTestFlightState(config, 0.60)));
	const OaLunarTransition failure = terminal.Step(0U);
	ASSERT_TRUE(failure.Valid_);
	ASSERT_TRUE(failure.Terminated_);
	EXPECT_DOUBLE_EQ(failure.RewardTerms_.Terminal_, config.FailurePenalty_);
	EXPECT_FALSE(terminal.Step(0U).Valid_);
}

TEST(LunarLander3d, PartialRestSoftContactRewardIsEpisodeBounded) {
	OaLunarLander3dConfig config;
	config.Gravity_ = 0.0;
	config.SafeDwellSteps_ = 1000U;
	for (std::size_t footIndex = 1U;
		footIndex < config.FootSupports_.size();
		++footIndex) {
		config.FootSupports_[footIndex].BodyOffset_.ComponentY_ = -0.5;
	}
	auto environment = OaLunarScalarEnvironment::CreateFlat(
		config, OaLunarTestManifestForConfig(config));
	ASSERT_TRUE(environment.IsValid()) << environment.Error();
	ASSERT_TRUE(environment.SetState(OaLunarTestFlightState(config, 1.15)));

	double totalSoftContact = 0.0;
	for (std::uint32_t step = 0U; step < 64U; ++step) {
		const OaLunarTransition transition = environment.Step(0U);
		ASSERT_TRUE(transition.Valid_);
		ASSERT_FALSE(transition.Terminated_);
		ASSERT_FALSE(transition.Truncated_);
		totalSoftContact += transition.RewardTerms_.SoftFootContact_;
		EXPECT_TRUE(environment.State().FootContacts_[0]);
		EXPECT_FALSE(environment.State().FootContacts_[1]);
		EXPECT_FALSE(environment.State().FootContacts_[2]);
		EXPECT_FALSE(environment.State().FootContacts_[3]);
	}
	EXPECT_DOUBLE_EQ(totalSoftContact, config.SoftFootContactReward_);
	EXPECT_TRUE(environment.State().FootContactRewarded_[0]);
	EXPECT_FALSE(environment.State().FootContactRewarded_[1]);

	OaLunarLander3dState lifted = environment.State();
	lifted.Position_.ComponentY_ = 2.0;
	ASSERT_TRUE(environment.SetState(lifted));
	const OaLunarTransition noContact = environment.Step(0U);
	ASSERT_TRUE(noContact.Valid_);
	EXPECT_DOUBLE_EQ(noContact.RewardTerms_.SoftFootContact_, 0.0);
	EXPECT_FALSE(environment.State().FootContacts_[0]);

	OaLunarLander3dState returned = environment.State();
	returned.Position_.ComponentY_ = 1.15;
	ASSERT_TRUE(environment.SetState(returned));
	const OaLunarTransition repeatedContact = environment.Step(0U);
	ASSERT_TRUE(repeatedContact.Valid_);
	EXPECT_TRUE(environment.State().FootContacts_[0]);
	EXPECT_DOUBLE_EQ(repeatedContact.RewardTerms_.SoftFootContact_, 0.0);
}

TEST(LunarLander3d, ScriptedControllerReachesSafeLanding) {
	OaLunarLander3dConfig config;
	config.SafeDwellSteps_ = 12U;
	config.MaxEpisodeSteps_ = 1200U;
	auto environment = OaLunarScalarEnvironment::CreateFlat(
		config, OaLunarTestManifestForConfig(config));
	ASSERT_TRUE(environment.IsValid()) << environment.Error();
	OaLunarLander3dState state = OaLunarTestFlightState(config, 4.0);
	state.LinearVelocity_.ComponentY_ = -0.2;
	ASSERT_TRUE(environment.SetState(state));

	for (std::uint32_t step = 0U;
		step < config.MaxEpisodeSteps_ and not environment.State().Terminated_
			and not environment.State().Truncated_;
		++step) {
		const OaLunarAction action = OaLunarScriptedLandingAction(
			config, environment.State());
		const OaLunarTransition transition = environment.Step(
			static_cast<std::uint32_t>(action));
		ASSERT_TRUE(transition.Valid_) << "step=" << step << " " << transition.Error_;
	}
	EXPECT_TRUE(environment.State().Terminated_);
	EXPECT_FALSE(environment.State().Truncated_);
	EXPECT_EQ(environment.State().EndReason_, OaLunarEndReason::SafeLanding);
}

TEST(LunarLander3d, ScriptedControllerCommandsCorrectiveBodyTorques) {
	OaLunarLander3dConfig config;
	OaLunarLander3dState state = OaLunarTestFlightState(config, 4.0);
	state.LinearVelocity_ = {};

	state.Orientation_ = OaLunarQuat::FromAxisAngle(
		{1.0, 0.0, 0.0}, 0.04);
	EXPECT_EQ(
		OaLunarScriptedLandingAction(config, state),
		OaLunarAction::PitchNegative);

	state.Orientation_ = OaLunarQuat::FromAxisAngle(
		{0.0, 0.0, 1.0}, 0.04);
	EXPECT_EQ(
		OaLunarScriptedLandingAction(config, state),
		OaLunarAction::RollNegative);

	state.Orientation_ = {};
	state.AngularVelocityBody_ = {0.04, 0.0, 0.0};
	EXPECT_EQ(
		OaLunarScriptedLandingAction(config, state),
		OaLunarAction::PitchNegative);

	state.AngularVelocityBody_ = {0.0, 0.0, 0.04};
	EXPECT_EQ(
		OaLunarScriptedLandingAction(config, state),
		OaLunarAction::RollNegative);

	state = OaLunarTestFlightState(config, 4.0);
	state.Position_.ComponentX_ = 1.0;
	EXPECT_EQ(
		OaLunarScriptedLandingAction(config, state),
		OaLunarAction::RollPositive);

	state = OaLunarTestFlightState(config, 4.0);
	state.Position_.ComponentZ_ = 1.0;
	EXPECT_EQ(
		OaLunarScriptedLandingAction(config, state),
		OaLunarAction::PitchNegative);
}

TEST(LunarLander3d, ScriptedControllerLandsAdversarialSpawnCorners) {
	OaLunarLander3dConfig config;
	const double padRange = config.Terrain_.PadHalfExtent_ * 0.35;
	std::uint32_t caseIndex = 0U;
	for (const double xSign : {-1.0, 1.0}) {
		for (const double zSign : {-1.0, 1.0}) {
			for (const double yawSign : {-1.0, 1.0}) {
				auto environment = OaLunarScalarEnvironment::CreateFlat(
					config, OaLunarTestManifestForConfig(
						config, 0x434f524e45525f54ULL, caseIndex++, 0U));
				ASSERT_TRUE(environment.IsValid()) << environment.Error();
				OaLunarLander3dState state = OaLunarTestFlightState(config, 7.0);
				state.Position_.ComponentX_ = xSign * padRange;
				state.Position_.ComponentZ_ = zSign * padRange;
				state.LinearVelocity_ = {xSign * 0.12, -0.3, zSign * 0.12};
				state.Orientation_ = (
					OaLunarQuat::FromAxisAngle(
						{0.0, 1.0, 0.0}, yawSign * 0.08)
					* OaLunarQuat::FromAxisAngle(
						{1.0, 0.0, 0.0}, xSign * 0.03)
					* OaLunarQuat::FromAxisAngle(
						{0.0, 0.0, 1.0}, zSign * 0.03)).Normalized();
				ASSERT_TRUE(environment.SetState(state));

				while (not environment.State().Terminated_
					and not environment.State().Truncated_) {
					const OaLunarTransition transition = environment.Step(
						static_cast<std::uint32_t>(OaLunarScriptedLandingAction(
							config, environment.State())));
					ASSERT_TRUE(transition.Valid_) << transition.Error_;
				}
				EXPECT_EQ(
					environment.State().EndReason_, OaLunarEndReason::SafeLanding)
					<< "x_sign=" << xSign << " z_sign=" << zSign
					<< " yaw_sign=" << yawSign;
			}
		}
	}
}

TEST(LunarLander3d, ScriptedControllerCoversHeldOutFlatSpawns) {
	constexpr std::uint64_t baseSeed = 0x50494c4f545f4556ULL;
	constexpr std::uint32_t episodes = 512U;
	OaLunarLander3dConfig config;
	std::uint32_t safeLandings = 0U;
	std::array<std::uint32_t, 9U> reasons{};
	std::array<std::uint64_t, 8U> actionCounts{};
	std::ostringstream timeoutDiagnostics;
	for (std::uint32_t lane = 0U; lane < episodes; ++lane) {
		const OaLunarEpisodeManifest manifest =
			OaLunarEpisodeManifest::Derive(
				baseSeed, lane, 0U, config.ContractFingerprint());
		auto environment = OaLunarScalarEnvironment::CreateFlat(config, manifest);
		ASSERT_TRUE(environment.IsValid()) << "lane=" << lane
			<< " " << environment.Error();
		for (std::uint32_t step = 0U;
			step < config.MaxEpisodeSteps_
				and not environment.State().Terminated_
				and not environment.State().Truncated_;
			++step) {
			const OaLunarAction action = OaLunarScriptedLandingAction(
				config, environment.State());
			++actionCounts[static_cast<std::size_t>(action)];
			const OaLunarTransition transition = environment.Step(
				static_cast<std::uint32_t>(action));
			ASSERT_TRUE(transition.Valid_) << "lane=" << lane
				<< " step=" << step << " " << transition.Error_;
		}
		const auto reasonIndex = static_cast<std::size_t>(
			environment.State().EndReason_);
		ASSERT_LT(reasonIndex, reasons.size());
		++reasons[reasonIndex];
		if (environment.State().EndReason_ == OaLunarEndReason::SafeLanding) {
			++safeLandings;
		} else if (environment.State().EndReason_ == OaLunarEndReason::TimeLimit) {
			const OaLunarLander3dState& state = environment.State();
			timeoutDiagnostics << " lane=" << lane
				<< " position=(" << state.Position_.ComponentX_ << ','
				<< state.Position_.ComponentY_ << ','
				<< state.Position_.ComponentZ_ << ')'
				<< " linear_speed=" << state.LinearVelocity_.Length()
				<< " angular_speed=" << state.AngularVelocityBody_.Length()
				<< " dwell=" << state.StableDwell_
				<< " contacts=" << state.FootContacts_[0]
				<< state.FootContacts_[1]
				<< state.FootContacts_[2]
				<< state.FootContacts_[3]
				<< " on_pad=" << state.FeetOnPad_[0]
				<< state.FeetOnPad_[1]
				<< state.FeetOnPad_[2]
				<< state.FeetOnPad_[3];
		}
	}
	RecordProperty("episodes", episodes);
	RecordProperty("safe_landings", safeLandings);
	RecordProperty("body_impacts", reasons[static_cast<std::size_t>(
		OaLunarEndReason::BodyImpact)]);
	RecordProperty("hard_foot_impacts", reasons[static_cast<std::size_t>(
		OaLunarEndReason::HardFootImpact)]);
	RecordProperty("out_of_bounds", reasons[static_cast<std::size_t>(
		OaLunarEndReason::OutOfBounds)]);
	RecordProperty("time_limits", reasons[static_cast<std::size_t>(
		OaLunarEndReason::TimeLimit)]);
	RecordProperty("action_coast", actionCounts[static_cast<std::size_t>(
		OaLunarAction::Coast)]);
	RecordProperty("action_main", actionCounts[static_cast<std::size_t>(
		OaLunarAction::MainEngine)]);
	RecordProperty("action_pitch_positive", actionCounts[static_cast<std::size_t>(
		OaLunarAction::PitchPositive)]);
	RecordProperty("action_pitch_negative", actionCounts[static_cast<std::size_t>(
		OaLunarAction::PitchNegative)]);
	RecordProperty("action_roll_positive", actionCounts[static_cast<std::size_t>(
		OaLunarAction::RollPositive)]);
	RecordProperty("action_roll_negative", actionCounts[static_cast<std::size_t>(
		OaLunarAction::RollNegative)]);
	EXPECT_EQ(safeLandings, episodes)
		<< "body=" << reasons[static_cast<std::size_t>(
			OaLunarEndReason::BodyImpact)]
		<< " hard-foot=" << reasons[static_cast<std::size_t>(
			OaLunarEndReason::HardFootImpact)]
		<< " out-of-bounds=" << reasons[static_cast<std::size_t>(
			OaLunarEndReason::OutOfBounds)]
		<< " timeout=" << reasons[static_cast<std::size_t>(
			OaLunarEndReason::TimeLimit)]
		<< timeoutDiagnostics.str();
	EXPECT_EQ(reasons[static_cast<std::size_t>(
		OaLunarEndReason::TimeLimit)], 0U);
	EXPECT_EQ(reasons[static_cast<std::size_t>(
		OaLunarEndReason::BodyImpact)], 0U);
	EXPECT_EQ(reasons[static_cast<std::size_t>(
		OaLunarEndReason::HardFootImpact)], 0U);
	EXPECT_EQ(reasons[static_cast<std::size_t>(
		OaLunarEndReason::OutOfBounds)], 0U);
	EXPECT_EQ(reasons[static_cast<std::size_t>(
		OaLunarEndReason::NumericalFailure)], 0U);
	EXPECT_EQ(reasons[static_cast<std::size_t>(
		OaLunarEndReason::ExternalStop)], 0U);
	EXPECT_EQ(reasons[static_cast<std::size_t>(
		OaLunarEndReason::InvalidAction)], 0U);
}
