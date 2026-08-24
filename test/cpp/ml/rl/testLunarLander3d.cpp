#include <ml/rl/lunarLander3d.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <vector>

class LunarTraceDigest {
public:
	void addU64(std::uint64_t inValue) noexcept {
		for (std::uint32_t byteIndex = 0U; byteIndex < 8U; ++byteIndex) {
			const std::uint8_t byte = static_cast<std::uint8_t>(
				(inValue >> (byteIndex * 8U)) & 0xffU);
			hash_ ^= byte;
			hash_ *= 1099511628211ULL;
		}
	}
	void addU32(std::uint32_t inValue) noexcept { addU64(inValue); }
	void addBool(bool inValue) noexcept { addU64(inValue ? 1U : 0U); }
	void addDouble(double inValue) noexcept {
		// freeze the numerical trace, not optimizer-specific FP encodings.
		addU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(
			std::llround(inValue * 1.0e9))));
	}
	void addFloat(float inValue) noexcept {
		addU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(
			std::llround(static_cast<double>(inValue) * 1.0e6))));
	}
	[[nodiscard]] std::uint64_t value() const noexcept { return hash_; }

private:
	std::uint64_t hash_ = 1469598103934665603ULL;
};

static void lunarDigestVector(
	LunarTraceDigest& inOutDigest,
	const oa::vlm::DVec3& inVector) noexcept {
	inOutDigest.addDouble(inVector.x);
	inOutDigest.addDouble(inVector.y);
	inOutDigest.addDouble(inVector.z);
}

static void lunarDigestQuaternion(
	LunarTraceDigest& inOutDigest,
	const oa::vlm::DQuat& inQuaternion) noexcept {
	inOutDigest.addDouble(inQuaternion.w);
	inOutDigest.addDouble(inQuaternion.x);
	inOutDigest.addDouble(inQuaternion.y);
	inOutDigest.addDouble(inQuaternion.z);
}

static void lunarDigestManifest(
	LunarTraceDigest& inOutDigest,
	const oa::LunarEpisodeManifest& inManifest) noexcept {
	inOutDigest.addU32(inManifest.environmentVersion_);
	inOutDigest.addU32(inManifest.randomVersion_);
	inOutDigest.addU32(inManifest.terrainVersion_);
	inOutDigest.addU32(inManifest.physicsVersion_);
	inOutDigest.addU32(inManifest.observationVersion_);
	inOutDigest.addU32(inManifest.rewardVersion_);
	inOutDigest.addU64(inManifest.configFingerprint_);
	inOutDigest.addU64(inManifest.baseSeed_);
	inOutDigest.addU32(inManifest.environmentLane_);
	inOutDigest.addU64(inManifest.episodeIndex_);
	inOutDigest.addU64(inManifest.terrainSeed_);
	inOutDigest.addU64(inManifest.spawnSeed_);
	inOutDigest.addU64(inManifest.domainSeed_);
}

static void lunarDigestState(
	LunarTraceDigest& inOutDigest,
	const oa::LunarLander3dState& inState) noexcept {
	lunarDigestVector(inOutDigest, inState.position_);
	lunarDigestVector(inOutDigest, inState.linearVelocity_);
	lunarDigestQuaternion(inOutDigest, inState.orientation_);
	lunarDigestVector(inOutDigest, inState.angularVelocityBody_);
	inOutDigest.addDouble(inState.fuel_);
	inOutDigest.addU32(static_cast<std::uint32_t>(inState.lastAction_));
	inOutDigest.addDouble(inState.mainThrottle_);
	lunarDigestVector(inOutDigest, inState.attitudeCommandBody_);
	for (const bool contact : inState.bodyContacts_) {
		inOutDigest.addBool(contact);
	}
	for (const double impulse : inState.bodyContactImpulses_) {
		inOutDigest.addDouble(impulse);
	}
	for (const bool contact : inState.footContacts_) {
		inOutDigest.addBool(contact);
	}
	for (const bool onPad : inState.feetOnPad_) {
		inOutDigest.addBool(onPad);
	}
	for (const double impulse : inState.footContactImpulses_) {
		inOutDigest.addDouble(impulse);
	}
	for (const bool rewarded : inState.footContactRewarded_) {
		inOutDigest.addBool(rewarded);
	}
	inOutDigest.addU32(inState.episodeStep_);
	inOutDigest.addU32(inState.stableDwell_);
	inOutDigest.addBool(inState.terminated_);
	inOutDigest.addBool(inState.truncated_);
	inOutDigest.addU32(static_cast<std::uint32_t>(inState.endReason_));
	inOutDigest.addDouble(inState.episodeReturn_);
}

static void lunarDigestReward(
	LunarTraceDigest& inOutDigest,
	const oa::LunarRewardTerms& inReward) noexcept {
	inOutDigest.addDouble(inReward.potentialBefore_);
	inOutDigest.addDouble(inReward.potentialAfter_);
	inOutDigest.addDouble(inReward.shaping_);
	inOutDigest.addDouble(inReward.mainFuelCost_);
	inOutDigest.addDouble(inReward.attitudeFuelCost_);
	inOutDigest.addDouble(inReward.softFootContact_);
	inOutDigest.addDouble(inReward.stableDwell_);
	inOutDigest.addDouble(inReward.terminal_);
	inOutDigest.addDouble(inReward.total_);
}

static oa::LunarEpisodeManifest lunarTestManifest(
	std::uint64_t inSeed = 0x123456789abcdef0ULL,
	std::uint32_t inLane = 3U,
	std::uint64_t inEpisode = 7U) {
	const oa::LunarLander3dConfig config;
	return oa::LunarEpisodeManifest::derive(
		inSeed, inLane, inEpisode, config.contractFingerprint());
}

static oa::LunarEpisodeManifest lunarTestManifestForConfig(
	const oa::LunarLander3dConfig& inConfig,
	std::uint64_t inSeed = 0x123456789abcdef0ULL,
	std::uint32_t inLane = 3U,
	std::uint64_t inEpisode = 7U) {
	return oa::LunarEpisodeManifest::deriveVersioned(
		inSeed, inLane, inEpisode,
		inConfig.environmentVersion_, oa::kLunarTerrainVersion,
		inConfig.physicsVersion_, inConfig.observationVersion_,
		inConfig.rewardVersion_, inConfig.contractFingerprint());
}

static oa::LunarLander3dState lunarTestFlightState(
	const oa::LunarLander3dConfig& inConfig,
	double inHeight = 20.0) {
	oa::LunarLander3dState state;
	state.position_ = {0.0, inHeight, 0.0};
	state.fuel_ = inConfig.fuelCapacity_;
	return state;
}

static double lunarTerrainSlope(
	const oa::LunarTerrainSample& inSample) {
	return std::hypot(inSample.normal_.x, inSample.normal_.z)
		/ inSample.normal_.y;
}

TEST(LunarLander3d, RejectsInvalidConfigurationTerrainAndState) {
	oa::LunarLander3dConfig config;
	EXPECT_TRUE(config.validationError().empty());
	config.mass_ = 0.0;
	EXPECT_FALSE(config.validationError().empty());
	config = {};
	config.physicsSubsteps_ = 0U;
	EXPECT_FALSE(config.validationError().empty());
	config = {};
	config.terrain_.cellsX_ = 0U;
	EXPECT_FALSE(config.validationError().empty());
	config = {};
	config.successReward_ = -1.0;
	EXPECT_FALSE(config.validationError().empty());
	config = {};
	config.failurePenalty_ = 1.0;
	EXPECT_FALSE(config.validationError().empty());

	config = {};
	const oa::LunarEpisodeManifest manifest = lunarTestManifestForConfig(config);
	const oa::LunarTerrain flat = oa::LunarTerrain::createFlat(config.terrain_);
	ASSERT_TRUE(flat.isValid()) << flat.error();
	auto environment = oa::LunarScalarEnvironment::createWithTerrain(
		config, manifest, flat);
	ASSERT_TRUE(environment.isValid()) << environment.error();
	const oa::LunarLander3dState before = environment.state();
	const oa::LunarTransition invalidAction = environment.step(8U);
	ASSERT_TRUE(invalidAction.valid_) << invalidAction.error_;
	EXPECT_TRUE(invalidAction.terminated_);
	EXPECT_FALSE(invalidAction.truncated_);
	EXPECT_EQ(invalidAction.endReason_, oa::LunarEndReason::InvalidAction);
	EXPECT_DOUBLE_EQ(invalidAction.reward_, config.failurePenalty_);
	EXPECT_DOUBLE_EQ(
		invalidAction.rewardTerms_.terminal_, config.failurePenalty_);
	EXPECT_DOUBLE_EQ(
		invalidAction.rewardTerms_.total_, config.failurePenalty_);
	EXPECT_EQ(environment.state().position_, before.position_);
	EXPECT_EQ(environment.state().episodeStep_, before.episodeStep_ + 1U);
	EXPECT_EQ(environment.state().endReason_, oa::LunarEndReason::InvalidAction);

	oa::LunarLander3dState invalidState = before;
	invalidState.fuel_ = config.fuelCapacity_ + 1.0;
	EXPECT_FALSE(environment.setState(invalidState));
	invalidState = before;
	invalidState.terminated_ = true;
	EXPECT_FALSE(environment.setState(invalidState));

	oa::LunarLander3dState directState = lunarTestFlightState(config);
	const oa::LunarPhysicsResult invalidEnum = oa::FnLunarLander::integrate(
		config, flat, static_cast<oa::LunarAction>(99U), directState);
	EXPECT_FALSE(invalidEnum.valid_);

	oa::LunarLander3dConfig mismatchedConfig = config;
	mismatchedConfig.terrain_.maxSlope_ = 0.2;
	directState = lunarTestFlightState(mismatchedConfig);
	const oa::LunarPhysicsResult mismatchedTerrain =
		oa::FnLunarLander::integrate(
			mismatchedConfig, flat, oa::LunarAction::Coast, directState);
	EXPECT_FALSE(mismatchedTerrain.valid_);
	EXPECT_FALSE(oa::LunarScalarEnvironment::createWithTerrain(
		mismatchedConfig, manifest, flat).isValid());
	oa::LunarLander3dConfig changedReward = config;
	changedReward.successReward_ += 1.0;
	const oa::LunarTerrain changedRewardFlat = oa::LunarTerrain::createFlat(
		changedReward.terrain_);
	ASSERT_TRUE(changedRewardFlat.isValid());
	EXPECT_FALSE(oa::LunarScalarEnvironment::createWithTerrain(
		changedReward, manifest, changedRewardFlat).isValid());
	oa::LunarEpisodeManifest tamperedFingerprint = manifest;
	tamperedFingerprint.configFingerprint_ ^= 1U;
	EXPECT_FALSE(oa::LunarScalarEnvironment::createWithTerrain(
		config, tamperedFingerprint, flat).isValid());
}

TEST(LunarLander3d, ManifestDerivationIsVersionedDeterministicAndTamperEvident) {
	const oa::LunarEpisodeManifest first = lunarTestManifest();
	const oa::LunarEpisodeManifest repeated = lunarTestManifest();
	EXPECT_EQ(first, repeated);
	EXPECT_TRUE(first.validationError().empty());
	EXPECT_NE(first.terrainSeed_, first.spawnSeed_);
	EXPECT_NE(first.spawnSeed_, first.domainSeed_);
	EXPECT_DOUBLE_EQ(
		first.sample01(oa::LunarRandomPurpose::Spawn, 11U),
		repeated.sample01(oa::LunarRandomPurpose::Spawn, 11U));

	EXPECT_NE(lunarTestManifest(first.baseSeed_, 4U, first.episodeIndex_), first);
	EXPECT_NE(lunarTestManifest(
		first.baseSeed_, first.environmentLane_, first.episodeIndex_ + 1U), first);

	oa::LunarEpisodeManifest tamperedSeed = first;
	tamperedSeed.spawnSeed_ ^= 1U;
	EXPECT_FALSE(tamperedSeed.validationError().empty());
	oa::LunarEpisodeManifest tamperedPhysics = first;
	++tamperedPhysics.physicsVersion_;
	EXPECT_FALSE(tamperedPhysics.validationError().empty());
	oa::LunarEpisodeManifest tamperedObservation = first;
	++tamperedObservation.observationVersion_;
	EXPECT_FALSE(tamperedObservation.validationError().empty());
	oa::LunarEpisodeManifest tamperedReward = first;
	++tamperedReward.rewardVersion_;
	EXPECT_FALSE(tamperedReward.validationError().empty());

	EXPECT_EQ(first.terrainSeed_, 0x87b6d5a424c9738bULL);
	EXPECT_EQ(first.spawnSeed_, 0xdb0f56e47eab3480ULL);
	EXPECT_EQ(first.domainSeed_, 0x2ce2d29d6f0762e7ULL);
}

TEST(LunarLander3d, SeededTerrainIsDeterministicBoundedAndPadFlat) {
	const oa::LunarTerrainConfig config;
	const oa::LunarEpisodeManifest manifest = lunarTestManifest();
	const oa::LunarTerrain first = oa::LunarTerrain::createSeeded(config, manifest);
	const oa::LunarTerrain repeated = oa::LunarTerrain::createSeeded(config, manifest);
	ASSERT_TRUE(first.isValid()) << first.error();
	ASSERT_TRUE(repeated.isValid()) << repeated.error();
	EXPECT_EQ(first.heights(), repeated.heights());
	const oa::LunarTerrain changed = oa::LunarTerrain::createSeeded(
		config, oa::LunarEpisodeManifest::derive(
			manifest.baseSeed_ + 1U,
			manifest.environmentLane_, manifest.episodeIndex_,
			manifest.configFingerprint_));
	ASSERT_TRUE(changed.isValid()) << changed.error();
	EXPECT_NE(first.heights(), changed.heights());

	for (std::uint32_t vertexZ = 0U; vertexZ <= config.cellsZ_; ++vertexZ) {
		for (std::uint32_t vertexX = 0U; vertexX <= config.cellsX_; ++vertexX) {
			EXPECT_LE(std::abs(first.vertexHeight(vertexX, vertexZ)),
				config.maxAbsHeight_ + 1.0e-12);
		}
	}
	for (std::uint32_t cellZ = 0U; cellZ < config.cellsZ_; ++cellZ) {
		for (std::uint32_t cellX = 0U; cellX < config.cellsX_; ++cellX) {
			const double baseX = first.minX()
				+ static_cast<double>(cellX) * config.cellSize_;
			const double baseZ = first.minZ()
				+ static_cast<double>(cellZ) * config.cellSize_;
			const oa::LunarTerrainSample lower = first.query(
				baseX + 0.75 * config.cellSize_,
				baseZ + 0.25 * config.cellSize_);
			const oa::LunarTerrainSample upper = first.query(
				baseX + 0.25 * config.cellSize_,
				baseZ + 0.75 * config.cellSize_);
			ASSERT_TRUE(lower.inBounds_);
			ASSERT_TRUE(upper.inBounds_);
			EXPECT_LE(lunarTerrainSlope(lower), config.maxSlope_ + 1.0e-12);
			EXPECT_LE(lunarTerrainSlope(upper), config.maxSlope_ + 1.0e-12);
		}
	}

	for (const double positionX : {-3.0, -1.0, 0.0, 1.0, 3.0}) {
		for (const double positionZ : {-3.0, -1.0, 0.0, 1.0, 3.0}) {
			const oa::LunarTerrainSample pad = first.query(positionX, positionZ);
			ASSERT_TRUE(pad.inBounds_);
			EXPECT_DOUBLE_EQ(pad.height_, 0.0);
			EXPECT_DOUBLE_EQ(pad.normal_.x, 0.0);
			EXPECT_DOUBLE_EQ(pad.normal_.y, 1.0);
			EXPECT_DOUBLE_EQ(pad.normal_.z, 0.0);
			EXPECT_TRUE(first.isOnPad(positionX, positionZ));
		}
	}
}

TEST(LunarLander3d, RecordedActionTraceHasFrozenSameBuildDigest) {
	oa::LunarLander3dConfig config;
	config.maxEpisodeSteps_ = 96U;
	const oa::LunarEpisodeManifest manifest =
		lunarTestManifestForConfig(config);
	auto environment = oa::LunarScalarEnvironment::createSeeded(config, manifest);
	ASSERT_TRUE(environment.isValid()) << environment.error();
	LunarTraceDigest digest;
	lunarDigestManifest(digest, manifest);
	for (const double height : environment.terrain().heights()) {
		digest.addDouble(height);
	}
	lunarDigestState(digest, environment.state());
	const std::array<oa::LunarAction, 8U> recordedActions = {
		oa::LunarAction::Coast,
		oa::LunarAction::MainEngine,
		oa::LunarAction::PitchPositive,
		oa::LunarAction::PitchNegative,
		oa::LunarAction::RollPositive,
		oa::LunarAction::RollNegative,
		oa::LunarAction::YawPositive,
		oa::LunarAction::YawNegative,
	};
	for (std::uint32_t step = 0U; step < config.maxEpisodeSteps_; ++step) {
		const oa::LunarAction action = recordedActions[step % recordedActions.size()];
		digest.addU32(static_cast<std::uint32_t>(action));
		const oa::LunarTransition transition = environment.step(
			static_cast<std::uint32_t>(action));
		ASSERT_TRUE(transition.valid_) << "step=" << step << " " << transition.error_;
		lunarDigestState(digest, environment.state());
		lunarDigestReward(digest, transition.rewardTerms_);
		digest.addDouble(transition.reward_);
		digest.addBool(transition.terminated_);
		digest.addBool(transition.truncated_);
		digest.addU32(static_cast<std::uint32_t>(transition.endReason_));
		for (const float observation : transition.observation_) {
			digest.addFloat(observation);
		}
	}
	EXPECT_FALSE(environment.state().terminated_);
	EXPECT_TRUE(environment.state().truncated_);
	EXPECT_EQ(environment.state().endReason_, oa::LunarEndReason::TimeLimit);
	EXPECT_EQ(digest.value(), 0xa85ea7bfd77f9ab5ULL);
}

TEST(LunarLander3d, TerrainFreezesVerticesEdgesDiagonalsAndNormals) {
	oa::LunarTerrainConfig config;
	config.cellsX_ = 2U;
	config.cellsZ_ = 2U;
	config.cellSize_ = 1.0;
	config.maxAbsHeight_ = 10.0;
	config.maxSlope_ = 20.0;
	config.padHalfExtent_ = 0.0;
	config.padTransitionWidth_ = 0.0;
	const std::vector<double> heights = {
		0.0, 1.0, 4.0,
		2.0, 4.0, 8.0,
		3.0, 6.0, 9.0,
	};
	const oa::LunarTerrain terrain = oa::LunarTerrain::createFromHeights(
		config, heights);
	ASSERT_TRUE(terrain.isValid()) << terrain.error();

	const oa::LunarTerrainSample vertex = terrain.query(0.0, 0.0);
	ASSERT_TRUE(vertex.inBounds_);
	EXPECT_EQ(vertex.cellX_, 1U);
	EXPECT_EQ(vertex.cellZ_, 1U);
	EXPECT_DOUBLE_EQ(vertex.localX_, 0.0);
	EXPECT_DOUBLE_EQ(vertex.localZ_, 0.0);
	EXPECT_DOUBLE_EQ(vertex.height_, 4.0);

	const oa::LunarTerrainSample edge = terrain.query(0.0, -0.5);
	ASSERT_TRUE(edge.inBounds_);
	EXPECT_EQ(edge.cellX_, 1U);
	EXPECT_EQ(edge.cellZ_, 0U);
	EXPECT_EQ(edge.triangle_, oa::LunarTerrainTriangle::UpperLeft);
	EXPECT_DOUBLE_EQ(edge.height_, 2.5);

	const oa::LunarTerrainSample diagonal = terrain.query(-0.5, -0.5);
	ASSERT_TRUE(diagonal.inBounds_);
	EXPECT_EQ(diagonal.triangle_, oa::LunarTerrainTriangle::LowerRight);
	EXPECT_DOUBLE_EQ(diagonal.height_, 2.0);
	const oa::LunarTerrainSample lower = terrain.query(-0.25, -0.75);
	const oa::LunarTerrainSample upper = terrain.query(-0.75, -0.25);
	ASSERT_TRUE(lower.inBounds_);
	ASSERT_TRUE(upper.inBounds_);
	EXPECT_DOUBLE_EQ(lower.height_, 1.5);
	EXPECT_DOUBLE_EQ(upper.height_, 2.0);
	const double lowerNormalScale = 1.0 / std::sqrt(11.0);
	EXPECT_NEAR(lower.normal_.x, -lowerNormalScale, 1.0e-15);
	EXPECT_NEAR(lower.normal_.y, lowerNormalScale, 1.0e-15);
	EXPECT_NEAR(lower.normal_.z, -3.0 * lowerNormalScale, 1.0e-15);
	EXPECT_NEAR(upper.normal_.x, -2.0 / 3.0, 1.0e-15);
	EXPECT_NEAR(upper.normal_.y, 1.0 / 3.0, 1.0e-15);
	EXPECT_NEAR(upper.normal_.z, -2.0 / 3.0, 1.0e-15);

	const oa::LunarTerrainSample maximum = terrain.query(1.0, 1.0);
	ASSERT_TRUE(maximum.inBounds_);
	EXPECT_EQ(maximum.cellX_, 1U);
	EXPECT_EQ(maximum.cellZ_, 1U);
	EXPECT_DOUBLE_EQ(maximum.localX_, 1.0);
	EXPECT_DOUBLE_EQ(maximum.localZ_, 1.0);
	EXPECT_DOUBLE_EQ(maximum.height_, 9.0);
	EXPECT_FALSE(terrain.query(terrain.minX() - 0.01, 0.0).inBounds_);
	EXPECT_FALSE(terrain.query(0.0, terrain.maxZ() + 0.01).inBounds_);
}

TEST(LunarLander3d, FreeFallMatchesSemiImplicitAnalyticSolution) {
	oa::LunarLander3dConfig config;
	const auto manifest = lunarTestManifestForConfig(config);
	auto environment = oa::LunarScalarEnvironment::createFlat(config, manifest);
	ASSERT_TRUE(environment.isValid()) << environment.error();
	oa::LunarLander3dState state = lunarTestFlightState(config);
	state.linearVelocity_ = {0.25, 1.0, -0.5};
	ASSERT_TRUE(environment.setState(state));
	const oa::LunarTransition transition = environment.step(
		static_cast<std::uint32_t>(oa::LunarAction::Coast));
	ASSERT_TRUE(transition.valid_) << transition.error_;
	ASSERT_FALSE(transition.terminated_);
	ASSERT_FALSE(transition.truncated_);

	const double substep = config.policyTimeStep_
		/ static_cast<double>(config.physicsSubsteps_);
	const double substepCount = static_cast<double>(config.physicsSubsteps_);
	const double expectedY = state.position_.y
		+ state.linearVelocity_.y * config.policyTimeStep_
		- config.gravity_ * substep * substep
			* substepCount * (substepCount + 1.0) * 0.5;
	EXPECT_NEAR(environment.state().position_.x,
		state.position_.x + state.linearVelocity_.x * config.policyTimeStep_,
		1.0e-14);
	EXPECT_NEAR(environment.state().position_.y, expectedY, 1.0e-14);
	EXPECT_NEAR(environment.state().position_.z,
		state.position_.z + state.linearVelocity_.z * config.policyTimeStep_,
		1.0e-14);
	EXPECT_NEAR(environment.state().linearVelocity_.y,
		state.linearVelocity_.y - config.gravity_ * config.policyTimeStep_,
		1.0e-14);
}

TEST(LunarLander3d, MainThrustAndFuelAreIsolated) {
	oa::LunarLander3dConfig config;
	config.gravity_ = 0.0;
	auto environment = oa::LunarScalarEnvironment::createFlat(
		config, lunarTestManifestForConfig(config));
	ASSERT_TRUE(environment.isValid()) << environment.error();
	const oa::LunarLander3dState state = lunarTestFlightState(config);
	ASSERT_TRUE(environment.setState(state));
	const oa::LunarTransition transition = environment.step(
		static_cast<std::uint32_t>(oa::LunarAction::MainEngine));
	ASSERT_TRUE(transition.valid_) << transition.error_;
	const double acceleration = config.mainThrust_ / config.mass_;
	const double substep = config.policyTimeStep_
		/ static_cast<double>(config.physicsSubsteps_);
	const double substepCount = static_cast<double>(config.physicsSubsteps_);
	EXPECT_NEAR(environment.state().linearVelocity_.y,
		acceleration * config.policyTimeStep_, 1.0e-14);
	EXPECT_NEAR(environment.state().position_.y,
		state.position_.y + acceleration * substep * substep
			* substepCount * (substepCount + 1.0) * 0.5,
		1.0e-14);
	EXPECT_DOUBLE_EQ(environment.state().linearVelocity_.x, 0.0);
	EXPECT_DOUBLE_EQ(environment.state().linearVelocity_.z, 0.0);
	EXPECT_NEAR(environment.state().fuel_,
		config.fuelCapacity_ - config.mainFuelRate_ * config.policyTimeStep_,
		1.0e-12);
	EXPECT_DOUBLE_EQ(environment.state().angularVelocityBody_.lengthSquared(), 0.0);
}

TEST(LunarLander3d, AttitudeTorqueIsIsolated) {
	oa::LunarLander3dConfig config;
	config.gravity_ = 0.0;
	auto environment = oa::LunarScalarEnvironment::createFlat(
		config, lunarTestManifestForConfig(config));
	ASSERT_TRUE(environment.isValid()) << environment.error();
	const oa::LunarLander3dState state = lunarTestFlightState(config);
	ASSERT_TRUE(environment.setState(state));
	const oa::LunarTransition transition = environment.step(
		static_cast<std::uint32_t>(oa::LunarAction::PitchPositive));
	ASSERT_TRUE(transition.valid_) << transition.error_;
	EXPECT_NEAR(environment.state().angularVelocityBody_.x,
		config.attitudeTorque_ / config.diagonalInertia_.x
			* config.policyTimeStep_, 1.0e-14);
	EXPECT_DOUBLE_EQ(environment.state().angularVelocityBody_.y, 0.0);
	EXPECT_DOUBLE_EQ(environment.state().angularVelocityBody_.z, 0.0);
	EXPECT_DOUBLE_EQ(environment.state().linearVelocity_.lengthSquared(), 0.0);
	EXPECT_NEAR(environment.state().fuel_,
		config.fuelCapacity_
			- config.attitudeFuelRate_ * config.policyTimeStep_,
		1.0e-14);
	EXPECT_NEAR(environment.state().orientation_.norm(), 1.0, 1.0e-15);
}

TEST(LunarLander3d, QuaternionRenormalizationSurvivesLongTorqueTrace) {
	oa::LunarLander3dConfig config;
	config.gravity_ = 0.0;
	config.maxEpisodeSteps_ = 2000U;
	auto environment = oa::LunarScalarEnvironment::createFlat(
		config, lunarTestManifestForConfig(config));
	ASSERT_TRUE(environment.isValid()) << environment.error();
	ASSERT_TRUE(environment.setState(lunarTestFlightState(config)));
	const std::array<oa::LunarAction, 6U> actions = {
		oa::LunarAction::PitchPositive, oa::LunarAction::RollNegative,
		oa::LunarAction::YawPositive, oa::LunarAction::PitchNegative,
		oa::LunarAction::RollPositive, oa::LunarAction::YawNegative,
	};
	for (std::uint32_t step = 0U; step < 600U; ++step) {
		const oa::LunarTransition transition = environment.step(
			static_cast<std::uint32_t>(actions[step % actions.size()]));
		ASSERT_TRUE(transition.valid_) << "step=" << step << " " << transition.error_;
		ASSERT_FALSE(transition.terminated_) << "step=" << step;
		ASSERT_FALSE(transition.truncated_) << "step=" << step;
		EXPECT_NEAR(environment.state().orientation_.norm(), 1.0, 2.0e-15);
	}
}

TEST(LunarLander3d, FuelDepletionClampsThrustWithoutChangingMass) {
	oa::LunarLander3dConfig config;
	config.gravity_ = 0.0;
	config.fuelCapacity_ = 0.01;
	auto environment = oa::LunarScalarEnvironment::createFlat(
		config, lunarTestManifestForConfig(config));
	ASSERT_TRUE(environment.isValid()) << environment.error();
	ASSERT_TRUE(environment.setState(lunarTestFlightState(config)));
	const oa::LunarTransition first = environment.step(
		static_cast<std::uint32_t>(oa::LunarAction::MainEngine));
	ASSERT_TRUE(first.valid_) << first.error_;
	const double burnDuration = config.fuelCapacity_ / config.mainFuelRate_;
	const double expectedVelocity = config.mainThrust_ / config.mass_
		* burnDuration;
	EXPECT_DOUBLE_EQ(environment.state().fuel_, 0.0);
	EXPECT_NEAR(environment.state().linearVelocity_.y, expectedVelocity, 1.0e-14);
	const double velocityBefore = environment.state().linearVelocity_.y;
	const oa::LunarTransition second = environment.step(
		static_cast<std::uint32_t>(oa::LunarAction::MainEngine));
	ASSERT_TRUE(second.valid_) << second.error_;
	EXPECT_DOUBLE_EQ(environment.state().fuel_, 0.0);
	EXPECT_DOUBLE_EQ(environment.state().linearVelocity_.y, velocityBefore);
}

TEST(LunarLander3d, ObservationHasFrozenThirtyThreeValueLayout) {
	oa::LunarLander3dConfig config;
	auto environment = oa::LunarScalarEnvironment::createFlat(
		config, lunarTestManifestForConfig(config));
	ASSERT_TRUE(environment.isValid()) << environment.error();
	oa::LunarLander3dState state = lunarTestFlightState(config, 6.0);
	state.position_.x = 2.0;
	state.position_.z = -3.0;
	state.linearVelocity_ = {0.4, -0.8, 1.2};
	state.angularVelocityBody_ = {0.1, -0.2, 0.3};
	state.fuel_ = config.fuelCapacity_ * 0.5;
	state.footContacts_ = {true, false, true, false};
	ASSERT_TRUE(environment.setState(state));
	const auto observation = environment.observation();
	EXPECT_EQ(observation.size(), 33U);
	for (const float value : observation) {
		EXPECT_TRUE(std::isfinite(value));
		EXPECT_GE(value, -1.0F);
		EXPECT_LE(value, 1.0F);
	}
	EXPECT_FLOAT_EQ(observation[0], static_cast<float>(
		state.position_.x / config.positionObservationScale_));
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
	oa::LunarLander3dConfig config;
	config.hardFootImpactSpeed_ = 10.0;
	config.safeDwellSteps_ = 1000U;
	auto environment = oa::LunarScalarEnvironment::createFlat(
		config, lunarTestManifestForConfig(config));
	ASSERT_TRUE(environment.isValid()) << environment.error();
	oa::LunarLander3dState state = lunarTestFlightState(config, 1.10);
	state.linearVelocity_.y = -0.1;
	ASSERT_TRUE(environment.setState(state));
	const oa::LunarTransition transition = environment.step(
		static_cast<std::uint32_t>(oa::LunarAction::Coast));
	ASSERT_TRUE(transition.valid_) << transition.error_;
	EXPECT_TRUE(environment.state().isFinite());
	EXPECT_TRUE(transition.contact_.isFinite());
	EXPECT_TRUE(transition.contact_.bounded_);
	EXPECT_TRUE(transition.contact_.footContactOccurred_);
	EXPECT_GT(transition.contact_.contactCount_, 0U);
	EXPECT_LE(transition.contact_.maximumNormalImpulse_, config.maxContactImpulse_);
	EXPECT_LE(transition.contact_.maximumFrictionImpulse_, config.maxContactImpulse_);
	const double maximumCorrection = static_cast<double>(
		config.physicsSubsteps_ * config.contactIterations_)
		* static_cast<double>(
			config.bodySupports_.size() + config.footSupports_.size())
		* config.maxPositionCorrectionPerContact_;
	EXPECT_LE(transition.contact_.totalPositionCorrection_,
		maximumCorrection + 1.0e-12);
}

TEST(LunarLander3d, SlopeContactIsFiniteAndBounded) {
	oa::LunarLander3dConfig config;
	config.hardFootImpactSpeed_ = 10.0;
	config.safeDwellSteps_ = 1000U;
	std::vector<double> heights;
	heights.reserve(static_cast<std::size_t>(config.terrain_.cellsX_ + 1U)
		* static_cast<std::size_t>(config.terrain_.cellsZ_ + 1U));
	for (std::uint32_t vertexZ = 0U;
		vertexZ <= config.terrain_.cellsZ_;
		++vertexZ) {
		for (std::uint32_t vertexX = 0U;
			vertexX <= config.terrain_.cellsX_;
			++vertexX) {
			const double positionX = -static_cast<double>(config.terrain_.cellsX_)
				* config.terrain_.cellSize_ * 0.5
				+ static_cast<double>(vertexX) * config.terrain_.cellSize_;
			heights.push_back(positionX * 0.08);
		}
	}
	const oa::LunarTerrain slope = oa::LunarTerrain::createFromHeights(
		config.terrain_, heights);
	ASSERT_TRUE(slope.isValid()) << slope.error();
	auto environment = oa::LunarScalarEnvironment::createWithTerrain(
		config, lunarTestManifestForConfig(config), slope);
	ASSERT_TRUE(environment.isValid()) << environment.error();
	oa::LunarLander3dState state = lunarTestFlightState(config, 1.20);
	state.linearVelocity_.y = -0.1;
	ASSERT_TRUE(environment.setState(state));
	const oa::LunarTransition transition = environment.step(
		static_cast<std::uint32_t>(oa::LunarAction::Coast));
	ASSERT_TRUE(transition.valid_) << transition.error_;
	EXPECT_TRUE(environment.state().isFinite());
	EXPECT_TRUE(transition.contact_.isFinite());
	EXPECT_TRUE(transition.contact_.bounded_);
	EXPECT_TRUE(transition.contact_.footContactOccurred_);
	EXPECT_LE(transition.contact_.maximumNormalImpulse_, config.maxContactImpulse_);
}

TEST(LunarLander3d, EndReasonsKeepTerminationAndTruncationDistinct) {
	oa::LunarLander3dConfig successConfig;
	successConfig.gravity_ = 0.0;
	successConfig.safeDwellSteps_ = 2U;
	auto success = oa::LunarScalarEnvironment::createFlat(
		successConfig, lunarTestManifestForConfig(successConfig));
	ASSERT_TRUE(success.isValid()) << success.error();
	ASSERT_TRUE(success.setState(lunarTestFlightState(successConfig, 1.15)));
	const oa::LunarTransition dwell = success.step(
		static_cast<std::uint32_t>(oa::LunarAction::Coast));
	ASSERT_TRUE(dwell.valid_);
	EXPECT_FALSE(dwell.terminated_);
	EXPECT_FALSE(dwell.truncated_);
	const oa::LunarTransition landed = success.step(
		static_cast<std::uint32_t>(oa::LunarAction::Coast));
	ASSERT_TRUE(landed.valid_);
	EXPECT_TRUE(landed.terminated_);
	EXPECT_FALSE(landed.truncated_);
	EXPECT_EQ(landed.endReason_, oa::LunarEndReason::SafeLanding);
	EXPECT_FALSE(success.step(0U).valid_);

	oa::LunarLander3dConfig impactConfig;
	impactConfig.gravity_ = 0.0;
	const oa::LunarEpisodeManifest impactManifest =
		lunarTestManifestForConfig(impactConfig);
	auto bodyImpact = oa::LunarScalarEnvironment::createFlat(
		impactConfig, impactManifest);
	ASSERT_TRUE(bodyImpact.isValid());
	ASSERT_TRUE(bodyImpact.setState(lunarTestFlightState(impactConfig, 0.60)));
	const oa::LunarTransition body = bodyImpact.step(0U);
	ASSERT_TRUE(body.valid_);
	EXPECT_TRUE(body.terminated_);
	EXPECT_FALSE(body.truncated_);
	EXPECT_EQ(body.endReason_, oa::LunarEndReason::BodyImpact);

	auto hardImpact = oa::LunarScalarEnvironment::createFlat(
		impactConfig, impactManifest);
	ASSERT_TRUE(hardImpact.isValid());
	oa::LunarLander3dState hardState = lunarTestFlightState(impactConfig, 1.17);
	hardState.linearVelocity_.y = -3.0;
	ASSERT_TRUE(hardImpact.setState(hardState));
	const oa::LunarTransition hard = hardImpact.step(0U);
	ASSERT_TRUE(hard.valid_);
	EXPECT_TRUE(hard.terminated_);
	EXPECT_FALSE(hard.truncated_);
	EXPECT_EQ(hard.endReason_, oa::LunarEndReason::HardFootImpact);

	auto outOfBounds = oa::LunarScalarEnvironment::createFlat(
		impactConfig, impactManifest);
	ASSERT_TRUE(outOfBounds.isValid());
	oa::LunarLander3dState outside = lunarTestFlightState(impactConfig);
	outside.position_.x = outOfBounds.terrain().maxX() + 0.5;
	ASSERT_TRUE(outOfBounds.setState(outside));
	const oa::LunarTransition bounds = outOfBounds.step(0U);
	ASSERT_TRUE(bounds.valid_);
	EXPECT_TRUE(bounds.terminated_);
	EXPECT_FALSE(bounds.truncated_);
	EXPECT_EQ(bounds.endReason_, oa::LunarEndReason::OutOfBounds);

	oa::LunarLander3dConfig horizonConfig;
	horizonConfig.gravity_ = 0.0;
	horizonConfig.maxEpisodeSteps_ = 1U;
	auto horizon = oa::LunarScalarEnvironment::createFlat(
		horizonConfig, lunarTestManifestForConfig(horizonConfig));
	ASSERT_TRUE(horizon.isValid());
	ASSERT_TRUE(horizon.setState(lunarTestFlightState(horizonConfig)));
	const oa::LunarTransition timeout = horizon.step(0U);
	ASSERT_TRUE(timeout.valid_);
	EXPECT_FALSE(timeout.terminated_);
	EXPECT_TRUE(timeout.truncated_);
	EXPECT_EQ(timeout.endReason_, oa::LunarEndReason::TimeLimit);
	EXPECT_FALSE(horizon.step(0U).valid_);

	auto stopped = oa::LunarScalarEnvironment::createFlat(
		impactConfig, impactManifest);
	ASSERT_TRUE(stopped.isValid());
	const oa::LunarTransition external = stopped.step(0U, true);
	ASSERT_TRUE(external.valid_);
	EXPECT_FALSE(external.terminated_);
	EXPECT_TRUE(external.truncated_);
	EXPECT_EQ(external.endReason_, oa::LunarEndReason::ExternalStop);

	auto numerical = oa::LunarScalarEnvironment::createFlat(
		impactConfig, impactManifest);
	ASSERT_TRUE(numerical.isValid());
	oa::LunarLander3dState huge = lunarTestFlightState(impactConfig);
	huge.position_.y = std::numeric_limits<double>::max();
	huge.linearVelocity_.y = std::numeric_limits<double>::max();
	ASSERT_TRUE(numerical.setState(huge));
	const oa::LunarTransition failed = numerical.step(0U);
	ASSERT_TRUE(failed.valid_);
	EXPECT_TRUE(failed.terminated_);
	EXPECT_FALSE(failed.truncated_);
	EXPECT_EQ(failed.endReason_, oa::LunarEndReason::NumericalFailure);
	EXPECT_TRUE(std::isfinite(failed.reward_));
}

TEST(LunarLander3d, RewardDiagnosticsSumAndFreezeTerminalPotentialHandling) {
	oa::LunarLander3dConfig config;
	config.gravity_ = 0.0;
	auto flight = oa::LunarScalarEnvironment::createFlat(
		config, lunarTestManifestForConfig(config));
	ASSERT_TRUE(flight.isValid());
	ASSERT_TRUE(flight.setState(lunarTestFlightState(config)));
	const oa::LunarTransition main = flight.step(
		static_cast<std::uint32_t>(oa::LunarAction::MainEngine));
	ASSERT_TRUE(main.valid_);
	EXPECT_TRUE(main.rewardTerms_.isFinite());
	EXPECT_DOUBLE_EQ(main.rewardTerms_.total_, main.rewardTerms_.sum());
	EXPECT_DOUBLE_EQ(main.reward_, main.rewardTerms_.total_);
	EXPECT_LT(main.rewardTerms_.mainFuelCost_, 0.0);
	EXPECT_DOUBLE_EQ(main.rewardTerms_.attitudeFuelCost_, 0.0);

	auto failure = oa::LunarScalarEnvironment::createFlat(
		config, lunarTestManifestForConfig(config));
	ASSERT_TRUE(failure.isValid());
	ASSERT_TRUE(failure.setState(lunarTestFlightState(config, 0.60)));
	const oa::LunarTransition terminal = failure.step(0U);
	ASSERT_TRUE(terminal.valid_);
	ASSERT_TRUE(terminal.terminated_);
	EXPECT_DOUBLE_EQ(terminal.rewardTerms_.shaping_,
		-terminal.rewardTerms_.potentialBefore_);
	EXPECT_DOUBLE_EQ(terminal.rewardTerms_.terminal_, config.failurePenalty_);
	EXPECT_DOUBLE_EQ(terminal.rewardTerms_.total_, terminal.rewardTerms_.sum());

	oa::LunarLander3dConfig horizonConfig = config;
	horizonConfig.maxEpisodeSteps_ = 1U;
	auto horizon = oa::LunarScalarEnvironment::createFlat(
		horizonConfig, lunarTestManifestForConfig(horizonConfig));
	ASSERT_TRUE(horizon.isValid());
	ASSERT_TRUE(horizon.setState(lunarTestFlightState(horizonConfig)));
	const oa::LunarTransition truncated = horizon.step(0U);
	ASSERT_TRUE(truncated.valid_);
	ASSERT_TRUE(truncated.truncated_);
	EXPECT_NEAR(truncated.rewardTerms_.shaping_,
		horizonConfig.rewardGamma_ * truncated.rewardTerms_.potentialAfter_
			- truncated.rewardTerms_.potentialBefore_,
		1.0e-15);
	EXPECT_DOUBLE_EQ(truncated.rewardTerms_.terminal_, 0.0);
}

TEST(LunarLander3d, PotentialShapingCannotFarmHoverCyclesOrDelayedTermination) {
	oa::LunarLander3dConfig config;
	config.gravity_ = 0.0;
	config.maxEpisodeSteps_ = 1000U;
	auto hover = oa::LunarScalarEnvironment::createFlat(
		config, lunarTestManifestForConfig(config));
	ASSERT_TRUE(hover.isValid()) << hover.error();
	ASSERT_TRUE(hover.setState(lunarTestFlightState(config, 10.0)));
	const double fixedPotential = oa::FnLunarLander::potential(
		config, hover.state());
	double discountedHoverShaping = 0.0;
	double discount = 1.0;
	constexpr std::uint32_t hoverSteps = 120U;
	for (std::uint32_t step = 0U; step < hoverSteps; ++step) {
		const oa::LunarTransition transition = hover.step(0U);
		ASSERT_TRUE(transition.valid_);
		ASSERT_FALSE(transition.terminated_);
		ASSERT_FALSE(transition.truncated_);
		EXPECT_DOUBLE_EQ(transition.rewardTerms_.mainFuelCost_, 0.0);
		EXPECT_DOUBLE_EQ(transition.rewardTerms_.attitudeFuelCost_, 0.0);
		EXPECT_DOUBLE_EQ(transition.rewardTerms_.softFootContact_, 0.0);
		EXPECT_DOUBLE_EQ(transition.rewardTerms_.stableDwell_, 0.0);
		EXPECT_DOUBLE_EQ(transition.rewardTerms_.terminal_, 0.0);
		discountedHoverShaping += discount * transition.rewardTerms_.shaping_;
		discount *= config.rewardGamma_;
	}
	const double expectedHoverShaping = fixedPotential
		* (std::pow(config.rewardGamma_, static_cast<double>(hoverSteps)) - 1.0);
	EXPECT_NEAR(discountedHoverShaping, expectedHoverShaping, 2.0e-14);
	EXPECT_LE(discountedHoverShaping, -fixedPotential + 1.0e-14);

	oa::LunarLander3dState cycleStateA = lunarTestFlightState(config, 9.0);
	cycleStateA.position_.x = 1.0;
	oa::LunarLander3dState cycleStateB = lunarTestFlightState(config, 7.0);
	cycleStateB.position_.x = -1.0;
	cycleStateB.linearVelocity_.z = 0.5;
	const double potentialA = oa::FnLunarLander::potential(config, cycleStateA);
	const double potentialB = oa::FnLunarLander::potential(config, cycleStateB);
	const double shapingAB = config.rewardGamma_ * potentialB - potentialA;
	const double shapingBA = config.rewardGamma_ * potentialA - potentialB;
	const double discountedCycle = shapingAB + config.rewardGamma_ * shapingBA;
	EXPECT_NEAR(discountedCycle,
		(std::pow(config.rewardGamma_, 2.0) - 1.0) * potentialA,
		1.0e-15);

	for (const std::uint32_t delaySteps : {0U, 1U, 17U, 120U}) {
		double discountedDelayedShaping = 0.0;
		double delayedDiscount = 1.0;
		for (std::uint32_t step = 0U; step < delaySteps; ++step) {
			discountedDelayedShaping += delayedDiscount
				* (config.rewardGamma_ * fixedPotential - fixedPotential);
			delayedDiscount *= config.rewardGamma_;
		}
		discountedDelayedShaping += delayedDiscount * (-fixedPotential);
		EXPECT_NEAR(discountedDelayedShaping, -fixedPotential, 2.0e-14)
			<< "delay_steps=" << delaySteps;
	}

	auto terminal = oa::LunarScalarEnvironment::createFlat(
		config, lunarTestManifestForConfig(config));
	ASSERT_TRUE(terminal.isValid());
	ASSERT_TRUE(terminal.setState(lunarTestFlightState(config, 0.60)));
	const oa::LunarTransition failure = terminal.step(0U);
	ASSERT_TRUE(failure.valid_);
	ASSERT_TRUE(failure.terminated_);
	EXPECT_DOUBLE_EQ(failure.rewardTerms_.terminal_, config.failurePenalty_);
	EXPECT_FALSE(terminal.step(0U).valid_);
}

TEST(LunarLander3d, PartialRestSoftContactRewardIsEpisodeBounded) {
	oa::LunarLander3dConfig config;
	config.gravity_ = 0.0;
	config.safeDwellSteps_ = 1000U;
	for (std::size_t footIndex = 1U;
		footIndex < config.footSupports_.size();
		++footIndex) {
		config.footSupports_[footIndex].bodyOffset_.y = -0.5;
	}
	auto environment = oa::LunarScalarEnvironment::createFlat(
		config, lunarTestManifestForConfig(config));
	ASSERT_TRUE(environment.isValid()) << environment.error();
	ASSERT_TRUE(environment.setState(lunarTestFlightState(config, 1.15)));

	double totalSoftContact = 0.0;
	for (std::uint32_t step = 0U; step < 64U; ++step) {
		const oa::LunarTransition transition = environment.step(0U);
		ASSERT_TRUE(transition.valid_);
		ASSERT_FALSE(transition.terminated_);
		ASSERT_FALSE(transition.truncated_);
		totalSoftContact += transition.rewardTerms_.softFootContact_;
		EXPECT_TRUE(environment.state().footContacts_[0]);
		EXPECT_FALSE(environment.state().footContacts_[1]);
		EXPECT_FALSE(environment.state().footContacts_[2]);
		EXPECT_FALSE(environment.state().footContacts_[3]);
	}
	EXPECT_DOUBLE_EQ(totalSoftContact, config.softFootContactReward_);
	EXPECT_TRUE(environment.state().footContactRewarded_[0]);
	EXPECT_FALSE(environment.state().footContactRewarded_[1]);

	oa::LunarLander3dState lifted = environment.state();
	lifted.position_.y = 2.0;
	ASSERT_TRUE(environment.setState(lifted));
	const oa::LunarTransition noContact = environment.step(0U);
	ASSERT_TRUE(noContact.valid_);
	EXPECT_DOUBLE_EQ(noContact.rewardTerms_.softFootContact_, 0.0);
	EXPECT_FALSE(environment.state().footContacts_[0]);

	oa::LunarLander3dState returned = environment.state();
	returned.position_.y = 1.15;
	ASSERT_TRUE(environment.setState(returned));
	const oa::LunarTransition repeatedContact = environment.step(0U);
	ASSERT_TRUE(repeatedContact.valid_);
	EXPECT_TRUE(environment.state().footContacts_[0]);
	EXPECT_DOUBLE_EQ(repeatedContact.rewardTerms_.softFootContact_, 0.0);
}

TEST(LunarLander3d, ScriptedControllerReachesSafeLanding) {
	oa::LunarLander3dConfig config;
	config.safeDwellSteps_ = 12U;
	config.maxEpisodeSteps_ = 1200U;
	auto environment = oa::LunarScalarEnvironment::createFlat(
		config, lunarTestManifestForConfig(config));
	ASSERT_TRUE(environment.isValid()) << environment.error();
	oa::LunarLander3dState state = lunarTestFlightState(config, 4.0);
	state.linearVelocity_.y = -0.2;
	ASSERT_TRUE(environment.setState(state));

	for (std::uint32_t step = 0U;
		step < config.maxEpisodeSteps_ and not environment.state().terminated_
			and not environment.state().truncated_;
		++step) {
		const oa::LunarAction action = oa::lunarScriptedLandingAction(
			config, environment.state());
		const oa::LunarTransition transition = environment.step(
			static_cast<std::uint32_t>(action));
		ASSERT_TRUE(transition.valid_) << "step=" << step << " " << transition.error_;
	}
	EXPECT_TRUE(environment.state().terminated_);
	EXPECT_FALSE(environment.state().truncated_);
	EXPECT_EQ(environment.state().endReason_, oa::LunarEndReason::SafeLanding);
}

TEST(LunarLander3d, ScriptedControllerCommandsCorrectiveBodyTorques) {
	oa::LunarLander3dConfig config;
	oa::LunarLander3dState state = lunarTestFlightState(config, 4.0);
	state.linearVelocity_ = {};

	state.orientation_ = oa::vlm::DQuat::fromAxisAngle(
		{1.0, 0.0, 0.0}, 0.04);
	EXPECT_EQ(
		oa::lunarScriptedLandingAction(config, state),
		oa::LunarAction::PitchNegative);

	state.orientation_ = oa::vlm::DQuat::fromAxisAngle(
		{0.0, 0.0, 1.0}, 0.04);
	EXPECT_EQ(
		oa::lunarScriptedLandingAction(config, state),
		oa::LunarAction::RollNegative);

	state.orientation_ = {};
	state.angularVelocityBody_ = {0.04, 0.0, 0.0};
	EXPECT_EQ(
		oa::lunarScriptedLandingAction(config, state),
		oa::LunarAction::PitchNegative);

	state.angularVelocityBody_ = {0.0, 0.0, 0.04};
	EXPECT_EQ(
		oa::lunarScriptedLandingAction(config, state),
		oa::LunarAction::RollNegative);

	state = lunarTestFlightState(config, 4.0);
	state.position_.x = 1.0;
	EXPECT_EQ(
		oa::lunarScriptedLandingAction(config, state),
		oa::LunarAction::RollPositive);

	state = lunarTestFlightState(config, 4.0);
	state.position_.z = 1.0;
	EXPECT_EQ(
		oa::lunarScriptedLandingAction(config, state),
		oa::LunarAction::PitchNegative);
}

TEST(LunarLander3d, ScriptedControllerLandsAdversarialSpawnCorners) {
	oa::LunarLander3dConfig config;
	const double padRange = config.terrain_.padHalfExtent_ * 0.35;
	std::uint32_t caseIndex = 0U;
	for (const double xSign : {-1.0, 1.0}) {
		for (const double zSign : {-1.0, 1.0}) {
			for (const double yawSign : {-1.0, 1.0}) {
				auto environment = oa::LunarScalarEnvironment::createFlat(
					config, lunarTestManifestForConfig(
						config, 0x434f524e45525f54ULL, caseIndex++, 0U));
				ASSERT_TRUE(environment.isValid()) << environment.error();
				oa::LunarLander3dState state = lunarTestFlightState(config, 7.0);
				state.position_.x = xSign * padRange;
				state.position_.z = zSign * padRange;
				state.linearVelocity_ = {xSign * 0.12, -0.3, zSign * 0.12};
				state.orientation_ = (
					oa::vlm::DQuat::fromAxisAngle(
						{0.0, 1.0, 0.0}, yawSign * 0.08)
					* oa::vlm::DQuat::fromAxisAngle(
						{1.0, 0.0, 0.0}, xSign * 0.03)
					* oa::vlm::DQuat::fromAxisAngle(
						{0.0, 0.0, 1.0}, zSign * 0.03)).normalized();
				ASSERT_TRUE(environment.setState(state));

				while (not environment.state().terminated_
					and not environment.state().truncated_) {
					const oa::LunarTransition transition = environment.step(
						static_cast<std::uint32_t>(oa::lunarScriptedLandingAction(
							config, environment.state())));
					ASSERT_TRUE(transition.valid_) << transition.error_;
				}
				EXPECT_EQ(
					environment.state().endReason_, oa::LunarEndReason::SafeLanding)
					<< "x_sign=" << xSign << " z_sign=" << zSign
					<< " yaw_sign=" << yawSign;
			}
		}
	}
}

TEST(LunarLander3d, ScriptedControllerCoversHeldOutFlatSpawns) {
	constexpr std::uint64_t baseSeed = 0x50494c4f545f4556ULL;
	constexpr std::uint32_t episodes = 512U;
	oa::LunarLander3dConfig config;
	std::uint32_t safeLandings = 0U;
	std::array<std::uint32_t, 9U> reasons{};
	std::array<std::uint64_t, 8U> actionCounts{};
	std::ostringstream timeoutDiagnostics;
	for (std::uint32_t lane = 0U; lane < episodes; ++lane) {
		const oa::LunarEpisodeManifest manifest =
			oa::LunarEpisodeManifest::derive(
				baseSeed, lane, 0U, config.contractFingerprint());
		auto environment = oa::LunarScalarEnvironment::createFlat(config, manifest);
		ASSERT_TRUE(environment.isValid()) << "lane=" << lane
			<< " " << environment.error();
		for (std::uint32_t step = 0U;
			step < config.maxEpisodeSteps_
				and not environment.state().terminated_
				and not environment.state().truncated_;
			++step) {
			const oa::LunarAction action = oa::lunarScriptedLandingAction(
				config, environment.state());
			++actionCounts[static_cast<std::size_t>(action)];
			const oa::LunarTransition transition = environment.step(
				static_cast<std::uint32_t>(action));
			ASSERT_TRUE(transition.valid_) << "lane=" << lane
				<< " step=" << step << " " << transition.error_;
		}
		const auto reasonIndex = static_cast<std::size_t>(
			environment.state().endReason_);
		ASSERT_LT(reasonIndex, reasons.size());
		++reasons[reasonIndex];
		if (environment.state().endReason_ == oa::LunarEndReason::SafeLanding) {
			++safeLandings;
		} else if (environment.state().endReason_ == oa::LunarEndReason::TimeLimit) {
			const oa::LunarLander3dState& state = environment.state();
			timeoutDiagnostics << " lane=" << lane
				<< " position=(" << state.position_.x << ','
				<< state.position_.y << ','
				<< state.position_.z << ')'
				<< " linear_speed=" << state.linearVelocity_.length()
				<< " angular_speed=" << state.angularVelocityBody_.length()
				<< " dwell=" << state.stableDwell_
				<< " contacts=" << state.footContacts_[0]
				<< state.footContacts_[1]
				<< state.footContacts_[2]
				<< state.footContacts_[3]
				<< " on_pad=" << state.feetOnPad_[0]
				<< state.feetOnPad_[1]
				<< state.feetOnPad_[2]
				<< state.feetOnPad_[3];
		}
	}
	RecordProperty("episodes", episodes);
	RecordProperty("safe_landings", safeLandings);
	RecordProperty("body_impacts", reasons[static_cast<std::size_t>(
		oa::LunarEndReason::BodyImpact)]);
	RecordProperty("hard_foot_impacts", reasons[static_cast<std::size_t>(
		oa::LunarEndReason::HardFootImpact)]);
	RecordProperty("out_of_bounds", reasons[static_cast<std::size_t>(
		oa::LunarEndReason::OutOfBounds)]);
	RecordProperty("time_limits", reasons[static_cast<std::size_t>(
		oa::LunarEndReason::TimeLimit)]);
	RecordProperty("action_coast", actionCounts[static_cast<std::size_t>(
		oa::LunarAction::Coast)]);
	RecordProperty("action_main", actionCounts[static_cast<std::size_t>(
		oa::LunarAction::MainEngine)]);
	RecordProperty("action_pitch_positive", actionCounts[static_cast<std::size_t>(
		oa::LunarAction::PitchPositive)]);
	RecordProperty("action_pitch_negative", actionCounts[static_cast<std::size_t>(
		oa::LunarAction::PitchNegative)]);
	RecordProperty("action_roll_positive", actionCounts[static_cast<std::size_t>(
		oa::LunarAction::RollPositive)]);
	RecordProperty("action_roll_negative", actionCounts[static_cast<std::size_t>(
		oa::LunarAction::RollNegative)]);
	EXPECT_EQ(safeLandings, episodes)
		<< "body=" << reasons[static_cast<std::size_t>(
			oa::LunarEndReason::BodyImpact)]
		<< " hard-foot=" << reasons[static_cast<std::size_t>(
			oa::LunarEndReason::HardFootImpact)]
		<< " out-of-bounds=" << reasons[static_cast<std::size_t>(
			oa::LunarEndReason::OutOfBounds)]
		<< " timeout=" << reasons[static_cast<std::size_t>(
			oa::LunarEndReason::TimeLimit)]
		<< timeoutDiagnostics.str();
	EXPECT_EQ(reasons[static_cast<std::size_t>(
		oa::LunarEndReason::TimeLimit)], 0U);
	EXPECT_EQ(reasons[static_cast<std::size_t>(
		oa::LunarEndReason::BodyImpact)], 0U);
	EXPECT_EQ(reasons[static_cast<std::size_t>(
		oa::LunarEndReason::HardFootImpact)], 0U);
	EXPECT_EQ(reasons[static_cast<std::size_t>(
		oa::LunarEndReason::OutOfBounds)], 0U);
	EXPECT_EQ(reasons[static_cast<std::size_t>(
		oa::LunarEndReason::NumericalFailure)], 0U);
	EXPECT_EQ(reasons[static_cast<std::size_t>(
		oa::LunarEndReason::ExternalStop)], 0U);
	EXPECT_EQ(reasons[static_cast<std::size_t>(
		oa::LunarEndReason::InvalidAction)], 0U);
}
