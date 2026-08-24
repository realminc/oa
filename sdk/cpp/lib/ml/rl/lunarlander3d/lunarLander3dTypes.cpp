#include <ml/rl/lunarlander3d/lunarLander3dTypes.h>

#include <cmath>

namespace oa {

static std::uint64_t lunarMix64(std::uint64_t inValue) noexcept {
	inValue += 0x9e3779b97f4a7c15ULL;
	inValue = (inValue ^ (inValue >> 30U)) * 0xbf58476d1ce4e5b9ULL;
	inValue = (inValue ^ (inValue >> 27U)) * 0x94d049bb133111ebULL;
	return inValue ^ (inValue >> 31U);
}

static std::uint64_t lunarCombineSeed(
	std::uint64_t inSeed,
	std::uint64_t inValue) noexcept {
	return lunarMix64(inSeed ^ lunarMix64(inValue));
}

LunarEpisodeManifest LunarEpisodeManifest::derive(
	std::uint64_t inBaseSeed,
	std::uint32_t inEnvironmentLane,
	std::uint64_t inEpisodeIndex,
	std::uint64_t inConfigFingerprint) noexcept {
	return deriveVersioned(
		inBaseSeed, inEnvironmentLane, inEpisodeIndex,
		kLunarEnvironmentVersion, kLunarTerrainVersion,
		kLunarPhysicsVersion, kLunarObservationVersion,
		kLunarRewardVersion, inConfigFingerprint);
}

LunarEpisodeManifest LunarEpisodeManifest::deriveVersioned(
	std::uint64_t inBaseSeed,
	std::uint32_t inEnvironmentLane,
	std::uint64_t inEpisodeIndex,
	std::uint32_t inEnvironmentVersion,
	std::uint32_t inTerrainVersion,
	std::uint32_t inPhysicsVersion,
	std::uint32_t inObservationVersion,
	std::uint32_t inRewardVersion,
	std::uint64_t inConfigFingerprint) noexcept {
	LunarEpisodeManifest manifest;
	manifest.environmentVersion_ = inEnvironmentVersion;
	manifest.randomVersion_ = kLunarRandomVersion;
	manifest.terrainVersion_ = inTerrainVersion;
	manifest.physicsVersion_ = inPhysicsVersion;
	manifest.observationVersion_ = inObservationVersion;
	manifest.rewardVersion_ = inRewardVersion;
	manifest.configFingerprint_ = inConfigFingerprint;
	manifest.baseSeed_ = inBaseSeed;
	manifest.environmentLane_ = inEnvironmentLane;
	manifest.episodeIndex_ = inEpisodeIndex;

	std::uint64_t root = 0x4f414c554e415231ULL;
	root = lunarCombineSeed(root, manifest.environmentVersion_);
	root = lunarCombineSeed(root, manifest.randomVersion_);
	root = lunarCombineSeed(root, manifest.terrainVersion_);
	root = lunarCombineSeed(root, manifest.physicsVersion_);
	root = lunarCombineSeed(root, manifest.observationVersion_);
	root = lunarCombineSeed(root, manifest.rewardVersion_);
	root = lunarCombineSeed(root, manifest.baseSeed_);
	root = lunarCombineSeed(root, manifest.environmentLane_);
	root = lunarCombineSeed(root, manifest.episodeIndex_);
	manifest.terrainSeed_ = lunarCombineSeed(
		root, static_cast<std::uint64_t>(LunarRandomPurpose::Terrain));
	manifest.spawnSeed_ = lunarCombineSeed(
		root, static_cast<std::uint64_t>(LunarRandomPurpose::Spawn));
	manifest.domainSeed_ = lunarCombineSeed(
		root, static_cast<std::uint64_t>(LunarRandomPurpose::Domain));
	return manifest;
}

std::string LunarEpisodeManifest::validationError() const {
	if (environmentVersion_ != kLunarEnvironmentVersion) {
		return "unsupported lunar environment version";
	}
	if (randomVersion_ != kLunarRandomVersion) {
		return "unsupported lunar random derivation version";
	}
	if (terrainVersion_ != kLunarTerrainVersion) {
		return "unsupported lunar terrain version";
	}
	if (physicsVersion_ != kLunarPhysicsVersion) {
		return "unsupported lunar physics version";
	}
	if (observationVersion_ != kLunarObservationVersion) {
		return "unsupported lunar observation version";
	}
	if (rewardVersion_ != kLunarRewardVersion) {
		return "unsupported lunar reward version";
	}
	const LunarEpisodeManifest expected = deriveVersioned(
		baseSeed_, environmentLane_, episodeIndex_,
		environmentVersion_, terrainVersion_, physicsVersion_,
		observationVersion_, rewardVersion_, configFingerprint_);
	if (terrainSeed_ != expected.terrainSeed_
		or spawnSeed_ != expected.spawnSeed_
		or domainSeed_ != expected.domainSeed_) {
		return "lunar manifest derived seeds do not match its versioned inputs";
	}
	return {};
}

std::uint64_t LunarEpisodeManifest::seedFor(
	LunarRandomPurpose inPurpose) const noexcept {
	switch (inPurpose) {
		case LunarRandomPurpose::Terrain: return terrainSeed_;
		case LunarRandomPurpose::Spawn: return spawnSeed_;
		case LunarRandomPurpose::Domain: return domainSeed_;
	}
	return 0U;
}

double LunarEpisodeManifest::sample01(
	LunarRandomPurpose inPurpose,
	std::uint64_t inCounter) const noexcept {
	const std::uint64_t bits = lunarCombineSeed(seedFor(inPurpose), inCounter);
	const std::uint64_t mantissa = bits >> 11U;
	return static_cast<double>(mantissa) * (1.0 / 9007199254740992.0);
}

} // namespace oa
