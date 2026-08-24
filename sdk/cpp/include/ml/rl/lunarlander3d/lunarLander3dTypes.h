#pragma once

#include <oa/core/vlm.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace oa {

inline constexpr std::uint32_t kLunarEnvironmentVersion = 1U;
inline constexpr std::uint32_t kLunarRandomVersion = 1U;
inline constexpr std::uint32_t kLunarTerrainVersion = 1U;
inline constexpr std::uint32_t kLunarPhysicsVersion = 1U;
inline constexpr std::uint32_t kLunarObservationVersion = 1U;
inline constexpr std::uint32_t kLunarRewardVersion = 1U;
inline constexpr std::size_t kLunarObservationSize = 33U;

enum class LunarRandomPurpose : std::uint64_t {
	Terrain = 0x5445525241494e31ULL,
	Spawn = 0x535041574e303031ULL,
	Domain = 0x444f4d41494e3031ULL,
};

class LunarEpisodeManifest {
public:
	std::uint32_t environmentVersion_ = kLunarEnvironmentVersion;
	std::uint32_t randomVersion_ = kLunarRandomVersion;
	std::uint32_t terrainVersion_ = kLunarTerrainVersion;
	std::uint32_t physicsVersion_ = kLunarPhysicsVersion;
	std::uint32_t observationVersion_ = kLunarObservationVersion;
	std::uint32_t rewardVersion_ = kLunarRewardVersion;
	std::uint64_t configFingerprint_ = 0U;
	std::uint64_t baseSeed_ = 0U;
	std::uint32_t environmentLane_ = 0U;
	std::uint64_t episodeIndex_ = 0U;
	std::uint64_t terrainSeed_ = 0U;
	std::uint64_t spawnSeed_ = 0U;
	std::uint64_t domainSeed_ = 0U;

	[[nodiscard]] static LunarEpisodeManifest derive(
		std::uint64_t inBaseSeed,
		std::uint32_t inEnvironmentLane,
		std::uint64_t inEpisodeIndex,
		std::uint64_t inConfigFingerprint) noexcept;
	[[nodiscard]] static LunarEpisodeManifest deriveVersioned(
		std::uint64_t inBaseSeed,
		std::uint32_t inEnvironmentLane,
		std::uint64_t inEpisodeIndex,
		std::uint32_t inEnvironmentVersion,
		std::uint32_t inTerrainVersion,
		std::uint32_t inPhysicsVersion,
		std::uint32_t inObservationVersion,
		std::uint32_t inRewardVersion,
		std::uint64_t inConfigFingerprint) noexcept;

	[[nodiscard]] std::string validationError() const;
	[[nodiscard]] std::uint64_t seedFor(
		LunarRandomPurpose inPurpose) const noexcept;
	[[nodiscard]] double sample01(
		LunarRandomPurpose inPurpose,
		std::uint64_t inCounter) const noexcept;

	[[nodiscard]] constexpr bool operator==(
		const LunarEpisodeManifest& inOther) const noexcept = default;
};

enum class LunarAction : std::uint8_t {
	Coast = 0U,
	MainEngine = 1U,
	PitchPositive = 2U,
	PitchNegative = 3U,
	RollPositive = 4U,
	RollNegative = 5U,
	YawPositive = 6U,
	YawNegative = 7U,
};

[[nodiscard]] constexpr bool lunarActionIsValid(
	std::uint32_t inAction) noexcept {
	return inAction <= static_cast<std::uint32_t>(LunarAction::YawNegative);
}

enum class LunarEndReason : std::uint8_t {
	None = 0U,
	SafeLanding,
	BodyImpact,
	HardFootImpact,
	OutOfBounds,
	NumericalFailure,
	TimeLimit,
	ExternalStop,
	InvalidAction,
};

} // namespace oa
