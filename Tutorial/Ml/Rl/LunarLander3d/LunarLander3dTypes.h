#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

inline constexpr std::uint32_t OA_LUNAR_ENVIRONMENT_VERSION = 1U;
inline constexpr std::uint32_t OA_LUNAR_RANDOM_VERSION = 1U;
inline constexpr std::uint32_t OA_LUNAR_TERRAIN_VERSION = 1U;
inline constexpr std::uint32_t OA_LUNAR_PHYSICS_VERSION = 1U;
inline constexpr std::uint32_t OA_LUNAR_OBSERVATION_VERSION = 1U;
inline constexpr std::uint32_t OA_LUNAR_REWARD_VERSION = 1U;
inline constexpr std::size_t OA_LUNAR_OBSERVATION_SIZE = 33U;

class OaLunarVec3 {
public:
	double ComponentX_ = 0.0;
	double ComponentY_ = 0.0;
	double ComponentZ_ = 0.0;

	constexpr OaLunarVec3() noexcept = default;
	constexpr OaLunarVec3(double InX, double InY, double InZ) noexcept
		: ComponentX_(InX), ComponentY_(InY), ComponentZ_(InZ) {}

	[[nodiscard]] bool IsFinite() const noexcept;
	[[nodiscard]] double LengthSquared() const noexcept;
	[[nodiscard]] double Length() const noexcept;
	[[nodiscard]] OaLunarVec3 Normalized() const noexcept;

	[[nodiscard]] constexpr OaLunarVec3 operator-() const noexcept {
		return {-ComponentX_, -ComponentY_, -ComponentZ_};
	}
	[[nodiscard]] constexpr OaLunarVec3 operator+(
		const OaLunarVec3& InOther) const noexcept {
		return {ComponentX_ + InOther.ComponentX_, ComponentY_ + InOther.ComponentY_, ComponentZ_ + InOther.ComponentZ_};
	}
	[[nodiscard]] constexpr OaLunarVec3 operator-(
		const OaLunarVec3& InOther) const noexcept {
		return {ComponentX_ - InOther.ComponentX_, ComponentY_ - InOther.ComponentY_, ComponentZ_ - InOther.ComponentZ_};
	}
	[[nodiscard]] constexpr OaLunarVec3 operator*(double InScale) const noexcept {
		return {ComponentX_ * InScale, ComponentY_ * InScale, ComponentZ_ * InScale};
	}
	[[nodiscard]] constexpr OaLunarVec3 operator/(double InScale) const noexcept {
		return {ComponentX_ / InScale, ComponentY_ / InScale, ComponentZ_ / InScale};
	}
	constexpr OaLunarVec3& operator+=(const OaLunarVec3& InOther) noexcept {
		ComponentX_ += InOther.ComponentX_;
		ComponentY_ += InOther.ComponentY_;
		ComponentZ_ += InOther.ComponentZ_;
		return *this;
	}
	constexpr OaLunarVec3& operator-=(const OaLunarVec3& InOther) noexcept {
		ComponentX_ -= InOther.ComponentX_;
		ComponentY_ -= InOther.ComponentY_;
		ComponentZ_ -= InOther.ComponentZ_;
		return *this;
	}
	constexpr OaLunarVec3& operator*=(double InScale) noexcept {
		ComponentX_ *= InScale;
		ComponentY_ *= InScale;
		ComponentZ_ *= InScale;
		return *this;
	}

	[[nodiscard]] constexpr bool operator==(
		const OaLunarVec3& InOther) const noexcept = default;
};

[[nodiscard]] constexpr OaLunarVec3 operator*(
	double InScale,
	const OaLunarVec3& InVector) noexcept {
	return InVector * InScale;
}

[[nodiscard]] constexpr double OaLunarDot(
	const OaLunarVec3& InLeft,
	const OaLunarVec3& InRight) noexcept {
	return InLeft.ComponentX_ * InRight.ComponentX_
		+ InLeft.ComponentY_ * InRight.ComponentY_
		+ InLeft.ComponentZ_ * InRight.ComponentZ_;
}

[[nodiscard]] constexpr OaLunarVec3 OaLunarCross(
	const OaLunarVec3& InLeft,
	const OaLunarVec3& InRight) noexcept {
	return {
		InLeft.ComponentY_ * InRight.ComponentZ_ - InLeft.ComponentZ_ * InRight.ComponentY_,
		InLeft.ComponentZ_ * InRight.ComponentX_ - InLeft.ComponentX_ * InRight.ComponentZ_,
		InLeft.ComponentX_ * InRight.ComponentY_ - InLeft.ComponentY_ * InRight.ComponentX_,
	};
}

class OaLunarQuat {
public:
	double Scalar_ = 1.0;
	double ComponentX_ = 0.0;
	double ComponentY_ = 0.0;
	double ComponentZ_ = 0.0;

	constexpr OaLunarQuat() noexcept = default;
	constexpr OaLunarQuat(
		double InW,
		double InX,
		double InY,
		double InZ) noexcept
		: Scalar_(InW), ComponentX_(InX), ComponentY_(InY), ComponentZ_(InZ) {}

	[[nodiscard]] static constexpr OaLunarQuat Identity() noexcept {
		return {};
	}
	[[nodiscard]] static OaLunarQuat FromAxisAngle(
		const OaLunarVec3& InAxis,
		double InRadians) noexcept;

	[[nodiscard]] bool IsFinite() const noexcept;
	[[nodiscard]] double NormSquared() const noexcept;
	[[nodiscard]] double Norm() const noexcept;
	[[nodiscard]] OaLunarQuat Normalized() const noexcept;
	[[nodiscard]] constexpr OaLunarQuat Conjugate() const noexcept {
		return {Scalar_, -ComponentX_, -ComponentY_, -ComponentZ_};
	}
	[[nodiscard]] OaLunarVec3 Rotate(const OaLunarVec3& InVector) const noexcept;
	[[nodiscard]] OaLunarVec3 InverseRotate(
		const OaLunarVec3& InVector) const noexcept;

	[[nodiscard]] constexpr OaLunarQuat operator+(
		const OaLunarQuat& InOther) const noexcept {
		return {
			Scalar_ + InOther.Scalar_, ComponentX_ + InOther.ComponentX_,
			ComponentY_ + InOther.ComponentY_, ComponentZ_ + InOther.ComponentZ_,
		};
	}
	[[nodiscard]] constexpr OaLunarQuat operator*(double InScale) const noexcept {
		return {Scalar_ * InScale, ComponentX_ * InScale, ComponentY_ * InScale, ComponentZ_ * InScale};
	}
	[[nodiscard]] constexpr OaLunarQuat operator*(
		const OaLunarQuat& InOther) const noexcept {
		return {
			Scalar_ * InOther.Scalar_ - ComponentX_ * InOther.ComponentX_
				- ComponentY_ * InOther.ComponentY_ - ComponentZ_ * InOther.ComponentZ_,
			Scalar_ * InOther.ComponentX_ + ComponentX_ * InOther.Scalar_
				+ ComponentY_ * InOther.ComponentZ_ - ComponentZ_ * InOther.ComponentY_,
			Scalar_ * InOther.ComponentY_ - ComponentX_ * InOther.ComponentZ_
				+ ComponentY_ * InOther.Scalar_ + ComponentZ_ * InOther.ComponentX_,
			Scalar_ * InOther.ComponentZ_ + ComponentX_ * InOther.ComponentY_
				- ComponentY_ * InOther.ComponentX_ + ComponentZ_ * InOther.Scalar_,
		};
	}

	[[nodiscard]] constexpr bool operator==(
		const OaLunarQuat& InOther) const noexcept = default;
};

enum class OaLunarRandomPurpose : std::uint64_t {
	Terrain = 0x5445525241494e31ULL,
	Spawn = 0x535041574e303031ULL,
	Domain = 0x444f4d41494e3031ULL,
};

class OaLunarEpisodeManifest {
public:
	std::uint32_t EnvironmentVersion_ = OA_LUNAR_ENVIRONMENT_VERSION;
	std::uint32_t RandomVersion_ = OA_LUNAR_RANDOM_VERSION;
	std::uint32_t TerrainVersion_ = OA_LUNAR_TERRAIN_VERSION;
	std::uint32_t PhysicsVersion_ = OA_LUNAR_PHYSICS_VERSION;
	std::uint32_t ObservationVersion_ = OA_LUNAR_OBSERVATION_VERSION;
	std::uint32_t RewardVersion_ = OA_LUNAR_REWARD_VERSION;
	std::uint64_t ConfigFingerprint_ = 0U;
	std::uint64_t BaseSeed_ = 0U;
	std::uint32_t EnvironmentLane_ = 0U;
	std::uint64_t EpisodeIndex_ = 0U;
	std::uint64_t TerrainSeed_ = 0U;
	std::uint64_t SpawnSeed_ = 0U;
	std::uint64_t DomainSeed_ = 0U;

	[[nodiscard]] static OaLunarEpisodeManifest Derive(
		std::uint64_t InBaseSeed,
		std::uint32_t InEnvironmentLane,
		std::uint64_t InEpisodeIndex,
		std::uint64_t InConfigFingerprint) noexcept;
	[[nodiscard]] static OaLunarEpisodeManifest DeriveVersioned(
		std::uint64_t InBaseSeed,
		std::uint32_t InEnvironmentLane,
		std::uint64_t InEpisodeIndex,
		std::uint32_t InEnvironmentVersion,
		std::uint32_t InTerrainVersion,
		std::uint32_t InPhysicsVersion,
		std::uint32_t InObservationVersion,
		std::uint32_t InRewardVersion,
		std::uint64_t InConfigFingerprint) noexcept;

	[[nodiscard]] std::string ValidationError() const;
	[[nodiscard]] std::uint64_t SeedFor(
		OaLunarRandomPurpose InPurpose) const noexcept;
	[[nodiscard]] double Sample01(
		OaLunarRandomPurpose InPurpose,
		std::uint64_t InCounter) const noexcept;

	[[nodiscard]] constexpr bool operator==(
		const OaLunarEpisodeManifest& InOther) const noexcept = default;
};

enum class OaLunarAction : std::uint8_t {
	Coast = 0U,
	MainEngine = 1U,
	PitchPositive = 2U,
	PitchNegative = 3U,
	RollPositive = 4U,
	RollNegative = 5U,
	YawPositive = 6U,
	YawNegative = 7U,
};

[[nodiscard]] constexpr bool OaLunarActionIsValid(
	std::uint32_t InAction) noexcept {
	return InAction <= static_cast<std::uint32_t>(OaLunarAction::YawNegative);
}

enum class OaLunarEndReason : std::uint8_t {
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
