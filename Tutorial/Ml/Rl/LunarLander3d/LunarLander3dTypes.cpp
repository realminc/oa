#include "LunarLander3dTypes.h"

#include <cmath>

static std::uint64_t OaLunarMix64(std::uint64_t InValue) noexcept {
	InValue += 0x9e3779b97f4a7c15ULL;
	InValue = (InValue ^ (InValue >> 30U)) * 0xbf58476d1ce4e5b9ULL;
	InValue = (InValue ^ (InValue >> 27U)) * 0x94d049bb133111ebULL;
	return InValue ^ (InValue >> 31U);
}

static std::uint64_t OaLunarCombineSeed(
	std::uint64_t InSeed,
	std::uint64_t InValue) noexcept {
	return OaLunarMix64(InSeed ^ OaLunarMix64(InValue));
}

bool OaLunarVec3::IsFinite() const noexcept {
	return std::isfinite(ComponentX_) and std::isfinite(ComponentY_) and std::isfinite(ComponentZ_);
}

double OaLunarVec3::LengthSquared() const noexcept {
	return OaLunarDot(*this, *this);
}

double OaLunarVec3::Length() const noexcept {
	return std::sqrt(LengthSquared());
}

OaLunarVec3 OaLunarVec3::Normalized() const noexcept {
	const double length = Length();
	if (not std::isfinite(length) or length <= 1.0e-15) {
		return {};
	}
	return *this / length;
}

OaLunarQuat OaLunarQuat::FromAxisAngle(
	const OaLunarVec3& InAxis,
	double InRadians) noexcept {
	const OaLunarVec3 axis = InAxis.Normalized();
	if (axis.LengthSquared() == 0.0 or not std::isfinite(InRadians)) {
		return Identity();
	}
	const double halfAngle = InRadians * 0.5;
	const double sine = std::sin(halfAngle);
	return {
		std::cos(halfAngle),
		axis.ComponentX_ * sine,
		axis.ComponentY_ * sine,
		axis.ComponentZ_ * sine,
	};
}

bool OaLunarQuat::IsFinite() const noexcept {
	return std::isfinite(Scalar_) and std::isfinite(ComponentX_)
		and std::isfinite(ComponentY_) and std::isfinite(ComponentZ_);
}

double OaLunarQuat::NormSquared() const noexcept {
	return Scalar_ * Scalar_ + ComponentX_ * ComponentX_ + ComponentY_ * ComponentY_ + ComponentZ_ * ComponentZ_;
}

double OaLunarQuat::Norm() const noexcept {
	return std::sqrt(NormSquared());
}

OaLunarQuat OaLunarQuat::Normalized() const noexcept {
	const double norm = Norm();
	if (not std::isfinite(norm) or norm <= 1.0e-15) {
		return Identity();
	}
	return {Scalar_ / norm, ComponentX_ / norm, ComponentY_ / norm, ComponentZ_ / norm};
}

OaLunarVec3 OaLunarQuat::Rotate(const OaLunarVec3& InVector) const noexcept {
	const OaLunarQuat normalized = Normalized();
	const OaLunarVec3 imaginary{normalized.ComponentX_, normalized.ComponentY_, normalized.ComponentZ_};
	const OaLunarVec3 twiceCross = 2.0 * OaLunarCross(imaginary, InVector);
	return InVector + normalized.Scalar_ * twiceCross
		+ OaLunarCross(imaginary, twiceCross);
}

OaLunarVec3 OaLunarQuat::InverseRotate(
	const OaLunarVec3& InVector) const noexcept {
	return Conjugate().Normalized().Rotate(InVector);
}

OaLunarEpisodeManifest OaLunarEpisodeManifest::Derive(
	std::uint64_t InBaseSeed,
	std::uint32_t InEnvironmentLane,
	std::uint64_t InEpisodeIndex,
	std::uint64_t InConfigFingerprint) noexcept {
	return DeriveVersioned(
		InBaseSeed, InEnvironmentLane, InEpisodeIndex,
		OA_LUNAR_ENVIRONMENT_VERSION, OA_LUNAR_TERRAIN_VERSION,
		OA_LUNAR_PHYSICS_VERSION, OA_LUNAR_OBSERVATION_VERSION,
		OA_LUNAR_REWARD_VERSION, InConfigFingerprint);
}

OaLunarEpisodeManifest OaLunarEpisodeManifest::DeriveVersioned(
	std::uint64_t InBaseSeed,
	std::uint32_t InEnvironmentLane,
	std::uint64_t InEpisodeIndex,
	std::uint32_t InEnvironmentVersion,
	std::uint32_t InTerrainVersion,
	std::uint32_t InPhysicsVersion,
	std::uint32_t InObservationVersion,
	std::uint32_t InRewardVersion,
	std::uint64_t InConfigFingerprint) noexcept {
	OaLunarEpisodeManifest manifest;
	manifest.EnvironmentVersion_ = InEnvironmentVersion;
	manifest.RandomVersion_ = OA_LUNAR_RANDOM_VERSION;
	manifest.TerrainVersion_ = InTerrainVersion;
	manifest.PhysicsVersion_ = InPhysicsVersion;
	manifest.ObservationVersion_ = InObservationVersion;
	manifest.RewardVersion_ = InRewardVersion;
	manifest.ConfigFingerprint_ = InConfigFingerprint;
	manifest.BaseSeed_ = InBaseSeed;
	manifest.EnvironmentLane_ = InEnvironmentLane;
	manifest.EpisodeIndex_ = InEpisodeIndex;

	std::uint64_t root = 0x4f414c554e415231ULL;
	root = OaLunarCombineSeed(root, manifest.EnvironmentVersion_);
	root = OaLunarCombineSeed(root, manifest.RandomVersion_);
	root = OaLunarCombineSeed(root, manifest.TerrainVersion_);
	root = OaLunarCombineSeed(root, manifest.PhysicsVersion_);
	root = OaLunarCombineSeed(root, manifest.ObservationVersion_);
	root = OaLunarCombineSeed(root, manifest.RewardVersion_);
	root = OaLunarCombineSeed(root, manifest.BaseSeed_);
	root = OaLunarCombineSeed(root, manifest.EnvironmentLane_);
	root = OaLunarCombineSeed(root, manifest.EpisodeIndex_);
	manifest.TerrainSeed_ = OaLunarCombineSeed(
		root, static_cast<std::uint64_t>(OaLunarRandomPurpose::Terrain));
	manifest.SpawnSeed_ = OaLunarCombineSeed(
		root, static_cast<std::uint64_t>(OaLunarRandomPurpose::Spawn));
	manifest.DomainSeed_ = OaLunarCombineSeed(
		root, static_cast<std::uint64_t>(OaLunarRandomPurpose::Domain));
	return manifest;
}

std::string OaLunarEpisodeManifest::ValidationError() const {
	if (EnvironmentVersion_ != OA_LUNAR_ENVIRONMENT_VERSION) {
		return "unsupported lunar environment version";
	}
	if (RandomVersion_ != OA_LUNAR_RANDOM_VERSION) {
		return "unsupported lunar random derivation version";
	}
	if (TerrainVersion_ != OA_LUNAR_TERRAIN_VERSION) {
		return "unsupported lunar terrain version";
	}
	if (PhysicsVersion_ != OA_LUNAR_PHYSICS_VERSION) {
		return "unsupported lunar physics version";
	}
	if (ObservationVersion_ != OA_LUNAR_OBSERVATION_VERSION) {
		return "unsupported lunar observation version";
	}
	if (RewardVersion_ != OA_LUNAR_REWARD_VERSION) {
		return "unsupported lunar reward version";
	}
	const OaLunarEpisodeManifest expected = DeriveVersioned(
		BaseSeed_, EnvironmentLane_, EpisodeIndex_,
		EnvironmentVersion_, TerrainVersion_, PhysicsVersion_,
		ObservationVersion_, RewardVersion_, ConfigFingerprint_);
	if (TerrainSeed_ != expected.TerrainSeed_
		or SpawnSeed_ != expected.SpawnSeed_
		or DomainSeed_ != expected.DomainSeed_) {
		return "lunar manifest derived seeds do not match its versioned inputs";
	}
	return {};
}

std::uint64_t OaLunarEpisodeManifest::SeedFor(
	OaLunarRandomPurpose InPurpose) const noexcept {
	switch (InPurpose) {
		case OaLunarRandomPurpose::Terrain: return TerrainSeed_;
		case OaLunarRandomPurpose::Spawn: return SpawnSeed_;
		case OaLunarRandomPurpose::Domain: return DomainSeed_;
	}
	return 0U;
}

double OaLunarEpisodeManifest::Sample01(
	OaLunarRandomPurpose InPurpose,
	std::uint64_t InCounter) const noexcept {
	const std::uint64_t bits = OaLunarCombineSeed(SeedFor(InPurpose), InCounter);
	const std::uint64_t mantissa = bits >> 11U;
	return static_cast<double>(mantissa) * (1.0 / 9007199254740992.0);
}
