#include "LunarTerrain.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

static constexpr std::uint64_t OA_LUNAR_TERRAIN_HASH_X =
	0x9e3779b185ebca87ULL;
static constexpr std::uint64_t OA_LUNAR_TERRAIN_HASH_Z =
	0xc2b2ae3d27d4eb4fULL;

static std::uint64_t OaLunarTerrainMix(std::uint64_t InValue) noexcept {
	InValue ^= InValue >> 30U;
	InValue *= 0xbf58476d1ce4e5b9ULL;
	InValue ^= InValue >> 27U;
	InValue *= 0x94d049bb133111ebULL;
	return InValue ^ (InValue >> 31U);
}

static double OaLunarTerrainSignedUnit(
	std::uint64_t InSeed,
	std::uint32_t InVertexX,
	std::uint32_t InVertexZ) noexcept {
	std::uint64_t bits = InSeed;
	bits ^= static_cast<std::uint64_t>(InVertexX) * OA_LUNAR_TERRAIN_HASH_X;
	bits ^= static_cast<std::uint64_t>(InVertexZ) * OA_LUNAR_TERRAIN_HASH_Z;
	bits = OaLunarTerrainMix(bits);
	const double unit = static_cast<double>(bits >> 11U)
		* (1.0 / 9007199254740992.0);
	return unit * 2.0 - 1.0;
}

static double OaLunarSmoothStep(double InValue) noexcept {
	const double clamped = std::clamp(InValue, 0.0, 1.0);
	return clamped * clamped * (3.0 - 2.0 * clamped);
}

static std::size_t OaLunarTerrainVertexCount(
	const OaLunarTerrainConfig& InConfig) noexcept {
	return static_cast<std::size_t>(InConfig.CellsX_ + 1U)
		* static_cast<std::size_t>(InConfig.CellsZ_ + 1U);
}

static std::size_t OaLunarTerrainIndex(
	const OaLunarTerrainConfig& InConfig,
	std::uint32_t InVertexX,
	std::uint32_t InVertexZ) noexcept {
	return static_cast<std::size_t>(InVertexZ)
		* static_cast<std::size_t>(InConfig.CellsX_ + 1U)
		+ static_cast<std::size_t>(InVertexX);
}

static double OaLunarTriangleSlope(
	double InHeight00,
	double InHeight10,
	double InHeight01,
	double InHeight11,
	double InCellSize,
	OaLunarTerrainTriangle InTriangle) noexcept {
	double heightDx = 0.0;
	double heightDz = 0.0;
	if (InTriangle == OaLunarTerrainTriangle::LowerRight) {
		heightDx = (InHeight10 - InHeight00) / InCellSize;
		heightDz = (InHeight11 - InHeight10) / InCellSize;
	} else {
		heightDx = (InHeight11 - InHeight01) / InCellSize;
		heightDz = (InHeight01 - InHeight00) / InCellSize;
	}
	return std::hypot(heightDx, heightDz);
}

std::string OaLunarTerrainConfig::ValidationError() const {
	if (CellsX_ == 0U or CellsZ_ == 0U) {
		return "lunar terrain requires at least one cell on each axis";
	}
	if (CellsX_ > 1048575U or CellsZ_ > 1048575U) {
		return "lunar terrain axis exceeds the host-oracle vertex bound";
	}
	const std::uint64_t verticesX = static_cast<std::uint64_t>(CellsX_) + 1U;
	const std::uint64_t verticesZ = static_cast<std::uint64_t>(CellsZ_) + 1U;
	if (verticesX * verticesZ > 1048576ULL) {
		return "lunar terrain exceeds the one-million-vertex host-oracle bound";
	}
	if (not std::isfinite(CellSize_) or CellSize_ <= 0.0) {
		return "lunar terrain cell size must be finite and positive";
	}
	if (not std::isfinite(MaxAbsHeight_) or MaxAbsHeight_ < 0.0) {
		return "lunar terrain height bound must be finite and non-negative";
	}
	if (not std::isfinite(MaxSlope_) or MaxSlope_ < 0.0) {
		return "lunar terrain slope bound must be finite and non-negative";
	}
	if (not std::isfinite(PadHalfExtent_) or PadHalfExtent_ < 0.0
		or not std::isfinite(PadTransitionWidth_)
		or PadTransitionWidth_ < 0.0) {
		return "lunar terrain pad dimensions must be finite and non-negative";
	}
	const double halfWidthX = static_cast<double>(CellsX_) * CellSize_ * 0.5;
	const double halfWidthZ = static_cast<double>(CellsZ_) * CellSize_ * 0.5;
	const double requiredHalfExtent = PadHalfExtent_ + CellSize_
		+ PadTransitionWidth_;
	if (requiredHalfExtent > std::min(halfWidthX, halfWidthZ)) {
		return "lunar terrain pad and guarded transition do not fit the tile";
	}
	return {};
}

OaLunarTerrain OaLunarTerrain::Invalid_(
	const OaLunarTerrainConfig& InConfig,
	std::string InError) {
	OaLunarTerrain terrain;
	terrain.Config_ = InConfig;
	terrain.Heights_.clear();
	terrain.Error_ = std::move(InError);
	return terrain;
}

OaLunarTerrain OaLunarTerrain::CreateFlat(
	const OaLunarTerrainConfig& InConfig) {
	const std::string configError = InConfig.ValidationError();
	if (not configError.empty()) {
		return Invalid_(InConfig, configError);
	}
	return CreateFromHeights(
		InConfig, std::vector<double>(OaLunarTerrainVertexCount(InConfig), 0.0));
}

OaLunarTerrain OaLunarTerrain::CreateSeeded(
	const OaLunarTerrainConfig& InConfig,
	const OaLunarEpisodeManifest& InManifest) {
	const std::string configError = InConfig.ValidationError();
	if (not configError.empty()) {
		return Invalid_(InConfig, configError);
	}
	const std::string manifestError = InManifest.ValidationError();
	if (not manifestError.empty()) {
		return Invalid_(InConfig, manifestError);
	}

	const std::size_t vertexCount = OaLunarTerrainVertexCount(InConfig);
	std::vector<double> heights(vertexCount, 0.0);
	for (std::uint32_t vertexZ = 0U; vertexZ <= InConfig.CellsZ_; ++vertexZ) {
		for (std::uint32_t vertexX = 0U; vertexX <= InConfig.CellsX_; ++vertexX) {
			heights[OaLunarTerrainIndex(InConfig, vertexX, vertexZ)] =
				OaLunarTerrainSignedUnit(
					InManifest.TerrainSeed_, vertexX, vertexZ);
		}
	}

	std::vector<double> smoothed(vertexCount, 0.0);
	for (std::uint32_t pass = 0U; pass < 4U; ++pass) {
		for (std::uint32_t vertexZ = 0U; vertexZ <= InConfig.CellsZ_; ++vertexZ) {
			for (std::uint32_t vertexX = 0U; vertexX <= InConfig.CellsX_; ++vertexX) {
				double sum = 4.0 * heights[
					OaLunarTerrainIndex(InConfig, vertexX, vertexZ)];
				double weight = 4.0;
				if (vertexX > 0U) {
					sum += heights[OaLunarTerrainIndex(
						InConfig, vertexX - 1U, vertexZ)];
					weight += 1.0;
				}
				if (vertexX < InConfig.CellsX_) {
					sum += heights[OaLunarTerrainIndex(
						InConfig, vertexX + 1U, vertexZ)];
					weight += 1.0;
				}
				if (vertexZ > 0U) {
					sum += heights[OaLunarTerrainIndex(
						InConfig, vertexX, vertexZ - 1U)];
					weight += 1.0;
				}
				if (vertexZ < InConfig.CellsZ_) {
					sum += heights[OaLunarTerrainIndex(
						InConfig, vertexX, vertexZ + 1U)];
					weight += 1.0;
				}
				smoothed[OaLunarTerrainIndex(InConfig, vertexX, vertexZ)] =
					sum / weight;
			}
		}
		heights.swap(smoothed);
	}

	const double minX = -static_cast<double>(InConfig.CellsX_)
		* InConfig.CellSize_ * 0.5;
	const double minZ = -static_cast<double>(InConfig.CellsZ_)
		* InConfig.CellSize_ * 0.5;
	const double guardedPad = InConfig.PadHalfExtent_ + InConfig.CellSize_;
	// One zero-height cell guards the declared pad. It keeps both triangles and
	// their normals flat at the inclusive pad boundary before the smooth ring.
	for (std::uint32_t vertexZ = 0U; vertexZ <= InConfig.CellsZ_; ++vertexZ) {
		const double positionZ = minZ
			+ static_cast<double>(vertexZ) * InConfig.CellSize_;
		for (std::uint32_t vertexX = 0U; vertexX <= InConfig.CellsX_; ++vertexX) {
			const double positionX = minX
				+ static_cast<double>(vertexX) * InConfig.CellSize_;
			const double squareRadius = std::max(
				std::abs(positionX), std::abs(positionZ));
			double reliefWeight = 0.0;
			if (squareRadius > guardedPad) {
				if (InConfig.PadTransitionWidth_ == 0.0) {
					reliefWeight = 1.0;
				} else {
					reliefWeight = OaLunarSmoothStep(
						(squareRadius - guardedPad)
						/ InConfig.PadTransitionWidth_);
				}
			}
			heights[OaLunarTerrainIndex(InConfig, vertexX, vertexZ)] *=
				reliefWeight;
		}
	}

	double maximumHeight = 0.0;
	double maximumSlope = 0.0;
	for (const double height : heights) {
		maximumHeight = std::max(maximumHeight, std::abs(height));
	}
	for (std::uint32_t cellZ = 0U; cellZ < InConfig.CellsZ_; ++cellZ) {
		for (std::uint32_t cellX = 0U; cellX < InConfig.CellsX_; ++cellX) {
			const double height00 = heights[OaLunarTerrainIndex(
				InConfig, cellX, cellZ)];
			const double height10 = heights[OaLunarTerrainIndex(
				InConfig, cellX + 1U, cellZ)];
			const double height01 = heights[OaLunarTerrainIndex(
				InConfig, cellX, cellZ + 1U)];
			const double height11 = heights[OaLunarTerrainIndex(
				InConfig, cellX + 1U, cellZ + 1U)];
			maximumSlope = std::max(maximumSlope, OaLunarTriangleSlope(
				height00, height10, height01, height11,
				InConfig.CellSize_, OaLunarTerrainTriangle::LowerRight));
			maximumSlope = std::max(maximumSlope, OaLunarTriangleSlope(
				height00, height10, height01, height11,
				InConfig.CellSize_, OaLunarTerrainTriangle::UpperLeft));
		}
	}
	double scale = 1.0;
	if (maximumHeight > 0.0) {
		scale = std::min(scale, InConfig.MaxAbsHeight_ / maximumHeight);
	}
	if (maximumSlope > 0.0) {
		scale = std::min(scale, InConfig.MaxSlope_ / maximumSlope);
	}
	for (double& height : heights) {
		height *= scale;
	}
	return CreateFromHeights(InConfig, heights);
}

OaLunarTerrain OaLunarTerrain::CreateFromHeights(
	const OaLunarTerrainConfig& InConfig,
	const std::vector<double>& InHeights) {
	const std::string configError = InConfig.ValidationError();
	if (not configError.empty()) {
		return Invalid_(InConfig, configError);
	}
	if (InHeights.size() != OaLunarTerrainVertexCount(InConfig)) {
		return Invalid_(InConfig, "lunar terrain height count does not match its grid");
	}
	for (const double height : InHeights) {
		if (not std::isfinite(height)) {
			return Invalid_(InConfig, "lunar terrain contains a non-finite height");
		}
		if (std::abs(height) > InConfig.MaxAbsHeight_ + 1.0e-12) {
			return Invalid_(InConfig, "lunar terrain exceeds its height bound");
		}
	}
	for (std::uint32_t cellZ = 0U; cellZ < InConfig.CellsZ_; ++cellZ) {
		for (std::uint32_t cellX = 0U; cellX < InConfig.CellsX_; ++cellX) {
			const double height00 = InHeights[OaLunarTerrainIndex(
				InConfig, cellX, cellZ)];
			const double height10 = InHeights[OaLunarTerrainIndex(
				InConfig, cellX + 1U, cellZ)];
			const double height01 = InHeights[OaLunarTerrainIndex(
				InConfig, cellX, cellZ + 1U)];
			const double height11 = InHeights[OaLunarTerrainIndex(
				InConfig, cellX + 1U, cellZ + 1U)];
			const double lowerSlope = OaLunarTriangleSlope(
				height00, height10, height01, height11,
				InConfig.CellSize_, OaLunarTerrainTriangle::LowerRight);
			const double upperSlope = OaLunarTriangleSlope(
				height00, height10, height01, height11,
				InConfig.CellSize_, OaLunarTerrainTriangle::UpperLeft);
			if (lowerSlope > InConfig.MaxSlope_ + 1.0e-12
				or upperSlope > InConfig.MaxSlope_ + 1.0e-12) {
				return Invalid_(InConfig, "lunar terrain exceeds its triangle-slope bound");
			}
		}
	}

	OaLunarTerrain terrain;
	terrain.Config_ = InConfig;
	terrain.Heights_ = InHeights;
	terrain.Error_.clear();
	return terrain;
}

double OaLunarTerrain::MinX() const noexcept {
	return -static_cast<double>(Config_.CellsX_) * Config_.CellSize_ * 0.5;
}

double OaLunarTerrain::MaxX() const noexcept {
	return -MinX();
}

double OaLunarTerrain::MinZ() const noexcept {
	return -static_cast<double>(Config_.CellsZ_) * Config_.CellSize_ * 0.5;
}

double OaLunarTerrain::MaxZ() const noexcept {
	return -MinZ();
}

bool OaLunarTerrain::Contains(double InX, double InZ) const noexcept {
	return IsValid() and std::isfinite(InX) and std::isfinite(InZ)
		and InX >= MinX() and InX <= MaxX()
		and InZ >= MinZ() and InZ <= MaxZ();
}

bool OaLunarTerrain::IsOnPad(double InX, double InZ) const noexcept {
	return Contains(InX, InZ)
		and std::max(std::abs(InX), std::abs(InZ))
			<= Config_.PadHalfExtent_;
}

std::size_t OaLunarTerrain::VertexIndex_(
	std::uint32_t InVertexX,
	std::uint32_t InVertexZ) const noexcept {
	return OaLunarTerrainIndex(Config_, InVertexX, InVertexZ);
}

double OaLunarTerrain::VertexHeight(
	std::uint32_t InVertexX,
	std::uint32_t InVertexZ) const noexcept {
	if (not IsValid() or InVertexX > Config_.CellsX_
		or InVertexZ > Config_.CellsZ_) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	return Heights_[VertexIndex_(InVertexX, InVertexZ)];
}

OaLunarTerrainSample OaLunarTerrain::Query(
	double InX,
	double InZ) const noexcept {
	OaLunarTerrainSample sample;
	if (not Contains(InX, InZ)) {
		return sample;
	}

	const double gridX = (InX - MinX()) / Config_.CellSize_;
	const double gridZ = (InZ - MinZ()) / Config_.CellSize_;
	if (gridX == static_cast<double>(Config_.CellsX_)) {
		sample.CellX_ = Config_.CellsX_ - 1U;
		sample.LocalX_ = 1.0;
	} else {
		sample.CellX_ = static_cast<std::uint32_t>(std::floor(gridX));
		sample.LocalX_ = gridX - static_cast<double>(sample.CellX_);
	}
	if (gridZ == static_cast<double>(Config_.CellsZ_)) {
		sample.CellZ_ = Config_.CellsZ_ - 1U;
		sample.LocalZ_ = 1.0;
	} else {
		sample.CellZ_ = static_cast<std::uint32_t>(std::floor(gridZ));
		sample.LocalZ_ = gridZ - static_cast<double>(sample.CellZ_);
	}

	const double height00 = VertexHeight(sample.CellX_, sample.CellZ_);
	const double height10 = VertexHeight(sample.CellX_ + 1U, sample.CellZ_);
	const double height01 = VertexHeight(sample.CellX_, sample.CellZ_ + 1U);
	const double height11 = VertexHeight(
		sample.CellX_ + 1U, sample.CellZ_ + 1U);
	double heightDx = 0.0;
	double heightDz = 0.0;
	if (sample.LocalZ_ <= sample.LocalX_) {
		sample.Triangle_ = OaLunarTerrainTriangle::LowerRight;
		sample.Height_ = height00
			+ sample.LocalX_ * (height10 - height00)
			+ sample.LocalZ_ * (height11 - height10);
		heightDx = (height10 - height00) / Config_.CellSize_;
		heightDz = (height11 - height10) / Config_.CellSize_;
	} else {
		sample.Triangle_ = OaLunarTerrainTriangle::UpperLeft;
		sample.Height_ = height00
			+ sample.LocalX_ * (height11 - height01)
			+ sample.LocalZ_ * (height01 - height00);
		heightDx = (height11 - height01) / Config_.CellSize_;
		heightDz = (height01 - height00) / Config_.CellSize_;
	}
	sample.Normal_ = OaLunarVec3(-heightDx, 1.0, -heightDz).Normalized();
	sample.InBounds_ = true;
	return sample;
}
