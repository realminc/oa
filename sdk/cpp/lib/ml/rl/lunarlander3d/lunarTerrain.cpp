#include <ml/rl/lunarlander3d/lunarTerrain.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace oa {

static constexpr std::uint64_t kLunarTerrainHashX =
	0x9e3779b185ebca87ULL;
static constexpr std::uint64_t kLunarTerrainHashZ =
	0xc2b2ae3d27d4eb4fULL;

static std::uint64_t lunarTerrainMix(std::uint64_t inValue) noexcept {
	inValue ^= inValue >> 30U;
	inValue *= 0xbf58476d1ce4e5b9ULL;
	inValue ^= inValue >> 27U;
	inValue *= 0x94d049bb133111ebULL;
	return inValue ^ (inValue >> 31U);
}

static double lunarTerrainSignedUnit(
	std::uint64_t inSeed,
	std::uint32_t inVertexX,
	std::uint32_t inVertexZ) noexcept {
	std::uint64_t bits = inSeed;
	bits ^= static_cast<std::uint64_t>(inVertexX) * kLunarTerrainHashX;
	bits ^= static_cast<std::uint64_t>(inVertexZ) * kLunarTerrainHashZ;
	bits = lunarTerrainMix(bits);
	const double unit = static_cast<double>(bits >> 11U)
		* (1.0 / 9007199254740992.0);
	return unit * 2.0 - 1.0;
}

static double lunarSmoothStep(double inValue) noexcept {
	const double clamped = std::clamp(inValue, 0.0, 1.0);
	return clamped * clamped * (3.0 - 2.0 * clamped);
}

static std::size_t lunarTerrainVertexCount(
	const LunarTerrainConfig& inConfig) noexcept {
	return static_cast<std::size_t>(inConfig.cellsX_ + 1U)
		* static_cast<std::size_t>(inConfig.cellsZ_ + 1U);
}

static std::size_t lunarTerrainIndex(
	const LunarTerrainConfig& inConfig,
	std::uint32_t inVertexX,
	std::uint32_t inVertexZ) noexcept {
	return static_cast<std::size_t>(inVertexZ)
		* static_cast<std::size_t>(inConfig.cellsX_ + 1U)
		+ static_cast<std::size_t>(inVertexX);
}

static double lunarTriangleSlope(
	double inHeight00,
	double inHeight10,
	double inHeight01,
	double inHeight11,
	double inCellSize,
	LunarTerrainTriangle inTriangle) noexcept {
	double heightDx = 0.0;
	double heightDz = 0.0;
	if (inTriangle == LunarTerrainTriangle::LowerRight) {
		heightDx = (inHeight10 - inHeight00) / inCellSize;
		heightDz = (inHeight11 - inHeight10) / inCellSize;
	} else {
		heightDx = (inHeight11 - inHeight01) / inCellSize;
		heightDz = (inHeight01 - inHeight00) / inCellSize;
	}
	return std::hypot(heightDx, heightDz);
}

std::string LunarTerrainConfig::validationError() const {
	if (cellsX_ == 0U or cellsZ_ == 0U) {
		return "lunar terrain requires at least one cell on each axis";
	}
	if (cellsX_ > 1048575U or cellsZ_ > 1048575U) {
		return "lunar terrain axis exceeds the host-oracle vertex bound";
	}
	const std::uint64_t verticesX = static_cast<std::uint64_t>(cellsX_) + 1U;
	const std::uint64_t verticesZ = static_cast<std::uint64_t>(cellsZ_) + 1U;
	if (verticesX * verticesZ > 1048576ULL) {
		return "lunar terrain exceeds the one-million-vertex host-oracle bound";
	}
	if (not std::isfinite(cellSize_) or cellSize_ <= 0.0) {
		return "lunar terrain cell size must be finite and positive";
	}
	if (not std::isfinite(maxAbsHeight_) or maxAbsHeight_ < 0.0) {
		return "lunar terrain height bound must be finite and non-negative";
	}
	if (not std::isfinite(maxSlope_) or maxSlope_ < 0.0) {
		return "lunar terrain slope bound must be finite and non-negative";
	}
	if (not std::isfinite(padHalfExtent_) or padHalfExtent_ < 0.0
		or not std::isfinite(padTransitionWidth_)
		or padTransitionWidth_ < 0.0) {
		return "lunar terrain pad dimensions must be finite and non-negative";
	}
	const double halfWidthX = static_cast<double>(cellsX_) * cellSize_ * 0.5;
	const double halfWidthZ = static_cast<double>(cellsZ_) * cellSize_ * 0.5;
	const double requiredHalfExtent = padHalfExtent_ + cellSize_
		+ padTransitionWidth_;
	if (requiredHalfExtent > std::min(halfWidthX, halfWidthZ)) {
		return "lunar terrain pad and guarded transition do not fit the tile";
	}
	return {};
}

LunarTerrain LunarTerrain::invalid_(
	const LunarTerrainConfig& inConfig,
	std::string inError) {
	LunarTerrain terrain;
	terrain.config_ = inConfig;
	terrain.heights_.clear();
	terrain.error_ = std::move(inError);
	return terrain;
}

LunarTerrain LunarTerrain::createFlat(
	const LunarTerrainConfig& inConfig) {
	const std::string configError = inConfig.validationError();
	if (not configError.empty()) {
		return invalid_(inConfig, configError);
	}
	return createFromHeights(
		inConfig, std::vector<double>(lunarTerrainVertexCount(inConfig), 0.0));
}

LunarTerrain LunarTerrain::createSeeded(
	const LunarTerrainConfig& inConfig,
	const LunarEpisodeManifest& inManifest) {
	const std::string configError = inConfig.validationError();
	if (not configError.empty()) {
		return invalid_(inConfig, configError);
	}
	const std::string manifestError = inManifest.validationError();
	if (not manifestError.empty()) {
		return invalid_(inConfig, manifestError);
	}

	const std::size_t vertexCount = lunarTerrainVertexCount(inConfig);
	std::vector<double> heights(vertexCount, 0.0);
	for (std::uint32_t vertexZ = 0U; vertexZ <= inConfig.cellsZ_; ++vertexZ) {
		for (std::uint32_t vertexX = 0U; vertexX <= inConfig.cellsX_; ++vertexX) {
			heights[lunarTerrainIndex(inConfig, vertexX, vertexZ)] =
				lunarTerrainSignedUnit(
					inManifest.terrainSeed_, vertexX, vertexZ);
		}
	}

	std::vector<double> smoothed(vertexCount, 0.0);
	for (std::uint32_t pass = 0U; pass < 4U; ++pass) {
		for (std::uint32_t vertexZ = 0U; vertexZ <= inConfig.cellsZ_; ++vertexZ) {
			for (std::uint32_t vertexX = 0U; vertexX <= inConfig.cellsX_; ++vertexX) {
				double sum = 4.0 * heights[
					lunarTerrainIndex(inConfig, vertexX, vertexZ)];
				double weight = 4.0;
				if (vertexX > 0U) {
					sum += heights[lunarTerrainIndex(
						inConfig, vertexX - 1U, vertexZ)];
					weight += 1.0;
				}
				if (vertexX < inConfig.cellsX_) {
					sum += heights[lunarTerrainIndex(
						inConfig, vertexX + 1U, vertexZ)];
					weight += 1.0;
				}
				if (vertexZ > 0U) {
					sum += heights[lunarTerrainIndex(
						inConfig, vertexX, vertexZ - 1U)];
					weight += 1.0;
				}
				if (vertexZ < inConfig.cellsZ_) {
					sum += heights[lunarTerrainIndex(
						inConfig, vertexX, vertexZ + 1U)];
					weight += 1.0;
				}
				smoothed[lunarTerrainIndex(inConfig, vertexX, vertexZ)] =
					sum / weight;
			}
		}
		heights.swap(smoothed);
	}

	const double minX = -static_cast<double>(inConfig.cellsX_)
		* inConfig.cellSize_ * 0.5;
	const double minZ = -static_cast<double>(inConfig.cellsZ_)
		* inConfig.cellSize_ * 0.5;
	const double guardedPad = inConfig.padHalfExtent_ + inConfig.cellSize_;
	// One zero-height cell guards the declared pad. It keeps both triangles and
	// their normals flat at the inclusive pad boundary before the smooth ring.
	for (std::uint32_t vertexZ = 0U; vertexZ <= inConfig.cellsZ_; ++vertexZ) {
		const double positionZ = minZ
			+ static_cast<double>(vertexZ) * inConfig.cellSize_;
		for (std::uint32_t vertexX = 0U; vertexX <= inConfig.cellsX_; ++vertexX) {
			const double positionX = minX
				+ static_cast<double>(vertexX) * inConfig.cellSize_;
			const double squareRadius = std::max(
				std::abs(positionX), std::abs(positionZ));
			double reliefWeight = 0.0;
			if (squareRadius > guardedPad) {
				if (inConfig.padTransitionWidth_ == 0.0) {
					reliefWeight = 1.0;
				} else {
					reliefWeight = lunarSmoothStep(
						(squareRadius - guardedPad)
						/ inConfig.padTransitionWidth_);
				}
			}
			heights[lunarTerrainIndex(inConfig, vertexX, vertexZ)] *=
				reliefWeight;
		}
	}

	double maximumHeight = 0.0;
	double maximumSlope = 0.0;
	for (const double height : heights) {
		maximumHeight = std::max(maximumHeight, std::abs(height));
	}
	for (std::uint32_t cellZ = 0U; cellZ < inConfig.cellsZ_; ++cellZ) {
		for (std::uint32_t cellX = 0U; cellX < inConfig.cellsX_; ++cellX) {
			const double height00 = heights[lunarTerrainIndex(
				inConfig, cellX, cellZ)];
			const double height10 = heights[lunarTerrainIndex(
				inConfig, cellX + 1U, cellZ)];
			const double height01 = heights[lunarTerrainIndex(
				inConfig, cellX, cellZ + 1U)];
			const double height11 = heights[lunarTerrainIndex(
				inConfig, cellX + 1U, cellZ + 1U)];
			maximumSlope = std::max(maximumSlope, lunarTriangleSlope(
				height00, height10, height01, height11,
				inConfig.cellSize_, LunarTerrainTriangle::LowerRight));
			maximumSlope = std::max(maximumSlope, lunarTriangleSlope(
				height00, height10, height01, height11,
				inConfig.cellSize_, LunarTerrainTriangle::UpperLeft));
		}
	}
	double scale = 1.0;
	if (maximumHeight > 0.0) {
		scale = std::min(scale, inConfig.maxAbsHeight_ / maximumHeight);
	}
	if (maximumSlope > 0.0) {
		scale = std::min(scale, inConfig.maxSlope_ / maximumSlope);
	}
	for (double& height : heights) {
		height *= scale;
	}
	return createFromHeights(inConfig, heights);
}

LunarTerrain LunarTerrain::createFromHeights(
	const LunarTerrainConfig& inConfig,
	const std::vector<double>& inHeights) {
	const std::string configError = inConfig.validationError();
	if (not configError.empty()) {
		return invalid_(inConfig, configError);
	}
	if (inHeights.size() != lunarTerrainVertexCount(inConfig)) {
		return invalid_(inConfig, "lunar terrain height count does not match its grid");
	}
	for (const double height : inHeights) {
		if (not std::isfinite(height)) {
			return invalid_(inConfig, "lunar terrain contains a non-finite height");
		}
		if (std::abs(height) > inConfig.maxAbsHeight_ + 1.0e-12) {
			return invalid_(inConfig, "lunar terrain exceeds its height bound");
		}
	}
	for (std::uint32_t cellZ = 0U; cellZ < inConfig.cellsZ_; ++cellZ) {
		for (std::uint32_t cellX = 0U; cellX < inConfig.cellsX_; ++cellX) {
			const double height00 = inHeights[lunarTerrainIndex(
				inConfig, cellX, cellZ)];
			const double height10 = inHeights[lunarTerrainIndex(
				inConfig, cellX + 1U, cellZ)];
			const double height01 = inHeights[lunarTerrainIndex(
				inConfig, cellX, cellZ + 1U)];
			const double height11 = inHeights[lunarTerrainIndex(
				inConfig, cellX + 1U, cellZ + 1U)];
			const double lowerSlope = lunarTriangleSlope(
				height00, height10, height01, height11,
				inConfig.cellSize_, LunarTerrainTriangle::LowerRight);
			const double upperSlope = lunarTriangleSlope(
				height00, height10, height01, height11,
				inConfig.cellSize_, LunarTerrainTriangle::UpperLeft);
			if (lowerSlope > inConfig.maxSlope_ + 1.0e-12
				or upperSlope > inConfig.maxSlope_ + 1.0e-12) {
				return invalid_(inConfig, "lunar terrain exceeds its triangle-slope bound");
			}
		}
	}

	LunarTerrain terrain;
	terrain.config_ = inConfig;
	terrain.heights_ = inHeights;
	terrain.error_.clear();
	return terrain;
}

double LunarTerrain::minX() const noexcept {
	return -static_cast<double>(config_.cellsX_) * config_.cellSize_ * 0.5;
}

double LunarTerrain::maxX() const noexcept {
	return -minX();
}

double LunarTerrain::minZ() const noexcept {
	return -static_cast<double>(config_.cellsZ_) * config_.cellSize_ * 0.5;
}

double LunarTerrain::maxZ() const noexcept {
	return -minZ();
}

bool LunarTerrain::contains(double inX, double inZ) const noexcept {
	return isValid() and std::isfinite(inX) and std::isfinite(inZ)
		and inX >= minX() and inX <= maxX()
		and inZ >= minZ() and inZ <= maxZ();
}

bool LunarTerrain::isOnPad(double inX, double inZ) const noexcept {
	return contains(inX, inZ)
		and std::max(std::abs(inX), std::abs(inZ))
			<= config_.padHalfExtent_;
}

std::size_t LunarTerrain::vertexIndex_(
	std::uint32_t inVertexX,
	std::uint32_t inVertexZ) const noexcept {
	return lunarTerrainIndex(config_, inVertexX, inVertexZ);
}

double LunarTerrain::vertexHeight(
	std::uint32_t inVertexX,
	std::uint32_t inVertexZ) const noexcept {
	if (not isValid() or inVertexX > config_.cellsX_
		or inVertexZ > config_.cellsZ_) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	return heights_[vertexIndex_(inVertexX, inVertexZ)];
}

LunarTerrainSample LunarTerrain::query(
	double inX,
	double inZ) const noexcept {
	LunarTerrainSample sample;
	if (not contains(inX, inZ)) {
		return sample;
	}

	const double gridX = (inX - minX()) / config_.cellSize_;
	const double gridZ = (inZ - minZ()) / config_.cellSize_;
	if (gridX == static_cast<double>(config_.cellsX_)) {
		sample.cellX_ = config_.cellsX_ - 1U;
		sample.localX_ = 1.0;
	} else {
		sample.cellX_ = static_cast<std::uint32_t>(std::floor(gridX));
		sample.localX_ = gridX - static_cast<double>(sample.cellX_);
	}
	if (gridZ == static_cast<double>(config_.cellsZ_)) {
		sample.cellZ_ = config_.cellsZ_ - 1U;
		sample.localZ_ = 1.0;
	} else {
		sample.cellZ_ = static_cast<std::uint32_t>(std::floor(gridZ));
		sample.localZ_ = gridZ - static_cast<double>(sample.cellZ_);
	}

	const double height00 = vertexHeight(sample.cellX_, sample.cellZ_);
	const double height10 = vertexHeight(sample.cellX_ + 1U, sample.cellZ_);
	const double height01 = vertexHeight(sample.cellX_, sample.cellZ_ + 1U);
	const double height11 = vertexHeight(
		sample.cellX_ + 1U, sample.cellZ_ + 1U);
	double heightDx = 0.0;
	double heightDz = 0.0;
	if (sample.localZ_ <= sample.localX_) {
		sample.triangle_ = LunarTerrainTriangle::LowerRight;
		sample.height_ = height00
			+ sample.localX_ * (height10 - height00)
			+ sample.localZ_ * (height11 - height10);
		heightDx = (height10 - height00) / config_.cellSize_;
		heightDz = (height11 - height10) / config_.cellSize_;
	} else {
		sample.triangle_ = LunarTerrainTriangle::UpperLeft;
		sample.height_ = height00
			+ sample.localX_ * (height11 - height01)
			+ sample.localZ_ * (height01 - height00);
		heightDx = (height11 - height01) / config_.cellSize_;
		heightDz = (height01 - height00) / config_.cellSize_;
	}
	sample.normal_ = oa::vlm::DVec3(-heightDx, 1.0, -heightDz).normalized();
	sample.inBounds_ = true;
	return sample;
}

} // namespace oa
