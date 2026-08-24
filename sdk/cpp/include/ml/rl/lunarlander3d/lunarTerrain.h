#pragma once

#include <ml/rl/lunarlander3d/lunarLander3dTypes.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace oa {

class LunarTerrainConfig {
public:
	std::uint32_t cellsX_ = 32U;
	std::uint32_t cellsZ_ = 32U;
	double cellSize_ = 1.0;
	double maxAbsHeight_ = 1.5;
	double maxSlope_ = 0.35;
	double padHalfExtent_ = 3.0;
	double padTransitionWidth_ = 3.0;

	[[nodiscard]] std::string validationError() const;
	[[nodiscard]] constexpr bool operator==(
		const LunarTerrainConfig& inOther) const noexcept = default;
};

enum class LunarTerrainTriangle : std::uint8_t {
	LowerRight = 0U,
	UpperLeft = 1U,
};

class LunarTerrainSample {
public:
	bool inBounds_ = false;
	double height_ = 0.0;
	oa::vlm::DVec3 normal_{0.0, 1.0, 0.0};
	std::uint32_t cellX_ = 0U;
	std::uint32_t cellZ_ = 0U;
	double localX_ = 0.0;
	double localZ_ = 0.0;
	LunarTerrainTriangle triangle_ = LunarTerrainTriangle::LowerRight;
};

class LunarTerrain {
public:
	LunarTerrain() = default;

	[[nodiscard]] static LunarTerrain createFlat(
		const LunarTerrainConfig& inConfig);
	[[nodiscard]] static LunarTerrain createSeeded(
		const LunarTerrainConfig& inConfig,
		const LunarEpisodeManifest& inManifest);
	[[nodiscard]] static LunarTerrain createFromHeights(
		const LunarTerrainConfig& inConfig,
		const std::vector<double>& inHeights);

	[[nodiscard]] bool isValid() const noexcept { return error_.empty(); }
	[[nodiscard]] const std::string& error() const noexcept { return error_; }
	[[nodiscard]] const LunarTerrainConfig& config() const noexcept {
		return config_;
	}
	[[nodiscard]] const std::vector<double>& heights() const noexcept {
		return heights_;
	}
	[[nodiscard]] double minX() const noexcept;
	[[nodiscard]] double maxX() const noexcept;
	[[nodiscard]] double minZ() const noexcept;
	[[nodiscard]] double maxZ() const noexcept;
	[[nodiscard]] bool contains(double inX, double inZ) const noexcept;
	[[nodiscard]] bool isOnPad(double inX, double inZ) const noexcept;
	[[nodiscard]] double vertexHeight(
		std::uint32_t inVertexX,
		std::uint32_t inVertexZ) const noexcept;
	// The tile bounds are inclusive. An internal grid edge selects the cell on
	// its positive side; the maximum tile edge selects the final cell. Every
	// cell uses the v00-to-v11 diagonal, and equality selects LowerRight. The
	// returned unit normal is the upward normal of that exact triangle plane.
	[[nodiscard]] LunarTerrainSample query(
		double inX,
		double inZ) const noexcept;

private:
	[[nodiscard]] static LunarTerrain invalid_(
		const LunarTerrainConfig& inConfig,
		std::string inError);
	[[nodiscard]] std::size_t vertexIndex_(
		std::uint32_t inVertexX,
		std::uint32_t inVertexZ) const noexcept;

	LunarTerrainConfig config_;
	std::vector<double> heights_;
	std::string error_ = "terrain has not been created";
};

} // namespace oa
