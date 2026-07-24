#pragma once

#include "LunarLander3dTypes.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class OaLunarTerrainConfig {
public:
	std::uint32_t CellsX_ = 32U;
	std::uint32_t CellsZ_ = 32U;
	double CellSize_ = 1.0;
	double MaxAbsHeight_ = 1.5;
	double MaxSlope_ = 0.35;
	double PadHalfExtent_ = 3.0;
	double PadTransitionWidth_ = 3.0;

	[[nodiscard]] std::string ValidationError() const;
	[[nodiscard]] constexpr bool operator==(
		const OaLunarTerrainConfig& InOther) const noexcept = default;
};

enum class OaLunarTerrainTriangle : std::uint8_t {
	LowerRight = 0U,
	UpperLeft = 1U,
};

class OaLunarTerrainSample {
public:
	bool InBounds_ = false;
	double Height_ = 0.0;
	OaLunarVec3 Normal_{0.0, 1.0, 0.0};
	std::uint32_t CellX_ = 0U;
	std::uint32_t CellZ_ = 0U;
	double LocalX_ = 0.0;
	double LocalZ_ = 0.0;
	OaLunarTerrainTriangle Triangle_ = OaLunarTerrainTriangle::LowerRight;
};

class OaLunarTerrain {
public:
	OaLunarTerrain() = default;

	[[nodiscard]] static OaLunarTerrain CreateFlat(
		const OaLunarTerrainConfig& InConfig);
	[[nodiscard]] static OaLunarTerrain CreateSeeded(
		const OaLunarTerrainConfig& InConfig,
		const OaLunarEpisodeManifest& InManifest);
	[[nodiscard]] static OaLunarTerrain CreateFromHeights(
		const OaLunarTerrainConfig& InConfig,
		const std::vector<double>& InHeights);

	[[nodiscard]] bool IsValid() const noexcept { return Error_.empty(); }
	[[nodiscard]] const std::string& Error() const noexcept { return Error_; }
	[[nodiscard]] const OaLunarTerrainConfig& Config() const noexcept {
		return Config_;
	}
	[[nodiscard]] const std::vector<double>& Heights() const noexcept {
		return Heights_;
	}
	[[nodiscard]] double MinX() const noexcept;
	[[nodiscard]] double MaxX() const noexcept;
	[[nodiscard]] double MinZ() const noexcept;
	[[nodiscard]] double MaxZ() const noexcept;
	[[nodiscard]] bool Contains(double InX, double InZ) const noexcept;
	[[nodiscard]] bool IsOnPad(double InX, double InZ) const noexcept;
	[[nodiscard]] double VertexHeight(
		std::uint32_t InVertexX,
		std::uint32_t InVertexZ) const noexcept;
	// The tile bounds are inclusive. An internal grid edge selects the cell on
	// its positive side; the maximum tile edge selects the final cell. Every
	// cell uses the v00-to-v11 diagonal, and equality selects LowerRight. The
	// returned unit normal is the upward normal of that exact triangle plane.
	[[nodiscard]] OaLunarTerrainSample Query(
		double InX,
		double InZ) const noexcept;

private:
	[[nodiscard]] static OaLunarTerrain Invalid_(
		const OaLunarTerrainConfig& InConfig,
		std::string InError);
	[[nodiscard]] std::size_t VertexIndex_(
		std::uint32_t InVertexX,
		std::uint32_t InVertexZ) const noexcept;

	OaLunarTerrainConfig Config_;
	std::vector<double> Heights_;
	std::string Error_ = "terrain has not been created";
};
