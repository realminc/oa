#pragma once

#include "LunarLander3dPhysics.h"

#include <string>

// Deterministic scalar reference controller used by the solvability oracle and
// the interactive diagnostic viewer. Vertical thrust has priority when the
// descent envelope is violated; otherwise position/velocity feedback targets a
// bounded lateral tilt and damps both controlled attitude axes. Yaw is
// irrelevant to the symmetric v0 lander and centered pad.
[[nodiscard]] OaLunarAction OaLunarScriptedLandingAction(
	const OaLunarLander3dConfig& InConfig,
	const OaLunarLander3dState& InState) noexcept;

class OaLunarScalarEnvironment {
public:
	OaLunarScalarEnvironment() = default;

	[[nodiscard]] static OaLunarScalarEnvironment CreateFlat(
		const OaLunarLander3dConfig& InConfig,
		const OaLunarEpisodeManifest& InManifest);
	[[nodiscard]] static OaLunarScalarEnvironment CreateSeeded(
		const OaLunarLander3dConfig& InConfig,
		const OaLunarEpisodeManifest& InManifest);
	[[nodiscard]] static OaLunarScalarEnvironment CreateWithTerrain(
		const OaLunarLander3dConfig& InConfig,
		const OaLunarEpisodeManifest& InManifest,
		const OaLunarTerrain& InTerrain);

	[[nodiscard]] bool IsValid() const noexcept { return Error_.empty(); }
	[[nodiscard]] const std::string& Error() const noexcept { return Error_; }
	[[nodiscard]] const OaLunarLander3dConfig& Config() const noexcept {
		return Config_;
	}
	[[nodiscard]] const OaLunarEpisodeManifest& Manifest() const noexcept {
		return Manifest_;
	}
	[[nodiscard]] const OaLunarTerrain& Terrain() const noexcept {
		return Terrain_;
	}
	[[nodiscard]] const OaLunarLander3dState& State() const noexcept {
		return State_;
	}
	[[nodiscard]] std::array<float, OA_LUNAR_OBSERVATION_SIZE>
		Observation() const noexcept;

	[[nodiscard]] bool Reset() noexcept;
	[[nodiscard]] bool SetState(
		const OaLunarLander3dState& InState) noexcept;
	[[nodiscard]] OaLunarTransition Step(
		std::uint32_t InAction,
		bool InExternalStop = false);

private:
	[[nodiscard]] static OaLunarScalarEnvironment Invalid_(
		const OaLunarLander3dConfig& InConfig,
		const OaLunarEpisodeManifest& InManifest,
		const OaLunarTerrain& InTerrain,
		std::string InError);
	[[nodiscard]] OaLunarLander3dState SpawnState_() const noexcept;

	OaLunarLander3dConfig Config_;
	OaLunarEpisodeManifest Manifest_;
	OaLunarTerrain Terrain_;
	OaLunarLander3dState State_;
	std::string Error_ = "lunar scalar environment has not been created";
};
