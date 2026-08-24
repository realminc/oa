#pragma once

#include <ml/rl/lunarlander3d/lunarLander3dPhysics.h>

#include <string>

namespace oa {

// Deterministic scalar reference controller used by the solvability oracle and
// the interactive diagnostic viewer. Vertical thrust has priority when the
// descent envelope is violated; otherwise position/velocity feedback targets a
// bounded lateral tilt and damps both controlled attitude axes. Yaw is
// irrelevant to the symmetric v0 lander and centered pad.
[[nodiscard]] LunarAction lunarScriptedLandingAction(
	const LunarLander3dConfig& inConfig,
	const LunarLander3dState& inState) noexcept;

class LunarScalarEnvironment {
public:
	LunarScalarEnvironment() = default;

	[[nodiscard]] static LunarScalarEnvironment createFlat(
		const LunarLander3dConfig& inConfig,
		const LunarEpisodeManifest& inManifest);
	[[nodiscard]] static LunarScalarEnvironment createSeeded(
		const LunarLander3dConfig& inConfig,
		const LunarEpisodeManifest& inManifest);
	[[nodiscard]] static LunarScalarEnvironment createWithTerrain(
		const LunarLander3dConfig& inConfig,
		const LunarEpisodeManifest& inManifest,
		const LunarTerrain& inTerrain);

	[[nodiscard]] bool isValid() const noexcept { return error_.empty(); }
	[[nodiscard]] const std::string& error() const noexcept { return error_; }
	[[nodiscard]] const LunarLander3dConfig& config() const noexcept {
		return config_;
	}
	[[nodiscard]] const LunarEpisodeManifest& manifest() const noexcept {
		return manifest_;
	}
	[[nodiscard]] const LunarTerrain& terrain() const noexcept {
		return terrain_;
	}
	[[nodiscard]] const LunarLander3dState& state() const noexcept {
		return state_;
	}
	[[nodiscard]] std::array<float, kLunarObservationSize>
		observation() const noexcept;

	[[nodiscard]] bool reset() noexcept;
	[[nodiscard]] bool setState(
		const LunarLander3dState& inState) noexcept;
	[[nodiscard]] LunarTransition step(
		std::uint32_t inAction,
		bool inExternalStop = false);

private:
	[[nodiscard]] static LunarScalarEnvironment invalid_(
		const LunarLander3dConfig& inConfig,
		const LunarEpisodeManifest& inManifest,
		const LunarTerrain& inTerrain,
		std::string inError);
	[[nodiscard]] LunarLander3dState spawnState_() const noexcept;

	LunarLander3dConfig config_;
	LunarEpisodeManifest manifest_;
	LunarTerrain terrain_;
	LunarLander3dState state_;
	std::string error_ = "lunar scalar environment has not been created";
};

} // namespace oa
