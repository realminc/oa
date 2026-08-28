#pragma once

#include <ml/rl/lunarlander3d/lunarLander3d.h>

#include <oa/ml/environment.h>

namespace oa {

inline constexpr oa::U32 kLunarVectorConfigLayoutVersion = 1U;
inline constexpr oa::U32 kLunarVectorStateLayoutVersion = 1U;

class LunarLander3dVectorConfig {
public:
	oa::U32 environments_ = 1U;
	oa::U64 seed_ = 1U;
	LunarLander3dConfig environment_;
};

class LunarLander3dVectorStep {
public:
	oa::Matrix observation_;
	oa::Matrix nextObservation_;
	oa::Matrix reward_;
	oa::Matrix terminated_;
	oa::Matrix truncated_;
	oa::Matrix endReason_;

	[[nodiscard]] bool isValid() const noexcept;
};

// Semantic host snapshot of one lane's current episode. This deliberately
// hides the private shader state layout; consumers can report learning evidence
// without depending on state-buffer offsets or widths.
class LunarLander3dEpisodeTelemetry {
public:
	oa::F32 episodeReturn_ = 0.0F;
	oa::F32 fuelRemaining_ = 0.0F;
	oa::F32 terminalLinearSpeed_ = 0.0F;
	oa::F32 terminalAngularSpeed_ = 0.0F;
	oa::F32 maximumFootImpulse_ = 0.0F;
	oa::U32 episodeStep_ = 0U;
	bool terminated_ = false;
	bool truncated_ = false;
	LunarEndReason endReason_ = LunarEndReason::None;

	[[nodiscard]] bool isFinite() const noexcept;
};

// Batched FP32 implementation of the scalar v0 contract. This installed
// environment accepts one immutable flat heightfield; procedural per-episode
// terrain remains a separately gated extension. The session borrows one engine
// and inherits exact-event submission from oa::Environment.
class LunarLander3dVector final : public oa::Environment {
public:
	using oa::Environment::reset;

	~LunarLander3dVector() override = default;
	LunarLander3dVector(const LunarLander3dVector&) = delete;
	LunarLander3dVector& operator=(
		const LunarLander3dVector&) = delete;
	LunarLander3dVector(LunarLander3dVector&&) noexcept = default;
	LunarLander3dVector& operator=(
		LunarLander3dVector&&) noexcept = default;

	[[nodiscard]] static oa::Result<LunarLander3dVector> createFlat(
		oa::Engine& inEngine,
		const LunarLander3dVectorConfig& inConfig);

	// CreateFlat and both reset calls record work. The caller submits and waits
	// on the exact returned event before observing their results.
	[[nodiscard]] oa::Status reset();
	[[nodiscard]] oa::Status resetDone();
	// out-of-range discrete(8) values consume one transition and terminate only
	// that lane. A completed lane retains its terminal flags with zero reward
	// until resetDone advances its private episode counter.
	[[nodiscard]] oa::Result<LunarLander3dVectorStep> step(
		const oa::Matrix& inAction);
	// external-stop entries are UInt8 [environments]. Invalid actions take
	// precedence; otherwise a nonzero entry truncates its lane without advancing
	// the episode step or changing its observation.
	[[nodiscard]] oa::Result<LunarLander3dVectorStep> step(
		const oa::Matrix& inAction,
		const oa::Matrix& inExternalStop);

	[[nodiscard]] bool isValid() const noexcept;
	[[nodiscard]] const oa::Matrix& observation() const noexcept override {
		return observation_;
	}
	[[nodiscard]] const oa::Matrix& endReason() const noexcept {
		return endReason_;
	}
	[[nodiscard]] const LunarLander3dVectorConfig& config() const noexcept {
		return config_;
	}
	[[nodiscard]] const oa::EnvironmentSpec& spec() const noexcept override {
		return spec_;
	}
	[[nodiscard]] oa::U32 environments() const noexcept override {
		return config_.environments_;
	}
	// Host observation is legal only with neither an active recording nor a
	// pending event. The copy exposes semantic episode telemetry, never
	// stateF32_/u32_.
	[[nodiscard]] oa::Result<oa::Vector<LunarLander3dEpisodeTelemetry>>
		copyEpisodeTelemetry() const;

private:
	explicit LunarLander3dVector(oa::Engine& inEngine);
	[[nodiscard]] oa::Status recordReset_(bool inOnlyCompleted);
	[[nodiscard]] oa::Result<LunarLander3dVectorStep> recordStep_(
		const oa::Matrix& inAction,
		const oa::Matrix& inExternalStop);
	[[nodiscard]] oa::U64 effectiveSeed_() const noexcept;

protected:
	[[nodiscard]] oa::Status recordReset_(oa::U64 inSeed) override;
	[[nodiscard]] oa::Result<oa::EnvironmentTransition>
		recordStep_(const oa::Matrix& inAction) override;
	[[nodiscard]] oa::Status recordResetCompleted_() override;
	void commitRecordedState_() noexcept override;
	void rollbackRecordedState_() noexcept override;

private:
	LunarLander3dVectorConfig config_;
	oa::EnvironmentSpec spec_;
	oa::Matrix configF32_;
	oa::Matrix configU32_;
	oa::Matrix terrainF32_;
	oa::Matrix stateF32_;
	oa::Matrix stateU32_;
	oa::Matrix observation_;
	oa::Matrix transitionObservation_;
	oa::Matrix reward_;
	oa::Matrix terminated_;
	oa::Matrix truncated_;
	oa::Matrix endReason_;
	// persistent storage is required because step records deferred work. The
	// zero-mask input must outlive submit, cancellation, and any queued event.
	oa::Matrix noExternalStop_;
	oa::U64 pendingSeed_ = 0U;
	bool hasPendingSeed_ = false;
	bool hasCommittedState_ = false;
	bool hasPendingFullReset_ = false;
};

} // namespace oa
