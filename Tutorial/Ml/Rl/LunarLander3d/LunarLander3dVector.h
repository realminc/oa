#pragma once

#include "LunarLander3d.h"

#include <Oa/Ml/Rl/Environment.h>

inline constexpr OaU32 OA_LUNAR_VECTOR_CONFIG_LAYOUT_VERSION = 1U;
inline constexpr OaU32 OA_LUNAR_VECTOR_STATE_LAYOUT_VERSION = 1U;

class OaLunarLander3dVectorConfig {
public:
	OaU32 Environments_ = 1U;
	OaU64 Seed_ = 1U;
	OaLunarLander3dConfig Environment_;
};

class OaLunarLander3dVectorStep {
public:
	OaMatrix Observation_;
	OaMatrix NextObservation_;
	OaMatrix Reward_;
	OaMatrix Terminated_;
	OaMatrix Truncated_;
	OaMatrix EndReason_;

	[[nodiscard]] bool IsValid() const noexcept;
};

// Semantic host snapshot of one lane's current episode. This deliberately
// hides the private shader state layout; consumers can report learning evidence
// without depending on state-buffer offsets or widths.
class OaLunarLander3dEpisodeTelemetry {
public:
	OaF32 EpisodeReturn_ = 0.0F;
	OaF32 FuelRemaining_ = 0.0F;
	OaF32 TerminalLinearSpeed_ = 0.0F;
	OaF32 TerminalAngularSpeed_ = 0.0F;
	OaF32 MaximumFootImpulse_ = 0.0F;
	OaU32 EpisodeStep_ = 0U;
	bool Terminated_ = false;
	bool Truncated_ = false;
	OaLunarEndReason EndReason_ = OaLunarEndReason::None;

	[[nodiscard]] bool IsFinite() const noexcept;
};

// Tutorial-local batched FP32 implementation of the scalar v0 contract. This
// L2 slice accepts one immutable flat heightfield; procedural per-episode
// terrain remains the separately gated L5 extension. The session borrows one
// engine and inherits exact-event submission from OaRlEnvironment.
class OaLunarLander3dVector final : public OaRlEnvironment {
public:
	~OaLunarLander3dVector() override = default;
	OaLunarLander3dVector(const OaLunarLander3dVector&) = delete;
	OaLunarLander3dVector& operator=(
		const OaLunarLander3dVector&) = delete;
	OaLunarLander3dVector(OaLunarLander3dVector&&) noexcept = default;
	OaLunarLander3dVector& operator=(
		OaLunarLander3dVector&&) noexcept = default;

	[[nodiscard]] static OaResult<OaLunarLander3dVector> CreateFlat(
		OaEngine& InEngine,
		const OaLunarLander3dVectorConfig& InConfig);

	// CreateFlat and both reset calls record work. The caller submits and waits
	// on the exact returned event before observing their results.
	[[nodiscard]] OaStatus Reset();
	[[nodiscard]] OaStatus ResetDone();
	// Out-of-range Discrete(8) values consume one transition and terminate only
	// that lane. A completed lane retains its terminal flags with zero reward
	// until ResetDone advances its private episode counter.
	[[nodiscard]] OaResult<OaLunarLander3dVectorStep> Step(
		const OaMatrix& InAction);
	// External-stop entries are UInt8 [Environments]. Invalid actions take
	// precedence; otherwise a nonzero entry truncates its lane without advancing
	// the episode step or changing its observation.
	[[nodiscard]] OaResult<OaLunarLander3dVectorStep> Step(
		const OaMatrix& InAction,
		const OaMatrix& InExternalStop);

	[[nodiscard]] bool IsValid() const noexcept;
	[[nodiscard]] const OaMatrix& Observation() const noexcept override {
		return Observation_;
	}
	[[nodiscard]] const OaMatrix& EndReason() const noexcept {
		return EndReason_;
	}
	[[nodiscard]] const OaLunarLander3dVectorConfig& Config() const noexcept {
		return Config_;
	}
	[[nodiscard]] const OaRlEnvironmentSpec& Spec() const noexcept override {
		return Spec_;
	}
	[[nodiscard]] OaU32 Environments() const noexcept override {
		return Config_.Environments_;
	}
	// Host observation is legal only with neither an active recording nor a
	// pending event. The copy exposes semantic episode telemetry, never
	// StateF32_/U32_.
	[[nodiscard]] OaResult<OaVec<OaLunarLander3dEpisodeTelemetry>>
		CopyEpisodeTelemetry() const;

private:
	explicit OaLunarLander3dVector(OaEngine& InEngine);
	[[nodiscard]] OaStatus RecordReset_(bool InOnlyCompleted);
	[[nodiscard]] OaResult<OaLunarLander3dVectorStep> RecordStep_(
		const OaMatrix& InAction,
		const OaMatrix& InExternalStop);
	[[nodiscard]] OaU64 EffectiveSeed_() const noexcept;

protected:
	[[nodiscard]] OaStatus RecordResetEnvironment_(OaU64 InSeed) override;
	[[nodiscard]] OaResult<OaRlEnvironmentTransition>
		RecordStepEnvironment_(const OaMatrix& InAction) override;
	[[nodiscard]] OaStatus RecordResetCompleted_() override;
	void CommitRecordedState_() noexcept override;
	void RollbackRecordedState_() noexcept override;

private:
	OaLunarLander3dVectorConfig Config_;
	OaRlEnvironmentSpec Spec_;
	OaMatrix ConfigF32_;
	OaMatrix ConfigU32_;
	OaMatrix TerrainF32_;
	OaMatrix StateF32_;
	OaMatrix StateU32_;
	OaMatrix Observation_;
	OaMatrix TransitionObservation_;
	OaMatrix Reward_;
	OaMatrix Terminated_;
	OaMatrix Truncated_;
	OaMatrix EndReason_;
	// Persistent storage is required because Step records deferred work. The
	// zero-mask input must outlive Submit, cancellation, and any queued event.
	OaMatrix NoExternalStop_;
	OaU64 PendingSeed_ = 0U;
	bool HasPendingSeed_ = false;
	bool HasCommittedState_ = false;
	bool HasPendingFullReset_ = false;
};
