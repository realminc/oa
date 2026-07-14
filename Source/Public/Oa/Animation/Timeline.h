// OaTimeline — Timeline with exact timecode, playback, and evaluation.
//
// Replaces ad-hoc local timers with a unified animation time system.
// Supports rational frame rates, deterministic stepping, and parent-to-local
// time transforms for clip/sequence composition.
//
// Design principles (CoreAnimationRenderArchitecture.md §3.2, §4):
// - No process-global mutable time (time as a scene convention, not a process singleton)
// - Time advancement and animation evaluation are separate
// - Use explicit time domains: wall -> app -> scene -> clip -> property
//
// Timeline hierarchy:
//   wall clock
//     -> OaTimeline (application root)
//          -> OaTimeline (scene)
//               -> OaTimeTransform -> clip local time
//                    -> OaCurve evaluation
//
// Usage:
//   OaTimeline timeline;
//   timeline.SetFrameRate(OaFrameRate::Fps24());
//   timeline.SetRange({0, 10});  // 0-10 seconds
//   timeline.Play();
//
//   // Per-frame update:
//   timeline.Update(deltaMs);
//   OaF32 evalTime = timeline.GetLocalTime().ToSeconds();
//
#pragma once

#include <Oa/Animation/TimeCode.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>

// ═════════════════════════════════════════════════════════════════════════════
// Time direction for playback
// ═════════════════════════════════════════════════════════════════════════════

enum class OaTimeDirection : OaU8 {
	Forward,
	Reverse,
};

// ═════════════════════════════════════════════════════════════════════════════
// Timeline playback state
// ═════════════════════════════════════════════════════════════════════════════

enum class OaTimelineState : OaU8 {
	Stopped,
	Playing,
	Paused,
	Scrubbing,   // Manual frame-by-frame seeking
};

// ═════════════════════════════════════════════════════════════════════════════
// OaTimeTransform — Maps parent time to local time with offset, scale, trim.
//
// Used for clip/sequence composition and retiming.
// local_time = (parent_time - offset) * scale
//
// ═════════════════════════════════════════════════════════════════════════════

struct OaTimeTransform {
	OaTimeCode Offset;    // Start time offset in parent time
	OaF32 Scale = 1.0f;   // Time scale (1.0 = normal, 2.0 = 2x fast, 0.5 = slow-mo)
	OaTimeCode TrimStart; // Local trim (skip first N seconds of clip)
	OaTimeCode TrimEnd;   // Local trim (end clip at local time)

	[[nodiscard]] static OaTimeTransform Identity() noexcept {
		return OaTimeTransform{};
	}

	// Create with offset only
	[[nodiscard]] static OaTimeTransform WithOffset(const OaTimeCode& InOffset) noexcept {
		return OaTimeTransform{.Offset = InOffset, .Scale = 1.0f};
	}

	// Create with scale only (retiming)
	[[nodiscard]] static OaTimeTransform WithScale(OaF32 InScale) noexcept {
		return OaTimeTransform{.Offset = OaTimeCode{}, .Scale = InScale};
	}

	// Transform parent time to local time
	[[nodiscard]] OaTimeCode ToLocal(const OaTimeCode& InParentTime) const noexcept {
		OaTimeCode local = (InParentTime - Offset) * Scale;
		return local;
	}

	// Transform local time back to parent time
	[[nodiscard]] OaTimeCode ToParent(const OaTimeCode& InLocalTime) const noexcept {
		if (Scale == 0.0f) return Offset;
		return InLocalTime * (1.0f / Scale) + Offset;
	}

	// Apply trim to local time
	[[nodiscard]] OaTimeCode ApplyTrim(const OaTimeCode& InLocalTime) const noexcept {
		if (InLocalTime < TrimStart) return TrimStart;
		if (!TrimEnd.IsZero() && InLocalTime > TrimEnd) return TrimEnd;
		return InLocalTime;
	}

	// Check if local time is within trim bounds
	[[nodiscard]] bool IsInTrim(const OaTimeCode& InLocalTime) const noexcept {
		if (InLocalTime < TrimStart) return false;
		if (!TrimEnd.IsZero() && InLocalTime > TrimEnd) return false;
		return true;
	}
};

// ═════════════════════════════════════════════════════════════════════════════
// OaTimeline — Exact timecode-based timeline with playback control.
//
// Core time evaluation for animation, video, and rendering.
// ═════════════════════════════════════════════════════════════════════════════

class OaTimeline {
public:
	OaTimeline() = default;
	~OaTimeline() = default;

	// ═════════════════════════════════════════════════════════════════════════
	// Configuration
	// ═════════════════════════════════════════════════════════════════════════

	void SetFrameRate(const OaFrameRate& InRate) noexcept { FrameRate_ = InRate; }
	[[nodiscard]] const OaFrameRate& GetFrameRate() const noexcept { return FrameRate_; }

	void SetRange(const OaTimeRange& InRange) noexcept { Range_ = InRange; }
	[[nodiscard]] const OaTimeRange& GetRange() const noexcept { return Range_; }

	void SetLooping(bool InLoop) noexcept { Range_.Loop = InLoop; }
	[[nodiscard]] bool IsLooping() const noexcept { return Range_.Loop; }

	void SetPlaybackDirection(OaTimeDirection InDir) noexcept { Direction_ = InDir; }
	[[nodiscard]] OaTimeDirection GetPlaybackDirection() const noexcept { return Direction_; }

	void SetPlaybackSpeed(OaF32 InSpeed) noexcept { PlaybackSpeed_ = InSpeed; }
	[[nodiscard]] OaF32 GetPlaybackSpeed() const noexcept { return PlaybackSpeed_; }

	void SetTimeTransform(const OaTimeTransform& InTransform) noexcept { Transform_ = InTransform; }
	[[nodiscard]] const OaTimeTransform& GetTimeTransform() const noexcept { return Transform_; }

	// ═════════════════════════════════════════════════════════════════════════
	// Playback Control
	// ═════════════════════════════════════════════════════════════════════════

	void Play() noexcept { State_ = OaTimelineState::Playing; }
	void Pause() noexcept { State_ = (State_ == OaTimelineState::Playing) ? OaTimelineState::Paused : State_; }
	void Stop() noexcept;
	void Seek(const OaTimeCode& InTime) noexcept;

	// Seek to specific frame (convenience)
	void SeekToFrame(OaI64 InFrameNumber) noexcept {
		Seek(OaTimeCode::FromFrame(InFrameNumber, FrameRate_, TicksPerSecond_));
	}

	// Step by frame delta (for scrubbing)
	void StepFrames(OaI64 InFrameDelta) noexcept {
		Seek(RootTime_.FrameDelta(FrameRate_, InFrameDelta));
	}

	[[nodiscard]] OaTimelineState GetState() const noexcept { return State_; }
	[[nodiscard]] bool IsPlaying() const noexcept { return State_ == OaTimelineState::Playing; }
	[[nodiscard]] bool IsPaused() const noexcept { return State_ == OaTimelineState::Paused; }
	[[nodiscard]] bool IsStopped() const noexcept { return State_ == OaTimelineState::Stopped; }

	// ═════════════════════════════════════════════════════════════════════════
	// Time Queries
	// ═════════════════════════════════════════════════════════════════════════

	// Root time (application/scene time, before transform)
	[[nodiscard]] const OaTimeCode& GetRootTime() const noexcept { return RootTime_; }

	// Local time (after time transform applied)
	[[nodiscard]] OaTimeCode GetLocalTime() const noexcept;

	// Current frame number (at current frame rate)
	[[nodiscard]] OaI64 GetCurrentFrame() const noexcept;

	// Time as seconds (API edge)
	[[nodiscard]] OaF32 GetLocalTimeSeconds() const noexcept { return GetLocalTime().ToSeconds(); }

	// Delta time from last update
	[[nodiscard]] OaF32 GetDeltaTimeSeconds() const noexcept { return DeltaTimeSeconds_; }

	// Duration of current frame
	[[nodiscard]] OaF32 GetFrameDurationSeconds() const noexcept { return FrameRate_.FrameDurationSeconds() / PlaybackSpeed_; }

	// ═════════════════════════════════════════════════════════════════════════
	// Update (call once per frame with wall-clock delta)
	// ═════════════════════════════════════════════════════════════════════════

	void Update(OaF32 InDeltaMs) noexcept;

	// Manual advance (for Fixed/Manual modes or batch rendering)
	void Advance(const OaTimeCode& InDelta) noexcept;

	// ═════════════════════════════════════════════════════════════════════════
	// Fixed timestep mode
	// ═════════════════════════════════════════════════════════════════════════

	void SetFixedTimestep(bool InFixed) noexcept { FixedTimestep_ = InFixed; }
	[[nodiscard]] bool IsFixedTimestep() const noexcept { return FixedTimestep_; }

	// ═════════════════════════════════════════════════════════════════════════
	// Parent/Child Hierarchy
	// ═════════════════════════════════════════════════════════════════════════

	void SetParent(const OaTimeline* InParent) noexcept { Parent_ = InParent; }
	[[nodiscard]] const OaTimeline* GetParent() const noexcept { return Parent_; }

private:
	// Time configuration
	OaFrameRate FrameRate_ = OaFrameRate::Fps24();
	OaI64 TicksPerSecond_ = 48000;
	OaTimeRange Range_{};
	OaTimeTransform Transform_ = OaTimeTransform::Identity();

	// Playback state
	OaTimelineState State_ = OaTimelineState::Stopped;
	OaTimeDirection Direction_ = OaTimeDirection::Forward;
	OaF32 PlaybackSpeed_ = 1.0f;

	// Time values
	OaTimeCode RootTime_{0, 48000};     // Current root time
	OaTimeCode PreviousRootTime_{0, 48000};
	OaF32 DeltaTimeSeconds_ = 0.0f;

	// Mode flags
	bool FixedTimestep_ = false;

	// Hierarchy (for NLE-style composition)
	const OaTimeline* Parent_ = nullptr;

	// Internal
	void ApplyRangeAndLoop() noexcept;
};

// ═════════════════════════════════════════════════════════════════════════════
// OaTimelineEvaluator — Helper for evaluating properties at timeline time
//
// Combines timeline with curves for property animation.
// ═════════════════════════════════════════════════════════════════════════════

class OaTimelineEvaluator {
public:
	explicit OaTimelineEvaluator(const OaTimeline& InTimeline) : Timeline_(InTimeline) {}

	// Evaluate a float curve at current timeline time
	// (requires OaCurveFloat - defined in Curve.h)
	// OaF32 Evaluate(const OaCurveFloat& InCurve) const;

	[[nodiscard]] const OaTimeline& GetTimeline() const noexcept { return Timeline_; }

private:
	const OaTimeline& Timeline_;
};
