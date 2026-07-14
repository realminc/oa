// OaTime — Global timeline and local time management.
//
// A scene time node: every scene has a time node that
// provides global and local time queries. Supports animation playback,
// video-style timeline control, and animated textures/UI elements.
//
// Design principles:
// - Global timeline accessible from OaContext: OaContext::GetTime()
// - Local time per object/scene for hierarchical animation
// - Query global/local time, control playback like video
// - Support for animated textures, UI elements, pan/zoom effects
// - Non-linear editing and video playback capabilities
//
// Usage:
//   auto& ctx = OaContext::GetDefault();
//   OaF32 globalTime = ctx.GetTime();
//   OaF32 localTime = animNode.GetLocalTime();
//   animNode.SetPlaybackSpeed(1.0f);  // 1.0 = normal speed
//   animNode.Play();
//   animNode.Pause();
//   animNode.Seek(5.0f);  // seek to 5 seconds

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Math.h>

// Time mode: how time advances
enum class OaTimeMode : OaU8 {
	Realtime,      // Wall-clock time (default)
	Fixed,         // Fixed timestep (e.g. 60 FPS)
	Manual,        // Manual advance (user-controlled)
	External,      // External time source (video, audio, etc.)
};

// Playback state
enum class OaPlaybackState : OaU8 {
	Stopped,
	Playing,
	Paused,
	Seeking,
};

// Time range (start, end, loop behavior)
struct OaTimeRange {
	OaF32 Start = 0.0f;
	OaF32 End   = 10.0f;  // 10 second default
	bool  Loop  = false;
};

// Global time node (time1 equivalent)
class OaTimeNode {
public:
	OaTimeNode() = default;
	~OaTimeNode() = default;

	// Time queries
	[[nodiscard]] OaF32 GetGlobalTime() const noexcept;
	[[nodiscard]] OaF32 GetLocalTime() const noexcept;
	[[nodiscard]] OaF32 GetDeltaTime() const noexcept;

	// Playback control
	void Play();
	void Pause();
	void Stop();
	void Seek(OaF32 InTime);
	void SetPlaybackSpeed(OaF32 InSpeed);
	[[nodiscard]] OaF32 GetPlaybackSpeed() const noexcept;

	// Time mode
	void SetTimeMode(OaTimeMode InMode);
	[[nodiscard]] OaTimeMode GetTimeMode() const noexcept;

	// Time range
	void SetTimeRange(const OaTimeRange& InRange);
	[[nodiscard]] const OaTimeRange& GetTimeRange() const noexcept;

	// State
	[[nodiscard]] OaPlaybackState GetPlaybackState() const noexcept;

	// Manual advance (for Fixed/Manual modes)
	void Advance(OaF32 InDeltaTime);

private:
	OaF32          GlobalTime_    = 0.0f;
	OaF32          LocalTime_     = 0.0f;
	OaF32          DeltaTime_     = 0.0f;
	OaF32          PlaybackSpeed_ = 1.0f;
	OaTimeMode     TimeMode_      = OaTimeMode::Realtime;
	OaPlaybackState State_        = OaPlaybackState::Stopped;
	OaTimeRange    Range_;
};

// Context integration (to be added to OaContext later)
// namespace OaContext {
//     [[nodiscard]] OaF32 GetTime() const noexcept;
//     void SetTimeNode(OaTimeNode* InNode);
// }
