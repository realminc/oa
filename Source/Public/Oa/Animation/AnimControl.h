// OaAnimControl — Animation playback and timeline control.
//
// Provides high-level animation control similar to video playback or
// non-linear editing systems. Supports keyframe-based animation,
// timeline scrubbing, and playback control.
//
// Features:
// - Timeline scrubbing and seeking
// - Play/pause/stop control
// - Playback speed adjustment
// - Looping and ping-pong playback
// - Animation blending and layering
// - Event callbacks for keyframe hits
//
// Usage:
//   OaAnimControl controller;
//   controller.SetTimeline(0.0f, 10.0f);  // 10 second animation
//   controller.Play();
//   while (controller.IsPlaying()) {
//       controller.Update(deltaTime);
//       // Evaluate animation at controller.GetCurrentTime()
//   }

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Animation/Time.h>

#include <functional>

// Animation event callback type
using OaAnimEventCallback = std::function<void(OaF32)>;

// Playback direction
enum class OaPlaybackDirection : OaU8 {
	Forward,
	Reverse,
	PingPong,  // Forward then reverse
};

// Animation control
class OaAnimControl {
public:
	OaAnimControl() = default;
	~OaAnimControl() = default;

	// Timeline control
	void SetTimeline(OaF32 InStart, OaF32 InEnd);
	void SetLooping(bool InLoop);
	void SetPlaybackDirection(OaPlaybackDirection InDirection);

	// Playback control
	void Play();
	void Pause();
	void Stop();
	void Seek(OaF32 InTime);
	void SetPlaybackSpeed(OaF32 InSpeed);

	// Update (call once per frame)
	void Update(OaF32 InDeltaTime);

	// Queries
	[[nodiscard]] OaF32 GetCurrentTime() const noexcept;
	[[nodiscard]] OaF32 GetDuration() const noexcept;
	[[nodiscard]] bool IsPlaying() const noexcept;
	[[nodiscard]] bool IsLooping() const noexcept;
	[[nodiscard]] OaPlaybackDirection GetPlaybackDirection() const noexcept;
	[[nodiscard]] OaF32 GetPlaybackSpeed() const noexcept;

	// Event callbacks (keyframe hits, loop events, etc.)
	void SetOnKeyframeHit(OaAnimEventCallback InCallback);
	void SetOnLoop(OaAnimEventCallback InCallback);
	void SetOnComplete(OaAnimEventCallback InCallback);

private:
	OaF32              CurrentTime_     = 0.0f;
	OaF32              Duration_        = 10.0f;
	OaF32              PlaybackSpeed_   = 1.0f;
	bool               Looping_         = false;
	bool               Playing_         = false;
	OaPlaybackDirection Direction_       = OaPlaybackDirection::Forward;
	OaAnimEventCallback OnKeyframeHit_;
	OaAnimEventCallback OnLoop_;
	OaAnimEventCallback OnComplete_;
};
