// OaTimeline — Implementation
//
// Exact timecode-based timeline with playback control.

#include <Oa/Animation/Timeline.h>
#include <algorithm>

namespace {

constexpr OaF32 MS_TO_SECONDS = 0.001f;

} // namespace

void OaTimeline::Stop() noexcept {
	State_ = OaTimelineState::Stopped;
	RootTime_ = Range_.Start;
	PreviousRootTime_ = RootTime_;
	DeltaTimeSeconds_ = 0.0f;
}

void OaTimeline::Seek(const OaTimeCode& InTime) noexcept {
	PreviousRootTime_ = RootTime_;
	RootTime_ = Range_.Clamp(InTime);
	DeltaTimeSeconds_ = RootTime_.ToSeconds() - PreviousRootTime_.ToSeconds();
}

OaTimeCode OaTimeline::GetLocalTime() const noexcept {
	return Transform_.ToLocal(RootTime_);
}

OaI64 OaTimeline::GetCurrentFrame() const noexcept {
	return RootTime_.ToFrame(FrameRate_);
}

void OaTimeline::Update(OaF32 InDeltaMs) noexcept {
	if (State_ != OaTimelineState::Playing) {
		DeltaTimeSeconds_ = 0.0f;
		return;
	}

	PreviousRootTime_ = RootTime_;

	// Calculate time advancement
	OaF32 deltaSeconds = InDeltaMs * MS_TO_SECONDS * PlaybackSpeed_;
	if (Direction_ == OaTimeDirection::Reverse) {
		deltaSeconds = -deltaSeconds;
	}

	if (FixedTimestep_) {
		// Fixed timestep: advance by exact frame duration
		deltaSeconds = GetFrameDurationSeconds();
		if (Direction_ == OaTimeDirection::Reverse) {
			deltaSeconds = -deltaSeconds;
		}
	}

	// Advance root time
	OaTimeCode deltaTimeCode = OaTimeCode::FromSeconds(deltaSeconds, TicksPerSecond_);
	RootTime_ = OaTimeCode{
		RootTime_.Ticks + deltaTimeCode.ToTicks(TicksPerSecond_),
		TicksPerSecond_
	};

	// Apply range and loop
	ApplyRangeAndLoop();

	// Calculate actual delta
	DeltaTimeSeconds_ = RootTime_.ToSeconds() - PreviousRootTime_.ToSeconds();
}

void OaTimeline::Advance(const OaTimeCode& InDelta) noexcept {
	PreviousRootTime_ = RootTime_;

	// Convert delta to our tick scale
	OaI64 deltaTicks = InDelta.ToTicks(TicksPerSecond_);
	RootTime_ = OaTimeCode{RootTime_.Ticks + deltaTicks, TicksPerSecond_};

	// Apply range and loop
	ApplyRangeAndLoop();

	DeltaTimeSeconds_ = RootTime_.ToSeconds() - PreviousRootTime_.ToSeconds();
}

void OaTimeline::ApplyRangeAndLoop() noexcept {
	if (RootTime_ < Range_.Start) {
		if (Range_.Loop) {
			RootTime_ = Range_.ApplyLoop(RootTime_);
		} else {
			RootTime_ = Range_.Start;
			State_ = OaTimelineState::Stopped;
		}
	} else if (RootTime_ > Range_.End) {
		if (Range_.Loop) {
			RootTime_ = Range_.ApplyLoop(RootTime_);
		} else {
			RootTime_ = Range_.End;
			State_ = OaTimelineState::Stopped;
		}
	}
}
