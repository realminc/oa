// OaItVideo — compatibility iterator/playback adapter over OaVideo.
//
// Core demux/decode/reorder ownership lives in OaVideo. This adapter supplies
// OaIterator and time-paced convenience without becoming a second media stack.

#pragma once

#include <Oa/Core/Iterator.h>
#include <Oa/Vision/Video.h>

using OaItVideoConfig = OaVideoConfig;

class OaItVideo : public OaIterator {
public:
	[[nodiscard]] static OaResult<OaItVideo> Create(OaEngine& InEngine, const OaItVideoConfig& InConfig);

	OaItVideo() = default;
	OaItVideo(OaItVideo&&) noexcept = default;
	OaItVideo& operator=(OaItVideo&&) noexcept = default;
	OaItVideo(const OaItVideo&) = delete;
	OaItVideo& operator=(const OaItVideo&) = delete;
	~OaItVideo() override = default;

	[[nodiscard]] OaStatus Close();
	void Destroy();
	[[nodiscard]] bool IsDone() const override;
	void Next() override;
	void Reset() override;
	[[nodiscard]] OaI64 Index() const override;

	void Play();
	void Pause();
	void TogglePlay();
	[[nodiscard]] bool IsPlaying() const;
	OaStatus StepForward();
	OaStatus StepBackward();
	OaStatus StepFrames(OaI32 InFrameDelta);
	OaStatus Seek(OaU64 InTimestamp);
	OaStatus Flush();
	void Tick(OaF32 InDeltaMs);

	[[nodiscard]] const OaVideoFrame& CurrentFrame() const;
	[[nodiscard]] OaResult<OaVec<OaU8>> ReadbackCurrentRgba();
	void MarkCurrentFrameConsumed(const OaEvent& InConsumed);
	void MarkCurrentFrameConsumed(const OaVkTimelineSemaphore& InSemaphore, OaU64 InValue);
	[[nodiscard]] OaU32 Width() const;
	[[nodiscard]] OaU32 Height() const;
	[[nodiscard]] OaU32 FrameRate() const;
	[[nodiscard]] OaUsize FrameCount() const;
	[[nodiscard]] OaF32 FrameIntervalMs() const;
	[[nodiscard]] bool IsEos() const;
	[[nodiscard]] const OaContainerInfo& GetContainerInfo() const;

private:
	OaOption<OaVideo> Video_;
};
