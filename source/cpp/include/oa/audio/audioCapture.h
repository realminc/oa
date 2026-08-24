// OA Audio — real-time device capture into a bounded lock-free F32 ring.

#pragma once

#include <oa/core/status.h>
#include <oa/core/std/uniquePtr.h>
#include <oa/core/types.h>

namespace oa {

class Engine;

struct AudioCaptureConfig {
	oa::U32 sampleRate = 48'000U;
	oa::U32 channelCount = 2U;
	oa::U32 ringMilliseconds = 500U;
};

struct AudioCaptureChunk {
	oa::Vec<oa::F32> interleaved;
	oa::U32 sampleRate = 0U;
	oa::U32 channelCount = 0U;
	oa::U64 frameCount = 0U;
	oa::U64 firstFrameIndex = 0U;
	oa::U64 presentationTimestamp = 0U; // monotonic microseconds
};

class AudioCapture {
public:
	struct Impl;

	AudioCapture() = default;
	AudioCapture(AudioCapture&& inOther) noexcept;
	AudioCapture& operator=(AudioCapture&& inOther) noexcept;
	AudioCapture(const AudioCapture&) = delete;
	AudioCapture& operator=(const AudioCapture&) = delete;
	~AudioCapture();

	[[nodiscard]] static oa::Result<AudioCapture> open(Engine& inEngine, const AudioCaptureConfig& inConfig = {});
	[[nodiscard]] oa::Status start();
	[[nodiscard]] oa::Status stop();
	// Non-blocking. Returns false when no complete captured frames are ready.
	bool poll(AudioCaptureChunk& outChunk, oa::U32 inMaxFrames = 4096U);
	// Stops callback delivery and releases the device. This is the explicit,
	// result-bearing completion boundary.
	[[nodiscard]] oa::Status close();

	[[nodiscard]] bool isStarted() const noexcept;
	[[nodiscard]] oa::U64 droppedFrameCount() const noexcept;

private:
	void abandon_() noexcept;
	static oa::Status completeRetired_(void* inPayload);
	static void releaseRetired_(void* inPayload);
	oa::UniquePtr<Impl> impl_;
};

} // namespace oa
