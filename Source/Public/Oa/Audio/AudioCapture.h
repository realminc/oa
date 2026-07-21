// OA Audio — real-time device capture into a bounded lock-free F32 ring.

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Std/UniquePtr.h>
#include <Oa/Core/Types.h>

class OaEngine;

struct OaAudioCaptureConfig {
	OaU32 SampleRate = 48'000U;
	OaU32 ChannelCount = 2U;
	OaU32 RingMilliseconds = 500U;
};

struct OaAudioCaptureChunk {
	OaVec<OaF32> Interleaved;
	OaU32 SampleRate = 0U;
	OaU32 ChannelCount = 0U;
	OaU64 FrameCount = 0U;
	OaU64 FirstFrameIndex = 0U;
	OaU64 PresentationTimestamp = 0U; // monotonic microseconds
};

class OaAudioCapture {
public:
	struct Impl;

	OaAudioCapture() = default;
	OaAudioCapture(OaAudioCapture&& InOther) noexcept;
	OaAudioCapture& operator=(OaAudioCapture&& InOther) noexcept;
	OaAudioCapture(const OaAudioCapture&) = delete;
	OaAudioCapture& operator=(const OaAudioCapture&) = delete;
	~OaAudioCapture();

	[[nodiscard]] static OaResult<OaAudioCapture> Open(
		OaEngine& InEngine,
		const OaAudioCaptureConfig& InConfig = {});
	[[nodiscard]] OaStatus Start();
	[[nodiscard]] OaStatus Stop();
	// Non-blocking. Returns false when no complete captured frames are ready.
	bool Poll(OaAudioCaptureChunk& OutChunk, OaU32 InMaxFrames = 4096U);
	// Stops callback delivery and releases the device. This is the explicit,
	// result-bearing completion boundary.
	[[nodiscard]] OaStatus Close();
	// Compatibility wrapper that logs Close() failures.
	void Destroy();

	[[nodiscard]] bool IsStarted() const noexcept;
	[[nodiscard]] OaU64 DroppedFrameCount() const noexcept;

private:
	void Abandon_() noexcept;
	static OaStatus CompleteRetired_(void* InPayload);
	static void ReleaseRetired_(void* InPayload);
	OaUniquePtr<Impl> Impl_;
};
