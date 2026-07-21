// OaAudioStream — incremental decode and realtime playback session.
//
// Container/codec and device I/O are explicit CPU boundaries. Decoded PCM is
// held in a bounded lock-free ring; callback code never allocates or locks.

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Std/UniquePtr.h>

class OaEngine;

struct OaAudioStreamConfig {
	OaString Uri;
	bool Loop = false;
	OaU32 RingMilliseconds = 500U;
};

class OaAudioStream {
public:
	struct Impl;

	OaAudioStream() = default;
	OaAudioStream(OaAudioStream&&) noexcept;
	OaAudioStream& operator=(OaAudioStream&&) noexcept;
	OaAudioStream(const OaAudioStream&) = delete;
	OaAudioStream& operator=(const OaAudioStream&) = delete;
	~OaAudioStream();

	[[nodiscard]] static OaResult<OaAudioStream> Open(
		OaEngine& InEngine,
		const OaAudioStreamConfig& InConfig);
	[[nodiscard]] static OaResult<OaAudioStream> Open(
		OaEngine& InEngine,
		OaStringView InUri);

	[[nodiscard]] OaStatus Play();
	void Pause();
	[[nodiscard]] OaStatus Seek(OaU64 InTimestampUs);
	void SetLoop(bool InLoop);
	// Stops playback, joins decoding, and releases codec/device state.
	[[nodiscard]] OaStatus Close();

	[[nodiscard]] bool IsOpen() const noexcept;
	[[nodiscard]] bool IsPlaying() const noexcept;
	[[nodiscard]] bool IsEos() const noexcept;
	[[nodiscard]] OaU32 SampleRate() const noexcept;
	[[nodiscard]] OaU32 ChannelCount() const noexcept;
	[[nodiscard]] OaU64 DurationUs() const noexcept;
	[[nodiscard]] OaU64 PositionUs() const noexcept;
	[[nodiscard]] OaU64 UnderrunFrameCount() const noexcept;

private:
	void Abandon_() noexcept;
	static OaStatus CompleteRetired_(void* InPayload);
	static void ReleaseRetired_(void* InPayload);
	OaUniquePtr<Impl> Impl_;
};
