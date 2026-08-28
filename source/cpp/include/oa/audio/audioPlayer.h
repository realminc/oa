// oa::AudioPlayer — incremental decode and realtime playback session.
//
// Container/codec and device I/O are explicit CPU boundaries. Decoded PCM is
// held in a bounded lock-free ring; callback code never allocates or locks.

#pragma once

#include <oa/core/status.h>
#include <oa/core/std/uniquePtr.h>

namespace oa {

class Engine;

struct AudioPlayerConfig {
  oa::String uri;
  bool loop = false;
  oa::U32 ringMilliseconds = 500U;
};

class AudioPlayer {
public:
  struct Impl;

  AudioPlayer();
  AudioPlayer(AudioPlayer &&) noexcept;
  AudioPlayer &operator=(AudioPlayer &&) noexcept;
  AudioPlayer(const AudioPlayer &) = delete;
  AudioPlayer &operator=(const AudioPlayer &) = delete;
  ~AudioPlayer();

  [[nodiscard]] static oa::Result<AudioPlayer>
  open(Engine &inEngine, const AudioPlayerConfig &inConfig);
  [[nodiscard]] static oa::Result<AudioPlayer> open(Engine &inEngine,
                                                    oa::StringView inUri);

  [[nodiscard]] oa::Status play();
  void pause();
  [[nodiscard]] oa::Status seek(oa::U64 inTimestampUs);
  void setLoop(bool inLoop);
  // Stops playback, joins decoding, and releases codec/device state.
  [[nodiscard]] oa::Status close();

  [[nodiscard]] bool isOpen() const noexcept;
  [[nodiscard]] bool isPlaying() const noexcept;
  [[nodiscard]] bool isEos() const noexcept;
  [[nodiscard]] oa::U32 sampleRate() const noexcept;
  [[nodiscard]] oa::U32 channelCount() const noexcept;
  [[nodiscard]] oa::U64 durationUs() const noexcept;
  [[nodiscard]] oa::U64 positionUs() const noexcept;
  [[nodiscard]] oa::U64 underrunFrameCount() const noexcept;

private:
  void abandon_() noexcept;
  static oa::Status completeRetired_(void *inPayload);
  static void releaseRetired_(void *inPayload);
  oa::UniquePtr<Impl> impl_;
};

} // namespace oa
