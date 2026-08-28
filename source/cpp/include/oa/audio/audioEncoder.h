// oa::AudioEncoder — stateful native streaming audio encode session.
//
// PcmS16 is uncompressed, deterministic, and has no external codec dependency.
// Converting FP32 capture samples to signed 16-bit PCM is intentionally
// quantizing. Additional codecs plug into this session boundary rather than
// hiding in a muxer or a stateless operation.

#pragma once

#include <oa/audio/type.h>
#include <oa/core/status.h>
#include <oa/core/std/uniquePtr.h>

namespace oa {

struct EncodedAudioPacket {
  oa::Vector<oa::U8> bitstream;
  oa::I64 presentationFrame = 0;
  oa::U32 durationFrames = 0U;
};

struct AudioEncodeProfile {
  AudioCodec codec = AudioCodec::PcmS16;
  oa::U32 sampleRate = 48'000U;
  oa::U32 channelCount = 2U;
  oa::U32 framesPerPacket = 1'024U;
};

class AudioEncoder {
public:
  struct Impl;

  AudioEncoder();
  AudioEncoder(AudioEncoder &&inOther) noexcept;
  AudioEncoder &operator=(AudioEncoder &&inOther) noexcept;
  AudioEncoder(const AudioEncoder &) = delete;
  AudioEncoder &operator=(const AudioEncoder &) = delete;
  ~AudioEncoder();

  [[nodiscard]] static oa::Result<AudioEncoder>
  create(const AudioEncodeProfile &inProfile);
  [[nodiscard]] oa::Status encode(oa::Span<const oa::F32> inInterleaved,
                                  oa::Vector<EncodedAudioPacket> &outPackets);
  [[nodiscard]] oa::Status flush(oa::Vector<EncodedAudioPacket> &outPackets);
  // Discards any unflushed partial packet and closes the session.
  [[nodiscard]] oa::Status close();

  [[nodiscard]] const AudioEncodeProfile &getProfile() const noexcept;
  [[nodiscard]] oa::Span<const oa::U8> getCodecConfig() const noexcept;
  [[nodiscard]] oa::U32 getPrimingFrames() const noexcept;
  [[nodiscard]] bool isOpen() const noexcept { return impl_ != nullptr; }

private:
  oa::UniquePtr<Impl> impl_;
};

} // namespace oa
