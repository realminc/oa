// oa::VideoRecorder — composed hardware video/audio recorder.
//
// oa::VideoRecorder is the file sink counterpart to oa::VideoPlayer/capture
// sources: it owns one vulkan Video encoder and one container muxer, accepts
// the common oa::VideoFrame contract, and finalizes a playable file.

#pragma once

#include <oa/audio/audioCapture.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/vision/videoDecoder.h>
#include <oa/vision/videoEncoder.h>
#include <oa/vision/videoMuxer.h>

namespace oa {
class Engine;
}
namespace oavk {
class Buffer;
}
namespace oa {
class Texture;
}

namespace oa {

struct VideoRecorderConfig {
  oa::String outputPath = "output.mp4";
  VideoEncodeProfile encode = {};
  YCbCrModel colorSpace = YCbCrModel::BT709;
  bool fullRange = false;
  bool audioEnabled = false;
  AudioEncodeProfile audio = {};
};

class VideoRecorder {
public:
  VideoRecorder() = default;
  VideoRecorder(VideoRecorder &&inOther) noexcept;
  VideoRecorder &operator=(VideoRecorder &&inOther) noexcept;
  VideoRecorder(const VideoRecorder &) = delete;
  VideoRecorder &operator=(const VideoRecorder &) = delete;
  ~VideoRecorder();

  [[nodiscard]] static oa::Result<VideoRecorder>
  create(Engine &inEngine, const VideoRecorderConfig &inConfig);

  // Record one packed RGBA8 bindless buffer.
  [[nodiscard]] oa::Status writeRgba(const oavk::Buffer &inRgba,
                                     oa::U32 inWidth, oa::U32 inHeight,
                                     oa::U64 inPts);

  // Record a common capture/decode/render frame.
  [[nodiscard]] oa::Status write(const VideoFrame &inFrame);
  // Non-blocking image-input variant. outInputConsumed signals after the
  // source image has returned to its published layout/queue family.
  [[nodiscard]] oa::Status writeAsync(const VideoFrame &inFrame,
                                      oa::Event &outInputConsumed);

  // Record a buffer- or image-backed render target.
  [[nodiscard]] oa::Status write(const oa::Texture &inTexture, oa::U64 inPts);

  // Add captured interleaved F32 audio.
  [[nodiscard]] oa::Status writeAudio(oa::Span<const oa::F32> inInterleaved,
                                      oa::U32 inSampleRate,
                                      oa::U32 inChannelCount, oa::U64 inPts);

  [[nodiscard]] oa::Status writeAudio(const AudioCaptureChunk &inChunk);

  // flush the encoder and finalize the container. idempotent.
  [[nodiscard]] oa::Status finalize();
  // Releases encoder and host-file state without finalizing.
  [[nodiscard]] oa::Status close();

  [[nodiscard]] bool isOpen() const noexcept {
    return engine_ != nullptr and not finalized_;
  }
  [[nodiscard]] oa::U32 getFrameCount() const noexcept {
    return submittedFrameCount_;
  }
  [[nodiscard]] const VideoRecorderConfig &getConfig() const noexcept {
    return config_;
  }

private:
  void moveFrom_(VideoRecorder &&inOther) noexcept;
  [[nodiscard]] oa::Status writeEncoded_(const EncodedVideoPacket &inFrame);
  [[nodiscard]] oa::Status
  writeAudioAligned_(oa::Span<const oa::F32> inInterleaved, oa::U64 inPts);
  [[nodiscard]] oa::Status
  writeAudioPackets_(oa::Vec<EncodedAudioPacket> &inPackets);
  [[nodiscard]] oa::Status setFirstVideoPts_(oa::U64 inPts);

  Engine *engine_ = nullptr;
  VideoRecorderConfig config_ = {};
  VideoEncoder encoder_;
  VideoMuxer muxer_;
  AudioEncoder audioEncoder_;
  struct PendingAudioChunk {
    oa::Vec<oa::F32> samples;
    oa::U64 pts = 0U;
  };
  oa::Vec<PendingAudioChunk> pendingAudio_;
  oa::Vec<oa::F32> audioScratch_;
  oa::U64 firstVideoPts_ = 0U;
  oa::U64 nextAudioFrame_ = 0U;
  bool hasFirstVideoPts_ = false;
  oa::U32 submittedFrameCount_ = 0;
  oa::U32 muxedFrameCount_ = 0;
  bool codecConfigWritten_ = false;
  bool finalized_ = false;
};

} // namespace oa
