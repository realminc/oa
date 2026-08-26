// oa::VideoPlayer — composed decoded media source/session
//
// Owns oa::VideoDemuxer + oa::VideoDecoder, presentation reordering, playback
// control, and retained frame resources.
//
// usage:
//   auto videoR = oa::VideoPlayer::open(engine, {.uri = "video.mp4"});
//   oa::VideoPlayer video = oa::move(*videoR);
//   // each frame:
//   OA_RETURN_IF_ERROR(video.next());
//   consume(video.currentFrame());
//
// playback control:
//   video.togglePlay();        // Space
//   video.next();              // Right arrow
//   video.stepBackward();      // Left arrow (seeks to prior IDR + replays
//   forward)
//
// looping (cfg.loop = true, default) is handled internally: when the
// underlying stream hits EOS, seek(0) is issued and the next frame comes
// back from the start.

#pragma once

#include <oa/audio/audioPlayer.h>
#include <oa/core/status.h>
#include <oa/core/std.h>
#include <oa/vision/videoDecoder.h>
#include <oa/vision/videoDemuxer.h>

namespace oa {

class Engine;

struct VideoPlayerConfig {
  // URI may be a local path or a supported network URL.
  oa::String uri;

  oa::U32 maxDpbSlots = 16;

  // When true, the stream wraps to t=0 on EOS so playback runs forever.
  // When false, isDone() flips to true once the final packet is decoded.
  bool loop = true;

  // Prefer the hardware YCbCr sampler path. The compute conversion remains
  // available as the compatibility and exact-control fallback.
  bool preferHardwareYCbCr = true;

  // 0 = use frameRate from the container (defaults to 30 if absent).
  oa::F32 frameRateOverride = 0.0F;

  // Start in the playing state. When false, the app must explicitly
  // call play() or next() to advance.
  bool startPlaying = true;
  // open and synchronize the first audio track when one is present. Failure
  // to find an audio track preserves video-only operation.
  bool audio = true;
  VideoDemuxerConfig demuxerConfig = {};

  // Reorder buffer depth (in decoded frames) used to convert decode order
  // → display order. H.264/H.265 streams with B-frames emit packets in
  // decode order (I P B B P B B …) but must be displayed in PTS order
  // (I B B P B B P …). Without this buffer, B-frame streams play as
  // "two forward, one back". 4 is enough for typical IBBBP GOPs; bump
  // higher for unusual encodes. set to 0 to disable reordering (correct
  // only when the stream has no B-frames).
  oa::U32 reorderDepth = 4;

  // conversion filter: Nearest = sharp pixel edges, no smoothing.
  // Linear = smoother but softer output. Default Nearest for video
  // to preserve decoded-frame sharpness.
  Filter filter = Filter::Nearest;
};

class VideoPlayer {
public:
  [[nodiscard]] static oa::Result<VideoPlayer>
  open(Engine &inEngine, const VideoPlayerConfig &inConfig);

  VideoPlayer() = default;
  VideoPlayer(VideoPlayer &&) noexcept;
  VideoPlayer &operator=(VideoPlayer &&) noexcept;
  VideoPlayer(const VideoPlayer &) = delete;
  VideoPlayer &operator=(const VideoPlayer &) = delete;
  ~VideoPlayer();

  // Explicit completion and resource-release boundary. Waits exact retained
  // frame-consumer events before releasing decoder-owned image storage.
  [[nodiscard]] oa::Status close();

  // isDone() is permanently false in looping mode. in non-looping mode,
  // flips to true once the underlying stream's EOS has been observed and
  // the last decoded frame consumed.
  [[nodiscard]] bool isDone() const;
  // Advances by exactly one decoded frame (ignores playing_, ignores the
  // wall-clock accumulator). Use tick() for time-paced playback.
  // advance to the next display-order frame. open() presents the first frame,
  // so the initial currentFrame() is immediately usable. next() reports
  // decode, demux and synchronization failures instead of hiding them.
  [[nodiscard]] oa::Status next();
  void reset();
  // Returns the number of frames decoded since the most recent rewind.
  [[nodiscard]] oa::I64 index() const { return index_; }

  // ─── playback control ─────────────────────────────────────────────────
  void play();
  void pause();
  void togglePlay();
  [[nodiscard]] bool isPlaying() const { return playing_; }
  [[nodiscard]] bool hasAudio() const { return audio_.hasValue(); }
  void setLoop(bool inLoop);
  [[nodiscard]] bool isLooping() const { return cfg_.loop; }
  // Unified viewer transport timebase. Public timeline APIs use microseconds;
  // container-specific ticks remain private to VideoPlayer::seek.
  [[nodiscard]] oa::U64 durationUs() const;
  [[nodiscard]] oa::U64 positionUs() const;
  [[nodiscard]] oa::Status seekUs(oa::U64 inTimestampUs);

  // seek to the previous keyframe and re-decode forward until one frame
  // before the current position. Costs up to GOP-size frames of decode.
  oa::Status stepBackward();

  // Scrub by a signed number of display-order frames. Positive movement
  // advances normally; negative movement seeks to the preceding keyframe
  // and replays through the reorder/presentation path.
  oa::Status stepFrames(oa::I32 inFrameDelta);
  // seek in the container video-track timebase. The decoder is recreated at
  // the preceding keyframe and replayed to the first display frame at/after
  // the requested timestamp.
  oa::Status seek(oa::U64 inTimestamp);
  // drain outstanding GPU work and clear queued display frames.
  oa::Status flush();

  // advance by wall clock when playing. When paused, accumulates nothing
  // (no implicit frame skipping after un-pause).
  void tick(oa::F32 inDeltaMs);

  // Most recently decoded frame. imageView is VK_NULL_HANDLE until the
  // first successful decode.
  [[nodiscard]] const VideoFrame &currentFrame() const { return frame_; }
  // convert the current decoder-owned frame to an ML tensor. conversion lives
  // on VideoPlayer rather than VideoFrame because the decoder session owns the
  // image views, YCbCr metadata, synchronization and conversion resources.
  [[nodiscard]] oa::Result<Matrix>
  currentFrameToMatrix(bool inNormalizeImageNet = true);
  [[nodiscard]] oa::Result<oa::Image>
  currentFrameToImage(bool inNormalizeImageNet = true);
  // Read back the current converted RGBA frame. This is intended for CPU
  // reference overlays and diagnostics; realtime inference should consume
  // currentFrame() directly on the GPU.
  [[nodiscard]] oa::Result<oa::Vec<oa::U8>> readbackCurrentRgba();
  // Record the compute submission that most recently sampled currentFrame().
  // The session will not recycle that RGBA image until inConsumed completes.
  // frames advanced without being rendered remain immediately reusable.
  void markCurrentFrameConsumed(const oa::Event &inConsumed);

  // ─── stream metadata passthroughs ─────────────────────────────────────
  [[nodiscard]] oa::U32 width() const;
  [[nodiscard]] oa::U32 height() const;
  [[nodiscard]] oa::U32 frameRate() const;
  [[nodiscard]] oa::Usize frameCount() const;
  [[nodiscard]] oa::F32 frameIntervalMs() const { return frameIntervalMs_; }
  [[nodiscard]] bool isEos() const;
  [[nodiscard]] const VideoContainerInfo &getContainerInfo() const;
  [[nodiscard]] const VideoDemuxerStats &getDemuxerStats() const;

private:
  // Each reorder buffer entry owns its own RGBA target image. We convert
  // NV12→RGBA at decode time so the DPB layer is free to be reused by the
  // next decode (the H.264 sliding-window otherwise corrupts data we'd
  // still be holding by reference).
  struct ReorderEntry {
    VideoFrame rgba = {};
    oa::U64 pts = 0;

    ReorderEntry() = default;
    ReorderEntry(const VideoFrame &inRgba, oa::U64 inPts)
        : rgba(inRgba), pts(inPts) {}
    ReorderEntry(const ReorderEntry &) = delete;
    ReorderEntry &operator=(const ReorderEntry &) = delete;
    ReorderEntry(ReorderEntry &&) noexcept = default;
    ReorderEntry &operator=(ReorderEntry &&) noexcept = default;
  };

  oa::Status decodeOneIntoReorder_();
  oa::Status fillReorderBuffer_();
  oa::Status popAndPresentLowestPts_();
  oa::Status seekDisplayFrame_(oa::Usize inTargetFrameIndex);
  oa::Status clearReorder_();
  oa::Status waitForPoolConsumers_();
  oa::Status restartDecoder_();
  void abandon_() noexcept;
  static oa::Status completeRetired_(void *inPayload);
  static void releaseRetired_(void *inPayload);
  // pool helpers — produce/release an RGBA target sized to the stream.
  [[nodiscard]] oa::Result<VideoFrame> acquireRgbaFromPool_();
  void releaseRgbaToPool_(const VideoFrame &inFrame);

  VideoPlayerConfig cfg_;
  Engine *engine_ = nullptr;
  oa::Optional<VideoDemuxer> demuxer_;
  oa::Optional<VideoDecoder> decoder_;
  oa::Optional<AudioPlayer> audio_;
  VideoFrame frame_ = {};
  oa::F32 frameIntervalMs_ = 1000.0F / 30.0F;
  oa::F32 accumulator_ = 0.0F;
  bool playing_ = true;
  bool reachedEos_ = false;
  bool demuxerEosCurrent_ = false;
  oa::Vec<ReorderEntry> reorder_;
  oa::Vec<oa::U64> displayPts_;
  // playback-owned pool of RGBA targets. Sized lazily to reorderDepth + 2
  // (held in reorder + currently displayed + one slack). The decoder owns
  // the actual VkImage lifetimes via its rgbImages_ table; we just track
  // who's holding which.
  oa::Vec<VideoFrame> rgbaPool_;
  oa::Vec<bool> rgbaPoolBusy_;
  oa::Vec<oa::Event> rgbaPoolConsumerEvents_;
  oa::I64 index_ = 0;
  oa::U64 demuxerFormatGeneration_ = 1U;
  oa::U64 demuxerReconnectCount_ = 0U;
};

} // namespace oa
