// OaVideo — decoded video source/session
//
// Owns OaVideoStream + OaVideoDecoder, presentation reordering and retained
// frame resources. OaItVideo is the optional iterator/playback adapter.
//
// Usage:
//   auto videoR = OaVideo::Open(engine, {.Uri = "video.mp4"});
//   OaVideo video = OaStdMove(*videoR);
//   // each frame:
//   OA_RETURN_IF_ERROR(video.Next());
//   consume(video.CurrentFrame());
//
// Playback control:
//   it.TogglePlay();        // Space
//   it.StepForward();       // Right arrow
//   it.StepBackward();      // Left arrow (seeks to prior IDR + replays forward)
//
// Looping (Cfg.Loop = true, default) is handled internally: when the
// underlying stream hits EOS, Seek(0) is issued and the next frame comes
// back from the start.

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Std.h>
#include <Oa/Vision/VideoStream.h>
#include <Oa/Vision/VideoDecoder.h>
#include <Oa/Audio/AudioStream.h>

class OaEngine;

class OaVideoConfig {
public:
	// URI is the canonical source and may be a local path or supported network
	// URL. Path is retained for source compatibility and is used when URI is
	// empty.
	OaString Uri;
	OaString Path;

	OaU32 MaxDpbSlots = 16;

	// When true, the stream wraps to t=0 on EOS so playback runs forever.
	// When false, IsDone() flips to true once the final packet is decoded.
	bool Loop = true;

	// Prefer the hardware YCbCr sampler path. The compute conversion remains
	// available as the compatibility and exact-control fallback.
	bool PreferHardwareYCbCr = true;

	// 0 = use FrameRate from the container (defaults to 30 if absent).
	OaF32 FrameRateOverride = 0.0F;

	// Start in the playing state. When false, the app must explicitly
	// call Play() or StepForward() to advance.
	bool StartPlaying = true;
	// Open and synchronize the first audio track when one is present. Failure
	// to find an audio track preserves video-only operation.
	bool Audio = true;
	OaVideoStreamOptions StreamOptions = {};

	// Reorder buffer depth (in decoded frames) used to convert decode order
	// → display order. H.264/H.265 streams with B-frames emit packets in
	// decode order (I P B B P B B …) but must be displayed in PTS order
	// (I B B P B B P …). Without this buffer, B-frame streams play as
	// "two forward, one back". 4 is enough for typical IBBBP GOPs; bump
	// higher for unusual encodes. Set to 0 to disable reordering (correct
	// only when the stream has no B-frames).
	OaU32 ReorderDepth = 4;

	// Conversion filter: Nearest = sharp pixel edges, no smoothing.
	// Linear = smoother but softer output. Default Nearest for video
	// to preserve decoded-frame sharpness.
	OaFilter Filter = OaFilter::Nearest;
};

class OaVideo {
public:
	[[nodiscard]] static OaResult<OaVideo> Open(
		OaEngine& InEngine,
		const OaVideoConfig& InConfig);
	[[nodiscard]] static OaResult<OaVideo> Create(
		OaEngine& InEngine,
		const OaVideoConfig& InCfg);

	OaVideo() = default;
	OaVideo(OaVideo&&) noexcept;
	OaVideo& operator=(OaVideo&&) noexcept;
	OaVideo(const OaVideo&)            = delete;
	OaVideo& operator=(const OaVideo&) = delete;
	~OaVideo();

	void Destroy();

	// IsDone() is permanently false in looping mode. In non-looping mode,
	// flips to true once the underlying stream's EOS has been observed and
	// the last decoded frame consumed.
	[[nodiscard]] bool IsDone() const;
	// Advances by exactly one decoded frame (ignores Playing_, ignores the
	// wall-clock accumulator). Use Tick() for time-paced playback.
	// Advance to the next display-order frame. Open() presents the first frame,
	// so the initial CurrentFrame() is immediately usable. Next() reports
	// decode, demux and synchronization failures instead of hiding them.
	[[nodiscard]] OaStatus Next();
	void Reset();
	// Returns the number of frames decoded since the most recent rewind.
	[[nodiscard]] OaI64 Index() const { return Index_; }

	// ─── Playback control ────────────────────────────────────────────────
	void Play();
	void Pause();
	void TogglePlay();
	[[nodiscard]] bool IsPlaying() const { return Playing_; }
	[[nodiscard]] bool HasAudio() const { return Audio_.HasValue(); }

	// Decode the next packet immediately (independent of play state). Loops
	// to t=0 if the stream is at EOS and Cfg.Loop is true.
	// Compatibility/playback spelling for Next().
	OaStatus StepForward();

	// Seek to the previous keyframe and re-decode forward until one frame
	// before the current position. Costs up to GOP-size frames of decode.
	OaStatus StepBackward();

	// Scrub by a signed number of display-order frames. Positive movement
	// advances normally; negative movement seeks to the preceding keyframe
	// and replays through the reorder/presentation path.
	OaStatus StepFrames(OaI32 InFrameDelta);
	// Seek in the container video-track timebase. The decoder is recreated at
	// the preceding keyframe and replayed to the first display frame at/after
	// the requested timestamp.
	OaStatus Seek(OaU64 InTimestamp);
	// Drain outstanding GPU work and clear queued display frames.
	OaStatus Flush();

	// Advance by wall clock when playing. When paused, accumulates nothing
	// (no implicit frame skipping after un-pause).
	void Tick(OaF32 InDeltaMs);

	// Most recently decoded frame. ImageView is VK_NULL_HANDLE until the
	// first successful decode.
	[[nodiscard]] const OaVideoFrame& CurrentFrame() const { return Frame_; }
	// Convert the current decoder-owned frame to an ML tensor. Conversion lives
	// on OaVideo rather than OaVideoFrame because the decoder session owns the
	// image views, YCbCr metadata, synchronization and conversion resources.
	[[nodiscard]] OaResult<OaMatrix> CurrentFrameToMatrix(
		bool InNormalizeImageNet = true);
	[[nodiscard]] OaResult<OaImage> CurrentFrameToImage(
		bool InNormalizeImageNet = true);
	// Read back the current converted RGBA frame. This is intended for CPU
	// reference overlays and diagnostics; realtime inference should consume
	// CurrentFrame() directly on the GPU.
	[[nodiscard]] OaResult<OaVec<OaU8>> ReadbackCurrentRgba();
	// Record the compute submission that most recently sampled CurrentFrame().
	// The iterator will not recycle that RGBA image until the timeline reaches
	// InValue. Frames advanced without being rendered remain immediately
	// reusable.
	void MarkCurrentFrameConsumed(
		const OaVkTimelineSemaphore& InSemaphore,
		OaU64 InValue);

	// ─── Stream metadata passthroughs ────────────────────────────────────
	[[nodiscard]] OaU32 Width() const;
	[[nodiscard]] OaU32 Height() const;
	[[nodiscard]] OaU32 FrameRate() const;
	[[nodiscard]] OaUsize FrameCount() const;
	[[nodiscard]] OaF32 FrameIntervalMs() const { return FrameIntervalMs_; }
	[[nodiscard]] bool  IsEos() const;
	[[nodiscard]] const OaContainerInfo& GetContainerInfo() const;
	[[nodiscard]] const OaVideoStreamStats& GetStreamStats() const;

private:
	// Each reorder buffer entry owns its own RGBA target image. We convert
	// NV12→RGBA at decode time so the DPB layer is free to be reused by the
	// next decode (the H.264 sliding-window otherwise corrupts data we'd
	// still be holding by reference).
	struct ReorderEntry {
		OaVideoFrame Rgba = {};
		OaU64        Pts  = 0;

		ReorderEntry() = default;
		ReorderEntry(const OaVideoFrame& InRgba, OaU64 InPts)
			: Rgba(InRgba)
			, Pts(InPts)
		{}
		ReorderEntry(const ReorderEntry&) = delete;
		ReorderEntry& operator=(const ReorderEntry&) = delete;
		ReorderEntry(ReorderEntry&&) noexcept = default;
		ReorderEntry& operator=(ReorderEntry&&) noexcept = default;
	};

	OaStatus DecodeOneIntoReorder_();
	OaStatus FillReorderBuffer_();
	OaStatus PopAndPresentLowestPts_();
	OaStatus SeekDisplayFrame_(OaUsize InTargetFrameIndex);
	OaStatus ClearReorder_();
	OaStatus WaitForCurrentFrameConsumer_();
	OaStatus RestartDecoder_();
	// Pool helpers — produce/release an RGBA target sized to the stream.
	[[nodiscard]] OaResult<OaVideoFrame> AcquireRgbaFromPool_();
	void ReleaseRgbaToPool_(const OaVideoFrame& InFrame);

	OaVideoConfig          Cfg_;
	OaEngine*                Engine_ = nullptr;
	OaOption<OaVideoStream>  Stream_;
	OaOption<OaVideoDecoder> Decoder_;
	OaOption<OaAudioStream>  Audio_;
	OaVideoFrame             Frame_           = {};
	OaF32                    FrameIntervalMs_ = 1000.0F / 30.0F;
	OaF32                    Accumulator_     = 0.0F;
	bool                     Playing_         = true;
	bool                     ReachedEos_      = false;
	bool                     StreamEosCurrent_ = false;
	OaVec<ReorderEntry>      Reorder_;
	OaVec<OaU64>             DisplayPts_;
	// Iterator-owned pool of RGBA targets. Sized lazily to ReorderDepth + 2
	// (held in reorder + currently displayed + one slack). The decoder owns
	// the actual VkImage lifetimes via its RgbImages_ table; we just track
	// who's holding which.
	OaVec<OaVideoFrame>      RgbaPool_;
	OaVec<bool>              RgbaPoolBusy_;
	OaVec<OaVkTimelineSemaphore> RgbaPoolConsumerSemaphores_;
	OaVec<OaU64>             RgbaPoolConsumerValues_;
	OaI64                    Index_ = 0;
	OaU64                    StreamFormatGeneration_ = 1U;
	OaU64                    StreamReconnectCount_ = 0U;
};
