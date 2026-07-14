// OA Vision — composed hardware video recorder.
//
// OaVideoRecorder is the file sink counterpart to OaVideo/capture sources:
// it owns one Vulkan Video encoder and one container muxer, accepts the common
// OaVideoFrame contract, and finalizes a playable file. Codec work remains in
// OaVideoEncoder; container work remains in OaVideoMuxer.

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Vision/VideoDecoder.h>
#include <Oa/Vision/VideoEncoder.h>
#include <Oa/Vision/VideoMuxer.h>
#include <Oa/Audio/AudioCapture.h>

class OaEngine;
class OaVkBuffer;
struct OaTexture;

struct OaVideoRecorderConfig {
	OaString OutputPath = "output.mp4";
	OaVideoEncodeProfile Encode = {};
	OaYCbCrModel ColorSpace = OaYCbCrModel::BT709;
	bool FullRange = false;
	bool AudioEnabled = false;
	OaAudioEncodeProfile Audio = {};
};

class OaVideoRecorder {
public:
	OaVideoRecorder() = default;
	OaVideoRecorder(OaVideoRecorder&& InOther) noexcept;
	OaVideoRecorder& operator=(OaVideoRecorder&& InOther) noexcept;
	OaVideoRecorder(const OaVideoRecorder&) = delete;
	OaVideoRecorder& operator=(const OaVideoRecorder&) = delete;
	~OaVideoRecorder();

	[[nodiscard]] static OaResult<OaVideoRecorder> Create(
		OaEngine& InEngine,
		const OaVideoRecorderConfig& InConfig);

	// Record one packed RGBA8 bindless buffer.
	[[nodiscard]] OaStatus WriteRgba(
		const OaVkBuffer& InRgba,
		OaU32 InWidth,
		OaU32 InHeight,
		OaU64 InPts);

	// Record a common capture/decode/render frame. Packed RGBA8/BGRA8 buffers
	// and images are supported; image readiness stays on the GPU timeline.
	[[nodiscard]] OaStatus Write(const OaVideoFrame& InFrame);
	// Non-blocking image-input variant. OutInputConsumed signals after the
	// source image has returned to its published layout/queue family and may be
	// recycled. Buffer-backed frames produce an already-complete empty token.
	[[nodiscard]] OaStatus WriteAsync(
		const OaVideoFrame& InFrame,
		OaCompletionToken& OutInputConsumed);

	// Record a buffer- or image-backed render target. Image-backed targets use
	// the same OaVideoFrame path as decoded/captured frames and never stage
	// pixels through host memory.
	[[nodiscard]] OaStatus Write(const OaTexture& InTexture, OaU64 InPts);

	// Add captured interleaved F32 audio. Timestamps share the monotonic
	// microsecond clock used by OaScreenCapture. The recorder trims overlap and
	// inserts silence for gaps so both tracks start at the first video frame.
	[[nodiscard]] OaStatus WriteAudio(
		OaSpan<const OaF32> InInterleaved,
		OaU32 InSampleRate,
		OaU32 InChannelCount,
		OaU64 InPts);
	[[nodiscard]] OaStatus WriteAudio(const OaAudioCaptureChunk& InChunk);

	// Flush the encoder and finalize the container. Idempotent.
	[[nodiscard]] OaStatus Finalize();
	void Destroy();

	[[nodiscard]] bool IsOpen() const noexcept { return Engine_ != nullptr and not Finalized_; }
	[[nodiscard]] OaU32 GetFrameCount() const noexcept { return SubmittedFrameCount_; }
	[[nodiscard]] const OaVideoRecorderConfig& GetConfig() const noexcept { return Config_; }

private:
	void MoveFrom_(OaVideoRecorder&& InOther) noexcept;
	[[nodiscard]] OaStatus WriteEncoded_(const OaEncodedFrame& InFrame);
	[[nodiscard]] OaStatus WriteAudioAligned_(
		OaSpan<const OaF32> InInterleaved, OaU64 InPts);
	[[nodiscard]] OaStatus WriteAudioPackets_(OaVec<OaEncodedAudioPacket>& InPackets);
	[[nodiscard]] OaStatus SetFirstVideoPts_(OaU64 InPts);

	OaEngine* Engine_ = nullptr;
	OaVideoRecorderConfig Config_ = {};
	OaVideoEncoder Encoder_;
	OaVideoMuxer Muxer_;
	OaAudioStreamEncoder AudioEncoder_;
	struct PendingAudioChunk {
		OaVec<OaF32> Samples;
		OaU64 Pts = 0U;
	};
	OaVec<PendingAudioChunk> PendingAudio_;
	OaVec<OaF32> AudioScratch_;
	OaU64 FirstVideoPts_ = 0U;
	OaU64 NextAudioFrame_ = 0U;
	bool HasFirstVideoPts_ = false;
	OaU32 SubmittedFrameCount_ = 0;
	OaU32 MuxedFrameCount_ = 0;
	bool CodecConfigWritten_ = false;
	bool Finalized_ = false;
};
