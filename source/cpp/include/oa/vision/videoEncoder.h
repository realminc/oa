// oa::VideoEncoder — hardware Video encoder
// VK_KHR_video_encode_queue + VK_KHR_video_encode_h264 / h265
// Zero-copy: vkImage (NV12) → compressed bitstream
//
// Mirrors oa::VideoDecoder's shape (source/cpp/include/oa/vision/videoDecoder.h):
//   queryEncodeCapabilities → create → encode → flush → close.
// The encoder MANUFACTURES the H.264 SPS/PPS at session-parameter creation
// time (the decoder parses them from the bitstream); see videoEncoderCodec.cpp
// once the codec implementation lands.
// session schema: Tools/FnAutogen/Schema/Vision/VisionVideoEncoder.toml.

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/runtime/oaVk.h>
#include <oa/vision/videoDecoder.h>

namespace oa { class Texture; }
namespace oavk { class Buffer; }

namespace oa {

struct VideoEncoderAccess;

enum class VideoRateControl : oa::U8 {
	ConstantQp = 0,
	Cbr        = 1,
	Vbr        = 2,
};

// Video encoding profile — what the user asks for.
struct VideoEncodeProfile {
	VideoCodec       codec          = VideoCodec::H264;
	oa::U32            width          = 0;
	oa::U32            height         = 0;
	VideoRateControl rateControl    = VideoRateControl::ConstantQp;
	// target average bitrate for CBR/VBR. Ignored by ConstantQp.
	oa::U32 bitrate                   = 4'000'000U;
	// VBR peak bitrate. Zero selects 2 * bitrate, clamped to device limits.
	// CBR always uses bitrate for both average and peak.
	oa::U32 maxBitrate                = 0U;
	// Per-slice QP used by constantQp (H.264 valid range is 0..51).
	oa::U32 constantQp                = 26U;
	oa::U32 frameRate                 = 30U;           // FPS
	oa::U32 gopSize                   = 30U;           // Keyframe interval (frames)
	oa::U32 maxBFrames                = 0U;            // Max consecutive B-frames
	oa::U32 maxDpbSlots               = 0U;            // 0 = pick from caps
	oa::U32 qualityLevel              = 0U;            // vulkan encode quality-level index
	// Number of independent compute/encode jobs kept in flight. One preserves
	// synchronous behavior; three is the live-recording default.
	oa::U32 asyncDepth                = 3U;
};


// Encoded frame output — bitstream + PTS + key/keyflag.
struct EncodedVideoPacket {
	oa::Vec<oa::U8> bitstream;
	oa::U64       presentationTimestamp = 0U;  // PTS in microseconds
	bool        isKeyframe            = false;
	oa::U32       frameSize             = 0U;  // Bytes in bitstream
};


// codec/profile-specific vulkan Video encode capabilities.
// Mirrors VideoDecodeCapabilities.
struct VideoEncodeCapabilities {
	bool  supported                           = false;
	oa::U32 maxWidth                            = 0;
	oa::U32 maxHeight                           = 0;
	oa::U32 minWidth                            = 0;
	oa::U32 minHeight                           = 0;
	oa::U32 pictureAccessGranularityWidth       = 1;
	oa::U32 pictureAccessGranularityHeight      = 1;
	oa::U32 maxDpbSlots                         = 0;
	oa::U32 maxActiveReferencePictures          = 0;
	oa::U32 maxBitrate                          = 0;   // bits/sec (0 = unknown / unlimited)
	oa::U32 maxQualityLevels                    = 1;
	oa::U64 minBitstreamBufferOffsetAlignment   = 0;
	oa::U64 minBitstreamBufferSizeAlignment     = 0;

	// Rate-control modes exposed by VK_KHR_video_encode_queue:
	//   VK_VIDEO_ENCODE_RATE_CONTROL_MODE_*_BIT_KHR
	VkVideoEncodeRateControlModeFlagsKHR rateControlModes = 0;

	// H.264-specific (zero when codec != H264)
	oa::U32 maxH264SliceCount               = 0;
	oa::U32 maxH264PPictureL0ReferenceCount = 0;
	oa::U32 maxH264BPictureL0ReferenceCount = 0;
	oa::U32 maxH264L1ReferenceCount         = 0;

	// H.265-specific (zero when codec != H265)
	oa::U32 maxH265SliceSegmentCount            = 0;
	oa::U32 maxH265PPictureL0ReferenceCount     = 0;
	oa::U32 maxH265BPictureL0ReferenceCount     = 0;
	oa::U32 maxH265L1ReferenceCount             = 0;
	VkVideoEncodeH265CtbSizeFlagsKHR            h265CtbSizes            = 0;
	VkVideoEncodeH265TransformBlockSizeFlagsKHR h265TransformBlockSizes = 0;
	VkVideoEncodeH265StdFlagsKHR                h265StdSyntaxFlags      = 0;
	oa::I32 minH265Qp                           = 0;
	oa::I32 maxH265Qp                           = 0;

	// input picture format the encoder consumes (NV12 by default).
	VkFormat pictureFormat         = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
	// Reference picture format the encoder writes into the DPB.
	VkFormat referencePictureFormat = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;

	VkExtensionProperties               stdHeaderVersion = {};
	VkVideoEncodeCapabilityFlagsKHR     encodeFlags      = 0;

	oa::Vec<VkVideoFormatPropertiesKHR> inputFormats;
	oa::Vec<VkVideoFormatPropertiesKHR> dpbFormats;
};


// hardware video encoder session — wraps VkVideoSessionKHR for encoding.
class VideoEncoder {
	friend class VideoTranscoder;
	friend struct VideoEncoderAccess;

public:
	VideoEncoder();
	VideoEncoder(VideoEncoder&& inOther) noexcept;
	VideoEncoder& operator=(VideoEncoder&& inOther) noexcept;
	VideoEncoder(const VideoEncoder&) = delete;
	VideoEncoder& operator=(const VideoEncoder&) = delete;
	~VideoEncoder();

	// query what this device's encode side can do for inCodec. Returns a
	// populated VideoEncodeCapabilities or an error if the codec is
	// unsupported / disabled.
	static oa::Result<VideoEncodeCapabilities> queryEncodeCapabilities(class Engine& inRt, VideoCodec inCodec);

	// Create encoder for specific codec and settings.
	static oa::Result<VideoEncoder> create(class Engine& inRt,	const VideoEncodeProfile& inProfile);

	[[nodiscard]] Engine* getEngine() const noexcept;

	// convert a buffer-backed RGBA texture and encode one frame through this
	// session's owning engine. pending compute producers are completed before
	// the encoder snapshots the source allocation.
	[[nodiscard]] oa::Status encode(
		const oa::Texture& inRgba,
		EncodedVideoPacket& outFrame,
		oa::U64 inPts = 0ULL
	);

	// flush encoder (get remaining buffered frames — B-frame reordering).
	oa::Status flush(oa::Vec<EncodedVideoPacket>& outFrames);

	// Explicit completion and resource-release boundary. pending encoded output
	// is discarded; call flush() first when the packets are required.
	[[nodiscard]] oa::Status close();

	[[nodiscard]] const VideoEncodeProfile& getProfile() const noexcept;
	[[nodiscard]] oa::U32 getCodedWidth() const noexcept;
	[[nodiscard]] oa::U32 getCodedHeight() const noexcept;

private:
	// Physical buffer/image routes. Public callers use encode(); media sessions
	// use VideoEncoderAccess to preserve pipelined recording and zero-copy
	// transcode paths without exporting vulkan resources as the semantic API.
	oa::Status uploadInputRgba(
		const oavk::Buffer& inRgba,
		oa::U32 inVisibleWidth,
		oa::U32 inVisibleHeight,
		YCbCrModel inColorSpace = YCbCrModel::BT709,
		bool inFullRange = false
	);
	oa::Status encodeFrame(
		VkImage inImage,
		oa::U64 inPts,
		EncodedVideoPacket& outFrame
	);
	oa::Status submitRgba(
		const oavk::Buffer& inRgba,
		oa::U32 inVisibleWidth,
		oa::U32 inVisibleHeight,
		oa::U64 inPts,
		oa::Vec<EncodedVideoPacket>& outReady,
		YCbCrModel inColorSpace = YCbCrModel::BT709,
		bool inFullRange = false
	);
	oa::Status submitRgbaImage(
		VkImage inImage,
		VkImageView inImageView,
		VkFormat inFormat,
		VkImageLayout inLayout,
		oa::U32 inVisibleWidth,
		oa::U32 inVisibleHeight,
		oa::U64 inPts,
		oa::Vec<EncodedVideoPacket>& outReady,
		YCbCrModel inColorSpace = YCbCrModel::BT709,
		bool inFullRange = false,
		oa::U32 inArrayLayer = 0U,
		oa::Event inReady = {},
		oa::U32 inExternalQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		oa::Event* outInputConsumed = nullptr
	);

	struct EncodeSlot;

	void     reset_() noexcept;
	oa::Status uploadInputRgba_(EncodeSlot& inSlot, const oavk::Buffer& inRgba,
		oa::U32 inVisibleWidth, oa::U32 inVisibleHeight,
		YCbCrModel inColorSpace, bool inFullRange
	);
	oa::Status uploadInputRgbaImage_(EncodeSlot& inSlot,
		VkImage inImage, VkImageView inImageView, VkFormat inFormat,
		VkImageLayout inLayout, oa::U32 inVisibleWidth, oa::U32 inVisibleHeight,
		YCbCrModel inColorSpace, bool inFullRange, oa::U32 inArrayLayer,
		oa::Event inReady,
		oa::U32 inExternalQueueFamilyIndex
	);
	oa::Status submitEncode_(EncodeSlot& inSlot, oa::U64 inPts);
	oa::Status harvest_(EncodeSlot& inSlot, bool inWait, EncodedVideoPacket& outFrame, bool& outReady);
	[[nodiscard]] oa::Status destroySlot_(EncodeSlot& inSlot);
	void abandon_() noexcept;
	static oa::Status completeRetired_(void* inPayload);
	static void releaseRetired_(void* inPayload);

	class Impl;
	oa::UniquePtr<Impl> impl_;
};


// Synchronous frame transcoder. The source packet is decoded on the vulkan
// video queue, converted to RGBA on compute, then submitted to the vulkan
// encoder without a CPU pixel round-trip. Container demux/mux remains the
// caller's responsibility.
class VideoTranscoder {
public:
	static oa::Result<VideoTranscoder> create(
		Engine& inRt,
		const VideoProfile& inDecodeProfile,
		const VideoEncodeProfile& inEncodeProfile);

	oa::Status transcodeFrame(
		const oa::Span<const oa::U8>& inBitstream,
		EncodedVideoPacket& outFrame);

	// Completes both codec sessions and reports the first shutdown failure.
	[[nodiscard]] oa::Status close();

private:
	VideoTranscoder() = default;

	VideoDecoder  decoder_;
	VideoEncoder  encoder_;
	Engine*       rt_              = nullptr;
	oa::U64         nextPtsUs_       = 0U;
	oa::U64         frameDurationUs_ = 0U;
};

} // namespace oa
