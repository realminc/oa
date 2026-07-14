// OA Vision — Hardware Video Encoder
// VK_KHR_video_encode_queue + VK_KHR_video_encode_h264 / h265
// Zero-copy: VkImage (NV12) → compressed bitstream
//
// Mirrors OaVideoDecoder's shape (Source/Public/Oa/Vision/VideoDecoder.h):
//   QueryEncodeCapabilities → Create → EncodeFrame → Flush → Destroy.
// The encoder MANUFACTURES the H.264 SPS/PPS at session-parameter creation
// time (the decoder parses them from the bitstream); see VideoEncoderCodec.cpp
// once the codec implementation lands.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Runtime/OaVkVideo.h>
#include <Oa/Runtime/ImageDispatch.h>
#include <Oa/Vision/VideoDecoder.h>


enum class OaVideoRateControl : OaU8 {
	ConstantQp = 0,
	Cbr = 1,
	Vbr = 2,
};

// Video encoding profile — what the user asks for.
struct OaVideoEncodeProfile
{
	OaVideoCodec Codec      = OaVideoCodec::H264;
	OaU32 Width             = 0;
	OaU32 Height            = 0;
	OaVideoRateControl RateControl = OaVideoRateControl::ConstantQp;
	// Target average bitrate for CBR/VBR. Ignored by ConstantQp.
	OaU32 Bitrate           = 4'000'000U;
	// VBR peak bitrate. Zero selects 2 * Bitrate, clamped to device limits.
	// CBR always uses Bitrate for both average and peak.
	OaU32 MaxBitrate        = 0U;
	// Per-slice QP used by ConstantQp (H.264 valid range is 0..51).
	OaU32 ConstantQp        = 26U;
	OaU32 FrameRate         = 30U;           // FPS
	OaU32 GopSize           = 30U;           // Keyframe interval (frames)
	OaU32 MaxBFrames        = 0U;            // Max consecutive B-frames
	OaU32 MaxDpbSlots       = 0U;            // 0 = pick from caps
	OaU32 QualityLevel      = 0U;            // Vulkan encode quality-level index
	// Number of independent compute/encode jobs kept in flight. One preserves
	// synchronous behavior; three is the live-recording default.
	OaU32 AsyncDepth        = 3U;
};


// Encoded frame output — bitstream + PTS + key/keyflag.
struct OaEncodedFrame
{
	OaVec<OaU8> Bitstream;
	OaU64 PresentationTimestamp = 0U;        // PTS in microseconds
	bool  IsKeyframe            = false;
	OaU32 FrameSize             = 0U;        // Bytes in Bitstream
};


// Codec/profile-specific Vulkan Video encode capabilities.
// Mirrors OaVideoDecodeCapabilities.
struct OaVideoEncodeCapabilities {
	bool  Supported                           = false;
	OaU32 MaxWidth                            = 0;
	OaU32 MaxHeight                           = 0;
	OaU32 MinWidth                            = 0;
	OaU32 MinHeight                           = 0;
	OaU32 PictureAccessGranularityWidth       = 1;
	OaU32 PictureAccessGranularityHeight      = 1;
	OaU32 MaxDpbSlots                         = 0;
	OaU32 MaxActiveReferencePictures          = 0;
	OaU32 MaxBitrate                          = 0;            // bits/sec (0 = unknown / unlimited)
	OaU32 MaxQualityLevels                    = 1;
	OaU64 MinBitstreamBufferOffsetAlignment   = 0;
	OaU64 MinBitstreamBufferSizeAlignment     = 0;

	// Rate-control modes exposed by VK_KHR_video_encode_queue:
	//   VK_VIDEO_ENCODE_RATE_CONTROL_MODE_*_BIT_KHR
	VkVideoEncodeRateControlModeFlagsKHR RateControlModes = 0;

	// H.264-specific (zero when codec != H264)
	OaU32 MaxH264SliceCount                   = 0;
	OaU32 MaxH264PPictureL0ReferenceCount     = 0;
	OaU32 MaxH264BPictureL0ReferenceCount     = 0;
	OaU32 MaxH264L1ReferenceCount             = 0;

	// H.265-specific (zero when codec != H265)
	OaU32 MaxH265SliceSegmentCount            = 0;
	OaU32 MaxH265PPictureL0ReferenceCount     = 0;
	OaU32 MaxH265BPictureL0ReferenceCount     = 0;
	OaU32 MaxH265L1ReferenceCount             = 0;
	VkVideoEncodeH265CtbSizeFlagsKHR H265CtbSizes = 0;
	VkVideoEncodeH265TransformBlockSizeFlagsKHR H265TransformBlockSizes = 0;
	VkVideoEncodeH265StdFlagsKHR H265StdSyntaxFlags = 0;
	OaI32 MinH265Qp                           = 0;
	OaI32 MaxH265Qp                           = 0;

	// Input picture format the encoder consumes (NV12 by default).
	VkFormat PictureFormat                    = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
	// Reference picture format the encoder writes into the DPB.
	VkFormat ReferencePictureFormat           = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;

	VkExtensionProperties StdHeaderVersion    = {};
	VkVideoEncodeCapabilityFlagsKHR EncodeFlags = 0;

	OaVec<VkVideoFormatPropertiesKHR> InputFormats;
	OaVec<VkVideoFormatPropertiesKHR> DpbFormats;
};


// Hardware video encoder session — wraps VkVideoSessionKHR for encoding.
class OaVideoEncoder
{
	friend class OaVideoTranscoder;
public:
	OaVideoEncoder() = default;
	OaVideoEncoder(OaVideoEncoder&& InOther) noexcept;
	OaVideoEncoder& operator=(OaVideoEncoder&& InOther) noexcept;
	OaVideoEncoder(const OaVideoEncoder&) = delete;
	OaVideoEncoder& operator=(const OaVideoEncoder&) = delete;
	~OaVideoEncoder();

	// Query what this device's encode side can do for InCodec. Returns a
	// populated OaVideoEncodeCapabilities or an error if the codec is
	// unsupported / disabled.
	static OaResult<OaVideoEncodeCapabilities> QueryEncodeCapabilities(
		class OaEngine& InRt,
		OaVideoCodec InCodec);

	// Convenience caps queries (call QueryEncodeCapabilities internally).
	static bool  IsCodecSupported(OaEngine& InRt, OaVideoCodec InCodec);
	static OaU32 GetMaxBitrate(OaEngine& InRt, OaVideoCodec InCodec);
	static OaU32 GetMaxWidth(OaEngine& InRt, OaVideoCodec InCodec);
	static OaU32 GetMaxHeight(OaEngine& InRt, OaVideoCodec InCodec);

	// Create encoder for specific codec and settings.
	static OaResult<OaVideoEncoder> Create(
		class OaEngine& InRt,
		const OaVideoEncodeProfile& InProfile);

	// Convert a packed RGBA8 source buffer into the encoder's NV12
	// input image via the CvtRgbaToNv12 compute shader (3g.2). InRgba
	// is the source's OaVkBuffer (typically OaTexture::DeviceBuf —
	// FromPixels and LoadFile both produce buffer-backed textures).
	// On success, compatibility slot 0 holds the converted picture and is
	// ready for the next synchronous EncodeFrame call.
	OaStatus UploadInputRgba(
		const OaVkBuffer& InRgba,
		OaU32 InVisibleWidth,
		OaU32 InVisibleHeight,
		OaYCbCrModel InColorSpace = OaYCbCrModel::BT709,
		bool   InFullRange        = false);

	// Encode one frame from VkImage (NV12 format).
	// Input image typically comes from RGBA→NV12 compute shader output.
	OaStatus EncodeFrame(
		VkImage InImage,
		OaU64 InPts,
		OaEncodedFrame& OutFrame);

	// Submit RGBA conversion + encode without a per-frame host wait. Completed
	// frames are returned in presentation order through OutReady. When the ring
	// wraps, only the oldest slot is waited and harvested.
	OaStatus SubmitRgba(
		const OaVkBuffer& InRgba,
		OaU32 InVisibleWidth,
		OaU32 InVisibleHeight,
		OaU64 InPts,
		OaVec<OaEncodedFrame>& OutReady,
		OaYCbCrModel InColorSpace = OaYCbCrModel::BT709,
		bool InFullRange = false);

	// Convert an image-backed RGBA/BGRA frame directly into the slot's NV12
	// encode image. The source is never mapped or copied through the CPU.
	// This call waits only for the conversion before returning, which makes the
	// caller-owned image safe to recycle immediately; video encoding remains
	// pipelined in the asynchronous job ring.
	OaStatus SubmitRgbaImage(
		VkImage InImage,
		VkImageView InImageView,
		VkFormat InFormat,
		VkImageLayout InLayout,
		OaU32 InVisibleWidth,
		OaU32 InVisibleHeight,
		OaU64 InPts,
		OaVec<OaEncodedFrame>& OutReady,
		OaYCbCrModel InColorSpace = OaYCbCrModel::BT709,
		bool InFullRange = false,
		OaU32 InArrayLayer = 0U,
		const OaVkTimelineSemaphore* InReadySemaphore = nullptr,
		OaU64 InReadyValue = 0U,
		OaU32 InExternalQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		OaCompletionToken* OutInputConsumed = nullptr);

	// Flush encoder (get remaining buffered frames — B-frame reordering).
	OaStatus Flush(OaVec<OaEncodedFrame>& OutFrames);

	// Release all resources.
	void Destroy();

	[[nodiscard]] const OaVideoEncodeProfile& GetProfile() const noexcept { return Profile_; }
	[[nodiscard]] OaU32 GetCodedWidth()  const noexcept { return CodedWidth_; }
	[[nodiscard]] OaU32 GetCodedHeight() const noexcept { return CodedHeight_; }

private:
	struct EncodeSlot {
		// Stable packed-RGBA snapshot. SubmitRgba owns the caller data before
		// returning so the producer may immediately recycle its buffer.
		OaVkBuffer RgbaSnapshot;
		VkImage InputImage = VK_NULL_HANDLE;
		VkImageView InputView = VK_NULL_HANDLE;
		VkImageView InputYView = VK_NULL_HANDLE;
		VkImageView InputUvView = VK_NULL_HANDLE;
		OaU32 InputYBindless = 0U;
		OaU32 InputUvBindless = 0U;
		bool InputBindlessRegistered = false;
		void* InputAllocation = nullptr;
		OaVkImageDispatchTicket InputTicket;
		bool InputInitialized = false;
		OaVkVideoBitstream Bitstream;
		OaU64 BitstreamDirtyEnd = 0U;
		VkCommandBuffer CommandBuffer = VK_NULL_HANDLE;
		VkFence Fence = VK_NULL_HANDLE;
		VkQueryPool QueryPool = VK_NULL_HANDLE;
		OaU64 PresentationTimestamp = 0U;
		bool IsKeyframe = false;
		bool Pending = false;

		EncodeSlot() = default;
		EncodeSlot(EncodeSlot&&) noexcept = default;
		EncodeSlot& operator=(EncodeSlot&&) noexcept = default;
		EncodeSlot(const EncodeSlot&) = delete;
		EncodeSlot& operator=(const EncodeSlot&) = delete;
	};

	void  Reset_() noexcept;
	OaStatus UploadInputRgba_(EncodeSlot& InSlot, const OaVkBuffer& InRgba,
		OaU32 InVisibleWidth, OaU32 InVisibleHeight,
		OaYCbCrModel InColorSpace, bool InFullRange);
	OaStatus UploadInputRgbaImage_(EncodeSlot& InSlot,
		VkImage InImage, VkImageView InImageView, VkFormat InFormat,
		VkImageLayout InLayout, OaU32 InVisibleWidth, OaU32 InVisibleHeight,
		OaYCbCrModel InColorSpace, bool InFullRange, OaU32 InArrayLayer,
		const OaVkTimelineSemaphore* InReadySemaphore, OaU64 InReadyValue,
		OaU32 InExternalQueueFamilyIndex);
	OaStatus SubmitEncode_(EncodeSlot& InSlot, OaU64 InPts);
	OaStatus Harvest_(EncodeSlot& InSlot, bool InWait,
		OaEncodedFrame& OutFrame, bool& OutReady);
	void DestroySlot_(EncodeSlot& InSlot) noexcept;

	OaVideoEncodeProfile Profile_                       = {};
	class OaEngine* Rt_                                 = nullptr;

	// Vulkan video session + parameters (Layer 1: OaVkVideo)
	OaVkVideoSession Session_;
	OaVkVideoParameters SessionParams_;
	OaVkVideoQueue Queue_;

	// DPB (Decoded Picture Buffer — also used for encode reference frames).
	OaVkVideoDpb Dpb_;
	OaU32 DpbSlotCapacity_ = 0;

	OaVec<EncodeSlot> Slots_;
	OaU32 SubmitSlot_ = 0U;
	OaU32 HarvestSlot_ = 0U;
	OaU32 PendingSlots_ = 0U;
	bool CompatibilityUploadReady_ = false;

	// Cached SPS+PPS NAL bytes — vkGetEncodedVideoSessionParametersKHR
	// returns these once after session params create; we prepend them
	// to the bitstream on every IDR frame so the output is self-decodable
	// without an external sidecar.
	OaVec<OaU8> CachedHeaders_;

	// Fence + feedback query pool for per-encode synchronisation and
	// bitstream-size readback.
	// Tracks whether we've issued the first CodingControlReset +
	// rate-control setup on this session (per-session, not per-frame).
	bool RateControlReset_                              = false;
	bool QueryResultStatusSupported_                   = false;

	// Vulkan mode selected from the public ConstantQp / CBR / VBR contract.
	// ConstantQp maps to DISABLED and carries QP in each codec slice.
	VkVideoEncodeRateControlModeFlagBitsKHR RateControlMode_ =
		VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR;

	// Aligned coded extent per encode-caps granularity.
	OaU32 CodedWidth_                                   = 0;
	OaU32 CodedHeight_                                  = 0;
	OaU64 MinBitstreamBufferOffsetAlignment_            = 1;
	OaU64 MinBitstreamBufferSizeAlignment_              = 1;

	// Per-frame state.
	OaU32 FrameCount_                                   = 0;     // Total frames encoded
	OaU32 LastKeyframeIndex_                            = 0;     // Frame index of most recent IDR
	
	// GOP structure for P-frame encoding (Phase D)
	OaU32 GopSize_                                      = 30;    // IDR interval (default: 30 frames)
	OaU32 CurrentGopFrame_                              = 0;     // Frame index within current GOP
};


// Synchronous frame transcoder. The source packet is decoded on the Vulkan
// video queue, converted to RGBA on compute, then submitted to the Vulkan
// encoder without a CPU pixel round-trip. Container demux/mux remains the
// caller's responsibility.
class OaVideoTranscoder
{
public:
	static OaResult<OaVideoTranscoder> Create(
		OaEngine& InRt,
		const OaVideoProfile& InDecodeProfile,
		const OaVideoEncodeProfile& InEncodeProfile);

	OaStatus TranscodeFrame(
		const OaSpan<const OaU8>& InBitstream,
		OaEncodedFrame& OutFrame);

	void Destroy();

private:
	OaVideoTranscoder() = default;

	OaVideoDecoder Decoder_;
	OaVideoEncoder Encoder_;
	OaEngine* Rt_ = nullptr;
	OaU64 NextPtsUs_ = 0U;
	OaU64 FrameDurationUs_ = 0U;
};
