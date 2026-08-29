// oa::VideoDecoder — hardware Video Decoder
// VK_KHR_video_decode_h264 / h265 / av1 / vp9
// Zero-copy: compressed bitstream → native 4:2:0 VkImage (NV12/P010) → compute
//
// Public surface: lifecycle, decode, decoder-owned conversion/readback, ML
// bridges, completion, and capability queries.

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/core/matrix.h>
#include <oa/core/image.h>
#include <vulkan/vulkan_core.h>
#include <vk_video/vulkan_video_codecs_common.h>
#include <oa/runtime/event.h>
#include <oa/vision/type.h>
#include <oa/vision/videoCodecParameterSets.h>
#include <oa/core/std/uniquePtr.h>

namespace oavk { class Buffer; }

namespace oa {

struct Av1PictureDesc;
struct Av1SequenceHeaderInfo;
struct H264MmcoCommand;
struct H264PictureDesc;
struct H265PictureDesc;
struct NormalizationParams;
struct VideoDecoderCodecAccess;
struct VideoDecoderInternal;
struct VideoDecoderRecordAccess;
struct Vp9PictureDesc;

// Decoded-frame resource path — selected once at decoder create() based on
// device capabilities + queue topology.
enum class VideoResourcePath : oa::U32 {
	// Decode + DPB→staging copy in one video submit. Requires dedicated video
	// queue with TRANSFER_BIT + coincide mode + DPB TRANSFER_SRC. NVIDIA dGPU.
	CoincidentFastStaging = 0,
	// Decode into a distinct output image, then decoded YUV→RGBA on
	// the compute queue with cross-family ownership transfer.
	DistinctComputeConvert = 1,
	// Coincide mode but copy on compute queue (cross-family barrier pair).
	CoincidentComputeStaging = 2,
	// Sample DPB directly through YCbCr conversion/view.
	DirectCoincidentSampling = 3,
	// No usable path — decoder creation returns Unavailable.
	Unavailable = 4,
};

// codec-standard profile. values are codec-qualified so a VideoProfile is
// self-describing without exposing vulkan std Video enums.
enum class VideoCodecProfile : oa::U8 {
	Unspecified = 0,
	H264Baseline,
	H264Main,
	H264High,
	H264High444Predictive,
	H265Main,
	H265Main10,
	H265MainStillPicture,
	H265FormatRangeExtensions,
	H265ScreenContentCodingExtensions,
	Av1Main,
	Av1High,
	Av1Professional,
	Vp9Profile0,
	Vp9Profile1,
	Vp9Profile2,
	Vp9Profile3,
};

enum class VideoChromaSubsampling : oa::U8 {
	Monochrome = 0,
	Yuv420,
	Yuv422,
	Yuv444,
};

enum class VideoBitDepth : oa::U8 {
	Bit8 = 8,
	Bit10 = 10,
	Bit12 = 12,
};

enum class VideoH264PictureLayout : oa::U8 {
	Progressive = 0,
	InterlacedInterleavedLines,
	InterlacedSeparatePlanes,
};

// exact stream profile for decoder capability queries and session creation.
struct VideoProfile {
	VideoCodec codec = VideoCodec::H264;
	oa::U32 width = 0;
	oa::U32 height = 0;
	oa::U32 maxDpbSlots = 0;  // Decoded Picture Buffer slots (reference frames)
	VideoCodecProfile standardProfile = VideoCodecProfile::Unspecified;
	VideoChromaSubsampling chromaSubsampling = VideoChromaSubsampling::Yuv420;
	VideoBitDepth lumaBitDepth = VideoBitDepth::Bit8;
	VideoBitDepth chromaBitDepth = VideoBitDepth::Bit8;
	VideoH264PictureLayout h264PictureLayout = VideoH264PictureLayout::Progressive;
	bool av1FilmGrain = false;
	oa::U32 level = 0;
	bool hasLevel = false;
	bool highTier = false;
};

// codec/profile-specific vulkan Video decode capabilities.
struct VideoDecodeCapabilities {
	bool supported = false;
	bool hardwareProfileSupported = false;
	bool oaDecodePathImplemented = false;
	bool supportsDpbAndOutputCoincide = false;
	bool supportsDpbAndOutputDistinct = false;
	bool supportsDecodedDpb = false;
	bool supportsDecodedDpbSampled = false;
	bool supportsDecodedDpbTransferSrc = false;
	bool supportsDecodedOutputSampled = false;
	// Compatibility detail for callers that specifically require 8-bit NV12.
	bool supportsNv12Dpb = false;
	bool supportsNv12DpbSampled = false;
	bool supportsNv12DpbTransferSrc = false;
	bool supportsNv12OutputSampled = false;
	oa::U32 maxWidth = 0;
	oa::U32 maxHeight = 0;
	oa::U32 minWidth = 0;
	oa::U32 minHeight = 0;
	oa::U32 pictureAccessGranularityWidth = 1;
	oa::U32 pictureAccessGranularityHeight = 1;
	oa::U32 maxDpbSlots = 0;
	oa::U32 maxActiveReferencePictures = 0;
	oa::U64 minBitstreamBufferOffsetAlignment = 0;
	oa::U64 minBitstreamBufferSizeAlignment = 0;
	oa::U32 maxLevel = 0;
	VkFormat pictureFormat = VK_FORMAT_UNDEFINED;
	VkFormat referencePictureFormat = VK_FORMAT_UNDEFINED;
	VkExtensionProperties stdHeaderVersion = {};
	VkVideoDecodeCapabilityFlagsKHR decodeFlags = 0;
	VideoProfile profile = {};
	oa::Vector<VkVideoFormatPropertiesKHR> dpbFormats;
	oa::Vector<VkVideoFormatPropertiesKHR> outputFormats;
};

// YCbCr to RGB conversion options
struct VideoConversionOptions {
	// Hardware sampler YCbCr conversion is the default when the exact decoded
	// format, color metadata, filter and image usage are supported. The manual
	// compute converter remains the correctness-preserving fallback.
	bool preferHardwareYCbCr = true;
	YCbCrModel colorSpace = YCbCrModel::Auto;  // BT.709, BT.2020, or auto-detect
	bool convertToRgb = true;          // convert decoded YUV→RGB (false = keep native YUV)
	Filter filter = Filter::Nearest;   // Nearest = sharp, no smoothing
};

// storage backing a video frame.
enum class VideoFrameResource : oa::U8 {
	None   = 0,
	Image  = 1,
	Buffer = 2,
};

// Timestamped frame shared by decode, capture, processing, presentation and
// encode. resource handles are non-owning and remain valid for the lifetime
// contract of their producer.
struct VideoFrame {
	VideoFrameResource resource = VideoFrameResource::Image;
	VkImage image = VK_NULL_HANDLE;
	VkImageView imageView = VK_NULL_HANDLE;
	VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
	oa::U32 externalQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	const oavk::Buffer* buffer = nullptr;
	VkFormat format = VK_FORMAT_UNDEFINED; // NV12, P010, RGBA8, or producer format
	oa::U32 width = 0;
	oa::U32 height = 0;
	oa::U64 presentationTimestamp = 0; // PTS in microseconds
	oa::U64 duration = 0;              // frame duration in microseconds, 0 if unknown
	bool isRgb = false;              // True for packed RGB/RGBA resources
	YCbCrModel colorSpace = YCbCrModel::Auto;
	bool fullRange = false;
	oa::U32 arrayLayer = 0;
	bool shown = true;
	oa::Event ready;
};

class VideoDecoder;
class VideoTranscoder;

// hardware video decoder session. Wraps VkVideoSessionKHR + DPB management.
class VideoDecoder {
	friend class VideoTranscoder;
	friend class VideoPlayer;
	friend struct VideoDecoderCodecAccess;
	friend struct VideoDecoderRecordAccess;
	friend struct VideoDecoderInternal;
public:
	VideoDecoder(VideoDecoder&& inOther) noexcept;
	VideoDecoder& operator=(VideoDecoder&& inOther) noexcept;
	VideoDecoder(const VideoDecoder&) = delete;
	VideoDecoder& operator=(const VideoDecoder&) = delete;
	~VideoDecoder();

	// Create decoder for specific codec and resolution.
	static oa::Result<VideoDecoder> create(class Engine& inRt, const VideoProfile& inProfile);

	// Decode one compressed access unit through this session's owning engine.
	[[nodiscard]] oa::Result<VideoFrame> decode(
		const oa::Span<const oa::U8>& inAccessUnit,
		oa::U64 inPts = 0ULL);
	[[nodiscard]] oa::Result<VideoFrame> decode(
		const oa::Span<const oa::U8>& inAccessUnit,
		const VideoConversionOptions& inOptions,
		oa::U64 inPts = 0ULL);
	[[nodiscard]] oa::Status decode(
		const oa::Span<const oa::U8>& inAccessUnit,
		const VideoConversionOptions& inOptions,
		VideoFrame& outFrame,
		oa::U64 inPts = 0ULL);

	// Decoder-owned color conversion, explicit targets, host observations, and
	// ML bridges.
	[[nodiscard]] oa::Result<VideoFrame> convert(
		const VideoFrame& inFrame,
		const VideoConversionOptions& inOptions = {});
	[[nodiscard]] oa::Status convertInto(
		const VideoFrame& inNv12Frame,
		const VideoConversionOptions& inOptions,
		VideoFrame& inOutRgbTarget);
	[[nodiscard]] oa::Result<oa::Event> convertIntoAsync(
		const VideoFrame& inNv12Frame,
		const VideoConversionOptions& inOptions,
		const VideoFrame& inRgbTarget);
	[[nodiscard]] oa::Result<VideoFrame> allocateRgbaFrame(
		oa::U32 inWidth,
		oa::U32 inHeight);
	[[nodiscard]] oa::Result<oa::Vector<oa::U8>> readbackLuma(const VideoFrame& inFrame);
	[[nodiscard]] oa::Result<oa::Vector<oa::U8>> readbackNv12(const VideoFrame& inFrame);
	// Returns tightly packed two-plane 4:2:0 bytes in the frame's native
	// storage representation: NV12 bytes or little-endian P010 16-bit words.
	[[nodiscard]] oa::Result<oa::Vector<oa::U8>> readbackYuv420(const VideoFrame& inFrame);
	[[nodiscard]] oa::Result<oa::Vector<oa::U8>> readbackRgba(const VideoFrame& inFrame);
	[[nodiscard]] oa::Result<Matrix> convertFrameToBf16(
		const VideoFrame& inFrame,
		bool inNormalizeImageNet = true);
	[[nodiscard]] oa::Result<Matrix> decodeFrameToBf16(
		const oa::Span<const oa::U8>& inBitstream,
		bool inNormalizeImageNet = true);
	[[nodiscard]] oa::Result<Matrix> decodeResizeNormalize(
		const oa::Span<const oa::U8>& inAccessUnit,
		oa::U32 inWidth = 224U,
		oa::U32 inHeight = 224U);
	[[nodiscard]] oa::Result<Matrix> decodeResizeNormalize(
		const oa::Span<const oa::U8>& inAccessUnit,
		oa::U32 inWidth,
		oa::U32 inHeight,
		const NormalizationParams& inNorm);

	// flush decoder state (call at stream end or seek).
	oa::Status flush();

	// wait for the last GPU submission on this decoder to complete.
	[[nodiscard]] oa::Status waitForCompletion(oa::U64 inTimeoutNs = UINT64_MAX);

	// Explicit completion and resource-release boundary.
	[[nodiscard]] oa::Status close();

	[[nodiscard]] bool isInitialized() const noexcept;
	[[nodiscard]] class Engine* getEngine() const noexcept;
	[[nodiscard]] bool hasSessionParameters() const noexcept;
	[[nodiscard]] oa::U32 getSessionParameterUpdateCount() const noexcept;
	[[nodiscard]] oa::U32 getDpbSlotCapacity() const noexcept;
	[[nodiscard]] oa::U32 getDpbViewCount() const noexcept { return 1; }
	[[nodiscard]] oa::U32 getOutputFrameCapacity() const noexcept;
	[[nodiscard]] oa::U32 getOutputViewCount() const noexcept;
	[[nodiscard]] oa::U32 getDpbInUseCount() const noexcept;
	[[nodiscard]] oa::U32 getDpbReferenceCount() const noexcept;
	[[nodiscard]] oa::U64 getCurrentFrameNumber() const noexcept;
	[[nodiscard]] oa::U32 getCodedWidth() const noexcept;
	[[nodiscard]] oa::U32 getCodedHeight() const noexcept;
	[[nodiscard]] VideoResourcePath getResourcePath() const noexcept;

	// query capabilities
	static oa::Result<VideoDecodeCapabilities> queryDecodeCapabilities(Engine& inRt, const VideoProfile& inProfile);
	static oa::Result<VideoDecodeCapabilities> queryDecodeCapabilities(Engine& inRt, VideoCodec inCodec);

private:
	oa::Status decodeFrame(
		const oa::Span<const oa::U8>& inBitstream,
		VideoFrame& outFrame);
	oa::Status decodeFrameWithConversion(
		const oa::Span<const oa::U8>& inBitstream,
		const VideoConversionOptions& inOptions,
		VideoFrame& outFrame);

	[[nodiscard]] oa::Result<Matrix> convertFrameToBf16Hardware(
		const VideoFrame& inFrame,
		bool inNormalizeImageNet = true);
	oa::Status convertFrameToRgba(
		const VideoFrame& inNv12Frame,
		const VideoConversionOptions& inOptions,
		VideoFrame& outRgbFrame);
	oa::Status convertNv12ToRgbInto(
		const VideoFrame& inNv12Frame,
		const VideoConversionOptions& inOptions,
		VideoFrame& inOutRgbTarget);
	[[nodiscard]] oa::Result<oa::Event> convertNv12ToRgbIntoAsync(
		const VideoFrame& inNv12Frame,
		const VideoConversionOptions& inOptions,
		const VideoFrame& inRgbTarget);
	[[nodiscard]] oa::Result<oa::Event> convertNv12ToRgbHardwareIntoAsync(
		const VideoFrame& inYcbcrFrame,
		const VideoConversionOptions& inOptions,
		const VideoFrame& inRgbTarget);
	oa::Status destroyHardwareYcbcr_();
	oa::Status restoreDpbLayerToDecodeLayout(const VideoFrame& inFrame);
	[[nodiscard]] oa::U64 getBitstreamBufferCapacity() const noexcept;
	[[nodiscard]] oa::U32 getBitstreamRingSize() const noexcept { return kBitstreamRingSize; }
	[[nodiscard]] oa::U32 getCachedSpsCount() const noexcept;
	[[nodiscard]] oa::U32 getCachedPpsCount() const noexcept;
	[[nodiscard]] oa::U32 getCachedH265VpsCount() const noexcept;
	[[nodiscard]] oa::U32 getCachedH265SpsCount() const noexcept;
	[[nodiscard]] oa::U32 getCachedH265PpsCount() const noexcept;
	[[nodiscard]] oa::U64 getHardwareYcbcrDispatchCount() const noexcept;
	static bool hasHardwareYCbCrConversion(Engine& inRt);
	VideoDecoder();
	void moveFrom(VideoDecoder&& inOther) noexcept;
	void abandon_() noexcept;
	static oa::Status completeRetired_(void* inPayload);
	static void releaseRetired_(void* inPayload);

	struct DpbSlot;

	oa::I32 allocateDpbSlot();
	void markSlotAsReference(oa::I32 inSlotIndex, oa::I32 inPicOrderCnt);
	void releaseDpbSlot(oa::I32 inSlotIndex);
	void buildRefPicList0(oa::I32 inCurrentPoc, oa::Vector<oa::I32>& outRefList);
	void buildRefPicList1(oa::I32 inCurrentPoc, oa::Vector<oa::I32>& outRefList);
	void buildH264RefPicList0P(oa::Vector<oa::I32>& outRefList);
	void applySlidingWindow(oa::U32 inMaxNumRefFrames);
	void applyMmco(
		const oa::Vector<H264MmcoCommand>& inMmcoCommands,
		oa::I32 inCurrentDpbSlot);
	void applyMmco(const oa::Vector<oa::U32>& inMmcoCommands);

	oa::Status cacheSps(oa::U32 inSpsId, const H264SpsData& inSps);
	oa::Status cachePps(oa::U32 inPpsId, const H264PpsData& inPps);
	const H264SpsData* getSps(oa::U32 inSpsId) const;
	const H264PpsData* getPps(oa::U32 inPpsId) const;
	void clearParameterSets();

	oa::Status updateH264SessionParametersFromSps(const H264SpsData& inSps);
	oa::Status updateH264SessionParametersFromPps(const H264PpsData& inPps);
	oa::Status updateH265SessionParametersFromVps(const H265VpsData& inVps);
	oa::Status updateH265SessionParametersFromSps(const H265SpsData& inSps);
	oa::Status updateH265SessionParametersFromPps(const H265PpsData& inPps);
	oa::Status updateAv1SessionParametersFromSequenceHeader(
		const Av1SequenceHeaderInfo& inSeq);
	oa::Status uploadBitstream(const oa::Span<const oa::U8>& inBitstream);
	oa::Status transitionFrameForSampledRead(const VideoFrame& inFrame);
	oa::Status restoreDpbLayerToDecodeLayoutAfter(const VideoFrame& inFrame, const oa::Event& inWait);
	oa::Status releaseDpbLayerForComputeCopy(const VideoFrame& inFrame);
	[[nodiscard]] VkImageLayout getFrameLayout(const VideoFrame& inFrame, bool& outIsOutput, oa::U32& outImageIndex) const;
	void setFrameLayout(bool inIsOutput, oa::U32 inImageIndex, VkImageLayout inLayout);
	oa::Status createOutputImages(
		class Engine& inRt,
		const VkVideoProfileInfoKHR& inProfile,
		VkFormat inFormat,
		VkExtent2D inCodedExtent,
		oa::U32 inSlotCount);
	oa::Status createSampleStagingImages(
		class Engine& inRt,
		const VkVideoProfileInfoKHR& inProfile,
		VkExtent2D inCodedExtent,
		oa::U32 inSlotCount);
	void recordDpbLayerToSampleImage(VkCommandBuffer inCommandBuffer, oa::I32 inDpbSlot);
	oa::Status copyDpbLayerToSampleImage(const VideoFrame& inDpbFrame);
	[[nodiscard]] oa::U32 getNv12PlaneArrayLayer(const VideoFrame& inFrame) const;
	[[nodiscard]] oa::Result<VideoFrame> allocateRgbaFrame_(oa::U32 inWidth, oa::U32 inHeight, oa::U64 inPts);
	[[nodiscard]] oa::Result<VideoFrame> acquireConvertedRgbaTarget(oa::U32 inWidth, oa::U32 inHeight, oa::U64 inPts);
	[[nodiscard]] VkImageView getCachedNv12PlaneView(VkImage inImage, oa::U32 inLayer, VkFormat inFormat, VkImageAspectFlagBits inPlane);
	[[nodiscard]] VkSampler getCachedNv12Sampler(Filter inFilter = Filter::Nearest);
	oa::Status recordH264DecodeCommands(
		oa::I32 inDpbSlot,
		const H264PictureDesc& inDesc,
		const oa::Vector<oa::I32>& inRefPicList0,
		const oa::Vector<oa::I32>& inRefPicList1);
	oa::Status recordH265DecodeCommands(
		oa::I32 inDpbSlot,
		const H265PictureDesc& inDesc,
		const oa::Vector<oa::I32>& inRefPicList0,
		const oa::Vector<oa::I32>& inRefPicList1);
	oa::Status recordAV1DecodeCommands(
		oa::I32 inDpbSlot,
		const Av1PictureDesc& inDesc,
		const oa::I32 inReferenceNameSlotIndices[7]);
	oa::Status recordVP9DecodeCommands(
		oa::I32 inDpbSlot,
		const Vp9PictureDesc& inDesc,
		const oa::I32 inReferenceNameSlotIndices[STD_VIDEO_VP9_REFS_PER_FRAME],
		const oa::Vector<oa::I32>& inReferenceSlots,
		const oa::Vector<VkExtent2D>& inReferenceExtents);
	oa::Status convertNv12ToRgb(
		const VideoFrame& inNv12Frame,
		const VideoConversionOptions& inOptions,
		VideoFrame& outRgbFrame);
	oa::Status convertNv12ToRgbCompute(
		const VideoFrame& inNv12Frame,
		YCbCrModel inColorSpace,
		VideoFrame& outRgbFrame,
		Filter inFilter = Filter::Nearest);

	struct VideoCmdSlot;
	VideoCmdSlot acquireVideoCmdSlot();
	void releaseVideoCmdSlot();
	void stampFrameReady(VideoFrame& outFrame) const noexcept;

	static constexpr oa::U32 kCmdBufferCount = 8;
	struct BitstreamSlot;
	static constexpr oa::U32 kBitstreamRingSize = 4;

	class Impl;
	oa::UniquePtr<Impl> impl_;
};

// Video frame resource pool (reuse decoded frames).
class VideoFramePool {
public:
	VideoFramePool(VideoFramePool&& inOther) noexcept;
	VideoFramePool& operator=(VideoFramePool&& inOther) noexcept;
	VideoFramePool(const VideoFramePool&) = delete;
	VideoFramePool& operator=(const VideoFramePool&) = delete;
	~VideoFramePool();

	static oa::Result<VideoFramePool> create(
		Engine& inRt,
		oa::U32 inWidth,
		oa::U32 inHeight,
		oa::U32 inPoolSize);

	// acquire frame from pool (blocks if all in use).
	VideoFrame acquire();

	// Return frame to pool.
	void release(const VideoFrame& inFrame);

private:
	VideoFramePool() = default;
	void moveFrom(VideoFramePool&& inOther) noexcept;
	void reset_() noexcept;

	oa::Vector<VideoFrame> frames_;
	oa::Vector<bool> inUse_;
	oa::Vector<void*> allocations_;
	Engine* rt_ = nullptr;
};

} // namespace oa
