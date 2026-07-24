// OA Vision — Hardware Video Decoder
// VK_KHR_video_decode_h264 / h265 / av1 / vp9
// Zero-copy: compressed bitstream → VkImage (NV12) → compute shader
//
// Public surface: Create, DecodeFrame, Flush, WaitForCompletion, caps query,
// and session getters. Color conversion, readback, and BF16 bridges live on
// OaFnVideo (OaFnVideo.h) and delegate through OaVideoDecoderInternal.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/Image.h>
#include <Oa/Runtime/ImageDispatch.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Runtime/OaVkVideo.h>
#include <Oa/Runtime/Sync.h>
#include <Oa/Vision/Type.h>
#include <Oa/Vision/VideoCodecParameterSets.h>
#include <Oa/Core/Std/UniquePtr.h>

class OaVkBuffer;

// Decoded-frame resource path — selected once at decoder Create() based on
// device capabilities + queue topology. Mirrors the GEMM capability-gating
// pattern (OaCapBit / ComputeCapsMask / Router). See
// VideoDecoderDeviceCompatibility.md for the full design.
enum class OaVideoResourcePath : OaU32 {
	// Decode + DPB→staging copy in one video submit. Requires dedicated video
	// queue with TRANSFER_BIT + coincide mode + DPB TRANSFER_SRC. NVIDIA dGPU.
	CoincidentFastStaging = 0,
	// Decode into distinct output image, then NV12→RGBA via compute shader on
	// the compute queue with cross-family ownership transfer. Safe for iGPUs
	// with no video-queue transfer capability.
	DistinctComputeConvert = 1,
	// Coincide mode but copy on compute queue (cross-family barrier pair).
	// Fallback when video queue lacks transfer but distinct output is unavailable.
	CoincidentComputeStaging = 2,
	// Sample DPB directly through YCbCr conversion/view. Lowest overhead,
	// highest driver dependency.
	DirectCoincidentSampling = 3,
	// No usable path — decoder creation returns Unavailable.
	Unavailable = 4,
};

// Codec-standard profile. Values are codec-qualified so an OaVideoProfile is
// self-describing without exposing Vulkan Std Video enums in the public
// contract. Unspecified preserves source compatibility for callers that only
// supplied codec/extent; decoder creation resolves it to the currently verified
// default for that codec before querying the device.
enum class OaVideoCodecProfile : OaU8 {
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

enum class OaVideoChromaSubsampling : OaU8 {
	Monochrome = 0,
	Yuv420,
	Yuv422,
	Yuv444,
};

enum class OaVideoBitDepth : OaU8 {
	Bit8 = 8,
	Bit10 = 10,
	Bit12 = 12,
};

enum class OaVideoH264PictureLayout : OaU8 {
	Progressive = 0,
	InterlacedInterleavedLines,
	InterlacedSeparatePlanes,
};

// Exact stream profile for decoder capability queries and session creation.
// Level uses the codec's Std Video numeric level value when HasLevel is true.
struct OaVideoProfile {
	OaVideoCodec Codec = OaVideoCodec::H264;
	OaU32 Width = 0;
	OaU32 Height = 0;
	OaU32 MaxDpbSlots = 0;  // Decoded Picture Buffer slots (reference frames)
	OaVideoCodecProfile StandardProfile = OaVideoCodecProfile::Unspecified;
	OaVideoChromaSubsampling ChromaSubsampling = OaVideoChromaSubsampling::Yuv420;
	OaVideoBitDepth LumaBitDepth = OaVideoBitDepth::Bit8;
	OaVideoBitDepth ChromaBitDepth = OaVideoBitDepth::Bit8;
	OaVideoH264PictureLayout H264PictureLayout = OaVideoH264PictureLayout::Progressive;
	bool Av1FilmGrain = false;
	OaU32 Level = 0;
	bool HasLevel = false;
	bool HighTier = false;
};

// Codec/profile-specific Vulkan Video decode capabilities.
struct OaVideoDecodeCapabilities {
	bool Supported = false;
	bool HardwareProfileSupported = false;
	bool OaDecodePathImplemented = false;
	bool SupportsDpbAndOutputCoincide = false;
	bool SupportsDpbAndOutputDistinct = false;
	bool SupportsNv12Dpb = false;
	bool SupportsNv12DpbSampled = false;
	bool SupportsNv12DpbTransferSrc = false;
	bool SupportsNv12OutputSampled = false;
	OaU32 MaxWidth = 0;
	OaU32 MaxHeight = 0;
	OaU32 MinWidth = 0;
	OaU32 MinHeight = 0;
	OaU32 PictureAccessGranularityWidth = 1;
	OaU32 PictureAccessGranularityHeight = 1;
	OaU32 MaxDpbSlots = 0;
	OaU32 MaxActiveReferencePictures = 0;
	OaU64 MinBitstreamBufferOffsetAlignment = 0;
	OaU64 MinBitstreamBufferSizeAlignment = 0;
	OaU32 MaxLevel = 0;
	VkFormat PictureFormat = VK_FORMAT_UNDEFINED;
	VkFormat ReferencePictureFormat = VK_FORMAT_UNDEFINED;
	VkExtensionProperties StdHeaderVersion = {};
	VkVideoDecodeCapabilityFlagsKHR DecodeFlags = 0;
	OaVideoProfile Profile = {};
	OaVec<VkVideoFormatPropertiesKHR> DpbFormats;
	OaVec<VkVideoFormatPropertiesKHR> OutputFormats;
};

// YCbCr to RGB conversion options
struct OaVideoConversionOptions {
	bool PreferHardwareYCbCr = true;   // Use VK_KHR_sampler_ycbcr_conversion if available
	OaYCbCrModel ColorSpace = OaYCbCrModel::Auto;  // BT.709, BT.2020, or auto-detect
	bool ConvertToRgb = true;          // Convert NV12→RGB (false = keep NV12)
	OaFilter Filter = OaFilter::Nearest; // Nearest = sharp, no smoothing
};

// Storage backing a video frame. Decode normally produces Image; camera and
// screen sources may initially produce a bindless Buffer when the platform
// cannot export DMA-BUF. Consumers must branch on Resource rather than infer
// ownership from IsRgb.
enum class OaVideoFrameResource : OaU8 {
	None   = 0,
	Image  = 1,
	Buffer = 2,
};

// Timestamped frame shared by decode, capture, processing, presentation and
// encode. Resource handles are non-owning and remain valid for the lifetime
// contract of their producer (decoder output pool or capture ring).
struct OaVideoFrame {
	// Image preserves the historical decode-frame default for aggregate `{}`
	// construction throughout the codec implementations.
	OaVideoFrameResource Resource = OaVideoFrameResource::Image;
	VkImage Image = VK_NULL_HANDLE;
	VkImageView ImageView = VK_NULL_HANDLE;
	// Current image layout. Image producers must publish this when Resource is
	// Image so downstream consumers can transition without discarding content.
	VkImageLayout Layout = VK_IMAGE_LAYOUT_UNDEFINED;
	OaU32 ExternalQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	const OaVkBuffer* Buffer = nullptr;
	VkFormat Format = VK_FORMAT_UNDEFINED; // NV12, RGBA8, or producer format
	OaU32 Width = 0;
	OaU32 Height = 0;
	OaU64 PresentationTimestamp = 0; // PTS in microseconds
	OaU64 Duration = 0;              // frame duration in microseconds, 0 if unknown
	bool IsRgb = false;              // True for packed RGB/RGBA resources
	OaYCbCrModel ColorSpace = OaYCbCrModel::Auto;
	bool FullRange = false;
	// Which array layer of the source image holds this frame. The decoder DPB
	// is one VkImage with one layer per slot; convert/sample paths must use
	// this when building views, otherwise they always read layer 0 and motion
	// shows up as cross-frame ghosting against an otherwise-static background.
	OaU32 ArrayLayer = 0;

	// Display gate for codecs with hidden frames. AV1/VP9 may code a frame with
	// show_frame=0 (a hidden alt-ref): it decodes into the DPB and yields a
	// valid image, but must NOT be presented — a later frame references it and
	// is shown in its place. Presenting it produces duplicated / out-of-order
	// frames. show_existing_frame re-displays a stored slot and IS shown.
	// H.264/H.265 have no hidden frames, so this stays true for them.
	bool Shown = true;

	// GPU completion for the work that produced this frame. The token is
	// non-owning and remains valid for the producer session lifetime. Consumers
	// should pass Ready.TimelineWait() to downstream queues instead of waiting
	// on the host.
	OaEvent Ready;

};

struct OaAv1PictureDesc;
struct OaH264PictureDesc;
struct OaH265PictureDesc;
struct OaFnVideoDecoderAccess;
class OaVideoCodecParser;
class OaVideoDecoder;

namespace FnVideoDecoderH264 {
OaStatus DecodeFrame(class OaVideoDecoder& InDecoder, const OaSpan<const OaU8>& InBitstream, OaVideoFrame& OutFrame);
}
namespace FnVideoDecoderH265 {
OaStatus DecodeFrame(class OaVideoDecoder& InDecoder, const OaSpan<const OaU8>& InBitstream, OaVideoFrame& OutFrame);
}
namespace FnVideoDecoderAv1 {
OaStatus DecodeFrame(class OaVideoDecoder& InDecoder, const OaSpan<const OaU8>& InBitstream, OaVideoFrame& OutFrame);
OaStatus DecodePicture(class OaVideoDecoder& InDecoder, const OaSpan<const OaU8>& InBitstream, const OaAv1PictureDesc& InDesc, OaVideoFrame& OutFrame);
}
namespace FnVideoDecoderVp9 {
OaStatus DecodeFrame(class OaVideoDecoder& InDecoder, const OaSpan<const OaU8>& InBitstream, OaVideoFrame& OutFrame);
}

// Hardware video decoder session
// Wraps VkVideoSessionKHR + DPB management
class OaVideoDecoder {
	friend class OaVideoTranscoder;
	friend struct OaFnVideoDecoderAccess;
	friend struct OaFnVideoDecoderRecord;
	friend struct OaVideoDecoderInternal;
	friend OaStatus FnVideoDecoderH264::DecodeFrame(OaVideoDecoder&, const OaSpan<const OaU8>&, OaVideoFrame&);
	friend OaStatus FnVideoDecoderH265::DecodeFrame(OaVideoDecoder&, const OaSpan<const OaU8>&, OaVideoFrame&);
	friend OaStatus FnVideoDecoderAv1::DecodeFrame(OaVideoDecoder&, const OaSpan<const OaU8>&, OaVideoFrame&);
	friend OaStatus FnVideoDecoderAv1::DecodePicture(OaVideoDecoder&, const OaSpan<const OaU8>&,
		const OaAv1PictureDesc&, OaVideoFrame&
	);
	friend OaStatus FnVideoDecoderVp9::DecodeFrame(OaVideoDecoder&, const OaSpan<const OaU8>&, OaVideoFrame&);
public:
	OaVideoDecoder(OaVideoDecoder&& InOther) noexcept;
	OaVideoDecoder& operator=(OaVideoDecoder&& InOther) noexcept;
	OaVideoDecoder(const OaVideoDecoder&) = delete;
	OaVideoDecoder& operator=(const OaVideoDecoder&) = delete;
	~OaVideoDecoder();

	// Create decoder for specific codec and resolution
	static OaResult<OaVideoDecoder> Create(class OaEngine &InRt, const OaVideoProfile &InProfile);

	// Decode one frame from compressed bitstream
	// Returns VkImage in NV12 format (or RGB if conversion enabled)
	OaStatus DecodeFrame(const OaSpan<const OaU8> &InBitstream, OaVideoFrame &OutFrame);
	
	// Decode with YCbCr→RGB conversion
	OaStatus DecodeFrameWithConversion(
		const OaSpan<const OaU8> &InBitstream,
		const OaVideoConversionOptions &InOptions,
		OaVideoFrame &OutFrame
	);

	// Flush decoder state (call at stream end or seek)
	OaStatus Flush();

	// Wait for the last GPU submission on this decoder to complete.
	// Used by OaVideo to ensure a frame is safe to read before presenting.
	[[nodiscard]] OaStatus WaitForCompletion(OaU64 InTimeoutNs = UINT64_MAX);

	// Explicit completion and resource-release boundary.
	[[nodiscard]] OaStatus Close();
	// Compatibility wrapper that logs Close() failures. Prefer Close() where
	// the shutdown result can be propagated.
	void Destroy();

	[[nodiscard]] bool IsInitialized() const noexcept { return Session_.Handle() != VK_NULL_HANDLE; }
	[[nodiscard]] class OaEngine* GetEngine() const noexcept { return Rt_; }
	[[nodiscard]] bool HasSessionParameters() const noexcept { return SessionParams_.Handle() != VK_NULL_HANDLE; }
	[[nodiscard]] OaU32 GetSessionParameterUpdateCount() const noexcept { return SessionParameterUpdateCount_; }
	[[nodiscard]] OaU32 GetDpbSlotCapacity() const noexcept { return DpbSlotCapacity_; }
	[[nodiscard]] OaU32 GetDpbViewCount() const noexcept { return 1; } // OaVkVideoDpb has a single view
	[[nodiscard]] OaU32 GetOutputFrameCapacity() const noexcept { return OutputFrameCapacity_; }
	[[nodiscard]] OaU32 GetOutputViewCount() const noexcept { return static_cast<OaU32>(OutputViews_.Size()); }
	[[nodiscard]] OaU32 GetDpbInUseCount() const noexcept;
	[[nodiscard]] OaU32 GetDpbReferenceCount() const noexcept;
	[[nodiscard]] OaU64 GetCurrentFrameNumber() const noexcept { return CurrentFrameNumber_; }
	[[nodiscard]] OaU32 GetCodedWidth() const noexcept { return CodedWidth_; }
	[[nodiscard]] OaU32 GetCodedHeight() const noexcept { return CodedHeight_; }
	[[nodiscard]] OaVideoResourcePath GetResourcePath() const noexcept { return ResourcePath_; }

	// Query capabilities
	static OaResult<OaVideoDecodeCapabilities> QueryDecodeCapabilities(OaEngine& InRt, const OaVideoProfile& InProfile);
	static OaResult<OaVideoDecodeCapabilities> QueryDecodeCapabilities(OaEngine& InRt, OaVideoCodec InCodec);
	static bool IsProfileSupported(OaEngine& InRt, const OaVideoProfile& InProfile);
	static bool IsCodecSupported(OaEngine &InRt, OaVideoCodec InCodec);
	static OaU32 GetMaxWidth(OaEngine &InRt, OaVideoCodec InCodec);
	static OaU32 GetMaxHeight(OaEngine &InRt, OaVideoCodec InCodec);

private:
	// Parsing state is stream-local. Sharing SPS/PPS/reference-header caches
	// through the codec registry corrupts independent or concurrent decoders.
	OaStdUniquePtr<OaVideoCodecParser> Parser_;
	[[nodiscard]] OaResult<OaVec<OaU8>> ReadbackLuma(const OaVideoFrame &InFrame);
	[[nodiscard]] OaResult<OaVec<OaU8>> ReadbackNv12(const OaVideoFrame &InFrame);
	[[nodiscard]] OaResult<OaVec<OaU8>> ReadbackRgba(const OaVideoFrame &InFrame);
	[[nodiscard]] OaResult<OaMatrix> ConvertFrameToBf16(
		const OaVideoFrame &InFrame,
		bool InNormalizeImageNet = true
	);
	[[nodiscard]] OaResult<OaMatrix> ConvertFrameToBf16Hardware(
		const OaVideoFrame &InFrame,
		bool InNormalizeImageNet = true
	);
	[[nodiscard]] OaResult<OaMatrix> DecodeFrameToBf16(
		const OaSpan<const OaU8> &InBitstream,
		bool InNormalizeImageNet = true
	);
	OaStatus ConvertFrameToRgba(
		const OaVideoFrame &InNv12Frame,
		const OaVideoConversionOptions &InOptions,
		OaVideoFrame &OutRgbFrame
	);
	OaStatus ConvertNv12ToRgbInto(
		const OaVideoFrame &InNv12Frame,
		const OaVideoConversionOptions &InOptions,
		OaVideoFrame &InOutRgbTarget
	);
	[[nodiscard]] OaResult<OaVkImageDispatchTicket> ConvertNv12ToRgbIntoAsync(
		const OaVideoFrame& InNv12Frame,
		const OaVideoConversionOptions& InOptions,
		const OaVideoFrame& InRgbTarget
	);
	[[nodiscard]] OaResult<OaVideoFrame> AllocateOutputRgbaFrame(OaU32 InWidth, OaU32 InHeight) {
		return AllocateRgbaFrame(InWidth, InHeight, 0);
	}
	OaStatus RestoreDpbLayerToDecodeLayout(const OaVideoFrame& InFrame);
	[[nodiscard]] OaU64 GetBitstreamBufferCapacity() const noexcept {
		return BitstreamRing_[CurrentBitstreamIndex_].Buffer.GetCapacity();
	}
	[[nodiscard]] OaU32 GetBitstreamRingSize() const noexcept { return kBitstreamRingSize; }
	[[nodiscard]] OaU32 GetCachedSpsCount() const noexcept { return static_cast<OaU32>(SpsCache_.Size()); }
	[[nodiscard]] OaU32 GetCachedPpsCount() const noexcept { return static_cast<OaU32>(PpsCache_.Size()); }
	[[nodiscard]] OaU32 GetCachedH265VpsCount() const noexcept { return static_cast<OaU32>(H265VpsCache_.Size()); }
	[[nodiscard]] OaU32 GetCachedH265SpsCount() const noexcept { return static_cast<OaU32>(H265SpsCache_.Size()); }
	[[nodiscard]] OaU32 GetCachedH265PpsCount() const noexcept { return static_cast<OaU32>(H265PpsCache_.Size()); }
	static bool HasHardwareYCbCrConversion(OaEngine &InRt);
	OaVideoDecoder() = default;  // OaVkVideo classes have default constructors
	void MoveFrom(OaVideoDecoder&& InOther) noexcept;
	void Abandon_() noexcept;
	static OaStatus CompleteRetired_(void* InPayload);
	static void ReleaseRetired_(void* InPayload);
	
	// Phase 2.4: DPB Management
	struct DpbSlot {
		bool InUse = false;
		OaI32 PicOrderCnt = -1;      // Picture order count (for H.264/H.265)
		OaU64 FrameNumber = 0;       // Sequential frame number (monotonic, ours)
		OaU32 H264FrameNum = 0;      // H.264 slice frame_num (mod MaxFrameNum)
		bool IsReference = false;    // Can be used as reference frame
		bool IsLongTerm  = false;    // Long-term reference (H.264 §8.2.5)
	};
	
	// Allocate DPB slot for decoded frame
	OaI32 AllocateDpbSlot();
	
	// Mark DPB slot as reference frame
	void MarkSlotAsReference(OaI32 InSlotIndex, OaI32 InPicOrderCnt);
	
	// Release DPB slot (no longer needed)
	void ReleaseDpbSlot(OaI32 InSlotIndex);
	
	// Phase 2.4.4: Reference Frame List Building
	// Build RefPicList0 for P/B-frame prediction (forward references)
	void BuildRefPicList0(OaI32 InCurrentPoc,	OaVec<OaI32>& OutRefList);
	
	// Build RefPicList1 for B-frame prediction (backward references)
	void BuildRefPicList1(OaI32 InCurrentPoc,	OaVec<OaI32>& OutRefList);

	// Build RefPicList0 for an H.264 P-slice per §8.2.4.2.1: short-term refs
	// ordered by PicNum (FrameNumWrap) descending, then long-term refs. Unlike
	// the POC-keyed builder above (correct for HEVC and H.264 B-slices), a
	// P-slice must include *every* short-term reference and order by decode
	// recency, not display order.
	void BuildH264RefPicList0P(OaVec<OaI32>& OutRefList);
	
	// Sliding window DPB management (mark oldest as unused when full)
	void ApplySlidingWindow(OaU32 InMaxNumRefFrames);
	
	// MMCO (Memory Management Control Operations) for explicit DPB control.
	// Walks the slice's parsed dec_ref_pic_marking commands and updates the
	// software-side DPB slot bookkeeping accordingly.
	void ApplyMmco(const OaVec<struct OaH264MmcoCommand>& InMmcoCommands,	OaI32 InCurrentDpbSlot);

	// Legacy raw-u32 overload (kept for backwards compat — no longer wired).
	void ApplyMmco(const OaVec<OaU32>& InMmcoCommands);
	
	// Phase 2.4.2: Parameter Set Cache Management
	// Cache SPS (Sequence Parameter Set)
	OaStatus CacheSps(OaU32 InSpsId, const OaH264SpsData& InSps);
	
	// Cache PPS (Picture Parameter Set)
	OaStatus CachePps(OaU32 InPpsId, const OaH264PpsData& InPps);
	
	// Get cached SPS (returns nullptr if not found)
	const OaH264SpsData* GetSps(OaU32 InSpsId) const;
	
	// Get cached PPS (returns nullptr if not found)
	const OaH264PpsData* GetPps(OaU32 InPpsId) const;
	
	// Clear all cached parameter sets
	void ClearParameterSets();

	OaStatus UpdateH264SessionParametersFromSps(const OaH264SpsData& InSps);
	OaStatus UpdateH264SessionParametersFromPps(const OaH264PpsData& InPps);
	OaStatus UpdateH265SessionParametersFromVps(const OaH265VpsData& InVps);
	OaStatus UpdateH265SessionParametersFromSps(const OaH265SpsData& InSps);
	OaStatus UpdateH265SessionParametersFromPps(const OaH265PpsData& InPps);
	OaStatus UpdateAv1SessionParametersFromSequenceHeader(const struct OaAv1SequenceHeaderInfo& InSeq);
	OaStatus UploadBitstream(const OaSpan<const OaU8>& InBitstream);
	OaStatus TransitionFrameForSampledRead(const OaVideoFrame& InFrame);
	OaStatus RestoreDpbLayerToDecodeLayoutAfter(
		const OaVideoFrame& InFrame,
		const OaVkTimelineSemaphore* InWaitSemaphore,
		OaU64 InWaitValue
	);
	OaStatus ReleaseDpbLayerForComputeCopy(const OaVideoFrame& InFrame);
	[[nodiscard]] VkImageLayout GetFrameLayout(const OaVideoFrame& InFrame, bool& OutIsOutput, OaU32& OutImageIndex) const;
	void SetFrameLayout(bool InIsOutput, OaU32 InImageIndex, VkImageLayout InLayout);
	OaStatus EnsureYcbcrSampler(OaYCbCrModel InColorSpace, OaFilter InFilter = OaFilter::Nearest);
	// Coincident DPB/output surfaces are copied into plain NV12 staging images
	// when the video format exposes TRANSFER_SRC. This avoids sampling video
	// profile images through mutable R8/R8G8 plane views on restrictive drivers.
	OaStatus CreateOutputImages(
		class OaEngine& InRt,
		const VkVideoProfileInfoKHR& InProfile,
		VkFormat InFormat,
		VkExtent2D InCodedExtent,
		OaU32 InSlotCount
	);
	OaStatus CreateSampleStagingImages(
		class OaEngine& InRt,
		const VkVideoProfileInfoKHR& InProfile,
		VkExtent2D InCodedExtent,
		OaU32 InSlotCount
	);
	void RecordDpbLayerToSampleImage(VkCommandBuffer InCommandBuffer,	OaI32 InDpbSlot);
	// Copies one DPB layer into its per-slot NV12 staging image on the compute
	// queue when the video queue lacks TRANSFER_BIT, then restores the DPB layer
	// on the video queue.
	OaStatus CopyDpbLayerToSampleImage(const OaVideoFrame& InDpbFrame);
	[[nodiscard]] OaU32 GetNv12PlaneArrayLayer(const OaVideoFrame& InFrame) const;
	[[nodiscard]] OaResult<OaVideoFrame> AllocateRgbaFrame(OaU32 InWidth, OaU32 InHeight, OaU64 InPts);
	// Reuses a single decoder-owned RGBA target if its size matches; allocates
	// on first call (and after a resize). Drops the AllocateRgbaFrame stall
	// on every frame, which is what kept playback at ~3 fps.
	[[nodiscard]] OaResult<OaVideoFrame> AcquireConvertedRgbaTarget(OaU32 InWidth, OaU32 InHeight, OaU64 InPts);
	// Caches a Y- or UV-plane VkImageView for a specific DPB array layer. The
	// shader-path convert recreated these every call before; with 30 fps + 16
	// slot DPB that was 60+ vkCreateImageView/vkDestroyImageView calls/sec
	// just for plane views.
	[[nodiscard]] VkImageView GetCachedNv12PlaneView(VkImage InImage, OaU32 InLayer, VkFormat InFormat, VkImageAspectFlagBits InPlane);
	[[nodiscard]] VkSampler GetCachedNv12Sampler(OaFilter InFilter = OaFilter::Nearest);
	OaStatus RecordH264DecodeCommands(
		OaI32 InDpbSlot,
		const OaH264PictureDesc& InDesc,
		const OaVec<OaI32>& InRefPicList0,
		const OaVec<OaI32>& InRefPicList1
	);
	OaStatus RecordH265DecodeCommands(
		OaI32 InDpbSlot,
		const OaH265PictureDesc& InDesc,
		const OaVec<OaI32>& InRefPicList0,
		const OaVec<OaI32>& InRefPicList1
	);
	OaStatus RecordAV1DecodeCommands(
		OaI32 InDpbSlot,
		const OaAv1PictureDesc& InDesc,
		const OaI32 InReferenceNameSlotIndices[7]
	);
	OaStatus RecordVP9DecodeCommands(
		OaI32 InDpbSlot,
		const struct OaVp9PictureDesc& InDesc,
		const OaI32 InReferenceNameSlotIndices[STD_VIDEO_VP9_REFS_PER_FRAME],
		const OaVec<OaI32>& InReferenceSlots,
		const OaVec<VkExtent2D>& InReferenceExtents
	);
	
	// Convert NV12 frame to RGB using hardware or compute shader
	OaStatus ConvertNv12ToRgb(
		const OaVideoFrame &InNv12Frame,
		const OaVideoConversionOptions &InOptions,
		OaVideoFrame &OutRgbFrame
	);
	
	// Hardware YCbCr conversion path (VK_KHR_sampler_ycbcr_conversion)
	OaStatus ConvertNv12ToRgbHardware(
		const OaVideoFrame &InNv12Frame,
		OaYCbCrModel InColorSpace,
		OaVideoFrame &OutRgbFrame,
		OaFilter InFilter = OaFilter::Nearest
	);

	// Software YCbCr conversion path (compute shader fallback)
	OaStatus ConvertNv12ToRgbCompute(
		const OaVideoFrame &InNv12Frame,
		OaYCbCrModel InColorSpace,
		OaVideoFrame &OutRgbFrame,
		OaFilter InFilter = OaFilter::Nearest
	);

	// Helper for acquiring/releasing the rotating video command buffer slot.
	// All video-queue operations (decode, transition, restore, readback) go
	// through this so they share the same fence-wait/reset/submit cycle.
	struct VideoCmdSlot {
		VkCommandBuffer cb = VK_NULL_HANDLE;
		VkFence fence = VK_NULL_HANDLE;
		OaStatus Status = OaStatus::Ok();
	};
	VideoCmdSlot AcquireVideoCmdSlot();
	void ReleaseVideoCmdSlot();
	void StampFrameReady(OaVideoFrame& OutFrame) const noexcept;

	// Vulkan video session + parameters (Layer 1: OaVkVideo)
	OaVkVideoSession Session_;
	OaVkVideoParameters SessionParams_;
	OaVkVideoQueue Queue_;

	// Command buffer ring: two primary CBs so decode N+1 can be recorded while
	// decode N is still in flight on the GPU. Index alternates every submit.
	// Increased to 8 for AI/ML/HPC sustained decode workloads (was 2, too small for pipelining).
	static constexpr OaU32 kCmdBufferCount = 8;
	OaArray<VkCommandBuffer, kCmdBufferCount> CmdBuffers_ = {};
	OaArray<VkFence, kCmdBufferCount> CmdFences_ = {};
	OaU32 CurrentCbIndex_ = 0;

	// Timeline semaphore for GPU-side completion tracking across all video
	// queue operations (decode, transition, restore). Each submit signals
	// an incremented value; callers can wait on the host or chain further
	// submits with VkTimelineSemaphoreSubmitInfo.
	OaVkTimelineSemaphore TimelineSem_;
	OaU64 TimelineValue_ = 0;
	struct BitstreamSlot {
		OaVkVideoBitstream Buffer;
		OaU64 Size = 0;
		OaU64 UseValue = 0;
	};
	static constexpr OaU32 kBitstreamRingSize = 4;
	OaArray<BitstreamSlot, kBitstreamRingSize> BitstreamRing_ = {};
	OaU32 CurrentBitstreamIndex_ = 0;
	OaU32 CodedWidth_ = 0;
	OaU32 CodedHeight_ = 0;
	
	// Decoded Picture Buffer (reference frames)
	OaVkVideoDpb Dpb_;
	// Per-array-layer layout tracking. The DPB is a single VkImage with one
	// layer per slot; different slots can be in different layouts simultaneously
	// (e.g. slot 0 in SHADER_READ_ONLY_OPTIMAL after conversion, slot 1 in
	// VIDEO_DECODE_DPB_KHR after decode). A single DpbImageLayout_ variable
	// caused the decode barrier to use the wrong oldLayout on slot reuse.
	OaArray<VkImageLayout, 16> DpbImageLayouts_ = {};

	// Resource path selected at Create() — see OaVideoResourcePath.
	OaVideoResourcePath ResourcePath_ = OaVideoResourcePath::Unavailable;

	// Per-slot NV12 staging images for plane-shader convert (Intel coincide path).
	// Legacy booleans derived from ResourcePath_ for existing code paths.
	bool UseSampleStaging_ = false;
	bool CopySampleStagingOnVideoQueue_ = false;
	OaVec<VkImage> SampleImages_;
	OaVec<VkImageView> SampleYViews_;
	OaVec<VkImageView> SampleUvViews_;
	OaVec<void*> SampleAllocations_;
	OaArray<VkImageLayout, 16> SampleImageLayouts_ = {};

	// Decode destination frames. Kept distinct from DPB when the driver supports it.
	OaVec<VkImage> OutputImages_;
	OaVec<VkImageView> OutputViews_;
	OaVec<void*> OutputAllocations_;
	OaArray<VkImageLayout, 16> OutputImageLayouts_ = {};
	// Distinct decode-output images may be sampled asynchronously while video
	// decode advances through other DPB slots. Reusing a particular output
	// image waits on the compute timeline value that last sampled that slot.
	OaArray<VkSemaphore, 16> OutputReuseSemaphores_ = {};
	OaArray<OaU64, 16> OutputReuseValues_ = {};

	// Converted RGBA frames owned by the decoder for preview/playback tests.
	OaVec<VkImage> RgbImages_;
	OaVec<VkImageView> RgbViews_;
	OaVec<void*> RgbAllocations_;
	OaVec<VkImageLayout> RgbImageLayouts_;
	
	// Phase 2.4.2: Cache parameter sets by ID
	OaHashMap<OaU32, OaH264SpsData> SpsCache_;
	OaHashMap<OaU32, OaH264PpsData> PpsCache_;
	OaHashMap<OaU32, OaH265VpsData> H265VpsCache_;
	OaHashMap<OaU32, OaH265SpsData> H265SpsCache_;
	OaHashMap<OaU32, OaH265PpsData> H265PpsCache_;
	
	// Phase 2.4: DPB slot tracking
	OaArray<DpbSlot, 16> DpbSlots_;
	OaI32 LastAllocatedDpbSlot_ = -1;  // round-robin cursor for AllocateDpbSlot
	// Tracks whether each DPB slot is currently *activated* in the Vulkan video
	// session (i.e. a prior decode set it up as a reconstructed picture since the
	// last session reset). The host DPB (DpbSlots_) releases non-reference frames
	// eagerly, but the device keeps a slot active until reset, so the two diverge.
	// When re-setting-up an already-active slot, BeginVideoCoding must bind it with
	// its real slotIndex; using -1 (the first-activation convention) makes the
	// driver treat the slot's picture resource as unowned and tears down other
	// slots' associations (VUID-07151 → 07239). Cleared on RESET / session reset.
	OaArray<bool, 16> SlotDeviceActivated_ = {};
	OaU32 DpbSlotCapacity_ = 0;
	OaU32 OutputFrameCapacity_ = 0;
	OaU32 SessionParameterUpdateCount_ = 0;
	OaArray<bool, 32> H264SpsUploaded_ = {};
	OaArray<bool, 256> H264PpsUploaded_ = {};
	OaArray<bool, 16> H265VpsUploaded_ = {};
	OaArray<bool, 32> H265SpsUploaded_ = {};
	OaArray<bool, 256> H265PpsUploaded_ = {};
	bool Av1SequenceHeaderUploaded_ = false;
	OaU64 CurrentFrameNumber_ = 0;
	// Set after the first vkCmdControlVideoCodingKHR(RESET_BIT) is issued
	// inside a BeginVideoCoding block. Vulkan Video sessions must be reset
	// once before any decode (VUID-vkCmdDecodeVideoKHR-None-07011); skipping
	// this makes the driver emit undefined output. Cleared on Flush().
	bool VideoSessionInitialized_ = false;

	// POC reconstruction state for H.264 type-0 (§8.2.1.1). Tracks the
	// previous picture's MSB+LSB so successive slices can recover the full
	// PicOrderCntVal when the LSB wraps. Reset on IDR + on Flush.
	OaI32 PrevPocLsb_ = 0;
	OaI32 PrevPocMsb_ = 0;

	// HEVC POC reconstruction is decoder state, not parser state. The slice
	// carries only an LSB; DPB lookup requires the full value across wraps.
	OaI32 H265PrevPocLsb_ = 0;
	OaI32 H265PrevPocMsb_ = 0;
	bool H265HasPrevPoc_ = false;

	// Current H.264 slice frame_num + log2_max_frame_num (set per slice before
	// ApplySlidingWindow so eviction picks the oldest *short-term* reference
	// by FrameNumWrap per §8.2.5.3, not by our monotonic decode counter
	// (the two differ in B-pyramid encodings).
	OaU32 CurrentH264FrameNum_      = 0;
	OaU32 CurrentLog2MaxFrameNum_   = 4;

	// VP9 reference buffer pool (8 logical slots → DPB array layers).
	OaArray<OaI32, STD_VIDEO_VP9_NUM_REF_FRAMES> Vp9BufferToDpbSlot_ = {};
	OaArray<VkExtent2D, STD_VIDEO_VP9_NUM_REF_FRAMES> Vp9BufferExtents_ = {};

	// AV1 reference frame pool (8 logical ref frames → DPB array layers).
	// ref_frame_idx[7] from the frame header are indices into this table (0..7).
	// refresh_frame_flags from the frame header tells us which entries to
	// point at the just-decoded DPB slot. Mirrors the VP9 scheme.
	OaArray<OaI32, STD_VIDEO_AV1_NUM_REF_FRAMES> Av1RefFrameToDpbSlot_ = {};
	OaArray<StdVideoDecodeAV1ReferenceInfo, 16> Av1DpbReferenceInfos_ = {};

	// YCbCr conversion resources (created on-demand)
	VkSamplerYcbcrConversion YcbcrConversion_ = VK_NULL_HANDLE;
	VkSampler YcbcrSampler_ = VK_NULL_HANDLE;
	VkSampler YcbcrSamplerNearest_ = VK_NULL_HANDLE;
	VkPipeline ConversionPipeline_ = VK_NULL_HANDLE;  // Compute shader fallback

	// Cached one-frame RGBA target reused across decodes — see AcquireConvertedRgbaTarget.
	OaI32 ReusedRgbaIndex_ = -1;
	OaU32 ReusedRgbaWidth_ = 0;
	OaU32 ReusedRgbaHeight_ = 0;
	// Per-layer NV12 plane views. The DPB has up to 16 layers, so we keep a
	// fixed-size cache and tie validity to the source VkImage (any reset
	// destroys these in Destroy()).
	VkImage CachedNv12Image_ = VK_NULL_HANDLE;
	OaArray<VkImageView, 16> CachedNv12YViews_ = {};
	OaArray<VkImageView, 16> CachedNv12UvViews_ = {};
	VkSampler CachedNv12Sampler_ = VK_NULL_HANDLE;
	VkSampler CachedNv12SamplerNearest_ = VK_NULL_HANDLE;
	
	OaVideoProfile Profile_ = {};
	class OaEngine *Rt_ = nullptr;
};

// Video frame resource pool (reuse decoded frames)
class OaVideoFramePool {
public:

	OaVideoFramePool(OaVideoFramePool&& InOther) noexcept;
	OaVideoFramePool& operator=(OaVideoFramePool&& InOther) noexcept;
	OaVideoFramePool(const OaVideoFramePool&) = delete;
	OaVideoFramePool& operator=(const OaVideoFramePool&) = delete;
	~OaVideoFramePool();

	static OaResult<OaVideoFramePool> Create(
		OaEngine &InRt,
		OaU32 InWidth,
		OaU32 InHeight,
		OaU32 InPoolSize
	);

	// Acquire frame from pool (blocks if all in use)
	OaVideoFrame Acquire();

	// Return frame to pool
	void Release(const OaVideoFrame &InFrame);

	void Destroy();

private:
	OaVideoFramePool() = default;
	void MoveFrom(OaVideoFramePool&& InOther) noexcept;

	OaVec<OaVideoFrame> Frames_;
	OaVec<bool> InUse_;
	OaVec<void*> Allocations_;
	OaEngine *Rt_ = nullptr;
};
