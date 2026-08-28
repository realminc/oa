#pragma once

#include <oa/vision/videoDecoder.h>
#include <oa/runtime/oaVkVideo.h>
#include <oa/runtime/sync.h>

namespace oa { class VideoCodecParser; }

struct oa::VideoDecoder::DpbSlot {
	bool inUse = false;
	oa::I32 picOrderCnt = -1;
	oa::U64 frameNumber = 0U;
	oa::U32 h264FrameNum = 0U;
	bool isReference = false;
	bool isLongTerm = false;
};

struct oa::VideoDecoder::VideoCmdSlot {
	VkCommandBuffer cb = VK_NULL_HANDLE;
	VkFence fence = VK_NULL_HANDLE;
	oa::Status status = oa::Status::ok();
};

struct oa::VideoDecoder::BitstreamSlot {
	oavk::VideoBitstream buffer;
	oa::U64 size = 0U;
	oa::U64 useValue = 0U;
};

class oa::VideoDecoder::Impl {
public:
	~Impl();

	oa::UniquePtr<VideoCodecParser> parser;
	oavk::VideoSession session;
	oavk::VideoParameters sessionParameters;
	oavk::VideoQueue queue;
	oa::Array<VkCommandBuffer, kCmdBufferCount> commandBuffers = {};
	oa::Array<VkFence, kCmdBufferCount> commandFences = {};
	oa::U32 currentCommandBufferIndex = 0U;
	oavk::TimelineSemaphore timelineSemaphore;
	oa::U64 timelineValue = 0U;
	oa::Array<BitstreamSlot, kBitstreamRingSize> bitstreamRing = {};
	oa::U32 currentBitstreamIndex = 0U;
	oa::U32 codedWidth = 0U;
	oa::U32 codedHeight = 0U;
	oavk::VideoDpb dpb;
	oa::Array<VkImageLayout, 16> dpbImageLayouts = {};
	oa::VideoResourcePath resourcePath = oa::VideoResourcePath::Unavailable;

	bool useSampleStaging = false;
	bool copySampleStagingOnVideoQueue = false;
	oa::Vector<VkImage> sampleImages;
	oa::Vector<VkImageView> sampleYViews;
	oa::Vector<VkImageView> sampleUvViews;
	oa::Vector<void*> sampleAllocations;
	oa::Array<VkImageLayout, 16> sampleImageLayouts = {};

	oa::Vector<VkImage> outputImages;
	oa::Vector<VkImageView> outputViews;
	oa::Vector<void*> outputAllocations;
	oa::Array<VkImageLayout, 16> outputImageLayouts = {};
	oa::Array<VkSemaphore, 16> outputReuseSemaphores = {};
	oa::Array<oa::U64, 16> outputReuseValues = {};

	oa::Vector<VkImage> rgbImages;
	oa::Vector<VkImageView> rgbViews;
	oa::Vector<void*> rgbAllocations;
	oa::Vector<VkImageLayout> rgbImageLayouts;

	oa::HashMap<oa::U32, H264SpsData> spsCache;
	oa::HashMap<oa::U32, H264PpsData> ppsCache;
	oa::HashMap<oa::U32, H265VpsData> h265VpsCache;
	oa::HashMap<oa::U32, H265SpsData> h265SpsCache;
	oa::HashMap<oa::U32, H265PpsData> h265PpsCache;

	oa::Array<DpbSlot, 16> dpbSlots = {};
	oa::I32 lastAllocatedDpbSlot = -1;
	oa::Array<bool, 16> slotDeviceActivated = {};
	oa::U32 dpbSlotCapacity = 0U;
	oa::U32 outputFrameCapacity = 0U;
	oa::U32 sessionParameterUpdateCount = 0U;
	oa::Array<bool, 32> h264SpsUploaded = {};
	oa::Array<bool, 256> h264PpsUploaded = {};
	oa::Array<bool, 16> h265VpsUploaded = {};
	oa::Array<bool, 32> h265SpsUploaded = {};
	oa::Array<bool, 256> h265PpsUploaded = {};
	bool av1SequenceHeaderUploaded = false;
	oa::U64 currentFrameNumber = 0U;
	bool videoSessionInitialized = false;

	oa::I32 previousPocLsb = 0;
	oa::I32 previousPocMsb = 0;
	oa::I32 h265PreviousPocLsb = 0;
	oa::I32 h265PreviousPocMsb = 0;
	bool h265HasPreviousPoc = false;
	oa::U32 currentH264FrameNumber = 0U;
	oa::U32 currentLog2MaxFrameNumber = 4U;

	oa::Array<oa::I32, STD_VIDEO_VP9_NUM_REF_FRAMES> vp9BufferToDpbSlot = {};
	oa::Array<VkExtent2D, STD_VIDEO_VP9_NUM_REF_FRAMES> vp9BufferExtents = {};
	oa::Array<oa::I32, STD_VIDEO_AV1_NUM_REF_FRAMES> av1RefFrameToDpbSlot = {};
	oa::Array<StdVideoDecodeAV1ReferenceInfo, 16> av1DpbReferenceInfos = {};

	VkSamplerYcbcrConversion ycbcrConversion = VK_NULL_HANDLE;
	VkSampler ycbcrSampler = VK_NULL_HANDLE;
	VkSampler ycbcrSamplerNearest = VK_NULL_HANDLE;
	VkPipeline conversionPipeline = VK_NULL_HANDLE;

	oa::I32 reusedRgbaIndex = -1;
	oa::U32 reusedRgbaWidth = 0U;
	oa::U32 reusedRgbaHeight = 0U;
	VkImage cachedNv12Image = VK_NULL_HANDLE;
	oa::Array<VkImageView, 16> cachedNv12YViews = {};
	oa::Array<VkImageView, 16> cachedNv12UvViews = {};
	VkSampler cachedNv12Sampler = VK_NULL_HANDLE;
	VkSampler cachedNv12SamplerNearest = VK_NULL_HANDLE;

	oa::VideoProfile profile = {};
	oa::Engine* engine = nullptr;
};
