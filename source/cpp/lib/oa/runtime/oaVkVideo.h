// OA Runtime — vulkan Video core (layer 1)
// Wraps VkVideoSessionKHR + memory + parameters + DPB + bitstream + queue
// This layer owns all Vk handle lifetime and provides the shared plumbing
// for both oa::VideoDecoder and oa::VideoEncoder.

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/runtime/oaVk.h>

namespace oa { class Engine; }

namespace oavk {

// VideoSession — wraps VkVideoSessionKHR + memory bindings
// Owns the session handle and all VMA allocations bound via vkBindVideoSessionMemoryKHR
class VideoSession {
public:
	static oa::Result<VideoSession> create(
		oa::Engine& inRt,
		const VkVideoProfileInfoKHR& inProfile,
		const VkExtent2D& inCodedExtent,
		VkFormat inPictureFormat,
		VkFormat inReferenceFormat,
		oa::U32 inMaxDpbSlots,
		oa::U32 inMaxActiveReferences,
		oa::U32 inEncodeQualityLevel = 0U);

	VideoSession(VideoSession&&) noexcept;
	VideoSession& operator=(VideoSession&&) noexcept;
	VideoSession(const VideoSession&) = delete;
	~VideoSession();

	[[nodiscard]] VkVideoSessionKHR handle() const noexcept { return session_; }
	[[nodiscard]] oa::U32 getMaxDpbSlots() const noexcept { return maxDpbSlots_; }
	[[nodiscard]] oa::U32 getMaxActiveReferences() const noexcept { return maxActiveReferences_; }
	[[nodiscard]] VkExtent2D getCodedExtent() const noexcept { return codedExtent_; }

	void destroy();

public:
	VideoSession() = default;
private:
	void moveFrom(VideoSession&& inOther) noexcept;

	oa::Engine* rt_ = nullptr;
	VkVideoSessionKHR session_ = VK_NULL_HANDLE;
	oa::Vec<void*> allocations_;  // VMA blocks bound via vkBindVideoSessionMemoryKHR
	VkExtent2D codedExtent_ = {0, 0};
	oa::U32 maxDpbSlots_ = 0;
	oa::U32 maxActiveReferences_ = 0;
};

// VideoParameters — wraps VkVideoSessionParametersKHR + cached SPS/PPS/VPS bytes
// Decoder parses these from the bitstream; encoder manufactures them.
class VideoParameters {
public:
	static oa::Result<VideoParameters> create(
		oa::Engine& inRt,
		VkVideoSessionKHR inSession,
		const VkVideoSessionParametersCreateInfoKHR& inCreateInfo);

	VideoParameters(VideoParameters&&) noexcept;
	VideoParameters& operator=(VideoParameters&&) noexcept;
	VideoParameters(const VideoParameters&) = delete;
	~VideoParameters();

	[[nodiscard]] VkVideoSessionParametersKHR handle() const noexcept { return params_; }

	// Transfer ownership of an already-created VkVideoSessionParametersKHR
	// handle into this wrapper. Required when the handle was produced via
	// vkCreateVideoSessionParametersKHR directly (legacy callsites) so that
	// destroy() can later call vkDestroyVideoSessionParametersKHR with the
	// right device.
	void attach(oa::Engine& inRt, VkVideoSessionParametersKHR inHandle) noexcept
	{
		rt_ = &inRt;
		params_ = inHandle;
	}

	void destroy();

public:
	VideoParameters() = default;
private:
	void moveFrom(VideoParameters&& inOther) noexcept;

	oa::Engine* rt_ = nullptr;
	VkVideoSessionParametersKHR params_ = VK_NULL_HANDLE;
};

// VideoDpb — wraps the DPB array image + per-slot views + slot management
// Hides the "layer N of an array image" plumbing from both decoder and encoder
class VideoDpb {
public:
	struct CreateInfo {
		VkVideoProfileInfoKHR profile;
		VkFormat format;
		VkExtent2D codedExtent;
		oa::U32 maxDpbSlots;
		VkImageUsageFlags usage;
	};

	static oa::Result<VideoDpb> create(
		oa::Engine& inRt,
		const CreateInfo& inInfo);

	VideoDpb(VideoDpb&&) noexcept;
	VideoDpb& operator=(VideoDpb&&) noexcept;
	VideoDpb(const VideoDpb&) = delete;
	~VideoDpb();

	[[nodiscard]] VkImage getImage() const noexcept { return image_; }
	[[nodiscard]] VkImageView getView() const noexcept { return view_; }
	[[nodiscard]] oa::U32 getSlotCapacity() const noexcept { return slotCapacity_; }

	void destroy();

public:
	VideoDpb() = default;
private:
	void moveFrom(VideoDpb&& inOther) noexcept;

	oa::Engine* rt_ = nullptr;
	VkImage image_ = VK_NULL_HANDLE;
	VkImageView view_ = VK_NULL_HANDLE;
	void* allocation_ = nullptr;
	oa::U32 slotCapacity_ = 0;
};

// VideoBitstream — wraps VMA buffer + offset/size alignment helpers
// Decoder: CPU_TO_GPU staging buffer
// encoder: GPU_TO_CPU mapped buffer for readback
class VideoBitstream {
public:
	enum class Direction {
		Decoder,  // CPU_TO_GPU staging
		Encoder,  // GPU_TO_CPU mapped
	};

	static oa::Result<VideoBitstream> create(
		oa::Engine& inRt,
		oa::U64 inSize,
		Direction inDirection,
		oa::U64 inOffsetAlignment = 1,
		oa::U64 inSizeAlignment = 1,
		// When non-null, attaches a VkVideoProfileListInfoKHR to the buffer
		// create so the driver knows what video profile this bitstream is
		// for. Validation layer requires this for video-decode-src/encode-dst
		// buffers; without it: VUID-VkBufferCreateInfo-usage-04813 +
		// VUID-vkCmdDecodeVideoKHR-pDecodeInfo-07135.
		const VkVideoProfileInfoKHR* inProfile = nullptr);

	VideoBitstream(VideoBitstream&&) noexcept;
	VideoBitstream& operator=(VideoBitstream&&) noexcept;
	VideoBitstream(const VideoBitstream&) = delete;
	~VideoBitstream();

	[[nodiscard]] VkBuffer getBuffer() const noexcept { return buffer_; }
	[[nodiscard]] void* getMappedPtr() const noexcept { return mappedPtr_; }
	[[nodiscard]] void* getAllocation() const noexcept { return allocation_; }
	[[nodiscard]] oa::U64 getCapacity() const noexcept { return capacity_; }
	[[nodiscard]] oa::U64 getOffsetAlignment() const noexcept { return offsetAlignment_; }
	[[nodiscard]] oa::U64 getSizeAlignment() const noexcept { return sizeAlignment_; }

	oa::Status resize(oa::U64 inNewSize);

	void destroy();

public:
	VideoBitstream() = default;
private:
	void moveFrom(VideoBitstream&& inOther) noexcept;

	oa::Engine* rt_ = nullptr;
	VkBuffer buffer_ = VK_NULL_HANDLE;
	void* allocation_ = nullptr;
	void* mappedPtr_ = nullptr;
	oa::U64 capacity_ = 0;
	oa::U64 offsetAlignment_ = 1;
	oa::U64 sizeAlignment_ = 1;
	Direction direction_;
};

// VideoQueue — single owner of video queue handle + family + command pool + fence pool
// Eliminates the queue-handle-not-fetched class of bug permanently
class VideoQueue {
public:
	enum class QueueType {
		Decode,
		Encode
	};

	static oa::Result<VideoQueue> create(
		oa::Engine& inRt,
		QueueType inType);

	VideoQueue(VideoQueue&&) noexcept;
	VideoQueue& operator=(VideoQueue&&) noexcept;
	VideoQueue(const VideoQueue&) = delete;
	~VideoQueue();

	[[nodiscard]] VkQueue getQueue() const noexcept { return queue_; }
	[[nodiscard]] oa::U32 getQueueFamilyIndex() const noexcept { return queueFamilyIndex_; }
	[[nodiscard]] VkCommandPool getCommandPool() const noexcept { return cmdPool_; }
	void setCommandPool(VkCommandPool inPool) noexcept { cmdPool_ = inPool; }

	// allocate a command buffer from the pool
	oa::Result<VkCommandBuffer> allocateCommandBuffer();

	// allocate a fence (simple pool - creates new fence each call, caller owns)
	oa::Result<VkFence> allocateFence();

	void destroy();

public:
	VideoQueue() = default;
private:
	void moveFrom(VideoQueue&& inOther) noexcept;

	oa::Engine* rt_ = nullptr;
	VkQueue queue_ = VK_NULL_HANDLE;
	oa::U32 queueFamilyIndex_ = 0;
	VkCommandPool cmdPool_ = VK_NULL_HANDLE;
	QueueType type_;
};

// VideoFormat — capability query + format negotiation helpers
// Same code path supports decode and encode caps (different pNext chains)
class VideoFormat {
public:
	// query video capabilities for a codec (decode or encode)
	static oa::Status queryCapabilities(
		oa::Engine& inRt,
		const VkVideoProfileInfoKHR& inProfile,
		bool inIsEncode,
		VkVideoCapabilitiesKHR& outCapabilities);

	// query supported formats for a given usage
	static oa::Status queryFormats(
		oa::Engine& inRt,
		const VkVideoProfileInfoKHR& inProfile,
		VkImageUsageFlags inUsage,
		oa::Vec<VkVideoFormatPropertiesKHR>& outFormats);

	// Check if a format is supported with specific usage
	static bool hasFormatWithUsage(
		const oa::Vec<VkVideoFormatPropertiesKHR>& inFormats,
		VkFormat inFormat,
		VkImageUsageFlags inUsage);

	// find format properties for a specific format
	static const VkVideoFormatPropertiesKHR* findFormatWithUsage(
		const oa::Vec<VkVideoFormatPropertiesKHR>& inFormats,
		VkFormat inFormat,
		VkImageUsageFlags inUsage);
};

} // namespace oavk
