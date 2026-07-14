// OA Runtime — Vulkan Video Core (Layer 1)
// Wraps VkVideoSessionKHR + memory + parameters + DPB + bitstream + queue
// This layer owns all Vk handle lifetime and provides the shared plumbing
// for both OaVideoDecoder and OaVideoEncoder.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/OaVk.h>

// OaVkVideoSession — wraps VkVideoSessionKHR + memory bindings
// Owns the session handle and all VMA allocations bound via vkBindVideoSessionMemoryKHR
class OaVkVideoSession {
public:
	static OaResult<OaVkVideoSession> Create(
		class OaComputeEngine& InRt,
		const VkVideoProfileInfoKHR& InProfile,
		const VkExtent2D& InCodedExtent,
		VkFormat InPictureFormat,
		VkFormat InReferenceFormat,
		OaU32 InMaxDpbSlots,
		OaU32 InMaxActiveReferences,
		OaU32 InEncodeQualityLevel = 0U);

	OaVkVideoSession(OaVkVideoSession&&) noexcept;
	OaVkVideoSession& operator=(OaVkVideoSession&&) noexcept;
	OaVkVideoSession(const OaVkVideoSession&) = delete;
	~OaVkVideoSession();

	[[nodiscard]] VkVideoSessionKHR Handle() const noexcept { return Session_; }
	[[nodiscard]] OaU32 GetMaxDpbSlots() const noexcept { return MaxDpbSlots_; }
	[[nodiscard]] OaU32 GetMaxActiveReferences() const noexcept { return MaxActiveReferences_; }
	[[nodiscard]] VkExtent2D GetCodedExtent() const noexcept { return CodedExtent_; }

	void Destroy();

public:
	OaVkVideoSession() = default;
private:
	void MoveFrom(OaVkVideoSession&& InOther) noexcept;

	class OaComputeEngine* Rt_ = nullptr;
	VkVideoSessionKHR Session_ = VK_NULL_HANDLE;
	OaVec<void*> Allocations_;  // VMA blocks bound via vkBindVideoSessionMemoryKHR
	VkExtent2D CodedExtent_ = {0, 0};
	OaU32 MaxDpbSlots_ = 0;
	OaU32 MaxActiveReferences_ = 0;
};

// OaVkVideoParameters — wraps VkVideoSessionParametersKHR + cached SPS/PPS/VPS bytes
// Decoder parses these from the bitstream; encoder manufactures them.
class OaVkVideoParameters {
public:
	static OaResult<OaVkVideoParameters> Create(
		class OaComputeEngine& InRt,
		VkVideoSessionKHR InSession,
		const VkVideoSessionParametersCreateInfoKHR& InCreateInfo);

	OaVkVideoParameters(OaVkVideoParameters&&) noexcept;
	OaVkVideoParameters& operator=(OaVkVideoParameters&&) noexcept;
	OaVkVideoParameters(const OaVkVideoParameters&) = delete;
	~OaVkVideoParameters();

	[[nodiscard]] VkVideoSessionParametersKHR Handle() const noexcept { return Params_; }
	void SetHandle(VkVideoSessionParametersKHR InHandle) noexcept { Params_ = InHandle; }

	// Transfer ownership of an already-created VkVideoSessionParametersKHR
	// handle into this wrapper. Required when the handle was produced via
	// vkCreateVideoSessionParametersKHR directly (legacy callsites) so that
	// Destroy() can later call vkDestroyVideoSessionParametersKHR with the
	// right device.
	void Attach(class OaComputeEngine& InRt, VkVideoSessionParametersKHR InHandle) noexcept
	{
		Rt_ = &InRt;
		Params_ = InHandle;
	}

	void Destroy();

public:
	OaVkVideoParameters() = default;
private:
	void MoveFrom(OaVkVideoParameters&& InOther) noexcept;

	class OaComputeEngine* Rt_ = nullptr;
	VkVideoSessionParametersKHR Params_ = VK_NULL_HANDLE;
};

// OaVkVideoDpb — wraps the DPB array image + per-slot views + slot management
// Hides the "layer N of an array image" plumbing from both decoder and encoder
class OaVkVideoDpb {
public:
	struct CreateInfo {
		VkVideoProfileInfoKHR Profile;
		VkFormat Format;
		VkExtent2D CodedExtent;
		OaU32 MaxDpbSlots;
		VkImageUsageFlags Usage;
	};

	static OaResult<OaVkVideoDpb> Create(
		class OaComputeEngine& InRt,
		const CreateInfo& InInfo);

	OaVkVideoDpb(OaVkVideoDpb&&) noexcept;
	OaVkVideoDpb& operator=(OaVkVideoDpb&&) noexcept;
	OaVkVideoDpb(const OaVkVideoDpb&) = delete;
	~OaVkVideoDpb();

	[[nodiscard]] VkImage GetImage() const noexcept { return Image_; }
	[[nodiscard]] VkImageView GetView() const noexcept { return View_; }
	[[nodiscard]] OaU32 GetSlotCapacity() const noexcept { return SlotCapacity_; }

	void Destroy();

public:
	OaVkVideoDpb() = default;
private:
	void MoveFrom(OaVkVideoDpb&& InOther) noexcept;

	class OaComputeEngine* Rt_ = nullptr;
	VkImage Image_ = VK_NULL_HANDLE;
	VkImageView View_ = VK_NULL_HANDLE;
	void* Allocation_ = nullptr;
	OaU32 SlotCapacity_ = 0;
};

// OaVkVideoBitstream — wraps VMA buffer + offset/size alignment helpers
// Decoder: CPU_TO_GPU staging buffer
// Encoder: GPU_TO_CPU mapped buffer for readback
class OaVkVideoBitstream {
public:
	enum class Direction {
		Decoder,  // CPU_TO_GPU staging
		Encoder   // GPU_TO_CPU mapped
	};

	static OaResult<OaVkVideoBitstream> Create(
		class OaComputeEngine& InRt,
		OaU64 InSize,
		Direction InDirection,
		OaU64 InOffsetAlignment = 1,
		OaU64 InSizeAlignment = 1,
		// When non-null, attaches a VkVideoProfileListInfoKHR to the buffer
		// create so the driver knows what video profile this bitstream is
		// for. Validation layer requires this for video-decode-src/encode-dst
		// buffers; without it: VUID-VkBufferCreateInfo-usage-04813 +
		// VUID-vkCmdDecodeVideoKHR-pDecodeInfo-07135.
		const VkVideoProfileInfoKHR* InProfile = nullptr);

	OaVkVideoBitstream(OaVkVideoBitstream&&) noexcept;
	OaVkVideoBitstream& operator=(OaVkVideoBitstream&&) noexcept;
	OaVkVideoBitstream(const OaVkVideoBitstream&) = delete;
	~OaVkVideoBitstream();

	[[nodiscard]] VkBuffer GetBuffer() const noexcept { return Buffer_; }
	[[nodiscard]] void* GetMappedPtr() const noexcept { return MappedPtr_; }
	[[nodiscard]] void* GetAllocation() const noexcept { return Allocation_; }
	[[nodiscard]] OaU64 GetCapacity() const noexcept { return Capacity_; }
	[[nodiscard]] OaU64 GetOffsetAlignment() const noexcept { return OffsetAlignment_; }
	[[nodiscard]] OaU64 GetSizeAlignment() const noexcept { return SizeAlignment_; }

	OaStatus Resize(OaU64 InNewSize);

	void Destroy();

public:
	OaVkVideoBitstream() = default;
private:
	void MoveFrom(OaVkVideoBitstream&& InOther) noexcept;

	class OaComputeEngine* Rt_ = nullptr;
	VkBuffer Buffer_ = VK_NULL_HANDLE;
	void* Allocation_ = nullptr;
	void* MappedPtr_ = nullptr;
	OaU64 Capacity_ = 0;
	OaU64 OffsetAlignment_ = 1;
	OaU64 SizeAlignment_ = 1;
	Direction Direction_;
};

// OaVkVideoQueue — single owner of video queue handle + family + command pool + fence pool
// Eliminates the queue-handle-not-fetched class of bug permanently
class OaVkVideoQueue {
public:
	enum class QueueType {
		Decode,
		Encode
	};

	static OaResult<OaVkVideoQueue> Create(
		class OaComputeEngine& InRt,
		QueueType InType);

	OaVkVideoQueue(OaVkVideoQueue&&) noexcept;
	OaVkVideoQueue& operator=(OaVkVideoQueue&&) noexcept;
	OaVkVideoQueue(const OaVkVideoQueue&) = delete;
	~OaVkVideoQueue();

	[[nodiscard]] VkQueue GetQueue() const noexcept { return Queue_; }
	[[nodiscard]] OaU32 GetQueueFamilyIndex() const noexcept { return QueueFamilyIndex_; }
	[[nodiscard]] VkCommandPool GetCommandPool() const noexcept { return CmdPool_; }
	void SetCommandPool(VkCommandPool InPool) noexcept { CmdPool_ = InPool; }

	// Allocate a command buffer from the pool
	OaResult<VkCommandBuffer> AllocateCommandBuffer();

	// Allocate a fence (simple pool - creates new fence each call, caller owns)
	OaResult<VkFence> AllocateFence();

	void Destroy();

public:
	OaVkVideoQueue() = default;
private:
	void MoveFrom(OaVkVideoQueue&& InOther) noexcept;

	class OaComputeEngine* Rt_ = nullptr;
	VkQueue Queue_ = VK_NULL_HANDLE;
	OaU32 QueueFamilyIndex_ = 0;
	VkCommandPool CmdPool_ = VK_NULL_HANDLE;
	QueueType Type_;
};

// OaVkVideoFormat — capability query + format negotiation helpers
// Same code path supports decode and encode caps (different pNext chains)
class OaVkVideoFormat {
public:
	// Query video capabilities for a codec (decode or encode)
	static OaStatus QueryCapabilities(
		class OaComputeEngine& InRt,
		const VkVideoProfileInfoKHR& InProfile,
		bool InIsEncode,
		VkVideoCapabilitiesKHR& OutCapabilities);

	// Query supported formats for a given usage
	static OaStatus QueryFormats(
		class OaComputeEngine& InRt,
		const VkVideoProfileInfoKHR& InProfile,
		VkImageUsageFlags InUsage,
		OaVec<VkVideoFormatPropertiesKHR>& OutFormats);

	// Check if a format is supported with specific usage
	static bool HasFormatWithUsage(
		const OaVec<VkVideoFormatPropertiesKHR>& InFormats,
		VkFormat InFormat,
		VkImageUsageFlags InUsage);

	// Find format properties for a specific format
	static const VkVideoFormatPropertiesKHR* FindFormatWithUsage(
		const OaVec<VkVideoFormatPropertiesKHR>& InFormats,
		VkFormat InFormat,
		VkImageUsageFlags InUsage);
};
