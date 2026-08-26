#include <assert.h>

#include <oa/runtime/allocator.h>
#include <oa/runtime/device.h>
#include <oa/runtime/oaVk.h>
#include <oa/runtime/oaVma.h>
#include <oa/core/memory.h>

namespace {

constexpr oa::U64 transferCapacity(oa::U64 inSize) {
	return (oa::max<oa::U64>(inSize, 1ULL) + 3ULL) & ~3ULL;
}

} // namespace

static OaVmaVulkanFunctions getOaVkFunctions(const oavk::Device& inDevice) {
	OaVmaVulkanFunctions fns{};
	fns.vkGetInstanceProcAddr = oaVkGetInstanceProcAddr();
	fns.vkGetDeviceProcAddr = inDevice.instanceDispatch.vkGetDeviceProcAddr;
	fns.vkGetPhysicalDeviceProperties = inDevice.instanceDispatch.vkGetPhysicalDeviceProperties;
	fns.vkGetPhysicalDeviceMemoryProperties = inDevice.instanceDispatch.vkGetPhysicalDeviceMemoryProperties;
	fns.vkAllocateMemory = inDevice.deviceDispatch.vkAllocateMemory;
	fns.vkFreeMemory = inDevice.deviceDispatch.vkFreeMemory;
	fns.vkMapMemory = inDevice.deviceDispatch.vkMapMemory;
	fns.vkUnmapMemory = inDevice.deviceDispatch.vkUnmapMemory;
	fns.vkFlushMappedMemoryRanges = inDevice.deviceDispatch.vkFlushMappedMemoryRanges;
	fns.vkInvalidateMappedMemoryRanges = inDevice.deviceDispatch.vkInvalidateMappedMemoryRanges;
	fns.vkBindBufferMemory = inDevice.deviceDispatch.vkBindBufferMemory;
	fns.vkBindImageMemory = inDevice.deviceDispatch.vkBindImageMemory;
	fns.vkGetBufferMemoryRequirements = inDevice.deviceDispatch.vkGetBufferMemoryRequirements;
	fns.vkGetImageMemoryRequirements = inDevice.deviceDispatch.vkGetImageMemoryRequirements;
	fns.vkCreateBuffer = inDevice.deviceDispatch.vkCreateBuffer;
	fns.vkDestroyBuffer = inDevice.deviceDispatch.vkDestroyBuffer;
	fns.vkCreateImage = inDevice.deviceDispatch.vkCreateImage;
	fns.vkDestroyImage = inDevice.deviceDispatch.vkDestroyImage;
	fns.vkCmdCopyBuffer = inDevice.deviceDispatch.vkCmdCopyBuffer;
	fns.vkGetBufferMemoryRequirements2KHR = inDevice.deviceDispatch.vkGetBufferMemoryRequirements2;
	fns.vkGetImageMemoryRequirements2KHR = inDevice.deviceDispatch.vkGetImageMemoryRequirements2;
	fns.vkBindBufferMemory2KHR = inDevice.deviceDispatch.vkBindBufferMemory2;
	fns.vkBindImageMemory2KHR = inDevice.deviceDispatch.vkBindImageMemory2;
	fns.vkGetPhysicalDeviceMemoryProperties2KHR =
		inDevice.instanceDispatch.vkGetPhysicalDeviceMemoryProperties2;
	return fns;
}

oa::Result<OaVma> OaVma::create(const oavk::Device& inDevice) {
	OaVmaVulkanFunctions fns = getOaVkFunctions(inDevice);

	OaVmaAllocatorCreateInfo ci{};
	ci.vulkanApiVersion = inDevice.info.software.apiVersionPacked;
	ci.instance = static_cast<VkInstance>(inDevice.instance);
	ci.physicalDevice = static_cast<VkPhysicalDevice>(inDevice.physicalDevice);
	ci.device = static_cast<VkDevice>(inDevice.device);
	ci.pVulkanFunctions = &fns;
	// Enable buffer device address support (required for bindless + GPU compute graphs)
	// Enable KHR_maintenance5 support (required for VkBufferUsageFlags2CreateInfo)
	ci.flags = OA_VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	for (const auto& extension : inDevice.info.software.enabledDeviceExtensions) {
		if (extension == "VK_KHR_maintenance5") {
			ci.flags |= OA_VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT;
			break;
		}
	}

	OaVmaAllocator alloc = VK_NULL_HANDLE;
	VkResult r = OaVmaCreateAllocator(&ci, &alloc);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::OutOfMemory, "OaVmaCreateAllocator failed");
	}

	OaVma a;
	a.allocator = alloc;
	a.hasSam = inDevice.info.hardware.hasSAM;
	return a;
}

void OaVma::destroy() {
	if (allocator) {
		OaVmaDestroyAllocator(static_cast<OaVmaAllocator>(allocator));
		allocator = nullptr;
	}
}

oa::Result<oavk::Buffer> OaVma::allocDevice(oa::U64 inSize) {
	const oa::U64 capacity = transferCapacity(inSize);
	VkBufferCreateInfo bufCI = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = capacity,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
			| VK_BUFFER_USAGE_TRANSFER_SRC_BIT
			| VK_BUFFER_USAGE_TRANSFER_DST_BIT
			| VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
			| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
	};

	OaVmaAllocationCreateInfo allocCI = {
		.usage = OA_VMA_MEMORY_USAGE_GPU_ONLY,
	};

	VkBuffer buffer = VK_NULL_HANDLE;
	OaVmaAllocation allocation = VK_NULL_HANDLE;
	VkResult r = OaVmaCreateBuffer(
		static_cast<OaVmaAllocator>(allocator),
		&bufCI, &allocCI, &buffer, &allocation, nullptr
	);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::OutOfMemory, "device buffer allocation failed");
	}

	oavk::Buffer buf;
	buf.buffer = buffer;
	buf.allocation = allocation;
	buf.allocatorIdentity = allocator;
	buf.size = inSize;
	buf.capacity = capacity;
	buf.mappedPtr = nullptr;
	buf.flags = OA_VK_BUFFER_FLAG_INDIRECT_DISPATCH;
	buf.placement = oa::MemoryPlacement::DeviceLocal;
	buf.mutationVersion_ = oa::makeShared<oa::U64>(0U);
	return buf;
}

oa::Result<oavk::Buffer> OaVma::allocHostVisible(oa::U64 inSize) {
	const oa::U64 capacity = transferCapacity(inSize);
	VkBufferCreateInfo bufCI{};
	bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufCI.size = capacity;
	bufCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
		| VK_BUFFER_USAGE_TRANSFER_SRC_BIT 
		| VK_BUFFER_USAGE_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
		| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

	OaVmaAllocationCreateInfo allocCI{};
	allocCI.usage = OA_VMA_MEMORY_USAGE_CPU_TO_GPU;
	allocCI.flags = OA_VMA_ALLOCATION_CREATE_MAPPED_BIT | OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	// oa::Matrix exposes its mapped pointer as a first-class CPU access path. Keep
	// that contract valid for direct Data()/dataAs() users; explicit flush and
	// invalidate calls remain in transfer primitives as a second line of defence.
	allocCI.requiredFlags =
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	VkBuffer buffer = VK_NULL_HANDLE;
	OaVmaAllocation allocation = VK_NULL_HANDLE;
	OaVmaAllocationInfo allocInfo{};
	VkResult r = OaVmaCreateBuffer(
		static_cast<OaVmaAllocator>(allocator),
		&bufCI, &allocCI, &buffer, &allocation, &allocInfo);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::OutOfMemory, "host-visible buffer allocation failed");
	}

	oavk::Buffer buf;
	buf.buffer = buffer;
	buf.allocation = allocation;
	buf.allocatorIdentity = allocator;
	buf.size = inSize;
	buf.capacity = capacity;
	buf.mappedPtr = allocInfo.pMappedData;
	buf.flags = OA_VK_BUFFER_FLAG_INDIRECT_DISPATCH;
	buf.placement = oa::MemoryPlacement::HostUpload;
	buf.mutationVersion_ = oa::makeShared<oa::U64>(0U);
	return buf;
}

oa::Result<oavk::Buffer> OaVma::allocHostReadback(oa::U64 inSize) {
	const oa::U64 capacity = transferCapacity(inSize);
	VkBufferCreateInfo bufCI{};
	bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufCI.size = capacity;
	bufCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
		| VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

	OaVmaAllocationCreateInfo allocCI{};
	allocCI.usage = OA_VMA_MEMORY_USAGE_GPU_TO_CPU;
	allocCI.flags = OA_VMA_ALLOCATION_CREATE_MAPPED_BIT
		| OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	allocCI.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

	VkBuffer buffer = VK_NULL_HANDLE;
	OaVmaAllocation allocation = VK_NULL_HANDLE;
	OaVmaAllocationInfo allocInfo{};
	const VkResult result = OaVmaCreateBuffer(
		static_cast<OaVmaAllocator>(allocator),
		&bufCI, &allocCI, &buffer, &allocation, &allocInfo);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::OutOfMemory, "host-readback buffer allocation failed");
	}

	oavk::Buffer out;
	out.buffer = buffer;
	out.allocation = allocation;
	out.allocatorIdentity = allocator;
	out.size = inSize;
	out.capacity = capacity;
	out.mappedPtr = allocInfo.pMappedData;
	out.flags = OA_VK_BUFFER_FLAG_INDIRECT_DISPATCH;
	out.placement = oa::MemoryPlacement::HostReadback;
	out.mutationVersion_ = oa::makeShared<oa::U64>(0U);
	return out;
}

oa::Bool OaVma::flushHostBuffer(const oavk::Buffer& inBuf, oa::U64 inOffset, oa::U64 inSize) {
	if (not allocator or not inBuf.allocation) {
		return true;
	}
	VkResult r = OaVmaFlushAllocation(
		static_cast<OaVmaAllocator>(allocator),
		static_cast<OaVmaAllocation>(inBuf.allocation),
		inOffset, inSize);
	return r == VK_SUCCESS;
}

oa::Bool OaVma::invalidateHostBuffer(
	const oavk::Buffer& inBuf, oa::U64 inOffset, oa::U64 inSize) {
	if (not allocator or not inBuf.allocation) {
		return true;
	}
	VkResult r = OaVmaInvalidateAllocation(
		static_cast<OaVmaAllocator>(allocator),
		static_cast<OaVmaAllocation>(inBuf.allocation),
		inOffset, inSize);
	return r == VK_SUCCESS;
}

oa::Result<oavk::Buffer> OaVma::allocBar(oa::U64 inSize) {
	if (!hasSam) {
		return allocHostVisible(inSize);
	}
	const oa::U64 capacity = transferCapacity(inSize);

	// Device-local + host-visible = BAR (requires SAM / resizable BAR)
	VkBufferCreateInfo bufCI{};
	bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufCI.size = capacity;
	bufCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
		| VK_BUFFER_USAGE_TRANSFER_SRC_BIT 
		| VK_BUFFER_USAGE_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
		| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

	OaVmaAllocationCreateInfo allocCI{};
	allocCI.usage = OA_VMA_MEMORY_USAGE_AUTO;
	allocCI.flags = OA_VMA_ALLOCATION_CREATE_MAPPED_BIT
		| OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	allocCI.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

	VkBuffer buffer = VK_NULL_HANDLE;
	OaVmaAllocation allocation = VK_NULL_HANDLE;
	OaVmaAllocationInfo allocInfo{};
	VkResult r = OaVmaCreateBuffer(
		static_cast<OaVmaAllocator>(allocator),
		&bufCI, &allocCI, &buffer, &allocation, &allocInfo);
	if (r != VK_SUCCESS) {
		// BAR allocation can fail if VRAM is full; fall back to host-visible
		return allocHostVisible(inSize);
	}

	oavk::Buffer buf;
	buf.buffer = buffer;
	buf.allocation = allocation;
	buf.allocatorIdentity = allocator;
	buf.size = inSize;
	buf.capacity = capacity;
	buf.mappedPtr = allocInfo.pMappedData;
	buf.flags = OA_VK_BUFFER_FLAG_BAR
		| OA_VK_BUFFER_FLAG_INDIRECT_DISPATCH;
	buf.placement = oa::MemoryPlacement::Unified;
	buf.mutationVersion_ = oa::makeShared<oa::U64>(0U);
	return buf;
}

oa::Result<oavk::Buffer> OaVma::allocPreprocessBuffer(oa::U64 inSize) {
	const oa::U64 capacity = transferCapacity(inSize);
	// allocate buffer with VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT for VK_EXT_device_generated_commands
	// Note: VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT requires VkBufferUsageFlags2CreateInfo in pNext
	VkBufferUsageFlags2CreateInfo usageFlags2{};
	usageFlags2.sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO;
	usageFlags2.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
		| VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
		| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		| VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT;  // Required for VK_EXT preprocessing
	
	VkBufferCreateInfo bufCI{};
	bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufCI.pNext = &usageFlags2;  // Chain the 64-bit usage flags
	bufCI.size = capacity;
	bufCI.usage = 0;  // Must be 0 when using VkBufferUsageFlags2CreateInfo

	OaVmaAllocationCreateInfo allocCI{};
	if (hasSam) {
		// Use BAR if available (device-local + host-visible)
		allocCI.usage = OA_VMA_MEMORY_USAGE_AUTO;
		allocCI.flags = OA_VMA_ALLOCATION_CREATE_MAPPED_BIT
			| OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
		allocCI.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
	} else {
		// Fall back to device-local only
		allocCI.usage = OA_VMA_MEMORY_USAGE_GPU_ONLY;
	}

	VkBuffer buffer = VK_NULL_HANDLE;
	OaVmaAllocation allocation = VK_NULL_HANDLE;
	OaVmaAllocationInfo allocInfo{};
	VkResult r = OaVmaCreateBuffer(
		static_cast<OaVmaAllocator>(allocator),
		&bufCI, &allocCI, &buffer, &allocation, &allocInfo);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::OutOfMemory, "preprocess buffer allocation failed");
	}

	oavk::Buffer buf;
	buf.buffer = buffer;
	buf.allocation = allocation;
	buf.allocatorIdentity = allocator;
	buf.size = inSize;
	buf.capacity = capacity;
	buf.mappedPtr = allocInfo.pMappedData;
	buf.flags = OA_VK_BUFFER_FLAG_INDIRECT_DISPATCH;
	if (hasSam && allocInfo.pMappedData) {
		buf.flags |= OA_VK_BUFFER_FLAG_BAR;
	}
	buf.placement = (hasSam && allocInfo.pMappedData)
		? oa::MemoryPlacement::Unified
		: oa::MemoryPlacement::DeviceLocal;
	buf.mutationVersion_ = oa::makeShared<oa::U64>(0U);
	return buf;
}

oa::Status OaVma::uploadWeights(oavk::Buffer& inDst, const void* inSrc, oa::U64 inSize) {
	if (!inDst.mappedPtr) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "buffer not mapped");
	}
	if (inSize > inDst.size) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "upload size exceeds buffer");
	}

	if (inDst.isBar()) {
		oa::memcpyStream(inDst.mappedPtr, inSrc, inSize);
		oa::storeFence();
	} else {
		oa::memcpy(inDst.mappedPtr, inSrc, inSize);
	}
	inDst.markMutation();
	return oa::Status::ok();
}

oa::Result<oavk::Buffer> OaVma::allocAliased(
	oa::U64 inSize, oa::MemoryPlacement inPlacement) {
	const oa::U64 capacity = transferCapacity(inSize);
	VkBufferCreateInfo bufCI{};
	bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufCI.size = capacity;
	bufCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
		| VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

	OaVmaAllocationCreateInfo allocCI{};
	allocCI.flags = OA_VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT;
	switch (inPlacement) {
		case oa::MemoryPlacement::DeviceLocal:
			allocCI.usage = OA_VMA_MEMORY_USAGE_GPU_ONLY;
			break;
		case oa::MemoryPlacement::Unified:
			allocCI.usage = OA_VMA_MEMORY_USAGE_AUTO;
			allocCI.flags |= OA_VMA_ALLOCATION_CREATE_MAPPED_BIT
				| OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
			allocCI.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
				| VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
			break;
		case oa::MemoryPlacement::HostReadback:
			allocCI.usage = OA_VMA_MEMORY_USAGE_GPU_TO_CPU;
			allocCI.flags |= OA_VMA_ALLOCATION_CREATE_MAPPED_BIT
				| OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
			allocCI.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
			break;
		case oa::MemoryPlacement::Auto:
		case oa::MemoryPlacement::HostUpload:
			inPlacement = oa::MemoryPlacement::HostUpload;
			allocCI.usage = OA_VMA_MEMORY_USAGE_CPU_TO_GPU;
			allocCI.flags |= OA_VMA_ALLOCATION_CREATE_MAPPED_BIT
				| OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
			allocCI.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
				| VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
			break;
	}

	VkBuffer buffer = VK_NULL_HANDLE;
	OaVmaAllocation allocation = VK_NULL_HANDLE;
	OaVmaAllocationInfo allocInfo{};
	VkResult res = OaVmaCreateBuffer(
		static_cast<OaVmaAllocator>(allocator),
		&bufCI, &allocCI, &buffer, &allocation, &allocInfo
	);
	if (res != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::OutOfMemory, "aliased buffer allocation failed");
	}

	oavk::Buffer buf;
	buf.buffer = buffer;
	buf.allocation = allocation;
	buf.allocatorIdentity = allocator;
	buf.aliasIdentity = allocation;
	buf.size = inSize;
	buf.capacity = capacity;
	buf.mappedPtr = allocInfo.pMappedData;
	buf.flags = OA_VK_BUFFER_FLAG_ALIAS
		| OA_VK_BUFFER_FLAG_INDIRECT_DISPATCH;
	buf.placement = inPlacement;
	buf.mutationVersion_ = oa::makeShared<oa::U64>(0U);
	return buf;
}

oa::Result<oavk::Buffer> OaVma::createAliasingBuffer(
	const oavk::Buffer& inExisting, oa::U64 inSize
) {
	const oa::U64 capacity = transferCapacity(inSize);
	if (!inExisting.allocation) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"createAliasingBuffer: source has no allocation");
	}

	VkBufferCreateInfo bufCI{};
	bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufCI.size = capacity;
	bufCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
		| VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

	VkBuffer buffer = VK_NULL_HANDLE;
	VkResult r = OaVmaCreateAliasingBuffer2(
		static_cast<OaVmaAllocator>(allocator),
		static_cast<OaVmaAllocation>(inExisting.allocation),
		0,
		&bufCI,
		&buffer);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::OutOfMemory, "aliasing buffer creation failed");
	}

	oavk::Buffer buf;
	buf.buffer = buffer;
	buf.allocation = nullptr;
	buf.allocatorIdentity = allocator;
	buf.aliasIdentity = inExisting.aliasIdentity
		? inExisting.aliasIdentity : inExisting.allocation;
	buf.size = inSize;
	buf.capacity = capacity;
	buf.mappedPtr = inExisting.mappedPtr;
	buf.flags = OA_VK_BUFFER_FLAG_ALIAS
		| OA_VK_BUFFER_FLAG_INDIRECT_DISPATCH;
	buf.placement = inExisting.placement;
	(void)inExisting.observeMutationVersion();
	buf.mutationVersion_ = inExisting.mutationVersion_;
	return buf;
}

void OaVma::freeAlias(oavk::Buffer& inOutBuffer) {
	// OaVmaDestroyBuffer safely handles null allocation — destroys only the VkBuffer.
	free(inOutBuffer);
}

OaVmaStats OaVma::getStats() const {
	OaVmaStats stats{};
	if (!allocator) return stats;

	OaVmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
	OaVmaGetHeapBudgets(static_cast<OaVmaAllocator>(allocator), budgets);

	for (oa::U32 i = 0; i < VK_MAX_MEMORY_HEAPS; ++i) {
		if (budgets[i].budget == 0) continue;
		stats.usedBytes += budgets[i].usage;
		stats.budgetBytes += budgets[i].budget;
	}

	// Detailed accounting: distinguishes bytes actually live (allocationBytes) from bytes
	// reserved in VMA blocks (blockBytes). A large blockBytes/allocationBytes ratio means
	// fragmentation / pooled slack rather than a true leak.
	OaVmaTotalStatistics total{};
	OaVmaCalculateStatistics(static_cast<OaVmaAllocator>(allocator), &total);
	stats.allocationBytes = total.total.statistics.allocationBytes;
	stats.blockBytes      = total.total.statistics.blockBytes;
	stats.allocationCount = total.total.statistics.allocationCount;
	stats.blockCount      = total.total.statistics.blockCount;
	return stats;
}

void OaVma::free(oavk::Buffer& inOutBuffer) {
	if (inOutBuffer.buffer && allocator) {
		// Descriptor ownership belongs to oa::Engine, not VMA. Looking up an
		// ambient engine here can deregister the same numeric slot from the wrong
		// heap when more than one engine exists. Engine-owned buffers therefore
		// pass through the engine's private resource boundary, which deregisters
		// first; raw allocator users may only free buffers that were never
		// registered.
		assert(inOutBuffer.bindlessIndex == UINT32_MAX
			&& "deregister an engine-owned buffer before raw allocator free");
		if (inOutBuffer.bindlessIndex != UINT32_MAX) return;

		OaVmaDestroyBuffer(
			static_cast<OaVmaAllocator>(allocator),
			static_cast<VkBuffer>(inOutBuffer.buffer),
			static_cast<OaVmaAllocation>(inOutBuffer.allocation)
		);
		inOutBuffer.buffer = nullptr;
		inOutBuffer.allocation = nullptr;
		inOutBuffer.allocatorIdentity = nullptr;
		inOutBuffer.aliasIdentity = nullptr;
		inOutBuffer.mappedPtr = nullptr;
		inOutBuffer.size = 0;
		inOutBuffer.capacity = 0;
		inOutBuffer.flags = OA_VK_BUFFER_FLAG_NONE;
		inOutBuffer.placement = oa::MemoryPlacement::Auto;
		inOutBuffer.mutationVersion_.reset();
	}
}

oa::U64 oavk::Buffer::observeMutationVersion() const {
	if (not mutationVersion_) {
		mutationVersion_ = oa::makeShared<oa::U64>(0U);
	}
	return *mutationVersion_;
}

oa::U64 oavk::Buffer::currentMutationVersion() const noexcept {
	return mutationVersion_ ? *mutationVersion_ : 0U;
}

void oavk::Buffer::markMutation() const noexcept {
	if (mutationVersion_) ++*mutationVersion_;
}
