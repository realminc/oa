#include <cassert>
#include <algorithm>

#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Device.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Runtime/OaVma.h>
#include <Oa/Core/Memory.h>
#include <Oa/Runtime/Engine.h>

namespace {

constexpr OaU64 TransferCapacity(OaU64 InSize) {
	return (std::max<OaU64>(InSize, 1ULL) + 3ULL) & ~3ULL;
}

} // namespace

static OaVmaVulkanFunctions GetOaVkFunctions() {
	OaVmaVulkanFunctions fns{};
	fns.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
	fns.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
	fns.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
	fns.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
	fns.vkAllocateMemory = vkAllocateMemory;
	fns.vkFreeMemory = vkFreeMemory;
	fns.vkMapMemory = vkMapMemory;
	fns.vkUnmapMemory = vkUnmapMemory;
	fns.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges;
	fns.vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges;
	fns.vkBindBufferMemory = vkBindBufferMemory;
	fns.vkBindImageMemory = vkBindImageMemory;
	fns.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements;
	fns.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements;
	fns.vkCreateBuffer = vkCreateBuffer;
	fns.vkDestroyBuffer = vkDestroyBuffer;
	fns.vkCreateImage = vkCreateImage;
	fns.vkDestroyImage = vkDestroyImage;
	fns.vkCmdCopyBuffer = vkCmdCopyBuffer;
	fns.vkGetBufferMemoryRequirements2KHR = vkGetBufferMemoryRequirements2;
	fns.vkGetImageMemoryRequirements2KHR = vkGetImageMemoryRequirements2;
	fns.vkBindBufferMemory2KHR = vkBindBufferMemory2;
	fns.vkBindImageMemory2KHR = vkBindImageMemory2;
	fns.vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2;
	return fns;
}

OaResult<OaVma> OaVma::Create(const OaVkDevice& InDevice) {
	OaVmaVulkanFunctions fns = GetOaVkFunctions();

	OaVmaAllocatorCreateInfo ci{};
	ci.vulkanApiVersion = InDevice.Info.Software.ApiVersionPacked;
	ci.instance = static_cast<VkInstance>(InDevice.Instance);
	ci.physicalDevice = static_cast<VkPhysicalDevice>(InDevice.PhysicalDevice);
	ci.device = static_cast<VkDevice>(InDevice.Device);
	ci.pVulkanFunctions = &fns;
	// Enable buffer device address support (required for bindless + GPU compute graphs)
	// Enable KHR_maintenance5 support (required for VkBufferUsageFlags2CreateInfo)
	ci.flags = OA_VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	for (const auto& extension : InDevice.Info.Software.EnabledDeviceExtensions) {
		if (extension == "VK_KHR_maintenance5") {
			ci.flags |= OA_VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT;
			break;
		}
	}

	OaVmaAllocator alloc = VK_NULL_HANDLE;
	VkResult r = OaVmaCreateAllocator(&ci, &alloc);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::OutOfMemory, "OaVmaCreateAllocator failed");
	}

	OaVma a;
	a.Allocator = alloc;
	a.HasSam = InDevice.Info.Hardware.HasSAM;
	return a;
}

void OaVma::Destroy() {
	if (Allocator) {
		OaVmaDestroyAllocator(static_cast<OaVmaAllocator>(Allocator));
		Allocator = nullptr;
	}
}

OaResult<OaVkBuffer> OaVma::AllocDevice(OaU64 InSize) {
	const OaU64 capacity = TransferCapacity(InSize);
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
		static_cast<OaVmaAllocator>(Allocator),
		&bufCI, &allocCI, &buffer, &allocation, nullptr
	);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::OutOfMemory, "device buffer allocation failed");
	}

	OaVkBuffer buf;
	buf.Buffer = buffer;
	buf.Allocation = allocation;
	buf.Size = InSize;
	buf.Capacity = capacity;
	buf.MappedPtr = nullptr;
	buf.Flags = OA_VK_BUFFER_FLAG_NONE;
	buf.Placement = OaMemoryPlacement::DeviceLocal;
	return buf;
}

OaResult<OaVkBuffer> OaVma::AllocHostVisible(OaU64 InSize) {
	const OaU64 capacity = TransferCapacity(InSize);
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
	// OaMatrix exposes its mapped pointer as a first-class CPU access path. Keep
	// that contract valid for direct Data()/DataAs() users; explicit flush and
	// invalidate calls remain in transfer primitives as a second line of defence.
	allocCI.requiredFlags =
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	VkBuffer buffer = VK_NULL_HANDLE;
	OaVmaAllocation allocation = VK_NULL_HANDLE;
	OaVmaAllocationInfo allocInfo{};
	VkResult r = OaVmaCreateBuffer(
		static_cast<OaVmaAllocator>(Allocator),
		&bufCI, &allocCI, &buffer, &allocation, &allocInfo);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::OutOfMemory, "host-visible buffer allocation failed");
	}

	OaVkBuffer buf;
	buf.Buffer = buffer;
	buf.Allocation = allocation;
	buf.Size = InSize;
	buf.Capacity = capacity;
	buf.MappedPtr = allocInfo.pMappedData;
	buf.Flags = OA_VK_BUFFER_FLAG_NONE;
	buf.Placement = OaMemoryPlacement::HostUpload;
	return buf;
}

OaResult<OaVkBuffer> OaVma::AllocHostReadback(OaU64 InSize) {
	const OaU64 capacity = TransferCapacity(InSize);
	VkBufferCreateInfo bufCI{};
	bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufCI.size = capacity;
	bufCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
		| VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	OaVmaAllocationCreateInfo allocCI{};
	allocCI.usage = OA_VMA_MEMORY_USAGE_GPU_TO_CPU;
	allocCI.flags = OA_VMA_ALLOCATION_CREATE_MAPPED_BIT
		| OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	allocCI.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

	VkBuffer buffer = VK_NULL_HANDLE;
	OaVmaAllocation allocation = VK_NULL_HANDLE;
	OaVmaAllocationInfo allocInfo{};
	const VkResult result = OaVmaCreateBuffer(
		static_cast<OaVmaAllocator>(Allocator),
		&bufCI, &allocCI, &buffer, &allocation, &allocInfo);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::OutOfMemory, "host-readback buffer allocation failed");
	}

	OaVkBuffer out;
	out.Buffer = buffer;
	out.Allocation = allocation;
	out.Size = InSize;
	out.Capacity = capacity;
	out.MappedPtr = allocInfo.pMappedData;
	out.Flags = OA_VK_BUFFER_FLAG_NONE;
	out.Placement = OaMemoryPlacement::HostReadback;
	return out;
}

OaBool OaVma::FlushHostBuffer(const OaVkBuffer& InBuf, OaU64 InOffset, OaU64 InSize) {
	if (not Allocator or not InBuf.Allocation) {
		return true;
	}
	VkResult r = OaVmaFlushAllocation(
		static_cast<OaVmaAllocator>(Allocator),
		static_cast<OaVmaAllocation>(InBuf.Allocation),
		InOffset, InSize);
	return r == VK_SUCCESS;
}

OaBool OaVma::InvalidateHostBuffer(
	const OaVkBuffer& InBuf, OaU64 InOffset, OaU64 InSize) {
	if (not Allocator or not InBuf.Allocation) {
		return true;
	}
	VkResult r = OaVmaInvalidateAllocation(
		static_cast<OaVmaAllocator>(Allocator),
		static_cast<OaVmaAllocation>(InBuf.Allocation),
		InOffset, InSize);
	return r == VK_SUCCESS;
}

OaResult<OaVkBuffer> OaVma::AllocBar(OaU64 InSize) {
	if (!HasSam) {
		return AllocHostVisible(InSize);
	}
	const OaU64 capacity = TransferCapacity(InSize);

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
		static_cast<OaVmaAllocator>(Allocator),
		&bufCI, &allocCI, &buffer, &allocation, &allocInfo);
	if (r != VK_SUCCESS) {
		// BAR allocation can fail if VRAM is full; fall back to host-visible
		return AllocHostVisible(InSize);
	}

	OaVkBuffer buf;
	buf.Buffer = buffer;
	buf.Allocation = allocation;
	buf.Size = InSize;
	buf.Capacity = capacity;
	buf.MappedPtr = allocInfo.pMappedData;
	buf.Flags = OA_VK_BUFFER_FLAG_BAR;
	buf.Placement = OaMemoryPlacement::Unified;
	return buf;
}

OaResult<OaVkBuffer> OaVma::AllocPreprocessBuffer(OaU64 InSize) {
	const OaU64 capacity = TransferCapacity(InSize);
	// Allocate buffer with VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT for VK_EXT_device_generated_commands
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
	if (HasSam) {
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
		static_cast<OaVmaAllocator>(Allocator),
		&bufCI, &allocCI, &buffer, &allocation, &allocInfo);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::OutOfMemory, "preprocess buffer allocation failed");
	}

	OaVkBuffer buf;
	buf.Buffer = buffer;
	buf.Allocation = allocation;
	buf.Size = InSize;
	buf.Capacity = capacity;
	buf.MappedPtr = allocInfo.pMappedData;
	buf.Flags = (HasSam && allocInfo.pMappedData) ? OA_VK_BUFFER_FLAG_BAR : OA_VK_BUFFER_FLAG_NONE;
	buf.Placement = (HasSam && allocInfo.pMappedData)
		? OaMemoryPlacement::Unified
		: OaMemoryPlacement::DeviceLocal;
	return buf;
}

OaStatus OaVma::UploadWeights(OaVkBuffer& InDst, const void* InSrc, OaU64 InSize) {
	if (!InDst.MappedPtr) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "buffer not mapped");
	}
	if (InSize > InDst.Size) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "upload size exceeds buffer");
	}

	if (InDst.IsBar()) {
		OaMemcpyStream(InDst.MappedPtr, InSrc, InSize);
		OaStoreFence();
	} else {
		OaMemcpy(InDst.MappedPtr, InSrc, InSize);
	}
	return OaStatus::Ok();
}

OaResult<OaVkBuffer> OaVma::AllocAliased(
	OaU64 InSize, OaMemoryPlacement InPlacement) {
	const OaU64 capacity = TransferCapacity(InSize);
	VkBufferCreateInfo bufCI{};
	bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufCI.size = capacity;
	bufCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
		| VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	OaVmaAllocationCreateInfo allocCI{};
	allocCI.flags = OA_VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT;
	switch (InPlacement) {
		case OaMemoryPlacement::DeviceLocal:
			allocCI.usage = OA_VMA_MEMORY_USAGE_GPU_ONLY;
			break;
		case OaMemoryPlacement::Unified:
			allocCI.usage = OA_VMA_MEMORY_USAGE_AUTO;
			allocCI.flags |= OA_VMA_ALLOCATION_CREATE_MAPPED_BIT
				| OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
			allocCI.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
				| VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
			break;
		case OaMemoryPlacement::HostReadback:
			allocCI.usage = OA_VMA_MEMORY_USAGE_GPU_TO_CPU;
			allocCI.flags |= OA_VMA_ALLOCATION_CREATE_MAPPED_BIT
				| OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
			allocCI.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
			break;
		case OaMemoryPlacement::Auto:
		case OaMemoryPlacement::HostUpload:
			InPlacement = OaMemoryPlacement::HostUpload;
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
		static_cast<OaVmaAllocator>(Allocator),
		&bufCI, &allocCI, &buffer, &allocation, &allocInfo
	);
	if (res != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::OutOfMemory, "aliased buffer allocation failed");
	}

	OaVkBuffer buf;
	buf.Buffer = buffer;
	buf.Allocation = allocation;
	buf.Size = InSize;
	buf.Capacity = capacity;
	buf.MappedPtr = allocInfo.pMappedData;
	buf.Flags = OA_VK_BUFFER_FLAG_ALIAS;
	buf.Placement = InPlacement;
	return buf;
}

OaResult<OaVkBuffer> OaVma::CreateAliasingBuffer(
	const OaVkBuffer& InExisting, OaU64 InSize
) {
	const OaU64 capacity = TransferCapacity(InSize);
	if (!InExisting.Allocation) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"CreateAliasingBuffer: source has no allocation");
	}

	VkBufferCreateInfo bufCI{};
	bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufCI.size = capacity;
	bufCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
		| VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	VkBuffer buffer = VK_NULL_HANDLE;
	VkResult r = OaVmaCreateAliasingBuffer2(
		static_cast<OaVmaAllocator>(Allocator),
		static_cast<OaVmaAllocation>(InExisting.Allocation),
		0,
		&bufCI,
		&buffer);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::OutOfMemory, "aliasing buffer creation failed");
	}

	OaVkBuffer buf;
	buf.Buffer = buffer;
	buf.Allocation = nullptr;
	buf.Size = InSize;
	buf.Capacity = capacity;
	buf.MappedPtr = InExisting.MappedPtr;
	buf.Flags = OA_VK_BUFFER_FLAG_ALIAS;
	buf.Placement = InExisting.Placement;
	return buf;
}

void OaVma::FreeAlias(OaVkBuffer& InOutBuffer) {
	// OaVmaDestroyBuffer safely handles null allocation — destroys only the VkBuffer.
	Free(InOutBuffer);
}

OaVmaStats OaVma::GetStats() const {
	OaVmaStats stats{};
	if (!Allocator) return stats;

	OaVmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
	OaVmaGetHeapBudgets(static_cast<OaVmaAllocator>(Allocator), budgets);

	for (OaU32 i = 0; i < VK_MAX_MEMORY_HEAPS; ++i) {
		if (budgets[i].budget == 0) continue;
		stats.UsedBytes += budgets[i].usage;
		stats.BudgetBytes += budgets[i].budget;
	}

	// Detailed accounting: distinguishes bytes actually live (AllocationBytes) from bytes
	// reserved in VMA blocks (BlockBytes). A large BlockBytes/AllocationBytes ratio means
	// fragmentation / pooled slack rather than a true leak.
	OaVmaTotalStatistics total{};
	OaVmaCalculateStatistics(static_cast<OaVmaAllocator>(Allocator), &total);
	stats.AllocationBytes = total.total.statistics.allocationBytes;
	stats.BlockBytes      = total.total.statistics.blockBytes;
	stats.AllocationCount = total.total.statistics.allocationCount;
	stats.BlockCount      = total.total.statistics.blockCount;
	return stats;
}

void OaVma::Free(OaVkBuffer& InOutBuffer) {
	if (InOutBuffer.Buffer && Allocator) {
		assert(!InOutBuffer.IsImported() && "imported buffers must use FreeImported()");

		// Deregister bindless slot to prevent heap exhaustion during long training runs
		if (InOutBuffer.BindlessIndex != UINT32_MAX) {
			auto* engine = OaComputeEngine::GetGlobal();
			if (engine) {
				engine->Bindless.Deregister(InOutBuffer.BindlessIndex);
			}
			InOutBuffer.BindlessIndex = UINT32_MAX;
		}

		OaVmaDestroyBuffer(
			static_cast<OaVmaAllocator>(Allocator),
			static_cast<VkBuffer>(InOutBuffer.Buffer),
			static_cast<OaVmaAllocation>(InOutBuffer.Allocation)
		);
		InOutBuffer.Buffer = nullptr;
		InOutBuffer.Allocation = nullptr;
			InOutBuffer.MappedPtr = nullptr;
			InOutBuffer.Size = 0;
			InOutBuffer.Capacity = 0;
			InOutBuffer.Flags = OA_VK_BUFFER_FLAG_NONE;
			InOutBuffer.Placement = OaMemoryPlacement::Auto;
	}
}

void OaVma::FreeImported(const OaVkDevice& InDevice, OaVkBuffer& InOutBuffer) {
	if (!InOutBuffer.Buffer) {
		return;
	};
	assert(InOutBuffer.IsImported() && "non-imported buffers must use Free()");
	VkDevice dev = static_cast<VkDevice>(InDevice.Device);
	vkDestroyBuffer(dev, static_cast<VkBuffer>(InOutBuffer.Buffer), nullptr);
	if (InOutBuffer.Allocation) {
		vkFreeMemory(dev, static_cast<VkDeviceMemory>(InOutBuffer.Allocation), nullptr);
	}
	InOutBuffer.Buffer = nullptr;
	InOutBuffer.Allocation = nullptr;
	InOutBuffer.MappedPtr = nullptr;
	InOutBuffer.Size = 0;
	InOutBuffer.Capacity = 0;
	InOutBuffer.Flags = OA_VK_BUFFER_FLAG_NONE;
	InOutBuffer.Placement = OaMemoryPlacement::Auto;
}
