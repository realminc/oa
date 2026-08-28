// SPDX: MIT. Derived from Vulkan Memory Allocator, (c) 2017-2026 Advanced Micro Devices, Inc.
// Full MIT copyright/permission notice: source/cpp/thirdparty/vma/vma.h. See NOTICE.md.
#ifndef _VMA_ALLOCATOR_T_FUNCTIONS
VmaAllocator_T::VmaAllocator_T(const VmaAllocatorCreateInfo* pCreateInfo) :
	m_UseMutex((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT) == 0),
	m_VulkanApiVersion(pCreateInfo->vulkanApiVersion != 0 ? pCreateInfo->vulkanApiVersion : VK_API_VERSION_1_0),
	m_UseKhrDedicatedAllocation((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT) != 0),
	m_UseKhrBindMemory2((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT) != 0),
	m_UseExtMemoryBudget((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT) != 0),
	m_UseAmdDeviceCoherentMemory((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT) != 0),
	m_UseKhrBufferDeviceAddress((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT) != 0),
	m_UseExtMemoryPriority((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT) != 0),
	m_UseKhrMaintenance4((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT) != 0),
	m_UseKhrMaintenance5((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT) != 0),
	m_UseKhrExternalMemoryWin32((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT) != 0),
	m_hDevice(pCreateInfo->device),
	m_hInstance(pCreateInfo->instance),
	m_AllocationCallbacksSpecified(pCreateInfo->pAllocationCallbacks != VMA_NULL),
	m_AllocationCallbacks(pCreateInfo->pAllocationCallbacks ? *pCreateInfo->pAllocationCallbacks : VmaEmptyAllocationCallbacks),
	m_AllocationObjectAllocator(&m_AllocationCallbacks),
	m_HeapSizeLimitMask(0),
	m_DeviceMemoryCount(0),
	m_PreferredLargeHeapBlockSize(0),
	m_PhysicalDevice(pCreateInfo->physicalDevice),
	m_GpuDefragmentationMemoryTypeBits(UINT32_MAX),
	m_NextPoolId(0),
	m_GlobalMemoryTypeBits(UINT32_MAX)
{
	if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0)) {
		m_UseKhrDedicatedAllocation = false;
		m_UseKhrBindMemory2 = false;
	}

	if(VMA_DEBUG_DETECT_CORRUPTION) {
		// Needs to be multiply of uint32_t size because we are going to write VMA_CORRUPTION_DETECTION_MAGIC_VALUE to it.
		VMA_ASSERT(VMA_DEBUG_MARGIN % sizeof(uint32_t) == 0);
	}

	VMA_ASSERT(pCreateInfo->physicalDevice && pCreateInfo->device && pCreateInfo->instance);

	if(m_VulkanApiVersion < VK_MAKE_VERSION(1, 1, 0)) {
#if !(VMA_DEDICATED_ALLOCATION)
		if((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT) != 0) {
			VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT set but required extensions are disabled by preprocessor macros.");
		}
#endif
#if !(VMA_BIND_MEMORY2)
		if((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT) != 0) {
			VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT set but required extension is disabled by preprocessor macros.");
		}
#endif
	}
#if !(VMA_MEMORY_BUDGET)
	if((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT) != 0) {
		VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT set but required extension is disabled by preprocessor macros.");
	}
#endif
#if !(VMA_BUFFER_DEVICE_ADDRESS)
	if(m_UseKhrBufferDeviceAddress) {
		VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT is set but required extension or Vulkan 1.2 is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
	}
#endif
#if VMA_VULKAN_VERSION < 1004000
	VMA_ASSERT(m_VulkanApiVersion < VK_MAKE_VERSION(1, 4, 0) && "vulkanApiVersion >= VK_API_VERSION_1_4 but required Vulkan version is disabled by preprocessor macros.");
#endif
#if VMA_VULKAN_VERSION < 1003000
	VMA_ASSERT(m_VulkanApiVersion < VK_MAKE_VERSION(1, 3, 0) && "vulkanApiVersion >= VK_API_VERSION_1_3 but required Vulkan version is disabled by preprocessor macros.");
#endif
#if VMA_VULKAN_VERSION < 1002000
	VMA_ASSERT(m_VulkanApiVersion < VK_MAKE_VERSION(1, 2, 0) && "vulkanApiVersion >= VK_API_VERSION_1_2 but required Vulkan version is disabled by preprocessor macros.");
#endif
#if VMA_VULKAN_VERSION < 1001000
	VMA_ASSERT(m_VulkanApiVersion < VK_MAKE_VERSION(1, 1, 0) && "vulkanApiVersion >= VK_API_VERSION_1_1 but required Vulkan version is disabled by preprocessor macros.");
#endif
#if !(VMA_MEMORY_PRIORITY)
	if(m_UseExtMemoryPriority) {
		VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT is set but required extension is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
	}
#endif
#if !(VMA_KHR_MAINTENANCE4)
	if(m_UseKhrMaintenance4) {
		VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT is set but required extension is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
	}
#endif
#if !(VMA_KHR_MAINTENANCE5)
	if(m_UseKhrMaintenance5) {
		VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT is set but required extension is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
	}
#endif
#if !(VMA_KHR_MAINTENANCE5)
	if(m_UseKhrMaintenance5) {
		VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT is set but required extension is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
	}
#endif

#if !(VMA_EXTERNAL_MEMORY_WIN32)
	if(m_UseKhrExternalMemoryWin32) {
		VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT is set but required extension is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
	}
#endif

	m_DeviceMemoryCallbacks = {};
	m_PhysicalDeviceProperties = {};
	m_MemProps = {};
	oa::fill(m_pBlockVectors,
		m_pBlockVectors
			+ sizeof(m_pBlockVectors) / sizeof(m_pBlockVectors[0]),
		nullptr);
	m_VulkanFunctions = {};

#if VMA_EXTERNAL_MEMORY
	oa::memzero(&m_TypeExternalMemoryHandleTypes,
		sizeof(m_TypeExternalMemoryHandleTypes));
#endif // #if VMA_EXTERNAL_MEMORY

	if(pCreateInfo->pDeviceMemoryCallbacks != VMA_NULL) {
		m_DeviceMemoryCallbacks.pUserData = pCreateInfo->pDeviceMemoryCallbacks->pUserData;
		m_DeviceMemoryCallbacks.pfnAllocate = pCreateInfo->pDeviceMemoryCallbacks->pfnAllocate;
		m_DeviceMemoryCallbacks.pfnFree = pCreateInfo->pDeviceMemoryCallbacks->pfnFree;
	}

	ImportVulkanFunctions(pCreateInfo->pVulkanFunctions);

	(*m_VulkanFunctions.vkGetPhysicalDeviceProperties)(m_PhysicalDevice, &m_PhysicalDeviceProperties);
	(*m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties)(m_PhysicalDevice, &m_MemProps);

	VMA_ASSERT(VmaIsPow2(VMA_MIN_ALIGNMENT));
	VMA_ASSERT(VmaIsPow2(VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY));
	VMA_ASSERT(VmaIsPow2(m_PhysicalDeviceProperties.limits.bufferImageGranularity));
	VMA_ASSERT(VmaIsPow2(m_PhysicalDeviceProperties.limits.nonCoherentAtomSize));

	m_PreferredLargeHeapBlockSize = (pCreateInfo->preferredLargeHeapBlockSize != 0) ?
		pCreateInfo->preferredLargeHeapBlockSize : static_cast<VkDeviceSize>(VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE);

	m_GlobalMemoryTypeBits = CalculateGlobalMemoryTypeBits();

#if VMA_EXTERNAL_MEMORY
	if(pCreateInfo->pTypeExternalMemoryHandleTypes != VMA_NULL) {
		oa::memcpy(m_TypeExternalMemoryHandleTypes,
			pCreateInfo->pTypeExternalMemoryHandleTypes,
			sizeof(VkExternalMemoryHandleTypeFlagsKHR) * GetMemoryTypeCount());
	}
#endif // #if VMA_EXTERNAL_MEMORY

	if(pCreateInfo->pHeapSizeLimit != VMA_NULL) {
		for(uint32_t heapIndex = 0; heapIndex < GetMemoryHeapCount(); ++heapIndex) {
			const VkDeviceSize limit = pCreateInfo->pHeapSizeLimit[heapIndex];
			if(limit != VK_WHOLE_SIZE) {
				m_HeapSizeLimitMask |= 1U << heapIndex;
				if(limit < m_MemProps.memoryHeaps[heapIndex].size) {
					m_MemProps.memoryHeaps[heapIndex].size = limit;
				}
			}
		}
	}

	for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex) {
		// Create only supported types
		if((m_GlobalMemoryTypeBits & (1U << memTypeIndex)) != 0) {
			const VkDeviceSize preferredBlockSize = CalcPreferredBlockSize(memTypeIndex);
			m_pBlockVectors[memTypeIndex] = Vma_new(this, VmaBlockVector)(
				this,
				VK_NULL_HANDLE, // hParentPool
				memTypeIndex,
				preferredBlockSize,
				0,
				SIZE_MAX,
				GetBufferImageGranularity(),
				false, // explicitBlockSize
				0, // algorithm
				0.5F, // priority (0.5 is the default per Vulkan spec)
				GetMemoryTypeMinAlignment(memTypeIndex), // minAllocationAlignment
				VMA_NULL
			); // // pMemoryAllocateNext
			// No need to call m_pBlockVectors[memTypeIndex][blockVectorTypeIndex]->CreateMinBlocks here,
			// because minBlockCount is 0.
		}
	}
}

VkResult VmaAllocator_T::Init(const VmaAllocatorCreateInfo* pCreateInfo) {
	VkResult res = VK_SUCCESS;

#if VMA_MEMORY_BUDGET
	if(m_UseExtMemoryBudget) {
		UpdateVulkanBudget();
	}
#endif // #if VMA_MEMORY_BUDGET

	return res;
}

VmaAllocator_T::~VmaAllocator_T() {
	VMA_ASSERT(m_Pools.IsEmpty());

	for(size_t memTypeIndex = GetMemoryTypeCount(); memTypeIndex--; ) {
		Vma_delete(this, m_pBlockVectors[memTypeIndex]);
	}
}

void VmaAllocator_T::ImportVulkanFunctions(const VmaVulkanFunctions* pVulkanFunctions) {
#if VMA_STATIC_VULKAN_FUNCTIONS == 1
	ImportVulkanFunctions_Static();
#endif

	if(pVulkanFunctions != VMA_NULL) {
		ImportVulkanFunctions_Custom(pVulkanFunctions);
	}

#if VMA_DYNAMIC_VULKAN_FUNCTIONS == 1
	ImportVulkanFunctions_Dynamic();
#endif

	ValidateVulkanFunctions();
}

#if VMA_STATIC_VULKAN_FUNCTIONS == 1

void VmaAllocator_T::ImportVulkanFunctions_Static() {
	// Vulkan 1.0
	m_VulkanFunctions.vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)vkGetInstanceProcAddr;
	m_VulkanFunctions.vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)vkGetDeviceProcAddr;
	m_VulkanFunctions.vkGetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)vkGetPhysicalDeviceProperties;
	m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)vkGetPhysicalDeviceMemoryProperties;
	m_VulkanFunctions.vkAllocateMemory = (PFN_vkAllocateMemory)vkAllocateMemory;
	m_VulkanFunctions.vkFreeMemory = (PFN_vkFreeMemory)vkFreeMemory;
	m_VulkanFunctions.vkMapMemory = (PFN_vkMapMemory)vkMapMemory;
	m_VulkanFunctions.vkUnmapMemory = (PFN_vkUnmapMemory)vkUnmapMemory;
	m_VulkanFunctions.vkFlushMappedMemoryRanges = (PFN_vkFlushMappedMemoryRanges)vkFlushMappedMemoryRanges;
	m_VulkanFunctions.vkInvalidateMappedMemoryRanges = (PFN_vkInvalidateMappedMemoryRanges)vkInvalidateMappedMemoryRanges;
	m_VulkanFunctions.vkBindBufferMemory = (PFN_vkBindBufferMemory)vkBindBufferMemory;
	m_VulkanFunctions.vkBindImageMemory = (PFN_vkBindImageMemory)vkBindImageMemory;
	m_VulkanFunctions.vkGetBufferMemoryRequirements = (PFN_vkGetBufferMemoryRequirements)vkGetBufferMemoryRequirements;
	m_VulkanFunctions.vkGetImageMemoryRequirements = (PFN_vkGetImageMemoryRequirements)vkGetImageMemoryRequirements;
	m_VulkanFunctions.vkCreateBuffer = (PFN_vkCreateBuffer)vkCreateBuffer;
	m_VulkanFunctions.vkDestroyBuffer = (PFN_vkDestroyBuffer)vkDestroyBuffer;
	m_VulkanFunctions.vkCreateImage = (PFN_vkCreateImage)vkCreateImage;
	m_VulkanFunctions.vkDestroyImage = (PFN_vkDestroyImage)vkDestroyImage;
	m_VulkanFunctions.vkCmdCopyBuffer = (PFN_vkCmdCopyBuffer)vkCmdCopyBuffer;

	// Vulkan 1.1
#if VMA_VULKAN_VERSION >= 1001000
	if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0)) {
		m_VulkanFunctions.vkGetBufferMemoryRequirements2KHR = (PFN_vkGetBufferMemoryRequirements2)vkGetBufferMemoryRequirements2;
		m_VulkanFunctions.vkGetImageMemoryRequirements2KHR = (PFN_vkGetImageMemoryRequirements2)vkGetImageMemoryRequirements2;
		m_VulkanFunctions.vkBindBufferMemory2KHR = (PFN_vkBindBufferMemory2)vkBindBufferMemory2;
		m_VulkanFunctions.vkBindImageMemory2KHR = (PFN_vkBindImageMemory2)vkBindImageMemory2;
	}
#endif

#if VMA_VULKAN_VERSION >= 1001000
	if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0)) {
		m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR = (PFN_vkGetPhysicalDeviceMemoryProperties2)vkGetPhysicalDeviceMemoryProperties2;
	}
#endif

#if VMA_VULKAN_VERSION >= 1003000
	if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 3, 0)) {
		m_VulkanFunctions.vkGetDeviceBufferMemoryRequirements = (PFN_vkGetDeviceBufferMemoryRequirements)vkGetDeviceBufferMemoryRequirements;
		m_VulkanFunctions.vkGetDeviceImageMemoryRequirements = (PFN_vkGetDeviceImageMemoryRequirements)vkGetDeviceImageMemoryRequirements;
	}
#endif
}

#endif // VMA_STATIC_VULKAN_FUNCTIONS == 1

void VmaAllocator_T::ImportVulkanFunctions_Custom(const VmaVulkanFunctions* pVulkanFunctions) {
	VMA_ASSERT(pVulkanFunctions != VMA_NULL);

#define VMA_COPY_IF_NOT_NULL(funcName) \
	if(pVulkanFunctions->funcName != VMA_NULL) m_VulkanFunctions.funcName = pVulkanFunctions->funcName;

	VMA_COPY_IF_NOT_NULL(vkGetInstanceProcAddr);
	VMA_COPY_IF_NOT_NULL(vkGetDeviceProcAddr);
	VMA_COPY_IF_NOT_NULL(vkGetPhysicalDeviceProperties);
	VMA_COPY_IF_NOT_NULL(vkGetPhysicalDeviceMemoryProperties);
	VMA_COPY_IF_NOT_NULL(vkAllocateMemory);
	VMA_COPY_IF_NOT_NULL(vkFreeMemory);
	VMA_COPY_IF_NOT_NULL(vkMapMemory);
	VMA_COPY_IF_NOT_NULL(vkUnmapMemory);
	VMA_COPY_IF_NOT_NULL(vkFlushMappedMemoryRanges);
	VMA_COPY_IF_NOT_NULL(vkInvalidateMappedMemoryRanges);
	VMA_COPY_IF_NOT_NULL(vkBindBufferMemory);
	VMA_COPY_IF_NOT_NULL(vkBindImageMemory);
	VMA_COPY_IF_NOT_NULL(vkGetBufferMemoryRequirements);
	VMA_COPY_IF_NOT_NULL(vkGetImageMemoryRequirements);
	VMA_COPY_IF_NOT_NULL(vkCreateBuffer);
	VMA_COPY_IF_NOT_NULL(vkDestroyBuffer);
	VMA_COPY_IF_NOT_NULL(vkCreateImage);
	VMA_COPY_IF_NOT_NULL(vkDestroyImage);
	VMA_COPY_IF_NOT_NULL(vkCmdCopyBuffer);

#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
	VMA_COPY_IF_NOT_NULL(vkGetBufferMemoryRequirements2KHR);
	VMA_COPY_IF_NOT_NULL(vkGetImageMemoryRequirements2KHR);
#endif

#if VMA_BIND_MEMORY2 || VMA_VULKAN_VERSION >= 1001000
	VMA_COPY_IF_NOT_NULL(vkBindBufferMemory2KHR);
	VMA_COPY_IF_NOT_NULL(vkBindImageMemory2KHR);
#endif

#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
	VMA_COPY_IF_NOT_NULL(vkGetPhysicalDeviceMemoryProperties2KHR);
#endif

#if VMA_KHR_MAINTENANCE4 || VMA_VULKAN_VERSION >= 1003000
	VMA_COPY_IF_NOT_NULL(vkGetDeviceBufferMemoryRequirements);
	VMA_COPY_IF_NOT_NULL(vkGetDeviceImageMemoryRequirements);
#endif
#if VMA_EXTERNAL_MEMORY_WIN32
	VMA_COPY_IF_NOT_NULL(vkGetMemoryWin32HandleKHR);
#endif
#undef VMA_COPY_IF_NOT_NULL
}

#if VMA_DYNAMIC_VULKAN_FUNCTIONS == 1

void VmaAllocator_T::ImportVulkanFunctions_Dynamic() {
	VMA_ASSERT(m_VulkanFunctions.vkGetInstanceProcAddr && m_VulkanFunctions.vkGetDeviceProcAddr &&
		"To use VMA_DYNAMIC_VULKAN_FUNCTIONS in new versions of VMA you now have to pass "
		"VmaVulkanFunctions::vkGetInstanceProcAddr and vkGetDeviceProcAddr as VmaAllocatorCreateInfo::pVulkanFunctions. "
		"Other members can be null.");

#define VMA_FETCH_INSTANCE_FUNC(memberName, functionPointerType, functionNameString) \
	if(m_VulkanFunctions.memberName == VMA_NULL) \
		m_VulkanFunctions.memberName = \
			(functionPointerType)m_VulkanFunctions.vkGetInstanceProcAddr(m_hInstance, functionNameString);
#define VMA_FETCH_DEVICE_FUNC(memberName, functionPointerType, functionNameString) \
	if(m_VulkanFunctions.memberName == VMA_NULL) \
		m_VulkanFunctions.memberName = \
			(functionPointerType)m_VulkanFunctions.vkGetDeviceProcAddr(m_hDevice, functionNameString);

	VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceProperties, PFN_vkGetPhysicalDeviceProperties, "vkGetPhysicalDeviceProperties");
	VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties, PFN_vkGetPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties");
	VMA_FETCH_DEVICE_FUNC(vkAllocateMemory, PFN_vkAllocateMemory, "vkAllocateMemory");
	VMA_FETCH_DEVICE_FUNC(vkFreeMemory, PFN_vkFreeMemory, "vkFreeMemory");
	VMA_FETCH_DEVICE_FUNC(vkMapMemory, PFN_vkMapMemory, "vkMapMemory");
	VMA_FETCH_DEVICE_FUNC(vkUnmapMemory, PFN_vkUnmapMemory, "vkUnmapMemory");
	VMA_FETCH_DEVICE_FUNC(vkFlushMappedMemoryRanges, PFN_vkFlushMappedMemoryRanges, "vkFlushMappedMemoryRanges");
	VMA_FETCH_DEVICE_FUNC(vkInvalidateMappedMemoryRanges, PFN_vkInvalidateMappedMemoryRanges, "vkInvalidateMappedMemoryRanges");
	VMA_FETCH_DEVICE_FUNC(vkBindBufferMemory, PFN_vkBindBufferMemory, "vkBindBufferMemory");
	VMA_FETCH_DEVICE_FUNC(vkBindImageMemory, PFN_vkBindImageMemory, "vkBindImageMemory");
	VMA_FETCH_DEVICE_FUNC(vkGetBufferMemoryRequirements, PFN_vkGetBufferMemoryRequirements, "vkGetBufferMemoryRequirements");
	VMA_FETCH_DEVICE_FUNC(vkGetImageMemoryRequirements, PFN_vkGetImageMemoryRequirements, "vkGetImageMemoryRequirements");
	VMA_FETCH_DEVICE_FUNC(vkCreateBuffer, PFN_vkCreateBuffer, "vkCreateBuffer");
	VMA_FETCH_DEVICE_FUNC(vkDestroyBuffer, PFN_vkDestroyBuffer, "vkDestroyBuffer");
	VMA_FETCH_DEVICE_FUNC(vkCreateImage, PFN_vkCreateImage, "vkCreateImage");
	VMA_FETCH_DEVICE_FUNC(vkDestroyImage, PFN_vkDestroyImage, "vkDestroyImage");
	VMA_FETCH_DEVICE_FUNC(vkCmdCopyBuffer, PFN_vkCmdCopyBuffer, "vkCmdCopyBuffer");

#if VMA_VULKAN_VERSION >= 1001000
	if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0)) {
		VMA_FETCH_DEVICE_FUNC(vkGetBufferMemoryRequirements2KHR, PFN_vkGetBufferMemoryRequirements2, "vkGetBufferMemoryRequirements2");
		VMA_FETCH_DEVICE_FUNC(vkGetImageMemoryRequirements2KHR, PFN_vkGetImageMemoryRequirements2, "vkGetImageMemoryRequirements2");
		VMA_FETCH_DEVICE_FUNC(vkBindBufferMemory2KHR, PFN_vkBindBufferMemory2, "vkBindBufferMemory2");
		VMA_FETCH_DEVICE_FUNC(vkBindImageMemory2KHR, PFN_vkBindImageMemory2, "vkBindImageMemory2");
	}
#endif

#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
	if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0)) {
		VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2");
		// Try to fetch the pointer from the other name, based on suspected driver bug - see issue #410.
		VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2KHR");
	} else if(m_UseExtMemoryBudget) {
		VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2KHR");
		// Try to fetch the pointer from the other name, based on suspected driver bug - see issue #410.
		VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2");
	}
#endif

#if VMA_DEDICATED_ALLOCATION
	if(m_UseKhrDedicatedAllocation) {
		VMA_FETCH_DEVICE_FUNC(vkGetBufferMemoryRequirements2KHR, PFN_vkGetBufferMemoryRequirements2KHR, "vkGetBufferMemoryRequirements2KHR");
		VMA_FETCH_DEVICE_FUNC(vkGetImageMemoryRequirements2KHR, PFN_vkGetImageMemoryRequirements2KHR, "vkGetImageMemoryRequirements2KHR");
	}
#endif

#if VMA_BIND_MEMORY2
	if(m_UseKhrBindMemory2) {
		VMA_FETCH_DEVICE_FUNC(vkBindBufferMemory2KHR, PFN_vkBindBufferMemory2KHR, "vkBindBufferMemory2KHR");
		VMA_FETCH_DEVICE_FUNC(vkBindImageMemory2KHR, PFN_vkBindImageMemory2KHR, "vkBindImageMemory2KHR");
	}
#endif // #if VMA_BIND_MEMORY2

#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
	if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0)) {
		VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2");
	} else if(m_UseExtMemoryBudget) {
		VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2KHR");
	}
#endif // #if VMA_MEMORY_BUDGET

#if VMA_VULKAN_VERSION >= 1003000
	if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 3, 0)) {
		VMA_FETCH_DEVICE_FUNC(vkGetDeviceBufferMemoryRequirements, PFN_vkGetDeviceBufferMemoryRequirements, "vkGetDeviceBufferMemoryRequirements");
		VMA_FETCH_DEVICE_FUNC(vkGetDeviceImageMemoryRequirements, PFN_vkGetDeviceImageMemoryRequirements, "vkGetDeviceImageMemoryRequirements");
	}
#endif
#if VMA_KHR_MAINTENANCE4
	if(m_UseKhrMaintenance4) {
		VMA_FETCH_DEVICE_FUNC(vkGetDeviceBufferMemoryRequirements, PFN_vkGetDeviceBufferMemoryRequirementsKHR, "vkGetDeviceBufferMemoryRequirementsKHR");
		VMA_FETCH_DEVICE_FUNC(vkGetDeviceImageMemoryRequirements, PFN_vkGetDeviceImageMemoryRequirementsKHR, "vkGetDeviceImageMemoryRequirementsKHR");
	}
#endif
#if VMA_EXTERNAL_MEMORY_WIN32
	if (m_UseKhrExternalMemoryWin32) {
		VMA_FETCH_DEVICE_FUNC(vkGetMemoryWin32HandleKHR, PFN_vkGetMemoryWin32HandleKHR, "vkGetMemoryWin32HandleKHR");
	}
#endif
#undef VMA_FETCH_DEVICE_FUNC
#undef VMA_FETCH_INSTANCE_FUNC
}

#endif // VMA_DYNAMIC_VULKAN_FUNCTIONS == 1

void VmaAllocator_T::ValidateVulkanFunctions() const {
	VMA_ASSERT(m_VulkanFunctions.vkGetPhysicalDeviceProperties != VMA_NULL);
	VMA_ASSERT(m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties != VMA_NULL);
	VMA_ASSERT(m_VulkanFunctions.vkAllocateMemory != VMA_NULL);
	VMA_ASSERT(m_VulkanFunctions.vkFreeMemory != VMA_NULL);
	VMA_ASSERT(m_VulkanFunctions.vkMapMemory != VMA_NULL);
	VMA_ASSERT(m_VulkanFunctions.vkUnmapMemory != VMA_NULL);
	VMA_ASSERT(m_VulkanFunctions.vkFlushMappedMemoryRanges != VMA_NULL);
	VMA_ASSERT(m_VulkanFunctions.vkInvalidateMappedMemoryRanges != VMA_NULL);
	VMA_ASSERT(m_VulkanFunctions.vkBindBufferMemory != VMA_NULL);
	VMA_ASSERT(m_VulkanFunctions.vkBindImageMemory != VMA_NULL);
	VMA_ASSERT(m_VulkanFunctions.vkGetBufferMemoryRequirements != VMA_NULL);
	VMA_ASSERT(m_VulkanFunctions.vkGetImageMemoryRequirements != VMA_NULL);
	VMA_ASSERT(m_VulkanFunctions.vkCreateBuffer != VMA_NULL);
	VMA_ASSERT(m_VulkanFunctions.vkDestroyBuffer != VMA_NULL);
	VMA_ASSERT(m_VulkanFunctions.vkCreateImage != VMA_NULL);
	VMA_ASSERT(m_VulkanFunctions.vkDestroyImage != VMA_NULL);
	VMA_ASSERT(m_VulkanFunctions.vkCmdCopyBuffer != VMA_NULL);

#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
	if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0) || m_UseKhrDedicatedAllocation) {
		VMA_ASSERT(m_VulkanFunctions.vkGetBufferMemoryRequirements2KHR != VMA_NULL);
		VMA_ASSERT(m_VulkanFunctions.vkGetImageMemoryRequirements2KHR != VMA_NULL);
	}
#endif

#if VMA_BIND_MEMORY2 || VMA_VULKAN_VERSION >= 1001000
	if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0) || m_UseKhrBindMemory2) {
		VMA_ASSERT(m_VulkanFunctions.vkBindBufferMemory2KHR != VMA_NULL);
		VMA_ASSERT(m_VulkanFunctions.vkBindImageMemory2KHR != VMA_NULL);
	}
#endif

#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
	if(m_UseExtMemoryBudget || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0)) {
		VMA_ASSERT(m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR != VMA_NULL);
	}
#endif
#if VMA_EXTERNAL_MEMORY_WIN32
	if (m_UseKhrExternalMemoryWin32) {
		VMA_ASSERT(m_VulkanFunctions.vkGetMemoryWin32HandleKHR != VMA_NULL);
	}
#endif

	// Not validating these due to suspected driver bugs with these function
	// pointers being null despite correct extension or Vulkan version is enabled.
	// See issue #397. Their usage in VMA is optional anyway.
	//
	// VMA_ASSERT(m_VulkanFunctions.vkGetDeviceBufferMemoryRequirements != VMA_NULL);
	// VMA_ASSERT(m_VulkanFunctions.vkGetDeviceImageMemoryRequirements != VMA_NULL);
}

VkDeviceSize VmaAllocator_T::CalcPreferredBlockSize(uint32_t memTypeIndex) {
	const uint32_t heapIndex = MemoryTypeIndexToHeapIndex(memTypeIndex);
	const VkDeviceSize heapSize = m_MemProps.memoryHeaps[heapIndex].size;
	const bool isSmallHeap = heapSize <= VMA_SMALL_HEAP_MAX_SIZE;
	return VmaAlignUp(isSmallHeap ? (heapSize / 8) : m_PreferredLargeHeapBlockSize, (VkDeviceSize)32);
}

VkResult VmaAllocator_T::AllocateMemoryOfType(
	VmaPool pool,
	VkDeviceSize size,
	VkDeviceSize alignment,
	bool dedicatedPreferred,
	VkBuffer dedicatedBuffer,
	VkImage dedicatedImage,
	VmaBufferImageUsage dedicatedBufferImageUsage,
	void* pMemoryAllocateNext,
	const VmaAllocationCreateInfo& createInfo,
	uint32_t memTypeIndex,
	VmaSuballocationType suballocType,
	VmaDedicatedAllocationList& dedicatedAllocations,
	VmaBlockVector& blockVector,
	size_t allocationCount,
	VmaAllocation* pAllocations
) {
	VMA_ASSERT(pAllocations != VMA_NULL);
	VMA_DEBUG_LOG_FORMAT("  AllocateMemory: MemoryTypeIndex=%" PRIu32 ", AllocationCount=%zu, Size=%" PRIu64, memTypeIndex, allocationCount, size);

	VmaAllocationCreateInfo finalCreateInfo = createInfo;
	VkResult res = CalcMemTypeParams(
		finalCreateInfo,
		memTypeIndex,
		size,
		allocationCount);
	if(res != VK_SUCCESS)
		return res;

	const void* allocateNextPtr = blockVector.GetAllocationNextPtr();
	if(pMemoryAllocateNext != VMA_NULL) {
		VMA_ASSERT(allocateNextPtr == VMA_NULL &&
			"You shouldn't create a dedicated allocation with a custom pMemoryAllocateNext if the pNext chain is already provided for this pool.");
		allocateNextPtr = pMemoryAllocateNext;
	}

	if((finalCreateInfo.flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0) {
		return AllocateDedicatedMemory(
			pool,
			size,
			suballocType,
			dedicatedAllocations,
			memTypeIndex,
			(finalCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0,
			(finalCreateInfo.flags & VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0,
			(finalCreateInfo.flags &
				(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0,
			(finalCreateInfo.flags & VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT) != 0,
			finalCreateInfo.pUserData,
			finalCreateInfo.priority,
			dedicatedBuffer,
			dedicatedImage,
			dedicatedBufferImageUsage,
			allocationCount,
			pAllocations,
			allocateNextPtr);
	}

	const bool canAllocateDedicated =
		(finalCreateInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) == 0 &&
		(pool == VK_NULL_HANDLE || !blockVector.HasExplicitBlockSize());

	if(canAllocateDedicated) {
		// Heuristics: Allocate dedicated memory if requested size if greater than half of preferred block size.
		if(size > blockVector.GetPreferredBlockSize() / 2) {
			dedicatedPreferred = true;
		}
		// Protection against creating each allocation as dedicated when we reach or exceed heap size/budget,
		// which can quickly deplete maxMemoryAllocationCount: Don't prefer dedicated allocations when above
		// 3/4 of the maximum allocation count.
		if(m_PhysicalDeviceProperties.limits.maxMemoryAllocationCount < UINT32_MAX / 4 &&
			m_DeviceMemoryCount.load() > m_PhysicalDeviceProperties.limits.maxMemoryAllocationCount * 3 / 4) {
			dedicatedPreferred = false;
		}

		if(dedicatedPreferred) {
			res = AllocateDedicatedMemory(
				pool,
				size,
				suballocType,
				dedicatedAllocations,
				memTypeIndex,
				(finalCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0,
				(finalCreateInfo.flags & VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0,
				(finalCreateInfo.flags &
					(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0,
				(finalCreateInfo.flags & VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT) != 0,
				finalCreateInfo.pUserData,
				finalCreateInfo.priority,
				dedicatedBuffer,
				dedicatedImage,
				dedicatedBufferImageUsage,
				allocationCount,
				pAllocations,
				allocateNextPtr);
			if(res == VK_SUCCESS) {
				// Succeeded: AllocateDedicatedMemory function already filled pMemory, nothing more to do here.
				VMA_DEBUG_LOG("    Allocated as DedicatedMemory");
				return VK_SUCCESS;
			}
		}
	}

	res = blockVector.Allocate(
		size,
		alignment,
		finalCreateInfo,
		suballocType,
		allocationCount,
		pAllocations);
	if(res == VK_SUCCESS)
		return VK_SUCCESS;

	// Try dedicated memory.
	if(canAllocateDedicated && !dedicatedPreferred) {
		res = AllocateDedicatedMemory(
			pool,
			size,
			suballocType,
			dedicatedAllocations,
			memTypeIndex,
			(finalCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0,
			(finalCreateInfo.flags & VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0,
			(finalCreateInfo.flags &
				(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0,
			(finalCreateInfo.flags & VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT) != 0,
			finalCreateInfo.pUserData,
			finalCreateInfo.priority,
			dedicatedBuffer,
			dedicatedImage,
			dedicatedBufferImageUsage,
			allocationCount,
			pAllocations,
			allocateNextPtr);
		if(res == VK_SUCCESS) {
			// Succeeded: AllocateDedicatedMemory function already filled pMemory, nothing more to do here.
			VMA_DEBUG_LOG("    Allocated as DedicatedMemory");
			return VK_SUCCESS;
		}
	}

	// Everything failed: Return error code.
	VMA_DEBUG_LOG("    vkAllocateMemory FAILED");
	return res;
}

VkResult VmaAllocator_T::AllocateDedicatedMemory(
	VmaPool pool,
	VkDeviceSize size,
	VmaSuballocationType suballocType,
	VmaDedicatedAllocationList& dedicatedAllocations,
	uint32_t memTypeIndex,
	bool map,
	bool isUserDataString,
	bool isMappingAllowed,
	bool canAliasMemory,
	void* pUserData,
	float priority,
	VkBuffer dedicatedBuffer,
	VkImage dedicatedImage,
	VmaBufferImageUsage dedicatedBufferImageUsage,
	size_t allocationCount,
	VmaAllocation* pAllocations,
	const void* pNextChain
) {
	VMA_ASSERT(allocationCount > 0 && pAllocations);

	VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
	allocInfo.memoryTypeIndex = memTypeIndex;
	allocInfo.allocationSize = size;
	allocInfo.pNext = pNextChain;

#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
	VkMemoryDedicatedAllocateInfoKHR dedicatedAllocInfo = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR };
	if(!canAliasMemory) {
		if(m_UseKhrDedicatedAllocation || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0)) {
			if(dedicatedBuffer != VK_NULL_HANDLE) {
				VMA_ASSERT(dedicatedImage == VK_NULL_HANDLE);
				dedicatedAllocInfo.buffer = dedicatedBuffer;
				VmaPnextChainPushFront(&allocInfo, &dedicatedAllocInfo);
			} else if(dedicatedImage != VK_NULL_HANDLE) {
				dedicatedAllocInfo.image = dedicatedImage;
				VmaPnextChainPushFront(&allocInfo, &dedicatedAllocInfo);
			}
		}
	}
#endif // #if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000

#if VMA_BUFFER_DEVICE_ADDRESS
	VkMemoryAllocateFlagsInfoKHR allocFlagsInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR };
	if(m_UseKhrBufferDeviceAddress) {
		bool canContainBufferWithDeviceAddress = true;
		if(dedicatedBuffer != VK_NULL_HANDLE) {
			canContainBufferWithDeviceAddress = dedicatedBufferImageUsage == VmaBufferImageUsage::UNKNOWN ||
				dedicatedBufferImageUsage.Contains(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT);
		} else if(dedicatedImage != VK_NULL_HANDLE) {
			canContainBufferWithDeviceAddress = false;
		}
		if(canContainBufferWithDeviceAddress) {
			allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
			VmaPnextChainPushFront(&allocInfo, &allocFlagsInfo);
		}
	}
#endif // #if VMA_BUFFER_DEVICE_ADDRESS

#if VMA_MEMORY_PRIORITY
	VkMemoryPriorityAllocateInfoEXT priorityInfo = { VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT };
	if(m_UseExtMemoryPriority) {
		VMA_ASSERT(priority >= 0.F && priority <= 1.F);
		priorityInfo.priority = priority;
		VmaPnextChainPushFront(&allocInfo, &priorityInfo);
	}
#endif // #if VMA_MEMORY_PRIORITY

#if VMA_EXTERNAL_MEMORY
	// Attach VkExportMemoryAllocateInfoKHR if necessary.
	VkExportMemoryAllocateInfoKHR exportMemoryAllocInfo = { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR };
	exportMemoryAllocInfo.handleTypes = GetExternalMemoryHandleTypeFlags(memTypeIndex);
	if(exportMemoryAllocInfo.handleTypes != 0) {
		VmaPnextChainPushFront(&allocInfo, &exportMemoryAllocInfo);
	}
#endif // #if VMA_EXTERNAL_MEMORY

	size_t allocIndex = 0;
	VkResult res = VK_SUCCESS;
	for(; allocIndex < allocationCount; ++allocIndex) {
		res = AllocateDedicatedMemoryPage(
			pool,
			size,
			suballocType,
			memTypeIndex,
			allocInfo,
			map,
			isUserDataString,
			isMappingAllowed,
			pUserData,
			pAllocations + allocIndex);
		if(res != VK_SUCCESS)
		{
			break;
		}
	}

	if(res == VK_SUCCESS) {
		for (allocIndex = 0; allocIndex < allocationCount; ++allocIndex) {
			dedicatedAllocations.Register(pAllocations[allocIndex]);
		}
		VMA_DEBUG_LOG_FORMAT("    Allocated DedicatedMemory Count=%zu, MemoryTypeIndex=#%" PRIu32, allocationCount, memTypeIndex);
	} else {
		// Free all already created allocations.
		while(allocIndex--) {
			VmaAllocation currAlloc = pAllocations[allocIndex];
			VkDeviceMemory hMemory = currAlloc->GetMemory();

			/*
			There is no need to call this, because Vulkan spec allows to skip vkUnmapMemory
			before vkFreeMemory.

			if(currAlloc->GetMappedData() != VMA_NULL)
			{
				(*m_VulkanFunctions.vkUnmapMemory)(m_hDevice, hMemory);
			}
			*/

			FreeVulkanMemory(memTypeIndex, currAlloc->GetSize(), hMemory);
			m_Budget.RemoveAllocation(MemoryTypeIndexToHeapIndex(memTypeIndex), currAlloc->GetSize());
			m_AllocationObjectAllocator.Free(currAlloc);
		}

		oa::fill(pAllocations, pAllocations + allocationCount, VK_NULL_HANDLE);
	}

	return res;
}

VkResult VmaAllocator_T::AllocateDedicatedMemoryPage(
	VmaPool pool,
	VkDeviceSize size,
	VmaSuballocationType suballocType,
	uint32_t memTypeIndex,
	const VkMemoryAllocateInfo& allocInfo,
	bool map,
	bool isUserDataString,
	bool isMappingAllowed,
	void* pUserData,
	VmaAllocation* pAllocation)
{
	VkDeviceMemory hMemory = VK_NULL_HANDLE;
	VkResult res = AllocateVulkanMemory(&allocInfo, &hMemory);
	if(res < 0)
	{
		VMA_DEBUG_LOG("    vkAllocateMemory FAILED");
		return res;
	}

	void* pMappedData = VMA_NULL;
	if(map)
	{
		res = (*m_VulkanFunctions.vkMapMemory)(
			m_hDevice,
			hMemory,
			0,
			VK_WHOLE_SIZE,
			0,
			&pMappedData);
		if(res < 0)
		{
			VMA_DEBUG_LOG("    vkMapMemory FAILED");
			FreeVulkanMemory(memTypeIndex, size, hMemory);
			return res;
		}
	}

	*pAllocation = m_AllocationObjectAllocator.Allocate(isMappingAllowed);
	(*pAllocation)->InitDedicatedAllocation(this, pool, memTypeIndex, hMemory, suballocType, pMappedData, size);
	if (isUserDataString)
		(*pAllocation)->SetName(this, (const char*)pUserData);
	else
		(*pAllocation)->SetUserData(this, pUserData);
	m_Budget.AddAllocation(MemoryTypeIndexToHeapIndex(memTypeIndex), size);

#if VMA_DEBUG_INITIALIZE_ALLOCATIONS
	FillAllocation(*pAllocation, VMA_ALLOCATION_FILL_PATTERN_CREATED);
#endif

	return VK_SUCCESS;
}

void VmaAllocator_T::GetBufferMemoryRequirements(
	VkBuffer hBuffer,
	VkMemoryRequirements& memReq,
	bool& requiresDedicatedAllocation,
	bool& prefersDedicatedAllocation) const
{
#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
	if(m_UseKhrDedicatedAllocation || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
	{
		VkBufferMemoryRequirementsInfo2KHR memReqInfo = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR };
		memReqInfo.buffer = hBuffer;

		VkMemoryDedicatedRequirementsKHR memDedicatedReq = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR };

		VkMemoryRequirements2KHR memReq2 = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR };
		VmaPnextChainPushFront(&memReq2, &memDedicatedReq);

		(*m_VulkanFunctions.vkGetBufferMemoryRequirements2KHR)(m_hDevice, &memReqInfo, &memReq2);

		memReq = memReq2.memoryRequirements;
		requiresDedicatedAllocation = (memDedicatedReq.requiresDedicatedAllocation != VK_FALSE);
		prefersDedicatedAllocation  = (memDedicatedReq.prefersDedicatedAllocation  != VK_FALSE);
	}
	else
#endif // #if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
	{
		(*m_VulkanFunctions.vkGetBufferMemoryRequirements)(m_hDevice, hBuffer, &memReq);
		requiresDedicatedAllocation = false;
		prefersDedicatedAllocation  = false;
	}
}

void VmaAllocator_T::GetImageMemoryRequirements(
	VkImage hImage,
	VkMemoryRequirements& memReq,
	bool& requiresDedicatedAllocation,
	bool& prefersDedicatedAllocation) const
{
#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
	if(m_UseKhrDedicatedAllocation || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
	{
		VkImageMemoryRequirementsInfo2KHR memReqInfo = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR };
		memReqInfo.image = hImage;

		VkMemoryDedicatedRequirementsKHR memDedicatedReq = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR };

		VkMemoryRequirements2KHR memReq2 = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR };
		VmaPnextChainPushFront(&memReq2, &memDedicatedReq);

		(*m_VulkanFunctions.vkGetImageMemoryRequirements2KHR)(m_hDevice, &memReqInfo, &memReq2);

		memReq = memReq2.memoryRequirements;
		requiresDedicatedAllocation = (memDedicatedReq.requiresDedicatedAllocation != VK_FALSE);
		prefersDedicatedAllocation  = (memDedicatedReq.prefersDedicatedAllocation  != VK_FALSE);
	}
	else
#endif // #if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
	{
		(*m_VulkanFunctions.vkGetImageMemoryRequirements)(m_hDevice, hImage, &memReq);
		requiresDedicatedAllocation = false;
		prefersDedicatedAllocation  = false;
	}
}

VkResult VmaAllocator_T::FindMemoryTypeIndex(
	uint32_t memoryTypeBits,
	const VmaAllocationCreateInfo* pAllocationCreateInfo,
	VmaBufferImageUsage bufImgUsage,
	uint32_t* pMemoryTypeIndex) const
{
	memoryTypeBits &= GetGlobalMemoryTypeBits();

	if(pAllocationCreateInfo->memoryTypeBits != 0)
	{
		memoryTypeBits &= pAllocationCreateInfo->memoryTypeBits;
	}

	VkMemoryPropertyFlags requiredFlags = 0;
	VkMemoryPropertyFlags preferredFlags = 0;
	VkMemoryPropertyFlags notPreferredFlags = 0;
	if(!FindMemoryPreferences(
		IsIntegratedGpu(),
		*pAllocationCreateInfo,
		bufImgUsage,
		requiredFlags, preferredFlags, notPreferredFlags))
	{
		return VK_ERROR_FEATURE_NOT_PRESENT;
	}

	*pMemoryTypeIndex = UINT32_MAX;
	uint32_t minCost = UINT32_MAX;
	for(uint32_t memTypeIndex = 0, memTypeBit = 1;
		memTypeIndex < GetMemoryTypeCount();
		++memTypeIndex, memTypeBit <<= 1)
	{
		// This memory type is acceptable according to memoryTypeBits bitmask.
		if((memTypeBit & memoryTypeBits) != 0) {
			const VkMemoryPropertyFlags currFlags =
				m_MemProps.memoryTypes[memTypeIndex].propertyFlags;
			// This memory type contains requiredFlags.
			if((requiredFlags & ~currFlags) == 0) {
				// Calculate cost as number of bits from preferredFlags not present in this memory type.
				uint32_t currCost = VMA_COUNT_BITS_SET(preferredFlags & ~currFlags) +
					VMA_COUNT_BITS_SET(currFlags & notPreferredFlags);
				// Remember memory type with lowest cost.
				if(currCost < minCost) {
					*pMemoryTypeIndex = memTypeIndex;
					if(currCost == 0) {
						return VK_SUCCESS;
					}
					minCost = currCost;
				}
			}
		}
	}
	return (*pMemoryTypeIndex != UINT32_MAX) ? VK_SUCCESS : VK_ERROR_FEATURE_NOT_PRESENT;
}

VkResult VmaAllocator_T::CalcMemTypeParams(
	VmaAllocationCreateInfo& inoutCreateInfo,
	uint32_t memTypeIndex,
	VkDeviceSize size,
	size_t allocationCount
) {
	// If memory type is not HOST_VISIBLE, disable MAPPED.
	if((inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0 &&
		(m_MemProps.memoryTypes[memTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
		inoutCreateInfo.flags &= ~VMA_ALLOCATION_CREATE_MAPPED_BIT;
	}

	if((inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0 &&
		(inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT) != 0) {
		const uint32_t heapIndex = MemoryTypeIndexToHeapIndex(memTypeIndex);
		VmaBudget heapBudget = {};
		GetHeapBudgets(&heapBudget, heapIndex, 1);
		if(size == 0 || allocationCount > UINT64_MAX / size) {
			return VK_ERROR_OUT_OF_DEVICE_MEMORY;
		}
		const VkDeviceSize allocationBytes = size * allocationCount;
		if(heapBudget.usage > heapBudget.budget ||
			allocationBytes > heapBudget.budget - heapBudget.usage) {
			return VK_ERROR_OUT_OF_DEVICE_MEMORY;
		}
	}
	return VK_SUCCESS;
}

VkResult VmaAllocator_T::CalcAllocationParams(
	VmaAllocationCreateInfo& inoutCreateInfo,
	bool dedicatedRequired)
{
	VMA_ASSERT((inoutCreateInfo.flags &
		(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) !=
		(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT) &&
		"Specifying both flags VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT and VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT is incorrect.");
	VMA_ASSERT((((inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT) == 0 ||
		(inoutCreateInfo.flags & (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0)) &&
		"Specifying VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT requires also VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.");
	if(inoutCreateInfo.usage == VMA_MEMORY_USAGE_AUTO || inoutCreateInfo.usage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE || inoutCreateInfo.usage == VMA_MEMORY_USAGE_AUTO_PREFER_HOST) {
		if((inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0) {
			VMA_ASSERT((inoutCreateInfo.flags & (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0 &&
				"When using VMA_ALLOCATION_CREATE_MAPPED_BIT and usage = VMA_MEMORY_USAGE_AUTO*, you must also specify VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.");
		}
	}

	// If memory is lazily allocated, it should be always dedicated.
	if(dedicatedRequired ||
		inoutCreateInfo.usage == VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED) {
		inoutCreateInfo.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
	}

	if(inoutCreateInfo.pool != VK_NULL_HANDLE) {
		if(inoutCreateInfo.pool->m_BlockVector.HasExplicitBlockSize() &&
			(inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0)
		{
			VMA_ASSERT(0 && "Specifying VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT while current custom pool doesn't support dedicated allocations.");
			return VK_ERROR_FEATURE_NOT_PRESENT;
		}
		inoutCreateInfo.priority = inoutCreateInfo.pool->m_BlockVector.GetPriority();
	}

	if((inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0 &&
		(inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) != 0) {
		VMA_ASSERT(0 && "Specifying VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT together with VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT makes no sense.");
		return VK_ERROR_FEATURE_NOT_PRESENT;
	}

#if VMA_DEBUG_ALWAYS_DEDICATED_MEMORY
	if((inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) != 0) {
		inoutCreateInfo.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
	}
#endif

	// Non-auto USAGE values imply HOST_ACCESS flags.
	// And so does VMA_MEMORY_USAGE_UNKNOWN because it is used with custom pools.
	// Which specific flag is used doesn't matter. They change things only when used with VMA_MEMORY_USAGE_AUTO*.
	// Otherwise they just protect from assert on mapping.
	if(inoutCreateInfo.usage != VMA_MEMORY_USAGE_AUTO &&
		inoutCreateInfo.usage != VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE &&
		inoutCreateInfo.usage != VMA_MEMORY_USAGE_AUTO_PREFER_HOST) {
		if((inoutCreateInfo.flags & (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) == 0) {
			inoutCreateInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		}
	}

	return VK_SUCCESS;
}

VkResult VmaAllocator_T::CreateBuffer(
	const VkBufferCreateInfo* pBufferCreateInfo,
	const VmaAllocationCreateInfo* pAllocationCreateInfo,
	void* pMemoryAllocateNext,
	VkBuffer* pBuffer,
	VmaAllocation* pAllocation,
	VmaAllocationInfo* pAllocationInfo)
{
	*pBuffer = VK_NULL_HANDLE;
	*pAllocation = VK_NULL_HANDLE;

	if (pBufferCreateInfo->size == 0) {
		return VK_ERROR_INITIALIZATION_FAILED;
	}
	if ((pBufferCreateInfo->usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_COPY) != 0 &&
		!m_UseKhrBufferDeviceAddress) {
		VMA_ASSERT(0 && "Creating a buffer with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT is not valid if VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT was not used.");
		return VK_ERROR_INITIALIZATION_FAILED;
	}

	// 1. Create VkBuffer.
	VkResult res = (*m_VulkanFunctions.vkCreateBuffer)(m_hDevice, pBufferCreateInfo,
		GetAllocationCallbacks(), pBuffer);
	if (res >= 0)
	{
		// 2. vkGetBufferMemoryRequirements.
		VkMemoryRequirements vkMemReq = {};
		bool requiresDedicatedAllocation = false;
		bool prefersDedicatedAllocation = false;
		GetBufferMemoryRequirements(*pBuffer, vkMemReq,
			requiresDedicatedAllocation, prefersDedicatedAllocation);

		// 3. Allocate memory using allocator.
		res = AllocateMemory(
			vkMemReq,
			requiresDedicatedAllocation,
			prefersDedicatedAllocation,
			*pBuffer, // dedicatedBuffer
			VK_NULL_HANDLE, // dedicatedImage
			VmaBufferImageUsage(*pBufferCreateInfo, m_UseKhrMaintenance5), // dedicatedBufferImageUsage
			pMemoryAllocateNext,
			*pAllocationCreateInfo,
			VMA_SUBALLOCATION_TYPE_BUFFER,
			1, // allocationCount
			pAllocation);
		if (res >= 0)
		{
			// 3. Bind buffer with memory.
			if ((pAllocationCreateInfo->flags & VMA_ALLOCATION_CREATE_DONT_BIND_BIT) == 0)
			{
				res = BindBufferMemory(*pAllocation, 0, *pBuffer, VMA_NULL);
			}
			if (res >= 0)
			{
				// All steps succeeded.
#if VMA_STATS_STRING_ENABLED
				(*pAllocation)->InitBufferUsage(*pBufferCreateInfo, m_UseKhrMaintenance5);
#endif
				if (pAllocationInfo != VMA_NULL)
				{
					GetAllocationInfo(*pAllocation, pAllocationInfo);
				}

				return VK_SUCCESS;
			}
			FreeMemory(1, pAllocation);
			*pAllocation = VK_NULL_HANDLE;
			(*m_VulkanFunctions.vkDestroyBuffer)(m_hDevice, *pBuffer, GetAllocationCallbacks());
			*pBuffer = VK_NULL_HANDLE;
			return res;
		}
		(*m_VulkanFunctions.vkDestroyBuffer)(m_hDevice, *pBuffer, GetAllocationCallbacks());
		*pBuffer = VK_NULL_HANDLE;
		return res;
	}
	return res;
}

VkResult VmaAllocator_T::CreateImage(
	const VkImageCreateInfo* pImageCreateInfo,
	const VmaAllocationCreateInfo* pAllocationCreateInfo,
	void* pMemoryAllocateNext,
	VkImage* pImage,
	VmaAllocation* pAllocation,
	VmaAllocationInfo* pAllocationInfo)
{
	*pImage = VK_NULL_HANDLE;
	*pAllocation = VK_NULL_HANDLE;

	if (pImageCreateInfo->extent.width == 0 ||
		pImageCreateInfo->extent.height == 0 ||
		pImageCreateInfo->extent.depth == 0 ||
		pImageCreateInfo->mipLevels == 0 ||
		pImageCreateInfo->arrayLayers == 0)
	{
		return VK_ERROR_INITIALIZATION_FAILED;
	}

	// 1. Create VkImage.
	VkResult res = (*m_VulkanFunctions.vkCreateImage)(m_hDevice, pImageCreateInfo,
		GetAllocationCallbacks(), pImage);
	if (res == VK_SUCCESS)
	{
		VmaSuballocationType suballocType = pImageCreateInfo->tiling == VK_IMAGE_TILING_OPTIMAL ?
			VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL :
			VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR;

		// 2. Allocate memory using allocator.
		VkMemoryRequirements vkMemReq = {};
		bool requiresDedicatedAllocation = false;
		bool prefersDedicatedAllocation = false;
		GetImageMemoryRequirements(*pImage, vkMemReq,
			requiresDedicatedAllocation, prefersDedicatedAllocation);

		res = AllocateMemory(
			vkMemReq,
			requiresDedicatedAllocation,
			prefersDedicatedAllocation,
			VK_NULL_HANDLE, // dedicatedBuffer
			*pImage, // dedicatedImage
			VmaBufferImageUsage(*pImageCreateInfo), // dedicatedBufferImageUsage
			pMemoryAllocateNext,
			*pAllocationCreateInfo,
			suballocType,
			1, // allocationCount
			pAllocation);
		if (res == VK_SUCCESS)
		{
			// 3. Bind image with memory.
			if ((pAllocationCreateInfo->flags & VMA_ALLOCATION_CREATE_DONT_BIND_BIT) == 0)
			{
				res = BindImageMemory(*pAllocation, 0, *pImage, VMA_NULL);
			}
			if (res == VK_SUCCESS)
			{
				// All steps succeeded.
#if VMA_STATS_STRING_ENABLED
				(*pAllocation)->InitImageUsage(*pImageCreateInfo);
#endif
				if (pAllocationInfo != VMA_NULL)
				{
					GetAllocationInfo(*pAllocation, pAllocationInfo);
				}

				return VK_SUCCESS;
			}
			FreeMemory(1, pAllocation);
			*pAllocation = VK_NULL_HANDLE;
			(*m_VulkanFunctions.vkDestroyImage)(m_hDevice, *pImage, GetAllocationCallbacks());
			*pImage = VK_NULL_HANDLE;
			return res;
		}
		(*m_VulkanFunctions.vkDestroyImage)(m_hDevice, *pImage, GetAllocationCallbacks());
		*pImage = VK_NULL_HANDLE;
		return res;
	}
	return res;
}

VkResult VmaAllocator_T::AllocateMemory(
	VkMemoryRequirements vkMemReq,
	bool requiresDedicatedAllocation,
	bool prefersDedicatedAllocation,
	VkBuffer dedicatedBuffer,
	VkImage dedicatedImage,
	VmaBufferImageUsage dedicatedBufferImageUsage,
	// pNext chain for VkMemoryAllocateInfo. When used, must specify requiresDedicatedAllocation = true.
	void* pMemoryAllocateNext,
	const VmaAllocationCreateInfo& createInfo,
	VmaSuballocationType suballocType,
	size_t allocationCount,
	VmaAllocation* pAllocations)
{
	oa::fill(pAllocations, pAllocations + allocationCount, VK_NULL_HANDLE);

	vkMemReq.alignment = VMA_MAX(vkMemReq.alignment, createInfo.minAlignment);
	VMA_ASSERT(VmaIsPow2(vkMemReq.alignment));

	// If using custom pNext chain for VkMemoryAllocateInfo, must require dedicated allocations.
	if(pMemoryAllocateNext != VMA_NULL)
	{
		requiresDedicatedAllocation = true;
	}

	if(vkMemReq.size == 0)
	{
		return VK_ERROR_INITIALIZATION_FAILED;
	}

	VmaAllocationCreateInfo createInfoFinal = createInfo;
	VkResult res = CalcAllocationParams(createInfoFinal, requiresDedicatedAllocation);
	if(res != VK_SUCCESS)
		return res;

	if(createInfoFinal.pool != VK_NULL_HANDLE)
	{
		VmaBlockVector& blockVector = createInfoFinal.pool->m_BlockVector;
		return AllocateMemoryOfType(
			createInfoFinal.pool,
			vkMemReq.size,
			vkMemReq.alignment,
			prefersDedicatedAllocation,
			dedicatedBuffer,
			dedicatedImage,
			dedicatedBufferImageUsage,
			pMemoryAllocateNext,
			createInfoFinal,
			blockVector.GetMemoryTypeIndex(),
			suballocType,
			createInfoFinal.pool->m_DedicatedAllocations,
			blockVector,
			allocationCount,
			pAllocations);
	}

	// Bit mask of memory Vulkan types acceptable for this allocation.
	uint32_t memoryTypeBits = vkMemReq.memoryTypeBits;
	uint32_t memTypeIndex = UINT32_MAX;
	res = FindMemoryTypeIndex(memoryTypeBits, &createInfoFinal, dedicatedBufferImageUsage, &memTypeIndex);
	// Can't find any single memory type matching requirements. res is VK_ERROR_FEATURE_NOT_PRESENT.
	if(res != VK_SUCCESS)
		return res;

	do
	{
		VmaBlockVector* blockVector = m_pBlockVectors[memTypeIndex];
		VMA_ASSERT(blockVector && "Trying to use unsupported memory type!");
		res = AllocateMemoryOfType(
			VK_NULL_HANDLE,
			vkMemReq.size,
			vkMemReq.alignment,
			requiresDedicatedAllocation || prefersDedicatedAllocation,
			dedicatedBuffer,
			dedicatedImage,
			dedicatedBufferImageUsage,
			pMemoryAllocateNext,
			createInfoFinal,
			memTypeIndex,
			suballocType,
			m_DedicatedAllocations[memTypeIndex],
			*blockVector,
			allocationCount,
			pAllocations);
		// Allocation succeeded
		if(res == VK_SUCCESS)
			return VK_SUCCESS;

		// Remove old memTypeIndex from list of possibilities.
		memoryTypeBits &= ~(1U << memTypeIndex);
		// Find alternative memTypeIndex.
		res = FindMemoryTypeIndex(memoryTypeBits, &createInfoFinal, dedicatedBufferImageUsage, &memTypeIndex);
	} while(res == VK_SUCCESS);

	// No other matching memory type index could be found.
	// Not returning res, which is VK_ERROR_FEATURE_NOT_PRESENT, because we already failed to allocate once.
	return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

void VmaAllocator_T::FreeMemory(
	size_t allocationCount,
	const VmaAllocation* pAllocations)
{
	VMA_ASSERT(pAllocations);

	for(size_t allocIndex = allocationCount; allocIndex--; )
	{
		VmaAllocation allocation = pAllocations[allocIndex];

		if(allocation != VK_NULL_HANDLE)
		{
#if VMA_DEBUG_INITIALIZE_ALLOCATIONS
			FillAllocation(allocation, VMA_ALLOCATION_FILL_PATTERN_DESTROYED);
#endif

			switch(allocation->GetType())
			{
			case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
				{
					VmaBlockVector* pBlockVector = VMA_NULL;
					VmaPool hPool = allocation->GetParentPool();
					if(hPool != VK_NULL_HANDLE)
					{
						pBlockVector = &hPool->m_BlockVector;
					}
					else
					{
						const uint32_t memTypeIndex = allocation->GetMemoryTypeIndex();
						pBlockVector = m_pBlockVectors[memTypeIndex];
						VMA_ASSERT(pBlockVector && "Trying to free memory of unsupported type!");
					}
					pBlockVector->Free(allocation);
				}
				break;
			case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
				FreeDedicatedMemory(allocation);
				break;
			default:
				VMA_ASSERT(0);
			}
		}
	}
}

void VmaAllocator_T::CalculateStatistics(VmaTotalStatistics* pStats)
{
	// Initialize.
	VmaClearDetailedStatistics(pStats->total);
	for(uint32_t i = 0; i < VK_MAX_MEMORY_TYPES; ++i)
		VmaClearDetailedStatistics(pStats->memoryType[i]);
	for(uint32_t i = 0; i < VK_MAX_MEMORY_HEAPS; ++i)
		VmaClearDetailedStatistics(pStats->memoryHeap[i]);

	// Process default pools.
	for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
	{
		VmaBlockVector* const pBlockVector = m_pBlockVectors[memTypeIndex];
		if (pBlockVector != VMA_NULL)
			pBlockVector->AddDetailedStatistics(pStats->memoryType[memTypeIndex]);
	}

	// Process custom pools.
	{
		VmaMutexLockRead lock(m_PoolsMutex, m_UseMutex);
		for(VmaPool pool = m_Pools.Front(); pool != VMA_NULL; pool = m_Pools.GetNext(pool))
		{
			VmaBlockVector& blockVector = pool->m_BlockVector;
			const uint32_t memTypeIndex = blockVector.GetMemoryTypeIndex();
			blockVector.AddDetailedStatistics(pStats->memoryType[memTypeIndex]);
			pool->m_DedicatedAllocations.AddDetailedStatistics(pStats->memoryType[memTypeIndex]);
		}
	}

	// Process dedicated allocations.
	for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
	{
		m_DedicatedAllocations[memTypeIndex].AddDetailedStatistics(pStats->memoryType[memTypeIndex]);
	}

	// Sum from memory types to memory heaps.
	for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
	{
		const uint32_t memHeapIndex = m_MemProps.memoryTypes[memTypeIndex].heapIndex;
		VmaAddDetailedStatistics(pStats->memoryHeap[memHeapIndex], pStats->memoryType[memTypeIndex]);
	}

	// Sum from memory heaps to total.
	for(uint32_t memHeapIndex = 0; memHeapIndex < GetMemoryHeapCount(); ++memHeapIndex)
		VmaAddDetailedStatistics(pStats->total, pStats->memoryHeap[memHeapIndex]);

	VMA_ASSERT(pStats->total.statistics.allocationCount == 0 ||
		pStats->total.allocationSizeMax >= pStats->total.allocationSizeMin);
	VMA_ASSERT(pStats->total.unusedRangeCount == 0 ||
		pStats->total.unusedRangeSizeMax >= pStats->total.unusedRangeSizeMin);
}

void VmaAllocator_T::GetHeapBudgets(VmaBudget* outBudgets, uint32_t firstHeap, uint32_t heapCount)
{
#if VMA_MEMORY_BUDGET
	if(m_UseExtMemoryBudget)
	{
		if(m_Budget.m_OperationsSinceBudgetFetch < 30)
		{
			VmaMutexLockRead lockRead(m_Budget.m_BudgetMutex, m_UseMutex);
			for(uint32_t i = 0; i < heapCount; ++i, ++outBudgets)
			{
				const uint32_t heapIndex = firstHeap + i;

				outBudgets->statistics.blockCount = m_Budget.m_BlockCount[heapIndex];
				outBudgets->statistics.allocationCount = m_Budget.m_AllocationCount[heapIndex];
				outBudgets->statistics.blockBytes = m_Budget.m_BlockBytes[heapIndex];
				outBudgets->statistics.allocationBytes = m_Budget.m_AllocationBytes[heapIndex];

				if(m_Budget.m_VulkanUsage[heapIndex] + outBudgets->statistics.blockBytes > m_Budget.m_BlockBytesAtBudgetFetch[heapIndex])
				{
					outBudgets->usage = m_Budget.m_VulkanUsage[heapIndex] +
						outBudgets->statistics.blockBytes - m_Budget.m_BlockBytesAtBudgetFetch[heapIndex];
				}
				else
				{
					outBudgets->usage = 0;
				}

				// Have to take MIN with heap size because explicit HeapSizeLimit is included in it.
				outBudgets->budget = VMA_MIN(
					m_Budget.m_VulkanBudget[heapIndex], m_MemProps.memoryHeaps[heapIndex].size);
			}
		}
		else
		{
			UpdateVulkanBudget(); // Outside of mutex lock
			GetHeapBudgets(outBudgets, firstHeap, heapCount); // Recursion
		}
	}
	else
#endif
	{
		for(uint32_t i = 0; i < heapCount; ++i, ++outBudgets)
		{
			const uint32_t heapIndex = firstHeap + i;

			outBudgets->statistics.blockCount = m_Budget.m_BlockCount[heapIndex];
			outBudgets->statistics.allocationCount = m_Budget.m_AllocationCount[heapIndex];
			outBudgets->statistics.blockBytes = m_Budget.m_BlockBytes[heapIndex];
			outBudgets->statistics.allocationBytes = m_Budget.m_AllocationBytes[heapIndex];

			outBudgets->usage = outBudgets->statistics.blockBytes;
			outBudgets->budget = m_MemProps.memoryHeaps[heapIndex].size * 8 / 10; // 80% heuristics.
		}
	}
}

void VmaAllocator_T::GetAllocationInfo(VmaAllocation hAllocation, VmaAllocationInfo* pAllocationInfo)
{
	pAllocationInfo->memoryType = hAllocation->GetMemoryTypeIndex();
	pAllocationInfo->deviceMemory = hAllocation->GetMemory();
	pAllocationInfo->offset = hAllocation->GetOffset();
	pAllocationInfo->size = hAllocation->GetSize();
	pAllocationInfo->pMappedData = hAllocation->GetMappedData();
	pAllocationInfo->pUserData = hAllocation->GetUserData();
	pAllocationInfo->pName = hAllocation->GetName();
}

void VmaAllocator_T::GetAllocationInfo2(VmaAllocation hAllocation, VmaAllocationInfo2* pAllocationInfo)
{
	GetAllocationInfo(hAllocation, &pAllocationInfo->allocationInfo);

	switch (hAllocation->GetType())
	{
	case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
		pAllocationInfo->blockSize = hAllocation->GetBlock()->m_pMetadata->GetSize();
		pAllocationInfo->dedicatedMemory = VK_FALSE;
		break;
	case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
		pAllocationInfo->blockSize = pAllocationInfo->allocationInfo.size;
		pAllocationInfo->dedicatedMemory = VK_TRUE;
		break;
	default:
		VMA_ASSERT(0);
	}
}

VkResult VmaAllocator_T::CreatePool(const VmaPoolCreateInfo* pCreateInfo, VmaPool* pPool)
{
	VMA_DEBUG_LOG_FORMAT("  CreatePool: MemoryTypeIndex=%" PRIu32 ", flags=%" PRIu32, pCreateInfo->memoryTypeIndex, pCreateInfo->flags);

	VmaPoolCreateInfo newCreateInfo = *pCreateInfo;

	// Protection against uninitialized new structure member. If garbage data are left there, this pointer dereference would crash.
	if(pCreateInfo->pMemoryAllocateNext)
	{
		VMA_ASSERT(((const VkBaseInStructure*)pCreateInfo->pMemoryAllocateNext)->sType != 0);
	}

	if(newCreateInfo.maxBlockCount == 0)
	{
		newCreateInfo.maxBlockCount = SIZE_MAX;
	}
	if(newCreateInfo.minBlockCount > newCreateInfo.maxBlockCount)
	{
		return VK_ERROR_INITIALIZATION_FAILED;
	}
	// Memory type index out of range or forbidden.
	if(pCreateInfo->memoryTypeIndex >= GetMemoryTypeCount() ||
		((1U << pCreateInfo->memoryTypeIndex) & m_GlobalMemoryTypeBits) == 0)
	{
		return VK_ERROR_FEATURE_NOT_PRESENT;
	}
	if(newCreateInfo.minAllocationAlignment > 0)
	{
		VMA_ASSERT(VmaIsPow2(newCreateInfo.minAllocationAlignment));
	}

	const VkDeviceSize preferredBlockSize = CalcPreferredBlockSize(newCreateInfo.memoryTypeIndex);

	*pPool = Vma_new(this, VmaPool_T)(this, newCreateInfo, preferredBlockSize);

	VkResult res = (*pPool)->m_BlockVector.CreateMinBlocks();
	if(res != VK_SUCCESS)
	{
		Vma_delete(this, *pPool);
		*pPool = VMA_NULL;
		return res;
	}

	// Add to m_Pools.
	{
		VmaMutexLockWrite lock(m_PoolsMutex, m_UseMutex);
		(*pPool)->SetId(m_NextPoolId++);
		m_Pools.PushBack(*pPool);
	}

	return VK_SUCCESS;
}

void VmaAllocator_T::DestroyPool(VmaPool pool)
{
	// Remove from m_Pools.
	{
		VmaMutexLockWrite lock(m_PoolsMutex, m_UseMutex);
		m_Pools.Remove(pool);
	}

	Vma_delete(this, pool);
}

void VmaAllocator_T::GetPoolStatistics(VmaPool pool, VmaStatistics* pPoolStats)
{
	VmaClearStatistics(*pPoolStats);
	pool->m_BlockVector.AddStatistics(*pPoolStats);
	pool->m_DedicatedAllocations.AddStatistics(*pPoolStats);
}

void VmaAllocator_T::CalculatePoolStatistics(VmaPool pool, VmaDetailedStatistics* pPoolStats)
{
	VmaClearDetailedStatistics(*pPoolStats);
	pool->m_BlockVector.AddDetailedStatistics(*pPoolStats);
	pool->m_DedicatedAllocations.AddDetailedStatistics(*pPoolStats);
}

void VmaAllocator_T::SetCurrentFrameIndex(uint32_t frameIndex)
{
	m_CurrentFrameIndex.store(frameIndex);

#if VMA_MEMORY_BUDGET
	if(m_UseExtMemoryBudget)
	{
		UpdateVulkanBudget();
	}
#endif // #if VMA_MEMORY_BUDGET
}

VkResult VmaAllocator_T::CheckPoolCorruption(VmaPool hPool)
{
	return hPool->m_BlockVector.CheckCorruption();
}

VkResult VmaAllocator_T::CheckCorruption(uint32_t memoryTypeBits)
{
	VkResult finalRes = VK_ERROR_FEATURE_NOT_PRESENT;

	// Process default pools.
	for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
	{
		VmaBlockVector* const pBlockVector = m_pBlockVectors[memTypeIndex];
		if(pBlockVector != VMA_NULL)
		{
			VkResult localRes = pBlockVector->CheckCorruption();
			switch(localRes)
			{
			case VK_ERROR_FEATURE_NOT_PRESENT:
				break;
			case VK_SUCCESS:
				finalRes = VK_SUCCESS;
				break;
			default:
				return localRes;
			}
		}
	}

	// Process custom pools.
	{
		VmaMutexLockRead lock(m_PoolsMutex, m_UseMutex);
		for(VmaPool pool = m_Pools.Front(); pool != VMA_NULL; pool = m_Pools.GetNext(pool))
		{
			if(((1U << pool->m_BlockVector.GetMemoryTypeIndex()) & memoryTypeBits) != 0)
			{
				VkResult localRes = pool->m_BlockVector.CheckCorruption();
				switch(localRes)
				{
				case VK_ERROR_FEATURE_NOT_PRESENT:
					break;
				case VK_SUCCESS:
					finalRes = VK_SUCCESS;
					break;
				default:
					return localRes;
				}
			}
		}
	}

	return finalRes;
}

VkResult VmaAllocator_T::AllocateVulkanMemory(const VkMemoryAllocateInfo* pAllocateInfo, VkDeviceMemory* pMemory)
{
	const uint32_t heapIndex = MemoryTypeIndexToHeapIndex(pAllocateInfo->memoryTypeIndex);

#if VMA_DEBUG_DONT_EXCEED_HEAP_SIZE_WITH_ALLOCATION_SIZE
	if (pAllocateInfo->allocationSize > m_MemProps.memoryHeaps[heapIndex].size)
	{
		return VK_ERROR_OUT_OF_DEVICE_MEMORY;
	}
#endif

	AtomicTransactionalIncrement<VMA_ATOMIC_UINT32> deviceMemoryCountIncrement;
	const uint64_t prevDeviceMemoryCount = deviceMemoryCountIncrement.Increment(&m_DeviceMemoryCount);
#if VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT
	if(prevDeviceMemoryCount >= m_PhysicalDeviceProperties.limits.maxMemoryAllocationCount)
	{
		return VK_ERROR_TOO_MANY_OBJECTS;
	}
#endif

	// HeapSizeLimit is in effect for this heap.
	if((m_HeapSizeLimitMask & (1U << heapIndex)) != 0)
	{
		const VkDeviceSize heapSize = m_MemProps.memoryHeaps[heapIndex].size;
		VkDeviceSize blockBytes = m_Budget.m_BlockBytes[heapIndex];
		for(;;)
		{
			const VkDeviceSize blockBytesAfterAllocation = blockBytes + pAllocateInfo->allocationSize;
			if(blockBytesAfterAllocation > heapSize)
			{
				return VK_ERROR_OUT_OF_DEVICE_MEMORY;
			}
			if(m_Budget.m_BlockBytes[heapIndex].compareExchangeStrong(blockBytes, blockBytesAfterAllocation))
			{
				break;
			}
		}
	}
	else
	{
		m_Budget.m_BlockBytes[heapIndex] += pAllocateInfo->allocationSize;
	}
	++m_Budget.m_BlockCount[heapIndex];

	// VULKAN CALL vkAllocateMemory.
	VkResult res = (*m_VulkanFunctions.vkAllocateMemory)(m_hDevice, pAllocateInfo, GetAllocationCallbacks(), pMemory);

	if(res == VK_SUCCESS)
	{
#if VMA_MEMORY_BUDGET
		++m_Budget.m_OperationsSinceBudgetFetch;
#endif

		// Informative callback.
		if(m_DeviceMemoryCallbacks.pfnAllocate != VMA_NULL)
		{
			(*m_DeviceMemoryCallbacks.pfnAllocate)(this, pAllocateInfo->memoryTypeIndex, *pMemory, pAllocateInfo->allocationSize, m_DeviceMemoryCallbacks.pUserData);
		}

		deviceMemoryCountIncrement.Commit();
	}
	else
	{
		--m_Budget.m_BlockCount[heapIndex];
		m_Budget.m_BlockBytes[heapIndex] -= pAllocateInfo->allocationSize;
	}

	return res;
}

void VmaAllocator_T::FreeVulkanMemory(uint32_t memoryType, VkDeviceSize size, VkDeviceMemory hMemory)
{
	// Informative callback.
	if(m_DeviceMemoryCallbacks.pfnFree != VMA_NULL)
	{
		(*m_DeviceMemoryCallbacks.pfnFree)(this, memoryType, hMemory, size, m_DeviceMemoryCallbacks.pUserData);
	}

	// VULKAN CALL vkFreeMemory.
	(*m_VulkanFunctions.vkFreeMemory)(m_hDevice, hMemory, GetAllocationCallbacks());

	const uint32_t heapIndex = MemoryTypeIndexToHeapIndex(memoryType);
	--m_Budget.m_BlockCount[heapIndex];
	m_Budget.m_BlockBytes[heapIndex] -= size;

	--m_DeviceMemoryCount;
}

VkResult VmaAllocator_T::BindVulkanBuffer(
	VkDeviceMemory memory,
	VkDeviceSize memoryOffset,
	VkBuffer buffer,
	const void* pNext) const
{
	if(pNext != VMA_NULL)
	{
#if VMA_VULKAN_VERSION >= 1001000 || VMA_BIND_MEMORY2
		if((m_UseKhrBindMemory2 || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0)) &&
			m_VulkanFunctions.vkBindBufferMemory2KHR != VMA_NULL)
		{
			VkBindBufferMemoryInfoKHR bindBufferMemoryInfo = { VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO_KHR };
			bindBufferMemoryInfo.pNext = pNext;
			bindBufferMemoryInfo.buffer = buffer;
			bindBufferMemoryInfo.memory = memory;
			bindBufferMemoryInfo.memoryOffset = memoryOffset;
			return (*m_VulkanFunctions.vkBindBufferMemory2KHR)(m_hDevice, 1, &bindBufferMemoryInfo);
		}
#endif // #if VMA_VULKAN_VERSION >= 1001000 || VMA_BIND_MEMORY2

		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
	else
	{
		return (*m_VulkanFunctions.vkBindBufferMemory)(m_hDevice, buffer, memory, memoryOffset);
	}
}

VkResult VmaAllocator_T::BindVulkanImage(
	VkDeviceMemory memory,
	VkDeviceSize memoryOffset,
	VkImage image,
	const void* pNext) const
{
	if(pNext != VMA_NULL)
	{
#if VMA_VULKAN_VERSION >= 1001000 || VMA_BIND_MEMORY2
		if((m_UseKhrBindMemory2 || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0)) &&
			m_VulkanFunctions.vkBindImageMemory2KHR != VMA_NULL)
		{
			VkBindImageMemoryInfoKHR bindBufferMemoryInfo = { VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO_KHR };
			bindBufferMemoryInfo.pNext = pNext;
			bindBufferMemoryInfo.image = image;
			bindBufferMemoryInfo.memory = memory;
			bindBufferMemoryInfo.memoryOffset = memoryOffset;
			return (*m_VulkanFunctions.vkBindImageMemory2KHR)(m_hDevice, 1, &bindBufferMemoryInfo);
		}
#endif // #if VMA_BIND_MEMORY2

		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}

	return (*m_VulkanFunctions.vkBindImageMemory)(m_hDevice, image, memory, memoryOffset);
}

VkResult VmaAllocator_T::Map(VmaAllocation hAllocation, void** ppData) {
	switch(hAllocation->GetType()) {
	case VmaAllocation_T::ALLOCATION_TYPE_BLOCK: {
			VmaDeviceMemoryBlock* const pBlock = hAllocation->GetBlock();
			char *pBytes = VMA_NULL;
			VkResult res = pBlock->Map(this, 1, (void**)&pBytes);
			if(res == VK_SUCCESS) {
				*ppData = pBytes + (ptrdiff_t)hAllocation->GetOffset();
				hAllocation->BlockAllocMap();
			}
			return res;
		}
	case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
		return hAllocation->DedicatedAllocMap(this, ppData);
	default:
		VMA_ASSERT(0);
		return VK_ERROR_MEMORY_MAP_FAILED;
	}
}

void VmaAllocator_T::Unmap(VmaAllocation hAllocation) {
	switch(hAllocation->GetType()) {
	case VmaAllocation_T::ALLOCATION_TYPE_BLOCK: {
			VmaDeviceMemoryBlock* const pBlock = hAllocation->GetBlock();
			hAllocation->BlockAllocUnmap();
			pBlock->Unmap(this, 1);
		}
		break;
	case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
		hAllocation->DedicatedAllocUnmap(this);
		break;
	default:
		VMA_ASSERT(0);
	}
}

VkResult VmaAllocator_T::BindBufferMemory(
	VmaAllocation hAllocation,
	VkDeviceSize allocationLocalOffset,
	VkBuffer hBuffer,
	const void* pNext
) {
	VkResult res = VK_ERROR_UNKNOWN_COPY;
	switch(hAllocation->GetType()) {
	case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
		res = BindVulkanBuffer(hAllocation->GetMemory(), allocationLocalOffset, hBuffer, pNext);
		break;
	case VmaAllocation_T::ALLOCATION_TYPE_BLOCK: {
		VmaDeviceMemoryBlock* const pBlock = hAllocation->GetBlock();
		VMA_ASSERT(pBlock && "Binding buffer to allocation that doesn't belong to any block.");
		res = pBlock->BindBufferMemory(this, hAllocation, allocationLocalOffset, hBuffer, pNext);
		break;
	}
	default:
		VMA_ASSERT(0);
	}
	return res;
}

VkResult VmaAllocator_T::BindImageMemory(
	VmaAllocation hAllocation,
	VkDeviceSize allocationLocalOffset,
	VkImage hImage,
	const void* pNext)
{
	VkResult res = VK_ERROR_UNKNOWN_COPY;
	switch(hAllocation->GetType()) {
	case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
		res = BindVulkanImage(hAllocation->GetMemory(), allocationLocalOffset, hImage, pNext);
		break;
	case VmaAllocation_T::ALLOCATION_TYPE_BLOCK: {
		VmaDeviceMemoryBlock* pBlock = hAllocation->GetBlock();
		VMA_ASSERT(pBlock && "Binding image to allocation that doesn't belong to any block.");
		res = pBlock->BindImageMemory(this, hAllocation, allocationLocalOffset, hImage, pNext);
		break;
	}
	default:
		VMA_ASSERT(0);
	}
	return res;
}

VkResult VmaAllocator_T::FlushOrInvalidateAllocation(
	VmaAllocation hAllocation,
	VkDeviceSize offset, VkDeviceSize size,
	VMA_CACHE_OPERATION op
) {
	VkResult res = VK_SUCCESS;

	VkMappedMemoryRange memRange = {};
	if(GetFlushOrInvalidateRange(hAllocation, offset, size, memRange)) {
		switch(op)
		{
		case VMA_CACHE_FLUSH:
			res = (*GetVulkanFunctions().vkFlushMappedMemoryRanges)(m_hDevice, 1, &memRange);
			break;
		case VMA_CACHE_INVALIDATE:
			res = (*GetVulkanFunctions().vkInvalidateMappedMemoryRanges)(m_hDevice, 1, &memRange);
			break;
		default:
			VMA_ASSERT(0);
		}
	}
	// else: Just ignore this call.
	return res;
}

VkResult VmaAllocator_T::FlushOrInvalidateAllocations(
	uint32_t allocationCount,
	const VmaAllocation* allocations,
	const VkDeviceSize* offsets, const VkDeviceSize* sizes,
	VMA_CACHE_OPERATION op)
{
	typedef VmaStlAllocator<VkMappedMemoryRange> RangeAllocator;
	typedef VmaSmallVector<VkMappedMemoryRange, RangeAllocator, 16> RangeVector;
	RangeVector ranges = RangeVector(RangeAllocator(GetAllocationCallbacks()));

	for(uint32_t allocIndex = 0; allocIndex < allocationCount; ++allocIndex)
	{
		const VmaAllocation alloc = allocations[allocIndex];
		const VkDeviceSize offset = offsets != VMA_NULL ? offsets[allocIndex] : 0;
		const VkDeviceSize size = sizes != VMA_NULL ? sizes[allocIndex] : VK_WHOLE_SIZE;
		VkMappedMemoryRange newRange;
		if(GetFlushOrInvalidateRange(alloc, offset, size, newRange))
		{
			ranges.push_back(newRange);
		}
	}

	VkResult res = VK_SUCCESS;
	if(!ranges.empty())
	{
		switch(op)
		{
		case VMA_CACHE_FLUSH:
			res = (*GetVulkanFunctions().vkFlushMappedMemoryRanges)(m_hDevice, (uint32_t)ranges.size(), ranges.data());
			break;
		case VMA_CACHE_INVALIDATE:
			res = (*GetVulkanFunctions().vkInvalidateMappedMemoryRanges)(m_hDevice, (uint32_t)ranges.size(), ranges.data());
			break;
		default:
			VMA_ASSERT(0);
		}
	}
	// else: Just ignore this call.
	return res;
}

VkResult VmaAllocator_T::CopyMemoryToAllocation(
	const void* pSrcHostPointer,
	VmaAllocation dstAllocation,
	VkDeviceSize dstAllocationLocalOffset,
	VkDeviceSize size)
{
	void* dstMappedData = VMA_NULL;
	VkResult res = Map(dstAllocation, &dstMappedData);
	if(res == VK_SUCCESS)
	{
		oa::memcpy((char*)dstMappedData + dstAllocationLocalOffset,
			pSrcHostPointer, (size_t)size);
		Unmap(dstAllocation);
		res = FlushOrInvalidateAllocation(dstAllocation, dstAllocationLocalOffset, size, VMA_CACHE_FLUSH);
	}
	return res;
}

VkResult VmaAllocator_T::CopyAllocationToMemory(
	VmaAllocation srcAllocation,
	VkDeviceSize srcAllocationLocalOffset,
	void* pDstHostPointer,
	VkDeviceSize size)
{
	void* srcMappedData = VMA_NULL;
	VkResult res = Map(srcAllocation, &srcMappedData);
	if(res == VK_SUCCESS)
	{
		res = FlushOrInvalidateAllocation(srcAllocation, srcAllocationLocalOffset, size, VMA_CACHE_INVALIDATE);
		if(res == VK_SUCCESS)
		{
			oa::memcpy(pDstHostPointer,
				(const char*)srcMappedData + srcAllocationLocalOffset, (size_t)size);
			Unmap(srcAllocation);
		}
	}
	return res;
}

void VmaAllocator_T::FreeDedicatedMemory(VmaAllocation allocation)
{
	VMA_ASSERT(allocation && allocation->GetType() == VmaAllocation_T::ALLOCATION_TYPE_DEDICATED);

	const uint32_t memTypeIndex = allocation->GetMemoryTypeIndex();
	VmaPool parentPool = allocation->GetParentPool();
	if(parentPool == VK_NULL_HANDLE)
	{
		// Default pool
		m_DedicatedAllocations[memTypeIndex].Unregister(allocation);
	}
	else
	{
		// Custom pool
		parentPool->m_DedicatedAllocations.Unregister(allocation);
	}

	VkDeviceMemory hMemory = allocation->GetMemory();

	/*
	There is no need to call this, because Vulkan spec allows to skip vkUnmapMemory
	before vkFreeMemory.

	if(allocation->GetMappedData() != VMA_NULL)
	{
		(*m_VulkanFunctions.vkUnmapMemory)(m_hDevice, hMemory);
	}
	*/

	FreeVulkanMemory(memTypeIndex, allocation->GetSize(), hMemory);

	m_Budget.RemoveAllocation(MemoryTypeIndexToHeapIndex(allocation->GetMemoryTypeIndex()), allocation->GetSize());
	allocation->Destroy(this);
	m_AllocationObjectAllocator.Free(allocation);

	VMA_DEBUG_LOG_FORMAT("    Freed DedicatedMemory MemoryTypeIndex=%" PRIu32, memTypeIndex);
}

uint32_t VmaAllocator_T::CalculateGpuDefragmentationMemoryTypeBits() const
{
	VkBufferCreateInfo dummyBufCreateInfo;
	VmaFillGpuDefragmentationBufferCreateInfo(dummyBufCreateInfo);

	uint32_t memoryTypeBits = 0;

	// Create buffer.
	VkBuffer buf = VK_NULL_HANDLE;
	VkResult res = (*GetVulkanFunctions().vkCreateBuffer)(
		m_hDevice, &dummyBufCreateInfo, GetAllocationCallbacks(), &buf);
	if(res == VK_SUCCESS)
	{
		// Query for supported memory types.
		VkMemoryRequirements memReq;
		(*GetVulkanFunctions().vkGetBufferMemoryRequirements)(m_hDevice, buf, &memReq);
		memoryTypeBits = memReq.memoryTypeBits;

		// Destroy buffer.
		(*GetVulkanFunctions().vkDestroyBuffer)(m_hDevice, buf, GetAllocationCallbacks());
	}

	return memoryTypeBits;
}

uint32_t VmaAllocator_T::CalculateGlobalMemoryTypeBits() const
{
	// Make sure memory information is already fetched.
	VMA_ASSERT(GetMemoryTypeCount() > 0);

	uint32_t memoryTypeBits = UINT32_MAX;

	if(!m_UseAmdDeviceCoherentMemory)
	{
		// Exclude memory types that have VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD.
		for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
		{
			if((m_MemProps.memoryTypes[memTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY) != 0)
			{
				memoryTypeBits &= ~(1U << memTypeIndex);
			}
		}
	}

	return memoryTypeBits;
}

bool VmaAllocator_T::GetFlushOrInvalidateRange(
	VmaAllocation allocation,
	VkDeviceSize offset, VkDeviceSize size,
	VkMappedMemoryRange& outRange) const
{
	const uint32_t memTypeIndex = allocation->GetMemoryTypeIndex();
	if(size > 0 && IsMemoryTypeNonCoherent(memTypeIndex))
	{
		const VkDeviceSize nonCoherentAtomSize = m_PhysicalDeviceProperties.limits.nonCoherentAtomSize;
		const VkDeviceSize allocationSize = allocation->GetSize();
		VMA_ASSERT(offset <= allocationSize);

		outRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
		outRange.pNext = VMA_NULL;
		outRange.memory = allocation->GetMemory();

		switch(allocation->GetType())
		{
		case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
			outRange.offset = VmaAlignDown(offset, nonCoherentAtomSize);
			if(size == VK_WHOLE_SIZE)
			{
				outRange.size = allocationSize - outRange.offset;
			}
			else
			{
				VMA_ASSERT(offset + size <= allocationSize);
				outRange.size = VMA_MIN(
					VmaAlignUp(size + (offset - outRange.offset), nonCoherentAtomSize),
					allocationSize - outRange.offset);
			}
			break;
		case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
		{
			// 1. Still within this allocation.
			outRange.offset = VmaAlignDown(offset, nonCoherentAtomSize);
			if(size == VK_WHOLE_SIZE)
			{
				size = allocationSize - offset;
			}
			else
			{
				VMA_ASSERT(offset + size <= allocationSize);
			}
			outRange.size = VmaAlignUp(size + (offset - outRange.offset), nonCoherentAtomSize);

			// 2. Adjust to whole block.
			const VkDeviceSize allocationOffset = allocation->GetOffset();
			VMA_ASSERT(allocationOffset % nonCoherentAtomSize == 0);
			const VkDeviceSize blockSize = allocation->GetBlock()->m_pMetadata->GetSize();
			outRange.offset += allocationOffset;
			outRange.size = VMA_MIN(outRange.size, blockSize - outRange.offset);

			break;
		}
		default:
			VMA_ASSERT(0);
		}
		return true;
	}
	return false;
}

#if VMA_MEMORY_BUDGET
void VmaAllocator_T::UpdateVulkanBudget()
{
	VMA_ASSERT(m_UseExtMemoryBudget);

	VkPhysicalDeviceMemoryProperties2KHR memProps = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR };

	VkPhysicalDeviceMemoryBudgetPropertiesEXT budgetProps = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT };
	VmaPnextChainPushFront(&memProps, &budgetProps);

	GetVulkanFunctions().vkGetPhysicalDeviceMemoryProperties2KHR(m_PhysicalDevice, &memProps);

	{
		VmaMutexLockWrite lockWrite(m_Budget.m_BudgetMutex, m_UseMutex);

		for(uint32_t heapIndex = 0; heapIndex < GetMemoryHeapCount(); ++heapIndex)
		{
			m_Budget.m_VulkanUsage[heapIndex] = budgetProps.heapUsage[heapIndex];
			m_Budget.m_VulkanBudget[heapIndex] = budgetProps.heapBudget[heapIndex];
			m_Budget.m_BlockBytesAtBudgetFetch[heapIndex] = m_Budget.m_BlockBytes[heapIndex].load();

			// Some bugged drivers return the budget incorrectly, e.g. 0 or much bigger than heap size.
			if(m_Budget.m_VulkanBudget[heapIndex] == 0)
			{
				m_Budget.m_VulkanBudget[heapIndex] = m_MemProps.memoryHeaps[heapIndex].size * 8 / 10; // 80% heuristics.
			}
			else if(m_Budget.m_VulkanBudget[heapIndex] > m_MemProps.memoryHeaps[heapIndex].size)
			{
				m_Budget.m_VulkanBudget[heapIndex] = m_MemProps.memoryHeaps[heapIndex].size;
			}
			if(m_Budget.m_VulkanUsage[heapIndex] == 0 && m_Budget.m_BlockBytesAtBudgetFetch[heapIndex] > 0)
			{
				m_Budget.m_VulkanUsage[heapIndex] = m_Budget.m_BlockBytesAtBudgetFetch[heapIndex];
			}
		}
		m_Budget.m_OperationsSinceBudgetFetch = 0;
	}
}
#endif // VMA_MEMORY_BUDGET

void VmaAllocator_T::FillAllocation(VmaAllocation hAllocation, uint8_t pattern)
{
#if VMA_DEBUG_INITIALIZE_ALLOCATIONS
	if(hAllocation->IsMappingAllowed() &&
		(m_MemProps.memoryTypes[hAllocation->GetMemoryTypeIndex()].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
	{
		void* pData = VMA_NULL;
		VkResult res = Map(hAllocation, &pData);
		if(res == VK_SUCCESS)
		{
			oa::memset(pData, (int)pattern, (size_t)hAllocation->GetSize());
			FlushOrInvalidateAllocation(hAllocation, 0, VK_WHOLE_SIZE, VMA_CACHE_FLUSH);
			Unmap(hAllocation);
		}
		else
		{
			VMA_ASSERT(0 && "VMA_DEBUG_INITIALIZE_ALLOCATIONS is enabled, but couldn't map memory to fill allocation.");
		}
	}
#endif // #if VMA_DEBUG_INITIALIZE_ALLOCATIONS
}

uint32_t VmaAllocator_T::GetGpuDefragmentationMemoryTypeBits()
{
	uint32_t memoryTypeBits = m_GpuDefragmentationMemoryTypeBits.load();
	if(memoryTypeBits == UINT32_MAX)
	{
		memoryTypeBits = CalculateGpuDefragmentationMemoryTypeBits();
		m_GpuDefragmentationMemoryTypeBits.store(memoryTypeBits);
	}
	return memoryTypeBits;
}

#if VMA_STATS_STRING_ENABLED
void VmaAllocator_T::PrintDetailedMap(VmaJsonWriter& json)
{
	json.WriteString("DefaultPools");
	json.BeginObject();
	{
		for (uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
		{
			VmaBlockVector* pBlockVector = m_pBlockVectors[memTypeIndex];
			VmaDedicatedAllocationList& dedicatedAllocList = m_DedicatedAllocations[memTypeIndex];
			if (pBlockVector != VMA_NULL)
			{
				json.BeginString("Type ");
				json.ContinueString(memTypeIndex);
				json.EndString();
				json.BeginObject();
				{
					json.WriteString("PreferredBlockSize");
					json.WriteNumber(pBlockVector->GetPreferredBlockSize());

					json.WriteString("Blocks");
					pBlockVector->PrintDetailedMap(json);

					json.WriteString("DedicatedAllocations");
					dedicatedAllocList.BuildStatsString(json);
				}
				json.EndObject();
			}
		}
	}
	json.EndObject();

	json.WriteString("CustomPools");
	json.BeginObject();
	{
		VmaMutexLockRead lock(m_PoolsMutex, m_UseMutex);
		if (!m_Pools.IsEmpty())
		{
			for (uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
			{
				bool displayType = true;
				size_t index = 0;
				for (VmaPool pool = m_Pools.Front(); pool != VMA_NULL; pool = m_Pools.GetNext(pool))
				{
					VmaBlockVector& blockVector = pool->m_BlockVector;
					if (blockVector.GetMemoryTypeIndex() == memTypeIndex)
					{
						if (displayType)
						{
							json.BeginString("Type ");
							json.ContinueString(memTypeIndex);
							json.EndString();
							json.BeginArray();
							displayType = false;
						}

						json.BeginObject();
						{
							json.WriteString("Name");
							json.BeginString();
							json.ContinueString((uint64_t)index++);
							if (pool->GetName())
							{
								json.ContinueString(" - ");
								json.ContinueString(pool->GetName());
							}
							json.EndString();

							json.WriteString("PreferredBlockSize");
							json.WriteNumber(blockVector.GetPreferredBlockSize());

							json.WriteString("Blocks");
							blockVector.PrintDetailedMap(json);

							json.WriteString("DedicatedAllocations");
							pool->m_DedicatedAllocations.BuildStatsString(json);
						}
						json.EndObject();
					}
				}

				if (!displayType)
					json.EndArray();
			}
		}
	}
	json.EndObject();
}
#endif // VMA_STATS_STRING_ENABLED
#endif // _VMA_ALLOCATOR_T_FUNCTIONS
