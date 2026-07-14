// SPDX: MIT. Derived from Vulkan Memory Allocator, (c) 2017-2026 Advanced Micro Devices, Inc.
// Full MIT copyright/permission notice: Source/Public/Oa/Runtime/OaVma.h. See NOTICE.md.
#ifndef _OA_VMA_ALLOCATOR_T_FUNCTIONS
OaVmaAllocator_T::OaVmaAllocator_T(const OaVmaAllocatorCreateInfo* pCreateInfo) :
    m_UseMutex((pCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT) == 0),
    m_VulkanApiVersion(pCreateInfo->vulkanApiVersion != 0 ? pCreateInfo->vulkanApiVersion : VK_API_VERSION_1_0),
    m_UseKhrDedicatedAllocation((pCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT) != 0),
    m_UseKhrBindMemory2((pCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT) != 0),
    m_UseExtMemoryBudget((pCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT) != 0),
    m_UseAmdDeviceCoherentMemory((pCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT) != 0),
    m_UseKhrBufferDeviceAddress((pCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT) != 0),
    m_UseExtMemoryPriority((pCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT) != 0),
    m_UseKhrMaintenance4((pCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT) != 0),
    m_UseKhrMaintenance5((pCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT) != 0),
    m_UseKhrExternalMemoryWin32((pCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT) != 0),
    m_hDevice(pCreateInfo->device),
    m_hInstance(pCreateInfo->instance),
    m_AllocationCallbacksSpecified(pCreateInfo->pAllocationCallbacks != OA_VMA_NULL),
    m_AllocationCallbacks(pCreateInfo->pAllocationCallbacks ?
        *pCreateInfo->pAllocationCallbacks : OaVmaEmptyAllocationCallbacks),
    m_AllocationObjectAllocator(&m_AllocationCallbacks),
    m_HeapSizeLimitMask(0),
    m_DeviceMemoryCount(0),
    m_PreferredLargeHeapBlockSize(0),
    m_PhysicalDevice(pCreateInfo->physicalDevice),
    m_GpuDefragmentationMemoryTypeBits(UINT32_MAX),
    m_NextPoolId(0),
    m_GlobalMemoryTypeBits(UINT32_MAX)
{
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        m_UseKhrDedicatedAllocation = false;
        m_UseKhrBindMemory2 = false;
    }

    if(OA_VMA_DEBUG_DETECT_CORRUPTION)
    {
        // Needs to be multiply of uint32_t size because we are going to write OA_VMA_CORRUPTION_DETECTION_MAGIC_VALUE to it.
        OA_VMA_ASSERT(OA_VMA_DEBUG_MARGIN % sizeof(uint32_t) == 0);
    }

    OA_VMA_ASSERT(pCreateInfo->physicalDevice && pCreateInfo->device && pCreateInfo->instance);

    if(m_VulkanApiVersion < VK_MAKE_VERSION(1, 1, 0))
    {
#if !(OA_VMA_DEDICATED_ALLOCATION)
        if((pCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT) != 0)
        {
            OA_VMA_ASSERT(0 && "OA_VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT set but required extensions are disabled by preprocessor macros.");
        }
#endif
#if !(OA_VMA_BIND_MEMORY2)
        if((pCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT) != 0)
        {
            OA_VMA_ASSERT(0 && "OA_VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT set but required extension is disabled by preprocessor macros.");
        }
#endif
    }
#if !(OA_VMA_MEMORY_BUDGET)
    if((pCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT) != 0)
    {
        OA_VMA_ASSERT(0 && "OA_VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT set but required extension is disabled by preprocessor macros.");
    }
#endif
#if !(OA_VMA_BUFFER_DEVICE_ADDRESS)
    if(m_UseKhrBufferDeviceAddress)
    {
        OA_VMA_ASSERT(0 && "OA_VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT is set but required extension or Vulkan 1.2 is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
    }
#endif
#if OA_VMA_VULKAN_VERSION < 1004000
    OA_VMA_ASSERT(m_VulkanApiVersion < VK_MAKE_VERSION(1, 4, 0) && "vulkanApiVersion >= VK_API_VERSION_1_4 but required Vulkan version is disabled by preprocessor macros.");
#endif
#if OA_VMA_VULKAN_VERSION < 1003000
    OA_VMA_ASSERT(m_VulkanApiVersion < VK_MAKE_VERSION(1, 3, 0) && "vulkanApiVersion >= VK_API_VERSION_1_3 but required Vulkan version is disabled by preprocessor macros.");
#endif
#if OA_VMA_VULKAN_VERSION < 1002000
    OA_VMA_ASSERT(m_VulkanApiVersion < VK_MAKE_VERSION(1, 2, 0) && "vulkanApiVersion >= VK_API_VERSION_1_2 but required Vulkan version is disabled by preprocessor macros.");
#endif
#if OA_VMA_VULKAN_VERSION < 1001000
    OA_VMA_ASSERT(m_VulkanApiVersion < VK_MAKE_VERSION(1, 1, 0) && "vulkanApiVersion >= VK_API_VERSION_1_1 but required Vulkan version is disabled by preprocessor macros.");
#endif
#if !(OA_VMA_MEMORY_PRIORITY)
    if(m_UseExtMemoryPriority)
    {
        OA_VMA_ASSERT(0 && "OA_VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT is set but required extension is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
    }
#endif
#if !(OA_VMA_KHR_MAINTENANCE4)
    if(m_UseKhrMaintenance4)
    {
        OA_VMA_ASSERT(0 && "OA_VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT is set but required extension is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
    }
#endif
#if !(OA_VMA_KHR_MAINTENANCE5)
    if(m_UseKhrMaintenance5)
    {
        OA_VMA_ASSERT(0 && "OA_VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT is set but required extension is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
    }
#endif
#if !(OA_VMA_KHR_MAINTENANCE5)
    if(m_UseKhrMaintenance5)
    {
        OA_VMA_ASSERT(0 && "OA_VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT is set but required extension is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
    }
#endif

#if !(OA_VMA_EXTERNAL_MEMORY_WIN32)
    if(m_UseKhrExternalMemoryWin32)
    {
        OA_VMA_ASSERT(0 && "OA_VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT is set but required extension is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
    }
#endif

    memset(&m_DeviceMemoryCallbacks, 0 ,sizeof(m_DeviceMemoryCallbacks));
    memset(&m_PhysicalDeviceProperties, 0, sizeof(m_PhysicalDeviceProperties));
    memset(&m_MemProps, 0, sizeof(m_MemProps));

    memset(&m_pBlockVectors, 0, sizeof(m_pBlockVectors));
    memset(&m_VulkanFunctions, 0, sizeof(m_VulkanFunctions));

#if OA_VMA_EXTERNAL_MEMORY
    memset(&m_TypeExternalMemoryHandleTypes, 0, sizeof(m_TypeExternalMemoryHandleTypes));
#endif // #if OA_VMA_EXTERNAL_MEMORY

    if(pCreateInfo->pDeviceMemoryCallbacks != OA_VMA_NULL)
    {
        m_DeviceMemoryCallbacks.pUserData = pCreateInfo->pDeviceMemoryCallbacks->pUserData;
        m_DeviceMemoryCallbacks.pfnAllocate = pCreateInfo->pDeviceMemoryCallbacks->pfnAllocate;
        m_DeviceMemoryCallbacks.pfnFree = pCreateInfo->pDeviceMemoryCallbacks->pfnFree;
    }

    ImportVulkanFunctions(pCreateInfo->pVulkanFunctions);

    (*m_VulkanFunctions.vkGetPhysicalDeviceProperties)(m_PhysicalDevice, &m_PhysicalDeviceProperties);
    (*m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties)(m_PhysicalDevice, &m_MemProps);

    OA_VMA_ASSERT(OaVmaIsPow2(OA_VMA_MIN_ALIGNMENT));
    OA_VMA_ASSERT(OaVmaIsPow2(OA_VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY));
    OA_VMA_ASSERT(OaVmaIsPow2(m_PhysicalDeviceProperties.limits.bufferImageGranularity));
    OA_VMA_ASSERT(OaVmaIsPow2(m_PhysicalDeviceProperties.limits.nonCoherentAtomSize));

    m_PreferredLargeHeapBlockSize = (pCreateInfo->preferredLargeHeapBlockSize != 0) ?
        pCreateInfo->preferredLargeHeapBlockSize : static_cast<VkDeviceSize>(OA_VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE);

    m_GlobalMemoryTypeBits = CalculateGlobalMemoryTypeBits();

#if OA_VMA_EXTERNAL_MEMORY
    if(pCreateInfo->pTypeExternalMemoryHandleTypes != OA_VMA_NULL)
    {
        memcpy(m_TypeExternalMemoryHandleTypes, pCreateInfo->pTypeExternalMemoryHandleTypes,
            sizeof(VkExternalMemoryHandleTypeFlagsKHR) * GetMemoryTypeCount());
    }
#endif // #if OA_VMA_EXTERNAL_MEMORY

    if(pCreateInfo->pHeapSizeLimit != OA_VMA_NULL)
    {
        for(uint32_t heapIndex = 0; heapIndex < GetMemoryHeapCount(); ++heapIndex)
        {
            const VkDeviceSize limit = pCreateInfo->pHeapSizeLimit[heapIndex];
            if(limit != VK_WHOLE_SIZE)
            {
                m_HeapSizeLimitMask |= 1U << heapIndex;
                if(limit < m_MemProps.memoryHeaps[heapIndex].size)
                {
                    m_MemProps.memoryHeaps[heapIndex].size = limit;
                }
            }
        }
    }

    for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
    {
        // Create only supported types
        if((m_GlobalMemoryTypeBits & (1U << memTypeIndex)) != 0)
        {
            const VkDeviceSize preferredBlockSize = CalcPreferredBlockSize(memTypeIndex);
            m_pBlockVectors[memTypeIndex] = OaVma_new(this, OaVmaBlockVector)(
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
                OA_VMA_NULL); // // pMemoryAllocateNext
            // No need to call m_pBlockVectors[memTypeIndex][blockVectorTypeIndex]->CreateMinBlocks here,
            // because minBlockCount is 0.
        }
    }
}

VkResult OaVmaAllocator_T::Init(const OaVmaAllocatorCreateInfo* pCreateInfo)
{
    VkResult res = VK_SUCCESS;

#if OA_VMA_MEMORY_BUDGET
    if(m_UseExtMemoryBudget)
    {
        UpdateVulkanBudget();
    }
#endif // #if OA_VMA_MEMORY_BUDGET

    return res;
}

OaVmaAllocator_T::~OaVmaAllocator_T()
{
    OA_VMA_ASSERT(m_Pools.IsEmpty());

    for(size_t memTypeIndex = GetMemoryTypeCount(); memTypeIndex--; )
    {
        OaVma_delete(this, m_pBlockVectors[memTypeIndex]);
    }
}

void OaVmaAllocator_T::ImportVulkanFunctions(const OaVmaVulkanFunctions* pVulkanFunctions)
{
#if OA_VMA_STATIC_VULKAN_FUNCTIONS == 1
    ImportVulkanFunctions_Static();
#endif

    if(pVulkanFunctions != OA_VMA_NULL)
    {
        ImportVulkanFunctions_Custom(pVulkanFunctions);
    }

#if OA_VMA_DYNAMIC_VULKAN_FUNCTIONS == 1
    ImportVulkanFunctions_Dynamic();
#endif

    ValidateVulkanFunctions();
}

#if OA_VMA_STATIC_VULKAN_FUNCTIONS == 1

void OaVmaAllocator_T::ImportVulkanFunctions_Static()
{
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
#if OA_VMA_VULKAN_VERSION >= 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        m_VulkanFunctions.vkGetBufferMemoryRequirements2KHR = (PFN_vkGetBufferMemoryRequirements2)vkGetBufferMemoryRequirements2;
        m_VulkanFunctions.vkGetImageMemoryRequirements2KHR = (PFN_vkGetImageMemoryRequirements2)vkGetImageMemoryRequirements2;
        m_VulkanFunctions.vkBindBufferMemory2KHR = (PFN_vkBindBufferMemory2)vkBindBufferMemory2;
        m_VulkanFunctions.vkBindImageMemory2KHR = (PFN_vkBindImageMemory2)vkBindImageMemory2;
    }
#endif

#if OA_VMA_VULKAN_VERSION >= 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR = (PFN_vkGetPhysicalDeviceMemoryProperties2)vkGetPhysicalDeviceMemoryProperties2;
    }
#endif

#if OA_VMA_VULKAN_VERSION >= 1003000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 3, 0))
    {
        m_VulkanFunctions.vkGetDeviceBufferMemoryRequirements = (PFN_vkGetDeviceBufferMemoryRequirements)vkGetDeviceBufferMemoryRequirements;
        m_VulkanFunctions.vkGetDeviceImageMemoryRequirements = (PFN_vkGetDeviceImageMemoryRequirements)vkGetDeviceImageMemoryRequirements;
    }
#endif
}

#endif // OA_VMA_STATIC_VULKAN_FUNCTIONS == 1

void OaVmaAllocator_T::ImportVulkanFunctions_Custom(const OaVmaVulkanFunctions* pVulkanFunctions)
{
    OA_VMA_ASSERT(pVulkanFunctions != OA_VMA_NULL);

#define OA_VMA_COPY_IF_NOT_NULL(funcName) \
    if(pVulkanFunctions->funcName != OA_VMA_NULL) m_VulkanFunctions.funcName = pVulkanFunctions->funcName;

    OA_VMA_COPY_IF_NOT_NULL(vkGetInstanceProcAddr);
    OA_VMA_COPY_IF_NOT_NULL(vkGetDeviceProcAddr);
    OA_VMA_COPY_IF_NOT_NULL(vkGetPhysicalDeviceProperties);
    OA_VMA_COPY_IF_NOT_NULL(vkGetPhysicalDeviceMemoryProperties);
    OA_VMA_COPY_IF_NOT_NULL(vkAllocateMemory);
    OA_VMA_COPY_IF_NOT_NULL(vkFreeMemory);
    OA_VMA_COPY_IF_NOT_NULL(vkMapMemory);
    OA_VMA_COPY_IF_NOT_NULL(vkUnmapMemory);
    OA_VMA_COPY_IF_NOT_NULL(vkFlushMappedMemoryRanges);
    OA_VMA_COPY_IF_NOT_NULL(vkInvalidateMappedMemoryRanges);
    OA_VMA_COPY_IF_NOT_NULL(vkBindBufferMemory);
    OA_VMA_COPY_IF_NOT_NULL(vkBindImageMemory);
    OA_VMA_COPY_IF_NOT_NULL(vkGetBufferMemoryRequirements);
    OA_VMA_COPY_IF_NOT_NULL(vkGetImageMemoryRequirements);
    OA_VMA_COPY_IF_NOT_NULL(vkCreateBuffer);
    OA_VMA_COPY_IF_NOT_NULL(vkDestroyBuffer);
    OA_VMA_COPY_IF_NOT_NULL(vkCreateImage);
    OA_VMA_COPY_IF_NOT_NULL(vkDestroyImage);
    OA_VMA_COPY_IF_NOT_NULL(vkCmdCopyBuffer);

#if OA_VMA_DEDICATED_ALLOCATION || OA_VMA_VULKAN_VERSION >= 1001000
    OA_VMA_COPY_IF_NOT_NULL(vkGetBufferMemoryRequirements2KHR);
    OA_VMA_COPY_IF_NOT_NULL(vkGetImageMemoryRequirements2KHR);
#endif

#if OA_VMA_BIND_MEMORY2 || OA_VMA_VULKAN_VERSION >= 1001000
    OA_VMA_COPY_IF_NOT_NULL(vkBindBufferMemory2KHR);
    OA_VMA_COPY_IF_NOT_NULL(vkBindImageMemory2KHR);
#endif

#if OA_VMA_MEMORY_BUDGET || OA_VMA_VULKAN_VERSION >= 1001000
    OA_VMA_COPY_IF_NOT_NULL(vkGetPhysicalDeviceMemoryProperties2KHR);
#endif

#if OA_VMA_KHR_MAINTENANCE4 || OA_VMA_VULKAN_VERSION >= 1003000
    OA_VMA_COPY_IF_NOT_NULL(vkGetDeviceBufferMemoryRequirements);
    OA_VMA_COPY_IF_NOT_NULL(vkGetDeviceImageMemoryRequirements);
#endif
#if OA_VMA_EXTERNAL_MEMORY_WIN32
    OA_VMA_COPY_IF_NOT_NULL(vkGetMemoryWin32HandleKHR);
#endif
#undef OA_VMA_COPY_IF_NOT_NULL
}

#if OA_VMA_DYNAMIC_VULKAN_FUNCTIONS == 1

void OaVmaAllocator_T::ImportVulkanFunctions_Dynamic()
{
    OA_VMA_ASSERT(m_VulkanFunctions.vkGetInstanceProcAddr && m_VulkanFunctions.vkGetDeviceProcAddr &&
        "To use OA_VMA_DYNAMIC_VULKAN_FUNCTIONS in new versions of VMA you now have to pass "
        "OaVmaVulkanFunctions::vkGetInstanceProcAddr and vkGetDeviceProcAddr as OaVmaAllocatorCreateInfo::pVulkanFunctions. "
        "Other members can be null.");

#define OA_VMA_FETCH_INSTANCE_FUNC(memberName, functionPointerType, functionNameString) \
    if(m_VulkanFunctions.memberName == OA_VMA_NULL) \
        m_VulkanFunctions.memberName = \
            (functionPointerType)m_VulkanFunctions.vkGetInstanceProcAddr(m_hInstance, functionNameString);
#define OA_VMA_FETCH_DEVICE_FUNC(memberName, functionPointerType, functionNameString) \
    if(m_VulkanFunctions.memberName == OA_VMA_NULL) \
        m_VulkanFunctions.memberName = \
            (functionPointerType)m_VulkanFunctions.vkGetDeviceProcAddr(m_hDevice, functionNameString);

    OA_VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceProperties, PFN_vkGetPhysicalDeviceProperties, "vkGetPhysicalDeviceProperties");
    OA_VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties, PFN_vkGetPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties");
    OA_VMA_FETCH_DEVICE_FUNC(vkAllocateMemory, PFN_vkAllocateMemory, "vkAllocateMemory");
    OA_VMA_FETCH_DEVICE_FUNC(vkFreeMemory, PFN_vkFreeMemory, "vkFreeMemory");
    OA_VMA_FETCH_DEVICE_FUNC(vkMapMemory, PFN_vkMapMemory, "vkMapMemory");
    OA_VMA_FETCH_DEVICE_FUNC(vkUnmapMemory, PFN_vkUnmapMemory, "vkUnmapMemory");
    OA_VMA_FETCH_DEVICE_FUNC(vkFlushMappedMemoryRanges, PFN_vkFlushMappedMemoryRanges, "vkFlushMappedMemoryRanges");
    OA_VMA_FETCH_DEVICE_FUNC(vkInvalidateMappedMemoryRanges, PFN_vkInvalidateMappedMemoryRanges, "vkInvalidateMappedMemoryRanges");
    OA_VMA_FETCH_DEVICE_FUNC(vkBindBufferMemory, PFN_vkBindBufferMemory, "vkBindBufferMemory");
    OA_VMA_FETCH_DEVICE_FUNC(vkBindImageMemory, PFN_vkBindImageMemory, "vkBindImageMemory");
    OA_VMA_FETCH_DEVICE_FUNC(vkGetBufferMemoryRequirements, PFN_vkGetBufferMemoryRequirements, "vkGetBufferMemoryRequirements");
    OA_VMA_FETCH_DEVICE_FUNC(vkGetImageMemoryRequirements, PFN_vkGetImageMemoryRequirements, "vkGetImageMemoryRequirements");
    OA_VMA_FETCH_DEVICE_FUNC(vkCreateBuffer, PFN_vkCreateBuffer, "vkCreateBuffer");
    OA_VMA_FETCH_DEVICE_FUNC(vkDestroyBuffer, PFN_vkDestroyBuffer, "vkDestroyBuffer");
    OA_VMA_FETCH_DEVICE_FUNC(vkCreateImage, PFN_vkCreateImage, "vkCreateImage");
    OA_VMA_FETCH_DEVICE_FUNC(vkDestroyImage, PFN_vkDestroyImage, "vkDestroyImage");
    OA_VMA_FETCH_DEVICE_FUNC(vkCmdCopyBuffer, PFN_vkCmdCopyBuffer, "vkCmdCopyBuffer");

#if OA_VMA_VULKAN_VERSION >= 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        OA_VMA_FETCH_DEVICE_FUNC(vkGetBufferMemoryRequirements2KHR, PFN_vkGetBufferMemoryRequirements2, "vkGetBufferMemoryRequirements2");
        OA_VMA_FETCH_DEVICE_FUNC(vkGetImageMemoryRequirements2KHR, PFN_vkGetImageMemoryRequirements2, "vkGetImageMemoryRequirements2");
        OA_VMA_FETCH_DEVICE_FUNC(vkBindBufferMemory2KHR, PFN_vkBindBufferMemory2, "vkBindBufferMemory2");
        OA_VMA_FETCH_DEVICE_FUNC(vkBindImageMemory2KHR, PFN_vkBindImageMemory2, "vkBindImageMemory2");
    }
#endif

#if OA_VMA_MEMORY_BUDGET || OA_VMA_VULKAN_VERSION >= 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        OA_VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2");
        // Try to fetch the pointer from the other name, based on suspected driver bug - see issue #410.
        OA_VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2KHR");
    }
    else if(m_UseExtMemoryBudget)
    {
        OA_VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2KHR");
        // Try to fetch the pointer from the other name, based on suspected driver bug - see issue #410.
        OA_VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2");
    }
#endif

#if OA_VMA_DEDICATED_ALLOCATION
    if(m_UseKhrDedicatedAllocation)
    {
        OA_VMA_FETCH_DEVICE_FUNC(vkGetBufferMemoryRequirements2KHR, PFN_vkGetBufferMemoryRequirements2KHR, "vkGetBufferMemoryRequirements2KHR");
        OA_VMA_FETCH_DEVICE_FUNC(vkGetImageMemoryRequirements2KHR, PFN_vkGetImageMemoryRequirements2KHR, "vkGetImageMemoryRequirements2KHR");
    }
#endif

#if OA_VMA_BIND_MEMORY2
    if(m_UseKhrBindMemory2)
    {
        OA_VMA_FETCH_DEVICE_FUNC(vkBindBufferMemory2KHR, PFN_vkBindBufferMemory2KHR, "vkBindBufferMemory2KHR");
        OA_VMA_FETCH_DEVICE_FUNC(vkBindImageMemory2KHR, PFN_vkBindImageMemory2KHR, "vkBindImageMemory2KHR");
    }
#endif // #if OA_VMA_BIND_MEMORY2

#if OA_VMA_MEMORY_BUDGET || OA_VMA_VULKAN_VERSION >= 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        OA_VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2");
    }
    else if(m_UseExtMemoryBudget)
    {
        OA_VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2KHR");
    }
#endif // #if OA_VMA_MEMORY_BUDGET

#if OA_VMA_VULKAN_VERSION >= 1003000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 3, 0))
    {
        OA_VMA_FETCH_DEVICE_FUNC(vkGetDeviceBufferMemoryRequirements, PFN_vkGetDeviceBufferMemoryRequirements, "vkGetDeviceBufferMemoryRequirements");
        OA_VMA_FETCH_DEVICE_FUNC(vkGetDeviceImageMemoryRequirements, PFN_vkGetDeviceImageMemoryRequirements, "vkGetDeviceImageMemoryRequirements");
    }
#endif
#if OA_VMA_KHR_MAINTENANCE4
    if(m_UseKhrMaintenance4)
    {
        OA_VMA_FETCH_DEVICE_FUNC(vkGetDeviceBufferMemoryRequirements, PFN_vkGetDeviceBufferMemoryRequirementsKHR, "vkGetDeviceBufferMemoryRequirementsKHR");
        OA_VMA_FETCH_DEVICE_FUNC(vkGetDeviceImageMemoryRequirements, PFN_vkGetDeviceImageMemoryRequirementsKHR, "vkGetDeviceImageMemoryRequirementsKHR");
    }
#endif
#if OA_VMA_EXTERNAL_MEMORY_WIN32
    if (m_UseKhrExternalMemoryWin32)
    {
        OA_VMA_FETCH_DEVICE_FUNC(vkGetMemoryWin32HandleKHR, PFN_vkGetMemoryWin32HandleKHR, "vkGetMemoryWin32HandleKHR");
    }
#endif
#undef OA_VMA_FETCH_DEVICE_FUNC
#undef OA_VMA_FETCH_INSTANCE_FUNC
}

#endif // OA_VMA_DYNAMIC_VULKAN_FUNCTIONS == 1

void OaVmaAllocator_T::ValidateVulkanFunctions() const
{
    OA_VMA_ASSERT(m_VulkanFunctions.vkGetPhysicalDeviceProperties != OA_VMA_NULL);
    OA_VMA_ASSERT(m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties != OA_VMA_NULL);
    OA_VMA_ASSERT(m_VulkanFunctions.vkAllocateMemory != OA_VMA_NULL);
    OA_VMA_ASSERT(m_VulkanFunctions.vkFreeMemory != OA_VMA_NULL);
    OA_VMA_ASSERT(m_VulkanFunctions.vkMapMemory != OA_VMA_NULL);
    OA_VMA_ASSERT(m_VulkanFunctions.vkUnmapMemory != OA_VMA_NULL);
    OA_VMA_ASSERT(m_VulkanFunctions.vkFlushMappedMemoryRanges != OA_VMA_NULL);
    OA_VMA_ASSERT(m_VulkanFunctions.vkInvalidateMappedMemoryRanges != OA_VMA_NULL);
    OA_VMA_ASSERT(m_VulkanFunctions.vkBindBufferMemory != OA_VMA_NULL);
    OA_VMA_ASSERT(m_VulkanFunctions.vkBindImageMemory != OA_VMA_NULL);
    OA_VMA_ASSERT(m_VulkanFunctions.vkGetBufferMemoryRequirements != OA_VMA_NULL);
    OA_VMA_ASSERT(m_VulkanFunctions.vkGetImageMemoryRequirements != OA_VMA_NULL);
    OA_VMA_ASSERT(m_VulkanFunctions.vkCreateBuffer != OA_VMA_NULL);
    OA_VMA_ASSERT(m_VulkanFunctions.vkDestroyBuffer != OA_VMA_NULL);
    OA_VMA_ASSERT(m_VulkanFunctions.vkCreateImage != OA_VMA_NULL);
    OA_VMA_ASSERT(m_VulkanFunctions.vkDestroyImage != OA_VMA_NULL);
    OA_VMA_ASSERT(m_VulkanFunctions.vkCmdCopyBuffer != OA_VMA_NULL);

#if OA_VMA_DEDICATED_ALLOCATION || OA_VMA_VULKAN_VERSION >= 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0) || m_UseKhrDedicatedAllocation)
    {
        OA_VMA_ASSERT(m_VulkanFunctions.vkGetBufferMemoryRequirements2KHR != OA_VMA_NULL);
        OA_VMA_ASSERT(m_VulkanFunctions.vkGetImageMemoryRequirements2KHR != OA_VMA_NULL);
    }
#endif

#if OA_VMA_BIND_MEMORY2 || OA_VMA_VULKAN_VERSION >= 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0) || m_UseKhrBindMemory2)
    {
        OA_VMA_ASSERT(m_VulkanFunctions.vkBindBufferMemory2KHR != OA_VMA_NULL);
        OA_VMA_ASSERT(m_VulkanFunctions.vkBindImageMemory2KHR != OA_VMA_NULL);
    }
#endif

#if OA_VMA_MEMORY_BUDGET || OA_VMA_VULKAN_VERSION >= 1001000
    if(m_UseExtMemoryBudget || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        OA_VMA_ASSERT(m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR != OA_VMA_NULL);
    }
#endif
#if OA_VMA_EXTERNAL_MEMORY_WIN32
    if (m_UseKhrExternalMemoryWin32)
    {
        OA_VMA_ASSERT(m_VulkanFunctions.vkGetMemoryWin32HandleKHR != OA_VMA_NULL);
    }
#endif

    // Not validating these due to suspected driver bugs with these function
    // pointers being null despite correct extension or Vulkan version is enabled.
    // See issue #397. Their usage in VMA is optional anyway.
    //
    // OA_VMA_ASSERT(m_VulkanFunctions.vkGetDeviceBufferMemoryRequirements != OA_VMA_NULL);
    // OA_VMA_ASSERT(m_VulkanFunctions.vkGetDeviceImageMemoryRequirements != OA_VMA_NULL);
}

VkDeviceSize OaVmaAllocator_T::CalcPreferredBlockSize(uint32_t memTypeIndex)
{
    const uint32_t heapIndex = MemoryTypeIndexToHeapIndex(memTypeIndex);
    const VkDeviceSize heapSize = m_MemProps.memoryHeaps[heapIndex].size;
    const bool isSmallHeap = heapSize <= OA_VMA_SMALL_HEAP_MAX_SIZE;
    return OaVmaAlignUp(isSmallHeap ? (heapSize / 8) : m_PreferredLargeHeapBlockSize, (VkDeviceSize)32);
}

VkResult OaVmaAllocator_T::AllocateMemoryOfType(
    OaVmaPool pool,
    VkDeviceSize size,
    VkDeviceSize alignment,
    bool dedicatedPreferred,
    VkBuffer dedicatedBuffer,
    VkImage dedicatedImage,
    OaVmaBufferImageUsage dedicatedBufferImageUsage,
    void* pMemoryAllocateNext,
    const OaVmaAllocationCreateInfo& createInfo,
    uint32_t memTypeIndex,
    OaVmaSuballocationType suballocType,
    OaVmaDedicatedAllocationList& dedicatedAllocations,
    OaVmaBlockVector& blockVector,
    size_t allocationCount,
    OaVmaAllocation* pAllocations)
{
    OA_VMA_ASSERT(pAllocations != OA_VMA_NULL);
    OA_VMA_DEBUG_LOG_FORMAT("  AllocateMemory: MemoryTypeIndex=%" PRIu32 ", AllocationCount=%zu, Size=%" PRIu64, memTypeIndex, allocationCount, size);

    OaVmaAllocationCreateInfo finalCreateInfo = createInfo;
    VkResult res = CalcMemTypeParams(
        finalCreateInfo,
        memTypeIndex,
        size,
        allocationCount);
    if(res != VK_SUCCESS)
        return res;

    const void* allocateNextPtr = blockVector.GetAllocationNextPtr();
    if(pMemoryAllocateNext != OA_VMA_NULL)
    {
        OA_VMA_ASSERT(allocateNextPtr == OA_VMA_NULL &&
            "You shouldn't create a dedicated allocation with a custom pMemoryAllocateNext if the pNext chain is already provided for this pool.");
        allocateNextPtr = pMemoryAllocateNext;
    }

    if((finalCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0)
    {
        return AllocateDedicatedMemory(
            pool,
            size,
            suballocType,
            dedicatedAllocations,
            memTypeIndex,
            (finalCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0,
            (finalCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0,
            (finalCreateInfo.flags &
                (OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0,
            (finalCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT) != 0,
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
        (finalCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) == 0 &&
        (pool == VK_NULL_HANDLE || !blockVector.HasExplicitBlockSize());

    if(canAllocateDedicated)
    {
        // Heuristics: Allocate dedicated memory if requested size if greater than half of preferred block size.
        if(size > blockVector.GetPreferredBlockSize() / 2)
        {
            dedicatedPreferred = true;
        }
        // Protection against creating each allocation as dedicated when we reach or exceed heap size/budget,
        // which can quickly deplete maxMemoryAllocationCount: Don't prefer dedicated allocations when above
        // 3/4 of the maximum allocation count.
        if(m_PhysicalDeviceProperties.limits.maxMemoryAllocationCount < UINT32_MAX / 4 &&
            m_DeviceMemoryCount.load() > m_PhysicalDeviceProperties.limits.maxMemoryAllocationCount * 3 / 4)
        {
            dedicatedPreferred = false;
        }

        if(dedicatedPreferred)
        {
            res = AllocateDedicatedMemory(
                pool,
                size,
                suballocType,
                dedicatedAllocations,
                memTypeIndex,
                (finalCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0,
                (finalCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0,
                (finalCreateInfo.flags &
                    (OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0,
                (finalCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT) != 0,
                finalCreateInfo.pUserData,
                finalCreateInfo.priority,
                dedicatedBuffer,
                dedicatedImage,
                dedicatedBufferImageUsage,
                allocationCount,
                pAllocations,
                allocateNextPtr);
            if(res == VK_SUCCESS)
            {
                // Succeeded: AllocateDedicatedMemory function already filled pMemory, nothing more to do here.
                OA_VMA_DEBUG_LOG("    Allocated as DedicatedMemory");
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
    if(canAllocateDedicated && !dedicatedPreferred)
    {
        res = AllocateDedicatedMemory(
            pool,
            size,
            suballocType,
            dedicatedAllocations,
            memTypeIndex,
            (finalCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0,
            (finalCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0,
            (finalCreateInfo.flags &
                (OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0,
            (finalCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT) != 0,
            finalCreateInfo.pUserData,
            finalCreateInfo.priority,
            dedicatedBuffer,
            dedicatedImage,
            dedicatedBufferImageUsage,
            allocationCount,
            pAllocations,
            allocateNextPtr);
        if(res == VK_SUCCESS)
        {
            // Succeeded: AllocateDedicatedMemory function already filled pMemory, nothing more to do here.
            OA_VMA_DEBUG_LOG("    Allocated as DedicatedMemory");
            return VK_SUCCESS;
        }
    }

    // Everything failed: Return error code.
    OA_VMA_DEBUG_LOG("    vkAllocateMemory FAILED");
    return res;
}

VkResult OaVmaAllocator_T::AllocateDedicatedMemory(
    OaVmaPool pool,
    VkDeviceSize size,
    OaVmaSuballocationType suballocType,
    OaVmaDedicatedAllocationList& dedicatedAllocations,
    uint32_t memTypeIndex,
    bool map,
    bool isUserDataString,
    bool isMappingAllowed,
    bool canAliasMemory,
    void* pUserData,
    float priority,
    VkBuffer dedicatedBuffer,
    VkImage dedicatedImage,
    OaVmaBufferImageUsage dedicatedBufferImageUsage,
    size_t allocationCount,
    OaVmaAllocation* pAllocations,
    const void* pNextChain)
{
    OA_VMA_ASSERT(allocationCount > 0 && pAllocations);

    VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.memoryTypeIndex = memTypeIndex;
    allocInfo.allocationSize = size;
    allocInfo.pNext = pNextChain;

#if OA_VMA_DEDICATED_ALLOCATION || OA_VMA_VULKAN_VERSION >= 1001000
    VkMemoryDedicatedAllocateInfoKHR dedicatedAllocInfo = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR };
    if(!canAliasMemory)
    {
        if(m_UseKhrDedicatedAllocation || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
        {
            if(dedicatedBuffer != VK_NULL_HANDLE)
            {
                OA_VMA_ASSERT(dedicatedImage == VK_NULL_HANDLE);
                dedicatedAllocInfo.buffer = dedicatedBuffer;
                OaVmaPnextChainPushFront(&allocInfo, &dedicatedAllocInfo);
            }
            else if(dedicatedImage != VK_NULL_HANDLE)
            {
                dedicatedAllocInfo.image = dedicatedImage;
                OaVmaPnextChainPushFront(&allocInfo, &dedicatedAllocInfo);
            }
        }
    }
#endif // #if OA_VMA_DEDICATED_ALLOCATION || OA_VMA_VULKAN_VERSION >= 1001000

#if OA_VMA_BUFFER_DEVICE_ADDRESS
    VkMemoryAllocateFlagsInfoKHR allocFlagsInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR };
    if(m_UseKhrBufferDeviceAddress)
    {
        bool canContainBufferWithDeviceAddress = true;
        if(dedicatedBuffer != VK_NULL_HANDLE)
        {
            canContainBufferWithDeviceAddress = dedicatedBufferImageUsage == OaVmaBufferImageUsage::UNKNOWN ||
                dedicatedBufferImageUsage.Contains(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT);
        }
        else if(dedicatedImage != VK_NULL_HANDLE)
        {
            canContainBufferWithDeviceAddress = false;
        }
        if(canContainBufferWithDeviceAddress)
        {
            allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
            OaVmaPnextChainPushFront(&allocInfo, &allocFlagsInfo);
        }
    }
#endif // #if OA_VMA_BUFFER_DEVICE_ADDRESS

#if OA_VMA_MEMORY_PRIORITY
    VkMemoryPriorityAllocateInfoEXT priorityInfo = { VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT };
    if(m_UseExtMemoryPriority)
    {
        OA_VMA_ASSERT(priority >= 0.F && priority <= 1.F);
        priorityInfo.priority = priority;
        OaVmaPnextChainPushFront(&allocInfo, &priorityInfo);
    }
#endif // #if OA_VMA_MEMORY_PRIORITY

#if OA_VMA_EXTERNAL_MEMORY
    // Attach VkExportMemoryAllocateInfoKHR if necessary.
    VkExportMemoryAllocateInfoKHR exportMemoryAllocInfo = { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR };
    exportMemoryAllocInfo.handleTypes = GetExternalMemoryHandleTypeFlags(memTypeIndex);
    if(exportMemoryAllocInfo.handleTypes != 0)
    {
        OaVmaPnextChainPushFront(&allocInfo, &exportMemoryAllocInfo);
    }
#endif // #if OA_VMA_EXTERNAL_MEMORY

    size_t allocIndex = 0;
    VkResult res = VK_SUCCESS;
    for(; allocIndex < allocationCount; ++allocIndex)
    {
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

    if(res == VK_SUCCESS)
    {
        for (allocIndex = 0; allocIndex < allocationCount; ++allocIndex)
        {
            dedicatedAllocations.Register(pAllocations[allocIndex]);
        }
        OA_VMA_DEBUG_LOG_FORMAT("    Allocated DedicatedMemory Count=%zu, MemoryTypeIndex=#%" PRIu32, allocationCount, memTypeIndex);
    }
    else
    {
        // Free all already created allocations.
        while(allocIndex--)
        {
            OaVmaAllocation currAlloc = pAllocations[allocIndex];
            VkDeviceMemory hMemory = currAlloc->GetMemory();

            /*
            There is no need to call this, because Vulkan spec allows to skip vkUnmapMemory
            before vkFreeMemory.

            if(currAlloc->GetMappedData() != OA_VMA_NULL)
            {
                (*m_VulkanFunctions.vkUnmapMemory)(m_hDevice, hMemory);
            }
            */

            FreeVulkanMemory(memTypeIndex, currAlloc->GetSize(), hMemory);
            m_Budget.RemoveAllocation(MemoryTypeIndexToHeapIndex(memTypeIndex), currAlloc->GetSize());
            m_AllocationObjectAllocator.Free(currAlloc);
        }

        memset(pAllocations, 0, sizeof(OaVmaAllocation) * allocationCount);
    }

    return res;
}

VkResult OaVmaAllocator_T::AllocateDedicatedMemoryPage(
    OaVmaPool pool,
    VkDeviceSize size,
    OaVmaSuballocationType suballocType,
    uint32_t memTypeIndex,
    const VkMemoryAllocateInfo& allocInfo,
    bool map,
    bool isUserDataString,
    bool isMappingAllowed,
    void* pUserData,
    OaVmaAllocation* pAllocation)
{
    VkDeviceMemory hMemory = VK_NULL_HANDLE;
    VkResult res = AllocateVulkanMemory(&allocInfo, &hMemory);
    if(res < 0)
    {
        OA_VMA_DEBUG_LOG("    vkAllocateMemory FAILED");
        return res;
    }

    void* pMappedData = OA_VMA_NULL;
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
            OA_VMA_DEBUG_LOG("    vkMapMemory FAILED");
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

#if OA_VMA_DEBUG_INITIALIZE_ALLOCATIONS
    FillAllocation(*pAllocation, OA_VMA_ALLOCATION_FILL_PATTERN_CREATED);
#endif

    return VK_SUCCESS;
}

void OaVmaAllocator_T::GetBufferMemoryRequirements(
    VkBuffer hBuffer,
    VkMemoryRequirements& memReq,
    bool& requiresDedicatedAllocation,
    bool& prefersDedicatedAllocation) const
{
#if OA_VMA_DEDICATED_ALLOCATION || OA_VMA_VULKAN_VERSION >= 1001000
    if(m_UseKhrDedicatedAllocation || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        VkBufferMemoryRequirementsInfo2KHR memReqInfo = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR };
        memReqInfo.buffer = hBuffer;

        VkMemoryDedicatedRequirementsKHR memDedicatedReq = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR };

        VkMemoryRequirements2KHR memReq2 = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR };
        OaVmaPnextChainPushFront(&memReq2, &memDedicatedReq);

        (*m_VulkanFunctions.vkGetBufferMemoryRequirements2KHR)(m_hDevice, &memReqInfo, &memReq2);

        memReq = memReq2.memoryRequirements;
        requiresDedicatedAllocation = (memDedicatedReq.requiresDedicatedAllocation != VK_FALSE);
        prefersDedicatedAllocation  = (memDedicatedReq.prefersDedicatedAllocation  != VK_FALSE);
    }
    else
#endif // #if OA_VMA_DEDICATED_ALLOCATION || OA_VMA_VULKAN_VERSION >= 1001000
    {
        (*m_VulkanFunctions.vkGetBufferMemoryRequirements)(m_hDevice, hBuffer, &memReq);
        requiresDedicatedAllocation = false;
        prefersDedicatedAllocation  = false;
    }
}

void OaVmaAllocator_T::GetImageMemoryRequirements(
    VkImage hImage,
    VkMemoryRequirements& memReq,
    bool& requiresDedicatedAllocation,
    bool& prefersDedicatedAllocation) const
{
#if OA_VMA_DEDICATED_ALLOCATION || OA_VMA_VULKAN_VERSION >= 1001000
    if(m_UseKhrDedicatedAllocation || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        VkImageMemoryRequirementsInfo2KHR memReqInfo = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR };
        memReqInfo.image = hImage;

        VkMemoryDedicatedRequirementsKHR memDedicatedReq = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR };

        VkMemoryRequirements2KHR memReq2 = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR };
        OaVmaPnextChainPushFront(&memReq2, &memDedicatedReq);

        (*m_VulkanFunctions.vkGetImageMemoryRequirements2KHR)(m_hDevice, &memReqInfo, &memReq2);

        memReq = memReq2.memoryRequirements;
        requiresDedicatedAllocation = (memDedicatedReq.requiresDedicatedAllocation != VK_FALSE);
        prefersDedicatedAllocation  = (memDedicatedReq.prefersDedicatedAllocation  != VK_FALSE);
    }
    else
#endif // #if OA_VMA_DEDICATED_ALLOCATION || OA_VMA_VULKAN_VERSION >= 1001000
    {
        (*m_VulkanFunctions.vkGetImageMemoryRequirements)(m_hDevice, hImage, &memReq);
        requiresDedicatedAllocation = false;
        prefersDedicatedAllocation  = false;
    }
}

VkResult OaVmaAllocator_T::FindMemoryTypeIndex(
    uint32_t memoryTypeBits,
    const OaVmaAllocationCreateInfo* pAllocationCreateInfo,
    OaVmaBufferImageUsage bufImgUsage,
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
        if((memTypeBit & memoryTypeBits) != 0)
        {
            const VkMemoryPropertyFlags currFlags =
                m_MemProps.memoryTypes[memTypeIndex].propertyFlags;
            // This memory type contains requiredFlags.
            if((requiredFlags & ~currFlags) == 0)
            {
                // Calculate cost as number of bits from preferredFlags not present in this memory type.
                uint32_t currCost = OA_VMA_COUNT_BITS_SET(preferredFlags & ~currFlags) +
                    OA_VMA_COUNT_BITS_SET(currFlags & notPreferredFlags);
                // Remember memory type with lowest cost.
                if(currCost < minCost)
                {
                    *pMemoryTypeIndex = memTypeIndex;
                    if(currCost == 0)
                    {
                        return VK_SUCCESS;
                    }
                    minCost = currCost;
                }
            }
        }
    }
    return (*pMemoryTypeIndex != UINT32_MAX) ? VK_SUCCESS : VK_ERROR_FEATURE_NOT_PRESENT;
}

VkResult OaVmaAllocator_T::CalcMemTypeParams(
    OaVmaAllocationCreateInfo& inoutCreateInfo,
    uint32_t memTypeIndex,
    VkDeviceSize size,
    size_t allocationCount)
{
    // If memory type is not HOST_VISIBLE, disable MAPPED.
    if((inoutCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0 &&
        (m_MemProps.memoryTypes[memTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
    {
        inoutCreateInfo.flags &= ~OA_VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    if((inoutCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0 &&
        (inoutCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT) != 0)
    {
        const uint32_t heapIndex = MemoryTypeIndexToHeapIndex(memTypeIndex);
        OaVmaBudget heapBudget = {};
        GetHeapBudgets(&heapBudget, heapIndex, 1);
        if(heapBudget.usage + size * allocationCount > heapBudget.budget)
        {
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }
    return VK_SUCCESS;
}

VkResult OaVmaAllocator_T::CalcAllocationParams(
    OaVmaAllocationCreateInfo& inoutCreateInfo,
    bool dedicatedRequired)
{
    OA_VMA_ASSERT((inoutCreateInfo.flags &
        (OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) !=
        (OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT) &&
        "Specifying both flags OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT and OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT is incorrect.");
    OA_VMA_ASSERT((((inoutCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT) == 0 ||
        (inoutCreateInfo.flags & (OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0)) &&
        "Specifying OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT requires also OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.");
    if(inoutCreateInfo.usage == OA_VMA_MEMORY_USAGE_AUTO || inoutCreateInfo.usage == OA_VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE || inoutCreateInfo.usage == OA_VMA_MEMORY_USAGE_AUTO_PREFER_HOST)
    {
        if((inoutCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0)
        {
            OA_VMA_ASSERT((inoutCreateInfo.flags & (OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0 &&
                "When using OA_VMA_ALLOCATION_CREATE_MAPPED_BIT and usage = OA_VMA_MEMORY_USAGE_AUTO*, you must also specify OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.");
        }
    }

    // If memory is lazily allocated, it should be always dedicated.
    if(dedicatedRequired ||
        inoutCreateInfo.usage == OA_VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED)
    {
        inoutCreateInfo.flags |= OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }

    if(inoutCreateInfo.pool != VK_NULL_HANDLE)
    {
        if(inoutCreateInfo.pool->m_BlockVector.HasExplicitBlockSize() &&
            (inoutCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0)
        {
            OA_VMA_ASSERT(0 && "Specifying OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT while current custom pool doesn't support dedicated allocations.");
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        inoutCreateInfo.priority = inoutCreateInfo.pool->m_BlockVector.GetPriority();
    }

    if((inoutCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0 &&
        (inoutCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) != 0)
    {
        OA_VMA_ASSERT(0 && "Specifying OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT together with OA_VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT makes no sense.");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

#if OA_VMA_DEBUG_ALWAYS_DEDICATED_MEMORY
    if((inoutCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) != 0)
    {
        inoutCreateInfo.flags |= OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }
#endif

    // Non-auto USAGE values imply HOST_ACCESS flags.
    // And so does OA_VMA_MEMORY_USAGE_UNKNOWN because it is used with custom pools.
    // Which specific flag is used doesn't matter. They change things only when used with OA_VMA_MEMORY_USAGE_AUTO*.
    // Otherwise they just protect from assert on mapping.
    if(inoutCreateInfo.usage != OA_VMA_MEMORY_USAGE_AUTO &&
        inoutCreateInfo.usage != OA_VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE &&
        inoutCreateInfo.usage != OA_VMA_MEMORY_USAGE_AUTO_PREFER_HOST)
    {
        if((inoutCreateInfo.flags & (OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) == 0)
        {
            inoutCreateInfo.flags |= OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        }
    }

    return VK_SUCCESS;
}

VkResult OaVmaAllocator_T::CreateBuffer(
    const VkBufferCreateInfo* pBufferCreateInfo,
    const OaVmaAllocationCreateInfo* pAllocationCreateInfo,
    void* pMemoryAllocateNext,
    VkBuffer* pBuffer,
    OaVmaAllocation* pAllocation,
    OaVmaAllocationInfo* pAllocationInfo)
{
    *pBuffer = VK_NULL_HANDLE;
    *pAllocation = VK_NULL_HANDLE;

    if (pBufferCreateInfo->size == 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if ((pBufferCreateInfo->usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_COPY) != 0 &&
        !m_UseKhrBufferDeviceAddress)
    {
        OA_VMA_ASSERT(0 && "Creating a buffer with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT is not valid if OA_VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT was not used.");
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
            OaVmaBufferImageUsage(*pBufferCreateInfo, m_UseKhrMaintenance5), // dedicatedBufferImageUsage
            pMemoryAllocateNext,
            *pAllocationCreateInfo,
            OA_VMA_SUBALLOCATION_TYPE_BUFFER,
            1, // allocationCount
            pAllocation);
        if (res >= 0)
        {
            // 3. Bind buffer with memory.
            if ((pAllocationCreateInfo->flags & OA_VMA_ALLOCATION_CREATE_DONT_BIND_BIT) == 0)
            {
                res = BindBufferMemory(*pAllocation, 0, *pBuffer, OA_VMA_NULL);
            }
            if (res >= 0)
            {
                // All steps succeeded.
#if OA_VMA_STATS_STRING_ENABLED
                (*pAllocation)->InitBufferUsage(*pBufferCreateInfo, m_UseKhrMaintenance5);
#endif
                if (pAllocationInfo != OA_VMA_NULL)
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

VkResult OaVmaAllocator_T::CreateImage(
    const VkImageCreateInfo* pImageCreateInfo,
    const OaVmaAllocationCreateInfo* pAllocationCreateInfo,
    void* pMemoryAllocateNext,
    VkImage* pImage,
    OaVmaAllocation* pAllocation,
    OaVmaAllocationInfo* pAllocationInfo)
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
        OaVmaSuballocationType suballocType = pImageCreateInfo->tiling == VK_IMAGE_TILING_OPTIMAL ?
            OA_VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL :
            OA_VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR;

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
            OaVmaBufferImageUsage(*pImageCreateInfo), // dedicatedBufferImageUsage
            pMemoryAllocateNext,
            *pAllocationCreateInfo,
            suballocType,
            1, // allocationCount
            pAllocation);
        if (res == VK_SUCCESS)
        {
            // 3. Bind image with memory.
            if ((pAllocationCreateInfo->flags & OA_VMA_ALLOCATION_CREATE_DONT_BIND_BIT) == 0)
            {
                res = BindImageMemory(*pAllocation, 0, *pImage, OA_VMA_NULL);
            }
            if (res == VK_SUCCESS)
            {
                // All steps succeeded.
#if OA_VMA_STATS_STRING_ENABLED
                (*pAllocation)->InitImageUsage(*pImageCreateInfo);
#endif
                if (pAllocationInfo != OA_VMA_NULL)
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

VkResult OaVmaAllocator_T::AllocateMemory(
    VkMemoryRequirements vkMemReq,
    bool requiresDedicatedAllocation,
    bool prefersDedicatedAllocation,
    VkBuffer dedicatedBuffer,
    VkImage dedicatedImage,
    OaVmaBufferImageUsage dedicatedBufferImageUsage,
    // pNext chain for VkMemoryAllocateInfo. When used, must specify requiresDedicatedAllocation = true.
    void* pMemoryAllocateNext,
    const OaVmaAllocationCreateInfo& createInfo,
    OaVmaSuballocationType suballocType,
    size_t allocationCount,
    OaVmaAllocation* pAllocations)
{
    memset(pAllocations, 0, sizeof(OaVmaAllocation) * allocationCount);

    vkMemReq.alignment = OA_VMA_MAX(vkMemReq.alignment, createInfo.minAlignment);
    OA_VMA_ASSERT(OaVmaIsPow2(vkMemReq.alignment));

    // If using custom pNext chain for VkMemoryAllocateInfo, must require dedicated allocations.
    if(pMemoryAllocateNext != OA_VMA_NULL)
    {
        requiresDedicatedAllocation = true;
    }

    if(vkMemReq.size == 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    OaVmaAllocationCreateInfo createInfoFinal = createInfo;
    VkResult res = CalcAllocationParams(createInfoFinal, requiresDedicatedAllocation);
    if(res != VK_SUCCESS)
        return res;

    if(createInfoFinal.pool != VK_NULL_HANDLE)
    {
        OaVmaBlockVector& blockVector = createInfoFinal.pool->m_BlockVector;
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
        OaVmaBlockVector* blockVector = m_pBlockVectors[memTypeIndex];
        OA_VMA_ASSERT(blockVector && "Trying to use unsupported memory type!");
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

void OaVmaAllocator_T::FreeMemory(
    size_t allocationCount,
    const OaVmaAllocation* pAllocations)
{
    OA_VMA_ASSERT(pAllocations);

    for(size_t allocIndex = allocationCount; allocIndex--; )
    {
        OaVmaAllocation allocation = pAllocations[allocIndex];

        if(allocation != VK_NULL_HANDLE)
        {
#if OA_VMA_DEBUG_INITIALIZE_ALLOCATIONS
            FillAllocation(allocation, OA_VMA_ALLOCATION_FILL_PATTERN_DESTROYED);
#endif

            switch(allocation->GetType())
            {
            case OaVmaAllocation_T::ALLOCATION_TYPE_BLOCK:
                {
                    OaVmaBlockVector* pBlockVector = OA_VMA_NULL;
                    OaVmaPool hPool = allocation->GetParentPool();
                    if(hPool != VK_NULL_HANDLE)
                    {
                        pBlockVector = &hPool->m_BlockVector;
                    }
                    else
                    {
                        const uint32_t memTypeIndex = allocation->GetMemoryTypeIndex();
                        pBlockVector = m_pBlockVectors[memTypeIndex];
                        OA_VMA_ASSERT(pBlockVector && "Trying to free memory of unsupported type!");
                    }
                    pBlockVector->Free(allocation);
                }
                break;
            case OaVmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
                FreeDedicatedMemory(allocation);
                break;
            default:
                OA_VMA_ASSERT(0);
            }
        }
    }
}

void OaVmaAllocator_T::CalculateStatistics(OaVmaTotalStatistics* pStats)
{
    // Initialize.
    OaVmaClearDetailedStatistics(pStats->total);
    for(uint32_t i = 0; i < VK_MAX_MEMORY_TYPES; ++i)
        OaVmaClearDetailedStatistics(pStats->memoryType[i]);
    for(uint32_t i = 0; i < VK_MAX_MEMORY_HEAPS; ++i)
        OaVmaClearDetailedStatistics(pStats->memoryHeap[i]);

    // Process default pools.
    for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
    {
        OaVmaBlockVector* const pBlockVector = m_pBlockVectors[memTypeIndex];
        if (pBlockVector != OA_VMA_NULL)
            pBlockVector->AddDetailedStatistics(pStats->memoryType[memTypeIndex]);
    }

    // Process custom pools.
    {
        OaVmaMutexLockRead lock(m_PoolsMutex, m_UseMutex);
        for(OaVmaPool pool = m_Pools.Front(); pool != OA_VMA_NULL; pool = m_Pools.GetNext(pool))
        {
            OaVmaBlockVector& blockVector = pool->m_BlockVector;
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
        OaVmaAddDetailedStatistics(pStats->memoryHeap[memHeapIndex], pStats->memoryType[memTypeIndex]);
    }

    // Sum from memory heaps to total.
    for(uint32_t memHeapIndex = 0; memHeapIndex < GetMemoryHeapCount(); ++memHeapIndex)
        OaVmaAddDetailedStatistics(pStats->total, pStats->memoryHeap[memHeapIndex]);

    OA_VMA_ASSERT(pStats->total.statistics.allocationCount == 0 ||
        pStats->total.allocationSizeMax >= pStats->total.allocationSizeMin);
    OA_VMA_ASSERT(pStats->total.unusedRangeCount == 0 ||
        pStats->total.unusedRangeSizeMax >= pStats->total.unusedRangeSizeMin);
}

void OaVmaAllocator_T::GetHeapBudgets(OaVmaBudget* outBudgets, uint32_t firstHeap, uint32_t heapCount)
{
#if OA_VMA_MEMORY_BUDGET
    if(m_UseExtMemoryBudget)
    {
        if(m_Budget.m_OperationsSinceBudgetFetch < 30)
        {
            OaVmaMutexLockRead lockRead(m_Budget.m_BudgetMutex, m_UseMutex);
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
                outBudgets->budget = OA_VMA_MIN(
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

void OaVmaAllocator_T::GetAllocationInfo(OaVmaAllocation hAllocation, OaVmaAllocationInfo* pAllocationInfo)
{
    pAllocationInfo->memoryType = hAllocation->GetMemoryTypeIndex();
    pAllocationInfo->deviceMemory = hAllocation->GetMemory();
    pAllocationInfo->offset = hAllocation->GetOffset();
    pAllocationInfo->size = hAllocation->GetSize();
    pAllocationInfo->pMappedData = hAllocation->GetMappedData();
    pAllocationInfo->pUserData = hAllocation->GetUserData();
    pAllocationInfo->pName = hAllocation->GetName();
}

void OaVmaAllocator_T::GetAllocationInfo2(OaVmaAllocation hAllocation, OaVmaAllocationInfo2* pAllocationInfo)
{
    GetAllocationInfo(hAllocation, &pAllocationInfo->allocationInfo);

    switch (hAllocation->GetType())
    {
    case OaVmaAllocation_T::ALLOCATION_TYPE_BLOCK:
        pAllocationInfo->blockSize = hAllocation->GetBlock()->m_pMetadata->GetSize();
        pAllocationInfo->dedicatedMemory = VK_FALSE;
        break;
    case OaVmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        pAllocationInfo->blockSize = pAllocationInfo->allocationInfo.size;
        pAllocationInfo->dedicatedMemory = VK_TRUE;
        break;
    default:
        OA_VMA_ASSERT(0);
    }
}

VkResult OaVmaAllocator_T::CreatePool(const OaVmaPoolCreateInfo* pCreateInfo, OaVmaPool* pPool)
{
    OA_VMA_DEBUG_LOG_FORMAT("  CreatePool: MemoryTypeIndex=%" PRIu32 ", flags=%" PRIu32, pCreateInfo->memoryTypeIndex, pCreateInfo->flags);

    OaVmaPoolCreateInfo newCreateInfo = *pCreateInfo;

    // Protection against uninitialized new structure member. If garbage data are left there, this pointer dereference would crash.
    if(pCreateInfo->pMemoryAllocateNext)
    {
        OA_VMA_ASSERT(((const VkBaseInStructure*)pCreateInfo->pMemoryAllocateNext)->sType != 0);
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
        OA_VMA_ASSERT(OaVmaIsPow2(newCreateInfo.minAllocationAlignment));
    }

    const VkDeviceSize preferredBlockSize = CalcPreferredBlockSize(newCreateInfo.memoryTypeIndex);

    *pPool = OaVma_new(this, OaVmaPool_T)(this, newCreateInfo, preferredBlockSize);

    VkResult res = (*pPool)->m_BlockVector.CreateMinBlocks();
    if(res != VK_SUCCESS)
    {
        OaVma_delete(this, *pPool);
        *pPool = OA_VMA_NULL;
        return res;
    }

    // Add to m_Pools.
    {
        OaVmaMutexLockWrite lock(m_PoolsMutex, m_UseMutex);
        (*pPool)->SetId(m_NextPoolId++);
        m_Pools.PushBack(*pPool);
    }

    return VK_SUCCESS;
}

void OaVmaAllocator_T::DestroyPool(OaVmaPool pool)
{
    // Remove from m_Pools.
    {
        OaVmaMutexLockWrite lock(m_PoolsMutex, m_UseMutex);
        m_Pools.Remove(pool);
    }

    OaVma_delete(this, pool);
}

void OaVmaAllocator_T::GetPoolStatistics(OaVmaPool pool, OaVmaStatistics* pPoolStats)
{
    OaVmaClearStatistics(*pPoolStats);
    pool->m_BlockVector.AddStatistics(*pPoolStats);
    pool->m_DedicatedAllocations.AddStatistics(*pPoolStats);
}

void OaVmaAllocator_T::CalculatePoolStatistics(OaVmaPool pool, OaVmaDetailedStatistics* pPoolStats)
{
    OaVmaClearDetailedStatistics(*pPoolStats);
    pool->m_BlockVector.AddDetailedStatistics(*pPoolStats);
    pool->m_DedicatedAllocations.AddDetailedStatistics(*pPoolStats);
}

void OaVmaAllocator_T::SetCurrentFrameIndex(uint32_t frameIndex)
{
    m_CurrentFrameIndex.store(frameIndex);

#if OA_VMA_MEMORY_BUDGET
    if(m_UseExtMemoryBudget)
    {
        UpdateVulkanBudget();
    }
#endif // #if OA_VMA_MEMORY_BUDGET
}

VkResult OaVmaAllocator_T::CheckPoolCorruption(OaVmaPool hPool)
{
    return hPool->m_BlockVector.CheckCorruption();
}

VkResult OaVmaAllocator_T::CheckCorruption(uint32_t memoryTypeBits)
{
    VkResult finalRes = VK_ERROR_FEATURE_NOT_PRESENT;

    // Process default pools.
    for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
    {
        OaVmaBlockVector* const pBlockVector = m_pBlockVectors[memTypeIndex];
        if(pBlockVector != OA_VMA_NULL)
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
        OaVmaMutexLockRead lock(m_PoolsMutex, m_UseMutex);
        for(OaVmaPool pool = m_Pools.Front(); pool != OA_VMA_NULL; pool = m_Pools.GetNext(pool))
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

VkResult OaVmaAllocator_T::AllocateVulkanMemory(const VkMemoryAllocateInfo* pAllocateInfo, VkDeviceMemory* pMemory)
{
    const uint32_t heapIndex = MemoryTypeIndexToHeapIndex(pAllocateInfo->memoryTypeIndex);

#if OA_VMA_DEBUG_DONT_EXCEED_HEAP_SIZE_WITH_ALLOCATION_SIZE
    if (pAllocateInfo->allocationSize > m_MemProps.memoryHeaps[heapIndex].size)
    {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
#endif

    AtomicTransactionalIncrement<OA_VMA_ATOMIC_UINT32> deviceMemoryCountIncrement;
    const uint64_t prevDeviceMemoryCount = deviceMemoryCountIncrement.Increment(&m_DeviceMemoryCount);
#if OA_VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT
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
            if(m_Budget.m_BlockBytes[heapIndex].compare_exchange_strong(blockBytes, blockBytesAfterAllocation))
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
#if OA_VMA_MEMORY_BUDGET
        ++m_Budget.m_OperationsSinceBudgetFetch;
#endif

        // Informative callback.
        if(m_DeviceMemoryCallbacks.pfnAllocate != OA_VMA_NULL)
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

void OaVmaAllocator_T::FreeVulkanMemory(uint32_t memoryType, VkDeviceSize size, VkDeviceMemory hMemory)
{
    // Informative callback.
    if(m_DeviceMemoryCallbacks.pfnFree != OA_VMA_NULL)
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

VkResult OaVmaAllocator_T::BindVulkanBuffer(
    VkDeviceMemory memory,
    VkDeviceSize memoryOffset,
    VkBuffer buffer,
    const void* pNext) const
{
    if(pNext != OA_VMA_NULL)
    {
#if OA_VMA_VULKAN_VERSION >= 1001000 || OA_VMA_BIND_MEMORY2
        if((m_UseKhrBindMemory2 || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0)) &&
            m_VulkanFunctions.vkBindBufferMemory2KHR != OA_VMA_NULL)
        {
            VkBindBufferMemoryInfoKHR bindBufferMemoryInfo = { VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO_KHR };
            bindBufferMemoryInfo.pNext = pNext;
            bindBufferMemoryInfo.buffer = buffer;
            bindBufferMemoryInfo.memory = memory;
            bindBufferMemoryInfo.memoryOffset = memoryOffset;
            return (*m_VulkanFunctions.vkBindBufferMemory2KHR)(m_hDevice, 1, &bindBufferMemoryInfo);
        }
#endif // #if OA_VMA_VULKAN_VERSION >= 1001000 || OA_VMA_BIND_MEMORY2

        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    else
    {
        return (*m_VulkanFunctions.vkBindBufferMemory)(m_hDevice, buffer, memory, memoryOffset);
    }
}

VkResult OaVmaAllocator_T::BindVulkanImage(
    VkDeviceMemory memory,
    VkDeviceSize memoryOffset,
    VkImage image,
    const void* pNext) const
{
    if(pNext != OA_VMA_NULL)
    {
#if OA_VMA_VULKAN_VERSION >= 1001000 || OA_VMA_BIND_MEMORY2
        if((m_UseKhrBindMemory2 || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0)) &&
            m_VulkanFunctions.vkBindImageMemory2KHR != OA_VMA_NULL)
        {
            VkBindImageMemoryInfoKHR bindBufferMemoryInfo = { VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO_KHR };
            bindBufferMemoryInfo.pNext = pNext;
            bindBufferMemoryInfo.image = image;
            bindBufferMemoryInfo.memory = memory;
            bindBufferMemoryInfo.memoryOffset = memoryOffset;
            return (*m_VulkanFunctions.vkBindImageMemory2KHR)(m_hDevice, 1, &bindBufferMemoryInfo);
        }
#endif // #if OA_VMA_BIND_MEMORY2

        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    return (*m_VulkanFunctions.vkBindImageMemory)(m_hDevice, image, memory, memoryOffset);
}

VkResult OaVmaAllocator_T::Map(OaVmaAllocation hAllocation, void** ppData)
{
    switch(hAllocation->GetType())
    {
    case OaVmaAllocation_T::ALLOCATION_TYPE_BLOCK:
        {
            OaVmaDeviceMemoryBlock* const pBlock = hAllocation->GetBlock();
            char *pBytes = OA_VMA_NULL;
            VkResult res = pBlock->Map(this, 1, (void**)&pBytes);
            if(res == VK_SUCCESS)
            {
                *ppData = pBytes + (ptrdiff_t)hAllocation->GetOffset();
                hAllocation->BlockAllocMap();
            }
            return res;
        }
    case OaVmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        return hAllocation->DedicatedAllocMap(this, ppData);
    default:
        OA_VMA_ASSERT(0);
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
}

void OaVmaAllocator_T::Unmap(OaVmaAllocation hAllocation)
{
    switch(hAllocation->GetType())
    {
    case OaVmaAllocation_T::ALLOCATION_TYPE_BLOCK:
        {
            OaVmaDeviceMemoryBlock* const pBlock = hAllocation->GetBlock();
            hAllocation->BlockAllocUnmap();
            pBlock->Unmap(this, 1);
        }
        break;
    case OaVmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        hAllocation->DedicatedAllocUnmap(this);
        break;
    default:
        OA_VMA_ASSERT(0);
    }
}

VkResult OaVmaAllocator_T::BindBufferMemory(
    OaVmaAllocation hAllocation,
    VkDeviceSize allocationLocalOffset,
    VkBuffer hBuffer,
    const void* pNext)
{
    VkResult res = VK_ERROR_UNKNOWN_COPY;
    switch(hAllocation->GetType())
    {
    case OaVmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        res = BindVulkanBuffer(hAllocation->GetMemory(), allocationLocalOffset, hBuffer, pNext);
        break;
    case OaVmaAllocation_T::ALLOCATION_TYPE_BLOCK:
    {
        OaVmaDeviceMemoryBlock* const pBlock = hAllocation->GetBlock();
        OA_VMA_ASSERT(pBlock && "Binding buffer to allocation that doesn't belong to any block.");
        res = pBlock->BindBufferMemory(this, hAllocation, allocationLocalOffset, hBuffer, pNext);
        break;
    }
    default:
        OA_VMA_ASSERT(0);
    }
    return res;
}

VkResult OaVmaAllocator_T::BindImageMemory(
    OaVmaAllocation hAllocation,
    VkDeviceSize allocationLocalOffset,
    VkImage hImage,
    const void* pNext)
{
    VkResult res = VK_ERROR_UNKNOWN_COPY;
    switch(hAllocation->GetType())
    {
    case OaVmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        res = BindVulkanImage(hAllocation->GetMemory(), allocationLocalOffset, hImage, pNext);
        break;
    case OaVmaAllocation_T::ALLOCATION_TYPE_BLOCK:
    {
        OaVmaDeviceMemoryBlock* pBlock = hAllocation->GetBlock();
        OA_VMA_ASSERT(pBlock && "Binding image to allocation that doesn't belong to any block.");
        res = pBlock->BindImageMemory(this, hAllocation, allocationLocalOffset, hImage, pNext);
        break;
    }
    default:
        OA_VMA_ASSERT(0);
    }
    return res;
}

VkResult OaVmaAllocator_T::FlushOrInvalidateAllocation(
    OaVmaAllocation hAllocation,
    VkDeviceSize offset, VkDeviceSize size,
    OA_VMA_CACHE_OPERATION op)
{
    VkResult res = VK_SUCCESS;

    VkMappedMemoryRange memRange = {};
    if(GetFlushOrInvalidateRange(hAllocation, offset, size, memRange))
    {
        switch(op)
        {
        case OA_VMA_CACHE_FLUSH:
            res = (*GetVulkanFunctions().vkFlushMappedMemoryRanges)(m_hDevice, 1, &memRange);
            break;
        case OA_VMA_CACHE_INVALIDATE:
            res = (*GetVulkanFunctions().vkInvalidateMappedMemoryRanges)(m_hDevice, 1, &memRange);
            break;
        default:
            OA_VMA_ASSERT(0);
        }
    }
    // else: Just ignore this call.
    return res;
}

VkResult OaVmaAllocator_T::FlushOrInvalidateAllocations(
    uint32_t allocationCount,
    const OaVmaAllocation* allocations,
    const VkDeviceSize* offsets, const VkDeviceSize* sizes,
    OA_VMA_CACHE_OPERATION op)
{
    typedef OaVmaStlAllocator<VkMappedMemoryRange> RangeAllocator;
    typedef OaVmaSmallVector<VkMappedMemoryRange, RangeAllocator, 16> RangeVector;
    RangeVector ranges = RangeVector(RangeAllocator(GetAllocationCallbacks()));

    for(uint32_t allocIndex = 0; allocIndex < allocationCount; ++allocIndex)
    {
        const OaVmaAllocation alloc = allocations[allocIndex];
        const VkDeviceSize offset = offsets != OA_VMA_NULL ? offsets[allocIndex] : 0;
        const VkDeviceSize size = sizes != OA_VMA_NULL ? sizes[allocIndex] : VK_WHOLE_SIZE;
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
        case OA_VMA_CACHE_FLUSH:
            res = (*GetVulkanFunctions().vkFlushMappedMemoryRanges)(m_hDevice, (uint32_t)ranges.size(), ranges.data());
            break;
        case OA_VMA_CACHE_INVALIDATE:
            res = (*GetVulkanFunctions().vkInvalidateMappedMemoryRanges)(m_hDevice, (uint32_t)ranges.size(), ranges.data());
            break;
        default:
            OA_VMA_ASSERT(0);
        }
    }
    // else: Just ignore this call.
    return res;
}

VkResult OaVmaAllocator_T::CopyMemoryToAllocation(
    const void* pSrcHostPointer,
    OaVmaAllocation dstAllocation,
    VkDeviceSize dstAllocationLocalOffset,
    VkDeviceSize size)
{
    void* dstMappedData = OA_VMA_NULL;
    VkResult res = Map(dstAllocation, &dstMappedData);
    if(res == VK_SUCCESS)
    {
        memcpy((char*)dstMappedData + dstAllocationLocalOffset, pSrcHostPointer, (size_t)size);
        Unmap(dstAllocation);
        res = FlushOrInvalidateAllocation(dstAllocation, dstAllocationLocalOffset, size, OA_VMA_CACHE_FLUSH);
    }
    return res;
}

VkResult OaVmaAllocator_T::CopyAllocationToMemory(
    OaVmaAllocation srcAllocation,
    VkDeviceSize srcAllocationLocalOffset,
    void* pDstHostPointer,
    VkDeviceSize size)
{
    void* srcMappedData = OA_VMA_NULL;
    VkResult res = Map(srcAllocation, &srcMappedData);
    if(res == VK_SUCCESS)
    {
        res = FlushOrInvalidateAllocation(srcAllocation, srcAllocationLocalOffset, size, OA_VMA_CACHE_INVALIDATE);
        if(res == VK_SUCCESS)
        {
            memcpy(pDstHostPointer, (const char*)srcMappedData + srcAllocationLocalOffset, (size_t)size);
            Unmap(srcAllocation);
        }
    }
    return res;
}

void OaVmaAllocator_T::FreeDedicatedMemory(OaVmaAllocation allocation)
{
    OA_VMA_ASSERT(allocation && allocation->GetType() == OaVmaAllocation_T::ALLOCATION_TYPE_DEDICATED);

    const uint32_t memTypeIndex = allocation->GetMemoryTypeIndex();
    OaVmaPool parentPool = allocation->GetParentPool();
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

    if(allocation->GetMappedData() != OA_VMA_NULL)
    {
        (*m_VulkanFunctions.vkUnmapMemory)(m_hDevice, hMemory);
    }
    */

    FreeVulkanMemory(memTypeIndex, allocation->GetSize(), hMemory);

    m_Budget.RemoveAllocation(MemoryTypeIndexToHeapIndex(allocation->GetMemoryTypeIndex()), allocation->GetSize());
    allocation->Destroy(this);
    m_AllocationObjectAllocator.Free(allocation);

    OA_VMA_DEBUG_LOG_FORMAT("    Freed DedicatedMemory MemoryTypeIndex=%" PRIu32, memTypeIndex);
}

uint32_t OaVmaAllocator_T::CalculateGpuDefragmentationMemoryTypeBits() const
{
    VkBufferCreateInfo dummyBufCreateInfo;
    OaVmaFillGpuDefragmentationBufferCreateInfo(dummyBufCreateInfo);

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

uint32_t OaVmaAllocator_T::CalculateGlobalMemoryTypeBits() const
{
    // Make sure memory information is already fetched.
    OA_VMA_ASSERT(GetMemoryTypeCount() > 0);

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

bool OaVmaAllocator_T::GetFlushOrInvalidateRange(
    OaVmaAllocation allocation,
    VkDeviceSize offset, VkDeviceSize size,
    VkMappedMemoryRange& outRange) const
{
    const uint32_t memTypeIndex = allocation->GetMemoryTypeIndex();
    if(size > 0 && IsMemoryTypeNonCoherent(memTypeIndex))
    {
        const VkDeviceSize nonCoherentAtomSize = m_PhysicalDeviceProperties.limits.nonCoherentAtomSize;
        const VkDeviceSize allocationSize = allocation->GetSize();
        OA_VMA_ASSERT(offset <= allocationSize);

        outRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        outRange.pNext = OA_VMA_NULL;
        outRange.memory = allocation->GetMemory();

        switch(allocation->GetType())
        {
        case OaVmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
            outRange.offset = OaVmaAlignDown(offset, nonCoherentAtomSize);
            if(size == VK_WHOLE_SIZE)
            {
                outRange.size = allocationSize - outRange.offset;
            }
            else
            {
                OA_VMA_ASSERT(offset + size <= allocationSize);
                outRange.size = OA_VMA_MIN(
                    OaVmaAlignUp(size + (offset - outRange.offset), nonCoherentAtomSize),
                    allocationSize - outRange.offset);
            }
            break;
        case OaVmaAllocation_T::ALLOCATION_TYPE_BLOCK:
        {
            // 1. Still within this allocation.
            outRange.offset = OaVmaAlignDown(offset, nonCoherentAtomSize);
            if(size == VK_WHOLE_SIZE)
            {
                size = allocationSize - offset;
            }
            else
            {
                OA_VMA_ASSERT(offset + size <= allocationSize);
            }
            outRange.size = OaVmaAlignUp(size + (offset - outRange.offset), nonCoherentAtomSize);

            // 2. Adjust to whole block.
            const VkDeviceSize allocationOffset = allocation->GetOffset();
            OA_VMA_ASSERT(allocationOffset % nonCoherentAtomSize == 0);
            const VkDeviceSize blockSize = allocation->GetBlock()->m_pMetadata->GetSize();
            outRange.offset += allocationOffset;
            outRange.size = OA_VMA_MIN(outRange.size, blockSize - outRange.offset);

            break;
        }
        default:
            OA_VMA_ASSERT(0);
        }
        return true;
    }
    return false;
}

#if OA_VMA_MEMORY_BUDGET
void OaVmaAllocator_T::UpdateVulkanBudget()
{
    OA_VMA_ASSERT(m_UseExtMemoryBudget);

    VkPhysicalDeviceMemoryProperties2KHR memProps = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR };

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budgetProps = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT };
    OaVmaPnextChainPushFront(&memProps, &budgetProps);

    GetVulkanFunctions().vkGetPhysicalDeviceMemoryProperties2KHR(m_PhysicalDevice, &memProps);

    {
        OaVmaMutexLockWrite lockWrite(m_Budget.m_BudgetMutex, m_UseMutex);

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
#endif // OA_VMA_MEMORY_BUDGET

void OaVmaAllocator_T::FillAllocation(OaVmaAllocation hAllocation, uint8_t pattern)
{
#if OA_VMA_DEBUG_INITIALIZE_ALLOCATIONS
    if(hAllocation->IsMappingAllowed() &&
        (m_MemProps.memoryTypes[hAllocation->GetMemoryTypeIndex()].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
    {
        void* pData = OA_VMA_NULL;
        VkResult res = Map(hAllocation, &pData);
        if(res == VK_SUCCESS)
        {
            memset(pData, (int)pattern, (size_t)hAllocation->GetSize());
            FlushOrInvalidateAllocation(hAllocation, 0, VK_WHOLE_SIZE, OA_VMA_CACHE_FLUSH);
            Unmap(hAllocation);
        }
        else
        {
            OA_VMA_ASSERT(0 && "OA_VMA_DEBUG_INITIALIZE_ALLOCATIONS is enabled, but couldn't map memory to fill allocation.");
        }
    }
#endif // #if OA_VMA_DEBUG_INITIALIZE_ALLOCATIONS
}

uint32_t OaVmaAllocator_T::GetGpuDefragmentationMemoryTypeBits()
{
    uint32_t memoryTypeBits = m_GpuDefragmentationMemoryTypeBits.load();
    if(memoryTypeBits == UINT32_MAX)
    {
        memoryTypeBits = CalculateGpuDefragmentationMemoryTypeBits();
        m_GpuDefragmentationMemoryTypeBits.store(memoryTypeBits);
    }
    return memoryTypeBits;
}

#if OA_VMA_STATS_STRING_ENABLED
void OaVmaAllocator_T::PrintDetailedMap(OaVmaJsonWriter& json)
{
    json.WriteString("DefaultPools");
    json.BeginObject();
    {
        for (uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
        {
            OaVmaBlockVector* pBlockVector = m_pBlockVectors[memTypeIndex];
            OaVmaDedicatedAllocationList& dedicatedAllocList = m_DedicatedAllocations[memTypeIndex];
            if (pBlockVector != OA_VMA_NULL)
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
        OaVmaMutexLockRead lock(m_PoolsMutex, m_UseMutex);
        if (!m_Pools.IsEmpty())
        {
            for (uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
            {
                bool displayType = true;
                size_t index = 0;
                for (OaVmaPool pool = m_Pools.Front(); pool != OA_VMA_NULL; pool = m_Pools.GetNext(pool))
                {
                    OaVmaBlockVector& blockVector = pool->m_BlockVector;
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
#endif // OA_VMA_STATS_STRING_ENABLED
#endif // _OA_VMA_ALLOCATOR_T_FUNCTIONS
