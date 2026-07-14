// SPDX: MIT. Derived from Vulkan Memory Allocator, (c) 2017-2026 Advanced Micro Devices, Inc.
// Full MIT copyright/permission notice: Source/Public/Oa/Runtime/OaVma.h. See NOTICE.md.
#ifndef _OA_VMA_PUBLIC_INTERFACE

#ifdef OAVK_HEADER_VERSION

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaImportVulkanFunctionsFromOaVk(
    const OaVmaAllocatorCreateInfo* OA_VMA_NOT_NULL pAllocatorCreateInfo,
    OaVmaVulkanFunctions* OA_VMA_NOT_NULL pDstVulkanFunctions)
{
    OA_VMA_ASSERT(pAllocatorCreateInfo != OA_VMA_NULL);
    OA_VMA_ASSERT(pAllocatorCreateInfo->instance != VK_NULL_HANDLE);
    OA_VMA_ASSERT(pAllocatorCreateInfo->device != VK_NULL_HANDLE);

    memset(pDstVulkanFunctions, 0, sizeof(*pDstVulkanFunctions));
    
    OaVkDeviceTable src = {};
    OaVkLoadDeviceTable(&src, pAllocatorCreateInfo->device);

#define COPY_GLOBAL_TO_OA_VMA_FUNC(volkName, OaVmaName) if(!pDstVulkanFunctions->OaVmaName) pDstVulkanFunctions->OaVmaName = volkName;
#define COPY_DEVICE_TO_OA_VMA_FUNC(volkName, OaVmaName) if(!pDstVulkanFunctions->OaVmaName) pDstVulkanFunctions->OaVmaName = src.volkName;

    COPY_GLOBAL_TO_OA_VMA_FUNC(vkGetInstanceProcAddr, vkGetInstanceProcAddr)
    COPY_GLOBAL_TO_OA_VMA_FUNC(vkGetDeviceProcAddr, vkGetDeviceProcAddr)
    COPY_GLOBAL_TO_OA_VMA_FUNC(vkGetPhysicalDeviceProperties, vkGetPhysicalDeviceProperties)
    COPY_GLOBAL_TO_OA_VMA_FUNC(vkGetPhysicalDeviceMemoryProperties, vkGetPhysicalDeviceMemoryProperties)
    COPY_DEVICE_TO_OA_VMA_FUNC(vkAllocateMemory, vkAllocateMemory)
    COPY_DEVICE_TO_OA_VMA_FUNC(vkFreeMemory, vkFreeMemory)
    COPY_DEVICE_TO_OA_VMA_FUNC(vkMapMemory, vkMapMemory)
    COPY_DEVICE_TO_OA_VMA_FUNC(vkUnmapMemory, vkUnmapMemory)
    COPY_DEVICE_TO_OA_VMA_FUNC(vkFlushMappedMemoryRanges, vkFlushMappedMemoryRanges)
    COPY_DEVICE_TO_OA_VMA_FUNC(vkInvalidateMappedMemoryRanges, vkInvalidateMappedMemoryRanges)
    COPY_DEVICE_TO_OA_VMA_FUNC(vkBindBufferMemory, vkBindBufferMemory)
    COPY_DEVICE_TO_OA_VMA_FUNC(vkBindImageMemory, vkBindImageMemory)
    COPY_DEVICE_TO_OA_VMA_FUNC(vkGetBufferMemoryRequirements, vkGetBufferMemoryRequirements)
    COPY_DEVICE_TO_OA_VMA_FUNC(vkGetImageMemoryRequirements, vkGetImageMemoryRequirements)
    COPY_DEVICE_TO_OA_VMA_FUNC(vkCreateBuffer, vkCreateBuffer)
    COPY_DEVICE_TO_OA_VMA_FUNC(vkDestroyBuffer, vkDestroyBuffer)
    COPY_DEVICE_TO_OA_VMA_FUNC(vkCreateImage, vkCreateImage)
    COPY_DEVICE_TO_OA_VMA_FUNC(vkDestroyImage, vkDestroyImage)
    COPY_DEVICE_TO_OA_VMA_FUNC(vkCmdCopyBuffer, vkCmdCopyBuffer)
#if OA_VMA_VULKAN_VERSION >= 1001000
    if (pAllocatorCreateInfo->vulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        COPY_GLOBAL_TO_OA_VMA_FUNC(vkGetPhysicalDeviceMemoryProperties2, vkGetPhysicalDeviceMemoryProperties2KHR)
        COPY_DEVICE_TO_OA_VMA_FUNC(vkGetBufferMemoryRequirements2, vkGetBufferMemoryRequirements2KHR)
        COPY_DEVICE_TO_OA_VMA_FUNC(vkGetImageMemoryRequirements2, vkGetImageMemoryRequirements2KHR)
        COPY_DEVICE_TO_OA_VMA_FUNC(vkBindBufferMemory2, vkBindBufferMemory2KHR)
        COPY_DEVICE_TO_OA_VMA_FUNC(vkBindImageMemory2, vkBindImageMemory2KHR)
    }
#endif
#if OA_VMA_VULKAN_VERSION >= 1003000
    if (pAllocatorCreateInfo->vulkanApiVersion >= VK_MAKE_VERSION(1, 3, 0))
    {
        COPY_DEVICE_TO_OA_VMA_FUNC(vkGetDeviceBufferMemoryRequirements, vkGetDeviceBufferMemoryRequirements)
        COPY_DEVICE_TO_OA_VMA_FUNC(vkGetDeviceImageMemoryRequirements, vkGetDeviceImageMemoryRequirements)
    }
#endif
#if OA_VMA_KHR_MAINTENANCE4
    if((pAllocatorCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT) != 0)
    {
        COPY_DEVICE_TO_OA_VMA_FUNC(vkGetDeviceBufferMemoryRequirementsKHR, vkGetDeviceBufferMemoryRequirements)
        COPY_DEVICE_TO_OA_VMA_FUNC(vkGetDeviceImageMemoryRequirementsKHR, vkGetDeviceImageMemoryRequirements)
    }
#endif
#if OA_VMA_DEDICATED_ALLOCATION
    if ((pAllocatorCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT) != 0)
    {
        COPY_DEVICE_TO_OA_VMA_FUNC(vkGetBufferMemoryRequirements2KHR, vkGetBufferMemoryRequirements2KHR)
        COPY_DEVICE_TO_OA_VMA_FUNC(vkGetImageMemoryRequirements2KHR, vkGetImageMemoryRequirements2KHR)
    }
#endif
#if OA_VMA_BIND_MEMORY2
    if ((pAllocatorCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT) != 0)
    {
        COPY_DEVICE_TO_OA_VMA_FUNC(vkBindBufferMemory2KHR, vkBindBufferMemory2KHR)
        COPY_DEVICE_TO_OA_VMA_FUNC(vkBindImageMemory2KHR, vkBindImageMemory2KHR)
    }
#endif
#if OA_VMA_MEMORY_BUDGET
    if ((pAllocatorCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT) != 0)
    {
        COPY_GLOBAL_TO_OA_VMA_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, vkGetPhysicalDeviceMemoryProperties2KHR)
    }
#endif
#if OA_VMA_EXTERNAL_MEMORY_WIN32
    if ((pAllocatorCreateInfo->flags & OA_VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT) != 0)
    {
        COPY_DEVICE_TO_OA_VMA_FUNC(vkGetMemoryWin32HandleKHR, vkGetMemoryWin32HandleKHR)
    }
#endif

#undef COPY_DEVICE_TO_OA_VMA_FUNC
#undef COPY_GLOBAL_TO_OA_VMA_FUNC

    return VK_SUCCESS;
}

#endif // #ifdef OAVK_HEADER_VERSION

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateAllocator(
    const OaVmaAllocatorCreateInfo* pCreateInfo,
    OaVmaAllocator* pAllocator)
{
    OA_VMA_ASSERT(pCreateInfo && pAllocator);
    OA_VMA_ASSERT(pCreateInfo->vulkanApiVersion == 0 ||
        (VK_VERSION_MAJOR(pCreateInfo->vulkanApiVersion) == 1 && VK_VERSION_MINOR(pCreateInfo->vulkanApiVersion) <= 4));
    OA_VMA_DEBUG_LOG("OaVmaCreateAllocator");
    *pAllocator = OaVma_new(pCreateInfo->pAllocationCallbacks, OaVmaAllocator_T)(pCreateInfo);
    VkResult result = (*pAllocator)->Init(pCreateInfo);
    if(result < 0)
    {
        OaVma_delete(pCreateInfo->pAllocationCallbacks, *pAllocator);
        *pAllocator = VK_NULL_HANDLE;
    }
    return result;
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaDestroyAllocator(
    OaVmaAllocator allocator)
{
    if(allocator != VK_NULL_HANDLE)
    {
        OA_VMA_DEBUG_LOG("OaVmaDestroyAllocator");
        VkAllocationCallbacks allocationCallbacks = allocator->m_AllocationCallbacks; // Have to copy the callbacks when destroying.
        OaVma_delete(&allocationCallbacks, allocator);
    }
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetAllocatorInfo(OaVmaAllocator allocator, OaVmaAllocatorInfo* pAllocatorInfo)
{
    OA_VMA_ASSERT(allocator && pAllocatorInfo);
    pAllocatorInfo->instance = allocator->m_hInstance;
    pAllocatorInfo->physicalDevice = allocator->GetPhysicalDevice();
    pAllocatorInfo->device = allocator->m_hDevice;
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetPhysicalDeviceProperties(
    OaVmaAllocator allocator,
    const VkPhysicalDeviceProperties **ppPhysicalDeviceProperties)
{
    OA_VMA_ASSERT(allocator && ppPhysicalDeviceProperties);
    *ppPhysicalDeviceProperties = &allocator->m_PhysicalDeviceProperties;
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetMemoryProperties(
    OaVmaAllocator allocator,
    const VkPhysicalDeviceMemoryProperties** ppPhysicalDeviceMemoryProperties)
{
    OA_VMA_ASSERT(allocator && ppPhysicalDeviceMemoryProperties);
    *ppPhysicalDeviceMemoryProperties = &allocator->m_MemProps;
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetMemoryTypeProperties(
    OaVmaAllocator allocator,
    uint32_t memoryTypeIndex,
    VkMemoryPropertyFlags* pFlags)
{
    OA_VMA_ASSERT(allocator && pFlags);
    OA_VMA_ASSERT(memoryTypeIndex < allocator->GetMemoryTypeCount());
    *pFlags = allocator->m_MemProps.memoryTypes[memoryTypeIndex].propertyFlags;
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaSetCurrentFrameIndex(
    OaVmaAllocator allocator,
    uint32_t frameIndex)
{
    OA_VMA_ASSERT(allocator);

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->SetCurrentFrameIndex(frameIndex);
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaCalculateStatistics(
    OaVmaAllocator allocator,
    OaVmaTotalStatistics* pStats)
{
    OA_VMA_ASSERT(allocator && pStats);
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK
    allocator->CalculateStatistics(pStats);
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetHeapBudgets(
    OaVmaAllocator allocator,
    OaVmaBudget* pBudgets)
{
    OA_VMA_ASSERT(allocator && pBudgets);
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK
    allocator->GetHeapBudgets(pBudgets, 0, allocator->GetMemoryHeapCount());
}

#if OA_VMA_STATS_STRING_ENABLED

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaBuildStatsString(
    OaVmaAllocator allocator,
    char** ppStatsString,
    VkBool32 detailedMap)
{
    OA_VMA_ASSERT(allocator && ppStatsString);
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    OaVmaStringBuilder sb(allocator->GetAllocationCallbacks());
    {
        OaVmaBudget budgets[VK_MAX_MEMORY_HEAPS];
        allocator->GetHeapBudgets(budgets, 0, allocator->GetMemoryHeapCount());

        OaVmaTotalStatistics stats;
        allocator->CalculateStatistics(&stats);

        OaVmaJsonWriter json(allocator->GetAllocationCallbacks(), sb);
        json.BeginObject();
        {
            json.WriteString("General");
            json.BeginObject();
            {
                const VkPhysicalDeviceProperties& deviceProperties = allocator->m_PhysicalDeviceProperties;
                const VkPhysicalDeviceMemoryProperties& memoryProperties = allocator->m_MemProps;

                json.WriteString("API");
                json.WriteString("Vulkan");

                json.WriteString("apiVersion");
                json.BeginString();
                json.ContinueString(VK_VERSION_MAJOR(deviceProperties.apiVersion));
                json.ContinueString(".");
                json.ContinueString(VK_VERSION_MINOR(deviceProperties.apiVersion));
                json.ContinueString(".");
                json.ContinueString(VK_VERSION_PATCH(deviceProperties.apiVersion));
                json.EndString();

                json.WriteString("GPU");
                json.WriteString(deviceProperties.deviceName);
                json.WriteString("deviceType");
                json.WriteNumber(static_cast<uint32_t>(deviceProperties.deviceType));

                json.WriteString("maxMemoryAllocationCount");
                json.WriteNumber(deviceProperties.limits.maxMemoryAllocationCount);
                json.WriteString("bufferImageGranularity");
                json.WriteNumber(deviceProperties.limits.bufferImageGranularity);
                json.WriteString("nonCoherentAtomSize");
                json.WriteNumber(deviceProperties.limits.nonCoherentAtomSize);

                json.WriteString("memoryHeapCount");
                json.WriteNumber(memoryProperties.memoryHeapCount);
                json.WriteString("memoryTypeCount");
                json.WriteNumber(memoryProperties.memoryTypeCount);
            }
            json.EndObject();
        }
        {
            json.WriteString("Total");
            OaVmaPrintDetailedStatistics(json, stats.total);
        }
        {
            json.WriteString("MemoryInfo");
            json.BeginObject();
            {
                for (uint32_t heapIndex = 0; heapIndex < allocator->GetMemoryHeapCount(); ++heapIndex)
                {
                    json.BeginString("Heap ");
                    json.ContinueString(heapIndex);
                    json.EndString();
                    json.BeginObject();
                    {
                        const VkMemoryHeap& heapInfo = allocator->m_MemProps.memoryHeaps[heapIndex];
                        json.WriteString("Flags");
                        json.BeginArray(true);
                        {
                            if (heapInfo.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                                json.WriteString("DEVICE_LOCAL");
                        #if OA_VMA_VULKAN_VERSION >= 1001000
                            if (heapInfo.flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT)
                                json.WriteString("MULTI_INSTANCE");
                        #endif

                            VkMemoryHeapFlags flags = heapInfo.flags &
                                ~(VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
                        #if OA_VMA_VULKAN_VERSION >= 1001000
                                    | VK_MEMORY_HEAP_MULTI_INSTANCE_BIT
                        #endif
                                    );
                            if (flags != 0)
                                json.WriteNumber(flags);
                        }
                        json.EndArray();

                        json.WriteString("Size");
                        json.WriteNumber(heapInfo.size);

                        json.WriteString("Budget");
                        json.BeginObject();
                        {
                            json.WriteString("BudgetBytes");
                            json.WriteNumber(budgets[heapIndex].budget);
                            json.WriteString("UsageBytes");
                            json.WriteNumber(budgets[heapIndex].usage);
                        }
                        json.EndObject();

                        json.WriteString("Stats");
                        OaVmaPrintDetailedStatistics(json, stats.memoryHeap[heapIndex]);

                        json.WriteString("MemoryPools");
                        json.BeginObject();
                        {
                            for (uint32_t typeIndex = 0; typeIndex < allocator->GetMemoryTypeCount(); ++typeIndex)
                            {
                                if (allocator->MemoryTypeIndexToHeapIndex(typeIndex) == heapIndex)
                                {
                                    json.BeginString("Type ");
                                    json.ContinueString(typeIndex);
                                    json.EndString();
                                    json.BeginObject();
                                    {
                                        json.WriteString("Flags");
                                        json.BeginArray(true);
                                        {
                                            VkMemoryPropertyFlags flags = allocator->m_MemProps.memoryTypes[typeIndex].propertyFlags;
                                            if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                                                json.WriteString("DEVICE_LOCAL");
                                            if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                                                json.WriteString("HOST_VISIBLE");
                                            if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                                json.WriteString("HOST_COHERENT");
                                            if (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
                                                json.WriteString("HOST_CACHED");
                                            if (flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)
                                                json.WriteString("LAZILY_ALLOCATED");
                                        #if OA_VMA_VULKAN_VERSION >= 1001000
                                            if (flags & VK_MEMORY_PROPERTY_PROTECTED_BIT)
                                                json.WriteString("PROTECTED");
                                        #endif
                                        #if VK_AMD_device_coherent_memory
                                            if (flags & VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY)
                                                json.WriteString("DEVICE_COHERENT_AMD");
                                            if (flags & VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY)
                                                json.WriteString("DEVICE_UNCACHED_AMD");
                                        #endif

                                            flags &= ~(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                        #if OA_VMA_VULKAN_VERSION >= 1001000
                                                | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT
                                        #endif
                                        #if VK_AMD_device_coherent_memory
                                                | VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY
                                                | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY
                                        #endif
                                                | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                                | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
                                            if (flags != 0)
                                                json.WriteNumber(flags);
                                        }
                                        json.EndArray();

                                        json.WriteString("Stats");
                                        OaVmaPrintDetailedStatistics(json, stats.memoryType[typeIndex]);
                                    }
                                    json.EndObject();
                                }
                            }

                        }
                        json.EndObject();
                    }
                    json.EndObject();
                }
            }
            json.EndObject();
        }

        if (detailedMap == VK_TRUE)
            allocator->PrintDetailedMap(json);

        json.EndObject();
    }

    *ppStatsString = OaVmaCreateStringCopy(allocator->GetAllocationCallbacks(), sb.GetData(), sb.GetLength());
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaFreeStatsString(
    OaVmaAllocator allocator,
    char* pStatsString)
{
    if(pStatsString != OA_VMA_NULL)
    {
        OA_VMA_ASSERT(allocator);
        OaVmaFreeString(allocator->GetAllocationCallbacks(), pStatsString);
    }
}

#endif // OA_VMA_STATS_STRING_ENABLED

/*
This function is not protected by any mutex because it just reads immutable data.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaFindMemoryTypeIndex(
    OaVmaAllocator allocator,
    uint32_t memoryTypeBits,
    const OaVmaAllocationCreateInfo* pAllocationCreateInfo,
    uint32_t* pMemoryTypeIndex)
{
    OA_VMA_ASSERT(allocator != VK_NULL_HANDLE);
    OA_VMA_ASSERT(pAllocationCreateInfo != OA_VMA_NULL);
    OA_VMA_ASSERT(pMemoryTypeIndex != OA_VMA_NULL);

    return allocator->FindMemoryTypeIndex(memoryTypeBits, pAllocationCreateInfo, OaVmaBufferImageUsage::UNKNOWN, pMemoryTypeIndex);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaFindMemoryTypeIndexForBufferInfo(
    OaVmaAllocator allocator,
    const VkBufferCreateInfo* pBufferCreateInfo,
    const OaVmaAllocationCreateInfo* pAllocationCreateInfo,
    uint32_t* pMemoryTypeIndex)
{
    OA_VMA_ASSERT(allocator != VK_NULL_HANDLE);
    OA_VMA_ASSERT(pBufferCreateInfo != OA_VMA_NULL);
    OA_VMA_ASSERT(pAllocationCreateInfo != OA_VMA_NULL);
    OA_VMA_ASSERT(pMemoryTypeIndex != OA_VMA_NULL);

    const VkDevice hDev = allocator->m_hDevice;
    const OaVmaVulkanFunctions* funcs = &allocator->GetVulkanFunctions();
    VkResult res = VK_SUCCESS;

#if OA_VMA_KHR_MAINTENANCE4 || OA_VMA_VULKAN_VERSION >= 1003000
    if (funcs->vkGetDeviceBufferMemoryRequirements &&
        (allocator->m_UseKhrMaintenance4 || allocator->m_VulkanApiVersion >= VK_MAKE_VERSION(1, 3, 0)))
    {
        // Can query straight from VkBufferCreateInfo :)
        VkDeviceBufferMemoryRequirementsKHR devBufMemReq = {VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS_KHR};
        devBufMemReq.pCreateInfo = pBufferCreateInfo;

        VkMemoryRequirements2 memReq = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        (*funcs->vkGetDeviceBufferMemoryRequirements)(hDev, &devBufMemReq, &memReq);

        return allocator->FindMemoryTypeIndex(
            memReq.memoryRequirements.memoryTypeBits, pAllocationCreateInfo,
            OaVmaBufferImageUsage(*pBufferCreateInfo, allocator->m_UseKhrMaintenance5), pMemoryTypeIndex);
    }
#endif // OA_VMA_KHR_MAINTENANCE4 || OA_VMA_VULKAN_VERSION >= 1003000

    // Must create a dummy buffer to query :(
    VkBuffer hBuffer = VK_NULL_HANDLE;
    res = funcs->vkCreateBuffer(
        hDev, pBufferCreateInfo, allocator->GetAllocationCallbacks(), &hBuffer);
    if(res == VK_SUCCESS)
    {
        VkMemoryRequirements memReq = {};
        funcs->vkGetBufferMemoryRequirements(hDev, hBuffer, &memReq);

        res = allocator->FindMemoryTypeIndex(
            memReq.memoryTypeBits, pAllocationCreateInfo,
            OaVmaBufferImageUsage(*pBufferCreateInfo, allocator->m_UseKhrMaintenance5), pMemoryTypeIndex);

        funcs->vkDestroyBuffer(
            hDev, hBuffer, allocator->GetAllocationCallbacks());
    }
    return res;
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaFindMemoryTypeIndexForImageInfo(
    OaVmaAllocator allocator,
    const VkImageCreateInfo* pImageCreateInfo,
    const OaVmaAllocationCreateInfo* pAllocationCreateInfo,
    uint32_t* pMemoryTypeIndex)
{
    OA_VMA_ASSERT(allocator != VK_NULL_HANDLE);
    OA_VMA_ASSERT(pImageCreateInfo != OA_VMA_NULL);
    OA_VMA_ASSERT(pAllocationCreateInfo != OA_VMA_NULL);
    OA_VMA_ASSERT(pMemoryTypeIndex != OA_VMA_NULL);

    const VkDevice hDev = allocator->m_hDevice;
    const OaVmaVulkanFunctions* funcs = &allocator->GetVulkanFunctions();
    VkResult res = VK_SUCCESS;

#if OA_VMA_KHR_MAINTENANCE4 || OA_VMA_VULKAN_VERSION >= 1003000
    if(funcs->vkGetDeviceImageMemoryRequirements &&
        (allocator->m_UseKhrMaintenance4 || allocator->m_VulkanApiVersion >= VK_MAKE_VERSION(1, 3, 0)))
    {
        // Can query straight from VkImageCreateInfo :)
        VkDeviceImageMemoryRequirementsKHR devImgMemReq = {VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS_KHR};
        devImgMemReq.pCreateInfo = pImageCreateInfo;
        OA_VMA_ASSERT(pImageCreateInfo->tiling != VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT_COPY && (pImageCreateInfo->flags & VK_IMAGE_CREATE_DISJOINT_BIT_COPY) == 0 &&
            "Cannot use this VkImageCreateInfo with OaVmaFindMemoryTypeIndexForImageInfo as I don't know what to pass as VkDeviceImageMemoryRequirements::planeAspect.");

        VkMemoryRequirements2 memReq = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        (*funcs->vkGetDeviceImageMemoryRequirements)(hDev, &devImgMemReq, &memReq);

        return allocator->FindMemoryTypeIndex(
            memReq.memoryRequirements.memoryTypeBits, pAllocationCreateInfo,
            OaVmaBufferImageUsage(*pImageCreateInfo), pMemoryTypeIndex);
    }
#endif // OA_VMA_KHR_MAINTENANCE4 || OA_VMA_VULKAN_VERSION >= 1003000
    
    // Must create a dummy image to query :(
    VkImage hImage = VK_NULL_HANDLE;
    res = funcs->vkCreateImage(
        hDev, pImageCreateInfo, allocator->GetAllocationCallbacks(), &hImage);
    if(res == VK_SUCCESS)
    {
        VkMemoryRequirements memReq = {};
        funcs->vkGetImageMemoryRequirements(hDev, hImage, &memReq);

        res = allocator->FindMemoryTypeIndex(
            memReq.memoryTypeBits, pAllocationCreateInfo,
            OaVmaBufferImageUsage(*pImageCreateInfo), pMemoryTypeIndex);

        funcs->vkDestroyImage(
            hDev, hImage, allocator->GetAllocationCallbacks());
    }
    return res;
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreatePool(
    OaVmaAllocator allocator,
    const OaVmaPoolCreateInfo* pCreateInfo,
    OaVmaPool* pPool)
{
    OA_VMA_ASSERT(allocator && pCreateInfo && pPool);

    OA_VMA_DEBUG_LOG("OaVmaCreatePool");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->CreatePool(pCreateInfo, pPool);
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaDestroyPool(
    OaVmaAllocator allocator,
    OaVmaPool pool)
{
    OA_VMA_ASSERT(allocator);

    if(pool == VK_NULL_HANDLE)
    {
        return;
    }

    OA_VMA_DEBUG_LOG("OaVmaDestroyPool");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->DestroyPool(pool);
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetPoolStatistics(
    OaVmaAllocator allocator,
    OaVmaPool pool,
    OaVmaStatistics* pPoolStats)
{
    OA_VMA_ASSERT(allocator && pool && pPoolStats);

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->GetPoolStatistics(pool, pPoolStats);
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaCalculatePoolStatistics(
    OaVmaAllocator allocator,
    OaVmaPool pool,
    OaVmaDetailedStatistics* pPoolStats)
{
    OA_VMA_ASSERT(allocator && pool && pPoolStats);

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->CalculatePoolStatistics(pool, pPoolStats);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCheckPoolCorruption(OaVmaAllocator allocator, OaVmaPool pool)
{
    OA_VMA_ASSERT(allocator && pool);

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    OA_VMA_DEBUG_LOG("OaVmaCheckPoolCorruption");

    return allocator->CheckPoolCorruption(pool);
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetPoolName(
    OaVmaAllocator allocator,
    OaVmaPool pool,
    const char** ppName)
{
    OA_VMA_ASSERT(allocator && pool && ppName);

    OA_VMA_DEBUG_LOG("OaVmaGetPoolName");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    *ppName = pool->GetName();
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaSetPoolName(
    OaVmaAllocator allocator,
    OaVmaPool pool,
    const char* pName)
{
    OA_VMA_ASSERT(allocator && pool);

    OA_VMA_DEBUG_LOG("OaVmaSetPoolName");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    pool->SetName(pName);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaAllocateMemory(
    OaVmaAllocator allocator,
    const VkMemoryRequirements* pVkMemoryRequirements,
    const OaVmaAllocationCreateInfo* pCreateInfo,
    OaVmaAllocation* pAllocation,
    OaVmaAllocationInfo* pAllocationInfo)
{
    OA_VMA_ASSERT(allocator && pVkMemoryRequirements && pCreateInfo && pAllocation);

    OA_VMA_DEBUG_LOG("OaVmaAllocateMemory");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VkResult result = allocator->AllocateMemory(
        *pVkMemoryRequirements,
        false, // requiresDedicatedAllocation
        false, // prefersDedicatedAllocation
        VK_NULL_HANDLE, // dedicatedBuffer
        VK_NULL_HANDLE, // dedicatedImage
        OaVmaBufferImageUsage::UNKNOWN, // dedicatedBufferImageUsage
        OA_VMA_NULL, // pMemoryAllocateNext
        *pCreateInfo,
        OA_VMA_SUBALLOCATION_TYPE_UNKNOWN,
        1, // allocationCount
        pAllocation);

    if(pAllocationInfo != OA_VMA_NULL && result == VK_SUCCESS)
    {
        allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
    }

    return result;
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaAllocateDedicatedMemory(
    OaVmaAllocator allocator,
    const VkMemoryRequirements* pVkMemoryRequirements,
    const OaVmaAllocationCreateInfo* pCreateInfo,
    void* pMemoryAllocateNext,
    OaVmaAllocation* pAllocation,
    OaVmaAllocationInfo* pAllocationInfo)
{
    OA_VMA_ASSERT(allocator && pVkMemoryRequirements && pCreateInfo && pAllocation);

    OA_VMA_DEBUG_LOG("OaVmaAllocateDedicatedMemory");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VkResult result = allocator->AllocateMemory(
        *pVkMemoryRequirements,
        true, // requiresDedicatedAllocation
        false, // prefersDedicatedAllocation
        VK_NULL_HANDLE, // dedicatedBuffer
        VK_NULL_HANDLE, // dedicatedImage
        OaVmaBufferImageUsage::UNKNOWN, // dedicatedBufferImageUsage
        pMemoryAllocateNext,
        *pCreateInfo,
        OA_VMA_SUBALLOCATION_TYPE_UNKNOWN,
        1, // allocationCount
        pAllocation);

    if(pAllocationInfo != OA_VMA_NULL && result == VK_SUCCESS)
    {
        allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
    }

    return result;
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaAllocateMemoryPages(
    OaVmaAllocator allocator,
    const VkMemoryRequirements* pVkMemoryRequirements,
    const OaVmaAllocationCreateInfo* pCreateInfo,
    size_t allocationCount,
    OaVmaAllocation* pAllocations,
    OaVmaAllocationInfo* pAllocationInfo)
{
    if(allocationCount == 0)
    {
        return VK_SUCCESS;
    }

    OA_VMA_ASSERT(allocator && pVkMemoryRequirements && pCreateInfo && pAllocations);

    OA_VMA_DEBUG_LOG("OaVmaAllocateMemoryPages");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VkResult result = allocator->AllocateMemory(
        *pVkMemoryRequirements,
        false, // requiresDedicatedAllocation
        false, // prefersDedicatedAllocation
        VK_NULL_HANDLE, // dedicatedBuffer
        VK_NULL_HANDLE, // dedicatedImage
        OaVmaBufferImageUsage::UNKNOWN, // dedicatedBufferImageUsage
        OA_VMA_NULL, // pMemoryAllocateNext
        *pCreateInfo,
        OA_VMA_SUBALLOCATION_TYPE_UNKNOWN,
        allocationCount,
        pAllocations);

    if(pAllocationInfo != OA_VMA_NULL && result == VK_SUCCESS)
    {
        for(size_t i = 0; i < allocationCount; ++i)
        {
            allocator->GetAllocationInfo(pAllocations[i], pAllocationInfo + i);
        }
    }

    return result;
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaAllocateMemoryForBuffer(
    OaVmaAllocator allocator,
    VkBuffer buffer,
    const OaVmaAllocationCreateInfo* pCreateInfo,
    OaVmaAllocation* pAllocation,
    OaVmaAllocationInfo* pAllocationInfo)
{
    OA_VMA_ASSERT(allocator && buffer != VK_NULL_HANDLE && pCreateInfo && pAllocation);

    OA_VMA_DEBUG_LOG("OaVmaAllocateMemoryForBuffer");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VkMemoryRequirements vkMemReq = {};
    bool requiresDedicatedAllocation = false;
    bool prefersDedicatedAllocation = false;
    allocator->GetBufferMemoryRequirements(buffer, vkMemReq,
        requiresDedicatedAllocation,
        prefersDedicatedAllocation);

    VkResult result = allocator->AllocateMemory(
        vkMemReq,
        requiresDedicatedAllocation,
        prefersDedicatedAllocation,
        buffer, // dedicatedBuffer
        VK_NULL_HANDLE, // dedicatedImage
        OaVmaBufferImageUsage::UNKNOWN, // dedicatedBufferImageUsage
        OA_VMA_NULL, // pMemoryAllocateNext
        *pCreateInfo,
        OA_VMA_SUBALLOCATION_TYPE_BUFFER,
        1, // allocationCount
        pAllocation);

    if(pAllocationInfo && result == VK_SUCCESS)
    {
        allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
    }

    return result;
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaAllocateMemoryForImage(
    OaVmaAllocator allocator,
    VkImage image,
    const OaVmaAllocationCreateInfo* pCreateInfo,
    OaVmaAllocation* pAllocation,
    OaVmaAllocationInfo* pAllocationInfo)
{
    OA_VMA_ASSERT(allocator && image != VK_NULL_HANDLE && pCreateInfo && pAllocation);

    OA_VMA_DEBUG_LOG("OaVmaAllocateMemoryForImage");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VkMemoryRequirements vkMemReq = {};
    bool requiresDedicatedAllocation = false;
    bool prefersDedicatedAllocation  = false;
    allocator->GetImageMemoryRequirements(image, vkMemReq,
        requiresDedicatedAllocation, prefersDedicatedAllocation);

    VkResult result = allocator->AllocateMemory(
        vkMemReq,
        requiresDedicatedAllocation,
        prefersDedicatedAllocation,
        VK_NULL_HANDLE, // dedicatedBuffer
        image, // dedicatedImage
        OaVmaBufferImageUsage::UNKNOWN, // dedicatedBufferImageUsage
        OA_VMA_NULL, // pMemoryAllocateNext
        *pCreateInfo,
        OA_VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN,
        1, // allocationCount
        pAllocation);

    if(pAllocationInfo && result == VK_SUCCESS)
    {
        allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
    }

    return result;
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaFreeMemory(
    OaVmaAllocator allocator,
    OaVmaAllocation allocation)
{
    OA_VMA_ASSERT(allocator);

    if(allocation == VK_NULL_HANDLE)
    {
        return;
    }

    OA_VMA_DEBUG_LOG("OaVmaFreeMemory");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->FreeMemory(
        1, // allocationCount
        &allocation);
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaFreeMemoryPages(
    OaVmaAllocator allocator,
    size_t allocationCount,
    const OaVmaAllocation* pAllocations)
{
    if(allocationCount == 0)
    {
        return;
    }

    OA_VMA_ASSERT(allocator);

    OA_VMA_DEBUG_LOG("OaVmaFreeMemoryPages");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->FreeMemory(allocationCount, pAllocations);
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetAllocationInfo(
    OaVmaAllocator allocator,
    OaVmaAllocation allocation,
    OaVmaAllocationInfo* pAllocationInfo)
{
    OA_VMA_ASSERT(allocator && allocation && pAllocationInfo);

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->GetAllocationInfo(allocation, pAllocationInfo);
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetAllocationInfo2(
    OaVmaAllocator allocator,
    OaVmaAllocation allocation,
    OaVmaAllocationInfo2* pAllocationInfo)
{
    OA_VMA_ASSERT(allocator && allocation && pAllocationInfo);

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->GetAllocationInfo2(allocation, pAllocationInfo);
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaSetAllocationUserData(
    OaVmaAllocator allocator,
    OaVmaAllocation allocation,
    void* pUserData)
{
    OA_VMA_ASSERT(allocator && allocation);

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocation->SetUserData(allocator, pUserData);
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaSetAllocationName(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    const char* OA_VMA_NULLABLE pName)
{
    allocation->SetName(allocator, pName);
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetAllocationMemoryProperties(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    VkMemoryPropertyFlags* OA_VMA_NOT_NULL pFlags)
{
    OA_VMA_ASSERT(allocator && allocation && pFlags);
    const uint32_t memTypeIndex = allocation->GetMemoryTypeIndex();
    *pFlags = allocator->m_MemProps.memoryTypes[memTypeIndex].propertyFlags;
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaMapMemory(
    OaVmaAllocator allocator,
    OaVmaAllocation allocation,
    void** ppData)
{
    OA_VMA_ASSERT(allocator && allocation && ppData);

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->Map(allocation, ppData);
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaUnmapMemory(
    OaVmaAllocator allocator,
    OaVmaAllocation allocation)
{
    OA_VMA_ASSERT(allocator && allocation);

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->Unmap(allocation);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaFlushAllocation(
    OaVmaAllocator allocator,
    OaVmaAllocation allocation,
    VkDeviceSize offset,
    VkDeviceSize size)
{
    OA_VMA_ASSERT(allocator && allocation);

    OA_VMA_DEBUG_LOG("OaVmaFlushAllocation");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->FlushOrInvalidateAllocation(allocation, offset, size, OA_VMA_CACHE_FLUSH);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaInvalidateAllocation(
    OaVmaAllocator allocator,
    OaVmaAllocation allocation,
    VkDeviceSize offset,
    VkDeviceSize size)
{
    OA_VMA_ASSERT(allocator && allocation);

    OA_VMA_DEBUG_LOG("OaVmaInvalidateAllocation");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->FlushOrInvalidateAllocation(allocation, offset, size, OA_VMA_CACHE_INVALIDATE);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaFlushAllocations(
    OaVmaAllocator allocator,
    uint32_t allocationCount,
    const OaVmaAllocation* allocations,
    const VkDeviceSize* offsets,
    const VkDeviceSize* sizes)
{
    OA_VMA_ASSERT(allocator);

    if(allocationCount == 0)
    {
        return VK_SUCCESS;
    }

    OA_VMA_ASSERT(allocations);

    OA_VMA_DEBUG_LOG("OaVmaFlushAllocations");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->FlushOrInvalidateAllocations(allocationCount, allocations, offsets, sizes, OA_VMA_CACHE_FLUSH);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaInvalidateAllocations(
    OaVmaAllocator allocator,
    uint32_t allocationCount,
    const OaVmaAllocation* allocations,
    const VkDeviceSize* offsets,
    const VkDeviceSize* sizes)
{
    OA_VMA_ASSERT(allocator);

    if(allocationCount == 0)
    {
        return VK_SUCCESS;
    }

    OA_VMA_ASSERT(allocations);

    OA_VMA_DEBUG_LOG("OaVmaInvalidateAllocations");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->FlushOrInvalidateAllocations(allocationCount, allocations, offsets, sizes, OA_VMA_CACHE_INVALIDATE);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCopyMemoryToAllocation(
    OaVmaAllocator allocator,
    const void* pSrcHostPointer,
    OaVmaAllocation dstAllocation,
    VkDeviceSize dstAllocationLocalOffset,
    VkDeviceSize size)
{
    OA_VMA_ASSERT(allocator && pSrcHostPointer && dstAllocation);

    if(size == 0)
    {
        return VK_SUCCESS;
    }

    OA_VMA_DEBUG_LOG("OaVmaCopyMemoryToAllocation");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->CopyMemoryToAllocation(pSrcHostPointer, dstAllocation, dstAllocationLocalOffset, size);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCopyAllocationToMemory(
    OaVmaAllocator allocator,
    OaVmaAllocation srcAllocation,
    VkDeviceSize srcAllocationLocalOffset,
    void* pDstHostPointer,
    VkDeviceSize size)
{
    OA_VMA_ASSERT(allocator && srcAllocation && pDstHostPointer);

    if(size == 0)
    {
        return VK_SUCCESS;
    }

    OA_VMA_DEBUG_LOG("OaVmaCopyAllocationToMemory");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->CopyAllocationToMemory(srcAllocation, srcAllocationLocalOffset, pDstHostPointer, size);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCheckCorruption(
    OaVmaAllocator allocator,
    uint32_t memoryTypeBits)
{
    OA_VMA_ASSERT(allocator);

    OA_VMA_DEBUG_LOG("OaVmaCheckCorruption");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->CheckCorruption(memoryTypeBits);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaBeginDefragmentation(
    OaVmaAllocator allocator,
    const OaVmaDefragmentationInfo* pInfo,
    OaVmaDefragmentationContext* pContext)
{
    OA_VMA_ASSERT(allocator && pInfo && pContext);

    OA_VMA_DEBUG_LOG("OaVmaBeginDefragmentation");

    if (pInfo->pool != OA_VMA_NULL)
    {
        // Check if run on supported algorithms
        if (pInfo->pool->m_BlockVector.GetAlgorithm() & OA_VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT)
            return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    *pContext = OaVma_new(allocator, OaVmaDefragmentationContext_T)(allocator, *pInfo);
    return VK_SUCCESS;
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaEndDefragmentation(
    OaVmaAllocator allocator,
    OaVmaDefragmentationContext context,
    OaVmaDefragmentationStats* pStats)
{
    OA_VMA_ASSERT(allocator && context);

    OA_VMA_DEBUG_LOG("OaVmaEndDefragmentation");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    if (pStats)
        context->GetStats(*pStats);
    OaVma_delete(allocator, context);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaBeginDefragmentationPass(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaDefragmentationContext OA_VMA_NOT_NULL context,
    OaVmaDefragmentationPassMoveInfo* OA_VMA_NOT_NULL pPassInfo)
{
    OA_VMA_ASSERT(context && pPassInfo);

    OA_VMA_DEBUG_LOG("OaVmaBeginDefragmentationPass");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return context->DefragmentPassBegin(*pPassInfo);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaEndDefragmentationPass(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaDefragmentationContext OA_VMA_NOT_NULL context,
    OaVmaDefragmentationPassMoveInfo* OA_VMA_NOT_NULL pPassInfo)
{
    OA_VMA_ASSERT(context && pPassInfo);

    OA_VMA_DEBUG_LOG("OaVmaEndDefragmentationPass");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return context->DefragmentPassEnd(*pPassInfo);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaBindBufferMemory(
    OaVmaAllocator allocator,
    OaVmaAllocation allocation,
    VkBuffer buffer)
{
    OA_VMA_ASSERT(allocator && allocation && buffer);

    OA_VMA_DEBUG_LOG("OaVmaBindBufferMemory");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->BindBufferMemory(allocation, 0, buffer, OA_VMA_NULL);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaBindBufferMemory2(
    OaVmaAllocator allocator,
    OaVmaAllocation allocation,
    VkDeviceSize allocationLocalOffset,
    VkBuffer buffer,
    const void* pNext)
{
    OA_VMA_ASSERT(allocator && allocation && buffer);

    OA_VMA_DEBUG_LOG("OaVmaBindBufferMemory2");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->BindBufferMemory(allocation, allocationLocalOffset, buffer, pNext);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaBindImageMemory(
    OaVmaAllocator allocator,
    OaVmaAllocation allocation,
    VkImage image)
{
    OA_VMA_ASSERT(allocator && allocation && image);

    OA_VMA_DEBUG_LOG("OaVmaBindImageMemory");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->BindImageMemory(allocation, 0, image, OA_VMA_NULL);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaBindImageMemory2(
    OaVmaAllocator allocator,
    OaVmaAllocation allocation,
    VkDeviceSize allocationLocalOffset,
    VkImage image,
    const void* pNext)
{
    OA_VMA_ASSERT(allocator && allocation && image);

    OA_VMA_DEBUG_LOG("OaVmaBindImageMemory2");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

        return allocator->BindImageMemory(allocation, allocationLocalOffset, image, pNext);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateBuffer(
    OaVmaAllocator allocator,
    const VkBufferCreateInfo* pBufferCreateInfo,
    const OaVmaAllocationCreateInfo* pAllocationCreateInfo,
    VkBuffer* pBuffer,
    OaVmaAllocation* pAllocation,
    OaVmaAllocationInfo* pAllocationInfo)
{
    OA_VMA_ASSERT(allocator && pBufferCreateInfo && pAllocationCreateInfo && pBuffer && pAllocation);
    OA_VMA_DEBUG_LOG("OaVmaCreateBuffer");
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;

    return allocator->CreateBuffer(pBufferCreateInfo, pAllocationCreateInfo,
        OA_VMA_NULL, // pMemoryAllocateNext
        pBuffer, pAllocation, pAllocationInfo);

}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateBufferWithAlignment(
    OaVmaAllocator allocator,
    const VkBufferCreateInfo* pBufferCreateInfo,
    const OaVmaAllocationCreateInfo* pAllocationCreateInfo,
    VkDeviceSize minAlignment,
    VkBuffer* pBuffer,
    OaVmaAllocation* pAllocation,
    OaVmaAllocationInfo* pAllocationInfo)
{
    OA_VMA_ASSERT(allocator && pBufferCreateInfo && pAllocationCreateInfo && OaVmaIsPow2(minAlignment) && pBuffer && pAllocation);
    OA_VMA_DEBUG_LOG("OaVmaCreateBufferWithAlignment");
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;

    OaVmaAllocationCreateInfo allocCreateInfoCopy = *pAllocationCreateInfo;
    allocCreateInfoCopy.minAlignment = OA_VMA_MAX(allocCreateInfoCopy.minAlignment, minAlignment);

    return allocator->CreateBuffer(pBufferCreateInfo, &allocCreateInfoCopy,
        OA_VMA_NULL, // pMemoryAllocateNext
        pBuffer, pAllocation, pAllocationInfo);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateDedicatedBuffer(
    OaVmaAllocator allocator,
    const VkBufferCreateInfo* pBufferCreateInfo,
    const OaVmaAllocationCreateInfo* pAllocationCreateInfo,
    void* pMemoryAllocateNext,
    VkBuffer* pBuffer,
    OaVmaAllocation* pAllocation,
    OaVmaAllocationInfo* pAllocationInfo)
{
    OA_VMA_ASSERT(allocator && pBufferCreateInfo && pAllocationCreateInfo && pBuffer && pAllocation);
    OA_VMA_DEBUG_LOG("OaVmaCreateDedicatedBuffer");
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;

    OaVmaAllocationCreateInfo allocCreateInfoCopy = *pAllocationCreateInfo;
    allocCreateInfoCopy.flags |= OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    return allocator->CreateBuffer(pBufferCreateInfo, &allocCreateInfoCopy,
        pMemoryAllocateNext, // pMemoryAllocateNext
        pBuffer, pAllocation, pAllocationInfo);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateAliasingBuffer(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    const VkBufferCreateInfo* OA_VMA_NOT_NULL pBufferCreateInfo,
    VkBuffer OA_VMA_NULLABLE_NON_DISPATCHABLE* OA_VMA_NOT_NULL pBuffer)
{
    return OaVmaCreateAliasingBuffer2(allocator, allocation, 0, pBufferCreateInfo, pBuffer);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateAliasingBuffer2(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    const VkBufferCreateInfo* OA_VMA_NOT_NULL pBufferCreateInfo,
    VkBuffer OA_VMA_NULLABLE_NON_DISPATCHABLE* OA_VMA_NOT_NULL pBuffer)
{
    OA_VMA_ASSERT(allocator && pBufferCreateInfo && pBuffer && allocation);
    OA_VMA_ASSERT(allocationLocalOffset + pBufferCreateInfo->size <= allocation->GetSize());

    OA_VMA_DEBUG_LOG("OaVmaCreateAliasingBuffer2");

    *pBuffer = VK_NULL_HANDLE;

    if (pBufferCreateInfo->size == 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if ((pBufferCreateInfo->usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_COPY) != 0 &&
        !allocator->m_UseKhrBufferDeviceAddress)
    {
        OA_VMA_ASSERT(0 && "Creating a buffer with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT is not valid if OA_VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT was not used.");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    // 1. Create VkBuffer.
    VkResult res = (*allocator->GetVulkanFunctions().vkCreateBuffer)(
        allocator->m_hDevice,
        pBufferCreateInfo,
        allocator->GetAllocationCallbacks(),
        pBuffer);
    if (res >= 0)
    {
        // 2. Bind buffer with memory.
        res = allocator->BindBufferMemory(allocation, allocationLocalOffset, *pBuffer, OA_VMA_NULL);
        if (res >= 0)
        {
            return VK_SUCCESS;
        }
        (*allocator->GetVulkanFunctions().vkDestroyBuffer)(allocator->m_hDevice, *pBuffer, allocator->GetAllocationCallbacks());
    }
    return res;
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaDestroyBuffer(
    OaVmaAllocator allocator,
    VkBuffer buffer,
    OaVmaAllocation allocation)
{
    OA_VMA_ASSERT(allocator);

    if(buffer == VK_NULL_HANDLE && allocation == VK_NULL_HANDLE)
    {
        return;
    }

    OA_VMA_DEBUG_LOG("OaVmaDestroyBuffer");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    if(buffer != VK_NULL_HANDLE)
    {
        (*allocator->GetVulkanFunctions().vkDestroyBuffer)(allocator->m_hDevice, buffer, allocator->GetAllocationCallbacks());
    }

    if(allocation != VK_NULL_HANDLE)
    {
        allocator->FreeMemory(
            1, // allocationCount
            &allocation);
    }
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateImage(
    OaVmaAllocator allocator,
    const VkImageCreateInfo* pImageCreateInfo,
    const OaVmaAllocationCreateInfo* pAllocationCreateInfo,
    VkImage* pImage,
    OaVmaAllocation* pAllocation,
    OaVmaAllocationInfo* pAllocationInfo)
{
    OA_VMA_ASSERT(allocator && pImageCreateInfo && pAllocationCreateInfo && pImage && pAllocation);
    OA_VMA_ASSERT((pImageCreateInfo->flags & VK_IMAGE_CREATE_DISJOINT_BIT_COPY) == 0 &&
        "OaVmaCreateImage() doesn't support disjoint multi-planar images. Please allocate memory for the planes using OaVmaAllocateMemory() and bind them using OaVmaBindImageMemory2().");
    OA_VMA_DEBUG_LOG("OaVmaCreateImage");
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;

    return allocator->CreateImage(pImageCreateInfo, pAllocationCreateInfo,
        OA_VMA_NULL, // pMemoryAllocateNext
        pImage, pAllocation, pAllocationInfo);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateDedicatedImage(
    OaVmaAllocator allocator,
    const VkImageCreateInfo* pImageCreateInfo,
    const OaVmaAllocationCreateInfo* pAllocationCreateInfo,
    void* pMemoryAllocateNext,
    VkImage* pImage,
    OaVmaAllocation* pAllocation,
    OaVmaAllocationInfo* pAllocationInfo)
{
    OA_VMA_ASSERT(allocator && pImageCreateInfo && pAllocationCreateInfo && pImage && pAllocation);
    OA_VMA_ASSERT((pImageCreateInfo->flags & VK_IMAGE_CREATE_DISJOINT_BIT_COPY) == 0 &&
        "OaVmaCreateDedicatedImage() doesn't support disjoint multi-planar images. Please allocate memory for the planes using OaVmaAllocateMemory() and bind them using OaVmaBindImageMemory2().");
    OA_VMA_DEBUG_LOG("OaVmaCreateDedicatedImage");
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;

    OaVmaAllocationCreateInfo allocCreateInfoCopy = *pAllocationCreateInfo;
    allocCreateInfoCopy.flags |= OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    return allocator->CreateImage(pImageCreateInfo, &allocCreateInfoCopy,
        pMemoryAllocateNext, // pMemoryAllocateNext
        pImage, pAllocation, pAllocationInfo);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateAliasingImage(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    const VkImageCreateInfo* OA_VMA_NOT_NULL pImageCreateInfo,
    VkImage OA_VMA_NULLABLE_NON_DISPATCHABLE* OA_VMA_NOT_NULL pImage)
{
    return OaVmaCreateAliasingImage2(allocator, allocation, 0, pImageCreateInfo, pImage);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateAliasingImage2(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    const VkImageCreateInfo* OA_VMA_NOT_NULL pImageCreateInfo,
    VkImage OA_VMA_NULLABLE_NON_DISPATCHABLE* OA_VMA_NOT_NULL pImage)
{
    OA_VMA_ASSERT(allocator && pImageCreateInfo && pImage && allocation);

    *pImage = VK_NULL_HANDLE;

    OA_VMA_DEBUG_LOG("OaVmaCreateImage2");

    if (pImageCreateInfo->extent.width == 0 ||
        pImageCreateInfo->extent.height == 0 ||
        pImageCreateInfo->extent.depth == 0 ||
        pImageCreateInfo->mipLevels == 0 ||
        pImageCreateInfo->arrayLayers == 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    // 1. Create VkImage.
    VkResult res = (*allocator->GetVulkanFunctions().vkCreateImage)(
        allocator->m_hDevice,
        pImageCreateInfo,
        allocator->GetAllocationCallbacks(),
        pImage);
    if (res >= 0)
    {
        // 2. Bind image with memory.
        res = allocator->BindImageMemory(allocation, allocationLocalOffset, *pImage, OA_VMA_NULL);
        if (res >= 0)
        {
            return VK_SUCCESS;
        }
        (*allocator->GetVulkanFunctions().vkDestroyImage)(allocator->m_hDevice, *pImage, allocator->GetAllocationCallbacks());
    }
    return res;
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaDestroyImage(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    VkImage OA_VMA_NULLABLE_NON_DISPATCHABLE image,
    OaVmaAllocation OA_VMA_NULLABLE allocation)
{
    OA_VMA_ASSERT(allocator);

    if(image == VK_NULL_HANDLE && allocation == VK_NULL_HANDLE)
    {
        return;
    }

    OA_VMA_DEBUG_LOG("OaVmaDestroyImage");

    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK

    if(image != VK_NULL_HANDLE)
    {
        (*allocator->GetVulkanFunctions().vkDestroyImage)(allocator->m_hDevice, image, allocator->GetAllocationCallbacks());
    }
    if(allocation != VK_NULL_HANDLE)
    {
        allocator->FreeMemory(
            1, // allocationCount
            &allocation);
    }
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateVirtualBlock(
    const OaVmaVirtualBlockCreateInfo* OA_VMA_NOT_NULL pCreateInfo,
    OaVmaVirtualBlock OA_VMA_NULLABLE * OA_VMA_NOT_NULL pVirtualBlock)
{
    OA_VMA_ASSERT(pCreateInfo && pVirtualBlock);
    OA_VMA_ASSERT(pCreateInfo->size > 0);
    OA_VMA_DEBUG_LOG("OaVmaCreateVirtualBlock");
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    *pVirtualBlock = OaVma_new(pCreateInfo->pAllocationCallbacks, OaVmaVirtualBlock_T)(*pCreateInfo);
    return VK_SUCCESS;

    /*
    Code for the future if we ever need a separate Init() method that could fail:

    VkResult res = (*pVirtualBlock)->Init();
    if(res < 0)
    {
        OaVma_delete(pCreateInfo->pAllocationCallbacks, *pVirtualBlock);
        *pVirtualBlock = VK_NULL_HANDLE;
    }
    return res;
    */
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaDestroyVirtualBlock(OaVmaVirtualBlock OA_VMA_NULLABLE virtualBlock)
{
    if(virtualBlock != VK_NULL_HANDLE)
    {
        OA_VMA_DEBUG_LOG("OaVmaDestroyVirtualBlock");
        OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;
        VkAllocationCallbacks allocationCallbacks = virtualBlock->m_AllocationCallbacks; // Have to copy the callbacks when destroying.
        OaVma_delete(&allocationCallbacks, virtualBlock);
    }
}

OA_VMA_CALL_PRE VkBool32 OA_VMA_CALL_POST OaVmaIsVirtualBlockEmpty(OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock)
{
    OA_VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
    OA_VMA_DEBUG_LOG("OaVmaIsVirtualBlockEmpty");
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    return virtualBlock->IsEmpty() ? VK_TRUE : VK_FALSE;
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetVirtualAllocationInfo(OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock,
    OaVmaVirtualAllocation OA_VMA_NOT_NULL_NON_DISPATCHABLE allocation, OaVmaVirtualAllocationInfo* OA_VMA_NOT_NULL pVirtualAllocInfo)
{
    OA_VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && pVirtualAllocInfo != OA_VMA_NULL);
    OA_VMA_DEBUG_LOG("OaVmaGetVirtualAllocationInfo");
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    virtualBlock->GetAllocationInfo(allocation, *pVirtualAllocInfo);
}

OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaVirtualAllocate(OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock,
    const OaVmaVirtualAllocationCreateInfo* OA_VMA_NOT_NULL pCreateInfo, OaVmaVirtualAllocation OA_VMA_NULLABLE_NON_DISPATCHABLE* OA_VMA_NOT_NULL pAllocation,
    VkDeviceSize* OA_VMA_NULLABLE pOffset)
{
    OA_VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && pCreateInfo != OA_VMA_NULL && pAllocation != OA_VMA_NULL);
    OA_VMA_DEBUG_LOG("OaVmaVirtualAllocate");
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    return virtualBlock->Allocate(*pCreateInfo, *pAllocation, pOffset);
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaVirtualFree(OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock, OaVmaVirtualAllocation OA_VMA_NULLABLE_NON_DISPATCHABLE allocation)
{
    if(allocation != VK_NULL_HANDLE)
    {
        OA_VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
        OA_VMA_DEBUG_LOG("OaVmaVirtualFree");
        OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;
        virtualBlock->Free(allocation);
    }
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaClearVirtualBlock(OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock)
{
    OA_VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
    OA_VMA_DEBUG_LOG("OaVmaClearVirtualBlock");
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    virtualBlock->Clear();
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaSetVirtualAllocationUserData(OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock,
    OaVmaVirtualAllocation OA_VMA_NOT_NULL_NON_DISPATCHABLE allocation, void* OA_VMA_NULLABLE pUserData)
{
    OA_VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
    OA_VMA_DEBUG_LOG("OaVmaSetVirtualAllocationUserData");
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    virtualBlock->SetAllocationUserData(allocation, pUserData);
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetVirtualBlockStatistics(OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock,
    OaVmaStatistics* OA_VMA_NOT_NULL pStats)
{
    OA_VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && pStats != OA_VMA_NULL);
    OA_VMA_DEBUG_LOG("OaVmaGetVirtualBlockStatistics");
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    virtualBlock->GetStatistics(*pStats);
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaCalculateVirtualBlockStatistics(OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock,
    OaVmaDetailedStatistics* OA_VMA_NOT_NULL pStats)
{
    OA_VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && pStats != OA_VMA_NULL);
    OA_VMA_DEBUG_LOG("OaVmaCalculateVirtualBlockStatistics");
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    virtualBlock->CalculateDetailedStatistics(*pStats);
}

#if OA_VMA_STATS_STRING_ENABLED

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaBuildVirtualBlockStatsString(OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock,
    char* OA_VMA_NULLABLE * OA_VMA_NOT_NULL ppStatsString, VkBool32 detailedMap)
{
    OA_VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && ppStatsString != OA_VMA_NULL);
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    const VkAllocationCallbacks* allocationCallbacks = virtualBlock->GetAllocationCallbacks();
    OaVmaStringBuilder sb(allocationCallbacks);
    virtualBlock->BuildStatsString(detailedMap != VK_FALSE, sb);
    *ppStatsString = OaVmaCreateStringCopy(allocationCallbacks, sb.GetData(), sb.GetLength());
}

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaFreeVirtualBlockStatsString(OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock,
    char* OA_VMA_NULLABLE pStatsString)
{
    if(pStatsString != OA_VMA_NULL)
    {
        OA_VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
        OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;
        OaVmaFreeString(virtualBlock->GetAllocationCallbacks(), pStatsString);
    }
}
#if OA_VMA_EXTERNAL_MEMORY_WIN32
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaGetMemoryWin32Handle(OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation, HANDLE hTargetProcess, HANDLE* OA_VMA_NOT_NULL pHandle)
{
    OA_VMA_ASSERT(allocator && allocation && pHandle);
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    return allocation->GetWin32Handle(allocator, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT, hTargetProcess, pHandle);
}
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaGetMemoryWin32Handle2(OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation, VkExternalMemoryHandleTypeFlagBits handleType, HANDLE hTargetProcess, HANDLE* OA_VMA_NOT_NULL pHandle)
{
    OA_VMA_ASSERT(allocator && allocation && pHandle);
    OA_VMA_ASSERT(handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR ||
        handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR ||
        handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT_KHR ||
        handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT_KHR ||
        handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT_KHR ||
        handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT_KHR);
    OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    return allocation->GetWin32Handle(allocator, handleType, hTargetProcess, pHandle);
}
#endif // OA_VMA_EXTERNAL_MEMORY_WIN32 
#endif // OA_VMA_STATS_STRING_ENABLED
#endif // _OA_VMA_PUBLIC_INTERFACE
