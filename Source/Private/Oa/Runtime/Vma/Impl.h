// SPDX: MIT. Derived from Vulkan Memory Allocator, (c) 2017-2026 Advanced Micro Devices, Inc.
// Full MIT copyright/permission notice: Source/Public/Oa/Runtime/OaVma.h. See NOTICE.md.
#ifndef _OA_VMA_MEMORY_FUNCTIONS
namespace
{
inline void* OaVmaMalloc(OaVmaAllocator hAllocator, size_t size, size_t alignment)
{
    return OaVmaMalloc(&hAllocator->m_AllocationCallbacks, size, alignment);
}

inline void OaVmaFree(OaVmaAllocator hAllocator, void* ptr)
{
    OaVmaFree(&hAllocator->m_AllocationCallbacks, ptr);
}

template<typename T>
T* OaVmaAllocate(OaVmaAllocator hAllocator)
{
    return (T*)OaVmaMalloc(hAllocator, sizeof(T), OA_VMA_ALIGN_OF(T));
}

template<typename T>
T* OaVmaAllocateArray(OaVmaAllocator hAllocator, size_t count)
{
    return (T*)OaVmaMalloc(hAllocator, sizeof(T) * count, OA_VMA_ALIGN_OF(T));
}

template<typename T>
void OaVma_delete(OaVmaAllocator hAllocator, T* ptr)
{
    if(ptr != OA_VMA_NULL)
    {
        ptr->~T();
        OaVmaFree(hAllocator, ptr);
    }
}

template<typename T>
void OaVma_delete_array(OaVmaAllocator hAllocator, T* ptr, size_t count)
{
    if(ptr != OA_VMA_NULL)
    {
        for(size_t i = count; i--; )
            ptr[i].~T();
        OaVmaFree(hAllocator, ptr);
    }
}
} // namespace
#endif // _OA_VMA_MEMORY_FUNCTIONS

#ifndef _OA_VMA_DEVICE_MEMORY_BLOCK_FUNCTIONS
OaVmaDeviceMemoryBlock::OaVmaDeviceMemoryBlock(OaVmaAllocator hAllocator)
    : m_pMetadata(OA_VMA_NULL),
    m_hParentPool(nullptr),
    m_MemoryTypeIndex(UINT32_MAX),
    m_Id(0),
    m_hMemory(VK_NULL_HANDLE),
    m_MapCount(0),
    m_pMappedData(OA_VMA_NULL){}

OaVmaDeviceMemoryBlock::~OaVmaDeviceMemoryBlock()
{
    OA_VMA_ASSERT_LEAK(m_MapCount == 0 && "VkDeviceMemory block is being destroyed while it is still mapped.");
    OA_VMA_ASSERT_LEAK(m_hMemory == VK_NULL_HANDLE);
}

void OaVmaDeviceMemoryBlock::Init(
    OaVmaAllocator hAllocator,
    OaVmaPool hParentPool,
    uint32_t newMemoryTypeIndex,
    VkDeviceMemory newMemory,
    VkDeviceSize newSize,
    uint32_t id,
    uint32_t algorithm,
    VkDeviceSize bufferImageGranularity)
{
    OA_VMA_ASSERT(m_hMemory == VK_NULL_HANDLE);

    m_hParentPool = hParentPool;
    m_MemoryTypeIndex = newMemoryTypeIndex;
    m_Id = id;
    m_hMemory = newMemory;

    switch (algorithm)
    {
    case 0:
        m_pMetadata = OaVma_new(hAllocator, OaVmaBlockMetadata_TLSF)(hAllocator->GetAllocationCallbacks(),
            bufferImageGranularity, false); // isVirtual
        break;
    case OA_VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT:
        m_pMetadata = OaVma_new(hAllocator, OaVmaBlockMetadata_Linear)(hAllocator->GetAllocationCallbacks(),
            bufferImageGranularity, false); // isVirtual
        break;
    default:
        OA_VMA_ASSERT(0);
        m_pMetadata = OaVma_new(hAllocator, OaVmaBlockMetadata_TLSF)(hAllocator->GetAllocationCallbacks(),
            bufferImageGranularity, false); // isVirtual
    }
    m_pMetadata->Init(newSize);
}

void OaVmaDeviceMemoryBlock::Destroy(OaVmaAllocator allocator)
{
    // Define macro OA_VMA_DEBUG_LOG_FORMAT or more specialized OA_VMA_LEAK_LOG_FORMAT
    // to receive the list of the unfreed allocations.
    if (!m_pMetadata->IsEmpty())
        m_pMetadata->DebugLogAllAllocations();
    // This is the most important assert in the entire library.
    // Hitting it means you have some memory leak - unreleased OaVmaAllocation objects.
    OA_VMA_ASSERT_LEAK(m_pMetadata->IsEmpty() && "Some allocations were not freed before destruction of this memory block!");

    OA_VMA_ASSERT_LEAK(m_hMemory != VK_NULL_HANDLE);
    allocator->FreeVulkanMemory(m_MemoryTypeIndex, m_pMetadata->GetSize(), m_hMemory);
    m_hMemory = VK_NULL_HANDLE;

    OaVma_delete(allocator, m_pMetadata);
    m_pMetadata = OA_VMA_NULL;
}

void OaVmaDeviceMemoryBlock::PostAlloc(OaVmaAllocator hAllocator)
{
    OaVmaMutexLock lock(m_MapAndBindMutex, hAllocator->m_UseMutex);
    m_MappingHysteresis.PostAlloc();
}

void OaVmaDeviceMemoryBlock::PostFree(OaVmaAllocator hAllocator)
{
    OaVmaMutexLock lock(m_MapAndBindMutex, hAllocator->m_UseMutex);
    if(m_MappingHysteresis.PostFree())
    {
        OA_VMA_ASSERT(m_MappingHysteresis.GetExtraMapping() == 0);
        if (m_MapCount == 0)
        {
            m_pMappedData = OA_VMA_NULL;
            (*hAllocator->GetVulkanFunctions().vkUnmapMemory)(hAllocator->m_hDevice, m_hMemory);
        }
    }
}

bool OaVmaDeviceMemoryBlock::Validate() const
{
    OA_VMA_VALIDATE((m_hMemory != VK_NULL_HANDLE) &&
        (m_pMetadata->GetSize() != 0));

    return m_pMetadata->Validate();
}

VkResult OaVmaDeviceMemoryBlock::CheckCorruption(OaVmaAllocator hAllocator)
{
    void* pData = OA_VMA_NULL;
    VkResult res = Map(hAllocator, 1, &pData);
    if (res != VK_SUCCESS)
    {
        return res;
    }

    res = m_pMetadata->CheckCorruption(pData);

    Unmap(hAllocator, 1);

    return res;
}

VkResult OaVmaDeviceMemoryBlock::Map(OaVmaAllocator hAllocator, uint32_t count, void** ppData)
{
    if (count == 0)
    {
        return VK_SUCCESS;
    }

    OaVmaMutexLock lock(m_MapAndBindMutex, hAllocator->m_UseMutex);
    const uint32_t oldTotalMapCount = m_MapCount + m_MappingHysteresis.GetExtraMapping();
    if (oldTotalMapCount != 0)
    {
        OA_VMA_ASSERT(m_pMappedData != OA_VMA_NULL);
        m_MappingHysteresis.PostMap();
        m_MapCount += count;
        if (ppData != OA_VMA_NULL)
        {
            *ppData = m_pMappedData;
        }
        return VK_SUCCESS;
    }

    VkResult result = (*hAllocator->GetVulkanFunctions().vkMapMemory)(
        hAllocator->m_hDevice,
        m_hMemory,
        0, // offset
        VK_WHOLE_SIZE,
        0, // flags
        &m_pMappedData);
    if (result == VK_SUCCESS)
    {
        OA_VMA_ASSERT(m_pMappedData != OA_VMA_NULL);
        m_MappingHysteresis.PostMap();
        m_MapCount = count;
        if (ppData != OA_VMA_NULL)
        {
            *ppData = m_pMappedData;
        }
    }
    return result;
}

void OaVmaDeviceMemoryBlock::Unmap(OaVmaAllocator hAllocator, uint32_t count)
{
    if (count == 0)
    {
        return;
    }

    OaVmaMutexLock lock(m_MapAndBindMutex, hAllocator->m_UseMutex);
    if (m_MapCount >= count)
    {
        m_MapCount -= count;
        const uint32_t totalMapCount = m_MapCount + m_MappingHysteresis.GetExtraMapping();
        if (totalMapCount == 0)
        {
            m_pMappedData = OA_VMA_NULL;
            (*hAllocator->GetVulkanFunctions().vkUnmapMemory)(hAllocator->m_hDevice, m_hMemory);
        }
        m_MappingHysteresis.PostUnmap();
    }
    else
    {
        OA_VMA_ASSERT(0 && "VkDeviceMemory block is being unmapped while it was not previously mapped.");
    }
}

VkResult OaVmaDeviceMemoryBlock::WriteMagicValueAfterAllocation(OaVmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize)
{
    OA_VMA_ASSERT(OA_VMA_DEBUG_MARGIN > 0 && OA_VMA_DEBUG_MARGIN % 4 == 0 && OA_VMA_DEBUG_DETECT_CORRUPTION);

    void* pData = OA_VMA_NULL;
    VkResult res = Map(hAllocator, 1, &pData);
    if (res != VK_SUCCESS)
    {
        return res;
    }

    OaVmaWriteMagicValue(pData, allocOffset + allocSize);

    Unmap(hAllocator, 1);
    return VK_SUCCESS;
}

VkResult OaVmaDeviceMemoryBlock::ValidateMagicValueAfterAllocation(OaVmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize)
{
    OA_VMA_ASSERT(OA_VMA_DEBUG_MARGIN > 0 && OA_VMA_DEBUG_MARGIN % 4 == 0 && OA_VMA_DEBUG_DETECT_CORRUPTION);

    void* pData = OA_VMA_NULL;
    VkResult res = Map(hAllocator, 1, &pData);
    if (res != VK_SUCCESS)
    {
        return res;
    }

    if (!OaVmaValidateMagicValue(pData, allocOffset + allocSize))
    {
        OA_VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER FREED ALLOCATION!");
    }

    Unmap(hAllocator, 1);
    return VK_SUCCESS;
}

VkResult OaVmaDeviceMemoryBlock::BindBufferMemory(
    OaVmaAllocator hAllocator,
    OaVmaAllocation hAllocation,
    VkDeviceSize allocationLocalOffset,
    VkBuffer hBuffer,
    const void* pNext)
{
    OA_VMA_ASSERT(hAllocation->GetType() == OaVmaAllocation_T::ALLOCATION_TYPE_BLOCK &&
        hAllocation->GetBlock() == this);
    OA_VMA_ASSERT(allocationLocalOffset < hAllocation->GetSize() &&
        "Invalid allocationLocalOffset. Did you forget that this offset is relative to the beginning of the allocation, not the whole memory block?");
    const VkDeviceSize memoryOffset = hAllocation->GetOffset() + allocationLocalOffset;
    // This lock is important so that we don't call vkBind... and/or vkMap... simultaneously on the same VkDeviceMemory from multiple threads.
    OaVmaMutexLock lock(m_MapAndBindMutex, hAllocator->m_UseMutex);
    return hAllocator->BindVulkanBuffer(m_hMemory, memoryOffset, hBuffer, pNext);
}

VkResult OaVmaDeviceMemoryBlock::BindImageMemory(
    OaVmaAllocator hAllocator,
    OaVmaAllocation hAllocation,
    VkDeviceSize allocationLocalOffset,
    VkImage hImage,
    const void* pNext)
{
    OA_VMA_ASSERT(hAllocation->GetType() == OaVmaAllocation_T::ALLOCATION_TYPE_BLOCK &&
        hAllocation->GetBlock() == this);
    OA_VMA_ASSERT(allocationLocalOffset < hAllocation->GetSize() &&
        "Invalid allocationLocalOffset. Did you forget that this offset is relative to the beginning of the allocation, not the whole memory block?");
    const VkDeviceSize memoryOffset = hAllocation->GetOffset() + allocationLocalOffset;
    // This lock is important so that we don't call vkBind... and/or vkMap... simultaneously on the same VkDeviceMemory from multiple threads.
    OaVmaMutexLock lock(m_MapAndBindMutex, hAllocator->m_UseMutex);
    return hAllocator->BindVulkanImage(m_hMemory, memoryOffset, hImage, pNext);
}

#if OA_VMA_EXTERNAL_MEMORY_WIN32
VkResult OaVmaDeviceMemoryBlock::CreateWin32Handle(const OaVmaAllocator hAllocator, PFN_vkGetMemoryWin32HandleKHR pvkGetMemoryWin32HandleKHR, VkExternalMemoryHandleTypeFlagBits handleType, HANDLE hTargetProcess, HANDLE* pHandle) noexcept
{
    OA_VMA_ASSERT(pHandle);
    return m_Handle.GetHandle(hAllocator->m_hDevice, m_hMemory, pvkGetMemoryWin32HandleKHR, handleType, hTargetProcess, hAllocator->m_UseMutex, pHandle);
}
#endif // OA_VMA_EXTERNAL_MEMORY_WIN32
#endif // _OA_VMA_DEVICE_MEMORY_BLOCK_FUNCTIONS

#ifndef _OA_VMA_ALLOCATION_T_FUNCTIONS
OaVmaAllocation_T::OaVmaAllocation_T(bool mappingAllowed)
    : m_Alignment{ 1 },
    m_Size{ 0 },
    m_pUserData{ OA_VMA_NULL },
    m_pName{ OA_VMA_NULL },
    m_MemoryTypeIndex{ 0 },
    m_Type{ (uint8_t)ALLOCATION_TYPE_NONE },
    m_SuballocationType{ (uint8_t)OA_VMA_SUBALLOCATION_TYPE_UNKNOWN },
    m_MapCount{ 0 },
    m_Flags{ 0 }
{
    if(mappingAllowed)
        m_Flags |= (uint8_t)FLAG_MAPPING_ALLOWED;
}

OaVmaAllocation_T::~OaVmaAllocation_T()
{
    OA_VMA_ASSERT_LEAK(m_MapCount == 0 && "Allocation was not unmapped before destruction.");

    // Check if owned string was freed.
    OA_VMA_ASSERT(m_pName == OA_VMA_NULL);
}

void OaVmaAllocation_T::InitBlockAllocation(
    OaVmaDeviceMemoryBlock* block,
    OaVmaAllocHandle allocHandle,
    VkDeviceSize alignment,
    VkDeviceSize size,
    uint32_t memoryTypeIndex,
    OaVmaSuballocationType suballocationType,
    bool mapped)
{
    OA_VMA_ASSERT(m_Type == ALLOCATION_TYPE_NONE);
    OA_VMA_ASSERT(block != OA_VMA_NULL);
    m_Type = (uint8_t)ALLOCATION_TYPE_BLOCK;
    m_Alignment = alignment;
    m_Size = size;
    m_MemoryTypeIndex = memoryTypeIndex;
    if(mapped)
    {
        OA_VMA_ASSERT(IsMappingAllowed() && "Mapping is not allowed on this allocation! Please use one of the new OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_* flags when creating it.");
        m_Flags |= (uint8_t)FLAG_PERSISTENT_MAP;
    }
    m_SuballocationType = (uint8_t)suballocationType;
    m_BlockAllocation.m_Block = block;
    m_BlockAllocation.m_AllocHandle = allocHandle;
}

void OaVmaAllocation_T::InitDedicatedAllocation(
    OaVmaAllocator allocator,
    OaVmaPool hParentPool,
    uint32_t memoryTypeIndex,
    VkDeviceMemory hMemory,
    OaVmaSuballocationType suballocationType,
    void* pMappedData,
    VkDeviceSize size)
{
    OA_VMA_ASSERT(m_Type == ALLOCATION_TYPE_NONE);
    OA_VMA_ASSERT(hMemory != VK_NULL_HANDLE);
    m_Type = (uint8_t)ALLOCATION_TYPE_DEDICATED;
    m_Alignment = 0;
    m_Size = size;
    m_MemoryTypeIndex = memoryTypeIndex;
    m_SuballocationType = (uint8_t)suballocationType;
    m_DedicatedAllocation.m_ExtraData = OA_VMA_NULL;
    m_DedicatedAllocation.m_hParentPool = hParentPool;
    m_DedicatedAllocation.m_hMemory = hMemory;
    m_DedicatedAllocation.m_Prev = OA_VMA_NULL;
    m_DedicatedAllocation.m_Next = OA_VMA_NULL;

    if (pMappedData != OA_VMA_NULL)
    {
        OA_VMA_ASSERT(IsMappingAllowed() && "Mapping is not allowed on this allocation! Please use one of the new OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_* flags when creating it.");
        m_Flags |= (uint8_t)FLAG_PERSISTENT_MAP;
        EnsureExtraData(allocator);
        m_DedicatedAllocation.m_ExtraData->m_pMappedData = pMappedData;
    }
}

void OaVmaAllocation_T::Destroy(OaVmaAllocator allocator)
{
    FreeName(allocator);

    if (GetType() == ALLOCATION_TYPE_DEDICATED)
    {
        OaVma_delete(allocator, m_DedicatedAllocation.m_ExtraData);
    }
}

void OaVmaAllocation_T::SetName(OaVmaAllocator hAllocator, const char* pName)
{
    OA_VMA_ASSERT(pName == OA_VMA_NULL || pName != m_pName);

    FreeName(hAllocator);

    if (pName != OA_VMA_NULL)
        m_pName = OaVmaCreateStringCopy(hAllocator->GetAllocationCallbacks(), pName);
}

uint8_t OaVmaAllocation_T::SwapBlockAllocation(OaVmaAllocator hAllocator, OaVmaAllocation allocation)
{
    OA_VMA_ASSERT(allocation != OA_VMA_NULL);
    OA_VMA_ASSERT(m_Type == ALLOCATION_TYPE_BLOCK);
    OA_VMA_ASSERT(allocation->m_Type == ALLOCATION_TYPE_BLOCK);

    if (m_MapCount != 0)
        m_BlockAllocation.m_Block->Unmap(hAllocator, m_MapCount);

    m_BlockAllocation.m_Block->m_pMetadata->SetAllocationUserData(m_BlockAllocation.m_AllocHandle, allocation);
    std::swap(m_BlockAllocation, allocation->m_BlockAllocation);
    m_BlockAllocation.m_Block->m_pMetadata->SetAllocationUserData(m_BlockAllocation.m_AllocHandle, this);

#if OA_VMA_STATS_STRING_ENABLED
    std::swap(m_BufferImageUsage, allocation->m_BufferImageUsage);
#endif
    return m_MapCount;
}

OaVmaAllocHandle OaVmaAllocation_T::GetAllocHandle() const
{
    switch (m_Type)
    {
    case ALLOCATION_TYPE_BLOCK:
        return m_BlockAllocation.m_AllocHandle;
    case ALLOCATION_TYPE_DEDICATED:
        return VK_NULL_HANDLE;
    default:
        OA_VMA_ASSERT(0);
        return VK_NULL_HANDLE;
    }
}

VkDeviceSize OaVmaAllocation_T::GetOffset() const
{
    switch (m_Type)
    {
    case ALLOCATION_TYPE_BLOCK:
        return m_BlockAllocation.m_Block->m_pMetadata->GetAllocationOffset(m_BlockAllocation.m_AllocHandle);
    case ALLOCATION_TYPE_DEDICATED:
        return 0;
    default:
        OA_VMA_ASSERT(0);
        return 0;
    }
}

OaVmaPool OaVmaAllocation_T::GetParentPool() const
{
    switch (m_Type)
    {
    case ALLOCATION_TYPE_BLOCK:
        return m_BlockAllocation.m_Block->GetParentPool();
    case ALLOCATION_TYPE_DEDICATED:
        return m_DedicatedAllocation.m_hParentPool;
    default:
        OA_VMA_ASSERT(0);
        return VK_NULL_HANDLE;
    }
}

VkDeviceMemory OaVmaAllocation_T::GetMemory() const
{
    switch (m_Type)
    {
    case ALLOCATION_TYPE_BLOCK:
        return m_BlockAllocation.m_Block->GetDeviceMemory();
    case ALLOCATION_TYPE_DEDICATED:
        return m_DedicatedAllocation.m_hMemory;
    default:
        OA_VMA_ASSERT(0);
        return VK_NULL_HANDLE;
    }
}

void* OaVmaAllocation_T::GetMappedData() const
{
    switch (m_Type)
    {
    case ALLOCATION_TYPE_BLOCK:
        if (m_MapCount != 0 || IsPersistentMap())
        {
            void* pBlockData = m_BlockAllocation.m_Block->GetMappedData();
            OA_VMA_ASSERT(pBlockData != OA_VMA_NULL);
            return (char*)pBlockData + GetOffset();
        }
        else
        {
            return OA_VMA_NULL;
        }
        break;
    case ALLOCATION_TYPE_DEDICATED:
        OA_VMA_ASSERT((m_DedicatedAllocation.m_ExtraData != OA_VMA_NULL && m_DedicatedAllocation.m_ExtraData->m_pMappedData != OA_VMA_NULL) ==
            (m_MapCount != 0 || IsPersistentMap()));
        return m_DedicatedAllocation.m_ExtraData != OA_VMA_NULL ? m_DedicatedAllocation.m_ExtraData->m_pMappedData : OA_VMA_NULL;
    default:
        OA_VMA_ASSERT(0);
        return OA_VMA_NULL;
    }
}

void OaVmaAllocation_T::BlockAllocMap()
{
    OA_VMA_ASSERT(GetType() == ALLOCATION_TYPE_BLOCK);
    OA_VMA_ASSERT(IsMappingAllowed() && "Mapping is not allowed on this allocation! Please use one of the new OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_* flags when creating it.");

    if (m_MapCount < 0xFF)
    {
        ++m_MapCount;
    }
    else
    {
        OA_VMA_ASSERT(0 && "Allocation mapped too many times simultaneously.");
    }
}

void OaVmaAllocation_T::BlockAllocUnmap()
{
    OA_VMA_ASSERT(GetType() == ALLOCATION_TYPE_BLOCK);

    if (m_MapCount > 0)
    {
        --m_MapCount;
    }
    else
    {
        OA_VMA_ASSERT(0 && "Unmapping allocation not previously mapped.");
    }
}

VkResult OaVmaAllocation_T::DedicatedAllocMap(OaVmaAllocator hAllocator, void** ppData)
{
    OA_VMA_ASSERT(GetType() == ALLOCATION_TYPE_DEDICATED);
    OA_VMA_ASSERT(IsMappingAllowed() && "Mapping is not allowed on this allocation! Please use one of the new OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_* flags when creating it.");

    EnsureExtraData(hAllocator);

    if (m_MapCount != 0 || IsPersistentMap())
    {
        if (m_MapCount < 0xFF)
        {
            OA_VMA_ASSERT(m_DedicatedAllocation.m_ExtraData->m_pMappedData != OA_VMA_NULL);
            *ppData = m_DedicatedAllocation.m_ExtraData->m_pMappedData;
            ++m_MapCount;
            return VK_SUCCESS;
        }

        OA_VMA_ASSERT(0 && "Dedicated allocation mapped too many times simultaneously.");
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    VkResult result = (*hAllocator->GetVulkanFunctions().vkMapMemory)(
        hAllocator->m_hDevice,
        m_DedicatedAllocation.m_hMemory,
        0, // offset
        VK_WHOLE_SIZE,
        0, // flags
        ppData);
    if (result == VK_SUCCESS)
    {
        m_DedicatedAllocation.m_ExtraData->m_pMappedData = *ppData;
        m_MapCount = 1;
    }
    return result;
}

void OaVmaAllocation_T::DedicatedAllocUnmap(OaVmaAllocator hAllocator)
{
    OA_VMA_ASSERT(GetType() == ALLOCATION_TYPE_DEDICATED);

    if (m_MapCount > 0)
    {
        --m_MapCount;
        if (m_MapCount == 0 && !IsPersistentMap())
        {
            OA_VMA_ASSERT(m_DedicatedAllocation.m_ExtraData != OA_VMA_NULL);
            m_DedicatedAllocation.m_ExtraData->m_pMappedData = OA_VMA_NULL;
            (*hAllocator->GetVulkanFunctions().vkUnmapMemory)(
                hAllocator->m_hDevice,
                m_DedicatedAllocation.m_hMemory);
        }
    }
    else
    {
        OA_VMA_ASSERT(0 && "Unmapping dedicated allocation not previously mapped.");
    }
}

#if OA_VMA_STATS_STRING_ENABLED
void OaVmaAllocation_T::PrintParameters(class OaVmaJsonWriter& json) const
{
    json.WriteString("Type");
    json.WriteString(OA_VMA_SUBALLOCATION_TYPE_NAMES[m_SuballocationType]);

    json.WriteString("Size");
    json.WriteNumber(m_Size);
    json.WriteString("Usage");
    json.WriteNumber(m_BufferImageUsage.Value); // It may be uint32_t or uint64_t.

    if (m_pUserData != OA_VMA_NULL)
    {
        json.WriteString("CustomData");
        json.BeginString();
        json.ContinueString_Pointer(m_pUserData);
        json.EndString();
    }
    if (m_pName != OA_VMA_NULL)
    {
        json.WriteString("Name");
        json.WriteString(m_pName);
    }
}
#if OA_VMA_EXTERNAL_MEMORY_WIN32
VkResult OaVmaAllocation_T::GetWin32Handle(OaVmaAllocator hAllocator, VkExternalMemoryHandleTypeFlagBits handleType, HANDLE hTargetProcess, HANDLE* pHandle) noexcept
{
    auto pvkGetMemoryWin32HandleKHR = hAllocator->GetVulkanFunctions().vkGetMemoryWin32HandleKHR;
    switch (m_Type)
    {
    case ALLOCATION_TYPE_BLOCK:
        return m_BlockAllocation.m_Block->CreateWin32Handle(hAllocator, pvkGetMemoryWin32HandleKHR, handleType, hTargetProcess, pHandle);
    case ALLOCATION_TYPE_DEDICATED:
        EnsureExtraData(hAllocator);
        return m_DedicatedAllocation.m_ExtraData->m_Handle.GetHandle(hAllocator->m_hDevice, m_DedicatedAllocation.m_hMemory, pvkGetMemoryWin32HandleKHR, handleType, hTargetProcess, hAllocator->m_UseMutex, pHandle);
    default:
        OA_VMA_ASSERT(0);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
}
#endif // OA_VMA_EXTERNAL_MEMORY_WIN32
#endif // OA_VMA_STATS_STRING_ENABLED

void OaVmaAllocation_T::EnsureExtraData(OaVmaAllocator hAllocator)
{
    if (m_DedicatedAllocation.m_ExtraData == OA_VMA_NULL)
    {
        m_DedicatedAllocation.m_ExtraData = OaVma_new(hAllocator, OaVmaAllocationExtraData)();
    }
}

void OaVmaAllocation_T::FreeName(OaVmaAllocator hAllocator)
{
    if(m_pName)
    {
        OaVmaFreeString(hAllocator->GetAllocationCallbacks(), m_pName);
        m_pName = OA_VMA_NULL;
    }
}
#endif // _OA_VMA_ALLOCATION_T_FUNCTIONS

#ifndef _OA_VMA_BLOCK_VECTOR_FUNCTIONS
OaVmaBlockVector::OaVmaBlockVector(
    OaVmaAllocator hAllocator,
    OaVmaPool hParentPool,
    uint32_t memoryTypeIndex,
    VkDeviceSize preferredBlockSize,
    size_t minBlockCount,
    size_t maxBlockCount,
    VkDeviceSize bufferImageGranularity,
    bool explicitBlockSize,
    uint32_t algorithm,
    float priority,
    VkDeviceSize minAllocationAlignment,
    void* pMemoryAllocateNext)
    : m_hAllocator(hAllocator),
    m_hParentPool(hParentPool),
    m_MemoryTypeIndex(memoryTypeIndex),
    m_PreferredBlockSize(preferredBlockSize),
    m_MinBlockCount(minBlockCount),
    m_MaxBlockCount(maxBlockCount),
    m_BufferImageGranularity(bufferImageGranularity),
    m_ExplicitBlockSize(explicitBlockSize),
    m_Algorithm(algorithm),
    m_Priority(priority),
    m_MinAllocationAlignment(minAllocationAlignment),
    m_pMemoryAllocateNext(pMemoryAllocateNext),
    m_Blocks(OaVmaStlAllocator<OaVmaDeviceMemoryBlock*>(hAllocator->GetAllocationCallbacks())),
    m_NextBlockId(0) {}

OaVmaBlockVector::~OaVmaBlockVector()
{
    for (size_t i = m_Blocks.size(); i--; )
    {
        m_Blocks[i]->Destroy(m_hAllocator);
        OaVma_delete(m_hAllocator, m_Blocks[i]);
    }
}

VkResult OaVmaBlockVector::CreateMinBlocks()
{
    for (size_t i = 0; i < m_MinBlockCount; ++i)
    {
        VkResult res = CreateBlock(m_PreferredBlockSize, OA_VMA_NULL);
        if (res != VK_SUCCESS)
        {
            return res;
        }
    }
    return VK_SUCCESS;
}

void OaVmaBlockVector::AddStatistics(OaVmaStatistics& inoutStats)
{
    OaVmaMutexLockRead lock(m_Mutex, m_hAllocator->m_UseMutex);

    const size_t blockCount = m_Blocks.size();
    for (uint32_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
    {
        const OaVmaDeviceMemoryBlock* const pBlock = m_Blocks[blockIndex];
        OA_VMA_ASSERT(pBlock);
        OA_VMA_HEAVY_ASSERT(pBlock->Validate());
        pBlock->m_pMetadata->AddStatistics(inoutStats);
    }
}

void OaVmaBlockVector::AddDetailedStatistics(OaVmaDetailedStatistics& inoutStats)
{
    OaVmaMutexLockRead lock(m_Mutex, m_hAllocator->m_UseMutex);

    const size_t blockCount = m_Blocks.size();
    for (uint32_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
    {
        const OaVmaDeviceMemoryBlock* const pBlock = m_Blocks[blockIndex];
        OA_VMA_ASSERT(pBlock);
        OA_VMA_HEAVY_ASSERT(pBlock->Validate());
        pBlock->m_pMetadata->AddDetailedStatistics(inoutStats);
    }
}

bool OaVmaBlockVector::IsEmpty()
{
    OaVmaMutexLockRead lock(m_Mutex, m_hAllocator->m_UseMutex);
    return m_Blocks.empty();
}

bool OaVmaBlockVector::IsCorruptionDetectionEnabled() const
{
#if (OA_VMA_DEBUG_DETECT_CORRUPTION == 0) || (OA_VMA_DEBUG_MARGIN == 0)
    return false;
#else
    constexpr uint32_t requiredMemFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return (m_Algorithm == 0 || m_Algorithm == OA_VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT) &&
        (m_hAllocator->m_MemProps.memoryTypes[m_MemoryTypeIndex].propertyFlags & requiredMemFlags) == requiredMemFlags;
#endif
}

VkResult OaVmaBlockVector::Allocate(
    VkDeviceSize size,
    VkDeviceSize alignment,
    const OaVmaAllocationCreateInfo& createInfo,
    OaVmaSuballocationType suballocType,
    size_t allocationCount,
    OaVmaAllocation* pAllocations)
{
    size_t allocIndex = 0;
    VkResult res = VK_SUCCESS;

    alignment = OA_VMA_MAX(alignment, m_MinAllocationAlignment);

    if (IsCorruptionDetectionEnabled())
    {
        size = OaVmaAlignUp<VkDeviceSize>(size, sizeof(OA_VMA_CORRUPTION_DETECTION_MAGIC_VALUE));
        alignment = OaVmaAlignUp<VkDeviceSize>(alignment, sizeof(OA_VMA_CORRUPTION_DETECTION_MAGIC_VALUE));
    }

    {
        OaVmaMutexLockWrite lock(m_Mutex, m_hAllocator->m_UseMutex);
        for (; allocIndex < allocationCount; ++allocIndex)
        {
            res = AllocatePage(
                size,
                alignment,
                createInfo,
                suballocType,
                pAllocations + allocIndex);
            if (res != VK_SUCCESS)
            {
                break;
            }
        }
    }

    if (res != VK_SUCCESS)
    {
        // Free all already created allocations.
        while (allocIndex--)
            Free(pAllocations[allocIndex]);
        memset(pAllocations, 0, sizeof(OaVmaAllocation) * allocationCount);
    }

    return res;
}

VkResult OaVmaBlockVector::AllocatePage(
    VkDeviceSize size,
    VkDeviceSize alignment,
    const OaVmaAllocationCreateInfo& createInfo,
    OaVmaSuballocationType suballocType,
    OaVmaAllocation* pAllocation)
{
    const bool isUpperAddress = (createInfo.flags & OA_VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT) != 0;

    VkDeviceSize freeMemory = 0;
    {
        const uint32_t heapIndex = m_hAllocator->MemoryTypeIndexToHeapIndex(m_MemoryTypeIndex);
        OaVmaBudget heapBudget = {};
        m_hAllocator->GetHeapBudgets(&heapBudget, heapIndex, 1);
        freeMemory = (heapBudget.usage < heapBudget.budget) ? (heapBudget.budget - heapBudget.usage) : 0;
    }

    const bool canFallbackToDedicated = !HasExplicitBlockSize() &&
        (createInfo.flags & OA_VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) == 0;
    const bool canCreateNewBlock =
        ((createInfo.flags & OA_VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) == 0) &&
        (m_Blocks.size() < m_MaxBlockCount) &&
        (freeMemory >= size || !canFallbackToDedicated);
    uint32_t strategy = createInfo.flags & OA_VMA_ALLOCATION_CREATE_STRATEGY_MASK;

    // Upper address can only be used with linear allocator and within single memory block.
    if (isUpperAddress &&
        (m_Algorithm != OA_VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT || m_MaxBlockCount > 1))
    {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    // Early reject: requested allocation size is larger that maximum block size for this block vector.
    if (size + OA_VMA_DEBUG_MARGIN > m_PreferredBlockSize)
    {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    // 1. Search existing allocations. Try to allocate.
    if (m_Algorithm == OA_VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT)
    {
        // Use only last block.
        if (!m_Blocks.empty())
        {
            OaVmaDeviceMemoryBlock* const pCurrBlock = m_Blocks.back();
            OA_VMA_ASSERT(pCurrBlock);
            VkResult res = AllocateFromBlock(
                pCurrBlock, size, alignment, createInfo.flags, createInfo.pUserData, suballocType, strategy, pAllocation);
            if (res == VK_SUCCESS)
            {
                OA_VMA_DEBUG_LOG_FORMAT("    Returned from last block #%" PRIu32, pCurrBlock->GetId());
                IncrementallySortBlocks();
                return VK_SUCCESS;
            }
        }
    }
    else
    {
        if (strategy != OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT) // MIN_MEMORY or default
        {
            const bool isHostVisible =
                (m_hAllocator->m_MemProps.memoryTypes[m_MemoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
            if(isHostVisible)
            {
                const bool isMappingAllowed = (createInfo.flags &
                    (OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0;
                /*
                For non-mappable allocations, check blocks that are not mapped first.
                For mappable allocations, check blocks that are already mapped first.
                This way, having many blocks, we will separate mappable and non-mappable allocations,
                hopefully limiting the number of blocks that are mapped, which will help tools like RenderDoc.
                */
                for(size_t mappingI = 0; mappingI < 2; ++mappingI)
                {
                    // Forward order in m_Blocks - prefer blocks with smallest amount of free space.
                    for (size_t blockIndex = 0; blockIndex < m_Blocks.size(); ++blockIndex)
                    {
                        OaVmaDeviceMemoryBlock* const pCurrBlock = m_Blocks[blockIndex];
                        OA_VMA_ASSERT(pCurrBlock);
                        const bool isBlockMapped = pCurrBlock->GetMappedData() != OA_VMA_NULL;
                        if((mappingI == 0) == (isMappingAllowed == isBlockMapped))
                        {
                            VkResult res = AllocateFromBlock(
                                pCurrBlock, size, alignment, createInfo.flags, createInfo.pUserData, suballocType, strategy, pAllocation);
                            if (res == VK_SUCCESS)
                            {
                                OA_VMA_DEBUG_LOG_FORMAT("    Returned from existing block #%" PRIu32, pCurrBlock->GetId());
                                IncrementallySortBlocks();
                                return VK_SUCCESS;
                            }
                        }
                    }
                }
            }
            else
            {
                // Forward order in m_Blocks - prefer blocks with smallest amount of free space.
                for (size_t blockIndex = 0; blockIndex < m_Blocks.size(); ++blockIndex)
                {
                    OaVmaDeviceMemoryBlock* const pCurrBlock = m_Blocks[blockIndex];
                    OA_VMA_ASSERT(pCurrBlock);
                    VkResult res = AllocateFromBlock(
                        pCurrBlock, size, alignment, createInfo.flags, createInfo.pUserData, suballocType, strategy, pAllocation);
                    if (res == VK_SUCCESS)
                    {
                        OA_VMA_DEBUG_LOG_FORMAT("    Returned from existing block #%" PRIu32, pCurrBlock->GetId());
                        IncrementallySortBlocks();
                        return VK_SUCCESS;
                    }
                }
            }
        }
        else // OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT
        {
            // Backward order in m_Blocks - prefer blocks with largest amount of free space.
            for (size_t blockIndex = m_Blocks.size(); blockIndex--; )
            {
                OaVmaDeviceMemoryBlock* const pCurrBlock = m_Blocks[blockIndex];
                OA_VMA_ASSERT(pCurrBlock);
                VkResult res = AllocateFromBlock(pCurrBlock, size, alignment, createInfo.flags, createInfo.pUserData, suballocType, strategy, pAllocation);
                if (res == VK_SUCCESS)
                {
                    OA_VMA_DEBUG_LOG_FORMAT("    Returned from existing block #%" PRIu32, pCurrBlock->GetId());
                    IncrementallySortBlocks();
                    return VK_SUCCESS;
                }
            }
        }
    }

    // 2. Try to create new block.
    if (canCreateNewBlock)
    {
        // Calculate optimal size for new block.
        VkDeviceSize newBlockSize = m_PreferredBlockSize;
        uint32_t newBlockSizeShift = 0;
        const uint32_t NEW_BLOCK_SIZE_SHIFT_MAX = 3;

        if (!m_ExplicitBlockSize)
        {
            // Allocate 1/8, 1/4, 1/2 as first blocks.
            const VkDeviceSize maxExistingBlockSize = CalcMaxBlockSize();
            for (uint32_t i = 0; i < NEW_BLOCK_SIZE_SHIFT_MAX; ++i)
            {
                const VkDeviceSize smallerNewBlockSize = newBlockSize / 2;
                if (smallerNewBlockSize > maxExistingBlockSize && smallerNewBlockSize >= size * 2)
                {
                    newBlockSize = smallerNewBlockSize;
                    ++newBlockSizeShift;
                }
                else
                {
                    break;
                }
            }
        }

        size_t newBlockIndex = 0;
        VkResult res = (newBlockSize <= freeMemory || !canFallbackToDedicated) ?
            CreateBlock(newBlockSize, &newBlockIndex) : VK_ERROR_OUT_OF_DEVICE_MEMORY;
        // Allocation of this size failed? Try 1/2, 1/4, 1/8 of m_PreferredBlockSize.
        if (!m_ExplicitBlockSize)
        {
            while (res < 0 && newBlockSizeShift < NEW_BLOCK_SIZE_SHIFT_MAX)
            {
                const VkDeviceSize smallerNewBlockSize = newBlockSize / 2;
                if (smallerNewBlockSize >= size)
                {
                    newBlockSize = smallerNewBlockSize;
                    ++newBlockSizeShift;
                    res = (newBlockSize <= freeMemory || !canFallbackToDedicated) ?
                        CreateBlock(newBlockSize, &newBlockIndex) : VK_ERROR_OUT_OF_DEVICE_MEMORY;
                }
                else
                {
                    break;
                }
            }
        }

        if (res == VK_SUCCESS)
        {
            OaVmaDeviceMemoryBlock* const pBlock = m_Blocks[newBlockIndex];
            OA_VMA_ASSERT(pBlock->m_pMetadata->GetSize() >= size);

            res = AllocateFromBlock(
                pBlock, size, alignment, createInfo.flags, createInfo.pUserData, suballocType, strategy, pAllocation);
            if (res == VK_SUCCESS)
            {
                OA_VMA_DEBUG_LOG_FORMAT("    Created new block #%" PRIu32 " Size=%" PRIu64, pBlock->GetId(), newBlockSize);
                IncrementallySortBlocks();
                return VK_SUCCESS;
            }

            // Allocation from new block failed, possibly due to OA_VMA_DEBUG_MARGIN or alignment.
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }

    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

void OaVmaBlockVector::Free(OaVmaAllocation hAllocation)
{
    OaVmaDeviceMemoryBlock* pBlockToDelete = OA_VMA_NULL;

    bool budgetExceeded = false;
    {
        const uint32_t heapIndex = m_hAllocator->MemoryTypeIndexToHeapIndex(m_MemoryTypeIndex);
        OaVmaBudget heapBudget = {};
        m_hAllocator->GetHeapBudgets(&heapBudget, heapIndex, 1);
        budgetExceeded = heapBudget.usage >= heapBudget.budget;
    }

    // Scope for lock.
    {
        OaVmaMutexLockWrite lock(m_Mutex, m_hAllocator->m_UseMutex);

        OaVmaDeviceMemoryBlock* pBlock = hAllocation->GetBlock();

        if (IsCorruptionDetectionEnabled())
        {
            VkResult res = pBlock->ValidateMagicValueAfterAllocation(m_hAllocator, hAllocation->GetOffset(), hAllocation->GetSize());
            OA_VMA_ASSERT(res == VK_SUCCESS && "Couldn't map block memory to validate magic value.");
        }

        if (hAllocation->IsPersistentMap())
        {
            pBlock->Unmap(m_hAllocator, 1);
        }

        const bool hadEmptyBlockBeforeFree = HasEmptyBlock();
        pBlock->m_pMetadata->Free(hAllocation->GetAllocHandle());
        pBlock->PostFree(m_hAllocator);
        OA_VMA_HEAVY_ASSERT(pBlock->Validate());

        OA_VMA_DEBUG_LOG_FORMAT("  Freed from MemoryTypeIndex=%" PRIu32, m_MemoryTypeIndex);

        const bool canDeleteBlock = m_Blocks.size() > m_MinBlockCount;
        // pBlock became empty after this deallocation.
        if (pBlock->m_pMetadata->IsEmpty())
        {
            // Already had empty block. We don't want to have two, so delete this one.
            if ((hadEmptyBlockBeforeFree || budgetExceeded) && canDeleteBlock)
            {
                pBlockToDelete = pBlock;
                Remove(pBlock);
            }
            // else: We now have one empty block - leave it. A hysteresis to avoid allocating whole block back and forth.
        }
        // pBlock didn't become empty, but we have another empty block - find and free that one.
        // (This is optional, heuristics.)
        else if (hadEmptyBlockBeforeFree && canDeleteBlock)
        {
            OaVmaDeviceMemoryBlock* pLastBlock = m_Blocks.back();
            if (pLastBlock->m_pMetadata->IsEmpty())
            {
                pBlockToDelete = pLastBlock;
                m_Blocks.pop_back();
            }
        }

        IncrementallySortBlocks();

        m_hAllocator->m_Budget.RemoveAllocation(m_hAllocator->MemoryTypeIndexToHeapIndex(m_MemoryTypeIndex), hAllocation->GetSize());
        hAllocation->Destroy(m_hAllocator);
        m_hAllocator->m_AllocationObjectAllocator.Free(hAllocation);
    }

    // Destruction of a free block. Deferred until this point, outside of mutex
    // lock, for performance reason.
    if (pBlockToDelete != OA_VMA_NULL)
    {
        OA_VMA_DEBUG_LOG_FORMAT("    Deleted empty block #%" PRIu32, pBlockToDelete->GetId());
        pBlockToDelete->Destroy(m_hAllocator);
        OaVma_delete(m_hAllocator, pBlockToDelete);
    }
}

VkDeviceSize OaVmaBlockVector::CalcMaxBlockSize() const
{
    VkDeviceSize result = 0;
    for (size_t i = m_Blocks.size(); i--; )
    {
        result = OA_VMA_MAX(result, m_Blocks[i]->m_pMetadata->GetSize());
        if (result >= m_PreferredBlockSize)
        {
            break;
        }
    }
    return result;
}

void OaVmaBlockVector::Remove(OaVmaDeviceMemoryBlock* pBlock)
{
    for (uint32_t blockIndex = 0; blockIndex < m_Blocks.size(); ++blockIndex)
    {
        if (m_Blocks[blockIndex] == pBlock)
        {
            OaVmaVectorRemove(m_Blocks, blockIndex);
            return;
        }
    }
    OA_VMA_ASSERT(0);
}

void OaVmaBlockVector::IncrementallySortBlocks()
{
    if (!m_IncrementalSort)
        return;
    if (m_Algorithm != OA_VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT)
    {
        // Bubble sort only until first swap.
        for (size_t i = 1; i < m_Blocks.size(); ++i)
        {
            if (m_Blocks[i - 1]->m_pMetadata->GetSumFreeSize() > m_Blocks[i]->m_pMetadata->GetSumFreeSize())
            {
                std::swap(m_Blocks[i - 1], m_Blocks[i]);
                return;
            }
        }
    }
}

void OaVmaBlockVector::SortByFreeSize()
{
    OA_VMA_SORT(m_Blocks.begin(), m_Blocks.end(),
        [](OaVmaDeviceMemoryBlock* b1, OaVmaDeviceMemoryBlock* b2) -> bool
        {
            return b1->m_pMetadata->GetSumFreeSize() < b2->m_pMetadata->GetSumFreeSize();
        });
}

VkResult OaVmaBlockVector::AllocateFromBlock(
    OaVmaDeviceMemoryBlock* pBlock,
    VkDeviceSize size,
    VkDeviceSize alignment,
    OaVmaAllocationCreateFlags allocFlags,
    void* pUserData,
    OaVmaSuballocationType suballocType,
    uint32_t strategy,
    OaVmaAllocation* pAllocation)
{
    const bool isUpperAddress = (allocFlags & OA_VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT) != 0;

    OaVmaAllocationRequest currRequest = {};
    if (pBlock->m_pMetadata->CreateAllocationRequest(
        size,
        alignment,
        isUpperAddress,
        suballocType,
        strategy,
        &currRequest))
    {
        return CommitAllocationRequest(currRequest, pBlock, alignment, allocFlags, pUserData, suballocType, pAllocation);
    }
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

VkResult OaVmaBlockVector::CommitAllocationRequest(
    OaVmaAllocationRequest& allocRequest,
    OaVmaDeviceMemoryBlock* pBlock,
    VkDeviceSize alignment,
    OaVmaAllocationCreateFlags allocFlags,
    void* pUserData,
    OaVmaSuballocationType suballocType,
    OaVmaAllocation* pAllocation)
{
    const bool mapped = (allocFlags & OA_VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0;
    const bool isUserDataString = (allocFlags & OA_VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0;
    const bool isMappingAllowed = (allocFlags &
        (OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0;

    pBlock->PostAlloc(m_hAllocator);
    // Allocate from pCurrBlock.
    if (mapped)
    {
        VkResult res = pBlock->Map(m_hAllocator, 1, OA_VMA_NULL);
        if (res != VK_SUCCESS)
        {
            return res;
        }
    }

    *pAllocation = m_hAllocator->m_AllocationObjectAllocator.Allocate(isMappingAllowed);
    pBlock->m_pMetadata->Alloc(allocRequest, suballocType, *pAllocation);
    (*pAllocation)->InitBlockAllocation(
        pBlock,
        allocRequest.allocHandle,
        alignment,
        allocRequest.size, // Not size, as actual allocation size may be larger than requested!
        m_MemoryTypeIndex,
        suballocType,
        mapped);
    OA_VMA_HEAVY_ASSERT(pBlock->Validate());
    if (isUserDataString)
        (*pAllocation)->SetName(m_hAllocator, (const char*)pUserData);
    else
        (*pAllocation)->SetUserData(m_hAllocator, pUserData);
    m_hAllocator->m_Budget.AddAllocation(m_hAllocator->MemoryTypeIndexToHeapIndex(m_MemoryTypeIndex), allocRequest.size);

#if OA_VMA_DEBUG_INITIALIZE_ALLOCATIONS
    m_hAllocator->FillAllocation(*pAllocation, OA_VMA_ALLOCATION_FILL_PATTERN_CREATED);
#endif

    if (IsCorruptionDetectionEnabled())
    {
        VkResult res = pBlock->WriteMagicValueAfterAllocation(m_hAllocator, (*pAllocation)->GetOffset(), allocRequest.size);
        OA_VMA_ASSERT(res == VK_SUCCESS && "Couldn't map block memory to write magic value.");
    }
    return VK_SUCCESS;
}

VkResult OaVmaBlockVector::CreateBlock(VkDeviceSize blockSize, size_t* pNewBlockIndex)
{
    VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.pNext = m_pMemoryAllocateNext;
    allocInfo.memoryTypeIndex = m_MemoryTypeIndex;
    allocInfo.allocationSize = blockSize;

#if OA_VMA_BUFFER_DEVICE_ADDRESS
    // Every standalone block can potentially contain a buffer with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT - always enable the feature.
    VkMemoryAllocateFlagsInfoKHR allocFlagsInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR };
    if (m_hAllocator->m_UseKhrBufferDeviceAddress)
    {
        allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
        OaVmaPnextChainPushFront(&allocInfo, &allocFlagsInfo);
    }
#endif // OA_VMA_BUFFER_DEVICE_ADDRESS

#if OA_VMA_MEMORY_PRIORITY
    VkMemoryPriorityAllocateInfoEXT priorityInfo = { VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT };
    if (m_hAllocator->m_UseExtMemoryPriority)
    {
        OA_VMA_ASSERT(m_Priority >= 0.F && m_Priority <= 1.F);
        priorityInfo.priority = m_Priority;
        OaVmaPnextChainPushFront(&allocInfo, &priorityInfo);
    }
#endif // OA_VMA_MEMORY_PRIORITY

#if OA_VMA_EXTERNAL_MEMORY
    // Attach VkExportMemoryAllocateInfoKHR if necessary.
    VkExportMemoryAllocateInfoKHR exportMemoryAllocInfo = { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR };
    exportMemoryAllocInfo.handleTypes = m_hAllocator->GetExternalMemoryHandleTypeFlags(m_MemoryTypeIndex);
    if (exportMemoryAllocInfo.handleTypes != 0)
    {
        OaVmaPnextChainPushFront(&allocInfo, &exportMemoryAllocInfo);
    }
#endif // OA_VMA_EXTERNAL_MEMORY

    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkResult res = m_hAllocator->AllocateVulkanMemory(&allocInfo, &mem);
    if (res < 0)
    {
        return res;
    }

    // New VkDeviceMemory successfully created.

    // Create new Allocation for it.
    OaVmaDeviceMemoryBlock* const pBlock = OaVma_new(m_hAllocator, OaVmaDeviceMemoryBlock)(m_hAllocator);
    pBlock->Init(
        m_hAllocator,
        m_hParentPool,
        m_MemoryTypeIndex,
        mem,
        allocInfo.allocationSize,
        m_NextBlockId++,
        m_Algorithm,
        m_BufferImageGranularity);

    m_Blocks.push_back(pBlock);
    if (pNewBlockIndex != OA_VMA_NULL)
    {
        *pNewBlockIndex = m_Blocks.size() - 1;
    }

    return VK_SUCCESS;
}

bool OaVmaBlockVector::HasEmptyBlock()
{
    for (size_t index = 0, count = m_Blocks.size(); index < count; ++index)
    {
        OaVmaDeviceMemoryBlock* const pBlock = m_Blocks[index];
        if (pBlock->m_pMetadata->IsEmpty())
        {
            return true;
        }
    }
    return false;
}

#if OA_VMA_STATS_STRING_ENABLED
void OaVmaBlockVector::PrintDetailedMap(class OaVmaJsonWriter& json)
{
    OaVmaMutexLockRead lock(m_Mutex, m_hAllocator->m_UseMutex);


    json.BeginObject();
    for (size_t i = 0; i < m_Blocks.size(); ++i)
    {
        json.BeginString();
        json.ContinueString(m_Blocks[i]->GetId());
        json.EndString();

        json.BeginObject();
        json.WriteString("MapRefCount");
        json.WriteNumber(m_Blocks[i]->GetMapRefCount());

        m_Blocks[i]->m_pMetadata->PrintDetailedMap(json);
        json.EndObject();
    }
    json.EndObject();
}
#endif // OA_VMA_STATS_STRING_ENABLED

VkResult OaVmaBlockVector::CheckCorruption()
{
    if (!IsCorruptionDetectionEnabled())
    {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    OaVmaMutexLockRead lock(m_Mutex, m_hAllocator->m_UseMutex);
    for (uint32_t blockIndex = 0; blockIndex < m_Blocks.size(); ++blockIndex)
    {
        OaVmaDeviceMemoryBlock* const pBlock = m_Blocks[blockIndex];
        OA_VMA_ASSERT(pBlock);
        VkResult res = pBlock->CheckCorruption(m_hAllocator);
        if (res != VK_SUCCESS)
        {
            return res;
        }
    }
    return VK_SUCCESS;
}

#endif // _OA_VMA_BLOCK_VECTOR_FUNCTIONS

#ifndef _OA_VMA_DEFRAGMENTATION_CONTEXT_FUNCTIONS
OaVmaDefragmentationContext_T::OaVmaDefragmentationContext_T(
    OaVmaAllocator hAllocator,
    const OaVmaDefragmentationInfo& info)
    : m_MaxPassBytes(info.maxBytesPerPass == 0 ? VK_WHOLE_SIZE : info.maxBytesPerPass),
    m_MaxPassAllocations(info.maxAllocationsPerPass == 0 ? UINT32_MAX : info.maxAllocationsPerPass),
    m_BreakCallback(info.pfnBreakCallback),
    m_BreakCallbackUserData(info.pBreakCallbackUserData),
    m_MoveAllocator(hAllocator->GetAllocationCallbacks()),
    m_Moves(m_MoveAllocator),
    m_Algorithm(info.flags & OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_MASK)
{
    if (info.pool != OA_VMA_NULL)
    {
        m_BlockVectorCount = 1;
        m_PoolBlockVector = &info.pool->m_BlockVector;
        m_pBlockVectors = &m_PoolBlockVector;
        m_PoolBlockVector->SetIncrementalSort(false);
        m_PoolBlockVector->SortByFreeSize();
    }
    else
    {
        m_BlockVectorCount = hAllocator->GetMemoryTypeCount();
        m_PoolBlockVector = OA_VMA_NULL;
        m_pBlockVectors = hAllocator->m_pBlockVectors;
        for (uint32_t i = 0; i < m_BlockVectorCount; ++i)
        {
            OaVmaBlockVector* vector = m_pBlockVectors[i];
            if (vector != OA_VMA_NULL)
            {
                vector->SetIncrementalSort(false);
                vector->SortByFreeSize();
            }
        }
    }

    switch (m_Algorithm)
    {
    case 0: // Default algorithm
        m_Algorithm = OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT;
        m_AlgorithmState = OaVma_new_array(hAllocator, StateBalanced, m_BlockVectorCount);
        break;
    case OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT:
        m_AlgorithmState = OaVma_new_array(hAllocator, StateBalanced, m_BlockVectorCount);
        break;
    case OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT:
        if (hAllocator->GetBufferImageGranularity() > 1)
        {
            m_AlgorithmState = OaVma_new_array(hAllocator, StateExtensive, m_BlockVectorCount);
        }
        break;
    default:
        ; // Do nothing.
    }
}

OaVmaDefragmentationContext_T::~OaVmaDefragmentationContext_T()
{
    if (m_PoolBlockVector != OA_VMA_NULL)
    {
        m_PoolBlockVector->SetIncrementalSort(true);
    }
    else
    {
        for (uint32_t i = 0; i < m_BlockVectorCount; ++i)
        {
            OaVmaBlockVector* vector = m_pBlockVectors[i];
            if (vector != OA_VMA_NULL)
                vector->SetIncrementalSort(true);
        }
    }

    if (m_AlgorithmState)
    {
        switch (m_Algorithm)
        {
        case OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT:
            OaVma_delete_array(m_MoveAllocator.m_pCallbacks, reinterpret_cast<StateBalanced*>(m_AlgorithmState), m_BlockVectorCount);
            break;
        case OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT:
            OaVma_delete_array(m_MoveAllocator.m_pCallbacks, reinterpret_cast<StateExtensive*>(m_AlgorithmState), m_BlockVectorCount);
            break;
        default:
            OA_VMA_ASSERT(0);
        }
    }
}

VkResult OaVmaDefragmentationContext_T::DefragmentPassBegin(OaVmaDefragmentationPassMoveInfo& moveInfo)
{
    if (m_PoolBlockVector != OA_VMA_NULL)
    {
        OaVmaMutexLockWrite lock(m_PoolBlockVector->GetMutex(), m_PoolBlockVector->GetAllocator()->m_UseMutex);

        if (m_PoolBlockVector->GetBlockCount() > 1)
            ComputeDefragmentation(*m_PoolBlockVector, 0);
        else if (m_PoolBlockVector->GetBlockCount() == 1)
            ReallocWithinBlock(*m_PoolBlockVector, m_PoolBlockVector->GetBlock(0));
    }
    else
    {
        for (uint32_t i = 0; i < m_BlockVectorCount; ++i)
        {
            if (m_pBlockVectors[i] != OA_VMA_NULL)
            {
                OaVmaMutexLockWrite lock(m_pBlockVectors[i]->GetMutex(), m_pBlockVectors[i]->GetAllocator()->m_UseMutex);

                if (m_pBlockVectors[i]->GetBlockCount() > 1)
                {
                    if (ComputeDefragmentation(*m_pBlockVectors[i], i))
                        break;
                }
                else if (m_pBlockVectors[i]->GetBlockCount() == 1)
                {
                    if (ReallocWithinBlock(*m_pBlockVectors[i], m_pBlockVectors[i]->GetBlock(0)))
                        break;
                }
            }
        }
    }

    moveInfo.moveCount = static_cast<uint32_t>(m_Moves.size());
    if (moveInfo.moveCount > 0)
    {
        moveInfo.pMoves = m_Moves.data();
        return VK_INCOMPLETE;
    }

    moveInfo.pMoves = OA_VMA_NULL;
    return VK_SUCCESS;
}

VkResult OaVmaDefragmentationContext_T::DefragmentPassEnd(OaVmaDefragmentationPassMoveInfo& moveInfo)
{
    OA_VMA_ASSERT(moveInfo.moveCount > 0 ? moveInfo.pMoves != OA_VMA_NULL : true);

    VkResult result = VK_SUCCESS;
    OaVmaStlAllocator<FragmentedBlock> blockAllocator(m_MoveAllocator.m_pCallbacks);
    OaVmaVector<FragmentedBlock, OaVmaStlAllocator<FragmentedBlock>> immovableBlocks(blockAllocator);
    OaVmaVector<FragmentedBlock, OaVmaStlAllocator<FragmentedBlock>> mappedBlocks(blockAllocator);

    OaVmaAllocator allocator = OA_VMA_NULL;
    for (uint32_t i = 0; i < moveInfo.moveCount; ++i)
    {
        OaVmaDefragmentationMove& move = moveInfo.pMoves[i];
        size_t prevCount = 0;
        size_t currentCount = 0;
        VkDeviceSize freedBlockSize = 0;

        uint32_t vectorIndex = 0;
        OaVmaBlockVector* vector = OA_VMA_NULL;
        if (m_PoolBlockVector != OA_VMA_NULL)
        {
            vector = m_PoolBlockVector;
        }
        else
        {
            vectorIndex = move.srcAllocation->GetMemoryTypeIndex();
            vector = m_pBlockVectors[vectorIndex];
            OA_VMA_ASSERT(vector != OA_VMA_NULL);
        }

        switch (move.operation)
        {
        case OA_VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY:
        {
            uint8_t mapCount = move.srcAllocation->SwapBlockAllocation(vector->m_hAllocator, move.dstTmpAllocation);
            if (mapCount > 0)
            {
                allocator = vector->m_hAllocator;
                OaVmaDeviceMemoryBlock* newMapBlock = move.srcAllocation->GetBlock();
                bool notPresent = true;
                for (FragmentedBlock& block : mappedBlocks)
                {
                    if (block.block == newMapBlock)
                    {
                        notPresent = false;
                        block.data += mapCount;
                        break;
                    }
                }
                if (notPresent)
                    mappedBlocks.push_back({ mapCount, newMapBlock });
            }

            // Scope for locks, Free have it's own lock
            {
                OaVmaMutexLockRead lock(vector->GetMutex(), vector->GetAllocator()->m_UseMutex);
                prevCount = vector->GetBlockCount();
                freedBlockSize = move.dstTmpAllocation->GetBlock()->m_pMetadata->GetSize();
            }
            vector->Free(move.dstTmpAllocation);
            {
                OaVmaMutexLockRead lock(vector->GetMutex(), vector->GetAllocator()->m_UseMutex);
                currentCount = vector->GetBlockCount();
            }

            result = VK_INCOMPLETE;
            break;
        }
        case OA_VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE:
        {
            m_PassStats.bytesMoved -= move.srcAllocation->GetSize();
            --m_PassStats.allocationsMoved;
            vector->Free(move.dstTmpAllocation);

            OaVmaDeviceMemoryBlock* newBlock = move.srcAllocation->GetBlock();
            bool notPresent = true;
            for (const FragmentedBlock& block : immovableBlocks)
            {
                if (block.block == newBlock)
                {
                    notPresent = false;
                    break;
                }
            }
            if (notPresent)
                immovableBlocks.push_back({ vectorIndex, newBlock });
            break;
        }
        case OA_VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY:
        {
            m_PassStats.bytesMoved -= move.srcAllocation->GetSize();
            --m_PassStats.allocationsMoved;
            // Scope for locks, Free have it's own lock
            {
                OaVmaMutexLockRead lock(vector->GetMutex(), vector->GetAllocator()->m_UseMutex);
                prevCount = vector->GetBlockCount();
                freedBlockSize = move.srcAllocation->GetBlock()->m_pMetadata->GetSize();
            }
            vector->Free(move.srcAllocation);
            {
                OaVmaMutexLockRead lock(vector->GetMutex(), vector->GetAllocator()->m_UseMutex);
                currentCount = vector->GetBlockCount();
            }
            freedBlockSize *= prevCount - currentCount;

            VkDeviceSize dstBlockSize = SIZE_MAX;
            {
                OaVmaMutexLockRead lock(vector->GetMutex(), vector->GetAllocator()->m_UseMutex);
                dstBlockSize = move.dstTmpAllocation->GetBlock()->m_pMetadata->GetSize();
            }
            vector->Free(move.dstTmpAllocation);
            {
                OaVmaMutexLockRead lock(vector->GetMutex(), vector->GetAllocator()->m_UseMutex);
                freedBlockSize += dstBlockSize * (currentCount - vector->GetBlockCount());
                currentCount = vector->GetBlockCount();
            }

            result = VK_INCOMPLETE;
            break;
        }
        default:
            OA_VMA_ASSERT(0);
        }

        if (prevCount > currentCount)
        {
            size_t freedBlocks = prevCount - currentCount;
            m_PassStats.deviceMemoryBlocksFreed += static_cast<uint32_t>(freedBlocks);
            m_PassStats.bytesFreed += freedBlockSize;
        }

        if(m_Algorithm == OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT &&
            m_AlgorithmState != OA_VMA_NULL)
        {
            // Avoid unnecessary tries to allocate when new free block is available
            StateExtensive& state = reinterpret_cast<StateExtensive*>(m_AlgorithmState)[vectorIndex];
            if (state.firstFreeBlock != SIZE_MAX)
            {
                const size_t diff = prevCount - currentCount;
                if (state.firstFreeBlock >= diff)
                {
                    state.firstFreeBlock -= diff;
                    if (state.firstFreeBlock != 0)
                        state.firstFreeBlock -= vector->GetBlock(state.firstFreeBlock - 1)->m_pMetadata->IsEmpty();
                }
                else
                    state.firstFreeBlock = 0;
            }
        }
    }
    moveInfo.moveCount = 0;
    moveInfo.pMoves = OA_VMA_NULL;
    m_Moves.clear();

    // Update stats
    m_GlobalStats.allocationsMoved += m_PassStats.allocationsMoved;
    m_GlobalStats.bytesFreed += m_PassStats.bytesFreed;
    m_GlobalStats.bytesMoved += m_PassStats.bytesMoved;
    m_GlobalStats.deviceMemoryBlocksFreed += m_PassStats.deviceMemoryBlocksFreed;
    m_PassStats = { 0 };

    // Move blocks with immovable allocations according to algorithm
    if (!immovableBlocks.empty())
    {
        do
        {
            if(m_Algorithm == OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT)
            {
                if (m_AlgorithmState != OA_VMA_NULL)
                {
                    bool swapped = false;
                    // Move to the start of free blocks range
                    for (const FragmentedBlock& block : immovableBlocks)
                    {
                        StateExtensive& state = reinterpret_cast<StateExtensive*>(m_AlgorithmState)[block.data];
                        if (state.operation != StateExtensive::Operation::Cleanup)
                        {
                            OaVmaBlockVector* vector = m_pBlockVectors[block.data];
                            OaVmaMutexLockWrite lock(vector->GetMutex(), vector->GetAllocator()->m_UseMutex);

                            for (size_t i = 0, count = vector->GetBlockCount() - m_ImmovableBlockCount; i < count; ++i)
                            {
                                if (vector->GetBlock(i) == block.block)
                                {
                                    std::swap(vector->m_Blocks[i], vector->m_Blocks[vector->GetBlockCount() - ++m_ImmovableBlockCount]);
                                    if (state.firstFreeBlock != SIZE_MAX)
                                    {
                                        if (i + 1 < state.firstFreeBlock)
                                        {
                                            if (state.firstFreeBlock > 1)
                                                std::swap(vector->m_Blocks[i], vector->m_Blocks[--state.firstFreeBlock]);
                                            else
                                                --state.firstFreeBlock;
                                        }
                                    }
                                    swapped = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (swapped)
                        result = VK_INCOMPLETE;
                    break;
                }
            }

            // Move to the beginning
            for (const FragmentedBlock& block : immovableBlocks)
            {
                OaVmaBlockVector* vector = m_pBlockVectors[block.data];
                OaVmaMutexLockWrite lock(vector->GetMutex(), vector->GetAllocator()->m_UseMutex);

                for (size_t i = m_ImmovableBlockCount; i < vector->GetBlockCount(); ++i)
                {
                    if (vector->GetBlock(i) == block.block)
                    {
                        std::swap(vector->m_Blocks[i], vector->m_Blocks[m_ImmovableBlockCount++]);
                        break;
                    }
                }
            }
        } while (false);
    }

    // Bulk-map destination blocks
    for (const FragmentedBlock& block : mappedBlocks)
    {
        VkResult res = block.block->Map(allocator, block.data, OA_VMA_NULL);
        OA_VMA_ASSERT(res == VK_SUCCESS);
    }
    return result;
}

bool OaVmaDefragmentationContext_T::ComputeDefragmentation(OaVmaBlockVector& vector, size_t index)
{
    switch (m_Algorithm)
    {
    case OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT:
        return ComputeDefragmentation_Fast(vector);
    case OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT:
        return ComputeDefragmentation_Balanced(vector, index, true);
    case OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL_BIT:
        return ComputeDefragmentation_Full(vector);
    case OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT:
        return ComputeDefragmentation_Extensive(vector, index);
    default:
        OA_VMA_ASSERT(0);
        return ComputeDefragmentation_Balanced(vector, index, true);
    }
}

OaVmaDefragmentationContext_T::MoveAllocationData OaVmaDefragmentationContext_T::GetMoveData(
    OaVmaAllocHandle handle, OaVmaBlockMetadata* metadata)
{
    MoveAllocationData moveData;
    moveData.move.srcAllocation = (OaVmaAllocation)metadata->GetAllocationUserData(handle);
    moveData.size = moveData.move.srcAllocation->GetSize();
    moveData.alignment = moveData.move.srcAllocation->GetAlignment();
    moveData.type = moveData.move.srcAllocation->GetSuballocationType();
    moveData.flags = 0;

    if (moveData.move.srcAllocation->IsPersistentMap())
        moveData.flags |= OA_VMA_ALLOCATION_CREATE_MAPPED_BIT;
    if (moveData.move.srcAllocation->IsMappingAllowed())
        moveData.flags |= OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

    return moveData;
}

OaVmaDefragmentationContext_T::CounterStatus OaVmaDefragmentationContext_T::CheckCounters(VkDeviceSize bytes)
{
    // Check custom criteria if exists
    if (m_BreakCallback && m_BreakCallback(m_BreakCallbackUserData))
        return CounterStatus::End;

    // Ignore allocation if will exceed max size for copy
    if (m_PassStats.bytesMoved + bytes > m_MaxPassBytes)
    {
        if (++m_IgnoredAllocs < MAX_ALLOCS_TO_IGNORE)
            return CounterStatus::Ignore;
        return CounterStatus::End;
    }

    m_IgnoredAllocs = 0;
    return CounterStatus::Pass;
}

bool OaVmaDefragmentationContext_T::IncrementCounters(VkDeviceSize bytes)
{
    m_PassStats.bytesMoved += bytes;
    // Early return when max found
    if (++m_PassStats.allocationsMoved >= m_MaxPassAllocations || m_PassStats.bytesMoved >= m_MaxPassBytes)
    {
        OA_VMA_ASSERT((m_PassStats.allocationsMoved == m_MaxPassAllocations ||
            m_PassStats.bytesMoved == m_MaxPassBytes) && "Exceeded maximal pass threshold!");
        return true;
    }
    return false;
}

bool OaVmaDefragmentationContext_T::ReallocWithinBlock(OaVmaBlockVector& vector, OaVmaDeviceMemoryBlock* block)
{
    OaVmaBlockMetadata* metadata = block->m_pMetadata;

    for (OaVmaAllocHandle handle = metadata->GetAllocationListBegin();
        handle != VK_NULL_HANDLE;
        handle = metadata->GetNextAllocation(handle))
    {
        MoveAllocationData moveData = GetMoveData(handle, metadata);
        // Ignore newly created allocations by defragmentation algorithm
        if (moveData.move.srcAllocation->GetUserData() == this)
            continue;
        switch (CheckCounters(moveData.move.srcAllocation->GetSize()))
        {
        case CounterStatus::Ignore:
            continue;
        case CounterStatus::End:
            return true;
        case CounterStatus::Pass:
            break;
        default:
            OA_VMA_ASSERT(0);
        }

        VkDeviceSize offset = moveData.move.srcAllocation->GetOffset();
        if (offset != 0 && metadata->GetSumFreeSize() >= moveData.size)
        {
            OaVmaAllocationRequest request = {};
            if (metadata->CreateAllocationRequest(
                moveData.size,
                moveData.alignment,
                false,
                moveData.type,
                OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT,
                &request))
            {
                if (metadata->GetAllocationOffset(request.allocHandle) < offset)
                {
                    if (vector.CommitAllocationRequest(
                        request,
                        block,
                        moveData.alignment,
                        moveData.flags,
                        this,
                        moveData.type,
                        &moveData.move.dstTmpAllocation) == VK_SUCCESS)
                    {
                        m_Moves.push_back(moveData.move);
                        if (IncrementCounters(moveData.size))
                            return true;
                    }
                }
            }
        }
    }
    return false;
}

bool OaVmaDefragmentationContext_T::AllocInOtherBlock(size_t start, size_t end, MoveAllocationData& data, OaVmaBlockVector& vector)
{
    for (; start < end; ++start)
    {
        OaVmaDeviceMemoryBlock* dstBlock = vector.GetBlock(start);
        if (dstBlock->m_pMetadata->GetSumFreeSize() >= data.size)
        {
            if (vector.AllocateFromBlock(dstBlock,
                data.size,
                data.alignment,
                data.flags,
                this,
                data.type,
                0,
                &data.move.dstTmpAllocation) == VK_SUCCESS)
            {
                m_Moves.push_back(data.move);
                if (IncrementCounters(data.size))
                    return true;
                break;
            }
        }
    }
    return false;
}

bool OaVmaDefragmentationContext_T::ComputeDefragmentation_Fast(OaVmaBlockVector& vector)
{
    // Move only between blocks

    // Go through allocations in last blocks and try to fit them inside first ones
    for (size_t i = vector.GetBlockCount() - 1; i > m_ImmovableBlockCount; --i)
    {
        OaVmaBlockMetadata* metadata = vector.GetBlock(i)->m_pMetadata;

        for (OaVmaAllocHandle handle = metadata->GetAllocationListBegin();
            handle != VK_NULL_HANDLE;
            handle = metadata->GetNextAllocation(handle))
        {
            MoveAllocationData moveData = GetMoveData(handle, metadata);
            // Ignore newly created allocations by defragmentation algorithm
            if (moveData.move.srcAllocation->GetUserData() == this)
                continue;
            switch (CheckCounters(moveData.move.srcAllocation->GetSize()))
            {
            case CounterStatus::Ignore:
                continue;
            case CounterStatus::End:
                return true;
            case CounterStatus::Pass:
                break;
            default:
                OA_VMA_ASSERT(0);
            }

            // Check all previous blocks for free space
            if (AllocInOtherBlock(0, i, moveData, vector))
                return true;
        }
    }
    return false;
}

bool OaVmaDefragmentationContext_T::ComputeDefragmentation_Balanced(OaVmaBlockVector& vector, size_t index, bool update)
{
    // Go over every allocation and try to fit it in previous blocks at lowest offsets,
    // if not possible: realloc within single block to minimize offset (exclude offset == 0),
    // but only if there are noticeable gaps between them (some heuristic, ex. average size of allocation in block)
    OA_VMA_ASSERT(m_AlgorithmState != OA_VMA_NULL);

    StateBalanced& vectorState = reinterpret_cast<StateBalanced*>(m_AlgorithmState)[index];
    if (update && vectorState.avgAllocSize == UINT64_MAX)
        UpdateVectorStatistics(vector, vectorState);

    const size_t startMoveCount = m_Moves.size();
    VkDeviceSize minimalFreeRegion = vectorState.avgFreeSize / 2;
    for (size_t i = vector.GetBlockCount() - 1; i > m_ImmovableBlockCount; --i)
    {
        OaVmaDeviceMemoryBlock* block = vector.GetBlock(i);
        OaVmaBlockMetadata* metadata = block->m_pMetadata;
        VkDeviceSize prevFreeRegionSize = 0;

        for (OaVmaAllocHandle handle = metadata->GetAllocationListBegin();
            handle != VK_NULL_HANDLE;
            handle = metadata->GetNextAllocation(handle))
        {
            MoveAllocationData moveData = GetMoveData(handle, metadata);
            // Ignore newly created allocations by defragmentation algorithm
            if (moveData.move.srcAllocation->GetUserData() == this)
                continue;
            switch (CheckCounters(moveData.move.srcAllocation->GetSize()))
            {
            case CounterStatus::Ignore:
                continue;
            case CounterStatus::End:
                return true;
            case CounterStatus::Pass:
                break;
            default:
                OA_VMA_ASSERT(0);
            }

            // Check all previous blocks for free space
            const size_t prevMoveCount = m_Moves.size();
            if (AllocInOtherBlock(0, i, moveData, vector))
                return true;

            VkDeviceSize nextFreeRegionSize = metadata->GetNextFreeRegionSize(handle);
            // If no room found then realloc within block for lower offset
            VkDeviceSize offset = moveData.move.srcAllocation->GetOffset();
            if (prevMoveCount == m_Moves.size() && offset != 0 && metadata->GetSumFreeSize() >= moveData.size)
            {
                // Check if realloc will make sense
                if (prevFreeRegionSize >= minimalFreeRegion ||
                    nextFreeRegionSize >= minimalFreeRegion ||
                    moveData.size <= vectorState.avgFreeSize ||
                    moveData.size <= vectorState.avgAllocSize)
                {
                    OaVmaAllocationRequest request = {};
                    if (metadata->CreateAllocationRequest(
                        moveData.size,
                        moveData.alignment,
                        false,
                        moveData.type,
                        OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT,
                        &request))
                    {
                        if (metadata->GetAllocationOffset(request.allocHandle) < offset)
                        {
                            if (vector.CommitAllocationRequest(
                                request,
                                block,
                                moveData.alignment,
                                moveData.flags,
                                this,
                                moveData.type,
                                &moveData.move.dstTmpAllocation) == VK_SUCCESS)
                            {
                                m_Moves.push_back(moveData.move);
                                if (IncrementCounters(moveData.size))
                                    return true;
                            }
                        }
                    }
                }
            }
            prevFreeRegionSize = nextFreeRegionSize;
        }
    }

    // No moves performed, update statistics to current vector state
    if (startMoveCount == m_Moves.size() && !update)
    {
        vectorState.avgAllocSize = UINT64_MAX;
        return ComputeDefragmentation_Balanced(vector, index, false);
    }
    return false;
}

bool OaVmaDefragmentationContext_T::ComputeDefragmentation_Full(OaVmaBlockVector& vector)
{
    // Go over every allocation and try to fit it in previous blocks at lowest offsets,
    // if not possible: realloc within single block to minimize offset (exclude offset == 0)

    for (size_t i = vector.GetBlockCount() - 1; i > m_ImmovableBlockCount; --i)
    {
        OaVmaDeviceMemoryBlock* block = vector.GetBlock(i);
        OaVmaBlockMetadata* metadata = block->m_pMetadata;

        for (OaVmaAllocHandle handle = metadata->GetAllocationListBegin();
            handle != VK_NULL_HANDLE;
            handle = metadata->GetNextAllocation(handle))
        {
            MoveAllocationData moveData = GetMoveData(handle, metadata);
            // Ignore newly created allocations by defragmentation algorithm
            if (moveData.move.srcAllocation->GetUserData() == this)
                continue;
            switch (CheckCounters(moveData.move.srcAllocation->GetSize()))
            {
            case CounterStatus::Ignore:
                continue;
            case CounterStatus::End:
                return true;
            case CounterStatus::Pass:
                break;
            default:
                OA_VMA_ASSERT(0);
            }

            // Check all previous blocks for free space
            const size_t prevMoveCount = m_Moves.size();
            if (AllocInOtherBlock(0, i, moveData, vector))
                return true;

            // If no room found then realloc within block for lower offset
            VkDeviceSize offset = moveData.move.srcAllocation->GetOffset();
            if (prevMoveCount == m_Moves.size() && offset != 0 && metadata->GetSumFreeSize() >= moveData.size)
            {
                OaVmaAllocationRequest request = {};
                if (metadata->CreateAllocationRequest(
                    moveData.size,
                    moveData.alignment,
                    false,
                    moveData.type,
                    OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT,
                    &request))
                {
                    if (metadata->GetAllocationOffset(request.allocHandle) < offset)
                    {
                        if (vector.CommitAllocationRequest(
                            request,
                            block,
                            moveData.alignment,
                            moveData.flags,
                            this,
                            moveData.type,
                            &moveData.move.dstTmpAllocation) == VK_SUCCESS)
                        {
                            m_Moves.push_back(moveData.move);
                            if (IncrementCounters(moveData.size))
                                return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

bool OaVmaDefragmentationContext_T::ComputeDefragmentation_Extensive(OaVmaBlockVector& vector, size_t index)
{
    // First free single block, then populate it to the brim, then free another block, and so on

    // Fallback to previous algorithm since without granularity conflicts it can achieve max packing
    if (vector.m_BufferImageGranularity == 1)
        return ComputeDefragmentation_Full(vector);

    OA_VMA_ASSERT(m_AlgorithmState != OA_VMA_NULL);

    StateExtensive& vectorState = reinterpret_cast<StateExtensive*>(m_AlgorithmState)[index];

    bool texturePresent = false;
    bool bufferPresent = false;
    bool otherPresent = false;
    switch (vectorState.operation)
    {
    case StateExtensive::Operation::Done: // Vector defragmented
        return false;
    case StateExtensive::Operation::FindFreeBlockBuffer:
    case StateExtensive::Operation::FindFreeBlockTexture:
    case StateExtensive::Operation::FindFreeBlockAll:
    {
        // No more blocks to free, just perform fast realloc and move to cleanup
        if (vectorState.firstFreeBlock == 0)
        {
            vectorState.operation = StateExtensive::Operation::Cleanup;
            return ComputeDefragmentation_Fast(vector);
        }

        // No free blocks, have to clear last one
        size_t last = (vectorState.firstFreeBlock == SIZE_MAX ? vector.GetBlockCount() : vectorState.firstFreeBlock) - 1;
        OaVmaBlockMetadata* freeMetadata = vector.GetBlock(last)->m_pMetadata;

        const size_t prevMoveCount = m_Moves.size();
        for (OaVmaAllocHandle handle = freeMetadata->GetAllocationListBegin();
            handle != VK_NULL_HANDLE;
            handle = freeMetadata->GetNextAllocation(handle))
        {
            MoveAllocationData moveData = GetMoveData(handle, freeMetadata);
            switch (CheckCounters(moveData.move.srcAllocation->GetSize()))
            {
            case CounterStatus::Ignore:
                continue;
            case CounterStatus::End:
                return true;
            case CounterStatus::Pass:
                break;
            default:
                OA_VMA_ASSERT(0);
            }

            // Check all previous blocks for free space
            if (AllocInOtherBlock(0, last, moveData, vector))
            {
                // Full clear performed already
                if (prevMoveCount != m_Moves.size() && freeMetadata->GetNextAllocation(handle) == VK_NULL_HANDLE)
                    vectorState.firstFreeBlock = last;
                return true;
            }
        }

        if (prevMoveCount == m_Moves.size())
        {
            // Cannot perform full clear, have to move data in other blocks around
            if (last != 0)
            {
                for (size_t i = last - 1; i; --i)
                {
                    if (ReallocWithinBlock(vector, vector.GetBlock(i)))
                        return true;
                }
            }

            if (prevMoveCount == m_Moves.size())
            {
                // No possible reallocs within blocks, try to move them around fast
                return ComputeDefragmentation_Fast(vector);
            }
        }
        else
        {
            switch (vectorState.operation)
            {
            case StateExtensive::Operation::FindFreeBlockBuffer:
                vectorState.operation = StateExtensive::Operation::MoveBuffers;
                break;
            case StateExtensive::Operation::FindFreeBlockTexture:
                vectorState.operation = StateExtensive::Operation::MoveTextures;
                break;
            case StateExtensive::Operation::FindFreeBlockAll:
                vectorState.operation = StateExtensive::Operation::MoveAll;
                break;
            default:
                OA_VMA_ASSERT(0);
                vectorState.operation = StateExtensive::Operation::MoveTextures;
            }
            vectorState.firstFreeBlock = last;
            // Nothing done, block found without reallocations, can perform another reallocs in same pass
            return ComputeDefragmentation_Extensive(vector, index);
        }
        break;
    }
    case StateExtensive::Operation::MoveTextures:
    {
        if (MoveDataToFreeBlocks(OA_VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL, vector,
            vectorState.firstFreeBlock, texturePresent, bufferPresent, otherPresent))
        {
            if (texturePresent)
            {
                vectorState.operation = StateExtensive::Operation::FindFreeBlockTexture;
                return ComputeDefragmentation_Extensive(vector, index);
            }

            if (!bufferPresent && !otherPresent)
            {
                vectorState.operation = StateExtensive::Operation::Cleanup;
                break;
            }

            // No more textures to move, check buffers
            vectorState.operation = StateExtensive::Operation::MoveBuffers;
            bufferPresent = false;
            otherPresent = false;
        }
        else
            break;
        OA_VMA_FALLTHROUGH; // Fallthrough
    }
    case StateExtensive::Operation::MoveBuffers:
    {
        if (MoveDataToFreeBlocks(OA_VMA_SUBALLOCATION_TYPE_BUFFER, vector,
            vectorState.firstFreeBlock, texturePresent, bufferPresent, otherPresent))
        {
            if (bufferPresent)
            {
                vectorState.operation = StateExtensive::Operation::FindFreeBlockBuffer;
                return ComputeDefragmentation_Extensive(vector, index);
            }

            if (!otherPresent)
            {
                vectorState.operation = StateExtensive::Operation::Cleanup;
                break;
            }

            // No more buffers to move, check all others
            vectorState.operation = StateExtensive::Operation::MoveAll;
            otherPresent = false;
        }
        else
            break;
        OA_VMA_FALLTHROUGH; // Fallthrough
    }
    case StateExtensive::Operation::MoveAll:
    {
        if (MoveDataToFreeBlocks(OA_VMA_SUBALLOCATION_TYPE_FREE, vector,
            vectorState.firstFreeBlock, texturePresent, bufferPresent, otherPresent))
        {
            if (otherPresent)
            {
                vectorState.operation = StateExtensive::Operation::FindFreeBlockBuffer;
                return ComputeDefragmentation_Extensive(vector, index);
            }
            // Everything moved
            vectorState.operation = StateExtensive::Operation::Cleanup;
        }
        break;
    }
    case StateExtensive::Operation::Cleanup:
        // Cleanup is handled below so that other operations may reuse the cleanup code. This case is here to prevent the unhandled enum value warning (C4062).
        break;
    }

    if (vectorState.operation == StateExtensive::Operation::Cleanup)
    {
        // All other work done, pack data in blocks even tighter if possible
        const size_t prevMoveCount = m_Moves.size();
        for (size_t i = 0; i < vector.GetBlockCount(); ++i)
        {
            if (ReallocWithinBlock(vector, vector.GetBlock(i)))
                return true;
        }

        if (prevMoveCount == m_Moves.size())
            vectorState.operation = StateExtensive::Operation::Done;
    }
    return false;
}

void OaVmaDefragmentationContext_T::UpdateVectorStatistics(OaVmaBlockVector& vector, StateBalanced& state)
{
    size_t allocCount = 0;
    size_t freeCount = 0;
    state.avgFreeSize = 0;
    state.avgAllocSize = 0;

    for (size_t i = 0; i < vector.GetBlockCount(); ++i)
    {
        OaVmaBlockMetadata* metadata = vector.GetBlock(i)->m_pMetadata;

        allocCount += metadata->GetAllocationCount();
        freeCount += metadata->GetFreeRegionsCount();
        state.avgFreeSize += metadata->GetSumFreeSize();
        state.avgAllocSize += metadata->GetSize();
    }

    state.avgAllocSize = (state.avgAllocSize - state.avgFreeSize) / allocCount;
    state.avgFreeSize /= freeCount;
}

bool OaVmaDefragmentationContext_T::MoveDataToFreeBlocks(OaVmaSuballocationType currentType,
    OaVmaBlockVector& vector, size_t firstFreeBlock,
    bool& texturePresent, bool& bufferPresent, bool& otherPresent)
{
    const size_t prevMoveCount = m_Moves.size();
    for (size_t i = firstFreeBlock ; i;)
    {
        OaVmaDeviceMemoryBlock* block = vector.GetBlock(--i);
        OaVmaBlockMetadata* metadata = block->m_pMetadata;

        for (OaVmaAllocHandle handle = metadata->GetAllocationListBegin();
            handle != VK_NULL_HANDLE;
            handle = metadata->GetNextAllocation(handle))
        {
            MoveAllocationData moveData = GetMoveData(handle, metadata);
            // Ignore newly created allocations by defragmentation algorithm
            if (moveData.move.srcAllocation->GetUserData() == this)
                continue;
            switch (CheckCounters(moveData.move.srcAllocation->GetSize()))
            {
            case CounterStatus::Ignore:
                continue;
            case CounterStatus::End:
                return true;
            case CounterStatus::Pass:
                break;
            default:
                OA_VMA_ASSERT(0);
            }

            // Move only single type of resources at once
            if (!OaVmaIsBufferImageGranularityConflict(moveData.type, currentType))
            {
                // Try to fit allocation into free blocks
                if (AllocInOtherBlock(firstFreeBlock, vector.GetBlockCount(), moveData, vector))
                    return false;
            }

            if (!OaVmaIsBufferImageGranularityConflict(moveData.type, OA_VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL))
                texturePresent = true;
            else if (!OaVmaIsBufferImageGranularityConflict(moveData.type, OA_VMA_SUBALLOCATION_TYPE_BUFFER))
                bufferPresent = true;
            else
                otherPresent = true;
        }
    }
    return prevMoveCount == m_Moves.size();
}
#endif // _OA_VMA_DEFRAGMENTATION_CONTEXT_FUNCTIONS

#ifndef _OA_VMA_POOL_T_FUNCTIONS
OaVmaPool_T::OaVmaPool_T(
    OaVmaAllocator hAllocator,
    const OaVmaPoolCreateInfo& createInfo,
    VkDeviceSize preferredBlockSize)
    : m_BlockVector(
        hAllocator,
        this, // hParentPool
        createInfo.memoryTypeIndex,
        createInfo.blockSize != 0 ? createInfo.blockSize : preferredBlockSize,
        createInfo.minBlockCount,
        createInfo.maxBlockCount,
        (createInfo.flags& OA_VMA_POOL_CREATE_IGNORE_BUFFER_IMAGE_GRANULARITY_BIT) != 0 ? 1 : hAllocator->GetBufferImageGranularity(),
        createInfo.blockSize != 0, // explicitBlockSize
        createInfo.flags & OA_VMA_POOL_CREATE_ALGORITHM_MASK, // algorithm
        createInfo.priority,
        OA_VMA_MAX(hAllocator->GetMemoryTypeMinAlignment(createInfo.memoryTypeIndex), createInfo.minAllocationAlignment),
        createInfo.pMemoryAllocateNext),
    m_Id(0),
    m_Name(OA_VMA_NULL) {}

OaVmaPool_T::~OaVmaPool_T()
{
    OA_VMA_ASSERT(m_PrevPool == OA_VMA_NULL && m_NextPool == OA_VMA_NULL);

    const VkAllocationCallbacks* allocs = m_BlockVector.GetAllocator()->GetAllocationCallbacks();
    OaVmaFreeString(allocs, m_Name);
}

void OaVmaPool_T::SetName(const char* pName)
{
    const VkAllocationCallbacks* allocs = m_BlockVector.GetAllocator()->GetAllocationCallbacks();
    OaVmaFreeString(allocs, m_Name);

    if (pName != OA_VMA_NULL)
    {
        m_Name = OaVmaCreateStringCopy(allocs, pName);
    }
    else
    {
        m_Name = OA_VMA_NULL;
    }
}
#endif // _OA_VMA_POOL_T_FUNCTIONS
