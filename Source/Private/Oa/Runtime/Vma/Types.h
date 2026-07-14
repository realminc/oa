// SPDX: MIT. Derived from Vulkan Memory Allocator, (c) 2017-2026 Advanced Micro Devices, Inc.
// Full MIT copyright/permission notice: Source/Public/Oa/Runtime/OaVma.h. See NOTICE.md.
#ifndef _OA_VMA_MAPPING_HYSTERESIS

class OaVmaMappingHysteresis
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaMappingHysteresis)
public:
    OaVmaMappingHysteresis() = default;

    uint32_t GetExtraMapping() const { return m_ExtraMapping; }

    // Call when Map was called.
    // Returns true if switched to extra +1 mapping reference count.
    bool PostMap()
    {
#if OA_VMA_MAPPING_HYSTERESIS_ENABLED
        if(m_ExtraMapping == 0)
        {
            ++m_MajorCounter;
            if(m_MajorCounter >= COUNTER_MIN_EXTRA_MAPPING)
            {
                m_ExtraMapping = 1;
                m_MajorCounter = 0;
                m_MinorCounter = 0;
                return true;
            }
        }
        else // m_ExtraMapping == 1
            PostMinorCounter();
#endif // #if OA_VMA_MAPPING_HYSTERESIS_ENABLED
        return false;
    }

    // Call when Unmap was called.
    void PostUnmap()
    {
#if OA_VMA_MAPPING_HYSTERESIS_ENABLED
        if(m_ExtraMapping == 0)
            ++m_MajorCounter;
        else // m_ExtraMapping == 1
            PostMinorCounter();
#endif // #if OA_VMA_MAPPING_HYSTERESIS_ENABLED
    }

    // Call when allocation was made from the memory block.
    void PostAlloc()
    {
#if OA_VMA_MAPPING_HYSTERESIS_ENABLED
        if(m_ExtraMapping == 1)
            ++m_MajorCounter;
        else // m_ExtraMapping == 0
            PostMinorCounter();
#endif // #if OA_VMA_MAPPING_HYSTERESIS_ENABLED
    }

    // Call when allocation was freed from the memory block.
    // Returns true if switched to extra -1 mapping reference count.
    bool PostFree()
    {
#if OA_VMA_MAPPING_HYSTERESIS_ENABLED
        if(m_ExtraMapping == 1)
        {
            ++m_MajorCounter;
            if(m_MajorCounter >= COUNTER_MIN_EXTRA_MAPPING &&
                m_MajorCounter > m_MinorCounter + 1)
            {
                m_ExtraMapping = 0;
                m_MajorCounter = 0;
                m_MinorCounter = 0;
                return true;
            }
        }
        else // m_ExtraMapping == 0
            PostMinorCounter();
#endif // #if OA_VMA_MAPPING_HYSTERESIS_ENABLED
        return false;
    }

private:
    static constexpr int32_t COUNTER_MIN_EXTRA_MAPPING = 7;

    uint32_t m_MinorCounter = 0;
    uint32_t m_MajorCounter = 0;
    uint32_t m_ExtraMapping = 0; // 0 or 1.

    void PostMinorCounter()
    {
        if(m_MinorCounter < m_MajorCounter)
        {
            ++m_MinorCounter;
        }
        else if(m_MajorCounter > 0)
        {
            --m_MajorCounter;
            --m_MinorCounter;
        }
    }
};

#endif // _OA_VMA_MAPPING_HYSTERESIS

#if OA_VMA_EXTERNAL_MEMORY_WIN32
class OaVmaWin32Handle
{
public:
    OaVmaWin32Handle() noexcept : m_hHandle(OA_VMA_NULL) { }
    explicit OaVmaWin32Handle(HANDLE hHandle) noexcept 
        : m_hHandle(hHandle)
        , m_IsNTHandle(IsNTHandle(hHandle))
    {
    }
    ~OaVmaWin32Handle() noexcept { if (m_hHandle != OA_VMA_NULL && m_IsNTHandle) { ::CloseHandle(m_hHandle); } }
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaWin32Handle)

public:
    // Strengthened
    VkResult GetHandle(VkDevice device, VkDeviceMemory memory, PFN_vkGetMemoryWin32HandleKHR pvkGetMemoryWin32HandleKHR, VkExternalMemoryHandleTypeFlagBits handleType, HANDLE hTargetProcess, bool useMutex, HANDLE* pHandle) noexcept
    {
        *pHandle = OA_VMA_NULL;
        // Try to get handle first.
        VkResult res = VK_SUCCESS;
        if (m_hHandle == OA_VMA_NULL)
        {
            OaVmaMutexLockWrite lock(m_Mutex, useMutex);
            if (m_hHandle == OA_VMA_NULL)
            {
                res = Create(device, memory, pvkGetMemoryWin32HandleKHR, handleType, &m_hHandle);
                if (res != VK_SUCCESS) {
                    m_hHandle = OA_VMA_NULL;
                    return res;
                }
                m_IsNTHandle = IsNTHandle(m_hHandle);
            }
        }
        if (res == VK_SUCCESS) {
            // KMT handle is returned as is.
            *pHandle = m_IsNTHandle ? Duplicate(hTargetProcess) : m_hHandle;
        }
        return res;
    }

    operator bool() const noexcept { return m_hHandle != OA_VMA_NULL; }
private:
    // Not atomic
    static VkResult Create(VkDevice device, VkDeviceMemory memory, PFN_vkGetMemoryWin32HandleKHR pvkGetMemoryWin32HandleKHR, VkExternalMemoryHandleTypeFlagBits handleType, HANDLE* pHandle) noexcept
    {
        VkResult res = VK_ERROR_FEATURE_NOT_PRESENT;
        if (pvkGetMemoryWin32HandleKHR != OA_VMA_NULL)
        {
            VkMemoryGetWin32HandleInfoKHR handleInfo{ };
            handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
            handleInfo.memory = memory;
            handleInfo.handleType = handleType;
            res = pvkGetMemoryWin32HandleKHR(device, &handleInfo, pHandle);
        }
        return res;
    }
    HANDLE Duplicate(HANDLE hTargetProcess = OA_VMA_NULL) const noexcept
    {
        if (!m_hHandle)
            return m_hHandle;

        HANDLE hCurrentProcess = ::GetCurrentProcess();
        HANDLE hDupHandle = OA_VMA_NULL;
        if (!::DuplicateHandle(hCurrentProcess, m_hHandle, hTargetProcess ? hTargetProcess : hCurrentProcess, &hDupHandle, 0, FALSE, DUPLICATE_SAME_ACCESS))
        {
            OA_VMA_ASSERT(0 && "Failed to duplicate handle.");
        }
        return hDupHandle;
    }
    static bool IsNTHandle(HANDLE hHandle) noexcept
    {
        DWORD flags = 0;
        return (hHandle != OA_VMA_NULL) ? (::GetHandleInformation(hHandle, &flags) != 0) : false;
    }
private:
    HANDLE m_hHandle;
    OA_VMA_RW_MUTEX m_Mutex; // Protects access m_Handle
    bool m_IsNTHandle = false; // True if m_Handle is NT handle, false if it's a KMT handle.
};
#else 
class OaVmaWin32Handle
{
    // ABI compatibility
    void* placeholder = OA_VMA_NULL;
    OA_VMA_RW_MUTEX placeholder2;
    bool placeholder3 = false;
};
#endif // OA_VMA_EXTERNAL_MEMORY_WIN32


#ifndef _OA_VMA_DEVICE_MEMORY_BLOCK
/*
Represents a single block of device memory (`VkDeviceMemory`) with all the
data about its regions (aka suballocations, #OaVmaAllocation), assigned and free.

Thread-safety:
- Access to m_pMetadata must be externally synchronized.
- Map, Unmap, Bind* are synchronized internally.
*/
class OaVmaDeviceMemoryBlock
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaDeviceMemoryBlock)
public:
    OaVmaBlockMetadata* m_pMetadata;

    explicit OaVmaDeviceMemoryBlock(OaVmaAllocator hAllocator);
    ~OaVmaDeviceMemoryBlock();

    // Always call after construction.
    void Init(
        OaVmaAllocator hAllocator,
        OaVmaPool hParentPool,
        uint32_t newMemoryTypeIndex,
        VkDeviceMemory newMemory,
        VkDeviceSize newSize,
        uint32_t id,
        uint32_t algorithm,
        VkDeviceSize bufferImageGranularity);
    // Always call before destruction.
    void Destroy(OaVmaAllocator allocator);

    OaVmaPool GetParentPool() const { return m_hParentPool; }
    VkDeviceMemory GetDeviceMemory() const { return m_hMemory; }
    uint32_t GetMemoryTypeIndex() const { return m_MemoryTypeIndex; }
    uint32_t GetId() const { return m_Id; }
    void* GetMappedData() const { return m_pMappedData; }
    uint32_t GetMapRefCount() const { return m_MapCount; }

    // Call when allocation/free was made from m_pMetadata.
    // Used for m_MappingHysteresis.
    void PostAlloc(OaVmaAllocator hAllocator);
    void PostFree(OaVmaAllocator hAllocator);

    // Validates all data structures inside this object. If not valid, returns false.
    bool Validate() const;
    VkResult CheckCorruption(OaVmaAllocator hAllocator);

    // ppData can be null.
    VkResult Map(OaVmaAllocator hAllocator, uint32_t count, void** ppData);
    void Unmap(OaVmaAllocator hAllocator, uint32_t count);

    VkResult WriteMagicValueAfterAllocation(OaVmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize);
    VkResult ValidateMagicValueAfterAllocation(OaVmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize);

    VkResult BindBufferMemory(
        OaVmaAllocator hAllocator,
        OaVmaAllocation hAllocation,
        VkDeviceSize allocationLocalOffset,
        VkBuffer hBuffer,
        const void* pNext);
    VkResult BindImageMemory(
        OaVmaAllocator hAllocator,
        OaVmaAllocation hAllocation,
        VkDeviceSize allocationLocalOffset,
        VkImage hImage,
        const void* pNext);
#if OA_VMA_EXTERNAL_MEMORY_WIN32
    VkResult CreateWin32Handle(
        const OaVmaAllocator hAllocator,
        PFN_vkGetMemoryWin32HandleKHR pvkGetMemoryWin32HandleKHR,
        VkExternalMemoryHandleTypeFlagBits handleType,
        HANDLE hTargetProcess,
        HANDLE* pHandle)noexcept;
#endif // OA_VMA_EXTERNAL_MEMORY_WIN32
private:
    OaVmaPool m_hParentPool; // VK_NULL_HANDLE if not belongs to custom pool.
    uint32_t m_MemoryTypeIndex;
    uint32_t m_Id;
    VkDeviceMemory m_hMemory;

    /*
    Protects access to m_hMemory so it is not used by multiple threads simultaneously, e.g. vkMapMemory, vkBindBufferMemory.
    Also protects m_MapCount, m_pMappedData.
    Allocations, deallocations, any change in m_pMetadata is protected by parent's OaVmaBlockVector::m_Mutex.
    */
    OA_VMA_MUTEX m_MapAndBindMutex;
    OaVmaMappingHysteresis m_MappingHysteresis;
    uint32_t m_MapCount;
    void* m_pMappedData;

    OaVmaWin32Handle m_Handle;
};
#endif // _OA_VMA_DEVICE_MEMORY_BLOCK

#ifndef _OA_VMA_ALLOCATION_T
struct OaVmaAllocationExtraData
{
    void* m_pMappedData = OA_VMA_NULL; // Not null means memory is mapped.
    OaVmaWin32Handle m_Handle;
};

struct OaVmaAllocation_T
{
    friend struct OaVmaDedicatedAllocationListItemTraits;

    enum FLAGS
    {
        FLAG_PERSISTENT_MAP   = 0x01,
        FLAG_MAPPING_ALLOWED  = 0x02,
    };

public:
    enum ALLOCATION_TYPE
    {
        ALLOCATION_TYPE_NONE,
        ALLOCATION_TYPE_BLOCK,
        ALLOCATION_TYPE_DEDICATED,
    };

    // This struct is allocated using OaVmaPoolAllocator.
    explicit OaVmaAllocation_T(bool mappingAllowed);
    ~OaVmaAllocation_T();

    void InitBlockAllocation(
        OaVmaDeviceMemoryBlock* block,
        OaVmaAllocHandle allocHandle,
        VkDeviceSize alignment,
        VkDeviceSize size,
        uint32_t memoryTypeIndex,
        OaVmaSuballocationType suballocationType,
        bool mapped);
    // pMappedData not null means allocation is created with MAPPED flag.
    void InitDedicatedAllocation(
        OaVmaAllocator allocator,
        OaVmaPool hParentPool,
        uint32_t memoryTypeIndex,
        VkDeviceMemory hMemory,
        OaVmaSuballocationType suballocationType,
        void* pMappedData,
        VkDeviceSize size);
    void Destroy(OaVmaAllocator allocator);

    ALLOCATION_TYPE GetType() const { return (ALLOCATION_TYPE)m_Type; }
    VkDeviceSize GetAlignment() const { return m_Alignment; }
    VkDeviceSize GetSize() const { return m_Size; }
    void* GetUserData() const { return m_pUserData; }
    const char* GetName() const { return m_pName; }
    OaVmaSuballocationType GetSuballocationType() const { return (OaVmaSuballocationType)m_SuballocationType; }

    OaVmaDeviceMemoryBlock* GetBlock() const { OA_VMA_ASSERT(m_Type == ALLOCATION_TYPE_BLOCK); return m_BlockAllocation.m_Block; }
    uint32_t GetMemoryTypeIndex() const { return m_MemoryTypeIndex; }
    bool IsPersistentMap() const { return (m_Flags & FLAG_PERSISTENT_MAP) != 0; }
    bool IsMappingAllowed() const { return (m_Flags & FLAG_MAPPING_ALLOWED) != 0; }

    void SetUserData(OaVmaAllocator hAllocator, void* pUserData) { m_pUserData = pUserData; }
    void SetName(OaVmaAllocator hAllocator, const char* pName);
    void FreeName(OaVmaAllocator hAllocator);
    uint8_t SwapBlockAllocation(OaVmaAllocator hAllocator, OaVmaAllocation allocation);
    OaVmaAllocHandle GetAllocHandle() const;
    VkDeviceSize GetOffset() const;
    OaVmaPool GetParentPool() const;
    VkDeviceMemory GetMemory() const;
    void* GetMappedData() const;

    void BlockAllocMap();
    void BlockAllocUnmap();
    VkResult DedicatedAllocMap(OaVmaAllocator hAllocator, void** ppData);
    void DedicatedAllocUnmap(OaVmaAllocator hAllocator);

#if OA_VMA_STATS_STRING_ENABLED
    OaVmaBufferImageUsage GetBufferImageUsage() const { return m_BufferImageUsage; }
    void InitBufferUsage(const VkBufferCreateInfo &createInfo, bool useKhrMaintenance5)
    {
        OA_VMA_ASSERT(m_BufferImageUsage == OaVmaBufferImageUsage::UNKNOWN);
        m_BufferImageUsage = OaVmaBufferImageUsage(createInfo, useKhrMaintenance5);
    }
    void InitImageUsage(const VkImageCreateInfo &createInfo)
    {
        OA_VMA_ASSERT(m_BufferImageUsage == OaVmaBufferImageUsage::UNKNOWN);
        m_BufferImageUsage = OaVmaBufferImageUsage(createInfo);
    }
    void PrintParameters(class OaVmaJsonWriter& json) const;
#endif

#if OA_VMA_EXTERNAL_MEMORY_WIN32
    VkResult GetWin32Handle(OaVmaAllocator hAllocator, VkExternalMemoryHandleTypeFlagBits handleType, HANDLE hTargetProcess, HANDLE* hHandle) noexcept;
#endif // OA_VMA_EXTERNAL_MEMORY_WIN32

private:
    // Allocation out of OaVmaDeviceMemoryBlock.
    struct BlockAllocation
    {
        OaVmaDeviceMemoryBlock* m_Block;
        OaVmaAllocHandle m_AllocHandle;
    };
    // Allocation for an object that has its own private VkDeviceMemory.
    struct DedicatedAllocation
    {
        OaVmaPool m_hParentPool; // VK_NULL_HANDLE if not belongs to custom pool.
        VkDeviceMemory m_hMemory;
        OaVmaAllocationExtraData* m_ExtraData;
        OaVmaAllocation_T* m_Prev;
        OaVmaAllocation_T* m_Next;
    };
    union
    {
        // Allocation out of OaVmaDeviceMemoryBlock.
        BlockAllocation m_BlockAllocation;
        // Allocation for an object that has its own private VkDeviceMemory.
        DedicatedAllocation m_DedicatedAllocation;
    };

    VkDeviceSize m_Alignment;
    VkDeviceSize m_Size;
    void* m_pUserData;
    char* m_pName;
    uint32_t m_MemoryTypeIndex;
    uint8_t m_Type; // ALLOCATION_TYPE
    uint8_t m_SuballocationType; // OaVmaSuballocationType
    // Reference counter for OaVmaMapMemory()/OaVmaUnmapMemory().
    uint8_t m_MapCount;
    uint8_t m_Flags; // enum FLAGS
#if OA_VMA_STATS_STRING_ENABLED
    OaVmaBufferImageUsage m_BufferImageUsage; // 0 if unknown.
#endif

    void EnsureExtraData(OaVmaAllocator hAllocator);
};
#endif // _OA_VMA_ALLOCATION_T

#ifndef _OA_VMA_DEDICATED_ALLOCATION_LIST_ITEM_TRAITS
struct OaVmaDedicatedAllocationListItemTraits
{
    typedef OaVmaAllocation_T ItemType;

    static ItemType* GetPrev(const ItemType* item)
    {
        OA_VMA_HEAVY_ASSERT(item->GetType() == OaVmaAllocation_T::ALLOCATION_TYPE_DEDICATED);
        return item->m_DedicatedAllocation.m_Prev;
    }
    static ItemType* GetNext(const ItemType* item)
    {
        OA_VMA_HEAVY_ASSERT(item->GetType() == OaVmaAllocation_T::ALLOCATION_TYPE_DEDICATED);
        return item->m_DedicatedAllocation.m_Next;
    }
    static ItemType*& AccessPrev(ItemType* item)
    {
        OA_VMA_HEAVY_ASSERT(item->GetType() == OaVmaAllocation_T::ALLOCATION_TYPE_DEDICATED);
        return item->m_DedicatedAllocation.m_Prev;
    }
    static ItemType*& AccessNext(ItemType* item)
    {
        OA_VMA_HEAVY_ASSERT(item->GetType() == OaVmaAllocation_T::ALLOCATION_TYPE_DEDICATED);
        return item->m_DedicatedAllocation.m_Next;
    }
};
#endif // _OA_VMA_DEDICATED_ALLOCATION_LIST_ITEM_TRAITS

#ifndef _OA_VMA_DEDICATED_ALLOCATION_LIST
/*
Stores linked list of OaVmaAllocation_T objects.
Thread-safe, synchronized internally.
*/
class OaVmaDedicatedAllocationList
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaDedicatedAllocationList)
public:
    OaVmaDedicatedAllocationList() = default;
    ~OaVmaDedicatedAllocationList();

    void Init(bool useMutex) { m_UseMutex = useMutex; }
    bool Validate();

    void AddDetailedStatistics(OaVmaDetailedStatistics& inoutStats);
    void AddStatistics(OaVmaStatistics& inoutStats);
#if OA_VMA_STATS_STRING_ENABLED
    // Writes JSON array with the list of allocations.
    void BuildStatsString(OaVmaJsonWriter& json);
#endif

    bool IsEmpty();
    void Register(OaVmaAllocation alloc);
    void Unregister(OaVmaAllocation alloc);

private:
    typedef OaVmaIntrusiveLinkedList<OaVmaDedicatedAllocationListItemTraits> DedicatedAllocationLinkedList;

    bool m_UseMutex = true;
    OA_VMA_RW_MUTEX m_Mutex;
    DedicatedAllocationLinkedList m_AllocationList;
};

#ifndef _OA_VMA_DEDICATED_ALLOCATION_LIST_FUNCTIONS

OaVmaDedicatedAllocationList::~OaVmaDedicatedAllocationList()
{
    OA_VMA_HEAVY_ASSERT(Validate());

    if (!m_AllocationList.IsEmpty())
    {
        OA_VMA_ASSERT_LEAK(false && "Unfreed dedicated allocations found!");
    }
}

bool OaVmaDedicatedAllocationList::Validate()
{
    const size_t declaredCount = m_AllocationList.GetCount();
    size_t actualCount = 0;
    OaVmaMutexLockRead lock(m_Mutex, m_UseMutex);
    for (OaVmaAllocation alloc = m_AllocationList.Front();
        alloc != OA_VMA_NULL; alloc = m_AllocationList.GetNext(alloc))
    {
        ++actualCount;
    }
    OA_VMA_VALIDATE(actualCount == declaredCount);

    return true;
}

void OaVmaDedicatedAllocationList::AddDetailedStatistics(OaVmaDetailedStatistics& inoutStats)
{
    for(auto* item = m_AllocationList.Front(); item != OA_VMA_NULL; item = DedicatedAllocationLinkedList::GetNext(item))
    {
        const VkDeviceSize size = item->GetSize();
        inoutStats.statistics.blockCount++;
        inoutStats.statistics.blockBytes += size;
        OaVmaAddDetailedStatisticsAllocation(inoutStats, item->GetSize());
    }
}

void OaVmaDedicatedAllocationList::AddStatistics(OaVmaStatistics& inoutStats)
{
    OaVmaMutexLockRead lock(m_Mutex, m_UseMutex);

    const uint32_t allocCount = (uint32_t)m_AllocationList.GetCount();
    inoutStats.blockCount += allocCount;
    inoutStats.allocationCount += allocCount;

    for(auto* item = m_AllocationList.Front(); item != OA_VMA_NULL; item = DedicatedAllocationLinkedList::GetNext(item))
    {
        const VkDeviceSize size = item->GetSize();
        inoutStats.blockBytes += size;
        inoutStats.allocationBytes += size;
    }
}

#if OA_VMA_STATS_STRING_ENABLED
void OaVmaDedicatedAllocationList::BuildStatsString(OaVmaJsonWriter& json)
{
    OaVmaMutexLockRead lock(m_Mutex, m_UseMutex);
    json.BeginArray();
    for (OaVmaAllocation alloc = m_AllocationList.Front();
        alloc != OA_VMA_NULL; alloc = m_AllocationList.GetNext(alloc))
    {
        json.BeginObject(true);
        alloc->PrintParameters(json);
        json.EndObject();
    }
    json.EndArray();
}
#endif // OA_VMA_STATS_STRING_ENABLED

bool OaVmaDedicatedAllocationList::IsEmpty()
{
    OaVmaMutexLockRead lock(m_Mutex, m_UseMutex);
    return m_AllocationList.IsEmpty();
}

void OaVmaDedicatedAllocationList::Register(OaVmaAllocation alloc)
{
    OaVmaMutexLockWrite lock(m_Mutex, m_UseMutex);
    m_AllocationList.PushBack(alloc);
}

void OaVmaDedicatedAllocationList::Unregister(OaVmaAllocation alloc)
{
    OaVmaMutexLockWrite lock(m_Mutex, m_UseMutex);
    m_AllocationList.Remove(alloc);
}
#endif // _OA_VMA_DEDICATED_ALLOCATION_LIST_FUNCTIONS
#endif // _OA_VMA_DEDICATED_ALLOCATION_LIST

#ifndef _OA_VMA_SUBALLOCATION
/*
Represents a region of OaVmaDeviceMemoryBlock that is either assigned and returned as
allocated memory block or free.
*/
struct OaVmaSuballocation
{
    VkDeviceSize offset;
    VkDeviceSize size;
    void* userData;
    OaVmaSuballocationType type;
};

// Comparator for offsets.
struct OaVmaSuballocationOffsetLess
{
    bool operator()(const OaVmaSuballocation& lhs, const OaVmaSuballocation& rhs) const
    {
        return lhs.offset < rhs.offset;
    }
};

struct OaVmaSuballocationOffsetGreater
{
    bool operator()(const OaVmaSuballocation& lhs, const OaVmaSuballocation& rhs) const
    {
        return lhs.offset > rhs.offset;
    }
};

struct OaVmaSuballocationItemSizeLess
{
    bool operator()(const OaVmaSuballocationList::iterator lhs,
        const OaVmaSuballocationList::iterator rhs) const
    {
        return lhs->size < rhs->size;
    }

    bool operator()(const OaVmaSuballocationList::iterator lhs,
        VkDeviceSize rhsSize) const
    {
        return lhs->size < rhsSize;
    }
};
#endif // _OA_VMA_SUBALLOCATION

#ifndef _OA_VMA_ALLOCATION_REQUEST
/*
Parameters of planned allocation inside a OaVmaDeviceMemoryBlock.
item points to a FREE suballocation.
*/
struct OaVmaAllocationRequest
{
    OaVmaAllocHandle allocHandle;
    VkDeviceSize size;
    OaVmaSuballocationList::iterator item;
    void* customData;
    uint64_t algorithmData;
    OaVmaAllocationRequestType type;
};
#endif // _OA_VMA_ALLOCATION_REQUEST
