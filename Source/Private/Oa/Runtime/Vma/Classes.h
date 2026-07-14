// SPDX: MIT. Derived from Vulkan Memory Allocator, (c) 2017-2026 Advanced Micro Devices, Inc.
// Full MIT copyright/permission notice: Source/Public/Oa/Runtime/OaVma.h. See NOTICE.md.
#ifndef _OA_VMA_BLOCK_VECTOR
/*
Sequence of OaVmaDeviceMemoryBlock. Represents memory blocks allocated for a specific
Vulkan memory type.

Synchronized internally with a mutex.
*/
class OaVmaBlockVector
{
    friend struct OaVmaDefragmentationContext_T;
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaBlockVector)
public:
    OaVmaBlockVector(
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
        void* pMemoryAllocateNext);
    ~OaVmaBlockVector();

    OaVmaAllocator GetAllocator() const { return m_hAllocator; }
    OaVmaPool GetParentPool() const { return m_hParentPool; }
    bool IsCustomPool() const { return m_hParentPool != OA_VMA_NULL; }
    uint32_t GetMemoryTypeIndex() const { return m_MemoryTypeIndex; }
    VkDeviceSize GetPreferredBlockSize() const { return m_PreferredBlockSize; }
    VkDeviceSize GetBufferImageGranularity() const { return m_BufferImageGranularity; }
    uint32_t GetAlgorithm() const { return m_Algorithm; }
    bool HasExplicitBlockSize() const { return m_ExplicitBlockSize; }
    float GetPriority() const { return m_Priority; }
    const void* GetAllocationNextPtr() const { return m_pMemoryAllocateNext; }
    // To be used only while the m_Mutex is locked. Used during defragmentation.
    size_t GetBlockCount() const { return m_Blocks.size(); }
    // To be used only while the m_Mutex is locked. Used during defragmentation.
    OaVmaDeviceMemoryBlock* GetBlock(size_t index) const { return m_Blocks[index]; }
    OA_VMA_RW_MUTEX &GetMutex() { return m_Mutex; }

    VkResult CreateMinBlocks();
    void AddStatistics(OaVmaStatistics& inoutStats);
    void AddDetailedStatistics(OaVmaDetailedStatistics& inoutStats);
    bool IsEmpty();
    bool IsCorruptionDetectionEnabled() const;

    VkResult Allocate(
        VkDeviceSize size,
        VkDeviceSize alignment,
        const OaVmaAllocationCreateInfo& createInfo,
        OaVmaSuballocationType suballocType,
        size_t allocationCount,
        OaVmaAllocation* pAllocations);

    void Free(OaVmaAllocation hAllocation);

#if OA_VMA_STATS_STRING_ENABLED
    void PrintDetailedMap(class OaVmaJsonWriter& json);
#endif

    VkResult CheckCorruption();

private:
    const OaVmaAllocator m_hAllocator;
    const OaVmaPool m_hParentPool;
    const uint32_t m_MemoryTypeIndex;
    const VkDeviceSize m_PreferredBlockSize;
    const size_t m_MinBlockCount;
    const size_t m_MaxBlockCount;
    const VkDeviceSize m_BufferImageGranularity;
    const bool m_ExplicitBlockSize;
    const uint32_t m_Algorithm;
    const float m_Priority;
    const VkDeviceSize m_MinAllocationAlignment;

    void* const m_pMemoryAllocateNext;
    OA_VMA_RW_MUTEX m_Mutex;
    // Incrementally sorted by sumFreeSize, ascending.
    OaVmaVector<OaVmaDeviceMemoryBlock*, OaVmaStlAllocator<OaVmaDeviceMemoryBlock*>> m_Blocks;
    uint32_t m_NextBlockId;
    bool m_IncrementalSort = true;

    void SetIncrementalSort(bool val) { m_IncrementalSort = val; }

    VkDeviceSize CalcMaxBlockSize() const;
    // Finds and removes given block from vector.
    void Remove(OaVmaDeviceMemoryBlock* pBlock);
    // Performs single step in sorting m_Blocks. They may not be fully sorted
    // after this call.
    void IncrementallySortBlocks();
    void SortByFreeSize();

    VkResult AllocatePage(
        VkDeviceSize size,
        VkDeviceSize alignment,
        const OaVmaAllocationCreateInfo& createInfo,
        OaVmaSuballocationType suballocType,
        OaVmaAllocation* pAllocation);

    VkResult AllocateFromBlock(
        OaVmaDeviceMemoryBlock* pBlock,
        VkDeviceSize size,
        VkDeviceSize alignment,
        OaVmaAllocationCreateFlags allocFlags,
        void* pUserData,
        OaVmaSuballocationType suballocType,
        uint32_t strategy,
        OaVmaAllocation* pAllocation);

    VkResult CommitAllocationRequest(
        OaVmaAllocationRequest& allocRequest,
        OaVmaDeviceMemoryBlock* pBlock,
        VkDeviceSize alignment,
        OaVmaAllocationCreateFlags allocFlags,
        void* pUserData,
        OaVmaSuballocationType suballocType,
        OaVmaAllocation* pAllocation);

    VkResult CreateBlock(VkDeviceSize blockSize, size_t* pNewBlockIndex);
    bool HasEmptyBlock();
};
#endif // _OA_VMA_BLOCK_VECTOR

#ifndef _OA_VMA_DEFRAGMENTATION_CONTEXT
struct OaVmaDefragmentationContext_T
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaDefragmentationContext_T)
public:
    OaVmaDefragmentationContext_T(
        OaVmaAllocator hAllocator,
        const OaVmaDefragmentationInfo& info);
    ~OaVmaDefragmentationContext_T();

    void GetStats(OaVmaDefragmentationStats& outStats) { outStats = m_GlobalStats; }

    VkResult DefragmentPassBegin(OaVmaDefragmentationPassMoveInfo& moveInfo);
    VkResult DefragmentPassEnd(OaVmaDefragmentationPassMoveInfo& moveInfo);

private:
    // Max number of allocations to ignore due to size constraints before ending single pass
    static constexpr uint8_t MAX_ALLOCS_TO_IGNORE = 16;
    enum class CounterStatus { Pass, Ignore, End };

    struct FragmentedBlock
    {
        uint32_t data;
        OaVmaDeviceMemoryBlock* block;
    };
    struct StateBalanced
    {
        VkDeviceSize avgFreeSize = 0;
        VkDeviceSize avgAllocSize = UINT64_MAX;
    };
    struct StateExtensive
    {
        enum class Operation : uint8_t
        {
            FindFreeBlockBuffer, FindFreeBlockTexture, FindFreeBlockAll,
            MoveBuffers, MoveTextures, MoveAll,
            Cleanup, Done
        };

        Operation operation = Operation::FindFreeBlockTexture;
        size_t firstFreeBlock = SIZE_MAX;
    };
    struct MoveAllocationData
    {
        VkDeviceSize size;
        VkDeviceSize alignment;
        OaVmaSuballocationType type;
        OaVmaAllocationCreateFlags flags;
        OaVmaDefragmentationMove move = {};
    };

    const VkDeviceSize m_MaxPassBytes;
    const uint32_t m_MaxPassAllocations;
    const PFN_OaVmaCheckDefragmentationBreakFunction m_BreakCallback;
    void* m_BreakCallbackUserData;

    OaVmaStlAllocator<OaVmaDefragmentationMove> m_MoveAllocator;
    OaVmaVector<OaVmaDefragmentationMove, OaVmaStlAllocator<OaVmaDefragmentationMove>> m_Moves;

    uint8_t m_IgnoredAllocs = 0;
    uint32_t m_Algorithm;
    uint32_t m_BlockVectorCount;
    OaVmaBlockVector* m_PoolBlockVector;
    OaVmaBlockVector** m_pBlockVectors;
    size_t m_ImmovableBlockCount = 0;
    OaVmaDefragmentationStats m_GlobalStats = { 0 };
    OaVmaDefragmentationStats m_PassStats = { 0 };
    void* m_AlgorithmState = OA_VMA_NULL;

    static MoveAllocationData GetMoveData(OaVmaAllocHandle handle, OaVmaBlockMetadata* metadata);
    CounterStatus CheckCounters(VkDeviceSize bytes);
    bool IncrementCounters(VkDeviceSize bytes);
    bool ReallocWithinBlock(OaVmaBlockVector& vector, OaVmaDeviceMemoryBlock* block);
    bool AllocInOtherBlock(size_t start, size_t end, MoveAllocationData& data, OaVmaBlockVector& vector);

    bool ComputeDefragmentation(OaVmaBlockVector& vector, size_t index);
    bool ComputeDefragmentation_Fast(OaVmaBlockVector& vector);
    bool ComputeDefragmentation_Balanced(OaVmaBlockVector& vector, size_t index, bool update);
    bool ComputeDefragmentation_Full(OaVmaBlockVector& vector);
    bool ComputeDefragmentation_Extensive(OaVmaBlockVector& vector, size_t index);

    static void UpdateVectorStatistics(OaVmaBlockVector& vector, StateBalanced& state);
    bool MoveDataToFreeBlocks(OaVmaSuballocationType currentType,
        OaVmaBlockVector& vector, size_t firstFreeBlock,
        bool& texturePresent, bool& bufferPresent, bool& otherPresent);
};
#endif // _OA_VMA_DEFRAGMENTATION_CONTEXT

#ifndef _OA_VMA_POOL_T
struct OaVmaPool_T
{
    friend struct OaVmaPoolListItemTraits;
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaPool_T)
public:
    OaVmaBlockVector m_BlockVector;
    OaVmaDedicatedAllocationList m_DedicatedAllocations;

    OaVmaPool_T(
        OaVmaAllocator hAllocator,
        const OaVmaPoolCreateInfo& createInfo,
        VkDeviceSize preferredBlockSize);
    ~OaVmaPool_T();

    uint32_t GetId() const { return m_Id; }
    void SetId(uint32_t id) { OA_VMA_ASSERT(m_Id == 0); m_Id = id; }

    const char* GetName() const { return m_Name; }
    void SetName(const char* pName);

#if OA_VMA_STATS_STRING_ENABLED
    //void PrintDetailedMap(class OaVmaStringBuilder& sb);
#endif

private:
    uint32_t m_Id;
    char* m_Name;
    OaVmaPool_T* m_PrevPool = OA_VMA_NULL;
    OaVmaPool_T* m_NextPool = OA_VMA_NULL;
};

struct OaVmaPoolListItemTraits
{
    typedef OaVmaPool_T ItemType;

    static ItemType* GetPrev(const ItemType* item) { return item->m_PrevPool; }
    static ItemType* GetNext(const ItemType* item) { return item->m_NextPool; }
    static ItemType*& AccessPrev(ItemType* item) { return item->m_PrevPool; }
    static ItemType*& AccessNext(ItemType* item) { return item->m_NextPool; }
};
#endif // _OA_VMA_POOL_T

#ifndef _OA_VMA_CURRENT_BUDGET_DATA
struct OaVmaCurrentBudgetData
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaCurrentBudgetData)
public:

    OA_VMA_ATOMIC_UINT32 m_BlockCount[VK_MAX_MEMORY_HEAPS];
    OA_VMA_ATOMIC_UINT32 m_AllocationCount[VK_MAX_MEMORY_HEAPS];
    OA_VMA_ATOMIC_UINT64 m_BlockBytes[VK_MAX_MEMORY_HEAPS];
    OA_VMA_ATOMIC_UINT64 m_AllocationBytes[VK_MAX_MEMORY_HEAPS];

#if OA_VMA_MEMORY_BUDGET
    OA_VMA_ATOMIC_UINT32 m_OperationsSinceBudgetFetch;
    OA_VMA_RW_MUTEX m_BudgetMutex;
    uint64_t m_VulkanUsage[VK_MAX_MEMORY_HEAPS];
    uint64_t m_VulkanBudget[VK_MAX_MEMORY_HEAPS];
    uint64_t m_BlockBytesAtBudgetFetch[VK_MAX_MEMORY_HEAPS];
#endif // OA_VMA_MEMORY_BUDGET

    OaVmaCurrentBudgetData();

    void AddAllocation(uint32_t heapIndex, VkDeviceSize allocationSize);
    void RemoveAllocation(uint32_t heapIndex, VkDeviceSize allocationSize);
};

#ifndef _OA_VMA_CURRENT_BUDGET_DATA_FUNCTIONS
OaVmaCurrentBudgetData::OaVmaCurrentBudgetData()
{
    for (uint32_t heapIndex = 0; heapIndex < VK_MAX_MEMORY_HEAPS; ++heapIndex)
    {
        m_BlockCount[heapIndex] = 0;
        m_AllocationCount[heapIndex] = 0;
        m_BlockBytes[heapIndex] = 0;
        m_AllocationBytes[heapIndex] = 0;
#if OA_VMA_MEMORY_BUDGET
        m_VulkanUsage[heapIndex] = 0;
        m_VulkanBudget[heapIndex] = 0;
        m_BlockBytesAtBudgetFetch[heapIndex] = 0;
#endif
    }

#if OA_VMA_MEMORY_BUDGET
    m_OperationsSinceBudgetFetch = 0;
#endif
}

void OaVmaCurrentBudgetData::AddAllocation(uint32_t heapIndex, VkDeviceSize allocationSize)
{
    m_AllocationBytes[heapIndex] += allocationSize;
    ++m_AllocationCount[heapIndex];
#if OA_VMA_MEMORY_BUDGET
    ++m_OperationsSinceBudgetFetch;
#endif
}

void OaVmaCurrentBudgetData::RemoveAllocation(uint32_t heapIndex, VkDeviceSize allocationSize)
{
    OA_VMA_ASSERT(m_AllocationBytes[heapIndex] >= allocationSize);
    m_AllocationBytes[heapIndex] -= allocationSize;
    OA_VMA_ASSERT(m_AllocationCount[heapIndex] > 0);
    --m_AllocationCount[heapIndex];
#if OA_VMA_MEMORY_BUDGET
    ++m_OperationsSinceBudgetFetch;
#endif
}
#endif // _OA_VMA_CURRENT_BUDGET_DATA_FUNCTIONS
#endif // _OA_VMA_CURRENT_BUDGET_DATA

#ifndef _OA_VMA_ALLOCATION_OBJECT_ALLOCATOR
/*
Thread-safe wrapper over OaVmaPoolAllocator free list, for allocation of OaVmaAllocation_T objects.
*/
class OaVmaAllocationObjectAllocator
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaAllocationObjectAllocator)
public:
    explicit OaVmaAllocationObjectAllocator(const VkAllocationCallbacks* pAllocationCallbacks)
        : m_Allocator(pAllocationCallbacks, 1024) {}

    template<typename... Types> OaVmaAllocation Allocate(Types&&... args);
    void Free(OaVmaAllocation hAlloc);

private:
    OA_VMA_MUTEX m_Mutex;
    OaVmaPoolAllocator<OaVmaAllocation_T> m_Allocator;
};

template<typename... Types>
OaVmaAllocation OaVmaAllocationObjectAllocator::Allocate(Types&&... args)
{
    OaVmaMutexLock mutexLock(m_Mutex);
    return m_Allocator.Alloc<Types...>(std::forward<Types>(args)...);
}

void OaVmaAllocationObjectAllocator::Free(OaVmaAllocation hAlloc)
{
    OaVmaMutexLock mutexLock(m_Mutex);
    m_Allocator.Free(hAlloc);
}
#endif // _OA_VMA_ALLOCATION_OBJECT_ALLOCATOR

#ifndef _OA_VMA_VIRTUAL_BLOCK_T
struct OaVmaVirtualBlock_T
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaVirtualBlock_T)
public:
    const bool m_AllocationCallbacksSpecified;
    const VkAllocationCallbacks m_AllocationCallbacks;

    explicit OaVmaVirtualBlock_T(const OaVmaVirtualBlockCreateInfo& createInfo);
    ~OaVmaVirtualBlock_T();

    bool IsEmpty() const { return m_Metadata->IsEmpty(); }
    void Free(OaVmaVirtualAllocation allocation) { m_Metadata->Free((OaVmaAllocHandle)allocation); }
    void SetAllocationUserData(OaVmaVirtualAllocation allocation, void* userData) { m_Metadata->SetAllocationUserData((OaVmaAllocHandle)allocation, userData); }
    void Clear() { m_Metadata->Clear(); }

    const VkAllocationCallbacks* GetAllocationCallbacks() const;
    void GetAllocationInfo(OaVmaVirtualAllocation allocation, OaVmaVirtualAllocationInfo& outInfo);
    VkResult Allocate(const OaVmaVirtualAllocationCreateInfo& createInfo, OaVmaVirtualAllocation& outAllocation,
        VkDeviceSize* outOffset);
    void GetStatistics(OaVmaStatistics& outStats) const;
    void CalculateDetailedStatistics(OaVmaDetailedStatistics& outStats) const;
#if OA_VMA_STATS_STRING_ENABLED
    void BuildStatsString(bool detailedMap, OaVmaStringBuilder& sb) const;
#endif

private:
    OaVmaBlockMetadata* m_Metadata;
};

#ifndef _OA_VMA_VIRTUAL_BLOCK_T_FUNCTIONS
OaVmaVirtualBlock_T::OaVmaVirtualBlock_T(const OaVmaVirtualBlockCreateInfo& createInfo)
    : m_AllocationCallbacksSpecified(createInfo.pAllocationCallbacks != OA_VMA_NULL),
    m_AllocationCallbacks(createInfo.pAllocationCallbacks != OA_VMA_NULL ? *createInfo.pAllocationCallbacks : OaVmaEmptyAllocationCallbacks)
{
    const uint32_t algorithm = createInfo.flags & OA_VMA_VIRTUAL_BLOCK_CREATE_ALGORITHM_MASK;
    switch (algorithm)
    {
    case 0:
        m_Metadata = OaVma_new(GetAllocationCallbacks(), OaVmaBlockMetadata_TLSF)(VK_NULL_HANDLE, 1, true);
        break;
    case OA_VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT:
        m_Metadata = OaVma_new(GetAllocationCallbacks(), OaVmaBlockMetadata_Linear)(VK_NULL_HANDLE, 1, true);
        break;
    default:
        OA_VMA_ASSERT(0);
        m_Metadata = OaVma_new(GetAllocationCallbacks(), OaVmaBlockMetadata_TLSF)(VK_NULL_HANDLE, 1, true);
    }

    m_Metadata->Init(createInfo.size);
}

OaVmaVirtualBlock_T::~OaVmaVirtualBlock_T()
{
    // Define macro OA_VMA_DEBUG_LOG_FORMAT or more specialized OA_VMA_LEAK_LOG_FORMAT
    // to receive the list of the unfreed allocations.
    if (!m_Metadata->IsEmpty())
        m_Metadata->DebugLogAllAllocations();
    // This is the most important assert in the entire library.
    // Hitting it means you have some memory leak - unreleased virtual allocations.
    OA_VMA_ASSERT_LEAK(m_Metadata->IsEmpty() && "Some virtual allocations were not freed before destruction of this virtual block!");

    OaVma_delete(GetAllocationCallbacks(), m_Metadata);
}

const VkAllocationCallbacks* OaVmaVirtualBlock_T::GetAllocationCallbacks() const
{
    return m_AllocationCallbacksSpecified ? &m_AllocationCallbacks : OA_VMA_NULL;
}

void OaVmaVirtualBlock_T::GetAllocationInfo(OaVmaVirtualAllocation allocation, OaVmaVirtualAllocationInfo& outInfo)
{
    m_Metadata->GetAllocationInfo((OaVmaAllocHandle)allocation, outInfo);
}

VkResult OaVmaVirtualBlock_T::Allocate(const OaVmaVirtualAllocationCreateInfo& createInfo, OaVmaVirtualAllocation& outAllocation,
    VkDeviceSize* outOffset)
{
    OaVmaAllocationRequest request = {};
    if (m_Metadata->CreateAllocationRequest(
        createInfo.size, // allocSize
        OA_VMA_MAX(createInfo.alignment, (VkDeviceSize)1), // allocAlignment
        (createInfo.flags & OA_VMA_VIRTUAL_ALLOCATION_CREATE_UPPER_ADDRESS_BIT) != 0, // upperAddress
        OA_VMA_SUBALLOCATION_TYPE_UNKNOWN, // allocType - unimportant
        createInfo.flags & OA_VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MASK, // strategy
        &request))
    {
        m_Metadata->Alloc(request,
            OA_VMA_SUBALLOCATION_TYPE_UNKNOWN, // type - unimportant
            createInfo.pUserData);
        outAllocation = (OaVmaVirtualAllocation)request.allocHandle;
        if(outOffset)
            *outOffset = m_Metadata->GetAllocationOffset(request.allocHandle);
        return VK_SUCCESS;
    }
    outAllocation = (OaVmaVirtualAllocation)VK_NULL_HANDLE;
    if (outOffset)
        *outOffset = UINT64_MAX;
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

void OaVmaVirtualBlock_T::GetStatistics(OaVmaStatistics& outStats) const
{
    OaVmaClearStatistics(outStats);
    m_Metadata->AddStatistics(outStats);
}

void OaVmaVirtualBlock_T::CalculateDetailedStatistics(OaVmaDetailedStatistics& outStats) const
{
    OaVmaClearDetailedStatistics(outStats);
    m_Metadata->AddDetailedStatistics(outStats);
}

#if OA_VMA_STATS_STRING_ENABLED
void OaVmaVirtualBlock_T::BuildStatsString(bool detailedMap, OaVmaStringBuilder& sb) const
{
    OaVmaJsonWriter json(GetAllocationCallbacks(), sb);
    json.BeginObject();

    OaVmaDetailedStatistics stats;
    CalculateDetailedStatistics(stats);

    json.WriteString("Stats");
    OaVmaPrintDetailedStatistics(json, stats);

    if (detailedMap)
    {
        json.WriteString("Details");
        json.BeginObject();
        m_Metadata->PrintDetailedMap(json);
        json.EndObject();
    }

    json.EndObject();
}
#endif // OA_VMA_STATS_STRING_ENABLED
#endif // _OA_VMA_VIRTUAL_BLOCK_T_FUNCTIONS
#endif // _OA_VMA_VIRTUAL_BLOCK_T


// Main allocator object.
struct OaVmaAllocator_T
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaAllocator_T)
public:
    const bool m_UseMutex;
    const uint32_t m_VulkanApiVersion;
    bool m_UseKhrDedicatedAllocation; // Can be set only if m_VulkanApiVersion < VK_MAKE_VERSION(1, 1, 0).
    bool m_UseKhrBindMemory2; // Can be set only if m_VulkanApiVersion < VK_MAKE_VERSION(1, 1, 0).
    bool m_UseExtMemoryBudget;
    bool m_UseAmdDeviceCoherentMemory;
    bool m_UseKhrBufferDeviceAddress;
    bool m_UseExtMemoryPriority;
    bool m_UseKhrMaintenance4;
    bool m_UseKhrMaintenance5;
    bool m_UseKhrExternalMemoryWin32;
    const VkDevice m_hDevice;
    const VkInstance m_hInstance;
    const bool m_AllocationCallbacksSpecified;
    const VkAllocationCallbacks m_AllocationCallbacks;
    OaVmaDeviceMemoryCallbacks m_DeviceMemoryCallbacks;
    OaVmaAllocationObjectAllocator m_AllocationObjectAllocator;

    // Each bit (1 << i) is set if HeapSizeLimit is enabled for that heap, so cannot allocate more than the heap size.
    uint32_t m_HeapSizeLimitMask;

    VkPhysicalDeviceProperties m_PhysicalDeviceProperties;
    VkPhysicalDeviceMemoryProperties m_MemProps;

    // Default pools.
    OaVmaBlockVector* m_pBlockVectors[VK_MAX_MEMORY_TYPES];
    OaVmaDedicatedAllocationList m_DedicatedAllocations[VK_MAX_MEMORY_TYPES];

    OaVmaCurrentBudgetData m_Budget;
    OA_VMA_ATOMIC_UINT32 m_DeviceMemoryCount; // Total number of VkDeviceMemory objects.

    explicit OaVmaAllocator_T(const OaVmaAllocatorCreateInfo* pCreateInfo);
    VkResult Init(const OaVmaAllocatorCreateInfo* pCreateInfo);
    ~OaVmaAllocator_T();

    const VkAllocationCallbacks* GetAllocationCallbacks() const
    {
        return m_AllocationCallbacksSpecified ? &m_AllocationCallbacks : OA_VMA_NULL;
    }
    const OaVmaVulkanFunctions& GetVulkanFunctions() const
    {
        return m_VulkanFunctions;
    }

    VkPhysicalDevice GetPhysicalDevice() const { return m_PhysicalDevice; }

    VkDeviceSize GetBufferImageGranularity() const
    {
        return OA_VMA_MAX(
            static_cast<VkDeviceSize>(OA_VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY),
            m_PhysicalDeviceProperties.limits.bufferImageGranularity);
    }

    uint32_t GetMemoryHeapCount() const { return m_MemProps.memoryHeapCount; }
    uint32_t GetMemoryTypeCount() const { return m_MemProps.memoryTypeCount; }

    uint32_t MemoryTypeIndexToHeapIndex(uint32_t memTypeIndex) const
    {
        OA_VMA_ASSERT(memTypeIndex < m_MemProps.memoryTypeCount);
        return m_MemProps.memoryTypes[memTypeIndex].heapIndex;
    }
    // True when specific memory type is HOST_VISIBLE but not HOST_COHERENT.
    bool IsMemoryTypeNonCoherent(uint32_t memTypeIndex) const
    {
        return (m_MemProps.memoryTypes[memTypeIndex].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    }
    // Minimum alignment for all allocations in specific memory type.
    VkDeviceSize GetMemoryTypeMinAlignment(uint32_t memTypeIndex) const
    {
        return IsMemoryTypeNonCoherent(memTypeIndex) ?
            OA_VMA_MAX((VkDeviceSize)OA_VMA_MIN_ALIGNMENT, m_PhysicalDeviceProperties.limits.nonCoherentAtomSize) :
            (VkDeviceSize)OA_VMA_MIN_ALIGNMENT;
    }

    bool IsIntegratedGpu() const
    {
        return m_PhysicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    }

    uint32_t GetGlobalMemoryTypeBits() const { return m_GlobalMemoryTypeBits; }

    void GetBufferMemoryRequirements(
        VkBuffer hBuffer,
        VkMemoryRequirements& memReq,
        bool& requiresDedicatedAllocation,
        bool& prefersDedicatedAllocation) const;
    void GetImageMemoryRequirements(
        VkImage hImage,
        VkMemoryRequirements& memReq,
        bool& requiresDedicatedAllocation,
        bool& prefersDedicatedAllocation) const;
    VkResult FindMemoryTypeIndex(
        uint32_t memoryTypeBits,
        const OaVmaAllocationCreateInfo* pAllocationCreateInfo,
        OaVmaBufferImageUsage bufImgUsage,
        uint32_t* pMemoryTypeIndex) const;

    // Common code for public functions OaVmaCreateBuffer, OaVmaCreateBufferWithAlignment, etc.
    VkResult CreateBuffer(
        const VkBufferCreateInfo* pBufferCreateInfo,
        const OaVmaAllocationCreateInfo* pAllocationCreateInfo,
        void* pMemoryAllocateNext, // pNext chain for VkMemoryAllocateInfo.
        VkBuffer* pBuffer,
        OaVmaAllocation* pAllocation,
        OaVmaAllocationInfo* pAllocationInfo);
    // Common code for public functions OaVmaCreateImage, OaVmaCreateDedicatedImage.
    VkResult CreateImage(
        const VkImageCreateInfo* pImageCreateInfo,
        const OaVmaAllocationCreateInfo* pAllocationCreateInfo,
        void* pMemoryAllocateNext, // pNext chain for VkMemoryAllocateInfo.
        VkImage* pImage,
        OaVmaAllocation* pAllocation,
        OaVmaAllocationInfo* pAllocationInfo);

    // Main allocation function.
    VkResult AllocateMemory(
        VkMemoryRequirements vkMemReq,
        bool requiresDedicatedAllocation,
        bool prefersDedicatedAllocation,
        VkBuffer dedicatedBuffer,
        VkImage dedicatedImage,
        OaVmaBufferImageUsage dedicatedBufferImageUsage,
        void* pMemoryAllocateNext, // Optional pNext chain for VkMemoryAllocateInfo.
        const OaVmaAllocationCreateInfo& createInfo,
        OaVmaSuballocationType suballocType,
        size_t allocationCount,
        OaVmaAllocation* pAllocations);

    // Main deallocation function.
    void FreeMemory(
        size_t allocationCount,
        const OaVmaAllocation* pAllocations);

    void CalculateStatistics(OaVmaTotalStatistics* pStats);

    void GetHeapBudgets(
        OaVmaBudget* outBudgets, uint32_t firstHeap, uint32_t heapCount);

#if OA_VMA_STATS_STRING_ENABLED
    void PrintDetailedMap(class OaVmaJsonWriter& json);
#endif

    static void GetAllocationInfo(OaVmaAllocation hAllocation, OaVmaAllocationInfo* pAllocationInfo);
    static void GetAllocationInfo2(OaVmaAllocation hAllocation, OaVmaAllocationInfo2* pAllocationInfo);

    VkResult CreatePool(const OaVmaPoolCreateInfo* pCreateInfo, OaVmaPool* pPool);
    void DestroyPool(OaVmaPool pool);
    static void GetPoolStatistics(OaVmaPool pool, OaVmaStatistics* pPoolStats);
    static void CalculatePoolStatistics(OaVmaPool pool, OaVmaDetailedStatistics* pPoolStats);

    void SetCurrentFrameIndex(uint32_t frameIndex);
    uint32_t GetCurrentFrameIndex() const { return m_CurrentFrameIndex.load(); }

    static VkResult CheckPoolCorruption(OaVmaPool hPool);
    VkResult CheckCorruption(uint32_t memoryTypeBits);

    // Call to Vulkan function vkAllocateMemory with accompanying bookkeeping.
    VkResult AllocateVulkanMemory(const VkMemoryAllocateInfo* pAllocateInfo, VkDeviceMemory* pMemory);
    // Call to Vulkan function vkFreeMemory with accompanying bookkeeping.
    void FreeVulkanMemory(uint32_t memoryType, VkDeviceSize size, VkDeviceMemory hMemory);
    // Call to Vulkan function vkBindBufferMemory or vkBindBufferMemory2KHR.
    VkResult BindVulkanBuffer(
        VkDeviceMemory memory,
        VkDeviceSize memoryOffset,
        VkBuffer buffer,
        const void* pNext) const;
    // Call to Vulkan function vkBindImageMemory or vkBindImageMemory2KHR.
    VkResult BindVulkanImage(
        VkDeviceMemory memory,
        VkDeviceSize memoryOffset,
        VkImage image,
        const void* pNext) const;

    VkResult Map(OaVmaAllocation hAllocation, void** ppData);
    void Unmap(OaVmaAllocation hAllocation);

    VkResult BindBufferMemory(
        OaVmaAllocation hAllocation,
        VkDeviceSize allocationLocalOffset,
        VkBuffer hBuffer,
        const void* pNext);
    VkResult BindImageMemory(
        OaVmaAllocation hAllocation,
        VkDeviceSize allocationLocalOffset,
        VkImage hImage,
        const void* pNext);

    VkResult FlushOrInvalidateAllocation(
        OaVmaAllocation hAllocation,
        VkDeviceSize offset, VkDeviceSize size,
        OA_VMA_CACHE_OPERATION op);
    VkResult FlushOrInvalidateAllocations(
        uint32_t allocationCount,
        const OaVmaAllocation* allocations,
        const VkDeviceSize* offsets, const VkDeviceSize* sizes,
        OA_VMA_CACHE_OPERATION op);

    VkResult CopyMemoryToAllocation(
        const void* pSrcHostPointer,
        OaVmaAllocation dstAllocation,
        VkDeviceSize dstAllocationLocalOffset,
        VkDeviceSize size);
    VkResult CopyAllocationToMemory(
        OaVmaAllocation srcAllocation,
        VkDeviceSize srcAllocationLocalOffset,
        void* pDstHostPointer,
        VkDeviceSize size);

    void FillAllocation(OaVmaAllocation hAllocation, uint8_t pattern);

    /*
    Returns bit mask of memory types that can support defragmentation on GPU as
    they support creation of required buffer for copy operations.
    */
    uint32_t GetGpuDefragmentationMemoryTypeBits();

#if OA_VMA_EXTERNAL_MEMORY
    VkExternalMemoryHandleTypeFlagsKHR GetExternalMemoryHandleTypeFlags(uint32_t memTypeIndex) const
    {
        return m_TypeExternalMemoryHandleTypes[memTypeIndex];
    }
#endif // #if OA_VMA_EXTERNAL_MEMORY

private:
    VkDeviceSize m_PreferredLargeHeapBlockSize;

    VkPhysicalDevice m_PhysicalDevice;
    OA_VMA_ATOMIC_UINT32 m_CurrentFrameIndex;
    OA_VMA_ATOMIC_UINT32 m_GpuDefragmentationMemoryTypeBits; // UINT32_MAX means uninitialized.
#if OA_VMA_EXTERNAL_MEMORY
    VkExternalMemoryHandleTypeFlagsKHR m_TypeExternalMemoryHandleTypes[VK_MAX_MEMORY_TYPES];
#endif // #if OA_VMA_EXTERNAL_MEMORY

    OA_VMA_RW_MUTEX m_PoolsMutex;
    typedef OaVmaIntrusiveLinkedList<OaVmaPoolListItemTraits> PoolList;
    // Protected by m_PoolsMutex.
    PoolList m_Pools;
    uint32_t m_NextPoolId;

    OaVmaVulkanFunctions m_VulkanFunctions;

    // Global bit mask AND-ed with any memoryTypeBits to disallow certain memory types.
    uint32_t m_GlobalMemoryTypeBits;

    void ImportVulkanFunctions(const OaVmaVulkanFunctions* pVulkanFunctions);

#if OA_VMA_STATIC_VULKAN_FUNCTIONS == 1
    void ImportVulkanFunctions_Static();
#endif

    void ImportVulkanFunctions_Custom(const OaVmaVulkanFunctions* pVulkanFunctions);

#if OA_VMA_DYNAMIC_VULKAN_FUNCTIONS == 1
    void ImportVulkanFunctions_Dynamic();
#endif

    void ValidateVulkanFunctions() const;

    VkDeviceSize CalcPreferredBlockSize(uint32_t memTypeIndex);

    VkResult AllocateMemoryOfType(
        OaVmaPool pool,
        VkDeviceSize size,
        VkDeviceSize alignment,
        bool dedicatedPreferred,
        VkBuffer dedicatedBuffer,
        VkImage dedicatedImage,
        OaVmaBufferImageUsage dedicatedBufferImageUsage,
        void* pMemoryAllocateNext, // Optional pNext chain for VkMemoryAllocateInfo.
        const OaVmaAllocationCreateInfo& createInfo,
        uint32_t memTypeIndex,
        OaVmaSuballocationType suballocType,
        OaVmaDedicatedAllocationList& dedicatedAllocations,
        OaVmaBlockVector& blockVector,
        size_t allocationCount,
        OaVmaAllocation* pAllocations);

    // Helper function only to be used inside AllocateDedicatedMemory.
    VkResult AllocateDedicatedMemoryPage(
        OaVmaPool pool,
        VkDeviceSize size,
        OaVmaSuballocationType suballocType,
        uint32_t memTypeIndex,
        const VkMemoryAllocateInfo& allocInfo,
        bool map,
        bool isUserDataString,
        bool isMappingAllowed,
        void* pUserData,
        OaVmaAllocation* pAllocation);

    // Allocates and registers new VkDeviceMemory specifically for dedicated allocations.
    VkResult AllocateDedicatedMemory(
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
        const void* pNextChain);

    void FreeDedicatedMemory(OaVmaAllocation allocation);

    VkResult CalcMemTypeParams(
        OaVmaAllocationCreateInfo& outCreateInfo,
        uint32_t memTypeIndex,
        VkDeviceSize size,
        size_t allocationCount);
    static VkResult CalcAllocationParams(
        OaVmaAllocationCreateInfo& outCreateInfo,
        bool dedicatedRequired);

    /*
    Calculates and returns bit mask of memory types that can support defragmentation
    on GPU as they support creation of required buffer for copy operations.
    */
    uint32_t CalculateGpuDefragmentationMemoryTypeBits() const;
    uint32_t CalculateGlobalMemoryTypeBits() const;

    bool GetFlushOrInvalidateRange(
        OaVmaAllocation allocation,
        VkDeviceSize offset, VkDeviceSize size,
        VkMappedMemoryRange& outRange) const;

#if OA_VMA_MEMORY_BUDGET
    void UpdateVulkanBudget();
#endif // #if OA_VMA_MEMORY_BUDGET
};

