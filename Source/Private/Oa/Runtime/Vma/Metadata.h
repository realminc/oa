// SPDX: MIT. Derived from Vulkan Memory Allocator, (c) 2017-2026 Advanced Micro Devices, Inc.
// Full MIT copyright/permission notice: Source/Public/Oa/Runtime/OaVma.h. See NOTICE.md.
#ifndef _OA_VMA_BLOCK_METADATA
/*
Data structure used for bookkeeping of allocations and unused ranges of memory
in a single VkDeviceMemory block.
*/
class OaVmaBlockMetadata
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaBlockMetadata)
public:
    // pAllocationCallbacks, if not null, must be owned externally - alive and unchanged for the whole lifetime of this object.
    OaVmaBlockMetadata(const VkAllocationCallbacks* pAllocationCallbacks,
        VkDeviceSize bufferImageGranularity, bool isVirtual);
    virtual ~OaVmaBlockMetadata() = default;

    virtual void Init(VkDeviceSize size) { m_Size = size; }
    bool IsVirtual() const { return m_IsVirtual; }
    VkDeviceSize GetSize() const { return m_Size; }

    // Validates all data structures inside this object. If not valid, returns false.
    virtual bool Validate() const = 0;
    virtual size_t GetAllocationCount() const = 0;
    virtual size_t GetFreeRegionsCount() const = 0;
    virtual VkDeviceSize GetSumFreeSize() const = 0;
    // Returns true if this block is empty - contains only single free suballocation.
    virtual bool IsEmpty() const = 0;
    virtual void GetAllocationInfo(OaVmaAllocHandle allocHandle, OaVmaVirtualAllocationInfo& outInfo) = 0;
    virtual VkDeviceSize GetAllocationOffset(OaVmaAllocHandle allocHandle) const = 0;
    virtual void* GetAllocationUserData(OaVmaAllocHandle allocHandle) const = 0;

    virtual OaVmaAllocHandle GetAllocationListBegin() const = 0;
    virtual OaVmaAllocHandle GetNextAllocation(OaVmaAllocHandle prevAlloc) const = 0;
    virtual VkDeviceSize GetNextFreeRegionSize(OaVmaAllocHandle alloc) const = 0;

    // Shouldn't modify blockCount.
    virtual void AddDetailedStatistics(OaVmaDetailedStatistics& inoutStats) const = 0;
    virtual void AddStatistics(OaVmaStatistics& inoutStats) const = 0;

#if OA_VMA_STATS_STRING_ENABLED
    virtual void PrintDetailedMap(class OaVmaJsonWriter& json) const = 0;
#endif

    // Tries to find a place for suballocation with given parameters inside this block.
    // If succeeded, fills pAllocationRequest and returns true.
    // If failed, returns false.
    virtual bool CreateAllocationRequest(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        bool upperAddress,
        OaVmaSuballocationType allocType,
        // Always one of OA_VMA_ALLOCATION_CREATE_STRATEGY_* or OA_VMA_ALLOCATION_INTERNAL_STRATEGY_* flags.
        uint32_t strategy,
        OaVmaAllocationRequest* pAllocationRequest) = 0;

    virtual VkResult CheckCorruption(const void* pBlockData) = 0;

    // Makes actual allocation based on request. Request must already be checked and valid.
    virtual void Alloc(
        const OaVmaAllocationRequest& request,
        OaVmaSuballocationType type,
        void* userData) = 0;

    // Frees suballocation assigned to given memory region.
    virtual void Free(OaVmaAllocHandle allocHandle) = 0;

    // Frees all allocations.
    // Careful! Don't call it if there are OaVmaAllocation objects owned by userData of cleared allocations!
    virtual void Clear() = 0;

    virtual void SetAllocationUserData(OaVmaAllocHandle allocHandle, void* userData) = 0;
    virtual void DebugLogAllAllocations() const = 0;

protected:
    const VkAllocationCallbacks* GetAllocationCallbacks() const { return m_pAllocationCallbacks; }
    VkDeviceSize GetBufferImageGranularity() const { return m_BufferImageGranularity; }
    VkDeviceSize GetDebugMargin() const { return VkDeviceSize(IsVirtual() ? 0 : OA_VMA_DEBUG_MARGIN); }

    void DebugLogAllocation(VkDeviceSize offset, VkDeviceSize size, void* userData) const;
#if OA_VMA_STATS_STRING_ENABLED
    // mapRefCount == UINT32_MAX means unspecified.
    void PrintDetailedMap_Begin(class OaVmaJsonWriter& json,
        VkDeviceSize unusedBytes,
        size_t allocationCount,
        size_t unusedRangeCount) const;
    void PrintDetailedMap_Allocation(class OaVmaJsonWriter& json,
        VkDeviceSize offset, VkDeviceSize size, void* userData) const;
    static void PrintDetailedMap_UnusedRange(class OaVmaJsonWriter& json,
        VkDeviceSize offset,
        VkDeviceSize size);
    static void PrintDetailedMap_End(class OaVmaJsonWriter& json);
#endif

private:
    VkDeviceSize m_Size;
    const VkAllocationCallbacks* m_pAllocationCallbacks;
    const VkDeviceSize m_BufferImageGranularity;
    const bool m_IsVirtual;
};

#ifndef _OA_VMA_BLOCK_METADATA_FUNCTIONS
OaVmaBlockMetadata::OaVmaBlockMetadata(const VkAllocationCallbacks* pAllocationCallbacks,
    VkDeviceSize bufferImageGranularity, bool isVirtual)
    : m_Size(0),
    m_pAllocationCallbacks(pAllocationCallbacks),
    m_BufferImageGranularity(bufferImageGranularity),
    m_IsVirtual(isVirtual) {}

void OaVmaBlockMetadata::DebugLogAllocation(VkDeviceSize offset, VkDeviceSize size, void* userData) const
{
    if (IsVirtual())
    {
        OA_VMA_LEAK_LOG_FORMAT("UNFREED VIRTUAL ALLOCATION; Offset: %" PRIu64 "; Size: %" PRIu64 "; UserData: %p", offset, size, userData);
    }
    else
    {
        OA_VMA_ASSERT(userData != OA_VMA_NULL);
        OaVmaAllocation allocation = reinterpret_cast<OaVmaAllocation>(userData);

        userData = allocation->GetUserData();
        const char* name = allocation->GetName();

#if OA_VMA_STATS_STRING_ENABLED
        OA_VMA_LEAK_LOG_FORMAT("UNFREED ALLOCATION; Offset: %" PRIu64 "; Size: %" PRIu64 "; UserData: %p; Name: %s; Type: %s; Usage: %" PRIu64,
            offset, size, userData, name ? name : "OaVma_empty",
            OA_VMA_SUBALLOCATION_TYPE_NAMES[allocation->GetSuballocationType()],
            (uint64_t)allocation->GetBufferImageUsage().Value);
#else
        OA_VMA_LEAK_LOG_FORMAT("UNFREED ALLOCATION; Offset: %" PRIu64 "; Size: %" PRIu64 "; UserData: %p; Name: %s; Type: %u",
            offset, size, userData, name ? name : "OaVma_empty",
            (unsigned)allocation->GetSuballocationType());
#endif // OA_VMA_STATS_STRING_ENABLED
    }

}

#if OA_VMA_STATS_STRING_ENABLED
void OaVmaBlockMetadata::PrintDetailedMap_Begin(class OaVmaJsonWriter& json,
    VkDeviceSize unusedBytes, size_t allocationCount, size_t unusedRangeCount) const
{
    json.WriteString("TotalBytes");
    json.WriteNumber(GetSize());

    json.WriteString("UnusedBytes");
    json.WriteNumber(unusedBytes);

    json.WriteString("Allocations");
    json.WriteNumber((uint64_t)allocationCount);

    json.WriteString("UnusedRanges");
    json.WriteNumber((uint64_t)unusedRangeCount);

    json.WriteString("Suballocations");
    json.BeginArray();
}

void OaVmaBlockMetadata::PrintDetailedMap_Allocation(class OaVmaJsonWriter& json,
    VkDeviceSize offset, VkDeviceSize size, void* userData) const
{
    json.BeginObject(true);

    json.WriteString("Offset");
    json.WriteNumber(offset);

    if (IsVirtual())
    {
        json.WriteString("Size");
        json.WriteNumber(size);
        if (userData)
        {
            json.WriteString("CustomData");
            json.BeginString();
            json.ContinueString_Pointer(userData);
            json.EndString();
        }
    }
    else
    {
        ((OaVmaAllocation)userData)->PrintParameters(json);
    }

    json.EndObject();
}

void OaVmaBlockMetadata::PrintDetailedMap_UnusedRange(class OaVmaJsonWriter& json,
    VkDeviceSize offset, VkDeviceSize size)
{
    json.BeginObject(true);

    json.WriteString("Offset");
    json.WriteNumber(offset);

    json.WriteString("Type");
    json.WriteString(OA_VMA_SUBALLOCATION_TYPE_NAMES[OA_VMA_SUBALLOCATION_TYPE_FREE]);

    json.WriteString("Size");
    json.WriteNumber(size);

    json.EndObject();
}

void OaVmaBlockMetadata::PrintDetailedMap_End(class OaVmaJsonWriter& json)
{
    json.EndArray();
}
#endif // OA_VMA_STATS_STRING_ENABLED
#endif // _OA_VMA_BLOCK_METADATA_FUNCTIONS
#endif // _OA_VMA_BLOCK_METADATA

#ifndef _OA_VMA_BLOCK_BUFFER_IMAGE_GRANULARITY
// Before deleting object of this class remember to call 'Destroy()'
class OaVmaBlockBufferImageGranularity final
{
public:
    struct ValidationContext
    {
        const VkAllocationCallbacks* allocCallbacks;
        uint16_t* pageAllocs;
    };

    explicit OaVmaBlockBufferImageGranularity(VkDeviceSize bufferImageGranularity);
    ~OaVmaBlockBufferImageGranularity();

    bool IsEnabled() const { return m_BufferImageGranularity > MAX_LOW_BUFFER_IMAGE_GRANULARITY; }

    void Init(const VkAllocationCallbacks* pAllocationCallbacks, VkDeviceSize size);
    // Before destroying object you must call free it's memory
    void Destroy(const VkAllocationCallbacks* pAllocationCallbacks);

    void RoundupAllocRequest(OaVmaSuballocationType allocType,
        VkDeviceSize& inOutAllocSize,
        VkDeviceSize& inOutAllocAlignment) const;

    bool CheckConflictAndAlignUp(VkDeviceSize& inOutAllocOffset,
        VkDeviceSize allocSize,
        VkDeviceSize blockOffset,
        VkDeviceSize blockSize,
        OaVmaSuballocationType allocType) const;

    void AllocPages(uint8_t allocType, VkDeviceSize offset, VkDeviceSize size);
    void FreePages(VkDeviceSize offset, VkDeviceSize size);
    void Clear();

    ValidationContext StartValidation(const VkAllocationCallbacks* pAllocationCallbacks,
        bool isVirutal) const;
    bool Validate(ValidationContext& ctx, VkDeviceSize offset, VkDeviceSize size) const;
    bool FinishValidation(ValidationContext& ctx) const;

private:
    static constexpr uint16_t MAX_LOW_BUFFER_IMAGE_GRANULARITY = 256;

    struct RegionInfo
    {
        uint8_t allocType;
        uint16_t allocCount;
    };

    VkDeviceSize m_BufferImageGranularity;
    uint32_t m_RegionCount;
    RegionInfo* m_RegionInfo;

    uint32_t GetStartPage(VkDeviceSize offset) const { return OffsetToPageIndex(offset & ~(m_BufferImageGranularity - 1)); }
    uint32_t GetEndPage(VkDeviceSize offset, VkDeviceSize size) const { return OffsetToPageIndex((offset + size - 1) & ~(m_BufferImageGranularity - 1)); }

    uint32_t OffsetToPageIndex(VkDeviceSize offset) const;
    static void AllocPage(RegionInfo& page, uint8_t allocType);
};

#ifndef _OA_VMA_BLOCK_BUFFER_IMAGE_GRANULARITY_FUNCTIONS
OaVmaBlockBufferImageGranularity::OaVmaBlockBufferImageGranularity(VkDeviceSize bufferImageGranularity)
    : m_BufferImageGranularity(bufferImageGranularity),
    m_RegionCount(0),
    m_RegionInfo(OA_VMA_NULL) {}

OaVmaBlockBufferImageGranularity::~OaVmaBlockBufferImageGranularity()
{
    OA_VMA_ASSERT(m_RegionInfo == OA_VMA_NULL && "Free not called before destroying object!");
}

void OaVmaBlockBufferImageGranularity::Init(const VkAllocationCallbacks* pAllocationCallbacks, VkDeviceSize size)
{
    if (IsEnabled())
    {
        m_RegionCount = static_cast<uint32_t>(OaVmaDivideRoundingUp(size, m_BufferImageGranularity));
        m_RegionInfo = OaVma_new_array(pAllocationCallbacks, RegionInfo, m_RegionCount);
        memset(m_RegionInfo, 0, m_RegionCount * sizeof(RegionInfo));
    }
}

void OaVmaBlockBufferImageGranularity::Destroy(const VkAllocationCallbacks* pAllocationCallbacks)
{
    if (m_RegionInfo)
    {
        OaVma_delete_array(pAllocationCallbacks, m_RegionInfo, m_RegionCount);
        m_RegionInfo = OA_VMA_NULL;
    }
}

void OaVmaBlockBufferImageGranularity::RoundupAllocRequest(OaVmaSuballocationType allocType,
    VkDeviceSize& inOutAllocSize,
    VkDeviceSize& inOutAllocAlignment) const
{
    if (m_BufferImageGranularity > 1 &&
        m_BufferImageGranularity <= MAX_LOW_BUFFER_IMAGE_GRANULARITY)
    {
        if (allocType == OA_VMA_SUBALLOCATION_TYPE_UNKNOWN ||
            allocType == OA_VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN ||
            allocType == OA_VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL)
        {
            inOutAllocAlignment = OA_VMA_MAX(inOutAllocAlignment, m_BufferImageGranularity);
            inOutAllocSize = OaVmaAlignUp(inOutAllocSize, m_BufferImageGranularity);
        }
    }
}

bool OaVmaBlockBufferImageGranularity::CheckConflictAndAlignUp(VkDeviceSize& inOutAllocOffset,
    VkDeviceSize allocSize,
    VkDeviceSize blockOffset,
    VkDeviceSize blockSize,
    OaVmaSuballocationType allocType) const
{
    if (IsEnabled())
    {
        uint32_t startPage = GetStartPage(inOutAllocOffset);
        if (m_RegionInfo[startPage].allocCount > 0 &&
            OaVmaIsBufferImageGranularityConflict(static_cast<OaVmaSuballocationType>(m_RegionInfo[startPage].allocType), allocType))
        {
            inOutAllocOffset = OaVmaAlignUp(inOutAllocOffset, m_BufferImageGranularity);
            if (blockSize < allocSize + inOutAllocOffset - blockOffset)
                return true;
            ++startPage;
        }
        uint32_t endPage = GetEndPage(inOutAllocOffset, allocSize);
        if (endPage != startPage &&
            m_RegionInfo[endPage].allocCount > 0 &&
            OaVmaIsBufferImageGranularityConflict(static_cast<OaVmaSuballocationType>(m_RegionInfo[endPage].allocType), allocType))
        {
            return true;
        }
    }
    return false;
}

void OaVmaBlockBufferImageGranularity::AllocPages(uint8_t allocType, VkDeviceSize offset, VkDeviceSize size)
{
    if (IsEnabled())
    {
        uint32_t startPage = GetStartPage(offset);
        AllocPage(m_RegionInfo[startPage], allocType);

        uint32_t endPage = GetEndPage(offset, size);
        if (startPage != endPage)
            AllocPage(m_RegionInfo[endPage], allocType);
    }
}

void OaVmaBlockBufferImageGranularity::FreePages(VkDeviceSize offset, VkDeviceSize size)
{
    if (IsEnabled())
    {
        uint32_t startPage = GetStartPage(offset);
        --m_RegionInfo[startPage].allocCount;
        if (m_RegionInfo[startPage].allocCount == 0)
            m_RegionInfo[startPage].allocType = OA_VMA_SUBALLOCATION_TYPE_FREE;
        uint32_t endPage = GetEndPage(offset, size);
        if (startPage != endPage)
        {
            --m_RegionInfo[endPage].allocCount;
            if (m_RegionInfo[endPage].allocCount == 0)
                m_RegionInfo[endPage].allocType = OA_VMA_SUBALLOCATION_TYPE_FREE;
        }
    }
}

void OaVmaBlockBufferImageGranularity::Clear()
{
    if (m_RegionInfo)
        memset(m_RegionInfo, 0, m_RegionCount * sizeof(RegionInfo));
}

OaVmaBlockBufferImageGranularity::ValidationContext OaVmaBlockBufferImageGranularity::StartValidation(
    const VkAllocationCallbacks* pAllocationCallbacks, bool isVirutal) const
{
    ValidationContext ctx{ pAllocationCallbacks, OA_VMA_NULL };
    if (!isVirutal && IsEnabled())
    {
        ctx.pageAllocs = OaVma_new_array(pAllocationCallbacks, uint16_t, m_RegionCount);
        memset(ctx.pageAllocs, 0, m_RegionCount * sizeof(uint16_t));
    }
    return ctx;
}

bool OaVmaBlockBufferImageGranularity::Validate(ValidationContext& ctx,
    VkDeviceSize offset, VkDeviceSize size) const
{
    if (IsEnabled())
    {
        uint32_t start = GetStartPage(offset);
        ++ctx.pageAllocs[start];
        OA_VMA_VALIDATE(m_RegionInfo[start].allocCount > 0);

        uint32_t end = GetEndPage(offset, size);
        if (start != end)
        {
            ++ctx.pageAllocs[end];
            OA_VMA_VALIDATE(m_RegionInfo[end].allocCount > 0);
        }
    }
    return true;
}

bool OaVmaBlockBufferImageGranularity::FinishValidation(ValidationContext& ctx) const
{
    // Check proper page structure
    if (IsEnabled())
    {
        OA_VMA_ASSERT(ctx.pageAllocs != OA_VMA_NULL && "Validation context not initialized!");

        for (uint32_t page = 0; page < m_RegionCount; ++page)
        {
            OA_VMA_VALIDATE(ctx.pageAllocs[page] == m_RegionInfo[page].allocCount);
        }
        OaVma_delete_array(ctx.allocCallbacks, ctx.pageAllocs, m_RegionCount);
        ctx.pageAllocs = OA_VMA_NULL;
    }
    return true;
}

uint32_t OaVmaBlockBufferImageGranularity::OffsetToPageIndex(VkDeviceSize offset) const
{
    return static_cast<uint32_t>(offset >> OA_VMA_BITSCAN_MSB(m_BufferImageGranularity));
}

void OaVmaBlockBufferImageGranularity::AllocPage(RegionInfo& page, uint8_t allocType)
{
    // When current alloc type is free then it can be overridden by new type
    if (page.allocCount == 0 || (page.allocCount > 0 && page.allocType == OA_VMA_SUBALLOCATION_TYPE_FREE))
        page.allocType = allocType;

    ++page.allocCount;
}
#endif // _OA_VMA_BLOCK_BUFFER_IMAGE_GRANULARITY_FUNCTIONS
#endif // _OA_VMA_BLOCK_BUFFER_IMAGE_GRANULARITY

#ifndef _OA_VMA_BLOCK_METADATA_LINEAR
/*
Allocations and their references in internal data structure look like this:

if(m_2ndVectorMode == SECOND_VECTOR_EMPTY):

        0 +-------+
          |       |
          |       |
          |       |
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount]
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount + 1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  1st[1st.size() - 1]
          +-------+
          |       |
          |       |
          |       |
GetSize() +-------+

if(m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER):

        0 +-------+
          | Alloc |  2nd[0]
          +-------+
          | Alloc |  2nd[1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  2nd[2nd.size() - 1]
          +-------+
          |       |
          |       |
          |       |
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount]
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount + 1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  1st[1st.size() - 1]
          +-------+
          |       |
GetSize() +-------+

if(m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK):

        0 +-------+
          |       |
          |       |
          |       |
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount]
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount + 1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  1st[1st.size() - 1]
          +-------+
          |       |
          |       |
          |       |
          +-------+
          | Alloc |  2nd[2nd.size() - 1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  2nd[1]
          +-------+
          | Alloc |  2nd[0]
GetSize() +-------+

*/
class OaVmaBlockMetadata_Linear : public OaVmaBlockMetadata
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaBlockMetadata_Linear)
public:
    OaVmaBlockMetadata_Linear(const VkAllocationCallbacks* pAllocationCallbacks,
        VkDeviceSize bufferImageGranularity, bool isVirtual);
    ~OaVmaBlockMetadata_Linear() override = default;

    VkDeviceSize GetSumFreeSize() const override { return m_SumFreeSize; }
    bool IsEmpty() const override { return GetAllocationCount() == 0; }
    VkDeviceSize GetAllocationOffset(OaVmaAllocHandle allocHandle) const override { return (VkDeviceSize)allocHandle - 1; }

    void Init(VkDeviceSize size) override;
    bool Validate() const override;
    size_t GetAllocationCount() const override;
    size_t GetFreeRegionsCount() const override;

    void AddDetailedStatistics(OaVmaDetailedStatistics& inoutStats) const override;
    void AddStatistics(OaVmaStatistics& inoutStats) const override;

#if OA_VMA_STATS_STRING_ENABLED
    void PrintDetailedMap(class OaVmaJsonWriter& json) const override;
#endif

    bool CreateAllocationRequest(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        bool upperAddress,
        OaVmaSuballocationType allocType,
        uint32_t strategy,
        OaVmaAllocationRequest* pAllocationRequest) override;

    VkResult CheckCorruption(const void* pBlockData) override;

    void Alloc(
        const OaVmaAllocationRequest& request,
        OaVmaSuballocationType type,
        void* userData) override;

    void Free(OaVmaAllocHandle allocHandle) override;
    void GetAllocationInfo(OaVmaAllocHandle allocHandle, OaVmaVirtualAllocationInfo& outInfo) override;
    void* GetAllocationUserData(OaVmaAllocHandle allocHandle) const override;
    OaVmaAllocHandle GetAllocationListBegin() const override;
    OaVmaAllocHandle GetNextAllocation(OaVmaAllocHandle prevAlloc) const override;
    VkDeviceSize GetNextFreeRegionSize(OaVmaAllocHandle alloc) const override;
    void Clear() override;
    void SetAllocationUserData(OaVmaAllocHandle allocHandle, void* userData) override;
    void DebugLogAllAllocations() const override;

private:
    /*
    There are two suballocation vectors, used in ping-pong way.
    The one with index m_1stVectorIndex is called 1st.
    The one with index (m_1stVectorIndex ^ 1) is called 2nd.
    2nd can be non-empty only when 1st is not empty.
    When 2nd is not empty, m_2ndVectorMode indicates its mode of operation.
    */
    typedef OaVmaVector<OaVmaSuballocation, OaVmaStlAllocator<OaVmaSuballocation>> SuballocationVectorType;

    enum SECOND_VECTOR_MODE
    {
        SECOND_VECTOR_EMPTY,
        /*
        Suballocations in 2nd vector are created later than the ones in 1st, but they
        all have smaller offset.
        */
        SECOND_VECTOR_RING_BUFFER,
        /*
        Suballocations in 2nd vector are upper side of double stack.
        They all have offsets higher than those in 1st vector.
        Top of this stack means smaller offsets, but higher indices in this vector.
        */
        SECOND_VECTOR_DOUBLE_STACK,
    };

    VkDeviceSize m_SumFreeSize;
    SuballocationVectorType m_Suballocations0, m_Suballocations1;
    uint32_t m_1stVectorIndex;
    SECOND_VECTOR_MODE m_2ndVectorMode;
    // Number of items in 1st vector with hAllocation = null at the beginning.
    size_t m_1stNullItemsBeginCount;
    // Number of other items in 1st vector with hAllocation = null somewhere in the middle.
    size_t m_1stNullItemsMiddleCount;
    // Number of items in 2nd vector with hAllocation = null.
    size_t m_2ndNullItemsCount;

    SuballocationVectorType& AccessSuballocations1st() { return m_1stVectorIndex ? m_Suballocations1 : m_Suballocations0; }
    SuballocationVectorType& AccessSuballocations2nd() { return m_1stVectorIndex ? m_Suballocations0 : m_Suballocations1; }
    const SuballocationVectorType& AccessSuballocations1st() const { return m_1stVectorIndex ? m_Suballocations1 : m_Suballocations0; }
    const SuballocationVectorType& AccessSuballocations2nd() const { return m_1stVectorIndex ? m_Suballocations0 : m_Suballocations1; }

    OaVmaSuballocation& FindSuballocation(VkDeviceSize offset) const;
    bool ShouldCompact1st() const;
    void CleanupAfterFree();

    bool CreateAllocationRequest_LowerAddress(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        OaVmaSuballocationType allocType,
        uint32_t strategy,
        OaVmaAllocationRequest* pAllocationRequest);
    bool CreateAllocationRequest_UpperAddress(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        OaVmaSuballocationType allocType,
        uint32_t strategy,
        OaVmaAllocationRequest* pAllocationRequest);
};

#ifndef _OA_VMA_BLOCK_METADATA_LINEAR_FUNCTIONS
OaVmaBlockMetadata_Linear::OaVmaBlockMetadata_Linear(const VkAllocationCallbacks* pAllocationCallbacks,
    VkDeviceSize bufferImageGranularity, bool isVirtual)
    : OaVmaBlockMetadata(pAllocationCallbacks, bufferImageGranularity, isVirtual),
    m_SumFreeSize(0),
    m_Suballocations0(OaVmaStlAllocator<OaVmaSuballocation>(pAllocationCallbacks)),
    m_Suballocations1(OaVmaStlAllocator<OaVmaSuballocation>(pAllocationCallbacks)),
    m_1stVectorIndex(0),
    m_2ndVectorMode(SECOND_VECTOR_EMPTY),
    m_1stNullItemsBeginCount(0),
    m_1stNullItemsMiddleCount(0),
    m_2ndNullItemsCount(0) {}

void OaVmaBlockMetadata_Linear::Init(VkDeviceSize size)
{
    OaVmaBlockMetadata::Init(size);
    m_SumFreeSize = size;
}

bool OaVmaBlockMetadata_Linear::Validate() const
{
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    OA_VMA_VALIDATE(suballocations2nd.empty() == (m_2ndVectorMode == SECOND_VECTOR_EMPTY));
    OA_VMA_VALIDATE(!suballocations1st.empty() ||
        suballocations2nd.empty() ||
        m_2ndVectorMode != SECOND_VECTOR_RING_BUFFER);

    if (!suballocations1st.empty())
    {
        // Null item at the beginning should be accounted into m_1stNullItemsBeginCount.
        OA_VMA_VALIDATE(suballocations1st[m_1stNullItemsBeginCount].type != OA_VMA_SUBALLOCATION_TYPE_FREE);
        // Null item at the end should be just pop_back().
        OA_VMA_VALIDATE(suballocations1st.back().type != OA_VMA_SUBALLOCATION_TYPE_FREE);
    }
    if (!suballocations2nd.empty())
    {
        // Null item at the end should be just pop_back().
        OA_VMA_VALIDATE(suballocations2nd.back().type != OA_VMA_SUBALLOCATION_TYPE_FREE);
    }

    OA_VMA_VALIDATE(m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount <= suballocations1st.size());
    OA_VMA_VALIDATE(m_2ndNullItemsCount <= suballocations2nd.size());

    VkDeviceSize sumUsedSize = 0;
    const size_t suballoc1stCount = suballocations1st.size();
    const VkDeviceSize debugMargin = GetDebugMargin();
    VkDeviceSize offset = 0;

    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const size_t suballoc2ndCount = suballocations2nd.size();
        size_t nullItem2ndCount = 0;
        for (size_t i = 0; i < suballoc2ndCount; ++i)
        {
            const OaVmaSuballocation& suballoc = suballocations2nd[i];
            const bool currFree = (suballoc.type == OA_VMA_SUBALLOCATION_TYPE_FREE);

            OaVmaAllocation const alloc = (OaVmaAllocation)suballoc.userData;
            if (!IsVirtual())
            {
                OA_VMA_VALIDATE(currFree == (alloc == VK_NULL_HANDLE));
            }
            OA_VMA_VALIDATE(suballoc.offset >= offset);

            if (!currFree)
            {
                if (!IsVirtual())
                {
                    OA_VMA_VALIDATE((VkDeviceSize)alloc->GetAllocHandle() == suballoc.offset + 1);
                    OA_VMA_VALIDATE(alloc->GetSize() == suballoc.size);
                }
                sumUsedSize += suballoc.size;
            }
            else
            {
                ++nullItem2ndCount;
            }

            offset = suballoc.offset + suballoc.size + debugMargin;
        }

        OA_VMA_VALIDATE(nullItem2ndCount == m_2ndNullItemsCount);
    }

    for (size_t i = 0; i < m_1stNullItemsBeginCount; ++i)
    {
        const OaVmaSuballocation& suballoc = suballocations1st[i];
        OA_VMA_VALIDATE(suballoc.type == OA_VMA_SUBALLOCATION_TYPE_FREE &&
            suballoc.userData == OA_VMA_NULL);
    }

    size_t nullItem1stCount = m_1stNullItemsBeginCount;

    for (size_t i = m_1stNullItemsBeginCount; i < suballoc1stCount; ++i)
    {
        const OaVmaSuballocation& suballoc = suballocations1st[i];
        const bool currFree = (suballoc.type == OA_VMA_SUBALLOCATION_TYPE_FREE);

        OaVmaAllocation const alloc = (OaVmaAllocation)suballoc.userData;
        if (!IsVirtual())
        {
            OA_VMA_VALIDATE(currFree == (alloc == VK_NULL_HANDLE));
        }
        OA_VMA_VALIDATE(suballoc.offset >= offset);
        OA_VMA_VALIDATE(i >= m_1stNullItemsBeginCount || currFree);

        if (!currFree)
        {
            if (!IsVirtual())
            {
                OA_VMA_VALIDATE((VkDeviceSize)alloc->GetAllocHandle() == suballoc.offset + 1);
                OA_VMA_VALIDATE(alloc->GetSize() == suballoc.size);
            }
            sumUsedSize += suballoc.size;
        }
        else
        {
            ++nullItem1stCount;
        }

        offset = suballoc.offset + suballoc.size + debugMargin;
    }
    OA_VMA_VALIDATE(nullItem1stCount == m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount);

    if (m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        const size_t suballoc2ndCount = suballocations2nd.size();
        size_t nullItem2ndCount = 0;
        for (size_t i = suballoc2ndCount; i--; )
        {
            const OaVmaSuballocation& suballoc = suballocations2nd[i];
            const bool currFree = (suballoc.type == OA_VMA_SUBALLOCATION_TYPE_FREE);

            OaVmaAllocation const alloc = (OaVmaAllocation)suballoc.userData;
            if (!IsVirtual())
            {
                OA_VMA_VALIDATE(currFree == (alloc == VK_NULL_HANDLE));
            }
            OA_VMA_VALIDATE(suballoc.offset >= offset);

            if (!currFree)
            {
                if (!IsVirtual())
                {
                    OA_VMA_VALIDATE((VkDeviceSize)alloc->GetAllocHandle() == suballoc.offset + 1);
                    OA_VMA_VALIDATE(alloc->GetSize() == suballoc.size);
                }
                sumUsedSize += suballoc.size;
            }
            else
            {
                ++nullItem2ndCount;
            }

            offset = suballoc.offset + suballoc.size + debugMargin;
        }

        OA_VMA_VALIDATE(nullItem2ndCount == m_2ndNullItemsCount);
    }

    OA_VMA_VALIDATE(offset <= GetSize());
    OA_VMA_VALIDATE(m_SumFreeSize == GetSize() - sumUsedSize);

    return true;
}

size_t OaVmaBlockMetadata_Linear::GetAllocationCount() const
{
    return AccessSuballocations1st().size() - m_1stNullItemsBeginCount - m_1stNullItemsMiddleCount +
        AccessSuballocations2nd().size() - m_2ndNullItemsCount;
}

size_t OaVmaBlockMetadata_Linear::GetFreeRegionsCount() const
{
    // Function only used for defragmentation, which is disabled for this algorithm
    OA_VMA_ASSERT(0);
    return SIZE_MAX;
}

void OaVmaBlockMetadata_Linear::AddDetailedStatistics(OaVmaDetailedStatistics& inoutStats) const
{
    const VkDeviceSize size = GetSize();
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    const size_t suballoc1stCount = suballocations1st.size();
    const size_t suballoc2ndCount = suballocations2nd.size();

    inoutStats.statistics.blockCount++;
    inoutStats.statistics.blockBytes += size;

    VkDeviceSize lastOffset = 0;

    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const VkDeviceSize freeSpace2ndTo1stEnd = suballocations1st[m_1stNullItemsBeginCount].offset;
        size_t nextAlloc2ndIndex = 0;
        while (lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAllocIndex to the end.
            while (nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].userData == OA_VMA_NULL)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex < suballoc2ndCount)
            {
                const OaVmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    OaVmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                OaVmaAddDetailedStatisticsAllocation(inoutStats, suballoc.size);

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                // There is free space from lastOffset to freeSpace2ndTo1stEnd.
                if (lastOffset < freeSpace2ndTo1stEnd)
                {
                    const VkDeviceSize unusedRangeSize = freeSpace2ndTo1stEnd - lastOffset;
                    OaVmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
                }

                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    size_t nextAlloc1stIndex = m_1stNullItemsBeginCount;
    const VkDeviceSize freeSpace1stTo2ndEnd =
        m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK ? suballocations2nd.back().offset : size;
    while (lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while (nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].userData == OA_VMA_NULL)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if (nextAlloc1stIndex < suballoc1stCount)
        {
            const OaVmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];

            // 1. Process free space before this allocation.
            if (lastOffset < suballoc.offset)
            {
                // There is free space from lastOffset to suballoc.offset.
                const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                OaVmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
            }

            // 2. Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            OaVmaAddDetailedStatisticsAllocation(inoutStats, suballoc.size);

            // 3. Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            // There is free space from lastOffset to freeSpace1stTo2ndEnd.
            if (lastOffset < freeSpace1stTo2ndEnd)
            {
                const VkDeviceSize unusedRangeSize = freeSpace1stTo2ndEnd - lastOffset;
                OaVmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
            }

            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if (m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while (lastOffset < size)
        {
            // Find next non-null allocation or move nextAllocIndex to the end.
            while (nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].userData == OA_VMA_NULL)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex != SIZE_MAX)
            {
                const OaVmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    OaVmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                OaVmaAddDetailedStatisticsAllocation(inoutStats, suballoc.size);

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                // There is free space from lastOffset to size.
                if (lastOffset < size)
                {
                    const VkDeviceSize unusedRangeSize = size - lastOffset;
                    OaVmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
                }

                // End of loop.
                lastOffset = size;
            }
        }
    }
}

void OaVmaBlockMetadata_Linear::AddStatistics(OaVmaStatistics& inoutStats) const
{
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    const VkDeviceSize size = GetSize();
    const size_t suballoc1stCount = suballocations1st.size();
    const size_t suballoc2ndCount = suballocations2nd.size();

    inoutStats.blockCount++;
    inoutStats.blockBytes += size;
    inoutStats.allocationBytes += size - m_SumFreeSize;

    VkDeviceSize lastOffset = 0;

    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const VkDeviceSize freeSpace2ndTo1stEnd = suballocations1st[m_1stNullItemsBeginCount].offset;
        size_t nextAlloc2ndIndex = m_1stNullItemsBeginCount;
        while (lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].userData == OA_VMA_NULL)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex < suballoc2ndCount)
            {
                const OaVmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++inoutStats.allocationCount;

                // Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    size_t nextAlloc1stIndex = m_1stNullItemsBeginCount;
    const VkDeviceSize freeSpace1stTo2ndEnd =
        m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK ? suballocations2nd.back().offset : size;
    while (lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while (nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].userData == OA_VMA_NULL)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if (nextAlloc1stIndex < suballoc1stCount)
        {
            const OaVmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];

            // Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            ++inoutStats.allocationCount;

            // Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if (m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while (lastOffset < size)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].userData == OA_VMA_NULL)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex != SIZE_MAX)
            {
                const OaVmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++inoutStats.allocationCount;

                // Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                // End of loop.
                lastOffset = size;
            }
        }
    }
}

#if OA_VMA_STATS_STRING_ENABLED
void OaVmaBlockMetadata_Linear::PrintDetailedMap(class OaVmaJsonWriter& json) const
{
    const VkDeviceSize size = GetSize();
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    const size_t suballoc1stCount = suballocations1st.size();
    const size_t suballoc2ndCount = suballocations2nd.size();

    // FIRST PASS

    size_t unusedRangeCount = 0;
    VkDeviceSize usedBytes = 0;

    VkDeviceSize lastOffset = 0;

    size_t alloc2ndCount = 0;
    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const VkDeviceSize freeSpace2ndTo1stEnd = suballocations1st[m_1stNullItemsBeginCount].offset;
        size_t nextAlloc2ndIndex = 0;
        while (lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].userData == OA_VMA_NULL)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex < suballoc2ndCount)
            {
                const OaVmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    ++unusedRangeCount;
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++alloc2ndCount;
                usedBytes += suballoc.size;

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if (lastOffset < freeSpace2ndTo1stEnd)
                {
                    // There is free space from lastOffset to freeSpace2ndTo1stEnd.
                    ++unusedRangeCount;
                }

                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    size_t nextAlloc1stIndex = m_1stNullItemsBeginCount;
    size_t alloc1stCount = 0;
    const VkDeviceSize freeSpace1stTo2ndEnd =
        m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK ? suballocations2nd.back().offset : size;
    while (lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while (nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].userData == OA_VMA_NULL)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if (nextAlloc1stIndex < suballoc1stCount)
        {
            const OaVmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];

            // 1. Process free space before this allocation.
            if (lastOffset < suballoc.offset)
            {
                // There is free space from lastOffset to suballoc.offset.
                ++unusedRangeCount;
            }

            // 2. Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            ++alloc1stCount;
            usedBytes += suballoc.size;

            // 3. Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            if (lastOffset < freeSpace1stTo2ndEnd)
            {
                // There is free space from lastOffset to freeSpace1stTo2ndEnd.
                ++unusedRangeCount;
            }

            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if (m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while (lastOffset < size)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].userData == OA_VMA_NULL)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex != SIZE_MAX)
            {
                const OaVmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    ++unusedRangeCount;
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++alloc2ndCount;
                usedBytes += suballoc.size;

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if (lastOffset < size)
                {
                    // There is free space from lastOffset to size.
                    ++unusedRangeCount;
                }

                // End of loop.
                lastOffset = size;
            }
        }
    }

    const VkDeviceSize unusedBytes = size - usedBytes;
    PrintDetailedMap_Begin(json, unusedBytes, alloc1stCount + alloc2ndCount, unusedRangeCount);

    // SECOND PASS
    lastOffset = 0;

    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const VkDeviceSize freeSpace2ndTo1stEnd = suballocations1st[m_1stNullItemsBeginCount].offset;
        size_t nextAlloc2ndIndex = 0;
        while (lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].userData == OA_VMA_NULL)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex < suballoc2ndCount)
            {
                const OaVmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                PrintDetailedMap_Allocation(json, suballoc.offset, suballoc.size, suballoc.userData);

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if (lastOffset < freeSpace2ndTo1stEnd)
                {
                    // There is free space from lastOffset to freeSpace2ndTo1stEnd.
                    const VkDeviceSize unusedRangeSize = freeSpace2ndTo1stEnd - lastOffset;
                    PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
                }

                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    nextAlloc1stIndex = m_1stNullItemsBeginCount;
    while (lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while (nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].userData == OA_VMA_NULL)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if (nextAlloc1stIndex < suballoc1stCount)
        {
            const OaVmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];

            // 1. Process free space before this allocation.
            if (lastOffset < suballoc.offset)
            {
                // There is free space from lastOffset to suballoc.offset.
                const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
            }

            // 2. Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            PrintDetailedMap_Allocation(json, suballoc.offset, suballoc.size, suballoc.userData);

            // 3. Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            if (lastOffset < freeSpace1stTo2ndEnd)
            {
                // There is free space from lastOffset to freeSpace1stTo2ndEnd.
                const VkDeviceSize unusedRangeSize = freeSpace1stTo2ndEnd - lastOffset;
                PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
            }

            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if (m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while (lastOffset < size)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].userData == OA_VMA_NULL)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex != SIZE_MAX)
            {
                const OaVmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                PrintDetailedMap_Allocation(json, suballoc.offset, suballoc.size, suballoc.userData);

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if (lastOffset < size)
                {
                    // There is free space from lastOffset to size.
                    const VkDeviceSize unusedRangeSize = size - lastOffset;
                    PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
                }

                // End of loop.
                lastOffset = size;
            }
        }
    }

    PrintDetailedMap_End(json);
}
#endif // OA_VMA_STATS_STRING_ENABLED

bool OaVmaBlockMetadata_Linear::CreateAllocationRequest(
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    bool upperAddress,
    OaVmaSuballocationType allocType,
    uint32_t strategy,
    OaVmaAllocationRequest* pAllocationRequest)
{
    OA_VMA_ASSERT(allocSize > 0);
    OA_VMA_ASSERT(allocType != OA_VMA_SUBALLOCATION_TYPE_FREE);
    OA_VMA_ASSERT(pAllocationRequest != OA_VMA_NULL);
    OA_VMA_HEAVY_ASSERT(Validate());

    if(allocSize > GetSize())
        return false;

    pAllocationRequest->size = allocSize;
    return upperAddress ?
        CreateAllocationRequest_UpperAddress(
            allocSize, allocAlignment, allocType, strategy, pAllocationRequest) :
        CreateAllocationRequest_LowerAddress(
            allocSize, allocAlignment, allocType, strategy, pAllocationRequest);
}

VkResult OaVmaBlockMetadata_Linear::CheckCorruption(const void* pBlockData)
{
    OA_VMA_ASSERT(!IsVirtual());
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    for (size_t i = m_1stNullItemsBeginCount, count = suballocations1st.size(); i < count; ++i)
    {
        const OaVmaSuballocation& suballoc = suballocations1st[i];
        if (suballoc.type != OA_VMA_SUBALLOCATION_TYPE_FREE)
        {
            if (!OaVmaValidateMagicValue(pBlockData, suballoc.offset + suballoc.size))
            {
                OA_VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER VALIDATED ALLOCATION!");
                return VK_ERROR_UNKNOWN_COPY;
            }
        }
    }

    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    for (size_t i = 0, count = suballocations2nd.size(); i < count; ++i)
    {
        const OaVmaSuballocation& suballoc = suballocations2nd[i];
        if (suballoc.type != OA_VMA_SUBALLOCATION_TYPE_FREE)
        {
            if (!OaVmaValidateMagicValue(pBlockData, suballoc.offset + suballoc.size))
            {
                OA_VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER VALIDATED ALLOCATION!");
                return VK_ERROR_UNKNOWN_COPY;
            }
        }
    }

    return VK_SUCCESS;
}

void OaVmaBlockMetadata_Linear::Alloc(
    const OaVmaAllocationRequest& request,
    OaVmaSuballocationType type,
    void* userData)
{
    const VkDeviceSize offset = (VkDeviceSize)request.allocHandle - 1;
    const OaVmaSuballocation newSuballoc = { offset, request.size, userData, type };

    switch (request.type)
    {
    case OaVmaAllocationRequestType::UpperAddress:
    {
        OA_VMA_ASSERT(m_2ndVectorMode != SECOND_VECTOR_RING_BUFFER &&
            "CRITICAL ERROR: Trying to use linear allocator as double stack while it was already used as ring buffer.");
        SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
        suballocations2nd.push_back(newSuballoc);
        m_2ndVectorMode = SECOND_VECTOR_DOUBLE_STACK;
    }
    break;
    case OaVmaAllocationRequestType::EndOf1st:
    {
        SuballocationVectorType& suballocations1st = AccessSuballocations1st();

        OA_VMA_ASSERT(suballocations1st.empty() ||
            offset >= suballocations1st.back().offset + suballocations1st.back().size);
        // Check if it fits before the end of the block.
        OA_VMA_ASSERT(offset + request.size <= GetSize());

        suballocations1st.push_back(newSuballoc);
    }
    break;
    case OaVmaAllocationRequestType::EndOf2nd:
    {
        SuballocationVectorType& suballocations1st = AccessSuballocations1st();
        // New allocation at the end of 2-part ring buffer, so before first allocation from 1st vector.
        OA_VMA_ASSERT(!suballocations1st.empty() &&
            offset + request.size <= suballocations1st[m_1stNullItemsBeginCount].offset);
        SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

        switch (m_2ndVectorMode)
        {
        case SECOND_VECTOR_EMPTY:
            // First allocation from second part ring buffer.
            OA_VMA_ASSERT(suballocations2nd.empty());
            m_2ndVectorMode = SECOND_VECTOR_RING_BUFFER;
            break;
        case SECOND_VECTOR_RING_BUFFER:
            // 2-part ring buffer is already started.
            OA_VMA_ASSERT(!suballocations2nd.empty());
            break;
        case SECOND_VECTOR_DOUBLE_STACK:
            OA_VMA_ASSERT(0 && "CRITICAL ERROR: Trying to use linear allocator as ring buffer while it was already used as double stack.");
            break;
        default:
            OA_VMA_ASSERT(0);
        }

        suballocations2nd.push_back(newSuballoc);
    }
    break;
    default:
        OA_VMA_ASSERT(0 && "CRITICAL INTERNAL ERROR.");
    }

    m_SumFreeSize -= newSuballoc.size;
}

void OaVmaBlockMetadata_Linear::Free(OaVmaAllocHandle allocHandle)
{
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    VkDeviceSize offset = (VkDeviceSize)allocHandle - 1;

    if (!suballocations1st.empty())
    {
        // First allocation: Mark it as next empty at the beginning.
        OaVmaSuballocation& firstSuballoc = suballocations1st[m_1stNullItemsBeginCount];
        if (firstSuballoc.offset == offset)
        {
            firstSuballoc.type = OA_VMA_SUBALLOCATION_TYPE_FREE;
            firstSuballoc.userData = OA_VMA_NULL;
            m_SumFreeSize += firstSuballoc.size;
            ++m_1stNullItemsBeginCount;
            CleanupAfterFree();
            return;
        }
    }

    // Last allocation in 2-part ring buffer or top of upper stack (same logic).
    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER ||
        m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        OaVmaSuballocation& lastSuballoc = suballocations2nd.back();
        if (lastSuballoc.offset == offset)
        {
            m_SumFreeSize += lastSuballoc.size;
            suballocations2nd.pop_back();
            CleanupAfterFree();
            return;
        }
    }
    // Last allocation in 1st vector.
    else if (m_2ndVectorMode == SECOND_VECTOR_EMPTY)
    {
        OaVmaSuballocation& lastSuballoc = suballocations1st.back();
        if (lastSuballoc.offset == offset)
        {
            m_SumFreeSize += lastSuballoc.size;
            suballocations1st.pop_back();
            CleanupAfterFree();
            return;
        }
    }

    OaVmaSuballocation refSuballoc;
    refSuballoc.offset = offset;
    // Rest of members stays uninitialized intentionally for better performance.

    // Item from the middle of 1st vector.
    {
        const SuballocationVectorType::iterator it = OaVmaBinaryFindSorted(
            suballocations1st.begin() + m_1stNullItemsBeginCount,
            suballocations1st.end(),
            refSuballoc,
            OaVmaSuballocationOffsetLess());
        if (it != suballocations1st.end())
        {
            it->type = OA_VMA_SUBALLOCATION_TYPE_FREE;
            it->userData = OA_VMA_NULL;
            ++m_1stNullItemsMiddleCount;
            m_SumFreeSize += it->size;
            CleanupAfterFree();
            return;
        }
    }

    if (m_2ndVectorMode != SECOND_VECTOR_EMPTY)
    {
        // Item from the middle of 2nd vector.
        const SuballocationVectorType::iterator it = m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER ?
            OaVmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, OaVmaSuballocationOffsetLess()) :
            OaVmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, OaVmaSuballocationOffsetGreater());
        if (it != suballocations2nd.end())
        {
            it->type = OA_VMA_SUBALLOCATION_TYPE_FREE;
            it->userData = OA_VMA_NULL;
            ++m_2ndNullItemsCount;
            m_SumFreeSize += it->size;
            CleanupAfterFree();
            return;
        }
    }

    OA_VMA_ASSERT(0 && "Allocation to free not found in linear allocator!");
}

void OaVmaBlockMetadata_Linear::GetAllocationInfo(OaVmaAllocHandle allocHandle, OaVmaVirtualAllocationInfo& outInfo)
{
    outInfo.offset = (VkDeviceSize)allocHandle - 1;
    OaVmaSuballocation& suballoc = FindSuballocation(outInfo.offset);
    outInfo.size = suballoc.size;
    outInfo.pUserData = suballoc.userData;
}

void* OaVmaBlockMetadata_Linear::GetAllocationUserData(OaVmaAllocHandle allocHandle) const
{
    return FindSuballocation((VkDeviceSize)allocHandle - 1).userData;
}

OaVmaAllocHandle OaVmaBlockMetadata_Linear::GetAllocationListBegin() const
{
    // Function only used for defragmentation, which is disabled for this algorithm
    OA_VMA_ASSERT(0);
    return VK_NULL_HANDLE;
}

OaVmaAllocHandle OaVmaBlockMetadata_Linear::GetNextAllocation(OaVmaAllocHandle prevAlloc) const
{
    // Function only used for defragmentation, which is disabled for this algorithm
    OA_VMA_ASSERT(0);
    return VK_NULL_HANDLE;
}

VkDeviceSize OaVmaBlockMetadata_Linear::GetNextFreeRegionSize(OaVmaAllocHandle alloc) const
{
    // Function only used for defragmentation, which is disabled for this algorithm
    OA_VMA_ASSERT(0);
    return 0;
}

void OaVmaBlockMetadata_Linear::Clear()
{
    m_SumFreeSize = GetSize();
    m_Suballocations0.clear();
    m_Suballocations1.clear();
    // Leaving m_1stVectorIndex unchanged - it doesn't matter.
    m_2ndVectorMode = SECOND_VECTOR_EMPTY;
    m_1stNullItemsBeginCount = 0;
    m_1stNullItemsMiddleCount = 0;
    m_2ndNullItemsCount = 0;
}

void OaVmaBlockMetadata_Linear::SetAllocationUserData(OaVmaAllocHandle allocHandle, void* userData)
{
    OaVmaSuballocation& suballoc = FindSuballocation((VkDeviceSize)allocHandle - 1);
    suballoc.userData = userData;
}

void OaVmaBlockMetadata_Linear::DebugLogAllAllocations() const
{
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    for (auto it = suballocations1st.begin() + m_1stNullItemsBeginCount; it != suballocations1st.end(); ++it)
        if (it->type != OA_VMA_SUBALLOCATION_TYPE_FREE)
            DebugLogAllocation(it->offset, it->size, it->userData);

    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    for (auto it = suballocations2nd.begin(); it != suballocations2nd.end(); ++it)
        if (it->type != OA_VMA_SUBALLOCATION_TYPE_FREE)
            DebugLogAllocation(it->offset, it->size, it->userData);
}

OaVmaSuballocation& OaVmaBlockMetadata_Linear::FindSuballocation(VkDeviceSize offset) const
{
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    OaVmaSuballocation refSuballoc;
    refSuballoc.offset = offset;
    // Rest of members stays uninitialized intentionally for better performance.

    // Item from the 1st vector.
    {
        SuballocationVectorType::const_iterator it = OaVmaBinaryFindSorted(
            suballocations1st.begin() + m_1stNullItemsBeginCount,
            suballocations1st.end(),
            refSuballoc,
            OaVmaSuballocationOffsetLess());
        if (it != suballocations1st.end())
        {
            return const_cast<OaVmaSuballocation&>(*it);
        }
    }

    if (m_2ndVectorMode != SECOND_VECTOR_EMPTY)
    {
        // Rest of members stays uninitialized intentionally for better performance.
        SuballocationVectorType::const_iterator it = m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER ?
            OaVmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, OaVmaSuballocationOffsetLess()) :
            OaVmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, OaVmaSuballocationOffsetGreater());
        if (it != suballocations2nd.end())
        {
            return const_cast<OaVmaSuballocation&>(*it);
        }
    }

    OA_VMA_ASSERT(0 && "Allocation not found in linear allocator!");
    return const_cast<OaVmaSuballocation&>(suballocations1st.back()); // Should never occur.
}

bool OaVmaBlockMetadata_Linear::ShouldCompact1st() const
{
    const size_t nullItemCount = m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount;
    const size_t suballocCount = AccessSuballocations1st().size();
    return suballocCount > 32 && nullItemCount * 2 >= (suballocCount - nullItemCount) * 3;
}

void OaVmaBlockMetadata_Linear::CleanupAfterFree()
{
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    if (IsEmpty())
    {
        suballocations1st.clear();
        suballocations2nd.clear();
        m_1stNullItemsBeginCount = 0;
        m_1stNullItemsMiddleCount = 0;
        m_2ndNullItemsCount = 0;
        m_2ndVectorMode = SECOND_VECTOR_EMPTY;
    }
    else
    {
        const size_t suballoc1stCount = suballocations1st.size();
        const size_t nullItem1stCount = m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount;
        OA_VMA_ASSERT(nullItem1stCount <= suballoc1stCount);

        // Find more null items at the beginning of 1st vector.
        while (m_1stNullItemsBeginCount < suballoc1stCount &&
            suballocations1st[m_1stNullItemsBeginCount].type == OA_VMA_SUBALLOCATION_TYPE_FREE)
        {
            ++m_1stNullItemsBeginCount;
            --m_1stNullItemsMiddleCount;
        }

        // Find more null items at the end of 1st vector.
        while (m_1stNullItemsMiddleCount > 0 &&
            suballocations1st.back().type == OA_VMA_SUBALLOCATION_TYPE_FREE)
        {
            --m_1stNullItemsMiddleCount;
            suballocations1st.pop_back();
        }

        // Find more null items at the end of 2nd vector.
        while (m_2ndNullItemsCount > 0 &&
            suballocations2nd.back().type == OA_VMA_SUBALLOCATION_TYPE_FREE)
        {
            --m_2ndNullItemsCount;
            suballocations2nd.pop_back();
        }

        // Find more null items at the beginning of 2nd vector.
        while (m_2ndNullItemsCount > 0 &&
            suballocations2nd[0].type == OA_VMA_SUBALLOCATION_TYPE_FREE)
        {
            --m_2ndNullItemsCount;
            OaVmaVectorRemove(suballocations2nd, 0);
        }

        if (ShouldCompact1st())
        {
            const size_t nonNullItemCount = suballoc1stCount - nullItem1stCount;
            size_t srcIndex = m_1stNullItemsBeginCount;
            for (size_t dstIndex = 0; dstIndex < nonNullItemCount; ++dstIndex)
            {
                while (suballocations1st[srcIndex].type == OA_VMA_SUBALLOCATION_TYPE_FREE)
                {
                    ++srcIndex;
                }
                if (dstIndex != srcIndex)
                {
                    suballocations1st[dstIndex] = suballocations1st[srcIndex];
                }
                ++srcIndex;
            }
            suballocations1st.resize(nonNullItemCount);
            m_1stNullItemsBeginCount = 0;
            m_1stNullItemsMiddleCount = 0;
        }

        // 2nd vector became empty.
        if (suballocations2nd.empty())
        {
            m_2ndVectorMode = SECOND_VECTOR_EMPTY;
        }

        // 1st vector became empty.
        if (suballocations1st.size() - m_1stNullItemsBeginCount == 0)
        {
            suballocations1st.clear();
            m_1stNullItemsBeginCount = 0;

            if (!suballocations2nd.empty() && m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
            {
                // Swap 1st with 2nd. Now 2nd is empty.
                m_2ndVectorMode = SECOND_VECTOR_EMPTY;
                m_1stNullItemsMiddleCount = m_2ndNullItemsCount;
                while (m_1stNullItemsBeginCount < suballocations2nd.size() &&
                    suballocations2nd[m_1stNullItemsBeginCount].type == OA_VMA_SUBALLOCATION_TYPE_FREE)
                {
                    ++m_1stNullItemsBeginCount;
                    --m_1stNullItemsMiddleCount;
                }
                m_2ndNullItemsCount = 0;
                m_1stVectorIndex ^= 1;
            }
        }
    }

    OA_VMA_HEAVY_ASSERT(Validate());
}

bool OaVmaBlockMetadata_Linear::CreateAllocationRequest_LowerAddress(
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    OaVmaSuballocationType allocType,
    uint32_t strategy,
    OaVmaAllocationRequest* pAllocationRequest)
{
    const VkDeviceSize blockSize = GetSize();
    const VkDeviceSize debugMargin = GetDebugMargin();
    const VkDeviceSize bufferImageGranularity = GetBufferImageGranularity();
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    if (m_2ndVectorMode == SECOND_VECTOR_EMPTY || m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        // Try to allocate at the end of 1st vector.

        VkDeviceSize resultBaseOffset = 0;
        if (!suballocations1st.empty())
        {
            const OaVmaSuballocation& lastSuballoc = suballocations1st.back();
            resultBaseOffset = lastSuballoc.offset + lastSuballoc.size + debugMargin;
        }

        // Start from offset equal to beginning of free space.
        VkDeviceSize resultOffset = resultBaseOffset;

        // Apply alignment.
        resultOffset = OaVmaAlignUp(resultOffset, allocAlignment);

        // Check previous suballocations for BufferImageGranularity conflicts.
        // Make bigger alignment if necessary.
        if (bufferImageGranularity > 1 && bufferImageGranularity != allocAlignment && !suballocations1st.empty())
        {
            bool bufferImageGranularityConflict = false;
            for (size_t prevSuballocIndex = suballocations1st.size(); prevSuballocIndex--; )
            {
                const OaVmaSuballocation& prevSuballoc = suballocations1st[prevSuballocIndex];
                if (OaVmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, resultOffset, bufferImageGranularity))
                {
                    if (OaVmaIsBufferImageGranularityConflict(prevSuballoc.type, allocType))
                    {
                        bufferImageGranularityConflict = true;
                        break;
                    }
                }
                else
                    // Already on previous page.
                    break;
            }
            if (bufferImageGranularityConflict)
            {
                resultOffset = OaVmaAlignUp(resultOffset, bufferImageGranularity);
            }
        }

        const VkDeviceSize freeSpaceEnd = m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK ?
            suballocations2nd.back().offset : blockSize;

        // There is enough free space at the end after alignment.
        if (resultOffset + allocSize + debugMargin <= freeSpaceEnd)
        {
            // Check next suballocations for BufferImageGranularity conflicts.
            // If conflict exists, allocation cannot be made here.
            if ((allocSize % bufferImageGranularity || resultOffset % bufferImageGranularity) && m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
            {
                for (size_t nextSuballocIndex = suballocations2nd.size(); nextSuballocIndex--; )
                {
                    const OaVmaSuballocation& nextSuballoc = suballocations2nd[nextSuballocIndex];
                    if (OaVmaBlocksOnSamePage(resultOffset, allocSize, nextSuballoc.offset, bufferImageGranularity))
                    {
                        if (OaVmaIsBufferImageGranularityConflict(allocType, nextSuballoc.type))
                        {
                            return false;
                        }
                    }
                    else
                    {
                        // Already on previous page.
                        break;
                    }
                }
            }

            // All tests passed: Success.
            pAllocationRequest->allocHandle = (OaVmaAllocHandle)(resultOffset + 1);
            // pAllocationRequest->item, customData unused.
            pAllocationRequest->type = OaVmaAllocationRequestType::EndOf1st;
            return true;
        }
    }

    // Wrap-around to end of 2nd vector. Try to allocate there, watching for the
    // beginning of 1st vector as the end of free space.
    if (m_2ndVectorMode == SECOND_VECTOR_EMPTY || m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        OA_VMA_ASSERT(!suballocations1st.empty());

        VkDeviceSize resultBaseOffset = 0;
        if (!suballocations2nd.empty())
        {
            const OaVmaSuballocation& lastSuballoc = suballocations2nd.back();
            resultBaseOffset = lastSuballoc.offset + lastSuballoc.size + debugMargin;
        }

        // Start from offset equal to beginning of free space.
        VkDeviceSize resultOffset = resultBaseOffset;

        // Apply alignment.
        resultOffset = OaVmaAlignUp(resultOffset, allocAlignment);

        // Check previous suballocations for BufferImageGranularity conflicts.
        // Make bigger alignment if necessary.
        if (bufferImageGranularity > 1 && bufferImageGranularity != allocAlignment && !suballocations2nd.empty())
        {
            bool bufferImageGranularityConflict = false;
            for (size_t prevSuballocIndex = suballocations2nd.size(); prevSuballocIndex--; )
            {
                const OaVmaSuballocation& prevSuballoc = suballocations2nd[prevSuballocIndex];
                if (OaVmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, resultOffset, bufferImageGranularity))
                {
                    if (OaVmaIsBufferImageGranularityConflict(prevSuballoc.type, allocType))
                    {
                        bufferImageGranularityConflict = true;
                        break;
                    }
                }
                else
                    // Already on previous page.
                    break;
            }
            if (bufferImageGranularityConflict)
            {
                resultOffset = OaVmaAlignUp(resultOffset, bufferImageGranularity);
            }
        }

        size_t index1st = m_1stNullItemsBeginCount;

        // There is enough free space at the end after alignment.
        if ((index1st == suballocations1st.size() && resultOffset + allocSize + debugMargin <= blockSize) ||
            (index1st < suballocations1st.size() && resultOffset + allocSize + debugMargin <= suballocations1st[index1st].offset))
        {
            // Check next suballocations for BufferImageGranularity conflicts.
            // If conflict exists, allocation cannot be made here.
            if (allocSize % bufferImageGranularity || resultOffset % bufferImageGranularity)
            {
                for (size_t nextSuballocIndex = index1st;
                    nextSuballocIndex < suballocations1st.size();
                    nextSuballocIndex++)
                {
                    const OaVmaSuballocation& nextSuballoc = suballocations1st[nextSuballocIndex];
                    if (OaVmaBlocksOnSamePage(resultOffset, allocSize, nextSuballoc.offset, bufferImageGranularity))
                    {
                        if (OaVmaIsBufferImageGranularityConflict(allocType, nextSuballoc.type))
                        {
                            return false;
                        }
                    }
                    else
                    {
                        // Already on next page.
                        break;
                    }
                }
            }

            // All tests passed: Success.
            pAllocationRequest->allocHandle = (OaVmaAllocHandle)(resultOffset + 1);
            pAllocationRequest->type = OaVmaAllocationRequestType::EndOf2nd;
            // pAllocationRequest->item, customData unused.
            return true;
        }
    }

    return false;
}

bool OaVmaBlockMetadata_Linear::CreateAllocationRequest_UpperAddress(
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    OaVmaSuballocationType allocType,
    uint32_t strategy,
    OaVmaAllocationRequest* pAllocationRequest)
{
    const VkDeviceSize blockSize = GetSize();
    const VkDeviceSize bufferImageGranularity = GetBufferImageGranularity();
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        OA_VMA_ASSERT(0 && "Trying to use pool with linear algorithm as double stack, while it is already being used as ring buffer.");
        return false;
    }

    // Try to allocate before 2nd.back(), or end of block if 2nd.empty().
    if (allocSize > blockSize)
    {
        return false;
    }
    VkDeviceSize resultBaseOffset = blockSize - allocSize;
    if (!suballocations2nd.empty())
    {
        const OaVmaSuballocation& lastSuballoc = suballocations2nd.back();
        resultBaseOffset = lastSuballoc.offset - allocSize;
        if (allocSize > lastSuballoc.offset)
        {
            return false;
        }
    }

    // Start from offset equal to end of free space.
    VkDeviceSize resultOffset = resultBaseOffset;

    const VkDeviceSize debugMargin = GetDebugMargin();

    // Apply debugMargin at the end.
    if (debugMargin > 0)
    {
        if (resultOffset < debugMargin)
        {
            return false;
        }
        resultOffset -= debugMargin;
    }

    // Apply alignment.
    resultOffset = OaVmaAlignDown(resultOffset, allocAlignment);

    // Check next suballocations from 2nd for BufferImageGranularity conflicts.
    // Make bigger alignment if necessary.
    if (bufferImageGranularity > 1 && bufferImageGranularity != allocAlignment && !suballocations2nd.empty())
    {
        bool bufferImageGranularityConflict = false;
        for (size_t nextSuballocIndex = suballocations2nd.size(); nextSuballocIndex--; )
        {
            const OaVmaSuballocation& nextSuballoc = suballocations2nd[nextSuballocIndex];
            if (OaVmaBlocksOnSamePage(resultOffset, allocSize, nextSuballoc.offset, bufferImageGranularity))
            {
                if (OaVmaIsBufferImageGranularityConflict(nextSuballoc.type, allocType))
                {
                    bufferImageGranularityConflict = true;
                    break;
                }
            }
            else
                // Already on previous page.
                break;
        }
        if (bufferImageGranularityConflict)
        {
            resultOffset = OaVmaAlignDown(resultOffset, bufferImageGranularity);
        }
    }

    // There is enough free space.
    const VkDeviceSize endOf1st = !suballocations1st.empty() ?
        suballocations1st.back().offset + suballocations1st.back().size :
        0;
    if (endOf1st + debugMargin <= resultOffset)
    {
        // Check previous suballocations for BufferImageGranularity conflicts.
        // If conflict exists, allocation cannot be made here.
        if (bufferImageGranularity > 1)
        {
            for (size_t prevSuballocIndex = suballocations1st.size(); prevSuballocIndex--; )
            {
                const OaVmaSuballocation& prevSuballoc = suballocations1st[prevSuballocIndex];
                if (OaVmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, resultOffset, bufferImageGranularity))
                {
                    if (OaVmaIsBufferImageGranularityConflict(allocType, prevSuballoc.type))
                    {
                        return false;
                    }
                }
                else
                {
                    // Already on next page.
                    break;
                }
            }
        }

        // All tests passed: Success.
        pAllocationRequest->allocHandle = (OaVmaAllocHandle)(resultOffset + 1);
        // pAllocationRequest->item unused.
        pAllocationRequest->type = OaVmaAllocationRequestType::UpperAddress;
        return true;
    }

    return false;
}
#endif // _OA_VMA_BLOCK_METADATA_LINEAR_FUNCTIONS
#endif // _OA_VMA_BLOCK_METADATA_LINEAR

#ifndef _OA_VMA_BLOCK_METADATA_TLSF
// To not search current larger region if first allocation won't succeed and skip to smaller range
// use with OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT as strategy in CreateAllocationRequest().
// When fragmentation and reusal of previous blocks doesn't matter then use with
// OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT for fastest alloc time possible.
class OaVmaBlockMetadata_TLSF : public OaVmaBlockMetadata
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaBlockMetadata_TLSF)
public:
    OaVmaBlockMetadata_TLSF(const VkAllocationCallbacks* pAllocationCallbacks,
        VkDeviceSize bufferImageGranularity, bool isVirtual);
    ~OaVmaBlockMetadata_TLSF() override;

    size_t GetAllocationCount() const override { return m_AllocCount; }
    size_t GetFreeRegionsCount() const override { return m_BlocksFreeCount + 1; }
    VkDeviceSize GetSumFreeSize() const override { return m_BlocksFreeSize + m_NullBlock->size; }
    bool IsEmpty() const override { return m_NullBlock->offset == 0; }
    VkDeviceSize GetAllocationOffset(OaVmaAllocHandle allocHandle) const override { return ((Block*)allocHandle)->offset; }

    void Init(VkDeviceSize size) override;
    bool Validate() const override;

    void AddDetailedStatistics(OaVmaDetailedStatistics& inoutStats) const override;
    void AddStatistics(OaVmaStatistics& inoutStats) const override;

#if OA_VMA_STATS_STRING_ENABLED
    void PrintDetailedMap(class OaVmaJsonWriter& json) const override;
#endif

    bool CreateAllocationRequest(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        bool upperAddress,
        OaVmaSuballocationType allocType,
        uint32_t strategy,
        OaVmaAllocationRequest* pAllocationRequest) override;

    VkResult CheckCorruption(const void* pBlockData) override;
    void Alloc(
        const OaVmaAllocationRequest& request,
        OaVmaSuballocationType type,
        void* userData) override;

    void Free(OaVmaAllocHandle allocHandle) override;
    void GetAllocationInfo(OaVmaAllocHandle allocHandle, OaVmaVirtualAllocationInfo& outInfo) override;
    void* GetAllocationUserData(OaVmaAllocHandle allocHandle) const override;
    OaVmaAllocHandle GetAllocationListBegin() const override;
    OaVmaAllocHandle GetNextAllocation(OaVmaAllocHandle prevAlloc) const override;
    VkDeviceSize GetNextFreeRegionSize(OaVmaAllocHandle alloc) const override;
    void Clear() override;
    void SetAllocationUserData(OaVmaAllocHandle allocHandle, void* userData) override;
    void DebugLogAllAllocations() const override;

private:
    // According to original paper it should be preferable 4 or 5:
    // M. Masmano, I. Ripoll, A. Crespo, and J. Real "TLSF: a New Dynamic Memory Allocator for Real-Time Systems"
    // http://www.gii.upv.es/tlsf/files/ecrts04_tlsf.pdf
    static constexpr uint8_t SECOND_LEVEL_INDEX = 5;
    static constexpr uint16_t SMALL_BUFFER_SIZE = 256;
    static constexpr uint32_t INITIAL_BLOCK_ALLOC_COUNT = 16;
    static constexpr uint8_t MEMORY_CLASS_SHIFT = 7;
    static constexpr uint8_t MAX_MEMORY_CLASSES = 65 - MEMORY_CLASS_SHIFT;

    class Block
    {
    public:
        VkDeviceSize offset;
        VkDeviceSize size;
        Block* prevPhysical;
        Block* nextPhysical;

        void MarkFree() { prevFree = OA_VMA_NULL; }
        void MarkTaken() { prevFree = this; }
        bool IsFree() const { return prevFree != this; }
        void*& UserData() { OA_VMA_HEAVY_ASSERT(!IsFree()); return userData; }
        Block*& PrevFree() { return prevFree; }
        Block*& NextFree() { OA_VMA_HEAVY_ASSERT(IsFree()); return nextFree; }

    private:
        Block* prevFree; // Address of the same block here indicates that block is taken
        union
        {
            Block* nextFree;
            void* userData;
        };
    };

    size_t m_AllocCount;
    // Total number of free blocks besides null block
    size_t m_BlocksFreeCount;
    // Total size of free blocks excluding null block
    VkDeviceSize m_BlocksFreeSize;
    uint32_t m_IsFreeBitmap;
    uint8_t m_MemoryClasses;
    uint32_t m_InnerIsFreeBitmap[MAX_MEMORY_CLASSES];
    uint32_t m_ListsCount;
    /*
    * 0: 0-3 lists for small buffers
    * 1+: 0-(2^SLI-1) lists for normal buffers
    */
    Block** m_FreeList;
    OaVmaPoolAllocator<Block> m_BlockAllocator;
    Block* m_NullBlock;
    OaVmaBlockBufferImageGranularity m_GranularityHandler;

    static uint8_t SizeToMemoryClass(VkDeviceSize size);
    uint16_t SizeToSecondIndex(VkDeviceSize size, uint8_t memoryClass) const;
    uint32_t GetListIndex(uint8_t memoryClass, uint16_t secondIndex) const;
    uint32_t GetListIndex(VkDeviceSize size) const;

    void RemoveFreeBlock(Block* block);
    void InsertFreeBlock(Block* block);
    void MergeBlock(Block* block, Block* prev);

    Block* FindFreeBlock(VkDeviceSize size, uint32_t& listIndex) const;
    bool CheckBlock(
        Block& block,
        uint32_t listIndex,
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        OaVmaSuballocationType allocType,
        OaVmaAllocationRequest* pAllocationRequest);
};

#ifndef _OA_VMA_BLOCK_METADATA_TLSF_FUNCTIONS
OaVmaBlockMetadata_TLSF::OaVmaBlockMetadata_TLSF(const VkAllocationCallbacks* pAllocationCallbacks,
    VkDeviceSize bufferImageGranularity, bool isVirtual)
    : OaVmaBlockMetadata(pAllocationCallbacks, bufferImageGranularity, isVirtual),
    m_AllocCount(0),
    m_BlocksFreeCount(0),
    m_BlocksFreeSize(0),
    m_IsFreeBitmap(0),
    m_MemoryClasses(0),
    m_ListsCount(0),
    m_FreeList(OA_VMA_NULL),
    m_BlockAllocator(pAllocationCallbacks, INITIAL_BLOCK_ALLOC_COUNT),
    m_NullBlock(OA_VMA_NULL),
    m_GranularityHandler(bufferImageGranularity) {}

OaVmaBlockMetadata_TLSF::~OaVmaBlockMetadata_TLSF()
{
    if (m_FreeList)
        OaVma_delete_array(GetAllocationCallbacks(), m_FreeList, m_ListsCount);
    m_GranularityHandler.Destroy(GetAllocationCallbacks());
}

void OaVmaBlockMetadata_TLSF::Init(VkDeviceSize size)
{
    OaVmaBlockMetadata::Init(size);

    if (!IsVirtual())
        m_GranularityHandler.Init(GetAllocationCallbacks(), size);

    m_NullBlock = m_BlockAllocator.Alloc();
    m_NullBlock->size = size;
    m_NullBlock->offset = 0;
    m_NullBlock->prevPhysical = OA_VMA_NULL;
    m_NullBlock->nextPhysical = OA_VMA_NULL;
    m_NullBlock->MarkFree();
    m_NullBlock->NextFree() = OA_VMA_NULL;
    m_NullBlock->PrevFree() = OA_VMA_NULL;
    uint8_t memoryClass = SizeToMemoryClass(size);
    uint16_t sli = SizeToSecondIndex(size, memoryClass);
    m_ListsCount = (memoryClass == 0 ? 0 : (memoryClass - 1) * (1UL << SECOND_LEVEL_INDEX) + sli) + 1;
    if (IsVirtual())
        m_ListsCount += 1UL << SECOND_LEVEL_INDEX;
    else
        m_ListsCount += 4;

    m_MemoryClasses = memoryClass + uint8_t(2);
    memset(m_InnerIsFreeBitmap, 0, MAX_MEMORY_CLASSES * sizeof(uint32_t));

    m_FreeList = OaVma_new_array(GetAllocationCallbacks(), Block*, m_ListsCount);
    memset(m_FreeList, 0, m_ListsCount * sizeof(Block*));
}

bool OaVmaBlockMetadata_TLSF::Validate() const
{
    OA_VMA_VALIDATE(GetSumFreeSize() <= GetSize());

    VkDeviceSize calculatedSize = m_NullBlock->size;
    VkDeviceSize calculatedFreeSize = m_NullBlock->size;
    size_t allocCount = 0;
    size_t freeCount = 0;

    // Check integrity of free lists
    for (uint32_t list = 0; list < m_ListsCount; ++list)
    {
        Block* block = m_FreeList[list];
        if (block != OA_VMA_NULL)
        {
            OA_VMA_VALIDATE(block->IsFree());
            OA_VMA_VALIDATE(block->PrevFree() == OA_VMA_NULL);
            while (block->NextFree())
            {
                OA_VMA_VALIDATE(block->NextFree()->IsFree());
                OA_VMA_VALIDATE(block->NextFree()->PrevFree() == block);
                block = block->NextFree();
            }
        }
    }

    VkDeviceSize nextOffset = m_NullBlock->offset;
    auto validateCtx = m_GranularityHandler.StartValidation(GetAllocationCallbacks(), IsVirtual());

    OA_VMA_VALIDATE(m_NullBlock->nextPhysical == OA_VMA_NULL);
    if (m_NullBlock->prevPhysical)
    {
        OA_VMA_VALIDATE(m_NullBlock->prevPhysical->nextPhysical == m_NullBlock);
    }
    // Check all blocks
    for (Block* prev = m_NullBlock->prevPhysical; prev != OA_VMA_NULL; prev = prev->prevPhysical)
    {
        OA_VMA_VALIDATE(prev->offset + prev->size == nextOffset);
        nextOffset = prev->offset;
        calculatedSize += prev->size;

        uint32_t listIndex = GetListIndex(prev->size);
        if (prev->IsFree())
        {
            ++freeCount;
            // Check if free block belongs to free list
            Block* freeBlock = m_FreeList[listIndex];
            OA_VMA_VALIDATE(freeBlock != OA_VMA_NULL);

            bool found = false;
            do
            {
                if (freeBlock == prev)
                    found = true;

                freeBlock = freeBlock->NextFree();
            } while (!found && freeBlock != OA_VMA_NULL);

            OA_VMA_VALIDATE(found);
            calculatedFreeSize += prev->size;
        }
        else
        {
            ++allocCount;
            // Check if taken block is not on a free list
            Block* freeBlock = m_FreeList[listIndex];
            while (freeBlock)
            {
                OA_VMA_VALIDATE(freeBlock != prev);
                freeBlock = freeBlock->NextFree();
            }

            if (!IsVirtual())
            {
                OA_VMA_VALIDATE(m_GranularityHandler.Validate(validateCtx, prev->offset, prev->size));
            }
        }

        if (prev->prevPhysical)
        {
            OA_VMA_VALIDATE(prev->prevPhysical->nextPhysical == prev);
        }
    }

    if (!IsVirtual())
    {
        OA_VMA_VALIDATE(m_GranularityHandler.FinishValidation(validateCtx));
    }

    OA_VMA_VALIDATE(nextOffset == 0);
    OA_VMA_VALIDATE(calculatedSize == GetSize());
    OA_VMA_VALIDATE(calculatedFreeSize == GetSumFreeSize());
    OA_VMA_VALIDATE(allocCount == m_AllocCount);
    OA_VMA_VALIDATE(freeCount == m_BlocksFreeCount);

    return true;
}

void OaVmaBlockMetadata_TLSF::AddDetailedStatistics(OaVmaDetailedStatistics& inoutStats) const
{
    inoutStats.statistics.blockCount++;
    inoutStats.statistics.blockBytes += GetSize();
    if (m_NullBlock->size > 0)
        OaVmaAddDetailedStatisticsUnusedRange(inoutStats, m_NullBlock->size);

    for (Block* block = m_NullBlock->prevPhysical; block != OA_VMA_NULL; block = block->prevPhysical)
    {
        if (block->IsFree())
            OaVmaAddDetailedStatisticsUnusedRange(inoutStats, block->size);
        else
            OaVmaAddDetailedStatisticsAllocation(inoutStats, block->size);
    }
}

void OaVmaBlockMetadata_TLSF::AddStatistics(OaVmaStatistics& inoutStats) const
{
    inoutStats.blockCount++;
    inoutStats.allocationCount += (uint32_t)m_AllocCount;
    inoutStats.blockBytes += GetSize();
    inoutStats.allocationBytes += GetSize() - GetSumFreeSize();
}

#if OA_VMA_STATS_STRING_ENABLED
void OaVmaBlockMetadata_TLSF::PrintDetailedMap(class OaVmaJsonWriter& json) const
{
    size_t blockCount = m_AllocCount + m_BlocksFreeCount;
    OaVmaStlAllocator<Block*> allocator(GetAllocationCallbacks());
    OaVmaVector<Block*, OaVmaStlAllocator<Block*>> blockList(blockCount, allocator);

    size_t i = blockCount;
    for (Block* block = m_NullBlock->prevPhysical; block != OA_VMA_NULL; block = block->prevPhysical)
    {
        blockList[--i] = block;
    }
    OA_VMA_ASSERT(i == 0);

    OaVmaDetailedStatistics stats;
    OaVmaClearDetailedStatistics(stats);
    AddDetailedStatistics(stats);

    PrintDetailedMap_Begin(json,
        stats.statistics.blockBytes - stats.statistics.allocationBytes,
        stats.statistics.allocationCount,
        stats.unusedRangeCount);

    for (; i < blockCount; ++i)
    {
        Block* block = blockList[i];
        if (block->IsFree())
            PrintDetailedMap_UnusedRange(json, block->offset, block->size);
        else
            PrintDetailedMap_Allocation(json, block->offset, block->size, block->UserData());
    }
    if (m_NullBlock->size > 0)
        PrintDetailedMap_UnusedRange(json, m_NullBlock->offset, m_NullBlock->size);

    PrintDetailedMap_End(json);
}
#endif

bool OaVmaBlockMetadata_TLSF::CreateAllocationRequest(
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    bool upperAddress,
    OaVmaSuballocationType allocType,
    uint32_t strategy,
    OaVmaAllocationRequest* pAllocationRequest)
{
    OA_VMA_ASSERT(allocSize > 0 && "Cannot allocate empty block!");
    OA_VMA_ASSERT(!upperAddress && "OA_VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT can be used only with linear algorithm.");

    // For small granularity round up
    if (!IsVirtual())
        m_GranularityHandler.RoundupAllocRequest(allocType, allocSize, allocAlignment);

    allocSize += GetDebugMargin();
    // Quick check for too small pool
    if (allocSize > GetSumFreeSize())
        return false;

    // If no free blocks in pool then check only null block
    if (m_BlocksFreeCount == 0)
        return CheckBlock(*m_NullBlock, m_ListsCount, allocSize, allocAlignment, allocType, pAllocationRequest);

    // Round up to the next block
    VkDeviceSize sizeForNextList = allocSize;
    VkDeviceSize smallSizeStep = VkDeviceSize(SMALL_BUFFER_SIZE / (IsVirtual() ? 1U << SECOND_LEVEL_INDEX : 4U));
    if (allocSize > SMALL_BUFFER_SIZE)
    {
        sizeForNextList += (1ULL << (OA_VMA_BITSCAN_MSB(allocSize) - SECOND_LEVEL_INDEX));
    }
    else if (allocSize > SMALL_BUFFER_SIZE - smallSizeStep)
        sizeForNextList = SMALL_BUFFER_SIZE + 1;
    else
        sizeForNextList += smallSizeStep;

    uint32_t nextListIndex = m_ListsCount;
    uint32_t prevListIndex = m_ListsCount;
    Block* nextListBlock = OA_VMA_NULL;
    Block* prevListBlock = OA_VMA_NULL;

    // Check blocks according to strategies
    if (strategy & OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT)
    {
        // Quick check for larger block first
        nextListBlock = FindFreeBlock(sizeForNextList, nextListIndex);
        if (nextListBlock != OA_VMA_NULL && CheckBlock(*nextListBlock, nextListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
            return true;

        // If not fitted then null block
        if (CheckBlock(*m_NullBlock, m_ListsCount, allocSize, allocAlignment, allocType, pAllocationRequest))
            return true;

        // Null block failed, search larger bucket
        while (nextListBlock)
        {
            if (CheckBlock(*nextListBlock, nextListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            nextListBlock = nextListBlock->NextFree();
        }

        // Failed again, check best fit bucket
        prevListBlock = FindFreeBlock(allocSize, prevListIndex);
        while (prevListBlock)
        {
            if (CheckBlock(*prevListBlock, prevListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            prevListBlock = prevListBlock->NextFree();
        }
    }
    else if (strategy & OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT)
    {
        // Check best fit bucket
        prevListBlock = FindFreeBlock(allocSize, prevListIndex);
        while (prevListBlock)
        {
            if (CheckBlock(*prevListBlock, prevListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            prevListBlock = prevListBlock->NextFree();
        }

        // If failed check null block
        if (CheckBlock(*m_NullBlock, m_ListsCount, allocSize, allocAlignment, allocType, pAllocationRequest))
            return true;

        // Check larger bucket
        nextListBlock = FindFreeBlock(sizeForNextList, nextListIndex);
        while (nextListBlock)
        {
            if (CheckBlock(*nextListBlock, nextListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            nextListBlock = nextListBlock->NextFree();
        }
    }
    else if (strategy & OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT )
    {
        // Perform search from the start
        OaVmaStlAllocator<Block*> allocator(GetAllocationCallbacks());
        OaVmaVector<Block*, OaVmaStlAllocator<Block*>> blockList(m_BlocksFreeCount, allocator);

        size_t i = m_BlocksFreeCount;
        for (Block* block = m_NullBlock->prevPhysical; block != OA_VMA_NULL; block = block->prevPhysical)
        {
            if (block->IsFree() && block->size >= allocSize)
                blockList[--i] = block;
        }

        for (; i < m_BlocksFreeCount; ++i)
        {
            Block& block = *blockList[i];
            if (CheckBlock(block, GetListIndex(block.size), allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
        }

        // If failed check null block
        if (CheckBlock(*m_NullBlock, m_ListsCount, allocSize, allocAlignment, allocType, pAllocationRequest))
            return true;

        // Whole range searched, no more memory
        return false;
    }
    else
    {
        // Check larger bucket
        nextListBlock = FindFreeBlock(sizeForNextList, nextListIndex);
        while (nextListBlock)
        {
            if (CheckBlock(*nextListBlock, nextListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            nextListBlock = nextListBlock->NextFree();
        }

        // If failed check null block
        if (CheckBlock(*m_NullBlock, m_ListsCount, allocSize, allocAlignment, allocType, pAllocationRequest))
            return true;

        // Check best fit bucket
        prevListBlock = FindFreeBlock(allocSize, prevListIndex);
        while (prevListBlock)
        {
            if (CheckBlock(*prevListBlock, prevListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            prevListBlock = prevListBlock->NextFree();
        }
    }

    // Worst case, full search has to be done
    while (++nextListIndex < m_ListsCount)
    {
        nextListBlock = m_FreeList[nextListIndex];
        while (nextListBlock)
        {
            if (CheckBlock(*nextListBlock, nextListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            nextListBlock = nextListBlock->NextFree();
        }
    }

    // No more memory sadly
    return false;
}

VkResult OaVmaBlockMetadata_TLSF::CheckCorruption(const void* pBlockData)
{
    for (Block* block = m_NullBlock->prevPhysical; block != OA_VMA_NULL; block = block->prevPhysical)
    {
        if (!block->IsFree())
        {
            if (!OaVmaValidateMagicValue(pBlockData, block->offset + block->size))
            {
                OA_VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER VALIDATED ALLOCATION!");
                return VK_ERROR_UNKNOWN_COPY;
            }
        }
    }

    return VK_SUCCESS;
}

void OaVmaBlockMetadata_TLSF::Alloc(
    const OaVmaAllocationRequest& request,
    OaVmaSuballocationType type,
    void* userData)
{
    OA_VMA_ASSERT(request.type == OaVmaAllocationRequestType::TLSF);

    // Get block and pop it from the free list
    Block* currentBlock = (Block*)request.allocHandle;
    VkDeviceSize offset = request.algorithmData;
    OA_VMA_ASSERT(currentBlock != OA_VMA_NULL);
    OA_VMA_ASSERT(currentBlock->offset <= offset);

    if (currentBlock != m_NullBlock)
        RemoveFreeBlock(currentBlock);

    VkDeviceSize debugMargin = GetDebugMargin();
    VkDeviceSize missingAlignment = offset - currentBlock->offset;

    // Append missing alignment to prev block or create new one
    if (missingAlignment)
    {
        Block* prevBlock = currentBlock->prevPhysical;
        OA_VMA_ASSERT(prevBlock != OA_VMA_NULL && "There should be no missing alignment at offset 0!");

        if (prevBlock->IsFree() && prevBlock->size != debugMargin)
        {
            uint32_t oldList = GetListIndex(prevBlock->size);
            prevBlock->size += missingAlignment;
            // Check if new size crosses list bucket
            if (oldList != GetListIndex(prevBlock->size))
            {
                prevBlock->size -= missingAlignment;
                RemoveFreeBlock(prevBlock);
                prevBlock->size += missingAlignment;
                InsertFreeBlock(prevBlock);
            }
            else
                m_BlocksFreeSize += missingAlignment;
        }
        else
        {
            Block* newBlock = m_BlockAllocator.Alloc();
            currentBlock->prevPhysical = newBlock;
            prevBlock->nextPhysical = newBlock;
            newBlock->prevPhysical = prevBlock;
            newBlock->nextPhysical = currentBlock;
            newBlock->size = missingAlignment;
            newBlock->offset = currentBlock->offset;
            newBlock->MarkTaken();

            InsertFreeBlock(newBlock);
        }

        currentBlock->size -= missingAlignment;
        currentBlock->offset += missingAlignment;
    }

    VkDeviceSize size = request.size + debugMargin;
    if (currentBlock->size == size)
    {
        if (currentBlock == m_NullBlock)
        {
            // Setup new null block
            m_NullBlock = m_BlockAllocator.Alloc();
            m_NullBlock->size = 0;
            m_NullBlock->offset = currentBlock->offset + size;
            m_NullBlock->prevPhysical = currentBlock;
            m_NullBlock->nextPhysical = OA_VMA_NULL;
            m_NullBlock->MarkFree();
            m_NullBlock->PrevFree() = OA_VMA_NULL;
            m_NullBlock->NextFree() = OA_VMA_NULL;
            currentBlock->nextPhysical = m_NullBlock;
            currentBlock->MarkTaken();
        }
    }
    else
    {
        OA_VMA_ASSERT(currentBlock->size > size && "Proper block already found, shouldn't find smaller one!");

        // Create new free block
        Block* newBlock = m_BlockAllocator.Alloc();
        newBlock->size = currentBlock->size - size;
        newBlock->offset = currentBlock->offset + size;
        newBlock->prevPhysical = currentBlock;
        newBlock->nextPhysical = currentBlock->nextPhysical;
        currentBlock->nextPhysical = newBlock;
        currentBlock->size = size;

        if (currentBlock == m_NullBlock)
        {
            m_NullBlock = newBlock;
            m_NullBlock->MarkFree();
            m_NullBlock->NextFree() = OA_VMA_NULL;
            m_NullBlock->PrevFree() = OA_VMA_NULL;
            currentBlock->MarkTaken();
        }
        else
        {
            newBlock->nextPhysical->prevPhysical = newBlock;
            newBlock->MarkTaken();
            InsertFreeBlock(newBlock);
        }
    }
    currentBlock->UserData() = userData;

    if (debugMargin > 0)
    {
        currentBlock->size -= debugMargin;
        Block* newBlock = m_BlockAllocator.Alloc();
        newBlock->size = debugMargin;
        newBlock->offset = currentBlock->offset + currentBlock->size;
        newBlock->prevPhysical = currentBlock;
        newBlock->nextPhysical = currentBlock->nextPhysical;
        newBlock->MarkTaken();
        currentBlock->nextPhysical->prevPhysical = newBlock;
        currentBlock->nextPhysical = newBlock;
        InsertFreeBlock(newBlock);
    }

    if (!IsVirtual())
        m_GranularityHandler.AllocPages((uint8_t)(uintptr_t)request.customData,
            currentBlock->offset, currentBlock->size);
    ++m_AllocCount;
}

void OaVmaBlockMetadata_TLSF::Free(OaVmaAllocHandle allocHandle)
{
    Block* block = (Block*)allocHandle;
    Block* next = block->nextPhysical;
    OA_VMA_ASSERT(!block->IsFree() && "Block is already free!");

    if (!IsVirtual())
        m_GranularityHandler.FreePages(block->offset, block->size);
    --m_AllocCount;

    VkDeviceSize debugMargin = GetDebugMargin();
    if (debugMargin > 0)
    {
        RemoveFreeBlock(next);
        MergeBlock(next, block);
        block = next;
        next = next->nextPhysical;
    }

    // Try merging
    Block* prev = block->prevPhysical;
    if (prev != OA_VMA_NULL && prev->IsFree() && prev->size != debugMargin)
    {
        RemoveFreeBlock(prev);
        MergeBlock(block, prev);
    }

    if (!next->IsFree())
        InsertFreeBlock(block);
    else if (next == m_NullBlock)
        MergeBlock(m_NullBlock, block);
    else
    {
        RemoveFreeBlock(next);
        MergeBlock(next, block);
        InsertFreeBlock(next);
    }
}

void OaVmaBlockMetadata_TLSF::GetAllocationInfo(OaVmaAllocHandle allocHandle, OaVmaVirtualAllocationInfo& outInfo)
{
    Block* block = (Block*)allocHandle;
    OA_VMA_ASSERT(!block->IsFree() && "Cannot get allocation info for free block!");
    outInfo.offset = block->offset;
    outInfo.size = block->size;
    outInfo.pUserData = block->UserData();
}

void* OaVmaBlockMetadata_TLSF::GetAllocationUserData(OaVmaAllocHandle allocHandle) const
{
    Block* block = (Block*)allocHandle;
    OA_VMA_ASSERT(!block->IsFree() && "Cannot get user data for free block!");
    return block->UserData();
}

OaVmaAllocHandle OaVmaBlockMetadata_TLSF::GetAllocationListBegin() const
{
    if (m_AllocCount == 0)
        return VK_NULL_HANDLE;

    for (Block* block = m_NullBlock->prevPhysical; block; block = block->prevPhysical)
    {
        if (!block->IsFree())
            return (OaVmaAllocHandle)block;
    }
    OA_VMA_ASSERT(false && "If m_AllocCount > 0 then should find any allocation!");
    return VK_NULL_HANDLE;
}

OaVmaAllocHandle OaVmaBlockMetadata_TLSF::GetNextAllocation(OaVmaAllocHandle prevAlloc) const
{
    Block* startBlock = (Block*)prevAlloc;
    OA_VMA_ASSERT(!startBlock->IsFree() && "Incorrect block!");

    for (Block* block = startBlock->prevPhysical; block; block = block->prevPhysical)
    {
        if (!block->IsFree())
            return (OaVmaAllocHandle)block;
    }
    return VK_NULL_HANDLE;
}

VkDeviceSize OaVmaBlockMetadata_TLSF::GetNextFreeRegionSize(OaVmaAllocHandle alloc) const
{
    Block* block = (Block*)alloc;
    OA_VMA_ASSERT(!block->IsFree() && "Incorrect block!");

    if (block->prevPhysical)
        return block->prevPhysical->IsFree() ? block->prevPhysical->size : 0;
    return 0;
}

void OaVmaBlockMetadata_TLSF::Clear()
{
    m_AllocCount = 0;
    m_BlocksFreeCount = 0;
    m_BlocksFreeSize = 0;
    m_IsFreeBitmap = 0;
    m_NullBlock->offset = 0;
    m_NullBlock->size = GetSize();
    Block* block = m_NullBlock->prevPhysical;
    m_NullBlock->prevPhysical = OA_VMA_NULL;
    while (block)
    {
        Block* prev = block->prevPhysical;
        m_BlockAllocator.Free(block);
        block = prev;
    }
    memset(m_FreeList, 0, m_ListsCount * sizeof(Block*));
    memset(m_InnerIsFreeBitmap, 0, m_MemoryClasses * sizeof(uint32_t));
    m_GranularityHandler.Clear();
}

void OaVmaBlockMetadata_TLSF::SetAllocationUserData(OaVmaAllocHandle allocHandle, void* userData)
{
    Block* block = (Block*)allocHandle;
    OA_VMA_ASSERT(!block->IsFree() && "Trying to set user data for not allocated block!");
    block->UserData() = userData;
}

void OaVmaBlockMetadata_TLSF::DebugLogAllAllocations() const
{
    for (Block* block = m_NullBlock->prevPhysical; block != OA_VMA_NULL; block = block->prevPhysical)
        if (!block->IsFree())
            DebugLogAllocation(block->offset, block->size, block->UserData());
}

uint8_t OaVmaBlockMetadata_TLSF::SizeToMemoryClass(VkDeviceSize size)
{
    if (size > SMALL_BUFFER_SIZE)
        return uint8_t(OA_VMA_BITSCAN_MSB(size) - MEMORY_CLASS_SHIFT);
    return 0;
}

uint16_t OaVmaBlockMetadata_TLSF::SizeToSecondIndex(VkDeviceSize size, uint8_t memoryClass) const
{
    if (memoryClass == 0)
    {
        if (IsVirtual())
            return static_cast<uint16_t>((size - 1) / 8);
        return static_cast<uint16_t>((size - 1) / 64);
    }
    return static_cast<uint16_t>((size >> (memoryClass + MEMORY_CLASS_SHIFT - SECOND_LEVEL_INDEX)) ^ (1U << SECOND_LEVEL_INDEX));
}

uint32_t OaVmaBlockMetadata_TLSF::GetListIndex(uint8_t memoryClass, uint16_t secondIndex) const
{
    if (memoryClass == 0)
        return secondIndex;

    const uint32_t index = static_cast<uint32_t>(memoryClass - 1) * (1 << SECOND_LEVEL_INDEX) + secondIndex;
    if (IsVirtual())
        return index + (1 << SECOND_LEVEL_INDEX);
    return index + 4;
}

uint32_t OaVmaBlockMetadata_TLSF::GetListIndex(VkDeviceSize size) const
{
    uint8_t memoryClass = SizeToMemoryClass(size);
    return GetListIndex(memoryClass, SizeToSecondIndex(size, memoryClass));
}

void OaVmaBlockMetadata_TLSF::RemoveFreeBlock(Block* block)
{
    OA_VMA_ASSERT(block != m_NullBlock);
    OA_VMA_ASSERT(block->IsFree());

    if (block->NextFree() != OA_VMA_NULL)
        block->NextFree()->PrevFree() = block->PrevFree();
    if (block->PrevFree() != OA_VMA_NULL)
        block->PrevFree()->NextFree() = block->NextFree();
    else
    {
        uint8_t memClass = SizeToMemoryClass(block->size);
        uint16_t secondIndex = SizeToSecondIndex(block->size, memClass);
        uint32_t index = GetListIndex(memClass, secondIndex);
        OA_VMA_ASSERT(m_FreeList[index] == block);
        m_FreeList[index] = block->NextFree();
        if (block->NextFree() == OA_VMA_NULL)
        {
            m_InnerIsFreeBitmap[memClass] &= ~(1U << secondIndex);
            if (m_InnerIsFreeBitmap[memClass] == 0)
                m_IsFreeBitmap &= ~(1UL << memClass);
        }
    }
    block->MarkTaken();
    block->UserData() = OA_VMA_NULL;
    --m_BlocksFreeCount;
    m_BlocksFreeSize -= block->size;
}

void OaVmaBlockMetadata_TLSF::InsertFreeBlock(Block* block)
{
    OA_VMA_ASSERT(block != m_NullBlock);
    OA_VMA_ASSERT(!block->IsFree() && "Cannot insert block twice!");

    uint8_t memClass = SizeToMemoryClass(block->size);
    uint16_t secondIndex = SizeToSecondIndex(block->size, memClass);
    uint32_t index = GetListIndex(memClass, secondIndex);
    OA_VMA_ASSERT(index < m_ListsCount);
    block->PrevFree() = OA_VMA_NULL;
    block->NextFree() = m_FreeList[index];
    m_FreeList[index] = block;
    if (block->NextFree() != OA_VMA_NULL)
        block->NextFree()->PrevFree() = block;
    else
    {
        m_InnerIsFreeBitmap[memClass] |= 1U << secondIndex;
        m_IsFreeBitmap |= 1UL << memClass;
    }
    ++m_BlocksFreeCount;
    m_BlocksFreeSize += block->size;
}

void OaVmaBlockMetadata_TLSF::MergeBlock(Block* block, Block* prev)
{
    OA_VMA_ASSERT(block->prevPhysical == prev && "Cannot merge separate physical regions!");
    OA_VMA_ASSERT(!prev->IsFree() && "Cannot merge block that belongs to free list!");

    block->offset = prev->offset;
    block->size += prev->size;
    block->prevPhysical = prev->prevPhysical;
    if (block->prevPhysical)
        block->prevPhysical->nextPhysical = block;
    m_BlockAllocator.Free(prev);
}

OaVmaBlockMetadata_TLSF::Block* OaVmaBlockMetadata_TLSF::FindFreeBlock(VkDeviceSize size, uint32_t& listIndex) const
{
    uint8_t memoryClass = SizeToMemoryClass(size);
    uint32_t innerFreeMap = m_InnerIsFreeBitmap[memoryClass] & (~0U << SizeToSecondIndex(size, memoryClass));
    if (!innerFreeMap)
    {
        // Check higher levels for available blocks
        uint32_t freeMap = m_IsFreeBitmap & (~0UL << (memoryClass + 1));
        if (!freeMap)
            return OA_VMA_NULL; // No more memory available

        // Find lowest free region
        memoryClass = OA_VMA_BITSCAN_LSB(freeMap);
        innerFreeMap = m_InnerIsFreeBitmap[memoryClass];
        OA_VMA_ASSERT(innerFreeMap != 0);
    }
    // Find lowest free subregion
    listIndex = GetListIndex(memoryClass, OA_VMA_BITSCAN_LSB(innerFreeMap));
    OA_VMA_ASSERT(m_FreeList[listIndex]);
    return m_FreeList[listIndex];
}

bool OaVmaBlockMetadata_TLSF::CheckBlock(
    Block& block,
    uint32_t listIndex,
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    OaVmaSuballocationType allocType,
    OaVmaAllocationRequest* pAllocationRequest)
{
    OA_VMA_ASSERT(block.IsFree() && "Block is already taken!");

    VkDeviceSize alignedOffset = OaVmaAlignUp(block.offset, allocAlignment);
    if (block.size < allocSize + alignedOffset - block.offset)
        return false;

    // Check for granularity conflicts
    if (!IsVirtual() &&
        m_GranularityHandler.CheckConflictAndAlignUp(alignedOffset, allocSize, block.offset, block.size, allocType))
        return false;

    // Alloc successful
    pAllocationRequest->type = OaVmaAllocationRequestType::TLSF;
    pAllocationRequest->allocHandle = (OaVmaAllocHandle)&block;
    pAllocationRequest->size = allocSize - GetDebugMargin();
    pAllocationRequest->customData = (void*)allocType;
    pAllocationRequest->algorithmData = alignedOffset;

    // Place block at the start of list if it's normal block
    if (listIndex != m_ListsCount && block.PrevFree())
    {
        block.PrevFree()->NextFree() = block.NextFree();
        if (block.NextFree())
            block.NextFree()->PrevFree() = block.PrevFree();
        block.PrevFree() = OA_VMA_NULL;
        block.NextFree() = m_FreeList[listIndex];
        m_FreeList[listIndex] = &block;
        if (block.NextFree())
            block.NextFree()->PrevFree() = &block;
    }

    return true;
}
#endif // _OA_VMA_BLOCK_METADATA_TLSF_FUNCTIONS
#endif // _OA_VMA_BLOCK_METADATA_TLSF
