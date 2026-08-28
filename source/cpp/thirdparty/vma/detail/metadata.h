// SPDX: MIT. Derived from Vulkan Memory Allocator, (c) 2017-2026 Advanced Micro Devices, Inc.
// Full MIT copyright/permission notice: source/cpp/thirdparty/vma/vma.h. See NOTICE.md.
#ifndef _VMA_BLOCK_METADATA
/*
Data structure used for bookkeeping of allocations and unused ranges of memory
in a single VkDeviceMemory block.
*/
class VmaBlockMetadata
{
	VMA_CLASS_NO_COPY_NO_MOVE(VmaBlockMetadata)
public:
	// pAllocationCallbacks, if not null, must be owned externally - alive and unchanged for the whole lifetime of this object.
	VmaBlockMetadata(const VkAllocationCallbacks* pAllocationCallbacks,
		VkDeviceSize bufferImageGranularity, bool isVirtual);
	virtual ~VmaBlockMetadata() = default;

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
	virtual void GetAllocationInfo(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo) = 0;
	virtual VkDeviceSize GetAllocationOffset(VmaAllocHandle allocHandle) const = 0;
	virtual void* GetAllocationUserData(VmaAllocHandle allocHandle) const = 0;

	virtual VmaAllocHandle GetAllocationListBegin() const = 0;
	virtual VmaAllocHandle GetNextAllocation(VmaAllocHandle prevAlloc) const = 0;
	virtual VkDeviceSize GetNextFreeRegionSize(VmaAllocHandle alloc) const = 0;

	// Shouldn't modify blockCount.
	virtual void AddDetailedStatistics(VmaDetailedStatistics& inoutStats) const = 0;
	virtual void AddStatistics(VmaStatistics& inoutStats) const = 0;

#if VMA_STATS_STRING_ENABLED
	virtual void PrintDetailedMap(class VmaJsonWriter& json) const = 0;
#endif

	// Tries to find a place for suballocation with given parameters inside this block.
	// If succeeded, fills pAllocationRequest and returns true.
	// If failed, returns false.
	virtual bool CreateAllocationRequest(
		VkDeviceSize allocSize,
		VkDeviceSize allocAlignment,
		bool upperAddress,
		VmaSuballocationType allocType,
		// Always one of VMA_ALLOCATION_CREATE_STRATEGY_* or VMA_ALLOCATION_INTERNAL_STRATEGY_* flags.
		uint32_t strategy,
		VmaAllocationRequest* pAllocationRequest) = 0;

	virtual VkResult CheckCorruption(const void* pBlockData) = 0;

	// Makes actual allocation based on request. Request must already be checked and valid.
	virtual void Alloc(
		const VmaAllocationRequest& request,
		VmaSuballocationType type,
		void* userData) = 0;

	// Frees suballocation assigned to given memory region.
	virtual void Free(VmaAllocHandle allocHandle) = 0;

	// Frees all allocations.
	// Careful! Don't call it if there are VmaAllocation objects owned by userData of cleared allocations!
	virtual void Clear() = 0;

	virtual void SetAllocationUserData(VmaAllocHandle allocHandle, void* userData) = 0;
	virtual void DebugLogAllAllocations() const = 0;

protected:
	const VkAllocationCallbacks* GetAllocationCallbacks() const { return m_pAllocationCallbacks; }
	VkDeviceSize GetBufferImageGranularity() const { return m_BufferImageGranularity; }
	VkDeviceSize GetDebugMargin() const { return VkDeviceSize(IsVirtual() ? 0 : VMA_DEBUG_MARGIN); }

	void DebugLogAllocation(VkDeviceSize offset, VkDeviceSize size, void* userData) const;
#if VMA_STATS_STRING_ENABLED
	// mapRefCount == UINT32_MAX means unspecified.
	void PrintDetailedMap_Begin(class VmaJsonWriter& json,
		VkDeviceSize unusedBytes,
		size_t allocationCount,
		size_t unusedRangeCount) const;
	void PrintDetailedMap_Allocation(class VmaJsonWriter& json,
		VkDeviceSize offset, VkDeviceSize size, void* userData) const;
	static void PrintDetailedMap_UnusedRange(class VmaJsonWriter& json,
		VkDeviceSize offset,
		VkDeviceSize size);
	static void PrintDetailedMap_End(class VmaJsonWriter& json);
#endif

private:
	VkDeviceSize m_Size;
	const VkAllocationCallbacks* m_pAllocationCallbacks;
	const VkDeviceSize m_BufferImageGranularity;
	const bool m_IsVirtual;
};

#ifndef _VMA_BLOCK_METADATA_FUNCTIONS
VmaBlockMetadata::VmaBlockMetadata(const VkAllocationCallbacks* pAllocationCallbacks,
	VkDeviceSize bufferImageGranularity, bool isVirtual)
	: m_Size(0),
	m_pAllocationCallbacks(pAllocationCallbacks),
	m_BufferImageGranularity(bufferImageGranularity),
	m_IsVirtual(isVirtual) {}

void VmaBlockMetadata::DebugLogAllocation(VkDeviceSize offset, VkDeviceSize size, void* userData) const
{
	if (IsVirtual())
	{
		VMA_LEAK_LOG_FORMAT("UNFREED VIRTUAL ALLOCATION; Offset: %" PRIu64 "; Size: %" PRIu64 "; UserData: %p", offset, size, userData);
	}
	else
	{
		VMA_ASSERT(userData != VMA_NULL);
		VmaAllocation allocation = reinterpret_cast<VmaAllocation>(userData);

		userData = allocation->GetUserData();
		const char* name = allocation->GetName();

#if VMA_STATS_STRING_ENABLED
		VMA_LEAK_LOG_FORMAT("UNFREED ALLOCATION; Offset: %" PRIu64 "; Size: %" PRIu64 "; UserData: %p; Name: %s; Type: %s; Usage: %" PRIu64,
			offset, size, userData, name ? name : "Vma_empty",
			VMA_SUBALLOCATION_TYPE_NAMES[allocation->GetSuballocationType()],
			(uint64_t)allocation->GetBufferImageUsage().Value);
#else
		VMA_LEAK_LOG_FORMAT("UNFREED ALLOCATION; Offset: %" PRIu64 "; Size: %" PRIu64 "; UserData: %p; Name: %s; Type: %u",
			offset, size, userData, name ? name : "Vma_empty",
			(unsigned)allocation->GetSuballocationType());
#endif // VMA_STATS_STRING_ENABLED
	}

}

#if VMA_STATS_STRING_ENABLED
void VmaBlockMetadata::PrintDetailedMap_Begin(class VmaJsonWriter& json,
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

void VmaBlockMetadata::PrintDetailedMap_Allocation(class VmaJsonWriter& json,
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
		((VmaAllocation)userData)->PrintParameters(json);
	}

	json.EndObject();
}

void VmaBlockMetadata::PrintDetailedMap_UnusedRange(class VmaJsonWriter& json,
	VkDeviceSize offset, VkDeviceSize size)
{
	json.BeginObject(true);

	json.WriteString("Offset");
	json.WriteNumber(offset);

	json.WriteString("Type");
	json.WriteString(VMA_SUBALLOCATION_TYPE_NAMES[VMA_SUBALLOCATION_TYPE_FREE]);

	json.WriteString("Size");
	json.WriteNumber(size);

	json.EndObject();
}

void VmaBlockMetadata::PrintDetailedMap_End(class VmaJsonWriter& json)
{
	json.EndArray();
}
#endif // VMA_STATS_STRING_ENABLED
#endif // _VMA_BLOCK_METADATA_FUNCTIONS
#endif // _VMA_BLOCK_METADATA

#ifndef _VMA_BLOCK_BUFFER_IMAGE_GRANULARITY
// Before deleting object of this class remember to call 'Destroy()'
class VmaBlockBufferImageGranularity final
{
public:
	struct ValidationContext
	{
		const VkAllocationCallbacks* allocCallbacks;
		uint16_t* pageAllocs;
	};

	explicit VmaBlockBufferImageGranularity(VkDeviceSize bufferImageGranularity);
	~VmaBlockBufferImageGranularity();

	bool IsEnabled() const { return m_BufferImageGranularity > MAX_LOW_BUFFER_IMAGE_GRANULARITY; }

	void Init(const VkAllocationCallbacks* pAllocationCallbacks, VkDeviceSize size);
	// Before destroying object you must call free it's memory
	void Destroy(const VkAllocationCallbacks* pAllocationCallbacks);

	void RoundupAllocRequest(VmaSuballocationType allocType,
		VkDeviceSize& inOutAllocSize,
		VkDeviceSize& inOutAllocAlignment) const;

	bool CheckConflictAndAlignUp(VkDeviceSize& inOutAllocOffset,
		VkDeviceSize allocSize,
		VkDeviceSize blockOffset,
		VkDeviceSize blockSize,
		VmaSuballocationType allocType) const;

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

#ifndef _VMA_BLOCK_BUFFER_IMAGE_GRANULARITY_FUNCTIONS
VmaBlockBufferImageGranularity::VmaBlockBufferImageGranularity(VkDeviceSize bufferImageGranularity)
	: m_BufferImageGranularity(bufferImageGranularity),
	m_RegionCount(0),
	m_RegionInfo(VMA_NULL) {}

VmaBlockBufferImageGranularity::~VmaBlockBufferImageGranularity()
{
	VMA_ASSERT(m_RegionInfo == VMA_NULL && "Free not called before destroying object!");
}

void VmaBlockBufferImageGranularity::Init(const VkAllocationCallbacks* pAllocationCallbacks, VkDeviceSize size)
{
	if (IsEnabled())
	{
		m_RegionCount = static_cast<uint32_t>(VmaDivideRoundingUp(size, m_BufferImageGranularity));
		m_RegionInfo = Vma_new_array(pAllocationCallbacks, RegionInfo, m_RegionCount);
		oa::memzero(m_RegionInfo, m_RegionCount * sizeof(RegionInfo));
	}
}

void VmaBlockBufferImageGranularity::Destroy(const VkAllocationCallbacks* pAllocationCallbacks)
{
	if (m_RegionInfo)
	{
		Vma_delete_array(pAllocationCallbacks, m_RegionInfo, m_RegionCount);
		m_RegionInfo = VMA_NULL;
	}
}

void VmaBlockBufferImageGranularity::RoundupAllocRequest(VmaSuballocationType allocType,
	VkDeviceSize& inOutAllocSize,
	VkDeviceSize& inOutAllocAlignment) const
{
	if (m_BufferImageGranularity > 1 &&
		m_BufferImageGranularity <= MAX_LOW_BUFFER_IMAGE_GRANULARITY)
	{
		if (allocType == VMA_SUBALLOCATION_TYPE_UNKNOWN ||
			allocType == VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN ||
			allocType == VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL)
		{
			inOutAllocAlignment = VMA_MAX(inOutAllocAlignment, m_BufferImageGranularity);
			inOutAllocSize = VmaAlignUp(inOutAllocSize, m_BufferImageGranularity);
		}
	}
}

bool VmaBlockBufferImageGranularity::CheckConflictAndAlignUp(VkDeviceSize& inOutAllocOffset,
	VkDeviceSize allocSize,
	VkDeviceSize blockOffset,
	VkDeviceSize blockSize,
	VmaSuballocationType allocType) const
{
	if (IsEnabled())
	{
		uint32_t startPage = GetStartPage(inOutAllocOffset);
		if (m_RegionInfo[startPage].allocCount > 0 &&
			VmaIsBufferImageGranularityConflict(static_cast<VmaSuballocationType>(m_RegionInfo[startPage].allocType), allocType))
		{
			inOutAllocOffset = VmaAlignUp(inOutAllocOffset, m_BufferImageGranularity);
			if (blockSize < allocSize + inOutAllocOffset - blockOffset)
				return true;
			++startPage;
		}
		uint32_t endPage = GetEndPage(inOutAllocOffset, allocSize);
		if (endPage != startPage &&
			m_RegionInfo[endPage].allocCount > 0 &&
			VmaIsBufferImageGranularityConflict(static_cast<VmaSuballocationType>(m_RegionInfo[endPage].allocType), allocType))
		{
			return true;
		}
	}
	return false;
}

void VmaBlockBufferImageGranularity::AllocPages(uint8_t allocType, VkDeviceSize offset, VkDeviceSize size)
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

void VmaBlockBufferImageGranularity::FreePages(VkDeviceSize offset, VkDeviceSize size)
{
	if (IsEnabled())
	{
		uint32_t startPage = GetStartPage(offset);
		--m_RegionInfo[startPage].allocCount;
		if (m_RegionInfo[startPage].allocCount == 0)
			m_RegionInfo[startPage].allocType = VMA_SUBALLOCATION_TYPE_FREE;
		uint32_t endPage = GetEndPage(offset, size);
		if (startPage != endPage)
		{
			--m_RegionInfo[endPage].allocCount;
			if (m_RegionInfo[endPage].allocCount == 0)
				m_RegionInfo[endPage].allocType = VMA_SUBALLOCATION_TYPE_FREE;
		}
	}
}

void VmaBlockBufferImageGranularity::Clear()
{
	if (m_RegionInfo)
		oa::memzero(m_RegionInfo, m_RegionCount * sizeof(RegionInfo));
}

VmaBlockBufferImageGranularity::ValidationContext VmaBlockBufferImageGranularity::StartValidation(
	const VkAllocationCallbacks* pAllocationCallbacks, bool isVirutal) const
{
	ValidationContext ctx{ pAllocationCallbacks, VMA_NULL };
	if (!isVirutal && IsEnabled())
	{
		ctx.pageAllocs = Vma_new_array(pAllocationCallbacks, uint16_t, m_RegionCount);
		oa::memzero(ctx.pageAllocs, m_RegionCount * sizeof(uint16_t));
	}
	return ctx;
}

bool VmaBlockBufferImageGranularity::Validate(ValidationContext& ctx,
	VkDeviceSize offset, VkDeviceSize size) const
{
	if (IsEnabled())
	{
		uint32_t start = GetStartPage(offset);
		++ctx.pageAllocs[start];
		VMA_VALIDATE(m_RegionInfo[start].allocCount > 0);

		uint32_t end = GetEndPage(offset, size);
		if (start != end)
		{
			++ctx.pageAllocs[end];
			VMA_VALIDATE(m_RegionInfo[end].allocCount > 0);
		}
	}
	return true;
}

bool VmaBlockBufferImageGranularity::FinishValidation(ValidationContext& ctx) const
{
	// Check proper page structure
	if (IsEnabled())
	{
		VMA_ASSERT(ctx.pageAllocs != VMA_NULL && "Validation context not initialized!");

		for (uint32_t page = 0; page < m_RegionCount; ++page)
		{
			VMA_VALIDATE(ctx.pageAllocs[page] == m_RegionInfo[page].allocCount);
		}
		Vma_delete_array(ctx.allocCallbacks, ctx.pageAllocs, m_RegionCount);
		ctx.pageAllocs = VMA_NULL;
	}
	return true;
}

uint32_t VmaBlockBufferImageGranularity::OffsetToPageIndex(VkDeviceSize offset) const
{
	return static_cast<uint32_t>(offset >> VMA_BITSCAN_MSB(m_BufferImageGranularity));
}

void VmaBlockBufferImageGranularity::AllocPage(RegionInfo& page, uint8_t allocType)
{
	// When current alloc type is free then it can be overridden by new type
	if (page.allocCount == 0 || (page.allocCount > 0 && page.allocType == VMA_SUBALLOCATION_TYPE_FREE))
		page.allocType = allocType;

	++page.allocCount;
}
#endif // _VMA_BLOCK_BUFFER_IMAGE_GRANULARITY_FUNCTIONS
#endif // _VMA_BLOCK_BUFFER_IMAGE_GRANULARITY

#ifndef _VMA_BLOCK_METADATA_LINEAR
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
class VmaBlockMetadata_Linear : public VmaBlockMetadata
{
	VMA_CLASS_NO_COPY_NO_MOVE(VmaBlockMetadata_Linear)
public:
	VmaBlockMetadata_Linear(const VkAllocationCallbacks* pAllocationCallbacks,
		VkDeviceSize bufferImageGranularity, bool isVirtual);
	~VmaBlockMetadata_Linear() override = default;

	VkDeviceSize GetSumFreeSize() const override { return m_SumFreeSize; }
	bool IsEmpty() const override { return GetAllocationCount() == 0; }
	VkDeviceSize GetAllocationOffset(VmaAllocHandle allocHandle) const override { return (VkDeviceSize)allocHandle - 1; }

	void Init(VkDeviceSize size) override;
	bool Validate() const override;
	size_t GetAllocationCount() const override;
	size_t GetFreeRegionsCount() const override;

	void AddDetailedStatistics(VmaDetailedStatistics& inoutStats) const override;
	void AddStatistics(VmaStatistics& inoutStats) const override;

#if VMA_STATS_STRING_ENABLED
	void PrintDetailedMap(class VmaJsonWriter& json) const override;
#endif

	bool CreateAllocationRequest(
		VkDeviceSize allocSize,
		VkDeviceSize allocAlignment,
		bool upperAddress,
		VmaSuballocationType allocType,
		uint32_t strategy,
		VmaAllocationRequest* pAllocationRequest) override;

	VkResult CheckCorruption(const void* pBlockData) override;

	void Alloc(
		const VmaAllocationRequest& request,
		VmaSuballocationType type,
		void* userData) override;

	void Free(VmaAllocHandle allocHandle) override;
	void GetAllocationInfo(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo) override;
	void* GetAllocationUserData(VmaAllocHandle allocHandle) const override;
	VmaAllocHandle GetAllocationListBegin() const override;
	VmaAllocHandle GetNextAllocation(VmaAllocHandle prevAlloc) const override;
	VkDeviceSize GetNextFreeRegionSize(VmaAllocHandle alloc) const override;
	void Clear() override;
	void SetAllocationUserData(VmaAllocHandle allocHandle, void* userData) override;
	void DebugLogAllAllocations() const override;

private:
	/*
	There are two suballocation vectors, used in ping-pong way.
	The one with index m_1stVectorIndex is called 1st.
	The one with index (m_1stVectorIndex ^ 1) is called 2nd.
	2nd can be non-empty only when 1st is not empty.
	When 2nd is not empty, m_2ndVectorMode indicates its mode of operation.
	*/
	typedef VmaVector<VmaSuballocation, VmaStlAllocator<VmaSuballocation>> SuballocationVectorType;

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

	VmaSuballocation& FindSuballocation(VkDeviceSize offset) const;
	bool ShouldCompact1st() const;
	void CleanupAfterFree();

	bool CreateAllocationRequest_LowerAddress(
		VkDeviceSize allocSize,
		VkDeviceSize allocAlignment,
		VmaSuballocationType allocType,
		uint32_t strategy,
		VmaAllocationRequest* pAllocationRequest);
	bool CreateAllocationRequest_UpperAddress(
		VkDeviceSize allocSize,
		VkDeviceSize allocAlignment,
		VmaSuballocationType allocType,
		uint32_t strategy,
		VmaAllocationRequest* pAllocationRequest);
};

#ifndef _VMA_BLOCK_METADATA_LINEAR_FUNCTIONS
VmaBlockMetadata_Linear::VmaBlockMetadata_Linear(const VkAllocationCallbacks* pAllocationCallbacks,
	VkDeviceSize bufferImageGranularity, bool isVirtual)
	: VmaBlockMetadata(pAllocationCallbacks, bufferImageGranularity, isVirtual),
	m_SumFreeSize(0),
	m_Suballocations0(VmaStlAllocator<VmaSuballocation>(pAllocationCallbacks)),
	m_Suballocations1(VmaStlAllocator<VmaSuballocation>(pAllocationCallbacks)),
	m_1stVectorIndex(0),
	m_2ndVectorMode(SECOND_VECTOR_EMPTY),
	m_1stNullItemsBeginCount(0),
	m_1stNullItemsMiddleCount(0),
	m_2ndNullItemsCount(0) {}

void VmaBlockMetadata_Linear::Init(VkDeviceSize size)
{
	VmaBlockMetadata::Init(size);
	m_SumFreeSize = size;
}

bool VmaBlockMetadata_Linear::Validate() const
{
	const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
	const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

	VMA_VALIDATE(suballocations2nd.empty() == (m_2ndVectorMode == SECOND_VECTOR_EMPTY));
	VMA_VALIDATE(!suballocations1st.empty() ||
		suballocations2nd.empty() ||
		m_2ndVectorMode != SECOND_VECTOR_RING_BUFFER);

	if (!suballocations1st.empty())
	{
		// Null item at the beginning should be accounted into m_1stNullItemsBeginCount.
		VMA_VALIDATE(suballocations1st[m_1stNullItemsBeginCount].type != VMA_SUBALLOCATION_TYPE_FREE);
		// Null item at the end should be just pop_back().
		VMA_VALIDATE(suballocations1st.back().type != VMA_SUBALLOCATION_TYPE_FREE);
	}
	if (!suballocations2nd.empty())
	{
		// Null item at the end should be just pop_back().
		VMA_VALIDATE(suballocations2nd.back().type != VMA_SUBALLOCATION_TYPE_FREE);
	}

	VMA_VALIDATE(m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount <= suballocations1st.size());
	VMA_VALIDATE(m_2ndNullItemsCount <= suballocations2nd.size());

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
			const VmaSuballocation& suballoc = suballocations2nd[i];
			const bool currFree = (suballoc.type == VMA_SUBALLOCATION_TYPE_FREE);

			VmaAllocation const alloc = (VmaAllocation)suballoc.userData;
			if (!IsVirtual())
			{
				VMA_VALIDATE(currFree == (alloc == VK_NULL_HANDLE));
			}
			VMA_VALIDATE(suballoc.offset >= offset);

			if (!currFree)
			{
				if (!IsVirtual())
				{
					VMA_VALIDATE((VkDeviceSize)alloc->GetAllocHandle() == suballoc.offset + 1);
					VMA_VALIDATE(alloc->GetSize() == suballoc.size);
				}
				sumUsedSize += suballoc.size;
			}
			else
			{
				++nullItem2ndCount;
			}

			offset = suballoc.offset + suballoc.size + debugMargin;
		}

		VMA_VALIDATE(nullItem2ndCount == m_2ndNullItemsCount);
	}

	for (size_t i = 0; i < m_1stNullItemsBeginCount; ++i)
	{
		const VmaSuballocation& suballoc = suballocations1st[i];
		VMA_VALIDATE(suballoc.type == VMA_SUBALLOCATION_TYPE_FREE &&
			suballoc.userData == VMA_NULL);
	}

	size_t nullItem1stCount = m_1stNullItemsBeginCount;

	for (size_t i = m_1stNullItemsBeginCount; i < suballoc1stCount; ++i)
	{
		const VmaSuballocation& suballoc = suballocations1st[i];
		const bool currFree = (suballoc.type == VMA_SUBALLOCATION_TYPE_FREE);

		VmaAllocation const alloc = (VmaAllocation)suballoc.userData;
		if (!IsVirtual())
		{
			VMA_VALIDATE(currFree == (alloc == VK_NULL_HANDLE));
		}
		VMA_VALIDATE(suballoc.offset >= offset);
		VMA_VALIDATE(i >= m_1stNullItemsBeginCount || currFree);

		if (!currFree)
		{
			if (!IsVirtual())
			{
				VMA_VALIDATE((VkDeviceSize)alloc->GetAllocHandle() == suballoc.offset + 1);
				VMA_VALIDATE(alloc->GetSize() == suballoc.size);
			}
			sumUsedSize += suballoc.size;
		}
		else
		{
			++nullItem1stCount;
		}

		offset = suballoc.offset + suballoc.size + debugMargin;
	}
	VMA_VALIDATE(nullItem1stCount == m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount);

	if (m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
	{
		const size_t suballoc2ndCount = suballocations2nd.size();
		size_t nullItem2ndCount = 0;
		for (size_t i = suballoc2ndCount; i--; )
		{
			const VmaSuballocation& suballoc = suballocations2nd[i];
			const bool currFree = (suballoc.type == VMA_SUBALLOCATION_TYPE_FREE);

			VmaAllocation const alloc = (VmaAllocation)suballoc.userData;
			if (!IsVirtual())
			{
				VMA_VALIDATE(currFree == (alloc == VK_NULL_HANDLE));
			}
			VMA_VALIDATE(suballoc.offset >= offset);

			if (!currFree)
			{
				if (!IsVirtual())
				{
					VMA_VALIDATE((VkDeviceSize)alloc->GetAllocHandle() == suballoc.offset + 1);
					VMA_VALIDATE(alloc->GetSize() == suballoc.size);
				}
				sumUsedSize += suballoc.size;
			}
			else
			{
				++nullItem2ndCount;
			}

			offset = suballoc.offset + suballoc.size + debugMargin;
		}

		VMA_VALIDATE(nullItem2ndCount == m_2ndNullItemsCount);
	}

	VMA_VALIDATE(offset <= GetSize());
	VMA_VALIDATE(m_SumFreeSize == GetSize() - sumUsedSize);

	return true;
}

size_t VmaBlockMetadata_Linear::GetAllocationCount() const
{
	return AccessSuballocations1st().size() - m_1stNullItemsBeginCount - m_1stNullItemsMiddleCount +
		AccessSuballocations2nd().size() - m_2ndNullItemsCount;
}

size_t VmaBlockMetadata_Linear::GetFreeRegionsCount() const
{
	// Function only used for defragmentation, which is disabled for this algorithm
	VMA_ASSERT(0);
	return SIZE_MAX;
}

void VmaBlockMetadata_Linear::AddDetailedStatistics(VmaDetailedStatistics& inoutStats) const
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
				suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
			{
				++nextAlloc2ndIndex;
			}

			// Found non-null allocation.
			if (nextAlloc2ndIndex < suballoc2ndCount)
			{
				const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

				// 1. Process free space before this allocation.
				if (lastOffset < suballoc.offset)
				{
					// There is free space from lastOffset to suballoc.offset.
					const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
					VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
				}

				// 2. Process this allocation.
				// There is allocation with suballoc.offset, suballoc.size.
				VmaAddDetailedStatisticsAllocation(inoutStats, suballoc.size);

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
					VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
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
			suballocations1st[nextAlloc1stIndex].userData == VMA_NULL)
		{
			++nextAlloc1stIndex;
		}

		// Found non-null allocation.
		if (nextAlloc1stIndex < suballoc1stCount)
		{
			const VmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];

			// 1. Process free space before this allocation.
			if (lastOffset < suballoc.offset)
			{
				// There is free space from lastOffset to suballoc.offset.
				const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
				VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
			}

			// 2. Process this allocation.
			// There is allocation with suballoc.offset, suballoc.size.
			VmaAddDetailedStatisticsAllocation(inoutStats, suballoc.size);

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
				VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
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
				suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
			{
				--nextAlloc2ndIndex;
			}

			// Found non-null allocation.
			if (nextAlloc2ndIndex != SIZE_MAX)
			{
				const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

				// 1. Process free space before this allocation.
				if (lastOffset < suballoc.offset)
				{
					// There is free space from lastOffset to suballoc.offset.
					const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
					VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
				}

				// 2. Process this allocation.
				// There is allocation with suballoc.offset, suballoc.size.
				VmaAddDetailedStatisticsAllocation(inoutStats, suballoc.size);

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
					VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
				}

				// End of loop.
				lastOffset = size;
			}
		}
	}
}

void VmaBlockMetadata_Linear::AddStatistics(VmaStatistics& inoutStats) const
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
				suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
			{
				++nextAlloc2ndIndex;
			}

			// Found non-null allocation.
			if (nextAlloc2ndIndex < suballoc2ndCount)
			{
				const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

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
			suballocations1st[nextAlloc1stIndex].userData == VMA_NULL)
		{
			++nextAlloc1stIndex;
		}

		// Found non-null allocation.
		if (nextAlloc1stIndex < suballoc1stCount)
		{
			const VmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];

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
				suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
			{
				--nextAlloc2ndIndex;
			}

			// Found non-null allocation.
			if (nextAlloc2ndIndex != SIZE_MAX)
			{
				const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

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

#if VMA_STATS_STRING_ENABLED
void VmaBlockMetadata_Linear::PrintDetailedMap(class VmaJsonWriter& json) const
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
				suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
			{
				++nextAlloc2ndIndex;
			}

			// Found non-null allocation.
			if (nextAlloc2ndIndex < suballoc2ndCount)
			{
				const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

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
			suballocations1st[nextAlloc1stIndex].userData == VMA_NULL)
		{
			++nextAlloc1stIndex;
		}

		// Found non-null allocation.
		if (nextAlloc1stIndex < suballoc1stCount)
		{
			const VmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];

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
				suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
			{
				--nextAlloc2ndIndex;
			}

			// Found non-null allocation.
			if (nextAlloc2ndIndex != SIZE_MAX)
			{
				const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

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
				suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
			{
				++nextAlloc2ndIndex;
			}

			// Found non-null allocation.
			if (nextAlloc2ndIndex < suballoc2ndCount)
			{
				const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

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
			suballocations1st[nextAlloc1stIndex].userData == VMA_NULL)
		{
			++nextAlloc1stIndex;
		}

		// Found non-null allocation.
		if (nextAlloc1stIndex < suballoc1stCount)
		{
			const VmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];

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
				suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
			{
				--nextAlloc2ndIndex;
			}

			// Found non-null allocation.
			if (nextAlloc2ndIndex != SIZE_MAX)
			{
				const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

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
#endif // VMA_STATS_STRING_ENABLED

bool VmaBlockMetadata_Linear::CreateAllocationRequest(
	VkDeviceSize allocSize,
	VkDeviceSize allocAlignment,
	bool upperAddress,
	VmaSuballocationType allocType,
	uint32_t strategy,
	VmaAllocationRequest* pAllocationRequest)
{
	VMA_ASSERT(allocSize > 0);
	VMA_ASSERT(allocType != VMA_SUBALLOCATION_TYPE_FREE);
	VMA_ASSERT(pAllocationRequest != VMA_NULL);
	VMA_HEAVY_ASSERT(Validate());

	if(allocSize > GetSize())
		return false;

	pAllocationRequest->size = allocSize;
	return upperAddress ?
		CreateAllocationRequest_UpperAddress(
			allocSize, allocAlignment, allocType, strategy, pAllocationRequest) :
		CreateAllocationRequest_LowerAddress(
			allocSize, allocAlignment, allocType, strategy, pAllocationRequest);
}

VkResult VmaBlockMetadata_Linear::CheckCorruption(const void* pBlockData)
{
	VMA_ASSERT(!IsVirtual());
	SuballocationVectorType& suballocations1st = AccessSuballocations1st();
	for (size_t i = m_1stNullItemsBeginCount, count = suballocations1st.size(); i < count; ++i)
	{
		const VmaSuballocation& suballoc = suballocations1st[i];
		if (suballoc.type != VMA_SUBALLOCATION_TYPE_FREE)
		{
			if (!VmaValidateMagicValue(pBlockData, suballoc.offset + suballoc.size))
			{
				VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER VALIDATED ALLOCATION!");
				return VK_ERROR_UNKNOWN_COPY;
			}
		}
	}

	SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
	for (size_t i = 0, count = suballocations2nd.size(); i < count; ++i)
	{
		const VmaSuballocation& suballoc = suballocations2nd[i];
		if (suballoc.type != VMA_SUBALLOCATION_TYPE_FREE)
		{
			if (!VmaValidateMagicValue(pBlockData, suballoc.offset + suballoc.size))
			{
				VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER VALIDATED ALLOCATION!");
				return VK_ERROR_UNKNOWN_COPY;
			}
		}
	}

	return VK_SUCCESS;
}

void VmaBlockMetadata_Linear::Alloc(
	const VmaAllocationRequest& request,
	VmaSuballocationType type,
	void* userData)
{
	const VkDeviceSize offset = (VkDeviceSize)request.allocHandle - 1;
	const VmaSuballocation newSuballoc = { offset, request.size, userData, type };

	switch (request.type)
	{
	case VmaAllocationRequestType::UpperAddress:
	{
		VMA_ASSERT(m_2ndVectorMode != SECOND_VECTOR_RING_BUFFER &&
			"CRITICAL ERROR: Trying to use linear allocator as double stack while it was already used as ring buffer.");
		SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
		suballocations2nd.push_back(newSuballoc);
		m_2ndVectorMode = SECOND_VECTOR_DOUBLE_STACK;
	}
	break;
	case VmaAllocationRequestType::EndOf1st:
	{
		SuballocationVectorType& suballocations1st = AccessSuballocations1st();

		VMA_ASSERT(suballocations1st.empty() ||
			offset >= suballocations1st.back().offset + suballocations1st.back().size);
		// Check if it fits before the end of the block.
		VMA_ASSERT(offset + request.size <= GetSize());

		suballocations1st.push_back(newSuballoc);
	}
	break;
	case VmaAllocationRequestType::EndOf2nd:
	{
		SuballocationVectorType& suballocations1st = AccessSuballocations1st();
		// New allocation at the end of 2-part ring buffer, so before first allocation from 1st vector.
		VMA_ASSERT(!suballocations1st.empty() &&
			offset + request.size <= suballocations1st[m_1stNullItemsBeginCount].offset);
		SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

		switch (m_2ndVectorMode)
		{
		case SECOND_VECTOR_EMPTY:
			// First allocation from second part ring buffer.
			VMA_ASSERT(suballocations2nd.empty());
			m_2ndVectorMode = SECOND_VECTOR_RING_BUFFER;
			break;
		case SECOND_VECTOR_RING_BUFFER:
			// 2-part ring buffer is already started.
			VMA_ASSERT(!suballocations2nd.empty());
			break;
		case SECOND_VECTOR_DOUBLE_STACK:
			VMA_ASSERT(0 && "CRITICAL ERROR: Trying to use linear allocator as ring buffer while it was already used as double stack.");
			break;
		default:
			VMA_ASSERT(0);
		}

		suballocations2nd.push_back(newSuballoc);
	}
	break;
	default:
		VMA_ASSERT(0 && "CRITICAL INTERNAL ERROR.");
	}

	m_SumFreeSize -= newSuballoc.size;
}

void VmaBlockMetadata_Linear::Free(VmaAllocHandle allocHandle)
{
	SuballocationVectorType& suballocations1st = AccessSuballocations1st();
	SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
	VkDeviceSize offset = (VkDeviceSize)allocHandle - 1;

	if (!suballocations1st.empty())
	{
		// First allocation: Mark it as next empty at the beginning.
		VmaSuballocation& firstSuballoc = suballocations1st[m_1stNullItemsBeginCount];
		if (firstSuballoc.offset == offset)
		{
			firstSuballoc.type = VMA_SUBALLOCATION_TYPE_FREE;
			firstSuballoc.userData = VMA_NULL;
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
		VmaSuballocation& lastSuballoc = suballocations2nd.back();
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
		VmaSuballocation& lastSuballoc = suballocations1st.back();
		if (lastSuballoc.offset == offset)
		{
			m_SumFreeSize += lastSuballoc.size;
			suballocations1st.pop_back();
			CleanupAfterFree();
			return;
		}
	}

	VmaSuballocation refSuballoc;
	refSuballoc.offset = offset;
	// Rest of members stays uninitialized intentionally for better performance.

	// Item from the middle of 1st vector.
	{
		const SuballocationVectorType::iterator it = VmaBinaryFindSorted(
			suballocations1st.begin() + m_1stNullItemsBeginCount,
			suballocations1st.end(),
			refSuballoc,
			VmaSuballocationOffsetLess());
		if (it != suballocations1st.end())
		{
			it->type = VMA_SUBALLOCATION_TYPE_FREE;
			it->userData = VMA_NULL;
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
			VmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, VmaSuballocationOffsetLess()) :
			VmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, VmaSuballocationOffsetGreater());
		if (it != suballocations2nd.end())
		{
			it->type = VMA_SUBALLOCATION_TYPE_FREE;
			it->userData = VMA_NULL;
			++m_2ndNullItemsCount;
			m_SumFreeSize += it->size;
			CleanupAfterFree();
			return;
		}
	}

	VMA_ASSERT(0 && "Allocation to free not found in linear allocator!");
}

void VmaBlockMetadata_Linear::GetAllocationInfo(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo)
{
	outInfo.offset = (VkDeviceSize)allocHandle - 1;
	VmaSuballocation& suballoc = FindSuballocation(outInfo.offset);
	outInfo.size = suballoc.size;
	outInfo.pUserData = suballoc.userData;
}

void* VmaBlockMetadata_Linear::GetAllocationUserData(VmaAllocHandle allocHandle) const
{
	return FindSuballocation((VkDeviceSize)allocHandle - 1).userData;
}

VmaAllocHandle VmaBlockMetadata_Linear::GetAllocationListBegin() const
{
	// Function only used for defragmentation, which is disabled for this algorithm
	VMA_ASSERT(0);
	return VK_NULL_HANDLE;
}

VmaAllocHandle VmaBlockMetadata_Linear::GetNextAllocation(VmaAllocHandle prevAlloc) const
{
	// Function only used for defragmentation, which is disabled for this algorithm
	VMA_ASSERT(0);
	return VK_NULL_HANDLE;
}

VkDeviceSize VmaBlockMetadata_Linear::GetNextFreeRegionSize(VmaAllocHandle alloc) const
{
	// Function only used for defragmentation, which is disabled for this algorithm
	VMA_ASSERT(0);
	return 0;
}

void VmaBlockMetadata_Linear::Clear()
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

void VmaBlockMetadata_Linear::SetAllocationUserData(VmaAllocHandle allocHandle, void* userData)
{
	VmaSuballocation& suballoc = FindSuballocation((VkDeviceSize)allocHandle - 1);
	suballoc.userData = userData;
}

void VmaBlockMetadata_Linear::DebugLogAllAllocations() const
{
	const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
	for (auto it = suballocations1st.begin() + m_1stNullItemsBeginCount; it != suballocations1st.end(); ++it)
		if (it->type != VMA_SUBALLOCATION_TYPE_FREE)
			DebugLogAllocation(it->offset, it->size, it->userData);

	const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
	for (auto it = suballocations2nd.begin(); it != suballocations2nd.end(); ++it)
		if (it->type != VMA_SUBALLOCATION_TYPE_FREE)
			DebugLogAllocation(it->offset, it->size, it->userData);
}

VmaSuballocation& VmaBlockMetadata_Linear::FindSuballocation(VkDeviceSize offset) const
{
	const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
	const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

	VmaSuballocation refSuballoc;
	refSuballoc.offset = offset;
	// Rest of members stays uninitialized intentionally for better performance.

	// Item from the 1st vector.
	{
		SuballocationVectorType::const_iterator it = VmaBinaryFindSorted(
			suballocations1st.begin() + m_1stNullItemsBeginCount,
			suballocations1st.end(),
			refSuballoc,
			VmaSuballocationOffsetLess());
		if (it != suballocations1st.end())
		{
			return const_cast<VmaSuballocation&>(*it);
		}
	}

	if (m_2ndVectorMode != SECOND_VECTOR_EMPTY)
	{
		// Rest of members stays uninitialized intentionally for better performance.
		SuballocationVectorType::const_iterator it = m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER ?
			VmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, VmaSuballocationOffsetLess()) :
			VmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, VmaSuballocationOffsetGreater());
		if (it != suballocations2nd.end())
		{
			return const_cast<VmaSuballocation&>(*it);
		}
	}

	VMA_ASSERT(0 && "Allocation not found in linear allocator!");
	return const_cast<VmaSuballocation&>(suballocations1st.back()); // Should never occur.
}

bool VmaBlockMetadata_Linear::ShouldCompact1st() const
{
	const size_t nullItemCount = m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount;
	const size_t suballocCount = AccessSuballocations1st().size();
	return suballocCount > 32 && nullItemCount * 2 >= (suballocCount - nullItemCount) * 3;
}

void VmaBlockMetadata_Linear::CleanupAfterFree()
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
		VMA_ASSERT(nullItem1stCount <= suballoc1stCount);

		// Find more null items at the beginning of 1st vector.
		while (m_1stNullItemsBeginCount < suballoc1stCount &&
			suballocations1st[m_1stNullItemsBeginCount].type == VMA_SUBALLOCATION_TYPE_FREE)
		{
			++m_1stNullItemsBeginCount;
			--m_1stNullItemsMiddleCount;
		}

		// Find more null items at the end of 1st vector.
		while (m_1stNullItemsMiddleCount > 0 &&
			suballocations1st.back().type == VMA_SUBALLOCATION_TYPE_FREE)
		{
			--m_1stNullItemsMiddleCount;
			suballocations1st.pop_back();
		}

		// Find more null items at the end of 2nd vector.
		while (m_2ndNullItemsCount > 0 &&
			suballocations2nd.back().type == VMA_SUBALLOCATION_TYPE_FREE)
		{
			--m_2ndNullItemsCount;
			suballocations2nd.pop_back();
		}

		// Find more null items at the beginning of 2nd vector.
		while (m_2ndNullItemsCount > 0 &&
			suballocations2nd[0].type == VMA_SUBALLOCATION_TYPE_FREE)
		{
			--m_2ndNullItemsCount;
			VmaVectorRemove(suballocations2nd, 0);
		}

		if (ShouldCompact1st())
		{
			const size_t nonNullItemCount = suballoc1stCount - nullItem1stCount;
			size_t srcIndex = m_1stNullItemsBeginCount;
			for (size_t dstIndex = 0; dstIndex < nonNullItemCount; ++dstIndex)
			{
				while (suballocations1st[srcIndex].type == VMA_SUBALLOCATION_TYPE_FREE)
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
					suballocations2nd[m_1stNullItemsBeginCount].type == VMA_SUBALLOCATION_TYPE_FREE)
				{
					++m_1stNullItemsBeginCount;
					--m_1stNullItemsMiddleCount;
				}
				m_2ndNullItemsCount = 0;
				m_1stVectorIndex ^= 1;
			}
		}
	}

	VMA_HEAVY_ASSERT(Validate());
}

bool VmaBlockMetadata_Linear::CreateAllocationRequest_LowerAddress(
	VkDeviceSize allocSize,
	VkDeviceSize allocAlignment,
	VmaSuballocationType allocType,
	uint32_t strategy,
	VmaAllocationRequest* pAllocationRequest)
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
			const VmaSuballocation& lastSuballoc = suballocations1st.back();
			resultBaseOffset = lastSuballoc.offset + lastSuballoc.size + debugMargin;
		}

		// Start from offset equal to beginning of free space.
		VkDeviceSize resultOffset = resultBaseOffset;

		// Apply alignment.
		resultOffset = VmaAlignUp(resultOffset, allocAlignment);

		// Check previous suballocations for BufferImageGranularity conflicts.
		// Make bigger alignment if necessary.
		if (bufferImageGranularity > 1 && bufferImageGranularity != allocAlignment && !suballocations1st.empty())
		{
			bool bufferImageGranularityConflict = false;
			for (size_t prevSuballocIndex = suballocations1st.size(); prevSuballocIndex--; )
			{
				const VmaSuballocation& prevSuballoc = suballocations1st[prevSuballocIndex];
				if (VmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, resultOffset, bufferImageGranularity))
				{
					if (VmaIsBufferImageGranularityConflict(prevSuballoc.type, allocType))
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
				resultOffset = VmaAlignUp(resultOffset, bufferImageGranularity);
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
					const VmaSuballocation& nextSuballoc = suballocations2nd[nextSuballocIndex];
					if (VmaBlocksOnSamePage(resultOffset, allocSize, nextSuballoc.offset, bufferImageGranularity))
					{
						if (VmaIsBufferImageGranularityConflict(allocType, nextSuballoc.type))
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
			pAllocationRequest->allocHandle = (VmaAllocHandle)(resultOffset + 1);
			// pAllocationRequest->item, customData unused.
			pAllocationRequest->type = VmaAllocationRequestType::EndOf1st;
			return true;
		}
	}

	// Wrap-around to end of 2nd vector. Try to allocate there, watching for the
	// beginning of 1st vector as the end of free space.
	if (m_2ndVectorMode == SECOND_VECTOR_EMPTY || m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
	{
		VMA_ASSERT(!suballocations1st.empty());

		VkDeviceSize resultBaseOffset = 0;
		if (!suballocations2nd.empty())
		{
			const VmaSuballocation& lastSuballoc = suballocations2nd.back();
			resultBaseOffset = lastSuballoc.offset + lastSuballoc.size + debugMargin;
		}

		// Start from offset equal to beginning of free space.
		VkDeviceSize resultOffset = resultBaseOffset;

		// Apply alignment.
		resultOffset = VmaAlignUp(resultOffset, allocAlignment);

		// Check previous suballocations for BufferImageGranularity conflicts.
		// Make bigger alignment if necessary.
		if (bufferImageGranularity > 1 && bufferImageGranularity != allocAlignment && !suballocations2nd.empty())
		{
			bool bufferImageGranularityConflict = false;
			for (size_t prevSuballocIndex = suballocations2nd.size(); prevSuballocIndex--; )
			{
				const VmaSuballocation& prevSuballoc = suballocations2nd[prevSuballocIndex];
				if (VmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, resultOffset, bufferImageGranularity))
				{
					if (VmaIsBufferImageGranularityConflict(prevSuballoc.type, allocType))
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
				resultOffset = VmaAlignUp(resultOffset, bufferImageGranularity);
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
					const VmaSuballocation& nextSuballoc = suballocations1st[nextSuballocIndex];
					if (VmaBlocksOnSamePage(resultOffset, allocSize, nextSuballoc.offset, bufferImageGranularity))
					{
						if (VmaIsBufferImageGranularityConflict(allocType, nextSuballoc.type))
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
			pAllocationRequest->allocHandle = (VmaAllocHandle)(resultOffset + 1);
			pAllocationRequest->type = VmaAllocationRequestType::EndOf2nd;
			// pAllocationRequest->item, customData unused.
			return true;
		}
	}

	return false;
}

bool VmaBlockMetadata_Linear::CreateAllocationRequest_UpperAddress(
	VkDeviceSize allocSize,
	VkDeviceSize allocAlignment,
	VmaSuballocationType allocType,
	uint32_t strategy,
	VmaAllocationRequest* pAllocationRequest)
{
	const VkDeviceSize blockSize = GetSize();
	const VkDeviceSize bufferImageGranularity = GetBufferImageGranularity();
	SuballocationVectorType& suballocations1st = AccessSuballocations1st();
	SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

	if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
	{
		VMA_ASSERT(0 && "Trying to use pool with linear algorithm as double stack, while it is already being used as ring buffer.");
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
		const VmaSuballocation& lastSuballoc = suballocations2nd.back();
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
	resultOffset = VmaAlignDown(resultOffset, allocAlignment);

	// Check next suballocations from 2nd for BufferImageGranularity conflicts.
	// Make bigger alignment if necessary.
	if (bufferImageGranularity > 1 && bufferImageGranularity != allocAlignment && !suballocations2nd.empty())
	{
		bool bufferImageGranularityConflict = false;
		for (size_t nextSuballocIndex = suballocations2nd.size(); nextSuballocIndex--; )
		{
			const VmaSuballocation& nextSuballoc = suballocations2nd[nextSuballocIndex];
			if (VmaBlocksOnSamePage(resultOffset, allocSize, nextSuballoc.offset, bufferImageGranularity))
			{
				if (VmaIsBufferImageGranularityConflict(nextSuballoc.type, allocType))
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
			resultOffset = VmaAlignDown(resultOffset, bufferImageGranularity);
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
				const VmaSuballocation& prevSuballoc = suballocations1st[prevSuballocIndex];
				if (VmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, resultOffset, bufferImageGranularity))
				{
					if (VmaIsBufferImageGranularityConflict(allocType, prevSuballoc.type))
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
		pAllocationRequest->allocHandle = (VmaAllocHandle)(resultOffset + 1);
		// pAllocationRequest->item unused.
		pAllocationRequest->type = VmaAllocationRequestType::UpperAddress;
		return true;
	}

	return false;
}
#endif // _VMA_BLOCK_METADATA_LINEAR_FUNCTIONS
#endif // _VMA_BLOCK_METADATA_LINEAR

#ifndef _VMA_BLOCK_METADATA_TLSF
// To not search current larger region if first allocation won't succeed and skip to smaller range
// use with VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT as strategy in CreateAllocationRequest().
// When fragmentation and reusal of previous blocks doesn't matter then use with
// VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT for fastest alloc time possible.
class VmaBlockMetadata_TLSF : public VmaBlockMetadata
{
	VMA_CLASS_NO_COPY_NO_MOVE(VmaBlockMetadata_TLSF)
public:
	VmaBlockMetadata_TLSF(const VkAllocationCallbacks* pAllocationCallbacks,
		VkDeviceSize bufferImageGranularity, bool isVirtual);
	~VmaBlockMetadata_TLSF() override;

	size_t GetAllocationCount() const override { return m_AllocCount; }
	size_t GetFreeRegionsCount() const override { return m_BlocksFreeCount + 1; }
	VkDeviceSize GetSumFreeSize() const override { return m_BlocksFreeSize + m_NullBlock->size; }
	bool IsEmpty() const override { return m_NullBlock->offset == 0; }
	VkDeviceSize GetAllocationOffset(VmaAllocHandle allocHandle) const override { return ((Block*)allocHandle)->offset; }

	void Init(VkDeviceSize size) override;
	bool Validate() const override;

	void AddDetailedStatistics(VmaDetailedStatistics& inoutStats) const override;
	void AddStatistics(VmaStatistics& inoutStats) const override;

#if VMA_STATS_STRING_ENABLED
	void PrintDetailedMap(class VmaJsonWriter& json) const override;
#endif

	bool CreateAllocationRequest(
		VkDeviceSize allocSize,
		VkDeviceSize allocAlignment,
		bool upperAddress,
		VmaSuballocationType allocType,
		uint32_t strategy,
		VmaAllocationRequest* pAllocationRequest) override;

	VkResult CheckCorruption(const void* pBlockData) override;
	void Alloc(
		const VmaAllocationRequest& request,
		VmaSuballocationType type,
		void* userData) override;

	void Free(VmaAllocHandle allocHandle) override;
	void GetAllocationInfo(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo) override;
	void* GetAllocationUserData(VmaAllocHandle allocHandle) const override;
	VmaAllocHandle GetAllocationListBegin() const override;
	VmaAllocHandle GetNextAllocation(VmaAllocHandle prevAlloc) const override;
	VkDeviceSize GetNextFreeRegionSize(VmaAllocHandle alloc) const override;
	void Clear() override;
	void SetAllocationUserData(VmaAllocHandle allocHandle, void* userData) override;
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

		void MarkFree() { prevFree = VMA_NULL; }
		void MarkTaken() { prevFree = this; }
		bool IsFree() const { return prevFree != this; }
		void*& UserData() { VMA_HEAVY_ASSERT(!IsFree()); return userData; }
		Block*& PrevFree() { return prevFree; }
		Block*& NextFree() { VMA_HEAVY_ASSERT(IsFree()); return nextFree; }

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
	VmaPoolAllocator<Block> m_BlockAllocator;
	Block* m_NullBlock;
	VmaBlockBufferImageGranularity m_GranularityHandler;

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
		VmaSuballocationType allocType,
		VmaAllocationRequest* pAllocationRequest);
};

#ifndef _VMA_BLOCK_METADATA_TLSF_FUNCTIONS
VmaBlockMetadata_TLSF::VmaBlockMetadata_TLSF(const VkAllocationCallbacks* pAllocationCallbacks,
	VkDeviceSize bufferImageGranularity, bool isVirtual)
	: VmaBlockMetadata(pAllocationCallbacks, bufferImageGranularity, isVirtual),
	m_AllocCount(0),
	m_BlocksFreeCount(0),
	m_BlocksFreeSize(0),
	m_IsFreeBitmap(0),
	m_MemoryClasses(0),
	m_ListsCount(0),
	m_FreeList(VMA_NULL),
	m_BlockAllocator(pAllocationCallbacks, INITIAL_BLOCK_ALLOC_COUNT),
	m_NullBlock(VMA_NULL),
	m_GranularityHandler(bufferImageGranularity) {}

VmaBlockMetadata_TLSF::~VmaBlockMetadata_TLSF()
{
	if (m_FreeList)
		Vma_delete_array(GetAllocationCallbacks(), m_FreeList, m_ListsCount);
	m_GranularityHandler.Destroy(GetAllocationCallbacks());
}

void VmaBlockMetadata_TLSF::Init(VkDeviceSize size)
{
	VmaBlockMetadata::Init(size);

	if (!IsVirtual())
		m_GranularityHandler.Init(GetAllocationCallbacks(), size);

	m_NullBlock = m_BlockAllocator.Alloc();
	m_NullBlock->size = size;
	m_NullBlock->offset = 0;
	m_NullBlock->prevPhysical = VMA_NULL;
	m_NullBlock->nextPhysical = VMA_NULL;
	m_NullBlock->MarkFree();
	m_NullBlock->NextFree() = VMA_NULL;
	m_NullBlock->PrevFree() = VMA_NULL;
	uint8_t memoryClass = SizeToMemoryClass(size);
	uint16_t sli = SizeToSecondIndex(size, memoryClass);
	m_ListsCount = (memoryClass == 0 ? 0 : (memoryClass - 1) * (1UL << SECOND_LEVEL_INDEX) + sli) + 1;
	if (IsVirtual())
		m_ListsCount += 1UL << SECOND_LEVEL_INDEX;
	else
		m_ListsCount += 4;

	m_MemoryClasses = memoryClass + uint8_t(2);
	oa::memzero(m_InnerIsFreeBitmap, MAX_MEMORY_CLASSES * sizeof(uint32_t));

	m_FreeList = Vma_new_array(GetAllocationCallbacks(), Block*, m_ListsCount);
	oa::fill(m_FreeList, m_FreeList + m_ListsCount, nullptr);
}

bool VmaBlockMetadata_TLSF::Validate() const
{
	VMA_VALIDATE(GetSumFreeSize() <= GetSize());

	VkDeviceSize calculatedSize = m_NullBlock->size;
	VkDeviceSize calculatedFreeSize = m_NullBlock->size;
	size_t allocCount = 0;
	size_t freeCount = 0;

	// Check integrity of free lists
	for (uint32_t list = 0; list < m_ListsCount; ++list)
	{
		Block* block = m_FreeList[list];
		if (block != VMA_NULL)
		{
			VMA_VALIDATE(block->IsFree());
			VMA_VALIDATE(block->PrevFree() == VMA_NULL);
			while (block->NextFree())
			{
				VMA_VALIDATE(block->NextFree()->IsFree());
				VMA_VALIDATE(block->NextFree()->PrevFree() == block);
				block = block->NextFree();
			}
		}
	}

	VkDeviceSize nextOffset = m_NullBlock->offset;
	auto validateCtx = m_GranularityHandler.StartValidation(GetAllocationCallbacks(), IsVirtual());

	VMA_VALIDATE(m_NullBlock->nextPhysical == VMA_NULL);
	if (m_NullBlock->prevPhysical)
	{
		VMA_VALIDATE(m_NullBlock->prevPhysical->nextPhysical == m_NullBlock);
	}
	// Check all blocks
	for (Block* prev = m_NullBlock->prevPhysical; prev != VMA_NULL; prev = prev->prevPhysical)
	{
		VMA_VALIDATE(prev->offset + prev->size == nextOffset);
		nextOffset = prev->offset;
		calculatedSize += prev->size;

		uint32_t listIndex = GetListIndex(prev->size);
		if (prev->IsFree())
		{
			++freeCount;
			// Check if free block belongs to free list
			Block* freeBlock = m_FreeList[listIndex];
			VMA_VALIDATE(freeBlock != VMA_NULL);

			bool found = false;
			do
			{
				if (freeBlock == prev)
					found = true;

				freeBlock = freeBlock->NextFree();
			} while (!found && freeBlock != VMA_NULL);

			VMA_VALIDATE(found);
			calculatedFreeSize += prev->size;
		}
		else
		{
			++allocCount;
			// Check if taken block is not on a free list
			Block* freeBlock = m_FreeList[listIndex];
			while (freeBlock)
			{
				VMA_VALIDATE(freeBlock != prev);
				freeBlock = freeBlock->NextFree();
			}

			if (!IsVirtual())
			{
				VMA_VALIDATE(m_GranularityHandler.Validate(validateCtx, prev->offset, prev->size));
			}
		}

		if (prev->prevPhysical)
		{
			VMA_VALIDATE(prev->prevPhysical->nextPhysical == prev);
		}
	}

	if (!IsVirtual())
	{
		VMA_VALIDATE(m_GranularityHandler.FinishValidation(validateCtx));
	}

	VMA_VALIDATE(nextOffset == 0);
	VMA_VALIDATE(calculatedSize == GetSize());
	VMA_VALIDATE(calculatedFreeSize == GetSumFreeSize());
	VMA_VALIDATE(allocCount == m_AllocCount);
	VMA_VALIDATE(freeCount == m_BlocksFreeCount);

	return true;
}

void VmaBlockMetadata_TLSF::AddDetailedStatistics(VmaDetailedStatistics& inoutStats) const
{
	inoutStats.statistics.blockCount++;
	inoutStats.statistics.blockBytes += GetSize();
	if (m_NullBlock->size > 0)
		VmaAddDetailedStatisticsUnusedRange(inoutStats, m_NullBlock->size);

	for (Block* block = m_NullBlock->prevPhysical; block != VMA_NULL; block = block->prevPhysical)
	{
		if (block->IsFree())
			VmaAddDetailedStatisticsUnusedRange(inoutStats, block->size);
		else
			VmaAddDetailedStatisticsAllocation(inoutStats, block->size);
	}
}

void VmaBlockMetadata_TLSF::AddStatistics(VmaStatistics& inoutStats) const
{
	inoutStats.blockCount++;
	inoutStats.allocationCount += (uint32_t)m_AllocCount;
	inoutStats.blockBytes += GetSize();
	inoutStats.allocationBytes += GetSize() - GetSumFreeSize();
}

#if VMA_STATS_STRING_ENABLED
void VmaBlockMetadata_TLSF::PrintDetailedMap(class VmaJsonWriter& json) const
{
	size_t blockCount = m_AllocCount + m_BlocksFreeCount;
	VmaStlAllocator<Block*> allocator(GetAllocationCallbacks());
	VmaVector<Block*, VmaStlAllocator<Block*>> blockList(blockCount, allocator);

	size_t i = blockCount;
	for (Block* block = m_NullBlock->prevPhysical; block != VMA_NULL; block = block->prevPhysical)
	{
		blockList[--i] = block;
	}
	VMA_ASSERT(i == 0);

	VmaDetailedStatistics stats;
	VmaClearDetailedStatistics(stats);
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

bool VmaBlockMetadata_TLSF::CreateAllocationRequest(
	VkDeviceSize allocSize,
	VkDeviceSize allocAlignment,
	bool upperAddress,
	VmaSuballocationType allocType,
	uint32_t strategy,
	VmaAllocationRequest* pAllocationRequest)
{
	VMA_ASSERT(allocSize > 0 && "Cannot allocate empty block!");
	VMA_ASSERT(!upperAddress && "VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT can be used only with linear algorithm.");

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
		sizeForNextList += (1ULL << (VMA_BITSCAN_MSB(allocSize) - SECOND_LEVEL_INDEX));
	}
	else if (allocSize > SMALL_BUFFER_SIZE - smallSizeStep)
		sizeForNextList = SMALL_BUFFER_SIZE + 1;
	else
		sizeForNextList += smallSizeStep;

	uint32_t nextListIndex = m_ListsCount;
	uint32_t prevListIndex = m_ListsCount;
	Block* nextListBlock = VMA_NULL;
	Block* prevListBlock = VMA_NULL;

	// Check blocks according to strategies
	if (strategy & VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT)
	{
		// Quick check for larger block first
		nextListBlock = FindFreeBlock(sizeForNextList, nextListIndex);
		if (nextListBlock != VMA_NULL && CheckBlock(*nextListBlock, nextListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
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
	else if (strategy & VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT)
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
	else if (strategy & VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT )
	{
		// Perform search from the start
		VmaStlAllocator<Block*> allocator(GetAllocationCallbacks());
		VmaVector<Block*, VmaStlAllocator<Block*>> blockList(m_BlocksFreeCount, allocator);

		size_t i = m_BlocksFreeCount;
		for (Block* block = m_NullBlock->prevPhysical; block != VMA_NULL; block = block->prevPhysical)
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

VkResult VmaBlockMetadata_TLSF::CheckCorruption(const void* pBlockData)
{
	for (Block* block = m_NullBlock->prevPhysical; block != VMA_NULL; block = block->prevPhysical)
	{
		if (!block->IsFree())
		{
			if (!VmaValidateMagicValue(pBlockData, block->offset + block->size))
			{
				VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER VALIDATED ALLOCATION!");
				return VK_ERROR_UNKNOWN_COPY;
			}
		}
	}

	return VK_SUCCESS;
}

void VmaBlockMetadata_TLSF::Alloc(
	const VmaAllocationRequest& request,
	VmaSuballocationType type,
	void* userData)
{
	VMA_ASSERT(request.type == VmaAllocationRequestType::TLSF);

	// Get block and pop it from the free list
	Block* currentBlock = (Block*)request.allocHandle;
	VkDeviceSize offset = request.algorithmData;
	VMA_ASSERT(currentBlock != VMA_NULL);
	VMA_ASSERT(currentBlock->offset <= offset);

	if (currentBlock != m_NullBlock)
		RemoveFreeBlock(currentBlock);

	VkDeviceSize debugMargin = GetDebugMargin();
	VkDeviceSize missingAlignment = offset - currentBlock->offset;

	// Append missing alignment to prev block or create new one
	if (missingAlignment)
	{
		Block* prevBlock = currentBlock->prevPhysical;
		VMA_ASSERT(prevBlock != VMA_NULL && "There should be no missing alignment at offset 0!");

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
			m_NullBlock->nextPhysical = VMA_NULL;
			m_NullBlock->MarkFree();
			m_NullBlock->PrevFree() = VMA_NULL;
			m_NullBlock->NextFree() = VMA_NULL;
			currentBlock->nextPhysical = m_NullBlock;
			currentBlock->MarkTaken();
		}
	}
	else
	{
		VMA_ASSERT(currentBlock->size > size && "Proper block already found, shouldn't find smaller one!");

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
			m_NullBlock->NextFree() = VMA_NULL;
			m_NullBlock->PrevFree() = VMA_NULL;
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

void VmaBlockMetadata_TLSF::Free(VmaAllocHandle allocHandle)
{
	Block* block = (Block*)allocHandle;
	Block* next = block->nextPhysical;
	VMA_ASSERT(!block->IsFree() && "Block is already free!");

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
	if (prev != VMA_NULL && prev->IsFree() && prev->size != debugMargin)
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

void VmaBlockMetadata_TLSF::GetAllocationInfo(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo)
{
	Block* block = (Block*)allocHandle;
	VMA_ASSERT(!block->IsFree() && "Cannot get allocation info for free block!");
	outInfo.offset = block->offset;
	outInfo.size = block->size;
	outInfo.pUserData = block->UserData();
}

void* VmaBlockMetadata_TLSF::GetAllocationUserData(VmaAllocHandle allocHandle) const
{
	Block* block = (Block*)allocHandle;
	VMA_ASSERT(!block->IsFree() && "Cannot get user data for free block!");
	return block->UserData();
}

VmaAllocHandle VmaBlockMetadata_TLSF::GetAllocationListBegin() const
{
	if (m_AllocCount == 0)
		return VK_NULL_HANDLE;

	for (Block* block = m_NullBlock->prevPhysical; block; block = block->prevPhysical)
	{
		if (!block->IsFree())
			return (VmaAllocHandle)block;
	}
	VMA_ASSERT(false && "If m_AllocCount > 0 then should find any allocation!");
	return VK_NULL_HANDLE;
}

VmaAllocHandle VmaBlockMetadata_TLSF::GetNextAllocation(VmaAllocHandle prevAlloc) const
{
	Block* startBlock = (Block*)prevAlloc;
	VMA_ASSERT(!startBlock->IsFree() && "Incorrect block!");

	for (Block* block = startBlock->prevPhysical; block; block = block->prevPhysical)
	{
		if (!block->IsFree())
			return (VmaAllocHandle)block;
	}
	return VK_NULL_HANDLE;
}

VkDeviceSize VmaBlockMetadata_TLSF::GetNextFreeRegionSize(VmaAllocHandle alloc) const
{
	Block* block = (Block*)alloc;
	VMA_ASSERT(!block->IsFree() && "Incorrect block!");

	if (block->prevPhysical)
		return block->prevPhysical->IsFree() ? block->prevPhysical->size : 0;
	return 0;
}

void VmaBlockMetadata_TLSF::Clear()
{
	m_AllocCount = 0;
	m_BlocksFreeCount = 0;
	m_BlocksFreeSize = 0;
	m_IsFreeBitmap = 0;
	m_NullBlock->offset = 0;
	m_NullBlock->size = GetSize();
	Block* block = m_NullBlock->prevPhysical;
	m_NullBlock->prevPhysical = VMA_NULL;
	while (block)
	{
		Block* prev = block->prevPhysical;
		m_BlockAllocator.Free(block);
		block = prev;
	}
	oa::fill(m_FreeList, m_FreeList + m_ListsCount, nullptr);
	oa::memzero(m_InnerIsFreeBitmap, m_MemoryClasses * sizeof(uint32_t));
	m_GranularityHandler.Clear();
}

void VmaBlockMetadata_TLSF::SetAllocationUserData(VmaAllocHandle allocHandle, void* userData)
{
	Block* block = (Block*)allocHandle;
	VMA_ASSERT(!block->IsFree() && "Trying to set user data for not allocated block!");
	block->UserData() = userData;
}

void VmaBlockMetadata_TLSF::DebugLogAllAllocations() const
{
	for (Block* block = m_NullBlock->prevPhysical; block != VMA_NULL; block = block->prevPhysical)
		if (!block->IsFree())
			DebugLogAllocation(block->offset, block->size, block->UserData());
}

uint8_t VmaBlockMetadata_TLSF::SizeToMemoryClass(VkDeviceSize size)
{
	if (size > SMALL_BUFFER_SIZE)
		return uint8_t(VMA_BITSCAN_MSB(size) - MEMORY_CLASS_SHIFT);
	return 0;
}

uint16_t VmaBlockMetadata_TLSF::SizeToSecondIndex(VkDeviceSize size, uint8_t memoryClass) const
{
	if (memoryClass == 0)
	{
		if (IsVirtual())
			return static_cast<uint16_t>((size - 1) / 8);
		return static_cast<uint16_t>((size - 1) / 64);
	}
	return static_cast<uint16_t>((size >> (memoryClass + MEMORY_CLASS_SHIFT - SECOND_LEVEL_INDEX)) ^ (1U << SECOND_LEVEL_INDEX));
}

uint32_t VmaBlockMetadata_TLSF::GetListIndex(uint8_t memoryClass, uint16_t secondIndex) const
{
	if (memoryClass == 0)
		return secondIndex;

	const uint32_t index = static_cast<uint32_t>(memoryClass - 1) * (1 << SECOND_LEVEL_INDEX) + secondIndex;
	if (IsVirtual())
		return index + (1 << SECOND_LEVEL_INDEX);
	return index + 4;
}

uint32_t VmaBlockMetadata_TLSF::GetListIndex(VkDeviceSize size) const
{
	uint8_t memoryClass = SizeToMemoryClass(size);
	return GetListIndex(memoryClass, SizeToSecondIndex(size, memoryClass));
}

void VmaBlockMetadata_TLSF::RemoveFreeBlock(Block* block)
{
	VMA_ASSERT(block != m_NullBlock);
	VMA_ASSERT(block->IsFree());

	if (block->NextFree() != VMA_NULL)
		block->NextFree()->PrevFree() = block->PrevFree();
	if (block->PrevFree() != VMA_NULL)
		block->PrevFree()->NextFree() = block->NextFree();
	else
	{
		uint8_t memClass = SizeToMemoryClass(block->size);
		uint16_t secondIndex = SizeToSecondIndex(block->size, memClass);
		uint32_t index = GetListIndex(memClass, secondIndex);
		VMA_ASSERT(m_FreeList[index] == block);
		m_FreeList[index] = block->NextFree();
		if (block->NextFree() == VMA_NULL)
		{
			m_InnerIsFreeBitmap[memClass] &= ~(1U << secondIndex);
			if (m_InnerIsFreeBitmap[memClass] == 0)
				m_IsFreeBitmap &= ~(1UL << memClass);
		}
	}
	block->MarkTaken();
	block->UserData() = VMA_NULL;
	--m_BlocksFreeCount;
	m_BlocksFreeSize -= block->size;
}

void VmaBlockMetadata_TLSF::InsertFreeBlock(Block* block)
{
	VMA_ASSERT(block != m_NullBlock);
	VMA_ASSERT(!block->IsFree() && "Cannot insert block twice!");

	uint8_t memClass = SizeToMemoryClass(block->size);
	uint16_t secondIndex = SizeToSecondIndex(block->size, memClass);
	uint32_t index = GetListIndex(memClass, secondIndex);
	VMA_ASSERT(index < m_ListsCount);
	block->PrevFree() = VMA_NULL;
	block->NextFree() = m_FreeList[index];
	m_FreeList[index] = block;
	if (block->NextFree() != VMA_NULL)
		block->NextFree()->PrevFree() = block;
	else
	{
		m_InnerIsFreeBitmap[memClass] |= 1U << secondIndex;
		m_IsFreeBitmap |= 1UL << memClass;
	}
	++m_BlocksFreeCount;
	m_BlocksFreeSize += block->size;
}

void VmaBlockMetadata_TLSF::MergeBlock(Block* block, Block* prev)
{
	VMA_ASSERT(block->prevPhysical == prev && "Cannot merge separate physical regions!");
	VMA_ASSERT(!prev->IsFree() && "Cannot merge block that belongs to free list!");

	block->offset = prev->offset;
	block->size += prev->size;
	block->prevPhysical = prev->prevPhysical;
	if (block->prevPhysical)
		block->prevPhysical->nextPhysical = block;
	m_BlockAllocator.Free(prev);
}

VmaBlockMetadata_TLSF::Block* VmaBlockMetadata_TLSF::FindFreeBlock(VkDeviceSize size, uint32_t& listIndex) const
{
	uint8_t memoryClass = SizeToMemoryClass(size);
	uint32_t innerFreeMap = m_InnerIsFreeBitmap[memoryClass] & (~0U << SizeToSecondIndex(size, memoryClass));
	if (!innerFreeMap)
	{
		// Check higher levels for available blocks
		uint32_t freeMap = m_IsFreeBitmap & (~0UL << (memoryClass + 1));
		if (!freeMap)
			return VMA_NULL; // No more memory available

		// Find lowest free region
		memoryClass = VMA_BITSCAN_LSB(freeMap);
		innerFreeMap = m_InnerIsFreeBitmap[memoryClass];
		VMA_ASSERT(innerFreeMap != 0);
	}
	// Find lowest free subregion
	listIndex = GetListIndex(memoryClass, VMA_BITSCAN_LSB(innerFreeMap));
	VMA_ASSERT(m_FreeList[listIndex]);
	return m_FreeList[listIndex];
}

bool VmaBlockMetadata_TLSF::CheckBlock(
	Block& block,
	uint32_t listIndex,
	VkDeviceSize allocSize,
	VkDeviceSize allocAlignment,
	VmaSuballocationType allocType,
	VmaAllocationRequest* pAllocationRequest)
{
	VMA_ASSERT(block.IsFree() && "Block is already taken!");

	VkDeviceSize alignedOffset = VmaAlignUp(block.offset, allocAlignment);
	if (block.size < allocSize + alignedOffset - block.offset)
		return false;

	// Check for granularity conflicts
	if (!IsVirtual() &&
		m_GranularityHandler.CheckConflictAndAlignUp(alignedOffset, allocSize, block.offset, block.size, allocType))
		return false;

	// Alloc successful
	pAllocationRequest->type = VmaAllocationRequestType::TLSF;
	pAllocationRequest->allocHandle = (VmaAllocHandle)&block;
	pAllocationRequest->size = allocSize - GetDebugMargin();
	pAllocationRequest->customData = (void*)allocType;
	pAllocationRequest->algorithmData = alignedOffset;

	// Place block at the start of list if it's normal block
	if (listIndex != m_ListsCount && block.PrevFree())
	{
		block.PrevFree()->NextFree() = block.NextFree();
		if (block.NextFree())
			block.NextFree()->PrevFree() = block.PrevFree();
		block.PrevFree() = VMA_NULL;
		block.NextFree() = m_FreeList[listIndex];
		m_FreeList[listIndex] = &block;
		if (block.NextFree())
			block.NextFree()->PrevFree() = &block;
	}

	return true;
}
#endif // _VMA_BLOCK_METADATA_TLSF_FUNCTIONS
#endif // _VMA_BLOCK_METADATA_TLSF
