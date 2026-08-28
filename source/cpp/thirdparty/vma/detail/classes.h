// SPDX: MIT. Derived from Vulkan Memory Allocator, (c) 2017-2026 Advanced Micro Devices, Inc.
// Full MIT copyright/permission notice: source/cpp/thirdparty/vma/vma.h. See NOTICE.md.
#ifndef _VMA_BLOCK_VECTOR
/*
Sequence of VmaDeviceMemoryBlock. Represents memory blocks allocated for a specific
Vulkan memory type.

Synchronized internally with a mutex.
*/
class VmaBlockVector {
	friend struct VmaDefragmentationContext_T;
	VMA_CLASS_NO_COPY_NO_MOVE(VmaBlockVector)
public:
	VmaBlockVector(
		VmaAllocator hAllocator,
		VmaPool hParentPool,
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
	~VmaBlockVector();

	VmaAllocator GetAllocator() const { return m_hAllocator; }
	VmaPool GetParentPool() const { return m_hParentPool; }
	bool IsCustomPool() const { return m_hParentPool != VMA_NULL; }
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
	VmaDeviceMemoryBlock* GetBlock(size_t index) const { return m_Blocks[index]; }
	VMA_RW_MUTEX &GetMutex() { return m_Mutex; }

	VkResult CreateMinBlocks();
	void AddStatistics(VmaStatistics& inoutStats);
	void AddDetailedStatistics(VmaDetailedStatistics& inoutStats);
	bool IsEmpty();
	bool IsCorruptionDetectionEnabled() const;

	VkResult Allocate(
		VkDeviceSize size,
		VkDeviceSize alignment,
		const VmaAllocationCreateInfo& createInfo,
		VmaSuballocationType suballocType,
		size_t allocationCount,
		VmaAllocation* pAllocations);

	void Free(VmaAllocation hAllocation);

#if VMA_STATS_STRING_ENABLED
	void PrintDetailedMap(class VmaJsonWriter& json);
#endif

	VkResult CheckCorruption();

private:
	const VmaAllocator m_hAllocator;
	const VmaPool m_hParentPool;
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
	VMA_RW_MUTEX m_Mutex;
	// Incrementally sorted by sumFreeSize, ascending.
	VmaVector<VmaDeviceMemoryBlock*, VmaStlAllocator<VmaDeviceMemoryBlock*>> m_Blocks;
	uint32_t m_NextBlockId;
	bool m_IncrementalSort = true;

	void SetIncrementalSort(bool val) { m_IncrementalSort = val; }

	VkDeviceSize CalcMaxBlockSize() const;
	// Finds and removes given block from vector.
	void Remove(VmaDeviceMemoryBlock* pBlock);
	// Performs single step in sorting m_Blocks. They may not be fully sorted
	// after this call.
	void IncrementallySortBlocks();
	void SortByFreeSize();

	VkResult AllocatePage(
		VkDeviceSize size,
		VkDeviceSize alignment,
		const VmaAllocationCreateInfo& createInfo,
		VmaSuballocationType suballocType,
		VmaAllocation* pAllocation);

	VkResult AllocateFromBlock(
		VmaDeviceMemoryBlock* pBlock,
		VkDeviceSize size,
		VkDeviceSize alignment,
		VmaAllocationCreateFlags allocFlags,
		void* pUserData,
		VmaSuballocationType suballocType,
		uint32_t strategy,
		VmaAllocation* pAllocation);

	VkResult CommitAllocationRequest(
		VmaAllocationRequest& allocRequest,
		VmaDeviceMemoryBlock* pBlock,
		VkDeviceSize alignment,
		VmaAllocationCreateFlags allocFlags,
		void* pUserData,
		VmaSuballocationType suballocType,
		VmaAllocation* pAllocation);

	VkResult CreateBlock(VkDeviceSize blockSize, size_t* pNewBlockIndex);
	bool HasEmptyBlock();
};
#endif // _VMA_BLOCK_VECTOR

#ifndef _VMA_DEFRAGMENTATION_CONTEXT
struct VmaDefragmentationContext_T
{
	VMA_CLASS_NO_COPY_NO_MOVE(VmaDefragmentationContext_T)
public:
	VmaDefragmentationContext_T(
		VmaAllocator hAllocator,
		const VmaDefragmentationInfo& info);
	~VmaDefragmentationContext_T();

	void GetStats(VmaDefragmentationStats& outStats) { outStats = m_GlobalStats; }

	VkResult DefragmentPassBegin(VmaDefragmentationPassMoveInfo& moveInfo);
	VkResult DefragmentPassEnd(VmaDefragmentationPassMoveInfo& moveInfo);

private:
	// Max number of allocations to ignore due to size constraints before ending single pass
	static constexpr uint8_t MAX_ALLOCS_TO_IGNORE = 16;
	enum class CounterStatus { Pass, Ignore, End };

	struct FragmentedBlock
	{
		uint32_t data;
		VmaDeviceMemoryBlock* block;
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
		VmaSuballocationType type;
		VmaAllocationCreateFlags flags;
		VmaDefragmentationMove move = {};
	};

	const VkDeviceSize m_MaxPassBytes;
	const uint32_t m_MaxPassAllocations;
	const PFN_VmaCheckDefragmentationBreakFunction m_BreakCallback;
	void* m_BreakCallbackUserData;

	VmaStlAllocator<VmaDefragmentationMove> m_MoveAllocator;
	VmaVector<VmaDefragmentationMove, VmaStlAllocator<VmaDefragmentationMove>> m_Moves;

	uint8_t m_IgnoredAllocs = 0;
	uint32_t m_Algorithm;
	uint32_t m_BlockVectorCount;
	VmaBlockVector* m_PoolBlockVector;
	VmaBlockVector** m_pBlockVectors;
	size_t m_ImmovableBlockCount = 0;
	VmaDefragmentationStats m_GlobalStats = { 0 };
	VmaDefragmentationStats m_PassStats = { 0 };
	void* m_AlgorithmState = VMA_NULL;

	static MoveAllocationData GetMoveData(VmaAllocHandle handle, VmaBlockMetadata* metadata);
	CounterStatus CheckCounters(VkDeviceSize bytes);
	bool IncrementCounters(VkDeviceSize bytes);
	bool ReallocWithinBlock(VmaBlockVector& vector, VmaDeviceMemoryBlock* block);
	bool AllocInOtherBlock(size_t start, size_t end, MoveAllocationData& data, VmaBlockVector& vector);

	bool ComputeDefragmentation(VmaBlockVector& vector, size_t index);
	bool ComputeDefragmentation_Fast(VmaBlockVector& vector);
	bool ComputeDefragmentation_Balanced(VmaBlockVector& vector, size_t index, bool update);
	bool ComputeDefragmentation_Full(VmaBlockVector& vector);
	bool ComputeDefragmentation_Extensive(VmaBlockVector& vector, size_t index);

	static void UpdateVectorStatistics(VmaBlockVector& vector, StateBalanced& state);
	bool MoveDataToFreeBlocks(VmaSuballocationType currentType,
		VmaBlockVector& vector, size_t firstFreeBlock,
		bool& texturePresent, bool& bufferPresent, bool& otherPresent);
};
#endif // _VMA_DEFRAGMENTATION_CONTEXT

#ifndef _VMA_POOL_T
struct VmaPool_T
{
	friend struct VmaPoolListItemTraits;
	VMA_CLASS_NO_COPY_NO_MOVE(VmaPool_T)
public:
	VmaBlockVector m_BlockVector;
	VmaDedicatedAllocationList m_DedicatedAllocations;

	VmaPool_T(
		VmaAllocator hAllocator,
		const VmaPoolCreateInfo& createInfo,
		VkDeviceSize preferredBlockSize);
	~VmaPool_T();

	uint32_t GetId() const { return m_Id; }
	void SetId(uint32_t id) { VMA_ASSERT(m_Id == 0); m_Id = id; }

	const char* GetName() const { return m_Name; }
	void SetName(const char* pName);

#if VMA_STATS_STRING_ENABLED
	//void PrintDetailedMap(class VmaStringBuilder& sb);
#endif

private:
	uint32_t m_Id;
	char* m_Name;
	VmaPool_T* m_PrevPool = VMA_NULL;
	VmaPool_T* m_NextPool = VMA_NULL;
};

struct VmaPoolListItemTraits
{
	typedef VmaPool_T ItemType;

	static ItemType* GetPrev(const ItemType* item) { return item->m_PrevPool; }
	static ItemType* GetNext(const ItemType* item) { return item->m_NextPool; }
	static ItemType*& AccessPrev(ItemType* item) { return item->m_PrevPool; }
	static ItemType*& AccessNext(ItemType* item) { return item->m_NextPool; }
};
#endif // _VMA_POOL_T

#ifndef _VMA_CURRENT_BUDGET_DATA
struct VmaCurrentBudgetData
{
	VMA_CLASS_NO_COPY_NO_MOVE(VmaCurrentBudgetData)
public:

	VMA_ATOMIC_UINT32 m_BlockCount[VK_MAX_MEMORY_HEAPS];
	VMA_ATOMIC_UINT32 m_AllocationCount[VK_MAX_MEMORY_HEAPS];
	VMA_ATOMIC_UINT64 m_BlockBytes[VK_MAX_MEMORY_HEAPS];
	VMA_ATOMIC_UINT64 m_AllocationBytes[VK_MAX_MEMORY_HEAPS];

#if VMA_MEMORY_BUDGET
	VMA_ATOMIC_UINT32 m_OperationsSinceBudgetFetch;
	VMA_RW_MUTEX m_BudgetMutex;
	uint64_t m_VulkanUsage[VK_MAX_MEMORY_HEAPS];
	uint64_t m_VulkanBudget[VK_MAX_MEMORY_HEAPS];
	uint64_t m_BlockBytesAtBudgetFetch[VK_MAX_MEMORY_HEAPS];
#endif // VMA_MEMORY_BUDGET

	VmaCurrentBudgetData();

	void AddAllocation(uint32_t heapIndex, VkDeviceSize allocationSize);
	void RemoveAllocation(uint32_t heapIndex, VkDeviceSize allocationSize);
};

#ifndef _VMA_CURRENT_BUDGET_DATA_FUNCTIONS
VmaCurrentBudgetData::VmaCurrentBudgetData()
{
	for (uint32_t heapIndex = 0; heapIndex < VK_MAX_MEMORY_HEAPS; ++heapIndex)
	{
		m_BlockCount[heapIndex] = 0;
		m_AllocationCount[heapIndex] = 0;
		m_BlockBytes[heapIndex] = 0;
		m_AllocationBytes[heapIndex] = 0;
#if VMA_MEMORY_BUDGET
		m_VulkanUsage[heapIndex] = 0;
		m_VulkanBudget[heapIndex] = 0;
		m_BlockBytesAtBudgetFetch[heapIndex] = 0;
#endif
	}

#if VMA_MEMORY_BUDGET
	m_OperationsSinceBudgetFetch = 0;
#endif
}

void VmaCurrentBudgetData::AddAllocation(uint32_t heapIndex, VkDeviceSize allocationSize)
{
	m_AllocationBytes[heapIndex] += allocationSize;
	++m_AllocationCount[heapIndex];
#if VMA_MEMORY_BUDGET
	++m_OperationsSinceBudgetFetch;
#endif
}

void VmaCurrentBudgetData::RemoveAllocation(uint32_t heapIndex, VkDeviceSize allocationSize)
{
	VMA_ASSERT(m_AllocationBytes[heapIndex] >= allocationSize);
	m_AllocationBytes[heapIndex] -= allocationSize;
	VMA_ASSERT(m_AllocationCount[heapIndex] > 0);
	--m_AllocationCount[heapIndex];
#if VMA_MEMORY_BUDGET
	++m_OperationsSinceBudgetFetch;
#endif
}
#endif // _VMA_CURRENT_BUDGET_DATA_FUNCTIONS
#endif // _VMA_CURRENT_BUDGET_DATA

#ifndef _VMA_ALLOCATION_OBJECT_ALLOCATOR
/*
Thread-safe wrapper over VmaPoolAllocator free list, for allocation of VmaAllocation_T objects.
*/
class VmaAllocationObjectAllocator
{
	VMA_CLASS_NO_COPY_NO_MOVE(VmaAllocationObjectAllocator)
public:
	explicit VmaAllocationObjectAllocator(const VkAllocationCallbacks* pAllocationCallbacks)
		: m_Allocator(pAllocationCallbacks, 1024) {}

	template<typename... Types> VmaAllocation Allocate(Types&&... args);
	void Free(VmaAllocation hAlloc);

private:
	VMA_MUTEX m_Mutex;
	VmaPoolAllocator<VmaAllocation_T> m_Allocator;
};

template<typename... Types>
VmaAllocation VmaAllocationObjectAllocator::Allocate(Types&&... args)
{
	VmaMutexLock mutexLock(m_Mutex);
	return m_Allocator.Alloc<Types...>(oa::forward<Types>(args)...);
}

void VmaAllocationObjectAllocator::Free(VmaAllocation hAlloc)
{
	VmaMutexLock mutexLock(m_Mutex);
	m_Allocator.Free(hAlloc);
}
#endif // _VMA_ALLOCATION_OBJECT_ALLOCATOR

#ifndef _VMA_VIRTUAL_BLOCK_T
struct VmaVirtualBlock_T
{
	VMA_CLASS_NO_COPY_NO_MOVE(VmaVirtualBlock_T)
public:
	const bool m_AllocationCallbacksSpecified;
	const VkAllocationCallbacks m_AllocationCallbacks;

	explicit VmaVirtualBlock_T(const VmaVirtualBlockCreateInfo& createInfo);
	~VmaVirtualBlock_T();

	bool IsEmpty() const { return m_Metadata->IsEmpty(); }
	void Free(VmaVirtualAllocation allocation) { m_Metadata->Free((VmaAllocHandle)allocation); }
	void SetAllocationUserData(VmaVirtualAllocation allocation, void* userData) { m_Metadata->SetAllocationUserData((VmaAllocHandle)allocation, userData); }
	void Clear() { m_Metadata->Clear(); }

	const VkAllocationCallbacks* GetAllocationCallbacks() const;
	void GetAllocationInfo(VmaVirtualAllocation allocation, VmaVirtualAllocationInfo& outInfo);
	VkResult Allocate(const VmaVirtualAllocationCreateInfo& createInfo, VmaVirtualAllocation& outAllocation,
		VkDeviceSize* outOffset);
	void GetStatistics(VmaStatistics& outStats) const;
	void CalculateDetailedStatistics(VmaDetailedStatistics& outStats) const;
#if VMA_STATS_STRING_ENABLED
	void BuildStatsString(bool detailedMap, VmaStringBuilder& sb) const;
#endif

private:
	VmaBlockMetadata* m_Metadata;
};

#ifndef _VMA_VIRTUAL_BLOCK_T_FUNCTIONS
VmaVirtualBlock_T::VmaVirtualBlock_T(const VmaVirtualBlockCreateInfo& createInfo)
	: m_AllocationCallbacksSpecified(createInfo.pAllocationCallbacks != VMA_NULL),
	m_AllocationCallbacks(createInfo.pAllocationCallbacks != VMA_NULL ? *createInfo.pAllocationCallbacks : VmaEmptyAllocationCallbacks)
{
	const uint32_t algorithm = createInfo.flags & VMA_VIRTUAL_BLOCK_CREATE_ALGORITHM_MASK;
	switch (algorithm)
	{
	case 0:
		m_Metadata = Vma_new(GetAllocationCallbacks(), VmaBlockMetadata_TLSF)(VK_NULL_HANDLE, 1, true);
		break;
	case VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT:
		m_Metadata = Vma_new(GetAllocationCallbacks(), VmaBlockMetadata_Linear)(VK_NULL_HANDLE, 1, true);
		break;
	default:
		VMA_ASSERT(0);
		m_Metadata = Vma_new(GetAllocationCallbacks(), VmaBlockMetadata_TLSF)(VK_NULL_HANDLE, 1, true);
	}

	m_Metadata->Init(createInfo.size);
}

VmaVirtualBlock_T::~VmaVirtualBlock_T()
{
	// Define macro VMA_DEBUG_LOG_FORMAT or more specialized VMA_LEAK_LOG_FORMAT
	// to receive the list of the unfreed allocations.
	if (!m_Metadata->IsEmpty())
		m_Metadata->DebugLogAllAllocations();
	// This is the most important assert in the entire library.
	// Hitting it means you have some memory leak - unreleased virtual allocations.
	VMA_ASSERT_LEAK(m_Metadata->IsEmpty() && "Some virtual allocations were not freed before destruction of this virtual block!");

	Vma_delete(GetAllocationCallbacks(), m_Metadata);
}

const VkAllocationCallbacks* VmaVirtualBlock_T::GetAllocationCallbacks() const
{
	return m_AllocationCallbacksSpecified ? &m_AllocationCallbacks : VMA_NULL;
}

void VmaVirtualBlock_T::GetAllocationInfo(VmaVirtualAllocation allocation, VmaVirtualAllocationInfo& outInfo)
{
	m_Metadata->GetAllocationInfo((VmaAllocHandle)allocation, outInfo);
}

VkResult VmaVirtualBlock_T::Allocate(const VmaVirtualAllocationCreateInfo& createInfo, VmaVirtualAllocation& outAllocation,
	VkDeviceSize* outOffset)
{
	VmaAllocationRequest request = {};
	if (m_Metadata->CreateAllocationRequest(
		createInfo.size, // allocSize
		VMA_MAX(createInfo.alignment, (VkDeviceSize)1), // allocAlignment
		(createInfo.flags & VMA_VIRTUAL_ALLOCATION_CREATE_UPPER_ADDRESS_BIT) != 0, // upperAddress
		VMA_SUBALLOCATION_TYPE_UNKNOWN, // allocType - unimportant
		createInfo.flags & VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MASK, // strategy
		&request))
	{
		m_Metadata->Alloc(request,
			VMA_SUBALLOCATION_TYPE_UNKNOWN, // type - unimportant
			createInfo.pUserData);
		outAllocation = (VmaVirtualAllocation)request.allocHandle;
		if(outOffset)
			*outOffset = m_Metadata->GetAllocationOffset(request.allocHandle);
		return VK_SUCCESS;
	}
	outAllocation = (VmaVirtualAllocation)VK_NULL_HANDLE;
	if (outOffset)
		*outOffset = UINT64_MAX;
	return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

void VmaVirtualBlock_T::GetStatistics(VmaStatistics& outStats) const
{
	VmaClearStatistics(outStats);
	m_Metadata->AddStatistics(outStats);
}

void VmaVirtualBlock_T::CalculateDetailedStatistics(VmaDetailedStatistics& outStats) const
{
	VmaClearDetailedStatistics(outStats);
	m_Metadata->AddDetailedStatistics(outStats);
}

#if VMA_STATS_STRING_ENABLED
void VmaVirtualBlock_T::BuildStatsString(bool detailedMap, VmaStringBuilder& sb) const
{
	VmaJsonWriter json(GetAllocationCallbacks(), sb);
	json.BeginObject();

	VmaDetailedStatistics stats;
	CalculateDetailedStatistics(stats);

	json.WriteString("Stats");
	VmaPrintDetailedStatistics(json, stats);

	if (detailedMap)
	{
		json.WriteString("Details");
		json.BeginObject();
		m_Metadata->PrintDetailedMap(json);
		json.EndObject();
	}

	json.EndObject();
}
#endif // VMA_STATS_STRING_ENABLED
#endif // _VMA_VIRTUAL_BLOCK_T_FUNCTIONS
#endif // _VMA_VIRTUAL_BLOCK_T


// Main allocator object.
struct VmaAllocator_T
{
	VMA_CLASS_NO_COPY_NO_MOVE(VmaAllocator_T)
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
	VmaDeviceMemoryCallbacks m_DeviceMemoryCallbacks;
	VmaAllocationObjectAllocator m_AllocationObjectAllocator;

	// Each bit (1 << i) is set if HeapSizeLimit is enabled for that heap, so cannot allocate more than the heap size.
	uint32_t m_HeapSizeLimitMask;

	VkPhysicalDeviceProperties m_PhysicalDeviceProperties;
	VkPhysicalDeviceMemoryProperties m_MemProps;

	// Default pools.
	VmaBlockVector* m_pBlockVectors[VK_MAX_MEMORY_TYPES];
	VmaDedicatedAllocationList m_DedicatedAllocations[VK_MAX_MEMORY_TYPES];

	VmaCurrentBudgetData m_Budget;
	VMA_ATOMIC_UINT32 m_DeviceMemoryCount; // Total number of VkDeviceMemory objects.

	explicit VmaAllocator_T(const VmaAllocatorCreateInfo* pCreateInfo);
	VkResult Init(const VmaAllocatorCreateInfo* pCreateInfo);
	~VmaAllocator_T();

	const VkAllocationCallbacks* GetAllocationCallbacks() const
	{
		return m_AllocationCallbacksSpecified ? &m_AllocationCallbacks : VMA_NULL;
	}
	const VmaVulkanFunctions& GetVulkanFunctions() const
	{
		return m_VulkanFunctions;
	}

	VkPhysicalDevice GetPhysicalDevice() const { return m_PhysicalDevice; }

	VkDeviceSize GetBufferImageGranularity() const
	{
		return VMA_MAX(
			static_cast<VkDeviceSize>(VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY),
			m_PhysicalDeviceProperties.limits.bufferImageGranularity);
	}

	uint32_t GetMemoryHeapCount() const { return m_MemProps.memoryHeapCount; }
	uint32_t GetMemoryTypeCount() const { return m_MemProps.memoryTypeCount; }

	uint32_t MemoryTypeIndexToHeapIndex(uint32_t memTypeIndex) const
	{
		VMA_ASSERT(memTypeIndex < m_MemProps.memoryTypeCount);
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
			VMA_MAX((VkDeviceSize)VMA_MIN_ALIGNMENT, m_PhysicalDeviceProperties.limits.nonCoherentAtomSize) :
			(VkDeviceSize)VMA_MIN_ALIGNMENT;
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
		const VmaAllocationCreateInfo* pAllocationCreateInfo,
		VmaBufferImageUsage bufImgUsage,
		uint32_t* pMemoryTypeIndex) const;

	// Common code for public functions vmaCreateBuffer, vmaCreateBufferWithAlignment, etc.
	VkResult CreateBuffer(
		const VkBufferCreateInfo* pBufferCreateInfo,
		const VmaAllocationCreateInfo* pAllocationCreateInfo,
		void* pMemoryAllocateNext, // pNext chain for VkMemoryAllocateInfo.
		VkBuffer* pBuffer,
		VmaAllocation* pAllocation,
		VmaAllocationInfo* pAllocationInfo);
	// Common code for public functions vmaCreateImage, vmaCreateDedicatedImage.
	VkResult CreateImage(
		const VkImageCreateInfo* pImageCreateInfo,
		const VmaAllocationCreateInfo* pAllocationCreateInfo,
		void* pMemoryAllocateNext, // pNext chain for VkMemoryAllocateInfo.
		VkImage* pImage,
		VmaAllocation* pAllocation,
		VmaAllocationInfo* pAllocationInfo);

	// Main allocation function.
	VkResult AllocateMemory(
		VkMemoryRequirements vkMemReq,
		bool requiresDedicatedAllocation,
		bool prefersDedicatedAllocation,
		VkBuffer dedicatedBuffer,
		VkImage dedicatedImage,
		VmaBufferImageUsage dedicatedBufferImageUsage,
		void* pMemoryAllocateNext, // Optional pNext chain for VkMemoryAllocateInfo.
		const VmaAllocationCreateInfo& createInfo,
		VmaSuballocationType suballocType,
		size_t allocationCount,
		VmaAllocation* pAllocations);

	// Main deallocation function.
	void FreeMemory(
		size_t allocationCount,
		const VmaAllocation* pAllocations);

	void CalculateStatistics(VmaTotalStatistics* pStats);

	void GetHeapBudgets(
		VmaBudget* outBudgets, uint32_t firstHeap, uint32_t heapCount);

#if VMA_STATS_STRING_ENABLED
	void PrintDetailedMap(class VmaJsonWriter& json);
#endif

	static void GetAllocationInfo(VmaAllocation hAllocation, VmaAllocationInfo* pAllocationInfo);
	static void GetAllocationInfo2(VmaAllocation hAllocation, VmaAllocationInfo2* pAllocationInfo);

	VkResult CreatePool(const VmaPoolCreateInfo* pCreateInfo, VmaPool* pPool);
	void DestroyPool(VmaPool pool);
	static void GetPoolStatistics(VmaPool pool, VmaStatistics* pPoolStats);
	static void CalculatePoolStatistics(VmaPool pool, VmaDetailedStatistics* pPoolStats);

	void SetCurrentFrameIndex(uint32_t frameIndex);
	uint32_t GetCurrentFrameIndex() const { return m_CurrentFrameIndex.load(); }

	static VkResult CheckPoolCorruption(VmaPool hPool);
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

	VkResult Map(VmaAllocation hAllocation, void** ppData);
	void Unmap(VmaAllocation hAllocation);

	VkResult BindBufferMemory(
		VmaAllocation hAllocation,
		VkDeviceSize allocationLocalOffset,
		VkBuffer hBuffer,
		const void* pNext);
	VkResult BindImageMemory(
		VmaAllocation hAllocation,
		VkDeviceSize allocationLocalOffset,
		VkImage hImage,
		const void* pNext);

	VkResult FlushOrInvalidateAllocation(
		VmaAllocation hAllocation,
		VkDeviceSize offset, VkDeviceSize size,
		VMA_CACHE_OPERATION op);
	VkResult FlushOrInvalidateAllocations(
		uint32_t allocationCount,
		const VmaAllocation* allocations,
		const VkDeviceSize* offsets, const VkDeviceSize* sizes,
		VMA_CACHE_OPERATION op);

	VkResult CopyMemoryToAllocation(
		const void* pSrcHostPointer,
		VmaAllocation dstAllocation,
		VkDeviceSize dstAllocationLocalOffset,
		VkDeviceSize size);
	VkResult CopyAllocationToMemory(
		VmaAllocation srcAllocation,
		VkDeviceSize srcAllocationLocalOffset,
		void* pDstHostPointer,
		VkDeviceSize size);

	void FillAllocation(VmaAllocation hAllocation, uint8_t pattern);

	/*
	Returns bit mask of memory types that can support defragmentation on GPU as
	they support creation of required buffer for copy operations.
	*/
	uint32_t GetGpuDefragmentationMemoryTypeBits();

#if VMA_EXTERNAL_MEMORY
	VkExternalMemoryHandleTypeFlagsKHR GetExternalMemoryHandleTypeFlags(uint32_t memTypeIndex) const
	{
		return m_TypeExternalMemoryHandleTypes[memTypeIndex];
	}
#endif // #if VMA_EXTERNAL_MEMORY

private:
	VkDeviceSize m_PreferredLargeHeapBlockSize;

	VkPhysicalDevice m_PhysicalDevice;
	VMA_ATOMIC_UINT32 m_CurrentFrameIndex;
	VMA_ATOMIC_UINT32 m_GpuDefragmentationMemoryTypeBits; // UINT32_MAX means uninitialized.
#if VMA_EXTERNAL_MEMORY
	VkExternalMemoryHandleTypeFlagsKHR m_TypeExternalMemoryHandleTypes[VK_MAX_MEMORY_TYPES];
#endif // #if VMA_EXTERNAL_MEMORY

	VMA_RW_MUTEX m_PoolsMutex;
	typedef VmaIntrusiveLinkedList<VmaPoolListItemTraits> PoolList;
	// Protected by m_PoolsMutex.
	PoolList m_Pools;
	uint32_t m_NextPoolId;

	VmaVulkanFunctions m_VulkanFunctions;

	// Global bit mask AND-ed with any memoryTypeBits to disallow certain memory types.
	uint32_t m_GlobalMemoryTypeBits;

	void ImportVulkanFunctions(const VmaVulkanFunctions* pVulkanFunctions);

#if VMA_STATIC_VULKAN_FUNCTIONS == 1
	void ImportVulkanFunctions_Static();
#endif

	void ImportVulkanFunctions_Custom(const VmaVulkanFunctions* pVulkanFunctions);

#if VMA_DYNAMIC_VULKAN_FUNCTIONS == 1
	void ImportVulkanFunctions_Dynamic();
#endif

	void ValidateVulkanFunctions() const;

	VkDeviceSize CalcPreferredBlockSize(uint32_t memTypeIndex);

	VkResult AllocateMemoryOfType(
		VmaPool pool,
		VkDeviceSize size,
		VkDeviceSize alignment,
		bool dedicatedPreferred,
		VkBuffer dedicatedBuffer,
		VkImage dedicatedImage,
		VmaBufferImageUsage dedicatedBufferImageUsage,
		void* pMemoryAllocateNext, // Optional pNext chain for VkMemoryAllocateInfo.
		const VmaAllocationCreateInfo& createInfo,
		uint32_t memTypeIndex,
		VmaSuballocationType suballocType,
		VmaDedicatedAllocationList& dedicatedAllocations,
		VmaBlockVector& blockVector,
		size_t allocationCount,
		VmaAllocation* pAllocations);

	// Helper function only to be used inside AllocateDedicatedMemory.
	VkResult AllocateDedicatedMemoryPage(
		VmaPool pool,
		VkDeviceSize size,
		VmaSuballocationType suballocType,
		uint32_t memTypeIndex,
		const VkMemoryAllocateInfo& allocInfo,
		bool map,
		bool isUserDataString,
		bool isMappingAllowed,
		void* pUserData,
		VmaAllocation* pAllocation);

	// Allocates and registers new VkDeviceMemory specifically for dedicated allocations.
	VkResult AllocateDedicatedMemory(
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
		const void* pNextChain);

	void FreeDedicatedMemory(VmaAllocation allocation);

	VkResult CalcMemTypeParams(
		VmaAllocationCreateInfo& outCreateInfo,
		uint32_t memTypeIndex,
		VkDeviceSize size,
		size_t allocationCount);
	static VkResult CalcAllocationParams(
		VmaAllocationCreateInfo& outCreateInfo,
		bool dedicatedRequired);

	/*
	Calculates and returns bit mask of memory types that can support defragmentation
	on GPU as they support creation of required buffer for copy operations.
	*/
	uint32_t CalculateGpuDefragmentationMemoryTypeBits() const;
	uint32_t CalculateGlobalMemoryTypeBits() const;

	bool GetFlushOrInvalidateRange(
		VmaAllocation allocation,
		VkDeviceSize offset, VkDeviceSize size,
		VkMappedMemoryRange& outRange) const;

#if VMA_MEMORY_BUDGET
	void UpdateVulkanBudget();
#endif // #if VMA_MEMORY_BUDGET
};
