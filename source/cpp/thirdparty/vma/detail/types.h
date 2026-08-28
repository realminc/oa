// SPDX: MIT. Derived from Vulkan Memory Allocator, (c) 2017-2026 Advanced Micro Devices, Inc.
// Full MIT copyright/permission notice: source/cpp/thirdparty/vma/vma.h. See NOTICE.md.
#ifndef _VMA_MAPPING_HYSTERESIS

class VmaMappingHysteresis {
	VMA_CLASS_NO_COPY_NO_MOVE(VmaMappingHysteresis)
public:
	VmaMappingHysteresis() = default;

	uint32_t GetExtraMapping() const { return m_ExtraMapping; }

	// Call when Map was called.
	// Returns true if switched to extra +1 mapping reference count.
	bool PostMap() {
#if VMA_MAPPING_HYSTERESIS_ENABLED
		if(m_ExtraMapping == 0) {
			++m_MajorCounter;
			if(m_MajorCounter >= COUNTER_MIN_EXTRA_MAPPING)	{
				m_ExtraMapping = 1;
				m_MajorCounter = 0;
				m_MinorCounter = 0;
				return true;
			}
		}	else { // m_ExtraMapping == 1
			PostMinorCounter();
		}
#endif // #if VMA_MAPPING_HYSTERESIS_ENABLED
		return false;
	}

	// Call when Unmap was called.
	void PostUnmap() {
#if VMA_MAPPING_HYSTERESIS_ENABLED
		if(m_ExtraMapping == 0) {
			++m_MajorCounter;
		}	else {
			PostMinorCounter();
		} // m_ExtraMapping == 1
#endif // #if VMA_MAPPING_HYSTERESIS_ENABLED
	}

	// Call when allocation was made from the memory block.
	void PostAlloc() {
#if VMA_MAPPING_HYSTERESIS_ENABLED
		if(m_ExtraMapping == 1) {
			++m_MajorCounter;
		}	else { // m_ExtraMapping == 0
			PostMinorCounter();
		}
#endif // #if VMA_MAPPING_HYSTERESIS_ENABLED
	}

	// Call when allocation was freed from the memory block.
	// Returns true if switched to extra -1 mapping reference count.
	bool PostFree()	{
#if VMA_MAPPING_HYSTERESIS_ENABLED
		if(m_ExtraMapping == 1)	{
			++m_MajorCounter;
			if(m_MajorCounter >= COUNTER_MIN_EXTRA_MAPPING &&	m_MajorCounter > m_MinorCounter + 1) {
				m_ExtraMapping = 0;
				m_MajorCounter = 0;
				m_MinorCounter = 0;
				return true;
			}
		}	else {// m_ExtraMapping == 0
			PostMinorCounter();
		}
#endif // #if VMA_MAPPING_HYSTERESIS_ENABLED
		return false;
	}

private:
	static constexpr int32_t COUNTER_MIN_EXTRA_MAPPING = 7;

	uint32_t m_MinorCounter = 0;
	uint32_t m_MajorCounter = 0;
	uint32_t m_ExtraMapping = 0; // 0 or 1.

	void PostMinorCounter()	{
		if(m_MinorCounter < m_MajorCounter)	{
			++m_MinorCounter;
		}	else if(m_MajorCounter > 0)	{
			--m_MajorCounter;
			--m_MinorCounter;
		}
	}
};

#endif // _VMA_MAPPING_HYSTERESIS

#if VMA_EXTERNAL_MEMORY_WIN32
class VmaWin32Handle {
public:
	VmaWin32Handle() noexcept
	 : m_hHandle(VMA_NULL)
	{}
	explicit VmaWin32Handle(HANDLE hHandle) noexcept
		: m_hHandle(hHandle)
		, m_IsNTHandle(IsNTHandle(hHandle))
	{}
	~VmaWin32Handle() noexcept { if (m_hHandle != VMA_NULL && m_IsNTHandle) { ::CloseHandle(m_hHandle); } }
	VMA_CLASS_NO_COPY_NO_MOVE(VmaWin32Handle)

public:
	// Strengthened
	VkResult GetHandle(VkDevice device, VkDeviceMemory memory, PFN_vkGetMemoryWin32HandleKHR pvkGetMemoryWin32HandleKHR, VkExternalMemoryHandleTypeFlagBits handleType, HANDLE hTargetProcess, bool useMutex, HANDLE* pHandle) noexcept	{
		*pHandle = VMA_NULL;
		// Try to get handle first.
		VkResult res = VK_SUCCESS;
		if (m_hHandle == VMA_NULL)	{
			VmaMutexLockWrite lock(m_Mutex, useMutex);
			if (m_hHandle == VMA_NULL)	{
				res = Create(device, memory, pvkGetMemoryWin32HandleKHR, handleType, &m_hHandle);
				if (res != VK_SUCCESS) {
					m_hHandle = VMA_NULL;
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

	operator bool() const noexcept { return m_hHandle != VMA_NULL; }
private:
	// Not atomic
	static VkResult Create(VkDevice device, VkDeviceMemory memory, PFN_vkGetMemoryWin32HandleKHR pvkGetMemoryWin32HandleKHR, VkExternalMemoryHandleTypeFlagBits handleType, HANDLE* pHandle) noexcept	{
		VkResult res = VK_ERROR_FEATURE_NOT_PRESENT;
		if (pvkGetMemoryWin32HandleKHR != VMA_NULL) {
			VkMemoryGetWin32HandleInfoKHR handleInfo{ };
			handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
			handleInfo.memory = memory;
			handleInfo.handleType = handleType;
			res = pvkGetMemoryWin32HandleKHR(device, &handleInfo, pHandle);
		}
		return res;
	}
	HANDLE Duplicate(HANDLE hTargetProcess = VMA_NULL) const noexcept {
		if (!m_hHandle) {
			return m_hHandle;
		}

		HANDLE hCurrentProcess = ::GetCurrentProcess();
		HANDLE hDupHandle = VMA_NULL;
		if (!::DuplicateHandle(hCurrentProcess, m_hHandle, hTargetProcess ? hTargetProcess : hCurrentProcess, &hDupHandle, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
			VMA_ASSERT(0 && "Failed to duplicate handle.");
		}
		return hDupHandle;
	}
	static bool IsNTHandle(HANDLE hHandle) noexcept	{
		DWORD flags = 0;
		return (hHandle != VMA_NULL) ? (::GetHandleInformation(hHandle, &flags) != 0) : false;
	}
private:
	HANDLE m_hHandle;
	VMA_RW_MUTEX m_Mutex; // Protects access m_Handle
	bool m_IsNTHandle = false; // True if m_Handle is NT handle, false if it's a KMT handle.
};
#else 
class VmaWin32Handle {
	// ABI compatibility
	void* placeholder = VMA_NULL;
	VMA_RW_MUTEX placeholder2;
	bool placeholder3 = false;
};
#endif // VMA_EXTERNAL_MEMORY_WIN32


#ifndef _VMA_DEVICE_MEMORY_BLOCK
/*
Represents a single block of device memory (`VkDeviceMemory`) with all the
data about its regions (aka suballocations, #VmaAllocation), assigned and free.

Thread-safety:
- Access to m_pMetadata must be externally synchronized.
- Map, Unmap, Bind* are synchronized internally.
*/
class VmaDeviceMemoryBlock {
	VMA_CLASS_NO_COPY_NO_MOVE(VmaDeviceMemoryBlock)
public:
	VmaBlockMetadata* m_pMetadata;

	explicit VmaDeviceMemoryBlock(VmaAllocator hAllocator);
	~VmaDeviceMemoryBlock();

	// Always call after construction.
	void Init(
		VmaAllocator hAllocator,
		VmaPool hParentPool,
		uint32_t newMemoryTypeIndex,
		VkDeviceMemory newMemory,
		VkDeviceSize newSize,
		uint32_t id,
		uint32_t algorithm,
		VkDeviceSize bufferImageGranularity
	);
	// Always call before destruction.
	void Destroy(VmaAllocator allocator);

	VmaPool GetParentPool() const { return m_hParentPool; }
	VkDeviceMemory GetDeviceMemory() const { return m_hMemory; }
	uint32_t GetMemoryTypeIndex() const { return m_MemoryTypeIndex; }
	uint32_t GetId() const { return m_Id; }
	void* GetMappedData() const { return m_pMappedData; }
	uint32_t GetMapRefCount() const { return m_MapCount; }

	// Call when allocation/free was made from m_pMetadata.
	// Used for m_MappingHysteresis.
	void PostAlloc(VmaAllocator hAllocator);
	void PostFree(VmaAllocator hAllocator);

	// Validates all data structures inside this object. If not valid, returns false.
	bool Validate() const;
	VkResult CheckCorruption(VmaAllocator hAllocator);

	// ppData can be null.
	VkResult Map(VmaAllocator hAllocator, uint32_t count, void** ppData);
	void Unmap(VmaAllocator hAllocator, uint32_t count);

	VkResult WriteMagicValueAfterAllocation(VmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize);
	VkResult ValidateMagicValueAfterAllocation(VmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize);

	VkResult BindBufferMemory(
		VmaAllocator hAllocator,
		VmaAllocation hAllocation,
		VkDeviceSize allocationLocalOffset,
		VkBuffer hBuffer,
		const void* pNext
	);
	VkResult BindImageMemory(
		VmaAllocator hAllocator,
		VmaAllocation hAllocation,
		VkDeviceSize allocationLocalOffset,
		VkImage hImage,
		const void* pNext
	);
#if VMA_EXTERNAL_MEMORY_WIN32
	VkResult CreateWin32Handle(
		const VmaAllocator hAllocator,
		PFN_vkGetMemoryWin32HandleKHR pvkGetMemoryWin32HandleKHR,
		VkExternalMemoryHandleTypeFlagBits handleType,
		HANDLE hTargetProcess,
		HANDLE* pHandle
	)noexcept;
#endif // VMA_EXTERNAL_MEMORY_WIN32
private:
	VmaPool m_hParentPool; // VK_NULL_HANDLE if not belongs to custom pool.
	uint32_t m_MemoryTypeIndex;
	uint32_t m_Id;
	VkDeviceMemory m_hMemory;

	/*
	Protects access to m_hMemory so it is not used by multiple threads simultaneously, e.g. vkMapMemory, vkBindBufferMemory.
	Also protects m_MapCount, m_pMappedData.
	Allocations, deallocations, any change in m_pMetadata is protected by parent's VmaBlockVector::m_Mutex.
	*/
	VMA_MUTEX m_MapAndBindMutex;
	VmaMappingHysteresis m_MappingHysteresis;
	uint32_t m_MapCount;
	void* m_pMappedData;

	VmaWin32Handle m_Handle;
};
#endif // _VMA_DEVICE_MEMORY_BLOCK

#ifndef _VMA_ALLOCATION_T
struct VmaAllocationExtraData {
	void* m_pMappedData = VMA_NULL; // Not null means memory is mapped.
	VmaWin32Handle m_Handle;
};

struct VmaAllocation_T {
	friend struct VmaDedicatedAllocationListItemTraits;

	enum FLAGS {
		FLAG_PERSISTENT_MAP   = 0x01,
		FLAG_MAPPING_ALLOWED  = 0x02,
	};

public:
	enum ALLOCATION_TYPE {
		ALLOCATION_TYPE_NONE,
		ALLOCATION_TYPE_BLOCK,
		ALLOCATION_TYPE_DEDICATED,
	};

	// This struct is allocated using VmaPoolAllocator.
	explicit VmaAllocation_T(bool mappingAllowed);
	~VmaAllocation_T();

	void InitBlockAllocation(
		VmaDeviceMemoryBlock* block,
		VmaAllocHandle allocHandle,
		VkDeviceSize alignment,
		VkDeviceSize size,
		uint32_t memoryTypeIndex,
		VmaSuballocationType suballocationType,
		bool mapped
	);
	// pMappedData not null means allocation is created with MAPPED flag.
	void InitDedicatedAllocation(
		VmaAllocator allocator,
		VmaPool hParentPool,
		uint32_t memoryTypeIndex,
		VkDeviceMemory hMemory,
		VmaSuballocationType suballocationType,
		void* pMappedData,
		VkDeviceSize size
	);
	void Destroy(VmaAllocator allocator);

	ALLOCATION_TYPE GetType() const { return (ALLOCATION_TYPE)m_Type; }
	VkDeviceSize GetAlignment() const { return m_Alignment; }
	VkDeviceSize GetSize() const { return m_Size; }
	void* GetUserData() const { return m_pUserData; }
	const char* GetName() const { return m_pName; }
	VmaSuballocationType GetSuballocationType() const { return (VmaSuballocationType)m_SuballocationType; }

	VmaDeviceMemoryBlock* GetBlock() const { VMA_ASSERT(m_Type == ALLOCATION_TYPE_BLOCK); return m_BlockAllocation.m_Block; }
	uint32_t GetMemoryTypeIndex() const { return m_MemoryTypeIndex; }
	bool IsPersistentMap() const { return (m_Flags & FLAG_PERSISTENT_MAP) != 0; }
	bool IsMappingAllowed() const { return (m_Flags & FLAG_MAPPING_ALLOWED) != 0; }

	void SetUserData(VmaAllocator hAllocator, void* pUserData) { m_pUserData = pUserData; }
	void SetName(VmaAllocator hAllocator, const char* pName);
	void FreeName(VmaAllocator hAllocator);
	uint8_t SwapBlockAllocation(VmaAllocator hAllocator, VmaAllocation allocation);
	VmaAllocHandle GetAllocHandle() const;
	VkDeviceSize GetOffset() const;
	VmaPool GetParentPool() const;
	VkDeviceMemory GetMemory() const;
	void* GetMappedData() const;

	void BlockAllocMap();
	void BlockAllocUnmap();
	VkResult DedicatedAllocMap(VmaAllocator hAllocator, void** ppData);
	void DedicatedAllocUnmap(VmaAllocator hAllocator);

#if VMA_STATS_STRING_ENABLED
	VmaBufferImageUsage GetBufferImageUsage() const { return m_BufferImageUsage; }
	void InitBufferUsage(const VkBufferCreateInfo &createInfo, bool useKhrMaintenance5)	{
		VMA_ASSERT(m_BufferImageUsage == VmaBufferImageUsage::UNKNOWN);
		m_BufferImageUsage = VmaBufferImageUsage(createInfo, useKhrMaintenance5);
	}
	void InitImageUsage(const VkImageCreateInfo &createInfo) {
		VMA_ASSERT(m_BufferImageUsage == VmaBufferImageUsage::UNKNOWN);
		m_BufferImageUsage = VmaBufferImageUsage(createInfo);
	}
	void PrintParameters(class VmaJsonWriter& json) const;
#endif

#if VMA_EXTERNAL_MEMORY_WIN32
	VkResult GetWin32Handle(VmaAllocator hAllocator, VkExternalMemoryHandleTypeFlagBits handleType, HANDLE hTargetProcess, HANDLE* hHandle) noexcept;
#endif // VMA_EXTERNAL_MEMORY_WIN32

private:
	// Allocation out of VmaDeviceMemoryBlock.
	struct BlockAllocation {
		VmaDeviceMemoryBlock* m_Block;
		VmaAllocHandle m_AllocHandle;
	};
	// Allocation for an object that has its own private VkDeviceMemory.
	struct DedicatedAllocation {
		VmaPool m_hParentPool; // VK_NULL_HANDLE if not belongs to custom pool.
		VkDeviceMemory m_hMemory;
		VmaAllocationExtraData* m_ExtraData;
		VmaAllocation_T* m_Prev;
		VmaAllocation_T* m_Next;
	};
	union {
		// Allocation out of VmaDeviceMemoryBlock.
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
	uint8_t m_SuballocationType; // VmaSuballocationType
	// Reference counter for vmaMapMemory()/vmaUnmapMemory().
	uint8_t m_MapCount;
	uint8_t m_Flags; // enum FLAGS
#if VMA_STATS_STRING_ENABLED
	VmaBufferImageUsage m_BufferImageUsage; // 0 if unknown.
#endif

	void EnsureExtraData(VmaAllocator hAllocator);
};
#endif // _VMA_ALLOCATION_T

#ifndef _VMA_DEDICATED_ALLOCATION_LIST_ITEM_TRAITS
struct VmaDedicatedAllocationListItemTraits {
	typedef VmaAllocation_T ItemType;

	static ItemType* GetPrev(const ItemType* item) {
		VMA_HEAVY_ASSERT(item->GetType() == VmaAllocation_T::ALLOCATION_TYPE_DEDICATED);
		return item->m_DedicatedAllocation.m_Prev;
	}
	static ItemType* GetNext(const ItemType* item) {
		VMA_HEAVY_ASSERT(item->GetType() == VmaAllocation_T::ALLOCATION_TYPE_DEDICATED);
		return item->m_DedicatedAllocation.m_Next;
	}
	static ItemType*& AccessPrev(ItemType* item) {
		VMA_HEAVY_ASSERT(item->GetType() == VmaAllocation_T::ALLOCATION_TYPE_DEDICATED);
		return item->m_DedicatedAllocation.m_Prev;
	}
	static ItemType*& AccessNext(ItemType* item) {
		VMA_HEAVY_ASSERT(item->GetType() == VmaAllocation_T::ALLOCATION_TYPE_DEDICATED);
		return item->m_DedicatedAllocation.m_Next;
	}
};
#endif // _VMA_DEDICATED_ALLOCATION_LIST_ITEM_TRAITS

#ifndef _VMA_DEDICATED_ALLOCATION_LIST
/*
Stores linked list of VmaAllocation_T objects.
Thread-safe, synchronized internally.
*/
class VmaDedicatedAllocationList {
	VMA_CLASS_NO_COPY_NO_MOVE(VmaDedicatedAllocationList)
public:
	VmaDedicatedAllocationList() = default;
	~VmaDedicatedAllocationList();

	void Init(bool useMutex) { m_UseMutex = useMutex; }
	bool Validate();

	void AddDetailedStatistics(VmaDetailedStatistics& inoutStats);
	void AddStatistics(VmaStatistics& inoutStats);
#if VMA_STATS_STRING_ENABLED
	// Writes JSON array with the list of allocations.
	void BuildStatsString(VmaJsonWriter& json);
#endif

	bool IsEmpty();
	void Register(VmaAllocation alloc);
	void Unregister(VmaAllocation alloc);

private:
	typedef VmaIntrusiveLinkedList<VmaDedicatedAllocationListItemTraits> DedicatedAllocationLinkedList;

	bool m_UseMutex = true;
	VMA_RW_MUTEX m_Mutex;
	DedicatedAllocationLinkedList m_AllocationList;
};

#ifndef _VMA_DEDICATED_ALLOCATION_LIST_FUNCTIONS

VmaDedicatedAllocationList::~VmaDedicatedAllocationList() {
	VMA_HEAVY_ASSERT(Validate());

	if (!m_AllocationList.IsEmpty()) {
		VMA_ASSERT_LEAK(false && "Unfreed dedicated allocations found!");
	}
}

bool VmaDedicatedAllocationList::Validate()
{
	const size_t declaredCount = m_AllocationList.GetCount();
	size_t actualCount = 0;
	VmaMutexLockRead lock(m_Mutex, m_UseMutex);
	for (VmaAllocation alloc = m_AllocationList.Front(); alloc != VMA_NULL; alloc = m_AllocationList.GetNext(alloc))	{
		++actualCount;
	}
	VMA_VALIDATE(actualCount == declaredCount);

	return true;
}

void VmaDedicatedAllocationList::AddDetailedStatistics(VmaDetailedStatistics& inoutStats) {
	for(auto* item = m_AllocationList.Front(); item != VMA_NULL; item = DedicatedAllocationLinkedList::GetNext(item)) {
		const VkDeviceSize size = item->GetSize();
		inoutStats.statistics.blockCount++;
		inoutStats.statistics.blockBytes += size;
		VmaAddDetailedStatisticsAllocation(inoutStats, item->GetSize());
	}
}

void VmaDedicatedAllocationList::AddStatistics(VmaStatistics& inoutStats) {
	VmaMutexLockRead lock(m_Mutex, m_UseMutex);

	const uint32_t allocCount = (uint32_t)m_AllocationList.GetCount();
	inoutStats.blockCount += allocCount;
	inoutStats.allocationCount += allocCount;

	for(auto* item = m_AllocationList.Front(); item != VMA_NULL; item = DedicatedAllocationLinkedList::GetNext(item)) {
		const VkDeviceSize size = item->GetSize();
		inoutStats.blockBytes += size;
		inoutStats.allocationBytes += size;
	}
}

#if VMA_STATS_STRING_ENABLED
void VmaDedicatedAllocationList::BuildStatsString(VmaJsonWriter& json) {
	VmaMutexLockRead lock(m_Mutex, m_UseMutex);
	json.BeginArray();
	for (VmaAllocation alloc = m_AllocationList.Front();
		alloc != VMA_NULL; alloc = m_AllocationList.GetNext(alloc))
	{
		json.BeginObject(true);
		alloc->PrintParameters(json);
		json.EndObject();
	}
	json.EndArray();
}
#endif // VMA_STATS_STRING_ENABLED

bool VmaDedicatedAllocationList::IsEmpty() {
	VmaMutexLockRead lock(m_Mutex, m_UseMutex);
	return m_AllocationList.IsEmpty();
}

void VmaDedicatedAllocationList::Register(VmaAllocation alloc) {
	VmaMutexLockWrite lock(m_Mutex, m_UseMutex);
	m_AllocationList.PushBack(alloc);
}

void VmaDedicatedAllocationList::Unregister(VmaAllocation alloc) {
	VmaMutexLockWrite lock(m_Mutex, m_UseMutex);
	m_AllocationList.Remove(alloc);
}
#endif // _VMA_DEDICATED_ALLOCATION_LIST_FUNCTIONS
#endif // _VMA_DEDICATED_ALLOCATION_LIST

#ifndef _VMA_SUBALLOCATION
/*
Represents a region of VmaDeviceMemoryBlock that is either assigned and returned as
allocated memory block or free.
*/
struct VmaSuballocation {
	VkDeviceSize offset;
	VkDeviceSize size;
	void* userData;
	VmaSuballocationType type;
};

// Comparator for offsets.
struct VmaSuballocationOffsetLess {
	bool operator()(const VmaSuballocation& lhs, const VmaSuballocation& rhs) const	{
		return lhs.offset < rhs.offset;
	}
};

struct VmaSuballocationOffsetGreater {
	bool operator()(const VmaSuballocation& lhs, const VmaSuballocation& rhs) const	{
		return lhs.offset > rhs.offset;
	}
};

struct VmaSuballocationItemSizeLess {
	bool operator()(const VmaSuballocationList::iterator lhs,	const VmaSuballocationList::iterator rhs) const	{
		return lhs->size < rhs->size;
	}

	bool operator()(const VmaSuballocationList::iterator lhs,	VkDeviceSize rhsSize) const	{
		return lhs->size < rhsSize;
	}
};
#endif // _VMA_SUBALLOCATION

#ifndef _VMA_ALLOCATION_REQUEST
/*
Parameters of planned allocation inside a VmaDeviceMemoryBlock.
item points to a FREE suballocation.
*/
struct VmaAllocationRequest {
	VmaAllocHandle allocHandle;
	VkDeviceSize size;
	VmaSuballocationList::iterator item;
	void* customData;
	uint64_t algorithmData;
	VmaAllocationRequestType type;
};
#endif // _VMA_ALLOCATION_REQUEST
