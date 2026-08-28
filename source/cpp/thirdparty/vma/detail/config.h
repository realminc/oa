// SPDX: MIT. Derived from Vulkan Memory Allocator, (c) 2017-2026 Advanced Micro Devices, Inc.
// Full MIT copyright/permission notice: source/cpp/thirdparty/vma/vma.h. See NOTICE.md.
#include <stdint.h>
#include <inttypes.h>

#include <oa/core/assert.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/allocator.h>
#include <oa/core/std/atomic.h>
#include <oa/core/std/cString.h>
#include <oa/core/std/lifetime.h>
#include <oa/core/std/memory.h>
#include <oa/core/std/sync.h>
#include <oa/core/std/utility.h>

#ifdef _MSC_VER
	#include <intrin.h> // For functions like __popcnt, _BitScanForward etc.
#endif
#if VMA_STATS_STRING_ENABLED
	#include <stdio.h> // For snprintf
#endif

/*******************************************************************************
CONFIGURATION SECTION

Define some of these macros before each #include of this header or change them
here if you need other then default behavior depending on your environment.
*/
#ifndef _VMA_CONFIGURATION

/*
Define this macro to 1 to make the library fetch pointers to Vulkan functions
internally, like:

	vulkanFunctions.vkAllocateMemory = &vkAllocateMemory;
*/
#if !defined(VMA_STATIC_VULKAN_FUNCTIONS) && !defined(VK_NO_PROTOTYPES)
	#define VMA_STATIC_VULKAN_FUNCTIONS 1
#endif

/*
Define this macro to 1 to make the library fetch pointers to Vulkan functions
internally, like:

	vulkanFunctions.vkAllocateMemory = (PFN_vkAllocateMemory)vkGetDeviceProcAddr(device, "vkAllocateMemory");

To use this feature in new versions of VMA you now have to pass
VmaVulkanFunctions::vkGetInstanceProcAddr and vkGetDeviceProcAddr as
VmaAllocatorCreateInfo::pVulkanFunctions. Other members can be null.
*/
#if !defined(VMA_DYNAMIC_VULKAN_FUNCTIONS)
	#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#endif

/*
Define this macro to include custom header files without having to edit this file directly, e.g.:

	// Inside of "my_Vma_configuration_user_includes.h":

	#include "my_custom_assert.h" // for MY_CUSTOM_ASSERT
	#include "my_custom_min.h" // for my_custom_min
	#include <oa/core/std/algo.h>
	#include <oa/core/std/sync.h>

	// Inside a different file, which includes Vma.h:

	#define VMA_CONFIGURATION_USER_INCLUDES_H "my_Vma_configuration_user_includes.h"
	#define VMA_ASSERT(expr) MY_CUSTOM_ASSERT(expr)
	#define VMA_MIN(v1, v2)  (my_custom_min(v1, v2))
	#include <oa/thirdparty/vma/vma.h>
	...

The following headers are used in this CONFIGURATION section only, so feel free to
remove them if not needed.
*/
#if defined(VMA_CONFIGURATION_USER_INCLUDES_H)
	#include VMA_CONFIGURATION_USER_INCLUDES_H
#endif

#ifndef VMA_NULL
	 // Value used as null pointer. Define it to e.g.: nullptr, NULL, 0, (void*)0.
	 #define VMA_NULL   nullptr
#endif

#ifndef VMA_FALLTHROUGH
	#if __cplusplus >= 201703L || _MSVC_LANG >= 201703L // C++17
		#define VMA_FALLTHROUGH [[fallthrough]]
	#else
		#define VMA_FALLTHROUGH
	#endif
#endif

// Normal assert to check for programmer's errors, especially in Debug configuration.
#ifndef VMA_ASSERT
	#define VMA_ASSERT(expr) OA_ASSERT(expr)
#endif

// Assert that will be called very often, like inside data structures e.g. operator[].
// Making it non-empty can make program slow.
#ifndef VMA_HEAVY_ASSERT
	 #ifdef NDEBUG
		 #define VMA_HEAVY_ASSERT(expr)
	 #else
		 #define VMA_HEAVY_ASSERT(expr)   //VMA_ASSERT(expr)
	 #endif
#endif

// Assert used for reporting memory leaks - unfreed allocations.
#ifndef VMA_ASSERT_LEAK
	#define VMA_ASSERT_LEAK(expr)   VMA_ASSERT(expr)
#endif

#ifndef VMA_ALIGN_OF
	#define VMA_ALIGN_OF(type) (alignof(type))
#endif

#ifndef VMA_SYSTEM_ALIGNED_MALLOC
	#define VMA_SYSTEM_ALIGNED_MALLOC(size, alignment) \
		::oa::allocBytes((size), (alignment))
#endif

#ifndef VMA_SYSTEM_ALIGNED_FREE
	// VMA_SYSTEM_FREE is the old name, but might have been defined by the user.
	#if defined(VMA_SYSTEM_FREE)
		#define VMA_SYSTEM_ALIGNED_FREE(ptr) VMA_SYSTEM_FREE(ptr)
	#else
		#define VMA_SYSTEM_ALIGNED_FREE(ptr) ::oa::freeBytes(ptr)
	#endif
#endif

#ifndef VMA_COUNT_BITS_SET
	// Returns number of bits set to 1 in (v)
	#define VMA_COUNT_BITS_SET(v) VmaCountBitsSet(v)
#endif

#ifndef VMA_BITSCAN_LSB
	// Scans integer for index of first nonzero value from the Least Significant Bit (LSB). If mask is 0 then returns UINT8_MAX
	#define VMA_BITSCAN_LSB(mask) VmaBitScanLSB(mask)
#endif

#ifndef VMA_BITSCAN_MSB
	// Scans integer for index of first nonzero value from the Most Significant Bit (MSB). If mask is 0 then returns UINT8_MAX
	#define VMA_BITSCAN_MSB(mask) VmaBitScanMSB(mask)
#endif

#ifndef VMA_MIN
	#define VMA_MIN(v1, v2) (::oa::min((v1), (v2)))
#endif

#ifndef VMA_MAX
	#define VMA_MAX(v1, v2) (::oa::max((v1), (v2)))
#endif

#ifndef VMA_SORT
	#define VMA_SORT(beg, end, cmp) ::oa::sort((beg), (end), (cmp))
#endif

#ifndef VMA_DEBUG_LOG_FORMAT
	 #define VMA_DEBUG_LOG_FORMAT(format, ...)
	 /*
	 #define VMA_DEBUG_LOG_FORMAT(format, ...) do { \
		 printf((format), __VA_ARGS__); \
		 printf("\n"); \
	 } while(false)
	 */
#endif

#ifndef VMA_DEBUG_LOG
	#define VMA_DEBUG_LOG(str)   VMA_DEBUG_LOG_FORMAT("%s", (str))
#endif

#ifndef VMA_LEAK_LOG_FORMAT
	#define VMA_LEAK_LOG_FORMAT(format, ...)   VMA_DEBUG_LOG_FORMAT(format, __VA_ARGS__)
#endif

#ifndef VMA_CLASS_NO_COPY
	#define VMA_CLASS_NO_COPY(className) \
		private: \
			className(const className&) = delete; \
			className& operator=(const className&) = delete;
#endif
#ifndef VMA_CLASS_NO_COPY_NO_MOVE
	#define VMA_CLASS_NO_COPY_NO_MOVE(className) \
		private: \
			className(const className&) = delete; \
			className(className&&) = delete; \
			className& operator=(const className&) = delete; \
			className& operator=(className&&) = delete;
#endif

// Define this macro to 1 to enable functions: vmaBuildStatsString, vmaFreeStatsString.
#if VMA_STATS_STRING_ENABLED
namespace {
	inline void VmaUint32ToStr(char* VMA_NOT_NULL outStr, size_t strLen, uint32_t num)
	{
		snprintf(outStr, strLen, "%" PRIu32, num);
	}
	inline void VmaUint64ToStr(char* VMA_NOT_NULL outStr, size_t strLen, uint64_t num)
	{
		snprintf(outStr, strLen, "%" PRIu64, num);
	}
	inline void VmaPtrToStr(char* VMA_NOT_NULL outStr, size_t strLen, const void* ptr)
	{
		snprintf(outStr, strLen, "%p", ptr);
	}
} // namespace
#endif

#ifndef VMA_MUTEX
	class VmaMutex
	{
	VMA_CLASS_NO_COPY_NO_MOVE(VmaMutex)
	public:
		VmaMutex() = default;
		void Lock() { m_Mutex.lock(); }
		void Unlock() { m_Mutex.unlock(); }
		bool TryLock() { return m_Mutex.tryLock(); }
	private:
		oa::Mutex m_Mutex;
	};
	#define VMA_MUTEX VmaMutex
#endif

// Read-write mutex, where "read" is shared access, "write" is exclusive access.
#ifndef VMA_RW_MUTEX
	class VmaRWMutex
	{
	public:
		void LockRead() { m_Mutex.lockShared(); }
		void UnlockRead() { m_Mutex.unlockShared(); }
		bool TryLockRead() { return m_Mutex.tryLockShared(); }
		void LockWrite() { m_Mutex.lock(); }
		void UnlockWrite() { m_Mutex.unlock(); }
		bool TryLockWrite() { return m_Mutex.tryLock(); }
	private:
		oa::SharedMutex m_Mutex;
	};
	#define VMA_RW_MUTEX VmaRWMutex
#endif // #ifndef VMA_RW_MUTEX

#ifndef VMA_ATOMIC_UINT32
	#define VMA_ATOMIC_UINT32 oa::Atomic<uint32_t>
#endif

#ifndef VMA_ATOMIC_UINT64
	#define VMA_ATOMIC_UINT64 oa::Atomic<uint64_t>
#endif

#ifndef VMA_DEBUG_ALWAYS_DEDICATED_MEMORY
	/**
	Every allocation will have its own memory block.
	Define to 1 for debugging purposes only.
	*/
	#define VMA_DEBUG_ALWAYS_DEDICATED_MEMORY (0)
#endif

#ifndef VMA_MIN_ALIGNMENT
	/**
	Minimum alignment of all allocations, in bytes.
	Set to more than 1 for debugging purposes. Must be power of two.
	*/
	#ifdef VMA_DEBUG_ALIGNMENT // Old name
		#define VMA_MIN_ALIGNMENT VMA_DEBUG_ALIGNMENT
	#else
		#define VMA_MIN_ALIGNMENT (1)
	#endif
#endif

#ifndef VMA_DEBUG_MARGIN
	/**
	Minimum margin after every allocation, in bytes.
	Set nonzero for debugging purposes only.
	*/
	#define VMA_DEBUG_MARGIN (0)
#endif

#ifndef VMA_DEBUG_INITIALIZE_ALLOCATIONS
	/**
	Define this macro to 1 to automatically fill new allocations and destroyed
	allocations with some bit pattern.
	*/
	#define VMA_DEBUG_INITIALIZE_ALLOCATIONS (0)
#endif

#ifndef VMA_DEBUG_DETECT_CORRUPTION
	/**
	Define this macro to 1 together with non-zero value of VMA_DEBUG_MARGIN to
	enable writing magic value to the margin after every allocation and
	validating it, so that memory corruptions (out-of-bounds writes) are detected.
	*/
	#define VMA_DEBUG_DETECT_CORRUPTION (0)
#endif

#ifndef VMA_DEBUG_GLOBAL_MUTEX
	/**
	Set this to 1 for debugging purposes only, to enable single mutex protecting all
	entry calls to the library. Can be useful for debugging multithreading issues.
	*/
	#define VMA_DEBUG_GLOBAL_MUTEX (0)
#endif

#ifndef VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY
	/**
	Minimum value for VkPhysicalDeviceLimits::bufferImageGranularity.
	Set to more than 1 for debugging purposes only. Must be power of two.
	*/
	#define VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY (1)
#endif

#ifndef VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT
	/*
	Set this to 1 to make VMA never exceed VkPhysicalDeviceLimits::maxMemoryAllocationCount
	and return error instead of leaving up to Vulkan implementation what to do in such cases.
	*/
	#define VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT (1)
#endif

#ifndef VMA_DEBUG_DONT_EXCEED_HEAP_SIZE_WITH_ALLOCATION_SIZE
	/*
	Set this to 1 to make VMA never exceed VkPhysicalDeviceMemoryProperties::memoryHeaps[i].size
	with a single allocation size VkMemoryAllocateInfo::allocationSize
	and return error instead of leaving up to Vulkan implementation what to do in such cases.
	It protects agaist validation error VUID-vkAllocateMemory-pAllocateInfo-01713.
	On the other hand, allowing exceeding this size may result in a successful allocation despite the validation error.
	*/
	#define VMA_DEBUG_DONT_EXCEED_HEAP_SIZE_WITH_ALLOCATION_SIZE (1)
#endif

#ifndef VMA_SMALL_HEAP_MAX_SIZE
	 /// Maximum size of a memory heap in Vulkan to consider it "small".
	 #define VMA_SMALL_HEAP_MAX_SIZE (1024ULL * 1024 * 1024)
#endif

#ifndef VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE
	 /// Default size of a block allocated as single VkDeviceMemory from a "large" heap.
	 #define VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE (256ULL * 1024 * 1024)
#endif

/*
Mapping hysteresis is a logic that launches when vmaMapMemory/vmaUnmapMemory is called
or a persistently mapped allocation is created and destroyed several times in a row.
It keeps additional +1 mapping of a device memory block to prevent calling actual
vkMapMemory/vkUnmapMemory too many times, which may improve performance and help
tools like RenderDoc.
*/
#ifndef VMA_MAPPING_HYSTERESIS_ENABLED
	#define VMA_MAPPING_HYSTERESIS_ENABLED 1
#endif

#define VMA_VALIDATE(cond) do { if(!(cond)) { \
		VMA_ASSERT(0 && "Validation failed: " #cond); \
		return false; \
	} } while(false)

/*******************************************************************************
END OF CONFIGURATION
*/
#endif // _VMA_CONFIGURATION

namespace
{
constexpr uint8_t VMA_ALLOCATION_FILL_PATTERN_CREATED = 0xDC;
constexpr uint8_t VMA_ALLOCATION_FILL_PATTERN_DESTROYED = 0xEF;
// Decimal 2139416166, float NaN, little-endian binary 66 E6 84 7F.
constexpr uint32_t VMA_CORRUPTION_DETECTION_MAGIC_VALUE = 0x7F84E666;

// Copy of some Vulkan definitions so we don't need to check their existence just to handle few constants.
constexpr uint32_t VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY = 0x00000040;
constexpr uint32_t VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY = 0x00000080;
constexpr uint32_t VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_COPY = 0x00020000;
constexpr uint32_t VK_IMAGE_CREATE_DISJOINT_BIT_COPY = 0x00000200;
constexpr int32_t VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT_COPY = 1000158000;
constexpr uint32_t VMA_ALLOCATION_INTERNAL_STRATEGY_MIN_OFFSET = 0x10000000U;
constexpr uint32_t VMA_ALLOCATION_TRY_COUNT = 32;
constexpr uint32_t VMA_VENDOR_ID_AMD = 4098;

// This one is tricky. Vulkan specification defines this code as available since
// Vulkan 1.0, but doesn't actually define it in Vulkan SDK earlier than 1.2.131.
// See pull request #207.
#define VK_ERROR_UNKNOWN_COPY ((VkResult)-13)
} // namespace


#if VMA_STATS_STRING_ENABLED
// Correspond to values of enum VmaSuballocationType.
const char* const VMA_SUBALLOCATION_TYPE_NAMES[] =
{
	"FREE",
	"UNKNOWN",
	"BUFFER",
	"IMAGE_UNKNOWN",
	"IMAGE_LINEAR",
	"IMAGE_OPTIMAL",
};
#endif

const VkAllocationCallbacks VmaEmptyAllocationCallbacks =
	{ VMA_NULL, VMA_NULL, VMA_NULL, VMA_NULL, VMA_NULL, VMA_NULL };


#ifndef _VMA_ENUM_DECLARATIONS

enum VmaSuballocationType
{
	VMA_SUBALLOCATION_TYPE_FREE = 0,
	VMA_SUBALLOCATION_TYPE_UNKNOWN = 1,
	VMA_SUBALLOCATION_TYPE_BUFFER = 2,
	VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN = 3,
	VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR = 4,
	VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL = 5,
	VMA_SUBALLOCATION_TYPE_MAX_ENUM = 0x7FFFFFFF
};

enum VMA_CACHE_OPERATION
{
	VMA_CACHE_FLUSH,
	VMA_CACHE_INVALIDATE
};

enum class VmaAllocationRequestType
{
	Normal,
	TLSF,
	// Used by "Linear" algorithm.
	UpperAddress,
	EndOf1st,
	EndOf2nd,
};

#endif // _VMA_ENUM_DECLARATIONS

#ifndef _VMA_FORWARD_DECLARATIONS
// Opaque handle used by allocation algorithms to identify single allocation in any conforming way.
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VmaAllocHandle);

struct VmaBufferImageUsage;

struct VmaMutexLock;
struct VmaMutexLockRead;
struct VmaMutexLockWrite;

template<typename T>
struct AtomicTransactionalIncrement;

template<typename T>
struct VmaStlAllocator;

template<typename T, typename AllocatorT>
class VmaVector;

template<typename T, typename AllocatorT, size_t N>
class VmaSmallVector;

template<typename T>
class VmaPoolAllocator;

template<typename T>
struct VmaListItem;

template<typename T>
class VmaRawList;

template<typename T, typename AllocatorT>
class VmaList;

template<typename ItemTypeTraits>
class VmaIntrusiveLinkedList;

#if VMA_STATS_STRING_ENABLED
class VmaStringBuilder;
class VmaJsonWriter;
#endif

class VmaDeviceMemoryBlock;

struct VmaDedicatedAllocationListItemTraits;
class VmaDedicatedAllocationList;

struct VmaSuballocation;
struct VmaSuballocationOffsetLess;
struct VmaSuballocationOffsetGreater;
struct VmaSuballocationItemSizeLess;

typedef VmaList<VmaSuballocation, VmaStlAllocator<VmaSuballocation>> VmaSuballocationList;

struct VmaAllocationRequest;

class VmaBlockMetadata;
class VmaBlockMetadata_Linear;
class VmaBlockMetadata_TLSF;

class VmaBlockVector;

struct VmaPoolListItemTraits;

struct VmaCurrentBudgetData;

class VmaAllocationObjectAllocator;

#endif // _VMA_FORWARD_DECLARATIONS

#ifndef _VMA_BUFFER_IMAGE_USAGE

// Finds structure with s->sType == sType in mainStruct->pNext chain.
// Returns pointer to it. If not found, returns null.
template<typename FindT, typename MainT>
inline const FindT* VmaPnextChainFind(const MainT* mainStruct, VkStructureType sType)
{
	for(const VkBaseInStructure* s = (const VkBaseInStructure*)mainStruct->pNext;
		s != VMA_NULL; s = s->pNext)
	{
		if(s->sType == sType)
		{
			return (const FindT*)s;
		}
	}
	return VMA_NULL;
}

// An abstraction over buffer or image `usage` flags, depending on available extensions.
struct VmaBufferImageUsage
{
#if VMA_KHR_MAINTENANCE5
	typedef uint64_t BaseType; // VkFlags64
#else
	typedef uint32_t BaseType; // VkFlags32
#endif

	static const VmaBufferImageUsage UNKNOWN;

	BaseType Value;

	VmaBufferImageUsage() { *this = UNKNOWN; }
	explicit VmaBufferImageUsage(BaseType usage) : Value(usage) { }
	VmaBufferImageUsage(const VkBufferCreateInfo &createInfo, bool useKhrMaintenance5);
	explicit VmaBufferImageUsage(const VkImageCreateInfo &createInfo);

	bool operator==(const VmaBufferImageUsage& rhs) const { return Value == rhs.Value; }
	bool operator!=(const VmaBufferImageUsage& rhs) const { return Value != rhs.Value; }

	bool Contains(BaseType flag) const { return (Value & flag) != 0; }
	bool ContainsDeviceAccess() const
	{
		// This relies on values of VK_IMAGE_USAGE_TRANSFER* being the same as VK_BUFFER_IMAGE_TRANSFER*.
		return (Value & ~BaseType(VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) != 0;
	}
};

const VmaBufferImageUsage VmaBufferImageUsage::UNKNOWN = VmaBufferImageUsage(0);

VmaBufferImageUsage::VmaBufferImageUsage(const VkBufferCreateInfo &createInfo,
	bool useKhrMaintenance5)
{
#if VMA_KHR_MAINTENANCE5
	if(useKhrMaintenance5)
	{
		// If VkBufferCreateInfo::pNext chain contains VkBufferUsageFlags2CreateInfoKHR,
		// take usage from it and ignore VkBufferCreateInfo::usage, per specification
		// of the VK_KHR_maintenance5 extension.
		const VkBufferUsageFlags2CreateInfoKHR* const usageFlags2 =
			VmaPnextChainFind<VkBufferUsageFlags2CreateInfoKHR>(&createInfo, VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR);
		if(usageFlags2 != VMA_NULL)
		{
			this->Value = usageFlags2->usage;
			return;
		}
	}
#endif

	this->Value = (BaseType)createInfo.usage;
}

VmaBufferImageUsage::VmaBufferImageUsage(const VkImageCreateInfo &createInfo)
	: Value((BaseType)createInfo.usage)
{
	// Maybe in the future there will be VK_KHR_maintenanceN extension with structure
	// VkImageUsageFlags2CreateInfoKHR, like the one for buffers...
}

#endif // _VMA_BUFFER_IMAGE_USAGE

#ifndef _VMA_FUNCTIONS

namespace
{

/*
Returns number of bits set to 1 in (v).

On specific platforms and compilers you can use intrinsics like:

Visual Studio:
	return __popcnt(v);
GCC, Clang:
	return static_cast<uint32_t>(__builtin_popcount(v));

Define macro VMA_COUNT_BITS_SET to provide your optimized implementation.
But you need to check in runtime whether user's CPU supports these, as some old processors don't.
*/
inline uint32_t VmaCountBitsSet(uint32_t v)
{
#if defined(_MSC_VER)
	return static_cast<uint32_t>(__popcnt(v));
#elif defined(__GNUC__) || defined(__clang__)
	return static_cast<uint32_t>(__builtin_popcount(v));
#else
	uint32_t c = v - ((v >> 1) & 0x55555555);
	c = ((c >> 2) & 0x33333333) + (c & 0x33333333);
	c = ((c >> 4) + c) & 0x0F0F0F0F;
	c = ((c >> 8) + c) & 0x00FF00FF;
	c = ((c >> 16) + c) & 0x0000FFFF;
	return c;
#endif
}

inline uint8_t VmaBitScanLSB(uint64_t mask)
{
#if defined(_MSC_VER) && defined(_WIN64)
	unsigned long pos;
	if (_BitScanForward64(&pos, mask))
		return static_cast<uint8_t>(pos);
	return UINT8_MAX;
#elif defined __GNUC__ || defined __clang__
	return mask != 0
		? static_cast<uint8_t>(__builtin_ctzll(
			static_cast<unsigned long long>(mask)))
		: UINT8_MAX;
#else
	uint8_t pos = 0;
	uint64_t bit = 1;
	do
	{
		if (mask & bit)
			return pos;
		bit <<= 1;
	} while (pos++ < 63);
	return UINT8_MAX;
#endif
}

inline uint8_t VmaBitScanLSB(uint32_t mask)
{
#ifdef _MSC_VER
	unsigned long pos;
	if (_BitScanForward(&pos, mask))
		return static_cast<uint8_t>(pos);
	return UINT8_MAX;
#elif defined __GNUC__ || defined __clang__
	return mask != 0
		? static_cast<uint8_t>(__builtin_ctz(mask))
		: UINT8_MAX;
#else
	uint8_t pos = 0;
	uint32_t bit = 1;
	do
	{
		if (mask & bit)
			return pos;
		bit <<= 1;
	} while (pos++ < 31);
	return UINT8_MAX;
#endif
}

inline uint8_t VmaBitScanMSB(uint64_t mask)
{
#if defined(_MSC_VER) && defined(_WIN64)
	unsigned long pos;
	if (_BitScanReverse64(&pos, mask))
		return static_cast<uint8_t>(pos);
#elif defined __GNUC__ || defined __clang__
	if (mask != 0)
		return 63 - static_cast<uint8_t>(__builtin_clzll(mask));
#else
	uint8_t pos = 63;
	uint64_t bit = 1ULL << 63;
	do
	{
		if (mask & bit)
			return pos;
		bit >>= 1;
	} while (pos-- > 0);
#endif
	return UINT8_MAX;
}

inline uint8_t VmaBitScanMSB(uint32_t mask)
{
#ifdef _MSC_VER
	unsigned long pos;
	if (_BitScanReverse(&pos, mask))
		return static_cast<uint8_t>(pos);
#elif defined __GNUC__ || defined __clang__
	if (mask != 0)
		return 31 - static_cast<uint8_t>(__builtin_clz(mask));
#else
	uint8_t pos = 31;
	uint32_t bit = 1UL << 31;
	do
	{
		if (mask & bit)
			return pos;
		bit >>= 1;
	} while (pos-- > 0);
#endif
	return UINT8_MAX;
}

/*
Returns true if given number is a power of two.
T must be unsigned integer number or signed integer but always nonnegative.
For 0 returns true.
*/
template <typename T>
inline bool VmaIsPow2(T x)
{
	return (x & (x - 1)) == 0;
}

// Aligns given value up to nearest multiply of align value. For example: VmaAlignUp(11, 8) = 16.
// Use types like uint32_t, uint64_t as T.
template <typename T>
inline T VmaAlignUp(T val, T alignment)
{
	VMA_HEAVY_ASSERT(VmaIsPow2(alignment));
	return (val + alignment - 1) & ~(alignment - 1);
}

// Aligns given value down to nearest multiply of align value. For example: VmaAlignDown(11, 8) = 8.
// Use types like uint32_t, uint64_t as T.
template <typename T>
inline T VmaAlignDown(T val, T alignment)
{
	VMA_HEAVY_ASSERT(VmaIsPow2(alignment));
	return val & ~(alignment - 1);
}

// Division with mathematical rounding to nearest number.
template <typename T>
inline T VmaRoundDiv(T x, T y)
{
	return (x + (y / (T)2)) / y;
}

// Divide by 'y' and round up to nearest integer.
template <typename T>
inline T VmaDivideRoundingUp(T x, T y)
{
	return (x + y - (T)1) / y;
}

// Returns smallest power of 2 greater or equal to v.
inline uint32_t VmaNextPow2(uint32_t v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

inline uint64_t VmaNextPow2(uint64_t v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v |= v >> 32;
	v++;
	return v;
}

// Returns largest power of 2 less or equal to v.
inline uint32_t VmaPrevPow2(uint32_t v)
{
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v = v ^ (v >> 1);
	return v;
}

inline uint64_t VmaPrevPow2(uint64_t v)
{
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v |= v >> 32;
	v = v ^ (v >> 1);
	return v;
}

inline bool VmaStrIsEmpty(const char* pStr)
{
	return pStr == VMA_NULL || *pStr == '\0';
}

/*
Returns true if two memory blocks occupy overlapping pages.
ResourceA must be in less memory offset than ResourceB.

Algorithm is based on "Vulkan 1.0.39 - A Specification (with all registered Vulkan extensions)"
chapter 11.6 "Resource Memory Association", paragraph "Buffer-Image Granularity".
*/
inline bool VmaBlocksOnSamePage(
	VkDeviceSize resourceAOffset,
	VkDeviceSize resourceASize,
	VkDeviceSize resourceBOffset,
	VkDeviceSize pageSize)
{
	VMA_ASSERT(resourceAOffset + resourceASize <= resourceBOffset && resourceASize > 0 && pageSize > 0);
	VkDeviceSize resourceAEnd = resourceAOffset + resourceASize - 1;
	VkDeviceSize resourceAEndPage = resourceAEnd & ~(pageSize - 1);
	VkDeviceSize resourceBStart = resourceBOffset;
	VkDeviceSize resourceBStartPage = resourceBStart & ~(pageSize - 1);
	return resourceAEndPage == resourceBStartPage;
}

/*
Returns true if given suballocation types could conflict and must respect
VkPhysicalDeviceLimits::bufferImageGranularity. They conflict if one is buffer
or linear image and another one is optimal image. If type is unknown, behave
conservatively.
*/
inline bool VmaIsBufferImageGranularityConflict(
	VmaSuballocationType suballocType1,
	VmaSuballocationType suballocType2)
{
	if (suballocType1 > suballocType2)
	{
		oa::swapValues(suballocType1, suballocType2);
	}

	switch (suballocType1)
	{
	case VMA_SUBALLOCATION_TYPE_FREE:
		return false;
	case VMA_SUBALLOCATION_TYPE_UNKNOWN:
		return true;
	case VMA_SUBALLOCATION_TYPE_BUFFER:
		return
			suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN ||
			suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL;
	case VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN:
		return
			suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN ||
			suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR ||
			suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL;
	case VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR:
		return
			suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL;
	case VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL:
		return false;
	default:
		VMA_ASSERT(0);
		return true;
	}
}

void VmaWriteMagicValue(void* pData, VkDeviceSize offset)
{
#if VMA_DEBUG_MARGIN > 0 && VMA_DEBUG_DETECT_CORRUPTION
	uint32_t* pDst = (uint32_t*)((char*)pData + offset);
	const size_t numberCount = VMA_DEBUG_MARGIN / sizeof(uint32_t);
	for (size_t i = 0; i < numberCount; ++i, ++pDst)
	{
		*pDst = VMA_CORRUPTION_DETECTION_MAGIC_VALUE;
	}
#else
	// no-op
#endif
}

bool VmaValidateMagicValue(const void* pData, VkDeviceSize offset)
{
#if VMA_DEBUG_MARGIN > 0 && VMA_DEBUG_DETECT_CORRUPTION
	const uint32_t* pSrc = (const uint32_t*)((const char*)pData + offset);
	const size_t numberCount = VMA_DEBUG_MARGIN / sizeof(uint32_t);
	for (size_t i = 0; i < numberCount; ++i, ++pSrc)
	{
		if (*pSrc != VMA_CORRUPTION_DETECTION_MAGIC_VALUE)
		{
			return false;
		}
	}
#endif
	return true;
}

/*
Fills structure with parameters of an example buffer to be used for transfers
during GPU memory defragmentation.
*/
void VmaFillGpuDefragmentationBufferCreateInfo(VkBufferCreateInfo& outBufCreateInfo)
{
	outBufCreateInfo = {};
	outBufCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	outBufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	outBufCreateInfo.size = (VkDeviceSize)VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE; // Example size.
}


/*
Performs binary search and returns iterator to first element that is greater or
equal to (key), according to comparison (cmp).

Cmp should return true if first argument is less than second argument.

Returned value is the found element, if present in the collection or place where
new element with value (key) should be inserted.
*/
template <typename CmpLess, typename IterT, typename KeyT>
IterT VmaBinaryFindFirstNotLess(IterT beg, IterT end, const KeyT& key, const CmpLess& cmp)
{
	size_t down = 0;
	size_t up = size_t(end - beg);
	while (down < up)
	{
		const size_t mid = down + (up - down) / 2;  // Overflow-safe midpoint calculation
		if (cmp(*(beg + mid), key))
		{
			down = mid + 1;
		}
		else
		{
			up = mid;
		}
	}
	return beg + down;
}

template<typename CmpLess, typename IterT, typename KeyT>
IterT VmaBinaryFindSorted(const IterT& beg, const IterT& end, const KeyT& value, const CmpLess& cmp)
{
	IterT it = VmaBinaryFindFirstNotLess<CmpLess, IterT, KeyT>(
		beg, end, value, cmp);
	if (it == end ||
		(!cmp(*it, value) && !cmp(value, *it)))
	{
		return it;
	}
	return end;
}

/*
Returns true if all pointers in the array are not-null and unique.
Warning! O(n^2) complexity. Use only inside VMA_HEAVY_ASSERT.
T must be pointer type, e.g. VmaAllocation, VmaPool.
*/
template<typename T>
bool VmaValidatePointerArray(uint32_t count, const T* arr)
{
	for (uint32_t i = 0; i < count; ++i)
	{
		const T iPtr = arr[i];
		if (iPtr == VMA_NULL)
		{
			return false;
		}
		for (uint32_t j = i + 1; j < count; ++j)
		{
			if (iPtr == arr[j])
			{
				return false;
			}
		}
	}
	return true;
}

template<typename MainT, typename NewT>
inline void VmaPnextChainPushFront(MainT* mainStruct, NewT* newStruct)
{
	newStruct->pNext = mainStruct->pNext;
	mainStruct->pNext = newStruct;
}

// This is the main algorithm that guides the selection of a memory type best for an allocation -
// converts usage to required/preferred/not preferred flags.
bool FindMemoryPreferences(
	bool isIntegratedGPU,
	const VmaAllocationCreateInfo& allocCreateInfo,
	VmaBufferImageUsage bufImgUsage,
	VkMemoryPropertyFlags& outRequiredFlags,
	VkMemoryPropertyFlags& outPreferredFlags,
	VkMemoryPropertyFlags& outNotPreferredFlags)
{
	outRequiredFlags = allocCreateInfo.requiredFlags;
	outPreferredFlags = allocCreateInfo.preferredFlags;
	outNotPreferredFlags = 0;

	switch(allocCreateInfo.usage)
	{
	case VMA_MEMORY_USAGE_UNKNOWN:
		break;
	case VMA_MEMORY_USAGE_GPU_ONLY:
		if(!isIntegratedGPU || (outPreferredFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
		{
			outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		}
		break;
	case VMA_MEMORY_USAGE_CPU_ONLY:
		outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		break;
	case VMA_MEMORY_USAGE_CPU_TO_GPU:
		outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
		if(!isIntegratedGPU || (outPreferredFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
		{
			outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		}
		break;
	case VMA_MEMORY_USAGE_GPU_TO_CPU:
		outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
		outPreferredFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
		break;
	case VMA_MEMORY_USAGE_CPU_COPY:
		outNotPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		break;
	case VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED:
		outRequiredFlags |= VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
		break;
	case VMA_MEMORY_USAGE_AUTO:
	case VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE:
	case VMA_MEMORY_USAGE_AUTO_PREFER_HOST:
	{
		if(bufImgUsage == VmaBufferImageUsage::UNKNOWN)
		{
			VMA_ASSERT(0 && "VMA_MEMORY_USAGE_AUTO* values can only be used with functions like vmaCreateBuffer, vmaCreateImage so that the details of the created resource are known."
				" Maybe you use VkBufferUsageFlags2CreateInfoKHR but forgot to use VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT?" );
			return false;
		}

		const bool deviceAccess = bufImgUsage.ContainsDeviceAccess();
		const bool hostAccessSequentialWrite = (allocCreateInfo.flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT) != 0;
		const bool hostAccessRandom = (allocCreateInfo.flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT) != 0;
		const bool hostAccessAllowTransferInstead = (allocCreateInfo.flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT) != 0;
		const bool preferDevice = allocCreateInfo.usage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		const bool preferHost = allocCreateInfo.usage == VMA_MEMORY_USAGE_AUTO_PREFER_HOST;

		// CPU random access - e.g. a buffer written to or transferred from GPU to read back on CPU.
		if(hostAccessRandom)
		{
			// Prefer cached. Cannot require it, because some platforms don't have it (e.g. Raspberry Pi - see #362)!
			outPreferredFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

			if (!isIntegratedGPU && deviceAccess && hostAccessAllowTransferInstead && !preferHost)
			{
				// Nice if it will end up in HOST_VISIBLE, but more importantly prefer DEVICE_LOCAL.
				// Omitting HOST_VISIBLE here is intentional.
				// In case there is DEVICE_LOCAL | HOST_VISIBLE | HOST_CACHED, it will pick that one.
				// Otherwise, this will give same weight to DEVICE_LOCAL as HOST_VISIBLE | HOST_CACHED and select the former if occurs first on the list.
				outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
			}
			else
			{
				if(hostAccessAllowTransferInstead)
					outPreferredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
				else
					// Always CPU memory.
					outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
			}
		}
		// CPU sequential write - may be CPU or host-visible GPU memory, uncached and write-combined.
		else if(hostAccessSequentialWrite)
		{
			// Want uncached and write-combined.
			outNotPreferredFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

			if(!isIntegratedGPU && deviceAccess && hostAccessAllowTransferInstead && !preferHost)
			{
				outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
			}
			else
			{
				if(hostAccessAllowTransferInstead)
					outPreferredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
				else
					// Always CPU memory.
					outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

				// Direct GPU access, CPU sequential write (e.g. a dynamic uniform buffer updated every frame)
				if(deviceAccess)
				{
					// Could go to CPU memory or GPU BAR/unified. Up to the user to decide. If no preference, choose GPU memory.
					if(preferHost)
						outNotPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
					else
						outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
				}
				// GPU no direct access, CPU sequential write (e.g. an upload buffer to be transferred to the GPU)
				else
				{
					// Could go to CPU memory or GPU BAR/unified. Up to the user to decide. If no preference, choose CPU memory.
					if(preferDevice)
						outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
					else
						outNotPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
				}
			}
		}
		// No CPU access
		else
		{
			// if(deviceAccess)
			//
			// GPU access, no CPU access (e.g. a color attachment image) - prefer GPU memory,
			// unless there is a clear preference from the user not to do so.
			//
			// else:
			//
			// No direct GPU access, no CPU access, just transfers.
			// It may be staging copy intended for e.g. preserving image for next frame (then better GPU memory) or
			// a "swap file" copy to free some GPU memory (then better CPU memory).
			// Up to the user to decide. If no preferece, assume the former and choose GPU memory.

			if(preferHost)
				outNotPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
			else
				outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		}
		break;
	}
	default:
		VMA_ASSERT(0);
	}

	// Avoid DEVICE_COHERENT unless explicitly requested.
	if(((allocCreateInfo.requiredFlags | allocCreateInfo.preferredFlags) &
		(VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY)) == 0)
	{
		outNotPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY;
	}

	return true;
}

////////////////////////////////////////////////////////////////////////////////
// Memory allocation

inline void* VmaMalloc(const VkAllocationCallbacks* pAllocationCallbacks, size_t size, size_t alignment)
{
	OA_REQUIRE(size > 0);
	OA_REQUIRE(alignment > 0 && (alignment & (alignment - 1)) == 0);
	void* result = VMA_NULL;
	if ((pAllocationCallbacks != VMA_NULL) &&
		(pAllocationCallbacks->pfnAllocation != VMA_NULL))
	{
		result = (*pAllocationCallbacks->pfnAllocation)(
			pAllocationCallbacks->pUserData,
			size,
			alignment,
			VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
	}
	else
	{
		result = VMA_SYSTEM_ALIGNED_MALLOC(size, alignment);
	}
	OA_REQUIRE(result != VMA_NULL);
	return result;
}

inline void VmaFree(const VkAllocationCallbacks* pAllocationCallbacks, void* ptr)
{
	if ((pAllocationCallbacks != VMA_NULL) &&
		(pAllocationCallbacks->pfnFree != VMA_NULL))
	{
		(*pAllocationCallbacks->pfnFree)(pAllocationCallbacks->pUserData, ptr);
	}
	else
	{
		VMA_SYSTEM_ALIGNED_FREE(ptr);
	}
}

template<typename T>
T* VmaAllocate(const VkAllocationCallbacks* pAllocationCallbacks)
{
	return (T*)VmaMalloc(pAllocationCallbacks, sizeof(T), VMA_ALIGN_OF(T));
}

template<typename T>
T* VmaAllocateArray(const VkAllocationCallbacks* pAllocationCallbacks, size_t count)
{
	OA_REQUIRE(count > 0);
	OA_REQUIRE(count <= static_cast<size_t>(-1) / sizeof(T));
	return (T*)VmaMalloc(pAllocationCallbacks, sizeof(T) * count, VMA_ALIGN_OF(T));
}

#define Vma_new(allocator, type)   new(VmaAllocate<type>(allocator), ::oa::Placement)(type)

#define Vma_new_array(allocator, type, count)   new(VmaAllocateArray<type>((allocator), (count)), ::oa::Placement)(type)

template<typename T>
void Vma_delete(const VkAllocationCallbacks* pAllocationCallbacks, T* ptr)
{
	ptr->~T();
	VmaFree(pAllocationCallbacks, ptr);
}

template<typename T>
void Vma_delete_array(const VkAllocationCallbacks* pAllocationCallbacks, T* ptr, size_t count)
{
	if (ptr != VMA_NULL)
	{
		for (size_t i = count; i--; )
		{
			ptr[i].~T();
		}
		VmaFree(pAllocationCallbacks, ptr);
	}
}

char* VmaCreateStringCopy(const VkAllocationCallbacks* allocs, const char* srcStr)
{
	if (srcStr != VMA_NULL)
	{
		const size_t len = oa::strlen(srcStr);
		OA_REQUIRE(len != static_cast<size_t>(-1));
		char* const result = Vma_new_array(allocs, char, len + 1);
		oa::memcpy(result, srcStr, len + 1);
		return result;
	}
	return VMA_NULL;
}

#if VMA_STATS_STRING_ENABLED
char* VmaCreateStringCopy(const VkAllocationCallbacks* allocs, const char* srcStr, size_t strLen)
{
	if (srcStr != VMA_NULL)
	{
		OA_REQUIRE(strLen != static_cast<size_t>(-1));
		char* const result = Vma_new_array(allocs, char, strLen + 1);
		oa::memcpy(result, srcStr, strLen);
		result[strLen] = '\0';
		return result;
	}
	return VMA_NULL;
}
#endif // VMA_STATS_STRING_ENABLED

void VmaFreeString(const VkAllocationCallbacks* allocs, char* str)
{
	if (str != VMA_NULL)
	{
		const size_t len = oa::strlen(str);
		OA_REQUIRE(len != static_cast<size_t>(-1));
		Vma_delete_array(allocs, str, len + 1);
	}
}

template<typename CmpLess, typename VectorT>
size_t VmaVectorInsertSorted(VectorT& vector, const typename VectorT::value_type& value)
{
	const size_t indexToInsert = VmaBinaryFindFirstNotLess(
		vector.data(),
		vector.data() + vector.size(),
		value,
		CmpLess()) - vector.data();
	VmaVectorInsert(vector, indexToInsert, value);
	return indexToInsert;
}

template<typename CmpLess, typename VectorT>
bool VmaVectorRemoveSorted(VectorT& vector, const typename VectorT::value_type& value)
{
	CmpLess comparator;
	typename VectorT::iterator it = VmaBinaryFindFirstNotLess(
		vector.begin(),
		vector.end(),
		value,
		comparator);
	if ((it != vector.end()) && !comparator(*it, value) && !comparator(value, *it))
	{
		size_t indexToRemove = it - vector.begin();
		VmaVectorRemove(vector, indexToRemove);
		return true;
	}
	return false;
}

} // namespace

#endif // _VMA_FUNCTIONS

#ifndef _VMA_STATISTICS_FUNCTIONS

namespace
{

void VmaClearStatistics(VmaStatistics& outStats)
{
	outStats.blockCount = 0;
	outStats.allocationCount = 0;
	outStats.blockBytes = 0;
	outStats.allocationBytes = 0;
}

void VmaAddStatistics(VmaStatistics& inoutStats, const VmaStatistics& src)
{
	inoutStats.blockCount += src.blockCount;
	inoutStats.allocationCount += src.allocationCount;
	inoutStats.blockBytes += src.blockBytes;
	inoutStats.allocationBytes += src.allocationBytes;
}

void VmaClearDetailedStatistics(VmaDetailedStatistics& outStats)
{
	VmaClearStatistics(outStats.statistics);
	outStats.unusedRangeCount = 0;
	outStats.allocationSizeMin = VK_WHOLE_SIZE;
	outStats.allocationSizeMax = 0;
	outStats.unusedRangeSizeMin = VK_WHOLE_SIZE;
	outStats.unusedRangeSizeMax = 0;
}

void VmaAddDetailedStatisticsAllocation(VmaDetailedStatistics& inoutStats, VkDeviceSize size)
{
	inoutStats.statistics.allocationCount++;
	inoutStats.statistics.allocationBytes += size;
	inoutStats.allocationSizeMin = VMA_MIN(inoutStats.allocationSizeMin, size);
	inoutStats.allocationSizeMax = VMA_MAX(inoutStats.allocationSizeMax, size);
}

void VmaAddDetailedStatisticsUnusedRange(VmaDetailedStatistics& inoutStats, VkDeviceSize size)
{
	inoutStats.unusedRangeCount++;
	inoutStats.unusedRangeSizeMin = VMA_MIN(inoutStats.unusedRangeSizeMin, size);
	inoutStats.unusedRangeSizeMax = VMA_MAX(inoutStats.unusedRangeSizeMax, size);
}

void VmaAddDetailedStatistics(VmaDetailedStatistics& inoutStats, const VmaDetailedStatistics& src)
{
	VmaAddStatistics(inoutStats.statistics, src.statistics);
	inoutStats.unusedRangeCount += src.unusedRangeCount;
	inoutStats.allocationSizeMin = VMA_MIN(inoutStats.allocationSizeMin, src.allocationSizeMin);
	inoutStats.allocationSizeMax = VMA_MAX(inoutStats.allocationSizeMax, src.allocationSizeMax);
	inoutStats.unusedRangeSizeMin = VMA_MIN(inoutStats.unusedRangeSizeMin, src.unusedRangeSizeMin);
	inoutStats.unusedRangeSizeMax = VMA_MAX(inoutStats.unusedRangeSizeMax, src.unusedRangeSizeMax);
}

} // namespace

#endif // _VMA_STATISTICS_FUNCTIONS

#ifndef _VMA_MUTEX_LOCK
// Helper RAII class to lock a mutex in constructor and unlock it in destructor (at the end of scope).
struct VmaMutexLock
{
	VMA_CLASS_NO_COPY_NO_MOVE(VmaMutexLock)
public:
	explicit VmaMutexLock(VMA_MUTEX& mutex, bool useMutex = true) :
		m_pMutex(useMutex ? &mutex : VMA_NULL)
	{
		if (m_pMutex) { m_pMutex->Lock(); }
	}
	~VmaMutexLock() {  if (m_pMutex) { m_pMutex->Unlock(); } }

private:
	VMA_MUTEX* m_pMutex;
};

// Helper RAII class to lock a RW mutex in constructor and unlock it in destructor (at the end of scope), for reading.
struct VmaMutexLockRead
{
	VMA_CLASS_NO_COPY_NO_MOVE(VmaMutexLockRead)
public:
	VmaMutexLockRead(VMA_RW_MUTEX& mutex, bool useMutex) :
		m_pMutex(useMutex ? &mutex : VMA_NULL)
	{
		if (m_pMutex) { m_pMutex->LockRead(); }
	}
	~VmaMutexLockRead() { if (m_pMutex) { m_pMutex->UnlockRead(); } }

private:
	VMA_RW_MUTEX* m_pMutex;
};

// Helper RAII class to lock a RW mutex in constructor and unlock it in destructor (at the end of scope), for writing.
struct VmaMutexLockWrite
{
	VMA_CLASS_NO_COPY_NO_MOVE(VmaMutexLockWrite)
public:
	VmaMutexLockWrite(VMA_RW_MUTEX& mutex, bool useMutex)
		: m_pMutex(useMutex ? &mutex : VMA_NULL)
	{
		if (m_pMutex) { m_pMutex->LockWrite(); }
	}
	~VmaMutexLockWrite() { if (m_pMutex) { m_pMutex->UnlockWrite(); } }

private:
	VMA_RW_MUTEX* m_pMutex;
};

#if VMA_DEBUG_GLOBAL_MUTEX
	static VMA_MUTEX gDebugGlobalMutex;
	#define VMA_DEBUG_GLOBAL_MUTEX_LOCK VmaMutexLock debugGlobalMutexLock(gDebugGlobalMutex, true);
#else
	#define VMA_DEBUG_GLOBAL_MUTEX_LOCK
#endif
#endif // _VMA_MUTEX_LOCK

#ifndef _VMA_ATOMIC_TRANSACTIONAL_INCREMENT
// An object that increments given atomic but decrements it back in the destructor unless Commit() is called.
template<typename AtomicT>
struct AtomicTransactionalIncrement
{
public:
	using T = decltype(AtomicT().load());

	~AtomicTransactionalIncrement()
	{
		if(m_Atomic)
			--(*m_Atomic);
	}

	void Commit() { m_Atomic = VMA_NULL; }
	T Increment(AtomicT* atomic)
	{
		m_Atomic = atomic;
		return m_Atomic->fetchAdd(1);
	}

private:
	AtomicT* m_Atomic = VMA_NULL;
};
#endif // _VMA_ATOMIC_TRANSACTIONAL_INCREMENT
