// SPDX: MIT. Derived from Vulkan Memory Allocator, (c) 2017-2026 Advanced Micro Devices, Inc.
// Full MIT copyright/permission notice: Source/Public/Oa/Runtime/OaVma.h. See NOTICE.md.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <utility>
#include <type_traits>

#if !defined(OA_VMA_CPP20)
    #if __cplusplus >= 202002L || _MSVC_LANG >= 202002L // C++20
        #define OA_VMA_CPP20 1
    #else
        #define OA_VMA_CPP20 0
    #endif
#endif

#ifdef _MSC_VER
    #include <intrin.h> // For functions like __popcnt, _BitScanForward etc.
#endif
#if OA_VMA_CPP20
    #include <bit>
#endif

#if OA_VMA_STATS_STRING_ENABLED
    #include <cstdio> // For snprintf
#endif

/*******************************************************************************
CONFIGURATION SECTION

Define some of these macros before each #include of this header or change them
here if you need other then default behavior depending on your environment.
*/
#ifndef _OA_VMA_CONFIGURATION

/*
Define this macro to 1 to make the library fetch pointers to Vulkan functions
internally, like:

    vulkanFunctions.vkAllocateMemory = &vkAllocateMemory;
*/
#if !defined(OA_VMA_STATIC_VULKAN_FUNCTIONS) && !defined(VK_NO_PROTOTYPES)
    #define OA_VMA_STATIC_VULKAN_FUNCTIONS 1
#endif

/*
Define this macro to 1 to make the library fetch pointers to Vulkan functions
internally, like:

    vulkanFunctions.vkAllocateMemory = (PFN_vkAllocateMemory)vkGetDeviceProcAddr(device, "vkAllocateMemory");

To use this feature in new versions of VMA you now have to pass
OaVmaVulkanFunctions::vkGetInstanceProcAddr and vkGetDeviceProcAddr as
OaVmaAllocatorCreateInfo::pVulkanFunctions. Other members can be null.
*/
#if !defined(OA_VMA_DYNAMIC_VULKAN_FUNCTIONS)
    #define OA_VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#endif

#ifndef OA_VMA_USE_STL_SHARED_MUTEX
    #if __cplusplus >= 201703L || _MSVC_LANG >= 201703L // C++17
        #define OA_VMA_USE_STL_SHARED_MUTEX 1
    // Visual studio defines __cplusplus properly only when passed additional parameter: /Zc:__cplusplus
    // Otherwise it is always 199711L, despite shared_mutex works since Visual Studio 2015 Update 2.
    #elif defined(_MSC_FULL_VER) && _MSC_FULL_VER >= 190023918 && __cplusplus == 199711L && _MSVC_LANG >= 201703L
        #define OA_VMA_USE_STL_SHARED_MUTEX 1
    #else
        #define OA_VMA_USE_STL_SHARED_MUTEX 0
    #endif
#endif

/*
Define this macro to include custom header files without having to edit this file directly, e.g.:

    // Inside of "my_OaVma_configuration_user_includes.h":

    #include "my_custom_assert.h" // for MY_CUSTOM_ASSERT
    #include "my_custom_min.h" // for my_custom_min
    #include <algorithm>
    #include <mutex>

    // Inside a different file, which includes OaVma.h:

    #define OA_VMA_CONFIGURATION_USER_INCLUDES_H "my_OaVma_configuration_user_includes.h"
    #define OA_VMA_ASSERT(expr) MY_CUSTOM_ASSERT(expr)
    #define OA_VMA_MIN(v1, v2)  (my_custom_min(v1, v2))
    #include <Oa/Runtime/OaVma.h>
    ...

The following headers are used in this CONFIGURATION section only, so feel free to
remove them if not needed.
*/
#if !defined(OA_VMA_CONFIGURATION_USER_INCLUDES_H)
    #include <cassert> // for assert
    #include <algorithm> // for min, max, swap
    #include <mutex>
#else
    #include OA_VMA_CONFIGURATION_USER_INCLUDES_H
#endif

#ifndef OA_VMA_NULL
   // Value used as null pointer. Define it to e.g.: nullptr, NULL, 0, (void*)0.
   #define OA_VMA_NULL   nullptr
#endif

#ifndef OA_VMA_FALLTHROUGH
    #if __cplusplus >= 201703L || _MSVC_LANG >= 201703L // C++17
        #define OA_VMA_FALLTHROUGH [[fallthrough]]
    #else
        #define OA_VMA_FALLTHROUGH
    #endif
#endif

// Normal assert to check for programmer's errors, especially in Debug configuration.
#ifndef OA_VMA_ASSERT
   #ifdef NDEBUG
       #define OA_VMA_ASSERT(expr)
   #else
       #define OA_VMA_ASSERT(expr)         assert(expr)
   #endif
#endif

// Assert that will be called very often, like inside data structures e.g. operator[].
// Making it non-empty can make program slow.
#ifndef OA_VMA_HEAVY_ASSERT
   #ifdef NDEBUG
       #define OA_VMA_HEAVY_ASSERT(expr)
   #else
       #define OA_VMA_HEAVY_ASSERT(expr)   //OA_VMA_ASSERT(expr)
   #endif
#endif

// Assert used for reporting memory leaks - unfreed allocations.
#ifndef OA_VMA_ASSERT_LEAK
    #define OA_VMA_ASSERT_LEAK(expr)   OA_VMA_ASSERT(expr)
#endif

// If your compiler is not compatible with C++17 and definition of
// aligned_alloc() function is missing, uncommenting following line may help:

//#include <malloc.h>

#if defined(__ANDROID_API__) && (__ANDROID_API__ < 16)
#include <cstdlib>
namespace
{
void* OaVma_aligned_alloc(size_t alignment, size_t size)
{
    // alignment must be >= sizeof(void*)
    if(alignment < sizeof(void*))
    {
        alignment = sizeof(void*);
    }

    return memalign(alignment, size);
}
} // namespace
#elif defined(__APPLE__) || defined(__ANDROID__) || (defined(__linux__) && defined(__GLIBCXX__) && !defined(_GLIBCXX_HAVE_ALIGNED_ALLOC))
#include <cstdlib>

#if defined(__APPLE__)
#include <AvailabilityMacros.h>
#endif

namespace
{
void* OaVma_aligned_alloc(size_t alignment, size_t size)
{
    // Unfortunately, aligned_alloc causes VMA to crash due to it returning null pointers. (At least under 11.4)
    // Therefore, for now disable this specific exception until a proper solution is found.
    //#if defined(__APPLE__) && (defined(MAC_OS_X_VERSION_10_16) || defined(__IPHONE_14_0))
    //#if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_16 || __IPHONE_OS_VERSION_MAX_ALLOWED >= __IPHONE_14_0
    //    // For C++14, usr/include/malloc/_malloc.h declares aligned_alloc()) only
    //    // with the MacOSX11.0 SDK in Xcode 12 (which is what adds
    //    // MAC_OS_X_VERSION_10_16), even though the function is marked
    //    // available for 10.15. That is why the preprocessor checks for 10.16 but
    //    // the __builtin_available checks for 10.15.
    //    // People who use C++17 could call aligned_alloc with the 10.15 SDK already.
    //    if (__builtin_available(macOS 10.15, iOS 13, *))
    //        return aligned_alloc(alignment, size);
    //#endif
    //#endif

    // alignment must be >= sizeof(void*)
    if(alignment < sizeof(void*))
    {
        alignment = sizeof(void*);
    }

    void *pointer;
    if(posix_memalign(&pointer, alignment, size) == 0)
        return pointer;
    return OA_VMA_NULL;
}
} // namespace
#elif defined(_WIN32)
namespace {
void* OaVma_aligned_alloc(size_t alignment, size_t size)
{
    return _aligned_malloc(size, alignment);
}
} // namespace
#elif __cplusplus >= 201703L || _MSVC_LANG >= 201703L // C++17
namespace {
void* OaVma_aligned_alloc(size_t alignment, size_t size)
{
    return aligned_alloc(alignment, size);
}
} // namespace
#else
namespace
{
void* OaVma_aligned_alloc(size_t alignment, size_t size)
{
    OA_VMA_ASSERT(0 && "Could not implement aligned_alloc automatically. Please enable C++17 or later in your compiler or provide custom implementation of macro OA_VMA_SYSTEM_ALIGNED_MALLOC (and OA_VMA_SYSTEM_ALIGNED_FREE if needed) using the API of your system.");
    return OA_VMA_NULL;
}
} // namespace
#endif

namespace
{
#if defined(_WIN32)
void OaVma_aligned_free(void* ptr)
{
    _aligned_free(ptr);
}
#else
void OaVma_aligned_free(void* OA_VMA_NULLABLE ptr)
{
    free(ptr);
}
#endif
} // namespace

#ifndef OA_VMA_ALIGN_OF
   #define OA_VMA_ALIGN_OF(type)       (alignof(type))
#endif

#ifndef OA_VMA_SYSTEM_ALIGNED_MALLOC
   #define OA_VMA_SYSTEM_ALIGNED_MALLOC(size, alignment) OaVma_aligned_alloc((alignment), (size))
#endif

#ifndef OA_VMA_SYSTEM_ALIGNED_FREE
   // OA_VMA_SYSTEM_FREE is the old name, but might have been defined by the user
   #if defined(OA_VMA_SYSTEM_FREE)
      #define OA_VMA_SYSTEM_ALIGNED_FREE(ptr)     OA_VMA_SYSTEM_FREE(ptr)
   #else
      #define OA_VMA_SYSTEM_ALIGNED_FREE(ptr)     OaVma_aligned_free(ptr)
    #endif
#endif

#ifndef OA_VMA_COUNT_BITS_SET
    // Returns number of bits set to 1 in (v)
    #define OA_VMA_COUNT_BITS_SET(v) OaVmaCountBitsSet(v)
#endif

#ifndef OA_VMA_BITSCAN_LSB
    // Scans integer for index of first nonzero value from the Least Significant Bit (LSB). If mask is 0 then returns UINT8_MAX
    #define OA_VMA_BITSCAN_LSB(mask) OaVmaBitScanLSB(mask)
#endif

#ifndef OA_VMA_BITSCAN_MSB
    // Scans integer for index of first nonzero value from the Most Significant Bit (MSB). If mask is 0 then returns UINT8_MAX
    #define OA_VMA_BITSCAN_MSB(mask) OaVmaBitScanMSB(mask)
#endif

#ifndef OA_VMA_MIN
   #define OA_VMA_MIN(v1, v2)    ((std::min)((v1), (v2)))
#endif

#ifndef OA_VMA_MAX
   #define OA_VMA_MAX(v1, v2)    ((std::max)((v1), (v2)))
#endif

#ifndef OA_VMA_SORT
   #define OA_VMA_SORT(beg, end, cmp)  std::sort(beg, end, cmp)
#endif

#ifndef OA_VMA_DEBUG_LOG_FORMAT
   #define OA_VMA_DEBUG_LOG_FORMAT(format, ...)
   /*
   #define OA_VMA_DEBUG_LOG_FORMAT(format, ...) do { \
       printf((format), __VA_ARGS__); \
       printf("\n"); \
   } while(false)
   */
#endif

#ifndef OA_VMA_DEBUG_LOG
    #define OA_VMA_DEBUG_LOG(str)   OA_VMA_DEBUG_LOG_FORMAT("%s", (str))
#endif

#ifndef OA_VMA_LEAK_LOG_FORMAT
    #define OA_VMA_LEAK_LOG_FORMAT(format, ...)   OA_VMA_DEBUG_LOG_FORMAT(format, __VA_ARGS__)
#endif

#ifndef OA_VMA_CLASS_NO_COPY
    #define OA_VMA_CLASS_NO_COPY(className) \
        private: \
            className(const className&) = delete; \
            className& operator=(const className&) = delete;
#endif
#ifndef OA_VMA_CLASS_NO_COPY_NO_MOVE
    #define OA_VMA_CLASS_NO_COPY_NO_MOVE(className) \
        private: \
            className(const className&) = delete; \
            className(className&&) = delete; \
            className& operator=(const className&) = delete; \
            className& operator=(className&&) = delete;
#endif

// Define this macro to 1 to enable functions: OaVmaBuildStatsString, OaVmaFreeStatsString.
#if OA_VMA_STATS_STRING_ENABLED
namespace {
    inline void OaVmaUint32ToStr(char* OA_VMA_NOT_NULL outStr, size_t strLen, uint32_t num)
    {
        snprintf(outStr, strLen, "%" PRIu32, num);
    }
    inline void OaVmaUint64ToStr(char* OA_VMA_NOT_NULL outStr, size_t strLen, uint64_t num)
    {
        snprintf(outStr, strLen, "%" PRIu64, num);
    }
    inline void OaVmaPtrToStr(char* OA_VMA_NOT_NULL outStr, size_t strLen, const void* ptr)
    {
        snprintf(outStr, strLen, "%p", ptr);
    }
} // namespace
#endif

#ifndef OA_VMA_MUTEX
    class OaVmaMutex
    {
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaMutex)
    public:
        OaVmaMutex() = default;
        void Lock() { m_Mutex.lock(); }
        void Unlock() { m_Mutex.unlock(); }
        bool TryLock() { return m_Mutex.try_lock(); }
    private:
        std::mutex m_Mutex;
    };
    #define OA_VMA_MUTEX OaVmaMutex
#endif

// Read-write mutex, where "read" is shared access, "write" is exclusive access.
#ifndef OA_VMA_RW_MUTEX
    #if OA_VMA_USE_STL_SHARED_MUTEX
        // Use std::shared_mutex from C++17.
        #include <shared_mutex>
        class OaVmaRWMutex
        {
        public:
            void LockRead() { m_Mutex.lock_shared(); }
            void UnlockRead() { m_Mutex.unlock_shared(); }
            bool TryLockRead() { return m_Mutex.try_lock_shared(); }
            void LockWrite() { m_Mutex.lock(); }
            void UnlockWrite() { m_Mutex.unlock(); }
            bool TryLockWrite() { return m_Mutex.try_lock(); }
        private:
            std::shared_mutex m_Mutex;
        };
        #define OA_VMA_RW_MUTEX OaVmaRWMutex
    #elif defined(_WIN32) && defined(WINVER) && defined(SRWLOCK_INIT) && WINVER >= 0x0600
        // Use SRWLOCK from WinAPI.
        // Minimum supported client = Windows Vista, server = Windows Server 2008.
        class OaVmaRWMutex
        {
        public:
            OaVmaRWMutex() { InitializeSRWLock(&m_Lock); }
            void LockRead() { AcquireSRWLockShared(&m_Lock); }
            void UnlockRead() { ReleaseSRWLockShared(&m_Lock); }
            bool TryLockRead() { return TryAcquireSRWLockShared(&m_Lock) != FALSE; }
            void LockWrite() { AcquireSRWLockExclusive(&m_Lock); }
            void UnlockWrite() { ReleaseSRWLockExclusive(&m_Lock); }
            bool TryLockWrite() { return TryAcquireSRWLockExclusive(&m_Lock) != FALSE; }
        private:
            SRWLOCK m_Lock;
        };
        #define OA_VMA_RW_MUTEX OaVmaRWMutex
    #else
        // Less efficient fallback: Use normal mutex.
        class OaVmaRWMutex
        {
        public:
            void LockRead() { m_Mutex.Lock(); }
            void UnlockRead() { m_Mutex.Unlock(); }
            bool TryLockRead() { return m_Mutex.TryLock(); }
            void LockWrite() { m_Mutex.Lock(); }
            void UnlockWrite() { m_Mutex.Unlock(); }
            bool TryLockWrite() { return m_Mutex.TryLock(); }
        private:
            OA_VMA_MUTEX m_Mutex;
        };
        #define OA_VMA_RW_MUTEX OaVmaRWMutex
    #endif // #if OA_VMA_USE_STL_SHARED_MUTEX
#endif // #ifndef OA_VMA_RW_MUTEX

/*
If providing your own implementation, you need to implement a subset of std::atomic.
*/
#ifndef OA_VMA_ATOMIC_UINT32
    #include <atomic>
    #define OA_VMA_ATOMIC_UINT32 std::atomic<uint32_t>
#endif

#ifndef OA_VMA_ATOMIC_UINT64
    #include <atomic>
    #define OA_VMA_ATOMIC_UINT64 std::atomic<uint64_t>
#endif

#ifndef OA_VMA_DEBUG_ALWAYS_DEDICATED_MEMORY
    /**
    Every allocation will have its own memory block.
    Define to 1 for debugging purposes only.
    */
    #define OA_VMA_DEBUG_ALWAYS_DEDICATED_MEMORY (0)
#endif

#ifndef OA_VMA_MIN_ALIGNMENT
    /**
    Minimum alignment of all allocations, in bytes.
    Set to more than 1 for debugging purposes. Must be power of two.
    */
    #ifdef OA_VMA_DEBUG_ALIGNMENT // Old name
        #define OA_VMA_MIN_ALIGNMENT OA_VMA_DEBUG_ALIGNMENT
    #else
        #define OA_VMA_MIN_ALIGNMENT (1)
    #endif
#endif

#ifndef OA_VMA_DEBUG_MARGIN
    /**
    Minimum margin after every allocation, in bytes.
    Set nonzero for debugging purposes only.
    */
    #define OA_VMA_DEBUG_MARGIN (0)
#endif

#ifndef OA_VMA_DEBUG_INITIALIZE_ALLOCATIONS
    /**
    Define this macro to 1 to automatically fill new allocations and destroyed
    allocations with some bit pattern.
    */
    #define OA_VMA_DEBUG_INITIALIZE_ALLOCATIONS (0)
#endif

#ifndef OA_VMA_DEBUG_DETECT_CORRUPTION
    /**
    Define this macro to 1 together with non-zero value of OA_VMA_DEBUG_MARGIN to
    enable writing magic value to the margin after every allocation and
    validating it, so that memory corruptions (out-of-bounds writes) are detected.
    */
    #define OA_VMA_DEBUG_DETECT_CORRUPTION (0)
#endif

#ifndef OA_VMA_DEBUG_GLOBAL_MUTEX
    /**
    Set this to 1 for debugging purposes only, to enable single mutex protecting all
    entry calls to the library. Can be useful for debugging multithreading issues.
    */
    #define OA_VMA_DEBUG_GLOBAL_MUTEX (0)
#endif

#ifndef OA_VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY
    /**
    Minimum value for VkPhysicalDeviceLimits::bufferImageGranularity.
    Set to more than 1 for debugging purposes only. Must be power of two.
    */
    #define OA_VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY (1)
#endif

#ifndef OA_VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT
    /*
    Set this to 1 to make VMA never exceed VkPhysicalDeviceLimits::maxMemoryAllocationCount
    and return error instead of leaving up to Vulkan implementation what to do in such cases.
    */
    #define OA_VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT (1)
#endif

#ifndef OA_VMA_DEBUG_DONT_EXCEED_HEAP_SIZE_WITH_ALLOCATION_SIZE
    /*
    Set this to 1 to make VMA never exceed VkPhysicalDeviceMemoryProperties::memoryHeaps[i].size
    with a single allocation size VkMemoryAllocateInfo::allocationSize
    and return error instead of leaving up to Vulkan implementation what to do in such cases.
    It protects agaist validation error VUID-vkAllocateMemory-pAllocateInfo-01713.
    On the other hand, allowing exceeding this size may result in a successful allocation despite the validation error.
    */
    #define OA_VMA_DEBUG_DONT_EXCEED_HEAP_SIZE_WITH_ALLOCATION_SIZE (1)
#endif

#ifndef OA_VMA_SMALL_HEAP_MAX_SIZE
   /// Maximum size of a memory heap in Vulkan to consider it "small".
   #define OA_VMA_SMALL_HEAP_MAX_SIZE (1024ULL * 1024 * 1024)
#endif

#ifndef OA_VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE
   /// Default size of a block allocated as single VkDeviceMemory from a "large" heap.
   #define OA_VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE (256ULL * 1024 * 1024)
#endif

/*
Mapping hysteresis is a logic that launches when OaVmaMapMemory/OaVmaUnmapMemory is called
or a persistently mapped allocation is created and destroyed several times in a row.
It keeps additional +1 mapping of a device memory block to prevent calling actual
vkMapMemory/vkUnmapMemory too many times, which may improve performance and help
tools like RenderDoc.
*/
#ifndef OA_VMA_MAPPING_HYSTERESIS_ENABLED
    #define OA_VMA_MAPPING_HYSTERESIS_ENABLED 1
#endif

#define OA_VMA_VALIDATE(cond) do { if(!(cond)) { \
        OA_VMA_ASSERT(0 && "Validation failed: " #cond); \
        return false; \
    } } while(false)

/*******************************************************************************
END OF CONFIGURATION
*/
#endif // _OA_VMA_CONFIGURATION

namespace
{
constexpr uint8_t OA_VMA_ALLOCATION_FILL_PATTERN_CREATED = 0xDC;
constexpr uint8_t OA_VMA_ALLOCATION_FILL_PATTERN_DESTROYED = 0xEF;
// Decimal 2139416166, float NaN, little-endian binary 66 E6 84 7F.
constexpr uint32_t OA_VMA_CORRUPTION_DETECTION_MAGIC_VALUE = 0x7F84E666;

// Copy of some Vulkan definitions so we don't need to check their existence just to handle few constants.
constexpr uint32_t VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY = 0x00000040;
constexpr uint32_t VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY = 0x00000080;
constexpr uint32_t VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_COPY = 0x00020000;
constexpr uint32_t VK_IMAGE_CREATE_DISJOINT_BIT_COPY = 0x00000200;
constexpr int32_t VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT_COPY = 1000158000;
constexpr uint32_t OA_VMA_ALLOCATION_INTERNAL_STRATEGY_MIN_OFFSET = 0x10000000U;
constexpr uint32_t OA_VMA_ALLOCATION_TRY_COUNT = 32;
constexpr uint32_t OA_VMA_VENDOR_ID_AMD = 4098;

// This one is tricky. Vulkan specification defines this code as available since
// Vulkan 1.0, but doesn't actually define it in Vulkan SDK earlier than 1.2.131.
// See pull request #207.
#define VK_ERROR_UNKNOWN_COPY ((VkResult)-13)
} // namespace


#if OA_VMA_STATS_STRING_ENABLED
// Correspond to values of enum OaVmaSuballocationType.
const char* const OA_VMA_SUBALLOCATION_TYPE_NAMES[] =
{
    "FREE",
    "UNKNOWN",
    "BUFFER",
    "IMAGE_UNKNOWN",
    "IMAGE_LINEAR",
    "IMAGE_OPTIMAL",
};
#endif

const VkAllocationCallbacks OaVmaEmptyAllocationCallbacks =
    { OA_VMA_NULL, OA_VMA_NULL, OA_VMA_NULL, OA_VMA_NULL, OA_VMA_NULL, OA_VMA_NULL };


#ifndef _OA_VMA_ENUM_DECLARATIONS

enum OaVmaSuballocationType
{
    OA_VMA_SUBALLOCATION_TYPE_FREE = 0,
    OA_VMA_SUBALLOCATION_TYPE_UNKNOWN = 1,
    OA_VMA_SUBALLOCATION_TYPE_BUFFER = 2,
    OA_VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN = 3,
    OA_VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR = 4,
    OA_VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL = 5,
    OA_VMA_SUBALLOCATION_TYPE_MAX_ENUM = 0x7FFFFFFF
};

enum OA_VMA_CACHE_OPERATION
{
    OA_VMA_CACHE_FLUSH,
    OA_VMA_CACHE_INVALIDATE
};

enum class OaVmaAllocationRequestType
{
    Normal,
    TLSF,
    // Used by "Linear" algorithm.
    UpperAddress,
    EndOf1st,
    EndOf2nd,
};

#endif // _OA_VMA_ENUM_DECLARATIONS

#ifndef _OA_VMA_FORWARD_DECLARATIONS
// Opaque handle used by allocation algorithms to identify single allocation in any conforming way.
VK_DEFINE_NON_DISPATCHABLE_HANDLE(OaVmaAllocHandle);

struct OaVmaBufferImageUsage;

struct OaVmaMutexLock;
struct OaVmaMutexLockRead;
struct OaVmaMutexLockWrite;

template<typename T>
struct AtomicTransactionalIncrement;

template<typename T>
struct OaVmaStlAllocator;

template<typename T, typename AllocatorT>
class OaVmaVector;

template<typename T, typename AllocatorT, size_t N>
class OaVmaSmallVector;

template<typename T>
class OaVmaPoolAllocator;

template<typename T>
struct OaVmaListItem;

template<typename T>
class OaVmaRawList;

template<typename T, typename AllocatorT>
class OaVmaList;

template<typename ItemTypeTraits>
class OaVmaIntrusiveLinkedList;

#if OA_VMA_STATS_STRING_ENABLED
class OaVmaStringBuilder;
class OaVmaJsonWriter;
#endif

class OaVmaDeviceMemoryBlock;

struct OaVmaDedicatedAllocationListItemTraits;
class OaVmaDedicatedAllocationList;

struct OaVmaSuballocation;
struct OaVmaSuballocationOffsetLess;
struct OaVmaSuballocationOffsetGreater;
struct OaVmaSuballocationItemSizeLess;

typedef OaVmaList<OaVmaSuballocation, OaVmaStlAllocator<OaVmaSuballocation>> OaVmaSuballocationList;

struct OaVmaAllocationRequest;

class OaVmaBlockMetadata;
class OaVmaBlockMetadata_Linear;
class OaVmaBlockMetadata_TLSF;

class OaVmaBlockVector;

struct OaVmaPoolListItemTraits;

struct OaVmaCurrentBudgetData;

class OaVmaAllocationObjectAllocator;

#endif // _OA_VMA_FORWARD_DECLARATIONS

#ifndef _OA_VMA_BUFFER_IMAGE_USAGE

// Finds structure with s->sType == sType in mainStruct->pNext chain.
// Returns pointer to it. If not found, returns null.
template<typename FindT, typename MainT>
inline const FindT* OaVmaPnextChainFind(const MainT* mainStruct, VkStructureType sType)
{
    for(const VkBaseInStructure* s = (const VkBaseInStructure*)mainStruct->pNext;
        s != OA_VMA_NULL; s = s->pNext)
    {
        if(s->sType == sType)
        {
            return (const FindT*)s;
        }
    }
    return OA_VMA_NULL;
}

// An abstraction over buffer or image `usage` flags, depending on available extensions.
struct OaVmaBufferImageUsage
{
#if OA_VMA_KHR_MAINTENANCE5
    typedef uint64_t BaseType; // VkFlags64
#else
    typedef uint32_t BaseType; // VkFlags32
#endif

    static const OaVmaBufferImageUsage UNKNOWN;

    BaseType Value;

    OaVmaBufferImageUsage() { *this = UNKNOWN; }
    explicit OaVmaBufferImageUsage(BaseType usage) : Value(usage) { }
    OaVmaBufferImageUsage(const VkBufferCreateInfo &createInfo, bool useKhrMaintenance5);
    explicit OaVmaBufferImageUsage(const VkImageCreateInfo &createInfo);

    bool operator==(const OaVmaBufferImageUsage& rhs) const { return Value == rhs.Value; }
    bool operator!=(const OaVmaBufferImageUsage& rhs) const { return Value != rhs.Value; }

    bool Contains(BaseType flag) const { return (Value & flag) != 0; }
    bool ContainsDeviceAccess() const
    {
        // This relies on values of VK_IMAGE_USAGE_TRANSFER* being the same as VK_BUFFER_IMAGE_TRANSFER*.
        return (Value & ~BaseType(VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) != 0;
    }
};

const OaVmaBufferImageUsage OaVmaBufferImageUsage::UNKNOWN = OaVmaBufferImageUsage(0);

OaVmaBufferImageUsage::OaVmaBufferImageUsage(const VkBufferCreateInfo &createInfo,
    bool useKhrMaintenance5)
{
#if OA_VMA_KHR_MAINTENANCE5
    if(useKhrMaintenance5)
    {
        // If VkBufferCreateInfo::pNext chain contains VkBufferUsageFlags2CreateInfoKHR,
        // take usage from it and ignore VkBufferCreateInfo::usage, per specification
        // of the VK_KHR_maintenance5 extension.
        const VkBufferUsageFlags2CreateInfoKHR* const usageFlags2 =
            OaVmaPnextChainFind<VkBufferUsageFlags2CreateInfoKHR>(&createInfo, VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR);
        if(usageFlags2 != OA_VMA_NULL)
        {
            this->Value = usageFlags2->usage;
            return;
        }
    }
#endif

    this->Value = (BaseType)createInfo.usage;
}

OaVmaBufferImageUsage::OaVmaBufferImageUsage(const VkImageCreateInfo &createInfo)
    : Value((BaseType)createInfo.usage)
{
    // Maybe in the future there will be VK_KHR_maintenanceN extension with structure
    // VkImageUsageFlags2CreateInfoKHR, like the one for buffers...
}

#endif // _OA_VMA_BUFFER_IMAGE_USAGE

#ifndef _OA_VMA_FUNCTIONS

namespace
{

/*
Returns number of bits set to 1 in (v).

On specific platforms and compilers you can use intrinsics like:

Visual Studio:
    return __popcnt(v);
GCC, Clang:
    return static_cast<uint32_t>(__builtin_popcount(v));

Define macro OA_VMA_COUNT_BITS_SET to provide your optimized implementation.
But you need to check in runtime whether user's CPU supports these, as some old processors don't.
*/
inline uint32_t OaVmaCountBitsSet(uint32_t v)
{
#if OA_VMA_CPP20
    return std::popcount(v);
#else
    uint32_t c = v - ((v >> 1) & 0x55555555);
    c = ((c >> 2) & 0x33333333) + (c & 0x33333333);
    c = ((c >> 4) + c) & 0x0F0F0F0F;
    c = ((c >> 8) + c) & 0x00FF00FF;
    c = ((c >> 16) + c) & 0x0000FFFF;
    return c;
#endif
}

inline uint8_t OaVmaBitScanLSB(uint64_t mask)
{
#if defined(_MSC_VER) && defined(_WIN64)
    unsigned long pos;
    if (_BitScanForward64(&pos, mask))
        return static_cast<uint8_t>(pos);
    return UINT8_MAX;
#elif OA_VMA_CPP20
    if(mask != 0)
        return static_cast<uint8_t>(std::countr_zero(mask));
    return UINT8_MAX;
#elif defined __GNUC__ || defined __clang__
    return static_cast<uint8_t>(__builtin_ffsll(mask)) - 1U;
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

inline uint8_t OaVmaBitScanLSB(uint32_t mask)
{
#ifdef _MSC_VER
    unsigned long pos;
    if (_BitScanForward(&pos, mask))
        return static_cast<uint8_t>(pos);
    return UINT8_MAX;
#elif OA_VMA_CPP20
    if(mask != 0)
        return static_cast<uint8_t>(std::countr_zero(mask));
    return UINT8_MAX;
#elif defined __GNUC__ || defined __clang__
    return static_cast<uint8_t>(__builtin_ffs(mask)) - 1U;
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

inline uint8_t OaVmaBitScanMSB(uint64_t mask)
{
#if defined(_MSC_VER) && defined(_WIN64)
    unsigned long pos;
    if (_BitScanReverse64(&pos, mask))
        return static_cast<uint8_t>(pos);
#elif OA_VMA_CPP20
    if(mask != 0)
        return 63 - static_cast<uint8_t>(std::countl_zero(mask));
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

inline uint8_t OaVmaBitScanMSB(uint32_t mask)
{
#ifdef _MSC_VER
    unsigned long pos;
    if (_BitScanReverse(&pos, mask))
        return static_cast<uint8_t>(pos);
#elif OA_VMA_CPP20
    if(mask != 0)
        return 31 - static_cast<uint8_t>(std::countl_zero(mask));
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
inline bool OaVmaIsPow2(T x)
{
    return (x & (x - 1)) == 0;
}

// Aligns given value up to nearest multiply of align value. For example: OaVmaAlignUp(11, 8) = 16.
// Use types like uint32_t, uint64_t as T.
template <typename T>
inline T OaVmaAlignUp(T val, T alignment)
{
    OA_VMA_HEAVY_ASSERT(OaVmaIsPow2(alignment));
    return (val + alignment - 1) & ~(alignment - 1);
}

// Aligns given value down to nearest multiply of align value. For example: OaVmaAlignDown(11, 8) = 8.
// Use types like uint32_t, uint64_t as T.
template <typename T>
inline T OaVmaAlignDown(T val, T alignment)
{
    OA_VMA_HEAVY_ASSERT(OaVmaIsPow2(alignment));
    return val & ~(alignment - 1);
}

// Division with mathematical rounding to nearest number.
template <typename T>
inline T OaVmaRoundDiv(T x, T y)
{
    return (x + (y / (T)2)) / y;
}

// Divide by 'y' and round up to nearest integer.
template <typename T>
inline T OaVmaDivideRoundingUp(T x, T y)
{
    return (x + y - (T)1) / y;
}

// Returns smallest power of 2 greater or equal to v.
inline uint32_t OaVmaNextPow2(uint32_t v)
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

inline uint64_t OaVmaNextPow2(uint64_t v)
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
inline uint32_t OaVmaPrevPow2(uint32_t v)
{
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v = v ^ (v >> 1);
    return v;
}

inline uint64_t OaVmaPrevPow2(uint64_t v)
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

inline bool OaVmaStrIsEmpty(const char* pStr)
{
    return pStr == OA_VMA_NULL || *pStr == '\0';
}

/*
Returns true if two memory blocks occupy overlapping pages.
ResourceA must be in less memory offset than ResourceB.

Algorithm is based on "Vulkan 1.0.39 - A Specification (with all registered Vulkan extensions)"
chapter 11.6 "Resource Memory Association", paragraph "Buffer-Image Granularity".
*/
inline bool OaVmaBlocksOnSamePage(
    VkDeviceSize resourceAOffset,
    VkDeviceSize resourceASize,
    VkDeviceSize resourceBOffset,
    VkDeviceSize pageSize)
{
    OA_VMA_ASSERT(resourceAOffset + resourceASize <= resourceBOffset && resourceASize > 0 && pageSize > 0);
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
inline bool OaVmaIsBufferImageGranularityConflict(
    OaVmaSuballocationType suballocType1,
    OaVmaSuballocationType suballocType2)
{
    if (suballocType1 > suballocType2)
    {
        std::swap(suballocType1, suballocType2);
    }

    switch (suballocType1)
    {
    case OA_VMA_SUBALLOCATION_TYPE_FREE:
        return false;
    case OA_VMA_SUBALLOCATION_TYPE_UNKNOWN:
        return true;
    case OA_VMA_SUBALLOCATION_TYPE_BUFFER:
        return
            suballocType2 == OA_VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN ||
            suballocType2 == OA_VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL;
    case OA_VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN:
        return
            suballocType2 == OA_VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN ||
            suballocType2 == OA_VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR ||
            suballocType2 == OA_VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL;
    case OA_VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR:
        return
            suballocType2 == OA_VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL;
    case OA_VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL:
        return false;
    default:
        OA_VMA_ASSERT(0);
        return true;
    }
}

void OaVmaWriteMagicValue(void* pData, VkDeviceSize offset)
{
#if OA_VMA_DEBUG_MARGIN > 0 && OA_VMA_DEBUG_DETECT_CORRUPTION
    uint32_t* pDst = (uint32_t*)((char*)pData + offset);
    const size_t numberCount = OA_VMA_DEBUG_MARGIN / sizeof(uint32_t);
    for (size_t i = 0; i < numberCount; ++i, ++pDst)
    {
        *pDst = OA_VMA_CORRUPTION_DETECTION_MAGIC_VALUE;
    }
#else
    // no-op
#endif
}

bool OaVmaValidateMagicValue(const void* pData, VkDeviceSize offset)
{
#if OA_VMA_DEBUG_MARGIN > 0 && OA_VMA_DEBUG_DETECT_CORRUPTION
    const uint32_t* pSrc = (const uint32_t*)((const char*)pData + offset);
    const size_t numberCount = OA_VMA_DEBUG_MARGIN / sizeof(uint32_t);
    for (size_t i = 0; i < numberCount; ++i, ++pSrc)
    {
        if (*pSrc != OA_VMA_CORRUPTION_DETECTION_MAGIC_VALUE)
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
void OaVmaFillGpuDefragmentationBufferCreateInfo(VkBufferCreateInfo& outBufCreateInfo)
{
    memset(&outBufCreateInfo, 0, sizeof(outBufCreateInfo));
    outBufCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    outBufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    outBufCreateInfo.size = (VkDeviceSize)OA_VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE; // Example size.
}


/*
Performs binary search and returns iterator to first element that is greater or
equal to (key), according to comparison (cmp).

Cmp should return true if first argument is less than second argument.

Returned value is the found element, if present in the collection or place where
new element with value (key) should be inserted.
*/
template <typename CmpLess, typename IterT, typename KeyT>
IterT OaVmaBinaryFindFirstNotLess(IterT beg, IterT end, const KeyT& key, const CmpLess& cmp)
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
IterT OaVmaBinaryFindSorted(const IterT& beg, const IterT& end, const KeyT& value, const CmpLess& cmp)
{
    IterT it = OaVmaBinaryFindFirstNotLess<CmpLess, IterT, KeyT>(
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
Warning! O(n^2) complexity. Use only inside OA_VMA_HEAVY_ASSERT.
T must be pointer type, e.g. OaVmaAllocation, OaVmaPool.
*/
template<typename T>
bool OaVmaValidatePointerArray(uint32_t count, const T* arr)
{
    for (uint32_t i = 0; i < count; ++i)
    {
        const T iPtr = arr[i];
        if (iPtr == OA_VMA_NULL)
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
inline void OaVmaPnextChainPushFront(MainT* mainStruct, NewT* newStruct)
{
    newStruct->pNext = mainStruct->pNext;
    mainStruct->pNext = newStruct;
}

// This is the main algorithm that guides the selection of a memory type best for an allocation -
// converts usage to required/preferred/not preferred flags.
bool FindMemoryPreferences(
    bool isIntegratedGPU,
    const OaVmaAllocationCreateInfo& allocCreateInfo,
    OaVmaBufferImageUsage bufImgUsage,
    VkMemoryPropertyFlags& outRequiredFlags,
    VkMemoryPropertyFlags& outPreferredFlags,
    VkMemoryPropertyFlags& outNotPreferredFlags)
{
    outRequiredFlags = allocCreateInfo.requiredFlags;
    outPreferredFlags = allocCreateInfo.preferredFlags;
    outNotPreferredFlags = 0;

    switch(allocCreateInfo.usage)
    {
    case OA_VMA_MEMORY_USAGE_UNKNOWN:
        break;
    case OA_VMA_MEMORY_USAGE_GPU_ONLY:
        if(!isIntegratedGPU || (outPreferredFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
        {
            outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }
        break;
    case OA_VMA_MEMORY_USAGE_CPU_ONLY:
        outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        break;
    case OA_VMA_MEMORY_USAGE_CPU_TO_GPU:
        outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        if(!isIntegratedGPU || (outPreferredFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
        {
            outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }
        break;
    case OA_VMA_MEMORY_USAGE_GPU_TO_CPU:
        outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        outPreferredFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        break;
    case OA_VMA_MEMORY_USAGE_CPU_COPY:
        outNotPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case OA_VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED:
        outRequiredFlags |= VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
        break;
    case OA_VMA_MEMORY_USAGE_AUTO:
    case OA_VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE:
    case OA_VMA_MEMORY_USAGE_AUTO_PREFER_HOST:
    {
        if(bufImgUsage == OaVmaBufferImageUsage::UNKNOWN)
        {
            OA_VMA_ASSERT(0 && "OA_VMA_MEMORY_USAGE_AUTO* values can only be used with functions like OaVmaCreateBuffer, OaVmaCreateImage so that the details of the created resource are known."
                " Maybe you use VkBufferUsageFlags2CreateInfoKHR but forgot to use OA_VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT?" );
            return false;
        }

        const bool deviceAccess = bufImgUsage.ContainsDeviceAccess();
        const bool hostAccessSequentialWrite = (allocCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT) != 0;
        const bool hostAccessRandom = (allocCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT) != 0;
        const bool hostAccessAllowTransferInstead = (allocCreateInfo.flags & OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT) != 0;
        const bool preferDevice = allocCreateInfo.usage == OA_VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        const bool preferHost = allocCreateInfo.usage == OA_VMA_MEMORY_USAGE_AUTO_PREFER_HOST;

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
        OA_VMA_ASSERT(0);
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

inline void* OaVmaMalloc(const VkAllocationCallbacks* pAllocationCallbacks, size_t size, size_t alignment)
{
    void* result = OA_VMA_NULL;
    if ((pAllocationCallbacks != OA_VMA_NULL) &&
        (pAllocationCallbacks->pfnAllocation != OA_VMA_NULL))
    {
        result = (*pAllocationCallbacks->pfnAllocation)(
            pAllocationCallbacks->pUserData,
            size,
            alignment,
            VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
    }
    else
    {
        result = OA_VMA_SYSTEM_ALIGNED_MALLOC(size, alignment);
    }
    OA_VMA_ASSERT(result != OA_VMA_NULL && "CPU memory allocation failed.");
    return result;
}

inline void OaVmaFree(const VkAllocationCallbacks* pAllocationCallbacks, void* ptr)
{
    if ((pAllocationCallbacks != OA_VMA_NULL) &&
        (pAllocationCallbacks->pfnFree != OA_VMA_NULL))
    {
        (*pAllocationCallbacks->pfnFree)(pAllocationCallbacks->pUserData, ptr);
    }
    else
    {
        OA_VMA_SYSTEM_ALIGNED_FREE(ptr);
    }
}

template<typename T>
T* OaVmaAllocate(const VkAllocationCallbacks* pAllocationCallbacks)
{
    return (T*)OaVmaMalloc(pAllocationCallbacks, sizeof(T), OA_VMA_ALIGN_OF(T));
}

template<typename T>
T* OaVmaAllocateArray(const VkAllocationCallbacks* pAllocationCallbacks, size_t count)
{
    return (T*)OaVmaMalloc(pAllocationCallbacks, sizeof(T) * count, OA_VMA_ALIGN_OF(T));
}

#define OaVma_new(allocator, type)   new(OaVmaAllocate<type>(allocator))(type)

#define OaVma_new_array(allocator, type, count)   new(OaVmaAllocateArray<type>((allocator), (count)))(type)

template<typename T>
void OaVma_delete(const VkAllocationCallbacks* pAllocationCallbacks, T* ptr)
{
    ptr->~T();
    OaVmaFree(pAllocationCallbacks, ptr);
}

template<typename T>
void OaVma_delete_array(const VkAllocationCallbacks* pAllocationCallbacks, T* ptr, size_t count)
{
    if (ptr != OA_VMA_NULL)
    {
        for (size_t i = count; i--; )
        {
            ptr[i].~T();
        }
        OaVmaFree(pAllocationCallbacks, ptr);
    }
}

char* OaVmaCreateStringCopy(const VkAllocationCallbacks* allocs, const char* srcStr)
{
    if (srcStr != OA_VMA_NULL)
    {
        const size_t len = strlen(srcStr);
        char* const result = OaVma_new_array(allocs, char, len + 1);
        memcpy(result, srcStr, len + 1);
        return result;
    }
    return OA_VMA_NULL;
}

#if OA_VMA_STATS_STRING_ENABLED
char* OaVmaCreateStringCopy(const VkAllocationCallbacks* allocs, const char* srcStr, size_t strLen)
{
    if (srcStr != OA_VMA_NULL)
    {
        char* const result = OaVma_new_array(allocs, char, strLen + 1);
        memcpy(result, srcStr, strLen);
        result[strLen] = '\0';
        return result;
    }
    return OA_VMA_NULL;
}
#endif // OA_VMA_STATS_STRING_ENABLED

void OaVmaFreeString(const VkAllocationCallbacks* allocs, char* str)
{
    if (str != OA_VMA_NULL)
    {
        const size_t len = strlen(str);
        OaVma_delete_array(allocs, str, len + 1);
    }
}

template<typename CmpLess, typename VectorT>
size_t OaVmaVectorInsertSorted(VectorT& vector, const typename VectorT::value_type& value)
{
    const size_t indexToInsert = OaVmaBinaryFindFirstNotLess(
        vector.data(),
        vector.data() + vector.size(),
        value,
        CmpLess()) - vector.data();
    OaVmaVectorInsert(vector, indexToInsert, value);
    return indexToInsert;
}

template<typename CmpLess, typename VectorT>
bool OaVmaVectorRemoveSorted(VectorT& vector, const typename VectorT::value_type& value)
{
    CmpLess comparator;
    typename VectorT::iterator it = OaVmaBinaryFindFirstNotLess(
        vector.begin(),
        vector.end(),
        value,
        comparator);
    if ((it != vector.end()) && !comparator(*it, value) && !comparator(value, *it))
    {
        size_t indexToRemove = it - vector.begin();
        OaVmaVectorRemove(vector, indexToRemove);
        return true;
    }
    return false;
}

} // namespace

#endif // _OA_VMA_FUNCTIONS

#ifndef _OA_VMA_STATISTICS_FUNCTIONS

namespace
{

void OaVmaClearStatistics(OaVmaStatistics& outStats)
{
    outStats.blockCount = 0;
    outStats.allocationCount = 0;
    outStats.blockBytes = 0;
    outStats.allocationBytes = 0;
}

void OaVmaAddStatistics(OaVmaStatistics& inoutStats, const OaVmaStatistics& src)
{
    inoutStats.blockCount += src.blockCount;
    inoutStats.allocationCount += src.allocationCount;
    inoutStats.blockBytes += src.blockBytes;
    inoutStats.allocationBytes += src.allocationBytes;
}

void OaVmaClearDetailedStatistics(OaVmaDetailedStatistics& outStats)
{
    OaVmaClearStatistics(outStats.statistics);
    outStats.unusedRangeCount = 0;
    outStats.allocationSizeMin = VK_WHOLE_SIZE;
    outStats.allocationSizeMax = 0;
    outStats.unusedRangeSizeMin = VK_WHOLE_SIZE;
    outStats.unusedRangeSizeMax = 0;
}

void OaVmaAddDetailedStatisticsAllocation(OaVmaDetailedStatistics& inoutStats, VkDeviceSize size)
{
    inoutStats.statistics.allocationCount++;
    inoutStats.statistics.allocationBytes += size;
    inoutStats.allocationSizeMin = OA_VMA_MIN(inoutStats.allocationSizeMin, size);
    inoutStats.allocationSizeMax = OA_VMA_MAX(inoutStats.allocationSizeMax, size);
}

void OaVmaAddDetailedStatisticsUnusedRange(OaVmaDetailedStatistics& inoutStats, VkDeviceSize size)
{
    inoutStats.unusedRangeCount++;
    inoutStats.unusedRangeSizeMin = OA_VMA_MIN(inoutStats.unusedRangeSizeMin, size);
    inoutStats.unusedRangeSizeMax = OA_VMA_MAX(inoutStats.unusedRangeSizeMax, size);
}

void OaVmaAddDetailedStatistics(OaVmaDetailedStatistics& inoutStats, const OaVmaDetailedStatistics& src)
{
    OaVmaAddStatistics(inoutStats.statistics, src.statistics);
    inoutStats.unusedRangeCount += src.unusedRangeCount;
    inoutStats.allocationSizeMin = OA_VMA_MIN(inoutStats.allocationSizeMin, src.allocationSizeMin);
    inoutStats.allocationSizeMax = OA_VMA_MAX(inoutStats.allocationSizeMax, src.allocationSizeMax);
    inoutStats.unusedRangeSizeMin = OA_VMA_MIN(inoutStats.unusedRangeSizeMin, src.unusedRangeSizeMin);
    inoutStats.unusedRangeSizeMax = OA_VMA_MAX(inoutStats.unusedRangeSizeMax, src.unusedRangeSizeMax);
}

} // namespace

#endif // _OA_VMA_STATISTICS_FUNCTIONS

#ifndef _OA_VMA_MUTEX_LOCK
// Helper RAII class to lock a mutex in constructor and unlock it in destructor (at the end of scope).
struct OaVmaMutexLock
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaMutexLock)
public:
    explicit OaVmaMutexLock(OA_VMA_MUTEX& mutex, bool useMutex = true) :
        m_pMutex(useMutex ? &mutex : OA_VMA_NULL)
    {
        if (m_pMutex) { m_pMutex->Lock(); }
    }
    ~OaVmaMutexLock() {  if (m_pMutex) { m_pMutex->Unlock(); } }

private:
    OA_VMA_MUTEX* m_pMutex;
};

// Helper RAII class to lock a RW mutex in constructor and unlock it in destructor (at the end of scope), for reading.
struct OaVmaMutexLockRead
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaMutexLockRead)
public:
    OaVmaMutexLockRead(OA_VMA_RW_MUTEX& mutex, bool useMutex) :
        m_pMutex(useMutex ? &mutex : OA_VMA_NULL)
    {
        if (m_pMutex) { m_pMutex->LockRead(); }
    }
    ~OaVmaMutexLockRead() { if (m_pMutex) { m_pMutex->UnlockRead(); } }

private:
    OA_VMA_RW_MUTEX* m_pMutex;
};

// Helper RAII class to lock a RW mutex in constructor and unlock it in destructor (at the end of scope), for writing.
struct OaVmaMutexLockWrite
{
    OA_VMA_CLASS_NO_COPY_NO_MOVE(OaVmaMutexLockWrite)
public:
    OaVmaMutexLockWrite(OA_VMA_RW_MUTEX& mutex, bool useMutex)
        : m_pMutex(useMutex ? &mutex : OA_VMA_NULL)
    {
        if (m_pMutex) { m_pMutex->LockWrite(); }
    }
    ~OaVmaMutexLockWrite() { if (m_pMutex) { m_pMutex->UnlockWrite(); } }

private:
    OA_VMA_RW_MUTEX* m_pMutex;
};

#if OA_VMA_DEBUG_GLOBAL_MUTEX
    static OA_VMA_MUTEX gDebugGlobalMutex;
    #define OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK OaVmaMutexLock debugGlobalMutexLock(gDebugGlobalMutex, true);
#else
    #define OA_VMA_DEBUG_GLOBAL_MUTEX_LOCK
#endif
#endif // _OA_VMA_MUTEX_LOCK

#ifndef _OA_VMA_ATOMIC_TRANSACTIONAL_INCREMENT
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

    void Commit() { m_Atomic = OA_VMA_NULL; }
    T Increment(AtomicT* atomic)
    {
        m_Atomic = atomic;
        return m_Atomic->fetch_add(1);
    }

private:
    AtomicT* m_Atomic = OA_VMA_NULL;
};
#endif // _OA_VMA_ATOMIC_TRANSACTIONAL_INCREMENT
