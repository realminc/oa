//
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#pragma once

/** \mainpage Vulkan Memory Allocator

<b>Version 3.4.0-development</b>

Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved. \n
License: MIT \n
See also: [product page on GPUOpen](https://gpuopen.com/vulkan-memory-allocator/),
[repository on GitHub](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)


<b>API documentation divided into groups:</b> [Topics](topics.html)

<b>General documentation chapters:</b>

- \subpage faq
- \subpage quick_start
  - [Project setup](@ref quick_start_project_setup)
  - [Initialization](@ref quick_start_initialization)
  - [Resource allocation](@ref quick_start_resource_allocation)
- \subpage choosing_memory_type
  - [Usage](@ref choosing_memory_type_usage)
  - [Required and preferred flags](@ref choosing_memory_type_required_preferred_flags)
  - [Explicit memory types](@ref choosing_memory_type_explicit_memory_types)
  - [Custom memory pools](@ref choosing_memory_type_custom_memory_pools)
  - [Dedicated allocations](@ref choosing_memory_type_dedicated_allocations)
- \subpage memory_mapping
  - [Copy functions](@ref memory_mapping_copy_functions)
  - [Mapping functions](@ref memory_mapping_mapping_functions)
  - [Persistently mapped memory](@ref memory_mapping_persistently_mapped_memory)
  - [Cache flush and invalidate](@ref memory_mapping_cache_control)
- \subpage staying_within_budget
  - [Querying for budget](@ref staying_within_budget_querying_for_budget)
  - [Controlling memory usage](@ref staying_within_budget_controlling_memory_usage)
- \subpage resource_aliasing
- \subpage custom_memory_pools
  - [Choosing memory type index](@ref custom_memory_pools_MemTypeIndex)
  - [When not to use custom pools](@ref custom_memory_pools_when_not_use)
  - [Linear allocation algorithm](@ref linear_algorithm)
    - [Free-at-once](@ref linear_algorithm_free_at_once)
    - [Stack](@ref linear_algorithm_stack)
    - [Double stack](@ref linear_algorithm_double_stack)
    - [Ring buffer](@ref linear_algorithm_ring_buffer)
- \subpage defragmentation
- \subpage statistics
  - [Numeric statistics](@ref statistics_numeric_statistics)
  - [JSON dump](@ref statistics_json_dump)
- \subpage allocation_annotation
  - [Allocation user data](@ref allocation_user_data)
  - [Allocation names](@ref allocation_names)
- \subpage virtual_allocator
- \subpage debugging_memory_usage
  - [Memory initialization](@ref debugging_memory_usage_initialization)
  - [Margins](@ref debugging_memory_usage_margins)
  - [Corruption detection](@ref debugging_memory_usage_corruption_detection)
  - [Leak detection features](@ref debugging_memory_usage_leak_detection)
- \subpage other_api_interop
  - [Exporting memory](@ref other_api_interop_exporting_memory)
  - [Importing memory](@ref other_api_interop_importing_memory)
- \subpage usage_patterns
    - [GPU-only resource](@ref usage_patterns_gpu_only)
    - [Staging copy for upload](@ref usage_patterns_staging_copy_upload)
    - [Readback](@ref usage_patterns_readback)
    - [Advanced data uploading](@ref usage_patterns_advanced_data_uploading)
    - [Other use cases](@ref usage_patterns_other_use_cases)
- \subpage configuration
  - [Pointers to Vulkan functions](@ref config_Vulkan_functions)
  - [Custom host memory allocator](@ref custom_memory_allocator)
  - [Device memory allocation callbacks](@ref allocation_callbacks)
  - [Device heap memory limit](@ref heap_memory_limit)
- <b>Extension support</b>
    - \subpage vk_khr_dedicated_allocation
    - \subpage enabling_buffer_device_address
    - \subpage vk_ext_memory_priority
    - \subpage vk_amd_device_coherent_memory
- \subpage general_considerations
  - [Thread safety](@ref general_considerations_thread_safety)
  - [Versioning and compatibility](@ref general_considerations_versioning_and_compatibility)
  - [Validation layer warnings](@ref general_considerations_validation_layer_warnings)
  - [Allocation algorithm](@ref general_considerations_allocation_algorithm)
  - [Features not supported](@ref general_considerations_features_not_supported)

\defgroup group_init Library initialization

\brief API elements related to the initialization and management of the entire library, especially #OaVmaAllocator object.

\defgroup group_alloc Memory allocation

\brief API elements related to the allocation, deallocation, and management of Vulkan memory, buffers, images.
Most basic ones being: OaVmaCreateBuffer(), OaVmaCreateImage().

\defgroup group_virtual Virtual allocator

\brief API elements related to the mechanism of \ref virtual_allocator - using the core allocation algorithm
for user-defined purpose without allocating any real GPU memory.

\defgroup group_stats Statistics

\brief API elements that query current status of the allocator, from memory usage, budget, to full dump of the internal state in JSON format.
See documentation chapter: \ref statistics.
*/


#ifdef __cplusplus
extern "C" {
#endif

#if !defined(VULKAN_H_)
#include <vulkan/vulkan.h>
#endif

#define OA_VMA_VERSION (VK_MAKE_VERSION(3, 4, 0))

#if !defined(OA_VMA_VULKAN_VERSION)
    #if defined(VK_VERSION_1_4)
        #define OA_VMA_VULKAN_VERSION 1004000
    #elif defined(VK_VERSION_1_3)
        #define OA_VMA_VULKAN_VERSION 1003000
    #elif defined(VK_VERSION_1_2)
        #define OA_VMA_VULKAN_VERSION 1002000
    #elif defined(VK_VERSION_1_1)
        #define OA_VMA_VULKAN_VERSION 1001000
    #else
        #define OA_VMA_VULKAN_VERSION 1000000
    #endif
#endif

#if defined(__ANDROID__) && defined(VK_NO_PROTOTYPES) && OA_VMA_STATIC_VULKAN_FUNCTIONS
    extern PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
    extern PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr;
    extern PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties;
    extern PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties;
    extern PFN_vkAllocateMemory vkAllocateMemory;
    extern PFN_vkFreeMemory vkFreeMemory;
    extern PFN_vkMapMemory vkMapMemory;
    extern PFN_vkUnmapMemory vkUnmapMemory;
    extern PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRanges;
    extern PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges;
    extern PFN_vkBindBufferMemory vkBindBufferMemory;
    extern PFN_vkBindImageMemory vkBindImageMemory;
    extern PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;
    extern PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements;
    extern PFN_vkCreateBuffer vkCreateBuffer;
    extern PFN_vkDestroyBuffer vkDestroyBuffer;
    extern PFN_vkCreateImage vkCreateImage;
    extern PFN_vkDestroyImage vkDestroyImage;
    extern PFN_vkCmdCopyBuffer vkCmdCopyBuffer;
    #if OA_VMA_VULKAN_VERSION >= 1001000
        extern PFN_vkGetBufferMemoryRequirements2 vkGetBufferMemoryRequirements2;
        extern PFN_vkGetImageMemoryRequirements2 vkGetImageMemoryRequirements2;
        extern PFN_vkBindBufferMemory2 vkBindBufferMemory2;
        extern PFN_vkBindImageMemory2 vkBindImageMemory2;
        extern PFN_vkGetPhysicalDeviceMemoryProperties2 vkGetPhysicalDeviceMemoryProperties2;
    #endif // #if OA_VMA_VULKAN_VERSION >= 1001000
#endif // #if defined(__ANDROID__) && OA_VMA_STATIC_VULKAN_FUNCTIONS && VK_NO_PROTOTYPES

#if !defined(OA_VMA_DEDICATED_ALLOCATION)
    #if VK_KHR_get_memory_requirements2 && VK_KHR_dedicated_allocation
        #define OA_VMA_DEDICATED_ALLOCATION 1
    #else
        #define OA_VMA_DEDICATED_ALLOCATION 0
    #endif
#endif

#if !defined(OA_VMA_BIND_MEMORY2)
    #if VK_KHR_bind_memory2
        #define OA_VMA_BIND_MEMORY2 1
    #else
        #define OA_VMA_BIND_MEMORY2 0
    #endif
#endif

#if !defined(OA_VMA_MEMORY_BUDGET)
    #if VK_EXT_memory_budget && (VK_KHR_get_physical_device_properties2 || OA_VMA_VULKAN_VERSION >= 1001000)
        #define OA_VMA_MEMORY_BUDGET 1
    #else
        #define OA_VMA_MEMORY_BUDGET 0
    #endif
#endif

// Defined to 1 when VK_KHR_buffer_device_address device extension or equivalent core Vulkan 1.2 feature is defined in its headers.
#if !defined(OA_VMA_BUFFER_DEVICE_ADDRESS)
    #if VK_KHR_buffer_device_address || OA_VMA_VULKAN_VERSION >= 1002000
        #define OA_VMA_BUFFER_DEVICE_ADDRESS 1
    #else
        #define OA_VMA_BUFFER_DEVICE_ADDRESS 0
    #endif
#endif

// Defined to 1 when VK_EXT_memory_priority device extension is defined in Vulkan headers.
#if !defined(OA_VMA_MEMORY_PRIORITY)
    #if VK_EXT_memory_priority
        #define OA_VMA_MEMORY_PRIORITY 1
    #else
        #define OA_VMA_MEMORY_PRIORITY 0
    #endif
#endif

// Defined to 1 when VK_KHR_maintenance4 device extension is defined in Vulkan headers.
#if !defined(OA_VMA_KHR_MAINTENANCE4)
    #if VK_KHR_maintenance4
        #define OA_VMA_KHR_MAINTENANCE4 1
    #else
        #define OA_VMA_KHR_MAINTENANCE4 0
    #endif
#endif

// Defined to 1 when VK_KHR_maintenance5 device extension is defined in Vulkan headers.
#if !defined(OA_VMA_KHR_MAINTENANCE5)
    #if VK_KHR_maintenance5
        #define OA_VMA_KHR_MAINTENANCE5 1
    #else
        #define OA_VMA_KHR_MAINTENANCE5 0
    #endif
#endif


// Defined to 1 when VK_KHR_external_memory device extension is defined in Vulkan headers.
#if !defined(OA_VMA_EXTERNAL_MEMORY)
    #if VK_KHR_external_memory
        #define OA_VMA_EXTERNAL_MEMORY 1
    #else
        #define OA_VMA_EXTERNAL_MEMORY 0
    #endif
#endif

// Defined to 1 when VK_KHR_external_memory_win32 device extension is defined in Vulkan headers.
#if !defined(OA_VMA_EXTERNAL_MEMORY_WIN32)
    #if VK_KHR_external_memory_win32
        #define OA_VMA_EXTERNAL_MEMORY_WIN32 1
    #else
        #define OA_VMA_EXTERNAL_MEMORY_WIN32 0
    #endif
#endif

// Define these macros to decorate all public functions with additional code,
// before and after returned type, appropriately. This may be useful for
// exporting the functions when compiling VMA as a separate library. Example:
// #define OA_VMA_CALL_PRE  __declspec(dllexport)
// #define OA_VMA_CALL_POST __cdecl
#ifndef OA_VMA_CALL_PRE
    #define OA_VMA_CALL_PRE
#endif
#ifndef OA_VMA_CALL_POST
    #define OA_VMA_CALL_POST
#endif

// Define this macro to decorate pNext pointers with an attribute specifying the Vulkan
// structure that will be extended via the pNext chain.
#ifndef OA_VMA_EXTENDS_VK_STRUCT
    #define OA_VMA_EXTENDS_VK_STRUCT(vkStruct)
#endif

// Define this macro to decorate pointers with an attribute specifying the
// length of the array they point to if they are not null.
//
// The length may be one of
// - The name of another parameter in the argument list where the pointer is declared
// - The name of another member in the struct where the pointer is declared
// - The name of a member of a struct type, meaning the value of that member in
//   the context of the call. For example
//   OA_VMA_LEN_IF_NOT_NULL("VkPhysicalDeviceMemoryProperties::memoryHeapCount"),
//   this means the number of memory heaps available in the device associated
//   with the OaVmaAllocator being dealt with.
#ifndef OA_VMA_LEN_IF_NOT_NULL
    #define OA_VMA_LEN_IF_NOT_NULL(len)
#endif

// In C/C++, Clang's _Nullable/_Nonnull raise -Wnullability-extension under -Wpedantic.
// Use them only for ObjC/ObjC++; consumers may #define OA_VMA_NULLABLE before include.
// see: https://clang.llvm.org/docs/AttributeReference.html#nullable
#ifndef OA_VMA_NULLABLE
    #if defined(__clang__) && defined(__OBJC__)
        #define OA_VMA_NULLABLE _Nullable
    #else
        #define OA_VMA_NULLABLE
    #endif
#endif

#ifndef OA_VMA_NOT_NULL
    #if defined(__clang__) && defined(__OBJC__)
        #define OA_VMA_NOT_NULL _Nonnull
    #else
        #define OA_VMA_NOT_NULL
    #endif
#endif

// If non-dispatchable handles are represented as pointers then we can give
// then nullability annotations
#ifndef OA_VMA_NOT_NULL_NON_DISPATCHABLE
    #if defined(__LP64__) || defined(_WIN64) || (defined(__x86_64__) && !defined(__ILP32__) ) || defined(_M_X64) || defined(__ia64) || defined (_M_IA64) || defined(__aarch64__) || defined(__powerpc64__)
        #define OA_VMA_NOT_NULL_NON_DISPATCHABLE OA_VMA_NOT_NULL
    #else
        #define OA_VMA_NOT_NULL_NON_DISPATCHABLE
    #endif
#endif

#ifndef OA_VMA_NULLABLE_NON_DISPATCHABLE
    #if defined(__LP64__) || defined(_WIN64) || (defined(__x86_64__) && !defined(__ILP32__) ) || defined(_M_X64) || defined(__ia64) || defined (_M_IA64) || defined(__aarch64__) || defined(__powerpc64__)
        #define OA_VMA_NULLABLE_NON_DISPATCHABLE OA_VMA_NULLABLE
    #else
        #define OA_VMA_NULLABLE_NON_DISPATCHABLE
    #endif
#endif

#ifndef OA_VMA_STATS_STRING_ENABLED
    #define OA_VMA_STATS_STRING_ENABLED 1
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
//    INTERFACE
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Sections for managing code placement in file, only for development purposes e.g. for convenient folding inside an IDE.
#ifndef _OA_VMA_ENUM_DECLARATIONS

/**
\addtogroup group_init
@{
*/

/// Flags for created #OaVmaAllocator.
typedef enum OaVmaAllocatorCreateFlagBits
{
    /** \brief Allocator and all objects created from it will not be synchronized internally, so you must guarantee they are used from only one thread at a time or synchronized externally by you.

    Using this flag may increase performance because internal mutexes are not used.
    */
    OA_VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT = 0x00000001,
    /** \brief Enables usage of VK_KHR_dedicated_allocation extension.

    The flag works only if OaVmaAllocatorCreateInfo::vulkanApiVersion `== VK_API_VERSION_1_0`.
    When it is `VK_API_VERSION_1_1`, the flag is ignored because the extension has been promoted to Vulkan 1.1.

    Using this extension will automatically allocate dedicated blocks of memory for
    some buffers and images instead of suballocating place for them out of bigger
    memory blocks (as if you explicitly used #OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
    flag) when it is recommended by the driver. It may improve performance on some
    GPUs.

    You may set this flag only if you found out that following device extensions are
    supported, you enabled them while creating Vulkan device passed as
    OaVmaAllocatorCreateInfo::device, and you want them to be used internally by this
    library:

    - VK_KHR_get_memory_requirements2 (device extension)
    - VK_KHR_dedicated_allocation (device extension)

    When this flag is set, you can experience following warnings reported by Vulkan
    validation layer. You can ignore them.

    > vkBindBufferMemory(): Binding memory to buffer 0x2d but vkGetBufferMemoryRequirements() has not been called on that buffer.
    */
    OA_VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT = 0x00000002,
    /**
    Enables usage of VK_KHR_bind_memory2 extension.

    The flag works only if OaVmaAllocatorCreateInfo::vulkanApiVersion `== VK_API_VERSION_1_0`.
    When it is `VK_API_VERSION_1_1`, the flag is ignored because the extension has been promoted to Vulkan 1.1.

    You may set this flag only if you found out that this device extension is supported,
    you enabled it while creating Vulkan device passed as OaVmaAllocatorCreateInfo::device,
    and you want it to be used internally by this library.

    The extension provides functions `vkBindBufferMemory2KHR` and `vkBindImageMemory2KHR`,
    which allow to pass a chain of `pNext` structures while binding.
    This flag is required if you use `pNext` parameter in OaVmaBindBufferMemory2() or OaVmaBindImageMemory2().
    */
    OA_VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT = 0x00000004,
    /**
    Enables usage of VK_EXT_memory_budget extension.

    You may set this flag only if you found out that this device extension is supported,
    you enabled it while creating Vulkan device passed as OaVmaAllocatorCreateInfo::device,
    and you want it to be used internally by this library, along with another instance extension
    VK_KHR_get_physical_device_properties2, which is required by it (or Vulkan 1.1, where this extension is promoted).

    The extension provides query for current memory usage and budget, which will probably
    be more accurate than an estimation used by the library otherwise.
    */
    OA_VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT = 0x00000008,
    /**
    Enables usage of VK_AMD_device_coherent_memory extension.

    You may set this flag only if you:

    - found out that this device extension is supported and enabled it while creating Vulkan device passed as OaVmaAllocatorCreateInfo::device,
    - checked that `VkPhysicalDeviceCoherentMemoryFeaturesAMD::deviceCoherentMemory` is true and set it while creating the Vulkan device,
    - want it to be used internally by this library.

    The extension and accompanying device feature provide access to memory types with
    `VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD` and `VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD` flags.
    They are useful mostly for writing breadcrumb markers - a common method for debugging GPU crash/hang/TDR.

    When the extension is not enabled, such memory types are still enumerated, but their usage is illegal.
    To protect from this error, if you don't create the allocator with this flag, it will refuse to allocate any memory or create a custom pool in such memory type,
    returning `VK_ERROR_FEATURE_NOT_PRESENT`.
    */
    OA_VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT = 0x00000010,
    /**
    Enables usage of "buffer device address" feature, which allows you to use function
    `vkGetBufferDeviceAddress*` to get raw GPU pointer to a buffer and pass it for usage inside a shader.

    You may set this flag only if you:

    1. (For Vulkan version < 1.2) Found as available and enabled device extension
    VK_KHR_buffer_device_address.
    This extension is promoted to core Vulkan 1.2.
    2. Found as available and enabled device feature `VkPhysicalDeviceBufferDeviceAddressFeatures::bufferDeviceAddress`.

    When this flag is set, you can create buffers with `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` using VMA.
    The library automatically adds `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` to
    allocated memory blocks wherever it might be needed.

    For more information, see documentation chapter \ref enabling_buffer_device_address.
    */
    OA_VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT = 0x00000020,
    /**
    Enables usage of VK_EXT_memory_priority extension in the library.

    You may set this flag only if you found available and enabled this device extension,
    along with `VkPhysicalDeviceMemoryPriorityFeaturesEXT::memoryPriority == VK_TRUE`,
    while creating Vulkan device passed as OaVmaAllocatorCreateInfo::device.

    When this flag is used, OaVmaAllocationCreateInfo::priority and OaVmaPoolCreateInfo::priority
    are used to set priorities of allocated Vulkan memory. Without it, these variables are ignored.

    A priority must be a floating-point value between 0 and 1, indicating the priority of the allocation relative to other memory allocations.
    Larger values are higher priority. The granularity of the priorities is implementation-dependent.
    It is automatically passed to every call to `vkAllocateMemory` done by the library using structure `VkMemoryPriorityAllocateInfoEXT`.
    The value to be used for default priority is 0.5.
    For more details, see the documentation of the VK_EXT_memory_priority extension.
    */
    OA_VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT = 0x00000040,
    /**
    Enables usage of VK_KHR_maintenance4 extension in the library.

    You may set this flag only if you found available and enabled this device extension,
    while creating Vulkan device passed as OaVmaAllocatorCreateInfo::device.
    */
    OA_VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT = 0x00000080,
    /**
    Enables usage of VK_KHR_maintenance5 extension in the library.

    You should set this flag if you found available and enabled this device extension,
    while creating Vulkan device passed as OaVmaAllocatorCreateInfo::device.
    */
    OA_VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT = 0x00000100,

    /**
    Enables usage of VK_KHR_external_memory_win32 extension in the library.

    You should set this flag if you found available and enabled this device extension,
    while creating Vulkan device passed as OaVmaAllocatorCreateInfo::device.
    For more information, see \ref other_api_interop.
    */
    OA_VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT = 0x00000200,

    OA_VMA_ALLOCATOR_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} OaVmaAllocatorCreateFlagBits;
/// See #OaVmaAllocatorCreateFlagBits.
typedef VkFlags OaVmaAllocatorCreateFlags;

/** @} */

/**
\addtogroup group_alloc
@{
*/

/// \brief Intended usage of the allocated memory.
typedef enum OaVmaMemoryUsage
{
    /** No intended memory usage specified.
    Use other members of OaVmaAllocationCreateInfo to specify your requirements.
    */
    OA_VMA_MEMORY_USAGE_UNKNOWN = 0,
    /**
    \deprecated Obsolete, preserved for backward compatibility.
    Prefers `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.
    */
    OA_VMA_MEMORY_USAGE_GPU_ONLY = 1,
    /**
    \deprecated Obsolete, preserved for backward compatibility.
    Guarantees `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` and `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`.
    */
    OA_VMA_MEMORY_USAGE_CPU_ONLY = 2,
    /**
    \deprecated Obsolete, preserved for backward compatibility.
    Guarantees `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`, prefers `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.
    */
    OA_VMA_MEMORY_USAGE_CPU_TO_GPU = 3,
    /**
    \deprecated Obsolete, preserved for backward compatibility.
    Guarantees `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`, prefers `VK_MEMORY_PROPERTY_HOST_CACHED_BIT`.
    */
    OA_VMA_MEMORY_USAGE_GPU_TO_CPU = 4,
    /**
    \deprecated Obsolete, preserved for backward compatibility.
    Prefers not `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.
    */
    OA_VMA_MEMORY_USAGE_CPU_COPY = 5,
    /**
    Lazily allocated GPU memory having `VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT`.
    Exists mostly on mobile platforms. Using it on desktop PC or other GPUs with no such memory type present will fail the allocation.

    Usage: Memory for transient attachment images (color attachments, depth attachments etc.), created with `VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT`.

    Allocations with this usage are always created as dedicated - it implies #OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
    */
    OA_VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED = 6,
    /**
    Selects best memory type automatically.
    This flag is recommended for most common use cases.

    When using this flag, if you want to map the allocation (using OaVmaMapMemory() or #OA_VMA_ALLOCATION_CREATE_MAPPED_BIT),
    you must pass one of the flags: #OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
    in OaVmaAllocationCreateInfo::flags.

    It can be used only with functions that let the library know `VkBufferCreateInfo` or `VkImageCreateInfo`, e.g.
    OaVmaCreateBuffer(), OaVmaCreateImage(), OaVmaFindMemoryTypeIndexForBufferInfo(), OaVmaFindMemoryTypeIndexForImageInfo()
    and not with generic memory allocation functions.
    */
    OA_VMA_MEMORY_USAGE_AUTO = 7,
    /**
    Selects best memory type automatically with preference for GPU (device) memory.

    When using this flag, if you want to map the allocation (using OaVmaMapMemory() or #OA_VMA_ALLOCATION_CREATE_MAPPED_BIT),
    you must pass one of the flags: #OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
    in OaVmaAllocationCreateInfo::flags.

    It can be used only with functions that let the library know `VkBufferCreateInfo` or `VkImageCreateInfo`, e.g.
    OaVmaCreateBuffer(), OaVmaCreateImage(), OaVmaFindMemoryTypeIndexForBufferInfo(), OaVmaFindMemoryTypeIndexForImageInfo()
    and not with generic memory allocation functions.
    */
    OA_VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE = 8,
    /**
    Selects best memory type automatically with preference for CPU (host) memory.

    When using this flag, if you want to map the allocation (using OaVmaMapMemory() or #OA_VMA_ALLOCATION_CREATE_MAPPED_BIT),
    you must pass one of the flags: #OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
    in OaVmaAllocationCreateInfo::flags.

    It can be used only with functions that let the library know `VkBufferCreateInfo` or `VkImageCreateInfo`, e.g.
    OaVmaCreateBuffer(), OaVmaCreateImage(), OaVmaFindMemoryTypeIndexForBufferInfo(), OaVmaFindMemoryTypeIndexForImageInfo()
    and not with generic memory allocation functions.
    */
    OA_VMA_MEMORY_USAGE_AUTO_PREFER_HOST = 9,

    OA_VMA_MEMORY_USAGE_MAX_ENUM = 0x7FFFFFFF
} OaVmaMemoryUsage;

/// Flags to be passed as OaVmaAllocationCreateInfo::flags.
typedef enum OaVmaAllocationCreateFlagBits
{
    /** \brief Set this flag if the allocation should have its own memory block.

    Use it for special, big resources, like fullscreen images used as attachments.

    If you use this flag while creating a buffer or an image, `VkMemoryDedicatedAllocateInfo`
    structure is applied if possible.
    */
    OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT = 0x00000001,

    /** \brief Set this flag to only try to allocate from existing `VkDeviceMemory` blocks and never create new such block.

    If new allocation cannot be placed in any of the existing blocks, allocation
    fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY` error.

    You should not use #OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT and
    #OA_VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT at the same time. It makes no sense.
    */
    OA_VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT = 0x00000002,
    /** \brief Set this flag to use a memory that will be persistently mapped and retrieve pointer to it.

    Pointer to mapped memory will be returned through OaVmaAllocationInfo::pMappedData.

    It is valid to use this flag for allocation made from memory type that is not
    `HOST_VISIBLE`. This flag is then ignored and memory is not mapped. This is
    useful if you need an allocation that is efficient to use on GPU
    (`DEVICE_LOCAL`) and still want to map it directly if possible on platforms that
    support it (e.g. Intel GPU).
    */
    OA_VMA_ALLOCATION_CREATE_MAPPED_BIT = 0x00000004,
    /** \deprecated Preserved for backward compatibility. Consider using OaVmaSetAllocationName() instead.

    Set this flag to treat OaVmaAllocationCreateInfo::pUserData as pointer to a
    null-terminated string. Instead of copying pointer value, a local copy of the
    string is made and stored in allocation's `pName`. The string is automatically
    freed together with the allocation. It is also used in OaVmaBuildStatsString().
    */
    OA_VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT = 0x00000020,
    /** Allocation will be created from upper stack in a double stack pool.

    This flag is only allowed for custom pools created with #OA_VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT flag.
    */
    OA_VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT = 0x00000040,
    /** Create both buffer/image and allocation, but don't bind them together.
    It is useful when you want to bind yourself to do some more advanced binding, e.g. using some extensions.
    The flag is meaningful only with functions that bind by default: OaVmaCreateBuffer(), OaVmaCreateImage().
    Otherwise it is ignored.

    If you want to make sure the new buffer/image is not tied to the new memory allocation
    through `VkMemoryDedicatedAllocateInfoKHR` structure in case the allocation ends up in its own memory block,
    use also flag #OA_VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT.
    */
    OA_VMA_ALLOCATION_CREATE_DONT_BIND_BIT = 0x00000080,
    /** Create allocation only if additional device memory required for it, if any, won't exceed
    memory budget. Otherwise return `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
    */
    OA_VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT = 0x00000100,
    /** \brief Set this flag if the allocated memory will have aliasing resources.

    Usage of this flag prevents supplying `VkMemoryDedicatedAllocateInfoKHR` when #OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT is specified.
    Otherwise created dedicated memory will not be suitable for aliasing resources, resulting in Vulkan Validation Layer errors.
    */
    OA_VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT = 0x00000200,
    /**
    Requests possibility to map the allocation (using OaVmaMapMemory() or #OA_VMA_ALLOCATION_CREATE_MAPPED_BIT).

    - If you use #OA_VMA_MEMORY_USAGE_AUTO or other `OA_VMA_MEMORY_USAGE_AUTO*` value,
      you must use this flag to be able to map the allocation. Otherwise, mapping is incorrect.
    - If you use other value of #OaVmaMemoryUsage, this flag is ignored and mapping is always possible in memory types that are `HOST_VISIBLE`.
      This includes allocations created in \ref custom_memory_pools.

    Declares that mapped memory will only be written sequentially, e.g. using `memcpy()` or a loop writing number-by-number,
    never read or accessed randomly, so a memory type can be selected that is uncached and write-combined.

    \warning Violating this declaration may work correctly, but will likely be very slow.
    Watch out for implicit reads introduced by doing e.g. `pMappedData[i] += x;`
    Better prepare your data in a local variable and `memcpy()` it to the mapped pointer all at once.
    */
    OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT = 0x00000400,
    /**
    Requests possibility to map the allocation (using OaVmaMapMemory() or #OA_VMA_ALLOCATION_CREATE_MAPPED_BIT).

    - If you use #OA_VMA_MEMORY_USAGE_AUTO or other `OA_VMA_MEMORY_USAGE_AUTO*` value,
      you must use this flag to be able to map the allocation. Otherwise, mapping is incorrect.
    - If you use other value of #OaVmaMemoryUsage, this flag is ignored and mapping is always possible in memory types that are `HOST_VISIBLE`.
      This includes allocations created in \ref custom_memory_pools.

    Declares that mapped memory can be read, written, and accessed in random order,
    so a `HOST_CACHED` memory type is preferred.
    */
    OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT = 0x00000800,
    /**
    Together with #OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
    it says that despite request for host access, a not-`HOST_VISIBLE` memory type can be selected
    if it may improve performance.

    By using this flag, you declare that you will check if the allocation ended up in a `HOST_VISIBLE` memory type
    (e.g. using OaVmaGetAllocationMemoryProperties()) and if not, you will create some "staging" buffer and
    issue an explicit transfer to write/read your data.
    To prepare for this possibility, don't forget to add appropriate flags like
    `VK_BUFFER_USAGE_TRANSFER_DST_BIT`, `VK_BUFFER_USAGE_TRANSFER_SRC_BIT` to the parameters of created buffer or image.
    */
    OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT = 0x00001000,
    /** Allocation strategy that chooses smallest possible free range for the allocation
    to minimize memory usage and fragmentation, possibly at the expense of allocation time.
    */
    OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT = 0x00010000,
    /** Allocation strategy that chooses first suitable free range for the allocation -
    not necessarily in terms of the smallest offset but the one that is easiest and fastest to find
    to minimize allocation time, possibly at the expense of allocation quality.
    */
    OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT = 0x00020000,
    /** Allocation strategy that chooses always the lowest offset in available space.
    This is not the most efficient strategy but achieves highly packed data.
    Used internally by defragmentation, not recommended in typical usage.
    */
    OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT  = 0x00040000,
    /** Alias to #OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT.
    */
    OA_VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT = OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT,
    /** Alias to #OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT.
    */
    OA_VMA_ALLOCATION_CREATE_STRATEGY_FIRST_FIT_BIT = OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT,
    /** A bit mask to extract only `STRATEGY` bits from entire set of flags.
    */
    OA_VMA_ALLOCATION_CREATE_STRATEGY_MASK =
        OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT |
        OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT |
        OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT,

    OA_VMA_ALLOCATION_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} OaVmaAllocationCreateFlagBits;
/// See #OaVmaAllocationCreateFlagBits.
typedef VkFlags OaVmaAllocationCreateFlags;

/// Flags to be passed as OaVmaPoolCreateInfo::flags.
typedef enum OaVmaPoolCreateFlagBits
{
    /** \brief Use this flag if you always allocate only buffers and linear images or only optimal images out of this pool and so Buffer-Image Granularity can be ignored.

    This is an optional optimization flag.

    If you always allocate using OaVmaCreateBuffer(), OaVmaCreateImage(),
    OaVmaAllocateMemoryForBuffer(), then you don't need to use it because allocator
    knows exact type of your allocations so it can handle Buffer-Image Granularity
    in the optimal way.

    If you also allocate using OaVmaAllocateMemoryForImage() or OaVmaAllocateMemory(),
    exact type of such allocations is not known, so allocator must be conservative
    in handling Buffer-Image Granularity, which can lead to suboptimal allocation
    (wasted memory). In that case, if you can make sure you always allocate only
    buffers and linear images or only optimal images out of this pool, use this flag
    to make allocator disregard Buffer-Image Granularity and so make allocations
    faster and more optimal.
    */
    OA_VMA_POOL_CREATE_IGNORE_BUFFER_IMAGE_GRANULARITY_BIT = 0x00000002,

    /** \brief Enables alternative, linear allocation algorithm in this pool.

    Specify this flag to enable linear allocation algorithm, which always creates
    new allocations after last one and doesn't reuse space from allocations freed in
    between. It trades memory consumption for simplified algorithm and data
    structure, which has better performance and uses less memory for metadata.

    By using this flag, you can achieve behavior of free-at-once, stack,
    ring buffer, and double stack.
    For details, see documentation chapter \ref linear_algorithm.
    */
    OA_VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT = 0x00000004,

    /** Bit mask to extract only `ALGORITHM` bits from entire set of flags.
    */
    OA_VMA_POOL_CREATE_ALGORITHM_MASK =
        OA_VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT,

    OA_VMA_POOL_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} OaVmaPoolCreateFlagBits;
/// Flags to be passed as OaVmaPoolCreateInfo::flags. See #OaVmaPoolCreateFlagBits.
typedef VkFlags OaVmaPoolCreateFlags;

/// Flags to be passed as OaVmaDefragmentationInfo::flags.
typedef enum OaVmaDefragmentationFlagBits
{
    /* \brief Use simple but fast algorithm for defragmentation.
    May not achieve best results but will require least time to compute and least allocations to copy.
    */
    OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT = 0x1,
    /* \brief Default defragmentation algorithm, applied also when no `ALGORITHM` flag is specified.
    Offers a balance between defragmentation quality and the amount of allocations and bytes that need to be moved.
    */
    OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT = 0x2,
    /* \brief Perform full defragmentation of memory.
    Can result in notably more time to compute and allocations to copy, but will achieve best memory packing.
    */
    OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL_BIT = 0x4,
    /** \brief Use the most roboust algorithm at the cost of time to compute and number of copies to make.
    Only available when bufferImageGranularity is greater than 1, since it aims to reduce
    alignment issues between different types of resources.
    Otherwise falls back to same behavior as #OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL_BIT.
    */
    OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT = 0x8,

    /// A bit mask to extract only `ALGORITHM` bits from entire set of flags.
    OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_MASK =
        OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT |
        OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT |
        OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL_BIT |
        OA_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT,

    OA_VMA_DEFRAGMENTATION_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} OaVmaDefragmentationFlagBits;
/// See #OaVmaDefragmentationFlagBits.
typedef VkFlags OaVmaDefragmentationFlags;

/// Operation performed on single defragmentation move. See structure #OaVmaDefragmentationMove.
typedef enum OaVmaDefragmentationMoveOperation
{
    /// Buffer/image has been recreated at `dstTmpAllocation`, data has been copied, old buffer/image has been destroyed. `srcAllocation` should be changed to point to the new place. This is the default value set by OaVmaBeginDefragmentationPass().
    OA_VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY = 0,
    /// Set this value if you cannot move the allocation. New place reserved at `dstTmpAllocation` will be freed. `srcAllocation` will remain unchanged.
    OA_VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE = 1,
    /// Set this value if you decide to abandon the allocation and you destroyed the buffer/image. New place reserved at `dstTmpAllocation` will be freed, along with `srcAllocation`, which will be destroyed.
    OA_VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY = 2,
} OaVmaDefragmentationMoveOperation;

/** @} */

/**
\addtogroup group_virtual
@{
*/

/// Flags to be passed as OaVmaVirtualBlockCreateInfo::flags.
typedef enum OaVmaVirtualBlockCreateFlagBits
{
    /** \brief Enables alternative, linear allocation algorithm in this virtual block.

    Specify this flag to enable linear allocation algorithm, which always creates
    new allocations after last one and doesn't reuse space from allocations freed in
    between. It trades memory consumption for simplified algorithm and data
    structure, which has better performance and uses less memory for metadata.

    By using this flag, you can achieve behavior of free-at-once, stack,
    ring buffer, and double stack.
    For details, see documentation chapter \ref linear_algorithm.
    */
    OA_VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT = 0x00000001,

    /** \brief Bit mask to extract only `ALGORITHM` bits from entire set of flags.
    */
    OA_VMA_VIRTUAL_BLOCK_CREATE_ALGORITHM_MASK =
        OA_VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT,

    OA_VMA_VIRTUAL_BLOCK_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} OaVmaVirtualBlockCreateFlagBits;
/// Flags to be passed as OaVmaVirtualBlockCreateInfo::flags. See #OaVmaVirtualBlockCreateFlagBits.
typedef VkFlags OaVmaVirtualBlockCreateFlags;

/// Flags to be passed as OaVmaVirtualAllocationCreateInfo::flags.
typedef enum OaVmaVirtualAllocationCreateFlagBits
{
    /** \brief Allocation will be created from upper stack in a double stack pool.

    This flag is only allowed for virtual blocks created with #OA_VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT flag.
    */
    OA_VMA_VIRTUAL_ALLOCATION_CREATE_UPPER_ADDRESS_BIT = OA_VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT,
    /** \brief Allocation strategy that tries to minimize memory usage.
    */
    OA_VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT = OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT,
    /** \brief Allocation strategy that tries to minimize allocation time.
    */
    OA_VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT = OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT,
    /** Allocation strategy that chooses always the lowest offset in available space.
    This is not the most efficient strategy but achieves highly packed data.
    */
    OA_VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT = OA_VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT,
    /** \brief A bit mask to extract only `STRATEGY` bits from entire set of flags.

    These strategy flags are binary compatible with equivalent flags in #OaVmaAllocationCreateFlagBits.
    */
    OA_VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MASK = OA_VMA_ALLOCATION_CREATE_STRATEGY_MASK,

    OA_VMA_VIRTUAL_ALLOCATION_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} OaVmaVirtualAllocationCreateFlagBits;
/// Flags to be passed as OaVmaVirtualAllocationCreateInfo::flags. See #OaVmaVirtualAllocationCreateFlagBits.
typedef VkFlags OaVmaVirtualAllocationCreateFlags;

/** @} */

#endif // _OA_VMA_ENUM_DECLARATIONS

#ifndef _OA_VMA_DATA_TYPES_DECLARATIONS

/**
\addtogroup group_init
@{ */

/** \struct OaVmaAllocator
\brief Represents main object of this library initialized.

Fill structure #OaVmaAllocatorCreateInfo and call function OaVmaCreateAllocator() to create it.
Call function OaVmaDestroyAllocator() to destroy it.

It is recommended to create just one object of this type per `VkDevice` object,
right after Vulkan is initialized and keep it alive until before Vulkan device is destroyed.
*/
VK_DEFINE_HANDLE(OaVmaAllocator)

/** @} */

/**
\addtogroup group_alloc
@{
*/

/** \struct OaVmaPool
\brief Represents custom memory pool

Fill structure OaVmaPoolCreateInfo and call function OaVmaCreatePool() to create it.
Call function OaVmaDestroyPool() to destroy it.

For more information see [Custom memory pools](@ref choosing_memory_type_custom_memory_pools).
*/
VK_DEFINE_HANDLE(OaVmaPool)

/** \struct OaVmaAllocation
\brief Represents single memory allocation.

It may be either dedicated block of `VkDeviceMemory` or a specific region of a bigger block of this type
plus unique offset.

There are multiple ways to create such object.
You need to fill structure OaVmaAllocationCreateInfo.
For more information see [Choosing memory type](@ref choosing_memory_type).

Although the library provides convenience functions that create Vulkan buffer or image,
allocate memory for it and bind them together,
binding of the allocation to a buffer or an image is out of scope of the allocation itself.
Allocation object can exist without buffer/image bound,
binding can be done manually by the user, and destruction of it can be done
independently of destruction of the allocation.

The object also remembers its size and some other information.
To retrieve this information, use function OaVmaGetAllocationInfo() and inspect
returned structure OaVmaAllocationInfo.
*/
VK_DEFINE_HANDLE(OaVmaAllocation)

/** \struct OaVmaDefragmentationContext
\brief An opaque object that represents started defragmentation process.

Fill structure #OaVmaDefragmentationInfo and call function OaVmaBeginDefragmentation() to create it.
Call function OaVmaEndDefragmentation() to destroy it.
*/
VK_DEFINE_HANDLE(OaVmaDefragmentationContext)

/** @} */

/**
\addtogroup group_virtual
@{
*/

/** \struct OaVmaVirtualAllocation
\brief Represents single memory allocation done inside OaVmaVirtualBlock.

Use it as a unique identifier to virtual allocation within the single block.

Use value `VK_NULL_HANDLE` to represent a null/invalid allocation.
*/
VK_DEFINE_NON_DISPATCHABLE_HANDLE(OaVmaVirtualAllocation)

/** @} */

/**
\addtogroup group_virtual
@{
*/

/** \struct OaVmaVirtualBlock
\brief Handle to a virtual block object that allows to use core allocation algorithm without allocating any real GPU memory.

Fill in #OaVmaVirtualBlockCreateInfo structure and use OaVmaCreateVirtualBlock() to create it. Use OaVmaDestroyVirtualBlock() to destroy it.
For more information, see documentation chapter \ref virtual_allocator.

This object is not thread-safe - should not be used from multiple threads simultaneously, must be synchronized externally.
*/
VK_DEFINE_HANDLE(OaVmaVirtualBlock)

/** @} */

/**
\addtogroup group_init
@{
*/

/// Callback function called after successful vkAllocateMemory.
typedef void (VKAPI_PTR* PFN_OaVmaAllocateDeviceMemoryFunction)(
    OaVmaAllocator OA_VMA_NOT_NULL                    allocator,
    uint32_t                                     memoryType,
    VkDeviceMemory OA_VMA_NOT_NULL_NON_DISPATCHABLE memory,
    VkDeviceSize                                 size,
    void* OA_VMA_NULLABLE                           pUserData);

/// Callback function called before vkFreeMemory.
typedef void (VKAPI_PTR* PFN_OaVmaFreeDeviceMemoryFunction)(
    OaVmaAllocator OA_VMA_NOT_NULL                    allocator,
    uint32_t                                     memoryType,
    VkDeviceMemory OA_VMA_NOT_NULL_NON_DISPATCHABLE memory,
    VkDeviceSize                                 size,
    void* OA_VMA_NULLABLE                           pUserData);

/** \brief Set of callbacks that the library will call for `vkAllocateMemory` and `vkFreeMemory`.

Provided for informative purpose, e.g. to gather statistics about number of
allocations or total amount of memory allocated in Vulkan.

Used in OaVmaAllocatorCreateInfo::pDeviceMemoryCallbacks.
*/
typedef struct OaVmaDeviceMemoryCallbacks
{
    /// Optional, can be null.
    PFN_OaVmaAllocateDeviceMemoryFunction OA_VMA_NULLABLE pfnAllocate;
    /// Optional, can be null.
    PFN_OaVmaFreeDeviceMemoryFunction OA_VMA_NULLABLE pfnFree;
    /// Optional, can be null.
    void* OA_VMA_NULLABLE pUserData;
} OaVmaDeviceMemoryCallbacks;

/** \brief Pointers to some Vulkan functions - a subset used by the library.

Used in OaVmaAllocatorCreateInfo::pVulkanFunctions.
*/
typedef struct OaVmaVulkanFunctions
{
    /// Required when using OA_VMA_DYNAMIC_VULKAN_FUNCTIONS.
    PFN_vkGetInstanceProcAddr OA_VMA_NULLABLE vkGetInstanceProcAddr;
    /// Required when using OA_VMA_DYNAMIC_VULKAN_FUNCTIONS.
    PFN_vkGetDeviceProcAddr OA_VMA_NULLABLE vkGetDeviceProcAddr;
    PFN_vkGetPhysicalDeviceProperties OA_VMA_NULLABLE vkGetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties OA_VMA_NULLABLE vkGetPhysicalDeviceMemoryProperties;
    PFN_vkAllocateMemory OA_VMA_NULLABLE vkAllocateMemory;
    PFN_vkFreeMemory OA_VMA_NULLABLE vkFreeMemory;
    PFN_vkMapMemory OA_VMA_NULLABLE vkMapMemory;
    PFN_vkUnmapMemory OA_VMA_NULLABLE vkUnmapMemory;
    PFN_vkFlushMappedMemoryRanges OA_VMA_NULLABLE vkFlushMappedMemoryRanges;
    PFN_vkInvalidateMappedMemoryRanges OA_VMA_NULLABLE vkInvalidateMappedMemoryRanges;
    PFN_vkBindBufferMemory OA_VMA_NULLABLE vkBindBufferMemory;
    PFN_vkBindImageMemory OA_VMA_NULLABLE vkBindImageMemory;
    PFN_vkGetBufferMemoryRequirements OA_VMA_NULLABLE vkGetBufferMemoryRequirements;
    PFN_vkGetImageMemoryRequirements OA_VMA_NULLABLE vkGetImageMemoryRequirements;
    PFN_vkCreateBuffer OA_VMA_NULLABLE vkCreateBuffer;
    PFN_vkDestroyBuffer OA_VMA_NULLABLE vkDestroyBuffer;
    PFN_vkCreateImage OA_VMA_NULLABLE vkCreateImage;
    PFN_vkDestroyImage OA_VMA_NULLABLE vkDestroyImage;
    PFN_vkCmdCopyBuffer OA_VMA_NULLABLE vkCmdCopyBuffer;
#if OA_VMA_DEDICATED_ALLOCATION || OA_VMA_VULKAN_VERSION >= 1001000
    /// Fetch "vkGetBufferMemoryRequirements2" on Vulkan >= 1.1, fetch "vkGetBufferMemoryRequirements2KHR" when using VK_KHR_dedicated_allocation extension.
    PFN_vkGetBufferMemoryRequirements2KHR OA_VMA_NULLABLE vkGetBufferMemoryRequirements2KHR;
    /// Fetch "vkGetImageMemoryRequirements2" on Vulkan >= 1.1, fetch "vkGetImageMemoryRequirements2KHR" when using VK_KHR_dedicated_allocation extension.
    PFN_vkGetImageMemoryRequirements2KHR OA_VMA_NULLABLE vkGetImageMemoryRequirements2KHR;
#endif
#if OA_VMA_BIND_MEMORY2 || OA_VMA_VULKAN_VERSION >= 1001000
    /// Fetch "vkBindBufferMemory2" on Vulkan >= 1.1, fetch "vkBindBufferMemory2KHR" when using VK_KHR_bind_memory2 extension.
    PFN_vkBindBufferMemory2KHR OA_VMA_NULLABLE vkBindBufferMemory2KHR;
    /// Fetch "vkBindImageMemory2" on Vulkan >= 1.1, fetch "vkBindImageMemory2KHR" when using VK_KHR_bind_memory2 extension.
    PFN_vkBindImageMemory2KHR OA_VMA_NULLABLE vkBindImageMemory2KHR;
#endif
#if OA_VMA_MEMORY_BUDGET || OA_VMA_VULKAN_VERSION >= 1001000
    /// Fetch from "vkGetPhysicalDeviceMemoryProperties2" on Vulkan >= 1.1, but you can also fetch it from "vkGetPhysicalDeviceMemoryProperties2KHR" if you enabled extension VK_KHR_get_physical_device_properties2.
    PFN_vkGetPhysicalDeviceMemoryProperties2KHR OA_VMA_NULLABLE vkGetPhysicalDeviceMemoryProperties2KHR;
#endif
#if OA_VMA_KHR_MAINTENANCE4 || OA_VMA_VULKAN_VERSION >= 1003000
    /// Fetch from "vkGetDeviceBufferMemoryRequirements" on Vulkan >= 1.3, but you can also fetch it from "vkGetDeviceBufferMemoryRequirementsKHR" if you enabled extension VK_KHR_maintenance4.
    PFN_vkGetDeviceBufferMemoryRequirementsKHR OA_VMA_NULLABLE vkGetDeviceBufferMemoryRequirements;
    /// Fetch from "vkGetDeviceImageMemoryRequirements" on Vulkan >= 1.3, but you can also fetch it from "vkGetDeviceImageMemoryRequirementsKHR" if you enabled extension VK_KHR_maintenance4.
    PFN_vkGetDeviceImageMemoryRequirementsKHR OA_VMA_NULLABLE vkGetDeviceImageMemoryRequirements;
#endif
#if OA_VMA_EXTERNAL_MEMORY_WIN32
    PFN_vkGetMemoryWin32HandleKHR OA_VMA_NULLABLE vkGetMemoryWin32HandleKHR;
#else
    void* OA_VMA_NULLABLE vkGetMemoryWin32HandleKHR;
#endif
} OaVmaVulkanFunctions;

/// Description of a Allocator to be created.
typedef struct OaVmaAllocatorCreateInfo
{
    /// Flags for created allocator. Use #OaVmaAllocatorCreateFlagBits enum.
    OaVmaAllocatorCreateFlags flags;
    /// Vulkan physical device.
    /** It must be valid throughout whole lifetime of created allocator. */
    VkPhysicalDevice OA_VMA_NOT_NULL physicalDevice;
    /// Vulkan device.
    /** It must be valid throughout whole lifetime of created allocator. */
    VkDevice OA_VMA_NOT_NULL device;
    /// Preferred size of a single `VkDeviceMemory` block to be allocated from large heaps > 1 GiB. Optional.
    /** Set to 0 to use default, which is currently 256 MiB. */
    VkDeviceSize preferredLargeHeapBlockSize;
    /// Custom CPU memory allocation callbacks. Optional.
    /** Optional, can be null. When specified, will also be used for all CPU-side memory allocations. */
    const VkAllocationCallbacks* OA_VMA_NULLABLE pAllocationCallbacks;
    /// Informative callbacks for `vkAllocateMemory`, `vkFreeMemory`. Optional.
    /** Optional, can be null. */
    const OaVmaDeviceMemoryCallbacks* OA_VMA_NULLABLE pDeviceMemoryCallbacks;
    /** \brief Either null or a pointer to an array of limits on maximum number of bytes that can be allocated out of particular Vulkan memory heap.

    If not NULL, it must be a pointer to an array of
    `VkPhysicalDeviceMemoryProperties::memoryHeapCount` elements, defining limit on
    maximum number of bytes that can be allocated out of particular Vulkan memory
    heap.

    Any of the elements may be equal to `VK_WHOLE_SIZE`, which means no limit on that
    heap. This is also the default in case of `pHeapSizeLimit` = NULL.

    If there is a limit defined for a heap:

    - If user tries to allocate more memory from that heap using this allocator,
      the allocation fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
    - If the limit is smaller than heap size reported in `VkMemoryHeap::size`, the
      value of this limit will be reported instead when using OaVmaGetMemoryProperties().

    Warning! Using this feature may not be equivalent to installing a GPU with
    smaller amount of memory, because graphics driver doesn't necessary fail new
    allocations with `VK_ERROR_OUT_OF_DEVICE_MEMORY` result when memory capacity is
    exceeded. It may return success and just silently migrate some device memory
    blocks to system RAM. This driver behavior can also be controlled using
    VK_AMD_memory_overallocation_behavior extension.
    */
    const VkDeviceSize* OA_VMA_NULLABLE OA_VMA_LEN_IF_NOT_NULL("VkPhysicalDeviceMemoryProperties::memoryHeapCount") pHeapSizeLimit;

    /** \brief Pointers to Vulkan functions. Can be null.

    For details see [Pointers to Vulkan functions](@ref config_Vulkan_functions).
    */
    const OaVmaVulkanFunctions* OA_VMA_NULLABLE pVulkanFunctions;
    /** \brief Handle to Vulkan instance object.

    Starting from version 3.0.0 this member is no longer optional, it must be set!
    */
    VkInstance OA_VMA_NOT_NULL instance;
    /** \brief Optional. Vulkan version that the application uses.

    It must be a value in the format as created by macro `VK_MAKE_VERSION` or a constant like: `VK_API_VERSION_1_1`, `VK_API_VERSION_1_0`.
    The patch version number specified is ignored. Only the major and minor versions are considered.
    Only versions 1.0...1.4 are supported by the current implementation.
    Leaving it initialized to zero is equivalent to `VK_API_VERSION_1_0`.
    It must match the Vulkan version used by the application and supported on the selected physical device,
    so it must be no higher than `VkApplicationInfo::apiVersion` passed to `vkCreateInstance`
    and no higher than `VkPhysicalDeviceProperties::apiVersion` found on the physical device used.
    */
    uint32_t vulkanApiVersion;
#if OA_VMA_EXTERNAL_MEMORY
    /** \brief Either null or a pointer to an array of external memory handle types for each Vulkan memory type.

    If not NULL, it must be a pointer to an array of `VkPhysicalDeviceMemoryProperties::memoryTypeCount`
    elements, defining external memory handle types of particular Vulkan memory type,
    to be passed using `VkExportMemoryAllocateInfoKHR`.

    Any of the elements may be equal to 0, which means not to use `VkExportMemoryAllocateInfoKHR` on this memory type.
    This is also the default in case of `pTypeExternalMemoryHandleTypes` = NULL.
    */
    const VkExternalMemoryHandleTypeFlagsKHR* OA_VMA_NULLABLE OA_VMA_LEN_IF_NOT_NULL("VkPhysicalDeviceMemoryProperties::memoryTypeCount") pTypeExternalMemoryHandleTypes;
#endif // #if OA_VMA_EXTERNAL_MEMORY
} OaVmaAllocatorCreateInfo;

/// Information about existing #OaVmaAllocator object.
typedef struct OaVmaAllocatorInfo
{
    /** \brief Handle to Vulkan instance object.

    This is the same value as has been passed through OaVmaAllocatorCreateInfo::instance.
    */
    VkInstance OA_VMA_NOT_NULL instance;
    /** \brief Handle to Vulkan physical device object.

    This is the same value as has been passed through OaVmaAllocatorCreateInfo::physicalDevice.
    */
    VkPhysicalDevice OA_VMA_NOT_NULL physicalDevice;
    /** \brief Handle to Vulkan device object.

    This is the same value as has been passed through OaVmaAllocatorCreateInfo::device.
    */
    VkDevice OA_VMA_NOT_NULL device;
} OaVmaAllocatorInfo;

/** @} */

/**
\addtogroup group_stats
@{
*/

/** \brief Calculated statistics of memory usage e.g. in a specific memory type, heap, custom pool, or total.

These are fast to calculate.
See functions: OaVmaGetHeapBudgets(), OaVmaGetPoolStatistics().
*/
typedef struct OaVmaStatistics
{
    /** \brief Number of `VkDeviceMemory` objects - Vulkan memory blocks allocated.
    */
    uint32_t blockCount;
    /** \brief Number of #OaVmaAllocation objects allocated.

    Dedicated allocations have their own blocks, so each one adds 1 to `allocationCount` as well as `blockCount`.
    */
    uint32_t allocationCount;
    /** \brief Number of bytes allocated in `VkDeviceMemory` blocks.

    \note To avoid confusion, please be aware that what Vulkan calls an "allocation" - a whole `VkDeviceMemory` object
    (e.g. as in `VkPhysicalDeviceLimits::maxMemoryAllocationCount`) is called a "block" in VMA, while VMA calls
    "allocation" a #OaVmaAllocation object that represents a memory region sub-allocated from such block, usually for a single buffer or image.
    */
    VkDeviceSize blockBytes;
    /** \brief Total number of bytes occupied by all #OaVmaAllocation objects.

    Always less or equal than `blockBytes`.
    Difference `(blockBytes - allocationBytes)` is the amount of memory allocated from Vulkan
    but unused by any #OaVmaAllocation.
    */
    VkDeviceSize allocationBytes;
} OaVmaStatistics;

/** \brief More detailed statistics than #OaVmaStatistics.

These are slower to calculate. Use for debugging purposes.
See functions: OaVmaCalculateStatistics(), OaVmaCalculatePoolStatistics().

Previous version of the statistics API provided averages, but they have been removed
because they can be easily calculated as:

\code
VkDeviceSize allocationSizeAvg = detailedStats.statistics.allocationBytes / detailedStats.statistics.allocationCount;
VkDeviceSize unusedBytes = detailedStats.statistics.blockBytes - detailedStats.statistics.allocationBytes;
VkDeviceSize unusedRangeSizeAvg = unusedBytes / detailedStats.unusedRangeCount;
\endcode
*/
typedef struct OaVmaDetailedStatistics
{
    /// Basic statistics.
    OaVmaStatistics statistics;
    /// Number of free ranges of memory between allocations.
    uint32_t unusedRangeCount;
    /// Smallest allocation size. `VK_WHOLE_SIZE` if there are 0 allocations.
    VkDeviceSize allocationSizeMin;
    /// Largest allocation size. 0 if there are 0 allocations.
    VkDeviceSize allocationSizeMax;
    /// Smallest empty range size. `VK_WHOLE_SIZE` if there are 0 empty ranges.
    VkDeviceSize unusedRangeSizeMin;
    /// Largest empty range size. 0 if there are 0 empty ranges.
    VkDeviceSize unusedRangeSizeMax;
} OaVmaDetailedStatistics;

/** \brief  General statistics from current state of the Allocator -
total memory usage across all memory heaps and types.

These are slower to calculate. Use for debugging purposes.
See function OaVmaCalculateStatistics().
*/
typedef struct OaVmaTotalStatistics
{
    OaVmaDetailedStatistics memoryType[VK_MAX_MEMORY_TYPES];
    OaVmaDetailedStatistics memoryHeap[VK_MAX_MEMORY_HEAPS];
    OaVmaDetailedStatistics total;
} OaVmaTotalStatistics;

/** \brief Statistics of current memory usage and available budget for a specific memory heap.

These are fast to calculate.
See function OaVmaGetHeapBudgets().
*/
typedef struct OaVmaBudget
{
    /** \brief Statistics fetched from the library.
    */
    OaVmaStatistics statistics;
    /** \brief Estimated current memory usage of the program, in bytes.

    Fetched from system using VK_EXT_memory_budget extension if enabled.

    It might be different than `statistics.blockBytes` (usually higher) due to additional implicit objects
    also occupying the memory, like swapchain, pipelines, descriptor heaps, command buffers, or
    `VkDeviceMemory` blocks allocated outside of this library, if any.
    */
    VkDeviceSize usage;
    /** \brief Estimated amount of memory available to the program, in bytes.

    Fetched from system using VK_EXT_memory_budget extension if enabled.

    It might be different (most probably smaller) than `VkMemoryHeap::size[heapIndex]` due to factors
    external to the program, decided by the operating system.
    Difference `budget - usage` is the amount of additional memory that can probably
    be allocated without problems. Exceeding the budget may result in various problems.
    */
    VkDeviceSize budget;
} OaVmaBudget;

/** @} */

/**
\addtogroup group_alloc
@{
*/

/** \brief Parameters of new #OaVmaAllocation.

To be used with functions like OaVmaCreateBuffer(), OaVmaCreateImage(), and many others.
*/
typedef struct OaVmaAllocationCreateInfo
{
    /// Use #OaVmaAllocationCreateFlagBits enum.
    OaVmaAllocationCreateFlags flags;
    /** \brief Intended usage of memory.

    You can leave #OA_VMA_MEMORY_USAGE_UNKNOWN if you specify memory requirements in other way. \n
    If `pool` is not null, this member is ignored.
    */
    OaVmaMemoryUsage usage;
    /** \brief Flags that must be set in a Memory Type chosen for an allocation.

    Leave 0 if you specify memory requirements in other way. \n
    If `pool` is not null, this member is ignored.*/
    VkMemoryPropertyFlags requiredFlags;
    /** \brief Flags that preferably should be set in a memory type chosen for an allocation.

    Set to 0 if no additional flags are preferred. \n
    If `pool` is not null, this member is ignored. */
    VkMemoryPropertyFlags preferredFlags;
    /** \brief Bitmask containing one bit set for every memory type acceptable for this allocation.

    Value 0 is equivalent to `UINT32_MAX` - it means any memory type is accepted if
    it meets other requirements specified by this structure, with no further
    restrictions on memory type index. \n
    If `pool` is not null, this member is ignored.
    */
    uint32_t memoryTypeBits;
    /** \brief Pool that this allocation should be created in.

    Leave `VK_NULL_HANDLE` to allocate from default pool. If not null, members:
    `usage`, `requiredFlags`, `preferredFlags`, `memoryTypeBits` are ignored.
    */
    OaVmaPool OA_VMA_NULLABLE pool;
    /** \brief Custom general-purpose pointer that will be stored in #OaVmaAllocation, can be read as OaVmaAllocationInfo::pUserData and changed using OaVmaSetAllocationUserData().

    If #OA_VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT is used, it must be either
    null or pointer to a null-terminated string. The string will be then copied to
    internal buffer, so it doesn't need to be valid after allocation call.
    */
    void* OA_VMA_NULLABLE pUserData;
    /** \brief A floating-point value between 0 and 1, indicating the priority of the allocation relative to other memory allocations.

    It is used only when #OA_VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT flag was used during creation of the #OaVmaAllocator object
    and this allocation ends up as dedicated or is explicitly forced as dedicated using #OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
    Otherwise, it has the priority of a memory block where it is placed and this variable is ignored.
    */
    float priority;
    /** \brief Additional minimum alignment to be used for this allocation. Can be 0.

    Leave 0 (default) not to impose any additional alignment. If not 0, it must be a power of two.
    
    When creating a buffer or an image, specifying a custom alignment is not needed in most cases,
    because Vulkan implementation inspects the `CreateInfo` structure (including intended usage flags)
    and returns required alignment through functions like `vkGetBufferMemoryRequirements2`, which VMA automatically
    uses and respects.
    Extra alignment may be needed in some cases, like when using a buffer for acceleration structure scratch
    (`VkPhysicalDeviceAccelerationStructurePropertiesKHR::minAccelerationStructureScratchOffsetAlignment`, see also issue #523)
    or when doing interop with OpenGL.
    */
    VkDeviceSize minAlignment;
} OaVmaAllocationCreateInfo;

/// Describes parameter of created #OaVmaPool.
typedef struct OaVmaPoolCreateInfo
{
    /** \brief Vulkan memory type index to allocate this pool from.
    */
    uint32_t memoryTypeIndex;
    /** \brief Use combination of #OaVmaPoolCreateFlagBits.
    */
    OaVmaPoolCreateFlags flags;
    /** \brief Size of a single `VkDeviceMemory` block to be allocated as part of this pool, in bytes. Optional.

    Specify nonzero to set explicit, constant size of memory blocks used by this
    pool.

    Leave 0 to use default and let the library manage block sizes automatically.
    Sizes of particular blocks may vary.
    In this case, the pool will also support dedicated allocations.
    */
    VkDeviceSize blockSize;
    /** \brief Minimum number of blocks to be always allocated in this pool, even if they stay empty.

    Set to 0 to have no preallocated blocks and allow the pool be completely empty.
    */
    size_t minBlockCount;
    /** \brief Maximum number of blocks that can be allocated in this pool. Optional.

    Set to 0 to use default, which is `SIZE_MAX`, which means no limit.

    Set to same value as OaVmaPoolCreateInfo::minBlockCount to have fixed amount of memory allocated
    throughout whole lifetime of this pool.
    */
    size_t maxBlockCount;
    /** \brief A floating-point value between 0 and 1, indicating the priority of the allocations in this pool relative to other memory allocations.

    It is used only when #OA_VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT flag was used during creation of the #OaVmaAllocator object.
    Otherwise, this variable is ignored.
    */
    float priority;
    /** \brief Additional minimum alignment to be used for all allocations created from this pool. Can be 0.

    Leave 0 (default) not to impose any additional alignment. If not 0, it must be a power of two.

    When creating a buffer or an image, specifying a custom alignment is not needed in most cases,
    because Vulkan implementation inspects the `CreateInfo` structure (including intended usage flags)
    and returns required alignment through functions like `vkGetBufferMemoryRequirements2`, which VMA automatically
    uses and respects.
    Extra alignment may be needed in some cases, like when using a buffer for acceleration structure scratch
    (`VkPhysicalDeviceAccelerationStructurePropertiesKHR::minAccelerationStructureScratchOffsetAlignment`, see also issue #523)
    or when doing interop with OpenGL.
    */
    VkDeviceSize minAllocationAlignment;
    /** \brief Additional `pNext` chain to be attached to `VkMemoryAllocateInfo` used for every allocation made by this pool. Optional.

    Optional, can be null. If not null, it must point to a `pNext` chain of structures that can be attached to `VkMemoryAllocateInfo`.
    It can be useful for special needs such as adding `VkExportMemoryAllocateInfoKHR`.
    Structures pointed by this member must remain alive and unchanged for the whole lifetime of the custom pool.

    Please note that some structures, e.g. `VkMemoryPriorityAllocateInfoEXT`, `VkMemoryDedicatedAllocateInfoKHR`,
    can be attached automatically by this library when using other, more convenient of its features.
    */
    void* OA_VMA_NULLABLE OA_VMA_EXTENDS_VK_STRUCT(VkMemoryAllocateInfo) pMemoryAllocateNext;
} OaVmaPoolCreateInfo;

/** @} */

/**
\addtogroup group_alloc
@{
*/

/**
Parameters of #OaVmaAllocation objects, that can be retrieved using function OaVmaGetAllocationInfo().

There is also an extended version of this structure that carries additional parameters: #OaVmaAllocationInfo2.
*/
typedef struct OaVmaAllocationInfo
{
    /** \brief Memory type index that this allocation was allocated from.

    It never changes.
    */
    uint32_t memoryType;
    /** \brief Handle to Vulkan memory object.

    Same memory object can be shared by multiple allocations.

    It can change after the allocation is moved during \ref defragmentation.
    */
    VkDeviceMemory OA_VMA_NULLABLE_NON_DISPATCHABLE deviceMemory;
    /** \brief Offset in `VkDeviceMemory` object to the beginning of this allocation, in bytes. `(deviceMemory, offset)` pair is unique to this allocation.

    You usually don't need to use this offset. If you create a buffer or an image together with the allocation using e.g. function
    OaVmaCreateBuffer(), OaVmaCreateImage(), functions that operate on these resources refer to the beginning of the buffer or image,
    not entire device memory block. Functions like OaVmaMapMemory(), OaVmaBindBufferMemory() also refer to the beginning of the allocation
    and apply this offset automatically.

    It can change after the allocation is moved during \ref defragmentation.
    */
    VkDeviceSize offset;
    /** \brief Size of this allocation, in bytes.

    It never changes.

    \note Allocation size returned in this variable may be greater than the size
    requested for the resource e.g. as `VkBufferCreateInfo::size`. Whole size of the
    allocation is accessible for operations on memory e.g. using a pointer after
    mapping with OaVmaMapMemory(), but operations on the resource e.g. using
    `vkCmdCopyBuffer` must be limited to the size of the resource.
    */
    VkDeviceSize size;
    /** \brief Pointer to the beginning of this allocation as mapped data.

    If the allocation hasn't been mapped using OaVmaMapMemory() and hasn't been
    created with #OA_VMA_ALLOCATION_CREATE_MAPPED_BIT flag, this value is null.

    It can change after call to OaVmaMapMemory(), OaVmaUnmapMemory().
    It can also change after the allocation is moved during \ref defragmentation.
    */
    void* OA_VMA_NULLABLE pMappedData;
    /** \brief Custom general-purpose pointer that was passed as OaVmaAllocationCreateInfo::pUserData or set using OaVmaSetAllocationUserData().

    It can change after call to OaVmaSetAllocationUserData() for this allocation.
    */
    void* OA_VMA_NULLABLE pUserData;
    /** \brief Custom allocation name that was set with OaVmaSetAllocationName().

    It can change after call to OaVmaSetAllocationName() for this allocation.

    Another way to set custom name is to pass it in OaVmaAllocationCreateInfo::pUserData with
    additional flag #OA_VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT set [DEPRECATED].
    */
    const char* OA_VMA_NULLABLE pName;
} OaVmaAllocationInfo;

/// Extended parameters of a #OaVmaAllocation object that can be retrieved using function OaVmaGetAllocationInfo2().
typedef struct OaVmaAllocationInfo2
{
    /** \brief Basic parameters of the allocation.
    
    If you need only these, you can use function OaVmaGetAllocationInfo() and structure #OaVmaAllocationInfo instead.
    */
    OaVmaAllocationInfo allocationInfo;
    /** \brief Size of the `VkDeviceMemory` block that the allocation belongs to.
    
    In case of an allocation with dedicated memory, it will be equal to `allocationInfo.size`.
    */
    VkDeviceSize blockSize;
    /** \brief `VK_TRUE` if the allocation has dedicated memory, `VK_FALSE` if it was placed as part of a larger memory block.
    
    When `VK_TRUE`, it also means `VkMemoryDedicatedAllocateInfo` was used when creating the allocation
    (if VK_KHR_dedicated_allocation extension or Vulkan version >= 1.1 is enabled).
    */
    VkBool32 dedicatedMemory;
} OaVmaAllocationInfo2;

/** Callback function called during OaVmaBeginDefragmentation() to check custom criterion about ending current defragmentation pass.

Should return true if the defragmentation needs to stop current pass.
*/
typedef VkBool32 (VKAPI_PTR* PFN_OaVmaCheckDefragmentationBreakFunction)(void* OA_VMA_NULLABLE pUserData);

/** \brief Parameters for defragmentation.

To be used with function OaVmaBeginDefragmentation().
*/
typedef struct OaVmaDefragmentationInfo
{
    /// \brief Use combination of #OaVmaDefragmentationFlagBits.
    OaVmaDefragmentationFlags flags;
    /** \brief Custom pool to be defragmented.

    If null then default pools will undergo defragmentation process.
    */
    OaVmaPool OA_VMA_NULLABLE pool;
    /** \brief Maximum numbers of bytes that can be copied during single pass, while moving allocations to different places.

    `0` means no limit.
    */
    VkDeviceSize maxBytesPerPass;
    /** \brief Maximum number of allocations that can be moved during single pass to a different place.

    `0` means no limit.
    */
    uint32_t maxAllocationsPerPass;
    /** \brief Optional custom callback for stopping OaVmaBeginDefragmentation().

    Have to return true for breaking current defragmentation pass.
    */
    PFN_OaVmaCheckDefragmentationBreakFunction OA_VMA_NULLABLE pfnBreakCallback;
    /// \brief Optional data to pass to custom callback for stopping pass of defragmentation.
    void* OA_VMA_NULLABLE pBreakCallbackUserData;
} OaVmaDefragmentationInfo;

/// Single move of an allocation to be done for defragmentation.
typedef struct OaVmaDefragmentationMove
{
    /// Operation to be performed on the allocation by OaVmaEndDefragmentationPass(). Default value is #OA_VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY. You can modify it.
    OaVmaDefragmentationMoveOperation operation;
    /// Allocation that should be moved.
    OaVmaAllocation OA_VMA_NOT_NULL srcAllocation;
    /** \brief Temporary allocation pointing to destination memory that will replace `srcAllocation`.

    \warning Do not store this allocation in your data structures! It exists only temporarily, for the duration of the defragmentation pass,
    to be used for binding new buffer/image to the destination memory using e.g. OaVmaBindBufferMemory().
    OaVmaEndDefragmentationPass() will destroy it and make `srcAllocation` point to this memory.
    */
    OaVmaAllocation OA_VMA_NOT_NULL dstTmpAllocation;
} OaVmaDefragmentationMove;

/** \brief Parameters for incremental defragmentation steps.

To be used with function OaVmaBeginDefragmentationPass().
*/
typedef struct OaVmaDefragmentationPassMoveInfo
{
    /// Number of elements in the `pMoves` array.
    uint32_t moveCount;
    /** \brief Array of moves to be performed by the user in the current defragmentation pass.

    Pointer to an array of `moveCount` elements, owned by VMA, created in OaVmaBeginDefragmentationPass(), destroyed in OaVmaEndDefragmentationPass().

    For each element, you should:

    1. Create a new buffer/image in the place pointed by OaVmaDefragmentationMove::dstMemory + OaVmaDefragmentationMove::dstOffset.
    2. Copy data from the OaVmaDefragmentationMove::srcAllocation e.g. using `vkCmdCopyBuffer`, `vkCmdCopyImage`.
    3. Make sure these commands finished executing on the GPU.
    4. Destroy the old buffer/image.

    Only then you can finish defragmentation pass by calling OaVmaEndDefragmentationPass().
    After this call, the allocation will point to the new place in memory.

    Alternatively, if you cannot move specific allocation, you can set OaVmaDefragmentationMove::operation to #OA_VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE.

    Alternatively, if you decide you want to completely remove the allocation:

    1. Destroy its buffer/image.
    2. Set OaVmaDefragmentationMove::operation to #OA_VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY.

    Then, after OaVmaEndDefragmentationPass() the allocation will be freed.
    */
    OaVmaDefragmentationMove* OA_VMA_NULLABLE OA_VMA_LEN_IF_NOT_NULL(moveCount) pMoves;
} OaVmaDefragmentationPassMoveInfo;

/// Statistics returned for defragmentation process in function OaVmaEndDefragmentation().
typedef struct OaVmaDefragmentationStats
{
    /// Total number of bytes that have been copied while moving allocations to different places.
    VkDeviceSize bytesMoved;
    /// Total number of bytes that have been released to the system by freeing empty `VkDeviceMemory` objects.
    VkDeviceSize bytesFreed;
    /// Number of allocations that have been moved to different places.
    uint32_t allocationsMoved;
    /// Number of empty `VkDeviceMemory` objects that have been released to the system.
    uint32_t deviceMemoryBlocksFreed;
} OaVmaDefragmentationStats;

/** @} */

/**
\addtogroup group_virtual
@{
*/

/// Parameters of created #OaVmaVirtualBlock object to be passed to OaVmaCreateVirtualBlock().
typedef struct OaVmaVirtualBlockCreateInfo
{
    /** \brief Total size of the virtual block.

    Sizes can be expressed in bytes or any units you want as long as you are consistent in using them.
    For example, if you allocate from some array of structures, 1 can mean single instance of entire structure.
    */
    VkDeviceSize size;

    /** \brief Use combination of #OaVmaVirtualBlockCreateFlagBits.
    */
    OaVmaVirtualBlockCreateFlags flags;

    /** \brief Custom CPU memory allocation callbacks. Optional.

    Optional, can be null. When specified, they will be used for all CPU-side memory allocations.
    */
    const VkAllocationCallbacks* OA_VMA_NULLABLE pAllocationCallbacks;
} OaVmaVirtualBlockCreateInfo;

/// Parameters of created virtual allocation to be passed to OaVmaVirtualAllocate().
typedef struct OaVmaVirtualAllocationCreateInfo
{
    /** \brief Size of the allocation.

    Cannot be zero.
    */
    VkDeviceSize size;
    /** \brief Required alignment of the allocation. Optional.

    Must be power of two. Special value 0 has the same meaning as 1 - means no special alignment is required, so allocation can start at any offset.
    */
    VkDeviceSize alignment;
    /** \brief Use combination of #OaVmaVirtualAllocationCreateFlagBits.
    */
    OaVmaVirtualAllocationCreateFlags flags;
    /** \brief Custom pointer to be associated with the allocation. Optional.

    It can be any value and can be used for user-defined purposes. It can be fetched or changed later.
    */
    void* OA_VMA_NULLABLE pUserData;
} OaVmaVirtualAllocationCreateInfo;

/// Parameters of an existing virtual allocation, returned by OaVmaGetVirtualAllocationInfo().
typedef struct OaVmaVirtualAllocationInfo
{
    /** \brief Offset of the allocation.

    Offset at which the allocation was made.
    */
    VkDeviceSize offset;
    /** \brief Size of the allocation.

    Same value as passed in OaVmaVirtualAllocationCreateInfo::size.
    */
    VkDeviceSize size;
    /** \brief Custom pointer associated with the allocation.

    Same value as passed in OaVmaVirtualAllocationCreateInfo::pUserData or to OaVmaSetVirtualAllocationUserData().
    */
    void* OA_VMA_NULLABLE pUserData;
} OaVmaVirtualAllocationInfo;

/** @} */

#endif // _OA_VMA_DATA_TYPES_DECLARATIONS

#ifndef _OA_VMA_FUNCTION_HEADERS

/**
\addtogroup group_init
@{
*/

#ifdef OAVK_HEADER_VERSION
/** \brief Fully initializes `pDstVulkanFunctions` structure with Vulkan functions needed by VMA
using [volk library](https://github.com/zeux/volk).

This function is defined in VMA header only if "volk.h" was included before it.

To use this function properly:

-# Initialize volk and Vulkan:
   -# Call `volkInitialize()`
   -# Create `VkInstance` object
   -# Call `volkLoadInstance()`
   -# Create `VkDevice` object
   -# Call `volkLoadDevice()`
-# Fill in structure #OaVmaAllocatorCreateInfo, especially members:
   - OaVmaAllocatorCreateInfo::device
   - OaVmaAllocatorCreateInfo::vulkanApiVersion
   - OaVmaAllocatorCreateInfo::flags - set appropriate flags for the Vulkan extensions you enabled
-# Create an instance of the #OaVmaVulkanFunctions structure.
-# Call OaVmaImportVulkanFunctionsFromOaVk().
   Parameter `pAllocatorCreateInfo` is read to find out which functions should be fetched for
   appropriate Vulkan version and extensions.
   Parameter `pDstVulkanFunctions` is filled with those function pointers, or null if not applicable.
-# Attach the #OaVmaVulkanFunctions structure to OaVmaAllocatorCreateInfo::pVulkanFunctions.
-# Call OaVmaCreateAllocator() to create the #OaVmaAllocator object.

Example:

\code
OaVmaAllocatorCreateInfo allocatorCreateInfo = {};
allocatorCreateInfo.physicalDevice = myPhysicalDevice;
allocatorCreateInfo.device = myDevice;
allocatorCreateInfo.instance = myInstance;
allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
allocatorCreateInfo.flags = OA_VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT |
    OA_VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT |
    OA_VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT;

OaVmaVulkanFunctions vulkanFunctions;
VkResult res = OaVmaImportVulkanFunctionsFromOaVk(&allocatorCreateInfo, &vulkanFunctions);
// Check res...
allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;

OaVmaAllocator allocator;
res = OaVmaCreateAllocator(&allocatorCreateInfo, &allocator);
// Check res...
\endcode

Internally in this function, pointers to functions related to the entire Vulkan instance are fetched using global function definitions,
while pointers to functions related to the Vulkan device are fetched using `volkLoadDeviceTable()` for given `pAllocatorCreateInfo->device`.
 */
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaImportVulkanFunctionsFromOaVk(
    const OaVmaAllocatorCreateInfo* OA_VMA_NOT_NULL pAllocatorCreateInfo,
    OaVmaVulkanFunctions* OA_VMA_NOT_NULL pDstVulkanFunctions);
#endif

/// Creates #OaVmaAllocator object.
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateAllocator(
    const OaVmaAllocatorCreateInfo* OA_VMA_NOT_NULL pCreateInfo,
    OaVmaAllocator OA_VMA_NULLABLE* OA_VMA_NOT_NULL pAllocator);

/// Destroys allocator object.
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaDestroyAllocator(
    OaVmaAllocator OA_VMA_NULLABLE allocator);

/** \brief Returns information about existing #OaVmaAllocator object - handle to Vulkan device etc.

It might be useful if you want to keep just the #OaVmaAllocator handle and fetch other required handles to
`VkPhysicalDevice`, `VkDevice` etc. every time using this function.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetAllocatorInfo(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocatorInfo* OA_VMA_NOT_NULL pAllocatorInfo);

/**
PhysicalDeviceProperties are fetched from physicalDevice by the allocator.
You can access it here, without fetching it again on your own.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetPhysicalDeviceProperties(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    const VkPhysicalDeviceProperties* OA_VMA_NULLABLE* OA_VMA_NOT_NULL ppPhysicalDeviceProperties);

/**
PhysicalDeviceMemoryProperties are fetched from physicalDevice by the allocator.
You can access it here, without fetching it again on your own.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetMemoryProperties(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    const VkPhysicalDeviceMemoryProperties* OA_VMA_NULLABLE* OA_VMA_NOT_NULL ppPhysicalDeviceMemoryProperties);

/**
\brief Given Memory Type Index, returns Property Flags of this memory type.

This is just a convenience function. Same information can be obtained using
OaVmaGetMemoryProperties().
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetMemoryTypeProperties(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    uint32_t memoryTypeIndex,
    VkMemoryPropertyFlags* OA_VMA_NOT_NULL pFlags);

/** \brief Sets index of the current frame.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaSetCurrentFrameIndex(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    uint32_t frameIndex);

/** @} */

/**
\addtogroup group_stats
@{
*/

/** \brief Retrieves statistics from current state of the Allocator.

This function is called "calculate" not "get" because it has to traverse all
internal data structures, so it may be quite slow. Use it for debugging purposes.
For faster but more brief statistics suitable to be called every frame or every allocation,
use OaVmaGetHeapBudgets().

Note that when using allocator from multiple threads, returned information may immediately
become outdated.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaCalculateStatistics(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaTotalStatistics* OA_VMA_NOT_NULL pStats);

/** \brief Retrieves information about current memory usage and budget for all memory heaps.

\param allocator
\param[out] pBudgets Must point to array with number of elements at least equal to number of memory heaps in physical device used.

This function is called "get" not "calculate" because it is very fast, suitable to be called
every frame or every allocation. For more detailed statistics use OaVmaCalculateStatistics().

Note that when using allocator from multiple threads, returned information may immediately
become outdated.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetHeapBudgets(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaBudget* OA_VMA_NOT_NULL OA_VMA_LEN_IF_NOT_NULL("VkPhysicalDeviceMemoryProperties::memoryHeapCount") pBudgets);

/** @} */

/**
\addtogroup group_alloc
@{
*/

/**
\brief Helps to find `memoryTypeIndex`, given `memoryTypeBits` and #OaVmaAllocationCreateInfo.

This algorithm tries to find a memory type that:

- Is allowed by `memoryTypeBits`.
- Contains all the flags from `pAllocationCreateInfo->requiredFlags`.
- Matches intended usage.
- Has as many flags from `pAllocationCreateInfo->preferredFlags` as possible.

\return Returns `VK_ERROR_FEATURE_NOT_PRESENT` if not found. Receiving such result
from this function or any other allocating function probably means that your
device doesn't support any memory type with requested features for the specific
type of resource you want to use it for. Please check parameters of your
resource, like image layout (`OPTIMAL` versus `LINEAR`) or mip level count.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaFindMemoryTypeIndex(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    uint32_t memoryTypeBits,
    const OaVmaAllocationCreateInfo* OA_VMA_NOT_NULL pAllocationCreateInfo,
    uint32_t* OA_VMA_NOT_NULL pMemoryTypeIndex);

/**
\brief Helps to find `memoryTypeIndex`, given `VkBufferCreateInfo` and #OaVmaAllocationCreateInfo.

It can be useful e.g. to determine value to be used as OaVmaPoolCreateInfo::memoryTypeIndex.
It may need to internally create a temporary, dummy buffer that never has memory bound.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaFindMemoryTypeIndexForBufferInfo(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    const VkBufferCreateInfo* OA_VMA_NOT_NULL pBufferCreateInfo,
    const OaVmaAllocationCreateInfo* OA_VMA_NOT_NULL pAllocationCreateInfo,
    uint32_t* OA_VMA_NOT_NULL pMemoryTypeIndex);

/**
\brief Helps to find `memoryTypeIndex`, given `VkImageCreateInfo` and #OaVmaAllocationCreateInfo.

It can be useful e.g. to determine value to be used as OaVmaPoolCreateInfo::memoryTypeIndex.
It may need to internally create a temporary, dummy image that never has memory bound.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaFindMemoryTypeIndexForImageInfo(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    const VkImageCreateInfo* OA_VMA_NOT_NULL pImageCreateInfo,
    const OaVmaAllocationCreateInfo* OA_VMA_NOT_NULL pAllocationCreateInfo,
    uint32_t* OA_VMA_NOT_NULL pMemoryTypeIndex);

/** \brief Allocates Vulkan device memory and creates #OaVmaPool object.

\param allocator Allocator object.
\param pCreateInfo Parameters of pool to create.
\param[out] pPool Handle to created pool.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreatePool(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    const OaVmaPoolCreateInfo* OA_VMA_NOT_NULL pCreateInfo,
    OaVmaPool OA_VMA_NULLABLE* OA_VMA_NOT_NULL pPool);

/** \brief Destroys #OaVmaPool object and frees Vulkan device memory.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaDestroyPool(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaPool OA_VMA_NULLABLE pool);

/** @} */

/**
\addtogroup group_stats
@{
*/

/** \brief Retrieves statistics of existing #OaVmaPool object.

\param allocator Allocator object.
\param pool Pool object.
\param[out] pPoolStats Statistics of specified pool.

Note that when using the pool from multiple threads, returned information may immediately
become outdated.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetPoolStatistics(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaPool OA_VMA_NOT_NULL pool,
    OaVmaStatistics* OA_VMA_NOT_NULL pPoolStats);

/** \brief Retrieves detailed statistics of existing #OaVmaPool object.

\param allocator Allocator object.
\param pool Pool object.
\param[out] pPoolStats Statistics of specified pool.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaCalculatePoolStatistics(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaPool OA_VMA_NOT_NULL pool,
    OaVmaDetailedStatistics* OA_VMA_NOT_NULL pPoolStats);

/** @} */

/**
\addtogroup group_alloc
@{
*/

/** \brief Checks magic number in margins around all allocations in given memory pool in search for corruptions.

Corruption detection is enabled only when `OA_VMA_DEBUG_DETECT_CORRUPTION` macro is defined to nonzero,
`OA_VMA_DEBUG_MARGIN` is defined to nonzero and the pool is created in memory type that is
`HOST_VISIBLE` and `HOST_COHERENT`. For more information, see [Corruption detection](@ref debugging_memory_usage_corruption_detection).

Possible return values:

- `VK_ERROR_FEATURE_NOT_PRESENT` - corruption detection is not enabled for specified pool.
- `VK_SUCCESS` - corruption detection has been performed and succeeded.
- `VK_ERROR_UNKNOWN` - corruption detection has been performed and found memory corruptions around one of the allocations.
  `OA_VMA_ASSERT` is also fired in that case.
- Other value: Error returned by Vulkan, e.g. memory mapping failure.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCheckPoolCorruption(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaPool OA_VMA_NOT_NULL pool);

/** \brief Retrieves name of a custom pool.

After the call `ppName` is either null or points to an internally-owned null-terminated string
containing name of the pool that was previously set. The pointer becomes invalid when the pool is
destroyed or its name is changed using OaVmaSetPoolName().
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetPoolName(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaPool OA_VMA_NOT_NULL pool,
    const char* OA_VMA_NULLABLE* OA_VMA_NOT_NULL ppName);

/** \brief Sets name of a custom pool.

`pName` can be either null or pointer to a null-terminated string with new name for the pool.
Function makes internal copy of the string, so it can be changed or freed immediately after this call.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaSetPoolName(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaPool OA_VMA_NOT_NULL pool,
    const char* OA_VMA_NULLABLE pName);

/** \brief General purpose memory allocation.

\param allocator The main allocator object.
\param pVkMemoryRequirements Requirements for the allocated memory.
\param pCreateInfo Allocation creation parameters.
\param[out] pAllocation Handle to allocated memory.
\param[out] pAllocationInfo Optional, can be null. Information about allocated memory. It can be also fetched later using OaVmaGetAllocationInfo().

The function creates a #OaVmaAllocation object without creating a buffer or an image together with it.

- It is recommended to use OaVmaAllocateMemoryForBuffer(), OaVmaAllocateMemoryForImage(),
  OaVmaCreateBuffer(), OaVmaCreateImage() instead whenever possible.
- You can also create a buffer or an image later in an existing allocation using
  OaVmaCreateAliasingBuffer2(), OaVmaCreateAliasingImage2().
- You can also create a buffer or an image on your own and bind it to an existing allocation
  using OaVmaBindBufferMemory2(), OaVmaBindImageMemory2().

You must free the returned allocation object using OaVmaFreeMemory() or OaVmaFreeMemoryPages().

There is also extended version of this function: OaVmaAllocateDedicatedMemory()
that offers additional parameter `pMemoryAllocateNext`.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaAllocateMemory(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    const VkMemoryRequirements* OA_VMA_NOT_NULL pVkMemoryRequirements,
    const OaVmaAllocationCreateInfo* OA_VMA_NOT_NULL pCreateInfo,
    OaVmaAllocation OA_VMA_NULLABLE* OA_VMA_NOT_NULL pAllocation,
    OaVmaAllocationInfo* OA_VMA_NULLABLE pAllocationInfo);

/** \brief General purpose allocation of a dedicated memory.

This function is similar OaVmaAllocateMemory(), but
it always allocates dedicated memory - flag #OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT is implied.
It offers additional parameter `pMemoryAllocateNext`,
which can be used to attach `pNext` chain to the `VkMemoryAllocateInfo` structure.
It can be useful for importing external memory. For more information, see \ref other_api_interop.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaAllocateDedicatedMemory(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    const VkMemoryRequirements* OA_VMA_NOT_NULL pVkMemoryRequirements,
    const OaVmaAllocationCreateInfo* OA_VMA_NOT_NULL pCreateInfo,
    void* OA_VMA_NULLABLE OA_VMA_EXTENDS_VK_STRUCT(VkMemoryAllocateInfo) pMemoryAllocateNext,
    OaVmaAllocation OA_VMA_NULLABLE* OA_VMA_NOT_NULL pAllocation,
    OaVmaAllocationInfo* OA_VMA_NULLABLE pAllocationInfo);

/** \brief General purpose memory allocation for multiple allocation objects at once.

\param allocator Allocator object.
\param pVkMemoryRequirements Memory requirements for each allocation.
\param pCreateInfo Creation parameters for each allocation.
\param allocationCount Number of allocations to make.
\param[out] pAllocations Pointer to array that will be filled with handles to created allocations.
\param[out] pAllocationInfo Optional. Pointer to array that will be filled with parameters of created allocations.

You should free the memory using OaVmaFreeMemory() or OaVmaFreeMemoryPages().

Word "pages" is just a suggestion to use this function to allocate pieces of memory needed for sparse binding.
It is just a general purpose allocation function able to make multiple allocations at once.
It may be internally optimized to be more efficient than calling OaVmaAllocateMemory() `allocationCount` times.

All allocations are made using same parameters. All of them are created out of the same memory pool and type.
If any allocation fails, all allocations already made within this function call are also freed, so that when
returned result is not `VK_SUCCESS`, `pAllocation` array is always entirely filled with `VK_NULL_HANDLE`.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaAllocateMemoryPages(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    const VkMemoryRequirements* OA_VMA_NOT_NULL OA_VMA_LEN_IF_NOT_NULL(allocationCount) pVkMemoryRequirements,
    const OaVmaAllocationCreateInfo* OA_VMA_NOT_NULL OA_VMA_LEN_IF_NOT_NULL(allocationCount) pCreateInfo,
    size_t allocationCount,
    OaVmaAllocation OA_VMA_NULLABLE* OA_VMA_NOT_NULL OA_VMA_LEN_IF_NOT_NULL(allocationCount) pAllocations,
    OaVmaAllocationInfo* OA_VMA_NULLABLE OA_VMA_LEN_IF_NOT_NULL(allocationCount) pAllocationInfo);

/** \brief Allocates memory suitable for given `VkBuffer`.

\param allocator
\param buffer
\param pCreateInfo
\param[out] pAllocation Handle to allocated memory.
\param[out] pAllocationInfo Optional. Information about allocated memory. It can be later fetched using function OaVmaGetAllocationInfo().

It only creates #OaVmaAllocation. To bind the memory to the buffer, use OaVmaBindBufferMemory().

This is a special-purpose function. In most cases you should use OaVmaCreateBuffer().

You must free the allocation using OaVmaFreeMemory() when no longer needed.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaAllocateMemoryForBuffer(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    VkBuffer OA_VMA_NOT_NULL_NON_DISPATCHABLE buffer,
    const OaVmaAllocationCreateInfo* OA_VMA_NOT_NULL pCreateInfo,
    OaVmaAllocation OA_VMA_NULLABLE* OA_VMA_NOT_NULL pAllocation,
    OaVmaAllocationInfo* OA_VMA_NULLABLE pAllocationInfo);

/** \brief Allocates memory suitable for given `VkImage`.

\param allocator
\param image
\param pCreateInfo
\param[out] pAllocation Handle to allocated memory.
\param[out] pAllocationInfo Optional. Information about allocated memory. It can be later fetched using function OaVmaGetAllocationInfo().

It only creates #OaVmaAllocation. To bind the memory to the buffer, use OaVmaBindImageMemory().

This is a special-purpose function. In most cases you should use OaVmaCreateImage().

You must free the allocation using OaVmaFreeMemory() when no longer needed.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaAllocateMemoryForImage(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    VkImage OA_VMA_NOT_NULL_NON_DISPATCHABLE image,
    const OaVmaAllocationCreateInfo* OA_VMA_NOT_NULL pCreateInfo,
    OaVmaAllocation OA_VMA_NULLABLE* OA_VMA_NOT_NULL pAllocation,
    OaVmaAllocationInfo* OA_VMA_NULLABLE pAllocationInfo);

/** \brief Frees memory previously allocated using OaVmaAllocateMemory(), OaVmaAllocateMemoryForBuffer(), or OaVmaAllocateMemoryForImage().

Passing `VK_NULL_HANDLE` as `allocation` is valid. Such function call is just skipped.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaFreeMemory(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NULLABLE allocation);

/** \brief Frees memory and destroys multiple allocations.

Word "pages" is just a suggestion to use this function to free pieces of memory used for sparse binding.
It is just a general purpose function to free memory and destroy allocations made using e.g. OaVmaAllocateMemory(),
OaVmaAllocateMemoryPages() and other functions.
It may be internally optimized to be more efficient than calling OaVmaFreeMemory() `allocationCount` times.

Allocations in `pAllocations` array can come from any memory pools and types.
Passing `VK_NULL_HANDLE` as elements of `pAllocations` array is valid. Such entries are just skipped.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaFreeMemoryPages(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    size_t allocationCount,
    const OaVmaAllocation OA_VMA_NULLABLE* OA_VMA_NOT_NULL OA_VMA_LEN_IF_NOT_NULL(allocationCount) pAllocations);

/** \brief Returns current information about specified allocation.

Current parameters of given allocation are returned in `pAllocationInfo`.

Although this function doesn't lock any mutex, so it should be quite efficient,
you should avoid calling it too often.
You can retrieve same OaVmaAllocationInfo structure while creating your resource, from function
OaVmaCreateBuffer(), OaVmaCreateImage(). You can remember it if you are sure parameters don't change
(e.g. due to defragmentation).

There is also a new function OaVmaGetAllocationInfo2() that offers extended information
about the allocation, returned using new structure #OaVmaAllocationInfo2.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetAllocationInfo(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    OaVmaAllocationInfo* OA_VMA_NOT_NULL pAllocationInfo);

/** \brief Returns extended information about specified allocation.

Current parameters of given allocation are returned in `pAllocationInfo`.
Extended parameters in structure #OaVmaAllocationInfo2 include memory block size
and a flag telling whether the allocation has dedicated memory.
It can be useful e.g. for interop with OpenGL.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetAllocationInfo2(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    OaVmaAllocationInfo2* OA_VMA_NOT_NULL pAllocationInfo);

/** \brief Sets pUserData in given allocation to new value.

The value of pointer `pUserData` is copied to allocation's `pUserData`.
It is opaque, so you can use it however you want - e.g.
as a pointer, ordinal number or some handle to you own data.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaSetAllocationUserData(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    void* OA_VMA_NULLABLE pUserData);

/** \brief Sets pName in given allocation to new value.

`pName` must be either null, or pointer to a null-terminated string. The function
makes local copy of the string and sets it as allocation's `pName`. String
passed as pName doesn't need to be valid for whole lifetime of the allocation -
you can free it after this call. String previously pointed by allocation's
`pName` is freed from memory.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaSetAllocationName(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    const char* OA_VMA_NULLABLE pName);

/**
\brief Given an allocation, returns Property Flags of its memory type.

This is just a convenience function. Same information can be obtained using
OaVmaGetAllocationInfo() + OaVmaGetMemoryProperties().
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetAllocationMemoryProperties(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    VkMemoryPropertyFlags* OA_VMA_NOT_NULL pFlags);


#if OA_VMA_EXTERNAL_MEMORY_WIN32
/**
\brief Given an allocation, returns Win32 handle that may be imported by other processes or APIs.

\param allocator The main allocator object.
\param allocation Allocation.
\param hTargetProcess A valid handle to target process or null. If it's null, the function returns
    handle for the current process.
\param[out] pHandle Output parameter that returns the handle.

The function fills `pHandle` with handle that can be used in target process.
The handle is fetched using function `vkGetMemoryWin32HandleKHR`.

Each call to this function creates a new handle that must be closed using:

\code
CloseHandle(handle);
\endcode

You can close it any time, before or after destroying the allocation object.
It is reference-counted internally by Windows.

Note the handle is returned for the entire `VkDeviceMemory` block that the allocation belongs to.
If the allocation is sub-allocated from a larger block, you may need to consider the offset of the allocation
(OaVmaAllocationInfo::offset).

This function always uses `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT`.
An extended version of this function is available as OaVmaGetMemoryWin32Handle2()
that allows using other handle type.

This function is available compile-time only when VK_KHR_external_memory_win32 extension is available.
It can be manually disabled by predefining `OA_VMA_EXTERNAL_MEMORY_WIN32=0` macro.

If the function fails with `VK_ERROR_FEATURE_NOT_PRESENT` error code, please double-check
that OaVmaVulkanFunctions::vkGetMemoryWin32HandleKHR function pointer is set, e.g.
either by using macro `OA_VMA_DYNAMIC_VULKAN_FUNCTIONS`
or by manually passing it through OaVmaAllocatorCreateInfo::pVulkanFunctions.

For more information, see chapter \ref other_api_interop.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaGetMemoryWin32Handle(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation, 
    HANDLE hTargetProcess, 
    HANDLE* OA_VMA_NOT_NULL pHandle);

/**
\brief Given an allocation, returns Win32 handle that may be imported by other processes or APIs.

\param allocator The main allocator object.
\param allocation Allocation.
\param handleType Type of handle to be exported. It should be one of:
    - `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR`
    - `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR`
    - `VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT_KHR`
    - `VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT_KHR`
    - `VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT_KHR`
    - `VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT_KHR`
\param hTargetProcess A valid handle to target process or null. If it's null, the function returns
    handle for the current process.
\param[out] pHandle Output parameter that returns the handle.

The function fills `pHandle` with handle that can be used in target process.
The handle is fetched using function `vkGetMemoryWin32HandleKHR`.

If `handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR`,
or other NT handle types,
each call to this function creates a new handle that must be closed using:

\code
CloseHandle(handle);
\endcode

You can close it any time, before or after destroying the allocation object.
It is reference-counted internally by Windows.

Note the handle is returned for the entire `VkDeviceMemory` block that the allocation belongs to.
If the allocation is sub-allocated from a larger block, you may need to consider the offset of the allocation
(OaVmaAllocationInfo::offset).

This function is available compile-time only when VK_KHR_external_memory_win32 extension is available.
It can be manually disabled by predefining `OA_VMA_EXTERNAL_MEMORY_WIN32=0` macro.

If the function fails with `VK_ERROR_FEATURE_NOT_PRESENT` error code, please double-check
that OaVmaVulkanFunctions::vkGetMemoryWin32HandleKHR function pointer is set, e.g.
either by using macro `OA_VMA_DYNAMIC_VULKAN_FUNCTIONS`
or by manually passing it through OaVmaAllocatorCreateInfo::pVulkanFunctions.

For more information, see chapter \ref other_api_interop.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaGetMemoryWin32Handle2(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation, 
    VkExternalMemoryHandleTypeFlagBits handleType, 
    HANDLE hTargetProcess, 
    HANDLE* OA_VMA_NOT_NULL pHandle);
#endif // OA_VMA_EXTERNAL_MEMORY_WIN32

/** \brief Maps memory represented by given allocation and returns pointer to it.

Maps memory represented by given allocation to make it accessible to CPU code.
When succeeded, `*ppData` contains pointer to first byte of this memory.

\warning
If the allocation is part of a bigger `VkDeviceMemory` block, returned pointer is
correctly offsetted to the beginning of region assigned to this particular allocation.
Unlike the result of `vkMapMemory`, it points to the allocation, not to the beginning of the whole block.
You should not add OaVmaAllocationInfo::offset to it!

Mapping is internally reference-counted and synchronized, so despite raw Vulkan
function `vkMapMemory()` cannot be used to map same block of `VkDeviceMemory`
multiple times simultaneously, it is safe to call this function on allocations
assigned to the same memory block. Actual Vulkan memory will be mapped on first
mapping and unmapped on last unmapping.

If the function succeeded, you must call OaVmaUnmapMemory() to unmap the
allocation when mapping is no longer needed or before freeing the allocation, at
the latest.

It also safe to call this function multiple times on the same allocation. You
must call OaVmaUnmapMemory() same number of times as you called OaVmaMapMemory().

It is also safe to call this function on allocation created with
#OA_VMA_ALLOCATION_CREATE_MAPPED_BIT flag. Its memory stays mapped all the time.
You must still call OaVmaUnmapMemory() same number of times as you called
OaVmaMapMemory(). You must not call OaVmaUnmapMemory() additional time to free the
"0-th" mapping made automatically due to #OA_VMA_ALLOCATION_CREATE_MAPPED_BIT flag.

This function fails when used on allocation made in memory type that is not
`HOST_VISIBLE`.

This function doesn't automatically flush or invalidate caches.
If the allocation is made from a memory types that is not `HOST_COHERENT`,
you also need to use OaVmaInvalidateAllocation() / OaVmaFlushAllocation(), as required by Vulkan specification.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaMapMemory(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    void* OA_VMA_NULLABLE* OA_VMA_NOT_NULL ppData);

/** \brief Unmaps memory represented by given allocation, mapped previously using OaVmaMapMemory().

For details, see description of OaVmaMapMemory().

This function doesn't automatically flush or invalidate caches.
If the allocation is made from a memory types that is not `HOST_COHERENT`,
you also need to use OaVmaInvalidateAllocation() / OaVmaFlushAllocation(), as required by Vulkan specification.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaUnmapMemory(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation);

/** \brief Flushes memory of given allocation.

Calls `vkFlushMappedMemoryRanges()` for memory associated with given range of given allocation.
It needs to be called after writing to a mapped memory for memory types that are not `HOST_COHERENT`.
Unmap operation doesn't do that automatically.

- `offset` must be relative to the beginning of allocation.
- `size` can be `VK_WHOLE_SIZE`. It means all memory from `offset` the the end of given allocation.
- `offset` and `size` don't have to be aligned.
  They are internally rounded down/up to multiply of `nonCoherentAtomSize`.
- If `size` is 0, this call is ignored.
- If memory type that the `allocation` belongs to is not `HOST_VISIBLE` or it is `HOST_COHERENT`,
  this call is ignored.

Warning! `offset` and `size` are relative to the contents of given `allocation`.
If you mean whole allocation, you can pass 0 and `VK_WHOLE_SIZE`, respectively.
Do not pass allocation's offset as `offset`!!!

This function returns the `VkResult` from `vkFlushMappedMemoryRanges` if it is
called, otherwise `VK_SUCCESS`.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaFlushAllocation(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    VkDeviceSize offset,
    VkDeviceSize size);

/** \brief Invalidates memory of given allocation.

Calls `vkInvalidateMappedMemoryRanges()` for memory associated with given range of given allocation.
It needs to be called before reading from a mapped memory for memory types that are not `HOST_COHERENT`.
Map operation doesn't do that automatically.

- `offset` must be relative to the beginning of allocation.
- `size` can be `VK_WHOLE_SIZE`. It means all memory from `offset` the the end of given allocation.
- `offset` and `size` don't have to be aligned.
  They are internally rounded down/up to multiply of `nonCoherentAtomSize`.
- If `size` is 0, this call is ignored.
- If memory type that the `allocation` belongs to is not `HOST_VISIBLE` or it is `HOST_COHERENT`,
  this call is ignored.

Warning! `offset` and `size` are relative to the contents of given `allocation`.
If you mean whole allocation, you can pass 0 and `VK_WHOLE_SIZE`, respectively.
Do not pass allocation's offset as `offset`!!!

This function returns the `VkResult` from `vkInvalidateMappedMemoryRanges` if
it is called, otherwise `VK_SUCCESS`.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaInvalidateAllocation(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    VkDeviceSize offset,
    VkDeviceSize size);

/** \brief Flushes memory of given set of allocations.

Calls `vkFlushMappedMemoryRanges()` for memory associated with given ranges of given allocations.
For more information, see documentation of OaVmaFlushAllocation().

\param allocator
\param allocationCount
\param allocations
\param offsets If not null, it must point to an array of offsets of regions to flush, relative to the beginning of respective allocations. Null means all offsets are zero.
\param sizes If not null, it must point to an array of sizes of regions to flush in respective allocations. Null means `VK_WHOLE_SIZE` for all allocations.

This function returns the `VkResult` from `vkFlushMappedMemoryRanges` if it is
called, otherwise `VK_SUCCESS`.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaFlushAllocations(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    uint32_t allocationCount,
    const OaVmaAllocation OA_VMA_NOT_NULL* OA_VMA_NULLABLE OA_VMA_LEN_IF_NOT_NULL(allocationCount) allocations,
    const VkDeviceSize* OA_VMA_NULLABLE OA_VMA_LEN_IF_NOT_NULL(allocationCount) offsets,
    const VkDeviceSize* OA_VMA_NULLABLE OA_VMA_LEN_IF_NOT_NULL(allocationCount) sizes);

/** \brief Invalidates memory of given set of allocations.

Calls `vkInvalidateMappedMemoryRanges()` for memory associated with given ranges of given allocations.
For more information, see documentation of OaVmaInvalidateAllocation().

\param allocator
\param allocationCount
\param allocations
\param offsets If not null, it must point to an array of offsets of regions to flush, relative to the beginning of respective allocations. Null means all offsets are zero.
\param sizes If not null, it must point to an array of sizes of regions to flush in respective allocations. Null means `VK_WHOLE_SIZE` for all allocations.

This function returns the `VkResult` from `vkInvalidateMappedMemoryRanges` if it is
called, otherwise `VK_SUCCESS`.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaInvalidateAllocations(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    uint32_t allocationCount,
    const OaVmaAllocation OA_VMA_NOT_NULL* OA_VMA_NULLABLE OA_VMA_LEN_IF_NOT_NULL(allocationCount) allocations,
    const VkDeviceSize* OA_VMA_NULLABLE OA_VMA_LEN_IF_NOT_NULL(allocationCount) offsets,
    const VkDeviceSize* OA_VMA_NULLABLE OA_VMA_LEN_IF_NOT_NULL(allocationCount) sizes);

/** \brief Maps the allocation temporarily if needed, copies data from specified host pointer to it, and flushes the memory from the host caches if needed.

\param allocator
\param pSrcHostPointer Pointer to the host data that become source of the copy.
\param dstAllocation   Handle to the allocation that becomes destination of the copy.
\param dstAllocationLocalOffset  Offset within `dstAllocation` where to write copied data, in bytes.
\param size            Number of bytes to copy.

This is a convenience function that allows to copy data from a host pointer to an allocation easily.
Same behavior can be achieved by calling OaVmaMapMemory(), `memcpy()`, OaVmaUnmapMemory(), OaVmaFlushAllocation().

This function can be called only for allocations created in a memory type that has `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` flag.
It can be ensured e.g. by using #OA_VMA_MEMORY_USAGE_AUTO and #OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or
#OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.
Otherwise, the function will fail and generate a Validation Layers error.

`dstAllocationLocalOffset` is relative to the contents of given `dstAllocation`.
If you mean whole allocation, you should pass 0.
Do not pass allocation's offset within device memory block this parameter!
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCopyMemoryToAllocation(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    const void* OA_VMA_NOT_NULL OA_VMA_LEN_IF_NOT_NULL(size) pSrcHostPointer,
    OaVmaAllocation OA_VMA_NOT_NULL dstAllocation,
    VkDeviceSize dstAllocationLocalOffset,
    VkDeviceSize size);

/** \brief Invalidates memory in the host caches if needed, maps the allocation temporarily if needed, and copies data from it to a specified host pointer.

\param allocator
\param srcAllocation   Handle to the allocation that becomes source of the copy.
\param srcAllocationLocalOffset  Offset within `srcAllocation` where to read copied data, in bytes.
\param pDstHostPointer Pointer to the host memory that become destination of the copy.
\param size            Number of bytes to copy.

This is a convenience function that allows to copy data from an allocation to a host pointer easily.
Same behavior can be achieved by calling OaVmaInvalidateAllocation(), OaVmaMapMemory(), `memcpy()`, OaVmaUnmapMemory().

This function should be called only for allocations created in a memory type that has `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`
and `VK_MEMORY_PROPERTY_HOST_CACHED_BIT` flag.
It can be ensured e.g. by using #OA_VMA_MEMORY_USAGE_AUTO and #OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.
Otherwise, the function may fail and generate a Validation Layers error.
It may also work very slowly when reading from an uncached memory.

`srcAllocationLocalOffset` is relative to the contents of given `srcAllocation`.
If you mean whole allocation, you should pass 0.
Do not pass allocation's offset within device memory block as this parameter!
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCopyAllocationToMemory(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL srcAllocation,
    VkDeviceSize srcAllocationLocalOffset,
    void* OA_VMA_NOT_NULL OA_VMA_LEN_IF_NOT_NULL(size) pDstHostPointer,
    VkDeviceSize size);

/** \brief Checks magic number in margins around all allocations in given memory types (in both default and custom pools) in search for corruptions.

\param allocator
\param memoryTypeBits Bit mask, where each bit set means that a memory type with that index should be checked.

Corruption detection is enabled only when `OA_VMA_DEBUG_DETECT_CORRUPTION` macro is defined to nonzero,
`OA_VMA_DEBUG_MARGIN` is defined to nonzero and only for memory types that are
`HOST_VISIBLE` and `HOST_COHERENT`. For more information, see [Corruption detection](@ref debugging_memory_usage_corruption_detection).

Possible return values:

- `VK_ERROR_FEATURE_NOT_PRESENT` - corruption detection is not enabled for any of specified memory types.
- `VK_SUCCESS` - corruption detection has been performed and succeeded.
- `VK_ERROR_UNKNOWN` - corruption detection has been performed and found memory corruptions around one of the allocations.
  `OA_VMA_ASSERT` is also fired in that case.
- Other value: Error returned by Vulkan, e.g. memory mapping failure.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCheckCorruption(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    uint32_t memoryTypeBits);

/** \brief Begins defragmentation process.

\param allocator Allocator object.
\param pInfo Structure filled with parameters of defragmentation.
\param[out] pContext Context object that must be passed to OaVmaEndDefragmentation() to finish defragmentation.
\returns
- `VK_SUCCESS` if defragmentation can begin.
- `VK_ERROR_FEATURE_NOT_PRESENT` if defragmentation is not supported.

For more information about defragmentation, see documentation chapter:
[Defragmentation](@ref defragmentation).
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaBeginDefragmentation(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    const OaVmaDefragmentationInfo* OA_VMA_NOT_NULL pInfo,
    OaVmaDefragmentationContext OA_VMA_NULLABLE* OA_VMA_NOT_NULL pContext);

/** \brief Ends defragmentation process.

\param allocator Allocator object.
\param context Context object that has been created by OaVmaBeginDefragmentation().
\param[out] pStats Optional stats for the defragmentation. Can be null.

Use this function to finish defragmentation started by OaVmaBeginDefragmentation().
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaEndDefragmentation(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaDefragmentationContext OA_VMA_NOT_NULL context,
    OaVmaDefragmentationStats* OA_VMA_NULLABLE pStats);

/** \brief Starts single defragmentation pass.

\param allocator Allocator object.
\param context Context object that has been created by OaVmaBeginDefragmentation().
\param[out] pPassInfo Computed information for current pass.
\returns
- `VK_SUCCESS` if no more moves are possible. Then you can omit call to OaVmaEndDefragmentationPass() and simply end whole defragmentation.
- `VK_INCOMPLETE` if there are pending moves returned in `pPassInfo`. You need to perform them, call OaVmaEndDefragmentationPass(),
  and then preferably try another pass with OaVmaBeginDefragmentationPass().
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaBeginDefragmentationPass(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaDefragmentationContext OA_VMA_NOT_NULL context,
    OaVmaDefragmentationPassMoveInfo* OA_VMA_NOT_NULL pPassInfo);

/** \brief Ends single defragmentation pass.

\param allocator Allocator object.
\param context Context object that has been created by OaVmaBeginDefragmentation().
\param pPassInfo Computed information for current pass filled by OaVmaBeginDefragmentationPass() and possibly modified by you.

Returns `VK_SUCCESS` if no more moves are possible or `VK_INCOMPLETE` if more defragmentations are possible.

Ends incremental defragmentation pass and commits all defragmentation moves from `pPassInfo`.
After this call:

- Allocations at `pPassInfo[i].srcAllocation` that had `pPassInfo[i].operation ==` #OA_VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY
  (which is the default) will be pointing to the new destination place.
- Allocation at `pPassInfo[i].srcAllocation` that had `pPassInfo[i].operation ==` #OA_VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY
  will be freed.

If no more moves are possible you can end whole defragmentation.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaEndDefragmentationPass(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaDefragmentationContext OA_VMA_NOT_NULL context,
    OaVmaDefragmentationPassMoveInfo* OA_VMA_NOT_NULL pPassInfo);

/** \brief Binds buffer to allocation.

Binds specified buffer to region of memory represented by specified allocation.
Gets `VkDeviceMemory` handle and offset from the allocation.
If you want to create a buffer, allocate memory for it and bind them together separately,
you should use this function for binding instead of standard `vkBindBufferMemory()`,
because it ensures proper synchronization so that when a `VkDeviceMemory` object is used by multiple
allocations, calls to `vkBind*Memory()` or `vkMapMemory()` won't happen from multiple threads simultaneously
(which is illegal in Vulkan).

It is recommended to use function OaVmaCreateBuffer() instead of this one.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaBindBufferMemory(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    VkBuffer OA_VMA_NOT_NULL_NON_DISPATCHABLE buffer);

/** \brief Binds buffer to allocation with additional parameters.

\param allocator
\param allocation
\param allocationLocalOffset Additional offset to be added while binding, relative to the beginning of the `allocation`. Normally it should be 0.
\param buffer
\param pNext A chain of structures to be attached to `VkBindBufferMemoryInfoKHR` structure used internally. Normally it should be null.

This function is similar to OaVmaBindBufferMemory(), but it provides additional parameters.

If `pNext` is not null, #OaVmaAllocator object must have been created with #OA_VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT flag
or with OaVmaAllocatorCreateInfo::vulkanApiVersion `>= VK_API_VERSION_1_1`. Otherwise the call fails.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaBindBufferMemory2(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    VkBuffer OA_VMA_NOT_NULL_NON_DISPATCHABLE buffer,
    const void* OA_VMA_NULLABLE OA_VMA_EXTENDS_VK_STRUCT(VkBindBufferMemoryInfoKHR) pNext);

/** \brief Binds image to allocation.

Binds specified image to region of memory represented by specified allocation.
Gets `VkDeviceMemory` handle and offset from the allocation.
If you want to create an image, allocate memory for it and bind them together separately,
you should use this function for binding instead of standard `vkBindImageMemory()`,
because it ensures proper synchronization so that when a `VkDeviceMemory` object is used by multiple
allocations, calls to `vkBind*Memory()` or `vkMapMemory()` won't happen from multiple threads simultaneously
(which is illegal in Vulkan).

It is recommended to use function OaVmaCreateImage() instead of this one.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaBindImageMemory(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    VkImage OA_VMA_NOT_NULL_NON_DISPATCHABLE image);

/** \brief Binds image to allocation with additional parameters.

\param allocator
\param allocation
\param allocationLocalOffset Additional offset to be added while binding, relative to the beginning of the `allocation`. Normally it should be 0.
\param image
\param pNext A chain of structures to be attached to `VkBindImageMemoryInfoKHR` structure used internally. Normally it should be null.

This function is similar to OaVmaBindImageMemory(), but it provides additional parameters.

If `pNext` is not null, #OaVmaAllocator object must have been created with #OA_VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT flag
or with OaVmaAllocatorCreateInfo::vulkanApiVersion `>= VK_API_VERSION_1_1`. Otherwise the call fails.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaBindImageMemory2(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    VkImage OA_VMA_NOT_NULL_NON_DISPATCHABLE image,
    const void* OA_VMA_NULLABLE OA_VMA_EXTENDS_VK_STRUCT(VkBindImageMemoryInfoKHR) pNext);

/** \brief Creates a new `VkBuffer`, allocates and binds memory for it.

\param allocator The main allocator object.
\param pBufferCreateInfo Buffer creation parameters.
\param pAllocationCreateInfo Allocation creation parameters.
\param[out] pBuffer Buffer that was created.
\param[out] pAllocation Allocation that was created.
\param[out] pAllocationInfo Optional, can be null. Information about allocated memory.
    It can be also fetched later using OaVmaGetAllocationInfo().

This function automatically:

-# Creates buffer.
-# Allocates appropriate memory for it.
-# Binds the buffer with the memory.

If any of these operations fail, buffer and allocation are not created,
returned value is negative error code, `*pBuffer` and `*pAllocation` are returned as null.

If the function succeeded, you must destroy both buffer and allocation when you
no longer need them using either convenience function OaVmaDestroyBuffer() or
separately, using `vkDestroyBuffer()` and OaVmaFreeMemory().

If VK_KHR_dedicated_allocation extenion or Vulkan version >= 1.1 is used,
the function queries the driver whether
it requires or prefers the new buffer to have dedicated allocation. If yes,
and if dedicated allocation is possible
(#OA_VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT is not used), it creates dedicated
allocation for this buffer, just like when using
#OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.

\note This function creates a new `VkBuffer`. Sub-allocation of parts of one large buffer,
although recommended as a good practice, is out of scope of this library and could be implemented
by the user as a higher-level logic on top of VMA.

There is also an extended versions of this function available with additional parameter `pMemoryAllocateNext` -
see OaVmaCreateDedicatedBuffer().
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateBuffer(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    const VkBufferCreateInfo* OA_VMA_NOT_NULL pBufferCreateInfo,
    const OaVmaAllocationCreateInfo* OA_VMA_NOT_NULL pAllocationCreateInfo,
    VkBuffer OA_VMA_NULLABLE_NON_DISPATCHABLE* OA_VMA_NOT_NULL pBuffer,
    OaVmaAllocation OA_VMA_NULLABLE* OA_VMA_NOT_NULL pAllocation,
    OaVmaAllocationInfo* OA_VMA_NULLABLE pAllocationInfo);

/** \brief Creates a buffer with additional minimum alignment.

Similar to OaVmaCreateBuffer() but provides additional parameter `minAlignment` which allows to specify custom,
minimum alignment to be used when placing the buffer inside a larger memory block, which may be needed e.g.
for interop with OpenGL.

\deprecated
This function in obsolete since new OaVmaAllocationCreateInfo::minAlignment member allows specifying custom
alignment while using any allocation function, like the standard OaVmaCreateBuffer().
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateBufferWithAlignment(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    const VkBufferCreateInfo* OA_VMA_NOT_NULL pBufferCreateInfo,
    const OaVmaAllocationCreateInfo* OA_VMA_NOT_NULL pAllocationCreateInfo,
    VkDeviceSize minAlignment,
    VkBuffer OA_VMA_NULLABLE_NON_DISPATCHABLE* OA_VMA_NOT_NULL pBuffer,
    OaVmaAllocation OA_VMA_NULLABLE* OA_VMA_NOT_NULL pAllocation,
    OaVmaAllocationInfo* OA_VMA_NULLABLE pAllocationInfo);

/** \brief Creates a dedicated buffer while offering extra parameter `pMemoryAllocateNext`.

This function is similar OaVmaCreateBuffer(), but
it always allocates dedicated memory for the buffer - flag #OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT is implied.
It offers additional parameter `pMemoryAllocateNext`,
which can be used to attach `pNext` chain to the `VkMemoryAllocateInfo` structure.
It can be useful for importing external memory. For more information, see \ref other_api_interop.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateDedicatedBuffer(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    const VkBufferCreateInfo* OA_VMA_NOT_NULL pBufferCreateInfo,
    const OaVmaAllocationCreateInfo* OA_VMA_NOT_NULL pAllocationCreateInfo,
    void* OA_VMA_NULLABLE OA_VMA_EXTENDS_VK_STRUCT(VkMemoryAllocateInfo) pMemoryAllocateNext,
    VkBuffer OA_VMA_NULLABLE_NON_DISPATCHABLE* OA_VMA_NOT_NULL pBuffer,
    OaVmaAllocation OA_VMA_NULLABLE* OA_VMA_NOT_NULL pAllocation,
    OaVmaAllocationInfo* OA_VMA_NULLABLE pAllocationInfo);

/** \brief Creates a new `VkBuffer`, binds already created memory for it.

\param allocator
\param allocation Allocation that provides memory to be used for binding new buffer to it.
\param pBufferCreateInfo
\param[out] pBuffer Buffer that was created.

This function automatically:

-# Creates buffer.
-# Binds the buffer with the supplied memory.

If any of these operations fail, buffer is not created,
returned value is negative error code and `*pBuffer` is null.

If the function succeeded, you must destroy the buffer when you
no longer need it using `vkDestroyBuffer()`. If you want to also destroy the corresponding
allocation you can use convenience function OaVmaDestroyBuffer().

\note There is a new version of this function augmented with parameter `allocationLocalOffset` - see OaVmaCreateAliasingBuffer2().
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateAliasingBuffer(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    const VkBufferCreateInfo* OA_VMA_NOT_NULL pBufferCreateInfo,
    VkBuffer OA_VMA_NULLABLE_NON_DISPATCHABLE* OA_VMA_NOT_NULL pBuffer);

/** \brief Creates a new `VkBuffer`, binds already created memory for it.

\param allocator
\param allocation Allocation that provides memory to be used for binding new buffer to it.
\param allocationLocalOffset Additional offset to be added while binding, relative to the beginning of the allocation. Normally it should be 0.
\param pBufferCreateInfo 
\param[out] pBuffer Buffer that was created.

This function automatically:

-# Creates buffer.
-# Binds the buffer with the supplied memory.

If any of these operations fail, buffer is not created,
returned value is negative error code and `*pBuffer` is null.

If the function succeeded, you must destroy the buffer when you
no longer need it using `vkDestroyBuffer()`. If you want to also destroy the corresponding
allocation you can use convenience function OaVmaDestroyBuffer().

\note This is a new version of the function augmented with parameter `allocationLocalOffset`.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateAliasingBuffer2(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    const VkBufferCreateInfo* OA_VMA_NOT_NULL pBufferCreateInfo,
    VkBuffer OA_VMA_NULLABLE_NON_DISPATCHABLE* OA_VMA_NOT_NULL pBuffer);

/** \brief Destroys Vulkan buffer and frees allocated memory.

This is just a convenience function equivalent to:

\code
vkDestroyBuffer(device, buffer, allocationCallbacks);
OaVmaFreeMemory(allocator, allocation);
\endcode

It is safe to pass null as buffer and/or allocation.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaDestroyBuffer(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    VkBuffer OA_VMA_NULLABLE_NON_DISPATCHABLE buffer,
    OaVmaAllocation OA_VMA_NULLABLE allocation);

/** \brief Function similar to OaVmaCreateBuffer() but for images.

There is also an extended version of this function available: OaVmaCreateDedicatedImage()
which offers additional parameter `pMemoryAllocateNext`.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateImage(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    const VkImageCreateInfo* OA_VMA_NOT_NULL pImageCreateInfo,
    const OaVmaAllocationCreateInfo* OA_VMA_NOT_NULL pAllocationCreateInfo,
    VkImage OA_VMA_NULLABLE_NON_DISPATCHABLE* OA_VMA_NOT_NULL pImage,
    OaVmaAllocation OA_VMA_NULLABLE* OA_VMA_NOT_NULL pAllocation,
    OaVmaAllocationInfo* OA_VMA_NULLABLE pAllocationInfo);

/** \brief Function similar to OaVmaCreateDedicatedBuffer() but for images.

This function is similar OaVmaCreateImage(), but
it always allocates dedicated memory for the image - flag #OA_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT is implied.
It offers additional parameter `pMemoryAllocateNext`,
which can be used to attach `pNext` chain to the `VkMemoryAllocateInfo` structure.
It can be useful for importing external memory. For more information, see \ref other_api_interop.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateDedicatedImage(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    const VkImageCreateInfo* OA_VMA_NOT_NULL pImageCreateInfo,
    const OaVmaAllocationCreateInfo* OA_VMA_NOT_NULL pAllocationCreateInfo,
    void* OA_VMA_NULLABLE OA_VMA_EXTENDS_VK_STRUCT(VkMemoryAllocateInfo) pMemoryAllocateNext,
    VkImage OA_VMA_NULLABLE_NON_DISPATCHABLE* OA_VMA_NOT_NULL pImage,
    OaVmaAllocation OA_VMA_NULLABLE* OA_VMA_NOT_NULL pAllocation,
    OaVmaAllocationInfo* OA_VMA_NULLABLE pAllocationInfo);

/// Function similar to OaVmaCreateAliasingBuffer() but for images.
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateAliasingImage(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    const VkImageCreateInfo* OA_VMA_NOT_NULL pImageCreateInfo,
    VkImage OA_VMA_NULLABLE_NON_DISPATCHABLE* OA_VMA_NOT_NULL pImage);

/// Function similar to OaVmaCreateAliasingBuffer2() but for images.
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateAliasingImage2(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    OaVmaAllocation OA_VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    const VkImageCreateInfo* OA_VMA_NOT_NULL pImageCreateInfo,
    VkImage OA_VMA_NULLABLE_NON_DISPATCHABLE* OA_VMA_NOT_NULL pImage);

/** \brief Destroys Vulkan image and frees allocated memory.

This is just a convenience function equivalent to:

\code
vkDestroyImage(device, image, allocationCallbacks);
OaVmaFreeMemory(allocator, allocation);
\endcode

It is safe to pass null as image and/or allocation.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaDestroyImage(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    VkImage OA_VMA_NULLABLE_NON_DISPATCHABLE image,
    OaVmaAllocation OA_VMA_NULLABLE allocation);

/** @} */

/**
\addtogroup group_virtual
@{
*/

/** \brief Creates new #OaVmaVirtualBlock object.

\param pCreateInfo Parameters for creation.
\param[out] pVirtualBlock Returned virtual block object or `OA_VMA_NULL` if creation failed.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaCreateVirtualBlock(
    const OaVmaVirtualBlockCreateInfo* OA_VMA_NOT_NULL pCreateInfo,
    OaVmaVirtualBlock OA_VMA_NULLABLE* OA_VMA_NOT_NULL pVirtualBlock);

/** \brief Destroys #OaVmaVirtualBlock object.

Please note that you should consciously handle virtual allocations that could remain unfreed in the block.
You should either free them individually using OaVmaVirtualFree() or call OaVmaClearVirtualBlock()
if you are sure this is what you want. If you do neither, an assert is called.

If you keep pointers to some additional metadata associated with your virtual allocations in their `pUserData`,
don't forget to free them.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaDestroyVirtualBlock(
    OaVmaVirtualBlock OA_VMA_NULLABLE virtualBlock);

/** \brief Returns true of the #OaVmaVirtualBlock is empty - contains 0 virtual allocations and has all its space available for new allocations.
*/
OA_VMA_CALL_PRE VkBool32 OA_VMA_CALL_POST OaVmaIsVirtualBlockEmpty(
    OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock);

/** \brief Returns information about a specific virtual allocation within a virtual block, like its size and `pUserData` pointer.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetVirtualAllocationInfo(
    OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock,
    OaVmaVirtualAllocation OA_VMA_NOT_NULL_NON_DISPATCHABLE allocation, OaVmaVirtualAllocationInfo* OA_VMA_NOT_NULL pVirtualAllocInfo);

/** \brief Allocates new virtual allocation inside given #OaVmaVirtualBlock.

If the allocation fails due to not enough free space available, `VK_ERROR_OUT_OF_DEVICE_MEMORY` is returned
(despite the function doesn't ever allocate actual GPU memory).
`pAllocation` is then set to `VK_NULL_HANDLE` and `pOffset`, if not null, it set to `UINT64_MAX`.

\param virtualBlock Virtual block
\param pCreateInfo Parameters for the allocation
\param[out] pAllocation Returned handle of the new allocation
\param[out] pOffset Returned offset of the new allocation. Optional, can be null.
*/
OA_VMA_CALL_PRE VkResult OA_VMA_CALL_POST OaVmaVirtualAllocate(
    OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock,
    const OaVmaVirtualAllocationCreateInfo* OA_VMA_NOT_NULL pCreateInfo,
    OaVmaVirtualAllocation OA_VMA_NULLABLE_NON_DISPATCHABLE* OA_VMA_NOT_NULL pAllocation,
    VkDeviceSize* OA_VMA_NULLABLE pOffset);

/** \brief Frees virtual allocation inside given #OaVmaVirtualBlock.

It is correct to call this function with `allocation == VK_NULL_HANDLE` - it does nothing.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaVirtualFree(
    OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock,
    OaVmaVirtualAllocation OA_VMA_NULLABLE_NON_DISPATCHABLE allocation);

/** \brief Frees all virtual allocations inside given #OaVmaVirtualBlock.

You must either call this function or free each virtual allocation individually with OaVmaVirtualFree()
before destroying a virtual block. Otherwise, an assert is called.

If you keep pointer to some additional metadata associated with your virtual allocation in its `pUserData`,
don't forget to free it as well.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaClearVirtualBlock(
    OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock);

/** \brief Changes custom pointer associated with given virtual allocation.
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaSetVirtualAllocationUserData(
    OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock,
    OaVmaVirtualAllocation OA_VMA_NOT_NULL_NON_DISPATCHABLE allocation,
    void* OA_VMA_NULLABLE pUserData);

/** \brief Calculates and returns statistics about virtual allocations and memory usage in given #OaVmaVirtualBlock.

This function is fast to call. For more detailed statistics, see OaVmaCalculateVirtualBlockStatistics().
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaGetVirtualBlockStatistics(
    OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock,
    OaVmaStatistics* OA_VMA_NOT_NULL pStats);

/** \brief Calculates and returns detailed statistics about virtual allocations and memory usage in given #OaVmaVirtualBlock.

This function is slow to call. Use for debugging purposes.
For less detailed statistics, see OaVmaGetVirtualBlockStatistics().
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaCalculateVirtualBlockStatistics(
    OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock,
    OaVmaDetailedStatistics* OA_VMA_NOT_NULL pStats);

/** @} */

#if OA_VMA_STATS_STRING_ENABLED
/**
\addtogroup group_stats
@{
*/

/** \brief Builds and returns a null-terminated string in JSON format with information about given #OaVmaVirtualBlock.
\param virtualBlock Virtual block.
\param[out] ppStatsString Returned string.
\param detailedMap Pass `VK_FALSE` to only obtain statistics as returned by OaVmaCalculateVirtualBlockStatistics(). Pass `VK_TRUE` to also obtain full list of allocations and free spaces.

Returned string must be freed using OaVmaFreeVirtualBlockStatsString().
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaBuildVirtualBlockStatsString(
    OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock,
    char* OA_VMA_NULLABLE* OA_VMA_NOT_NULL ppStatsString,
    VkBool32 detailedMap);

/// Frees a string returned by OaVmaBuildVirtualBlockStatsString().
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaFreeVirtualBlockStatsString(
    OaVmaVirtualBlock OA_VMA_NOT_NULL virtualBlock,
    char* OA_VMA_NULLABLE pStatsString);

/** \brief Builds and returns statistics as a null-terminated string in JSON format.
\param allocator
\param[out] ppStatsString Must be freed using OaVmaFreeStatsString() function.
\param detailedMap
*/
OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaBuildStatsString(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    char* OA_VMA_NULLABLE* OA_VMA_NOT_NULL ppStatsString,
    VkBool32 detailedMap);

OA_VMA_CALL_PRE void OA_VMA_CALL_POST OaVmaFreeStatsString(
    OaVmaAllocator OA_VMA_NOT_NULL allocator,
    char* OA_VMA_NULLABLE pStatsString);

/** @} */

#endif // OA_VMA_STATS_STRING_ENABLED

#endif // _OA_VMA_FUNCTION_HEADERS

#ifdef __cplusplus
}
#endif
