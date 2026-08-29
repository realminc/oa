#pragma once

// OA's C++ boundary for the vendored Vulkan Memory Allocator fork.
//
// vma.h retains mechanically traceable upstream-derived storage types and
// private implementation declarations. Product code uses this namespace
// instead of depending on fork prefixes or preprocessor constants.

#include "vma.h"

#include <oa/core/std/assert.h>

namespace vma {

using Allocator = VmaAllocator;
using Allocation = VmaAllocation;
using AllocatorCreateFlags = VmaAllocatorCreateFlags;
using AllocationCreateFlags = VmaAllocationCreateFlags;
using MemoryUsage = VmaMemoryUsage;
using VulkanFunctions = VmaVulkanFunctions;
using AllocatorCreateInfo = VmaAllocatorCreateInfo;
using AllocationCreateInfo = VmaAllocationCreateInfo;
using AllocationInfo = VmaAllocationInfo;
using Budget = VmaBudget;
using TotalStatistics = VmaTotalStatistics;
using Statistics = VmaStatistics;
using DetailedStatistics = VmaDetailedStatistics;
using VirtualBlock = VmaVirtualBlock;
using VirtualAllocation = VmaVirtualAllocation;
using VirtualBlockCreateFlags = VmaVirtualBlockCreateFlags;
using VirtualAllocationCreateFlags = VmaVirtualAllocationCreateFlags;
using VirtualBlockCreateInfo = VmaVirtualBlockCreateInfo;
using VirtualAllocationCreateInfo = VmaVirtualAllocationCreateInfo;
using VirtualAllocationInfo = VmaVirtualAllocationInfo;

inline constexpr AllocatorCreateFlags allocatorCreateBufferDeviceAddressBit =
	VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
inline constexpr AllocatorCreateFlags allocatorCreateKhrMaintenance5Bit =
	VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT;

inline constexpr AllocationCreateFlags allocationCreateCanAliasBit =
	VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT;
inline constexpr AllocationCreateFlags allocationCreateMappedBit =
	VMA_ALLOCATION_CREATE_MAPPED_BIT;
inline constexpr AllocationCreateFlags allocationCreateHostAccessSequentialWriteBit =
	VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
inline constexpr AllocationCreateFlags allocationCreateHostAccessRandomBit =
	VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

inline constexpr MemoryUsage memoryUsageAuto = VMA_MEMORY_USAGE_AUTO;
inline constexpr MemoryUsage memoryUsageCpuOnly = VMA_MEMORY_USAGE_CPU_ONLY;
inline constexpr MemoryUsage memoryUsageCpuToGpu = VMA_MEMORY_USAGE_CPU_TO_GPU;
inline constexpr MemoryUsage memoryUsageGpuOnly = VMA_MEMORY_USAGE_GPU_ONLY;
inline constexpr MemoryUsage memoryUsageGpuToCpu = VMA_MEMORY_USAGE_GPU_TO_CPU;

namespace facadeDetail {
[[nodiscard]] constexpr bool isPowerOfTwo(VkDeviceSize inValue) noexcept {
	return inValue != 0U and (inValue & (inValue - 1U)) == 0U;
}
} // namespace facadeDetail

[[nodiscard]] inline VkResult createVirtualBlock(
	const VirtualBlockCreateInfo* inCreateInfo,
	VirtualBlock* outVirtualBlock
) noexcept {
	constexpr VirtualBlockCreateFlags knownFlags =
		VMA_VIRTUAL_BLOCK_CREATE_ALGORITHM_MASK;
	OA_REQUIRE(inCreateInfo != nullptr
		and outVirtualBlock != nullptr
		and inCreateInfo->size > 0U
		and (inCreateInfo->flags & ~knownFlags) == 0U);
	return detail::vmaCreateVirtualBlock(inCreateInfo, outVirtualBlock);
}

inline void destroyVirtualBlock(VirtualBlock inVirtualBlock) noexcept {
	detail::vmaDestroyVirtualBlock(inVirtualBlock);
}

[[nodiscard]] inline bool isVirtualBlockEmpty(
	VirtualBlock inVirtualBlock
) noexcept {
	OA_REQUIRE(inVirtualBlock != VK_NULL_HANDLE);
	return detail::vmaIsVirtualBlockEmpty(inVirtualBlock) != VK_FALSE;
}

[[nodiscard]] inline VkResult virtualAllocate(
	VirtualBlock inVirtualBlock,
	const VirtualAllocationCreateInfo* inCreateInfo,
	VirtualAllocation* outAllocation,
	VkDeviceSize* outOffset = nullptr
) noexcept {
	constexpr VirtualAllocationCreateFlags knownFlags =
		VMA_VIRTUAL_ALLOCATION_CREATE_UPPER_ADDRESS_BIT
		| VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MASK;
	OA_REQUIRE(inVirtualBlock != VK_NULL_HANDLE
		and inCreateInfo != nullptr
		and outAllocation != nullptr
		and inCreateInfo->size > 0U
		and (inCreateInfo->alignment == 0U
			or facadeDetail::isPowerOfTwo(inCreateInfo->alignment))
		and (inCreateInfo->flags & ~knownFlags) == 0U);
	return detail::vmaVirtualAllocate(
		inVirtualBlock, inCreateInfo, outAllocation, outOffset);
}

inline void virtualFree(
	VirtualBlock inVirtualBlock,
	VirtualAllocation inAllocation
) noexcept {
	if (inAllocation == VK_NULL_HANDLE) return;
	OA_REQUIRE(inVirtualBlock != VK_NULL_HANDLE);
	detail::vmaVirtualFree(inVirtualBlock, inAllocation);
}

inline void clearVirtualBlock(VirtualBlock inVirtualBlock) noexcept {
	OA_REQUIRE(inVirtualBlock != VK_NULL_HANDLE);
	detail::vmaClearVirtualBlock(inVirtualBlock);
}

inline void getVirtualBlockStatistics(
	VirtualBlock inVirtualBlock,
	Statistics* outStatistics
) noexcept {
	OA_REQUIRE(inVirtualBlock != VK_NULL_HANDLE and outStatistics != nullptr);
	detail::vmaGetVirtualBlockStatistics(inVirtualBlock, outStatistics);
}

inline void calculateVirtualBlockStatistics(
	VirtualBlock inVirtualBlock,
	DetailedStatistics* outStatistics
) noexcept {
	OA_REQUIRE(inVirtualBlock != VK_NULL_HANDLE and outStatistics != nullptr);
	detail::vmaCalculateVirtualBlockStatistics(inVirtualBlock, outStatistics);
}

[[nodiscard]] inline VkResult createAllocator(
	const AllocatorCreateInfo* inCreateInfo,
	Allocator* outAllocator
) noexcept {
	OA_REQUIRE(inCreateInfo != nullptr
		and outAllocator != nullptr
		and inCreateInfo->instance != VK_NULL_HANDLE
		and inCreateInfo->physicalDevice != VK_NULL_HANDLE
		and inCreateInfo->device != VK_NULL_HANDLE
		and inCreateInfo->pVulkanFunctions != nullptr);
	return detail::vmaCreateAllocator(inCreateInfo, outAllocator);
}

inline void destroyAllocator(Allocator inAllocator) noexcept {
	detail::vmaDestroyAllocator(inAllocator);
}

inline void calculateStatistics(
	Allocator inAllocator,
	TotalStatistics* outStatistics
) noexcept {
	OA_REQUIRE(inAllocator != VK_NULL_HANDLE);
	OA_REQUIRE(outStatistics != nullptr);
	detail::vmaCalculateStatistics(inAllocator, outStatistics);
}

inline void getHeapBudgets(Allocator inAllocator, Budget* outBudgets) noexcept {
	OA_REQUIRE(inAllocator != VK_NULL_HANDLE);
	OA_REQUIRE(outBudgets != nullptr);
	detail::vmaGetHeapBudgets(inAllocator, outBudgets);
}

[[nodiscard]] inline VkResult allocateMemory(
	Allocator inAllocator,
	const VkMemoryRequirements* inMemoryRequirements,
	const AllocationCreateInfo* inCreateInfo,
	Allocation* outAllocation,
	AllocationInfo* outAllocationInfo = nullptr
) noexcept {
	OA_REQUIRE(inAllocator != VK_NULL_HANDLE);
	OA_REQUIRE(inMemoryRequirements != nullptr);
	OA_REQUIRE(inCreateInfo != nullptr);
	OA_REQUIRE(outAllocation != nullptr);
	return detail::vmaAllocateMemory(
		inAllocator,
		inMemoryRequirements,
		inCreateInfo,
		outAllocation,
		outAllocationInfo);
}

inline void freeMemory(Allocator inAllocator, Allocation inAllocation) noexcept {
	OA_REQUIRE(inAllocator != VK_NULL_HANDLE);
	detail::vmaFreeMemory(inAllocator, inAllocation);
}

[[nodiscard]] inline VkResult mapMemory(
	Allocator inAllocator,
	Allocation inAllocation,
	void** outData
) noexcept {
	OA_REQUIRE(inAllocator != VK_NULL_HANDLE);
	OA_REQUIRE(inAllocation != VK_NULL_HANDLE);
	OA_REQUIRE(outData != nullptr);
	return detail::vmaMapMemory(inAllocator, inAllocation, outData);
}

inline void unmapMemory(Allocator inAllocator, Allocation inAllocation) noexcept {
	OA_REQUIRE(inAllocator != VK_NULL_HANDLE);
	OA_REQUIRE(inAllocation != VK_NULL_HANDLE);
	detail::vmaUnmapMemory(inAllocator, inAllocation);
}

[[nodiscard]] inline VkResult flushAllocation(
	Allocator inAllocator,
	Allocation inAllocation,
	VkDeviceSize inOffset,
	VkDeviceSize inSize
) noexcept {
	OA_REQUIRE(inAllocator != VK_NULL_HANDLE);
	OA_REQUIRE(inAllocation != VK_NULL_HANDLE);
	return detail::vmaFlushAllocation(inAllocator, inAllocation, inOffset, inSize);
}

[[nodiscard]] inline VkResult invalidateAllocation(
	Allocator inAllocator,
	Allocation inAllocation,
	VkDeviceSize inOffset,
	VkDeviceSize inSize
) noexcept {
	OA_REQUIRE(inAllocator != VK_NULL_HANDLE);
	OA_REQUIRE(inAllocation != VK_NULL_HANDLE);
	return detail::vmaInvalidateAllocation(inAllocator, inAllocation, inOffset, inSize);
}

[[nodiscard]] inline VkResult createBuffer(
	Allocator inAllocator,
	const VkBufferCreateInfo* inBufferCreateInfo,
	const AllocationCreateInfo* inAllocationCreateInfo,
	VkBuffer* outBuffer,
	Allocation* outAllocation,
	AllocationInfo* outAllocationInfo = nullptr
) noexcept {
	OA_REQUIRE(inAllocator != VK_NULL_HANDLE);
	OA_REQUIRE(inBufferCreateInfo != nullptr);
	OA_REQUIRE(inAllocationCreateInfo != nullptr);
	OA_REQUIRE(outBuffer != nullptr);
	OA_REQUIRE(outAllocation != nullptr);
	return detail::vmaCreateBuffer(
		inAllocator,
		inBufferCreateInfo,
		inAllocationCreateInfo,
		outBuffer,
		outAllocation,
		outAllocationInfo);
}

[[nodiscard]] inline VkResult createAliasingBuffer(
	Allocator inAllocator,
	Allocation inAllocation,
	VkDeviceSize inAllocationLocalOffset,
	const VkBufferCreateInfo* inBufferCreateInfo,
	VkBuffer* outBuffer
) noexcept {
	OA_REQUIRE(inAllocator != VK_NULL_HANDLE);
	OA_REQUIRE(inAllocation != VK_NULL_HANDLE);
	OA_REQUIRE(inBufferCreateInfo != nullptr);
	OA_REQUIRE(outBuffer != nullptr);
	return detail::vmaCreateAliasingBuffer2(
		inAllocator,
		inAllocation,
		inAllocationLocalOffset,
		inBufferCreateInfo,
		outBuffer);
}

inline void destroyBuffer(
	Allocator inAllocator,
	VkBuffer inBuffer,
	Allocation inAllocation
) noexcept {
	OA_REQUIRE(inAllocator != VK_NULL_HANDLE);
	detail::vmaDestroyBuffer(inAllocator, inBuffer, inAllocation);
}

[[nodiscard]] inline VkResult createImage(
	Allocator inAllocator,
	const VkImageCreateInfo* inImageCreateInfo,
	const AllocationCreateInfo* inAllocationCreateInfo,
	VkImage* outImage,
	Allocation* outAllocation,
	AllocationInfo* outAllocationInfo = nullptr
) noexcept {
	OA_REQUIRE(inAllocator != VK_NULL_HANDLE);
	OA_REQUIRE(inImageCreateInfo != nullptr);
	OA_REQUIRE(inAllocationCreateInfo != nullptr);
	OA_REQUIRE(outImage != nullptr);
	OA_REQUIRE(outAllocation != nullptr);
	return detail::vmaCreateImage(
		inAllocator,
		inImageCreateInfo,
		inAllocationCreateInfo,
		outImage,
		outAllocation,
		outAllocationInfo);
}

inline void destroyImage(
	Allocator inAllocator,
	VkImage inImage,
	Allocation inAllocation
) noexcept {
	OA_REQUIRE(inAllocator != VK_NULL_HANDLE);
	detail::vmaDestroyImage(inAllocator, inImage, inAllocation);
}

} // namespace vma
