#include <gtest/gtest.h>

#include <vma/vma.hpp>

#include <oa/runtime/allocator.h>
#include <oa/core/std/limits.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

struct LiveAllocation {
	vma::VirtualAllocation allocation = VK_NULL_HANDLE;
	VkDeviceSize offset = 0U;
	VkDeviceSize size = 0U;
};

std::uint64_t nextRandom(std::uint64_t& inOutState) {
	inOutState ^= inOutState >> 12U;
	inOutState ^= inOutState << 25U;
	inOutState ^= inOutState >> 27U;
	return inOutState * 0x2545F4914F6CDD1DULL;
}

void expectStatistics(
	vma::VirtualBlock inBlock,
	const std::vector<LiveAllocation>& inLive
) {
	VkDeviceSize bytes = 0U;
	for (const LiveAllocation& allocation : inLive) bytes += allocation.size;
	vma::Statistics statistics{};
	vma::getVirtualBlockStatistics(inBlock, &statistics);
	EXPECT_EQ(statistics.allocationCount, inLive.size());
	EXPECT_EQ(statistics.allocationBytes, bytes);
}

} // namespace

TEST(VmaVirtual, DeterministicFragmentationModel) {
	constexpr VkDeviceSize blockSize = 2U * 1024U * 1024U;
	const vma::VirtualBlockCreateInfo blockInfo{.size = blockSize};
	vma::VirtualBlock block = VK_NULL_HANDLE;
	ASSERT_EQ(vma::createVirtualBlock(&blockInfo, &block), VK_SUCCESS);
	ASSERT_NE(block, VK_NULL_HANDLE);

	std::vector<LiveAllocation> live;
	live.reserve(2048U);
	std::uint64_t state = 0xD1B54A32D192ED03ULL;
	for (std::size_t operation = 0; operation < 50000U; ++operation) {
		const std::uint64_t random = nextRandom(state);
		const bool allocate = live.empty()
			or (live.size() < 1536U and (random & 3U) != 0U);
		if (allocate) {
			const VkDeviceSize size = 1U + (random % 8192U);
			const VkDeviceSize alignment = VkDeviceSize{1}
				<< static_cast<unsigned>((random >> 17U) & 8U);
			const vma::VirtualAllocationCreateInfo info{
				.size = size,
				.alignment = alignment,
			};
			vma::VirtualAllocation allocation = VK_NULL_HANDLE;
			VkDeviceSize offset = 0U;
			const VkResult result =
				vma::virtualAllocate(block, &info, &allocation, &offset);
			if (result == VK_SUCCESS) {
				ASSERT_NE(allocation, VK_NULL_HANDLE);
				ASSERT_EQ(offset & (alignment - 1U), 0U);
				ASSERT_LE(size, blockSize);
				ASSERT_LE(offset, blockSize - size);
				for (const LiveAllocation& other : live) {
					EXPECT_TRUE(offset + size <= other.offset
						or other.offset + other.size <= offset);
				}
				live.push_back({allocation, offset, size});
			} else {
				EXPECT_EQ(result, VK_ERROR_OUT_OF_DEVICE_MEMORY);
				EXPECT_EQ(allocation, VK_NULL_HANDLE);
				EXPECT_EQ(offset, std::numeric_limits<VkDeviceSize>::max());
			}
		} else {
			const std::size_t index = static_cast<std::size_t>(
				random % static_cast<std::uint64_t>(live.size()));
			vma::virtualFree(block, live[index].allocation);
			live[index] = live.back();
			live.pop_back();
		}
		if ((operation & 255U) == 0U) expectStatistics(block, live);
	}

	expectStatistics(block, live);
	vma::clearVirtualBlock(block);
	EXPECT_TRUE(vma::isVirtualBlockEmpty(block));
	vma::destroyVirtualBlock(block);
}

TEST(VmaVirtual, OversizedAllocationFailsWithoutStateChange) {
	const vma::VirtualBlockCreateInfo blockInfo{.size = 4096U};
	vma::VirtualBlock block = VK_NULL_HANDLE;
	ASSERT_EQ(vma::createVirtualBlock(&blockInfo, &block), VK_SUCCESS);
	const vma::VirtualAllocationCreateInfo allocationInfo{
		.size = 4097U,
		.alignment = 256U,
	};
	vma::VirtualAllocation allocation =
		reinterpret_cast<vma::VirtualAllocation>(std::uintptr_t{1U});
	VkDeviceSize offset = 0U;
	EXPECT_EQ(vma::virtualAllocate(block, &allocationInfo, &allocation, &offset),
		VK_ERROR_OUT_OF_DEVICE_MEMORY);
	EXPECT_EQ(allocation, VK_NULL_HANDLE);
	EXPECT_EQ(offset, std::numeric_limits<VkDeviceSize>::max());
	EXPECT_TRUE(vma::isVirtualBlockEmpty(block));
	vma::destroyVirtualBlock(block);
}

TEST(VmaFacadeDeath, RejectsInvalidVirtualContracts) {
	vma::VirtualBlock block = VK_NULL_HANDLE;
	vma::VirtualBlockCreateInfo blockInfo{.size = 4096U};
	EXPECT_DEATH(
		static_cast<void>(vma::createVirtualBlock(nullptr, &block)),
		"OA contract failed");
	blockInfo.size = 0U;
	EXPECT_DEATH(
		static_cast<void>(vma::createVirtualBlock(&blockInfo, &block)),
		"OA contract failed");

	blockInfo.size = 4096U;
	ASSERT_EQ(vma::createVirtualBlock(&blockInfo, &block), VK_SUCCESS);
	vma::VirtualAllocation allocation = VK_NULL_HANDLE;
	vma::VirtualAllocationCreateInfo allocationInfo{
		.size = 0U,
		.alignment = 1U,
	};
	EXPECT_DEATH(
		static_cast<void>(
			vma::virtualAllocate(block, &allocationInfo, &allocation)),
		"OA contract failed");
	allocationInfo.size = 16U;
	allocationInfo.alignment = 3U;
	EXPECT_DEATH(
		static_cast<void>(
			vma::virtualAllocate(block, &allocationInfo, &allocation)),
		"OA contract failed");
	vma::virtualFree(VK_NULL_HANDLE, VK_NULL_HANDLE);
	vma::destroyVirtualBlock(block);
}

TEST(RuntimeAllocatorSecurity, RejectsCapacityOverflowBeforeVulkan) {
	RuntimeAllocator allocator;
	const oa::U64 impossible = oa::Limits<oa::U64>::max();
	const auto device = allocator.allocDevice(impossible);
	EXPECT_FALSE(device.isOk());
	EXPECT_EQ(device.getStatus().getCode(), oa::StatusCode::OutOfRange);
	const auto alias = allocator.allocAliased(
		impossible, oa::MemoryPlacement::DeviceLocal);
	EXPECT_FALSE(alias.isOk());
	EXPECT_EQ(alias.getStatus().getCode(), oa::StatusCode::OutOfRange);
}

TEST(RuntimeAllocatorSecurity, RejectsInvalidHostAndOwnershipRanges) {
	RuntimeAllocator allocator;
	allocator.allocator = reinterpret_cast<void*>(std::uintptr_t{1U});
	oavk::Buffer buffer;
	buffer.buffer = reinterpret_cast<void*>(std::uintptr_t{2U});
	buffer.allocation = reinterpret_cast<void*>(std::uintptr_t{3U});
	buffer.allocatorIdentity = allocator.allocator;
	buffer.capacity = 16U;
	EXPECT_FALSE(allocator.flushHostBuffer(buffer, 17U, 0U));
	EXPECT_FALSE(allocator.invalidateHostBuffer(buffer, 8U, 9U));
	EXPECT_EQ(
		allocator.uploadWeights(buffer, nullptr, 1U).getCode(),
		oa::StatusCode::InvalidArgument);

	buffer.allocatorIdentity = reinterpret_cast<void*>(std::uintptr_t{4U});
	EXPECT_DEATH(allocator.free(buffer), "buffer belongs to a different allocator");
}
