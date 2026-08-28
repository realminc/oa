#include <gtest/gtest.h>

// Compile the private derived implementation into this test so allocation
// failure and internal POD-container contracts can be exercised directly.
#include <vma/vma.cpp>

#include <oa/core/std/allocator.h>

#include <cstddef>
#include <cstdint>

namespace {

struct CallbackState {
	std::size_t allocations = 0;
	std::size_t frees = 0;
};

void* VKAPI_PTR trackedAllocation(
	void* inUserData,
	std::size_t inSize,
	std::size_t inAlignment,
	VkSystemAllocationScope /*inScope*/
) {
	auto* state = static_cast<CallbackState*>(inUserData);
	++state->allocations;
	return oa::allocBytes(inSize, inAlignment);
}

void VKAPI_PTR trackedFree(void* inUserData, void* inMemory) {
	auto* state = static_cast<CallbackState*>(inUserData);
	if (inMemory != nullptr) {
		++state->frees;
	}
	oa::freeBytes(inMemory);
}

void* VKAPI_PTR rejectedAllocation(
	void* /*inUserData*/,
	std::size_t /*inSize*/,
	std::size_t /*inAlignment*/,
	VkSystemAllocationScope /*inScope*/
) {
	return nullptr;
}

} // namespace

TEST(VmaInternalMemory, ContainersPreserveEveryValueAndCallbacks) {
	CallbackState state;
	const VkAllocationCallbacks callbacks{
		.pUserData = &state,
		.pfnAllocation = trackedAllocation,
		.pfnReallocation = nullptr,
		.pfnFree = trackedFree,
		.pfnInternalAllocation = nullptr,
		.pfnInternalFree = nullptr,
	};
	using Allocator = VmaStlAllocator<std::uint64_t>;
	const Allocator allocator(&callbacks);
	{
		VmaVector<std::uint64_t, Allocator> values(allocator);
		for (std::size_t index = 0; index < 4096U; ++index) {
			values.push_back(
				static_cast<std::uint64_t>(index) * 0x9E3779B185EBCA87ULL);
		}
		ASSERT_EQ(values.size(), 4096U);
		for (std::size_t index = 0; index < values.size(); ++index) {
			EXPECT_EQ(values[index],
				static_cast<std::uint64_t>(index) * 0x9E3779B185EBCA87ULL);
		}

		const VmaVector<std::uint64_t, Allocator> copied(values);
		ASSERT_EQ(copied.size(), values.size());
		for (std::size_t index = 0; index < copied.size(); ++index) {
			EXPECT_EQ(copied[index], values[index]);
		}

		VmaVector<std::uint64_t, Allocator> aliases(allocator);
		for (std::size_t index = 0; index < 8U; ++index) {
			aliases.push_back(100U + index);
		}
		aliases.push_back(aliases[0]);
		EXPECT_EQ(aliases.back(), 100U);
		aliases.insert(1U, aliases[4]);
		ASSERT_EQ(aliases.size(), 10U);
		EXPECT_EQ(aliases[1], 104U);

		VmaSmallVector<std::uint64_t, Allocator, 16U> small(allocator);
		for (std::size_t index = 0; index < 33U; ++index) {
			small.push_back(0xD1B54A32D192ED03ULL + index);
		}
		small.resize(16U, true);
		ASSERT_EQ(small.size(), 16U);
		for (std::size_t index = 0; index < small.size(); ++index) {
			EXPECT_EQ(small[index], 0xD1B54A32D192ED03ULL + index);
		}
		small.insert(0U, small[8]);
		EXPECT_EQ(small.front(), 0xD1B54A32D192ED03ULL + 8U);
		small.insert(1U, small[8]);
		EXPECT_EQ(small[1], 0xD1B54A32D192ED03ULL + 7U);
	}
	EXPECT_GT(state.allocations, 0U);
	EXPECT_EQ(state.allocations, state.frees);
}

TEST(VmaInternalMemoryDeath, RejectsInvalidAndOverflowingHostAllocations) {
	EXPECT_DEATH(
		static_cast<void>(VmaMalloc(
			static_cast<const VkAllocationCallbacks*>(nullptr), 64U, 3U)),
		"OA contract failed");
	EXPECT_DEATH(
		static_cast<void>(VmaAllocateArray<std::uint64_t>(
			static_cast<const VkAllocationCallbacks*>(nullptr),
			static_cast<std::size_t>(-1) / sizeof(std::uint64_t) + 1U)),
		"OA contract failed");

	const VkAllocationCallbacks callbacks{
		.pUserData = nullptr,
		.pfnAllocation = rejectedAllocation,
		.pfnReallocation = nullptr,
		.pfnFree = nullptr,
		.pfnInternalAllocation = nullptr,
		.pfnInternalFree = nullptr,
	};
	EXPECT_DEATH(
		static_cast<void>(VmaMalloc(&callbacks, 64U, 8U)),
		"OA contract failed");
}
