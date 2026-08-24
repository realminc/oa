// SPDX: MIT. Derived from Vulkan Memory Allocator, (c) 2017-2026 Advanced Micro Devices, Inc.
// Full MIT copyright/permission notice: source/cpp/lib/oa/runtime/oaVma.h. See NOTICE.md.
// oa::VkSlab — bitmap slab allocator
//
// 64 slots per slab, oa::U64 bitmap.
// allocSlot: TZCNT + bit clear = ~2ns.
// freeSlot: OR = ~1ns.
// Backed by a single contiguous allocation.

#pragma once

#include <oa/core/types.h>
#include <bit>

namespace oa {

struct VkSlab {
	U64 freeBitmap = 0;
	U64 slotSize = 0;
	U8* basePtr = nullptr;
	U64 totalSize = 0;
	U32 capacity = 0;

	void init(U8* inBase, U64 inSlotSize, U32 inCapacity) {
		basePtr = inBase;
		slotSize = inSlotSize;
		capacity = inCapacity > 64 ? 64 : inCapacity;
		totalSize = slotSize * capacity;
		freeBitmap = capacity == 64 ? ~U64(0) : ((U64(1) << capacity) - 1);
	}

	[[nodiscard]] OA_FORCEINLINE bool isFull() const {
		return freeBitmap == 0;
	}

	[[nodiscard]] OA_FORCEINLINE bool isEmpty() const {
		U64 allFree = capacity == 64 ? ~U64(0) : ((U64(1) << capacity) - 1);
		return freeBitmap == allFree;
	}

	[[nodiscard]] OA_FORCEINLINE U32 freeCount() const {
		return static_cast<U32>(std::popcount(freeBitmap));
	}

	[[nodiscard]] OA_FORCEINLINE U32 usedCount() const {
		return capacity - freeCount();
	}

	// Returns slot index, or 64 if full
	[[nodiscard]] OA_FORCEINLINE U32 allocSlot() {
		if (OA_UNLIKELY(freeBitmap == 0)) return 64;
		const U32 slot = static_cast<U32>(std::countr_zero(freeBitmap));
		freeBitmap &= freeBitmap - 1;
		return slot;
	}

	OA_FORCEINLINE void freeSlot(U32 inSlot) {
		freeBitmap |= (U64(1) << inSlot);
	}

	[[nodiscard]] OA_FORCEINLINE void* slotPtr(U32 inSlot) const {
		return basePtr + (static_cast<U64>(inSlot) * slotSize);
	}

	// Allocate and return pointer (or nullptr if full)
	[[nodiscard]] OA_FORCEINLINE void* alloc() {
		U32 slot = allocSlot();
		if (OA_UNLIKELY(slot >= capacity)) return nullptr;
		return slotPtr(slot);
	}

	// Free by pointer (must belong to this slab)
	OA_FORCEINLINE void free(void* inPtr) {
		Usize offset = static_cast<Usize>(static_cast<U8*>(inPtr) - basePtr);
		U32 slot = static_cast<U32>(offset / slotSize);
		freeSlot(slot);
	}
};

} // namespace oa
