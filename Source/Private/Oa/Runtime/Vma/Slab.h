// SPDX: MIT. Derived from Vulkan Memory Allocator, (c) 2017-2026 Advanced Micro Devices, Inc.
// Full MIT copyright/permission notice: Source/Public/Oa/Runtime/OaVma.h. See NOTICE.md.
// OaVkSlab — Bitmap slab allocator
//
// 64 slots per slab, OaU64 bitmap.
// AllocSlot: TZCNT + bit clear = ~2ns.
// FreeSlot: OR = ~1ns.
// Backed by a single contiguous allocation.

#pragma once

#include <Oa/Core/Types.h>
#include <bit>

struct OaVkSlab {
	OaU64 FreeBitmap = 0;
	OaU64 SlotSize = 0;
	OaU8* BasePtr = nullptr;
	OaU64 TotalSize = 0;
	OaU32 Capacity = 0;

	void Init(OaU8* InBase, OaU64 InSlotSize, OaU32 InCapacity) {
		BasePtr = InBase;
		SlotSize = InSlotSize;
		Capacity = InCapacity > 64 ? 64 : InCapacity;
		TotalSize = SlotSize * Capacity;
		FreeBitmap = Capacity == 64 ? ~OaU64(0) : ((OaU64(1) << Capacity) - 1);
	}

	[[nodiscard]] OA_FORCEINLINE bool IsFull() const {
		return FreeBitmap == 0;
	}

	[[nodiscard]] OA_FORCEINLINE bool IsEmpty() const {
		OaU64 allFree = Capacity == 64 ? ~OaU64(0) : ((OaU64(1) << Capacity) - 1);
		return FreeBitmap == allFree;
	}

	[[nodiscard]] OA_FORCEINLINE OaU32 FreeCount() const {
		return static_cast<OaU32>(std::popcount(FreeBitmap));
	}

	[[nodiscard]] OA_FORCEINLINE OaU32 UsedCount() const {
		return Capacity - FreeCount();
	}

	// Returns slot index, or 64 if full
	[[nodiscard]] OA_FORCEINLINE OaU32 AllocSlot() {
		if (OA_UNLIKELY(FreeBitmap == 0)) return 64;
		const OaU32 slot = static_cast<OaU32>(std::countr_zero(FreeBitmap));
		FreeBitmap &= FreeBitmap - 1;
		return slot;
	}

	OA_FORCEINLINE void FreeSlot(OaU32 InSlot) {
		FreeBitmap |= (OaU64(1) << InSlot);
	}

	[[nodiscard]] OA_FORCEINLINE void* SlotPtr(OaU32 InSlot) const {
		return BasePtr + (static_cast<OaU64>(InSlot) * SlotSize);
	}

	// Allocate and return pointer (or nullptr if full)
	[[nodiscard]] OA_FORCEINLINE void* Alloc() {
		OaU32 slot = AllocSlot();
		if (OA_UNLIKELY(slot >= Capacity)) return nullptr;
		return SlotPtr(slot);
	}

	// Free by pointer (must belong to this slab)
	OA_FORCEINLINE void Free(void* InPtr) {
		OaUsize offset = static_cast<OaU8*>(InPtr) - BasePtr;
		OaU32 slot = static_cast<OaU32>(offset / SlotSize);
		FreeSlot(slot);
	}
};
