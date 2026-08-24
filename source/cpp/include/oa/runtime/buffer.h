// OA Runtime physical buffer descriptor.
//
// This record carries shared storage identity and placement across semantic
// values. allocation, registration, mutation, and destruction are private
// oa::Engine services; installed callers must not manufacture device buffers.

#pragma once

#include <oa/core/device.h>
#include <oa/core/types.h>

static constexpr oa::U8 OA_VK_BUFFER_FLAG_NONE = 0;
static constexpr oa::U8 OA_VK_BUFFER_FLAG_BAR = 1;
static constexpr oa::U8 OA_VK_BUFFER_FLAG_ALIAS = 2;
static constexpr oa::U8 OA_VK_BUFFER_FLAG_TRANSIENT = 8;
// The VkBuffer was created with VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT.
// Keep this creation-time capability because it cannot be queried from a
// VkBuffer handle when an indirect dispatch is admitted later.
static constexpr oa::U8 OA_VK_BUFFER_FLAG_INDIRECT_DISPATCH = 16;

namespace oavk {

class Buffer {
public:
	void* buffer = nullptr;
	void* allocation = nullptr;
	// Opaque VMA allocator that created the allocation. This distinguishes
	// primary-engine buffers from handles owned by another engine/device.
	void* allocatorIdentity = nullptr;
	// Non-null only for buffers deliberately bound to the same physical
	// allocation. Distinct VkBuffer handles with the same identity require a
	// global memory dependency at lifetime hand-off; a buffer barrier scopes
	// accesses through only one handle and is insufficient for aliases.
	void* aliasIdentity = nullptr;
	// Logical range visible to callers. capacity is the addressable VkBuffer
	// creation size retained by caches/arenas and may be larger after reuse (the
	// backing VMA allocation itself may be larger still).
	oa::U64 size = 0;
	oa::U64 capacity = 0;
	void* mappedPtr = nullptr;
	oa::U8 flags = OA_VK_BUFFER_FLAG_NONE;
	oa::MemoryPlacement placement = oa::MemoryPlacement::Auto;
	oa::U32 bindlessIndex = UINT32_MAX;
	// OA allocators create one counter with the storage. Buffer descriptor copies
	// retain it, so a descriptor copied before an autograd snapshot still
	// invalidates every observed view when it records a write. observe provides a
	// lazy fallback only for externally assembled descriptors.
	mutable oa::SharedPtr<oa::U64> mutationVersion_;

	[[nodiscard]] OA_FORCEINLINE bool isBar() const {
		return flags & OA_VK_BUFFER_FLAG_BAR;
	}
	[[nodiscard]] OA_FORCEINLINE bool isTransient() const {
		return flags & OA_VK_BUFFER_FLAG_TRANSIENT;
	}
	[[nodiscard]] OA_FORCEINLINE bool supportsIndirectDispatch() const {
		return flags & OA_VK_BUFFER_FLAG_INDIRECT_DISPATCH;
	}
	// storage.slang may access packed sub-word scalars through their enclosing
	// 32-bit word. Expose only that padded logical word, never the full reusable
	// capacity: a wider descriptor would hide logical out-of-bounds shader access
	// from GPU-assisted validation. capacity is still the hard VkBuffer bound.
	[[nodiscard]] OA_FORCEINLINE oa::U64 descriptorRange() const {
		if (capacity == 0) return size;
		oa::U64 padded = size == 0 ? 1U : size;
		if (padded <= ~oa::U64{0} - 3U) padded = (padded + 3U) & ~oa::U64{3U};
		return padded < capacity ? padded : capacity;
	}
	[[nodiscard]] OA_FORCEINLINE void* synchronizationIdentity() const {
		return aliasIdentity ? aliasIdentity : buffer;
	}
	[[nodiscard]] OA_FORCEINLINE bool isHostVisible() const {
		return mappedPtr != nullptr;
	}
	[[nodiscard]] OA_FORCEINLINE bool isDeviceLocal() const {
		return placement == oa::MemoryPlacement::DeviceLocal
			or placement == oa::MemoryPlacement::Unified;
	}
	[[nodiscard]] oa::U64 observeMutationVersion() const;
	[[nodiscard]] oa::U64 currentMutationVersion() const noexcept;
	void markMutation() const noexcept;
};

} // namespace oavk
