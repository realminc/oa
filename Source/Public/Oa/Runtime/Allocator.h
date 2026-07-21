// OaVma — GPU memory allocator (forked VMA)
//
// Thin C++ facade over the OaVma C API.
// SAM path: VMOVNTPS direct to VRAM via AllocBar + UploadWeights.
// Budget: VK_EXT_memory_budget for available VRAM query.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>


class OaVkDevice;


static constexpr OaU8 OA_VK_BUFFER_FLAG_NONE     = 0;
static constexpr OaU8 OA_VK_BUFFER_FLAG_BAR      = 1;
static constexpr OaU8 OA_VK_BUFFER_FLAG_ALIAS    = 2;
static constexpr OaU8 OA_VK_BUFFER_FLAG_IMPORTED = 4;
static constexpr OaU8 OA_VK_BUFFER_FLAG_TRANSIENT = 8;

// Memory intent is part of the buffer contract. `Auto` is resolved by the
// engine from the selected device: unified/host-visible on integrated GPUs,
// device-local on discrete GPUs. Transfer-only placements remain explicitly
// mapped and must not be used as long-lived compute storage.
enum class OaMemoryPlacement : OaU8 {
	Auto = 0,
	DeviceLocal,
	HostUpload,
	HostReadback,
	Unified,
};


class OaVkBuffer {
public:

	// Data, class members.
	void* Buffer = nullptr;
	void* Allocation = nullptr;
	// Opaque VMA allocator that created the allocation. This distinguishes
	// primary-engine buffers from handles owned by another engine/device.
	void* AllocatorIdentity = nullptr;
	// Non-null only for buffers deliberately bound to the same physical
	// allocation. Distinct VkBuffer handles with the same identity require a
	// global memory dependency at lifetime hand-off; a buffer barrier scopes
	// accesses through only one handle and is insufficient for aliases.
	void* AliasIdentity = nullptr;
	// Logical range visible to callers. Capacity is the addressable VkBuffer
	// creation size retained by caches/arenas and may be larger after reuse (the
	// backing VMA allocation itself may be larger still).
	OaU64 Size = 0;
	OaU64 Capacity = 0;
	void* MappedPtr = nullptr;
	OaU8 Flags = OA_VK_BUFFER_FLAG_NONE;
	OaMemoryPlacement Placement = OaMemoryPlacement::Auto;
	OaU32 BindlessIndex = UINT32_MAX;
	// Owning mesh node: matches OaDeviceNode::Index when OaEngine has a device mesh (0 = primary).
	OaU32 NodeIndex = 0;

	// Methods.
	[[nodiscard]] OA_FORCEINLINE bool IsBar() const { return Flags & OA_VK_BUFFER_FLAG_BAR; }
	[[nodiscard]] OA_FORCEINLINE bool IsImported() const { return Flags & OA_VK_BUFFER_FLAG_IMPORTED; }
	[[nodiscard]] OA_FORCEINLINE bool IsTransient() const { return Flags & OA_VK_BUFFER_FLAG_TRANSIENT; }
	// Storage.slang may access packed sub-word scalars through their enclosing
	// 32-bit word. Expose only that padded logical word, never the full reusable
	// capacity: a wider descriptor would hide logical out-of-bounds shader access
	// from GPU-assisted validation. Capacity is still the hard VkBuffer bound.
	[[nodiscard]] OA_FORCEINLINE OaU64 DescriptorRange() const {
		if (Capacity == 0) return Size;
		OaU64 padded = Size == 0 ? 1U : Size;
		if (padded <= ~OaU64{0} - 3U) padded = (padded + 3U) & ~OaU64{3U};
		return padded < Capacity ? padded : Capacity;
	}
	[[nodiscard]] OA_FORCEINLINE void* SynchronizationIdentity() const {
		return AliasIdentity ? AliasIdentity : Buffer;
	}
	[[nodiscard]] OA_FORCEINLINE bool IsHostVisible() const { return MappedPtr != nullptr; }
	[[nodiscard]] OA_FORCEINLINE bool IsDeviceLocal() const {
		return Placement == OaMemoryPlacement::DeviceLocal || Placement == OaMemoryPlacement::Unified;
	}
};


class OaVmaStats {
public:
	OaU64 UsedBytes = 0;
	OaU64 BudgetBytes = 0;
	// Detailed (VmaCalculateStatistics): live allocation bytes vs reserved block bytes.
	// AllocationBytes = bytes actually handed out; BlockBytes = bytes VMA reserved from
	// the driver. (BlockBytes - AllocationBytes) is internal fragmentation / pooled slack.
	OaU64 AllocationBytes = 0;
	OaU64 BlockBytes = 0;
	OaU64 AllocationCount = 0;
	OaU64 BlockCount = 0;
};


class OaVma {
public:
	void* Allocator = nullptr;
	OaBool HasSam = false;

	[[nodiscard]] static OaResult<OaVma> Create(const OaVkDevice& InDevice);
	void Destroy();

	[[nodiscard]] OaResult<OaVkBuffer> AllocDevice(OaU64 InSize);
	[[nodiscard]] OaResult<OaVkBuffer> AllocHostVisible(OaU64 InSize);
	[[nodiscard]] OaResult<OaVkBuffer> AllocHostReadback(OaU64 InSize);
	[[nodiscard]] OaResult<OaVkBuffer> AllocBar(OaU64 InSize);
	[[nodiscard]] OaResult<OaVkBuffer> AllocPreprocessBuffer(OaU64 InSize);
	void Free(OaVkBuffer& InOutBuffer);

	// Free a buffer imported via OaImportBufferFd (raw VkDeviceMemory, not VMA).
	// Imported buffers MUST use this — OaVma::Free will assert.
	void FreeImported(const OaVkDevice& InDevice, OaVkBuffer& InOutBuffer);

	// SAM-aware weight upload: VMOVNTPS when BAR, memcpy otherwise
	OaStatus UploadWeights(OaVkBuffer& InDst, const void* InSrc, OaU64 InSize);

	// Flush a host write to a mapped allocation so a subsequent GPU read sees it.
	// No-op on HOST_COHERENT memory (VMA handles that internally); required on
	// non-coherent HOST_VISIBLE memory, which VMA can hand out under pressure.
	// Returns false only on a VMA error; a null/unallocated buffer is a no-op true.
	OaBool FlushHostBuffer(const OaVkBuffer& InBuf, OaU64 InOffset, OaU64 InSize);
	// Invalidate a mapped allocation after GPU writes so subsequent CPU reads see
	// device memory. Symmetric with FlushHostBuffer; a no-op on HOST_COHERENT.
	OaBool InvalidateHostBuffer(const OaVkBuffer& InBuf, OaU64 InOffset, OaU64 InSize);

	// Aliased allocation — CAN_ALIAS flag, shareable backing memory.
	// The returned buffer can have other buffers bound to the same memory.
	[[nodiscard]] OaResult<OaVkBuffer> AllocAliased(
		OaU64 InSize,
		OaMemoryPlacement InPlacement = OaMemoryPlacement::HostUpload);

	// Create a new VkBuffer bound to an existing allocation's memory.
	// The existing buffer must have been allocated with AllocAliased().
	// Only destroys the VkBuffer on Free — the allocation stays alive.
	[[nodiscard]] OaResult<OaVkBuffer> CreateAliasingBuffer(const OaVkBuffer& InExisting, OaU64 InSize);

	// Free an aliasing buffer (destroys VkBuffer only, not the allocation).
	void FreeAlias(OaVkBuffer& InOutBuffer);

	// VK_EXT_memory_budget — aggregate across all device-local heaps
	[[nodiscard]] OaVmaStats GetStats() const;
};
