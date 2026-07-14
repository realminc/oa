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


class OaVkBuffer {
public:

	// Data, class members.
	void* Buffer = nullptr;
	void* Allocation = nullptr;
	OaU64 Size = 0;
	void* MappedPtr = nullptr;
	OaU8 Flags = OA_VK_BUFFER_FLAG_NONE;
	OaU32 BindlessIndex = UINT32_MAX;
	// Owning mesh node: matches OaDeviceNode::Index when OaComputeEngine has a device mesh (0 = primary).
	OaU32 NodeIndex = 0;

	// Methods.
	[[nodiscard]] OA_FORCEINLINE bool IsBar() const { return Flags & OA_VK_BUFFER_FLAG_BAR; }
	[[nodiscard]] OA_FORCEINLINE bool IsImported() const { return Flags & OA_VK_BUFFER_FLAG_IMPORTED; }
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
	[[nodiscard]] OaResult<OaVkBuffer> AllocAliased(OaU64 InSize);

	// Create a new VkBuffer bound to an existing allocation's memory.
	// The existing buffer must have been allocated with AllocAliased().
	// Only destroys the VkBuffer on Free — the allocation stays alive.
	[[nodiscard]] OaResult<OaVkBuffer> CreateAliasingBuffer(const OaVkBuffer& InExisting, OaU64 InSize);

	// Free an aliasing buffer (destroys VkBuffer only, not the allocation).
	void FreeAlias(OaVkBuffer& InOutBuffer);

	// VK_EXT_memory_budget — aggregate across all device-local heaps
	[[nodiscard]] OaVmaStats GetStats() const;
};
