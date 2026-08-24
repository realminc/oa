#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/core/thread.h>
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan_core.h>

namespace oavk { class Device; }
namespace oavk { class Buffer; }

// Fallback limits for devices that don't support descriptor indexing
static constexpr oa::U32 OA_BINDLESS_CAPACITY_FALLBACK = 65536;
static constexpr oa::U32 OA_BINDLESS_IMAGE_CAPACITY_FALLBACK = 8192;
static constexpr oa::U32 OA_BINDLESS_SAMPLER_CAPACITY_FALLBACK = 1024;
static constexpr oa::U32 OA_BINDLESS_INVALID = UINT32_MAX;

namespace oavk {

class BindlessCapacities {
public:
	oa::U32 buffers = 0;
	oa::U32 images = 0;
	oa::U32 samplers = 0;
};

// vulkan min guaranteed push constant size (oa pipelines use this cap).
static constexpr oa::U32 OA_VK_MAX_PUSH_CONSTANT_BYTES = 128;

// Bytes left for shader-specific push after bindless buffer-index header (numBufs * 4).
[[nodiscard]] constexpr oa::U32 bindlessMaxUserPushBytes(oa::U32 inNumBuffers) noexcept {
	const oa::U64 header = static_cast<oa::U64>(inNumBuffers) * sizeof(oa::U32);
	if (header > OA_VK_MAX_PUSH_CONSTANT_BYTES) {
		return 0;
	}
	return static_cast<oa::U32>(OA_VK_MAX_PUSH_CONSTANT_BYTES - header);
}

[[nodiscard]] constexpr bool bindlessPushFits(
	oa::U32 inNumBuffers, oa::U32 inUserPushSize) noexcept {
	const oa::U64 header = static_cast<oa::U64>(inNumBuffers) * sizeof(oa::U32);
	return header + static_cast<oa::U64>(inUserPushSize) <= OA_VK_MAX_PUSH_CONSTANT_BYTES;
}

// Global descriptor heap — one descriptor set with bindless arrays:
//   binding 0: RWByteAddressBuffer heap[]      (storage buffers)
//   binding 1: RWTexture2D/Texture2D images[]  (storage images)
//   binding 2: Texture2D sampled_images[]      (sampled images)
//   binding 3: SamplerState samplers[]         (samplers)
// Every resource gets an index at registration time. Shaders index into arrays
// via push constant resource indices. No per-dispatch descriptor set allocation.
// The resource layout is visible to compute, vertex, and fragment stages.

// Requires VK 1.2 descriptorIndexing features (enabled in oavk::Device::Create).
class BindlessHeap {
public:
	void* descriptorPool = nullptr;
	void* descriptorSetLayout = nullptr;
	void* descriptorSet = nullptr;
	void* pipelineLayout = nullptr;

	BindlessHeap() = default;
	BindlessHeap(BindlessHeap&& inOther) noexcept
		: descriptorPool(inOther.descriptorPool)
		, descriptorSetLayout(inOther.descriptorSetLayout)
		, descriptorSet(inOther.descriptorSet)
		, pipelineLayout(inOther.pipelineLayout)
		, freeList_(std::move(inOther.freeList_))
		, storageImageFreeList_(std::move(inOther.storageImageFreeList_))
		, sampledImageFreeList_(std::move(inOther.sampledImageFreeList_))
		, samplerFreeList_(std::move(inOther.samplerFreeList_))
		, capacities_(inOther.capacities_)
	{
		inOther.descriptorPool = nullptr;
		inOther.descriptorSetLayout = nullptr;
		inOther.descriptorSet = nullptr;
		inOther.pipelineLayout = nullptr;
	}

	BindlessHeap& operator=(BindlessHeap&& inOther) noexcept {
		if (this != &inOther) {
			descriptorPool = inOther.descriptorPool;
			descriptorSetLayout = inOther.descriptorSetLayout;
			descriptorSet = inOther.descriptorSet;
			pipelineLayout = inOther.pipelineLayout;
			freeList_ = std::move(inOther.freeList_);
			storageImageFreeList_ = std::move(inOther.storageImageFreeList_);
			sampledImageFreeList_ = std::move(inOther.sampledImageFreeList_);
			samplerFreeList_ = std::move(inOther.samplerFreeList_);
			capacities_ = inOther.capacities_;
			inOther.descriptorPool = nullptr;
			inOther.descriptorSetLayout = nullptr;
			inOther.descriptorSet = nullptr;
			inOther.pipelineLayout = nullptr;
		}
		return *this;
	}

	BindlessHeap(const BindlessHeap&) = delete;
	BindlessHeap& operator=(const BindlessHeap&) = delete;

	[[nodiscard]] static oa::Result<BindlessHeap> create(
		const oavk::Device& inDevice,
		BindlessCapacities inOverride = {});
	void destroy(const oavk::Device& inDevice);

	// register a buffer — returns a device-sized slot in
	// [1, the engine-local buffer capacity), or OA_BINDLESS_INVALID on
	// exhaustion.
	// Thread-safe. Engine descriptor-backed allocation paths call this after VMA
	// storage succeeds and roll that storage back if no slot remains.
	[[nodiscard]] oa::U32 registerBuffer(const oavk::Device& inDevice, const oavk::Buffer& inBuffer);

	// release a buffer slot. Thread-safe. Called by oa::Engine before VMA release.
	void deregister(oa::U32 inIndex);

	// Update a slot's buffer (e.g. after resize). Thread-safe and fail-closed:
	// an invalid replacement leaves the previously written descriptor intact.
	[[nodiscard]] oa::Status update(
		const oavk::Device& inDevice,
		oa::U32 inIndex,
		const oavk::Buffer& inBuffer);

	[[nodiscard]] oa::U32 registerStorageImage(const oavk::Device& inDevice, VkImageView inView, VkImageLayout inLayout);
	void updateStorageImage(const oavk::Device& inDevice, oa::U32 inIndex, VkImageView inView, VkImageLayout inLayout);
	void deregisterStorageImage(oa::U32 inIndex);

	[[nodiscard]] oa::U32 registerSampledImage(const oavk::Device& inDevice, VkImageView inView, VkImageLayout inLayout);
	void updateSampledImage(const oavk::Device& inDevice, oa::U32 inIndex, VkImageView inView, VkImageLayout inLayout);
	void deregisterSampledImage(oa::U32 inIndex);

	[[nodiscard]] oa::U32 registerSampler(const oavk::Device& inDevice, VkSampler inSampler);
	void updateSampler(const oavk::Device& inDevice, oa::U32 inIndex, VkSampler inSampler);
	void deregisterSampler(oa::U32 inIndex);

private:
	oa::Vec<oa::U32> freeList_;
	oa::Vec<oa::U32> storageImageFreeList_;
	oa::Vec<oa::U32> sampledImageFreeList_;
	oa::Vec<oa::U32> samplerFreeList_;
	BindlessCapacities capacities_;
	oa::Spinlock lock_;
};

} // namespace oavk
