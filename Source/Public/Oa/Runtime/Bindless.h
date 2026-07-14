#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Thread.h>
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan_core.h>

class OaVkDevice;
class OaVkBuffer;

// Fallback limits for devices that don't support descriptor indexing
static constexpr OaU32 OA_BINDLESS_CAPACITY_FALLBACK = 65536;
static constexpr OaU32 OA_BINDLESS_IMAGE_CAPACITY_FALLBACK = 8192;
static constexpr OaU32 OA_BINDLESS_SAMPLER_CAPACITY_FALLBACK = 1024;
static constexpr OaU32 OA_BINDLESS_INVALID = UINT32_MAX;

// Runtime bindless capacity (queried from device at init)
extern OaU32 OA_BINDLESS_CAPACITY;
extern OaU32 OA_BINDLESS_IMAGE_CAPACITY;
extern OaU32 OA_BINDLESS_SAMPLER_CAPACITY;

// Vulkan min guaranteed push constant size (oa pipelines use this cap).
static constexpr OaU32 OA_VK_MAX_PUSH_CONSTANT_BYTES = 128;

// Bytes left for shader-specific push after bindless buffer-index header (numBufs * 4).
[[nodiscard]] constexpr OaU32 OaVkBindlessMaxUserPushBytes(OaU32 InNumBuffers) noexcept {
	const OaU64 header = static_cast<OaU64>(InNumBuffers) * sizeof(OaU32);
	if (header > OA_VK_MAX_PUSH_CONSTANT_BYTES) {
		return 0;
	}
	return static_cast<OaU32>(OA_VK_MAX_PUSH_CONSTANT_BYTES - header);
}

[[nodiscard]] constexpr bool OaVkBindlessPushFits(
	OaU32 InNumBuffers, OaU32 InUserPushSize) noexcept {
	const OaU64 header = static_cast<OaU64>(InNumBuffers) * sizeof(OaU32);
	return header + static_cast<OaU64>(InUserPushSize) <= OA_VK_MAX_PUSH_CONSTANT_BYTES;
}

// Global descriptor heap — one descriptor set with bindless arrays:
//   binding 0: RWByteAddressBuffer heap[]      (storage buffers)
//   binding 1: RWTexture2D/Texture2D images[]  (storage images)
//   binding 2: Texture2D sampled_images[]      (sampled images)
//   binding 3: SamplerState samplers[]         (samplers)
// Every resource gets an index at registration time. Shaders index into arrays
// via push constant resource indices. No per-dispatch descriptor set allocation.
// The resource layout is visible to compute, vertex, and fragment stages.

// Requires VK 1.2 descriptorIndexing features (enabled in OaVkDevice::Create).
class OaBindlessHeap {
public:
	void* DescriptorPool = nullptr;
	void* DescriptorSetLayout = nullptr;
	void* DescriptorSet = nullptr;
	void* PipelineLayout = nullptr;

	OaBindlessHeap() = default;
	OaBindlessHeap(OaBindlessHeap&& InOther) noexcept
		: DescriptorPool(InOther.DescriptorPool)
		, DescriptorSetLayout(InOther.DescriptorSetLayout)
		, DescriptorSet(InOther.DescriptorSet)
		, PipelineLayout(InOther.PipelineLayout)
		, FreeList_(std::move(InOther.FreeList_))
		, StorageImageFreeList_(std::move(InOther.StorageImageFreeList_))
		, SampledImageFreeList_(std::move(InOther.SampledImageFreeList_))
		, SamplerFreeList_(std::move(InOther.SamplerFreeList_))
	{
		InOther.DescriptorPool = nullptr;
		InOther.DescriptorSetLayout = nullptr;
		InOther.DescriptorSet = nullptr;
		InOther.PipelineLayout = nullptr;
	}

	OaBindlessHeap& operator=(OaBindlessHeap&& InOther) noexcept {
		if (this != &InOther) {
			DescriptorPool = InOther.DescriptorPool;
			DescriptorSetLayout = InOther.DescriptorSetLayout;
			DescriptorSet = InOther.DescriptorSet;
			PipelineLayout = InOther.PipelineLayout;
			FreeList_ = std::move(InOther.FreeList_);
			StorageImageFreeList_ = std::move(InOther.StorageImageFreeList_);
			SampledImageFreeList_ = std::move(InOther.SampledImageFreeList_);
			SamplerFreeList_ = std::move(InOther.SamplerFreeList_);
			InOther.DescriptorPool = nullptr;
			InOther.DescriptorSetLayout = nullptr;
			InOther.DescriptorSet = nullptr;
			InOther.PipelineLayout = nullptr;
		}
		return *this;
	}

	OaBindlessHeap(const OaBindlessHeap&) = delete;
	OaBindlessHeap& operator=(const OaBindlessHeap&) = delete;

	[[nodiscard]] static OaResult<OaBindlessHeap> Create(const OaVkDevice& InDevice);
	void Destroy(const OaVkDevice& InDevice);

	// Register a buffer — returns a slot index (0..65535).
	// Thread-safe. Automatically called by OaVma::Alloc*.
	[[nodiscard]] OaU32 Register(const OaVkDevice& InDevice, const OaVkBuffer& InBuffer);

	// Release a buffer slot. Thread-safe. Called by OaVma::Free.
	void Deregister(OaU32 InIndex);

	// Update a slot's buffer (e.g. after resize). Thread-safe.
	void Update(const OaVkDevice& InDevice, OaU32 InIndex, const OaVkBuffer& InBuffer);

	[[nodiscard]] OaU32 RegisterStorageImage(const OaVkDevice& InDevice, VkImageView InView, VkImageLayout InLayout);
	void UpdateStorageImage(const OaVkDevice& InDevice, OaU32 InIndex, VkImageView InView, VkImageLayout InLayout);
	void DeregisterStorageImage(OaU32 InIndex);

	[[nodiscard]] OaU32 RegisterSampledImage(const OaVkDevice& InDevice, VkImageView InView, VkImageLayout InLayout);
	void UpdateSampledImage(const OaVkDevice& InDevice, OaU32 InIndex, VkImageView InView, VkImageLayout InLayout);
	void DeregisterSampledImage(OaU32 InIndex);

	[[nodiscard]] OaU32 RegisterSampler(const OaVkDevice& InDevice, VkSampler InSampler);
	void UpdateSampler(const OaVkDevice& InDevice, OaU32 InIndex, VkSampler InSampler);
	void DeregisterSampler(OaU32 InIndex);

private:
	OaVec<OaU32> FreeList_;
	OaVec<OaU32> StorageImageFreeList_;
	OaVec<OaU32> SampledImageFreeList_;
	OaVec<OaU32> SamplerFreeList_;
	OaSpinlock Lock_;
};
