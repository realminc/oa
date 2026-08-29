#include <oa/runtime/bindless.h>
#include <oa/runtime/device.h>
#include <oa/runtime/allocator.h>
#include <vkl/vkl.h>
#include <oa/core/log.h>
#include <oa/core/std/algo.h>
#include "descriptorValidation.h"

namespace oavk {

oa::Result<BindlessHeap> BindlessHeap::create(
	const oavk::Device& inDevice,
	BindlessCapacities inOverride)
{
	VkDevice dev = static_cast<VkDevice>(inDevice.device);

	// Try to create bindless heap with the queried capacity
	// If allocation fails, retry with progressively smaller capacities
	// This handles drivers that report high theoretical limits but cannot allocate them
	const auto& hardware = inDevice.info.hardware;
	oa::U32 bufferCapacity = inOverride.buffers != 0U
		? inOverride.buffers : hardware.bindlessBufferCapacity;
	oa::U32 imageCapacity = inOverride.images != 0U
		? inOverride.images : hardware.bindlessImageCapacity;
	oa::U32 samplerCapacity = inOverride.samplers != 0U
		? inOverride.samplers : hardware.bindlessSamplerCapacity;
	if (bufferCapacity < 2U or imageCapacity < 2U or samplerCapacity < 2U) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"bindless descriptor limits must provide at least two slots per resource kind");
	}

	// All capacities are now capped in deviceBuilder (1M buffers, 16K images, 2K samplers).
	// Use those values directly; the retry loop below will back off if a driver still
	// cannot satisfy the capped request.

	const oa::U32 minBufferCapacity = oa::min(
		bufferCapacity, OA_BINDLESS_CAPACITY_FALLBACK);
	const oa::U32 minImageCapacity = oa::min(
		imageCapacity, OA_BINDLESS_IMAGE_CAPACITY_FALLBACK);
	const oa::U32 minSamplerCapacity = oa::min(
		samplerCapacity, OA_BINDLESS_SAMPLER_CAPACITY_FALLBACK);

	// Track retry stage: 0 = first attempt, 1 = halve buffers only, 2 = halve everything
	oa::U32 retryStage = 0;

	// Unified bindless set layout:
	//   binding 0: storage buffers
	//   binding 1: storage images
	//   binding 2: sampled images
	//   binding 3: samplers
	constexpr VkShaderStageFlags resourceStages =
		VK_SHADER_STAGE_COMPUTE_BIT |
		VK_SHADER_STAGE_VERTEX_BIT |
		VK_SHADER_STAGE_FRAGMENT_BIT;

	VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
	VkDescriptorPool pool = VK_NULL_HANDLE;
	VkResult r;

	// Retry loop with exponential backoff on capacity
	while (true) {
		VkDescriptorSetLayoutBinding bindings[4]{};
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		bindings[0].descriptorCount = bufferCapacity;
		bindings[0].stageFlags = resourceStages;
		bindings[1].binding = 1;
		bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		bindings[1].descriptorCount = imageCapacity;
		bindings[1].stageFlags = resourceStages;
		bindings[2].binding = 2;
		bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
		bindings[2].descriptorCount = imageCapacity;
		bindings[2].stageFlags = resourceStages;
		bindings[3].binding = 3;
		bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
		bindings[3].descriptorCount = samplerCapacity;
		bindings[3].stageFlags = resourceStages;

		VkDescriptorBindingFlags bindingFlags[4]{};
		for (oa::U32 i = 0; i < 4; ++i) {
			bindingFlags[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
		}

		VkDescriptorSetLayoutBindingFlagsCreateInfo flagsCI{};
		flagsCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
		flagsCI.bindingCount = 4;
		flagsCI.pBindingFlags = bindingFlags;

		VkDescriptorSetLayoutCreateInfo dslCI{};
		dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		dslCI.pNext = &flagsCI;
		dslCI.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
		dslCI.bindingCount = 4;
		dslCI.pBindings = bindings;

		r = inDevice.deviceDispatch.vkCreateDescriptorSetLayout(dev, &dslCI, nullptr, &dsl);

		// Descriptor pool with UPDATE_AFTER_BIND flag.
		VkDescriptorPoolSize poolSizes[4]{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[0].descriptorCount = bufferCapacity;
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[1].descriptorCount = imageCapacity;
		poolSizes[2].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
		poolSizes[2].descriptorCount = imageCapacity;
		poolSizes[3].type = VK_DESCRIPTOR_TYPE_SAMPLER;
		poolSizes[3].descriptorCount = samplerCapacity;

		VkDescriptorPoolCreateInfo dpCI{};
		dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		dpCI.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
		dpCI.maxSets = 1;
		dpCI.poolSizeCount = 4;
		dpCI.pPoolSizes = poolSizes;

		if (r == VK_SUCCESS) {
			r = inDevice.deviceDispatch.vkCreateDescriptorPool(dev, &dpCI, nullptr, &pool);
		}
		if (r == VK_SUCCESS) break;

		// Failed - clean up and retry with smaller capacity
		if (dsl != VK_NULL_HANDLE) {
			inDevice.deviceDispatch.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
		}
		dsl = VK_NULL_HANDLE;
		const bool alreadyAtMinimum =
			bufferCapacity == minBufferCapacity and
			imageCapacity == minImageCapacity and
			samplerCapacity == minSamplerCapacity;
		if (alreadyAtMinimum) {
			OaLogError(oa::LogComponent::Runtime,
				"bindless creation failed at minimum capacity (VkResult={})", r);
			return oa::Status::error(oa::StatusCode::PipelineError,
				"bindless descriptor layout/pool creation failed at minimum capacity");
		}

		// Staged retry: first halve buffers only, then halve everything
		if (retryStage == 0) {
			// first failure: halve buffers only
			if (bufferCapacity > minBufferCapacity) {
				bufferCapacity = oa::max(bufferCapacity / 2, minBufferCapacity);
			}
			retryStage = 1;
		} else {
			// Subsequent failures: halve everything
			if (bufferCapacity > minBufferCapacity) {
				bufferCapacity = oa::max(bufferCapacity / 2, minBufferCapacity);
			}
			if (imageCapacity > minImageCapacity) {
				imageCapacity = oa::max(imageCapacity / 2, minImageCapacity);
			}
			if (samplerCapacity > minSamplerCapacity) {
				samplerCapacity = oa::max(samplerCapacity / 2, minSamplerCapacity);
			}
		}

		OaLogWarn(oa::LogComponent::Runtime, "bindless creation failed, retrying with smaller capacity (buffers={} images={} samplers={})",
			bufferCapacity, imageCapacity, samplerCapacity);
	}

	OaLogInfo(oa::LogComponent::Runtime, "bindless heap: buffers={} imageSlots={} samplerSlots={}, UPDATE_AFTER_BIND",
		bufferCapacity - 1, imageCapacity - 1, samplerCapacity - 1);

	// allocate the one global descriptor set.
	VkDescriptorSetAllocateInfo dsAI{};
	dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAI.descriptorPool = pool;
	dsAI.descriptorSetCount = 1;
	dsAI.pSetLayouts = &dsl;

	VkDescriptorSet ds = VK_NULL_HANDLE;
	r = inDevice.deviceDispatch.vkAllocateDescriptorSets(dev, &dsAI, &ds);
	if (r != VK_SUCCESS) {
		OaLogError(oa::LogComponent::Runtime, "bindless: vkAllocateDescriptorSets failed (VkResult={})", r);
		inDevice.deviceDispatch.vkDestroyDescriptorPool(dev, pool, nullptr);
		inDevice.deviceDispatch.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
		return oa::Status::error(oa::StatusCode::PipelineError,
			"bindless: vkAllocateDescriptorSets failed");
	}

	// Shared pipeline layout: bindless set + 128-byte push range
	VkPushConstantRange pushRange{};
	pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pushRange.offset = 0;
	pushRange.size = 128;

	VkPipelineLayoutCreateInfo plCI{};
	plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	plCI.setLayoutCount = 1;
	plCI.pSetLayouts = &dsl;
	plCI.pushConstantRangeCount = 1;
	plCI.pPushConstantRanges = &pushRange;

	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	r = inDevice.deviceDispatch.vkCreatePipelineLayout(dev, &plCI, nullptr, &pipelineLayout);
	if (r != VK_SUCCESS) {
		OaLogError(oa::LogComponent::Runtime, "bindless: vkCreatePipelineLayout failed (VkResult={})", r);
		inDevice.deviceDispatch.vkDestroyDescriptorPool(dev, pool, nullptr);
		inDevice.deviceDispatch.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
		return oa::Status::error(oa::StatusCode::PipelineError,
			"bindless: vkCreatePipelineLayout failed");
	}

	BindlessHeap heap;
	heap.descriptorPool = pool;
	heap.descriptorSetLayout = dsl;
	heap.descriptorSet = ds;
	heap.pipelineLayout = pipelineLayout;
	heap.capacities_ = {
		.buffers = bufferCapacity,
		.images = imageCapacity,
		.samplers = samplerCapacity,
	};

	// initialize free lists (reverse order, skip index 0 which is OA_BINDLESS_INVALID).
	heap.freeList_.resize(bufferCapacity - 1);
	for (oa::U32 i = 1; i < bufferCapacity; ++i) {
		heap.freeList_[i - 1] = bufferCapacity - i;
	}
	heap.storageImageFreeList_.resize(imageCapacity - 1);
	heap.sampledImageFreeList_.resize(imageCapacity - 1);
	for (oa::U32 i = 1; i < imageCapacity; ++i) {
		heap.storageImageFreeList_[i - 1] = imageCapacity - i;
		heap.sampledImageFreeList_[i - 1] = imageCapacity - i;
	}
	heap.samplerFreeList_.resize(samplerCapacity - 1);
	for (oa::U32 i = 1; i < samplerCapacity; ++i) {
		heap.samplerFreeList_[i - 1] = samplerCapacity - i;
	}

	return heap;
}

void BindlessHeap::destroy(const oavk::Device& inDevice) {
	VkDevice dev = static_cast<VkDevice>(inDevice.device);
	if (pipelineLayout) {
		inDevice.deviceDispatch.vkDestroyPipelineLayout(dev, static_cast<VkPipelineLayout>(pipelineLayout), nullptr);
		pipelineLayout = nullptr;
	}
	if (descriptorPool) {
		inDevice.deviceDispatch.vkDestroyDescriptorPool(dev, static_cast<VkDescriptorPool>(descriptorPool), nullptr);
		descriptorPool = nullptr;
		descriptorSet = nullptr;
	}
	if (descriptorSetLayout) {
		inDevice.deviceDispatch.vkDestroyDescriptorSetLayout(dev, static_cast<VkDescriptorSetLayout>(descriptorSetLayout), nullptr);
		descriptorSetLayout = nullptr;
	}
	freeList_.clear();
	storageImageFreeList_.clear();
	sampledImageFreeList_.clear();
	samplerFreeList_.clear();
	capacities_ = {};
}

oa::U32 BindlessHeap::registerBuffer(
	const oavk::Device& inDevice, const oavk::Buffer& inBuffer)
{
	const auto validation =
		oavk::validateStorageBufferDescriptor(inDevice, inBuffer);
	if (not validation.isOk()) {
		OaLogError(oa::LogComponent::Runtime,
			"bindless heap: refusing storage-buffer registration: {}",
			validation.getMessage().cStr());
		return OA_BINDLESS_INVALID;
	}
	oa::SpinlockGuard guard(lock_);
	if (freeList_.empty()) {
		OaLogError(oa::LogComponent::Runtime, "bindless heap: out of slots ({} max)",
			capacities_.buffers);
		return OA_BINDLESS_INVALID;
	}

	oa::U32 index = freeList_.back();
	freeList_.popBack();

	VkDevice dev = static_cast<VkDevice>(inDevice.device);

	VkDescriptorBufferInfo bufInfo{};
	bufInfo.buffer = static_cast<VkBuffer>(inBuffer.buffer);
	bufInfo.offset = 0;
	bufInfo.range = inBuffer.descriptorRange();

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = static_cast<VkDescriptorSet>(descriptorSet);
	write.dstBinding = 0;
	write.dstArrayElement = index;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	write.pBufferInfo = &bufInfo;

	inDevice.deviceDispatch.vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
	return index;
}

void BindlessHeap::deregister(oa::U32 inIndex) {
	if (inIndex == OA_BINDLESS_INVALID) return;
	oa::SpinlockGuard guard(lock_);
	freeList_.pushBack(inIndex);
}

oa::Status BindlessHeap::update(
	const oavk::Device& inDevice, oa::U32 inIndex, const oavk::Buffer& inBuffer)
{
	if (inIndex == OA_BINDLESS_INVALID) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"bindless update requires a valid storage-buffer slot");
	}
	OA_RETURN_IF_ERROR(
		oavk::validateStorageBufferDescriptor(inDevice, inBuffer));
	oa::SpinlockGuard guard(lock_);

	VkDevice dev = static_cast<VkDevice>(inDevice.device);

	VkDescriptorBufferInfo bufInfo{};
	bufInfo.buffer = static_cast<VkBuffer>(inBuffer.buffer);
	bufInfo.offset = 0;
	bufInfo.range = inBuffer.descriptorRange();

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = static_cast<VkDescriptorSet>(descriptorSet);
	write.dstBinding = 0;
	write.dstArrayElement = inIndex;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	write.pBufferInfo = &bufInfo;

	inDevice.deviceDispatch.vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
	return oa::Status::ok();
}

static oa::U32 bindlessPopSlot(oa::Vector<oa::U32>& inOutFreeList, const char* inKind)
{
	if (inOutFreeList.empty()) {
		OaLogError(oa::LogComponent::Runtime, "bindless heap: out of {} slots", inKind);
		return OA_BINDLESS_INVALID;
	}
	oa::U32 index = inOutFreeList.back();
	inOutFreeList.popBack();
	return index;
}

static void bindlessWriteImage(
	const oavk::Device& inDevice,
	VkDescriptorSet inSet,
	oa::U32 inBinding,
	oa::U32 inIndex,
	VkDescriptorType inType,
	VkImageView inView,
	VkImageLayout inLayout,
	VkSampler inSampler = VK_NULL_HANDLE)
{
	VkDescriptorImageInfo info{};
	info.imageView = inView;
	info.imageLayout = inLayout;
	info.sampler = inSampler;

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = inSet;
	write.dstBinding = inBinding;
	write.dstArrayElement = inIndex;
	write.descriptorCount = 1;
	write.descriptorType = inType;
	write.pImageInfo = &info;
	inDevice.deviceDispatch.vkUpdateDescriptorSets(
		static_cast<VkDevice>(inDevice.device), 1, &write, 0, nullptr);
}

oa::U32 BindlessHeap::registerStorageImage(const oavk::Device& inDevice, VkImageView inView, VkImageLayout inLayout)
{
	if (!inView) return OA_BINDLESS_INVALID;
	oa::SpinlockGuard guard(lock_);
	oa::U32 index = bindlessPopSlot(storageImageFreeList_, "storage image");
	if (index == OA_BINDLESS_INVALID) return index;
	updateStorageImage(inDevice, index, inView, inLayout);
	return index;
}

void BindlessHeap::updateStorageImage(const oavk::Device& inDevice, oa::U32 inIndex, VkImageView inView, VkImageLayout inLayout)
{
	if (inIndex == OA_BINDLESS_INVALID || !inView) return;
	bindlessWriteImage(
		inDevice,
		static_cast<VkDescriptorSet>(descriptorSet),
		1,
		inIndex,
		VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		inView,
		inLayout);
}

void BindlessHeap::deregisterStorageImage(oa::U32 inIndex) {
	if (inIndex == OA_BINDLESS_INVALID) return;
	oa::SpinlockGuard guard(lock_);
	storageImageFreeList_.pushBack(inIndex);
}

oa::U32 BindlessHeap::registerSampledImage(const oavk::Device& inDevice, VkImageView inView, VkImageLayout inLayout) {
	if (!inView) return OA_BINDLESS_INVALID;
	oa::SpinlockGuard guard(lock_);
	oa::U32 index = bindlessPopSlot(sampledImageFreeList_, "sampled image");
	if (index == OA_BINDLESS_INVALID) return index;
	updateSampledImage(inDevice, index, inView, inLayout);
	return index;
}

void BindlessHeap::updateSampledImage(const oavk::Device& inDevice, oa::U32 inIndex, VkImageView inView, VkImageLayout inLayout) {
	if (inIndex == OA_BINDLESS_INVALID || !inView) return;
	bindlessWriteImage(
		inDevice,
		static_cast<VkDescriptorSet>(descriptorSet),
		2,
		inIndex,
		VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
		inView,
		inLayout
	);
}

void BindlessHeap::deregisterSampledImage(oa::U32 inIndex) {
	if (inIndex == OA_BINDLESS_INVALID) return;
	oa::SpinlockGuard guard(lock_);
	sampledImageFreeList_.pushBack(inIndex);
}

oa::U32 BindlessHeap::registerSampler(const oavk::Device& inDevice, VkSampler inSampler) {
	if (!inSampler) return OA_BINDLESS_INVALID;
	oa::SpinlockGuard guard(lock_);
	oa::U32 index = bindlessPopSlot(samplerFreeList_, "sampler");
	if (index == OA_BINDLESS_INVALID) return index;
	updateSampler(inDevice, index, inSampler);
	return index;
}

void BindlessHeap::updateSampler(const oavk::Device& inDevice, oa::U32 inIndex, VkSampler inSampler) {
	if (inIndex == OA_BINDLESS_INVALID || !inSampler) return;
	bindlessWriteImage(
		inDevice,
		static_cast<VkDescriptorSet>(descriptorSet),
		3,
		inIndex,
		VK_DESCRIPTOR_TYPE_SAMPLER,
		VK_NULL_HANDLE,
		VK_IMAGE_LAYOUT_UNDEFINED,
		inSampler);
}

void BindlessHeap::deregisterSampler(oa::U32 inIndex){
	if (inIndex == OA_BINDLESS_INVALID) return;
	oa::SpinlockGuard guard(lock_);
	samplerFreeList_.pushBack(inIndex);
}

} // namespace oavk
