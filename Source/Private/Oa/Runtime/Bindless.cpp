#include <Oa/Runtime/Bindless.h>
#include <Oa/Runtime/Device.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Core/Log.h>

// Runtime bindless capacity (initialized from device limits)
OaU32 OA_BINDLESS_CAPACITY = OA_BINDLESS_CAPACITY_FALLBACK;
OaU32 OA_BINDLESS_IMAGE_CAPACITY = OA_BINDLESS_IMAGE_CAPACITY_FALLBACK;
OaU32 OA_BINDLESS_SAMPLER_CAPACITY = OA_BINDLESS_SAMPLER_CAPACITY_FALLBACK;

OaResult<OaBindlessHeap> OaBindlessHeap::Create(const OaVkDevice& InDevice) {
	VkDevice dev = static_cast<VkDevice>(InDevice.Device);

	// Try to create bindless heap with the queried capacity
	// If allocation fails, retry with progressively smaller capacities
	// This handles drivers that report high theoretical limits but cannot allocate them
	OaU32 bufferCapacity = OA_BINDLESS_CAPACITY;
	OaU32 imageCapacity = OA_BINDLESS_IMAGE_CAPACITY;
	OaU32 samplerCapacity = OA_BINDLESS_SAMPLER_CAPACITY;

	// All capacities are now capped in DeviceBuilder (1M buffers, 16K images, 2K samplers).
	// Use those values directly; the retry loop below will back off if a driver still
	// cannot satisfy the capped request.

	constexpr OaU32 kMinBufferCapacity = OA_BINDLESS_CAPACITY_FALLBACK;
	constexpr OaU32 kMinImageCapacity = OA_BINDLESS_IMAGE_CAPACITY_FALLBACK;
	constexpr OaU32 kMinSamplerCapacity = OA_BINDLESS_SAMPLER_CAPACITY_FALLBACK;

	// Track retry stage: 0 = first attempt, 1 = halve buffers only, 2 = halve everything
	OaU32 retryStage = 0;

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
		for (OaU32 i = 0; i < 4; ++i) {
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

		VkResult r = vkCreateDescriptorSetLayout(dev, &dslCI, nullptr, &dsl);
		if (r != VK_SUCCESS) {
			OA_LOG_ERROR(OaLogComponent::Core, "Bindless: vkCreateDescriptorSetLayout failed (VkResult=%d)", r);
			return OaStatus::Error(OaStatusCode::PipelineError,
				"bindless: vkCreateDescriptorSetLayout failed");
		}

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

		r = vkCreateDescriptorPool(dev, &dpCI, nullptr, &pool);
		if (r == VK_SUCCESS) {
			// Success - update global capacities to what we actually allocated
			OA_BINDLESS_CAPACITY = bufferCapacity;
			OA_BINDLESS_IMAGE_CAPACITY = imageCapacity;
			OA_BINDLESS_SAMPLER_CAPACITY = samplerCapacity;
			break;
		}

		// Failed - clean up and retry with smaller capacity
		vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
		dsl = VK_NULL_HANDLE;

		// Staged retry: first halve buffers only, then halve everything
		if (retryStage == 0) {
			// First failure: halve buffers only
			if (bufferCapacity > kMinBufferCapacity) {
				bufferCapacity = std::max(bufferCapacity / 2, kMinBufferCapacity);
			}
			retryStage = 1;
		} else {
			// Subsequent failures: halve everything
			if (bufferCapacity > kMinBufferCapacity) {
				bufferCapacity = std::max(bufferCapacity / 2, kMinBufferCapacity);
			}
			if (imageCapacity > kMinImageCapacity) {
				imageCapacity = std::max(imageCapacity / 2, kMinImageCapacity);
			}
			if (samplerCapacity > kMinSamplerCapacity) {
				samplerCapacity = std::max(samplerCapacity / 2, kMinSamplerCapacity);
			}
		}

		// If we're at minimum and still failing, give up
		if (bufferCapacity == kMinBufferCapacity &&
		    imageCapacity == kMinImageCapacity &&
		    samplerCapacity == kMinSamplerCapacity) {
			OA_LOG_ERROR(OaLogComponent::Core, "Bindless: vkCreateDescriptorPool failed even at minimum capacity (VkResult=%d)", r);
			return OaStatus::Error(OaStatusCode::PipelineError,
				"bindless: vkCreateDescriptorPool failed at minimum capacity");
		}

		OA_LOG_WARN(OaLogComponent::Core, "Bindless: vkCreateDescriptorPool failed, retrying with smaller capacity (buffers=%u images=%u samplers=%u)",
			bufferCapacity, imageCapacity, samplerCapacity);
	}

	OA_LOG_INFO(OaLogComponent::Core, "Bindless heap: buffers=%u imageSlots=%u samplerSlots=%u, UPDATE_AFTER_BIND",
		bufferCapacity - 1, imageCapacity - 1, samplerCapacity - 1);

	// Allocate the one global descriptor set.
	VkDescriptorSetAllocateInfo dsAI{};
	dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAI.descriptorPool = pool;
	dsAI.descriptorSetCount = 1;
	dsAI.pSetLayouts = &dsl;

	VkDescriptorSet ds = VK_NULL_HANDLE;
	r = vkAllocateDescriptorSets(dev, &dsAI, &ds);
	if (r != VK_SUCCESS) {
		OA_LOG_ERROR(OaLogComponent::Core, "Bindless: vkAllocateDescriptorSets failed (VkResult=%d)", r);
		vkDestroyDescriptorPool(dev, pool, nullptr);
		vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
		return OaStatus::Error(OaStatusCode::PipelineError,
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
	r = vkCreatePipelineLayout(dev, &plCI, nullptr, &pipelineLayout);
	if (r != VK_SUCCESS) {
		OA_LOG_ERROR(OaLogComponent::Core, "Bindless: vkCreatePipelineLayout failed (VkResult=%d)", r);
		vkDestroyDescriptorPool(dev, pool, nullptr);
		vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
		return OaStatus::Error(OaStatusCode::PipelineError,
			"bindless: vkCreatePipelineLayout failed");
	}

	OaBindlessHeap heap;
	heap.DescriptorPool = pool;
	heap.DescriptorSetLayout = dsl;
	heap.DescriptorSet = ds;
	heap.PipelineLayout = pipelineLayout;

	// Initialize free lists (reverse order, skip index 0 which is OA_BINDLESS_INVALID).
	heap.FreeList_.Resize(OA_BINDLESS_CAPACITY - 1);
	for (OaU32 i = 1; i < OA_BINDLESS_CAPACITY; ++i) {
		heap.FreeList_[i - 1] = OA_BINDLESS_CAPACITY - i;
	}
	heap.StorageImageFreeList_.Resize(OA_BINDLESS_IMAGE_CAPACITY - 1);
	heap.SampledImageFreeList_.Resize(OA_BINDLESS_IMAGE_CAPACITY - 1);
	for (OaU32 i = 1; i < OA_BINDLESS_IMAGE_CAPACITY; ++i) {
		heap.StorageImageFreeList_[i - 1] = OA_BINDLESS_IMAGE_CAPACITY - i;
		heap.SampledImageFreeList_[i - 1] = OA_BINDLESS_IMAGE_CAPACITY - i;
	}
	heap.SamplerFreeList_.Resize(OA_BINDLESS_SAMPLER_CAPACITY - 1);
	for (OaU32 i = 1; i < OA_BINDLESS_SAMPLER_CAPACITY; ++i) {
		heap.SamplerFreeList_[i - 1] = OA_BINDLESS_SAMPLER_CAPACITY - i;
	}

	return heap;
}

void OaBindlessHeap::Destroy(const OaVkDevice& InDevice) {
	VkDevice dev = static_cast<VkDevice>(InDevice.Device);
	if (PipelineLayout) {
		vkDestroyPipelineLayout(dev, static_cast<VkPipelineLayout>(PipelineLayout), nullptr);
		PipelineLayout = nullptr;
	}
	if (DescriptorPool) {
		vkDestroyDescriptorPool(dev, static_cast<VkDescriptorPool>(DescriptorPool), nullptr);
		DescriptorPool = nullptr;
		DescriptorSet = nullptr;
	}
	if (DescriptorSetLayout) {
		vkDestroyDescriptorSetLayout(dev, static_cast<VkDescriptorSetLayout>(DescriptorSetLayout), nullptr);
		DescriptorSetLayout = nullptr;
	}
	FreeList_.Clear();
	StorageImageFreeList_.Clear();
	SampledImageFreeList_.Clear();
	SamplerFreeList_.Clear();
}

OaU32 OaBindlessHeap::Register(
	const OaVkDevice& InDevice, const OaVkBuffer& InBuffer)
{
	OaSpinlockGuard guard(Lock_);
	if (FreeList_.Empty()) {
		OA_LOG_ERROR(OaLogComponent::Core, "Bindless heap: out of slots (%u max)",
			OA_BINDLESS_CAPACITY);
		return OA_BINDLESS_INVALID;
	}

	OaU32 index = FreeList_.Back();
	FreeList_.PopBack();

	VkDevice dev = static_cast<VkDevice>(InDevice.Device);

	VkDescriptorBufferInfo bufInfo{};
	bufInfo.buffer = static_cast<VkBuffer>(InBuffer.Buffer);
	bufInfo.offset = 0;
	bufInfo.range = InBuffer.DescriptorRange();

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = static_cast<VkDescriptorSet>(DescriptorSet);
	write.dstBinding = 0;
	write.dstArrayElement = index;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	write.pBufferInfo = &bufInfo;

	vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
	return index;
}

void OaBindlessHeap::Deregister(OaU32 InIndex) {
	if (InIndex == OA_BINDLESS_INVALID) return;
	OaSpinlockGuard guard(Lock_);
	FreeList_.PushBack(InIndex);
}

void OaBindlessHeap::Update(
	const OaVkDevice& InDevice, OaU32 InIndex, const OaVkBuffer& InBuffer)
{
	if (InIndex == OA_BINDLESS_INVALID) return;
	OaSpinlockGuard guard(Lock_);

	VkDevice dev = static_cast<VkDevice>(InDevice.Device);

	VkDescriptorBufferInfo bufInfo{};
	bufInfo.buffer = static_cast<VkBuffer>(InBuffer.Buffer);
	bufInfo.offset = 0;
	bufInfo.range = InBuffer.DescriptorRange();

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = static_cast<VkDescriptorSet>(DescriptorSet);
	write.dstBinding = 0;
	write.dstArrayElement = InIndex;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	write.pBufferInfo = &bufInfo;

	vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
}

static OaU32 OaBindlessPopSlot(OaVec<OaU32>& InOutFreeList, const char* InKind)
{
	if (InOutFreeList.Empty()) {
		OA_LOG_ERROR(OaLogComponent::Core, "Bindless heap: out of %s slots", InKind);
		return OA_BINDLESS_INVALID;
	}
	OaU32 index = InOutFreeList.Back();
	InOutFreeList.PopBack();
	return index;
}

static void OaBindlessWriteImage(
	VkDevice InDevice,
	VkDescriptorSet InSet,
	OaU32 InBinding,
	OaU32 InIndex,
	VkDescriptorType InType,
	VkImageView InView,
	VkImageLayout InLayout,
	VkSampler InSampler = VK_NULL_HANDLE)
{
	VkDescriptorImageInfo info{};
	info.imageView = InView;
	info.imageLayout = InLayout;
	info.sampler = InSampler;

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = InSet;
	write.dstBinding = InBinding;
	write.dstArrayElement = InIndex;
	write.descriptorCount = 1;
	write.descriptorType = InType;
	write.pImageInfo = &info;
	vkUpdateDescriptorSets(InDevice, 1, &write, 0, nullptr);
}

OaU32 OaBindlessHeap::RegisterStorageImage(const OaVkDevice& InDevice, VkImageView InView, VkImageLayout InLayout)
{
	if (!InView) return OA_BINDLESS_INVALID;
	OaSpinlockGuard guard(Lock_);
	OaU32 index = OaBindlessPopSlot(StorageImageFreeList_, "storage image");
	if (index == OA_BINDLESS_INVALID) return index;
	UpdateStorageImage(InDevice, index, InView, InLayout);
	return index;
}

void OaBindlessHeap::UpdateStorageImage(const OaVkDevice& InDevice, OaU32 InIndex, VkImageView InView, VkImageLayout InLayout)
{
	if (InIndex == OA_BINDLESS_INVALID || !InView) return;
	OaBindlessWriteImage(
		static_cast<VkDevice>(InDevice.Device),
		static_cast<VkDescriptorSet>(DescriptorSet),
		1,
		InIndex,
		VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		InView,
		InLayout);
}

void OaBindlessHeap::DeregisterStorageImage(OaU32 InIndex) {
	if (InIndex == OA_BINDLESS_INVALID) return;
	OaSpinlockGuard guard(Lock_);
	StorageImageFreeList_.PushBack(InIndex);
}

OaU32 OaBindlessHeap::RegisterSampledImage(const OaVkDevice& InDevice, VkImageView InView, VkImageLayout InLayout) {
	if (!InView) return OA_BINDLESS_INVALID;
	OaSpinlockGuard guard(Lock_);
	OaU32 index = OaBindlessPopSlot(SampledImageFreeList_, "sampled image");
	if (index == OA_BINDLESS_INVALID) return index;
	UpdateSampledImage(InDevice, index, InView, InLayout);
	return index;
}

void OaBindlessHeap::UpdateSampledImage(const OaVkDevice& InDevice, OaU32 InIndex, VkImageView InView, VkImageLayout InLayout) {
	if (InIndex == OA_BINDLESS_INVALID || !InView) return;
	OaBindlessWriteImage(
		static_cast<VkDevice>(InDevice.Device),
		static_cast<VkDescriptorSet>(DescriptorSet),
		2,
		InIndex,
		VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
		InView,
		InLayout
	);
}

void OaBindlessHeap::DeregisterSampledImage(OaU32 InIndex) {
	if (InIndex == OA_BINDLESS_INVALID) return;
	OaSpinlockGuard guard(Lock_);
	SampledImageFreeList_.PushBack(InIndex);
}

OaU32 OaBindlessHeap::RegisterSampler(const OaVkDevice& InDevice, VkSampler InSampler) {
	if (!InSampler) return OA_BINDLESS_INVALID;
	OaSpinlockGuard guard(Lock_);
	OaU32 index = OaBindlessPopSlot(SamplerFreeList_, "sampler");
	if (index == OA_BINDLESS_INVALID) return index;
	UpdateSampler(InDevice, index, InSampler);
	return index;
}

void OaBindlessHeap::UpdateSampler(const OaVkDevice& InDevice, OaU32 InIndex, VkSampler InSampler) {
	if (InIndex == OA_BINDLESS_INVALID || !InSampler) return;
	OaBindlessWriteImage(
		static_cast<VkDevice>(InDevice.Device),
		static_cast<VkDescriptorSet>(DescriptorSet),
		3,
		InIndex,
		VK_DESCRIPTOR_TYPE_SAMPLER,
		VK_NULL_HANDLE,
		VK_IMAGE_LAYOUT_UNDEFINED,
		InSampler);
}

void OaBindlessHeap::DeregisterSampler(OaU32 InIndex){
	if (InIndex == OA_BINDLESS_INVALID) return;
	OaSpinlockGuard guard(Lock_);
	SamplerFreeList_.PushBack(InIndex);
}
