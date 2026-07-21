#include <Oa/Runtime/Cmd.h>
#include <Oa/Runtime/Device.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Pipeline.h>
#include <Oa/Runtime/OaVk.h>

OaResult<OaVkCmd> OaVkCmd::Create(const OaVkDevice& InDevice) {
	VkDevice dev = static_cast<VkDevice>(InDevice.Device);

	VkCommandPoolCreateInfo cpCI = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = InDevice.Queues.ComputeQueueFamily,
	};

	VkCommandPool pool = VK_NULL_HANDLE;
	VkResult r = vkCreateCommandPool(dev, &cpCI, nullptr, &pool);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkCreateCommandPool failed");
	}

	VkCommandBufferAllocateInfo cbAI = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1
	};

	VkCommandBuffer cb = VK_NULL_HANDLE;
	r = vkAllocateCommandBuffers(dev, &cbAI, &cb);
	if (r != VK_SUCCESS) {
		vkDestroyCommandPool(dev, pool, nullptr);
		return OaStatus::Error(OaStatusCode::VulkanError, "vkAllocateCommandBuffers failed");
	}

	OaVkCmd cmd;
	cmd.CommandPool = pool;
	cmd.CommandBuffer = cb;
	return cmd;
}

void OaVkCmd::Destroy(const OaVkDevice& InDevice) {
	if (CommandPool) {
		VkDevice dev = static_cast<VkDevice>(InDevice.Device);
		// Explicitly free command buffer before destroying pool (Vulkan validation requirement)
		if (CommandBuffer) {
			VkCommandBuffer cb = static_cast<VkCommandBuffer>(CommandBuffer);
			vkFreeCommandBuffers(dev, static_cast<VkCommandPool>(CommandPool), 1, &cb);
			CommandBuffer = nullptr;
		}
		vkDestroyCommandPool(dev, static_cast<VkCommandPool>(CommandPool), nullptr);
		CommandPool = nullptr;
	}
}

OaStatus OaVkCmd::Begin() {
	VkCommandBufferBeginInfo bi ={
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};
	VkResult r = vkBeginCommandBuffer(static_cast<VkCommandBuffer>(CommandBuffer), &bi);
	if (r != VK_SUCCESS) return OaStatus::Error(OaStatusCode::VulkanError, "vkBeginCommandBuffer failed");
	return OaStatus::Ok();
}

OaStatus OaVkCmd::End() {
	VkResult r = vkEndCommandBuffer(static_cast<VkCommandBuffer>(CommandBuffer));
	if (r != VK_SUCCESS) return OaStatus::Error(OaStatusCode::VulkanError, "vkEndCommandBuffer failed");
	return OaStatus::Ok();
}

void OaVkCmd::BindPipeline(const OaComputePipeline& InPipeline) {
	vkCmdBindPipeline(
		static_cast<VkCommandBuffer>(CommandBuffer),
		VK_PIPELINE_BIND_POINT_COMPUTE,
		static_cast<VkPipeline>(InPipeline.Pipeline)
	);
}

void OaVkCmd::BindDescriptors(const OaComputePipeline& InPipeline) {
	BindDescriptorSet(InPipeline.PipelineLayout, InPipeline.DescriptorSet);
}

void OaVkCmd::BindDescriptorSet(void* InPipelineLayout, void* InDescriptorSet) {
	VkDescriptorSet ds = static_cast<VkDescriptorSet>(InDescriptorSet);
	vkCmdBindDescriptorSets(
		static_cast<VkCommandBuffer>(CommandBuffer),
		VK_PIPELINE_BIND_POINT_COMPUTE,
		static_cast<VkPipelineLayout>(InPipelineLayout),
		0, 1, &ds, 0, nullptr
	);
}

void OaVkCmd::PushConstants(const OaComputePipeline& InPipeline, const void* InData, OaU32 InSize) {
	vkCmdPushConstants(
		static_cast<VkCommandBuffer>(CommandBuffer),
		static_cast<VkPipelineLayout>(InPipeline.PipelineLayout),
		VK_SHADER_STAGE_COMPUTE_BIT,
		0, InSize, InData
	);
}

void OaVkCmd::Dispatch(OaU32 InGroupsX, OaU32 InGroupsY, OaU32 InGroupsZ) {
	vkCmdDispatch(static_cast<VkCommandBuffer>(CommandBuffer), InGroupsX, InGroupsY, InGroupsZ);
}

void OaVkCmd::BufferBarrier() {
	VkMemoryBarrier2 barrier = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_HOST_BIT,
		.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_HOST_READ_BIT,
	};

	VkDependencyInfo dep = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.memoryBarrierCount = 1,
		.pMemoryBarriers = &barrier,
	};

	vkCmdPipelineBarrier2(static_cast<VkCommandBuffer>(CommandBuffer), &dep);
}

void OaVkCmd::CopyBuffer(const OaVkBuffer& InSrc, const OaVkBuffer& InDst, OaU64 InSize) {
	VkBufferCopy region = {
		.size = InSize,
	};
	vkCmdCopyBuffer(
		static_cast<VkCommandBuffer>(CommandBuffer),
		static_cast<VkBuffer>(InSrc.Buffer),
		static_cast<VkBuffer>(InDst.Buffer),
		1, &region
	);
}

OaStatus OaVkCmd::Submit(const OaVkDevice& InDevice) {
	VkCommandBuffer cb = static_cast<VkCommandBuffer>(CommandBuffer);
	VkSubmitInfo si = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &cb,
	};

	VkResult res = vkQueueSubmit(static_cast<VkQueue>(InDevice.Queues.ComputeQueue), 1, &si, VK_NULL_HANDLE);
	if (res != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkQueueSubmit failed");
	}
	return OaStatus::Ok();
}

OaStatus OaVkCmd::SubmitAndWait(const OaVkDevice& InDevice) {
	VkDevice device = static_cast<VkDevice>(InDevice.Device);
	VkFenceCreateInfo fenceInfo = {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
	};
	VkFence fence = VK_NULL_HANDLE;
	VkResult r = vkCreateFence(device, &fenceInfo, nullptr, &fence);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError,
			"vkCreateFence (command completion) failed");
	}

	VkCommandBuffer cb = static_cast<VkCommandBuffer>(CommandBuffer);
	VkSubmitInfo submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &cb,
	};
	r = vkQueueSubmit(
		static_cast<VkQueue>(InDevice.Queues.ComputeQueue), 1, &submitInfo, fence);
	if (r == VK_SUCCESS) {
		r = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
	}
	vkDestroyFence(device, fence, nullptr);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError,
			"command submission or fence wait failed");
	}
	return OaStatus::Ok();
}
