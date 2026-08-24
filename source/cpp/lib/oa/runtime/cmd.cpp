#include <oa/runtime/cmd.h>
#include <oa/runtime/device.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/pipeline.h>
#include <oa/runtime/oaVk.h>
#include "dispatchValidation.h"

oa::Result<oavk::Command> oavk::Command::create(const oavk::Device& inDevice) {
	VkDevice dev = static_cast<VkDevice>(inDevice.device);

	VkCommandPoolCreateInfo cpCI = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = inDevice.queues.computeQueueFamily,
	};

	VkCommandPool pool = VK_NULL_HANDLE;
	VkResult r = inDevice.deviceDispatch.vkCreateCommandPool(dev, &cpCI, nullptr, &pool);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkCreateCommandPool failed");
	}

	VkCommandBufferAllocateInfo cbAI = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1
	};

	VkCommandBuffer cb = VK_NULL_HANDLE;
	r = inDevice.deviceDispatch.vkAllocateCommandBuffers(dev, &cbAI, &cb);
	if (r != VK_SUCCESS) {
		inDevice.deviceDispatch.vkDestroyCommandPool(dev, pool, nullptr);
		return oa::Status::error(oa::StatusCode::VulkanError, "vkAllocateCommandBuffers failed");
	}

	oavk::Command cmd;
	cmd.commandPool = pool;
	cmd.commandBuffer = cb;
	cmd.deviceDispatch = &inDevice.deviceDispatch;
	return cmd;
}

void oavk::Command::destroy(const oavk::Device& inDevice) {
	if (commandPool) {
		VkDevice dev = static_cast<VkDevice>(inDevice.device);
		// Explicitly free command buffer before destroying pool (vulkan validation requirement)
		if (commandBuffer) {
			VkCommandBuffer cb = static_cast<VkCommandBuffer>(commandBuffer);
			deviceDispatch->vkFreeCommandBuffers(dev, static_cast<VkCommandPool>(commandPool), 1, &cb);
			commandBuffer = nullptr;
		}
		deviceDispatch->vkDestroyCommandPool(dev, static_cast<VkCommandPool>(commandPool), nullptr);
		commandPool = nullptr;
	}
	deviceDispatch = nullptr;
}

oa::Status oavk::Command::begin() {
	VkCommandBufferBeginInfo bi ={
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};
	VkResult r = deviceDispatch->vkBeginCommandBuffer(static_cast<VkCommandBuffer>(commandBuffer), &bi);
	if (r != VK_SUCCESS) return oa::Status::error(oa::StatusCode::VulkanError, "vkBeginCommandBuffer failed");
	return oa::Status::ok();
}

oa::Status oavk::Command::end() {
	VkResult r = deviceDispatch->vkEndCommandBuffer(static_cast<VkCommandBuffer>(commandBuffer));
	if (r != VK_SUCCESS) return oa::Status::error(oa::StatusCode::VulkanError, "vkEndCommandBuffer failed");
	return oa::Status::ok();
}

void oavk::Command::bindPipeline(const oa::ComputePipeline& inPipeline) {
	deviceDispatch->vkCmdBindPipeline(
		static_cast<VkCommandBuffer>(commandBuffer),
		VK_PIPELINE_BIND_POINT_COMPUTE,
		static_cast<VkPipeline>(inPipeline.pipeline)
	);
}

void oavk::Command::bindDescriptors(const oa::ComputePipeline& inPipeline) {
	bindDescriptorSet(inPipeline.pipelineLayout, inPipeline.descriptorSet);
}

void oavk::Command::bindDescriptorSet(void* inPipelineLayout, void* inDescriptorSet) {
	VkDescriptorSet ds = static_cast<VkDescriptorSet>(inDescriptorSet);
	deviceDispatch->vkCmdBindDescriptorSets(
		static_cast<VkCommandBuffer>(commandBuffer),
		VK_PIPELINE_BIND_POINT_COMPUTE,
		static_cast<VkPipelineLayout>(inPipelineLayout),
		0, 1, &ds, 0, nullptr
	);
}

void oavk::Command::pushConstants(const oa::ComputePipeline& inPipeline, const void* inData, oa::U32 inSize) {
	deviceDispatch->vkCmdPushConstants(
		static_cast<VkCommandBuffer>(commandBuffer),
		static_cast<VkPipelineLayout>(inPipeline.pipelineLayout),
		VK_SHADER_STAGE_COMPUTE_BIT,
		0, inSize, inData
	);
}

oa::Status oavk::Command::dispatch(
	const oavk::Device& inDevice,
	oa::U32 inGroupsX,
	oa::U32 inGroupsY,
	oa::U32 inGroupsZ)
{
	OA_RETURN_IF_ERROR(oavk::validateDirectComputeDispatch(
		inDevice, inGroupsX, inGroupsY, inGroupsZ));
	deviceDispatch->vkCmdDispatch(static_cast<VkCommandBuffer>(commandBuffer), inGroupsX, inGroupsY, inGroupsZ);
	return oa::Status::ok();
}

void oavk::Command::bufferBarrier() {
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

	deviceDispatch->vkCmdPipelineBarrier2(static_cast<VkCommandBuffer>(commandBuffer), &dep);
}

void oavk::Command::copyBuffer(const oavk::Buffer& inSrc, const oavk::Buffer& inDst, oa::U64 inSize) {
	VkBufferCopy region = {
		.size = inSize,
	};
	deviceDispatch->vkCmdCopyBuffer(
		static_cast<VkCommandBuffer>(commandBuffer),
		static_cast<VkBuffer>(inSrc.buffer),
		static_cast<VkBuffer>(inDst.buffer),
		1, &region
	);
}

oa::Status oavk::Command::submit(const oavk::Device& inDevice) {
	VkCommandBuffer cb = static_cast<VkCommandBuffer>(commandBuffer);
	VkSubmitInfo si = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &cb,
	};

	VkResult res = deviceDispatch->vkQueueSubmit(static_cast<VkQueue>(inDevice.queues.computeQueue), 1, &si, VK_NULL_HANDLE);
	if (res != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkQueueSubmit failed");
	}
	return oa::Status::ok();
}

oa::Status oavk::Command::submitAndWait(const oavk::Device& inDevice) {
	VkDevice device = static_cast<VkDevice>(inDevice.device);
	VkFenceCreateInfo fenceInfo = {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
	};
	VkFence fence = VK_NULL_HANDLE;
	VkResult r = deviceDispatch->vkCreateFence(device, &fenceInfo, nullptr, &fence);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			"vkCreateFence (command completion) failed");
	}

	VkCommandBuffer cb = static_cast<VkCommandBuffer>(commandBuffer);
	VkSubmitInfo submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &cb,
	};
	r = deviceDispatch->vkQueueSubmit(
		static_cast<VkQueue>(inDevice.queues.computeQueue), 1, &submitInfo, fence);
	if (r == VK_SUCCESS) {
		r = deviceDispatch->vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
	}
	deviceDispatch->vkDestroyFence(device, fence, nullptr);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			"command submission or fence wait failed");
	}
	return oa::Status::ok();
}
