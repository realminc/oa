#pragma once

#include <oa/core/std.h>
#include <oa/runtime/stream.h>
#include <oa/runtime/swapchain.h>
#include "engine/queueSubmitRoute.h"

#include <vulkan/vulkan.h>

namespace oa {

// Engine-owned lifetime payload for a presenter facade abandoned without an
// explicit close. Abandonment transfers the still-attached surface too, because
// it must outlive the swapchain and cannot be destroyed after engine close has
// destroyed the VkInstance. Every WSI child stays here until completion is
// proven at that explicit boundary.
struct RetiredPresenter {
	oa::Swapchain swapchain;
	VkRenderPass renderPass = VK_NULL_HANDLE;
	oa::Vec<VkFramebuffer> framebuffers;
	VkCommandPool commandPool = VK_NULL_HANDLE;
	oa::Vec<VkCommandBuffer> commandBuffers;
	VkDescriptorPool imGuiPool = VK_NULL_HANDLE;
	oa::Bool imGuiReady = false;
	oa::Vec<oa::UniquePtr<oavk::Stream>> graphicsStreams;
	void* presentQueue = nullptr;
	oavk::QueueSubmitRoute presentQueueRoute = oavk::QueueSubmitRoute::Unknown;
	oa::Bool hasSwapchainMaintenance1 = false;
	oa::Bool ownsAbandonedSurface = false;
};

} // namespace oa
