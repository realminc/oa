#pragma once

#include <Oa/Core/Std.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Runtime/Swapchain.h>
#include "Engine/QueueSubmitRoute.h"

#include <vector>
#include <vulkan/vulkan.h>

// Engine-owned lifetime payload for a presenter facade abandoned without an
// explicit close. Abandonment transfers the still-attached surface too, because
// it must outlive the swapchain and cannot be destroyed after engine close has
// destroyed the VkInstance. Every WSI child stays here until completion is
// proven at that explicit boundary.
struct OaRetiredPresenter {
	OaSwapchain Swapchain;
	VkRenderPass RenderPass = VK_NULL_HANDLE;
	std::vector<VkFramebuffer> Framebuffers;
	VkCommandPool CommandPool = VK_NULL_HANDLE;
	std::vector<VkCommandBuffer> CommandBuffers;
	VkDescriptorPool ImGuiPool = VK_NULL_HANDLE;
	OaBool ImGuiReady = false;
	OaVec<OaUniquePtr<OaVkStream>> GraphicsStreams;
	void* PresentQueue = nullptr;
	OaQueueSubmitRoute PresentQueueRoute = OaQueueSubmitRoute::Unknown;
	OaBool HasSwapchainMaintenance1 = false;
	OaBool OwnsAbandonedSurface = false;
};
