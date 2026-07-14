// OA Vulkan Render Device Implementation
#include <Oa/Runtime/Device.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Core/Log.h>


VkSurfaceFormatKHR OaVkRenderDevice::SelectSwapchainFormat(VkSurfaceKHR InSurface) const {
	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(PhysicalDevice);
	
	OaU32 formatCount = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR(phys, InSurface, &formatCount, nullptr);
	
	if (formatCount == 0) {
		OA_LOG_ERROR(OaLogComponent::Core, "No surface formats available");
		return VkSurfaceFormatKHR{
			.format = VK_FORMAT_B8G8R8A8_SRGB,
			.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
		};
	}
	
	OaVec<VkSurfaceFormatKHR> formats(formatCount);
	vkGetPhysicalDeviceSurfaceFormatsKHR(phys, InSurface, &formatCount, formats.Data());
	
	// Prefer BGRA8 SRGB with nonlinear color space
	for (const auto& format : formats) {
		if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
			format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			return format;
		}
	}
	
	// Fallback to first available format
	return formats[0];
}


VkPresentModeKHR OaVkRenderDevice::SelectPresentMode(VkSurfaceKHR InSurface) const {
	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(PhysicalDevice);
	
	OaU32 modeCount = 0;
	vkGetPhysicalDeviceSurfacePresentModesKHR(phys, InSurface, &modeCount, nullptr);
	
	if (modeCount == 0) {
		return VK_PRESENT_MODE_FIFO_KHR;  // Always available
	}
	
	OaVec<VkPresentModeKHR> modes(modeCount);
	vkGetPhysicalDeviceSurfacePresentModesKHR(phys, InSurface, &modeCount, modes.Data());
	
	// Prefer mailbox (triple buffering) for low latency
	for (const auto& mode : modes) {
		if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
			return mode;
		}
	}
	
	// Fallback to FIFO (vsync, always available)
	return VK_PRESENT_MODE_FIFO_KHR;
}
