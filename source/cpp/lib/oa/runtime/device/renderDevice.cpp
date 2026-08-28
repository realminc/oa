// OA vulkan Render device Implementation
#include <oa/runtime/device.h>
#include <vkl/vkl.h>
#include <oa/core/log.h>


VkSurfaceFormatKHR oavk::RenderDevice::selectSwapchainFormat(VkSurfaceKHR inSurface) const {
	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(physicalDevice);
	
	oa::U32 formatCount = 0;
	instanceDispatch.vkGetPhysicalDeviceSurfaceFormatsKHR(
		phys, inSurface, &formatCount, nullptr);
	
	if (formatCount == 0) {
		OaLogError(oa::LogComponent::Runtime, "No surface formats available");
		return VkSurfaceFormatKHR{
			.format = VK_FORMAT_B8G8R8A8_SRGB,
			.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
		};
	}
	
	oa::Vector<VkSurfaceFormatKHR> formats(formatCount);
	instanceDispatch.vkGetPhysicalDeviceSurfaceFormatsKHR(
		phys, inSurface, &formatCount, formats.data());
	
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


VkPresentModeKHR oavk::RenderDevice::selectPresentMode(VkSurfaceKHR inSurface) const {
	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(physicalDevice);
	
	oa::U32 modeCount = 0;
	instanceDispatch.vkGetPhysicalDeviceSurfacePresentModesKHR(
		phys, inSurface, &modeCount, nullptr);
	
	if (modeCount == 0) {
		return VK_PRESENT_MODE_FIFO_KHR;  // Always available
	}
	
	oa::Vector<VkPresentModeKHR> modes(modeCount);
	instanceDispatch.vkGetPhysicalDeviceSurfacePresentModesKHR(
		phys, inSurface, &modeCount, modes.data());
	
	// Prefer mailbox (triple buffering) for low latency
	for (const auto& mode : modes) {
		if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
			return mode;
		}
	}
	
	// Fallback to FIFO (vsync, always available)
	return VK_PRESENT_MODE_FIFO_KHR;
}
