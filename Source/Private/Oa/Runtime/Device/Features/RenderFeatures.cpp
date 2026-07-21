// OA Vulkan Render Features Module
// Handles graphics/presentation features:
// - Graphics queue
// - Swapchain support
// - Present queue

#include "../FeatureModule.h"
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Init.h>
#include <cstring>


class OaVkRenderFeatures : public OaVkFeatureModule {
public:
	OaStringView Name() const override {
		return "Render";
	}

	void ProbeExtensions(
		const OaVec<VkExtensionProperties>& InAvailableExtensions,
		OaVkPhysExtProbe& OutProbe
	) override {
		for (const auto& ext : InAvailableExtensions) {
			if (strcmp(ext.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
				OutProbe.KhrSwapchain = true;
			}
			else if (strcmp(ext.extensionName,
				VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME) == 0) {
				OutProbe.KhrSwapchainMaintenance1 = true;
				MaintenanceExtensionPresent_ = true;
			}
			else if (strcmp(ext.extensionName,
				VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME) == 0) {
				OutProbe.ExtSwapchainMaintenance1 = true;
				MaintenanceExtensionPresent_ = true;
			}
		}
	}

	void QueryFeatures(
		VkPhysicalDevice InPhysicalDevice,
		OaVkDeviceFeatureBundle& OutBundle
	) override {
		if (!MaintenanceExtensionPresent_) return;
		OutBundle.SupportedSwapchainMaintenance1 = {};
		OutBundle.SupportedSwapchainMaintenance1.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR;
		VkPhysicalDeviceFeatures2 features{};
		features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		features.pNext = &OutBundle.SupportedSwapchainMaintenance1;
		vkGetPhysicalDeviceFeatures2(InPhysicalDevice, &features);
		OutBundle.HasSwapchainMaintenance1 =
			OutBundle.SupportedSwapchainMaintenance1.swapchainMaintenance1 == VK_TRUE;
	}

	void BuildFeatureChain(
		OaVkDeviceFeatureBundle& InOutBundle
	) override {
		if (!InOutBundle.HasSwapchainMaintenance1) return;
		InOutBundle.SwapchainMaintenance1Features = {};
		InOutBundle.SwapchainMaintenance1Features.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR;
		InOutBundle.SwapchainMaintenance1Features.pNext =
			InOutBundle.Features13.pNext;
		InOutBundle.SwapchainMaintenance1Features.swapchainMaintenance1 = VK_TRUE;
		InOutBundle.Features13.pNext = &InOutBundle.SwapchainMaintenance1Features;
	}

	void CollectExtensions(
		const OaVkPhysExtProbe& InProbe,
		const OaVkDeviceFeatureBundle& InBundle,
		OaVec<const char*>& OutExtensions
	) override {
		// Swapchain extension (required for presentation)
		if (InProbe.KhrSwapchain) {
			OutExtensions.PushBack(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
			if (InBundle.HasSwapchainMaintenance1) {
				if (InProbe.KhrSwapchainMaintenance1) {
					OutExtensions.PushBack(VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
				} else if (InProbe.ExtSwapchainMaintenance1) {
					OutExtensions.PushBack(VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
				}
			}
		}
	}

	OaVec<OaStringView> Dependencies() const override {
		return {"Core"};
	}

private:
	bool MaintenanceExtensionPresent_ = false;
};


OaUniquePtr<OaVkFeatureModule> OaVkCreateRenderFeatures() {
	return OaMakeUniquePtr<OaVkRenderFeatures>();
}
