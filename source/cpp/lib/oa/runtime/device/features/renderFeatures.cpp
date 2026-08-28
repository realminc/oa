// OA vulkan Render features Module
// Handles graphics/presentation features:
// - Graphics queue
// - Swapchain support
// - Present queue

#include "../featureModule.h"
#include <oa/core/log.h>
#include <oa/runtime/init.h>
#include <string.h>


class RenderFeatures : public oavk::FeatureModule {
public:
	oa::StringView name() const override {
		return "Render";
	}

	void probeExtensions(
		const oa::Vector<VkExtensionProperties>& inAvailableExtensions,
		oavk::PhysicalExtensionProbe& outProbe
	) override {
		for (const auto& ext : inAvailableExtensions) {
			if (strcmp(ext.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
				outProbe.khrSwapchain = true;
			}
			else if (strcmp(ext.extensionName,
				VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME) == 0) {
				outProbe.khrSwapchainMaintenance1 = true;
				maintenanceExtensionPresent_ = true;
			}
			else if (strcmp(ext.extensionName,
				VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME) == 0) {
				outProbe.extSwapchainMaintenance1 = true;
				maintenanceExtensionPresent_ = true;
			}
		}
	}

	void queryFeatures(
		const VklInstanceTable& inDispatch,
		VkPhysicalDevice inPhysicalDevice,
		oavk::DeviceFeatureBundle& outBundle
	) override {
		if (!maintenanceExtensionPresent_) return;
		outBundle.supportedSwapchainMaintenance1 = {};
		outBundle.supportedSwapchainMaintenance1.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR;
		VkPhysicalDeviceFeatures2 features{};
		features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		features.pNext = &outBundle.supportedSwapchainMaintenance1;
		inDispatch.vkGetPhysicalDeviceFeatures2(inPhysicalDevice, &features);
		outBundle.hasSwapchainMaintenance1 =
			outBundle.supportedSwapchainMaintenance1.swapchainMaintenance1 == VK_TRUE;
	}

	void buildFeatureChain(
		oavk::DeviceFeatureBundle& inOutBundle
	) override {
		if (!inOutBundle.hasSwapchainMaintenance1) return;
		inOutBundle.swapchainMaintenance1Features = {};
		inOutBundle.swapchainMaintenance1Features.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR;
		inOutBundle.swapchainMaintenance1Features.pNext =
			inOutBundle.features13.pNext;
		inOutBundle.swapchainMaintenance1Features.swapchainMaintenance1 = VK_TRUE;
		inOutBundle.features13.pNext = &inOutBundle.swapchainMaintenance1Features;
	}

	void collectExtensions(
		const oavk::PhysicalExtensionProbe& inProbe,
		const oavk::DeviceFeatureBundle& inBundle,
		oa::Vector<const char*>& outExtensions
	) override {
		// Swapchain extension (required for presentation)
		if (inProbe.khrSwapchain) {
			outExtensions.pushBack(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
			if (inBundle.hasSwapchainMaintenance1) {
				if (inProbe.khrSwapchainMaintenance1) {
					outExtensions.pushBack(VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
				} else if (inProbe.extSwapchainMaintenance1) {
					outExtensions.pushBack(VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
				}
			}
		}
	}

	oa::Vector<oa::StringView> dependencies() const override {
		return {"Core"};
	}

private:
	bool maintenanceExtensionPresent_ = false;
};


oa::UniquePtr<oavk::FeatureModule> oavk::createRenderFeatures() {
	return oa::makeUnique<RenderFeatures>();
}
