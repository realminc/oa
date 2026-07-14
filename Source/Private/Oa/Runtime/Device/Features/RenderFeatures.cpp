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
		}
	}

	void QueryFeatures(
		VkPhysicalDevice InPhysicalDevice,
		OaVkDeviceFeatureBundle& OutBundle
	) override {
		// Graphics/present features are queue-based, no additional feature queries needed
		// Queue family detection happens in queue planning phase
	}

	void BuildFeatureChain(
		OaVkDeviceFeatureBundle& InOutBundle
	) override {
		// Swapchain doesn't require feature enables beyond the extension itself
		// Graphics queue capabilities are detected during queue planning
	}

	void CollectExtensions(
		const OaVkPhysExtProbe& InProbe,
		const OaVkDeviceFeatureBundle& InBundle,
		OaVec<const char*>& OutExtensions
	) override {
		// Swapchain extension (required for presentation)
		if (InProbe.KhrSwapchain) {
			OutExtensions.PushBack(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
		}
	}

	OaVec<OaStringView> Dependencies() const override {
		return {"Core"};
	}
};


OaUniquePtr<OaVkFeatureModule> OaVkCreateRenderFeatures() {
	return OaMakeUniquePtr<OaVkRenderFeatures>();
}
