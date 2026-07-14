// OA Vulkan Audio Features Module
// Handles audio compute features (future-proofing)
// Currently a stub - audio processing uses compute shaders

#include "../FeatureModule.h"
#include <Oa/Core/Log.h>


class OaVkAudioFeatures : public OaVkFeatureModule {
public:
	OaStringView Name() const override {
		return "Audio";
	}

	void ProbeExtensions(
		const OaVec<VkExtensionProperties>& InAvailableExtensions,
		OaVkPhysExtProbe& OutProbe
	) override {
		// Audio processing currently uses standard compute shaders
		// No audio-specific Vulkan extensions at this time
		// Future: audio-specific DSP extensions if they emerge
	}

	void QueryFeatures(
		VkPhysicalDevice InPhysicalDevice,
		OaVkDeviceFeatureBundle& OutBundle
	) override {
		// No audio-specific features to query yet
		// Audio processing relies on compute capabilities from Core module
	}

	void BuildFeatureChain(
		OaVkDeviceFeatureBundle& InOutBundle
	) override {
		// No audio-specific feature chain needed
	}

	void CollectExtensions(
		const OaVkPhysExtProbe& InProbe,
		const OaVkDeviceFeatureBundle& InBundle,
		OaVec<const char*>& OutExtensions
	) override {
		// No audio-specific extensions to enable
	}

	OaVec<OaStringView> Dependencies() const override {
		return {"Core"};
	}
};


OaUniquePtr<OaVkFeatureModule> OaVkCreateAudioFeatures() {
	return OaMakeUniquePtr<OaVkAudioFeatures>();
}
