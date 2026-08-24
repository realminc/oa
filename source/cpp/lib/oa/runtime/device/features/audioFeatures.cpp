// OA vulkan Audio features Module
// Handles audio compute features (future-proofing)
// Currently a stub - audio processing uses compute shaders

#include "../featureModule.h"
#include <oa/core/log.h>


class AudioFeatures : public oavk::FeatureModule {
public:
	oa::StringView name() const override {
		return "Audio";
	}

	void probeExtensions(
		const oa::Vec<VkExtensionProperties>& inAvailableExtensions,
		oavk::PhysicalExtensionProbe& outProbe
	) override {
		// Audio processing currently uses standard compute shaders
		// No audio-specific vulkan extensions at this time
		// Future: audio-specific DSP extensions if they emerge
	}

	void queryFeatures(
		const OaVkInstanceTable& inDispatch,
		VkPhysicalDevice inPhysicalDevice,
		oavk::DeviceFeatureBundle& outBundle
	) override {
		(void)inDispatch;
		(void)inPhysicalDevice;
		(void)outBundle;
		// No audio-specific features to query yet
		// Audio processing relies on compute capabilities from Core module
	}

	void buildFeatureChain(
		oavk::DeviceFeatureBundle& inOutBundle
	) override {
		// No audio-specific feature chain needed
	}

	void collectExtensions(
		const oavk::PhysicalExtensionProbe& inProbe,
		const oavk::DeviceFeatureBundle& inBundle,
		oa::Vec<const char*>& outExtensions
	) override {
		// No audio-specific extensions to enable
	}

	oa::Vec<oa::StringView> dependencies() const override {
		return {"Core"};
	}
};


oa::UniquePtr<oavk::FeatureModule> oavk::createAudioFeatures() {
	return oa::makeUnique<AudioFeatures>();
}
