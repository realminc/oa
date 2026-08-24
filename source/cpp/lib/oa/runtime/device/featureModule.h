// OA vulkan Device Feature Module System
// Modular device capability probing and feature chain building.
#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/runtime/oaVk.h>
#include <oa/runtime/init.h>  // PhysicalExtensionProbe, DeviceFeatureBundle

namespace oavk {

// ─────────────────────────────────────────────────────────────────────────────
// FeatureModule — base interface for device feature modules
//
// Each module is responsible for:
// 1. Probing available extensions
// 2. Querying supported features
// 3. Building the feature chain for device creation
// 4. collecting enabled extensions
// 5. Declaring dependencies on other modules
// ─────────────────────────────────────────────────────────────────────────────
class FeatureModule {
public:
	virtual ~FeatureModule() = default;

	// Module identification
	virtual oa::StringView name() const = 0;

	// phase 1: probe which extensions are available
	virtual void probeExtensions(
		const oa::Vec<VkExtensionProperties>& inAvailableExtensions,
		PhysicalExtensionProbe& outProbe
	) = 0;

	// phase 2: query which features are supported
	virtual void queryFeatures(
		const OaVkInstanceTable& inDispatch,
		VkPhysicalDevice inPhysicalDevice,
		DeviceFeatureBundle& outBundle
	) = 0;

	// phase 3: Build the feature chain for device creation
	virtual void buildFeatureChain(
		DeviceFeatureBundle& inOutBundle
	) = 0;

	// phase 4: Collect extensions to enable
	virtual void collectExtensions(
		const PhysicalExtensionProbe& inProbe,
		const DeviceFeatureBundle& inBundle,
		oa::Vec<const char*>& outExtensions
	) = 0;

	// Module dependencies (must be loaded before this module)
	virtual oa::Vec<oa::StringView> dependencies() const { return {}; }
};

[[nodiscard]] oa::UniquePtr<FeatureModule> createCoreFeatures();
[[nodiscard]] oa::UniquePtr<FeatureModule> createMlFeatures();
[[nodiscard]] oa::UniquePtr<FeatureModule> createVisionFeatures();
[[nodiscard]] oa::UniquePtr<FeatureModule> createAudioFeatures();
[[nodiscard]] oa::UniquePtr<FeatureModule> createRenderFeatures();

} // namespace oavk
