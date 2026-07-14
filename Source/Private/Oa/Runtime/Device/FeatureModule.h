// OA Vulkan Device Feature Module System
// Modular device capability probing and feature chain building.
#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Runtime/Init.h>  // OaVkPhysExtProbe, OaVkDeviceFeatureBundle


// ─────────────────────────────────────────────────────────────────────────────
// OaVkFeatureModule — Base interface for device feature modules
//
// Each module is responsible for:
// 1. Probing available extensions
// 2. Querying supported features
// 3. Building the feature chain for device creation
// 4. Collecting enabled extensions
// 5. Declaring dependencies on other modules
// ─────────────────────────────────────────────────────────────────────────────
class OaVkFeatureModule {
public:
	virtual ~OaVkFeatureModule() = default;

	// Module identification
	virtual OaStringView Name() const = 0;

	// Phase 1: Probe which extensions are available
	virtual void ProbeExtensions(
		const OaVec<VkExtensionProperties>& InAvailableExtensions,
		OaVkPhysExtProbe& OutProbe
	) = 0;

	// Phase 2: Query which features are supported
	virtual void QueryFeatures(
		VkPhysicalDevice InPhysicalDevice,
		OaVkDeviceFeatureBundle& OutBundle
	) = 0;

	// Phase 3: Build the feature chain for device creation
	virtual void BuildFeatureChain(
		OaVkDeviceFeatureBundle& InOutBundle
	) = 0;

	// Phase 4: Collect extensions to enable
	virtual void CollectExtensions(
		const OaVkPhysExtProbe& InProbe,
		const OaVkDeviceFeatureBundle& InBundle,
		OaVec<const char*>& OutExtensions
	) = 0;

	// Module dependencies (must be loaded before this module)
	virtual OaVec<OaStringView> Dependencies() const { return {}; }
};
