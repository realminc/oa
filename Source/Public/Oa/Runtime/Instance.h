#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Runtime/Init.h>

// VkInstance creation shared by single-device and mesh paths.
// Flow: OaVkInit → OaVkInstance::CreateInstance → OaVkLoadInstance (caller).

class OaVkInstance {
public:

	// Methods
	// Merges OaVkInstanceExtensionNames with InExtraInstanceExtensions (null/empty skipped, strcmp dedupe).
	// App patch is clamped to 12 bits for VK_MAKE_VERSION(0, 0, patch). Engine name "OA", api OaVkMinApiVersion.
	[[nodiscard]] static OaResult<VkInstance> CreateInstance(
		OaStringView InAppName,
		OaU32 InAppVersionPatch,
		OaBool InEnableValidation,
		OaSpan<const char* const> InExtraInstanceExtensions = {},
		OaBool InWantsPresentation = false
	);

	static void DestroyInstance(VkInstance InInstance) noexcept;
};

