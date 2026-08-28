#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <vkl/vkl.h>
#include <oa/runtime/init.h>

// VkInstance creation shared by single-device and mesh paths.
// Flow: process loader selection → createInstance → caller-owned table.

namespace oavk {

class Instance {
public:

	// Methods
	// Merges oavk::InstanceExtensionNames with inExtraInstanceExtensions (null/empty skipped, strcmp dedupe).
	// App patch is clamped to 12 bits for VK_MAKE_VERSION(0, 0, patch). Engine name "OA", api oavk::MinApiVersion.
	[[nodiscard]] static oa::Result<VkInstance> createInstance(
		oa::StringView inAppName,
		oa::U32 inAppVersionPatch,
		oa::Bool inEnableValidation,
		oa::Span<const char* const> inExtraInstanceExtensions = {},
		oa::Bool inWantsPresentation = false,
		PFN_vkGetInstanceProcAddr inCustomLoader = nullptr
	);

	static void destroyInstance(
		const VklInstanceTable& inDispatch,
		VkInstance inInstance) noexcept;
};

} // namespace oavk
