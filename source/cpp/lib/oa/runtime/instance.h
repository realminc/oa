#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/runtime/oaVk.h>
#include <oa/runtime/init.h>

// VkInstance creation shared by single-device and mesh paths.
// Flow: oaVkInit → createInstance → caller-owned OaVkInstanceTable.

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
		oa::Bool inWantsPresentation = false
	);

	static void destroyInstance(
		const OaVkInstanceTable& inDispatch,
		VkInstance inInstance) noexcept;
};

} // namespace oavk
