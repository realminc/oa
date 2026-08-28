#pragma once

#include <oa/core/status.h>
#include <vulkan/vulkan_core.h>

namespace oavk {

// Selects the process-global Vulkan loader exactly once. Reusing the selected
// loader is valid; attempting to replace it while OA may own live engines is
// rejected. Instance and device dispatch remain engine-owned tables.
[[nodiscard]] oa::Status initializeLoader(
	PFN_vkGetInstanceProcAddr inCustomLoader = nullptr
);

} // namespace oavk
