#include <oa/runtime/loader.h>

#include <oa/core/std/sync.h>
#include <vkl/vkl.h>

namespace {

oa::Mutex& loaderMutex() {
	static oa::Mutex mutex;
	return mutex;
}

} // namespace

oa::Status oavk::initializeLoader(
	PFN_vkGetInstanceProcAddr inCustomLoader
) {
	oa::ScopedLock lock(loaderMutex());

	const PFN_vkGetInstanceProcAddr selected = vklGetInstanceProcAddr();
	if (selected != nullptr) {
		if (inCustomLoader != nullptr && inCustomLoader != selected) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"a different process-global Vulkan loader is already selected");
		}
		return oa::Status::ok();
	}

	if (inCustomLoader != nullptr) {
		vklInitCustom(inCustomLoader);
	} else if (vklInit() != VK_SUCCESS) {
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			"Vulkan loader initialization failed");
	}

	if (vklGetInstanceProcAddr() == nullptr) {
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			"Vulkan loader did not provide vkGetInstanceProcAddr");
	}
	return oa::Status::ok();
}
