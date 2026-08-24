#include <oa/runtime/window.h>

bool oa::VulkanWindow::getPresenterInstanceExtensions(std::vector<char const *> *out_extensions) const {
	(void)out_extensions;
	return false;
}

bool oa::VulkanWindow::createPresenterVkSurface(VkInstance instance, VkSurfaceKHR *out_surface) const {
	(void)instance;
	(void)out_surface;
	return false;
}
