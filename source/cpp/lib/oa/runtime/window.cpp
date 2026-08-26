#include <oa/runtime/window.h>

bool oa::VulkanWindow::getPresenterInstanceExtensions(
	oa::Vec<const char*>* outExtensions) const {
	(void)outExtensions;
	return false;
}

bool oa::VulkanWindow::createPresenterVkSurface(VkInstance instance, VkSurfaceKHR *out_surface) const {
	(void)instance;
	(void)out_surface;
	return false;
}
