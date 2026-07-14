#include <Oa/Runtime/Window.h>

bool OaVkWindow::GetPresenterInstanceExtensions(std::vector<char const *> *out_extensions) const {
	(void)out_extensions;
	return false;
}

bool OaVkWindow::CreatePresenterVkSurface(VkInstance instance, VkSurfaceKHR *out_surface) const {
	(void)instance;
	(void)out_surface;
	return false;
}
