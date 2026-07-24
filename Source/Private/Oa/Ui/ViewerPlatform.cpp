// Engine first — VK_NO_PROTOTYPES must precede SDL's Vulkan declarations.
#include <Oa/Runtime/Engine.h>

#include "ViewerPlatform.h"

#include <Oa/Core/Log.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

std::mutex PlatformMutex;
OaU32 PlatformLeaseCount = 0;
bool PlatformOwnsVideo = false;
bool PlatformLoadedVulkan = false;

OaStatus PlatformError(const char* InOperation) {
	return OaStatus::Error(
		OaStatusCode::Unavailable,
		OaString(InOperation) + ": " + SDL_GetError());
}

void ConfigureVideoBackend() {
	if (not SDL_SetAppMetadata("OA", nullptr, "com.empyrealm.oa")) {
		OA_LOG_WARN(OaLogComponent::App,
			"SDL_SetAppMetadata failed: %s", SDL_GetError());
	}

	if (const char* backend = std::getenv("OA_UI_BACKEND");
		backend != nullptr and backend[0] != '\0') {
		if (not SDL_SetHint(SDL_HINT_VIDEO_DRIVER, backend)) {
			OA_LOG_WARN(OaLogComponent::App,
				"SDL video backend override '%s' was rejected", backend);
		}
	} else if (const char* session = std::getenv("XDG_SESSION_TYPE");
		session != nullptr and std::strcmp(session, "wayland") == 0) {
		(void)SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "wayland");
	}

#if !defined(_WIN32)
	if (std::getenv("LC_ALL") == nullptr and std::getenv("LANG") == nullptr) {
		::setenv("LC_ALL", "C.UTF-8", 0);
	}
#endif
}

void ReleasePlatformLocked() noexcept {
	if (PlatformLoadedVulkan) {
		SDL_Vulkan_UnloadLibrary();
		PlatformLoadedVulkan = false;
	}
	if (PlatformOwnsVideo) {
		SDL_QuitSubSystem(SDL_INIT_VIDEO);
		PlatformOwnsVideo = false;
	}
}

} // namespace

OaViewerPlatformLease::~OaViewerPlatformLease() {
	Release();
}

OaStatus OaViewerPlatformLease::Acquire(OaEngineConfig* InOutEngineConfig) {
	if (Acquired_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaViewerPlatformLease is already acquired");
	}

	std::lock_guard lock(PlatformMutex);
	if (PlatformLeaseCount == 0U) {
		ConfigureVideoBackend();
		if ((SDL_WasInit(SDL_INIT_VIDEO) & SDL_INIT_VIDEO) == 0U) {
			if (not SDL_InitSubSystem(SDL_INIT_VIDEO)) {
				return PlatformError("SDL_InitSubSystem(SDL_INIT_VIDEO) failed");
			}
			PlatformOwnsVideo = true;
		}
		if (not SDL_Vulkan_LoadLibrary(nullptr)) {
			const OaStatus status = PlatformError(
				"SDL_Vulkan_LoadLibrary failed");
			ReleasePlatformLocked();
			return status;
		}
		PlatformLoadedVulkan = true;
	}

	if (InOutEngineConfig != nullptr) {
		OaU32 extensionCount = 0U;
		const char* const* extensions =
			SDL_Vulkan_GetInstanceExtensions(&extensionCount);
		if (extensions == nullptr or extensionCount == 0U) {
			if (PlatformLeaseCount == 0U) ReleasePlatformLocked();
			return PlatformError(
				"SDL_Vulkan_GetInstanceExtensions failed");
		}
		InOutEngineConfig->PresentationMode = OaPresentationMode::Swapchain;
		for (OaU32 i = 0U; i < extensionCount; ++i) {
			InOutEngineConfig->InstanceExtraExtensions.PushBack(extensions[i]);
		}
	}

	++PlatformLeaseCount;
	Acquired_ = true;
	return OaStatus::Ok();
}

void OaViewerPlatformLease::Release() noexcept {
	if (not Acquired_) return;
	std::lock_guard lock(PlatformMutex);
	Acquired_ = false;
	if (PlatformLeaseCount == 0U) return;
	--PlatformLeaseCount;
	if (PlatformLeaseCount == 0U) ReleasePlatformLocked();
}
