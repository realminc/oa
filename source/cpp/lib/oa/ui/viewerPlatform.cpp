// Engine first — VK_NO_PROTOTYPES must precede SDL's vulkan declarations.
#include <oa/runtime/engine.h>

#include "viewerPlatform.h"

#include <oa/core/log.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

std::mutex PlatformMutex;
oa::U32 PlatformLeaseCount = 0;
bool PlatformOwnsVideo = false;
bool PlatformLoadedVulkan = false;

oa::Status platformError(const char* inOperation) {
	return oa::Status::error(
		oa::StatusCode::Unavailable,
		oa::String(inOperation) + ": " + SDL_GetError());
}

void configureVideoBackend() {
	if (not SDL_SetAppMetadata("OA", nullptr, "com.empyrealm.oa")) {
		OaLogWarn(oa::LogComponent::Ui,
			"SDL_SetAppMetadata failed: %s", SDL_GetError());
	}
	// oa::Ui renders the IME composition string at its scalar-safe caret. Keep the
	// candidate chooser native; OA does not own platform language policy.
	if (not SDL_SetHint(SDL_HINT_IME_IMPLEMENTED_UI, "composition")) {
		OaLogWarn(oa::LogComponent::Ui,
			"SDL IME composition hint was rejected: %s", SDL_GetError());
	}

	if (const char* backend = std::getenv("OA_UI_BACKEND");
		backend != nullptr and backend[0] != '\0') {
		if (not SDL_SetHint(SDL_HINT_VIDEO_DRIVER, backend)) {
			OaLogWarn(oa::LogComponent::Ui,
				"SDL video backend override '%s' was rejected", backend);
		}
	} else if (const char* session = std::getenv("XDG_SESSION_TYPE");
		session != nullptr and std::strcmp(session, "wayland") == 0) {
		(void)SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "wayland");
	}

#if !defined(_WIN32)
	if (std::getenv("LC_ALL") == nullptr and std::getenv("LANG") == nullptr) {
		::setenv("LC_ALL", "C.uTF-8", 0);
	}
#endif
}

void releasePlatformLocked() noexcept {
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

oa::ViewerPlatformLease::~ViewerPlatformLease() {
	release();
}

oa::Status oa::ViewerPlatformLease::acquire(oa::EngineConfig* inOutEngineConfig) {
	if (acquired_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::ViewerPlatformLease is already acquired");
	}

	std::lock_guard lock(PlatformMutex);
	if (PlatformLeaseCount == 0U) {
		configureVideoBackend();
		if ((SDL_WasInit(SDL_INIT_VIDEO) & SDL_INIT_VIDEO) == 0U) {
			if (not SDL_InitSubSystem(SDL_INIT_VIDEO)) {
				return platformError("SDL_InitSubSystem(SDL_INIT_VIDEO) failed");
			}
			PlatformOwnsVideo = true;
		}
		if (not SDL_Vulkan_LoadLibrary(nullptr)) {
			const oa::Status status = platformError(
				"SDL_Vulkan_LoadLibrary failed");
			releasePlatformLocked();
			return status;
		}
		PlatformLoadedVulkan = true;
	}

	if (inOutEngineConfig != nullptr) {
		oa::U32 extensionCount = 0U;
		const char* const* extensions =
			SDL_Vulkan_GetInstanceExtensions(&extensionCount);
		if (extensions == nullptr or extensionCount == 0U) {
			if (PlatformLeaseCount == 0U) releasePlatformLocked();
			return platformError(
				"SDL_Vulkan_GetInstanceExtensions failed");
		}
		inOutEngineConfig->presentationMode = oa::PresentationMode::Swapchain;
		for (oa::U32 i = 0U; i < extensionCount; ++i) {
			inOutEngineConfig->instanceExtraExtensions.pushBack(extensions[i]);
		}
	}

	++PlatformLeaseCount;
	acquired_ = true;
	return oa::Status::ok();
}

void oa::ViewerPlatformLease::release() noexcept {
	if (not acquired_) return;
	std::lock_guard lock(PlatformMutex);
	acquired_ = false;
	if (PlatformLeaseCount == 0U) return;
	--PlatformLeaseCount;
	if (PlatformLeaseCount == 0U) releasePlatformLocked();
}
