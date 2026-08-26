// Engine first — VK_NO_PROTOTYPES must precede SDL's vulkan declarations.
#include <oa/runtime/engine.h>

#include "presentationPlatform.h"

#include <oa/core/log.h>
#include <oa/core/std/cString.h>
#include <oa/core/std/sync.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <stdlib.h>

namespace {

oa::Mutex PlatformMutex;
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

	if (const char* backend = ::getenv("OA_UI_BACKEND");
		backend != nullptr and backend[0] != '\0') {
		if (not SDL_SetHint(SDL_HINT_VIDEO_DRIVER, backend)) {
			OaLogWarn(oa::LogComponent::Ui,
				"SDL video backend override '%s' was rejected", backend);
		}
	} else if (const char* session = ::getenv("XDG_SESSION_TYPE");
		session != nullptr and oa::strcmp(session, "wayland") == 0) {
		(void)SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "wayland");
	}

#if !defined(_WIN32)
	if (::getenv("LC_ALL") == nullptr and ::getenv("LANG") == nullptr) {
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

oa::PresentationPlatformLease::~PresentationPlatformLease() {
	release();
}

oa::Status oa::PresentationPlatformLease::acquire(
	oa::EngineConfig* inOutEngineConfig)
{
	if (acquired_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::PresentationPlatformLease is already acquired");
	}

	oa::ScopedLock lock(PlatformMutex);
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
		for (oa::U32 index = 0U; index < extensionCount; ++index) {
			inOutEngineConfig->instanceExtraExtensions.pushBack(extensions[index]);
		}
	}

	++PlatformLeaseCount;
	acquired_ = true;
	return oa::Status::ok();
}

void oa::PresentationPlatformLease::release() noexcept {
	if (not acquired_) return;
	oa::ScopedLock lock(PlatformMutex);
	acquired_ = false;
	if (PlatformLeaseCount == 0U) return;
	--PlatformLeaseCount;
	if (PlatformLeaseCount == 0U) releasePlatformLocked();
}
