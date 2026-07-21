#include "../../OaTest.h"

#include <Oa/Core/EnvFlag.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/OaVk.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

namespace {

struct SdlWindowScope {
	SDL_Window* Window = nullptr;
	~SdlWindowScope() {
		if (Window) SDL_DestroyWindow(Window);
		SDL_Vulkan_UnloadLibrary();
		SDL_Quit();
	}
};

} // namespace

TEST(PresenterWsi, AbandonedAttachedPresenterRetiresAtEngineClose) {
	if (!SDL_Init(SDL_INIT_VIDEO)) {
		GTEST_SKIP() << "SDL video unavailable: " << SDL_GetError();
	}
	SdlWindowScope sdl;
	if (!SDL_Vulkan_LoadLibrary(nullptr)) {
		GTEST_SKIP() << "SDL Vulkan unavailable: " << SDL_GetError();
	}
	sdl.Window = SDL_CreateWindow(
		"OA presenter retirement test", 96, 96,
		SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
	if (!sdl.Window) {
		GTEST_SKIP() << "SDL window unavailable: " << SDL_GetError();
	}

	OaEngineConfig config = OaTestEngineConfig(OaPrecision::FP32);
	config.PresentationMode = OaPresentationMode::Swapchain;
	config.RegisterAsGlobal = false;
	config.PreloadEmbeddedPipelines = false;
	config.EnablePipelineCache = false;
	if (OaEnvFlag::IsSet("OA_VK_VALIDATION")) config.EnableValidation = true;
	OaU32 extensionCount = 0;
	const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
	if (!extensions || extensionCount == 0) {
		GTEST_SKIP() << "SDL returned no Vulkan instance extensions";
	}
	for (OaU32 index = 0; index < extensionCount; ++index) {
		config.InstanceExtraExtensions.PushBack(extensions[index]);
	}

	auto engineResult = OaEngine::Create(config);
	ASSERT_TRUE(engineResult.IsOk()) << engineResult.GetStatus().ToString();
	auto engine = OaStdMove(*engineResult);
	VkSurfaceKHR surface = VK_NULL_HANDLE;
	ASSERT_TRUE(SDL_Vulkan_CreateSurface(
		sdl.Window, static_cast<VkInstance>(engine->Device.Instance), nullptr,
		&surface)) << SDL_GetError();

	OaBool transferredSurface = false;
	{
		OaPresenter presenter(*engine);
		if (!presenter.InitPresentation(surface, VkExtent2D{96, 96})) {
			vkDestroySurfaceKHR(
				static_cast<VkInstance>(engine->Device.Instance), surface, nullptr);
			ASSERT_TRUE(engine->Close().IsOk());
			FAIL() << "presenter initialization failed";
		}

		OaPresenter::AcquireResult acquired;
		OaBool acquiredFrame = false;
		for (OaU32 attempt = 0; attempt < 3 && !acquiredFrame; ++attempt) {
			if (!presenter.AcquireSwapchainImage(presenter.Swapchain(), acquired)) {
				break;
			}
			acquiredFrame = !acquired.Recreated;
		}
		if (!acquiredFrame) {
			ASSERT_TRUE(presenter.Close().IsOk());
			vkDestroySurfaceKHR(
				static_cast<VkInstance>(engine->Device.Instance), surface, nullptr);
			ASSERT_TRUE(engine->Close().IsOk());
			FAIL() << "failed to acquire a stable swapchain image";
		}

		const OaF32 clear[4] = {0.10F, 0.20F, 0.30F, 1.0F};
		OaPresenter::PresentArgs args;
		args.ClearRgba = clear;
		if (!presenter.PresentSwapchainImage(
			presenter.Swapchain(), acquired.ImageIndex, acquired.FrameSlot, args)) {
			ASSERT_TRUE(presenter.Close().IsOk());
			vkDestroySurfaceKHR(
				static_cast<VkInstance>(engine->Device.Instance), surface, nullptr);
			ASSERT_TRUE(engine->Close().IsOk());
			FAIL() << "failed to submit and present the WSI frame";
		}

		// No Close/Detach: destruction must transfer both pending WSI resources
		// and the still-attached surface to engine retirement without waiting.
		transferredSurface = true;
	}
	ASSERT_TRUE(transferredSurface);
	ASSERT_TRUE(engine->Close().IsOk());
}
