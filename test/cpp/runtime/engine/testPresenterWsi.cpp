#include "../../oaTest.h"

#include <oa/core/envFlag.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/presenter.h>
#include <oa/runtime/window.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

namespace {

struct SdlWindowScope final : oa::VulkanWindow {
	SDL_Window* window = nullptr;
	~SdlWindowScope() override {
		if (window) SDL_DestroyWindow(window);
		SDL_Vulkan_UnloadLibrary();
		SDL_Quit();
	}

	bool pumpEvents(bool& outShouldQuit) override {
		outShouldQuit = false;
		return false;
	}
	[[nodiscard]] int drawableWidthPx() const override { return 96; }
	[[nodiscard]] int drawableHeightPx() const override { return 96; }
	[[nodiscard]] void* getNativeWindowHandle() const override { return window; }

protected:
	[[nodiscard]] bool createPresenterVkSurface(
		VkInstance inInstance,
		VkSurfaceKHR* outSurface) const override
	{
		return window != nullptr
			and SDL_Vulkan_CreateSurface(
				window, inInstance, nullptr, outSurface);
	}
};

} // namespace

TEST(PresenterWsi, AbandonedAttachedPresenterRetiresAtEngineClose) {
	if (!SDL_Init(SDL_INIT_VIDEO)) {
		GTEST_SKIP() << "SDL video unavailable: " << SDL_GetError();
	}
	SdlWindowScope sdl;
	if (!SDL_Vulkan_LoadLibrary(nullptr)) {
		GTEST_SKIP() << "SDL vulkan unavailable: " << SDL_GetError();
	}
	sdl.window = SDL_CreateWindow(
		"OA presenter retirement test", 96, 96,
		SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
	if (!sdl.window) {
		GTEST_SKIP() << "SDL window unavailable: " << SDL_GetError();
	}

	oa::EngineConfig config = testEngineConfig(oa::Precision::FP32);
	config.presentationMode = oa::PresentationMode::Swapchain;
	config.selectForThread = false;
	config.preloadEmbeddedPipelines = false;
	config.enablePipelineCache = false;
	if (oa::EnvFlag::isSet("OA_VK_VALIDATION")) config.enableValidation = true;
	oa::U32 extensionCount = 0;
	const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
	if (!extensions || extensionCount == 0) {
		GTEST_SKIP() << "SDL returned no vulkan instance extensions";
	}
	for (oa::U32 index = 0; index < extensionCount; ++index) {
		config.instanceExtraExtensions.pushBack(extensions[index]);
	}

	auto engineResult = oa::Engine::create(config);
	ASSERT_TRUE(engineResult.isOk()) << engineResult.getStatus().toString();
	auto engine = oa::move(*engineResult);
	oa::Bool transferredSurface = false;
	{
		oa::Presenter presenter(*engine);
		auto surfaceResult = presenter.createSurface(sdl);
		ASSERT_TRUE(surfaceResult.isOk())
			<< surfaceResult.getStatus().toString();
		void* surface = *surfaceResult;
		if (!presenter.initPresentation(surface, VkExtent2D{96, 96})) {
			ASSERT_TRUE(presenter.destroySurface(surface).isOk());
			ASSERT_TRUE(engine->close().isOk());
			FAIL() << "presenter initialization failed";
		}

		oa::Presenter::AcquireResult acquired;
		oa::Bool acquiredFrame = false;
		for (oa::U32 attempt = 0; attempt < 3 && !acquiredFrame; ++attempt) {
			if (!presenter.acquireSwapchainImage(presenter.swapchain(), acquired)) {
				break;
			}
			acquiredFrame = !acquired.recreated;
		}
		if (!acquiredFrame) {
			ASSERT_TRUE(presenter.close().isOk());
			ASSERT_TRUE(presenter.destroySurface(surface).isOk());
			ASSERT_TRUE(engine->close().isOk());
			FAIL() << "failed to acquire a stable swapchain image";
		}

		const oa::F32 clear[4] = {0.10F, 0.20F, 0.30F, 1.0F};
		oa::Presenter::PresentArgs args;
		args.clearRgba = clear;
		if (!presenter.presentSwapchainImage(
			presenter.swapchain(), acquired.imageIndex, acquired.frameSlot, args)) {
			ASSERT_TRUE(presenter.close().isOk());
			ASSERT_TRUE(presenter.destroySurface(surface).isOk());
			ASSERT_TRUE(engine->close().isOk());
			FAIL() << "failed to submit and present the WSI frame";
		}

		// No Close/detach: destruction must transfer both pending WSI resources
		// and the still-attached surface to engine retirement without waiting.
		transferredSurface = true;
	}
	ASSERT_TRUE(transferredSurface);
	ASSERT_TRUE(engine->close().isOk());
}
