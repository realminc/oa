// OA Tutorial: completion-safe UI render target -> Viewer.
//
// This is the reusable editor/tooling shape: build one oa::Ui frame without a
// window, then choose a sink. oa::Viewer::show returns its final graphics
// completion to oa::Renderer before the frame target can be recycled.

#include <oa/runtime/engine.h>
#include <oa/ui/canvas.h>
#include <oa/render/renderer.h>
#include <oa/ui/renderConfig.h>
#include <oa/ui/viewer.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <array>
#include <cstdio>

namespace {

class SdlPlatformLease {
public:
	[[nodiscard]] oa::Status open(oa::EngineConfig& outConfig) {
		if (not SDL_InitSubSystem(SDL_INIT_VIDEO)) {
			return oa::Status::error(
				oa::StatusCode::Unavailable,
				oa::String("SDL video initialization failed: ") + SDL_GetError());
		}
		video_ = true;
		if (not SDL_Vulkan_LoadLibrary(nullptr)) {
			return oa::Status::error(
				oa::StatusCode::Unavailable,
				oa::String("SDL vulkan loading failed: ") + SDL_GetError());
		}
		vulkan_ = true;
		oa::U32 extensionCount = 0U;
		const char* const* extensions =
			SDL_Vulkan_GetInstanceExtensions(&extensionCount);
		if (extensions == nullptr or extensionCount == 0U) {
			return oa::Status::error(
				oa::StatusCode::Unavailable,
				oa::String("SDL vulkan extension query failed: ") + SDL_GetError());
		}
		outConfig.presentationMode = oa::PresentationMode::Swapchain;
		for (oa::U32 index = 0U; index < extensionCount; ++index) {
			outConfig.instanceExtraExtensions.pushBack(extensions[index]);
		}
		return oa::Status::ok();
	}

	~SdlPlatformLease() {
		if (vulkan_) SDL_Vulkan_UnloadLibrary();
		if (video_) SDL_QuitSubSystem(SDL_INIT_VIDEO);
	}

private:
	bool video_ = false;
	bool vulkan_ = false;
};

oa::Status runTutorial() {
	oa::EngineConfig engineConfig;
	engineConfig.appName = "TutorialRendererUi";
	engineConfig.selectForThread = false;
	engineConfig.preloadEmbeddedPipelines = false;
	SdlPlatformLease platform;
	OA_RETURN_IF_ERROR(platform.open(engineConfig));

	auto engineResult = oa::Engine::create(engineConfig);
	if (not engineResult.isOk()) return engineResult.getStatus();
	oa::UniquePtr<oa::Engine> engine = oa::move(*engineResult);

	oa::UiRenderConfig rendererConfig;
	rendererConfig.width_ = 960U;
	rendererConfig.height_ = 600U;
	rendererConfig.targetSlotCount_ = 1U;
	rendererConfig.style_ = oa::UiStyle::editorDark();
	auto rendererResult = oa::Renderer::create(*engine, rendererConfig);
	if (not rendererResult.isOk()) {
		const oa::Status status = rendererResult.getStatus();
		(void)engine->close();
		return status;
	}
	oa::UniquePtr<oa::Renderer> renderer = oa::move(*rendererResult);

	oa::Status status = renderer->beginFrame(16.0F);
	if (status.isOk()) {
		oa::Ui& ui = *renderer->ui();
		const oa::UiStyle& style = ui.currentStyle();
		ui.rect({0, 0, 960, 600}, style.background);

		std::array<oa::UiTabItem, 3U> tabs{{
			{.id = "scene", .label = "Scene.oa", .dirty = true},
			{.id = "material", .label = "Material"},
			{.id = "output", .label = "output"},
		}};
		oa::UiTabBarState tabState{.selected = 0, .firstVisible = 0};
		(void)ui.tabBar(
			"documents", {0, 0, 960, 34},
			oa::Span<const oa::UiTabItem>(tabs.data(), tabs.size()), tabState);

		oa::NodeCanvas canvas;
		status = canvas.setViewSize(570.0F, 520.0F);
		if (status.isOk()) {
			ui.nodeCanvasGrid(canvas, {190, 42, 570, 520});
			ui.line({304.0F, 183.0F}, {448.0F, 287.0F}, style.accent, 2.0F);
			ui.line({536.0F, 287.0F}, {650.0F, 399.0F}, style.accent, 2.0F);
			ui.rect({236, 142, 132, 82}, style.surface);
			ui.rectOutline({236, 142, 132, 82}, style.accent, 2U);
			ui.rect({432, 246, 120, 82}, style.surface);
			ui.rectOutline({432, 246, 120, 82}, style.border, 1U);
			ui.rect({594, 358, 116, 82}, style.surface);
			ui.rectOutline({594, 358, 116, 82}, style.border, 1U);
		}

		ui.beginPanel("outliner", {8, 42, 174, 520});
		ui.label("OUTLINER");
		(void)ui.treeRow(
			"world", {16, 82, 158, 26}, "World",
			{.hasChildren = true, .open = true, .selected = true});
		(void)ui.treeRow(
			"camera", {16, 110, 158, 26}, "camera", {.depth = 1});
		(void)ui.treeRow(
			"character", {16, 138, 158, 26}, "character", {.depth = 1});
		(void)ui.treeRow(
			"lights", {16, 166, 158, 26}, "Lights", {.depth = 1});
		ui.endPanel();

		bool enabled = true;
		oa::F32 exposure = 0.65F;
		oa::I32 samples = 64;
		oa::String name("HeroSurface");
		ui.beginPanel("inspector", {768, 42, 184, 520});
		ui.label("INSPECTOR");
		(void)ui.inputText("Name", name);
		(void)ui.checkbox("enabled", enabled);
		(void)ui.sliderF32("Exposure", &exposure, 0.0F, 2.0F, "%.2F");
		(void)ui.sliderI32("samples", &samples, 1, 256);
		ui.separator();
		ui.label("base color");
		ui.colorSwatch({0.16F, 0.48F, 0.92F, 1.0F}, {152.0F, 18.0F});
		ui.progressBar(0.72F, "Compiling shaders");
		(void)ui.button("Render frame");
		ui.endPanel();
	}

	oa::Result<oa::RenderFrame> frame = status.isOk()
		? renderer->submitFrame()
		: oa::Result<oa::RenderFrame>(status);
	if (not frame.isOk()) {
		if (status.isOk()) status = frame.getStatus();
	} else {
		oa::ViewerConfig viewerConfig;
		viewerConfig.title = "OA renderer · UI Composition";
		viewerConfig.width = 960U;
		viewerConfig.height = 600U;
		viewerConfig.showHelp = false;
		viewerConfig.showStats = false;
		viewerConfig.showTimeline = false;
		viewerConfig.presentFilter = oa::Filter::Nearest;
		status = oa::Viewer::show(*engine, *renderer, *frame, viewerConfig);
		// One target slot proves show returned and collected the exact frame: a
		// new frame must be immediately recordable without waiting or resizing.
		if (status.isOk()) status = renderer->beginFrame(0.0F);
		if (status.isOk()) status = renderer->cancelFrame();
	}

	const oa::Status rendererStatus = renderer->close();
	const oa::Status engineStatus = engine->close();
	if (not status.isOk()) return status;
	if (not rendererStatus.isOk()) return rendererStatus;
	return engineStatus;
}

} // namespace

int main() {
	const oa::Status status = runTutorial();
	if (not status.isOk()) {
		std::fprintf(stderr, "TutorialRendererUi failed: %s\n",
			status.toString().cStr());
		return 1;
	}
	return 0;
}
