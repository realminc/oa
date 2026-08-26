// oa::Viewer::run — SDL3-backed windowed application loop.
// Creates oa::Presenter + SDL3 window + vulkan surface, then drives
// oa::Ui per-frame for the single windowed viewer lifecycle.

// Engine first: Device.h → OaVk.h → VK_NO_PROTOTYPES before any vulkan.h pull-in.
#include <oa/runtime/engine.h>
#include <oa/runtime/presenter.h>
#include <oa/runtime/window.h>
#include <oa/core/log.h>
#include <oa/runtime/stream.h>
#include <oa/ui/viewer.h>
#include <oa/core/envFlag.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/chrono.h>
#include <oa/core/std/cString.h>
#include <oa/core/std/scalarMath.h>
#include <oa/ui/platformInput.h>
#include <oa/core/validation.h>

#include "../runtime/presentationPlatform.h"
#include "windowDecoration.h"

#include <stdlib.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

namespace {

constexpr oa::WindowDecorationMetrics kWindowDecorationMetrics;
constexpr oa::U32 kMaxWindowTitleGlyphs = 1024U;

class ViewerSdlWindow final : public oa::VulkanWindow {
public:
	explicit ViewerSdlWindow(SDL_Window* inWindow) noexcept
		: window_(inWindow) {}

	bool pumpEvents(bool& outShouldQuit) override {
		outShouldQuit = false;
		return false;
	}
	[[nodiscard]] int drawableWidthPx() const override {
		int width = 0;
		int height = 0;
		if (window_) (void)SDL_GetWindowSizeInPixels(window_, &width, &height);
		return width;
	}
	[[nodiscard]] int drawableHeightPx() const override {
		int width = 0;
		int height = 0;
		if (window_) (void)SDL_GetWindowSizeInPixels(window_, &width, &height);
		return height;
	}
	[[nodiscard]] void* getNativeWindowHandle() const override {
		return window_;
	}

protected:
	[[nodiscard]] bool createPresenterVkSurface(
		VkInstance inInstance,
		VkSurfaceKHR* outSurface) const override
	{
		return window_ != nullptr
			and SDL_Vulkan_CreateSurface(
				window_, inInstance, nullptr, outSurface);
	}

private:
	SDL_Window* window_ = nullptr;
};

SDL_HitTestResult toSdlHitTest(oa::WindowDecorationHit inHit) {
	switch (inHit) {
		case oa::WindowDecorationHit::Draggable:
			return SDL_HITTEST_DRAGGABLE;
		case oa::WindowDecorationHit::ResizeTopLeft:
			return SDL_HITTEST_RESIZE_TOPLEFT;
		case oa::WindowDecorationHit::ResizeTop:
			return SDL_HITTEST_RESIZE_TOP;
		case oa::WindowDecorationHit::ResizeTopRight:
			return SDL_HITTEST_RESIZE_TOPRIGHT;
		case oa::WindowDecorationHit::ResizeRight:
			return SDL_HITTEST_RESIZE_RIGHT;
		case oa::WindowDecorationHit::ResizeBottomRight:
			return SDL_HITTEST_RESIZE_BOTTOMRIGHT;
		case oa::WindowDecorationHit::ResizeBottom:
			return SDL_HITTEST_RESIZE_BOTTOM;
		case oa::WindowDecorationHit::ResizeBottomLeft:
			return SDL_HITTEST_RESIZE_BOTTOMLEFT;
		case oa::WindowDecorationHit::ResizeLeft:
			return SDL_HITTEST_RESIZE_LEFT;
		case oa::WindowDecorationHit::Normal:
			break;
	}
	return SDL_HITTEST_NORMAL;
}

SDL_HitTestResult SDLCALL viewerWindowHitTest(
	SDL_Window* inWindow,
	const SDL_Point* inArea,
	void*) {
	if (inWindow == nullptr or inArea == nullptr) return SDL_HITTEST_NORMAL;
	int width = 0;
	int height = 0;
	if (not SDL_GetWindowSize(inWindow, &width, &height)) {
		return SDL_HITTEST_NORMAL;
	}
	const SDL_WindowFlags flags = SDL_GetWindowFlags(inWindow);
	return toSdlHitTest(oa::windowDecorationHitTest(
		inArea->x,
		inArea->y,
		width,
		height,
		(flags & SDL_WINDOW_RESIZABLE) != 0,
		(flags & SDL_WINDOW_MAXIMIZED) != 0,
		kWindowDecorationMetrics));
}

void appendWindowTitleGlyphs(
	const oa::TextAtlas& inAtlas,
	oa::StringView inTitle,
	oa::F32 inScaleX,
	oa::F32 inScaleY,
	oa::Vec<oa::GlyphInstance>& inOutGlyphs) {
	const oa::F32 fontSize = 13.0F * inScaleY;
	const oa::F32 baseline = 23.0F * inScaleY;
	oa::TextLayout layout;
	oa::TextLayoutConfig config{
		.font = oa::FontId::Sans,
		.size = fontSize,
	};
	oa::Vec<oa::PositionedGlyph> positioned;
	layout.shape(
		inAtlas, inTitle, {20.0F * inScaleX, baseline}, config,
		oa::Color{0.80F, 0.80F, 0.80F, 1.0F}.toU32(), positioned);
	for (const auto& item : positioned) {
		if (inOutGlyphs.size() >= kMaxWindowTitleGlyphs) break;
		const oa::GlyphInfo* glyph = inAtlas.findGlyph(
			item.font, item.codepoint, fontSize);
		if (glyph == nullptr) continue;
		const oa::F32 glyphScale = fontSize / glyph->rasterSize;
		inOutGlyphs.pushBack({
			.anchorX = 0.0F,
			.anchorY = 0.0F,
			.offsetX = item.x + glyph->bearingX * glyphScale,
			.offsetY = item.y - glyph->bearingY * glyphScale,
			.width = glyph->atlasW * glyphScale,
			.height = glyph->atlasH * glyphScale,
			.atlasX = static_cast<oa::U32>(glyph->atlasX),
			.atlasY = static_cast<oa::U32>(glyph->atlasY),
			.atlasW = static_cast<oa::U32>(glyph->atlasW),
			.atlasH = static_cast<oa::U32>(glyph->atlasH),
			.color = oa::Color{0.80F, 0.80F, 0.80F, 1.0F}.toU32(),
		});
	}
}

} // namespace


static oa::UiEvent convertSdlEvent(const SDL_Event& inSdl, SDL_Window* inWindow) {
	oa::UiEvent e;
	oa::F32 pixelScaleX = 1.0F;
	oa::F32 pixelScaleY = 1.0F;
	if (inWindow != nullptr) {
		int logicalWidth = 0;
		int logicalHeight = 0;
		int pixelWidth = 0;
		int pixelHeight = 0;
		if (SDL_GetWindowSize(inWindow, &logicalWidth, &logicalHeight)
			and SDL_GetWindowSizeInPixels(inWindow, &pixelWidth, &pixelHeight)
			and logicalWidth > 0 and logicalHeight > 0) {
			pixelScaleX = static_cast<oa::F32>(pixelWidth)
				/ static_cast<oa::F32>(logicalWidth);
			pixelScaleY = static_cast<oa::F32>(pixelHeight)
				/ static_cast<oa::F32>(logicalHeight);
		}
	}
	const auto pixelX = [pixelScaleX](oa::F32 inX) { return inX * pixelScaleX; };
	const auto pixelY = [pixelScaleY](oa::F32 inY) { return inY * pixelScaleY; };
	const SDL_Keymod mods = SDL_GetModState();
	if ((mods & SDL_KMOD_SHIFT) != 0) { e.modifiers |= oa::UiModifierShift; }
	if ((mods & SDL_KMOD_CTRL)  != 0) { e.modifiers |= oa::UiModifierCtrl; }
	if ((mods & SDL_KMOD_ALT)   != 0) { e.modifiers |= oa::UiModifierAlt; }
	if ((mods & SDL_KMOD_GUI)   != 0) { e.modifiers |= oa::UiModifierSuper; }
	switch (inSdl.type) {
		case SDL_EVENT_MOUSE_MOTION:
			e.type   = oa::UiEventType::MouseMove;
			e.mouseX = pixelX(inSdl.motion.x);
			e.mouseY = pixelY(inSdl.motion.y);
			e.mouseDX = pixelX(inSdl.motion.xrel);
			e.mouseDY = pixelY(inSdl.motion.yrel);
			break;
		case SDL_EVENT_MOUSE_BUTTON_DOWN:
			e.type   = oa::UiEventType::MouseDown;
			e.mouseX = pixelX(inSdl.button.x); e.mouseY = pixelY(inSdl.button.y);
			e.button = static_cast<oa::I32>(inSdl.button.button);
			e.clickCount = static_cast<oa::I32>(inSdl.button.clicks);
			break;
		case SDL_EVENT_MOUSE_BUTTON_UP:
			e.type   = oa::UiEventType::MouseUp;
			e.mouseX = pixelX(inSdl.button.x); e.mouseY = pixelY(inSdl.button.y);
			e.button = static_cast<oa::I32>(inSdl.button.button);
			break;
		case SDL_EVENT_MOUSE_WHEEL:
			e.type    = oa::UiEventType::MouseScroll;
			e.scrollX = inSdl.wheel.x; e.scrollY = inSdl.wheel.y;
			e.mouseX  = pixelX(inSdl.wheel.mouse_x); e.mouseY = pixelY(inSdl.wheel.mouse_y);
			e.integerScrollX = inSdl.wheel.integer_x;
			e.integerScrollY = inSdl.wheel.integer_y;
			e.timestampNs    = inSdl.wheel.timestamp;
			e.scrollGesture  = oa::input::classifyScroll(e);
			break;
		case SDL_EVENT_PINCH_BEGIN:
			e.type = oa::UiEventType::Pinch;
			e.pinchPhase = oa::UiPinchPhase::Begin;
			e.gestureScale = 1.0F;
			break;
		case SDL_EVENT_PINCH_UPDATE:
			e.type = oa::UiEventType::Pinch;
			e.pinchPhase = oa::UiPinchPhase::Update;
			e.gestureScale = inSdl.pinch.scale;
			break;
		case SDL_EVENT_PINCH_END:
			e.type = oa::UiEventType::Pinch;
			e.pinchPhase = oa::UiPinchPhase::End;
			e.gestureScale = 1.0F;
			break;
		case SDL_EVENT_KEY_DOWN:
			e.type = oa::UiEventType::KeyDown;
			e.key  = static_cast<oa::UiKey>(inSdl.key.scancode);
			e.keyRepeat = inSdl.key.repeat;
			break;
		case SDL_EVENT_KEY_UP:
			e.type = oa::UiEventType::KeyUp;
			e.key  = static_cast<oa::UiKey>(inSdl.key.scancode);
			e.keyRepeat = inSdl.key.repeat;
			break;
		case SDL_EVENT_TEXT_INPUT:
			e.type = oa::UiEventType::KeyChar;
			e.text = oa::String(inSdl.text.text);
			break;
		case SDL_EVENT_TEXT_EDITING:
			e.type = oa::UiEventType::TextEditing;
			e.text = oa::String(inSdl.edit.text);
			e.textSelectionStart = inSdl.edit.start;
			e.textSelectionLength = inSdl.edit.length;
			break;
		case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
		case SDL_EVENT_WINDOW_RESIZED: {
			e.type = oa::UiEventType::WindowResize;
			int wPx = inSdl.window.data1;
			int hPx = inSdl.window.data2;
			if (SDL_Window* win = SDL_GetWindowFromID(inSdl.window.windowID)) {
				(void)SDL_GetWindowSizeInPixels(win, &wPx, &hPx);
			}
			e.windowW = wPx;
			e.windowH = hPx;
			break;
		}
		default:
			break;
	}
	return e;
}

oa::Status oa::Viewer::rebuildWindowTitleGlyphs() {
	if (not windowDecorationActive_ or config_.title.empty()) {
		return oa::Status::ok();
	}
	if (not windowTitleGlyphs_.isValid()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Viewer window title glyph buffer is not initialized");
	}
	oa::Vec<oa::GlyphInstance> glyphs;
	glyphs.reserve(oa::min<oa::Usize>(
		config_.title.size(),
		static_cast<oa::Usize>(kMaxWindowTitleGlyphs)));
	appendWindowTitleGlyphs(
		textAtlas_,
		config_.title,
		windowPixelScaleX_,
		windowPixelScaleY_,
		glyphs);
	return windowTitleGlyphs_.upload(
		oa::Span<const oa::GlyphInstance>(glyphs.data(), glyphs.size()));
}

void oa::Viewer::refreshWindowDecorationScale() {
	if (window_ == nullptr) return;
	SDL_Window* window = static_cast<SDL_Window*>(window_);
	int logicalWidth = 0;
	int logicalHeight = 0;
	int pixelWidth = 0;
	int pixelHeight = 0;
	if (not SDL_GetWindowSize(window, &logicalWidth, &logicalHeight)
		or not SDL_GetWindowSizeInPixels(window, &pixelWidth, &pixelHeight)
		or logicalWidth <= 0 or logicalHeight <= 0
		or pixelWidth <= 0 or pixelHeight <= 0) {
		return;
	}
	const oa::F32 nextScaleX =
		static_cast<oa::F32>(pixelWidth) / static_cast<oa::F32>(logicalWidth);
	const oa::F32 nextScaleY =
		static_cast<oa::F32>(pixelHeight) / static_cast<oa::F32>(logicalHeight);
	const bool changed =
		oa::abs(nextScaleX - windowPixelScaleX_) > 0.001F
		or oa::abs(nextScaleY - windowPixelScaleY_) > 0.001F;
	windowPixelScaleX_ = nextScaleX;
	windowPixelScaleY_ = nextScaleY;
	if (changed and windowDecorationActive_ and windowTitleGlyphs_.isValid()) {
		if (const oa::Status status = rebuildWindowTitleGlyphs();
			not status.isOk()) {
			OaLogWarn(oa::LogComponent::Ui,
				"oa::Viewer title scale refresh failed: %s",
				status.toString().cStr());
		}
	}
}

oa::Status oa::Viewer::initWindowDecoration() {
	refreshWindowDecorationScale();
	if (not windowDecorationActive_) return oa::Status::ok();
	if (engine_ == nullptr or window_ == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Viewer window decoration requires a window and engine");
	}
	if (config_.title.empty()) return oa::Status::ok();
	const oa::Usize requestedCapacity = oa::min<oa::Usize>(
		oa::max<oa::Usize>(config_.title.size(), 1U),
		static_cast<oa::Usize>(kMaxWindowTitleGlyphs));
	auto buffer = oa::GlyphBuffer::createHostUpload(
		*engine_, static_cast<oa::U32>(requestedCapacity));
	if (not buffer.isOk()) return buffer.getStatus();
	windowTitleGlyphs_ = oa::move(*buffer);
	return rebuildWindowTitleGlyphs();
}

void oa::Viewer::destroyWindowDecoration() {
	windowTitleGlyphs_ = {};
	windowHoveredControl_ = 0;
	windowPressedControl_ = 0;
	windowDecorationActive_ = false;
	windowMaximized_ = false;
}

bool oa::Viewer::routeWindowDecorationEvent(void* inEvent) {
	if (not windowDecorationActive_ or window_ == nullptr or inEvent == nullptr) {
		return false;
	}
	auto& event = *static_cast<SDL_Event*>(inEvent);
	SDL_Window* window = static_cast<SDL_Window*>(window_);
	const SDL_WindowFlags flags = SDL_GetWindowFlags(window);
	windowMaximized_ = (flags & SDL_WINDOW_MAXIMIZED) != 0;

	if (event.type == SDL_EVENT_WINDOW_MOUSE_LEAVE
		or event.type == SDL_EVENT_WINDOW_FOCUS_LOST) {
		const bool consumed = windowPressedControl_ != 0;
		windowHoveredControl_ = 0;
		windowPressedControl_ = 0;
		return consumed;
	}

	oa::F32 mouseX = 0.0F;
	oa::F32 mouseY = 0.0F;
	if (event.type == SDL_EVENT_MOUSE_MOTION) {
		mouseX = event.motion.x;
		mouseY = event.motion.y;
	} else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN
		or event.type == SDL_EVENT_MOUSE_BUTTON_UP) {
		mouseX = event.button.x;
		mouseY = event.button.y;
	} else {
		return false;
	}

	int width = 0;
	int height = 0;
	if (not SDL_GetWindowSize(window, &width, &height)) return false;
	const auto control = oa::windowDecorationControlAt(
		static_cast<oa::I32>(mouseX),
		static_cast<oa::I32>(mouseY),
		width,
		kWindowDecorationMetrics);
	windowHoveredControl_ = static_cast<oa::I32>(control);

	if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN
		and event.button.button == SDL_BUTTON_LEFT
		and control != oa::WindowDecorationControl::None) {
		windowPressedControl_ = static_cast<oa::I32>(control);
		return true;
	}

	if (event.type == SDL_EVENT_MOUSE_BUTTON_UP
		and event.button.button == SDL_BUTTON_LEFT
		and windowPressedControl_ != 0) {
		const auto pressed = static_cast<oa::WindowDecorationControl>(
			windowPressedControl_);
		windowPressedControl_ = 0;
		if (pressed != control) return true;

		bool succeeded = true;
		switch (control) {
			case oa::WindowDecorationControl::Minimize:
				succeeded = SDL_MinimizeWindow(window);
				break;
			case oa::WindowDecorationControl::Maximize:
				succeeded = windowMaximized_
					? SDL_RestoreWindow(window)
					: SDL_MaximizeWindow(window);
				break;
			case oa::WindowDecorationControl::Close:
				running_ = false;
				break;
			case oa::WindowDecorationControl::None:
				break;
		}
		if (not succeeded) {
			OaLogWarn(oa::LogComponent::Ui,
				"oa::Viewer window control failed: %s", SDL_GetError());
		}
		return true;
	}
	return control != oa::WindowDecorationControl::None
		or windowPressedControl_ != 0;
}

oa::U32 oa::Viewer::windowDecorationHeight() const noexcept {
	if (not windowDecorationActive_) return 0U;
	return static_cast<oa::U32>(oa::max(
		1.0F,
		oa::ceil(
			static_cast<oa::F32>(kWindowDecorationMetrics.titleHeight)
			* windowPixelScaleY_)));
}

void oa::Viewer::renderWindowDecoration(oa::Ui& inUi) {
	if (not windowDecorationActive_) return;
	const oa::I32 width = static_cast<oa::I32>(this->width());
	const oa::I32 height = static_cast<oa::I32>(windowDecorationHeight());
	if (width <= 0 or height <= 0) return;

	const oa::I32 controlWidth = oa::max(
		1,
		static_cast<oa::I32>(oa::ceil(
			static_cast<oa::F32>(kWindowDecorationMetrics.controlWidth)
			* windowPixelScaleX_)));
	const oa::I32 controlsBegin = width - controlWidth * 3;
	const oa::UiStyle& style = config_.style;
	const oa::Color titleBackground = style.background;
	const oa::Color controlHover = style.surfaceHover;
	const oa::Color controlPressed = style.accent;
	const oa::Color closeHover = style.error;
	const oa::Color icon = style.text;

	inUi.rect({0, 0, width, height}, titleBackground);
	inUi.rect({0, height - 1, width, 1}, style.border);
	inUi.rect({
		oa::max(8, static_cast<oa::I32>(12.0F * windowPixelScaleX_)),
		oa::max(7, static_cast<oa::I32>(9.0F * windowPixelScaleY_)),
		oa::max(2, static_cast<oa::I32>(3.0F * windowPixelScaleX_)),
		oa::max(12, static_cast<oa::I32>(18.0F * windowPixelScaleY_))},
		style.accent);

	for (oa::I32 index = 0; index < 3; ++index) {
		const oa::I32 control = index + 1;
		const oa::PixelRect rect{
			controlsBegin + index * controlWidth, 0, controlWidth, height};
		if (windowPressedControl_ == control) {
			inUi.rect(rect, controlPressed);
		} else if (windowHoveredControl_ == control) {
			inUi.rect(
				rect,
				control == static_cast<oa::I32>(
					oa::WindowDecorationControl::Close)
					? closeHover : controlHover);
		}

		const oa::F32 centerX =
			static_cast<oa::F32>(rect.x) + static_cast<oa::F32>(rect.w) * 0.5F;
		const oa::F32 centerY =
			static_cast<oa::F32>(rect.h) * 0.5F;
		const oa::F32 half = oa::max(4.0F, 5.0F * windowPixelScaleX_);
		if (control == static_cast<oa::I32>(
			oa::WindowDecorationControl::Minimize)) {
			inUi.line(
				{centerX - half, centerY + 3.0F * windowPixelScaleY_},
				{centerX + half, centerY + 3.0F * windowPixelScaleY_},
				icon, 1.0F);
		} else if (control == static_cast<oa::I32>(
			oa::WindowDecorationControl::Maximize)) {
			const oa::I32 box = oa::max(
				8, static_cast<oa::I32>(10.0F * windowPixelScaleX_));
			const oa::I32 boxX = static_cast<oa::I32>(centerX) - box / 2;
			const oa::I32 boxY = static_cast<oa::I32>(centerY) - box / 2;
			if (windowMaximized_) {
				inUi.rectOutline(
					{boxX - 2, boxY + 2, box, box}, icon, 1);
				inUi.rectOutline(
					{boxX + 2, boxY - 2, box, box}, icon, 1);
			} else {
				inUi.rectOutline({boxX, boxY, box, box}, icon, 1);
			}
		} else {
			inUi.line(
				{centerX - half, centerY - half},
				{centerX + half, centerY + half},
				icon, 1.25F);
			inUi.line(
				{centerX + half, centerY - half},
				{centerX - half, centerY + half},
				icon, 1.25F);
		}
	}

	if (windowTitleGlyphs_.isValid()) {
		const oa::I32 titleClipWidth = oa::max(1, controlsBegin - 8);
		const oa::PixelRect screen{
			0, 0, width, static_cast<oa::I32>(this->height())};
		inUi.glyphs(
			windowTitleGlyphs_,
			textAtlas_,
			screen,
			{0, 0, titleClipWidth, height});
	}
}


void oa::Viewer::resizeWindow(oa::U32 inWidth, oa::U32 inHeight) noexcept {
	if (not window_ or inWidth == 0 or inHeight == 0) return;
	SDL_Window* win = static_cast<SDL_Window*>(window_);
	refreshWindowDecorationScale();
	const int logicalWidth = oa::windowLogicalSizeForPixels(
		inWidth, windowPixelScaleX_);
	const int logicalHeight = oa::windowLogicalSizeForPixels(
		inHeight, windowPixelScaleY_);
	SDL_SetWindowSize(win, logicalWidth, logicalHeight);
	SDL_SetWindowPosition(win, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
	int wPx = 0;
	int hPx = 0;
	SDL_GetWindowSizeInPixels(win, &wPx, &hPx);
	(void)resize(static_cast<oa::U32>(wPx), static_cast<oa::U32>(hPx));
}

void oa::Viewer::capturePointer(bool inEnabled) noexcept {
	if (not window_) { return; }
	// capture only — NOT mouse-grab (grab confines cursor inside the window).
	(void)SDL_CaptureMouse(inEnabled);
}

void oa::Viewer::captureRelativeMouse(bool inEnabled) noexcept {
	if (not window_) { return; }
	SDL_Window* win = static_cast<SDL_Window*>(window_);
	(void)SDL_SetWindowRelativeMouseMode(win, inEnabled);
	(void)SDL_CaptureMouse(inEnabled);
}

oa::Status oa::Viewer::run() {
	return runApplication(nullptr);
}

oa::Status oa::Viewer::run(oa::Engine& inEngine) {
	return runApplication(&inEngine);
}

oa::Status oa::Viewer::runApplication(oa::Engine* inBorrowedEngine) {
	if (running_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Viewer::run cannot enter an already-running viewer");
	}

	// Deterministic, graceful smoke-test boundary for windowed tutorials. Zero
	// or an invalid value preserves the normal interactive run-until-closed loop.
	oa::U64 maxFrames = 0;
	if (const char* value = ::getenv("OA_UI_MAX_FRAMES"); value and *value) {
		char* end = nullptr;
		const unsigned long long parsed = ::strtoull(value, &end, 10);
		if (end != value and *end == '\0') {
			maxFrames = static_cast<oa::U64>(parsed);
		}
	}
	oa::U64 renderedFrames = 0;

	oa::EngineConfig engineCfg;
	oa::PresentationPlatformLease platform;
	const oa::Status platformStatus = platform.acquire(
		inBorrowedEngine == nullptr ? &engineCfg : nullptr);
	if (not platformStatus.isOk()) return platformStatus;

	// Keep windowed applications inside the same validation contract as tests
	// and headless workloads. tools/diagnostics/runValidation.py sets this
	// process-local flag and verifies that OA actually enabled the requested
	// validation profile.
	if (inBorrowedEngine == nullptr
		and oa::EnvFlag::isSet("OA_VK_VALIDATION")) {
		engineCfg.enableValidation = true;
	}

	// Respect OA_DEVICE env var (same semantics as gtest harness).
	if (inBorrowedEngine == nullptr) {
		if (const char* dev = ::getenv("OA_DEVICE"); dev and *dev) {
			if (oa::strcmp(dev, "integrated") == 0
				or oa::strcmp(dev, "igpu") == 0) {
				engineCfg.devicePref = oa::DevicePreference::Integrated;
			} else if (oa::strcmp(dev, "discrete") == 0
				or oa::strcmp(dev, "dgpu") == 0) {
				engineCfg.devicePref = oa::DevicePreference::Discrete;
			} else if (oa::strcmp(dev, "cpu") == 0) {
				engineCfg.devicePref = oa::DevicePreference::Cpu;
			} else {
				char* end = nullptr;
				unsigned long idx = ::strtoul(dev, &end, 10);
				if (end != dev and *end == '\0' and idx <= 0xFFFFu) {
					engineCfg.devicePref = oa::DevicePreference::ByIndex;
					engineCfg.deviceIndex = static_cast<oa::U32>(idx);
				}
			}
		}
	}

	// phase A — one engine with a graphics-capable queue, no surface yet.
	oa::UniquePtr<oa::Engine> ownedEngine;
	oa::Engine* engine = inBorrowedEngine;
	if (engine == nullptr) {
		auto engineResult = oa::Engine::create(engineCfg);
		if (not engineResult.isOk()) {
			OaLogError(oa::LogComponent::Ui, "Engine create failed: %s",
				engineResult.getStatus().getMessage().cStr());
			return engineResult.getStatus();
		}
		ownedEngine = oa::move(*engineResult);
		engine = ownedEngine.get();
	}
	if (not engine->isReady() or not engine->hasGraphics()) {
		if (ownedEngine != nullptr) (void)engine->close();
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Viewer::run requires a ready presentation-capable oa::Engine");
	}
	oa::Engine& rt = *engine;
	const auto closeOwnedEngine = [&]() -> oa::Status {
		return ownedEngine != nullptr ? rt.close() : oa::Status::ok();
	};

	oa::input::initialize();
	oa::Presenter presenter(rt);

	const SDL_WindowFlags baseFlags =
		SDL_WINDOW_VULKAN
		| SDL_WINDOW_RESIZABLE
		| SDL_WINDOW_HIGH_PIXEL_DENSITY;
	SDL_WindowFlags flags = baseFlags;
	if (config_.customWindowDecoration) flags |= SDL_WINDOW_BORDERLESS;
	SDL_Window* win = SDL_CreateWindow(
		config_.title.cStr(),
		static_cast<int>(config_.width),
		static_cast<int>(config_.height),
		flags);
	if (win == nullptr) {
		OaLogError(oa::LogComponent::Ui, "SDL_CreateWindow failed: %s", SDL_GetError());
		oa::input::shutdown();
		(void)closeOwnedEngine();
		return oa::Status::error("SDL_CreateWindow failed");
	}
	if (not SDL_SetWindowMinimumSize(win, 320, 240)) {
		OaLogWarn(oa::LogComponent::Ui,
			"SDL_SetWindowMinimumSize failed: %s", SDL_GetError());
	}
	if (not config_.customWindowDecoration) {
		windowDecorationActive_ = false;
	} else if (SDL_SetWindowHitTest(win, viewerWindowHitTest, nullptr)) {
		windowDecorationActive_ = true;
		OaLogInfo(oa::LogComponent::Ui,
			"oa::Viewer custom window decoration active (%s)",
			SDL_GetCurrentVideoDriver() != nullptr
				? SDL_GetCurrentVideoDriver() : "unknown video driver");
	} else {
		OaLogWarn(oa::LogComponent::Ui,
			"SDL window hit-testing is unavailable; using system decorations: %s",
			SDL_GetError());
		SDL_DestroyWindow(win);
		win = SDL_CreateWindow(
			config_.title.cStr(),
			static_cast<int>(config_.width),
			static_cast<int>(config_.height),
			baseFlags);
		if (win == nullptr) {
			OaLogError(oa::LogComponent::Ui,
				"SDL_CreateWindow fallback failed: %s", SDL_GetError());
			oa::input::shutdown();
			(void)closeOwnedEngine();
			return oa::Status::error("SDL_CreateWindow fallback failed");
		}
		windowDecorationActive_ = false;
		if (not SDL_SetWindowMinimumSize(win, 320, 240)) {
			OaLogWarn(oa::LogComponent::Ui,
				"SDL_SetWindowMinimumSize fallback failed: %s",
				SDL_GetError());
		}
	}

	// phase B — opaque surface through the SDL window backend. The raw instance
	// remains inside oa::Presenter.
	ViewerSdlWindow surfaceWindow(win);
	auto surfaceResult = presenter.createSurface(surfaceWindow);
	if (not surfaceResult.isOk()) {
		OaLogError(oa::LogComponent::Ui, "vulkan surface creation failed: %s",
			surfaceResult.getStatus().toString().cStr());
		SDL_DestroyWindow(win);
		oa::input::shutdown();
		(void)closeOwnedEngine();
		return surfaceResult.getStatus();
	}
	void* surface = *surfaceResult;

	// Use actual pixel size for HiDPI displays.
	int wPx = 0;
	int hPx = 0;
	SDL_GetWindowSizeInPixels(win, &wPx, &hPx);
	config_.width  = static_cast<oa::U32>(wPx);
	config_.height = static_cast<oa::U32>(hPx);

	window_ = static_cast<void*>(win);
	if (auto s = openSource(rt); not s.isOk()) {
		OaLogError(oa::LogComponent::Ui, "Device-ready initialization failed: %s",
			s.getMessage().cStr());
		const oa::Status closeStatus = closeSource();
		if (not closeStatus.isOk()) {
			OaLogError(oa::LogComponent::Ui,
				"source cleanup after open failure failed: %s",
				closeStatus.toString().cStr());
		}
		(void)presenter.destroySurface(surface);
		SDL_DestroyWindow(win);
		window_ = nullptr;
		oa::input::shutdown();
		(void)closeOwnedEngine();
		return s;
	}

	// phase C — the presenter owns WSI; the viewer owns its UI render target.
	if (auto s = initPresentation(presenter, surface); not s.isOk()) {
		OaLogError(oa::LogComponent::Ui, "oa::Viewer initialization failed: %s", s.getMessage().cStr());
		(void)closeSource();
		(void)presenter.destroySurface(surface);
		SDL_DestroyWindow(win);
		window_ = nullptr;
		oa::input::shutdown();
		(void)closeOwnedEngine();
		return s;
	}

	running_ = true;
	if (auto s = initView(); not s.isOk()) {
		running_ = false;
		(void)closeSource();
		const oa::Status presentationStatus = destroyPresentation();
		if (not presentationStatus.isOk()) {
			OaLogError(oa::LogComponent::Ui,
				"Presentation cleanup after view initialization failed: %s",
				presentationStatus.toString().cStr());
		}
		(void)presenter.destroySurface(surface);
		SDL_DestroyWindow(win);
		window_ = nullptr;
		oa::input::shutdown();
		(void)closeOwnedEngine();
		return s;
	}
	oa::Vec<oa::UiEvent> events;
	using Clock = oa::SteadyClock;
	auto tPrev = Clock::now();
	oa::Status runStatus = oa::Status::ok();
	bool nativeTextInputActive = false;

	while (running_) {
		events.clear();
		SDL_Event sdlEvent;
		while (SDL_PollEvent(&sdlEvent)) {
			if (routeWindowDecorationEvent(&sdlEvent)) continue;
			oa::input::processEvent(&sdlEvent);
			if (sdlEvent.type == SDL_EVENT_QUIT
			 or sdlEvent.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
				running_ = false;
			} else if (sdlEvent.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
				if (auto s = resize(
					static_cast<oa::U32>(sdlEvent.window.data1),
					static_cast<oa::U32>(sdlEvent.window.data2)); not s.isOk()) {
					OaLogError(oa::LogComponent::Ui,
						"Presenter resize failed: %s", s.getMessage().cStr());
					runStatus = s;
					running_ = false;
				}
			}
			oa::UiEvent e = convertSdlEvent(sdlEvent, win);
			events.pushBack(e);
			const bool primaryModifier = e.ctrl()
				|| (e.modifiers & oa::UiModifierSuper) != 0U;
			if (e.type == oa::UiEventType::KeyDown && e.key == oa::UiKey::V
				&& primaryModifier && !e.keyRepeat && ui_.wantsTextInput()) {
				char* clipboard = SDL_GetClipboardText();
				if (clipboard != nullptr && clipboard[0] != '\0') {
					oa::UiEvent paste;
					paste.type = oa::UiEventType::KeyChar;
					paste.modifiers = e.modifiers;
					paste.text = oa::String(clipboard);
					events.pushBack(oa::move(paste));
				}
				SDL_free(clipboard);
			}
		}
		oa::input::update();
		if (runStatus.isError()) break;

		auto tNow = Clock::now();
		oa::F32 deltaMs = static_cast<oa::F32>((tNow - tPrev).toMilliseconds());
		tPrev = tNow;

		if (const oa::Status updateStatus = update(deltaMs); not updateStatus.isOk()) {
			OaLogError(oa::LogComponent::Ui,
				"oa::Viewer update failed: %s", updateStatus.toString().cStr());
			runStatus = updateStatus;
			running_ = false;
			break;
		}
		beginFrame(deltaMs);
		if (const oa::Status routeStatus = routeUiEvents(
			oa::Span<const oa::UiEvent>(events.data(), events.size()));
			not routeStatus.isOk()) {
			OaLogError(oa::LogComponent::Ui,
				"oa::Viewer input failed: %s", routeStatus.toString().cStr());
			runStatus = routeStatus;
			running_ = false;
			endFrame();
			break;
		}
		if (const oa::Status renderStatus = render(ui_);
			not renderStatus.isOk()) {
			OaLogError(oa::LogComponent::Ui,
				"oa::Viewer source render failed: %s",
				renderStatus.toString().cStr());
			runStatus = renderStatus;
			running_ = false;
			endFrame();
			break;
		}
		const bool wantsTextInput = ui_.wantsTextInput();
		if (wantsTextInput != nativeTextInputActive) {
			const bool changed = wantsTextInput
				? SDL_StartTextInput(win)
				: SDL_StopTextInput(win);
			if (changed) {
				nativeTextInputActive = wantsTextInput;
			} else {
				OaLogWarn(oa::LogComponent::Ui,
					"SDL text-input state change failed: %s", SDL_GetError());
			}
		}
		if (wantsTextInput && nativeTextInputActive) {
			const oa::PixelRect caret = ui_.textInputRect();
			int logicalWidth = 0;
			int logicalHeight = 0;
			int pixelWidth = 0;
			int pixelHeight = 0;
			if (SDL_GetWindowSize(win, &logicalWidth, &logicalHeight)
				&& SDL_GetWindowSizeInPixels(win, &pixelWidth, &pixelHeight)
				&& pixelWidth > 0 && pixelHeight > 0) {
				const oa::F32 inverseScaleX = static_cast<oa::F32>(logicalWidth)
					/ static_cast<oa::F32>(pixelWidth);
				const oa::F32 inverseScaleY = static_cast<oa::F32>(logicalHeight)
					/ static_cast<oa::F32>(pixelHeight);
				const SDL_Rect area{
					.x = static_cast<int>(oa::floor(
						static_cast<oa::F32>(caret.x) * inverseScaleX)),
					.y = static_cast<int>(oa::floor(
						static_cast<oa::F32>(caret.y) * inverseScaleY)),
					.w = oa::max(1, static_cast<int>(oa::ceil(
						static_cast<oa::F32>(caret.w) * inverseScaleX))),
					.h = oa::max(1, static_cast<int>(oa::ceil(
						static_cast<oa::F32>(caret.h) * inverseScaleY))),
				};
				(void)SDL_SetTextInputArea(win, &area, 0);
			}
		}
		oa::String clipboardWrite;
		if (ui_.takeClipboardWrite(clipboardWrite)
			&& !SDL_SetClipboardText(clipboardWrite.cStr())) {
			OaLogWarn(oa::LogComponent::Ui,
				"SDL clipboard write failed: %s", SDL_GetError());
		}

		oa::Status frameStatus = oa::Status::ok();
		if (renderDependency_.isValid()) {
			if (not rt.ownsEvent(renderDependency_)) {
				frameStatus = oa::Status::invalidArgument(
					"oa::Viewer source readiness event belongs to another engine");
			}
		}
		if (frameStatus.isOk()) frameStatus = presenter.beginGraphicsBatch();
		if (frameStatus.isOk()) {
			oavk::Stream* stream = presenter.activeGraphicsBatchStream();
			const oa::Status recordStatus =
				recordRender(static_cast<VkCommandBuffer>(stream->commandBuffer));
			auto flush = presenter.flushGraphicsBatch(renderDependency_);
			if (not flush.isOk()) {
				frameStatus = flush.getStatus();
			} else {
				const oa::Event completion = *flush;
				frameStatus = setRenderCompletion(completion);
				if (frameStatus.isOk()) {
					frameStatus = markRenderSubmitted(completion);
				}
				if (frameStatus.isOk() and not recordStatus.isOk()) {
					frameStatus = recordStatus;
				}
			}
		}

		// OUT_OF_DATE handled internally; resize already applied above.
		if (frameStatus.isOk()) frameStatus = present();
		if (frameStatus.isError()) {
			OaLogError(oa::LogComponent::Ui,
				"oa::Viewer frame failed: %s", frameStatus.toString().cStr());
			runStatus = frameStatus;
			running_ = false;
		}

		endFrame();
		oa::input::clearForNextFrame();
		if (maxFrames != 0 and ++renderedFrames >= maxFrames) {
			running_ = false;
		}
	}

	// The last frame no longer host-waits before present. drain its compute
	// sampling work before user resources and the compose image are destroyed.
	if (const oa::Status syncStatus = presenter.syncGraphicsBatch(); not syncStatus.isOk()) {
		OaLogError(oa::LogComponent::Ui,
			"syncGraphicsBatch failed during UI shutdown: %s",
			syncStatus.getMessage().cStr());
		if (runStatus.isOk()) runStatus = syncStatus;
	}
	// Presentation is submitted after the compose batch and is not covered by
	// the graphics-batch timeline. drain WSI before application-owned images,
	// buffers, and the compose target are released below.
	if (const oa::Status presentStatus = presenter.waitPresentationIdle();
		not presentStatus.isOk()) {
		OaLogError(oa::LogComponent::Ui,
			"presentation drain failed during UI shutdown: %s",
			presentStatus.toString().cStr());
		if (runStatus.isOk()) runStatus = presentStatus;
	}
	const oa::Status closeStatus = closeSource();
	const oa::Status presentationStatus = destroyPresentation();
	window_ = nullptr;
	const oa::Status surfaceStatus = presenter.destroySurface(surface);
	if (nativeTextInputActive) (void)SDL_StopTextInput(win);
	SDL_DestroyWindow(win);
	const oa::Status engineStatus = closeOwnedEngine();
	oa::input::shutdown();
	if (not runStatus.isOk()) return runStatus;
	if (not closeStatus.isOk()) return closeStatus;
	if (not presentationStatus.isOk()) return presentationStatus;
	if (not surfaceStatus.isOk()) return surfaceStatus;
	return engineStatus;
}
