// Viewer — the single windowed/headless media inspection application.
//
// `Viewer` owns the application lifecycle, input and presentation. Media
// implementations remain in their modules and feed the viewer through explicit
// OA resources. `Viewport` in <oa/ui/viewport.h> is a passive render
// description; it is not another application.

#pragma once

#include <oa/core/constant.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/runtime/oaVk.h>
#include <oa/runtime/event.h>
#include <oa/ui/detectionOverlay.h>
#include <oa/ui/image.h>
#include <oa/ui/input.h>
#include <oa/ui/navigation.h>
#include <oa/ui/text.h>
#include <oa/ui/ui.h>
#include <oa/audio/audioPlayer.h>
#include <oa/vision/videoPlayer.h>

namespace oa { class Engine; }
namespace oa { class Presenter; }
namespace oa {
class Renderer;
class RenderFrame;
}

namespace oa {

enum class ViewerMode : oa::U8 {
	Auto,
	Image,
	Video,
	Audio,
	Live,
};

enum class ViewerAudioView : oa::U8 {
	Waveform,
	Spectrum,
	Mel,
};

struct ViewerLiveCapabilities {
	// Raw events are offered after Ui focus/popup ownership and before named
	// Viewer shortcuts. Registered InputSystem actions do not require this.
	bool receivesEvents = false;
	// renderReady supplies the producer event for resources recorded this frame.
	bool publishesRenderDependency = false;
	// markConsumed receives the exact Viewer submission for source-side reuse.
	bool retainsConsumerCompletion = false;
};

// Non-owning live producer attached to the one Viewer application lifecycle.
// It may render domain-specific overlays, but it never owns the window,
// swapchain, input pump or presentation submission.
class ViewerLiveSource {
public:
	virtual ~ViewerLiveSource() = default;
	[[nodiscard]] virtual ViewerLiveCapabilities capabilities() const noexcept = 0;
	[[nodiscard]] virtual oa::Status open(oa::Engine&) = 0;
	virtual oa::Status init(
		InputSystem&,
		oa::Fn<void(bool)>) = 0;
	[[nodiscard]] virtual oa::Status update(oa::F32) = 0;
	[[nodiscard]] virtual oa::Status render(
		Ui&, const TextAtlas&, oa::U32, oa::U32) = 0;
	[[nodiscard]] virtual oa::Status event(const UiEvent&) {
		return oa::Status::error(
			oa::StatusCode::Unimplemented,
			"ViewerLiveSource does not declare raw-event capability");
	}
	[[nodiscard]] virtual oa::Result<oa::Event> renderReady() const {
		return oa::Status::error(
			oa::StatusCode::Unimplemented,
			"ViewerLiveSource does not declare render-dependency capability");
	}
	// Retain the exact Viewer submission until every resource rendered by this
	// frame can be recycled. A source may reject stale or foreign completion.
	[[nodiscard]] virtual oa::Status markConsumed(const oa::Event&) {
		return oa::Status::error(
			oa::StatusCode::Unimplemented,
			"ViewerLiveSource does not declare consumer-completion capability");
	}
	[[nodiscard]] virtual oa::Status close() = 0;
};

struct ViewerConfig {
	// Auto probes the actual decoders in a deterministic order. Explicit modes
	// skip probing and return the source decoder's error directly.
	ViewerMode mode = ViewerMode::Auto;
	oa::String path = "asset/image/SpaceCathedral.jpg";
	// Required when Mode == Live. The caller owns this object and must keep it
	// alive until run() returns.
	ViewerLiveSource* liveSource = nullptr;

	oa::String title = "Viewer";
	oa::U32 width = 1280;
	oa::U32 height = 720;
	UiStyle style = UiStyle::editorDark();
	bool showHelp = true;
	bool showStats = true;
	bool showTimeline = true;
	bool vsync = true;
	// Use the OA-rendered client-side frame instead of the platform decoration.
	// This is opt-in so desktop applications follow the host environment.
	bool customWindowDecoration = false;
	oa::Filter presentFilter = oa::Filter::Nearest;

	// Temporal media options. Ignored for still images.
	bool loop = true;
	bool startPlaying = true;
	oa::F32 frameRateOverride = 0.0F;
	oa::U32 reorderDepth = 4;
	oa::U32 audioRingMilliseconds = 500;
	oa::U64 audioStepUs = 5'000'000ULL;
	oa::U32 audioWaveformBins = 2048;
	// file-backed audio analysis is recorded once on the source engine and
	// published to the graphics queue through one exact producer event. The
	// effective hop grows for long files so Spectrum/Mel stay bounded by
	// audioAnalysisFrames rather than allocating proportional to duration.
	ViewerAudioView audioView = ViewerAudioView::Waveform;
	oa::U32 audioAnalysisFrames = 2048;
	oa::U32 audioFftSize = 1024;
	oa::U32 audioHopSize = 256;
	oa::U32 audioMelBins = 80;
	bool showAudioViewSelector = true;
	bool preferHardwareYCbCr = true;
	oa::Filter filter = oa::Filter::Nearest;

	// Optional normalized CV annotations rendered without reading back or
	// replacing the source frame.
	oa::Vec<DetectionOverlayItem> annotations;
	DetectionOverlayConfig annotationStyle;

	// keyboard shortcuts.
	UiKey keyQuit = UiKey::Escape;
	UiKey keyQuitQ = UiKey::Q;
	UiKey keyRed = UiKey::Num1;
	UiKey keyGreen = UiKey::Num2;
	UiKey keyBlue = UiKey::Num3;
	UiKey keyAlpha = UiKey::Num4;
	UiKey keyRgb = UiKey::Num5;
	UiKey keyZoomIn = UiKey::Equals;
	UiKey keyZoomOut = UiKey::Minus;
	UiKey keyZoomFit = UiKey::Num0;
	UiKey keyZoom100 = UiKey::Num9;
	// Arrow keys are temporal frame controls. keyboard panning follows the
	// shared numeric-keypad bindings; pointer and touch navigation are unchanged.
	UiKey keyPanUp = UiKey::Kp8;
	UiKey keyPanDown = UiKey::Kp2;
	UiKey keyPanLeft = UiKey::Kp4;
	UiKey keyPanRight = UiKey::Kp6;
};

// Windowed or headless application session for inspecting OA images, video,
// audio and live GPU output. The Viewer owns UI, input, window and presenter
// state. It either creates one engine for run() or borrows the caller's engine
// for run(oa::Engine&); it never creates a second device runtime in that path.
class Viewer {
public:
	Viewer() = default;
	explicit Viewer(const char* inPath) { config_.path = inPath; }
	explicit Viewer(const oa::String& inPath) { config_.path = inPath; }
	explicit Viewer(const ViewerConfig& inConfig) : config_(inConfig) {}
	~Viewer() = default;
	Viewer(const Viewer&) = delete;
	Viewer& operator=(const Viewer&) = delete;
	Viewer(Viewer&&) = delete;
	Viewer& operator=(Viewer&&) = delete;

	void setMode(ViewerMode inMode) { config_.mode = inMode; }
	void setPath(const oa::String& inPath) { config_.path = inPath; }
	void setPath(const char* inPath) { config_.path = inPath; }
	void setConfig(const ViewerConfig& inConfig) { config_ = inConfig; }

	[[nodiscard]] oa::Status run();
	// run the windowed session against an existing presentation-capable engine.
	// The viewer borrows the engine and closes only its presenter/window state.
	[[nodiscard]] oa::Status run(oa::Engine& inEngine);

	// Blocking one-shot display sinks. Matrix and image overloads record the
	// RGBA8 conversion into the engine's matching private recorder, submit it
	// once, and pass its exact completion into presentation.
	[[nodiscard]] static oa::Status show(
		oa::Engine& inEngine,
		const oa::Matrix& inMatrix,
		const ViewerConfig& inConfig = {});
	[[nodiscard]] static oa::Status show(
		oa::Engine& inEngine,
		const oa::Image& inImage,
		const ViewerConfig& inConfig = {});
	// Ready buffer-backed textures can be displayed without conversion.
	[[nodiscard]] static oa::Status show(
		oa::Engine& inEngine,
		const oa::Texture& inTexture,
		const ViewerConfig& inConfig = {});
	// Image-backed renderer targets require their generation-safe frame and
	// owning session. The blocking viewer returns its final graphics completion
	// to that session before the presentation timelines are destroyed.
	[[nodiscard]] static oa::Status show(
		oa::Engine& inEngine,
		oa::Renderer& inRenderer,
		const oa::RenderFrame& inFrame,
		const ViewerConfig& inConfig = {});

	// Headless sink for a ready buffer-backed texture. Image-backed renderer
	// frames use their owning session's saveTo/consumeReadback contract.
	[[nodiscard]] static oa::Status save(
		oa::Engine& inEngine,
		const oa::Texture& inTexture,
		const char* inPath);


private:
	enum class ImageViewMode : oa::U8 { RGB, R, G, B, A };

	[[nodiscard]] oa::Status runApplication(oa::Engine* inBorrowedEngine);
	[[nodiscard]] oa::Status openSource(oa::Engine& inEngine);
	[[nodiscard]] oa::Status initView();
	[[nodiscard]] oa::Status update(oa::F32 inDeltaMs);
	[[nodiscard]] oa::Status render(Ui& inUi);
	[[nodiscard]] oa::Status routeEvent(const UiEvent& inEvent);
	[[nodiscard]] oa::Status markRenderSubmitted(const oa::Event& inCompletion);
	[[nodiscard]] oa::Status finalizeBorrowedFrame();
	[[nodiscard]] oa::Status closeSource();
	[[nodiscard]] const oa::Texture& imageSource() const noexcept;

	[[nodiscard]] oa::Status initPresentation(
		oa::Presenter& inPresenter,
		void* inSurface);
	[[nodiscard]] oa::Status destroyPresentation();
	[[nodiscard]] oa::Status buildComposeImage(oa::U32 inWidth, oa::U32 inHeight);
	void destroyComposeImage();
	void beginFrame(oa::F32 inDeltaMs);
	[[nodiscard]] oa::Status routeUiEvents(oa::Span<const UiEvent> inEvents);
	[[nodiscard]] oa::Status recordRender(VkCommandBuffer inCommandBuffer);
	void endFrame();
	[[nodiscard]] oa::Status resize(oa::U32 inWidth, oa::U32 inHeight);
	[[nodiscard]] oa::Status present();
	void setRenderDependency(const oa::Event& inEvent);
	[[nodiscard]] oa::Status setRenderCompletion(const oa::Event& inCompletion);
	[[nodiscard]] oa::U32 width() const noexcept;
	[[nodiscard]] oa::U32 height() const noexcept;
	void quit() noexcept { running_ = false; }
	void resizeWindow(oa::U32 inWidth, oa::U32 inHeight) noexcept;
	void capturePointer(bool inEnabled) noexcept;
	void captureRelativeMouse(bool inEnabled) noexcept;

	ViewerConfig config_;
	ViewerMode resolvedMode_ = ViewerMode::Auto;
	// Snapshotted once when a live source opens. Capability changes cannot alter
	// event or completion routing halfway through a session.
	ViewerLiveCapabilities liveCapabilities_;

	oa::Texture image_;
	// Direct show() inputs are borrowed for the blocking run() call. The caller
	// retains ownership; CloseSource never destroys this resource.
	const oa::Texture* borrowedImage_ = nullptr;
	oa::Event borrowedImageReady_;
	VkPipelineStageFlags2 borrowedImageSourceStageMask_ =
		VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
	VkAccessFlags2 borrowedImageSourceAccessMask_ =
		VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
	oa::Fn<oa::Status(const oa::Event&)> borrowedFrameMarkConsumed_;
	oa::Fn<oa::Status()> borrowedFrameAbandon_;
	oa::Fn<oa::Status()> borrowedFrameCollect_;
	ImagePlanes planes_;
	oa::Optional<oa::VideoPlayer> video_;
	oa::Optional<oa::AudioPlayer> audio_;
	oa::Matrix audioEnvelope_;
	oa::Matrix audioSpectrum_;
	oa::Matrix audioMel_;
	oa::Event audioAnalysisReady_;
	ViewerAudioView audioView_ = ViewerAudioView::Waveform;
	UiTabBarState audioViewTabs_;
	ImageViewMode imageMode_ = ImageViewMode::RGB;

	Navigation nav_;
	DetectionOverlay detectionOverlay_;
	oa::F32 statsAccumMs_ = 0.0F;
	oa::U32 statsFrameCount_ = 0;
	oa::F32 displayFps_ = 0.0F;
	oa::F32 displayFrameMs_ = 0.0F;

	oa::Presenter* presenter_ = nullptr;
	oa::Engine* sourceEngine_ = nullptr;
	oa::Engine* engine_ = nullptr;
	void* window_ = nullptr;
	void* composeImage_ = nullptr;
	void* composeView_ = nullptr;
	void* composeAllocation_ = nullptr;
	oa::U32 composeBindlessIndex_ = UINT32_MAX;
	oa::U32 composeWidth_ = 0;
	oa::U32 composeHeight_ = 0;
	TextAtlas textAtlas_;
	Ui ui_;
	InputSystem input_;
	oa::Event renderCompletion_;
	oa::Event renderDependency_;
	bool running_ = false;

	[[nodiscard]] oa::Status openImage(oa::Engine& inEngine);
	[[nodiscard]] oa::Status openVideo(oa::Engine& inEngine);
	[[nodiscard]] oa::Status openAudio(oa::Engine& inEngine);
	[[nodiscard]] bool hasVisualContent() const noexcept;
	[[nodiscard]] bool hasTimeline() const noexcept;
	[[nodiscard]] bool isMediaPlaying() const noexcept;
	[[nodiscard]] bool isMediaLooping() const noexcept;
	[[nodiscard]] oa::U64 mediaDurationUs() const noexcept;
	[[nodiscard]] oa::U64 mediaPositionUs() const noexcept;
	[[nodiscard]] PixelRect timelineRect() const noexcept;
	[[nodiscard]] PixelRect temporalButtonsRect() const noexcept;
	[[nodiscard]] PixelRect audioViewTabRect() const noexcept;
	[[nodiscard]] PixelRect audioVisualizationRect() const noexcept;
	void toggleMediaPlayback();
	void toggleMediaLoop();
	void seekMediaUs(oa::U64 inTimestampUs);
	void seekMediaFraction(oa::F32 inFraction);
	void stepTemporal(oa::I32 inAmount);
	[[nodiscard]] oa::Status configureNavigation();
	[[nodiscard]] oa::Status configureOverlay();
	[[nodiscard]] oa::Status registerCommonInput();
	void registerImageInput();
	void registerTemporalInput();
	void renderImage(Ui& inUi);
	void renderVideo(Ui& inUi);
	void renderAudio(Ui& inUi);
	void renderAudioViewSelector(Ui& inUi);
	void renderTimeline(Ui& inUi);
	void renderTemporalButtons(Ui& inUi);
	void drawOverlay(Ui& inUi, PixelRect inDestination);
	[[nodiscard]] oa::Status initWindowDecoration();
	void destroyWindowDecoration();
	void refreshWindowDecorationScale();
	[[nodiscard]] oa::Status rebuildWindowTitleGlyphs();
	[[nodiscard]] bool routeWindowDecorationEvent(void* inEvent);
	void renderWindowDecoration(Ui& inUi);
	[[nodiscard]] oa::U32 windowDecorationHeight() const noexcept;

	GlyphBuffer windowTitleGlyphs_;
	oa::F32 windowPixelScaleX_ = 1.0F;
	oa::F32 windowPixelScaleY_ = 1.0F;
	oa::I32 windowHoveredControl_ = 0;
	oa::I32 windowPressedControl_ = 0;
	bool windowDecorationActive_ = false;
	bool windowMaximized_ = false;
};

}  // namespace oa
