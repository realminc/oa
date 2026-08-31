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
#include <vulkan/vulkan_core.h>
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

enum class ViewerCanvasBackground : oa::U8 {
	Dark,
	Gradient,
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
	virtual oa::Status init(InputSystem&,	oa::Fn<void(bool)>) = 0;
	[[nodiscard]] virtual oa::Status update(oa::F32) = 0;
	[[nodiscard]] virtual oa::Status render(Ui&, const TextAtlas&, oa::U32, oa::U32) = 0;
	[[nodiscard]] virtual oa::Status event(const UiEvent&) {
		return oa::Status::error(
			oa::StatusCode::Unimplemented,
			"ViewerLiveSource does not declare raw-event capability"
		);
	}
	[[nodiscard]] virtual oa::Result<oa::Event> renderReady() const {
		return oa::Status::error(
			oa::StatusCode::Unimplemented,
			"ViewerLiveSource does not declare render-dependency capability"
		);
	}
	// Retain the exact Viewer submission until every resource rendered by this
	// frame can be recycled. A source may reject stale or foreign completion.
	[[nodiscard]] virtual oa::Status markConsumed(const oa::Event&) {
		return oa::Status::error(
			oa::StatusCode::Unimplemented,
			"ViewerLiveSource does not declare consumer-completion capability"
		);
	}
	[[nodiscard]] virtual oa::Status close() = 0;
};

struct ViewerConfig {
	// Auto probes the actual decoders in a deterministic order. Explicit modes
	// skip probing and return the source decoder's error directly.
	ViewerMode mode = ViewerMode::Auto;
	// Empty by default: reusable library code does not assume an SDK checkout.
	oa::String path;
	// Required when Mode == Live. The caller owns this object and must keep it
	// alive until run() returns.
	ViewerLiveSource* liveSource = nullptr;

	oa::String title = "Viewer";
	oa::U32 width = 1280;
	oa::U32 height = 720;
	UiStyle style = UiStyle::viewerDark();
	bool showHelp = true;
	bool showStats = false;
	bool showTimeline = true;
	bool vsync = true;
	// The media-first client frame is the default. Unsupported SDL backends
	// fail safely to the platform decoration before the Vulkan surface exists.
	bool customWindowDecoration = true;
	// Playing video hides inactive chrome after this delay. Pointer, keyboard,
	// popup and paused states reveal it without changing playback ownership.
	bool autoHideControls = true;
	oa::U32 controlsHideDelayMs = 2'000U;
	// One policy controls viewport fit/reset and retained widget transitions.
	oa::UiMotionSpeed motionSpeed = oa::UiMotionSpeed::Fast;
	oa::Filter presentFilter = oa::Filter::Nearest;
	// Visual-source workspace behind the opaque image/video canvas. The grid is
	// image-space anchored: one source pixel is one world unit.
	ViewerCanvasBackground canvasBackground = ViewerCanvasBackground::Gradient;
	bool showCanvasGrid = true;

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
	bool preferHardwareYCbCr = true;
	oa::Filter filter = oa::Filter::Nearest;

	// Optional normalized CV annotations rendered without reading back or
	// replacing the source frame.
	oa::Vector<DetectionOverlayItem> annotations;
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
	UiKey keyCanvasBackground = UiKey::B;
	UiKey keyCanvasGrid = UiKey::G;
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

	// Unified blocking preview front door. Paths are decoder-probed as image,
	// video, then audio and may either own an application engine or borrow the
	// caller's presentation-capable engine. Direct GPU values always borrow the
	// caller's engine and retain the exact completion contract of show().
	[[nodiscard]] static oa::Status preview(
		const char* inPath,
		const ViewerConfig& inConfig = {});
	[[nodiscard]] static oa::Status preview(
		const oa::String& inPath,
		const ViewerConfig& inConfig = {});
	[[nodiscard]] static oa::Status preview(
		oa::Engine& inEngine,
		const char* inPath,
		const ViewerConfig& inConfig = {});
	[[nodiscard]] static oa::Status preview(
		oa::Engine& inEngine,
		const oa::String& inPath,
		const ViewerConfig& inConfig = {});
	[[nodiscard]] static oa::Status preview(
		oa::Engine& inEngine,
		const oa::Matrix& inMatrix,
		const ViewerConfig& inConfig = {});
	[[nodiscard]] static oa::Status preview(
		oa::Engine& inEngine,
		const oa::Image& inImage,
		const ViewerConfig& inConfig = {});
	[[nodiscard]] static oa::Status preview(
		oa::Engine& inEngine,
		const oa::Texture& inTexture,
		const ViewerConfig& inConfig = {});
	[[nodiscard]] static oa::Status preview(
		oa::Engine& inEngine,
		oa::Renderer& inRenderer,
		const oa::RenderFrame& inFrame,
		const ViewerConfig& inConfig = {});

	// Compatibility spelling for blocking one-shot display sinks. Matrix and image overloads record the
	// RGBA8 conversion into the engine's matching private recorder, submit it
	// once, and pass its exact completion into presentation. New code uses
	// preview() so file-backed and direct values share one discoverable verb.
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

	[[nodiscard]] oa::Status initPresentation(oa::Presenter& inPresenter, void* inSurface);
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
	VkPipelineStageFlags2 borrowedImageSourceStageMask_ =	VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
	VkAccessFlags2 borrowedImageSourceAccessMask_ =	VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
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
	oa::Optional<oa::F32> pendingTimelineSeekFraction_;
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
	[[nodiscard]] bool hasMediaAudio() const noexcept;
	[[nodiscard]] bool isMediaMuted() const noexcept;
	[[nodiscard]] oa::U64 mediaDurationUs() const noexcept;
	[[nodiscard]] oa::U64 mediaPositionUs() const noexcept;
	[[nodiscard]] oa::F32 mediaPositionFraction() const noexcept;
	[[nodiscard]] oa::F32 displayedMediaFraction() const noexcept;
	[[nodiscard]] PixelRect timelineRect() const noexcept;
	[[nodiscard]] PixelRect temporalButtonsRect() const noexcept;
	[[nodiscard]] PixelRect audioVisualizationRect() const noexcept;
	void toggleMediaPlayback();
	void toggleMediaLoop();
	void toggleMediaMuted();
	void pauseMedia();
	void seekMediaUs(oa::U64 inTimestampUs);
	void seekMediaFraction(oa::F32 inFraction);
	void handleTimelineSeek(oa::F32 inFraction, bool inChanged, bool inActive);
	void stepTemporal(oa::I32 inAmount);
	[[nodiscard]] oa::Status configureNavigation();
	[[nodiscard]] oa::Status configureOverlay();
	[[nodiscard]] oa::Status registerCommonInput();
	void registerImageInput();
	void registerTemporalInput();
	void renderVisualWorkspace(
		Ui& inUi,
		PixelRect inDestination,
		bool inDrawGrid);
	void renderImage(Ui& inUi);
	void renderVideo(Ui& inUi);
	void renderAudio(Ui& inUi);
	void renderTimeline(Ui& inUi);
	void renderTemporalButtons(Ui& inUi);
	[[nodiscard]] bool mediaChromeVisible() const noexcept;
	void revealMediaChrome() noexcept;
	void drawOverlay(Ui& inUi, PixelRect inDestination);
	[[nodiscard]] oa::Status initWindowDecoration();
	void destroyWindowDecoration();
	void refreshWindowDecorationScale();
	[[nodiscard]] bool routeWindowDecorationEvent(void* inEvent);
	void renderWindowDecoration(Ui& inUi);
	[[nodiscard]] oa::U32 windowDecorationHeight() const noexcept;

	oa::F32 windowPixelScaleX_ = 1.0F;
	oa::F32 windowPixelScaleY_ = 1.0F;
	oa::F32 mediaChromeIdleMs_ = 0.0F;
	bool mediaPointerInside_ = true;
	bool showRemainingTime_ = false;
	bool windowDecorationActive_ = false;
	bool windowTransparent_ = false;
	bool windowFullscreen_ = false;
	bool windowMaximized_ = false;
};

}  // namespace oa
