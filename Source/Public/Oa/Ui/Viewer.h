// OaViewer — the single windowed/headless media inspection application.
//
// `OaViewer` owns the application lifecycle, input and presentation. Media
// implementations remain in their modules and feed the viewer through explicit
// OA resources. `OaViewport` in <Oa/Ui/Viewport.h> is a passive render
// description; it is not another application.

#pragma once

#include <Oa/Core/Constant.h>
#include <Oa/Core/Navigation.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Runtime/Sync.h>
#include <Oa/Ui/DetectionOverlay.h>
#include <Oa/Ui/Image.h>
#include <Oa/Ui/Input.h>
#include <Oa/Ui/Text.h>
#include <Oa/Ui/Ui.h>
#include <Oa/Audio/AudioStream.h>
#include <Oa/Vision/Video.h>

class OaEngine;
class OaPresenter;

enum class OaViewerMode : OaU8 {
	Auto,
	Image,
	Video,
	Audio,
	Live,
};

// Non-owning live producer attached to the one OaViewer application lifecycle.
// It may render domain-specific overlays, but it never owns the window,
// swapchain, input pump or presentation submission.
class OaViewerLiveSource {
public:
	virtual ~OaViewerLiveSource() = default;
	virtual OaStatus Open(OaEngine&) { return OaStatus::Ok(); }
	virtual OaStatus Init(
		OaInputSystem&,
		OaFunc<void(bool)>) { return OaStatus::Ok(); }
	virtual void Update(OaF32) {}
	virtual void Render(OaUi&, const OaTextAtlas&, OaU32, OaU32) {}
	virtual void Event(const OaUiEvent&) {}
	[[nodiscard]] virtual OaEvent RenderReady() const { return {}; }
	virtual void MarkConsumed(const OaVkTimelineSemaphore&, OaU64) {}
	[[nodiscard]] virtual OaStatus Close() { return OaStatus::Ok(); }
};

struct OaViewerConfig {
	// Auto probes the actual decoders in a deterministic order. Explicit modes
	// skip probing and return the source decoder's error directly.
	OaViewerMode Mode = OaViewerMode::Auto;
	OaString Path = "Asset/Image/Realm1024px.jpg";
	// Required when Mode == Live. The caller owns this object and must keep it
	// alive until Run() returns.
	OaViewerLiveSource* LiveSource = nullptr;

	OaString Title = "OaViewer";
	OaU32 Width = 1280;
	OaU32 Height = 720;
	OaUiStyle Style;
	bool ShowHelp = true;
	bool ShowStats = true;
	bool ShowTimeline = true;
	bool Vsync = true;
	OaFilter PresentFilter = OaFilter::Nearest;

	// Temporal media options. Ignored for still images.
	bool Loop = true;
	bool StartPlaying = true;
	OaF32 FrameRateOverride = 0.0F;
	OaU32 ReorderDepth = 4;
	OaU32 AudioRingMilliseconds = 500;
	OaU64 AudioStepUs = 5'000'000ULL;
	OaU32 AudioWaveformBins = 2048;
	bool PreferHardwareYCbCr = true;
	OaFilter Filter = OaFilter::Nearest;

	// Optional normalized CV annotations rendered without reading back or
	// replacing the source frame.
	OaVec<OaDetectionOverlayItem> Annotations;
	OaDetectionOverlayConfig AnnotationStyle;

	// Keyboard shortcuts.
	OuiKey KeyQuit = OuiKey::Escape;
	OuiKey KeyQuitQ = OuiKey::Q;
	OuiKey KeyRed = OuiKey::Num1;
	OuiKey KeyGreen = OuiKey::Num2;
	OuiKey KeyBlue = OuiKey::Num3;
	OuiKey KeyAlpha = OuiKey::Num4;
	OuiKey KeyRGB = OuiKey::Num5;
	OuiKey KeyZoomIn = OuiKey::Equals;
	OuiKey KeyZoomOut = OuiKey::Minus;
	OuiKey KeyZoomFit = OuiKey::Num0;
	OuiKey KeyZoom100 = OuiKey::Num9;
	// Arrow keys are temporal frame controls. Keyboard panning follows the
	// shared numeric-keypad bindings; pointer and touch navigation are unchanged.
	OuiKey KeyPanUp = OuiKey::Kp8;
	OuiKey KeyPanDown = OuiKey::Kp2;
	OuiKey KeyPanLeft = OuiKey::Kp4;
	OuiKey KeyPanRight = OuiKey::Kp6;
};

class OaViewer {
public:
	OaViewer() = default;
	explicit OaViewer(const char* InPath) { Config_.Path = InPath; }
	explicit OaViewer(const OaString& InPath) { Config_.Path = InPath; }
	explicit OaViewer(const OaViewerConfig& InConfig) : Config_(InConfig) {}
	~OaViewer() = default;
	OaViewer(const OaViewer&) = delete;
	OaViewer& operator=(const OaViewer&) = delete;
	OaViewer(OaViewer&&) = delete;
	OaViewer& operator=(OaViewer&&) = delete;

	void SetMode(OaViewerMode InMode) { Config_.Mode = InMode; }
	void SetPath(const OaString& InPath) { Config_.Path = InPath; }
	void SetPath(const char* InPath) { Config_.Path = InPath; }
	void SetConfig(const OaViewerConfig& InConfig) { Config_ = InConfig; }

	[[nodiscard]] OaStatus Run();

	// Headless image sink. The same viewer render-body abstraction will replace
	// this direct file call when render-target consolidation lands.
	[[nodiscard]] static OaStatus Save(
		OaEngine& InEngine,
		const OaTexture& InTexture,
		const char* InPath);


private:
	enum class ImageViewMode : OaU8 { RGB, R, G, B, A };

	[[nodiscard]] OaStatus OpenSource(OaEngine& InEngine);
	[[nodiscard]] OaStatus InitView();
	void Update(OaF32 InDeltaMs);
	void Render(OaUi& InUi);
	void RouteEvent(const OaUiEvent& InEvent);
	void MarkRenderSubmitted(
		const OaVkTimelineSemaphore& InSemaphore,
		OaU64 InValue);
	[[nodiscard]] OaStatus CloseSource();

	[[nodiscard]] OaStatus InitPresentation(
		OaPresenter& InPresenter,
		void* InSurface);
	[[nodiscard]] OaStatus DestroyPresentation();
	[[nodiscard]] OaStatus BuildComposeImage(OaU32 InWidth, OaU32 InHeight);
	void DestroyComposeImage();
	void BeginFrame(OaF32 InDeltaMs);
	void RouteUiEvents(OaSpan<const OaUiEvent> InEvents);
	void RecordRender(VkCommandBuffer InCommandBuffer);
	void EndFrame();
	[[nodiscard]] OaStatus Resize(OaU32 InWidth, OaU32 InHeight);
	[[nodiscard]] OaStatus Present();
	void SetRenderDependency(const OaEvent& InEvent);
	void SetRenderCompletion(
		const OaVkTimelineSemaphore& InSemaphore,
		OaU64 InValue);
	[[nodiscard]] OaU32 Width() const noexcept;
	[[nodiscard]] OaU32 Height() const noexcept;
	void Quit() noexcept { Running_ = false; }
	void ResizeWindow(OaU32 InWidth, OaU32 InHeight) noexcept;
	void CapturePointer(bool InEnabled) noexcept;
	void CaptureRelativeMouse(bool InEnabled) noexcept;

	OaViewerConfig Config_;
	OaViewerMode ResolvedMode_ = OaViewerMode::Auto;

	OaTexture Image_;
	OaImagePlanes Planes_;
	OaOption<OaVideo> Video_;
	OaOption<OaAudioStream> Audio_;
	OaMatrix AudioEnvelope_;
	ImageViewMode ImageMode_ = ImageViewMode::RGB;

	OaNavigation Nav_;
	OaDetectionOverlay DetectionOverlay_;
	OaF32 StatsAccumMs_ = 0.0F;
	OaU32 StatsFrameCount_ = 0;
	OaF32 DisplayFps_ = 0.0F;
	OaF32 DisplayFrameMs_ = 0.0F;

	OaPresenter* Presenter_ = nullptr;
	OaEngine* Engine_ = nullptr;
	void* Window_ = nullptr;
	void* ComposeImage_ = nullptr;
	void* ComposeView_ = nullptr;
	void* ComposeAllocation_ = nullptr;
	OaU32 ComposeBindlessIndex_ = UINT32_MAX;
	OaU32 ComposeWidth_ = 0;
	OaU32 ComposeHeight_ = 0;
	OaTextAtlas TextAtlas_;
	OaUi Ui_;
	OaInputSystem Input_;
	OaVkTimelineSemaphore RenderCompletionSemaphore_;
	OaU64 RenderCompletionValue_ = 0;
	const OaVkTimelineSemaphore* RenderDependencySemaphore_ = nullptr;
	OaU64 RenderDependencyValue_ = 0;
	bool Running_ = false;

	[[nodiscard]] OaStatus OpenImage(OaEngine& InEngine);
	[[nodiscard]] OaStatus OpenVideo(OaEngine& InEngine);
	[[nodiscard]] OaStatus OpenAudio(OaEngine& InEngine);
	[[nodiscard]] bool HasVisualContent() const noexcept;
	[[nodiscard]] bool HasTimeline() const noexcept;
	[[nodiscard]] bool IsMediaPlaying() const noexcept;
	[[nodiscard]] bool IsMediaLooping() const noexcept;
	[[nodiscard]] OaU64 MediaDurationUs() const noexcept;
	[[nodiscard]] OaU64 MediaPositionUs() const noexcept;
	[[nodiscard]] OaPixelRect TimelineRect() noexcept;
	void ToggleMediaPlayback();
	void ToggleMediaLoop();
	void SeekMediaUs(OaU64 InTimestampUs);
	void SeekMediaFraction(OaF32 InFraction);
	void StepTemporal(OaI32 InAmount);
	void ConfigureNavigation();
	[[nodiscard]] OaStatus ConfigureOverlay();
	void RegisterCommonInput();
	void RegisterImageInput();
	void RegisterTemporalInput();
	void RenderImage(OaUi& InUi);
	void RenderVideo(OaUi& InUi);
	void RenderAudio(OaUi& InUi);
	void RenderTimeline(OaUi& InUi);
	void DrawOverlay(OaUi& InUi, OaPixelRect InDestination);
};
