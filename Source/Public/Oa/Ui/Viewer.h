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
#include <Oa/Ui/DetectionOverlay.h>
#include <Oa/Ui/DeviceUi.h>
#include <Oa/Ui/Image.h>
#include <Oa/Ui/Input.h>
#include <Oa/Audio/AudioStream.h>
#include <Oa/Vision/Video.h>

class OaComputeEngine;

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
	virtual OaStatus Open(OaGraphicsEngine&) { return OaStatus::Ok(); }
	virtual void Init(OaDeviceUi&) {}
	virtual void Update(OaF32) {}
	virtual void Render(OaUi&, OaDeviceUi&) {}
	virtual void Event(const OaUiEvent&) {}
	virtual void MarkConsumed(const OaVkTimelineSemaphore&, OaU64) {}
	virtual void Close(OaDeviceUi&) {}
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
	bool ShowHelp = true;
	bool ShowStats = true;
	bool ShowTimeline = true;

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

class OaViewer : public OaDeviceUiApp {
public:
	OaViewer() = default;
	explicit OaViewer(const char* InPath) { Config_.Path = InPath; }
	explicit OaViewer(const OaString& InPath) { Config_.Path = InPath; }
	explicit OaViewer(const OaViewerConfig& InConfig) : Config_(InConfig) {}

	void SetMode(OaViewerMode InMode) { Config_.Mode = InMode; }
	void SetPath(const OaString& InPath) { Config_.Path = InPath; }
	void SetPath(const char* InPath) { Config_.Path = InPath; }
	void SetConfig(const OaViewerConfig& InConfig) { Config_ = InConfig; }

	[[nodiscard]] OaStatus Run();

	// Headless image sink. The same viewer render-body abstraction will replace
	// this direct file call when render-target consolidation lands.
	[[nodiscard]] static OaStatus Save(
		OaComputeEngine& InEngine,
		const OaTexture& InTexture,
		const char* InPath);

	OaStatus OnDeviceReady(OaGraphicsEngine& InEngine) override;
	void OnInit(OaDeviceUi& InGpui) override;
	void OnUpdate(OaF32 InDeltaMs) override;
	void OnRender(OaUi& InUi) override;
	void OnEvent(const OaUiEvent& InEvent) override;
	void OnRenderSubmitted(
		const OaVkTimelineSemaphore& InSemaphore,
		OaU64 InValue) override;
	void OnShutdown(OaDeviceUi& InGpui) override;

private:
	enum class ImageViewMode : OaU8 { RGB, R, G, B, A };

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

	[[nodiscard]] OaStatus OpenImage(OaGraphicsEngine& InEngine);
	[[nodiscard]] OaStatus OpenVideo(OaGraphicsEngine& InEngine);
	[[nodiscard]] OaStatus OpenAudio(OaGraphicsEngine& InEngine);
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
	void ConfigureNavigation(OaDeviceUi& InGpui);
	void ConfigureOverlay(OaDeviceUi& InGpui);
	void RegisterCommonInput(OaDeviceUi& InGpui);
	void RegisterImageInput(OaDeviceUi& InGpui);
	void RegisterTemporalInput(OaDeviceUi& InGpui);
	void RenderImage(OaUi& InUi);
	void RenderVideo(OaUi& InUi);
	void RenderAudio(OaUi& InUi);
	void RenderTimeline(OaUi& InUi);
	void DrawOverlay(OaUi& InUi, OaPixelRect InDestination);
};
