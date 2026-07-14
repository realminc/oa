// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial: OaViewportCanvas — Unified 2D/3D Canvas
//
// Image or video on a 2D orthographic canvas with OaNavigation input
// Controls:
//   Space         — toggle play / pause (video)
//   Left/Right    — scrub one frame (video)
//   +/-           — zoom in/out
//   0             — fit to window
//   9             — 100% zoom
//   F / 0         — fit to window
//   LMB / MMB drag — pan (tracks past window edge)
//   RMB drag       — Maya zoom (horizontal)
//   Wheel          — pan up/down
//   Side wheel     — pan left/right (MX Master)
//   Ctrl + wheel   — dolly zoom in/out
//   2-finger      — pan (touchpad)
//   pinch         — zoom (touchpad)
//   Q / Esc       — quit
//
// Usage:
//   ./TutorialViewportCanvas [image.jpg|video.mp4]
// ═══════════════════════════════════════════════════════════════════════════

#include <Oa/Core/Log.h>
#include <Oa/Core/Navigation.h>
#include <Oa/Ui/Camera.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Ui/DeviceUi.h>
#include <Oa/Ui/Image.h>
#include <Oa/Vision/ItVideo.h>
#include "TutorialVision.h"

#include <cstdlib>
#include <cstring>

class OaCanvasApp : public OaDeviceUiApp {
public:
	std::string ContentPath = TutorialVideoPath("shibuya_crossing_1080p30_av1.mp4");

	void OnInit(OaDeviceUi& InGpui) override {
		auto& rt = *OaComputeEngine::GetGlobal();

		if (IsVideoPath(ContentPath.c_str())) {
			InitVideo(rt);
		} else {
			InitImage(rt);
		}

		if (!HasContent()) {
			OA_LOG_ERROR(OaLogComponent::App, "Failed to load content: %s", ContentPath.c_str());
			Quit();
			return;
		}

		Nav_.SetCapturePointer([this](bool InOn) { CaptureRelativeMouse(InOn); });
		Nav_.SetContentSize(static_cast<OaF32>(ImageWidth_),
		                    static_cast<OaF32>(ImageHeight_));
		Nav_.SetWindowSize(static_cast<OaF32>(Gpui().Width()),
		                   static_cast<OaF32>(Gpui().Height()));

		Camera_ = OaCamera(static_cast<OaF32>(ImageWidth_), static_cast<OaF32>(ImageHeight_));
		Camera_.SetOrthographic(static_cast<OaF32>(ImageWidth_), static_cast<OaF32>(ImageHeight_));

		Nav_.FitToWindow(false);
		SyncCamera();

		RegisterInput(InGpui);

		OA_LOG_INFO(OaLogComponent::App, "═══════════════════════════════════════════════════");
		OA_LOG_INFO(OaLogComponent::App, "OA Canvas Viewport");
		OA_LOG_INFO(OaLogComponent::App, "  Content: %s", ContentPath.c_str());
		OA_LOG_INFO(OaLogComponent::App, "  Type: %s", Video_.HasValue() ? "video" : "image");
		OA_LOG_INFO(OaLogComponent::App, "  Size: %ux%u", ImageWidth_, ImageHeight_);
		if (Video_.HasValue()) {
			OA_LOG_INFO(OaLogComponent::App, "  Space=play/pause  Arrows=scrub");
		}
		OA_LOG_INFO(OaLogComponent::App, "%s", OaNavigationHelpLine());
		OA_LOG_INFO(OaLogComponent::App, "═══════════════════════════════════════════════════");
	}

	void OnUpdate(OaF32 InDeltaMs) override {
		if (Video_.HasValue()) {
			Video_->Tick(InDeltaMs);
		}
		Nav_.Update(InDeltaMs);
		SyncCamera();
	}

	void OnEvent(const OaUiEvent& InEvent) override {
		if (!HasContent()) { return; }
		(void)Nav_.HandleEvent(InEvent);
	}

	void OnRender(OaUi& InOui) override {
		if (!HasContent()) { return; }

		if (Video_.HasValue()) {
			const auto& frame = Video_->CurrentFrame();
			if (frame.ImageView == VK_NULL_HANDLE) { return; }
			if (frame.ReadySemaphore != nullptr && frame.ReadyValue > 0) {
				Gpui().SetRenderDependency(*frame.ReadySemaphore, frame.ReadyValue);
			}
			RenderPanel(InOui, frame.Image, frame.ImageView, frame.Width, frame.Height);
			return;
		}

		if (Image_.IsValid()) {
			const OaF32 zoom = Nav_.Zoom();
			const OaI32 displayW = static_cast<OaI32>(static_cast<OaF32>(ImageWidth_) * zoom);
			const OaI32 displayH = static_cast<OaI32>(static_cast<OaF32>(ImageHeight_) * zoom);
			InOui.BeginPanel("canvas", {
				.X = static_cast<OaI32>(Nav_.PanX()),
				.Y = static_cast<OaI32>(Nav_.PanY()),
				.W = displayW,
				.H = displayH
			});
			InOui.Image(Image_.BindlessIndex(), ImageWidth_, ImageHeight_);
			InOui.EndPanel();
		}
	}

	void OnShutdown(OaDeviceUi&) override {
		auto& rt = *OaComputeEngine::GetGlobal();
		if (Image_.IsValid()) { Image_.Destroy(rt); }
		if (Video_.HasValue()) { Video_->Destroy(); }
	}

private:
	OaTexture Image_;
	OaOption<OaItVideo> Video_;
	OaU32 ImageWidth_ = 0;
	OaU32 ImageHeight_ = 0;

	OaCamera Camera_;
	OaNavigation Nav_;

	bool IsVideoPath(const char* InPath) const {
		const char* ext = strrchr(InPath, '.');
		if (!ext) { return false; }
		return (strcmp(ext, ".mp4") == 0 ||
		        strcmp(ext, ".mkv") == 0 ||
		        strcmp(ext, ".avi") == 0 ||
		        strcmp(ext, ".mov") == 0);
	}

	void InitVideo(OaComputeEngine& rt) {
		if (not OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::AV1)) {
			OA_LOG_ERROR(OaLogComponent::App, "AV1 decode not supported");
			return;
		}
		OaItVideoConfig cfg;
		cfg.Path = ContentPath.c_str();
		cfg.Loop = true;
		if (auto result = OaItVideo::Create(rt, cfg); result.IsOk()) {
			Video_.Emplace(OaStdMove(*result));
			ImageWidth_ = Video_->Width();
			ImageHeight_ = Video_->Height();
		}
	}

	void InitImage(OaComputeEngine& rt) {
		if (auto result = OaTexture::LoadFile(rt, ContentPath.c_str()); result.IsOk()) {
			Image_ = OaStdMove(*result);
			ImageWidth_ = Image_.Width;
			ImageHeight_ = Image_.Height;
		}
	}

	bool HasContent() const {
		return Image_.IsValid() || (Video_.HasValue() && Video_->Width() > 0);
	}

	void SyncCamera() {
		Nav_.UpdateCamera(Camera_);
	}

	void RegisterInput(OaDeviceUi& InGpui) {
		auto& input = InGpui.Input();

		input.RegisterAction({.Name = "quit", .Binding = {.Key = OuiKey::Escape},
			.Callback = [this] { Quit(); }});
		input.RegisterAction({.Name = "quitq", .Binding = {.Key = OuiKey::Q},
			.Callback = [this] { Quit(); }});

		input.RegisterAction({.Name = "play", .Binding = {.Key = OuiKey::Space},
			.Callback = [this] {
				if (Video_.HasValue()) { Video_->TogglePlay(); }
			}});
		input.RegisterAction({.Name = "stepf", .Binding = {.Key = OuiKey::Right},
			.Callback = [this] {
				if (Video_.HasValue()) { Video_->Pause(); Video_->StepFrames(1); }
			}});
		input.RegisterAction({.Name = "stepb", .Binding = {.Key = OuiKey::Left},
			.Callback = [this] {
				if (Video_.HasValue()) { Video_->Pause(); Video_->StepFrames(-1); }
			}});

		RegisterViewportShortcuts(input, Nav_);
	}

	void RenderPanel(OaUi& InOui, VkImage InImage, VkImageView InView, OaU32 InW, OaU32 InH) {
		const OaF32 zoom = Nav_.Zoom();
		const OaI32 displayW = static_cast<OaI32>(static_cast<OaF32>(InW) * zoom);
		const OaI32 displayH = static_cast<OaI32>(static_cast<OaF32>(InH) * zoom);
		InOui.BeginPanel("canvas", {
			.X = static_cast<OaI32>(Nav_.PanX()),
			.Y = static_cast<OaI32>(Nav_.PanY()),
			.W = displayW,
			.H = displayH
		});
		InOui.ImageVkRgba(InImage, InView, static_cast<OaI32>(InW), static_cast<OaI32>(InH),
			VK_IMAGE_LAYOUT_GENERAL);
		InOui.EndPanel();
	}
};

int main(int argc, char** argv) {
	OaCanvasApp app;
	if (argc > 1) { app.ContentPath = argv[1]; }

	OaUiStyle style;
	style.Background = {0.01F, 0.01F, 0.01F, 1.0F};

	return app.Run({
		.Title = "OA Canvas Viewport",
		.Width = 1280,
		.Height = 720,
		.Style = style
	}).IsOk() ? 0 : 1;
}
