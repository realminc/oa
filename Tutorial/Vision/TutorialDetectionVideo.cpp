// OA Tutorial: GPU-resident video with reusable detection overlays.
//
// The annotations are synthetic so the display path can be validated without
// bundling a model checkpoint. Replace BuildAnnotations() with inference/NMS
// output; the OaDetection record and GPU consumer stay unchanged.
//
// Pipeline:
//   Vulkan Video decode -> RGBA image -> boxes + IBM Plex SDF labels -> present
//
// Usage:
//   TutorialDetectionVideo [video.mp4]


#include "../Ml/TutorialMl.h"
#include "TutorialVision.h"

#include <Oa/Core/Log.h>
#include <Oa/Core/Navigation.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Ui/DetectionOverlay.h>
#include <Oa/Ui/DeviceUi.h>
#include <Oa/Vision/Video.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

class OaDetectionVideoApp final : public OaDeviceUiApp {
public:
	OaString Path = TutorialVideoPath("shibuya_crossing_1080p30_av1.mp4").c_str();

	OaStatus OnDeviceReady(OaGraphicsEngine& InEngine) override {
		OaVideoConfig config;
		config.Uri = Path;
		config.Loop = true;
		config.Filter = OaFilter::Nearest;
		auto video = OaVideo::Open(InEngine, config);
		if (!video.IsOk()) return video.GetStatus();
		Video_.Emplace(OaStdMove(*video));
		return OaStatus::Ok();
	}

	void OnInit(OaDeviceUi& InGpui) override {
		if (!Video_.HasValue()) {
			Quit();
			return;
		}
		auto& runtime = *OaComputeEngine::GetGlobal();
		auto overlay = OaDetectionOverlay::Create(runtime, {
			.MaxDetections = 64,
			.MaxGlyphs = 2048,
			.ThicknessPixels = 3.0F,
			.FontSize = 18.0F,
		});
		if (!overlay.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::App,
				"Detection overlay creation failed: %s",
				overlay.GetStatus().ToString().c_str());
			Quit();
			return;
		}
		Overlay_ = OaStdMove(*overlay);

		Nav_.SetCapturePointer([this](bool InOn) { CapturePointer(InOn); });
		Nav_.SetContentSize(
			static_cast<OaF32>(Video_->Width()),
			static_cast<OaF32>(Video_->Height()));
		Nav_.SetWindowSize(
			static_cast<OaF32>(InGpui.Width()),
			static_cast<OaF32>(InGpui.Height()));
		Nav_.FitToWindow(false);
		RegisterViewportShortcuts(InGpui.Input(), Nav_);

		auto& input = InGpui.Input();
		input.RegisterAction({.Name = "quit", .Binding = {.Key = OuiKey::Escape},
			.Callback = [this] { Quit(); }});
		input.RegisterAction({.Name = "quitq", .Binding = {.Key = OuiKey::Q},
			.Callback = [this] { Quit(); }});
		input.RegisterAction({.Name = "play", .Binding = {.Key = OuiKey::Space},
			.Callback = [this] { Video_->TogglePlay(); }});
		input.RegisterAction({.Name = "stepf", .Binding = {.Key = OuiKey::Right},
			.Callback = [this] { Scrub(1); }});
		input.RegisterAction({.Name = "stepb", .Binding = {.Key = OuiKey::Left},
			.Callback = [this] { Scrub(-1); }});

		UpdateAnnotations();
		OA_LOG_INFO(OaLogComponent::App,
			"Detection display: %s (%ux%u)",
			Path.c_str(), Video_->Width(), Video_->Height());
		OA_LOG_INFO(OaLogComponent::App,
			"GPU path: decode -> image compose -> boxes + SDF labels -> present");
		OA_LOG_INFO(OaLogComponent::App,
			"Annotations are synthetic display fixtures; replace with model/NMS output");
	}

	void OnUpdate(OaF32 InDeltaMs) override {
		if (!Video_.HasValue()) return;
		Video_->Tick(InDeltaMs);
		Nav_.Update(InDeltaMs);
		if (Video_->Index() != AnnotatedFrame_) UpdateAnnotations();
	}

	void OnEvent(const OaUiEvent& InEvent) override {
		(void)Nav_.HandleEvent(InEvent);
	}

	void OnRender(OaUi& InUi) override {
		if (!Video_.HasValue()) return;
		const auto& frame = Video_->CurrentFrame();
		if (frame.ImageView == VK_NULL_HANDLE) return;
		if (frame.ReadySemaphore != nullptr && frame.ReadyValue > 0) {
			Gpui().SetRenderDependency(*frame.ReadySemaphore, frame.ReadyValue);
		}

		const OaPixelRect destination = {
			.X = static_cast<OaI32>(Nav_.PanX()),
			.Y = static_cast<OaI32>(Nav_.PanY()),
			.W = static_cast<OaI32>(static_cast<OaF32>(frame.Width) * Nav_.Zoom()),
			.H = static_cast<OaI32>(static_cast<OaF32>(frame.Height) * Nav_.Zoom()),
		};
		const OaPixelRect clip = {
			.X = 0,
			.Y = 0,
			.W = static_cast<OaI32>(Gpui().Width()),
			.H = static_cast<OaI32>(Gpui().Height()),
		};

		InUi.BeginPanel("detection-video", destination);
		InUi.ImageVkRgba(
			frame.Image,
			frame.ImageView,
			static_cast<OaI32>(frame.Width),
			static_cast<OaI32>(frame.Height),
			VK_IMAGE_LAYOUT_GENERAL);
		InUi.EndPanel();
		Overlay_.Draw(InUi, Gpui().TextAtlas(), destination, clip);
	}

	void OnRenderSubmitted(
		const OaVkTimelineSemaphore& InSemaphore,
		OaU64 InValue) override {
		if (Video_.HasValue()) {
			Video_->MarkCurrentFrameConsumed(InSemaphore, InValue);
		}
		Overlay_.MarkConsumed(InSemaphore, InValue);
	}

	void OnShutdown(OaDeviceUi&) override {
		Overlay_.Destroy();
		if (Video_.HasValue()) {
			Video_->Destroy();
			Video_.Reset();
		}
	}

private:
	void Scrub(OaI32 InFrames) {
		if (!Video_.HasValue()) return;
		Video_->Pause();
		const OaStatus status = Video_->StepFrames(InFrames);
		if (!status.IsOk()) {
			OA_LOG_WARN(OaLogComponent::App,
				"Video scrub failed: %s", status.ToString().c_str());
		}
	}

	void UpdateAnnotations() {
		if (!Video_.HasValue() || !Overlay_.IsValid()) return;
		const OaF32 phase = static_cast<OaF32>(Video_->Index() % 240) / 240.0F;
		const OaF32 wave = std::sin(phase * 6.28318530718F);

		OaVec<OaDetectionOverlayItem> items;
		items.PushBack({
			.Detection = {
				.CenterX = 0.29F + wave * 0.035F,
				.CenterY = 0.53F,
				.Width = 0.12F,
				.Height = 0.42F,
				.Confidence = 0.98F,
				.ClassId = 0,
				.ColorRgba = OaColor::Success().ToU32(),
				.TrackId = 17,
			},
			.Label = "person 98% / track 17",
		});
		items.PushBack({
			.Detection = {
				.CenterX = 0.62F - wave * 0.02F,
				.CenterY = 0.67F,
				.Width = 0.24F,
				.Height = 0.20F,
				.Confidence = 0.93F,
				.ClassId = 2,
				.ColorRgba = OaColor::Cyan().ToU32(),
				.TrackId = 31,
			},
			.Label = "car 93% / track 31",
		});
		items.PushBack({
			.Detection = {
				.CenterX = 0.79F,
				.CenterY = 0.48F + wave * 0.025F,
				.Width = 0.09F,
				.Height = 0.25F,
				.Confidence = 0.87F,
				.ClassId = 1,
				.ColorRgba = OaColor::Warning().ToU32(),
				.TrackId = 44,
			},
			.Label = "bicycle 87% / track 44",
		});

		const OaStatus status = Overlay_.Update(
			OaSpan<const OaDetectionOverlayItem>(items.Data(), items.Size()),
			Gpui().TextAtlas());
		if (status.IsOk()) {
			AnnotatedFrame_ = Video_->Index();
		} else if (status.GetCode() != OaStatusCode::Unavailable) {
			OA_LOG_WARN(OaLogComponent::App,
				"Detection overlay update failed: %s", status.ToString().c_str());
		}
	}

	OaOption<OaVideo> Video_;
	OaDetectionOverlay Overlay_;
	OaNavigation Nav_;
	OaI64 AnnotatedFrame_ = -1;
};

int main(int argc, char** argv) {
	const OaI32 deviceIndex = TutorialPreParseDeviceIndex(argc, argv);
	if (deviceIndex >= 0) {
		const std::string index = std::to_string(deviceIndex);
#if defined(_WIN32)
		_putenv_s("OA_DEVICE", index.c_str());
#else
		::setenv("OA_DEVICE", index.c_str(), 1);
#endif
	}

	OaDetectionVideoApp app;
	if (argc > 1) app.Path = argv[1];
	return app.Run({
		.Title = "OA Detection Video",
		.Width = 1280,
		.Height = 720,
	}).IsOk() ? 0 : 1;
}
