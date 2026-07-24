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
#include <Oa/Ui/Viewer.h>
#include <Oa/Vision/Video.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

class OaDetectionVideoSource final : public OaViewerLiveSource {
public:
	OaString Path = TutorialVideoPath("shibuya_crossing_1080p30_av1.mp4").c_str();

	OaStatus Open(OaEngine& InEngine) override {
		OaVideoConfig config;
		config.Uri = Path;
		config.Loop = true;
		config.Filter = OaFilter::Nearest;
		auto video = OaVideo::Open(InEngine, config);
		if (!video.IsOk()) return video.GetStatus();
		Video_.Emplace(OaStdMove(*video));
		return OaStatus::Ok();
	}

	OaStatus Init(
		OaInputSystem& InInput,
		OaFunc<void(bool)> InCapturePointer) override {
		if (!Video_.HasValue()) {
			return OaStatus::Error("Detection video source is not open");
		}
		auto& runtime = *OaEngine::GetGlobal();
		auto overlay = OaDetectionOverlay::Create(runtime, {
			.MaxDetections = 64,
			.MaxGlyphs = 2048,
			.ThicknessPixels = 3.0F,
			.FontSize = 18.0F,
		});
		if (!overlay.IsOk()) {
			return overlay.GetStatus();
		}
		Overlay_ = OaStdMove(*overlay);

		Nav_.SetCapturePointer(OaStdMove(InCapturePointer));
		Nav_.SetContentSize(
			static_cast<OaF32>(Video_->Width()),
			static_cast<OaF32>(Video_->Height()));
		RegisterViewportShortcuts(InInput, Nav_);

		InInput.RegisterAction({.Name = "play", .Binding = {.Key = OuiKey::Space},
			.Callback = [this] { Video_->TogglePlay(); }});
		InInput.RegisterAction({.Name = "stepf", .Binding = {.Key = OuiKey::Right},
			.Callback = [this] { Scrub(1); }});
		InInput.RegisterAction({.Name = "stepb", .Binding = {.Key = OuiKey::Left},
			.Callback = [this] { Scrub(-1); }});

		OA_LOG_INFO(OaLogComponent::App,
			"Detection display: %s (%ux%u)",
			Path.c_str(), Video_->Width(), Video_->Height());
		OA_LOG_INFO(OaLogComponent::App,
			"GPU path: decode -> image compose -> boxes + SDF labels -> present");
		OA_LOG_INFO(OaLogComponent::App,
			"Annotations are synthetic display fixtures; replace with model/NMS output");
		return OaStatus::Ok();
	}

	void Update(OaF32 InDeltaMs) override {
		if (!Video_.HasValue()) return;
		Video_->Tick(InDeltaMs);
		Nav_.Update(InDeltaMs);
	}

	void Event(const OaUiEvent& InEvent) override {
		(void)Nav_.HandleEvent(InEvent);
	}

	void Render(
		OaUi& InUi,
		const OaTextAtlas& InTextAtlas,
		OaU32 InWidth,
		OaU32 InHeight) override {
		if (!Video_.HasValue()) return;
		if (InWidth != ViewWidth_ or InHeight != ViewHeight_) {
			ViewWidth_ = InWidth;
			ViewHeight_ = InHeight;
			Nav_.SetWindowSize(
				static_cast<OaF32>(InWidth),
				static_cast<OaF32>(InHeight));
			if (not ViewInitialized_) {
				Nav_.FitToWindow(false);
				ViewInitialized_ = true;
			}
		}
		if (Video_->Index() != AnnotatedFrame_) {
			UpdateAnnotations(InTextAtlas);
		}
		const auto& frame = Video_->CurrentFrame();
		if (frame.ImageView == VK_NULL_HANDLE) return;

		const OaPixelRect destination = {
			.X = static_cast<OaI32>(Nav_.PanX()),
			.Y = static_cast<OaI32>(Nav_.PanY()),
			.W = static_cast<OaI32>(static_cast<OaF32>(frame.Width) * Nav_.Zoom()),
			.H = static_cast<OaI32>(static_cast<OaF32>(frame.Height) * Nav_.Zoom()),
		};
		const OaPixelRect clip = {
			.X = 0,
			.Y = 0,
			.W = static_cast<OaI32>(InWidth),
			.H = static_cast<OaI32>(InHeight),
		};

		InUi.BeginPanel("detection-video", destination);
		InUi.ImageVkRgba(
			frame.Image,
			frame.ImageView,
			static_cast<OaI32>(frame.Width),
			static_cast<OaI32>(frame.Height),
			VK_IMAGE_LAYOUT_GENERAL);
		InUi.EndPanel();
		Overlay_.Draw(InUi, InTextAtlas, destination, clip);
	}

	[[nodiscard]] OaEvent RenderReady() const override {
		return Video_.HasValue() ? Video_->CurrentFrame().Ready : OaEvent{};
	}

	OaStatus MarkConsumed(const OaEvent& InCompletion) override {
		if (not InCompletion.IsValid()) {
			return OaStatus::InvalidArgument(
				"detection viewer consumption requires a valid completion event");
		}
		const OaVkTimelineWait wait = InCompletion.TimelineWait();
		if (wait.Semaphore == nullptr or wait.Value == 0U) {
			return OaStatus::InvalidArgument(
				"detection viewer consumption requires a timeline completion");
		}
		if (Video_.HasValue()) {
			Video_->MarkCurrentFrameConsumed(*wait.Semaphore, wait.Value);
		}
		Overlay_.MarkConsumed(*wait.Semaphore, wait.Value);
		return OaStatus::Ok();
	}

	OaStatus Close() override {
		Overlay_.Destroy();
		if (Video_.HasValue()) {
			const OaStatus status = Video_->Close();
			Video_.Reset();
			return status;
		}
		return OaStatus::Ok();
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

	void UpdateAnnotations(const OaTextAtlas& InTextAtlas) {
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
			InTextAtlas);
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
	OaU32 ViewWidth_ = 0;
	OaU32 ViewHeight_ = 0;
	bool ViewInitialized_ = false;
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

	OaDetectionVideoSource source;
	if (argc > 1) source.Path = argv[1];
	OaViewer viewer({
		.Mode = OaViewerMode::Live,
		.LiveSource = &source,
		.Title = "OA Detection Video",
		.Width = 1280,
		.Height = 720,
		.ShowHelp = false,
		.ShowStats = false,
		.ShowTimeline = false,
	});
	return viewer.Run().IsOk() ? 0 : 1;
}
