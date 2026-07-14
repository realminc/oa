// OaVideoPlayer - reusable windowed video player app.

#include <Oa/Ui/VideoPlayer.h>

#include <Oa/Core/Log.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Ui/Ui.h>
#include <Oa/Vision/VideoDecoder.h>

namespace {

[[nodiscard]] const char* VideoCodecName(OaVideoCodec InCodec) noexcept {
	switch (InCodec) {
		case OaVideoCodec::H264: return "H.264";
		case OaVideoCodec::H265: return "H.265";
		case OaVideoCodec::AV1:  return "AV1";
		case OaVideoCodec::VP9:  return "VP9";
		default:                 return "Unknown";
	}
}

} // namespace

OaVideoPlayer::OaVideoPlayer(const OaString& InPath, OaVideoCodec InCodec) {
	Config_.Path = InPath;
	Config_.Codec = InCodec;
}

OaVideoPlayer::OaVideoPlayer(const char* InPath, OaVideoCodec InCodec) {
	Config_.Path = InPath;
	Config_.Codec = InCodec;
}

OaStatus OaVideoPlayer::Run() {
	OaUiStyle style;
	style.Background = {0.01F, 0.01F, 0.01F, 1.0F};
	return OaDeviceUiApp::Run({
		.Title  = Config_.Title,
		.Width  = Config_.Width,
		.Height = Config_.Height,
		.Style  = style
	});
}

OaStatus OaVideoPlayer::OnDeviceReady(OaGraphicsEngine& InRt) {
	OaVideoConfig cfg;
	cfg.Uri = Config_.Path;
	cfg.Loop = Config_.Loop;
	cfg.FrameRateOverride = Config_.FrameRateOverride;
	cfg.ReorderDepth = Config_.ReorderDepth;
	cfg.PreferHardwareYCbCr = Config_.PreferHardwareYCbCr;
	cfg.Filter = Config_.Filter;

	auto videoResult = OaVideo::Open(InRt, cfg);
	if (not videoResult.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App, "OaVideo::Open failed: %s",
			videoResult.GetStatus().ToString().c_str());
		return videoResult.GetStatus();
	}
	Video_.Emplace(OaStdMove(*videoResult));
	return OaStatus::Ok();
}

void OaVideoPlayer::OnInit(OaDeviceUi& InGpui) {
	if (not Video_.HasValue()) {
		Quit();
		return;
	}

	const OaU32 winW = (Video_->Width()  >= Config_.Width)
		? Video_->Width() : Video_->Width() * 2U;
	const OaU32 winH = (Video_->Height() >= Config_.Height)
		? Video_->Height() : Video_->Height() * 2U;
	ResizeWindow(winW, winH);

	Nav_.SetCapturePointer([this](bool InOn) { CaptureRelativeMouse(InOn); });
	Nav_.SetContentSize(static_cast<OaF32>(Video_->Width()),
	                    static_cast<OaF32>(Video_->Height()));
	Nav_.SetWindowSize(static_cast<OaF32>(Gpui().Width()),
	                   static_cast<OaF32>(Gpui().Height()));
	Nav_.FitToWindow(false);

	if (Config_.ShowHelp) {
		OA_LOG_INFO(OaLogComponent::App, "OA Video Player");
		OA_LOG_INFO(OaLogComponent::App, "  Video: %s", Config_.Path.c_str());
		OA_LOG_INFO(OaLogComponent::App, "  Codec: %s",
			VideoCodecName(Video_->GetContainerInfo().Codec));
		OA_LOG_INFO(OaLogComponent::App, "  Size:  %ux%u @ %u fps",
			Video_->Width(), Video_->Height(), Video_->FrameRate());
		OA_LOG_INFO(OaLogComponent::App, "  Space=play/pause  Arrows=scrub -1/+1/-5/+5 frames");
		OA_LOG_INFO(OaLogComponent::App, "%s", OaNavigationHelpLine());
		OA_LOG_INFO(OaLogComponent::App, "  Q/Esc=Quit");
	}

	auto& input = InGpui.Input();
	input.RegisterAction({.Name = "quit",  .Binding = {.Key = OuiKey::Escape}, .Callback = [this] { Quit(); }});
	input.RegisterAction({.Name = "quitq", .Binding = {.Key = OuiKey::Q},      .Callback = [this] { Quit(); }});
	input.RegisterAction({.Name = "play",  .Binding = {.Key = OuiKey::Space},  .Callback = [this] {
		if (Video_.HasValue()) {
			Video_->TogglePlay();
			OA_LOG_INFO(OaLogComponent::App, "Playback %s",
				Video_->IsPlaying() ? "playing" : "paused");
		}
	}});
	input.RegisterAction({.Name = "stepf",  .Binding = {.Key = OuiKey::Right}, .Callback = [this] { Scrub(1); }});
	input.RegisterAction({.Name = "stepb",  .Binding = {.Key = OuiKey::Left},  .Callback = [this] { Scrub(-1); }});
	input.RegisterAction({.Name = "stepf5", .Binding = {.Key = OuiKey::Up},    .Callback = [this] { Scrub(5); }});
	input.RegisterAction({.Name = "stepb5", .Binding = {.Key = OuiKey::Down},  .Callback = [this] { Scrub(-5); }});

	RegisterViewportShortcuts(input, Nav_);
}

void OaVideoPlayer::OnUpdate(OaF32 InDeltaMs) {
	if (Video_.HasValue()) {
		Video_->Tick(InDeltaMs);
	}
	Nav_.Update(InDeltaMs);

	StatsAccumMs_ += InDeltaMs;
	++StatsFrameCount_;
	if (StatsAccumMs_ >= 500.0F) {
		DisplayFrameMs_ = StatsAccumMs_ / static_cast<OaF32>(StatsFrameCount_);
		DisplayFps_ = static_cast<OaF32>(StatsFrameCount_) * 1000.0F / StatsAccumMs_;
		StatsAccumMs_ = 0.0F;
		StatsFrameCount_ = 0;
	}
}

void OaVideoPlayer::OnEvent(const OaUiEvent& InEvent) {
	if (not Video_.HasValue()) { return; }
	(void)Nav_.HandleEvent(InEvent);
}

void OaVideoPlayer::OnRender(OaUi& InOui) {
	if (not Video_.HasValue()) { return; }
	const auto& frame = Video_->CurrentFrame();
	if (frame.ImageView == VK_NULL_HANDLE) { return; }
	if (frame.ReadySemaphore != nullptr && frame.ReadyValue > 0) {
		Gpui().SetRenderDependency(*frame.ReadySemaphore, frame.ReadyValue);
	}

	const OaF32 scaledW = static_cast<OaF32>(frame.Width) * Nav_.Zoom();
	const OaF32 scaledH = static_cast<OaF32>(frame.Height) * Nav_.Zoom();
	const OaI32 x = static_cast<OaI32>(Nav_.PanX());
	const OaI32 y = static_cast<OaI32>(Nav_.PanY());
	const OaI32 dW = static_cast<OaI32>(scaledW);
	const OaI32 dH = static_cast<OaI32>(scaledH);

	InOui.BeginPanel("video", {.X = x, .Y = y, .W = dW, .H = dH});
	InOui.ImageVkRgba(frame.Image, frame.ImageView,
		static_cast<OaI32>(frame.Width),
		static_cast<OaI32>(frame.Height),
		VK_IMAGE_LAYOUT_GENERAL);
	InOui.EndPanel();

	if (not Config_.ShowStats) {
		return;
	}

	InOui.PushStyle({
		.BorderWidth = 0.0F,
		.ShadowBlur = 0.0F,
		.ShadowOffset = 0.0F,
		.Background = {0.0F, 0.0F, 0.0F, 0.6F},
		.Text = {1.0F, 1.0F, 1.0F, 1.0F},
	});
	InOui.BeginPanel("stats", {12, 12, 155, 52},
		{.Width = OaUiSizing::Fixed(155), .Height = OaUiSizing::Fixed(52)});
	InOui.LabelFmt("FPS:  %.1f", DisplayFps_);
	InOui.LabelFmt("Frame: %.2f ms", DisplayFrameMs_);
	InOui.EndPanel();
	InOui.PopStyle();
}

void OaVideoPlayer::OnRenderSubmitted(
	const OaVkTimelineSemaphore& InSemaphore,
	OaU64 InValue) {
	if (Video_.HasValue()) {
		Video_->MarkCurrentFrameConsumed(InSemaphore, InValue);
	}
}

void OaVideoPlayer::OnShutdown(OaDeviceUi&) {
	if (Video_.HasValue()) {
		Video_->Destroy();
		Video_.Reset();
	}
}

void OaVideoPlayer::Scrub(OaI32 InFrameDelta) {
	if (not Video_.HasValue()) { return; }
	Video_->Pause();
	const auto status = Video_->StepFrames(InFrameDelta);
	if (not status.IsOk()) {
		OA_LOG_WARN(OaLogComponent::App,
			"Video scrub failed: %s", status.ToString().c_str());
	}
}
