// OaViewer — one application lifecycle for still images, video and audio.

#include <Oa/Ui/Viewer.h>

#include <Oa/Core/Log.h>
#include <Oa/Audio/AudioDecoder.h>
#include <Oa/Audio/FnAudio.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Ui/Ui.h>
#include <Oa/Vision/FnImage.h>

#include <algorithm>
#include <cstdlib>

namespace {

const char* ViewerModeName(OaViewerMode InMode) {
	switch (InMode) {
		case OaViewerMode::Image: return "image";
		case OaViewerMode::Video: return "video";
		case OaViewerMode::Audio: return "audio";
		case OaViewerMode::Live: return "live";
		default: return "auto";
	}
}

const char* VideoCodecName(OaVideoCodec InCodec) {
	switch (InCodec) {
		case OaVideoCodec::H264: return "H.264";
		case OaVideoCodec::H265: return "H.265";
		case OaVideoCodec::AV1: return "AV1";
		case OaVideoCodec::VP9: return "VP9";
		default: return "Unknown";
	}
}

} // namespace

OaStatus OaViewer::Show(
	OaContext& InContext,
	const OaMatrix& InMatrix,
	const OaViewerConfig& InConfig)
{
	OaEngine& engine = InContext.Engine();
	if (not engine.HasGraphics()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaViewer::Show requires a presentation-capable OaEngine");
	}

	auto textureResult = OaTexture::FromMatrix(InContext, InMatrix);
	if (not textureResult.IsOk()) return textureResult.GetStatus();
	OaTexture texture = OaStdMove(*textureResult);

	auto submitResult = InContext.Submit();
	if (not submitResult.IsOk()) {
		texture.Destroy(engine);
		return submitResult.GetStatus();
	}
	const OaEvent ready = *submitResult;

	OaViewer viewer(InConfig);
	viewer.Config_.Mode = OaViewerMode::Image;
	viewer.Config_.Path.Clear();
	viewer.BorrowedImage_ = &texture;
	viewer.BorrowedImageReady_ = ready;
	const OaStatus runStatus = viewer.Run(engine);
	const OaStatus waitStatus = InContext.Wait(ready);
	texture.Destroy(engine);
	if (not runStatus.IsOk()) return runStatus;
	return waitStatus;
}

OaStatus OaViewer::Show(
	OaContext& InContext,
	const OaImage& InImage,
	const OaViewerConfig& InConfig)
{
	OaEngine& engine = InContext.Engine();
	if (not engine.HasGraphics()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaViewer::Show requires a presentation-capable OaEngine");
	}

	auto textureResult = OaTexture::FromImage(InContext, InImage);
	if (not textureResult.IsOk()) return textureResult.GetStatus();
	OaTexture texture = OaStdMove(*textureResult);

	auto submitResult = InContext.Submit();
	if (not submitResult.IsOk()) {
		texture.Destroy(engine);
		return submitResult.GetStatus();
	}
	const OaEvent ready = *submitResult;

	OaViewer viewer(InConfig);
	viewer.Config_.Mode = OaViewerMode::Image;
	viewer.Config_.Path.Clear();
	viewer.BorrowedImage_ = &texture;
	viewer.BorrowedImageReady_ = ready;
	const OaStatus runStatus = viewer.Run(engine);
	const OaStatus waitStatus = InContext.Wait(ready);
	texture.Destroy(engine);
	if (not runStatus.IsOk()) return runStatus;
	return waitStatus;
}

OaStatus OaViewer::Show(
	OaEngine& InEngine,
	const OaTexture& InTexture,
	const OaViewerConfig& InConfig)
{
	if (not InEngine.HasGraphics()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaViewer::Show requires a presentation-capable OaEngine");
	}
	OaViewer viewer(InConfig);
	viewer.Config_.Mode = OaViewerMode::Image;
	viewer.Config_.Path.Clear();
	viewer.BorrowedImage_ = &InTexture;
	return viewer.Run(InEngine);
}

OaStatus OaViewer::Save(
	OaEngine& InEngine,
	const OaTexture& InTexture,
	const char* InPath) {
	if (not InTexture.IsValid()) {
		return OaStatus::InvalidArgument("OaViewer::Save: invalid texture");
	}
	if (InPath == nullptr or InPath[0] == '\0') {
		return OaStatus::InvalidArgument("OaViewer::Save: empty output path");
	}
	return OaFnImage::SaveFile(InEngine, InTexture, InPath);
}

OaStatus OaViewer::OpenImage(OaEngine& InEngine) {
	if (BorrowedImage_ != nullptr) {
		const OaTexture& image = *BorrowedImage_;
		const OaVkBuffer& buffer = image.DeviceBuf;
		if (not image.IsValid() or image.IsImageBacked()
			or buffer.Buffer == nullptr or buffer.NodeIndex != 0U
			or buffer.AllocatorIdentity != InEngine.Allocator.Allocator) {
			return OaStatus::InvalidArgument(
				"OaViewer::Show requires a buffer-backed texture owned by its engine");
		}
		if (BorrowedImageReady_.IsValid()) {
			if (not InEngine.OwnsEvent(BorrowedImageReady_)) {
				return OaStatus::InvalidArgument(
					"OaViewer::Show readiness event belongs to another engine");
			}
			SetRenderDependency(BorrowedImageReady_);
		}
		ResolvedMode_ = OaViewerMode::Image;
		return OaStatus::Ok();
	}

	auto image = OaTexture::LoadFile(InEngine, Config_.Path);
	if (not image.IsOk()) return image.GetStatus();
	Image_ = OaStdMove(*image);

	auto planes = OaImagePlanes::LoadFile(InEngine, Config_.Path);
	if (planes.IsOk()) Planes_ = OaStdMove(*planes);
	ResolvedMode_ = OaViewerMode::Image;
	return OaStatus::Ok();
}

OaStatus OaViewer::OpenVideo(OaEngine& InEngine) {
	OaVideoConfig config;
	config.Uri = Config_.Path;
	config.Loop = Config_.Loop;
	config.FrameRateOverride = Config_.FrameRateOverride;
	config.ReorderDepth = Config_.ReorderDepth;
	config.PreferHardwareYCbCr = Config_.PreferHardwareYCbCr;
	config.Filter = Config_.Filter;
	config.StartPlaying = Config_.StartPlaying;

	auto video = OaVideo::Open(InEngine, config);
	if (not video.IsOk()) return video.GetStatus();
	Video_.Emplace(OaStdMove(*video));
	ResolvedMode_ = OaViewerMode::Video;
	return OaStatus::Ok();
}

OaStatus OaViewer::OpenAudio(OaEngine& InEngine) {
	OaAudioStreamConfig config;
	config.Uri = Config_.Path;
	config.Loop = Config_.Loop;
	config.RingMilliseconds = Config_.AudioRingMilliseconds;
	auto audio = OaAudioStream::Open(InEngine, config);
	if (not audio.IsOk()) return audio.GetStatus();
	Audio_.Emplace(OaStdMove(*audio));
	auto decoded = OaAudioDecoder::LoadFile(OaPath(config.Uri));
	if (decoded.IsOk()) {
		AudioEnvelope_ = OaFnAudio::WaveformEnvelope(
			*decoded,
			Config_.AudioWaveformBins);
		if (not AudioEnvelope_.IsEmpty()) {
			auto& context = OaContext::GetDefault();
			OA_RETURN_IF_ERROR(context.Execute());
			OA_RETURN_IF_ERROR(context.Sync());
		}
	} else {
		OA_LOG_WARN(OaLogComponent::App,
			"OaViewer waveform unavailable: %s",
			decoded.GetStatus().ToString().c_str());
	}
	if (Config_.StartPlaying) {
		OA_RETURN_IF_ERROR(Audio_->Play());
	}
	ResolvedMode_ = OaViewerMode::Audio;
	return OaStatus::Ok();
}

OaStatus OaViewer::OpenSource(OaEngine& InEngine) {
	switch (Config_.Mode) {
		case OaViewerMode::Image:
			return OpenImage(InEngine);
		case OaViewerMode::Video:
			return OpenVideo(InEngine);
		case OaViewerMode::Audio:
			return OpenAudio(InEngine);
		case OaViewerMode::Live:
			if (Config_.LiveSource == nullptr) {
				return OaStatus::InvalidArgument(
					"OaViewer live mode requires a live source");
			}
			ResolvedMode_ = OaViewerMode::Live;
			return Config_.LiveSource->Open(InEngine);
		case OaViewerMode::Auto:
			break;
	}

	const OaStatus imageStatus = OpenImage(InEngine);
	if (imageStatus.IsOk()) return imageStatus;

	const OaStatus videoStatus = OpenVideo(InEngine);
	if (videoStatus.IsOk()) return videoStatus;

	const OaStatus audioStatus = OpenAudio(InEngine);
	if (audioStatus.IsOk()) return audioStatus;

	OA_LOG_ERROR(OaLogComponent::App,
		"OaViewer could not open '%s' as image (%s), video (%s) or audio (%s)",
		Config_.Path.c_str(),
		imageStatus.ToString().c_str(),
		videoStatus.ToString().c_str(),
		audioStatus.ToString().c_str());
	return OaStatus::Error(
		OaStatusCode::InvalidArgument,
		"OaViewer: unsupported or invalid media source");
}

bool OaViewer::HasVisualContent() const noexcept {
	return ResolvedMode_ == OaViewerMode::Image or ResolvedMode_ == OaViewerMode::Video;
}

bool OaViewer::HasTimeline() const noexcept {
	return Config_.ShowTimeline
		and ((ResolvedMode_ == OaViewerMode::Video and Video_.HasValue())
			or (ResolvedMode_ == OaViewerMode::Audio and Audio_.HasValue()))
		and MediaDurationUs() > 0U;
}

bool OaViewer::IsMediaPlaying() const noexcept {
	if (Video_.HasValue()) return Video_->IsPlaying();
	if (Audio_.HasValue()) return Audio_->IsPlaying();
	return false;
}

bool OaViewer::IsMediaLooping() const noexcept {
	if (Video_.HasValue()) return Video_->IsLooping();
	return Config_.Loop;
}

OaU64 OaViewer::MediaDurationUs() const noexcept {
	if (Video_.HasValue()) return Video_->DurationUs();
	if (Audio_.HasValue()) return Audio_->DurationUs();
	return 0U;
}

OaU64 OaViewer::MediaPositionUs() const noexcept {
	if (Video_.HasValue()) return Video_->PositionUs();
	if (Audio_.HasValue()) return Audio_->PositionUs();
	return 0U;
}

OaPixelRect OaViewer::TimelineRect() noexcept {
	const OaI32 margin = ResolvedMode_ == OaViewerMode::Audio ? 32 : 24;
	const OaI32 height = ResolvedMode_ == OaViewerMode::Audio
		? std::max<OaI32>(1, static_cast<OaI32>(Height()) - 96)
		: 20;
	const OaI32 width = std::max<OaI32>(1, static_cast<OaI32>(Width()) - margin * 2);
	const OaI32 y = ResolvedMode_ == OaViewerMode::Audio
		? std::max<OaI32>(24, (static_cast<OaI32>(Height()) - height) / 2)
		: std::max<OaI32>(0, static_cast<OaI32>(Height()) - 52);
	return {margin, y, width, height};
}

void OaViewer::ToggleMediaPlayback() {
	if (Video_.HasValue()) {
		Video_->TogglePlay();
	} else if (Audio_.HasValue()) {
		if (Audio_->IsPlaying()) Audio_->Pause();
		else if (const OaStatus status = Audio_->Play(); not status.IsOk()) {
			OA_LOG_WARN(OaLogComponent::App,
				"OaViewer audio playback failed: %s", status.ToString().c_str());
		}
	}
}

void OaViewer::ToggleMediaLoop() {
	const bool loop = not IsMediaLooping();
	Config_.Loop = loop;
	if (Video_.HasValue()) Video_->SetLoop(loop);
	if (Audio_.HasValue()) Audio_->SetLoop(loop);
}

void OaViewer::SeekMediaUs(OaU64 InTimestampUs) {
	OaStatus status = OaStatus::Ok();
	if (Video_.HasValue()) status = Video_->SeekUs(InTimestampUs);
	else if (Audio_.HasValue()) status = Audio_->Seek(InTimestampUs);
	if (not status.IsOk()) {
		OA_LOG_WARN(OaLogComponent::App,
			"OaViewer media seek failed: %s", status.ToString().c_str());
	}
}

void OaViewer::SeekMediaFraction(OaF32 InFraction) {
	const OaU64 duration = MediaDurationUs();
	if (duration == 0U) return;
	const long double clamped = static_cast<long double>(
		std::clamp(InFraction, 0.0F, 1.0F));
	SeekMediaUs(static_cast<OaU64>(clamped * static_cast<long double>(duration)));
}

void OaViewer::StepTemporal(OaI32 InAmount) {
	if (Video_.HasValue()) {
		Video_->Pause();
		const OaStatus status = Video_->StepFrames(InAmount);
		if (not status.IsOk()) {
			OA_LOG_WARN(OaLogComponent::App,
				"OaViewer video scrub failed: %s", status.ToString().c_str());
		}
		return;
	}
	if (not Audio_.HasValue()) return;
	Audio_->Pause();
	const OaU64 position = Audio_->PositionUs();
	const OaU64 step = Config_.AudioStepUs * static_cast<OaU64>(std::abs(InAmount));
	const OaU64 target = InAmount < 0
		? (position > step ? position - step : 0U)
		: std::min(MediaDurationUs(), position + step);
	SeekMediaUs(target);
}

void OaViewer::ConfigureNavigation() {
	if (not HasVisualContent()) {
		ResizeWindow(Config_.Width, Config_.Height);
		return;
	}
	OaU32 contentWidth = 0;
	OaU32 contentHeight = 0;
	if (ResolvedMode_ == OaViewerMode::Image and ImageSource().IsValid()) {
		contentWidth = static_cast<OaU32>(ImageSource().Width);
		contentHeight = static_cast<OaU32>(ImageSource().Height);
	} else if (ResolvedMode_ == OaViewerMode::Video and Video_.HasValue()) {
		contentWidth = Video_->Width();
		contentHeight = Video_->Height();
	}

	if (contentWidth == 0 or contentHeight == 0) {
		Quit();
		return;
	}

	const OaU32 windowWidth = std::min(
		Config_.Width, std::max<OaU32>(contentWidth, contentWidth * 2U));
	const OaU32 windowHeight = std::min(
		Config_.Height, std::max<OaU32>(contentHeight, contentHeight * 2U));
	ResizeWindow(windowWidth, windowHeight);

	Nav_.SetCapturePointer([this](bool InOn) { CaptureRelativeMouse(InOn); });
	Nav_.SetContentSize(
		static_cast<OaF32>(contentWidth),
		static_cast<OaF32>(contentHeight));
	Nav_.SetWindowSize(
		static_cast<OaF32>(Width()),
		static_cast<OaF32>(Height()));
	Nav_.FitToWindow(false);
}

OaStatus OaViewer::ConfigureOverlay() {
	if (Config_.Annotations.Empty()) return OaStatus::Ok();
	if (Engine_ == nullptr) {
		return OaStatus::Error(
			"OaViewer overlay creation requires an attached engine");
	}
	auto overlay = OaDetectionOverlay::Create(*Engine_, Config_.AnnotationStyle);
	if (not overlay.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App,
			"OaViewer overlay creation failed: %s",
			overlay.GetStatus().ToString().c_str());
		return overlay.GetStatus();
	}
	DetectionOverlay_ = OaStdMove(*overlay);
	const OaStatus update = DetectionOverlay_.Update(
		OaSpan<const OaDetectionOverlayItem>(
			Config_.Annotations.Data(), Config_.Annotations.Size()),
		TextAtlas_);
	if (not update.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App,
			"OaViewer overlay update failed: %s", update.ToString().c_str());
		DetectionOverlay_.Destroy();
		return update;
	}
	return OaStatus::Ok();
}

void OaViewer::RegisterCommonInput() {
	auto& input = Input_;
	input.RegisterAction({.Name = "quit", .Binding = {.Key = Config_.KeyQuit},
		.Callback = [this] { Quit(); }});
	input.RegisterAction({.Name = "quitq", .Binding = {.Key = Config_.KeyQuitQ},
		.Callback = [this] { Quit(); }});

	if (HasVisualContent()) {
		OaNavigationShortcuts keys;
		keys.ZoomIn = Config_.KeyZoomIn;
		keys.ZoomOut = Config_.KeyZoomOut;
		keys.ZoomFit = Config_.KeyZoomFit;
		keys.Zoom100 = Config_.KeyZoom100;
		keys.PanUp = Config_.KeyPanUp;
		keys.PanDown = Config_.KeyPanDown;
		keys.PanLeft = Config_.KeyPanLeft;
		keys.PanRight = Config_.KeyPanRight;
		RegisterViewportShortcuts(input, Nav_, keys);
	}
}

void OaViewer::RegisterImageInput() {
	auto& input = Input_;
	input.RegisterAction({.Name = "red", .Binding = {.Key = Config_.KeyRed},
		.Callback = [this] { ImageMode_ = ImageViewMode::R; }});
	input.RegisterAction({.Name = "green", .Binding = {.Key = Config_.KeyGreen},
		.Callback = [this] { ImageMode_ = ImageViewMode::G; }});
	input.RegisterAction({.Name = "blue", .Binding = {.Key = Config_.KeyBlue},
		.Callback = [this] { ImageMode_ = ImageViewMode::B; }});
	input.RegisterAction({.Name = "alpha", .Binding = {.Key = Config_.KeyAlpha},
		.Callback = [this] { ImageMode_ = ImageViewMode::A; }});
	input.RegisterAction({.Name = "rgb", .Binding = {.Key = Config_.KeyRGB},
		.Callback = [this] { ImageMode_ = ImageViewMode::RGB; }});
}

void OaViewer::RegisterTemporalInput() {
	auto& input = Input_;
	input.RegisterAction({.Name = "play", .Binding = {.Key = OuiKey::Space},
		.Callback = [this] { ToggleMediaPlayback(); }});
	input.RegisterAction({.Name = "loop", .Binding = {.Key = OuiKey::L},
		.Callback = [this] { ToggleMediaLoop(); }});
	input.RegisterAction({.Name = "stepf", .Binding = {.Key = OuiKey::Right},
		.Callback = [this] { StepTemporal(1); }});
	input.RegisterAction({.Name = "stepb", .Binding = {.Key = OuiKey::Left},
		.Callback = [this] { StepTemporal(-1); }});
	input.RegisterAction({.Name = "stepf5", .Binding = {.Key = OuiKey::Up},
		.Callback = [this] { StepTemporal(5); }});
	input.RegisterAction({.Name = "stepb5", .Binding = {.Key = OuiKey::Down},
		.Callback = [this] { StepTemporal(-5); }});
}

OaStatus OaViewer::InitView() {
	if (ResolvedMode_ == OaViewerMode::Auto) {
		return OaStatus::Error("OaViewer has no resolved source");
	}

	ConfigureNavigation();
	OA_RETURN_IF_ERROR(ConfigureOverlay());
	RegisterCommonInput();

	if (ResolvedMode_ == OaViewerMode::Image) {
		RegisterImageInput();
	} else if (ResolvedMode_ != OaViewerMode::Live) {
		RegisterTemporalInput();
	}
	if (ResolvedMode_ == OaViewerMode::Live) {
		OA_RETURN_IF_ERROR(Config_.LiveSource->Init(
			Input_,
			[this](bool InEnabled) { CapturePointer(InEnabled); }));
	}

	if (not Config_.ShowHelp) return OaStatus::Ok();
	OA_LOG_INFO(OaLogComponent::App, "═══════════════════════════════════════════════════");
	OA_LOG_INFO(OaLogComponent::App, "OaViewer (%s)", ViewerModeName(ResolvedMode_));
	if (ResolvedMode_ != OaViewerMode::Live and not Config_.Path.Empty()) {
		OA_LOG_INFO(OaLogComponent::App, "  Source: %s", Config_.Path.c_str());
	}
	if (ResolvedMode_ == OaViewerMode::Image) {
		OA_LOG_INFO(OaLogComponent::App, "  Size: %dx%d",
			ImageSource().Width, ImageSource().Height);
		OA_LOG_INFO(OaLogComponent::App,
			"  Channels: 1=R  2=G  3=B  4=A  5=RGB");
	} else if (Video_.HasValue()) {
		OA_LOG_INFO(OaLogComponent::App, "  Codec: %s",
			VideoCodecName(Video_->GetContainerInfo().Codec));
		OA_LOG_INFO(OaLogComponent::App, "  Size: %ux%u @ %u fps",
			Video_->Width(), Video_->Height(), Video_->FrameRate());
		OA_LOG_INFO(OaLogComponent::App, "  Duration: %.2f s",
			static_cast<double>(Video_->DurationUs()) / 1'000'000.0);
		OA_LOG_INFO(OaLogComponent::App,
			"  Space=play/pause  Arrows=step 1/5 frames  L=loop  timeline=seek");
	} else if (Audio_.HasValue()) {
		OA_LOG_INFO(OaLogComponent::App, "  Audio: %u Hz · %u channels · %.2f s",
			Audio_->SampleRate(), Audio_->ChannelCount(),
			static_cast<double>(Audio_->DurationUs()) / 1'000'000.0);
		OA_LOG_INFO(OaLogComponent::App,
			"  Space=play/pause  Left/Right=5 s  Up/Down=25 s  L=loop  timeline=seek");
	} else if (ResolvedMode_ == OaViewerMode::Live) {
		OA_LOG_INFO(OaLogComponent::App, "  Source: attached live producer");
	}
	if (HasVisualContent()) OA_LOG_INFO(OaLogComponent::App, "%s", OaNavigationHelpLine());
	OA_LOG_INFO(OaLogComponent::App, "  Q/Esc=Quit");
	OA_LOG_INFO(OaLogComponent::App, "═══════════════════════════════════════════════════");
	return OaStatus::Ok();
}

void OaViewer::Update(OaF32 InDeltaMs) {
	if (ResolvedMode_ == OaViewerMode::Live and Config_.LiveSource != nullptr) {
		Config_.LiveSource->Update(InDeltaMs);
	}
	if (Video_.HasValue()) Video_->Tick(InDeltaMs);
	if (HasVisualContent()) Nav_.Update(InDeltaMs);

	StatsAccumMs_ += InDeltaMs;
	++StatsFrameCount_;
	if (StatsAccumMs_ >= 500.0F) {
		DisplayFrameMs_ = StatsAccumMs_ / static_cast<OaF32>(StatsFrameCount_);
		DisplayFps_ = static_cast<OaF32>(StatsFrameCount_) * 1000.0F / StatsAccumMs_;
		StatsAccumMs_ = 0.0F;
		StatsFrameCount_ = 0;
	}
}

void OaViewer::RouteEvent(const OaUiEvent& InEvent) {
	if (ResolvedMode_ == OaViewerMode::Live and Config_.LiveSource != nullptr) {
		Config_.LiveSource->Event(InEvent);
		return;
	}
	if (not HasVisualContent()) return;
	const OaPixelRect controls = TimelineRect();
	const bool inControls = InEvent.MouseY >= static_cast<OaF32>(controls.Y - 12)
		and InEvent.MouseY < static_cast<OaF32>(controls.Y + controls.H + 12);
	const bool timelineActive = Ui_.Input().ActiveId != 0U;
	if (not HasTimeline() or (not inControls and not timelineActive)) {
		(void)Nav_.HandleEvent(InEvent);
	}
}

void OaViewer::DrawOverlay(OaUi& InUi, OaPixelRect InDestination) {
	if (not DetectionOverlay_.IsValid()) return;
	DetectionOverlay_.Draw(
		InUi,
		TextAtlas_,
		InDestination,
		{.X = 0, .Y = 0,
		 .W = static_cast<OaI32>(Width()),
		 .H = static_cast<OaI32>(Height())});
}

void OaViewer::RenderImage(OaUi& InUi) {
	const OaTexture& image = ImageSource();
	if (not image.IsValid()) return;
	const OaPixelRect destination = {
		.X = static_cast<OaI32>(Nav_.PanX()),
		.Y = static_cast<OaI32>(Nav_.PanY()),
		.W = static_cast<OaI32>(static_cast<OaF32>(image.Width) * Nav_.Zoom()),
		.H = static_cast<OaI32>(static_cast<OaF32>(image.Height) * Nav_.Zoom()),
	};

	if (ImageMode_ == ImageViewMode::RGB or not Planes_.IsValid()) {
		InUi.BeginPanel("viewer-image", destination);
		InUi.Image(image.BindlessIndex(), image.Width, image.Height);
		InUi.EndPanel();
	} else {
		const OaU32 channel = static_cast<OaU32>(ImageMode_) - 1U;
		if (channel < static_cast<OaU32>(Planes_.ChannelCount)) {
			OaImagePlanes plane;
			plane.Planes[0] = Planes_.Planes[channel];
			plane.Dtypes[0] = Planes_.Dtypes[channel];
			plane.Width = Planes_.Width;
			plane.Height = Planes_.Height;
			plane.ChannelCount = 1;
			InUi.BeginPanel("viewer-image", destination);
			InUi.ImagePlanar(plane);
			InUi.EndPanel();
		}
	}
	DrawOverlay(InUi, destination);
}

void OaViewer::RenderVideo(OaUi& InUi) {
	if (not Video_.HasValue()) return;
	const auto& frame = Video_->CurrentFrame();
	if (frame.ImageView == VK_NULL_HANDLE) return;
	if (frame.Ready.IsValid()) {
		SetRenderDependency(frame.Ready);
	}

	const OaPixelRect destination = {
		.X = static_cast<OaI32>(Nav_.PanX()),
		.Y = static_cast<OaI32>(Nav_.PanY()),
		.W = static_cast<OaI32>(static_cast<OaF32>(frame.Width) * Nav_.Zoom()),
		.H = static_cast<OaI32>(static_cast<OaF32>(frame.Height) * Nav_.Zoom()),
	};
	InUi.BeginPanel("viewer-video", destination);
	InUi.ImageVkRgba(
		frame.Image,
		frame.ImageView,
		static_cast<OaI32>(frame.Width),
		static_cast<OaI32>(frame.Height),
		VK_IMAGE_LAYOUT_GENERAL);
	InUi.EndPanel();
	DrawOverlay(InUi, destination);

	if (Config_.ShowStats) {
		InUi.BeginPanel("viewer-stats", {12, 12, 176, 56});
		InUi.LabelFmt("FPS: %.1f", DisplayFps_);
		InUi.LabelFmt("Frame: %.2f ms", DisplayFrameMs_);
		InUi.EndPanel();
	}
}

void OaViewer::RenderAudio(OaUi& InUi) {
	if (not Audio_.HasValue()) return;
	const OaU64 duration = MediaDurationUs();
	if (duration == 0U) return;
	OaF32 fraction = static_cast<OaF32>(std::min(
		1.0L,
		static_cast<long double>(MediaPositionUs())
			/ static_cast<long double>(duration)));
	const OaPixelRect rect = TimelineRect();
	if (not AudioEnvelope_.IsEmpty()) {
		if (InUi.WaveformTimeline(
			"viewer-audio-waveform", rect, AudioEnvelope_, fraction)) {
			SeekMediaFraction(fraction);
		}
	} else if (InUi.Timeline("viewer-audio-transport", rect, fraction)) {
		SeekMediaFraction(fraction);
	}
}

void OaViewer::RenderTimeline(OaUi& InUi) {
	if (not HasTimeline() or ResolvedMode_ == OaViewerMode::Audio) return;
	const OaU64 duration = MediaDurationUs();
	if (duration == 0U) return;
	OaF32 fraction = static_cast<OaF32>(std::min(
		1.0L,
		static_cast<long double>(MediaPositionUs())
			/ static_cast<long double>(duration)));
	if (InUi.Timeline("viewer-transport", TimelineRect(), fraction)) {
		SeekMediaFraction(fraction);
	}

	const OaPixelRect rect = TimelineRect();
	const OaColor playing = IsMediaPlaying()
		? OaColor::Success()
		: OaColor::Warning();
	InUi.Rect({8, rect.Y, 6, rect.H}, playing);
	if (IsMediaLooping()) {
		InUi.Rect({rect.X + rect.W + 10, rect.Y, 6, rect.H}, OaColor::Accent());
	}
}

void OaViewer::Render(OaUi& InUi) {
	if (ResolvedMode_ == OaViewerMode::Image) {
		RenderImage(InUi);
	} else if (ResolvedMode_ == OaViewerMode::Video) {
		RenderVideo(InUi);
	} else if (ResolvedMode_ == OaViewerMode::Audio) {
		RenderAudio(InUi);
	} else if (ResolvedMode_ == OaViewerMode::Live
		and Config_.LiveSource != nullptr) {
		Config_.LiveSource->Render(
			InUi, TextAtlas_, Width(), Height());
		SetRenderDependency(Config_.LiveSource->RenderReady());
	}
	RenderTimeline(InUi);
}

OaStatus OaViewer::MarkRenderSubmitted(const OaEvent& InCompletion) {
	if (not InCompletion.IsValid()) {
		return OaStatus::InvalidArgument(
			"OaViewer render consumption requires a valid completion event");
	}
	if (Engine_ == nullptr or not Engine_->OwnsEvent(InCompletion)) {
		return OaStatus::InvalidArgument(
			"OaViewer render consumption requires an event from its engine");
	}
	const OaVkTimelineWait wait = InCompletion.TimelineWait();
	if (wait.Semaphore == nullptr or wait.Value == 0U) {
		return OaStatus::InvalidArgument(
			"OaViewer render consumption requires a timeline completion");
	}
	if (ResolvedMode_ == OaViewerMode::Live and Config_.LiveSource != nullptr) {
		OA_RETURN_IF_ERROR(Config_.LiveSource->MarkConsumed(InCompletion));
	}
	if (Video_.HasValue()) {
		Video_->MarkCurrentFrameConsumed(*wait.Semaphore, wait.Value);
	}
	DetectionOverlay_.MarkConsumed(*wait.Semaphore, wait.Value);
	return OaStatus::Ok();
}

const OaTexture& OaViewer::ImageSource() const noexcept {
	return BorrowedImage_ != nullptr ? *BorrowedImage_ : Image_;
}

OaStatus OaViewer::CloseSource(OaEngine& InEngine) {
	OaStatus sourceStatus = OaStatus::Ok();
	if (ResolvedMode_ == OaViewerMode::Live and Config_.LiveSource != nullptr) {
		sourceStatus = Config_.LiveSource->Close();
	}
	DetectionOverlay_.Destroy();
	if (Video_.HasValue()) {
		const OaStatus videoStatus = Video_->Close();
		if (sourceStatus.IsOk() and not videoStatus.IsOk()) {
			sourceStatus = videoStatus;
		}
		Video_.Reset();
	}
	if (Audio_.HasValue()) {
		const OaStatus audioStatus = Audio_->Close();
		if (sourceStatus.IsOk() and not audioStatus.IsOk()) {
			sourceStatus = audioStatus;
		}
		Audio_.Reset();
	}
	AudioEnvelope_ = {};
	if (Image_.IsValid()) Image_.Destroy(InEngine);
	if (Planes_.IsValid()) Planes_.Destroy(InEngine);
	BorrowedImage_ = nullptr;
	BorrowedImageReady_ = {};
	ResolvedMode_ = OaViewerMode::Auto;
	return sourceStatus;
}
