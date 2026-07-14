// OaViewer — Implementation with OaNavigation pan/zoom.

#include <Oa/Runtime/Engine.h>
#include <Oa/Ui/Viewer.h>
#include <Oa/Core/Log.h>
#include <Oa/Vision/FnImage.h>


OaStatus OaViewer::Run() {
	OaUiStyle style;
	style.Background = {0.01F, 0.01F, 0.01F, 1.0F};
	return OaDeviceUiApp::Run({
		.Title  = Config_.Title,
		.Width  = Config_.Width,
		.Height = Config_.Height,
		.Style  = style
	});
}

OaStatus OaViewer::Save(OaComputeEngine& InEngine,
                        const OaTexture&   InTex,
                        const char*        InPath) {
	if (not InTex.IsValid()) {
		return OaStatus::Error("OaViewer::Save: invalid OaTexture");
	}
	if (InPath == nullptr or InPath[0] == '\0') {
		return OaStatus::Error("OaViewer::Save: null/empty output path");
	}
	return OaFnImage::SaveFile(InEngine, InTex, InPath);
}

void OaViewer::OnInit(OaDeviceUi& InGpui) {
	auto& rt = *OaComputeEngine::GetGlobal();

	Nav_.SetCapturePointer([this](bool InOn) { CaptureRelativeMouse(InOn); });

	if (auto r = OaTexture::LoadFile(rt, Config_.Path); r.IsOk()) {
		Image_ = *r;
	} else {
		OA_LOG_ERROR(OaLogComponent::App, "Failed to load image: %s", Config_.Path.c_str());
		Quit();
		return;
	}

	if (auto r = OaImagePlanes::LoadFile(rt, Config_.Path); r.IsOk()) {
		Planes_ = *r;
	}

	if (Image_.IsValid()) {
		ResizeWindow(static_cast<OaU32>(Image_.Width), static_cast<OaU32>(Image_.Height));
		Nav_.SetContentSize(static_cast<OaF32>(Image_.Width),
		                    static_cast<OaF32>(Image_.Height));
	}

	Nav_.SetWindowSize(static_cast<OaF32>(Gpui().Width()),
	                   static_cast<OaF32>(Gpui().Height()));

	InternalCamera_ = OaCamera(static_cast<OaF32>(Image_.Width), static_cast<OaF32>(Image_.Height));
	InternalCamera_.SetOrthographic(static_cast<OaF32>(Image_.Width), static_cast<OaF32>(Image_.Height));

	Nav_.FitToWindow(false);

	if (Config_.ShowHelp) {
		OA_LOG_INFO(OaLogComponent::App, "═══════════════════════════════════════════════════");
		OA_LOG_INFO(OaLogComponent::App, "OaViewer");
		OA_LOG_INFO(OaLogComponent::App, "  Image: %s (%dx%d)", Config_.Path.c_str(), Image_.Width, Image_.Height);
		OA_LOG_INFO(OaLogComponent::App, "%s", OaNavigationHelpLine());
		OA_LOG_INFO(OaLogComponent::App, "  Channels: 1=R  2=G  3=B  4=A  5=RGB  Q/Esc=Quit");
		OA_LOG_INFO(OaLogComponent::App, "═══════════════════════════════════════════════════");
	}

	auto& input = InGpui.Input();
	input.RegisterAction({.Name = "quit",  .Binding = {.Key = Config_.KeyQuit},  .Callback = [this] { Quit(); }});
	input.RegisterAction({.Name = "quitq", .Binding = {.Key = Config_.KeyQuitQ}, .Callback = [this] { Quit(); }});
	input.RegisterAction({.Name = "red",   .Binding = {.Key = Config_.KeyRed},   .Callback = [this] { ImageMode_ = ImageViewMode::R; }});
	input.RegisterAction({.Name = "green", .Binding = {.Key = Config_.KeyGreen}, .Callback = [this] { ImageMode_ = ImageViewMode::G; }});
	input.RegisterAction({.Name = "blue",  .Binding = {.Key = Config_.KeyBlue},  .Callback = [this] { ImageMode_ = ImageViewMode::B; }});
	input.RegisterAction({.Name = "alpha", .Binding = {.Key = Config_.KeyAlpha}, .Callback = [this] { ImageMode_ = ImageViewMode::A; }});
	input.RegisterAction({.Name = "rgb",   .Binding = {.Key = Config_.KeyRGB},   .Callback = [this] { ImageMode_ = ImageViewMode::RGB; }});

	OaNavigationShortcuts navKeys;
	navKeys.ZoomIn  = Config_.KeyZoomIn;
	navKeys.ZoomOut = Config_.KeyZoomOut;
	navKeys.ZoomFit = Config_.KeyZoomFit;
	navKeys.Zoom100 = Config_.KeyZoom100;
	navKeys.PanUp   = Config_.KeyPanUp;
	navKeys.PanDown = Config_.KeyPanDown;
	navKeys.PanLeft = Config_.KeyPanLeft;
	navKeys.PanRight = Config_.KeyPanRight;
	RegisterViewportShortcuts(input, Nav_, navKeys);
}

void OaViewer::OnEvent(const OaUiEvent& InEvent) {
	(void)Nav_.HandleEvent(InEvent);
}

void OaViewer::OnUpdate(OaF32 InDeltaMs) {
	Nav_.Update(InDeltaMs);
	Nav_.UpdateCamera(InternalCamera_);
}

void OaViewer::OnRender(OaUi& InOui) {
	switch (Config_.Mode) {
		case OaViewerMode::Image2D:
			RenderImage2D(InOui);
			break;
		case OaViewerMode::Scene3D:
			RenderScene3D(InOui);
			break;
		default:
			RenderImage2D(InOui);
			break;
	}
}

void OaViewer::RenderImage2D(OaUi& InOui) {
	if (not Image_.IsValid()) return;

	const OaI32 displayW = static_cast<OaI32>(static_cast<OaF32>(Image_.Width) * Nav_.Zoom());
	const OaI32 displayH = static_cast<OaI32>(static_cast<OaF32>(Image_.Height) * Nav_.Zoom());
	const OaI32 x = static_cast<OaI32>(Nav_.PanX());
	const OaI32 y = static_cast<OaI32>(Nav_.PanY());

	switch (ImageMode_) {
		case ImageViewMode::RGB:
			InOui.BeginPanel("viewer", {.X = x, .Y = y, .W = displayW, .H = displayH});
			InOui.Image(Image_.BindlessIndex(), Image_.Width, Image_.Height);
			InOui.EndPanel();
			break;
		case ImageViewMode::R:
		case ImageViewMode::G:
		case ImageViewMode::B:
		case ImageViewMode::A:
			if (Planes_.IsValid()) {
				const OaU32 ch = static_cast<OaU32>(ImageMode_) - 1U;
				if (ch >= static_cast<OaU32>(Planes_.ChannelCount)) {
					InOui.BeginPanel("viewer", {.X = x, .Y = y, .W = displayW, .H = displayH});
					InOui.Image(Image_.BindlessIndex(), Image_.Width, Image_.Height);
					InOui.EndPanel();
					break;
				}
				OaImagePlanes single;
				single.Planes[0] = Planes_.Planes[ch];
				single.Dtypes[0] = Planes_.Dtypes[ch];
				single.Width = Planes_.Width;
				single.Height = Planes_.Height;
				single.ChannelCount = 1;

				InOui.BeginPanel("viewer", {.X = x, .Y = y, .W = displayW, .H = displayH});
				InOui.ImagePlanar(single);
				InOui.EndPanel();
			}
			break;
	}
}

void OaViewer::RenderScene3D(OaUi& InOui) {
	RenderImage2D(InOui);
}

void OaViewer::OnShutdown(OaDeviceUi&) {
	auto& rt = *OaComputeEngine::GetGlobal();
	if (Image_.IsValid()) Image_.Destroy(rt);
	if (Planes_.IsValid()) Planes_.Destroy(rt);
}
