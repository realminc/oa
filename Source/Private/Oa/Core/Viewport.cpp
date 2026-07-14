// OaViewport — implementation. See Source/Public/Oa/Core/Viewport.h.

#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/Viewport.h>
#include <Oa/Core/Log.h>
#include <Oa/Vision/FnImage.h>


OaStatus OaViewport::Run() {
	OaUiStyle style;
	style.Background = {0.01F, 0.01F, 0.01F, 1.0F};
	return OaDeviceUiApp::Run({
		.Title  = BrandViewport(Config_.Title),
		.Width  = Config_.Width,
		.Height = Config_.Height,
		.Style  = style
	});
}

OaStatus OaViewport::Save(OaComputeEngine& InEngine,
                          const OaTexture&   InTex,
                          const char*        InPath) {
	if (not InTex.IsValid()) {
		return OaStatus::Error("OaViewport::Save: invalid OaTexture (no image and no buffer)");
	}
	if (InPath == nullptr or InPath[0] == '\0') {
		return OaStatus::Error("OaViewport::Save: null/empty output path");
	}
	return OaFnImage::SaveFile(InEngine, InTex, InPath);
}

void OaViewport::OnInit(OaDeviceUi& InGpui) {
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

	if (Config_.ShowHelp) {
		OA_LOG_INFO(OaLogComponent::App, "═══════════════════════════════════════════════════");
		OA_LOG_INFO(OaLogComponent::App, "OaViewport");
		OA_LOG_INFO(OaLogComponent::App, "  Image: %s (%dx%d)", Config_.Path.c_str(), Image_.Width, Image_.Height);
		OA_LOG_INFO(OaLogComponent::App, "  Channels: %u", Planes_.ChannelCount);
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

	Nav_.FitToWindow(false);
}

void OaViewport::OnUpdate(OaF32 InDeltaMs) {
	Nav_.Update(InDeltaMs);
}

void OaViewport::OnEvent(const OaUiEvent& InEvent) {
	if (not Image_.IsValid()) { return; }
	(void)Nav_.HandleEvent(InEvent);
}

void OaViewport::OnRender(OaUi& InOui) {
	if (not Image_.IsValid()) { return; }

	const OaU32 W = Gpui().Width();
	const OaU32 H = Gpui().Height();
	if (W == 0U or H == 0U) { return; }

	const OaF32 scaledW = static_cast<OaF32>(Image_.Width) * Nav_.Zoom();
	const OaF32 scaledH = static_cast<OaF32>(Image_.Height) * Nav_.Zoom();

	const OaI32 x  = static_cast<OaI32>(Nav_.PanX());
	const OaI32 y  = static_cast<OaI32>(Nav_.PanY());
	const OaI32 dW = static_cast<OaI32>(scaledW);
	const OaI32 dH = static_cast<OaI32>(scaledH);

	if (ImageMode_ == ImageViewMode::RGB) {
		InOui.BeginPanel("image", {.X = x, .Y = y, .W = dW, .H = dH});
		InOui.Image(Image_.BindlessIndex(), Image_.Width, Image_.Height);
		InOui.EndPanel();
	} else {
		InOui.BeginPanel("image", {.X = x, .Y = y, .W = dW, .H = dH});
		const OaU32 ch = static_cast<OaU32>(ImageMode_) - 1U;
		if (ch < Planes_.ChannelCount) {
			OaImagePlanes single;
			single.Width        = Planes_.Width;
			single.Height       = Planes_.Height;
			single.ChannelCount = 1;
			single.Planes[0]    = Planes_.Planes[ch];
			single.Dtypes[0]    = Planes_.Dtypes[ch];
			InOui.ImagePlanar(single);
		}
		InOui.EndPanel();
	}
}

void OaViewport::OnShutdown(OaDeviceUi& /*InGpui*/) {
	auto& rt = *OaComputeEngine::GetGlobal();
	Planes_.Destroy(rt);
	Image_.Destroy(rt);
}