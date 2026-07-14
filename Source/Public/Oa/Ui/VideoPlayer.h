// OaVideoPlayer - preconfigured Vulkan Video playback app.

#pragma once

#include <Oa/Core/Navigation.h>
#include <Oa/Ui/DeviceUi.h>
#include <Oa/Vision/Video.h>

class OaVideoPlayerConfig {
public:
	OaString     Path;
	OaVideoCodec Codec = OaVideoCodec::H264;
	OaString     Title = "OA Video Player";
	OaU32        Width = 960;
	OaU32        Height = 540;
	bool         Loop = true;
	bool         ShowHelp = true;
	bool         ShowStats = true;
	OaF32        FrameRateOverride = 0.0F;
	OaU32        ReorderDepth = 4;
	bool         PreferHardwareYCbCr = true;
	OaFilter     Filter = OaFilter::Nearest;
};

class OaVideoPlayer : public OaDeviceUiApp {
public:
	OaVideoPlayer() = default;
	explicit OaVideoPlayer(const OaVideoPlayerConfig& InConfig)
		: Config_(InConfig)
	{}
	OaVideoPlayer(const OaString& InPath, OaVideoCodec InCodec);
	OaVideoPlayer(const char* InPath, OaVideoCodec InCodec);

	void SetConfig(const OaVideoPlayerConfig& InConfig) { Config_ = InConfig; }
	void SetPath(const OaString& InPath) { Config_.Path = InPath; }
	void SetPath(const char* InPath) { Config_.Path = InPath; }
	void SetCodec(OaVideoCodec InCodec) { Config_.Codec = InCodec; }

	[[nodiscard]] OaStatus Run();

	OaStatus OnDeviceReady(OaGraphicsEngine& InRt) override;
	void OnInit(OaDeviceUi& InGpui) override;
	void OnUpdate(OaF32 InDeltaMs) override;
	void OnEvent(const OaUiEvent& InEvent) override;
	void OnRender(OaUi& InOui) override;
	void OnRenderSubmitted(
		const OaVkTimelineSemaphore& InSemaphore,
		OaU64 InValue) override;
	void OnShutdown(OaDeviceUi& InGpui) override;

private:
	OaVideoPlayerConfig Config_;
	OaOption<OaVideo> Video_;
	OaNavigation Nav_;
	OaF32 StatsAccumMs_ = 0.0F;
	OaU32 StatsFrameCount_ = 0;
	OaF32 DisplayFps_ = 0.0F;
	OaF32 DisplayFrameMs_ = 0.0F;

	void Scrub(OaI32 InFrameDelta);
};
